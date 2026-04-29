#include "argsort.cuh"
#include "top-k.cuh"

#include <cfloat>
#include <type_traits>

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
#    if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2)
#        define CUB_TOP_K_AVAILABLE
using namespace cub;
#    endif  // CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2
#endif      // GGML_CUDA_USE_CUB

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

#endif                            // CUB_TOP_K_AVAILABLE

static __device__ __forceinline__ bool top_k_large_better(
        const float lhs_v, const int lhs_i,
        const float rhs_v, const int rhs_i) {
    return lhs_v > rhs_v || (lhs_v == rhs_v && (rhs_i < 0 || lhs_i < rhs_i));
}

template <int MAX_K>
static __device__ __forceinline__ void top_k_large_insert(
        const float v, const int idx, const int k,
        float (& vals)[MAX_K], int (& ids)[MAX_K]) {
    if (!top_k_large_better(v, idx, vals[k - 1], ids[k - 1])) {
        return;
    }

    int pos = k - 1;
    while (pos > 0 && top_k_large_better(v, idx, vals[pos - 1], ids[pos - 1])) {
        vals[pos] = vals[pos - 1];
        ids[pos]  = ids[pos - 1];
        --pos;
    }

    vals[pos] = v;
    ids[pos]  = idx;
}

template <int MAX_K, int BLOCK_SIZE>
static __global__ void top_k_large_f32_i32(
        const float * __restrict__ src,
        int *         __restrict__ dst,
        const int ncols,
        const int k) {
    const int row = blockIdx.x;
    const float * row_src = src + (int64_t) row * ncols;
    int * row_dst = dst + (int64_t) row * k;

    float vals[MAX_K];
    int ids[MAX_K];

    for (int i = 0; i < MAX_K; ++i) {
        vals[i] = -FLT_MAX;
        ids[i]  = -1;
    }

    for (int col = threadIdx.x; col < ncols; col += BLOCK_SIZE) {
        top_k_large_insert<MAX_K>(row_src[col], col, k, vals, ids);
    }

    extern __shared__ char smem[];
    float * shared_vals = reinterpret_cast<float *>(smem);
    int * shared_ids = reinterpret_cast<int *>(shared_vals + BLOCK_SIZE * MAX_K);

    const int base = threadIdx.x * MAX_K;
    for (int i = 0; i < MAX_K; ++i) {
        shared_vals[base + i] = i < k ? vals[i] : -FLT_MAX;
        shared_ids [base + i] = i < k ? ids[i]  : -1;
    }

    __syncthreads();

    if (threadIdx.x != 0) {
        return;
    }

    float out_vals[MAX_K];
    int out_ids[MAX_K];

    for (int i = 0; i < MAX_K; ++i) {
        out_vals[i] = -FLT_MAX;
        out_ids[i]  = -1;
    }

    for (int t = 0; t < BLOCK_SIZE; ++t) {
        const int thread_base = t * MAX_K;
        for (int i = 0; i < k; ++i) {
            const int idx = shared_ids[thread_base + i];
            if (idx < 0) {
                continue;
            }
            top_k_large_insert<MAX_K>(shared_vals[thread_base + i], idx, k, out_vals, out_ids);
        }
    }

    for (int i = 0; i < k; ++i) {
        row_dst[i] = out_ids[i];
    }
}

static bool ggml_cuda_top_k_large_supported(const ggml_tensor * dst) {
    if (dst == nullptr || dst->src[0] == nullptr) {
        return false;
    }

    const ggml_tensor * src0 = dst->src[0];
    return src0->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_I32 &&
        ggml_is_contiguous(src0) &&
        dst->ne[0] > 0 &&
        dst->ne[0] <= 32;
}

bool ggml_cuda_top_k_large_supported_op(const ggml_tensor * dst) {
    return dst != nullptr && dst->op == GGML_OP_TOP_K && ggml_cuda_top_k_large_supported(dst);
}

static void top_k_large_f32_i32_cuda(
        const float * src,
        int * dst,
        const int ncols,
        const int nrows,
        const int k,
        cudaStream_t stream) {
    constexpr int block_size = 128;

    auto launch = [&](auto max_k_tag) {
        constexpr int max_k = decltype(max_k_tag)::value;
        const size_t smem = block_size * max_k * (sizeof(float) + sizeof(int));
        top_k_large_f32_i32<max_k, block_size><<<nrows, block_size, smem, stream>>>(src, dst, ncols, k);
    };

    if (k <= 16) {
        launch(std::integral_constant<int, 16>{});
    } else {
        launch(std::integral_constant<int, 32>{});
    }
}

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();
#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
    // https://github.com/NVIDIA/cccl/issues/6391
    // TODO: investigate if there exists a point where parallelized argsort is faster than sequential top-k
    for (int i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    // Fall back to argsort + copy
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;

    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
    int *                     tmp_dst = temp_dst_alloc.get();

    if (shared_mem > max_shared_mem || ncols > 1024) {
        argsort_f32_i32_cuda_cub(pool, src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
    } else {
        argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
    }
    CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                 cudaMemcpyDeviceToDevice, stream));
#else                             // GGML_CUDA_USE_CUB
    if (ncols > 1024 && ggml_cuda_top_k_large_supported(dst)) {
        top_k_large_f32_i32_cuda(src0_d, dst_d, ncols, nrows, k, stream);
    } else {
        ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
        int *                     tmp_dst = temp_dst_alloc.get();
        argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                     cudaMemcpyDeviceToDevice, stream));
    }
#endif
}
