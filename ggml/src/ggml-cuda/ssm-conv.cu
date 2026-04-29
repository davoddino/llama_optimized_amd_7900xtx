#include "ssm-conv.cuh"
#include "unary.cuh"

template <bool apply_silu, size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_f32(const float * __restrict__ src0, const float * __restrict__ src1,
                                    const int src0_nb0, const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                    float * __restrict__ dst, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                    const int64_t n_t, float * __restrict__ state_out, const int64_t state_out_nb1) {
    GGML_UNUSED(src0_nb0);
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);
    float *       s_block = state_out == nullptr ? nullptr : (float *) ((char *) state_out + bidx * state_out_nb1);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    for (int64_t i = 0; i < n_t; i++) {
        float sumf = 0.0f;

        if (i == 0) {
            for (size_t j = 0; j < d_conv; j++) {
                x[j] = x_block[tid * stride_x + j];
            }
        } else {
            x[(i - 1) % d_conv] = x_block[tid * stride_x + i + d_conv - 1];
        }

#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[(i + j) % d_conv] * w[j];
        }
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }

    if (s_block != nullptr) {
        const int channel = bidy * split_d_inner + tid;
#pragma unroll
        for (size_t j = 0; j < d_conv - 1; ++j) {
            s_block[channel * (d_conv - 1) + j] = x_block[tid * stride_x + j + 1];
        }
    }
}

template <bool apply_silu, size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_state_token_f32(
        const float * __restrict__ state, const float * __restrict__ token, const float * __restrict__ weights,
        const int state_nb0, const int state_nb1, const int state_nb2,
        const int token_nb1, const int token_nb2, const int weights_nb1,
        float * __restrict__ dst, const int dst_nb0, const int dst_nb2,
        float * __restrict__ state_out, const int64_t state_out_nb1) {
    GGML_UNUSED(state_nb0);

    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const float * s_block = (const float *) ((const char *) state + bidx * state_nb2 + bidy * split_d_inner * state_nb1);
    const float * t_block = (const float *) ((const char *) token + bidx * token_nb2 + bidy * split_d_inner * token_nb1);
    const float * w_block = (const float *) ((const char *) weights + bidy * split_d_inner * weights_nb1);
    float *       y_block = (float *) ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);
    float *       o_block = (float *) ((char *) state_out + bidx * state_out_nb1);

    const int stride_s = state_nb1 / sizeof(float);
    const int stride_t = token_nb1 / sizeof(float);
    const int stride_w = weights_nb1 / sizeof(float);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

#pragma unroll
    for (size_t j = 0; j < d_conv - 1; ++j) {
        x[j] = s_block[tid * stride_s + j];
    }
    x[d_conv - 1] = t_block[tid * stride_t];

#pragma unroll
    for (size_t j = 0; j < d_conv; ++j) {
        w[j] = w_block[tid * stride_w + j];
    }

    float sumf = 0.0f;
#pragma unroll
    for (size_t j = 0; j < d_conv; ++j) {
        sumf += x[j] * w[j];
    }

    y_block[tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;

    const int channel = bidy * split_d_inner + tid;
#pragma unroll
    for (size_t j = 0; j < d_conv - 1; ++j) {
        o_block[channel * (d_conv - 1) + j] = x[j + 1];
    }
}

template <bool apply_silu, size_t split_d_inner, size_t d_conv, int64_t split_n_t>
static __global__ void ssm_conv_long_token_f32(const float * __restrict__ src0, const float * __restrict__ src1,
                                               const int src0_nb0, const int src0_nb1, const int src0_nb2,
                                               const int src1_nb1, float * __restrict__ dst, const int dst_nb0,
                                               const int dst_nb1, const int dst_nb2, const int64_t n_t) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1 +
                                             bidz * split_n_t * src0_nb0);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block =
        (float *) ((char *) dst + bidx * dst_nb2 + bidz * split_n_t * dst_nb1 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    const int64_t local_n_t = min(split_n_t, n_t - bidz * split_n_t);
    const int     n_cols    = d_conv - 1 + split_n_t;

    extern __shared__ float smem[];

    constexpr int load_cols   = d_conv - 1 + split_n_t;
    constexpr int total_elems = split_d_inner * load_cols;
    int row = tid / load_cols;
    int col = tid % load_cols;
#pragma unroll
    for (int idx = 0; idx < total_elems; idx += split_d_inner) {
        if (row < (int)split_d_inner) {
            smem[row * n_cols + col] = x_block[row * stride_x + col];
        }

        col += split_d_inner;
        row += col / load_cols;
        col  = col % load_cols;
        if (idx >= total_elems - tid - split_d_inner) {
            break;
        }
    }
    __syncthreads();

    // Load weights into registers (done once, small)
    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    // Compute from shared memory
    for (int64_t i = 0; i < local_n_t; i++) {
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += smem[tid * n_cols + i + j] * w[j];
        }
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu>
static void ssm_conv_state_token_f32_cuda(
        const float * state, const float * token, const float * weights,
        const int state_nb0, const int state_nb1, const int state_nb2,
        const int token_nb1, const int token_nb2, const int weights_nb1,
        float * dst, const int dst_nb0, const int dst_nb2,
        const int64_t nc, const int64_t nr, const int64_t n_s,
        float * state_out, const int64_t state_out_nb1, cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
        ssm_conv_state_token_f32<apply_silu, threads, kNC><<<blocks, threads, 0, stream>>>(
                state, token, weights,
                state_nb0, state_nb1, state_nb2, token_nb1, token_nb2, weights_nb1,
                dst, dst_nb0, dst_nb2, state_out, state_out_nb1);
    };

    switch (nc) {
        case 3: launch_kernel(std::integral_constant<int, 3>{}); break;
        case 4: launch_kernel(std::integral_constant<int, 4>{}); break;
        case 5: launch_kernel(std::integral_constant<int, 5>{}); break;
        case 9: launch_kernel(std::integral_constant<int, 9>{}); break;
        default: GGML_ABORT("Only support kernel sizes 3, 4, 5, 9 right now.");
    }
}

template <bool apply_silu>
static void ssm_conv_f32_cuda(const float * src0, const float * src1, const int src0_nb0, const int src0_nb1,
                              const int src0_nb2, const int src1_nb1, float * dst, const int dst_nb0, const int dst_nb1,
                              const int dst_nb2, const int64_t nc, const int64_t nr, const int64_t n_t,
                              const int64_t n_s, float * state_out, const int64_t state_out_nb1, cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if (n_t <= 32) {
            const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
            ssm_conv_f32<apply_silu, threads, kNC><<<blocks, threads, 0, stream>>>(src0, src1, src0_nb0, src0_nb1, src0_nb2, src1_nb1,
                                                                       dst, dst_nb0, dst_nb1, dst_nb2, n_t, state_out, state_out_nb1);
        } else {
            GGML_ASSERT(state_out == nullptr);
            const int64_t split_n_t = 32;
            dim3          blocks(n_s, (nr + threads - 1) / threads, (n_t + split_n_t - 1) / split_n_t);
            const size_t  smem_size = threads * (kNC - 1 + split_n_t) * sizeof(float);
            ssm_conv_long_token_f32<apply_silu, threads, kNC, split_n_t><<<blocks, threads, smem_size, stream>>>(
                src0, src1, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        }
    };

    switch (nc) {
        case 3: launch_kernel(std::integral_constant<int, 3>{}); break;
        case 4: launch_kernel(std::integral_constant<int, 4>{}); break;
        case 5: launch_kernel(std::integral_constant<int, 5>{}); break;
        case 9: launch_kernel(std::integral_constant<int, 9>{}); break;
        default: GGML_ABORT("Only support kernel sizes 3, 4, 5, 9 right now.");
    }
}

bool ggml_cuda_ssm_conv_uses_state_token_concat(const ggml_tensor * dst, const ggml_tensor * silu_dst) {
    if (dst == nullptr || dst->op != GGML_OP_SSM_CONV || dst->src[0] == nullptr ||
            dst->src[1] == nullptr || dst->src[2] == nullptr) {
        return false;
    }

    const ggml_tensor * src0      = dst->src[0];
    const ggml_tensor * conv      = dst->src[1];
    const ggml_tensor * state_out = dst->src[2];
    const ggml_tensor * out       = silu_dst != nullptr ? silu_dst : dst;

    if (src0->op != GGML_OP_CONCAT || ggml_get_op_params_i32(src0, 0) != 0 ||
            src0->src[0] == nullptr || src0->src[1] == nullptr) {
        return false;
    }

    const ggml_tensor * state = src0->src[0];
    const ggml_tensor * token = src0->src[1];

    if (dst->src[3] != state || dst->src[4] != token) {
        return false;
    }

    if (state->type != GGML_TYPE_F32 || token->type != GGML_TYPE_F32 ||
            conv->type != GGML_TYPE_F32 || out->type != GGML_TYPE_F32 ||
            state_out->type != GGML_TYPE_F32) {
        return false;
    }

    const int64_t nc  = conv->ne[0];
    const int64_t nr  = conv->ne[1];
    const int64_t n_s = out->ne[2];

    if (out->ne[1] != 1 || src0->ne[0] != nc || src0->ne[1] != nr || src0->ne[2] != n_s ||
            state->ne[0] != nc - 1 || token->ne[0] != 1 ||
            state->ne[1] != nr || token->ne[1] != nr ||
            state->ne[2] != n_s || token->ne[2] != n_s) {
        return false;
    }

    if (state_out->ne[0] != (nc - 1) * nr || state_out->ne[1] != n_s ||
            state->nb[0] != sizeof(float) || token->nb[1] != sizeof(float) ||
            conv->nb[0] != sizeof(float) || out->nb[0] != sizeof(float) ||
            !ggml_is_contiguous(state_out)) {
        return false;
    }

    return true;
}

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * silu_dst) {
    const struct ggml_tensor * src0 = dst->src[0];  // conv_x
    const struct ggml_tensor * src1 = dst->src[1];  // conv1d.weight
    const struct ggml_tensor * src2 = dst->src[2];  // optional conv state update target
    const bool fuse_silu = silu_dst != nullptr;

    // When fusing, write to silu_dst (the node downstream references).
    const struct ggml_tensor * out = fuse_silu ? silu_dst : dst;

    const int64_t nc  = src1->ne[0];                // d_conv
    const int64_t nr  = src0->ne[1];                // d_inner
    const int64_t n_t = out->ne[1];                 // tokens per sequence
    const int64_t n_s = out->ne[2];                 // number of sequences in the batch

    GGML_ASSERT(out->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float *       dst_d  = (float *) out->data;
    float *       state_out_d = src2 == nullptr ? nullptr : (float *) src2->data;
    cudaStream_t  stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(out->type == GGML_TYPE_F32);
    GGML_ASSERT(src2 == nullptr || ggml_is_contiguous(src2));
    GGML_ASSERT(src2 == nullptr || ggml_nelements(src2) == (nc - 1) * nr * n_s);
    GGML_ASSERT(src2 == nullptr || n_t == 1);

    const int64_t state_out_nb1 = src2 == nullptr ? 0 : src2->nb[1];

    if (ggml_cuda_ssm_conv_uses_state_token_concat(dst, silu_dst)) {
        const ggml_tensor * state = src0->src[0];
        const ggml_tensor * token = src0->src[1];
        const float * state_d = (const float *) state->data;
        const float * token_d = (const float *) token->data;

        if (fuse_silu) {
            ssm_conv_state_token_f32_cuda<true>(
                    state_d, token_d, src1_d,
                    state->nb[0], state->nb[1], state->nb[2],
                    token->nb[1], token->nb[2], src1->nb[1],
                    dst_d, out->nb[0], out->nb[2],
                    nc, nr, n_s, state_out_d, state_out_nb1, stream);
        } else {
            ssm_conv_state_token_f32_cuda<false>(
                    state_d, token_d, src1_d,
                    state->nb[0], state->nb[1], state->nb[2],
                    token->nb[1], token->nb[2], src1->nb[1],
                    dst_d, out->nb[0], out->nb[2],
                    nc, nr, n_s, state_out_d, state_out_nb1, stream);
        }
        return;
    }

    if (fuse_silu) {
        ssm_conv_f32_cuda<true>(src0_d, src1_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, state_out_d, state_out_nb1, stream);
    } else {
        ssm_conv_f32_cuda<false>(src0_d, src1_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, state_out_d, state_out_nb1, stream);
    }
}
