#include "set-rows.cuh"
#include "cpy-utils.cuh"

typedef void (*set_rows_kernel_t)(const char * src, char * dst);

// Generic quantized set_rows kernel template
template <typename idx_t, typename block_type, int qk, void (*quantize_func)(const float *, block_type *)>
static __global__ void k_set_rows_quant(const float * __restrict__ src0,
                                        const idx_t * __restrict__ src1,
                                        block_type * __restrict__ dst,
                                        const int64_t ne_total,
                                        const int64_t ne10,
                                        const int64_t ne11,
                                        const int64_t ne12,
                                        const int64_t ne13,
                                        const int64_t s01,
                                        const int64_t s02,
                                        const int64_t s03,
                                        const int64_t s10,
                                        const int64_t s11,
                                        const int64_t s12,
                                        const int64_t s1,
                                        const int64_t s2,
                                        const int64_t s3,
                                        const uint3   ne00,
                                        const uint3   ne01,
                                        const uint3   ne02,
                                        const uint3   ne11_fd,
                                        const uint3   ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    const int64_t i_base = i * qk;
    uint32_t      tmp    = (uint32_t) i_base;
    uint2         div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_type * dst_row_ptr = dst + (dst_row*s1 + i02*s2 + i03*s3) / sizeof(block_type);

    const float * src_block = src0_row + i00;
    block_type * dst_block = dst_row_ptr + i00 / qk;

    quantize_func(src_block, dst_block);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

template <typename idx_t, typename block_type, int qk, void (*quantize_func)(const float *, block_type *)>
static __global__ void k_set_rows_quant_pair(const float * __restrict__ src0_0,
                                             const float * __restrict__ src0_1,
                                             const idx_t * __restrict__ src1,
                                             block_type * __restrict__ dst0,
                                             block_type * __restrict__ dst1,
                                             const int64_t ne_total,
                                             const int64_t ne10,
                                             const int64_t ne11,
                                             const int64_t ne12,
                                             const int64_t ne13,
                                             const int64_t s01,
                                             const int64_t s02,
                                             const int64_t s03,
                                             const int64_t s10,
                                             const int64_t s11,
                                             const int64_t s12,
                                             const int64_t s1,
                                             const int64_t s2,
                                             const int64_t s3,
                                             const uint3   ne00,
                                             const uint3   ne01,
                                             const uint3   ne02,
                                             const uint3   ne11_fd,
                                             const uint3   ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    const int pair = blockIdx.y;

    const int64_t i_base = i * qk;
    uint32_t      tmp    = (uint32_t) i_base;
    uint2         div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float * src0 = pair == 0 ? src0_0 : src0_1;
    block_type *  dst  = pair == 0 ? dst0   : dst1;

    const float * src0_row    = src0 + i01*s01 + i02*s02 + i03*s03;
    block_type *  dst_row_ptr = dst + (dst_row*s1 + i02*s2 + i03*s3) / sizeof(block_type);

    const float * src_block = src0_row + i00;
    block_type *  dst_block = dst_row_ptr + i00 / qk;

    quantize_func(src_block, dst_block);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

// Template dispatch function for quantized set_rows
template<typename idx_t, typename block_type, int qk, void (*quantize_func)(const float*, block_type*)>
static void set_rows_cuda_quant(
        const float * src0_d, const idx_t * src1_d, block_type * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    GGML_ASSERT(ne00 % qk == 0);
    const int64_t ne_total = (ne00 * ne01 * ne02 * ne03) / qk;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);

    const int64_t s01 = nb01/sizeof(float);
    const int64_t s02 = nb02/sizeof(float);
    const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1;
    const int64_t s2  = nb2;
    const int64_t s3  = nb3;

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        k_set_rows_quant<idx_t, block_type, qk, quantize_func><<<grid_size, block_size, 0, stream>>>(
            src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01, s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd,
            ne01_fd, ne02_fd, ne11_fd, ne12_fd);
    }
}

template<typename idx_t, typename block_type, int qk, void (*quantize_func)(const float*, block_type*)>
static void set_rows_pair_cuda_quant(
        const float * src0_0_d, const float * src0_1_d, const idx_t * src1_d, block_type * dst0_d, block_type * dst1_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    GGML_ASSERT(ne00 % qk == 0);
    const int64_t ne_total = (ne00 * ne01 * ne02 * ne03) / qk;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks, 2);

    const int64_t s01 = nb01/sizeof(float);
    const int64_t s02 = nb02/sizeof(float);
    const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1;
    const int64_t s2  = nb2;
    const int64_t s3  = nb3;

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        k_set_rows_quant_pair<idx_t, block_type, qk, quantize_func><<<grid_size, block_size, 0, stream>>>(
            src0_0_d, src0_1_d, src1_d, dst0_d, dst1_d, ne_total, ne10, ne11, ne12, ne13, s01, s02, s03,
            s10, s11, s12, s1, s2, s3, ne00_fd, ne01_fd, ne02_fd, ne11_fd, ne12_fd);
    }
}

template <typename src_t, typename idx_t, typename dst_t>
static __global__ void k_set_rows(const src_t * __restrict__ src0,
                                  const idx_t * __restrict__ src1,
                                  dst_t * __restrict__ dst,
                                  const int64_t ne_total,
                                  const int64_t ne10,
                                  const int64_t ne11,
                                  const int64_t ne12,
                                  const int64_t ne13,
                                  const int64_t s01,
                                  const int64_t s02,
                                  const int64_t s03,
                                  const int64_t s10,
                                  const int64_t s11,
                                  const int64_t s12,
                                  const int64_t s1,
                                  const int64_t s2,
                                  const int64_t s3,
                                  const uint3   ne00,
                                  const uint3   ne01,
                                  const uint3   ne02,
                                  const uint3   ne11_fd,
                                  const uint3   ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    uint32_t tmp = (uint32_t) i;
    uint2    div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const src_t * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    dst_t * dst_row_ptr    = dst + dst_row*s1 + i02*s2 + i03*s3;

    dst_row_ptr[i00] = ggml_cuda_cast<dst_t>(src0_row[i00]);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

template <typename src_t, typename idx_t, typename dst_t>
static __global__ void k_set_rows_pair(const src_t * __restrict__ src0_0,
                                       const src_t * __restrict__ src0_1,
                                       const idx_t * __restrict__ src1,
                                       dst_t * __restrict__ dst0,
                                       dst_t * __restrict__ dst1,
                                       const int64_t ne_total,
                                       const int64_t ne10,
                                       const int64_t ne11,
                                       const int64_t ne12,
                                       const int64_t ne13,
                                       const int64_t s01,
                                       const int64_t s02,
                                       const int64_t s03,
                                       const int64_t s10,
                                       const int64_t s11,
                                       const int64_t s12,
                                       const int64_t s1,
                                       const int64_t s2,
                                       const int64_t s3,
                                       const uint3   ne00,
                                       const uint3   ne01,
                                       const uint3   ne02,
                                       const uint3   ne11_fd,
                                       const uint3   ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    const int pair = blockIdx.y;

    uint32_t tmp = (uint32_t) i;
    uint2    div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const src_t * src0 = pair == 0 ? src0_0 : src0_1;
    dst_t *       dst  = pair == 0 ? dst0   : dst1;

    const src_t * src0_row    = src0 + i01*s01 + i02*s02 + i03*s03;
    dst_t *       dst_row_ptr = dst + dst_row*s1 + i02*s2 + i03*s3;

    dst_row_ptr[i00] = ggml_cuda_cast<dst_t>(src0_row[i00]);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

template<typename src_t, typename idx_t, typename dst_t>
static void set_rows_cuda(
        const src_t * src0_d, const idx_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    const int64_t ne_total = ne00 * ne01 * ne02 * ne03;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);


    const int64_t s01 = nb01/sizeof(src_t);
    const int64_t s02 = nb02/sizeof(src_t);
    const int64_t s03 = nb03/sizeof(src_t);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1/sizeof(dst_t);
    const int64_t s2  = nb2/sizeof(dst_t);
    const int64_t s3  = nb3/sizeof(dst_t);

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        k_set_rows<<<grid_size, block_size, 0, stream>>>(src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01,
                                                         s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd, ne01_fd, ne02_fd,
                                                         ne11_fd, ne12_fd);
    }
}

template<typename src_t, typename idx_t, typename dst_t>
static void set_rows_pair_cuda(
        const src_t * src0_0_d, const src_t * src0_1_d, const idx_t * src1_d, dst_t * dst0_d, dst_t * dst1_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    const int64_t ne_total = ne00 * ne01 * ne02 * ne03;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks, 2);

    const int64_t s01 = nb01/sizeof(src_t);
    const int64_t s02 = nb02/sizeof(src_t);
    const int64_t s03 = nb03/sizeof(src_t);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1/sizeof(dst_t);
    const int64_t s2  = nb2/sizeof(dst_t);
    const int64_t s3  = nb3/sizeof(dst_t);

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        k_set_rows_pair<<<grid_size, block_size, 0, stream>>>(src0_0_d, src0_1_d, src1_d, dst0_d, dst1_d,
                                                              ne_total, ne10, ne11, ne12, ne13, s01, s02, s03,
                                                              s10, s11, s12, s1, s2, s3, ne00_fd, ne01_fd,
                                                              ne02_fd, ne11_fd, ne12_fd);
    }
}

template<typename src_t, typename idx_t>
static void set_rows_cuda(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const src_t * src0_d = (const src_t *)src0->data;
    const idx_t * src1_d = (const idx_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();


    if (dst->type == GGML_TYPE_F32) {
        set_rows_cuda(
            src0_d, src1_d, (float*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_F16) {
        set_rows_cuda(
            src0_d, src1_d, (half*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_BF16) {
        set_rows_cuda(
            src0_d, src1_d, (nv_bfloat16*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_0) {
        set_rows_cuda_quant<idx_t, block_q4_0, QK4_0, quantize_f32_q4_0_block>(
            src0_d, src1_d, (block_q4_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_1) {
        set_rows_cuda_quant<idx_t, block_q4_1, QK4_1, quantize_f32_q4_1_block>(
            src0_d, src1_d, (block_q4_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_0) {
        set_rows_cuda_quant<idx_t, block_q5_0, QK5_0, quantize_f32_q5_0_block>(
            src0_d, src1_d, (block_q5_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_1) {
        set_rows_cuda_quant<idx_t, block_q5_1, QK5_1, quantize_f32_q5_1_block>(
            src0_d, src1_d, (block_q5_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q8_0) {
        set_rows_cuda_quant<idx_t, block_q8_0, QK8_0, quantize_f32_q8_0_block>(
            src0_d, src1_d, (block_q8_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_IQ4_NL) {
        set_rows_cuda_quant<idx_t, block_iq4_nl, QK4_NL, quantize_f32_iq4_nl_block>(
            src0_d, src1_d, (block_iq4_nl*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_TQKV_2_0) {
        set_rows_cuda_quant<idx_t, block_tqkv_2_0, QK_TQKV, quantize_f32_tqkv_2_0_block>(
            src0_d, src1_d, (block_tqkv_2_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_TQKV_2_5) {
        set_rows_cuda_quant<idx_t, block_tqkv_2_5, QK_TQKV, quantize_f32_tqkv_2_5_block>(
            src0_d, src1_d, (block_tqkv_2_5*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_TQKV_3_0) {
        set_rows_cuda_quant<idx_t, block_tqkv_3_0, QK_TQKV, quantize_f32_tqkv_3_0_block>(
            src0_d, src1_d, (block_tqkv_3_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_TQKV_3_5) {
        set_rows_cuda_quant<idx_t, block_tqkv_3_5, QK_TQKV, quantize_f32_tqkv_3_5_block>(
            src0_d, src1_d, (block_tqkv_3_5*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_TQKV_4_0) {
        set_rows_cuda_quant<idx_t, block_tqkv_4_0, QK_TQKV, quantize_f32_tqkv_4_0_block>(
            src0_d, src1_d, (block_tqkv_4_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_TQKV_2_0_IP) {
        set_rows_cuda_quant<idx_t, block_tqkv_2_0_ip, QK_TQKV, quantize_f32_tqkv_2_0_ip_block>(
            src0_d, src1_d, (block_tqkv_2_0_ip*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_TQKV_2_5_IP) {
        set_rows_cuda_quant<idx_t, block_tqkv_2_5_ip, QK_TQKV, quantize_f32_tqkv_2_5_ip_block>(
            src0_d, src1_d, (block_tqkv_2_5_ip*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_TQKV_3_0_IP) {
        set_rows_cuda_quant<idx_t, block_tqkv_3_0_ip, QK_TQKV, quantize_f32_tqkv_3_0_ip_block>(
            src0_d, src1_d, (block_tqkv_3_0_ip*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_TQKV_3_5_IP) {
        set_rows_cuda_quant<idx_t, block_tqkv_3_5_ip, QK_TQKV, quantize_f32_tqkv_3_5_ip_block>(
            src0_d, src1_d, (block_tqkv_3_5_ip*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_TQKV_4_0_IP) {
        set_rows_cuda_quant<idx_t, block_tqkv_4_0_ip, QK_TQKV, quantize_f32_tqkv_4_0_ip_block>(
            src0_d, src1_d, (block_tqkv_4_0_ip*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));
    }
}

template<typename src_t, typename idx_t>
static void set_rows_pair_cuda(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0_0,
        const ggml_tensor * src0_1,
        const ggml_tensor * src1,
        ggml_tensor * dst0,
        ggml_tensor * dst1) {
    const src_t * src0_0_d = (const src_t *)src0_0->data;
    const src_t * src0_1_d = (const src_t *)src0_1->data;
    const idx_t * src1_d   = (const idx_t *)src1->data;

    const ggml_tensor * src0 = src0_0;
    ggml_tensor * dst = dst0;
    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0_0->type == src0_1->type);
    GGML_ASSERT(dst0->type == dst1->type);

    if (dst0->type == GGML_TYPE_F32) {
        set_rows_pair_cuda(
            src0_0_d, src0_1_d, src1_d, (float*)dst0->data, (float*)dst1->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst0->type == GGML_TYPE_F16) {
        set_rows_pair_cuda(
            src0_0_d, src0_1_d, src1_d, (half*)dst0->data, (half*)dst1->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst0->type == GGML_TYPE_BF16) {
        set_rows_pair_cuda(
            src0_0_d, src0_1_d, src1_d, (nv_bfloat16*)dst0->data, (nv_bfloat16*)dst1->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    }
#define GGML_CUDA_SET_ROWS_PAIR_QUANT(type_name, block_type, qk, quant_func)                         \
    else if (dst0->type == type_name) {                                                              \
        set_rows_pair_cuda_quant<idx_t, block_type, qk, quant_func>(                                  \
            src0_0_d, src0_1_d, src1_d, (block_type*)dst0->data, (block_type*)dst1->data,             \
            ne00, ne01, ne02, ne03,                                                                  \
            ne10, ne11, ne12, ne13,                                                                  \
            nb01, nb02, nb03,                                                                        \
            nb10, nb11, nb12,                                                                        \
            nb1, nb2, nb3,                                                                           \
            stream                                                                                   \
        );                                                                                            \
    }
    GGML_CUDA_SET_ROWS_PAIR_QUANT(GGML_TYPE_Q4_0,      block_q4_0,       QK4_0,    quantize_f32_q4_0_block)
    GGML_CUDA_SET_ROWS_PAIR_QUANT(GGML_TYPE_Q4_1,      block_q4_1,       QK4_1,    quantize_f32_q4_1_block)
    GGML_CUDA_SET_ROWS_PAIR_QUANT(GGML_TYPE_Q5_0,      block_q5_0,       QK5_0,    quantize_f32_q5_0_block)
    GGML_CUDA_SET_ROWS_PAIR_QUANT(GGML_TYPE_Q5_1,      block_q5_1,       QK5_1,    quantize_f32_q5_1_block)
    GGML_CUDA_SET_ROWS_PAIR_QUANT(GGML_TYPE_Q8_0,      block_q8_0,       QK8_0,    quantize_f32_q8_0_block)
    GGML_CUDA_SET_ROWS_PAIR_QUANT(GGML_TYPE_IQ4_NL,    block_iq4_nl,     QK4_NL,   quantize_f32_iq4_nl_block)
    GGML_CUDA_SET_ROWS_PAIR_QUANT(GGML_TYPE_TQKV_2_0,    block_tqkv_2_0,    QK_TQKV, quantize_f32_tqkv_2_0_block)
    GGML_CUDA_SET_ROWS_PAIR_QUANT(GGML_TYPE_TQKV_2_5,    block_tqkv_2_5,    QK_TQKV, quantize_f32_tqkv_2_5_block)
    GGML_CUDA_SET_ROWS_PAIR_QUANT(GGML_TYPE_TQKV_3_0,    block_tqkv_3_0,    QK_TQKV, quantize_f32_tqkv_3_0_block)
    GGML_CUDA_SET_ROWS_PAIR_QUANT(GGML_TYPE_TQKV_3_5,    block_tqkv_3_5,    QK_TQKV, quantize_f32_tqkv_3_5_block)
    GGML_CUDA_SET_ROWS_PAIR_QUANT(GGML_TYPE_TQKV_4_0,    block_tqkv_4_0,    QK_TQKV, quantize_f32_tqkv_4_0_block)
    GGML_CUDA_SET_ROWS_PAIR_QUANT(GGML_TYPE_TQKV_2_0_IP, block_tqkv_2_0_ip, QK_TQKV, quantize_f32_tqkv_2_0_ip_block)
    GGML_CUDA_SET_ROWS_PAIR_QUANT(GGML_TYPE_TQKV_2_5_IP, block_tqkv_2_5_ip, QK_TQKV, quantize_f32_tqkv_2_5_ip_block)
    GGML_CUDA_SET_ROWS_PAIR_QUANT(GGML_TYPE_TQKV_3_0_IP, block_tqkv_3_0_ip, QK_TQKV, quantize_f32_tqkv_3_0_ip_block)
    GGML_CUDA_SET_ROWS_PAIR_QUANT(GGML_TYPE_TQKV_3_5_IP, block_tqkv_3_5_ip, QK_TQKV, quantize_f32_tqkv_3_5_ip_block)
    GGML_CUDA_SET_ROWS_PAIR_QUANT(GGML_TYPE_TQKV_4_0_IP, block_tqkv_4_0_ip, QK_TQKV, quantize_f32_tqkv_4_0_ip_block)
#undef GGML_CUDA_SET_ROWS_PAIR_QUANT
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(dst0->type));
    }
}


void ggml_cuda_op_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32);

    if (src1->type == GGML_TYPE_I64) {
        set_rows_cuda<float, int64_t>(ctx, src0, src1, dst);
    } else {
        set_rows_cuda<float, int32_t>(ctx, src0, src1, dst);
    }
}

void ggml_cuda_op_set_rows_pair(ggml_backend_cuda_context & ctx, ggml_tensor * dst0, ggml_tensor * dst1) {
    const ggml_tensor * src0_0 = dst0->src[0];
    const ggml_tensor * src0_1 = dst1->src[0];
    const ggml_tensor * src1   = dst0->src[1];

    GGML_ASSERT(src0_0->type == GGML_TYPE_F32);
    GGML_ASSERT(src0_1->type == GGML_TYPE_F32);
    GGML_ASSERT(src1 == dst1->src[1]);
    GGML_ASSERT(src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32);
    GGML_ASSERT(dst0->type == dst1->type);

    if (src1->type == GGML_TYPE_I64) {
        set_rows_pair_cuda<float, int64_t>(ctx, src0_0, src0_1, src1, dst0, dst1);
    } else {
        set_rows_pair_cuda<float, int32_t>(ctx, src0_0, src0_1, src1, dst0, dst1);
    }
}
