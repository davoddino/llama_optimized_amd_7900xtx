#include "common.cuh"

bool ggml_cuda_top_k_large_supported_op(const ggml_tensor * dst);
void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
