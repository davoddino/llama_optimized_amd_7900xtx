#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * silu_dst = nullptr);

bool ggml_cuda_ssm_conv_uses_state_token_concat(const ggml_tensor * dst, const ggml_tensor * silu_dst = nullptr);
