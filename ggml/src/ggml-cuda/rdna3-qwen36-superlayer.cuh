#pragma once

#include "ggml.h"

#include <cstdint>
#include <string>

struct ggml_backend_cuda_context;

bool ggml_cuda_rdna3_qwen36_superlayer_enabled(int device);
bool ggml_cuda_rdna3_qwen36_superlayer_required(int device);
bool ggml_cuda_rdna3_qwen36_superlayer_runtime_enabled(int device);
bool ggml_cuda_rdna3_qwen36_superlayer_final_enabled(int device);
bool ggml_cuda_rdna3_qwen36_superlayer_replace_l0_enabled(int device);
bool ggml_cuda_rdna3_qwen36_superlayer_replace_l0_rms_enabled(int device);
bool ggml_cuda_rdna3_qwen36_superlayer_replace_l0_qkv_enabled(int device);
bool ggml_cuda_rdna3_qwen36_superlayer_replace_l0_proj_enabled(int device);
bool ggml_cuda_rdna3_qwen36_superlayer_replace_l0_proj_z_enabled(int device);
bool ggml_cuda_rdna3_qwen36_superlayer_replace_l0_proj_beta_enabled(int device);
bool ggml_cuda_rdna3_qwen36_superlayer_replace_l0_proj_alpha_enabled(int device);
bool ggml_cuda_rdna3_qwen36_superlayer_replace_l0_ssm_enabled(int device);
bool ggml_cuda_rdna3_qwen36_superlayer_replace_l0_l2_enabled(int device);
bool ggml_cuda_rdna3_qwen36_superlayer_replace_l0_gdn_enabled(int device);
bool ggml_cuda_rdna3_qwen36_superlayer_replace_l0_gated_norm_enabled(int device);
bool ggml_cuda_rdna3_qwen36_superlayer_replace_l0_attn_out_enabled(int device);
bool ggml_cuda_rdna3_qwen36_superlayer_replace_l0_post_attn_enabled(int device);
bool ggml_cuda_rdna3_qwen36_superlayer_replace_l0_moe_router_enabled(int device);
bool ggml_cuda_rdna3_qwen36_superlayer_replace_l0_moe_gate_up_enabled(int device);
bool ggml_cuda_rdna3_qwen36_superlayer_replace_l0_moe_down_enabled(int device);

bool ggml_cuda_rdna3_qwen36_superlayer_prepare(
        ggml_backend_cuda_context * cuda_ctx,
        const ggml_cgraph * cgraph,
        std::string * blocker);

bool ggml_cuda_rdna3_qwen36_superlayer_maybe_launch_contract(
        ggml_backend_cuda_context * cuda_ctx,
        const ggml_cgraph * cgraph,
        std::string * blocker,
        uint32_t forced_l0_stage_mask);

bool ggml_cuda_rdna3_qwen36_superlayer_maybe_launch_smoke(
        ggml_backend_cuda_context * cuda_ctx,
        const ggml_cgraph * cgraph,
        std::string * blocker);
