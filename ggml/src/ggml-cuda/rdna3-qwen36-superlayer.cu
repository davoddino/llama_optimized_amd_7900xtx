#include "rdna3-qwen36-superlayer.cuh"

#include "ggml-cuda/common.cuh"

#ifdef GGML_USE_HIP
#include <hip/hip_cooperative_groups.h>
#else
#include <cooperative_groups.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cfloat>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

static bool qwen36_superlayer_env_enabled(const char * name) {
    const char * value = getenv(name);
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

static bool qwen36_superlayer_trace_enabled() {
    return qwen36_superlayer_env_enabled("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_TRACE");
}

static bool qwen36_superlayer_direct_l0_weights_requested() {
    return qwen36_superlayer_env_enabled("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_DIRECT_L0_WEIGHTS");
}

static bool qwen36_superlayer_direct_l0_norm_weights_requested() {
    return qwen36_superlayer_direct_l0_weights_requested() ||
        qwen36_superlayer_env_enabled("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_DIRECT_L0_NORM_WEIGHTS");
}

static bool qwen36_superlayer_direct_l0_qkv_weights_requested() {
    return qwen36_superlayer_direct_l0_weights_requested() ||
        qwen36_superlayer_env_enabled("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_DIRECT_L0_QKV_WEIGHTS");
}

static bool qwen36_superlayer_direct_l0_proj_weights_requested() {
    return qwen36_superlayer_direct_l0_weights_requested() ||
        qwen36_superlayer_env_enabled("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_DIRECT_L0_PROJ_WEIGHTS");
}

static bool qwen36_superlayer_direct_l0_ssm_weights_requested() {
    return qwen36_superlayer_direct_l0_weights_requested() ||
        qwen36_superlayer_env_enabled("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_DIRECT_L0_SSM_WEIGHTS");
}

static bool qwen36_superlayer_direct_l0_out_weights_requested() {
    return qwen36_superlayer_direct_l0_weights_requested() ||
        qwen36_superlayer_env_enabled("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_DIRECT_L0_OUT_WEIGHTS");
}

static bool qwen36_superlayer_direct_l0_moe_weights_requested() {
    return qwen36_superlayer_direct_l0_weights_requested() ||
        qwen36_superlayer_env_enabled("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_DIRECT_L0_MOE_WEIGHTS");
}

static bool qwen36_superlayer_replace_l0_all_stages_requested() {
    return qwen36_superlayer_env_enabled("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_FINAL") ||
        qwen36_superlayer_env_enabled("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_FINAL_PHYSICAL_L0") ||
        qwen36_superlayer_env_enabled("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REPLACE_L0");
}

static uint32_t qwen36_superlayer_direct_l0_stage_mask() {
    uint32_t mask = 0;
    if (qwen36_superlayer_direct_l0_norm_weights_requested()) {
        mask |= 0x1u;
    }
    if (qwen36_superlayer_direct_l0_qkv_weights_requested()) {
        mask |= 0x2u;
    }
    if (qwen36_superlayer_direct_l0_proj_weights_requested()) {
        mask |= 0x3cu;
    }
    if (qwen36_superlayer_direct_l0_ssm_weights_requested()) {
        mask |= 0x40u;
    }
    if (qwen36_superlayer_direct_l0_norm_weights_requested()) {
        mask |= 0x200u;
    }
    if (qwen36_superlayer_direct_l0_out_weights_requested()) {
        mask |= 0x400u;
    }
    if (qwen36_superlayer_direct_l0_norm_weights_requested()) {
        mask |= 0x800u;
    }
    if (qwen36_superlayer_direct_l0_moe_weights_requested()) {
        mask |= 0x7000u;
    }
    return mask;
}

static int64_t qwen36_superlayer_env_i64(const char * name, const int64_t default_value) {
    const char * value = getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }

    char * end = nullptr;
    const int64_t parsed = std::strtoll(value, &end, 10);
    return end == value ? default_value : parsed;
}

static bool tensor_name_starts_with(const ggml_tensor * tensor, const char * prefix) {
    return tensor != nullptr && std::strncmp(tensor->name, prefix, std::strlen(prefix)) == 0;
}

static bool tensor_name_equals(const ggml_tensor * tensor, const char * name) {
    return tensor != nullptr && std::strcmp(tensor->name, name) == 0;
}

static bool tensor_name_matches_layer(const ggml_tensor * tensor, const char * prefix, const int layer) {
    char name[64];
    std::snprintf(name, sizeof(name), "%s-%d", prefix, layer);
    return tensor_name_equals(tensor, name);
}

static uint64_t fnv1a_update(uint64_t h, const void * data, const size_t size) {
    const uint8_t * bytes = (const uint8_t *) data;
    for (size_t i = 0; i < size; ++i) {
        h ^= bytes[i];
        h *= 1099511628211ull;
    }
    return h;
}

template <typename T>
static uint64_t fnv1a_update_pod(uint64_t h, const T & value) {
    return fnv1a_update(h, &value, sizeof(value));
}

static uint64_t fnv1a_update_cstr(uint64_t h, const char * value) {
    if (value == nullptr) {
        const uint8_t zero = 0;
        return fnv1a_update(h, &zero, sizeof(zero));
    }
    return fnv1a_update(fnv1a_update(h, value, std::strlen(value)), "", 1);
}

struct qwen36_superlayer_plan {
    int n_nodes          = 0;
    int n_fattn          = 0;
    int n_gdn            = 0;
    int n_ssm_conv       = 0;
    int n_topk_moe       = 0;
    int n_moe_gate_up    = 0;
    int n_moe_down       = 0;
    int n_mmid           = 0;
    bool has_decode_out  = false;
    bool is_decode_token = false;
    std::array<int, 40>  layer_start;
    std::array<int, 40>  layer_end;
    std::array<bool, 40> layer_recurrent;
    uint64_t fingerprint = 0;
    std::filesystem::path artifact_dir;
    std::string blocker;

    qwen36_superlayer_plan() {
        layer_start.fill(-1);
        layer_end.fill(-1);
        layer_recurrent.fill(false);
    }
};

struct qwen36_superlayer_pack_ref {
    const ggml_tensor * tensor = nullptr;
    std::string role;
    int layer = -1;
    size_t offset = 0;
    size_t nbytes = 0;
    size_t alignment = 256;
};

struct qwen36_superlayer_pack_plan {
    std::vector<qwen36_superlayer_pack_ref> refs;
    size_t total_bytes = 0;
};

struct qwen36_superlayer_layer_pack_desc {
    uint64_t byte_begin = 0;
    uint64_t byte_end = 0;
    uint32_t first_ref = 0;
    uint32_t n_refs = 0;
    uint32_t kind = 0;
    uint32_t reserved = 0;
};

struct qwen36_superlayer_runtime_layout {
    size_t activation_slot_bytes = 0;
    size_t projection_slot_bytes = 0;
    size_t logits_bytes = 0;
    size_t router_slot_bytes = 0;
    size_t scratch_bytes = 0;
    int64_t n_embd = 0;
    int64_t n_vocab = 0;
};

struct qwen36_superlayer_runtime_ref {
    const ggml_tensor * tensor = nullptr;
    std::string role;
    size_t nbytes = 0;
};

struct qwen36_superlayer_runtime_binding_plan {
    std::vector<qwen36_superlayer_runtime_ref> refs;
};

struct qwen36_superlayer_runtime_tensor_desc {
    void * data = nullptr;
    uint64_t nbytes = 0;
    int32_t type = 0;
    int64_t ne[4] = { 0, 0, 0, 0 };
    uint64_t nb[4] = { 0, 0, 0, 0 };
    uint64_t role_hash = 0;
    uint64_t name_hash = 0;
};

struct qwen36_superlayer_l0_norm_desc {
    const float * x = nullptr;
    float * norm_out = nullptr;
    const float * norm_w_data = nullptr;
    uint64_t norm_w_offset = 0;
    uint32_t n_embd = 0;
    float eps = 0.0f;
    uint32_t has_norm_w = 0;
    uint32_t use_direct_weights = 0;
    uint32_t ready = 0;
};

struct qwen36_superlayer_l0_qkv_desc {
    float * qkv_math_out = nullptr;
    float * qkv_out = nullptr;
    float * qkv_named_out = nullptr;
    const char * wqkv_data = nullptr;
    const float * qkv_scale_data = nullptr;
    uint64_t wqkv_offset = 0;
    uint64_t qkv_scale_offset = 0;
    uint64_t qkv_scratch_offset = 0;
    uint64_t wqkv_nb1 = 0;
    uint32_t n_embd = 0;
    uint32_t n_out = 0;
    int32_t wqkv_type = GGML_TYPE_COUNT;
    uint32_t qkv_scale_n = 0;
    uint32_t has_qkv_scale = 0;
    uint32_t use_direct_weights = 0;
    uint32_t ready = 0;
};

struct qwen36_superlayer_l0_proj_desc {
    float * z_math_dst = nullptr;
    float * z_dst = nullptr;
    float * beta_math_dst = nullptr;
    float * beta_raw_dst = nullptr;
    float * beta_dst = nullptr;
    float * alpha_math_dst = nullptr;
    float * alpha_raw_dst = nullptr;
    float * alpha_biased_dst = nullptr;
    float * alpha_softplus_dst = nullptr;
    float * alpha_dst = nullptr;
    const char * wz_data = nullptr;
    const char * wbeta_data = nullptr;
    const char * walpha_data = nullptr;
    const float * alpha_dt_data = nullptr;
    const float * alpha_a_data = nullptr;
    const float * z_scale_data = nullptr;
    const float * beta_scale_data = nullptr;
    const float * alpha_scale_data = nullptr;
    uint64_t wz_offset = 0;
    uint64_t wbeta_offset = 0;
    uint64_t walpha_offset = 0;
    uint64_t alpha_dt_offset = 0;
    uint64_t alpha_a_offset = 0;
    uint64_t z_scale_offset = 0;
    uint64_t beta_scale_offset = 0;
    uint64_t alpha_scale_offset = 0;
    uint64_t z_scratch_offset = 0;
    uint64_t beta_scratch_offset = 0;
    uint64_t alpha_scratch_offset = 0;
    uint64_t wz_nb1 = 0;
    uint64_t wbeta_nb1 = 0;
    uint64_t walpha_nb1 = 0;
    uint32_t n_embd = 0;
    uint32_t z_out = 0;
    uint32_t beta_out = 0;
    uint32_t alpha_out = 0;
    int32_t wz_type = GGML_TYPE_COUNT;
    int32_t wbeta_type = GGML_TYPE_COUNT;
    int32_t walpha_type = GGML_TYPE_COUNT;
    uint32_t z_scale_n = 0;
    uint32_t beta_scale_n = 0;
    uint32_t alpha_scale_n = 0;
    uint32_t has_z_scale = 0;
    uint32_t has_beta_scale = 0;
    uint32_t has_alpha_scale = 0;
    uint32_t use_direct_weights = 0;
    uint32_t ready = 0;
};

struct qwen36_superlayer_l0_ssm_desc {
    const float * state = nullptr;
    const float * token = nullptr;
    const float * conv_w_data = nullptr;
    float * raw_dst = nullptr;
    float * silu_dst = nullptr;
    float * state_out = nullptr;
    uint64_t conv_w_offset = 0;
    uint64_t state_nb1 = 0;
    uint64_t state_nb2 = 0;
    uint64_t token_nb1 = 0;
    uint64_t token_nb2 = 0;
    uint64_t conv_w_nb1 = 0;
    uint64_t raw_nb0 = 0;
    uint64_t raw_nb2 = 0;
    uint64_t silu_nb0 = 0;
    uint64_t silu_nb2 = 0;
    uint64_t state_out_nb1 = 0;
    uint32_t d_conv = 0;
    uint32_t n_channels = 0;
    uint32_t n_seqs = 0;
    uint32_t use_direct_weights = 0;
    uint32_t ready = 0;
};

struct qwen36_superlayer_l0_l2_desc {
    const float * q_src = nullptr;
    const float * k_src = nullptr;
    float * q_dst = nullptr;
    float * k_dst = nullptr;
    uint64_t q_src_nb1 = 0;
    uint64_t q_src_nb2 = 0;
    uint64_t q_src_nb3 = 0;
    uint64_t k_src_nb1 = 0;
    uint64_t k_src_nb2 = 0;
    uint64_t k_src_nb3 = 0;
    uint64_t q_dst_nb1 = 0;
    uint64_t q_dst_nb2 = 0;
    uint64_t q_dst_nb3 = 0;
    uint64_t k_dst_nb1 = 0;
    uint64_t k_dst_nb2 = 0;
    uint64_t k_dst_nb3 = 0;
    uint32_t ncols = 0;
    uint32_t nrows = 0;
    uint32_t nchannels = 0;
    uint32_t nsamples = 0;
    float eps = 0.0f;
    uint32_t ready = 0;
};

struct qwen36_superlayer_l0_gdn_desc {
    const float * q = nullptr;
    const float * k = nullptr;
    const float * v = nullptr;
    const float * g = nullptr;
    const float * beta = nullptr;
    const float * state = nullptr;
    float * dst = nullptr;
    float * state_out = nullptr;
    uint64_t q_nb1 = 0;
    uint64_t q_nb3 = 0;
    uint64_t k_nb1 = 0;
    uint64_t k_nb3 = 0;
    uint64_t v_nb1 = 0;
    uint64_t v_nb3 = 0;
    uint64_t g_nb1 = 0;
    uint64_t g_nb3 = 0;
    uint64_t beta_nb1 = 0;
    uint64_t beta_nb3 = 0;
    uint64_t state_nb2 = 0;
    uint64_t state_nb3 = 0;
    uint64_t state_out_nb2 = 0;
    uint64_t state_out_nb3 = 0;
    uint32_t s_v = 0;
    uint32_t h_v = 0;
    uint32_t h_k = 0;
    uint32_t n_seqs = 0;
    uint32_t kda = 0;
    uint32_t ready = 0;
};

struct qwen36_superlayer_l0_gated_norm_desc {
    const float * x = nullptr;
    const float * gate = nullptr;
    const float * norm_w_data = nullptr;
    float * rms_dst = nullptr;
    float * norm_dst = nullptr;
    float * silu_dst = nullptr;
    float * out_dst = nullptr;
    float * final_dst = nullptr;
    uint64_t norm_w_offset = 0;
    uint32_t ncols = 0;
    uint32_t nrows = 0;
    float eps = 0.0f;
    uint32_t use_direct_weights = 0;
    uint32_t ready = 0;
};

struct qwen36_superlayer_l0_attn_out_desc {
    const float * x = nullptr;
    float * math_dst = nullptr;
    float * out_dst = nullptr;
    float * named_dst = nullptr;
    const char * w_data = nullptr;
    const float * scale_data = nullptr;
    uint64_t w_offset = 0;
    uint64_t scale_offset = 0;
    uint64_t w_nb1 = 0;
    uint32_t n_embd = 0;
    uint32_t n_out = 0;
    int32_t w_type = GGML_TYPE_COUNT;
    uint32_t scale_n = 0;
    uint32_t has_scale = 0;
    uint32_t use_direct_weights = 0;
    uint32_t ready = 0;
};

struct qwen36_superlayer_l0_post_attn_desc {
    const float * attn = nullptr;
    const float * skip = nullptr;
    const float * norm_w_data = nullptr;
    float * residual_dst = nullptr;
    float * residual_named_dst = nullptr;
    float * rms_dst = nullptr;
    float * norm_dst = nullptr;
    float * named_dst = nullptr;
    uint64_t norm_w_offset = 0;
    uint32_t ncols = 0;
    uint32_t nrows = 0;
    float eps = 0.0f;
    uint32_t has_norm_w = 0;
    uint32_t use_direct_weights = 0;
    uint32_t ready = 0;
};

struct qwen36_superlayer_l0_moe_router_desc {
    const float * x = nullptr;
    float * logits_math_dst = nullptr;
    float * logits_dst = nullptr;
    float * logits_named_dst = nullptr;
    float * probs_dst = nullptr;
    int32_t * argsort_dst = nullptr;
    int32_t * topk_dst = nullptr;
    float * weights_dst = nullptr;
    float * weights_sum_dst = nullptr;
    float * weights_sum_clamped_dst = nullptr;
    float * weights_norm_dst = nullptr;
    float * weights_scaled_dst = nullptr;
    const char * w_data = nullptr;
    const float * scale_data = nullptr;
    uint64_t w_offset = 0;
    uint64_t scale_offset = 0;
    uint64_t w_nb1 = 0;
    uint32_t n_embd = 0;
    uint32_t n_expert = 0;
    uint32_t n_expert_used = 0;
    int32_t w_type = GGML_TYPE_COUNT;
    uint32_t scale_n = 0;
    float weights_scale = 1.0f;
    float clamp_min = 6.103515625e-5f;
    uint32_t has_scale = 0;
    uint32_t has_weights_sum = 0;
    uint32_t has_weights_norm = 0;
    uint32_t has_weights_scaled = 0;
    uint32_t use_direct_weights = 0;
    uint32_t ready = 0;
};

struct qwen36_superlayer_l0_moe_gate_up_desc {
    const float * x = nullptr;
    const int32_t * ids = nullptr;
    float * gate_up_dst = nullptr;
    float * gate_dst = nullptr;
    float * up_dst = nullptr;
    float * swiglu_dst = nullptr;
    const char * w_data = nullptr;
    uint64_t w_offset = 0;
    uint64_t w_nb1 = 0;
    uint64_t w_nb2 = 0;
    uint64_t gate_up_nb1 = 0;
    uint64_t gate_nb1 = 0;
    uint64_t up_nb1 = 0;
    uint64_t swiglu_nb1 = 0;
    uint32_t n_embd = 0;
    uint32_t n_ff = 0;
    uint32_t n_expert = 0;
    uint32_t n_expert_used = 0;
    int32_t w_type = GGML_TYPE_COUNT;
    uint32_t use_direct_weights = 0;
    uint32_t ready = 0;
};

struct qwen36_superlayer_l0_moe_down_desc {
    const float * x = nullptr;
    const int32_t * ids = nullptr;
    const float * weights = nullptr;
    float * down_dst = nullptr;
    float * weighted_dst = nullptr;
    float * out_dst = nullptr;
    const char * w_data = nullptr;
    uint64_t w_offset = 0;
    uint64_t w_nb1 = 0;
    uint64_t w_nb2 = 0;
    uint64_t x_nb1 = 0;
    uint64_t weights_nb1 = 0;
    uint64_t down_nb1 = 0;
    uint64_t weighted_nb1 = 0;
    uint64_t out_nb1 = 0;
    uint32_t n_ff = 0;
    uint32_t n_embd = 0;
    uint32_t n_expert = 0;
    uint32_t n_expert_used = 0;
    int32_t w_type = GGML_TYPE_COUNT;
    uint32_t use_direct_weights = 0;
    uint32_t ready = 0;
};

struct qwen36_superlayer_device_pack_entry {
    int device = -1;
    uint64_t fingerprint = 0;
    uint64_t source_signature = 0;
    uint64_t runtime_signature = 0;
    void * data = nullptr;
    qwen36_superlayer_layer_pack_desc * layers = nullptr;
    qwen36_superlayer_runtime_tensor_desc * io_descs = nullptr;
    qwen36_superlayer_l0_norm_desc * l0_norm = nullptr;
    qwen36_superlayer_l0_qkv_desc * l0_qkv = nullptr;
    qwen36_superlayer_l0_proj_desc * l0_proj = nullptr;
    qwen36_superlayer_l0_ssm_desc * l0_ssm = nullptr;
    qwen36_superlayer_l0_l2_desc * l0_l2 = nullptr;
    qwen36_superlayer_l0_gdn_desc * l0_gdn = nullptr;
    qwen36_superlayer_l0_gated_norm_desc * l0_gated_norm = nullptr;
    qwen36_superlayer_l0_attn_out_desc * l0_attn_out = nullptr;
    qwen36_superlayer_l0_post_attn_desc * l0_post_attn = nullptr;
    qwen36_superlayer_l0_moe_router_desc * l0_moe_router = nullptr;
    qwen36_superlayer_l0_moe_gate_up_desc * l0_moe_gate_up = nullptr;
    qwen36_superlayer_l0_moe_down_desc * l0_moe_down = nullptr;
    void * scratch = nullptr;
    size_t bytes = 0;
    size_t tensors = 0;
    size_t io_count = 0;
    size_t io_capacity = 0;
    qwen36_superlayer_runtime_layout runtime;
    cudaEvent_t ready_event = nullptr;
};

struct qwen36_superlayer_device_pack_view {
    void * data = nullptr;
    qwen36_superlayer_layer_pack_desc * layers = nullptr;
    qwen36_superlayer_runtime_tensor_desc * io_descs = nullptr;
    qwen36_superlayer_l0_norm_desc * l0_norm = nullptr;
    qwen36_superlayer_l0_qkv_desc * l0_qkv = nullptr;
    qwen36_superlayer_l0_proj_desc * l0_proj = nullptr;
    qwen36_superlayer_l0_ssm_desc * l0_ssm = nullptr;
    qwen36_superlayer_l0_l2_desc * l0_l2 = nullptr;
    qwen36_superlayer_l0_gdn_desc * l0_gdn = nullptr;
    qwen36_superlayer_l0_gated_norm_desc * l0_gated_norm = nullptr;
    qwen36_superlayer_l0_attn_out_desc * l0_attn_out = nullptr;
    qwen36_superlayer_l0_post_attn_desc * l0_post_attn = nullptr;
    qwen36_superlayer_l0_moe_router_desc * l0_moe_router = nullptr;
    qwen36_superlayer_l0_moe_gate_up_desc * l0_moe_gate_up = nullptr;
    qwen36_superlayer_l0_moe_down_desc * l0_moe_down = nullptr;
    void * scratch = nullptr;
    size_t bytes = 0;
    size_t tensors = 0;
    size_t io_count = 0;
    size_t io_capacity = 0;
    qwen36_superlayer_runtime_layout runtime;
    cudaEvent_t ready_event = nullptr;
};

static std::filesystem::path qwen36_superlayer_cache_root() {
    const char * explicit_root = getenv("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_CACHE");
    if (explicit_root != nullptr && explicit_root[0] != '\0') {
        return std::filesystem::path(explicit_root);
    }

    const char * xdg_cache = getenv("XDG_CACHE_HOME");
    if (xdg_cache != nullptr && xdg_cache[0] != '\0') {
        return std::filesystem::path(xdg_cache) / "llama.cpp" / "rdna3-qwen36-superlayer";
    }

    const char * home = getenv("HOME");
    if (home != nullptr && home[0] != '\0') {
        return std::filesystem::path(home) / ".cache" / "llama.cpp" / "rdna3-qwen36-superlayer";
    }

    return std::filesystem::path(".rdna3-qwen36-superlayer-cache");
}

static size_t align_up(const size_t value, const size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

static bool qwen36_superlayer_runtime_tensor(const ggml_tensor * t) {
    if (t == nullptr) {
        return true;
    }
    if (t->flags & GGML_TENSOR_FLAG_INPUT) {
        return true;
    }

    return tensor_name_starts_with(t, "inp_") ||
        tensor_name_starts_with(t, "cache_") ||
        tensor_name_starts_with(t, "positions") ||
        tensor_name_starts_with(t, "KQ_mask") ||
        tensor_name_starts_with(t, "attn_inp") ||
        tensor_name_starts_with(t, "conv_states") ||
        tensor_name_starts_with(t, "state_") ||
        tensor_name_starts_with(t, "result_");
}

static bool qwen36_superlayer_packable_tensor(const ggml_tensor * t) {
    if (t == nullptr || t->data == nullptr || qwen36_superlayer_runtime_tensor(t)) {
        return false;
    }
    if (t->op != GGML_OP_NONE) {
        return false;
    }
    if (t->name[0] == '\0') {
        return false;
    }

    if (t->flags & GGML_TENSOR_FLAG_PARAM) {
        return true;
    }

    return std::strstr(t->name, ".weight") != nullptr ||
        std::strstr(t->name, ".bias") != nullptr ||
        std::strstr(t->name, ".scale") != nullptr ||
        std::strstr(t->name, "blk.") != nullptr ||
        std::strstr(t->name, "output") != nullptr ||
        std::strstr(t->name, "token_embd") != nullptr;
}

static void qwen36_superlayer_sanitize_role(std::string & role) {
    for (char & c : role) {
        if (!(std::isalnum((unsigned char) c) || c == '_' || c == '.')) {
            c = '_';
        }
    }
}

static std::string qwen36_superlayer_role_for_src(
        const ggml_tensor * node, const ggml_tensor * src, const int src_idx, const int layer) {
    std::string role;
    if (layer >= 0) {
        role = "L" + std::to_string(layer) + ".";
    } else {
        role = "global.";
    }

    if (node != nullptr && node->name[0] != '\0') {
        role += node->name;
    } else {
        role += "leaf";
    }
    role += ".src";
    role += std::to_string(src_idx);

    if (src != nullptr && src->name[0] != '\0') {
        role += ".";
        role += src->name;
    }

    qwen36_superlayer_sanitize_role(role);

    return role;
}

static std::string qwen36_superlayer_role_for_runtime(
        const ggml_tensor * node, const ggml_tensor * tensor, const int src_idx, const int layer) {
    std::string role = "runtime.";
    if (layer >= 0) {
        role += "L" + std::to_string(layer) + ".";
    } else {
        role += "global.";
    }

    if (src_idx < 0) {
        role += "dst.";
    } else {
        role += "src" + std::to_string(src_idx) + ".";
    }

    if (node != nullptr && node->name[0] != '\0') {
        role += node->name;
        role += ".";
    }
    if (tensor != nullptr && tensor->name[0] != '\0') {
        role += tensor->name;
    } else {
        role += "tensor";
    }

    qwen36_superlayer_sanitize_role(role);
    return role;
}

static int qwen36_superlayer_layer_for_node(const qwen36_superlayer_plan & plan, const int node_idx) {
    for (int layer = 0; layer < 40; ++layer) {
        if (plan.layer_start[layer] >= 0 && node_idx >= plan.layer_start[layer] && node_idx <= plan.layer_end[layer]) {
            return layer;
        }
    }
    return -1;
}

static qwen36_superlayer_pack_plan qwen36_superlayer_make_pack_plan(
        const ggml_cgraph * cgraph, const qwen36_superlayer_plan & plan) {
    qwen36_superlayer_pack_plan pack;
    std::unordered_map<const ggml_tensor *, size_t> seen;

    auto add_ref = [&](const ggml_tensor * tensor, const std::string & role, const int layer) {
        if (!qwen36_superlayer_packable_tensor(tensor)) {
            return;
        }
        if (seen.find(tensor) != seen.end()) {
            return;
        }

        qwen36_superlayer_pack_ref ref;
        ref.tensor = tensor;
        ref.role = role;
        ref.layer = layer;
        ref.nbytes = ggml_nbytes(tensor);
        ref.alignment = ggml_is_quantized(tensor->type) ? 4096 : 256;
        seen.emplace(tensor, pack.refs.size());
        pack.refs.push_back(std::move(ref));
    };

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (node == nullptr) {
            continue;
        }

        const int layer = qwen36_superlayer_layer_for_node(plan, i);
        const bool in_decode_span = layer >= 0 || tensor_name_starts_with(node, "result_output");
        if (!in_decode_span) {
            continue;
        }

        for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
            const ggml_tensor * src = node->src[src_idx];
            add_ref(src, qwen36_superlayer_role_for_src(node, src, src_idx, layer), layer);
        }
    }

    std::stable_sort(pack.refs.begin(), pack.refs.end(),
            [](const qwen36_superlayer_pack_ref & a, const qwen36_superlayer_pack_ref & b) {
        if (a.layer != b.layer) {
            return a.layer < b.layer;
        }
        return a.role < b.role;
    });

    size_t offset = 0;
    for (qwen36_superlayer_pack_ref & ref : pack.refs) {
        offset = align_up(offset, ref.alignment);
        ref.offset = offset;
        offset += ref.nbytes;
    }
    pack.total_bytes = align_up(offset, 4096);

    return pack;
}

static qwen36_superlayer_pack_plan qwen36_superlayer_repack_refs(
        const std::vector<qwen36_superlayer_pack_ref> & refs) {
    qwen36_superlayer_pack_plan pack;
    pack.refs.reserve(refs.size());

    size_t offset = 0;
    for (qwen36_superlayer_pack_ref ref : refs) {
        offset = align_up(offset, ref.alignment);
        ref.offset = offset;
        offset += ref.nbytes;
        pack.refs.push_back(std::move(ref));
    }
    pack.total_bytes = align_up(offset, 4096);
    return pack;
}

static qwen36_superlayer_pack_plan qwen36_superlayer_make_runtime_pack_plan(
        const qwen36_superlayer_pack_plan & full_pack) {
    if (qwen36_superlayer_env_enabled("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_FULL_WEIGHTPACK")) {
        return full_pack;
    }

    std::vector<qwen36_superlayer_pack_ref> refs;
    refs.reserve(full_pack.refs.size());
    for (const qwen36_superlayer_pack_ref & ref : full_pack.refs) {
        if (ref.layer == 0) {
            refs.push_back(ref);
        }
    }
    return qwen36_superlayer_repack_refs(refs);
}

static std::string hex_u64(const uint64_t value) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%016" PRIx64, value);
    return std::string(buf);
}

static uint64_t qwen36_superlayer_source_signature(const qwen36_superlayer_pack_plan & pack) {
    uint64_t h = 1469598103934665603ull;
    h = fnv1a_update_cstr(h, "qwen36-rdna3-superlayer-device-pack-v0");
    h = fnv1a_update_pod(h, pack.refs.size());
    h = fnv1a_update_pod(h, pack.total_bytes);

    for (const qwen36_superlayer_pack_ref & ref : pack.refs) {
        const ggml_tensor * t = ref.tensor;
        const uintptr_t data = (uintptr_t) t->data;
        h = fnv1a_update_cstr(h, ref.role.c_str());
        h = fnv1a_update_cstr(h, t->name);
        h = fnv1a_update_pod(h, data);
        h = fnv1a_update_pod(h, ref.offset);
        h = fnv1a_update_pod(h, ref.nbytes);
        h = fnv1a_update_pod(h, t->type);
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            h = fnv1a_update_pod(h, t->ne[d]);
            h = fnv1a_update_pod(h, t->nb[d]);
        }
    }

    return h;
}

static std::array<qwen36_superlayer_layer_pack_desc, 40> qwen36_superlayer_make_layer_pack_descs(
        const qwen36_superlayer_plan & plan, const qwen36_superlayer_pack_plan & pack) {
    std::array<qwen36_superlayer_layer_pack_desc, 40> layers;

    for (int layer = 0; layer < 40; ++layer) {
        qwen36_superlayer_layer_pack_desc & desc = layers[layer];
        desc.kind = plan.layer_recurrent[layer] ? 1u : 0u;
        desc.first_ref = (uint32_t) -1;
    }

    for (size_t i = 0; i < pack.refs.size(); ++i) {
        const qwen36_superlayer_pack_ref & ref = pack.refs[i];
        if (ref.layer < 0 || ref.layer >= 40) {
            continue;
        }

        qwen36_superlayer_layer_pack_desc & desc = layers[ref.layer];
        if (desc.n_refs == 0) {
            desc.first_ref = (uint32_t) i;
            desc.byte_begin = ref.offset;
            desc.byte_end = ref.offset + ref.nbytes;
        } else {
            desc.byte_begin = std::min<uint64_t>(desc.byte_begin, ref.offset);
            desc.byte_end = std::max<uint64_t>(desc.byte_end, ref.offset + ref.nbytes);
        }
        desc.n_refs++;
    }

    for (qwen36_superlayer_layer_pack_desc & desc : layers) {
        if (desc.n_refs == 0) {
            desc.first_ref = 0;
        }
    }

    return layers;
}

static qwen36_superlayer_runtime_layout qwen36_superlayer_make_runtime_layout(
        const ggml_cgraph * cgraph, const qwen36_superlayer_plan & plan) {
    qwen36_superlayer_runtime_layout layout;
    size_t l0_projection_bytes = 0;

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (node == nullptr) {
            continue;
        }

        const int layer = qwen36_superlayer_layer_for_node(plan, i);
        const bool in_decode_span = layer >= 0 || tensor_name_starts_with(node, "result_output");
        if (!in_decode_span) {
            continue;
        }

        const size_t node_bytes = ggml_nbytes(node);
        layout.activation_slot_bytes = std::max(layout.activation_slot_bytes, node_bytes);

        if (tensor_name_starts_with(node, "result_output")) {
            layout.logits_bytes = std::max(layout.logits_bytes, node_bytes);
            layout.n_vocab = std::max<int64_t>(layout.n_vocab, node->ne[0]);
            if (node->src[1] != nullptr) {
                layout.n_embd = std::max<int64_t>(layout.n_embd, node->src[1]->ne[0]);
            }
        }

        if (std::strstr(node->name, "ffn_moe") != nullptr ||
                std::strstr(node->name, "moe") != nullptr) {
            layout.router_slot_bytes = std::max(layout.router_slot_bytes, node_bytes);
        }

        if (tensor_name_matches_layer(node, "z", 0) ||
                tensor_name_matches_layer(node, "beta_sigmoid", 0) ||
                tensor_name_matches_layer(node, "gate", 0)) {
            l0_projection_bytes += align_up(node_bytes, 256);
        }
    }

    layout.activation_slot_bytes = align_up(std::max<size_t>(layout.activation_slot_bytes, 4096), 4096);
    layout.projection_slot_bytes = align_up(std::max<size_t>(l0_projection_bytes, 4096), 4096);
    layout.logits_bytes = align_up(layout.logits_bytes, 4096);
    layout.router_slot_bytes = align_up(layout.router_slot_bytes, 4096);
    layout.scratch_bytes = align_up(
            2*layout.activation_slot_bytes + layout.projection_slot_bytes +
            layout.logits_bytes + layout.router_slot_bytes, 4096);

    return layout;
}

static bool qwen36_superlayer_runtime_bindable_tensor(const ggml_tensor * t, const int device) {
    return t != nullptr &&
        t->data != nullptr &&
        t->buffer != nullptr &&
        ggml_backend_buffer_get_type(t->buffer) == ggml_backend_cuda_buffer_type(device) &&
        qwen36_superlayer_runtime_tensor(t);
}

static qwen36_superlayer_runtime_binding_plan qwen36_superlayer_make_runtime_binding_plan(
        const ggml_cgraph * cgraph, const qwen36_superlayer_plan & plan, const int device) {
    qwen36_superlayer_runtime_binding_plan bindings;
    std::unordered_map<const ggml_tensor *, size_t> seen;

    auto add_ref = [&](const ggml_tensor * node, const ggml_tensor * tensor, const int src_idx, const int layer) {
        if (!qwen36_superlayer_runtime_bindable_tensor(tensor, device)) {
            return;
        }
        if (seen.find(tensor) != seen.end()) {
            return;
        }

        qwen36_superlayer_runtime_ref ref;
        ref.tensor = tensor;
        ref.role = qwen36_superlayer_role_for_runtime(node, tensor, src_idx, layer);
        ref.nbytes = ggml_nbytes(tensor);
        seen.emplace(tensor, bindings.refs.size());
        bindings.refs.push_back(std::move(ref));
    };

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (node == nullptr) {
            continue;
        }

        const int layer = qwen36_superlayer_layer_for_node(plan, i);
        const bool in_decode_span = layer >= 0 || tensor_name_starts_with(node, "result_output");
        if (!in_decode_span) {
            continue;
        }

        add_ref(node, node, -1, layer);
        for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
            add_ref(node, node->src[src_idx], src_idx, layer);
        }
    }

    std::stable_sort(bindings.refs.begin(), bindings.refs.end(),
            [](const qwen36_superlayer_runtime_ref & a, const qwen36_superlayer_runtime_ref & b) {
        return a.role < b.role;
    });

    return bindings;
}

static uint64_t qwen36_superlayer_hash_string(const char * value) {
    return fnv1a_update_cstr(1469598103934665603ull, value);
}

static std::vector<qwen36_superlayer_runtime_tensor_desc> qwen36_superlayer_make_runtime_descs(
        const qwen36_superlayer_runtime_binding_plan & bindings) {
    std::vector<qwen36_superlayer_runtime_tensor_desc> descs;
    descs.reserve(bindings.refs.size());

    for (const qwen36_superlayer_runtime_ref & ref : bindings.refs) {
        const ggml_tensor * t = ref.tensor;
        qwen36_superlayer_runtime_tensor_desc desc;
        desc.data = t->data;
        desc.nbytes = ref.nbytes;
        desc.type = (int32_t) t->type;
        desc.role_hash = qwen36_superlayer_hash_string(ref.role.c_str());
        desc.name_hash = qwen36_superlayer_hash_string(t->name);
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            desc.ne[d] = t->ne[d];
            desc.nb[d] = (uint64_t) t->nb[d];
        }
        descs.push_back(desc);
    }

    return descs;
}

static uint64_t qwen36_superlayer_runtime_signature(
        const qwen36_superlayer_runtime_binding_plan & bindings) {
    uint64_t h = 1469598103934665603ull;
    h = fnv1a_update_cstr(h, "qwen36-rdna3-superlayer-runtime-bindings-v0");
    h = fnv1a_update_pod(h, bindings.refs.size());

    for (const qwen36_superlayer_runtime_ref & ref : bindings.refs) {
        const ggml_tensor * t = ref.tensor;
        const uintptr_t data = (uintptr_t) t->data;
        h = fnv1a_update_cstr(h, ref.role.c_str());
        h = fnv1a_update_cstr(h, t->name);
        h = fnv1a_update_pod(h, data);
        h = fnv1a_update_pod(h, ref.nbytes);
        h = fnv1a_update_pod(h, t->type);
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            h = fnv1a_update_pod(h, t->ne[d]);
            h = fnv1a_update_pod(h, t->nb[d]);
        }
    }

    return h;
}

static std::string qwen36_superlayer_device_pack_key(
        const int device,
        const uint64_t fingerprint,
        const uint64_t source_signature) {
    return "d" + std::to_string(device) + "-" + hex_u64(fingerprint) +
        "-" + hex_u64(source_signature);
}

static void qwen36_superlayer_set_cuda_blocker(
        std::string * blocker, const char * action, const cudaError_t err) {
    if (blocker != nullptr) {
        *blocker = std::string(action) + ": " + cudaGetErrorString(err);
    }
}

static bool qwen36_superlayer_tensor_on_device(
        const ggml_tensor * t, const int device, std::string * blocker) {
    if (t == nullptr || t->data == nullptr) {
        if (blocker != nullptr) {
            *blocker = "weightpack tensor is missing data";
        }
        return false;
    }
    if (t->buffer == nullptr ||
            ggml_backend_buffer_get_type(t->buffer) != ggml_backend_cuda_buffer_type(device)) {
        if (blocker != nullptr) {
            *blocker = "weightpack tensor is not resident on target CUDA/HIP device: ";
            *blocker += t->name;
        }
        return false;
    }
    return true;
}

static bool qwen36_superlayer_find_pack_offset(
        const qwen36_superlayer_pack_plan & pack,
        const ggml_tensor * tensor,
        uint64_t * offset) {
    for (const qwen36_superlayer_pack_ref & ref : pack.refs) {
        if (ref.tensor == tensor) {
            if (offset != nullptr) {
                *offset = ref.offset;
            }
            return true;
        }
    }
    return false;
}

static bool qwen36_superlayer_tensor_data_on_device(
        const ggml_tensor * t,
        const int device,
        const char * role,
        std::string * blocker) {
    if (t == nullptr || t->data == nullptr) {
        if (blocker != nullptr) {
            *blocker = std::string(role) + " is missing device data";
        }
        return false;
    }
    if (t->buffer == nullptr ||
            ggml_backend_buffer_get_type(t->buffer) != ggml_backend_cuda_buffer_type(device)) {
        if (blocker != nullptr) {
            *blocker = std::string(role) + " is not resident on target CUDA/HIP device: ";
            *blocker += t->name;
        }
        return false;
    }
    return true;
}

static bool qwen36_superlayer_make_l0_norm_desc(
        const ggml_cgraph * cgraph,
        const qwen36_superlayer_plan & plan,
        const qwen36_superlayer_pack_plan & pack,
        const int device,
        qwen36_superlayer_l0_norm_desc * desc,
        std::string * blocker) {
    if (desc == nullptr) {
        if (blocker != nullptr) {
            *blocker = "missing L0 RMSNorm descriptor output";
        }
        return false;
    }
    *desc = qwen36_superlayer_l0_norm_desc{};

    const ggml_tensor * attn_norm = nullptr;
    const int begin = plan.layer_start[0] >= 0 ? plan.layer_start[0] : 0;
    const int end   = plan.layer_end[0]   >= begin ? plan.layer_end[0]   : cgraph->n_nodes - 1;
    for (int i = begin; i <= end; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (tensor_name_matches_layer(node, "attn_norm", 0)) {
            attn_norm = node;
            break;
        }
    }
    if (attn_norm == nullptr) {
        if (blocker != nullptr) {
            *blocker = "L0 RMSNorm fusion could not find attn_norm-0";
        }
        return false;
    }

    const ggml_tensor * rms = nullptr;
    const ggml_tensor * norm_w = nullptr;
    if (attn_norm->op == GGML_OP_MUL) {
        for (int i = 0; i < 2; ++i) {
            const ggml_tensor * src = attn_norm->src[i];
            if (src != nullptr && src->op == GGML_OP_RMS_NORM) {
                rms = src;
                norm_w = attn_norm->src[1 - i];
                break;
            }
        }
    } else if (attn_norm->op == GGML_OP_RMS_NORM) {
        rms = attn_norm;
    }

    if (rms == nullptr || rms->src[0] == nullptr) {
        if (blocker != nullptr) {
            *blocker = "L0 RMSNorm fusion expected attn_norm-0 to contain RMS_NORM";
        }
        return false;
    }

    const ggml_tensor * x = rms->src[0];
    if (x->type != GGML_TYPE_F32 || rms->type != GGML_TYPE_F32 || attn_norm->type != GGML_TYPE_F32) {
        if (blocker != nullptr) {
            *blocker = "L0 RMSNorm fusion currently requires F32 activation/output tensors";
        }
        return false;
    }
    if (x->ne[0] <= 0 || x->nb[0] != (int64_t) sizeof(float) ||
            attn_norm->nb[0] != (int64_t) sizeof(float) ||
            ggml_nelements(x) / x->ne[0] != 1 ||
            ggml_nelements(attn_norm) != x->ne[0]) {
        if (blocker != nullptr) {
            *blocker = "L0 RMSNorm fusion requires dense one-token F32 hidden tensors";
        }
        return false;
    }
    if (!qwen36_superlayer_tensor_data_on_device(x, device, "L0 RMSNorm input", blocker)) {
        return false;
    }
    if (!qwen36_superlayer_tensor_data_on_device(attn_norm, device, "L0 RMSNorm output", blocker)) {
        return false;
    }

    const bool use_direct_weights = qwen36_superlayer_direct_l0_norm_weights_requested();
    uint64_t norm_w_offset = 0;
    if (norm_w != nullptr) {
        if (norm_w->type != GGML_TYPE_F32 || norm_w->ne[0] != x->ne[0] ||
                norm_w->nb[0] != (int64_t) sizeof(float)) {
            if (blocker != nullptr) {
                *blocker = "L0 RMSNorm fusion requires a dense F32 norm weight matching n_embd";
            }
            return false;
        }
        if (!qwen36_superlayer_find_pack_offset(pack, norm_w, &norm_w_offset)) {
            if (blocker != nullptr) {
                *blocker = "L0 RMSNorm norm weight is not present in the fused weightpack: ";
                *blocker += norm_w->name;
            }
            return false;
        }
        if (use_direct_weights &&
                !qwen36_superlayer_tensor_data_on_device(norm_w, device, "L0 RMSNorm weight", blocker)) {
            return false;
        }
    }

    const float eps = ggml_get_op_params_f32(rms, 0);
    if (eps < 0.0f) {
        if (blocker != nullptr) {
            *blocker = "L0 RMSNorm epsilon is negative";
        }
        return false;
    }

    desc->x = (const float *) x->data;
    desc->norm_out = (float *) attn_norm->data;
    desc->norm_w_data = norm_w != nullptr ? (const float *) norm_w->data : nullptr;
    desc->norm_w_offset = norm_w_offset;
    desc->n_embd = (uint32_t) x->ne[0];
    desc->eps = eps;
    desc->has_norm_w = norm_w != nullptr ? 1u : 0u;
    desc->use_direct_weights = use_direct_weights ? 1u : 0u;
    desc->ready = 1u;
    return true;
}

static const ggml_tensor * qwen36_superlayer_strip_view_ops(const ggml_tensor * t) {
    for (int depth = 0; t != nullptr && depth < 8; ++depth) {
        switch (t->op) {
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
            case GGML_OP_CONT:
                t = t->src[0];
                break;
            default:
                return t;
        }
    }
    return t;
}

static bool qwen36_superlayer_same_tensor_or_view(const ggml_tensor * a, const ggml_tensor * b) {
    return a == b || qwen36_superlayer_strip_view_ops(a) == qwen36_superlayer_strip_view_ops(b);
}

static bool qwen36_superlayer_projection_weight_supported(const ggml_tensor * w) {
    if (w == nullptr) {
        return false;
    }
    switch (w->type) {
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_F16:
        case GGML_TYPE_F32:
            return true;
        default:
            return false;
    }
}

static bool qwen36_superlayer_make_l0_qkv_desc(
        const ggml_cgraph * cgraph,
        const qwen36_superlayer_plan & plan,
        const qwen36_superlayer_pack_plan & pack,
        const qwen36_superlayer_runtime_layout & runtime,
        const int device,
        qwen36_superlayer_l0_qkv_desc * desc,
        std::string * blocker) {
    if (desc == nullptr) {
        if (blocker != nullptr) {
            *blocker = "missing L0 QKV descriptor output";
        }
        return false;
    }
    *desc = qwen36_superlayer_l0_qkv_desc{};

    const int begin = plan.layer_start[0] >= 0 ? plan.layer_start[0] : 0;
    const int end   = plan.layer_end[0]   >= begin ? plan.layer_end[0]   : cgraph->n_nodes - 1;
    const ggml_tensor * attn_norm = nullptr;
    const ggml_tensor * qkv_named = nullptr;
    for (int i = begin; i <= end; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (attn_norm == nullptr && tensor_name_matches_layer(node, "attn_norm", 0)) {
            attn_norm = node;
        }
        if (qkv_named == nullptr && tensor_name_matches_layer(node, "linear_attn_qkv_mixed", 0)) {
            qkv_named = node;
        }
    }
    if (attn_norm == nullptr || qkv_named == nullptr) {
        if (blocker != nullptr) {
            *blocker = "L0 QKV fusion requires attn_norm-0 and linear_attn_qkv_mixed-0";
        }
        return false;
    }

    const ggml_tensor * qkv_out = qwen36_superlayer_strip_view_ops(qkv_named);
    const ggml_tensor * qkv_math = qkv_out;
    const ggml_tensor * qkv_scale = nullptr;
    if (qkv_out != nullptr && qkv_out->op == GGML_OP_MUL) {
        if (qkv_out->src[0] != nullptr && qkv_out->src[0]->op == GGML_OP_MUL_MAT) {
            qkv_math = qkv_out->src[0];
            qkv_scale = qkv_out->src[1];
        } else if (qkv_out->src[1] != nullptr && qkv_out->src[1]->op == GGML_OP_MUL_MAT) {
            qkv_math = qkv_out->src[1];
            qkv_scale = qkv_out->src[0];
        }
    }

    if (qkv_math == nullptr || qkv_math->op != GGML_OP_MUL_MAT ||
            qkv_math->src[0] == nullptr || qkv_math->src[1] == nullptr) {
        if (blocker != nullptr) {
            *blocker = "L0 QKV fusion could not resolve linear_attn_qkv_mixed-0 to MUL_MAT";
        }
        return false;
    }
    if (qkv_math->src[1] != attn_norm) {
        if (blocker != nullptr) {
            *blocker = "L0 QKV MUL_MAT input is not attn_norm-0";
        }
        return false;
    }
    if (qkv_out == nullptr || qkv_out->type != GGML_TYPE_F32 ||
            qkv_named->type != GGML_TYPE_F32 || qkv_math->type != GGML_TYPE_F32) {
        if (blocker != nullptr) {
            *blocker = "L0 QKV fusion currently requires F32 QKV output tensors";
        }
        return false;
    }

    const ggml_tensor * wqkv = qkv_math->src[0];
    if (!qwen36_superlayer_projection_weight_supported(wqkv)) {
        if (blocker != nullptr) {
            *blocker = "L0 QKV weight type is not implemented in the superlayer";
        }
        return false;
    }
    if (wqkv->ne[0] <= 0 || wqkv->ne[1] <= 0 || qkv_math->ne[0] != wqkv->ne[1]) {
        if (blocker != nullptr) {
            *blocker = "L0 QKV tensor dimensions do not match";
        }
        return false;
    }
    if (qkv_math->src[1]->ne[0] != wqkv->ne[0]) {
        if (blocker != nullptr) {
            *blocker = "L0 QKV input hidden dimension does not match weight columns";
        }
        return false;
    }
    if (ggml_nelements(qkv_out) / qkv_out->ne[0] != 1) {
        if (blocker != nullptr) {
            *blocker = "L0 QKV fusion is limited to one-token decode";
        }
        return false;
    }
    if (qkv_out->nb[0] != (int64_t) sizeof(float) ||
            qkv_named->nb[0] != (int64_t) sizeof(float) ||
            qkv_math->nb[0] != (int64_t) sizeof(float) ||
            !ggml_is_contiguous_1(qkv_out) ||
            !ggml_is_contiguous_1(qkv_named) ||
            !ggml_is_contiguous_1(qkv_math) ||
            ggml_nelements(qkv_out) != qkv_math->ne[0] ||
            ggml_nelements(qkv_named) != qkv_math->ne[0] ||
            ggml_nelements(qkv_math) != qkv_math->ne[0]) {
        if (blocker != nullptr) {
            *blocker = "L0 QKV fusion requires dense one-token QKV output";
        }
        return false;
    }
    if (!qwen36_superlayer_tensor_data_on_device(qkv_out, device, "L0 QKV output", blocker)) {
        return false;
    }
    if (qkv_math != qkv_out && qkv_math != qkv_named &&
            !qwen36_superlayer_tensor_data_on_device(qkv_math, device, "L0 QKV math output", blocker)) {
        return false;
    }
    if (qkv_named != qkv_out &&
            !qwen36_superlayer_tensor_data_on_device(qkv_named, device, "L0 QKV named output", blocker)) {
        return false;
    }

    const bool use_direct_weights = qwen36_superlayer_direct_l0_qkv_weights_requested();
    uint64_t wqkv_offset = 0;
    if (!qwen36_superlayer_find_pack_offset(pack, wqkv, &wqkv_offset)) {
        if (blocker != nullptr) {
            *blocker = "L0 QKV weight is not present in the fused weightpack: ";
            *blocker += wqkv->name;
        }
        return false;
    }
    if (use_direct_weights &&
            !qwen36_superlayer_tensor_data_on_device(wqkv, device, "L0 QKV weight", blocker)) {
        return false;
    }

    uint64_t qkv_scale_offset = 0;
    uint32_t qkv_scale_n = 0;
    if (qkv_scale != nullptr) {
        if (qkv_scale->type != GGML_TYPE_F32 || qkv_scale->nb[0] != (int64_t) sizeof(float)) {
            if (blocker != nullptr) {
                *blocker = "L0 QKV scale must be dense F32";
            }
            return false;
        }
        const int64_t scale_ne = ggml_nelements(qkv_scale);
        if (scale_ne != 1 && scale_ne != qkv_math->ne[0]) {
            if (blocker != nullptr) {
                *blocker = "L0 QKV scale has unexpected size";
            }
            return false;
        }
        if (!qwen36_superlayer_find_pack_offset(pack, qkv_scale, &qkv_scale_offset)) {
            if (blocker != nullptr) {
                *blocker = "L0 QKV scale is not present in the fused weightpack: ";
                *blocker += qkv_scale->name;
            }
            return false;
        }
        if (use_direct_weights &&
                !qwen36_superlayer_tensor_data_on_device(qkv_scale, device, "L0 QKV scale", blocker)) {
            return false;
        }
        qkv_scale_n = (uint32_t) scale_ne;
    }

    const uint64_t qkv_scratch_offset = runtime.activation_slot_bytes;
    const uint64_t qkv_bytes = (uint64_t) qkv_math->ne[0]*sizeof(float);
    if (qkv_scratch_offset + qkv_bytes > runtime.scratch_bytes) {
        if (blocker != nullptr) {
            *blocker = "L0 QKV scratch range does not fit in the superlayer runtime scratch";
        }
        return false;
    }

    desc->qkv_math_out = qkv_math != qkv_out && qkv_math != qkv_named &&
        qkv_math->data != qkv_out->data && qkv_math->data != qkv_named->data ?
        (float *) qkv_math->data : nullptr;
    desc->qkv_out = (float *) qkv_out->data;
    desc->qkv_named_out = qkv_named != qkv_out && qkv_named->data != qkv_out->data ?
        (float *) qkv_named->data : nullptr;
    desc->wqkv_data = (const char *) wqkv->data;
    desc->qkv_scale_data = qkv_scale != nullptr ? (const float *) qkv_scale->data : nullptr;
    desc->wqkv_offset = wqkv_offset;
    desc->qkv_scale_offset = qkv_scale_offset;
    desc->qkv_scratch_offset = qkv_scratch_offset;
    desc->wqkv_nb1 = (uint64_t) wqkv->nb[1];
    desc->n_embd = (uint32_t) wqkv->ne[0];
    desc->n_out = (uint32_t) qkv_math->ne[0];
    desc->wqkv_type = (int32_t) wqkv->type;
    desc->qkv_scale_n = qkv_scale_n;
    desc->has_qkv_scale = qkv_scale != nullptr ? 1u : 0u;
    desc->use_direct_weights = use_direct_weights ? 1u : 0u;
    desc->ready = 1u;
    return true;
}

static const ggml_tensor * qwen36_superlayer_resolve_mul_mat_with_scale(
        const ggml_tensor * node,
        const ggml_tensor ** scale) {
    if (scale != nullptr) {
        *scale = nullptr;
    }

    const ggml_tensor * out = qwen36_superlayer_strip_view_ops(node);
    const ggml_tensor * mm = out;
    if (out != nullptr && out->op == GGML_OP_MUL) {
        if (out->src[0] != nullptr && out->src[0]->op == GGML_OP_MUL_MAT) {
            mm = out->src[0];
            if (scale != nullptr) {
                *scale = out->src[1];
            }
        } else if (out->src[1] != nullptr && out->src[1]->op == GGML_OP_MUL_MAT) {
            mm = out->src[1];
            if (scale != nullptr) {
                *scale = out->src[0];
            }
        }
    }
    return mm != nullptr && mm->op == GGML_OP_MUL_MAT ? mm : nullptr;
}

static bool qwen36_superlayer_find_pack_offset_named(
        const qwen36_superlayer_pack_plan & pack,
        const ggml_tensor * tensor,
        const char * role,
        uint64_t * offset,
        std::string * blocker) {
    if (!qwen36_superlayer_find_pack_offset(pack, tensor, offset)) {
        if (blocker != nullptr) {
            *blocker = std::string(role) + " is not present in the fused weightpack: ";
            *blocker += tensor != nullptr ? tensor->name : "<null>";
        }
        return false;
    }
    return true;
}

static bool qwen36_superlayer_make_l0_proj_desc(
        const ggml_cgraph * cgraph,
        const qwen36_superlayer_plan & plan,
        const qwen36_superlayer_pack_plan & pack,
        const qwen36_superlayer_runtime_layout & runtime,
        const int device,
        qwen36_superlayer_l0_proj_desc * desc,
        std::string * blocker) {
    if (desc == nullptr) {
        if (blocker != nullptr) {
            *blocker = "missing L0 projection descriptor output";
        }
        return false;
    }
    *desc = qwen36_superlayer_l0_proj_desc{};

    const int begin = plan.layer_start[0] >= 0 ? plan.layer_start[0] : 0;
    const int end   = plan.layer_end[0]   >= begin ? plan.layer_end[0]   : cgraph->n_nodes - 1;
    const ggml_tensor * attn_norm = nullptr;
    const ggml_tensor * z = nullptr;
    const ggml_tensor * beta = nullptr;
    const ggml_tensor * beta_sigmoid = nullptr;
    const ggml_tensor * alpha = nullptr;
    const ggml_tensor * alpha_softplus = nullptr;
    const ggml_tensor * alpha_gate = nullptr;
    for (int i = begin; i <= end; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (attn_norm == nullptr && tensor_name_matches_layer(node, "attn_norm", 0)) {
            attn_norm = node;
        } else if (z == nullptr && tensor_name_matches_layer(node, "z", 0)) {
            z = node;
        } else if (beta == nullptr && tensor_name_matches_layer(node, "beta", 0)) {
            beta = node;
        } else if (beta_sigmoid == nullptr && tensor_name_matches_layer(node, "beta_sigmoid", 0)) {
            beta_sigmoid = node;
        } else if (alpha == nullptr && tensor_name_matches_layer(node, "alpha", 0)) {
            alpha = node;
        } else if (alpha_softplus == nullptr && tensor_name_matches_layer(node, "a_softplus", 0)) {
            alpha_softplus = node;
        } else if (alpha_gate == nullptr && tensor_name_matches_layer(node, "gate", 0)) {
            alpha_gate = node;
        }
    }

    if (attn_norm == nullptr || z == nullptr || beta == nullptr || beta_sigmoid == nullptr ||
            alpha == nullptr || alpha_softplus == nullptr || alpha_gate == nullptr) {
        if (blocker != nullptr) {
            *blocker = "L0 projection fusion requires z-0, beta-0, beta_sigmoid-0, alpha-0, a_softplus-0, and gate-0";
        }
        return false;
    }

    const ggml_tensor * z_scale = nullptr;
    const ggml_tensor * beta_scale = nullptr;
    const ggml_tensor * alpha_scale = nullptr;
    const ggml_tensor * z_math = qwen36_superlayer_resolve_mul_mat_with_scale(z, &z_scale);
    const ggml_tensor * beta_math = nullptr;
    if (beta_sigmoid->op == GGML_OP_UNARY && ggml_get_unary_op(beta_sigmoid) == GGML_UNARY_OP_SIGMOID) {
        if (!qwen36_superlayer_same_tensor_or_view(beta_sigmoid->src[0], beta)) {
            if (blocker != nullptr) {
                *blocker = "L0 projection beta_sigmoid-0 does not consume beta-0";
            }
            return false;
        }
        beta_math = qwen36_superlayer_resolve_mul_mat_with_scale(beta, &beta_scale);
    }
    const ggml_tensor * alpha_math = qwen36_superlayer_resolve_mul_mat_with_scale(alpha, &alpha_scale);
    if (z_math == nullptr || beta_math == nullptr || alpha_math == nullptr ||
            z_math->src[0] == nullptr || beta_math->src[0] == nullptr || alpha_math->src[0] == nullptr) {
        if (blocker != nullptr) {
            *blocker = "L0 projection fusion could not resolve z/beta/alpha MUL_MAT nodes";
        }
        return false;
    }
    if (z_math->src[1] != attn_norm || beta_math->src[1] != attn_norm || alpha_math->src[1] != attn_norm) {
        if (blocker != nullptr) {
            *blocker = "L0 projection MUL_MAT inputs are not attn_norm-0";
        }
        return false;
    }

    const ggml_tensor * alpha_base = qwen36_superlayer_strip_view_ops(alpha);
    const ggml_tensor * alpha_gate_math = qwen36_superlayer_strip_view_ops(alpha_gate);
    if (alpha_softplus->op != GGML_OP_UNARY ||
            ggml_get_unary_op(alpha_softplus) != GGML_UNARY_OP_SOFTPLUS ||
            alpha_softplus->src[0] == nullptr) {
        if (blocker != nullptr) {
            *blocker = "L0 projection fusion could not resolve a_softplus-0";
        }
        return false;
    }

    const ggml_tensor * alpha_biased = qwen36_superlayer_strip_view_ops(alpha_softplus->src[0]);
    if (alpha_biased == nullptr || alpha_biased->op != GGML_OP_ADD ||
            (!qwen36_superlayer_same_tensor_or_view(alpha_biased->src[0], alpha_base) &&
             !qwen36_superlayer_same_tensor_or_view(alpha_biased->src[1], alpha_base))) {
        alpha_biased = nullptr;
        for (int i = begin; i <= end; ++i) {
            const ggml_tensor * node = cgraph->nodes[i];
            if (node != nullptr && node->op == GGML_OP_ADD &&
                    (qwen36_superlayer_same_tensor_or_view(node->src[0], alpha_base) ||
                     qwen36_superlayer_same_tensor_or_view(node->src[1], alpha_base))) {
                alpha_biased = node;
                break;
            }
        }
    }
    if (alpha_biased == nullptr ||
            !qwen36_superlayer_same_tensor_or_view(alpha_softplus->src[0], alpha_biased)) {
        if (blocker != nullptr) {
            *blocker = "L0 projection fusion could not resolve alpha bias feeding a_softplus-0";
        }
        return false;
    }
    if (alpha_gate_math == nullptr || alpha_gate_math->op != GGML_OP_MUL) {
        if (blocker != nullptr) {
            *blocker = "L0 projection fusion could not resolve gate-0 to MUL";
        }
        return false;
    }

    const ggml_tensor * alpha_dt = nullptr;
    if (qwen36_superlayer_same_tensor_or_view(alpha_biased->src[0], alpha_base)) {
        alpha_dt = alpha_biased->src[1];
    } else if (qwen36_superlayer_same_tensor_or_view(alpha_biased->src[1], alpha_base)) {
        alpha_dt = alpha_biased->src[0];
    }
    const ggml_tensor * alpha_a = nullptr;
    if (qwen36_superlayer_same_tensor_or_view(alpha_gate_math->src[0], alpha_softplus)) {
        alpha_a = alpha_gate_math->src[1];
    } else if (qwen36_superlayer_same_tensor_or_view(alpha_gate_math->src[1], alpha_softplus)) {
        alpha_a = alpha_gate_math->src[0];
    }

    const ggml_tensor * wz = z_math->src[0];
    const ggml_tensor * wbeta = beta_math->src[0];
    const ggml_tensor * walpha = alpha_math->src[0];
    if (!qwen36_superlayer_projection_weight_supported(wz) ||
            !qwen36_superlayer_projection_weight_supported(wbeta) ||
            !qwen36_superlayer_projection_weight_supported(walpha)) {
        if (blocker != nullptr) {
            *blocker = "L0 projection weight type is not implemented in the superlayer";
        }
        return false;
    }
    if (z->type != GGML_TYPE_F32 || z_math->type != GGML_TYPE_F32 ||
            beta->type != GGML_TYPE_F32 || beta_math->type != GGML_TYPE_F32 ||
            beta_sigmoid->type != GGML_TYPE_F32 ||
            alpha->type != GGML_TYPE_F32 || alpha_math->type != GGML_TYPE_F32 ||
            alpha_biased->type != GGML_TYPE_F32 || alpha_softplus->type != GGML_TYPE_F32 ||
            alpha_gate->type != GGML_TYPE_F32 || alpha_dt == nullptr || alpha_a == nullptr ||
            alpha_dt->type != GGML_TYPE_F32 || alpha_a->type != GGML_TYPE_F32 ||
            (z_scale != nullptr && z_scale->type != GGML_TYPE_F32) ||
            (beta_scale != nullptr && beta_scale->type != GGML_TYPE_F32) ||
            (alpha_scale != nullptr && alpha_scale->type != GGML_TYPE_F32) ||
            z->nb[0] != (int64_t) sizeof(float) ||
            z_math->nb[0] != (int64_t) sizeof(float) ||
            beta->nb[0] != (int64_t) sizeof(float) ||
            beta_math->nb[0] != (int64_t) sizeof(float) ||
            beta_sigmoid->nb[0] != (int64_t) sizeof(float) ||
            alpha->nb[0] != (int64_t) sizeof(float) ||
            alpha_math->nb[0] != (int64_t) sizeof(float) ||
            alpha_biased->nb[0] != (int64_t) sizeof(float) ||
            alpha_softplus->nb[0] != (int64_t) sizeof(float) ||
            alpha_gate->nb[0] != (int64_t) sizeof(float) ||
            alpha_dt->nb[0] != (int64_t) sizeof(float) ||
            alpha_a->nb[0] != (int64_t) sizeof(float) ||
            (z_scale != nullptr && z_scale->nb[0] != (int64_t) sizeof(float)) ||
            (beta_scale != nullptr && beta_scale->nb[0] != (int64_t) sizeof(float)) ||
            (alpha_scale != nullptr && alpha_scale->nb[0] != (int64_t) sizeof(float)) ||
            !ggml_is_contiguous_1(z) ||
            !ggml_is_contiguous_1(z_math) ||
            !ggml_is_contiguous_1(beta) ||
            !ggml_is_contiguous_1(beta_math) ||
            !ggml_is_contiguous_1(beta_sigmoid) ||
            !ggml_is_contiguous_1(alpha) ||
            !ggml_is_contiguous_1(alpha_math) ||
            !ggml_is_contiguous_1(alpha_biased) ||
            !ggml_is_contiguous_1(alpha_softplus) ||
            !ggml_is_contiguous_1(alpha_gate) ||
            (z_scale != nullptr && !ggml_is_contiguous_1(z_scale)) ||
            (beta_scale != nullptr && !ggml_is_contiguous_1(beta_scale)) ||
            (alpha_scale != nullptr && !ggml_is_contiguous_1(alpha_scale))) {
        if (blocker != nullptr) {
            *blocker = "L0 projection fusion requires dense F32 projection outputs and alpha constants";
        }
        return false;
    }

    const int64_t n_embd = attn_norm->ne[0];
    if (n_embd <= 0 || wz->ne[0] != n_embd || wbeta->ne[0] != n_embd || walpha->ne[0] != n_embd ||
            z_math->ne[0] != wz->ne[1] || beta_math->ne[0] != wbeta->ne[1] ||
            alpha_math->ne[0] != walpha->ne[1]) {
        if (blocker != nullptr) {
            *blocker = "L0 projection tensor dimensions do not match";
        }
        return false;
    }
    if (ggml_nelements(z) != z_math->ne[0] ||
            ggml_nelements(z_math) != z_math->ne[0] ||
            ggml_nelements(beta) != beta_math->ne[0] ||
            ggml_nelements(beta_math) != beta_math->ne[0] ||
            ggml_nelements(beta_sigmoid) != beta_math->ne[0] ||
            ggml_nelements(alpha) != alpha_math->ne[0] ||
            ggml_nelements(alpha_math) != alpha_math->ne[0] ||
            ggml_nelements(alpha_biased) != alpha_math->ne[0] ||
            ggml_nelements(alpha_softplus) != alpha_math->ne[0] ||
            ggml_nelements(alpha_gate) != alpha_math->ne[0] ||
            ggml_nelements(alpha_dt) != ggml_nelements(alpha_gate) ||
            ggml_nelements(alpha_a) != ggml_nelements(alpha_gate) ||
            (z_scale != nullptr && ggml_nelements(z_scale) != 1 && ggml_nelements(z_scale) != z_math->ne[0]) ||
            (beta_scale != nullptr && ggml_nelements(beta_scale) != 1 && ggml_nelements(beta_scale) != beta_math->ne[0]) ||
            (alpha_scale != nullptr && ggml_nelements(alpha_scale) != 1 && ggml_nelements(alpha_scale) != alpha_math->ne[0])) {
        if (blocker != nullptr) {
            *blocker = "L0 projection output/token dimensions do not match";
        }
        return false;
    }
    if (!qwen36_superlayer_tensor_data_on_device(z, device, "L0 z output", blocker) ||
            (z_math != z && !qwen36_superlayer_tensor_data_on_device(z_math, device, "L0 z math output", blocker)) ||
            !qwen36_superlayer_tensor_data_on_device(beta, device, "L0 beta raw output", blocker) ||
            (beta_math != beta && !qwen36_superlayer_tensor_data_on_device(beta_math, device, "L0 beta math output", blocker)) ||
            !qwen36_superlayer_tensor_data_on_device(beta_sigmoid, device, "L0 beta output", blocker) ||
            !qwen36_superlayer_tensor_data_on_device(alpha, device, "L0 alpha raw output", blocker) ||
            (alpha_math != alpha && !qwen36_superlayer_tensor_data_on_device(alpha_math, device, "L0 alpha math output", blocker)) ||
            !qwen36_superlayer_tensor_data_on_device(alpha_biased, device, "L0 alpha biased output", blocker) ||
            !qwen36_superlayer_tensor_data_on_device(alpha_softplus, device, "L0 alpha softplus output", blocker) ||
            !qwen36_superlayer_tensor_data_on_device(alpha_gate, device, "L0 alpha gate output", blocker)) {
        return false;
    }

    const bool use_direct_weights = qwen36_superlayer_direct_l0_proj_weights_requested();
    if (use_direct_weights) {
        if (!qwen36_superlayer_tensor_data_on_device(wz, device, "L0 z weight", blocker) ||
                !qwen36_superlayer_tensor_data_on_device(wbeta, device, "L0 beta weight", blocker) ||
                !qwen36_superlayer_tensor_data_on_device(walpha, device, "L0 alpha weight", blocker) ||
                !qwen36_superlayer_tensor_data_on_device(alpha_dt, device, "L0 alpha dt", blocker) ||
                !qwen36_superlayer_tensor_data_on_device(alpha_a, device, "L0 alpha gate scale", blocker) ||
                (z_scale != nullptr &&
                 !qwen36_superlayer_tensor_data_on_device(z_scale, device, "L0 z post scale", blocker)) ||
                (beta_scale != nullptr &&
                 !qwen36_superlayer_tensor_data_on_device(beta_scale, device, "L0 beta post scale", blocker)) ||
                (alpha_scale != nullptr &&
                 !qwen36_superlayer_tensor_data_on_device(alpha_scale, device, "L0 alpha post scale", blocker))) {
            return false;
        }
    }

    uint64_t wz_offset = 0;
    uint64_t wbeta_offset = 0;
    uint64_t walpha_offset = 0;
    uint64_t alpha_dt_offset = 0;
    uint64_t alpha_a_offset = 0;
    uint64_t z_scale_offset = 0;
    uint64_t beta_scale_offset = 0;
    uint64_t alpha_scale_offset = 0;
    if (!qwen36_superlayer_find_pack_offset_named(pack, wz, "L0 z weight", &wz_offset, blocker) ||
            !qwen36_superlayer_find_pack_offset_named(pack, wbeta, "L0 beta weight", &wbeta_offset, blocker) ||
            !qwen36_superlayer_find_pack_offset_named(pack, walpha, "L0 alpha weight", &walpha_offset, blocker) ||
            !qwen36_superlayer_find_pack_offset_named(pack, alpha_dt, "L0 alpha dt", &alpha_dt_offset, blocker) ||
            !qwen36_superlayer_find_pack_offset_named(pack, alpha_a, "L0 alpha gate scale", &alpha_a_offset, blocker)) {
        return false;
    }
    if (z_scale != nullptr &&
            !qwen36_superlayer_find_pack_offset_named(pack, z_scale, "L0 z post scale", &z_scale_offset, blocker)) {
        return false;
    }
    if (beta_scale != nullptr &&
            !qwen36_superlayer_find_pack_offset_named(pack, beta_scale, "L0 beta post scale", &beta_scale_offset, blocker)) {
        return false;
    }
    if (alpha_scale != nullptr &&
            !qwen36_superlayer_find_pack_offset_named(pack, alpha_scale, "L0 alpha post scale", &alpha_scale_offset, blocker)) {
        return false;
    }

    const uint64_t z_bytes = (uint64_t) ggml_nelements(z)*sizeof(float);
    const uint64_t beta_bytes = (uint64_t) ggml_nelements(beta_sigmoid)*sizeof(float);
    const uint64_t alpha_bytes = (uint64_t) ggml_nelements(alpha_gate)*sizeof(float);
    const uint64_t proj_base = 2*runtime.activation_slot_bytes;
    const uint64_t z_off = proj_base;
    const uint64_t beta_off = z_off + align_up(z_bytes, 256);
    const uint64_t alpha_off = beta_off + align_up(beta_bytes, 256);
    if (alpha_off + alpha_bytes > proj_base + runtime.projection_slot_bytes ||
            alpha_off + alpha_bytes > runtime.scratch_bytes) {
        if (blocker != nullptr) {
            *blocker = "L0 projection scratch range does not fit in the superlayer runtime scratch";
        }
        return false;
    }

    desc->z_math_dst = z_math != z && z_math->data != z->data ? (float *) z_math->data : nullptr;
    desc->z_dst = (float *) z->data;
    desc->beta_math_dst = beta_math != beta && beta_math->data != beta->data ? (float *) beta_math->data : nullptr;
    desc->beta_raw_dst = (float *) beta->data;
    desc->beta_dst = (float *) beta_sigmoid->data;
    desc->alpha_math_dst = alpha_math != alpha && alpha_math->data != alpha->data ? (float *) alpha_math->data : nullptr;
    desc->alpha_raw_dst = (float *) alpha->data;
    desc->alpha_biased_dst = (float *) alpha_biased->data;
    desc->alpha_softplus_dst = (float *) alpha_softplus->data;
    desc->alpha_dst = (float *) alpha_gate->data;
    desc->wz_data = (const char *) wz->data;
    desc->wbeta_data = (const char *) wbeta->data;
    desc->walpha_data = (const char *) walpha->data;
    desc->alpha_dt_data = (const float *) alpha_dt->data;
    desc->alpha_a_data = (const float *) alpha_a->data;
    desc->z_scale_data = z_scale != nullptr ? (const float *) z_scale->data : nullptr;
    desc->beta_scale_data = beta_scale != nullptr ? (const float *) beta_scale->data : nullptr;
    desc->alpha_scale_data = alpha_scale != nullptr ? (const float *) alpha_scale->data : nullptr;
    desc->wz_offset = wz_offset;
    desc->wbeta_offset = wbeta_offset;
    desc->walpha_offset = walpha_offset;
    desc->alpha_dt_offset = alpha_dt_offset;
    desc->alpha_a_offset = alpha_a_offset;
    desc->z_scale_offset = z_scale_offset;
    desc->beta_scale_offset = beta_scale_offset;
    desc->alpha_scale_offset = alpha_scale_offset;
    desc->z_scratch_offset = z_off;
    desc->beta_scratch_offset = beta_off;
    desc->alpha_scratch_offset = alpha_off;
    desc->wz_nb1 = (uint64_t) wz->nb[1];
    desc->wbeta_nb1 = (uint64_t) wbeta->nb[1];
    desc->walpha_nb1 = (uint64_t) walpha->nb[1];
    desc->n_embd = (uint32_t) n_embd;
    desc->z_out = (uint32_t) z_math->ne[0];
    desc->beta_out = (uint32_t) beta_math->ne[0];
    desc->alpha_out = (uint32_t) alpha_math->ne[0];
    desc->wz_type = (int32_t) wz->type;
    desc->wbeta_type = (int32_t) wbeta->type;
    desc->walpha_type = (int32_t) walpha->type;
    desc->z_scale_n = z_scale != nullptr ? (uint32_t) ggml_nelements(z_scale) : 0u;
    desc->beta_scale_n = beta_scale != nullptr ? (uint32_t) ggml_nelements(beta_scale) : 0u;
    desc->alpha_scale_n = alpha_scale != nullptr ? (uint32_t) ggml_nelements(alpha_scale) : 0u;
    desc->has_z_scale = z_scale != nullptr ? 1u : 0u;
    desc->has_beta_scale = beta_scale != nullptr ? 1u : 0u;
    desc->has_alpha_scale = alpha_scale != nullptr ? 1u : 0u;
    desc->use_direct_weights = use_direct_weights ? 1u : 0u;
    desc->ready = 1u;
    return true;
}

static bool qwen36_superlayer_l0_ssm_requested() {
    return qwen36_superlayer_replace_l0_all_stages_requested() ||
        qwen36_superlayer_env_i64("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REPLACE_L0_SSM", 0) != 0;
}

static bool qwen36_superlayer_make_l0_ssm_desc(
        const ggml_cgraph * cgraph,
        const qwen36_superlayer_plan & plan,
        const qwen36_superlayer_pack_plan & pack,
        const int device,
        qwen36_superlayer_l0_ssm_desc * desc,
        std::string * blocker) {
    if (desc == nullptr) {
        if (blocker != nullptr) {
            *blocker = "missing L0 SSM descriptor output";
        }
        return false;
    }
    *desc = qwen36_superlayer_l0_ssm_desc{};

    const bool requested = qwen36_superlayer_l0_ssm_requested();
    const int begin = plan.layer_start[0] >= 0 ? plan.layer_start[0] : 0;
    const int end   = plan.layer_end[0]   >= begin ? plan.layer_end[0]   : cgraph->n_nodes - 1;
    const ggml_tensor * conv_raw = nullptr;
    const ggml_tensor * conv_silu = nullptr;
    for (int i = begin; i <= end; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (conv_raw == nullptr && tensor_name_matches_layer(node, "conv_output_raw", 0)) {
            conv_raw = node;
        } else if (conv_silu == nullptr && tensor_name_matches_layer(node, "conv_output_silu", 0)) {
            conv_silu = node;
        }
    }

    if (conv_raw == nullptr || conv_silu == nullptr) {
        if (requested && blocker != nullptr) {
            *blocker = "L0 SSM fusion requires conv_output_raw-0 and conv_output_silu-0";
        }
        return !requested;
    }
    if (conv_raw->op != GGML_OP_SSM_CONV || conv_silu->op != GGML_OP_UNARY ||
            ggml_get_unary_op(conv_silu) != GGML_UNARY_OP_SILU || conv_silu->src[0] != conv_raw) {
        if (requested && blocker != nullptr) {
            *blocker = "L0 SSM fusion requires SSM_CONV followed by SILU";
        }
        return !requested;
    }

    const ggml_tensor * conv_input = conv_raw->src[0];
    const ggml_tensor * conv_w = conv_raw->src[1];
    const ggml_tensor * state_out = conv_raw->src[2];
    const ggml_tensor * state = conv_raw->src[3];
    const ggml_tensor * token = conv_raw->src[4];
    if (conv_input == nullptr || conv_w == nullptr || state_out == nullptr ||
            state == nullptr || token == nullptr) {
        if (requested && blocker != nullptr) {
            *blocker = "L0 SSM fusion requires state-token concat and in-place state target";
        }
        return !requested;
    }
    if (conv_input->op != GGML_OP_CONCAT || ggml_get_op_params_i32(conv_input, 0) != 0 ||
            conv_input->src[0] != state || conv_input->src[1] != token) {
        if (requested && blocker != nullptr) {
            *blocker = "L0 SSM fusion requires conv_input-0 to be concat(state, token)";
        }
        return !requested;
    }
    if (conv_w->type != GGML_TYPE_F32 || state->type != GGML_TYPE_F32 || token->type != GGML_TYPE_F32 ||
            conv_raw->type != GGML_TYPE_F32 || conv_silu->type != GGML_TYPE_F32 ||
            state_out->type != GGML_TYPE_F32) {
        if (requested && blocker != nullptr) {
            *blocker = "L0 SSM fusion currently requires F32 state, token, weight, output, and state target";
        }
        return !requested;
    }

    const int64_t d_conv = conv_w->ne[0];
    const int64_t n_channels = conv_w->ne[1];
    const int64_t n_seqs = conv_silu->ne[2];
    if (d_conv <= 1 || d_conv > 16 || n_channels <= 0 || n_seqs != 1 ||
            conv_raw->ne[0] != n_channels || conv_silu->ne[0] != n_channels ||
            conv_raw->ne[1] != 1 || conv_silu->ne[1] != 1 ||
            conv_raw->ne[2] != n_seqs ||
            conv_input->ne[0] != d_conv || conv_input->ne[1] != n_channels || conv_input->ne[2] != n_seqs ||
            state->ne[0] != d_conv - 1 || state->ne[1] != n_channels || state->ne[2] != n_seqs ||
            token->ne[0] != 1 || token->ne[1] != n_channels || token->ne[2] != n_seqs ||
            state_out->ne[0] != (d_conv - 1)*n_channels || state_out->ne[1] != n_seqs) {
        if (requested && blocker != nullptr) {
            *blocker = "L0 SSM fusion tensor dimensions do not match state-token decode";
        }
        return !requested;
    }
    if (state->nb[0] != (int64_t) sizeof(float) || token->nb[1] != (int64_t) sizeof(float) ||
            conv_w->nb[0] != (int64_t) sizeof(float) ||
            conv_raw->nb[0] != (int64_t) sizeof(float) || conv_silu->nb[0] != (int64_t) sizeof(float) ||
            !ggml_is_contiguous(state_out)) {
        if (requested && blocker != nullptr) {
            *blocker = "L0 SSM fusion requires dense F32 channel/state layout";
        }
        return !requested;
    }
    if (!qwen36_superlayer_tensor_data_on_device(state, device, "L0 SSM conv state", blocker) ||
            !qwen36_superlayer_tensor_data_on_device(token, device, "L0 SSM token", blocker) ||
            !qwen36_superlayer_tensor_data_on_device(conv_raw, device, "L0 SSM raw output", blocker) ||
            !qwen36_superlayer_tensor_data_on_device(conv_silu, device, "L0 SSM silu output", blocker) ||
            !qwen36_superlayer_tensor_data_on_device(state_out, device, "L0 SSM state output", blocker)) {
        return false;
    }

    const bool use_direct_weights = qwen36_superlayer_direct_l0_ssm_weights_requested();
    if (use_direct_weights &&
            !qwen36_superlayer_tensor_data_on_device(conv_w, device, "L0 SSM conv weight", blocker)) {
        return false;
    }

    uint64_t conv_w_offset = 0;
    if (!qwen36_superlayer_find_pack_offset_named(pack, conv_w, "L0 SSM conv weight", &conv_w_offset, blocker)) {
        return false;
    }

    desc->state = (const float *) state->data;
    desc->token = (const float *) token->data;
    desc->conv_w_data = (const float *) conv_w->data;
    desc->raw_dst = (float *) conv_raw->data;
    desc->silu_dst = (float *) conv_silu->data;
    desc->state_out = (float *) state_out->data;
    desc->conv_w_offset = conv_w_offset;
    desc->state_nb1 = (uint64_t) state->nb[1];
    desc->state_nb2 = (uint64_t) state->nb[2];
    desc->token_nb1 = (uint64_t) token->nb[1];
    desc->token_nb2 = (uint64_t) token->nb[2];
    desc->conv_w_nb1 = (uint64_t) conv_w->nb[1];
    desc->raw_nb0 = (uint64_t) conv_raw->nb[0];
    desc->raw_nb2 = (uint64_t) conv_raw->nb[2];
    desc->silu_nb0 = (uint64_t) conv_silu->nb[0];
    desc->silu_nb2 = (uint64_t) conv_silu->nb[2];
    desc->state_out_nb1 = (uint64_t) state_out->nb[1];
    desc->d_conv = (uint32_t) d_conv;
    desc->n_channels = (uint32_t) n_channels;
    desc->n_seqs = (uint32_t) n_seqs;
    desc->use_direct_weights = use_direct_weights ? 1u : 0u;
    desc->ready = 1u;
    return true;
}

static bool qwen36_superlayer_l0_l2_requested() {
    return qwen36_superlayer_replace_l0_all_stages_requested() ||
        qwen36_superlayer_env_i64("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REPLACE_L0_L2", 0) != 0;
}

static bool qwen36_superlayer_make_l0_l2_desc(
        const ggml_cgraph * cgraph,
        const qwen36_superlayer_plan & plan,
        const int device,
        qwen36_superlayer_l0_l2_desc * desc,
        std::string * blocker) {
    if (desc == nullptr) {
        if (blocker != nullptr) {
            *blocker = "missing L0 L2Norm descriptor output";
        }
        return false;
    }
    *desc = qwen36_superlayer_l0_l2_desc{};

    const bool requested = qwen36_superlayer_l0_l2_requested();
    if (!requested) {
        return true;
    }

    const int begin = plan.layer_start[0] >= 0 ? plan.layer_start[0] : 0;
    const int end   = plan.layer_end[0]   >= begin ? plan.layer_end[0]   : cgraph->n_nodes - 1;
    const ggml_tensor * q_l2 = nullptr;
    const ggml_tensor * k_l2 = nullptr;
    for (int i = begin; i <= end; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (q_l2 == nullptr && tensor_name_matches_layer(node, "q_conv_predelta", 0)) {
            q_l2 = node;
        } else if (k_l2 == nullptr && tensor_name_matches_layer(node, "k_conv_predelta", 0)) {
            k_l2 = node;
        }
    }

    if (q_l2 == nullptr || k_l2 == nullptr ||
            q_l2->op != GGML_OP_L2_NORM || k_l2->op != GGML_OP_L2_NORM ||
            q_l2->src[0] == nullptr || k_l2->src[0] == nullptr) {
        if (blocker != nullptr) {
            *blocker = "L0 L2Norm fusion requires q_conv_predelta-0 and k_conv_predelta-0 L2_NORM nodes";
        }
        return false;
    }

    const ggml_tensor * q_src = q_l2->src[0];
    const ggml_tensor * k_src = k_l2->src[0];
    if (q_src->type != GGML_TYPE_F32 || k_src->type != GGML_TYPE_F32 ||
            q_l2->type != GGML_TYPE_F32 || k_l2->type != GGML_TYPE_F32) {
        if (blocker != nullptr) {
            *blocker = "L0 L2Norm fusion currently requires F32 Q/K tensors";
        }
        return false;
    }
    if (!ggml_are_same_shape(q_src, k_src) ||
            !ggml_are_same_shape(q_src, q_l2) ||
            !ggml_are_same_shape(k_src, k_l2) ||
            q_src->ne[0] <= 0 || q_src->ne[1] <= 0 ||
            q_src->ne[2] != 1 || q_src->ne[3] != 1) {
        if (blocker != nullptr) {
            *blocker = "L0 L2Norm fusion requires matching one-token Q/K decode tensors";
        }
        return false;
    }
    if (ggml_get_op_params_f32(q_l2, 0) != ggml_get_op_params_f32(k_l2, 0)) {
        if (blocker != nullptr) {
            *blocker = "L0 L2Norm fusion requires identical Q/K epsilon";
        }
        return false;
    }
    if (ggml_get_op_params_f32(q_l2, 0) < 0.0f) {
        if (blocker != nullptr) {
            *blocker = "L0 L2Norm epsilon is negative";
        }
        return false;
    }
    if (q_src->nb[0] != (int64_t) sizeof(float) ||
            k_src->nb[0] != (int64_t) sizeof(float) ||
            q_l2->nb[0] != (int64_t) sizeof(float) ||
            k_l2->nb[0] != (int64_t) sizeof(float) ||
            !ggml_is_contiguous_rows(q_src) ||
            !ggml_is_contiguous_rows(k_src) ||
            !ggml_is_contiguous(q_l2) ||
            !ggml_is_contiguous(k_l2)) {
        if (blocker != nullptr) {
            *blocker = "L0 L2Norm fusion requires dense row-contiguous Q/K layout";
        }
        return false;
    }
    if (!qwen36_superlayer_tensor_data_on_device(q_src, device, "L0 Q L2 input", blocker) ||
            !qwen36_superlayer_tensor_data_on_device(k_src, device, "L0 K L2 input", blocker) ||
            !qwen36_superlayer_tensor_data_on_device(q_l2, device, "L0 Q L2 output", blocker) ||
            !qwen36_superlayer_tensor_data_on_device(k_l2, device, "L0 K L2 output", blocker)) {
        return false;
    }

    desc->q_src = (const float *) q_src->data;
    desc->k_src = (const float *) k_src->data;
    desc->q_dst = (float *) q_l2->data;
    desc->k_dst = (float *) k_l2->data;
    desc->q_src_nb1 = (uint64_t) q_src->nb[1];
    desc->q_src_nb2 = (uint64_t) q_src->nb[2];
    desc->q_src_nb3 = (uint64_t) q_src->nb[3];
    desc->k_src_nb1 = (uint64_t) k_src->nb[1];
    desc->k_src_nb2 = (uint64_t) k_src->nb[2];
    desc->k_src_nb3 = (uint64_t) k_src->nb[3];
    desc->q_dst_nb1 = (uint64_t) q_l2->nb[1];
    desc->q_dst_nb2 = (uint64_t) q_l2->nb[2];
    desc->q_dst_nb3 = (uint64_t) q_l2->nb[3];
    desc->k_dst_nb1 = (uint64_t) k_l2->nb[1];
    desc->k_dst_nb2 = (uint64_t) k_l2->nb[2];
    desc->k_dst_nb3 = (uint64_t) k_l2->nb[3];
    desc->ncols = (uint32_t) q_src->ne[0];
    desc->nrows = (uint32_t) q_src->ne[1];
    desc->nchannels = (uint32_t) q_src->ne[2];
    desc->nsamples = (uint32_t) q_src->ne[3];
    desc->eps = ggml_get_op_params_f32(q_l2, 0);
    desc->ready = 1u;
    return true;
}

static bool qwen36_superlayer_l0_gdn_requested() {
    return qwen36_superlayer_replace_l0_all_stages_requested() ||
        qwen36_superlayer_env_i64("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REPLACE_L0_GDN", 0) != 0;
}

static bool qwen36_superlayer_make_l0_gdn_desc(
        const ggml_cgraph * cgraph,
        const qwen36_superlayer_plan & plan,
        const int device,
        qwen36_superlayer_l0_gdn_desc * desc,
        std::string * blocker) {
    if (desc == nullptr) {
        if (blocker != nullptr) {
            *blocker = "missing L0 GDN descriptor output";
        }
        return false;
    }
    *desc = qwen36_superlayer_l0_gdn_desc{};

    const bool requested = qwen36_superlayer_l0_gdn_requested();
    if (!requested) {
        return true;
    }

    const int begin = plan.layer_start[0] >= 0 ? plan.layer_start[0] : 0;
    const int end   = plan.layer_end[0]   >= begin ? plan.layer_end[0]   : cgraph->n_nodes - 1;
    const ggml_tensor * gdn = nullptr;
    for (int i = begin; i <= end; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (tensor_name_matches_layer(node, "__fgdn_ar__", 0)) {
            gdn = node;
            break;
        }
    }

    if (gdn == nullptr || gdn->op != GGML_OP_GATED_DELTA_NET) {
        if (blocker != nullptr) {
            *blocker = "L0 GDN fusion requires __fgdn_ar__-0 GATED_DELTA_NET";
        }
        return false;
    }

    const ggml_tensor * q = gdn->src[0];
    const ggml_tensor * k = gdn->src[1];
    const ggml_tensor * v = gdn->src[2];
    const ggml_tensor * g = gdn->src[3];
    const ggml_tensor * beta = gdn->src[4];
    const ggml_tensor * state = gdn->src[5];
    const ggml_tensor * state_out = gdn->src[6];
    if (q == nullptr || k == nullptr || v == nullptr || g == nullptr || beta == nullptr ||
            state == nullptr || state_out == nullptr || ggml_get_op_params_i32(gdn, 0) == 0) {
        if (blocker != nullptr) {
            *blocker = "L0 GDN fusion requires autoregressive in-place GDN state output";
        }
        return false;
    }
    if (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F32 || v->type != GGML_TYPE_F32 ||
            g->type != GGML_TYPE_F32 || beta->type != GGML_TYPE_F32 ||
            state->type != GGML_TYPE_F32 || state_out->type != GGML_TYPE_F32 ||
            gdn->type != GGML_TYPE_F32) {
        if (blocker != nullptr) {
            *blocker = "L0 GDN fusion currently requires F32 tensors";
        }
        return false;
    }

    const int64_t s_v = v->ne[0];
    const int64_t h_v = v->ne[1];
    const int64_t h_k = q->ne[1];
    const int64_t n_tokens = v->ne[2];
    const int64_t n_seqs = v->ne[3];
    const bool kda = g->ne[0] == s_v;
    if (s_v <= 0 || h_v <= 0 || h_k <= 0 || n_tokens != 1 || n_seqs != 1 ||
            q->ne[0] != s_v || k->ne[0] != s_v ||
            k->ne[1] != h_k || q->ne[2] != 1 || k->ne[2] != 1 ||
            q->ne[3] != n_seqs || k->ne[3] != n_seqs ||
            h_v % h_k != 0 ||
            g->ne[0] != (kda ? s_v : 1) || g->ne[1] != h_v || g->ne[2] != 1 || g->ne[3] != n_seqs ||
            beta->ne[0] != 1 || beta->ne[1] != h_v || beta->ne[2] != 1 || beta->ne[3] != n_seqs ||
            state->ne[0] != s_v || state->ne[1] != s_v || state->ne[2] != h_v || state->ne[3] != n_seqs ||
            ggml_nelements(state_out) != s_v*s_v*h_v*n_seqs ||
            ggml_nelements(gdn) != s_v*h_v*n_seqs) {
        if (blocker != nullptr) {
            *blocker = "L0 GDN fusion tensor dimensions do not match one-token decode";
        }
        return false;
    }
    if (!ggml_is_contiguous_rows(q) || !ggml_is_contiguous_rows(k) || !ggml_is_contiguous_rows(v) ||
            !ggml_is_contiguous(g) || !ggml_is_contiguous(beta) ||
            !ggml_is_contiguous(state) || !ggml_is_contiguous(state_out) ||
            !ggml_is_contiguous(gdn)) {
        if (blocker != nullptr) {
            *blocker = "L0 GDN fusion requires contiguous F32 GDN layout";
        }
        return false;
    }
    if (!qwen36_superlayer_tensor_data_on_device(q, device, "L0 GDN Q", blocker) ||
            !qwen36_superlayer_tensor_data_on_device(k, device, "L0 GDN K", blocker) ||
            !qwen36_superlayer_tensor_data_on_device(v, device, "L0 GDN V", blocker) ||
            !qwen36_superlayer_tensor_data_on_device(g, device, "L0 GDN gate", blocker) ||
            !qwen36_superlayer_tensor_data_on_device(beta, device, "L0 GDN beta", blocker) ||
            !qwen36_superlayer_tensor_data_on_device(state, device, "L0 GDN state", blocker) ||
            !qwen36_superlayer_tensor_data_on_device(state_out, device, "L0 GDN state output", blocker) ||
            !qwen36_superlayer_tensor_data_on_device(gdn, device, "L0 GDN output", blocker)) {
        return false;
    }

    desc->q = (const float *) q->data;
    desc->k = (const float *) k->data;
    desc->v = (const float *) v->data;
    desc->g = (const float *) g->data;
    desc->beta = (const float *) beta->data;
    desc->state = (const float *) state->data;
    desc->dst = (float *) gdn->data;
    desc->state_out = (float *) state_out->data;
    desc->q_nb1 = (uint64_t) q->nb[1];
    desc->q_nb3 = (uint64_t) q->nb[3];
    desc->k_nb1 = (uint64_t) k->nb[1];
    desc->k_nb3 = (uint64_t) k->nb[3];
    desc->v_nb1 = (uint64_t) v->nb[1];
    desc->v_nb3 = (uint64_t) v->nb[3];
    desc->g_nb1 = (uint64_t) g->nb[1];
    desc->g_nb3 = (uint64_t) g->nb[3];
    desc->beta_nb1 = (uint64_t) beta->nb[1];
    desc->beta_nb3 = (uint64_t) beta->nb[3];
    desc->state_nb2 = (uint64_t) state->nb[2];
    desc->state_nb3 = (uint64_t) state->nb[3];
    desc->state_out_nb2 = (uint64_t) state_out->nb[2];
    desc->state_out_nb3 = (uint64_t) state_out->nb[3];
    desc->s_v = (uint32_t) s_v;
    desc->h_v = (uint32_t) h_v;
    desc->h_k = (uint32_t) h_k;
    desc->n_seqs = (uint32_t) n_seqs;
    desc->kda = kda ? 1u : 0u;
    desc->ready = 1u;
    return true;
}

static bool qwen36_superlayer_l0_gated_norm_requested() {
    return qwen36_superlayer_replace_l0_all_stages_requested() ||
        qwen36_superlayer_env_i64("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REPLACE_L0_GATED_NORM", 0) != 0;
}

static bool qwen36_superlayer_make_l0_gated_norm_desc(
        const ggml_cgraph * cgraph,
        const qwen36_superlayer_plan & plan,
        const qwen36_superlayer_pack_plan & pack,
        const int device,
        qwen36_superlayer_l0_gated_norm_desc * desc,
        std::string * blocker) {
    if (desc == nullptr) {
        if (blocker != nullptr) {
            *blocker = "missing L0 gated norm descriptor output";
        }
        return false;
    }
    *desc = qwen36_superlayer_l0_gated_norm_desc{};

    const bool requested = qwen36_superlayer_l0_gated_norm_requested();
    if (!requested) {
        return true;
    }

    const int begin = plan.layer_start[0] >= 0 ? plan.layer_start[0] : 0;
    const int end   = plan.layer_end[0]   >= begin ? plan.layer_end[0]   : cgraph->n_nodes - 1;
    const ggml_tensor * attn_output = nullptr;
    const ggml_tensor * final_output = nullptr;
    for (int i = begin; i <= end; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (attn_output == nullptr && tensor_name_matches_layer(node, "attn_output", 0)) {
            attn_output = node;
        } else if (final_output == nullptr && tensor_name_matches_layer(node, "final_output", 0)) {
            final_output = node;
        }
    }

    const ggml_tensor * out = qwen36_superlayer_strip_view_ops(final_output);
    if (attn_output == nullptr || final_output == nullptr || out == nullptr || out->op != GGML_OP_MUL) {
        if (blocker != nullptr) {
            *blocker = "L0 gated norm fusion requires attn_output-0 and final_output-0";
        }
        return false;
    }

    const ggml_tensor * norm_mul = nullptr;
    const ggml_tensor * silu = nullptr;
    if (out->src[0] != nullptr && out->src[0]->op == GGML_OP_UNARY &&
            ggml_get_unary_op(out->src[0]) == GGML_UNARY_OP_SILU) {
        silu = out->src[0];
        norm_mul = out->src[1];
    } else if (out->src[1] != nullptr && out->src[1]->op == GGML_OP_UNARY &&
            ggml_get_unary_op(out->src[1]) == GGML_UNARY_OP_SILU) {
        silu = out->src[1];
        norm_mul = out->src[0];
    }
    if (norm_mul == nullptr || silu == nullptr || silu->src[0] == nullptr) {
        if (blocker != nullptr) {
            *blocker = "L0 gated norm fusion requires MUL(RMSNorm*weight, SILU(z))";
        }
        return false;
    }

    const ggml_tensor * rms = nullptr;
    const ggml_tensor * norm_w = nullptr;
    if (norm_mul->op == GGML_OP_MUL) {
        if (norm_mul->src[0] != nullptr && norm_mul->src[0]->op == GGML_OP_RMS_NORM) {
            rms = norm_mul->src[0];
            norm_w = norm_mul->src[1];
        } else if (norm_mul->src[1] != nullptr && norm_mul->src[1]->op == GGML_OP_RMS_NORM) {
            rms = norm_mul->src[1];
            norm_w = norm_mul->src[0];
        }
    } else if (norm_mul->op == GGML_OP_RMS_NORM) {
        rms = norm_mul;
    }
    if (rms == nullptr || rms->src[0] == nullptr ||
            !qwen36_superlayer_same_tensor_or_view(rms->src[0], attn_output)) {
        if (blocker != nullptr) {
            *blocker = "L0 gated norm fusion could not resolve RMS_NORM(attn_output-0)";
        }
        return false;
    }

    const ggml_tensor * gate = silu->src[0];
    if (rms->type != GGML_TYPE_F32 || rms->src[0]->type != GGML_TYPE_F32 ||
            gate->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32 ||
            norm_mul->type != GGML_TYPE_F32 || out->type != GGML_TYPE_F32 ||
            final_output->type != GGML_TYPE_F32 ||
            (norm_w != nullptr && norm_w->type != GGML_TYPE_F32)) {
        if (blocker != nullptr) {
            *blocker = "L0 gated norm fusion currently requires F32 tensors";
        }
        return false;
    }

    const int64_t ncols = rms->src[0]->ne[0];
    const int64_t nrows = ggml_nelements(rms->src[0]) / ncols;
    if (ncols <= 0 || nrows <= 0 ||
            ggml_nelements(gate) != ggml_nelements(rms->src[0]) ||
            ggml_nelements(rms) != ggml_nelements(rms->src[0]) ||
            ggml_nelements(norm_mul) != ggml_nelements(rms->src[0]) ||
            ggml_nelements(silu) != ggml_nelements(rms->src[0]) ||
            ggml_nelements(out) != ggml_nelements(rms->src[0]) ||
            ggml_nelements(final_output) != ggml_nelements(rms->src[0]) ||
            (norm_w != nullptr && ggml_nelements(norm_w) != ncols)) {
        if (blocker != nullptr) {
            *blocker = "L0 gated norm fusion tensor dimensions do not match";
        }
        return false;
    }
    if (rms->src[0]->nb[0] != (int64_t) sizeof(float) ||
            gate->nb[0] != (int64_t) sizeof(float) ||
            rms->nb[0] != (int64_t) sizeof(float) ||
            norm_mul->nb[0] != (int64_t) sizeof(float) ||
            silu->nb[0] != (int64_t) sizeof(float) ||
            out->nb[0] != (int64_t) sizeof(float) ||
            final_output->nb[0] != (int64_t) sizeof(float) ||
            (norm_w != nullptr && norm_w->nb[0] != (int64_t) sizeof(float)) ||
            !ggml_is_contiguous(rms->src[0]) ||
            !ggml_is_contiguous(gate) ||
            !ggml_is_contiguous(rms) ||
            !ggml_is_contiguous(norm_mul) ||
            !ggml_is_contiguous(silu) ||
            !ggml_is_contiguous(out) ||
            !ggml_is_contiguous(final_output)) {
        if (blocker != nullptr) {
            *blocker = "L0 gated norm fusion requires contiguous F32 layout";
        }
        return false;
    }
    const float eps = ggml_get_op_params_f32(rms, 0);
    if (eps < 0.0f) {
        if (blocker != nullptr) {
            *blocker = "L0 gated norm epsilon is negative";
        }
        return false;
    }
    if (!qwen36_superlayer_tensor_data_on_device(rms->src[0], device, "L0 gated norm input", blocker) ||
            !qwen36_superlayer_tensor_data_on_device(gate, device, "L0 gated norm gate", blocker) ||
            !qwen36_superlayer_tensor_data_on_device(rms, device, "L0 gated norm RMS output", blocker) ||
            !qwen36_superlayer_tensor_data_on_device(norm_mul, device, "L0 gated norm weighted output", blocker) ||
            !qwen36_superlayer_tensor_data_on_device(silu, device, "L0 gated norm silu output", blocker) ||
            !qwen36_superlayer_tensor_data_on_device(out, device, "L0 gated norm output", blocker) ||
            !qwen36_superlayer_tensor_data_on_device(final_output, device, "L0 final output", blocker)) {
        return false;
    }

    const bool use_direct_weights = qwen36_superlayer_direct_l0_norm_weights_requested();
    uint64_t norm_w_offset = 0;
    if (norm_w != nullptr) {
        if (!qwen36_superlayer_find_pack_offset_named(pack, norm_w, "L0 gated norm weight", &norm_w_offset, blocker)) {
            return false;
        }
        if (use_direct_weights &&
                !qwen36_superlayer_tensor_data_on_device(norm_w, device, "L0 gated norm weight", blocker)) {
            return false;
        }
    }

    desc->x = (const float *) rms->src[0]->data;
    desc->gate = (const float *) gate->data;
    desc->norm_w_data = norm_w != nullptr ? (const float *) norm_w->data : nullptr;
    desc->rms_dst = (float *) rms->data;
    desc->norm_dst = (float *) norm_mul->data;
    desc->silu_dst = (float *) silu->data;
    desc->out_dst = (float *) out->data;
    desc->final_dst = final_output->data != out->data ? (float *) final_output->data : nullptr;
    desc->norm_w_offset = norm_w_offset;
    desc->ncols = (uint32_t) ncols;
    desc->nrows = (uint32_t) nrows;
    desc->eps = eps;
    desc->use_direct_weights = use_direct_weights ? 1u : 0u;
    desc->ready = 1u;
    return true;
}

static bool qwen36_superlayer_l0_attn_out_requested() {
    return qwen36_superlayer_replace_l0_all_stages_requested() ||
        qwen36_superlayer_env_i64("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REPLACE_L0_ATTN_OUT", 0) != 0;
}

static bool qwen36_superlayer_make_l0_attn_out_desc(
        const ggml_cgraph * cgraph,
        const qwen36_superlayer_plan & plan,
        const qwen36_superlayer_pack_plan & pack,
        const int device,
        qwen36_superlayer_l0_attn_out_desc * desc,
        std::string * blocker) {
    if (desc == nullptr) {
        if (blocker != nullptr) {
            *blocker = "missing L0 attention output descriptor output";
        }
        return false;
    }
    *desc = qwen36_superlayer_l0_attn_out_desc{};

    const bool requested = qwen36_superlayer_l0_attn_out_requested();
    if (!requested) {
        return true;
    }

    const int begin = plan.layer_start[0] >= 0 ? plan.layer_start[0] : 0;
    const int end   = plan.layer_end[0]   >= begin ? plan.layer_end[0]   : cgraph->n_nodes - 1;
    const ggml_tensor * final_output = nullptr;
    const ggml_tensor * linear_attn_out = nullptr;
    for (int i = begin; i <= end; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (final_output == nullptr && tensor_name_matches_layer(node, "final_output", 0)) {
            final_output = node;
        } else if (linear_attn_out == nullptr && tensor_name_matches_layer(node, "linear_attn_out", 0)) {
            linear_attn_out = node;
        }
    }

    const ggml_tensor * out = qwen36_superlayer_strip_view_ops(linear_attn_out);
    const ggml_tensor * scale = nullptr;
    const ggml_tensor * mm = qwen36_superlayer_resolve_mul_mat_with_scale(linear_attn_out, &scale);
    if (final_output == nullptr || linear_attn_out == nullptr || out == nullptr ||
            mm == nullptr || mm->src[0] == nullptr || mm->src[1] == nullptr) {
        if (blocker != nullptr) {
            *blocker = "L0 attention output fusion requires final_output-0 and linear_attn_out-0 MUL_MAT";
        }
        return false;
    }
    if (!qwen36_superlayer_same_tensor_or_view(mm->src[1], final_output)) {
        if (blocker != nullptr) {
            *blocker = "L0 attention output MUL_MAT input is not final_output-0";
        }
        return false;
    }

    const ggml_tensor * w = mm->src[0];
    if (!qwen36_superlayer_projection_weight_supported(w)) {
        if (blocker != nullptr) {
            *blocker = "L0 attention output weight type is not implemented in the superlayer";
        }
        return false;
    }
    if (final_output->type != GGML_TYPE_F32 || mm->src[1]->type != GGML_TYPE_F32 ||
            mm->type != GGML_TYPE_F32 || out->type != GGML_TYPE_F32 ||
            linear_attn_out->type != GGML_TYPE_F32 ||
            (scale != nullptr && scale->type != GGML_TYPE_F32)) {
        if (blocker != nullptr) {
            *blocker = "L0 attention output fusion currently requires F32 activation/output tensors";
        }
        return false;
    }

    const int64_t n_embd = w->ne[0];
    const int64_t n_out = mm->ne[0];
    if (n_embd <= 0 || n_out <= 0 ||
            w->ne[1] != n_out ||
            mm->src[1]->ne[0] != n_embd ||
            ggml_nelements(mm->src[1]) != n_embd ||
            ggml_nelements(final_output) != n_embd ||
            ggml_nelements(mm) != n_out ||
            ggml_nelements(out) != n_out ||
            ggml_nelements(linear_attn_out) != n_out ||
            (scale != nullptr && ggml_nelements(scale) != 1 && ggml_nelements(scale) != n_out)) {
        if (blocker != nullptr) {
            *blocker = "L0 attention output tensor dimensions do not match one-token decode";
        }
        return false;
    }
    if (mm->src[1]->nb[0] != (int64_t) sizeof(float) ||
            final_output->nb[0] != (int64_t) sizeof(float) ||
            mm->nb[0] != (int64_t) sizeof(float) ||
            out->nb[0] != (int64_t) sizeof(float) ||
            linear_attn_out->nb[0] != (int64_t) sizeof(float) ||
            (scale != nullptr && scale->nb[0] != (int64_t) sizeof(float)) ||
            !ggml_is_contiguous(mm->src[1]) ||
            !ggml_is_contiguous(final_output) ||
            !ggml_is_contiguous_1(mm) ||
            !ggml_is_contiguous_1(out) ||
            !ggml_is_contiguous_1(linear_attn_out) ||
            (scale != nullptr && !ggml_is_contiguous_1(scale))) {
        if (blocker != nullptr) {
            *blocker = "L0 attention output fusion requires dense F32 one-token layout";
        }
        return false;
    }
    if (!qwen36_superlayer_tensor_data_on_device(mm->src[1], device, "L0 attention output input", blocker) ||
            (mm->src[1] != final_output &&
             !qwen36_superlayer_tensor_data_on_device(final_output, device, "L0 final output", blocker)) ||
            !qwen36_superlayer_tensor_data_on_device(out, device, "L0 attention output", blocker) ||
            (mm != out && mm != linear_attn_out &&
             !qwen36_superlayer_tensor_data_on_device(mm, device, "L0 attention output math output", blocker)) ||
            (linear_attn_out != out &&
             !qwen36_superlayer_tensor_data_on_device(linear_attn_out, device, "L0 attention named output", blocker))) {
        return false;
    }

    const bool use_direct_weights = qwen36_superlayer_direct_l0_out_weights_requested();
    uint64_t w_offset = 0;
    uint64_t scale_offset = 0;
    if (!qwen36_superlayer_find_pack_offset_named(pack, w, "L0 attention output weight", &w_offset, blocker)) {
        return false;
    }
    if (scale != nullptr &&
            !qwen36_superlayer_find_pack_offset_named(pack, scale, "L0 attention output post scale", &scale_offset, blocker)) {
        return false;
    }
    if (use_direct_weights) {
        if (!qwen36_superlayer_tensor_data_on_device(w, device, "L0 attention output weight", blocker) ||
                (scale != nullptr &&
                 !qwen36_superlayer_tensor_data_on_device(scale, device, "L0 attention output post scale", blocker))) {
            return false;
        }
    }

    desc->x = (const float *) mm->src[1]->data;
    desc->math_dst = mm != out && mm != linear_attn_out &&
        mm->data != out->data && mm->data != linear_attn_out->data ? (float *) mm->data : nullptr;
    desc->out_dst = (float *) out->data;
    desc->named_dst = linear_attn_out != out && linear_attn_out->data != out->data ?
        (float *) linear_attn_out->data : nullptr;
    desc->w_data = (const char *) w->data;
    desc->scale_data = scale != nullptr ? (const float *) scale->data : nullptr;
    desc->w_offset = w_offset;
    desc->scale_offset = scale_offset;
    desc->w_nb1 = (uint64_t) w->nb[1];
    desc->n_embd = (uint32_t) n_embd;
    desc->n_out = (uint32_t) n_out;
    desc->w_type = (int32_t) w->type;
    desc->scale_n = scale != nullptr ? (uint32_t) ggml_nelements(scale) : 0u;
    desc->has_scale = scale != nullptr ? 1u : 0u;
    desc->use_direct_weights = use_direct_weights ? 1u : 0u;
    desc->ready = 1u;
    return true;
}

static bool qwen36_superlayer_l0_post_attn_requested() {
    return qwen36_superlayer_replace_l0_all_stages_requested() ||
        qwen36_superlayer_env_i64("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REPLACE_L0_POST_ATTN", 0) != 0;
}

static bool qwen36_superlayer_make_l0_post_attn_desc(
        const ggml_cgraph * cgraph,
        const qwen36_superlayer_plan & plan,
        const qwen36_superlayer_pack_plan & pack,
        const int device,
        qwen36_superlayer_l0_post_attn_desc * desc,
        std::string * blocker) {
    if (desc == nullptr) {
        if (blocker != nullptr) {
            *blocker = "missing L0 post-attention descriptor output";
        }
        return false;
    }
    *desc = qwen36_superlayer_l0_post_attn_desc{};

    const bool requested = qwen36_superlayer_l0_post_attn_requested();
    if (!requested) {
        return true;
    }

    const int begin = plan.layer_start[0] >= 0 ? plan.layer_start[0] : 0;
    const int end   = plan.layer_end[0]   >= begin ? plan.layer_end[0]   : cgraph->n_nodes - 1;
    const ggml_tensor * linear_attn_out = nullptr;
    const ggml_tensor * attn_residual = nullptr;
    const ggml_tensor * attn_post_norm = nullptr;
    for (int i = begin; i <= end; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (linear_attn_out == nullptr && tensor_name_matches_layer(node, "linear_attn_out", 0)) {
            linear_attn_out = node;
        } else if (attn_residual == nullptr && tensor_name_matches_layer(node, "attn_residual", 0)) {
            attn_residual = node;
        } else if (attn_post_norm == nullptr && tensor_name_matches_layer(node, "attn_post_norm", 0)) {
            attn_post_norm = node;
        }
    }

    const ggml_tensor * residual = qwen36_superlayer_strip_view_ops(attn_residual);
    const ggml_tensor * post_out = qwen36_superlayer_strip_view_ops(attn_post_norm);
    if (linear_attn_out == nullptr || attn_residual == nullptr || attn_post_norm == nullptr ||
            residual == nullptr || residual->op != GGML_OP_ADD ||
            post_out == nullptr) {
        if (blocker != nullptr) {
            *blocker = "L0 post-attention fusion requires linear_attn_out-0, attn_residual-0, and attn_post_norm-0";
        }
        return false;
    }

    const ggml_tensor * attn = nullptr;
    const ggml_tensor * skip = nullptr;
    if (qwen36_superlayer_same_tensor_or_view(residual->src[0], linear_attn_out)) {
        attn = residual->src[0];
        skip = residual->src[1];
    } else if (qwen36_superlayer_same_tensor_or_view(residual->src[1], linear_attn_out)) {
        attn = residual->src[1];
        skip = residual->src[0];
    }
    if (attn == nullptr || skip == nullptr) {
        if (blocker != nullptr) {
            *blocker = "L0 post-attention residual does not consume linear_attn_out-0";
        }
        return false;
    }

    const ggml_tensor * rms = nullptr;
    const ggml_tensor * norm_w = nullptr;
    if (post_out->op == GGML_OP_MUL) {
        if (post_out->src[0] != nullptr && post_out->src[0]->op == GGML_OP_RMS_NORM) {
            rms = post_out->src[0];
            norm_w = post_out->src[1];
        } else if (post_out->src[1] != nullptr && post_out->src[1]->op == GGML_OP_RMS_NORM) {
            rms = post_out->src[1];
            norm_w = post_out->src[0];
        }
    } else if (post_out->op == GGML_OP_RMS_NORM) {
        rms = post_out;
    }
    if (rms == nullptr || rms->src[0] == nullptr ||
            !qwen36_superlayer_same_tensor_or_view(rms->src[0], attn_residual)) {
        if (blocker != nullptr) {
            *blocker = "L0 post-attention fusion could not resolve RMS_NORM(attn_residual-0)";
        }
        return false;
    }

    if (attn->type != GGML_TYPE_F32 || skip->type != GGML_TYPE_F32 ||
            residual->type != GGML_TYPE_F32 || attn_residual->type != GGML_TYPE_F32 ||
            rms->type != GGML_TYPE_F32 || post_out->type != GGML_TYPE_F32 ||
            attn_post_norm->type != GGML_TYPE_F32 ||
            (norm_w != nullptr && norm_w->type != GGML_TYPE_F32)) {
        if (blocker != nullptr) {
            *blocker = "L0 post-attention fusion currently requires F32 tensors";
        }
        return false;
    }

    const int64_t ncols = residual->ne[0];
    if (ncols <= 0) {
        if (blocker != nullptr) {
            *blocker = "L0 post-attention tensor hidden dimension is empty";
        }
        return false;
    }
    const int64_t nrows = ggml_nelements(residual) / ncols;
    if (nrows <= 0 ||
            ggml_nelements(attn) != ggml_nelements(residual) ||
            ggml_nelements(skip) != ggml_nelements(residual) ||
            ggml_nelements(attn_residual) != ggml_nelements(residual) ||
            ggml_nelements(rms) != ggml_nelements(residual) ||
            ggml_nelements(post_out) != ggml_nelements(residual) ||
            ggml_nelements(attn_post_norm) != ggml_nelements(residual) ||
            (norm_w != nullptr && ggml_nelements(norm_w) != ncols)) {
        if (blocker != nullptr) {
            *blocker = "L0 post-attention tensor dimensions do not match";
        }
        return false;
    }
    if (attn->nb[0] != (int64_t) sizeof(float) ||
            skip->nb[0] != (int64_t) sizeof(float) ||
            residual->nb[0] != (int64_t) sizeof(float) ||
            attn_residual->nb[0] != (int64_t) sizeof(float) ||
            rms->nb[0] != (int64_t) sizeof(float) ||
            post_out->nb[0] != (int64_t) sizeof(float) ||
            attn_post_norm->nb[0] != (int64_t) sizeof(float) ||
            (norm_w != nullptr && norm_w->nb[0] != (int64_t) sizeof(float)) ||
            !ggml_is_contiguous(attn) ||
            !ggml_is_contiguous(skip) ||
            !ggml_is_contiguous(residual) ||
            !ggml_is_contiguous(attn_residual) ||
            !ggml_is_contiguous(rms) ||
            !ggml_is_contiguous(post_out) ||
            !ggml_is_contiguous(attn_post_norm)) {
        if (blocker != nullptr) {
            *blocker = "L0 post-attention fusion requires contiguous F32 layout";
        }
        return false;
    }

    const float eps = ggml_get_op_params_f32(rms, 0);
    if (eps < 0.0f) {
        if (blocker != nullptr) {
            *blocker = "L0 post-attention epsilon is negative";
        }
        return false;
    }
    if (!qwen36_superlayer_tensor_data_on_device(attn, device, "L0 post-attention attention output", blocker) ||
            !qwen36_superlayer_tensor_data_on_device(skip, device, "L0 post-attention residual input", blocker) ||
            !qwen36_superlayer_tensor_data_on_device(residual, device, "L0 post-attention residual output", blocker) ||
            (attn_residual != residual &&
             !qwen36_superlayer_tensor_data_on_device(attn_residual, device, "L0 attn_residual named output", blocker)) ||
            !qwen36_superlayer_tensor_data_on_device(rms, device, "L0 attn_post_norm RMS output", blocker) ||
            !qwen36_superlayer_tensor_data_on_device(post_out, device, "L0 attn_post_norm output", blocker) ||
            (attn_post_norm != post_out &&
             !qwen36_superlayer_tensor_data_on_device(attn_post_norm, device, "L0 attn_post_norm named output", blocker))) {
        return false;
    }

    const bool use_direct_weights = qwen36_superlayer_direct_l0_norm_weights_requested();
    uint64_t norm_w_offset = 0;
    if (norm_w != nullptr) {
        if (!qwen36_superlayer_find_pack_offset_named(
                    pack, norm_w, "L0 post-attention norm weight", &norm_w_offset, blocker)) {
            return false;
        }
        if (use_direct_weights &&
                !qwen36_superlayer_tensor_data_on_device(
                    norm_w, device, "L0 post-attention norm weight", blocker)) {
            return false;
        }
    }

    desc->attn = (const float *) attn->data;
    desc->skip = (const float *) skip->data;
    desc->norm_w_data = norm_w != nullptr ? (const float *) norm_w->data : nullptr;
    desc->residual_dst = (float *) residual->data;
    desc->residual_named_dst = attn_residual != residual && attn_residual->data != residual->data ?
        (float *) attn_residual->data : nullptr;
    desc->rms_dst = (float *) rms->data;
    desc->norm_dst = (float *) post_out->data;
    desc->named_dst = attn_post_norm != post_out && attn_post_norm->data != post_out->data ?
        (float *) attn_post_norm->data : nullptr;
    desc->norm_w_offset = norm_w_offset;
    desc->ncols = (uint32_t) ncols;
    desc->nrows = (uint32_t) nrows;
    desc->eps = eps;
    desc->has_norm_w = norm_w != nullptr ? 1u : 0u;
    desc->use_direct_weights = use_direct_weights ? 1u : 0u;
    desc->ready = 1u;
    return true;
}

static bool qwen36_superlayer_l0_moe_router_requested() {
    return qwen36_superlayer_replace_l0_all_stages_requested() ||
        qwen36_superlayer_env_i64("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REPLACE_L0_MOE_ROUTER", 0) != 0;
}

static bool qwen36_superlayer_make_l0_moe_router_desc(
        const ggml_cgraph * cgraph,
        const qwen36_superlayer_plan & plan,
        const qwen36_superlayer_pack_plan & pack,
        const int device,
        qwen36_superlayer_l0_moe_router_desc * desc,
        std::string * blocker) {
    if (desc == nullptr) {
        if (blocker != nullptr) {
            *blocker = "missing L0 MoE router descriptor output";
        }
        return false;
    }
    *desc = qwen36_superlayer_l0_moe_router_desc{};

    const bool requested = qwen36_superlayer_l0_moe_router_requested();
    if (!requested) {
        return true;
    }

    const int begin = plan.layer_start[0] >= 0 ? plan.layer_start[0] : 0;
    const int end   = plan.layer_end[0]   >= begin ? plan.layer_end[0]   : cgraph->n_nodes - 1;
    const ggml_tensor * attn_post_norm = nullptr;
    const ggml_tensor * logits = nullptr;
    const ggml_tensor * probs = nullptr;
    const ggml_tensor * argsort = nullptr;
    const ggml_tensor * topk = nullptr;
    const ggml_tensor * weights = nullptr;
    const ggml_tensor * weights_sum = nullptr;
    const ggml_tensor * weights_sum_clamped = nullptr;
    const ggml_tensor * weights_norm = nullptr;
    const ggml_tensor * weights_scaled = nullptr;
    const ggml_tensor * unsupported = nullptr;
    for (int i = begin; i <= end; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (attn_post_norm == nullptr && tensor_name_matches_layer(node, "attn_post_norm", 0)) {
            attn_post_norm = node;
        } else if (logits == nullptr && tensor_name_matches_layer(node, "ffn_moe_logits", 0)) {
            logits = node;
        } else if (probs == nullptr && tensor_name_matches_layer(node, "ffn_moe_probs", 0)) {
            probs = node;
        } else if (argsort == nullptr && tensor_name_matches_layer(node, "ffn_moe_argsort", 0)) {
            argsort = node;
        } else if (topk == nullptr && tensor_name_matches_layer(node, "ffn_moe_topk", 0)) {
            topk = node;
        } else if (weights == nullptr && tensor_name_matches_layer(node, "ffn_moe_weights", 0)) {
            weights = node;
        } else if (weights_sum == nullptr && tensor_name_matches_layer(node, "ffn_moe_weights_sum", 0)) {
            weights_sum = node;
        } else if (weights_sum_clamped == nullptr && tensor_name_matches_layer(node, "ffn_moe_weights_sum_clamped", 0)) {
            weights_sum_clamped = node;
        } else if (weights_norm == nullptr && tensor_name_matches_layer(node, "ffn_moe_weights_norm", 0)) {
            weights_norm = node;
        } else if (weights_scaled == nullptr && tensor_name_matches_layer(node, "ffn_moe_weights_scaled", 0)) {
            weights_scaled = node;
        } else if (tensor_name_matches_layer(node, "ffn_moe_logits_biased", 0) ||
                tensor_name_matches_layer(node, "ffn_moe_probs_biased", 0) ||
                tensor_name_matches_layer(node, "ffn_moe_probs_masked", 0) ||
                tensor_name_matches_layer(node, "ffn_moe_group_topk", 0) ||
                tensor_name_matches_layer(node, "ffn_moe_weights_softmax", 0)) {
            unsupported = node;
        }
    }

    if (unsupported != nullptr) {
        if (blocker != nullptr) {
            *blocker = "L0 MoE router fusion does not support biased/grouped/delayed router node: ";
            *blocker += unsupported->name;
        }
        return false;
    }
    if (attn_post_norm == nullptr || logits == nullptr || probs == nullptr ||
            argsort == nullptr || topk == nullptr || weights == nullptr) {
        if (blocker != nullptr) {
            *blocker = "L0 MoE router fusion requires attn_post_norm-0 and ffn_moe logits/probs/topk/weights";
        }
        return false;
    }

    const ggml_tensor * scale = nullptr;
    const ggml_tensor * mm = qwen36_superlayer_resolve_mul_mat_with_scale(logits, &scale);
    const ggml_tensor * logits_out = qwen36_superlayer_strip_view_ops(logits);
    if (mm == nullptr || mm->src[0] == nullptr || mm->src[1] == nullptr || logits_out == nullptr) {
        if (blocker != nullptr) {
            *blocker = "L0 MoE router fusion could not resolve ffn_moe_logits-0 to MUL_MAT";
        }
        return false;
    }
    if (!qwen36_superlayer_same_tensor_or_view(mm->src[1], attn_post_norm)) {
        if (blocker != nullptr) {
            *blocker = "L0 MoE router logits input is not attn_post_norm-0";
        }
        return false;
    }
    if (probs->op != GGML_OP_SOFT_MAX || probs->src[0] == nullptr ||
            !qwen36_superlayer_same_tensor_or_view(probs->src[0], logits)) {
        if (blocker != nullptr) {
            *blocker = "L0 MoE router fusion currently requires SOFT_MAX(ffn_moe_logits-0)";
        }
        return false;
    }
    if (argsort->op != GGML_OP_ARGSORT || argsort->src[0] == nullptr ||
            !qwen36_superlayer_same_tensor_or_view(argsort->src[0], probs)) {
        if (blocker != nullptr) {
            *blocker = "L0 MoE router fusion requires ARGSORT(ffn_moe_probs-0)";
        }
        return false;
    }
    if (topk->op != GGML_OP_VIEW || topk->src[0] == nullptr ||
            !qwen36_superlayer_same_tensor_or_view(topk->src[0], argsort)) {
        if (blocker != nullptr) {
            *blocker = "L0 MoE router fusion requires ffn_moe_topk-0 VIEW of ffn_moe_argsort-0";
        }
        return false;
    }
    if (weights->op != GGML_OP_GET_ROWS || weights->src[0] == nullptr || weights->src[1] == nullptr ||
            !qwen36_superlayer_same_tensor_or_view(weights->src[0], probs) ||
            !qwen36_superlayer_same_tensor_or_view(weights->src[1], topk)) {
        if (blocker != nullptr) {
            *blocker = "L0 MoE router fusion requires GET_ROWS(reshape(ffn_moe_probs-0), ffn_moe_topk-0)";
        }
        return false;
    }

    const bool has_norm =
        weights_sum != nullptr || weights_sum_clamped != nullptr || weights_norm != nullptr;
    if (has_norm &&
            (weights_sum == nullptr || weights_sum_clamped == nullptr || weights_norm == nullptr)) {
        if (blocker != nullptr) {
            *blocker = "L0 MoE router fusion requires the complete weights sum/clamp/div normalization chain";
        }
        return false;
    }
    if (has_norm) {
        if (weights_sum->op != GGML_OP_SUM_ROWS || weights_sum->src[0] == nullptr ||
                !qwen36_superlayer_same_tensor_or_view(weights_sum->src[0], weights) ||
                weights_sum_clamped->op != GGML_OP_CLAMP || weights_sum_clamped->src[0] != weights_sum ||
                weights_norm->op != GGML_OP_DIV || weights_norm->src[0] == nullptr ||
                !qwen36_superlayer_same_tensor_or_view(weights_norm->src[0], weights) ||
                weights_norm->src[1] != weights_sum_clamped) {
            if (blocker != nullptr) {
                *blocker = "L0 MoE router fusion weights normalization chain has unexpected topology";
            }
            return false;
        }
    }
    if (weights_scaled != nullptr) {
        const ggml_tensor * expected = has_norm ? weights_norm : weights;
        if (weights_scaled->op != GGML_OP_SCALE || weights_scaled->src[0] == nullptr ||
                !qwen36_superlayer_same_tensor_or_view(weights_scaled->src[0], expected) ||
                ggml_get_op_params_f32(weights_scaled, 1) != 0.0f) {
            if (blocker != nullptr) {
                *blocker = "L0 MoE router fusion requires SCALE(weights) with zero bias";
            }
            return false;
        }
    }

    const ggml_tensor * w = mm->src[0];
    if (!qwen36_superlayer_projection_weight_supported(w)) {
        if (blocker != nullptr) {
            *blocker = "L0 MoE router weight type is not implemented in the superlayer";
        }
        return false;
    }
    if (mm->src[1]->type != GGML_TYPE_F32 || mm->type != GGML_TYPE_F32 ||
            logits_out->type != GGML_TYPE_F32 || logits->type != GGML_TYPE_F32 ||
            probs->type != GGML_TYPE_F32 || weights->type != GGML_TYPE_F32 ||
            argsort->type != GGML_TYPE_I32 || topk->type != GGML_TYPE_I32 ||
            (scale != nullptr && scale->type != GGML_TYPE_F32) ||
            (weights_sum != nullptr && weights_sum->type != GGML_TYPE_F32) ||
            (weights_sum_clamped != nullptr && weights_sum_clamped->type != GGML_TYPE_F32) ||
            (weights_norm != nullptr && weights_norm->type != GGML_TYPE_F32) ||
            (weights_scaled != nullptr && weights_scaled->type != GGML_TYPE_F32)) {
        if (blocker != nullptr) {
            *blocker = "L0 MoE router fusion currently requires F32 logits/probs/weights and I32 ids";
        }
        return false;
    }

    const int64_t n_embd = w->ne[0];
    const int64_t n_expert = mm->ne[0];
    const int64_t n_expert_used = ggml_nelements(topk);
    if (n_embd <= 0 || n_expert <= 0 || n_expert_used <= 0 || n_expert_used > n_expert ||
            n_expert_used > 64 ||
            w->ne[1] != n_expert ||
            mm->src[1]->ne[0] != n_embd ||
            ggml_nelements(mm->src[1]) != n_embd ||
            ggml_nelements(attn_post_norm) != n_embd ||
            ggml_nelements(mm) != n_expert ||
            ggml_nelements(logits_out) != n_expert ||
            ggml_nelements(logits) != n_expert ||
            ggml_nelements(probs) != n_expert ||
            ggml_nelements(argsort) < n_expert_used ||
            ggml_nelements(weights) != n_expert_used ||
            (scale != nullptr && ggml_nelements(scale) != 1 && ggml_nelements(scale) != n_expert) ||
            (weights_sum != nullptr && ggml_nelements(weights_sum) != 1) ||
            (weights_sum_clamped != nullptr && ggml_nelements(weights_sum_clamped) != 1) ||
            (weights_norm != nullptr && ggml_nelements(weights_norm) != n_expert_used) ||
            (weights_scaled != nullptr && ggml_nelements(weights_scaled) != n_expert_used)) {
        if (blocker != nullptr) {
            *blocker = "L0 MoE router tensor dimensions do not match one-token decode";
        }
        return false;
    }
    if (mm->src[1]->nb[0] != (int64_t) sizeof(float) ||
            attn_post_norm->nb[0] != (int64_t) sizeof(float) ||
            mm->nb[0] != (int64_t) sizeof(float) ||
            logits_out->nb[0] != (int64_t) sizeof(float) ||
            logits->nb[0] != (int64_t) sizeof(float) ||
            probs->nb[0] != (int64_t) sizeof(float) ||
            argsort->nb[0] != (int64_t) sizeof(int32_t) ||
            topk->nb[0] != (int64_t) sizeof(int32_t) ||
            weights->nb[0] != (int64_t) sizeof(float) ||
            (scale != nullptr && scale->nb[0] != (int64_t) sizeof(float)) ||
            (weights_sum != nullptr && weights_sum->nb[0] != (int64_t) sizeof(float)) ||
            (weights_sum_clamped != nullptr && weights_sum_clamped->nb[0] != (int64_t) sizeof(float)) ||
            (weights_norm != nullptr && weights_norm->nb[0] != (int64_t) sizeof(float)) ||
            (weights_scaled != nullptr && weights_scaled->nb[0] != (int64_t) sizeof(float)) ||
            !ggml_is_contiguous(mm->src[1]) ||
            !ggml_is_contiguous(attn_post_norm) ||
            !ggml_is_contiguous_1(mm) ||
            !ggml_is_contiguous_1(logits_out) ||
            !ggml_is_contiguous_1(logits) ||
            !ggml_is_contiguous_1(probs) ||
            !ggml_is_contiguous_1(argsort) ||
            !ggml_is_contiguous_1(topk) ||
            !ggml_is_contiguous_1(weights) ||
            (scale != nullptr && !ggml_is_contiguous_1(scale)) ||
            (weights_sum != nullptr && !ggml_is_contiguous_1(weights_sum)) ||
            (weights_sum_clamped != nullptr && !ggml_is_contiguous_1(weights_sum_clamped)) ||
            (weights_norm != nullptr && !ggml_is_contiguous_1(weights_norm)) ||
            (weights_scaled != nullptr && !ggml_is_contiguous_1(weights_scaled))) {
        if (blocker != nullptr) {
            *blocker = "L0 MoE router fusion requires dense one-token F32/I32 layout";
        }
        return false;
    }
    if (!qwen36_superlayer_tensor_data_on_device(mm->src[1], device, "L0 MoE router input", blocker) ||
            (mm->src[1] != attn_post_norm &&
             !qwen36_superlayer_tensor_data_on_device(attn_post_norm, device, "L0 MoE router attn_post_norm", blocker)) ||
            !qwen36_superlayer_tensor_data_on_device(logits_out, device, "L0 MoE logits output", blocker) ||
            (mm != logits_out && mm != logits &&
             !qwen36_superlayer_tensor_data_on_device(mm, device, "L0 MoE logits math output", blocker)) ||
            (logits != logits_out &&
             !qwen36_superlayer_tensor_data_on_device(logits, device, "L0 MoE logits named output", blocker)) ||
            !qwen36_superlayer_tensor_data_on_device(probs, device, "L0 MoE probs output", blocker) ||
            !qwen36_superlayer_tensor_data_on_device(argsort, device, "L0 MoE argsort output", blocker) ||
            !qwen36_superlayer_tensor_data_on_device(topk, device, "L0 MoE topk output", blocker) ||
            !qwen36_superlayer_tensor_data_on_device(weights, device, "L0 MoE weights output", blocker) ||
            (weights_sum != nullptr &&
             !qwen36_superlayer_tensor_data_on_device(weights_sum, device, "L0 MoE weights sum output", blocker)) ||
            (weights_sum_clamped != nullptr &&
             !qwen36_superlayer_tensor_data_on_device(weights_sum_clamped, device, "L0 MoE weights sum clamped output", blocker)) ||
            (weights_norm != nullptr &&
             !qwen36_superlayer_tensor_data_on_device(weights_norm, device, "L0 MoE weights norm output", blocker)) ||
            (weights_scaled != nullptr &&
             !qwen36_superlayer_tensor_data_on_device(weights_scaled, device, "L0 MoE weights scaled output", blocker))) {
        return false;
    }

    const bool use_direct_weights = qwen36_superlayer_direct_l0_moe_weights_requested();
    uint64_t w_offset = 0;
    uint64_t scale_offset = 0;
    if (!qwen36_superlayer_find_pack_offset_named(pack, w, "L0 MoE router weight", &w_offset, blocker)) {
        return false;
    }
    if (scale != nullptr &&
            !qwen36_superlayer_find_pack_offset_named(pack, scale, "L0 MoE router post scale", &scale_offset, blocker)) {
        return false;
    }
    if (use_direct_weights) {
        if (!qwen36_superlayer_tensor_data_on_device(w, device, "L0 MoE router weight", blocker) ||
                (scale != nullptr &&
                 !qwen36_superlayer_tensor_data_on_device(scale, device, "L0 MoE router post scale", blocker))) {
            return false;
        }
    }

    desc->x = (const float *) mm->src[1]->data;
    desc->logits_math_dst = mm != logits_out && mm != logits &&
        mm->data != logits_out->data && mm->data != logits->data ? (float *) mm->data : nullptr;
    desc->logits_dst = (float *) logits_out->data;
    desc->logits_named_dst = logits != logits_out && logits->data != logits_out->data ?
        (float *) logits->data : nullptr;
    desc->probs_dst = (float *) probs->data;
    desc->argsort_dst = (int32_t *) argsort->data;
    desc->topk_dst = (int32_t *) topk->data;
    desc->weights_dst = (float *) weights->data;
    desc->weights_sum_dst = weights_sum != nullptr ? (float *) weights_sum->data : nullptr;
    desc->weights_sum_clamped_dst = weights_sum_clamped != nullptr ? (float *) weights_sum_clamped->data : nullptr;
    desc->weights_norm_dst = weights_norm != nullptr ? (float *) weights_norm->data : nullptr;
    desc->weights_scaled_dst = weights_scaled != nullptr ? (float *) weights_scaled->data : nullptr;
    desc->w_data = (const char *) w->data;
    desc->scale_data = scale != nullptr ? (const float *) scale->data : nullptr;
    desc->w_offset = w_offset;
    desc->scale_offset = scale_offset;
    desc->w_nb1 = (uint64_t) w->nb[1];
    desc->n_embd = (uint32_t) n_embd;
    desc->n_expert = (uint32_t) n_expert;
    desc->n_expert_used = (uint32_t) n_expert_used;
    desc->w_type = (int32_t) w->type;
    desc->scale_n = scale != nullptr ? (uint32_t) ggml_nelements(scale) : 0u;
    desc->weights_scale = weights_scaled != nullptr ? ggml_get_op_params_f32(weights_scaled, 0) : 1.0f;
    desc->clamp_min = weights_sum_clamped != nullptr ? ggml_get_op_params_f32(weights_sum_clamped, 0) : 0.0f;
    desc->has_scale = scale != nullptr ? 1u : 0u;
    desc->has_weights_sum = weights_sum != nullptr ? 1u : 0u;
    desc->has_weights_norm = weights_norm != nullptr ? 1u : 0u;
    desc->has_weights_scaled = weights_scaled != nullptr ? 1u : 0u;
    desc->use_direct_weights = use_direct_weights ? 1u : 0u;
    desc->ready = 1u;
    return true;
}

static bool qwen36_superlayer_l0_moe_gate_up_requested() {
    return qwen36_superlayer_replace_l0_all_stages_requested() ||
        qwen36_superlayer_env_i64("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REPLACE_L0_MOE_GATE_UP", 0) != 0;
}

static bool qwen36_superlayer_make_l0_moe_gate_up_desc(
        const ggml_cgraph * cgraph,
        const qwen36_superlayer_plan & plan,
        const qwen36_superlayer_pack_plan & pack,
        const int device,
        qwen36_superlayer_l0_moe_gate_up_desc * desc,
        std::string * blocker) {
    if (desc == nullptr) {
        if (blocker != nullptr) {
            *blocker = "missing L0 MoE gate/up descriptor output";
        }
        return false;
    }
    *desc = qwen36_superlayer_l0_moe_gate_up_desc{};

    const bool requested = qwen36_superlayer_l0_moe_gate_up_requested();
    if (!requested) {
        return true;
    }

    const int begin = plan.layer_start[0] >= 0 ? plan.layer_start[0] : 0;
    const int end   = plan.layer_end[0]   >= begin ? plan.layer_end[0]   : cgraph->n_nodes - 1;
    const ggml_tensor * attn_post_norm = nullptr;
    const ggml_tensor * topk = nullptr;
    const ggml_tensor * gate_up = nullptr;
    const ggml_tensor * gate = nullptr;
    const ggml_tensor * up = nullptr;
    const ggml_tensor * swiglu = nullptr;
    const ggml_tensor * unsupported = nullptr;
    for (int i = begin; i <= end; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (attn_post_norm == nullptr && tensor_name_matches_layer(node, "attn_post_norm", 0)) {
            attn_post_norm = node;
        } else if (topk == nullptr && tensor_name_matches_layer(node, "ffn_moe_topk", 0)) {
            topk = node;
        } else if (gate_up == nullptr && tensor_name_matches_layer(node, "ffn_moe_gate_up", 0)) {
            gate_up = node;
        } else if (gate == nullptr && tensor_name_matches_layer(node, "ffn_moe_gate", 0)) {
            gate = node;
        } else if (up == nullptr && tensor_name_matches_layer(node, "ffn_moe_up", 0)) {
            up = node;
        } else if (swiglu == nullptr && tensor_name_matches_layer(node, "ffn_moe_swiglu", 0)) {
            swiglu = node;
        } else if (tensor_name_matches_layer(node, "ffn_moe_gate_up_biased", 0) ||
                tensor_name_matches_layer(node, "ffn_moe_gate_up_scaled", 0) ||
                tensor_name_matches_layer(node, "ffn_moe_gate_biased", 0) ||
                tensor_name_matches_layer(node, "ffn_moe_gate_scaled", 0) ||
                tensor_name_matches_layer(node, "ffn_moe_up_biased", 0) ||
                tensor_name_matches_layer(node, "ffn_moe_up_scaled", 0) ||
                tensor_name_matches_layer(node, "ffn_moe_silu", 0) ||
                tensor_name_matches_layer(node, "ffn_moe_geglu", 0) ||
                tensor_name_matches_layer(node, "ffn_moe_reglu", 0) ||
                tensor_name_matches_layer(node, "ffn_moe_swiglu_oai", 0) ||
                tensor_name_matches_layer(node, "ffn_moe_swiglu_limited", 0)) {
            unsupported = node;
        }
    }

    if (unsupported != nullptr) {
        if (blocker != nullptr) {
            *blocker = "L0 MoE gate/up fusion does not support biased/scaled/non-SwiGLU node: ";
            *blocker += unsupported->name;
        }
        return false;
    }
    if (attn_post_norm == nullptr || topk == nullptr || gate_up == nullptr ||
            gate == nullptr || up == nullptr || swiglu == nullptr) {
        if (blocker != nullptr) {
            *blocker = "L0 MoE gate/up fusion requires attn_post_norm/topk/gate_up/gate/up/swiglu nodes";
        }
        return false;
    }
    if (gate_up->op != GGML_OP_MUL_MAT_ID || gate_up->src[0] == nullptr ||
            gate_up->src[1] == nullptr || gate_up->src[2] == nullptr ||
            !qwen36_superlayer_same_tensor_or_view(gate_up->src[1], attn_post_norm) ||
            !qwen36_superlayer_same_tensor_or_view(gate_up->src[2], topk)) {
        if (blocker != nullptr) {
            *blocker = "L0 MoE gate/up fusion requires MUL_MAT_ID(weight, reshape(attn_post_norm-0), ffn_moe_topk-0)";
        }
        return false;
    }
    if (gate->op != GGML_OP_VIEW || up->op != GGML_OP_VIEW ||
            gate->src[0] != gate_up || up->src[0] != gate_up ||
            swiglu->op != GGML_OP_GLU ||
            ggml_get_glu_op(swiglu) != GGML_GLU_OP_SWIGLU ||
            ggml_get_op_params_i32(swiglu, 1) != 0 ||
            swiglu->src[0] != gate || swiglu->src[1] != up) {
        if (blocker != nullptr) {
            *blocker = "L0 MoE gate/up fusion requires gate/up views followed by SWIGLU";
        }
        return false;
    }

    const ggml_tensor * w = gate_up->src[0];
    if (!qwen36_superlayer_projection_weight_supported(w)) {
        if (blocker != nullptr) {
            *blocker = "L0 MoE gate/up weight type is not implemented in the superlayer";
        }
        return false;
    }
    if (gate_up->src[1]->type != GGML_TYPE_F32 || gate_up->src[2]->type != GGML_TYPE_I32 ||
            gate_up->type != GGML_TYPE_F32 || gate->type != GGML_TYPE_F32 ||
            up->type != GGML_TYPE_F32 || swiglu->type != GGML_TYPE_F32) {
        if (blocker != nullptr) {
            *blocker = "L0 MoE gate/up fusion currently requires F32 activations and I32 ids";
        }
        return false;
    }

    const int64_t n_embd = w->ne[0];
    const int64_t n_ff2 = gate_up->ne[0];
    const int64_t n_expert_used = gate_up->ne[1];
    const int64_t n_tokens = gate_up->ne[2];
    const int64_t n_expert = w->ne[2];
    if (n_embd <= 0 || n_ff2 <= 0 || (n_ff2 % 2) != 0 || n_expert <= 0 ||
            n_expert_used <= 0 || n_expert_used > 64 || n_tokens != 1 ||
            w->ne[1] != n_ff2 || gate_up->src[1]->ne[0] != n_embd ||
            ggml_nelements(gate_up->src[1]) != n_embd ||
            ggml_nelements(attn_post_norm) != n_embd ||
            ggml_nelements(topk) != n_expert_used ||
            gate->ne[0] != n_ff2/2 || up->ne[0] != n_ff2/2 ||
            gate->ne[1] != n_expert_used || up->ne[1] != n_expert_used ||
            gate->ne[2] != 1 || up->ne[2] != 1 ||
            swiglu->ne[0] != n_ff2/2 || swiglu->ne[1] != n_expert_used ||
            swiglu->ne[2] != 1) {
        if (blocker != nullptr) {
            *blocker = "L0 MoE gate/up tensor dimensions do not match one-token decode";
        }
        return false;
    }
    if (gate_up->src[1]->nb[0] != (int64_t) sizeof(float) ||
            gate_up->src[2]->nb[0] != (int64_t) sizeof(int32_t) ||
            attn_post_norm->nb[0] != (int64_t) sizeof(float) ||
            gate_up->nb[0] != (int64_t) sizeof(float) ||
            gate->nb[0] != (int64_t) sizeof(float) ||
            up->nb[0] != (int64_t) sizeof(float) ||
            swiglu->nb[0] != (int64_t) sizeof(float) ||
            !ggml_is_contiguous(gate_up->src[1]) ||
            !ggml_is_contiguous_1(gate_up->src[2]) ||
            !ggml_is_contiguous(attn_post_norm) ||
            !ggml_is_contiguous_1(gate_up) ||
            !ggml_is_contiguous_1(gate) ||
            !ggml_is_contiguous_1(up) ||
            !ggml_is_contiguous_1(swiglu)) {
        if (blocker != nullptr) {
            *blocker = "L0 MoE gate/up fusion requires dense one-token F32/I32 layout";
        }
        return false;
    }
    if (!qwen36_superlayer_tensor_data_on_device(gate_up->src[1], device, "L0 MoE gate/up input", blocker) ||
            (gate_up->src[1] != attn_post_norm &&
             !qwen36_superlayer_tensor_data_on_device(attn_post_norm, device, "L0 MoE gate/up attn_post_norm", blocker)) ||
            !qwen36_superlayer_tensor_data_on_device(gate_up->src[2], device, "L0 MoE gate/up ids", blocker) ||
            !qwen36_superlayer_tensor_data_on_device(gate_up, device, "L0 MoE gate_up output", blocker) ||
            !qwen36_superlayer_tensor_data_on_device(gate, device, "L0 MoE gate view", blocker) ||
            !qwen36_superlayer_tensor_data_on_device(up, device, "L0 MoE up view", blocker) ||
            !qwen36_superlayer_tensor_data_on_device(swiglu, device, "L0 MoE swiglu output", blocker)) {
        return false;
    }

    const bool use_direct_weights = qwen36_superlayer_direct_l0_moe_weights_requested();
    uint64_t w_offset = 0;
    if (!qwen36_superlayer_find_pack_offset_named(pack, w, "L0 MoE gate/up weight", &w_offset, blocker)) {
        return false;
    }
    if (use_direct_weights &&
            !qwen36_superlayer_tensor_data_on_device(w, device, "L0 MoE gate/up weight", blocker)) {
        return false;
    }

    desc->x = (const float *) gate_up->src[1]->data;
    desc->ids = (const int32_t *) gate_up->src[2]->data;
    desc->gate_up_dst = (float *) gate_up->data;
    desc->gate_dst = (float *) gate->data;
    desc->up_dst = (float *) up->data;
    desc->swiglu_dst = (float *) swiglu->data;
    desc->w_data = (const char *) w->data;
    desc->w_offset = w_offset;
    desc->w_nb1 = (uint64_t) w->nb[1];
    desc->w_nb2 = (uint64_t) w->nb[2];
    desc->gate_up_nb1 = (uint64_t) gate_up->nb[1];
    desc->gate_nb1 = (uint64_t) gate->nb[1];
    desc->up_nb1 = (uint64_t) up->nb[1];
    desc->swiglu_nb1 = (uint64_t) swiglu->nb[1];
    desc->n_embd = (uint32_t) n_embd;
    desc->n_ff = (uint32_t) (n_ff2/2);
    desc->n_expert = (uint32_t) n_expert;
    desc->n_expert_used = (uint32_t) n_expert_used;
    desc->w_type = (int32_t) w->type;
    desc->use_direct_weights = use_direct_weights ? 1u : 0u;
    desc->ready = 1u;
    return true;
}

static bool qwen36_superlayer_l0_moe_down_requested() {
    return qwen36_superlayer_replace_l0_all_stages_requested() ||
        qwen36_superlayer_env_i64("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REPLACE_L0_MOE_DOWN", 0) != 0;
}

static bool qwen36_superlayer_make_l0_moe_down_desc(
        const ggml_cgraph * cgraph,
        const qwen36_superlayer_plan & plan,
        const qwen36_superlayer_pack_plan & pack,
        const int device,
        qwen36_superlayer_l0_moe_down_desc * desc,
        std::string * blocker) {
    if (desc == nullptr) {
        if (blocker != nullptr) {
            *blocker = "missing L0 MoE down descriptor output";
        }
        return false;
    }
    *desc = qwen36_superlayer_l0_moe_down_desc{};

    const bool requested = qwen36_superlayer_l0_moe_down_requested();
    if (!requested) {
        return true;
    }

    const int begin = plan.layer_start[0] >= 0 ? plan.layer_start[0] : 0;
    const int end   = plan.layer_end[0]   >= begin ? plan.layer_end[0]   : cgraph->n_nodes - 1;
    const ggml_tensor * swiglu = nullptr;
    const ggml_tensor * topk = nullptr;
    const ggml_tensor * down = nullptr;
    const ggml_tensor * weighted = nullptr;
    const ggml_tensor * out = nullptr;
    const ggml_tensor * unsupported = nullptr;
    int down_idx = -1;
    int weighted_idx = -1;
    int out_idx = -1;
    for (int i = begin; i <= end; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (swiglu == nullptr && tensor_name_matches_layer(node, "ffn_moe_swiglu", 0)) {
            swiglu = node;
        } else if (topk == nullptr && tensor_name_matches_layer(node, "ffn_moe_topk", 0)) {
            topk = node;
        } else if (down == nullptr && tensor_name_matches_layer(node, "ffn_moe_down", 0)) {
            down = node;
            down_idx = i;
        } else if (weighted == nullptr && tensor_name_matches_layer(node, "ffn_moe_weighted", 0)) {
            weighted = node;
            weighted_idx = i;
        } else if (out == nullptr && tensor_name_matches_layer(node, "ffn_moe_out", 0)) {
            out = node;
            out_idx = i;
        } else if (tensor_name_matches_layer(node, "ffn_moe_down_biased", 0) ||
                tensor_name_matches_layer(node, "ffn_moe_down_scaled", 0)) {
            unsupported = node;
        }
    }

    if (unsupported != nullptr) {
        if (blocker != nullptr) {
            *blocker = "L0 MoE down fusion does not support biased/scaled down node: ";
            *blocker += unsupported->name;
        }
        return false;
    }
    if (swiglu == nullptr || topk == nullptr || down == nullptr || weighted == nullptr || out == nullptr ||
            down_idx < 0 || weighted_idx < 0 || out_idx < 0) {
        if (blocker != nullptr) {
            *blocker = "L0 MoE down fusion requires swiglu/topk/down/weighted/out nodes";
        }
        return false;
    }
    if (down->op != GGML_OP_MUL_MAT_ID || down->src[0] == nullptr ||
            down->src[1] == nullptr || down->src[2] == nullptr ||
            !qwen36_superlayer_same_tensor_or_view(down->src[1], swiglu) ||
            !qwen36_superlayer_same_tensor_or_view(down->src[2], topk)) {
        if (blocker != nullptr) {
            *blocker = "L0 MoE down fusion requires MUL_MAT_ID(weight, ffn_moe_swiglu-0, ffn_moe_topk-0)";
        }
        return false;
    }

    auto is_factor_tensor = [](const ggml_tensor * tensor, const int64_t n_expert_used, const int64_t n_tokens) {
        return tensor != nullptr &&
            tensor->type == GGML_TYPE_F32 &&
            tensor->ne[0] == 1 &&
            tensor->ne[1] == n_expert_used &&
            tensor->ne[2] == n_tokens;
    };
    auto mul_factor = [&](const ggml_tensor * mul, const ggml_tensor * expected,
            const int64_t n_expert_used, const int64_t n_tokens, const ggml_tensor *& factor) {
        if (mul == nullptr || mul->op != GGML_OP_MUL || mul->type != GGML_TYPE_F32) {
            return false;
        }
        if (mul->src[0] == expected && is_factor_tensor(mul->src[1], n_expert_used, n_tokens)) {
            factor = mul->src[1];
            return true;
        }
        if (mul->src[1] == expected && is_factor_tensor(mul->src[0], n_expert_used, n_tokens)) {
            factor = mul->src[0];
            return true;
        }
        return false;
    };

    const int64_t n_embd = down->ne[0];
    const int64_t n_expert_used = down->ne[1];
    const int64_t n_tokens = down->ne[2];
    const ggml_tensor * weights = nullptr;
    if (!mul_factor(weighted, down, n_expert_used, n_tokens, weights)) {
        if (blocker != nullptr) {
            *blocker = "L0 MoE down fusion requires ffn_moe_weighted-0 = ffn_moe_down-0 * weights";
        }
        return false;
    }

    const int views_start = weighted_idx + 1;
    const int adds_start = views_start + (int) n_expert_used;
    const int last_node = adds_start + (int) n_expert_used - 2;
    if (last_node != out_idx || last_node >= cgraph->n_nodes) {
        if (blocker != nullptr) {
            *blocker = "L0 MoE down fusion requires ffn_moe_out-0 to terminate the weighted view/add tail";
        }
        return false;
    }

    for (int i = 0; i < (int) n_expert_used; ++i) {
        const ggml_tensor * view = cgraph->nodes[views_start + i];
        if (view->op != GGML_OP_VIEW || view->src[0] != weighted || view->view_src != weighted ||
                view->type != GGML_TYPE_F32 || view->ne[0] != n_embd || view->ne[1] != n_tokens ||
                view->nb[0] != weighted->nb[0] || view->nb[1] != weighted->nb[2] ||
                view->view_offs != size_t(i)*weighted->nb[1]) {
            if (blocker != nullptr) {
                *blocker = "L0 MoE down fusion requires ordered expert views of ffn_moe_weighted-0";
            }
            return false;
        }
    }
    const ggml_tensor * prev = cgraph->nodes[views_start];
    for (int i = 1; i < (int) n_expert_used; ++i) {
        const ggml_tensor * add = cgraph->nodes[adds_start + i - 1];
        if (add->op != GGML_OP_ADD || add->src[0] != prev || add->src[1] != cgraph->nodes[views_start + i] ||
                add->type != GGML_TYPE_F32 || add->ne[0] != n_embd || add->ne[1] != n_tokens) {
            if (blocker != nullptr) {
                *blocker = "L0 MoE down fusion requires ordered add tail ending at ffn_moe_out-0";
            }
            return false;
        }
        prev = add;
    }

    const ggml_tensor * w = down->src[0];
    if (!qwen36_superlayer_projection_weight_supported(w)) {
        if (blocker != nullptr) {
            *blocker = "L0 MoE down weight type is not implemented in the superlayer";
        }
        return false;
    }
    if (down->src[1]->type != GGML_TYPE_F32 || down->src[2]->type != GGML_TYPE_I32 ||
            down->type != GGML_TYPE_F32 || weighted->type != GGML_TYPE_F32 || out->type != GGML_TYPE_F32) {
        if (blocker != nullptr) {
            *blocker = "L0 MoE down fusion currently requires F32 activations and I32 ids";
        }
        return false;
    }

    const int64_t n_ff = w->ne[0];
    const int64_t n_expert = w->ne[2];
    if (n_embd <= 0 || n_ff <= 0 || n_expert <= 0 ||
            n_expert_used <= 0 || n_expert_used > 64 || n_tokens != 1 ||
            w->ne[1] != n_embd || down->src[1]->ne[0] != n_ff ||
            down->src[1]->ne[1] != n_expert_used || down->src[1]->ne[2] != 1 ||
            ggml_nelements(topk) != n_expert_used ||
            down->ne[0] != n_embd || down->ne[1] != n_expert_used || down->ne[2] != 1 ||
            weighted->ne[0] != n_embd || weighted->ne[1] != n_expert_used || weighted->ne[2] != 1 ||
            out->ne[0] != n_embd || out->ne[1] != 1) {
        if (blocker != nullptr) {
            *blocker = "L0 MoE down tensor dimensions do not match one-token decode";
        }
        return false;
    }
    if (down->src[1]->nb[0] != (int64_t) sizeof(float) ||
            down->src[2]->nb[0] != (int64_t) sizeof(int32_t) ||
            weights->nb[0] != (int64_t) sizeof(float) ||
            down->nb[0] != (int64_t) sizeof(float) ||
            weighted->nb[0] != (int64_t) sizeof(float) ||
            out->nb[0] != (int64_t) sizeof(float) ||
            !ggml_is_contiguous_1(down->src[1]) ||
            !ggml_is_contiguous_1(down->src[2]) ||
            !ggml_is_contiguous_1(weights) ||
            !ggml_is_contiguous_1(down) ||
            !ggml_is_contiguous_1(weighted) ||
            !ggml_is_contiguous(out)) {
        if (blocker != nullptr) {
            *blocker = "L0 MoE down fusion requires dense one-token F32/I32 layout";
        }
        return false;
    }
    if (!qwen36_superlayer_tensor_data_on_device(down->src[1], device, "L0 MoE down input", blocker) ||
            !qwen36_superlayer_tensor_data_on_device(down->src[2], device, "L0 MoE down ids", blocker) ||
            !qwen36_superlayer_tensor_data_on_device(weights, device, "L0 MoE down weights", blocker) ||
            !qwen36_superlayer_tensor_data_on_device(down, device, "L0 MoE down output", blocker) ||
            !qwen36_superlayer_tensor_data_on_device(weighted, device, "L0 MoE weighted output", blocker) ||
            !qwen36_superlayer_tensor_data_on_device(out, device, "L0 MoE final output", blocker)) {
        return false;
    }

    const bool use_direct_weights = qwen36_superlayer_direct_l0_moe_weights_requested();
    uint64_t w_offset = 0;
    if (!qwen36_superlayer_find_pack_offset_named(pack, w, "L0 MoE down weight", &w_offset, blocker)) {
        return false;
    }
    if (use_direct_weights &&
            !qwen36_superlayer_tensor_data_on_device(w, device, "L0 MoE down weight", blocker)) {
        return false;
    }

    desc->x = (const float *) down->src[1]->data;
    desc->ids = (const int32_t *) down->src[2]->data;
    desc->weights = (const float *) weights->data;
    desc->down_dst = (float *) down->data;
    desc->weighted_dst = (float *) weighted->data;
    desc->out_dst = (float *) out->data;
    desc->w_data = (const char *) w->data;
    desc->w_offset = w_offset;
    desc->w_nb1 = (uint64_t) w->nb[1];
    desc->w_nb2 = (uint64_t) w->nb[2];
    desc->x_nb1 = (uint64_t) down->src[1]->nb[1];
    desc->weights_nb1 = (uint64_t) weights->nb[1];
    desc->down_nb1 = (uint64_t) down->nb[1];
    desc->weighted_nb1 = (uint64_t) weighted->nb[1];
    desc->out_nb1 = (uint64_t) out->nb[1];
    desc->n_ff = (uint32_t) n_ff;
    desc->n_embd = (uint32_t) n_embd;
    desc->n_expert = (uint32_t) n_expert;
    desc->n_expert_used = (uint32_t) n_expert_used;
    desc->w_type = (int32_t) w->type;
    desc->use_direct_weights = use_direct_weights ? 1u : 0u;
    desc->ready = 1u;
    return true;
}

static bool qwen36_superlayer_materialize_device_pack(
        ggml_backend_cuda_context * cuda_ctx,
        const ggml_cgraph * cgraph,
        const qwen36_superlayer_plan & plan,
        const qwen36_superlayer_pack_plan & pack,
        const bool device_weightpack_required,
        qwen36_superlayer_device_pack_view * view,
        std::string * blocker) {
    const uint64_t source_signature = qwen36_superlayer_source_signature(pack);
    const qwen36_superlayer_runtime_layout runtime =
        qwen36_superlayer_make_runtime_layout(cgraph, plan);
    const qwen36_superlayer_runtime_binding_plan bindings =
        qwen36_superlayer_make_runtime_binding_plan(cgraph, plan, cuda_ctx->device);
    if (bindings.refs.empty()) {
        if (blocker != nullptr) {
            *blocker = "superlayer runtime binding table would be empty";
        }
        return false;
    }
    const uint64_t runtime_signature = qwen36_superlayer_runtime_signature(bindings);
    const std::string key =
        qwen36_superlayer_device_pack_key(cuda_ctx->device, plan.fingerprint, source_signature) +
        (device_weightpack_required ? "-wp" : "-meta");
    const std::vector<qwen36_superlayer_runtime_tensor_desc> io_descs_host =
        qwen36_superlayer_make_runtime_descs(bindings);
    qwen36_superlayer_l0_norm_desc l0_norm_host;
    if (!qwen36_superlayer_make_l0_norm_desc(
                cgraph, plan, pack, cuda_ctx->device, &l0_norm_host, blocker)) {
        return false;
    }
    qwen36_superlayer_l0_qkv_desc l0_qkv_host;
    if (!qwen36_superlayer_make_l0_qkv_desc(
                cgraph, plan, pack, runtime, cuda_ctx->device, &l0_qkv_host, blocker)) {
        return false;
    }
    qwen36_superlayer_l0_proj_desc l0_proj_host;
    if (!qwen36_superlayer_make_l0_proj_desc(
                cgraph, plan, pack, runtime, cuda_ctx->device, &l0_proj_host, blocker)) {
        return false;
    }
    qwen36_superlayer_l0_ssm_desc l0_ssm_host;
    if (!qwen36_superlayer_make_l0_ssm_desc(
                cgraph, plan, pack, cuda_ctx->device, &l0_ssm_host, blocker)) {
        return false;
    }
    qwen36_superlayer_l0_l2_desc l0_l2_host;
    if (!qwen36_superlayer_make_l0_l2_desc(
                cgraph, plan, cuda_ctx->device, &l0_l2_host, blocker)) {
        return false;
    }
    qwen36_superlayer_l0_gdn_desc l0_gdn_host;
    if (!qwen36_superlayer_make_l0_gdn_desc(
                cgraph, plan, cuda_ctx->device, &l0_gdn_host, blocker)) {
        return false;
    }
    qwen36_superlayer_l0_gated_norm_desc l0_gated_norm_host;
    if (!qwen36_superlayer_make_l0_gated_norm_desc(
                cgraph, plan, pack, cuda_ctx->device, &l0_gated_norm_host, blocker)) {
        return false;
    }
    qwen36_superlayer_l0_attn_out_desc l0_attn_out_host;
    if (!qwen36_superlayer_make_l0_attn_out_desc(
                cgraph, plan, pack, cuda_ctx->device, &l0_attn_out_host, blocker)) {
        return false;
    }
    qwen36_superlayer_l0_post_attn_desc l0_post_attn_host;
    if (!qwen36_superlayer_make_l0_post_attn_desc(
                cgraph, plan, pack, cuda_ctx->device, &l0_post_attn_host, blocker)) {
        return false;
    }
    qwen36_superlayer_l0_moe_router_desc l0_moe_router_host;
    if (!qwen36_superlayer_make_l0_moe_router_desc(
                cgraph, plan, pack, cuda_ctx->device, &l0_moe_router_host, blocker)) {
        return false;
    }
    qwen36_superlayer_l0_moe_gate_up_desc l0_moe_gate_up_host;
    if (!qwen36_superlayer_make_l0_moe_gate_up_desc(
                cgraph, plan, pack, cuda_ctx->device, &l0_moe_gate_up_host, blocker)) {
        return false;
    }
    qwen36_superlayer_l0_moe_down_desc l0_moe_down_host;
    if (!qwen36_superlayer_make_l0_moe_down_desc(
                cgraph, plan, pack, cuda_ctx->device, &l0_moe_down_host, blocker)) {
        return false;
    }

    static std::mutex device_pack_mutex;
    static std::unordered_map<std::string, qwen36_superlayer_device_pack_entry> device_packs;
    static std::unordered_set<std::string> reported_device_packs;

    qwen36_superlayer_device_pack_view local_view;
    bool should_report = false;

    {
        std::lock_guard<std::mutex> lock(device_pack_mutex);
        auto it = device_packs.find(key);
        if (it == device_packs.end()) {
            if (device_weightpack_required) {
                for (const qwen36_superlayer_pack_ref & ref : pack.refs) {
                    if (!qwen36_superlayer_tensor_on_device(ref.tensor, cuda_ctx->device, blocker)) {
                        return false;
                    }
                }
            }

            ggml_cuda_set_device(cuda_ctx->device);

            void * data = nullptr;
            cudaError_t err = cudaSuccess;
            if (device_weightpack_required) {
                err = cudaMalloc(&data, pack.total_bytes);
                if (err != cudaSuccess) {
                    (void) cudaGetLastError();
                    qwen36_superlayer_set_cuda_blocker(blocker, "failed to allocate superlayer device weightpack", err);
                    return false;
                }
            }

            qwen36_superlayer_layer_pack_desc * layer_descs = nullptr;
            err = cudaMalloc((void **) &layer_descs, sizeof(qwen36_superlayer_layer_pack_desc)*40);
            if (err != cudaSuccess) {
                (void) cudaFree(data);
                (void) cudaGetLastError();
                qwen36_superlayer_set_cuda_blocker(blocker, "failed to allocate superlayer layer descriptors", err);
                return false;
            }

            void * scratch = nullptr;
            err = cudaMalloc(&scratch, runtime.scratch_bytes);
            if (err != cudaSuccess) {
                (void) cudaFree(layer_descs);
                (void) cudaFree(data);
                (void) cudaGetLastError();
                qwen36_superlayer_set_cuda_blocker(blocker, "failed to allocate superlayer runtime scratch", err);
                return false;
            }

            qwen36_superlayer_runtime_tensor_desc * io_descs = nullptr;
            err = cudaMalloc((void **) &io_descs, sizeof(qwen36_superlayer_runtime_tensor_desc)*io_descs_host.size());
            if (err != cudaSuccess) {
                (void) cudaFree(scratch);
                (void) cudaFree(layer_descs);
                (void) cudaFree(data);
                (void) cudaGetLastError();
                qwen36_superlayer_set_cuda_blocker(blocker, "failed to allocate superlayer runtime bindings", err);
                return false;
            }

            if (device_weightpack_required) {
                err = cudaMemsetAsync(data, 0, pack.total_bytes, cuda_ctx->stream());
                if (err != cudaSuccess) {
                    (void) cudaFree(io_descs);
                    (void) cudaFree(scratch);
                    (void) cudaFree(layer_descs);
                    (void) cudaFree(data);
                    qwen36_superlayer_set_cuda_blocker(blocker, "failed to clear superlayer device weightpack", err);
                    return false;
                }
            }
            err = cudaMemsetAsync(scratch, 0, runtime.scratch_bytes, cuda_ctx->stream());
            if (err != cudaSuccess) {
                (void) cudaFree(io_descs);
                (void) cudaFree(scratch);
                (void) cudaFree(layer_descs);
                (void) cudaFree(data);
                qwen36_superlayer_set_cuda_blocker(blocker, "failed to clear superlayer runtime scratch", err);
                return false;
            }

            const auto layer_descs_host = qwen36_superlayer_make_layer_pack_descs(plan, pack);
            err = cudaMemcpy(
                    layer_descs, layer_descs_host.data(), sizeof(qwen36_superlayer_layer_pack_desc)*40,
                    cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                (void) cudaFree(io_descs);
                (void) cudaFree(scratch);
                (void) cudaFree(layer_descs);
                (void) cudaFree(data);
                qwen36_superlayer_set_cuda_blocker(blocker, "failed to upload superlayer layer descriptors", err);
                return false;
            }
            err = cudaMemcpy(
                    io_descs, io_descs_host.data(),
                    sizeof(qwen36_superlayer_runtime_tensor_desc)*io_descs_host.size(),
                    cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                (void) cudaFree(io_descs);
                (void) cudaFree(scratch);
                (void) cudaFree(layer_descs);
                (void) cudaFree(data);
                qwen36_superlayer_set_cuda_blocker(blocker, "failed to upload superlayer runtime bindings", err);
                return false;
            }

            if (device_weightpack_required) {
                for (const qwen36_superlayer_pack_ref & ref : pack.refs) {
                    err = cudaMemcpyAsync(
                            (char *) data + ref.offset, ref.tensor->data, ref.nbytes,
                            cudaMemcpyDeviceToDevice, cuda_ctx->stream());
                    if (err != cudaSuccess) {
                        (void) cudaFree(io_descs);
                        (void) cudaFree(scratch);
                        (void) cudaFree(layer_descs);
                        (void) cudaFree(data);
                        qwen36_superlayer_set_cuda_blocker(
                                blocker, "failed to copy tensor into superlayer device weightpack", err);
                        return false;
                    }
                }
            }

            qwen36_superlayer_l0_norm_desc * l0_norm = nullptr;
            err = cudaMalloc((void **) &l0_norm, sizeof(qwen36_superlayer_l0_norm_desc));
            if (err != cudaSuccess) {
                (void) cudaFree(io_descs);
                (void) cudaFree(scratch);
                (void) cudaFree(layer_descs);
                (void) cudaFree(data);
                (void) cudaGetLastError();
                qwen36_superlayer_set_cuda_blocker(blocker, "failed to allocate L0 RMSNorm descriptor", err);
                return false;
            }
            err = cudaMemcpy(
                    l0_norm, &l0_norm_host, sizeof(qwen36_superlayer_l0_norm_desc),
                    cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                (void) cudaFree(l0_norm);
                (void) cudaFree(io_descs);
                (void) cudaFree(scratch);
                (void) cudaFree(layer_descs);
                (void) cudaFree(data);
                qwen36_superlayer_set_cuda_blocker(blocker, "failed to upload L0 RMSNorm descriptor", err);
                return false;
            }

            qwen36_superlayer_l0_qkv_desc * l0_qkv = nullptr;
            err = cudaMalloc((void **) &l0_qkv, sizeof(qwen36_superlayer_l0_qkv_desc));
            if (err != cudaSuccess) {
                (void) cudaFree(l0_norm);
                (void) cudaFree(io_descs);
                (void) cudaFree(scratch);
                (void) cudaFree(layer_descs);
                (void) cudaFree(data);
                (void) cudaGetLastError();
                qwen36_superlayer_set_cuda_blocker(blocker, "failed to allocate L0 QKV descriptor", err);
                return false;
            }
            err = cudaMemcpy(
                    l0_qkv, &l0_qkv_host, sizeof(qwen36_superlayer_l0_qkv_desc),
                    cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                (void) cudaFree(l0_qkv);
                (void) cudaFree(l0_norm);
                (void) cudaFree(io_descs);
                (void) cudaFree(scratch);
                (void) cudaFree(layer_descs);
                (void) cudaFree(data);
                qwen36_superlayer_set_cuda_blocker(blocker, "failed to upload L0 QKV descriptor", err);
                return false;
            }

            qwen36_superlayer_l0_proj_desc * l0_proj = nullptr;
            err = cudaMalloc((void **) &l0_proj, sizeof(qwen36_superlayer_l0_proj_desc));
            if (err != cudaSuccess) {
                (void) cudaFree(l0_qkv);
                (void) cudaFree(l0_norm);
                (void) cudaFree(io_descs);
                (void) cudaFree(scratch);
                (void) cudaFree(layer_descs);
                (void) cudaFree(data);
                (void) cudaGetLastError();
                qwen36_superlayer_set_cuda_blocker(blocker, "failed to allocate L0 projection descriptor", err);
                return false;
            }
            err = cudaMemcpy(
                    l0_proj, &l0_proj_host, sizeof(qwen36_superlayer_l0_proj_desc),
                    cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                (void) cudaFree(l0_proj);
                (void) cudaFree(l0_qkv);
                (void) cudaFree(l0_norm);
                (void) cudaFree(io_descs);
                (void) cudaFree(scratch);
                (void) cudaFree(layer_descs);
                (void) cudaFree(data);
                qwen36_superlayer_set_cuda_blocker(blocker, "failed to upload L0 projection descriptor", err);
                return false;
            }

            qwen36_superlayer_l0_ssm_desc * l0_ssm = nullptr;
            err = cudaMalloc((void **) &l0_ssm, sizeof(qwen36_superlayer_l0_ssm_desc));
            if (err != cudaSuccess) {
                (void) cudaFree(l0_proj);
                (void) cudaFree(l0_qkv);
                (void) cudaFree(l0_norm);
                (void) cudaFree(io_descs);
                (void) cudaFree(scratch);
                (void) cudaFree(layer_descs);
                (void) cudaFree(data);
                (void) cudaGetLastError();
                qwen36_superlayer_set_cuda_blocker(blocker, "failed to allocate L0 SSM descriptor", err);
                return false;
            }
            err = cudaMemcpy(
                    l0_ssm, &l0_ssm_host, sizeof(qwen36_superlayer_l0_ssm_desc),
                    cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                (void) cudaFree(l0_ssm);
                (void) cudaFree(l0_proj);
                (void) cudaFree(l0_qkv);
                (void) cudaFree(l0_norm);
                (void) cudaFree(io_descs);
                (void) cudaFree(scratch);
                (void) cudaFree(layer_descs);
                (void) cudaFree(data);
                qwen36_superlayer_set_cuda_blocker(blocker, "failed to upload L0 SSM descriptor", err);
                return false;
            }

            qwen36_superlayer_l0_l2_desc * l0_l2 = nullptr;
            err = cudaMalloc((void **) &l0_l2, sizeof(qwen36_superlayer_l0_l2_desc));
            if (err != cudaSuccess) {
                (void) cudaFree(l0_ssm);
                (void) cudaFree(l0_proj);
                (void) cudaFree(l0_qkv);
                (void) cudaFree(l0_norm);
                (void) cudaFree(io_descs);
                (void) cudaFree(scratch);
                (void) cudaFree(layer_descs);
                (void) cudaFree(data);
                (void) cudaGetLastError();
                qwen36_superlayer_set_cuda_blocker(blocker, "failed to allocate L0 L2Norm descriptor", err);
                return false;
            }
            err = cudaMemcpy(
                    l0_l2, &l0_l2_host, sizeof(qwen36_superlayer_l0_l2_desc),
                    cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                (void) cudaFree(l0_l2);
                (void) cudaFree(l0_ssm);
                (void) cudaFree(l0_proj);
                (void) cudaFree(l0_qkv);
                (void) cudaFree(l0_norm);
                (void) cudaFree(io_descs);
                (void) cudaFree(scratch);
                (void) cudaFree(layer_descs);
                (void) cudaFree(data);
                qwen36_superlayer_set_cuda_blocker(blocker, "failed to upload L0 L2Norm descriptor", err);
                return false;
            }

            qwen36_superlayer_l0_gdn_desc * l0_gdn = nullptr;
            err = cudaMalloc((void **) &l0_gdn, sizeof(qwen36_superlayer_l0_gdn_desc));
            if (err != cudaSuccess) {
                (void) cudaFree(l0_l2);
                (void) cudaFree(l0_ssm);
                (void) cudaFree(l0_proj);
                (void) cudaFree(l0_qkv);
                (void) cudaFree(l0_norm);
                (void) cudaFree(io_descs);
                (void) cudaFree(scratch);
                (void) cudaFree(layer_descs);
                (void) cudaFree(data);
                (void) cudaGetLastError();
                qwen36_superlayer_set_cuda_blocker(blocker, "failed to allocate L0 GDN descriptor", err);
                return false;
            }
            err = cudaMemcpy(
                    l0_gdn, &l0_gdn_host, sizeof(qwen36_superlayer_l0_gdn_desc),
                    cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                (void) cudaFree(l0_gdn);
                (void) cudaFree(l0_l2);
                (void) cudaFree(l0_ssm);
                (void) cudaFree(l0_proj);
                (void) cudaFree(l0_qkv);
                (void) cudaFree(l0_norm);
                (void) cudaFree(io_descs);
                (void) cudaFree(scratch);
                (void) cudaFree(layer_descs);
                (void) cudaFree(data);
                qwen36_superlayer_set_cuda_blocker(blocker, "failed to upload L0 GDN descriptor", err);
                return false;
            }

            qwen36_superlayer_l0_gated_norm_desc * l0_gated_norm = nullptr;
            err = cudaMalloc((void **) &l0_gated_norm, sizeof(qwen36_superlayer_l0_gated_norm_desc));
            if (err != cudaSuccess) {
                (void) cudaFree(l0_gdn);
                (void) cudaFree(l0_l2);
                (void) cudaFree(l0_ssm);
                (void) cudaFree(l0_proj);
                (void) cudaFree(l0_qkv);
                (void) cudaFree(l0_norm);
                (void) cudaFree(io_descs);
                (void) cudaFree(scratch);
                (void) cudaFree(layer_descs);
                (void) cudaFree(data);
                (void) cudaGetLastError();
                qwen36_superlayer_set_cuda_blocker(blocker, "failed to allocate L0 gated norm descriptor", err);
                return false;
            }
            err = cudaMemcpy(
                    l0_gated_norm, &l0_gated_norm_host, sizeof(qwen36_superlayer_l0_gated_norm_desc),
                    cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                (void) cudaFree(l0_gated_norm);
                (void) cudaFree(l0_gdn);
                (void) cudaFree(l0_l2);
                (void) cudaFree(l0_ssm);
                (void) cudaFree(l0_proj);
                (void) cudaFree(l0_qkv);
                (void) cudaFree(l0_norm);
                (void) cudaFree(io_descs);
                (void) cudaFree(scratch);
                (void) cudaFree(layer_descs);
                (void) cudaFree(data);
                qwen36_superlayer_set_cuda_blocker(blocker, "failed to upload L0 gated norm descriptor", err);
                return false;
            }

            qwen36_superlayer_l0_attn_out_desc * l0_attn_out = nullptr;
            err = cudaMalloc((void **) &l0_attn_out, sizeof(qwen36_superlayer_l0_attn_out_desc));
            if (err != cudaSuccess) {
                (void) cudaFree(l0_gated_norm);
                (void) cudaFree(l0_gdn);
                (void) cudaFree(l0_l2);
                (void) cudaFree(l0_ssm);
                (void) cudaFree(l0_proj);
                (void) cudaFree(l0_qkv);
                (void) cudaFree(l0_norm);
                (void) cudaFree(io_descs);
                (void) cudaFree(scratch);
                (void) cudaFree(layer_descs);
                (void) cudaFree(data);
                (void) cudaGetLastError();
                qwen36_superlayer_set_cuda_blocker(blocker, "failed to allocate L0 attention output descriptor", err);
                return false;
            }
            err = cudaMemcpy(
                    l0_attn_out, &l0_attn_out_host, sizeof(qwen36_superlayer_l0_attn_out_desc),
                    cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                (void) cudaFree(l0_attn_out);
                (void) cudaFree(l0_gated_norm);
                (void) cudaFree(l0_gdn);
                (void) cudaFree(l0_l2);
                (void) cudaFree(l0_ssm);
                (void) cudaFree(l0_proj);
                (void) cudaFree(l0_qkv);
                (void) cudaFree(l0_norm);
                (void) cudaFree(io_descs);
                (void) cudaFree(scratch);
                (void) cudaFree(layer_descs);
                (void) cudaFree(data);
                qwen36_superlayer_set_cuda_blocker(blocker, "failed to upload L0 attention output descriptor", err);
                return false;
            }

            qwen36_superlayer_l0_post_attn_desc * l0_post_attn = nullptr;
            err = cudaMalloc((void **) &l0_post_attn, sizeof(qwen36_superlayer_l0_post_attn_desc));
            if (err != cudaSuccess) {
                (void) cudaFree(l0_attn_out);
                (void) cudaFree(l0_gated_norm);
                (void) cudaFree(l0_gdn);
                (void) cudaFree(l0_l2);
                (void) cudaFree(l0_ssm);
                (void) cudaFree(l0_proj);
                (void) cudaFree(l0_qkv);
                (void) cudaFree(l0_norm);
                (void) cudaFree(io_descs);
                (void) cudaFree(scratch);
                (void) cudaFree(layer_descs);
                (void) cudaFree(data);
                (void) cudaGetLastError();
                qwen36_superlayer_set_cuda_blocker(blocker, "failed to allocate L0 post-attention descriptor", err);
                return false;
            }
            err = cudaMemcpy(
                    l0_post_attn, &l0_post_attn_host, sizeof(qwen36_superlayer_l0_post_attn_desc),
                    cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                (void) cudaFree(l0_post_attn);
                (void) cudaFree(l0_attn_out);
                (void) cudaFree(l0_gated_norm);
                (void) cudaFree(l0_gdn);
                (void) cudaFree(l0_l2);
                (void) cudaFree(l0_ssm);
                (void) cudaFree(l0_proj);
                (void) cudaFree(l0_qkv);
                (void) cudaFree(l0_norm);
                (void) cudaFree(io_descs);
                (void) cudaFree(scratch);
                (void) cudaFree(layer_descs);
                (void) cudaFree(data);
                qwen36_superlayer_set_cuda_blocker(blocker, "failed to upload L0 post-attention descriptor", err);
                return false;
            }

            qwen36_superlayer_l0_moe_router_desc * l0_moe_router = nullptr;
            err = cudaMalloc((void **) &l0_moe_router, sizeof(qwen36_superlayer_l0_moe_router_desc));
            if (err != cudaSuccess) {
                (void) cudaFree(l0_post_attn);
                (void) cudaFree(l0_attn_out);
                (void) cudaFree(l0_gated_norm);
                (void) cudaFree(l0_gdn);
                (void) cudaFree(l0_l2);
                (void) cudaFree(l0_ssm);
                (void) cudaFree(l0_proj);
                (void) cudaFree(l0_qkv);
                (void) cudaFree(l0_norm);
                (void) cudaFree(io_descs);
                (void) cudaFree(scratch);
                (void) cudaFree(layer_descs);
                (void) cudaFree(data);
                (void) cudaGetLastError();
                qwen36_superlayer_set_cuda_blocker(blocker, "failed to allocate L0 MoE router descriptor", err);
                return false;
            }
            err = cudaMemcpy(
                    l0_moe_router, &l0_moe_router_host, sizeof(qwen36_superlayer_l0_moe_router_desc),
                    cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                (void) cudaFree(l0_moe_router);
                (void) cudaFree(l0_post_attn);
                (void) cudaFree(l0_attn_out);
                (void) cudaFree(l0_gated_norm);
                (void) cudaFree(l0_gdn);
                (void) cudaFree(l0_l2);
                (void) cudaFree(l0_ssm);
                (void) cudaFree(l0_proj);
                (void) cudaFree(l0_qkv);
                (void) cudaFree(l0_norm);
                (void) cudaFree(io_descs);
                (void) cudaFree(scratch);
                (void) cudaFree(layer_descs);
                (void) cudaFree(data);
                qwen36_superlayer_set_cuda_blocker(blocker, "failed to upload L0 MoE router descriptor", err);
                return false;
            }

            qwen36_superlayer_l0_moe_gate_up_desc * l0_moe_gate_up = nullptr;
            err = cudaMalloc((void **) &l0_moe_gate_up, sizeof(qwen36_superlayer_l0_moe_gate_up_desc));
            if (err != cudaSuccess) {
                (void) cudaFree(l0_moe_router);
                (void) cudaFree(l0_post_attn);
                (void) cudaFree(l0_attn_out);
                (void) cudaFree(l0_gated_norm);
                (void) cudaFree(l0_gdn);
                (void) cudaFree(l0_l2);
                (void) cudaFree(l0_ssm);
                (void) cudaFree(l0_proj);
                (void) cudaFree(l0_qkv);
                (void) cudaFree(l0_norm);
                (void) cudaFree(io_descs);
                (void) cudaFree(scratch);
                (void) cudaFree(layer_descs);
                (void) cudaFree(data);
                (void) cudaGetLastError();
                qwen36_superlayer_set_cuda_blocker(blocker, "failed to allocate L0 MoE gate/up descriptor", err);
                return false;
            }
            err = cudaMemcpy(
                    l0_moe_gate_up, &l0_moe_gate_up_host, sizeof(qwen36_superlayer_l0_moe_gate_up_desc),
                    cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                (void) cudaFree(l0_moe_gate_up);
                (void) cudaFree(l0_moe_router);
                (void) cudaFree(l0_post_attn);
                (void) cudaFree(l0_attn_out);
                (void) cudaFree(l0_gated_norm);
                (void) cudaFree(l0_gdn);
                (void) cudaFree(l0_l2);
                (void) cudaFree(l0_ssm);
                (void) cudaFree(l0_proj);
                (void) cudaFree(l0_qkv);
                (void) cudaFree(l0_norm);
                (void) cudaFree(io_descs);
                (void) cudaFree(scratch);
                (void) cudaFree(layer_descs);
                (void) cudaFree(data);
                qwen36_superlayer_set_cuda_blocker(blocker, "failed to upload L0 MoE gate/up descriptor", err);
                return false;
            }

            qwen36_superlayer_l0_moe_down_desc * l0_moe_down = nullptr;
            err = cudaMalloc((void **) &l0_moe_down, sizeof(qwen36_superlayer_l0_moe_down_desc));
            if (err != cudaSuccess) {
                (void) cudaFree(l0_moe_gate_up);
                (void) cudaFree(l0_moe_router);
                (void) cudaFree(l0_post_attn);
                (void) cudaFree(l0_attn_out);
                (void) cudaFree(l0_gated_norm);
                (void) cudaFree(l0_gdn);
                (void) cudaFree(l0_l2);
                (void) cudaFree(l0_ssm);
                (void) cudaFree(l0_proj);
                (void) cudaFree(l0_qkv);
                (void) cudaFree(l0_norm);
                (void) cudaFree(io_descs);
                (void) cudaFree(scratch);
                (void) cudaFree(layer_descs);
                (void) cudaFree(data);
                (void) cudaGetLastError();
                qwen36_superlayer_set_cuda_blocker(blocker, "failed to allocate L0 MoE down descriptor", err);
                return false;
            }
            err = cudaMemcpy(
                    l0_moe_down, &l0_moe_down_host, sizeof(qwen36_superlayer_l0_moe_down_desc),
                    cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                (void) cudaFree(l0_moe_down);
                (void) cudaFree(l0_moe_gate_up);
                (void) cudaFree(l0_moe_router);
                (void) cudaFree(l0_post_attn);
                (void) cudaFree(l0_attn_out);
                (void) cudaFree(l0_gated_norm);
                (void) cudaFree(l0_gdn);
                (void) cudaFree(l0_l2);
                (void) cudaFree(l0_ssm);
                (void) cudaFree(l0_proj);
                (void) cudaFree(l0_qkv);
                (void) cudaFree(l0_norm);
                (void) cudaFree(io_descs);
                (void) cudaFree(scratch);
                (void) cudaFree(layer_descs);
                (void) cudaFree(data);
                qwen36_superlayer_set_cuda_blocker(blocker, "failed to upload L0 MoE down descriptor", err);
                return false;
            }

            cudaEvent_t ready_event = nullptr;
            err = cudaEventCreateWithFlags(&ready_event, cudaEventDisableTiming);
            if (err != cudaSuccess) {
                (void) cudaFree(l0_moe_down);
                (void) cudaFree(l0_moe_gate_up);
                (void) cudaFree(l0_moe_router);
                (void) cudaFree(l0_post_attn);
                (void) cudaFree(l0_attn_out);
                (void) cudaFree(l0_gated_norm);
                (void) cudaFree(l0_gdn);
                (void) cudaFree(l0_l2);
                (void) cudaFree(l0_ssm);
                (void) cudaFree(l0_proj);
                (void) cudaFree(l0_qkv);
                (void) cudaFree(l0_norm);
                (void) cudaFree(io_descs);
                (void) cudaFree(scratch);
                (void) cudaFree(layer_descs);
                (void) cudaFree(data);
                qwen36_superlayer_set_cuda_blocker(blocker, "failed to create superlayer weightpack event", err);
                return false;
            }

            err = cudaEventRecord(ready_event, cuda_ctx->stream());
            if (err != cudaSuccess) {
                (void) cudaEventDestroy(ready_event);
                (void) cudaFree(l0_moe_down);
                (void) cudaFree(l0_moe_gate_up);
                (void) cudaFree(l0_moe_router);
                (void) cudaFree(l0_post_attn);
                (void) cudaFree(l0_attn_out);
                (void) cudaFree(l0_gated_norm);
                (void) cudaFree(l0_gdn);
                (void) cudaFree(l0_l2);
                (void) cudaFree(l0_ssm);
                (void) cudaFree(l0_proj);
                (void) cudaFree(l0_qkv);
                (void) cudaFree(l0_norm);
                (void) cudaFree(io_descs);
                (void) cudaFree(scratch);
                (void) cudaFree(layer_descs);
                (void) cudaFree(data);
                qwen36_superlayer_set_cuda_blocker(blocker, "failed to record superlayer weightpack event", err);
                return false;
            }

            qwen36_superlayer_device_pack_entry entry;
            entry.device = cuda_ctx->device;
            entry.fingerprint = plan.fingerprint;
            entry.source_signature = source_signature;
            entry.runtime_signature = runtime_signature;
            entry.data = data;
            entry.layers = layer_descs;
            entry.io_descs = io_descs;
            entry.l0_norm = l0_norm;
            entry.l0_qkv = l0_qkv;
            entry.l0_proj = l0_proj;
            entry.l0_ssm = l0_ssm;
            entry.l0_l2 = l0_l2;
            entry.l0_gdn = l0_gdn;
            entry.l0_gated_norm = l0_gated_norm;
            entry.l0_attn_out = l0_attn_out;
            entry.l0_post_attn = l0_post_attn;
            entry.l0_moe_router = l0_moe_router;
            entry.l0_moe_gate_up = l0_moe_gate_up;
            entry.l0_moe_down = l0_moe_down;
            entry.scratch = scratch;
            entry.bytes = device_weightpack_required ? pack.total_bytes : 0;
            entry.tensors = device_weightpack_required ? pack.refs.size() : 0;
            entry.io_count = io_descs_host.size();
            entry.io_capacity = io_descs_host.size();
            entry.runtime = runtime;
            entry.ready_event = ready_event;

            it = device_packs.emplace(key, entry).first;
            should_report = reported_device_packs.insert(key).second;
        } else {
            ggml_cuda_set_device(cuda_ctx->device);
            if (io_descs_host.size() > it->second.io_capacity) {
                qwen36_superlayer_runtime_tensor_desc * io_descs = nullptr;
                cudaError_t err = cudaMalloc(
                        (void **) &io_descs,
                        sizeof(qwen36_superlayer_runtime_tensor_desc)*io_descs_host.size());
                if (err != cudaSuccess) {
                    (void) cudaGetLastError();
                    qwen36_superlayer_set_cuda_blocker(
                            blocker, "failed to resize superlayer runtime bindings", err);
                    return false;
                }
                (void) cudaFree(it->second.io_descs);
                it->second.io_descs = io_descs;
                it->second.io_capacity = io_descs_host.size();
            }
            if (runtime.scratch_bytes > it->second.runtime.scratch_bytes) {
                void * scratch = nullptr;
                cudaError_t err = cudaMalloc(&scratch, runtime.scratch_bytes);
                if (err != cudaSuccess) {
                    (void) cudaGetLastError();
                    qwen36_superlayer_set_cuda_blocker(
                            blocker, "failed to grow superlayer runtime scratch", err);
                    return false;
                }
                err = cudaMemsetAsync(scratch, 0, runtime.scratch_bytes, cuda_ctx->stream());
                if (err != cudaSuccess) {
                    (void) cudaFree(scratch);
                    qwen36_superlayer_set_cuda_blocker(
                            blocker, "failed to clear grown superlayer runtime scratch", err);
                    return false;
                }
                (void) cudaFree(it->second.scratch);
                it->second.scratch = scratch;
            }

            cudaError_t err = cudaMemcpy(
                    it->second.io_descs, io_descs_host.data(),
                    sizeof(qwen36_superlayer_runtime_tensor_desc)*io_descs_host.size(),
                    cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                qwen36_superlayer_set_cuda_blocker(blocker, "failed to refresh superlayer runtime bindings", err);
                return false;
            }

            if (it->second.l0_norm == nullptr) {
                qwen36_superlayer_l0_norm_desc * l0_norm = nullptr;
                cudaError_t err = cudaMalloc((void **) &l0_norm, sizeof(qwen36_superlayer_l0_norm_desc));
                if (err != cudaSuccess) {
                    (void) cudaGetLastError();
                    qwen36_superlayer_set_cuda_blocker(blocker, "failed to allocate L0 RMSNorm descriptor", err);
                    return false;
                }
                it->second.l0_norm = l0_norm;
            }
            err = cudaMemcpy(
                    it->second.l0_norm, &l0_norm_host, sizeof(qwen36_superlayer_l0_norm_desc),
                    cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                qwen36_superlayer_set_cuda_blocker(blocker, "failed to refresh L0 RMSNorm descriptor", err);
                return false;
            }

            if (it->second.l0_qkv == nullptr) {
                qwen36_superlayer_l0_qkv_desc * l0_qkv = nullptr;
                cudaError_t err = cudaMalloc((void **) &l0_qkv, sizeof(qwen36_superlayer_l0_qkv_desc));
                if (err != cudaSuccess) {
                    (void) cudaGetLastError();
                    qwen36_superlayer_set_cuda_blocker(blocker, "failed to allocate L0 QKV descriptor", err);
                    return false;
                }
                it->second.l0_qkv = l0_qkv;
            }
            err = cudaMemcpy(
                    it->second.l0_qkv, &l0_qkv_host, sizeof(qwen36_superlayer_l0_qkv_desc),
                    cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                qwen36_superlayer_set_cuda_blocker(blocker, "failed to refresh L0 QKV descriptor", err);
                return false;
            }

            if (it->second.l0_proj == nullptr) {
                qwen36_superlayer_l0_proj_desc * l0_proj = nullptr;
                cudaError_t err = cudaMalloc((void **) &l0_proj, sizeof(qwen36_superlayer_l0_proj_desc));
                if (err != cudaSuccess) {
                    (void) cudaGetLastError();
                    qwen36_superlayer_set_cuda_blocker(blocker, "failed to allocate L0 projection descriptor", err);
                    return false;
                }
                it->second.l0_proj = l0_proj;
            }
            err = cudaMemcpy(
                    it->second.l0_proj, &l0_proj_host, sizeof(qwen36_superlayer_l0_proj_desc),
                    cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                qwen36_superlayer_set_cuda_blocker(blocker, "failed to refresh L0 projection descriptor", err);
                return false;
            }

            if (it->second.l0_ssm == nullptr) {
                qwen36_superlayer_l0_ssm_desc * l0_ssm = nullptr;
                cudaError_t err = cudaMalloc((void **) &l0_ssm, sizeof(qwen36_superlayer_l0_ssm_desc));
                if (err != cudaSuccess) {
                    (void) cudaGetLastError();
                    qwen36_superlayer_set_cuda_blocker(blocker, "failed to allocate L0 SSM descriptor", err);
                    return false;
                }
                it->second.l0_ssm = l0_ssm;
            }
            err = cudaMemcpy(
                    it->second.l0_ssm, &l0_ssm_host, sizeof(qwen36_superlayer_l0_ssm_desc),
                    cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                qwen36_superlayer_set_cuda_blocker(blocker, "failed to refresh L0 SSM descriptor", err);
                return false;
            }

            if (it->second.l0_l2 == nullptr) {
                qwen36_superlayer_l0_l2_desc * l0_l2 = nullptr;
                cudaError_t err = cudaMalloc((void **) &l0_l2, sizeof(qwen36_superlayer_l0_l2_desc));
                if (err != cudaSuccess) {
                    (void) cudaGetLastError();
                    qwen36_superlayer_set_cuda_blocker(blocker, "failed to allocate L0 L2Norm descriptor", err);
                    return false;
                }
                it->second.l0_l2 = l0_l2;
            }
            err = cudaMemcpy(
                    it->second.l0_l2, &l0_l2_host, sizeof(qwen36_superlayer_l0_l2_desc),
                    cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                qwen36_superlayer_set_cuda_blocker(blocker, "failed to refresh L0 L2Norm descriptor", err);
                return false;
            }

            if (it->second.l0_gdn == nullptr) {
                qwen36_superlayer_l0_gdn_desc * l0_gdn = nullptr;
                cudaError_t err = cudaMalloc((void **) &l0_gdn, sizeof(qwen36_superlayer_l0_gdn_desc));
                if (err != cudaSuccess) {
                    (void) cudaGetLastError();
                    qwen36_superlayer_set_cuda_blocker(blocker, "failed to allocate L0 GDN descriptor", err);
                    return false;
                }
                it->second.l0_gdn = l0_gdn;
            }
            err = cudaMemcpy(
                    it->second.l0_gdn, &l0_gdn_host, sizeof(qwen36_superlayer_l0_gdn_desc),
                    cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                qwen36_superlayer_set_cuda_blocker(blocker, "failed to refresh L0 GDN descriptor", err);
                return false;
            }

            if (it->second.l0_gated_norm == nullptr) {
                qwen36_superlayer_l0_gated_norm_desc * l0_gated_norm = nullptr;
                cudaError_t err = cudaMalloc(
                        (void **) &l0_gated_norm, sizeof(qwen36_superlayer_l0_gated_norm_desc));
                if (err != cudaSuccess) {
                    (void) cudaGetLastError();
                    qwen36_superlayer_set_cuda_blocker(blocker, "failed to allocate L0 gated norm descriptor", err);
                    return false;
                }
                it->second.l0_gated_norm = l0_gated_norm;
            }
            err = cudaMemcpy(
                    it->second.l0_gated_norm, &l0_gated_norm_host,
                    sizeof(qwen36_superlayer_l0_gated_norm_desc),
                    cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                qwen36_superlayer_set_cuda_blocker(blocker, "failed to refresh L0 gated norm descriptor", err);
                return false;
            }

            if (it->second.l0_attn_out == nullptr) {
                qwen36_superlayer_l0_attn_out_desc * l0_attn_out = nullptr;
                cudaError_t err = cudaMalloc(
                        (void **) &l0_attn_out, sizeof(qwen36_superlayer_l0_attn_out_desc));
                if (err != cudaSuccess) {
                    (void) cudaGetLastError();
                    qwen36_superlayer_set_cuda_blocker(
                            blocker, "failed to allocate L0 attention output descriptor", err);
                    return false;
                }
                it->second.l0_attn_out = l0_attn_out;
            }
            err = cudaMemcpy(
                    it->second.l0_attn_out, &l0_attn_out_host,
                    sizeof(qwen36_superlayer_l0_attn_out_desc),
                    cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                qwen36_superlayer_set_cuda_blocker(blocker, "failed to refresh L0 attention output descriptor", err);
                return false;
            }

            if (it->second.l0_post_attn == nullptr) {
                qwen36_superlayer_l0_post_attn_desc * l0_post_attn = nullptr;
                cudaError_t err = cudaMalloc(
                        (void **) &l0_post_attn, sizeof(qwen36_superlayer_l0_post_attn_desc));
                if (err != cudaSuccess) {
                    (void) cudaGetLastError();
                    qwen36_superlayer_set_cuda_blocker(
                            blocker, "failed to allocate L0 post-attention descriptor", err);
                    return false;
                }
                it->second.l0_post_attn = l0_post_attn;
            }
            err = cudaMemcpy(
                    it->second.l0_post_attn, &l0_post_attn_host,
                    sizeof(qwen36_superlayer_l0_post_attn_desc),
                    cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                qwen36_superlayer_set_cuda_blocker(blocker, "failed to refresh L0 post-attention descriptor", err);
                return false;
            }

            if (it->second.l0_moe_router == nullptr) {
                qwen36_superlayer_l0_moe_router_desc * l0_moe_router = nullptr;
                cudaError_t err = cudaMalloc(
                        (void **) &l0_moe_router, sizeof(qwen36_superlayer_l0_moe_router_desc));
                if (err != cudaSuccess) {
                    (void) cudaGetLastError();
                    qwen36_superlayer_set_cuda_blocker(
                            blocker, "failed to allocate L0 MoE router descriptor", err);
                    return false;
                }
                it->second.l0_moe_router = l0_moe_router;
            }
            err = cudaMemcpy(
                    it->second.l0_moe_router, &l0_moe_router_host,
                    sizeof(qwen36_superlayer_l0_moe_router_desc),
                    cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                qwen36_superlayer_set_cuda_blocker(blocker, "failed to refresh L0 MoE router descriptor", err);
                return false;
            }

            if (it->second.l0_moe_gate_up == nullptr) {
                qwen36_superlayer_l0_moe_gate_up_desc * l0_moe_gate_up = nullptr;
                cudaError_t err = cudaMalloc(
                        (void **) &l0_moe_gate_up, sizeof(qwen36_superlayer_l0_moe_gate_up_desc));
                if (err != cudaSuccess) {
                    (void) cudaGetLastError();
                    qwen36_superlayer_set_cuda_blocker(
                            blocker, "failed to allocate L0 MoE gate/up descriptor", err);
                    return false;
                }
                it->second.l0_moe_gate_up = l0_moe_gate_up;
            }
            err = cudaMemcpy(
                    it->second.l0_moe_gate_up, &l0_moe_gate_up_host,
                    sizeof(qwen36_superlayer_l0_moe_gate_up_desc),
                    cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                qwen36_superlayer_set_cuda_blocker(blocker, "failed to refresh L0 MoE gate/up descriptor", err);
                return false;
            }

            if (it->second.l0_moe_down == nullptr) {
                qwen36_superlayer_l0_moe_down_desc * l0_moe_down = nullptr;
                cudaError_t err = cudaMalloc(
                        (void **) &l0_moe_down, sizeof(qwen36_superlayer_l0_moe_down_desc));
                if (err != cudaSuccess) {
                    (void) cudaGetLastError();
                    qwen36_superlayer_set_cuda_blocker(
                            blocker, "failed to allocate L0 MoE down descriptor", err);
                    return false;
                }
                it->second.l0_moe_down = l0_moe_down;
            }
            err = cudaMemcpy(
                    it->second.l0_moe_down, &l0_moe_down_host,
                    sizeof(qwen36_superlayer_l0_moe_down_desc),
                    cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                qwen36_superlayer_set_cuda_blocker(blocker, "failed to refresh L0 MoE down descriptor", err);
                return false;
            }

            it->second.runtime_signature = runtime_signature;
            it->second.io_count = io_descs_host.size();
            it->second.runtime = runtime;
        }

        local_view.data = it->second.data;
        local_view.layers = it->second.layers;
        local_view.io_descs = it->second.io_descs;
        local_view.l0_norm = it->second.l0_norm;
        local_view.l0_qkv = it->second.l0_qkv;
        local_view.l0_proj = it->second.l0_proj;
        local_view.l0_ssm = it->second.l0_ssm;
        local_view.l0_l2 = it->second.l0_l2;
        local_view.l0_gdn = it->second.l0_gdn;
        local_view.l0_gated_norm = it->second.l0_gated_norm;
        local_view.l0_attn_out = it->second.l0_attn_out;
        local_view.l0_post_attn = it->second.l0_post_attn;
        local_view.l0_moe_router = it->second.l0_moe_router;
        local_view.l0_moe_gate_up = it->second.l0_moe_gate_up;
        local_view.l0_moe_down = it->second.l0_moe_down;
        local_view.scratch = it->second.scratch;
        local_view.bytes = it->second.bytes;
        local_view.tensors = it->second.tensors;
        local_view.io_count = it->second.io_count;
        local_view.io_capacity = it->second.io_capacity;
        local_view.runtime = it->second.runtime;
        local_view.ready_event = it->second.ready_event;
    }

    const cudaError_t err = cudaStreamWaitEvent(cuda_ctx->stream(), local_view.ready_event, 0);
    if (err != cudaSuccess) {
        qwen36_superlayer_set_cuda_blocker(blocker, "failed to wait for superlayer device weightpack", err);
        return false;
    }

    if (view != nullptr) {
        *view = local_view;
    }

    if (should_report && qwen36_superlayer_trace_enabled()) {
        fprintf(stderr,
                "rdna3_qwen36_superlayer: device-pack-ready fingerprint=%s source=%s runtime=%s"
                " ptr=%p layer_descs=%p io_descs=%p l0_norm=%p l0_qkv=%p l0_proj=%p l0_ssm=%p l0_l2=%p l0_gdn=%p l0_gated_norm=%p l0_attn_out=%p l0_post_attn=%p l0_moe_router=%p l0_moe_gate_up=%p l0_moe_down=%p"
                " scratch=%p tensors=%zu io=%zu bytes=%zu"
                " scratch_bytes=%zu activation_slot=%zu projection_slot=%zu logits_bytes=%zu router_slot=%zu"
                " n_embd=%" PRId64 " n_vocab=%" PRId64 "\n",
                hex_u64(plan.fingerprint).c_str(), hex_u64(source_signature).c_str(),
                hex_u64(runtime_signature).c_str(),
                local_view.data, local_view.layers, local_view.io_descs, local_view.l0_norm, local_view.l0_qkv,
                local_view.l0_proj, local_view.l0_ssm, local_view.l0_l2, local_view.l0_gdn,
                local_view.l0_gated_norm, local_view.l0_attn_out, local_view.l0_post_attn,
                local_view.l0_moe_router, local_view.l0_moe_gate_up, local_view.l0_moe_down, local_view.scratch,
                local_view.tensors, local_view.io_count, local_view.bytes,
                local_view.runtime.scratch_bytes, local_view.runtime.activation_slot_bytes,
                local_view.runtime.projection_slot_bytes, local_view.runtime.logits_bytes, local_view.runtime.router_slot_bytes,
                local_view.runtime.n_embd, local_view.runtime.n_vocab);
        fflush(stderr);
    }

    return true;
}

static uint64_t qwen36_superlayer_fingerprint(const ggml_cgraph * cgraph, const int device) {
    uint64_t h = 1469598103934665603ull;
    h = fnv1a_update_cstr(h, "qwen36-rdna3-7900xtx-superlayer-v0");
    h = fnv1a_update_pod(h, ggml_cuda_info().devices[device].cc);
    h = fnv1a_update_pod(h, ggml_cuda_info().devices[device].nsm);
    h = fnv1a_update_pod(h, cgraph->n_nodes);
    h = fnv1a_update_pod(h, cgraph->n_leafs);

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (node == nullptr) {
            h = fnv1a_update_cstr(h, "<null-node>");
            continue;
        }

        h = fnv1a_update_cstr(h, node->name);
        h = fnv1a_update_pod(h, node->op);
        h = fnv1a_update_pod(h, node->type);
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            h = fnv1a_update_pod(h, node->ne[d]);
            h = fnv1a_update_pod(h, node->nb[d]);
        }

        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            const ggml_tensor * src = node->src[s];
            if (src == nullptr) {
                continue;
            }
            h = fnv1a_update_cstr(h, src->name);
            h = fnv1a_update_pod(h, src->op);
            h = fnv1a_update_pod(h, src->type);
            for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                h = fnv1a_update_pod(h, src->ne[d]);
                h = fnv1a_update_pod(h, src->nb[d]);
            }
        }
    }

    return h;
}

static qwen36_superlayer_plan qwen36_superlayer_make_plan(const ggml_cgraph * cgraph, const int device) {
    qwen36_superlayer_plan plan;

    if (cgraph == nullptr || cgraph->n_nodes <= 0) {
        plan.blocker = "empty graph";
        return plan;
    }

    const int cc = ggml_cuda_info().devices[device].cc;
    if (!GGML_CUDA_CC_IS_RDNA3(cc)) {
        plan.blocker = "device is not RDNA3/gfx11";
        return plan;
    }

    plan.n_nodes = cgraph->n_nodes;

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (node == nullptr) {
            continue;
        }

        switch (node->op) {
            case GGML_OP_FLASH_ATTN_EXT:
                plan.n_fattn++;
                break;
            case GGML_OP_GATED_DELTA_NET:
                plan.n_gdn++;
                break;
            case GGML_OP_SSM_CONV:
                plan.n_ssm_conv++;
                break;
            case GGML_OP_MUL_MAT_ID:
                plan.n_mmid++;
                if (tensor_name_starts_with(node, "ffn_moe_down")) {
                    plan.n_moe_down++;
                } else if (tensor_name_starts_with(node, "ffn_moe_gate")) {
                    plan.n_moe_gate_up++;
                }
                break;
            case GGML_OP_SOFT_MAX:
                if (tensor_name_starts_with(node, "ffn_moe_probs")) {
                    plan.n_topk_moe++;
                }
                break;
            case GGML_OP_MUL_MAT:
                if (tensor_name_starts_with(node, "result_output") && node->ne[1] == 1) {
                    plan.has_decode_out = true;
                    plan.is_decode_token = true;
                }
                break;
            default:
                break;
        }

        if (tensor_name_starts_with(node, "result_output") && node->ne[1] == 1) {
            plan.has_decode_out = true;
            plan.is_decode_token = true;
        }

        for (int layer = 0; layer < 40; ++layer) {
            if (plan.layer_start[layer] < 0 && tensor_name_matches_layer(node, "attn_norm", layer)) {
                plan.layer_start[layer] = i;
            }
        }
    }

    for (int layer = 0; layer < 40; ++layer) {
        if (plan.layer_start[layer] < 0) {
            continue;
        }
        if (layer + 1 < 40 && plan.layer_start[layer + 1] > plan.layer_start[layer]) {
            plan.layer_end[layer] = plan.layer_start[layer + 1] - 1;
        } else {
            plan.layer_end[layer] = cgraph->n_nodes - 1;
        }
        for (int i = plan.layer_start[layer]; i <= plan.layer_end[layer]; ++i) {
            const ggml_tensor * node = cgraph->nodes[i];
            if (node != nullptr && node->op == GGML_OP_GATED_DELTA_NET) {
                plan.layer_recurrent[layer] = true;
                break;
            }
        }
    }

    int layer_count = 0;
    for (int layer = 0; layer < 40; ++layer) {
        layer_count += plan.layer_start[layer] >= 0 ? 1 : 0;
    }
    int64_t l0_tokens = -1;
    if (plan.layer_start[0] >= 0) {
        const ggml_tensor * l0 = cgraph->nodes[plan.layer_start[0]];
        if (l0 != nullptr && l0->ne[0] > 0) {
            l0_tokens = ggml_nelements(l0) / l0->ne[0];
        }
    }

    if (!plan.has_decode_out || !plan.is_decode_token) {
        plan.blocker = "not a one-token decoder graph ending in result_output";
    } else if (layer_count != 40) {
        plan.blocker = "Qwen3.6 superlayer requires exactly 40 named layer spans";
    } else if (l0_tokens != 1) {
        plan.blocker = "not a one-token decoder graph internally; attn_norm-0 tokens=" + std::to_string(l0_tokens);
    } else if (plan.n_fattn != 10) {
        plan.blocker = "Qwen3.6 signature mismatch: expected 10 full-attention layers";
    } else if (plan.n_gdn != 30) {
        plan.blocker = "Qwen3.6 signature mismatch: expected 30 GDN/recurrent layers";
    } else if (plan.n_topk_moe != 40) {
        plan.blocker = "Qwen3.6 signature mismatch: expected 40 MoE router nodes";
    } else if (plan.n_moe_down != 40) {
        plan.blocker = "Qwen3.6 signature mismatch: expected 40 MoE down projections";
    } else if (plan.n_moe_gate_up < 40) {
        plan.blocker = "Qwen3.6 signature mismatch: expected fused MoE gate/up projections";
    }

    if (plan.blocker.empty()) {
        plan.fingerprint = qwen36_superlayer_fingerprint(cgraph, device);
        plan.artifact_dir = qwen36_superlayer_cache_root() / hex_u64(plan.fingerprint);
    }

    return plan;
}

static bool write_text_file(const std::filesystem::path & path, const std::string & text, std::string * blocker) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        if (blocker != nullptr) {
            *blocker = "failed to open artifact file for writing: " + path.string();
        }
        return false;
    }

    out << text;
    if (!out) {
        if (blocker != nullptr) {
            *blocker = "failed to write artifact file: " + path.string();
        }
        return false;
    }

    return true;
}

static std::string c_escape(const std::string & value) {
    std::string escaped;
    escaped.reserve(value.size() + 8);
    for (char c : value) {
        switch (c) {
            case '\\': escaped += "\\\\"; break;
            case '"':  escaped += "\\\""; break;
            case '\n': escaped += "\\n";  break;
            case '\r': escaped += "\\r";  break;
            case '\t': escaped += "\\t";  break;
            default:   escaped += c;       break;
        }
    }
    return escaped;
}

static std::string qwen36_superlayer_layout_source(
        const qwen36_superlayer_plan & plan,
        const qwen36_superlayer_pack_plan & pack,
        const qwen36_superlayer_runtime_binding_plan & bindings,
        const int device) {
    std::ostringstream out;
    out << "#pragma once\n\n";
    out << "#include <cstdint>\n\n";
    out << "#define QWEN36_RDNA3_SUPERLAYER_FINGERPRINT 0x" << hex_u64(plan.fingerprint) << "ull\n";
    out << "#define QWEN36_RDNA3_SUPERLAYER_DEVICE_CC " << ggml_cuda_info().devices[device].cc << "\n";
    out << "#define QWEN36_RDNA3_SUPERLAYER_DEVICE_CU " << ggml_cuda_info().devices[device].nsm << "\n";
    out << "#define QWEN36_RDNA3_SUPERLAYER_LOGICAL_LAYERS 40\n\n";
    out << "#define QWEN36_RDNA3_WEIGHTPACK_TENSORS " << pack.refs.size() << "\n";
    out << "#define QWEN36_RDNA3_WEIGHTPACK_BYTES " << pack.total_bytes << "ull\n\n";
    out << "#define QWEN36_RDNA3_RUNTIME_BINDINGS " << bindings.refs.size() << "\n\n";
    out << "enum qwen36_rdna3_layer_kind {\n";
    out << "    QWEN36_LAYER_ATTN = 0,\n";
    out << "    QWEN36_LAYER_GDN  = 1,\n";
    out << "};\n\n";
    out << "static constexpr qwen36_rdna3_layer_kind QWEN36_LAYER_KINDS[40] = {\n";
    for (int layer = 0; layer < 40; ++layer) {
        out << "    " << (plan.layer_recurrent[layer] ? "QWEN36_LAYER_GDN" : "QWEN36_LAYER_ATTN")
            << (layer == 39 ? "\n" : ",\n");
    }
    out << "};\n\n";

    out << "struct qwen36_rdna3_layer_pack_desc {\n";
    out << "    uint64_t byte_begin;\n";
    out << "    uint64_t byte_end;\n";
    out << "    uint32_t first_ref;\n";
    out << "    uint32_t n_refs;\n";
    out << "    uint32_t kind;\n";
    out << "    uint32_t reserved;\n";
    out << "};\n\n";
    out << "struct qwen36_rdna3_runtime_tensor_desc {\n";
    out << "    void * data;\n";
    out << "    uint64_t nbytes;\n";
    out << "    int32_t type;\n";
    out << "    int64_t ne[4];\n";
    out << "    uint64_t nb[4];\n";
    out << "    uint64_t role_hash;\n";
    out << "    uint64_t name_hash;\n";
    out << "};\n\n";
    out << "static constexpr qwen36_rdna3_layer_pack_desc QWEN36_LAYER_PACK[40] = {\n";
    const auto layer_descs = qwen36_superlayer_make_layer_pack_descs(plan, pack);
    for (int layer = 0; layer < 40; ++layer) {
        const qwen36_superlayer_layer_pack_desc & desc = layer_descs[layer];
        out << "    { " << desc.byte_begin << "ull, " << desc.byte_end << "ull, "
            << desc.first_ref << "u, " << desc.n_refs << "u, " << desc.kind << "u, 0u }"
            << (layer == 39 ? "\n" : ",\n");
    }
    out << "};\n";
    return out.str();
}

static std::string qwen36_superlayer_weightpack_header(const qwen36_superlayer_pack_plan & pack) {
    std::ostringstream out;
    out << "#pragma once\n\n";
    out << "#include <cstdint>\n\n";
    out << "struct qwen36_rdna3_weight_ref {\n";
    out << "    uint64_t offset;\n";
    out << "    uint64_t nbytes;\n";
    out << "    int32_t  layer;\n";
    out << "    int32_t  type;\n";
    out << "    int64_t  ne[4];\n";
    out << "    int64_t  nb[4];\n";
    out << "    const char * role;\n";
    out << "    const char * tensor;\n";
    out << "};\n\n";
    out << "static constexpr qwen36_rdna3_weight_ref QWEN36_RDNA3_WEIGHTPACK[] = {\n";
    for (const qwen36_superlayer_pack_ref & ref : pack.refs) {
        const ggml_tensor * t = ref.tensor;
        out << "    { " << ref.offset << "ull, " << ref.nbytes << "ull, " << ref.layer
            << ", " << (int) t->type << ", { "
            << t->ne[0] << ", " << t->ne[1] << ", " << t->ne[2] << ", " << t->ne[3]
            << " }, { "
            << t->nb[0] << ", " << t->nb[1] << ", " << t->nb[2] << ", " << t->nb[3]
            << " }, \"" << c_escape(ref.role) << "\", \"" << c_escape(t->name) << "\" },\n";
    }
    out << "};\n";
    return out.str();
}

static std::string qwen36_superlayer_generated_source(const qwen36_superlayer_plan & plan) {
    std::ostringstream out;
    out << "// Generated scaffold for the RDNA3 Qwen3.6 physical superlayer.\n";
    out << "// This file is intentionally topology-specific. It is not a ggml graph executor.\n\n";
    out << "#include <cstdint>\n\n";
    out << "#include \"layout.generated.h\"\n\n";
    out << "#include \"weightpack.generated.h\"\n\n";
    out << "struct qwen36_superlayer_runtime {\n";
    out << "    const uint8_t * weightpack;\n";
    out << "    const qwen36_rdna3_runtime_tensor_desc * io_descs;\n";
    out << "    uint32_t io_count;\n";
    out << "    uint8_t * scratch;\n";
    out << "    uint64_t scratch_bytes;\n";
    out << "    uint64_t checksum;\n";
    out << "};\n\n";
    for (int layer = 0; layer < 40; ++layer) {
        out << "__device__ __forceinline__ void qwen36_l" << (layer < 10 ? "0" : "") << layer
            << "_" << (plan.layer_recurrent[layer] ? "gdn" : "attn")
            << "_fused(qwen36_superlayer_runtime & rt) {\n";
        out << "    constexpr qwen36_rdna3_layer_pack_desc desc = QWEN36_LAYER_PACK[" << layer << "];\n";
        out << "    if (rt.weightpack != nullptr && desc.n_refs != 0 && desc.byte_end > desc.byte_begin) {\n";
        out << "        rt.checksum ^= ((uint64_t) rt.weightpack[desc.byte_begin] << " << ((layer & 7)*8)
            << ") ^ desc.byte_end;\n";
        out << "    }\n";
        out << "    if (rt.scratch != nullptr && rt.scratch_bytes > " << layer << ") {\n";
        out << "        rt.scratch[" << layer << "] = (uint8_t) rt.checksum;\n";
        out << "    }\n";
        out << "    if (rt.io_descs != nullptr && rt.io_count != 0) {\n";
        out << "        const qwen36_rdna3_runtime_tensor_desc io = rt.io_descs[" << layer
            << " % rt.io_count];\n";
        out << "        rt.checksum ^= io.role_hash ^ io.name_hash ^ io.nbytes;\n";
        out << "    }\n";
        out << "    // TODO: generated RDNA3 dataflow for physical layer " << layer << ".\n";
        out << "}\n\n";
    }

    out << "__device__ __forceinline__ void qwen36_7900xtx_fused_superlayer(qwen36_superlayer_runtime & rt) {\n";
    for (int layer = 0; layer < 40; ++layer) {
        out << "    qwen36_l" << (layer < 10 ? "0" : "") << layer
            << "_" << (plan.layer_recurrent[layer] ? "gdn" : "attn") << "_fused(rt);\n";
    }
    out << "}\n";
    return out.str();
}

static std::string qwen36_superlayer_manifest_json(
        const qwen36_superlayer_plan & plan,
        const qwen36_superlayer_pack_plan & pack,
        const qwen36_superlayer_runtime_layout & runtime,
        const qwen36_superlayer_runtime_binding_plan & bindings,
        const int device) {
    std::ostringstream out;
    out << "{\n";
    out << "  \"artifact\": \"qwen36-rdna3-7900xtx-superlayer\",\n";
    out << "  \"state\": \"device-pack-layout\",\n";
    out << "  \"fingerprint\": \"" << hex_u64(plan.fingerprint) << "\",\n";
    out << "  \"device_cc\": " << ggml_cuda_info().devices[device].cc << ",\n";
    out << "  \"device_cu\": " << ggml_cuda_info().devices[device].nsm << ",\n";
    out << "  \"n_nodes\": " << plan.n_nodes << ",\n";
    out << "  \"n_fattn\": " << plan.n_fattn << ",\n";
    out << "  \"n_gdn\": " << plan.n_gdn << ",\n";
    out << "  \"n_ssm_conv\": " << plan.n_ssm_conv << ",\n";
    out << "  \"n_topk_moe\": " << plan.n_topk_moe << ",\n";
    out << "  \"n_moe_gate_up\": " << plan.n_moe_gate_up << ",\n";
    out << "  \"n_moe_down\": " << plan.n_moe_down << ",\n";
    out << "  \"n_mmid\": " << plan.n_mmid << ",\n";
    out << "  \"weightpack_tensors\": " << pack.refs.size() << ",\n";
    out << "  \"weightpack_bytes\": " << pack.total_bytes << ",\n";
    out << "  \"runtime_bindings\": " << bindings.refs.size() << ",\n";
    out << "  \"activation_slot_bytes\": " << runtime.activation_slot_bytes << ",\n";
    out << "  \"projection_slot_bytes\": " << runtime.projection_slot_bytes << ",\n";
    out << "  \"logits_bytes\": " << runtime.logits_bytes << ",\n";
    out << "  \"router_slot_bytes\": " << runtime.router_slot_bytes << ",\n";
    out << "  \"scratch_bytes\": " << runtime.scratch_bytes << ",\n";
    out << "  \"n_embd\": " << runtime.n_embd << ",\n";
    out << "  \"n_vocab\": " << runtime.n_vocab << ",\n";
    out << "  \"runtime_device_weightpack\": true,\n";
    out << "  \"fusion_contract\": \"load-time generated physical superlayer; no generic layer loop in generated source\"\n";
    out << "}\n";
    return out.str();
}

static std::string qwen36_superlayer_runtime_json(
        const qwen36_superlayer_runtime_layout & runtime,
        const qwen36_superlayer_runtime_binding_plan & bindings) {
    std::ostringstream out;
    out << "{\n";
    out << "  \"runtime_bindings\": " << bindings.refs.size() << ",\n";
    out << "  \"activation_slot_bytes\": " << runtime.activation_slot_bytes << ",\n";
    out << "  \"projection_slot_bytes\": " << runtime.projection_slot_bytes << ",\n";
    out << "  \"logits_bytes\": " << runtime.logits_bytes << ",\n";
    out << "  \"router_slot_bytes\": " << runtime.router_slot_bytes << ",\n";
    out << "  \"scratch_bytes\": " << runtime.scratch_bytes << ",\n";
    out << "  \"n_embd\": " << runtime.n_embd << ",\n";
    out << "  \"n_vocab\": " << runtime.n_vocab << ",\n";
    out << "  \"buffers\": {\n";
    out << "    \"activation_a_offset\": 0,\n";
    out << "    \"activation_b_offset\": " << runtime.activation_slot_bytes << ",\n";
    out << "    \"projection_offset\": " << 2*runtime.activation_slot_bytes << ",\n";
    out << "    \"logits_offset\": " << 2*runtime.activation_slot_bytes + runtime.projection_slot_bytes << ",\n";
    out << "    \"router_offset\": "
        << 2*runtime.activation_slot_bytes + runtime.projection_slot_bytes + runtime.logits_bytes << "\n";
    out << "  }\n";
    out << "}\n";
    return out.str();
}

static std::string qwen36_superlayer_runtime_bindings_json(
        const qwen36_superlayer_runtime_binding_plan & bindings) {
    std::ostringstream out;
    out << "{\n";
    out << "  \"binding_count\": " << bindings.refs.size() << ",\n";
    out << "  \"bindings\": [\n";
    for (size_t i = 0; i < bindings.refs.size(); ++i) {
        const qwen36_superlayer_runtime_ref & ref = bindings.refs[i];
        const ggml_tensor * t = ref.tensor;
        out << "    { \"index\": " << i
            << ", \"role\": \"" << c_escape(ref.role) << "\""
            << ", \"tensor\": \"" << c_escape(t->name) << "\""
            << ", \"nbytes\": " << ref.nbytes
            << ", \"type\": \"" << ggml_type_name(t->type) << "\""
            << ", \"role_hash\": \"" << hex_u64(qwen36_superlayer_hash_string(ref.role.c_str())) << "\""
            << ", \"name_hash\": \"" << hex_u64(qwen36_superlayer_hash_string(t->name)) << "\""
            << ", \"ne\": [" << t->ne[0] << ", " << t->ne[1] << ", " << t->ne[2] << ", " << t->ne[3] << "]"
            << ", \"nb\": [" << t->nb[0] << ", " << t->nb[1] << ", " << t->nb[2] << ", " << t->nb[3] << "] }"
            << (i + 1 == bindings.refs.size() ? "\n" : ",\n");
    }
    out << "  ]\n";
    out << "}\n";
    return out.str();
}

static std::string qwen36_superlayer_weightpack_plan(const qwen36_superlayer_plan & plan, const qwen36_superlayer_pack_plan & pack) {
    std::ostringstream out;
    out << "# RDNA3 Qwen3.6 7900 XTX Weight Pack Plan\n\n";
    out << "fingerprint: " << hex_u64(plan.fingerprint) << "\n";
    out << "state: device-pack-layout\n";
    out << "tensor_count: " << pack.refs.size() << "\n";
    out << "total_bytes: " << pack.total_bytes << "\n\n";
    out << "This is the deterministic physical pack order for gfx1100. At runtime the backend\n";
    out << "allocates one device-side buffer and copies these tensors into the listed offsets.\n";
    out << "The next step is to replace original tensor loads with offsets into that buffer.\n\n";
    out << "Packed tensors:\n\n";
    for (const qwen36_superlayer_pack_ref & ref : pack.refs) {
        const ggml_tensor * t = ref.tensor;
        out << "- offset=" << ref.offset
            << " bytes=" << ref.nbytes
            << " align=" << ref.alignment
            << " layer=" << ref.layer
            << " type=" << ggml_type_name(t->type)
            << " role=" << ref.role
            << " tensor=" << t->name
            << " ne=[" << t->ne[0] << "," << t->ne[1] << "," << t->ne[2] << "," << t->ne[3] << "]"
            << " nb=[" << t->nb[0] << "," << t->nb[1] << "," << t->nb[2] << "," << t->nb[3] << "]\n";
    }
    out << "\n";
    out << "Layer spans detected from the decode graph:\n\n";
    for (int layer = 0; layer < 40; ++layer) {
        out << "- layer " << layer << ": "
            << (plan.layer_recurrent[layer] ? "gdn" : "attention")
            << " nodes [" << plan.layer_start[layer] << ", " << plan.layer_end[layer] << "]\n";
    }
    return out.str();
}

static std::string qwen36_superlayer_weightpack_json(const qwen36_superlayer_pack_plan & pack) {
    std::ostringstream out;
    out << "{\n";
    out << "  \"tensor_count\": " << pack.refs.size() << ",\n";
    out << "  \"total_bytes\": " << pack.total_bytes << ",\n";
    out << "  \"tensors\": [\n";
    for (size_t i = 0; i < pack.refs.size(); ++i) {
        const qwen36_superlayer_pack_ref & ref = pack.refs[i];
        const ggml_tensor * t = ref.tensor;
        out << "    { \"offset\": " << ref.offset
            << ", \"nbytes\": " << ref.nbytes
            << ", \"alignment\": " << ref.alignment
            << ", \"layer\": " << ref.layer
            << ", \"type\": \"" << ggml_type_name(t->type) << "\""
            << ", \"role\": \"" << c_escape(ref.role) << "\""
            << ", \"tensor\": \"" << c_escape(t->name) << "\""
            << ", \"ne\": [" << t->ne[0] << ", " << t->ne[1] << ", " << t->ne[2] << ", " << t->ne[3] << "]"
            << ", \"nb\": [" << t->nb[0] << ", " << t->nb[1] << ", " << t->nb[2] << ", " << t->nb[3] << "] }"
            << (i + 1 == pack.refs.size() ? "\n" : ",\n");
    }
    out << "  ]\n";
    out << "}\n";
    return out.str();
}

static bool qwen36_superlayer_materialize_artifact(
        const qwen36_superlayer_plan & plan,
        const qwen36_superlayer_pack_plan & pack,
        const qwen36_superlayer_runtime_layout & runtime,
        const qwen36_superlayer_runtime_binding_plan & bindings,
        const int device,
        std::string * blocker) {
    try {
        std::filesystem::create_directories(plan.artifact_dir);
    } catch (const std::filesystem::filesystem_error & e) {
        if (blocker != nullptr) {
            *blocker = std::string("failed to create superlayer artifact directory: ") + e.what();
        }
        return false;
    }

    const std::filesystem::path manifest_path = plan.artifact_dir / "manifest.json";
    const std::filesystem::path layout_path   = plan.artifact_dir / "layout.generated.h";
    const std::filesystem::path source_path   = plan.artifact_dir / "fused_model.hip";
    const std::filesystem::path pack_path     = plan.artifact_dir / "weightpack.plan";
    const std::filesystem::path pack_h_path   = plan.artifact_dir / "weightpack.generated.h";
    const std::filesystem::path pack_json_path = plan.artifact_dir / "weightpack.layout.json";
    const std::filesystem::path runtime_json_path = plan.artifact_dir / "runtime.layout.json";
    const std::filesystem::path runtime_bindings_path = plan.artifact_dir / "runtime-bindings.layout.json";

    if (!write_text_file(manifest_path, qwen36_superlayer_manifest_json(plan, pack, runtime, bindings, device), blocker)) {
        return false;
    }
    if (!write_text_file(layout_path, qwen36_superlayer_layout_source(plan, pack, bindings, device), blocker)) {
        return false;
    }
    if (!write_text_file(source_path, qwen36_superlayer_generated_source(plan), blocker)) {
        return false;
    }
    if (!write_text_file(pack_path, qwen36_superlayer_weightpack_plan(plan, pack), blocker)) {
        return false;
    }
    if (!write_text_file(pack_h_path, qwen36_superlayer_weightpack_header(pack), blocker)) {
        return false;
    }
    if (!write_text_file(pack_json_path, qwen36_superlayer_weightpack_json(pack), blocker)) {
        return false;
    }
    if (!write_text_file(runtime_json_path, qwen36_superlayer_runtime_json(runtime, bindings), blocker)) {
        return false;
    }
    if (!write_text_file(runtime_bindings_path, qwen36_superlayer_runtime_bindings_json(bindings), blocker)) {
        return false;
    }

    return true;
}

struct qwen36_superlayer_contract_state {
    uint64_t fingerprint;
    uint64_t weightpack_bytes;
    uint64_t scratch_bytes;
    uint64_t checksum;
    uint32_t n_nodes;
    uint32_t n_blocks;
    uint32_t weightpack_tensors;
    uint32_t touched_layers;
    uint32_t active_lanes;
};

template<int L>
__device__ __forceinline__ uint64_t qwen36_rdna3_superlayer_contract_layer(
        const uint8_t * weightpack,
        const qwen36_superlayer_layer_pack_desc * layer_descs,
        const uint64_t lane) {
    const qwen36_superlayer_layer_pack_desc desc = layer_descs[L];
    if (weightpack == nullptr || desc.n_refs == 0 || desc.byte_end <= desc.byte_begin) {
        return 0x9e3779b97f4a7c15ull ^ ((uint64_t) L << 32);
    }

    const uint64_t span = desc.byte_end - desc.byte_begin;
    const uint64_t rel = (lane*1315423911ull + (uint64_t) L*2654435761ull) % span;
    const uint64_t value = (uint64_t) weightpack[desc.byte_begin + rel];

    return (value << ((L & 7)*8)) ^
        desc.byte_begin ^
        (desc.byte_end << 1) ^
        ((uint64_t) desc.n_refs << 40) ^
        ((uint64_t) desc.kind << 56) ^
        ((uint64_t) L << 48);
}

static __device__ __forceinline__ void qwen36_rdna3_superlayer_get_scale_min_k4(
        const int j,
        const uint8_t * __restrict__ q,
        uint8_t & d,
        uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

static __device__ __forceinline__ float qwen36_rdna3_superlayer_dequant_q4_k(
        const char * __restrict__ row,
        const int64_t col) {
    const block_q4_K * blocks = (const block_q4_K *) row;
    const block_q4_K & b = blocks[col / QK_K];
    const int k = (int) (col % QK_K);
    const int g64 = k >> 6;
    const int in64 = k & 63;
    const int half = in64 >> 5;
    const int q_index = 32*g64 + (in64 & 31);

    uint8_t sc;
    uint8_t m;
    qwen36_rdna3_superlayer_get_scale_min_k4(2*g64 + half, b.scales, sc, m);

    const uint8_t q_byte = b.qs[q_index];
    const int q = half == 0 ? (q_byte & 0x0F) : (q_byte >> 4);
    const float d_all = __low2half(b.dm);
    const float m_all = __high2half(b.dm);
    return d_all * sc * q - m_all * m;
}

static __device__ __forceinline__ float qwen36_rdna3_superlayer_dequant_q6_k(
        const char * __restrict__ row,
        const int64_t col) {
    const block_q6_K * blocks = (const block_q6_K *) row;
    const block_q6_K & b = blocks[col / QK_K];
    const int k = (int) (col % QK_K);
    const int ip = k >> 7;
    const int in128 = k & 127;
    const int il = in128 & 31;
    const uint8_t qh = b.qh[32*ip + il];
    const int scale_base = 8*ip + il/16;
    int q;
    int scale_index;

    if (in128 < 32) {
        q = (b.ql[64*ip + il] & 0x0F) | (((qh >> 0) & 3) << 4);
        scale_index = scale_base + 0;
    } else if (in128 < 64) {
        q = (b.ql[64*ip + il + 32] & 0x0F) | (((qh >> 2) & 3) << 4);
        scale_index = scale_base + 2;
    } else if (in128 < 96) {
        q = (b.ql[64*ip + il] >> 4) | (((qh >> 4) & 3) << 4);
        scale_index = scale_base + 4;
    } else {
        q = (b.ql[64*ip + il + 32] >> 4) | (((qh >> 6) & 3) << 4);
        scale_index = scale_base + 6;
    }

    return (float) b.d * b.scales[scale_index] * ((int8_t) q - 32);
}

static __device__ __forceinline__ float qwen36_rdna3_superlayer_dequant_weight(
        const char * __restrict__ w,
        const ggml_type wtype,
        const int64_t w_nb1,
        const int64_t row,
        const int64_t col) {
    const char * wrow = w + row*w_nb1;
    switch (wtype) {
        case GGML_TYPE_Q4_K:
            return qwen36_rdna3_superlayer_dequant_q4_k(wrow, col);
        case GGML_TYPE_Q6_K:
            return qwen36_rdna3_superlayer_dequant_q6_k(wrow, col);
        case GGML_TYPE_Q8_0: {
            const block_q8_0 * blocks = (const block_q8_0 *) wrow;
            const block_q8_0 & b = blocks[col / QK8_0];
            return (float) b.d * b.qs[col % QK8_0];
        }
        case GGML_TYPE_F16:
            return (float) ((const ggml_half *) wrow)[col];
        case GGML_TYPE_F32:
            return ((const float *) wrow)[col];
        default:
            return 0.0f;
    }
}

__device__ __forceinline__ void qwen36_rdna3_superlayer_l0_rms_norm(
        const qwen36_superlayer_l0_norm_desc * desc,
        const uint8_t * weightpack,
        uint8_t * scratch,
        const uint64_t scratch_bytes,
        cooperative_groups::grid_group grid,
        float * partial_sums,
        float * sums,
        const uint32_t write_outputs) {
    if (desc == nullptr || desc->ready == 0 || desc->x == nullptr ||
            scratch == nullptr || partial_sums == nullptr || sums == nullptr) {
        return;
    }

    const uint32_t n_embd = desc->n_embd;
    if (n_embd == 0 || scratch_bytes < (uint64_t) n_embd*sizeof(float) ||
            scratch_bytes < (uint64_t) (gridDim.x + 1)*sizeof(float)) {
        return;
    }
    const bool use_direct_weights = desc->use_direct_weights != 0;
    if (desc->has_norm_w != 0 && !use_direct_weights && weightpack == nullptr) {
        return;
    }
    if (desc->has_norm_w != 0 && use_direct_weights && desc->norm_w_data == nullptr) {
        return;
    }

    const int tid = threadIdx.x;
    float sum = 0.0f;

    for (uint32_t i = blockIdx.x*blockDim.x + tid; i < n_embd; i += gridDim.x*blockDim.x) {
        const float x = desc->x[i];
        sum += x*x;
    }

    sums[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sums[tid] += sums[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = sums[0];
    }
    grid.sync();

    if (blockIdx.x == 0) {
        float total = 0.0f;
        for (uint32_t i = tid; i < gridDim.x; i += blockDim.x) {
            total += partial_sums[i];
        }
        sums[tid] = total;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                sums[tid] += sums[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            partial_sums[gridDim.x] = rsqrtf(sums[0]/(float) n_embd + desc->eps);
        }
    }
    grid.sync();

    const float inv_rms = partial_sums[gridDim.x];
    const float * norm_w = desc->has_norm_w != 0 ?
        (use_direct_weights ? desc->norm_w_data : (const float *) (weightpack + desc->norm_w_offset)) : nullptr;
    float * dst = (float *) scratch;

    for (uint32_t i = blockIdx.x*blockDim.x + tid; i < n_embd; i += gridDim.x*blockDim.x) {
        const float w = norm_w != nullptr ? norm_w[i] : 1.0f;
        const float v = desc->x[i]*inv_rms*w;
        dst[i] = v;
        if (write_outputs != 0 && desc->norm_out != nullptr) {
            desc->norm_out[i] = v;
        }
    }
    grid.sync();
}

__device__ __forceinline__ void qwen36_rdna3_superlayer_l0_qkv(
        const qwen36_superlayer_l0_qkv_desc * desc,
        const uint8_t * weightpack,
        uint8_t * scratch,
        const uint64_t scratch_bytes,
        float * sums,
        const uint32_t write_outputs) {
    if (desc == nullptr || desc->ready == 0 || scratch == nullptr || sums == nullptr) {
        return;
    }

    const bool use_direct_weights = desc->use_direct_weights != 0;
    if (!use_direct_weights && weightpack == nullptr) {
        return;
    }
    if (use_direct_weights && desc->wqkv_data == nullptr) {
        return;
    }

    const uint32_t n_embd = desc->n_embd;
    const uint32_t n_out = desc->n_out;
    const uint64_t qkv_bytes = (uint64_t) n_out*sizeof(float);
    if (n_embd == 0 || n_out == 0 ||
            desc->qkv_scratch_offset + qkv_bytes > scratch_bytes) {
        return;
    }

    const float * norm = (const float *) scratch;
    float * qkv = (float *) (scratch + desc->qkv_scratch_offset);
    const char * wqkv = use_direct_weights ? desc->wqkv_data : (const char *) (weightpack + desc->wqkv_offset);
    const float * qkv_scale = desc->has_qkv_scale != 0 ?
        (use_direct_weights ? desc->qkv_scale_data : (const float *) (weightpack + desc->qkv_scale_offset)) : nullptr;
    if (desc->has_qkv_scale != 0 && qkv_scale == nullptr) {
        return;
    }
    const ggml_type wqkv_type = (ggml_type) desc->wqkv_type;

    (void) sums;

    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp = threadIdx.x / WARP_SIZE;
    const int rows_per_block = max(1, blockDim.x / WARP_SIZE);
    for (uint32_t row = blockIdx.x*rows_per_block + warp; row < n_out; row += gridDim.x*rows_per_block) {
        float acc = 0.0f;
        for (uint32_t col = lane; col < n_embd; col += WARP_SIZE) {
            const float w = qwen36_rdna3_superlayer_dequant_weight(
                    wqkv, wqkv_type, (int64_t) desc->wqkv_nb1, row, col);
            acc += norm[col]*w;
        }

        const float raw = warp_reduce_sum<WARP_SIZE>(acc);
        if (lane == 0) {
            if (write_outputs != 0 && desc->qkv_math_out != nullptr) {
                desc->qkv_math_out[row] = raw;
            }
            const float scale = qkv_scale == nullptr ? 1.0f :
                qkv_scale[desc->qkv_scale_n == 1 ? 0 : row];
            const float v = raw*scale;
            qkv[row] = v;
            if (write_outputs != 0 && desc->qkv_out != nullptr) {
                desc->qkv_out[row] = v;
            }
            if (write_outputs != 0 && desc->qkv_named_out != nullptr) {
                desc->qkv_named_out[row] = v;
            }
        }
    }
}

__device__ __forceinline__ void qwen36_rdna3_superlayer_l0_projection_bundle(
        const qwen36_superlayer_l0_proj_desc * desc,
        const uint8_t * weightpack,
        uint8_t * scratch,
        const uint64_t scratch_bytes,
        float * sums,
        const uint32_t write_outputs) {
    if (desc == nullptr || desc->ready == 0 || scratch == nullptr || sums == nullptr) {
        return;
    }

    const bool use_direct_weights = desc->use_direct_weights != 0;
    if (!use_direct_weights && weightpack == nullptr) {
        return;
    }

    const uint32_t n_embd = desc->n_embd;
    const uint32_t total_out = desc->z_out + desc->beta_out + desc->alpha_out;
    if (n_embd == 0 || total_out == 0 ||
            desc->z_scratch_offset + (uint64_t) desc->z_out*sizeof(float) > scratch_bytes ||
            desc->beta_scratch_offset + (uint64_t) desc->beta_out*sizeof(float) > scratch_bytes ||
            desc->alpha_scratch_offset + (uint64_t) desc->alpha_out*sizeof(float) > scratch_bytes) {
        return;
    }

    const float * norm = (const float *) scratch;
    float * z = (float *) (scratch + desc->z_scratch_offset);
    float * beta = (float *) (scratch + desc->beta_scratch_offset);
    float * alpha = (float *) (scratch + desc->alpha_scratch_offset);
    const float * alpha_dt = use_direct_weights ?
        desc->alpha_dt_data : (const float *) (weightpack + desc->alpha_dt_offset);
    const float * alpha_a = use_direct_weights ?
        desc->alpha_a_data : (const float *) (weightpack + desc->alpha_a_offset);
    const float * z_scale = desc->has_z_scale != 0 ?
        (use_direct_weights ? desc->z_scale_data : (const float *) (weightpack + desc->z_scale_offset)) : nullptr;
    const float * beta_scale = desc->has_beta_scale != 0 ?
        (use_direct_weights ? desc->beta_scale_data : (const float *) (weightpack + desc->beta_scale_offset)) : nullptr;
    const float * alpha_scale = desc->has_alpha_scale != 0 ?
        (use_direct_weights ? desc->alpha_scale_data : (const float *) (weightpack + desc->alpha_scale_offset)) : nullptr;
    if ((use_direct_weights &&
         (desc->wz_data == nullptr || desc->wbeta_data == nullptr || desc->walpha_data == nullptr)) ||
            alpha_dt == nullptr || alpha_a == nullptr ||
            (desc->has_z_scale != 0 && z_scale == nullptr) ||
            (desc->has_beta_scale != 0 && beta_scale == nullptr) ||
            (desc->has_alpha_scale != 0 && alpha_scale == nullptr)) {
        return;
    }
    (void) sums;
    const bool write_z = (write_outputs & 0x4u) != 0;
    const bool write_beta = (write_outputs & 0x8u) != 0;
    const bool write_alpha = (write_outputs & 0x10u) != 0;
    const bool z_math_only = (write_outputs & 0x20u) != 0;

    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp = threadIdx.x / WARP_SIZE;
    const int rows_per_block = max(1, blockDim.x / WARP_SIZE);
    for (uint32_t row = blockIdx.x*rows_per_block + warp; row < total_out; row += gridDim.x*rows_per_block) {
        const char * w = nullptr;
        uint64_t w_nb1 = 0;
        ggml_type wtype = GGML_TYPE_COUNT;
        uint32_t local_row = row;
        if (local_row < desc->z_out) {
            if (!write_z) {
                continue;
            }
            w = use_direct_weights ? desc->wz_data : (const char *) (weightpack + desc->wz_offset);
            w_nb1 = desc->wz_nb1;
            wtype = (ggml_type) desc->wz_type;
        } else {
            local_row -= desc->z_out;
            if (local_row < desc->beta_out) {
                if (!write_beta) {
                    continue;
                }
                w = use_direct_weights ? desc->wbeta_data : (const char *) (weightpack + desc->wbeta_offset);
                w_nb1 = desc->wbeta_nb1;
                wtype = (ggml_type) desc->wbeta_type;
            } else {
                local_row -= desc->beta_out;
                if (!write_alpha) {
                    continue;
                }
                w = use_direct_weights ? desc->walpha_data : (const char *) (weightpack + desc->walpha_offset);
                w_nb1 = desc->walpha_nb1;
                wtype = (ggml_type) desc->walpha_type;
            }
        }

        float acc = 0.0f;
        for (uint32_t col = lane; col < n_embd; col += WARP_SIZE) {
            const float ww = qwen36_rdna3_superlayer_dequant_weight(
                    w, wtype, (int64_t) w_nb1, local_row, col);
            acc += norm[col]*ww;
        }

        const float sum = warp_reduce_sum<WARP_SIZE>(acc);
        if (lane == 0) {
            local_row = row;
            if (local_row < desc->z_out) {
                const float scale = z_scale == nullptr ? 1.0f :
                    z_scale[desc->z_scale_n == 1 ? 0 : local_row];
                const float v = sum*scale;
                if (desc->z_math_dst != nullptr) {
                    desc->z_math_dst[local_row] = sum;
                }
                if (!z_math_only) {
                    z[local_row] = v;
                }
                if (!z_math_only && desc->z_dst != nullptr) {
                    desc->z_dst[local_row] = v;
                }
            } else {
                local_row -= desc->z_out;
                if (local_row < desc->beta_out) {
                    const float scale = beta_scale == nullptr ? 1.0f :
                        beta_scale[desc->beta_scale_n == 1 ? 0 : local_row];
                    const float raw = sum*scale;
                    const float v = 1.0f / (1.0f + expf(-raw));
                    beta[local_row] = v;
                    if (desc->beta_math_dst != nullptr) {
                        desc->beta_math_dst[local_row] = sum;
                    }
                    if (desc->beta_raw_dst != nullptr) {
                        desc->beta_raw_dst[local_row] = raw;
                    }
                    if (desc->beta_dst != nullptr) {
                        desc->beta_dst[local_row] = v;
                    }
                } else {
                    local_row -= desc->beta_out;
                    const float scale = alpha_scale == nullptr ? 1.0f :
                        alpha_scale[desc->alpha_scale_n == 1 ? 0 : local_row];
                    const float raw = sum*scale;
                    const float biased = raw + alpha_dt[local_row];
                    const float softplus = biased > 20.0f ? biased : log1pf(expf(biased));
                    const float v = softplus * alpha_a[local_row];
                    alpha[local_row] = v;
                    if (desc->alpha_math_dst != nullptr) {
                        desc->alpha_math_dst[local_row] = sum;
                    }
                    if (desc->alpha_raw_dst != nullptr) {
                        desc->alpha_raw_dst[local_row] = raw;
                    }
                    if (desc->alpha_biased_dst != nullptr) {
                        desc->alpha_biased_dst[local_row] = biased;
                    }
                    if (desc->alpha_softplus_dst != nullptr) {
                        desc->alpha_softplus_dst[local_row] = softplus;
                    }
                    if (desc->alpha_dst != nullptr) {
                        desc->alpha_dst[local_row] = v;
                    }
                }
            }
        }
    }
}

__device__ __forceinline__ void qwen36_rdna3_superlayer_l0_ssm_conv(
        const qwen36_superlayer_l0_ssm_desc * desc,
        const uint8_t * weightpack,
        const uint32_t write_outputs) {
    if (desc == nullptr || desc->ready == 0 || write_outputs == 0 ||
            desc->state == nullptr || desc->token == nullptr ||
            desc->raw_dst == nullptr || desc->silu_dst == nullptr || desc->state_out == nullptr) {
        return;
    }

    const bool use_direct_weights = desc->use_direct_weights != 0;
    if (!use_direct_weights && weightpack == nullptr) {
        return;
    }

    const float * conv_w = use_direct_weights ?
        desc->conv_w_data : (const float *) (weightpack + desc->conv_w_offset);
    if (conv_w == nullptr) {
        return;
    }

    const uint32_t d_conv = desc->d_conv;
    const uint32_t n_channels = desc->n_channels;
    const uint32_t n_seqs = desc->n_seqs;
    if (d_conv <= 1 || d_conv > 16 || n_channels == 0 || n_seqs == 0) {
        return;
    }

    const uint64_t total = (uint64_t) n_channels*n_seqs;
    const uint64_t stride = (uint64_t) gridDim.x*blockDim.x;
    const uint64_t first = (uint64_t) blockIdx.x*blockDim.x + threadIdx.x;
    for (uint64_t idx = first; idx < total; idx += stride) {
        const uint32_t seq = (uint32_t) (idx / n_channels);
        const uint32_t ch = (uint32_t) (idx - (uint64_t) seq*n_channels);

        const float * state_ch = (const float *) ((const char *) desc->state +
                (uint64_t) seq*desc->state_nb2 + (uint64_t) ch*desc->state_nb1);
        const float * token_ch = (const float *) ((const char *) desc->token +
                (uint64_t) seq*desc->token_nb2 + (uint64_t) ch*desc->token_nb1);
        const float * weight_ch = (const float *) ((const char *) conv_w + (uint64_t) ch*desc->conv_w_nb1);

        float sum = 0.0f;
        for (uint32_t j = 0; j + 1 < d_conv; ++j) {
            sum += state_ch[j]*weight_ch[j];
        }
        const float token_v = token_ch[0];
        sum += token_v*weight_ch[d_conv - 1];

        float * raw_ch = (float *) ((char *) desc->raw_dst +
                (uint64_t) seq*desc->raw_nb2 + (uint64_t) ch*desc->raw_nb0);
        float * silu_ch = (float *) ((char *) desc->silu_dst +
                (uint64_t) seq*desc->silu_nb2 + (uint64_t) ch*desc->silu_nb0);
        raw_ch[0] = sum;
        silu_ch[0] = sum / (1.0f + expf(-sum));

        float * state_out_seq = (float *) ((char *) desc->state_out +
                (uint64_t) seq*desc->state_out_nb1);
        float * state_out_ch = state_out_seq + (uint64_t) ch*(d_conv - 1);
        for (uint32_t j = 0; j + 2 < d_conv; ++j) {
            state_out_ch[j] = state_ch[j + 1];
        }
        state_out_ch[d_conv - 2] = token_v;
    }
}

__device__ __forceinline__ void qwen36_rdna3_superlayer_l0_l2_pair(
        const qwen36_superlayer_l0_l2_desc * desc,
        float * sums,
        const uint32_t write_outputs) {
    if (desc == nullptr || desc->ready == 0 || write_outputs == 0 ||
            sums == nullptr || desc->q_src == nullptr || desc->k_src == nullptr ||
            desc->q_dst == nullptr || desc->k_dst == nullptr) {
        return;
    }

    const uint32_t ncols = desc->ncols;
    const uint32_t nrows = desc->nrows;
    const uint32_t nchannels = desc->nchannels;
    const uint32_t nsamples = desc->nsamples;
    if (ncols == 0 || nrows == 0 || nchannels == 0 || nsamples == 0) {
        return;
    }

    const uint64_t rows_per_tensor = (uint64_t) nrows*nchannels*nsamples;
    const uint64_t total_rows = rows_per_tensor*2;
    for (uint64_t row2 = blockIdx.x; row2 < total_rows; row2 += gridDim.x) {
        const bool is_k = row2 >= rows_per_tensor;
        const uint64_t local = is_k ? row2 - rows_per_tensor : row2;
        const uint32_t row = (uint32_t) (local % nrows);
        const uint32_t channel = (uint32_t) ((local / nrows) % nchannels);
        const uint32_t sample = (uint32_t) (local / ((uint64_t) nrows*nchannels));

        const uint64_t src_off = is_k ?
            (uint64_t) row*desc->k_src_nb1 + (uint64_t) channel*desc->k_src_nb2 +
                (uint64_t) sample*desc->k_src_nb3 :
            (uint64_t) row*desc->q_src_nb1 + (uint64_t) channel*desc->q_src_nb2 +
                (uint64_t) sample*desc->q_src_nb3;
        const uint64_t dst_off = is_k ?
            (uint64_t) row*desc->k_dst_nb1 + (uint64_t) channel*desc->k_dst_nb2 +
                (uint64_t) sample*desc->k_dst_nb3 :
            (uint64_t) row*desc->q_dst_nb1 + (uint64_t) channel*desc->q_dst_nb2 +
                (uint64_t) sample*desc->q_dst_nb3;
        const float * src = (const float *) ((const char *) (is_k ? desc->k_src : desc->q_src) + src_off);
        float * dst = (float *) ((char *) (is_k ? desc->k_dst : desc->q_dst) + dst_off);

        float acc = 0.0f;
        for (uint32_t col = threadIdx.x; col < ncols; col += blockDim.x) {
            const float v = src[col];
            acc += v*v;
        }

        sums[threadIdx.x] = acc;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                sums[threadIdx.x] += sums[threadIdx.x + stride];
            }
            __syncthreads();
        }

        const float scale = rsqrtf(fmaxf(sums[0], desc->eps*desc->eps));
        for (uint32_t col = threadIdx.x; col < ncols; col += blockDim.x) {
            dst[col] = src[col]*scale;
        }
        __syncthreads();
    }
}

__device__ __forceinline__ void qwen36_rdna3_superlayer_l0_gdn(
        const qwen36_superlayer_l0_gdn_desc * desc,
        float * sums,
        const uint32_t write_outputs) {
    if (desc == nullptr || desc->ready == 0 || write_outputs == 0 || sums == nullptr ||
            desc->q == nullptr || desc->k == nullptr || desc->v == nullptr ||
            desc->g == nullptr || desc->beta == nullptr || desc->state == nullptr ||
            desc->dst == nullptr || desc->state_out == nullptr) {
        return;
    }

    const uint32_t s_v = desc->s_v;
    const uint32_t h_v = desc->h_v;
    const uint32_t h_k = desc->h_k;
    const uint32_t n_seqs = desc->n_seqs;
    if (s_v == 0 || h_v == 0 || h_k == 0 || n_seqs == 0 || h_v % h_k != 0) {
        return;
    }

    const float scale = rsqrtf((float) s_v);
    const uint64_t total_cols = (uint64_t) n_seqs*h_v*s_v;
    for (uint64_t idx = blockIdx.x; idx < total_cols; idx += gridDim.x) {
        const uint32_t col = (uint32_t) (idx % s_v);
        const uint32_t h = (uint32_t) ((idx / s_v) % h_v);
        const uint32_t seq = (uint32_t) (idx / ((uint64_t) s_v*h_v));
        const uint32_t hq = h % h_k;

        const float * q_h = (const float *) ((const char *) desc->q +
                (uint64_t) hq*desc->q_nb1 + (uint64_t) seq*desc->q_nb3);
        const float * k_h = (const float *) ((const char *) desc->k +
                (uint64_t) hq*desc->k_nb1 + (uint64_t) seq*desc->k_nb3);
        const float * v_h = (const float *) ((const char *) desc->v +
                (uint64_t) h*desc->v_nb1 + (uint64_t) seq*desc->v_nb3);
        const float * g_h = (const float *) ((const char *) desc->g +
                (uint64_t) h*desc->g_nb1 + (uint64_t) seq*desc->g_nb3);
        const float * beta_h = (const float *) ((const char *) desc->beta +
                (uint64_t) h*desc->beta_nb1 + (uint64_t) seq*desc->beta_nb3);
        const float * state_h = (const float *) ((const char *) desc->state +
                (uint64_t) h*desc->state_nb2 + (uint64_t) seq*desc->state_nb3);
        float * state_out_h = desc->state_out + ((uint64_t) seq*h_v + h)*(uint64_t) s_v*s_v;
        const float * state_col = state_h + (uint64_t) col*s_v;
        float * state_out_col = state_out_h + (uint64_t) col*s_v;

        const bool kda = desc->kda != 0;
        float kv_part = 0.0f;
        for (uint32_t i = threadIdx.x; i < s_v; i += blockDim.x) {
            const float g_val = kda ? expf(g_h[i]) : 1.0f;
            kv_part += g_val*state_col[i]*k_h[i];
        }
        sums[threadIdx.x] = kv_part;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                sums[threadIdx.x] += sums[threadIdx.x + stride];
            }
            __syncthreads();
        }

        const float g_scalar = kda ? 1.0f : expf(g_h[0]);
        const float kv = kda ? sums[0] : g_scalar*sums[0];
        const float delta = (v_h[col] - kv)*beta_h[0];

        float attn_part = 0.0f;
        for (uint32_t i = threadIdx.x; i < s_v; i += blockDim.x) {
            const float g_val = kda ? expf(g_h[i]) : g_scalar;
            const float s_new = g_val*state_col[i] + k_h[i]*delta;
            state_out_col[i] = s_new;
            attn_part += s_new*q_h[i];
        }
        sums[threadIdx.x] = attn_part;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                sums[threadIdx.x] += sums[threadIdx.x + stride];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            desc->dst[((uint64_t) seq*h_v + h)*(uint64_t) s_v + col] = sums[0]*scale;
        }
        __syncthreads();
    }
}

__device__ __forceinline__ void qwen36_rdna3_superlayer_l0_gated_norm(
        const qwen36_superlayer_l0_gated_norm_desc * desc,
        const uint8_t * weightpack,
        float * sums,
        const uint32_t write_outputs) {
    if (desc == nullptr || desc->ready == 0 || write_outputs == 0 || sums == nullptr ||
            desc->x == nullptr || desc->gate == nullptr || desc->rms_dst == nullptr ||
            desc->norm_dst == nullptr || desc->silu_dst == nullptr || desc->out_dst == nullptr) {
        return;
    }

    const uint32_t ncols = desc->ncols;
    const uint32_t nrows = desc->nrows;
    if (ncols == 0 || nrows == 0) {
        return;
    }

    const bool has_norm_w = desc->norm_w_data != nullptr;
    const bool use_direct_weights = desc->use_direct_weights != 0;
    if (has_norm_w && !use_direct_weights && weightpack == nullptr) {
        return;
    }
    const float * norm_w = has_norm_w ?
        (use_direct_weights ? desc->norm_w_data : (const float *) (weightpack + desc->norm_w_offset)) : nullptr;
    if (has_norm_w && norm_w == nullptr) {
        return;
    }

    for (uint32_t row = blockIdx.x; row < nrows; row += gridDim.x) {
        const uint64_t row_off = (uint64_t) row*ncols;

        float acc = 0.0f;
        for (uint32_t col = threadIdx.x; col < ncols; col += blockDim.x) {
            const float x = desc->x[row_off + col];
            acc += x*x;
        }

        sums[threadIdx.x] = acc;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                sums[threadIdx.x] += sums[threadIdx.x + stride];
            }
            __syncthreads();
        }

        const float inv_rms = rsqrtf(sums[0]/(float) ncols + desc->eps);
        for (uint32_t col = threadIdx.x; col < ncols; col += blockDim.x) {
            const uint64_t idx = row_off + col;
            const float rms = desc->x[idx]*inv_rms;
            const float norm = rms*(norm_w != nullptr ? norm_w[col] : 1.0f);
            const float gate = desc->gate[idx];
            const float silu = gate/(1.0f + expf(-gate));
            const float out = norm*silu;
            desc->rms_dst[idx] = rms;
            desc->norm_dst[idx] = norm;
            desc->silu_dst[idx] = silu;
            desc->out_dst[idx] = out;
            if (desc->final_dst != nullptr) {
                desc->final_dst[idx] = out;
            }
        }
        __syncthreads();
    }
}

__device__ __forceinline__ void qwen36_rdna3_superlayer_l0_attn_out(
        const qwen36_superlayer_l0_attn_out_desc * desc,
        const uint8_t * weightpack,
        const uint32_t write_outputs) {
    if (desc == nullptr || desc->ready == 0 || write_outputs == 0 ||
            desc->x == nullptr || desc->out_dst == nullptr) {
        return;
    }

    const bool use_direct_weights = desc->use_direct_weights != 0;
    if (!use_direct_weights && weightpack == nullptr) {
        return;
    }
    if (use_direct_weights && desc->w_data == nullptr) {
        return;
    }

    const uint32_t n_embd = desc->n_embd;
    const uint32_t n_out = desc->n_out;
    if (n_embd == 0 || n_out == 0) {
        return;
    }

    const char * w = use_direct_weights ? desc->w_data : (const char *) (weightpack + desc->w_offset);
    const float * scale = desc->has_scale != 0 ?
        (use_direct_weights ? desc->scale_data : (const float *) (weightpack + desc->scale_offset)) : nullptr;
    if (w == nullptr || (desc->has_scale != 0 && scale == nullptr)) {
        return;
    }

    const ggml_type wtype = (ggml_type) desc->w_type;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp = threadIdx.x / WARP_SIZE;
    const int rows_per_block = max(1, blockDim.x / WARP_SIZE);
    for (uint32_t row = blockIdx.x*rows_per_block + warp; row < n_out; row += gridDim.x*rows_per_block) {
        float acc = 0.0f;
        for (uint32_t col = lane; col < n_embd; col += WARP_SIZE) {
            const float ww = qwen36_rdna3_superlayer_dequant_weight(
                    w, wtype, (int64_t) desc->w_nb1, row, col);
            acc += desc->x[col]*ww;
        }

        const float raw = warp_reduce_sum<WARP_SIZE>(acc);
        if (lane == 0) {
            if (desc->math_dst != nullptr) {
                desc->math_dst[row] = raw;
            }
            const float scale_v = scale == nullptr ? 1.0f :
                scale[desc->scale_n == 1 ? 0 : row];
            const float v = raw*scale_v;
            desc->out_dst[row] = v;
            if (desc->named_dst != nullptr) {
                desc->named_dst[row] = v;
            }
        }
    }
}

__device__ __forceinline__ void qwen36_rdna3_superlayer_l0_post_attn(
        const qwen36_superlayer_l0_post_attn_desc * desc,
        const uint8_t * weightpack,
        float * sums,
        const uint32_t write_outputs) {
    if (desc == nullptr || desc->ready == 0 || write_outputs == 0 || sums == nullptr ||
            desc->attn == nullptr || desc->skip == nullptr || desc->residual_dst == nullptr ||
            desc->rms_dst == nullptr || desc->norm_dst == nullptr) {
        return;
    }

    const uint32_t ncols = desc->ncols;
    const uint32_t nrows = desc->nrows;
    if (ncols == 0 || nrows == 0) {
        return;
    }

    const bool use_direct_weights = desc->use_direct_weights != 0;
    if (desc->has_norm_w != 0 && !use_direct_weights && weightpack == nullptr) {
        return;
    }
    if (desc->has_norm_w != 0 && use_direct_weights && desc->norm_w_data == nullptr) {
        return;
    }
    const float * norm_w = desc->has_norm_w != 0 ?
        (use_direct_weights ? desc->norm_w_data : (const float *) (weightpack + desc->norm_w_offset)) : nullptr;

    for (uint32_t row = blockIdx.x; row < nrows; row += gridDim.x) {
        const uint64_t row_off = (uint64_t) row*ncols;

        float acc = 0.0f;
        for (uint32_t col = threadIdx.x; col < ncols; col += blockDim.x) {
            const uint64_t idx = row_off + col;
            const float residual = desc->attn[idx] + desc->skip[idx];
            desc->residual_dst[idx] = residual;
            if (desc->residual_named_dst != nullptr) {
                desc->residual_named_dst[idx] = residual;
            }
            acc += residual*residual;
        }

        sums[threadIdx.x] = acc;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                sums[threadIdx.x] += sums[threadIdx.x + stride];
            }
            __syncthreads();
        }

        const float inv_rms = rsqrtf(sums[0]/(float) ncols + desc->eps);
        for (uint32_t col = threadIdx.x; col < ncols; col += blockDim.x) {
            const uint64_t idx = row_off + col;
            const float rms = desc->residual_dst[idx]*inv_rms;
            const float norm = rms*(norm_w != nullptr ? norm_w[col] : 1.0f);
            desc->rms_dst[idx] = rms;
            desc->norm_dst[idx] = norm;
            if (desc->named_dst != nullptr) {
                desc->named_dst[idx] = norm;
            }
        }
        __syncthreads();
    }
}

__device__ __forceinline__ void qwen36_rdna3_superlayer_l0_moe_router(
        const qwen36_superlayer_l0_moe_router_desc * desc,
        const uint8_t * weightpack,
        float * sums,
        const uint32_t write_outputs) {
    if (blockIdx.x != 0) {
        return;
    }
    if (desc == nullptr || desc->ready == 0 || write_outputs == 0 || sums == nullptr ||
            desc->x == nullptr || desc->logits_dst == nullptr || desc->probs_dst == nullptr ||
            desc->argsort_dst == nullptr || desc->topk_dst == nullptr || desc->weights_dst == nullptr) {
        return;
    }

    const bool use_direct_weights = desc->use_direct_weights != 0;
    if (!use_direct_weights && weightpack == nullptr) {
        return;
    }
    if (use_direct_weights && desc->w_data == nullptr) {
        return;
    }

    const uint32_t n_embd = desc->n_embd;
    const uint32_t n_expert = desc->n_expert;
    const uint32_t n_expert_used = desc->n_expert_used;
    if (n_embd == 0 || n_expert == 0 || n_expert_used == 0 || n_expert_used > n_expert ||
            n_expert_used > 64) {
        return;
    }

    const char * w = use_direct_weights ? desc->w_data : (const char *) (weightpack + desc->w_offset);
    const float * scale = desc->has_scale != 0 ?
        (use_direct_weights ? desc->scale_data : (const float *) (weightpack + desc->scale_offset)) : nullptr;
    if (w == nullptr || (desc->has_scale != 0 && scale == nullptr)) {
        return;
    }

    const ggml_type wtype = (ggml_type) desc->w_type;
    float local_max = -FLT_MAX;
    for (uint32_t expert = threadIdx.x; expert < n_expert; expert += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t col = 0; col < n_embd; ++col) {
            const float ww = qwen36_rdna3_superlayer_dequant_weight(
                    w, wtype, (int64_t) desc->w_nb1, expert, col);
            acc += desc->x[col]*ww;
        }
        if (__isnanf(acc)) {
            acc = -FLT_MAX;
        }
        if (desc->logits_math_dst != nullptr) {
            desc->logits_math_dst[expert] = acc;
        }
        const float scale_v = scale == nullptr ? 1.0f : scale[desc->scale_n == 1 ? 0 : expert];
        float v = acc*scale_v;
        if (__isnanf(v)) {
            v = -FLT_MAX;
        }
        desc->logits_dst[expert] = v;
        if (desc->logits_named_dst != nullptr) {
            desc->logits_named_dst[expert] = v;
        }
        local_max = fmaxf(local_max, v);
    }

    sums[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sums[threadIdx.x] = fmaxf(sums[threadIdx.x], sums[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    const float max_logit = sums[0];
    float local_sum = 0.0f;
    for (uint32_t expert = threadIdx.x; expert < n_expert; expert += blockDim.x) {
        const float p = expf(desc->logits_dst[expert] - max_logit);
        desc->probs_dst[expert] = p;
        local_sum += p;
    }

    sums[threadIdx.x] = local_sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sums[threadIdx.x] += sums[threadIdx.x + stride];
        }
        __syncthreads();
    }

    const float inv_sum = sums[0] > 0.0f ? 1.0f/sums[0] : 0.0f;
    for (uint32_t expert = threadIdx.x; expert < n_expert; expert += blockDim.x) {
        desc->probs_dst[expert] *= inv_sum;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int32_t selected_ids[64];
        float selected_weights[64];
        for (uint32_t k = 0; k < n_expert_used; ++k) {
            int32_t best_id = -1;
            float best_val = -FLT_MAX;
            for (uint32_t expert = 0; expert < n_expert; ++expert) {
                bool used = false;
                for (uint32_t prev = 0; prev < k; ++prev) {
                    used = used || selected_ids[prev] == (int32_t) expert;
                }
                if (used) {
                    continue;
                }
                const float v = desc->logits_dst[expert];
                if (best_id < 0 || v > best_val || (v == best_val && expert < (uint32_t) best_id)) {
                    best_val = v;
                    best_id = (int32_t) expert;
                }
            }
            selected_ids[k] = best_id;
            const float weight = best_id >= 0 ? desc->probs_dst[best_id] : 0.0f;
            selected_weights[k] = weight;
            desc->argsort_dst[k] = best_id;
            desc->topk_dst[k] = best_id;
            desc->weights_dst[k] = weight;
        }

        float weight_sum = 0.0f;
        for (uint32_t k = 0; k < n_expert_used; ++k) {
            weight_sum += selected_weights[k];
        }
        if (desc->weights_sum_dst != nullptr) {
            desc->weights_sum_dst[0] = weight_sum;
        }

        const float clamped_sum = fmaxf(weight_sum, desc->clamp_min);
        if (desc->weights_sum_clamped_dst != nullptr) {
            desc->weights_sum_clamped_dst[0] = clamped_sum;
        }

        for (uint32_t k = 0; k < n_expert_used; ++k) {
            const float norm = desc->has_weights_norm != 0 && clamped_sum > 0.0f ?
                selected_weights[k]/clamped_sum : selected_weights[k];
            if (desc->weights_norm_dst != nullptr) {
                desc->weights_norm_dst[k] = norm;
            }
            if (desc->weights_scaled_dst != nullptr) {
                desc->weights_scaled_dst[k] = norm*desc->weights_scale;
            }
        }
    }
}

__device__ __forceinline__ void qwen36_rdna3_superlayer_l0_moe_gate_up(
        const qwen36_superlayer_l0_moe_gate_up_desc * desc,
        const uint8_t * weightpack,
        const uint32_t write_outputs) {
    if (desc == nullptr || desc->ready == 0 || write_outputs == 0 ||
            desc->x == nullptr || desc->ids == nullptr || desc->swiglu_dst == nullptr) {
        return;
    }

    const bool use_direct_weights = desc->use_direct_weights != 0;
    if (!use_direct_weights && weightpack == nullptr) {
        return;
    }
    if (use_direct_weights && desc->w_data == nullptr) {
        return;
    }

    const uint32_t n_embd = desc->n_embd;
    const uint32_t n_ff = desc->n_ff;
    const uint32_t n_expert = desc->n_expert;
    const uint32_t n_expert_used = desc->n_expert_used;
    if (n_embd == 0 || n_ff == 0 || n_expert == 0 || n_expert_used == 0) {
        return;
    }

    const char * w = use_direct_weights ? desc->w_data : (const char *) (weightpack + desc->w_offset);
    if (w == nullptr) {
        return;
    }

    const ggml_type wtype = (ggml_type) desc->w_type;
    const uint64_t total = (uint64_t) n_ff*n_expert_used;
    const uint64_t stride = (uint64_t) blockDim.x*gridDim.x;
    for (uint64_t idx = (uint64_t) blockIdx.x*blockDim.x + threadIdx.x; idx < total; idx += stride) {
        const uint32_t expert_slot = (uint32_t) (idx / n_ff);
        const uint32_t row = (uint32_t) (idx - (uint64_t) expert_slot*n_ff);
        const int32_t expert = desc->ids[expert_slot];
        if (expert < 0 || (uint32_t) expert >= n_expert) {
            continue;
        }

        const char * w_expert = w + (uint64_t) expert*desc->w_nb2;
        float gate_acc = 0.0f;
        float up_acc = 0.0f;
        for (uint32_t col = 0; col < n_embd; ++col) {
            const float x = desc->x[col];
            gate_acc += x*qwen36_rdna3_superlayer_dequant_weight(
                    w_expert, wtype, (int64_t) desc->w_nb1, row, col);
            up_acc += x*qwen36_rdna3_superlayer_dequant_weight(
                    w_expert, wtype, (int64_t) desc->w_nb1, row + n_ff, col);
        }

        if (__isnanf(gate_acc)) {
            gate_acc = 0.0f;
        }
        if (__isnanf(up_acc)) {
            up_acc = 0.0f;
        }

        const float silu = gate_acc/(1.0f + expf(-gate_acc));
        const float out = silu*up_acc;

        if (desc->gate_up_dst != nullptr) {
            char * gate_up_base = (char *) desc->gate_up_dst + (uint64_t) expert_slot*desc->gate_up_nb1;
            ((float *) gate_up_base)[row] = gate_acc;
            ((float *) gate_up_base)[row + n_ff] = up_acc;
        }
        if (desc->gate_dst != nullptr) {
            char * gate_base = (char *) desc->gate_dst + (uint64_t) expert_slot*desc->gate_nb1;
            ((float *) gate_base)[row] = gate_acc;
        }
        if (desc->up_dst != nullptr) {
            char * up_base = (char *) desc->up_dst + (uint64_t) expert_slot*desc->up_nb1;
            ((float *) up_base)[row] = up_acc;
        }
        char * swiglu_base = (char *) desc->swiglu_dst + (uint64_t) expert_slot*desc->swiglu_nb1;
        ((float *) swiglu_base)[row] = out;
    }
}

__device__ __forceinline__ void qwen36_rdna3_superlayer_l0_moe_down(
        const qwen36_superlayer_l0_moe_down_desc * desc,
        const uint8_t * weightpack,
        const uint32_t write_outputs) {
    if (desc == nullptr || desc->ready == 0 || write_outputs == 0 ||
            desc->x == nullptr || desc->ids == nullptr || desc->weights == nullptr ||
            desc->down_dst == nullptr || desc->weighted_dst == nullptr || desc->out_dst == nullptr) {
        return;
    }

    const bool use_direct_weights = desc->use_direct_weights != 0;
    if (!use_direct_weights && weightpack == nullptr) {
        return;
    }
    if (use_direct_weights && desc->w_data == nullptr) {
        return;
    }

    const uint32_t n_ff = desc->n_ff;
    const uint32_t n_embd = desc->n_embd;
    const uint32_t n_expert = desc->n_expert;
    const uint32_t n_expert_used = desc->n_expert_used;
    if (n_ff == 0 || n_embd == 0 || n_expert == 0 || n_expert_used == 0) {
        return;
    }

    const char * w = use_direct_weights ? desc->w_data : (const char *) (weightpack + desc->w_offset);
    if (w == nullptr) {
        return;
    }

    const ggml_type wtype = (ggml_type) desc->w_type;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp = threadIdx.x / WARP_SIZE;
    const int rows_per_block = max(1, blockDim.x / WARP_SIZE);
    for (uint32_t row = blockIdx.x*rows_per_block + warp; row < n_embd; row += gridDim.x*rows_per_block) {
        float final_acc = 0.0f;

        for (uint32_t expert_slot = 0; expert_slot < n_expert_used; ++expert_slot) {
            const int32_t expert = desc->ids[expert_slot];
            if (expert < 0 || (uint32_t) expert >= n_expert) {
                continue;
            }

            const char * w_expert = w + (uint64_t) expert*desc->w_nb2;
            const char * x_expert = (const char *) desc->x + (uint64_t) expert_slot*desc->x_nb1;
            float acc = 0.0f;
            for (uint32_t col = lane; col < n_ff; col += WARP_SIZE) {
                const float x = ((const float *) x_expert)[col];
                const float ww = qwen36_rdna3_superlayer_dequant_weight(
                        w_expert, wtype, (int64_t) desc->w_nb1, row, col);
                acc += x*ww;
            }

            const float raw0 = warp_reduce_sum<WARP_SIZE>(acc);
            if (lane == 0) {
                const float raw = __isnanf(raw0) ? 0.0f : raw0;
                const float weight =
                    *((const float *) ((const char *) desc->weights + (uint64_t) expert_slot*desc->weights_nb1));
                const float weighted = raw*weight;

                char * down_base = (char *) desc->down_dst + (uint64_t) expert_slot*desc->down_nb1;
                ((float *) down_base)[row] = raw;
                char * weighted_base = (char *) desc->weighted_dst + (uint64_t) expert_slot*desc->weighted_nb1;
                ((float *) weighted_base)[row] = weighted;
                final_acc += weighted;
            }
        }

        if (lane == 0) {
            desc->out_dst[row] = final_acc;
        }
    }
}

__global__ void qwen36_rdna3_superlayer_contract_kernel(
        qwen36_superlayer_contract_state * state,
        const uint8_t * weightpack,
        const qwen36_superlayer_layer_pack_desc * layer_descs,
        const qwen36_superlayer_runtime_tensor_desc * io_descs,
        const uint32_t io_count,
        const qwen36_superlayer_l0_norm_desc * l0_norm,
        const qwen36_superlayer_l0_qkv_desc * l0_qkv,
        const qwen36_superlayer_l0_proj_desc * l0_proj,
        const qwen36_superlayer_l0_ssm_desc * l0_ssm,
        const qwen36_superlayer_l0_l2_desc * l0_l2,
        const qwen36_superlayer_l0_gdn_desc * l0_gdn,
        const qwen36_superlayer_l0_gated_norm_desc * l0_gated_norm,
        const qwen36_superlayer_l0_attn_out_desc * l0_attn_out,
        const qwen36_superlayer_l0_post_attn_desc * l0_post_attn,
        const qwen36_superlayer_l0_moe_router_desc * l0_moe_router,
        const qwen36_superlayer_l0_moe_gate_up_desc * l0_moe_gate_up,
        const qwen36_superlayer_l0_moe_down_desc * l0_moe_down,
        uint8_t * scratch,
        const uint64_t scratch_bytes,
        const uint64_t weightpack_bytes,
        const uint32_t l0_stage_mask) {
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();
    __shared__ float l0_rms_sums[256];

    const uint64_t lane = (uint64_t) blockIdx.x*blockDim.x + threadIdx.x;
    uint64_t checksum = 0x7900011000360001ull ^ lane ^ weightpack_bytes ^ scratch_bytes ^ io_count;

#define QWEN36_SUPERLAYER_CONTRACT_LAYER(L) \
    checksum ^= qwen36_rdna3_superlayer_contract_layer<L>(weightpack, layer_descs, lane)

    QWEN36_SUPERLAYER_CONTRACT_LAYER(0);
    QWEN36_SUPERLAYER_CONTRACT_LAYER(1);
    QWEN36_SUPERLAYER_CONTRACT_LAYER(2);
    QWEN36_SUPERLAYER_CONTRACT_LAYER(3);
    QWEN36_SUPERLAYER_CONTRACT_LAYER(4);
    QWEN36_SUPERLAYER_CONTRACT_LAYER(5);
    QWEN36_SUPERLAYER_CONTRACT_LAYER(6);
    QWEN36_SUPERLAYER_CONTRACT_LAYER(7);
    QWEN36_SUPERLAYER_CONTRACT_LAYER(8);
    QWEN36_SUPERLAYER_CONTRACT_LAYER(9);
    QWEN36_SUPERLAYER_CONTRACT_LAYER(10);
    QWEN36_SUPERLAYER_CONTRACT_LAYER(11);
    QWEN36_SUPERLAYER_CONTRACT_LAYER(12);
    QWEN36_SUPERLAYER_CONTRACT_LAYER(13);
    QWEN36_SUPERLAYER_CONTRACT_LAYER(14);
    QWEN36_SUPERLAYER_CONTRACT_LAYER(15);
    QWEN36_SUPERLAYER_CONTRACT_LAYER(16);
    QWEN36_SUPERLAYER_CONTRACT_LAYER(17);
    QWEN36_SUPERLAYER_CONTRACT_LAYER(18);
    QWEN36_SUPERLAYER_CONTRACT_LAYER(19);
    QWEN36_SUPERLAYER_CONTRACT_LAYER(20);
    QWEN36_SUPERLAYER_CONTRACT_LAYER(21);
    QWEN36_SUPERLAYER_CONTRACT_LAYER(22);
    QWEN36_SUPERLAYER_CONTRACT_LAYER(23);
    QWEN36_SUPERLAYER_CONTRACT_LAYER(24);
    QWEN36_SUPERLAYER_CONTRACT_LAYER(25);
    QWEN36_SUPERLAYER_CONTRACT_LAYER(26);
    QWEN36_SUPERLAYER_CONTRACT_LAYER(27);
    QWEN36_SUPERLAYER_CONTRACT_LAYER(28);
    QWEN36_SUPERLAYER_CONTRACT_LAYER(29);
    QWEN36_SUPERLAYER_CONTRACT_LAYER(30);
    QWEN36_SUPERLAYER_CONTRACT_LAYER(31);
    QWEN36_SUPERLAYER_CONTRACT_LAYER(32);
    QWEN36_SUPERLAYER_CONTRACT_LAYER(33);
    QWEN36_SUPERLAYER_CONTRACT_LAYER(34);
    QWEN36_SUPERLAYER_CONTRACT_LAYER(35);
    QWEN36_SUPERLAYER_CONTRACT_LAYER(36);
    QWEN36_SUPERLAYER_CONTRACT_LAYER(37);
    QWEN36_SUPERLAYER_CONTRACT_LAYER(38);
    QWEN36_SUPERLAYER_CONTRACT_LAYER(39);

#undef QWEN36_SUPERLAYER_CONTRACT_LAYER

    if (io_descs != nullptr && io_count > 0) {
        const qwen36_superlayer_runtime_tensor_desc io = io_descs[lane % io_count];
        checksum ^= io.role_hash ^ io.name_hash ^ io.nbytes;
        if (io.data != nullptr && io.nbytes > 0) {
            const uint8_t * io_data = (const uint8_t *) io.data;
            checksum ^= (uint64_t) io_data[lane % io.nbytes] << 32;
        }
    }

    float * l0_grid_scratch = nullptr;
    if (scratch != nullptr && scratch_bytes >= (uint64_t) (gridDim.x + 1)*sizeof(float)) {
        l0_grid_scratch = (float *) (scratch + scratch_bytes) - ((uint64_t) gridDim.x + 1);
    }

    const uint64_t l0_norm_requested =
        l0_norm != nullptr && l0_norm->ready != 0 ? (uint64_t) l0_norm->n_embd*sizeof(float) : 0;
    const bool l0_norm_ready =
        l0_stage_mask != 0 && l0_norm_requested != 0 &&
        scratch != nullptr && scratch_bytes >= l0_norm_requested &&
        l0_grid_scratch != nullptr &&
        (l0_norm->has_norm_w == 0 || l0_norm->use_direct_weights != 0 || weightpack != nullptr);
    const uint64_t l0_qkv_requested =
        l0_qkv != nullptr && l0_qkv->ready != 0 ? (uint64_t) l0_qkv->n_out*sizeof(float) : 0;
    const uint64_t l0_qkv_begin = l0_qkv != nullptr ? l0_qkv->qkv_scratch_offset : 0;
    const uint64_t l0_qkv_end = l0_qkv_begin + l0_qkv_requested;
    const bool l0_qkv_ready =
        (l0_stage_mask & 0x2u) != 0 && l0_qkv_requested != 0 &&
        scratch != nullptr && (l0_qkv->use_direct_weights != 0 || weightpack != nullptr) &&
        l0_qkv_end <= scratch_bytes;
    const uint64_t l0_proj_z_end =
        l0_proj != nullptr ? l0_proj->z_scratch_offset + (uint64_t) l0_proj->z_out*sizeof(float) : 0;
    const uint64_t l0_proj_beta_end =
        l0_proj != nullptr ? l0_proj->beta_scratch_offset + (uint64_t) l0_proj->beta_out*sizeof(float) : 0;
    const uint64_t l0_proj_alpha_end =
        l0_proj != nullptr ? l0_proj->alpha_scratch_offset + (uint64_t) l0_proj->alpha_out*sizeof(float) : 0;
    const uint32_t l0_proj_stage_mask = l0_stage_mask & 0x3cu;
    const bool l0_proj_ready =
        l0_proj_stage_mask != 0 &&
        l0_proj != nullptr && l0_proj->ready != 0 && scratch != nullptr &&
        (l0_proj->use_direct_weights != 0 || weightpack != nullptr) &&
        l0_proj->z_out != 0 && l0_proj->beta_out != 0 && l0_proj->alpha_out != 0 &&
        l0_proj_z_end <= scratch_bytes && l0_proj_beta_end <= scratch_bytes &&
        l0_proj_alpha_end <= scratch_bytes && l0_grid_scratch != nullptr;
    const bool l0_ssm_ready =
        (l0_stage_mask & 0x40u) != 0 &&
        l0_ssm != nullptr && l0_ssm->ready != 0 &&
        (l0_ssm->use_direct_weights != 0 || weightpack != nullptr);
    const bool l0_l2_ready =
        (l0_stage_mask & 0x80u) != 0 &&
        l0_l2 != nullptr && l0_l2->ready != 0;
    const bool l0_gdn_ready =
        (l0_stage_mask & 0x100u) != 0 &&
        l0_gdn != nullptr && l0_gdn->ready != 0;
    const bool l0_gated_norm_ready =
        (l0_stage_mask & 0x200u) != 0 &&
        l0_gated_norm != nullptr && l0_gated_norm->ready != 0 &&
        (l0_gated_norm->norm_w_data == nullptr ||
         l0_gated_norm->use_direct_weights != 0 ||
         weightpack != nullptr);
    const bool l0_attn_out_ready =
        (l0_stage_mask & 0x400u) != 0 &&
        l0_attn_out != nullptr && l0_attn_out->ready != 0 &&
        (l0_attn_out->use_direct_weights != 0 || weightpack != nullptr) &&
        (l0_attn_out->has_scale == 0 ||
         l0_attn_out->use_direct_weights != 0 ||
         weightpack != nullptr);
    const bool l0_post_attn_ready =
        (l0_stage_mask & 0x800u) != 0 &&
        l0_post_attn != nullptr && l0_post_attn->ready != 0 &&
        (l0_post_attn->has_norm_w == 0 ||
         l0_post_attn->use_direct_weights != 0 ||
         weightpack != nullptr);
    const bool l0_moe_router_ready =
        (l0_stage_mask & 0x1000u) != 0 &&
        l0_moe_router != nullptr && l0_moe_router->ready != 0 &&
        (l0_moe_router->use_direct_weights != 0 || weightpack != nullptr) &&
        (l0_moe_router->has_scale == 0 ||
         l0_moe_router->use_direct_weights != 0 ||
         weightpack != nullptr);
    const bool l0_moe_gate_up_ready =
        (l0_stage_mask & 0x2000u) != 0 &&
        l0_moe_gate_up != nullptr && l0_moe_gate_up->ready != 0 &&
        (l0_moe_gate_up->use_direct_weights != 0 || weightpack != nullptr);
    const bool l0_moe_down_ready =
        (l0_stage_mask & 0x4000u) != 0 &&
        l0_moe_down != nullptr && l0_moe_down->ready != 0 &&
        (l0_moe_down->use_direct_weights != 0 || weightpack != nullptr);

    if (l0_norm_ready) {
        qwen36_rdna3_superlayer_l0_rms_norm(
                l0_norm, weightpack, scratch, scratch_bytes, grid, l0_grid_scratch, l0_rms_sums,
                l0_stage_mask & 0x1u);
    }
    if (l0_norm_ready && l0_qkv_ready) {
        qwen36_rdna3_superlayer_l0_qkv(
                l0_qkv, weightpack, scratch, scratch_bytes, l0_rms_sums, l0_stage_mask & 0x2u);
    }
    if (l0_norm_ready && l0_proj_ready) {
        qwen36_rdna3_superlayer_l0_projection_bundle(
                l0_proj, weightpack, scratch, scratch_bytes, l0_rms_sums, l0_proj_stage_mask);
    }
    if (l0_ssm_ready) {
        qwen36_rdna3_superlayer_l0_ssm_conv(l0_ssm, weightpack, l0_stage_mask & 0x40u);
    }
    if (l0_l2_ready) {
        qwen36_rdna3_superlayer_l0_l2_pair(l0_l2, l0_rms_sums, l0_stage_mask & 0x80u);
    }
    if (l0_gdn_ready) {
        qwen36_rdna3_superlayer_l0_gdn(l0_gdn, l0_rms_sums, l0_stage_mask & 0x100u);
    }
    if (l0_gated_norm_ready) {
        qwen36_rdna3_superlayer_l0_gated_norm(
                l0_gated_norm, weightpack, l0_rms_sums, l0_stage_mask & 0x200u);
    }
    if (l0_attn_out_ready) {
        qwen36_rdna3_superlayer_l0_attn_out(
                l0_attn_out, weightpack, l0_stage_mask & 0x400u);
    }
    if (l0_post_attn_ready) {
        qwen36_rdna3_superlayer_l0_post_attn(
                l0_post_attn, weightpack, l0_rms_sums, l0_stage_mask & 0x800u);
    }
    if (l0_moe_router_ready) {
        qwen36_rdna3_superlayer_l0_moe_router(
                l0_moe_router, weightpack, l0_rms_sums, l0_stage_mask & 0x1000u);
    }
    if (l0_moe_gate_up_ready) {
        qwen36_rdna3_superlayer_l0_moe_gate_up(
                l0_moe_gate_up, weightpack, l0_stage_mask & 0x2000u);
    }
    if (l0_moe_down_ready) {
        qwen36_rdna3_superlayer_l0_moe_down(
                l0_moe_down, weightpack, l0_stage_mask & 0x4000u);
    }

    if (lane == 0) {
        state->fingerprint ^= 0x7900'0110'0036'0001ull;
        state->fingerprint ^= checksum;
        state->checksum = checksum ^
            (l0_norm_ready ? 0x10'0000'0000ull : 0ull) ^
            (l0_qkv_ready ? 0x20'0000'0000ull : 0ull) ^
            (l0_proj_ready ? 0x40'0000'0000ull : 0ull) ^
            (l0_ssm_ready ? 0x80'0000'0000ull : 0ull) ^
            (l0_l2_ready ? 0x100'0000'0000ull : 0ull) ^
            (l0_gdn_ready ? 0x200'0000'0000ull : 0ull) ^
            (l0_gated_norm_ready ? 0x400'0000'0000ull : 0ull) ^
            (l0_attn_out_ready ? 0x800'0000'0000ull : 0ull) ^
            (l0_post_attn_ready ? 0x1000'0000'0000ull : 0ull) ^
            (l0_moe_router_ready ? 0x2000'0000'0000ull : 0ull) ^
            (l0_moe_gate_up_ready ? 0x4000'0000'0000ull : 0ull) ^
            (l0_moe_down_ready ? 0x8000'0000'0000ull : 0ull);
        state->n_blocks = gridDim.x;
        state->touched_layers = 40;
        state->active_lanes = gridDim.x*blockDim.x;
    }
}

static bool qwen36_superlayer_final_requested() {
    return qwen36_superlayer_env_enabled("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_FINAL");
}

static bool qwen36_superlayer_final_numeric_ready() {
    return false;
}

static const char * qwen36_superlayer_final_blocker() {
    return "GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_FINAL requires a real numeric 40-layer "
           "Qwen3.6 RDNA3 megakernel; the current physical superlayer only implements "
           "L0 numeric replacement plus a 40-layer contract scaffold";
}

static bool qwen36_superlayer_final_physical_l0_requested() {
    return qwen36_superlayer_env_enabled("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_FINAL_PHYSICAL_L0");
}

static bool qwen36_superlayer_l0_env_requested() {
    if (qwen36_superlayer_final_requested()) {
        return true;
    }
    if (qwen36_superlayer_final_physical_l0_requested()) {
        return true;
    }
    return qwen36_superlayer_env_enabled("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_RUN_L0") ||
        qwen36_superlayer_env_enabled("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REPLACE_L0") ||
        qwen36_superlayer_env_enabled("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REPLACE_L0_RMS") ||
        qwen36_superlayer_env_enabled("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REPLACE_L0_QKV") ||
        qwen36_superlayer_env_enabled("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REPLACE_L0_PROJ") ||
        qwen36_superlayer_env_enabled("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REPLACE_L0_PROJ_Z") ||
        qwen36_superlayer_env_enabled("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REPLACE_L0_PROJ_Z_MATH_ONLY") ||
        qwen36_superlayer_env_enabled("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REPLACE_L0_PROJ_BETA") ||
        qwen36_superlayer_env_enabled("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REPLACE_L0_PROJ_ALPHA") ||
        qwen36_superlayer_env_enabled("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REPLACE_L0_SSM") ||
        qwen36_superlayer_env_enabled("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REPLACE_L0_L2") ||
        qwen36_superlayer_env_enabled("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REPLACE_L0_GDN") ||
        qwen36_superlayer_env_enabled("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REPLACE_L0_GATED_NORM") ||
        qwen36_superlayer_env_enabled("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REPLACE_L0_ATTN_OUT") ||
        qwen36_superlayer_env_enabled("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REPLACE_L0_POST_ATTN") ||
        qwen36_superlayer_env_enabled("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REPLACE_L0_MOE_ROUTER") ||
        qwen36_superlayer_env_enabled("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REPLACE_L0_MOE_GATE_UP") ||
        qwen36_superlayer_env_enabled("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REPLACE_L0_MOE_DOWN") ||
        qwen36_superlayer_env_enabled("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_SINGLE_L0_DISPATCH");
}

static bool qwen36_superlayer_replace_l0_all_requested() {
    return qwen36_superlayer_replace_l0_all_stages_requested() ||
        qwen36_superlayer_env_i64("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_SINGLE_L0_DISPATCH", 0) != 0;
}

static bool qwen36_superlayer_replace_l0_rms_requested() {
    return qwen36_superlayer_replace_l0_all_requested() ||
        qwen36_superlayer_env_i64("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REPLACE_L0_RMS", 0) != 0;
}

static bool qwen36_superlayer_replace_l0_qkv_requested() {
    return qwen36_superlayer_replace_l0_all_requested() ||
        qwen36_superlayer_env_i64("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REPLACE_L0_QKV", 0) != 0;
}

static bool qwen36_superlayer_replace_l0_proj_requested() {
    return qwen36_superlayer_replace_l0_all_requested() ||
        qwen36_superlayer_env_i64("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REPLACE_L0_PROJ", 0) != 0 ||
        qwen36_superlayer_env_i64("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REPLACE_L0_PROJ_Z", 0) != 0 ||
        qwen36_superlayer_env_i64("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REPLACE_L0_PROJ_Z_MATH_ONLY", 0) != 0 ||
        qwen36_superlayer_env_i64("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REPLACE_L0_PROJ_BETA", 0) != 0 ||
        qwen36_superlayer_env_i64("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REPLACE_L0_PROJ_ALPHA", 0) != 0;
}

static bool qwen36_superlayer_replace_l0_proj_all_requested() {
    return qwen36_superlayer_replace_l0_all_requested() ||
        qwen36_superlayer_env_i64("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REPLACE_L0_PROJ", 0) != 0;
}

static bool qwen36_superlayer_replace_l0_proj_z_requested() {
    return qwen36_superlayer_replace_l0_proj_all_requested() ||
        qwen36_superlayer_env_i64("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REPLACE_L0_PROJ_Z", 0) != 0 ||
        qwen36_superlayer_env_i64("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REPLACE_L0_PROJ_Z_MATH_ONLY", 0) != 0;
}

static bool qwen36_superlayer_replace_l0_proj_z_math_only_requested() {
    return qwen36_superlayer_env_i64("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REPLACE_L0_PROJ_Z_MATH_ONLY", 0) != 0;
}

static bool qwen36_superlayer_replace_l0_proj_beta_requested() {
    return qwen36_superlayer_replace_l0_proj_all_requested() ||
        qwen36_superlayer_env_i64("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REPLACE_L0_PROJ_BETA", 0) != 0;
}

static bool qwen36_superlayer_replace_l0_proj_alpha_requested() {
    return qwen36_superlayer_replace_l0_proj_all_requested() ||
        qwen36_superlayer_env_i64("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REPLACE_L0_PROJ_ALPHA", 0) != 0;
}

static bool qwen36_superlayer_replace_l0_ssm_requested() {
    return qwen36_superlayer_l0_ssm_requested();
}

static bool qwen36_superlayer_replace_l0_l2_requested() {
    return qwen36_superlayer_l0_l2_requested();
}

static bool qwen36_superlayer_replace_l0_gdn_requested() {
    return qwen36_superlayer_l0_gdn_requested();
}

static bool qwen36_superlayer_replace_l0_gated_norm_requested() {
    return qwen36_superlayer_l0_gated_norm_requested();
}

static bool qwen36_superlayer_replace_l0_attn_out_requested() {
    return qwen36_superlayer_l0_attn_out_requested();
}

static bool qwen36_superlayer_replace_l0_post_attn_requested() {
    return qwen36_superlayer_l0_post_attn_requested();
}

static bool qwen36_superlayer_replace_l0_moe_router_requested() {
    return qwen36_superlayer_l0_moe_router_requested();
}

static bool qwen36_superlayer_replace_l0_moe_gate_up_requested() {
    return qwen36_superlayer_l0_moe_gate_up_requested();
}

static bool qwen36_superlayer_replace_l0_moe_down_requested() {
    return qwen36_superlayer_l0_moe_down_requested();
}

static bool qwen36_superlayer_replace_l0_any_requested() {
    return qwen36_superlayer_replace_l0_rms_requested() ||
        qwen36_superlayer_replace_l0_qkv_requested() ||
        qwen36_superlayer_replace_l0_proj_requested() ||
        qwen36_superlayer_replace_l0_ssm_requested() ||
        qwen36_superlayer_replace_l0_l2_requested() ||
        qwen36_superlayer_replace_l0_gdn_requested() ||
        qwen36_superlayer_replace_l0_gated_norm_requested() ||
        qwen36_superlayer_replace_l0_attn_out_requested() ||
        qwen36_superlayer_replace_l0_post_attn_requested() ||
        qwen36_superlayer_replace_l0_moe_router_requested() ||
        qwen36_superlayer_replace_l0_moe_gate_up_requested() ||
        qwen36_superlayer_replace_l0_moe_down_requested();
}

static uint32_t qwen36_superlayer_l0_stage_mask() {
    if (qwen36_superlayer_env_enabled("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_RUN_L0")) {
        return 0x1fu;
    }

    uint32_t mask = 0;
    if (qwen36_superlayer_replace_l0_rms_requested()) {
        mask |= 0x1u;
    }
    if (qwen36_superlayer_replace_l0_qkv_requested()) {
        mask |= 0x2u;
    }
    if (qwen36_superlayer_replace_l0_proj_z_requested()) {
        mask |= 0x4u;
    }
    if (qwen36_superlayer_replace_l0_proj_z_math_only_requested()) {
        mask |= 0x20u;
    }
    if (qwen36_superlayer_replace_l0_proj_beta_requested()) {
        mask |= 0x8u;
    }
    if (qwen36_superlayer_replace_l0_proj_alpha_requested()) {
        mask |= 0x10u;
    }
    if (qwen36_superlayer_replace_l0_ssm_requested()) {
        mask |= 0x40u;
    }
    if (qwen36_superlayer_replace_l0_l2_requested()) {
        mask |= 0x80u;
    }
    if (qwen36_superlayer_replace_l0_gdn_requested()) {
        mask |= 0x100u;
    }
    if (qwen36_superlayer_replace_l0_gated_norm_requested()) {
        mask |= 0x200u;
    }
    if (qwen36_superlayer_replace_l0_attn_out_requested()) {
        mask |= 0x400u;
    }
    if (qwen36_superlayer_replace_l0_post_attn_requested()) {
        mask |= 0x800u;
    }
    if (qwen36_superlayer_replace_l0_moe_router_requested()) {
        mask |= 0x1000u;
    }
    if (qwen36_superlayer_replace_l0_moe_gate_up_requested()) {
        mask |= 0x2000u;
    }
    if (qwen36_superlayer_replace_l0_moe_down_requested()) {
        mask |= 0x4000u;
    }
    return mask;
}

static bool qwen36_superlayer_run_l0_math_enabled() {
    return qwen36_superlayer_l0_stage_mask() != 0;
}

static bool qwen36_superlayer_contract_kernel_enabled() {
    return qwen36_superlayer_final_physical_l0_requested() ||
        qwen36_superlayer_env_enabled("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_CONTRACT") ||
        qwen36_superlayer_run_l0_math_enabled();
}

static bool qwen36_superlayer_contract_weightpack_requested() {
    return qwen36_superlayer_env_enabled("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER") ||
        qwen36_superlayer_env_enabled("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REQUIRED") ||
        qwen36_superlayer_env_enabled("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_CONTRACT") ||
        qwen36_superlayer_env_enabled("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_SMOKE");
}

static bool qwen36_superlayer_contract_dispatch_enabled() {
    return qwen36_superlayer_final_requested() ||
        qwen36_superlayer_final_physical_l0_requested() ||
        qwen36_superlayer_env_enabled("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_DISPATCH") ||
        qwen36_superlayer_env_enabled("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_SMOKE") ||
        qwen36_superlayer_env_enabled("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER") ||
        qwen36_superlayer_env_enabled("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REQUIRED") ||
        qwen36_superlayer_env_enabled("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_CONTRACT") ||
        qwen36_superlayer_l0_env_requested();
}

static bool qwen36_superlayer_requested() {
    return qwen36_superlayer_final_requested() ||
        qwen36_superlayer_final_physical_l0_requested() ||
        qwen36_superlayer_env_enabled("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER") ||
        qwen36_superlayer_env_enabled("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REQUIRED") ||
        qwen36_superlayer_env_enabled("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_CONTRACT") ||
        qwen36_superlayer_l0_env_requested();
}

} // namespace

bool ggml_cuda_rdna3_qwen36_superlayer_enabled(const int device) {
    const int cc = ggml_cuda_info().devices[device].cc;
    const bool requested = qwen36_superlayer_requested();
    const bool rdna3 = GGML_CUDA_CC_IS_RDNA3(cc);
    const bool enabled = requested && rdna3;
    if (qwen36_superlayer_trace_enabled()) {
        static std::atomic<int64_t> enabled_reports{0};
        const int64_t report_id = enabled_reports.fetch_add(1, std::memory_order_relaxed);
        if (report_id < 64) {
            fprintf(stderr,
                    "rdna3_qwen36_superlayer: enabled-check device=%d requested=%d rdna3=%d"
                    " enabled=%d cc_raw=0x%x cc_visible=0x%x dispatch=%d contract=%d"
                    " run_l0_math=%d replace_l0_any=%d\n",
                    device,
                    requested ? 1 : 0,
                    rdna3 ? 1 : 0,
                    enabled ? 1 : 0,
                    cc,
                    cc & 0xffff,
                    qwen36_superlayer_contract_dispatch_enabled() ? 1 : 0,
                    qwen36_superlayer_contract_kernel_enabled() ? 1 : 0,
                    qwen36_superlayer_run_l0_math_enabled() ? 1 : 0,
                    qwen36_superlayer_replace_l0_any_requested() ? 1 : 0);
            fflush(stderr);
        }
    }
    return enabled;
}

bool ggml_cuda_rdna3_qwen36_superlayer_required(const int device) {
    return ggml_cuda_rdna3_qwen36_superlayer_enabled(device) &&
        qwen36_superlayer_env_enabled("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REQUIRED");
}

bool ggml_cuda_rdna3_qwen36_superlayer_runtime_enabled(const int device) {
    return ggml_cuda_rdna3_qwen36_superlayer_enabled(device) &&
        qwen36_superlayer_contract_dispatch_enabled() &&
        qwen36_superlayer_contract_kernel_enabled();
}

bool ggml_cuda_rdna3_qwen36_superlayer_final_enabled(const int device) {
    return ggml_cuda_rdna3_qwen36_superlayer_enabled(device) &&
        qwen36_superlayer_contract_dispatch_enabled() &&
        qwen36_superlayer_final_requested();
}

bool ggml_cuda_rdna3_qwen36_superlayer_replace_l0_enabled(const int device) {
    return ggml_cuda_rdna3_qwen36_superlayer_enabled(device) &&
        qwen36_superlayer_contract_dispatch_enabled() &&
        qwen36_superlayer_replace_l0_any_requested();
}

bool ggml_cuda_rdna3_qwen36_superlayer_replace_l0_rms_enabled(const int device) {
    return ggml_cuda_rdna3_qwen36_superlayer_enabled(device) &&
        qwen36_superlayer_contract_dispatch_enabled() &&
        qwen36_superlayer_replace_l0_rms_requested();
}

bool ggml_cuda_rdna3_qwen36_superlayer_replace_l0_qkv_enabled(const int device) {
    return ggml_cuda_rdna3_qwen36_superlayer_enabled(device) &&
        qwen36_superlayer_contract_dispatch_enabled() &&
        qwen36_superlayer_replace_l0_qkv_requested();
}

bool ggml_cuda_rdna3_qwen36_superlayer_replace_l0_proj_enabled(const int device) {
    return ggml_cuda_rdna3_qwen36_superlayer_enabled(device) &&
        qwen36_superlayer_contract_dispatch_enabled() &&
        qwen36_superlayer_replace_l0_proj_requested();
}

bool ggml_cuda_rdna3_qwen36_superlayer_replace_l0_proj_z_enabled(const int device) {
    return ggml_cuda_rdna3_qwen36_superlayer_enabled(device) &&
        qwen36_superlayer_contract_dispatch_enabled() &&
        qwen36_superlayer_replace_l0_proj_z_requested();
}

bool ggml_cuda_rdna3_qwen36_superlayer_replace_l0_proj_beta_enabled(const int device) {
    return ggml_cuda_rdna3_qwen36_superlayer_enabled(device) &&
        qwen36_superlayer_contract_dispatch_enabled() &&
        qwen36_superlayer_replace_l0_proj_beta_requested();
}

bool ggml_cuda_rdna3_qwen36_superlayer_replace_l0_proj_alpha_enabled(const int device) {
    return ggml_cuda_rdna3_qwen36_superlayer_enabled(device) &&
        qwen36_superlayer_contract_dispatch_enabled() &&
        qwen36_superlayer_replace_l0_proj_alpha_requested();
}

bool ggml_cuda_rdna3_qwen36_superlayer_replace_l0_ssm_enabled(const int device) {
    return ggml_cuda_rdna3_qwen36_superlayer_enabled(device) &&
        qwen36_superlayer_contract_dispatch_enabled() &&
        qwen36_superlayer_replace_l0_ssm_requested();
}

bool ggml_cuda_rdna3_qwen36_superlayer_replace_l0_l2_enabled(const int device) {
    return ggml_cuda_rdna3_qwen36_superlayer_enabled(device) &&
        qwen36_superlayer_contract_dispatch_enabled() &&
        qwen36_superlayer_replace_l0_l2_requested();
}

bool ggml_cuda_rdna3_qwen36_superlayer_replace_l0_gdn_enabled(const int device) {
    return ggml_cuda_rdna3_qwen36_superlayer_enabled(device) &&
        qwen36_superlayer_contract_dispatch_enabled() &&
        qwen36_superlayer_replace_l0_gdn_requested();
}

bool ggml_cuda_rdna3_qwen36_superlayer_replace_l0_gated_norm_enabled(const int device) {
    return ggml_cuda_rdna3_qwen36_superlayer_enabled(device) &&
        qwen36_superlayer_contract_dispatch_enabled() &&
        qwen36_superlayer_replace_l0_gated_norm_requested();
}

bool ggml_cuda_rdna3_qwen36_superlayer_replace_l0_attn_out_enabled(const int device) {
    return ggml_cuda_rdna3_qwen36_superlayer_enabled(device) &&
        qwen36_superlayer_contract_dispatch_enabled() &&
        qwen36_superlayer_replace_l0_attn_out_requested();
}

bool ggml_cuda_rdna3_qwen36_superlayer_replace_l0_post_attn_enabled(const int device) {
    return ggml_cuda_rdna3_qwen36_superlayer_enabled(device) &&
        qwen36_superlayer_contract_dispatch_enabled() &&
        qwen36_superlayer_replace_l0_post_attn_requested();
}

bool ggml_cuda_rdna3_qwen36_superlayer_replace_l0_moe_router_enabled(const int device) {
    return ggml_cuda_rdna3_qwen36_superlayer_enabled(device) &&
        qwen36_superlayer_contract_dispatch_enabled() &&
        qwen36_superlayer_replace_l0_moe_router_requested();
}

bool ggml_cuda_rdna3_qwen36_superlayer_replace_l0_moe_gate_up_enabled(const int device) {
    return ggml_cuda_rdna3_qwen36_superlayer_enabled(device) &&
        qwen36_superlayer_contract_dispatch_enabled() &&
        qwen36_superlayer_replace_l0_moe_gate_up_requested();
}

bool ggml_cuda_rdna3_qwen36_superlayer_replace_l0_moe_down_enabled(const int device) {
    return ggml_cuda_rdna3_qwen36_superlayer_enabled(device) &&
        qwen36_superlayer_contract_dispatch_enabled() &&
        qwen36_superlayer_replace_l0_moe_down_requested();
}

bool ggml_cuda_rdna3_qwen36_superlayer_prepare(
        ggml_backend_cuda_context * cuda_ctx,
        const ggml_cgraph * cgraph,
        std::string * blocker) {
    if (cuda_ctx == nullptr) {
        if (blocker != nullptr) {
            *blocker = "missing CUDA/HIP backend context";
        }
        return false;
    }

    qwen36_superlayer_plan plan = qwen36_superlayer_make_plan(cgraph, cuda_ctx->device);
    if (!plan.blocker.empty()) {
        if (blocker != nullptr) {
            *blocker = plan.blocker;
        }
        return false;
    }
    qwen36_superlayer_pack_plan artifact_pack = qwen36_superlayer_make_pack_plan(cgraph, plan);
    if (artifact_pack.refs.empty()) {
        if (blocker != nullptr) {
            *blocker = "superlayer weight pack would be empty";
        }
        return false;
    }
    qwen36_superlayer_pack_plan runtime_pack =
        qwen36_superlayer_make_runtime_pack_plan(artifact_pack);
    if (runtime_pack.refs.empty()) {
        if (blocker != nullptr) {
            *blocker = "superlayer runtime weight pack would be empty";
        }
        return false;
    }
    const qwen36_superlayer_runtime_layout runtime = qwen36_superlayer_make_runtime_layout(cgraph, plan);
    const qwen36_superlayer_runtime_binding_plan bindings =
        qwen36_superlayer_make_runtime_binding_plan(cgraph, plan, cuda_ctx->device);
    if (bindings.refs.empty()) {
        if (blocker != nullptr) {
            *blocker = "superlayer runtime binding table would be empty";
        }
        return false;
    }

    static std::mutex prepared_mutex;
    static std::unordered_set<uint64_t> prepared_fingerprints;
    static std::unordered_set<uint64_t> reported_fingerprints;

    bool already_prepared = false;
    {
        std::lock_guard<std::mutex> lock(prepared_mutex);
        already_prepared = prepared_fingerprints.find(plan.fingerprint) != prepared_fingerprints.end();
    }

    if (!already_prepared) {
        if (!qwen36_superlayer_materialize_artifact(plan, artifact_pack, runtime, bindings, cuda_ctx->device, blocker)) {
            return false;
        }
        std::lock_guard<std::mutex> lock(prepared_mutex);
        prepared_fingerprints.insert(plan.fingerprint);
    }

    bool should_report = false;
    {
        std::lock_guard<std::mutex> lock(prepared_mutex);
        should_report = reported_fingerprints.insert(plan.fingerprint).second;
    }

    if (should_report && qwen36_superlayer_trace_enabled()) {
        fprintf(stderr,
                "rdna3_qwen36_superlayer: artifact-ready fingerprint=%s dir=%s nodes=%d"
                " layers=40 fattn=%d gdn=%d topk=%d moe_gate_up=%d moe_down=%d mmid=%d"
                " artifact_weightpack_tensors=%zu artifact_weightpack_bytes=%zu"
                " runtime_weightpack_tensors=%zu runtime_bindings=%zu runtime_weightpack_bytes=%zu"
                " runtime_scratch_bytes=%zu"
                " state=artifact-ready\n",
                hex_u64(plan.fingerprint).c_str(), plan.artifact_dir.string().c_str(), plan.n_nodes,
                plan.n_fattn, plan.n_gdn, plan.n_topk_moe, plan.n_moe_gate_up, plan.n_moe_down, plan.n_mmid,
                artifact_pack.refs.size(), artifact_pack.total_bytes,
                runtime_pack.refs.size(), bindings.refs.size(), runtime_pack.total_bytes,
                runtime.scratch_bytes);
        fflush(stderr);
    }

    return true;
}

bool ggml_cuda_rdna3_qwen36_superlayer_maybe_launch_contract(
        ggml_backend_cuda_context * cuda_ctx,
        const ggml_cgraph * cgraph,
        std::string * blocker,
        const uint32_t forced_l0_stage_mask) {
    const uint32_t l0_stage_mask_arg =
        forced_l0_stage_mask == UINT32_MAX ? qwen36_superlayer_l0_stage_mask() : forced_l0_stage_mask;
    if (qwen36_superlayer_final_requested() && !qwen36_superlayer_final_numeric_ready() &&
            l0_stage_mask_arg == 0) {
        if (blocker != nullptr) {
            *blocker = qwen36_superlayer_final_blocker();
        }
        return false;
    }

    if (!qwen36_superlayer_contract_dispatch_enabled()) {
        return true;
    }
    if (!qwen36_superlayer_contract_kernel_enabled()) {
        static std::atomic<int64_t> skipped_reports{0};
        const int64_t report_id = skipped_reports.fetch_add(1, std::memory_order_relaxed);
        if (report_id < 4) {
            fprintf(stderr,
                    "rdna3_qwen36_superlayer: contract-kernel-skipped"
                    " reason=no per-token superlayer work enabled l0_math=0 replace_l0=0\n");
            fflush(stderr);
        }
        return true;
    }
    if (!ggml_cuda_info().devices[cuda_ctx->device].supports_cooperative_launch) {
        if (blocker != nullptr) {
            *blocker = "superlayer dispatch requires cooperative launch support";
        }
        return false;
    }

    qwen36_superlayer_plan plan = qwen36_superlayer_make_plan(cgraph, cuda_ctx->device);
    if (!plan.blocker.empty()) {
        if (blocker != nullptr) {
            *blocker = plan.blocker;
        }
        return false;
    }
    qwen36_superlayer_pack_plan full_pack = qwen36_superlayer_make_pack_plan(cgraph, plan);
    if (full_pack.refs.empty()) {
        if (blocker != nullptr) {
            *blocker = "superlayer weight pack would be empty";
        }
        return false;
    }
    qwen36_superlayer_pack_plan pack = qwen36_superlayer_make_runtime_pack_plan(full_pack);
    if (pack.refs.empty()) {
        if (blocker != nullptr) {
            *blocker = "superlayer runtime weight pack would be empty";
        }
        return false;
    }
    const uint32_t direct_l0_stage_mask = qwen36_superlayer_direct_l0_stage_mask();
    const uint32_t l0_weighted_stage_mask = l0_stage_mask_arg & ~0x180u;
    const bool l0_weightpack_required = (l0_weighted_stage_mask & ~direct_l0_stage_mask) != 0;
    const bool device_weightpack_required =
        qwen36_superlayer_contract_weightpack_requested() ||
        l0_weightpack_required ||
        (!qwen36_superlayer_final_requested() && l0_stage_mask_arg == 0);
    qwen36_superlayer_device_pack_view device_pack;
    if (!qwen36_superlayer_materialize_device_pack(
                cuda_ctx, cgraph, plan, pack, device_weightpack_required, &device_pack, blocker)) {
        return false;
    }

    ggml_cuda_pool_alloc<qwen36_superlayer_contract_state> contract_state(cuda_ctx->pool(), 1);
    qwen36_superlayer_contract_state host_state = {
        /*.fingerprint         =*/ plan.fingerprint,
        /*.weightpack_bytes    =*/ (uint64_t) device_pack.bytes,
        /*.scratch_bytes       =*/ (uint64_t) device_pack.runtime.scratch_bytes,
        /*.checksum            =*/ 0,
        /*.n_nodes             =*/ (uint32_t) plan.n_nodes,
        /*.n_blocks            =*/ 0,
        /*.weightpack_tensors  =*/ (uint32_t) device_pack.tensors,
        /*.touched_layers      =*/ 0,
        /*.active_lanes        =*/ 0,
    };
    CUDA_CHECK(cudaMemcpyAsync(
            contract_state.ptr, &host_state, sizeof(host_state), cudaMemcpyHostToDevice, cuda_ctx->stream()));

    const int default_blocks = std::max(1, ggml_cuda_info().devices[cuda_ctx->device].nsm);
    const int blocks = (int) std::max<int64_t>(1, qwen36_superlayer_env_i64(
            "GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_BLOCKS", default_blocks));
    const int64_t requested_threads = qwen36_superlayer_env_i64(
            "GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_THREADS", 256);
    const int threads =
        requested_threads >= 256 ? 256 :
        requested_threads >= 128 ? 128 :
        requested_threads >=  64 ?  64 : 32;

    qwen36_superlayer_contract_state * state_arg = contract_state.ptr;
    const uint8_t * weightpack_arg = (const uint8_t *) device_pack.data;
    qwen36_superlayer_layer_pack_desc * layers_arg = device_pack.layers;
    qwen36_superlayer_runtime_tensor_desc * io_descs_arg = device_pack.io_descs;
    const uint32_t io_count_arg = (uint32_t) device_pack.io_count;
    qwen36_superlayer_l0_norm_desc * l0_norm_arg = device_pack.l0_norm;
    qwen36_superlayer_l0_qkv_desc * l0_qkv_arg = device_pack.l0_qkv;
    qwen36_superlayer_l0_proj_desc * l0_proj_arg = device_pack.l0_proj;
    qwen36_superlayer_l0_ssm_desc * l0_ssm_arg = device_pack.l0_ssm;
    qwen36_superlayer_l0_l2_desc * l0_l2_arg = device_pack.l0_l2;
    qwen36_superlayer_l0_gdn_desc * l0_gdn_arg = device_pack.l0_gdn;
    qwen36_superlayer_l0_gated_norm_desc * l0_gated_norm_arg = device_pack.l0_gated_norm;
    qwen36_superlayer_l0_attn_out_desc * l0_attn_out_arg = device_pack.l0_attn_out;
    qwen36_superlayer_l0_post_attn_desc * l0_post_attn_arg = device_pack.l0_post_attn;
    qwen36_superlayer_l0_moe_router_desc * l0_moe_router_arg = device_pack.l0_moe_router;
    qwen36_superlayer_l0_moe_gate_up_desc * l0_moe_gate_up_arg = device_pack.l0_moe_gate_up;
    qwen36_superlayer_l0_moe_down_desc * l0_moe_down_arg = device_pack.l0_moe_down;
    uint8_t * scratch_arg = (uint8_t *) device_pack.scratch;
    const uint64_t scratch_bytes_arg = (uint64_t) device_pack.runtime.scratch_bytes;
    const uint64_t weightpack_bytes_arg = (uint64_t) device_pack.bytes;
    void * kernel_args[] = {
        (void *) &state_arg,
        (void *) &weightpack_arg,
        (void *) &layers_arg,
        (void *) &io_descs_arg,
        (void *) &io_count_arg,
        (void *) &l0_norm_arg,
        (void *) &l0_qkv_arg,
        (void *) &l0_proj_arg,
        (void *) &l0_ssm_arg,
        (void *) &l0_l2_arg,
        (void *) &l0_gdn_arg,
        (void *) &l0_gated_norm_arg,
        (void *) &l0_attn_out_arg,
        (void *) &l0_post_attn_arg,
        (void *) &l0_moe_router_arg,
        (void *) &l0_moe_gate_up_arg,
        (void *) &l0_moe_down_arg,
        (void *) &scratch_arg,
        (void *) &scratch_bytes_arg,
        (void *) &weightpack_bytes_arg,
        (void *) &l0_stage_mask_arg,
    };
    CUDA_CHECK(cudaLaunchCooperativeKernel(
            (void *) qwen36_rdna3_superlayer_contract_kernel,
            dim3((unsigned int) blocks, 1, 1), dim3((unsigned int) threads, 1, 1),
            kernel_args, 0, cuda_ctx->stream()));
    CUDA_CHECK(cudaGetLastError());

    static std::atomic<int64_t> contract_reports{0};
    const int64_t report_id = contract_reports.fetch_add(1, std::memory_order_relaxed);
    (void) report_id;
    if (qwen36_superlayer_trace_enabled() || (qwen36_superlayer_final_requested() && report_id < 4)) {
        const bool final_requested = qwen36_superlayer_final_requested();
        fprintf(stderr,
                "rdna3_qwen36_superlayer: %s fingerprint=%s blocks=%d threads=%d"
                " weightpack_tensors=%zu runtime_bindings=%zu weightpack_bytes=%zu scratch_bytes=%zu"
                " l0_stage_mask=0x%x l0_rms_norm=%s l0_qkv=%s l0_projection=%s replace_l0=%d"
                " l0_proj_z=%s l0_proj_z_math_only=%s l0_proj_beta=%s l0_proj_alpha=%s"
                " l0_ssm=%s l0_l2=%s l0_gdn=%s l0_gated_norm=%s l0_attn_out=%s l0_post_attn=%s l0_moe_router=%s l0_moe_gate_up=%s l0_moe_down=%s"
                " direct_l0_weights=%d direct_l0_norm_weights=%d direct_l0_qkv_weights=%d"
                " direct_l0_proj_weights=%d direct_l0_ssm_weights=%d direct_l0_out_weights=%d direct_l0_moe_weights=%d"
                " weightpack_required=%d final_requested=%d numeric_layers=%d/40"
                " note=%s\n",
                final_requested ? "final-kernel-launched" : "contract-kernel-launched",
                hex_u64(plan.fingerprint).c_str(), blocks, threads,
                device_pack.tensors, device_pack.io_count, device_pack.bytes, device_pack.runtime.scratch_bytes,
                l0_stage_mask_arg,
                (l0_stage_mask_arg & 0x1u) != 0 ? "on" : "off",
                (l0_stage_mask_arg & 0x2u) != 0 ? "on" : "off",
                (l0_stage_mask_arg & 0x1cu) != 0 ? "on" : "off",
                ggml_cuda_rdna3_qwen36_superlayer_replace_l0_enabled(cuda_ctx->device) ? 1 : 0,
                (l0_stage_mask_arg & 0x4u) != 0 ? "on" : "off",
                (l0_stage_mask_arg & 0x20u) != 0 ? "on" : "off",
                (l0_stage_mask_arg & 0x8u) != 0 ? "on" : "off",
                (l0_stage_mask_arg & 0x10u) != 0 ? "on" : "off",
                (l0_stage_mask_arg & 0x40u) != 0 ? "on" : "off",
                (l0_stage_mask_arg & 0x80u) != 0 ? "on" : "off",
                (l0_stage_mask_arg & 0x100u) != 0 ? "on" : "off",
                (l0_stage_mask_arg & 0x200u) != 0 ? "on" : "off",
                (l0_stage_mask_arg & 0x400u) != 0 ? "on" : "off",
                (l0_stage_mask_arg & 0x800u) != 0 ? "on" : "off",
                (l0_stage_mask_arg & 0x1000u) != 0 ? "on" : "off",
                (l0_stage_mask_arg & 0x2000u) != 0 ? "on" : "off",
                (l0_stage_mask_arg & 0x4000u) != 0 ? "on" : "off",
                qwen36_superlayer_direct_l0_weights_requested() ? 1 : 0,
                qwen36_superlayer_direct_l0_norm_weights_requested() ? 1 : 0,
                qwen36_superlayer_direct_l0_qkv_weights_requested() ? 1 : 0,
                qwen36_superlayer_direct_l0_proj_weights_requested() ? 1 : 0,
                qwen36_superlayer_direct_l0_ssm_weights_requested() ? 1 : 0,
                qwen36_superlayer_direct_l0_out_weights_requested() ? 1 : 0,
                qwen36_superlayer_direct_l0_moe_weights_requested() ? 1 : 0,
                device_weightpack_required ? 1 : 0,
                final_requested ? 1 : 0,
                l0_stage_mask_arg != 0 ? 1 : 0,
                final_requested ?
                    (l0_stage_mask_arg != 0 ?
                        "final physical L0 cooperative dispatch; 40-layer dataflow must replace scaffold" :
                        "final cooperative 40-layer contract scaffold; numeric dataflow still runs in graph") :
                    "single physical cooperative 40-layer dispatch scaffold");
        fflush(stderr);
    }

    return true;
}

bool ggml_cuda_rdna3_qwen36_superlayer_maybe_launch_smoke(
        ggml_backend_cuda_context * cuda_ctx,
        const ggml_cgraph * cgraph,
        std::string * blocker) {
    return ggml_cuda_rdna3_qwen36_superlayer_maybe_launch_contract(cuda_ctx, cgraph, blocker, UINT32_MAX);
}
