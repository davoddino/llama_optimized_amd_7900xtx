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
    uint64_t norm_w_offset = 0;
    uint32_t n_embd = 0;
    float eps = 0.0f;
    uint32_t has_norm_w = 0;
    uint32_t ready = 0;
};

struct qwen36_superlayer_l0_qkv_desc {
    float * qkv_math_out = nullptr;
    float * qkv_out = nullptr;
    float * qkv_named_out = nullptr;
    uint64_t wqkv_offset = 0;
    uint64_t qkv_scale_offset = 0;
    uint64_t qkv_scratch_offset = 0;
    uint64_t wqkv_nb1 = 0;
    uint32_t n_embd = 0;
    uint32_t n_out = 0;
    int32_t wqkv_type = GGML_TYPE_COUNT;
    uint32_t qkv_scale_n = 0;
    uint32_t has_qkv_scale = 0;
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
    uint64_t wz_offset = 0;
    uint64_t wbeta_offset = 0;
    uint64_t walpha_offset = 0;
    uint64_t alpha_dt_offset = 0;
    uint64_t alpha_a_offset = 0;
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
    desc->norm_w_offset = norm_w_offset;
    desc->n_embd = (uint32_t) x->ne[0];
    desc->eps = eps;
    desc->has_norm_w = norm_w != nullptr ? 1u : 0u;
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

    uint64_t wqkv_offset = 0;
    if (!qwen36_superlayer_find_pack_offset(pack, wqkv, &wqkv_offset)) {
        if (blocker != nullptr) {
            *blocker = "L0 QKV weight is not present in the fused weightpack: ";
            *blocker += wqkv->name;
        }
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
    desc->wqkv_offset = wqkv_offset;
    desc->qkv_scale_offset = qkv_scale_offset;
    desc->qkv_scratch_offset = qkv_scratch_offset;
    desc->wqkv_nb1 = (uint64_t) wqkv->nb[1];
    desc->n_embd = (uint32_t) wqkv->ne[0];
    desc->n_out = (uint32_t) qkv_math->ne[0];
    desc->wqkv_type = (int32_t) wqkv->type;
    desc->qkv_scale_n = qkv_scale_n;
    desc->has_qkv_scale = qkv_scale != nullptr ? 1u : 0u;
    desc->ready = 1u;
    return true;
}

static const ggml_tensor * qwen36_superlayer_resolve_mul_mat(const ggml_tensor * node) {
    const ggml_tensor * out = qwen36_superlayer_strip_view_ops(node);
    const ggml_tensor * mm = out;
    if (out != nullptr && out->op == GGML_OP_MUL) {
        if (out->src[0] != nullptr && out->src[0]->op == GGML_OP_MUL_MAT) {
            mm = out->src[0];
        } else if (out->src[1] != nullptr && out->src[1]->op == GGML_OP_MUL_MAT) {
            mm = out->src[1];
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

    const ggml_tensor * z_math = qwen36_superlayer_resolve_mul_mat(z);
    const ggml_tensor * beta_math = nullptr;
    if (beta_sigmoid->op == GGML_OP_UNARY && ggml_get_unary_op(beta_sigmoid) == GGML_UNARY_OP_SIGMOID) {
        if (!qwen36_superlayer_same_tensor_or_view(beta_sigmoid->src[0], beta)) {
            if (blocker != nullptr) {
                *blocker = "L0 projection beta_sigmoid-0 does not consume beta-0";
            }
            return false;
        }
        beta_math = qwen36_superlayer_resolve_mul_mat(beta);
    }
    const ggml_tensor * alpha_math = qwen36_superlayer_resolve_mul_mat(alpha);
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
            !ggml_is_contiguous_1(z) ||
            !ggml_is_contiguous_1(z_math) ||
            !ggml_is_contiguous_1(beta) ||
            !ggml_is_contiguous_1(beta_math) ||
            !ggml_is_contiguous_1(beta_sigmoid) ||
            !ggml_is_contiguous_1(alpha) ||
            !ggml_is_contiguous_1(alpha_math) ||
            !ggml_is_contiguous_1(alpha_biased) ||
            !ggml_is_contiguous_1(alpha_softplus) ||
            !ggml_is_contiguous_1(alpha_gate)) {
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
            ggml_nelements(alpha_a) != ggml_nelements(alpha_gate)) {
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

    uint64_t wz_offset = 0;
    uint64_t wbeta_offset = 0;
    uint64_t walpha_offset = 0;
    uint64_t alpha_dt_offset = 0;
    uint64_t alpha_a_offset = 0;
    if (!qwen36_superlayer_find_pack_offset_named(pack, wz, "L0 z weight", &wz_offset, blocker) ||
            !qwen36_superlayer_find_pack_offset_named(pack, wbeta, "L0 beta weight", &wbeta_offset, blocker) ||
            !qwen36_superlayer_find_pack_offset_named(pack, walpha, "L0 alpha weight", &walpha_offset, blocker) ||
            !qwen36_superlayer_find_pack_offset_named(pack, alpha_dt, "L0 alpha dt", &alpha_dt_offset, blocker) ||
            !qwen36_superlayer_find_pack_offset_named(pack, alpha_a, "L0 alpha gate scale", &alpha_a_offset, blocker)) {
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
    desc->wz_offset = wz_offset;
    desc->wbeta_offset = wbeta_offset;
    desc->walpha_offset = walpha_offset;
    desc->alpha_dt_offset = alpha_dt_offset;
    desc->alpha_a_offset = alpha_a_offset;
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
    desc->ready = 1u;
    return true;
}

static bool qwen36_superlayer_materialize_device_pack(
        ggml_backend_cuda_context * cuda_ctx,
        const ggml_cgraph * cgraph,
        const qwen36_superlayer_plan & plan,
        const qwen36_superlayer_pack_plan & pack,
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
        qwen36_superlayer_device_pack_key(cuda_ctx->device, plan.fingerprint, source_signature);
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

    static std::mutex device_pack_mutex;
    static std::unordered_map<std::string, qwen36_superlayer_device_pack_entry> device_packs;
    static std::unordered_set<std::string> reported_device_packs;

    qwen36_superlayer_device_pack_view local_view;
    bool should_report = false;

    {
        std::lock_guard<std::mutex> lock(device_pack_mutex);
        auto it = device_packs.find(key);
        if (it == device_packs.end()) {
            for (const qwen36_superlayer_pack_ref & ref : pack.refs) {
                if (!qwen36_superlayer_tensor_on_device(ref.tensor, cuda_ctx->device, blocker)) {
                    return false;
                }
            }

            ggml_cuda_set_device(cuda_ctx->device);

            void * data = nullptr;
            cudaError_t err = cudaMalloc(&data, pack.total_bytes);
            if (err != cudaSuccess) {
                (void) cudaGetLastError();
                qwen36_superlayer_set_cuda_blocker(blocker, "failed to allocate superlayer device weightpack", err);
                return false;
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

            err = cudaMemsetAsync(data, 0, pack.total_bytes, cuda_ctx->stream());
            if (err != cudaSuccess) {
                (void) cudaFree(io_descs);
                (void) cudaFree(scratch);
                (void) cudaFree(layer_descs);
                (void) cudaFree(data);
                qwen36_superlayer_set_cuda_blocker(blocker, "failed to clear superlayer device weightpack", err);
                return false;
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

            cudaEvent_t ready_event = nullptr;
            err = cudaEventCreateWithFlags(&ready_event, cudaEventDisableTiming);
            if (err != cudaSuccess) {
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
            entry.scratch = scratch;
            entry.bytes = pack.total_bytes;
            entry.tensors = pack.refs.size();
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

    if (should_report) {
        GGML_LOG_INFO(
                "rdna3_qwen36_superlayer: device-pack-ready fingerprint=%s source=%s runtime=%s"
                " ptr=%p layer_descs=%p io_descs=%p l0_norm=%p l0_qkv=%p l0_proj=%p scratch=%p tensors=%zu io=%zu bytes=%zu"
                " scratch_bytes=%zu activation_slot=%zu projection_slot=%zu logits_bytes=%zu router_slot=%zu"
                " n_embd=%" PRId64 " n_vocab=%" PRId64 "\n",
                hex_u64(plan.fingerprint).c_str(), hex_u64(source_signature).c_str(),
                hex_u64(runtime_signature).c_str(),
                local_view.data, local_view.layers, local_view.io_descs, local_view.l0_norm, local_view.l0_qkv,
                local_view.l0_proj, local_view.scratch,
                local_view.tensors, local_view.io_count, local_view.bytes,
                local_view.runtime.scratch_bytes, local_view.runtime.activation_slot_bytes,
                local_view.runtime.projection_slot_bytes, local_view.runtime.logits_bytes, local_view.runtime.router_slot_bytes,
                local_view.runtime.n_embd, local_view.runtime.n_vocab);
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
    if (desc->has_norm_w != 0 && weightpack == nullptr) {
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
        (const float *) (weightpack + desc->norm_w_offset) : nullptr;
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
    if (desc == nullptr || desc->ready == 0 ||
            weightpack == nullptr || scratch == nullptr || sums == nullptr) {
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
    const char * wqkv = (const char *) (weightpack + desc->wqkv_offset);
    const float * qkv_scale = desc->has_qkv_scale != 0 ?
        (const float *) (weightpack + desc->qkv_scale_offset) : nullptr;
    const ggml_type wqkv_type = (ggml_type) desc->wqkv_type;

    const int tid = threadIdx.x;
    for (uint32_t row = blockIdx.x; row < n_out; row += gridDim.x) {
        float acc = 0.0f;
        for (uint32_t col = tid; col < n_embd; col += blockDim.x) {
            const float w = qwen36_rdna3_superlayer_dequant_weight(
                    wqkv, wqkv_type, (int64_t) desc->wqkv_nb1, row, col);
            acc += norm[col]*w;
        }

        sums[tid] = acc;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                sums[tid] += sums[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            const float raw = sums[0];
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
        __syncthreads();
    }
}

__device__ __forceinline__ void qwen36_rdna3_superlayer_l0_projection_bundle(
        const qwen36_superlayer_l0_proj_desc * desc,
        const uint8_t * weightpack,
        uint8_t * scratch,
        const uint64_t scratch_bytes,
        float * sums,
        const uint32_t write_outputs) {
    if (desc == nullptr || desc->ready == 0 ||
            weightpack == nullptr || scratch == nullptr || sums == nullptr) {
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
    const float * alpha_dt = (const float *) (weightpack + desc->alpha_dt_offset);
    const float * alpha_a = (const float *) (weightpack + desc->alpha_a_offset);
    const int tid = threadIdx.x;

    for (uint32_t row = blockIdx.x; row < total_out; row += gridDim.x) {
        const char * w = nullptr;
        uint64_t w_nb1 = 0;
        ggml_type wtype = GGML_TYPE_COUNT;
        uint32_t local_row = row;
        if (local_row < desc->z_out) {
            w = (const char *) (weightpack + desc->wz_offset);
            w_nb1 = desc->wz_nb1;
            wtype = (ggml_type) desc->wz_type;
        } else {
            local_row -= desc->z_out;
            if (local_row < desc->beta_out) {
                w = (const char *) (weightpack + desc->wbeta_offset);
                w_nb1 = desc->wbeta_nb1;
                wtype = (ggml_type) desc->wbeta_type;
            } else {
                local_row -= desc->beta_out;
                w = (const char *) (weightpack + desc->walpha_offset);
                w_nb1 = desc->walpha_nb1;
                wtype = (ggml_type) desc->walpha_type;
            }
        }

        float acc = 0.0f;
        for (uint32_t col = tid; col < n_embd; col += blockDim.x) {
            const float ww = qwen36_rdna3_superlayer_dequant_weight(
                    w, wtype, (int64_t) w_nb1, local_row, col);
            acc += norm[col]*ww;
        }

        sums[tid] = acc;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                sums[tid] += sums[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            local_row = row;
            if (local_row < desc->z_out) {
                const float v = sums[0];
                z[local_row] = v;
                if (write_outputs != 0 && desc->z_math_dst != nullptr) {
                    desc->z_math_dst[local_row] = v;
                }
                if (write_outputs != 0 && desc->z_dst != nullptr) {
                    desc->z_dst[local_row] = v;
                }
            } else {
                local_row -= desc->z_out;
                if (local_row < desc->beta_out) {
                    const float raw = sums[0];
                    const float v = 1.0f / (1.0f + expf(-raw));
                    beta[local_row] = v;
                    if (write_outputs != 0 && desc->beta_math_dst != nullptr) {
                        desc->beta_math_dst[local_row] = raw;
                    }
                    if (write_outputs != 0 && desc->beta_raw_dst != nullptr) {
                        desc->beta_raw_dst[local_row] = raw;
                    }
                    if (write_outputs != 0 && desc->beta_dst != nullptr) {
                        desc->beta_dst[local_row] = v;
                    }
                } else {
                    local_row -= desc->beta_out;
                    const float raw = sums[0];
                    const float biased = raw + alpha_dt[local_row];
                    const float softplus = biased > 20.0f ? biased : log1pf(expf(biased));
                    const float v = softplus * alpha_a[local_row];
                    alpha[local_row] = v;
                    if (write_outputs != 0 && desc->alpha_math_dst != nullptr) {
                        desc->alpha_math_dst[local_row] = raw;
                    }
                    if (write_outputs != 0 && desc->alpha_raw_dst != nullptr) {
                        desc->alpha_raw_dst[local_row] = raw;
                    }
                    if (write_outputs != 0 && desc->alpha_biased_dst != nullptr) {
                        desc->alpha_biased_dst[local_row] = biased;
                    }
                    if (write_outputs != 0 && desc->alpha_softplus_dst != nullptr) {
                        desc->alpha_softplus_dst[local_row] = softplus;
                    }
                    if (write_outputs != 0 && desc->alpha_dst != nullptr) {
                        desc->alpha_dst[local_row] = v;
                    }
                }
            }
        }
        __syncthreads();
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
        l0_grid_scratch != nullptr && (l0_norm->has_norm_w == 0 || weightpack != nullptr);
    const uint64_t l0_qkv_requested =
        l0_qkv != nullptr && l0_qkv->ready != 0 ? (uint64_t) l0_qkv->n_out*sizeof(float) : 0;
    const uint64_t l0_qkv_begin = l0_qkv != nullptr ? l0_qkv->qkv_scratch_offset : 0;
    const uint64_t l0_qkv_end = l0_qkv_begin + l0_qkv_requested;
    const bool l0_qkv_ready =
        (l0_stage_mask & 0x2u) != 0 && l0_qkv_requested != 0 &&
        scratch != nullptr && weightpack != nullptr &&
        l0_qkv_end <= scratch_bytes;
    const uint64_t l0_proj_z_end =
        l0_proj != nullptr ? l0_proj->z_scratch_offset + (uint64_t) l0_proj->z_out*sizeof(float) : 0;
    const uint64_t l0_proj_beta_end =
        l0_proj != nullptr ? l0_proj->beta_scratch_offset + (uint64_t) l0_proj->beta_out*sizeof(float) : 0;
    const uint64_t l0_proj_alpha_end =
        l0_proj != nullptr ? l0_proj->alpha_scratch_offset + (uint64_t) l0_proj->alpha_out*sizeof(float) : 0;
    const bool l0_proj_ready =
        (l0_stage_mask & 0x4u) != 0 &&
        l0_proj != nullptr && l0_proj->ready != 0 && scratch != nullptr && weightpack != nullptr &&
        l0_proj->z_out != 0 && l0_proj->beta_out != 0 && l0_proj->alpha_out != 0 &&
        l0_proj_z_end <= scratch_bytes && l0_proj_beta_end <= scratch_bytes &&
        l0_proj_alpha_end <= scratch_bytes && l0_grid_scratch != nullptr;

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
                l0_proj, weightpack, scratch, scratch_bytes, l0_rms_sums, l0_stage_mask & 0x4u);
    }

    if (lane == 0) {
        state->fingerprint ^= 0x7900'0110'0036'0001ull;
        state->fingerprint ^= checksum;
        state->checksum = checksum ^
            (l0_norm_ready ? 0x10'0000'0000ull : 0ull) ^
            (l0_qkv_ready ? 0x20'0000'0000ull : 0ull) ^
            (l0_proj_ready ? 0x40'0000'0000ull : 0ull);
        state->n_blocks = gridDim.x;
        state->touched_layers = 40;
        state->active_lanes = gridDim.x*blockDim.x;
    }
}

static bool qwen36_superlayer_contract_dispatch_enabled() {
    return qwen36_superlayer_env_enabled("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_DISPATCH") ||
        qwen36_superlayer_env_enabled("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_SMOKE");
}

static bool qwen36_superlayer_replace_l0_all_requested() {
    return qwen36_superlayer_env_i64("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REPLACE_L0", 0) != 0;
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
        qwen36_superlayer_env_i64("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_REPLACE_L0_PROJ", 0) != 0;
}

static bool qwen36_superlayer_replace_l0_any_requested() {
    return qwen36_superlayer_replace_l0_rms_requested() ||
        qwen36_superlayer_replace_l0_qkv_requested() ||
        qwen36_superlayer_replace_l0_proj_requested();
}

static uint32_t qwen36_superlayer_l0_stage_mask() {
    if (qwen36_superlayer_env_enabled("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_RUN_L0")) {
        return 0x7u;
    }

    uint32_t mask = 0;
    if (qwen36_superlayer_replace_l0_rms_requested()) {
        mask |= 0x1u;
    }
    if (qwen36_superlayer_replace_l0_qkv_requested()) {
        mask |= 0x2u;
    }
    if (qwen36_superlayer_replace_l0_proj_requested()) {
        mask |= 0x4u;
    }
    return mask;
}

static bool qwen36_superlayer_run_l0_math_enabled() {
    return qwen36_superlayer_l0_stage_mask() != 0;
}

static bool qwen36_superlayer_contract_kernel_enabled() {
    return qwen36_superlayer_env_enabled("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_CONTRACT") ||
        qwen36_superlayer_run_l0_math_enabled();
}

} // namespace

bool ggml_cuda_rdna3_qwen36_superlayer_enabled(const int device) {
    return qwen36_superlayer_env_enabled("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER") &&
        GGML_CUDA_CC_IS_RDNA3(ggml_cuda_info().devices[device].cc);
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

    if (should_report) {
        GGML_LOG_INFO(
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
    }

    return true;
}

bool ggml_cuda_rdna3_qwen36_superlayer_maybe_launch_contract(
        ggml_backend_cuda_context * cuda_ctx,
        const ggml_cgraph * cgraph,
        std::string * blocker) {
    if (!qwen36_superlayer_contract_dispatch_enabled()) {
        return true;
    }
    if (!qwen36_superlayer_contract_kernel_enabled()) {
        static std::atomic<int64_t> skipped_reports{0};
        const int64_t report_id = skipped_reports.fetch_add(1, std::memory_order_relaxed);
        if (report_id < 4) {
            GGML_LOG_INFO(
                    "rdna3_qwen36_superlayer: contract-kernel-skipped"
                    " reason=no per-token superlayer work enabled l0_math=0 replace_l0=0\n");
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
    qwen36_superlayer_device_pack_view device_pack;
    if (!qwen36_superlayer_materialize_device_pack(cuda_ctx, cgraph, plan, pack, &device_pack, blocker)) {
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
    uint8_t * scratch_arg = (uint8_t *) device_pack.scratch;
    const uint64_t scratch_bytes_arg = (uint64_t) device_pack.runtime.scratch_bytes;
    const uint64_t weightpack_bytes_arg = (uint64_t) device_pack.bytes;
    const uint32_t l0_stage_mask_arg = qwen36_superlayer_l0_stage_mask();
    void * kernel_args[] = {
        (void *) &state_arg,
        (void *) &weightpack_arg,
        (void *) &layers_arg,
        (void *) &io_descs_arg,
        (void *) &io_count_arg,
        (void *) &l0_norm_arg,
        (void *) &l0_qkv_arg,
        (void *) &l0_proj_arg,
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
    if (report_id < 8) {
        GGML_LOG_INFO(
                "rdna3_qwen36_superlayer: contract-kernel-launched fingerprint=%s blocks=%d threads=%d"
                " weightpack_tensors=%zu runtime_bindings=%zu weightpack_bytes=%zu scratch_bytes=%zu"
                " l0_stage_mask=0x%x l0_rms_norm=%s l0_qkv=%s l0_projection=%s replace_l0=%d"
                " note=single physical cooperative 40-layer dispatch scaffold\n",
                hex_u64(plan.fingerprint).c_str(), blocks, threads,
                device_pack.tensors, device_pack.io_count, device_pack.bytes, device_pack.runtime.scratch_bytes,
                l0_stage_mask_arg,
                (l0_stage_mask_arg & 0x1u) != 0 ? "on" : "off",
                (l0_stage_mask_arg & 0x2u) != 0 ? "on" : "off",
                (l0_stage_mask_arg & 0x4u) != 0 ? "on" : "off",
                ggml_cuda_rdna3_qwen36_superlayer_replace_l0_enabled(cuda_ctx->device) ? 1 : 0);
    }

    return true;
}

bool ggml_cuda_rdna3_qwen36_superlayer_maybe_launch_smoke(
        ggml_backend_cuda_context * cuda_ctx,
        const ggml_cgraph * cgraph,
        std::string * blocker) {
    return ggml_cuda_rdna3_qwen36_superlayer_maybe_launch_contract(cuda_ctx, cgraph, blocker);
}
