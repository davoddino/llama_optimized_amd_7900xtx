#include "rdna3-qwen36-superlayer.cuh"

#include "ggml-cuda/common.cuh"

#include <algorithm>
#include <array>
#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_set>

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

static std::string hex_u64(const uint64_t value) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%016" PRIx64, value);
    return std::string(buf);
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

    if (!plan.has_decode_out || !plan.is_decode_token) {
        plan.blocker = "not a one-token decoder graph ending in result_output";
    } else if (layer_count != 40) {
        plan.blocker = "Qwen3.6 superlayer requires exactly 40 named layer spans";
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

static std::string qwen36_superlayer_layout_source(const qwen36_superlayer_plan & plan, const int device) {
    std::ostringstream out;
    out << "#pragma once\n\n";
    out << "#define QWEN36_RDNA3_SUPERLAYER_FINGERPRINT 0x" << hex_u64(plan.fingerprint) << "ull\n";
    out << "#define QWEN36_RDNA3_SUPERLAYER_DEVICE_CC " << ggml_cuda_info().devices[device].cc << "\n";
    out << "#define QWEN36_RDNA3_SUPERLAYER_DEVICE_CU " << ggml_cuda_info().devices[device].nsm << "\n";
    out << "#define QWEN36_RDNA3_SUPERLAYER_LOGICAL_LAYERS 40\n\n";
    out << "enum qwen36_rdna3_layer_kind {\n";
    out << "    QWEN36_LAYER_ATTN = 0,\n";
    out << "    QWEN36_LAYER_GDN  = 1,\n";
    out << "};\n\n";
    out << "static constexpr qwen36_rdna3_layer_kind QWEN36_LAYER_KINDS[40] = {\n";
    for (int layer = 0; layer < 40; ++layer) {
        out << "    " << (plan.layer_recurrent[layer] ? "QWEN36_LAYER_GDN" : "QWEN36_LAYER_ATTN")
            << (layer == 39 ? "\n" : ",\n");
    }
    out << "};\n";
    return out.str();
}

static std::string qwen36_superlayer_generated_source(const qwen36_superlayer_plan & plan) {
    std::ostringstream out;
    out << "// Generated scaffold for the RDNA3 Qwen3.6 physical superlayer.\n";
    out << "// This file is intentionally topology-specific. It is not a ggml graph executor.\n\n";
    out << "#include \"layout.generated.h\"\n\n";
    out << "struct qwen36_superlayer_runtime;\n\n";
    for (int layer = 0; layer < 40; ++layer) {
        out << "__device__ __forceinline__ void qwen36_l" << (layer < 10 ? "0" : "") << layer
            << "_" << (plan.layer_recurrent[layer] ? "gdn" : "attn")
            << "_fused(qwen36_superlayer_runtime &) {\n";
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

static std::string qwen36_superlayer_manifest_json(const qwen36_superlayer_plan & plan, const int device) {
    std::ostringstream out;
    out << "{\n";
    out << "  \"artifact\": \"qwen36-rdna3-7900xtx-superlayer\",\n";
    out << "  \"state\": \"scaffold\",\n";
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
    out << "  \"fusion_contract\": \"load-time generated physical superlayer; no generic layer loop in generated source\"\n";
    out << "}\n";
    return out.str();
}

static std::string qwen36_superlayer_weightpack_plan(const qwen36_superlayer_plan & plan) {
    std::ostringstream out;
    out << "# RDNA3 Qwen3.6 7900 XTX Weight Pack Plan\n\n";
    out << "fingerprint: " << hex_u64(plan.fingerprint) << "\n";
    out << "state: scaffold\n\n";
    out << "The final artifact must replace this plan with a physical weight pack ordered for gfx1100 wave32 access.\n";
    out << "Layer spans detected from the decode graph:\n\n";
    for (int layer = 0; layer < 40; ++layer) {
        out << "- layer " << layer << ": "
            << (plan.layer_recurrent[layer] ? "gdn" : "attention")
            << " nodes [" << plan.layer_start[layer] << ", " << plan.layer_end[layer] << "]\n";
    }
    return out.str();
}

static bool qwen36_superlayer_materialize_artifact(const qwen36_superlayer_plan & plan, const int device, std::string * blocker) {
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

    if (!write_text_file(manifest_path, qwen36_superlayer_manifest_json(plan, device), blocker)) {
        return false;
    }
    if (!write_text_file(layout_path, qwen36_superlayer_layout_source(plan, device), blocker)) {
        return false;
    }
    if (!write_text_file(source_path, qwen36_superlayer_generated_source(plan), blocker)) {
        return false;
    }
    if (!write_text_file(pack_path, qwen36_superlayer_weightpack_plan(plan), blocker)) {
        return false;
    }

    return true;
}

struct qwen36_superlayer_smoke_state {
    uint64_t fingerprint;
    uint32_t n_nodes;
    uint32_t n_blocks;
};

__global__ void qwen36_rdna3_superlayer_smoke_kernel(qwen36_superlayer_smoke_state * state) {
    const uint32_t tid = (uint32_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (tid == 0) {
        state->fingerprint ^= 0x7900'0110'0036'0001ull;
        state->n_blocks = gridDim.x;
    }
}

static bool qwen36_superlayer_smoke_enabled() {
    return qwen36_superlayer_env_enabled("GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_SMOKE");
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

    static std::mutex prepared_mutex;
    static std::unordered_set<uint64_t> prepared_fingerprints;
    static std::unordered_set<uint64_t> reported_fingerprints;

    bool already_prepared = false;
    {
        std::lock_guard<std::mutex> lock(prepared_mutex);
        already_prepared = prepared_fingerprints.find(plan.fingerprint) != prepared_fingerprints.end();
    }

    if (!already_prepared) {
        if (!qwen36_superlayer_materialize_artifact(plan, cuda_ctx->device, blocker)) {
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
                " state=scaffold\n",
                hex_u64(plan.fingerprint).c_str(), plan.artifact_dir.string().c_str(), plan.n_nodes,
                plan.n_fattn, plan.n_gdn, plan.n_topk_moe, plan.n_moe_gate_up, plan.n_moe_down, plan.n_mmid);
    }

    return true;
}

bool ggml_cuda_rdna3_qwen36_superlayer_maybe_launch_smoke(
        ggml_backend_cuda_context * cuda_ctx,
        const ggml_cgraph * cgraph,
        std::string * blocker) {
    if (!qwen36_superlayer_smoke_enabled()) {
        return true;
    }

    qwen36_superlayer_plan plan = qwen36_superlayer_make_plan(cgraph, cuda_ctx->device);
    if (!plan.blocker.empty()) {
        if (blocker != nullptr) {
            *blocker = plan.blocker;
        }
        return false;
    }

    ggml_cuda_pool_alloc<qwen36_superlayer_smoke_state> smoke_state(cuda_ctx->pool(), 1);
    qwen36_superlayer_smoke_state host_state = {
        /*.fingerprint =*/ plan.fingerprint,
        /*.n_nodes     =*/ (uint32_t) plan.n_nodes,
        /*.n_blocks    =*/ 0,
    };
    CUDA_CHECK(cudaMemcpyAsync(smoke_state.ptr, &host_state, sizeof(host_state), cudaMemcpyHostToDevice, cuda_ctx->stream()));

    const int default_blocks = std::max(1, ggml_cuda_info().devices[cuda_ctx->device].nsm);
    const int blocks = (int) std::max<int64_t>(1, qwen36_superlayer_env_i64(
            "GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_BLOCKS", default_blocks));
    const int threads = (int) std::max<int64_t>(32, qwen36_superlayer_env_i64(
            "GGML_CUDA_RDNA3_QWEN36_SUPERLAYER_THREADS", 256));

    qwen36_rdna3_superlayer_smoke_kernel<<<blocks, threads, 0, cuda_ctx->stream()>>>(smoke_state.ptr);
    CUDA_CHECK(cudaGetLastError());

    static std::atomic<int64_t> smoke_reports{0};
    const int64_t report_id = smoke_reports.fetch_add(1, std::memory_order_relaxed);
    if (report_id < 8) {
        GGML_LOG_INFO(
                "rdna3_qwen36_superlayer: smoke-kernel-launched fingerprint=%s blocks=%d threads=%d"
                " note=single physical dispatch scaffold\n",
                hex_u64(plan.fingerprint).c_str(), blocks, threads);
    }

    return true;
}
