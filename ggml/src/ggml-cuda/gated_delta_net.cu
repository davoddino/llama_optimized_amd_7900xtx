#include "gated_delta_net.cuh"

#include <cinttypes>
#include <cstdlib>
#include <type_traits>

static int ggml_cuda_rdna3_gdn_ar_cols() {
    int cols_per_block = 8;
    if (const char * env = getenv("GGML_CUDA_RDNA3_GDN_AR_COLS")) {
        const int requested = atoi(env);
        if (requested == 4 || requested == 8 || requested == 16) {
            cols_per_block = requested;
        }
    }

    return cols_per_block;
}

static bool ggml_cuda_gated_delta_net_uses_ar_tiled(const int device, const ggml_tensor * dst) {
    if (dst == nullptr || dst->src[2] == nullptr) {
        return false;
    }

    const ggml_tensor * src_v = dst->src[2];
    return GGML_CUDA_CC_IS_RDNA3(ggml_cuda_info().devices[device].cc) &&
        getenv("GGML_CUDA_RDNA3_DISABLE_GDN_AR_TILED") == nullptr &&
        src_v->ne[0] == 128 && src_v->ne[2] == 1;
}

static const char * ggml_cuda_gated_delta_net_ar_tiled_name(const bool kda, const int cols_per_block) {
    if (kda) {
        switch (cols_per_block) {
            case 4:
                return "GDN_AR_TILED_KDA_C4";
            case 16:
                return "GDN_AR_TILED_KDA_C16";
            case 8:
            default:
                return "GDN_AR_TILED_KDA_C8";
        }
    }

    switch (cols_per_block) {
        case 4:
            return "GDN_AR_TILED_C4";
        case 16:
            return "GDN_AR_TILED_C16";
        case 8:
        default:
            return "GDN_AR_TILED_C8";
    }
}

const char * ggml_cuda_gated_delta_net_kernel_name(int device, const ggml_tensor * dst) {
    if (ggml_cuda_gated_delta_net_uses_ar_tiled(device, dst)) {
        const ggml_tensor * src_v = dst->src[2];
        const ggml_tensor * src_g = dst->src[3];
        const bool kda = src_g != nullptr && src_g->ne[0] == src_v->ne[0];
        return ggml_cuda_gated_delta_net_ar_tiled_name(kda, ggml_cuda_rdna3_gdn_ar_cols());
    }

    return "GATED_DELTA_NET";
}

template <int S_v, bool KDA, int COLS_PER_BLOCK>
__global__ void __launch_bounds__(ggml_cuda_get_physical_warp_size() * COLS_PER_BLOCK, COLS_PER_BLOCK <= 8 ? 2 : 1)
gated_delta_net_ar_tiled_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state_out,
                                     int64_t       H,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    static_assert(S_v == 128, "RDNA3 tiled autoregressive GDN currently targets S_v=128");
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    static_assert(S_v % COLS_PER_BLOCK == 0, "S_v must be a multiple of COLS_PER_BLOCK");

    const int h_idx    = blockIdx.x;
    const int sequence = blockIdx.y;
    const int lane     = threadIdx.x;
    const int col_lane = threadIdx.y;
    const int col      = blockIdx.z * COLS_PER_BLOCK + col_lane;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    constexpr int rows_per_lane = S_v / warp_size;

    __shared__ float q_shared[S_v];
    __shared__ float k_shared[S_v];
    __shared__ float g_shared[S_v];
    __shared__ float g_scalar_shared;
    __shared__ float kq_scalar_shared;

#pragma unroll
    for (int r = 0; r < rows_per_lane; ++r) {
        const int i = r * warp_size + lane;
        if (col_lane == 0) {
            const float * q_t = q + iq3 * sq3 + iq1 * sq1;
            const float * k_t = k + iq3 * sq3 + iq1 * sq1;
            q_shared[i] = q_t[i];
            k_shared[i] = k_t[i];
        }
    }

    const int64_t gb_offset = sequence * sb3 + h_idx * sb1;
    const float * g_t       = g + gb_offset * (KDA ? S_v : 1);

    if constexpr (KDA) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; ++r) {
            const int i = r * warp_size + lane;
            if (col_lane == 0) {
                g_shared[i] = expf(g_t[i]);
            }
        }
    } else {
        if (lane == 0 && col_lane == 0) {
            g_scalar_shared = expf(*g_t);
        }
    }
    __syncthreads();

    float * attn_data = dst;
    float * state     = state_out;

    const int64_t state_offset = (sequence * H + h_idx) * S_v * S_v;
    state      += state_offset;
    curr_state += state_offset + col * S_v;
    attn_data  += (sequence * H + h_idx) * S_v;

    const float * v_t = v + sequence * sv3 + h_idx * sv1;
    const float beta_val = beta[gb_offset];

    float s_shard[rows_per_lane];
#pragma unroll
    for (int r = 0; r < rows_per_lane; ++r) {
        const int i = r * warp_size + lane;
        s_shard[r] = curr_state[i];
    }

    if constexpr (!KDA) {
        const float g_val = g_scalar_shared;

        float2 sk_sq_shard = make_float2(0.0f, 0.0f);
        float kq_shard = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; ++r) {
            const int i = r * warp_size + lane;
            const float k_val = k_shared[i];
            const float q_val = q_shared[i];

            sk_sq_shard.x += s_shard[r] * k_val;
            sk_sq_shard.y += s_shard[r] * q_val;

            if (col_lane == 0) {
                kq_shard += k_val * q_val;
            }
        }

        if (col_lane == 0) {
            const float kq_col = warp_reduce_sum<warp_size>(kq_shard);
            if (lane == 0) {
                kq_scalar_shared = kq_col;
            }
        }

        const float2 sk_sq_col = warp_reduce_sum<warp_size>(sk_sq_shard);
        const float delta_col = (v_t[col] - g_val * sk_sq_col.x) * beta_val;

        __syncthreads();

        if (lane == 0) {
            const float attn_col = g_val * sk_sq_col.y + delta_col * kq_scalar_shared;
            attn_data[col] = attn_col * scale;
        }

#pragma unroll
        for (int r = 0; r < rows_per_lane; ++r) {
            const int i = r * warp_size + lane;
            s_shard[r] = g_val * s_shard[r] + k_shared[i] * delta_col;
        }
    } else {
        float kv_shard = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; ++r) {
            const int i = r * warp_size + lane;
            kv_shard += g_shared[i] * s_shard[r] * k_shared[i];
        }

        const float kv_col    = warp_reduce_sum<warp_size>(kv_shard);
        const float delta_col = (v_t[col] - kv_col) * beta_val;

        float attn_partial = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; ++r) {
            const int i = r * warp_size + lane;
            s_shard[r] = g_shared[i] * s_shard[r] + k_shared[i] * delta_col;
            attn_partial += s_shard[r] * q_shared[i];
        }

        const float attn_col = warp_reduce_sum<warp_size>(attn_partial);
        if (lane == 0) {
            attn_data[col] = attn_col * scale;
        }
    }

#pragma unroll
    for (int r = 0; r < rows_per_lane; ++r) {
        const int i = r * warp_size + lane;
        state[col * S_v + i] = s_shard[r];
    }
}

template <int S_v, bool KDA, int N_WARPS, bool CACHE_G>
__global__ void __launch_bounds__(
        (ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * N_WARPS,
        N_WARPS <= 4 ? 2 : 1)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state_out,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    static_assert(S_v % N_WARPS == 0, "S_v must be a multiple of N_WARPS");

    float * attn_data = dst;
    float * state     = state_out;

    const int64_t state_offset = (sequence * H + h_idx) * S_v * S_v;
    state += state_offset;
    curr_state += state_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    __shared__ float g_shared[S_v];
    __shared__ float g_scalar_shared;

#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        if constexpr (CACHE_G) {
            if constexpr (KDA) {
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    if (threadIdx.y == 0) {
                        g_shared[i] = expf(g_t[i]);
                    }
                }
            } else if (threadIdx.x == 0 && threadIdx.y == 0) {
                g_scalar_shared = expf(*g_t);
            }
            __syncthreads();
        }

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = CACHE_G ? g_scalar_shared : expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += (CACHE_G ? g_shared[i] : expf(g_t[i])) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = (CACHE_G ? g_shared[i] : expf(g_t[i])) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;
    }

    // Write state back to global memory (transposed layout)
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i          = r * warp_size + lane;
        state[col * S_v + i] = s_shard[r];
    }
}

template <bool KDA>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_out_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    if (GGML_CUDA_CC_IS_RDNA3(cc) && n_tokens == 1 && S_v == 128 &&
            getenv("GGML_CUDA_RDNA3_DISABLE_GDN_AR_TILED") == nullptr) {
        const int cols_per_block = ggml_cuda_rdna3_gdn_ar_cols();

        if (getenv("GGML_CUDA_RDNA3_PROFILE_LOG") != nullptr) {
            GGML_LOG_INFO("launch_gated_delta_net: GDN_AR_TILED S_v=%" PRId64 ", KDA=%d, cols/block=%d, H=%" PRId64
                    ", seqs=%" PRId64 "\n", S_v, int(KDA), cols_per_block, H, n_seqs);
        }

        auto launch_tiled = [&](auto cols_tag) {
            constexpr int cols_per_block_ct = decltype(cols_tag)::value;
            const dim3 grid_dims(H, n_seqs, S_v / cols_per_block_ct);
            const dim3 block_dims(warp_size, cols_per_block_ct, 1);

            gated_delta_net_ar_tiled_cuda<128, KDA, cols_per_block_ct><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_out_d, H, n_seqs,
                sq1, sq3, sv1, sv3, sb1, sb3, neqk1_magic, rq3_magic, scale);
        };

        switch (cols_per_block) {
            case 4:
                launch_tiled(std::integral_constant<int, 4>{});
                return;
            case 16:
                launch_tiled(std::integral_constant<int, 16>{});
                return;
            case 8:
            default:
                launch_tiled(std::integral_constant<int, 8>{});
                return;
        }
    }

    int num_warps = 4;
    if (GGML_CUDA_CC_IS_RDNA3(cc) && n_tokens == 1 && S_v >= 64) {
        num_warps = 8;
    }
    if (const char * env = getenv("GGML_CUDA_RDNA3_GDN_WARPS")) {
        const int requested = atoi(env);
        if (requested == 4 || requested == 8) {
            num_warps = requested;
        }
    }

    if (getenv("GGML_CUDA_RDNA3_PROFILE_LOG") != nullptr) {
        GGML_LOG_INFO("launch_gated_delta_net: GATED_DELTA_NET S_v=%" PRId64 ", KDA=%d, warps=%d, cache_g=%d, H=%" PRId64
                ", tokens=%" PRId64 ", seqs=%" PRId64 "\n",
            S_v, int(KDA), num_warps, int(n_tokens == 1), H, n_tokens, n_seqs);
    }

    auto launch = [&](auto warps_tag, auto cache_g_tag) {
        constexpr int num_warps_ct = decltype(warps_tag)::value;
        constexpr bool cache_g_ct  = decltype(cache_g_tag)::value;

        dim3 grid_dims(H, n_seqs, (S_v + num_warps_ct - 1) / num_warps_ct);
        dim3 block_dims(warp_size <= S_v ? warp_size : S_v, num_warps_ct, 1);

        switch (S_v) {
            case 16:
                gated_delta_net_cuda<16, KDA, num_warps_ct, cache_g_ct><<<grid_dims, block_dims, 0, stream>>>(
                    q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_out_d, H,
                    n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
                break;
            case 32:
                gated_delta_net_cuda<32, KDA, num_warps_ct, cache_g_ct><<<grid_dims, block_dims, 0, stream>>>(
                    q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_out_d, H,
                    n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
                break;
            case 64:
                gated_delta_net_cuda<64, KDA, num_warps_ct, cache_g_ct><<<grid_dims, block_dims, 0, stream>>>(
                    q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_out_d, H,
                    n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
                break;
            case 128:
                gated_delta_net_cuda<128, KDA, num_warps_ct, cache_g_ct><<<grid_dims, block_dims, 0, stream>>>(
                    q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_out_d, H,
                    n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
                break;
            default:
                GGML_ABORT("fatal error");
                break;
        }
    };

    if (num_warps == 8) {
        if (n_tokens == 1) {
            launch(std::integral_constant<int, 8>{}, std::true_type{});
        } else {
            launch(std::integral_constant<int, 8>{}, std::false_type{});
        }
    } else {
        if (n_tokens == 1) {
            launch(std::integral_constant<int, 4>{}, std::true_type{});
        } else {
            launch(std::integral_constant<int, 4>{}, std::false_type{});
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;
    const bool    state_inplace = ggml_get_op_params_i32(dst, 0) != 0;
    ggml_tensor * src_state_out = state_inplace ? dst->src[6] : nullptr;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));
    GGML_ASSERT(!state_inplace || src_state_out != nullptr);
    GGML_ASSERT(!state_inplace || ggml_is_contiguous(src_state_out));
    GGML_ASSERT(!state_inplace || ggml_nelements(src_state_out) == ggml_nelements(src_state));

    float * state_out_d = state_inplace ?
        (float *) src_state_out->data :
        dst_d + S_v * H * n_tokens * n_seqs;

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    if (kda) {
        launch_gated_delta_net<true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_out_d,
            S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
            sb1, sb2, sb3, neqk1, rq3, scale, stream);
    } else {
        launch_gated_delta_net<false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_out_d,
            S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
            sb1, sb2, sb3, neqk1, rq3, scale, stream);
    }
}
