#pragma once

#include "ggml-common.h"
#include "convert.cuh"

static __device__ __forceinline__ int best_index_int8(int n, const int8_t * val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu-ml > 1) {
        int mav = (ml+mu)/2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

static __device__ void quantize_f32_q4_0_block(const float * __restrict__ x, block_q4_0 * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_0; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -8;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    for (int j = 0; j < QK4_0/2; ++j) {
        const float x0 = x[0       + j]*id;
        const float x1 = x[QK4_0/2 + j]*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 8.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 8.5f));

        y->qs[j]  = xi0;
        y->qs[j] |= xi1 << 4;
    }
}

static __device__ void quantize_f32_q4_1_block(const float * __restrict__ x, block_q4_1 * __restrict__ y) {
    float vmin = FLT_MAX;
    float vmax = -FLT_MAX;

    for (int j = 0; j < QK4_1; ++j) {
        const float v = x[j];
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
    }

    const float d  = (vmax - vmin) / ((1 << 4) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->dm.x = d;
    y->dm.y = vmin;

    for (int j = 0; j < QK4_1/2; ++j) {
        const float x0 = (x[0       + j] - vmin)*id;
        const float x1 = (x[QK4_1/2 + j] - vmin)*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 0.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 0.5f));

        y->qs[j]  = xi0;
        y->qs[j] |= xi1 << 4;
    }
}

static __device__ void quantize_f32_q5_0_block(const float * __restrict__ x, block_q5_0 * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK5_0; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -16;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_0/2; ++j) {
        const float x0 = x[0       + j]*id;
        const float x1 = x[QK5_0/2 + j]*id;

        const uint8_t xi0 = min(31, (int8_t)(x0 + 16.5f));
        const uint8_t xi1 = min(31, (int8_t)(x1 + 16.5f));

        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

static __device__ void quantize_f32_q5_1_block(const float * __restrict__ x, block_q5_1 * __restrict__ y) {
    float min = x[0];
    float max = x[0];

    for (int j = 1; j < QK5_1; ++j) {
        const float v = x[j];
        min = v < min ? v : min;
        max = v > max ? v : max;
    }

    const float d  = (max - min) / 31;
    const float id = d ? 1.0f/d : 0.0f;

    y->dm.x = d;
    y->dm.y = min;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_1/2; ++j) {
        const float x0 = (x[0       + j] - min)*id;
        const float x1 = (x[QK5_1/2 + j] - min)*id;

        const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
        const uint8_t xi1 = (uint8_t)(x1 + 0.5f);

        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

static __device__ void quantize_f32_q8_0_block(const float * __restrict__ x, block_q8_0 * __restrict__ y) {
    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK8_0; j++) {
        const float v = x[j];
        amax = fmaxf(amax, fabsf(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    for (int j = 0; j < QK8_0; ++j) {
        const float x0 = x[j]*id;
        y->qs[j] = roundf(x0);
    }
}

static __device__ void quantize_f32_iq4_nl_block(const float * __restrict__ x, block_iq4_nl * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_NL; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    float d = vmax / kvalues_iq4nl[0];
    const float id = d ? 1.0f/d : 0.0f;

    float sumqx = 0, sumq2 = 0;
    for (int j = 0; j < QK4_NL/2; ++j) {
        const float x0 = x[0        + j]*id;
        const float x1 = x[QK4_NL/2 + j]*id;
        const uint8_t xi0 = best_index_int8(16, kvalues_iq4nl, x0);
        const uint8_t xi1 = best_index_int8(16, kvalues_iq4nl, x1);
        y->qs[j] = xi0 | (xi1 << 4);
        const float v0 = kvalues_iq4nl[xi0];
        const float v1 = kvalues_iq4nl[xi1];
        const float w0 = x[0        + j]*x[0        + j];
        const float w1 = x[QK4_NL/2 + j]*x[QK4_NL/2 + j];
        sumqx += w0*v0*x[j] + w1*v1*x[QK4_NL/2 + j];
        sumq2 += w0*v0*v0 + w1*v1*v1;
    }

    y->d = sumq2 > 0 ? sumqx/sumq2 : d;
}

static __device__ __forceinline__ void tqkv_set_bit_device(uint8_t * q, int i, bool v) {
    if (v) {
        q[i >> 3] |= uint8_t(1u << (i & 7));
    }
}

static __device__ __forceinline__ void tqkv_set_bits_device(uint8_t * q, int bit, int nbits, uint8_t v) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        if (i < nbits && ((v >> i) & 1u)) {
            tqkv_set_bit_device(q, bit + i, true);
        }
    }
}

template<typename block_t, int bits, bool half_residual, bool ip_residual>
static __device__ void quantize_f32_tqkv_block(const float * __restrict__ x, block_t * __restrict__ y) {
    for (int i = 0; i < int(sizeof(y->qs)); ++i) {
        y->qs[i] = 0;
    }
    if constexpr (half_residual) {
        for (int i = 0; i < int(sizeof(y->qh)); ++i) {
            y->qh[i] = 0;
        }
    }
    if constexpr (ip_residual) {
        for (int i = 0; i < int(sizeof(y->rs)); ++i) {
            y->rs[i] = 0;
        }
    }

    float amax = 0.0f;
    for (int j = 0; j < QK_TQKV; ++j) {
        amax = fmaxf(amax, fabsf(x[j]));
    }

    constexpr int levels = 1 << bits;
    constexpr float center = 0.5f * float(levels - 1);
    const float d = amax > 0.0f ? amax / center : 0.0f;
    const float id = d ? 1.0f/d : 0.0f;

    float yv[QK_TQKV];
    for (int j = 0; j < QK_TQKV; ++j) {
        int q = int(roundf(x[j]*id + center));
        q = max(0, min(levels - 1, q));
        tqkv_set_bits_device(y->qs, j*bits, bits, uint8_t(q));
        yv[j] = (float(q) - center) * d;
    }

    if constexpr (half_residual) {
        for (int j = 0; j < QK_TQKV; j += 2) {
            const float y_pos = yv[j] + 0.25f*d;
            const float y_neg = yv[j] - 0.25f*d;
            const bool use_pos = fabsf(x[j] - y_pos) < fabsf(x[j] - y_neg);
            tqkv_set_bit_device(y->qh, j >> 1, use_pos);
            yv[j] = use_pos ? y_pos : y_neg;
        }
    }

    float dr = 0.0f;
    if constexpr (ip_residual) {
        for (int j = 0; j < QK_TQKV; ++j) {
            dr += fabsf(x[j] - yv[j]);
        }
        dr /= QK_TQKV;

        for (int j = 0; j < QK_TQKV; ++j) {
            tqkv_set_bit_device(y->rs, j, x[j] >= yv[j]);
        }
    }

    y->d = d;
    if constexpr (ip_residual) {
        y->dr = dr;
    }
}

static __device__ void quantize_f32_tqkv_2_0_block(const float * __restrict__ x, block_tqkv_2_0 * __restrict__ y) {
    quantize_f32_tqkv_block<block_tqkv_2_0, 2, false, false>(x, y);
}

static __device__ void quantize_f32_tqkv_2_5_block(const float * __restrict__ x, block_tqkv_2_5 * __restrict__ y) {
    quantize_f32_tqkv_block<block_tqkv_2_5, 2, true, false>(x, y);
}

static __device__ void quantize_f32_tqkv_3_0_block(const float * __restrict__ x, block_tqkv_3_0 * __restrict__ y) {
    quantize_f32_tqkv_block<block_tqkv_3_0, 3, false, false>(x, y);
}

static __device__ void quantize_f32_tqkv_3_5_block(const float * __restrict__ x, block_tqkv_3_5 * __restrict__ y) {
    quantize_f32_tqkv_block<block_tqkv_3_5, 3, true, false>(x, y);
}

static __device__ void quantize_f32_tqkv_4_0_block(const float * __restrict__ x, block_tqkv_4_0 * __restrict__ y) {
    quantize_f32_tqkv_block<block_tqkv_4_0, 4, false, false>(x, y);
}

static __device__ void quantize_f32_tqkv_2_0_ip_block(const float * __restrict__ x, block_tqkv_2_0_ip * __restrict__ y) {
    quantize_f32_tqkv_block<block_tqkv_2_0_ip, 2, false, true>(x, y);
}

static __device__ void quantize_f32_tqkv_2_5_ip_block(const float * __restrict__ x, block_tqkv_2_5_ip * __restrict__ y) {
    quantize_f32_tqkv_block<block_tqkv_2_5_ip, 2, true, true>(x, y);
}

static __device__ void quantize_f32_tqkv_3_0_ip_block(const float * __restrict__ x, block_tqkv_3_0_ip * __restrict__ y) {
    quantize_f32_tqkv_block<block_tqkv_3_0_ip, 3, false, true>(x, y);
}

static __device__ void quantize_f32_tqkv_3_5_ip_block(const float * __restrict__ x, block_tqkv_3_5_ip * __restrict__ y) {
    quantize_f32_tqkv_block<block_tqkv_3_5_ip, 3, true, true>(x, y);
}

static __device__ void quantize_f32_tqkv_4_0_ip_block(const float * __restrict__ x, block_tqkv_4_0_ip * __restrict__ y) {
    quantize_f32_tqkv_block<block_tqkv_4_0_ip, 4, false, true>(x, y);
}

// Wrapper functions for cpy.cu compatibility
static __device__ void cpy_blck_f32_q4_0(const char * cxi, char * cdsti) {
    quantize_f32_q4_0_block((const float *)cxi, (block_q4_0 *)cdsti);
}

static __device__ void cpy_blck_f32_q4_1(const char * cxi, char * cdsti) {
    quantize_f32_q4_1_block((const float *)cxi, (block_q4_1 *)cdsti);
}

static __device__ void cpy_blck_f32_q5_0(const char * cxi, char * cdsti) {
    quantize_f32_q5_0_block((const float *)cxi, (block_q5_0 *)cdsti);
}

static __device__ void cpy_blck_f32_q5_1(const char * cxi, char * cdsti) {
    quantize_f32_q5_1_block((const float *)cxi, (block_q5_1 *)cdsti);
}

static __device__ void cpy_blck_f32_q8_0(const char * cxi, char * cdsti) {
    quantize_f32_q8_0_block((const float *)cxi, (block_q8_0 *)cdsti);
}

static __device__ void cpy_blck_f32_iq4_nl(const char * cxi, char * cdsti) {
    quantize_f32_iq4_nl_block((const float *)cxi, (block_iq4_nl *)cdsti);
}

template<typename src_t, typename dst_t>
static __device__ void cpy_1_scalar(const char * cxi, char * cdsti) {
    *(dst_t *) cdsti = ggml_cuda_cast<dst_t>(*(const src_t *) cxi);
}
