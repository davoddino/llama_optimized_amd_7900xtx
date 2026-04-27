#include "common.cuh"

static __device__ __forceinline__ void dequantize_q1_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q1_0 * x = (const block_q1_0 *) vx;

    const float d = x[ib].d;

    const int bit_index_0 = iqs;
    const int bit_index_1 = iqs + 1;

    const int byte_index_0 = bit_index_0 / 8;
    const int bit_offset_0 = bit_index_0 % 8;

    const int byte_index_1 = bit_index_1 / 8;
    const int bit_offset_1 = bit_index_1 % 8;

    // Extract bits: 1 = +d, 0 = -d (branchless)
    const int bit_0 = (x[ib].qs[byte_index_0] >> bit_offset_0) & 1;
    const int bit_1 = (x[ib].qs[byte_index_1] >> bit_offset_1) & 1;

    v.x = (2*bit_0 - 1) * d;
    v.y = (2*bit_1 - 1) * d;
}

static __device__ __forceinline__ void dequantize_q4_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const float d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x - 8.0f) * d;
    v.y = (v.y - 8.0f) * d;
}

static __device__ __forceinline__ void dequantize_q4_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q5_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const float d = x[ib].d;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x - 16.0f) * d;
    v.y = (v.y - 16.0f) * d;
}

static __device__ __forceinline__ void dequantize_q5_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ bool dequantize_tqkv_get_bit(const uint8_t * q, int i) {
    return (q[i >> 3] >> (i & 7)) & 1u;
}

static __device__ __forceinline__ uint8_t dequantize_tqkv_get_bits(const uint8_t * q, int bit, int nbits) {
    uint8_t v = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        if (i < nbits) {
            v |= uint8_t(dequantize_tqkv_get_bit(q, bit + i)) << i;
        }
    }
    return v;
}

template<typename block_t, int bits, bool half_residual, bool ip_residual>
static __device__ __forceinline__ float dequantize_tqkv_scalar(const block_t * x, int64_t ib, int j) {
    constexpr int levels = 1 << bits;
    constexpr float center = 0.5f * float(levels - 1);

    const uint8_t q = dequantize_tqkv_get_bits(x[ib].qs, j*bits, bits);
    const float d = x[ib].d;

    float v = (float(q) - center) * d;

    if constexpr (half_residual) {
        if ((j & 1) == 0) {
            v += (dequantize_tqkv_get_bit(x[ib].qh, j >> 1) ? 0.25f : -0.25f) * d;
        }
    }

    if constexpr (ip_residual) {
        const float dr = x[ib].dr;
        v += dequantize_tqkv_get_bit(x[ib].rs, j) ? dr : -dr;
    }

    return v;
}

template<typename block_t, int bits, bool half_residual, bool ip_residual>
static __device__ __forceinline__ void dequantize_tqkv(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_t * x = (const block_t *) vx;

    v.x = dequantize_tqkv_scalar<block_t, bits, half_residual, ip_residual>(x, ib, iqs + 0);
    v.y = dequantize_tqkv_scalar<block_t, bits, half_residual, ip_residual>(x, ib, iqs + 1);
}

static __device__ __forceinline__ void dequantize_tqkv_2_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    dequantize_tqkv<block_tqkv_2_0, 2, false, false>(vx, ib, iqs, v);
}

static __device__ __forceinline__ void dequantize_tqkv_2_5(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    dequantize_tqkv<block_tqkv_2_5, 2, true, false>(vx, ib, iqs, v);
}

static __device__ __forceinline__ void dequantize_tqkv_3_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    dequantize_tqkv<block_tqkv_3_0, 3, false, false>(vx, ib, iqs, v);
}

static __device__ __forceinline__ void dequantize_tqkv_3_5(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    dequantize_tqkv<block_tqkv_3_5, 3, true, false>(vx, ib, iqs, v);
}

static __device__ __forceinline__ void dequantize_tqkv_4_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    dequantize_tqkv<block_tqkv_4_0, 4, false, false>(vx, ib, iqs, v);
}

static __device__ __forceinline__ void dequantize_tqkv_2_0_ip(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    dequantize_tqkv<block_tqkv_2_0_ip, 2, false, true>(vx, ib, iqs, v);
}

static __device__ __forceinline__ void dequantize_tqkv_2_5_ip(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    dequantize_tqkv<block_tqkv_2_5_ip, 2, true, true>(vx, ib, iqs, v);
}

static __device__ __forceinline__ void dequantize_tqkv_3_0_ip(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    dequantize_tqkv<block_tqkv_3_0_ip, 3, false, true>(vx, ib, iqs, v);
}

static __device__ __forceinline__ void dequantize_tqkv_3_5_ip(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    dequantize_tqkv<block_tqkv_3_5_ip, 3, true, true>(vx, ib, iqs, v);
}

static __device__ __forceinline__ void dequantize_tqkv_4_0_ip(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    dequantize_tqkv<block_tqkv_4_0_ip, 4, false, true>(vx, ib, iqs, v);
}

static __device__ __forceinline__ void dequantize_q8_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const float d = x[ib].d;

    v.x = x[ib].qs[iqs + 0];
    v.y = x[ib].qs[iqs + 1];

    v.x *= d;
    v.y *= d;
}
