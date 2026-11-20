// Fused affine 3-bit and 4-bit (MLX group_size=64) dequantization GEMV and
// GEMM kernels (G1, G2). Specialized at startup via Metal function constants
// so hot loops carry zero bit-width branching (ADR-0014).
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant uint Q_BITS [[function_constant(0)]];
constant uint Q_GROUP_SIZE [[function_constant(1)]];

struct DequantParams {
    uint m;
    uint n;
    uint k;
    uint num_groups;
    uint row_bytes;
};

// Accumulate 8 affine-quantized weights from `w_ptr` against prescaled x0/x1.
// `x0`/`x1` hold `x[0..7]` already scaled by exact powers of two for Q_BITS==4,
// or raw float4 values for Q_BITS==3 (handled directly via exact bit-field scaling).
static inline float qdot8_q4(uint32_t w, float4 px0, float4 px1) {
    uint lo = w & 0xFFFFu;
    uint hi = w >> 16;
    return fma(float(lo & 0x000Fu), px0.x,
           fma(float(lo & 0x00F0u), px0.y,
           fma(float(lo & 0x0F00u), px0.z,
           fma(float(lo & 0xF000u), px0.w,
           fma(float(hi & 0x000Fu), px1.x,
           fma(float(hi & 0x00F0u), px1.y,
           fma(float(hi & 0x0F00u), px1.z,
               float(hi & 0xF000u) * px1.w)))))));
}

static inline float qdot8_q3(uchar b0, uchar b1, uchar b2,
                             float4 px0, float4 px1,
                             float x2_hi, float x5_hi) {
    float acc = 0.0f;
    acc = fma(float(b0 & 0x07u), px0.x, acc);
    acc = fma(float(b0 & 0x38u), px0.y, acc);
    acc = fma(float(b0 & 0xC0u), px0.z, acc);
    acc = fma(float(b1 & 0x01u), x2_hi, acc);
    acc = fma(float(b1 & 0x0Eu), px0.w, acc);
    acc = fma(float(b1 & 0x70u), px1.x, acc);
    acc = fma(float(b1 & 0x80u), px1.y, acc);
    acc = fma(float(b2 & 0x03u), x5_hi, acc);
    acc = fma(float(b2 & 0x1Cu), px1.z, acc);
    acc = fma(float(b2 & 0xE0u), px1.w, acc);
    return acc;
}

// ---------------------------------------------------------------------------
// Kernel 1: Fused Dequant GEMV (M = 1 decode hot path)
//
// Geometry: 4 simdgroups/threadgroup (128 threads), 4 output rows/simdgroup
// (16 rows/threadgroup). Each 32-lane simdgroup processes 4 groups of 64
// weights (256 elements) per K-step (8 lanes per 64-element group, 8 elements
// per lane), reusing X in registers across all 4 rows.
// ---------------------------------------------------------------------------
kernel void ami_dequant_gemv(const device half *x [[buffer(0)]],
                             const device uchar *qweight [[buffer(1)]],
                             const device half2 *scale_bias [[buffer(2)]],
                             device half *y [[buffer(3)]],
                             constant DequantParams &p [[buffer(4)]],
                             uint tg_id [[threadgroup_position_in_grid]],
                             uint sg_id [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint ROWS_PER_SG = 4u;
    const uint row0 = tg_id * 16u + sg_id * ROWS_PER_SG;
    if (row0 >= p.n) {
        return;
    }

    const uint sub_group = lane >> 3;   // 0..3 (which of 4 64-elem groups)
    const uint lane_in_grp = lane & 7u; // 0..7 (8 contiguous elements/lane)

    float acc[ROWS_PER_SG] = {0.0f, 0.0f, 0.0f, 0.0f};
    const bool full_rows = (row0 + ROWS_PER_SG <= p.n);

    for (uint g_base = 0u; g_base < p.num_groups; g_base += 4u) {
        const uint g = g_base + sub_group;
        if (g >= p.num_groups) {
            continue;
        }

        const uint k_off = g * Q_GROUP_SIZE + lane_in_grp * 8u;
        const float4 fx0 = float4(*(const device half4 *)(x + k_off));
        const float4 fx1 = float4(*(const device half4 *)(x + k_off + 4u));
        const float x_sum = (fx0.x + fx0.y) + (fx0.z + fx0.w) +
                            (fx1.x + fx1.y) + (fx1.z + fx1.w);

        if (Q_BITS == 4u) {
            const float4 px0 = float4(fx0.x,
                                      fx0.y * (1.0f / 16.0f),
                                      fx0.z * (1.0f / 256.0f),
                                      fx0.w * (1.0f / 4096.0f));
            const float4 px1 = float4(fx1.x,
                                      fx1.y * (1.0f / 16.0f),
                                      fx1.z * (1.0f / 256.0f),
                                      fx1.w * (1.0f / 4096.0f));
            const uint w_off = g * 32u + lane_in_grp * 4u;

            if (full_rows) {
                for (uint r = 0u; r < ROWS_PER_SG; r++) {
                    const uint row = row0 + r;
                    const uint32_t w = *(const device uint32_t *)(qweight + ulong(row) * p.row_bytes + w_off);
                    const half2 sb = scale_bias[ulong(row) * p.num_groups + g];
                    const float qd = qdot8_q4(w, px0, px1);
                    acc[r] = fma(qd, float(sb.x), fma(x_sum, float(sb.y), acc[r]));
                }
            } else {
                for (uint r = 0u; r < ROWS_PER_SG; r++) {
                    const uint row = row0 + r;
                    if (row < p.n) {
                        const uint32_t w = *(const device uint32_t *)(qweight + ulong(row) * p.row_bytes + w_off);
                        const half2 sb = scale_bias[ulong(row) * p.num_groups + g];
                        const float qd = qdot8_q4(w, px0, px1);
                        acc[r] = fma(qd, float(sb.x), fma(x_sum, float(sb.y), acc[r]));
                    }
                }
            }
        } else if (Q_BITS == 3u) {
            const float4 px0 = float4(fx0.x,
                                      fx0.y * (1.0f / 8.0f),
                                      fx0.z * (1.0f / 64.0f),
                                      fx0.w * (1.0f / 2.0f));
            const float4 px1 = float4(fx1.x * (1.0f / 16.0f),
                                      fx1.y * (1.0f / 128.0f),
                                      fx1.z * (1.0f / 4.0f),
                                      fx1.w * (1.0f / 32.0f));
            const float x2_hi = fx0.z * 4.0f;
            const float x5_hi = fx1.y * 2.0f;
            const uint w_off = g * 24u + lane_in_grp * 3u;

            if (full_rows) {
                for (uint r = 0u; r < ROWS_PER_SG; r++) {
                    const uint row = row0 + r;
                    const device uchar *wp = qweight + ulong(row) * p.row_bytes + w_off;
                    const half2 sb = scale_bias[ulong(row) * p.num_groups + g];
                    const float qd = qdot8_q3(wp[0], wp[1], wp[2], px0, px1, x2_hi, x5_hi);
                    acc[r] = fma(qd, float(sb.x), fma(x_sum, float(sb.y), acc[r]));
                }
            } else {
                for (uint r = 0u; r < ROWS_PER_SG; r++) {
                    const uint row = row0 + r;
                    if (row < p.n) {
                        const device uchar *wp = qweight + ulong(row) * p.row_bytes + w_off;
                        const half2 sb = scale_bias[ulong(row) * p.num_groups + g];
                        const float qd = qdot8_q3(wp[0], wp[1], wp[2], px0, px1, x2_hi, x5_hi);
                        acc[r] = fma(qd, float(sb.x), fma(x_sum, float(sb.y), acc[r]));
                    }
                }
            }
        }
    }

    for (uint r = 0u; r < ROWS_PER_SG; r++) {
        const float total = simd_sum(acc[r]);
        if (lane == 0u && (row0 + r) < p.n) {
            y[row0 + r] = half(total);
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: Fused Dequant GEMM (M > 1 prefill & speculative tree verification)
//
// Tile: BM = 32, BN = 32, BK = 64 (= Q_GROUP_SIZE).
// 4 simdgroups/threadgroup (128 threads), arranged 2x2: each simdgroup
// accumulates a 16x16 tile with four simdgroup_float8x8 matrix tiles.
// ---------------------------------------------------------------------------
kernel void ami_dequant_gemm(const device half *x [[buffer(0)]],
                             const device uchar *qweight [[buffer(1)]],
                             const device half2 *scale_bias [[buffer(2)]],
                             device half *y [[buffer(3)]],
                             constant DequantParams &p [[buffer(4)]],
                             uint2 tg_pos [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]],
                             uint sg_id [[simdgroup_index_in_threadgroup]]) {
    constexpr uint BM = 32u;
    constexpr uint BN = 32u;
    constexpr uint BK = 64u;
    constexpr uint BK_PAD = 72u; // +8 halfs avoids threadgroup bank conflicts

    const uint n_base = tg_pos.x * BN;
    const uint m_base = tg_pos.y * BM;
    if (m_base >= p.m || n_base >= p.n) {
        return;
    }

    threadgroup half As[BM * BK_PAD];
    threadgroup half Bs[BN * BK_PAD];
    threadgroup float Cs[BM * BN];

    simdgroup_float8x8 acc[2][2];
    for (uint i = 0u; i < 2u; i++) {
        for (uint j = 0u; j < 2u; j++) {
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    const uint sg_m = (sg_id >> 1) * 16u;
    const uint sg_n = (sg_id & 1u) * 16u;
    const uint load_row = tid >> 2;         // 0..31
    const uint load_chunk = tid & 3u;       // 0..3 (16 elements per thread)
    const uint load_col = load_chunk * 16u; // 0, 16, 32, 48

    for (uint g = 0u; g < p.num_groups; g++) {
        const uint k_base = g * BK;

        // 1. Cooperative load of X tile [BM, BK] -> As
        {
            const uint gm = m_base + load_row;
            threadgroup half *a_dst = &As[load_row * BK_PAD + load_col];
            if (gm < p.m) {
                const device half4 *x_src = (const device half4 *)(x + ulong(gm) * p.k + k_base + load_col);
                *(threadgroup half4 *)(a_dst + 0u) = x_src[0];
                *(threadgroup half4 *)(a_dst + 4u) = x_src[1];
                *(threadgroup half4 *)(a_dst + 8u) = x_src[2];
                *(threadgroup half4 *)(a_dst + 12u) = x_src[3];
            } else {
                const half4 z = half4(0.0h);
                *(threadgroup half4 *)(a_dst + 0u) = z;
                *(threadgroup half4 *)(a_dst + 4u) = z;
                *(threadgroup half4 *)(a_dst + 8u) = z;
                *(threadgroup half4 *)(a_dst + 12u) = z;
            }
        }

        // 2. Cooperative fused dequantization of W tile [BN, BK] -> Bs
        {
            const uint gn = n_base + load_row;
            threadgroup half *b_dst = &Bs[load_row * BK_PAD + load_col];
            if (gn < p.n) {
                const half2 sb = scale_bias[ulong(gn) * p.num_groups + g];
                const float scale = float(sb.x);
                const float bias = float(sb.y);

                if (Q_BITS == 4u) {
                    const uint2 w = *(const device uint2 *)(qweight + ulong(gn) * p.row_bytes + g * 32u + load_chunk * 8u);
                    uint v0 = w.x;
                    uint v1 = w.y;
                    for (uint j = 0u; j < 8u; j++) {
                        b_dst[j] = half(fma(float((v0 >> (j * 4u)) & 0xFu), scale, bias));
                        b_dst[8u + j] = half(fma(float((v1 >> (j * 4u)) & 0xFu), scale, bias));
                    }
                } else if (Q_BITS == 3u) {
                    const device uchar *wp = qweight + ulong(gn) * p.row_bytes + g * 24u + load_chunk * 6u;
                    for (uint blk = 0u; blk < 2u; blk++) {
                        const uint word = uint(wp[blk * 3u + 0u]) |
                                          (uint(wp[blk * 3u + 1u]) << 8) |
                                          (uint(wp[blk * 3u + 2u]) << 16);
                        for (uint j = 0u; j < 8u; j++) {
                            b_dst[blk * 8u + j] = half(fma(float((word >> (j * 3u)) & 0x7u), scale, bias));
                        }
                    }
                }
            } else {
                const half4 z = half4(0.0h);
                *(threadgroup half4 *)(b_dst + 0u) = z;
                *(threadgroup half4 *)(b_dst + 4u) = z;
                *(threadgroup half4 *)(b_dst + 8u) = z;
                *(threadgroup half4 *)(b_dst + 12u) = z;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 3. Hardware 8x8 simdgroup matrix multiply-accumulate across BK=64
        for (uint kk = 0u; kk < BK; kk += 8u) {
            simdgroup_half8x8 a_frag[2];
            simdgroup_half8x8 b_frag[2];
            simdgroup_load(a_frag[0], &As[(sg_m + 0u) * BK_PAD + kk], BK_PAD);
            simdgroup_load(a_frag[1], &As[(sg_m + 8u) * BK_PAD + kk], BK_PAD);
            simdgroup_load(b_frag[0], &Bs[(sg_n + 0u) * BK_PAD + kk], BK_PAD, ulong2(0, 0), true);
            simdgroup_load(b_frag[1], &Bs[(sg_n + 8u) * BK_PAD + kk], BK_PAD, ulong2(0, 0), true);

            simdgroup_multiply_accumulate(acc[0][0], a_frag[0], b_frag[0], acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1], a_frag[0], b_frag[1], acc[0][1]);
            simdgroup_multiply_accumulate(acc[1][0], a_frag[1], b_frag[0], acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1], a_frag[1], b_frag[1], acc[1][1]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 4. Stage accumulators through Cs and write guarded [M, N] fp16 outputs
    for (uint i = 0u; i < 2u; i++) {
        for (uint j = 0u; j < 2u; j++) {
            simdgroup_store(acc[i][j], &Cs[(sg_m + i * 8u) * BN + (sg_n + j * 8u)], BN);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tid; idx < BM * BN; idx += 128u) {
        const uint r = idx / BN;
        const uint c = idx % BN;
        const uint gm = m_base + r;
        const uint gn = n_base + c;
        if (gm < p.m && gn < p.n) {
            y[ulong(gm) * p.n + gn] = half(Cs[idx]);
        }
    }
}
