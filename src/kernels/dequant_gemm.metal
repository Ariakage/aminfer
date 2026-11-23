// Fused affine 2-bit, 3-bit and 4-bit (MLX group_size=64) dequantization GEMV
// and GEMM kernels (G1, G2, G13). Specialized at startup via Metal function
// constants so hot loops carry zero bit-width branching (ADR-0014, ADR-0017).
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant uint Q_BITS [[function_constant(0)]];
constant uint Q_GROUP_SIZE [[function_constant(1)]];
constant bool DENSE_W_BF16 [[function_constant(2)]];
constant bool HAS_RESIDUAL_EPILOGUE_RAW [[function_constant(3)]];
constant bool HAS_RESIDUAL_EPILOGUE = is_function_constant_defined(HAS_RESIDUAL_EPILOGUE_RAW)
    ? HAS_RESIDUAL_EPILOGUE_RAW
    : false;

struct DequantParams {
    uint m;
    uint n;
    uint k;
    uint num_groups;
    uint row_bytes;
};

struct FusedQkvParams {
    uint m;
    uint n_q;
    uint n_k;
    uint n_v;
    uint k;
    uint num_groups;
    uint row_bytes;
    uint tgs_q;
    uint tgs_k;
    uint tgs_v;
    uint _pad0;
    uint _pad1;
};

static_assert(sizeof(FusedQkvParams) == 48, "FusedQkvParams must be 48 bytes");

static inline float dequant_fast_silu(float x) {
    return x / (1.0f + exp(-x));
}

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

static inline float qdot8_q2(uint16_t w, float4 px0, float4 px1) {
    const uint lo = uint(w & 0xFFu);
    const uint hi = uint(w >> 8);
    return fma(float(lo & 0x03u), px0.x,
           fma(float(lo & 0x0Cu), px0.y,
           fma(float(lo & 0x30u), px0.z,
           fma(float(lo & 0xC0u), px0.w,
           fma(float(hi & 0x03u), px1.x,
           fma(float(hi & 0x0Cu), px1.y,
           fma(float(hi & 0x30u), px1.z,
               float(hi & 0xC0u) * px1.w)))))));
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
                             const device half *residual [[buffer(4)]],
                             constant DequantParams &p [[buffer(5)]],
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
        } else if (Q_BITS == 2u) {
            const float4 px0 = float4(fx0.x,
                                      fx0.y * (1.0f / 4.0f),
                                      fx0.z * (1.0f / 16.0f),
                                      fx0.w * (1.0f / 64.0f));
            const float4 px1 = float4(fx1.x,
                                      fx1.y * (1.0f / 4.0f),
                                      fx1.z * (1.0f / 16.0f),
                                      fx1.w * (1.0f / 64.0f));
            const uint w_off = g * 16u + lane_in_grp * 2u;

            if (full_rows) {
                for (uint r = 0u; r < ROWS_PER_SG; r++) {
                    const uint row = row0 + r;
                    const uint16_t w = *(const device uint16_t *)(qweight + ulong(row) * p.row_bytes + w_off);
                    const half2 sb = scale_bias[ulong(row) * p.num_groups + g];
                    const float qd = qdot8_q2(w, px0, px1);
                    acc[r] = fma(qd, float(sb.x), fma(x_sum, float(sb.y), acc[r]));
                }
            } else {
                for (uint r = 0u; r < ROWS_PER_SG; r++) {
                    const uint row = row0 + r;
                    if (row < p.n) {
                        const uint16_t w = *(const device uint16_t *)(qweight + ulong(row) * p.row_bytes + w_off);
                        const half2 sb = scale_bias[ulong(row) * p.num_groups + g];
                        const float qd = qdot8_q2(w, px0, px1);
                        acc[r] = fma(qd, float(sb.x), fma(x_sum, float(sb.y), acc[r]));
                    }
                }
            }
        }
    }

    #pragma clang loop unroll(full)
    for (uint r = 0u; r < ROWS_PER_SG; r++) {
        const float total = simd_sum(acc[r]);
        if (lane == 0u && (row0 + r) < p.n) {
            const uint row = row0 + r;
            const float out_val = HAS_RESIDUAL_EPILOGUE ? (total + float(residual[row])) : total;
            y[row] = half(out_val);
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
                             const device half *residual [[buffer(4)]],
                             constant DequantParams &p [[buffer(5)]],
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
                } else if (Q_BITS == 2u) {
                    const uint32_t w = *(const device uint32_t *)(qweight + ulong(gn) * p.row_bytes + g * 16u + load_chunk * 4u);
                    for (uint j = 0u; j < 16u; j++) {
                        b_dst[j] = half(fma(float((w >> (j * 2u)) & 0x3u), scale, bias));
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
            const ulong out_idx = ulong(gm) * p.n + gn;
            const float val = HAS_RESIDUAL_EPILOGUE ? (Cs[idx] + float(residual[out_idx])) : Cs[idx];
            y[out_idx] = half(val);
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 2b: Weight-Stationary Batched Dequant GEMV (M = 1..64 tree verify)
//
// Geometry: 4 simdgroups/threadgroup (128 threads), 2 output rows/simdgroup
// (8 output rows/threadgroup), up to 8 tokens per tg_pos.y block.
// Issues 2 independent DRAM weight/scale_bias loads per step (full M4 MLP),
// computes `px0, px1, x_sum` ONCE per token and reuses across both rows, and
// uses a single `x_base` pointer for low register pressure and high occupancy.
// Preserves exact bit-for-bit FP32 FMA order with `ami_dequant_gemv`.
// ---------------------------------------------------------------------------
static inline void qdot8_pair_q4(const device half *x_row,
                                 uint k_off,
                                 uint32_t w0, float s0, float b0,
                                 uint32_t w1, float s1, float b1,
                                 thread float &a0,
                                 thread float &a1) {
    const float4 fx0 = float4(*(const device half4 *)(x_row + k_off));
    const float4 fx1 = float4(*(const device half4 *)(x_row + k_off + 4u));
    const float x_sum = (fx0.x + fx0.y) + (fx0.z + fx0.w) +
                        (fx1.x + fx1.y) + (fx1.z + fx1.w);
    const float4 px0 = float4(fx0.x,
                              fx0.y * (1.0f / 16.0f),
                              fx0.z * (1.0f / 256.0f),
                              fx0.w * (1.0f / 4096.0f));
    const float4 px1 = float4(fx1.x,
                              fx1.y * (1.0f / 16.0f),
                              fx1.z * (1.0f / 256.0f),
                              fx1.w * (1.0f / 4096.0f));
    a0 = fma(qdot8_q4(w0, px0, px1), s0, fma(x_sum, b0, a0));
    a1 = fma(qdot8_q4(w1, px0, px1), s1, fma(x_sum, b1, a1));
}

static inline void qdot8_pair_q3(const device half *x_row,
                                 uint k_off,
                                 uchar r0_b0, uchar r0_b1, uchar r0_b2, float s0, float b0,
                                 uchar r1_b0, uchar r1_b1, uchar r1_b2, float s1, float b1,
                                 thread float &a0,
                                 thread float &a1) {
    const float4 fx0 = float4(*(const device half4 *)(x_row + k_off));
    const float4 fx1 = float4(*(const device half4 *)(x_row + k_off + 4u));
    const float x_sum = (fx0.x + fx0.y) + (fx0.z + fx0.w) +
                        (fx1.x + fx1.y) + (fx1.z + fx1.w);
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
    a0 = fma(qdot8_q3(r0_b0, r0_b1, r0_b2, px0, px1, x2_hi, x5_hi), s0, fma(x_sum, b0, a0));
    a1 = fma(qdot8_q3(r1_b0, r1_b1, r1_b2, px0, px1, x2_hi, x5_hi), s1, fma(x_sum, b1, a1));
}

static inline void qdot8_pair_q2(const device half *x_row,
                                 uint k_off,
                                 uint16_t w0, float s0, float b0,
                                 uint16_t w1, float s1, float b1,
                                 thread float &a0,
                                 thread float &a1) {
    const float4 fx0 = float4(*(const device half4 *)(x_row + k_off));
    const float4 fx1 = float4(*(const device half4 *)(x_row + k_off + 4u));
    const float x_sum = (fx0.x + fx0.y) + (fx0.z + fx0.w) +
                        (fx1.x + fx1.y) + (fx1.z + fx1.w);
    const float4 px0 = float4(fx0.x,
                              fx0.y * (1.0f / 4.0f),
                              fx0.z * (1.0f / 16.0f),
                              fx0.w * (1.0f / 64.0f));
    const float4 px1 = float4(fx1.x,
                              fx1.y * (1.0f / 4.0f),
                              fx1.z * (1.0f / 16.0f),
                              fx1.w * (1.0f / 64.0f));
    a0 = fma(qdot8_q2(w0, px0, px1), s0, fma(x_sum, b0, a0));
    a1 = fma(qdot8_q2(w1, px0, px1), s1, fma(x_sum, b1, a1));
}

static inline void gemv_batch_store_pair(device half *y,
                                         const device half *residual,
                                         uint m_idx,
                                         uint n,
                                         uint row0,
                                         uint row1,
                                         bool has_row1,
                                         float t0,
                                         float t1) {
    const ulong base = ulong(m_idx) * n;
    const ulong idx0 = base + row0;
    y[idx0] = half(HAS_RESIDUAL_EPILOGUE ? (t0 + float(residual[idx0])) : t0);
    if (has_row1) {
        const ulong idx1 = base + row1;
        y[idx1] = half(HAS_RESIDUAL_EPILOGUE ? (t1 + float(residual[idx1])) : t1);
    }
}

kernel void ami_dequant_gemv_batch(const device half *x [[buffer(0)]],
                                   const device uchar *qweight [[buffer(1)]],
                                   const device half2 *scale_bias [[buffer(2)]],
                                   device half *y [[buffer(3)]],
                                   const device half *residual [[buffer(4)]],
                                   constant DequantParams &p [[buffer(5)]],
                                   uint2 tg_pos [[threadgroup_position_in_grid]],
                                   uint sg_id [[simdgroup_index_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]]) {
    const uint row0 = (tg_pos.x * 4u + sg_id) * 2u;
    const uint m_base = tg_pos.y * 8u;
    if (row0 >= p.n || m_base >= p.m) {
        return;
    }
    const uint row1 = min(row0 + 1u, p.n - 1u);
    const bool has_row1 = (row0 + 1u) < p.n;
    const uint tile_m = min(8u, p.m - m_base);
    const uint sub_group = lane >> 3;
    const uint lane_in_grp = lane & 7u;

    const device uchar *w_row0 = qweight + ulong(row0) * p.row_bytes;
    const device uchar *w_row1 = qweight + ulong(row1) * p.row_bytes;
    const device half2 *sb_row0 = scale_bias + ulong(row0) * p.num_groups;
    const device half2 *sb_row1 = scale_bias + ulong(row1) * p.num_groups;

    const device half *x_base = x + ulong(m_base) * p.k;
    const ulong x_stride = ulong(p.k);

    float a0_0 = 0.0f, a0_1 = 0.0f, a0_2 = 0.0f, a0_3 = 0.0f;
    float a0_4 = 0.0f, a0_5 = 0.0f, a0_6 = 0.0f, a0_7 = 0.0f;
    float a1_0 = 0.0f, a1_1 = 0.0f, a1_2 = 0.0f, a1_3 = 0.0f;
    float a1_4 = 0.0f, a1_5 = 0.0f, a1_6 = 0.0f, a1_7 = 0.0f;

    for (uint g_base = 0u; g_base < p.num_groups; g_base += 4u) {
        const uint g = g_base + sub_group;
        if (g >= p.num_groups) {
            continue;
        }
        const uint k_off = g * Q_GROUP_SIZE + lane_in_grp * 8u;
        const half2 sb0 = sb_row0[g];
        const half2 sb1 = sb_row1[g];
        const float s0 = float(sb0.x), b0 = float(sb0.y);
        const float s1 = float(sb1.x), b1 = float(sb1.y);

        if (Q_BITS == 4u) {
            const uint w_off = g * 32u + lane_in_grp * 4u;
            const uint32_t w0 = *(const device uint32_t *)(w_row0 + w_off);
            const uint32_t w1 = *(const device uint32_t *)(w_row1 + w_off);
            qdot8_pair_q4(x_base + 0ul * x_stride, k_off, w0, s0, b0, w1, s1, b1, a0_0, a1_0);
            if (tile_m > 1u) qdot8_pair_q4(x_base + 1ul * x_stride, k_off, w0, s0, b0, w1, s1, b1, a0_1, a1_1);
            if (tile_m > 2u) qdot8_pair_q4(x_base + 2ul * x_stride, k_off, w0, s0, b0, w1, s1, b1, a0_2, a1_2);
            if (tile_m > 3u) qdot8_pair_q4(x_base + 3ul * x_stride, k_off, w0, s0, b0, w1, s1, b1, a0_3, a1_3);
            if (tile_m > 4u) {
                qdot8_pair_q4(x_base + 4ul * x_stride, k_off, w0, s0, b0, w1, s1, b1, a0_4, a1_4);
                if (tile_m > 5u) qdot8_pair_q4(x_base + 5ul * x_stride, k_off, w0, s0, b0, w1, s1, b1, a0_5, a1_5);
                if (tile_m > 6u) qdot8_pair_q4(x_base + 6ul * x_stride, k_off, w0, s0, b0, w1, s1, b1, a0_6, a1_6);
                if (tile_m > 7u) qdot8_pair_q4(x_base + 7ul * x_stride, k_off, w0, s0, b0, w1, s1, b1, a0_7, a1_7);
            }
        } else if (Q_BITS == 3u) {
            const uint w_off = g * 24u + lane_in_grp * 3u;
            const device uchar *wp0 = w_row0 + w_off;
            const device uchar *wp1 = w_row1 + w_off;
            const uchar r0_0 = wp0[0], r0_1 = wp0[1], r0_2 = wp0[2];
            const uchar r1_0 = wp1[0], r1_1 = wp1[1], r1_2 = wp1[2];
            qdot8_pair_q3(x_base + 0ul * x_stride, k_off, r0_0, r0_1, r0_2, s0, b0, r1_0, r1_1, r1_2, s1, b1, a0_0, a1_0);
            if (tile_m > 1u) qdot8_pair_q3(x_base + 1ul * x_stride, k_off, r0_0, r0_1, r0_2, s0, b0, r1_0, r1_1, r1_2, s1, b1, a0_1, a1_1);
            if (tile_m > 2u) qdot8_pair_q3(x_base + 2ul * x_stride, k_off, r0_0, r0_1, r0_2, s0, b0, r1_0, r1_1, r1_2, s1, b1, a0_2, a1_2);
            if (tile_m > 3u) qdot8_pair_q3(x_base + 3ul * x_stride, k_off, r0_0, r0_1, r0_2, s0, b0, r1_0, r1_1, r1_2, s1, b1, a0_3, a1_3);
            if (tile_m > 4u) {
                qdot8_pair_q3(x_base + 4ul * x_stride, k_off, r0_0, r0_1, r0_2, s0, b0, r1_0, r1_1, r1_2, s1, b1, a0_4, a1_4);
                if (tile_m > 5u) qdot8_pair_q3(x_base + 5ul * x_stride, k_off, r0_0, r0_1, r0_2, s0, b0, r1_0, r1_1, r1_2, s1, b1, a0_5, a1_5);
                if (tile_m > 6u) qdot8_pair_q3(x_base + 6ul * x_stride, k_off, r0_0, r0_1, r0_2, s0, b0, r1_0, r1_1, r1_2, s1, b1, a0_6, a1_6);
                if (tile_m > 7u) qdot8_pair_q3(x_base + 7ul * x_stride, k_off, r0_0, r0_1, r0_2, s0, b0, r1_0, r1_1, r1_2, s1, b1, a0_7, a1_7);
            }
        } else if (Q_BITS == 2u) {
            const uint w_off = g * 16u + lane_in_grp * 2u;
            const uint16_t w0 = *(const device uint16_t *)(w_row0 + w_off);
            const uint16_t w1 = *(const device uint16_t *)(w_row1 + w_off);
            qdot8_pair_q2(x_base + 0ul * x_stride, k_off, w0, s0, b0, w1, s1, b1, a0_0, a1_0);
            if (tile_m > 1u) qdot8_pair_q2(x_base + 1ul * x_stride, k_off, w0, s0, b0, w1, s1, b1, a0_1, a1_1);
            if (tile_m > 2u) qdot8_pair_q2(x_base + 2ul * x_stride, k_off, w0, s0, b0, w1, s1, b1, a0_2, a1_2);
            if (tile_m > 3u) qdot8_pair_q2(x_base + 3ul * x_stride, k_off, w0, s0, b0, w1, s1, b1, a0_3, a1_3);
            if (tile_m > 4u) {
                qdot8_pair_q2(x_base + 4ul * x_stride, k_off, w0, s0, b0, w1, s1, b1, a0_4, a1_4);
                if (tile_m > 5u) qdot8_pair_q2(x_base + 5ul * x_stride, k_off, w0, s0, b0, w1, s1, b1, a0_5, a1_5);
                if (tile_m > 6u) qdot8_pair_q2(x_base + 6ul * x_stride, k_off, w0, s0, b0, w1, s1, b1, a0_6, a1_6);
                if (tile_m > 7u) qdot8_pair_q2(x_base + 7ul * x_stride, k_off, w0, s0, b0, w1, s1, b1, a0_7, a1_7);
            }
        }
    }

    const float t0_0 = simd_sum(a0_0), t1_0 = simd_sum(a1_0);
    if (lane == 0u) {
        gemv_batch_store_pair(y, residual, m_base + 0u, p.n, row0, row1, has_row1, t0_0, t1_0);
    }
    if (tile_m > 1u) {
        const float t0_1 = simd_sum(a0_1), t1_1 = simd_sum(a1_1);
        if (lane == 0u) {
            gemv_batch_store_pair(y, residual, m_base + 1u, p.n, row0, row1, has_row1, t0_1, t1_1);
        }
    }
    if (tile_m > 2u) {
        const float t0_2 = simd_sum(a0_2), t1_2 = simd_sum(a1_2);
        if (lane == 0u) {
            gemv_batch_store_pair(y, residual, m_base + 2u, p.n, row0, row1, has_row1, t0_2, t1_2);
        }
    }
    if (tile_m > 3u) {
        const float t0_3 = simd_sum(a0_3), t1_3 = simd_sum(a1_3);
        if (lane == 0u) {
            gemv_batch_store_pair(y, residual, m_base + 3u, p.n, row0, row1, has_row1, t0_3, t1_3);
        }
    }
    if (tile_m > 4u) {
        const float t0_4 = simd_sum(a0_4), t1_4 = simd_sum(a1_4);
        if (lane == 0u) {
            gemv_batch_store_pair(y, residual, m_base + 4u, p.n, row0, row1, has_row1, t0_4, t1_4);
        }
        if (tile_m > 5u) {
            const float t0_5 = simd_sum(a0_5), t1_5 = simd_sum(a1_5);
            if (lane == 0u) {
                gemv_batch_store_pair(y, residual, m_base + 5u, p.n, row0, row1, has_row1, t0_5, t1_5);
            }
        }
        if (tile_m > 6u) {
            const float t0_6 = simd_sum(a0_6), t1_6 = simd_sum(a1_6);
            if (lane == 0u) {
                gemv_batch_store_pair(y, residual, m_base + 6u, p.n, row0, row1, has_row1, t0_6, t1_6);
            }
        }
        if (tile_m > 7u) {
            const float t0_7 = simd_sum(a0_7), t1_7 = simd_sum(a1_7);
            if (lane == 0u) {
                gemv_batch_store_pair(y, residual, m_base + 7u, p.n, row0, row1, has_row1, t0_7, t1_7);
            }
        }
    }
}

// ===========================================================================
// Native unquantized FP16 / BF16 dense GEMV and GEMM (G12, ADR-0016).
//
// Specialized at startup on DENSE_W_BF16 (false: `half` weights, true:
// `bfloat` weights). Activations are always `half` and outputs `half`; every
// dot product and matrix multiply-accumulate accumulates in FP32.
// ===========================================================================

struct DenseParams {
    uint m;
    uint n;
    uint k;
    uint _pad;
};

static inline float4 dense_load_w4(const device uchar *w, ulong elem) {
    if (DENSE_W_BF16) {
        return float4(*(const device bfloat4 *)(w + elem * 2ul));
    }
    return float4(*(const device half4 *)(w + elem * 2ul));
}

static inline float dense_load_w1(const device uchar *w, ulong elem) {
    if (DENSE_W_BF16) {
        return float(*(const device bfloat *)(w + elem * 2ul));
    }
    return float(*(const device half *)(w + elem * 2ul));
}

// ---------------------------------------------------------------------------
// Kernel 3: Dense GEMV (M = 1 decode hot path, projections and lm_head)
//
// Geometry: 4 simdgroups/threadgroup (128 threads), 4 output rows/simdgroup
// (16 rows/threadgroup). With K % 4 == 0 every lane issues coalesced 8-byte
// `half4` / `bfloat4` loads (two independent loads per step, 256 elements per
// simdgroup step) and reuses each X vector across 4 rows; otherwise a scalar
// strided path handles arbitrary K.
// ---------------------------------------------------------------------------
kernel void ami_dense_gemv(const device half *x [[buffer(0)]],
                           const device uchar *w [[buffer(1)]],
                           device half *y [[buffer(2)]],
                           const device half *residual [[buffer(3)]],
                           constant DenseParams &p [[buffer(4)]],
                           uint tg_id [[threadgroup_position_in_grid]],
                           uint sg_id [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint ROWS_PER_SG = 4u;
    const uint row0 = tg_id * 16u + sg_id * ROWS_PER_SG;
    if (row0 >= p.n) {
        return;
    }

    // Clamp out-of-range rows onto the last valid row so loads stay in bounds
    // without divergent branches; their results are never stored.
    ulong row_base[ROWS_PER_SG];
    for (uint r = 0u; r < ROWS_PER_SG; r++) {
        row_base[r] = ulong(min(row0 + r, p.n - 1u)) * p.k;
    }
    float acc[ROWS_PER_SG] = {0.0f, 0.0f, 0.0f, 0.0f};

    if ((p.k & 3u) == 0u) {
        const uint vec_k = p.k >> 2;
        const device half4 *x4 = (const device half4 *)x;
        uint i = lane;
        for (; i + 32u < vec_k; i += 64u) {
            const float4 xa = float4(x4[i]);
            const float4 xb = float4(x4[i + 32u]);
            for (uint r = 0u; r < ROWS_PER_SG; r++) {
                const float4 wa = dense_load_w4(w, row_base[r] + ulong(i) * 4ul);
                const float4 wb = dense_load_w4(w, row_base[r] + ulong(i + 32u) * 4ul);
                acc[r] = fma(xa.x, wa.x, acc[r]);
                acc[r] = fma(xa.y, wa.y, acc[r]);
                acc[r] = fma(xa.z, wa.z, acc[r]);
                acc[r] = fma(xa.w, wa.w, acc[r]);
                acc[r] = fma(xb.x, wb.x, acc[r]);
                acc[r] = fma(xb.y, wb.y, acc[r]);
                acc[r] = fma(xb.z, wb.z, acc[r]);
                acc[r] = fma(xb.w, wb.w, acc[r]);
            }
        }
        for (; i < vec_k; i += 32u) {
            const float4 xa = float4(x4[i]);
            for (uint r = 0u; r < ROWS_PER_SG; r++) {
                const float4 wa = dense_load_w4(w, row_base[r] + ulong(i) * 4ul);
                acc[r] = fma(xa.x, wa.x, acc[r]);
                acc[r] = fma(xa.y, wa.y, acc[r]);
                acc[r] = fma(xa.z, wa.z, acc[r]);
                acc[r] = fma(xa.w, wa.w, acc[r]);
            }
        }
    } else {
        for (uint kk = lane; kk < p.k; kk += 32u) {
            const float xv = float(x[kk]);
            for (uint r = 0u; r < ROWS_PER_SG; r++) {
                acc[r] = fma(xv, dense_load_w1(w, row_base[r] + kk), acc[r]);
            }
        }
    }

    #pragma clang loop unroll(full)
    for (uint r = 0u; r < ROWS_PER_SG; r++) {
        const float total = simd_sum(acc[r]);
        if (lane == 0u && (row0 + r) < p.n) {
            const uint row = row0 + r;
            const float out_val = HAS_RESIDUAL_EPILOGUE ? (total + float(residual[row])) : total;
            y[row] = half(out_val);
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 4: Dense GEMM (M > 1 prefill & speculative tree verification)
//
// Tile: BM = 32, BN = 32, BK = 32. 4 simdgroups/threadgroup (128 threads),
// arranged 2x2: each simdgroup accumulates a 16x16 FP32 tile with four
// `simdgroup_float8x8` accumulators. FP16 weights use `simdgroup_half8x8`
// fragments; BF16 weights stage both the X tile (converted from `half`) and
// the W tile as `bfloat` and use `simdgroup_bfloat8x8` fragments. Arbitrary
// M >= 1, N and K are supported with zero-filled boundary tiles.
// ---------------------------------------------------------------------------
constant constexpr uint DG_BM = 32u;
constant constexpr uint DG_BN = 32u;
constant constexpr uint DG_BK = 32u;
constant constexpr uint DG_BK_PAD = 40u; // +8 elements avoids bank conflicts

template <typename T>
static inline void dense_gemm_stage(const device half *x,
                                    const device uchar *w,
                                    threadgroup T *As,
                                    threadgroup T *Bs,
                                    constant DenseParams &p,
                                    uint m_base,
                                    uint n_base,
                                    uint k_base,
                                    uint tid) {
    const uint ld_row = tid >> 2;          // 0..31
    const uint ld_col = (tid & 3u) * 8u;   // 0, 8, 16, 24
    const uint gk = k_base + ld_col;
    const bool vec_ok = ((p.k & 3u) == 0u) && (gk + 8u <= p.k);

    // X tile [BM, BK] -> As (converted to T)
    {
        threadgroup T *dst = As + ld_row * DG_BK_PAD + ld_col;
        const uint gm = m_base + ld_row;
        if (gm < p.m && vec_ok) {
            const device half4 *src = (const device half4 *)(x + ulong(gm) * p.k + gk);
            const float4 a0 = float4(src[0]);
            const float4 a1 = float4(src[1]);
            dst[0] = T(a0.x); dst[1] = T(a0.y); dst[2] = T(a0.z); dst[3] = T(a0.w);
            dst[4] = T(a1.x); dst[5] = T(a1.y); dst[6] = T(a1.z); dst[7] = T(a1.w);
        } else {
            for (uint j = 0u; j < 8u; j++) {
                const uint kk = gk + j;
                const float v = (gm < p.m && kk < p.k) ? float(x[ulong(gm) * p.k + kk]) : 0.0f;
                dst[j] = T(v);
            }
        }
    }

    // W tile [BN, BK] -> Bs (stored row-major by output column n)
    {
        threadgroup T *dst = Bs + ld_row * DG_BK_PAD + ld_col;
        const uint gn = n_base + ld_row;
        if (gn < p.n && vec_ok) {
            const ulong e = ulong(gn) * p.k + gk;
            const float4 b0 = dense_load_w4(w, e);
            const float4 b1 = dense_load_w4(w, e + 4ul);
            dst[0] = T(b0.x); dst[1] = T(b0.y); dst[2] = T(b0.z); dst[3] = T(b0.w);
            dst[4] = T(b1.x); dst[5] = T(b1.y); dst[6] = T(b1.z); dst[7] = T(b1.w);
        } else {
            for (uint j = 0u; j < 8u; j++) {
                const uint kk = gk + j;
                const float v = (gn < p.n && kk < p.k) ? dense_load_w1(w, ulong(gn) * p.k + kk) : 0.0f;
                dst[j] = T(v);
            }
        }
    }
}

template <typename T>
static inline void dense_gemm_mma(threadgroup T *As,
                                  threadgroup T *Bs,
                                  thread simdgroup_float8x8 (&acc)[2][2],
                                  uint sg_m,
                                  uint sg_n) {
    for (uint kk = 0u; kk < DG_BK; kk += 8u) {
        simdgroup_matrix<T, 8, 8> a_frag[2];
        simdgroup_matrix<T, 8, 8> b_frag[2];
        simdgroup_load(a_frag[0], As + (sg_m + 0u) * DG_BK_PAD + kk, DG_BK_PAD);
        simdgroup_load(a_frag[1], As + (sg_m + 8u) * DG_BK_PAD + kk, DG_BK_PAD);
        simdgroup_load(b_frag[0], Bs + (sg_n + 0u) * DG_BK_PAD + kk, DG_BK_PAD, ulong2(0, 0), true);
        simdgroup_load(b_frag[1], Bs + (sg_n + 8u) * DG_BK_PAD + kk, DG_BK_PAD, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(acc[0][0], a_frag[0], b_frag[0], acc[0][0]);
        simdgroup_multiply_accumulate(acc[0][1], a_frag[0], b_frag[1], acc[0][1]);
        simdgroup_multiply_accumulate(acc[1][0], a_frag[1], b_frag[0], acc[1][0]);
        simdgroup_multiply_accumulate(acc[1][1], a_frag[1], b_frag[1], acc[1][1]);
    }
}

kernel void ami_dense_gemm(const device half *x [[buffer(0)]],
                           const device uchar *w [[buffer(1)]],
                           device half *y [[buffer(2)]],
                           const device half *residual [[buffer(3)]],
                           constant DenseParams &p [[buffer(4)]],
                           uint2 tg_pos [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint sg_id [[simdgroup_index_in_threadgroup]]) {
    const uint n_base = tg_pos.x * DG_BN;
    const uint m_base = tg_pos.y * DG_BM;
    if (m_base >= p.m || n_base >= p.n) {
        return;
    }

    // Raw 16-bit storage reinterpreted as `half` or `bfloat` per specialization.
    threadgroup half As_raw[DG_BM * DG_BK_PAD];
    threadgroup half Bs_raw[DG_BN * DG_BK_PAD];
    threadgroup float Cs[DG_BM * DG_BN];

    simdgroup_float8x8 acc[2][2];
    for (uint i = 0u; i < 2u; i++) {
        for (uint j = 0u; j < 2u; j++) {
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    const uint sg_m = (sg_id >> 1) * 16u;
    const uint sg_n = (sg_id & 1u) * 16u;

    for (uint k_base = 0u; k_base < p.k; k_base += DG_BK) {
        if (DENSE_W_BF16) {
            threadgroup bfloat *As = (threadgroup bfloat *)As_raw;
            threadgroup bfloat *Bs = (threadgroup bfloat *)Bs_raw;
            dense_gemm_stage<bfloat>(x, w, As, Bs, p, m_base, n_base, k_base, tid);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            dense_gemm_mma<bfloat>(As, Bs, acc, sg_m, sg_n);
        } else {
            dense_gemm_stage<half>(x, w, As_raw, Bs_raw, p, m_base, n_base, k_base, tid);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            dense_gemm_mma<half>(As_raw, Bs_raw, acc, sg_m, sg_n);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0u; i < 2u; i++) {
        for (uint j = 0u; j < 2u; j++) {
            simdgroup_store(acc[i][j], &Cs[(sg_m + i * 8u) * DG_BN + (sg_n + j * 8u)], DG_BN);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tid; idx < DG_BM * DG_BN; idx += 128u) {
        const uint r = idx / DG_BN;
        const uint cc = idx % DG_BN;
        const uint gm = m_base + r;
        const uint gn = n_base + cc;
        if (gm < p.m && gn < p.n) {
            const ulong out_idx = ulong(gm) * p.n + gn;
            const float val = HAS_RESIDUAL_EPILOGUE ? (Cs[idx] + float(residual[out_idx])) : Cs[idx];
            y[out_idx] = half(val);
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 4b: Weight-Stationary Batched Dense FP16/BF16 GEMV (M = 1..64)
//
// Geometry: 4 simdgroups/threadgroup (128 threads), 2 output rows/simdgroup
// (8 output rows/threadgroup), up to 8 tokens per tg_pos.y block.
// Issues 2x2 independent DRAM weight loads per step and reuses each token's
// activation vector across both rows without truncating `half` activations to
// `bfloat`, matching `ami_dense_gemv` bit-for-bit in FP32 accumulation order.
// ---------------------------------------------------------------------------
static inline float dense_fma4(float4 x, float4 w, float acc) {
    acc = fma(x.x, w.x, acc);
    acc = fma(x.y, w.y, acc);
    acc = fma(x.z, w.z, acc);
    acc = fma(x.w, w.w, acc);
    return acc;
}

static inline void dense_pair_step2(const device half4 *x_row,
                                    uint i,
                                    float4 w0a, float4 w0b,
                                    float4 w1a, float4 w1b,
                                    thread float &a0,
                                    thread float &a1) {
    const float4 xa = float4(x_row[i]);
    const float4 xb = float4(x_row[i + 32u]);
    a0 = dense_fma4(xb, w0b, dense_fma4(xa, w0a, a0));
    a1 = dense_fma4(xb, w1b, dense_fma4(xa, w1a, a1));
}

static inline void dense_pair_step1(const device half4 *x_row,
                                    uint i,
                                    float4 w0a,
                                    float4 w1a,
                                    thread float &a0,
                                    thread float &a1) {
    const float4 xa = float4(x_row[i]);
    a0 = dense_fma4(xa, w0a, a0);
    a1 = dense_fma4(xa, w1a, a1);
}

kernel void ami_dense_gemv_batch(const device half *x [[buffer(0)]],
                                 const device uchar *w [[buffer(1)]],
                                 device half *y [[buffer(2)]],
                                 const device half *residual [[buffer(3)]],
                                 constant DenseParams &p [[buffer(4)]],
                                 uint2 tg_pos [[threadgroup_position_in_grid]],
                                 uint sg_id [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
    const uint row0 = (tg_pos.x * 4u + sg_id) * 2u;
    const uint m_base = tg_pos.y * 8u;
    if (row0 >= p.n || m_base >= p.m) {
        return;
    }
    const uint row1 = min(row0 + 1u, p.n - 1u);
    const bool has_row1 = (row0 + 1u) < p.n;
    const uint tile_m = min(8u, p.m - m_base);
    const ulong row_base0 = ulong(row0) * p.k;
    const ulong row_base1 = ulong(row1) * p.k;

    const device half *x_base = x + ulong(m_base) * p.k;
    const ulong x_stride = ulong(p.k);

    float a0_0 = 0.0f, a0_1 = 0.0f, a0_2 = 0.0f, a0_3 = 0.0f;
    float a0_4 = 0.0f, a0_5 = 0.0f, a0_6 = 0.0f, a0_7 = 0.0f;
    float a1_0 = 0.0f, a1_1 = 0.0f, a1_2 = 0.0f, a1_3 = 0.0f;
    float a1_4 = 0.0f, a1_5 = 0.0f, a1_6 = 0.0f, a1_7 = 0.0f;

    if ((p.k & 3u) == 0u) {
        const uint vec_k = p.k >> 2;
        const device half4 *x_base4 = (const device half4 *)x_base;
        const ulong vec_stride = ulong(vec_k);

        uint i = lane;
        for (; i + 32u < vec_k; i += 64u) {
            const float4 w0a = dense_load_w4(w, row_base0 + ulong(i) * 4ul);
            const float4 w0b = dense_load_w4(w, row_base0 + ulong(i + 32u) * 4ul);
            const float4 w1a = dense_load_w4(w, row_base1 + ulong(i) * 4ul);
            const float4 w1b = dense_load_w4(w, row_base1 + ulong(i + 32u) * 4ul);
            dense_pair_step2(x_base4 + 0ul * vec_stride, i, w0a, w0b, w1a, w1b, a0_0, a1_0);
            if (tile_m > 1u) dense_pair_step2(x_base4 + 1ul * vec_stride, i, w0a, w0b, w1a, w1b, a0_1, a1_1);
            if (tile_m > 2u) dense_pair_step2(x_base4 + 2ul * vec_stride, i, w0a, w0b, w1a, w1b, a0_2, a1_2);
            if (tile_m > 3u) dense_pair_step2(x_base4 + 3ul * vec_stride, i, w0a, w0b, w1a, w1b, a0_3, a1_3);
            if (tile_m > 4u) {
                dense_pair_step2(x_base4 + 4ul * vec_stride, i, w0a, w0b, w1a, w1b, a0_4, a1_4);
                if (tile_m > 5u) dense_pair_step2(x_base4 + 5ul * vec_stride, i, w0a, w0b, w1a, w1b, a0_5, a1_5);
                if (tile_m > 6u) dense_pair_step2(x_base4 + 6ul * vec_stride, i, w0a, w0b, w1a, w1b, a0_6, a1_6);
                if (tile_m > 7u) dense_pair_step2(x_base4 + 7ul * vec_stride, i, w0a, w0b, w1a, w1b, a0_7, a1_7);
            }
        }
        for (; i < vec_k; i += 32u) {
            const float4 w0a = dense_load_w4(w, row_base0 + ulong(i) * 4ul);
            const float4 w1a = dense_load_w4(w, row_base1 + ulong(i) * 4ul);
            dense_pair_step1(x_base4 + 0ul * vec_stride, i, w0a, w1a, a0_0, a1_0);
            if (tile_m > 1u) dense_pair_step1(x_base4 + 1ul * vec_stride, i, w0a, w1a, a0_1, a1_1);
            if (tile_m > 2u) dense_pair_step1(x_base4 + 2ul * vec_stride, i, w0a, w1a, a0_2, a1_2);
            if (tile_m > 3u) dense_pair_step1(x_base4 + 3ul * vec_stride, i, w0a, w1a, a0_3, a1_3);
            if (tile_m > 4u) {
                dense_pair_step1(x_base4 + 4ul * vec_stride, i, w0a, w1a, a0_4, a1_4);
                if (tile_m > 5u) dense_pair_step1(x_base4 + 5ul * vec_stride, i, w0a, w1a, a0_5, a1_5);
                if (tile_m > 6u) dense_pair_step1(x_base4 + 6ul * vec_stride, i, w0a, w1a, a0_6, a1_6);
                if (tile_m > 7u) dense_pair_step1(x_base4 + 7ul * vec_stride, i, w0a, w1a, a0_7, a1_7);
            }
        }
    } else {
        for (uint kk = lane; kk < p.k; kk += 32u) {
            const float w0v = dense_load_w1(w, row_base0 + kk);
            const float w1v = dense_load_w1(w, row_base1 + kk);
            { const float xv = float(x_base[0ul * x_stride + kk]); a0_0 = fma(xv, w0v, a0_0); a1_0 = fma(xv, w1v, a1_0); }
            if (tile_m > 1u) { const float xv = float(x_base[1ul * x_stride + kk]); a0_1 = fma(xv, w0v, a0_1); a1_1 = fma(xv, w1v, a1_1); }
            if (tile_m > 2u) { const float xv = float(x_base[2ul * x_stride + kk]); a0_2 = fma(xv, w0v, a0_2); a1_2 = fma(xv, w1v, a1_2); }
            if (tile_m > 3u) { const float xv = float(x_base[3ul * x_stride + kk]); a0_3 = fma(xv, w0v, a0_3); a1_3 = fma(xv, w1v, a1_3); }
            if (tile_m > 4u) {
                { const float xv = float(x_base[4ul * x_stride + kk]); a0_4 = fma(xv, w0v, a0_4); a1_4 = fma(xv, w1v, a1_4); }
                if (tile_m > 5u) { const float xv = float(x_base[5ul * x_stride + kk]); a0_5 = fma(xv, w0v, a0_5); a1_5 = fma(xv, w1v, a1_5); }
                if (tile_m > 6u) { const float xv = float(x_base[6ul * x_stride + kk]); a0_6 = fma(xv, w0v, a0_6); a1_6 = fma(xv, w1v, a1_6); }
                if (tile_m > 7u) { const float xv = float(x_base[7ul * x_stride + kk]); a0_7 = fma(xv, w0v, a0_7); a1_7 = fma(xv, w1v, a1_7); }
            }
        }
    }

    const float t0_0 = simd_sum(a0_0), t1_0 = simd_sum(a1_0);
    if (lane == 0u) {
        gemv_batch_store_pair(y, residual, m_base + 0u, p.n, row0, row1, has_row1, t0_0, t1_0);
    }
    if (tile_m > 1u) {
        const float t0_1 = simd_sum(a0_1), t1_1 = simd_sum(a1_1);
        if (lane == 0u) {
            gemv_batch_store_pair(y, residual, m_base + 1u, p.n, row0, row1, has_row1, t0_1, t1_1);
        }
    }
    if (tile_m > 2u) {
        const float t0_2 = simd_sum(a0_2), t1_2 = simd_sum(a1_2);
        if (lane == 0u) {
            gemv_batch_store_pair(y, residual, m_base + 2u, p.n, row0, row1, has_row1, t0_2, t1_2);
        }
    }
    if (tile_m > 3u) {
        const float t0_3 = simd_sum(a0_3), t1_3 = simd_sum(a1_3);
        if (lane == 0u) {
            gemv_batch_store_pair(y, residual, m_base + 3u, p.n, row0, row1, has_row1, t0_3, t1_3);
        }
    }
    if (tile_m > 4u) {
        const float t0_4 = simd_sum(a0_4), t1_4 = simd_sum(a1_4);
        if (lane == 0u) {
            gemv_batch_store_pair(y, residual, m_base + 4u, p.n, row0, row1, has_row1, t0_4, t1_4);
        }
        if (tile_m > 5u) {
            const float t0_5 = simd_sum(a0_5), t1_5 = simd_sum(a1_5);
            if (lane == 0u) {
                gemv_batch_store_pair(y, residual, m_base + 5u, p.n, row0, row1, has_row1, t0_5, t1_5);
            }
        }
        if (tile_m > 6u) {
            const float t0_6 = simd_sum(a0_6), t1_6 = simd_sum(a1_6);
            if (lane == 0u) {
                gemv_batch_store_pair(y, residual, m_base + 6u, p.n, row0, row1, has_row1, t0_6, t1_6);
            }
        }
        if (tile_m > 7u) {
            const float t0_7 = simd_sum(a0_7), t1_7 = simd_sum(a1_7);
            if (lane == 0u) {
                gemv_batch_store_pair(y, residual, m_base + 7u, p.n, row0, row1, has_row1, t0_7, t1_7);
            }
        }
    }
}

// ===========================================================================
// Milestone M6.8 (G15, ADR-0020): Wide-Tile (BM = 64, BN = 64) Prefill GEMM
//
// Specialized on Q_BITS / Q_GROUP_SIZE (dequant) and DENSE_W_BF16 (dense).
// Geometry: 8 simdgroups / threadgroup (256 threads) arranged 4x2 along (M, N);
// each simdgroup computes a 16x32 output sub-tile via 2x4 hardware
// `simdgroup_float8x8` accumulators (FP32 accumulation). Stages BK = 32 K-slices
// (two half-groups per 64-element quant group, loading scale_bias once per
// group) and unions A/B staging with per-simdgroup 8x32 Cs epilogue slices to
// fit in 10,240 B (10 KiB) threadgroup memory — allowing >= 2 concurrent
// threadgroups/core on M4 and passing MTL_SHADER_VALIDATION=1 (<= 32 KiB).
// ===========================================================================
constant constexpr uint DGW_BM = 64u;
constant constexpr uint DGW_BN = 64u;
constant constexpr uint DGW_BK = 32u;
constant constexpr uint DGW_BK_PAD = 40u; // +8 elements avoids bank conflicts

union WideGemmSmem {
    struct {
        half As[DGW_BM * DGW_BK_PAD]; // 64 * 40 * 2 = 5,120 B
        half Bs[DGW_BN * DGW_BK_PAD]; // 64 * 40 * 2 = 5,120 B
    } ab;                             // 10,240 B total
    float Cs[8u][8u * 32u];           // 8 simdgroups x 8x32 float = 8,192 B
};

template <bool ADD_RESIDUAL>
static inline void wide_gemm_store_epilogue(threadgroup WideGemmSmem &smem,
                                            thread simdgroup_float8x8 (&acc)[2][4],
                                            device half *y,
                                            const device half *residual,
                                            uint m,
                                            uint n,
                                            uint m_base,
                                            uint n_base,
                                            uint sg_id,
                                            uint sg_m,
                                            uint sg_n,
                                            uint lane) {
    threadgroup float *sg_cs = smem.Cs[sg_id];
    const uint gn = n_base + sg_n + lane;
    for (uint i = 0u; i < 2u; i++) {
        for (uint j = 0u; j < 4u; j++) {
            simdgroup_store(acc[i][j], sg_cs + j * 8u, 32u);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (gn < n) {
            for (uint r = 0u; r < 8u; r++) {
                const uint gm = m_base + sg_m + i * 8u + r;
                if (gm < m) {
                    const ulong idx = ulong(gm) * n + gn;
                    float val = sg_cs[r * 32u + lane];
                    if (ADD_RESIDUAL) {
                        val += float(residual[idx]);
                    }
                    y[idx] = half(val);
                }
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void ami_dequant_gemm_wide(const device half *x [[buffer(0)]],
                                  const device uchar *qweight [[buffer(1)]],
                                  const device half2 *scale_bias [[buffer(2)]],
                                  device half *y [[buffer(3)]],
                                  const device half *residual [[buffer(4)]],
                                  constant DequantParams &p [[buffer(5)]],
                                  uint2 tg_pos [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]],
                                  uint sg_id [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
    const uint n_base = tg_pos.x * DGW_BN;
    const uint m_base = tg_pos.y * DGW_BM;
    if (m_base >= p.m || n_base >= p.n) {
        return;
    }

    threadgroup WideGemmSmem smem;

    simdgroup_float8x8 acc[2][4];
    for (uint i = 0u; i < 2u; i++) {
        for (uint j = 0u; j < 4u; j++) {
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    const uint sg_m = (sg_id >> 1) * 16u;   // 0, 16, 32, 48
    const uint sg_n = (sg_id & 1u) * 32u;   // 0, 32
    const uint load_row = tid >> 2;         // 0..63
    const uint load_chunk = tid & 3u;       // 0..3 (8 elements per thread)
    const uint load_col = load_chunk * 8u;  // 0, 8, 16, 24
    const uint gm_load = m_base + load_row;
    const uint gn_load = n_base + load_row;

    for (uint g = 0u; g < p.num_groups; g++) {
        float scale = 0.0f;
        float bias = 0.0f;
        if (gn_load < p.n) {
            const half2 sb = scale_bias[ulong(gn_load) * p.num_groups + g];
            scale = float(sb.x);
            bias = float(sb.y);
        }

        for (uint sub = 0u; sub < 2u; sub++) {
            const uint k_base = g * 64u + sub * DGW_BK;

            // 1. Cooperative load of X tile [64, 32] -> As
            {
                threadgroup half *a_dst = &smem.ab.As[load_row * DGW_BK_PAD + load_col];
                if (gm_load < p.m) {
                    const device half4 *x_src = (const device half4 *)(x + ulong(gm_load) * p.k + k_base + load_col);
                    *(threadgroup half4 *)(a_dst + 0u) = x_src[0];
                    *(threadgroup half4 *)(a_dst + 4u) = x_src[1];
                } else {
                    const half4 z = half4(0.0h);
                    *(threadgroup half4 *)(a_dst + 0u) = z;
                    *(threadgroup half4 *)(a_dst + 4u) = z;
                }
            }

            // 2. Cooperative fused dequantization of W tile [64, 32] -> Bs
            {
                threadgroup half *b_dst = &smem.ab.Bs[load_row * DGW_BK_PAD + load_col];
                if (gn_load < p.n) {
                    if (Q_BITS == 4u) {
                        const uint w = *(const device uint *)(qweight + ulong(gn_load) * p.row_bytes + g * 32u + sub * 16u + load_chunk * 4u);
                        for (uint j = 0u; j < 8u; j++) {
                            b_dst[j] = half(fma(float((w >> (j * 4u)) & 0xFu), scale, bias));
                        }
                    } else if (Q_BITS == 3u) {
                        const device uchar *wp = qweight + ulong(gn_load) * p.row_bytes + g * 24u + sub * 12u + load_chunk * 3u;
                        const uint word = uint(wp[0]) |
                                          (uint(wp[1]) << 8) |
                                          (uint(wp[2]) << 16);
                        for (uint j = 0u; j < 8u; j++) {
                            b_dst[j] = half(fma(float((word >> (j * 3u)) & 0x7u), scale, bias));
                        }
                    } else if (Q_BITS == 2u) {
                        const ushort w = *(const device ushort *)(qweight + ulong(gn_load) * p.row_bytes + g * 16u + sub * 8u + load_chunk * 2u);
                        for (uint j = 0u; j < 8u; j++) {
                            b_dst[j] = half(fma(float((uint(w) >> (j * 2u)) & 0x3u), scale, bias));
                        }
                    }
                } else {
                    const half4 z = half4(0.0h);
                    *(threadgroup half4 *)(b_dst + 0u) = z;
                    *(threadgroup half4 *)(b_dst + 4u) = z;
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // 3. Hardware 8x8 simdgroup MMA across BK = 32
            for (uint kk = 0u; kk < DGW_BK; kk += 8u) {
                simdgroup_half8x8 a_frag[2];
                simdgroup_half8x8 b_frag[4];
                simdgroup_load(a_frag[0], &smem.ab.As[(sg_m + 0u) * DGW_BK_PAD + kk], DGW_BK_PAD);
                simdgroup_load(a_frag[1], &smem.ab.As[(sg_m + 8u) * DGW_BK_PAD + kk], DGW_BK_PAD);
                simdgroup_load(b_frag[0], &smem.ab.Bs[(sg_n + 0u) * DGW_BK_PAD + kk], DGW_BK_PAD, ulong2(0, 0), true);
                simdgroup_load(b_frag[1], &smem.ab.Bs[(sg_n + 8u) * DGW_BK_PAD + kk], DGW_BK_PAD, ulong2(0, 0), true);
                simdgroup_load(b_frag[2], &smem.ab.Bs[(sg_n + 16u) * DGW_BK_PAD + kk], DGW_BK_PAD, ulong2(0, 0), true);
                simdgroup_load(b_frag[3], &smem.ab.Bs[(sg_n + 24u) * DGW_BK_PAD + kk], DGW_BK_PAD, ulong2(0, 0), true);

                for (uint i = 0u; i < 2u; i++) {
                    for (uint j = 0u; j < 4u; j++) {
                        simdgroup_multiply_accumulate(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // 4. Stage accumulators through per-simdgroup union Cs slices and write guarded [M, N] fp16 outputs
    if (HAS_RESIDUAL_EPILOGUE) {
        wide_gemm_store_epilogue<true>(smem, acc, y, residual, p.m, p.n, m_base, n_base, sg_id, sg_m, sg_n, lane);
    } else {
        wide_gemm_store_epilogue<false>(smem, acc, y, residual, p.m, p.n, m_base, n_base, sg_id, sg_m, sg_n, lane);
    }
}

template <typename T>
static inline void dense_gemm_wide_stage_dims(const device half *x,
                                              const device uchar *w,
                                              threadgroup T *As,
                                              threadgroup T *Bs,
                                              uint m_dim,
                                              uint n_dim,
                                              uint k_dim,
                                              uint m_base,
                                              uint n_base,
                                              uint k_base,
                                              uint tid) {
    const uint ld_row = tid >> 2;          // 0..63
    const uint ld_col = (tid & 3u) * 8u;   // 0, 8, 16, 24
    const uint gk = k_base + ld_col;
    const bool vec_ok = ((k_dim & 3u) == 0u) && (gk + 8u <= k_dim);

    // X tile [64, 32] -> As (converted to T)
    {
        threadgroup T *dst = As + ld_row * DGW_BK_PAD + ld_col;
        const uint gm = m_base + ld_row;
        if (gm < m_dim && vec_ok) {
            const device half4 *src = (const device half4 *)(x + ulong(gm) * k_dim + gk);
            const float4 a0 = float4(src[0]);
            const float4 a1 = float4(src[1]);
            dst[0] = T(a0.x); dst[1] = T(a0.y); dst[2] = T(a0.z); dst[3] = T(a0.w);
            dst[4] = T(a1.x); dst[5] = T(a1.y); dst[6] = T(a1.z); dst[7] = T(a1.w);
        } else {
            for (uint j = 0u; j < 8u; j++) {
                const uint kk = gk + j;
                const float v = (gm < m_dim && kk < k_dim) ? float(x[ulong(gm) * k_dim + kk]) : 0.0f;
                dst[j] = T(v);
            }
        }
    }

    // W tile [64, 32] -> Bs (stored row-major by output column n)
    {
        threadgroup T *dst = Bs + ld_row * DGW_BK_PAD + ld_col;
        const uint gn = n_base + ld_row;
        if (gn < n_dim && vec_ok) {
            const ulong e = ulong(gn) * k_dim + gk;
            const float4 b0 = dense_load_w4(w, e);
            const float4 b1 = dense_load_w4(w, e + 4ul);
            dst[0] = T(b0.x); dst[1] = T(b0.y); dst[2] = T(b0.z); dst[3] = T(b0.w);
            dst[4] = T(b1.x); dst[5] = T(b1.y); dst[6] = T(b1.z); dst[7] = T(b1.w);
        } else {
            for (uint j = 0u; j < 8u; j++) {
                const uint kk = gk + j;
                const float v = (gn < n_dim && kk < k_dim) ? dense_load_w1(w, ulong(gn) * k_dim + kk) : 0.0f;
                dst[j] = T(v);
            }
        }
    }
}

template <typename T>
static inline void dense_gemm_wide_stage(const device half *x,
                                         const device uchar *w,
                                         threadgroup T *As,
                                         threadgroup T *Bs,
                                         constant DenseParams &p,
                                         uint m_base,
                                         uint n_base,
                                         uint k_base,
                                         uint tid) {
    dense_gemm_wide_stage_dims<T>(x, w, As, Bs, p.m, p.n, p.k, m_base, n_base, k_base, tid);
}

template <typename T>
static inline void dense_gemm_wide_mma(threadgroup T *As,
                                       threadgroup T *Bs,
                                       thread simdgroup_float8x8 (&acc)[2][4],
                                       uint sg_m,
                                       uint sg_n) {
    for (uint kk = 0u; kk < DGW_BK; kk += 8u) {
        simdgroup_matrix<T, 8, 8> a_frag[2];
        simdgroup_matrix<T, 8, 8> b_frag[4];
        simdgroup_load(a_frag[0], As + (sg_m + 0u) * DGW_BK_PAD + kk, DGW_BK_PAD);
        simdgroup_load(a_frag[1], As + (sg_m + 8u) * DGW_BK_PAD + kk, DGW_BK_PAD);
        simdgroup_load(b_frag[0], Bs + (sg_n + 0u) * DGW_BK_PAD + kk, DGW_BK_PAD, ulong2(0, 0), true);
        simdgroup_load(b_frag[1], Bs + (sg_n + 8u) * DGW_BK_PAD + kk, DGW_BK_PAD, ulong2(0, 0), true);
        simdgroup_load(b_frag[2], Bs + (sg_n + 16u) * DGW_BK_PAD + kk, DGW_BK_PAD, ulong2(0, 0), true);
        simdgroup_load(b_frag[3], Bs + (sg_n + 24u) * DGW_BK_PAD + kk, DGW_BK_PAD, ulong2(0, 0), true);
        for (uint i = 0u; i < 2u; i++) {
            for (uint j = 0u; j < 4u; j++) {
                simdgroup_multiply_accumulate(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
            }
        }
    }
}

kernel void ami_dense_gemm_wide(const device half *x [[buffer(0)]],
                                const device uchar *w [[buffer(1)]],
                                device half *y [[buffer(2)]],
                                const device half *residual [[buffer(3)]],
                                constant DenseParams &p [[buffer(4)]],
                                uint2 tg_pos [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]],
                                uint sg_id [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    const uint n_base = tg_pos.x * DGW_BN;
    const uint m_base = tg_pos.y * DGW_BM;
    if (m_base >= p.m || n_base >= p.n) {
        return;
    }

    threadgroup WideGemmSmem smem;

    simdgroup_float8x8 acc[2][4];
    for (uint i = 0u; i < 2u; i++) {
        for (uint j = 0u; j < 4u; j++) {
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    const uint sg_m = (sg_id >> 1) * 16u;
    const uint sg_n = (sg_id & 1u) * 32u;

    for (uint k_base = 0u; k_base < p.k; k_base += DGW_BK) {
        if (DENSE_W_BF16) {
            threadgroup bfloat *As = (threadgroup bfloat *)smem.ab.As;
            threadgroup bfloat *Bs = (threadgroup bfloat *)smem.ab.Bs;
            dense_gemm_wide_stage<bfloat>(x, w, As, Bs, p, m_base, n_base, k_base, tid);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            dense_gemm_wide_mma<bfloat>(As, Bs, acc, sg_m, sg_n);
        } else {
            dense_gemm_wide_stage<half>(x, w, smem.ab.As, smem.ab.Bs, p, m_base, n_base, k_base, tid);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            dense_gemm_wide_mma<half>(smem.ab.As, smem.ab.Bs, acc, sg_m, sg_n);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (HAS_RESIDUAL_EPILOGUE) {
        wide_gemm_store_epilogue<true>(smem, acc, y, residual, p.m, p.n, m_base, n_base, sg_id, sg_m, sg_n, lane);
    } else {
        wide_gemm_store_epilogue<false>(smem, acc, y, residual, p.m, p.n, m_base, n_base, sg_id, sg_m, sg_n, lane);
    }
}

// ===========================================================================
// Milestone M7.8 (G18, ADR-0023): Homogeneous-DType Fused QKV & Gate+Up+SwiGLU
//
// Specialized on Q_BITS ({0 = dense f16/bf16, 2/3/4 = affine group-64}),
// Q_GROUP_SIZE (= 64), and DENSE_W_BF16 ({false = f16, true = bf16}).
// Strictly restricted to homogeneous layers (`q.dtype == k.dtype == v.dtype`,
// `gate.dtype == up.dtype`) by `src/gpu/dequant.zig`; mixed-DType GSQ layers
// keep the unfused dispatch path with zero divergence.
// ===========================================================================

// ---------------------------------------------------------------------------
// Kernel 5: Fused Homogeneous QKV GEMV (`M = 1`, single dispatch reading x once
// into L1/L2 across Q/K/V threadgroups with zero warp divergence or register
// inflation). Grid: (tgs_q + tgs_k + tgs_v, 1, 1), 128 threads (16 rows/TG).
// ---------------------------------------------------------------------------
kernel void ami_fused_qkv_gemv(const device half *x [[buffer(0)]],
                               const device uchar *w_q [[buffer(1)]],
                               const device half2 *sb_q [[buffer(2)]],
                               const device uchar *w_k [[buffer(3)]],
                               const device half2 *sb_k [[buffer(4)]],
                               const device uchar *w_v [[buffer(5)]],
                               const device half2 *sb_v [[buffer(6)]],
                               device half *y_q [[buffer(7)]],
                               device half *y_k [[buffer(8)]],
                               device half *y_v [[buffer(9)]],
                               constant FusedQkvParams &p [[buffer(10)]],
                               uint tg_id [[threadgroup_position_in_grid]],
                               uint sg_id [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint ROWS_PER_SG = 4u;
    const device uchar *w_ptr;
    const device half2 *sb_ptr;
    device half *y_ptr;
    uint n_lim;
    uint local_tg;

    if (tg_id < p.tgs_q) {
        w_ptr = w_q;
        sb_ptr = sb_q;
        y_ptr = y_q;
        n_lim = p.n_q;
        local_tg = tg_id;
    } else if (tg_id < p.tgs_q + p.tgs_k) {
        w_ptr = w_k;
        sb_ptr = sb_k;
        y_ptr = y_k;
        n_lim = p.n_k;
        local_tg = tg_id - p.tgs_q;
    } else {
        w_ptr = w_v;
        sb_ptr = sb_v;
        y_ptr = y_v;
        n_lim = p.n_v;
        local_tg = tg_id - p.tgs_q - p.tgs_k;
    }

    const uint row0 = local_tg * 16u + sg_id * ROWS_PER_SG;
    if (row0 >= n_lim) {
        return;
    }

    float acc[ROWS_PER_SG] = {0.0f, 0.0f, 0.0f, 0.0f};

    if (Q_BITS == 0u) {
        ulong row_base[ROWS_PER_SG];
        #pragma clang loop unroll(full)
        for (uint r = 0u; r < ROWS_PER_SG; r++) {
            row_base[r] = ulong(min(row0 + r, n_lim - 1u)) * p.k;
        }
        if ((p.k & 3u) == 0u) {
            const uint vec_k = p.k >> 2;
            const device half4 *x4 = (const device half4 *)x;
            uint i = lane;
            for (; i + 32u < vec_k; i += 64u) {
                const float4 xa = float4(x4[i]);
                const float4 xb = float4(x4[i + 32u]);
                #pragma clang loop unroll(full)
                for (uint r = 0u; r < ROWS_PER_SG; r++) {
                    const float4 wa = dense_load_w4(w_ptr, row_base[r] + ulong(i) * 4ul);
                    const float4 wb = dense_load_w4(w_ptr, row_base[r] + ulong(i + 32u) * 4ul);
                    acc[r] = dense_fma4(xb, wb, dense_fma4(xa, wa, acc[r]));
                }
            }
            for (; i < vec_k; i += 32u) {
                const float4 xa = float4(x4[i]);
                #pragma clang loop unroll(full)
                for (uint r = 0u; r < ROWS_PER_SG; r++) {
                    const float4 wa = dense_load_w4(w_ptr, row_base[r] + ulong(i) * 4ul);
                    acc[r] = dense_fma4(xa, wa, acc[r]);
                }
            }
        } else {
            for (uint kk = lane; kk < p.k; kk += 32u) {
                const float xv = float(x[kk]);
                #pragma clang loop unroll(full)
                for (uint r = 0u; r < ROWS_PER_SG; r++) {
                    acc[r] = fma(xv, dense_load_w1(w_ptr, row_base[r] + kk), acc[r]);
                }
            }
        }
    } else {
        const uint sub_group = lane >> 3;
        const uint lane_in_grp = lane & 7u;
        const bool full_rows = (row0 + ROWS_PER_SG <= n_lim);

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
                #pragma clang loop unroll(full)
                for (uint r = 0u; r < ROWS_PER_SG; r++) {
                    const uint row = row0 + r;
                    if (full_rows || row < n_lim) {
                        const uint32_t w = *(const device uint32_t *)(w_ptr + ulong(row) * p.row_bytes + w_off);
                        const half2 sb = sb_ptr[ulong(row) * p.num_groups + g];
                        acc[r] = fma(qdot8_q4(w, px0, px1), float(sb.x), fma(x_sum, float(sb.y), acc[r]));
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
                #pragma clang loop unroll(full)
                for (uint r = 0u; r < ROWS_PER_SG; r++) {
                    const uint row = row0 + r;
                    if (full_rows || row < n_lim) {
                        const device uchar *wp = w_ptr + ulong(row) * p.row_bytes + w_off;
                        const half2 sb = sb_ptr[ulong(row) * p.num_groups + g];
                        acc[r] = fma(qdot8_q3(wp[0], wp[1], wp[2], px0, px1, x2_hi, x5_hi),
                                     float(sb.x), fma(x_sum, float(sb.y), acc[r]));
                    }
                }
            } else if (Q_BITS == 2u) {
                const float4 px0 = float4(fx0.x,
                                          fx0.y * (1.0f / 4.0f),
                                          fx0.z * (1.0f / 16.0f),
                                          fx0.w * (1.0f / 64.0f));
                const float4 px1 = float4(fx1.x,
                                          fx1.y * (1.0f / 4.0f),
                                          fx1.z * (1.0f / 16.0f),
                                          fx1.w * (1.0f / 64.0f));
                const uint w_off = g * 16u + lane_in_grp * 2u;
                #pragma clang loop unroll(full)
                for (uint r = 0u; r < ROWS_PER_SG; r++) {
                    const uint row = row0 + r;
                    if (full_rows || row < n_lim) {
                        const uint16_t w = *(const device uint16_t *)(w_ptr + ulong(row) * p.row_bytes + w_off);
                        const half2 sb = sb_ptr[ulong(row) * p.num_groups + g];
                        acc[r] = fma(qdot8_q2(w, px0, px1), float(sb.x), fma(x_sum, float(sb.y), acc[r]));
                    }
                }
            }
        }
    }

    #pragma clang loop unroll(full)
    for (uint r = 0u; r < ROWS_PER_SG; r++) {
        const float total = simd_sum(acc[r]);
        if (lane == 0u && (row0 + r) < n_lim) {
            y_ptr[row0 + r] = half(total);
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 6: Fused Homogeneous QKV Wide-Tile GEMM (`M >= 1`, BM = 64, BN = 64,
// BK = 32, 256 threads / 8 simdgroups, 10 KiB union SRAM).
// Grid: (tgs_q + tgs_k + tgs_v, ceil(m / 64), 1), 256 threads.
// ---------------------------------------------------------------------------
kernel void ami_fused_qkv_gemm(const device half *x [[buffer(0)]],
                               const device uchar *w_q [[buffer(1)]],
                               const device half2 *sb_q [[buffer(2)]],
                               const device uchar *w_k [[buffer(3)]],
                               const device half2 *sb_k [[buffer(4)]],
                               const device uchar *w_v [[buffer(5)]],
                               const device half2 *sb_v [[buffer(6)]],
                               device half *y_q [[buffer(7)]],
                               device half *y_k [[buffer(8)]],
                               device half *y_v [[buffer(9)]],
                               constant FusedQkvParams &p [[buffer(10)]],
                               uint2 tg_pos [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]],
                               uint sg_id [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
    const device uchar *w_ptr;
    const device half2 *sb_ptr;
    device half *y_ptr;
    uint n_lim;
    uint local_tg_x;

    if (tg_pos.x < p.tgs_q) {
        w_ptr = w_q;
        sb_ptr = sb_q;
        y_ptr = y_q;
        n_lim = p.n_q;
        local_tg_x = tg_pos.x;
    } else if (tg_pos.x < p.tgs_q + p.tgs_k) {
        w_ptr = w_k;
        sb_ptr = sb_k;
        y_ptr = y_k;
        n_lim = p.n_k;
        local_tg_x = tg_pos.x - p.tgs_q;
    } else {
        w_ptr = w_v;
        sb_ptr = sb_v;
        y_ptr = y_v;
        n_lim = p.n_v;
        local_tg_x = tg_pos.x - p.tgs_q - p.tgs_k;
    }

    const uint n_base = local_tg_x * DGW_BN;
    const uint m_base = tg_pos.y * DGW_BM;
    if (m_base >= p.m || n_base >= n_lim) {
        return;
    }

    threadgroup WideGemmSmem smem;
    simdgroup_float8x8 acc[2][4];
    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 2u; i++) {
        #pragma clang loop unroll(full)
        for (uint j = 0u; j < 4u; j++) {
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    const uint sg_m = (sg_id >> 1) * 16u;
    const uint sg_n = (sg_id & 1u) * 32u;

    if (Q_BITS == 0u) {
        for (uint k_base = 0u; k_base < p.k; k_base += DGW_BK) {
            if (DENSE_W_BF16) {
                threadgroup bfloat *As = (threadgroup bfloat *)smem.ab.As;
                threadgroup bfloat *Bs = (threadgroup bfloat *)smem.ab.Bs;
                dense_gemm_wide_stage_dims<bfloat>(x, w_ptr, As, Bs, p.m, n_lim, p.k, m_base, n_base, k_base, tid);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                dense_gemm_wide_mma<bfloat>(As, Bs, acc, sg_m, sg_n);
            } else {
                dense_gemm_wide_stage_dims<half>(x, w_ptr, smem.ab.As, smem.ab.Bs, p.m, n_lim, p.k, m_base, n_base, k_base, tid);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                dense_gemm_wide_mma<half>(smem.ab.As, smem.ab.Bs, acc, sg_m, sg_n);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        const uint load_row = tid >> 2;
        const uint load_chunk = tid & 3u;
        const uint load_col = load_chunk * 8u;
        const uint gm_load = m_base + load_row;
        const uint gn_load = n_base + load_row;

        for (uint g = 0u; g < p.num_groups; g++) {
            float scale = 0.0f;
            float bias = 0.0f;
            if (gn_load < n_lim) {
                const half2 sb = sb_ptr[ulong(gn_load) * p.num_groups + g];
                scale = float(sb.x);
                bias = float(sb.y);
            }
            for (uint sub = 0u; sub < 2u; sub++) {
                const uint k_base = g * 64u + sub * DGW_BK;
                {
                    threadgroup half *a_dst = &smem.ab.As[load_row * DGW_BK_PAD + load_col];
                    if (gm_load < p.m) {
                        const device half4 *x_src = (const device half4 *)(x + ulong(gm_load) * p.k + k_base + load_col);
                        *(threadgroup half4 *)(a_dst + 0u) = x_src[0];
                        *(threadgroup half4 *)(a_dst + 4u) = x_src[1];
                    } else {
                        const half4 z = half4(0.0h);
                        *(threadgroup half4 *)(a_dst + 0u) = z;
                        *(threadgroup half4 *)(a_dst + 4u) = z;
                    }
                }
                {
                    threadgroup half *b_dst = &smem.ab.Bs[load_row * DGW_BK_PAD + load_col];
                    if (gn_load < n_lim) {
                        if (Q_BITS == 4u) {
                            const uint w = *(const device uint *)(w_ptr + ulong(gn_load) * p.row_bytes + g * 32u + sub * 16u + load_chunk * 4u);
                            for (uint j = 0u; j < 8u; j++) {
                                b_dst[j] = half(fma(float((w >> (j * 4u)) & 0xFu), scale, bias));
                            }
                        } else if (Q_BITS == 3u) {
                            const device uchar *wp = w_ptr + ulong(gn_load) * p.row_bytes + g * 24u + sub * 12u + load_chunk * 3u;
                            const uint word = uint(wp[0]) | (uint(wp[1]) << 8) | (uint(wp[2]) << 16);
                            for (uint j = 0u; j < 8u; j++) {
                                b_dst[j] = half(fma(float((word >> (j * 3u)) & 0x7u), scale, bias));
                            }
                        } else if (Q_BITS == 2u) {
                            const ushort w = *(const device ushort *)(w_ptr + ulong(gn_load) * p.row_bytes + g * 16u + sub * 8u + load_chunk * 2u);
                            for (uint j = 0u; j < 8u; j++) {
                                b_dst[j] = half(fma(float((uint(w) >> (j * 2u)) & 0x3u), scale, bias));
                            }
                        }
                    } else {
                        const half4 z = half4(0.0h);
                        *(threadgroup half4 *)(b_dst + 0u) = z;
                        *(threadgroup half4 *)(b_dst + 4u) = z;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                dense_gemm_wide_mma<half>(smem.ab.As, smem.ab.Bs, acc, sg_m, sg_n);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }

    wide_gemm_store_epilogue<false>(smem, acc, y_ptr, y_ptr, p.m, n_lim, m_base, n_base, sg_id, sg_m, sg_n, lane);
}

// ---------------------------------------------------------------------------
// Kernel 7: Fused Dense FFN `Gate + Up + SwiGLU` GEMV (`M = 1`)
//
// Computes `y[i] = SiLU(dot(W_gate[i], x)) * dot(W_up[i], x)` entirely in
// registers without writing `2 x intermediate_size` scratch activations to
// unified DRAM. Geometry: 4 simdgroups (128 threads), 2 output rows/simdgroup
// (8 rows/threadgroup) -> 4 float accumulators/thread (identical to single GEMV,
// zero stack spills). Grid: (ceil(n / 8), 1, 1), 128 threads.
// ---------------------------------------------------------------------------
kernel void ami_dense_gate_up_swiglu(const device half *x [[buffer(0)]],
                                     const device uchar *w_gate [[buffer(1)]],
                                     const device half2 *sb_gate [[buffer(2)]],
                                     const device uchar *w_up [[buffer(3)]],
                                     const device half2 *sb_up [[buffer(4)]],
                                     device half *y [[buffer(5)]],
                                     constant DequantParams &p [[buffer(6)]],
                                     uint tg_id [[threadgroup_position_in_grid]],
                                     uint sg_id [[simdgroup_index_in_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint ROWS_PER_SG = 2u;
    const uint row0 = tg_id * 8u + sg_id * ROWS_PER_SG;
    if (row0 >= p.n) {
        return;
    }

    float acc_g[ROWS_PER_SG] = {0.0f, 0.0f};
    float acc_u[ROWS_PER_SG] = {0.0f, 0.0f};

    if (Q_BITS == 0u) {
        const ulong row_base0 = ulong(row0) * p.k;
        const ulong row_base1 = ulong(min(row0 + 1u, p.n - 1u)) * p.k;
        if ((p.k & 3u) == 0u) {
            const uint vec_k = p.k >> 2;
            const device half4 *x4 = (const device half4 *)x;
            uint i = lane;
            for (; i + 32u < vec_k; i += 64u) {
                const float4 xa = float4(x4[i]);
                const float4 xb = float4(x4[i + 32u]);
                const ulong ea = ulong(i) * 4ul;
                const ulong eb = ulong(i + 32u) * 4ul;
                acc_g[0] = dense_fma4(xb, dense_load_w4(w_gate, row_base0 + eb),
                           dense_fma4(xa, dense_load_w4(w_gate, row_base0 + ea), acc_g[0]));
                acc_u[0] = dense_fma4(xb, dense_load_w4(w_up,   row_base0 + eb),
                           dense_fma4(xa, dense_load_w4(w_up,   row_base0 + ea), acc_u[0]));
                acc_g[1] = dense_fma4(xb, dense_load_w4(w_gate, row_base1 + eb),
                           dense_fma4(xa, dense_load_w4(w_gate, row_base1 + ea), acc_g[1]));
                acc_u[1] = dense_fma4(xb, dense_load_w4(w_up,   row_base1 + eb),
                           dense_fma4(xa, dense_load_w4(w_up,   row_base1 + ea), acc_u[1]));
            }
            for (; i < vec_k; i += 32u) {
                const float4 xa = float4(x4[i]);
                const ulong ea = ulong(i) * 4ul;
                acc_g[0] = dense_fma4(xa, dense_load_w4(w_gate, row_base0 + ea), acc_g[0]);
                acc_u[0] = dense_fma4(xa, dense_load_w4(w_up,   row_base0 + ea), acc_u[0]);
                acc_g[1] = dense_fma4(xa, dense_load_w4(w_gate, row_base1 + ea), acc_g[1]);
                acc_u[1] = dense_fma4(xa, dense_load_w4(w_up,   row_base1 + ea), acc_u[1]);
            }
        } else {
            for (uint kk = lane; kk < p.k; kk += 32u) {
                const float xv = float(x[kk]);
                acc_g[0] = fma(xv, dense_load_w1(w_gate, row_base0 + kk), acc_g[0]);
                acc_u[0] = fma(xv, dense_load_w1(w_up,   row_base0 + kk), acc_u[0]);
                acc_g[1] = fma(xv, dense_load_w1(w_gate, row_base1 + kk), acc_g[1]);
                acc_u[1] = fma(xv, dense_load_w1(w_up,   row_base1 + kk), acc_u[1]);
            }
        }
    } else {
        const uint sub_group = lane >> 3;
        const uint lane_in_grp = lane & 7u;
        const uint row1 = min(row0 + 1u, p.n - 1u);

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
                #pragma clang loop unroll(full)
                for (uint r = 0u; r < ROWS_PER_SG; r++) {
                    const uint row = (r == 0u) ? row0 : row1;
                    const ulong rb = ulong(row) * p.row_bytes + w_off;
                    const ulong sb_i = ulong(row) * p.num_groups + g;
                    const uint32_t wg = *(const device uint32_t *)(w_gate + rb);
                    const uint32_t wu = *(const device uint32_t *)(w_up + rb);
                    const half2 sbg = sb_gate[sb_i];
                    const half2 sbu = sb_up[sb_i];
                    acc_g[r] = fma(qdot8_q4(wg, px0, px1), float(sbg.x), fma(x_sum, float(sbg.y), acc_g[r]));
                    acc_u[r] = fma(qdot8_q4(wu, px0, px1), float(sbu.x), fma(x_sum, float(sbu.y), acc_u[r]));
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
                #pragma clang loop unroll(full)
                for (uint r = 0u; r < ROWS_PER_SG; r++) {
                    const uint row = (r == 0u) ? row0 : row1;
                    const ulong rb = ulong(row) * p.row_bytes + w_off;
                    const ulong sb_i = ulong(row) * p.num_groups + g;
                    const device uchar *wgp = w_gate + rb;
                    const device uchar *wup = w_up + rb;
                    const half2 sbg = sb_gate[sb_i];
                    const half2 sbu = sb_up[sb_i];
                    acc_g[r] = fma(qdot8_q3(wgp[0], wgp[1], wgp[2], px0, px1, x2_hi, x5_hi),
                                   float(sbg.x), fma(x_sum, float(sbg.y), acc_g[r]));
                    acc_u[r] = fma(qdot8_q3(wup[0], wup[1], wup[2], px0, px1, x2_hi, x5_hi),
                                   float(sbu.x), fma(x_sum, float(sbu.y), acc_u[r]));
                }
            } else if (Q_BITS == 2u) {
                const float4 px0 = float4(fx0.x,
                                          fx0.y * (1.0f / 4.0f),
                                          fx0.z * (1.0f / 16.0f),
                                          fx0.w * (1.0f / 64.0f));
                const float4 px1 = float4(fx1.x,
                                          fx1.y * (1.0f / 4.0f),
                                          fx1.z * (1.0f / 16.0f),
                                          fx1.w * (1.0f / 64.0f));
                const uint w_off = g * 16u + lane_in_grp * 2u;
                #pragma clang loop unroll(full)
                for (uint r = 0u; r < ROWS_PER_SG; r++) {
                    const uint row = (r == 0u) ? row0 : row1;
                    const ulong rb = ulong(row) * p.row_bytes + w_off;
                    const ulong sb_i = ulong(row) * p.num_groups + g;
                    const uint16_t wg = *(const device uint16_t *)(w_gate + rb);
                    const uint16_t wu = *(const device uint16_t *)(w_up + rb);
                    const half2 sbg = sb_gate[sb_i];
                    const half2 sbu = sb_up[sb_i];
                    acc_g[r] = fma(qdot8_q2(wg, px0, px1), float(sbg.x), fma(x_sum, float(sbg.y), acc_g[r]));
                    acc_u[r] = fma(qdot8_q2(wu, px0, px1), float(sbu.x), fma(x_sum, float(sbu.y), acc_u[r]));
                }
            }
        }
    }

    #pragma clang loop unroll(full)
    for (uint r = 0u; r < ROWS_PER_SG; r++) {
        const float g_tot = simd_sum(acc_g[r]);
        const float u_tot = simd_sum(acc_u[r]);
        if (lane == 0u && (row0 + r) < p.n) {
            y[row0 + r] = half(dequant_fast_silu(g_tot) * u_tot);
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 8: Fused Dense FFN `Gate + Up + SwiGLU` GEMM (`M >= 1`)
//
// Tile: BM = 32, BN = 32, BK = 32 (BK_PAD = 40), 4 simdgroups (128 threads),
// 8 KiB union SRAM. Reads X once into As, runs hardware 8x8 MMA for both
// W_gate and W_up into 4+4 = 8 `simdgroup_float8x8` accumulators (same count
// as wide GEMM -> zero spills), and emits `SiLU(gate) * up` directly without
// writing `2 x M x intermediate_size` scratch to DRAM.
// Grid: (ceil(n / 32), ceil(m / 32), 1), 128 threads.
// ---------------------------------------------------------------------------
union SwigluGemmSmem {
    struct {
        half As[DG_BM * DG_BK_PAD];  // 32 * 40 * 2 = 2,560 B
        half Bg[DG_BN * DG_BK_PAD];  // 32 * 40 * 2 = 2,560 B
        half Bu[DG_BN * DG_BK_PAD];  // 32 * 40 * 2 = 2,560 B
    } ab;                            // 7,680 B
    struct {
        float Cg[DG_BM * DG_BN];     // 32 * 32 * 4 = 4,096 B
        float Cu[DG_BM * DG_BN];     // 32 * 32 * 4 = 4,096 B
    } cs;                            // 8,192 B (8 KiB total)
};

template <typename T>
static inline void swiglu_gemm_dense_stage(const device half *x,
                                           const device uchar *w_gate,
                                           const device uchar *w_up,
                                           threadgroup T *As,
                                           threadgroup T *Bg,
                                           threadgroup T *Bu,
                                           constant DequantParams &p,
                                           uint m_base,
                                           uint n_base,
                                           uint k_base,
                                           uint tid) {
    const uint ld_row = tid >> 2;          // 0..31
    const uint ld_col = (tid & 3u) * 8u;   // 0, 8, 16, 24
    const uint gk = k_base + ld_col;
    const bool vec_ok = ((p.k & 3u) == 0u) && (gk + 8u <= p.k);

    {
        threadgroup T *dst = As + ld_row * DG_BK_PAD + ld_col;
        const uint gm = m_base + ld_row;
        if (gm < p.m && vec_ok) {
            const device half4 *src = (const device half4 *)(x + ulong(gm) * p.k + gk);
            const float4 a0 = float4(src[0]);
            const float4 a1 = float4(src[1]);
            dst[0] = T(a0.x); dst[1] = T(a0.y); dst[2] = T(a0.z); dst[3] = T(a0.w);
            dst[4] = T(a1.x); dst[5] = T(a1.y); dst[6] = T(a1.z); dst[7] = T(a1.w);
        } else {
            for (uint j = 0u; j < 8u; j++) {
                const uint kk = gk + j;
                const float v = (gm < p.m && kk < p.k) ? float(x[ulong(gm) * p.k + kk]) : 0.0f;
                dst[j] = T(v);
            }
        }
    }

    const uint gn = n_base + ld_row;
    {
        threadgroup T *dst_g = Bg + ld_row * DG_BK_PAD + ld_col;
        threadgroup T *dst_u = Bu + ld_row * DG_BK_PAD + ld_col;
        if (gn < p.n && vec_ok) {
            const ulong e = ulong(gn) * p.k + gk;
            const float4 g0 = dense_load_w4(w_gate, e);
            const float4 g1 = dense_load_w4(w_gate, e + 4ul);
            dst_g[0] = T(g0.x); dst_g[1] = T(g0.y); dst_g[2] = T(g0.z); dst_g[3] = T(g0.w);
            dst_g[4] = T(g1.x); dst_g[5] = T(g1.y); dst_g[6] = T(g1.z); dst_g[7] = T(g1.w);

            const float4 u0 = dense_load_w4(w_up, e);
            const float4 u1 = dense_load_w4(w_up, e + 4ul);
            dst_u[0] = T(u0.x); dst_u[1] = T(u0.y); dst_u[2] = T(u0.z); dst_u[3] = T(u0.w);
            dst_u[4] = T(u1.x); dst_u[5] = T(u1.y); dst_u[6] = T(u1.z); dst_u[7] = T(u1.w);
        } else {
            for (uint j = 0u; j < 8u; j++) {
                const uint kk = gk + j;
                const ulong idx = ulong(gn) * p.k + kk;
                dst_g[j] = T((gn < p.n && kk < p.k) ? dense_load_w1(w_gate, idx) : 0.0f);
                dst_u[j] = T((gn < p.n && kk < p.k) ? dense_load_w1(w_up, idx) : 0.0f);
            }
        }
    }
}

template <typename T>
static inline void swiglu_gemm_mma(threadgroup T *As,
                                   threadgroup T *Bg,
                                   threadgroup T *Bu,
                                   thread simdgroup_float8x8 (&acc_g)[2][2],
                                   thread simdgroup_float8x8 (&acc_u)[2][2],
                                   uint sg_m,
                                   uint sg_n) {
    for (uint kk = 0u; kk < DG_BK; kk += 8u) {
        simdgroup_matrix<T, 8, 8> a_frag[2];
        simdgroup_matrix<T, 8, 8> b_frag[2];
        simdgroup_load(a_frag[0], As + (sg_m + 0u) * DG_BK_PAD + kk, DG_BK_PAD);
        simdgroup_load(a_frag[1], As + (sg_m + 8u) * DG_BK_PAD + kk, DG_BK_PAD);

        simdgroup_load(b_frag[0], Bg + (sg_n + 0u) * DG_BK_PAD + kk, DG_BK_PAD, ulong2(0, 0), true);
        simdgroup_load(b_frag[1], Bg + (sg_n + 8u) * DG_BK_PAD + kk, DG_BK_PAD, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(acc_g[0][0], a_frag[0], b_frag[0], acc_g[0][0]);
        simdgroup_multiply_accumulate(acc_g[0][1], a_frag[0], b_frag[1], acc_g[0][1]);
        simdgroup_multiply_accumulate(acc_g[1][0], a_frag[1], b_frag[0], acc_g[1][0]);
        simdgroup_multiply_accumulate(acc_g[1][1], a_frag[1], b_frag[1], acc_g[1][1]);

        simdgroup_load(b_frag[0], Bu + (sg_n + 0u) * DG_BK_PAD + kk, DG_BK_PAD, ulong2(0, 0), true);
        simdgroup_load(b_frag[1], Bu + (sg_n + 8u) * DG_BK_PAD + kk, DG_BK_PAD, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(acc_u[0][0], a_frag[0], b_frag[0], acc_u[0][0]);
        simdgroup_multiply_accumulate(acc_u[0][1], a_frag[0], b_frag[1], acc_u[0][1]);
        simdgroup_multiply_accumulate(acc_u[1][0], a_frag[1], b_frag[0], acc_u[1][0]);
        simdgroup_multiply_accumulate(acc_u[1][1], a_frag[1], b_frag[1], acc_u[1][1]);
    }
}

kernel void ami_dense_gate_up_swiglu_gemm(const device half *x [[buffer(0)]],
                                          const device uchar *w_gate [[buffer(1)]],
                                          const device half2 *sb_gate [[buffer(2)]],
                                          const device uchar *w_up [[buffer(3)]],
                                          const device half2 *sb_up [[buffer(4)]],
                                          device half *y [[buffer(5)]],
                                          constant DequantParams &p [[buffer(6)]],
                                          uint2 tg_pos [[threadgroup_position_in_grid]],
                                          uint tid [[thread_index_in_threadgroup]],
                                          uint sg_id [[simdgroup_index_in_threadgroup]]) {
    const uint n_base = tg_pos.x * DG_BN;
    const uint m_base = tg_pos.y * DG_BM;
    if (m_base >= p.m || n_base >= p.n) {
        return;
    }

    threadgroup SwigluGemmSmem smem;
    simdgroup_float8x8 acc_g[2][2];
    simdgroup_float8x8 acc_u[2][2];
    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 2u; i++) {
        #pragma clang loop unroll(full)
        for (uint j = 0u; j < 2u; j++) {
            acc_g[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            acc_u[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    const uint sg_m = (sg_id >> 1) * 16u;
    const uint sg_n = (sg_id & 1u) * 16u;

    if (Q_BITS == 0u) {
        for (uint k_base = 0u; k_base < p.k; k_base += DG_BK) {
            if (DENSE_W_BF16) {
                threadgroup bfloat *As = (threadgroup bfloat *)smem.ab.As;
                threadgroup bfloat *Bg = (threadgroup bfloat *)smem.ab.Bg;
                threadgroup bfloat *Bu = (threadgroup bfloat *)smem.ab.Bu;
                swiglu_gemm_dense_stage<bfloat>(x, w_gate, w_up, As, Bg, Bu, p, m_base, n_base, k_base, tid);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                swiglu_gemm_mma<bfloat>(As, Bg, Bu, acc_g, acc_u, sg_m, sg_n);
            } else {
                swiglu_gemm_dense_stage<half>(x, w_gate, w_up, smem.ab.As, smem.ab.Bg, smem.ab.Bu, p, m_base, n_base, k_base, tid);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                swiglu_gemm_mma<half>(smem.ab.As, smem.ab.Bg, smem.ab.Bu, acc_g, acc_u, sg_m, sg_n);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        const uint load_row = tid >> 2;         // 0..31
        const uint load_chunk = tid & 3u;       // 0..3 (8 elements/thread)
        const uint load_col = load_chunk * 8u;  // 0, 8, 16, 24
        const uint gm_load = m_base + load_row;
        const uint gn_load = n_base + load_row;

        for (uint g = 0u; g < p.num_groups; g++) {
            float sc_g = 0.0f, bi_g = 0.0f;
            float sc_u = 0.0f, bi_u = 0.0f;
            if (gn_load < p.n) {
                const ulong sb_idx = ulong(gn_load) * p.num_groups + g;
                const half2 sbg = sb_gate[sb_idx];
                const half2 sbu = sb_up[sb_idx];
                sc_g = float(sbg.x); bi_g = float(sbg.y);
                sc_u = float(sbu.x); bi_u = float(sbu.y);
            }
            for (uint sub = 0u; sub < 2u; sub++) {
                const uint k_base = g * 64u + sub * DG_BK;
                {
                    threadgroup half *a_dst = smem.ab.As + load_row * DG_BK_PAD + load_col;
                    if (gm_load < p.m) {
                        const device half4 *x_src = (const device half4 *)(x + ulong(gm_load) * p.k + k_base + load_col);
                        *(threadgroup half4 *)(a_dst + 0u) = x_src[0];
                        *(threadgroup half4 *)(a_dst + 4u) = x_src[1];
                    } else {
                        const half4 z = half4(0.0h);
                        *(threadgroup half4 *)(a_dst + 0u) = z;
                        *(threadgroup half4 *)(a_dst + 4u) = z;
                    }
                }
                {
                    threadgroup half *bg_dst = smem.ab.Bg + load_row * DG_BK_PAD + load_col;
                    threadgroup half *bu_dst = smem.ab.Bu + load_row * DG_BK_PAD + load_col;
                    if (gn_load < p.n) {
                        if (Q_BITS == 4u) {
                            const ulong w_off = ulong(gn_load) * p.row_bytes + g * 32u + sub * 16u + load_chunk * 4u;
                            const uint wg = *(const device uint *)(w_gate + w_off);
                            const uint wu = *(const device uint *)(w_up + w_off);
                            for (uint j = 0u; j < 8u; j++) {
                                bg_dst[j] = half(fma(float((wg >> (j * 4u)) & 0xFu), sc_g, bi_g));
                                bu_dst[j] = half(fma(float((wu >> (j * 4u)) & 0xFu), sc_u, bi_u));
                            }
                        } else if (Q_BITS == 3u) {
                            const ulong w_off = ulong(gn_load) * p.row_bytes + g * 24u + sub * 12u + load_chunk * 3u;
                            const device uchar *wgp = w_gate + w_off;
                            const device uchar *wup = w_up + w_off;
                            const uint wg = uint(wgp[0]) | (uint(wgp[1]) << 8) | (uint(wgp[2]) << 16);
                            const uint wu = uint(wup[0]) | (uint(wup[1]) << 8) | (uint(wup[2]) << 16);
                            for (uint j = 0u; j < 8u; j++) {
                                bg_dst[j] = half(fma(float((wg >> (j * 3u)) & 0x7u), sc_g, bi_g));
                                bu_dst[j] = half(fma(float((wu >> (j * 3u)) & 0x7u), sc_u, bi_u));
                            }
                        } else if (Q_BITS == 2u) {
                            const ulong w_off = ulong(gn_load) * p.row_bytes + g * 16u + sub * 8u + load_chunk * 2u;
                            const ushort wg = *(const device ushort *)(w_gate + w_off);
                            const ushort wu = *(const device ushort *)(w_up + w_off);
                            for (uint j = 0u; j < 8u; j++) {
                                bg_dst[j] = half(fma(float((uint(wg) >> (j * 2u)) & 0x3u), sc_g, bi_g));
                                bu_dst[j] = half(fma(float((uint(wu) >> (j * 2u)) & 0x3u), sc_u, bi_u));
                            }
                        }
                    } else {
                        const half4 z = half4(0.0h);
                        *(threadgroup half4 *)(bg_dst + 0u) = z;
                        *(threadgroup half4 *)(bg_dst + 4u) = z;
                        *(threadgroup half4 *)(bu_dst + 0u) = z;
                        *(threadgroup half4 *)(bu_dst + 4u) = z;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                swiglu_gemm_mma<half>(smem.ab.As, smem.ab.Bg, smem.ab.Bu, acc_g, acc_u, sg_m, sg_n);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }

    #pragma clang loop unroll(full)
    for (uint i = 0u; i < 2u; i++) {
        #pragma clang loop unroll(full)
        for (uint j = 0u; j < 2u; j++) {
            const uint off = (sg_m + i * 8u) * DG_BN + (sg_n + j * 8u);
            simdgroup_store(acc_g[i][j], smem.cs.Cg + off, DG_BN);
            simdgroup_store(acc_u[i][j], smem.cs.Cu + off, DG_BN);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tid; idx < DG_BM * DG_BN; idx += 128u) {
        const uint r = idx / DG_BN;
        const uint cc = idx % DG_BN;
        const uint gm = m_base + r;
        const uint gn = n_base + cc;
        if (gm < p.m && gn < p.n) {
            y[ulong(gm) * p.n + gn] = half(dequant_fast_silu(smem.cs.Cg[idx]) * smem.cs.Cu[idx]);
        }
    }
}
