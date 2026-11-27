// Multi-format Metal 4 register-dequantization GEMV, batched-GEMV, MoE gather,
// tiled fp16 prefill staging (G23), and opt-in activation quantization emulation
// (G24) kernels for M8.1 (ADR-0024).
//
// Zero dynamic heap allocation, zero stack spill (-Wframe-larger-than=1),
// strict FP32 accumulation, and <= 16 KiB static threadgroup SRAM.
#include <metal_stdlib>
using namespace metal;

constant uint QF_DTYPE_RAW [[function_constant(0)]];
constant uint QF_GROUP_SIZE_RAW [[function_constant(1)]];
constant bool QF_HAS_RESIDUAL_RAW [[function_constant(2)]];
constant uint QF_ACT_QUANT_MODE_RAW [[function_constant(3)]];

constant uint QF_DTYPE = is_function_constant_defined(QF_DTYPE_RAW) ? QF_DTYPE_RAW : 2u;
constant uint QF_GROUP_SIZE = is_function_constant_defined(QF_GROUP_SIZE_RAW) ? QF_GROUP_SIZE_RAW : 64u;
constant bool QF_HAS_RESIDUAL = is_function_constant_defined(QF_HAS_RESIDUAL_RAW) ? QF_HAS_RESIDUAL_RAW : false;
constant uint QF_ACT_QUANT_MODE = is_function_constant_defined(QF_ACT_QUANT_MODE_RAW) ? QF_ACT_QUANT_MODE_RAW : 0u;

// Canonical .amf DType discriminants (ADR-0024).
constant constexpr uint QF_DT_Q4_AFFINE = 2u;
constant constexpr uint QF_DT_Q3_AFFINE = 3u;
constant constexpr uint QF_DT_Q2_AFFINE = 6u;
constant constexpr uint QF_DT_Q5_AFFINE = 7u;
constant constexpr uint QF_DT_Q6_AFFINE = 8u;
constant constexpr uint QF_DT_Q8_AFFINE = 9u;

constant constexpr uint QF_DT_FP8_BLOCK = 10u;
constant constexpr uint QF_DT_FP8_ROW = 11u;
constant constexpr uint QF_DT_FP8_TENSOR = 12u;
constant constexpr uint QF_DT_MXFP8_G32 = 13u;
constant constexpr uint QF_DT_NVFP4_G16 = 14u;

constant constexpr uint QF_DT_GGUF_Q4_0 = 15u;
constant constexpr uint QF_DT_GGUF_Q4_1 = 16u;
constant constexpr uint QF_DT_GGUF_Q5_0 = 17u;
constant constexpr uint QF_DT_GGUF_Q5_1 = 18u;
constant constexpr uint QF_DT_GGUF_Q8_0 = 19u;
constant constexpr uint QF_DT_GGUF_Q2_K = 20u;
constant constexpr uint QF_DT_GGUF_Q3_K = 21u;
constant constexpr uint QF_DT_GGUF_Q4_K = 22u;
constant constexpr uint QF_DT_GGUF_Q5_K = 23u;
constant constexpr uint QF_DT_GGUF_Q6_K = 24u;
constant constexpr uint QF_DT_GGUF_IQ1_S = 25u;
constant constexpr uint QF_DT_GGUF_IQ1_M = 26u;
constant constexpr uint QF_DT_GGUF_IQ2_XXS = 27u;
constant constexpr uint QF_DT_GGUF_IQ2_XS = 28u;
constant constexpr uint QF_DT_GGUF_IQ2_S = 29u;
constant constexpr uint QF_DT_GGUF_IQ3_XXS = 30u;
constant constexpr uint QF_DT_GGUF_IQ3_S = 31u;
constant constexpr uint QF_DT_GGUF_IQ4_NL = 32u;
constant constexpr uint QF_DT_GGUF_IQ4_XS = 33u;
constant constexpr uint QF_DT_GGUF_TQ1_0 = 34u;
constant constexpr uint QF_DT_GGUF_TQ2_0 = 35u;
constant constexpr uint QF_DT_GGUF_MXFP4 = 36u;
constant constexpr uint QF_DT_GGUF_Q2_0 = 37u;

// G24 ActQuantMode discriminants.
constant constexpr uint QF_ACT_NONE = 0u;
constant constexpr uint QF_ACT_MXFP8_G32 = 1u;
constant constexpr uint QF_ACT_FP8_G128 = 2u;
constant constexpr uint QF_ACT_NVFP4_G16 = 3u;
constant constexpr uint QF_ACT_NVFP4_FOUROVERSIX_MAE = 4u;

struct QuantFormatParams {
    uint m;
    uint n;
    uint k;
    uint num_groups;
    uint row_bytes;
    uint scale_stride;
    uint outer_scale_offset;
    float outer_scale;
    uint tile_n_start;
    uint tile_k_start;
    uint tile_n;
    uint tile_k;
    uint dst_stride;
    uint has_residual;
    uint _pad0;
    uint _pad1;
};

static_assert(sizeof(QuantFormatParams) == 64, "QuantFormatParams must be 64 bytes");

struct ActQuantParams {
    uint m;
    uint k;
    uint mode;
    uint _pad;
};

static_assert(sizeof(ActQuantParams) == 16, "ActQuantParams must be 16 bytes");

struct QfExpertWeights {
    ulong gate_qweight_offset;
    ulong gate_params_offset;
    ulong up_qweight_offset;
    ulong up_params_offset;
    ulong down_qweight_offset;
    ulong down_params_offset;
};

static_assert(sizeof(QfExpertWeights) == 48, "QfExpertWeights must be 48 bytes");

struct QfMoeScheduleHeader {
    uint num_unique_experts;
    uint total_assignments;
    uint m;
    uint top_k;
};

static_assert(sizeof(QfMoeScheduleHeader) == 16, "QfMoeScheduleHeader must be 16 bytes");

struct QfMoeParams {
    uint hidden_size;
    uint intermediate_size;
    uint num_groups;
    uint row_bytes;
    uint top_k;
    uint scale_stride;
    uint outer_scale_offset;
    float outer_scale;
    float swiglu_limit;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

static_assert(sizeof(QfMoeParams) == 48, "QfMoeParams must be 48 bytes");

// ===========================================================================
// Common unaligned-safe readers & scalar FP8/FP4/UE8M0 codecs
// ===========================================================================

static inline ushort qf_read_u16(const device uchar *p) {
    return ushort(p[0]) | (ushort(p[1]) << 8);
}

static inline uint qf_read_u32(const device uchar *p) {
    return uint(p[0]) | (uint(p[1]) << 8) | (uint(p[2]) << 16) | (uint(p[3]) << 24);
}

static inline half qf_read_half(const device uchar *p) {
    return as_type<half>(qf_read_u16(p));
}

static inline float qf_read_f16_as_f32(const device uchar *p) {
    return float(qf_read_half(p));
}

static inline float4 qf_load_half4_safe(const device half *p, bool vec_aligned) {
    if (vec_aligned) {
        return float4(*(const device half4 *)p);
    }
    return float4(float(p[0]), float(p[1]), float(p[2]), float(p[3]));
}

static inline float qf_fast_silu(float x) {
    return x / (1.0f + exp(-x));
}

static inline float qf_apply_swiglu(float g, float u, float limit) {
    float v = qf_fast_silu(g) * u;
    if (limit > 0.0f) {
        v = clamp(v, -limit, limit);
    }
    return v;
}

// OCP FP8 E4M3FN -> f32 (0x7F / 0xFF -> NaN).
static inline float qf_e4m3_to_f32(uchar b) {
    const uint ax = uint(b) & 0x7Fu;
    if (ax == 0x7Fu) {
        return as_type<float>(0x7FC00000u);
    }
    const uint exp4 = (ax >> 3) & 0xFu;
    const uint man3 = ax & 0x7u;
    const uint norm_bits = ((exp4 + 120u) << 23) | (man3 << 20);
    const float mag = (exp4 == 0u) ? (float(man3) * (1.0f / 512.0f)) : as_type<float>(norm_bits);
    return as_type<float>(as_type<uint>(mag) | ((uint(b) & 0x80u) << 24));
}

// Unsigned/scale E4M3 -> f32 (0x7F / 0xFF -> 0.0f, matching ggml_ue4m3_to_fp32).
static inline float qf_e4m3_scale_to_f32(uchar b) {
    const uint ax = uint(b) & 0x7Fu;
    if (ax == 0u || ax == 0x7Fu) {
        return 0.0f;
    }
    const uint exp4 = (ax >> 3) & 0xFu;
    const uint man3 = ax & 0x7u;
    const uint norm_bits = ((exp4 + 120u) << 23) | (man3 << 20);
    const float mag = (exp4 == 0u) ? (float(man3) * (1.0f / 512.0f)) : as_type<float>(norm_bits);
    return as_type<float>(as_type<uint>(mag) | ((uint(b) & 0x80u) << 24));
}

static inline uchar qf_f32_to_e4m3(float x) {
    if (isnan(x)) {
        return 0u;
    }
    const uint bits = as_type<uint>(x);
    const uint sign_bit = (bits >> 31) & 1u;
    const float a = fabs(x);
    if (a == 0.0f) {
        return 0u;
    }
    if (a >= 448.0f) {
        return uchar((sign_bit << 7) | 0x7Eu);
    }
    if (a < (1.0f / 64.0f)) {
        const float scaled = a * 512.0f;
        const uint q_floor = uint(floor(scaled));
        const float frac = scaled - float(q_floor);
        uint q = q_floor;
        if (frac > 0.5f || (frac == 0.5f && (q_floor & 1u) != 0u)) {
            q += 1u;
        }
        if (q == 0u) {
            return 0u;
        }
        if (q >= 8u) {
            return uchar((sign_bit << 7) | (1u << 3));
        }
        return uchar((sign_bit << 7) | q);
    }
    const uint abs_bits = as_type<uint>(a);
    const int f_exp = int((abs_bits >> 23) & 0xFFu) - 127;
    const uint f_mant = abs_bits & 0x7FFFFFu;
    uint m3 = f_mant >> 20;
    const uint rem = f_mant & 0xFFFFFu;
    uint e4 = uint(f_exp + 7);
    if (rem > 0x80000u || (rem == 0x80000u && (m3 & 1u) != 0u)) {
        m3 += 1u;
        if (m3 == 8u) {
            m3 = 0u;
            e4 += 1u;
        }
    }
    if (e4 > 15u || (e4 == 15u && m3 >= 7u)) {
        e4 = 15u;
        m3 = 6u;
    }
    return uchar((sign_bit << 7) | (e4 << 3) | m3);
}

// UE8M0 -> f32: 255 -> NaN, 0 -> 2^-127, else 2^(e - 127).
static inline float qf_ue8m0_to_f32(uchar e) {
    if (e == 255u) {
        return as_type<float>(0x7FC00000u);
    }
    if (e == 0u) {
        return as_type<float>(0x00400000u);
    }
    return as_type<float>(uint(e) << 23);
}

// E2M1 nibble -> f32.
constant constexpr float QF_E2M1_LUT[8] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
};

static inline float qf_e2m1_to_f32(uint nib) {
    const uint mag = nib & 0x7u;
    const float v = QF_E2M1_LUT[mag];
    return ((nib & 0x8u) != 0u && mag != 0u) ? -v : v;
}

static inline uchar qf_f32_to_e2m1(float x) {
    if (isnan(x)) {
        return 0u;
    }
    const bool neg = (x < 0.0f);
    const float a = fabs(x);
    uchar mag;
    if (a <= 0.25f) {
        mag = 0u;
    } else if (a < 0.75f) {
        mag = 1u;
    } else if (a <= 1.25f) {
        mag = 2u;
    } else if (a < 1.75f) {
        mag = 3u;
    } else if (a <= 2.50f) {
        mag = 4u;
    } else if (a < 3.50f) {
        mag = 5u;
    } else if (a <= 5.00f) {
        mag = 6u;
    } else {
        mag = 7u;
    }
    if (mag == 0u) {
        return 0u;
    }
    return neg ? uchar(mag | 0x8u) : mag;
}

// IQ4_NL 16-entry non-linear LUT (exact match to ggml kvalues_iq4nl).
constant constexpr int QF_KVALUES_IQ4NL[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113,
};

// ===========================================================================
// Family 1: Extended MLX Affine (q2/q3/q4/q5/q6/q8_affine, group_size in {32, 64, 128})
// ===========================================================================

static inline uint qf_affine_bits(uint dtype) {
    switch (dtype) {
        case QF_DT_Q2_AFFINE: return 2u;
        case QF_DT_Q3_AFFINE: return 3u;
        case QF_DT_Q4_AFFINE: return 4u;
        case QF_DT_Q5_AFFINE: return 5u;
        case QF_DT_Q6_AFFINE: return 6u;
        case QF_DT_Q8_AFFINE: return 8u;
        default: return 4u;
    }
}

static inline float qf_affine_dot8(const device uchar *wp,
                                     float4 fx0,
                                     float4 fx1,
                                     float scale,
                                     float bias) {
    const uint bits = qf_affine_bits(QF_DTYPE);
    const float x_sum = (fx0.x + fx0.y) + (fx0.z + fx0.w) +
                        (fx1.x + fx1.y) + (fx1.z + fx1.w);
    float qd = 0.0f;
    if (bits == 2u) {
        const uint w = uint(wp[0]) | (uint(wp[1]) << 8);
        qd = fma(float(w & 0x0003u), fx0.x,
             fma(float((w >> 2) & 0x0003u), fx0.y,
             fma(float((w >> 4) & 0x0003u), fx0.z,
             fma(float((w >> 6) & 0x0003u), fx0.w,
             fma(float((w >> 8) & 0x0003u), fx1.x,
             fma(float((w >> 10) & 0x0003u), fx1.y,
             fma(float((w >> 12) & 0x0003u), fx1.z,
                 float((w >> 14) & 0x0003u) * fx1.w)))))));
    } else if (bits == 3u) {
        const uint b0 = wp[0], b1 = wp[1], b2 = wp[2];
        qd = fma(float(b0 & 0x07u), fx0.x,
             fma(float((b0 >> 3) & 0x07u), fx0.y,
             fma(float(((b0 >> 6) & 0x03u) | ((b1 & 0x01u) << 2)), fx0.z,
             fma(float((b1 >> 1) & 0x07u), fx0.w,
             fma(float((b1 >> 4) & 0x07u), fx1.x,
             fma(float(((b1 >> 7) & 0x01u) | ((b2 & 0x03u) << 1)), fx1.y,
             fma(float((b2 >> 2) & 0x07u), fx1.z,
                 float((b2 >> 5) & 0x07u) * fx1.w)))))));
    } else if (bits == 4u) {
        const uint w = qf_read_u32(wp);
        qd = fma(float(w & 0xFu), fx0.x,
             fma(float((w >> 4) & 0xFu), fx0.y,
             fma(float((w >> 8) & 0xFu), fx0.z,
             fma(float((w >> 12) & 0xFu), fx0.w,
             fma(float((w >> 16) & 0xFu), fx1.x,
             fma(float((w >> 20) & 0xFu), fx1.y,
             fma(float((w >> 24) & 0xFu), fx1.z,
                 float((w >> 28) & 0xFu) * fx1.w)))))));
    } else if (bits == 5u) {
        const uint b0 = wp[0], b1 = wp[1], b2 = wp[2], b3 = wp[3], b4 = wp[4];
        qd = fma(float(b0 & 0x1Fu), fx0.x,
             fma(float(((b0 >> 5) & 0x07u) | ((b1 & 0x03u) << 3)), fx0.y,
             fma(float((b1 >> 2) & 0x1Fu), fx0.z,
             fma(float(((b1 >> 7) & 0x01u) | ((b2 & 0x0Fu) << 1)), fx0.w,
             fma(float(((b2 >> 4) & 0x0Fu) | ((b3 & 0x01u) << 4)), fx1.x,
             fma(float((b3 >> 1) & 0x1Fu), fx1.y,
             fma(float(((b3 >> 6) & 0x03u) | ((b4 & 0x07u) << 2)), fx1.z,
                 float((b4 >> 3) & 0x1Fu) * fx1.w)))))));
    } else if (bits == 6u) {
        const uint a0 = wp[0], a1 = wp[1], a2 = wp[2];
        const uint b0 = wp[3], b1 = wp[4], b2 = wp[5];
        qd = fma(float(a0 & 0x3Fu), fx0.x,
             fma(float(((a0 >> 6) & 0x03u) | ((a1 & 0x0Fu) << 2)), fx0.y,
             fma(float(((a1 >> 4) & 0x0Fu) | ((a2 & 0x03u) << 4)), fx0.z,
             fma(float((a2 >> 2) & 0x3Fu), fx0.w,
             fma(float(b0 & 0x3Fu), fx1.x,
             fma(float(((b0 >> 6) & 0x03u) | ((b1 & 0x0Fu) << 2)), fx1.y,
             fma(float(((b1 >> 4) & 0x0Fu) | ((b2 & 0x03u) << 4)), fx1.z,
                 float((b2 >> 2) & 0x3Fu) * fx1.w)))))));
    } else { // bits == 8u
        qd = fma(float(wp[0]), fx0.x,
             fma(float(wp[1]), fx0.y,
             fma(float(wp[2]), fx0.z,
             fma(float(wp[3]), fx0.w,
             fma(float(wp[4]), fx1.x,
             fma(float(wp[5]), fx1.y,
             fma(float(wp[6]), fx1.z,
                 float(wp[7]) * fx1.w)))))));
    }
    return fma(qd, scale, x_sum * bias);
}

static inline void qf_affine_dequant8(const device uchar *wp,
                                      float scale,
                                      float bias,
                                      thread float4 &w0,
                                      thread float4 &w1) {
    const uint bits = qf_affine_bits(QF_DTYPE);
    if (bits == 2u) {
        const uint w = uint(wp[0]) | (uint(wp[1]) << 8);
        w0 = float4(float((w >> 0) & 3u), float((w >> 2) & 3u), float((w >> 4) & 3u), float((w >> 6) & 3u)) * scale + bias;
        w1 = float4(float((w >> 8) & 3u), float((w >> 10) & 3u), float((w >> 12) & 3u), float((w >> 14) & 3u)) * scale + bias;
    } else if (bits == 3u) {
        const uint b0 = wp[0], b1 = wp[1], b2 = wp[2];
        w0 = float4(float(b0 & 7u),
                    float((b0 >> 3) & 7u),
                    float(((b0 >> 6) & 3u) | ((b1 & 1u) << 2)),
                    float((b1 >> 1) & 7u)) * scale + bias;
        w1 = float4(float((b1 >> 4) & 7u),
                    float(((b1 >> 7) & 1u) | ((b2 & 3u) << 1)),
                    float((b2 >> 2) & 7u),
                    float((b2 >> 5) & 7u)) * scale + bias;
    } else if (bits == 4u) {
        const uint w = qf_read_u32(wp);
        w0 = float4(float(w & 0xFu), float((w >> 4) & 0xFu), float((w >> 8) & 0xFu), float((w >> 12) & 0xFu)) * scale + bias;
        w1 = float4(float((w >> 16) & 0xFu), float((w >> 20) & 0xFu), float((w >> 24) & 0xFu), float((w >> 28) & 0xFu)) * scale + bias;
    } else if (bits == 5u) {
        const uint b0 = wp[0], b1 = wp[1], b2 = wp[2], b3 = wp[3], b4 = wp[4];
        w0 = float4(float(b0 & 0x1Fu),
                    float(((b0 >> 5) & 0x07u) | ((b1 & 0x03u) << 3)),
                    float((b1 >> 2) & 0x1Fu),
                    float(((b1 >> 7) & 0x01u) | ((b2 & 0x0Fu) << 1))) * scale + bias;
        w1 = float4(float(((b2 >> 4) & 0x0Fu) | ((b3 & 0x01u) << 4)),
                    float((b3 >> 1) & 0x1Fu),
                    float(((b3 >> 6) & 0x03u) | ((b4 & 0x07u) << 2)),
                    float((b4 >> 3) & 0x1Fu)) * scale + bias;
    } else if (bits == 6u) {
        const uint a0 = wp[0], a1 = wp[1], a2 = wp[2];
        const uint b0 = wp[3], b1 = wp[4], b2 = wp[5];
        w0 = float4(float(a0 & 0x3Fu),
                    float(((a0 >> 6) & 0x03u) | ((a1 & 0x0Fu) << 2)),
                    float(((a1 >> 4) & 0x0Fu) | ((a2 & 0x03u) << 4)),
                    float((a2 >> 2) & 0x3Fu)) * scale + bias;
        w1 = float4(float(b0 & 0x3Fu),
                    float(((b0 >> 6) & 0x03u) | ((b1 & 0x0Fu) << 2)),
                    float(((b1 >> 4) & 0x0Fu) | ((b2 & 0x03u) << 4)),
                    float((b2 >> 2) & 0x3Fu)) * scale + bias;
    } else {
        w0 = float4(float(wp[0]), float(wp[1]), float(wp[2]), float(wp[3])) * scale + bias;
        w1 = float4(float(wp[4]), float(wp[5]), float(wp[6]), float(wp[7])) * scale + bias;
    }
}

kernel void ami_qf_affine_gemv(const device half *x [[buffer(0)]],
                               const device uchar *qweight [[buffer(1)]],
                               const device half2 *scale_bias [[buffer(2)]],
                               device half *y [[buffer(3)]],
                               const device half *residual [[buffer(4)]],
                               constant QuantFormatParams &p [[buffer(5)]],
                               uint tg_id [[threadgroup_position_in_grid]],
                               uint sg_id [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint ROWS_PER_SG = 4u;
    const uint row0 = tg_id * 16u + sg_id * ROWS_PER_SG;
    if (row0 >= p.n) {
        return;
    }

    const uint bits = qf_affine_bits(QF_DTYPE);
    const uint lanes_per_grp = QF_GROUP_SIZE / 8u;
    const uint grps_per_sg = 32u / lanes_per_grp;
    const uint sub_group = lane / lanes_per_grp;
    const uint lane_in_grp = lane & (lanes_per_grp - 1u);
    const uint grp_bytes = (QF_GROUP_SIZE * bits) / 8u;

    float acc[ROWS_PER_SG] = {0.0f, 0.0f, 0.0f, 0.0f};
    const bool full_rows = (row0 + ROWS_PER_SG <= p.n);

    for (uint g_base = 0u; g_base < p.num_groups; g_base += grps_per_sg) {
        const uint g = g_base + sub_group;
        if (g >= p.num_groups) {
            continue;
        }
        const uint k_off = g * QF_GROUP_SIZE + lane_in_grp * 8u;
        const float4 fx0 = float4(*(const device half4 *)(x + k_off));
        const float4 fx1 = float4(*(const device half4 *)(x + k_off + 4u));
        const uint w_off = g * grp_bytes + lane_in_grp * bits;

        for (uint r = 0u; r < ROWS_PER_SG; r++) {
            const uint row = row0 + r;
            if (full_rows || row < p.n) {
                const half2 sb = scale_bias[ulong(row) * p.num_groups + g];
                const device uchar *wp = qweight + ulong(row) * p.row_bytes + w_off;
                acc[r] += qf_affine_dot8(wp, fx0, fx1, float(sb.x), float(sb.y));
            }
        }
    }

    #pragma clang loop unroll(full)
    for (uint r = 0u; r < ROWS_PER_SG; r++) {
        const float total = simd_sum(acc[r]);
        if (lane == 0u && (row0 + r) < p.n) {
            const uint row = row0 + r;
            const bool add_res = QF_HAS_RESIDUAL || (p.has_residual != 0u);
            const float out_val = add_res ? (total + float(residual[row])) : total;
            y[row] = half(out_val);
        }
    }
}

kernel void ami_qf_affine_gemv_batch(const device half *x [[buffer(0)]],
                                     const device uchar *qweight [[buffer(1)]],
                                     const device half2 *scale_bias [[buffer(2)]],
                                     device half *y [[buffer(3)]],
                                     const device half *residual [[buffer(4)]],
                                     constant QuantFormatParams &p [[buffer(5)]],
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

    const uint bits = qf_affine_bits(QF_DTYPE);
    const uint lanes_per_grp = QF_GROUP_SIZE / 8u;
    const uint grps_per_sg = 32u / lanes_per_grp;
    const uint sub_group = lane / lanes_per_grp;
    const uint lane_in_grp = lane & (lanes_per_grp - 1u);
    const uint grp_bytes = (QF_GROUP_SIZE * bits) / 8u;

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

    for (uint g_base = 0u; g_base < p.num_groups; g_base += grps_per_sg) {
        const uint g = g_base + sub_group;
        if (g >= p.num_groups) {
            continue;
        }
        const uint k_off = g * QF_GROUP_SIZE + lane_in_grp * 8u;
        const uint w_off = g * grp_bytes + lane_in_grp * bits;
        const half2 sb0 = sb_row0[g];
        const half2 sb1 = sb_row1[g];
        float4 w0_a, w0_b, w1_a, w1_b;
        qf_affine_dequant8(w_row0 + w_off, float(sb0.x), float(sb0.y), w0_a, w0_b);
        qf_affine_dequant8(w_row1 + w_off, float(sb1.x), float(sb1.y), w1_a, w1_b);

        { const float4 fx0 = float4(*(const device half4 *)(x_base + 0ul * x_stride + k_off)); const float4 fx1 = float4(*(const device half4 *)(x_base + 0ul * x_stride + k_off + 4u)); a0_0 += dot(fx0, w0_a) + dot(fx1, w0_b); a1_0 += dot(fx0, w1_a) + dot(fx1, w1_b); }
        if (tile_m > 1u) { const float4 fx0 = float4(*(const device half4 *)(x_base + 1ul * x_stride + k_off)); const float4 fx1 = float4(*(const device half4 *)(x_base + 1ul * x_stride + k_off + 4u)); a0_1 += dot(fx0, w0_a) + dot(fx1, w0_b); a1_1 += dot(fx0, w1_a) + dot(fx1, w1_b); }
        if (tile_m > 2u) { const float4 fx0 = float4(*(const device half4 *)(x_base + 2ul * x_stride + k_off)); const float4 fx1 = float4(*(const device half4 *)(x_base + 2ul * x_stride + k_off + 4u)); a0_2 += dot(fx0, w0_a) + dot(fx1, w0_b); a1_2 += dot(fx0, w1_a) + dot(fx1, w1_b); }
        if (tile_m > 3u) { const float4 fx0 = float4(*(const device half4 *)(x_base + 3ul * x_stride + k_off)); const float4 fx1 = float4(*(const device half4 *)(x_base + 3ul * x_stride + k_off + 4u)); a0_3 += dot(fx0, w0_a) + dot(fx1, w0_b); a1_3 += dot(fx0, w1_a) + dot(fx1, w1_b); }
        if (tile_m > 4u) {
            const float4 fx0 = float4(*(const device half4 *)(x_base + 4ul * x_stride + k_off)); const float4 fx1 = float4(*(const device half4 *)(x_base + 4ul * x_stride + k_off + 4u)); a0_4 += dot(fx0, w0_a) + dot(fx1, w0_b); a1_4 += dot(fx0, w1_a) + dot(fx1, w1_b);
            if (tile_m > 5u) { const float4 fx0 = float4(*(const device half4 *)(x_base + 5ul * x_stride + k_off)); const float4 fx1 = float4(*(const device half4 *)(x_base + 5ul * x_stride + k_off + 4u)); a0_5 += dot(fx0, w0_a) + dot(fx1, w0_b); a1_5 += dot(fx0, w1_a) + dot(fx1, w1_b); }
            if (tile_m > 6u) { const float4 fx0 = float4(*(const device half4 *)(x_base + 6ul * x_stride + k_off)); const float4 fx1 = float4(*(const device half4 *)(x_base + 6ul * x_stride + k_off + 4u)); a0_6 += dot(fx0, w0_a) + dot(fx1, w0_b); a1_6 += dot(fx0, w1_a) + dot(fx1, w1_b); }
            if (tile_m > 7u) { const float4 fx0 = float4(*(const device half4 *)(x_base + 7ul * x_stride + k_off)); const float4 fx1 = float4(*(const device half4 *)(x_base + 7ul * x_stride + k_off + 4u)); a0_7 += dot(fx0, w0_a) + dot(fx1, w0_b); a1_7 += dot(fx0, w1_a) + dot(fx1, w1_b); }
        }
    }

    const bool add_res = QF_HAS_RESIDUAL || (p.has_residual != 0u);
    #define QF_STORE_PAIR(M_OFF, A0, A1) \
        { \
            const float t0 = simd_sum(A0); \
            const float t1 = simd_sum(A1); \
            if (lane == 0u) { \
                const ulong b = ulong(m_base + M_OFF) * p.n; \
                y[b + row0] = half(add_res ? (t0 + float(residual[b + row0])) : t0); \
                if (has_row1) { \
                    y[b + row1] = half(add_res ? (t1 + float(residual[b + row1])) : t1); \
                } \
            } \
        }
    QF_STORE_PAIR(0u, a0_0, a1_0);
    if (tile_m > 1u) QF_STORE_PAIR(1u, a0_1, a1_1);
    if (tile_m > 2u) QF_STORE_PAIR(2u, a0_2, a1_2);
    if (tile_m > 3u) QF_STORE_PAIR(3u, a0_3, a1_3);
    if (tile_m > 4u) {
        QF_STORE_PAIR(4u, a0_4, a1_4);
        if (tile_m > 5u) QF_STORE_PAIR(5u, a0_5, a1_5);
        if (tile_m > 6u) QF_STORE_PAIR(6u, a0_6, a1_6);
        if (tile_m > 7u) QF_STORE_PAIR(7u, a0_7, a1_7);
    }
    #undef QF_STORE_PAIR
}

kernel void ami_qf_affine_dequant_tile_f16(const device uchar *qweight [[buffer(0)]],
                                           const device half2 *scale_bias [[buffer(1)]],
                                           device half *dst_f16 [[buffer(2)]],
                                           constant QuantFormatParams &p [[buffer(3)]],
                                           uint2 tg_pos [[threadgroup_position_in_grid]],
                                           uint sg_id [[simdgroup_index_in_threadgroup]],
                                           uint lane [[thread_index_in_simdgroup]]) {
    const uint local_r = tg_pos.y * 4u + sg_id;
    if (local_r >= p.tile_n) {
        return;
    }
    const uint col_local = tg_pos.x * 256u + lane * 8u;
    if (col_local >= p.tile_k) {
        return;
    }
    const uint row = p.tile_n_start + local_r;
    const uint col = p.tile_k_start + col_local;
    const uint g = col / QF_GROUP_SIZE;
    const uint in_grp = col - g * QF_GROUP_SIZE;
    const uint bits = qf_affine_bits(QF_DTYPE);
    const uint grp_bytes = (QF_GROUP_SIZE * bits) / 8u;
    const ulong w_off = ulong(row) * p.row_bytes + g * grp_bytes + (in_grp / 8u) * bits;
    const half2 sb = scale_bias[ulong(row) * p.num_groups + g];
    float4 w0, w1;
    qf_affine_dequant8(qweight + w_off, float(sb.x), float(sb.y), w0, w1);
    device half *out = dst_f16 + ulong(local_r) * p.dst_stride + col_local;
    *(device half4 *)(out + 0u) = half4(w0);
    *(device half4 *)(out + 4u) = half4(w1);
}

kernel void ami_qf_affine_moe_gate_up_swiglu(const device half *x [[buffer(0)]],
                                             const device uchar *weights_base [[buffer(1)]],
                                             const device QfExpertWeights *expert_table [[buffer(2)]],
                                             const device QfMoeScheduleHeader *sched_hdr [[buffer(3)]],
                                             const device uint *unique_experts [[buffer(4)]],
                                             const device uint *expert_token_offsets [[buffer(5)]],
                                             const device uint *token_indices [[buffer(6)]],
                                             device half *inter [[buffer(7)]],
                                             constant QfMoeParams &p [[buffer(8)]],
                                             uint2 tg_pos [[threadgroup_position_in_grid]],
                                             uint sg_id [[simdgroup_index_in_threadgroup]],
                                             uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint ROWS_PER_SG = 2u;
    const uint u = tg_pos.y;
    if (u >= sched_hdr->num_unique_experts) {
        return;
    }
    const uint row0 = tg_pos.x * 8u + sg_id * ROWS_PER_SG;
    if (row0 >= p.intermediate_size) {
        return;
    }
    const uint s0 = expert_token_offsets[u];
    const uint s1 = expert_token_offsets[u + 1u];
    const uint count_u = s1 - s0;
    if (count_u == 0u) {
        return;
    }

    const QfExpertWeights ew = expert_table[unique_experts[u]];
    const device uchar *gate_qw = weights_base + ew.gate_qweight_offset;
    const device half2 *gate_sb = (const device half2 *)(weights_base + ew.gate_params_offset);
    const device uchar *up_qw = weights_base + ew.up_qweight_offset;
    const device half2 *up_sb = (const device half2 *)(weights_base + ew.up_params_offset);

    const uint bits = qf_affine_bits(QF_DTYPE);
    const uint lanes_per_grp = QF_GROUP_SIZE / 8u;
    const uint grps_per_sg = 32u / lanes_per_grp;
    const uint sub_group = lane / lanes_per_grp;
    const uint lane_in_grp = lane & (lanes_per_grp - 1u);
    const uint grp_bytes = (QF_GROUP_SIZE * bits) / 8u;

    for (uint t_base = 0u; t_base < count_u; ) {
        const uint rem = min(4u, count_u - t_base);
        const device half *x0 = x + ulong(token_indices[s0 + t_base + 0u]) * p.hidden_size;
        const device half *x1 = x + ulong(token_indices[s0 + t_base + min(1u, rem - 1u)]) * p.hidden_size;
        const device half *x2 = x + ulong(token_indices[s0 + t_base + min(2u, rem - 1u)]) * p.hidden_size;
        const device half *x3 = x + ulong(token_indices[s0 + t_base + min(3u, rem - 1u)]) * p.hidden_size;

        for (uint r = 0u; r < ROWS_PER_SG; r++) {
            const uint row = row0 + r;
            if (row >= p.intermediate_size) {
                break;
            }
            float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
            float u0 = 0.0f, u1 = 0.0f, u2 = 0.0f, u3 = 0.0f;
            const ulong row_w_base = ulong(row) * p.row_bytes;
            const ulong row_sb_base = ulong(row) * p.num_groups;

            for (uint g_base = 0u; g_base < p.num_groups; g_base += grps_per_sg) {
                const uint g = g_base + sub_group;
                if (g >= p.num_groups) {
                    continue;
                }
                const uint k_off = g * QF_GROUP_SIZE + lane_in_grp * 8u;
                const uint w_off = g * grp_bytes + lane_in_grp * bits;
                const half2 gsb = gate_sb[row_sb_base + g];
                const half2 usb = up_sb[row_sb_base + g];
                float4 gw0, gw1, uw0, uw1;
                qf_affine_dequant8(gate_qw + row_w_base + w_off, float(gsb.x), float(gsb.y), gw0, gw1);
                qf_affine_dequant8(up_qw + row_w_base + w_off, float(usb.x), float(usb.y), uw0, uw1);

                { const float4 a = float4(*(const device half4 *)(x0 + k_off)); const float4 b = float4(*(const device half4 *)(x0 + k_off + 4u)); g0 += dot(a, gw0) + dot(b, gw1); u0 += dot(a, uw0) + dot(b, uw1); }
                if (rem > 1u) { const float4 a = float4(*(const device half4 *)(x1 + k_off)); const float4 b = float4(*(const device half4 *)(x1 + k_off + 4u)); g1 += dot(a, gw0) + dot(b, gw1); u1 += dot(a, uw0) + dot(b, uw1); }
                if (rem > 2u) { const float4 a = float4(*(const device half4 *)(x2 + k_off)); const float4 b = float4(*(const device half4 *)(x2 + k_off + 4u)); g2 += dot(a, gw0) + dot(b, gw1); u2 += dot(a, uw0) + dot(b, uw1); }
                if (rem > 3u) { const float4 a = float4(*(const device half4 *)(x3 + k_off)); const float4 b = float4(*(const device half4 *)(x3 + k_off + 4u)); g3 += dot(a, gw0) + dot(b, gw1); u3 += dot(a, uw0) + dot(b, uw1); }
            }

            const float sg0 = simd_sum(g0), su0 = simd_sum(u0);
            const float sg1 = simd_sum(g1), su1 = simd_sum(u1);
            const float sg2 = simd_sum(g2), su2 = simd_sum(u2);
            const float sg3 = simd_sum(g3), su3 = simd_sum(u3);
            if (lane == 0u) {
                inter[ulong(s0 + t_base + 0u) * p.intermediate_size + row] = half(qf_apply_swiglu(sg0, su0, p.swiglu_limit));
                if (rem > 1u) inter[ulong(s0 + t_base + 1u) * p.intermediate_size + row] = half(qf_apply_swiglu(sg1, su1, p.swiglu_limit));
                if (rem > 2u) inter[ulong(s0 + t_base + 2u) * p.intermediate_size + row] = half(qf_apply_swiglu(sg2, su2, p.swiglu_limit));
                if (rem > 3u) inter[ulong(s0 + t_base + 3u) * p.intermediate_size + row] = half(qf_apply_swiglu(sg3, su3, p.swiglu_limit));
            }
        }
        t_base += rem;
    }
}

kernel void ami_qf_affine_moe_down_proj(const device half *inter [[buffer(0)]],
                                        const device uchar *weights_base [[buffer(1)]],
                                        const device QfExpertWeights *expert_table [[buffer(2)]],
                                        const device QfMoeScheduleHeader *sched_hdr [[buffer(3)]],
                                        const device uint *unique_experts [[buffer(4)]],
                                        const device uint *expert_token_offsets [[buffer(5)]],
                                        const device uint *token_indices [[buffer(6)]],
                                        const device float *routing_weights [[buffer(7)]],
                                        const device uint *slot_k_indices [[buffer(8)]],
                                        device float *down_out [[buffer(9)]],
                                        constant QfMoeParams &p [[buffer(10)]],
                                        uint2 tg_pos [[threadgroup_position_in_grid]],
                                        uint sg_id [[simdgroup_index_in_threadgroup]],
                                        uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint ROWS_PER_SG = 2u;
    const uint u = tg_pos.y;
    if (u >= sched_hdr->num_unique_experts) {
        return;
    }
    const uint row0 = tg_pos.x * 8u + sg_id * ROWS_PER_SG;
    if (row0 >= p.hidden_size) {
        return;
    }
    const uint s0 = expert_token_offsets[u];
    const uint s1 = expert_token_offsets[u + 1u];
    const uint count_u = s1 - s0;
    if (count_u == 0u) {
        return;
    }

    const QfExpertWeights ew = expert_table[unique_experts[u]];
    const device uchar *down_qw = weights_base + ew.down_qweight_offset;
    const device half2 *down_sb = (const device half2 *)(weights_base + ew.down_params_offset);

    const uint bits = qf_affine_bits(QF_DTYPE);
    const uint lanes_per_grp = QF_GROUP_SIZE / 8u;
    const uint grps_per_sg = 32u / lanes_per_grp;
    const uint sub_group = lane / lanes_per_grp;
    const uint lane_in_grp = lane & (lanes_per_grp - 1u);
    const uint grp_bytes = (QF_GROUP_SIZE * bits) / 8u;

    for (uint t_base = 0u; t_base < count_u; ) {
        const uint rem = min(4u, count_u - t_base);
        const device half *i0 = inter + ulong(s0 + t_base + 0u) * p.intermediate_size;
        const device half *i1 = inter + ulong(s0 + t_base + min(1u, rem - 1u)) * p.intermediate_size;
        const device half *i2 = inter + ulong(s0 + t_base + min(2u, rem - 1u)) * p.intermediate_size;
        const device half *i3 = inter + ulong(s0 + t_base + min(3u, rem - 1u)) * p.intermediate_size;

        for (uint r = 0u; r < ROWS_PER_SG; r++) {
            const uint row = row0 + r;
            if (row >= p.hidden_size) {
                break;
            }
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            const ulong row_w_base = ulong(row) * p.row_bytes;
            const ulong row_sb_base = ulong(row) * p.num_groups;

            for (uint g_base = 0u; g_base < p.num_groups; g_base += grps_per_sg) {
                const uint g = g_base + sub_group;
                if (g >= p.num_groups) {
                    continue;
                }
                const uint k_off = g * QF_GROUP_SIZE + lane_in_grp * 8u;
                const uint w_off = g * grp_bytes + lane_in_grp * bits;
                const half2 sb = down_sb[row_sb_base + g];
                float4 dw0, dw1;
                qf_affine_dequant8(down_qw + row_w_base + w_off, float(sb.x), float(sb.y), dw0, dw1);

                { const float4 a = float4(*(const device half4 *)(i0 + k_off)); const float4 b = float4(*(const device half4 *)(i0 + k_off + 4u)); a0 += dot(a, dw0) + dot(b, dw1); }
                if (rem > 1u) { const float4 a = float4(*(const device half4 *)(i1 + k_off)); const float4 b = float4(*(const device half4 *)(i1 + k_off + 4u)); a1 += dot(a, dw0) + dot(b, dw1); }
                if (rem > 2u) { const float4 a = float4(*(const device half4 *)(i2 + k_off)); const float4 b = float4(*(const device half4 *)(i2 + k_off + 4u)); a2 += dot(a, dw0) + dot(b, dw1); }
                if (rem > 3u) { const float4 a = float4(*(const device half4 *)(i3 + k_off)); const float4 b = float4(*(const device half4 *)(i3 + k_off + 4u)); a3 += dot(a, dw0) + dot(b, dw1); }
            }

            const float t0 = simd_sum(a0), t1 = simd_sum(a1), t2 = simd_sum(a2), t3 = simd_sum(a3);
            if (lane == 0u) {
                { const uint s = s0 + t_base + 0u; down_out[(ulong(token_indices[s]) * p.top_k + slot_k_indices[s]) * p.hidden_size + row] = t0 * routing_weights[s]; }
                if (rem > 1u) { const uint s = s0 + t_base + 1u; down_out[(ulong(token_indices[s]) * p.top_k + slot_k_indices[s]) * p.hidden_size + row] = t1 * routing_weights[s]; }
                if (rem > 2u) { const uint s = s0 + t_base + 2u; down_out[(ulong(token_indices[s]) * p.top_k + slot_k_indices[s]) * p.hidden_size + row] = t2 * routing_weights[s]; }
                if (rem > 3u) { const uint s = s0 + t_base + 3u; down_out[(ulong(token_indices[s]) * p.top_k + slot_k_indices[s]) * p.hidden_size + row] = t3 * routing_weights[s]; }
            }
        }
        t_base += rem;
    }
}

// ===========================================================================
// Family 2: FP8 (block/row/tensor), MXFP8_G32, and NVFP4_G16
// ===========================================================================

static inline float qf_fp_row_scale_at(const device uchar *scales,
                                        uint row,
                                        uint col,
                                        uint n_cols,
                                        uint num_groups,
                                        uint scale_stride,
                                        float outer_scale) {
    (void)n_cols;
    if (QF_DTYPE == QF_DT_FP8_BLOCK) {
        const device float *sf = (const device float *)scales;
        return sf[ulong(row >> 7) * scale_stride + (col >> 7)];
    } else if (QF_DTYPE == QF_DT_FP8_ROW) {
        const device float *sf = (const device float *)scales;
        return sf[row];
    } else if (QF_DTYPE == QF_DT_FP8_TENSOR) {
        const device float *sf = (const device float *)scales;
        return sf[(scale_stride != 0u) ? row : 0u];
    } else if (QF_DTYPE == QF_DT_MXFP8_G32) {
        return qf_ue8m0_to_f32(scales[ulong(row) * num_groups + (col >> 5)]);
    } else { // QF_DT_NVFP4_G16
        return qf_e4m3_scale_to_f32(scales[ulong(row) * num_groups + (col >> 4)]) * outer_scale;
    }
}

static inline float qf_fp_resolve_outer_scale(const device uchar *scales,
                                              uint outer_scale_offset,
                                              float fallback_outer) {
    if (QF_DTYPE != QF_DT_NVFP4_G16) {
        return 1.0f;
    }
    if (outer_scale_offset != 0xFFFFFFFFu) {
        return *(const device float *)(scales + outer_scale_offset);
    }
    return fallback_outer;
}

static inline void qf_fp_dequant8(const device uchar *w_row,
                                  const device uchar *scales,
                                  uint row,
                                  uint col,
                                  uint n_cols,
                                  uint num_groups,
                                  uint scale_stride,
                                  float outer_scale,
                                  thread float4 &w0,
                                  thread float4 &w1) {
    const float sc = qf_fp_row_scale_at(scales, row, col, n_cols, num_groups, scale_stride, outer_scale);
    if (QF_DTYPE == QF_DT_NVFP4_G16) {
        const uint p4 = qf_read_u32(w_row + (col >> 1));
        w0 = float4(qf_e2m1_to_f32((p4 >> 0) & 0xFu),
                    qf_e2m1_to_f32((p4 >> 4) & 0xFu),
                    qf_e2m1_to_f32((p4 >> 8) & 0xFu),
                    qf_e2m1_to_f32((p4 >> 12) & 0xFu)) * sc;
        w1 = float4(qf_e2m1_to_f32((p4 >> 16) & 0xFu),
                    qf_e2m1_to_f32((p4 >> 20) & 0xFu),
                    qf_e2m1_to_f32((p4 >> 24) & 0xFu),
                    qf_e2m1_to_f32((p4 >> 28) & 0xFu)) * sc;
    } else {
        const device uchar *wp = w_row + col;
        w0 = float4(qf_e4m3_to_f32(wp[0]), qf_e4m3_to_f32(wp[1]),
                    qf_e4m3_to_f32(wp[2]), qf_e4m3_to_f32(wp[3])) * sc;
        w1 = float4(qf_e4m3_to_f32(wp[4]), qf_e4m3_to_f32(wp[5]),
                    qf_e4m3_to_f32(wp[6]), qf_e4m3_to_f32(wp[7])) * sc;
    }
}

static inline float qf_fp_dot_tail(const device half *x,
                                   const device uchar *w_row,
                                   const device uchar *scales,
                                   uint row,
                                   uint col,
                                   uint k_limit,
                                   uint num_groups,
                                   uint scale_stride,
                                   float outer_scale) {
    float acc = 0.0f;
    const uint rem = min(8u, k_limit - col);
    if (QF_DTYPE == QF_DT_NVFP4_G16) {
        const float sc = qf_fp_row_scale_at(scales, row, col, k_limit, num_groups, scale_stride, outer_scale);
        const device uchar *wp = w_row + (col >> 1);
        for (uint t = 0u; t < rem; t++) {
            const uchar byte_val = wp[t >> 1];
            const uint nib = ((t & 1u) != 0u) ? (byte_val >> 4) : (byte_val & 0xFu);
            acc = fma(float(x[col + t]), qf_e2m1_to_f32(nib) * sc, acc);
        }
    } else {
        for (uint t = 0u; t < rem; t++) {
            const float sc = qf_fp_row_scale_at(scales, row, col + t, k_limit, num_groups, scale_stride, outer_scale);
            acc = fma(float(x[col + t]), qf_e4m3_to_f32(w_row[col + t]) * sc, acc);
        }
    }
    return acc;
}

kernel void ami_qf_fp_gemv(const device half *x [[buffer(0)]],
                           const device uchar *qweight [[buffer(1)]],
                           const device uchar *scales [[buffer(2)]],
                           device half *y [[buffer(3)]],
                           const device half *residual [[buffer(4)]],
                           constant QuantFormatParams &p [[buffer(5)]],
                           uint tg_id [[threadgroup_position_in_grid]],
                           uint sg_id [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint ROWS_PER_SG = 4u;
    const uint row0 = tg_id * 16u + sg_id * ROWS_PER_SG;
    if (row0 >= p.n) {
        return;
    }

    const float outer_scale = qf_fp_resolve_outer_scale(scales, p.outer_scale_offset, p.outer_scale);
    float acc[ROWS_PER_SG] = {0.0f, 0.0f, 0.0f, 0.0f};
    const bool full_rows = (row0 + ROWS_PER_SG <= p.n);

    for (uint k_base = 0u; k_base < p.k; k_base += 256u) {
        const uint col = k_base + lane * 8u;
        if (col >= p.k) {
            continue;
        }
        if (col + 8u <= p.k) {
            const float4 fx0 = float4(*(const device half4 *)(x + col));
            const float4 fx1 = float4(*(const device half4 *)(x + col + 4u));
            for (uint r = 0u; r < ROWS_PER_SG; r++) {
                const uint row = row0 + r;
                if (full_rows || row < p.n) {
                    float4 w0, w1;
                    qf_fp_dequant8(qweight + ulong(row) * p.row_bytes, scales, row, col,
                                   p.k, p.num_groups, p.scale_stride, outer_scale, w0, w1);
                    acc[r] += dot(fx0, w0) + dot(fx1, w1);
                }
            }
        } else {
            for (uint r = 0u; r < ROWS_PER_SG; r++) {
                const uint row = row0 + r;
                if (full_rows || row < p.n) {
                    acc[r] += qf_fp_dot_tail(x, qweight + ulong(row) * p.row_bytes, scales, row, col,
                                             p.k, p.num_groups, p.scale_stride, outer_scale);
                }
            }
        }
    }

    #pragma clang loop unroll(full)
    for (uint r = 0u; r < ROWS_PER_SG; r++) {
        const float total = simd_sum(acc[r]);
        if (lane == 0u && (row0 + r) < p.n) {
            const uint row = row0 + r;
            const bool add_res = QF_HAS_RESIDUAL || (p.has_residual != 0u);
            y[row] = half(add_res ? (total + float(residual[row])) : total);
        }
    }
}

kernel void ami_qf_fp_gemv_batch(const device half *x [[buffer(0)]],
                                 const device uchar *qweight [[buffer(1)]],
                                 const device uchar *scales [[buffer(2)]],
                                 device half *y [[buffer(3)]],
                                 const device half *residual [[buffer(4)]],
                                 constant QuantFormatParams &p [[buffer(5)]],
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
    const float outer_scale = qf_fp_resolve_outer_scale(scales, p.outer_scale_offset, p.outer_scale);

    const device uchar *w_row0 = qweight + ulong(row0) * p.row_bytes;
    const device uchar *w_row1 = qweight + ulong(row1) * p.row_bytes;
    const device half *x_base = x + ulong(m_base) * p.k;
    const ulong x_stride = ulong(p.k);
    const bool vec_x = ((p.k & 3u) == 0u);

    float a0_0 = 0.0f, a0_1 = 0.0f, a0_2 = 0.0f, a0_3 = 0.0f;
    float a0_4 = 0.0f, a0_5 = 0.0f, a0_6 = 0.0f, a0_7 = 0.0f;
    float a1_0 = 0.0f, a1_1 = 0.0f, a1_2 = 0.0f, a1_3 = 0.0f;
    float a1_4 = 0.0f, a1_5 = 0.0f, a1_6 = 0.0f, a1_7 = 0.0f;

    for (uint k_base = 0u; k_base < p.k; k_base += 256u) {
        const uint col = k_base + lane * 8u;
        if (col >= p.k) {
            continue;
        }
        if (col + 8u <= p.k) {
            float4 w0_a, w0_b, w1_a, w1_b;
            qf_fp_dequant8(w_row0, scales, row0, col, p.k, p.num_groups, p.scale_stride, outer_scale, w0_a, w0_b);
            qf_fp_dequant8(w_row1, scales, row1, col, p.k, p.num_groups, p.scale_stride, outer_scale, w1_a, w1_b);
            { const float4 fx0 = qf_load_half4_safe(x_base + 0ul * x_stride + col, vec_x); const float4 fx1 = qf_load_half4_safe(x_base + 0ul * x_stride + col + 4u, vec_x); a0_0 += dot(fx0, w0_a) + dot(fx1, w0_b); a1_0 += dot(fx0, w1_a) + dot(fx1, w1_b); }
            if (tile_m > 1u) { const float4 fx0 = qf_load_half4_safe(x_base + 1ul * x_stride + col, vec_x); const float4 fx1 = qf_load_half4_safe(x_base + 1ul * x_stride + col + 4u, vec_x); a0_1 += dot(fx0, w0_a) + dot(fx1, w0_b); a1_1 += dot(fx0, w1_a) + dot(fx1, w1_b); }
            if (tile_m > 2u) { const float4 fx0 = qf_load_half4_safe(x_base + 2ul * x_stride + col, vec_x); const float4 fx1 = qf_load_half4_safe(x_base + 2ul * x_stride + col + 4u, vec_x); a0_2 += dot(fx0, w0_a) + dot(fx1, w0_b); a1_2 += dot(fx0, w1_a) + dot(fx1, w1_b); }
            if (tile_m > 3u) { const float4 fx0 = qf_load_half4_safe(x_base + 3ul * x_stride + col, vec_x); const float4 fx1 = qf_load_half4_safe(x_base + 3ul * x_stride + col + 4u, vec_x); a0_3 += dot(fx0, w0_a) + dot(fx1, w0_b); a1_3 += dot(fx0, w1_a) + dot(fx1, w1_b); }
            if (tile_m > 4u) {
                const float4 fx0 = qf_load_half4_safe(x_base + 4ul * x_stride + col, vec_x); const float4 fx1 = qf_load_half4_safe(x_base + 4ul * x_stride + col + 4u, vec_x); a0_4 += dot(fx0, w0_a) + dot(fx1, w0_b); a1_4 += dot(fx0, w1_a) + dot(fx1, w1_b);
                if (tile_m > 5u) { const float4 fx0 = qf_load_half4_safe(x_base + 5ul * x_stride + col, vec_x); const float4 fx1 = qf_load_half4_safe(x_base + 5ul * x_stride + col + 4u, vec_x); a0_5 += dot(fx0, w0_a) + dot(fx1, w0_b); a1_5 += dot(fx0, w1_a) + dot(fx1, w1_b); }
                if (tile_m > 6u) { const float4 fx0 = qf_load_half4_safe(x_base + 6ul * x_stride + col, vec_x); const float4 fx1 = qf_load_half4_safe(x_base + 6ul * x_stride + col + 4u, vec_x); a0_6 += dot(fx0, w0_a) + dot(fx1, w0_b); a1_6 += dot(fx0, w1_a) + dot(fx1, w1_b); }
                if (tile_m > 7u) { const float4 fx0 = qf_load_half4_safe(x_base + 7ul * x_stride + col, vec_x); const float4 fx1 = qf_load_half4_safe(x_base + 7ul * x_stride + col + 4u, vec_x); a0_7 += dot(fx0, w0_a) + dot(fx1, w0_b); a1_7 += dot(fx0, w1_a) + dot(fx1, w1_b); }
            }
        } else {
            a0_0 += qf_fp_dot_tail(x_base + 0ul * x_stride, w_row0, scales, row0, col, p.k, p.num_groups, p.scale_stride, outer_scale);
            a1_0 += qf_fp_dot_tail(x_base + 0ul * x_stride, w_row1, scales, row1, col, p.k, p.num_groups, p.scale_stride, outer_scale);
            if (tile_m > 1u) { a0_1 += qf_fp_dot_tail(x_base + 1ul * x_stride, w_row0, scales, row0, col, p.k, p.num_groups, p.scale_stride, outer_scale); a1_1 += qf_fp_dot_tail(x_base + 1ul * x_stride, w_row1, scales, row1, col, p.k, p.num_groups, p.scale_stride, outer_scale); }
            if (tile_m > 2u) { a0_2 += qf_fp_dot_tail(x_base + 2ul * x_stride, w_row0, scales, row0, col, p.k, p.num_groups, p.scale_stride, outer_scale); a1_2 += qf_fp_dot_tail(x_base + 2ul * x_stride, w_row1, scales, row1, col, p.k, p.num_groups, p.scale_stride, outer_scale); }
            if (tile_m > 3u) { a0_3 += qf_fp_dot_tail(x_base + 3ul * x_stride, w_row0, scales, row0, col, p.k, p.num_groups, p.scale_stride, outer_scale); a1_3 += qf_fp_dot_tail(x_base + 3ul * x_stride, w_row1, scales, row1, col, p.k, p.num_groups, p.scale_stride, outer_scale); }
            if (tile_m > 4u) {
                a0_4 += qf_fp_dot_tail(x_base + 4ul * x_stride, w_row0, scales, row0, col, p.k, p.num_groups, p.scale_stride, outer_scale); a1_4 += qf_fp_dot_tail(x_base + 4ul * x_stride, w_row1, scales, row1, col, p.k, p.num_groups, p.scale_stride, outer_scale);
                if (tile_m > 5u) { a0_5 += qf_fp_dot_tail(x_base + 5ul * x_stride, w_row0, scales, row0, col, p.k, p.num_groups, p.scale_stride, outer_scale); a1_5 += qf_fp_dot_tail(x_base + 5ul * x_stride, w_row1, scales, row1, col, p.k, p.num_groups, p.scale_stride, outer_scale); }
                if (tile_m > 6u) { a0_6 += qf_fp_dot_tail(x_base + 6ul * x_stride, w_row0, scales, row0, col, p.k, p.num_groups, p.scale_stride, outer_scale); a1_6 += qf_fp_dot_tail(x_base + 6ul * x_stride, w_row1, scales, row1, col, p.k, p.num_groups, p.scale_stride, outer_scale); }
                if (tile_m > 7u) { a0_7 += qf_fp_dot_tail(x_base + 7ul * x_stride, w_row0, scales, row0, col, p.k, p.num_groups, p.scale_stride, outer_scale); a1_7 += qf_fp_dot_tail(x_base + 7ul * x_stride, w_row1, scales, row1, col, p.k, p.num_groups, p.scale_stride, outer_scale); }
            }
        }
    }

    const bool add_res = QF_HAS_RESIDUAL || (p.has_residual != 0u);
    #define QF_STORE_PAIR_FP(M_OFF, A0, A1) \
        { \
            const float t0 = simd_sum(A0); \
            const float t1 = simd_sum(A1); \
            if (lane == 0u) { \
                const ulong b = ulong(m_base + M_OFF) * p.n; \
                y[b + row0] = half(add_res ? (t0 + float(residual[b + row0])) : t0); \
                if (has_row1) { \
                    y[b + row1] = half(add_res ? (t1 + float(residual[b + row1])) : t1); \
                } \
            } \
        }
    QF_STORE_PAIR_FP(0u, a0_0, a1_0);
    if (tile_m > 1u) QF_STORE_PAIR_FP(1u, a0_1, a1_1);
    if (tile_m > 2u) QF_STORE_PAIR_FP(2u, a0_2, a1_2);
    if (tile_m > 3u) QF_STORE_PAIR_FP(3u, a0_3, a1_3);
    if (tile_m > 4u) {
        QF_STORE_PAIR_FP(4u, a0_4, a1_4);
        if (tile_m > 5u) QF_STORE_PAIR_FP(5u, a0_5, a1_5);
        if (tile_m > 6u) QF_STORE_PAIR_FP(6u, a0_6, a1_6);
        if (tile_m > 7u) QF_STORE_PAIR_FP(7u, a0_7, a1_7);
    }
    #undef QF_STORE_PAIR_FP
}

kernel void ami_qf_fp_dequant_tile_f16(const device uchar *qweight [[buffer(0)]],
                                       const device uchar *scales [[buffer(1)]],
                                       device half *dst_f16 [[buffer(2)]],
                                       constant QuantFormatParams &p [[buffer(3)]],
                                       uint2 tg_pos [[threadgroup_position_in_grid]],
                                       uint sg_id [[simdgroup_index_in_threadgroup]],
                                       uint lane [[thread_index_in_simdgroup]]) {
    const uint local_r = tg_pos.y * 4u + sg_id;
    if (local_r >= p.tile_n) {
        return;
    }
    const uint col_local = tg_pos.x * 256u + lane * 8u;
    if (col_local >= p.tile_k) {
        return;
    }
    const uint row = p.tile_n_start + local_r;
    const uint col = p.tile_k_start + col_local;
    const float outer_scale = qf_fp_resolve_outer_scale(scales, p.outer_scale_offset, p.outer_scale);
    const device uchar *w_row = qweight + ulong(row) * p.row_bytes;
    device half *out = dst_f16 + ulong(local_r) * p.dst_stride + col_local;

    if (col_local + 8u <= p.tile_k && (QF_DTYPE != QF_DT_FP8_BLOCK || (p.tile_k_start & 7u) == 0u)) {
        float4 w0, w1;
        qf_fp_dequant8(w_row, scales, row, col, p.k, p.num_groups, p.scale_stride, outer_scale, w0, w1);
        if ((p.dst_stride & 3u) == 0u) {
            *(device half4 *)(out + 0u) = half4(w0);
            *(device half4 *)(out + 4u) = half4(w1);
        } else {
            out[0] = half(w0.x); out[1] = half(w0.y); out[2] = half(w0.z); out[3] = half(w0.w);
            out[4] = half(w1.x); out[5] = half(w1.y); out[6] = half(w1.z); out[7] = half(w1.w);
        }
    } else {
        const uint rem = min(8u, p.tile_k - col_local);
        if (QF_DTYPE == QF_DT_NVFP4_G16) {
            for (uint t = 0u; t < rem; t++) {
                const uint cc = col + t;
                const float sc = qf_fp_row_scale_at(scales, row, cc, p.k, p.num_groups, p.scale_stride, outer_scale);
                const uchar b = w_row[cc >> 1];
                const uint nib = ((cc & 1u) != 0u) ? (b >> 4) : (b & 0xFu);
                out[t] = half(qf_e2m1_to_f32(nib) * sc);
            }
        } else {
            for (uint t = 0u; t < rem; t++) {
                const uint cc = col + t;
                const float sc = qf_fp_row_scale_at(scales, row, cc, p.k, p.num_groups, p.scale_stride, outer_scale);
                out[t] = half(qf_e4m3_to_f32(w_row[cc]) * sc);
            }
        }
    }
}

kernel void ami_qf_fp_moe_gate_up_swiglu(const device half *x [[buffer(0)]],
                                         const device uchar *weights_base [[buffer(1)]],
                                         const device QfExpertWeights *expert_table [[buffer(2)]],
                                         const device QfMoeScheduleHeader *sched_hdr [[buffer(3)]],
                                         const device uint *unique_experts [[buffer(4)]],
                                         const device uint *expert_token_offsets [[buffer(5)]],
                                         const device uint *token_indices [[buffer(6)]],
                                         device half *inter [[buffer(7)]],
                                         constant QfMoeParams &p [[buffer(8)]],
                                         uint2 tg_pos [[threadgroup_position_in_grid]],
                                         uint sg_id [[simdgroup_index_in_threadgroup]],
                                         uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint ROWS_PER_SG = 2u;
    const uint u = tg_pos.y;
    if (u >= sched_hdr->num_unique_experts) {
        return;
    }
    const uint row0 = tg_pos.x * 8u + sg_id * ROWS_PER_SG;
    if (row0 >= p.intermediate_size) {
        return;
    }
    const uint s0 = expert_token_offsets[u];
    const uint s1 = expert_token_offsets[u + 1u];
    const uint count_u = s1 - s0;
    if (count_u == 0u) {
        return;
    }

    const QfExpertWeights ew = expert_table[unique_experts[u]];
    const device uchar *gate_qw = weights_base + ew.gate_qweight_offset;
    const device uchar *gate_sc = weights_base + ew.gate_params_offset;
    const device uchar *up_qw = weights_base + ew.up_qweight_offset;
    const device uchar *up_sc = weights_base + ew.up_params_offset;
    const float gate_outer = qf_fp_resolve_outer_scale(gate_sc, p.outer_scale_offset, p.outer_scale);
    const float up_outer = qf_fp_resolve_outer_scale(up_sc, p.outer_scale_offset, p.outer_scale);

    for (uint t_base = 0u; t_base < count_u; ) {
        const uint rem = min(4u, count_u - t_base);
        const device half *x0 = x + ulong(token_indices[s0 + t_base + 0u]) * p.hidden_size;
        const device half *x1 = x + ulong(token_indices[s0 + t_base + min(1u, rem - 1u)]) * p.hidden_size;
        const device half *x2 = x + ulong(token_indices[s0 + t_base + min(2u, rem - 1u)]) * p.hidden_size;
        const device half *x3 = x + ulong(token_indices[s0 + t_base + min(3u, rem - 1u)]) * p.hidden_size;

        for (uint r = 0u; r < ROWS_PER_SG; r++) {
            const uint row = row0 + r;
            if (row >= p.intermediate_size) {
                break;
            }
            float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
            float u0 = 0.0f, u1 = 0.0f, u2 = 0.0f, u3 = 0.0f;
            const device uchar *gw_row = gate_qw + ulong(row) * p.row_bytes;
            const device uchar *uw_row = up_qw + ulong(row) * p.row_bytes;
            const bool vec_x = ((p.hidden_size & 3u) == 0u);

            for (uint k_base = 0u; k_base < p.hidden_size; k_base += 256u) {
                const uint col = k_base + lane * 8u;
                if (col >= p.hidden_size) {
                    continue;
                }
                if (col + 8u <= p.hidden_size) {
                    float4 gw0, gw1, uw0, uw1;
                    qf_fp_dequant8(gw_row, gate_sc, row, col, p.hidden_size, p.num_groups, p.scale_stride, gate_outer, gw0, gw1);
                    qf_fp_dequant8(uw_row, up_sc, row, col, p.hidden_size, p.num_groups, p.scale_stride, up_outer, uw0, uw1);
                    { const float4 a = qf_load_half4_safe(x0 + col, vec_x); const float4 b = qf_load_half4_safe(x0 + col + 4u, vec_x); g0 += dot(a, gw0) + dot(b, gw1); u0 += dot(a, uw0) + dot(b, uw1); }
                    if (rem > 1u) { const float4 a = qf_load_half4_safe(x1 + col, vec_x); const float4 b = qf_load_half4_safe(x1 + col + 4u, vec_x); g1 += dot(a, gw0) + dot(b, gw1); u1 += dot(a, uw0) + dot(b, uw1); }
                    if (rem > 2u) { const float4 a = qf_load_half4_safe(x2 + col, vec_x); const float4 b = qf_load_half4_safe(x2 + col + 4u, vec_x); g2 += dot(a, gw0) + dot(b, gw1); u2 += dot(a, uw0) + dot(b, uw1); }
                    if (rem > 3u) { const float4 a = qf_load_half4_safe(x3 + col, vec_x); const float4 b = qf_load_half4_safe(x3 + col + 4u, vec_x); g3 += dot(a, gw0) + dot(b, gw1); u3 += dot(a, uw0) + dot(b, uw1); }
                } else {
                    g0 += qf_fp_dot_tail(x0, gw_row, gate_sc, row, col, p.hidden_size, p.num_groups, p.scale_stride, gate_outer);
                    u0 += qf_fp_dot_tail(x0, uw_row, up_sc, row, col, p.hidden_size, p.num_groups, p.scale_stride, up_outer);
                    if (rem > 1u) { g1 += qf_fp_dot_tail(x1, gw_row, gate_sc, row, col, p.hidden_size, p.num_groups, p.scale_stride, gate_outer); u1 += qf_fp_dot_tail(x1, uw_row, up_sc, row, col, p.hidden_size, p.num_groups, p.scale_stride, up_outer); }
                    if (rem > 2u) { g2 += qf_fp_dot_tail(x2, gw_row, gate_sc, row, col, p.hidden_size, p.num_groups, p.scale_stride, gate_outer); u2 += qf_fp_dot_tail(x2, uw_row, up_sc, row, col, p.hidden_size, p.num_groups, p.scale_stride, up_outer); }
                    if (rem > 3u) { g3 += qf_fp_dot_tail(x3, gw_row, gate_sc, row, col, p.hidden_size, p.num_groups, p.scale_stride, gate_outer); u3 += qf_fp_dot_tail(x3, uw_row, up_sc, row, col, p.hidden_size, p.num_groups, p.scale_stride, up_outer); }
                }
            }

            const float sg0 = simd_sum(g0), su0 = simd_sum(u0);
            const float sg1 = simd_sum(g1), su1 = simd_sum(u1);
            const float sg2 = simd_sum(g2), su2 = simd_sum(u2);
            const float sg3 = simd_sum(g3), su3 = simd_sum(u3);
            if (lane == 0u) {
                inter[ulong(s0 + t_base + 0u) * p.intermediate_size + row] = half(qf_apply_swiglu(sg0, su0, p.swiglu_limit));
                if (rem > 1u) inter[ulong(s0 + t_base + 1u) * p.intermediate_size + row] = half(qf_apply_swiglu(sg1, su1, p.swiglu_limit));
                if (rem > 2u) inter[ulong(s0 + t_base + 2u) * p.intermediate_size + row] = half(qf_apply_swiglu(sg2, su2, p.swiglu_limit));
                if (rem > 3u) inter[ulong(s0 + t_base + 3u) * p.intermediate_size + row] = half(qf_apply_swiglu(sg3, su3, p.swiglu_limit));
            }
        }
        t_base += rem;
    }
}

kernel void ami_qf_fp_moe_down_proj(const device half *inter [[buffer(0)]],
                                    const device uchar *weights_base [[buffer(1)]],
                                    const device QfExpertWeights *expert_table [[buffer(2)]],
                                    const device QfMoeScheduleHeader *sched_hdr [[buffer(3)]],
                                    const device uint *unique_experts [[buffer(4)]],
                                    const device uint *expert_token_offsets [[buffer(5)]],
                                    const device uint *token_indices [[buffer(6)]],
                                    const device float *routing_weights [[buffer(7)]],
                                    const device uint *slot_k_indices [[buffer(8)]],
                                    device float *down_out [[buffer(9)]],
                                    constant QfMoeParams &p [[buffer(10)]],
                                    uint2 tg_pos [[threadgroup_position_in_grid]],
                                    uint sg_id [[simdgroup_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint ROWS_PER_SG = 2u;
    const uint u = tg_pos.y;
    if (u >= sched_hdr->num_unique_experts) {
        return;
    }
    const uint row0 = tg_pos.x * 8u + sg_id * ROWS_PER_SG;
    if (row0 >= p.hidden_size) {
        return;
    }
    const uint s0 = expert_token_offsets[u];
    const uint s1 = expert_token_offsets[u + 1u];
    const uint count_u = s1 - s0;
    if (count_u == 0u) {
        return;
    }

    const QfExpertWeights ew = expert_table[unique_experts[u]];
    const device uchar *down_qw = weights_base + ew.down_qweight_offset;
    const device uchar *down_sc = weights_base + ew.down_params_offset;
    const float down_outer = qf_fp_resolve_outer_scale(down_sc, p.outer_scale_offset, p.outer_scale);

    for (uint t_base = 0u; t_base < count_u; ) {
        const uint rem = min(4u, count_u - t_base);
        const device half *i0 = inter + ulong(s0 + t_base + 0u) * p.intermediate_size;
        const device half *i1 = inter + ulong(s0 + t_base + min(1u, rem - 1u)) * p.intermediate_size;
        const device half *i2 = inter + ulong(s0 + t_base + min(2u, rem - 1u)) * p.intermediate_size;
        const device half *i3 = inter + ulong(s0 + t_base + min(3u, rem - 1u)) * p.intermediate_size;

        for (uint r = 0u; r < ROWS_PER_SG; r++) {
            const uint row = row0 + r;
            if (row >= p.hidden_size) {
                break;
            }
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            const device uchar *dw_row = down_qw + ulong(row) * p.row_bytes;
            const bool vec_i = ((p.intermediate_size & 3u) == 0u);

            for (uint k_base = 0u; k_base < p.intermediate_size; k_base += 256u) {
                const uint col = k_base + lane * 8u;
                if (col >= p.intermediate_size) {
                    continue;
                }
                if (col + 8u <= p.intermediate_size) {
                    float4 dw0, dw1;
                    qf_fp_dequant8(dw_row, down_sc, row, col, p.intermediate_size, p.num_groups, p.scale_stride, down_outer, dw0, dw1);
                    { const float4 a = qf_load_half4_safe(i0 + col, vec_i); const float4 b = qf_load_half4_safe(i0 + col + 4u, vec_i); a0 += dot(a, dw0) + dot(b, dw1); }
                    if (rem > 1u) { const float4 a = qf_load_half4_safe(i1 + col, vec_i); const float4 b = qf_load_half4_safe(i1 + col + 4u, vec_i); a1 += dot(a, dw0) + dot(b, dw1); }
                    if (rem > 2u) { const float4 a = qf_load_half4_safe(i2 + col, vec_i); const float4 b = qf_load_half4_safe(i2 + col + 4u, vec_i); a2 += dot(a, dw0) + dot(b, dw1); }
                    if (rem > 3u) { const float4 a = qf_load_half4_safe(i3 + col, vec_i); const float4 b = qf_load_half4_safe(i3 + col + 4u, vec_i); a3 += dot(a, dw0) + dot(b, dw1); }
                } else {
                    a0 += qf_fp_dot_tail(i0, dw_row, down_sc, row, col, p.intermediate_size, p.num_groups, p.scale_stride, down_outer);
                    if (rem > 1u) a1 += qf_fp_dot_tail(i1, dw_row, down_sc, row, col, p.intermediate_size, p.num_groups, p.scale_stride, down_outer);
                    if (rem > 2u) a2 += qf_fp_dot_tail(i2, dw_row, down_sc, row, col, p.intermediate_size, p.num_groups, p.scale_stride, down_outer);
                    if (rem > 3u) a3 += qf_fp_dot_tail(i3, dw_row, down_sc, row, col, p.intermediate_size, p.num_groups, p.scale_stride, down_outer);
                }
            }

            const float t0 = simd_sum(a0), t1 = simd_sum(a1), t2 = simd_sum(a2), t3 = simd_sum(a3);
            if (lane == 0u) {
                { const uint s = s0 + t_base + 0u; down_out[(ulong(token_indices[s]) * p.top_k + slot_k_indices[s]) * p.hidden_size + row] = t0 * routing_weights[s]; }
                if (rem > 1u) { const uint s = s0 + t_base + 1u; down_out[(ulong(token_indices[s]) * p.top_k + slot_k_indices[s]) * p.hidden_size + row] = t1 * routing_weights[s]; }
                if (rem > 2u) { const uint s = s0 + t_base + 2u; down_out[(ulong(token_indices[s]) * p.top_k + slot_k_indices[s]) * p.hidden_size + row] = t2 * routing_weights[s]; }
                if (rem > 3u) { const uint s = s0 + t_base + 3u; down_out[(ulong(token_indices[s]) * p.top_k + slot_k_indices[s]) * p.hidden_size + row] = t3 * routing_weights[s]; }
            }
        }
        t_base += rem;
    }
}

// ===========================================================================
// Family 3: GGUF Block Formats (QK = 32 / 64):
// q4_0, q4_1, q5_0, q5_1, q8_0, iq4_nl, mxfp4, q2_0
// ===========================================================================

static inline uint qf_gguf_blk_bytes(uint dtype) {
    switch (dtype) {
        case QF_DT_GGUF_Q4_0: return 18u;
        case QF_DT_GGUF_Q4_1: return 20u;
        case QF_DT_GGUF_Q5_0: return 22u;
        case QF_DT_GGUF_Q5_1: return 24u;
        case QF_DT_GGUF_Q8_0: return 34u;
        case QF_DT_GGUF_IQ4_NL: return 18u;
        case QF_DT_GGUF_MXFP4: return 17u;
        case QF_DT_GGUF_Q2_0: return 18u;
        default: return 18u;
    }
}

static inline void qf_gguf_blk_k_indices(uint blk_step,
                                          uint lane,
                                          thread uint &k0,
                                          thread uint &k1) {
    if (QF_DTYPE == QF_DT_GGUF_Q2_0) {
        const uint blk_idx = blk_step + (lane >> 3);
        const uint lane_in_blk = lane & 7u;
        k0 = blk_idx * 64u + lane_in_blk * 8u;
        k1 = k0 + 4u;
    } else {
        const uint blk_idx = blk_step + (lane >> 2);
        const uint lane_in_blk = lane & 3u;
        k0 = blk_idx * 32u + lane_in_blk * 4u;
        k1 = k0 + 16u;
    }
}

// Decodes 8 weights for `lane` inside a 256-elem step starting at block `blk_step`:
// - For Q2_0 (QK=64): 4 blocks/step, 8 lanes/blk -> contiguous 8 weights at k0..k0+7 (k1 = k0 + 4).
// - For QK=32 formats: 8 blocks/step, 4 lanes/blk -> split-half 4+4 weights at k0..k0+3 and k1 = k0 + 16..k0+19.
static inline void qf_gguf_blk_dequant8(const device uchar *row_ptr,
                                        uint blk_step,
                                        uint lane,
                                        uint k_total,
                                        thread uint &k0,
                                        thread uint &k1,
                                        thread float4 &w0,
                                        thread float4 &w1) {
    qf_gguf_blk_k_indices(blk_step, lane, k0, k1);
    if (QF_DTYPE == QF_DT_GGUF_Q2_0) {
        const uint blk_idx = blk_step + (lane >> 3);
        const uint lane_in_blk = lane & 7u;
        if (k0 >= k_total) {
            w0 = float4(0.0f);
            w1 = float4(0.0f);
            return;
        }
        const device uchar *blk = row_ptr + ulong(blk_idx) * 18ul;
        const float d = qf_read_f16_as_f32(blk);
        const uchar b0 = blk[2u + lane_in_blk * 2u];
        const uchar b1 = blk[2u + lane_in_blk * 2u + 1u];
        w0 = float4(float(int((b0 >> 0) & 3u) - 1),
                    float(int((b0 >> 2) & 3u) - 1),
                    float(int((b0 >> 4) & 3u) - 1),
                    float(int((b0 >> 6) & 3u) - 1)) * d;
        w1 = float4(float(int((b1 >> 0) & 3u) - 1),
                    float(int((b1 >> 2) & 3u) - 1),
                    float(int((b1 >> 4) & 3u) - 1),
                    float(int((b1 >> 6) & 3u) - 1)) * d;
        return;
    }

    const uint blk_idx = blk_step + (lane >> 2);
    const uint lane_in_blk = lane & 3u;
    if (k0 >= k_total) {
        w0 = float4(0.0f);
        w1 = float4(0.0f);
        return;
    }
    const device uchar *blk = row_ptr + ulong(blk_idx) * qf_gguf_blk_bytes(QF_DTYPE);
    const uint j0 = lane_in_blk * 4u;

    if (QF_DTYPE == QF_DT_GGUF_Q4_0) {
        const float d = qf_read_f16_as_f32(blk);
        const device uchar *qs = blk + 2u + j0;
        w0 = float4(float(int(qs[0] & 0xFu) - 8), float(int(qs[1] & 0xFu) - 8),
                    float(int(qs[2] & 0xFu) - 8), float(int(qs[3] & 0xFu) - 8)) * d;
        w1 = float4(float(int(qs[0] >> 4) - 8), float(int(qs[1] >> 4) - 8),
                    float(int(qs[2] >> 4) - 8), float(int(qs[3] >> 4) - 8)) * d;
    } else if (QF_DTYPE == QF_DT_GGUF_Q4_1) {
        const float d = qf_read_f16_as_f32(blk);
        const float m = qf_read_f16_as_f32(blk + 2u);
        const device uchar *qs = blk + 4u + j0;
        w0 = float4(float(qs[0] & 0xFu), float(qs[1] & 0xFu),
                    float(qs[2] & 0xFu), float(qs[3] & 0xFu)) * d + m;
        w1 = float4(float(qs[0] >> 4), float(qs[1] >> 4),
                    float(qs[2] >> 4), float(qs[3] >> 4)) * d + m;
    } else if (QF_DTYPE == QF_DT_GGUF_Q5_0) {
        const float d = qf_read_f16_as_f32(blk);
        const uint qh = qf_read_u32(blk + 2u);
        const device uchar *qs = blk + 6u + j0;
        const uint h0_0 = ((qh >> (j0 + 0u)) << 4) & 0x10u;
        const uint h0_1 = ((qh >> (j0 + 1u)) << 4) & 0x10u;
        const uint h0_2 = ((qh >> (j0 + 2u)) << 4) & 0x10u;
        const uint h0_3 = ((qh >> (j0 + 3u)) << 4) & 0x10u;
        const uint h1_0 = (qh >> (j0 + 12u)) & 0x10u;
        const uint h1_1 = (qh >> (j0 + 13u)) & 0x10u;
        const uint h1_2 = (qh >> (j0 + 14u)) & 0x10u;
        const uint h1_3 = (qh >> (j0 + 15u)) & 0x10u;
        w0 = float4(float(int((qs[0] & 0xFu) | h0_0) - 16), float(int((qs[1] & 0xFu) | h0_1) - 16),
                    float(int((qs[2] & 0xFu) | h0_2) - 16), float(int((qs[3] & 0xFu) | h0_3) - 16)) * d;
        w1 = float4(float(int((qs[0] >> 4) | h1_0) - 16), float(int((qs[1] >> 4) | h1_1) - 16),
                    float(int((qs[2] >> 4) | h1_2) - 16), float(int((qs[3] >> 4) | h1_3) - 16)) * d;
    } else if (QF_DTYPE == QF_DT_GGUF_Q5_1) {
        const float d = qf_read_f16_as_f32(blk);
        const float m = qf_read_f16_as_f32(blk + 2u);
        const uint qh = qf_read_u32(blk + 4u);
        const device uchar *qs = blk + 8u + j0;
        const uint h0_0 = ((qh >> (j0 + 0u)) << 4) & 0x10u;
        const uint h0_1 = ((qh >> (j0 + 1u)) << 4) & 0x10u;
        const uint h0_2 = ((qh >> (j0 + 2u)) << 4) & 0x10u;
        const uint h0_3 = ((qh >> (j0 + 3u)) << 4) & 0x10u;
        const uint h1_0 = (qh >> (j0 + 12u)) & 0x10u;
        const uint h1_1 = (qh >> (j0 + 13u)) & 0x10u;
        const uint h1_2 = (qh >> (j0 + 14u)) & 0x10u;
        const uint h1_3 = (qh >> (j0 + 15u)) & 0x10u;
        w0 = float4(float((qs[0] & 0xFu) | h0_0), float((qs[1] & 0xFu) | h0_1),
                    float((qs[2] & 0xFu) | h0_2), float((qs[3] & 0xFu) | h0_3)) * d + m;
        w1 = float4(float((qs[0] >> 4) | h1_0), float((qs[1] >> 4) | h1_1),
                    float((qs[2] >> 4) | h1_2), float((qs[3] >> 4) | h1_3)) * d + m;
    } else if (QF_DTYPE == QF_DT_GGUF_Q8_0) {
        const float d = qf_read_f16_as_f32(blk);
        const device char *qs = (const device char *)(blk + 2u);
        w0 = float4(float(qs[j0 + 0u]), float(qs[j0 + 1u]), float(qs[j0 + 2u]), float(qs[j0 + 3u])) * d;
        w1 = float4(float(qs[j0 + 16u]), float(qs[j0 + 17u]), float(qs[j0 + 18u]), float(qs[j0 + 19u])) * d;
    } else if (QF_DTYPE == QF_DT_GGUF_IQ4_NL) {
        const float d = qf_read_f16_as_f32(blk);
        const device uchar *qs = blk + 2u + j0;
        w0 = float4(float(QF_KVALUES_IQ4NL[qs[0] & 0xFu]), float(QF_KVALUES_IQ4NL[qs[1] & 0xFu]),
                    float(QF_KVALUES_IQ4NL[qs[2] & 0xFu]), float(QF_KVALUES_IQ4NL[qs[3] & 0xFu])) * d;
        w1 = float4(float(QF_KVALUES_IQ4NL[qs[0] >> 4]), float(QF_KVALUES_IQ4NL[qs[1] >> 4]),
                    float(QF_KVALUES_IQ4NL[qs[2] >> 4]), float(QF_KVALUES_IQ4NL[qs[3] >> 4])) * d;
    } else { // QF_DT_GGUF_MXFP4
        const float d = qf_ue8m0_to_f32(blk[0]);
        const device uchar *qs = blk + 1u + j0;
        w0 = float4(qf_e2m1_to_f32(qs[0] & 0xFu), qf_e2m1_to_f32(qs[1] & 0xFu),
                    qf_e2m1_to_f32(qs[2] & 0xFu), qf_e2m1_to_f32(qs[3] & 0xFu)) * d;
        w1 = float4(qf_e2m1_to_f32(qs[0] >> 4), qf_e2m1_to_f32(qs[1] >> 4),
                    qf_e2m1_to_f32(qs[2] >> 4), qf_e2m1_to_f32(qs[3] >> 4)) * d;
    }
}

kernel void ami_qf_gguf_blk_gemv(const device half *x [[buffer(0)]],
                                 const device uchar *qweight [[buffer(1)]],
                                 const device uchar *scales [[buffer(2)]],
                                 device half *y [[buffer(3)]],
                                 const device half *residual [[buffer(4)]],
                                 constant QuantFormatParams &p [[buffer(5)]],
                                 uint tg_id [[threadgroup_position_in_grid]],
                                 uint sg_id [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
    (void)scales;
    constexpr uint ROWS_PER_SG = 4u;
    const uint row0 = tg_id * 16u + sg_id * ROWS_PER_SG;
    if (row0 >= p.n) {
        return;
    }
    const uint blks_per_step = (QF_DTYPE == QF_DT_GGUF_Q2_0) ? 4u : 8u;
    float acc[ROWS_PER_SG] = {0.0f, 0.0f, 0.0f, 0.0f};
    const bool full_rows = (row0 + ROWS_PER_SG <= p.n);

    for (uint blk_step = 0u; blk_step < p.num_groups; blk_step += blks_per_step) {
        uint k0, k1;
        qf_gguf_blk_k_indices(blk_step, lane, k0, k1);
        if (k0 >= p.k) {
            continue;
        }
        const float4 fx0 = float4(*(const device half4 *)(x + k0));
        const float4 fx1 = float4(*(const device half4 *)(x + k1));
        for (uint r = 0u; r < ROWS_PER_SG; r++) {
            const uint row = row0 + r;
            if (full_rows || row < p.n) {
                uint tk0, tk1;
                float4 w0, w1;
                qf_gguf_blk_dequant8(qweight + ulong(row) * p.row_bytes, blk_step, lane, p.k, tk0, tk1, w0, w1);
                acc[r] += dot(fx0, w0) + dot(fx1, w1);
            }
        }
    }

    #pragma clang loop unroll(full)
    for (uint r = 0u; r < ROWS_PER_SG; r++) {
        const float total = simd_sum(acc[r]);
        if (lane == 0u && (row0 + r) < p.n) {
            const uint row = row0 + r;
            const bool add_res = QF_HAS_RESIDUAL || (p.has_residual != 0u);
            y[row] = half(add_res ? (total + float(residual[row])) : total);
        }
    }
}

kernel void ami_qf_gguf_blk_gemv_batch(const device half *x [[buffer(0)]],
                                       const device uchar *qweight [[buffer(1)]],
                                       const device uchar *scales [[buffer(2)]],
                                       device half *y [[buffer(3)]],
                                       const device half *residual [[buffer(4)]],
                                       constant QuantFormatParams &p [[buffer(5)]],
                                       uint2 tg_pos [[threadgroup_position_in_grid]],
                                       uint sg_id [[simdgroup_index_in_threadgroup]],
                                       uint lane [[thread_index_in_simdgroup]]) {
    (void)scales;
    const uint row0 = (tg_pos.x * 4u + sg_id) * 2u;
    const uint m_base = tg_pos.y * 8u;
    if (row0 >= p.n || m_base >= p.m) {
        return;
    }
    const uint row1 = min(row0 + 1u, p.n - 1u);
    const bool has_row1 = (row0 + 1u) < p.n;
    const uint tile_m = min(8u, p.m - m_base);
    const uint blks_per_step = (QF_DTYPE == QF_DT_GGUF_Q2_0) ? 4u : 8u;

    const device uchar *w_row0 = qweight + ulong(row0) * p.row_bytes;
    const device uchar *w_row1 = qweight + ulong(row1) * p.row_bytes;
    const device half *x_base = x + ulong(m_base) * p.k;
    const ulong x_stride = ulong(p.k);

    float a0_0 = 0.0f, a0_1 = 0.0f, a0_2 = 0.0f, a0_3 = 0.0f;
    float a0_4 = 0.0f, a0_5 = 0.0f, a0_6 = 0.0f, a0_7 = 0.0f;
    float a1_0 = 0.0f, a1_1 = 0.0f, a1_2 = 0.0f, a1_3 = 0.0f;
    float a1_4 = 0.0f, a1_5 = 0.0f, a1_6 = 0.0f, a1_7 = 0.0f;

    for (uint blk_step = 0u; blk_step < p.num_groups; blk_step += blks_per_step) {
        uint k0, k1;
        float4 w0_a, w0_b, w1_a, w1_b;
        qf_gguf_blk_dequant8(w_row0, blk_step, lane, p.k, k0, k1, w0_a, w0_b);
        if (k0 >= p.k) {
            continue;
        }
        qf_gguf_blk_dequant8(w_row1, blk_step, lane, p.k, k0, k1, w1_a, w1_b);
        { const float4 fx0 = float4(*(const device half4 *)(x_base + 0ul * x_stride + k0)); const float4 fx1 = float4(*(const device half4 *)(x_base + 0ul * x_stride + k1)); a0_0 += dot(fx0, w0_a) + dot(fx1, w0_b); a1_0 += dot(fx0, w1_a) + dot(fx1, w1_b); }
        if (tile_m > 1u) { const float4 fx0 = float4(*(const device half4 *)(x_base + 1ul * x_stride + k0)); const float4 fx1 = float4(*(const device half4 *)(x_base + 1ul * x_stride + k1)); a0_1 += dot(fx0, w0_a) + dot(fx1, w0_b); a1_1 += dot(fx0, w1_a) + dot(fx1, w1_b); }
        if (tile_m > 2u) { const float4 fx0 = float4(*(const device half4 *)(x_base + 2ul * x_stride + k0)); const float4 fx1 = float4(*(const device half4 *)(x_base + 2ul * x_stride + k1)); a0_2 += dot(fx0, w0_a) + dot(fx1, w0_b); a1_2 += dot(fx0, w1_a) + dot(fx1, w1_b); }
        if (tile_m > 3u) { const float4 fx0 = float4(*(const device half4 *)(x_base + 3ul * x_stride + k0)); const float4 fx1 = float4(*(const device half4 *)(x_base + 3ul * x_stride + k1)); a0_3 += dot(fx0, w0_a) + dot(fx1, w0_b); a1_3 += dot(fx0, w1_a) + dot(fx1, w1_b); }
        if (tile_m > 4u) {
            const float4 fx0 = float4(*(const device half4 *)(x_base + 4ul * x_stride + k0)); const float4 fx1 = float4(*(const device half4 *)(x_base + 4ul * x_stride + k1)); a0_4 += dot(fx0, w0_a) + dot(fx1, w0_b); a1_4 += dot(fx0, w1_a) + dot(fx1, w1_b);
            if (tile_m > 5u) { const float4 fx0 = float4(*(const device half4 *)(x_base + 5ul * x_stride + k0)); const float4 fx1 = float4(*(const device half4 *)(x_base + 5ul * x_stride + k1)); a0_5 += dot(fx0, w0_a) + dot(fx1, w0_b); a1_5 += dot(fx0, w1_a) + dot(fx1, w1_b); }
            if (tile_m > 6u) { const float4 fx0 = float4(*(const device half4 *)(x_base + 6ul * x_stride + k0)); const float4 fx1 = float4(*(const device half4 *)(x_base + 6ul * x_stride + k1)); a0_6 += dot(fx0, w0_a) + dot(fx1, w0_b); a1_6 += dot(fx0, w1_a) + dot(fx1, w1_b); }
            if (tile_m > 7u) { const float4 fx0 = float4(*(const device half4 *)(x_base + 7ul * x_stride + k0)); const float4 fx1 = float4(*(const device half4 *)(x_base + 7ul * x_stride + k1)); a0_7 += dot(fx0, w0_a) + dot(fx1, w0_b); a1_7 += dot(fx0, w1_a) + dot(fx1, w1_b); }
        }
    }

    const bool add_res = QF_HAS_RESIDUAL || (p.has_residual != 0u);
    #define QF_STORE_PAIR_BLK(M_OFF, A0, A1) \
        { \
            const float t0 = simd_sum(A0); \
            const float t1 = simd_sum(A1); \
            if (lane == 0u) { \
                const ulong b = ulong(m_base + M_OFF) * p.n; \
                y[b + row0] = half(add_res ? (t0 + float(residual[b + row0])) : t0); \
                if (has_row1) { \
                    y[b + row1] = half(add_res ? (t1 + float(residual[b + row1])) : t1); \
                } \
            } \
        }
    QF_STORE_PAIR_BLK(0u, a0_0, a1_0);
    if (tile_m > 1u) QF_STORE_PAIR_BLK(1u, a0_1, a1_1);
    if (tile_m > 2u) QF_STORE_PAIR_BLK(2u, a0_2, a1_2);
    if (tile_m > 3u) QF_STORE_PAIR_BLK(3u, a0_3, a1_3);
    if (tile_m > 4u) {
        QF_STORE_PAIR_BLK(4u, a0_4, a1_4);
        if (tile_m > 5u) QF_STORE_PAIR_BLK(5u, a0_5, a1_5);
        if (tile_m > 6u) QF_STORE_PAIR_BLK(6u, a0_6, a1_6);
        if (tile_m > 7u) QF_STORE_PAIR_BLK(7u, a0_7, a1_7);
    }
    #undef QF_STORE_PAIR_BLK
}

kernel void ami_qf_gguf_blk_dequant_tile_f16(const device uchar *qweight [[buffer(0)]],
                                             const device uchar *scales [[buffer(1)]],
                                             device half *dst_f16 [[buffer(2)]],
                                             constant QuantFormatParams &p [[buffer(3)]],
                                             uint2 tg_pos [[threadgroup_position_in_grid]],
                                             uint sg_id [[simdgroup_index_in_threadgroup]],
                                             uint lane [[thread_index_in_simdgroup]]) {
    (void)scales;
    const uint local_r = tg_pos.y * 4u + sg_id;
    if (local_r >= p.tile_n) {
        return;
    }
    const uint k_step_base = p.tile_k_start + tg_pos.x * 256u;
    const uint qk = (QF_DTYPE == QF_DT_GGUF_Q2_0) ? 64u : 32u;
    const uint blk_step = k_step_base / qk;
    const uint row = p.tile_n_start + local_r;
    uint k0, k1;
    float4 w0, w1;
    qf_gguf_blk_dequant8(qweight + ulong(row) * p.row_bytes, blk_step, lane, p.k, k0, k1, w0, w1);
    if (k0 < p.tile_k_start + p.tile_k) {
        device half *out = dst_f16 + ulong(local_r) * p.dst_stride;
        *(device half4 *)(out + (k0 - p.tile_k_start)) = half4(w0);
        *(device half4 *)(out + (k1 - p.tile_k_start)) = half4(w1);
    }
}

kernel void ami_qf_gguf_blk_moe_gate_up_swiglu(const device half *x [[buffer(0)]],
                                               const device uchar *weights_base [[buffer(1)]],
                                               const device QfExpertWeights *expert_table [[buffer(2)]],
                                               const device QfMoeScheduleHeader *sched_hdr [[buffer(3)]],
                                               const device uint *unique_experts [[buffer(4)]],
                                               const device uint *expert_token_offsets [[buffer(5)]],
                                               const device uint *token_indices [[buffer(6)]],
                                               device half *inter [[buffer(7)]],
                                               constant QfMoeParams &p [[buffer(8)]],
                                               uint2 tg_pos [[threadgroup_position_in_grid]],
                                               uint sg_id [[simdgroup_index_in_threadgroup]],
                                               uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint ROWS_PER_SG = 2u;
    const uint u = tg_pos.y;
    if (u >= sched_hdr->num_unique_experts) {
        return;
    }
    const uint row0 = tg_pos.x * 8u + sg_id * ROWS_PER_SG;
    if (row0 >= p.intermediate_size) {
        return;
    }
    const uint s0 = expert_token_offsets[u];
    const uint s1 = expert_token_offsets[u + 1u];
    const uint count_u = s1 - s0;
    if (count_u == 0u) {
        return;
    }
    const QfExpertWeights ew = expert_table[unique_experts[u]];
    const device uchar *gate_qw = weights_base + ew.gate_qweight_offset;
    const device uchar *up_qw = weights_base + ew.up_qweight_offset;
    const uint blks_per_step = (QF_DTYPE == QF_DT_GGUF_Q2_0) ? 4u : 8u;

    for (uint t_base = 0u; t_base < count_u; ) {
        const uint rem = min(4u, count_u - t_base);
        const device half *x0 = x + ulong(token_indices[s0 + t_base + 0u]) * p.hidden_size;
        const device half *x1 = x + ulong(token_indices[s0 + t_base + min(1u, rem - 1u)]) * p.hidden_size;
        const device half *x2 = x + ulong(token_indices[s0 + t_base + min(2u, rem - 1u)]) * p.hidden_size;
        const device half *x3 = x + ulong(token_indices[s0 + t_base + min(3u, rem - 1u)]) * p.hidden_size;

        for (uint r = 0u; r < ROWS_PER_SG; r++) {
            const uint row = row0 + r;
            if (row >= p.intermediate_size) {
                break;
            }
            float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
            float u0 = 0.0f, u1 = 0.0f, u2 = 0.0f, u3 = 0.0f;
            const device uchar *gw_row = gate_qw + ulong(row) * p.row_bytes;
            const device uchar *uw_row = up_qw + ulong(row) * p.row_bytes;

            for (uint blk_step = 0u; blk_step < p.num_groups; blk_step += blks_per_step) {
                uint k0, k1;
                float4 gw0, gw1, uw0, uw1;
                qf_gguf_blk_dequant8(gw_row, blk_step, lane, p.hidden_size, k0, k1, gw0, gw1);
                if (k0 >= p.hidden_size) {
                    continue;
                }
                qf_gguf_blk_dequant8(uw_row, blk_step, lane, p.hidden_size, k0, k1, uw0, uw1);
                { const float4 a = float4(*(const device half4 *)(x0 + k0)); const float4 b = float4(*(const device half4 *)(x0 + k1)); g0 += dot(a, gw0) + dot(b, gw1); u0 += dot(a, uw0) + dot(b, uw1); }
                if (rem > 1u) { const float4 a = float4(*(const device half4 *)(x1 + k0)); const float4 b = float4(*(const device half4 *)(x1 + k1)); g1 += dot(a, gw0) + dot(b, gw1); u1 += dot(a, uw0) + dot(b, uw1); }
                if (rem > 2u) { const float4 a = float4(*(const device half4 *)(x2 + k0)); const float4 b = float4(*(const device half4 *)(x2 + k1)); g2 += dot(a, gw0) + dot(b, gw1); u2 += dot(a, uw0) + dot(b, uw1); }
                if (rem > 3u) { const float4 a = float4(*(const device half4 *)(x3 + k0)); const float4 b = float4(*(const device half4 *)(x3 + k1)); g3 += dot(a, gw0) + dot(b, gw1); u3 += dot(a, uw0) + dot(b, uw1); }
            }

            const float sg0 = simd_sum(g0), su0 = simd_sum(u0);
            const float sg1 = simd_sum(g1), su1 = simd_sum(u1);
            const float sg2 = simd_sum(g2), su2 = simd_sum(u2);
            const float sg3 = simd_sum(g3), su3 = simd_sum(u3);
            if (lane == 0u) {
                inter[ulong(s0 + t_base + 0u) * p.intermediate_size + row] = half(qf_apply_swiglu(sg0, su0, p.swiglu_limit));
                if (rem > 1u) inter[ulong(s0 + t_base + 1u) * p.intermediate_size + row] = half(qf_apply_swiglu(sg1, su1, p.swiglu_limit));
                if (rem > 2u) inter[ulong(s0 + t_base + 2u) * p.intermediate_size + row] = half(qf_apply_swiglu(sg2, su2, p.swiglu_limit));
                if (rem > 3u) inter[ulong(s0 + t_base + 3u) * p.intermediate_size + row] = half(qf_apply_swiglu(sg3, su3, p.swiglu_limit));
            }
        }
        t_base += rem;
    }
}

kernel void ami_qf_gguf_blk_moe_down_proj(const device half *inter [[buffer(0)]],
                                          const device uchar *weights_base [[buffer(1)]],
                                          const device QfExpertWeights *expert_table [[buffer(2)]],
                                          const device QfMoeScheduleHeader *sched_hdr [[buffer(3)]],
                                          const device uint *unique_experts [[buffer(4)]],
                                          const device uint *expert_token_offsets [[buffer(5)]],
                                          const device uint *token_indices [[buffer(6)]],
                                          const device float *routing_weights [[buffer(7)]],
                                          const device uint *slot_k_indices [[buffer(8)]],
                                          device float *down_out [[buffer(9)]],
                                          constant QfMoeParams &p [[buffer(10)]],
                                          uint2 tg_pos [[threadgroup_position_in_grid]],
                                          uint sg_id [[simdgroup_index_in_threadgroup]],
                                          uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint ROWS_PER_SG = 2u;
    const uint u = tg_pos.y;
    if (u >= sched_hdr->num_unique_experts) {
        return;
    }
    const uint row0 = tg_pos.x * 8u + sg_id * ROWS_PER_SG;
    if (row0 >= p.hidden_size) {
        return;
    }
    const uint s0 = expert_token_offsets[u];
    const uint s1 = expert_token_offsets[u + 1u];
    const uint count_u = s1 - s0;
    if (count_u == 0u) {
        return;
    }
    const QfExpertWeights ew = expert_table[unique_experts[u]];
    const device uchar *down_qw = weights_base + ew.down_qweight_offset;
    const uint blks_per_step = (QF_DTYPE == QF_DT_GGUF_Q2_0) ? 4u : 8u;

    for (uint t_base = 0u; t_base < count_u; ) {
        const uint rem = min(4u, count_u - t_base);
        const device half *i0 = inter + ulong(s0 + t_base + 0u) * p.intermediate_size;
        const device half *i1 = inter + ulong(s0 + t_base + min(1u, rem - 1u)) * p.intermediate_size;
        const device half *i2 = inter + ulong(s0 + t_base + min(2u, rem - 1u)) * p.intermediate_size;
        const device half *i3 = inter + ulong(s0 + t_base + min(3u, rem - 1u)) * p.intermediate_size;

        for (uint r = 0u; r < ROWS_PER_SG; r++) {
            const uint row = row0 + r;
            if (row >= p.hidden_size) {
                break;
            }
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            const device uchar *dw_row = down_qw + ulong(row) * p.row_bytes;

            for (uint blk_step = 0u; blk_step < p.num_groups; blk_step += blks_per_step) {
                uint k0, k1;
                float4 dw0, dw1;
                qf_gguf_blk_dequant8(dw_row, blk_step, lane, p.intermediate_size, k0, k1, dw0, dw1);
                if (k0 >= p.intermediate_size) {
                    continue;
                }
                { const float4 a = float4(*(const device half4 *)(i0 + k0)); const float4 b = float4(*(const device half4 *)(i0 + k1)); a0 += dot(a, dw0) + dot(b, dw1); }
                if (rem > 1u) { const float4 a = float4(*(const device half4 *)(i1 + k0)); const float4 b = float4(*(const device half4 *)(i1 + k1)); a1 += dot(a, dw0) + dot(b, dw1); }
                if (rem > 2u) { const float4 a = float4(*(const device half4 *)(i2 + k0)); const float4 b = float4(*(const device half4 *)(i2 + k1)); a2 += dot(a, dw0) + dot(b, dw1); }
                if (rem > 3u) { const float4 a = float4(*(const device half4 *)(i3 + k0)); const float4 b = float4(*(const device half4 *)(i3 + k1)); a3 += dot(a, dw0) + dot(b, dw1); }
            }

            const float t0 = simd_sum(a0), t1 = simd_sum(a1), t2 = simd_sum(a2), t3 = simd_sum(a3);
            if (lane == 0u) {
                { const uint s = s0 + t_base + 0u; down_out[(ulong(token_indices[s]) * p.top_k + slot_k_indices[s]) * p.hidden_size + row] = t0 * routing_weights[s]; }
                if (rem > 1u) { const uint s = s0 + t_base + 1u; down_out[(ulong(token_indices[s]) * p.top_k + slot_k_indices[s]) * p.hidden_size + row] = t1 * routing_weights[s]; }
                if (rem > 2u) { const uint s = s0 + t_base + 2u; down_out[(ulong(token_indices[s]) * p.top_k + slot_k_indices[s]) * p.hidden_size + row] = t2 * routing_weights[s]; }
                if (rem > 3u) { const uint s = s0 + t_base + 3u; down_out[(ulong(token_indices[s]) * p.top_k + slot_k_indices[s]) * p.hidden_size + row] = t3 * routing_weights[s]; }
            }
        }
        t_base += rem;
    }
}

// ===========================================================================
// Family 4: GGUF Super-Block K-Quants, IQ4_XS, and Ternary (QK = 256):
// q2_k, q3_k, q4_k, q5_k, q6_k, iq4_xs, tq1_0, tq2_0
// ===========================================================================

static inline uint qf_gguf_k_bytes(uint dtype) {
    switch (dtype) {
        case QF_DT_GGUF_Q2_K: return 84u;
        case QF_DT_GGUF_Q3_K: return 110u;
        case QF_DT_GGUF_Q4_K: return 144u;
        case QF_DT_GGUF_Q5_K: return 176u;
        case QF_DT_GGUF_Q6_K: return 210u;
        case QF_DT_GGUF_IQ4_XS: return 136u;
        case QF_DT_GGUF_TQ1_0: return 54u;
        case QF_DT_GGUF_TQ2_0: return 66u;
        default: return 144u;
    }
}

static inline uchar2 qf_get_scale_min_k4(uint j, const device uchar *q) {
    if (j < 4u) {
        return uchar2(q[j] & 63u, q[j + 4u] & 63u);
    }
    return uchar2((q[j + 4u] & 0xFu) | ((q[j - 4u] >> 6) << 4),
                  (q[j + 4u] >> 4) | ((q[j] >> 6) << 4));
}

static inline int qf_q3k_scale_at(const device uchar *scales, uint is) {
    const uint u0 = qf_read_u32(scales + 0u);
    const uint u1 = qf_read_u32(scales + 4u);
    const uint u2 = qf_read_u32(scales + 8u);
    const uint word_idx = is >> 2;
    uint word;
    if (word_idx == 0u) {
        word = (u0 & 0x0F0F0F0Fu) | ((u2 & 0x03030303u) << 4);
    } else if (word_idx == 1u) {
        word = (u1 & 0x0F0F0F0Fu) | (((u2 >> 2) & 0x03030303u) << 4);
    } else if (word_idx == 2u) {
        word = ((u0 >> 4) & 0x0F0F0F0Fu) | (((u2 >> 4) & 0x03030303u) << 4);
    } else {
        word = ((u1 >> 4) & 0x0F0F0F0Fu) | (((u2 >> 6) & 0x03030303u) << 4);
    }
    return int((word >> ((is & 3u) * 8u)) & 0xFFu) - 32;
}

static inline void qf_gguf_k_dequant8(const device uchar *blk,
                                      uint sb_idx,
                                      uint lane,
                                      thread uint &k0,
                                      thread uint &k1,
                                      thread float4 &w0,
                                      thread float4 &w1) {
    if (QF_DTYPE == QF_DT_GGUF_IQ4_XS) {
        const uint ib32 = lane >> 2;
        const uint l = lane & 3u;
        k0 = sb_idx * 256u + ib32 * 32u + l * 4u;
        k1 = k0 + 16u;
        const float d = qf_read_f16_as_f32(blk);
        const ushort scales_h = qf_read_u16(blk + 2u);
        const int ls = int(((blk[4u + (ib32 >> 1)] >> (4u * (ib32 & 1u))) & 0xFu) |
                           (((scales_h >> (2u * ib32)) & 3u) << 4));
        const float dl = d * float(ls - 32);
        const device uchar *qs = blk + 8u + ib32 * 16u + l * 4u;
        w0 = float4(float(QF_KVALUES_IQ4NL[qs[0] & 0xFu]), float(QF_KVALUES_IQ4NL[qs[1] & 0xFu]),
                    float(QF_KVALUES_IQ4NL[qs[2] & 0xFu]), float(QF_KVALUES_IQ4NL[qs[3] & 0xFu])) * dl;
        w1 = float4(float(QF_KVALUES_IQ4NL[qs[0] >> 4]), float(QF_KVALUES_IQ4NL[qs[1] >> 4]),
                    float(QF_KVALUES_IQ4NL[qs[2] >> 4]), float(QF_KVALUES_IQ4NL[qs[3] >> 4])) * dl;
        return;
    }

    k0 = sb_idx * 256u + lane * 8u;
    k1 = k0 + 4u;

    if (QF_DTYPE == QF_DT_GGUF_Q2_K) {
        const float d = qf_read_f16_as_f32(blk + 80u);
        const float dmin = qf_read_f16_as_f32(blk + 82u);
        const uint is = lane >> 1;
        const uchar sc = blk[is];
        const float dl = d * float(sc & 0xFu);
        const float ml = dmin * float(sc >> 4);
        const uint n_half = lane >> 4;
        const uint shift = ((lane >> 2) & 3u) * 2u;
        const uint sub16 = (lane >> 1) & 1u;
        const uint oct = lane & 1u;
        const device uchar *q = blk + 16u + n_half * 32u + sub16 * 16u + oct * 8u;
        w0 = float4(float((q[0] >> shift) & 3u), float((q[1] >> shift) & 3u),
                    float((q[2] >> shift) & 3u), float((q[3] >> shift) & 3u)) * dl - ml;
        w1 = float4(float((q[4] >> shift) & 3u), float((q[5] >> shift) & 3u),
                    float((q[6] >> shift) & 3u), float((q[7] >> shift) & 3u)) * dl - ml;
    } else if (QF_DTYPE == QF_DT_GGUF_Q3_K) {
        const float d_all = qf_read_f16_as_f32(blk + 108u);
        const uint is = lane >> 1;
        const float dl = d_all * float(qf_q3k_scale_at(blk + 96u, is));
        const uint n_half = lane >> 4;
        const uint shift = ((lane >> 2) & 3u) * 2u;
        const uint sub16 = (lane >> 1) & 1u;
        const uint oct = lane & 1u;
        const uint m_bit = 1u << (lane >> 2);
        const device uchar *hm = blk + sub16 * 16u + oct * 8u;
        const device uchar *q = blk + 32u + n_half * 32u + sub16 * 16u + oct * 8u;
        w0 = float4(float(int((q[0] >> shift) & 3u) - ((hm[0] & m_bit) != 0u ? 0 : 4)),
                    float(int((q[1] >> shift) & 3u) - ((hm[1] & m_bit) != 0u ? 0 : 4)),
                    float(int((q[2] >> shift) & 3u) - ((hm[2] & m_bit) != 0u ? 0 : 4)),
                    float(int((q[3] >> shift) & 3u) - ((hm[3] & m_bit) != 0u ? 0 : 4))) * dl;
        w1 = float4(float(int((q[4] >> shift) & 3u) - ((hm[4] & m_bit) != 0u ? 0 : 4)),
                    float(int((q[5] >> shift) & 3u) - ((hm[5] & m_bit) != 0u ? 0 : 4)),
                    float(int((q[6] >> shift) & 3u) - ((hm[6] & m_bit) != 0u ? 0 : 4)),
                    float(int((q[7] >> shift) & 3u) - ((hm[7] & m_bit) != 0u ? 0 : 4))) * dl;
    } else if (QF_DTYPE == QF_DT_GGUF_Q4_K) {
        const float d = qf_read_f16_as_f32(blk);
        const float dmin = qf_read_f16_as_f32(blk + 2u);
        const uint blk64 = lane >> 3;
        const uint hi = (lane >> 2) & 1u;
        const uint oct = lane & 3u;
        const uchar2 sc_m = qf_get_scale_min_k4(blk64 * 2u + hi, blk + 4u);
        const float dl = d * float(sc_m.x);
        const float ml = dmin * float(sc_m.y);
        const device uchar *q = blk + 16u + blk64 * 32u + oct * 8u;
        const uint sh = hi * 4u;
        w0 = float4(float((q[0] >> sh) & 0xFu), float((q[1] >> sh) & 0xFu),
                    float((q[2] >> sh) & 0xFu), float((q[3] >> sh) & 0xFu)) * dl - ml;
        w1 = float4(float((q[4] >> sh) & 0xFu), float((q[5] >> sh) & 0xFu),
                    float((q[6] >> sh) & 0xFu), float((q[7] >> sh) & 0xFu)) * dl - ml;
    } else if (QF_DTYPE == QF_DT_GGUF_Q5_K) {
        const float d = qf_read_f16_as_f32(blk);
        const float dmin = qf_read_f16_as_f32(blk + 2u);
        const uint blk64 = lane >> 3;
        const uint hi = (lane >> 2) & 1u;
        const uint oct = lane & 3u;
        const uint is = blk64 * 2u + hi;
        const uchar2 sc_m = qf_get_scale_min_k4(is, blk + 4u);
        const float dl = d * float(sc_m.x);
        const float ml = dmin * float(sc_m.y);
        const device uchar *qh = blk + 16u + oct * 8u;
        const device uchar *q = blk + 48u + blk64 * 32u + oct * 8u;
        const uint sh = hi * 4u;
        const uint u_bit = 1u << is;
        w0 = float4(float(((q[0] >> sh) & 0xFu) + (((qh[0] & u_bit) != 0u) ? 16u : 0u)),
                    float(((q[1] >> sh) & 0xFu) + (((qh[1] & u_bit) != 0u) ? 16u : 0u)),
                    float(((q[2] >> sh) & 0xFu) + (((qh[2] & u_bit) != 0u) ? 16u : 0u)),
                    float(((q[3] >> sh) & 0xFu) + (((qh[3] & u_bit) != 0u) ? 16u : 0u))) * dl - ml;
        w1 = float4(float(((q[4] >> sh) & 0xFu) + (((qh[4] & u_bit) != 0u) ? 16u : 0u)),
                    float(((q[5] >> sh) & 0xFu) + (((qh[5] & u_bit) != 0u) ? 16u : 0u)),
                    float(((q[6] >> sh) & 0xFu) + (((qh[6] & u_bit) != 0u) ? 16u : 0u)),
                    float(((q[7] >> sh) & 0xFu) + (((qh[7] & u_bit) != 0u) ? 16u : 0u))) * dl - ml;
    } else if (QF_DTYPE == QF_DT_GGUF_Q6_K) {
        const float d = qf_read_f16_as_f32(blk + 208u);
        const uint n_half = lane >> 4;
        const uint quarter = (lane >> 2) & 3u;
        const uint oct = lane & 3u;
        const uint is = n_half * 8u + quarter * 2u + (oct >> 1);
        const float dl = d * float(int(as_type<char>(blk[192u + is])));
        const device uchar *ql = blk + n_half * 64u + (quarter & 1u) * 32u + oct * 8u;
        const device uchar *qh = blk + 128u + n_half * 32u + oct * 8u;
        const uint ql_sh = (quarter >> 1) * 4u;
        const uint qh_sh = quarter * 2u;
        w0 = float4(float(int(((ql[0] >> ql_sh) & 0xFu) | (((qh[0] >> qh_sh) & 3u) << 4)) - 32),
                    float(int(((ql[1] >> ql_sh) & 0xFu) | (((qh[1] >> qh_sh) & 3u) << 4)) - 32),
                    float(int(((ql[2] >> ql_sh) & 0xFu) | (((qh[2] >> qh_sh) & 3u) << 4)) - 32),
                    float(int(((ql[3] >> ql_sh) & 0xFu) | (((qh[3] >> qh_sh) & 3u) << 4)) - 32)) * dl;
        w1 = float4(float(int(((ql[4] >> ql_sh) & 0xFu) | (((qh[4] >> qh_sh) & 3u) << 4)) - 32),
                    float(int(((ql[5] >> ql_sh) & 0xFu) | (((qh[5] >> qh_sh) & 3u) << 4)) - 32),
                    float(int(((ql[6] >> ql_sh) & 0xFu) | (((qh[6] >> qh_sh) & 3u) << 4)) - 32),
                    float(int(((ql[7] >> ql_sh) & 0xFu) | (((qh[7] >> qh_sh) & 3u) << 4)) - 32)) * dl;
    } else if (QF_DTYPE == QF_DT_GGUF_TQ1_0) {
        const float d = qf_read_f16_as_f32(blk + 52u);
        const uint pow3_lut[5] = {1u, 3u, 9u, 27u, 81u};
        if (lane < 20u) {
            const uint p3 = pow3_lut[lane >> 2];
            const device uchar *q = blk + (lane & 3u) * 8u;
            w0 = float4(float(int((((uint(q[0]) * p3) & 0xFFu) * 3u) >> 8) - 1),
                        float(int((((uint(q[1]) * p3) & 0xFFu) * 3u) >> 8) - 1),
                        float(int((((uint(q[2]) * p3) & 0xFFu) * 3u) >> 8) - 1),
                        float(int((((uint(q[3]) * p3) & 0xFFu) * 3u) >> 8) - 1)) * d;
            w1 = float4(float(int((((uint(q[4]) * p3) & 0xFFu) * 3u) >> 8) - 1),
                        float(int((((uint(q[5]) * p3) & 0xFFu) * 3u) >> 8) - 1),
                        float(int((((uint(q[6]) * p3) & 0xFFu) * 3u) >> 8) - 1),
                        float(int((((uint(q[7]) * p3) & 0xFFu) * 3u) >> 8) - 1)) * d;
        } else if (lane < 30u) {
            const uint l2 = lane - 20u;
            const uint p3 = pow3_lut[l2 >> 1];
            const device uchar *q = blk + 32u + (l2 & 1u) * 8u;
            w0 = float4(float(int((((uint(q[0]) * p3) & 0xFFu) * 3u) >> 8) - 1),
                        float(int((((uint(q[1]) * p3) & 0xFFu) * 3u) >> 8) - 1),
                        float(int((((uint(q[2]) * p3) & 0xFFu) * 3u) >> 8) - 1),
                        float(int((((uint(q[3]) * p3) & 0xFFu) * 3u) >> 8) - 1)) * d;
            w1 = float4(float(int((((uint(q[4]) * p3) & 0xFFu) * 3u) >> 8) - 1),
                        float(int((((uint(q[5]) * p3) & 0xFFu) * 3u) >> 8) - 1),
                        float(int((((uint(q[6]) * p3) & 0xFFu) * 3u) >> 8) - 1),
                        float(int((((uint(q[7]) * p3) & 0xFFu) * 3u) >> 8) - 1)) * d;
        } else {
            const uint l3 = lane - 30u;
            const uint p3_0 = pow3_lut[l3 * 2u];
            const uint p3_1 = pow3_lut[l3 * 2u + 1u];
            const device uchar *qh = blk + 48u;
            w0 = float4(float(int((((uint(qh[0]) * p3_0) & 0xFFu) * 3u) >> 8) - 1),
                        float(int((((uint(qh[1]) * p3_0) & 0xFFu) * 3u) >> 8) - 1),
                        float(int((((uint(qh[2]) * p3_0) & 0xFFu) * 3u) >> 8) - 1),
                        float(int((((uint(qh[3]) * p3_0) & 0xFFu) * 3u) >> 8) - 1)) * d;
            w1 = float4(float(int((((uint(qh[0]) * p3_1) & 0xFFu) * 3u) >> 8) - 1),
                        float(int((((uint(qh[1]) * p3_1) & 0xFFu) * 3u) >> 8) - 1),
                        float(int((((uint(qh[2]) * p3_1) & 0xFFu) * 3u) >> 8) - 1),
                        float(int((((uint(qh[3]) * p3_1) & 0xFFu) * 3u) >> 8) - 1)) * d;
        }
    } else { // QF_DT_GGUF_TQ2_0
        const float d = qf_read_f16_as_f32(blk + 64u);
        const uint half128 = lane >> 4;
        const uint sh = ((lane >> 2) & 3u) * 2u;
        const uint oct = lane & 3u;
        const device uchar *q = blk + half128 * 32u + oct * 8u;
        w0 = float4(float(int((q[0] >> sh) & 3u) - 1), float(int((q[1] >> sh) & 3u) - 1),
                    float(int((q[2] >> sh) & 3u) - 1), float(int((q[3] >> sh) & 3u) - 1)) * d;
        w1 = float4(float(int((q[4] >> sh) & 3u) - 1), float(int((q[5] >> sh) & 3u) - 1),
                    float(int((q[6] >> sh) & 3u) - 1), float(int((q[7] >> sh) & 3u) - 1)) * d;
    }
}

kernel void ami_qf_gguf_k_gemv(const device half *x [[buffer(0)]],
                               const device uchar *qweight [[buffer(1)]],
                               const device uchar *scales [[buffer(2)]],
                               device half *y [[buffer(3)]],
                               const device half *residual [[buffer(4)]],
                               constant QuantFormatParams &p [[buffer(5)]],
                               uint tg_id [[threadgroup_position_in_grid]],
                               uint sg_id [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
    (void)scales;
    constexpr uint ROWS_PER_SG = 4u;
    const uint row0 = tg_id * 16u + sg_id * ROWS_PER_SG;
    if (row0 >= p.n) {
        return;
    }
    const uint blk_bytes = qf_gguf_k_bytes(QF_DTYPE);
    float acc[ROWS_PER_SG] = {0.0f, 0.0f, 0.0f, 0.0f};
    const bool full_rows = (row0 + ROWS_PER_SG <= p.n);

    for (uint sb = 0u; sb < p.num_groups; sb++) {
        uint k0, k1;
        if (QF_DTYPE == QF_DT_GGUF_IQ4_XS) {
            k0 = sb * 256u + (lane >> 2) * 32u + (lane & 3u) * 4u;
            k1 = k0 + 16u;
        } else {
            k0 = sb * 256u + lane * 8u;
            k1 = k0 + 4u;
        }
        const float4 fx0 = float4(*(const device half4 *)(x + k0));
        const float4 fx1 = float4(*(const device half4 *)(x + k1));

        for (uint r = 0u; r < ROWS_PER_SG; r++) {
            const uint row = row0 + r;
            if (full_rows || row < p.n) {
                uint tk0, tk1;
                float4 w0, w1;
                const device uchar *blk = qweight + ulong(row) * p.row_bytes + ulong(sb) * blk_bytes;
                qf_gguf_k_dequant8(blk, sb, lane, tk0, tk1, w0, w1);
                acc[r] += dot(fx0, w0) + dot(fx1, w1);
            }
        }
    }

    #pragma clang loop unroll(full)
    for (uint r = 0u; r < ROWS_PER_SG; r++) {
        const float total = simd_sum(acc[r]);
        if (lane == 0u && (row0 + r) < p.n) {
            const uint row = row0 + r;
            const bool add_res = QF_HAS_RESIDUAL || (p.has_residual != 0u);
            y[row] = half(add_res ? (total + float(residual[row])) : total);
        }
    }
}

kernel void ami_qf_gguf_k_gemv_batch(const device half *x [[buffer(0)]],
                                     const device uchar *qweight [[buffer(1)]],
                                     const device uchar *scales [[buffer(2)]],
                                     device half *y [[buffer(3)]],
                                     const device half *residual [[buffer(4)]],
                                     constant QuantFormatParams &p [[buffer(5)]],
                                     uint2 tg_pos [[threadgroup_position_in_grid]],
                                     uint sg_id [[simdgroup_index_in_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]]) {
    (void)scales;
    const uint row0 = (tg_pos.x * 4u + sg_id) * 2u;
    const uint m_base = tg_pos.y * 8u;
    if (row0 >= p.n || m_base >= p.m) {
        return;
    }
    const uint row1 = min(row0 + 1u, p.n - 1u);
    const bool has_row1 = (row0 + 1u) < p.n;
    const uint tile_m = min(8u, p.m - m_base);
    const uint blk_bytes = qf_gguf_k_bytes(QF_DTYPE);

    const device uchar *w_row0 = qweight + ulong(row0) * p.row_bytes;
    const device uchar *w_row1 = qweight + ulong(row1) * p.row_bytes;
    const device half *x_base = x + ulong(m_base) * p.k;
    const ulong x_stride = ulong(p.k);

    float a0_0 = 0.0f, a0_1 = 0.0f, a0_2 = 0.0f, a0_3 = 0.0f;
    float a0_4 = 0.0f, a0_5 = 0.0f, a0_6 = 0.0f, a0_7 = 0.0f;
    float a1_0 = 0.0f, a1_1 = 0.0f, a1_2 = 0.0f, a1_3 = 0.0f;
    float a1_4 = 0.0f, a1_5 = 0.0f, a1_6 = 0.0f, a1_7 = 0.0f;

    for (uint sb = 0u; sb < p.num_groups; sb++) {
        uint k0, k1;
        float4 w0_a, w0_b, w1_a, w1_b;
        qf_gguf_k_dequant8(w_row0 + ulong(sb) * blk_bytes, sb, lane, k0, k1, w0_a, w0_b);
        qf_gguf_k_dequant8(w_row1 + ulong(sb) * blk_bytes, sb, lane, k0, k1, w1_a, w1_b);
        { const float4 fx0 = float4(*(const device half4 *)(x_base + 0ul * x_stride + k0)); const float4 fx1 = float4(*(const device half4 *)(x_base + 0ul * x_stride + k1)); a0_0 += dot(fx0, w0_a) + dot(fx1, w0_b); a1_0 += dot(fx0, w1_a) + dot(fx1, w1_b); }
        if (tile_m > 1u) { const float4 fx0 = float4(*(const device half4 *)(x_base + 1ul * x_stride + k0)); const float4 fx1 = float4(*(const device half4 *)(x_base + 1ul * x_stride + k1)); a0_1 += dot(fx0, w0_a) + dot(fx1, w0_b); a1_1 += dot(fx0, w1_a) + dot(fx1, w1_b); }
        if (tile_m > 2u) { const float4 fx0 = float4(*(const device half4 *)(x_base + 2ul * x_stride + k0)); const float4 fx1 = float4(*(const device half4 *)(x_base + 2ul * x_stride + k1)); a0_2 += dot(fx0, w0_a) + dot(fx1, w0_b); a1_2 += dot(fx0, w1_a) + dot(fx1, w1_b); }
        if (tile_m > 3u) { const float4 fx0 = float4(*(const device half4 *)(x_base + 3ul * x_stride + k0)); const float4 fx1 = float4(*(const device half4 *)(x_base + 3ul * x_stride + k1)); a0_3 += dot(fx0, w0_a) + dot(fx1, w0_b); a1_3 += dot(fx0, w1_a) + dot(fx1, w1_b); }
        if (tile_m > 4u) {
            const float4 fx0 = float4(*(const device half4 *)(x_base + 4ul * x_stride + k0)); const float4 fx1 = float4(*(const device half4 *)(x_base + 4ul * x_stride + k1)); a0_4 += dot(fx0, w0_a) + dot(fx1, w0_b); a1_4 += dot(fx0, w1_a) + dot(fx1, w1_b);
            if (tile_m > 5u) { const float4 fx0 = float4(*(const device half4 *)(x_base + 5ul * x_stride + k0)); const float4 fx1 = float4(*(const device half4 *)(x_base + 5ul * x_stride + k1)); a0_5 += dot(fx0, w0_a) + dot(fx1, w0_b); a1_5 += dot(fx0, w1_a) + dot(fx1, w1_b); }
            if (tile_m > 6u) { const float4 fx0 = float4(*(const device half4 *)(x_base + 6ul * x_stride + k0)); const float4 fx1 = float4(*(const device half4 *)(x_base + 6ul * x_stride + k1)); a0_6 += dot(fx0, w0_a) + dot(fx1, w0_b); a1_6 += dot(fx0, w1_a) + dot(fx1, w1_b); }
            if (tile_m > 7u) { const float4 fx0 = float4(*(const device half4 *)(x_base + 7ul * x_stride + k0)); const float4 fx1 = float4(*(const device half4 *)(x_base + 7ul * x_stride + k1)); a0_7 += dot(fx0, w0_a) + dot(fx1, w0_b); a1_7 += dot(fx0, w1_a) + dot(fx1, w1_b); }
        }
    }

    const bool add_res = QF_HAS_RESIDUAL || (p.has_residual != 0u);
    #define QF_STORE_PAIR_K(M_OFF, A0, A1) \
        { \
            const float t0 = simd_sum(A0); \
            const float t1 = simd_sum(A1); \
            if (lane == 0u) { \
                const ulong b = ulong(m_base + M_OFF) * p.n; \
                y[b + row0] = half(add_res ? (t0 + float(residual[b + row0])) : t0); \
                if (has_row1) { \
                    y[b + row1] = half(add_res ? (t1 + float(residual[b + row1])) : t1); \
                } \
            } \
        }
    QF_STORE_PAIR_K(0u, a0_0, a1_0);
    if (tile_m > 1u) QF_STORE_PAIR_K(1u, a0_1, a1_1);
    if (tile_m > 2u) QF_STORE_PAIR_K(2u, a0_2, a1_2);
    if (tile_m > 3u) QF_STORE_PAIR_K(3u, a0_3, a1_3);
    if (tile_m > 4u) {
        QF_STORE_PAIR_K(4u, a0_4, a1_4);
        if (tile_m > 5u) QF_STORE_PAIR_K(5u, a0_5, a1_5);
        if (tile_m > 6u) QF_STORE_PAIR_K(6u, a0_6, a1_6);
        if (tile_m > 7u) QF_STORE_PAIR_K(7u, a0_7, a1_7);
    }
    #undef QF_STORE_PAIR_K
}

kernel void ami_qf_gguf_k_dequant_tile_f16(const device uchar *qweight [[buffer(0)]],
                                           const device uchar *scales [[buffer(1)]],
                                           device half *dst_f16 [[buffer(2)]],
                                           constant QuantFormatParams &p [[buffer(3)]],
                                           uint2 tg_pos [[threadgroup_position_in_grid]],
                                           uint sg_id [[simdgroup_index_in_threadgroup]],
                                           uint lane [[thread_index_in_simdgroup]]) {
    (void)scales;
    const uint local_r = tg_pos.y * 4u + sg_id;
    if (local_r >= p.tile_n) {
        return;
    }
    const uint sb = (p.tile_k_start >> 8) + tg_pos.x;
    if (sb * 256u >= p.tile_k_start + p.tile_k) {
        return;
    }
    const uint row = p.tile_n_start + local_r;
    const uint blk_bytes = qf_gguf_k_bytes(QF_DTYPE);
    uint k0, k1;
    float4 w0, w1;
    qf_gguf_k_dequant8(qweight + ulong(row) * p.row_bytes + ulong(sb) * blk_bytes, sb, lane, k0, k1, w0, w1);
    device half *out = dst_f16 + ulong(local_r) * p.dst_stride;
    *(device half4 *)(out + (k0 - p.tile_k_start)) = half4(w0);
    *(device half4 *)(out + (k1 - p.tile_k_start)) = half4(w1);
}

kernel void ami_qf_gguf_k_moe_gate_up_swiglu(const device half *x [[buffer(0)]],
                                             const device uchar *weights_base [[buffer(1)]],
                                             const device QfExpertWeights *expert_table [[buffer(2)]],
                                             const device QfMoeScheduleHeader *sched_hdr [[buffer(3)]],
                                             const device uint *unique_experts [[buffer(4)]],
                                             const device uint *expert_token_offsets [[buffer(5)]],
                                             const device uint *token_indices [[buffer(6)]],
                                             device half *inter [[buffer(7)]],
                                             constant QfMoeParams &p [[buffer(8)]],
                                             uint2 tg_pos [[threadgroup_position_in_grid]],
                                             uint sg_id [[simdgroup_index_in_threadgroup]],
                                             uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint ROWS_PER_SG = 2u;
    const uint u = tg_pos.y;
    if (u >= sched_hdr->num_unique_experts) {
        return;
    }
    const uint row0 = tg_pos.x * 8u + sg_id * ROWS_PER_SG;
    if (row0 >= p.intermediate_size) {
        return;
    }
    const uint s0 = expert_token_offsets[u];
    const uint s1 = expert_token_offsets[u + 1u];
    const uint count_u = s1 - s0;
    if (count_u == 0u) {
        return;
    }
    const QfExpertWeights ew = expert_table[unique_experts[u]];
    const device uchar *gate_qw = weights_base + ew.gate_qweight_offset;
    const device uchar *up_qw = weights_base + ew.up_qweight_offset;
    const uint blk_bytes = qf_gguf_k_bytes(QF_DTYPE);

    for (uint t_base = 0u; t_base < count_u; ) {
        const uint rem = min(4u, count_u - t_base);
        const device half *x0 = x + ulong(token_indices[s0 + t_base + 0u]) * p.hidden_size;
        const device half *x1 = x + ulong(token_indices[s0 + t_base + min(1u, rem - 1u)]) * p.hidden_size;
        const device half *x2 = x + ulong(token_indices[s0 + t_base + min(2u, rem - 1u)]) * p.hidden_size;
        const device half *x3 = x + ulong(token_indices[s0 + t_base + min(3u, rem - 1u)]) * p.hidden_size;

        for (uint r = 0u; r < ROWS_PER_SG; r++) {
            const uint row = row0 + r;
            if (row >= p.intermediate_size) {
                break;
            }
            float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
            float u0 = 0.0f, u1 = 0.0f, u2 = 0.0f, u3 = 0.0f;
            const device uchar *gw_row = gate_qw + ulong(row) * p.row_bytes;
            const device uchar *uw_row = up_qw + ulong(row) * p.row_bytes;

            for (uint sb = 0u; sb < p.num_groups; sb++) {
                uint k0, k1;
                float4 gw0, gw1, uw0, uw1;
                qf_gguf_k_dequant8(gw_row + ulong(sb) * blk_bytes, sb, lane, k0, k1, gw0, gw1);
                qf_gguf_k_dequant8(uw_row + ulong(sb) * blk_bytes, sb, lane, k0, k1, uw0, uw1);
                { const float4 a = float4(*(const device half4 *)(x0 + k0)); const float4 b = float4(*(const device half4 *)(x0 + k1)); g0 += dot(a, gw0) + dot(b, gw1); u0 += dot(a, uw0) + dot(b, uw1); }
                if (rem > 1u) { const float4 a = float4(*(const device half4 *)(x1 + k0)); const float4 b = float4(*(const device half4 *)(x1 + k1)); g1 += dot(a, gw0) + dot(b, gw1); u1 += dot(a, uw0) + dot(b, uw1); }
                if (rem > 2u) { const float4 a = float4(*(const device half4 *)(x2 + k0)); const float4 b = float4(*(const device half4 *)(x2 + k1)); g2 += dot(a, gw0) + dot(b, gw1); u2 += dot(a, uw0) + dot(b, uw1); }
                if (rem > 3u) { const float4 a = float4(*(const device half4 *)(x3 + k0)); const float4 b = float4(*(const device half4 *)(x3 + k1)); g3 += dot(a, gw0) + dot(b, gw1); u3 += dot(a, uw0) + dot(b, uw1); }
            }

            const float sg0 = simd_sum(g0), su0 = simd_sum(u0);
            const float sg1 = simd_sum(g1), su1 = simd_sum(u1);
            const float sg2 = simd_sum(g2), su2 = simd_sum(u2);
            const float sg3 = simd_sum(g3), su3 = simd_sum(u3);
            if (lane == 0u) {
                inter[ulong(s0 + t_base + 0u) * p.intermediate_size + row] = half(qf_apply_swiglu(sg0, su0, p.swiglu_limit));
                if (rem > 1u) inter[ulong(s0 + t_base + 1u) * p.intermediate_size + row] = half(qf_apply_swiglu(sg1, su1, p.swiglu_limit));
                if (rem > 2u) inter[ulong(s0 + t_base + 2u) * p.intermediate_size + row] = half(qf_apply_swiglu(sg2, su2, p.swiglu_limit));
                if (rem > 3u) inter[ulong(s0 + t_base + 3u) * p.intermediate_size + row] = half(qf_apply_swiglu(sg3, su3, p.swiglu_limit));
            }
        }
        t_base += rem;
    }
}

kernel void ami_qf_gguf_k_moe_down_proj(const device half *inter [[buffer(0)]],
                                        const device uchar *weights_base [[buffer(1)]],
                                        const device QfExpertWeights *expert_table [[buffer(2)]],
                                        const device QfMoeScheduleHeader *sched_hdr [[buffer(3)]],
                                        const device uint *unique_experts [[buffer(4)]],
                                        const device uint *expert_token_offsets [[buffer(5)]],
                                        const device uint *token_indices [[buffer(6)]],
                                        const device float *routing_weights [[buffer(7)]],
                                        const device uint *slot_k_indices [[buffer(8)]],
                                        device float *down_out [[buffer(9)]],
                                        constant QfMoeParams &p [[buffer(10)]],
                                        uint2 tg_pos [[threadgroup_position_in_grid]],
                                        uint sg_id [[simdgroup_index_in_threadgroup]],
                                        uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint ROWS_PER_SG = 2u;
    const uint u = tg_pos.y;
    if (u >= sched_hdr->num_unique_experts) {
        return;
    }
    const uint row0 = tg_pos.x * 8u + sg_id * ROWS_PER_SG;
    if (row0 >= p.hidden_size) {
        return;
    }
    const uint s0 = expert_token_offsets[u];
    const uint s1 = expert_token_offsets[u + 1u];
    const uint count_u = s1 - s0;
    if (count_u == 0u) {
        return;
    }
    const QfExpertWeights ew = expert_table[unique_experts[u]];
    const device uchar *down_qw = weights_base + ew.down_qweight_offset;
    const uint blk_bytes = qf_gguf_k_bytes(QF_DTYPE);

    for (uint t_base = 0u; t_base < count_u; ) {
        const uint rem = min(4u, count_u - t_base);
        const device half *i0 = inter + ulong(s0 + t_base + 0u) * p.intermediate_size;
        const device half *i1 = inter + ulong(s0 + t_base + min(1u, rem - 1u)) * p.intermediate_size;
        const device half *i2 = inter + ulong(s0 + t_base + min(2u, rem - 1u)) * p.intermediate_size;
        const device half *i3 = inter + ulong(s0 + t_base + min(3u, rem - 1u)) * p.intermediate_size;

        for (uint r = 0u; r < ROWS_PER_SG; r++) {
            const uint row = row0 + r;
            if (row >= p.hidden_size) {
                break;
            }
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            const device uchar *dw_row = down_qw + ulong(row) * p.row_bytes;

            for (uint sb = 0u; sb < p.num_groups; sb++) {
                uint k0, k1;
                float4 dw0, dw1;
                qf_gguf_k_dequant8(dw_row + ulong(sb) * blk_bytes, sb, lane, k0, k1, dw0, dw1);
                { const float4 a = float4(*(const device half4 *)(i0 + k0)); const float4 b = float4(*(const device half4 *)(i0 + k1)); a0 += dot(a, dw0) + dot(b, dw1); }
                if (rem > 1u) { const float4 a = float4(*(const device half4 *)(i1 + k0)); const float4 b = float4(*(const device half4 *)(i1 + k1)); a1 += dot(a, dw0) + dot(b, dw1); }
                if (rem > 2u) { const float4 a = float4(*(const device half4 *)(i2 + k0)); const float4 b = float4(*(const device half4 *)(i2 + k1)); a2 += dot(a, dw0) + dot(b, dw1); }
                if (rem > 3u) { const float4 a = float4(*(const device half4 *)(i3 + k0)); const float4 b = float4(*(const device half4 *)(i3 + k1)); a3 += dot(a, dw0) + dot(b, dw1); }
            }

            const float t0 = simd_sum(a0), t1 = simd_sum(a1), t2 = simd_sum(a2), t3 = simd_sum(a3);
            if (lane == 0u) {
                { const uint s = s0 + t_base + 0u; down_out[(ulong(token_indices[s]) * p.top_k + slot_k_indices[s]) * p.hidden_size + row] = t0 * routing_weights[s]; }
                if (rem > 1u) { const uint s = s0 + t_base + 1u; down_out[(ulong(token_indices[s]) * p.top_k + slot_k_indices[s]) * p.hidden_size + row] = t1 * routing_weights[s]; }
                if (rem > 2u) { const uint s = s0 + t_base + 2u; down_out[(ulong(token_indices[s]) * p.top_k + slot_k_indices[s]) * p.hidden_size + row] = t2 * routing_weights[s]; }
                if (rem > 3u) { const uint s = s0 + t_base + 3u; down_out[(ulong(token_indices[s]) * p.top_k + slot_k_indices[s]) * p.hidden_size + row] = t3 * routing_weights[s]; }
            }
        }
        t_base += rem;
    }
}

// ===========================================================================
// Family 5: GGUF IQ Codebook-SRAM Formats (QK = 256):
// iq1_s, iq1_m, iq2_xxs, iq2_xs, iq2_s, iq3_xxs, iq3_s
//
// `codebook` buffer layout (packed per IQ dtype via `DType.codebookOffset`):
//   - iq1_s / iq1_m: 2048 x u32 (8,192 B)
//   - iq2_xxs: 256 x u64 (512 x u32 = 2,048 B) + 128 x u8 signs (32 x u32 = 128 B)
//   - iq2_xs:  512 x u64 (1024 x u32 = 4,096 B) + 128 x u8 signs (32 x u32 = 128 B)
//   - iq2_s:   1024 x u64 (2048 x u32 = 8,192 B)
//   - iq3_xxs: 256 x u32 (1,024 B) + 128 x u8 signs (32 x u32 = 128 B)
//   - iq3_s:   512 x u32 (2,048 B)
// Staged cooperatively into 8,320 B threadgroup SRAM (<= 16 KiB).
// ===========================================================================

struct IqCodebookSmem {
    alignas(16) uint32_t grid32[2048]; // 8,192 B (direct 32-bit reads, 16-byte aligned for uint4 staging)
    uint8_t signs[128];                // 128 B (ksigns_iq2xs)
};

static_assert(sizeof(IqCodebookSmem) == 8320u, "IqCodebookSmem must fit within 16 KiB");

static inline uint qf_iq_blk_bytes(uint dtype) {
    switch (dtype) {
        case QF_DT_GGUF_IQ1_S: return 50u;
        case QF_DT_GGUF_IQ1_M: return 56u;
        case QF_DT_GGUF_IQ2_XXS: return 66u;
        case QF_DT_GGUF_IQ2_XS: return 74u;
        case QF_DT_GGUF_IQ2_S: return 82u;
        case QF_DT_GGUF_IQ3_XXS: return 98u;
        case QF_DT_GGUF_IQ3_S: return 110u;
        default: return 66u;
    }
}

static inline uint qf_iq_grid_u16_count(uint dtype) {
    switch (dtype) {
        case QF_DT_GGUF_IQ1_S:
        case QF_DT_GGUF_IQ1_M:
        case QF_DT_GGUF_IQ2_S: return 4096u;
        case QF_DT_GGUF_IQ2_XS: return 2048u;
        case QF_DT_GGUF_IQ2_XXS:
        case QF_DT_GGUF_IQ3_S: return 1024u;
        case QF_DT_GGUF_IQ3_XXS: return 512u;
        default: return 1024u;
    }
}

static inline bool qf_iq_needs_signs(uint dtype) {
    return (dtype == QF_DT_GGUF_IQ2_XXS) || (dtype == QF_DT_GGUF_IQ2_XS) || (dtype == QF_DT_GGUF_IQ3_XXS);
}

static inline void qf_iq_stage_codebook(const device uchar *codebook,
                                        threadgroup IqCodebookSmem &smem,
                                        uint tid) {
    const uint n16 = qf_iq_grid_u16_count(QF_DTYPE);
    const uint n_vec4 = n16 >> 3;
    const device uint4 *src4 = (const device uint4 *)codebook;
    threadgroup uint4 *dst4 = (threadgroup uint4 *)smem.grid32;
    for (uint i = tid; i < n_vec4; i += 128u) {
        dst4[i] = src4[i];
    }
    if (qf_iq_needs_signs(QF_DTYPE)) {
        const device uchar *signs_src = codebook + ulong(n16) * 2ul;
        if (tid < 128u) {
            smem.signs[tid] = signs_src[tid];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

static inline uint qf_smem_read_u32(threadgroup const IqCodebookSmem &smem, uint u32_idx) {
    return smem.grid32[u32_idx];
}

static inline void qf_iq_dequant8(const device uchar *blk,
                                  threadgroup const IqCodebookSmem &smem,
                                  uint lane,
                                  thread float4 &w0,
                                  thread float4 &w1) {
    const uint ib32 = lane >> 2;
    const uint l = lane & 3u;

    if (QF_DTYPE == QF_DT_GGUF_IQ1_S) {
        const float d = qf_read_f16_as_f32(blk);
        const ushort qh = qf_read_u16(blk + 34u + ib32 * 2u);
        const float dl = d * float(2u * ((qh >> 12) & 7u) + 1u);
        const float delta = ((qh & 0x8000u) != 0u) ? -0.125f : 0.125f;
        const float shift = -1.0f + delta;
        const uint idx = uint(blk[2u + lane]) | (((uint(qh) >> (3u * l)) & 7u) << 8);
        const uint g = qf_smem_read_u32(smem, idx);
        w0 = float4(float((g >> 0) & 0xFu) + shift, float((g >> 8) & 0xFu) + shift,
                    float((g >> 16) & 0xFu) + shift, float((g >> 24) & 0xFu) + shift) * dl;
        w1 = float4(float((g >> 4) & 0xFu) + shift, float((g >> 12) & 0xFu) + shift,
                    float((g >> 20) & 0xFu) + shift, float((g >> 28) & 0xFu) + shift) * dl;
    } else if (QF_DTYPE == QF_DT_GGUF_IQ1_M) {
        const ushort sc0 = qf_read_u16(blk + 48u);
        const ushort sc1 = qf_read_u16(blk + 50u);
        const ushort sc2 = qf_read_u16(blk + 52u);
        const ushort sc3 = qf_read_u16(blk + 54u);
        const ushort d_u16 = (sc0 >> 12) | ((sc1 >> 8) & 0x00F0u) | ((sc2 >> 4) & 0x0F00u) | (sc3 & 0xF000u);
        const float d = float(as_type<half>(d_u16));
        const ushort sc_ib = qf_read_u16(blk + 48u + (ib32 >> 1) * 2u);
        const float dl = d * float(2u * ((sc_ib >> (6u * (ib32 & 1u) + 3u * (l >> 1))) & 7u) + 1u);
        const uchar qh_b = blk[32u + ib32 * 2u + (l >> 1)];
        const float delta = ((qh_b & (((l & 1u) == 0u) ? 0x08u : 0x80u)) != 0u) ? -0.125f : 0.125f;
        const float shift = -1.0f + delta;
        const uint idx = uint(blk[lane]) | ((uint(qh_b) << (((l & 1u) == 0u) ? 8u : 4u)) & 0x700u);
        const uint g = qf_smem_read_u32(smem, idx);
        w0 = float4(float((g >> 0) & 0xFu) + shift, float((g >> 8) & 0xFu) + shift,
                    float((g >> 16) & 0xFu) + shift, float((g >> 24) & 0xFu) + shift) * dl;
        w1 = float4(float((g >> 4) & 0xFu) + shift, float((g >> 12) & 0xFu) + shift,
                    float((g >> 20) & 0xFu) + shift, float((g >> 28) & 0xFu) + shift) * dl;
    } else if (QF_DTYPE == QF_DT_GGUF_IQ2_XXS) {
        const float d = qf_read_f16_as_f32(blk);
        const uint aux0 = qf_read_u32(blk + 2u + ib32 * 8u);
        const uint aux1 = qf_read_u32(blk + 2u + ib32 * 8u + 4u);
        const float db = d * (0.5f + float(aux1 >> 28)) * 0.25f;
        const uint grid_idx = (aux0 >> (8u * l)) & 0xFFu;
        const uchar sg = smem.signs[(aux1 >> (7u * l)) & 127u];
        const uint g0 = qf_smem_read_u32(smem, grid_idx * 2u + 0u);
        const uint g1 = qf_smem_read_u32(smem, grid_idx * 2u + 1u);
        w0 = float4(float((g0 >> 0) & 0xFFu) * (((sg & 1u) != 0u) ? -db : db),
                    float((g0 >> 8) & 0xFFu) * (((sg & 2u) != 0u) ? -db : db),
                    float((g0 >> 16) & 0xFFu) * (((sg & 4u) != 0u) ? -db : db),
                    float((g0 >> 24) & 0xFFu) * (((sg & 8u) != 0u) ? -db : db));
        w1 = float4(float((g1 >> 0) & 0xFFu) * (((sg & 16u) != 0u) ? -db : db),
                    float((g1 >> 8) & 0xFFu) * (((sg & 32u) != 0u) ? -db : db),
                    float((g1 >> 16) & 0xFFu) * (((sg & 64u) != 0u) ? -db : db),
                    float((g1 >> 24) & 0xFFu) * (((sg & 128u) != 0u) ? -db : db));
    } else if (QF_DTYPE == QF_DT_GGUF_IQ2_XS) {
        const float d = qf_read_f16_as_f32(blk);
        const ushort q2 = qf_read_u16(blk + 2u + lane * 2u);
        const uchar sc_b = blk[66u + ib32];
        const float db = d * (0.5f + float((sc_b >> (4u * (l >> 1))) & 0xFu)) * 0.25f;
        const uint grid_idx = q2 & 511u;
        const uchar sg = smem.signs[q2 >> 9];
        const uint g0 = qf_smem_read_u32(smem, grid_idx * 2u + 0u);
        const uint g1 = qf_smem_read_u32(smem, grid_idx * 2u + 1u);
        w0 = float4(float((g0 >> 0) & 0xFFu) * (((sg & 1u) != 0u) ? -db : db),
                    float((g0 >> 8) & 0xFFu) * (((sg & 2u) != 0u) ? -db : db),
                    float((g0 >> 16) & 0xFFu) * (((sg & 4u) != 0u) ? -db : db),
                    float((g0 >> 24) & 0xFFu) * (((sg & 8u) != 0u) ? -db : db));
        w1 = float4(float((g1 >> 0) & 0xFFu) * (((sg & 16u) != 0u) ? -db : db),
                    float((g1 >> 8) & 0xFFu) * (((sg & 32u) != 0u) ? -db : db),
                    float((g1 >> 16) & 0xFFu) * (((sg & 64u) != 0u) ? -db : db),
                    float((g1 >> 24) & 0xFFu) * (((sg & 128u) != 0u) ? -db : db));
    } else if (QF_DTYPE == QF_DT_GGUF_IQ2_S) {
        const float d = qf_read_f16_as_f32(blk);
        const uchar sc_b = blk[74u + ib32];
        const float dl = d * (0.5f + float((sc_b >> (4u * (l >> 1))) & 0xFu)) * 0.25f;
        const uchar q_b = blk[2u + lane];
        const uchar sg = blk[34u + lane];
        const uchar qh_b = blk[66u + ib32];
        const uint grid_idx = uint(q_b) | ((uint(qh_b) << (8u - 2u * l)) & 0x300u);
        const uint g0 = qf_smem_read_u32(smem, grid_idx * 2u + 0u);
        const uint g1 = qf_smem_read_u32(smem, grid_idx * 2u + 1u);
        w0 = float4(float((g0 >> 0) & 0xFFu) * (((sg & 1u) != 0u) ? -dl : dl),
                    float((g0 >> 8) & 0xFFu) * (((sg & 2u) != 0u) ? -dl : dl),
                    float((g0 >> 16) & 0xFFu) * (((sg & 4u) != 0u) ? -dl : dl),
                    float((g0 >> 24) & 0xFFu) * (((sg & 8u) != 0u) ? -dl : dl));
        w1 = float4(float((g1 >> 0) & 0xFFu) * (((sg & 16u) != 0u) ? -dl : dl),
                    float((g1 >> 8) & 0xFFu) * (((sg & 32u) != 0u) ? -dl : dl),
                    float((g1 >> 16) & 0xFFu) * (((sg & 64u) != 0u) ? -dl : dl),
                    float((g1 >> 24) & 0xFFu) * (((sg & 128u) != 0u) ? -dl : dl));
    } else if (QF_DTYPE == QF_DT_GGUF_IQ3_XXS) {
        const float d = qf_read_f16_as_f32(blk);
        const uint aux = qf_read_u32(blk + 66u + ib32 * 4u);
        const float db = d * (0.5f + float(aux >> 28)) * 0.5f;
        const uchar sg = smem.signs[(aux >> (7u * l)) & 127u];
        const uint g0 = qf_smem_read_u32(smem, blk[2u + lane * 2u + 0u]);
        const uint g1 = qf_smem_read_u32(smem, blk[2u + lane * 2u + 1u]);
        w0 = float4(float((g0 >> 0) & 0xFFu) * (((sg & 1u) != 0u) ? -db : db),
                    float((g0 >> 8) & 0xFFu) * (((sg & 2u) != 0u) ? -db : db),
                    float((g0 >> 16) & 0xFFu) * (((sg & 4u) != 0u) ? -db : db),
                    float((g0 >> 24) & 0xFFu) * (((sg & 8u) != 0u) ? -db : db));
        w1 = float4(float((g1 >> 0) & 0xFFu) * (((sg & 16u) != 0u) ? -db : db),
                    float((g1 >> 8) & 0xFFu) * (((sg & 32u) != 0u) ? -db : db),
                    float((g1 >> 16) & 0xFFu) * (((sg & 64u) != 0u) ? -db : db),
                    float((g1 >> 24) & 0xFFu) * (((sg & 128u) != 0u) ? -db : db));
    } else { // QF_DT_GGUF_IQ3_S
        const float d = qf_read_f16_as_f32(blk);
        const uchar sc_b = blk[106u + (ib32 >> 1)];
        const float db = d * float(1u + 2u * ((sc_b >> (4u * (ib32 & 1u))) & 0xFu));
        const uchar qh_b = blk[66u + ib32];
        const uint idx1 = uint(blk[2u + lane * 2u + 0u]) | ((uint(qh_b) << (8u - 2u * l)) & 256u);
        const uint idx2 = uint(blk[2u + lane * 2u + 1u]) | ((uint(qh_b) << (7u - 2u * l)) & 256u);
        const uchar sg = blk[74u + lane];
        const uint g0 = qf_smem_read_u32(smem, idx1);
        const uint g1 = qf_smem_read_u32(smem, idx2);
        w0 = float4(float((g0 >> 0) & 0xFFu) * (((sg & 1u) != 0u) ? -db : db),
                    float((g0 >> 8) & 0xFFu) * (((sg & 2u) != 0u) ? -db : db),
                    float((g0 >> 16) & 0xFFu) * (((sg & 4u) != 0u) ? -db : db),
                    float((g0 >> 24) & 0xFFu) * (((sg & 8u) != 0u) ? -db : db));
        w1 = float4(float((g1 >> 0) & 0xFFu) * (((sg & 16u) != 0u) ? -db : db),
                    float((g1 >> 8) & 0xFFu) * (((sg & 32u) != 0u) ? -db : db),
                    float((g1 >> 16) & 0xFFu) * (((sg & 64u) != 0u) ? -db : db),
                    float((g1 >> 24) & 0xFFu) * (((sg & 128u) != 0u) ? -db : db));
    }
}

kernel void ami_qf_iq_gemv(const device half *x [[buffer(0)]],
                           const device uchar *qweight [[buffer(1)]],
                           const device uchar *scales [[buffer(2)]],
                           device half *y [[buffer(3)]],
                           const device half *residual [[buffer(4)]],
                           const device uchar *codebook [[buffer(5)]],
                           constant QuantFormatParams &p [[buffer(6)]],
                           uint tg_id [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint sg_id [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    (void)scales;
    threadgroup IqCodebookSmem smem;
    qf_iq_stage_codebook(codebook, smem, tid);

    constexpr uint ROWS_PER_SG = 4u;
    const uint row0 = tg_id * 16u + sg_id * ROWS_PER_SG;
    if (row0 >= p.n) {
        return;
    }
    const uint blk_bytes = qf_iq_blk_bytes(QF_DTYPE);
    float acc[ROWS_PER_SG] = {0.0f, 0.0f, 0.0f, 0.0f};
    const bool full_rows = (row0 + ROWS_PER_SG <= p.n);

    for (uint sb = 0u; sb < p.num_groups; sb++) {
        const uint k_off = sb * 256u + lane * 8u;
        const float4 fx0 = float4(*(const device half4 *)(x + k_off));
        const float4 fx1 = float4(*(const device half4 *)(x + k_off + 4u));
        for (uint r = 0u; r < ROWS_PER_SG; r++) {
            const uint row = row0 + r;
            if (full_rows || row < p.n) {
                float4 w0, w1;
                qf_iq_dequant8(qweight + ulong(row) * p.row_bytes + ulong(sb) * blk_bytes, smem, lane, w0, w1);
                acc[r] += dot(fx0, w0) + dot(fx1, w1);
            }
        }
    }

    #pragma clang loop unroll(full)
    for (uint r = 0u; r < ROWS_PER_SG; r++) {
        const float total = simd_sum(acc[r]);
        if (lane == 0u && (row0 + r) < p.n) {
            const uint row = row0 + r;
            const bool add_res = QF_HAS_RESIDUAL || (p.has_residual != 0u);
            y[row] = half(add_res ? (total + float(residual[row])) : total);
        }
    }
}

kernel void ami_qf_iq_gemv_batch(const device half *x [[buffer(0)]],
                                 const device uchar *qweight [[buffer(1)]],
                                 const device uchar *scales [[buffer(2)]],
                                 device half *y [[buffer(3)]],
                                 const device half *residual [[buffer(4)]],
                                 const device uchar *codebook [[buffer(5)]],
                                 constant QuantFormatParams &p [[buffer(6)]],
                                 uint2 tg_pos [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]],
                                 uint sg_id [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
    (void)scales;
    threadgroup IqCodebookSmem smem;
    qf_iq_stage_codebook(codebook, smem, tid);

    const uint row0 = (tg_pos.x * 4u + sg_id) * 2u;
    const uint m_base = tg_pos.y * 8u;
    if (row0 >= p.n || m_base >= p.m) {
        return;
    }
    const uint row1 = min(row0 + 1u, p.n - 1u);
    const bool has_row1 = (row0 + 1u) < p.n;
    const uint tile_m = min(8u, p.m - m_base);
    const uint blk_bytes = qf_iq_blk_bytes(QF_DTYPE);

    const device uchar *w_row0 = qweight + ulong(row0) * p.row_bytes;
    const device uchar *w_row1 = qweight + ulong(row1) * p.row_bytes;
    const device half *x_base = x + ulong(m_base) * p.k;
    const ulong x_stride = ulong(p.k);

    float a0_0 = 0.0f, a0_1 = 0.0f, a0_2 = 0.0f, a0_3 = 0.0f;
    float a0_4 = 0.0f, a0_5 = 0.0f, a0_6 = 0.0f, a0_7 = 0.0f;
    float a1_0 = 0.0f, a1_1 = 0.0f, a1_2 = 0.0f, a1_3 = 0.0f;
    float a1_4 = 0.0f, a1_5 = 0.0f, a1_6 = 0.0f, a1_7 = 0.0f;

    for (uint sb = 0u; sb < p.num_groups; sb++) {
        const uint k_off = sb * 256u + lane * 8u;
        float4 w0_a, w0_b, w1_a, w1_b;
        qf_iq_dequant8(w_row0 + ulong(sb) * blk_bytes, smem, lane, w0_a, w0_b);
        qf_iq_dequant8(w_row1 + ulong(sb) * blk_bytes, smem, lane, w1_a, w1_b);
        { const float4 fx0 = float4(*(const device half4 *)(x_base + 0ul * x_stride + k_off)); const float4 fx1 = float4(*(const device half4 *)(x_base + 0ul * x_stride + k_off + 4u)); a0_0 += dot(fx0, w0_a) + dot(fx1, w0_b); a1_0 += dot(fx0, w1_a) + dot(fx1, w1_b); }
        if (tile_m > 1u) { const float4 fx0 = float4(*(const device half4 *)(x_base + 1ul * x_stride + k_off)); const float4 fx1 = float4(*(const device half4 *)(x_base + 1ul * x_stride + k_off + 4u)); a0_1 += dot(fx0, w0_a) + dot(fx1, w0_b); a1_1 += dot(fx0, w1_a) + dot(fx1, w1_b); }
        if (tile_m > 2u) { const float4 fx0 = float4(*(const device half4 *)(x_base + 2ul * x_stride + k_off)); const float4 fx1 = float4(*(const device half4 *)(x_base + 2ul * x_stride + k_off + 4u)); a0_2 += dot(fx0, w0_a) + dot(fx1, w0_b); a1_2 += dot(fx0, w1_a) + dot(fx1, w1_b); }
        if (tile_m > 3u) { const float4 fx0 = float4(*(const device half4 *)(x_base + 3ul * x_stride + k_off)); const float4 fx1 = float4(*(const device half4 *)(x_base + 3ul * x_stride + k_off + 4u)); a0_3 += dot(fx0, w0_a) + dot(fx1, w0_b); a1_3 += dot(fx0, w1_a) + dot(fx1, w1_b); }
        if (tile_m > 4u) {
            const float4 fx0 = float4(*(const device half4 *)(x_base + 4ul * x_stride + k_off)); const float4 fx1 = float4(*(const device half4 *)(x_base + 4ul * x_stride + k_off + 4u)); a0_4 += dot(fx0, w0_a) + dot(fx1, w0_b); a1_4 += dot(fx0, w1_a) + dot(fx1, w1_b);
            if (tile_m > 5u) { const float4 fx0 = float4(*(const device half4 *)(x_base + 5ul * x_stride + k_off)); const float4 fx1 = float4(*(const device half4 *)(x_base + 5ul * x_stride + k_off + 4u)); a0_5 += dot(fx0, w0_a) + dot(fx1, w0_b); a1_5 += dot(fx0, w1_a) + dot(fx1, w1_b); }
            if (tile_m > 6u) { const float4 fx0 = float4(*(const device half4 *)(x_base + 6ul * x_stride + k_off)); const float4 fx1 = float4(*(const device half4 *)(x_base + 6ul * x_stride + k_off + 4u)); a0_6 += dot(fx0, w0_a) + dot(fx1, w0_b); a1_6 += dot(fx0, w1_a) + dot(fx1, w1_b); }
            if (tile_m > 7u) { const float4 fx0 = float4(*(const device half4 *)(x_base + 7ul * x_stride + k_off)); const float4 fx1 = float4(*(const device half4 *)(x_base + 7ul * x_stride + k_off + 4u)); a0_7 += dot(fx0, w0_a) + dot(fx1, w0_b); a1_7 += dot(fx0, w1_a) + dot(fx1, w1_b); }
        }
    }

    const bool add_res = QF_HAS_RESIDUAL || (p.has_residual != 0u);
    #define QF_STORE_PAIR_IQ(M_OFF, A0, A1) \
        { \
            const float t0 = simd_sum(A0); \
            const float t1 = simd_sum(A1); \
            if (lane == 0u) { \
                const ulong b = ulong(m_base + M_OFF) * p.n; \
                y[b + row0] = half(add_res ? (t0 + float(residual[b + row0])) : t0); \
                if (has_row1) { \
                    y[b + row1] = half(add_res ? (t1 + float(residual[b + row1])) : t1); \
                } \
            } \
        }
    QF_STORE_PAIR_IQ(0u, a0_0, a1_0);
    if (tile_m > 1u) QF_STORE_PAIR_IQ(1u, a0_1, a1_1);
    if (tile_m > 2u) QF_STORE_PAIR_IQ(2u, a0_2, a1_2);
    if (tile_m > 3u) QF_STORE_PAIR_IQ(3u, a0_3, a1_3);
    if (tile_m > 4u) {
        QF_STORE_PAIR_IQ(4u, a0_4, a1_4);
        if (tile_m > 5u) QF_STORE_PAIR_IQ(5u, a0_5, a1_5);
        if (tile_m > 6u) QF_STORE_PAIR_IQ(6u, a0_6, a1_6);
        if (tile_m > 7u) QF_STORE_PAIR_IQ(7u, a0_7, a1_7);
    }
    #undef QF_STORE_PAIR_IQ
}

kernel void ami_qf_iq_dequant_tile_f16(const device uchar *qweight [[buffer(0)]],
                                       const device uchar *scales [[buffer(1)]],
                                       device half *dst_f16 [[buffer(2)]],
                                       const device uchar *codebook [[buffer(3)]],
                                       constant QuantFormatParams &p [[buffer(4)]],
                                       uint2 tg_pos [[threadgroup_position_in_grid]],
                                       uint tid [[thread_index_in_threadgroup]],
                                       uint sg_id [[simdgroup_index_in_threadgroup]],
                                       uint lane [[thread_index_in_simdgroup]]) {
    (void)scales;
    threadgroup IqCodebookSmem smem;
    qf_iq_stage_codebook(codebook, smem, tid);

    const uint local_r = tg_pos.y * 4u + sg_id;
    if (local_r >= p.tile_n) {
        return;
    }
    const uint sb = (p.tile_k_start >> 8) + tg_pos.x;
    if (sb * 256u >= p.tile_k_start + p.tile_k) {
        return;
    }
    const uint row = p.tile_n_start + local_r;
    const uint blk_bytes = qf_iq_blk_bytes(QF_DTYPE);
    float4 w0, w1;
    qf_iq_dequant8(qweight + ulong(row) * p.row_bytes + ulong(sb) * blk_bytes, smem, lane, w0, w1);
    const uint col_local = tg_pos.x * 256u + lane * 8u;
    device half *out = dst_f16 + ulong(local_r) * p.dst_stride + col_local;
    *(device half4 *)(out + 0u) = half4(w0);
    *(device half4 *)(out + 4u) = half4(w1);
}

kernel void ami_qf_iq_moe_gate_up_swiglu(const device half *x [[buffer(0)]],
                                         const device uchar *weights_base [[buffer(1)]],
                                         const device QfExpertWeights *expert_table [[buffer(2)]],
                                         const device QfMoeScheduleHeader *sched_hdr [[buffer(3)]],
                                         const device uint *unique_experts [[buffer(4)]],
                                         const device uint *expert_token_offsets [[buffer(5)]],
                                         const device uint *token_indices [[buffer(6)]],
                                         device half *inter [[buffer(7)]],
                                         const device uchar *codebook [[buffer(8)]],
                                         constant QfMoeParams &p [[buffer(9)]],
                                         uint2 tg_pos [[threadgroup_position_in_grid]],
                                         uint tid [[thread_index_in_threadgroup]],
                                         uint sg_id [[simdgroup_index_in_threadgroup]],
                                         uint lane [[thread_index_in_simdgroup]]) {
    threadgroup IqCodebookSmem smem;
    qf_iq_stage_codebook(codebook, smem, tid);

    constexpr uint ROWS_PER_SG = 2u;
    const uint u = tg_pos.y;
    if (u >= sched_hdr->num_unique_experts) {
        return;
    }
    const uint row0 = tg_pos.x * 8u + sg_id * ROWS_PER_SG;
    if (row0 >= p.intermediate_size) {
        return;
    }
    const uint s0 = expert_token_offsets[u];
    const uint s1 = expert_token_offsets[u + 1u];
    const uint count_u = s1 - s0;
    if (count_u == 0u) {
        return;
    }
    const QfExpertWeights ew = expert_table[unique_experts[u]];
    const device uchar *gate_qw = weights_base + ew.gate_qweight_offset;
    const device uchar *up_qw = weights_base + ew.up_qweight_offset;
    const uint blk_bytes = qf_iq_blk_bytes(QF_DTYPE);

    for (uint t_base = 0u; t_base < count_u; ) {
        const uint rem = min(4u, count_u - t_base);
        const device half *x0 = x + ulong(token_indices[s0 + t_base + 0u]) * p.hidden_size;
        const device half *x1 = x + ulong(token_indices[s0 + t_base + min(1u, rem - 1u)]) * p.hidden_size;
        const device half *x2 = x + ulong(token_indices[s0 + t_base + min(2u, rem - 1u)]) * p.hidden_size;
        const device half *x3 = x + ulong(token_indices[s0 + t_base + min(3u, rem - 1u)]) * p.hidden_size;

        for (uint r = 0u; r < ROWS_PER_SG; r++) {
            const uint row = row0 + r;
            if (row >= p.intermediate_size) {
                break;
            }
            float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
            float u0 = 0.0f, u1 = 0.0f, u2 = 0.0f, u3 = 0.0f;
            const device uchar *gw_row = gate_qw + ulong(row) * p.row_bytes;
            const device uchar *uw_row = up_qw + ulong(row) * p.row_bytes;

            for (uint sb = 0u; sb < p.num_groups; sb++) {
                const uint k_off = sb * 256u + lane * 8u;
                float4 gw0, gw1, uw0, uw1;
                qf_iq_dequant8(gw_row + ulong(sb) * blk_bytes, smem, lane, gw0, gw1);
                qf_iq_dequant8(uw_row + ulong(sb) * blk_bytes, smem, lane, uw0, uw1);
                { const float4 a = float4(*(const device half4 *)(x0 + k_off)); const float4 b = float4(*(const device half4 *)(x0 + k_off + 4u)); g0 += dot(a, gw0) + dot(b, gw1); u0 += dot(a, uw0) + dot(b, uw1); }
                if (rem > 1u) { const float4 a = float4(*(const device half4 *)(x1 + k_off)); const float4 b = float4(*(const device half4 *)(x1 + k_off + 4u)); g1 += dot(a, gw0) + dot(b, gw1); u1 += dot(a, uw0) + dot(b, uw1); }
                if (rem > 2u) { const float4 a = float4(*(const device half4 *)(x2 + k_off)); const float4 b = float4(*(const device half4 *)(x2 + k_off + 4u)); g2 += dot(a, gw0) + dot(b, gw1); u2 += dot(a, uw0) + dot(b, uw1); }
                if (rem > 3u) { const float4 a = float4(*(const device half4 *)(x3 + k_off)); const float4 b = float4(*(const device half4 *)(x3 + k_off + 4u)); g3 += dot(a, gw0) + dot(b, gw1); u3 += dot(a, uw0) + dot(b, uw1); }
            }

            const float sg0 = simd_sum(g0), su0 = simd_sum(u0);
            const float sg1 = simd_sum(g1), su1 = simd_sum(u1);
            const float sg2 = simd_sum(g2), su2 = simd_sum(u2);
            const float sg3 = simd_sum(g3), su3 = simd_sum(u3);
            if (lane == 0u) {
                inter[ulong(s0 + t_base + 0u) * p.intermediate_size + row] = half(qf_apply_swiglu(sg0, su0, p.swiglu_limit));
                if (rem > 1u) inter[ulong(s0 + t_base + 1u) * p.intermediate_size + row] = half(qf_apply_swiglu(sg1, su1, p.swiglu_limit));
                if (rem > 2u) inter[ulong(s0 + t_base + 2u) * p.intermediate_size + row] = half(qf_apply_swiglu(sg2, su2, p.swiglu_limit));
                if (rem > 3u) inter[ulong(s0 + t_base + 3u) * p.intermediate_size + row] = half(qf_apply_swiglu(sg3, su3, p.swiglu_limit));
            }
        }
        t_base += rem;
    }
}

kernel void ami_qf_iq_moe_down_proj(const device half *inter [[buffer(0)]],
                                    const device uchar *weights_base [[buffer(1)]],
                                    const device QfExpertWeights *expert_table [[buffer(2)]],
                                    const device QfMoeScheduleHeader *sched_hdr [[buffer(3)]],
                                    const device uint *unique_experts [[buffer(4)]],
                                    const device uint *expert_token_offsets [[buffer(5)]],
                                    const device uint *token_indices [[buffer(6)]],
                                    const device float *routing_weights [[buffer(7)]],
                                    const device uint *slot_k_indices [[buffer(8)]],
                                    device float *down_out [[buffer(9)]],
                                    const device uchar *codebook [[buffer(10)]],
                                    constant QfMoeParams &p [[buffer(11)]],
                                    uint2 tg_pos [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]],
                                    uint sg_id [[simdgroup_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint ROWS_PER_SG = 2u;
    const uint u = tg_pos.y;
    if (u >= sched_hdr->num_unique_experts) {
        return;
    }
    if (tg_pos.x * 8u >= p.hidden_size) {
        return;
    }
    const uint s0 = expert_token_offsets[u];
    const uint s1 = expert_token_offsets[u + 1u];
    const uint count_u = s1 - s0;
    if (count_u == 0u) {
        return;
    }

    threadgroup IqCodebookSmem smem;
    qf_iq_stage_codebook(codebook, smem, tid);

    const uint row0 = tg_pos.x * 8u + sg_id * ROWS_PER_SG;
    if (row0 >= p.hidden_size) {
        return;
    }
    const QfExpertWeights ew = expert_table[unique_experts[u]];
    const device uchar *down_qw = weights_base + ew.down_qweight_offset;
    const uint blk_bytes = qf_iq_blk_bytes(QF_DTYPE);

    for (uint t_base = 0u; t_base < count_u; ) {
        const uint rem = min(4u, count_u - t_base);
        const device half *i0 = inter + ulong(s0 + t_base + 0u) * p.intermediate_size;
        const device half *i1 = inter + ulong(s0 + t_base + min(1u, rem - 1u)) * p.intermediate_size;
        const device half *i2 = inter + ulong(s0 + t_base + min(2u, rem - 1u)) * p.intermediate_size;
        const device half *i3 = inter + ulong(s0 + t_base + min(3u, rem - 1u)) * p.intermediate_size;

        for (uint r = 0u; r < ROWS_PER_SG; r++) {
            const uint row = row0 + r;
            if (row >= p.hidden_size) {
                break;
            }
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            const device uchar *dw_row = down_qw + ulong(row) * p.row_bytes;

            for (uint sb = 0u; sb < p.num_groups; sb++) {
                const uint k_off = sb * 256u + lane * 8u;
                float4 dw0, dw1;
                qf_iq_dequant8(dw_row + ulong(sb) * blk_bytes, smem, lane, dw0, dw1);
                { const float4 a = float4(*(const device half4 *)(i0 + k_off)); const float4 b = float4(*(const device half4 *)(i0 + k_off + 4u)); a0 += dot(a, dw0) + dot(b, dw1); }
                if (rem > 1u) { const float4 a = float4(*(const device half4 *)(i1 + k_off)); const float4 b = float4(*(const device half4 *)(i1 + k_off + 4u)); a1 += dot(a, dw0) + dot(b, dw1); }
                if (rem > 2u) { const float4 a = float4(*(const device half4 *)(i2 + k_off)); const float4 b = float4(*(const device half4 *)(i2 + k_off + 4u)); a2 += dot(a, dw0) + dot(b, dw1); }
                if (rem > 3u) { const float4 a = float4(*(const device half4 *)(i3 + k_off)); const float4 b = float4(*(const device half4 *)(i3 + k_off + 4u)); a3 += dot(a, dw0) + dot(b, dw1); }
            }

            const float t0 = simd_sum(a0), t1 = simd_sum(a1), t2 = simd_sum(a2), t3 = simd_sum(a3);
            if (lane == 0u) {
                { const uint s = s0 + t_base + 0u; down_out[(ulong(token_indices[s]) * p.top_k + slot_k_indices[s]) * p.hidden_size + row] = t0 * routing_weights[s]; }
                if (rem > 1u) { const uint s = s0 + t_base + 1u; down_out[(ulong(token_indices[s]) * p.top_k + slot_k_indices[s]) * p.hidden_size + row] = t1 * routing_weights[s]; }
                if (rem > 2u) { const uint s = s0 + t_base + 2u; down_out[(ulong(token_indices[s]) * p.top_k + slot_k_indices[s]) * p.hidden_size + row] = t2 * routing_weights[s]; }
                if (rem > 3u) { const uint s = s0 + t_base + 3u; down_out[(ulong(token_indices[s]) * p.top_k + slot_k_indices[s]) * p.hidden_size + row] = t3 * routing_weights[s]; }
            }
        }
        t_base += rem;
    }
}

// ===========================================================================
// G24: Opt-in Activation Quantization-Dequantization Emulation
//   Mode 1: mxfp8_g32 (dynamic [1, 32] UE8M0 + E4M3)
//   Mode 2: fp8_g128  (dynamic [1, 128] F32 + E4M3)
//   Mode 3: nvfp4_g16 (per-token outer + g16 E4M3 + E2M1)
//   Mode 4: nvfp4_fouroversix_mae (per-token outer + g16 E4M3 + E2M1 MAE 4/6 vs 6/6)
// ===========================================================================

static inline uchar qf_mxfp8_ue8m0_from_amax(float amax) {
    if (amax <= 0.0f) {
        return 0u;
    }
    const int e = clamp(int(floor(log2(amax))) - 8 + 127, 0, 254);
    return uchar(e);
}

static inline float qf_half16_sum(float v) {
    v += simd_shuffle_xor(v, 8);
    v += simd_shuffle_xor(v, 4);
    v += simd_shuffle_xor(v, 2);
    v += simd_shuffle_xor(v, 1);
    return v;
}

static inline float qf_half16_max(float v) {
    v = fmax(v, simd_shuffle_xor(v, 8));
    v = fmax(v, simd_shuffle_xor(v, 4));
    v = fmax(v, simd_shuffle_xor(v, 2));
    v = fmax(v, simd_shuffle_xor(v, 1));
    return v;
}

kernel void ami_act_quant_emulate_f16(const device half *x_in [[buffer(0)]],
                                      device half *x_out [[buffer(1)]],
                                      constant ActQuantParams &p [[buffer(2)]],
                                      uint2 tg_pos [[threadgroup_position_in_grid]],
                                      uint tid [[thread_index_in_threadgroup]],
                                      uint sg_id [[simdgroup_index_in_threadgroup]],
                                      uint lane [[thread_index_in_simdgroup]]) {
    if (QF_ACT_QUANT_MODE == QF_ACT_NONE) {
        return;
    }
    const uint row = tg_pos.y;
    if (row >= p.m) {
        return;
    }
    const ulong row_off = ulong(row) * p.k;

    if (QF_ACT_QUANT_MODE == QF_ACT_MXFP8_G32) {
        // Dispatch grid.x = ceil((K / 32) / 4), tg = {128, 1, 1}.
        const uint g = tg_pos.x * 4u + sg_id;
        const uint col = g * 32u + lane;
        if (col >= p.k) {
            return;
        }
        const float v = float(x_in[row_off + col]);
        const float amax = simd_max(fabs(v));
        float out_v = 0.0f;
        if (amax > 0.0f) {
            const float scale = qf_ue8m0_to_f32(qf_mxfp8_ue8m0_from_amax(amax));
            const uchar q = qf_f32_to_e4m3(v / scale);
            out_v = qf_e4m3_to_f32(q) * scale;
        }
        x_out[row_off + col] = half(out_v);
        return;
    }

    if (QF_ACT_QUANT_MODE == QF_ACT_FP8_G128) {
        // Dispatch grid.x = K / 128, tg = {128, 1, 1}.
        threadgroup float sg_max_buf[4];
        const uint col = tg_pos.x * 128u + tid;
        if (col >= p.k) {
            return;
        }
        const float v = float(x_in[row_off + col]);
        const float sm = simd_max(fabs(v));
        if (lane == 0u) {
            sg_max_buf[sg_id] = sm;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float amax = fmax(fmax(sg_max_buf[0], sg_max_buf[1]),
                                fmax(sg_max_buf[2], sg_max_buf[3]));
        float out_v = 0.0f;
        if (amax > 0.0f) {
            const float scale = amax / 448.0f;
            const uchar q = qf_f32_to_e4m3(v / scale);
            out_v = qf_e4m3_to_f32(q) * scale;
        }
        x_out[row_off + col] = half(out_v);
        return;
    }

    if (QF_ACT_QUANT_MODE == QF_ACT_NVFP4_G16 || QF_ACT_QUANT_MODE == QF_ACT_NVFP4_FOUROVERSIX_MAE) {
        // Dispatch grid = {1, M, 1}, tg = {128, 1, 1}: 1 threadgroup owns 1 token row.
        threadgroup float sg_max_buf[4];
        float local_max = 0.0f;
        for (uint col = tid; col < p.k; col += 128u) {
            local_max = fmax(local_max, fabs(float(x_in[row_off + col])));
        }
        const float sm = simd_max(local_max);
        if (lane == 0u) {
            sg_max_buf[sg_id] = sm;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float row_amax = fmax(fmax(sg_max_buf[0], sg_max_buf[1]),
                                    fmax(sg_max_buf[2], sg_max_buf[3]));
        const float outer = (row_amax > 0.0f) ? (row_amax / 2688.0f) : 1.0f;

        const uint num_g16 = p.k / 16u;
        const uint half_sg_id = sg_id * 2u + (lane >> 4); // 0..7 (8 blocks of 16 per step)
        const uint lane_in_16 = lane & 15u;

        for (uint g = half_sg_id; g < num_g16; g += 8u) {
            const uint col = g * 16u + lane_in_16;
            const float v = float(x_in[row_off + col]);
            const float b_amax = qf_half16_max(fabs(v));

            const uchar d6_b = (b_amax > 0.0f) ? qf_f32_to_e4m3(b_amax / (6.0f * outer)) : 0u;
            const float s6 = qf_e4m3_scale_to_f32(d6_b) * outer;
            float x6 = 0.0f;
            if (s6 > 0.0f) {
                x6 = qf_e2m1_to_f32(qf_f32_to_e2m1(v / s6)) * s6;
            }

            if (QF_ACT_QUANT_MODE == QF_ACT_NVFP4_FOUROVERSIX_MAE) {
                const uchar d4_b = (b_amax > 0.0f) ? qf_f32_to_e4m3(b_amax / (4.0f * outer)) : 0u;
                const float s4 = qf_e4m3_scale_to_f32(d4_b) * outer;
                float x4 = 0.0f;
                if (s4 > 0.0f) {
                    x4 = qf_e2m1_to_f32(qf_f32_to_e2m1(v / s4)) * s4;
                }
                const float mae6 = qf_half16_sum(fabs(x6 - v));
                const float mae4 = qf_half16_sum(fabs(x4 - v));
                x_out[row_off + col] = half((mae4 < mae6) ? x4 : x6);
            } else {
                x_out[row_off + col] = half(x6);
            }
        }
    }
}
