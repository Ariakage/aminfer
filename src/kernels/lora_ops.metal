// M9.5 (W30) Universal Multi-LoRA (`r = 1..64`) + PEFT DoRA Metal 4 compute kernels.
//
// Zero-flush two-dispatch single-encoder fusion:
//   1) `ami_lora_down_a`:   z_f32[row, j] = scale * dot(x[row, 0..K], A[j, 0..K])
//   2) `ami_lora_up_b_dora`: y[row, col]  = (has_dora ? dora_scale[col] : 1.0f) *
//                                           (float(y[row, col]) + dot(z_f32[row, 0..rank], B[col, 0..rank]))
//
// Supports:
//   - M >= 1 rows (decode M=1, batched verify/prefill, chunked arbitrary M)
//   - Ranks 1..64 (arbitrary non-power-of-two ranks supported)
//   - Activations x in f16 / bf16 / f32; outputs y in-place f16 / bf16 / f32
//   - Weights A/B in row_major (f16 / bf16) and sme2_panel64 (f16 / bf16)
//   - Both uniform single-adapter mode and heterogeneous per-row C14 batching
//   - Zero thread-private stack spills (`-Wframe-larger-than=0`) and zero warnings
//     under `-std=metal4.0 -Wall -Werror -mmacosx-version-min=26.0`.

#include <metal_stdlib>
using namespace metal;

constant uint LORA_DTYPE_F16 = 0u;
constant uint LORA_DTYPE_BF16 = 1u;

constant uint LORA_FLAG_W_BF16 = 1u << 0;
constant uint LORA_FLAG_PANEL64 = 1u << 1;
constant uint LORA_FLAG_HAS_DORA = 1u << 2;
constant uint LORA_FLAG_ACTIVE = 1u << 3;

constant uint LORA_PARAM_VEC4_X_SAFE = 1u << 0;

constant uint LORA_BATCH_SLOTS_MODE = 0xFFFFFFFEu;

// Exact ABI match with `LoraSlotDesc` in `src/gpu/lora_gpu.zig`.
struct LoraSlotDesc {
    ulong a_byte_offset;
    ulong b_byte_offset;
    ulong dora_scale_byte_offset;
    float scale;
    uint rank;
    uint flags;
    uint _pad0;
};
static_assert(sizeof(LoraSlotDesc) == 40, "LoraSlotDesc must be 40 bytes");

// Exact ABI match with `LoraParams` in `src/gpu/lora_gpu.zig`.
struct LoraParams {
    uint m;
    uint k;
    uint n;
    uint max_rank;
    uint z_stride;
    uint x_dtype;
    uint y_dtype;
    uint single_slot;
    uint num_slots;
    uint x_row_stride;
    uint y_row_stride;
    uint param_flags;
};
static_assert(sizeof(LoraParams) == 48, "LoraParams must be 48 bytes");

static inline float lora_load_scalar(const device uchar *raw, ulong elem_idx, uint dtype) {
    if (dtype == LORA_DTYPE_F16) {
        return float(((const device half *)raw)[elem_idx]);
    }
    if (dtype == LORA_DTYPE_BF16) {
        return float(((const device bfloat *)raw)[elem_idx]);
    }
    return ((const device float *)raw)[elem_idx];
}

static inline float4 lora_load_vec4(const device uchar *raw, ulong elem_idx, uint dtype) {
    if (dtype == LORA_DTYPE_F16) {
        return float4(*(const device half4 *)(raw + elem_idx * 2ul));
    }
    if (dtype == LORA_DTYPE_BF16) {
        return float4(*(const device bfloat4 *)(raw + elem_idx * 2ul));
    }
    return *(const device float4 *)(raw + elem_idx * 4ul);
}

static inline void lora_store_scalar(device uchar *raw, ulong elem_idx, uint dtype, float v) {
    if (dtype == LORA_DTYPE_F16) {
        ((device half *)raw)[elem_idx] = half(v);
    } else if (dtype == LORA_DTYPE_BF16) {
        ((device bfloat *)raw)[elem_idx] = bfloat(v);
    } else {
        ((device float *)raw)[elem_idx] = v;
    }
}

static inline float lora_load_w1(const device uchar *pool_raw, ulong byte_off, bool w_bf16) {
    if (w_bf16) {
        return float(*(const device bfloat *)(pool_raw + byte_off));
    }
    return float(*(const device half *)(pool_raw + byte_off));
}

static inline float2 lora_load_w2(const device uchar *pool_raw, ulong byte_off, bool w_bf16) {
    if (w_bf16) {
        return float2(*(const device bfloat2 *)(pool_raw + byte_off));
    }
    return float2(*(const device half2 *)(pool_raw + byte_off));
}

static inline float4 lora_load_w4(const device uchar *pool_raw, ulong byte_off, bool w_bf16) {
    if (w_bf16) {
        return float4(*(const device bfloat4 *)(pool_raw + byte_off));
    }
    return float4(*(const device half4 *)(pool_raw + byte_off));
}

// ===========================================================================
// Dispatch 1: z_f32[row, j] = scale * dot(x[row, 0..K], A[j, 0..K])
//
// Threadgroup: (128, 1, 1) = 4 simdgroups of 32 lanes; each simdgroup computes
// one rank index `j = tg_id.x * 4 + sg_id` for row `tg_id.y`.
// Grid (threadgroups): ((max_rank + 3) / 4, m, 1).
// ===========================================================================
kernel void ami_lora_down_a(const device uchar *x_raw [[buffer(0)]],
                            const device uchar *pool_raw [[buffer(1)]],
                            const device LoraSlotDesc *descriptors [[buffer(2)]],
                            const device uint *row_slots [[buffer(3)]],
                            device float *z_out [[buffer(4)]],
                            constant LoraParams &p [[buffer(5)]],
                            uint3 tg_id [[threadgroup_position_in_grid]],
                            uint sg_id [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
    const uint row = tg_id.y;
    const uint j = tg_id.x * 4u + sg_id;
    if (row >= p.m || j >= p.max_rank) {
        return;
    }

    const uint sid = (p.single_slot == LORA_BATCH_SLOTS_MODE) ? row_slots[row] : p.single_slot;
    const ulong z_idx = ulong(row) * ulong(p.z_stride) + ulong(j);
    if (sid >= p.num_slots) {
        if (lane == 0u) {
            z_out[z_idx] = 0.0f;
        }
        return;
    }

    const LoraSlotDesc desc = descriptors[sid];
    if (((desc.flags & LORA_FLAG_ACTIVE) == 0u) || j >= desc.rank || desc.scale == 0.0f) {
        if (lane == 0u) {
            z_out[z_idx] = 0.0f;
        }
        return;
    }

    const bool w_bf16 = (desc.flags & LORA_FLAG_W_BF16) != 0u;
    const bool is_panel64 = (desc.flags & LORA_FLAG_PANEL64) != 0u;
    const ulong x_row_base = ulong(row) * ulong(p.x_row_stride);

    float acc = 0.0f;
    if (is_panel64) {
        // SME2 panel64 layout: 16 rows per panel, 64 K-cols per group (2048 B/group).
        // Within each group, 32 kp pairs (lane 0..31) each store 16 adjacent 4-byte pairs.
        const uint k_groups = p.k >> 6;
        const uint p_idx = j >> 4;
        const uint pc = j & 15u;
        const ulong a_panel_base = desc.a_byte_offset +
                                   (ulong(p_idx) * ulong(k_groups)) * 2048ul +
                                   ulong(lane) * 64ul +
                                   ulong(pc) * 4ul;
        for (uint g = 0u; g < k_groups; ++g) {
            const ulong x_idx = x_row_base + ulong(g) * 64ul + ulong(lane) * 2ul;
            const float x0 = lora_load_scalar(x_raw, x_idx, p.x_dtype);
            const float x1 = lora_load_scalar(x_raw, x_idx + 1ul, p.x_dtype);
            const float2 w = lora_load_w2(pool_raw, a_panel_base + ulong(g) * 2048ul, w_bf16);
            acc = fma(x0, w.x, acc);
            acc = fma(x1, w.y, acc);
        }
    } else {
        const ulong a_row_base = desc.a_byte_offset + ulong(j) * ulong(p.k) * 2ul;
        if ((p.param_flags & LORA_PARAM_VEC4_X_SAFE) != 0u) {
            for (uint t = lane * 4u; t < p.k; t += 128u) {
                const float4 xv = lora_load_vec4(x_raw, x_row_base + ulong(t), p.x_dtype);
                const float4 wv = lora_load_w4(pool_raw, a_row_base + ulong(t) * 2ul, w_bf16);
                acc += dot(xv, wv);
            }
        } else {
            for (uint t = lane; t < p.k; t += 32u) {
                const float xv = lora_load_scalar(x_raw, x_row_base + ulong(t), p.x_dtype);
                const float wv = lora_load_w1(pool_raw, a_row_base + ulong(t) * 2ul, w_bf16);
                acc = fma(xv, wv, acc);
            }
        }
    }

    const float sum = simd_sum(acc);
    if (lane == 0u) {
        z_out[z_idx] = sum * desc.scale;
    }
}

// ===========================================================================
// Dispatch 2: y[row, col] = (has_dora ? dora_scale[col] : 1.0f) *
//                           (float(y[row, col]) + dot(z_f32[row, 0..rank], B[col, 0..rank]))
//
// Threadgroup: (256, 1, 1); Grid (threadgroups): ((n + 255) / 256, m, 1).
// ===========================================================================
kernel void ami_lora_up_b_dora(const device float *z_in [[buffer(0)]],
                               const device uchar *pool_raw [[buffer(1)]],
                               const device LoraSlotDesc *descriptors [[buffer(2)]],
                               const device uint *row_slots [[buffer(3)]],
                               device uchar *y_raw [[buffer(4)]],
                               constant LoraParams &p [[buffer(5)]],
                               uint3 gid [[thread_position_in_grid]]) {
    const uint col = gid.x;
    const uint row = gid.y;
    if (col >= p.n || row >= p.m) {
        return;
    }

    const uint sid = (p.single_slot == LORA_BATCH_SLOTS_MODE) ? row_slots[row] : p.single_slot;
    if (sid >= p.num_slots) {
        return;
    }

    const LoraSlotDesc desc = descriptors[sid];
    if (((desc.flags & LORA_FLAG_ACTIVE) == 0u) || desc.rank == 0u) {
        return;
    }

    const bool has_dora = (desc.flags & LORA_FLAG_HAS_DORA) != 0u;
    if (desc.scale == 0.0f && !has_dora) {
        return;
    }

    float delta = 0.0f;
    if (desc.scale != 0.0f) {
        const bool w_bf16 = (desc.flags & LORA_FLAG_W_BF16) != 0u;
        const bool is_panel64 = (desc.flags & LORA_FLAG_PANEL64) != 0u;
        const device float *z_row = z_in + ulong(row) * ulong(p.z_stride);

        if (is_panel64) {
            // B in sme2_panel64 is [N, 64] (1 group per 16-row panel = 2048 B/panel).
            const uint p_idx = col >> 4;
            const uint pc = col & 15u;
            const ulong b_base = desc.b_byte_offset + ulong(p_idx) * 2048ul + ulong(pc) * 4ul;
            const uint kp_lim = (desc.rank + 1u) >> 1;
            for (uint kp = 0u; kp < kp_lim; ++kp) {
                const float z0 = z_row[2u * kp];
                const float z1 = (2u * kp + 1u < desc.rank) ? z_row[2u * kp + 1u] : 0.0f;
                const float2 w = lora_load_w2(pool_raw, b_base + ulong(kp) * 64ul, w_bf16);
                delta = fma(z0, w.x, delta);
                delta = fma(z1, w.y, delta);
            }
        } else {
            const ulong b_row_base = desc.b_byte_offset + ulong(col) * ulong(desc.rank) * 2ul;
            if ((desc.rank & 3u) == 0u) {
                for (uint j = 0u; j < desc.rank; j += 4u) {
                    const float4 zv = *(const device float4 *)(z_row + j);
                    const float4 wv = lora_load_w4(pool_raw, b_row_base + ulong(j) * 2ul, w_bf16);
                    delta += dot(zv, wv);
                }
            } else {
                for (uint j = 0u; j < desc.rank; ++j) {
                    const float wv = lora_load_w1(pool_raw, b_row_base + ulong(j) * 2ul, w_bf16);
                    delta = fma(z_row[j], wv, delta);
                }
            }
        }
    }

    const ulong y_idx = ulong(row) * ulong(p.y_row_stride) + ulong(col);
    float out_val = lora_load_scalar(y_raw, y_idx, p.y_dtype) + delta;
    if (has_dora) {
        const device float *dora_scale = (const device float *)(pool_raw + desc.dora_scale_byte_offset);
        out_val *= dora_scale[col];
    }
    lora_store_scalar(y_raw, y_idx, p.y_dtype, out_val);
}
