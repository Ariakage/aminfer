// M8.2 (G25) Inkling 2 and Hy4 (`hy_v4`, `hy_v4_vl`, `hy_v4_flash`) fused Metal 4 operators.
//
// Compiled with `-std=metal4.0 -Wall -Werror -Wframe-larger-than=1 -mmacosx-version-min=26.0`:
// zero warnings and zero thread-private stack spills across all kernels.
#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

constant bool NORM_W_BF16 [[function_constant(0)]];
constant uint INK_HEAD_DIM [[function_constant(1)]];
constant uint HY4_KV_LORA_RANK [[function_constant(2)]];
constant bool INK_LOGITS_F16 [[function_constant(3)]];
constant uint INK_MOE_DTYPE_RAW [[function_constant(4)]];

constant uint INK_MOE_DTYPE = is_function_constant_defined(INK_MOE_DTYPE_RAW) ? INK_MOE_DTYPE_RAW : 2u;

static inline float ink_load_norm_w(const device uchar *raw, uint idx, bool is_bf16) {
    if (is_bf16) {
        return float(((const device bfloat *)raw)[idx]);
    }
    return float(((const device half *)raw)[idx]);
}

static inline float ink_sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

static inline float ink_silu(float x) {
    return x / (1.0f + exp(-x));
}

static inline float ink_tanh_safe(float x) {
    return tanh(clamp(x, -20.0f, 20.0f));
}

static inline float ink_softcap(float x, float cap) {
    if (cap <= 0.0f) {
        return x;
    }
    return cap * ink_tanh_safe(x / cap);
}

static inline float ink_apply_gate(float val, float g, uint gate_mode) {
    if (gate_mode == 0u) {
        return val * ink_sigmoid(g);
    }
    if (gate_mode == 1u) {
        return val * ink_silu(g);
    }
    return val * g;
}

static inline uint ink_f32_to_ordered_u32(float x) {
    if (x == 0.0f) {
        x = 0.0f;
    }
    const uint u = as_type<uint>(x);
    const uint mask = ((u >> 31) != 0u) ? 0xFFFFFFFFu : 0x80000000u;
    return u ^ mask;
}

// ===========================================================================
// Uniform Parameter Structs (exact ABI match with src/gpu/inkling2_hy4.zig)
// ===========================================================================

struct Sconv4Params {
    uint m;
    uint hidden_dim;
    uint state_pos;
    uint write_state;
    uint is_dynamic;
    uint has_bias;
    uint apply_rms_norm;
    uint apply_silu;
    uint conv_w_f16;
    uint serial_tiles;
    float rms_eps;
    float post_residual_scale;
};
static_assert(sizeof(Sconv4Params) == 48, "Sconv4Params must be 48 bytes");

struct Inkling2RelGqaParams {
    uint num_nodes;
    uint num_q_heads;
    uint num_kv_heads;
    uint head_dim;
    uint kv_len;
    uint window_size;
    uint ring_capacity;
    uint chunk_tokens;
    uint causal_q_start_pos;
    uint d_rel;
    uint rel_extent;
    uint rel_per_head;
    uint use_log_scaling;
    uint has_sink;
    uint sink_per_token;
    uint has_gate_attn;
    uint gate_mode;
    uint has_tree;
    float scale;
    float qk_l2_clip;
    float rel_l2_clip;
    float rel_scale;
    float log_scale_alpha;
    float attn_softcap;
};
static_assert(sizeof(Inkling2RelGqaParams) == 96, "Inkling2RelGqaParams must be 96 bytes");

struct Inkling2MoeRouteParams {
    uint m;
    uint num_experts;
    uint top_k;
    uint logits_f16;
    uint has_bias;
    uint bias_f16;
    uint bias_mode;
    uint norm_topk_prob;
    float route_scale;
    float router_softcap;
    uint scoring_sigmoid;
    uint _pad1;
};
static_assert(sizeof(Inkling2MoeRouteParams) == 48, "Inkling2MoeRouteParams must be 48 bytes");

struct Inkling2MoeSwigluParams {
    uint num_rows;
    uint intermediate_size;
    uint interleaved_gate_up;
    float softcap;
    float gate_softcap;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};
static_assert(sizeof(Inkling2MoeSwigluParams) == 32, "Inkling2MoeSwigluParams must be 32 bytes");

struct Inkling2MoeEpilogueParams {
    uint m;
    uint hidden_dim;
    uint top_k;
    uint has_residual;
    uint weights_f16;
    float mlp_softcap;
    float res_scale;
    uint _pad0;
};
static_assert(sizeof(Inkling2MoeEpilogueParams) == 32, "Inkling2MoeEpilogueParams must be 32 bytes");

struct Inkling2AttnResParams {
    uint m;
    uint hidden_dim;
    uint num_blocks;
    uint block_stride_elems;
    uint apply_rms_norm;
    uint has_norm_w;
    uint query_per_token;
    float rms_eps;
    float scale;
    uint normalize_values;
    uint _pad1;
    uint _pad2;
    uint block_slots[8];
};
static_assert(sizeof(Inkling2AttnResParams) == 80, "Inkling2AttnResParams must be 80 bytes");

struct Hy4IndexCompressParams {
    uint kv_len;
    uint idx_dim;
    uint num_k_heads;
    uint compress_ratio;
    uint pool_mode;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};
static_assert(sizeof(Hy4IndexCompressParams) == 32, "Hy4IndexCompressParams must be 32 bytes");

struct Hy4IndexPoolTopkParams {
    uint num_nodes;
    uint num_q_heads;
    uint num_k_heads;
    uint idx_dim;
    uint kv_len;
    uint causal_q_start_pos;
    uint compress_ratio;
    uint group_topk;
    uint candidate_topk;
    uint score_mode;
    uint has_head_weights;
    uint swa_window;
    uint max_groups_stride;
    uint max_cand_stride;
    float scale;
    uint _pad0;
};
static_assert(sizeof(Hy4IndexPoolTopkParams) == 64, "Hy4IndexPoolTopkParams must be 64 bytes");

struct Hy4LatentMoeRouteParams {
    uint m;
    uint num_experts;
    uint top_k;
    uint logits_f16;
    uint has_bias;
    uint bias_f16;
    uint norm_topk_prob;
    uint _pad0;
    float routed_scaling_factor;
    float eps;
    uint _pad1;
    uint _pad2;
};
static_assert(sizeof(Hy4LatentMoeRouteParams) == 48, "Hy4LatentMoeRouteParams must be 48 bytes");

struct Hy4SwigluClampedParams {
    uint num_rows;
    uint intermediate_size;
    uint interleaved_gate_up;
    uint clamp_mode;
    uint has_linear_gate;
    uint linear_gate_mode;
    float swiglu_limit;
    uint _pad0;
};
static_assert(sizeof(Hy4SwigluClampedParams) == 32, "Hy4SwigluClampedParams must be 32 bytes");

struct Hy4CombineSandwichParams {
    uint m;
    uint hidden_dim;
    uint top_k;
    uint has_shared;
    uint has_residual;
    uint apply_combine_norm;
    uint apply_post_norm;
    uint weights_f16;
    float rms_eps;
    float res_scale;
    uint _pad0;
    uint _pad1;
};
static_assert(sizeof(Hy4CombineSandwichParams) == 48, "Hy4CombineSandwichParams must be 48 bytes");

struct Hy4MlaSinkAttnParams {
    uint num_nodes;
    uint num_q_heads;
    uint kv_lora_rank;
    uint qk_rope_dim;
    uint kv_len;
    uint causal_q_start_pos;
    uint swa_window;
    uint candidate_topk;
    uint has_sink;
    uint has_linear_gate;
    uint linear_gate_mode;
    uint has_tree;
    float scale;
    float attn_softcap;
    uint _pad0;
    uint _pad1;
};
static_assert(sizeof(Hy4MlaSinkAttnParams) == 64, "Hy4MlaSinkAttnParams must be 64 bytes");

struct InkKvTilePair {
    texture2d<half, access::read> k_tex;
    texture2d<half, access::read> v_tex;
};
static_assert(sizeof(InkKvTilePair) == 16, "InkKvTilePair must be 16 bytes");

// ===========================================================================
// Helper: Threadgroup Radix Top-K Select (zero stack arrays)
// ===========================================================================
// Selects `k` indices from `scores[0..n)` (values in `scores`), breaking ties
// strictly by lowest logical index, and writes selected token/group IDs in
// ascending element order into `out_ids[0..k)`.
static inline void ink_tg_radix_topk(const device float *scores,
                                     const device uint *src_ids,
                                     bool use_src_ids,
                                     uint n,
                                     uint k,
                                     device uint *out_ids,
                                     uint tid,
                                     uint sg_id,
                                     uint lane,
                                     threadgroup atomic_uint *tg_hist,
                                     threadgroup uint *tg_scan,
                                     threadgroup uint *tg_win_nib,
                                     threadgroup uint *tg_scalar) {
    if (k == 0u || n == 0u) {
        return;
    }
    if (n <= k) {
        for (uint i = tid; i < n; i += 256u) {
            out_ids[i] = use_src_ids ? src_ids[i] : i;
        }
        return;
    }

    uint prefix = 0u;
    uint mask_above = 0u;
    uint k_rem = k;

    for (uint pass = 0u; pass < 8u; pass++) {
        const uint shift = (7u - pass) * 4u;
        if (tid < 16u) {
            atomic_store_explicit(&tg_hist[tid], 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = tid; i < n; i += 256u) {
            const uint ord = ink_f32_to_ordered_u32(scores[i]);
            if ((ord & mask_above) == prefix) {
                const uint nib = (ord >> shift) & 0xFu;
                atomic_fetch_add_explicit(&tg_hist[nib], 1u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0u) {
            uint cum = 0u;
            uint win = 0u;
            uint prev_cum = 0u;
            for (int nb = 15; nb >= 0; --nb) {
                const uint cnt = atomic_load_explicit(&tg_hist[uint(nb)], memory_order_relaxed);
                if (cum + cnt >= k_rem) {
                    win = uint(nb);
                    prev_cum = cum;
                    break;
                }
                cum += cnt;
            }
            *tg_win_nib = win;
            *tg_scalar = k_rem - prev_cum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        prefix |= (*tg_win_nib << shift);
        mask_above |= (0xFu << shift);
        k_rem = *tg_scalar;
    }

    // Count strictly greater elements in one parallel pass.
    uint local_gt = 0u;
    for (uint i = tid; i < n; i += 256u) {
        if (ink_f32_to_ordered_u32(scores[i]) > prefix) {
            local_gt += 1u;
        }
    }
    const uint sg_gt = simd_sum(local_gt);
    if (lane == 0u) {
        tg_scan[sg_id] = sg_gt;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        uint total_gt = 0u;
        for (uint s = 0u; s < 8u; s++) {
            total_gt += tg_scan[s];
        }
        *tg_scalar = (k > total_gt) ? (k - total_gt) : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint ties_needed = *tg_scalar;
    uint eq_written = 0u;
    uint out_written = 0u;

    for (uint base = 0u; base < n; base += 256u) {
        const uint idx = base + tid;
        const bool in_bounds = (idx < n);
        uint ord = 0u;
        if (in_bounds) {
            ord = ink_f32_to_ordered_u32(scores[idx]);
        }
        const bool is_gt = in_bounds && (ord > prefix);
        const bool is_eq = in_bounds && (ord == prefix);

        // Prefix sum of is_eq across 256 threads in ascending idx order.
        const uint eq_u = is_eq ? 1u : 0u;
        const uint eq_lane_ex = simd_prefix_exclusive_sum(eq_u);
        if (lane == 31u) {
            tg_scan[sg_id] = eq_lane_ex + eq_u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            uint run = 0u;
            for (uint s = 0u; s < 8u; s++) {
                const uint v = tg_scan[s];
                tg_scan[s] = run;
                run += v;
            }
            *tg_win_nib = run;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint eq_rank = eq_written + tg_scan[sg_id] + eq_lane_ex;
        const uint chunk_eq_total = *tg_win_nib;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const bool take = is_gt || (is_eq && (eq_rank < ties_needed));
        const uint take_u = take ? 1u : 0u;
        const uint out_lane_ex = simd_prefix_exclusive_sum(take_u);
        if (lane == 31u) {
            tg_scan[sg_id] = out_lane_ex + take_u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            uint run = 0u;
            for (uint s = 0u; s < 8u; s++) {
                const uint v = tg_scan[s];
                tg_scan[s] = run;
                run += v;
            }
            *tg_win_nib = run;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint out_pos = out_written + tg_scan[sg_id] + out_lane_ex;
        if (take && out_pos < k) {
            out_ids[out_pos] = use_src_ids ? src_ids[idx] : idx;
        }
        eq_written += chunk_eq_total;
        out_written += *tg_win_nib;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ===========================================================================
// 1. Inkling 2: ami_inkling2_rmsnorm_sconv4
// ===========================================================================
// Register-fused RMSNorm + k=4 causal depthwise short conv (`sconv`) with 0
// intermediate global-memory roundtrips. Supports static `conv_w` [D, 4] and
// dynamic per-token `dyn_w` [M, D, 4] (`weight_proj`), in-place 3-step ring
// updates for `M = 1`, and causal scans for `M > 1` (e.g. `M = 8` and `M = 33`).
//
// Grid: `(serial_tiles != 0) ? 1 : ceil(M / 64)` threadgroups of 256 threads.
kernel void ami_inkling2_rmsnorm_sconv4(const device half *x [[buffer(0)]],
                                        const device uchar *norm_w [[buffer(1)]],
                                        const device uchar *conv_w [[buffer(2)]],
                                        const device half *dyn_w [[buffer(3)]],
                                        const device float *conv_bias [[buffer(4)]],
                                        const device half *conv_state_in [[buffer(5)]],
                                        device half *conv_state_out [[buffer(6)]],
                                        device half *out [[buffer(7)]],
                                        constant Sconv4Params &p [[buffer(8)]],
                                        uint tg_x [[threadgroup_position_in_grid]],
                                        uint tid [[thread_index_in_threadgroup]],
                                        uint sg_id [[simdgroup_index_in_threadgroup]],
                                        uint lane [[thread_index_in_simdgroup]]) {
    if (p.m == 0u || p.hidden_dim == 0u) {
        return;
    }

    constexpr uint T_TILE = 64u;
    threadgroup float tg_inv_rms[68];
    threadgroup float tg_reduce[8];

    const uint d = p.hidden_dim;
    const uint num_tiles = (p.serial_tiles != 0u) ? ((p.m + T_TILE - 1u) / T_TILE) : 1u;
    const uint start_tile = (p.serial_tiles != 0u) ? 0u : tg_x;

    for (uint t_step = 0u; t_step < num_tiles; t_step++) {
        const uint tile_idx = start_tile + t_step;
        const uint t_base = tile_idx * T_TILE;
        if (t_base >= p.m) {
            return;
        }
        const uint tile_len = min(T_TILE, p.m - t_base);
        const uint halo = (t_base >= 3u) ? 3u : 0u;
        const uint total_rms_toks = halo + tile_len;

        // Compute inverse RMS scalar per token in the tile (+ 3 halo tokens if t_base > 0).
        for (uint u = 0u; u < total_rms_toks; u++) {
            float inv_rms = 1.0f;
            if (p.apply_rms_norm != 0u) {
                const uint tok = (t_base + u) - halo;
                const device half *x_row = x + ulong(tok) * d;
                float sum_sq = 0.0f;
                if ((d & 3u) == 0u) {
                    const uint vlen = d >> 2;
                    const device half4 *x4 = (const device half4 *)x_row;
                    for (uint i = tid; i < vlen; i += 256u) {
                        const float4 v = float4(x4[i]);
                        sum_sq += dot(v, v);
                    }
                } else {
                    for (uint i = tid; i < d; i += 256u) {
                        const float v = float(x_row[i]);
                        sum_sq = fma(v, v, sum_sq);
                    }
                }
                const float sg_s = simd_sum(sum_sq);
                if (lane == 0u) {
                    tg_reduce[sg_id] = sg_s;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid == 0u) {
                    float tot = 0.0f;
                    for (uint s = 0u; s < 8u; s++) {
                        tot += tg_reduce[s];
                    }
                    tg_inv_rms[u] = rsqrt(tot / float(d) + p.rms_eps);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                inv_rms = tg_inv_rms[u];
            } else if (tid == 0u) {
                tg_inv_rms[u] = inv_rms;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint s0 = (p.state_pos + 0u) % 3u;
        const uint s1 = (p.state_pos + 1u) % 3u;
        const uint s2 = (p.state_pos + 2u) % 3u;

        // Stream each channel independently with 3 scalar history registers.
        for (uint c = tid; c < d; c += 256u) {
            const float nw = (p.apply_rms_norm != 0u) ? ink_load_norm_w(norm_w, c, NORM_W_BF16) : 1.0f;
            float h_m3 = 0.0f;
            float h_m2 = 0.0f;
            float h_m1 = 0.0f;

            if (t_base == 0u) {
                h_m3 = float(conv_state_in[ulong(s0) * d + c]);
                h_m2 = float(conv_state_in[ulong(s1) * d + c]);
                h_m1 = float(conv_state_in[ulong(s2) * d + c]);
            } else {
                h_m3 = float(x[ulong(t_base - 3u) * d + c]) * tg_inv_rms[0] * nw;
                h_m2 = float(x[ulong(t_base - 2u) * d + c]) * tg_inv_rms[1] * nw;
                h_m1 = float(x[ulong(t_base - 1u) * d + c]) * tg_inv_rms[2] * nw;
            }

            float sw0 = 0.0f;
            float sw1 = 0.0f;
            float sw2 = 0.0f;
            float sw3 = 0.0f;
            if (p.is_dynamic == 0u) {
                const ulong w_off = ulong(c) * 4ul;
                if (p.conv_w_f16 != 0u) {
                    const device half *cw = (const device half *)conv_w;
                    sw0 = float(cw[w_off + 0ul]);
                    sw1 = float(cw[w_off + 1ul]);
                    sw2 = float(cw[w_off + 2ul]);
                    sw3 = float(cw[w_off + 3ul]);
                } else {
                    const device float *cw = (const device float *)conv_w;
                    sw0 = cw[w_off + 0ul];
                    sw1 = cw[w_off + 1ul];
                    sw2 = cw[w_off + 2ul];
                    sw3 = cw[w_off + 3ul];
                }
            }
            const float bias_c = (p.has_bias != 0u) ? conv_bias[c] : 0.0f;

            for (uint u = 0u; u < tile_len; u++) {
                const uint tok = t_base + u;
                const float h_cur = float(x[ulong(tok) * d + c]) * tg_inv_rms[halo + u] * nw;

                float w0 = sw0;
                float w1 = sw1;
                float w2 = sw2;
                float w3 = sw3;
                if (p.is_dynamic != 0u) {
                    const ulong dw_off = (ulong(tok) * d + c) * 4ul;
                    w0 = float(dyn_w[dw_off + 0ul]);
                    w1 = float(dyn_w[dw_off + 1ul]);
                    w2 = float(dyn_w[dw_off + 2ul]);
                    w3 = float(dyn_w[dw_off + 3ul]);
                }

                float acc = bias_c;
                acc = fma(w0, h_m3, acc);
                acc = fma(w1, h_m2, acc);
                acc = fma(w2, h_m1, acc);
                acc = fma(w3, h_cur, acc);
                if (p.post_residual_scale > 0.0f) {
                    acc = (acc + h_cur) * p.post_residual_scale;
                }
                if (p.apply_silu != 0u) {
                    acc = ink_silu(acc);
                }
                out[ulong(tok) * d + c] = half(acc);

                h_m3 = h_m2;
                h_m2 = h_m1;
                h_m1 = h_cur;
            }

            if (p.write_state != 0u && (t_base + tile_len == p.m)) {
                const uint dst0 = (p.state_pos + p.m + 0u) % 3u;
                const uint dst1 = (p.state_pos + p.m + 1u) % 3u;
                const uint dst2 = (p.state_pos + p.m + 2u) % 3u;
                conv_state_out[ulong(dst0) * d + c] = half(h_m3);
                conv_state_out[ulong(dst1) * d + c] = half(h_m2);
                conv_state_out[ulong(dst2) * d + c] = half(h_m1);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ===========================================================================
// 2. Inkling 2: ami_inkling2_rel_gqa_attn
// ===========================================================================
// Single-pass register-resident GQA / 1024-token HybridKvRing SWA attention
// fusing qk_l2_clip (22.627), log_scaling (1 + log(pos + 1)), d_rel=16
// (rel_extent=1024) + rel_l2_clip (8.0), attn_sink_proj, attn_softcap, and
// elementwise gate_attn output gating with 0 intermediate score buffer writes.
//
// Grid: (num_kv_heads * q_groups_per_kv, num_nodes, 1) threadgroups, 256 threads.
static inline float ink_rel_gqa_score(float4 q0,
                                        float4 q1,
                                        float4 k0,
                                        float4 k1,
                                        float q_rel_v,
                                        const device half *rel_emb,
                                        ulong rel_head_off,
                                        uint q_pos,
                                        uint k_pos,
                                        uint d_rel,
                                        uint rel_extent,
                                        float qk_l2_clip,
                                        float rel_l2_clip,
                                        float scale_eff,
                                        float rel_scale,
                                        float attn_softcap,
                                        bool rel_zero_oob,
                                        uint lane) {
    float k_clip_fac = 1.0f;
    if (qk_l2_clip > 0.0f) {
        const float k_sq = simd_sum(dot(k0, k0) + dot(k1, k1));
        const float k_norm = sqrt(k_sq);
        if (k_norm > qk_l2_clip) {
            k_clip_fac = qk_l2_clip / k_norm;
        }
    }
    const float qk_dot = simd_sum(dot(q0, k0) + dot(q1, k1)) * k_clip_fac;

    float rel_score = 0.0f;
    if (d_rel > 0u && rel_extent > 0u) {
        const bool in_range = (q_pos >= k_pos) && ((q_pos - k_pos) < rel_extent);
        if (!rel_zero_oob || in_range) {
            const uint dist = (q_pos >= k_pos) ? min(q_pos - k_pos, rel_extent - 1u) : 0u;
            const float rk = (lane < d_rel)
                ? float(rel_emb[(rel_head_off + dist) * d_rel + lane])
                : 0.0f;
            float rk_clip = 1.0f;
            if (rel_l2_clip > 0.0f) {
                const float rk_sq = simd_sum(rk * rk);
                const float rk_norm = sqrt(rk_sq);
                if (rk_norm > rel_l2_clip) {
                    rk_clip = rel_l2_clip / rk_norm;
                }
            }
            rel_score = simd_sum(q_rel_v * rk) * rk_clip * rel_scale;
        }
    }

    return ink_softcap(qk_dot * scale_eff + rel_score, attn_softcap);
}

kernel void ami_inkling2_rel_gqa_attn(const device half *q [[buffer(0)]],
                                      const device half *q_rel [[buffer(1)]],
                                      const device half *rel_emb [[buffer(2)]],
                                      const device InkKvTilePair *kv_tiles [[buffer(3)]],
                                      const device half *k_tree [[buffer(4)]],
                                      const device half *v_tree [[buffer(5)]],
                                      const device ulong *ancestor_masks [[buffer(6)]],
                                      const device float *sink_logits [[buffer(7)]],
                                      const device half *gate_attn [[buffer(8)]],
                                      device half *out [[buffer(9)]],
                                      constant Inkling2RelGqaParams &p [[buffer(10)]],
                                      uint2 tg_pos [[threadgroup_position_in_grid]],
                                      uint tid [[thread_index_in_threadgroup]],
                                      uint sg_id [[simdgroup_index_in_threadgroup]],
                                      uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint Q_TILE = 8u;
    constexpr float LOG2E = 1.4426950408889634f;

    if (p.num_nodes == 0u || p.num_kv_heads == 0u || p.num_q_heads == 0u) {
        return;
    }
    const uint d_dim = (INK_HEAD_DIM > 0u) ? INK_HEAD_DIM : p.head_dim;
    if (d_dim == 0u || d_dim > 256u) {
        return;
    }

    const uint gqa_ratio = p.num_q_heads / p.num_kv_heads;
    const uint q_groups_per_kv = (gqa_ratio + Q_TILE - 1u) / Q_TILE;
    const uint kv_head = tg_pos.x / q_groups_per_kv;
    const uint q_grp = tg_pos.x % q_groups_per_kv;
    const uint node = tg_pos.y;
    if (kv_head >= p.num_kv_heads || node >= p.num_nodes) {
        return;
    }

    const uint q_start = kv_head * gqa_ratio + q_grp * Q_TILE;
    const uint q_end = min(q_start + Q_TILE, (kv_head + 1u) * gqa_ratio);
    const uint q_cnt = q_end - q_start;
    const bool active_q = (sg_id < q_cnt);
    const uint qh = q_start + min(sg_id, q_cnt - 1u);

    const ulong anc_mask = (p.has_tree != 0u) ? ancestor_masks[node] : 0ul;
    const uint tree_anc = (p.has_tree != 0u) ? uint(popcount(anc_mask)) : 0u;
    const uint tree_depth = (tree_anc > 0u) ? (tree_anc - 1u) : 0u;
    const uint q_pos = (p.has_tree != 0u)
        ? (p.kv_len + tree_depth)
        : ((p.causal_q_start_pos != 0xFFFFFFFFu)
            ? (p.causal_q_start_pos + node)
            : ((p.kv_len > 0u) ? (p.kv_len - 1u) : 0u));

    const uint eff_kv_len = (p.has_tree != 0u)
        ? p.kv_len
        : ((p.causal_q_start_pos != 0xFFFFFFFFu)
            ? min(p.kv_len, q_pos + 1u)
            : p.kv_len);
    const uint win_start = (p.window_size > 0u && (q_pos + 1u) > p.window_size)
        ? ((q_pos + 1u) - p.window_size)
        : 0u;

    const float log_mult = (p.use_log_scaling != 0u)
        ? (1.0f + p.log_scale_alpha * log(float(q_pos) + 1.0f))
        : 1.0f;
    const float scale_eff = p.scale * log_mult;

    // Load Q into up to 2 float4 registers per lane (handles d <= 256 with 0 arrays).
    const ulong q_row_off = (ulong(node) * p.num_q_heads + qh) * d_dim;
    const uint c0 = lane * 4u;
    const uint c1 = 128u + lane * 4u;
    const bool has_c0 = (c0 < d_dim);
    const bool has_c1 = (c1 < d_dim);

    float4 q0 = (active_q && has_c0) ? float4(*(const device half4 *)(q + q_row_off + c0)) : float4(0.0f);
    float4 q1 = (active_q && has_c1) ? float4(*(const device half4 *)(q + q_row_off + c1)) : float4(0.0f);

    if (p.qk_l2_clip > 0.0f && active_q) {
        const float q_sq = simd_sum(dot(q0, q0) + dot(q1, q1));
        const float q_norm = sqrt(q_sq);
        if (q_norm > p.qk_l2_clip) {
            const float fac = p.qk_l2_clip / q_norm;
            q0 *= fac;
            q1 *= fac;
        }
    }

    // Load and clip d_rel <= 16 low-rank relative query vector in lanes 0..d_rel-1.
    const bool active_rel_lane = active_q && (p.d_rel > 0u) && (lane < p.d_rel);
    const ulong q_rel_off = (ulong(node) * p.num_q_heads + qh) * p.d_rel;
    float q_rel_v = active_rel_lane ? float(q_rel[q_rel_off + lane]) : 0.0f;
    if (p.d_rel > 0u && p.rel_l2_clip > 0.0f && active_q) {
        const float qr_sq = simd_sum(q_rel_v * q_rel_v);
        const float qr_norm = sqrt(qr_sq);
        if (qr_norm > p.rel_l2_clip) {
            q_rel_v *= (p.rel_l2_clip / qr_norm);
        }
    }
    const ulong rel_head_off = ((p.rel_per_head & 1u) != 0u)
        ? (ulong(qh) * p.rel_extent)
        : 0ul;
    const bool rel_zero_oob = ((p.rel_per_head & 2u) != 0u);

    float m_val = -INFINITY;
    float l_val = 0.0f;
    if (p.has_sink != 0u && active_q) {
        const ulong sink_idx = (p.sink_per_token != 0u)
            ? (ulong(node) * p.num_q_heads + qh)
            : ulong(qh);
        m_val = sink_logits[sink_idx] * LOG2E;
        l_val = 1.0f;
    }
    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);

    threadgroup half4 tg_k[256];
    threadgroup half4 tg_v[256];

    const uint vec_cols = d_dim >> 2; // 16 for d=64, 32 for d=128, 64 for d=256
    const uint toks_per_tile = (vec_cols <= 16u) ? 16u : ((vec_cols <= 32u) ? 8u : 4u);
    const uint load_stride = (vec_cols <= 16u) ? 16u : ((vec_cols <= 32u) ? 32u : 64u);
    const uint load_tok = tid / load_stride;
    const uint load_col = tid - load_tok * load_stride;

    for (uint tile_t = win_start; tile_t < eff_kv_len; tile_t += toks_per_tile) {
        const uint tile_len = min(toks_per_tile, eff_kv_len - tile_t);
        if (load_tok < tile_len && load_col < vec_cols) {
            const uint t = tile_t + load_tok;
            const uint phys_t = (p.ring_capacity > 0u) ? (t % p.ring_capacity) : t;
            const uint chunk = phys_t / p.chunk_tokens;
            const uint slot = phys_t - chunk * p.chunk_tokens;
            const uint row = kv_head * p.chunk_tokens + slot;
            tg_k[load_tok * vec_cols + load_col] = kv_tiles[chunk].k_tex.read(uint2(load_col, row));
            tg_v[load_tok * vec_cols + load_col] = kv_tiles[chunk].v_tex.read(uint2(load_col, row));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active_q) {
            for (uint dt = 0u; dt < tile_len; dt++) {
                const uint base_idx = dt * vec_cols;
                const float4 k0 = has_c0 ? float4(tg_k[base_idx + lane]) : float4(0.0f);
                const float4 k1 = has_c1 ? float4(tg_k[base_idx + 32u + lane]) : float4(0.0f);
                const float4 v0 = has_c0 ? float4(tg_v[base_idx + lane]) : float4(0.0f);
                const float4 v1 = has_c1 ? float4(tg_v[base_idx + 32u + lane]) : float4(0.0f);

                const float score = ink_rel_gqa_score(
                    q0, q1, k0, k1, q_rel_v, rel_emb, rel_head_off,
                    q_pos, tile_t + dt, p.d_rel, p.rel_extent,
                    p.qk_l2_clip, p.rel_l2_clip, scale_eff, p.rel_scale, p.attn_softcap, rel_zero_oob, lane);

                const float score_log2 = score * LOG2E;
                const float new_m = max(m_val, score_log2);
                const float corr = (l_val > 0.0f) ? exp2(m_val - new_m) : 0.0f;
                const float p_w = exp2(score_log2 - new_m);
                l_val = fma(l_val, corr, p_w);
                acc0 = fma(acc0, float4(corr), v0 * p_w);
                acc1 = fma(acc1, float4(corr), v1 * p_w);
                m_val = new_m;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Speculative tree verification ancestors from k_tree/v_tree.
    if (p.has_tree != 0u && active_q) {
        ulong rem = anc_mask;
        while (rem != 0ul) {
            const uint j = uint(ctz(rem));
            rem &= (rem - 1ul);
            if (j >= p.num_nodes) {
                continue;
            }
            const uint j_anc = uint(popcount(ancestor_masks[j]));
            const uint j_depth = (j_anc > 0u) ? (j_anc - 1u) : 0u;
            const uint k_pos_j = p.kv_len + j_depth;
            if (p.window_size > 0u && k_pos_j < win_start) {
                continue;
            }

            const ulong kv_off = (ulong(j) * p.num_kv_heads + kv_head) * d_dim;
            const float4 k0 = has_c0 ? float4(*(const device half4 *)(k_tree + kv_off + c0)) : float4(0.0f);
            const float4 k1 = has_c1 ? float4(*(const device half4 *)(k_tree + kv_off + c1)) : float4(0.0f);
            const float4 v0 = has_c0 ? float4(*(const device half4 *)(v_tree + kv_off + c0)) : float4(0.0f);
            const float4 v1 = has_c1 ? float4(*(const device half4 *)(v_tree + kv_off + c1)) : float4(0.0f);

            const float score = ink_rel_gqa_score(
                q0, q1, k0, k1, q_rel_v, rel_emb, rel_head_off,
                q_pos, k_pos_j, p.d_rel, p.rel_extent,
                p.qk_l2_clip, p.rel_l2_clip, scale_eff, p.rel_scale, p.attn_softcap, rel_zero_oob, lane);

            const float score_log2 = score * LOG2E;
            const float new_m = max(m_val, score_log2);
            const float corr = (l_val > 0.0f) ? exp2(m_val - new_m) : 0.0f;
            const float p_w = exp2(score_log2 - new_m);
            l_val = fma(l_val, corr, p_w);
            acc0 = fma(acc0, float4(corr), v0 * p_w);
            acc1 = fma(acc1, float4(corr), v1 * p_w);
            m_val = new_m;
        }
    }

    if (active_q) {
        const float inv_l = (l_val > 0.0f) ? (1.0f / l_val) : 0.0f;
        float4 o0 = acc0 * inv_l;
        float4 o1 = acc1 * inv_l;
        if (p.has_gate_attn != 0u) {
            if (has_c0) {
                const float4 g0 = float4(*(const device half4 *)(gate_attn + q_row_off + c0));
                o0 = float4(
                    ink_apply_gate(o0.x, g0.x, p.gate_mode),
                    ink_apply_gate(o0.y, g0.y, p.gate_mode),
                    ink_apply_gate(o0.z, g0.z, p.gate_mode),
                    ink_apply_gate(o0.w, g0.w, p.gate_mode)
                );
            }
            if (has_c1) {
                const float4 g1 = float4(*(const device half4 *)(gate_attn + q_row_off + c1));
                o1 = float4(
                    ink_apply_gate(o1.x, g1.x, p.gate_mode),
                    ink_apply_gate(o1.y, g1.y, p.gate_mode),
                    ink_apply_gate(o1.z, g1.z, p.gate_mode),
                    ink_apply_gate(o1.w, g1.w, p.gate_mode)
                );
            }
        }
        if (has_c0) {
            *(device half4 *)(out + q_row_off + c0) = half4(o0);
        }
        if (has_c1) {
            *(device half4 *)(out + q_row_off + c1) = half4(o1);
        }
    }
}

// ===========================================================================
// 3. Inkling 2 MoE: Router, Fused SwiGLU + Softcap, and Route Softcap Epilogue
// ===========================================================================
// Inkling 2 top-k router (`num_experts <= 1024`, `top_k <= 16`) with router
// logit softcap (4.0), bias (`bias_mode == 0`: add to logit; `1`: sel-only),
// softmax, top-k selection, normalization, and `route_scale` (8.0).
//
// Grid: (M, 1, 1) threadgroups of 256 threads (zero stack arrays).
kernel void ami_inkling2_moe_route(const device uchar *logits_raw [[buffer(0)]],
                                   const device uchar *bias_raw [[buffer(1)]],
                                   device uint *topk_experts [[buffer(2)]],
                                   device float *topk_weights [[buffer(3)]],
                                   constant Inkling2MoeRouteParams &p [[buffer(4)]],
                                   uint tok [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]],
                                   uint sg_id [[simdgroup_index_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]]) {
    if (tok >= p.m || p.num_experts == 0u || p.num_experts > 1024u || p.top_k == 0u) {
        return;
    }

    threadgroup float tg_probs[1024];
    threadgroup uchar tg_picked[1024];
    threadgroup float tg_freduce[8];
    threadgroup float tg_best_s[8];
    threadgroup uint tg_best_e[8];
    threadgroup uint tg_sel_e[16];
    threadgroup float tg_sel_w[16];

    const ulong row_base = ulong(tok) * p.num_experts;
    const uint k_eff = min(p.top_k, min(p.num_experts, 16u));

    if (p.scoring_sigmoid != 0u) {
        for (uint e = tid; e < p.num_experts; e += 256u) {
            float v = (p.logits_f16 != 0u)
                ? float(((const device half *)logits_raw)[row_base + e])
                : ((const device float *)logits_raw)[row_base + e];
            v = ink_softcap(v, p.router_softcap);
            if (p.has_bias != 0u && p.bias_mode == 0u) {
                const float b = (p.bias_f16 != 0u)
                    ? float(((const device half *)bias_raw)[e])
                    : ((const device float *)bias_raw)[e];
                v += b;
            }
            tg_probs[e] = ink_sigmoid(v);
            tg_picked[e] = 0u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    } else {
        float local_max = -INFINITY;
        for (uint e = tid; e < p.num_experts; e += 256u) {
            float v = (p.logits_f16 != 0u)
                ? float(((const device half *)logits_raw)[row_base + e])
                : ((const device float *)logits_raw)[row_base + e];
            v = ink_softcap(v, p.router_softcap);
            if (p.has_bias != 0u && p.bias_mode == 0u) {
                const float b = (p.bias_f16 != 0u)
                    ? float(((const device half *)bias_raw)[e])
                    : ((const device float *)bias_raw)[e];
                v += b;
            }
            tg_probs[e] = v;
            tg_picked[e] = 0u;
            local_max = max(local_max, v);
        }
        const float sg_max = simd_max(local_max);
        if (lane == 0u) {
            tg_freduce[sg_id] = sg_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float gm = -INFINITY;
            for (uint s = 0u; s < 8u; s++) {
                gm = max(gm, tg_freduce[s]);
            }
            tg_freduce[0] = gm;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float max_v = tg_freduce[0];

        float local_sum = 0.0f;
        for (uint e = tid; e < p.num_experts; e += 256u) {
            const float ex = exp(tg_probs[e] - max_v);
            tg_probs[e] = ex;
            local_sum += ex;
        }
        const float sg_sum = simd_sum(local_sum);
        if (lane == 0u) {
            tg_freduce[sg_id] = sg_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float tot = 0.0f;
            for (uint s = 0u; s < 8u; s++) {
                tot += tg_freduce[s];
            }
            tg_freduce[0] = (tot > 0.0f) ? (1.0f / tot) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float inv_sum = tg_freduce[0];

        for (uint e = tid; e < p.num_experts; e += 256u) {
            tg_probs[e] *= inv_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint k = 0u; k < k_eff; k++) {
        float best_s = -INFINITY;
        uint best_e = 0xFFFFFFFFu;
        for (uint e = tid; e < p.num_experts; e += 256u) {
            if (tg_picked[e] == 0u) {
                float s = tg_probs[e];
                if (p.has_bias != 0u && p.bias_mode == 1u) {
                    const float b = (p.bias_f16 != 0u)
                        ? float(((const device half *)bias_raw)[e])
                        : ((const device float *)bias_raw)[e];
                    s += b;
                }
                if (s > best_s || (s == best_s && e < best_e)) {
                    best_s = s;
                    best_e = e;
                }
            }
        }
        for (ushort off = 16u; off > 0u; off >>= 1u) {
            const float os = simd_shuffle_down(best_s, off);
            const uint oe = simd_shuffle_down(best_e, off);
            if (os > best_s || (os == best_s && oe < best_e)) {
                best_s = os;
                best_e = oe;
            }
        }
        if (lane == 0u) {
            tg_best_s[sg_id] = best_s;
            tg_best_e[sg_id] = best_e;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0u) {
            float gs = -INFINITY;
            uint ge = 0xFFFFFFFFu;
            for (uint s = 0u; s < 8u; s++) {
                const float cs = tg_best_s[s];
                const uint ce = tg_best_e[s];
                if (ce != 0xFFFFFFFFu && (cs > gs || (cs == gs && ce < ge))) {
                    gs = cs;
                    ge = ce;
                }
            }
            const uint win_e = (ge < p.num_experts) ? ge : 0u;
            tg_picked[win_e] = 1u;
            tg_sel_e[k] = win_e;
            tg_sel_w[k] = tg_probs[win_e];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        float w_sum = 0.0f;
        for (uint k = 0u; k < k_eff; k++) {
            w_sum += tg_sel_w[k];
        }
        const float norm_fac = (p.norm_topk_prob != 0u && w_sum > 0.0f) ? (1.0f / w_sum) : 1.0f;
        const ulong out_base = ulong(tok) * p.top_k;
        for (uint k = 0u; k < k_eff; k++) {
            topk_experts[out_base + k] = tg_sel_e[k];
            topk_weights[out_base + k] = tg_sel_w[k] * norm_fac * p.route_scale;
        }
    }
}

// Fused SwiGLU + softcap (`25.0`/`30.0`), supporting arbitrary non-256 widths
// (e.g., `intermediate_size = 1152`) and interleaved `[M, 2*I]` or separate `[M, I]` inputs.
//
// Grid (threads): (intermediate_size, num_rows, 1).
kernel void ami_inkling2_moe_swiglu_softcap(const device half *gate_in [[buffer(0)]],
                                            const device half *up_in [[buffer(1)]],
                                            device half *out [[buffer(2)]],
                                            constant Inkling2MoeSwigluParams &p [[buffer(3)]],
                                            uint2 gid [[thread_position_in_grid]]) {
    const uint col = gid.x;
    const uint row = gid.y;
    if (row >= p.num_rows || col >= p.intermediate_size) {
        return;
    }

    float g;
    float u;
    if (p.interleaved_gate_up == 2u) {
        const ulong base = ulong(row) * (2ul * p.intermediate_size);
        g = float(gate_in[base + 2ul * col]);
        u = float(gate_in[base + 2ul * col + 1ul]);
    } else if (p.interleaved_gate_up != 0u) {
        const ulong base = ulong(row) * (2ul * p.intermediate_size);
        g = float(gate_in[base + col]);
        u = float(gate_in[base + p.intermediate_size + col]);
    } else {
        const ulong idx = ulong(row) * p.intermediate_size + col;
        g = float(gate_in[idx]);
        u = float(up_in[idx]);
    }
    const float v = (p.gate_softcap > 0.0f)
        ? (ink_softcap(g, p.gate_softcap) * ink_sigmoid(g) * ink_softcap(u, p.softcap))
        : ink_softcap(ink_silu(g) * u, p.softcap);
    out[ulong(row) * p.intermediate_size + col] = half(v);
}

// Weighted MoE expert combine + MLP softcap (25.0/30.0) + optional residual epilogue.
// Handles arbitrary hidden_dim including non-256-divisible widths (e.g. 1152).
//
// Grid (threads): (hidden_dim, m, 1).
kernel void ami_inkling2_moe_route_softcap_epilogue(const device half *expert_out [[buffer(0)]],
                                                    const device uchar *topk_weights_raw [[buffer(1)]],
                                                    const device half *residual [[buffer(2)]],
                                                    device half *out [[buffer(3)]],
                                                    constant Inkling2MoeEpilogueParams &p [[buffer(4)]],
                                                    uint2 gid [[thread_position_in_grid]]) {
    const uint col = gid.x;
    const uint tok = gid.y;
    if (tok >= p.m || col >= p.hidden_dim) {
        return;
    }

    const ulong w_base = ulong(tok) * p.top_k;
    float acc = 0.0f;
    for (uint k = 0u; k < p.top_k; k++) {
        const float w = (p.weights_f16 != 0u)
            ? float(((const device half *)topk_weights_raw)[w_base + k])
            : ((const device float *)topk_weights_raw)[w_base + k];
        const float ev = float(expert_out[(w_base + k) * p.hidden_dim + col]);
        acc = fma(w, ev, acc);
    }

    acc = ink_softcap(acc, p.mlp_softcap);
    if (p.has_residual != 0u) {
        const float r = float(residual[ulong(tok) * p.hidden_dim + col]);
        acc = fma(p.res_scale, acc, r);
    }
    out[ulong(tok) * p.hidden_dim + col] = half(acc);
}

// ===========================================================================
// 4. Inkling 2: ami_inkling2_attn_res
// ===========================================================================
// 6-layer Attention Residuals (`num_blocks <= 8`): RMSNorm each prior block state,
// score against learned cross-layer query (`[hidden_dim]` or `[M, hidden_dim]`),
// softmax across blocks, and emit weighted residual aggregation with 0 intermediate
// global memory writes and 0 stack spills.
//
// Grid: (M, 1, 1) threadgroups of 256 threads.
kernel void ami_inkling2_attn_res(const device half *block_states [[buffer(0)]],
                                  const device half *query [[buffer(1)]],
                                  const device uchar *norm_w [[buffer(2)]],
                                  device half *out [[buffer(3)]],
                                  constant Inkling2AttnResParams &p [[buffer(4)]],
                                  uint tok [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]],
                                  uint sg_id [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
    if (tok >= p.m || p.hidden_dim == 0u || p.num_blocks == 0u || p.num_blocks > 8u) {
        return;
    }

    threadgroup float tg_logits[8];
    threadgroup float tg_alpha[8];
    threadgroup float tg_inv_rms_b[8];
    threadgroup float tg_sum_sq[8];
    threadgroup float tg_dot[8];

    const uint d = p.hidden_dim;
    const ulong q_base = (p.query_per_token != 0u) ? (ulong(tok) * d) : 0ul;
    const ulong tok_row_off = ulong(tok) * d;

    for (uint b = 0u; b < p.num_blocks; b++) {
        const uint phys_b = p.block_slots[b];
        const device half *vb = block_states + ulong(phys_b) * p.block_stride_elems + tok_row_off;

        float sum_sq = 0.0f;
        float dot_raw = 0.0f;
        for (uint c = tid; c < d; c += 256u) {
            const float v = float(vb[c]);
            const float qv = float(query[q_base + c]);
            const float nw = (p.has_norm_w != 0u) ? ink_load_norm_w(norm_w, c, NORM_W_BF16) : 1.0f;
            sum_sq = fma(v, v, sum_sq);
            dot_raw = fma(v * nw, qv, dot_raw);
        }
        const float sg_ss = simd_sum(sum_sq);
        const float sg_dt = simd_sum(dot_raw);
        if (lane == 0u) {
            tg_sum_sq[sg_id] = sg_ss;
            tg_dot[sg_id] = sg_dt;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0u) {
            float tot_ss = 0.0f;
            float tot_dt = 0.0f;
            for (uint s = 0u; s < 8u; s++) {
                tot_ss += tg_sum_sq[s];
                tot_dt += tg_dot[s];
            }
            const float inv_rms = (p.apply_rms_norm != 0u || p.normalize_values != 0u)
                ? rsqrt(tot_ss / float(d) + p.rms_eps)
                : 1.0f;
            tg_inv_rms_b[b] = inv_rms;
            tg_logits[b] = ((p.apply_rms_norm != 0u) ? (tot_dt * inv_rms) : tot_dt) * p.scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        float max_l = -INFINITY;
        for (uint b = 0u; b < p.num_blocks; b++) {
            max_l = max(max_l, tg_logits[b]);
        }
        float exp_sum = 0.0f;
        for (uint b = 0u; b < p.num_blocks; b++) {
            const float ex = exp(tg_logits[b] - max_l);
            tg_alpha[b] = ex;
            exp_sum += ex;
        }
        const float inv_e = (exp_sum > 0.0f) ? (1.0f / exp_sum) : 0.0f;
        for (uint b = 0u; b < p.num_blocks; b++) {
            tg_alpha[b] *= inv_e;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint c = tid; c < d; c += 256u) {
        float acc = 0.0f;
        for (uint b = 0u; b < p.num_blocks; b++) {
            const uint phys_b = p.block_slots[b];
            const float raw_v = float(block_states[ulong(phys_b) * p.block_stride_elems + tok_row_off + c]);
            const float v = (p.normalize_values != 0u) ? (raw_v * tg_inv_rms_b[b]) : raw_v;
            acc = fma(tg_alpha[b], v, acc);
        }
        out[tok_row_off + c] = half(acc);
    }
}

// ===========================================================================
// 5. Hy4: IndexPool Compression & 2-Stage Sparse Token Selector
// ===========================================================================
// Compresses key indexer vectors `[kv_len, num_k_heads, idx_dim]` by
// `compress_ratio` (`= 4`) via mean (`pool_mode == 0`) or max (`pool_mode == 1`).
//
// Grid (threads): (num_k_heads * idx_dim, ceil(kv_len / compress_ratio), 1).
kernel void ami_hy4_index_pool_compress(const device half *k_fine [[buffer(0)]],
                                        device half *k_pool [[buffer(1)]],
                                        constant Hy4IndexCompressParams &p [[buffer(2)]],
                                        uint2 gid [[thread_position_in_grid]]) {
    const uint stride = p.num_k_heads * p.idx_dim;
    const uint ch = gid.x;
    const uint grp = gid.y;
    const uint num_groups = (p.kv_len + p.compress_ratio - 1u) / p.compress_ratio;
    if (ch >= stride || grp >= num_groups || p.compress_ratio == 0u) {
        return;
    }

    const uint t0 = grp * p.compress_ratio;
    const uint t1 = min(t0 + p.compress_ratio, p.kv_len);
    const uint cnt = t1 - t0;

    if (p.pool_mode == 0u) {
        float acc = 0.0f;
        for (uint t = t0; t < t1; t++) {
            acc += float(k_fine[ulong(t) * stride + ch]);
        }
        k_pool[ulong(grp) * stride + ch] = half(acc / float(cnt));
    } else {
        float mx = -INFINITY;
        for (uint t = t0; t < t1; t++) {
            mx = max(mx, float(k_fine[ulong(t) * stride + ch]));
        }
        k_pool[ulong(grp) * stride + ch] = half(mx);
    }
}

static inline float hy4_score_key_simd(const device half *q_idx,
                                       ulong q_node_base,
                                       const device half *head_weights,
                                       ulong w_node_base,
                                       const device half *k_vec_base,
                                       uint num_q_heads,
                                       uint num_k_heads,
                                       uint q_per_k,
                                       uint idx_dim,
                                       uint score_mode,
                                       uint has_head_weights,
                                       float scale,
                                       uint lane) {
    float agg_score = (score_mode == 2u) ? -INFINITY : 0.0f;
    for (uint h = 0u; h < num_q_heads; h++) {
        const uint kh = (num_k_heads > 1u) ? (h / q_per_k) : 0u;
        const device half *qh_ptr = q_idx + q_node_base + ulong(h) * idx_dim;
        const device half *kh_ptr = k_vec_base + ulong(kh) * idx_dim;
        float dot_acc = 0.0f;
        for (uint d = lane; d < idx_dim; d += 32u) {
            dot_acc = fma(float(qh_ptr[d]), float(kh_ptr[d]), dot_acc);
        }
        const float head_dot = simd_sum(dot_acc);
        if (score_mode == 0u) {
            const float hw = (has_head_weights != 0u) ? float(head_weights[w_node_base + h]) : 1.0f;
            agg_score += hw * max(0.0f, head_dot) * scale;
        } else if (score_mode == 1u) {
            const float hw = (has_head_weights != 0u) ? float(head_weights[w_node_base + h]) : 1.0f;
            agg_score += hw * head_dot * scale;
        } else {
            agg_score = max(agg_score, head_dot * scale);
        }
    }
    return agg_score;
}

// Two-stage IndexPool sparse selector (`compress_ratio = 4` -> `group_topk` ->
// `candidate_topk`, supporting both tiny dims and full `512` / `2048` dims).
//
// Grid: (num_nodes, 1, 1) threadgroups of 256 threads (8 simdgroups).
kernel void ami_hy4_index_pool_topk(const device half *q_idx [[buffer(0)]],
                                    const device half *head_weights [[buffer(1)]],
                                    const device half *k_fine [[buffer(2)]],
                                    const device half *k_pool [[buffer(3)]],
                                    device float *scratch_group_scores [[buffer(4)]],
                                    device float *scratch_cand_scores [[buffer(5)]],
                                    device uint *scratch_sel_groups [[buffer(6)]],
                                    device uint *scratch_cand_ids [[buffer(7)]],
                                    device uint *selected_tokens [[buffer(8)]],
                                    device uint *selected_counts [[buffer(9)]],
                                    constant Hy4IndexPoolTopkParams &p [[buffer(10)]],
                                    uint node [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]],
                                    uint sg_id [[simdgroup_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]]) {
    if (node >= p.num_nodes || p.num_q_heads == 0u || p.idx_dim == 0u || p.compress_ratio == 0u) {
        return;
    }

    threadgroup atomic_uint tg_hist[16];
    threadgroup uint tg_scan[8];
    threadgroup uint tg_win_nib;
    threadgroup uint tg_scalar;

    const ulong out_base = ulong(node) * p.candidate_topk;
    const uint raw_kv = p.kv_len;
    const uint eff_kv_len = (p.causal_q_start_pos != 0xFFFFFFFFu)
        ? min(raw_kv, p.causal_q_start_pos + node + 1u)
        : raw_kv;
    const uint sparse_end = (p.swa_window > 0u && eff_kv_len > p.swa_window)
        ? (eff_kv_len - p.swa_window)
        : ((p.swa_window > 0u) ? 0u : eff_kv_len);

    if (sparse_end == 0u || p.candidate_topk == 0u) {
        for (uint i = tid; i < p.candidate_topk; i += 256u) {
            selected_tokens[out_base + i] = 0xFFFFFFFFu;
        }
        if (tid == 0u) {
            selected_counts[node] = 0u;
        }
        return;
    }

    const uint k_cand_cap = min(sparse_end, p.candidate_topk);
    if (sparse_end <= p.candidate_topk) {
        for (uint i = tid; i < p.candidate_topk; i += 256u) {
            selected_tokens[out_base + i] = (i < sparse_end) ? i : 0xFFFFFFFFu;
        }
        if (tid == 0u) {
            selected_counts[node] = sparse_end;
        }
        return;
    }

    const uint num_groups = (sparse_end + p.compress_ratio - 1u) / p.compress_ratio;
    const uint k_stride = p.num_k_heads * p.idx_dim;
    const ulong q_node_base = ulong(node) * p.num_q_heads * p.idx_dim;
    const ulong w_node_base = ulong(node) * p.num_q_heads;
    const uint q_per_k = (p.num_k_heads > 0u) ? max(p.num_q_heads / p.num_k_heads, 1u) : p.num_q_heads;

    device float *g_scores = scratch_group_scores + ulong(node) * p.max_groups_stride;
    device uint *sel_groups = scratch_sel_groups + ulong(node) * p.group_topk;
    device float *c_scores = scratch_cand_scores + ulong(node) * p.max_cand_stride;
    device uint *c_ids = scratch_cand_ids + ulong(node) * p.max_cand_stride;
    device uint *out_ptr = selected_tokens + out_base;

    uint num_cand_tokens = 0u;
    const bool skip_group_stage = (p.group_topk == 0u || num_groups <= p.group_topk);

    if (!skip_group_stage) {
        // Stage 1: Score every compressed group across 8 simdgroups in parallel.
        for (uint g = sg_id; g < num_groups; g += 8u) {
            const float agg_score = hy4_score_key_simd(
                q_idx, q_node_base, head_weights, w_node_base,
                k_pool + ulong(g) * k_stride,
                p.num_q_heads, p.num_k_heads, q_per_k, p.idx_dim,
                p.score_mode, p.has_head_weights, p.scale, lane);
            if (lane == 0u) {
                g_scores[g] = agg_score;
            }
        }
        threadgroup_barrier(mem_flags::mem_device);

        const uint g_keep = min(num_groups, p.group_topk);
        ink_tg_radix_topk(g_scores, sel_groups, false, num_groups, g_keep, sel_groups,
                          tid, sg_id, lane, tg_hist, tg_scan, &tg_win_nib, &tg_scalar);
        threadgroup_barrier(mem_flags::mem_device);

        // Expand selected groups into candidate fine tokens (bounded by sparse_end).
        for (uint gi = 0u; gi < g_keep; gi++) {
            const uint g = sel_groups[gi];
            const uint t0 = g * p.compress_ratio;
            const uint t1 = min(t0 + p.compress_ratio, sparse_end);
            const uint cnt = (t1 > t0) ? (t1 - t0) : 0u;
            for (uint r = tid; r < cnt; r += 256u) {
                if ((num_cand_tokens + r) < p.max_cand_stride) {
                    c_ids[num_cand_tokens + r] = t0 + r;
                }
            }
            num_cand_tokens += cnt;
        }
        num_cand_tokens = min(num_cand_tokens, p.max_cand_stride);
        threadgroup_barrier(mem_flags::mem_device);
    } else {
        num_cand_tokens = min(sparse_end, p.max_cand_stride);
        for (uint i = tid; i < num_cand_tokens; i += 256u) {
            c_ids[i] = i;
        }
        threadgroup_barrier(mem_flags::mem_device);
    }

    if (num_cand_tokens <= k_cand_cap) {
        for (uint i = tid; i < p.candidate_topk; i += 256u) {
            out_ptr[i] = (i < num_cand_tokens) ? c_ids[i] : 0xFFFFFFFFu;
        }
        if (tid == 0u) {
            selected_counts[node] = num_cand_tokens;
        }
        return;
    }

    // Stage 2: Score every candidate fine token across 8 simdgroups in parallel.
    for (uint ci = sg_id; ci < num_cand_tokens; ci += 8u) {
        const uint t = c_ids[ci];
        const float agg_score = hy4_score_key_simd(
            q_idx, q_node_base, head_weights, w_node_base,
            k_fine + ulong(t) * k_stride,
            p.num_q_heads, p.num_k_heads, q_per_k, p.idx_dim,
            p.score_mode, p.has_head_weights, p.scale, lane);
        if (lane == 0u) {
            c_scores[ci] = agg_score;
        }
    }
    threadgroup_barrier(mem_flags::mem_device);

    ink_tg_radix_topk(c_scores, c_ids, true, num_cand_tokens, k_cand_cap, out_ptr,
                      tid, sg_id, lane, tg_hist, tg_scan, &tg_win_nib, &tg_scalar);
    threadgroup_barrier(mem_flags::mem_device);

    for (uint i = k_cand_cap + tid; i < p.candidate_topk; i += 256u) {
        out_ptr[i] = 0xFFFFFFFFu;
    }
    if (tid == 0u) {
        selected_counts[node] = k_cand_cap;
    }
}

// ===========================================================================
// 6. Hy4: ami_hy4_latent_moe_route
// ===========================================================================
// Sigmoid router (`1 / (1 + exp(-x))`) + `e_score_correction_bias` selection-only
// bias + top-k normalization * `routed_scaling_factor` (3.998).
//
// Grid: (M, 1, 1) threadgroups of 256 threads (zero stack arrays).
kernel void ami_hy4_latent_moe_route(const device uchar *logits_raw [[buffer(0)]],
                                     const device uchar *bias_raw [[buffer(1)]],
                                     device uint *topk_experts [[buffer(2)]],
                                     device float *topk_weights [[buffer(3)]],
                                     constant Hy4LatentMoeRouteParams &p [[buffer(4)]],
                                     uint tok [[threadgroup_position_in_grid]],
                                     uint tid [[thread_index_in_threadgroup]],
                                     uint sg_id [[simdgroup_index_in_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]]) {
    if (tok >= p.m || p.num_experts == 0u || p.num_experts > 1024u || p.top_k == 0u) {
        return;
    }

    threadgroup float tg_scores[1024];
    threadgroup uchar tg_picked[1024];
    threadgroup float tg_best_s[8];
    threadgroup uint tg_best_e[8];
    threadgroup uint tg_sel_e[16];
    threadgroup float tg_sel_w[16];

    const ulong row_base = ulong(tok) * p.num_experts;
    const uint k_eff = min(p.top_k, min(p.num_experts, 16u));

    for (uint e = tid; e < p.num_experts; e += 256u) {
        const float x = (p.logits_f16 != 0u)
            ? float(((const device half *)logits_raw)[row_base + e])
            : ((const device float *)logits_raw)[row_base + e];
        tg_scores[e] = ink_sigmoid(x);
        tg_picked[e] = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 0u; k < k_eff; k++) {
        float best_s = -INFINITY;
        uint best_e = 0xFFFFFFFFu;
        for (uint e = tid; e < p.num_experts; e += 256u) {
            if (tg_picked[e] == 0u) {
                float s = tg_scores[e];
                if (p.has_bias != 0u) {
                    const float b = (p.bias_f16 != 0u)
                        ? float(((const device half *)bias_raw)[e])
                        : ((const device float *)bias_raw)[e];
                    s += b;
                }
                if (s > best_s || (s == best_s && e < best_e)) {
                    best_s = s;
                    best_e = e;
                }
            }
        }
        for (ushort off = 16u; off > 0u; off >>= 1u) {
            const float os = simd_shuffle_down(best_s, off);
            const uint oe = simd_shuffle_down(best_e, off);
            if (os > best_s || (os == best_s && oe < best_e)) {
                best_s = os;
                best_e = oe;
            }
        }
        if (lane == 0u) {
            tg_best_s[sg_id] = best_s;
            tg_best_e[sg_id] = best_e;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0u) {
            float gs = -INFINITY;
            uint ge = 0xFFFFFFFFu;
            for (uint s = 0u; s < 8u; s++) {
                const float cs = tg_best_s[s];
                const uint ce = tg_best_e[s];
                if (ce != 0xFFFFFFFFu && (cs > gs || (cs == gs && ce < ge))) {
                    gs = cs;
                    ge = ce;
                }
            }
            const uint win_e = (ge < p.num_experts) ? ge : 0u;
            tg_picked[win_e] = 1u;
            tg_sel_e[k] = win_e;
            tg_sel_w[k] = tg_scores[win_e];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        float w_sum = 0.0f;
        for (uint k = 0u; k < k_eff; k++) {
            w_sum += tg_sel_w[k];
        }
        const float norm_fac = (p.norm_topk_prob != 0u)
            ? (1.0f / (w_sum + p.eps))
            : 1.0f;
        const ulong out_base = ulong(tok) * p.top_k;
        for (uint k = 0u; k < k_eff; k++) {
            topk_experts[out_base + k] = tg_sel_e[k];
            topk_weights[out_base + k] = tg_sel_w[k] * norm_fac * p.routed_scaling_factor;
        }
    }
}

// ===========================================================================
// 7. Hy4: ami_hy4_swiglu_clamped_linear_gate
// ===========================================================================
// SwiGLU with `swiglu_limit` (`[-limit, +limit]` gate/up clamp, `limit = 10.0`)
// plus optional elementwise `linear_gate`. Supports arbitrary widths (incl. 1152).
//
// Grid (threads): (intermediate_size, num_rows, 1).
kernel void ami_hy4_swiglu_clamped_linear_gate(const device half *gate_in [[buffer(0)]],
                                               const device half *up_in [[buffer(1)]],
                                               const device half *linear_gate [[buffer(2)]],
                                               device half *out [[buffer(3)]],
                                               constant Hy4SwigluClampedParams &p [[buffer(4)]],
                                               uint2 gid [[thread_position_in_grid]]) {
    const uint col = gid.x;
    const uint row = gid.y;
    if (row >= p.num_rows || col >= p.intermediate_size) {
        return;
    }

    const ulong idx = ulong(row) * p.intermediate_size + col;
    float g;
    float u;
    if (p.interleaved_gate_up != 0u) {
        const ulong base = ulong(row) * (2ul * p.intermediate_size);
        g = float(gate_in[base + col]);
        u = float(gate_in[base + p.intermediate_size + col]);
    } else {
        g = float(gate_in[idx]);
        u = float(up_in[idx]);
    }

    if (p.swiglu_limit > 0.0f && (p.clamp_mode == 0u || p.clamp_mode == 2u)) {
        g = clamp(g, -p.swiglu_limit, p.swiglu_limit);
        u = clamp(u, -p.swiglu_limit, p.swiglu_limit);
    }
    float v = ink_silu(g) * u;
    if (p.swiglu_limit > 0.0f && (p.clamp_mode == 1u || p.clamp_mode == 2u)) {
        v = clamp(v, -p.swiglu_limit, p.swiglu_limit);
    }
    if (p.has_linear_gate != 0u) {
        v = ink_apply_gate(v, float(linear_gate[idx]), p.linear_gate_mode);
    }
    out[idx] = half(v);
}

// ===========================================================================
// 8. Hy4: ami_hy4_combine_sandwich_norm
// ===========================================================================
// Fused LatentMoE weighted combine (`combine_output_norm`) + optional shared
// branch + residual + `sandwich_norm`/`post_proj_norm` in a single TG pass.
//
// Grid: (M, 1, 1) threadgroups of 256 threads.
static inline float hy4_eval_comb_channel(const device half *expert_or_x,
                                          const device uchar *topk_weights_raw,
                                          const device half *shared_out,
                                          ulong tok_off,
                                          ulong w_base,
                                          uint d,
                                          uint c,
                                          uint top_k,
                                          uint weights_f16,
                                          uint has_shared) {
    float val = 0.0f;
    if (top_k > 0u) {
        for (uint k = 0u; k < top_k; k++) {
            const float w = (weights_f16 != 0u)
                ? float(((const device half *)topk_weights_raw)[w_base + k])
                : ((const device float *)topk_weights_raw)[w_base + k];
            val = fma(w, float(expert_or_x[(w_base + k) * d + c]), val);
        }
    } else {
        val = float(expert_or_x[tok_off + c]);
    }
    if (has_shared != 0u) {
        val += float(shared_out[tok_off + c]);
    }
    return val;
}

kernel void ami_hy4_combine_sandwich_norm(const device half *expert_or_x [[buffer(0)]],
                                          const device uchar *topk_weights_raw [[buffer(1)]],
                                          const device half *shared_out [[buffer(2)]],
                                          const device half *residual [[buffer(3)]],
                                          const device uchar *combine_norm_w [[buffer(4)]],
                                          const device uchar *post_norm_w [[buffer(5)]],
                                          device half *out [[buffer(6)]],
                                          constant Hy4CombineSandwichParams &p [[buffer(7)]],
                                          uint tok [[threadgroup_position_in_grid]],
                                          uint tid [[thread_index_in_threadgroup]],
                                          uint sg_id [[simdgroup_index_in_threadgroup]],
                                          uint lane [[thread_index_in_simdgroup]]) {
    if (tok >= p.m || p.hidden_dim == 0u) {
        return;
    }

    threadgroup float tg_comb[2048];
    threadgroup float tg_reduce[8];
    threadgroup float tg_inv_rms;

    const uint d = p.hidden_dim;
    const ulong tok_off = ulong(tok) * d;
    const ulong w_base = ulong(tok) * p.top_k;
    const bool cache_in_tg = (d <= 2048u);

    float sum_sq = 0.0f;
    for (uint c = tid; c < d; c += 256u) {
        const float val = hy4_eval_comb_channel(
            expert_or_x, topk_weights_raw, shared_out, tok_off, w_base, d, c,
            p.top_k, p.weights_f16, p.has_shared);
        if (cache_in_tg) {
            tg_comb[c] = val;
        }
        sum_sq = fma(val, val, sum_sq);
    }

    if (p.apply_combine_norm != 0u) {
        const float sg_ss = simd_sum(sum_sq);
        if (lane == 0u) {
            tg_reduce[sg_id] = sg_ss;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float tot = 0.0f;
            for (uint s = 0u; s < 8u; s++) {
                tot += tg_reduce[s];
            }
            tg_inv_rms = rsqrt(tot / float(d) + p.rms_eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv_rms1 = (p.apply_combine_norm != 0u) ? tg_inv_rms : 1.0f;

    float post_sum_sq = 0.0f;
    for (uint c = tid; c < d; c += 256u) {
        float val = cache_in_tg
            ? tg_comb[c]
            : hy4_eval_comb_channel(
                expert_or_x, topk_weights_raw, shared_out, tok_off, w_base, d, c,
                p.top_k, p.weights_f16, p.has_shared);
        if (p.apply_combine_norm != 0u) {
            val = (val * inv_rms1) * ink_load_norm_w(combine_norm_w, c, NORM_W_BF16);
        }
        if (p.has_residual != 0u) {
            val = fma(p.res_scale, val, float(residual[tok_off + c]));
        }
        if (p.apply_post_norm != 0u) {
            if (cache_in_tg) {
                tg_comb[c] = val;
            }
            post_sum_sq = fma(val, val, post_sum_sq);
        } else {
            out[tok_off + c] = half(val);
        }
    }

    if (p.apply_post_norm != 0u) {
        const float sg_ss2 = simd_sum(post_sum_sq);
        if (lane == 0u) {
            tg_reduce[sg_id] = sg_ss2;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float tot = 0.0f;
            for (uint s = 0u; s < 8u; s++) {
                tot += tg_reduce[s];
            }
            tg_inv_rms = rsqrt(tot / float(d) + p.rms_eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float inv_rms2 = tg_inv_rms;

        for (uint c = tid; c < d; c += 256u) {
            float val;
            if (cache_in_tg) {
                val = tg_comb[c];
            } else {
                val = hy4_eval_comb_channel(
                    expert_or_x, topk_weights_raw, shared_out, tok_off, w_base, d, c,
                    p.top_k, p.weights_f16, p.has_shared);
                if (p.apply_combine_norm != 0u) {
                    val = (val * inv_rms1) * ink_load_norm_w(combine_norm_w, c, NORM_W_BF16);
                }
                if (p.has_residual != 0u) {
                    val = fma(p.res_scale, val, float(residual[tok_off + c]));
                }
            }
            out[tok_off + c] = half((val * inv_rms2) * ink_load_norm_w(post_norm_w, c, NORM_W_BF16));
        }
    }
}

// ===========================================================================
// 9. Hy4: ami_hy4_mla_sink_attn
// ===========================================================================
// Sparse MLA decode / verify / prefill attention over `IndexCache` / `IndexPool`
// selected token indices + optional SWA window `[swa_lo, eff_kv_len)` + optional
// `kv_tree` ancestors + `learnable_sink_param` in online softmax + optional
// elementwise `linear_gate`. Zero stack arrays (`kv_lora_rank <= 2048`).
//
// Grid: (num_q_heads, num_nodes, 1) threadgroups of 256 threads (8 simdgroups).
static inline void hy4_mla_step(const device half *kl_row,
                                const device half *kr_row,
                                float2 q0,
                                float2 q1,
                                float2 q2,
                                float2 q3,
                                float q_pe_reg,
                                uint idx0,
                                uint idx1,
                                uint idx2,
                                uint idx3,
                                bool v0,
                                bool v1,
                                bool v2,
                                bool v3,
                                uint qk_rope_dim,
                                float scale,
                                float attn_softcap,
                                uint tid,
                                uint sg_id,
                                uint lane,
                                threadgroup float *tg_reduce,
                                thread float &m_val,
                                thread float &l_val,
                                thread float2 &acc0,
                                thread float2 &acc1,
                                thread float2 &acc2,
                                thread float2 &acc3) {
    constexpr float LOG2E = 1.4426950408889634f;
    const float2 k0 = v0 ? float2(float(kl_row[idx0 * 2u]), float(kl_row[idx0 * 2u + 1u])) : float2(0.0f);
    const float2 k1 = v1 ? float2(float(kl_row[idx1 * 2u]), float(kl_row[idx1 * 2u + 1u])) : float2(0.0f);
    const float2 k2 = v2 ? float2(float(kl_row[idx2 * 2u]), float(kl_row[idx2 * 2u + 1u])) : float2(0.0f);
    const float2 k3 = v3 ? float2(float(kl_row[idx3 * 2u]), float(kl_row[idx3 * 2u + 1u])) : float2(0.0f);

    float dot_p = dot(q0, k0) + dot(q1, k1) + dot(q2, k2) + dot(q3, k3);
    if (qk_rope_dim > 0u && tid < qk_rope_dim) {
        dot_p = fma(q_pe_reg, float(kr_row[tid]), dot_p);
    }
    const float sg_sum = simd_sum(dot_p);
    if (lane == 0u) {
        tg_reduce[sg_id] = sg_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float tot_dot = ((tg_reduce[0] + tg_reduce[1]) + (tg_reduce[2] + tg_reduce[3])) +
        ((tg_reduce[4] + tg_reduce[5]) + (tg_reduce[6] + tg_reduce[7]));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float score = ink_softcap(tot_dot * scale, attn_softcap) * LOG2E;
    const float new_m = max(m_val, score);
    const float corr = (l_val > 0.0f) ? exp2(m_val - new_m) : 0.0f;
    const float p_w = exp2(score - new_m);
    l_val = fma(l_val, corr, p_w);
    acc0 = fma(acc0, float2(corr), k0 * p_w);
    acc1 = fma(acc1, float2(corr), k1 * p_w);
    acc2 = fma(acc2, float2(corr), k2 * p_w);
    acc3 = fma(acc3, float2(corr), k3 * p_w);
    m_val = new_m;
}

kernel void ami_hy4_mla_sink_attn(const device half *q_nope [[buffer(0)]],
                                  const device half *q_pe [[buffer(1)]],
                                  const device half *kv_latent [[buffer(2)]],
                                  const device half *kv_rope [[buffer(3)]],
                                  const device uint *selected_tokens [[buffer(4)]],
                                  const device uint *selected_counts [[buffer(5)]],
                                  const device half *kv_tree_latent [[buffer(6)]],
                                  const device half *kv_tree_rope [[buffer(7)]],
                                  const device ulong *ancestor_masks [[buffer(8)]],
                                  const device float *sink_params [[buffer(9)]],
                                  const device half *linear_gate [[buffer(10)]],
                                  device half *out [[buffer(11)]],
                                  constant Hy4MlaSinkAttnParams &p [[buffer(12)]],
                                  uint2 tg_pos [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]],
                                  uint sg_id [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
    constexpr float LOG2E = 1.4426950408889634f;
    const uint q_head = tg_pos.x;
    const uint node = tg_pos.y;
    const uint d_lat = (HY4_KV_LORA_RANK > 0u) ? HY4_KV_LORA_RANK : p.kv_lora_rank;
    if (q_head >= p.num_q_heads || node >= p.num_nodes || d_lat == 0u || d_lat > 2048u) {
        return;
    }

    const uint num_pairs = d_lat >> 1; // <= 1024 -> at most 4 float2 slots/thread
    const ulong q_nope_off = (ulong(node) * p.num_q_heads + q_head) * d_lat;
    const ulong q_pe_off = (ulong(node) * p.num_q_heads + q_head) * p.qk_rope_dim;

    const uint idx0 = tid;
    const uint idx1 = tid + 256u;
    const uint idx2 = tid + 512u;
    const uint idx3 = tid + 768u;
    const bool v0 = (idx0 < num_pairs);
    const bool v1 = (idx1 < num_pairs);
    const bool v2 = (idx2 < num_pairs);
    const bool v3 = (idx3 < num_pairs);

    const float2 q0 = v0 ? float2(float(q_nope[q_nope_off + idx0 * 2u]), float(q_nope[q_nope_off + idx0 * 2u + 1u])) : float2(0.0f);
    const float2 q1 = v1 ? float2(float(q_nope[q_nope_off + idx1 * 2u]), float(q_nope[q_nope_off + idx1 * 2u + 1u])) : float2(0.0f);
    const float2 q2 = v2 ? float2(float(q_nope[q_nope_off + idx2 * 2u]), float(q_nope[q_nope_off + idx2 * 2u + 1u])) : float2(0.0f);
    const float2 q3 = v3 ? float2(float(q_nope[q_nope_off + idx3 * 2u]), float(q_nope[q_nope_off + idx3 * 2u + 1u])) : float2(0.0f);
    const float q_pe_reg = (p.qk_rope_dim > 0u && tid < p.qk_rope_dim)
        ? float(q_pe[q_pe_off + tid])
        : 0.0f;

    float2 acc0 = float2(0.0f);
    float2 acc1 = float2(0.0f);
    float2 acc2 = float2(0.0f);
    float2 acc3 = float2(0.0f);

    float m_val = (p.has_sink != 0u) ? (sink_params[q_head] * LOG2E) : -INFINITY;
    float l_val = (p.has_sink != 0u) ? 1.0f : 0.0f;

    const uint eff_kv_len = (p.has_tree != 0u)
        ? p.kv_len
        : ((p.causal_q_start_pos != 0xFFFFFFFFu)
            ? min(p.kv_len, p.causal_q_start_pos + node + 1u)
            : p.kv_len);
    const uint swa_lo = (p.swa_window > 0u)
        ? ((eff_kv_len > p.swa_window) ? (eff_kv_len - p.swa_window) : 0u)
        : eff_kv_len;

    threadgroup float tg_reduce[8];

    // Pass 1: Selected sparse tokens (< swa_lo if SWA is active, else < eff_kv_len).
    const uint sel_cnt = (p.candidate_topk > 0u) ? min(selected_counts[node], p.candidate_topk) : 0u;
    const ulong sel_base = ulong(node) * p.candidate_topk;
    const uint sparse_cap = (p.swa_window > 0u) ? swa_lo : eff_kv_len;

    for (uint si = 0u; si < sel_cnt; si++) {
        const uint t = selected_tokens[sel_base + si];
        if (t == 0xFFFFFFFFu || t >= sparse_cap) {
            continue;
        }
        hy4_mla_step(
            kv_latent + ulong(t) * d_lat, kv_rope + ulong(t) * p.qk_rope_dim,
            q0, q1, q2, q3, q_pe_reg, idx0, idx1, idx2, idx3, v0, v1, v2, v3,
            p.qk_rope_dim, p.scale, p.attn_softcap, tid, sg_id, lane, tg_reduce,
            m_val, l_val, acc0, acc1, acc2, acc3);
    }

    // Pass 2: Dense SWA recent window [swa_lo, eff_kv_len).
    for (uint t = swa_lo; t < eff_kv_len; t++) {
        hy4_mla_step(
            kv_latent + ulong(t) * d_lat, kv_rope + ulong(t) * p.qk_rope_dim,
            q0, q1, q2, q3, q_pe_reg, idx0, idx1, idx2, idx3, v0, v1, v2, v3,
            p.qk_rope_dim, p.scale, p.attn_softcap, tid, sg_id, lane, tg_reduce,
            m_val, l_val, acc0, acc1, acc2, acc3);
    }

    // Pass 3: Speculative tree / block-verify ancestors from kv_tree_latent/rope.
    if (p.has_tree != 0u) {
        ulong rem = ancestor_masks[node];
        while (rem != 0ul) {
            const uint j = uint(ctz(rem));
            rem &= (rem - 1ul);
            if (j >= p.num_nodes) {
                continue;
            }
            hy4_mla_step(
                kv_tree_latent + ulong(j) * d_lat, kv_tree_rope + ulong(j) * p.qk_rope_dim,
                q0, q1, q2, q3, q_pe_reg, idx0, idx1, idx2, idx3, v0, v1, v2, v3,
                p.qk_rope_dim, p.scale, p.attn_softcap, tid, sg_id, lane, tg_reduce,
                m_val, l_val, acc0, acc1, acc2, acc3);
        }
    }

    const float inv_l = (l_val > 0.0f) ? (1.0f / l_val) : 0.0f;
    float2 o0 = acc0 * inv_l;
    float2 o1 = acc1 * inv_l;
    float2 o2 = acc2 * inv_l;
    float2 o3 = acc3 * inv_l;

    if (p.has_linear_gate != 0u) {
        if (v0) {
            const float2 g = float2(float(linear_gate[q_nope_off + idx0 * 2u]), float(linear_gate[q_nope_off + idx0 * 2u + 1u]));
            o0 = float2(ink_apply_gate(o0.x, g.x, p.linear_gate_mode), ink_apply_gate(o0.y, g.y, p.linear_gate_mode));
        }
        if (v1) {
            const float2 g = float2(float(linear_gate[q_nope_off + idx1 * 2u]), float(linear_gate[q_nope_off + idx1 * 2u + 1u]));
            o1 = float2(ink_apply_gate(o1.x, g.x, p.linear_gate_mode), ink_apply_gate(o1.y, g.y, p.linear_gate_mode));
        }
        if (v2) {
            const float2 g = float2(float(linear_gate[q_nope_off + idx2 * 2u]), float(linear_gate[q_nope_off + idx2 * 2u + 1u]));
            o2 = float2(ink_apply_gate(o2.x, g.x, p.linear_gate_mode), ink_apply_gate(o2.y, g.y, p.linear_gate_mode));
        }
        if (v3) {
            const float2 g = float2(float(linear_gate[q_nope_off + idx3 * 2u]), float(linear_gate[q_nope_off + idx3 * 2u + 1u]));
            o3 = float2(ink_apply_gate(o3.x, g.x, p.linear_gate_mode), ink_apply_gate(o3.y, g.y, p.linear_gate_mode));
        }
    }

    if (v0) {
        out[q_nope_off + idx0 * 2u] = half(o0.x);
        out[q_nope_off + idx0 * 2u + 1u] = half(o0.y);
    }
    if (v1) {
        out[q_nope_off + idx1 * 2u] = half(o1.x);
        out[q_nope_off + idx1 * 2u + 1u] = half(o1.y);
    }
    if (v2) {
        out[q_nope_off + idx2 * 2u] = half(o2.x);
        out[q_nope_off + idx2 * 2u + 1u] = half(o2.y);
    }
    if (v3) {
        out[q_nope_off + idx3 * 2u] = half(o3.x);
        out[q_nope_off + idx3 * 2u + 1u] = half(o3.y);
    }
}

// ===========================================================================
// 9. M8.25 (G29) Zero-Spill Inkling 2 Sublayer & Indirect Softcap MoE Kernels
// ===========================================================================

struct Inkling2ScaledL2ClipParams {
    uint m;
    uint num_heads;
    uint head_dim;
    uint has_scale_vec;
    uint use_log_scaling;
    uint has_positions;
    uint base_pos;
    uint log_floor;
    float threshold;
    float log_alpha;
    uint _pad0;
    uint _pad1;
};
static_assert(sizeof(Inkling2ScaledL2ClipParams) == 48, "Inkling2ScaledL2ClipParams must be 48 bytes");

struct Inkling2UnitRmsNormParams {
    uint m;
    uint num_heads;
    uint head_dim;
    float rms_eps;
};
static_assert(sizeof(Inkling2UnitRmsNormParams) == 16, "Inkling2UnitRmsNormParams must be 16 bytes");

struct Inkling2F16ToF32Params {
    uint count;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};
static_assert(sizeof(Inkling2F16ToF32Params) == 16, "Inkling2F16ToF32Params must be 16 bytes");

struct Inkling2AttnPostProjParams {
    uint m;
    uint num_q_heads;
    uint num_kv_heads;
    uint head_dim;
    uint d_rel;
    uint has_q_scale_vec;
    uint has_rel_scale_vec;
    uint has_rel;
    uint do_v_norm;
    uint has_sink;
    uint use_log_scaling;
    uint has_positions;
    uint base_pos;
    uint log_floor;
    float q_clip;
    float k_clip;
    float rel_clip;
    float log_alpha;
    float rms_eps;
    uint _pad0;
};
static_assert(sizeof(Inkling2AttnPostProjParams) == 80, "Inkling2AttnPostProjParams must be 80 bytes");

struct Inkling2DynSconvAssembleParams {
    uint m;
    uint kv_dim;
    uint num_kv_heads;
    uint head_dim;
    uint proj_n;
    uint has_static_w;
    uint static_w_f16;
    float weight_scale;
};
static_assert(sizeof(Inkling2DynSconvAssembleParams) == 32, "Inkling2DynSconvAssembleParams must be 32 bytes");

struct Inkling2MsgBankNormParams {
    uint m;
    uint hidden_dim;
    uint num_ring_blocks;
    uint append_cur;
    uint block_stride_elems;
    float rms_eps;
    uint _pad0;
    uint _pad1;
    uint block_slots[8];
};
static_assert(sizeof(Inkling2MsgBankNormParams) == 64, "Inkling2MsgBankNormParams must be 64 bytes");

struct Inkling2CrossPassMsgParams {
    uint m;
    uint hidden_dim;
    uint bank_len;
    uint q_rows;
    uint kv_rows;
    uint loop_layer_idx;
    uint has_kv_q;
    uint queries_f16;
    float scale;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};
static_assert(sizeof(Inkling2CrossPassMsgParams) == 48, "Inkling2CrossPassMsgParams must be 48 bytes");

struct Inkling2MessageGateParams {
    uint m;
    uint state_dim;
    uint down_in_dim;
    uint gate_rank;
    uint has_norm_w;
    uint has_forget_bias;
    uint has_gain_bias;
    uint msg_f16;
    uint gain_act;
    float rms_eps;
    uint _pad0;
    uint _pad1;
};
static_assert(sizeof(Inkling2MessageGateParams) == 48, "Inkling2MessageGateParams must be 48 bytes");

struct Inkling2GatherRowsParams {
    uint count;
    uint row_elems;
    uint _pad0;
    uint _pad1;
};
static_assert(sizeof(Inkling2GatherRowsParams) == 16, "Inkling2GatherRowsParams must be 16 bytes");

struct Inkling2ScatterRowsParams {
    uint count;
    uint row_elems;
    uint slot_stride;
    uint _pad0;
};
static_assert(sizeof(Inkling2ScatterRowsParams) == 16, "Inkling2ScatterRowsParams must be 16 bytes");

struct InkExpertWeights {
    ulong gate_qweight_offset;
    ulong gate_sb_offset;
    ulong up_qweight_offset;
    ulong up_sb_offset;
    ulong down_qweight_offset;
    ulong down_sb_offset;
};
static_assert(sizeof(InkExpertWeights) == 48, "InkExpertWeights must be 48 bytes");

struct InkMoeScheduleHeader {
    uint num_unique_experts;
    uint total_assignments;
    uint m;
    uint top_k;
};
static_assert(sizeof(InkMoeScheduleHeader) == 16, "InkMoeScheduleHeader must be 16 bytes");

struct InkMoeIndirectDispatchArgs {
    uint threadgroups_per_grid[3];
};
static_assert(sizeof(InkMoeIndirectDispatchArgs) == 12, "InkMoeIndirectDispatchArgs must be 12 bytes");

struct Inkling2MoeRouteCsrParams {
    uint m;
    uint num_experts;
    uint top_k;
    uint has_bias;
    uint bias_f16;
    uint bias_mode;
    uint norm_topk_prob;
    uint scoring_sigmoid;
    uint is_heterogeneous;
    uint bucket_unique_stride;
    uint bucket_offsets_stride;
    uint gate_up_groups_x_q;
    uint gate_up_groups_x_f;
    uint down_groups_x;
    uint down_groups_x_q8;
    float route_scale;
    float router_softcap;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};
static_assert(sizeof(Inkling2MoeRouteCsrParams) == 80, "Inkling2MoeRouteCsrParams must be 80 bytes");

struct Inkling2MoeGateUpSoftcapParams {
    uint hidden_size;
    uint intermediate_size;
    uint num_groups;
    uint row_bytes;
    uint group_size;
    uint pair_interleaved;
    uint scale_stride;
    uint gate_outer_scale_off;
    uint up_outer_scale_off;
    float outer_scale_fallback;
    float gate_softcap;
    float up_softcap;
};
static_assert(sizeof(Inkling2MoeGateUpSoftcapParams) == 48, "Inkling2MoeGateUpSoftcapParams must be 48 bytes");

// ---------------------------------------------------------------------------
// 9.1 Per-Head ScaledL2Clip + LogScale + UnitRMSNorm + Sink f16->f32
// ---------------------------------------------------------------------------

static inline float ink_compute_log_tau(uint tok,
                                          uint use_log_scaling,
                                          uint has_positions,
                                          uint base_pos,
                                          uint log_floor,
                                          float log_alpha,
                                          const device uint *positions) {
    if (use_log_scaling == 0u || log_alpha == 0.0f) {
        return 1.0f;
    }
    const uint pos = (has_positions != 0u) ? positions[tok] : (base_pos + tok);
    const float floor_f = (log_floor > 0u) ? float(log_floor) : 1.0f;
    const float ratio = max(1.0f, (float(pos) + 1.0f) / floor_f);
    return 1.0f + log_alpha * log(ratio);
}

static inline void ink_scaled_l2_clip_head(device half *row,
                                           const device float *scale_vec,
                                           uint sv_off,
                                           bool has_sv,
                                           uint head_dim,
                                           float threshold,
                                           float tau,
                                           uint lane) {
    float sq = 0.0f;
    for (uint i = lane; i < head_dim; i += 32u) {
        float fv = float(row[i]);
        if (has_sv) {
            fv *= scale_vec[sv_off + i];
        }
        sq = fma(fv, fv, sq);
    }
    const float norm = sqrt(simd_sum(sq));
    const float clip = (threshold > 0.0f) ? min(1.0f, threshold / (norm + 1e-6f)) : 1.0f;
    const float fac = clip * tau;
    for (uint i = lane; i < head_dim; i += 32u) {
        float fv = float(row[i]);
        if (has_sv) {
            fv *= scale_vec[sv_off + i];
        }
        row[i] = half(fv * fac);
    }
}

static inline void ink_unit_rms_norm_head(device half *row,
                                          uint head_dim,
                                          float rms_eps,
                                          uint lane) {
    float sq = 0.0f;
    for (uint i = lane; i < head_dim; i += 32u) {
        const float v = float(row[i]);
        sq = fma(v, v, sq);
    }
    const float inv_rms = rsqrt(simd_sum(sq) / float(head_dim) + rms_eps);
    for (uint i = lane; i < head_dim; i += 32u) {
        row[i] = half(float(row[i]) * inv_rms);
    }
}

// Grid: (num_heads, m, 1) threadgroups of 32 threads (1 simdgroup/head).
kernel void ami_inkling2_scaled_l2_clip_per_head(device half *buf [[buffer(0)]],
                                                 const device float *scale_vec [[buffer(1)]],
                                                 const device uint *positions [[buffer(2)]],
                                                 constant Inkling2ScaledL2ClipParams &p [[buffer(3)]],
                                                 uint2 tg_pos [[threadgroup_position_in_grid]],
                                                 uint lane [[thread_index_in_simdgroup]]) {
    const uint nh = tg_pos.x;
    const uint tok = tg_pos.y;
    if (nh >= p.num_heads || tok >= p.m || p.head_dim == 0u) {
        return;
    }
    const float tau = ink_compute_log_tau(tok, p.use_log_scaling, p.has_positions,
                                          p.base_pos, p.log_floor, p.log_alpha, positions);
    device half *row = buf + (ulong(tok) * p.num_heads + nh) * p.head_dim;
    ink_scaled_l2_clip_head(row, scale_vec, nh * p.head_dim, p.has_scale_vec != 0u,
                            p.head_dim, p.threshold, tau, lane);
}

// Grid: (num_heads, m, 1) threadgroups of 32 threads (1 simdgroup/head).
kernel void ami_inkling2_unit_rms_norm_per_head(device half *buf [[buffer(0)]],
                                                constant Inkling2UnitRmsNormParams &p [[buffer(1)]],
                                                uint2 tg_pos [[threadgroup_position_in_grid]],
                                                uint lane [[thread_index_in_simdgroup]]) {
    const uint nh = tg_pos.x;
    const uint tok = tg_pos.y;
    if (nh >= p.num_heads || tok >= p.m || p.head_dim == 0u) {
        return;
    }
    device half *row = buf + (ulong(tok) * p.num_heads + nh) * p.head_dim;
    ink_unit_rms_norm_head(row, p.head_dim, p.rms_eps, lane);
}

// Grid (threads): (count, 1, 1).
kernel void ami_inkling2_f16_to_f32(const device half *src [[buffer(0)]],
                                    device float *dst [[buffer(1)]],
                                    constant Inkling2F16ToF32Params &p [[buffer(2)]],
                                    uint gid [[thread_position_in_grid]]) {
    if (gid >= p.count) {
        return;
    }
    dst[gid] = float(src[gid]);
}

// Single-dispatch post-projection prep: Q clip+log, K clip, optional R clip+log,
// optional V unit RMSNorm, optional sink f16->f32.
// Grid: (n_q + n_kv + (has_rel ? n_q : 0) + (do_v_norm ? n_kv : 0) + (has_sink ? 1 : 0), m, 1), 32 threads.
kernel void ami_inkling2_attn_post_proj(device half *q [[buffer(0)]],
                                        device half *k [[buffer(1)]],
                                        device half *q_rel [[buffer(2)]],
                                        device half *v [[buffer(3)]],
                                        const device float *q_scale_vec [[buffer(4)]],
                                        const device float *rel_scale_vec [[buffer(5)]],
                                        const device uint *positions [[buffer(6)]],
                                        const device half *sink_f16 [[buffer(7)]],
                                        device float *sink_f32 [[buffer(8)]],
                                        constant Inkling2AttnPostProjParams &p [[buffer(9)]],
                                        uint2 tg_pos [[threadgroup_position_in_grid]],
                                        uint lane [[thread_index_in_simdgroup]]) {
    const uint tok = tg_pos.y;
    if (tok >= p.m || p.head_dim == 0u) {
        return;
    }
    uint x = tg_pos.x;
    if (x < p.num_q_heads) {
        const float tau = ink_compute_log_tau(tok, p.use_log_scaling, p.has_positions,
                                              p.base_pos, p.log_floor, p.log_alpha, positions);
        device half *row = q + (ulong(tok) * p.num_q_heads + x) * p.head_dim;
        ink_scaled_l2_clip_head(row, q_scale_vec, x * p.head_dim, p.has_q_scale_vec != 0u,
                                p.head_dim, p.q_clip, tau, lane);
        return;
    }
    x -= p.num_q_heads;
    if (x < p.num_kv_heads) {
        device half *row = k + (ulong(tok) * p.num_kv_heads + x) * p.head_dim;
        ink_scaled_l2_clip_head(row, q_scale_vec, 0u, false, p.head_dim, p.k_clip, 1.0f, lane);
        return;
    }
    x -= p.num_kv_heads;
    if (p.has_rel != 0u) {
        if (x < p.num_q_heads) {
            const float tau = ink_compute_log_tau(tok, p.use_log_scaling, p.has_positions,
                                                  p.base_pos, p.log_floor, p.log_alpha, positions);
            device half *row = q_rel + (ulong(tok) * p.num_q_heads + x) * p.d_rel;
            ink_scaled_l2_clip_head(row, rel_scale_vec, x * p.d_rel, p.has_rel_scale_vec != 0u,
                                    p.d_rel, p.rel_clip, tau, lane);
            return;
        }
        x -= p.num_q_heads;
    }
    if (p.do_v_norm != 0u) {
        if (x < p.num_kv_heads) {
            device half *row = v + (ulong(tok) * p.num_kv_heads + x) * p.head_dim;
            ink_unit_rms_norm_head(row, p.head_dim, p.rms_eps, lane);
            return;
        }
        x -= p.num_kv_heads;
    }
    if (p.has_sink != 0u && x == 0u) {
        const ulong base = ulong(tok) * p.num_q_heads;
        for (uint h = lane; h < p.num_q_heads; h += 32u) {
            sink_f32[base + h] = float(sink_f16[base + h]);
        }
    }
}

// ---------------------------------------------------------------------------
// 9.2 Global dyn_sconv [t-3..t] Lag-Reversed Weight Assembly
// ---------------------------------------------------------------------------
// Grid (threads): (kv_dim, m, 1).
kernel void ami_inkling2_assemble_dyn_sconv_w(const device half *proj [[buffer(0)]],
                                              const device uchar *static_w [[buffer(1)]],
                                              device half *dyn_w [[buffer(2)]],
                                              constant Inkling2DynSconvAssembleParams &p [[buffer(3)]],
                                              uint2 gid [[thread_position_in_grid]]) {
    const uint c = gid.x;
    const uint tok = gid.y;
    if (tok >= p.m || c >= p.kv_dim) {
        return;
    }
    ulong p_off;
    if (p.proj_n == p.kv_dim * 4u) {
        p_off = (ulong(tok) * p.kv_dim + c) * 4ul;
    } else if (p.proj_n == p.num_kv_heads * 4u && p.head_dim > 0u) {
        p_off = (ulong(tok) * p.num_kv_heads + (c / p.head_dim)) * 4ul;
    } else {
        p_off = ulong(tok) * 4ul;
    }

    float w0 = float(proj[p_off + 3ul]);
    float w1 = float(proj[p_off + 2ul]);
    float w2 = float(proj[p_off + 1ul]);
    float w3 = float(proj[p_off + 0ul]);
    if (p.has_static_w != 0u) {
        const ulong sw_off = ulong(c) * 4ul;
        if (p.static_w_f16 != 0u) {
            const device half *sf = (const device half *)static_w;
            w0 += float(sf[sw_off + 0ul]);
            w1 += float(sf[sw_off + 1ul]);
            w2 += float(sf[sw_off + 2ul]);
            w3 += float(sf[sw_off + 3ul]);
        } else {
            const device float *sf = (const device float *)static_w;
            w0 += sf[sw_off + 0ul];
            w1 += sf[sw_off + 1ul];
            w2 += sf[sw_off + 2ul];
            w3 += sf[sw_off + 3ul];
        }
    }
    const float sc = p.weight_scale;
    device half *dst = dyn_w + (ulong(tok) * p.kv_dim + c) * 4ul;
    dst[0] = half(w0 * sc);
    dst[1] = half(w1 * sc);
    dst[2] = half(w2 * sc);
    dst[3] = half(w3 * sc);
}

// ---------------------------------------------------------------------------
// 9.3 Cross-Pass Message Bank Normalization, Message Attention & Gate Rank=64
// ---------------------------------------------------------------------------

// Grid: (num_ring_blocks + append_cur, m, 1) threadgroups of 256 threads.
kernel void ami_inkling2_msg_bank_norm(const device half *block_states [[buffer(0)]],
                                       const device half *cur [[buffer(1)]],
                                       device float *msg_bank_out [[buffer(2)]],
                                       constant Inkling2MsgBankNormParams &p [[buffer(3)]],
                                       uint2 tg_pos [[threadgroup_position_in_grid]],
                                       uint tid [[thread_index_in_threadgroup]],
                                       uint sg_id [[simdgroup_index_in_threadgroup]],
                                       uint lane [[thread_index_in_simdgroup]]) {
    const uint b = tg_pos.x;
    const uint tok = tg_pos.y;
    const uint total_blocks = p.num_ring_blocks + ((p.append_cur != 0u) ? 1u : 0u);
    if (b >= total_blocks || tok >= p.m || p.hidden_dim == 0u) {
        return;
    }

    threadgroup float tg_sum[8];
    const uint d = p.hidden_dim;
    const device half *src = (b < p.num_ring_blocks)
        ? (block_states + ulong(p.block_slots[b]) * p.block_stride_elems + ulong(tok) * d)
        : (cur + ulong(tok) * d);

    float sum_sq = 0.0f;
    for (uint c = tid; c < d; c += 256u) {
        const float v = float(src[c]);
        sum_sq = fma(v, v, sum_sq);
    }
    const float sg_s = simd_sum(sum_sq);
    if (lane == 0u) {
        tg_sum[sg_id] = sg_s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float tot = 0.0f;
        for (uint s = 0u; s < 8u; s++) {
            tot += tg_sum[s];
        }
        tg_sum[0] = rsqrt(tot / float(d) + p.rms_eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_rms = tg_sum[0];

    device float *dst = msg_bank_out + (ulong(b) * p.m + tok) * d;
    for (uint c = tid; c < d; c += 256u) {
        dst[c] = float(src[c]) * inv_rms;
    }
}

// Computes 4 attended messages [msg_q, msg_k, msg_v, msg_mlp] into out_msgs [4, m, hidden_dim].
// Grid: (4, m, 1) threadgroups of 256 threads.
kernel void ami_inkling2_cross_pass_msgs(const device float *msg_bank [[buffer(0)]],
                                         const device uchar *q_src [[buffer(1)]],
                                         const device uchar *kv_q_src [[buffer(2)]],
                                         device float *out_msgs [[buffer(3)]],
                                         constant Inkling2CrossPassMsgParams &p [[buffer(4)]],
                                         uint2 tg_pos [[threadgroup_position_in_grid]],
                                         uint tid [[thread_index_in_threadgroup]],
                                         uint sg_id [[simdgroup_index_in_threadgroup]],
                                         uint lane [[thread_index_in_simdgroup]]) {
    const uint qi = tg_pos.x;
    const uint tok = tg_pos.y;
    if (qi >= 4u || tok >= p.m || p.hidden_dim == 0u || p.bank_len == 0u || p.bank_len > 8u) {
        return;
    }

    threadgroup float tg_dot[8];
    threadgroup float tg_scores[8];
    threadgroup float tg_weights[8];

    const uint d = p.hidden_dim;
    const bool use_kv = (qi == 1u || qi == 2u) && (p.has_kv_q != 0u);
    const uint raw_q_idx = (qi == 0u || qi == 1u) ? (2u * p.loop_layer_idx) : (2u * p.loop_layer_idx + 1u);
    const uint rows = use_kv ? max(p.kv_rows, 1u) : max(p.q_rows, 1u);
    const uint q_idx = raw_q_idx % rows;
    const device uchar *q_raw = use_kv ? kv_q_src : q_src;
    const ulong q_off = ulong(q_idx) * d;
    const ulong mh = ulong(p.m) * d;

    for (uint b = 0u; b < p.bank_len; b++) {
        const device float *bv = msg_bank + ulong(b) * mh + ulong(tok) * d;
        float dot_v = 0.0f;
        for (uint c = tid; c < d; c += 256u) {
            const float qv = (p.queries_f16 != 0u)
                ? float(((const device half *)q_raw)[q_off + c])
                : ((const device float *)q_raw)[q_off + c];
            dot_v = fma(qv, bv[c], dot_v);
        }
        const float sg_d = simd_sum(dot_v);
        if (lane == 0u) {
            tg_dot[sg_id] = sg_d;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float tot = 0.0f;
            for (uint s = 0u; s < 8u; s++) {
                tot += tg_dot[s];
            }
            tg_scores[b] = tot * p.scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        float max_s = -INFINITY;
        for (uint b = 0u; b < p.bank_len; b++) {
            max_s = max(max_s, tg_scores[b]);
        }
        float exp_sum = 0.0f;
        for (uint b = 0u; b < p.bank_len; b++) {
            const float ex = exp(tg_scores[b] - max_s);
            tg_weights[b] = ex;
            exp_sum += ex;
        }
        const float inv_sum = (exp_sum > 0.0f) ? (1.0f / exp_sum) : 0.0f;
        for (uint b = 0u; b < p.bank_len; b++) {
            tg_weights[b] *= inv_sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float *dst = out_msgs + ulong(qi) * mh + ulong(tok) * d;
    for (uint c = tid; c < d; c += 256u) {
        float acc = 0.0f;
        for (uint b = 0u; b < p.bank_len; b++) {
            const float bv = msg_bank[ulong(b) * mh + ulong(tok) * d + c];
            acc = fma(tg_weights[b], bv, acc);
        }
        dst[c] = acc;
    }
}

static inline float ink_load_ctrl_w1(const device uchar *raw, ulong idx) {
    return NORM_W_BF16 ? float(((const device bfloat *)raw)[idx]) : float(((const device half *)raw)[idx]);
}

static inline float4 ink_load_ctrl_w4(const device uchar *raw, ulong idx) {
    return NORM_W_BF16 ? float4(*(const device bfloat4 *)(raw + idx * 2ul))
                       : float4(*(const device half4 *)(raw + idx * 2ul));
}

// Message gate Stage 1: z = down_weight @ [rms(cur), rms(msg)] -> [m, gate_rank] f32.
// Grid: (ceil(gate_rank / 4), ceil(m / 4), 1) threadgroups of 128 threads (4 simdgroups).
kernel void ami_inkling2_msg_gate_down(const device half *cur [[buffer(0)]],
                                       const device uchar *msg_raw [[buffer(1)]],
                                       const device uchar *down_weight [[buffer(2)]],
                                       const device float *norm_weight [[buffer(3)]],
                                       device float *z_out [[buffer(4)]],
                                       constant Inkling2MessageGateParams &p [[buffer(5)]],
                                       uint2 tg_pos [[threadgroup_position_in_grid]],
                                       uint tid [[thread_index_in_threadgroup]],
                                       uint sg_id [[simdgroup_index_in_threadgroup]],
                                       uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint TB = 4u;
    const uint m_base = tg_pos.y * TB;
    if (m_base >= p.m || p.state_dim == 0u || p.gate_rank == 0u) {
        return;
    }
    const uint tile_m = min(TB, p.m - m_base);
    const uint row = tg_pos.x * 4u + sg_id;
    const bool has_msg_in = (p.down_in_dim == p.state_dim * 2u);
    const uint sd = p.state_dim;

    threadgroup float tg_inv_cur[4];
    threadgroup float tg_inv_msg[4];
    threadgroup float tg_red_c[4];
    threadgroup float tg_red_m[4];

    for (uint t = 0u; t < tile_m; t++) {
        const ulong r_off = ulong(m_base + t) * sd;
        float sc = 0.0f;
        float sm = 0.0f;
        for (uint c = tid; c < sd; c += 128u) {
            const float vc = float(cur[r_off + c]);
            sc = fma(vc, vc, sc);
            if (has_msg_in) {
                const float vm = (p.msg_f16 != 0u)
                    ? float(((const device half *)msg_raw)[r_off + c])
                    : ((const device float *)msg_raw)[r_off + c];
                sm = fma(vm, vm, sm);
            }
        }
        const float sgc = simd_sum(sc);
        const float sgm = simd_sum(sm);
        if (lane == 0u) {
            tg_red_c[sg_id] = sgc;
            tg_red_m[sg_id] = sgm;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float tc = 0.0f;
            float tm = 0.0f;
            for (uint s = 0u; s < 4u; s++) {
                tc += tg_red_c[s];
                tm += tg_red_m[s];
            }
            tg_inv_cur[t] = rsqrt(tc / float(sd) + p.rms_eps);
            tg_inv_msg[t] = has_msg_in ? rsqrt(tm / float(sd) + p.rms_eps) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row >= p.gate_rank) {
        return;
    }

    const ulong w_row_off = ulong(row) * p.down_in_dim;
    const ulong off0 = ulong(m_base + 0u) * sd;
    const ulong off1 = ulong(m_base + min(1u, tile_m - 1u)) * sd;
    const ulong off2 = ulong(m_base + min(2u, tile_m - 1u)) * sd;
    const ulong off3 = ulong(m_base + min(3u, tile_m - 1u)) * sd;

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    const bool vec_sd = ((sd & 3u) == 0u) && ((p.down_in_dim & 3u) == 0u);
    if (vec_sd) {
        const uint vec_len = sd >> 2;
        for (uint vi = lane; vi < vec_len; vi += 32u) {
            const uint c = vi * 4u;
            const float4 nw = (p.has_norm_w != 0u) ? *(const device float4 *)(norm_weight + c) : float4(1.0f);
            const float4 w = ink_load_ctrl_w4(down_weight, w_row_off + c) * nw;
            a0 += dot(float4(*(const device half4 *)(cur + off0 + c)) * tg_inv_cur[0], w);
            if (tile_m > 1u) a1 += dot(float4(*(const device half4 *)(cur + off1 + c)) * tg_inv_cur[1], w);
            if (tile_m > 2u) a2 += dot(float4(*(const device half4 *)(cur + off2 + c)) * tg_inv_cur[2], w);
            if (tile_m > 3u) a3 += dot(float4(*(const device half4 *)(cur + off3 + c)) * tg_inv_cur[3], w);
        }
        if (has_msg_in) {
            for (uint vi = lane; vi < vec_len; vi += 32u) {
                const uint c = vi * 4u;
                const float4 nw = (p.has_norm_w != 0u) ? *(const device float4 *)(norm_weight + sd + c) : float4(1.0f);
                const float4 w = ink_load_ctrl_w4(down_weight, w_row_off + sd + c) * nw;
                if (p.msg_f16 != 0u) {
                    const device half *mh = (const device half *)msg_raw;
                    a0 += dot(float4(*(const device half4 *)(mh + off0 + c)) * tg_inv_msg[0], w);
                    if (tile_m > 1u) a1 += dot(float4(*(const device half4 *)(mh + off1 + c)) * tg_inv_msg[1], w);
                    if (tile_m > 2u) a2 += dot(float4(*(const device half4 *)(mh + off2 + c)) * tg_inv_msg[2], w);
                    if (tile_m > 3u) a3 += dot(float4(*(const device half4 *)(mh + off3 + c)) * tg_inv_msg[3], w);
                } else {
                    const device float *mf = (const device float *)msg_raw;
                    a0 += dot(*(const device float4 *)(mf + off0 + c) * tg_inv_msg[0], w);
                    if (tile_m > 1u) a1 += dot(*(const device float4 *)(mf + off1 + c) * tg_inv_msg[1], w);
                    if (tile_m > 2u) a2 += dot(*(const device float4 *)(mf + off2 + c) * tg_inv_msg[2], w);
                    if (tile_m > 3u) a3 += dot(*(const device float4 *)(mf + off3 + c) * tg_inv_msg[3], w);
                }
            }
        }
    } else {
        for (uint c = lane; c < sd; c += 32u) {
            const float nw = (p.has_norm_w != 0u) ? norm_weight[c] : 1.0f;
            const float w = ink_load_ctrl_w1(down_weight, w_row_off + c) * nw;
            a0 = fma(float(cur[off0 + c]) * tg_inv_cur[0], w, a0);
            if (tile_m > 1u) a1 = fma(float(cur[off1 + c]) * tg_inv_cur[1], w, a1);
            if (tile_m > 2u) a2 = fma(float(cur[off2 + c]) * tg_inv_cur[2], w, a2);
            if (tile_m > 3u) a3 = fma(float(cur[off3 + c]) * tg_inv_cur[3], w, a3);
        }
        if (has_msg_in) {
            for (uint c = lane; c < sd; c += 32u) {
                const float nw = (p.has_norm_w != 0u) ? norm_weight[sd + c] : 1.0f;
                const float w = ink_load_ctrl_w1(down_weight, w_row_off + sd + c) * nw;
                if (p.msg_f16 != 0u) {
                    const device half *mh = (const device half *)msg_raw;
                    a0 = fma(float(mh[off0 + c]) * tg_inv_msg[0], w, a0);
                    if (tile_m > 1u) a1 = fma(float(mh[off1 + c]) * tg_inv_msg[1], w, a1);
                    if (tile_m > 2u) a2 = fma(float(mh[off2 + c]) * tg_inv_msg[2], w, a2);
                    if (tile_m > 3u) a3 = fma(float(mh[off3 + c]) * tg_inv_msg[3], w, a3);
                } else {
                    const device float *mf = (const device float *)msg_raw;
                    a0 = fma(mf[off0 + c] * tg_inv_msg[0], w, a0);
                    if (tile_m > 1u) a1 = fma(mf[off1 + c] * tg_inv_msg[1], w, a1);
                    if (tile_m > 2u) a2 = fma(mf[off2 + c] * tg_inv_msg[2], w, a2);
                    if (tile_m > 3u) a3 = fma(mf[off3 + c] * tg_inv_msg[3], w, a3);
                }
            }
        }
    }

    const float s0 = simd_sum(a0), s1 = simd_sum(a1), s2 = simd_sum(a2), s3 = simd_sum(a3);
    if (lane == 0u) {
        z_out[ulong(m_base + 0u) * p.gate_rank + row] = s0;
        if (tile_m > 1u) z_out[ulong(m_base + 1u) * p.gate_rank + row] = s1;
        if (tile_m > 2u) z_out[ulong(m_base + 2u) * p.gate_rank + row] = s2;
        if (tile_m > 3u) z_out[ulong(m_base + 3u) * p.gate_rank + row] = s3;
    }
}

// Message gate Stage 2: forget = tanh(forget_weight @ z), gain = tanh(gain_weight @ z),
// out = (1 - forget) * cur + gain * msg.
// Grid: (ceil(state_dim / 8), ceil(m / 4), 1) threadgroups of 128 threads.
kernel void ami_inkling2_msg_gate_up(const device float *z_in [[buffer(0)]],
                                     const device uchar *forget_weight [[buffer(1)]],
                                     const device uchar *gain_weight [[buffer(2)]],
                                     const device float *forget_bias [[buffer(3)]],
                                     const device float *gain_bias [[buffer(4)]],
                                     const device half *cur [[buffer(5)]],
                                     const device uchar *msg_raw [[buffer(6)]],
                                     device half *out [[buffer(7)]],
                                     constant Inkling2MessageGateParams &p [[buffer(8)]],
                                     uint2 tg_pos [[threadgroup_position_in_grid]],
                                     uint tid [[thread_index_in_threadgroup]],
                                     uint sg_id [[simdgroup_index_in_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint TB = 4u;
    constexpr uint RPS = 2u;
    const uint m_base = tg_pos.y * TB;
    const uint row0 = tg_pos.x * 8u + sg_id * RPS;
    if (m_base >= p.m || row0 >= p.state_dim || p.gate_rank == 0u) {
        return;
    }
    const uint tile_m = min(TB, p.m - m_base);
    const uint gr = p.gate_rank;
    const uint sd = p.state_dim;

    threadgroup float tg_z[4][64];
    if (gr <= 64u) {
        for (uint idx = tid; idx < tile_m * gr; idx += 128u) {
            const uint t = idx / gr;
            const uint k = idx - t * gr;
            tg_z[t][k] = z_in[ulong(m_base + t) * gr + k];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint r = 0u; r < RPS; r++) {
        const uint row = row0 + r;
        if (row >= sd) {
            break;
        }
        float f0 = 0.0f, f1 = 0.0f, f2 = 0.0f, f3 = 0.0f;
        float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
        const ulong w_base = ulong(row) * gr;

        for (uint k = lane * 2u; k < gr; k += 64u) {
            const bool has_k1 = (k + 1u) < gr;
            const float fw0 = ink_load_ctrl_w1(forget_weight, w_base + k);
            const float fw1 = has_k1 ? ink_load_ctrl_w1(forget_weight, w_base + k + 1u) : 0.0f;
            const float gw0 = ink_load_ctrl_w1(gain_weight, w_base + k);
            const float gw1 = has_k1 ? ink_load_ctrl_w1(gain_weight, w_base + k + 1u) : 0.0f;

            {
                const float z0 = (gr <= 64u) ? tg_z[0][k] : z_in[ulong(m_base + 0u) * gr + k];
                const float z1 = has_k1 ? ((gr <= 64u) ? tg_z[0][k + 1u] : z_in[ulong(m_base + 0u) * gr + k + 1u]) : 0.0f;
                f0 = fma(fw0, z0, fma(fw1, z1, f0));
                g0 = fma(gw0, z0, fma(gw1, z1, g0));
            }
            if (tile_m > 1u) {
                const float z0 = (gr <= 64u) ? tg_z[1][k] : z_in[ulong(m_base + 1u) * gr + k];
                const float z1 = has_k1 ? ((gr <= 64u) ? tg_z[1][k + 1u] : z_in[ulong(m_base + 1u) * gr + k + 1u]) : 0.0f;
                f1 = fma(fw0, z0, fma(fw1, z1, f1));
                g1 = fma(gw0, z0, fma(gw1, z1, g1));
            }
            if (tile_m > 2u) {
                const float z0 = (gr <= 64u) ? tg_z[2][k] : z_in[ulong(m_base + 2u) * gr + k];
                const float z1 = has_k1 ? ((gr <= 64u) ? tg_z[2][k + 1u] : z_in[ulong(m_base + 2u) * gr + k + 1u]) : 0.0f;
                f2 = fma(fw0, z0, fma(fw1, z1, f2));
                g2 = fma(gw0, z0, fma(gw1, z1, g2));
            }
            if (tile_m > 3u) {
                const float z0 = (gr <= 64u) ? tg_z[3][k] : z_in[ulong(m_base + 3u) * gr + k];
                const float z1 = has_k1 ? ((gr <= 64u) ? tg_z[3][k + 1u] : z_in[ulong(m_base + 3u) * gr + k + 1u]) : 0.0f;
                f3 = fma(fw0, z0, fma(fw1, z1, f3));
                g3 = fma(gw0, z0, fma(gw1, z1, g3));
            }
        }

        const float sf0 = simd_sum(f0), sg0 = simd_sum(g0);
        const float sf1 = simd_sum(f1), sg1 = simd_sum(g1);
        const float sf2 = simd_sum(f2), sg2 = simd_sum(g2);
        const float sf3 = simd_sum(f3), sg3 = simd_sum(g3);
        if (lane == 0u) {
            const float fb = (p.has_forget_bias != 0u) ? forget_bias[row] : 0.0f;
            const float gb = (p.has_gain_bias != 0u) ? gain_bias[row] : 0.0f;
            #define INK_WRITE_GATE_OUT(T_IDX, SF_VAL, SG_VAL) \
                { \
                    const ulong idx = ulong(m_base + T_IDX) * sd + row; \
                    const float fg = ink_tanh_safe(SF_VAL + fb); \
                    const float mg = (p.gain_act == 1u) ? ink_silu(SG_VAL + gb) : ink_tanh_safe(SG_VAL + gb); \
                    const float cv = float(cur[idx]); \
                    const float mv = (p.msg_f16 != 0u) ? float(((const device half *)msg_raw)[idx]) : ((const device float *)msg_raw)[idx]; \
                    out[idx] = half((1.0f - fg) * cv + mg * mv); \
                }
            INK_WRITE_GATE_OUT(0u, sf0, sg0);
            if (tile_m > 1u) INK_WRITE_GATE_OUT(1u, sf1, sg1);
            if (tile_m > 2u) INK_WRITE_GATE_OUT(2u, sf2, sg2);
            if (tile_m > 3u) INK_WRITE_GATE_OUT(3u, sf3, sg3);
            #undef INK_WRITE_GATE_OUT
        }
    }
}

// ---------------------------------------------------------------------------
// 9.4 Gather / Scatter Rows + Inkling 2 CSR Router + Indirect Softcap MoE
// ---------------------------------------------------------------------------

// Grid: (ceil(row_elems / 256), count, 1) threadgroups, 256 threads.
kernel void ami_inkling2_gather_rows_f16(const device half *src [[buffer(0)]],
                                         const device uint *indices [[buffer(1)]],
                                         device half *dst [[buffer(2)]],
                                         constant Inkling2GatherRowsParams &p [[buffer(3)]],
                                         uint2 tg_pos [[threadgroup_position_in_grid]],
                                         uint tid [[thread_index_in_threadgroup]]) {
    const uint out_row = tg_pos.y;
    const uint col = tg_pos.x * 256u + tid;
    if (out_row >= p.count || col >= p.row_elems) {
        return;
    }
    const uint src_row = indices[out_row];
    dst[ulong(out_row) * p.row_elems + col] = src[ulong(src_row) * p.row_elems + col];
}

// Scatters src[i, :] into dst[slot, :] where slot = (slot_stride == 0) ? slots_or_tok[i]
// : (slots_or_tok[i] * slot_stride + slot_k_indices[i]).
// Grid: (ceil(row_elems / 256), count, 1) threadgroups, 256 threads.
kernel void ami_inkling2_scatter_rows_f16(const device half *src [[buffer(0)]],
                                          const device uint *slots_or_tok [[buffer(1)]],
                                          const device uint *slot_k_indices [[buffer(2)]],
                                          device half *dst [[buffer(3)]],
                                          constant Inkling2ScatterRowsParams &p [[buffer(4)]],
                                          uint2 tg_pos [[threadgroup_position_in_grid]],
                                          uint tid [[thread_index_in_threadgroup]]) {
    const uint in_row = tg_pos.y;
    const uint col = tg_pos.x * 256u + tid;
    if (in_row >= p.count || col >= p.row_elems) {
        return;
    }
    const uint dst_row = (p.slot_stride == 0u)
        ? slots_or_tok[in_row]
        : (slots_or_tok[in_row] * p.slot_stride + slot_k_indices[in_row]);
    dst[ulong(dst_row) * p.row_elems + col] = src[ulong(in_row) * p.row_elems + col];
}

static inline void ink_write_schedule_indirect(device uchar *slot_base,
                                               uint active_experts,
                                               uint gate_up_groups_x_q,
                                               uint gate_up_groups_x_f,
                                               uint down_groups_x,
                                               uint down_groups_x_q8) {
    device InkMoeIndirectDispatchArgs *gu_q = (device InkMoeIndirectDispatchArgs *)(slot_base + 16ul);
    device InkMoeIndirectDispatchArgs *gu_f = (device InkMoeIndirectDispatchArgs *)(slot_base + 28ul);
    device InkMoeIndirectDispatchArgs *dn   = (device InkMoeIndirectDispatchArgs *)(slot_base + 40ul);
    device InkMoeIndirectDispatchArgs *dn8  = (device InkMoeIndirectDispatchArgs *)(slot_base + 52ul);
    gu_q->threadgroups_per_grid[0] = gate_up_groups_x_q;
    gu_q->threadgroups_per_grid[1] = active_experts;
    gu_q->threadgroups_per_grid[2] = 1u;
    gu_f->threadgroups_per_grid[0] = gate_up_groups_x_f;
    gu_f->threadgroups_per_grid[1] = active_experts;
    gu_f->threadgroups_per_grid[2] = 1u;
    dn->threadgroups_per_grid[0]   = down_groups_x;
    dn->threadgroups_per_grid[1]   = active_experts;
    dn->threadgroups_per_grid[2]   = 1u;
    dn8->threadgroups_per_grid[0]  = down_groups_x_q8;
    dn8->threadgroups_per_grid[1]  = active_experts;
    dn8->threadgroups_per_grid[2]  = 1u;
}

// Single-threadgroup Inkling 2 router (sigmoid/softmax, router_softcap, selection/add bias,
// route_scale) + CSR deduplication + optional 5-bucket scheme partitioning + G20 IndirectDispatchArgs.
// Grid: (1, 1, 1) threadgroup of 256 threads.
kernel void ami_inkling2_moe_route_csr(const device uchar *logits_raw [[buffer(0)]],
                                       const device uchar *bias_raw [[buffer(1)]],
                                       device InkMoeScheduleHeader *sched_hdr [[buffer(2)]],
                                       device uint *unique_experts [[buffer(3)]],
                                       device uint *expert_token_offsets [[buffer(4)]],
                                       device uint *token_indices [[buffer(5)]],
                                       device float *routing_weights [[buffer(6)]],
                                       device uint *slot_k_indices [[buffer(7)]],
                                       device uint *scratch_topk_experts [[buffer(8)]],
                                       device float *scratch_topk_weights [[buffer(9)]],
                                       const device uchar *expert_schemes [[buffer(10)]],
                                       device uchar *bucket_headers [[buffer(11)]],
                                       device uint *bucket_unique_experts [[buffer(12)]],
                                       device uint *bucket_expert_token_offsets [[buffer(13)]],
                                       constant Inkling2MoeRouteCsrParams &p [[buffer(14)]],
                                       uint tid [[thread_index_in_threadgroup]],
                                       uint sg_id [[simdgroup_index_in_threadgroup]],
                                       uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint NUM_SCHEMES = 5u;
    threadgroup atomic_uint s_counts[1024];
    threadgroup atomic_uint s_cursors[1024];
    threadgroup uint tg_sel_e[8][16];
    threadgroup float tg_sel_w[8][16];
    threadgroup uint s_prefix_tmp[8];
    threadgroup uint s_u_cnt;
    threadgroup uint s_a_cnt;

    const uint k_eff = min(p.top_k, min(p.num_experts, 16u));
    for (uint e = tid; e < p.num_experts; e += 256u) {
        atomic_store_explicit(&s_counts[e], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint tok = sg_id; tok < p.m; tok += 8u) {
        const ulong row_base = ulong(tok) * p.num_experts;
        const uint chunks = (p.num_experts + 31u) >> 5;
        float inv_sum = 1.0f;
        float max_v = 0.0f;
        if (p.scoring_sigmoid == 0u) {
            float local_max = -INFINITY;
            for (uint r = 0u; r < chunks; r++) {
                const uint e = lane + r * 32u;
                if (e < p.num_experts) {
                    float v = INK_LOGITS_F16
                        ? float(((const device half *)logits_raw)[row_base + e])
                        : ((const device float *)logits_raw)[row_base + e];
                    v = ink_softcap(v, p.router_softcap);
                    if (p.has_bias != 0u && p.bias_mode == 0u) {
                        const float b = (p.bias_f16 != 0u)
                            ? float(((const device half *)bias_raw)[e])
                            : ((const device float *)bias_raw)[e];
                        v += b;
                    }
                    local_max = max(local_max, v);
                }
            }
            max_v = simd_max(local_max);
            float local_sum = 0.0f;
            for (uint r = 0u; r < chunks; r++) {
                const uint e = lane + r * 32u;
                if (e < p.num_experts) {
                    float v = INK_LOGITS_F16
                        ? float(((const device half *)logits_raw)[row_base + e])
                        : ((const device float *)logits_raw)[row_base + e];
                    v = ink_softcap(v, p.router_softcap);
                    if (p.has_bias != 0u && p.bias_mode == 0u) {
                        const float b = (p.bias_f16 != 0u)
                            ? float(((const device half *)bias_raw)[e])
                            : ((const device float *)bias_raw)[e];
                        v += b;
                    }
                    local_sum += exp(v - max_v);
                }
            }
            const float sum_exp = simd_sum(local_sum);
            inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
        }

        uint picked_mask = 0u;
        float w_sum = 0.0f;
        for (uint k = 0u; k < k_eff; k++) {
            float best_s = -INFINITY;
            float best_prob = 0.0f;
            uint best_e = 0xFFFFFFFFu;
            for (uint r = 0u; r < chunks; r++) {
                const uint e = lane + r * 32u;
                if (e < p.num_experts && ((picked_mask >> r) & 1u) == 0u) {
                    float v = INK_LOGITS_F16
                        ? float(((const device half *)logits_raw)[row_base + e])
                        : ((const device float *)logits_raw)[row_base + e];
                    v = ink_softcap(v, p.router_softcap);
                    if (p.has_bias != 0u && p.bias_mode == 0u) {
                        const float b = (p.bias_f16 != 0u)
                            ? float(((const device half *)bias_raw)[e])
                            : ((const device float *)bias_raw)[e];
                        v += b;
                    }
                    const float prob = (p.scoring_sigmoid != 0u)
                        ? ink_sigmoid(v)
                        : (exp(v - max_v) * inv_sum);
                    float sel_s = prob;
                    if (p.has_bias != 0u && p.bias_mode == 1u) {
                        const float b = (p.bias_f16 != 0u)
                            ? float(((const device half *)bias_raw)[e])
                            : ((const device float *)bias_raw)[e];
                        sel_s += b;
                    }
                    if (sel_s > best_s || (sel_s == best_s && e < best_e)) {
                        best_s = sel_s;
                        best_prob = prob;
                        best_e = e;
                    }
                }
            }
            for (ushort off = 16u; off > 0u; off >>= 1u) {
                const float os = simd_shuffle_down(best_s, off);
                const float op = simd_shuffle_down(best_prob, off);
                const uint oe = simd_shuffle_down(best_e, off);
                if (oe != 0xFFFFFFFFu && (best_e == 0xFFFFFFFFu || os > best_s || (os == best_s && oe < best_e))) {
                    best_s = os;
                    best_prob = op;
                    best_e = oe;
                }
            }
            const uint raw_win_e = simd_broadcast_first(best_e);
            const uint win_e = (raw_win_e < p.num_experts) ? raw_win_e : 0u;
            const float win_prob = simd_broadcast_first(best_prob);
            if ((win_e & 31u) == lane) {
                picked_mask |= (1u << (win_e >> 5));
            }
            if (lane == 0u) {
                tg_sel_e[sg_id][k] = win_e;
                tg_sel_w[sg_id][k] = win_prob;
                w_sum += win_prob;
            }
        }

        if (lane == 0u) {
            const float norm_fac = (p.norm_topk_prob != 0u && w_sum > 0.0f) ? (1.0f / w_sum) : 1.0f;
            const ulong out_base = ulong(tok) * p.top_k;
            for (uint k = 0u; k < k_eff; k++) {
                const uint e = tg_sel_e[sg_id][k];
                scratch_topk_experts[out_base + k] = e;
                scratch_topk_weights[out_base + k] = tg_sel_w[sg_id][k] * norm_fac * p.route_scale;
                atomic_fetch_add_explicit(&s_counts[e], 1u, memory_order_relaxed);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    // Phase 2: compact unique experts & CSR offsets (either unified or 5 scheme buckets).
    const uint num_passes = (p.is_heterogeneous != 0u) ? NUM_SCHEMES : 1u;
    uint running_u_base = 0u;
    uint running_a_base = 0u;
    for (uint s = 0u; s < num_passes; s++) {
        const ulong u_base_off = ulong(s) * p.bucket_unique_stride;
        const ulong o_base_off = ulong(s) * p.bucket_offsets_stride;
        uint u_cnt_s = 0u;
        uint a_cnt_s = 0u;

        for (uint blk_base = 0u; blk_base < p.num_experts; blk_base += 256u) {
            const uint e = blk_base + tid;
            const uint cnt = (e < p.num_experts)
                ? atomic_load_explicit(&s_counts[e], memory_order_relaxed)
                : 0u;
            const bool match_s = (p.is_heterogeneous == 0u) ||
                ((e < p.num_experts) && (min(uint(expert_schemes[e]), NUM_SCHEMES - 1u) == s));
            const uint in_s = (cnt > 0u && match_s) ? 1u : 0u;
            const uint cnt_s = match_s ? cnt : 0u;

            const uint sg_act_excl = simd_prefix_exclusive_sum(in_s);
            if (lane == 31u) {
                s_prefix_tmp[sg_id] = sg_act_excl + in_s;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg_id == 0u) {
                const uint v = (lane < 8u) ? s_prefix_tmp[lane] : 0u;
                const uint ex = simd_prefix_exclusive_sum(v);
                if (lane < 8u) s_prefix_tmp[lane] = ex;
                if (lane == 7u) s_u_cnt = ex + v;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint u_in_blk = s_prefix_tmp[sg_id] + sg_act_excl;
            const uint blk_u_cnt = s_u_cnt;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const uint sg_cnt_excl = simd_prefix_exclusive_sum(cnt_s);
            if (lane == 31u) {
                s_prefix_tmp[sg_id] = sg_cnt_excl + cnt_s;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg_id == 0u) {
                const uint v = (lane < 8u) ? s_prefix_tmp[lane] : 0u;
                const uint ex = simd_prefix_exclusive_sum(v);
                if (lane < 8u) s_prefix_tmp[lane] = ex;
                if (lane == 7u) s_a_cnt = ex + v;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint a_in_blk = s_prefix_tmp[sg_id] + sg_cnt_excl;
            const uint blk_a_cnt = s_a_cnt;

            if (in_s != 0u) {
                const uint slot_off = running_a_base + a_cnt_s + a_in_blk;
                atomic_store_explicit(&s_cursors[e], slot_off, memory_order_relaxed);
                if (p.is_heterogeneous != 0u) {
                    bucket_unique_experts[u_base_off + u_cnt_s + u_in_blk] = e;
                    bucket_expert_token_offsets[o_base_off + u_cnt_s + u_in_blk] = slot_off;
                }
                unique_experts[running_u_base + u_cnt_s + u_in_blk] = e;
                expert_token_offsets[running_u_base + u_cnt_s + u_in_blk] = slot_off;
            }
            u_cnt_s += blk_u_cnt;
            a_cnt_s += blk_a_cnt;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0u && p.is_heterogeneous != 0u) {
            bucket_expert_token_offsets[o_base_off + u_cnt_s] = running_a_base + a_cnt_s;
            device uchar *slot_ptr = bucket_headers + ulong(s) * 64ul;
            device InkMoeScheduleHeader *bhdr = (device InkMoeScheduleHeader *)slot_ptr;
            bhdr->num_unique_experts = u_cnt_s;
            bhdr->total_assignments = a_cnt_s;
            bhdr->m = p.m;
            bhdr->top_k = k_eff;
            ink_write_schedule_indirect(slot_ptr, u_cnt_s, p.gate_up_groups_x_q,
                                        p.gate_up_groups_x_f, p.down_groups_x, p.down_groups_x_q8);
        }
        running_u_base += u_cnt_s;
        running_a_base += a_cnt_s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        expert_token_offsets[running_u_base] = running_a_base;
        sched_hdr->num_unique_experts = running_u_base;
        sched_hdr->total_assignments = running_a_base;
        sched_hdr->m = p.m;
        sched_hdr->top_k = k_eff;
        ink_write_schedule_indirect((device uchar *)sched_hdr, running_u_base,
                                    p.gate_up_groups_x_q, p.gate_up_groups_x_f,
                                    p.down_groups_x, p.down_groups_x_q8);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: scatter token assignments into contiguous CSR slots.
    const uint total_assign = p.m * k_eff;
    for (uint idx = tid; idx < total_assign; idx += 256u) {
        const uint tok = idx / k_eff;
        const uint k_idx = idx - tok * k_eff;
        const uint e = scratch_topk_experts[idx];
        const float w = scratch_topk_weights[idx];
        const uint dst = atomic_fetch_add_explicit(&s_cursors[e], 1u, memory_order_relaxed);
        token_indices[dst] = tok;
        routing_weights[dst] = w;
        slot_k_indices[dst] = k_idx;
    }
}

// ---------------------------------------------------------------------------
// 9.5 Inkling 2 Indirect Pair-Interleaved + SoftcappedSwiGLU Gate+Up Kernels
// ---------------------------------------------------------------------------

static inline float ink_eval_swiglu_softcap(float g, float u, float gate_softcap, float up_softcap) {
    return (gate_softcap > 0.0f)
        ? (ink_softcap(g, gate_softcap) * ink_sigmoid(g) * ink_softcap(u, up_softcap))
        : ink_softcap(ink_silu(g) * u, up_softcap);
}

// Canonical .amf DType discriminants for INK_MOE_DTYPE.
constant constexpr uint INK_DT_Q4_AFFINE = 2u;
constant constexpr uint INK_DT_Q3_AFFINE = 3u;
constant constexpr uint INK_DT_Q2_AFFINE = 6u;
constant constexpr uint INK_DT_Q5_AFFINE = 7u;
constant constexpr uint INK_DT_Q6_AFFINE = 8u;
constant constexpr uint INK_DT_Q8_AFFINE = 9u;
constant constexpr uint INK_DT_FP8_BLOCK = 10u;
constant constexpr uint INK_DT_FP8_ROW = 11u;
constant constexpr uint INK_DT_FP8_TENSOR = 12u;
constant constexpr uint INK_DT_MXFP8_G32 = 13u;
constant constexpr uint INK_DT_NVFP4_G16 = 14u;
constant constexpr uint INK_DT_GGUF_Q4_0 = 15u;
constant constexpr uint INK_DT_GGUF_Q4_1 = 16u;
constant constexpr uint INK_DT_GGUF_Q5_0 = 17u;
constant constexpr uint INK_DT_GGUF_Q5_1 = 18u;
constant constexpr uint INK_DT_GGUF_Q8_0 = 19u;
constant constexpr uint INK_DT_GGUF_Q2_K = 20u;
constant constexpr uint INK_DT_GGUF_Q3_K = 21u;
constant constexpr uint INK_DT_GGUF_Q4_K = 22u;
constant constexpr uint INK_DT_GGUF_Q5_K = 23u;
constant constexpr uint INK_DT_GGUF_Q6_K = 24u;
constant constexpr uint INK_DT_GGUF_IQ1_S = 25u;
constant constexpr uint INK_DT_GGUF_IQ1_M = 26u;
constant constexpr uint INK_DT_GGUF_IQ2_XXS = 27u;
constant constexpr uint INK_DT_GGUF_IQ2_XS = 28u;
constant constexpr uint INK_DT_GGUF_IQ2_S = 29u;
constant constexpr uint INK_DT_GGUF_IQ3_XXS = 30u;
constant constexpr uint INK_DT_GGUF_IQ3_S = 31u;
constant constexpr uint INK_DT_GGUF_IQ4_NL = 32u;
constant constexpr uint INK_DT_GGUF_IQ4_XS = 33u;
constant constexpr uint INK_DT_GGUF_TQ1_0 = 34u;
constant constexpr uint INK_DT_GGUF_TQ2_0 = 35u;
constant constexpr uint INK_DT_GGUF_MXFP4 = 36u;
constant constexpr uint INK_DT_GGUF_Q2_0 = 37u;

static inline ushort ink_read_u16(const device uchar *p) {
    return ushort(p[0]) | (ushort(p[1]) << 8);
}

static inline uint ink_read_u32(const device uchar *p) {
    return uint(p[0]) | (uint(p[1]) << 8) | (uint(p[2]) << 16) | (uint(p[3]) << 24);
}

static inline float ink_read_f16(const device uchar *p) {
    return float(as_type<half>(ink_read_u16(p)));
}

static inline float4 ink_load_half4_safe(const device half *p, bool vec_aligned) {
    if (vec_aligned) {
        return float4(*(const device half4 *)p);
    }
    return float4(float(p[0]), float(p[1]), float(p[2]), float(p[3]));
}

static inline float ink_e4m3_to_f32(uchar b) {
    const uint ax = uint(b) & 0x7Fu;
    if (ax == 0x7Fu) return as_type<float>(0x7FC00000u);
    const uint exp4 = (ax >> 3) & 0xFu;
    const uint man3 = ax & 0x7u;
    const uint norm_bits = ((exp4 + 120u) << 23) | (man3 << 20);
    const float mag = (exp4 == 0u) ? (float(man3) * (1.0f / 512.0f)) : as_type<float>(norm_bits);
    return as_type<float>(as_type<uint>(mag) | ((uint(b) & 0x80u) << 24));
}

static inline float ink_e4m3_scale_to_f32(uchar b) {
    const uint ax = uint(b) & 0x7Fu;
    if (ax == 0u || ax == 0x7Fu) return 0.0f;
    const uint exp4 = (ax >> 3) & 0xFu;
    const uint man3 = ax & 0x7u;
    const uint norm_bits = ((exp4 + 120u) << 23) | (man3 << 20);
    const float mag = (exp4 == 0u) ? (float(man3) * (1.0f / 512.0f)) : as_type<float>(norm_bits);
    return as_type<float>(as_type<uint>(mag) | ((uint(b) & 0x80u) << 24));
}

static inline float ink_ue8m0_to_f32(uchar e) {
    if (e == 0xFFu) return as_type<float>(0x7FC00000u);
    if (e == 0u) return as_type<float>(0x00400000u);
    return as_type<float>(uint(e) << 23);
}

constant constexpr float INK_E2M1_LUT[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
static inline float ink_e2m1_to_f32(uint nib) {
    const uint mag = nib & 0x7u;
    const float v = INK_E2M1_LUT[mag];
    return ((nib & 0x8u) != 0u && mag != 0u) ? -v : v;
}

constant constexpr int INK_KVALUES_IQ4NL[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

// 1. Dense FP16 / BF16 (`NORM_W_BF16`), Grid X: ceil(intermediate_size / 16).
kernel void ami_inkling2_moe_gate_up_softcap_float(const device half *x [[buffer(0)]],
                                                   const device uchar *weights_base [[buffer(1)]],
                                                   const device InkExpertWeights *expert_table [[buffer(2)]],
                                                   const device InkMoeScheduleHeader *sched_hdr [[buffer(3)]],
                                                   const device uint *unique_experts [[buffer(4)]],
                                                   const device uint *expert_token_offsets [[buffer(5)]],
                                                   const device uint *token_indices [[buffer(6)]],
                                                   device half *inter [[buffer(7)]],
                                                   constant Inkling2MoeGateUpSoftcapParams &p [[buffer(8)]],
                                                   uint2 tg_pos [[threadgroup_position_in_grid]],
                                                   uint sg_id [[simdgroup_index_in_threadgroup]],
                                                   uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint RPS = 4u;
    const uint u = tg_pos.y;
    if (u >= sched_hdr->num_unique_experts) return;
    const uint row0 = tg_pos.x * 16u + sg_id * RPS;
    if (row0 >= p.intermediate_size) return;
    const uint s0 = expert_token_offsets[u];
    const uint s1 = expert_token_offsets[u + 1u];
    const uint count_u = s1 - s0;
    if (count_u == 0u) return;

    const InkExpertWeights ew = expert_table[unique_experts[u]];
    const device uchar *gate_w = weights_base + ew.gate_qweight_offset;
    const device uchar *up_w = (p.pair_interleaved != 0u) ? gate_w : (weights_base + ew.up_qweight_offset);
    const bool vec_x = ((p.hidden_size & 3u) == 0u);
    const uint vec_k = p.hidden_size >> 2;

    for (uint t_base = 0u; t_base < count_u; ) {
        const uint rem = min(4u, count_u - t_base);
        const device half *x0 = x + ulong(token_indices[s0 + t_base + 0u]) * p.hidden_size;
        const device half *x1 = x + ulong(token_indices[s0 + t_base + min(1u, rem - 1u)]) * p.hidden_size;
        const device half *x2 = x + ulong(token_indices[s0 + t_base + min(2u, rem - 1u)]) * p.hidden_size;
        const device half *x3 = x + ulong(token_indices[s0 + t_base + min(3u, rem - 1u)]) * p.hidden_size;

        for (uint r = 0u; r < RPS; r++) {
            const uint row = row0 + r;
            if (row >= p.intermediate_size) break;
            const ulong g_row = (p.pair_interleaved != 0u) ? (ulong(row) * 2ul) : ulong(row);
            const ulong u_row = (p.pair_interleaved != 0u) ? (ulong(row) * 2ul + 1ul) : ulong(row);
            const ulong g_base = g_row * p.hidden_size;
            const ulong u_base = u_row * p.hidden_size;
            float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
            float u0 = 0.0f, u1 = 0.0f, u2 = 0.0f, u3 = 0.0f;

            if (vec_x) {
                for (uint vi = lane; vi < vec_k; vi += 32u) {
                    const uint col = vi * 4u;
                    const float4 gw = ink_load_ctrl_w4(gate_w, g_base + col);
                    const float4 uw = ink_load_ctrl_w4(up_w, u_base + col);
                    { const float4 xv = float4(*(const device half4 *)(x0 + col)); g0 += dot(xv, gw); u0 += dot(xv, uw); }
                    if (rem > 1u) { const float4 xv = float4(*(const device half4 *)(x1 + col)); g1 += dot(xv, gw); u1 += dot(xv, uw); }
                    if (rem > 2u) { const float4 xv = float4(*(const device half4 *)(x2 + col)); g2 += dot(xv, gw); u2 += dot(xv, uw); }
                    if (rem > 3u) { const float4 xv = float4(*(const device half4 *)(x3 + col)); g3 += dot(xv, gw); u3 += dot(xv, uw); }
                }
            } else {
                for (uint col = lane; col < p.hidden_size; col += 32u) {
                    const float gw = ink_load_ctrl_w1(gate_w, g_base + col);
                    const float uw = ink_load_ctrl_w1(up_w, u_base + col);
                    { const float xv = float(x0[col]); g0 = fma(xv, gw, g0); u0 = fma(xv, uw, u0); }
                    if (rem > 1u) { const float xv = float(x1[col]); g1 = fma(xv, gw, g1); u1 = fma(xv, uw, u1); }
                    if (rem > 2u) { const float xv = float(x2[col]); g2 = fma(xv, gw, g2); u2 = fma(xv, uw, u2); }
                    if (rem > 3u) { const float xv = float(x3[col]); g3 = fma(xv, gw, g3); u3 = fma(xv, uw, u3); }
                }
            }

            const float sg0 = simd_sum(g0), su0 = simd_sum(u0);
            const float sg1 = simd_sum(g1), su1 = simd_sum(u1);
            const float sg2 = simd_sum(g2), su2 = simd_sum(u2);
            const float sg3 = simd_sum(g3), su3 = simd_sum(u3);
            if (lane == 0u) {
                inter[ulong(s0 + t_base + 0u) * p.intermediate_size + row] = half(ink_eval_swiglu_softcap(sg0, su0, p.gate_softcap, p.up_softcap));
                if (rem > 1u) inter[ulong(s0 + t_base + 1u) * p.intermediate_size + row] = half(ink_eval_swiglu_softcap(sg1, su1, p.gate_softcap, p.up_softcap));
                if (rem > 2u) inter[ulong(s0 + t_base + 2u) * p.intermediate_size + row] = half(ink_eval_swiglu_softcap(sg2, su2, p.gate_softcap, p.up_softcap));
                if (rem > 3u) inter[ulong(s0 + t_base + 3u) * p.intermediate_size + row] = half(ink_eval_swiglu_softcap(sg3, su3, p.gate_softcap, p.up_softcap));
            }
        }
        t_base += rem;
    }
}

// 2. Affine (q2/q3/q4/q5/q6/q8_affine, group_size 32/64/128), Grid X: ceil(intermediate_size / 8).
static inline uint ink_affine_bits(uint dtype) {
    switch (dtype) {
        case INK_DT_Q2_AFFINE: return 2u;
        case INK_DT_Q3_AFFINE: return 3u;
        case INK_DT_Q4_AFFINE: return 4u;
        case INK_DT_Q5_AFFINE: return 5u;
        case INK_DT_Q6_AFFINE: return 6u;
        case INK_DT_Q8_AFFINE: return 8u;
        default: return 4u;
    }
}

static inline void ink_affine_dequant8(const device uchar *wp,
                                       float scale,
                                       float bias,
                                       thread float4 &w0,
                                       thread float4 &w1) {
    const uint bits = ink_affine_bits(INK_MOE_DTYPE);
    if (bits == 2u) {
        const uint w = uint(wp[0]) | (uint(wp[1]) << 8);
        w0 = float4(float((w >> 0) & 3u), float((w >> 2) & 3u), float((w >> 4) & 3u), float((w >> 6) & 3u)) * scale + bias;
        w1 = float4(float((w >> 8) & 3u), float((w >> 10) & 3u), float((w >> 12) & 3u), float((w >> 14) & 3u)) * scale + bias;
    } else if (bits == 3u) {
        const uint b0 = wp[0], b1 = wp[1], b2 = wp[2];
        w0 = float4(float(b0 & 7u), float((b0 >> 3) & 7u),
                    float(((b0 >> 6) & 3u) | ((b1 & 1u) << 2)), float((b1 >> 1) & 7u)) * scale + bias;
        w1 = float4(float((b1 >> 4) & 7u), float(((b1 >> 7) & 1u) | ((b2 & 3u) << 1)),
                    float((b2 >> 2) & 7u), float((b2 >> 5) & 7u)) * scale + bias;
    } else if (bits == 4u) {
        const uint w = ink_read_u32(wp);
        w0 = float4(float(w & 0xFu), float((w >> 4) & 0xFu), float((w >> 8) & 0xFu), float((w >> 12) & 0xFu)) * scale + bias;
        w1 = float4(float((w >> 16) & 0xFu), float((w >> 20) & 0xFu), float((w >> 24) & 0xFu), float((w >> 28) & 0xFu)) * scale + bias;
    } else if (bits == 5u) {
        const uint b0 = wp[0], b1 = wp[1], b2 = wp[2], b3 = wp[3], b4 = wp[4];
        w0 = float4(float(b0 & 0x1Fu), float(((b0 >> 5) & 0x07u) | ((b1 & 0x03u) << 3)),
                    float((b1 >> 2) & 0x1Fu), float(((b1 >> 7) & 0x01u) | ((b2 & 0x0Fu) << 1))) * scale + bias;
        w1 = float4(float(((b2 >> 4) & 0x0Fu) | ((b3 & 0x01u) << 4)), float((b3 >> 1) & 0x1Fu),
                    float(((b3 >> 6) & 0x03u) | ((b4 & 0x07u) << 2)), float((b4 >> 3) & 0x1Fu)) * scale + bias;
    } else if (bits == 6u) {
        const uint a0 = wp[0], a1 = wp[1], a2 = wp[2];
        const uint b0 = wp[3], b1 = wp[4], b2 = wp[5];
        w0 = float4(float(a0 & 0x3Fu), float(((a0 >> 6) & 0x03u) | ((a1 & 0x0Fu) << 2)),
                    float(((a1 >> 4) & 0x0Fu) | ((a2 & 0x03u) << 4)), float((a2 >> 2) & 0x3Fu)) * scale + bias;
        w1 = float4(float(b0 & 0x3Fu), float(((b0 >> 6) & 0x03u) | ((b1 & 0x0Fu) << 2)),
                    float(((b1 >> 4) & 0x0Fu) | ((b2 & 0x03u) << 4)), float((b2 >> 2) & 0x3Fu)) * scale + bias;
    } else {
        w0 = float4(float(wp[0]), float(wp[1]), float(wp[2]), float(wp[3])) * scale + bias;
        w1 = float4(float(wp[4]), float(wp[5]), float(wp[6]), float(wp[7])) * scale + bias;
    }
}

kernel void ami_inkling2_moe_gate_up_softcap_affine(const device half *x [[buffer(0)]],
                                                    const device uchar *weights_base [[buffer(1)]],
                                                    const device InkExpertWeights *expert_table [[buffer(2)]],
                                                    const device InkMoeScheduleHeader *sched_hdr [[buffer(3)]],
                                                    const device uint *unique_experts [[buffer(4)]],
                                                    const device uint *expert_token_offsets [[buffer(5)]],
                                                    const device uint *token_indices [[buffer(6)]],
                                                    device half *inter [[buffer(7)]],
                                                    constant Inkling2MoeGateUpSoftcapParams &p [[buffer(8)]],
                                                    uint2 tg_pos [[threadgroup_position_in_grid]],
                                                    uint sg_id [[simdgroup_index_in_threadgroup]],
                                                    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint ROWS_PER_SG = 2u;
    const uint u = tg_pos.y;
    if (u >= sched_hdr->num_unique_experts) return;
    const uint row0 = tg_pos.x * 8u + sg_id * ROWS_PER_SG;
    if (row0 >= p.intermediate_size) return;
    const uint s0 = expert_token_offsets[u];
    const uint s1 = expert_token_offsets[u + 1u];
    const uint count_u = s1 - s0;
    if (count_u == 0u) return;

    const InkExpertWeights ew = expert_table[unique_experts[u]];
    const device uchar *gate_qw = weights_base + ew.gate_qweight_offset;
    const device half2 *gate_sb = (const device half2 *)(weights_base + ew.gate_sb_offset);
    const device uchar *up_qw = (p.pair_interleaved != 0u) ? gate_qw : (weights_base + ew.up_qweight_offset);
    const device half2 *up_sb = (p.pair_interleaved != 0u) ? gate_sb : (const device half2 *)(weights_base + ew.up_sb_offset);

    const uint bits = ink_affine_bits(INK_MOE_DTYPE);
    const uint lanes_per_grp = p.group_size >> 3;
    const uint grps_per_sg = 32u / lanes_per_grp;
    const uint sub_group = lane / lanes_per_grp;
    const uint lane_in_grp = lane & (lanes_per_grp - 1u);
    const uint grp_bytes = (p.group_size * bits) >> 3;

    for (uint t_base = 0u; t_base < count_u; ) {
        const uint rem = min(4u, count_u - t_base);
        const device half *x0 = x + ulong(token_indices[s0 + t_base + 0u]) * p.hidden_size;
        const device half *x1 = x + ulong(token_indices[s0 + t_base + min(1u, rem - 1u)]) * p.hidden_size;
        const device half *x2 = x + ulong(token_indices[s0 + t_base + min(2u, rem - 1u)]) * p.hidden_size;
        const device half *x3 = x + ulong(token_indices[s0 + t_base + min(3u, rem - 1u)]) * p.hidden_size;

        for (uint r = 0u; r < ROWS_PER_SG; r++) {
            const uint row = row0 + r;
            if (row >= p.intermediate_size) break;
            const ulong g_row = (p.pair_interleaved != 0u) ? (ulong(row) * 2ul) : ulong(row);
            const ulong u_row = (p.pair_interleaved != 0u) ? (ulong(row) * 2ul + 1ul) : ulong(row);
            const ulong gw_base = g_row * p.row_bytes;
            const ulong uw_base = u_row * p.row_bytes;
            const ulong gsb_base = g_row * p.num_groups;
            const ulong usb_base = u_row * p.num_groups;
            float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
            float u0 = 0.0f, u1 = 0.0f, u2 = 0.0f, u3 = 0.0f;

            for (uint g_base = 0u; g_base < p.num_groups; g_base += grps_per_sg) {
                const uint g = g_base + sub_group;
                if (g >= p.num_groups) continue;
                const uint k_off = g * p.group_size + lane_in_grp * 8u;
                const uint w_off = g * grp_bytes + lane_in_grp * bits;
                const half2 gsb = gate_sb[gsb_base + g];
                const half2 usb = up_sb[usb_base + g];
                float4 gw0, gw1, uw0, uw1;
                ink_affine_dequant8(gate_qw + gw_base + w_off, float(gsb.x), float(gsb.y), gw0, gw1);
                ink_affine_dequant8(up_qw + uw_base + w_off, float(usb.x), float(usb.y), uw0, uw1);

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
                inter[ulong(s0 + t_base + 0u) * p.intermediate_size + row] = half(ink_eval_swiglu_softcap(sg0, su0, p.gate_softcap, p.up_softcap));
                if (rem > 1u) inter[ulong(s0 + t_base + 1u) * p.intermediate_size + row] = half(ink_eval_swiglu_softcap(sg1, su1, p.gate_softcap, p.up_softcap));
                if (rem > 2u) inter[ulong(s0 + t_base + 2u) * p.intermediate_size + row] = half(ink_eval_swiglu_softcap(sg2, su2, p.gate_softcap, p.up_softcap));
                if (rem > 3u) inter[ulong(s0 + t_base + 3u) * p.intermediate_size + row] = half(ink_eval_swiglu_softcap(sg3, su3, p.gate_softcap, p.up_softcap));
            }
        }
        t_base += rem;
    }
}

// 3. FP Family (fp8_block, fp8_row, fp8_tensor, mxfp8_g32, nvfp4_g16), Grid X: ceil(intermediate_size / 8).
static inline float ink_fp_row_scale_at(const device uchar *scales,
                                        uint row,
                                        uint col,
                                        uint num_groups,
                                        uint scale_stride,
                                        float outer_scale) {
    if (INK_MOE_DTYPE == INK_DT_FP8_BLOCK) {
        const device float *sf = (const device float *)scales;
        return sf[ulong(row >> 7) * scale_stride + (col >> 7)];
    } else if (INK_MOE_DTYPE == INK_DT_FP8_ROW) {
        const device float *sf = (const device float *)scales;
        return sf[row];
    } else if (INK_MOE_DTYPE == INK_DT_FP8_TENSOR) {
        const device float *sf = (const device float *)scales;
        return sf[(scale_stride != 0u) ? row : 0u];
    } else if (INK_MOE_DTYPE == INK_DT_MXFP8_G32) {
        return ink_ue8m0_to_f32(scales[ulong(row) * num_groups + (col >> 5)]);
    } else {
        return ink_e4m3_scale_to_f32(scales[ulong(row) * num_groups + (col >> 4)]) * outer_scale;
    }
}

static inline float ink_fp_resolve_outer(const device uchar *scales, uint off, float fallback) {
    if (INK_MOE_DTYPE != INK_DT_NVFP4_G16) return 1.0f;
    if (off != 0xFFFFFFFFu) return *(const device float *)(scales + off);
    return fallback;
}

static inline void ink_fp_dequant8(const device uchar *w_row,
                                   const device uchar *scales,
                                   uint row,
                                   uint col,
                                   uint num_groups,
                                   uint scale_stride,
                                   float outer_scale,
                                   thread float4 &w0,
                                   thread float4 &w1) {
    const float sc = ink_fp_row_scale_at(scales, row, col, num_groups, scale_stride, outer_scale);
    if (INK_MOE_DTYPE == INK_DT_NVFP4_G16) {
        const uint p4 = ink_read_u32(w_row + (col >> 1));
        w0 = float4(ink_e2m1_to_f32((p4 >> 0) & 0xFu), ink_e2m1_to_f32((p4 >> 4) & 0xFu),
                    ink_e2m1_to_f32((p4 >> 8) & 0xFu), ink_e2m1_to_f32((p4 >> 12) & 0xFu)) * sc;
        w1 = float4(ink_e2m1_to_f32((p4 >> 16) & 0xFu), ink_e2m1_to_f32((p4 >> 20) & 0xFu),
                    ink_e2m1_to_f32((p4 >> 24) & 0xFu), ink_e2m1_to_f32((p4 >> 28) & 0xFu)) * sc;
    } else {
        const device uchar *wp = w_row + col;
        w0 = float4(ink_e4m3_to_f32(wp[0]), ink_e4m3_to_f32(wp[1]),
                    ink_e4m3_to_f32(wp[2]), ink_e4m3_to_f32(wp[3])) * sc;
        w1 = float4(ink_e4m3_to_f32(wp[4]), ink_e4m3_to_f32(wp[5]),
                    ink_e4m3_to_f32(wp[6]), ink_e4m3_to_f32(wp[7])) * sc;
    }
}

static inline float ink_fp_dot_tail(const device half *x,
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
    if (INK_MOE_DTYPE == INK_DT_NVFP4_G16) {
        const float sc = ink_fp_row_scale_at(scales, row, col, num_groups, scale_stride, outer_scale);
        const device uchar *wp = w_row + (col >> 1);
        for (uint t = 0u; t < rem; t++) {
            const uchar byte_val = wp[t >> 1];
            const uint nib = ((t & 1u) != 0u) ? (byte_val >> 4) : (byte_val & 0xFu);
            acc = fma(float(x[col + t]), ink_e2m1_to_f32(nib) * sc, acc);
        }
    } else {
        for (uint t = 0u; t < rem; t++) {
            const float sc = ink_fp_row_scale_at(scales, row, col + t, num_groups, scale_stride, outer_scale);
            acc = fma(float(x[col + t]), ink_e4m3_to_f32(w_row[col + t]) * sc, acc);
        }
    }
    return acc;
}

kernel void ami_inkling2_moe_gate_up_softcap_fp(const device half *x [[buffer(0)]],
                                                const device uchar *weights_base [[buffer(1)]],
                                                const device InkExpertWeights *expert_table [[buffer(2)]],
                                                const device InkMoeScheduleHeader *sched_hdr [[buffer(3)]],
                                                const device uint *unique_experts [[buffer(4)]],
                                                const device uint *expert_token_offsets [[buffer(5)]],
                                                const device uint *token_indices [[buffer(6)]],
                                                device half *inter [[buffer(7)]],
                                                constant Inkling2MoeGateUpSoftcapParams &p [[buffer(8)]],
                                                uint2 tg_pos [[threadgroup_position_in_grid]],
                                                uint sg_id [[simdgroup_index_in_threadgroup]],
                                                uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint ROWS_PER_SG = 2u;
    const uint u = tg_pos.y;
    if (u >= sched_hdr->num_unique_experts) return;
    const uint row0 = tg_pos.x * 8u + sg_id * ROWS_PER_SG;
    if (row0 >= p.intermediate_size) return;
    const uint s0 = expert_token_offsets[u];
    const uint s1 = expert_token_offsets[u + 1u];
    const uint count_u = s1 - s0;
    if (count_u == 0u) return;

    const InkExpertWeights ew = expert_table[unique_experts[u]];
    const device uchar *gate_qw = weights_base + ew.gate_qweight_offset;
    const device uchar *gate_sc = weights_base + ew.gate_sb_offset;
    const device uchar *up_qw = (p.pair_interleaved != 0u) ? gate_qw : (weights_base + ew.up_qweight_offset);
    const device uchar *up_sc = (p.pair_interleaved != 0u) ? gate_sc : (weights_base + ew.up_sb_offset);
    const float gate_outer = ink_fp_resolve_outer(gate_sc, p.gate_outer_scale_off, p.outer_scale_fallback);
    const float up_outer = ink_fp_resolve_outer(up_sc, p.up_outer_scale_off, p.outer_scale_fallback);
    const bool vec_x = ((p.hidden_size & 3u) == 0u);

    for (uint t_base = 0u; t_base < count_u; ) {
        const uint rem = min(4u, count_u - t_base);
        const device half *x0 = x + ulong(token_indices[s0 + t_base + 0u]) * p.hidden_size;
        const device half *x1 = x + ulong(token_indices[s0 + t_base + min(1u, rem - 1u)]) * p.hidden_size;
        const device half *x2 = x + ulong(token_indices[s0 + t_base + min(2u, rem - 1u)]) * p.hidden_size;
        const device half *x3 = x + ulong(token_indices[s0 + t_base + min(3u, rem - 1u)]) * p.hidden_size;

        for (uint r = 0u; r < ROWS_PER_SG; r++) {
            const uint row = row0 + r;
            if (row >= p.intermediate_size) break;
            const uint g_row = (p.pair_interleaved != 0u) ? (row * 2u) : row;
            const uint u_row = (p.pair_interleaved != 0u) ? (row * 2u + 1u) : row;
            const device uchar *gw_row = gate_qw + ulong(g_row) * p.row_bytes;
            const device uchar *uw_row = up_qw + ulong(u_row) * p.row_bytes;
            float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
            float u0 = 0.0f, u1 = 0.0f, u2 = 0.0f, u3 = 0.0f;

            for (uint k_base = 0u; k_base < p.hidden_size; k_base += 256u) {
                const uint col = k_base + lane * 8u;
                if (col >= p.hidden_size) continue;
                if (col + 8u <= p.hidden_size) {
                    float4 gw0, gw1, uw0, uw1;
                    ink_fp_dequant8(gw_row, gate_sc, g_row, col, p.num_groups, p.scale_stride, gate_outer, gw0, gw1);
                    ink_fp_dequant8(uw_row, up_sc, u_row, col, p.num_groups, p.scale_stride, up_outer, uw0, uw1);
                    { const float4 a = ink_load_half4_safe(x0 + col, vec_x); const float4 b = ink_load_half4_safe(x0 + col + 4u, vec_x); g0 += dot(a, gw0) + dot(b, gw1); u0 += dot(a, uw0) + dot(b, uw1); }
                    if (rem > 1u) { const float4 a = ink_load_half4_safe(x1 + col, vec_x); const float4 b = ink_load_half4_safe(x1 + col + 4u, vec_x); g1 += dot(a, gw0) + dot(b, gw1); u1 += dot(a, uw0) + dot(b, uw1); }
                    if (rem > 2u) { const float4 a = ink_load_half4_safe(x2 + col, vec_x); const float4 b = ink_load_half4_safe(x2 + col + 4u, vec_x); g2 += dot(a, gw0) + dot(b, gw1); u2 += dot(a, uw0) + dot(b, uw1); }
                    if (rem > 3u) { const float4 a = ink_load_half4_safe(x3 + col, vec_x); const float4 b = ink_load_half4_safe(x3 + col + 4u, vec_x); g3 += dot(a, gw0) + dot(b, gw1); u3 += dot(a, uw0) + dot(b, uw1); }
                } else {
                    g0 += ink_fp_dot_tail(x0, gw_row, gate_sc, g_row, col, p.hidden_size, p.num_groups, p.scale_stride, gate_outer);
                    u0 += ink_fp_dot_tail(x0, uw_row, up_sc, u_row, col, p.hidden_size, p.num_groups, p.scale_stride, up_outer);
                    if (rem > 1u) { g1 += ink_fp_dot_tail(x1, gw_row, gate_sc, g_row, col, p.hidden_size, p.num_groups, p.scale_stride, gate_outer); u1 += ink_fp_dot_tail(x1, uw_row, up_sc, u_row, col, p.hidden_size, p.num_groups, p.scale_stride, up_outer); }
                    if (rem > 2u) { g2 += ink_fp_dot_tail(x2, gw_row, gate_sc, g_row, col, p.hidden_size, p.num_groups, p.scale_stride, gate_outer); u2 += ink_fp_dot_tail(x2, uw_row, up_sc, u_row, col, p.hidden_size, p.num_groups, p.scale_stride, up_outer); }
                    if (rem > 3u) { g3 += ink_fp_dot_tail(x3, gw_row, gate_sc, g_row, col, p.hidden_size, p.num_groups, p.scale_stride, gate_outer); u3 += ink_fp_dot_tail(x3, uw_row, up_sc, u_row, col, p.hidden_size, p.num_groups, p.scale_stride, up_outer); }
                }
            }

            const float sg0 = simd_sum(g0), su0 = simd_sum(u0);
            const float sg1 = simd_sum(g1), su1 = simd_sum(u1);
            const float sg2 = simd_sum(g2), su2 = simd_sum(u2);
            const float sg3 = simd_sum(g3), su3 = simd_sum(u3);
            if (lane == 0u) {
                inter[ulong(s0 + t_base + 0u) * p.intermediate_size + row] = half(ink_eval_swiglu_softcap(sg0, su0, p.gate_softcap, p.up_softcap));
                if (rem > 1u) inter[ulong(s0 + t_base + 1u) * p.intermediate_size + row] = half(ink_eval_swiglu_softcap(sg1, su1, p.gate_softcap, p.up_softcap));
                if (rem > 2u) inter[ulong(s0 + t_base + 2u) * p.intermediate_size + row] = half(ink_eval_swiglu_softcap(sg2, su2, p.gate_softcap, p.up_softcap));
                if (rem > 3u) inter[ulong(s0 + t_base + 3u) * p.intermediate_size + row] = half(ink_eval_swiglu_softcap(sg3, su3, p.gate_softcap, p.up_softcap));
            }
        }
        t_base += rem;
    }
}

// 4. GGUF Block Formats (q4_0, q4_1, q5_0, q5_1, q8_0, iq4_nl, mxfp4, q2_0), Grid X: ceil(intermediate_size / 8).
static inline uint ink_gguf_blk_bytes(uint dtype) {
    switch (dtype) {
        case INK_DT_GGUF_Q4_0: return 18u;
        case INK_DT_GGUF_Q4_1: return 20u;
        case INK_DT_GGUF_Q5_0: return 22u;
        case INK_DT_GGUF_Q5_1: return 24u;
        case INK_DT_GGUF_Q8_0: return 34u;
        case INK_DT_GGUF_IQ4_NL: return 18u;
        case INK_DT_GGUF_MXFP4: return 17u;
        case INK_DT_GGUF_Q2_0: return 18u;
        default: return 18u;
    }
}

static inline void ink_gguf_blk_dequant8(const device uchar *row_ptr,
                                         uint blk_step,
                                         uint lane,
                                         uint k_total,
                                         thread uint &k0,
                                         thread uint &k1,
                                         thread float4 &w0,
                                         thread float4 &w1) {
    if (INK_MOE_DTYPE == INK_DT_GGUF_Q2_0) {
        const uint blk_idx = blk_step + (lane >> 3);
        const uint lane_in_blk = lane & 7u;
        k0 = blk_idx * 64u + lane_in_blk * 8u;
        k1 = k0 + 4u;
        if (k0 >= k_total) { w0 = float4(0.0f); w1 = float4(0.0f); return; }
        const device uchar *blk = row_ptr + ulong(blk_idx) * 18ul;
        const float d = ink_read_f16(blk);
        const uchar b0 = blk[2u + lane_in_blk * 2u];
        const uchar b1 = blk[2u + lane_in_blk * 2u + 1u];
        w0 = float4(float(int((b0 >> 0) & 3u) - 1), float(int((b0 >> 2) & 3u) - 1),
                    float(int((b0 >> 4) & 3u) - 1), float(int((b0 >> 6) & 3u) - 1)) * d;
        w1 = float4(float(int((b1 >> 0) & 3u) - 1), float(int((b1 >> 2) & 3u) - 1),
                    float(int((b1 >> 4) & 3u) - 1), float(int((b1 >> 6) & 3u) - 1)) * d;
        return;
    }

    const uint blk_idx = blk_step + (lane >> 2);
    const uint lane_in_blk = lane & 3u;
    k0 = blk_idx * 32u + lane_in_blk * 4u;
    k1 = k0 + 16u;
    if (k0 >= k_total) { w0 = float4(0.0f); w1 = float4(0.0f); return; }
    const device uchar *blk = row_ptr + ulong(blk_idx) * ink_gguf_blk_bytes(INK_MOE_DTYPE);
    const uint j0 = lane_in_blk * 4u;

    if (INK_MOE_DTYPE == INK_DT_GGUF_Q4_0) {
        const float d = ink_read_f16(blk);
        const device uchar *qs = blk + 2u + j0;
        w0 = float4(float(int(qs[0] & 0xFu) - 8), float(int(qs[1] & 0xFu) - 8),
                    float(int(qs[2] & 0xFu) - 8), float(int(qs[3] & 0xFu) - 8)) * d;
        w1 = float4(float(int(qs[0] >> 4) - 8), float(int(qs[1] >> 4) - 8),
                    float(int(qs[2] >> 4) - 8), float(int(qs[3] >> 4) - 8)) * d;
    } else if (INK_MOE_DTYPE == INK_DT_GGUF_Q4_1) {
        const float d = ink_read_f16(blk);
        const float m = ink_read_f16(blk + 2u);
        const device uchar *qs = blk + 4u + j0;
        w0 = float4(float(qs[0] & 0xFu), float(qs[1] & 0xFu), float(qs[2] & 0xFu), float(qs[3] & 0xFu)) * d + m;
        w1 = float4(float(qs[0] >> 4), float(qs[1] >> 4), float(qs[2] >> 4), float(qs[3] >> 4)) * d + m;
    } else if (INK_MOE_DTYPE == INK_DT_GGUF_Q5_0) {
        const float d = ink_read_f16(blk);
        const uint qh = ink_read_u32(blk + 2u);
        const device uchar *qs = blk + 6u + j0;
        w0 = float4(float(int((qs[0] & 0xFu) | (((qh >> (j0 + 0u)) << 4) & 0x10u)) - 16),
                    float(int((qs[1] & 0xFu) | (((qh >> (j0 + 1u)) << 4) & 0x10u)) - 16),
                    float(int((qs[2] & 0xFu) | (((qh >> (j0 + 2u)) << 4) & 0x10u)) - 16),
                    float(int((qs[3] & 0xFu) | (((qh >> (j0 + 3u)) << 4) & 0x10u)) - 16)) * d;
        w1 = float4(float(int((qs[0] >> 4) | ((qh >> (j0 + 12u)) & 0x10u)) - 16),
                    float(int((qs[1] >> 4) | ((qh >> (j0 + 13u)) & 0x10u)) - 16),
                    float(int((qs[2] >> 4) | ((qh >> (j0 + 14u)) & 0x10u)) - 16),
                    float(int((qs[3] >> 4) | ((qh >> (j0 + 15u)) & 0x10u)) - 16)) * d;
    } else if (INK_MOE_DTYPE == INK_DT_GGUF_Q5_1) {
        const float d = ink_read_f16(blk);
        const float m = ink_read_f16(blk + 2u);
        const uint qh = ink_read_u32(blk + 4u);
        const device uchar *qs = blk + 8u + j0;
        w0 = float4(float((qs[0] & 0xFu) | (((qh >> (j0 + 0u)) << 4) & 0x10u)),
                    float((qs[1] & 0xFu) | (((qh >> (j0 + 1u)) << 4) & 0x10u)),
                    float((qs[2] & 0xFu) | (((qh >> (j0 + 2u)) << 4) & 0x10u)),
                    float((qs[3] & 0xFu) | (((qh >> (j0 + 3u)) << 4) & 0x10u))) * d + m;
        w1 = float4(float((qs[0] >> 4) | ((qh >> (j0 + 12u)) & 0x10u)),
                    float((qs[1] >> 4) | ((qh >> (j0 + 13u)) & 0x10u)),
                    float((qs[2] >> 4) | ((qh >> (j0 + 14u)) & 0x10u)),
                    float((qs[3] >> 4) | ((qh >> (j0 + 15u)) & 0x10u))) * d + m;
    } else if (INK_MOE_DTYPE == INK_DT_GGUF_Q8_0) {
        const float d = ink_read_f16(blk);
        const device char *qs = (const device char *)(blk + 2u);
        w0 = float4(float(qs[j0 + 0u]), float(qs[j0 + 1u]), float(qs[j0 + 2u]), float(qs[j0 + 3u])) * d;
        w1 = float4(float(qs[j0 + 16u]), float(qs[j0 + 17u]), float(qs[j0 + 18u]), float(qs[j0 + 19u])) * d;
    } else if (INK_MOE_DTYPE == INK_DT_GGUF_IQ4_NL) {
        const float d = ink_read_f16(blk);
        const device uchar *qs = blk + 2u + j0;
        w0 = float4(float(INK_KVALUES_IQ4NL[qs[0] & 0xFu]), float(INK_KVALUES_IQ4NL[qs[1] & 0xFu]),
                    float(INK_KVALUES_IQ4NL[qs[2] & 0xFu]), float(INK_KVALUES_IQ4NL[qs[3] & 0xFu])) * d;
        w1 = float4(float(INK_KVALUES_IQ4NL[qs[0] >> 4]), float(INK_KVALUES_IQ4NL[qs[1] >> 4]),
                    float(INK_KVALUES_IQ4NL[qs[2] >> 4]), float(INK_KVALUES_IQ4NL[qs[3] >> 4])) * d;
    } else {
        const float d = ink_ue8m0_to_f32(blk[0]) * 0.5f;
        const device uchar *qs = blk + 1u + j0;
        w0 = float4(ink_e2m1_to_f32(qs[0] & 0xFu), ink_e2m1_to_f32(qs[1] & 0xFu),
                    ink_e2m1_to_f32(qs[2] & 0xFu), ink_e2m1_to_f32(qs[3] & 0xFu)) * d;
        w1 = float4(ink_e2m1_to_f32(qs[0] >> 4), ink_e2m1_to_f32(qs[1] >> 4),
                    ink_e2m1_to_f32(qs[2] >> 4), ink_e2m1_to_f32(qs[3] >> 4)) * d;
    }
}

kernel void ami_inkling2_moe_gate_up_softcap_gguf_blk(const device half *x [[buffer(0)]],
                                                      const device uchar *weights_base [[buffer(1)]],
                                                      const device InkExpertWeights *expert_table [[buffer(2)]],
                                                      const device InkMoeScheduleHeader *sched_hdr [[buffer(3)]],
                                                      const device uint *unique_experts [[buffer(4)]],
                                                      const device uint *expert_token_offsets [[buffer(5)]],
                                                      const device uint *token_indices [[buffer(6)]],
                                                      device half *inter [[buffer(7)]],
                                                      constant Inkling2MoeGateUpSoftcapParams &p [[buffer(8)]],
                                                      uint2 tg_pos [[threadgroup_position_in_grid]],
                                                      uint sg_id [[simdgroup_index_in_threadgroup]],
                                                      uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint ROWS_PER_SG = 2u;
    const uint u = tg_pos.y;
    if (u >= sched_hdr->num_unique_experts) return;
    const uint row0 = tg_pos.x * 8u + sg_id * ROWS_PER_SG;
    if (row0 >= p.intermediate_size) return;
    const uint s0 = expert_token_offsets[u];
    const uint s1 = expert_token_offsets[u + 1u];
    const uint count_u = s1 - s0;
    if (count_u == 0u) return;

    const InkExpertWeights ew = expert_table[unique_experts[u]];
    const device uchar *gate_qw = weights_base + ew.gate_qweight_offset;
    const device uchar *up_qw = (p.pair_interleaved != 0u) ? gate_qw : (weights_base + ew.up_qweight_offset);
    const uint blks_per_step = (INK_MOE_DTYPE == INK_DT_GGUF_Q2_0) ? 4u : 8u;

    for (uint t_base = 0u; t_base < count_u; ) {
        const uint rem = min(4u, count_u - t_base);
        const device half *x0 = x + ulong(token_indices[s0 + t_base + 0u]) * p.hidden_size;
        const device half *x1 = x + ulong(token_indices[s0 + t_base + min(1u, rem - 1u)]) * p.hidden_size;
        const device half *x2 = x + ulong(token_indices[s0 + t_base + min(2u, rem - 1u)]) * p.hidden_size;
        const device half *x3 = x + ulong(token_indices[s0 + t_base + min(3u, rem - 1u)]) * p.hidden_size;

        for (uint r = 0u; r < ROWS_PER_SG; r++) {
            const uint row = row0 + r;
            if (row >= p.intermediate_size) break;
            const ulong g_row = (p.pair_interleaved != 0u) ? (ulong(row) * 2ul) : ulong(row);
            const ulong u_row = (p.pair_interleaved != 0u) ? (ulong(row) * 2ul + 1ul) : ulong(row);
            const device uchar *gw_row = gate_qw + g_row * p.row_bytes;
            const device uchar *uw_row = up_qw + u_row * p.row_bytes;
            float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
            float u0 = 0.0f, u1 = 0.0f, u2 = 0.0f, u3 = 0.0f;

            for (uint blk_step = 0u; blk_step < p.num_groups; blk_step += blks_per_step) {
                uint k0, k1;
                float4 gw0, gw1, uw0, uw1;
                ink_gguf_blk_dequant8(gw_row, blk_step, lane, p.hidden_size, k0, k1, gw0, gw1);
                if (k0 >= p.hidden_size) continue;
                ink_gguf_blk_dequant8(uw_row, blk_step, lane, p.hidden_size, k0, k1, uw0, uw1);
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
                inter[ulong(s0 + t_base + 0u) * p.intermediate_size + row] = half(ink_eval_swiglu_softcap(sg0, su0, p.gate_softcap, p.up_softcap));
                if (rem > 1u) inter[ulong(s0 + t_base + 1u) * p.intermediate_size + row] = half(ink_eval_swiglu_softcap(sg1, su1, p.gate_softcap, p.up_softcap));
                if (rem > 2u) inter[ulong(s0 + t_base + 2u) * p.intermediate_size + row] = half(ink_eval_swiglu_softcap(sg2, su2, p.gate_softcap, p.up_softcap));
                if (rem > 3u) inter[ulong(s0 + t_base + 3u) * p.intermediate_size + row] = half(ink_eval_swiglu_softcap(sg3, su3, p.gate_softcap, p.up_softcap));
            }
        }
        t_base += rem;
    }
}

// 5. GGUF K-Quant Super-Blocks (QK = 256), Grid X: ceil(intermediate_size / 8).
static inline uint ink_gguf_k_bytes(uint dtype) {
    switch (dtype) {
        case INK_DT_GGUF_Q2_K: return 84u;
        case INK_DT_GGUF_Q3_K: return 110u;
        case INK_DT_GGUF_Q4_K: return 144u;
        case INK_DT_GGUF_Q5_K: return 176u;
        case INK_DT_GGUF_Q6_K: return 210u;
        case INK_DT_GGUF_IQ4_XS: return 136u;
        case INK_DT_GGUF_TQ1_0: return 54u;
        case INK_DT_GGUF_TQ2_0: return 66u;
        default: return 144u;
    }
}

static inline uchar2 ink_get_scale_min_k4(uint j, const device uchar *q) {
    if (j < 4u) return uchar2(q[j] & 63u, q[j + 4u] & 63u);
    return uchar2((q[j + 4u] & 0x0Fu) | ((q[j - 4u] >> 6) << 4),
                  (q[j + 4u] >> 4) | ((q[j] >> 6) << 4));
}

static inline int ink_q3k_scale_at(const device uchar *scales, uint is) {
    const uint a0 = ink_read_u32(scales + 0u);
    const uint a1 = ink_read_u32(scales + 4u);
    const uint a2 = ink_read_u32(scales + 8u);
    uint word;
    if (is < 4u) {
        word = (a0 & 0x0F0F0F0Fu) | (((a2 >> 0) & 0x03030303u) << 4);
    } else if (is < 8u) {
        word = (a1 & 0x0F0F0F0Fu) | (((a2 >> 2) & 0x03030303u) << 4);
    } else if (is < 12u) {
        word = ((a0 >> 4) & 0x0F0F0F0Fu) | (((a2 >> 4) & 0x03030303u) << 4);
    } else {
        word = ((a1 >> 4) & 0x0F0F0F0Fu) | (((a2 >> 6) & 0x03030303u) << 4);
    }
    return int((word >> ((is & 3u) * 8u)) & 0xFFu) - 32;
}

static inline void ink_gguf_k_dequant8(const device uchar *blk,
                                       uint sb_idx,
                                       uint lane,
                                       thread uint &k0,
                                       thread uint &k1,
                                       thread float4 &w0,
                                       thread float4 &w1) {
    if (INK_MOE_DTYPE == INK_DT_GGUF_IQ4_XS) {
        const uint ib32 = lane >> 2;
        const uint l = lane & 3u;
        k0 = sb_idx * 256u + ib32 * 32u + l * 4u;
        k1 = k0 + 16u;
        const float d = ink_read_f16(blk);
        const ushort scales_h = ink_read_u16(blk + 2u);
        const int ls = int(((blk[4u + (ib32 >> 1)] >> (4u * (ib32 & 1u))) & 0xFu) |
                           (((scales_h >> (2u * ib32)) & 3u) << 4));
        const float dl = d * float(ls - 32);
        const device uchar *qs = blk + 8u + ib32 * 16u + l * 4u;
        w0 = float4(float(INK_KVALUES_IQ4NL[qs[0] & 0xFu]), float(INK_KVALUES_IQ4NL[qs[1] & 0xFu]),
                    float(INK_KVALUES_IQ4NL[qs[2] & 0xFu]), float(INK_KVALUES_IQ4NL[qs[3] & 0xFu])) * dl;
        w1 = float4(float(INK_KVALUES_IQ4NL[qs[0] >> 4]), float(INK_KVALUES_IQ4NL[qs[1] >> 4]),
                    float(INK_KVALUES_IQ4NL[qs[2] >> 4]), float(INK_KVALUES_IQ4NL[qs[3] >> 4])) * dl;
        return;
    }

    k0 = sb_idx * 256u + lane * 8u;
    k1 = k0 + 4u;

    if (INK_MOE_DTYPE == INK_DT_GGUF_Q2_K) {
        const float d = ink_read_f16(blk + 80u);
        const float dmin = ink_read_f16(blk + 82u);
        const uchar sc = blk[lane >> 1];
        const float dl = d * float(sc & 0xFu);
        const float ml = dmin * float(sc >> 4);
        const uint shift = ((lane >> 2) & 3u) * 2u;
        const device uchar *q = blk + 16u + (lane >> 4) * 32u + ((lane >> 1) & 1u) * 16u + (lane & 1u) * 8u;
        w0 = float4(float((q[0] >> shift) & 3u), float((q[1] >> shift) & 3u),
                    float((q[2] >> shift) & 3u), float((q[3] >> shift) & 3u)) * dl - ml;
        w1 = float4(float((q[4] >> shift) & 3u), float((q[5] >> shift) & 3u),
                    float((q[6] >> shift) & 3u), float((q[7] >> shift) & 3u)) * dl - ml;
    } else if (INK_MOE_DTYPE == INK_DT_GGUF_Q3_K) {
        const float d_all = ink_read_f16(blk + 108u);
        const float dl = d_all * float(ink_q3k_scale_at(blk + 96u, lane >> 1));
        const uint shift = ((lane >> 2) & 3u) * 2u;
        const uint sub16 = (lane >> 1) & 1u;
        const uint oct = lane & 1u;
        const uint m_bit = 1u << (lane >> 2);
        const device uchar *hm = blk + sub16 * 16u + oct * 8u;
        const device uchar *q = blk + 32u + (lane >> 4) * 32u + sub16 * 16u + oct * 8u;
        w0 = float4(float(int((q[0] >> shift) & 3u) - ((hm[0] & m_bit) != 0u ? 0 : 4)),
                    float(int((q[1] >> shift) & 3u) - ((hm[1] & m_bit) != 0u ? 0 : 4)),
                    float(int((q[2] >> shift) & 3u) - ((hm[2] & m_bit) != 0u ? 0 : 4)),
                    float(int((q[3] >> shift) & 3u) - ((hm[3] & m_bit) != 0u ? 0 : 4))) * dl;
        w1 = float4(float(int((q[4] >> shift) & 3u) - ((hm[4] & m_bit) != 0u ? 0 : 4)),
                    float(int((q[5] >> shift) & 3u) - ((hm[5] & m_bit) != 0u ? 0 : 4)),
                    float(int((q[6] >> shift) & 3u) - ((hm[6] & m_bit) != 0u ? 0 : 4)),
                    float(int((q[7] >> shift) & 3u) - ((hm[7] & m_bit) != 0u ? 0 : 4))) * dl;
    } else if (INK_MOE_DTYPE == INK_DT_GGUF_Q4_K) {
        const float d = ink_read_f16(blk);
        const float dmin = ink_read_f16(blk + 2u);
        const uint blk64 = lane >> 3;
        const uint hi = (lane >> 2) & 1u;
        const uchar2 sc_m = ink_get_scale_min_k4(blk64 * 2u + hi, blk + 4u);
        const float dl = d * float(sc_m.x);
        const float ml = dmin * float(sc_m.y);
        const device uchar *q = blk + 16u + blk64 * 32u + (lane & 3u) * 8u;
        const uint sh = hi * 4u;
        w0 = float4(float((q[0] >> sh) & 0xFu), float((q[1] >> sh) & 0xFu),
                    float((q[2] >> sh) & 0xFu), float((q[3] >> sh) & 0xFu)) * dl - ml;
        w1 = float4(float((q[4] >> sh) & 0xFu), float((q[5] >> sh) & 0xFu),
                    float((q[6] >> sh) & 0xFu), float((q[7] >> sh) & 0xFu)) * dl - ml;
    } else if (INK_MOE_DTYPE == INK_DT_GGUF_Q5_K) {
        const float d = ink_read_f16(blk);
        const float dmin = ink_read_f16(blk + 2u);
        const uint blk64 = lane >> 3;
        const uint hi = (lane >> 2) & 1u;
        const uint oct = lane & 3u;
        const uint is = blk64 * 2u + hi;
        const uchar2 sc_m = ink_get_scale_min_k4(is, blk + 4u);
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
    } else if (INK_MOE_DTYPE == INK_DT_GGUF_Q6_K) {
        const float d = ink_read_f16(blk + 208u);
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
    } else if (INK_MOE_DTYPE == INK_DT_GGUF_TQ1_0) {
        const float d = ink_read_f16(blk + 52u);
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
    } else {
        const float d = ink_read_f16(blk + 64u);
        const uint sh = ((lane >> 2) & 3u) * 2u;
        const device uchar *q = blk + (lane >> 4) * 32u + (lane & 3u) * 8u;
        w0 = float4(float(int((q[0] >> sh) & 3u) - 1), float(int((q[1] >> sh) & 3u) - 1),
                    float(int((q[2] >> sh) & 3u) - 1), float(int((q[3] >> sh) & 3u) - 1)) * d;
        w1 = float4(float(int((q[4] >> sh) & 3u) - 1), float(int((q[5] >> sh) & 3u) - 1),
                    float(int((q[6] >> sh) & 3u) - 1), float(int((q[7] >> sh) & 3u) - 1)) * d;
    }
}

kernel void ami_inkling2_moe_gate_up_softcap_gguf_k(const device half *x [[buffer(0)]],
                                                    const device uchar *weights_base [[buffer(1)]],
                                                    const device InkExpertWeights *expert_table [[buffer(2)]],
                                                    const device InkMoeScheduleHeader *sched_hdr [[buffer(3)]],
                                                    const device uint *unique_experts [[buffer(4)]],
                                                    const device uint *expert_token_offsets [[buffer(5)]],
                                                    const device uint *token_indices [[buffer(6)]],
                                                    device half *inter [[buffer(7)]],
                                                    constant Inkling2MoeGateUpSoftcapParams &p [[buffer(8)]],
                                                    uint2 tg_pos [[threadgroup_position_in_grid]],
                                                    uint sg_id [[simdgroup_index_in_threadgroup]],
                                                    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint ROWS_PER_SG = 2u;
    const uint u = tg_pos.y;
    if (u >= sched_hdr->num_unique_experts) return;
    const uint row0 = tg_pos.x * 8u + sg_id * ROWS_PER_SG;
    if (row0 >= p.intermediate_size) return;
    const uint s0 = expert_token_offsets[u];
    const uint s1 = expert_token_offsets[u + 1u];
    const uint count_u = s1 - s0;
    if (count_u == 0u) return;

    const InkExpertWeights ew = expert_table[unique_experts[u]];
    const device uchar *gate_qw = weights_base + ew.gate_qweight_offset;
    const device uchar *up_qw = (p.pair_interleaved != 0u) ? gate_qw : (weights_base + ew.up_qweight_offset);
    const uint blk_bytes = ink_gguf_k_bytes(INK_MOE_DTYPE);

    for (uint t_base = 0u; t_base < count_u; ) {
        const uint rem = min(4u, count_u - t_base);
        const device half *x0 = x + ulong(token_indices[s0 + t_base + 0u]) * p.hidden_size;
        const device half *x1 = x + ulong(token_indices[s0 + t_base + min(1u, rem - 1u)]) * p.hidden_size;
        const device half *x2 = x + ulong(token_indices[s0 + t_base + min(2u, rem - 1u)]) * p.hidden_size;
        const device half *x3 = x + ulong(token_indices[s0 + t_base + min(3u, rem - 1u)]) * p.hidden_size;

        for (uint r = 0u; r < ROWS_PER_SG; r++) {
            const uint row = row0 + r;
            if (row >= p.intermediate_size) break;
            const ulong g_row = (p.pair_interleaved != 0u) ? (ulong(row) * 2ul) : ulong(row);
            const ulong u_row = (p.pair_interleaved != 0u) ? (ulong(row) * 2ul + 1ul) : ulong(row);
            const device uchar *gw_row = gate_qw + g_row * p.row_bytes;
            const device uchar *uw_row = up_qw + u_row * p.row_bytes;
            float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
            float u0 = 0.0f, u1 = 0.0f, u2 = 0.0f, u3 = 0.0f;

            for (uint sb = 0u; sb < p.num_groups; sb++) {
                uint k0, k1;
                float4 gw0, gw1, uw0, uw1;
                ink_gguf_k_dequant8(gw_row + ulong(sb) * blk_bytes, sb, lane, k0, k1, gw0, gw1);
                ink_gguf_k_dequant8(uw_row + ulong(sb) * blk_bytes, sb, lane, k0, k1, uw0, uw1);
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
                inter[ulong(s0 + t_base + 0u) * p.intermediate_size + row] = half(ink_eval_swiglu_softcap(sg0, su0, p.gate_softcap, p.up_softcap));
                if (rem > 1u) inter[ulong(s0 + t_base + 1u) * p.intermediate_size + row] = half(ink_eval_swiglu_softcap(sg1, su1, p.gate_softcap, p.up_softcap));
                if (rem > 2u) inter[ulong(s0 + t_base + 2u) * p.intermediate_size + row] = half(ink_eval_swiglu_softcap(sg2, su2, p.gate_softcap, p.up_softcap));
                if (rem > 3u) inter[ulong(s0 + t_base + 3u) * p.intermediate_size + row] = half(ink_eval_swiglu_softcap(sg3, su3, p.gate_softcap, p.up_softcap));
            }
        }
        t_base += rem;
    }
}

// 6. GGUF IQ Codebook-SRAM Formats (iq1_s..iq3_s), Grid X: ceil(intermediate_size / 8).
struct InkIqCodebookSmem {
    uint16_t grid16[4096];
    uint8_t signs[128];
};
static_assert(sizeof(InkIqCodebookSmem) == 8320u, "InkIqCodebookSmem must fit within 16 KiB");

static inline uint ink_iq_blk_bytes(uint dtype) {
    switch (dtype) {
        case INK_DT_GGUF_IQ1_S: return 50u;
        case INK_DT_GGUF_IQ1_M: return 56u;
        case INK_DT_GGUF_IQ2_XXS: return 66u;
        case INK_DT_GGUF_IQ2_XS: return 74u;
        case INK_DT_GGUF_IQ2_S: return 82u;
        case INK_DT_GGUF_IQ3_XXS: return 98u;
        case INK_DT_GGUF_IQ3_S: return 110u;
        default: return 66u;
    }
}

static inline uint ink_iq_grid_u16_count(uint dtype) {
    switch (dtype) {
        case INK_DT_GGUF_IQ1_S:
        case INK_DT_GGUF_IQ1_M:
        case INK_DT_GGUF_IQ2_S: return 4096u;
        case INK_DT_GGUF_IQ2_XS: return 2048u;
        case INK_DT_GGUF_IQ2_XXS:
        case INK_DT_GGUF_IQ3_S: return 1024u;
        case INK_DT_GGUF_IQ3_XXS: return 512u;
        default: return 1024u;
    }
}

static inline bool ink_iq_needs_signs(uint dtype) {
    return (dtype == INK_DT_GGUF_IQ2_XXS) || (dtype == INK_DT_GGUF_IQ2_XS) || (dtype == INK_DT_GGUF_IQ3_XXS);
}

static inline void ink_iq_stage_codebook(const device uchar *codebook,
                                         threadgroup InkIqCodebookSmem &smem,
                                         uint tid) {
    const device uint16_t *src16 = (const device uint16_t *)codebook;
    const uint n16 = ink_iq_grid_u16_count(INK_MOE_DTYPE);
    for (uint i = tid; i < n16; i += 128u) {
        smem.grid16[i] = src16[i];
    }
    if (ink_iq_needs_signs(INK_MOE_DTYPE)) {
        const device uchar *signs_src = codebook + ulong(n16) * 2ul;
        if (tid < 128u) {
            smem.signs[tid] = signs_src[tid];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

static inline uint ink_smem_read_u32(threadgroup const InkIqCodebookSmem &smem, uint u32_idx) {
    return uint(smem.grid16[u32_idx * 2u]) | (uint(smem.grid16[u32_idx * 2u + 1u]) << 16);
}

static inline void ink_iq_dequant8(const device uchar *blk,
                                   threadgroup const InkIqCodebookSmem &smem,
                                   uint lane,
                                   thread float4 &w0,
                                   thread float4 &w1) {
    const uint ib32 = lane >> 2;
    const uint l = lane & 3u;

    if (INK_MOE_DTYPE == INK_DT_GGUF_IQ1_S) {
        const float d = ink_read_f16(blk);
        const ushort qh = ink_read_u16(blk + 34u + ib32 * 2u);
        const float dl = d * float(2u * ((qh >> 12) & 7u) + 1u);
        const float shift = -1.0f + (((qh & 0x8000u) != 0u) ? -0.125f : 0.125f);
        const uint g = ink_smem_read_u32(smem, uint(blk[2u + lane]) | (((uint(qh) >> (3u * l)) & 7u) << 8));
        w0 = float4(float((g >> 0) & 0xFu) + shift, float((g >> 8) & 0xFu) + shift,
                    float((g >> 16) & 0xFu) + shift, float((g >> 24) & 0xFu) + shift) * dl;
        w1 = float4(float((g >> 4) & 0xFu) + shift, float((g >> 12) & 0xFu) + shift,
                    float((g >> 20) & 0xFu) + shift, float((g >> 28) & 0xFu) + shift) * dl;
    } else if (INK_MOE_DTYPE == INK_DT_GGUF_IQ1_M) {
        const ushort sc0 = ink_read_u16(blk + 48u);
        const ushort sc1 = ink_read_u16(blk + 50u);
        const ushort sc2 = ink_read_u16(blk + 52u);
        const ushort sc3 = ink_read_u16(blk + 54u);
        const ushort d_u16 = (sc0 >> 12) | ((sc1 >> 8) & 0x00F0u) | ((sc2 >> 4) & 0x0F00u) | (sc3 & 0xF000u);
        const float d = float(as_type<half>(d_u16));
        const ushort sc_ib = ink_read_u16(blk + 48u + (ib32 >> 1) * 2u);
        const float dl = d * float(2u * ((sc_ib >> (6u * (ib32 & 1u) + 3u * (l >> 1))) & 7u) + 1u);
        const uchar qh_b = blk[32u + ib32 * 2u + (l >> 1)];
        const float shift = -1.0f + (((qh_b & (((l & 1u) == 0u) ? 0x08u : 0x80u)) != 0u) ? -0.125f : 0.125f);
        const uint g = ink_smem_read_u32(smem, uint(blk[lane]) | ((uint(qh_b) << (((l & 1u) == 0u) ? 8u : 4u)) & 0x700u));
        w0 = float4(float((g >> 0) & 0xFu) + shift, float((g >> 8) & 0xFu) + shift,
                    float((g >> 16) & 0xFu) + shift, float((g >> 24) & 0xFu) + shift) * dl;
        w1 = float4(float((g >> 4) & 0xFu) + shift, float((g >> 12) & 0xFu) + shift,
                    float((g >> 20) & 0xFu) + shift, float((g >> 28) & 0xFu) + shift) * dl;
    } else if (INK_MOE_DTYPE == INK_DT_GGUF_IQ2_XXS) {
        const float d = ink_read_f16(blk);
        const uint aux0 = ink_read_u32(blk + 2u + ib32 * 8u);
        const uint aux1 = ink_read_u32(blk + 2u + ib32 * 8u + 4u);
        const float db = d * (0.5f + float(aux1 >> 28)) * 0.25f;
        const uint grid_idx = (aux0 >> (8u * l)) & 0xFFu;
        const uchar sg = smem.signs[(aux1 >> (7u * l)) & 127u];
        const uint g0 = ink_smem_read_u32(smem, grid_idx * 2u + 0u);
        const uint g1 = ink_smem_read_u32(smem, grid_idx * 2u + 1u);
        w0 = float4(float((g0 >> 0) & 0xFFu) * (((sg & 1u) != 0u) ? -db : db),
                    float((g0 >> 8) & 0xFFu) * (((sg & 2u) != 0u) ? -db : db),
                    float((g0 >> 16) & 0xFFu) * (((sg & 4u) != 0u) ? -db : db),
                    float((g0 >> 24) & 0xFFu) * (((sg & 8u) != 0u) ? -db : db));
        w1 = float4(float((g1 >> 0) & 0xFFu) * (((sg & 16u) != 0u) ? -db : db),
                    float((g1 >> 8) & 0xFFu) * (((sg & 32u) != 0u) ? -db : db),
                    float((g1 >> 16) & 0xFFu) * (((sg & 64u) != 0u) ? -db : db),
                    float((g1 >> 24) & 0xFFu) * (((sg & 128u) != 0u) ? -db : db));
    } else if (INK_MOE_DTYPE == INK_DT_GGUF_IQ2_XS) {
        const float d = ink_read_f16(blk);
        const ushort q2 = ink_read_u16(blk + 2u + lane * 2u);
        const uchar sc_b = blk[66u + ib32];
        const float db = d * (0.5f + float((sc_b >> (4u * (l >> 1))) & 0xFu)) * 0.25f;
        const uint grid_idx = q2 & 511u;
        const uchar sg = smem.signs[q2 >> 9];
        const uint g0 = ink_smem_read_u32(smem, grid_idx * 2u + 0u);
        const uint g1 = ink_smem_read_u32(smem, grid_idx * 2u + 1u);
        w0 = float4(float((g0 >> 0) & 0xFFu) * (((sg & 1u) != 0u) ? -db : db),
                    float((g0 >> 8) & 0xFFu) * (((sg & 2u) != 0u) ? -db : db),
                    float((g0 >> 16) & 0xFFu) * (((sg & 4u) != 0u) ? -db : db),
                    float((g0 >> 24) & 0xFFu) * (((sg & 8u) != 0u) ? -db : db));
        w1 = float4(float((g1 >> 0) & 0xFFu) * (((sg & 16u) != 0u) ? -db : db),
                    float((g1 >> 8) & 0xFFu) * (((sg & 32u) != 0u) ? -db : db),
                    float((g1 >> 16) & 0xFFu) * (((sg & 64u) != 0u) ? -db : db),
                    float((g1 >> 24) & 0xFFu) * (((sg & 128u) != 0u) ? -db : db));
    } else if (INK_MOE_DTYPE == INK_DT_GGUF_IQ2_S) {
        const float d = ink_read_f16(blk);
        const uchar sc_b = blk[74u + ib32];
        const float dl = d * (0.5f + float((sc_b >> (4u * (l >> 1))) & 0xFu)) * 0.25f;
        const uchar sg = blk[34u + lane];
        const uint grid_idx = uint(blk[2u + lane]) | ((uint(blk[66u + ib32]) << (8u - 2u * l)) & 0x300u);
        const uint g0 = ink_smem_read_u32(smem, grid_idx * 2u + 0u);
        const uint g1 = ink_smem_read_u32(smem, grid_idx * 2u + 1u);
        w0 = float4(float((g0 >> 0) & 0xFFu) * (((sg & 1u) != 0u) ? -dl : dl),
                    float((g0 >> 8) & 0xFFu) * (((sg & 2u) != 0u) ? -dl : dl),
                    float((g0 >> 16) & 0xFFu) * (((sg & 4u) != 0u) ? -dl : dl),
                    float((g0 >> 24) & 0xFFu) * (((sg & 8u) != 0u) ? -dl : dl));
        w1 = float4(float((g1 >> 0) & 0xFFu) * (((sg & 16u) != 0u) ? -dl : dl),
                    float((g1 >> 8) & 0xFFu) * (((sg & 32u) != 0u) ? -dl : dl),
                    float((g1 >> 16) & 0xFFu) * (((sg & 64u) != 0u) ? -dl : dl),
                    float((g1 >> 24) & 0xFFu) * (((sg & 128u) != 0u) ? -dl : dl));
    } else if (INK_MOE_DTYPE == INK_DT_GGUF_IQ3_XXS) {
        const float d = ink_read_f16(blk);
        const uint aux = ink_read_u32(blk + 66u + ib32 * 4u);
        const float db = d * (0.5f + float(aux >> 28)) * 0.5f;
        const uchar sg = smem.signs[(aux >> (7u * l)) & 127u];
        const uint g0 = ink_smem_read_u32(smem, blk[2u + lane * 2u + 0u]);
        const uint g1 = ink_smem_read_u32(smem, blk[2u + lane * 2u + 1u]);
        w0 = float4(float((g0 >> 0) & 0xFFu) * (((sg & 1u) != 0u) ? -db : db),
                    float((g0 >> 8) & 0xFFu) * (((sg & 2u) != 0u) ? -db : db),
                    float((g0 >> 16) & 0xFFu) * (((sg & 4u) != 0u) ? -db : db),
                    float((g0 >> 24) & 0xFFu) * (((sg & 8u) != 0u) ? -db : db));
        w1 = float4(float((g1 >> 0) & 0xFFu) * (((sg & 16u) != 0u) ? -db : db),
                    float((g1 >> 8) & 0xFFu) * (((sg & 32u) != 0u) ? -db : db),
                    float((g1 >> 16) & 0xFFu) * (((sg & 64u) != 0u) ? -db : db),
                    float((g1 >> 24) & 0xFFu) * (((sg & 128u) != 0u) ? -db : db));
    } else {
        const float d = ink_read_f16(blk);
        const uchar sc_b = blk[106u + (ib32 >> 1)];
        const float db = d * float(1u + 2u * ((sc_b >> (4u * (ib32 & 1u))) & 0xFu));
        const uchar qh_b = blk[66u + ib32];
        const uint idx1 = uint(blk[2u + lane * 2u + 0u]) | ((uint(qh_b) << (8u - 2u * l)) & 256u);
        const uint idx2 = uint(blk[2u + lane * 2u + 1u]) | ((uint(qh_b) << (7u - 2u * l)) & 256u);
        const uchar sg = blk[74u + lane];
        const uint g0 = ink_smem_read_u32(smem, idx1);
        const uint g1 = ink_smem_read_u32(smem, idx2);
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

kernel void ami_inkling2_moe_gate_up_softcap_iq(const device half *x [[buffer(0)]],
                                                const device uchar *weights_base [[buffer(1)]],
                                                const device InkExpertWeights *expert_table [[buffer(2)]],
                                                const device InkMoeScheduleHeader *sched_hdr [[buffer(3)]],
                                                const device uint *unique_experts [[buffer(4)]],
                                                const device uint *expert_token_offsets [[buffer(5)]],
                                                const device uint *token_indices [[buffer(6)]],
                                                device half *inter [[buffer(7)]],
                                                const device uchar *codebook [[buffer(8)]],
                                                constant Inkling2MoeGateUpSoftcapParams &p [[buffer(9)]],
                                                uint2 tg_pos [[threadgroup_position_in_grid]],
                                                uint tid [[thread_index_in_threadgroup]],
                                                uint sg_id [[simdgroup_index_in_threadgroup]],
                                                uint lane [[thread_index_in_simdgroup]]) {
    threadgroup InkIqCodebookSmem smem;
    ink_iq_stage_codebook(codebook, smem, tid);

    constexpr uint ROWS_PER_SG = 2u;
    const uint u = tg_pos.y;
    if (u >= sched_hdr->num_unique_experts) return;
    const uint row0 = tg_pos.x * 8u + sg_id * ROWS_PER_SG;
    if (row0 >= p.intermediate_size) return;
    const uint s0 = expert_token_offsets[u];
    const uint s1 = expert_token_offsets[u + 1u];
    const uint count_u = s1 - s0;
    if (count_u == 0u) return;

    const InkExpertWeights ew = expert_table[unique_experts[u]];
    const device uchar *gate_qw = weights_base + ew.gate_qweight_offset;
    const device uchar *up_qw = (p.pair_interleaved != 0u) ? gate_qw : (weights_base + ew.up_qweight_offset);
    const uint blk_bytes = ink_iq_blk_bytes(INK_MOE_DTYPE);

    for (uint t_base = 0u; t_base < count_u; ) {
        const uint rem = min(4u, count_u - t_base);
        const device half *x0 = x + ulong(token_indices[s0 + t_base + 0u]) * p.hidden_size;
        const device half *x1 = x + ulong(token_indices[s0 + t_base + min(1u, rem - 1u)]) * p.hidden_size;
        const device half *x2 = x + ulong(token_indices[s0 + t_base + min(2u, rem - 1u)]) * p.hidden_size;
        const device half *x3 = x + ulong(token_indices[s0 + t_base + min(3u, rem - 1u)]) * p.hidden_size;

        for (uint r = 0u; r < ROWS_PER_SG; r++) {
            const uint row = row0 + r;
            if (row >= p.intermediate_size) break;
            const ulong g_row = (p.pair_interleaved != 0u) ? (ulong(row) * 2ul) : ulong(row);
            const ulong u_row = (p.pair_interleaved != 0u) ? (ulong(row) * 2ul + 1ul) : ulong(row);
            const device uchar *gw_row = gate_qw + g_row * p.row_bytes;
            const device uchar *uw_row = up_qw + u_row * p.row_bytes;
            float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
            float u0 = 0.0f, u1 = 0.0f, u2 = 0.0f, u3 = 0.0f;

            for (uint sb = 0u; sb < p.num_groups; sb++) {
                const uint k_off = sb * 256u + lane * 8u;
                float4 gw0, gw1, uw0, uw1;
                ink_iq_dequant8(gw_row + ulong(sb) * blk_bytes, smem, lane, gw0, gw1);
                ink_iq_dequant8(uw_row + ulong(sb) * blk_bytes, smem, lane, uw0, uw1);
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
                inter[ulong(s0 + t_base + 0u) * p.intermediate_size + row] = half(ink_eval_swiglu_softcap(sg0, su0, p.gate_softcap, p.up_softcap));
                if (rem > 1u) inter[ulong(s0 + t_base + 1u) * p.intermediate_size + row] = half(ink_eval_swiglu_softcap(sg1, su1, p.gate_softcap, p.up_softcap));
                if (rem > 2u) inter[ulong(s0 + t_base + 2u) * p.intermediate_size + row] = half(ink_eval_swiglu_softcap(sg2, su2, p.gate_softcap, p.up_softcap));
                if (rem > 3u) inter[ulong(s0 + t_base + 3u) * p.intermediate_size + row] = half(ink_eval_swiglu_softcap(sg3, su3, p.gate_softcap, p.up_softcap));
            }
        }
        t_base += rem;
    }
}
