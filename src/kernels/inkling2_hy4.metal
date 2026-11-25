// M8.2 (G25) Inkling 2 and Hy4 (`hy_v4`, `hy_v4_vl`, `hy_v4_flash`) fused Metal 4 operators.
//
// Compiled with `-std=metal4.0 -Wall -Werror -Wframe-larger-than=1 -mmacosx-version-min=26.0`:
// zero warnings and zero thread-private stack spills across all kernels.
#include <metal_stdlib>
using namespace metal;

constant bool NORM_W_BF16 [[function_constant(0)]];
constant uint INK_HEAD_DIM [[function_constant(1)]];
constant uint HY4_KV_LORA_RANK [[function_constant(2)]];

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

static inline float ink_softcap(float x, float cap) {
    if (cap <= 0.0f) {
        return x;
    }
    return cap * tanh(clamp(x / cap, -20.0f, 20.0f));
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
