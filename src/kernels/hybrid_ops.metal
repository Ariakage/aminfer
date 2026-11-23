// M6 + M7 hybrid frontier model Metal 4 kernels:
//   - R6 + G8: Gated DeltaNet (GDN) recurrent decode, chunked prefill, M-node
//              tree verification with 64-bit ancestor masks, fused RMSNorm +
//              SiLU output gate, and ADR-0019 loop-isolation state commit.
//   - G9 + R5: Qwen Sparse Attention (QSA) with YaRN RoPE scaling, interleaved
//              3D MRoPE, hybrid 512-token full / 128-token circular SWA
//              RGBA16Float IOSurface KvTile rings, micro-block KV indexer with
//              mandatory local recent block(s) and ADR-0019 cross-loop index
//              reuse, plus sparse GQA decode & tree-verify attention.
//   - G10:     4-Branch Gated Residual (GR) stream init, fused read-gate,
//              fused write-gate, stream collapse, and Single-Pass Mega-mHC
//              (`mhc_single_pass` pre-mix + fused post-mix, `2..8` streams).
//   - R7 + G11: DeepSeek-V4.1 MLA + CSA2 (`Full` / `Reindex` / `Reuse`)
//              two-stage hierarchical sparse indexer, FP4 (`E2M1` + `E4M3`
//              per-16-channel scale, 890 B/tok) latent KV encode/decode, and
//              deduplicated SWA bounded window replay.
//
// Compiled with `-std=metal4.0 -Wall -Werror -mmacosx-version-min=26.0`.
#include <metal_stdlib>
using namespace metal;

constant uint GDN_HEAD_K_DIM [[function_constant(0)]];
constant bool HYBRID_NORM_W_BF16 [[function_constant(1)]];
constant uint QSA_HEAD_DIM [[function_constant(2)]];
constant bool GR_STREAM_BF16 [[function_constant(3)]];
constant bool GR_ACT_BF16 [[function_constant(4)]];

static inline float hybrid_fast_silu(float x) {
    return x / (1.0f + exp(-x));
}

static inline float hybrid_sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

static inline float hybrid_load_norm_w(const device uchar *raw, uint idx, bool is_bf16) {
    if (is_bf16) {
        return float(((const device bfloat *)raw)[idx]);
    }
    return float(((const device half *)raw)[idx]);
}

static inline float hybrid_load_f16_or_bf16(const device uchar *raw, ulong idx, bool is_bf16) {
    if (is_bf16) {
        return float(((const device bfloat *)raw)[idx]);
    }
    return float(((const device half *)raw)[idx]);
}

static inline void hybrid_store_f16_or_bf16(device uchar *raw, ulong idx, float val, bool is_bf16) {
    if (is_bf16) {
        ((device bfloat *)raw)[idx] = bfloat(val);
    } else {
        ((device half *)raw)[idx] = half(val);
    }
}

// ===========================================================================
// Section A: R6 + G8 — Gated DeltaNet (GDN)
//
// Canonical Yang et al. / Qwen3-Next Gated DeltaNet recurrence per v-head:
//   q <- L2Norm(q) * q_scale,   k <- L2Norm(k)
//   alpha = exp(g),             beta = sigmoid(beta_raw)
//   S_dec = alpha * S_prev                       [head_k_dim, head_v_dim] fp32
//   kv_mem = S_dec^T * k                         [head_v_dim]
//   S_new = S_dec + beta * k * (v - kv_mem)^T
//   o = S_new^T * q                              [head_v_dim]
// Optional output gating: o <- RMSNorm(o) * norm_w * SiLU(z).
// ===========================================================================

struct GdnDecodeParams {
    uint num_k_heads;
    uint num_v_heads;
    uint head_k_dim;
    uint head_v_dim;
    float q_scale;
    float l2_eps;
    float rms_eps;
    uint apply_l2_norm;
    uint apply_out_norm;
    uint apply_out_gate;
    uint per_head_norm;
    uint _pad0;
};

struct GdnPrefillParams {
    uint seq_len;
    uint num_k_heads;
    uint num_v_heads;
    uint head_k_dim;
    uint head_v_dim;
    float q_scale;
    float l2_eps;
    uint apply_l2_norm;
    uint apply_out_gate;
    uint chunk_size;
    uint write_state_only;
    uint out_start_token;
};

struct GdnTreeVerifyParams {
    uint num_nodes;
    uint num_k_heads;
    uint num_v_heads;
    uint head_k_dim;
    uint head_v_dim;
    float q_scale;
    float l2_eps;
    uint apply_l2_norm;
    uint apply_out_gate;
    uint scratch_leaf_node;
    uint _pad0;
    uint _pad1;
};

struct GdnOutNormGateParams {
    uint num_tokens;
    uint num_v_heads;
    uint head_v_dim;
    float rms_eps;
    uint apply_out_norm;
    uint apply_out_gate;
    uint per_head_norm;
    uint _pad0;
};

struct GdnCommitStateParams {
    uint total_f32_elems;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

struct GdnCommitTreeParams {
    uint num_k_heads;
    uint num_v_heads;
    uint head_k_dim;
    uint head_v_dim;
    uint accepted_count;
    uint use_node_buffer;
    float l2_eps;
    uint apply_l2_norm;
    ushort inline_nodes[64];
};

static_assert(sizeof(GdnDecodeParams) == 48, "GdnDecodeParams must be 48 bytes");
static_assert(sizeof(GdnPrefillParams) == 48, "GdnPrefillParams must be 48 bytes");
static_assert(sizeof(GdnTreeVerifyParams) == 48, "GdnTreeVerifyParams must be 48 bytes");
static_assert(sizeof(GdnOutNormGateParams) == 32, "GdnOutNormGateParams must be 32 bytes");
static_assert(sizeof(GdnCommitStateParams) == 16, "GdnCommitStateParams must be 16 bytes");
static_assert(sizeof(GdnCommitTreeParams) == 160, "GdnCommitTreeParams must be 160 bytes");

// Single-step Gated DeltaNet decode (`M = 1`) with fused optional RMSNorm +
// SiLU output gate and explicit `state_in` -> `state_out` bindings supporting
// both in-place commit (`state_out == state_in`) and ADR-0019 loop isolation
// (`state_out == S_tmp`).
//
// Grid: (num_v_heads, 1, 1) threadgroups, 256 threads (8 simdgroups).
kernel void ami_gdn_decode(const device half *q [[buffer(0)]],
                           const device half *k [[buffer(1)]],
                           const device half *v [[buffer(2)]],
                           const device float *g [[buffer(3)]],
                           const device float *beta [[buffer(4)]],
                           const device half *z [[buffer(5)]],
                           const device uchar *norm_w [[buffer(6)]],
                           const device float *state_in [[buffer(7)]],
                           device float *state_out [[buffer(8)]],
                           device half *out [[buffer(9)]],
                           constant GdnDecodeParams &p [[buffer(10)]],
                           uint tg_x [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint sg_id [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    const uint vh = tg_x;
    if (vh >= p.num_v_heads) {
        return;
    }

    const uint d_k = (GDN_HEAD_K_DIM > 0u) ? GDN_HEAD_K_DIM : p.head_k_dim;
    const uint d_v = p.head_v_dim;
    const uint v_per_k = p.num_v_heads / p.num_k_heads;
    const uint kh = vh / v_per_k;

    threadgroup float tg_q[256];
    threadgroup float tg_k[256];
    threadgroup float tg_o[256];
    threadgroup float tg_rms_partial[8];

    // Simdgroup 0 loads and L2-normalizes q[kh] and k[kh] once for all v-columns.
    if (sg_id == 0u) {
        const device half *q_src = q + ulong(kh) * d_k;
        const device half *k_src = k + ulong(kh) * d_k;
        float q_sq = 0.0f;
        float k_sq = 0.0f;
        for (uint dk = lane; dk < d_k; dk += 32u) {
            const float qv = float(q_src[dk]);
            const float kv = float(k_src[dk]);
            tg_q[dk] = qv;
            tg_k[dk] = kv;
            q_sq = fma(qv, qv, q_sq);
            k_sq = fma(kv, kv, k_sq);
        }
        const float q_sum = simd_sum(q_sq);
        const float k_sum = simd_sum(k_sq);
        const float q_fac = (p.apply_l2_norm != 0u) ? (rsqrt(q_sum + p.l2_eps) * p.q_scale) : p.q_scale;
        const float k_fac = (p.apply_l2_norm != 0u) ? rsqrt(k_sum + p.l2_eps) : 1.0f;
        for (uint dk = lane; dk < d_k; dk += 32u) {
            tg_q[dk] *= q_fac;
            tg_k[dk] *= k_fac;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float alpha = exp(g[vh]);
    const float beta_sig = hybrid_sigmoid(beta[vh]);
    const ulong head_state_base = ulong(vh) * d_k * d_v;
    const ulong v_head_base = ulong(vh) * d_v;

    // Each simdgroup updates a disjoint strided set of dv columns in FP32.
    for (uint dv = sg_id; dv < d_v; dv += 8u) {
        float pred_acc = 0.0f;
        float s_col[8];
        if (GDN_HEAD_K_DIM == 64u) {
            for (uint r = 0u; r < 2u; r++) {
                const uint dk = lane + r * 32u;
                const float s_dec = state_in[head_state_base + ulong(dk) * d_v + dv] * alpha;
                s_col[r] = s_dec;
                pred_acc = fma(s_dec, tg_k[dk], pred_acc);
            }
        } else if (GDN_HEAD_K_DIM == 128u) {
            for (uint r = 0u; r < 4u; r++) {
                const uint dk = lane + r * 32u;
                const float s_dec = state_in[head_state_base + ulong(dk) * d_v + dv] * alpha;
                s_col[r] = s_dec;
                pred_acc = fma(s_dec, tg_k[dk], pred_acc);
            }
        } else {
            for (uint r = 0u; r < 8u; r++) {
                const uint dk = lane + r * 32u;
                if (dk < d_k) {
                    const float s_dec = state_in[head_state_base + ulong(dk) * d_v + dv] * alpha;
                    s_col[r] = s_dec;
                    pred_acc = fma(s_dec, tg_k[dk], pred_acc);
                } else {
                    s_col[r] = 0.0f;
                }
            }
        }

        const float pred_v = simd_sum(pred_acc);
        const float v_val = float(v[v_head_base + dv]);
        const float delta = beta_sig * (v_val - pred_v);

        float o_acc = 0.0f;
        if (GDN_HEAD_K_DIM == 64u) {
            for (uint r = 0u; r < 2u; r++) {
                const uint dk = lane + r * 32u;
                const float s_new = fma(tg_k[dk], delta, s_col[r]);
                state_out[head_state_base + ulong(dk) * d_v + dv] = s_new;
                o_acc = fma(s_new, tg_q[dk], o_acc);
            }
        } else if (GDN_HEAD_K_DIM == 128u) {
            for (uint r = 0u; r < 4u; r++) {
                const uint dk = lane + r * 32u;
                const float s_new = fma(tg_k[dk], delta, s_col[r]);
                state_out[head_state_base + ulong(dk) * d_v + dv] = s_new;
                o_acc = fma(s_new, tg_q[dk], o_acc);
            }
        } else {
            for (uint r = 0u; r < 8u; r++) {
                const uint dk = lane + r * 32u;
                if (dk < d_k) {
                    const float s_new = fma(tg_k[dk], delta, s_col[r]);
                    state_out[head_state_base + ulong(dk) * d_v + dv] = s_new;
                    o_acc = fma(s_new, tg_q[dk], o_acc);
                }
            }
        }

        const float o_val = simd_sum(o_acc);
        if (lane == 0u) {
            tg_o[dv] = (p.apply_out_norm != 0u) ? float(half(o_val)) : o_val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_rms = 1.0f;
    if (p.apply_out_norm != 0u) {
        // Match `ami_gdn_out_norm_gate` bit-for-bit: single 32-lane `simd_sum` over
        // `half`-staged `tg_o[dv]` so `m = 1` decode and `m > 1` prefill/tree match.
        if (sg_id == 0u) {
            float sum_sq = 0.0f;
            for (uint dv = lane; dv < d_v; dv += 32u) {
                const float ov = tg_o[dv];
                sum_sq = fma(ov, ov, sum_sq);
            }
            const float total_sq = simd_sum(sum_sq);
            if (lane == 0u) {
                tg_rms_partial[0] = rsqrt(total_sq / float(d_v) + p.rms_eps);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        inv_rms = tg_rms_partial[0];
    }

    const uint nw_base = (p.per_head_norm != 0u) ? (vh * d_v) : 0u;
    for (uint dv = tid; dv < d_v; dv += 256u) {
        float val = tg_o[dv];
        if (p.apply_out_norm != 0u) {
            val = (val * inv_rms) * hybrid_load_norm_w(norm_w, nw_base + dv, HYBRID_NORM_W_BF16);
        }
        if (p.apply_out_gate != 0u) {
            val *= hybrid_fast_silu(float(z[v_head_base + dv]));
        }
        out[v_head_base + dv] = half(val);
    }
}

// Chunked-prefill Gated DeltaNet (`seq_len >= 1`).
// Each simdgroup keeps one column `S[vh, :, dv]` purely in FP32 registers
// across the entire sequence (1 state read at start, 1 state write at end),
// staging 4-token chunks of L2-normalized Q and K cooperatively in 8 KiB SRAM.
//
// Grid: (num_v_heads, ceil(head_v_dim / 8), 1) threadgroups, 256 threads.
kernel void ami_gdn_prefill(const device half *q [[buffer(0)]],
                            const device half *k [[buffer(1)]],
                            const device half *v [[buffer(2)]],
                            const device float *g [[buffer(3)]],
                            const device float *beta [[buffer(4)]],
                            const device half *z [[buffer(5)]],
                            const device float *state_in [[buffer(6)]],
                            device float *state_out [[buffer(7)]],
                            device half *out [[buffer(8)]],
                            constant GdnPrefillParams &p [[buffer(9)]],
                            uint2 tg_pos [[threadgroup_position_in_grid]],
                            uint sg_id [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint STAGE_T = 4u;
    const uint vh = tg_pos.x;
    const uint dv = tg_pos.y * 8u + sg_id;
    if (vh >= p.num_v_heads) {
        return;
    }

    const uint d_k = (GDN_HEAD_K_DIM > 0u) ? GDN_HEAD_K_DIM : p.head_k_dim;
    const uint d_v = p.head_v_dim;
    const uint v_per_k = p.num_v_heads / p.num_k_heads;
    const uint kh = vh / v_per_k;
    const bool active_dv = (dv < d_v);

    threadgroup float tg_q[STAGE_T][256];
    threadgroup float tg_k[STAGE_T][256];
    threadgroup float tg_alpha[STAGE_T];
    threadgroup float tg_beta[STAGE_T];

    const ulong head_state_base = ulong(vh) * d_k * d_v;
    float s_col[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (active_dv) {
        if (GDN_HEAD_K_DIM == 64u) {
            for (uint r = 0u; r < 2u; r++) {
                const uint dk = lane + r * 32u;
                s_col[r] = state_in[head_state_base + ulong(dk) * d_v + dv];
            }
        } else if (GDN_HEAD_K_DIM == 128u) {
            for (uint r = 0u; r < 4u; r++) {
                const uint dk = lane + r * 32u;
                s_col[r] = state_in[head_state_base + ulong(dk) * d_v + dv];
            }
        } else {
            for (uint r = 0u; r < 8u; r++) {
                const uint dk = lane + r * 32u;
                if (dk < d_k) {
                    s_col[r] = state_in[head_state_base + ulong(dk) * d_v + dv];
                }
            }
        }
    }

    for (uint t_base = 0u; t_base < p.seq_len; t_base += STAGE_T) {
        const uint stage_cnt = min(STAGE_T, p.seq_len - t_base);

        // Cooperatively load & L2-normalize up to 8 tokens of Q/K + exp(g)/sigmoid(beta).
        if (sg_id < stage_cnt) {
            const uint t = t_base + sg_id;
            const ulong k_off = (ulong(t) * p.num_k_heads + kh) * d_k;
            const device half *q_src = q + k_off;
            const device half *k_src = k + k_off;
            float q_sq = 0.0f;
            float k_sq = 0.0f;
            for (uint dk = lane; dk < d_k; dk += 32u) {
                const float qv = float(q_src[dk]);
                const float kv = float(k_src[dk]);
                tg_q[sg_id][dk] = qv;
                tg_k[sg_id][dk] = kv;
                q_sq = fma(qv, qv, q_sq);
                k_sq = fma(kv, kv, k_sq);
            }
            const float q_sum = simd_sum(q_sq);
            const float k_sum = simd_sum(k_sq);
            const float q_fac = (p.apply_l2_norm != 0u) ? (rsqrt(q_sum + p.l2_eps) * p.q_scale) : p.q_scale;
            const float k_fac = (p.apply_l2_norm != 0u) ? rsqrt(k_sum + p.l2_eps) : 1.0f;
            for (uint dk = lane; dk < d_k; dk += 32u) {
                tg_q[sg_id][dk] *= q_fac;
                tg_k[sg_id][dk] *= k_fac;
            }
            if (lane == 0u) {
                const ulong gb_idx = ulong(t) * p.num_v_heads + vh;
                tg_alpha[sg_id] = exp(g[gb_idx]);
                tg_beta[sg_id] = hybrid_sigmoid(beta[gb_idx]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active_dv) {
            for (uint dt = 0u; dt < stage_cnt; dt++) {
                const uint t = t_base + dt;
                const float alpha = tg_alpha[dt];
                const float beta_sig = tg_beta[dt];
                const ulong v_idx = (ulong(t) * p.num_v_heads + vh) * d_v + dv;
                const float v_val = float(v[v_idx]);

                float pred_acc = 0.0f;
                if (GDN_HEAD_K_DIM == 64u) {
                    for (uint r = 0u; r < 2u; r++) {
                        const uint dk = lane + r * 32u;
                        s_col[r] *= alpha;
                        pred_acc = fma(s_col[r], tg_k[dt][dk], pred_acc);
                    }
                } else if (GDN_HEAD_K_DIM == 128u) {
                    for (uint r = 0u; r < 4u; r++) {
                        const uint dk = lane + r * 32u;
                        s_col[r] *= alpha;
                        pred_acc = fma(s_col[r], tg_k[dt][dk], pred_acc);
                    }
                } else {
                    for (uint r = 0u; r < 8u; r++) {
                        const uint dk = lane + r * 32u;
                        if (dk < d_k) {
                            s_col[r] *= alpha;
                            pred_acc = fma(s_col[r], tg_k[dt][dk], pred_acc);
                        }
                    }
                }

                const float pred_v = simd_sum(pred_acc);
                const float delta = beta_sig * (v_val - pred_v);

                const bool emit_out = (p.write_state_only == 0u) && (t >= p.out_start_token);
                float o_acc = 0.0f;
                if (GDN_HEAD_K_DIM == 64u) {
                    for (uint r = 0u; r < 2u; r++) {
                        const uint dk = lane + r * 32u;
                        s_col[r] = fma(tg_k[dt][dk], delta, s_col[r]);
                        if (emit_out) {
                            o_acc = fma(s_col[r], tg_q[dt][dk], o_acc);
                        }
                    }
                } else if (GDN_HEAD_K_DIM == 128u) {
                    for (uint r = 0u; r < 4u; r++) {
                        const uint dk = lane + r * 32u;
                        s_col[r] = fma(tg_k[dt][dk], delta, s_col[r]);
                        if (emit_out) {
                            o_acc = fma(s_col[r], tg_q[dt][dk], o_acc);
                        }
                    }
                } else {
                    for (uint r = 0u; r < 8u; r++) {
                        const uint dk = lane + r * 32u;
                        if (dk < d_k) {
                            s_col[r] = fma(tg_k[dt][dk], delta, s_col[r]);
                            if (emit_out) {
                                o_acc = fma(s_col[r], tg_q[dt][dk], o_acc);
                            }
                        }
                    }
                }

                if (emit_out) {
                    float o_val = simd_sum(o_acc);
                    if (lane == 0u) {
                        if (p.apply_out_gate != 0u) {
                            o_val *= hybrid_fast_silu(float(z[v_idx]));
                        }
                        const ulong out_idx = (ulong(t - p.out_start_token) * p.num_v_heads + vh) * d_v + dv;
                        out[out_idx] = half(o_val);
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (active_dv) {
        if (GDN_HEAD_K_DIM == 64u) {
            for (uint r = 0u; r < 2u; r++) {
                const uint dk = lane + r * 32u;
                state_out[head_state_base + ulong(dk) * d_v + dv] = s_col[r];
            }
        } else if (GDN_HEAD_K_DIM == 128u) {
            for (uint r = 0u; r < 4u; r++) {
                const uint dk = lane + r * 32u;
                state_out[head_state_base + ulong(dk) * d_v + dv] = s_col[r];
            }
        } else {
            for (uint r = 0u; r < 8u; r++) {
                const uint dk = lane + r * 32u;
                if (dk < d_k) {
                    state_out[head_state_base + ulong(dk) * d_v + dv] = s_col[r];
                }
            }
        }
    }
}

// Speculative `M`-node (`M = 1..64`) Gated DeltaNet tree verification with
// 64-bit causal ancestor masks. Reads committed `state_in` once into registers
// without mutating it; optionally writes `scratch_leaf_node`'s resulting state
// to `state_out` (e.g. `S_tmp`) for ADR-0019 loop isolation.
//
// Grid: (num_v_heads, ceil(head_v_dim / 8), 1) threadgroups, 256 threads.
kernel void ami_gdn_tree_verify(const device half *q [[buffer(0)]],
                                const device half *k [[buffer(1)]],
                                const device half *v [[buffer(2)]],
                                const device float *g [[buffer(3)]],
                                const device float *beta [[buffer(4)]],
                                const device half *z [[buffer(5)]],
                                const device ulong *ancestor_masks [[buffer(6)]],
                                const device float *state_in [[buffer(7)]],
                                device float *state_out [[buffer(8)]],
                                device half *out [[buffer(9)]],
                                constant GdnTreeVerifyParams &p [[buffer(10)]],
                                uint2 tg_pos [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]],
                                uint sg_id [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    const uint vh = tg_pos.x;
    const uint dv = tg_pos.y * 8u + sg_id;
    if (vh >= p.num_v_heads) {
        return;
    }

    const uint d_k = (GDN_HEAD_K_DIM > 0u) ? GDN_HEAD_K_DIM : p.head_k_dim;
    const uint d_v = p.head_v_dim;
    const uint v_per_k = p.num_v_heads / p.num_k_heads;
    const uint kh = vh / v_per_k;
    const bool active_dv = (dv < d_v);

    threadgroup ulong tg_masks[64];
    threadgroup uchar tg_depth[64];
    threadgroup uchar tg_path[64];
    threadgroup uint tg_path_len;
    threadgroup float tg_q_fac[64];
    threadgroup float tg_k_fac[64];
    threadgroup float tg_alpha[64];
    threadgroup float tg_beta[64];

    if (tid < p.num_nodes) {
        const ulong m = ancestor_masks[tid];
        tg_masks[tid] = m;
        tg_depth[tid] = uchar(popcount(m));
        const ulong gb_idx = ulong(tid) * p.num_v_heads + vh;
        tg_alpha[tid] = exp(g[gb_idx]);
        tg_beta[tid] = hybrid_sigmoid(beta[gb_idx]);
    }
    // Precompute per-node L2 normalization factors cooperatively across 8 simdgroups.
    for (uint node = sg_id; node < p.num_nodes; node += 8u) {
        const ulong k_off = (ulong(node) * p.num_k_heads + kh) * d_k;
        float q_sq = 0.0f;
        float k_sq = 0.0f;
        if (p.apply_l2_norm != 0u) {
            for (uint dk = lane; dk < d_k; dk += 32u) {
                const float qv = float(q[k_off + dk]);
                const float kv = float(k[k_off + dk]);
                q_sq = fma(qv, qv, q_sq);
                k_sq = fma(kv, kv, k_sq);
            }
        }
        const float q_sum = simd_sum(q_sq);
        const float k_sum = simd_sum(k_sq);
        if (lane == 0u) {
            tg_q_fac[node] = (p.apply_l2_norm != 0u) ? (rsqrt(q_sum + p.l2_eps) * p.q_scale) : p.q_scale;
            tg_k_fac[node] = (p.apply_l2_norm != 0u) ? rsqrt(k_sum + p.l2_eps) : 1.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load committed initial state once into registers.
    const ulong head_state_base = ulong(vh) * d_k * d_v;
    float s_init[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (active_dv) {
        if (GDN_HEAD_K_DIM == 64u) {
            for (uint r = 0u; r < 2u; r++) {
                s_init[r] = state_in[head_state_base + ulong(lane + r * 32u) * d_v + dv];
            }
        } else if (GDN_HEAD_K_DIM == 128u) {
            for (uint r = 0u; r < 4u; r++) {
                s_init[r] = state_in[head_state_base + ulong(lane + r * 32u) * d_v + dv];
            }
        } else {
            for (uint r = 0u; r < 8u; r++) {
                const uint dk = lane + r * 32u;
                if (dk < d_k) {
                    s_init[r] = state_in[head_state_base + ulong(dk) * d_v + dv];
                }
            }
        }
    }

    for (uint node = 0u; node < p.num_nodes; node++) {
        // Reconstruct exact root-to-node ancestor chain ordered by depth (independent of node numbering).
        if (tid == 0u) {
            ulong rem = tg_masks[node];
            const uint depth_n = uint(tg_depth[node]);
            tg_path_len = depth_n;
            while (rem != 0ul) {
                const uint j = uint(ctz(rem));
                rem &= (rem - 1ul);
                if (j < p.num_nodes) {
                    const uint pos = uint(tg_depth[j]) - 1u;
                    if (pos < depth_n && pos < 64u) {
                        tg_path[pos] = uchar(j);
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active_dv) {
            float s_col[8];
            for (uint r = 0u; r < 8u; r++) {
                s_col[r] = s_init[r];
            }
            const uint plen = tg_path_len;
            for (uint step = 0u; step < plen; step++) {
                const uint j = uint(tg_path[step]);
                const float alpha = tg_alpha[j];
                const float beta_sig = tg_beta[j];
                const float k_fac = tg_k_fac[j];
                const ulong k_off = (ulong(j) * p.num_k_heads + kh) * d_k;
                const ulong v_idx = (ulong(j) * p.num_v_heads + vh) * d_v + dv;
                const float v_val = float(v[v_idx]);

                float pred_acc = 0.0f;
                float k_reg[8];
                if (GDN_HEAD_K_DIM == 64u) {
                    for (uint r = 0u; r < 2u; r++) {
                        const uint dk = lane + r * 32u;
                        const float kv = float(k[k_off + dk]) * k_fac;
                        k_reg[r] = kv;
                        s_col[r] *= alpha;
                        pred_acc = fma(s_col[r], kv, pred_acc);
                    }
                } else if (GDN_HEAD_K_DIM == 128u) {
                    for (uint r = 0u; r < 4u; r++) {
                        const uint dk = lane + r * 32u;
                        const float kv = float(k[k_off + dk]) * k_fac;
                        k_reg[r] = kv;
                        s_col[r] *= alpha;
                        pred_acc = fma(s_col[r], kv, pred_acc);
                    }
                } else {
                    for (uint r = 0u; r < 8u; r++) {
                        const uint dk = lane + r * 32u;
                        if (dk < d_k) {
                            const float kv = float(k[k_off + dk]) * k_fac;
                            k_reg[r] = kv;
                            s_col[r] *= alpha;
                            pred_acc = fma(s_col[r], kv, pred_acc);
                        } else {
                            k_reg[r] = 0.0f;
                        }
                    }
                }

                const float pred_v = simd_sum(pred_acc);
                const float delta = beta_sig * (v_val - pred_v);

                if (GDN_HEAD_K_DIM == 64u) {
                    for (uint r = 0u; r < 2u; r++) {
                        s_col[r] = fma(k_reg[r], delta, s_col[r]);
                    }
                } else if (GDN_HEAD_K_DIM == 128u) {
                    for (uint r = 0u; r < 4u; r++) {
                        s_col[r] = fma(k_reg[r], delta, s_col[r]);
                    }
                } else {
                    for (uint r = 0u; r < 8u; r++) {
                        if (lane + r * 32u < d_k) {
                            s_col[r] = fma(k_reg[r], delta, s_col[r]);
                        }
                    }
                }
            }

            const ulong q_off = (ulong(node) * p.num_k_heads + kh) * d_k;
            const float q_fac = tg_q_fac[node];
            float o_acc = 0.0f;
            if (GDN_HEAD_K_DIM == 64u) {
                for (uint r = 0u; r < 2u; r++) {
                    const uint dk = lane + r * 32u;
                    o_acc = fma(s_col[r], float(q[q_off + dk]) * q_fac, o_acc);
                }
            } else if (GDN_HEAD_K_DIM == 128u) {
                for (uint r = 0u; r < 4u; r++) {
                    const uint dk = lane + r * 32u;
                    o_acc = fma(s_col[r], float(q[q_off + dk]) * q_fac, o_acc);
                }
            } else {
                for (uint r = 0u; r < 8u; r++) {
                    const uint dk = lane + r * 32u;
                    if (dk < d_k) {
                        o_acc = fma(s_col[r], float(q[q_off + dk]) * q_fac, o_acc);
                    }
                }
            }

            float o_val = simd_sum(o_acc);
            const ulong out_idx = (ulong(node) * p.num_v_heads + vh) * d_v + dv;
            if (lane == 0u) {
                if (p.apply_out_gate != 0u) {
                    o_val *= hybrid_fast_silu(float(z[out_idx]));
                }
                out[out_idx] = half(o_val);
            }

            if (node == p.scratch_leaf_node) {
                if (GDN_HEAD_K_DIM == 64u) {
                    for (uint r = 0u; r < 2u; r++) {
                        state_out[head_state_base + ulong(lane + r * 32u) * d_v + dv] = s_col[r];
                    }
                } else if (GDN_HEAD_K_DIM == 128u) {
                    for (uint r = 0u; r < 4u; r++) {
                        state_out[head_state_base + ulong(lane + r * 32u) * d_v + dv] = s_col[r];
                    }
                } else {
                    for (uint r = 0u; r < 8u; r++) {
                        const uint dk = lane + r * 32u;
                        if (dk < d_k) {
                            state_out[head_state_base + ulong(dk) * d_v + dv] = s_col[r];
                        }
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// In-place per-head RMSNorm + optional SiLU output gate for batched GDN prefill
// / tree-verify outputs (`[num_tokens, num_v_heads, head_v_dim]` f16).
// Grid: (num_v_heads, num_tokens, 1) threadgroups, 32 threads.
kernel void ami_gdn_out_norm_gate(device half *out [[buffer(0)]],
                                  const device half *z [[buffer(1)]],
                                  const device uchar *norm_w [[buffer(2)]],
                                  constant GdnOutNormGateParams &p [[buffer(3)]],
                                  uint2 tg_pos [[threadgroup_position_in_grid]],
                                  uint lane [[thread_index_in_simdgroup]]) {
    const uint vh = tg_pos.x;
    const uint tok = tg_pos.y;
    if (vh >= p.num_v_heads || tok >= p.num_tokens) {
        return;
    }

    const uint d_v = p.head_v_dim;
    const ulong base = (ulong(tok) * p.num_v_heads + vh) * d_v;

    float inv_rms = 1.0f;
    if (p.apply_out_norm != 0u) {
        float sum_sq = 0.0f;
        for (uint dv = lane; dv < d_v; dv += 32u) {
            const float v = float(out[base + dv]);
            sum_sq = fma(v, v, sum_sq);
        }
        const float total_sq = simd_sum(sum_sq);
        inv_rms = rsqrt(total_sq / float(d_v) + p.rms_eps);
    }

    const uint nw_base = (p.per_head_norm != 0u) ? (vh * d_v) : 0u;
    for (uint dv = lane; dv < d_v; dv += 32u) {
        float val = float(out[base + dv]);
        if (p.apply_out_norm != 0u) {
            val = (val * inv_rms) * hybrid_load_norm_w(norm_w, nw_base + dv, HYBRID_NORM_W_BF16);
        }
        if (p.apply_out_gate != 0u) {
            val *= hybrid_fast_silu(float(z[base + dv]));
        }
        out[base + dv] = half(val);
    }
}

// ADR-0019 loop-isolation state commit: copies `S_tmp` -> `S_t` on final loop pass `R - 1`.
kernel void ami_gdn_commit_state(const device float *src [[buffer(0)]],
                                 device float *dst [[buffer(1)]],
                                 constant GdnCommitStateParams &p [[buffer(2)]],
                                 uint gid [[thread_position_in_grid]]) {
    const uint vec_n = p.total_f32_elems >> 2;
    if (gid < vec_n) {
        ((device float4 *)dst)[gid] = ((const device float4 *)src)[gid];
        return;
    }
    const uint tail_idx = (vec_n << 2) + (gid - vec_n);
    if (tail_idx < p.total_f32_elems) {
        dst[tail_idx] = src[tail_idx];
    }
}

// Commits an accepted speculative tree path `accepted_nodes[0..accepted_count)`
// onto `state_in` -> `state_out` with zero per-node state storage.
// Grid: (num_v_heads, ceil(head_v_dim / 8), 1) threadgroups, 256 threads.
kernel void ami_gdn_commit_tree(const device half *k_tree [[buffer(0)]],
                                const device half *v_tree [[buffer(1)]],
                                const device float *g_tree [[buffer(2)]],
                                const device float *beta_tree [[buffer(3)]],
                                const device uint *accepted_nodes [[buffer(4)]],
                                const device float *state_in [[buffer(5)]],
                                device float *state_out [[buffer(6)]],
                                constant GdnCommitTreeParams &p [[buffer(7)]],
                                uint2 tg_pos [[threadgroup_position_in_grid]],
                                uint sg_id [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    const uint vh = tg_pos.x;
    const uint dv = tg_pos.y * 8u + sg_id;
    if (vh >= p.num_v_heads) {
        return;
    }

    const uint d_k = (GDN_HEAD_K_DIM > 0u) ? GDN_HEAD_K_DIM : p.head_k_dim;
    const uint d_v = p.head_v_dim;
    const uint v_per_k = p.num_v_heads / p.num_k_heads;
    const uint kh = vh / v_per_k;
    const bool active_dv = (dv < d_v);

    threadgroup float tg_k[256];
    threadgroup float tg_alpha;
    threadgroup float tg_beta;

    const ulong head_state_base = ulong(vh) * d_k * d_v;
    float s_col[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (active_dv) {
        for (uint r = 0u; r < 8u; r++) {
            const uint dk = lane + r * 32u;
            if (dk < d_k) {
                s_col[r] = state_in[head_state_base + ulong(dk) * d_v + dv];
            }
        }
    }

    for (uint step = 0u; step < p.accepted_count; step++) {
        const uint node_idx = (p.use_node_buffer != 0u) ? accepted_nodes[step] : uint(p.inline_nodes[step]);
        if (sg_id == 0u) {
            const ulong k_off = (ulong(node_idx) * p.num_k_heads + kh) * d_k;
            float k_sq = 0.0f;
            for (uint dk = lane; dk < d_k; dk += 32u) {
                const float kv = float(k_tree[k_off + dk]);
                tg_k[dk] = kv;
                k_sq = fma(kv, kv, k_sq);
            }
            const float k_sum = simd_sum(k_sq);
            const float k_fac = (p.apply_l2_norm != 0u) ? rsqrt(k_sum + p.l2_eps) : 1.0f;
            for (uint dk = lane; dk < d_k; dk += 32u) {
                tg_k[dk] *= k_fac;
            }
            if (lane == 0u) {
                const ulong gb_idx = ulong(node_idx) * p.num_v_heads + vh;
                tg_alpha = exp(g_tree[gb_idx]);
                tg_beta = hybrid_sigmoid(beta_tree[gb_idx]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active_dv) {
            const float alpha = tg_alpha;
            const float beta_sig = tg_beta;
            const ulong v_idx = (ulong(node_idx) * p.num_v_heads + vh) * d_v + dv;
            const float v_val = float(v_tree[v_idx]);

            float pred_acc = 0.0f;
            for (uint r = 0u; r < 8u; r++) {
                const uint dk = lane + r * 32u;
                if (dk < d_k) {
                    s_col[r] *= alpha;
                    pred_acc = fma(s_col[r], tg_k[dk], pred_acc);
                }
            }
            const float pred_v = simd_sum(pred_acc);
            const float delta = beta_sig * (v_val - pred_v);
            for (uint r = 0u; r < 8u; r++) {
                const uint dk = lane + r * 32u;
                if (dk < d_k) {
                    s_col[r] = fma(tg_k[dk], delta, s_col[r]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (active_dv) {
        for (uint r = 0u; r < 8u; r++) {
            const uint dk = lane + r * 32u;
            if (dk < d_k) {
                state_out[head_state_base + ulong(dk) * d_v + dv] = s_col[r];
            }
        }
    }
}

// ===========================================================================
// Section B: G9 + R5 — Qwen Sparse Attention (QSA)
//   - YaRN RoPE scaling (`factor`, `mscale`, `beta_fast`, `beta_slow`)
//   - Interleaved 3D MRoPE (`mrope_section = [s0, s1, s2]` across `rotary_dim/2`)
//   - Hybrid RGBA16Float IOSurface KvTile ring (512-tok full & 128-tok circular SWA)
//   - Micro-block KV indexer (`ami_qsa_index_blocks`) with mandatory local block(s)
//     and ADR-0019 cross-loop index reuse
//   - Sparse GQA decode & 64-bit ancestor-masked tree-verify (`ami_qsa_sparse_attn`)
// ===========================================================================

struct QsaKvTilePair {
    texture2d<half, access::read> k_tex;
    texture2d<half, access::read> v_tex;
};

static_assert(sizeof(QsaKvTilePair) == 16, "QsaKvTilePair must be 16 bytes");

struct QsaRopeKvParams {
    uint num_q_heads;
    uint num_kv_heads;
    uint head_dim;
    uint rotary_dim;
    uint pos_t;
    uint pos_h;
    uint pos_w;
    uint mrope_s0;
    uint mrope_s1;
    uint mrope_s2;
    uint chunk_tokens;
    uint slot_in_chunk;
    float rms_eps;
    float rope_theta;
    float yarn_factor;
    float yarn_mscale;
    float yarn_beta_fast;
    float yarn_beta_slow;
    uint original_max_pos;
    uint apply_qk_norm;
};

struct QsaTreeRopeParams {
    uint num_nodes;
    uint num_q_heads;
    uint num_kv_heads;
    uint head_dim;
    uint rotary_dim;
    uint mrope_s0;
    uint mrope_s1;
    uint mrope_s2;
    float rms_eps;
    float rope_theta;
    float yarn_factor;
    float yarn_mscale;
    float yarn_beta_fast;
    float yarn_beta_slow;
    uint original_max_pos;
    uint apply_qk_norm;
};

struct QsaKvTreeCommitParams {
    uint num_kv_heads;
    uint head_dim;
    uint base_pos;
    uint chunk_tokens;
    uint ring_capacity;
    uint accepted_count;
    uint use_node_buffer;
    uint has_second_chunk;
    ushort inline_nodes[64];
};

struct QsaBlockSummaryParams {
    uint num_kv_heads;
    uint head_dim;
    uint kv_len;
    uint window_size;
    uint ring_capacity;
    uint chunk_tokens;
    uint block_tokens;
    uint start_block;
    uint num_blocks;
    uint num_retired_chunks;
    uint _pad1;
    uint _pad2;
};

struct QsaIndexParams {
    uint num_nodes;
    uint num_q_heads;
    uint num_kv_heads;
    uint head_dim;
    uint kv_len;
    uint window_size;
    uint ring_capacity;
    uint block_tokens;
    uint top_k_blocks;
    uint num_local_blocks;
    float scale;
    uint causal_q_start_pos;
};

struct QsaSparseAttnParams {
    uint num_nodes;
    uint num_q_heads;
    uint num_kv_heads;
    uint head_dim;
    uint kv_len;
    uint window_size;
    uint ring_capacity;
    uint chunk_tokens;
    uint block_tokens;
    uint top_k_blocks;
    uint has_tree;
    float scale;
    uint num_retired_chunks;
    uint retired_plane_stride_bytes;
    uint retired_chunk_stride_bytes;
    uint causal_q_start_pos;
};

static_assert(sizeof(QsaRopeKvParams) == 80, "QsaRopeKvParams must be 80 bytes");
static_assert(sizeof(QsaTreeRopeParams) == 64, "QsaTreeRopeParams must be 64 bytes");
static_assert(sizeof(QsaKvTreeCommitParams) == 160, "QsaKvTreeCommitParams must be 160 bytes");
static_assert(sizeof(QsaBlockSummaryParams) == 48, "QsaBlockSummaryParams must be 48 bytes");
static_assert(sizeof(QsaIndexParams) == 48, "QsaIndexParams must be 48 bytes");
static_assert(sizeof(QsaSparseAttnParams) == 64, "QsaSparseAttnParams must be 64 bytes");

// Round-robin interleaved 3D MRoPE axis selector (`0 = T, 1 = H, 2 = W`).
// For `[11, 11, 10]` across `half_rot = 32`, yields `T,H,W,...,T,H,W,T,H`,
// matching HuggingFace Qwen3-VL / Qwen3.8 interleaved MRoPE identically while
// supporting arbitrary `[s0, s1, s2]` summing to `rotary_dim / 2`.
static inline uint mrope_interleaved_axis(uint j, uint s0, uint s1, uint s2) {
    const uint m1 = min(s0, min(s1, s2));
    const uint three_m1 = m1 * 3u;
    if (j < three_m1) {
        return j % 3u;
    }
    const uint rem = j - three_m1;
    const uint a0 = s0 - m1;
    const uint a1 = s1 - m1;
    const uint a2 = s2 - m1;

    // At least one of (a0, a1, a2) is zero; collect the remaining two axes in order.
    uint ax_a = 0u, cnt_a = 0u;
    uint ax_b = 0u, cnt_b = 0u;
    if (a0 == 0u) {
        ax_a = 1u; cnt_a = a1;
        ax_b = 2u; cnt_b = a2;
    } else if (a1 == 0u) {
        ax_a = 0u; cnt_a = a0;
        ax_b = 2u; cnt_b = a2;
    } else {
        ax_a = 0u; cnt_a = a0;
        ax_b = 1u; cnt_b = a1;
    }
    const uint m2 = min(cnt_a, cnt_b);
    if (rem < m2 * 2u) {
        return ((rem & 1u) == 0u) ? ax_a : ax_b;
    }
    return (cnt_a > cnt_b) ? ax_a : ax_b;
}

// YaRN inverse frequency for pair index `j in [0 .. rotary_dim/2)`.
// Reduces identically to standard RoPE (`theta^(-2j/rotary_dim)`) when `factor <= 1.0`.
static inline float yarn_inv_freq(uint j,
                                  uint rotary_dim,
                                  float rope_theta,
                                  float yarn_factor,
                                  float beta_fast,
                                  float beta_slow,
                                  uint original_max_pos) {
    const float log_theta = log(rope_theta);
    const float inv_freq_extrap = exp(-log_theta * (2.0f * float(j) / float(rotary_dim)));
    if (yarn_factor <= 1.0f || original_max_pos == 0u) {
        return inv_freq_extrap;
    }
    constexpr float TWO_PI = 6.2831853071795864769f;
    const float half_rot = float(rotary_dim >> 1);
    const float f_orig = float(original_max_pos);
    const float raw_low = (float(rotary_dim) * log(f_orig / (beta_fast * TWO_PI))) / (2.0f * log_theta);
    const float raw_high = (float(rotary_dim) * log(f_orig / (beta_slow * TWO_PI))) / (2.0f * log_theta);
    const float low = clamp(floor(raw_low), 0.0f, half_rot - 1.0f);
    float high = clamp(ceil(raw_high), 0.0f, half_rot - 1.0f);
    if (high <= low) {
        high = low + 0.001f;
    }
    const float ramp = clamp((float(j) - low) / (high - low), 0.0f, 1.0f);
    const float extrap_weight = 1.0f - ramp;
    const float inv_freq_interp = inv_freq_extrap / yarn_factor;
    return fma(inv_freq_extrap, extrap_weight, inv_freq_interp * ramp);
}

// Single-token QSA QK-Norm + YaRN + Interleaved 3D MRoPE + Hybrid KvTile write
// (supports both 512-tok full chunks and 128-tok circular SWA tiles).
// Grid: (num_q_heads + num_kv_heads, 1, 1) threadgroups, 32 threads.
kernel void ami_qsa_qknorm_mrope_kv_write(device half *q [[buffer(0)]],
                                          const device half *k [[buffer(1)]],
                                          const device half *v [[buffer(2)]],
                                          const device uchar *q_norm [[buffer(3)]],
                                          const device uchar *k_norm [[buffer(4)]],
                                          constant QsaRopeKvParams &p [[buffer(5)]],
                                          texture2d<half, access::write> k_tex [[texture(0)]],
                                          texture2d<half, access::write> v_tex [[texture(1)]],
                                          uint tg_id [[threadgroup_position_in_grid]],
                                          uint lane [[thread_index_in_simdgroup]]) {
    const uint total_heads = p.num_q_heads + p.num_kv_heads;
    if (tg_id >= total_heads) {
        return;
    }

    threadgroup float vec[256];
    const bool is_q = (tg_id < p.num_q_heads);
    const uint head_idx = is_q ? tg_id : (tg_id - p.num_q_heads);
    const uint d_dim = p.head_dim;
    const uint half_rot = p.rotary_dim >> 1;

    const device half *src = is_q ? (q + ulong(head_idx) * d_dim)
                                  : (k + ulong(head_idx) * d_dim);
    float sum_sq = 0.0f;
    for (uint d = lane; d < d_dim; d += 32u) {
        const float val = float(src[d]);
        vec[d] = val;
        sum_sq = fma(val, val, sum_sq);
    }

    if (p.apply_qk_norm != 0u) {
        const float total_sq = simd_sum(sum_sq);
        const float inv_rms = rsqrt(total_sq / float(d_dim) + p.rms_eps);
        const device uchar *nw = is_q ? q_norm : k_norm;
        for (uint d = lane; d < d_dim; d += 32u) {
            vec[d] = (vec[d] * inv_rms) * hybrid_load_norm_w(nw, d, HYBRID_NORM_W_BF16);
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // Partial or full NeoX half-split YaRN + Interleaved 3D MRoPE on [0..rotary_dim).
    for (uint j = lane; j < half_rot; j += 32u) {
        const uint axis = mrope_interleaved_axis(j, p.mrope_s0, p.mrope_s1, p.mrope_s2);
        const uint axis_pos = (axis == 0u) ? p.pos_t : ((axis == 1u) ? p.pos_h : p.pos_w);
        const float inv_f = yarn_inv_freq(
            j, p.rotary_dim, p.rope_theta, p.yarn_factor,
            p.yarn_beta_fast, p.yarn_beta_slow, p.original_max_pos);
        const float angle = float(axis_pos) * inv_f;
        const float c = cos(angle) * p.yarn_mscale;
        const float s = sin(angle) * p.yarn_mscale;
        const float x0 = vec[j];
        const float x1 = vec[j + half_rot];
        vec[j] = fma(x0, c, -x1 * s);
        vec[j + half_rot] = fma(x1, c, x0 * s);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    if (is_q) {
        device half *dst = q + ulong(head_idx) * d_dim;
        for (uint d = lane; d < d_dim; d += 32u) {
            dst[d] = half(vec[d]);
        }
    } else {
        const uint row = head_idx * p.chunk_tokens + p.slot_in_chunk;
        const uint num_cols = d_dim >> 2;
        const device half4 *v_src = (const device half4 *)(v + ulong(head_idx) * d_dim);
        for (uint col = lane; col < num_cols; col += 32u) {
            const uint base = col * 4u;
            const half4 k4 = half4(half(vec[base + 0u]),
                                   half(vec[base + 1u]),
                                   half(vec[base + 2u]),
                                   half(vec[base + 3u]));
            k_tex.write(k4, uint2(col, row));
            v_tex.write(v_src[col], uint2(col, row));
        }
    }
}

// Batched speculative tree QSA QK-Norm + YaRN + Interleaved 3D MRoPE.
// Accepts `[num_nodes, 3]` u32 3D positions `(t, h, w)` (text tokens set t=h=w=pos).
// Grid: (num_q_heads + num_kv_heads, num_nodes, 1) threadgroups, 32 threads.
kernel void ami_qsa_tree_qknorm_mrope(device half *q [[buffer(0)]],
                                      const device half *k [[buffer(1)]],
                                      const device half *v [[buffer(2)]],
                                      const device uchar *q_norm [[buffer(3)]],
                                      const device uchar *k_norm [[buffer(4)]],
                                      const device uint *positions_3d [[buffer(5)]],
                                      device half *k_tree_out [[buffer(6)]],
                                      device half *v_tree_out [[buffer(7)]],
                                      constant QsaTreeRopeParams &p [[buffer(8)]],
                                      uint2 tg_pos [[threadgroup_position_in_grid]],
                                      uint lane [[thread_index_in_simdgroup]]) {
    const uint total_heads = p.num_q_heads + p.num_kv_heads;
    const uint node = tg_pos.y;
    if (tg_pos.x >= total_heads || node >= p.num_nodes) {
        return;
    }

    threadgroup float vec[256];
    const bool is_q = (tg_pos.x < p.num_q_heads);
    const uint head_idx = is_q ? tg_pos.x : (tg_pos.x - p.num_q_heads);
    const uint d_dim = p.head_dim;
    const uint half_rot = p.rotary_dim >> 1;

    const uint pos_t = positions_3d[node * 3u + 0u];
    const uint pos_h = positions_3d[node * 3u + 1u];
    const uint pos_w = positions_3d[node * 3u + 2u];

    const ulong q_off = (ulong(node) * p.num_q_heads + head_idx) * d_dim;
    const ulong kv_off = (ulong(node) * p.num_kv_heads + head_idx) * d_dim;
    const device half *src = is_q ? (q + q_off) : (k + kv_off);

    float sum_sq = 0.0f;
    for (uint d = lane; d < d_dim; d += 32u) {
        const float val = float(src[d]);
        vec[d] = val;
        sum_sq = fma(val, val, sum_sq);
    }

    if (p.apply_qk_norm != 0u) {
        const float total_sq = simd_sum(sum_sq);
        const float inv_rms = rsqrt(total_sq / float(d_dim) + p.rms_eps);
        const device uchar *nw = is_q ? q_norm : k_norm;
        for (uint d = lane; d < d_dim; d += 32u) {
            vec[d] = (vec[d] * inv_rms) * hybrid_load_norm_w(nw, d, HYBRID_NORM_W_BF16);
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    for (uint j = lane; j < half_rot; j += 32u) {
        const uint axis = mrope_interleaved_axis(j, p.mrope_s0, p.mrope_s1, p.mrope_s2);
        const uint axis_pos = (axis == 0u) ? pos_t : ((axis == 1u) ? pos_h : pos_w);
        const float inv_f = yarn_inv_freq(
            j, p.rotary_dim, p.rope_theta, p.yarn_factor,
            p.yarn_beta_fast, p.yarn_beta_slow, p.original_max_pos);
        const float angle = float(axis_pos) * inv_f;
        const float c = cos(angle) * p.yarn_mscale;
        const float s = sin(angle) * p.yarn_mscale;
        const float x0 = vec[j];
        const float x1 = vec[j + half_rot];
        vec[j] = fma(x0, c, -x1 * s);
        vec[j + half_rot] = fma(x1, c, x0 * s);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    if (is_q) {
        device half *dst = q + q_off;
        for (uint d = lane; d < d_dim; d += 32u) {
            dst[d] = half(vec[d]);
        }
    } else {
        device half *k_dst = k_tree_out + kv_off;
        device half *v_dst = v_tree_out + kv_off;
        const device half *v_src = v + kv_off;
        for (uint d = lane; d < d_dim; d += 32u) {
            k_dst[d] = half(vec[d]);
            v_dst[d] = v_src[d];
        }
    }
}

// Accepted-path KV commit supporting both linear full-chunk tiles (512-tok)
// and circular sliding-window tiles (128-tok SWA ring wrap-around).
// Grid: (num_kv_heads, accepted_count, 1) threadgroups, 32 threads.
kernel void ami_qsa_kv_tree_commit(const device half *k_tree [[buffer(0)]],
                                   const device half *v_tree [[buffer(1)]],
                                   const device uint *accepted_nodes [[buffer(2)]],
                                   constant QsaKvTreeCommitParams &p [[buffer(3)]],
                                   texture2d<half, access::write> k_tile0 [[texture(0)]],
                                   texture2d<half, access::write> v_tile0 [[texture(1)]],
                                   texture2d<half, access::write> k_tile1 [[texture(2)]],
                                   texture2d<half, access::write> v_tile1 [[texture(3)]],
                                   uint2 tg_pos [[threadgroup_position_in_grid]],
                                   uint lane [[thread_index_in_simdgroup]]) {
    const uint kvh = tg_pos.x;
    const uint step = tg_pos.y;
    if (kvh >= p.num_kv_heads || step >= p.accepted_count) {
        return;
    }

    const uint node_idx = (p.use_node_buffer != 0u) ? accepted_nodes[step] : uint(p.inline_nodes[step]);
    const uint base_phys = (p.ring_capacity > 0u) ? (p.base_pos % p.ring_capacity) : p.base_pos;
    const uint chunk0 = base_phys / p.chunk_tokens;

    const uint pos = p.base_pos + step;
    const uint pos_phys = (p.ring_capacity > 0u) ? (pos % p.ring_capacity) : pos;
    const uint pos_chunk = pos_phys / p.chunk_tokens;
    const uint slot_in_chunk = pos_phys - pos_chunk * p.chunk_tokens;
    const uint row = kvh * p.chunk_tokens + slot_in_chunk;
    const uint num_cols = p.head_dim >> 2;
    const ulong src_off = (ulong(node_idx) * p.num_kv_heads + kvh) * p.head_dim;

    const device half4 *k_src = (const device half4 *)(k_tree + src_off);
    const device half4 *v_src = (const device half4 *)(v_tree + src_off);
    const bool use_tile1 = (p.has_second_chunk != 0u) && (pos_chunk != chunk0);

    for (uint col = lane; col < num_cols; col += 32u) {
        if (use_tile1) {
            k_tile1.write(k_src[col], uint2(col, row));
            v_tile1.write(v_src[col], uint2(col, row));
        } else {
            k_tile0.write(k_src[col], uint2(col, row));
            v_tile0.write(v_src[col], uint2(col, row));
        }
    }
}

// Computes mean-pooled coarse/summary micro-block keys directly from resident
// RGBA16Float IOSurface KvTiles (supports both full-512 and circular-128 rings).
// Grid: (num_kv_heads, num_blocks, 1) threadgroups, 32 threads.
kernel void ami_qsa_update_block_summary(const device QsaKvTilePair *kv_tiles [[buffer(0)]],
                                         device half *block_keys [[buffer(1)]],
                                         constant QsaBlockSummaryParams &p [[buffer(2)]],
                                         uint2 tg_pos [[threadgroup_position_in_grid]],
                                         uint lane [[thread_index_in_simdgroup]]) {
    const uint kvh = tg_pos.x;
    const uint bi = tg_pos.y;
    if (kvh >= p.num_kv_heads || bi >= p.num_blocks || p.kv_len == 0u) {
        return;
    }

    const uint win_start = (p.window_size > 0u && p.kv_len > p.window_size)
        ? (p.kv_len - p.window_size)
        : 0u;
    const uint b = p.start_block + bi;
    const uint t0 = max(b * p.block_tokens, win_start);
    const uint t1 = min((b + 1u) * p.block_tokens, p.kv_len);
    if (t0 >= t1) {
        return;
    }

    const uint ring_blocks = (p.ring_capacity > 0u) ? (p.ring_capacity / p.block_tokens) : 0u;
    const uint phys_b = (ring_blocks > 0u) ? (b % ring_blocks) : b;
    const uint num_cols = p.head_dim >> 2;
    const float inv_cnt = 1.0f / float(t1 - t0);
    device half4 *dst = (device half4 *)(block_keys + (ulong(phys_b) * p.num_kv_heads + kvh) * p.head_dim);

    for (uint col = lane; col < num_cols; col += 32u) {
        float4 acc = float4(0.0f);
        for (uint t = t0; t < t1; t++) {
            const uint phys_t = (p.ring_capacity > 0u) ? (t % p.ring_capacity) : t;
            const uint chunk = phys_t / p.chunk_tokens;
            const uint slot = phys_t - chunk * p.chunk_tokens;
            const uint row = kvh * p.chunk_tokens + slot;
            const uint active_chunk = (p.ring_capacity == 0u) ? (chunk - p.num_retired_chunks) : chunk;
            acc += float4(kv_tiles[active_chunk].k_tex.read(uint2(col, row)));
        }
        dst[col] = half4(acc * inv_cnt);
    }
}

// Micro-block KV indexer (`G9`): scores coarse/summary block keys against the
// GQA query group sharing `kv_head` (`score(b) = max_g dot(q[g], key[b]) * scale`)
// and selects top-`K_blocks` valid KV micro-blocks + mandatory trailing local
// recent block(s). Writes into wired `selected_blocks` / `selected_counts` so
// ADR-0019 loop passes `r >= 1` can skip this dispatch entirely.
//
// Grid: (num_kv_heads, num_nodes, 1) threadgroups, 256 threads (8 simdgroups).
kernel void ami_qsa_index_blocks(const device half *q [[buffer(0)]],
                                 const device half *block_keys [[buffer(1)]],
                                 device uint *selected_blocks [[buffer(2)]],
                                 device uint *selected_counts [[buffer(3)]],
                                 constant QsaIndexParams &p [[buffer(4)]],
                                 uint2 tg_pos [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]],
                                 uint sg_id [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
    const uint kvh = tg_pos.x;
    const uint node = tg_pos.y;
    if (kvh >= p.num_kv_heads || node >= p.num_nodes) {
        return;
    }

    const uint k_cap = min(p.top_k_blocks, 64u);
    const ulong cnt_idx = ulong(node) * p.num_kv_heads + kvh;
    const ulong sel_base = cnt_idx * p.top_k_blocks;
    const uint eff_kv_len = (p.causal_q_start_pos != 0xFFFFFFFFu)
        ? min(p.kv_len, p.causal_q_start_pos + node + 1u)
        : p.kv_len;

    if (eff_kv_len == 0u || k_cap == 0u) {
        if (tid < p.top_k_blocks) {
            selected_blocks[sel_base + tid] = 0xFFFFFFFFu;
        }
        if (tid == 0u) {
            selected_counts[cnt_idx] = 0u;
        }
        return;
    }

    const uint win_start = (p.window_size > 0u && eff_kv_len > p.window_size)
        ? (eff_kv_len - p.window_size)
        : 0u;
    const uint first_block = win_start / p.block_tokens;
    const uint last_block = (eff_kv_len - 1u) / p.block_tokens;
    const uint num_valid = last_block - first_block + 1u;

    // Fast path: all valid blocks fit within top_k_blocks budget.
    if (num_valid <= k_cap) {
        for (uint i = tid; i < p.top_k_blocks; i += 256u) {
            selected_blocks[sel_base + i] = (i < num_valid) ? (first_block + i) : 0xFFFFFFFFu;
        }
        if (tid == 0u) {
            selected_counts[cnt_idx] = num_valid;
        }
        return;
    }

    const uint local_cnt = min(max(p.num_local_blocks, 1u), k_cap);
    const uint local_start = last_block + 1u - local_cnt;
    const uint k_sel = k_cap - local_cnt;
    const uint gqa_ratio = p.num_q_heads / p.num_kv_heads;
    const uint ring_blocks = (p.ring_capacity > 0u) ? (p.ring_capacity / p.block_tokens) : 0u;

    // Constant 4 KiB threadgroup memory supports arbitrary context lengths.
    threadgroup float tg_scores[8][64];
    threadgroup uint tg_blocks[8][64];

    if (lane == 0u) {
        for (uint i = 0u; i < k_sel; i++) {
            tg_scores[sg_id][i] = -INFINITY;
            tg_blocks[sg_id][i] = 0xFFFFFFFFu;
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // Each simdgroup streams non-local blocks [first_block, local_start) strided by 8.
    if (k_sel > 0u) {
        const ulong q_node_base = (ulong(node) * p.num_q_heads + kvh * gqa_ratio) * p.head_dim;
        const uint vec_d = p.head_dim >> 2;
        uint worst_slot = 0u;
        float worst_score = -INFINITY;
        uint worst_block = 0xFFFFFFFFu;

        for (uint b = first_block + sg_id; b < local_start; b += 8u) {
            const uint phys_b = (ring_blocks > 0u) ? (b % ring_blocks) : b;
            const device half4 *bk_vec = (const device half4 *)(block_keys + (ulong(phys_b) * p.num_kv_heads + kvh) * p.head_dim);

            float best_head_score = -INFINITY;
            for (uint g = 0u; g < gqa_ratio; g++) {
                const device half4 *q_vec = (const device half4 *)(q + q_node_base + ulong(g) * p.head_dim);
                float dot_acc = 0.0f;
                for (uint c = lane; c < vec_d; c += 32u) {
                    dot_acc += dot(float4(q_vec[c]), float4(bk_vec[c]));
                }
                const float h_score = simd_sum(dot_acc) * p.scale;
                best_head_score = max(best_head_score, h_score);
            }

            if (lane == 0u) {
                if (best_head_score > worst_score || (best_head_score == worst_score && b < worst_block)) {
                    tg_scores[sg_id][worst_slot] = best_head_score;
                    tg_blocks[sg_id][worst_slot] = b;

                    // Recompute worst entry in this simdgroup's top-k_sel table.
                    uint new_w = 0u;
                    float min_s = tg_scores[sg_id][0];
                    uint max_b = tg_blocks[sg_id][0];
                    for (uint i = 1u; i < k_sel; i++) {
                        const float si = tg_scores[sg_id][i];
                        const uint bi = tg_blocks[sg_id][i];
                        if (si < min_s || (si == min_s && bi > max_b)) {
                            min_s = si;
                            max_b = bi;
                            new_w = i;
                        }
                    }
                    worst_slot = new_w;
                    worst_score = min_s;
                    worst_block = max_b;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Simdgroup 0 merges the `8 * k_sel <= 512` candidates deterministically.
    if (sg_id == 0u && k_sel > 0u) {
        const uint total_cands = 8u * k_sel;
        for (uint k = 0u; k < k_sel; k++) {
            float best_s = -INFINITY;
            uint best_b = 0xFFFFFFFFu;
            uint best_c = 0xFFFFFFFFu;
            for (uint c = lane; c < total_cands; c += 32u) {
                const uint sg = c / k_sel;
                const uint slot = c - sg * k_sel;
                const uint cb = tg_blocks[sg][slot];
                if (cb != 0xFFFFFFFFu) {
                    const float cs = tg_scores[sg][slot];
                    if (cs > best_s || (cs == best_s && cb < best_b)) {
                        best_s = cs;
                        best_b = cb;
                        best_c = c;
                    }
                }
            }
            for (ushort off = 16u; off > 0u; off >>= 1u) {
                const float os = simd_shuffle_down(best_s, off);
                const uint ob = simd_shuffle_down(best_b, off);
                const uint oc = simd_shuffle_down(best_c, off);
                if (os > best_s || (os == best_s && ob < best_b)) {
                    best_s = os;
                    best_b = ob;
                    best_c = oc;
                }
            }
            const uint win_b = simd_broadcast_first(best_b);
            const uint win_c = simd_broadcast_first(best_c);
            if (lane == 0u) {
                selected_blocks[sel_base + k] = win_b;
                if (win_c < total_cands) {
                    const uint w_sg = win_c / k_sel;
                    const uint w_slot = win_c - w_sg * k_sel;
                    tg_scores[w_sg][w_slot] = -INFINITY;
                    tg_blocks[w_sg][w_slot] = 0xFFFFFFFFu;
                }
            }
        }
    }

    // Append mandatory local recent blocks and finalize count.
    if (tid < local_cnt) {
        selected_blocks[sel_base + k_sel + tid] = local_start + tid;
    }
    if (tid >= k_cap && tid < p.top_k_blocks) {
        selected_blocks[sel_base + tid] = 0xFFFFFFFFu;
    }
    if (tid == 0u) {
        selected_counts[cnt_idx] = k_cap;
    }
}

// Sparse GQA decode (`num_nodes = 1, has_tree = 0`) & speculative tree-verify
// (`num_nodes = 1..64, has_tree = 1` with 64-bit ancestor masks) over selected
// micro-blocks in resident RGBA16Float IOSurface KvTiles (supporting both
// Group-64 q8 retired KV dequant helper matching `RetiredKvQ8Pool` (`68 B / 64 dims`).
static inline half4 hybrid_load_q8_half4(const device uchar *plane_base,
                                         uint row,
                                         uint groups_per_head,
                                         uint col) {
    const uint grp = col >> 4;
    const uint col_in_grp = col & 15u;
    const ulong off = (ulong(row) * groups_per_head + grp) * 68ul;
    const char4 q4 = *(const device char4 *)(plane_base + off + col_in_grp * 4u);
    const half2 sb = *(const device half2 *)(plane_base + off + 64u);
    return half4(fma(float4(q4), float4(float(sb.x)), float4(float(sb.y))));
}

// 512-tok full chunks, R5 retired group-64 q8 chunks, and 128-tok circular SWA tiles).
//
// Specialized on QSA_HEAD_DIM (`64`, `128`, or `0` dynamic <= 256 multiple of 4).
// Grid: (num_kv_heads * q_groups_per_kv, num_nodes, 1) threadgroups, 256 threads.
kernel void ami_qsa_sparse_attn(const device half *q [[buffer(0)]],
                                const device QsaKvTilePair *kv_tiles [[buffer(1)]],
                                const device uint *selected_blocks [[buffer(2)]],
                                const device uint *selected_counts [[buffer(3)]],
                                const device half *k_tree [[buffer(4)]],
                                const device half *v_tree [[buffer(5)]],
                                const device ulong *ancestor_masks [[buffer(6)]],
                                device half *out [[buffer(7)]],
                                const device uchar *retired_pool [[buffer(8)]],
                                const device uint *retired_chunk_ids [[buffer(9)]],
                                constant QsaSparseAttnParams &p [[buffer(10)]],
                                uint2 tg_pos [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]],
                                uint sg_id [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint Q_TILE = 8u;
    constexpr float LOG2E = 1.4426950408889634f;

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

    const uint d_dim = (QSA_HEAD_DIM > 0u) ? QSA_HEAD_DIM : p.head_dim;
    const uint num_cols = d_dim >> 2;
    const float scale_log2e = p.scale * LOG2E;
    const uint eff_kv_len = (p.causal_q_start_pos != 0xFFFFFFFFu)
        ? min(p.kv_len, p.causal_q_start_pos + node + 1u)
        : p.kv_len;
    const uint win_start = (p.window_size > 0u && eff_kv_len > p.window_size)
        ? (eff_kv_len - p.window_size)
        : 0u;

    const ulong cnt_idx = ulong(node) * p.num_kv_heads + kv_head;
    const uint sel_cnt = (eff_kv_len > 0u) ? min(selected_counts[cnt_idx], p.top_k_blocks) : 0u;
    const ulong sel_base = cnt_idx * p.top_k_blocks;
    const ulong q_row_off = (ulong(node) * p.num_q_heads + qh) * d_dim;

    // 4 KiB each for staging up to 8 (d=128) or 16 (d=64) or 4 (d=256) tokens.
    threadgroup half4 tg_k[256];
    threadgroup half4 tg_v[256];

    if (QSA_HEAD_DIM == 128u) {
        const float4 q_vec = active_q
            ? float4(*(const device half4 *)(q + q_row_off + lane * 4u))
            : float4(0.0f);
        float m_val = -INFINITY;
        float l_val = 0.0f;
        float4 acc = float4(0.0f);

        const uint load_tok = tid >> 5;  // 0..7
        const uint load_col = tid & 31u; // 0..31

        for (uint bi = 0u; bi < sel_cnt; bi++) {
            const uint b = selected_blocks[sel_base + bi];
            if (b == 0xFFFFFFFFu) {
                continue;
            }
            const uint t0 = max(b * p.block_tokens, win_start);
            const uint t1 = min((b + 1u) * p.block_tokens, eff_kv_len);
            if (t0 >= t1) {
                continue;
            }
            for (uint tile_t = t0; tile_t < t1; tile_t += 8u) {
                const uint tile_len = min(8u, t1 - tile_t);
                if (load_tok < tile_len) {
                    const uint t = tile_t + load_tok;
                    const uint phys_t = (p.ring_capacity > 0u) ? (t % p.ring_capacity) : t;
                    const uint chunk = phys_t / p.chunk_tokens;
                    const uint slot = phys_t - chunk * p.chunk_tokens;
                    const uint row = kv_head * p.chunk_tokens + slot;
                    if (p.ring_capacity == 0u && chunk < p.num_retired_chunks) {
                        const uint phys_slot = retired_chunk_ids[chunk];
                        const device uchar *k_plane = retired_pool + ulong(phys_slot) * p.retired_chunk_stride_bytes;
                        const device uchar *v_plane = k_plane + p.retired_plane_stride_bytes;
                        tg_k[tid] = hybrid_load_q8_half4(k_plane, row, 2u, load_col);
                        tg_v[tid] = hybrid_load_q8_half4(v_plane, row, 2u, load_col);
                    } else {
                        const uint active_chunk = (p.ring_capacity == 0u) ? (chunk - p.num_retired_chunks) : chunk;
                        tg_k[tid] = kv_tiles[active_chunk].k_tex.read(uint2(load_col, row));
                        tg_v[tid] = kv_tiles[active_chunk].v_tex.read(uint2(load_col, row));
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (active_q) {
                    for (uint dt = 0u; dt < tile_len; dt++) {
                        const uint idx = dt * 32u + lane;
                        const float score = simd_sum(dot(q_vec, float4(tg_k[idx]))) * scale_log2e;
                        const float new_m = max(m_val, score);
                        const float corr = (l_val > 0.0f) ? exp2(m_val - new_m) : 0.0f;
                        const float p_w = exp2(score - new_m);
                        l_val = fma(l_val, corr, p_w);
                        acc = fma(acc, float4(corr), float4(tg_v[idx]) * p_w);
                        m_val = new_m;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        if (p.has_tree != 0u && active_q) {
            ulong rem = ancestor_masks[node];
            while (rem != 0ul) {
                const uint j = uint(ctz(rem));
                rem &= (rem - 1ul);
                if (j < p.num_nodes) {
                    const ulong kv_off = (ulong(j) * p.num_kv_heads + kv_head) * 128u + lane * 4u;
                    const float4 kv = float4(*(const device half4 *)(k_tree + kv_off));
                    const float4 vv = float4(*(const device half4 *)(v_tree + kv_off));
                    const float score = simd_sum(dot(q_vec, kv)) * scale_log2e;
                    const float new_m = max(m_val, score);
                    const float corr = (l_val > 0.0f) ? exp2(m_val - new_m) : 0.0f;
                    const float p_w = exp2(score - new_m);
                    l_val = fma(l_val, corr, p_w);
                    acc = fma(acc, float4(corr), vv * p_w);
                    m_val = new_m;
                }
            }
        }

        if (active_q) {
            const float inv_l = (l_val > 0.0f) ? (1.0f / l_val) : 0.0f;
            *(device half4 *)(out + q_row_off + lane * 4u) = half4(acc * inv_l);
        }
        return;
    }

    if (QSA_HEAD_DIM == 64u) {
        const bool active_lane = (lane < 16u);
        const float4 q_vec = (active_q && active_lane)
            ? float4(*(const device half4 *)(q + q_row_off + lane * 4u))
            : float4(0.0f);
        float m_val = -INFINITY;
        float l_val = 0.0f;
        float4 acc = float4(0.0f);

        const uint load_tok = tid >> 4;  // 0..15
        const uint load_col = tid & 15u; // 0..15

        for (uint bi = 0u; bi < sel_cnt; bi++) {
            const uint b = selected_blocks[sel_base + bi];
            if (b == 0xFFFFFFFFu) {
                continue;
            }
            const uint t0 = max(b * p.block_tokens, win_start);
            const uint t1 = min((b + 1u) * p.block_tokens, eff_kv_len);
            if (t0 >= t1) {
                continue;
            }
            for (uint tile_t = t0; tile_t < t1; tile_t += 16u) {
                const uint tile_len = min(16u, t1 - tile_t);
                if (load_tok < tile_len) {
                    const uint t = tile_t + load_tok;
                    const uint phys_t = (p.ring_capacity > 0u) ? (t % p.ring_capacity) : t;
                    const uint chunk = phys_t / p.chunk_tokens;
                    const uint slot = phys_t - chunk * p.chunk_tokens;
                    const uint row = kv_head * p.chunk_tokens + slot;
                    if (p.ring_capacity == 0u && chunk < p.num_retired_chunks) {
                        const uint phys_slot = retired_chunk_ids[chunk];
                        const device uchar *k_plane = retired_pool + ulong(phys_slot) * p.retired_chunk_stride_bytes;
                        const device uchar *v_plane = k_plane + p.retired_plane_stride_bytes;
                        tg_k[tid] = hybrid_load_q8_half4(k_plane, row, 1u, load_col);
                        tg_v[tid] = hybrid_load_q8_half4(v_plane, row, 1u, load_col);
                    } else {
                        const uint active_chunk = (p.ring_capacity == 0u) ? (chunk - p.num_retired_chunks) : chunk;
                        tg_k[tid] = kv_tiles[active_chunk].k_tex.read(uint2(load_col, row));
                        tg_v[tid] = kv_tiles[active_chunk].v_tex.read(uint2(load_col, row));
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (active_q) {
                    for (uint dt = 0u; dt < tile_len; dt++) {
                        const uint idx = dt * 16u + (lane & 15u);
                        const float partial = active_lane ? dot(q_vec, float4(tg_k[idx])) : 0.0f;
                        const float score = simd_sum(partial) * scale_log2e;
                        const float new_m = max(m_val, score);
                        const float corr = (l_val > 0.0f) ? exp2(m_val - new_m) : 0.0f;
                        const float p_w = exp2(score - new_m);
                        l_val = fma(l_val, corr, p_w);
                        if (active_lane) {
                            acc = fma(acc, float4(corr), float4(tg_v[idx]) * p_w);
                        }
                        m_val = new_m;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        if (p.has_tree != 0u && active_q) {
            ulong rem = ancestor_masks[node];
            while (rem != 0ul) {
                const uint j = uint(ctz(rem));
                rem &= (rem - 1ul);
                if (j < p.num_nodes) {
                    const ulong kv_off = (ulong(j) * p.num_kv_heads + kv_head) * 64u + (lane & 15u) * 4u;
                    const float4 kv = active_lane ? float4(*(const device half4 *)(k_tree + kv_off)) : float4(0.0f);
                    const float4 vv = active_lane ? float4(*(const device half4 *)(v_tree + kv_off)) : float4(0.0f);
                    const float partial = active_lane ? dot(q_vec, kv) : 0.0f;
                    const float score = simd_sum(partial) * scale_log2e;
                    const float new_m = max(m_val, score);
                    const float corr = (l_val > 0.0f) ? exp2(m_val - new_m) : 0.0f;
                    const float p_w = exp2(score - new_m);
                    l_val = fma(l_val, corr, p_w);
                    if (active_lane) {
                        acc = fma(acc, float4(corr), vv * p_w);
                    }
                    m_val = new_m;
                }
            }
        }

        if (active_q && active_lane) {
            const float inv_l = (l_val > 0.0f) ? (1.0f / l_val) : 0.0f;
            *(device half4 *)(out + q_row_off + lane * 4u) = half4(acc * inv_l);
        }
        return;
    }

    // Dynamic head_dim <= 256 (multiple of 4; multiple of 64 when retired chunks are active):
    // 2 float4 registers per lane (up to 64 cols = 256 dims).
    const uint groups_per_head = d_dim >> 6;
    float4 q_reg[2] = {float4(0.0f), float4(0.0f)};
    float4 acc_reg[2] = {float4(0.0f), float4(0.0f)};
    if (active_q) {
        if (lane < num_cols) {
            q_reg[0] = float4(*(const device half4 *)(q + q_row_off + lane * 4u));
        }
        if (lane + 32u < num_cols) {
            q_reg[1] = float4(*(const device half4 *)(q + q_row_off + (lane + 32u) * 4u));
        }
    }
    float m_val = -INFINITY;
    float l_val = 0.0f;

    for (uint bi = 0u; bi < sel_cnt; bi++) {
        const uint b = selected_blocks[sel_base + bi];
        if (b == 0xFFFFFFFFu) {
            continue;
        }
        const uint t0 = max(b * p.block_tokens, win_start);
        const uint t1 = min((b + 1u) * p.block_tokens, eff_kv_len);
        if (t0 >= t1) {
            continue;
        }
        for (uint t = t0; t < t1; t++) {
            const uint phys_t = (p.ring_capacity > 0u) ? (t % p.ring_capacity) : t;
            const uint chunk = phys_t / p.chunk_tokens;
            const uint slot = phys_t - chunk * p.chunk_tokens;
            const uint row = kv_head * p.chunk_tokens + slot;
            if (active_q) {
                float4 k0 = float4(0.0f), k1 = float4(0.0f);
                float4 v0 = float4(0.0f), v1 = float4(0.0f);
                float partial = 0.0f;
                if (p.ring_capacity == 0u && chunk < p.num_retired_chunks) {
                    const uint phys_slot = retired_chunk_ids[chunk];
                    const device uchar *k_plane = retired_pool + ulong(phys_slot) * p.retired_chunk_stride_bytes;
                    const device uchar *v_plane = k_plane + p.retired_plane_stride_bytes;
                    if (lane < num_cols) {
                        k0 = float4(hybrid_load_q8_half4(k_plane, row, groups_per_head, lane));
                        v0 = float4(hybrid_load_q8_half4(v_plane, row, groups_per_head, lane));
                        partial += dot(q_reg[0], k0);
                    }
                    if (lane + 32u < num_cols) {
                        k1 = float4(hybrid_load_q8_half4(k_plane, row, groups_per_head, lane + 32u));
                        v1 = float4(hybrid_load_q8_half4(v_plane, row, groups_per_head, lane + 32u));
                        partial += dot(q_reg[1], k1);
                    }
                } else {
                    const uint active_chunk = (p.ring_capacity == 0u) ? (chunk - p.num_retired_chunks) : chunk;
                    if (lane < num_cols) {
                        k0 = float4(kv_tiles[active_chunk].k_tex.read(uint2(lane, row)));
                        v0 = float4(kv_tiles[active_chunk].v_tex.read(uint2(lane, row)));
                        partial += dot(q_reg[0], k0);
                    }
                    if (lane + 32u < num_cols) {
                        k1 = float4(kv_tiles[active_chunk].k_tex.read(uint2(lane + 32u, row)));
                        v1 = float4(kv_tiles[active_chunk].v_tex.read(uint2(lane + 32u, row)));
                        partial += dot(q_reg[1], k1);
                    }
                }
                const float score = simd_sum(partial) * scale_log2e;
                const float new_m = max(m_val, score);
                const float corr = (l_val > 0.0f) ? exp2(m_val - new_m) : 0.0f;
                const float p_w = exp2(score - new_m);
                l_val = fma(l_val, corr, p_w);
                acc_reg[0] = fma(acc_reg[0], float4(corr), v0 * p_w);
                acc_reg[1] = fma(acc_reg[1], float4(corr), v1 * p_w);
                m_val = new_m;
            }
        }
    }

    if (p.has_tree != 0u && active_q) {
        ulong rem = ancestor_masks[node];
        while (rem != 0ul) {
            const uint j = uint(ctz(rem));
            rem &= (rem - 1ul);
            if (j < p.num_nodes) {
                const ulong kv_base = (ulong(j) * p.num_kv_heads + kv_head) * d_dim;
                float4 k0 = float4(0.0f), k1 = float4(0.0f);
                float4 v0 = float4(0.0f), v1 = float4(0.0f);
                float partial = 0.0f;
                if (lane < num_cols) {
                    k0 = float4(*(const device half4 *)(k_tree + kv_base + lane * 4u));
                    v0 = float4(*(const device half4 *)(v_tree + kv_base + lane * 4u));
                    partial += dot(q_reg[0], k0);
                }
                if (lane + 32u < num_cols) {
                    k1 = float4(*(const device half4 *)(k_tree + kv_base + (lane + 32u) * 4u));
                    v1 = float4(*(const device half4 *)(v_tree + kv_base + (lane + 32u) * 4u));
                    partial += dot(q_reg[1], k1);
                }
                const float score = simd_sum(partial) * scale_log2e;
                const float new_m = max(m_val, score);
                const float corr = exp2(m_val - new_m);
                const float p_w = exp2(score - new_m);
                l_val = fma(l_val, corr, p_w);
                acc_reg[0] = fma(acc_reg[0], float4(corr), v0 * p_w);
                acc_reg[1] = fma(acc_reg[1], float4(corr), v1 * p_w);
                m_val = new_m;
            }
        }
    }

    if (active_q) {
        const float inv_l = (l_val > 0.0f) ? (1.0f / l_val) : 0.0f;
        if (lane < num_cols) {
            *(device half4 *)(out + q_row_off + lane * 4u) = half4(acc_reg[0] * inv_l);
        }
        if (lane + 32u < num_cols) {
            *(device half4 *)(out + q_row_off + (lane + 32u) * 4u) = half4(acc_reg[1] * inv_l);
        }
    }
}

// ===========================================================================
// Section C: G10 — 4-Branch Gated Residual (GR)
//
// Stream layout: `[4, branch_token_stride, hidden_dim]` (`f16` or `bf16`,
// controlled by `GR_STREAM_BF16`), activations `x_in`/`y_delta`/`x0`/`out`
// (`f16` or `bf16`, controlled by `GR_ACT_BF16`), all arithmetic in FP32.
// ===========================================================================

struct GrParams {
    uint m;
    uint hidden_dim;
    uint branch_token_stride;
    uint mode;
    float w0;
    float w1;
    float w2;
    float w3;
    uint _pad0;
    uint _pad1;
    uint _pad2;
    uint _pad3;
};

static_assert(sizeof(GrParams) == 48, "GrParams must be 48 bytes");

// Stream-0 initializer: `stream[0, m, d] = x0[m, d]`; `stream[1..3, m, d] = 0`
// (or broadcasts `x0` to all 4 branches when `p.mode == 1`).
kernel void ami_gr_init_stream0(const device uchar *x0 [[buffer(0)]],
                                device uchar *stream [[buffer(1)]],
                                constant GrParams &p [[buffer(2)]],
                                uint gid [[thread_position_in_grid]]) {
    const uint total = p.m * p.hidden_dim;
    if (gid >= total) {
        return;
    }

    const uint tok = gid / p.hidden_dim;
    const uint d = gid - tok * p.hidden_dim;
    const ulong branch_elems = ulong(p.branch_token_stride) * p.hidden_dim;
    const ulong tok_off = ulong(tok) * p.hidden_dim + d;
    const float v0 = hybrid_load_f16_or_bf16(x0, tok_off, GR_ACT_BF16);
    const float v_other = (p.mode != 0u) ? v0 : 0.0f;

    hybrid_store_f16_or_bf16(stream, 0ul * branch_elems + tok_off, v0, GR_STREAM_BF16);
    hybrid_store_f16_or_bf16(stream, 1ul * branch_elems + tok_off, v_other, GR_STREAM_BF16);
    hybrid_store_f16_or_bf16(stream, 2ul * branch_elems + tok_off, v_other, GR_STREAM_BF16);
    hybrid_store_f16_or_bf16(stream, 3ul * branch_elems + tok_off, v_other, GR_STREAM_BF16);
}

// Fused 4-branch read-gate: `x_in[m, d] = sum_{b=0..3} g_read[m, b] * stream[b, m, d]`.
kernel void ami_gr_read_gate(const device uchar *stream [[buffer(0)]],
                             const device float4 *g_read [[buffer(1)]],
                             device uchar *x_in [[buffer(2)]],
                             constant GrParams &p [[buffer(3)]],
                             uint gid [[thread_position_in_grid]]) {
    const uint total = p.m * p.hidden_dim;
    if (gid >= total) {
        return;
    }

    const uint tok = gid / p.hidden_dim;
    const uint d = gid - tok * p.hidden_dim;
    const ulong branch_elems = ulong(p.branch_token_stride) * p.hidden_dim;
    const ulong tok_off = ulong(tok) * p.hidden_dim + d;
    const float4 g = g_read[tok];

    const float s0 = hybrid_load_f16_or_bf16(stream, 0ul * branch_elems + tok_off, GR_STREAM_BF16);
    const float s1 = hybrid_load_f16_or_bf16(stream, 1ul * branch_elems + tok_off, GR_STREAM_BF16);
    const float s2 = hybrid_load_f16_or_bf16(stream, 2ul * branch_elems + tok_off, GR_STREAM_BF16);
    const float s3 = hybrid_load_f16_or_bf16(stream, 3ul * branch_elems + tok_off, GR_STREAM_BF16);

    const float acc = fma(g.w, s3, fma(g.z, s2, fma(g.y, s1, g.x * s0)));
    hybrid_store_f16_or_bf16(x_in, tok_off, acc, GR_ACT_BF16);
}

// Fused 4-branch write-gate:
// `stream[b, m, d] = decay[m, b] * stream[b, m, d] + g_write[m, b] * y_delta[m, d]`.
kernel void ami_gr_write_gate(device uchar *stream [[buffer(0)]],
                              const device uchar *y_delta [[buffer(1)]],
                              const device float4 *decay [[buffer(2)]],
                              const device float4 *g_write [[buffer(3)]],
                              constant GrParams &p [[buffer(4)]],
                              uint gid [[thread_position_in_grid]]) {
    const uint total = p.m * p.hidden_dim;
    if (gid >= total) {
        return;
    }

    const uint tok = gid / p.hidden_dim;
    const uint d = gid - tok * p.hidden_dim;
    const ulong branch_elems = ulong(p.branch_token_stride) * p.hidden_dim;
    const ulong tok_off = ulong(tok) * p.hidden_dim + d;

    const float yd = hybrid_load_f16_or_bf16(y_delta, tok_off, GR_ACT_BF16);
    const float4 dec = decay[tok];
    const float4 gw = g_write[tok];

    const ulong o0 = 0ul * branch_elems + tok_off;
    const ulong o1 = 1ul * branch_elems + tok_off;
    const ulong o2 = 2ul * branch_elems + tok_off;
    const ulong o3 = 3ul * branch_elems + tok_off;

    const float n0 = fma(dec.x, hybrid_load_f16_or_bf16(stream, o0, GR_STREAM_BF16), gw.x * yd);
    const float n1 = fma(dec.y, hybrid_load_f16_or_bf16(stream, o1, GR_STREAM_BF16), gw.y * yd);
    const float n2 = fma(dec.z, hybrid_load_f16_or_bf16(stream, o2, GR_STREAM_BF16), gw.z * yd);
    const float n3 = fma(dec.w, hybrid_load_f16_or_bf16(stream, o3, GR_STREAM_BF16), gw.w * yd);

    hybrid_store_f16_or_bf16(stream, o0, n0, GR_STREAM_BF16);
    hybrid_store_f16_or_bf16(stream, o1, n1, GR_STREAM_BF16);
    hybrid_store_f16_or_bf16(stream, o2, n2, GR_STREAM_BF16);
    hybrid_store_f16_or_bf16(stream, o3, n3, GR_STREAM_BF16);
}

// 4-branch stream collapse into single-stream `out [m, hidden_dim]`:
//   mode == 0: stream 0
//   mode == 1: sum of all 4 branches
//   mode == 2: mean of all 4 branches (`0.25 * sum`)
//   mode == 3: weighted sum using uniform `(p.w0, p.w1, p.w2, p.w3)`
//   mode == 4: weighted sum using per-token `weights [m, 4]`
kernel void ami_gr_collapse(const device uchar *stream [[buffer(0)]],
                            const device float4 *weights [[buffer(1)]],
                            device uchar *out [[buffer(2)]],
                            constant GrParams &p [[buffer(3)]],
                            uint gid [[thread_position_in_grid]]) {
    const uint total = p.m * p.hidden_dim;
    if (gid >= total) {
        return;
    }

    const uint tok = gid / p.hidden_dim;
    const uint d = gid - tok * p.hidden_dim;
    const ulong branch_elems = ulong(p.branch_token_stride) * p.hidden_dim;
    const ulong tok_off = ulong(tok) * p.hidden_dim + d;

    const float s0 = hybrid_load_f16_or_bf16(stream, 0ul * branch_elems + tok_off, GR_STREAM_BF16);
    if (p.mode == 0u) {
        hybrid_store_f16_or_bf16(out, tok_off, s0, GR_ACT_BF16);
        return;
    }

    const float s1 = hybrid_load_f16_or_bf16(stream, 1ul * branch_elems + tok_off, GR_STREAM_BF16);
    const float s2 = hybrid_load_f16_or_bf16(stream, 2ul * branch_elems + tok_off, GR_STREAM_BF16);
    const float s3 = hybrid_load_f16_or_bf16(stream, 3ul * branch_elems + tok_off, GR_STREAM_BF16);

    float acc;
    if (p.mode == 1u) {
        acc = (s0 + s1) + (s2 + s3);
    } else if (p.mode == 2u) {
        acc = 0.25f * ((s0 + s1) + (s2 + s3));
    } else {
        const float4 w = (p.mode == 4u) ? weights[tok] : float4(p.w0, p.w1, p.w2, p.w3);
        acc = fma(w.w, s3, fma(w.z, s2, fma(w.y, s1, w.x * s0)));
    }
    hybrid_store_f16_or_bf16(out, tok_off, acc, GR_ACT_BF16);
}

// ===========================================================================
// Section D: Graph Executor Fused Gate, Conv1D, and Loop-Blend Helpers
// ===========================================================================

struct GrComputeGatesParams {
    uint m;
    uint hidden_dim;
    uint branch_token_stride;
    uint _pad0;
};

static_assert(sizeof(GrComputeGatesParams) == 16, "GrComputeGatesParams must be 16 bytes");

// Computes 4-branch GR per-token gates from stream branch 0 (`f16` / `bf16` weights):
//   g_read[m, b]  = sigmoid(dot(stream[0, m, :], w_read[b, :]))
//   g_write[m, b] = sigmoid(dot(stream[0, m, :], w_write[b, :]))
//   decay[m, b]   = sigmoid(dot(stream[0, m, :], w_decay[b, :]))
// Dispatch: `m * 4` threadgroups of 32 threads (1 SIMD group per (tok, branch)).
kernel void ami_gr_compute_gates(const device half *stream [[buffer(0)]],
                                 const device uchar *w_read [[buffer(1)]],
                                 const device uchar *w_write [[buffer(2)]],
                                 const device uchar *w_decay [[buffer(3)]],
                                 device float *g_read_out [[buffer(4)]],
                                 device float *g_write_out [[buffer(5)]],
                                 device float *decay_out [[buffer(6)]],
                                 constant GrComputeGatesParams &p [[buffer(7)]],
                                 uint tgid [[threadgroup_position_in_grid]],
                                 uint lane [[thread_index_in_simdgroup]]) {
    const uint tok = tgid >> 2u;
    const uint b = tgid & 3u;
    if (tok >= p.m) {
        return;
    }

    const device half *x_row = stream + ulong(tok) * p.hidden_dim;
    const ulong w_off = ulong(b) * p.hidden_dim;
    float acc_r = 0.0f;
    float acc_w = 0.0f;
    float acc_d = 0.0f;
    for (uint d = lane; d < p.hidden_dim; d += 32u) {
        const float xv = float(x_row[d]);
        acc_r = fma(xv, hybrid_load_f16_or_bf16(w_read, w_off + d, HYBRID_NORM_W_BF16), acc_r);
        acc_w = fma(xv, hybrid_load_f16_or_bf16(w_write, w_off + d, HYBRID_NORM_W_BF16), acc_w);
        acc_d = fma(xv, hybrid_load_f16_or_bf16(w_decay, w_off + d, HYBRID_NORM_W_BF16), acc_d);
    }
    acc_r = simd_sum(acc_r);
    acc_w = simd_sum(acc_w);
    acc_d = simd_sum(acc_d);
    if (lane == 0u) {
        const uint out_idx = tok * 4u + b;
        g_read_out[out_idx] = hybrid_sigmoid(acc_r);
        g_write_out[out_idx] = hybrid_sigmoid(acc_w);
        decay_out[out_idx] = hybrid_sigmoid(acc_d);
    }
}

struct GdnComputeGatesParams {
    uint m;
    uint hidden_dim;
    uint num_v_heads;
    uint _pad0;
};

static_assert(sizeof(GdnComputeGatesParams) == 16, "GdnComputeGatesParams must be 16 bytes");

// Computes Qwen3.8 Gated DeltaNet per-head log-decay `g` and pre-sigmoid `beta`:
//   a = dot(x[tok, :], w_a[vh, :]) + dt_bias[vh]
//   g[tok, vh] = -exp(a_log[vh]) * softplus(a)    (<= 0 so exp(g) in (0, 1])
//   beta[tok, vh] = dot(x[tok, :], w_b[vh, :])
// Dispatch: `m * num_v_heads` threadgroups of 32 threads.
kernel void ami_gdn_compute_gates(const device half *x [[buffer(0)]],
                                  const device uchar *w_a [[buffer(1)]],
                                  const device uchar *w_b [[buffer(2)]],
                                  const device float *a_log [[buffer(3)]],
                                  const device float *dt_bias [[buffer(4)]],
                                  device float *g_out [[buffer(5)]],
                                  device float *beta_out [[buffer(6)]],
                                  constant GdnComputeGatesParams &p [[buffer(7)]],
                                  uint tgid [[threadgroup_position_in_grid]],
                                  uint lane [[thread_index_in_simdgroup]]) {
    const uint tok = tgid / p.num_v_heads;
    const uint vh = tgid - tok * p.num_v_heads;
    if (tok >= p.m) {
        return;
    }

    const device half *x_row = x + ulong(tok) * p.hidden_dim;
    const ulong w_off = ulong(vh) * p.hidden_dim;
    float acc_a = 0.0f;
    float acc_b = 0.0f;
    for (uint d = lane; d < p.hidden_dim; d += 32u) {
        const float xv = float(x_row[d]);
        acc_a = fma(xv, hybrid_load_f16_or_bf16(w_a, w_off + d, HYBRID_NORM_W_BF16), acc_a);
        acc_b = fma(xv, hybrid_load_f16_or_bf16(w_b, w_off + d, HYBRID_NORM_W_BF16), acc_b);
    }
    acc_a = simd_sum(acc_a);
    acc_b = simd_sum(acc_b);
    if (lane == 0u) {
        const float a = acc_a + dt_bias[vh];
        const float sp = (a > 20.0f) ? a : log(1.0f + exp(a));
        const uint out_idx = tok * p.num_v_heads + vh;
        g_out[out_idx] = -exp(a_log[vh]) * sp;
        beta_out[out_idx] = acc_b;
    }
}

struct GdnConv1dParams {
    uint m;
    uint gdn_qk;
    uint gdn_v;
    uint conv_k;
    uint is_tree;
    uint write_state;
    uint _pad0;
    uint _pad1;
};

static_assert(sizeof(GdnConv1dParams) == 32, "GdnConv1dParams must be 32 bytes");

static inline float load_qkv_channel(const device half *q,
                                     const device half *k,
                                     const device half *v,
                                     uint tok,
                                     uint c,
                                     uint gdn_qk,
                                     uint gdn_v) {
    if (c < gdn_qk) {
        return float(q[ulong(tok) * gdn_qk + c]);
    }
    if (c < 2u * gdn_qk) {
        return float(k[ulong(tok) * gdn_qk + (c - gdn_qk)]);
    }
    return float(v[ulong(tok) * gdn_v + (c - 2u * gdn_qk)]);
}

static inline void store_qkv_channel(device half *q,
                                     device half *k,
                                     device half *v,
                                     uint tok,
                                     uint c,
                                     uint gdn_qk,
                                     uint gdn_v,
                                     float val) {
    const half hval = half(val);
    if (c < gdn_qk) {
        q[ulong(tok) * gdn_qk + c] = hval;
    } else if (c < 2u * gdn_qk) {
        k[ulong(tok) * gdn_qk + (c - gdn_qk)] = hval;
    } else {
        v[ulong(tok) * gdn_v + (c - 2u * gdn_qk)] = hval;
    }
}

// Causal depthwise Conv1D + SiLU over concatenated `(q, k, v)` channels
// (`conv_dim = 2 * gdn_qk + gdn_v`, `conv_k in [1..8]`).
// Supports decode (`m=1, is_tree=0`), prefill (`m>=1, is_tree=0`), and
// tree-verify (`is_tree=1` with 64-bit `ancestor_masks` without mutating
// committed `conv_state_in`).
kernel void ami_gdn_conv1d_silu(const device half *q_in [[buffer(0)]],
                                const device half *k_in [[buffer(1)]],
                                const device half *v_in [[buffer(2)]],
                                const device float *conv_w [[buffer(3)]],
                                const device half *conv_state_in [[buffer(4)]],
                                device half *conv_state_out [[buffer(5)]],
                                const device ulong *ancestor_masks [[buffer(6)]],
                                device half *q_out [[buffer(7)]],
                                device half *k_out [[buffer(8)]],
                                device half *v_out [[buffer(9)]],
                                constant GdnConv1dParams &p [[buffer(10)]],
                                uint gid [[thread_position_in_grid]]) {
    const uint conv_dim = 2u * p.gdn_qk + p.gdn_v;
    if (gid >= conv_dim) {
        return;
    }

    const uint hist = (p.conv_k > 0u) ? (p.conv_k - 1u) : 0u;
    const device float *w_row = conv_w + ulong(gid) * p.conv_k;

    if (p.is_tree == 0u) {
        for (uint tok = 0u; tok < p.m; ++tok) {
            float acc = 0.0f;
            for (uint tap = 0u; tap < p.conv_k; ++tap) {
                const int rel = int(tok) - int(hist - tap);
                float xv;
                if (rel < 0) {
                    const uint s_idx = uint(int(hist) + rel);
                    xv = float(conv_state_in[ulong(s_idx) * conv_dim + gid]);
                } else {
                    xv = load_qkv_channel(q_in, k_in, v_in, uint(rel), gid, p.gdn_qk, p.gdn_v);
                }
                acc = fma(xv, w_row[tap], acc);
            }
            store_qkv_channel(q_out, k_out, v_out, tok, gid, p.gdn_qk, p.gdn_v, hybrid_fast_silu(acc));
        }

        if (p.write_state != 0u && hist > 0u) {
            for (uint s = 0u; s < hist; ++s) {
                const int rel = int(p.m) - int(hist - s);
                float xv;
                if (rel < 0) {
                    const uint old_s = uint(int(hist) + rel);
                    xv = float(conv_state_in[ulong(old_s) * conv_dim + gid]);
                } else {
                    xv = load_qkv_channel(q_in, k_in, v_in, uint(rel), gid, p.gdn_qk, p.gdn_v);
                }
                conv_state_out[ulong(s) * conv_dim + gid] = half(xv);
            }
        }
    } else {
        // Tree verification: for each node, walk up to `hist` strict ancestors
        // from `ancestor_masks[node]` (most recent first), falling back to `conv_state_in`.
        for (uint node = 0u; node < p.m; ++node) {
            uint anc_nodes[8];
            uint n_anc = 0u;
            const ulong mask = ancestor_masks[node];
            if (node > 0u && hist > 0u) {
                for (int cand = int(node) - 1; cand >= 0 && n_anc < hist; --cand) {
                    if (((mask >> uint(cand)) & 1ul) != 0ul) {
                        anc_nodes[n_anc++] = uint(cand);
                    }
                }
            }
            float acc = 0.0f;
            for (uint tap = 0u; tap < p.conv_k; ++tap) {
                float xv;
                if (tap == hist) {
                    xv = load_qkv_channel(q_in, k_in, v_in, node, gid, p.gdn_qk, p.gdn_v);
                } else {
                    const uint back = hist - tap; // 1 = parent, 2 = grandparent, ...
                    if (back <= n_anc) {
                        xv = load_qkv_channel(q_in, k_in, v_in, anc_nodes[back - 1u], gid, p.gdn_qk, p.gdn_v);
                    } else {
                        const uint past_back = back - n_anc; // 1 = newest slot in conv_state_in
                        const uint s_idx = hist - past_back;
                        xv = float(conv_state_in[ulong(s_idx) * conv_dim + gid]);
                    }
                }
                acc = fma(xv, w_row[tap], acc);
            }
            store_qkv_channel(q_out, k_out, v_out, node, gid, p.gdn_qk, p.gdn_v, hybrid_fast_silu(acc));
        }
    }
}

struct GdnConv1dQParams {
    uint m;
    uint gdn_qk;
    uint gdn_v;
    uint conv_k;
    uint is_tree;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

static_assert(sizeof(GdnConv1dQParams) == 32, "GdnConv1dQParams must be 32 bytes");

// ADR-0019 `r >= 1` causal depthwise Conv1D + SiLU for GDN queries ONLY (`gid < gdn_qk`).
// Reads current-pass `q_cur[node]` at `tap == hist`, and `r = 0` ancestor history
// `q_hist_r0[anc]` (or committed `conv_state_in`) at `tap < hist`, writing ONLY `q_out`
// without mutating `tree_{q,k,v}_raw`, `tree_{k,v}`, or `conv_state_in`.
// Supports both linear causal prefill (`is_tree == 0`, arbitrary `m >= 1`) and
// speculative tree verification (`is_tree != 0`, `m <= 64` with 64-bit masks).
kernel void ami_gdn_conv1d_q_silu(const device half *q_cur [[buffer(0)]],
                                  const device half *q_hist_r0 [[buffer(1)]],
                                  const device float *conv_w [[buffer(2)]],
                                  const device half *conv_state_in [[buffer(3)]],
                                  const device ulong *ancestor_masks [[buffer(4)]],
                                  device half *q_out [[buffer(5)]],
                                  constant GdnConv1dQParams &p [[buffer(6)]],
                                  uint gid [[thread_position_in_grid]]) {
    if (gid >= p.gdn_qk) {
        return;
    }

    const uint conv_dim = 2u * p.gdn_qk + p.gdn_v;
    const uint hist = (p.conv_k > 0u) ? (p.conv_k - 1u) : 0u;
    const device float *w_row = conv_w + ulong(gid) * p.conv_k;

    if (p.is_tree == 0u) {
        for (uint tok = 0u; tok < p.m; ++tok) {
            float acc = 0.0f;
            for (uint tap = 0u; tap < p.conv_k; ++tap) {
                float xv;
                if (tap == hist) {
                    xv = float(q_cur[ulong(tok) * p.gdn_qk + gid]);
                } else {
                    const int rel = int(tok) - int(hist - tap);
                    if (rel < 0) {
                        const uint s_idx = uint(int(hist) + rel);
                        xv = float(conv_state_in[ulong(s_idx) * conv_dim + gid]);
                    } else {
                        xv = float(q_hist_r0[ulong(uint(rel)) * p.gdn_qk + gid]);
                    }
                }
                acc = fma(xv, w_row[tap], acc);
            }
            q_out[ulong(tok) * p.gdn_qk + gid] = half(hybrid_fast_silu(acc));
        }
        return;
    }

    for (uint node = 0u; node < p.m; ++node) {
        uint anc_nodes[8];
        uint n_anc = 0u;
        const ulong mask = ancestor_masks[node];
        if (node > 0u && hist > 0u) {
            for (int cand = int(node) - 1; cand >= 0 && n_anc < hist; --cand) {
                if (((mask >> uint(cand)) & 1ul) != 0ul) {
                    anc_nodes[n_anc++] = uint(cand);
                }
            }
        }
        float acc = 0.0f;
        for (uint tap = 0u; tap < p.conv_k; ++tap) {
            float xv;
            if (tap == hist) {
                xv = float(q_cur[ulong(node) * p.gdn_qk + gid]);
            } else {
                const uint back = hist - tap; // 1 = parent, 2 = grandparent, ...
                if (back <= n_anc) {
                    xv = float(q_hist_r0[ulong(anc_nodes[back - 1u]) * p.gdn_qk + gid]);
                } else {
                    const uint past_back = back - n_anc; // 1 = newest slot in conv_state_in
                    const uint s_idx = hist - past_back;
                    xv = float(conv_state_in[ulong(s_idx) * conv_dim + gid]);
                }
            }
            acc = fma(xv, w_row[tap], acc);
        }
        q_out[ulong(node) * p.gdn_qk + gid] = half(hybrid_fast_silu(acc));
    }
}

struct GdnConv1dCommitParams {
    uint gdn_qk;
    uint gdn_v;
    uint conv_k;
    uint accepted_count;
    ushort inline_nodes[64];
};

static_assert(sizeof(GdnConv1dCommitParams) == 144, "GdnConv1dCommitParams must be 144 bytes");

// Advances committed GDN Conv1D history along `accepted_nodes[0..accepted_count]`.
kernel void ami_gdn_conv1d_commit(const device half *q_raw [[buffer(0)]],
                                  const device half *k_raw [[buffer(1)]],
                                  const device half *v_raw [[buffer(2)]],
                                  const device half *conv_state_in [[buffer(3)]],
                                  device half *conv_state_out [[buffer(4)]],
                                  constant GdnConv1dCommitParams &p [[buffer(5)]],
                                  uint gid [[thread_position_in_grid]]) {
    const uint conv_dim = 2u * p.gdn_qk + p.gdn_v;
    const uint hist = (p.conv_k > 0u) ? (p.conv_k - 1u) : 0u;
    if (gid >= conv_dim || hist == 0u || p.accepted_count == 0u) {
        return;
    }

    float next_vals[8];
    for (uint s = 0u; s < hist; ++s) {
        const int rel = int(p.accepted_count) - int(hist - s);
        if (rel < 0) {
            const uint old_s = uint(int(hist) + rel);
            next_vals[s] = float(conv_state_in[ulong(old_s) * conv_dim + gid]);
        } else {
            const uint node = uint(p.inline_nodes[uint(rel)]);
            next_vals[s] = load_qkv_channel(q_raw, k_raw, v_raw, node, gid, p.gdn_qk, p.gdn_v);
        }
    }
    for (uint s = 0u; s < hist; ++s) {
        conv_state_out[ulong(s) * conv_dim + gid] = half(next_vals[s]);
    }
}

struct BlendF16Params {
    uint count;
    float a;
    float b;
    uint _pad0;
};

static_assert(sizeof(BlendF16Params) == 16, "BlendF16Params must be 16 bytes");

// Elementwise FP16 linear blend: `dst[i] = half(a * float(dst[i]) + b * float(src[i]))`.
// Used for native LoopLM `inject_x_init` (`0.5 * h + 0.5 * x_init` on `r >= 1`).
kernel void ami_blend_f16(device half *dst [[buffer(0)]],
                          const device half *src [[buffer(1)]],
                          constant BlendF16Params &p [[buffer(2)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= p.count) {
        return;
    }
    dst[gid] = half(fma(p.a, float(dst[gid]), p.b * float(src[gid])));
}

// ===========================================================================
// Milestone M6.5: C14 Multi-Slot Batched GDN Decode & R8 GDN Snapshot Clone
// ===========================================================================

struct GdnBatchDecodeParams {
    uint batch_size;
    uint num_k_heads;
    uint num_v_heads;
    uint head_k_dim;
    uint head_v_dim;
    uint slot_stride_f32;
    float q_scale;
    float l2_eps;
    float rms_eps;
    uint apply_l2_norm;
    uint apply_out_norm;
    uint apply_out_gate;
    uint per_head_norm;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

struct GdnCloneStateParams {
    uint num_gdn_layers;
    uint state_f32_per_layer;
    uint src_state_layer_stride_f32;
    uint dst_state_layer_stride_f32;
    uint conv_f16_per_layer;
    uint src_conv_layer_stride_f16;
    uint dst_conv_layer_stride_f16;
    uint _pad0;
};

static_assert(sizeof(GdnBatchDecodeParams) == 64, "GdnBatchDecodeParams must be 64 bytes");
static_assert(sizeof(GdnCloneStateParams) == 32, "GdnCloneStateParams must be 32 bytes");

// Multi-request batched Gated DeltaNet decode (`M_batch = 1..16+`) executing
// `batch_size` independent slots in ONE GPU pass using per-slot `state_in_slots[b]`
// and `state_out_slots[b]`. Bit-for-bit identical recurrence and RMSNorm + SiLU
// gate math to `ami_gdn_decode`.
//
// Specialized on GDN_HEAD_K_DIM ({64, 128, 0}) and HYBRID_NORM_W_BF16 ({false, true}).
// Grid: (num_v_heads, batch_size, 1) threadgroups, 256 threads (8 simdgroups).
kernel void ami_gdn_batch_decode(const device half *q [[buffer(0)]],
                                 const device half *k [[buffer(1)]],
                                 const device half *v [[buffer(2)]],
                                 const device float *g [[buffer(3)]],
                                 const device float *beta [[buffer(4)]],
                                 const device half *z [[buffer(5)]],
                                 const device uchar *norm_w [[buffer(6)]],
                                 const device float *state_in_pool [[buffer(7)]],
                                 device float *state_out_pool [[buffer(8)]],
                                 device half *out [[buffer(9)]],
                                 const device uint *state_in_slots [[buffer(10)]],
                                 const device uint *state_out_slots [[buffer(11)]],
                                 constant GdnBatchDecodeParams &p [[buffer(12)]],
                                 uint2 tg_pos [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]],
                                 uint sg_id [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
    const uint vh = tg_pos.x;
    const uint b = tg_pos.y;
    if (vh >= p.num_v_heads || b >= p.batch_size) {
        return;
    }

    const uint d_k = (GDN_HEAD_K_DIM > 0u) ? GDN_HEAD_K_DIM : p.head_k_dim;
    const uint d_v = p.head_v_dim;
    const uint v_per_k = p.num_v_heads / p.num_k_heads;
    const uint kh = vh / v_per_k;

    threadgroup float tg_q[256];
    threadgroup float tg_k[256];
    threadgroup float tg_o[256];
    threadgroup float tg_rms_partial[8];

    if (sg_id == 0u) {
        const ulong qk_base = (ulong(b) * p.num_k_heads + kh) * d_k;
        const device half *q_src = q + qk_base;
        const device half *k_src = k + qk_base;
        float q_sq = 0.0f;
        float k_sq = 0.0f;
        for (uint dk = lane; dk < d_k; dk += 32u) {
            const float qv = float(q_src[dk]);
            const float kv = float(k_src[dk]);
            tg_q[dk] = qv;
            tg_k[dk] = kv;
            q_sq = fma(qv, qv, q_sq);
            k_sq = fma(kv, kv, k_sq);
        }
        const float q_sum = simd_sum(q_sq);
        const float k_sum = simd_sum(k_sq);
        const float q_fac = (p.apply_l2_norm != 0u) ? (rsqrt(q_sum + p.l2_eps) * p.q_scale) : p.q_scale;
        const float k_fac = (p.apply_l2_norm != 0u) ? rsqrt(k_sum + p.l2_eps) : 1.0f;
        for (uint dk = lane; dk < d_k; dk += 32u) {
            tg_q[dk] *= q_fac;
            tg_k[dk] *= k_fac;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const ulong gb_idx = ulong(b) * p.num_v_heads + vh;
    const float alpha = exp(g[gb_idx]);
    const float beta_sig = hybrid_sigmoid(beta[gb_idx]);
    const uint in_slot = state_in_slots[b];
    const uint out_slot = state_out_slots[b];
    const ulong in_state_base = ulong(in_slot) * p.slot_stride_f32 + ulong(vh) * d_k * d_v;
    const ulong out_state_base = ulong(out_slot) * p.slot_stride_f32 + ulong(vh) * d_k * d_v;
    const ulong v_head_base = gb_idx * d_v;

    for (uint dv = sg_id; dv < d_v; dv += 8u) {
        float pred_acc = 0.0f;
        float s_col[8];
        if (GDN_HEAD_K_DIM == 64u) {
            for (uint r = 0u; r < 2u; r++) {
                const uint dk = lane + r * 32u;
                const float s_dec = state_in_pool[in_state_base + ulong(dk) * d_v + dv] * alpha;
                s_col[r] = s_dec;
                pred_acc = fma(s_dec, tg_k[dk], pred_acc);
            }
        } else if (GDN_HEAD_K_DIM == 128u) {
            for (uint r = 0u; r < 4u; r++) {
                const uint dk = lane + r * 32u;
                const float s_dec = state_in_pool[in_state_base + ulong(dk) * d_v + dv] * alpha;
                s_col[r] = s_dec;
                pred_acc = fma(s_dec, tg_k[dk], pred_acc);
            }
        } else {
            for (uint r = 0u; r < 8u; r++) {
                const uint dk = lane + r * 32u;
                if (dk < d_k) {
                    const float s_dec = state_in_pool[in_state_base + ulong(dk) * d_v + dv] * alpha;
                    s_col[r] = s_dec;
                    pred_acc = fma(s_dec, tg_k[dk], pred_acc);
                } else {
                    s_col[r] = 0.0f;
                }
            }
        }

        const float pred_v = simd_sum(pred_acc);
        const float v_val = float(v[v_head_base + dv]);
        const float delta = beta_sig * (v_val - pred_v);

        float o_acc = 0.0f;
        if (GDN_HEAD_K_DIM == 64u) {
            for (uint r = 0u; r < 2u; r++) {
                const uint dk = lane + r * 32u;
                const float s_new = fma(tg_k[dk], delta, s_col[r]);
                state_out_pool[out_state_base + ulong(dk) * d_v + dv] = s_new;
                o_acc = fma(s_new, tg_q[dk], o_acc);
            }
        } else if (GDN_HEAD_K_DIM == 128u) {
            for (uint r = 0u; r < 4u; r++) {
                const uint dk = lane + r * 32u;
                const float s_new = fma(tg_k[dk], delta, s_col[r]);
                state_out_pool[out_state_base + ulong(dk) * d_v + dv] = s_new;
                o_acc = fma(s_new, tg_q[dk], o_acc);
            }
        } else {
            for (uint r = 0u; r < 8u; r++) {
                const uint dk = lane + r * 32u;
                if (dk < d_k) {
                    const float s_new = fma(tg_k[dk], delta, s_col[r]);
                    state_out_pool[out_state_base + ulong(dk) * d_v + dv] = s_new;
                    o_acc = fma(s_new, tg_q[dk], o_acc);
                }
            }
        }

        const float o_val = simd_sum(o_acc);
        if (lane == 0u) {
            tg_o[dv] = (p.apply_out_norm != 0u) ? float(half(o_val)) : o_val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_rms = 1.0f;
    if (p.apply_out_norm != 0u) {
        if (sg_id == 0u) {
            float sum_sq = 0.0f;
            for (uint dv = lane; dv < d_v; dv += 32u) {
                const float ov = tg_o[dv];
                sum_sq = fma(ov, ov, sum_sq);
            }
            const float total_sq = simd_sum(sum_sq);
            if (lane == 0u) {
                tg_rms_partial[0] = rsqrt(total_sq / float(d_v) + p.rms_eps);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        inv_rms = tg_rms_partial[0];
    }

    const uint nw_base = (p.per_head_norm != 0u) ? (vh * d_v) : 0u;
    for (uint dv = tid; dv < d_v; dv += 256u) {
        float val = tg_o[dv];
        if (p.apply_out_norm != 0u) {
            val = (val * inv_rms) * hybrid_load_norm_w(norm_w, nw_base + dv, HYBRID_NORM_W_BF16);
        }
        if (p.apply_out_gate != 0u) {
            val *= hybrid_fast_silu(float(z[v_head_base + dv]));
        }
        out[v_head_base + dv] = half(val);
    }
}

// GPU-side snapshot clone (`R8`): copies a full multi-layer GDN recurrent state
// `[num_gdn_layers, num_v_heads, head_k_dim, head_v_dim]` (`f32`) AND optional
// `conv1d` history `[num_gdn_layers, conv_f16_per_layer]` (`f16`) between a
// request slot's `GdnStateArena` slab and a `PrefixCache` snapshot slab in ONE
// dispatch with zero CPU `memcpy`.
//
// Grid: (ceil(max(state_f32_per_layer, conv_f16_per_layer) / 1024), num_gdn_layers, 1)
//       threadgroups, 256 threads (each thread copies up to 4 elements).
kernel void ami_gdn_clone_state(const device float *src_state [[buffer(0)]],
                                device float *dst_state [[buffer(1)]],
                                const device half *src_conv [[buffer(2)]],
                                device half *dst_conv [[buffer(3)]],
                                constant GdnCloneStateParams &p [[buffer(4)]],
                                uint2 tg_pos [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]]) {
    const uint layer = tg_pos.y;
    if (layer >= p.num_gdn_layers) {
        return;
    }

    const uint base = (tg_pos.x * 256u + tid) * 4u;
    if (base < p.state_f32_per_layer) {
        const ulong s_off = ulong(layer) * p.src_state_layer_stride_f32 + base;
        const ulong d_off = ulong(layer) * p.dst_state_layer_stride_f32 + base;
        if (base + 4u <= p.state_f32_per_layer &&
            ((p.src_state_layer_stride_f32 | p.dst_state_layer_stride_f32) & 3u) == 0u) {
            *(device float4 *)(dst_state + d_off) = *(const device float4 *)(src_state + s_off);
        } else {
            const uint n = min(4u, p.state_f32_per_layer - base);
            for (uint i = 0u; i < n; ++i) {
                dst_state[d_off + i] = src_state[s_off + i];
            }
        }
    }

    if (base < p.conv_f16_per_layer) {
        const ulong s_off = ulong(layer) * p.src_conv_layer_stride_f16 + base;
        const ulong d_off = ulong(layer) * p.dst_conv_layer_stride_f16 + base;
        if (base + 4u <= p.conv_f16_per_layer &&
            ((p.src_conv_layer_stride_f16 | p.dst_conv_layer_stride_f16) & 3u) == 0u) {
            *(device half4 *)(dst_conv + d_off) = *(const device half4 *)(src_conv + s_off);
        } else {
            const uint n = min(4u, p.conv_f16_per_layer - base);
            for (uint i = 0u; i < n; ++i) {
                dst_conv[d_off + i] = src_conv[s_off + i];
            }
        }
    }
}

// ===========================================================================
// Milestone M6.8 (G15, ADR-0020): Wide-Chunk QSA & 128-Token SWA Ring Prefill
//   - `ami_qsa_prefill_qknorm_mrope`: QK-Norm + YaRN + 3D MRoPE for `1..1024+`
//     tokens with either direct bindless `HybridKvRing` tile writes or in-place
//     K rotation (preserving circular SWA ring history until after attention).
//   - `ami_qsa_prefill_kv_ring_commit`: race-free commit of the trailing
//     `min(num_tokens, ring_capacity)` K/V rows into bindless `HybridKvRing`.
//   - `ami_qsa_swa_prefill_attn`: causal sliding-window prefill attention over
//     unmodified circular `HybridKvRing` history + in-flight chunk K/V buffers,
//     strictly skipping upper-triangular (`kv_pos > q_pos`) and out-of-window
//     positions for any `num_q` (`1..1024+`).
// ===========================================================================

struct QsaKvTileWritePair {
    texture2d<half, access::write> k_tex;
    texture2d<half, access::write> v_tex;
};

static_assert(sizeof(QsaKvTileWritePair) == 16, "QsaKvTileWritePair must match QsaKvTilePair");

struct QsaPrefillRopeParams {
    uint num_tokens;
    uint num_q_heads;
    uint num_kv_heads;
    uint head_dim;
    uint rotary_dim;
    uint start_pos;
    uint chunk_tokens;
    uint num_retired_chunks;
    uint ring_capacity;
    uint mrope_s0;
    uint mrope_s1;
    uint mrope_s2;
    uint use_positions_3d;
    uint apply_qk_norm;
    uint kv_only;
    uint write_to_tiles;
    float rms_eps;
    float rope_theta;
    float yarn_factor;
    float yarn_mscale;
    float yarn_beta_fast;
    float yarn_beta_slow;
    uint original_max_pos;
    uint _pad0;
};

struct QsaPrefillKvCommitParams {
    uint num_tokens;
    uint num_kv_heads;
    uint head_dim;
    uint start_pos;
    uint chunk_tokens;
    uint num_retired_chunks;
    uint ring_capacity;
    uint _pad0;
};

struct QsaSwaPrefillAttnParams {
    uint num_q;
    uint q_start_pos;
    uint chunk_start_pos;
    uint kv_len;
    uint num_q_heads;
    uint num_kv_heads;
    uint head_dim;
    uint window_size;
    uint ring_capacity;
    uint chunk_tokens;
    float scale;
    uint num_retired_chunks;
    uint retired_plane_stride_bytes;
    uint retired_chunk_stride_bytes;
    uint _pad0;
    uint _pad1;
};

static_assert(sizeof(QsaPrefillRopeParams) == 96, "QsaPrefillRopeParams must be 96 bytes");
static_assert(sizeof(QsaPrefillKvCommitParams) == 32, "QsaPrefillKvCommitParams must be 32 bytes");
static_assert(sizeof(QsaSwaPrefillAttnParams) == 64, "QsaSwaPrefillAttnParams must be 64 bytes");

// Wide prefill QSA QK-Norm + YaRN + Interleaved 3D MRoPE (`1..1024+` tokens).
// Specialized on HYBRID_NORM_W_BF16 (`function_constant(1)`).
// Grid: ((kv_only ? num_kv_heads : num_q_heads + num_kv_heads), num_tokens, 1), 32 threads.
kernel void ami_qsa_prefill_qknorm_mrope(device half *q [[buffer(0)]],
                                         device half *k [[buffer(1)]],
                                         const device half *v [[buffer(2)]],
                                         const device uchar *q_norm [[buffer(3)]],
                                         const device uchar *k_norm [[buffer(4)]],
                                         const device uint *positions_3d [[buffer(5)]],
                                         const device QsaKvTileWritePair *kv_tiles [[buffer(6)]],
                                         constant QsaPrefillRopeParams &p [[buffer(7)]],
                                         uint2 tg_pos [[threadgroup_position_in_grid]],
                                         uint lane [[thread_index_in_simdgroup]]) {
    const uint total_heads = (p.kv_only != 0u) ? p.num_kv_heads : (p.num_q_heads + p.num_kv_heads);
    const uint tok = tg_pos.y;
    if (tg_pos.x >= total_heads || tok >= p.num_tokens) {
        return;
    }

    const bool is_q = (p.kv_only == 0u) && (tg_pos.x < p.num_q_heads);
    const uint head_idx = is_q ? tg_pos.x : (tg_pos.x - ((p.kv_only != 0u) ? 0u : p.num_q_heads));

    // When writing directly into a circular ring smaller than num_tokens, skip
    // overwritten prefix KV tokens so no two threadgroups race on a ring slot.
    if (!is_q && p.write_to_tiles != 0u && p.ring_capacity > 0u && (tok + p.ring_capacity < p.num_tokens)) {
        return;
    }

    threadgroup float vec[256];
    const uint d_dim = p.head_dim;
    const uint half_rot = p.rotary_dim >> 1;

    const ulong q_off = (ulong(tok) * p.num_q_heads + head_idx) * d_dim;
    const ulong kv_off = (ulong(tok) * p.num_kv_heads + head_idx) * d_dim;
    const device half *src = is_q ? (q + q_off) : (k + kv_off);

    float sum_sq = 0.0f;
    for (uint d = lane; d < d_dim; d += 32u) {
        const float val = float(src[d]);
        vec[d] = val;
        sum_sq = fma(val, val, sum_sq);
    }

    if (p.apply_qk_norm != 0u) {
        const float total_sq = simd_sum(sum_sq);
        const float inv_rms = rsqrt(total_sq / float(d_dim) + p.rms_eps);
        const device uchar *nw = is_q ? q_norm : k_norm;
        for (uint d = lane; d < d_dim; d += 32u) {
            vec[d] = (vec[d] * inv_rms) * hybrid_load_norm_w(nw, d, HYBRID_NORM_W_BF16);
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    const uint lin_pos = p.start_pos + tok;
    const uint pos_t = (p.use_positions_3d != 0u) ? positions_3d[ulong(tok) * 3ul + 0ul] : lin_pos;
    const uint pos_h = (p.use_positions_3d != 0u) ? positions_3d[ulong(tok) * 3ul + 1ul] : lin_pos;
    const uint pos_w = (p.use_positions_3d != 0u) ? positions_3d[ulong(tok) * 3ul + 2ul] : lin_pos;

    for (uint j = lane; j < half_rot; j += 32u) {
        const uint axis = mrope_interleaved_axis(j, p.mrope_s0, p.mrope_s1, p.mrope_s2);
        const uint axis_pos = (axis == 0u) ? pos_t : ((axis == 1u) ? pos_h : pos_w);
        const float inv_f = yarn_inv_freq(
            j, p.rotary_dim, p.rope_theta, p.yarn_factor,
            p.yarn_beta_fast, p.yarn_beta_slow, p.original_max_pos);
        const float angle = float(axis_pos) * inv_f;
        const float c = cos(angle) * p.yarn_mscale;
        const float s = sin(angle) * p.yarn_mscale;
        const float x0 = vec[j];
        const float x1 = vec[j + half_rot];
        vec[j] = fma(x0, c, -x1 * s);
        vec[j + half_rot] = fma(x1, c, x0 * s);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    if (is_q) {
        device half *dst = q + q_off;
        for (uint d = lane; d < d_dim; d += 32u) {
            dst[d] = half(vec[d]);
        }
    } else if (p.write_to_tiles != 0u) {
        const uint phys_pos = (p.ring_capacity > 0u) ? (lin_pos % p.ring_capacity) : lin_pos;
        const uint chunk = phys_pos / p.chunk_tokens;
        const uint slot = phys_pos - chunk * p.chunk_tokens;
        const uint active_chunk = (p.ring_capacity == 0u) ? (chunk - p.num_retired_chunks) : chunk;
        const uint row = head_idx * p.chunk_tokens + slot;
        const QsaKvTileWritePair tile = kv_tiles[active_chunk];
        const uint num_cols = d_dim >> 2;
        const device half4 *v_src = (const device half4 *)(v + kv_off);
        for (uint col = lane; col < num_cols; col += 32u) {
            const uint base = col * 4u;
            const half4 k4 = half4(half(vec[base + 0u]),
                                   half(vec[base + 1u]),
                                   half(vec[base + 2u]),
                                   half(vec[base + 3u]));
            tile.k_tex.write(k4, uint2(col, row));
            tile.v_tex.write(v_src[col], uint2(col, row));
        }
    } else {
        device half *k_dst = k + kv_off;
        for (uint d = lane; d < d_dim; d += 32u) {
            k_dst[d] = half(vec[d]);
        }
    }
}

// Race-free commit of the trailing `min(num_tokens, ring_capacity)` tokens
// from contiguous `k`/`v` buffers into bindless `HybridKvRing` `RGBA16Float` tiles.
// Grid: (num_kv_heads, commit_count, 1) threadgroups, 32 threads.
kernel void ami_qsa_prefill_kv_ring_commit(const device half *k [[buffer(0)]],
                                           const device half *v [[buffer(1)]],
                                           const device QsaKvTileWritePair *kv_tiles [[buffer(2)]],
                                           constant QsaPrefillKvCommitParams &p [[buffer(3)]],
                                           uint2 tg_pos [[threadgroup_position_in_grid]],
                                           uint lane [[thread_index_in_simdgroup]]) {
    const uint kvh = tg_pos.x;
    const uint commit_cnt = (p.ring_capacity > 0u) ? min(p.num_tokens, p.ring_capacity) : p.num_tokens;
    const uint ci = tg_pos.y;
    if (kvh >= p.num_kv_heads || ci >= commit_cnt) {
        return;
    }

    const uint tok = (p.num_tokens - commit_cnt) + ci;
    const uint pos = p.start_pos + tok;
    const uint phys_pos = (p.ring_capacity > 0u) ? (pos % p.ring_capacity) : pos;
    const uint chunk = phys_pos / p.chunk_tokens;
    const uint slot = phys_pos - chunk * p.chunk_tokens;
    const uint active_chunk = (p.ring_capacity == 0u) ? (chunk - p.num_retired_chunks) : chunk;
    const uint row = kvh * p.chunk_tokens + slot;

    const ulong kv_off = (ulong(tok) * p.num_kv_heads + kvh) * p.head_dim;
    const device half4 *k_src = (const device half4 *)(k + kv_off);
    const device half4 *v_src = (const device half4 *)(v + kv_off);
    const QsaKvTileWritePair tile = kv_tiles[active_chunk];
    const uint num_cols = p.head_dim >> 2;
    for (uint col = lane; col < num_cols; col += 32u) {
        tile.k_tex.write(k_src[col], uint2(col, row));
        tile.v_tex.write(v_src[col], uint2(col, row));
    }
}

// Causal SWA prefill attention (`1..1024+` queries) reading unmodified
// `HybridKvRing` prefix (`t < chunk_start_pos`) plus contiguous in-flight chunk
// K/V (`t >= chunk_start_pos`), strictly skipping `kv_pos > q_pos` and
// `q_pos - kv_pos >= window_size`.
// Specialized on QSA_HEAD_DIM (`64`, `128`, or `0` dynamic <= 256 multiple of 4).
// Grid: (num_kv_heads * q_groups_per_kv, num_q, 1) threadgroups, 256 threads.
kernel void ami_qsa_swa_prefill_attn(const device half *q [[buffer(0)]],
                                     const device QsaKvTilePair *kv_tiles [[buffer(1)]],
                                     const device half *k_chunk [[buffer(2)]],
                                     const device half *v_chunk [[buffer(3)]],
                                     device half *out [[buffer(4)]],
                                     const device uchar *retired_pool [[buffer(5)]],
                                     const device uint *retired_chunk_ids [[buffer(6)]],
                                     constant QsaSwaPrefillAttnParams &p [[buffer(7)]],
                                     uint2 tg_pos [[threadgroup_position_in_grid]],
                                     uint tid [[thread_index_in_threadgroup]],
                                     uint sg_id [[simdgroup_index_in_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint Q_TILE = 8u;
    constexpr float LOG2E = 1.4426950408889634f;

    const uint gqa_ratio = p.num_q_heads / p.num_kv_heads;
    const uint q_groups_per_kv = (gqa_ratio + Q_TILE - 1u) / Q_TILE;
    const uint kv_head = tg_pos.x / q_groups_per_kv;
    const uint q_grp = tg_pos.x % q_groups_per_kv;
    const uint qi = tg_pos.y;
    if (kv_head >= p.num_kv_heads || qi >= p.num_q) {
        return;
    }

    const uint q_start = kv_head * gqa_ratio + q_grp * Q_TILE;
    const uint q_end = min(q_start + Q_TILE, (kv_head + 1u) * gqa_ratio);
    const uint q_cnt = q_end - q_start;
    const bool active_q = (sg_id < q_cnt);
    const uint qh = q_start + min(sg_id, q_cnt - 1u);

    const uint d_dim = (QSA_HEAD_DIM > 0u) ? QSA_HEAD_DIM : p.head_dim;
    const uint num_cols = d_dim >> 2;
    const float scale_log2e = p.scale * LOG2E;

    // Strictly causal upper bound `q_pos + 1` and sliding-window lower bound.
    const uint q_pos = p.q_start_pos + qi;
    const uint t_hi = min(p.kv_len, q_pos + 1u);
    const uint t_lo = (p.window_size > 0u && t_hi > p.window_size) ? (t_hi - p.window_size) : 0u;
    const ulong q_row_off = (ulong(qi) * p.num_q_heads + qh) * d_dim;

    threadgroup half4 tg_k[256];
    threadgroup half4 tg_v[256];

    if (QSA_HEAD_DIM == 128u) {
        const float4 q_vec = active_q
            ? float4(*(const device half4 *)(q + q_row_off + lane * 4u))
            : float4(0.0f);
        float m_val = -INFINITY;
        float l_val = 0.0f;
        float4 acc = float4(0.0f);

        const uint load_tok = tid >> 5;  // 0..7
        const uint load_col = tid & 31u; // 0..31

        for (uint tile_t = t_lo; tile_t < t_hi; tile_t += 8u) {
            const uint tile_len = min(8u, t_hi - tile_t);
            if (load_tok < tile_len) {
                const uint t = tile_t + load_tok;
                if (t >= p.chunk_start_pos) {
                    const ulong c_off = (ulong(t - p.chunk_start_pos) * p.num_kv_heads + kv_head) * 128u + load_col * 4u;
                    tg_k[tid] = *(const device half4 *)(k_chunk + c_off);
                    tg_v[tid] = *(const device half4 *)(v_chunk + c_off);
                } else {
                    const uint phys_t = (p.ring_capacity > 0u) ? (t % p.ring_capacity) : t;
                    const uint chunk = phys_t / p.chunk_tokens;
                    const uint slot = phys_t - chunk * p.chunk_tokens;
                    const uint row = kv_head * p.chunk_tokens + slot;
                    if (p.ring_capacity == 0u && chunk < p.num_retired_chunks) {
                        const uint phys_slot = retired_chunk_ids[chunk];
                        const device uchar *k_plane = retired_pool + ulong(phys_slot) * p.retired_chunk_stride_bytes;
                        const device uchar *v_plane = k_plane + p.retired_plane_stride_bytes;
                        tg_k[tid] = hybrid_load_q8_half4(k_plane, row, 2u, load_col);
                        tg_v[tid] = hybrid_load_q8_half4(v_plane, row, 2u, load_col);
                    } else {
                        const uint active_chunk = (p.ring_capacity == 0u) ? (chunk - p.num_retired_chunks) : chunk;
                        tg_k[tid] = kv_tiles[active_chunk].k_tex.read(uint2(load_col, row));
                        tg_v[tid] = kv_tiles[active_chunk].v_tex.read(uint2(load_col, row));
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (active_q) {
                for (uint dt = 0u; dt < tile_len; dt++) {
                    const uint idx = dt * 32u + lane;
                    const float score = simd_sum(dot(q_vec, float4(tg_k[idx]))) * scale_log2e;
                    const float new_m = max(m_val, score);
                    const float corr = (l_val > 0.0f) ? exp2(m_val - new_m) : 0.0f;
                    const float p_w = exp2(score - new_m);
                    l_val = fma(l_val, corr, p_w);
                    acc = fma(acc, float4(corr), float4(tg_v[idx]) * p_w);
                    m_val = new_m;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (active_q) {
            const float inv_l = (l_val > 0.0f) ? (1.0f / l_val) : 0.0f;
            *(device half4 *)(out + q_row_off + lane * 4u) = half4(acc * inv_l);
        }
        return;
    }

    if (QSA_HEAD_DIM == 64u) {
        const bool active_lane = (lane < 16u);
        const float4 q_vec = (active_q && active_lane)
            ? float4(*(const device half4 *)(q + q_row_off + lane * 4u))
            : float4(0.0f);
        float m_val = -INFINITY;
        float l_val = 0.0f;
        float4 acc = float4(0.0f);

        const uint load_tok = tid >> 4;  // 0..15
        const uint load_col = tid & 15u; // 0..15

        for (uint tile_t = t_lo; tile_t < t_hi; tile_t += 16u) {
            const uint tile_len = min(16u, t_hi - tile_t);
            if (load_tok < tile_len) {
                const uint t = tile_t + load_tok;
                if (t >= p.chunk_start_pos) {
                    const ulong c_off = (ulong(t - p.chunk_start_pos) * p.num_kv_heads + kv_head) * 64u + load_col * 4u;
                    tg_k[tid] = *(const device half4 *)(k_chunk + c_off);
                    tg_v[tid] = *(const device half4 *)(v_chunk + c_off);
                } else {
                    const uint phys_t = (p.ring_capacity > 0u) ? (t % p.ring_capacity) : t;
                    const uint chunk = phys_t / p.chunk_tokens;
                    const uint slot = phys_t - chunk * p.chunk_tokens;
                    const uint row = kv_head * p.chunk_tokens + slot;
                    if (p.ring_capacity == 0u && chunk < p.num_retired_chunks) {
                        const uint phys_slot = retired_chunk_ids[chunk];
                        const device uchar *k_plane = retired_pool + ulong(phys_slot) * p.retired_chunk_stride_bytes;
                        const device uchar *v_plane = k_plane + p.retired_plane_stride_bytes;
                        tg_k[tid] = hybrid_load_q8_half4(k_plane, row, 1u, load_col);
                        tg_v[tid] = hybrid_load_q8_half4(v_plane, row, 1u, load_col);
                    } else {
                        const uint active_chunk = (p.ring_capacity == 0u) ? (chunk - p.num_retired_chunks) : chunk;
                        tg_k[tid] = kv_tiles[active_chunk].k_tex.read(uint2(load_col, row));
                        tg_v[tid] = kv_tiles[active_chunk].v_tex.read(uint2(load_col, row));
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (active_q) {
                for (uint dt = 0u; dt < tile_len; dt++) {
                    const uint idx = dt * 16u + (lane & 15u);
                    const float partial = active_lane ? dot(q_vec, float4(tg_k[idx])) : 0.0f;
                    const float score = simd_sum(partial) * scale_log2e;
                    const float new_m = max(m_val, score);
                    const float corr = (l_val > 0.0f) ? exp2(m_val - new_m) : 0.0f;
                    const float p_w = exp2(score - new_m);
                    l_val = fma(l_val, corr, p_w);
                    if (active_lane) {
                        acc = fma(acc, float4(corr), float4(tg_v[idx]) * p_w);
                    }
                    m_val = new_m;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (active_q && active_lane) {
            const float inv_l = (l_val > 0.0f) ? (1.0f / l_val) : 0.0f;
            *(device half4 *)(out + q_row_off + lane * 4u) = half4(acc * inv_l);
        }
        return;
    }

    // Dynamic head_dim <= 256 (multiple of 4).
    const uint groups_per_head = d_dim >> 6;
    float4 q_reg[2] = {float4(0.0f), float4(0.0f)};
    float4 acc_reg[2] = {float4(0.0f), float4(0.0f)};
    if (active_q) {
        if (lane < num_cols) {
            q_reg[0] = float4(*(const device half4 *)(q + q_row_off + lane * 4u));
        }
        if (lane + 32u < num_cols) {
            q_reg[1] = float4(*(const device half4 *)(q + q_row_off + (lane + 32u) * 4u));
        }
    }
    float m_val = -INFINITY;
    float l_val = 0.0f;

    for (uint t = t_lo; t < t_hi; t++) {
        if (active_q) {
            float4 k0 = float4(0.0f), k1 = float4(0.0f);
            float4 v0 = float4(0.0f), v1 = float4(0.0f);
            float partial = 0.0f;
            if (t >= p.chunk_start_pos) {
                const ulong c_base = (ulong(t - p.chunk_start_pos) * p.num_kv_heads + kv_head) * d_dim;
                if (lane < num_cols) {
                    k0 = float4(*(const device half4 *)(k_chunk + c_base + lane * 4u));
                    v0 = float4(*(const device half4 *)(v_chunk + c_base + lane * 4u));
                    partial += dot(q_reg[0], k0);
                }
                if (lane + 32u < num_cols) {
                    k1 = float4(*(const device half4 *)(k_chunk + c_base + (lane + 32u) * 4u));
                    v1 = float4(*(const device half4 *)(v_chunk + c_base + (lane + 32u) * 4u));
                    partial += dot(q_reg[1], k1);
                }
            } else {
                const uint phys_t = (p.ring_capacity > 0u) ? (t % p.ring_capacity) : t;
                const uint chunk = phys_t / p.chunk_tokens;
                const uint slot = phys_t - chunk * p.chunk_tokens;
                const uint row = kv_head * p.chunk_tokens + slot;
                if (p.ring_capacity == 0u && chunk < p.num_retired_chunks) {
                    const uint phys_slot = retired_chunk_ids[chunk];
                    const device uchar *k_plane = retired_pool + ulong(phys_slot) * p.retired_chunk_stride_bytes;
                    const device uchar *v_plane = k_plane + p.retired_plane_stride_bytes;
                    if (lane < num_cols) {
                        k0 = float4(hybrid_load_q8_half4(k_plane, row, groups_per_head, lane));
                        v0 = float4(hybrid_load_q8_half4(v_plane, row, groups_per_head, lane));
                        partial += dot(q_reg[0], k0);
                    }
                    if (lane + 32u < num_cols) {
                        k1 = float4(hybrid_load_q8_half4(k_plane, row, groups_per_head, lane + 32u));
                        v1 = float4(hybrid_load_q8_half4(v_plane, row, groups_per_head, lane + 32u));
                        partial += dot(q_reg[1], k1);
                    }
                } else {
                    const uint active_chunk = (p.ring_capacity == 0u) ? (chunk - p.num_retired_chunks) : chunk;
                    if (lane < num_cols) {
                        k0 = float4(kv_tiles[active_chunk].k_tex.read(uint2(lane, row)));
                        v0 = float4(kv_tiles[active_chunk].v_tex.read(uint2(lane, row)));
                        partial += dot(q_reg[0], k0);
                    }
                    if (lane + 32u < num_cols) {
                        k1 = float4(kv_tiles[active_chunk].k_tex.read(uint2(lane + 32u, row)));
                        v1 = float4(kv_tiles[active_chunk].v_tex.read(uint2(lane + 32u, row)));
                        partial += dot(q_reg[1], k1);
                    }
                }
            }
            const float score = simd_sum(partial) * scale_log2e;
            const float new_m = max(m_val, score);
            const float corr = (l_val > 0.0f) ? exp2(m_val - new_m) : 0.0f;
            const float p_w = exp2(score - new_m);
            l_val = fma(l_val, corr, p_w);
            acc_reg[0] = fma(acc_reg[0], float4(corr), v0 * p_w);
            acc_reg[1] = fma(acc_reg[1], float4(corr), v1 * p_w);
            m_val = new_m;
        }
    }

    if (active_q) {
        const float inv_l = (l_val > 0.0f) ? (1.0f / l_val) : 0.0f;
        if (lane < num_cols) {
            *(device half4 *)(out + q_row_off + lane * 4u) = half4(acc_reg[0] * inv_l);
        }
        if (lane + 32u < num_cols) {
            *(device half4 *)(out + q_row_off + (lane + 32u) * 4u) = half4(acc_reg[1] * inv_l);
        }
    }
}

// ===========================================================================
// Section E: M7 (R7 + G11) — MLA + CSA2 (Full / Reindex / Reuse) + FP4 KV
// (`E2M1` + `E4M3` per-16-channel scale, 890 B/tok geometry) + SWA Replay
// ===========================================================================

static inline float mla_e2m1_to_f32(uint nib) {
    const uint mag = nib & 0x7u;
    float v;
    switch (mag) {
        case 0u: v = 0.0f; break;
        case 1u: v = 0.5f; break;
        case 2u: v = 1.0f; break;
        case 3u: v = 1.5f; break;
        case 4u: v = 2.0f; break;
        case 5u: v = 3.0f; break;
        case 6u: v = 4.0f; break;
        default: v = 6.0f; break;
    }
    return ((nib & 0x8u) != 0u && mag != 0u) ? -v : v;
}

static inline uchar mla_f32_to_e2m1(float x) {
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

// OCP FP8 E4M3FN: 1 sign, 4 exponent (bias 7), 3 mantissa, max finite 448.0.
static inline float mla_e4m3_to_f32(uchar b) {
    const uint s = (uint(b) >> 7) & 1u;
    const uint e = (uint(b) >> 3) & 0xFu;
    const uint m = uint(b) & 0x7u;
    float val;
    if (e == 0u) {
        val = float(m) * (1.0f / 512.0f);
    } else if (e == 15u && m == 7u) {
        val = 0.0f;
    } else {
        val = ldexp(float(8u + m), int(e) - 10);
    }
    return (s != 0u) ? -val : val;
}

static inline uchar mla_f32_to_e4m3(float x) {
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

struct MlaFp4KvWriteParams {
    uint num_tokens;
    uint start_pos;
    uint kv_lora_rank;
    uint qk_rope_dim;
    uint rope_tail_format; // 0 = f16, 1 = e4m3
    uint max_tokens;
    uint swa_ring_capacity; // 0 = disabled
    uint flags;             // bit 0: CED encoder->decoder projection
    ulong e2m1_plane_offset;
    ulong scale_plane_offset;
    ulong rope_plane_offset;
    ulong swa_rope_plane_offset;
};

static_assert(sizeof(MlaFp4KvWriteParams) == 64, "MlaFp4KvWriteParams must be 64 bytes");

struct Csa2SummaryParams {
    uint kv_len;
    uint kv_lora_rank;
    uint index_dim;
    uint block_tokens;
    uint coarse_group_blocks;
    uint start_coarse_block;
    uint num_coarse_blocks;
    uint _pad0;
    ulong e2m1_plane_offset;
    ulong scale_plane_offset;
    ulong fine_plane_offset;
    ulong coarse_plane_offset;
};

static_assert(sizeof(Csa2SummaryParams) == 64, "Csa2SummaryParams must be 64 bytes");

struct Csa2IndexParams {
    uint num_nodes;
    uint num_q_heads;
    uint kv_lora_rank;
    uint index_dim;
    uint kv_len;
    uint encoder_prefix_len;
    uint causal_q_start_pos; // 0xFFFFFFFF = non-causal / full visibility
    uint swa_window;
    uint block_tokens;
    uint coarse_group_blocks;
    uint top_k_coarse;
    uint top_k_blocks;
    uint num_local_blocks;
    uint has_batch_slots;
    float scale;
    uint _pad0;
    ulong fine_batch_stride_bytes;
    ulong coarse_batch_stride_bytes;
};

static_assert(sizeof(Csa2IndexParams) == 80, "Csa2IndexParams must be 80 bytes");

struct MlaCsa2AttnParams {
    uint num_nodes;
    uint num_q_heads;
    uint kv_lora_rank;
    uint qk_rope_dim;
    uint rope_tail_format; // 0 = f16, 1 = e4m3
    uint kv_len;
    uint encoder_prefix_len;
    uint causal_q_start_pos; // 0xFFFFFFFF = non-causal / full visibility
    uint swa_window;
    uint swa_ring_capacity;
    uint block_tokens;
    uint top_k_blocks;
    uint has_tree;
    uint has_batch_slots;
    float scale;
    uint _pad0;
    ulong e2m1_plane_offset;
    ulong scale_plane_offset;
    ulong rope_plane_offset;
    ulong swa_rope_plane_offset;
    ulong kv_batch_stride_bytes;
    ulong swa_batch_stride_bytes;
};

static_assert(sizeof(MlaCsa2AttnParams) == 112, "MlaCsa2AttnParams must be 112 bytes");

// Quantizes `[M, kv_lora_rank]` latent rows to FP4 (`E2M1` + per-16 `E4M3`
// scale) and writes `[M, qk_rope_dim]` RoPE keys (`f16` or `E4M3`) into a
// 16 KiB-aligned `MlaFp4KvArena` layer slab, optionally mirroring exact `f16`
// rows into a circular SWA replay ring (`swa_ring_capacity > 0`).
//
// Grid: (num_tokens, 1, 1) threadgroups, 256 threads.
kernel void ami_mla_fp4_kv_write(const device half *latent_in [[buffer(0)]],
                                 const device half *rope_in [[buffer(1)]],
                                 device uchar *kv_slab [[buffer(2)]],
                                 device uchar *swa_ring [[buffer(3)]],
                                 constant MlaFp4KvWriteParams &p [[buffer(4)]],
                                 uint tg_x [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]]) {
    const uint tok = tg_x;
    if (tok >= p.num_tokens) {
        return;
    }
    const uint dst_pos = p.start_pos + tok;
    if (dst_pos >= p.max_tokens) {
        return;
    }

    const uint num_groups = p.kv_lora_rank >> 4; // per-16 channels
    const uint e2m1_row_bytes = p.kv_lora_rank >> 1;
    const ulong in_lat_base = ulong(tok) * p.kv_lora_rank;
    device uchar *e2m1_row = kv_slab + p.e2m1_plane_offset + ulong(dst_pos) * e2m1_row_bytes;
    device uchar *scale_row = kv_slab + p.scale_plane_offset + ulong(dst_pos) * num_groups;

    const bool write_swa = (p.swa_ring_capacity > 0u);
    const uint swa_slot = write_swa ? (dst_pos % p.swa_ring_capacity) : 0u;
    device half *swa_lat_row = write_swa
        ? ((device half *)swa_ring + ulong(swa_slot) * p.kv_lora_rank)
        : nullptr;

    // Each thread quantizes one or more 16-channel groups.
    for (uint g = tid; g < num_groups; g += 256u) {
        const uint ch0 = g << 4;
        float vals[16];
        float amax = 0.0f;
        for (uint i = 0u; i < 16u; i++) {
            const half h_val = latent_in[in_lat_base + ch0 + i];
            if (write_swa) {
                swa_lat_row[ch0 + i] = h_val;
            }
            const float fv = float(h_val);
            vals[i] = fv;
            amax = max(amax, fabs(fv));
        }
        uchar s_byte = 0u;
        float inv_s = 0.0f;
        if (amax > 0.0f) {
            s_byte = mla_f32_to_e4m3(amax * (1.0f / 6.0f));
            if (s_byte == 0u) {
                s_byte = 1u; // min positive E4M3 subnormal (1/512)
            }
            inv_s = 1.0f / mla_e4m3_to_f32(s_byte);
        }
        scale_row[g] = s_byte;

        const uint byte0 = g << 3;
        for (uint b = 0u; b < 8u; b++) {
            const uchar q0 = (s_byte != 0u) ? mla_f32_to_e2m1(vals[b * 2u] * inv_s) : 0u;
            const uchar q1 = (s_byte != 0u) ? mla_f32_to_e2m1(vals[b * 2u + 1u] * inv_s) : 0u;
            e2m1_row[byte0 + b] = uchar((q1 << 4) | (q0 & 0xFu));
        }
    }

    // Write RoPE key tail (`f16` or `E4M3`), plus optional SWA `f16` mirror.
    if (p.qk_rope_dim > 0u) {
        const ulong in_rope_base = ulong(tok) * p.qk_rope_dim;
        const ulong rope_row_bytes = (p.rope_tail_format == 1u)
            ? ulong(p.qk_rope_dim)
            : (ulong(p.qk_rope_dim) * 2ul);
        device uchar *rope_row = kv_slab + p.rope_plane_offset + ulong(dst_pos) * rope_row_bytes;
        device half *swa_rope_row = write_swa
            ? ((device half *)(swa_ring + p.swa_rope_plane_offset) + ulong(swa_slot) * p.qk_rope_dim)
            : nullptr;

        for (uint d = tid; d < p.qk_rope_dim; d += 256u) {
            const half hv = rope_in[in_rope_base + d];
            if (write_swa) {
                swa_rope_row[d] = hv;
            }
            if (p.rope_tail_format == 1u) {
                rope_row[d] = mla_f32_to_e4m3(float(hv));
            } else {
                ((device half *)rope_row)[d] = hv;
            }
        }
    }
}

// Stage-1 hierarchical summary updater: dequantizes FP4 (`E2M1` + `E4M3`)
// latent KV blocks into mean-pooled fine block summaries (`[num_fine, index_dim]`)
// AND coarse super-block summaries (`[num_coarse, index_dim]`) in one pass.
//
// Grid: (num_coarse_blocks, 1, 1) threadgroups, 256 threads.
kernel void ami_csa2_update_summaries(const device uchar *kv_slab [[buffer(0)]],
                                      device uchar *summary_slab [[buffer(1)]],
                                      constant Csa2SummaryParams &p [[buffer(2)]],
                                      uint tg_x [[threadgroup_position_in_grid]],
                                      uint tid [[thread_index_in_threadgroup]]) {
    if (tg_x >= p.num_coarse_blocks || p.kv_len == 0u) {
        return;
    }
    const uint sb = p.start_coarse_block + tg_x;
    const uint g_blocks = max(p.coarse_group_blocks, 1u);
    const uint first_fine = sb * g_blocks;
    const uint total_fine = (p.kv_len + p.block_tokens - 1u) / p.block_tokens;
    if (first_fine >= total_fine) {
        return;
    }
    const uint end_fine = min(first_fine + g_blocks, total_fine);
    const uint valid_sub = end_fine - first_fine;

    const uint e2m1_row_bytes = p.kv_lora_rank >> 1;
    const uint num_groups = p.kv_lora_rank >> 4;
    const device uchar *e2m1_base = kv_slab + p.e2m1_plane_offset;
    const device uchar *scale_base = kv_slab + p.scale_plane_offset;
    device half *fine_keys = (device half *)(summary_slab + p.fine_plane_offset);
    device half *coarse_keys = (device half *)(summary_slab + p.coarse_plane_offset);

    for (uint d = tid; d < p.index_dim; d += 256u) {
        const uint byte_idx = d >> 1;
        const bool hi_nib = ((d & 1u) != 0u);
        const uint grp_idx = d >> 4;
        float coarse_acc = 0.0f;

        for (uint fb = first_fine; fb < end_fine; fb++) {
            const uint t0 = fb * p.block_tokens;
            const uint t1 = min(t0 + p.block_tokens, p.kv_len);
            float fine_acc = 0.0f;
            for (uint t = t0; t < t1; t++) {
                const uchar packed = e2m1_base[ulong(t) * e2m1_row_bytes + byte_idx];
                const uint nib = hi_nib ? ((uint(packed) >> 4) & 0xFu) : (uint(packed) & 0xFu);
                const float sc = mla_e4m3_to_f32(scale_base[ulong(t) * num_groups + grp_idx]);
                fine_acc += mla_e2m1_to_f32(nib) * sc;
            }
            const float fine_mean = fine_acc / float(t1 - t0);
            fine_keys[ulong(fb) * p.index_dim + d] = half(fine_mean);
            coarse_acc += fine_mean;
        }
        coarse_keys[ulong(sb) * p.index_dim + d] = half(coarse_acc / float(valid_sub));
    }
}

// Stage-2 hierarchical CSA2 sparse block selector (`Full` and `Reindex` modes;
// skipped in `Reuse` mode and on `LoopSchedule` `r >= 1` passes).
// Strictly skips upper-triangular blocks (`kv_pos > q_pos`) in causal mode,
// while keeping the full encoder prefix `[0, encoder_prefix_len)` visible in
// CED decoder cross/hybrid mode.
//
// Grid: (num_nodes, 1, 1) threadgroups, 256 threads (8 simdgroups).
kernel void ami_csa2_index_blocks(const device half *q_nope [[buffer(0)]],
                                  const device uchar *fine_keys_raw [[buffer(1)]],
                                  const device uchar *coarse_keys_raw [[buffer(2)]],
                                  const device uint *batch_kv_lens [[buffer(3)]],
                                  device uint *selected_blocks [[buffer(4)]],
                                  device uint *selected_counts [[buffer(5)]],
                                  constant Csa2IndexParams &p [[buffer(6)]],
                                  uint tg_x [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]],
                                  uint sg_id [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
    const uint node = tg_x;
    if (node >= p.num_nodes) {
        return;
    }

    const uint k_cap = min(p.top_k_blocks, 64u);
    const ulong sel_base = ulong(node) * p.top_k_blocks;
    const uint raw_kv_len = (p.has_batch_slots != 0u) ? batch_kv_lens[node] : p.kv_len;
    const uint eff_kv_len = (p.causal_q_start_pos != 0xFFFFFFFFu)
        ? min(raw_kv_len, max(p.encoder_prefix_len, p.causal_q_start_pos + node + 1u))
        : raw_kv_len;

    if (eff_kv_len == 0u || k_cap == 0u) {
        for (uint i = tid; i < p.top_k_blocks; i += 256u) {
            selected_blocks[sel_base + i] = 0xFFFFFFFFu;
        }
        if (tid == 0u) {
            selected_counts[node] = 0u;
        }
        return;
    }

    const uint swa_lo = (p.swa_window > 0u && eff_kv_len > p.swa_window)
        ? (eff_kv_len - p.swa_window)
        : 0u;
    const uint num_valid = (eff_kv_len + p.block_tokens - 1u) / p.block_tokens;

    // When SWA covers the entire visible prefix, no extra sparse blocks are needed.
    if (p.swa_window > 0u && swa_lo == 0u) {
        for (uint i = tid; i < p.top_k_blocks; i += 256u) {
            selected_blocks[sel_base + i] = 0xFFFFFFFFu;
        }
        if (tid == 0u) {
            selected_counts[node] = 0u;
        }
        return;
    }

    const uint num_cand = (p.swa_window > 0u)
        ? ((swa_lo + p.block_tokens - 1u) / p.block_tokens)
        : num_valid;

    // Fast path: all candidate blocks fit inside top_k_blocks.
    if (num_cand <= k_cap) {
        for (uint i = tid; i < p.top_k_blocks; i += 256u) {
            selected_blocks[sel_base + i] = (i < num_cand) ? i : 0xFFFFFFFFu;
        }
        if (tid == 0u) {
            selected_counts[node] = num_cand;
        }
        return;
    }

    const uint local_cnt = (p.swa_window > 0u)
        ? 0u
        : min(max(p.num_local_blocks, 1u), k_cap);
    const uint search_end = num_cand - local_cnt;
    const uint k_sel = k_cap - local_cnt;

    const device half *fine_keys = (const device half *)(fine_keys_raw +
        ((p.has_batch_slots != 0u) ? (ulong(node) * p.fine_batch_stride_bytes) : 0ul));
    const device half *coarse_keys = (const device half *)(coarse_keys_raw +
        ((p.has_batch_slots != 0u) ? (ulong(node) * p.coarse_batch_stride_bytes) : 0ul));
    const ulong q_node_base = ulong(node) * p.num_q_heads * p.kv_lora_rank;
    const uint vec_d = p.index_dim >> 2;

    threadgroup float tg_scores[8][64];
    threadgroup uint tg_blocks[8][64];
    threadgroup uint tg_coarse[64];
    threadgroup uint tg_coarse_cnt;
    threadgroup uint tg_win_cnt;

    const uint g_blocks = max(p.coarse_group_blocks, 1u);
    const uint num_coarse = (search_end + g_blocks - 1u) / g_blocks;
    const uint k_coarse = min(p.top_k_coarse, 64u);
    const bool use_coarse = (g_blocks > 1u && k_coarse > 0u && num_coarse > k_coarse);

    if (tid == 0u) {
        tg_coarse_cnt = 0u;
        tg_win_cnt = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage 2a: Coarse super-block shortlist (when context exceeds coarse budget).
    if (use_coarse && k_sel > 0u) {
        if (lane == 0u) {
            for (uint i = 0u; i < k_coarse; i++) {
                tg_scores[sg_id][i] = -INFINITY;
                tg_blocks[sg_id][i] = 0xFFFFFFFFu;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        uint worst_slot = 0u;
        float worst_score = -INFINITY;
        uint worst_block = 0xFFFFFFFFu;

        for (uint sb = sg_id; sb < num_coarse; sb += 8u) {
            const device half4 *ck_vec = (const device half4 *)(coarse_keys + ulong(sb) * p.index_dim);
            float best_h = -INFINITY;
            for (uint h = 0u; h < p.num_q_heads; h++) {
                const device half4 *q_vec = (const device half4 *)(q_nope + q_node_base + ulong(h) * p.kv_lora_rank);
                float dot_acc = 0.0f;
                for (uint c = lane; c < vec_d; c += 32u) {
                    dot_acc += dot(float4(q_vec[c]), float4(ck_vec[c]));
                }
                best_h = max(best_h, simd_sum(dot_acc) * p.scale);
            }
            if (lane == 0u) {
                if (best_h > worst_score || (best_h == worst_score && sb < worst_block)) {
                    tg_scores[sg_id][worst_slot] = best_h;
                    tg_blocks[sg_id][worst_slot] = sb;
                    uint new_w = 0u;
                    float min_s = tg_scores[sg_id][0];
                    uint max_b = tg_blocks[sg_id][0];
                    for (uint i = 1u; i < k_coarse; i++) {
                        const float si = tg_scores[sg_id][i];
                        const uint bi = tg_blocks[sg_id][i];
                        if (si < min_s || (si == min_s && bi > max_b)) {
                            min_s = si;
                            max_b = bi;
                            new_w = i;
                        }
                    }
                    worst_slot = new_w;
                    worst_score = min_s;
                    worst_block = max_b;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg_id == 0u) {
            const uint total_cands = 8u * k_coarse;
            uint out_c = 0u;
            for (uint k = 0u; k < k_coarse; k++) {
                float best_s = -INFINITY;
                uint best_b = 0xFFFFFFFFu;
                uint best_c = 0xFFFFFFFFu;
                for (uint c = lane; c < total_cands; c += 32u) {
                    const uint sg = c / k_coarse;
                    const uint slot = c - sg * k_coarse;
                    const uint cb = tg_blocks[sg][slot];
                    if (cb != 0xFFFFFFFFu) {
                        const float cs = tg_scores[sg][slot];
                        if (cs > best_s || (cs == best_s && cb < best_b)) {
                            best_s = cs;
                            best_b = cb;
                            best_c = c;
                        }
                    }
                }
                for (ushort off = 16u; off > 0u; off >>= 1u) {
                    const float os = simd_shuffle_down(best_s, off);
                    const uint ob = simd_shuffle_down(best_b, off);
                    const uint oc = simd_shuffle_down(best_c, off);
                    if (os > best_s || (os == best_s && ob < best_b)) {
                        best_s = os;
                        best_b = ob;
                        best_c = oc;
                    }
                }
                const uint win_b = simd_broadcast_first(best_b);
                const uint win_c = simd_broadcast_first(best_c);
                if (lane == 0u && win_b != 0xFFFFFFFFu) {
                    tg_coarse[out_c] = win_b;
                    out_c += 1u;
                    if (win_c < total_cands) {
                        const uint w_sg = win_c / k_coarse;
                        const uint w_slot = win_c - w_sg * k_coarse;
                        tg_scores[w_sg][w_slot] = -INFINITY;
                        tg_blocks[w_sg][w_slot] = 0xFFFFFFFFu;
                    }
                }
            }
            if (lane == 0u) {
                tg_coarse_cnt = out_c;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Stage 2b: Fine block top-k selection (either within coarse shortlist or flat).
    if (k_sel > 0u) {
        if (lane == 0u) {
            for (uint i = 0u; i < k_sel; i++) {
                tg_scores[sg_id][i] = -INFINITY;
                tg_blocks[sg_id][i] = 0xFFFFFFFFu;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        uint worst_slot = 0u;
        float worst_score = -INFINITY;
        uint worst_block = 0xFFFFFFFFu;
        const uint fine_iters = use_coarse ? (tg_coarse_cnt * g_blocks) : search_end;

        for (uint idx = sg_id; idx < fine_iters; idx += 8u) {
            uint b;
            if (use_coarse) {
                const uint ci = idx / g_blocks;
                const uint sub = idx - ci * g_blocks;
                b = tg_coarse[ci] * g_blocks + sub;
                if (b >= search_end) {
                    continue;
                }
            } else {
                b = idx;
            }

            const device half4 *fk_vec = (const device half4 *)(fine_keys + ulong(b) * p.index_dim);
            float best_h = -INFINITY;
            for (uint h = 0u; h < p.num_q_heads; h++) {
                const device half4 *q_vec = (const device half4 *)(q_nope + q_node_base + ulong(h) * p.kv_lora_rank);
                float dot_acc = 0.0f;
                for (uint c = lane; c < vec_d; c += 32u) {
                    dot_acc += dot(float4(q_vec[c]), float4(fk_vec[c]));
                }
                best_h = max(best_h, simd_sum(dot_acc) * p.scale);
            }
            if (lane == 0u) {
                if (best_h > worst_score || (best_h == worst_score && b < worst_block)) {
                    tg_scores[sg_id][worst_slot] = best_h;
                    tg_blocks[sg_id][worst_slot] = b;
                    uint new_w = 0u;
                    float min_s = tg_scores[sg_id][0];
                    uint max_b = tg_blocks[sg_id][0];
                    for (uint i = 1u; i < k_sel; i++) {
                        const float si = tg_scores[sg_id][i];
                        const uint bi = tg_blocks[sg_id][i];
                        if (si < min_s || (si == min_s && bi > max_b)) {
                            min_s = si;
                            max_b = bi;
                            new_w = i;
                        }
                    }
                    worst_slot = new_w;
                    worst_score = min_s;
                    worst_block = max_b;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg_id == 0u) {
            const uint total_cands = 8u * k_sel;
            uint out_w = 0u;
            for (uint k = 0u; k < k_sel; k++) {
                float best_s = -INFINITY;
                uint best_b = 0xFFFFFFFFu;
                uint best_c = 0xFFFFFFFFu;
                for (uint c = lane; c < total_cands; c += 32u) {
                    const uint sg = c / k_sel;
                    const uint slot = c - sg * k_sel;
                    const uint cb = tg_blocks[sg][slot];
                    if (cb != 0xFFFFFFFFu) {
                        const float cs = tg_scores[sg][slot];
                        if (cs > best_s || (cs == best_s && cb < best_b)) {
                            best_s = cs;
                            best_b = cb;
                            best_c = c;
                        }
                    }
                }
                for (ushort off = 16u; off > 0u; off >>= 1u) {
                    const float os = simd_shuffle_down(best_s, off);
                    const uint ob = simd_shuffle_down(best_b, off);
                    const uint oc = simd_shuffle_down(best_c, off);
                    if (os > best_s || (os == best_s && ob < best_b)) {
                        best_s = os;
                        best_b = ob;
                        best_c = oc;
                    }
                }
                const uint win_b = simd_broadcast_first(best_b);
                const uint win_c = simd_broadcast_first(best_c);
                if (lane == 0u && win_b != 0xFFFFFFFFu) {
                    selected_blocks[sel_base + out_w] = win_b;
                    out_w += 1u;
                    if (win_c < total_cands) {
                        const uint w_sg = win_c / k_sel;
                        const uint w_slot = win_c - w_sg * k_sel;
                        tg_scores[w_sg][w_slot] = -INFINITY;
                        tg_blocks[w_sg][w_slot] = 0xFFFFFFFFu;
                    }
                }
            }
            if (lane == 0u) {
                tg_win_cnt = out_w;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint base_cnt = tg_win_cnt;
    if (tid < local_cnt) {
        selected_blocks[sel_base + base_cnt + tid] = search_end + tid;
    }
    const uint final_cnt = base_cnt + local_cnt;
    for (uint i = final_cnt + tid; i < p.top_k_blocks; i += 256u) {
        selected_blocks[sel_base + i] = 0xFFFFFFFFu;
    }
    if (tid == 0u) {
        selected_counts[node] = final_cnt;
    }
}

// Online-softmax absorbed MLA + CSA2 attention over selected FP4 (`E2M1` +
// `E4M3`) latent KV blocks + deduplicated SWA bounded window replay + optional
// 64-bit ancestor-masked speculative tree tokens.
//
// Supports:
//   - wide causal prefill (`M = 1..1024`),
//   - CED encoder->decoder cross/hybrid prefix (`encoder_prefix_len > 0`),
//   - batched decode (`M = 1..16`, `has_batch_slots = 1`),
//   - tree verification (`M = 1..64`, `has_tree = 1`, `u64` ancestor masks),
//   - `kv_lora_rank` multiples of 32 up to `2048` (`1536` for `deepseek41`).
//
// Grid: (num_q_heads, num_nodes, 1) threadgroups, 256 threads (8 simdgroups).
kernel void ami_mla_csa2_attend(const device half *q_nope [[buffer(0)]],
                                const device half *q_pe [[buffer(1)]],
                                const device uchar *kv_slab_base [[buffer(2)]],
                                const device uchar *swa_ring_base [[buffer(3)]],
                                const device uint *selected_blocks [[buffer(4)]],
                                const device uint *selected_counts [[buffer(5)]],
                                const device half *kv_tree_latent [[buffer(6)]],
                                const device half *kv_tree_rope [[buffer(7)]],
                                const device ulong *ancestor_masks [[buffer(8)]],
                                const device uint *batch_kv_lens [[buffer(9)]],
                                device half *out [[buffer(10)]],
                                constant MlaCsa2AttnParams &p [[buffer(11)]],
                                uint2 tg_pos [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]],
                                uint sg_id [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    const uint q_head = tg_pos.x;
    const uint node = tg_pos.y;
    if (q_head >= p.num_q_heads || node >= p.num_nodes) {
        return;
    }

    const uint num_bytes = p.kv_lora_rank >> 1; // <= 1024 (at most 4 pairs/thread)
    const uint num_groups = p.kv_lora_rank >> 4;
    const uint num_iters = (num_bytes + 255u) >> 8;

    const ulong q_nope_off = (ulong(node) * p.num_q_heads + q_head) * p.kv_lora_rank;
    const ulong q_pe_off = (ulong(node) * p.num_q_heads + q_head) * p.qk_rope_dim;

    float2 q_nope_reg[4] = {float2(0.0f), float2(0.0f), float2(0.0f), float2(0.0f)};
    float2 acc_reg[4] = {float2(0.0f), float2(0.0f), float2(0.0f), float2(0.0f)};
    for (uint u = 0u; u < num_iters; u++) {
        const uint b_idx = tid + u * 256u;
        if (b_idx < num_bytes) {
            const ulong ch = q_nope_off + ulong(b_idx) * 2ul;
            q_nope_reg[u] = float2(float(q_nope[ch]), float(q_nope[ch + 1ul]));
        }
    }
    const float q_pe_reg = (tid < p.qk_rope_dim) ? float(q_pe[q_pe_off + tid]) : 0.0f;

    const device uchar *kv_slab = kv_slab_base +
        ((p.has_batch_slots != 0u) ? (ulong(node) * p.kv_batch_stride_bytes) : 0ul);
    const device uchar *e2m1_base = kv_slab + p.e2m1_plane_offset;
    const device uchar *scale_base = kv_slab + p.scale_plane_offset;
    const device uchar *rope_base = kv_slab + p.rope_plane_offset;
    const ulong rope_row_bytes = (p.rope_tail_format == 1u)
        ? ulong(p.qk_rope_dim)
        : (ulong(p.qk_rope_dim) * 2ul);

    const device uchar *swa_ring = swa_ring_base +
        ((p.has_batch_slots != 0u) ? (ulong(node) * p.swa_batch_stride_bytes) : 0ul);
    const device half *swa_lat_base = (const device half *)swa_ring;
    const device half *swa_rope_base = (const device half *)(swa_ring + p.swa_rope_plane_offset);

    const uint raw_kv_len = (p.has_batch_slots != 0u) ? batch_kv_lens[node] : p.kv_len;
    const uint eff_kv_len = (p.causal_q_start_pos != 0xFFFFFFFFu)
        ? min(raw_kv_len, max(p.encoder_prefix_len, p.causal_q_start_pos + node + 1u))
        : raw_kv_len;
    const uint swa_hi = eff_kv_len;
    const uint swa_lo = (p.swa_window > 0u)
        ? ((eff_kv_len > p.swa_window) ? (eff_kv_len - p.swa_window) : 0u)
        : eff_kv_len;
    const uint tree_anc = (p.has_tree != 0u) ? uint(popcount(ancestor_masks[node])) : 0u;
    const uint depth_above_root = (tree_anc > 1u) ? (tree_anc - 1u) : 0u;
    const uint eff_ring_pos = raw_kv_len + depth_above_root;
    const uint ring_valid_lo = (p.swa_ring_capacity > 0u && eff_ring_pos > p.swa_ring_capacity)
        ? (eff_ring_pos - p.swa_ring_capacity)
        : 0u;

    const float scale_log2e = p.scale * 1.4426950408889634f;
    float m_val = -INFINITY;
    float l_val = 0.0f;
    threadgroup float tg_reduce[8];

    // Pass 1: Selected CSA2 sparse FP4 blocks (deduplicated against [swa_lo, swa_hi)).
    const uint sel_count = min(selected_counts[node], p.top_k_blocks);
    const ulong sel_base = ulong(node) * p.top_k_blocks;
    for (uint sb = 0u; sb < sel_count; sb++) {
        const uint block_id = selected_blocks[sel_base + sb];
        if (block_id == 0xFFFFFFFFu) {
            continue;
        }
        const uint t_start = block_id * p.block_tokens;
        if (t_start >= eff_kv_len) {
            continue;
        }
        const uint t_end = min(t_start + p.block_tokens, eff_kv_len);
        for (uint t = t_start; t < t_end; t++) {
            if (t >= swa_lo && t < swa_hi) {
                continue;
            }
            const device uchar *e2m1_row = e2m1_base + ulong(t) * num_bytes;
            const device uchar *scale_row = scale_base + ulong(t) * num_groups;
            float2 kv_reg[4];
            float dot_partial = 0.0f;
            for (uint u = 0u; u < num_iters; u++) {
                const uint b_idx = tid + u * 256u;
                if (b_idx < num_bytes) {
                    const uchar packed = e2m1_row[b_idx];
                    const float sc = mla_e4m3_to_f32(scale_row[b_idx >> 3]);
                    const float2 kv = float2(
                        mla_e2m1_to_f32(uint(packed) & 0xFu),
                        mla_e2m1_to_f32((uint(packed) >> 4) & 0xFu)) * sc;
                    kv_reg[u] = kv;
                    dot_partial += dot(q_nope_reg[u], kv);
                } else {
                    kv_reg[u] = float2(0.0f);
                }
            }
            if (tid < p.qk_rope_dim) {
                const device uchar *rope_row = rope_base + ulong(t) * rope_row_bytes;
                const float k_pe_val = (p.rope_tail_format == 1u)
                    ? mla_e4m3_to_f32(rope_row[tid])
                    : float(((const device half *)rope_row)[tid]);
                dot_partial += q_pe_reg * k_pe_val;
            }
            const float sg_sum = simd_sum(dot_partial);
            if (lane == 0u) {
                tg_reduce[sg_id] = sg_sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float total_dot = ((tg_reduce[0] + tg_reduce[1]) + (tg_reduce[2] + tg_reduce[3])) +
                ((tg_reduce[4] + tg_reduce[5]) + (tg_reduce[6] + tg_reduce[7]));
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const float score = total_dot * scale_log2e;
            const float new_m = max(m_val, score);
            const float corr = (l_val > 0.0f) ? exp2(m_val - new_m) : 0.0f;
            const float p_w = exp2(score - new_m);
            l_val = fma(l_val, corr, p_w);
            for (uint u = 0u; u < num_iters; u++) {
                acc_reg[u] = fma(acc_reg[u], float2(corr), kv_reg[u] * p_w);
            }
            m_val = new_m;
        }
    }

    // Pass 2: SWA bounded window replay `[swa_lo, swa_hi)`. Reads exact `f16`
    // from circular SWA ring when resident (`t >= ring_valid_lo`), else falls
    // back seamlessly to the non-destructive FP4 `MlaFp4KvArena`.
    for (uint t = swa_lo; t < swa_hi; t++) {
        const bool use_swa_ring = (p.swa_ring_capacity > 0u && t >= ring_valid_lo);
        float2 kv_reg[4];
        float dot_partial = 0.0f;
        if (use_swa_ring) {
            const uint slot = t % p.swa_ring_capacity;
            const device half *s_lat = swa_lat_base + ulong(slot) * p.kv_lora_rank;
            for (uint u = 0u; u < num_iters; u++) {
                const uint b_idx = tid + u * 256u;
                if (b_idx < num_bytes) {
                    const uint ch = b_idx * 2u;
                    const float2 kv = float2(float(s_lat[ch]), float(s_lat[ch + 1u]));
                    kv_reg[u] = kv;
                    dot_partial += dot(q_nope_reg[u], kv);
                } else {
                    kv_reg[u] = float2(0.0f);
                }
            }
            if (tid < p.qk_rope_dim) {
                const device half *s_rope = swa_rope_base + ulong(slot) * p.qk_rope_dim;
                dot_partial += q_pe_reg * float(s_rope[tid]);
            }
        } else {
            const device uchar *e2m1_row = e2m1_base + ulong(t) * num_bytes;
            const device uchar *scale_row = scale_base + ulong(t) * num_groups;
            for (uint u = 0u; u < num_iters; u++) {
                const uint b_idx = tid + u * 256u;
                if (b_idx < num_bytes) {
                    const uchar packed = e2m1_row[b_idx];
                    const float sc = mla_e4m3_to_f32(scale_row[b_idx >> 3]);
                    const float2 kv = float2(
                        mla_e2m1_to_f32(uint(packed) & 0xFu),
                        mla_e2m1_to_f32((uint(packed) >> 4) & 0xFu)) * sc;
                    kv_reg[u] = kv;
                    dot_partial += dot(q_nope_reg[u], kv);
                } else {
                    kv_reg[u] = float2(0.0f);
                }
            }
            if (tid < p.qk_rope_dim) {
                const device uchar *rope_row = rope_base + ulong(t) * rope_row_bytes;
                const float k_pe_val = (p.rope_tail_format == 1u)
                    ? mla_e4m3_to_f32(rope_row[tid])
                    : float(((const device half *)rope_row)[tid]);
                dot_partial += q_pe_reg * k_pe_val;
            }
        }
        const float sg_sum = simd_sum(dot_partial);
        if (lane == 0u) {
            tg_reduce[sg_id] = sg_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float total_dot = ((tg_reduce[0] + tg_reduce[1]) + (tg_reduce[2] + tg_reduce[3])) +
            ((tg_reduce[4] + tg_reduce[5]) + (tg_reduce[6] + tg_reduce[7]));
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float score = total_dot * scale_log2e;
        const float new_m = max(m_val, score);
        const float corr = (l_val > 0.0f) ? exp2(m_val - new_m) : 0.0f;
        const float p_w = exp2(score - new_m);
        l_val = fma(l_val, corr, p_w);
        for (uint u = 0u; u < num_iters; u++) {
            acc_reg[u] = fma(acc_reg[u], float2(corr), kv_reg[u] * p_w);
        }
        m_val = new_m;
    }

    // Pass 3: Optional speculative tree ancestors (`u64` ancestor bitmask).
    if (p.has_tree != 0u) {
        const ulong mask = ancestor_masks[node];
        for (uint j = 0u; j < p.num_nodes; j++) {
            if (((mask >> j) & 1ul) == 0ul) {
                continue;
            }
            const device half *t_lat = kv_tree_latent + ulong(j) * p.kv_lora_rank;
            float2 kv_reg[4];
            float dot_partial = 0.0f;
            for (uint u = 0u; u < num_iters; u++) {
                const uint b_idx = tid + u * 256u;
                if (b_idx < num_bytes) {
                    const uint ch = b_idx * 2u;
                    const float2 kv = float2(float(t_lat[ch]), float(t_lat[ch + 1u]));
                    kv_reg[u] = kv;
                    dot_partial += dot(q_nope_reg[u], kv);
                } else {
                    kv_reg[u] = float2(0.0f);
                }
            }
            if (tid < p.qk_rope_dim) {
                const device half *t_rope = kv_tree_rope + ulong(j) * p.qk_rope_dim;
                dot_partial += q_pe_reg * float(t_rope[tid]);
            }
            const float sg_sum = simd_sum(dot_partial);
            if (lane == 0u) {
                tg_reduce[sg_id] = sg_sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float total_dot = ((tg_reduce[0] + tg_reduce[1]) + (tg_reduce[2] + tg_reduce[3])) +
                ((tg_reduce[4] + tg_reduce[5]) + (tg_reduce[6] + tg_reduce[7]));
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const float score = total_dot * scale_log2e;
            const float new_m = max(m_val, score);
            const float corr = (l_val > 0.0f) ? exp2(m_val - new_m) : 0.0f;
            const float p_w = exp2(score - new_m);
            l_val = fma(l_val, corr, p_w);
            for (uint u = 0u; u < num_iters; u++) {
                acc_reg[u] = fma(acc_reg[u], float2(corr), kv_reg[u] * p_w);
            }
            m_val = new_m;
        }
    }

    const float inv_l = (l_val > 0.0f) ? (1.0f / l_val) : 0.0f;
    for (uint u = 0u; u < num_iters; u++) {
        const uint b_idx = tid + u * 256u;
        if (b_idx < num_bytes) {
            const ulong ch = q_nope_off + ulong(b_idx) * 2ul;
            out[ch] = half(acc_reg[u].x * inv_l);
            out[ch + 1ul] = half(acc_reg[u].y * inv_l);
        }
    }
}

// ===========================================================================
// Section F: M7 (G10) — Single-Pass Mega-mHC (`mhc_single_pass`)
// ===========================================================================

struct MhcParams {
    uint m;
    uint num_streams; // 2..8 (default 4)
    uint hidden_size;
    uint flags;          // bit0=has_norm_w, bit1=norm_w_bf16, bit2=apply_h_res, bit3=activate_coeffs, bit4=shared_coeffs
    uint sinkhorn_iters; // 0 = direct H_res, >0 = exp + Sinkhorn-Knopp Birkhoff projection
    float rms_eps;
    float post_scale;
    uint _pad0;
};

static_assert(sizeof(MhcParams) == 32, "MhcParams must be 32 bytes");

static inline void mhc_load_h_res(const device float *h_res_ptr,
                                  uint n,
                                  uint sinkhorn_iters,
                                  threadgroup float *tg_hres) {
    float max_v = -INFINITY;
    for (uint o = 0u; o < n; o++) {
        for (uint i = 0u; i < n; i++) {
            const float v = h_res_ptr[o * n + i];
            tg_hres[o * 8u + i] = v;
            max_v = max(max_v, v);
        }
    }
    if (sinkhorn_iters == 0u) {
        return;
    }
    for (uint o = 0u; o < n; o++) {
        for (uint i = 0u; i < n; i++) {
            tg_hres[o * 8u + i] = exp(tg_hres[o * 8u + i] - max_v);
        }
    }
    for (uint it = 0u; it < sinkhorn_iters; it++) {
        for (uint o = 0u; o < n; o++) {
            float rsum = 0.0f;
            for (uint i = 0u; i < n; i++) {
                rsum += tg_hres[o * 8u + i];
            }
            const float inv_r = 1.0f / max(rsum, 1e-12f);
            for (uint i = 0u; i < n; i++) {
                tg_hres[o * 8u + i] *= inv_r;
            }
        }
        for (uint i = 0u; i < n; i++) {
            float csum = 0.0f;
            for (uint o = 0u; o < n; o++) {
                csum += tg_hres[o * 8u + i];
            }
            const float inv_c = 1.0f / max(csum, 1e-12f);
            for (uint o = 0u; o < n; o++) {
                tg_hres[o * 8u + i] *= inv_c;
            }
        }
    }
}

// Single-Pass Mega-mHC pre-sublayer kernel:
// Fuses multi-stream mixing across `num_streams` (`2..8`) hyper-streams
// (`streams [M, num_streams, hidden_size]`), optional RMSNorm (`norm_w`), and
// optional in-place Birkhoff-projected `H_res` stream mixing with 0 extra
// global intermediate buffers.
//
// Grid: (M, 1, 1) threadgroups, 256 threads (8 simdgroups).
kernel void ami_mhc_single_pass(device uchar *streams [[buffer(0)]],
                                const device float *h_pre [[buffer(1)]],
                                const device float *h_res [[buffer(2)]],
                                const device uchar *norm_w [[buffer(3)]],
                                device uchar *branch_out [[buffer(4)]],
                                constant MhcParams &p [[buffer(5)]],
                                uint tg_x [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]],
                                uint sg_id [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    const uint tok = tg_x;
    if (tok >= p.m) {
        return;
    }

    const uint n = p.num_streams;
    const bool has_norm_w = ((p.flags & 1u) != 0u);
    const bool norm_w_bf16 = ((p.flags & 2u) != 0u);
    const bool apply_h_res = ((p.flags & 4u) != 0u);
    const bool activate_coeffs = ((p.flags & 8u) != 0u);
    const bool shared_coeffs = ((p.flags & 16u) != 0u);
    const uint coeff_tok = shared_coeffs ? 0u : tok;

    threadgroup float tg_hpre[8];
    threadgroup float tg_hres[64];
    threadgroup float tg_reduce[8];

    if (tid == 0u) {
        const device float *hp_row = h_pre + ulong(coeff_tok) * n;
        for (uint i = 0u; i < n; i++) {
            const float raw = hp_row[i];
            tg_hpre[i] = activate_coeffs ? hybrid_sigmoid(raw) : raw;
        }
        if (apply_h_res) {
            const device float *hr_row = h_res + ulong(coeff_tok) * n * n;
            mhc_load_h_res(hr_row, n, p.sinkhorn_iters, tg_hres);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const ulong tok_stream_base = ulong(tok) * n * p.hidden_size;
    const ulong tok_out_base = ulong(tok) * p.hidden_size;

    float inv_rms = 1.0f;
    if (has_norm_w) {
        float sq_partial = 0.0f;
        for (uint d = tid; d < p.hidden_size; d += 256u) {
            float mix = 0.0f;
            for (uint i = 0u; i < n; i++) {
                const float s_val = hybrid_load_f16_or_bf16(
                    streams, tok_stream_base + ulong(i) * p.hidden_size + d, GR_STREAM_BF16);
                mix = fma(tg_hpre[i], s_val, mix);
            }
            sq_partial = fma(mix, mix, sq_partial);
        }
        const float sg_sq = simd_sum(sq_partial);
        if (lane == 0u) {
            tg_reduce[sg_id] = sg_sq;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float total_sq = ((tg_reduce[0] + tg_reduce[1]) + (tg_reduce[2] + tg_reduce[3])) +
            ((tg_reduce[4] + tg_reduce[5]) + (tg_reduce[6] + tg_reduce[7]));
        inv_rms = rsqrt(total_sq / float(p.hidden_size) + p.rms_eps);
    }

    for (uint d = tid; d < p.hidden_size; d += 256u) {
        float s_reg[8];
        float mix = 0.0f;
        for (uint i = 0u; i < n; i++) {
            const float s_val = hybrid_load_f16_or_bf16(
                streams, tok_stream_base + ulong(i) * p.hidden_size + d, GR_STREAM_BF16);
            s_reg[i] = s_val;
            mix = fma(tg_hpre[i], s_val, mix);
        }
        if (has_norm_w) {
            const float nw = hybrid_load_norm_w(norm_w, d, norm_w_bf16);
            mix = (mix * inv_rms) * nw;
        }
        hybrid_store_f16_or_bf16(branch_out, tok_out_base + d, mix, GR_ACT_BF16);

        if (apply_h_res) {
            for (uint o = 0u; o < n; o++) {
                float acc = 0.0f;
                for (uint i = 0u; i < n; i++) {
                    acc = fma(tg_hres[o * 8u + i], s_reg[i], acc);
                }
                hybrid_store_f16_or_bf16(
                    streams, tok_stream_base + ulong(o) * p.hidden_size + d, acc, GR_STREAM_BF16);
            }
        }
    }
}

// Single-Pass Mega-mHC post-sublayer fused residual update:
// Updates `streams [M, num_streams, hidden_size]` in-place:
//   `streams[m, o, d] = (apply_h_res ? sum_i H_res[m, o, i] * streams[m, i, d] : streams[m, o, d])`
//                     `+ H_post[m, o] * sublayer_out[m, d]`
// with optional in-kernel Sinkhorn-Knopp doubly-stochastic projection and 0
// extra global intermediate buffers.
//
// Grid: (M, 1, 1) threadgroups, 256 threads.
kernel void ami_mhc_post_mix(device uchar *streams [[buffer(0)]],
                             const device uchar *sublayer_out [[buffer(1)]],
                             const device float *h_post [[buffer(2)]],
                             const device float *h_res [[buffer(3)]],
                             constant MhcParams &p [[buffer(4)]],
                             uint tg_x [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]]) {
    const uint tok = tg_x;
    if (tok >= p.m) {
        return;
    }

    const uint n = p.num_streams;
    const bool apply_h_res = ((p.flags & 4u) != 0u);
    const bool activate_coeffs = ((p.flags & 8u) != 0u);
    const bool shared_coeffs = ((p.flags & 16u) != 0u);
    const uint coeff_tok = shared_coeffs ? 0u : tok;

    threadgroup float tg_hpost[8];
    threadgroup float tg_hres[64];

    if (tid == 0u) {
        const device float *hp_row = h_post + ulong(coeff_tok) * n;
        for (uint i = 0u; i < n; i++) {
            const float raw = hp_row[i];
            tg_hpost[i] = activate_coeffs ? (p.post_scale * hybrid_sigmoid(raw)) : (p.post_scale * raw);
        }
        if (apply_h_res) {
            const device float *hr_row = h_res + ulong(coeff_tok) * n * n;
            mhc_load_h_res(hr_row, n, p.sinkhorn_iters, tg_hres);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const ulong tok_stream_base = ulong(tok) * n * p.hidden_size;
    const ulong tok_in_base = ulong(tok) * p.hidden_size;

    for (uint d = tid; d < p.hidden_size; d += 256u) {
        const float y_val = hybrid_load_f16_or_bf16(sublayer_out, tok_in_base + d, GR_ACT_BF16);
        float s_reg[8];
        for (uint i = 0u; i < n; i++) {
            s_reg[i] = hybrid_load_f16_or_bf16(
                streams, tok_stream_base + ulong(i) * p.hidden_size + d, GR_STREAM_BF16);
        }
        for (uint o = 0u; o < n; o++) {
            float base_val;
            if (apply_h_res) {
                base_val = 0.0f;
                for (uint i = 0u; i < n; i++) {
                    base_val = fma(tg_hres[o * 8u + i], s_reg[i], base_val);
                }
            } else {
                base_val = s_reg[o];
            }
            const float out_val = fma(tg_hpost[o], y_val, base_val);
            hybrid_store_f16_or_bf16(
                streams, tok_stream_base + ulong(o) * p.hidden_size + d, out_val, GR_STREAM_BF16);
        }
    }
}
