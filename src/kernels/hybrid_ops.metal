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
#include <metal_atomic>
#include <metal_simdgroup_matrix>
using namespace metal;

constant uint GDN_HEAD_K_DIM [[function_constant(0)]];
constant bool HYBRID_NORM_W_BF16 [[function_constant(1)]];
constant uint QSA_HEAD_DIM [[function_constant(2)]];
constant bool GR_STREAM_BF16 [[function_constant(3)]];
constant bool GR_ACT_BF16 [[function_constant(4)]];
constant uint G26_K_DIM [[function_constant(5)]];
constant uint G26_V_DIM [[function_constant(6)]];
constant bool G26_LOGITS_F16 [[function_constant(7)]];

static inline float hybrid_fast_silu(float x) {
    return x / (1.0f + exp(-x));
}

static inline float hybrid_sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

// GDN output gate mode: 1 = SiLU (legacy/Qwen3-Next), 2 = Sigmoid (real-HF Qwen3.8-Flash-Next).
static inline float hybrid_gdn_out_gate(float z, uint mode) {
    return (mode == 2u) ? hybrid_sigmoid(z) : hybrid_fast_silu(z);
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
            val *= hybrid_gdn_out_gate(float(z[v_head_base + dv]), p.apply_out_gate);
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
                            o_val *= hybrid_gdn_out_gate(float(z[v_idx]), p.apply_out_gate);
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
                    o_val *= hybrid_gdn_out_gate(float(z[out_idx]), p.apply_out_gate);
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
            val *= hybrid_gdn_out_gate(float(z[base + dv]), p.apply_out_gate);
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
        const float w_add = (p.apply_qk_norm == 2u) ? 1.0f : 0.0f;
        for (uint d = lane; d < d_dim; d += 32u) {
            vec[d] = (vec[d] * inv_rms) * (w_add + hybrid_load_norm_w(nw, d, HYBRID_NORM_W_BF16));
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
        const float w_add = (p.apply_qk_norm == 2u) ? 1.0f : 0.0f;
        for (uint d = lane; d < d_dim; d += 32u) {
            vec[d] = (vec[d] * inv_rms) * (w_add + hybrid_load_norm_w(nw, d, HYBRID_NORM_W_BF16));
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
            val *= hybrid_gdn_out_gate(float(z[v_head_base + dv]), p.apply_out_gate);
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
        const float w_add = (p.apply_qk_norm == 2u) ? 1.0f : 0.0f;
        for (uint d = lane; d < d_dim; d += 32u) {
            vec[d] = (vec[d] * inv_rms) * (w_add + hybrid_load_norm_w(nw, d, HYBRID_NORM_W_BF16));
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

// ---------------------------------------------------------------------------
// Kernel (`G9` + `G19`, ADR-0023 §2): `ami_qsa_swa_prefill_attn_mma`
//
// Hardware `simdgroup_half8x8` (`Apple9`) tiled FlashAttention-2 prefill for
// QSA full & SWA prefill (`BQ = 32, BK = 64` for `D = 64` in 15,872 B SRAM;
// `BQ = 32, BK = 32` for `D = 128` in 15,360 B SRAM; both `<= 16 KiB` with
// bank-conflict-free padding). Strictly gated on `Q_len >= 32` and
// `KV_len >= 64` by `src/gpu/hybrid.zig`.
// Grid: (num_kv_heads, ceil(num_q * gqa_ratio / 32), 1) threadgroups, 128 threads.
// ---------------------------------------------------------------------------
template <uint D, uint D_PAD, uint BK, bool IS_V>
static inline void qsa_mma_load_kv(threadgroup half4 *KVs4,
                                   const device QsaKvTilePair *kv_tiles,
                                   const device half *kv_chunk,
                                   const device uchar *retired_pool,
                                   const device uint *retired_chunk_ids,
                                   constant QsaSwaPrefillAttnParams &p,
                                   uint kv_head,
                                   uint kv_block_start,
                                   uint kv_block_end,
                                   uint tid) {
    constexpr uint D_VEC4 = D >> 2;
    constexpr uint D_PAD4 = D_PAD >> 2;
    constexpr uint ROW_STEP = 128u / D_VEC4;
    constexpr uint GROUPS_PER_HEAD = D >> 6;

    const uint load_row0 = tid / D_VEC4;
    const uint load_col4 = tid & (D_VEC4 - 1u);

    for (uint row = load_row0; row < BK; row += ROW_STEP) {
        const uint t = kv_block_start + row;
        half4 val = half4(0.0h);
        if (t < kv_block_end) {
            if (t >= p.chunk_start_pos) {
                const ulong c_off = (ulong(t - p.chunk_start_pos) * p.num_kv_heads + kv_head) * D + load_col4 * 4u;
                val = *(const device half4 *)(kv_chunk + c_off);
            } else {
                const uint phys_t = (p.ring_capacity > 0u) ? (t % p.ring_capacity) : t;
                const uint chunk = phys_t / p.chunk_tokens;
                const uint slot = phys_t - chunk * p.chunk_tokens;
                const uint tex_row = kv_head * p.chunk_tokens + slot;
                if (p.ring_capacity == 0u && chunk < p.num_retired_chunks) {
                    const uint phys_slot = retired_chunk_ids[chunk];
                    const device uchar *plane = retired_pool +
                        ulong(phys_slot) * p.retired_chunk_stride_bytes +
                        (IS_V ? p.retired_plane_stride_bytes : 0u);
                    val = hybrid_load_q8_half4(plane, tex_row, GROUPS_PER_HEAD, load_col4);
                } else {
                    const uint active_chunk = (p.ring_capacity == 0u) ? (chunk - p.num_retired_chunks) : chunk;
                    val = IS_V ? kv_tiles[active_chunk].v_tex.read(uint2(load_col4, tex_row))
                               : kv_tiles[active_chunk].k_tex.read(uint2(load_col4, tex_row));
                }
            }
        }
        KVs4[row * D_PAD4 + load_col4] = val;
    }
}

union QsaMmaD64Union {
    half4 KVs4[64u * 18u];  // 9,216 B
    float Ss[32u * 64u];    // 8,192 B
};

static inline void qsa_mma_exec_d128(const device half *q,
                                     const device QsaKvTilePair *kv_tiles,
                                     const device half *k_chunk,
                                     const device half *v_chunk,
                                     device half *out,
                                     const device uchar *retired_pool,
                                     const device uint *retired_chunk_ids,
                                     constant QsaSwaPrefillAttnParams &p,
                                     threadgroup half4 *KVs4,
                                     threadgroup float *Ss,
                                     threadgroup half *Ps,
                                     uint2 tg_pos,
                                     uint tid,
                                     uint sg_id,
                                     uint lane) {
    constexpr uint D = 128u;
    constexpr uint D_PAD = 136u;
    constexpr uint D_PAD4 = 34u;
    constexpr uint BQ = 32u;
    constexpr uint BK = 32u;
    constexpr uint D_VEC4 = 32u;
    constexpr uint ROW_STEP = 4u;
    constexpr uint D_TILES8 = 16u;
    constexpr float LOG2E = 1.4426950408889634f;

    const uint kv_head = tg_pos.x;
    const uint gqa_ratio = p.num_q_heads / p.num_kv_heads;
    const uint total_q = p.num_q * gqa_ratio;
    const uint flat_q_base = tg_pos.y * BQ;
    if (kv_head >= p.num_kv_heads || flat_q_base >= total_q) {
        return;
    }

    const uint q_cnt = min(BQ, total_q - flat_q_base);
    const uint min_q_pos = p.q_start_pos + (flat_q_base / gqa_ratio);
    const uint max_q_pos = p.q_start_pos + ((flat_q_base + q_cnt - 1u) / gqa_ratio);
    const uint kv_end = min(p.kv_len, max_q_pos + 1u);
    const uint kv_begin = (p.window_size > 0u && (min_q_pos + 1u) > p.window_size)
        ? (min_q_pos + 1u - p.window_size)
        : 0u;

    {
        const uint load_row0 = tid / D_VEC4;
        const uint load_col4 = tid & (D_VEC4 - 1u);
        for (uint row = load_row0; row < BQ; row += ROW_STEP) {
            const uint flat_q = flat_q_base + row;
            if (flat_q < total_q) {
                const uint qi = flat_q / gqa_ratio;
                const uint qh = kv_head * gqa_ratio + (flat_q - qi * gqa_ratio);
                const ulong q_off = (ulong(qi) * p.num_q_heads + qh) * D + load_col4 * 4u;
                KVs4[row * D_PAD4 + load_col4] = *(const device half4 *)(q + q_off);
            } else {
                KVs4[row * D_PAD4 + load_col4] = half4(0.0h);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint r0 = sg_id * 8u;
    const threadgroup half *KVs = (const threadgroup half *)KVs4;
    simdgroup_half8x8 q_frag[D_TILES8];
    #pragma clang loop unroll(full)
    for (uint kt = 0u; kt < D_TILES8; kt++) {
        simdgroup_load(q_frag[kt], KVs + r0 * D_PAD + kt * 8u, D_PAD);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_val[8];
    float l_val[8];
    float o_reg[8][4];
    #pragma clang loop unroll(full)
    for (uint r = 0u; r < 8u; r++) {
        m_val[r] = -INFINITY;
        l_val[r] = 0.0f;
        #pragma clang loop unroll(full)
        for (uint c = 0u; c < 4u; c++) {
            o_reg[r][c] = 0.0f;
        }
    }

    const float scale_log2e = p.scale * LOG2E;

    for (uint kv_block_start = kv_begin; kv_block_start < kv_end; kv_block_start += BK) {
        const uint kv_block_end = min(kv_block_start + BK, kv_end);

        qsa_mma_load_kv<D, D_PAD, BK, false>(
            KVs4, kv_tiles, k_chunk, retired_pool, retired_chunk_ids, p,
            kv_head, kv_block_start, kv_block_end, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 s_frag[4];
        #pragma clang loop unroll(full)
        for (uint bj = 0u; bj < 4u; bj++) {
            s_frag[bj] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
        for (uint kt = 0u; kt < D_TILES8; kt++) {
            const uint kk = kt * 8u;
            #pragma clang loop unroll(full)
            for (uint bj = 0u; bj < 4u; bj++) {
                simdgroup_half8x8 k_frag;
                simdgroup_load(k_frag, KVs + (bj * 8u) * D_PAD + kk, D_PAD, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(s_frag[bj], q_frag[kt], k_frag, s_frag[bj]);
            }
        }
        #pragma clang loop unroll(full)
        for (uint bj = 0u; bj < 4u; bj++) {
            simdgroup_store(s_frag[bj], Ss + r0 * BK + bj * 8u, BK);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        float alpha[8];
        const uint kv_pos = kv_block_start + lane;
        const bool in_blk = (kv_pos < kv_block_end);
        #pragma clang loop unroll(full)
        for (uint r = 0u; r < 8u; r++) {
            const uint flat_q = flat_q_base + r0 + r;
            const bool act = (flat_q < total_q);
            const uint qp = p.q_start_pos + (act ? (flat_q / gqa_ratio) : 0u);
            const float raw_s = Ss[(r0 + r) * BK + lane];
            const bool valid = act && in_blk && (kv_pos <= qp) &&
                               (p.window_size == 0u || (qp - kv_pos) < p.window_size);
            const float score = valid ? (raw_s * scale_log2e) : -INFINITY;
            const float row_m = simd_max(score);
            float p_val = 0.0f;
            if (row_m > -INFINITY) {
                const float m_new = max(m_val[r], row_m);
                const float a = (l_val[r] > 0.0f) ? exp2(m_val[r] - m_new) : 0.0f;
                p_val = valid ? exp2(score - m_new) : 0.0f;
                const float row_l = simd_sum(p_val);
                l_val[r] = fma(l_val[r], a, row_l);
                m_val[r] = m_new;
                alpha[r] = a;
            } else {
                alpha[r] = 1.0f;
            }
            Ps[(r0 + r) * 40u + lane] = half(p_val);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        qsa_mma_load_kv<D, D_PAD, BK, true>(
            KVs4, kv_tiles, v_chunk, retired_pool, retired_chunk_ids, p,
            kv_head, kv_block_start, kv_block_end, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (uint c = 0u; c < 4u; c++) {
            const uint d_blk = c * 32u;
            simdgroup_float8x8 pv[4];
            #pragma clang loop unroll(full)
            for (uint bj = 0u; bj < 4u; bj++) {
                pv[bj] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            }
            #pragma clang loop unroll(full)
            for (uint kk = 0u; kk < BK; kk += 8u) {
                simdgroup_half8x8 p_frag;
                simdgroup_load(p_frag, Ps + r0 * 40u + kk, 40u);
                #pragma clang loop unroll(full)
                for (uint bj = 0u; bj < 4u; bj++) {
                    simdgroup_half8x8 v_frag;
                    simdgroup_load(v_frag, KVs + kk * D_PAD + d_blk + bj * 8u, D_PAD);
                    simdgroup_multiply_accumulate(pv[bj], p_frag, v_frag, pv[bj]);
                }
            }
            #pragma clang loop unroll(full)
            for (uint bj = 0u; bj < 4u; bj++) {
                simdgroup_store(pv[bj], Ss + r0 * 32u + bj * 8u, 32u);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            #pragma clang loop unroll(full)
            for (uint r = 0u; r < 8u; r++) {
                o_reg[r][c] = fma(o_reg[r][c], alpha[r], Ss[(r0 + r) * 32u + lane]);
            }
            if (c + 1u < 4u) {
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma clang loop unroll(full)
    for (uint r = 0u; r < 8u; r++) {
        const uint flat_q = flat_q_base + r0 + r;
        if (flat_q >= total_q) {
            continue;
        }
        const uint qi = flat_q / gqa_ratio;
        const uint qh = kv_head * gqa_ratio + (flat_q - qi * gqa_ratio);
        const float inv_l = (l_val[r] > 0.0f) ? (1.0f / l_val[r]) : 0.0f;
        const ulong out_base = (ulong(qi) * p.num_q_heads + qh) * D;
        #pragma clang loop unroll(full)
        for (uint c = 0u; c < 4u; c++) {
            out[out_base + c * 32u + lane] = half(o_reg[r][c] * inv_l);
        }
    }
}

static inline void qsa_mma_exec_d64(const device half *q,
                                    const device QsaKvTilePair *kv_tiles,
                                    const device half *k_chunk,
                                    const device half *v_chunk,
                                    device half *out,
                                    const device uchar *retired_pool,
                                    const device uint *retired_chunk_ids,
                                    constant QsaSwaPrefillAttnParams &p,
                                    threadgroup QsaMmaD64Union &u_smem,
                                    threadgroup half *Ps,
                                    threadgroup float *Os,
                                    uint2 tg_pos,
                                    uint tid,
                                    uint sg_id,
                                    uint lane) {
    constexpr uint D = 64u;
    constexpr uint D_PAD = 72u;
    constexpr uint D_PAD4 = 18u;
    constexpr uint BQ = 32u;
    constexpr uint BK = 64u;
    constexpr uint BK_PAD = 72u;
    constexpr uint D_VEC4 = 16u;
    constexpr uint ROW_STEP = 8u;
    constexpr uint D_TILES8 = 8u;
    constexpr float LOG2E = 1.4426950408889634f;

    const uint kv_head = tg_pos.x;
    const uint gqa_ratio = p.num_q_heads / p.num_kv_heads;
    const uint total_q = p.num_q * gqa_ratio;
    const uint flat_q_base = tg_pos.y * BQ;
    if (kv_head >= p.num_kv_heads || flat_q_base >= total_q) {
        return;
    }

    const uint q_cnt = min(BQ, total_q - flat_q_base);
    const uint min_q_pos = p.q_start_pos + (flat_q_base / gqa_ratio);
    const uint max_q_pos = p.q_start_pos + ((flat_q_base + q_cnt - 1u) / gqa_ratio);
    const uint kv_end = min(p.kv_len, max_q_pos + 1u);
    const uint kv_begin = (p.window_size > 0u && (min_q_pos + 1u) > p.window_size)
        ? (min_q_pos + 1u - p.window_size)
        : 0u;

    {
        const uint load_row0 = tid / D_VEC4;
        const uint load_col4 = tid & (D_VEC4 - 1u);
        for (uint row = load_row0; row < BQ; row += ROW_STEP) {
            const uint flat_q = flat_q_base + row;
            if (flat_q < total_q) {
                const uint qi = flat_q / gqa_ratio;
                const uint qh = kv_head * gqa_ratio + (flat_q - qi * gqa_ratio);
                const ulong q_off = (ulong(qi) * p.num_q_heads + qh) * D + load_col4 * 4u;
                u_smem.KVs4[row * D_PAD4 + load_col4] = *(const device half4 *)(q + q_off);
            } else {
                u_smem.KVs4[row * D_PAD4 + load_col4] = half4(0.0h);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint r0 = sg_id * 8u;
    const threadgroup half *KVs = (const threadgroup half *)u_smem.KVs4;
    simdgroup_half8x8 q_frag[D_TILES8];
    #pragma clang loop unroll(full)
    for (uint kt = 0u; kt < D_TILES8; kt++) {
        simdgroup_load(q_frag[kt], KVs + r0 * D_PAD + kt * 8u, D_PAD);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_val[8];
    float l_val[8];
    float o_reg[8][2];
    #pragma clang loop unroll(full)
    for (uint r = 0u; r < 8u; r++) {
        m_val[r] = -INFINITY;
        l_val[r] = 0.0f;
        o_reg[r][0] = 0.0f;
        o_reg[r][1] = 0.0f;
    }

    const float scale_log2e = p.scale * LOG2E;

    for (uint kv_block_start = kv_begin; kv_block_start < kv_end; kv_block_start += BK) {
        const uint kv_block_end = min(kv_block_start + BK, kv_end);

        qsa_mma_load_kv<D, D_PAD, BK, false>(
            u_smem.KVs4, kv_tiles, k_chunk, retired_pool, retired_chunk_ids, p,
            kv_head, kv_block_start, kv_block_end, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 s_frag[8];
        #pragma clang loop unroll(full)
        for (uint bj = 0u; bj < 8u; bj++) {
            s_frag[bj] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
        #pragma clang loop unroll(full)
        for (uint kt = 0u; kt < D_TILES8; kt++) {
            const uint kk = kt * 8u;
            #pragma clang loop unroll(full)
            for (uint bj = 0u; bj < 8u; bj++) {
                simdgroup_half8x8 k_frag;
                simdgroup_load(k_frag, KVs + (bj * 8u) * D_PAD + kk, D_PAD, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(s_frag[bj], q_frag[kt], k_frag, s_frag[bj]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (uint bj = 0u; bj < 8u; bj++) {
            simdgroup_store(s_frag[bj], u_smem.Ss + r0 * BK + bj * 8u, BK);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        float alpha[8];
        const uint kv_pos0 = kv_block_start + lane;
        const uint kv_pos1 = kv_block_start + lane + 32u;
        const bool in_blk0 = (kv_pos0 < kv_block_end);
        const bool in_blk1 = (kv_pos1 < kv_block_end);
        #pragma clang loop unroll(full)
        for (uint r = 0u; r < 8u; r++) {
            const uint flat_q = flat_q_base + r0 + r;
            const bool act = (flat_q < total_q);
            const uint qp = p.q_start_pos + (act ? (flat_q / gqa_ratio) : 0u);
            const float raw_s0 = u_smem.Ss[(r0 + r) * BK + lane];
            const float raw_s1 = u_smem.Ss[(r0 + r) * BK + lane + 32u];
            const bool v0 = act && in_blk0 && (kv_pos0 <= qp) &&
                            (p.window_size == 0u || (qp - kv_pos0) < p.window_size);
            const bool v1 = act && in_blk1 && (kv_pos1 <= qp) &&
                            (p.window_size == 0u || (qp - kv_pos1) < p.window_size);
            const float sc0 = v0 ? (raw_s0 * scale_log2e) : -INFINITY;
            const float sc1 = v1 ? (raw_s1 * scale_log2e) : -INFINITY;
            const float row_m = simd_max(max(sc0, sc1));
            float p0 = 0.0f;
            float p1 = 0.0f;
            if (row_m > -INFINITY) {
                const float m_new = max(m_val[r], row_m);
                const float a = (l_val[r] > 0.0f) ? exp2(m_val[r] - m_new) : 0.0f;
                p0 = v0 ? exp2(sc0 - m_new) : 0.0f;
                p1 = v1 ? exp2(sc1 - m_new) : 0.0f;
                const float row_l = simd_sum(p0 + p1);
                l_val[r] = fma(l_val[r], a, row_l);
                m_val[r] = m_new;
                alpha[r] = a;
            } else {
                alpha[r] = 1.0f;
            }
            Ps[(r0 + r) * BK_PAD + lane] = half(p0);
            Ps[(r0 + r) * BK_PAD + lane + 32u] = half(p1);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        qsa_mma_load_kv<D, D_PAD, BK, true>(
            u_smem.KVs4, kv_tiles, v_chunk, retired_pool, retired_chunk_ids, p,
            kv_head, kv_block_start, kv_block_end, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (uint p16 = 0u; p16 < 4u; p16++) {
            const uint d_blk = p16 * 16u;
            simdgroup_float8x8 pv[2];
            pv[0] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            pv[1] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            #pragma clang loop unroll(full)
            for (uint kk = 0u; kk < BK; kk += 8u) {
                simdgroup_half8x8 p_frag;
                simdgroup_load(p_frag, Ps + r0 * BK_PAD + kk, BK_PAD);
                simdgroup_half8x8 v0, v1;
                simdgroup_load(v0, KVs + kk * D_PAD + d_blk + 0u, D_PAD);
                simdgroup_load(v1, KVs + kk * D_PAD + d_blk + 8u, D_PAD);
                simdgroup_multiply_accumulate(pv[0], p_frag, v0, pv[0]);
                simdgroup_multiply_accumulate(pv[1], p_frag, v1, pv[1]);
            }
            simdgroup_store(pv[0], Os + r0 * 16u + 0u, 16u);
            simdgroup_store(pv[1], Os + r0 * 16u + 8u, 16u);
            simdgroup_barrier(mem_flags::mem_threadgroup);

            const uint c = p16 >> 1;
            const bool hi_half = ((p16 & 1u) != 0u);
            if (!hi_half) {
                if (lane < 16u) {
                    #pragma clang loop unroll(full)
                    for (uint r = 0u; r < 8u; r++) {
                        o_reg[r][c] = fma(o_reg[r][c], alpha[r], Os[(r0 + r) * 16u + lane]);
                    }
                }
            } else {
                if (lane >= 16u) {
                    const uint col = lane - 16u;
                    #pragma clang loop unroll(full)
                    for (uint r = 0u; r < 8u; r++) {
                        o_reg[r][c] = fma(o_reg[r][c], alpha[r], Os[(r0 + r) * 16u + col]);
                    }
                }
            }
            if (p16 + 1u < 4u) {
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma clang loop unroll(full)
    for (uint r = 0u; r < 8u; r++) {
        const uint flat_q = flat_q_base + r0 + r;
        if (flat_q >= total_q) {
            continue;
        }
        const uint qi = flat_q / gqa_ratio;
        const uint qh = kv_head * gqa_ratio + (flat_q - qi * gqa_ratio);
        const float inv_l = (l_val[r] > 0.0f) ? (1.0f / l_val[r]) : 0.0f;
        const ulong out_base = (ulong(qi) * p.num_q_heads + qh) * D;
        out[out_base + lane] = half(o_reg[r][0] * inv_l);
        out[out_base + 32u + lane] = half(o_reg[r][1] * inv_l);
    }
}

kernel void ami_qsa_swa_prefill_attn_mma(const device half *q [[buffer(0)]],
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
    union QsaMmaSmem {
        struct {
            half4 KVs4[32u * 34u];  // 8,704 B
            float Ss[32u * 32u];    // 4,096 B
            half Ps[32u * 40u];     // 2,560 B
        } d128;                     // 15,360 B total
        struct {
            QsaMmaD64Union u;       // 9,216 B
            half Ps[32u * 72u];     // 4,608 B
            float Os[32u * 16u];    // 2,048 B
        } d64;                      // 15,872 B total (<= 16 KiB)
    };
    threadgroup QsaMmaSmem smem;

    if (QSA_HEAD_DIM == 128u) {
        qsa_mma_exec_d128(q, kv_tiles, k_chunk, v_chunk, out, retired_pool, retired_chunk_ids, p,
                          smem.d128.KVs4, smem.d128.Ss, smem.d128.Ps, tg_pos, tid, sg_id, lane);
    } else {
        qsa_mma_exec_d64(q, kv_tiles, k_chunk, v_chunk, out, retired_pool, retired_chunk_ids, p,
                         smem.d64.u, smem.d64.Ps, smem.d64.Os, tg_pos, tid, sg_id, lane);
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

// ---------------------------------------------------------------------------
// Kernel (`G11` + `G19`, ADR-0023 §2): `ami_mla_csa2_attend_mma`
//
// Hardware `simdgroup_half8x8` (`Apple9`) tiled FlashAttention-2 prefill for
// DeepSeek-V4.1 MLA + CSA2 prefill (`BQ = 32, BK = 32` per tile, `q_head` x `q_tile`).
// Uses <= 16 KiB bank-conflict-free padded threadgroup SRAM:
//   - Qs[32, 40] (2,560 B) + KVs[32, 40] (2,560 B) = 5,120 B
//   - Ss[32, 32] (4,096 B)
//   - Ps[32, 40] (2,560 B)
//   - Os[32, 32] (4,096 B)
//   Total = 15,872 B (<= 16 KiB).
// Streams `32`-dim slices of `kv_lora_rank` and `qk_rope_dim` for `Q * K^T`
// and `32`-dim slices of `kv_lora_rank` for `O += exp(S - m) * V` via
// `simdgroup_half8x8`, strictly gated on `Q_len >= 32` and `KV_len >= 64`
// (non-tree, non-batch prefill) by `src/gpu/hybrid.zig`.
// Grid: (num_q_heads, ceil(num_nodes / 32), 1) threadgroups, 128 threads.
// ---------------------------------------------------------------------------
static inline void mla_mma_stage_kv_slice(threadgroup half *dst,
                                          const device uchar *e2m1_base,
                                          const device uchar *scale_base,
                                          const device uchar *rope_base,
                                          const device half *swa_lat_base,
                                          const device half *swa_rope_base,
                                          constant MlaCsa2AttnParams &p,
                                          uint kv_pos,
                                          uint ring_valid_lo,
                                          uint slice_off,
                                          uint slice_len,
                                          bool is_rope,
                                          uint num_bytes,
                                          ulong rope_row_bytes,
                                          uint lane) {
    half v = half(0.0h);
    if (lane < slice_len) {
        const uint d = slice_off + lane;
        const bool use_swa = (p.swa_ring_capacity > 0u && kv_pos >= ring_valid_lo);
        if (is_rope) {
            if (use_swa) {
                const uint slot = kv_pos % p.swa_ring_capacity;
                v = swa_rope_base[ulong(slot) * p.qk_rope_dim + d];
            } else {
                const device uchar *rope_row = rope_base + ulong(kv_pos) * rope_row_bytes;
                v = (p.rope_tail_format == 1u)
                    ? half(mla_e4m3_to_f32(rope_row[d]))
                    : ((const device half *)rope_row)[d];
            }
        } else {
            if (use_swa) {
                const uint slot = kv_pos % p.swa_ring_capacity;
                v = swa_lat_base[ulong(slot) * p.kv_lora_rank + d];
            } else {
                const uchar packed = e2m1_base[ulong(kv_pos) * num_bytes + (d >> 1)];
                const float sc = mla_e4m3_to_f32(scale_base[ulong(kv_pos) * (p.kv_lora_rank >> 4) + (d >> 4)]);
                const uint nib = ((d & 1u) == 0u) ? (uint(packed) & 0xFu) : ((uint(packed) >> 4) & 0xFu);
                v = half(mla_e2m1_to_f32(nib) * sc);
            }
        }
    }
    dst[lane] = v;
}

static inline void mla_mma_attend_kv_block(const device half *q_nope,
                                           const device half *q_pe,
                                           const device uchar *e2m1_base,
                                           const device uchar *scale_base,
                                           const device uchar *rope_base,
                                           const device half *swa_lat_base,
                                           const device half *swa_rope_base,
                                           constant MlaCsa2AttnParams &p,
                                           threadgroup half *Qs,
                                           threadgroup half *KVs,
                                           threadgroup float *Ss,
                                           threadgroup half *Ps,
                                           threadgroup float *Os,
                                           uint q_head,
                                           uint q_base,
                                           uint kv_start,
                                           uint kv_end,
                                           uint out_d0,
                                           uint num_slices32,
                                           uint num_bytes,
                                           ulong rope_row_bytes,
                                           uint ring_valid_lo,
                                           float scale_log2e,
                                           bool is_swa_pass,
                                           uint tile_row_mask,
                                           uint sg_id,
                                           uint lane,
                                           thread float *m_val,
                                           thread float *l_val,
                                           thread float *o_reg) {
    constexpr uint BQ = 32u;
    constexpr uint BK = 32u;
    constexpr uint SLICE_PAD = 40u;
    const uint r0 = sg_id * 8u;

    // 1. S = Q[32, kv_lora_rank + qk_rope_dim] @ K[32, ...]^T via simdgroup_half8x8.
    simdgroup_float8x8 s_frag[4];
    #pragma clang loop unroll(full)
    for (uint bj = 0u; bj < 4u; bj++) {
        s_frag[bj] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    for (uint sl = 0u; sl < p.kv_lora_rank; sl += 32u) {
        const uint sl_len = min(32u, p.kv_lora_rank - sl);
        for (uint rr = sg_id; rr < BQ; rr += 4u) {
            const uint q_idx = q_base + rr;
            const half qv = (q_idx < p.num_nodes && lane < sl_len)
                ? q_nope[(ulong(q_idx) * p.num_q_heads + q_head) * p.kv_lora_rank + sl + lane]
                : half(0.0h);
            Qs[rr * SLICE_PAD + lane] = qv;

            const uint kv_pos = kv_start + rr;
            if (kv_pos < kv_end) {
                mla_mma_stage_kv_slice(KVs + rr * SLICE_PAD, e2m1_base, scale_base, rope_base,
                                       swa_lat_base, swa_rope_base, p, kv_pos, ring_valid_lo,
                                       sl, sl_len, false, num_bytes, rope_row_bytes, lane);
            } else {
                KVs[rr * SLICE_PAD + lane] = half(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (uint kk = 0u; kk < 32u; kk += 8u) {
            simdgroup_half8x8 q_f;
            simdgroup_load(q_f, Qs + r0 * SLICE_PAD + kk, SLICE_PAD);
            #pragma clang loop unroll(full)
            for (uint bj = 0u; bj < 4u; bj++) {
                simdgroup_half8x8 k_f;
                simdgroup_load(k_f, KVs + (bj * 8u) * SLICE_PAD + kk, SLICE_PAD, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(s_frag[bj], q_f, k_f, s_frag[bj]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint sl = 0u; sl < p.qk_rope_dim; sl += 32u) {
        const uint sl_len = min(32u, p.qk_rope_dim - sl);
        for (uint rr = sg_id; rr < BQ; rr += 4u) {
            const uint q_idx = q_base + rr;
            const half qv = (q_idx < p.num_nodes && lane < sl_len)
                ? q_pe[(ulong(q_idx) * p.num_q_heads + q_head) * p.qk_rope_dim + sl + lane]
                : half(0.0h);
            Qs[rr * SLICE_PAD + lane] = qv;

            const uint kv_pos = kv_start + rr;
            if (kv_pos < kv_end) {
                mla_mma_stage_kv_slice(KVs + rr * SLICE_PAD, e2m1_base, scale_base, rope_base,
                                       swa_lat_base, swa_rope_base, p, kv_pos, ring_valid_lo,
                                       sl, sl_len, true, num_bytes, rope_row_bytes, lane);
            } else {
                KVs[rr * SLICE_PAD + lane] = half(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (uint kk = 0u; kk < 32u; kk += 8u) {
            simdgroup_half8x8 q_f;
            simdgroup_load(q_f, Qs + r0 * SLICE_PAD + kk, SLICE_PAD);
            #pragma clang loop unroll(full)
            for (uint bj = 0u; bj < 4u; bj++) {
                simdgroup_half8x8 k_f;
                simdgroup_load(k_f, KVs + (bj * 8u) * SLICE_PAD + kk, SLICE_PAD, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(s_frag[bj], q_f, k_f, s_frag[bj]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma clang loop unroll(full)
    for (uint bj = 0u; bj < 4u; bj++) {
        simdgroup_store(s_frag[bj], Ss + r0 * BK + bj * 8u, BK);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // 2. Causal, SWA-dedup, and block-selection masking + online softmax.
    float alpha[8];
    const uint kv_pos = kv_start + lane;
    const bool in_blk = (kv_pos < kv_end);
    #pragma clang loop unroll(full)
    for (uint r = 0u; r < 8u; r++) {
        const uint q_local = r0 + r;
        const uint q_idx = q_base + q_local;
        const bool q_in = (q_idx < p.num_nodes);
        const bool sel_ok = is_swa_pass || (((tile_row_mask >> q_local) & 1u) != 0u);
        const bool act = q_in && sel_ok;
        const uint eff_kv = q_in ? ((p.causal_q_start_pos != 0xFFFFFFFFu)
            ? min(p.kv_len, max(p.encoder_prefix_len, p.causal_q_start_pos + q_idx + 1u))
            : p.kv_len) : 0u;
        const uint swa_lo = (p.swa_window > 0u)
            ? ((eff_kv > p.swa_window) ? (eff_kv - p.swa_window) : 0u)
            : eff_kv;
        const bool in_swa = (kv_pos >= swa_lo && kv_pos < eff_kv);
        const bool valid = act && in_blk && (kv_pos < eff_kv) && (is_swa_pass ? in_swa : !in_swa);
        const float score = valid ? (Ss[q_local * BK + lane] * scale_log2e) : -INFINITY;
        const float row_m = simd_max(score);
        float p_val = 0.0f;
        if (row_m > -INFINITY) {
            const float m_new = max(m_val[r], row_m);
            const float a = (l_val[r] > 0.0f) ? exp2(m_val[r] - m_new) : 0.0f;
            p_val = valid ? exp2(score - m_new) : 0.0f;
            const float row_l = simd_sum(p_val);
            l_val[r] = fma(l_val[r], a, row_l);
            m_val[r] = m_new;
            alpha[r] = a;
        } else {
            alpha[r] = 1.0f;
        }
        Ps[q_local * SLICE_PAD + lane] = half(p_val);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. O += P[32, 32] @ V[32, chunk_cols] via simdgroup_half8x8 in 32-col slices.
    #pragma clang loop unroll(full)
    for (uint sl_i = 0u; sl_i < 4u; sl_i++) {
        if (sl_i < num_slices32) {
            const uint d_off = out_d0 + sl_i * 32u;
            const uint sl_len = min(32u, p.kv_lora_rank - d_off);
            for (uint rr = sg_id; rr < BK; rr += 4u) {
                const uint kpos = kv_start + rr;
                if (kpos < kv_end) {
                    mla_mma_stage_kv_slice(KVs + rr * SLICE_PAD, e2m1_base, scale_base, rope_base,
                                           swa_lat_base, swa_rope_base, p, kpos, ring_valid_lo,
                                           d_off, sl_len, false, num_bytes, rope_row_bytes, lane);
                } else {
                    KVs[rr * SLICE_PAD + lane] = half(0.0h);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_float8x8 pv[4];
            #pragma clang loop unroll(full)
            for (uint bj = 0u; bj < 4u; bj++) {
                pv[bj] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            }
            #pragma clang loop unroll(full)
            for (uint kk = 0u; kk < BK; kk += 8u) {
                simdgroup_half8x8 p_frag;
                simdgroup_load(p_frag, Ps + r0 * SLICE_PAD + kk, SLICE_PAD);
                #pragma clang loop unroll(full)
                for (uint bj = 0u; bj < 4u; bj++) {
                    simdgroup_half8x8 v_frag;
                    simdgroup_load(v_frag, KVs + kk * SLICE_PAD + bj * 8u, SLICE_PAD);
                    simdgroup_multiply_accumulate(pv[bj], p_frag, v_frag, pv[bj]);
                }
            }
            #pragma clang loop unroll(full)
            for (uint bj = 0u; bj < 4u; bj++) {
                simdgroup_store(pv[bj], Os + r0 * 32u + bj * 8u, 32u);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            #pragma clang loop unroll(full)
            for (uint r = 0u; r < 8u; r++) {
                o_reg[r * 4u + sl_i] = fma(o_reg[r * 4u + sl_i], alpha[r], Os[(r0 + r) * 32u + lane]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

kernel void ami_mla_csa2_attend_mma(const device half *q_nope [[buffer(0)]],
                                    const device half *q_pe [[buffer(1)]],
                                    const device uchar *kv_slab [[buffer(2)]],
                                    const device uchar *swa_ring [[buffer(3)]],
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
    (void)kv_tree_latent;
    (void)kv_tree_rope;
    (void)ancestor_masks;
    (void)batch_kv_lens;

    constexpr uint BQ = 32u;
    constexpr uint BK = 32u;
    constexpr uint SLICE_PAD = 40u;

    const uint q_head = tg_pos.x;
    const uint q_base = tg_pos.y * BQ;
    if (q_head >= p.num_q_heads || q_base >= p.num_nodes) {
        return;
    }

    // 15,872 B total threadgroup SRAM (<= 16 KiB).
    threadgroup half Qs[BQ * SLICE_PAD];   // 2,560 B
    threadgroup half KVs[BK * SLICE_PAD];  // 2,560 B
    threadgroup float Ss[BQ * BK];         // 4,096 B
    threadgroup half Ps[BQ * SLICE_PAD];   // 2,560 B
    threadgroup float Os[BQ * 32u];        // 4,096 B

    const uint num_bytes = p.kv_lora_rank >> 1;
    const device uchar *e2m1_base = kv_slab + p.e2m1_plane_offset;
    const device uchar *scale_base = kv_slab + p.scale_plane_offset;
    const device uchar *rope_base = kv_slab + p.rope_plane_offset;
    const ulong rope_row_bytes = (p.rope_tail_format == 1u)
        ? ulong(p.qk_rope_dim)
        : (ulong(p.qk_rope_dim) * 2ul);
    const device half *swa_lat_base = (const device half *)swa_ring;
    const device half *swa_rope_base = (const device half *)(swa_ring + p.swa_rope_plane_offset);

    const uint r0 = sg_id * 8u;
    const uint q_cnt = min(BQ, p.num_nodes - q_base);

    uint min_eff_kv = p.kv_len;
    uint max_eff_kv = 0u;
    for (uint r = 0u; r < q_cnt; r++) {
        const uint q_idx = q_base + r;
        const uint eff = (p.causal_q_start_pos != 0xFFFFFFFFu)
            ? min(p.kv_len, max(p.encoder_prefix_len, p.causal_q_start_pos + q_idx + 1u))
            : p.kv_len;
        min_eff_kv = min(min_eff_kv, eff);
        max_eff_kv = max(max_eff_kv, eff);
    }

    const uint ring_valid_lo = (p.swa_ring_capacity > 0u && p.kv_len > p.swa_ring_capacity)
        ? (p.kv_len - p.swa_ring_capacity)
        : 0u;
    const float scale_log2e = p.scale * 1.4426950408889634f;
    const uint swa_lo_tile = (p.swa_window > 0u)
        ? ((min_eff_kv > p.swa_window) ? (min_eff_kv - p.swa_window) : 0u)
        : max_eff_kv;
    const uint num_blocks = (max_eff_kv + p.block_tokens - 1u) / p.block_tokens;

    for (uint out_d0 = 0u; out_d0 < p.kv_lora_rank; out_d0 += 128u) {
        const uint chunk_cols = min(128u, p.kv_lora_rank - out_d0);
        const uint num_slices32 = (chunk_cols + 31u) >> 5;

        // Keep online softmax stats and O accumulators in scalar registers (0 allocas).
        float m_val[8];
        float l_val[8];
        float o_reg[32];
        #pragma clang loop unroll(full)
        for (uint r = 0u; r < 8u; r++) {
            m_val[r] = -INFINITY;
            l_val[r] = 0.0f;
            #pragma clang loop unroll(full)
            for (uint c = 0u; c < 4u; c++) {
                o_reg[r * 4u + c] = 0.0f;
            }
        }

        // Phase 1: Selected sparse CSA2 blocks + Phase 2: SWA bounded replay.
        // Unified single loop (`pass == 0`: sparse blocks, `pass == 1`: SWA) so
        // `mla_mma_attend_kv_block` is called once inside the loop and fully inlined.
        for (uint pass = 0u; pass < 2u; pass++) {
            const bool is_swa = (pass == 1u);
            const uint n_iter = is_swa ? 1u : num_blocks;
            for (uint it = 0u; it < n_iter; it++) {
                uint tile_mask = 0xFFFFFFFFu;
                uint seg_start = swa_lo_tile;
                uint seg_end = max_eff_kv;
                if (!is_swa) {
                    tile_mask = 0u;
                    for (uint r = 0u; r < q_cnt; r++) {
                        const uint q_idx = q_base + r;
                        const uint sc = min(selected_counts[q_idx], p.top_k_blocks);
                        const ulong sb_off = ulong(q_idx) * p.top_k_blocks;
                        for (uint j = 0u; j < sc; j++) {
                            if (selected_blocks[sb_off + j] == it) {
                                tile_mask |= (1u << r);
                                break;
                            }
                        }
                    }
                    if (tile_mask == 0u) {
                        continue;
                    }
                    seg_start = it * p.block_tokens;
                    seg_end = min(seg_start + p.block_tokens, max_eff_kv);
                }
                for (uint kv_start = seg_start; kv_start < seg_end; kv_start += BK) {
                    const uint kv_end = min(kv_start + BK, seg_end);
                    mla_mma_attend_kv_block(
                        q_nope, q_pe, e2m1_base, scale_base, rope_base, swa_lat_base, swa_rope_base,
                        p, Qs, KVs, Ss, Ps, Os, q_head, q_base, kv_start, kv_end, out_d0, num_slices32,
                        num_bytes, rope_row_bytes, ring_valid_lo, scale_log2e, is_swa, tile_mask,
                        sg_id, lane, m_val, l_val, o_reg);
                }
            }
        }

        // Write normalized FP16 output slice for this simdgroup's 8 rows.
        #pragma clang loop unroll(full)
        for (uint r = 0u; r < 8u; r++) {
            const uint q_idx = q_base + r0 + r;
            if (q_idx >= p.num_nodes) {
                continue;
            }
            const float inv_l = (l_val[r] > 0.0f) ? (1.0f / l_val[r]) : 0.0f;
            const ulong out_base = (ulong(q_idx) * p.num_q_heads + q_head) * p.kv_lora_rank + out_d0;
            if (0u < num_slices32 && lane < chunk_cols) out[out_base + lane] = half(o_reg[r * 4u + 0u] * inv_l);
            if (1u < num_slices32 && 32u + lane < chunk_cols) out[out_base + 32u + lane] = half(o_reg[r * 4u + 1u] * inv_l);
            if (2u < num_slices32 && 64u + lane < chunk_cols) out[out_base + 64u + lane] = half(o_reg[r * 4u + 2u] * inv_l);
            if (3u < num_slices32 && 96u + lane < chunk_cols) out[out_base + 96u + lane] = half(o_reg[r * 4u + 3u] * inv_l);
        }
    }
    (void)tid;
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

// ===========================================================================
// Section E: G26 Asymmetric K/V (`192/128`, `256/256`, dynamic <= 256) SWA
// (`128 + sink`), Value Scaling, Partial/3D-MRoPE + Optional `(1 + w)` QK-Norm,
// Sigmoid + `e_score_correction_bias` MoE Router, and DFlash Tap Ring Capture
// ===========================================================================

static inline float g26_hybrid_load_scalar_f32(const device uchar *buf, uint idx, uint dtype) {
    if (dtype == 2u) {
        return ((const device float *)buf)[idx];
    }
    if (dtype == 1u) {
        return float(((const device bfloat *)buf)[idx]);
    }
    return float(((const device half *)buf)[idx]);
}

struct G26QkNormRopeKvParams {
    uint num_tokens;
    uint num_q_heads;
    uint num_kv_heads;
    uint k_head_dim;
    uint v_head_dim;
    uint rotary_dim;
    uint start_pos;
    uint chunk_tokens;
    uint num_retired_chunks;
    uint ring_capacity;
    uint tile_table_offset;
    uint mrope_s0;
    uint mrope_s1;
    uint mrope_s2;
    uint pos_mode;             // 0 = linear (start_pos + tok), 1 = positions_1d[tok], 2 = positions_3d[tok*3 + axis]
    uint apply_qk_norm;
    uint qk_norm_unit_offset;  // 1 = (1.0 + w) for qwen3_5, 0 = plain w
    uint kv_only;
    uint write_to_tiles;       // 1 = write into bindless kv_tiles, 0 = write into k_out/v_out
    uint _pad0;
    float rms_eps;
    float rope_theta;
    float v_scale;
    float yarn_factor;
    float yarn_mscale;
    float yarn_beta_fast;
    float yarn_beta_slow;
    uint original_max_pos;
};

static_assert(sizeof(G26QkNormRopeKvParams) == 112, "G26QkNormRopeKvParams must be 112 bytes");

// Asymmetric K/V QK-Norm (optional `(1 + w)`) + Partial NeoX/YaRN/3D-MRoPE +
// `attention_value_scale` + either bindless `HybridKvRing` tile write or
// contiguous `k_out`/`v_out` staging (zero spills).
// Specialized on HYBRID_NORM_W_BF16 (`function_constant(1)`).
// Grid: ((kv_only ? num_kv_heads : num_q_heads + num_kv_heads), num_tokens, 1), 32 threads.
kernel void ami_g26_qknorm_rope_kv(device half *q [[buffer(0)]],
                                   const device half *k [[buffer(1)]],
                                   const device half *v [[buffer(2)]],
                                   const device uchar *q_norm [[buffer(3)]],
                                   const device uchar *k_norm [[buffer(4)]],
                                   const device uint *positions [[buffer(5)]],
                                   device half *k_out [[buffer(6)]],
                                   device half *v_out [[buffer(7)]],
                                   const device QsaKvTileWritePair *kv_tiles [[buffer(8)]],
                                   constant G26QkNormRopeKvParams &p [[buffer(9)]],
                                   uint2 tg_pos [[threadgroup_position_in_grid]],
                                   uint lane [[thread_index_in_simdgroup]]) {
    const uint total_heads = (p.kv_only != 0u) ? p.num_kv_heads : (p.num_q_heads + p.num_kv_heads);
    const uint tok = tg_pos.y;
    if (tg_pos.x >= total_heads || tok >= p.num_tokens) {
        return;
    }

    const bool is_q = (p.kv_only == 0u) && (tg_pos.x < p.num_q_heads);
    const uint head_idx = is_q ? tg_pos.x : (tg_pos.x - ((p.kv_only != 0u) ? 0u : p.num_q_heads));

    if (!is_q && p.write_to_tiles != 0u && p.ring_capacity > 0u && (tok + p.ring_capacity < p.num_tokens)) {
        return;
    }

    threadgroup float vec[256];
    const uint k_dim = p.k_head_dim;
    const uint v_dim = p.v_head_dim;
    const uint rot_dim = (p.rotary_dim > 0u) ? p.rotary_dim : k_dim;
    const uint half_rot = rot_dim >> 1;

    const ulong q_off = (ulong(tok) * p.num_q_heads + head_idx) * k_dim;
    const ulong k_off = (ulong(tok) * p.num_kv_heads + head_idx) * k_dim;
    const ulong v_off = (ulong(tok) * p.num_kv_heads + head_idx) * v_dim;
    const device half *src = is_q ? (q + q_off) : (k + k_off);

    float sum_sq = 0.0f;
    for (uint d = lane; d < k_dim; d += 32u) {
        const float val = float(src[d]);
        vec[d] = val;
        sum_sq = fma(val, val, sum_sq);
    }

    if (p.apply_qk_norm != 0u) {
        const float total_sq = simd_sum(sum_sq);
        const float inv_rms = rsqrt(total_sq / float(k_dim) + p.rms_eps);
        const device uchar *nw = is_q ? q_norm : k_norm;
        const float w_add = (p.qk_norm_unit_offset != 0u) ? 1.0f : 0.0f;
        for (uint d = lane; d < k_dim; d += 32u) {
            vec[d] = (vec[d] * inv_rms) * (hybrid_load_norm_w(nw, d, HYBRID_NORM_W_BF16) + w_add);
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    const uint lin_pos = (p.pos_mode == 1u)
        ? positions[tok]
        : ((p.pos_mode == 2u) ? positions[ulong(tok) * 3ul + 0ul] : (p.start_pos + tok));
    const uint pos_t = (p.pos_mode == 2u) ? positions[ulong(tok) * 3ul + 0ul] : lin_pos;
    const uint pos_h = (p.pos_mode == 2u) ? positions[ulong(tok) * 3ul + 1ul] : lin_pos;
    const uint pos_w = (p.pos_mode == 2u) ? positions[ulong(tok) * 3ul + 2ul] : lin_pos;

    for (uint j = lane; j < half_rot; j += 32u) {
        uint axis_pos = lin_pos;
        if (p.pos_mode == 2u) {
            const uint axis = mrope_interleaved_axis(j, p.mrope_s0, p.mrope_s1, p.mrope_s2);
            axis_pos = (axis == 0u) ? pos_t : ((axis == 1u) ? pos_h : pos_w);
        }
        const float inv_f = yarn_inv_freq(
            j, rot_dim, p.rope_theta, p.yarn_factor,
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
        for (uint d = lane; d < k_dim; d += 32u) {
            dst[d] = half(vec[d]);
        }
        return;
    }

    const uint k_cols = k_dim >> 2;
    const uint v_cols = v_dim >> 2;
    const device half4 *v_src = (const device half4 *)(v + v_off);
    const bool do_v_scale = (p.v_scale != 1.0f);

    if (p.write_to_tiles != 0u) {
        const uint phys_pos = (p.ring_capacity > 0u) ? (lin_pos % p.ring_capacity) : lin_pos;
        const uint chunk = phys_pos / p.chunk_tokens;
        const uint slot = phys_pos - chunk * p.chunk_tokens;
        const uint active_chunk = (p.ring_capacity == 0u) ? (chunk - p.num_retired_chunks) : chunk;
        const uint row = head_idx * p.chunk_tokens + slot;
        const QsaKvTileWritePair tile = kv_tiles[p.tile_table_offset + active_chunk];
        for (uint col = lane; col < k_cols; col += 32u) {
            const uint base = col * 4u;
            const half4 k4 = half4(half(vec[base + 0u]),
                                   half(vec[base + 1u]),
                                   half(vec[base + 2u]),
                                   half(vec[base + 3u]));
            tile.k_tex.write(k4, uint2(col, row));
        }
        for (uint col = lane; col < v_cols; col += 32u) {
            const half4 raw_v = v_src[col];
            const half4 out_v = do_v_scale ? half4(float4(raw_v) * p.v_scale) : raw_v;
            tile.v_tex.write(out_v, uint2(col, row));
        }
    } else {
        device half *k_dst = k_out + k_off;
        for (uint d = lane; d < k_dim; d += 32u) {
            k_dst[d] = half(vec[d]);
        }
        device half4 *v_dst = (device half4 *)(v_out + v_off);
        for (uint col = lane; col < v_cols; col += 32u) {
            const half4 raw_v = v_src[col];
            v_dst[col] = do_v_scale ? half4(float4(raw_v) * p.v_scale) : raw_v;
        }
    }
}

struct G26KvCommitParams {
    uint num_kv_heads;
    uint k_head_dim;
    uint v_head_dim;
    uint start_pos;
    uint num_tokens;
    uint chunk_tokens;
    uint num_retired_chunks;
    uint ring_capacity;
    uint tile_table_offset;
    uint use_node_list;    // 0 = trailing min(num_tokens, ring_capacity) prefill chunk, 1 = accepted tree nodes
    uint use_node_buffer;  // 0 = inline_nodes[64], 1 = accepted_nodes buffer
    uint _pad0;
    ushort inline_nodes[64];
};

static_assert(sizeof(G26KvCommitParams) == 176, "G26KvCommitParams must be 176 bytes");

// Asymmetric K/V commit from prefill chunk OR accepted speculative tree nodes
// into bindless `HybridKvRing` / global `KvTileWritePair` tiles (zero spills).
// Grid: (num_kv_heads, commit_count, 1) threadgroups, 32 threads.
kernel void ami_g26_kv_commit(const device half *k_src [[buffer(0)]],
                              const device half *v_src [[buffer(1)]],
                              const device uint *accepted_nodes [[buffer(2)]],
                              const device QsaKvTileWritePair *kv_tiles [[buffer(3)]],
                              constant G26KvCommitParams &p [[buffer(4)]],
                              uint2 tg_pos [[threadgroup_position_in_grid]],
                              uint lane [[thread_index_in_simdgroup]]) {
    const uint kvh = tg_pos.x;
    const uint ci = tg_pos.y;
    const uint commit_cnt = (p.use_node_list == 0u && p.ring_capacity > 0u)
        ? min(p.num_tokens, p.ring_capacity)
        : p.num_tokens;
    if (kvh >= p.num_kv_heads || ci >= commit_cnt) {
        return;
    }

    uint src_tok;
    uint dst_pos;
    if (p.use_node_list != 0u) {
        src_tok = (p.use_node_buffer != 0u) ? accepted_nodes[ci] : uint(p.inline_nodes[ci]);
        dst_pos = p.start_pos + ci;
    } else {
        src_tok = (p.num_tokens - commit_cnt) + ci;
        dst_pos = p.start_pos + src_tok;
    }

    const uint phys_pos = (p.ring_capacity > 0u) ? (dst_pos % p.ring_capacity) : dst_pos;
    const uint chunk = phys_pos / p.chunk_tokens;
    const uint slot = phys_pos - chunk * p.chunk_tokens;
    const uint active_chunk = (p.ring_capacity == 0u) ? (chunk - p.num_retired_chunks) : chunk;
    const uint row = kvh * p.chunk_tokens + slot;
    const QsaKvTileWritePair tile = kv_tiles[p.tile_table_offset + active_chunk];

    const uint k_cols = p.k_head_dim >> 2;
    const uint v_cols = p.v_head_dim >> 2;
    const device half4 *k4 = (const device half4 *)(k_src + (ulong(src_tok) * p.num_kv_heads + kvh) * p.k_head_dim);
    const device half4 *v4 = (const device half4 *)(v_src + (ulong(src_tok) * p.num_kv_heads + kvh) * p.v_head_dim);

    for (uint col = lane; col < k_cols; col += 32u) {
        tile.k_tex.write(k4[col], uint2(col, row));
    }
    for (uint col = lane; col < v_cols; col += 32u) {
        tile.v_tex.write(v4[col], uint2(col, row));
    }
}

struct G26SwaAttnParams {
    uint num_q;
    uint q_start_pos;      // prefill: first query pos; decode/tree: base_pos
    uint chunk_start_pos;  // prefill: start pos of k_extra/v_extra chunk; decode/tree: base_pos
    uint kv_len;           // prefill: total kv_len including chunk; decode/tree: base_pos
    uint num_q_heads;
    uint num_kv_heads;
    uint k_head_dim;
    uint v_head_dim;
    uint window_size;      // e.g. 128 (0 = full causal)
    uint ring_capacity;    // e.g. 128 (0 = linear)
    uint chunk_tokens;     // e.g. 128 for SWA, 512 for linear
    uint tile_table_offset;
    uint is_prefill;       // 1 = prefill (ring prefix + contiguous chunk), 0 = decode/tree (ring prefix + ancestor_masks tree)
    uint has_tree;         // when is_prefill == 0: 1 = include tree k_extra/v_extra with ancestor_masks
    uint has_positions;    // when is_prefill == 0: 1 = read node_positions[qi] for exact per-node SWA window
    uint has_sink;
    uint sink_dtype;       // 0 = f16, 1 = bf16, 2 = f32
    uint _pad0;
    float scale;
    uint _pad1;
};

static_assert(sizeof(G26SwaAttnParams) == 80, "G26SwaAttnParams must be 80 bytes");

// Asymmetric K/V (`192/128`, `256/256`, dynamic <= 256) `HybridKvRing` SWA
// Attention across ALL 3 modes (`M = 1` decode, chunked prefill, and `M = 1..64`
// speculative `verifyTree` with 64-bit ancestor masks + per-node SWA window)
// with exact FP32 online-softmax `sink_bias` (0 stack spills).
// Specialized on G26_K_DIM (`function_constant(5)`) & G26_V_DIM (`function_constant(6)`).
// Grid: (num_kv_heads * q_groups_per_kv, num_q, 1) threadgroups, 256 threads.
kernel void ami_g26_swa_attn(const device half *q [[buffer(0)]],
                             const device QsaKvTilePair *kv_tiles [[buffer(1)]],
                             const device half *k_extra [[buffer(2)]],
                             const device half *v_extra [[buffer(3)]],
                             const device ulong *ancestor_masks [[buffer(4)]],
                             const device uint *node_positions [[buffer(5)]],
                             const device uchar *sink_bias [[buffer(6)]],
                             device half *out [[buffer(7)]],
                             constant G26SwaAttnParams &p [[buffer(8)]],
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

    const uint k_dim = (G26_K_DIM > 0u) ? G26_K_DIM : p.k_head_dim;
    const uint v_dim = (G26_V_DIM > 0u) ? G26_V_DIM : p.v_head_dim;
    const uint k_cols = k_dim >> 2;
    const uint v_cols = v_dim >> 2;
    const float scale_log2e = p.scale * LOG2E;

    const uint q_pos = (p.is_prefill != 0u)
        ? (p.q_start_pos + qi)
        : ((p.has_positions != 0u) ? node_positions[qi] : p.q_start_pos);
    const uint t_hi = (p.is_prefill != 0u) ? min(p.kv_len, q_pos + 1u) : p.kv_len;
    const uint t_lo = (p.window_size > 0u && (q_pos + 1u) > p.window_size)
        ? min(t_hi, (q_pos + 1u) - p.window_size)
        : 0u;

    const ulong q_row_off = (ulong(qi) * p.num_q_heads + qh) * k_dim;
    const float4 q0 = (active_q && lane < k_cols)
        ? float4(*(const device half4 *)(q + q_row_off + lane * 4u))
        : float4(0.0f);
    const float4 q1 = (active_q && (lane + 32u) < k_cols)
        ? float4(*(const device half4 *)(q + q_row_off + (lane + 32u) * 4u))
        : float4(0.0f);

    float m_val = -INFINITY;
    float l_val = 0.0f;
    if (p.has_sink != 0u && active_q) {
        m_val = g26_hybrid_load_scalar_f32(sink_bias, qh, p.sink_dtype) * LOG2E;
        l_val = 1.0f;
    }
    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);

    threadgroup half4 tg_k[256];
    threadgroup half4 tg_v[256];
    const uint load_tok = tid >> 6;  // 0..3
    const uint load_col = tid & 63u; // 0..63
    const uint head_row_base = kv_head * p.chunk_tokens;

    for (uint tile_t = t_lo; tile_t < t_hi; tile_t += 4u) {
        const uint tile_len = min(4u, t_hi - tile_t);
        if (load_tok < tile_len) {
            const uint t = tile_t + load_tok;
            if (p.is_prefill != 0u && t >= p.chunk_start_pos) {
                const uint c_tok = t - p.chunk_start_pos;
                if (load_col < k_cols) {
                    const ulong ck_off = (ulong(c_tok) * p.num_kv_heads + kv_head) * k_dim + load_col * 4u;
                    tg_k[tid] = *(const device half4 *)(k_extra + ck_off);
                }
                if (load_col < v_cols) {
                    const ulong cv_off = (ulong(c_tok) * p.num_kv_heads + kv_head) * v_dim + load_col * 4u;
                    tg_v[tid] = *(const device half4 *)(v_extra + cv_off);
                }
            } else {
                const uint phys_t = (p.ring_capacity > 0u) ? (t % p.ring_capacity) : t;
                const uint chunk = phys_t / p.chunk_tokens;
                const uint slot = phys_t - chunk * p.chunk_tokens;
                const uint row = head_row_base + slot;
                const QsaKvTilePair pair = kv_tiles[p.tile_table_offset + chunk];
                if (load_col < k_cols) {
                    tg_k[tid] = pair.k_tex.read(uint2(load_col, row));
                }
                if (load_col < v_cols) {
                    tg_v[tid] = pair.v_tex.read(uint2(load_col, row));
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active_q) {
            for (uint dt = 0u; dt < tile_len; dt++) {
                const uint base_idx = dt * 64u;
                float partial = (lane < k_cols) ? dot(q0, float4(tg_k[base_idx + lane])) : 0.0f;
                if ((lane + 32u) < k_cols) {
                    partial += dot(q1, float4(tg_k[base_idx + lane + 32u]));
                }
                const float score = simd_sum(partial) * scale_log2e;
                const float new_m = max(m_val, score);
                const float corr = (l_val > 0.0f) ? exp2(m_val - new_m) : 0.0f;
                const float p_w = exp2(score - new_m);
                l_val = fma(l_val, corr, p_w);
                if (lane < v_cols) {
                    acc0 = fma(acc0, float4(corr), float4(tg_v[base_idx + lane]) * p_w);
                }
                if ((lane + 32u) < v_cols) {
                    acc1 = fma(acc1, float4(corr), float4(tg_v[base_idx + lane + 32u]) * p_w);
                }
                m_val = new_m;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (p.is_prefill == 0u && p.has_tree != 0u && active_q) {
        ulong rem = ancestor_masks[qi];
        while (rem != 0ul) {
            const uint j = uint(ctz(rem));
            rem &= (rem - 1ul);
            if (j < p.num_q) {
                const uint j_pos = (p.has_positions != 0u) ? node_positions[j] : q_pos;
                if (p.window_size == 0u || (j_pos <= q_pos && (q_pos - j_pos) < p.window_size)) {
                    const ulong k_off = (ulong(j) * p.num_kv_heads + kv_head) * k_dim;
                    const ulong v_off = (ulong(j) * p.num_kv_heads + kv_head) * v_dim;
                    float partial = 0.0f;
                    if (lane < k_cols) {
                        partial = dot(q0, float4(*(const device half4 *)(k_extra + k_off + lane * 4u)));
                    }
                    if ((lane + 32u) < k_cols) {
                        partial += dot(q1, float4(*(const device half4 *)(k_extra + k_off + (lane + 32u) * 4u)));
                    }
                    const float score = simd_sum(partial) * scale_log2e;
                    const float new_m = max(m_val, score);
                    const float corr = (l_val > 0.0f) ? exp2(m_val - new_m) : 0.0f;
                    const float p_w = exp2(score - new_m);
                    l_val = fma(l_val, corr, p_w);
                    if (lane < v_cols) {
                        const float4 v0 = float4(*(const device half4 *)(v_extra + v_off + lane * 4u));
                        acc0 = fma(acc0, float4(corr), v0 * p_w);
                    }
                    if ((lane + 32u) < v_cols) {
                        const float4 v1 = float4(*(const device half4 *)(v_extra + v_off + (lane + 32u) * 4u));
                        acc1 = fma(acc1, float4(corr), v1 * p_w);
                    }
                    m_val = new_m;
                }
            }
        }
    }

    if (active_q) {
        const float inv_l = (l_val > 0.0f) ? (1.0f / l_val) : 0.0f;
        const ulong out_off = (ulong(qi) * p.num_q_heads + qh) * v_dim;
        if (lane < v_cols) {
            *(device half4 *)(out + out_off + lane * 4u) = half4(acc0 * inv_l);
        }
        if ((lane + 32u) < v_cols) {
            *(device half4 *)(out + out_off + (lane + 32u) * 4u) = half4(acc1 * inv_l);
        }
    }
}

// ---------------------------------------------------------------------------
// G26 MoE Sigmoid + `e_score_correction_bias` + Group Top-K + CSR Schedule
// Writes exact `MoeWorkspace` CSR & G20 indirect dispatch structures (uniform
// or G13 scheme-bucketed) with 0 thread-local arrays (0 stack spills).
// ---------------------------------------------------------------------------

struct G26MoeScheduleHeader {
    uint num_unique_experts;
    uint total_assignments;
    uint m;
    uint top_k;
};

struct G26MoeIndirectArgs {
    uint threadgroups_per_grid[3];
};

static inline void g26_write_schedule_indirect(device uchar *slot_base,
                                               uint active_experts,
                                               uint gate_up_groups_x_q,
                                               uint gate_up_groups_x_f,
                                               uint down_groups_x,
                                               uint down_groups_x_q8) {
    device G26MoeIndirectArgs *gu_q = (device G26MoeIndirectArgs *)(slot_base + 16ul);
    device G26MoeIndirectArgs *gu_f = (device G26MoeIndirectArgs *)(slot_base + 28ul);
    device G26MoeIndirectArgs *dn   = (device G26MoeIndirectArgs *)(slot_base + 40ul);
    device G26MoeIndirectArgs *dn8  = (device G26MoeIndirectArgs *)(slot_base + 52ul);
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

struct G26MoeRouteSigmoidParams {
    uint m;
    uint num_experts;
    uint top_k;
    uint norm_topk_prob;
    uint has_bias;
    uint bias_dtype;       // 0 = f16, 1 = bf16, 2 = f32
    uint n_group;          // >= 1
    uint topk_group;       // >= 1
    uint is_bucketed;      // 0 = uniform, 1 = heterogeneous 5-bucket CSR
    uint bucket_unique_stride;
    uint bucket_offsets_stride;
    uint gate_up_groups_x_q;
    uint gate_up_groups_x_f;
    uint down_groups_x;
    uint down_groups_x_q8;
    float routed_scaling_factor;
    uint group_score_topk; // 1 = max-1 per group; 0 or >= 2 = top-2 sum per group (DeepSeek-V3 / BailingMoeV3)
    uint score_mode;       // 0 = sigmoid, 1 = sqrt(softplus(x)) (DeepSeek-V4.1)
    uint _pad1;
    uint _pad2;
};

static inline float g26_route_score(float raw, uint score_mode) {
    if (score_mode == 1u) {
        const float sp = (raw > 20.0f) ? raw : ((raw < -20.0f) ? exp(raw) : log(1.0f + exp(raw)));
        return sqrt(max(sp, 0.0f));
    }
    return hybrid_sigmoid(raw);
}

static_assert(sizeof(G26MoeRouteSigmoidParams) == 80, "G26MoeRouteSigmoidParams must be 80 bytes");

// Single threadgroup (256 threads / 8 simdgroups). Supports num_experts <= 1024,
// top_k <= 16, n_group <= 32. Zero stack spills (`-Wframe-larger-than=0`).
// Specialized on G26_LOGITS_F16 (`function_constant(7)`).
kernel void ami_moe_route_topk_sigmoid(const device uchar *logits_raw [[buffer(0)]],
                                       const device uchar *bias_raw [[buffer(1)]],
                                       device G26MoeScheduleHeader *sched_hdr [[buffer(2)]],
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
                                       constant G26MoeRouteSigmoidParams &p [[buffer(14)]],
                                       uint tid [[thread_index_in_threadgroup]],
                                       uint sg_id [[simdgroup_index_in_threadgroup]],
                                       uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint NUM_SCHEMES = 5u;
    threadgroup atomic_uint s_counts[1024];
    threadgroup atomic_uint s_cursors[1024];
    threadgroup uint s_sel_e[8][16];
    threadgroup float s_sel_w[8][16];
    threadgroup uint s_picked[8][32];
    threadgroup uint s_prefix_tmp[8];
    threadgroup uint s_u_cnt;
    threadgroup uint s_a_cnt;

    const uint top_k = min(p.top_k, 16u);
    const uint chunks = (p.num_experts + 31u) >> 5;

    for (uint e = tid; e < p.num_experts; e += 256u) {
        atomic_store_explicit(&s_counts[e], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const bool use_groups = (p.n_group > 1u && p.topk_group < p.n_group && (p.num_experts % p.n_group) == 0u);
    const uint group_size = use_groups ? (p.num_experts / p.n_group) : p.num_experts;

    for (uint tok = sg_id; tok < p.m; tok += 8u) {
        const ulong row_base = ulong(tok) * p.num_experts;
        if (lane < chunks) {
            s_picked[sg_id][lane] = 0u;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // Optional DeepSeek/MiMo group top-k masking (`n_group <= 32`).
        uint group_allow_mask = 0xFFFFFFFFu;
        if (use_groups && p.n_group <= 32u) {
            uint picked_g = 0u;
            for (uint gk = 0u; gk < p.topk_group; gk++) {
                float best_gs = -INFINITY;
                uint best_g = 0xFFFFFFFFu;
                for (uint g = 0u; g < p.n_group; g++) {
                    if (((picked_g >> g) & 1u) != 0u) {
                        continue;
                    }
                    const uint g_start = g * group_size;
                    float top1 = -INFINITY;
                    float top2 = -INFINITY;
                    for (uint off = lane; off < group_size; off += 32u) {
                        const uint e = g_start + off;
                        const float raw = G26_LOGITS_F16
                            ? float(((const device half *)logits_raw)[row_base + e])
                            : ((const device float *)logits_raw)[row_base + e];
                        float c_val = -INFINITY;
                        if (isfinite(raw) && raw > -1.0e29f) {
                            const float sc = g26_route_score(raw, p.score_mode);
                            const float b = (p.has_bias != 0u) ? g26_hybrid_load_scalar_f32(bias_raw, e, p.bias_dtype) : 0.0f;
                            c_val = sc + b;
                        }
                        if (c_val >= top1) {
                            top2 = top1;
                            top1 = c_val;
                        } else if (c_val > top2) {
                            top2 = c_val;
                        }
                    }
                    // Reduce top-2 across 32 lanes.
                    for (ushort step = 16u; step > 0u; step >>= 1u) {
                        const float o1 = simd_shuffle_down(top1, step);
                        const float o2 = simd_shuffle_down(top2, step);
                        if (o1 >= top1) {
                            top2 = max(top1, o2);
                            top1 = o1;
                        } else if (o1 > top2) {
                            top2 = o1;
                        }
                    }
                    const float g_score = simd_broadcast_first(
                        (p.group_score_topk == 1u)
                            ? top1
                            : (top1 + ((top2 > -1.0e30f) ? top2 : 0.0f)));
                    if (g_score > best_gs || (g_score == best_gs && g < best_g)) {
                        best_gs = g_score;
                        best_g = g;
                    }
                }
                if (best_g < p.n_group) {
                    picked_g |= (1u << best_g);
                }
            }
            group_allow_mask = picked_g;
        }

        float w_sum = 0.0f;
        for (uint k = 0u; k < top_k; k++) {
            float best_choice = -INFINITY;
            float best_score = 0.0f;
            uint best_e = 0xFFFFFFFFu;
            for (uint r = 0u; r < chunks; r++) {
                const uint e = lane + r * 32u;
                if (e < p.num_experts && ((s_picked[sg_id][r] >> lane) & 1u) == 0u) {
                    const uint g_idx = use_groups ? (e / group_size) : 0u;
                    if (!use_groups || ((group_allow_mask >> g_idx) & 1u) != 0u) {
                        const float raw = G26_LOGITS_F16
                            ? float(((const device half *)logits_raw)[row_base + e])
                            : ((const device float *)logits_raw)[row_base + e];
                        float sc = 0.0f;
                        float ch = -INFINITY;
                        if (isfinite(raw) && raw > -1.0e29f) {
                            sc = g26_route_score(raw, p.score_mode);
                            const float b = (p.has_bias != 0u)
                                ? g26_hybrid_load_scalar_f32(bias_raw, e, p.bias_dtype)
                                : 0.0f;
                            ch = sc + b;
                        }
                        if (ch > best_choice || (ch == best_choice && e < best_e)) {
                            best_choice = ch;
                            best_score = sc;
                            best_e = e;
                        }
                    }
                }
            }
            for (ushort offset = 16u; offset > 0u; offset >>= 1u) {
                const float oc = simd_shuffle_down(best_choice, offset);
                const float os = simd_shuffle_down(best_score, offset);
                const uint oe = simd_shuffle_down(best_e, offset);
                if (oc > best_choice || (oc == best_choice && oe < best_e)) {
                    best_choice = oc;
                    best_score = os;
                    best_e = oe;
                }
            }
            const uint raw_win_e = simd_broadcast_first(best_e);
            const uint win_e = (raw_win_e < p.num_experts) ? raw_win_e : 0u;
            const float win_s = simd_broadcast_first(best_score);
            if (lane == 0u) {
                s_picked[sg_id][win_e >> 5] |= (1u << (win_e & 31u));
                s_sel_e[sg_id][k] = win_e;
                s_sel_w[sg_id][k] = win_s;
                w_sum += win_s;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lane == 0u) {
            const float renorm = (p.norm_topk_prob != 0u && top_k > 1u)
                ? (p.routed_scaling_factor / (w_sum + 1.0e-20f))
                : p.routed_scaling_factor;
            const ulong base = ulong(tok) * top_k;
            for (uint k = 0u; k < top_k; k++) {
                const uint e = s_sel_e[sg_id][k];
                scratch_topk_experts[base + k] = e;
                scratch_topk_weights[base + k] = s_sel_w[sg_id][k] * renorm;
                atomic_fetch_add_explicit(&s_counts[e], 1u, memory_order_relaxed);
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    if (p.is_bucketed == 0u) {
        uint u_base = 0u;
        uint tok_base = 0u;
        for (uint blk_base = 0u; blk_base < p.num_experts; blk_base += 256u) {
            const uint e = blk_base + tid;
            const uint cnt = (e < p.num_experts)
                ? atomic_load_explicit(&s_counts[e], memory_order_relaxed)
                : 0u;
            const uint is_active = (cnt > 0u) ? 1u : 0u;

            const uint sg_act_excl = simd_prefix_exclusive_sum(is_active);
            if (lane == 31u) {
                s_prefix_tmp[sg_id] = sg_act_excl + is_active;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg_id == 0u) {
                const uint v = (lane < 8u) ? s_prefix_tmp[lane] : 0u;
                const uint ex = simd_prefix_exclusive_sum(v);
                if (lane < 8u) {
                    s_prefix_tmp[lane] = ex;
                }
                if (lane == 7u) {
                    s_u_cnt = ex + v;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint u_idx = u_base + s_prefix_tmp[sg_id] + sg_act_excl;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const uint sg_cnt_excl = simd_prefix_exclusive_sum(cnt);
            if (lane == 31u) {
                s_prefix_tmp[sg_id] = sg_cnt_excl + cnt;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg_id == 0u) {
                const uint v = (lane < 8u) ? s_prefix_tmp[lane] : 0u;
                const uint ex = simd_prefix_exclusive_sum(v);
                if (lane < 8u) {
                    s_prefix_tmp[lane] = ex;
                }
                if (lane == 7u) {
                    s_a_cnt = ex + v;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint tok_off = tok_base + s_prefix_tmp[sg_id] + sg_cnt_excl;

            if (e < p.num_experts) {
                atomic_store_explicit(&s_cursors[e], tok_off, memory_order_relaxed);
            }
            if (is_active != 0u) {
                unique_experts[u_idx] = e;
                expert_token_offsets[u_idx] = tok_off;
            }
            u_base += s_u_cnt;
            tok_base += s_a_cnt;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0u) {
            const uint total_assign = p.m * top_k;
            expert_token_offsets[u_base] = total_assign;
            sched_hdr->num_unique_experts = u_base;
            sched_hdr->total_assignments = total_assign;
            sched_hdr->m = p.m;
            sched_hdr->top_k = top_k;
            g26_write_schedule_indirect((device uchar *)sched_hdr,
                                        u_base,
                                        p.gate_up_groups_x_q,
                                        p.gate_up_groups_x_f,
                                        p.down_groups_x,
                                        p.down_groups_x_q8);
        }
    } else {
        uint running_u_base = 0u;
        uint running_a_base = 0u;
        for (uint s = 0u; s < NUM_SCHEMES; s++) {
            const ulong u_base_off = ulong(s) * p.bucket_unique_stride;
            const ulong o_base_off = ulong(s) * p.bucket_offsets_stride;
            uint u_cnt_s = 0u;
            uint a_cnt_s = 0u;

            for (uint blk_base = 0u; blk_base < p.num_experts; blk_base += 256u) {
                const uint e = blk_base + tid;
                const uint cnt = (e < p.num_experts)
                    ? atomic_load_explicit(&s_counts[e], memory_order_relaxed)
                    : 0u;
                const uint my_scheme = (e < p.num_experts)
                    ? min(uint(expert_schemes[e]), NUM_SCHEMES - 1u)
                    : NUM_SCHEMES;
                const uint in_s = (cnt > 0u && my_scheme == s) ? 1u : 0u;
                const uint cnt_s = (my_scheme == s) ? cnt : 0u;

                const uint sg_act_excl = simd_prefix_exclusive_sum(in_s);
                if (lane == 31u) {
                    s_prefix_tmp[sg_id] = sg_act_excl + in_s;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (sg_id == 0u) {
                    const uint v = (lane < 8u) ? s_prefix_tmp[lane] : 0u;
                    const uint ex = simd_prefix_exclusive_sum(v);
                    if (lane < 8u) {
                        s_prefix_tmp[lane] = ex;
                    }
                    if (lane == 7u) {
                        s_u_cnt = ex + v;
                    }
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
                    if (lane < 8u) {
                        s_prefix_tmp[lane] = ex;
                    }
                    if (lane == 7u) {
                        s_a_cnt = ex + v;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                const uint a_in_blk = s_prefix_tmp[sg_id] + sg_cnt_excl;
                const uint blk_a_cnt = s_a_cnt;

                if (in_s != 0u) {
                    const uint slot_off = running_a_base + a_cnt_s + a_in_blk;
                    atomic_store_explicit(&s_cursors[e], slot_off, memory_order_relaxed);
                    bucket_unique_experts[u_base_off + u_cnt_s + u_in_blk] = e;
                    bucket_expert_token_offsets[o_base_off + u_cnt_s + u_in_blk] = slot_off;
                    unique_experts[running_u_base + u_cnt_s + u_in_blk] = e;
                    expert_token_offsets[running_u_base + u_cnt_s + u_in_blk] = slot_off;
                }
                u_cnt_s += blk_u_cnt;
                a_cnt_s += blk_a_cnt;
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (tid == 0u) {
                bucket_expert_token_offsets[o_base_off + u_cnt_s] = running_a_base + a_cnt_s;
                device uchar *slot_ptr = bucket_headers + ulong(s) * 64ul;
                device G26MoeScheduleHeader *bhdr = (device G26MoeScheduleHeader *)slot_ptr;
                bhdr->num_unique_experts = u_cnt_s;
                bhdr->total_assignments = a_cnt_s;
                bhdr->m = p.m;
                bhdr->top_k = top_k;
                g26_write_schedule_indirect(slot_ptr,
                                            u_cnt_s,
                                            p.gate_up_groups_x_q,
                                            p.gate_up_groups_x_f,
                                            p.down_groups_x,
                                            p.down_groups_x_q8);
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
            sched_hdr->top_k = top_k;
            g26_write_schedule_indirect((device uchar *)sched_hdr,
                                        running_u_base,
                                        p.gate_up_groups_x_q,
                                        p.gate_up_groups_x_f,
                                        p.down_groups_x,
                                        p.down_groups_x_q8);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint total_assign = p.m * top_k;
    for (uint idx = tid; idx < total_assign; idx += 256u) {
        const uint tok = idx / top_k;
        const uint k_idx = idx - tok * top_k;
        const uint e = scratch_topk_experts[idx];
        const float w = scratch_topk_weights[idx];
        const uint dst = atomic_fetch_add_explicit(&s_cursors[e], 1u, memory_order_relaxed);
        token_indices[dst] = tok;
        routing_weights[dst] = w;
        slot_k_indices[dst] = k_idx;
    }
}

// ---------------------------------------------------------------------------
// G26 DFlash Target Hidden Tap Ring Capture (`[cap, num_taps, H]` f16 ring).
// Copies either contiguous `[num_rows, H]` rows at one `tap_idx` (prefill/decode)
// or accepted tree nodes across all `num_taps` from `[num_taps, src_stride_rows, H]`
// into the circular `GpuContextHistory` ring with zero CPU-GPU syncs.
// ---------------------------------------------------------------------------

struct G26TapRingCaptureParams {
    uint num_rows;
    uint hidden_size;
    uint base_pos;
    uint ring_cap;
    uint num_taps;
    uint tap_idx;         // used when mode == 0 (single tap from layer output)
    uint src_stride_rows; // used when mode == 1 (tree commit across all taps)
    uint mode;            // 0 = single tap [num_rows, H], 1 = gather inline_nodes[0..num_rows) across all taps
    ushort inline_nodes[64];
};

static_assert(sizeof(G26TapRingCaptureParams) == 160, "G26TapRingCaptureParams must be 160 bytes");

// Grid: (num_rows, (mode == 1 ? num_taps : 1), 1) threadgroups, 256 threads.
kernel void ami_g26_capture_tap_ring(const device half *src [[buffer(0)]],
                                     device half *ring [[buffer(1)]],
                                     constant G26TapRingCaptureParams &p [[buffer(2)]],
                                     uint2 tg_pos [[threadgroup_position_in_grid]],
                                     uint tid [[thread_index_in_threadgroup]]) {
    const uint ri = tg_pos.x;
    const uint ti = (p.mode != 0u) ? tg_pos.y : p.tap_idx;
    if (ri >= p.num_rows || ti >= p.num_taps || p.ring_cap == 0u) {
        return;
    }

    const uint dst_slot = (p.base_pos + ri) % p.ring_cap;
    const ulong dst_off = (ulong(dst_slot) * p.num_taps + ti) * p.hidden_size;
    ulong src_off;
    if (p.mode != 0u) {
        const uint node = uint(p.inline_nodes[ri]);
        src_off = (ulong(ti) * p.src_stride_rows + node) * p.hidden_size;
    } else {
        src_off = ulong(ri) * p.hidden_size;
    }

    const device half *s = src + src_off;
    device half *d = ring + dst_off;
    if ((p.hidden_size & 3u) == 0u) {
        const uint vec_len = p.hidden_size >> 2;
        const device half4 *s4 = (const device half4 *)s;
        device half4 *d4 = (device half4 *)d;
        for (uint i = tid; i < vec_len; i += 256u) {
            d4[i] = s4[i];
        }
    } else {
        for (uint i = tid; i < p.hidden_size; i += 256u) {
            d[i] = s[i];
        }
    }
}

// ===========================================================================
// Section G: M9.2 (`G27`) — GLM-5.2 / GLM-5.3 (`glm_moe_dsa`) &
// GLM-5.3-Flash (`glm5_next`) Metal 4 Operators
//   1. Single-kernel register-fused 20-step `hc_mult = 4` mHC Sinkhorn
//      (`ami_g27_mhc_pre_sinkhorn`, `ami_g27_mhc_post_mix`)
//   2. Kimi Delta Attention (`KDA`): separate `{q,k,v}_conv1d + SiLU`,
//      per-channel bounded/softplus forget gate, decode / prefill / tree_verify /
//      commit_tree / sigmoid-gated RMSNorm (`ami_kda_*`)
//   3. `GlmMoeDsaIndexer` + `NoPE MLA` (`qk_rope = 0`, `index_kpool = 4`):
//      interleaved RoPE, LayerNorm with bias, FP16 MLA & indexer KV write,
//      k-pool key compression, DSA parallel scoring + exact radix Top-K +
//      tail selection, and sparse absorbed MLA attention (`ami_g27_*`)
// ===========================================================================

static inline float g27_load_scalar_f32(const device uchar *buf, ulong idx, uint dtype) {
    if (dtype == 2u) {
        return ((const device float *)buf)[idx];
    }
    if (dtype == 1u) {
        return float(((const device bfloat *)buf)[idx]);
    }
    return float(((const device half *)buf)[idx]);
}

static inline float4 g27_softmax4_plus_eps(float4 z, float eps) {
    const float m = max(max(z.x, z.y), max(z.z, z.w));
    const float4 e = exp(z - m);
    const float inv_s = 1.0f / (e.x + e.y + e.z + e.w);
    return e * inv_s + eps;
}

static inline float g27_row_norm4(float4 v, float eps) {
    return 1.0f / (v.x + v.y + v.z + v.w + eps);
}

// ---------------------------------------------------------------------------
// G.1: Single-Kernel Register-Fused 20-Step `hc_mult = 4` mHC Sinkhorn
// Matches `Glm5NextTextHyperConnection` verbatim:
//   flat = UnweightedRMSNorm(streams.view(M, 4*H), rms_eps)
//   raw[24] = flat @ fn^T
//   pre  = sigmoid(raw[0..4] * scale[0] + base[0..4]) + hc_eps
//   post = 2 * sigmoid(raw[4..8] * scale[1] + base[4..8])
//   comb = softmax(raw[8..24].view(4,4) * scale[2] + base[8..24].view(4,4), dim=-1) + hc_eps
//   comb /= (col_sum(comb) + hc_eps)
//   (sinkhorn_iters - 1) times:
//     comb /= (row_sum(comb) + hc_eps)
//     comb /= (col_sum(comb) + hc_eps)
//   collapsed = sum_{s=0..3} pre[s] * streams[s]
//   optional sublayer RMSNorm(collapsed, norm_w, rms_eps) -> branch_out
// ---------------------------------------------------------------------------

struct G27MhcSinkhornParams {
    uint m;
    uint hidden_size;
    uint sinkhorn_iters;
    uint flags;      // bit0=has_norm_w
    uint fn_dtype;   // 0=f16, 1=bf16, 2=f32
    uint norm_dtype; // 0=f16, 1=bf16, 2=f32
    float rms_eps;
    float hc_eps;
};

static_assert(sizeof(G27MhcSinkhornParams) == 32, "G27MhcSinkhornParams must be 32 bytes");

// Grid: (M, 1, 1) threadgroups, 256 threads (8 simdgroups).
kernel void ami_g27_mhc_pre_sinkhorn(const device uchar *streams [[buffer(0)]],
                                     const device uchar *fn_w [[buffer(1)]],
                                     const device float *base_w [[buffer(2)]],
                                     const device float *scale_w [[buffer(3)]],
                                     const device uchar *norm_w [[buffer(4)]],
                                     device uchar *branch_out [[buffer(5)]],
                                     device float *coeffs_out [[buffer(6)]],
                                     constant G27MhcSinkhornParams &p [[buffer(7)]],
                                     uint tg_x [[threadgroup_position_in_grid]],
                                     uint tid [[thread_index_in_threadgroup]],
                                     uint sg_id [[simdgroup_index_in_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]]) {
    const uint tok = tg_x;
    if (tok >= p.m || p.hidden_size == 0u) {
        return;
    }

    const uint total_dim = 4u * p.hidden_size;
    const ulong tok_stream_base = ulong(tok) * total_dim;

    threadgroup float tg_raw[24];
    threadgroup float tg_pre[4];
    threadgroup float tg_reduce[8];
    threadgroup float tg_inv_rms[1];

    // 8 simdgroups x 3 rows/simdgroup = 24 rows of `fn [24, 4*H]`.
    // Exploits linearity: `fn @ (streams * inv_rms) == (fn @ streams) * inv_rms`
    // so `streams` and `fn_w` are streamed in a single coalesced pass!
    const uint r0 = sg_id * 3u + 0u;
    const uint r1 = sg_id * 3u + 1u;
    const uint r2 = sg_id * 3u + 2u;
    const ulong fn_off0 = ulong(r0) * total_dim;
    const ulong fn_off1 = ulong(r1) * total_dim;
    const ulong fn_off2 = ulong(r2) * total_dim;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float sq_sum = 0.0f;

    for (uint j = lane; j < total_dim; j += 32u) {
        const float sv = hybrid_load_f16_or_bf16(streams, tok_stream_base + j, GR_STREAM_BF16);
        if (sg_id == 0u) {
            sq_sum = fma(sv, sv, sq_sum);
        }
        const float w0 = g27_load_scalar_f32(fn_w, fn_off0 + j, p.fn_dtype);
        const float w1 = g27_load_scalar_f32(fn_w, fn_off1 + j, p.fn_dtype);
        const float w2 = g27_load_scalar_f32(fn_w, fn_off2 + j, p.fn_dtype);
        acc0 = fma(sv, w0, acc0);
        acc1 = fma(sv, w1, acc1);
        acc2 = fma(sv, w2, acc2);
    }

    const float sum0 = simd_sum(acc0);
    const float sum1 = simd_sum(acc1);
    const float sum2 = simd_sum(acc2);
    const float total_sq = (sg_id == 0u) ? simd_sum(sq_sum) : 0.0f;
    if (lane == 0u) {
        tg_raw[r0] = sum0;
        tg_raw[r1] = sum1;
        tg_raw[r2] = sum2;
        if (sg_id == 0u) {
            tg_inv_rms[0] = rsqrt(total_sq / float(total_dim) + p.rms_eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Single-thread register-fused 20-step Sinkhorn-Knopp in `float4 c0..c3` (0 spills).
    if (tid == 0u) {
        const float inv_rms = tg_inv_rms[0];
        const float eps = p.hc_eps;
        const float s_pre = scale_w[0] * inv_rms;
        const float s_post = scale_w[1] * inv_rms;
        const float s_comb = scale_w[2] * inv_rms;

        tg_pre[0] = hybrid_sigmoid(fma(tg_raw[0], s_pre, base_w[0])) + eps;
        tg_pre[1] = hybrid_sigmoid(fma(tg_raw[1], s_pre, base_w[1])) + eps;
        tg_pre[2] = hybrid_sigmoid(fma(tg_raw[2], s_pre, base_w[2])) + eps;
        tg_pre[3] = hybrid_sigmoid(fma(tg_raw[3], s_pre, base_w[3])) + eps;

        const float p0 = 2.0f * hybrid_sigmoid(fma(tg_raw[4], s_post, base_w[4]));
        const float p1 = 2.0f * hybrid_sigmoid(fma(tg_raw[5], s_post, base_w[5]));
        const float p2 = 2.0f * hybrid_sigmoid(fma(tg_raw[6], s_post, base_w[6]));
        const float p3 = 2.0f * hybrid_sigmoid(fma(tg_raw[7], s_post, base_w[7]));

        float4 c0 = g27_softmax4_plus_eps(float4(
            fma(tg_raw[8],  s_comb, base_w[8]),
            fma(tg_raw[9],  s_comb, base_w[9]),
            fma(tg_raw[10], s_comb, base_w[10]),
            fma(tg_raw[11], s_comb, base_w[11])), eps);
        float4 c1 = g27_softmax4_plus_eps(float4(
            fma(tg_raw[12], s_comb, base_w[12]),
            fma(tg_raw[13], s_comb, base_w[13]),
            fma(tg_raw[14], s_comb, base_w[14]),
            fma(tg_raw[15], s_comb, base_w[15])), eps);
        float4 c2 = g27_softmax4_plus_eps(float4(
            fma(tg_raw[16], s_comb, base_w[16]),
            fma(tg_raw[17], s_comb, base_w[17]),
            fma(tg_raw[18], s_comb, base_w[18]),
            fma(tg_raw[19], s_comb, base_w[19])), eps);
        float4 c3 = g27_softmax4_plus_eps(float4(
            fma(tg_raw[20], s_comb, base_w[20]),
            fma(tg_raw[21], s_comb, base_w[21]),
            fma(tg_raw[22], s_comb, base_w[22]),
            fma(tg_raw[23], s_comb, base_w[23])), eps);

        if (p.sinkhorn_iters > 0u) {
            const float4 inv_c0 = 1.0f / (c0 + c1 + c2 + c3 + eps);
            c0 *= inv_c0;
            c1 *= inv_c0;
            c2 *= inv_c0;
            c3 *= inv_c0;
            for (uint it = 1u; it < p.sinkhorn_iters; ++it) {
                c0 *= g27_row_norm4(c0, eps);
                c1 *= g27_row_norm4(c1, eps);
                c2 *= g27_row_norm4(c2, eps);
                c3 *= g27_row_norm4(c3, eps);
                const float4 inv_c = 1.0f / (c0 + c1 + c2 + c3 + eps);
                c0 *= inv_c;
                c1 *= inv_c;
                c2 *= inv_c;
                c3 *= inv_c;
            }
        }

        device float *dst_c = coeffs_out + ulong(tok) * 20u;
        dst_c[0] = p0; dst_c[1] = p1; dst_c[2] = p2; dst_c[3] = p3;
        dst_c[4] = c0.x; dst_c[5] = c0.y; dst_c[6] = c0.z; dst_c[7] = c0.w;
        dst_c[8] = c1.x; dst_c[9] = c1.y; dst_c[10] = c1.z; dst_c[11] = c1.w;
        dst_c[12] = c2.x; dst_c[13] = c2.y; dst_c[14] = c2.z; dst_c[15] = c2.w;
        dst_c[16] = c3.x; dst_c[17] = c3.y; dst_c[18] = c3.z; dst_c[19] = c3.w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float pre0 = tg_pre[0];
    const float pre1 = tg_pre[1];
    const float pre2 = tg_pre[2];
    const float pre3 = tg_pre[3];
    const bool has_norm_w = ((p.flags & 1u) != 0u);
    const ulong h = ulong(p.hidden_size);
    const ulong tok_out_base = ulong(tok) * h;

    float inv_rms_sub = 1.0f;
    if (has_norm_w) {
        float sq_partial = 0.0f;
        for (uint d = tid; d < p.hidden_size; d += 256u) {
            const float s0 = hybrid_load_f16_or_bf16(streams, tok_stream_base + 0ul * h + d, GR_STREAM_BF16);
            const float s1 = hybrid_load_f16_or_bf16(streams, tok_stream_base + 1ul * h + d, GR_STREAM_BF16);
            const float s2 = hybrid_load_f16_or_bf16(streams, tok_stream_base + 2ul * h + d, GR_STREAM_BF16);
            const float s3 = hybrid_load_f16_or_bf16(streams, tok_stream_base + 3ul * h + d, GR_STREAM_BF16);
            const float mix = fma(pre0, s0, fma(pre1, s1, fma(pre2, s2, pre3 * s3)));
            sq_partial = fma(mix, mix, sq_partial);
        }
        const float sg_sq = simd_sum(sq_partial);
        if (lane == 0u) {
            tg_reduce[sg_id] = sg_sq;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float sub_sq = ((tg_reduce[0] + tg_reduce[1]) + (tg_reduce[2] + tg_reduce[3])) +
                             ((tg_reduce[4] + tg_reduce[5]) + (tg_reduce[6] + tg_reduce[7]));
        inv_rms_sub = rsqrt(sub_sq / float(p.hidden_size) + p.rms_eps);
    }

    for (uint d = tid; d < p.hidden_size; d += 256u) {
        const float s0 = hybrid_load_f16_or_bf16(streams, tok_stream_base + 0ul * h + d, GR_STREAM_BF16);
        const float s1 = hybrid_load_f16_or_bf16(streams, tok_stream_base + 1ul * h + d, GR_STREAM_BF16);
        const float s2 = hybrid_load_f16_or_bf16(streams, tok_stream_base + 2ul * h + d, GR_STREAM_BF16);
        const float s3 = hybrid_load_f16_or_bf16(streams, tok_stream_base + 3ul * h + d, GR_STREAM_BF16);
        float mix = fma(pre0, s0, fma(pre1, s1, fma(pre2, s2, pre3 * s3)));
        if (has_norm_w) {
            const float nw = g27_load_scalar_f32(norm_w, d, p.norm_dtype);
            mix = (mix * inv_rms_sub) * nw;
        }
        hybrid_store_f16_or_bf16(branch_out, tok_out_base + d, mix, GR_ACT_BF16);
    }
}

// Single-kernel mHC post-mix for `hc_mult = 4`:
//   streams[tok, j, d] = post[j] * sub_out[tok, d] + sum_{i=0..3} comb[i, j] * streams[tok, i, d]
// Grid: (M, 1, 1) threadgroups, 256 threads.
kernel void ami_g27_mhc_post_mix(device uchar *streams [[buffer(0)]],
                                 const device uchar *sublayer_out [[buffer(1)]],
                                 const device float *coeffs [[buffer(2)]],
                                 constant G27MhcSinkhornParams &p [[buffer(3)]],
                                 uint tg_x [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]]) {
    const uint tok = tg_x;
    if (tok >= p.m || p.hidden_size == 0u) {
        return;
    }

    threadgroup float tg_c[20];
    if (tid < 20u) {
        tg_c[tid] = coeffs[ulong(tok) * 20u + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float4 post = float4(tg_c[0], tg_c[1], tg_c[2], tg_c[3]);
    const float4 c0 = float4(tg_c[4], tg_c[5], tg_c[6], tg_c[7]);
    const float4 c1 = float4(tg_c[8], tg_c[9], tg_c[10], tg_c[11]);
    const float4 c2 = float4(tg_c[12], tg_c[13], tg_c[14], tg_c[15]);
    const float4 c3 = float4(tg_c[16], tg_c[17], tg_c[18], tg_c[19]);

    const ulong h = ulong(p.hidden_size);
    const ulong tok_stream_base = ulong(tok) * 4ul * h;
    const ulong tok_in_base = ulong(tok) * h;

    for (uint d = tid; d < p.hidden_size; d += 256u) {
        const float y = hybrid_load_f16_or_bf16(sublayer_out, tok_in_base + d, GR_ACT_BF16);
        const float s0 = hybrid_load_f16_or_bf16(streams, tok_stream_base + 0ul * h + d, GR_STREAM_BF16);
        const float s1 = hybrid_load_f16_or_bf16(streams, tok_stream_base + 1ul * h + d, GR_STREAM_BF16);
        const float s2 = hybrid_load_f16_or_bf16(streams, tok_stream_base + 2ul * h + d, GR_STREAM_BF16);
        const float s3 = hybrid_load_f16_or_bf16(streams, tok_stream_base + 3ul * h + d, GR_STREAM_BF16);

        const float4 out_s = fma(post, y, fma(c0, s0, fma(c1, s1, fma(c2, s2, c3 * s3))));
        hybrid_store_f16_or_bf16(streams, tok_stream_base + 0ul * h + d, out_s.x, GR_STREAM_BF16);
        hybrid_store_f16_or_bf16(streams, tok_stream_base + 1ul * h + d, out_s.y, GR_STREAM_BF16);
        hybrid_store_f16_or_bf16(streams, tok_stream_base + 2ul * h + d, out_s.z, GR_STREAM_BF16);
        hybrid_store_f16_or_bf16(streams, tok_stream_base + 3ul * h + d, out_s.w, GR_STREAM_BF16);
    }
}

// ---------------------------------------------------------------------------
// G.2: Kimi Delta Attention (`KDA`, `GLM-5.3-Flash` 34L `linear_attention`)
// Reusable for M9.3 `Ling-3.1` via configurable `lower_bound` (`< 0`: safe
// bounded sigmoid gate; `== 0`: softplus gate).
// ---------------------------------------------------------------------------

struct KdaConv1dParams {
    uint m;
    uint qkv_dim;
    uint conv_k;
    uint write_state;
    uint is_tree;
    uint w_dtype; // 0=f16, 1=bf16, 2=f32
    uint _pad0;
    uint _pad1;
};

struct KdaComputeGatesParams {
    uint m;
    uint num_heads;
    uint head_dim;
    float lower_bound; // < 0: lower_bound * sigmoid(exp(A_log) * (f + dt_bias)); == 0: -exp(A_log) * softplus(...)
    uint a_log_dtype;  // 0=f16, 1=bf16, 2=f32
    uint dt_bias_dtype;
    uint _pad0;
    uint _pad1;
};

struct KdaDecodeParams {
    uint num_heads;
    uint head_k_dim;
    uint head_v_dim;
    float q_scale;
    float l2_eps;
    float rms_eps;
    uint apply_l2_norm;
    uint apply_out_norm;
    uint apply_out_gate;
    uint per_head_norm;
    uint beta_apply_sigmoid;
    uint _pad0;
};

struct KdaPrefillParams {
    uint seq_len;
    uint num_heads;
    uint head_k_dim;
    uint head_v_dim;
    float q_scale;
    float l2_eps;
    uint apply_l2_norm;
    uint beta_apply_sigmoid;
    uint write_state_only;
    uint out_start_token;
    uint _pad0;
    uint _pad1;
};

struct KdaTreeVerifyParams {
    uint num_nodes;
    uint num_heads;
    uint head_k_dim;
    uint head_v_dim;
    float q_scale;
    float l2_eps;
    uint apply_l2_norm;
    uint beta_apply_sigmoid;
    uint scratch_leaf_node;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

struct KdaCommitTreeParams {
    uint num_heads;
    uint head_k_dim;
    uint head_v_dim;
    uint accepted_count;
    uint use_node_buffer;
    float l2_eps;
    uint apply_l2_norm;
    uint beta_apply_sigmoid;
    ushort inline_nodes[64];
};

static_assert(sizeof(KdaConv1dParams) == 32, "KdaConv1dParams must be 32 bytes");
static_assert(sizeof(KdaComputeGatesParams) == 32, "KdaComputeGatesParams must be 32 bytes");
static_assert(sizeof(KdaDecodeParams) == 48, "KdaDecodeParams must be 48 bytes");
static_assert(sizeof(KdaPrefillParams) == 48, "KdaPrefillParams must be 48 bytes");
static_assert(sizeof(KdaTreeVerifyParams) == 48, "KdaTreeVerifyParams must be 48 bytes");
static_assert(sizeof(KdaCommitTreeParams) == 160, "KdaCommitTreeParams must be 160 bytes");

// Separate `{q,k,v}` causal depthwise Conv1D + SiLU (`conv_k in [1..8]`, `3 * qkv_dim` total channels).
// State layout matches `ami_gdn_conv1d_commit`: `[conv_k - 1, 3 * qkv_dim]` f16.
kernel void ami_kda_conv1d_silu(const device half *q_in [[buffer(0)]],
                                const device half *k_in [[buffer(1)]],
                                const device half *v_in [[buffer(2)]],
                                const device uchar *q_conv_w [[buffer(3)]],
                                const device uchar *k_conv_w [[buffer(4)]],
                                const device uchar *v_conv_w [[buffer(5)]],
                                const device half *conv_state_in [[buffer(6)]],
                                device half *conv_state_out [[buffer(7)]],
                                const device ulong *ancestor_masks [[buffer(8)]],
                                device half *q_out [[buffer(9)]],
                                device half *k_out [[buffer(10)]],
                                device half *v_out [[buffer(11)]],
                                constant KdaConv1dParams &p [[buffer(12)]],
                                uint gid [[thread_position_in_grid]]) {
    const uint conv_dim = 3u * p.qkv_dim;
    if (gid >= conv_dim) {
        return;
    }

    const uint hist = (p.conv_k > 0u) ? (p.conv_k - 1u) : 0u;
    const uint part = (gid < p.qkv_dim) ? 0u : ((gid < 2u * p.qkv_dim) ? 1u : 2u);
    const uint local_c = gid - part * p.qkv_dim;
    const device uchar *w_buf = (part == 0u) ? q_conv_w : ((part == 1u) ? k_conv_w : v_conv_w);
    const device half *in_buf = (part == 0u) ? q_in : ((part == 1u) ? k_in : v_in);
    device half *out_buf = (part == 0u) ? q_out : ((part == 1u) ? k_out : v_out);
    const ulong w_base = ulong(local_c) * p.conv_k;

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
                    xv = float(in_buf[ulong(uint(rel)) * p.qkv_dim + local_c]);
                }
                acc = fma(xv, g27_load_scalar_f32(w_buf, w_base + tap, p.w_dtype), acc);
            }
            out_buf[ulong(tok) * p.qkv_dim + local_c] = half(hybrid_fast_silu(acc));
        }

        if (p.write_state != 0u && hist > 0u) {
            for (uint s = 0u; s < hist; ++s) {
                const int rel = int(p.m) - int(hist - s);
                float xv;
                if (rel < 0) {
                    const uint old_s = uint(int(hist) + rel);
                    xv = float(conv_state_in[ulong(old_s) * conv_dim + gid]);
                } else {
                    xv = float(in_buf[ulong(uint(rel)) * p.qkv_dim + local_c]);
                }
                conv_state_out[ulong(s) * conv_dim + gid] = half(xv);
            }
        }
    } else {
        for (uint node = 0u; node < p.m; ++node) {
            const ulong mask = ancestor_masks[node];
            float acc = 0.0f;
            for (uint tap = 0u; tap < p.conv_k; ++tap) {
                float xv;
                if (tap == hist) {
                    xv = float(in_buf[ulong(node) * p.qkv_dim + local_c]);
                } else {
                    const uint back = hist - tap; // 1 = parent, 2 = grandparent, ...
                    uint found_cnt = 0u;
                    int anc_node = -1;
                    if (node > 0u) {
                        for (int cand = int(node) - 1; cand >= 0; --cand) {
                            if (((mask >> uint(cand)) & 1ul) != 0ul) {
                                found_cnt++;
                                if (found_cnt == back) {
                                    anc_node = cand;
                                    break;
                                }
                            }
                        }
                    }
                    if (anc_node >= 0) {
                        xv = float(in_buf[ulong(uint(anc_node)) * p.qkv_dim + local_c]);
                    } else {
                        const uint past_back = back - found_cnt;
                        const uint s_idx = hist - past_back;
                        xv = float(conv_state_in[ulong(s_idx) * conv_dim + gid]);
                    }
                }
                acc = fma(xv, g27_load_scalar_f32(w_buf, w_base + tap, p.w_dtype), acc);
            }
            out_buf[ulong(node) * p.qkv_dim + local_c] = half(hybrid_fast_silu(acc));
        }
    }
}

// Per-key-element KDA forget gate `g [M, num_heads, head_dim]` + `beta = sigmoid(b_raw) [M, num_heads]`.
kernel void ami_kda_compute_gates(const device half *f_raw [[buffer(0)]],
                                  const device half *b_raw [[buffer(1)]],
                                  const device uchar *a_log [[buffer(2)]],
                                  const device uchar *dt_bias [[buffer(3)]],
                                  device float *g_out [[buffer(4)]],
                                  device float *beta_out [[buffer(5)]],
                                  constant KdaComputeGatesParams &p [[buffer(6)]],
                                  uint gid [[thread_position_in_grid]]) {
    const uint qkv_dim = p.num_heads * p.head_dim;
    const uint total = p.m * qkv_dim;
    if (gid >= total) {
        return;
    }

    const uint tok = gid / qkv_dim;
    const uint rem = gid - tok * qkv_dim;
    const uint h = rem / p.head_dim;
    const uint d = rem - h * p.head_dim;

    const float decay_rate = exp(g27_load_scalar_f32(a_log, h, p.a_log_dtype));
    const float g_in = float(f_raw[gid]) + g27_load_scalar_f32(dt_bias, rem, p.dt_bias_dtype);

    float g_val;
    if (p.lower_bound < 0.0f) {
        g_val = p.lower_bound * hybrid_sigmoid(decay_rate * g_in);
    } else {
        const float sp = (g_in > 20.0f) ? g_in : log(1.0f + exp(g_in));
        g_val = -decay_rate * sp;
    }
    g_out[gid] = g_val;

    if (d == 0u) {
        const uint bh_idx = tok * p.num_heads + h;
        beta_out[bh_idx] = hybrid_sigmoid(float(b_raw[bh_idx]));
    }
}

// Single-step KDA decode (`M = 1`) with per-dk decay `exp(g[h, dk])`, FP32 L2Norm,
// and fused optional RMSNorm + sigmoid output gate (`Glm5NextTextRMSNormGated`).
// Grid: (num_heads, 1, 1) threadgroups, 256 threads (8 simdgroups).
kernel void ami_kda_decode(const device half *q [[buffer(0)]],
                           const device half *k [[buffer(1)]],
                           const device half *v [[buffer(2)]],
                           const device float *g [[buffer(3)]],
                           const device float *beta [[buffer(4)]],
                           const device half *z [[buffer(5)]],
                           const device uchar *norm_w [[buffer(6)]],
                           const device float *state_in [[buffer(7)]],
                           device float *state_out [[buffer(8)]],
                           device half *out [[buffer(9)]],
                           constant KdaDecodeParams &p [[buffer(10)]],
                           uint tg_x [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint sg_id [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    const uint h = tg_x;
    if (h >= p.num_heads) {
        return;
    }

    const uint d_k = (GDN_HEAD_K_DIM > 0u) ? GDN_HEAD_K_DIM : p.head_k_dim;
    const uint d_v = p.head_v_dim;

    threadgroup float tg_q[256];
    threadgroup float tg_k[256];
    threadgroup float tg_alpha[256];
    threadgroup float tg_o[256];
    threadgroup float tg_rms_val[1];

    if (sg_id == 0u) {
        const device half *q_src = q + ulong(h) * d_k;
        const device half *k_src = k + ulong(h) * d_k;
        const device float *g_src = g + ulong(h) * d_k;
        float q_sq = 0.0f;
        float k_sq = 0.0f;
        for (uint dk = lane; dk < d_k; dk += 32u) {
            const float qv = float(q_src[dk]);
            const float kv = float(k_src[dk]);
            tg_q[dk] = qv;
            tg_k[dk] = kv;
            tg_alpha[dk] = exp(g_src[dk]);
            q_sq = fma(qv, qv, q_sq);
            k_sq = fma(kv, kv, k_sq);
        }
        const float q_sum = simd_sum(q_sq);
        const float k_sum = simd_sum(k_sq);
        const float q_fac = (p.apply_l2_norm != 0u) ? (p.q_scale / sqrt(q_sum + p.l2_eps)) : p.q_scale;
        const float k_fac = (p.apply_l2_norm != 0u) ? (1.0f / sqrt(k_sum + p.l2_eps)) : 1.0f;
        for (uint dk = lane; dk < d_k; dk += 32u) {
            tg_q[dk] *= q_fac;
            tg_k[dk] *= k_fac;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float beta_val = (p.beta_apply_sigmoid != 0u) ? hybrid_sigmoid(beta[h]) : beta[h];
    const ulong head_state_base = ulong(h) * d_k * d_v;
    const ulong v_head_base = ulong(h) * d_v;

    for (uint dv = sg_id; dv < d_v; dv += 8u) {
        float pred_acc = 0.0f;
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        float s4 = 0.0f, s5 = 0.0f, s6 = 0.0f, s7 = 0.0f;

        if (GDN_HEAD_K_DIM == 64u) {
            const uint dk0 = lane;
            const uint dk1 = lane + 32u;
            s0 = state_in[head_state_base + ulong(dk0) * d_v + dv] * tg_alpha[dk0];
            s1 = state_in[head_state_base + ulong(dk1) * d_v + dv] * tg_alpha[dk1];
            pred_acc = fma(s0, tg_k[dk0], s1 * tg_k[dk1]);
        } else if (GDN_HEAD_K_DIM == 128u) {
            const uint dk0 = lane;
            const uint dk1 = lane + 32u;
            const uint dk2 = lane + 64u;
            const uint dk3 = lane + 96u;
            s0 = state_in[head_state_base + ulong(dk0) * d_v + dv] * tg_alpha[dk0];
            s1 = state_in[head_state_base + ulong(dk1) * d_v + dv] * tg_alpha[dk1];
            s2 = state_in[head_state_base + ulong(dk2) * d_v + dv] * tg_alpha[dk2];
            s3 = state_in[head_state_base + ulong(dk3) * d_v + dv] * tg_alpha[dk3];
            pred_acc = fma(s0, tg_k[dk0], fma(s1, tg_k[dk1], fma(s2, tg_k[dk2], s3 * tg_k[dk3])));
        } else {
            if (lane < d_k)        { s0 = state_in[head_state_base + ulong(lane) * d_v + dv] * tg_alpha[lane]; pred_acc = fma(s0, tg_k[lane], pred_acc); }
            if (lane + 32u < d_k)  { s1 = state_in[head_state_base + ulong(lane + 32u) * d_v + dv] * tg_alpha[lane + 32u]; pred_acc = fma(s1, tg_k[lane + 32u], pred_acc); }
            if (lane + 64u < d_k)  { s2 = state_in[head_state_base + ulong(lane + 64u) * d_v + dv] * tg_alpha[lane + 64u]; pred_acc = fma(s2, tg_k[lane + 64u], pred_acc); }
            if (lane + 96u < d_k)  { s3 = state_in[head_state_base + ulong(lane + 96u) * d_v + dv] * tg_alpha[lane + 96u]; pred_acc = fma(s3, tg_k[lane + 96u], pred_acc); }
            if (lane + 128u < d_k) { s4 = state_in[head_state_base + ulong(lane + 128u) * d_v + dv] * tg_alpha[lane + 128u]; pred_acc = fma(s4, tg_k[lane + 128u], pred_acc); }
            if (lane + 160u < d_k) { s5 = state_in[head_state_base + ulong(lane + 160u) * d_v + dv] * tg_alpha[lane + 160u]; pred_acc = fma(s5, tg_k[lane + 160u], pred_acc); }
            if (lane + 192u < d_k) { s6 = state_in[head_state_base + ulong(lane + 192u) * d_v + dv] * tg_alpha[lane + 192u]; pred_acc = fma(s6, tg_k[lane + 192u], pred_acc); }
            if (lane + 224u < d_k) { s7 = state_in[head_state_base + ulong(lane + 224u) * d_v + dv] * tg_alpha[lane + 224u]; pred_acc = fma(s7, tg_k[lane + 224u], pred_acc); }
        }

        const float pred_v = simd_sum(pred_acc);
        const float delta = beta_val * (float(v[v_head_base + dv]) - pred_v);

        float o_acc = 0.0f;
        if (GDN_HEAD_K_DIM == 64u) {
            const uint dk0 = lane;
            const uint dk1 = lane + 32u;
            s0 = fma(tg_k[dk0], delta, s0); state_out[head_state_base + ulong(dk0) * d_v + dv] = s0;
            s1 = fma(tg_k[dk1], delta, s1); state_out[head_state_base + ulong(dk1) * d_v + dv] = s1;
            o_acc = fma(s0, tg_q[dk0], s1 * tg_q[dk1]);
        } else if (GDN_HEAD_K_DIM == 128u) {
            const uint dk0 = lane;
            const uint dk1 = lane + 32u;
            const uint dk2 = lane + 64u;
            const uint dk3 = lane + 96u;
            s0 = fma(tg_k[dk0], delta, s0); state_out[head_state_base + ulong(dk0) * d_v + dv] = s0;
            s1 = fma(tg_k[dk1], delta, s1); state_out[head_state_base + ulong(dk1) * d_v + dv] = s1;
            s2 = fma(tg_k[dk2], delta, s2); state_out[head_state_base + ulong(dk2) * d_v + dv] = s2;
            s3 = fma(tg_k[dk3], delta, s3); state_out[head_state_base + ulong(dk3) * d_v + dv] = s3;
            o_acc = fma(s0, tg_q[dk0], fma(s1, tg_q[dk1], fma(s2, tg_q[dk2], s3 * tg_q[dk3])));
        } else {
            if (lane < d_k)        { s0 = fma(tg_k[lane], delta, s0); state_out[head_state_base + ulong(lane) * d_v + dv] = s0; o_acc = fma(s0, tg_q[lane], o_acc); }
            if (lane + 32u < d_k)  { s1 = fma(tg_k[lane + 32u], delta, s1); state_out[head_state_base + ulong(lane + 32u) * d_v + dv] = s1; o_acc = fma(s1, tg_q[lane + 32u], o_acc); }
            if (lane + 64u < d_k)  { s2 = fma(tg_k[lane + 64u], delta, s2); state_out[head_state_base + ulong(lane + 64u) * d_v + dv] = s2; o_acc = fma(s2, tg_q[lane + 64u], o_acc); }
            if (lane + 96u < d_k)  { s3 = fma(tg_k[lane + 96u], delta, s3); state_out[head_state_base + ulong(lane + 96u) * d_v + dv] = s3; o_acc = fma(s3, tg_q[lane + 96u], o_acc); }
            if (lane + 128u < d_k) { s4 = fma(tg_k[lane + 128u], delta, s4); state_out[head_state_base + ulong(lane + 128u) * d_v + dv] = s4; o_acc = fma(s4, tg_q[lane + 128u], o_acc); }
            if (lane + 160u < d_k) { s5 = fma(tg_k[lane + 160u], delta, s5); state_out[head_state_base + ulong(lane + 160u) * d_v + dv] = s5; o_acc = fma(s5, tg_q[lane + 160u], o_acc); }
            if (lane + 192u < d_k) { s6 = fma(tg_k[lane + 192u], delta, s6); state_out[head_state_base + ulong(lane + 192u) * d_v + dv] = s6; o_acc = fma(s6, tg_q[lane + 192u], o_acc); }
            if (lane + 224u < d_k) { s7 = fma(tg_k[lane + 224u], delta, s7); state_out[head_state_base + ulong(lane + 224u) * d_v + dv] = s7; o_acc = fma(s7, tg_q[lane + 224u], o_acc); }
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
                tg_rms_val[0] = rsqrt(total_sq / float(d_v) + p.rms_eps);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        inv_rms = tg_rms_val[0];
    }

    const uint nw_base = (p.per_head_norm != 0u) ? (h * d_v) : 0u;
    for (uint dv = tid; dv < d_v; dv += 256u) {
        float val = tg_o[dv];
        if (p.apply_out_norm != 0u) {
            val = (val * inv_rms) * hybrid_load_norm_w(norm_w, nw_base + dv, HYBRID_NORM_W_BF16);
        }
        if (p.apply_out_gate != 0u) {
            val *= hybrid_sigmoid(float(z[v_head_base + dv]));
        }
        out[v_head_base + dv] = half(val);
    }
}

// Chunked-prefill KDA (`seq_len >= 1`), register-resident state columns with 4-token SRAM chunks.
// Grid: (num_heads, ceil(head_v_dim / 8), 1) threadgroups, 256 threads.
kernel void ami_kda_prefill(const device half *q [[buffer(0)]],
                            const device half *k [[buffer(1)]],
                            const device half *v [[buffer(2)]],
                            const device float *g [[buffer(3)]],
                            const device float *beta [[buffer(4)]],
                            const device float *state_in [[buffer(5)]],
                            device float *state_out [[buffer(6)]],
                            device half *out [[buffer(7)]],
                            constant KdaPrefillParams &p [[buffer(8)]],
                            uint2 tg_pos [[threadgroup_position_in_grid]],
                            uint sg_id [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
    const uint h = tg_pos.x;
    const uint dv = tg_pos.y * 8u + sg_id;
    if (h >= p.num_heads) {
        return;
    }

    const uint d_k = (GDN_HEAD_K_DIM > 0u) ? GDN_HEAD_K_DIM : p.head_k_dim;
    const uint d_v = p.head_v_dim;
    const bool active_dv = (dv < d_v);
    const ulong head_state_base = ulong(h) * d_k * d_v;

    threadgroup float tg_q[4][256];
    threadgroup float tg_k[4][256];
    threadgroup float tg_alpha[4][256];
    threadgroup float tg_beta[4];

    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    float s4 = 0.0f, s5 = 0.0f, s6 = 0.0f, s7 = 0.0f;
    if (active_dv) {
        if (GDN_HEAD_K_DIM == 64u) {
            s0 = state_in[head_state_base + ulong(lane) * d_v + dv];
            s1 = state_in[head_state_base + ulong(lane + 32u) * d_v + dv];
        } else if (GDN_HEAD_K_DIM == 128u) {
            s0 = state_in[head_state_base + ulong(lane) * d_v + dv];
            s1 = state_in[head_state_base + ulong(lane + 32u) * d_v + dv];
            s2 = state_in[head_state_base + ulong(lane + 64u) * d_v + dv];
            s3 = state_in[head_state_base + ulong(lane + 96u) * d_v + dv];
        } else {
            if (lane < d_k)        s0 = state_in[head_state_base + ulong(lane) * d_v + dv];
            if (lane + 32u < d_k)  s1 = state_in[head_state_base + ulong(lane + 32u) * d_v + dv];
            if (lane + 64u < d_k)  s2 = state_in[head_state_base + ulong(lane + 64u) * d_v + dv];
            if (lane + 96u < d_k)  s3 = state_in[head_state_base + ulong(lane + 96u) * d_v + dv];
            if (lane + 128u < d_k) s4 = state_in[head_state_base + ulong(lane + 128u) * d_v + dv];
            if (lane + 160u < d_k) s5 = state_in[head_state_base + ulong(lane + 160u) * d_v + dv];
            if (lane + 192u < d_k) s6 = state_in[head_state_base + ulong(lane + 192u) * d_v + dv];
            if (lane + 224u < d_k) s7 = state_in[head_state_base + ulong(lane + 224u) * d_v + dv];
        }
    }

    for (uint t_base = 0u; t_base < p.seq_len; t_base += 4u) {
        const uint chunk_n = min(4u, p.seq_len - t_base);
        // Simdgroups 0..3 cooperatively stage Q, K, exp(g), beta for up to 4 tokens.
        if (sg_id < chunk_n) {
            const uint tok = t_base + sg_id;
            const ulong qk_off = (ulong(tok) * p.num_heads + h) * d_k;
            float q_sq = 0.0f;
            float k_sq = 0.0f;
            for (uint dk = lane; dk < d_k; dk += 32u) {
                const float qv = float(q[qk_off + dk]);
                const float kv = float(k[qk_off + dk]);
                tg_q[sg_id][dk] = qv;
                tg_k[sg_id][dk] = kv;
                tg_alpha[sg_id][dk] = exp(g[qk_off + dk]);
                q_sq = fma(qv, qv, q_sq);
                k_sq = fma(kv, kv, k_sq);
            }
            const float q_sum = simd_sum(q_sq);
            const float k_sum = simd_sum(k_sq);
            const float q_fac = (p.apply_l2_norm != 0u) ? (p.q_scale / sqrt(q_sum + p.l2_eps)) : p.q_scale;
            const float k_fac = (p.apply_l2_norm != 0u) ? (1.0f / sqrt(k_sum + p.l2_eps)) : 1.0f;
            for (uint dk = lane; dk < d_k; dk += 32u) {
                tg_q[sg_id][dk] *= q_fac;
                tg_k[sg_id][dk] *= k_fac;
            }
            if (lane == 0u) {
                const float bv = beta[ulong(tok) * p.num_heads + h];
                tg_beta[sg_id] = (p.beta_apply_sigmoid != 0u) ? hybrid_sigmoid(bv) : bv;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active_dv) {
            for (uint c = 0u; c < chunk_n; ++c) {
                const uint tok = t_base + c;
                const float beta_val = tg_beta[c];
                const ulong v_idx = (ulong(tok) * p.num_heads + h) * d_v + dv;
                const float v_val = float(v[v_idx]);

                float pred_acc = 0.0f;
                if (GDN_HEAD_K_DIM == 64u) {
                    s0 *= tg_alpha[c][lane];
                    s1 *= tg_alpha[c][lane + 32u];
                    pred_acc = fma(s0, tg_k[c][lane], s1 * tg_k[c][lane + 32u]);
                } else if (GDN_HEAD_K_DIM == 128u) {
                    s0 *= tg_alpha[c][lane];
                    s1 *= tg_alpha[c][lane + 32u];
                    s2 *= tg_alpha[c][lane + 64u];
                    s3 *= tg_alpha[c][lane + 96u];
                    pred_acc = fma(s0, tg_k[c][lane], fma(s1, tg_k[c][lane + 32u], fma(s2, tg_k[c][lane + 64u], s3 * tg_k[c][lane + 96u])));
                } else {
                    if (lane < d_k)        { s0 *= tg_alpha[c][lane]; pred_acc = fma(s0, tg_k[c][lane], pred_acc); }
                    if (lane + 32u < d_k)  { s1 *= tg_alpha[c][lane + 32u]; pred_acc = fma(s1, tg_k[c][lane + 32u], pred_acc); }
                    if (lane + 64u < d_k)  { s2 *= tg_alpha[c][lane + 64u]; pred_acc = fma(s2, tg_k[c][lane + 64u], pred_acc); }
                    if (lane + 96u < d_k)  { s3 *= tg_alpha[c][lane + 96u]; pred_acc = fma(s3, tg_k[c][lane + 96u], pred_acc); }
                    if (lane + 128u < d_k) { s4 *= tg_alpha[c][lane + 128u]; pred_acc = fma(s4, tg_k[c][lane + 128u], pred_acc); }
                    if (lane + 160u < d_k) { s5 *= tg_alpha[c][lane + 160u]; pred_acc = fma(s5, tg_k[c][lane + 160u], pred_acc); }
                    if (lane + 192u < d_k) { s6 *= tg_alpha[c][lane + 192u]; pred_acc = fma(s6, tg_k[c][lane + 192u], pred_acc); }
                    if (lane + 224u < d_k) { s7 *= tg_alpha[c][lane + 224u]; pred_acc = fma(s7, tg_k[c][lane + 224u], pred_acc); }
                }

                const float pred_v = simd_sum(pred_acc);
                const float delta = beta_val * (v_val - pred_v);

                float o_acc = 0.0f;
                if (GDN_HEAD_K_DIM == 64u) {
                    s0 = fma(tg_k[c][lane], delta, s0);
                    s1 = fma(tg_k[c][lane + 32u], delta, s1);
                    o_acc = fma(s0, tg_q[c][lane], s1 * tg_q[c][lane + 32u]);
                } else if (GDN_HEAD_K_DIM == 128u) {
                    s0 = fma(tg_k[c][lane], delta, s0);
                    s1 = fma(tg_k[c][lane + 32u], delta, s1);
                    s2 = fma(tg_k[c][lane + 64u], delta, s2);
                    s3 = fma(tg_k[c][lane + 96u], delta, s3);
                    o_acc = fma(s0, tg_q[c][lane], fma(s1, tg_q[c][lane + 32u], fma(s2, tg_q[c][lane + 64u], s3 * tg_q[c][lane + 96u])));
                } else {
                    if (lane < d_k)        { s0 = fma(tg_k[c][lane], delta, s0); o_acc = fma(s0, tg_q[c][lane], o_acc); }
                    if (lane + 32u < d_k)  { s1 = fma(tg_k[c][lane + 32u], delta, s1); o_acc = fma(s1, tg_q[c][lane + 32u], o_acc); }
                    if (lane + 64u < d_k)  { s2 = fma(tg_k[c][lane + 64u], delta, s2); o_acc = fma(s2, tg_q[c][lane + 64u], o_acc); }
                    if (lane + 96u < d_k)  { s3 = fma(tg_k[c][lane + 96u], delta, s3); o_acc = fma(s3, tg_q[c][lane + 96u], o_acc); }
                    if (lane + 128u < d_k) { s4 = fma(tg_k[c][lane + 128u], delta, s4); o_acc = fma(s4, tg_q[c][lane + 128u], o_acc); }
                    if (lane + 160u < d_k) { s5 = fma(tg_k[c][lane + 160u], delta, s5); o_acc = fma(s5, tg_q[c][lane + 160u], o_acc); }
                    if (lane + 192u < d_k) { s6 = fma(tg_k[c][lane + 192u], delta, s6); o_acc = fma(s6, tg_q[c][lane + 192u], o_acc); }
                    if (lane + 224u < d_k) { s7 = fma(tg_k[c][lane + 224u], delta, s7); o_acc = fma(s7, tg_q[c][lane + 224u], o_acc); }
                }

                if (p.write_state_only == 0u && tok >= p.out_start_token) {
                    const float o_val = simd_sum(o_acc);
                    if (lane == 0u) {
                        const ulong out_idx = (ulong(tok - p.out_start_token) * p.num_heads + h) * d_v + dv;
                        out[out_idx] = half(o_val);
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (active_dv) {
        if (GDN_HEAD_K_DIM == 64u) {
            state_out[head_state_base + ulong(lane) * d_v + dv] = s0;
            state_out[head_state_base + ulong(lane + 32u) * d_v + dv] = s1;
        } else if (GDN_HEAD_K_DIM == 128u) {
            state_out[head_state_base + ulong(lane) * d_v + dv] = s0;
            state_out[head_state_base + ulong(lane + 32u) * d_v + dv] = s1;
            state_out[head_state_base + ulong(lane + 64u) * d_v + dv] = s2;
            state_out[head_state_base + ulong(lane + 96u) * d_v + dv] = s3;
        } else {
            if (lane < d_k)        state_out[head_state_base + ulong(lane) * d_v + dv] = s0;
            if (lane + 32u < d_k)  state_out[head_state_base + ulong(lane + 32u) * d_v + dv] = s1;
            if (lane + 64u < d_k)  state_out[head_state_base + ulong(lane + 64u) * d_v + dv] = s2;
            if (lane + 96u < d_k)  state_out[head_state_base + ulong(lane + 96u) * d_v + dv] = s3;
            if (lane + 128u < d_k) state_out[head_state_base + ulong(lane + 128u) * d_v + dv] = s4;
            if (lane + 160u < d_k) state_out[head_state_base + ulong(lane + 160u) * d_v + dv] = s5;
            if (lane + 192u < d_k) state_out[head_state_base + ulong(lane + 192u) * d_v + dv] = s6;
            if (lane + 224u < d_k) state_out[head_state_base + ulong(lane + 224u) * d_v + dv] = s7;
        }
    }
}

// Tree-verification KDA (`num_nodes <= 64`, 64-bit `ancestor_masks`), non-mutating committed state.
// Grid: (num_heads, ceil(head_v_dim / 8), 1) threadgroups, 256 threads.
kernel void ami_kda_tree_verify(const device half *q [[buffer(0)]],
                                const device half *k [[buffer(1)]],
                                const device half *v [[buffer(2)]],
                                const device float *g [[buffer(3)]],
                                const device float *beta [[buffer(4)]],
                                const device ulong *ancestor_masks [[buffer(5)]],
                                const device float *state_in [[buffer(6)]],
                                device float *state_out [[buffer(7)]],
                                device half *out [[buffer(8)]],
                                constant KdaTreeVerifyParams &p [[buffer(9)]],
                                uint2 tg_pos [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]],
                                uint sg_id [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    const uint h = tg_pos.x;
    const uint dv = tg_pos.y * 8u + sg_id;
    if (h >= p.num_heads) {
        return;
    }

    const uint d_k = (GDN_HEAD_K_DIM > 0u) ? GDN_HEAD_K_DIM : p.head_k_dim;
    const uint d_v = p.head_v_dim;
    const bool active_dv = (dv < d_v);
    const ulong head_state_base = ulong(h) * d_k * d_v;

    threadgroup ulong tg_masks[64];
    threadgroup uchar tg_depth[64];
    threadgroup uchar tg_path[64];
    threadgroup uint tg_path_len;
    threadgroup float tg_beta[64];
    threadgroup float tg_q_fac[64];
    threadgroup float tg_k_fac[64];

    if (tid < p.num_nodes) {
        const ulong mask = ancestor_masks[tid];
        tg_masks[tid] = mask;
        tg_depth[tid] = uchar(popcount(mask));
        const float bv = beta[ulong(tid) * p.num_heads + h];
        tg_beta[tid] = (p.beta_apply_sigmoid != 0u) ? hybrid_sigmoid(bv) : bv;
    }
    for (uint node = sg_id; node < p.num_nodes; node += 8u) {
        const ulong qk_off = (ulong(node) * p.num_heads + h) * d_k;
        float q_sq = 0.0f;
        float k_sq = 0.0f;
        for (uint dk = lane; dk < d_k; dk += 32u) {
            const float qv = float(q[qk_off + dk]);
            const float kv = float(k[qk_off + dk]);
            q_sq = fma(qv, qv, q_sq);
            k_sq = fma(kv, kv, k_sq);
        }
        const float q_sum = simd_sum(q_sq);
        const float k_sum = simd_sum(k_sq);
        if (lane == 0u) {
            tg_q_fac[node] = (p.apply_l2_norm != 0u) ? (p.q_scale / sqrt(q_sum + p.l2_eps)) : p.q_scale;
            tg_k_fac[node] = (p.apply_l2_norm != 0u) ? (1.0f / sqrt(k_sum + p.l2_eps)) : 1.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float si0 = 0.0f, si1 = 0.0f, si2 = 0.0f, si3 = 0.0f;
    float si4 = 0.0f, si5 = 0.0f, si6 = 0.0f, si7 = 0.0f;
    if (active_dv) {
        if (lane < d_k)        si0 = state_in[head_state_base + ulong(lane) * d_v + dv];
        if (lane + 32u < d_k)  si1 = state_in[head_state_base + ulong(lane + 32u) * d_v + dv];
        if (lane + 64u < d_k)  si2 = state_in[head_state_base + ulong(lane + 64u) * d_v + dv];
        if (lane + 96u < d_k)  si3 = state_in[head_state_base + ulong(lane + 96u) * d_v + dv];
        if (lane + 128u < d_k) si4 = state_in[head_state_base + ulong(lane + 128u) * d_v + dv];
        if (lane + 160u < d_k) si5 = state_in[head_state_base + ulong(lane + 160u) * d_v + dv];
        if (lane + 192u < d_k) si6 = state_in[head_state_base + ulong(lane + 192u) * d_v + dv];
        if (lane + 224u < d_k) si7 = state_in[head_state_base + ulong(lane + 224u) * d_v + dv];
    }

    for (uint node = 0u; node < p.num_nodes; ++node) {
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
            float s0 = si0, s1 = si1, s2 = si2, s3 = si3;
            float s4 = si4, s5 = si5, s6 = si6, s7 = si7;
            const uint plen = tg_path_len;
            for (uint step = 0u; step < plen; ++step) {
                const uint j = uint(tg_path[step]);
                const float beta_val = tg_beta[j];
                const float k_fac = tg_k_fac[j];
                const ulong qk_off = (ulong(j) * p.num_heads + h) * d_k;
                const ulong v_idx = (ulong(j) * p.num_heads + h) * d_v + dv;
                const float v_val = float(v[v_idx]);

                float pred_acc = 0.0f;
                float k0 = 0.0f, k1 = 0.0f, k2 = 0.0f, k3 = 0.0f;
                float k4 = 0.0f, k5 = 0.0f, k6 = 0.0f, k7 = 0.0f;
                if (GDN_HEAD_K_DIM == 64u) {
                    k0 = float(k[qk_off + lane]) * k_fac;       s0 *= exp(g[qk_off + lane]);       pred_acc = fma(s0, k0, pred_acc);
                    k1 = float(k[qk_off + lane + 32u]) * k_fac; s1 *= exp(g[qk_off + lane + 32u]); pred_acc = fma(s1, k1, pred_acc);
                } else if (GDN_HEAD_K_DIM == 128u) {
                    k0 = float(k[qk_off + lane]) * k_fac;       s0 *= exp(g[qk_off + lane]);       pred_acc = fma(s0, k0, pred_acc);
                    k1 = float(k[qk_off + lane + 32u]) * k_fac; s1 *= exp(g[qk_off + lane + 32u]); pred_acc = fma(s1, k1, pred_acc);
                    k2 = float(k[qk_off + lane + 64u]) * k_fac; s2 *= exp(g[qk_off + lane + 64u]); pred_acc = fma(s2, k2, pred_acc);
                    k3 = float(k[qk_off + lane + 96u]) * k_fac; s3 *= exp(g[qk_off + lane + 96u]); pred_acc = fma(s3, k3, pred_acc);
                } else {
                    if (lane < d_k)        { k0 = float(k[qk_off + lane]) * k_fac;        s0 *= exp(g[qk_off + lane]);        pred_acc = fma(s0, k0, pred_acc); }
                    if (lane + 32u < d_k)  { k1 = float(k[qk_off + lane + 32u]) * k_fac;  s1 *= exp(g[qk_off + lane + 32u]);  pred_acc = fma(s1, k1, pred_acc); }
                    if (lane + 64u < d_k)  { k2 = float(k[qk_off + lane + 64u]) * k_fac;  s2 *= exp(g[qk_off + lane + 64u]);  pred_acc = fma(s2, k2, pred_acc); }
                    if (lane + 96u < d_k)  { k3 = float(k[qk_off + lane + 96u]) * k_fac;  s3 *= exp(g[qk_off + lane + 96u]);  pred_acc = fma(s3, k3, pred_acc); }
                    if (lane + 128u < d_k) { k4 = float(k[qk_off + lane + 128u]) * k_fac; s4 *= exp(g[qk_off + lane + 128u]); pred_acc = fma(s4, k4, pred_acc); }
                    if (lane + 160u < d_k) { k5 = float(k[qk_off + lane + 160u]) * k_fac; s5 *= exp(g[qk_off + lane + 160u]); pred_acc = fma(s5, k5, pred_acc); }
                    if (lane + 192u < d_k) { k6 = float(k[qk_off + lane + 192u]) * k_fac; s6 *= exp(g[qk_off + lane + 192u]); pred_acc = fma(s6, k6, pred_acc); }
                    if (lane + 224u < d_k) { k7 = float(k[qk_off + lane + 224u]) * k_fac; s7 *= exp(g[qk_off + lane + 224u]); pred_acc = fma(s7, k7, pred_acc); }
                }

                const float pred_v = simd_sum(pred_acc);
                const float delta = beta_val * (v_val - pred_v);

                if (GDN_HEAD_K_DIM == 64u) {
                    s0 = fma(k0, delta, s0); s1 = fma(k1, delta, s1);
                } else if (GDN_HEAD_K_DIM == 128u) {
                    s0 = fma(k0, delta, s0); s1 = fma(k1, delta, s1);
                    s2 = fma(k2, delta, s2); s3 = fma(k3, delta, s3);
                } else {
                    if (lane < d_k)        s0 = fma(k0, delta, s0);
                    if (lane + 32u < d_k)  s1 = fma(k1, delta, s1);
                    if (lane + 64u < d_k)  s2 = fma(k2, delta, s2);
                    if (lane + 96u < d_k)  s3 = fma(k3, delta, s3);
                    if (lane + 128u < d_k) s4 = fma(k4, delta, s4);
                    if (lane + 160u < d_k) s5 = fma(k5, delta, s5);
                    if (lane + 192u < d_k) s6 = fma(k6, delta, s6);
                    if (lane + 224u < d_k) s7 = fma(k7, delta, s7);
                }
            }

            const ulong q_off = (ulong(node) * p.num_heads + h) * d_k;
            const float q_fac = tg_q_fac[node];
            float o_acc = 0.0f;
            if (GDN_HEAD_K_DIM == 64u) {
                o_acc = fma(s0, float(q[q_off + lane]) * q_fac, s1 * (float(q[q_off + lane + 32u]) * q_fac));
            } else if (GDN_HEAD_K_DIM == 128u) {
                o_acc = fma(s0, float(q[q_off + lane]) * q_fac,
                        fma(s1, float(q[q_off + lane + 32u]) * q_fac,
                        fma(s2, float(q[q_off + lane + 64u]) * q_fac,
                            s3 * (float(q[q_off + lane + 96u]) * q_fac))));
            } else {
                if (lane < d_k)        o_acc = fma(s0, float(q[q_off + lane]) * q_fac, o_acc);
                if (lane + 32u < d_k)  o_acc = fma(s1, float(q[q_off + lane + 32u]) * q_fac, o_acc);
                if (lane + 64u < d_k)  o_acc = fma(s2, float(q[q_off + lane + 64u]) * q_fac, o_acc);
                if (lane + 96u < d_k)  o_acc = fma(s3, float(q[q_off + lane + 96u]) * q_fac, o_acc);
                if (lane + 128u < d_k) o_acc = fma(s4, float(q[q_off + lane + 128u]) * q_fac, o_acc);
                if (lane + 160u < d_k) o_acc = fma(s5, float(q[q_off + lane + 160u]) * q_fac, o_acc);
                if (lane + 192u < d_k) o_acc = fma(s6, float(q[q_off + lane + 192u]) * q_fac, o_acc);
                if (lane + 224u < d_k) o_acc = fma(s7, float(q[q_off + lane + 224u]) * q_fac, o_acc);
            }

            const float o_val = simd_sum(o_acc);
            if (lane == 0u) {
                out[(ulong(node) * p.num_heads + h) * d_v + dv] = half(o_val);
            }

            if (node == p.scratch_leaf_node) {
                if (lane < d_k)        state_out[head_state_base + ulong(lane) * d_v + dv] = s0;
                if (lane + 32u < d_k)  state_out[head_state_base + ulong(lane + 32u) * d_v + dv] = s1;
                if (lane + 64u < d_k)  state_out[head_state_base + ulong(lane + 64u) * d_v + dv] = s2;
                if (lane + 96u < d_k)  state_out[head_state_base + ulong(lane + 96u) * d_v + dv] = s3;
                if (lane + 128u < d_k) state_out[head_state_base + ulong(lane + 128u) * d_v + dv] = s4;
                if (lane + 160u < d_k) state_out[head_state_base + ulong(lane + 160u) * d_v + dv] = s5;
                if (lane + 192u < d_k) state_out[head_state_base + ulong(lane + 192u) * d_v + dv] = s6;
                if (lane + 224u < d_k) state_out[head_state_base + ulong(lane + 224u) * d_v + dv] = s7;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// In-place per-head FP32 RMSNorm + sigmoid output gate (`Glm5NextTextRMSNormGated`).
// Grid: (num_heads, num_tokens, 1) threadgroups, 32 threads.
kernel void ami_kda_out_norm_gate(device half *out [[buffer(0)]],
                                  const device half *z [[buffer(1)]],
                                  const device uchar *norm_w [[buffer(2)]],
                                  constant GdnOutNormGateParams &p [[buffer(3)]],
                                  uint2 tg_pos [[threadgroup_position_in_grid]],
                                  uint lane [[thread_index_in_simdgroup]]) {
    const uint h = tg_pos.x;
    const uint tok = tg_pos.y;
    if (h >= p.num_v_heads || tok >= p.num_tokens) {
        return;
    }

    const uint d_v = p.head_v_dim;
    const ulong base = (ulong(tok) * p.num_v_heads + h) * d_v;

    float inv_rms = 1.0f;
    if (p.apply_out_norm != 0u) {
        float sum_sq = 0.0f;
        for (uint dv = lane; dv < d_v; dv += 32u) {
            const float ov = float(out[base + dv]);
            sum_sq = fma(ov, ov, sum_sq);
        }
        const float total_sq = simd_sum(sum_sq);
        inv_rms = rsqrt(total_sq / float(d_v) + p.rms_eps);
    }

    const uint nw_base = (p.per_head_norm != 0u) ? (h * d_v) : 0u;
    for (uint dv = lane; dv < d_v; dv += 32u) {
        float val = float(out[base + dv]);
        if (p.apply_out_norm != 0u) {
            val = (val * inv_rms) * hybrid_load_norm_w(norm_w, nw_base + dv, HYBRID_NORM_W_BF16);
        }
        if (p.apply_out_gate != 0u) {
            val *= hybrid_sigmoid(float(z[base + dv]));
        }
        out[base + dv] = half(val);
    }
}

// Advances committed KDA recurrent state along `accepted_nodes[0..accepted_count]`.
// Grid: (num_heads, ceil(head_v_dim / 8), 1) threadgroups, 256 threads.
kernel void ami_kda_commit_tree(const device half *k_tree [[buffer(0)]],
                                const device half *v_tree [[buffer(1)]],
                                const device float *g_tree [[buffer(2)]],
                                const device float *beta_tree [[buffer(3)]],
                                const device ushort *accepted_nodes [[buffer(4)]],
                                const device float *state_in [[buffer(5)]],
                                device float *state_out [[buffer(6)]],
                                constant KdaCommitTreeParams &p [[buffer(7)]],
                                uint2 tg_pos [[threadgroup_position_in_grid]],
                                uint sg_id [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    const uint h = tg_pos.x;
    const uint dv = tg_pos.y * 8u + sg_id;
    if (h >= p.num_heads || dv >= p.head_v_dim) {
        return;
    }

    const uint d_k = (GDN_HEAD_K_DIM > 0u) ? GDN_HEAD_K_DIM : p.head_k_dim;
    const uint d_v = p.head_v_dim;
    const ulong head_state_base = ulong(h) * d_k * d_v;

    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    float s4 = 0.0f, s5 = 0.0f, s6 = 0.0f, s7 = 0.0f;
    if (lane < d_k)        s0 = state_in[head_state_base + ulong(lane) * d_v + dv];
    if (lane + 32u < d_k)  s1 = state_in[head_state_base + ulong(lane + 32u) * d_v + dv];
    if (lane + 64u < d_k)  s2 = state_in[head_state_base + ulong(lane + 64u) * d_v + dv];
    if (lane + 96u < d_k)  s3 = state_in[head_state_base + ulong(lane + 96u) * d_v + dv];
    if (lane + 128u < d_k) s4 = state_in[head_state_base + ulong(lane + 128u) * d_v + dv];
    if (lane + 160u < d_k) s5 = state_in[head_state_base + ulong(lane + 160u) * d_v + dv];
    if (lane + 192u < d_k) s6 = state_in[head_state_base + ulong(lane + 192u) * d_v + dv];
    if (lane + 224u < d_k) s7 = state_in[head_state_base + ulong(lane + 224u) * d_v + dv];

    for (uint step = 0u; step < p.accepted_count; ++step) {
        const uint node = (p.use_node_buffer != 0u) ? uint(accepted_nodes[step]) : uint(p.inline_nodes[step]);
        const ulong qk_off = (ulong(node) * p.num_heads + h) * d_k;
        const ulong v_idx = (ulong(node) * p.num_heads + h) * d_v + dv;

        float k0 = 0.0f, k1 = 0.0f, k2 = 0.0f, k3 = 0.0f;
        float k4 = 0.0f, k5 = 0.0f, k6 = 0.0f, k7 = 0.0f;
        float k_sq = 0.0f;
        if (lane < d_k)        { k0 = float(k_tree[qk_off + lane]);        k_sq = fma(k0, k0, k_sq); }
        if (lane + 32u < d_k)  { k1 = float(k_tree[qk_off + lane + 32u]);  k_sq = fma(k1, k1, k_sq); }
        if (lane + 64u < d_k)  { k2 = float(k_tree[qk_off + lane + 64u]);  k_sq = fma(k2, k2, k_sq); }
        if (lane + 96u < d_k)  { k3 = float(k_tree[qk_off + lane + 96u]);  k_sq = fma(k3, k3, k_sq); }
        if (lane + 128u < d_k) { k4 = float(k_tree[qk_off + lane + 128u]); k_sq = fma(k4, k4, k_sq); }
        if (lane + 160u < d_k) { k5 = float(k_tree[qk_off + lane + 160u]); k_sq = fma(k5, k5, k_sq); }
        if (lane + 192u < d_k) { k6 = float(k_tree[qk_off + lane + 192u]); k_sq = fma(k6, k6, k_sq); }
        if (lane + 224u < d_k) { k7 = float(k_tree[qk_off + lane + 224u]); k_sq = fma(k7, k7, k_sq); }
        const float k_fac = (p.apply_l2_norm != 0u) ? (1.0f / sqrt(simd_sum(k_sq) + p.l2_eps)) : 1.0f;
        k0 *= k_fac; k1 *= k_fac; k2 *= k_fac; k3 *= k_fac;
        k4 *= k_fac; k5 *= k_fac; k6 *= k_fac; k7 *= k_fac;

        float pred_acc = 0.0f;
        if (lane < d_k)        { s0 *= exp(g_tree[qk_off + lane]);        pred_acc = fma(s0, k0, pred_acc); }
        if (lane + 32u < d_k)  { s1 *= exp(g_tree[qk_off + lane + 32u]);  pred_acc = fma(s1, k1, pred_acc); }
        if (lane + 64u < d_k)  { s2 *= exp(g_tree[qk_off + lane + 64u]);  pred_acc = fma(s2, k2, pred_acc); }
        if (lane + 96u < d_k)  { s3 *= exp(g_tree[qk_off + lane + 96u]);  pred_acc = fma(s3, k3, pred_acc); }
        if (lane + 128u < d_k) { s4 *= exp(g_tree[qk_off + lane + 128u]); pred_acc = fma(s4, k4, pred_acc); }
        if (lane + 160u < d_k) { s5 *= exp(g_tree[qk_off + lane + 160u]); pred_acc = fma(s5, k5, pred_acc); }
        if (lane + 192u < d_k) { s6 *= exp(g_tree[qk_off + lane + 192u]); pred_acc = fma(s6, k6, pred_acc); }
        if (lane + 224u < d_k) { s7 *= exp(g_tree[qk_off + lane + 224u]); pred_acc = fma(s7, k7, pred_acc); }

        const float pred_v = simd_sum(pred_acc);
        const float bv = beta_tree[ulong(node) * p.num_heads + h];
        const float beta_val = (p.beta_apply_sigmoid != 0u) ? hybrid_sigmoid(bv) : bv;
        const float delta = beta_val * (float(v_tree[v_idx]) - pred_v);

        if (lane < d_k)        s0 = fma(k0, delta, s0);
        if (lane + 32u < d_k)  s1 = fma(k1, delta, s1);
        if (lane + 64u < d_k)  s2 = fma(k2, delta, s2);
        if (lane + 96u < d_k)  s3 = fma(k3, delta, s3);
        if (lane + 128u < d_k) s4 = fma(k4, delta, s4);
        if (lane + 160u < d_k) s5 = fma(k5, delta, s5);
        if (lane + 192u < d_k) s6 = fma(k6, delta, s6);
        if (lane + 224u < d_k) s7 = fma(k7, delta, s7);
    }

    if (lane < d_k)        state_out[head_state_base + ulong(lane) * d_v + dv] = s0;
    if (lane + 32u < d_k)  state_out[head_state_base + ulong(lane + 32u) * d_v + dv] = s1;
    if (lane + 64u < d_k)  state_out[head_state_base + ulong(lane + 64u) * d_v + dv] = s2;
    if (lane + 96u < d_k)  state_out[head_state_base + ulong(lane + 96u) * d_v + dv] = s3;
    if (lane + 128u < d_k) state_out[head_state_base + ulong(lane + 128u) * d_v + dv] = s4;
    if (lane + 160u < d_k) state_out[head_state_base + ulong(lane + 160u) * d_v + dv] = s5;
    if (lane + 192u < d_k) state_out[head_state_base + ulong(lane + 192u) * d_v + dv] = s6;
    if (lane + 224u < d_k) state_out[head_state_base + ulong(lane + 224u) * d_v + dv] = s7;
}

// ---------------------------------------------------------------------------
// G.3: `GlmMoeDsaIndexer` (`GLM-5.2/5.3`), `NoPE MLA` + `index_kpool = 4`
// (`GLM-5.3-Flash`), and Sparse Absorbed MLA Attention
// ---------------------------------------------------------------------------

struct G27RopeInterleaveParams {
    uint num_tokens;
    uint num_heads;
    uint head_stride;
    uint rot_dim;
    uint rot_offset;
    uint base_pos;
    uint use_pos_buf;
    float rope_theta;
    float attn_scaling;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

struct G27LayerNormParams {
    uint num_rows;
    uint dim;
    uint w_dtype; // 0=f16, 1=bf16, 2=f32
    uint b_dtype; // 0=f16, 1=bf16, 2=f32
    float eps;
    uint has_bias;
    uint _pad0;
    uint _pad1;
};

struct G27MlaKvWriteParams {
    uint num_tokens;
    uint kv_lora_rank;
    uint qk_rope_head_dim;
    uint start_pos;
    uint is_tree_commit;
    uint _pad0;
    uint _pad1;
    uint _pad2;
    ushort inline_nodes[64];
};

struct G27DsaIdxKvWriteParams {
    uint num_tokens;
    uint index_head_dim;
    uint start_pos;
    uint has_gate;
    uint is_tree_commit;
    uint _pad0;
    uint _pad1;
    uint _pad2;
    ushort inline_nodes[64];
};

struct G27DsaKpoolBuildParams {
    uint start_pool;
    uint end_pool;
    uint index_head_dim;
    uint kpool;
    uint ape_dtype; // 0=f16, 1=bf16, 2=f32
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

struct G27DsaScoresParams {
    uint num_rows;
    uint index_n_heads;
    uint index_head_dim;
    uint committed_kv_len; // committed tokens in k_cache (or all tokens in linear prefill)
    uint num_candidates;   // total score slots per row
    uint kpool;            // 1 = token-level (`glm_moe_dsa`), 4 = pooled (`glm5_next`)
    uint causal_base_pos;  // for linear causal mode (is_tree == 0)
    uint is_tree;          // 1 = use ancestor_masks + k_tree for uncommitted tokens
    uint weights_f16;      // 1 = weights buffer is f16, 0 = f32
    uint ape_dtype;        // 0=f16, 1=bf16, 2=f32
    float softmax_scale;   // index_head_dim^-0.5
    float head_weight_scale; // index_n_heads^-0.5
};

struct G27DsaSelectTopKParams {
    uint num_rows;
    uint committed_kv_len;
    uint num_candidates;
    uint index_topk;
    uint max_out_k;
    uint kpool;
    uint always_select_tail;
    uint causal_base_pos;
    uint is_tree;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

struct G27MlaSparseAttnParams {
    uint num_rows;
    uint num_q_heads;
    uint kv_lora_rank;
    uint qk_rope_head_dim; // 0 for NoPE MLA (`glm5_next`)
    uint committed_kv_len;
    uint max_out_k;
    float scale;
    uint _pad0;
};

static_assert(sizeof(G27RopeInterleaveParams) == 48, "G27RopeInterleaveParams must be 48 bytes");
static_assert(sizeof(G27LayerNormParams) == 32, "G27LayerNormParams must be 32 bytes");
static_assert(sizeof(G27MlaKvWriteParams) == 160, "G27MlaKvWriteParams must be 160 bytes");
static_assert(sizeof(G27DsaIdxKvWriteParams) == 160, "G27DsaIdxKvWriteParams must be 160 bytes");
static_assert(sizeof(G27DsaKpoolBuildParams) == 32, "G27DsaKpoolBuildParams must be 32 bytes");
static_assert(sizeof(G27DsaScoresParams) == 48, "G27DsaScoresParams must be 48 bytes");
static_assert(sizeof(G27DsaSelectTopKParams) == 48, "G27DsaSelectTopKParams must be 48 bytes");
static_assert(sizeof(G27MlaSparseAttnParams) == 32, "G27MlaSparseAttnParams must be 32 bytes");

// Interleaved RoPE matching `apply_rotary_pos_emb_interleave`:
//   out[j]         = in[2*j] * cos - in[2*j + 1] * sin
//   out[j + D/2]   = in[2*j + 1] * cos + in[2*j] * sin
// Grid: (num_heads, num_tokens, 1), 32 threads (1 simdgroup).
kernel void ami_g27_rope_interleave(device half *x [[buffer(0)]],
                                    const device uint *positions [[buffer(1)]],
                                    constant G27RopeInterleaveParams &p [[buffer(2)]],
                                    uint2 tg_pos [[threadgroup_position_in_grid]],
                                    uint lane [[thread_index_in_simdgroup]]) {
    const uint head = tg_pos.x;
    const uint tok = tg_pos.y;
    if (head >= p.num_heads || tok >= p.num_tokens || p.rot_dim < 2u || p.rot_dim > 256u) {
        return;
    }

    const uint half_d = p.rot_dim >> 1;
    const uint pos = (p.use_pos_buf != 0u) ? positions[tok] : (p.base_pos + tok);
    const ulong base = (ulong(tok) * p.num_heads + head) * p.head_stride + p.rot_offset;

    threadgroup float tg_out[256];
    const float inv_rot_dim = 1.0f / float(p.rot_dim);
    const float log_theta = log(p.rope_theta);

    for (uint j = lane; j < half_d; j += 32u) {
        const float x0 = float(x[base + 2u * j]);
        const float x1 = float(x[base + 2u * j + 1u]);
        const float inv_freq = exp(-log_theta * (float(2u * j) * inv_rot_dim));
        const float angle = float(pos) * inv_freq;
        const float c = cos(angle) * p.attn_scaling;
        const float s = sin(angle) * p.attn_scaling;
        tg_out[j] = x0 * c - x1 * s;
        tg_out[j + half_d] = x1 * c + x0 * s;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = lane; d < p.rot_dim; d += 32u) {
        x[base + d] = half(tg_out[d]);
    }
}

// Standard FP32 LayerNorm with weight + optional bias (`k_norm` in `GlmMoeDsaIndexer` & `Glm5NextTextIndexer`).
// Grid: (num_rows, 1, 1), 128 threads (4 simdgroups).
kernel void ami_g27_layernorm(const device half *x [[buffer(0)]],
                              const device uchar *weight [[buffer(1)]],
                              const device uchar *bias [[buffer(2)]],
                              device half *y [[buffer(3)]],
                              constant G27LayerNormParams &p [[buffer(4)]],
                              uint tg_x [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]],
                              uint sg_id [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
    const uint row = tg_x;
    if (row >= p.num_rows || p.dim == 0u) {
        return;
    }

    threadgroup float tg_red[4];
    const ulong base = ulong(row) * p.dim;

    float sum_x = 0.0f;
    for (uint d = tid; d < p.dim; d += 128u) {
        sum_x += float(x[base + d]);
    }
    const float sg_sum = simd_sum(sum_x);
    if (lane == 0u) tg_red[sg_id] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float mean = ((tg_red[0] + tg_red[1]) + (tg_red[2] + tg_red[3])) / float(p.dim);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sum_sq = 0.0f;
    for (uint d = tid; d < p.dim; d += 128u) {
        const float diff = float(x[base + d]) - mean;
        sum_sq = fma(diff, diff, sum_sq);
    }
    const float sg_sq = simd_sum(sum_sq);
    if (lane == 0u) tg_red[sg_id] = sg_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float var = ((tg_red[0] + tg_red[1]) + (tg_red[2] + tg_red[3])) / float(p.dim);
    const float inv_std = rsqrt(var + p.eps);

    for (uint d = tid; d < p.dim; d += 128u) {
        const float norm_v = (float(x[base + d]) - mean) * inv_std;
        const float wv = g27_load_scalar_f32(weight, d, p.w_dtype);
        const float bv = (p.has_bias != 0u) ? g27_load_scalar_f32(bias, d, p.b_dtype) : 0.0f;
        y[base + d] = half(fma(norm_v, wv, bv));
    }
}

// Writes compressed FP16 MLA KV (`latent [kv_lora_rank]` + optional `rope [qk_rope_head_dim]`)
// into `[max_seq_len, kv_lora_rank + qk_rope_head_dim]` cache (linear or tree commit).
// Grid: (num_tokens, 1, 1), 256 threads.
kernel void ami_g27_mla_kv_write(const device half *latent_in [[buffer(0)]],
                                 const device half *rope_in [[buffer(1)]],
                                 device half *kv_cache [[buffer(2)]],
                                 constant G27MlaKvWriteParams &p [[buffer(3)]],
                                 uint tg_x [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]]) {
    const uint i = tg_x;
    if (i >= p.num_tokens) {
        return;
    }

    const uint src_row = (p.is_tree_commit != 0u) ? uint(p.inline_nodes[i]) : i;
    const uint dst_pos = p.start_pos + i;
    const uint row_stride = p.kv_lora_rank + p.qk_rope_head_dim;
    device half *dst = kv_cache + ulong(dst_pos) * row_stride;

    const device half *lat_src = latent_in + ulong(src_row) * p.kv_lora_rank;
    for (uint d = tid; d < p.kv_lora_rank; d += 256u) {
        dst[d] = lat_src[d];
    }
    if (p.qk_rope_head_dim > 0u) {
        const device half *rope_src = rope_in + ulong(src_row) * p.qk_rope_head_dim;
        for (uint r = tid; r < p.qk_rope_head_dim; r += 256u) {
            dst[p.kv_lora_rank + r] = rope_src[r];
        }
    }
}

// Writes FP16 indexer K (`[M, D]`) + optional k-pool gate scores (`[M, D]`) into indexer caches.
// Grid: (num_tokens, 1, 1), 128 threads.
kernel void ami_g27_dsa_idx_kv_write(const device half *k_in [[buffer(0)]],
                                     const device half *gate_in [[buffer(1)]],
                                     device half *k_cache [[buffer(2)]],
                                     device half *gate_cache [[buffer(3)]],
                                     constant G27DsaIdxKvWriteParams &p [[buffer(4)]],
                                     uint tg_x [[threadgroup_position_in_grid]],
                                     uint tid [[thread_index_in_threadgroup]]) {
    const uint i = tg_x;
    if (i >= p.num_tokens) {
        return;
    }

    const uint src_row = (p.is_tree_commit != 0u) ? uint(p.inline_nodes[i]) : i;
    const uint dst_pos = p.start_pos + i;
    const ulong src_off = ulong(src_row) * p.index_head_dim;
    const ulong dst_off = ulong(dst_pos) * p.index_head_dim;

    for (uint d = tid; d < p.index_head_dim; d += 128u) {
        k_cache[dst_off + d] = k_in[src_off + d];
        if (p.has_gate != 0u) {
            gate_cache[dst_off + d] = gate_in[src_off + d];
        }
    }
}

// Builds compressed k-pool keys `pool_k[pool, d] = sum_{j=0..kpool-1} softmax_j(gate[pool*kpool+j, d] + ape[j, d]) * k[pool*kpool+j, d]`.
// Grid: (end_pool - start_pool, 1, 1), 128 threads.
kernel void ami_g27_dsa_kpool_build(const device half *k_cache [[buffer(0)]],
                                    const device half *gate_cache [[buffer(1)]],
                                    const device uchar *kpool_ape [[buffer(2)]],
                                    device half *pool_k_cache [[buffer(3)]],
                                    constant G27DsaKpoolBuildParams &p [[buffer(4)]],
                                    uint tg_x [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]]) {
    const uint pool = p.start_pool + tg_x;
    if (pool >= p.end_pool || p.kpool == 0u) {
        return;
    }

    const uint dim = p.index_head_dim;
    const uint tok0 = pool * p.kpool;

    if (p.kpool == 4u) {
        for (uint d = tid; d < dim; d += 128u) {
            const float g0 = float(gate_cache[ulong(tok0 + 0u) * dim + d]) + g27_load_scalar_f32(kpool_ape, 0ul * dim + d, p.ape_dtype);
            const float g1 = float(gate_cache[ulong(tok0 + 1u) * dim + d]) + g27_load_scalar_f32(kpool_ape, 1ul * dim + d, p.ape_dtype);
            const float g2 = float(gate_cache[ulong(tok0 + 2u) * dim + d]) + g27_load_scalar_f32(kpool_ape, 2ul * dim + d, p.ape_dtype);
            const float g3 = float(gate_cache[ulong(tok0 + 3u) * dim + d]) + g27_load_scalar_f32(kpool_ape, 3ul * dim + d, p.ape_dtype);
            const float m = max(max(g0, g1), max(g2, g3));
            const float4 e = exp(float4(g0, g1, g2, g3) - m);
            const float4 pr = e * (1.0f / (e.x + e.y + e.z + e.w));
            const float4 kv = float4(
                float(k_cache[ulong(tok0 + 0u) * dim + d]),
                float(k_cache[ulong(tok0 + 1u) * dim + d]),
                float(k_cache[ulong(tok0 + 2u) * dim + d]),
                float(k_cache[ulong(tok0 + 3u) * dim + d]));
            pool_k_cache[ulong(pool) * dim + d] = half(dot(pr, kv));
        }
    } else {
        for (uint d = tid; d < dim; d += 128u) {
            float m = -INFINITY;
            for (uint j = 0u; j < p.kpool; ++j) {
                const float lv = float(gate_cache[ulong(tok0 + j) * dim + d]) + g27_load_scalar_f32(kpool_ape, ulong(j) * dim + d, p.ape_dtype);
                m = max(m, lv);
            }
            float denom = 0.0f;
            float num = 0.0f;
            for (uint j = 0u; j < p.kpool; ++j) {
                const float lv = float(gate_cache[ulong(tok0 + j) * dim + d]) + g27_load_scalar_f32(kpool_ape, ulong(j) * dim + d, p.ape_dtype);
                const float ev = exp(lv - m);
                denom += ev;
                num = fma(ev, float(k_cache[ulong(tok0 + j) * dim + d]), num);
            }
            pool_k_cache[ulong(pool) * dim + d] = half((denom > 0.0f) ? (num / denom) : 0.0f);
        }
    }
}

// Reconstructs root-to-node ancestor chain into `tg_path` (length = popcount(mask)).
static inline void g27_build_tree_path(const device ulong *ancestor_masks,
                                       uint num_rows,
                                       uint node,
                                       threadgroup uchar *tg_path,
                                       threadgroup uint &tg_path_len) {
    ulong rem = ancestor_masks[node];
    const uint depth_n = uint(popcount(rem));
    tg_path_len = depth_n;
    while (rem != 0ul) {
        const uint j = uint(ctz(rem));
        rem &= (rem - 1ul);
        if (j < num_rows) {
            const uint pos = uint(popcount(ancestor_masks[j])) - 1u;
            if (pos < depth_n && pos < 64u) {
                tg_path[pos] = uchar(j);
            }
        }
    }
}

// Parallel DSA Indexer scoring kernel for BOTH token-level (`kpool = 1`, `glm_moe_dsa`)
// and pooled-key (`kpool = 4`, `glm5_next`) modes, supporting linear causal and tree masks.
//   score[tok, c] = sum_{h=0..H_idx-1} (weights[tok, h] * H_idx^-0.5) * relu((q[tok, h] . k_cand[c]) * D_idx^-0.5)
// Grid: (ceil(num_candidates / 64), num_rows, 1), 64 threads (2 simdgroups).
kernel void ami_g27_dsa_scores(const device half *q_idx [[buffer(0)]],
                               const device uchar *weights_raw [[buffer(1)]],
                               const device half *k_cand_cache [[buffer(2)]],
                               const device half *k_raw_cache [[buffer(3)]],
                               const device half *k_tree [[buffer(4)]],
                               const device half *gate_cache [[buffer(5)]],
                               const device half *gate_tree [[buffer(6)]],
                               const device uchar *kpool_ape [[buffer(7)]],
                               const device ulong *ancestor_masks [[buffer(8)]],
                               device float *scores_out [[buffer(9)]],
                               constant G27DsaScoresParams &p [[buffer(10)]],
                               uint2 tg_pos [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]]) {
    const uint c = tg_pos.x * 64u + tid;
    const uint tok = tg_pos.y;
    if (tok >= p.num_rows) {
        return;
    }

    threadgroup uchar tg_path[64];
    threadgroup uint tg_path_len;
    if (p.is_tree != 0u && tid == 0u) {
        g27_build_tree_path(ancestor_masks, p.num_rows, tok, tg_path, tg_path_len);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (c >= p.num_candidates) {
        return;
    }

    const uint dim = p.index_head_dim;
    const ulong out_idx = ulong(tok) * p.num_candidates + c;

    // Determine candidate visibility and whether it is a boundary tree pool (kpool > 1).
    bool visible = false;
    bool use_tree_key = false;
    uint tree_node_idx = 0u;
    bool is_boundary_pool = false;

    if (p.kpool <= 1u) {
        if (p.is_tree == 0u) {
            const uint q_pos = p.causal_base_pos + tok;
            visible = (c <= q_pos);
        } else {
            if (c < p.committed_kv_len) {
                visible = true;
            } else {
                const uint j = c - p.committed_kv_len;
                if (j < p.num_rows && (((ancestor_masks[tok] >> j) & 1ul) != 0ul)) {
                    visible = true;
                    use_tree_key = true;
                    tree_node_idx = j;
                }
            }
        }
    } else {
        const uint pool_end = c * p.kpool + (p.kpool - 1u);
        if (p.is_tree == 0u) {
            const uint q_pos = p.causal_base_pos + tok;
            visible = (pool_end <= q_pos);
        } else {
            const uint visible_count = p.committed_kv_len + tg_path_len;
            if (pool_end < visible_count) {
                visible = true;
                if (pool_end >= p.committed_kv_len) {
                    is_boundary_pool = true;
                }
            }
        }
    }

    if (!visible) {
        scores_out[out_idx] = -INFINITY;
        return;
    }

    const ulong q_tok_base = ulong(tok) * p.index_n_heads * dim;
    const ulong w_tok_base = ulong(tok) * p.index_n_heads;
    float total_score = 0.0f;

    if (!is_boundary_pool) {
        const device half *k_ptr = use_tree_key
            ? (k_tree + ulong(tree_node_idx) * dim)
            : (k_cand_cache + ulong(c) * dim);

        for (uint h = 0u; h < p.index_n_heads; ++h) {
            const device half *q_ptr = q_tok_base + q_idx + ulong(h) * dim;
            float dot_v = 0.0f;
            if ((dim & 3u) == 0u) {
                const uint vdim = dim >> 2;
                const device half4 *q4 = (const device half4 *)q_ptr;
                const device half4 *k4 = (const device half4 *)k_ptr;
                for (uint d = 0u; d < vdim; ++d) {
                    dot_v += dot(float4(q4[d]), float4(k4[d]));
                }
            } else {
                for (uint d = 0u; d < dim; ++d) {
                    dot_v = fma(float(q_ptr[d]), float(k_ptr[d]), dot_v);
                }
            }
            const float relu_s = max(dot_v * p.softmax_scale, 0.0f);
            const float w_raw = (p.weights_f16 != 0u)
                ? float(((const device half *)weights_raw)[w_tok_base + h])
                : ((const device float *)weights_raw)[w_tok_base + h];
            total_score = fma(w_raw * p.head_weight_scale, relu_s, total_score);
        }
    } else {
        // Boundary pool formed by uncommitted prefix tail + tree ancestors (`kpool == 4` fast path).
        const uint tok0 = c * p.kpool;
        for (uint h = 0u; h < p.index_n_heads; ++h) {
            const device half *q_ptr = q_idx + q_tok_base + ulong(h) * dim;
            float dot_v = 0.0f;
            for (uint d = 0u; d < dim; ++d) {
                float m_l = -INFINITY;
                for (uint j = 0u; j < p.kpool; ++j) {
                    const uint seq_p = tok0 + j;
                    const float gv = (seq_p < p.committed_kv_len)
                        ? float(gate_cache[ulong(seq_p) * dim + d])
                        : float(gate_tree[ulong(uint(tg_path[seq_p - p.committed_kv_len])) * dim + d]);
                    m_l = max(m_l, gv + g27_load_scalar_f32(kpool_ape, ulong(j) * dim + d, p.ape_dtype));
                }
                float denom = 0.0f;
                float num = 0.0f;
                for (uint j = 0u; j < p.kpool; ++j) {
                    const uint seq_p = tok0 + j;
                    const float gv = (seq_p < p.committed_kv_len)
                        ? float(gate_cache[ulong(seq_p) * dim + d])
                        : float(gate_tree[ulong(uint(tg_path[seq_p - p.committed_kv_len])) * dim + d]);
                    const float kv = (seq_p < p.committed_kv_len)
                        ? float(k_raw_cache[ulong(seq_p) * dim + d])
                        : float(k_tree[ulong(uint(tg_path[seq_p - p.committed_kv_len])) * dim + d]);
                    const float ev = exp(gv + g27_load_scalar_f32(kpool_ape, ulong(j) * dim + d, p.ape_dtype) - m_l);
                    denom += ev;
                    num = fma(ev, kv, num);
                }
                const float pk = (denom > 0.0f) ? float(half(num / denom)) : 0.0f;
                dot_v = fma(float(q_ptr[d]), pk, dot_v);
            }
            const float relu_s = max(dot_v * p.softmax_scale, 0.0f);
            const float w_raw = (p.weights_f16 != 0u)
                ? float(((const device half *)weights_raw)[w_tok_base + h])
                : ((const device float *)weights_raw)[w_tok_base + h];
            total_score = fma(w_raw * p.head_weight_scale, relu_s, total_score);
        }
    }

    scores_out[out_idx] = total_score;
}

static inline uint g27_float_to_ordered_key(float s) {
    const uint bits = as_type<uint>(s);
    return ((bits & 0x80000000u) != 0u) ? ~bits : (bits ^ 0x80000000u);
}

static inline uint g27_map_seq_pos_to_kv_index(uint seq_pos,
                                               uint committed_kv_len,
                                               uint is_tree,
                                               const threadgroup uchar *tg_path) {
    if (is_tree == 0u || seq_pos < committed_kv_len) {
        return seq_pos;
    }
    return committed_kv_len + uint(tg_path[seq_pos - committed_kv_len]);
}

// Exact 4-pass 8-bit MSB radix Top-K selector in 1 KiB SRAM (`0` stack spills!).
// Handles arbitrary `index_topk` (including `2048`), k-pool expansion (`kpool = 4`),
// and mandatory visible tail selection (`always_select_tail = 1`).
// Grid: (num_rows, 1, 1), 256 threads (8 simdgroups).
kernel void ami_g27_dsa_select_topk(const device float *scores [[buffer(0)]],
                                    const device ulong *ancestor_masks [[buffer(1)]],
                                    device uint *out_indices [[buffer(2)]],
                                    device uint *out_counts [[buffer(3)]],
                                    constant G27DsaSelectTopKParams &p [[buffer(4)]],
                                    uint tg_x [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]]) {
    const uint tok = tg_x;
    if (tok >= p.num_rows) {
        return;
    }

    threadgroup atomic_uint tg_hist[256];
    threadgroup atomic_uint tg_count;
    threadgroup atomic_uint tg_tie_taken;
    threadgroup uint tg_meta[4]; // [0]=prefix, [1]=remaining, [2]=num_valid, [3]=path_len
    threadgroup uchar tg_path[64];

    if (tid == 0u) {
        atomic_store_explicit(&tg_count, 0u, memory_order_relaxed);
        tg_meta[3] = 0u;
        if (p.is_tree != 0u) {
            g27_build_tree_path(ancestor_masks, p.num_rows, tok, tg_path, tg_meta[3]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const device float *row_s = scores + ulong(tok) * p.num_candidates;
    device uint *row_out = out_indices + ulong(tok) * p.max_out_k;

    // Count valid (finite) candidates for this query row.
    uint local_valid = 0u;
    for (uint c = tid; c < p.num_candidates; c += 256u) {
        if (isfinite(row_s[c])) {
            local_valid++;
        }
    }
    if (local_valid > 0u) {
        atomic_fetch_add_explicit(&tg_count, local_valid, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        tg_meta[2] = atomic_load_explicit(&tg_count, memory_order_relaxed);
        atomic_store_explicit(&tg_count, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint num_valid = tg_meta[2];
    const uint kpool = max(p.kpool, 1u);
    const uint pool_budget = p.index_topk / kpool;
    const uint select_k = min(pool_budget, num_valid);

    if (select_k > 0u) {
        if (num_valid <= select_k) {
            // Fast path: emit ALL valid candidates directly.
            for (uint c = tid; c < p.num_candidates; c += 256u) {
                if (isfinite(row_s[c])) {
                    const uint slot = atomic_fetch_add_explicit(&tg_count, kpool, memory_order_relaxed);
                    if (kpool == 1u) {
                        if (slot < p.max_out_k) row_out[slot] = c;
                    } else {
                        for (uint j = 0u; j < kpool; ++j) {
                            if (slot + j < p.max_out_k) {
                                row_out[slot + j] = g27_map_seq_pos_to_kv_index(
                                    c * kpool + j, p.committed_kv_len, p.is_tree, tg_path);
                            }
                        }
                    }
                }
            }
        } else {
            // Exact 4-pass 8-bit MSB radix threshold selection in 1 KiB SRAM.
            if (tid == 0u) {
                tg_meta[0] = 0u;
                tg_meta[1] = select_k;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint pass = 0u; pass < 4u; ++pass) {
                const uint shift = 24u - pass * 8u;
                atomic_store_explicit(&tg_hist[tid], 0u, memory_order_relaxed);
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const uint cur_prefix = tg_meta[0];
                for (uint c = tid; c < p.num_candidates; c += 256u) {
                    const float s = row_s[c];
                    if (isfinite(s)) {
                        const uint key = g27_float_to_ordered_key(s);
                        const bool matches = (pass == 0u) || ((key >> (shift + 8u)) == cur_prefix);
                        if (matches) {
                            const uint bucket = (key >> shift) & 0xFFu;
                            atomic_fetch_add_explicit(&tg_hist[bucket], 1u, memory_order_relaxed);
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (tid == 0u) {
                    uint acc = 0u;
                    uint chosen = 0u;
                    const uint rem = tg_meta[1];
                    for (int b = 255; b >= 0; --b) {
                        const uint cnt = atomic_load_explicit(&tg_hist[uint(b)], memory_order_relaxed);
                        if (acc + cnt >= rem) {
                            chosen = uint(b);
                            tg_meta[1] = rem - acc;
                            break;
                        }
                        acc += cnt;
                    }
                    tg_meta[0] = (cur_prefix << 8u) | chosen;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            const uint thr_key = tg_meta[0];
            const uint tie_limit = tg_meta[1];
            if (tid == 0u) {
                atomic_store_explicit(&tg_tie_taken, 0u, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Emit strictly greater candidates first, then ties up to exact `select_k`.
            for (uint c = tid; c < p.num_candidates; c += 256u) {
                const float s = row_s[c];
                if (isfinite(s) && g27_float_to_ordered_key(s) > thr_key) {
                    const uint slot = atomic_fetch_add_explicit(&tg_count, kpool, memory_order_relaxed);
                    if (kpool == 1u) {
                        if (slot < p.max_out_k) row_out[slot] = c;
                    } else {
                        for (uint j = 0u; j < kpool; ++j) {
                            if (slot + j < p.max_out_k) {
                                row_out[slot + j] = g27_map_seq_pos_to_kv_index(
                                    c * kpool + j, p.committed_kv_len, p.is_tree, tg_path);
                            }
                        }
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint c = tid; c < p.num_candidates; c += 256u) {
                const float s = row_s[c];
                if (isfinite(s) && g27_float_to_ordered_key(s) == thr_key) {
                    const uint taken = atomic_fetch_add_explicit(&tg_tie_taken, 1u, memory_order_relaxed);
                    if (taken < tie_limit) {
                        const uint slot = atomic_fetch_add_explicit(&tg_count, kpool, memory_order_relaxed);
                        if (kpool == 1u) {
                            if (slot < p.max_out_k) row_out[slot] = c;
                        } else {
                            for (uint j = 0u; j < kpool; ++j) {
                                if (slot + j < p.max_out_k) {
                                    row_out[slot + j] = g27_map_seq_pos_to_kv_index(
                                        c * kpool + j, p.committed_kv_len, p.is_tree, tg_path);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Append visible incomplete tail tokens when `kpool > 1` and `always_select_tail != 0`.
    if (tid == 0u) {
        uint cur_cnt = min(atomic_load_explicit(&tg_count, memory_order_relaxed), p.max_out_k);
        if (kpool > 1u && p.always_select_tail != 0u) {
            const uint visible_count = (p.is_tree != 0u)
                ? (p.committed_kv_len + tg_meta[3])
                : (p.causal_base_pos + tok + 1u);
            const uint tail_cnt = visible_count % kpool;
            const uint tail_start = visible_count - tail_cnt;
            for (uint t = 0u; t < tail_cnt && cur_cnt < p.max_out_k; ++t) {
                row_out[cur_cnt++] = g27_map_seq_pos_to_kv_index(
                    tail_start + t, p.committed_kv_len, p.is_tree, tg_path);
            }
        }
        out_counts[tok] = cur_cnt;
        atomic_store_explicit(&tg_count, cur_cnt, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pad remaining slots with sentinel `0xFFFFFFFFu`.
    const uint final_cnt = atomic_load_explicit(&tg_count, memory_order_relaxed);
    for (uint i = final_cnt + tid; i < p.max_out_k; i += 256u) {
        row_out[i] = 0xFFFFFFFFu;
    }
}

// Sparse absorbed MLA attention over DSA-selected token indices.
// Supports BOTH `qk_rope_head_dim > 0` (`glm_moe_dsa`) and `qk_rope_head_dim == 0` (`NoPE MLA`, `glm5_next`),
// across decode, prefill, and `verifyTree` (reading `idx >= committed_kv_len` from tree scratch).
// Grid: (num_q_heads, num_rows, 1), 256 threads (8 simdgroups).
kernel void ami_g27_mla_sparse_attend(const device half *q_nope [[buffer(0)]],
                                      const device half *q_pe [[buffer(1)]],
                                      const device half *kv_cache [[buffer(2)]],
                                      const device half *kv_tree_latent [[buffer(3)]],
                                      const device half *kv_tree_rope [[buffer(4)]],
                                      const device uint *selected_indices [[buffer(5)]],
                                      const device uint *selected_counts [[buffer(6)]],
                                      device half *out [[buffer(7)]],
                                      constant G27MlaSparseAttnParams &p [[buffer(8)]],
                                      uint2 tg_pos [[threadgroup_position_in_grid]],
                                      uint tid [[thread_index_in_threadgroup]],
                                      uint sg_id [[simdgroup_index_in_threadgroup]],
                                      uint lane [[thread_index_in_simdgroup]]) {
    const uint q_head = tg_pos.x;
    const uint tok = tg_pos.y;
    if (q_head >= p.num_q_heads || tok >= p.num_rows) {
        return;
    }

    const uint d_lat = p.kv_lora_rank;
    const uint d_rope = p.qk_rope_head_dim;
    const uint kv_stride = d_lat + d_rope;
    const uint k_cnt = min(selected_counts[tok], p.max_out_k);
    const ulong qn_base = (ulong(tok) * p.num_q_heads + q_head) * d_lat;
    const ulong qp_base = (ulong(tok) * p.num_q_heads + q_head) * d_rope;
    const device uint *idx_row = selected_indices + ulong(tok) * p.max_out_k;
    device half *out_row = out + qn_base;

    if (k_cnt == 0u) {
        for (uint d = tid; d < d_lat; d += 256u) {
            out_row[d] = half(0.0f);
        }
        return;
    }

    // Stage query in threadgroup SRAM (up to 1024-dim latent + 256-dim RoPE = 5 KiB SRAM).
    threadgroup float tg_qn[1024];
    threadgroup float tg_qp[256];
    threadgroup float tg_red[8];

    if (d_lat <= 1024u) {
        for (uint d = tid; d < d_lat; d += 256u) {
            tg_qn[d] = float(q_nope[qn_base + d]);
        }
    }
    if (d_rope > 0u) {
        for (uint r = tid; r < d_rope; r += 256u) {
            tg_qp[r] = float(q_pe[qp_base + r]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (d_lat <= 1024u) {
        // Fast single-pass path (covers `kv_lora_rank = 512` on GLM-5.2/5.3/5.3-Flash!).
        // 4 scalar registers `o0..o3` per thread -> 0 stack spills, 1 KV read per selected token.
        float m_val = -INFINITY;
        float l_val = 0.0f;
        float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;

        const uint d0 = tid;
        const uint d1 = tid + 256u;
        const uint d2 = tid + 512u;
        const uint d3 = tid + 768u;
        const float q0 = (d0 < d_lat) ? tg_qn[d0] : 0.0f;
        const float q1 = (d1 < d_lat) ? tg_qn[d1] : 0.0f;
        const float q2 = (d2 < d_lat) ? tg_qn[d2] : 0.0f;
        const float q3 = (d3 < d_lat) ? tg_qn[d3] : 0.0f;
        const float qr = (d_rope > 0u && tid < d_rope) ? tg_qp[tid] : 0.0f;

        for (uint si = 0u; si < k_cnt; ++si) {
            const uint kv_idx = idx_row[si];
            if (kv_idx == 0xFFFFFFFFu) {
                continue;
            }
            const bool is_tree_tok = (kv_idx >= p.committed_kv_len);
            const device half *k_lat_ptr = is_tree_tok
                ? (kv_tree_latent + ulong(kv_idx - p.committed_kv_len) * d_lat)
                : (kv_cache + ulong(kv_idx) * kv_stride);
            const device half *k_rope_ptr = is_tree_tok
                ? (kv_tree_rope + ulong(kv_idx - p.committed_kv_len) * d_rope)
                : (kv_cache + ulong(kv_idx) * kv_stride + d_lat);

            const float k0 = (d0 < d_lat) ? float(k_lat_ptr[d0]) : 0.0f;
            const float k1 = (d1 < d_lat) ? float(k_lat_ptr[d1]) : 0.0f;
            const float k2 = (d2 < d_lat) ? float(k_lat_ptr[d2]) : 0.0f;
            const float k3 = (d3 < d_lat) ? float(k_lat_ptr[d3]) : 0.0f;

            float part = fma(q0, k0, fma(q1, k1, fma(q2, k2, q3 * k3)));
            if (d_rope > 0u && tid < d_rope) {
                part = fma(qr, float(k_rope_ptr[tid]), part);
            }

            const float sg_dot = simd_sum(part);
            if (lane == 0u) tg_red[sg_id] = sg_dot;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float score = (((tg_red[0] + tg_red[1]) + (tg_red[2] + tg_red[3])) +
                                 ((tg_red[4] + tg_red[5]) + (tg_red[6] + tg_red[7]))) * p.scale;

            const float m_new = max(m_val, score);
            const float alpha = (m_val == -INFINITY) ? 0.0f : exp(m_val - m_new);
            const float w = exp(score - m_new);
            l_val = fma(l_val, alpha, w);
            m_val = m_new;

            o0 = fma(w, k0, o0 * alpha);
            o1 = fma(w, k1, o1 * alpha);
            o2 = fma(w, k2, o2 * alpha);
            o3 = fma(w, k3, o3 * alpha);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        const float inv_l = (l_val > 0.0f) ? (1.0f / l_val) : 0.0f;
        if (d0 < d_lat) out_row[d0] = half(o0 * inv_l);
        if (d1 < d_lat) out_row[d1] = half(o1 * inv_l);
        if (d2 < d_lat) out_row[d2] = half(o2 * inv_l);
        if (d3 < d_lat) out_row[d3] = half(o3 * inv_l);
    } else {
        // General tiled path for `kv_lora_rank > 1024` (0 stack spills).
        for (uint d_base = 0u; d_base < d_lat; d_base += 256u) {
            const uint d_cur = d_base + tid;
            float m_val = -INFINITY;
            float l_val = 0.0f;
            float o_acc = 0.0f;

            for (uint si = 0u; si < k_cnt; ++si) {
                const uint kv_idx = idx_row[si];
                if (kv_idx == 0xFFFFFFFFu) {
                    continue;
                }
                const bool is_tree_tok = (kv_idx >= p.committed_kv_len);
                const device half *k_lat_ptr = is_tree_tok
                    ? (kv_tree_latent + ulong(kv_idx - p.committed_kv_len) * d_lat)
                    : (kv_cache + ulong(kv_idx) * kv_stride);
                const device half *k_rope_ptr = is_tree_tok
                    ? (kv_tree_rope + ulong(kv_idx - p.committed_kv_len) * d_rope)
                    : (kv_cache + ulong(kv_idx) * kv_stride + d_lat);

                float part = 0.0f;
                for (uint d = tid; d < d_lat; d += 256u) {
                    part = fma(float(q_nope[qn_base + d]), float(k_lat_ptr[d]), part);
                }
                if (d_rope > 0u && tid < d_rope) {
                    part = fma(tg_qp[tid], float(k_rope_ptr[tid]), part);
                }
                const float sg_dot = simd_sum(part);
                if (lane == 0u) tg_red[sg_id] = sg_dot;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                const float score = (((tg_red[0] + tg_red[1]) + (tg_red[2] + tg_red[3])) +
                                     ((tg_red[4] + tg_red[5]) + (tg_red[6] + tg_red[7]))) * p.scale;

                const float m_new = max(m_val, score);
                const float alpha = (m_val == -INFINITY) ? 0.0f : exp(m_val - m_new);
                const float w = exp(score - m_new);
                l_val = fma(l_val, alpha, w);
                m_val = m_new;

                const float kv_d = (d_cur < d_lat) ? float(k_lat_ptr[d_cur]) : 0.0f;
                o_acc = fma(w, kv_d, o_acc * alpha);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            const float inv_l = (l_val > 0.0f) ? (1.0f / l_val) : 0.0f;
            if (d_cur < d_lat) {
                out_row[d_cur] = half(o_acc * inv_l);
            }
        }
    }
}

// ===========================================================================
// M9.3 (`G28`): Full-Causal Absorbed MLA + Head-Wise/Element-Wise Output Gate
// (`Ling-3.1-flash-1019` `bailing_hybrid`: `64H`, `kv_lora_rank=512`, `qk_rope=64`,
// `v_head_dim=128`, `g_proj` sigmoid gate before `o_proj`). Zero stack spills.
// ===========================================================================

struct G28MlaFullAttnParams {
    uint num_rows;
    uint num_q_heads;
    uint kv_lora_rank;
    uint qk_rope_head_dim;
    uint committed_kv_len;
    uint causal_base_pos;
    uint is_tree;
    float scale;
};

struct G28MlaOutGateParams {
    uint num_rows;
    uint num_heads;
    uint v_head_dim;
    uint head_wise; // 1 = gate [num_rows, num_heads]; 0 = gate [num_rows, num_heads * v_head_dim]
};

static_assert(sizeof(G28MlaFullAttnParams) == 32, "G28MlaFullAttnParams must be 32 bytes");
static_assert(sizeof(G28MlaOutGateParams) == 16, "G28MlaOutGateParams must be 16 bytes");

// Full-causal absorbed MLA attention without sparse indexer.
// Supports decode (`num_rows=1`), wide prefill (`is_tree=0`), and speculative tree
// verification (`is_tree=1` with `ancestor_masks`), plus both `d_rope > 0` and `d_rope == 0`.
// Grid: (num_q_heads, num_rows, 1), 256 threads (8 simdgroups). Zero stack spills.
kernel void ami_g28_mla_full_attend(const device half *q_nope [[buffer(0)]],
                                    const device half *q_pe [[buffer(1)]],
                                    const device half *kv_cache [[buffer(2)]],
                                    const device half *kv_tree_latent [[buffer(3)]],
                                    const device half *kv_tree_rope [[buffer(4)]],
                                    const device ulong *ancestor_masks [[buffer(5)]],
                                    device half *out [[buffer(6)]],
                                    constant G28MlaFullAttnParams &p [[buffer(7)]],
                                    uint2 tg_pos [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]],
                                    uint sg_id [[simdgroup_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]]) {
    const uint q_head = tg_pos.x;
    const uint tok = tg_pos.y;
    if (q_head >= p.num_q_heads || tok >= p.num_rows) {
        return;
    }

    const uint d_lat = p.kv_lora_rank;
    const uint d_rope = p.qk_rope_head_dim;
    const uint kv_stride = d_lat + d_rope;
    const ulong qn_base = (ulong(tok) * p.num_q_heads + q_head) * d_lat;
    const ulong qp_base = (ulong(tok) * p.num_q_heads + q_head) * d_rope;
    device half *out_row = out + qn_base;

    const uint total_steps = (p.is_tree != 0u)
        ? (p.committed_kv_len + p.num_rows)
        : min(p.causal_base_pos + tok + 1u, p.committed_kv_len);
    const ulong tree_mask = (p.is_tree != 0u) ? ancestor_masks[tok] : 0ul;

    if (total_steps == 0u) {
        for (uint d = tid; d < d_lat; d += 256u) {
            out_row[d] = half(0.0f);
        }
        return;
    }

    threadgroup float tg_qn[1024];
    threadgroup float tg_qp[256];
    threadgroup float tg_red[8];

    if (d_lat <= 1024u) {
        for (uint d = tid; d < d_lat; d += 256u) {
            tg_qn[d] = float(q_nope[qn_base + d]);
        }
    }
    if (d_rope > 0u) {
        for (uint r = tid; r < d_rope; r += 256u) {
            tg_qp[r] = float(q_pe[qp_base + r]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (d_lat <= 1024u) {
        float m_val = -INFINITY;
        float l_val = 0.0f;
        float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;

        const uint d0 = tid;
        const uint d1 = tid + 256u;
        const uint d2 = tid + 512u;
        const uint d3 = tid + 768u;
        const float q0 = (d0 < d_lat) ? tg_qn[d0] : 0.0f;
        const float q1 = (d1 < d_lat) ? tg_qn[d1] : 0.0f;
        const float q2 = (d2 < d_lat) ? tg_qn[d2] : 0.0f;
        const float q3 = (d3 < d_lat) ? tg_qn[d3] : 0.0f;
        const float qr = (d_rope > 0u && tid < d_rope) ? tg_qp[tid] : 0.0f;

        for (uint step = 0u; step < total_steps; ++step) {
            const bool is_tree_tok = (p.is_tree != 0u && step >= p.committed_kv_len);
            if (is_tree_tok) {
                const uint j = step - p.committed_kv_len;
                if (((tree_mask >> j) & 1ul) == 0ul) {
                    continue;
                }
            }
            const device half *k_lat_ptr = is_tree_tok
                ? (kv_tree_latent + ulong(step - p.committed_kv_len) * d_lat)
                : (kv_cache + ulong(step) * kv_stride);
            const device half *k_rope_ptr = is_tree_tok
                ? (kv_tree_rope + ulong(step - p.committed_kv_len) * d_rope)
                : (kv_cache + ulong(step) * kv_stride + d_lat);

            const float k0 = (d0 < d_lat) ? float(k_lat_ptr[d0]) : 0.0f;
            const float k1 = (d1 < d_lat) ? float(k_lat_ptr[d1]) : 0.0f;
            const float k2 = (d2 < d_lat) ? float(k_lat_ptr[d2]) : 0.0f;
            const float k3 = (d3 < d_lat) ? float(k_lat_ptr[d3]) : 0.0f;

            float part = fma(q0, k0, fma(q1, k1, fma(q2, k2, q3 * k3)));
            if (d_rope > 0u && tid < d_rope) {
                part = fma(qr, float(k_rope_ptr[tid]), part);
            }

            const float sg_dot = simd_sum(part);
            if (lane == 0u) tg_red[sg_id] = sg_dot;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float score = (((tg_red[0] + tg_red[1]) + (tg_red[2] + tg_red[3])) +
                                 ((tg_red[4] + tg_red[5]) + (tg_red[6] + tg_red[7]))) * p.scale;

            const float m_new = max(m_val, score);
            const float alpha = (m_val == -INFINITY) ? 0.0f : exp(m_val - m_new);
            const float w = exp(score - m_new);
            l_val = fma(l_val, alpha, w);
            m_val = m_new;

            o0 = fma(w, k0, o0 * alpha);
            o1 = fma(w, k1, o1 * alpha);
            o2 = fma(w, k2, o2 * alpha);
            o3 = fma(w, k3, o3 * alpha);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        const float inv_l = (l_val > 0.0f) ? (1.0f / l_val) : 0.0f;
        if (d0 < d_lat) out_row[d0] = half(o0 * inv_l);
        if (d1 < d_lat) out_row[d1] = half(o1 * inv_l);
        if (d2 < d_lat) out_row[d2] = half(o2 * inv_l);
        if (d3 < d_lat) out_row[d3] = half(o3 * inv_l);
    } else {
        for (uint d_base = 0u; d_base < d_lat; d_base += 256u) {
            const uint d_cur = d_base + tid;
            float m_val = -INFINITY;
            float l_val = 0.0f;
            float o_acc = 0.0f;

            for (uint step = 0u; step < total_steps; ++step) {
                const bool is_tree_tok = (p.is_tree != 0u && step >= p.committed_kv_len);
                if (is_tree_tok) {
                    const uint j = step - p.committed_kv_len;
                    if (((tree_mask >> j) & 1ul) == 0ul) {
                        continue;
                    }
                }
                const device half *k_lat_ptr = is_tree_tok
                    ? (kv_tree_latent + ulong(step - p.committed_kv_len) * d_lat)
                    : (kv_cache + ulong(step) * kv_stride);
                const device half *k_rope_ptr = is_tree_tok
                    ? (kv_tree_rope + ulong(step - p.committed_kv_len) * d_rope)
                    : (kv_cache + ulong(step) * kv_stride + d_lat);

                float part = 0.0f;
                for (uint d = tid; d < d_lat; d += 256u) {
                    part = fma(float(q_nope[qn_base + d]), float(k_lat_ptr[d]), part);
                }
                if (d_rope > 0u && tid < d_rope) {
                    part = fma(tg_qp[tid], float(k_rope_ptr[tid]), part);
                }
                const float sg_dot = simd_sum(part);
                if (lane == 0u) tg_red[sg_id] = sg_dot;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                const float score = (((tg_red[0] + tg_red[1]) + (tg_red[2] + tg_red[3])) +
                                     ((tg_red[4] + tg_red[5]) + (tg_red[6] + tg_red[7]))) * p.scale;

                const float m_new = max(m_val, score);
                const float alpha = (m_val == -INFINITY) ? 0.0f : exp(m_val - m_new);
                const float w = exp(score - m_new);
                l_val = fma(l_val, alpha, w);
                m_val = m_new;

                const float kv_d = (d_cur < d_lat) ? float(k_lat_ptr[d_cur]) : 0.0f;
                o_acc = fma(w, kv_d, o_acc * alpha);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            const float inv_l = (l_val > 0.0f) ? (1.0f / l_val) : 0.0f;
            if (d_cur < d_lat) {
                out_row[d_cur] = half(o_acc * inv_l);
            }
        }
    }
}

// In-place head-wise or element-wise sigmoid output gate applied to MLA output
// before `self_attn.o_proj` (`BailingMoeV3MultiLatentAttention.g_proj`).
// Grid: ceil((num_rows * num_heads * v_head_dim) / 256), 256 threads. Zero stack spills.
kernel void ami_g28_mla_out_gate(device half *attn_out [[buffer(0)]],
                                 const device half *gate [[buffer(1)]],
                                 constant G28MlaOutGateParams &p [[buffer(2)]],
                                 uint gid [[thread_position_in_grid]]) {
    const ulong total = ulong(p.num_rows) * p.num_heads * p.v_head_dim;
    if (ulong(gid) >= total) {
        return;
    }
    const ulong gate_idx = (p.head_wise != 0u) ? (ulong(gid) / p.v_head_dim) : ulong(gid);
    const float g_val = hybrid_sigmoid(float(gate[gate_idx]));
    attn_out[gid] = half(float(attn_out[gid]) * g_val);
}

// ===========================================================================
// Section J: M9.4 — Real-HF `Qwen3.8-Flash-Next` (`qwen4_exp`) &
//            `DeepSeek-V4.1-Flash-1017` (`deepseek_v41`) Metal 4 Operators
// Zero dynamic stack arrays (`-Wframe-larger-than=0`), 0 stack spills.
// ===========================================================================

struct M94HcNormParams {
    uint m;
    uint num_groups;
    uint group_size;
    uint w_dtype;     // 0 = f16, 1 = bf16, 2 = f32
    uint unit_offset; // 1 = (1.0 + w), 0 = plain w
    float rms_eps;
    uint _pad0;
    uint _pad1;
};

struct M94ScaleSiluParams {
    uint count;
    float scale;
    uint _pad0;
    uint _pad1;
};

struct M94HcMixParams {
    uint m;
    uint hc_count;
    uint hidden_size;
    uint has_inj;
    float inv_hc_count;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

struct M94HcInjectParams {
    uint m;
    uint hc_count;
    uint hidden_size;
    uint _pad0;
};

struct M94PleGateParams {
    uint m;
    uint hc_count;
    uint hidden_size;
    uint norm_dtype; // 0 = f16, 1 = bf16, 2 = f32
    float rms_eps;
    float inv_sqrt_h;
    uint _pad0;
    uint _pad1;
};

struct M94PleConvParams {
    uint m;
    uint channels;
    uint kernel_size; // <= 8 (real-HF: 4)
    uint dilation;    // real-HF: ngram_size = 3
    uint state_len;   // (kernel_size - 1) * dilation (real-HF: 9)
    uint w_dtype;     // 0 = f16, 1 = bf16, 2 = f32
    uint is_tree;     // 0 = sequential, 1 = ancestor_masks tree
    uint _pad0;
};

struct M94PleConvCommitParams {
    uint channels;
    uint state_len;
    uint accepted_count;
    uint _pad0;
    ushort inline_nodes[64];
};

struct M94QsaIdxPrepParams {
    uint num_tokens;
    uint idx_n_heads;
    uint idx_head_dim;
    uint rotary_dim;
    uint start_pos;
    uint use_positions_3d;
    uint write_raw_to_cache; // 1 = raw_k_dst[start_pos + tok], 0 = raw_k_dst[tok]
    uint mrope_s0;
    uint mrope_s1;
    uint mrope_s2;
    uint norm_dtype;     // 0 = f16, 1 = bf16, 2 = f32
    uint unit_offset;    // 1 = (1 + w)
    float rms_eps;
    float rope_theta;
    float yarn_factor;
    float yarn_mscale;
    float yarn_beta_fast;
    float yarn_beta_slow;
    uint original_max_pos;
    uint _pad0;
};

struct M94QsaIdxPoolParams {
    uint start_block;
    uint num_blocks;
    uint compress_ratio;
    uint idx_head_dim;
    uint rotary_dim;
    uint mrope_s0;
    uint mrope_s1;
    uint mrope_s2;
    uint norm_dtype;
    uint unit_offset;
    float rms_eps;
    float rope_theta;
    float yarn_factor;
    float yarn_mscale;
    float yarn_beta_fast;
    float yarn_beta_slow;
    uint original_max_pos;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

struct M94QsaIdxSelectParams {
    uint num_nodes;
    uint idx_n_heads;
    uint idx_head_dim;
    uint num_kv_heads;
    uint kv_len;             // committed KV tokens before current nodes
    uint compress_ratio;
    uint block_topk;
    uint top_k_blocks_cap;   // QsaIndexBuffer capacity (<= 64)
    uint causal_q_start_pos; // 0xFFFFFFFF = tree/decode over kv_len; else wide causal
    float inv_sqrt_d;
    uint _pad0;
    uint _pad1;
};

struct M94MhcInitParams {
    uint m;
    uint hc_mult;
    uint _pad0;
    uint _pad1;
};

struct M94MhcCollapseParams {
    uint m;
    uint hidden_size;
    uint has_norm_w;
    uint norm_dtype; // 0 = f16, 1 = bf16, 2 = f32
    float rms_eps;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

struct M94RopeComplexParams {
    uint num_tokens;
    uint num_heads;
    uint head_stride;
    uint rot_offset;
    uint rot_dim;
    uint base_pos;
    uint use_pos_buf;
    uint inverse;          // 0 = forward, 1 = conjugate inverse
    uint pos_stride;       // > 1 for compressed group positions (e.g. ratio)
    uint original_seq_len; // 0 = standard RoPE, > 0 = YaRN frequency ramp
    float rope_theta;
    float rope_factor;
    float beta_fast;
    float beta_slow;
    uint _pad0;
    uint _pad1;
};

struct M94DsCompressorParams {
    uint num_tokens;
    uint start_pos;
    uint head_dim;
    uint ratio;
    uint gate_dim;   // == head_dim (real-HF) or == ratio (synth)
    uint norm_dtype; // 0 = f16, 1 = bf16, 2 = f32
    float rms_eps;
    uint write_state; // 1 = commit partial group state
    uint out_count;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

struct M94DsIndexerParams {
    uint num_tokens;
    uint start_pos;
    uint idx_n_heads;
    uint idx_head_dim;
    uint ratio;
    uint total_comp_len;
    uint index_topk;
    uint offset;
    uint is_candidate_source;
    uint uses_candidates;
    uint cand_topk_blocks;
    uint cand_block_size;
    uint cand_num_blocks;
    float weight_scale;
    uint is_causal_prefill;
    uint _pad0;
};

struct M94DsLatentAttnParams {
    uint num_q;
    uint n_heads;
    uint q_head_dim;
    uint kv_dim;
    uint v_head_dim;
    uint nope_dim;
    uint rope_dim;
    uint window_size;
    uint start_pos;
    uint has_compress;
    uint compress_topk;
    uint has_sink;
    uint is_tree;
    uint apply_inv_rope;
    uint original_seq_len;
    float scale;
    float rope_theta;
    float rope_factor;
    float beta_fast;
    float beta_slow;
};

struct M94DsEngramParams {
    uint m;
    uint hidden_size;
    uint wkv_out_dim; // 5*H (real-HF) or H (synth)
    uint emb_dim;
    uint k_dim;       // H (real-HF) or emb_dim (synth)
    uint qk_dtype;    // 0 = f16, 1 = bf16, 2 = f32
    float rms_eps;
    float inv_sqrt_d;
};

static_assert(sizeof(M94HcNormParams) == 32, "M94HcNormParams must be 32 bytes");
static_assert(sizeof(M94ScaleSiluParams) == 16, "M94ScaleSiluParams must be 16 bytes");
static_assert(sizeof(M94HcMixParams) == 32, "M94HcMixParams must be 32 bytes");
static_assert(sizeof(M94HcInjectParams) == 16, "M94HcInjectParams must be 16 bytes");
static_assert(sizeof(M94PleGateParams) == 32, "M94PleGateParams must be 32 bytes");
static_assert(sizeof(M94PleConvParams) == 32, "M94PleConvParams must be 32 bytes");
static_assert(sizeof(M94PleConvCommitParams) == 144, "M94PleConvCommitParams must be 144 bytes");
static_assert(sizeof(M94QsaIdxPrepParams) == 80, "M94QsaIdxPrepParams must be 80 bytes");
static_assert(sizeof(M94QsaIdxPoolParams) == 80, "M94QsaIdxPoolParams must be 80 bytes");
static_assert(sizeof(M94QsaIdxSelectParams) == 48, "M94QsaIdxSelectParams must be 48 bytes");
static_assert(sizeof(M94MhcInitParams) == 16, "M94MhcInitParams must be 16 bytes");
static_assert(sizeof(M94MhcCollapseParams) == 32, "M94MhcCollapseParams must be 32 bytes");
static_assert(sizeof(M94RopeComplexParams) == 64, "M94RopeComplexParams must be 64 bytes");
static_assert(sizeof(M94DsCompressorParams) == 48, "M94DsCompressorParams must be 48 bytes");
static_assert(sizeof(M94DsIndexerParams) == 64, "M94DsIndexerParams must be 64 bytes");
static_assert(sizeof(M94DsLatentAttnParams) == 80, "M94DsLatentAttnParams must be 80 bytes");
static_assert(sizeof(M94DsEngramParams) == 32, "M94DsEngramParams must be 32 bytes");

// Exact DeepSeek-V4.1 `precompute_freqs_cis` frequency helper (with optional YaRN ramp).
static inline float m94_ds_inv_freq(uint j,
                                    uint rot_dim,
                                    float rope_theta,
                                    uint original_seq_len,
                                    float rope_factor,
                                    float beta_fast,
                                    float beta_slow) {
    const float dim_f = float(rot_dim);
    float freq = exp(-log(rope_theta) * (float(2u * j) / dim_f));
    if (original_seq_len > 0u && rope_factor > 1.0f) {
        const float two_pi = 6.2831853071795864769f;
        const float two_log_base = 2.0f * log(rope_theta);
        const float orig_f = float(original_seq_len);
        const float low_raw = dim_f * log(orig_f / (beta_fast * two_pi)) / two_log_base;
        const float high_raw = dim_f * log(orig_f / (beta_slow * two_pi)) / two_log_base;
        const float low = max(floor(low_raw), 0.0f);
        const float high = min(ceil(high_raw), dim_f - 1.0f);
        const float ramp = clamp((float(j) - low) / max(high - low, 1.0e-3f), 0.0f, 1.0f);
        const float smooth = 1.0f - ramp;
        freq = (freq / rope_factor) * (1.0f - smooth) + freq * smooth;
    }
    return freq;
}

// 1. Grouped RMSNorm (`(1.0 + w)` unit-offset when `unit_offset == 1`).
// Grid: (num_groups, m, 1) threadgroups, 256 threads.
kernel void ami_qwen4_hc_norm(const device half *x [[buffer(0)]],
                              const device uchar *w [[buffer(1)]],
                              device half *y [[buffer(2)]],
                              constant M94HcNormParams &p [[buffer(3)]],
                              uint2 tg_pos [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]],
                              uint sg_id [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
    const uint grp = tg_pos.x;
    const uint tok = tg_pos.y;
    if (grp >= p.num_groups || tok >= p.m || p.group_size == 0u) {
        return;
    }
    const ulong total_dim = ulong(p.num_groups) * p.group_size;
    const ulong grp_off = ulong(grp) * p.group_size;
    const ulong base = ulong(tok) * total_dim + grp_off;

    threadgroup float tg_red[8];
    float sq = 0.0f;
    for (uint d = tid; d < p.group_size; d += 256u) {
        const float v = float(x[base + d]);
        sq = fma(v, v, sq);
    }
    const float sg_sq = simd_sum(sq);
    if (lane == 0u) tg_red[sg_id] = sg_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total_sq = ((tg_red[0] + tg_red[1]) + (tg_red[2] + tg_red[3])) +
                           ((tg_red[4] + tg_red[5]) + (tg_red[6] + tg_red[7]));
    const float inv_rms = rsqrt(total_sq / float(p.group_size) + p.rms_eps);
    const float w_add = (p.unit_offset != 0u) ? 1.0f : 0.0f;

    for (uint d = tid; d < p.group_size; d += 256u) {
        const float v = float(x[base + d]);
        const float wt = w_add + g27_load_scalar_f32(w, grp_off + d, p.w_dtype);
        y[base + d] = half((v * inv_rms) * wt);
    }
}

// 2. In-place `silu(x * scale)` (`scale = 1.0 / hc_count = 0.25`).
// Grid: ceil(count / 256), 256 threads.
kernel void ami_qwen4_hc_scale_silu(device half *x [[buffer(0)]],
                                    constant M94ScaleSiluParams &p [[buffer(1)]],
                                    uint gid [[thread_position_in_grid]]) {
    if (gid >= p.count) return;
    x[gid] = half(hybrid_fast_silu(float(x[gid]) * p.scale));
}

// 3. Fused Qwen3.8 Hyper-Connection pre-mix reduction + optional injection gate:
//    `mixed[tok, d] = inv_hc * sum_b(sigmoid(w_mix[tok, b*H + d]) * h_normed[tok, b*H + d])`
//    `inj_out[tok, b] = 2.0 * sigmoid(inj_raw[tok, b] * inv_hc)` (f32).
// Grid: (m, 1, 1) threadgroups, 256 threads.
kernel void ami_qwen4_hc_mix_reduce(const device half *h_normed [[buffer(0)]],
                                    const device half *w_mix [[buffer(1)]],
                                    const device half *inj_raw [[buffer(2)]],
                                    device half *mixed_out [[buffer(3)]],
                                    device float *inj_out [[buffer(4)]],
                                    constant M94HcMixParams &p [[buffer(5)]],
                                    uint tg_x [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]]) {
    const uint tok = tg_x;
    if (tok >= p.m) return;
    const ulong h = ulong(p.hidden_size);
    const ulong hc = ulong(p.hc_count);
    const ulong tok_hc_base = ulong(tok) * hc * h;
    const ulong tok_h_base = ulong(tok) * h;

    for (uint d = tid; d < p.hidden_size; d += 256u) {
        float acc = 0.0f;
        for (uint b = 0u; b < p.hc_count; ++b) {
            const ulong idx = tok_hc_base + ulong(b) * h + d;
            acc = fma(hybrid_sigmoid(float(w_mix[idx])), float(h_normed[idx]), acc);
        }
        mixed_out[tok_h_base + d] = half(acc * p.inv_hc_count);
    }
    if (p.has_inj != 0u && tid < p.hc_count) {
        const ulong i_idx = ulong(tok) * hc + tid;
        inj_out[i_idx] = 2.0f * hybrid_sigmoid(float(inj_raw[i_idx]) * p.inv_hc_count);
    }
}

// 4. Qwen3.8 Hyper-Connection post-inject: `h[tok, b*H + d] += inj[tok, b] * sub_out[tok, d]`.
// Grid: (m, 1, 1) threadgroups, 256 threads.
kernel void ami_qwen4_hc_inject(device half *h_streams [[buffer(0)]],
                                const device half *sub_out [[buffer(1)]],
                                const device float *inj [[buffer(2)]],
                                constant M94HcInjectParams &p [[buffer(3)]],
                                uint tg_x [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]]) {
    const uint tok = tg_x;
    if (tok >= p.m) return;
    const ulong h = ulong(p.hidden_size);
    const ulong hc = ulong(p.hc_count);
    const ulong tok_hc_base = ulong(tok) * hc * h;
    const ulong tok_h_base = ulong(tok) * h;
    const ulong inj_base = ulong(tok) * hc;

    for (uint d = tid; d < p.hidden_size; d += 256u) {
        const float y = float(sub_out[tok_h_base + d]);
        for (uint b = 0u; b < p.hc_count; ++b) {
            const ulong idx = tok_hc_base + ulong(b) * h + d;
            h_streams[idx] = half(fma(inj[inj_base + b], y, float(h_streams[idx])));
        }
    }
}

// 5. Fused `Qwen4ExpTextPLELayer` stream gating + `norm_conv`:
//    k = norm_key(k_raw), q = norm_query(h_streams),
//    dot = sum_d(k * q) / sqrt(H),
//    gate = sigmoid(copysign(sqrt(max(|dot|, 1e-6)), dot)),
//    gv = gate * value, gv_normed = norm_conv(gv).
// Grid: (hc_count, m, 1) threadgroups, 256 threads.
kernel void ami_qwen4_ple_gate(const device half *h_streams [[buffer(0)]],
                               const device half *k_raw [[buffer(1)]],
                               const device half *value_in [[buffer(2)]],
                               const device uchar *norm_key_w [[buffer(3)]],
                               const device uchar *norm_query_w [[buffer(4)]],
                               const device uchar *norm_conv_w [[buffer(5)]],
                               device half *gv_out [[buffer(6)]],
                               device half *gv_normed_out [[buffer(7)]],
                               constant M94PleGateParams &p [[buffer(8)]],
                               uint2 tg_pos [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]],
                               uint sg_id [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
    const uint b = tg_pos.x;
    const uint tok = tg_pos.y;
    if (b >= p.hc_count || tok >= p.m || p.hidden_size == 0u) return;

    const ulong h = ulong(p.hidden_size);
    const ulong total_dim = ulong(p.hc_count) * h;
    const ulong grp_off = ulong(b) * h;
    const ulong base = ulong(tok) * total_dim + grp_off;
    const ulong v_base = ulong(tok) * h;

    threadgroup float tg_red_a[8];
    threadgroup float tg_red_b[8];

    // Pass 1: RMS statistics for `k_raw[b]` and `h_streams[b]`.
    float sq_k = 0.0f;
    float sq_q = 0.0f;
    for (uint d = tid; d < p.hidden_size; d += 256u) {
        const float kv = float(k_raw[base + d]);
        const float qv = float(h_streams[base + d]);
        sq_k = fma(kv, kv, sq_k);
        sq_q = fma(qv, qv, sq_q);
    }
    const float sg_k = simd_sum(sq_k);
    const float sg_q = simd_sum(sq_q);
    if (lane == 0u) {
        tg_red_a[sg_id] = sg_k;
        tg_red_b[sg_id] = sg_q;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float tot_k = ((tg_red_a[0] + tg_red_a[1]) + (tg_red_a[2] + tg_red_a[3])) +
                        ((tg_red_a[4] + tg_red_a[5]) + (tg_red_a[6] + tg_red_a[7]));
    const float tot_q = ((tg_red_b[0] + tg_red_b[1]) + (tg_red_b[2] + tg_red_b[3])) +
                        ((tg_red_b[4] + tg_red_b[5]) + (tg_red_b[6] + tg_red_b[7]));
    const float inv_rms_k = rsqrt(tot_k / float(p.hidden_size) + p.rms_eps);
    const float inv_rms_q = rsqrt(tot_q / float(p.hidden_size) + p.rms_eps);

    // Pass 2: dot product `(norm_key(k) * norm_query(q))` and `sum_sq(value)` (since `gv = gate * value`).
    float dot_part = 0.0f;
    float sq_v = 0.0f;
    for (uint d = tid; d < p.hidden_size; d += 256u) {
        const float kw = 1.0f + g27_load_scalar_f32(norm_key_w, grp_off + d, p.norm_dtype);
        const float qw = 1.0f + g27_load_scalar_f32(norm_query_w, grp_off + d, p.norm_dtype);
        const float kn = (float(k_raw[base + d]) * inv_rms_k) * kw;
        const float qn = (float(h_streams[base + d]) * inv_rms_q) * qw;
        dot_part = fma(kn, qn, dot_part);
        const float vv = float(value_in[v_base + d]);
        sq_v = fma(vv, vv, sq_v);
    }
    const float sg_dot = simd_sum(dot_part);
    const float sg_v = simd_sum(sq_v);
    if (lane == 0u) {
        tg_red_a[sg_id] = sg_dot;
        tg_red_b[sg_id] = sg_v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float dot_val = (((tg_red_a[0] + tg_red_a[1]) + (tg_red_a[2] + tg_red_a[3])) +
                           ((tg_red_a[4] + tg_red_a[5]) + (tg_red_a[6] + tg_red_a[7]))) * p.inv_sqrt_h;
    const float tot_v = ((tg_red_b[0] + tg_red_b[1]) + (tg_red_b[2] + tg_red_b[3])) +
                        ((tg_red_b[4] + tg_red_b[5]) + (tg_red_b[6] + tg_red_b[7]));
    const float signed_sqrt = copysign(sqrt(max(fabs(dot_val), 1.0e-6f)), dot_val);
    const float gate = hybrid_sigmoid(signed_sqrt);
    const float inv_rms_gv = rsqrt((tot_v * gate * gate) / float(p.hidden_size) + p.rms_eps);

    // Pass 3: write `gv` and `norm_conv(gv)`.
    for (uint d = tid; d < p.hidden_size; d += 256u) {
        const float gv = gate * float(value_in[v_base + d]);
        const float cw = 1.0f + g27_load_scalar_f32(norm_conv_w, grp_off + d, p.norm_dtype);
        gv_out[base + d] = half(gv);
        gv_normed_out[base + d] = half((gv * inv_rms_gv) * cw);
    }
}

// 6. Causal depthwise dilated Conv1D (`K <= 8`, `dilation = 3`) + SiLU + residual add into `h_streams`.
//    `conv_state_in` layout: `[state_len, channels]` oldest-first (`state_len = (K - 1) * dilation`).
// Grid: (ceil(channels / 256), m, 1) threadgroups, 256 threads.
kernel void ami_qwen4_ple_dilated_conv_add(device half *h_streams [[buffer(0)]],
                                           const device half *gv_in [[buffer(1)]],
                                           const device half *gv_normed [[buffer(2)]],
                                           const device uchar *conv_w [[buffer(3)]],
                                           const device half *conv_state_in [[buffer(4)]],
                                           const device ulong *ancestor_masks [[buffer(5)]],
                                           constant M94PleConvParams &p [[buffer(6)]],
                                           uint2 gid [[thread_position_in_grid]]) {
    const uint c = gid.x;
    const uint tok = gid.y;
    if (c >= p.channels || tok >= p.m) return;

    const ulong stride = ulong(p.channels);
    const uint ksz = p.kernel_size;
    const uint dil = p.dilation;
    const uint slen = p.state_len;

    float conv_acc = 0.0f;
    if (p.is_tree == 0u) {
        // Sequential chunk `[0..m)` appended after `conv_state_in[0..slen)`.
        for (uint tap = 0u; tap < ksz; ++tap) {
            const uint back = (ksz - 1u - tap) * dil;
            float x_tap = 0.0f;
            if (back <= tok) {
                x_tap = float(gv_normed[ulong(tok - back) * stride + c]);
            } else {
                const uint hist_back = back - tok; // 1..slen
                if (hist_back <= slen) {
                    x_tap = float(conv_state_in[ulong(slen - hist_back) * stride + c]);
                }
            }
            const float w_tap = g27_load_scalar_f32(conv_w, ulong(c) * ksz + tap, p.w_dtype);
            conv_acc = fma(x_tap, w_tap, conv_acc);
        }
    } else {
        // Speculative tree: walk strict ancestors in `ancestor_masks[tok]`.
        const ulong mask = ancestor_masks[tok];
        const ulong strict_anc = (tok > 0u) ? (mask & ((1ul << tok) - 1ul)) : 0ul;
        const uint depth = uint(popcount(strict_anc));
        for (uint tap = 0u; tap < ksz; ++tap) {
            const uint back = (ksz - 1u - tap) * dil;
            float x_tap = 0.0f;
            if (back == 0u) {
                x_tap = float(gv_normed[ulong(tok) * stride + c]);
            } else if (back <= depth) {
                // Find the `back`-th ancestor going backwards from `tok - 1`.
                uint rem = back;
                uint found = 0u;
                for (int cand = int(tok) - 1; cand >= 0; --cand) {
                    if (((strict_anc >> uint(cand)) & 1ul) != 0ul) {
                        rem -= 1u;
                        if (rem == 0u) {
                            found = uint(cand);
                            break;
                        }
                    }
                }
                x_tap = float(gv_normed[ulong(found) * stride + c]);
            } else {
                const uint hist_back = back - depth;
                if (hist_back <= slen) {
                    x_tap = float(conv_state_in[ulong(slen - hist_back) * stride + c]);
                }
            }
            const float w_tap = g27_load_scalar_f32(conv_w, ulong(c) * ksz + tap, p.w_dtype);
            conv_acc = fma(x_tap, w_tap, conv_acc);
        }
    }

    const ulong idx = ulong(tok) * stride + c;
    const float out_val = float(gv_in[idx]) + hybrid_fast_silu(conv_acc);
    h_streams[idx] = half(float(h_streams[idx]) + out_val);
}

// 7. Commits accepted path `accepted_nodes[0..accepted_count)` into `conv_state [state_len, channels]`.
// Grid: ceil(channels / 256), 256 threads.
kernel void ami_qwen4_ple_conv_commit(const device half *gv_normed [[buffer(0)]],
                                      device half *conv_state [[buffer(1)]],
                                      constant M94PleConvCommitParams &p [[buffer(2)]],
                                      uint gid [[thread_position_in_grid]]) {
    const uint c = gid;
    if (c >= p.channels || p.state_len == 0u || p.accepted_count == 0u) return;
    const ulong stride = ulong(p.channels);
    const uint slen = p.state_len;
    const uint acc = p.accepted_count;

    if (acc >= slen) {
        const uint start = acc - slen;
        for (uint s = 0u; s < slen; ++s) {
            const uint node = uint(p.inline_nodes[start + s]);
            conv_state[ulong(s) * stride + c] = gv_normed[ulong(node) * stride + c];
        }
    } else {
        const uint keep = slen - acc;
        for (uint s = 0u; s < keep; ++s) {
            conv_state[ulong(s) * stride + c] = conv_state[ulong(s + acc) * stride + c];
        }
        for (uint s = 0u; s < acc; ++s) {
            const uint node = uint(p.inline_nodes[s]);
            conv_state[ulong(keep + s) * stride + c] = gv_normed[ulong(node) * stride + c];
        }
    }
}

// 8. Real-HF Qwen3.8 QSA Indexer Q norm + partial 3D MRoPE & raw_k extraction.
//    `qk_in`: `[num_tokens, (idx_n_heads + 1) * idx_head_dim]`.
// Grid: (idx_n_heads + 1, num_tokens, 1) threadgroups, 32 threads.
kernel void ami_qsa_real_idx_prep(const device half *qk_in [[buffer(0)]],
                                  const device uchar *q_norm [[buffer(1)]],
                                  const device uint *positions_3d [[buffer(2)]],
                                  device half *q_idx_out [[buffer(3)]],
                                  device half *raw_k_dst [[buffer(4)]],
                                  constant M94QsaIdxPrepParams &p [[buffer(5)]],
                                  uint2 tg_pos [[threadgroup_position_in_grid]],
                                  uint lane [[thread_index_in_simdgroup]]) {
    const uint h = tg_pos.x;
    const uint tok = tg_pos.y;
    const uint total_h = p.idx_n_heads + 1u;
    if (h >= total_h || tok >= p.num_tokens || p.idx_head_dim == 0u || p.idx_head_dim > 256u) {
        return;
    }
    const uint d_dim = p.idx_head_dim;
    const ulong in_off = (ulong(tok) * total_h + h) * d_dim;
    if (h == p.idx_n_heads) {
        // Store raw unrotated, unnormed key for block mean-pooling.
        const ulong dst_tok = (p.write_raw_to_cache != 0u) ? ulong(p.start_pos + tok) : ulong(tok);
        const ulong k_off = dst_tok * d_dim;
        for (uint d = lane; d < d_dim; d += 32u) {
            raw_k_dst[k_off + d] = qk_in[in_off + d];
        }
        return;
    }

    threadgroup float vec[256];
    float sum_sq = 0.0f;
    for (uint d = lane; d < d_dim; d += 32u) {
        const float v = float(qk_in[in_off + d]);
        vec[d] = v;
        sum_sq = fma(v, v, sum_sq);
    }
    const float total_sq = simd_sum(sum_sq);
    const float inv_rms = rsqrt(total_sq / float(d_dim) + p.rms_eps);
    const float w_add = (p.unit_offset != 0u) ? 1.0f : 0.0f;
    for (uint d = lane; d < d_dim; d += 32u) {
        vec[d] = (vec[d] * inv_rms) * (w_add + g27_load_scalar_f32(q_norm, d, p.norm_dtype));
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    const uint lin_pos = p.start_pos + tok;
    const uint pos_t = (p.use_positions_3d != 0u) ? positions_3d[ulong(tok) * 3ul + 0ul] : lin_pos;
    const uint pos_h = (p.use_positions_3d != 0u) ? positions_3d[ulong(tok) * 3ul + 1ul] : lin_pos;
    const uint pos_w = (p.use_positions_3d != 0u) ? positions_3d[ulong(tok) * 3ul + 2ul] : lin_pos;
    const uint half_rot = p.rotary_dim >> 1;
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

    const ulong out_off = (ulong(tok) * p.idx_n_heads + h) * d_dim;
    for (uint d = lane; d < d_dim; d += 32u) {
        q_idx_out[out_off + d] = half(vec[d]);
    }
}

// 9. Pools completed compressed blocks from `raw_k_cache` -> `(1+k_norm)` RMSNorm ->
//    partial 3D MRoPE at block start position `b * compress_ratio` -> `pooled_k_cache`.
// Grid: (num_blocks, 1, 1) threadgroups, 32 threads.
kernel void ami_qsa_real_idx_pool_commit(const device half *raw_k_cache [[buffer(0)]],
                                         const device uchar *k_norm [[buffer(1)]],
                                         device half *pooled_k_cache [[buffer(2)]],
                                         constant M94QsaIdxPoolParams &p [[buffer(3)]],
                                         uint tg_x [[threadgroup_position_in_grid]],
                                         uint lane [[thread_index_in_simdgroup]]) {
    if (tg_x >= p.num_blocks || p.idx_head_dim == 0u || p.idx_head_dim > 256u) return;
    const uint blk = p.start_block + tg_x;
    const uint d_dim = p.idx_head_dim;
    const uint ratio = p.compress_ratio;
    const uint start_tok = blk * ratio;
    const float inv_r = 1.0f / float(ratio);

    threadgroup float vec[256];
    float sum_sq = 0.0f;
    for (uint d = lane; d < d_dim; d += 32u) {
        float acc = 0.0f;
        for (uint r = 0u; r < ratio; ++r) {
            acc += float(raw_k_cache[ulong(start_tok + r) * d_dim + d]);
        }
        const float pooled = float(half(acc * inv_r));
        vec[d] = pooled;
        sum_sq = fma(pooled, pooled, sum_sq);
    }
    const float total_sq = simd_sum(sum_sq);
    const float inv_rms = rsqrt(total_sq / float(d_dim) + p.rms_eps);
    const float w_add = (p.unit_offset != 0u) ? 1.0f : 0.0f;
    for (uint d = lane; d < d_dim; d += 32u) {
        vec[d] = (vec[d] * inv_rms) * (w_add + g27_load_scalar_f32(k_norm, d, p.norm_dtype));
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    const uint half_rot = p.rotary_dim >> 1;
    for (uint j = lane; j < half_rot; j += 32u) {
        const float inv_f = yarn_inv_freq(
            j, p.rotary_dim, p.rope_theta, p.yarn_factor,
            p.yarn_beta_fast, p.yarn_beta_slow, p.original_max_pos);
        const float angle = float(start_tok) * inv_f;
        const float c = cos(angle) * p.yarn_mscale;
        const float s = sin(angle) * p.yarn_mscale;
        const float x0 = vec[j];
        const float x1 = vec[j + half_rot];
        vec[j] = fma(x0, c, -x1 * s);
        vec[j + half_rot] = fma(x1, c, x0 * s);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    const ulong out_off = ulong(blk) * d_dim;
    for (uint d = lane; d < d_dim; d += 32u) {
        pooled_k_cache[out_off + d] = half(vec[d]);
    }
}

// 10. Real-HF QSA Indexer block selection writing `QsaIndexBuffer` (`selected_blocks`, `selected_counts`).
//     Computes `score[b] = sum_h(relu(q_idx[node, h] . pooled_k[b])) / sqrt(idx_hd)`,
//     selects top-`min(block_topk, num_complete_blocks)` blocks plus trailing partial block.
// Grid: (num_nodes, 1, 1) threadgroups, 256 threads (8 simdgroups). Zero stack spills.
kernel void ami_qsa_real_idx_select(const device half *q_idx [[buffer(0)]],
                                    const device half *pooled_k_cache [[buffer(1)]],
                                    device uint *selected_blocks [[buffer(2)]],
                                    device uint *selected_counts [[buffer(3)]],
                                    constant M94QsaIdxSelectParams &p [[buffer(4)]],
                                    uint tg_x [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]],
                                    uint sg_id [[simdgroup_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]]) {
    const uint node = tg_x;
    if (node >= p.num_nodes) return;

    const uint eff_kv_len = (p.causal_q_start_pos == 0xFFFFFFFFu)
        ? p.kv_len
        : min(p.kv_len, p.causal_q_start_pos + node + 1u);
    const uint num_complete = eff_kv_len / p.compress_ratio;
    const bool has_tail = ((eff_kv_len % p.compress_ratio) != 0u);
    const uint cap = min(p.top_k_blocks_cap, 64u);
    const uint tail_cnt = has_tail ? 1u : 0u;
    const uint k_want = min(p.block_topk, cap - tail_cnt);
    const uint k_sel = min(k_want, num_complete);

    threadgroup float tg_top_s[64];
    threadgroup uint tg_top_b[64];
    threadgroup float tg_sg_s[8];
    threadgroup uint tg_sg_b[8];

    if (tid < 64u) {
        tg_top_s[tid] = -INFINITY;
        tg_top_b[tid] = 0xFFFFFFFFu;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint d_dim = p.idx_head_dim;
    const uint n_ih = p.idx_n_heads;
    const ulong q_node_base = ulong(node) * n_ih * d_dim;

    // Fast path: all complete blocks fit within budget.
    if (num_complete <= k_want) {
        if (tid < num_complete) {
            tg_top_b[tid] = tid;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    } else if (k_sel > 0u) {
        // Streaming top-`k_sel` via iterative argmax over complete blocks (0 dynamic arrays).
        for (uint k = 0u; k < k_sel; ++k) {
            float best_s = -INFINITY;
            uint best_b = 0xFFFFFFFFu;
            for (uint b = sg_id; b < num_complete; b += 8u) {
                bool already = false;
                for (uint prev = 0u; prev < k; ++prev) {
                    if (tg_top_b[prev] == b) {
                        already = true;
                        break;
                    }
                }
                if (already) continue;
                const ulong k_off = ulong(b) * d_dim;
                float sc = 0.0f;
                for (uint ih = 0u; ih < n_ih; ++ih) {
                    float dot = 0.0f;
                    const ulong q_off = q_node_base + ulong(ih) * d_dim;
                    for (uint d = lane; d < d_dim; d += 32u) {
                        dot = fma(float(q_idx[q_off + d]), float(pooled_k_cache[k_off + d]), dot);
                    }
                    const float hdot = simd_sum(dot);
                    sc += max(hdot, 0.0f);
                }
                sc *= p.inv_sqrt_d;
                if (lane == 0u && (sc > best_s || (sc == best_s && b < best_b))) {
                    best_s = sc;
                    best_b = b;
                }
            }
            if (lane == 0u) {
                tg_sg_s[sg_id] = best_s;
                tg_sg_b[sg_id] = best_b;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u) {
                float gs = -INFINITY;
                uint gb = 0xFFFFFFFFu;
                for (uint s = 0u; s < 8u; ++s) {
                    if (tg_sg_s[s] > gs || (tg_sg_s[s] == gs && tg_sg_b[s] < gb)) {
                        gs = tg_sg_s[s];
                        gb = tg_sg_b[s];
                    }
                }
                tg_top_s[k] = gs;
                tg_top_b[k] = gb;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    const uint total_out = k_sel + tail_cnt;
    if (tid == 0u && has_tail && k_sel < cap) {
        tg_top_b[k_sel] = num_complete;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Replicate selected blocks across all KV heads for `ami_qsa_sparse_attn`.
    for (uint kh = 0u; kh < p.num_kv_heads; ++kh) {
        const ulong cnt_idx = ulong(node) * p.num_kv_heads + kh;
        const ulong sel_base = cnt_idx * p.top_k_blocks_cap;
        if (tid < total_out) {
            selected_blocks[sel_base + tid] = tg_top_b[tid];
        } else if (tid < p.top_k_blocks_cap) {
            selected_blocks[sel_base + tid] = 0xFFFFFFFFu;
        }
        if (tid == 0u) {
            selected_counts[cnt_idx] = total_out;
        }
    }
}

// 11. DeepSeek-V4.1 `make_identity_pre_mix`: writes `[1, 0, ..., 0]` per token (`f32`).
// Grid: ceil((m * hc_mult) / 256), 256 threads.
kernel void ami_m94_mhc_init_pre_mix(device float *pre_mix [[buffer(0)]],
                                     constant M94MhcInitParams &p [[buffer(1)]],
                                     uint gid [[thread_position_in_grid]]) {
    const uint total = p.m * p.hc_mult;
    if (gid >= total) return;
    const uint lane_s = gid % p.hc_mult;
    pre_mix[gid] = (lane_s == 0u) ? 1.0f : 0.0f;
}

// 12. DeepSeek-V4.1 exact 20-step Sinkhorn `mHC` with `pre_mix` handoff:
//     Computes `(pre_new, post, comb) = hc_mixes(streams, fn_w, scale_w, base_w)`,
//     stores `post + comb` into `coeffs_out [M, 20]`, stores `pre_new` into `pre_mix_out [M, 4]`,
//     and collapses `streams` with INCOMING `pre_mix_in [M, 4]` + optional RMSNorm into `branch_out`!
// Grid: (m, 1, 1) threadgroups, 256 threads (8 simdgroups). Zero stack spills.
kernel void ami_m94_mhc_pre_sinkhorn_handoff(const device uchar *streams [[buffer(0)]],
                                             const device uchar *fn_w [[buffer(1)]],
                                             const device float *base_w [[buffer(2)]],
                                             const device float *scale_w [[buffer(3)]],
                                             const device uchar *norm_w [[buffer(4)]],
                                             device uchar *branch_out [[buffer(5)]],
                                             device float *coeffs_out [[buffer(6)]],
                                             const device float *pre_mix_in [[buffer(7)]],
                                             device float *pre_mix_out [[buffer(8)]],
                                             constant G27MhcSinkhornParams &p [[buffer(9)]],
                                             uint tg_x [[threadgroup_position_in_grid]],
                                             uint tid [[thread_index_in_threadgroup]],
                                             uint sg_id [[simdgroup_index_in_threadgroup]],
                                             uint lane [[thread_index_in_simdgroup]]) {
    const uint tok = tg_x;
    if (tok >= p.m || p.hidden_size == 0u) return;

    const uint total_dim = 4u * p.hidden_size;
    const ulong tok_stream_base = ulong(tok) * total_dim;

    threadgroup float tg_raw[24];
    threadgroup float tg_pre_in[4];
    threadgroup float tg_reduce[8];
    threadgroup float tg_inv_rms[1];

    const uint r0 = sg_id * 3u + 0u;
    const uint r1 = sg_id * 3u + 1u;
    const uint r2 = sg_id * 3u + 2u;
    const ulong fn_off0 = ulong(r0) * total_dim;
    const ulong fn_off1 = ulong(r1) * total_dim;
    const ulong fn_off2 = ulong(r2) * total_dim;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float sq_sum = 0.0f;
    for (uint j = lane; j < total_dim; j += 32u) {
        const float sv = hybrid_load_f16_or_bf16(streams, tok_stream_base + j, GR_STREAM_BF16);
        if (sg_id == 0u) sq_sum = fma(sv, sv, sq_sum);
        acc0 = fma(sv, g27_load_scalar_f32(fn_w, fn_off0 + j, p.fn_dtype), acc0);
        acc1 = fma(sv, g27_load_scalar_f32(fn_w, fn_off1 + j, p.fn_dtype), acc1);
        acc2 = fma(sv, g27_load_scalar_f32(fn_w, fn_off2 + j, p.fn_dtype), acc2);
    }

    const float sum0 = simd_sum(acc0);
    const float sum1 = simd_sum(acc1);
    const float sum2 = simd_sum(acc2);
    const float total_sq = (sg_id == 0u) ? simd_sum(sq_sum) : 0.0f;
    if (lane == 0u) {
        tg_raw[r0] = sum0;
        tg_raw[r1] = sum1;
        tg_raw[r2] = sum2;
        if (sg_id == 0u) tg_inv_rms[0] = rsqrt(total_sq / float(total_dim) + p.rms_eps);
    }
    if (tid < 4u) {
        tg_pre_in[tid] = pre_mix_in[ulong(tok) * 4u + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        const float inv_rms = tg_inv_rms[0];
        const float eps = p.hc_eps;
        const float s_pre = scale_w[0] * inv_rms;
        const float s_post = scale_w[1] * inv_rms;
        const float s_comb = scale_w[2] * inv_rms;

        device float *dst_pre = pre_mix_out + ulong(tok) * 4u;
        dst_pre[0] = hybrid_sigmoid(fma(tg_raw[0], s_pre, base_w[0])) + eps;
        dst_pre[1] = hybrid_sigmoid(fma(tg_raw[1], s_pre, base_w[1])) + eps;
        dst_pre[2] = hybrid_sigmoid(fma(tg_raw[2], s_pre, base_w[2])) + eps;
        dst_pre[3] = hybrid_sigmoid(fma(tg_raw[3], s_pre, base_w[3])) + eps;

        const float p0 = 2.0f * hybrid_sigmoid(fma(tg_raw[4], s_post, base_w[4]));
        const float p1 = 2.0f * hybrid_sigmoid(fma(tg_raw[5], s_post, base_w[5]));
        const float p2 = 2.0f * hybrid_sigmoid(fma(tg_raw[6], s_post, base_w[6]));
        const float p3 = 2.0f * hybrid_sigmoid(fma(tg_raw[7], s_post, base_w[7]));

        float4 c0 = g27_softmax4_plus_eps(float4(
            fma(tg_raw[8],  s_comb, base_w[8]),
            fma(tg_raw[9],  s_comb, base_w[9]),
            fma(tg_raw[10], s_comb, base_w[10]),
            fma(tg_raw[11], s_comb, base_w[11])), eps);
        float4 c1 = g27_softmax4_plus_eps(float4(
            fma(tg_raw[12], s_comb, base_w[12]),
            fma(tg_raw[13], s_comb, base_w[13]),
            fma(tg_raw[14], s_comb, base_w[14]),
            fma(tg_raw[15], s_comb, base_w[15])), eps);
        float4 c2 = g27_softmax4_plus_eps(float4(
            fma(tg_raw[16], s_comb, base_w[16]),
            fma(tg_raw[17], s_comb, base_w[17]),
            fma(tg_raw[18], s_comb, base_w[18]),
            fma(tg_raw[19], s_comb, base_w[19])), eps);
        float4 c3 = g27_softmax4_plus_eps(float4(
            fma(tg_raw[20], s_comb, base_w[20]),
            fma(tg_raw[21], s_comb, base_w[21]),
            fma(tg_raw[22], s_comb, base_w[22]),
            fma(tg_raw[23], s_comb, base_w[23])), eps);

        if (p.sinkhorn_iters > 0u) {
            const float4 inv_c0 = 1.0f / (c0 + c1 + c2 + c3 + eps);
            c0 *= inv_c0; c1 *= inv_c0; c2 *= inv_c0; c3 *= inv_c0;
            for (uint it = 1u; it < p.sinkhorn_iters; ++it) {
                c0 *= g27_row_norm4(c0, eps);
                c1 *= g27_row_norm4(c1, eps);
                c2 *= g27_row_norm4(c2, eps);
                c3 *= g27_row_norm4(c3, eps);
                const float4 inv_c = 1.0f / (c0 + c1 + c2 + c3 + eps);
                c0 *= inv_c; c1 *= inv_c; c2 *= inv_c; c3 *= inv_c;
            }
        }

        device float *dst_c = coeffs_out + ulong(tok) * 20u;
        dst_c[0] = p0; dst_c[1] = p1; dst_c[2] = p2; dst_c[3] = p3;
        dst_c[4] = c0.x; dst_c[5] = c0.y; dst_c[6] = c0.z; dst_c[7] = c0.w;
        dst_c[8] = c1.x; dst_c[9] = c1.y; dst_c[10] = c1.z; dst_c[11] = c1.w;
        dst_c[12] = c2.x; dst_c[13] = c2.y; dst_c[14] = c2.z; dst_c[15] = c2.w;
        dst_c[16] = c3.x; dst_c[17] = c3.y; dst_c[18] = c3.z; dst_c[19] = c3.w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float pre0 = tg_pre_in[0];
    const float pre1 = tg_pre_in[1];
    const float pre2 = tg_pre_in[2];
    const float pre3 = tg_pre_in[3];
    const bool has_norm_w = ((p.flags & 1u) != 0u);
    const ulong h = ulong(p.hidden_size);
    const ulong tok_out_base = ulong(tok) * h;

    float inv_rms_sub = 1.0f;
    if (has_norm_w) {
        float sq_partial = 0.0f;
        for (uint d = tid; d < p.hidden_size; d += 256u) {
            const float s0 = hybrid_load_f16_or_bf16(streams, tok_stream_base + 0ul * h + d, GR_STREAM_BF16);
            const float s1 = hybrid_load_f16_or_bf16(streams, tok_stream_base + 1ul * h + d, GR_STREAM_BF16);
            const float s2 = hybrid_load_f16_or_bf16(streams, tok_stream_base + 2ul * h + d, GR_STREAM_BF16);
            const float s3 = hybrid_load_f16_or_bf16(streams, tok_stream_base + 3ul * h + d, GR_STREAM_BF16);
            const float mix = fma(pre0, s0, fma(pre1, s1, fma(pre2, s2, pre3 * s3)));
            sq_partial = fma(mix, mix, sq_partial);
        }
        const float sg_sq = simd_sum(sq_partial);
        if (lane == 0u) tg_reduce[sg_id] = sg_sq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float sub_sq = ((tg_reduce[0] + tg_reduce[1]) + (tg_reduce[2] + tg_reduce[3])) +
                             ((tg_reduce[4] + tg_reduce[5]) + (tg_reduce[6] + tg_reduce[7]));
        inv_rms_sub = rsqrt(sub_sq / float(p.hidden_size) + p.rms_eps);
    }

    for (uint d = tid; d < p.hidden_size; d += 256u) {
        const float s0 = hybrid_load_f16_or_bf16(streams, tok_stream_base + 0ul * h + d, GR_STREAM_BF16);
        const float s1 = hybrid_load_f16_or_bf16(streams, tok_stream_base + 1ul * h + d, GR_STREAM_BF16);
        const float s2 = hybrid_load_f16_or_bf16(streams, tok_stream_base + 2ul * h + d, GR_STREAM_BF16);
        const float s3 = hybrid_load_f16_or_bf16(streams, tok_stream_base + 3ul * h + d, GR_STREAM_BF16);
        float mix = fma(pre0, s0, fma(pre1, s1, fma(pre2, s2, pre3 * s3)));
        if (has_norm_w) {
            mix = (mix * inv_rms_sub) * g27_load_scalar_f32(norm_w, d, p.norm_dtype);
        }
        hybrid_store_f16_or_bf16(branch_out, tok_out_base + d, mix, GR_ACT_BF16);
    }
}

// 13. DeepSeek-V4.1 final `hc_pre(h, pre_mix)` collapse + optional RMSNorm (`norm.weight`).
// Grid: (m, 1, 1) threadgroups, 256 threads.
kernel void ami_m94_mhc_collapse_pre_mix(const device uchar *streams [[buffer(0)]],
                                         const device float *pre_mix [[buffer(1)]],
                                         const device uchar *norm_w [[buffer(2)]],
                                         device half *out [[buffer(3)]],
                                         constant M94MhcCollapseParams &p [[buffer(4)]],
                                         uint tg_x [[threadgroup_position_in_grid]],
                                         uint tid [[thread_index_in_threadgroup]],
                                         uint sg_id [[simdgroup_index_in_threadgroup]],
                                         uint lane [[thread_index_in_simdgroup]]) {
    const uint tok = tg_x;
    if (tok >= p.m || p.hidden_size == 0u) return;

    const ulong h = ulong(p.hidden_size);
    const ulong tok_stream_base = ulong(tok) * 4ul * h;
    const ulong tok_out_base = ulong(tok) * h;

    threadgroup float tg_pre[4];
    threadgroup float tg_red[8];
    if (tid < 4u) tg_pre[tid] = pre_mix[ulong(tok) * 4u + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float p0 = tg_pre[0];
    const float p1 = tg_pre[1];
    const float p2 = tg_pre[2];
    const float p3 = tg_pre[3];

    float inv_rms = 1.0f;
    if (p.has_norm_w != 0u) {
        float sq = 0.0f;
        for (uint d = tid; d < p.hidden_size; d += 256u) {
            const float s0 = hybrid_load_f16_or_bf16(streams, tok_stream_base + 0ul * h + d, GR_STREAM_BF16);
            const float s1 = hybrid_load_f16_or_bf16(streams, tok_stream_base + 1ul * h + d, GR_STREAM_BF16);
            const float s2 = hybrid_load_f16_or_bf16(streams, tok_stream_base + 2ul * h + d, GR_STREAM_BF16);
            const float s3 = hybrid_load_f16_or_bf16(streams, tok_stream_base + 3ul * h + d, GR_STREAM_BF16);
            const float v = fma(p0, s0, fma(p1, s1, fma(p2, s2, p3 * s3)));
            sq = fma(v, v, sq);
        }
        const float sg_sq = simd_sum(sq);
        if (lane == 0u) tg_red[sg_id] = sg_sq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float tot_sq = ((tg_red[0] + tg_red[1]) + (tg_red[2] + tg_red[3])) +
                             ((tg_red[4] + tg_red[5]) + (tg_red[6] + tg_red[7]));
        inv_rms = rsqrt(tot_sq / float(p.hidden_size) + p.rms_eps);
    }

    for (uint d = tid; d < p.hidden_size; d += 256u) {
        const float s0 = hybrid_load_f16_or_bf16(streams, tok_stream_base + 0ul * h + d, GR_STREAM_BF16);
        const float s1 = hybrid_load_f16_or_bf16(streams, tok_stream_base + 1ul * h + d, GR_STREAM_BF16);
        const float s2 = hybrid_load_f16_or_bf16(streams, tok_stream_base + 2ul * h + d, GR_STREAM_BF16);
        const float s3 = hybrid_load_f16_or_bf16(streams, tok_stream_base + 3ul * h + d, GR_STREAM_BF16);
        float v = fma(p0, s0, fma(p1, s1, fma(p2, s2, p3 * s3)));
        if (p.has_norm_w != 0u) {
            v = (v * inv_rms) * g27_load_scalar_f32(norm_w, d, p.norm_dtype);
        }
        out[tok_out_base + d] = half(v);
    }
}

// 14. Exact DeepSeek-V4.1 complex-pair RoPE (`apply_rotary_emb`, forward or `inverse`, optional YaRN).
//     Keeps adjacent-pair layout: `(x[2j], x[2j+1]) * exp(±i * pos * freq[j])`.
// Grid: (num_heads, num_tokens, 1) threadgroups, 32 threads.
kernel void ami_m94_rope_complex(device half *x [[buffer(0)]],
                                 const device uint *positions [[buffer(1)]],
                                 constant M94RopeComplexParams &p [[buffer(2)]],
                                 uint2 tg_pos [[threadgroup_position_in_grid]],
                                 uint lane [[thread_index_in_simdgroup]]) {
    const uint head = tg_pos.x;
    const uint tok = tg_pos.y;
    if (head >= p.num_heads || tok >= p.num_tokens || p.rot_dim < 2u) return;

    const uint half_d = p.rot_dim >> 1;
    const uint stride_pos = (p.pos_stride > 0u) ? p.pos_stride : 1u;
    const uint pos = (p.use_pos_buf != 0u) ? positions[tok] : (p.base_pos + tok * stride_pos);
    const ulong base = (ulong(tok) * p.num_heads + head) * p.head_stride + p.rot_offset;

    for (uint j = lane; j < half_d; j += 32u) {
        const float x0 = float(x[base + 2u * j]);
        const float x1 = float(x[base + 2u * j + 1u]);
        const float inv_f = m94_ds_inv_freq(
            j, p.rot_dim, p.rope_theta, p.original_seq_len, p.rope_factor, p.beta_fast, p.beta_slow);
        const float angle = float(pos) * inv_f;
        const float c = cos(angle);
        const float s = (p.inverse != 0u) ? -sin(angle) : sin(angle);
        x[base + 2u * j]      = half(fma(x0, c, -x1 * s));
        x[base + 2u * j + 1u] = half(fma(x1, c,  x0 * s));
    }
}

// 15. DeepSeek-V4.1 `Compressor` softmax-gated group pooling (`ratio >= 2`) + RMSNorm.
//     Supports both real-HF (`gate_dim == head_dim`) and synth (`gate_dim == ratio`),
//     handling both prefill (`num_tokens >= 1` starting at `start_pos`) and decode (`num_tokens == 1`).
// Grid: (max(out_count, 1), 1, 1) threadgroups, 256 threads.
kernel void ami_m94_ds_compressor_pool(const device half *kv_in [[buffer(0)]],
                                       const device half *gate_in [[buffer(1)]],
                                       const device uchar *norm_w [[buffer(2)]],
                                       device float *kv_state [[buffer(3)]],
                                       device float *score_state [[buffer(4)]],
                                       device half *latent_out [[buffer(5)]],
                                       constant M94DsCompressorParams &p [[buffer(6)]],
                                       uint tg_x [[threadgroup_position_in_grid]],
                                       uint tid [[thread_index_in_threadgroup]],
                                       uint sg_id [[simdgroup_index_in_threadgroup]],
                                       uint lane [[thread_index_in_simdgroup]]) {
    const uint d_dim = p.head_dim;
    const uint ratio = p.ratio;
    if (d_dim == 0u || ratio == 0u) return;

    // Group `g_idx = tg_x` produces `latent_out[g_idx]` for completed group starting at
    // global token index `g_tok_start = ((start_pos / ratio) + g_idx) * ratio`?
    // Caller ensures `out_count = (start_pos + num_tokens) / ratio - (start_pos / ratio)`.
    // First update partial group state on threadgroup 0 for trailing remainder (if `write_state != 0`).
    const uint end_pos = p.start_pos + p.num_tokens;
    const uint first_complete_group = p.start_pos / ratio;
    const uint end_complete_group = end_pos / ratio;
    const uint num_out = end_complete_group - first_complete_group;

    threadgroup float tg_red[8];

    if (tg_x < num_out) {
        const uint grp = first_complete_group + tg_x;
        const uint grp_tok0 = grp * ratio;

        float sq_part = 0.0f;
        for (uint d = tid; d < d_dim; d += 256u) {
            float max_sc = -INFINITY;
            for (uint r = 0u; r < ratio; ++r) {
                const uint gpos = grp_tok0 + r;
                float sc = 0.0f;
                if (gpos < p.start_pos) {
                    sc = score_state[ulong(r) * d_dim + d];
                } else {
                    const uint local_t = gpos - p.start_pos;
                    const uint g_col = (p.gate_dim >= d_dim) ? d : (r % p.gate_dim);
                    sc = float(gate_in[ulong(local_t) * p.gate_dim + g_col]);
                }
                max_sc = max(max_sc, sc);
            }
            float denom = 0.0f;
            float acc = 0.0f;
            for (uint r = 0u; r < ratio; ++r) {
                const uint gpos = grp_tok0 + r;
                float sc = 0.0f;
                float kv = 0.0f;
                if (gpos < p.start_pos) {
                    sc = score_state[ulong(r) * d_dim + d];
                    kv = kv_state[ulong(r) * d_dim + d];
                } else {
                    const uint local_t = gpos - p.start_pos;
                    const uint g_col = (p.gate_dim >= d_dim) ? d : (r % p.gate_dim);
                    sc = float(gate_in[ulong(local_t) * p.gate_dim + g_col]);
                    kv = float(kv_in[ulong(local_t) * d_dim + d]);
                }
                const float w = exp(sc - max_sc);
                denom += w;
                acc = fma(w, kv, acc);
            }
            const float pooled = float(half(acc / max(denom, 1.0e-20f)));
            latent_out[ulong(tg_x) * d_dim + d] = half(pooled);
            sq_part = fma(pooled, pooled, sq_part);
        }
        const float sg_sq = simd_sum(sq_part);
        if (lane == 0u) tg_red[sg_id] = sg_sq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float tot_sq = ((tg_red[0] + tg_red[1]) + (tg_red[2] + tg_red[3])) +
                             ((tg_red[4] + tg_red[5]) + (tg_red[6] + tg_red[7]));
        const float inv_rms = rsqrt(tot_sq / float(d_dim) + p.rms_eps);
        for (uint d = tid; d < d_dim; d += 256u) {
            const float v = float(latent_out[ulong(tg_x) * d_dim + d]);
            const float nw = g27_load_scalar_f32(norm_w, d, p.norm_dtype);
            latent_out[ulong(tg_x) * d_dim + d] = half((v * inv_rms) * nw);
        }
    }

    // Commit trailing partial group into `(kv_state, score_state)` on threadgroup 0.
    if (tg_x == 0u && p.write_state != 0u) {
        const uint rem_start = end_complete_group * ratio;
        if (rem_start < end_pos) {
            const uint first_new = max(rem_start, p.start_pos);
            for (uint gpos = first_new; gpos < end_pos; ++gpos) {
                const uint slot = gpos - rem_start;
                const uint local_t = gpos - p.start_pos;
                for (uint d = tid; d < d_dim; d += 256u) {
                    const uint g_col = (p.gate_dim >= d_dim) ? d : (slot % p.gate_dim);
                    kv_state[ulong(slot) * d_dim + d] = float(kv_in[ulong(local_t) * d_dim + d]);
                    score_state[ulong(slot) * d_dim + d] = float(gate_in[ulong(local_t) * p.gate_dim + g_col]);
                }
            }
        }
    }
}

// 16. DeepSeek-V4.1 two-level `Indexer` scoring, candidate block mask, and top-k selection.
//     Computes `score[t] = sum_h (relu(q_idx[tok, h] . index_k[t]) * weights[tok, h] * weight_scale)`.
//     Optional Level-1 (`is_candidate_source`): max-pools scores per `cand_block_size`, pins last
//     reachable block `(compress_len - 1) / cand_block_size` to `+inf`, writes `candidate_mask`.
//     Optional Level-2 (`uses_candidates`): masks scores outside `candidate_mask`.
//     Outputs top-`min(index_topk, total_comp_len)` indices (sorted ascending, `+ offset`, `-1` unreachable).
// Grid: (num_tokens, 1, 1) threadgroups, 256 threads (8 simdgroups). Zero stack spills.
kernel void ami_m94_ds_indexer_select(const device half *q_idx [[buffer(0)]],
                                      const device half *index_k_cache [[buffer(1)]],
                                      const device half *weights [[buffer(2)]],
                                      device uchar *candidate_mask [[buffer(3)]],
                                      device int *topk_idxs_out [[buffer(4)]],
                                      device float *scores_scratch [[buffer(5)]],
                                      constant M94DsIndexerParams &p [[buffer(6)]],
                                      uint tg_x [[threadgroup_position_in_grid]],
                                      uint tid [[thread_index_in_threadgroup]],
                                      uint sg_id [[simdgroup_index_in_threadgroup]],
                                      uint lane [[thread_index_in_simdgroup]]) {
    const uint tok = tg_x;
    if (tok >= p.num_tokens) return;

    const uint comp_len = p.is_causal_prefill != 0u
        ? ((p.start_pos + tok + 1u) / p.ratio)
        : p.total_comp_len;
    const uint total_c = p.total_comp_len;
    const uint topk = min(p.index_topk, total_c);
    device float *row_scores = scores_scratch + ulong(tok) * max(total_c, 1u);

    // Pass 1: score every compressed key position `t in 0..total_c-1`.
    const uint n_h = p.idx_n_heads;
    const uint d_dim = p.idx_head_dim;
    const ulong q_tok_base = ulong(tok) * n_h * d_dim;
    const ulong w_tok_base = ulong(tok) * n_h;

    for (uint t = sg_id; t < total_c; t += 8u) {
        if (t >= comp_len) {
            if (lane == 0u) row_scores[t] = -INFINITY;
            continue;
        }
        const ulong k_off = ulong(t) * d_dim;
        float total_s = 0.0f;
        for (uint h = 0u; h < n_h; ++h) {
            const ulong q_off = q_tok_base + ulong(h) * d_dim;
            float dot = 0.0f;
            for (uint d = lane; d < d_dim; d += 32u) {
                dot = fma(float(q_idx[q_off + d]), float(index_k_cache[k_off + d]), dot);
            }
            const float hdot = simd_sum(dot);
            if (lane == 0u) {
                const float w_h = float(weights[w_tok_base + h]) * p.weight_scale;
                total_s = fma(max(hdot, 0.0f), w_h, total_s);
            }
        }
        if (lane == 0u) {
            row_scores[t] = total_s;
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    // Optional Level-1 (`candidate_source_layer`): select `cand_topk_blocks` blocks, always pinning last block.
    if (p.is_candidate_source != 0u && p.cand_block_size > 0u && p.cand_num_blocks > 0u) {
        device uchar *cm_row = candidate_mask + ulong(tok) * p.cand_num_blocks;
        const uint last_blk = (comp_len > 0u) ? ((comp_len - 1u) / p.cand_block_size) : 0xFFFFFFFFu;
        const uint k_blk = min(p.cand_topk_blocks, p.cand_num_blocks);
        for (uint b = tid; b < p.cand_num_blocks; b += 256u) {
            cm_row[b] = (p.cand_num_blocks <= k_blk && b * p.cand_block_size < comp_len) ? 1u : 0u;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

        if (p.cand_num_blocks > k_blk && comp_len > 0u) {
            threadgroup float tg_bs[8];
            threadgroup uint tg_bi[8];
            for (uint pick = 0u; pick < k_blk; ++pick) {
                float best_s = -INFINITY;
                uint best_b = 0xFFFFFFFFu;
                for (uint b = tid; b < p.cand_num_blocks; b += 256u) {
                    if (cm_row[b] != 0u) continue;
                    const uint t0 = b * p.cand_block_size;
                    if (t0 >= comp_len) continue;
                    float bs = (b == last_blk) ? INFINITY : -INFINITY;
                    if (b != last_blk) {
                        const uint t1 = min(t0 + p.cand_block_size, comp_len);
                        for (uint t = t0; t < t1; ++t) {
                            bs = max(bs, row_scores[t]);
                        }
                    }
                    if (bs > best_s || (bs == best_s && b < best_b)) {
                        best_s = bs;
                        best_b = b;
                    }
                }
                // Simdgroup reduce.
                for (ushort off = 16u; off > 0u; off >>= 1u) {
                    const float os = simd_shuffle_down(best_s, off);
                    const uint ob = simd_shuffle_down(best_b, off);
                    if (os > best_s || (os == best_s && ob < best_b)) {
                        best_s = os;
                        best_b = ob;
                    }
                }
                if (lane == 0u) {
                    tg_bs[sg_id] = best_s;
                    tg_bi[sg_id] = best_b;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid == 0u) {
                    float gs = -INFINITY;
                    uint gb = 0xFFFFFFFFu;
                    for (uint s = 0u; s < 8u; ++s) {
                        if (tg_bs[s] > gs || (tg_bs[s] == gs && tg_bi[s] < gb)) {
                            gs = tg_bs[s];
                            gb = tg_bi[s];
                        }
                    }
                    if (gb < p.cand_num_blocks && gs > -INFINITY) {
                        cm_row[gb] = 1u;
                    }
                }
                threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
            }
        }
    } else if (p.uses_candidates != 0u && p.cand_block_size > 0u && p.cand_num_blocks > 0u) {
        const device uchar *cm_row = candidate_mask + ulong(tok) * p.cand_num_blocks;
        for (uint t = tid; t < total_c; t += 256u) {
            const uint b = t / p.cand_block_size;
            if (b >= p.cand_num_blocks || cm_row[b] == 0u) {
                row_scores[t] = -INFINITY;
            }
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }

    // Pass 2: select `topk` highest-scoring positions (exact match to `topk(..., sorted=False)` + `sort()`).
    device int *out_row = topk_idxs_out + ulong(tok) * max(topk, 1u);
    if (topk == 0u) return;

    if (total_c <= topk) {
        for (uint k = tid; k < topk; k += 256u) {
            out_row[k] = (k < comp_len && row_scores[k] > -INFINITY) ? int(k + p.offset) : -1;
        }
        return;
    }

    threadgroup float tg_s[8];
    threadgroup uint tg_i[8];
    for (uint k = 0u; k < topk; ++k) {
        float best_s = -INFINITY;
        uint best_t = 0xFFFFFFFFu;
        for (uint t = tid; t < total_c; t += 256u) {
            const float sc = row_scores[t];
            if (sc > best_s || (sc == best_s && t < best_t)) {
                best_s = sc;
                best_t = t;
            }
        }
        for (ushort off = 16u; off > 0u; off >>= 1u) {
            const float os = simd_shuffle_down(best_s, off);
            const uint ot = simd_shuffle_down(best_t, off);
            if (os > best_s || (os == best_s && ot < best_t)) {
                best_s = os;
                best_t = ot;
            }
        }
        if (lane == 0u) {
            tg_s[sg_id] = best_s;
            tg_i[sg_id] = best_t;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float gs = -INFINITY;
            uint gt = 0xFFFFFFFFu;
            for (uint s = 0u; s < 8u; ++s) {
                if (tg_s[s] > gs || (tg_s[s] == gs && tg_i[s] < gt)) {
                    gs = tg_s[s];
                    gt = tg_i[s];
                }
            }
            if (gt < comp_len && gs > -INFINITY) {
                out_row[k] = int(gt + p.offset);
                row_scores[gt] = -INFINITY;
            } else {
                out_row[k] = -1;
            }
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }
}

// 17. DeepSeek-V4.1 shared-KV sparse latent attention over sliding window + compressed
//     latent KV with exact FP32 `attn_sink` and fused inverse complex RoPE on `out[..., -rd:]`.
//     Supports real-HF (`q_head_dim == kv_dim == v_head_dim <= 512`) and synth compact shapes.
// Grid: (n_heads, num_q, 1) threadgroups, 256 threads (8 simdgroups). Zero stack spills.
kernel void ami_m94_ds_latent_attn(const device half *q [[buffer(0)]],
                                   const device half *win_kv_cache [[buffer(1)]],
                                   const device half *win_kv_cur [[buffer(2)]],
                                   const device half *comp_kv_cache [[buffer(3)]],
                                   const device int *topk_idxs [[buffer(4)]],
                                   const device float *attn_sink [[buffer(5)]],
                                   const device ulong *ancestor_masks [[buffer(6)]],
                                   device half *out [[buffer(7)]],
                                   constant M94DsLatentAttnParams &p [[buffer(8)]],
                                   uint2 tg_pos [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]],
                                   uint sg_id [[simdgroup_index_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]]) {
    const uint head = tg_pos.x;
    const uint q_idx = tg_pos.y;
    if (head >= p.n_heads || q_idx >= p.num_q || p.kv_dim > 512u || p.q_head_dim > 512u) return;

    threadgroup float tg_q[512];
    threadgroup float tg_red[8];

    const ulong q_base = (ulong(q_idx) * p.n_heads + head) * p.q_head_dim;
    for (uint d = tid; d < p.q_head_dim; d += 256u) {
        tg_q[d] = float(q[q_base + d]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Two FP32 register accumulators per thread cover `v_head_dim <= 512` with 0 stack spills.
    const uint d0 = tid;
    const uint d1 = tid + 256u;
    float o0 = 0.0f;
    float o1 = 0.0f;
    float m_val = -1.0e30f;
    float l_val = 0.0f;

    const bool split_q_kv = (p.q_head_dim != p.kv_dim);
    const uint kv_rope_off = p.kv_dim - p.rope_dim;

    // Helper: attend to one KV vector `kv_ptr` of length `p.kv_dim`.
    // Phase A: committed sliding-window positions in `win_kv_cache` (`[0, p.start_pos)` within window).
    const uint q_global_pos = p.start_pos + q_idx;
    const uint win_lo = (q_global_pos + 1u > p.window_size) ? (q_global_pos + 1u - p.window_size) : 0u;
    const uint win_committed_end = min(p.start_pos, q_global_pos + 1u);

    for (uint pos = win_lo; pos < win_committed_end; ++pos) {
        const uint slot = pos % p.window_size;
        const device half *kv_ptr = win_kv_cache + ulong(slot) * p.kv_dim;
        float dot_p = 0.0f;
        if (!split_q_kv) {
            if (d0 < p.kv_dim) dot_p = fma(tg_q[d0], float(kv_ptr[d0]), dot_p);
            if (d1 < p.kv_dim) dot_p = fma(tg_q[d1], float(kv_ptr[d1]), dot_p);
        } else {
            if (d0 < p.nope_dim) dot_p = fma(tg_q[d0], float(kv_ptr[d0]), dot_p);
            if (d0 < p.rope_dim) dot_p = fma(tg_q[p.nope_dim + d0], float(kv_ptr[kv_rope_off + d0]), dot_p);
        }
        const float sg_dot = simd_sum(dot_p);
        if (lane == 0u) tg_red[sg_id] = sg_dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float score = (((tg_red[0] + tg_red[1]) + (tg_red[2] + tg_red[3])) +
                             ((tg_red[4] + tg_red[5]) + (tg_red[6] + tg_red[7]))) * p.scale;

        const float m_new = max(m_val, score);
        const float alpha = exp(m_val - m_new);
        const float w = exp(score - m_new);
        l_val = fma(l_val, alpha, w);
        m_val = m_new;
        if (d0 < p.v_head_dim) o0 = fma(w, float(kv_ptr[d0]), o0 * alpha);
        if (d1 < p.v_head_dim) o1 = fma(w, float(kv_ptr[d1]), o1 * alpha);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Phase B: current-chunk / tree uncommitted window positions in `win_kv_cur`.
    const ulong anc_mask = (p.is_tree != 0u) ? ancestor_masks[q_idx] : 0ul;
    const uint cur_lo = (win_lo > p.start_pos) ? (win_lo - p.start_pos) : 0u;
    for (uint k_i = cur_lo; k_i <= q_idx; ++k_i) {
        if (p.is_tree != 0u && (((anc_mask >> k_i) & 1ul) == 0ul)) continue;
        const device half *kv_ptr = win_kv_cur + ulong(k_i) * p.kv_dim;
        float dot_p = 0.0f;
        if (!split_q_kv) {
            if (d0 < p.kv_dim) dot_p = fma(tg_q[d0], float(kv_ptr[d0]), dot_p);
            if (d1 < p.kv_dim) dot_p = fma(tg_q[d1], float(kv_ptr[d1]), dot_p);
        } else {
            if (d0 < p.nope_dim) dot_p = fma(tg_q[d0], float(kv_ptr[d0]), dot_p);
            if (d0 < p.rope_dim) dot_p = fma(tg_q[p.nope_dim + d0], float(kv_ptr[kv_rope_off + d0]), dot_p);
        }
        const float sg_dot = simd_sum(dot_p);
        if (lane == 0u) tg_red[sg_id] = sg_dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float score = (((tg_red[0] + tg_red[1]) + (tg_red[2] + tg_red[3])) +
                             ((tg_red[4] + tg_red[5]) + (tg_red[6] + tg_red[7]))) * p.scale;

        const float m_new = max(m_val, score);
        const float alpha = exp(m_val - m_new);
        const float w = exp(score - m_new);
        l_val = fma(l_val, alpha, w);
        m_val = m_new;
        if (d0 < p.v_head_dim) o0 = fma(w, float(kv_ptr[d0]), o0 * alpha);
        if (d1 < p.v_head_dim) o1 = fma(w, float(kv_ptr[d1]), o1 * alpha);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Phase C: selected compressed latent KV positions (`comp_kv_cache[idx]`).
    if (p.has_compress != 0u && p.compress_topk > 0u) {
        const device int *row_idx = topk_idxs + ulong(q_idx) * p.compress_topk;
        for (uint k = 0u; k < p.compress_topk; ++k) {
            const int c_slot = row_idx[k];
            if (c_slot < 0) continue;
            const device half *kv_ptr = comp_kv_cache + ulong(uint(c_slot)) * p.kv_dim;
            float dot_p = 0.0f;
            if (!split_q_kv) {
                if (d0 < p.kv_dim) dot_p = fma(tg_q[d0], float(kv_ptr[d0]), dot_p);
                if (d1 < p.kv_dim) dot_p = fma(tg_q[d1], float(kv_ptr[d1]), dot_p);
            } else {
                if (d0 < p.nope_dim) dot_p = fma(tg_q[d0], float(kv_ptr[d0]), dot_p);
                if (d0 < p.rope_dim) dot_p = fma(tg_q[p.nope_dim + d0], float(kv_ptr[kv_rope_off + d0]), dot_p);
            }
            const float sg_dot = simd_sum(dot_p);
            if (lane == 0u) tg_red[sg_id] = sg_dot;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float score = (((tg_red[0] + tg_red[1]) + (tg_red[2] + tg_red[3])) +
                                 ((tg_red[4] + tg_red[5]) + (tg_red[6] + tg_red[7]))) * p.scale;

            const float m_new = max(m_val, score);
            const float alpha = exp(m_val - m_new);
            const float w = exp(score - m_new);
            l_val = fma(l_val, alpha, w);
            m_val = m_new;
            if (d0 < p.v_head_dim) o0 = fma(w, float(kv_ptr[d0]), o0 * alpha);
            if (d1 < p.v_head_dim) o1 = fma(w, float(kv_ptr[d1]), o1 * alpha);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Exact FP32 `attn_sink` contribution (numerically stable even if `sink > m_val`).
    if (p.has_sink != 0u) {
        const float sink_s = attn_sink[head];
        const float m_new = max(m_val, sink_s);
        const float alpha = exp(m_val - m_new);
        o0 *= alpha;
        o1 *= alpha;
        l_val = fma(l_val, alpha, exp(sink_s - m_new));
    }

    const float inv_l = (l_val > 0.0f) ? (1.0f / l_val) : 0.0f;
    // Reuse `tg_q` to stage normalized output when fused inverse complex RoPE is requested.
    if (p.apply_inv_rope != 0u && p.rope_dim >= 2u && p.v_head_dim >= p.rope_dim) {
        if (d0 < p.v_head_dim) tg_q[d0] = o0 * inv_l;
        if (d1 < p.v_head_dim) tg_q[d1] = o1 * inv_l;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint out_rope_off = p.v_head_dim - p.rope_dim;
        const uint half_rd = p.rope_dim >> 1;
        if (tid < half_rd) {
            const float x0 = tg_q[out_rope_off + 2u * tid];
            const float x1 = tg_q[out_rope_off + 2u * tid + 1u];
            const float inv_f = m94_ds_inv_freq(
                tid, p.rope_dim, p.rope_theta, p.original_seq_len, p.rope_factor, p.beta_fast, p.beta_slow);
            const float angle = float(q_global_pos) * inv_f;
            const float c = cos(angle);
            const float s = -sin(angle); // conjugate inverse
            tg_q[out_rope_off + 2u * tid]      = fma(x0, c, -x1 * s);
            tg_q[out_rope_off + 2u * tid + 1u] = fma(x1, c,  x0 * s);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const ulong out_base = (ulong(q_idx) * p.n_heads + head) * p.v_head_dim;
        if (d0 < p.v_head_dim) out[out_base + d0] = half(tg_q[d0]);
        if (d1 < p.v_head_dim) out[out_base + d1] = half(tg_q[d1]);
    } else {
        const ulong out_base = (ulong(q_idx) * p.n_heads + head) * p.v_head_dim;
        if (d0 < p.v_head_dim) out[out_base + d0] = half(o0 * inv_l);
        if (d1 < p.v_head_dim) out[out_base + d1] = half(o1 * inv_l);
    }
}

// 18. DeepSeek-V4.1 real-HF `Engram` per-stream `rstd` signed-sqrt sigmoid gate + value residual add.
//     Supports both real-HF (`wkv_out_dim == 5 * H`, `k_dim == H`) and synth (`wkv_out_dim == H`, `k_dim == emb_dim`).
// Grid: (4, m, 1) threadgroups, 256 threads (8 simdgroups). Zero stack spills.
kernel void ami_m94_ds_engram_gate_add(device uchar *streams [[buffer(0)]],
                                       const device half *wkv_out [[buffer(1)]],
                                       const device half *emb_in [[buffer(2)]],
                                       const device uchar *q_weight [[buffer(3)]],
                                       const device uchar *k_weight [[buffer(4)]],
                                       constant M94DsEngramParams &p [[buffer(5)]],
                                       uint2 tg_pos [[threadgroup_position_in_grid]],
                                       uint tid [[thread_index_in_threadgroup]],
                                       uint sg_id [[simdgroup_index_in_threadgroup]],
                                       uint lane [[thread_index_in_simdgroup]]) {
    const uint s = tg_pos.x;
    const uint tok = tg_pos.y;
    if (s >= 4u || tok >= p.m || p.hidden_size == 0u) return;

    const ulong h = ulong(p.hidden_size);
    const ulong st_base = (ulong(tok) * 4ul + s) * h;
    const ulong wkv_base = ulong(tok) * p.wkv_out_dim;
    const bool real_hf_wkv = (p.wkv_out_dim >= 5u * p.hidden_size);

    threadgroup float tg_a[8];
    threadgroup float tg_b[8];
    threadgroup float tg_c[8];

    float sq_h = 0.0f;
    float sq_k = 0.0f;
    float dot_p = 0.0f;

    if (real_hf_wkv) {
        const ulong key_base = wkv_base + ulong(s) * h;
        const ulong w_off = ulong(s) * h;
        for (uint d = tid; d < p.hidden_size; d += 256u) {
            const float hv = hybrid_load_f16_or_bf16(streams, st_base + d, GR_STREAM_BF16);
            const float kv = float(wkv_out[key_base + d]);
            const float qw = g27_load_scalar_f32(q_weight, w_off + d, p.qk_dtype);
            const float kw = g27_load_scalar_f32(k_weight, w_off + d, p.qk_dtype);
            sq_h = fma(hv, hv, sq_h);
            sq_k = fma(kv, kv, sq_k);
            dot_p = fma(hv * qw, kw * kv, dot_p);
        }
    } else {
        const uint k_len = min(p.hidden_size, min(p.k_dim, p.emb_dim));
        const ulong emb_base = ulong(tok) * p.emb_dim;
        const ulong qw_off = ulong(s) * h;
        const ulong kw_off = ulong(s) * p.k_dim;
        for (uint d = tid; d < k_len; d += 256u) {
            const float hv = hybrid_load_f16_or_bf16(streams, st_base + d, GR_STREAM_BF16);
            const float kv = float(emb_in[emb_base + d]);
            const float qw = g27_load_scalar_f32(q_weight, qw_off + d, p.qk_dtype);
            const float kw = g27_load_scalar_f32(k_weight, kw_off + d, p.qk_dtype);
            sq_h = fma(hv, hv, sq_h);
            sq_k = fma(kv, kv, sq_k);
            dot_p = fma(hv * qw, kw * kv, dot_p);
        }
    }

    const float sh = simd_sum(sq_h);
    const float sk = simd_sum(sq_k);
    const float sd = simd_sum(dot_p);
    if (lane == 0u) {
        tg_a[sg_id] = sh;
        tg_b[sg_id] = sk;
        tg_c[sg_id] = sd;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float norm_len = real_hf_wkv ? float(p.hidden_size) : float(max(1u, min(p.hidden_size, min(p.k_dim, p.emb_dim))));
    const float tot_h = ((tg_a[0] + tg_a[1]) + (tg_a[2] + tg_a[3])) + ((tg_a[4] + tg_a[5]) + (tg_a[6] + tg_a[7]));
    const float tot_k = ((tg_b[0] + tg_b[1]) + (tg_b[2] + tg_b[3])) + ((tg_b[4] + tg_b[5]) + (tg_b[6] + tg_b[7]));
    const float tot_d = ((tg_c[0] + tg_c[1]) + (tg_c[2] + tg_c[3])) + ((tg_c[4] + tg_c[5]) + (tg_c[6] + tg_c[7]));
    const float rstd = rsqrt(tot_h / norm_len + p.rms_eps) * rsqrt(tot_k / norm_len + p.rms_eps);
    const float dot_val = tot_d * rstd * p.inv_sqrt_d;
    const float gate = hybrid_sigmoid(copysign(sqrt(max(fabs(dot_val), 1.0e-6f)), dot_val));

    const ulong val_base = real_hf_wkv ? (wkv_base + 4ul * h) : wkv_base;
    for (uint d = tid; d < p.hidden_size; d += 256u) {
        const float hv = hybrid_load_f16_or_bf16(streams, st_base + d, GR_STREAM_BF16);
        const float vv = float(wkv_out[val_base + d]);
        hybrid_store_f16_or_bf16(streams, st_base + d, fma(gate, vv, hv), GR_STREAM_BF16);
    }
}
