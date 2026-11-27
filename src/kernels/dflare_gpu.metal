// Dedicated Metal 4 compute kernels for M8.25 `D16` `DFlareProposer` (`Inkling 2`
// `3L`/`5L` and `arch = dflash`). Every kernel is engineered for `-Wframe-larger-than=0`
// (`0` stack spills) and zero steady-state allocations.
#include <metal_stdlib>
using namespace metal;

struct DFlarePackTapsParams {
    uint num_taps;
    uint hidden_size;
    uint src_is_f16;
    uint dst_is_f16;
};

struct DFlareMixTapsParams {
    uint num_pos;
    uint num_taps;
    uint hidden_size;
    uint taps_dtype; // 0 = f32, 1 = f16, 2 = bf16
    uint ring_cap;
    uint ring_start;
    uint has_mix_w;
    uint apply_norm;
    uint has_norm_w;
    uint write_f32;
    float rms_eps;
    uint _pad0;
};

struct DFlareSeedBlockParams {
    uint block_size;
    uint hidden_size;
    uint is_real_gru;
    uint mask_rows;
    uint anchor_is_f16;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

struct DFlareRmsNormParams {
    uint num_rows;
    uint dim;
    uint x_is_f32;
    uint has_weight;
    uint write_f32;
    float eps;
    uint _pad0;
    uint _pad1;
};

struct DFlareQkNormRopeParams {
    uint seq_len;
    uint num_q_heads;
    uint num_kv_heads;
    uint head_dim;
    uint anchor_pos;
    uint has_q_norm;
    uint has_k_norm;
    uint rope_dim; // 0 => full head_dim; >0 => partial NeoX RoPE over [0..rope_dim)
    float rms_eps;
    float rope_theta;
    uint _pad1;
    uint _pad2;
};

struct DFlareCtxKvStoreParams {
    uint num_pos;
    uint num_kv_heads;
    uint head_dim;
    uint start_pos;
    uint max_ctx;
    uint layer_elem_off;
    uint has_k_norm;
    uint rope_zero_pos;
    float rms_eps;
    float rope_theta;
    uint rope_dim; // 0 => full head_dim; >0 => partial NeoX RoPE over [0..rope_dim)
    uint _pad1;
};

struct DFlareBlockGqaParams {
    uint seq_len;
    uint active_ctx_len;
    uint num_q_heads;
    uint num_kv_heads;
    uint head_dim;
    uint layer_elem_off;
    float scale;
    float v_scale;
    uint max_ctx;
    uint anchor_pos;
    uint sliding_window;
    uint has_sink_bias;
};

struct DFlarePackCtxTapsParams {
    uint num_pos;
    uint num_taps;
    uint hidden_size;
    uint taps_dtype; // 0 = f32, 1 = f16, 2 = bf16
    uint ring_cap;
    uint ring_start;
    uint _pad0;
    uint _pad1;
};

struct MtpTreeQkNormRopeParams {
    uint num_nodes;
    uint num_q_heads;
    uint num_kv_heads;
    uint head_dim;
    uint v_head_dim;
    uint rope_dim; // 0 => full head_dim
    uint has_q_norm;
    uint has_k_norm;
    uint norm_dtype; // 0 = f16, 1 = bf16, 2 = f32
    float rms_eps;
    float rope_theta;
    float v_scale;
};

struct MtpKvCommitParams {
    uint num_nodes;
    uint num_kv_heads;
    uint head_dim;
    uint v_head_dim;
    uint base_pos;
    uint kv_cap;
    uint layer_k_elem_off;
    uint layer_v_elem_off;
};

struct MtpGqaTreeAttnParams {
    uint num_nodes;
    uint slot_start;
    uint q_count;
    uint base_pos;
    uint kv_cap;
    uint sliding_window;
    uint num_q_heads;
    uint num_kv_heads;
    uint head_dim;
    uint v_head_dim;
    uint layer_k_elem_off;
    uint layer_v_elem_off;
    uint has_sink_bias;
    uint sink_dtype; // 0 = f16, 1 = bf16, 2 = f32
    float scale;
    uint has_sparse_indices;
    uint sparse_k;
    uint sparse_row_stride;
    uint sparse_num_rows;
    uint head_gate_mode; // 0 = none, 1 = head_wise [m, num_q_heads], 2 = element_wise [m, num_q_heads * v_head_dim]
};

struct MtpMlaPrepParams {
    uint num_nodes;
    uint num_heads;
    uint kv_lora_rank;
    uint qk_nope_head_dim;
    uint qk_rope_head_dim;
    uint v_head_dim;
    uint rope_interleave; // 0 = NeoX, 1 = interleaved pairs (GLM-5.2/5.3)
    uint _pad0;
    float rope_theta;
    uint _pad1;
    uint _pad2;
    uint _pad3;
};

struct MtpSwigluClampParams {
    uint count;
    float limit; // <= 0 disables clamping
    uint _pad0;
    uint _pad1;
};

struct MtpMoeRouteParams {
    uint m;
    uint num_experts;
    uint top_k;
    uint use_sigmoid; // 0 = softmax, 1 = sigmoid
    uint has_bias;
    uint bias_dtype; // 0 = f16, 1 = bf16, 2 = f32
    uint norm_topk_prob;
    float routed_scaling_factor;
    uint n_group;     // <= 1 disables group_limited_topk (`8` on Ling-3.1)
    uint topk_group;  // active groups selected by top-2 sum (`4` on Ling-3.1)
    uint _pad0;
    uint _pad1;
};

struct MtpExpertMeta {
    ulong gate_off;
    ulong up_off;
    ulong down_off;
    ulong gate_sb_off;
    ulong up_sb_off;
    ulong down_sb_off;
    uint weight_dtype; // 0 = f16, 1 = bf16, 2 = q4_affine, 3 = q3_affine, 4 = q2_affine
    uint group_size;
    uint _pad0;
    uint _pad1;
};

struct MtpMoeExpertParams {
    uint m;
    uint hidden_size;
    uint moe_inter;
    uint top_k;
    uint weight_dtype; // 0 = f16, 1 = bf16, 2 = q4_affine, 3 = q3_affine, 4 = q2_affine
    uint group_size;
    float swiglu_limit; // <= 0 disables clamping
    uint _pad0;
};

struct DFlash2GroupedConvParams {
    uint seq_len;
    uint hidden_size;
    uint block_size;
    uint taps;
    uint num_groups;
    uint group_size;
    uint side; // 0 = input prepare, 1 = output finish
    uint add_residual;
};

struct DFlash2TopKUnaryParams {
    uint num_slots;
    uint vocab_size;
    uint eff_vocab_size;
    uint top_k; // <= 16
    uint logits_is_f16;
    uint aligned_vec4;
    float output_multiplier;
    float softcap;
};

struct DFlash2SelectorScoreParams {
    uint num_slots;
    uint top_k; // <= 16
    uint state_rank; // <= 512
    uint vocab_size;
    uint anchor_token_id;
    uint pred_dtype; // 0 = f32, 1 = bf16, 2 = f16
    uint succ_dtype; // 0 = f32, 1 = bf16, 2 = f16
    uint hidden_is_f32;
};

struct DFlash2SelectorWalkParams {
    uint num_slots;
    uint top_k;
    uint out_k; // 1 or 2
    float inv_temp;
    uint _pad0;
    uint _pad1;
    uint _pad2;
    uint _pad3;
};

struct DFlareElemParams {
    uint count;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

struct DFlareGemvF32Params {
    uint m;
    uint n;
    uint k;
    uint has_residual;
    uint x_is_f32;
    uint y_is_f32;
    uint res_is_f32;
    uint _pad0;
};

struct DFlareMarkovAssembleParams {
    uint hidden_size;
    uint markov_rank;
    uint slot_pos;
    uint w1_rows;
    uint w1_dtype; // 0 = bf16, 1 = f16, 2 = f32
    uint use_tok_buf;
    uint tok_slot;
    uint imm_tok;
    uint write_f32;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

struct DFlareMarkovGruUpdateParams {
    uint markov_rank;
    uint gates_is_f32;
    uint has_bias;
    uint _pad0;
};

struct DFlareSimpleMarkovPrepParams {
    uint hidden_size;
    uint vocab_size;
    uint slot_pos;
    uint use_tok_buf;
    uint tok_slot;
    uint imm_tok;
    uint tok_dtype; // 0 = f32, 1 = f16, 2 = bf16
    uint has_tok_norm;
    uint concat_2h;
    uint write_f32;
    float rms_eps;
    uint _pad0;
};

struct DFlareSimpleMarkovSiluParams {
    uint markov_rank;
    uint u_is_f32;
    uint add_parent_rnn;
    uint has_b1;
};

struct DFlareMarkovCombineParams {
    uint hidden_size;
    uint w2_is_f32;
    uint has_bias;
    uint has_norm;
    uint slot_pos;
    float rms_eps;
    uint _pad0;
    uint _pad1;
};

struct DFlareW2VocabParams {
    uint vocab_size;
    uint v_rows;
    uint markov_rank;
    uint w2_dtype; // 0 = bf16, 1 = f16, 2 = f32
    uint has_bias;
    uint has_base_logits;
    uint base_row_elem_off;
    float softcap;
};

struct DFlareLogitsEpilogueParams {
    uint vocab_size;
    uint v_rows;
    uint has_bias;
    uint has_base_logits;
    uint base_row_elem_off;
    uint in_is_f16;
    float softcap;
    uint _pad0;
};

struct DFlareTopKCandidate {
    uint token_id;
    float prob;
    float log_prob;
    uint _pad;
};

struct DFlareTopKParams {
    uint eff_vocab_size;
    uint top_k;
    uint logits_is_f16;
    uint has_mask;
    uint out_cand_offset;
    uint write_next_tok;
    uint next_tok_slot;
    uint aligned_vec4;
    float inv_temp;
    float softcap;
    uint _pad0;
    uint _pad1;
};

static_assert(sizeof(DFlarePackTapsParams) == 16, "DFlarePackTapsParams size");
static_assert(sizeof(DFlareMixTapsParams) == 48, "DFlareMixTapsParams size");
static_assert(sizeof(DFlareSeedBlockParams) == 32, "DFlareSeedBlockParams size");
static_assert(sizeof(DFlareRmsNormParams) == 32, "DFlareRmsNormParams size");
static_assert(sizeof(DFlareQkNormRopeParams) == 48, "DFlareQkNormRopeParams size");
static_assert(sizeof(DFlareCtxKvStoreParams) == 48, "DFlareCtxKvStoreParams size");
static_assert(sizeof(DFlareBlockGqaParams) == 48, "DFlareBlockGqaParams size");
static_assert(sizeof(DFlarePackCtxTapsParams) == 32, "DFlarePackCtxTapsParams size");
static_assert(sizeof(MtpTreeQkNormRopeParams) == 48, "MtpTreeQkNormRopeParams size");
static_assert(sizeof(MtpKvCommitParams) == 32, "MtpKvCommitParams size");
static_assert(sizeof(MtpGqaTreeAttnParams) == 80, "MtpGqaTreeAttnParams size");
static_assert(sizeof(MtpMlaPrepParams) == 48, "MtpMlaPrepParams size");
static_assert(sizeof(MtpSwigluClampParams) == 16, "MtpSwigluClampParams size");
static_assert(sizeof(MtpMoeRouteParams) == 48, "MtpMoeRouteParams size");
static_assert(sizeof(MtpExpertMeta) == 64, "MtpExpertMeta size");
static_assert(sizeof(MtpMoeExpertParams) == 32, "MtpMoeExpertParams size");
static_assert(sizeof(DFlash2GroupedConvParams) == 32, "DFlash2GroupedConvParams size");
static_assert(sizeof(DFlash2TopKUnaryParams) == 32, "DFlash2TopKUnaryParams size");
static_assert(sizeof(DFlash2SelectorScoreParams) == 32, "DFlash2SelectorScoreParams size");
static_assert(sizeof(DFlash2SelectorWalkParams) == 32, "DFlash2SelectorWalkParams size");
static_assert(sizeof(DFlareElemParams) == 16, "DFlareElemParams size");
static_assert(sizeof(DFlareGemvF32Params) == 32, "DFlareGemvF32Params size");
static_assert(sizeof(DFlareMarkovAssembleParams) == 48, "DFlareMarkovAssembleParams size");
static_assert(sizeof(DFlareMarkovGruUpdateParams) == 16, "DFlareMarkovGruUpdateParams size");
static_assert(sizeof(DFlareSimpleMarkovPrepParams) == 48, "DFlareSimpleMarkovPrepParams size");
static_assert(sizeof(DFlareSimpleMarkovSiluParams) == 16, "DFlareSimpleMarkovSiluParams size");
static_assert(sizeof(DFlareMarkovCombineParams) == 32, "DFlareMarkovCombineParams size");
static_assert(sizeof(DFlareW2VocabParams) == 32, "DFlareW2VocabParams size");
static_assert(sizeof(DFlareLogitsEpilogueParams) == 32, "DFlareLogitsEpilogueParams size");
static_assert(sizeof(DFlareTopKCandidate) == 16, "DFlareTopKCandidate size");
static_assert(sizeof(DFlareTopKParams) == 48, "DFlareTopKParams size");

static inline float dflare_silu(float x) {
    return x / (1.0f + exp(-clamp(x, -80.0f, 80.0f)));
}

static inline float dflare_sigmoid(float x) {
    return 1.0f / (1.0f + exp(-clamp(x, -80.0f, 80.0f)));
}

static inline float dflare_load_tap_elem(const device uchar *base, ulong idx, uint dtype) {
    if (dtype == 0u) {
        return ((const device float *)base)[idx];
    } else if (dtype == 1u) {
        return float(((const device half *)base)[idx]);
    }
    return float(((const device bfloat *)base)[idx]);
}

static inline float dflare_eval_logit(float raw, float softcap, float inv_temp) {
    float v = raw;
    if (softcap > 0.0f) {
        v = softcap * tanh(clamp(v / softcap, -20.0f, 20.0f));
    }
    return v * inv_temp;
}

// ---------------------------------------------------------------------------
// Kernel 1: Pack up to 8 GPU target tap bindings into contiguous taps_flat.
// Grid: (ceil(hidden_size / 256), num_taps, 1), TG: (256, 1, 1)
// ---------------------------------------------------------------------------
kernel void dflare_pack_taps8(const device uchar *tap0 [[buffer(0)]],
                              const device uchar *tap1 [[buffer(1)]],
                              const device uchar *tap2 [[buffer(2)]],
                              const device uchar *tap3 [[buffer(3)]],
                              const device uchar *tap4 [[buffer(4)]],
                              const device uchar *tap5 [[buffer(5)]],
                              const device uchar *tap6 [[buffer(6)]],
                              const device uchar *tap7 [[buffer(7)]],
                              device uchar *out_raw [[buffer(8)]],
                              constant DFlarePackTapsParams &p [[buffer(9)]],
                              uint2 gid [[thread_position_in_grid]]) {
    const uint d = gid.x;
    const uint ti = gid.y;
    if (d >= p.hidden_size || ti >= p.num_taps) {
        return;
    }
    const device uchar *src = tap0;
    switch (ti) {
        case 0u: src = tap0; break;
        case 1u: src = tap1; break;
        case 2u: src = tap2; break;
        case 3u: src = tap3; break;
        case 4u: src = tap4; break;
        case 5u: src = tap5; break;
        case 6u: src = tap6; break;
        default: src = tap7; break;
    }
    const float val = (p.src_is_f16 != 0u)
        ? float(((const device half *)src)[d])
        : ((const device float *)src)[d];
    const ulong out_idx = ulong(ti) * p.hidden_size + d;
    if (p.dst_is_f16 != 0u) {
        ((device half *)out_raw)[out_idx] = half(val);
    } else {
        ((device float *)out_raw)[out_idx] = val;
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: Scalar-weighted target tap mixing + optional RMSNorm (`[num_pos, h]`)
// Supports contiguous `[num_pos, num_taps, h]` (`ring_cap == 0`) or circular
// `[ring_cap, num_taps, h]` (`ring_cap > 0`).
// Grid: (num_pos, 1, 1), TG: (256, 1, 1)
// ---------------------------------------------------------------------------
kernel void dflare_mix_taps_rmsnorm(const device uchar *taps_raw [[buffer(0)]],
                                    const device float *mix_w [[buffer(1)]],
                                    const device float *norm_w [[buffer(2)]],
                                    device half *out_f16 [[buffer(3)]],
                                    device float *out_f32 [[buffer(4)]],
                                    constant DFlareMixTapsParams &p [[buffer(5)]],
                                    uint pos [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]],
                                    uint sg_id [[simdgroup_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]]) {
    if (pos >= p.num_pos) {
        return;
    }
    const uint src_slot = (p.ring_cap > 0u) ? ((p.ring_start + pos) % p.ring_cap) : pos;
    const ulong pos_tap_base = ulong(src_slot) * p.num_taps * p.hidden_size;

    float sum_sq = 0.0f;
    for (uint d = tid; d < p.hidden_size; d += 256u) {
        float acc = 0.0f;
        if (p.has_mix_w != 0u) {
            for (uint t = 0u; t < p.num_taps; t++) {
                const float tv = dflare_load_tap_elem(taps_raw, pos_tap_base + ulong(t) * p.hidden_size + d, p.taps_dtype);
                acc = fma(mix_w[t], tv, acc);
            }
        } else {
            acc = dflare_load_tap_elem(taps_raw, pos_tap_base + d, p.taps_dtype);
        }
        const ulong out_idx = ulong(pos) * p.hidden_size + d;
        out_f16[out_idx] = half(acc);
        if (p.write_f32 != 0u) {
            out_f32[out_idx] = acc;
        }
        sum_sq = fma(acc, acc, sum_sq);
    }

    if (p.apply_norm == 0u) {
        return;
    }

    threadgroup float sg_sums[8];
    const float sg_tot = simd_sum(sum_sq);
    if (lane == 0u) {
        sg_sums[sg_id] = sg_tot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float tot = 0.0f;
        for (uint s = 0u; s < 8u; s++) {
            tot += sg_sums[s];
        }
        sg_sums[0] = rsqrt(tot / float(p.hidden_size) + p.rms_eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inv_rms = sg_sums[0];
    for (uint d = tid; d < p.hidden_size; d += 256u) {
        const ulong out_idx = ulong(pos) * p.hidden_size + d;
        const float raw = (p.write_f32 != 0u) ? out_f32[out_idx] : float(out_f16[out_idx]);
        const float w = (p.has_norm_w != 0u) ? norm_w[d] : 1.0f;
        const float normed = raw * inv_rms * w;
        out_f16[out_idx] = half(normed);
        if (p.write_f32 != 0u) {
            out_f32[out_idx] = normed;
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: Seed `slot_hiddens[0..block_size-1]` (`f16`).
// Grid: (ceil(hidden_size / 256), block_size, 1), TG: (256, 1, 1)
// ---------------------------------------------------------------------------
kernel void dflare_seed_block(const device uchar *anchor_raw [[buffer(0)]],
                              const device half *ctx_hidden_f16 [[buffer(1)]],
                              const device float *mask_embed_f32 [[buffer(2)]],
                              device half *out_slots_f16 [[buffer(3)]],
                              constant DFlareSeedBlockParams &p [[buffer(4)]],
                              uint2 gid [[thread_position_in_grid]]) {
    const uint d = gid.x;
    const uint pos = gid.y;
    if (d >= p.hidden_size || pos >= p.block_size) {
        return;
    }
    float val = 0.0f;
    if (p.is_real_gru != 0u) {
        if (pos == 0u) {
            val = (p.anchor_is_f16 != 0u)
                ? float(((const device half *)anchor_raw)[d])
                : ((const device float *)anchor_raw)[d];
        } else {
            uint m_row = 0u;
            if (p.block_size > 1u && p.mask_rows == p.block_size - 1u) {
                m_row = pos - 1u;
            } else if (p.mask_rows == p.block_size) {
                m_row = pos;
            }
            val = mask_embed_f32[ulong(m_row) * p.hidden_size + d];
        }
    } else {
        const float ch = float(ctx_hidden_f16[d]);
        if (p.block_size > 1u && p.mask_rows == p.block_size - 1u) {
            val = (pos == 0u) ? ch : (ch + mask_embed_f32[ulong(pos - 1u) * p.hidden_size + d]);
        } else {
            const uint m_row = (p.mask_rows == p.block_size) ? pos : 0u;
            val = ch + mask_embed_f32[ulong(m_row) * p.hidden_size + d];
        }
    }
    out_slots_f16[ulong(pos) * p.hidden_size + d] = half(val);
}

// ---------------------------------------------------------------------------
// Kernel 4: General RMSNorm (`[num_rows, dim]`, weighted or unweighted, in-place safe)
// Grid: (num_rows, 1, 1), TG: (256, 1, 1)
// ---------------------------------------------------------------------------
kernel void dflare_rmsnorm_rows(const device uchar *x_raw [[buffer(0)]],
                                const device float *w_f32 [[buffer(1)]],
                                device half *y_f16 [[buffer(2)]],
                                device float *y_f32 [[buffer(3)]],
                                constant DFlareRmsNormParams &p [[buffer(4)]],
                                uint row [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]],
                                uint sg_id [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    if (row >= p.num_rows) {
        return;
    }
    const ulong row_base = ulong(row) * p.dim;
    float sum_sq = 0.0f;
    if (p.x_is_f32 != 0u) {
        const device float *xr = ((const device float *)x_raw) + row_base;
        for (uint i = tid; i < p.dim; i += 256u) {
            const float v = xr[i];
            sum_sq = fma(v, v, sum_sq);
        }
    } else {
        const device half *xr = ((const device half *)x_raw) + row_base;
        for (uint i = tid; i < p.dim; i += 256u) {
            const float v = float(xr[i]);
            sum_sq = fma(v, v, sum_sq);
        }
    }

    threadgroup float sg_sums[8];
    const float sg_tot = simd_sum(sum_sq);
    if (lane == 0u) {
        sg_sums[sg_id] = sg_tot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float tot = 0.0f;
        for (uint s = 0u; s < 8u; s++) {
            tot += sg_sums[s];
        }
        sg_sums[0] = rsqrt(tot / float(p.dim) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inv_rms = sg_sums[0];
    if (p.x_is_f32 != 0u) {
        const device float *xr = ((const device float *)x_raw) + row_base;
        for (uint i = tid; i < p.dim; i += 256u) {
            const float w = (p.has_weight != 0u) ? w_f32[i] : 1.0f;
            const float outv = xr[i] * inv_rms * w;
            y_f16[row_base + i] = half(outv);
            if (p.write_f32 != 0u) {
                y_f32[row_base + i] = outv;
            }
        }
    } else {
        const device half *xr = ((const device half *)x_raw) + row_base;
        for (uint i = tid; i < p.dim; i += 256u) {
            const float w = (p.has_weight != 0u) ? w_f32[i] : 1.0f;
            const float outv = float(xr[i]) * inv_rms * w;
            y_f16[row_base + i] = half(outv);
            if (p.write_f32 != 0u) {
                y_f32[row_base + i] = outv;
            }
        }
    }
}

// Helper for shared-memory per-head RMSNorm + optional partial NeoX RoPE (head_dim <= 256).
static inline void dflare_head_norm_rope_tg(threadgroup float *tg_h,
                                            device half *head_ptr,
                                            const device float *norm_w,
                                            uint head_dim,
                                            uint rope_dim,
                                            uint apply_norm,
                                            uint tok_pos,
                                            float rms_eps,
                                            float rope_theta,
                                            uint tid,
                                            uint sg_id,
                                            uint lane,
                                            threadgroup float *sg_sums) {
    float sum_sq = 0.0f;
    for (uint d = tid; d < head_dim; d += 128u) {
        const float v = float(head_ptr[d]);
        tg_h[d] = v;
        sum_sq = fma(v, v, sum_sq);
    }

    if (apply_norm != 0u) {
        const float sg_tot = simd_sum(sum_sq);
        if (lane == 0u) {
            sg_sums[sg_id] = sg_tot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float tot = 0.0f;
            for (uint s = 0u; s < 4u; s++) {
                tot += sg_sums[s];
            }
            sg_sums[0] = rsqrt(tot / float(head_dim) + rms_eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float inv_rms = sg_sums[0];
        for (uint d = tid; d < head_dim; d += 128u) {
            tg_h[d] = tg_h[d] * inv_rms * norm_w[d];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint eff_rope = (rope_dim > 0u && rope_dim <= head_dim) ? rope_dim : head_dim;
    if (rope_theta > 0.0f && eff_rope >= 2u) {
        const uint half_rot = eff_rope >> 1;
        const float inv_rot = 2.0f / float(eff_rope);
        const float pos_f = float(tok_pos);
        for (uint i = tid; i < half_rot; i += 128u) {
            const float freq = 1.0f / pow(rope_theta, float(i) * inv_rot);
            const float angle = pos_f * freq;
            float c_val;
            const float s_val = sincos(angle, c_val);
            const float x0 = tg_h[i];
            const float x1 = tg_h[i + half_rot];
            head_ptr[i] = half(x0 * c_val - x1 * s_val);
            head_ptr[i + half_rot] = half(x0 * s_val + x1 * c_val);
        }
        for (uint d = eff_rope + tid; d < head_dim; d += 128u) {
            head_ptr[d] = half(tg_h[d]);
        }
    } else {
        for (uint d = tid; d < head_dim; d += 128u) {
            head_ptr[d] = half(tg_h[d]);
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 5: Fused Q/K Per-Head RMSNorm + Partial NeoX RoPE (`head_dim <= 256`)
// Grid: (num_q_heads + num_kv_heads, seq_len, 1), TG: (128, 1, 1)
// ---------------------------------------------------------------------------
kernel void dflare_qk_norm_rope(device half *q_f16 [[buffer(0)]],
                                device half *k_f16 [[buffer(1)]],
                                const device float *qn_f32 [[buffer(2)]],
                                const device float *kn_f32 [[buffer(3)]],
                                constant DFlareQkNormRopeParams &p [[buffer(4)]],
                                uint2 tg_pos [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]],
                                uint sg_id [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    const uint head_global = tg_pos.x;
    const uint seq_idx = tg_pos.y;
    const uint total_heads = p.num_q_heads + p.num_kv_heads;
    if (head_global >= total_heads || seq_idx >= p.seq_len || p.head_dim > 256u) {
        return;
    }

    threadgroup float tg_h[256];
    threadgroup float sg_sums[4];

    const bool is_q = (head_global < p.num_q_heads);
    const uint h_idx = is_q ? head_global : (head_global - p.num_q_heads);
    const uint stride = (is_q ? p.num_q_heads : p.num_kv_heads) * p.head_dim;
    device half *head_ptr = (is_q ? q_f16 : k_f16) + ulong(seq_idx) * stride + ulong(h_idx) * p.head_dim;
    const device float *norm_w = is_q ? qn_f32 : kn_f32;
    const uint apply_norm = is_q ? p.has_q_norm : p.has_k_norm;

    dflare_head_norm_rope_tg(
        tg_h,
        head_ptr,
        norm_w,
        p.head_dim,
        p.rope_dim,
        apply_norm,
        p.anchor_pos + seq_idx,
        p.rms_eps,
        p.rope_theta,
        tid,
        sg_id,
        lane,
        sg_sums);
}

// ---------------------------------------------------------------------------
// Kernel 6: Context K/V Per-Head Norm + Partial NeoX RoPE + Circular Cache Store
// Grid: (num_kv_heads, num_pos, 1), TG: (128, 1, 1)
// ---------------------------------------------------------------------------
kernel void dflare_ctx_kv_norm_rope_store(const device half *k_in [[buffer(0)]],
                                          const device half *v_in [[buffer(1)]],
                                          const device float *kn_f32 [[buffer(2)]],
                                          device half *ctx_k_cache [[buffer(3)]],
                                          device half *ctx_v_cache [[buffer(4)]],
                                          constant DFlareCtxKvStoreParams &p [[buffer(5)]],
                                          uint2 tg_pos [[threadgroup_position_in_grid]],
                                          uint tid [[thread_index_in_threadgroup]],
                                          uint sg_id [[simdgroup_index_in_threadgroup]],
                                          uint lane [[thread_index_in_simdgroup]]) {
    const uint kvh = tg_pos.x;
    const uint pos_idx = tg_pos.y;
    if (kvh >= p.num_kv_heads || pos_idx >= p.num_pos || p.head_dim > 256u) {
        return;
    }

    const uint kv_dim = p.num_kv_heads * p.head_dim;
    const ulong src_off = ulong(pos_idx) * kv_dim + ulong(kvh) * p.head_dim;
    const uint slot = (p.start_pos + pos_idx) % p.max_ctx;
    const ulong dst_off = ulong(p.layer_elem_off) + ulong(slot) * kv_dim + ulong(kvh) * p.head_dim;

    // Copy V directly.
    for (uint d = tid; d < p.head_dim; d += 128u) {
        ctx_v_cache[dst_off + d] = v_in[src_off + d];
    }

    threadgroup float tg_h[256];
    threadgroup float sg_sums[4];

    float sum_sq = 0.0f;
    for (uint d = tid; d < p.head_dim; d += 128u) {
        const float v = float(k_in[src_off + d]);
        tg_h[d] = v;
        sum_sq = fma(v, v, sum_sq);
    }

    if (p.has_k_norm != 0u) {
        const float sg_tot = simd_sum(sum_sq);
        if (lane == 0u) {
            sg_sums[sg_id] = sg_tot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float tot = 0.0f;
            for (uint s = 0u; s < 4u; s++) {
                tot += sg_sums[s];
            }
            sg_sums[0] = rsqrt(tot / float(p.head_dim) + p.rms_eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float inv_rms = sg_sums[0];
        for (uint d = tid; d < p.head_dim; d += 128u) {
            tg_h[d] = tg_h[d] * inv_rms * kn_f32[d];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint eff_rope = (p.rope_dim > 0u && p.rope_dim <= p.head_dim) ? p.rope_dim : p.head_dim;
    if (p.rope_theta > 0.0f && eff_rope >= 2u) {
        const uint half_rot = eff_rope >> 1;
        const float inv_rot = 2.0f / float(eff_rope);
        const uint rope_pos = (p.rope_zero_pos != 0u) ? 0u : (p.start_pos + pos_idx);
        const float pos_f = float(rope_pos);
        for (uint i = tid; i < half_rot; i += 128u) {
            const float freq = 1.0f / pow(p.rope_theta, float(i) * inv_rot);
            const float angle = pos_f * freq;
            float c_val;
            const float s_val = sincos(angle, c_val);
            const float x0 = tg_h[i];
            const float x1 = tg_h[i + half_rot];
            ctx_k_cache[dst_off + i] = half(x0 * c_val - x1 * s_val);
            ctx_k_cache[dst_off + i + half_rot] = half(x0 * s_val + x1 * c_val);
        }
        for (uint d = eff_rope + tid; d < p.head_dim; d += 128u) {
            ctx_k_cache[dst_off + d] = half(tg_h[d]);
        }
    } else {
        for (uint d = tid; d < p.head_dim; d += 128u) {
            ctx_k_cache[dst_off + d] = half(tg_h[d]);
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 7: Bidirectional Block GQA Attention over `[ctx_kv, block_kv]` (`f16`)
// with optional causal SWA over circular context ring, `attention_sink_bias`,
// and `attention_value_scale` (`D15` DFlash + `D16` Inkling 2).
// Grid: (num_q_heads, seq_len, 1), TG: (128, 1, 1) (4 simdgroups splitting KV)
// ---------------------------------------------------------------------------
kernel void dflare_block_gqa_attn(const device half *q_f16 [[buffer(0)]],
                                  const device half *ctx_k_f16 [[buffer(1)]],
                                  const device half *ctx_v_f16 [[buffer(2)]],
                                  const device half *blk_k_f16 [[buffer(3)]],
                                  const device half *blk_v_f16 [[buffer(4)]],
                                  device half *out_f16 [[buffer(5)]],
                                  const device float *sink_bias_f32 [[buffer(6)]],
                                  constant DFlareBlockGqaParams &p [[buffer(7)]],
                                  uint2 tg_pos [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]],
                                  uint sg_id [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
    const uint qh = tg_pos.x;
    const uint q_pos = tg_pos.y;
    if (qh >= p.num_q_heads || q_pos >= p.seq_len || p.head_dim > 256u) {
        return;
    }

    const uint gqa_group = p.num_q_heads / p.num_kv_heads;
    const uint kvh = qh / gqa_group;
    const uint q_dim = p.num_q_heads * p.head_dim;
    const uint kv_dim = p.num_kv_heads * p.head_dim;
    const ulong q_off = ulong(q_pos) * q_dim + ulong(qh) * p.head_dim;

    threadgroup float tg_q[256];
    threadgroup float tg_acc[4][256];
    threadgroup float tg_m[5];
    threadgroup float tg_l[5];

    for (uint d = tid; d < p.head_dim; d += 128u) {
        tg_q[d] = float(q_f16[q_off + d]);
        tg_acc[0][d] = 0.0f;
        tg_acc[1][d] = 0.0f;
        tg_acc[2][d] = 0.0f;
        tg_acc[3][d] = 0.0f;
    }
    if (tid < 5u) {
        tg_m[tid] = -INFINITY;
        tg_l[tid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Causal SWA window over circular context ring + non-causal bidirectional block `[0..seq_len-1]`.
    const uint ctx_base = (p.anchor_pos >= p.active_ctx_len) ? (p.anchor_pos - p.active_ctx_len) : 0u;
    uint eff_ctx_start = ctx_base;
    uint eff_ctx_len = p.active_ctx_len;
    if (p.sliding_window > 0u && p.anchor_pos >= p.active_ctx_len) {
        const uint q_log_pos = p.anchor_pos + q_pos;
        const uint min_k_pos = (q_log_pos + 1u > p.sliding_window) ? (q_log_pos + 1u - p.sliding_window) : 0u;
        eff_ctx_start = max(ctx_base, min_k_pos);
        eff_ctx_len = (p.anchor_pos > eff_ctx_start) ? (p.anchor_pos - eff_ctx_start) : 0u;
    }

    float m_sg = -INFINITY;
    float l_sg = 0.0f;
    const uint total_kv = eff_ctx_len + p.seq_len;

    for (uint kv_idx = sg_id; kv_idx < total_kv; kv_idx += 4u) {
        const bool is_ctx = (kv_idx < eff_ctx_len);
        const uint ctx_slot = (p.max_ctx > 0u) ? ((eff_ctx_start + kv_idx) % p.max_ctx) : (eff_ctx_start + kv_idx);
        const ulong kv_off = is_ctx
            ? (ulong(p.layer_elem_off) + ulong(ctx_slot) * kv_dim + ulong(kvh) * p.head_dim)
            : (ulong(kv_idx - eff_ctx_len) * kv_dim + ulong(kvh) * p.head_dim);
        const device half *k_head = (is_ctx ? ctx_k_f16 : blk_k_f16) + kv_off;
        const device half *v_head = (is_ctx ? ctx_v_f16 : blk_v_f16) + kv_off;

        float dot_partial = 0.0f;
        for (uint d = lane; d < p.head_dim; d += 32u) {
            dot_partial = fma(tg_q[d], float(k_head[d]), dot_partial);
        }
        const float score = simd_sum(dot_partial) * p.scale;
        const float m_new = max(m_sg, score);
        const float alpha = (l_sg > 0.0f) ? exp(m_sg - m_new) : 0.0f;
        const float prob = exp(score - m_new);
        l_sg = fma(l_sg, alpha, prob);
        m_sg = m_new;

        for (uint d = lane; d < p.head_dim; d += 32u) {
            tg_acc[sg_id][d] = fma(prob, float(v_head[d]), tg_acc[sg_id][d] * alpha);
        }
    }

    if (lane == 0u) {
        tg_m[sg_id] = m_sg;
        tg_l[sg_id] = l_sg;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float gm = -INFINITY;
        for (uint s = 0u; s < 4u; s++) {
            if (tg_l[s] > 0.0f) {
                gm = max(gm, tg_m[s]);
            }
        }
        if (p.has_sink_bias != 0u) {
            gm = max(gm, sink_bias_f32[qh]);
        }
        float gl = 0.0f;
        for (uint s = 0u; s < 4u; s++) {
            if (tg_l[s] > 0.0f) {
                gl += tg_l[s] * exp(tg_m[s] - gm);
            }
        }
        if (p.has_sink_bias != 0u) {
            gl += exp(sink_bias_f32[qh] - gm);
        }
        tg_m[4] = gm;
        tg_l[4] = (gl > 0.0f) ? (1.0f / gl) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float gm = tg_m[4];
    const float out_scale = tg_l[4] * p.v_scale;
    for (uint d = tid; d < p.head_dim; d += 128u) {
        float acc = 0.0f;
        for (uint s = 0u; s < 4u; s++) {
            if (isfinite(tg_m[s])) {
                acc = fma(tg_acc[s][d], exp(tg_m[s] - gm), acc);
            }
        }
        out_f16[q_off + d] = half(acc * out_scale);
    }
}

// ---------------------------------------------------------------------------
// Kernel 8: Elementwise SwiGLU (`gate[i] = silu(gate[i]) * up[i]`, `f16`)
// Grid: (ceil(count / 256), 1, 1), TG: (256, 1, 1)
// ---------------------------------------------------------------------------
kernel void dflare_swiglu_mul(device half *gate_f16 [[buffer(0)]],
                              const device half *up_f16 [[buffer(1)]],
                              constant DFlareElemParams &p [[buffer(2)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= p.count) {
        return;
    }
    const float g = float(gate_f16[gid]);
    const float u = float(up_f16[gid]);
    gate_f16[gid] = half(dflare_silu(g) * u);
}

// ---------------------------------------------------------------------------
// Kernel 9: Batched FP32-Weight GEMV (`W[n, k]` f32, arbitrary `m, n, k`)
// Grid: (ceil(n / 4), m, 1), TG: (128, 1, 1) (4 simdgroups, 1 row/simdgroup)
// ---------------------------------------------------------------------------
kernel void dflare_gemv_batch_f32(const device uchar *x_raw [[buffer(0)]],
                                  const device float *w_f32 [[buffer(1)]],
                                  device uchar *y_raw [[buffer(2)]],
                                  const device uchar *res_raw [[buffer(3)]],
                                  constant DFlareGemvF32Params &p [[buffer(4)]],
                                  uint2 tg_pos [[threadgroup_position_in_grid]],
                                  uint sg_id [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
    const uint row = tg_pos.x * 4u + sg_id;
    const uint m_idx = tg_pos.y;
    if (row >= p.n || m_idx >= p.m) {
        return;
    }

    const device float *w_row = w_f32 + ulong(row) * p.k;
    const ulong x_off = ulong(m_idx) * p.k;
    float acc = 0.0f;

    if (p.x_is_f32 != 0u) {
        const device float *xr = ((const device float *)x_raw) + x_off;
        for (uint c = lane; c < p.k; c += 32u) {
            acc = fma(w_row[c], xr[c], acc);
        }
    } else {
        const device half *xr = ((const device half *)x_raw) + x_off;
        for (uint c = lane; c < p.k; c += 32u) {
            acc = fma(w_row[c], float(xr[c]), acc);
        }
    }

    const float tot = simd_sum(acc);
    if (lane == 0u) {
        const ulong y_idx = ulong(m_idx) * p.n + row;
        float out_v = tot;
        if (p.has_residual != 0u) {
            const float rv = (p.res_is_f32 != 0u)
                ? ((const device float *)res_raw)[y_idx]
                : float(((const device half *)res_raw)[y_idx]);
            out_v += rv;
        }
        if (p.y_is_f32 != 0u) {
            ((device float *)y_raw)[y_idx] = out_v;
        } else {
            ((device half *)y_raw)[y_idx] = half(out_v);
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 10: Assemble Real 3-Gate Markov GRU Input `[parent_rnn || w1[tok] || b_j || f_rnn]`
// Grid: (ceil((2*mr + 2*h) / 256), 1, 1), TG: (256, 1, 1)
// ---------------------------------------------------------------------------
kernel void dflare_markov_gru_assemble(const device float *parent_rnn_f32 [[buffer(0)]],
                                       const device uchar *w1_vocab_raw [[buffer(1)]],
                                       const device half *slot_hiddens_f16 [[buffer(2)]],
                                       const device half *ctx_markov_f16 [[buffer(3)]],
                                       const device uint *token_chain_u32 [[buffer(4)]],
                                       device half *out_concat_f16 [[buffer(5)]],
                                       device float *out_concat_f32 [[buffer(6)]],
                                       constant DFlareMarkovAssembleParams &p [[buffer(7)]],
                                       uint gid [[thread_position_in_grid]]) {
    const uint mr = p.markov_rank;
    const uint h = p.hidden_size;
    const uint total = 2u * mr + 2u * h;
    if (gid >= total) {
        return;
    }

    float val = 0.0f;
    if (gid < mr) {
        val = parent_rnn_f32[gid];
    } else if (gid < 2u * mr) {
        const uint raw_tok = (p.use_tok_buf != 0u) ? token_chain_u32[p.tok_slot] : p.imm_tok;
        const uint tok_clamped = (raw_tok < p.w1_rows) ? raw_tok : 0u;
        const ulong elem = ulong(tok_clamped) * mr + (gid - mr);
        if (p.w1_dtype == 0u) {
            val = float(((const device bfloat *)w1_vocab_raw)[elem]);
        } else if (p.w1_dtype == 1u) {
            val = float(((const device half *)w1_vocab_raw)[elem]);
        } else if (p.w1_dtype == 3u) {
            // Native GGUF Q8_0 row lookup: 34 B per 32-element block (f16 scale + 32 x int8).
            const uint col = gid - mr;
            const uint blks_per_row = mr >> 5;
            const uint blk_idx = col >> 5;
            const uint in_blk = col & 31u;
            const device uchar *bp = w1_vocab_raw + (ulong(tok_clamped) * blks_per_row + blk_idx) * 34ul;
            const ushort d_bits = ushort(bp[0]) | (ushort(bp[1]) << 8);
            const float d_sc = float(as_type<half>(d_bits));
            val = float(as_type<char>(bp[2u + in_blk])) * d_sc;
        } else {
            val = ((const device float *)w1_vocab_raw)[elem];
        }
    } else if (gid < 2u * mr + h) {
        val = float(slot_hiddens_f16[ulong(p.slot_pos) * h + (gid - 2u * mr)]);
    } else {
        val = float(ctx_markov_f16[gid - (2u * mr + h)]);
    }

    out_concat_f16[gid] = half(val);
    if (p.write_f32 != 0u) {
        out_concat_f32[gid] = val;
    }
}

// ---------------------------------------------------------------------------
// Kernel 11: Real 3-Chunk Markov GRU Gate Update
// Grid: (ceil(markov_rank / 256), 1, 1), TG: (256, 1, 1)
// ---------------------------------------------------------------------------
kernel void dflare_markov_gru_update(const device uchar *gates_raw [[buffer(0)]],
                                     const device float *bias_f32 [[buffer(1)]],
                                     const device float *parent_rnn_f32 [[buffer(2)]],
                                     device float *out_rnn_f32 [[buffer(3)]],
                                     device half *out_vec_f16 [[buffer(4)]],
                                     device float *out_vec_f32 [[buffer(5)]],
                                     constant DFlareMarkovGruUpdateParams &p [[buffer(6)]],
                                     uint i [[thread_position_in_grid]]) {
    const uint mr = p.markov_rank;
    if (i >= mr) {
        return;
    }
    float g0, g1, g2;
    if (p.gates_is_f32 != 0u) {
        const device float *gf = (const device float *)gates_raw;
        g0 = gf[i];
        g1 = gf[mr + i];
        g2 = gf[2u * mr + i];
    } else {
        const device half *gh = (const device half *)gates_raw;
        g0 = float(gh[i]);
        g1 = float(gh[mr + i]);
        g2 = float(gh[2u * mr + i]);
    }
    if (p.has_bias != 0u) {
        g0 += bias_f32[i];
        g1 += bias_f32[mr + i];
        g2 += bias_f32[2u * mr + i];
    }
    const float gate = dflare_sigmoid(g0);
    const float cand = tanh(clamp(g1, -20.0f, 20.0f));
    const float pr = parent_rnn_f32[i];
    const float mvec = tanh(clamp(g2, -20.0f, 20.0f));

    out_rnn_f32[i] = fma(gate, pr, (1.0f - gate) * cand);
    out_vec_f16[i] = half(mvec);
    out_vec_f32[i] = mvec;
}

// ---------------------------------------------------------------------------
// Kernel 12: Prepare Simple (non-GRU) Markov Input `[b_j || tok_e]` or `b_j + tok_e`
// Grid: (1, 1, 1), TG: (256, 1, 1)
// ---------------------------------------------------------------------------
kernel void dflare_simple_markov_prep(const device uchar *tok_embed_raw [[buffer(0)]],
                                      const device float *tok_norm_f32 [[buffer(1)]],
                                      const device half *slot_hiddens_f16 [[buffer(2)]],
                                      const device uint *token_chain_u32 [[buffer(3)]],
                                      device half *out_concat_f16 [[buffer(4)]],
                                      device float *out_concat_f32 [[buffer(5)]],
                                      constant DFlareSimpleMarkovPrepParams &p [[buffer(6)]],
                                      uint tid [[thread_index_in_threadgroup]],
                                      uint sg_id [[simdgroup_index_in_threadgroup]],
                                      uint lane [[thread_index_in_simdgroup]]) {
    const uint raw_tok = (p.use_tok_buf != 0u) ? token_chain_u32[p.tok_slot] : p.imm_tok;
    const uint tok = (raw_tok < p.vocab_size) ? raw_tok : 0u;
    const ulong tok_base = ulong(tok) * p.hidden_size;

    threadgroup float sg_sums[8];
    float inv_rms = 1.0f;
    if (p.has_tok_norm != 0u) {
        float sum_sq = 0.0f;
        for (uint d = tid; d < p.hidden_size; d += 256u) {
            const float tv = dflare_load_tap_elem(tok_embed_raw, tok_base + d, p.tok_dtype);
            sum_sq = fma(tv, tv, sum_sq);
        }
        const float sg_tot = simd_sum(sum_sq);
        if (lane == 0u) {
            sg_sums[sg_id] = sg_tot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float tot = 0.0f;
            for (uint s = 0u; s < 8u; s++) {
                tot += sg_sums[s];
            }
            sg_sums[0] = rsqrt(tot / float(p.hidden_size) + p.rms_eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        inv_rms = sg_sums[0];
    }

    const ulong bj_base = ulong(p.slot_pos) * p.hidden_size;
    for (uint d = tid; d < p.hidden_size; d += 256u) {
        const float bj = float(slot_hiddens_f16[bj_base + d]);
        float te = dflare_load_tap_elem(tok_embed_raw, tok_base + d, p.tok_dtype);
        if (p.has_tok_norm != 0u) {
            te = te * inv_rms * tok_norm_f32[d];
        }
        if (p.concat_2h != 0u) {
            out_concat_f16[d] = half(bj);
            out_concat_f16[p.hidden_size + d] = half(te);
            if (p.write_f32 != 0u) {
                out_concat_f32[d] = bj;
                out_concat_f32[p.hidden_size + d] = te;
            }
        } else {
            const float s = bj + te;
            out_concat_f16[d] = half(s);
            if (p.write_f32 != 0u) {
                out_concat_f32[d] = s;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 13: Simple Markov SiLU Activation (`out_rnn = silu(u + [parent_rnn] + [b1])`)
// Grid: (ceil(markov_rank / 256), 1, 1), TG: (256, 1, 1)
// ---------------------------------------------------------------------------
kernel void dflare_simple_markov_silu(const device uchar *u_raw [[buffer(0)]],
                                      const device float *parent_rnn_f32 [[buffer(1)]],
                                      const device float *b1_f32 [[buffer(2)]],
                                      device float *out_rnn_f32 [[buffer(3)]],
                                      device half *out_rnn_f16 [[buffer(4)]],
                                      constant DFlareSimpleMarkovSiluParams &p [[buffer(5)]],
                                      uint i [[thread_position_in_grid]]) {
    if (i >= p.markov_rank) {
        return;
    }
    float u = (p.u_is_f32 != 0u)
        ? ((const device float *)u_raw)[i]
        : float(((const device half *)u_raw)[i]);
    if (p.add_parent_rnn != 0u) {
        u += parent_rnn_f32[i];
    }
    if (p.has_b1 != 0u) {
        u += b1_f32[i];
    }
    const float y = dflare_silu(u);
    out_rnn_f32[i] = y;
    out_rnn_f16[i] = half(y);
}

// ---------------------------------------------------------------------------
// Kernel 14: Combine `head_hidden = b_j + w2_out + [hidden_bias]` + optional RMSNorm
// Grid: (1, 1, 1), TG: (256, 1, 1)
// ---------------------------------------------------------------------------
kernel void dflare_markov_head_combine(const device uchar *w2_out_raw [[buffer(0)]],
                                       const device float *hidden_bias_f32 [[buffer(1)]],
                                       const device half *slot_hiddens_f16 [[buffer(2)]],
                                       const device float *markov_norm_f32 [[buffer(3)]],
                                       device half *out_head_f16 [[buffer(4)]],
                                       constant DFlareMarkovCombineParams &p [[buffer(5)]],
                                       uint tid [[thread_index_in_threadgroup]],
                                       uint sg_id [[simdgroup_index_in_threadgroup]],
                                       uint lane [[thread_index_in_simdgroup]]) {
    const ulong bj_base = ulong(p.slot_pos) * p.hidden_size;
    float sum_sq = 0.0f;

    for (uint d = tid; d < p.hidden_size; d += 256u) {
        float val = (p.w2_is_f32 != 0u)
            ? ((const device float *)w2_out_raw)[d]
            : float(((const device half *)w2_out_raw)[d]);
        if (p.has_bias != 0u) {
            val += hidden_bias_f32[d];
        }
        val += float(slot_hiddens_f16[bj_base + d]);
        out_head_f16[d] = half(val);
        sum_sq = fma(val, val, sum_sq);
    }

    if (p.has_norm == 0u) {
        return;
    }

    threadgroup float sg_sums[8];
    const float sg_tot = simd_sum(sum_sq);
    if (lane == 0u) {
        sg_sums[sg_id] = sg_tot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float tot = 0.0f;
        for (uint s = 0u; s < 8u; s++) {
            tot += sg_sums[s];
        }
        sg_sums[0] = rsqrt(tot / float(p.hidden_size) + p.rms_eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inv_rms = sg_sums[0];
    for (uint d = tid; d < p.hidden_size; d += 256u) {
        const float val = float(out_head_f16[d]) * inv_rms * markov_norm_f32[d];
        out_head_f16[d] = half(val);
    }
}

// ---------------------------------------------------------------------------
// Kernel 15a: Fused `markov_w2_vocab @ markov_vec + bias + base_logits + softcap -> f32`
// Supports `bf16` (`0`), `f16` (`1`), and `f32` (`2`) `w2_vocab`.
// Grid: (ceil(vocab_size / 4), 1, 1), TG: (128, 1, 1) (1 row per simdgroup)
// ---------------------------------------------------------------------------
kernel void dflare_markov_w2_vocab_score(const device float *markov_vec_f32 [[buffer(0)]],
                                         const device uchar *w2_vocab_raw [[buffer(1)]],
                                         const device float *bias_f32 [[buffer(2)]],
                                         const device half *base_logits_f16 [[buffer(3)]],
                                         device float *out_logits_f32 [[buffer(4)]],
                                         constant DFlareW2VocabParams &p [[buffer(5)]],
                                         uint tg_id [[threadgroup_position_in_grid]],
                                         uint sg_id [[simdgroup_index_in_threadgroup]],
                                         uint lane [[thread_index_in_simdgroup]]) {
    const uint row = tg_id * 4u + sg_id;
    if (row >= p.vocab_size) {
        return;
    }
    if (row >= p.v_rows) {
        if (lane == 0u) {
            out_logits_f32[row] = -1e30f;
        }
        return;
    }

    const ulong row_base = ulong(row) * p.markov_rank;
    float acc = 0.0f;

    if ((p.markov_rank & 3u) == 0u) {
        const uint vec_k = p.markov_rank >> 2;
        const device float4 *x4 = (const device float4 *)markov_vec_f32;
        if (p.w2_dtype == 0u) {
            const device bfloat4 *w4 = ((const device bfloat4 *)w2_vocab_raw) + (row_base >> 2);
            for (uint i = lane; i < vec_k; i += 32u) {
                acc += dot(float4(w4[i]), x4[i]);
            }
        } else if (p.w2_dtype == 1u) {
            const device half4 *w4 = ((const device half4 *)w2_vocab_raw) + (row_base >> 2);
            for (uint i = lane; i < vec_k; i += 32u) {
                acc += dot(float4(w4[i]), x4[i]);
            }
        } else {
            const device float4 *w4 = ((const device float4 *)w2_vocab_raw) + (row_base >> 2);
            for (uint i = lane; i < vec_k; i += 32u) {
                acc += dot(w4[i], x4[i]);
            }
        }
    } else {
        for (uint c = lane; c < p.markov_rank; c += 32u) {
            const float wv = dflare_load_tap_elem(w2_vocab_raw, row_base + c, p.w2_dtype);
            acc = fma(wv, markov_vec_f32[c], acc);
        }
    }

    const float tot = simd_sum(acc);
    if (lane == 0u) {
        float lv = tot;
        if (p.has_bias != 0u) {
            lv += bias_f32[row];
        }
        if (p.has_base_logits != 0u) {
            lv += float(base_logits_f16[ulong(p.base_row_elem_off) + row]);
        }
        if (p.softcap > 0.0f) {
            lv = p.softcap * tanh(clamp(lv / p.softcap, -20.0f, 20.0f));
        }
        out_logits_f32[row] = lv;
    }
}

// ---------------------------------------------------------------------------
// Kernel 15b: Logits epilogue (for quantized `w2_vocab` or `base_logits + bias`)
// Grid: (ceil(vocab_size / 256), 1, 1), TG: (256, 1, 1)
// ---------------------------------------------------------------------------
kernel void dflare_logits_epilogue(const device uchar *logits_in_raw [[buffer(0)]],
                                   const device float *bias_f32 [[buffer(1)]],
                                   const device half *base_logits_f16 [[buffer(2)]],
                                   device float *out_logits_f32 [[buffer(3)]],
                                   constant DFlareLogitsEpilogueParams &p [[buffer(4)]],
                                   uint row [[thread_position_in_grid]]) {
    if (row >= p.vocab_size) {
        return;
    }
    if (row >= p.v_rows) {
        out_logits_f32[row] = -1e30f;
        return;
    }
    float lv = (p.in_is_f16 != 0u)
        ? float(((const device half *)logits_in_raw)[row])
        : ((const device float *)logits_in_raw)[row];
    if (p.has_bias != 0u) {
        lv += bias_f32[row];
    }
    if (p.has_base_logits != 0u) {
        lv += float(base_logits_f16[ulong(p.base_row_elem_off) + row]);
    }
    if (p.softcap > 0.0f) {
        lv = p.softcap * tanh(clamp(lv / p.softcap, -20.0f, 20.0f));
    }
    out_logits_f32[row] = lv;
}

// ---------------------------------------------------------------------------
// Kernel 16: Greedy Top-1 Softmax + Argmax (`top_k == 1`, zero TG array)
// Writes 1 `DFlareTopKCandidate` + optional `out_tok_chain[next_tok_slot]`.
// Grid: (1, 1, 1), TG: (256, 1, 1)
// ---------------------------------------------------------------------------
kernel void dflare_top1_logits(const device uchar *logits_raw [[buffer(0)]],
                               const device uint *allowed_mask [[buffer(1)]],
                               device DFlareTopKCandidate *out_cands [[buffer(2)]],
                               device uint *out_tok_chain [[buffer(3)]],
                               constant DFlareTopKParams &p [[buffer(4)]],
                               uint tid [[thread_index_in_threadgroup]],
                               uint sg_id [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
    float best_v = -INFINITY;
    uint best_i = 0xFFFFFFFFu;
    float local_m = -INFINITY;
    float local_sum = 0.0f;

    const bool use_vec4 = (p.aligned_vec4 != 0u) && (p.has_mask == 0u);
    if (p.logits_is_f16 != 0u) {
        const device half *logits = (const device half *)logits_raw;
        if (use_vec4) {
            const uint vec_n = p.eff_vocab_size >> 2;
            const device half4 *vlogits = (const device half4 *)logits;
            for (uint vi = tid; vi < vec_n; vi += 256u) {
                const float4 fv = float4(vlogits[vi]);
                const uint base = vi << 2;
                #pragma clang loop unroll(full)
                for (uint c = 0u; c < 4u; c++) {
                    const float sc = dflare_eval_logit(fv[c], p.softcap, p.inv_temp);
                    const uint idx = base + c;
                    if (sc > local_m) {
                        local_sum = (local_sum > 0.0f) ? fma(local_sum, exp(local_m - sc), 1.0f) : 1.0f;
                        local_m = sc;
                    } else if (isfinite(sc)) {
                        local_sum += exp(sc - local_m);
                    }
                    if (sc > best_v || (sc == best_v && idx < best_i)) {
                        best_v = sc;
                        best_i = idx;
                    }
                }
            }
            for (uint i = (vec_n << 2) + tid; i < p.eff_vocab_size; i += 256u) {
                const float sc = dflare_eval_logit(float(logits[i]), p.softcap, p.inv_temp);
                if (sc > local_m) {
                    local_sum = (local_sum > 0.0f) ? fma(local_sum, exp(local_m - sc), 1.0f) : 1.0f;
                    local_m = sc;
                } else if (isfinite(sc)) {
                    local_sum += exp(sc - local_m);
                }
                if (sc > best_v || (sc == best_v && i < best_i)) {
                    best_v = sc;
                    best_i = i;
                }
            }
        } else {
            for (uint i = tid; i < p.eff_vocab_size; i += 256u) {
                if (p.has_mask != 0u && (((allowed_mask[i >> 5] >> (i & 31u)) & 1u) == 0u)) {
                    continue;
                }
                const float sc = dflare_eval_logit(float(logits[i]), p.softcap, p.inv_temp);
                if (sc > local_m) {
                    local_sum = (local_sum > 0.0f) ? fma(local_sum, exp(local_m - sc), 1.0f) : 1.0f;
                    local_m = sc;
                } else if (isfinite(sc)) {
                    local_sum += exp(sc - local_m);
                }
                if (sc > best_v || (sc == best_v && i < best_i)) {
                    best_v = sc;
                    best_i = i;
                }
            }
        }
    } else {
        const device float *logits = (const device float *)logits_raw;
        if (use_vec4) {
            const uint vec_n = p.eff_vocab_size >> 2;
            const device float4 *vlogits = (const device float4 *)logits;
            for (uint vi = tid; vi < vec_n; vi += 256u) {
                const float4 fv = vlogits[vi];
                const uint base = vi << 2;
                #pragma clang loop unroll(full)
                for (uint c = 0u; c < 4u; c++) {
                    const float sc = dflare_eval_logit(fv[c], p.softcap, p.inv_temp);
                    const uint idx = base + c;
                    if (sc > local_m) {
                        local_sum = (local_sum > 0.0f) ? fma(local_sum, exp(local_m - sc), 1.0f) : 1.0f;
                        local_m = sc;
                    } else if (isfinite(sc)) {
                        local_sum += exp(sc - local_m);
                    }
                    if (sc > best_v || (sc == best_v && idx < best_i)) {
                        best_v = sc;
                        best_i = idx;
                    }
                }
            }
            for (uint i = (vec_n << 2) + tid; i < p.eff_vocab_size; i += 256u) {
                const float sc = dflare_eval_logit(logits[i], p.softcap, p.inv_temp);
                if (sc > local_m) {
                    local_sum = (local_sum > 0.0f) ? fma(local_sum, exp(local_m - sc), 1.0f) : 1.0f;
                    local_m = sc;
                } else if (isfinite(sc)) {
                    local_sum += exp(sc - local_m);
                }
                if (sc > best_v || (sc == best_v && i < best_i)) {
                    best_v = sc;
                    best_i = i;
                }
            }
        } else {
            for (uint i = tid; i < p.eff_vocab_size; i += 256u) {
                if (p.has_mask != 0u && (((allowed_mask[i >> 5] >> (i & 31u)) & 1u) == 0u)) {
                    continue;
                }
                const float sc = dflare_eval_logit(logits[i], p.softcap, p.inv_temp);
                if (sc > local_m) {
                    local_sum = (local_sum > 0.0f) ? fma(local_sum, exp(local_m - sc), 1.0f) : 1.0f;
                    local_m = sc;
                } else if (isfinite(sc)) {
                    local_sum += exp(sc - local_m);
                }
                if (sc > best_v || (sc == best_v && i < best_i)) {
                    best_v = sc;
                    best_i = i;
                }
            }
        }
    }

    threadgroup float s_m[8];
    threadgroup float s_sum[8];
    threadgroup float s_bv[8];
    threadgroup uint s_bi[8];

    const float sg_m = simd_max(local_m);
    const float sg_sum = simd_sum((local_sum > 0.0f) ? (local_sum * exp(local_m - sg_m)) : 0.0f);
    for (ushort offset = 16u; offset > 0u; offset >>= 1u) {
        const float ov = simd_shuffle_down(best_v, offset);
        const uint oi = simd_shuffle_down(best_i, offset);
        if (ov > best_v || (ov == best_v && oi < best_i)) {
            best_v = ov;
            best_i = oi;
        }
    }
    if (lane == 0u) {
        s_m[sg_id] = sg_m;
        s_sum[sg_id] = sg_sum;
        s_bv[sg_id] = best_v;
        s_bi[sg_id] = best_i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float rm = -INFINITY;
        float fv = -INFINITY;
        uint fi = 0xFFFFFFFFu;
        for (uint s = 0u; s < 8u; s++) {
            if (s_sum[s] > 0.0f) {
                rm = max(rm, s_m[s]);
            }
            if (s_bv[s] > fv || (s_bv[s] == fv && s_bi[s] < fi)) {
                fv = s_bv[s];
                fi = s_bi[s];
            }
        }
        float rsum = 0.0f;
        for (uint s = 0u; s < 8u; s++) {
            if (s_sum[s] > 0.0f) {
                rsum += s_sum[s] * exp(s_m[s] - rm);
            }
        }
        DFlareTopKCandidate cand;
        if (fi != 0xFFFFFFFFu && rsum > 0.0f) {
            cand.token_id = fi;
            cand.prob = exp(fv - rm) / rsum;
            cand.log_prob = (fv - rm) - log(rsum);
            cand._pad = 0u;
        } else {
            cand.token_id = 0xFFFFFFFFu;
            cand.prob = 0.0f;
            cand.log_prob = -INFINITY;
            cand._pad = 0u;
        }
        out_cands[p.out_cand_offset] = cand;
        if (p.write_next_tok != 0u) {
            out_tok_chain[p.next_tok_slot] = (fi == 0xFFFFFFFFu) ? 0u : fi;
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 17: General Top-K Softmax (`1 <= top_k <= 8`, TG-backed zero-spill)
// Grid: (1, 1, 1), TG: (128, 1, 1)
// ---------------------------------------------------------------------------
static inline void dflare_topk_consider_tg(float v,
                                           uint idx,
                                           uint k,
                                           threadgroup float *tv,
                                           threadgroup uint *ti,
                                           thread float &min_v,
                                           thread uint &min_i,
                                           thread float &local_m,
                                           thread float &local_sum) {
    if (v > local_m) {
        local_sum = (local_sum > 0.0f) ? fma(local_sum, exp(local_m - v), 1.0f) : 1.0f;
        local_m = v;
    } else if (isfinite(v)) {
        local_sum += exp(v - local_m);
    }

    if (v > min_v || (v == min_v && idx < min_i)) {
        uint pos = k - 1u;
        while (pos > 0u) {
            const float pv = tv[pos - 1u];
            const uint pi = ti[pos - 1u];
            if (v > pv || (v == pv && idx < pi)) {
                tv[pos] = pv;
                ti[pos] = pi;
                pos--;
            } else {
                break;
            }
        }
        tv[pos] = v;
        ti[pos] = idx;
        min_v = tv[k - 1u];
        min_i = ti[k - 1u];
    }
}

kernel void dflare_topk_logits(const device uchar *logits_raw [[buffer(0)]],
                               const device uint *allowed_mask [[buffer(1)]],
                               device DFlareTopKCandidate *out_cands [[buffer(2)]],
                               device uint *out_tok_chain [[buffer(3)]],
                               constant DFlareTopKParams &p [[buffer(4)]],
                               uint tid [[thread_index_in_threadgroup]],
                               uint sg_id [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
    const uint k = clamp(p.top_k, 1u, 8u);
    threadgroup float tg_tv[128 * 8];
    threadgroup uint tg_ti[128 * 8];
    threadgroup float *my_tv = tg_tv + tid * 8u;
    threadgroup uint *my_ti = tg_ti + tid * 8u;

    for (uint j = 0u; j < k; j++) {
        my_tv[j] = -INFINITY;
        my_ti[j] = 0xFFFFFFFFu;
    }
    float min_v = -INFINITY;
    uint min_i = 0xFFFFFFFFu;
    float local_m = -INFINITY;
    float local_sum = 0.0f;

    const bool use_vec4 = (p.aligned_vec4 != 0u) && (p.has_mask == 0u);
    if (p.logits_is_f16 != 0u) {
        const device half *logits = (const device half *)logits_raw;
        if (use_vec4) {
            const uint vec_n = p.eff_vocab_size >> 2;
            const device half4 *vlogits = (const device half4 *)logits;
            for (uint vi = tid; vi < vec_n; vi += 128u) {
                const float4 fv = float4(vlogits[vi]);
                const uint base = vi << 2;
                #pragma clang loop unroll(full)
                for (uint c = 0u; c < 4u; c++) {
                    const float sc = dflare_eval_logit(fv[c], p.softcap, p.inv_temp);
                    dflare_topk_consider_tg(sc, base + c, k, my_tv, my_ti, min_v, min_i, local_m, local_sum);
                }
            }
            for (uint i = (vec_n << 2) + tid; i < p.eff_vocab_size; i += 128u) {
                const float sc = dflare_eval_logit(float(logits[i]), p.softcap, p.inv_temp);
                dflare_topk_consider_tg(sc, i, k, my_tv, my_ti, min_v, min_i, local_m, local_sum);
            }
        } else {
            for (uint i = tid; i < p.eff_vocab_size; i += 128u) {
                if (p.has_mask != 0u && (((allowed_mask[i >> 5] >> (i & 31u)) & 1u) == 0u)) {
                    continue;
                }
                const float sc = dflare_eval_logit(float(logits[i]), p.softcap, p.inv_temp);
                dflare_topk_consider_tg(sc, i, k, my_tv, my_ti, min_v, min_i, local_m, local_sum);
            }
        }
    } else {
        const device float *logits = (const device float *)logits_raw;
        if (use_vec4) {
            const uint vec_n = p.eff_vocab_size >> 2;
            const device float4 *vlogits = (const device float4 *)logits;
            for (uint vi = tid; vi < vec_n; vi += 128u) {
                const float4 fv = vlogits[vi];
                const uint base = vi << 2;
                #pragma clang loop unroll(full)
                for (uint c = 0u; c < 4u; c++) {
                    const float sc = dflare_eval_logit(fv[c], p.softcap, p.inv_temp);
                    dflare_topk_consider_tg(sc, base + c, k, my_tv, my_ti, min_v, min_i, local_m, local_sum);
                }
            }
            for (uint i = (vec_n << 2) + tid; i < p.eff_vocab_size; i += 128u) {
                const float sc = dflare_eval_logit(logits[i], p.softcap, p.inv_temp);
                dflare_topk_consider_tg(sc, i, k, my_tv, my_ti, min_v, min_i, local_m, local_sum);
            }
        } else {
            for (uint i = tid; i < p.eff_vocab_size; i += 128u) {
                if (p.has_mask != 0u && (((allowed_mask[i >> 5] >> (i & 31u)) & 1u) == 0u)) {
                    continue;
                }
                const float sc = dflare_eval_logit(logits[i], p.softcap, p.inv_temp);
                dflare_topk_consider_tg(sc, i, k, my_tv, my_ti, min_v, min_i, local_m, local_sum);
            }
        }
    }

    threadgroup float s_m[4];
    threadgroup float s_sum[4];
    threadgroup float s_vals[4];
    threadgroup uint s_idxs[4];
    threadgroup float s_row_max;
    threadgroup float s_row_sum;
    threadgroup float s_log_row_sum;
    threadgroup uint s_win_i;

    const float sg_m = simd_max(local_m);
    const float sg_sum = simd_sum((local_sum > 0.0f) ? (local_sum * exp(local_m - sg_m)) : 0.0f);
    if (lane == 0u) {
        s_m[sg_id] = sg_m;
        s_sum[sg_id] = sg_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float rm = -INFINITY;
        for (uint s = 0u; s < 4u; s++) {
            if (s_sum[s] > 0.0f) {
                rm = max(rm, s_m[s]);
            }
        }
        float rsum = 0.0f;
        for (uint s = 0u; s < 4u; s++) {
            if (s_sum[s] > 0.0f) {
                rsum += s_sum[s] * exp(s_m[s] - rm);
            }
        }
        s_row_max = rm;
        s_row_sum = rsum;
        s_log_row_sum = (rsum > 0.0f) ? log(rsum) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint cur = 0u;
    for (uint out_k = 0u; out_k < k; out_k++) {
        float cv = (cur < k) ? my_tv[cur] : -INFINITY;
        uint ci = (cur < k) ? my_ti[cur] : 0xFFFFFFFFu;
        for (ushort offset = 16u; offset > 0u; offset >>= 1u) {
            const float ov = simd_shuffle_down(cv, offset);
            const uint oi = simd_shuffle_down(ci, offset);
            if (ov > cv || (ov == cv && oi < ci)) {
                cv = ov;
                ci = oi;
            }
        }
        if (lane == 0u) {
            s_vals[sg_id] = cv;
            s_idxs[sg_id] = ci;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg_id == 0u) {
            float fv = (lane < 4u) ? s_vals[lane] : -INFINITY;
            uint fi = (lane < 4u) ? s_idxs[lane] : 0xFFFFFFFFu;
            for (ushort offset = 2u; offset > 0u; offset >>= 1u) {
                const float ov = simd_shuffle_down(fv, offset);
                const uint oi = simd_shuffle_down(fi, offset);
                if (ov > fv || (ov == fv && oi < fi)) {
                    fv = ov;
                    fi = oi;
                }
            }
            if (lane == 0u) {
                s_win_i = fi;
                DFlareTopKCandidate cand;
                if (fi != 0xFFFFFFFFu && s_row_sum > 0.0f) {
                    cand.token_id = fi;
                    cand.log_prob = (fv - s_row_max) - s_log_row_sum;
                    cand.prob = exp(fv - s_row_max) / s_row_sum;
                    cand._pad = 0u;
                } else {
                    cand.token_id = 0xFFFFFFFFu;
                    cand.prob = 0.0f;
                    cand.log_prob = -INFINITY;
                    cand._pad = 0u;
                }
                out_cands[p.out_cand_offset + out_k] = cand;
                if (out_k == 0u && p.write_next_tok != 0u) {
                    out_tok_chain[p.next_tok_slot] = (fi == 0xFFFFFFFFu) ? 0u : fi;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (cur < k && my_ti[cur] == s_win_i) {
            cur++;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ---------------------------------------------------------------------------
// Kernel 18: Pack Context Tap Ring (`f32`/`f16`/`bf16`) -> Contiguous `f16`
// `[num_pos, num_taps * hidden_size]` for `DFlash` `fc.weight [H, taps*H]`.
// Grid: (num_taps * hidden_size, num_pos, 1), TG: (256, 1, 1)
// ---------------------------------------------------------------------------
kernel void dflare_pack_ctx_taps_f16(const device uchar *taps_raw [[buffer(0)]],
                                     device half *out_f16 [[buffer(1)]],
                                     constant DFlarePackCtxTapsParams &p [[buffer(2)]],
                                     uint2 gid [[thread_position_in_grid]]) {
    const uint col = gid.x;
    const uint pos = gid.y;
    const uint row_cols = p.num_taps * p.hidden_size;
    if (col >= row_cols || pos >= p.num_pos) {
        return;
    }
    const uint src_slot = (p.ring_cap > 0u) ? ((p.ring_start + pos) % p.ring_cap) : pos;
    const ulong src_idx = ulong(src_slot) * row_cols + col;
    const float v = dflare_load_tap_elem(taps_raw, src_idx, p.taps_dtype);
    out_f16[ulong(pos) * row_cols + col] = half(v);
}

// Helper for loading f16 (0), bf16 (1), or f32 (2) 1D norm/sink weights.
static inline float mtp_load_weight_elem(const device uchar *base, ulong idx, uint dtype) {
    if (dtype == 0u) {
        return float(((const device half *)base)[idx]);
    }
    if (dtype == 1u) {
        return float(((const device bfloat *)base)[idx]);
    }
    return ((const device float *)base)[idx];
}

// ---------------------------------------------------------------------------
// Kernel 19 (`D15` Native 3L MTP): Per-Head Optional Q/K RMSNorm + Partial
// NeoX RoPE (`rope_dim <= head_dim`) + `v_scale` + Tree K/V Store.
// Grid: (num_q_heads + num_kv_heads, num_nodes, 1), TG: (128, 1, 1)
// ---------------------------------------------------------------------------
kernel void mtp_tree_qk_norm_rope_vscale(device half *q_f16 [[buffer(0)]],
                                         const device half *k_f16 [[buffer(1)]],
                                         const device half *v_f16 [[buffer(2)]],
                                         const device uchar *qn_raw [[buffer(3)]],
                                         const device uchar *kn_raw [[buffer(4)]],
                                         const device uint *positions_u32 [[buffer(5)]],
                                         device half *k_tree_out [[buffer(6)]],
                                         device half *v_tree_out [[buffer(7)]],
                                         constant MtpTreeQkNormRopeParams &p [[buffer(8)]],
                                         uint2 tg_pos [[threadgroup_position_in_grid]],
                                         uint tid [[thread_index_in_threadgroup]],
                                         uint sg_id [[simdgroup_index_in_threadgroup]],
                                         uint lane [[thread_index_in_simdgroup]]) {
    const uint head_global = tg_pos.x;
    const uint node_idx = tg_pos.y;
    const uint total_heads = p.num_q_heads + p.num_kv_heads;
    if (head_global >= total_heads || node_idx >= p.num_nodes || p.head_dim > 256u || p.v_head_dim > 256u) {
        return;
    }

    threadgroup float tg_h[256];
    threadgroup float sg_sums[4];

    const bool is_q = (head_global < p.num_q_heads);
    const uint h_idx = is_q ? head_global : (head_global - p.num_q_heads);
    const uint tok_pos = positions_u32[node_idx];

    if (!is_q) {
        const uint kv_v_dim = p.num_kv_heads * p.v_head_dim;
        const ulong v_off = ulong(node_idx) * kv_v_dim + ulong(h_idx) * p.v_head_dim;
        for (uint d = tid; d < p.v_head_dim; d += 128u) {
            v_tree_out[v_off + d] = half(float(v_f16[v_off + d]) * p.v_scale);
        }
    }

    const uint kq_stride = (is_q ? p.num_q_heads : p.num_kv_heads) * p.head_dim;
    const ulong off = ulong(node_idx) * kq_stride + ulong(h_idx) * p.head_dim;
    const device half *src_ptr = (is_q ? (const device half *)q_f16 : k_f16) + off;
    device half *dst_ptr = (is_q ? q_f16 : k_tree_out) + off;
    const device uchar *norm_raw = is_q ? qn_raw : kn_raw;
    const uint apply_norm = is_q ? p.has_q_norm : p.has_k_norm;

    float sum_sq = 0.0f;
    for (uint d = tid; d < p.head_dim; d += 128u) {
        const float v = float(src_ptr[d]);
        tg_h[d] = v;
        sum_sq = fma(v, v, sum_sq);
    }

    if (apply_norm != 0u) {
        const float sg_tot = simd_sum(sum_sq);
        if (lane == 0u) {
            sg_sums[sg_id] = sg_tot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float tot = 0.0f;
            for (uint s = 0u; s < 4u; s++) {
                tot += sg_sums[s];
            }
            sg_sums[0] = rsqrt(tot / float(p.head_dim) + p.rms_eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float inv_rms = sg_sums[0];
        for (uint d = tid; d < p.head_dim; d += 128u) {
            tg_h[d] = tg_h[d] * inv_rms * mtp_load_weight_elem(norm_raw, d, p.norm_dtype);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint eff_rope = (p.rope_dim > 0u && p.rope_dim <= p.head_dim) ? p.rope_dim : p.head_dim;
    if (p.rope_theta > 0.0f && eff_rope >= 2u) {
        const uint half_rot = eff_rope >> 1;
        const float inv_rot = 2.0f / float(eff_rope);
        const float pos_f = float(tok_pos);
        for (uint i = tid; i < half_rot; i += 128u) {
            const float freq = 1.0f / pow(p.rope_theta, float(i) * inv_rot);
            const float angle = pos_f * freq;
            float c_val;
            const float s_val = sincos(angle, c_val);
            const float x0 = tg_h[i];
            const float x1 = tg_h[i + half_rot];
            dst_ptr[i] = half(x0 * c_val - x1 * s_val);
            dst_ptr[i + half_rot] = half(x0 * s_val + x1 * c_val);
        }
        for (uint d = eff_rope + tid; d < p.head_dim; d += 128u) {
            dst_ptr[d] = half(tg_h[d]);
        }
    } else {
        for (uint d = tid; d < p.head_dim; d += 128u) {
            dst_ptr[d] = half(tg_h[d]);
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 20 (`D15` Native 3L MTP): Commit Accepted K/V Rows into Circular Ring
// Grid: (num_kv_heads, num_nodes, 1), TG: (128, 1, 1)
// ---------------------------------------------------------------------------
kernel void mtp_kv_ring_commit(const device half *k_tree [[buffer(0)]],
                               const device half *v_tree [[buffer(1)]],
                               device half *kv_ring_k [[buffer(2)]],
                               device half *kv_ring_v [[buffer(3)]],
                               constant MtpKvCommitParams &p [[buffer(4)]],
                               uint2 tg_pos [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]]) {
    const uint kvh = tg_pos.x;
    const uint idx = tg_pos.y;
    if (kvh >= p.num_kv_heads || idx >= p.num_nodes || p.kv_cap == 0u) {
        return;
    }
    const uint slot = (p.base_pos + idx) % p.kv_cap;
    const uint kv_k_dim = p.num_kv_heads * p.head_dim;
    const uint kv_v_dim = p.num_kv_heads * p.v_head_dim;

    const ulong src_k = ulong(idx) * kv_k_dim + ulong(kvh) * p.head_dim;
    const ulong dst_k = ulong(p.layer_k_elem_off) + ulong(slot) * kv_k_dim + ulong(kvh) * p.head_dim;
    for (uint d = tid; d < p.head_dim; d += 128u) {
        kv_ring_k[dst_k + d] = k_tree[src_k + d];
    }

    const ulong src_v = ulong(idx) * kv_v_dim + ulong(kvh) * p.v_head_dim;
    const ulong dst_v = ulong(p.layer_v_elem_off) + ulong(slot) * kv_v_dim + ulong(kvh) * p.v_head_dim;
    for (uint d = tid; d < p.v_head_dim; d += 128u) {
        kv_ring_v[dst_v + d] = v_tree[src_v + d];
    }
}

// ---------------------------------------------------------------------------
// Kernel 21 (`D15` Native 3L MTP): Asymmetric `head_dim/v_head_dim` GQA Tree
// Attention over Circular SWA/Causal KV Ring + Tree-Local KV + `sink_bias`.
// Grid: (num_q_heads, q_count, 1), TG: (128, 1, 1)
// ---------------------------------------------------------------------------
kernel void mtp_gqa_tree_verify_attn(const device half *q_f16 [[buffer(0)]],
                                     const device half *kv_ring_k [[buffer(1)]],
                                     const device half *kv_ring_v [[buffer(2)]],
                                     const device half *k_tree [[buffer(3)]],
                                     const device half *v_tree [[buffer(4)]],
                                     const device uint *positions_u32 [[buffer(5)]],
                                     const device ulong *ancestor_masks_u64 [[buffer(6)]],
                                     const device uchar *sink_bias_raw [[buffer(7)]],
                                     device half *out_f16 [[buffer(8)]],
                                     const device uint *sparse_indices_u32 [[buffer(9)]],
                                     const device half *gate_f16 [[buffer(10)]],
                                     constant MtpGqaTreeAttnParams &p [[buffer(11)]],
                                     uint2 tg_pos [[threadgroup_position_in_grid]],
                                     uint tid [[thread_index_in_threadgroup]],
                                     uint sg_id [[simdgroup_index_in_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]]) {
    const uint qh = tg_pos.x;
    const uint q_idx = tg_pos.y;
    if (qh >= p.num_q_heads || q_idx >= p.q_count || p.head_dim > 256u || p.v_head_dim > 256u) {
        return;
    }

    const uint q_node = p.slot_start + q_idx;
    if (q_node >= p.num_nodes) {
        return;
    }

    const uint gqa_group = p.num_q_heads / p.num_kv_heads;
    const uint kvh = qh / gqa_group;
    const uint q_dim = p.num_q_heads * p.head_dim;
    const uint o_dim = p.num_q_heads * p.v_head_dim;
    const uint kv_k_dim = p.num_kv_heads * p.head_dim;
    const uint kv_v_dim = p.num_kv_heads * p.v_head_dim;
    const ulong q_off = ulong(q_idx) * q_dim + ulong(qh) * p.head_dim;
    const ulong out_off = ulong(q_idx) * o_dim + ulong(qh) * p.v_head_dim;

    const uint q_pos = positions_u32[q_node];
    const ulong q_mask = ancestor_masks_u64[q_node];

    threadgroup float tg_q[256];
    threadgroup float tg_acc[4][256];
    threadgroup float tg_m[5];
    threadgroup float tg_l[5];

    for (uint d = tid; d < p.head_dim; d += 128u) {
        tg_q[d] = float(q_f16[q_off + d]);
    }
    for (uint d = tid; d < p.v_head_dim; d += 128u) {
        tg_acc[0][d] = 0.0f;
        tg_acc[1][d] = 0.0f;
        tg_acc[2][d] = 0.0f;
        tg_acc[3][d] = 0.0f;
    }
    if (tid < 5u) {
        tg_m[tid] = -INFINITY;
        tg_l[tid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_sg = -INFINITY;
    float l_sg = 0.0f;

    // 1. Committed circular KV ring (`[eff_start .. base_pos)` under causal SWA window,
    //    or zero-reindex shared sparse indices when `has_sparse_indices != 0`).
    if (p.base_pos > 0u && p.kv_cap > 0u) {
        const uint ctx_base = (p.base_pos > p.kv_cap) ? (p.base_pos - p.kv_cap) : 0u;
        if (p.has_sparse_indices != 0u && p.sparse_k > 0u) {
            const uint sp_row = (p.sparse_num_rows > 1u) ? min(q_idx, p.sparse_num_rows - 1u) : 0u;
            const ulong sp_off = ulong(sp_row) * p.sparse_row_stride;
            for (uint si = sg_id; si < p.sparse_k; si += 4u) {
                const uint k_pos = sparse_indices_u32[sp_off + si];
                if (k_pos >= p.base_pos || k_pos < ctx_base || k_pos > q_pos) {
                    continue;
                }
                const uint ring_slot = k_pos % p.kv_cap;
                const device half *k_head = kv_ring_k + ulong(p.layer_k_elem_off) + ulong(ring_slot) * kv_k_dim + ulong(kvh) * p.head_dim;
                const device half *v_head = kv_ring_v + ulong(p.layer_v_elem_off) + ulong(ring_slot) * kv_v_dim + ulong(kvh) * p.v_head_dim;

                float dot_partial = 0.0f;
                for (uint d = lane; d < p.head_dim; d += 32u) {
                    dot_partial = fma(tg_q[d], float(k_head[d]), dot_partial);
                }
                const float score = simd_sum(dot_partial) * p.scale;
                const float m_new = max(m_sg, score);
                const float alpha = (l_sg > 0.0f) ? exp(m_sg - m_new) : 0.0f;
                const float prob = exp(score - m_new);
                l_sg = fma(l_sg, alpha, prob);
                m_sg = m_new;

                for (uint d = lane; d < p.v_head_dim; d += 32u) {
                    tg_acc[sg_id][d] = fma(prob, float(v_head[d]), tg_acc[sg_id][d] * alpha);
                }
            }
        } else {
            uint eff_start = ctx_base;
            if (p.sliding_window > 0u) {
                const uint min_k = (q_pos + 1u > p.sliding_window) ? (q_pos + 1u - p.sliding_window) : 0u;
                eff_start = max(ctx_base, min_k);
            }
            const uint eff_ctx = (p.base_pos > eff_start) ? (p.base_pos - eff_start) : 0u;
            for (uint ci = sg_id; ci < eff_ctx; ci += 4u) {
                const uint k_pos = eff_start + ci;
                if (k_pos > q_pos) {
                    continue;
                }
                const uint ring_slot = k_pos % p.kv_cap;
                const device half *k_head = kv_ring_k + ulong(p.layer_k_elem_off) + ulong(ring_slot) * kv_k_dim + ulong(kvh) * p.head_dim;
                const device half *v_head = kv_ring_v + ulong(p.layer_v_elem_off) + ulong(ring_slot) * kv_v_dim + ulong(kvh) * p.v_head_dim;

                float dot_partial = 0.0f;
                for (uint d = lane; d < p.head_dim; d += 32u) {
                    dot_partial = fma(tg_q[d], float(k_head[d]), dot_partial);
                }
                const float score = simd_sum(dot_partial) * p.scale;
                const float m_new = max(m_sg, score);
                const float alpha = (l_sg > 0.0f) ? exp(m_sg - m_new) : 0.0f;
                const float prob = exp(score - m_new);
                l_sg = fma(l_sg, alpha, prob);
                m_sg = m_new;

                for (uint d = lane; d < p.v_head_dim; d += 32u) {
                    tg_acc[sg_id][d] = fma(prob, float(v_head[d]), tg_acc[sg_id][d] * alpha);
                }
            }
        }
    }

    // 2. Tree-local KV slots masked by `ancestor_masks_u64[q_node]` and SWA window.
    for (uint ti = sg_id; ti < p.num_nodes; ti += 4u) {
        if (((q_mask >> ti) & 1ull) == 0ull) {
            continue;
        }
        const uint tk_pos = positions_u32[ti];
        if (tk_pos > q_pos) {
            continue;
        }
        if (p.sliding_window > 0u && (q_pos - tk_pos) >= p.sliding_window) {
            continue;
        }
        const device half *k_head = k_tree + ulong(ti) * kv_k_dim + ulong(kvh) * p.head_dim;
        const device half *v_head = v_tree + ulong(ti) * kv_v_dim + ulong(kvh) * p.v_head_dim;

        float dot_partial = 0.0f;
        for (uint d = lane; d < p.head_dim; d += 32u) {
            dot_partial = fma(tg_q[d], float(k_head[d]), dot_partial);
        }
        const float score = simd_sum(dot_partial) * p.scale;
        const float m_new = max(m_sg, score);
        const float alpha = (l_sg > 0.0f) ? exp(m_sg - m_new) : 0.0f;
        const float prob = exp(score - m_new);
        l_sg = fma(l_sg, alpha, prob);
        m_sg = m_new;

        for (uint d = lane; d < p.v_head_dim; d += 32u) {
            tg_acc[sg_id][d] = fma(prob, float(v_head[d]), tg_acc[sg_id][d] * alpha);
        }
    }

    if (lane == 0u) {
        tg_m[sg_id] = m_sg;
        tg_l[sg_id] = l_sg;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float gm = -INFINITY;
        for (uint s = 0u; s < 4u; s++) {
            if (tg_l[s] > 0.0f) {
                gm = max(gm, tg_m[s]);
            }
        }
        float sink_val = -INFINITY;
        if (p.has_sink_bias != 0u) {
            sink_val = mtp_load_weight_elem(sink_bias_raw, qh, p.sink_dtype);
            gm = max(gm, sink_val);
        }
        float gl = 0.0f;
        for (uint s = 0u; s < 4u; s++) {
            if (tg_l[s] > 0.0f) {
                gl += tg_l[s] * exp(tg_m[s] - gm);
            }
        }
        if (p.has_sink_bias != 0u) {
            gl += exp(sink_val - gm);
        }
        tg_m[4] = gm;
        tg_l[4] = (gl > 0.0f) ? (1.0f / gl) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float gm = tg_m[4];
    const float inv_gl = tg_l[4];
    const float head_gate = (p.head_gate_mode == 1u)
        ? dflare_sigmoid(float(gate_f16[ulong(q_idx) * p.num_q_heads + qh]))
        : 1.0f;
    for (uint d = tid; d < p.v_head_dim; d += 128u) {
        float acc = 0.0f;
        for (uint s = 0u; s < 4u; s++) {
            if (isfinite(tg_m[s])) {
                acc = fma(tg_acc[s][d], exp(tg_m[s] - gm), acc);
            }
        }
        float out_v = acc * inv_gl;
        if (p.head_gate_mode == 1u) {
            out_v *= head_gate;
        } else if (p.head_gate_mode == 2u) {
            out_v *= dflare_sigmoid(float(gate_f16[out_off + d]));
        }
        out_f16[out_off + d] = half(out_v);
    }
}

// ---------------------------------------------------------------------------
// Kernel 22 (`D15` GLM-5.2/5.3 MLA & GLM-5.3-Flash NoPE-MLA 1L MTP):
// Applies optional interleaved/NeoX RoPE to `q_rot` & shared `k_rot`, splits
// `kv_b_proj` output into `(k_nope, v)`, and writes expanded `(k_tree, v_tree)`.
// Grid: (num_heads, num_nodes, 1), TG: (128, 1, 1)
// ---------------------------------------------------------------------------
kernel void mtp_mla_prep_rope_kv(device half *q_f16 [[buffer(0)]],
                                 const device half *kv_a_f16 [[buffer(1)]],
                                 const device half *kv_b_f16 [[buffer(2)]],
                                 const device uint *positions_u32 [[buffer(3)]],
                                 device half *k_tree_out [[buffer(4)]],
                                 device half *v_tree_out [[buffer(5)]],
                                 constant MtpMlaPrepParams &p [[buffer(6)]],
                                 uint2 tg_pos [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]]) {
    const uint head = tg_pos.x;
    const uint node = tg_pos.y;
    if (head >= p.num_heads || node >= p.num_nodes || p.qk_rope_head_dim > 128u) {
        return;
    }

    const uint qk_head = p.qk_nope_head_dim + p.qk_rope_head_dim;
    const uint kv_b_head = p.qk_nope_head_dim + p.v_head_dim;
    const ulong qk_base = (ulong(node) * p.num_heads + head) * qk_head;
    const ulong kv_b_base = (ulong(node) * p.num_heads + head) * kv_b_head;
    const ulong v_base = (ulong(node) * p.num_heads + head) * p.v_head_dim;

    for (uint d = tid; d < p.qk_nope_head_dim; d += 128u) {
        k_tree_out[qk_base + d] = kv_b_f16[kv_b_base + d];
    }
    for (uint d = tid; d < p.v_head_dim; d += 128u) {
        v_tree_out[v_base + d] = kv_b_f16[kv_b_base + p.qk_nope_head_dim + d];
    }

    if (p.qk_rope_head_dim >= 2u) {
        threadgroup float tg_qrot[128];
        threadgroup float tg_krot[128];
        const ulong kv_a_stride = ulong(p.kv_lora_rank + p.qk_rope_head_dim);
        const ulong krot_src = ulong(node) * kv_a_stride + p.kv_lora_rank;
        for (uint d = tid; d < p.qk_rope_head_dim; d += 128u) {
            tg_qrot[d] = float(q_f16[qk_base + p.qk_nope_head_dim + d]);
            tg_krot[d] = float(kv_a_f16[krot_src + d]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (p.rope_theta > 0.0f) {
            const uint half_rot = p.qk_rope_head_dim >> 1;
            const float inv_rot = 2.0f / float(p.qk_rope_head_dim);
            const float pos_f = float(positions_u32[node]);
            for (uint i = tid; i < half_rot; i += 128u) {
                const float freq = 1.0f / pow(p.rope_theta, float(i) * inv_rot);
                const float angle = pos_f * freq;
                float c_val;
                const float s_val = sincos(angle, c_val);
                float q0, q1, k0, k1;
                if (p.rope_interleave != 0u) {
                    q0 = tg_qrot[2u * i];
                    q1 = tg_qrot[2u * i + 1u];
                    k0 = tg_krot[2u * i];
                    k1 = tg_krot[2u * i + 1u];
                } else {
                    q0 = tg_qrot[i];
                    q1 = tg_qrot[i + half_rot];
                    k0 = tg_krot[i];
                    k1 = tg_krot[i + half_rot];
                }
                q_f16[qk_base + p.qk_nope_head_dim + i] = half(q0 * c_val - q1 * s_val);
                q_f16[qk_base + p.qk_nope_head_dim + i + half_rot] = half(q1 * c_val + q0 * s_val);
                k_tree_out[qk_base + p.qk_nope_head_dim + i] = half(k0 * c_val - k1 * s_val);
                k_tree_out[qk_base + p.qk_nope_head_dim + i + half_rot] = half(k1 * c_val + k0 * s_val);
            }
        } else {
            for (uint d = tid; d < p.qk_rope_head_dim; d += 128u) {
                k_tree_out[qk_base + p.qk_nope_head_dim + d] = half(tg_krot[d]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 23 (`D15` GLM-5.3-Flash `swiglu_limit = 10.0`):
// `out[i] = silu(min(gate[i], limit)) * clamp(up[i], -limit, limit)`
// Grid: (count, 1, 1), TG: (<=256, 1, 1)
// ---------------------------------------------------------------------------
kernel void mtp_swiglu_clamped_mul(const device half *gate_f16 [[buffer(0)]],
                                   const device half *up_f16 [[buffer(1)]],
                                   device half *out_f16 [[buffer(2)]],
                                   constant MtpSwigluClampParams &p [[buffer(3)]],
                                   uint gid [[thread_position_in_grid]]) {
    if (gid >= p.count) {
        return;
    }
    float g = float(gate_f16[gid]);
    float u = float(up_f16[gid]);
    if (p.limit > 0.0f) {
        g = min(g, p.limit);
        u = clamp(u, -p.limit, p.limit);
    }
    out_f16[gid] = half(dflare_silu(g) * u);
}

// ---------------------------------------------------------------------------
// Kernel 24 (`D15` 1L MTP MoE Router): Sigmoid/Softmax + `e_score_correction_bias`
// + Top-K + optional `norm_topk_prob` + `routed_scaling_factor`.
// Grid: (m, 1, 1), TG: (128, 1, 1)
// ---------------------------------------------------------------------------
kernel void mtp_moe_route_topk_sigmoid(const device half *router_logits_f16 [[buffer(0)]],
                                       const device uchar *bias_raw [[buffer(1)]],
                                       device uint *out_expert_ids [[buffer(2)]],
                                       device float *out_expert_weights [[buffer(3)]],
                                       constant MtpMoeRouteParams &p [[buffer(4)]],
                                       uint tg_pos [[threadgroup_position_in_grid]],
                                       uint tid [[thread_index_in_threadgroup]],
                                       uint sg_id [[simdgroup_index_in_threadgroup]],
                                       uint lane [[thread_index_in_simdgroup]]) {
    const uint tok = tg_pos;
    if (tok >= p.m || p.num_experts == 0u || p.num_experts > 512u || p.top_k == 0u || p.top_k > 16u) {
        return;
    }

    threadgroup float tg_raw_w[512];
    threadgroup float tg_sel[512];
    threadgroup float tg_sg_m[4];
    threadgroup float tg_sg_sum[4];
    threadgroup float tg_sel_v[4];
    threadgroup uint tg_sel_i[4];
    threadgroup float tg_gscore[32];
    threadgroup uint tg_group_mask[1];

    const ulong row_off = ulong(tok) * p.num_experts;
    if (p.use_sigmoid != 0u) {
        for (uint e = tid; e < p.num_experts; e += 128u) {
            const float rw = dflare_sigmoid(float(router_logits_f16[row_off + e]));
            tg_raw_w[e] = rw;
            const float b = (p.has_bias != 0u) ? mtp_load_weight_elem(bias_raw, e, p.bias_dtype) : 0.0f;
            tg_sel[e] = rw + b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    } else {
        float l_max = -INFINITY;
        for (uint e = tid; e < p.num_experts; e += 128u) {
            const float lv = float(router_logits_f16[row_off + e]);
            tg_raw_w[e] = lv;
            l_max = max(l_max, lv);
        }
        const float sm = simd_max(l_max);
        if (lane == 0u) tg_sg_m[sg_id] = sm;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            tg_sg_m[0] = max(max(tg_sg_m[0], tg_sg_m[1]), max(tg_sg_m[2], tg_sg_m[3]));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float row_max = tg_sg_m[0];
        float l_sum = 0.0f;
        for (uint e = tid; e < p.num_experts; e += 128u) {
            const float ev = exp(tg_raw_w[e] - row_max);
            tg_raw_w[e] = ev;
            l_sum += ev;
        }
        const float ss = simd_sum(l_sum);
        if (lane == 0u) tg_sg_sum[sg_id] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            const float tot = tg_sg_sum[0] + tg_sg_sum[1] + tg_sg_sum[2] + tg_sg_sum[3];
            tg_sg_sum[0] = (tot > 0.0f) ? (1.0f / tot) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float inv_s = tg_sg_sum[0];
        for (uint e = tid; e < p.num_experts; e += 128u) {
            const float rw = tg_raw_w[e] * inv_s;
            tg_raw_w[e] = rw;
            const float b = (p.has_bias != 0u) ? mtp_load_weight_elem(bias_raw, e, p.bias_dtype) : 0.0f;
            tg_sel[e] = rw + b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Optional `BailingMoeV3` / DeepSeek `group_limited_topk` (`n_group=8, topk_group=4`, top-2 sum per group).
    const bool use_groups = (p.n_group > 1u && p.topk_group > 0u && p.topk_group < p.n_group &&
                             p.n_group <= 32u && (p.num_experts % p.n_group) == 0u);
    if (use_groups) {
        const uint group_size = p.num_experts / p.n_group;
        if (tid < p.n_group) {
            const uint g_start = tid * group_size;
            float top1 = -INFINITY;
            float top2 = -INFINITY;
            for (uint off = 0u; off < group_size; off++) {
                const float v = tg_sel[g_start + off];
                if (v >= top1) {
                    top2 = top1;
                    top1 = v;
                } else if (v > top2) {
                    top2 = v;
                }
            }
            tg_gscore[tid] = top1 + ((group_size >= 2u && isfinite(top2)) ? top2 : 0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            uint mask = 0u;
            for (uint gk = 0u; gk < p.topk_group; gk++) {
                float best_v = -INFINITY;
                uint best_g = 0xFFFFFFFFu;
                for (uint g = 0u; g < p.n_group; g++) {
                    if (((mask >> g) & 1u) != 0u) {
                        continue;
                    }
                    const float gs = tg_gscore[g];
                    if (gs > best_v || (gs == best_v && g < best_g)) {
                        best_v = gs;
                        best_g = g;
                    }
                }
                if (best_g < p.n_group) {
                    mask |= (1u << best_g);
                }
            }
            tg_group_mask[0] = mask;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint allow_mask = tg_group_mask[0];
        for (uint e = tid; e < p.num_experts; e += 128u) {
            const uint g = e / group_size;
            if (((allow_mask >> g) & 1u) == 0u) {
                tg_sel[e] = -INFINITY;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint eff_k = min(p.top_k, p.num_experts);
    const ulong out_base = ulong(tok) * p.top_k;
    for (uint k = 0u; k < eff_k; k++) {
        float best_v = -INFINITY;
        uint best_i = 0xFFFFFFFFu;
        for (uint e = tid; e < p.num_experts; e += 128u) {
            const float v = tg_sel[e];
            if (v > best_v || (v == best_v && e < best_i)) {
                best_v = v;
                best_i = e;
            }
        }
        for (ushort off = 16u; off > 0u; off >>= 1u) {
            const float ov = simd_shuffle_down(best_v, off);
            const uint oi = simd_shuffle_down(best_i, off);
            if (ov > best_v || (ov == best_v && oi < best_i)) {
                best_v = ov;
                best_i = oi;
            }
        }
        if (lane == 0u) {
            tg_sel_v[sg_id] = best_v;
            tg_sel_i[sg_id] = best_i;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float gv = tg_sel_v[0];
            uint gi = tg_sel_i[0];
            for (uint s = 1u; s < 4u; s++) {
                if (tg_sel_v[s] > gv || (tg_sel_v[s] == gv && tg_sel_i[s] < gi)) {
                    gv = tg_sel_v[s];
                    gi = tg_sel_i[s];
                }
            }
            out_expert_ids[out_base + k] = gi;
            out_expert_weights[out_base + k] = tg_raw_w[gi];
            tg_sel[gi] = -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        float w_sum = 0.0f;
        for (uint k = 0u; k < eff_k; k++) {
            w_sum += out_expert_weights[out_base + k];
        }
        const float denom = (p.norm_topk_prob != 0u) ? (w_sum + 1e-20f) : 1.0f;
        const float mul = p.routed_scaling_factor / denom;
        for (uint k = 0u; k < eff_k; k++) {
            out_expert_weights[out_base + k] *= mul;
        }
    }
}

static inline float mtp_moe_dot_row(const device half *x_row,
                                    const device uchar *weights_base,
                                    ulong w_off,
                                    ulong sb_off,
                                    uint row,
                                    uint k_dim,
                                    uint weight_dtype,
                                    uint group_size,
                                    uint lane) {
    float acc = 0.0f;
    if (weight_dtype == 0u) {
        const device half *w = (const device half *)(weights_base + w_off) + ulong(row) * k_dim;
        for (uint c = lane; c < k_dim; c += 32u) {
            acc = fma(float(x_row[c]), float(w[c]), acc);
        }
    } else if (weight_dtype == 1u) {
        const device bfloat *w = (const device bfloat *)(weights_base + w_off) + ulong(row) * k_dim;
        for (uint c = lane; c < k_dim; c += 32u) {
            acc = fma(float(x_row[c]), float(w[c]), acc);
        }
    } else {
        const uint gs = (group_size > 0u) ? group_size : 64u;
        const uint num_groups = k_dim / gs;
        const device half2 *sb_ptr = (const device half2 *)(weights_base + sb_off) + ulong(row) * num_groups;
        if (weight_dtype == 2u) {
            const ulong row_bytes = ulong(k_dim) >> 1;
            const device uchar *w = weights_base + w_off + ulong(row) * row_bytes;
            for (uint c = lane; c < k_dim; c += 32u) {
                const uint g = c / gs;
                const half2 sb = sb_ptr[g];
                const uchar b = w[c >> 1];
                const uint q = (b >> ((c & 1u) * 4u)) & 0xFu;
                const float wv = fma(float(q), float(sb.x), float(sb.y));
                acc = fma(float(x_row[c]), wv, acc);
            }
        } else if (weight_dtype == 3u) {
            const ulong row_bytes = (ulong(k_dim) * 3ul) >> 3;
            const device uchar *w = weights_base + w_off + ulong(row) * row_bytes;
            for (uint c = lane; c < k_dim; c += 32u) {
                const uint g = c / gs;
                const half2 sb = sb_ptr[g];
                const uint blk = c >> 3;
                const uint sub = c & 7u;
                const device uchar *bp = w + ulong(blk) * 3ul;
                const uint bits24 = uint(bp[0]) | (uint(bp[1]) << 8u) | (uint(bp[2]) << 16u);
                const uint q = (bits24 >> (sub * 3u)) & 0x7u;
                const float wv = fma(float(q), float(sb.x), float(sb.y));
                acc = fma(float(x_row[c]), wv, acc);
            }
        } else {
            const ulong row_bytes = ulong(k_dim) >> 2;
            const device uchar *w = weights_base + w_off + ulong(row) * row_bytes;
            for (uint c = lane; c < k_dim; c += 32u) {
                const uint g = c / gs;
                const half2 sb = sb_ptr[g];
                const uchar b = w[c >> 2];
                const uint q = (b >> ((c & 3u) * 2u)) & 0x3u;
                const float wv = fma(float(q), float(sb.x), float(sb.y));
                acc = fma(float(x_row[c]), wv, acc);
            }
        }
    }
    return simd_sum(acc);
}

// ---------------------------------------------------------------------------
// Kernel 25 (`D15` 1L MTP MoE Gate+Up+SwiGLU with optional `swiglu_limit`):
// Grid: (ceil(moe_inter / 4), top_k, m), TG: (128, 1, 1)
// ---------------------------------------------------------------------------
kernel void mtp_moe_gate_up_swiglu(const device half *x_norm_f16 [[buffer(0)]],
                                   const device uchar *weights_base [[buffer(1)]],
                                   const device MtpExpertMeta *expert_meta [[buffer(2)]],
                                   const device uint *expert_ids [[buffer(3)]],
                                   device half *act_out_f16 [[buffer(4)]],
                                   constant MtpMoeExpertParams &p [[buffer(5)]],
                                   uint3 tg_pos [[threadgroup_position_in_grid]],
                                   uint sg_id [[simdgroup_index_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]]) {
    const uint row = tg_pos.x * 4u + sg_id;
    const uint k_slot = tg_pos.y;
    const uint tok = tg_pos.z;
    if (row >= p.moe_inter || k_slot >= p.top_k || tok >= p.m) {
        return;
    }

    const uint eid = expert_ids[ulong(tok) * p.top_k + k_slot];
    const MtpExpertMeta meta = expert_meta[eid];
    if (meta.gate_off == 0xFFFFFFFFFFFFFFFFull) {
        if (lane == 0u) {
            act_out_f16[(ulong(tok) * p.top_k + k_slot) * p.moe_inter + row] = half(0.0f);
        }
        return;
    }
    const device half *x_row = x_norm_f16 + ulong(tok) * p.hidden_size;
    const uint exp_dt = meta.weight_dtype;
    const uint exp_gs = (meta.group_size > 0u) ? meta.group_size : p.group_size;

    float g = mtp_moe_dot_row(x_row, weights_base, meta.gate_off, meta.gate_sb_off, row, p.hidden_size, exp_dt, exp_gs, lane);
    float u = mtp_moe_dot_row(x_row, weights_base, meta.up_off, meta.up_sb_off, row, p.hidden_size, exp_dt, exp_gs, lane);
    if (lane == 0u) {
        if (p.swiglu_limit > 0.0f) {
            g = min(g, p.swiglu_limit);
            u = clamp(u, -p.swiglu_limit, p.swiglu_limit);
        }
        act_out_f16[(ulong(tok) * p.top_k + k_slot) * p.moe_inter + row] = half(dflare_silu(g) * u);
    }
}

// ---------------------------------------------------------------------------
// Kernel 26 (`D15` 1L MTP MoE Down + Weighted Reduce + Residual Add):
// Grid: (ceil(hidden_size / 4), m, 1), TG: (128, 1, 1)
// ---------------------------------------------------------------------------
kernel void mtp_moe_down_reduce(const device half *act_f16 [[buffer(0)]],
                                const device uchar *weights_base [[buffer(1)]],
                                const device MtpExpertMeta *expert_meta [[buffer(2)]],
                                const device uint *expert_ids [[buffer(3)]],
                                const device float *expert_weights [[buffer(4)]],
                                device half *x_inout_f16 [[buffer(5)]],
                                constant MtpMoeExpertParams &p [[buffer(6)]],
                                uint2 tg_pos [[threadgroup_position_in_grid]],
                                uint sg_id [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    const uint row = tg_pos.x * 4u + sg_id;
    const uint tok = tg_pos.y;
    if (row >= p.hidden_size || tok >= p.m) {
        return;
    }

    float total = 0.0f;
    for (uint k_slot = 0u; k_slot < p.top_k; k_slot++) {
        const ulong slot_idx = ulong(tok) * p.top_k + k_slot;
        const uint eid = expert_ids[slot_idx];
        const float w = expert_weights[slot_idx];
        const MtpExpertMeta meta = expert_meta[eid];
        if (meta.down_off == 0xFFFFFFFFFFFFFFFFull) {
            continue;
        }
        const uint exp_dt = meta.weight_dtype;
        const uint exp_gs = (meta.group_size > 0u) ? meta.group_size : p.group_size;
        const device half *act_row = act_f16 + slot_idx * p.moe_inter;
        const float y_k = mtp_moe_dot_row(act_row, weights_base, meta.down_off, meta.down_sb_off, row, p.moe_inter, exp_dt, exp_gs, lane);
        total = fma(w, y_k, total);
    }
    if (lane == 0u) {
        const ulong x_off = ulong(tok) * p.hidden_size + row;
        x_inout_f16[x_off] = half(float(x_inout_f16[x_off]) + total);
    }
}

// ---------------------------------------------------------------------------
// Kernel 27 (`D15` `DFlash2` `DFlashGroupedConv`):
// Exact grouped dynamic depthwise K-tap block convolution (`sglang_dflash.py:_grouped_conv`)
// with optional fused residual add on `side == 1` (`finish`).
// Grid: (hidden_size, seq_len, 1), TG: (<=256, 1, 1)
// ---------------------------------------------------------------------------
kernel void dflash2_grouped_conv(const device half *x_f16 [[buffer(0)]],
                                 const device half *delta_f16 [[buffer(1)]],
                                 const device float *base_f32 [[buffer(2)]],
                                 const device half *residual_f16 [[buffer(3)]],
                                 device half *out_f16 [[buffer(4)]],
                                 constant DFlash2GroupedConvParams &p [[buffer(5)]],
                                 uint2 gid [[thread_position_in_grid]]) {
    const uint c = gid.x;
    const uint t = gid.y;
    if (c >= p.hidden_size || t >= p.seq_len || p.group_size == 0u) {
        return;
    }

    const uint pos = (p.block_size > 0u) ? (t % p.block_size) : t;
    const uint g = c / p.group_size;
    const uint side_taps = p.taps * p.num_groups;
    const ulong delta_row = ulong(t) * (2u * side_taps) + ulong(p.side) * side_taps;
    const ulong base_side = ulong(p.side) * ulong(p.taps) * p.hidden_size;

    float acc = 0.0f;
    for (uint tap = 0u; tap < p.taps; tap++) {
        if (tap <= pos) {
            const float coeff = base_f32[base_side + ulong(tap) * p.hidden_size + c] +
                                float(delta_f16[delta_row + ulong(tap) * p.num_groups + g]);
            acc = fma(coeff, float(x_f16[ulong(t - tap) * p.hidden_size + c]), acc);
        }
    }
    if (p.add_residual != 0u) {
        acc += float(residual_f16[ulong(t) * p.hidden_size + c]);
    }
    out_f16[ulong(t) * p.hidden_size + c] = half(acc);
}

static inline float dflash2_transform_unary(float v, float mult, float softcap) {
    float out = v * mult;
    if (softcap > 0.0f) {
        out = tanh(out / softcap) * softcap;
    }
    return out;
}

static inline void dflash2_topk16_consider(float v,
                                           uint idx,
                                           uint k,
                                           threadgroup float *tv,
                                           threadgroup uint *ti,
                                           thread float &min_v,
                                           thread uint &min_i) {
    if (v > min_v || (v == min_v && idx < min_i)) {
        uint pos = k - 1u;
        while (pos > 0u) {
            const float pv = tv[pos - 1u];
            const uint pi = ti[pos - 1u];
            if (v > pv || (v == pv && idx < pi)) {
                tv[pos] = pv;
                ti[pos] = pi;
                pos--;
            } else {
                break;
            }
        }
        tv[pos] = v;
        ti[pos] = idx;
        min_v = tv[k - 1u];
        min_i = ti[k - 1u];
    }
}

// ---------------------------------------------------------------------------
// Kernel 28 (`D15` `DFlash2` `CandidateSelector` Unary Top-K <= 16):
// Extracts deterministic top-K `(token_id, transformed_unary_logit)` per slot.
// Uses 64 threads (`2` simdgroups, `8 KiB` TG memory) so Metal GPU shader
// validation (`MTL_SHADER_VALIDATION=1`) stays well under the `32 KiB` TG limit.
// Grid: (num_slots, 1, 1), TG: (64, 1, 1)
// ---------------------------------------------------------------------------
kernel void dflash2_topk_unary_logits(const device uchar *logits_raw [[buffer(0)]],
                                      device uint *out_cand_ids [[buffer(1)]],
                                      device float *out_unary_logits [[buffer(2)]],
                                      constant DFlash2TopKUnaryParams &p [[buffer(3)]],
                                      uint tg_pos [[threadgroup_position_in_grid]],
                                      uint tid [[thread_index_in_threadgroup]],
                                      uint sg_id [[simdgroup_index_in_threadgroup]],
                                      uint lane [[thread_index_in_simdgroup]]) {
    const uint slot = tg_pos;
    if (slot >= p.num_slots || p.eff_vocab_size == 0u) {
        return;
    }
    const uint k = clamp(p.top_k, 1u, 16u);
    threadgroup float tg_tv[64 * 16];
    threadgroup uint tg_ti[64 * 16];
    threadgroup float *my_tv = tg_tv + tid * 16u;
    threadgroup uint *my_ti = tg_ti + tid * 16u;

    for (uint j = 0u; j < k; j++) {
        my_tv[j] = -INFINITY;
        my_ti[j] = 0xFFFFFFFFu;
    }
    float min_v = -INFINITY;
    uint min_i = 0xFFFFFFFFu;

    const ulong row_elems = ulong(slot) * p.vocab_size;
    if (p.logits_is_f16 != 0u) {
        const device half *logits = ((const device half *)logits_raw) + row_elems;
        if (p.aligned_vec4 != 0u && ((row_elems & 3ul) == 0ul)) {
            const uint vec_n = p.eff_vocab_size >> 2;
            const device half4 *vlogits = (const device half4 *)logits;
            for (uint vi = tid; vi < vec_n; vi += 64u) {
                const float4 fv = float4(vlogits[vi]);
                const uint base = vi << 2;
                #pragma clang loop unroll(full)
                for (uint c = 0u; c < 4u; c++) {
                    const float sc = dflash2_transform_unary(fv[c], p.output_multiplier, p.softcap);
                    dflash2_topk16_consider(sc, base + c, k, my_tv, my_ti, min_v, min_i);
                }
            }
            for (uint i = (vec_n << 2) + tid; i < p.eff_vocab_size; i += 64u) {
                const float sc = dflash2_transform_unary(float(logits[i]), p.output_multiplier, p.softcap);
                dflash2_topk16_consider(sc, i, k, my_tv, my_ti, min_v, min_i);
            }
        } else {
            for (uint i = tid; i < p.eff_vocab_size; i += 64u) {
                const float sc = dflash2_transform_unary(float(logits[i]), p.output_multiplier, p.softcap);
                dflash2_topk16_consider(sc, i, k, my_tv, my_ti, min_v, min_i);
            }
        }
    } else {
        const device float *logits = ((const device float *)logits_raw) + row_elems;
        for (uint i = tid; i < p.eff_vocab_size; i += 64u) {
            const float sc = dflash2_transform_unary(logits[i], p.output_multiplier, p.softcap);
            dflash2_topk16_consider(sc, i, k, my_tv, my_ti, min_v, min_i);
        }
    }

    threadgroup float s_vals[2];
    threadgroup uint s_idxs[2];
    threadgroup uint s_win_i;

    uint cur = 0u;
    const ulong out_base = ulong(slot) * k;
    for (uint out_k = 0u; out_k < k; out_k++) {
        float cv = (cur < k) ? my_tv[cur] : -INFINITY;
        uint ci = (cur < k) ? my_ti[cur] : 0xFFFFFFFFu;
        for (ushort offset = 16u; offset > 0u; offset >>= 1u) {
            const float ov = simd_shuffle_down(cv, offset);
            const uint oi = simd_shuffle_down(ci, offset);
            if (ov > cv || (ov == cv && oi < ci)) {
                cv = ov;
                ci = oi;
            }
        }
        if (lane == 0u) {
            s_vals[sg_id] = cv;
            s_idxs[sg_id] = ci;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg_id == 0u) {
            float fv = (lane < 2u) ? s_vals[lane] : -INFINITY;
            uint fi = (lane < 2u) ? s_idxs[lane] : 0xFFFFFFFFu;
            const float ov = simd_shuffle_down(fv, 1u);
            const uint oi = simd_shuffle_down(fi, 1u);
            if (ov > fv || (ov == fv && oi < fi)) {
                fv = ov;
                fi = oi;
            }
            if (lane == 0u) {
                s_win_i = fi;
                out_cand_ids[out_base + out_k] = fi;
                out_unary_logits[out_base + out_k] = fv;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (cur < k && my_ti[cur] == s_win_i) {
            cur++;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

static inline float dflash2_load_cb_elem(const device uchar *raw, ulong idx, uint dtype) {
    if (dtype == 0u) {
        return ((const device float *)raw)[idx];
    } else if (dtype == 1u) {
        return float(((const device bfloat *)raw)[idx]);
    }
    return float(((const device half *)raw)[idx]);
}

// ---------------------------------------------------------------------------
// Kernel 29 (`D15` `DFlash2` `CandidateSelector` Pairwise Edge Lattice):
// `scores[e, p, c] = unary[e, c] + <pred_cb[pred_id] * hidden[e], succ_cb[cand_ids[e, c]]>`
// Grid: (top_k, num_slots, 1), TG: (128, 1, 1)
// ---------------------------------------------------------------------------
kernel void dflash2_selector_score_edges(const device uchar *pred_cb_raw [[buffer(0)]],
                                         const device uchar *succ_cb_raw [[buffer(1)]],
                                         const device uchar *hidden_raw [[buffer(2)]],
                                         const device uint *cand_ids [[buffer(3)]],
                                         const device float *unary_logits [[buffer(4)]],
                                         device float *out_scores [[buffer(5)]],
                                         constant DFlash2SelectorScoreParams &p [[buffer(6)]],
                                         uint2 tg_pos [[threadgroup_position_in_grid]],
                                         uint tid [[thread_index_in_threadgroup]],
                                         uint sg_id [[simdgroup_index_in_threadgroup]],
                                         uint lane [[thread_index_in_simdgroup]]) {
    const uint pred_idx = tg_pos.x;
    const uint slot = tg_pos.y;
    if (pred_idx >= p.top_k || slot >= p.num_slots || p.state_rank == 0u || p.state_rank > 512u) {
        return;
    }

    threadgroup float tg_ph[512];
    const uint pred_tok = (slot == 0u) ? p.anchor_token_id : cand_ids[ulong(slot - 1u) * p.top_k + pred_idx];
    const bool pred_valid = (pred_tok < p.vocab_size);
    const ulong pred_row = ulong(pred_valid ? pred_tok : 0u) * p.state_rank;
    const ulong hid_row = ulong(slot) * p.state_rank;

    for (uint r = tid; r < p.state_rank; r += 128u) {
        const float pv = pred_valid ? dflash2_load_cb_elem(pred_cb_raw, pred_row + r, p.pred_dtype) : 0.0f;
        const float hv = (p.hidden_is_f32 != 0u)
                             ? ((const device float *)hidden_raw)[hid_row + r]
                             : float(((const device half *)hidden_raw)[hid_row + r]);
        tg_ph[r] = pv * hv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const ulong slot_c_base = ulong(slot) * p.top_k;
    const ulong score_base = (ulong(slot) * p.top_k + pred_idx) * p.top_k;
    for (uint c = sg_id; c < p.top_k; c += 4u) {
        const uint cand_tok = cand_ids[slot_c_base + c];
        const bool cand_valid = (cand_tok < p.vocab_size);
        float dot = 0.0f;
        if (cand_valid) {
            const ulong succ_row = ulong(cand_tok) * p.state_rank;
            for (uint r = lane; r < p.state_rank; r += 32u) {
                dot = fma(tg_ph[r], dflash2_load_cb_elem(succ_cb_raw, succ_row + r, p.succ_dtype), dot);
            }
        }
        const float total_dot = simd_sum(dot);
        if (lane == 0u) {
            out_scores[score_base + c] = cand_valid ? (unary_logits[slot_c_base + c] + total_dot) : -INFINITY;
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 30 (`D15` `DFlash2` `CandidateSelector` Greedy Lattice Path Walk):
// Walks `e = 0 .. num_slots - 1`, selecting best & optional runner-up candidate
// with softmax probabilities over the `K` transition scores.
// Grid: (1, 1, 1), TG: (32, 1, 1)
// ---------------------------------------------------------------------------
kernel void dflash2_selector_walk_path(const device uint *cand_ids [[buffer(0)]],
                                       const device float *scores [[buffer(1)]],
                                       device DFlareTopKCandidate *out_cands [[buffer(2)]],
                                       constant DFlash2SelectorWalkParams &p [[buffer(3)]],
                                       uint lane [[thread_index_in_simdgroup]]) {
    if (p.num_slots == 0u || p.top_k == 0u || p.top_k > 16u || p.out_k == 0u || p.out_k > 2u) {
        return;
    }

    uint prev_idx = 0u;
    for (uint slot = 0u; slot < p.num_slots; slot++) {
        const ulong row_base = (ulong(slot) * p.top_k + prev_idx) * p.top_k;
        const ulong cid_base = ulong(slot) * p.top_k;

        const bool active = (lane < p.top_k);
        const uint tok = active ? cand_ids[cid_base + lane] : 0xFFFFFFFFu;
        const float sc = (active && tok != 0xFFFFFFFFu) ? (scores[row_base + lane] * p.inv_temp) : -INFINITY;

        const float row_max = simd_max(sc);
        const float ew = (active && isfinite(sc)) ? exp(sc - row_max) : 0.0f;
        const float row_sum = simd_sum(ew);
        const float log_sum = (row_sum > 0.0f) ? log(row_sum) : 0.0f;

        // Pick best (top-1) candidate deterministically: highest score, then smaller token_id.
        float b1_v = sc;
        uint b1_tok = tok;
        uint b1_c = active ? lane : 0xFFFFFFFFu;
        for (ushort off = 16u; off > 0u; off >>= 1u) {
            const float ov = simd_shuffle_down(b1_v, off);
            const uint ot = simd_shuffle_down(b1_tok, off);
            const uint oc = simd_shuffle_down(b1_c, off);
            if (ov > b1_v || (ov == b1_v && ot < b1_tok)) {
                b1_v = ov;
                b1_tok = ot;
                b1_c = oc;
            }
        }
        const uint win1_c = simd_broadcast(b1_c, 0u);
        const float win1_v = simd_broadcast(b1_v, 0u);
        const uint win1_tok = simd_broadcast(b1_tok, 0u);

        if (lane == 0u) {
            DFlareTopKCandidate c1;
            if (win1_tok != 0xFFFFFFFFu && row_sum > 0.0f) {
                c1.token_id = win1_tok;
                c1.log_prob = (win1_v - row_max) - log_sum;
                c1.prob = exp(win1_v - row_max) / row_sum;
                c1._pad = 0u;
            } else {
                c1.token_id = 0xFFFFFFFFu;
                c1.prob = 0.0f;
                c1.log_prob = -INFINITY;
                c1._pad = 0u;
            }
            out_cands[ulong(slot) * p.out_k + 0u] = c1;
        }

        if (p.out_k >= 2u) {
            const bool active2 = active && (lane != win1_c);
            float b2_v = active2 ? sc : -INFINITY;
            uint b2_tok = active2 ? tok : 0xFFFFFFFFu;
            for (ushort off = 16u; off > 0u; off >>= 1u) {
                const float ov = simd_shuffle_down(b2_v, off);
                const uint ot = simd_shuffle_down(b2_tok, off);
                if (ov > b2_v || (ov == b2_v && ot < b2_tok)) {
                    b2_v = ov;
                    b2_tok = ot;
                }
            }
            if (lane == 0u) {
                DFlareTopKCandidate c2;
                if (b2_tok != 0xFFFFFFFFu && row_sum > 0.0f) {
                    c2.token_id = b2_tok;
                    c2.log_prob = (b2_v - row_max) - log_sum;
                    c2.prob = exp(b2_v - row_max) / row_sum;
                    c2._pad = 0u;
                } else {
                    c2.token_id = 0xFFFFFFFFu;
                    c2.prob = 0.0f;
                    c2.log_prob = -INFINITY;
                    c2._pad = 0u;
                }
                out_cands[ulong(slot) * p.out_k + 1u] = c2;
            }
        }

        prev_idx = (win1_c < p.top_k) ? win1_c : 0u;
    }
}
