// MoE router, deduplicated active-expert fused 3-bit / 4-bit affine SwiGLU
// FFN (G3), dense SwiGLU utilities, and single-dispatch greedy argmax.
//
// Specialized at startup via Metal 4 function constants so hot loops carry
// zero bit-width or top-k branching (ADR-0009, ADR-0011, ADR-0014).
#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

constant uint Q_BITS [[function_constant(0)]];
constant uint Q_GROUP_SIZE [[function_constant(1)]];
constant uint MOE_TOP_K [[function_constant(2)]];
constant bool LOGITS_F16 [[function_constant(3)]];
constant bool ROUTER_W_F16 [[function_constant(4)]];
// Native BF16 router weights (ADR-0016); takes precedence over ROUTER_W_F16.
constant bool ROUTER_W_BF16 [[function_constant(5)]];
// Native FP16 / BF16 expert weights for the dense-float MoE SwiGLU (ADR-0016).
constant bool MOE_W_BF16 [[function_constant(6)]];
// Embedding table storage: 0 = f16, 1 = bf16, 2 = q4_affine, 3 = q3_affine,
// 4 = q2_affine (matches `quant.WeightDType`).
constant uint EMBED_KIND [[function_constant(7)]];

struct ExpertWeights {
    ulong gate_qweight_offset;
    ulong gate_sb_offset;
    ulong up_qweight_offset;
    ulong up_sb_offset;
    ulong down_qweight_offset;
    ulong down_sb_offset;
};

struct MoeScheduleHeader {
    uint num_unique_experts;
    uint total_assignments;
    uint m;
    uint top_k;
};

struct RouterProjParams {
    uint m;
    uint num_experts;
    uint hidden_size;
    uint reserved;
};

struct RouteParams {
    uint m;
    uint num_experts;
    uint top_k;
    uint norm_topk_prob;
};

struct MoeGateUpParams {
    uint hidden_size;
    uint intermediate_size;
    uint num_groups;
    uint row_bytes;
};

struct MoeDownParams {
    uint hidden_size;
    uint intermediate_size;
    uint num_groups;
    uint row_bytes;
    uint top_k;
    uint reserved;
};

struct MoeReduceParams {
    uint m;
    uint hidden_size;
    uint top_k;
    uint add_residual;
};

struct ElemParams {
    uint count;
};

struct ArgmaxParams {
    uint vocab_size;
    uint aligned_half4;
};

static inline float fast_silu(float x) {
    return x / (1.0f + exp(-x));
}

// Prescaled bit-field 8-element dot products for single-token fast path (M=1 / count_u=1).
static inline float qdot8_q4(uint32_t w, float4 px0, float4 px1) {
    const uint lo = w & 0xFFFFu;
    const uint hi = w >> 16;
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

// Weight-stationary 8-element affine dequantization shared by both `count_u == 1`
// and `count_u > 1` deduplicated MoE paths for bit-exact FP32 accumulation parity.
static inline void dequant8_q4(uint32_t w, float scale, float bias,
                               thread float4 &w0, thread float4 &w1) {
    w0 = float4(fma(float((w >>  0u) & 0xFu), scale, bias),
                fma(float((w >>  4u) & 0xFu), scale, bias),
                fma(float((w >>  8u) & 0xFu), scale, bias),
                fma(float((w >> 12u) & 0xFu), scale, bias));
    w1 = float4(fma(float((w >> 16u) & 0xFu), scale, bias),
                fma(float((w >> 20u) & 0xFu), scale, bias),
                fma(float((w >> 24u) & 0xFu), scale, bias),
                fma(float((w >> 28u) & 0xFu), scale, bias));
}

static inline void dequant8_q3(uchar b0, uchar b1, uchar b2,
                               float scale, float bias,
                               thread float4 &w0, thread float4 &w1) {
    const uint word = uint(b0) | (uint(b1) << 8) | (uint(b2) << 16);
    w0 = float4(fma(float((word >>  0u) & 0x7u), scale, bias),
                fma(float((word >>  3u) & 0x7u), scale, bias),
                fma(float((word >>  6u) & 0x7u), scale, bias),
                fma(float((word >>  9u) & 0x7u), scale, bias));
    w1 = float4(fma(float((word >> 12u) & 0x7u), scale, bias),
                fma(float((word >> 15u) & 0x7u), scale, bias),
                fma(float((word >> 18u) & 0x7u), scale, bias),
                fma(float((word >> 21u) & 0x7u), scale, bias));
}

static inline void dequant8_q2(uint16_t w, float scale, float bias,
                               thread float4 &w0, thread float4 &w1) {
    const uint u = uint(w);
    w0 = float4(fma(float((u >>  0u) & 0x3u), scale, bias),
                fma(float((u >>  2u) & 0x3u), scale, bias),
                fma(float((u >>  4u) & 0x3u), scale, bias),
                fma(float((u >>  6u) & 0x3u), scale, bias));
    w1 = float4(fma(float((u >>  8u) & 0x3u), scale, bias),
                fma(float((u >> 10u) & 0x3u), scale, bias),
                fma(float((u >> 12u) & 0x3u), scale, bias),
                fma(float((u >> 14u) & 0x3u), scale, bias));
}

// ---------------------------------------------------------------------------
// Kernel 1: MoE Router Projection (`[M, hidden] @ router_w^T -> [M, E] f32`)
// with f32, f16 (ROUTER_W_F16) or bf16 (ROUTER_W_BF16) router weights.
//
// 4 simdgroups / threadgroup (128 threads), 1 simdgroup per (tok, expert).
// ---------------------------------------------------------------------------
kernel void ami_moe_router_proj(const device half *x [[buffer(0)]],
                                const device uchar *router_w [[buffer(1)]],
                                device float *logits [[buffer(2)]],
                                constant RouterProjParams &p [[buffer(3)]],
                                uint2 tg_pos [[threadgroup_position_in_grid]],
                                uint sg_id [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    const uint e = tg_pos.x * 4u + sg_id;
    const uint tok = tg_pos.y;
    if (e >= p.num_experts || tok >= p.m) {
        return;
    }

    float acc = 0.0f;
    if ((p.hidden_size & 3u) == 0u) {
        const uint vec_k = p.hidden_size >> 2;
        const device half4 *x_row = (const device half4 *)(x + ulong(tok) * p.hidden_size);
        if (ROUTER_W_BF16) {
            const device bfloat4 *w_row =
                (const device bfloat4 *)(((const device bfloat *)router_w) + ulong(e) * p.hidden_size);
            for (uint k = lane; k < vec_k; k += 32u) {
                acc += dot(float4(x_row[k]), float4(w_row[k]));
            }
        } else if (ROUTER_W_F16) {
            const device half4 *w_row =
                (const device half4 *)(((const device half *)router_w) + ulong(e) * p.hidden_size);
            for (uint k = lane; k < vec_k; k += 32u) {
                acc += dot(float4(x_row[k]), float4(w_row[k]));
            }
        } else {
            const device float4 *w_row =
                (const device float4 *)(((const device float *)router_w) + ulong(e) * p.hidden_size);
            for (uint k = lane; k < vec_k; k += 32u) {
                acc += dot(float4(x_row[k]), w_row[k]);
            }
        }
    } else {
        const device half *x_row = x + ulong(tok) * p.hidden_size;
        if (ROUTER_W_BF16) {
            const device bfloat *w_row = ((const device bfloat *)router_w) + ulong(e) * p.hidden_size;
            for (uint k = lane; k < p.hidden_size; k += 32u) {
                acc = fma(float(x_row[k]), float(w_row[k]), acc);
            }
        } else if (ROUTER_W_F16) {
            const device half *w_row = ((const device half *)router_w) + ulong(e) * p.hidden_size;
            for (uint k = lane; k < p.hidden_size; k += 32u) {
                acc = fma(float(x_row[k]), float(w_row[k]), acc);
            }
        } else {
            const device float *w_row = ((const device float *)router_w) + ulong(e) * p.hidden_size;
            for (uint k = lane; k < p.hidden_size; k += 32u) {
                acc = fma(float(x_row[k]), w_row[k], acc);
            }
        }
    }

    const float total = simd_sum(acc);
    if (lane == 0u) {
        logits[ulong(tok) * p.num_experts + e] = total;
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: MoE Softmax + Top-K + Deduplicated Active-Expert CSR Schedule
//
// Single threadgroup of 256 threads (8 simdgroups). Supports num_experts <= 1024
// and top_k <= 16. Each simdgroup handles one token fully in registers with
// zero barriers for softmax and top-k selection (E <= 256 fast path retains
// 8-float register buffering; 256 < E <= 1024 streams with a 32-bit picked mask
// to avoid dynamic register spills), then all 256 threads compact active experts
// into sorted unique_experts[0..U) and scatter CSR token slots.
// ---------------------------------------------------------------------------
kernel void ami_moe_route_topk(const device uchar *logits_raw [[buffer(0)]],
                               device MoeScheduleHeader *sched_hdr [[buffer(1)]],
                               device uint *unique_experts [[buffer(2)]],
                               device uint *expert_token_offsets [[buffer(3)]],
                               device uint *token_indices [[buffer(4)]],
                               device float *routing_weights [[buffer(5)]],
                               device uint *slot_k_indices [[buffer(6)]],
                               device uint *scratch_topk_experts [[buffer(7)]],
                               device float *scratch_topk_weights [[buffer(8)]],
                               constant RouteParams &p [[buffer(9)]],
                               uint tid [[thread_index_in_threadgroup]],
                               uint sg_id [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
    threadgroup atomic_uint s_counts[1024];
    threadgroup atomic_uint s_cursors[1024];
    threadgroup uint s_prefix_tmp[8];
    threadgroup uint s_blk_unique;
    threadgroup uint s_blk_toks;

    const uint top_k = (MOE_TOP_K > 0u) ? MOE_TOP_K : min(p.top_k, 16u);

    // Phase 0: clear per-expert token histograms.
    for (uint e = tid; e < p.num_experts; e += 256u) {
        atomic_store_explicit(&s_counts[e], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 1: parallel per-token fp32 softmax & top-k in registers (8 tokens/wave).
    for (uint tok = sg_id; tok < p.m; tok += 8u) {
        uint sel_e[16];
        float sel_w[16];
        float w_sum = 0.0f;

        if (p.num_experts <= 256u) {
            float vals[8];
            float local_max = -INFINITY;
            for (uint r = 0u; r < 8u; r++) {
                const uint e = lane + r * 32u;
                if (e < p.num_experts) {
                    const float v = LOGITS_F16
                        ? float(((const device half *)logits_raw)[ulong(tok) * p.num_experts + e])
                        : ((const device float *)logits_raw)[ulong(tok) * p.num_experts + e];
                    vals[r] = v;
                    local_max = max(local_max, v);
                } else {
                    vals[r] = -INFINITY;
                }
            }

            const float max_v = simd_max(local_max);
            float local_sum = 0.0f;
            for (uint r = 0u; r < 8u; r++) {
                const uint e = lane + r * 32u;
                if (e < p.num_experts) {
                    const float ex = exp(vals[r] - max_v);
                    vals[r] = ex;
                    local_sum += ex;
                } else {
                    vals[r] = -INFINITY;
                }
            }

            const float sum_exp = simd_sum(local_sum);
            const float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
            for (uint r = 0u; r < 8u; r++) {
                const uint e = lane + r * 32u;
                if (e < p.num_experts) {
                    vals[r] *= inv_sum;
                }
            }

            const uint k_limit = (MOE_TOP_K > 0u) ? MOE_TOP_K : top_k;
            for (uint k = 0u; k < k_limit; k++) {
                float best_v = -INFINITY;
                uint best_e = 0xFFFFFFFFu;
                for (uint r = 0u; r < 8u; r++) {
                    const uint e = lane + r * 32u;
                    if (e < p.num_experts) {
                        const float v = vals[r];
                        if (v > best_v || (v == best_v && e < best_e)) {
                            best_v = v;
                            best_e = e;
                        }
                    }
                }
                for (ushort offset = 16u; offset > 0u; offset >>= 1u) {
                    const float ov = simd_shuffle_down(best_v, offset);
                    const uint oe = simd_shuffle_down(best_e, offset);
                    if (ov > best_v || (ov == best_v && oe < best_e)) {
                        best_v = ov;
                        best_e = oe;
                    }
                }
                const uint raw_win_e = simd_broadcast_first(best_e);
                const uint win_e = (raw_win_e < p.num_experts) ? raw_win_e : 0u;
                const float win_v = simd_broadcast_first(best_v);
                if ((win_e & 31u) == lane) {
                    vals[win_e >> 5] = -INFINITY;
                }
                if (lane == 0u) {
                    sel_e[k] = win_e;
                    sel_w[k] = win_v;
                    w_sum += win_v;
                }
            }
        } else {
            const uint chunks = (p.num_experts + 31u) >> 5;
            const ulong row_base = ulong(tok) * p.num_experts;
            float local_max = -INFINITY;
            for (uint r = 0u; r < chunks; r++) {
                const uint e = lane + r * 32u;
                if (e < p.num_experts) {
                    const float v = LOGITS_F16
                        ? float(((const device half *)logits_raw)[row_base + e])
                        : ((const device float *)logits_raw)[row_base + e];
                    local_max = max(local_max, v);
                }
            }
            const float max_v = simd_max(local_max);

            float local_sum = 0.0f;
            for (uint r = 0u; r < chunks; r++) {
                const uint e = lane + r * 32u;
                if (e < p.num_experts) {
                    const float v = LOGITS_F16
                        ? float(((const device half *)logits_raw)[row_base + e])
                        : ((const device float *)logits_raw)[row_base + e];
                    local_sum += exp(v - max_v);
                }
            }
            const float sum_exp = simd_sum(local_sum);
            const float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;

            uint picked_mask = 0u;
            const uint k_limit = (MOE_TOP_K > 0u) ? MOE_TOP_K : top_k;
            for (uint k = 0u; k < k_limit; k++) {
                float best_v = -INFINITY;
                uint best_e = 0xFFFFFFFFu;
                for (uint r = 0u; r < chunks; r++) {
                    const uint e = lane + r * 32u;
                    if (e < p.num_experts && ((picked_mask >> r) & 1u) == 0u) {
                        const float raw_v = LOGITS_F16
                            ? float(((const device half *)logits_raw)[row_base + e])
                            : ((const device float *)logits_raw)[row_base + e];
                        const float v = exp(raw_v - max_v) * inv_sum;
                        if (v > best_v || (v == best_v && e < best_e)) {
                            best_v = v;
                            best_e = e;
                        }
                    }
                }
                for (ushort offset = 16u; offset > 0u; offset >>= 1u) {
                    const float ov = simd_shuffle_down(best_v, offset);
                    const uint oe = simd_shuffle_down(best_e, offset);
                    if (ov > best_v || (ov == best_v && oe < best_e)) {
                        best_v = ov;
                        best_e = oe;
                    }
                }
                const uint raw_win_e = simd_broadcast_first(best_e);
                const uint win_e = (raw_win_e < p.num_experts) ? raw_win_e : 0u;
                const float win_v = simd_broadcast_first(best_v);
                if ((win_e & 31u) == lane) {
                    picked_mask |= (1u << (win_e >> 5));
                }
                if (lane == 0u) {
                    sel_e[k] = win_e;
                    sel_w[k] = win_v;
                    w_sum += win_v;
                }
            }
        }

        if (lane == 0u) {
            const float renorm = (p.norm_topk_prob != 0u && w_sum > 0.0f) ? (1.0f / w_sum) : 1.0f;
            const ulong base = ulong(tok) * top_k;
            for (uint k = 0u; k < top_k; k++) {
                const uint e = sel_e[k];
                scratch_topk_experts[base + k] = e;
                scratch_topk_weights[base + k] = sel_w[k] * renorm;
                atomic_fetch_add_explicit(&s_counts[e], 1u, memory_order_relaxed);
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    // Phase 2: compact active experts [0..U) and compute CSR token offsets in 256-expert blocks.
    uint u_base = 0u;
    uint tok_base = 0u;
    for (uint blk_base = 0u; blk_base < p.num_experts; blk_base += 256u) {
        const uint e = blk_base + tid;
        const uint cnt = (e < p.num_experts)
            ? atomic_load_explicit(&s_counts[e], memory_order_relaxed)
            : 0u;
        const uint is_active = (cnt > 0u) ? 1u : 0u;

        // 2a. Exclusive prefix sum of `is_active`.
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
                s_blk_unique = ex + v;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint u_idx = u_base + s_prefix_tmp[sg_id] + sg_act_excl;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2b. Exclusive prefix sum of `cnt`.
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
                s_blk_toks = ex + v;
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
        u_base += s_blk_unique;
        tok_base += s_blk_toks;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        const uint total_assign = p.m * top_k;
        expert_token_offsets[u_base] = total_assign;
        sched_hdr->num_unique_experts = u_base;
        sched_hdr->total_assignments = total_assign;
        sched_hdr->m = p.m;
        sched_hdr->top_k = top_k;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: scatter token assignments into each active expert's contiguous CSR range.
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
// Kernel 3: Deduplicated MoE Fused 3-bit / 4-bit Gate + Up + SwiGLU
//
// Geometry: 4 simdgroups / threadgroup (128 threads), 2 output rows / simdgroup
// (8 rows / threadgroup). tg_pos.y indexes active unique expert `u in 0..U-1`.
// Each active expert reads gate_proj and up_proj weights ONCE across all tokens
// assigned to it (count_u == 1 fast path matching M1 GEMV; count_u > 1
// weight-stationary register batching up to 16 tokens per pass).
// ---------------------------------------------------------------------------
kernel void ami_moe_gate_up_swiglu(const device half *x [[buffer(0)]],
                                   const device uchar *weights_base [[buffer(1)]],
                                   const device ExpertWeights *expert_table [[buffer(2)]],
                                   const device MoeScheduleHeader *sched_hdr [[buffer(3)]],
                                   const device uint *unique_experts [[buffer(4)]],
                                   const device uint *expert_token_offsets [[buffer(5)]],
                                   const device uint *token_indices [[buffer(6)]],
                                   device half *inter [[buffer(7)]],
                                   constant MoeGateUpParams &p [[buffer(8)]],
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

    const uint expert_id = unique_experts[u];
    const ExpertWeights ew = expert_table[expert_id];
    const device uchar *gate_qw = weights_base + ew.gate_qweight_offset;
    const device half2 *gate_sb = (const device half2 *)(weights_base + ew.gate_sb_offset);
    const device uchar *up_qw = weights_base + ew.up_qweight_offset;
    const device half2 *up_sb = (const device half2 *)(weights_base + ew.up_sb_offset);

    const uint sub_group = lane >> 3;   // 0..3 (4 groups of 64 per K-step)
    const uint lane_in_grp = lane & 7u; // 0..7 (8 contiguous elements/lane)

    if (count_u == 1u) {
        const uint tok = token_indices[s0];
        const device half *x_tok = x + ulong(tok) * p.hidden_size;
        const bool full_rows = (row0 + ROWS_PER_SG <= p.intermediate_size);

        float g_acc[ROWS_PER_SG] = {0.0f, 0.0f};
        float u_acc[ROWS_PER_SG] = {0.0f, 0.0f};

        for (uint g_base = 0u; g_base < p.num_groups; g_base += 4u) {
            const uint g = g_base + sub_group;
            if (g >= p.num_groups) {
                continue;
            }

            const uint k_off = g * Q_GROUP_SIZE + lane_in_grp * 8u;
            const float4 fx0 = float4(*(const device half4 *)(x_tok + k_off));
            const float4 fx1 = float4(*(const device half4 *)(x_tok + k_off + 4u));

            for (uint r = 0u; r < ROWS_PER_SG; r++) {
                const uint row = row0 + r;
                if (full_rows || row < p.intermediate_size) {
                    const ulong row_sb_off = ulong(row) * p.num_groups + g;
                    const half2 gsb = gate_sb[row_sb_off];
                    const half2 usb = up_sb[row_sb_off];
                    float4 gw0, gw1, uw0, uw1;
                    if (Q_BITS == 4u) {
                        const ulong row_w_off = ulong(row) * p.row_bytes + g * 32u + lane_in_grp * 4u;
                        dequant8_q4(*(const device uint32_t *)(gate_qw + row_w_off), float(gsb.x), float(gsb.y), gw0, gw1);
                        dequant8_q4(*(const device uint32_t *)(up_qw + row_w_off), float(usb.x), float(usb.y), uw0, uw1);
                    } else if (Q_BITS == 3u) {
                        const ulong row_w_off = ulong(row) * p.row_bytes + g * 24u + lane_in_grp * 3u;
                        const device uchar *gwp = gate_qw + row_w_off;
                        const device uchar *uwp = up_qw + row_w_off;
                        dequant8_q3(gwp[0], gwp[1], gwp[2], float(gsb.x), float(gsb.y), gw0, gw1);
                        dequant8_q3(uwp[0], uwp[1], uwp[2], float(usb.x), float(usb.y), uw0, uw1);
                    } else {
                        const ulong row_w_off = ulong(row) * p.row_bytes + g * 16u + lane_in_grp * 2u;
                        dequant8_q2(*(const device uint16_t *)(gate_qw + row_w_off), float(gsb.x), float(gsb.y), gw0, gw1);
                        dequant8_q2(*(const device uint16_t *)(up_qw + row_w_off), float(usb.x), float(usb.y), uw0, uw1);
                    }
                    g_acc[r] += dot(fx0, gw0) + dot(fx1, gw1);
                    u_acc[r] += dot(fx0, uw0) + dot(fx1, uw1);
                }
            }
        }

        const ulong out_base = ulong(s0) * p.intermediate_size;
        for (uint r = 0u; r < ROWS_PER_SG; r++) {
            const float gt = simd_sum(g_acc[r]);
            const float ut = simd_sum(u_acc[r]);
            if (lane == 0u && (row0 + r) < p.intermediate_size) {
                inter[out_base + row0 + r] = half(fast_silu(gt) * ut);
            }
        }
        return;
    }

    // Multi-token deduplicated path: zero-spill scalar-register batching
    // dequantizing each expert row ONCE across up to 8 tokens (with 4-token fast
    // path), bit-for-bit identical in FP32 accumulation order to `count_u == 1u`.
    for (uint t_base = 0u; t_base < count_u; ) {
        const uint rem = count_u - t_base;
        if (rem <= 4u) {
            const uint t0 = token_indices[s0 + t_base + 0u];
            const uint t1 = token_indices[s0 + t_base + min(1u, rem - 1u)];
            const uint t2 = token_indices[s0 + t_base + min(2u, rem - 1u)];
            const uint t3 = token_indices[s0 + t_base + min(3u, rem - 1u)];
            const device half *x0 = x + ulong(t0) * p.hidden_size;
            const device half *x1 = x + ulong(t1) * p.hidden_size;
            const device half *x2 = x + ulong(t2) * p.hidden_size;
            const device half *x3 = x + ulong(t3) * p.hidden_size;

            for (uint r = 0u; r < ROWS_PER_SG; r++) {
                const uint row = row0 + r;
                if (row >= p.intermediate_size) {
                    break;
                }
                float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
                float u0 = 0.0f, u1 = 0.0f, u2 = 0.0f, u3 = 0.0f;
                const ulong row_w_base = ulong(row) * p.row_bytes;
                const ulong row_sb_base = ulong(row) * p.num_groups;

                for (uint g_base = 0u; g_base < p.num_groups; g_base += 4u) {
                    const uint g = g_base + sub_group;
                    if (g >= p.num_groups) {
                        continue;
                    }
                    const half2 gsb = gate_sb[row_sb_base + g];
                    const half2 usb = up_sb[row_sb_base + g];
                    float4 gw0, gw1, uw0, uw1;
                    if (Q_BITS == 4u) {
                        const uint w_off = g * 32u + lane_in_grp * 4u;
                        dequant8_q4(*(const device uint32_t *)(gate_qw + row_w_base + w_off), float(gsb.x), float(gsb.y), gw0, gw1);
                        dequant8_q4(*(const device uint32_t *)(up_qw + row_w_base + w_off), float(usb.x), float(usb.y), uw0, uw1);
                    } else if (Q_BITS == 3u) {
                        const uint w_off = g * 24u + lane_in_grp * 3u;
                        const device uchar *gwp = gate_qw + row_w_base + w_off;
                        const device uchar *uwp = up_qw + row_w_base + w_off;
                        dequant8_q3(gwp[0], gwp[1], gwp[2], float(gsb.x), float(gsb.y), gw0, gw1);
                        dequant8_q3(uwp[0], uwp[1], uwp[2], float(usb.x), float(usb.y), uw0, uw1);
                    } else {
                        const uint w_off = g * 16u + lane_in_grp * 2u;
                        dequant8_q2(*(const device uint16_t *)(gate_qw + row_w_base + w_off), float(gsb.x), float(gsb.y), gw0, gw1);
                        dequant8_q2(*(const device uint16_t *)(up_qw + row_w_base + w_off), float(usb.x), float(usb.y), uw0, uw1);
                    }

                    const uint k_off = g * Q_GROUP_SIZE + lane_in_grp * 8u;
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
                    inter[ulong(s0 + t_base + 0u) * p.intermediate_size + row] = half(fast_silu(sg0) * su0);
                    if (rem > 1u) inter[ulong(s0 + t_base + 1u) * p.intermediate_size + row] = half(fast_silu(sg1) * su1);
                    if (rem > 2u) inter[ulong(s0 + t_base + 2u) * p.intermediate_size + row] = half(fast_silu(sg2) * su2);
                    if (rem > 3u) inter[ulong(s0 + t_base + 3u) * p.intermediate_size + row] = half(fast_silu(sg3) * su3);
                }
            }
            t_base += rem;
        } else {
            const uint t_cnt = min(8u, rem);
            const device half *x0 = x + ulong(token_indices[s0 + t_base + 0u]) * p.hidden_size;
            const device half *x1 = x + ulong(token_indices[s0 + t_base + 1u]) * p.hidden_size;
            const device half *x2 = x + ulong(token_indices[s0 + t_base + 2u]) * p.hidden_size;
            const device half *x3 = x + ulong(token_indices[s0 + t_base + 3u]) * p.hidden_size;
            const device half *x4 = x + ulong(token_indices[s0 + t_base + 4u]) * p.hidden_size;
            const device half *x5 = x + ulong(token_indices[s0 + t_base + min(5u, t_cnt - 1u)]) * p.hidden_size;
            const device half *x6 = x + ulong(token_indices[s0 + t_base + min(6u, t_cnt - 1u)]) * p.hidden_size;
            const device half *x7 = x + ulong(token_indices[s0 + t_base + min(7u, t_cnt - 1u)]) * p.hidden_size;

            for (uint r = 0u; r < ROWS_PER_SG; r++) {
                const uint row = row0 + r;
                if (row >= p.intermediate_size) {
                    break;
                }
                float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
                float g4 = 0.0f, g5 = 0.0f, g6 = 0.0f, g7 = 0.0f;
                float u0 = 0.0f, u1 = 0.0f, u2 = 0.0f, u3 = 0.0f;
                float u4 = 0.0f, u5 = 0.0f, u6 = 0.0f, u7 = 0.0f;
                const ulong row_w_base = ulong(row) * p.row_bytes;
                const ulong row_sb_base = ulong(row) * p.num_groups;

                for (uint g_base = 0u; g_base < p.num_groups; g_base += 4u) {
                    const uint g = g_base + sub_group;
                    if (g >= p.num_groups) {
                        continue;
                    }
                    const half2 gsb = gate_sb[row_sb_base + g];
                    const half2 usb = up_sb[row_sb_base + g];
                    float4 gw0, gw1, uw0, uw1;
                    if (Q_BITS == 4u) {
                        const uint w_off = g * 32u + lane_in_grp * 4u;
                        dequant8_q4(*(const device uint32_t *)(gate_qw + row_w_base + w_off), float(gsb.x), float(gsb.y), gw0, gw1);
                        dequant8_q4(*(const device uint32_t *)(up_qw + row_w_base + w_off), float(usb.x), float(usb.y), uw0, uw1);
                    } else if (Q_BITS == 3u) {
                        const uint w_off = g * 24u + lane_in_grp * 3u;
                        const device uchar *gwp = gate_qw + row_w_base + w_off;
                        const device uchar *uwp = up_qw + row_w_base + w_off;
                        dequant8_q3(gwp[0], gwp[1], gwp[2], float(gsb.x), float(gsb.y), gw0, gw1);
                        dequant8_q3(uwp[0], uwp[1], uwp[2], float(usb.x), float(usb.y), uw0, uw1);
                    } else {
                        const uint w_off = g * 16u + lane_in_grp * 2u;
                        dequant8_q2(*(const device uint16_t *)(gate_qw + row_w_base + w_off), float(gsb.x), float(gsb.y), gw0, gw1);
                        dequant8_q2(*(const device uint16_t *)(up_qw + row_w_base + w_off), float(usb.x), float(usb.y), uw0, uw1);
                    }

                    const uint k_off = g * Q_GROUP_SIZE + lane_in_grp * 8u;
                    { const float4 a = float4(*(const device half4 *)(x0 + k_off)); const float4 b = float4(*(const device half4 *)(x0 + k_off + 4u)); g0 += dot(a, gw0) + dot(b, gw1); u0 += dot(a, uw0) + dot(b, uw1); }
                    { const float4 a = float4(*(const device half4 *)(x1 + k_off)); const float4 b = float4(*(const device half4 *)(x1 + k_off + 4u)); g1 += dot(a, gw0) + dot(b, gw1); u1 += dot(a, uw0) + dot(b, uw1); }
                    { const float4 a = float4(*(const device half4 *)(x2 + k_off)); const float4 b = float4(*(const device half4 *)(x2 + k_off + 4u)); g2 += dot(a, gw0) + dot(b, gw1); u2 += dot(a, uw0) + dot(b, uw1); }
                    { const float4 a = float4(*(const device half4 *)(x3 + k_off)); const float4 b = float4(*(const device half4 *)(x3 + k_off + 4u)); g3 += dot(a, gw0) + dot(b, gw1); u3 += dot(a, uw0) + dot(b, uw1); }
                    { const float4 a = float4(*(const device half4 *)(x4 + k_off)); const float4 b = float4(*(const device half4 *)(x4 + k_off + 4u)); g4 += dot(a, gw0) + dot(b, gw1); u4 += dot(a, uw0) + dot(b, uw1); }
                    if (t_cnt > 5u) { const float4 a = float4(*(const device half4 *)(x5 + k_off)); const float4 b = float4(*(const device half4 *)(x5 + k_off + 4u)); g5 += dot(a, gw0) + dot(b, gw1); u5 += dot(a, uw0) + dot(b, uw1); }
                    if (t_cnt > 6u) { const float4 a = float4(*(const device half4 *)(x6 + k_off)); const float4 b = float4(*(const device half4 *)(x6 + k_off + 4u)); g6 += dot(a, gw0) + dot(b, gw1); u6 += dot(a, uw0) + dot(b, uw1); }
                    if (t_cnt > 7u) { const float4 a = float4(*(const device half4 *)(x7 + k_off)); const float4 b = float4(*(const device half4 *)(x7 + k_off + 4u)); g7 += dot(a, gw0) + dot(b, gw1); u7 += dot(a, uw0) + dot(b, uw1); }
                }

                const float sg0 = simd_sum(g0), su0 = simd_sum(u0);
                const float sg1 = simd_sum(g1), su1 = simd_sum(u1);
                const float sg2 = simd_sum(g2), su2 = simd_sum(u2);
                const float sg3 = simd_sum(g3), su3 = simd_sum(u3);
                const float sg4 = simd_sum(g4), su4 = simd_sum(u4);
                const float sg5 = simd_sum(g5), su5 = simd_sum(u5);
                const float sg6 = simd_sum(g6), su6 = simd_sum(u6);
                const float sg7 = simd_sum(g7), su7 = simd_sum(u7);
                if (lane == 0u) {
                    inter[ulong(s0 + t_base + 0u) * p.intermediate_size + row] = half(fast_silu(sg0) * su0);
                    inter[ulong(s0 + t_base + 1u) * p.intermediate_size + row] = half(fast_silu(sg1) * su1);
                    inter[ulong(s0 + t_base + 2u) * p.intermediate_size + row] = half(fast_silu(sg2) * su2);
                    inter[ulong(s0 + t_base + 3u) * p.intermediate_size + row] = half(fast_silu(sg3) * su3);
                    inter[ulong(s0 + t_base + 4u) * p.intermediate_size + row] = half(fast_silu(sg4) * su4);
                    if (t_cnt > 5u) inter[ulong(s0 + t_base + 5u) * p.intermediate_size + row] = half(fast_silu(sg5) * su5);
                    if (t_cnt > 6u) inter[ulong(s0 + t_base + 6u) * p.intermediate_size + row] = half(fast_silu(sg6) * su6);
                    if (t_cnt > 7u) inter[ulong(s0 + t_base + 7u) * p.intermediate_size + row] = half(fast_silu(sg7) * su7);
                }
            }
            t_base += t_cnt;
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 4: Deduplicated MoE Fused 3-bit / 4-bit Down Projection * Weight
//
// Geometry: 4 simdgroups / threadgroup (128 threads), 4 output rows / simdgroup
// (16 rows / threadgroup). Reads each active expert's down_proj weights ONCE
// across all assigned tokens, scales by routing_weights[s], and writes fp32
// partial outputs to down_out[(tok * top_k + k_idx) * hidden_size + row] with
// zero atomics and zero write races.
// ---------------------------------------------------------------------------
kernel void ami_moe_down_proj(const device half *inter [[buffer(0)]],
                              const device uchar *weights_base [[buffer(1)]],
                              const device ExpertWeights *expert_table [[buffer(2)]],
                              const device MoeScheduleHeader *sched_hdr [[buffer(3)]],
                              const device uint *unique_experts [[buffer(4)]],
                              const device uint *expert_token_offsets [[buffer(5)]],
                              const device uint *token_indices [[buffer(6)]],
                              const device float *routing_weights [[buffer(7)]],
                              const device uint *slot_k_indices [[buffer(8)]],
                              device float *down_out [[buffer(9)]],
                              constant MoeDownParams &p [[buffer(10)]],
                              uint2 tg_pos [[threadgroup_position_in_grid]],
                              uint sg_id [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint ROWS_PER_SG = 4u;
    const uint u = tg_pos.y;
    if (u >= sched_hdr->num_unique_experts) {
        return;
    }

    const uint row0 = tg_pos.x * 16u + sg_id * ROWS_PER_SG;
    if (row0 >= p.hidden_size) {
        return;
    }

    const uint s0 = expert_token_offsets[u];
    const uint s1 = expert_token_offsets[u + 1u];
    const uint count_u = s1 - s0;
    if (count_u == 0u) {
        return;
    }

    const uint expert_id = unique_experts[u];
    const ExpertWeights ew = expert_table[expert_id];
    const device uchar *down_qw = weights_base + ew.down_qweight_offset;
    const device half2 *down_sb = (const device half2 *)(weights_base + ew.down_sb_offset);

    const uint sub_group = lane >> 3;
    const uint lane_in_grp = lane & 7u;

    if (count_u == 1u) {
        const device half *in_row = inter + ulong(s0) * p.intermediate_size;
        const bool full_rows = (row0 + ROWS_PER_SG <= p.hidden_size);
        float acc[ROWS_PER_SG] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (uint g_base = 0u; g_base < p.num_groups; g_base += 4u) {
            const uint g = g_base + sub_group;
            if (g >= p.num_groups) {
                continue;
            }

            const uint k_off = g * Q_GROUP_SIZE + lane_in_grp * 8u;
            const float4 fx0 = float4(*(const device half4 *)(in_row + k_off));
            const float4 fx1 = float4(*(const device half4 *)(in_row + k_off + 4u));

            for (uint r = 0u; r < ROWS_PER_SG; r++) {
                const uint row = row0 + r;
                if (full_rows || row < p.hidden_size) {
                    const half2 sb = down_sb[ulong(row) * p.num_groups + g];
                    float4 dw0, dw1;
                    if (Q_BITS == 4u) {
                        const ulong row_w_off = ulong(row) * p.row_bytes + g * 32u + lane_in_grp * 4u;
                        dequant8_q4(*(const device uint32_t *)(down_qw + row_w_off), float(sb.x), float(sb.y), dw0, dw1);
                    } else if (Q_BITS == 3u) {
                        const ulong row_w_off = ulong(row) * p.row_bytes + g * 24u + lane_in_grp * 3u;
                        const device uchar *wp = down_qw + row_w_off;
                        dequant8_q3(wp[0], wp[1], wp[2], float(sb.x), float(sb.y), dw0, dw1);
                    } else {
                        const ulong row_w_off = ulong(row) * p.row_bytes + g * 16u + lane_in_grp * 2u;
                        dequant8_q2(*(const device uint16_t *)(down_qw + row_w_off), float(sb.x), float(sb.y), dw0, dw1);
                    }
                    acc[r] += dot(fx0, dw0) + dot(fx1, dw1);
                }
            }
        }

        const uint tok = token_indices[s0];
        const uint k_idx = slot_k_indices[s0];
        const float rw = routing_weights[s0];
        const ulong out_base = (ulong(tok) * p.top_k + k_idx) * p.hidden_size;

        for (uint r = 0u; r < ROWS_PER_SG; r++) {
            const float total = simd_sum(acc[r]);
            if (lane == 0u && (row0 + r) < p.hidden_size) {
                down_out[out_base + row0 + r] = total * rw;
            }
        }
        return;
    }

    // Multi-token deduplicated path: zero-spill scalar-register batching
    // dequantizing each down_proj row ONCE across up to 8 tokens (with 4-token
    // fast path), bit-for-bit identical in FP32 accumulation order to `count_u == 1u`.
    for (uint t_base = 0u; t_base < count_u; ) {
        const uint rem = count_u - t_base;
        if (rem <= 4u) {
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

                for (uint g_base = 0u; g_base < p.num_groups; g_base += 4u) {
                    const uint g = g_base + sub_group;
                    if (g >= p.num_groups) {
                        continue;
                    }
                    const half2 sb = down_sb[row_sb_base + g];
                    float4 dw0, dw1;
                    if (Q_BITS == 4u) {
                        const uint w_off = g * 32u + lane_in_grp * 4u;
                        dequant8_q4(*(const device uint32_t *)(down_qw + row_w_base + w_off), float(sb.x), float(sb.y), dw0, dw1);
                    } else if (Q_BITS == 3u) {
                        const uint w_off = g * 24u + lane_in_grp * 3u;
                        const device uchar *wp = down_qw + row_w_base + w_off;
                        dequant8_q3(wp[0], wp[1], wp[2], float(sb.x), float(sb.y), dw0, dw1);
                    } else {
                        const uint w_off = g * 16u + lane_in_grp * 2u;
                        dequant8_q2(*(const device uint16_t *)(down_qw + row_w_base + w_off), float(sb.x), float(sb.y), dw0, dw1);
                    }

                    const uint k_off = g * Q_GROUP_SIZE + lane_in_grp * 8u;
                    a0 += dot(float4(*(const device half4 *)(i0 + k_off)), dw0) + dot(float4(*(const device half4 *)(i0 + k_off + 4u)), dw1);
                    if (rem > 1u) a1 += dot(float4(*(const device half4 *)(i1 + k_off)), dw0) + dot(float4(*(const device half4 *)(i1 + k_off + 4u)), dw1);
                    if (rem > 2u) a2 += dot(float4(*(const device half4 *)(i2 + k_off)), dw0) + dot(float4(*(const device half4 *)(i2 + k_off + 4u)), dw1);
                    if (rem > 3u) a3 += dot(float4(*(const device half4 *)(i3 + k_off)), dw0) + dot(float4(*(const device half4 *)(i3 + k_off + 4u)), dw1);
                }

                const float sa0 = simd_sum(a0), sa1 = simd_sum(a1), sa2 = simd_sum(a2), sa3 = simd_sum(a3);
                if (lane == 0u) {
                    { const uint s = s0 + t_base + 0u; down_out[(ulong(token_indices[s]) * p.top_k + slot_k_indices[s]) * p.hidden_size + row] = sa0 * routing_weights[s]; }
                    if (rem > 1u) { const uint s = s0 + t_base + 1u; down_out[(ulong(token_indices[s]) * p.top_k + slot_k_indices[s]) * p.hidden_size + row] = sa1 * routing_weights[s]; }
                    if (rem > 2u) { const uint s = s0 + t_base + 2u; down_out[(ulong(token_indices[s]) * p.top_k + slot_k_indices[s]) * p.hidden_size + row] = sa2 * routing_weights[s]; }
                    if (rem > 3u) { const uint s = s0 + t_base + 3u; down_out[(ulong(token_indices[s]) * p.top_k + slot_k_indices[s]) * p.hidden_size + row] = sa3 * routing_weights[s]; }
                }
            }
            t_base += rem;
        } else {
            const uint t_cnt = min(8u, rem);
            const device half *i0 = inter + ulong(s0 + t_base + 0u) * p.intermediate_size;
            const device half *i1 = inter + ulong(s0 + t_base + 1u) * p.intermediate_size;
            const device half *i2 = inter + ulong(s0 + t_base + 2u) * p.intermediate_size;
            const device half *i3 = inter + ulong(s0 + t_base + 3u) * p.intermediate_size;
            const device half *i4 = inter + ulong(s0 + t_base + 4u) * p.intermediate_size;
            const device half *i5 = inter + ulong(s0 + t_base + min(5u, t_cnt - 1u)) * p.intermediate_size;
            const device half *i6 = inter + ulong(s0 + t_base + min(6u, t_cnt - 1u)) * p.intermediate_size;
            const device half *i7 = inter + ulong(s0 + t_base + min(7u, t_cnt - 1u)) * p.intermediate_size;

            for (uint r = 0u; r < ROWS_PER_SG; r++) {
                const uint row = row0 + r;
                if (row >= p.hidden_size) {
                    break;
                }
                float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
                float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
                const ulong row_w_base = ulong(row) * p.row_bytes;
                const ulong row_sb_base = ulong(row) * p.num_groups;

                for (uint g_base = 0u; g_base < p.num_groups; g_base += 4u) {
                    const uint g = g_base + sub_group;
                    if (g >= p.num_groups) {
                        continue;
                    }
                    const half2 sb = down_sb[row_sb_base + g];
                    float4 dw0, dw1;
                    if (Q_BITS == 4u) {
                        const uint w_off = g * 32u + lane_in_grp * 4u;
                        dequant8_q4(*(const device uint32_t *)(down_qw + row_w_base + w_off), float(sb.x), float(sb.y), dw0, dw1);
                    } else if (Q_BITS == 3u) {
                        const uint w_off = g * 24u + lane_in_grp * 3u;
                        const device uchar *wp = down_qw + row_w_base + w_off;
                        dequant8_q3(wp[0], wp[1], wp[2], float(sb.x), float(sb.y), dw0, dw1);
                    } else {
                        const uint w_off = g * 16u + lane_in_grp * 2u;
                        dequant8_q2(*(const device uint16_t *)(down_qw + row_w_base + w_off), float(sb.x), float(sb.y), dw0, dw1);
                    }

                    const uint k_off = g * Q_GROUP_SIZE + lane_in_grp * 8u;
                    a0 += dot(float4(*(const device half4 *)(i0 + k_off)), dw0) + dot(float4(*(const device half4 *)(i0 + k_off + 4u)), dw1);
                    a1 += dot(float4(*(const device half4 *)(i1 + k_off)), dw0) + dot(float4(*(const device half4 *)(i1 + k_off + 4u)), dw1);
                    a2 += dot(float4(*(const device half4 *)(i2 + k_off)), dw0) + dot(float4(*(const device half4 *)(i2 + k_off + 4u)), dw1);
                    a3 += dot(float4(*(const device half4 *)(i3 + k_off)), dw0) + dot(float4(*(const device half4 *)(i3 + k_off + 4u)), dw1);
                    a4 += dot(float4(*(const device half4 *)(i4 + k_off)), dw0) + dot(float4(*(const device half4 *)(i4 + k_off + 4u)), dw1);
                    if (t_cnt > 5u) a5 += dot(float4(*(const device half4 *)(i5 + k_off)), dw0) + dot(float4(*(const device half4 *)(i5 + k_off + 4u)), dw1);
                    if (t_cnt > 6u) a6 += dot(float4(*(const device half4 *)(i6 + k_off)), dw0) + dot(float4(*(const device half4 *)(i6 + k_off + 4u)), dw1);
                    if (t_cnt > 7u) a7 += dot(float4(*(const device half4 *)(i7 + k_off)), dw0) + dot(float4(*(const device half4 *)(i7 + k_off + 4u)), dw1);
                }

                const float sa0 = simd_sum(a0), sa1 = simd_sum(a1), sa2 = simd_sum(a2), sa3 = simd_sum(a3);
                const float sa4 = simd_sum(a4), sa5 = simd_sum(a5), sa6 = simd_sum(a6), sa7 = simd_sum(a7);
                if (lane == 0u) {
                    { const uint s = s0 + t_base + 0u; down_out[(ulong(token_indices[s]) * p.top_k + slot_k_indices[s]) * p.hidden_size + row] = sa0 * routing_weights[s]; }
                    { const uint s = s0 + t_base + 1u; down_out[(ulong(token_indices[s]) * p.top_k + slot_k_indices[s]) * p.hidden_size + row] = sa1 * routing_weights[s]; }
                    { const uint s = s0 + t_base + 2u; down_out[(ulong(token_indices[s]) * p.top_k + slot_k_indices[s]) * p.hidden_size + row] = sa2 * routing_weights[s]; }
                    { const uint s = s0 + t_base + 3u; down_out[(ulong(token_indices[s]) * p.top_k + slot_k_indices[s]) * p.hidden_size + row] = sa3 * routing_weights[s]; }
                    { const uint s = s0 + t_base + 4u; down_out[(ulong(token_indices[s]) * p.top_k + slot_k_indices[s]) * p.hidden_size + row] = sa4 * routing_weights[s]; }
                    if (t_cnt > 5u) { const uint s = s0 + t_base + 5u; down_out[(ulong(token_indices[s]) * p.top_k + slot_k_indices[s]) * p.hidden_size + row] = sa5 * routing_weights[s]; }
                    if (t_cnt > 6u) { const uint s = s0 + t_base + 6u; down_out[(ulong(token_indices[s]) * p.top_k + slot_k_indices[s]) * p.hidden_size + row] = sa6 * routing_weights[s]; }
                    if (t_cnt > 7u) { const uint s = s0 + t_base + 7u; down_out[(ulong(token_indices[s]) * p.top_k + slot_k_indices[s]) * p.hidden_size + row] = sa7 * routing_weights[s]; }
                }
            }
            t_base += t_cnt;
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 5: Deterministic Top-K Reduction + Optional Fused Residual Add
// ---------------------------------------------------------------------------
kernel void ami_moe_reduce_f16(const device float *down_out [[buffer(0)]],
                               const device half *residual [[buffer(1)]],
                               device half *out [[buffer(2)]],
                               constant MoeReduceParams &p [[buffer(3)]],
                               uint gid [[thread_position_in_grid]]) {
    const uint total = p.m * p.hidden_size;
    if (gid >= total) {
        return;
    }

    const uint tok = gid / p.hidden_size;
    const uint col = gid - tok * p.hidden_size;
    const ulong base = ulong(tok) * p.top_k * p.hidden_size + col;
    float acc = (p.add_residual != 0u) ? float(residual[gid]) : 0.0f;

    if (MOE_TOP_K > 0u) {
        for (uint k = 0u; k < MOE_TOP_K; k++) {
            acc += down_out[base + ulong(k) * p.hidden_size];
        }
    } else {
        for (uint k = 0u; k < p.top_k; k++) {
            acc += down_out[base + ulong(k) * p.hidden_size];
        }
    }

    out[gid] = half(acc);
}

// ---------------------------------------------------------------------------
// Kernel 6: Dense SwiGLU Activation (`out[i] = half(silu(gate[i]) * up[i])`)
// ---------------------------------------------------------------------------
kernel void ami_silu_mul_f16(const device half *gate [[buffer(0)]],
                             const device half *up [[buffer(1)]],
                             device half *out [[buffer(2)]],
                             constant ElemParams &p [[buffer(3)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= p.count) {
        return;
    }
    const float g = float(gate[gid]);
    const float u = float(up[gid]);
    out[gid] = half(fast_silu(g) * u);
}

// ---------------------------------------------------------------------------
// Kernel 7: In-Place Residual Add (`x[i] = half(float(x[i]) + float(delta[i]))`)
// ---------------------------------------------------------------------------
kernel void ami_residual_add_f16(device half *x [[buffer(0)]],
                                 const device half *delta [[buffer(1)]],
                                 constant ElemParams &p [[buffer(2)]],
                                 uint gid [[thread_position_in_grid]]) {
    if (gid >= p.count) {
        return;
    }
    x[gid] = half(float(x[gid]) + float(delta[gid]));
}

// ---------------------------------------------------------------------------
// Kernel 8: Parallel Multi-Simdgroup Greedy Argmax over fp16 Logits
//
// Single threadgroup (up to 1024 threads = 32 simdgroups) with vectorized
// half4 loads when 8-byte aligned and deterministic smallest-index tie
// breaking. Writes a single uint32_t token ID directly into wired shared RAM.
// ---------------------------------------------------------------------------
kernel void ami_argmax_f16(const device half *logits [[buffer(0)]],
                           device uint *out_token_id [[buffer(1)]],
                           constant ArgmaxParams &p [[buffer(2)]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint tg_size [[threads_per_threadgroup]],
                           uint sg_id [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float s_vals[32];
    threadgroup uint s_idxs[32];

    float best_v = -INFINITY;
    uint best_i = 0xFFFFFFFFu;

    if (p.aligned_half4 != 0u) {
        const uint vec_n = p.vocab_size >> 2;
        const device half4 *vlogits = (const device half4 *)logits;
        for (uint vi = tid; vi < vec_n; vi += tg_size) {
            const float4 fv = float4(vlogits[vi]);
            const uint base = vi << 2;
            if (fv.x > best_v || (fv.x == best_v && (base + 0u) < best_i)) {
                best_v = fv.x;
                best_i = base + 0u;
            }
            if (fv.y > best_v || (fv.y == best_v && (base + 1u) < best_i)) {
                best_v = fv.y;
                best_i = base + 1u;
            }
            if (fv.z > best_v || (fv.z == best_v && (base + 2u) < best_i)) {
                best_v = fv.z;
                best_i = base + 2u;
            }
            if (fv.w > best_v || (fv.w == best_v && (base + 3u) < best_i)) {
                best_v = fv.w;
                best_i = base + 3u;
            }
        }
        const uint tail = vec_n << 2;
        for (uint i = tail + tid; i < p.vocab_size; i += tg_size) {
            const float v = float(logits[i]);
            if (v > best_v || (v == best_v && i < best_i)) {
                best_v = v;
                best_i = i;
            }
        }
    } else {
        for (uint i = tid; i < p.vocab_size; i += tg_size) {
            const float v = float(logits[i]);
            if (v > best_v || (v == best_v && i < best_i)) {
                best_v = v;
                best_i = i;
            }
        }
    }

    for (ushort offset = 16u; offset > 0u; offset >>= 1u) {
        const float ov = simd_shuffle_down(best_v, offset);
        const uint oi = simd_shuffle_down(best_i, offset);
        if (ov > best_v || (ov == best_v && oi < best_i)) {
            best_v = ov;
            best_i = oi;
        }
    }

    if (lane == 0u) {
        s_vals[sg_id] = best_v;
        s_idxs[sg_id] = best_i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_id == 0u) {
        const uint num_sgs = tg_size >> 5;
        float fv = (lane < num_sgs) ? s_vals[lane] : -INFINITY;
        uint fi = (lane < num_sgs) ? s_idxs[lane] : 0xFFFFFFFFu;
        for (ushort offset = 16u; offset > 0u; offset >>= 1u) {
            const float ov = simd_shuffle_down(fv, offset);
            const uint oi = simd_shuffle_down(fi, offset);
            if (ov > fv || (ov == fv && oi < fi)) {
                fv = ov;
                fi = oi;
            }
        }
        if (lane == 0u) {
            out_token_id[0] = (fi == 0xFFFFFFFFu) ? 0u : fi;
        }
    }
}

// ===========================================================================
// M2.5 (G12, ADR-0016): quantized router, native FP16 / BF16 deduplicated MoE
// SwiGLU, and GPU embedding lookup. All reductions accumulate in FP32.
// ===========================================================================

struct RouterQParams {
    uint m;
    uint num_experts;
    uint hidden_size;
    uint num_groups;
    uint row_bytes;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

struct EmbedParams {
    uint m;
    uint vocab_size;
    uint hidden_size;
    uint row_bytes;
    uint num_groups;
    uint use_token_buffer;
    uint token_id;
    uint _pad;
};

// ---------------------------------------------------------------------------
// Kernel 9: MLX affine 3-bit / 4-bit group-64 Router Projection
// (`[M, hidden] @ dequant(router_qw)^T -> [M, E] f32`), e.g. the real
// Qwen3-30B-A3B-3bit `mlp.gate` (`q3_affine [128, 2048]`).
//
// Geometry: 4 simdgroups/threadgroup (128 threads), 4 experts/simdgroup
// (16 experts/threadgroup), grid (ceil(E / 16), M).
// ---------------------------------------------------------------------------
kernel void ami_moe_router_proj_q(const device half *x [[buffer(0)]],
                                  const device uchar *qweight [[buffer(1)]],
                                  const device half2 *scale_bias [[buffer(2)]],
                                  device float *logits [[buffer(3)]],
                                  constant RouterQParams &p [[buffer(4)]],
                                  uint2 tg_pos [[threadgroup_position_in_grid]],
                                  uint sg_id [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint ROWS_PER_SG = 4u;
    const uint tok = tg_pos.y;
    const uint row0 = tg_pos.x * 16u + sg_id * ROWS_PER_SG;
    if (tok >= p.m || row0 >= p.num_experts) {
        return;
    }

    const device half *x_tok = x + ulong(tok) * p.hidden_size;
    const uint sub_group = lane >> 3;
    const uint lane_in_grp = lane & 7u;
    float acc[ROWS_PER_SG] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint g_base = 0u; g_base < p.num_groups; g_base += 4u) {
        const uint g = g_base + sub_group;
        if (g >= p.num_groups) {
            continue;
        }
        const uint k_off = g * Q_GROUP_SIZE + lane_in_grp * 8u;
        const float4 fx0 = float4(*(const device half4 *)(x_tok + k_off));
        const float4 fx1 = float4(*(const device half4 *)(x_tok + k_off + 4u));
        const float x_sum = (fx0.x + fx0.y) + (fx0.z + fx0.w) + (fx1.x + fx1.y) + (fx1.z + fx1.w);

        if (Q_BITS == 4u) {
            const float4 px0 = float4(fx0.x, fx0.y * (1.0f / 16.0f), fx0.z * (1.0f / 256.0f), fx0.w * (1.0f / 4096.0f));
            const float4 px1 = float4(fx1.x, fx1.y * (1.0f / 16.0f), fx1.z * (1.0f / 256.0f), fx1.w * (1.0f / 4096.0f));
            const uint w_off = g * 32u + lane_in_grp * 4u;
            for (uint r = 0u; r < ROWS_PER_SG; r++) {
                const uint row = min(row0 + r, p.num_experts - 1u);
                const uint32_t w = *(const device uint32_t *)(qweight + ulong(row) * p.row_bytes + w_off);
                const half2 sb = scale_bias[ulong(row) * p.num_groups + g];
                acc[r] = fma(qdot8_q4(w, px0, px1), float(sb.x), fma(x_sum, float(sb.y), acc[r]));
            }
        } else if (Q_BITS == 3u) {
            const float4 px0 = float4(fx0.x, fx0.y * (1.0f / 8.0f), fx0.z * (1.0f / 64.0f), fx0.w * (1.0f / 2.0f));
            const float4 px1 = float4(fx1.x * (1.0f / 16.0f), fx1.y * (1.0f / 128.0f), fx1.z * (1.0f / 4.0f), fx1.w * (1.0f / 32.0f));
            const float x2_hi = fx0.z * 4.0f;
            const float x5_hi = fx1.y * 2.0f;
            const uint w_off = g * 24u + lane_in_grp * 3u;
            for (uint r = 0u; r < ROWS_PER_SG; r++) {
                const uint row = min(row0 + r, p.num_experts - 1u);
                const device uchar *wp = qweight + ulong(row) * p.row_bytes + w_off;
                const half2 sb = scale_bias[ulong(row) * p.num_groups + g];
                acc[r] = fma(qdot8_q3(wp[0], wp[1], wp[2], px0, px1, x2_hi, x5_hi),
                             float(sb.x), fma(x_sum, float(sb.y), acc[r]));
            }
        } else {
            const float4 px0 = float4(fx0.x, fx0.y * (1.0f / 4.0f), fx0.z * (1.0f / 16.0f), fx0.w * (1.0f / 64.0f));
            const float4 px1 = float4(fx1.x, fx1.y * (1.0f / 4.0f), fx1.z * (1.0f / 16.0f), fx1.w * (1.0f / 64.0f));
            const uint w_off = g * 16u + lane_in_grp * 2u;
            for (uint r = 0u; r < ROWS_PER_SG; r++) {
                const uint row = min(row0 + r, p.num_experts - 1u);
                const uint16_t w = *(const device uint16_t *)(qweight + ulong(row) * p.row_bytes + w_off);
                const half2 sb = scale_bias[ulong(row) * p.num_groups + g];
                acc[r] = fma(qdot8_q2(w, px0, px1), float(sb.x), fma(x_sum, float(sb.y), acc[r]));
            }
        }
    }

    for (uint r = 0u; r < ROWS_PER_SG; r++) {
        const float total = simd_sum(acc[r]);
        if (lane == 0u && (row0 + r) < p.num_experts) {
            logits[ulong(tok) * p.num_experts + row0 + r] = total;
        }
    }
}

static inline float4 moe_load_w4(const device uchar *w, ulong elem) {
    if (MOE_W_BF16) {
        return float4(*(const device bfloat4 *)(w + elem * 2ul));
    }
    return float4(*(const device half4 *)(w + elem * 2ul));
}

// ---------------------------------------------------------------------------
// Kernel 10: Deduplicated MoE FP16 / BF16 Gate + Up + SwiGLU
//
// Geometry: 4 simdgroups/threadgroup (128 threads), 4 output rows/simdgroup
// (16 rows/threadgroup); tg_pos.y indexes active unique expert `u`. Each
// active expert streams its gate/up rows with coalesced `half4` / `bfloat4`
// loads ONCE per block of up to 4 assigned tokens (weight-stationary register
// batching: 1-token slots read weights exactly once; multi-token slots reuse
// each weight vector across the token block). Requires hidden_size % 4 == 0.
// Expert `ExpertWeights.*_qweight_offset` entries are byte offsets of
// row-major `[N, K]` 16-bit matrices (`*_sb_offset` unused).
// ---------------------------------------------------------------------------
kernel void ami_moe_gate_up_swiglu_float(const device half *x [[buffer(0)]],
                                         const device uchar *weights_base [[buffer(1)]],
                                         const device ExpertWeights *expert_table [[buffer(2)]],
                                         const device MoeScheduleHeader *sched_hdr [[buffer(3)]],
                                         const device uint *unique_experts [[buffer(4)]],
                                         const device uint *expert_token_offsets [[buffer(5)]],
                                         const device uint *token_indices [[buffer(6)]],
                                         device half *inter [[buffer(7)]],
                                         constant MoeGateUpParams &p [[buffer(8)]],
                                         uint2 tg_pos [[threadgroup_position_in_grid]],
                                         uint sg_id [[simdgroup_index_in_threadgroup]],
                                         uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint RPS = 4u;
    constexpr uint TB = 4u;
    const uint u = tg_pos.y;
    if (u >= sched_hdr->num_unique_experts) {
        return;
    }
    const uint row0 = tg_pos.x * 16u + sg_id * RPS;
    if (row0 >= p.intermediate_size) {
        return;
    }
    const uint s0 = expert_token_offsets[u];
    const uint count_u = expert_token_offsets[u + 1u] - s0;
    if (count_u == 0u) {
        return;
    }

    const ExpertWeights ew = expert_table[unique_experts[u]];
    const device uchar *gate_w = weights_base + ew.gate_qweight_offset;
    const device uchar *up_w = weights_base + ew.up_qweight_offset;

    ulong row_base[RPS];
    for (uint r = 0u; r < RPS; r++) {
        row_base[r] = ulong(min(row0 + r, p.intermediate_size - 1u)) * p.hidden_size;
    }
    const uint vec_k = p.hidden_size >> 2;

    for (uint t_base = 0u; t_base < count_u; t_base += TB) {
        const uint t_cnt = min(TB, count_u - t_base);
        const device half4 *xt[TB];
        for (uint t = 0u; t < TB; t++) {
            const uint tok = token_indices[s0 + t_base + min(t, t_cnt - 1u)];
            xt[t] = (const device half4 *)(x + ulong(tok) * p.hidden_size);
        }

        float g_acc[RPS][TB];
        float u_acc[RPS][TB];
        for (uint r = 0u; r < RPS; r++) {
            for (uint t = 0u; t < TB; t++) {
                g_acc[r][t] = 0.0f;
                u_acc[r][t] = 0.0f;
            }
        }

        for (uint i = lane; i < vec_k; i += 32u) {
            float4 xv[TB];
            for (uint t = 0u; t < TB; t++) {
                xv[t] = float4(xt[t][i]);
            }
            for (uint r = 0u; r < RPS; r++) {
                const float4 gw = moe_load_w4(gate_w, row_base[r] + ulong(i) * 4ul);
                const float4 uw = moe_load_w4(up_w, row_base[r] + ulong(i) * 4ul);
                for (uint t = 0u; t < TB; t++) {
                    g_acc[r][t] += dot(xv[t], gw);
                    u_acc[r][t] += dot(xv[t], uw);
                }
            }
        }

        for (uint r = 0u; r < RPS; r++) {
            for (uint t = 0u; t < TB; t++) {
                const float gt = simd_sum(g_acc[r][t]);
                const float ut = simd_sum(u_acc[r][t]);
                if (lane == 0u && t < t_cnt && (row0 + r) < p.intermediate_size) {
                    const ulong slot = ulong(s0 + t_base + t);
                    inter[slot * p.intermediate_size + row0 + r] = half(fast_silu(gt) * ut);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 11: Deduplicated MoE FP16 / BF16 Down Projection * Routing Weight
//
// Geometry: 4 simdgroups/threadgroup, 4 output rows/simdgroup (16 rows per
// threadgroup), weight-stationary token blocks of 4. Writes fp32 partials to
// down_out[(tok * top_k + k_idx) * hidden_size + row] with zero atomics.
// Requires intermediate_size % 4 == 0.
// ---------------------------------------------------------------------------
kernel void ami_moe_down_proj_float(const device half *inter [[buffer(0)]],
                                    const device uchar *weights_base [[buffer(1)]],
                                    const device ExpertWeights *expert_table [[buffer(2)]],
                                    const device MoeScheduleHeader *sched_hdr [[buffer(3)]],
                                    const device uint *unique_experts [[buffer(4)]],
                                    const device uint *expert_token_offsets [[buffer(5)]],
                                    const device uint *token_indices [[buffer(6)]],
                                    const device float *routing_weights [[buffer(7)]],
                                    const device uint *slot_k_indices [[buffer(8)]],
                                    device float *down_out [[buffer(9)]],
                                    constant MoeDownParams &p [[buffer(10)]],
                                    uint2 tg_pos [[threadgroup_position_in_grid]],
                                    uint sg_id [[simdgroup_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint RPS = 4u;
    constexpr uint TB = 4u;
    const uint u = tg_pos.y;
    if (u >= sched_hdr->num_unique_experts) {
        return;
    }
    const uint row0 = tg_pos.x * 16u + sg_id * RPS;
    if (row0 >= p.hidden_size) {
        return;
    }
    const uint s0 = expert_token_offsets[u];
    const uint count_u = expert_token_offsets[u + 1u] - s0;
    if (count_u == 0u) {
        return;
    }

    const ExpertWeights ew = expert_table[unique_experts[u]];
    const device uchar *down_w = weights_base + ew.down_qweight_offset;

    ulong row_base[RPS];
    for (uint r = 0u; r < RPS; r++) {
        row_base[r] = ulong(min(row0 + r, p.hidden_size - 1u)) * p.intermediate_size;
    }
    const uint vec_k = p.intermediate_size >> 2;

    for (uint t_base = 0u; t_base < count_u; t_base += TB) {
        const uint t_cnt = min(TB, count_u - t_base);
        const device half4 *it[TB];
        for (uint t = 0u; t < TB; t++) {
            const uint slot = s0 + t_base + min(t, t_cnt - 1u);
            it[t] = (const device half4 *)(inter + ulong(slot) * p.intermediate_size);
        }

        float acc[RPS][TB];
        for (uint r = 0u; r < RPS; r++) {
            for (uint t = 0u; t < TB; t++) {
                acc[r][t] = 0.0f;
            }
        }

        for (uint i = lane; i < vec_k; i += 32u) {
            float4 iv[TB];
            for (uint t = 0u; t < TB; t++) {
                iv[t] = float4(it[t][i]);
            }
            for (uint r = 0u; r < RPS; r++) {
                const float4 dw = moe_load_w4(down_w, row_base[r] + ulong(i) * 4ul);
                for (uint t = 0u; t < TB; t++) {
                    acc[r][t] += dot(iv[t], dw);
                }
            }
        }

        for (uint t = 0u; t < TB; t++) {
            const uint s = s0 + t_base + min(t, t_cnt - 1u);
            const uint tok = token_indices[s];
            const uint k_idx = slot_k_indices[s];
            const float rw = routing_weights[s];
            const ulong out_base = (ulong(tok) * p.top_k + k_idx) * p.hidden_size;
            for (uint r = 0u; r < RPS; r++) {
                const float total = simd_sum(acc[r][t]);
                if (lane == 0u && t < t_cnt && (row0 + r) < p.hidden_size) {
                    down_out[out_base + row0 + r] = total * rw;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 12: GPU Embedding Lookup (`out[m, :] = embed[token_id[m], :]`)
//
// Reads one `[vocab_size, hidden_size]` table row per token in f16, bf16,
// q4_affine or q3_affine (EMBED_KIND) and writes `half [M, hidden_size]`
// directly into the residual stream (zero CPU copy; the same table can back a
// tied `lm_head`). Token IDs come from a uniform (single-token decode) or a
// `uint [M]` buffer. Out-of-range IDs produce zero rows instead of faulting.
// Each thread produces 8 consecutive outputs; grid (ceil(hidden / 8), M).
// ---------------------------------------------------------------------------
kernel void ami_embed_lookup(const device uchar *weight [[buffer(0)]],
                             const device half2 *scale_bias [[buffer(1)]],
                             const device uint *token_ids [[buffer(2)]],
                             device half *out [[buffer(3)]],
                             constant EmbedParams &p [[buffer(4)]],
                             uint2 gid [[thread_position_in_grid]]) {
    const uint row = gid.y;
    const uint col0 = gid.x * 8u;
    if (row >= p.m || col0 >= p.hidden_size) {
        return;
    }
    const uint tok = (p.use_token_buffer != 0u) ? token_ids[row] : p.token_id;
    device half *dst = out + ulong(row) * p.hidden_size + col0;
    const uint n = min(8u, p.hidden_size - col0);

    if (tok >= p.vocab_size) {
        for (uint j = 0u; j < n; j++) {
            dst[j] = half(0.0h);
        }
        return;
    }

    if (EMBED_KIND == 0u) {
        const device half *src = (const device half *)weight + ulong(tok) * p.hidden_size + col0;
        for (uint j = 0u; j < n; j++) {
            dst[j] = src[j];
        }
    } else if (EMBED_KIND == 1u) {
        // bf16 -> f16: saturate instead of overflowing to infinity.
        const device bfloat *src = (const device bfloat *)weight + ulong(tok) * p.hidden_size + col0;
        for (uint j = 0u; j < n; j++) {
            dst[j] = half(clamp(float(src[j]), -65504.0f, 65504.0f));
        }
    } else {
        const uint g = col0 / Q_GROUP_SIZE;
        const uint lane_in_grp = (col0 % Q_GROUP_SIZE) >> 3;
        const half2 sb = scale_bias[ulong(tok) * p.num_groups + g];
        const float scale = float(sb.x);
        const float bias = float(sb.y);
        const device uchar *row_w = weight + ulong(tok) * p.row_bytes;
        if (EMBED_KIND == 2u) {
            const uint32_t w = *(const device uint32_t *)(row_w + g * 32u + lane_in_grp * 4u);
            for (uint j = 0u; j < 8u; j++) {
                dst[j] = half(fma(float((w >> (j * 4u)) & 0xFu), scale, bias));
            }
        } else if (EMBED_KIND == 3u) {
            const device uchar *wp = row_w + g * 24u + lane_in_grp * 3u;
            const uint word = uint(wp[0]) | (uint(wp[1]) << 8) | (uint(wp[2]) << 16);
            for (uint j = 0u; j < 8u; j++) {
                dst[j] = half(fma(float((word >> (j * 3u)) & 0x7u), scale, bias));
            }
        } else {
            const uint16_t w = *(const device uint16_t *)(row_w + g * 16u + lane_in_grp * 2u);
            const uint u = uint(w);
            for (uint j = 0u; j < 8u; j++) {
                dst[j] = half(fma(float((u >> (j * 2u)) & 0x3u), scale, bias));
            }
        }
    }
}

// ===========================================================================
// Milestone M3: Batched Argmax Rows, Fused Top-K Draft Logits Extraction,
// and Non-Deduplicated Per-Slot MoE SwiGLU Kernels (Spike S5 / ADR-0009).
// ===========================================================================

struct ArgmaxRowsParams {
    uint num_rows;
    uint vocab_size;
    uint aligned_vec4;
    uint _pad;
};

struct TopKCandidate {
    uint token_id;
    float prob;
    float log_prob;
    uint _pad;
};

struct TopKLogitsRowsParams {
    uint num_rows;
    uint vocab_size;
    uint top_k;
    uint aligned_vec4;
};

struct MoePerSlotParams {
    uint hidden_size;
    uint intermediate_size;
    uint num_groups;
    uint row_bytes;
    uint m;
    uint top_k;
};

static_assert(sizeof(ArgmaxRowsParams) == 16, "ArgmaxRowsParams must be 16 bytes");
static_assert(sizeof(TopKCandidate) == 16, "TopKCandidate must be 16 bytes");
static_assert(sizeof(TopKLogitsRowsParams) == 16, "TopKLogitsRowsParams must be 16 bytes");
static_assert(sizeof(MoePerSlotParams) == 24, "MoePerSlotParams must be 24 bytes");

// ---------------------------------------------------------------------------
// Kernel 13: Batched M-Row Greedy Argmax (`[M, vocab_size]` f16/f32 -> `[M]u32`)
//
// Specialized on LOGITS_F16 (function_constant(3)). Deterministic smallest-
// index tie breaking. Grid: (num_rows, 1, 1) threadgroups, up to 1024 threads.
// ---------------------------------------------------------------------------
kernel void ami_argmax_rows(const device uchar *logits_raw [[buffer(0)]],
                            device uint *out_token_ids [[buffer(1)]],
                            constant ArgmaxRowsParams &p [[buffer(2)]],
                            uint row [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint tg_size [[threads_per_threadgroup]],
                            uint sg_id [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
    if (row >= p.num_rows) {
        return;
    }

    threadgroup float s_vals[32];
    threadgroup uint s_idxs[32];

    float best_v = -INFINITY;
    uint best_i = 0xFFFFFFFFu;

    if (LOGITS_F16) {
        const device half *logits = ((const device half *)logits_raw) + ulong(row) * p.vocab_size;
        if (p.aligned_vec4 != 0u) {
            const uint vec_n = p.vocab_size >> 2;
            const device half4 *vlogits = (const device half4 *)logits;
            for (uint vi = tid; vi < vec_n; vi += tg_size) {
                const float4 fv = float4(vlogits[vi]);
                const uint base = vi << 2;
                if (fv.x > best_v || (fv.x == best_v && (base + 0u) < best_i)) { best_v = fv.x; best_i = base + 0u; }
                if (fv.y > best_v || (fv.y == best_v && (base + 1u) < best_i)) { best_v = fv.y; best_i = base + 1u; }
                if (fv.z > best_v || (fv.z == best_v && (base + 2u) < best_i)) { best_v = fv.z; best_i = base + 2u; }
                if (fv.w > best_v || (fv.w == best_v && (base + 3u) < best_i)) { best_v = fv.w; best_i = base + 3u; }
            }
            const uint tail = vec_n << 2;
            for (uint i = tail + tid; i < p.vocab_size; i += tg_size) {
                const float v = float(logits[i]);
                if (v > best_v || (v == best_v && i < best_i)) { best_v = v; best_i = i; }
            }
        } else {
            for (uint i = tid; i < p.vocab_size; i += tg_size) {
                const float v = float(logits[i]);
                if (v > best_v || (v == best_v && i < best_i)) { best_v = v; best_i = i; }
            }
        }
    } else {
        const device float *logits = ((const device float *)logits_raw) + ulong(row) * p.vocab_size;
        if (p.aligned_vec4 != 0u) {
            const uint vec_n = p.vocab_size >> 2;
            const device float4 *vlogits = (const device float4 *)logits;
            for (uint vi = tid; vi < vec_n; vi += tg_size) {
                const float4 fv = vlogits[vi];
                const uint base = vi << 2;
                if (fv.x > best_v || (fv.x == best_v && (base + 0u) < best_i)) { best_v = fv.x; best_i = base + 0u; }
                if (fv.y > best_v || (fv.y == best_v && (base + 1u) < best_i)) { best_v = fv.y; best_i = base + 1u; }
                if (fv.z > best_v || (fv.z == best_v && (base + 2u) < best_i)) { best_v = fv.z; best_i = base + 2u; }
                if (fv.w > best_v || (fv.w == best_v && (base + 3u) < best_i)) { best_v = fv.w; best_i = base + 3u; }
            }
            const uint tail = vec_n << 2;
            for (uint i = tail + tid; i < p.vocab_size; i += tg_size) {
                const float v = logits[i];
                if (v > best_v || (v == best_v && i < best_i)) { best_v = v; best_i = i; }
            }
        } else {
            for (uint i = tid; i < p.vocab_size; i += tg_size) {
                const float v = logits[i];
                if (v > best_v || (v == best_v && i < best_i)) { best_v = v; best_i = i; }
            }
        }
    }

    for (ushort offset = 16u; offset > 0u; offset >>= 1u) {
        const float ov = simd_shuffle_down(best_v, offset);
        const uint oi = simd_shuffle_down(best_i, offset);
        if (ov > best_v || (ov == best_v && oi < best_i)) {
            best_v = ov;
            best_i = oi;
        }
    }

    if (lane == 0u) {
        s_vals[sg_id] = best_v;
        s_idxs[sg_id] = best_i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_id == 0u) {
        const uint num_sgs = tg_size >> 5;
        float fv = (lane < num_sgs) ? s_vals[lane] : -INFINITY;
        uint fi = (lane < num_sgs) ? s_idxs[lane] : 0xFFFFFFFFu;
        for (ushort offset = 16u; offset > 0u; offset >>= 1u) {
            const float ov = simd_shuffle_down(fv, offset);
            const uint oi = simd_shuffle_down(fi, offset);
            if (ov > fv || (ov == fv && oi < fi)) {
                fv = ov;
                fi = oi;
            }
        }
        if (lane == 0u) {
            out_token_ids[row] = (fi == 0xFFFFFFFFu) ? 0u : fi;
        }
    }
}

static inline void topk_consider(float v,
                                 uint idx,
                                 uint k,
                                 thread float *top_v,
                                 thread uint *top_i,
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
            const float pv = top_v[pos - 1u];
            const uint pi = top_i[pos - 1u];
            if (v > pv || (v == pv && idx < pi)) {
                top_v[pos] = pv;
                top_i[pos] = pi;
                pos--;
            } else {
                break;
            }
        }
        top_v[pos] = v;
        top_i[pos] = idx;
        min_v = top_v[k - 1u];
        min_i = top_i[k - 1u];
    }
}

// ---------------------------------------------------------------------------
// Kernel 14: Fused Single-Dispatch Row-Wise FP32 Softmax + Top-K Extraction
//
// `[M, vocab_size]` (f16 or f32 via LOGITS_F16) -> `[M, top_k]` TopKCandidate
// (`top_k <= 16`). One 256-thread (8-simdgroup) threadgroup per row performs a
// single streaming pass computing exact FP32 online `(max, sum_exp)` and
// per-thread top-K, then merges exact top-K candidates with deterministic
// smallest-index tie breaking.
// ---------------------------------------------------------------------------
kernel void ami_topk_logits_rows(const device uchar *logits_raw [[buffer(0)]],
                                 device TopKCandidate *out_cands [[buffer(1)]],
                                 constant TopKLogitsRowsParams &p [[buffer(2)]],
                                 uint row [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]],
                                 uint sg_id [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
    if (row >= p.num_rows) {
        return;
    }

    const uint k = min(p.top_k, 16u);
    float top_v[16];
    uint top_i[16];
    for (uint j = 0u; j < k; j++) {
        top_v[j] = -INFINITY;
        top_i[j] = 0xFFFFFFFFu;
    }
    float min_v = -INFINITY;
    uint min_i = 0xFFFFFFFFu;
    float local_m = -INFINITY;
    float local_sum = 0.0f;

    if (LOGITS_F16) {
        const device half *logits = ((const device half *)logits_raw) + ulong(row) * p.vocab_size;
        if (p.aligned_vec4 != 0u) {
            const uint vec_n = p.vocab_size >> 2;
            const device half4 *vlogits = (const device half4 *)logits;
            for (uint vi = tid; vi < vec_n; vi += 256u) {
                const float4 fv = float4(vlogits[vi]);
                const uint base = vi << 2;
                topk_consider(fv.x, base + 0u, k, top_v, top_i, min_v, min_i, local_m, local_sum);
                topk_consider(fv.y, base + 1u, k, top_v, top_i, min_v, min_i, local_m, local_sum);
                topk_consider(fv.z, base + 2u, k, top_v, top_i, min_v, min_i, local_m, local_sum);
                topk_consider(fv.w, base + 3u, k, top_v, top_i, min_v, min_i, local_m, local_sum);
            }
            const uint tail = vec_n << 2;
            for (uint i = tail + tid; i < p.vocab_size; i += 256u) {
                topk_consider(float(logits[i]), i, k, top_v, top_i, min_v, min_i, local_m, local_sum);
            }
        } else {
            for (uint i = tid; i < p.vocab_size; i += 256u) {
                topk_consider(float(logits[i]), i, k, top_v, top_i, min_v, min_i, local_m, local_sum);
            }
        }
    } else {
        const device float *logits = ((const device float *)logits_raw) + ulong(row) * p.vocab_size;
        if (p.aligned_vec4 != 0u) {
            const uint vec_n = p.vocab_size >> 2;
            const device float4 *vlogits = (const device float4 *)logits;
            for (uint vi = tid; vi < vec_n; vi += 256u) {
                const float4 fv = vlogits[vi];
                const uint base = vi << 2;
                topk_consider(fv.x, base + 0u, k, top_v, top_i, min_v, min_i, local_m, local_sum);
                topk_consider(fv.y, base + 1u, k, top_v, top_i, min_v, min_i, local_m, local_sum);
                topk_consider(fv.z, base + 2u, k, top_v, top_i, min_v, min_i, local_m, local_sum);
                topk_consider(fv.w, base + 3u, k, top_v, top_i, min_v, min_i, local_m, local_sum);
            }
            const uint tail = vec_n << 2;
            for (uint i = tail + tid; i < p.vocab_size; i += 256u) {
                topk_consider(logits[i], i, k, top_v, top_i, min_v, min_i, local_m, local_sum);
            }
        } else {
            for (uint i = tid; i < p.vocab_size; i += 256u) {
                topk_consider(logits[i], i, k, top_v, top_i, min_v, min_i, local_m, local_sum);
            }
        }
    }

    threadgroup float s_m[8];
    threadgroup float s_sum[8];
    threadgroup float s_vals[8];
    threadgroup uint s_idxs[8];
    threadgroup float s_row_max;
    threadgroup float s_row_sum;
    threadgroup float s_log_row_sum;
    threadgroup uint s_win_i;

    // Reduce exact FP32 (max, sum_exp) across all 256 threads.
    const float sg_m = simd_max(local_m);
    const float sg_sum = simd_sum((local_sum > 0.0f) ? (local_sum * exp(local_m - sg_m)) : 0.0f);
    if (lane == 0u) {
        s_m[sg_id] = sg_m;
        s_sum[sg_id] = sg_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float rm = -INFINITY;
        for (uint s = 0u; s < 8u; s++) {
            if (s_sum[s] > 0.0f) {
                rm = max(rm, s_m[s]);
            }
        }
        float rsum = 0.0f;
        for (uint s = 0u; s < 8u; s++) {
            if (s_sum[s] > 0.0f) {
                rsum += s_sum[s] * exp(s_m[s] - rm);
            }
        }
        s_row_max = rm;
        s_row_sum = rsum;
        s_log_row_sum = (rsum > 0.0f) ? log(rsum) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const ulong out_base = ulong(row) * k;
    uint cur = 0u;
    for (uint out_k = 0u; out_k < k; out_k++) {
        float cv = (cur < k) ? top_v[cur] : -INFINITY;
        uint ci = (cur < k) ? top_i[cur] : 0xFFFFFFFFu;
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
            float fv = (lane < 8u) ? s_vals[lane] : -INFINITY;
            uint fi = (lane < 8u) ? s_idxs[lane] : 0xFFFFFFFFu;
            for (ushort offset = 4u; offset > 0u; offset >>= 1u) {
                const float ov = simd_shuffle_down(fv, offset);
                const uint oi = simd_shuffle_down(fi, offset);
                if (ov > fv || (ov == fv && oi < fi)) {
                    fv = ov;
                    fi = oi;
                }
            }
            if (lane == 0u) {
                s_win_i = fi;
                TopKCandidate cand;
                if (fi != 0xFFFFFFFFu && s_row_sum > 0.0f) {
                    cand.token_id = fi;
                    cand.log_prob = (fv - s_row_max) - s_log_row_sum;
                    cand.prob = exp(fv - s_row_max) / s_row_sum;
                    cand._pad = 0u;
                } else {
                    cand.token_id = 0u;
                    cand.prob = 0.0f;
                    cand.log_prob = -INFINITY;
                    cand._pad = 0u;
                }
                out_cands[out_base + out_k] = cand;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (cur < k && top_i[cur] == s_win_i) {
            cur++;
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 15: Non-Deduplicated Per-Slot MoE Fused 3-bit / 4-bit Gate + Up + SwiGLU
// (Spike S5 comparison path vs CSR-deduplicated Kernel 3, ADR-0009).
//
// Grid: (ceil(intermediate_size / 8), m * top_k, 1) threadgroups, 128 threads.
// ---------------------------------------------------------------------------
kernel void ami_moe_gate_up_swiglu_perslot(const device half *x [[buffer(0)]],
                                           const device uchar *weights_base [[buffer(1)]],
                                           const device ExpertWeights *expert_table [[buffer(2)]],
                                           const device uint *topk_experts [[buffer(3)]],
                                           device half *inter [[buffer(4)]],
                                           constant MoePerSlotParams &p [[buffer(5)]],
                                           uint2 tg_pos [[threadgroup_position_in_grid]],
                                           uint sg_id [[simdgroup_index_in_threadgroup]],
                                           uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint ROWS_PER_SG = 2u;
    const uint slot = tg_pos.y;
    const uint total_slots = p.m * p.top_k;
    if (slot >= total_slots) {
        return;
    }

    const uint row0 = tg_pos.x * 8u + sg_id * ROWS_PER_SG;
    if (row0 >= p.intermediate_size) {
        return;
    }

    const uint tok = slot / p.top_k;
    const uint expert_id = topk_experts[slot];
    const ExpertWeights ew = expert_table[expert_id];
    const device uchar *gate_qw = weights_base + ew.gate_qweight_offset;
    const device half2 *gate_sb = (const device half2 *)(weights_base + ew.gate_sb_offset);
    const device uchar *up_qw = weights_base + ew.up_qweight_offset;
    const device half2 *up_sb = (const device half2 *)(weights_base + ew.up_sb_offset);

    const uint sub_group = lane >> 3;
    const uint lane_in_grp = lane & 7u;
    const device half *x_tok = x + ulong(tok) * p.hidden_size;
    const bool full_rows = (row0 + ROWS_PER_SG <= p.intermediate_size);

    float g_acc[ROWS_PER_SG] = {0.0f, 0.0f};
    float u_acc[ROWS_PER_SG] = {0.0f, 0.0f};

    for (uint g_base = 0u; g_base < p.num_groups; g_base += 4u) {
        const uint g = g_base + sub_group;
        if (g >= p.num_groups) {
            continue;
        }
        const uint k_off = g * Q_GROUP_SIZE + lane_in_grp * 8u;
        const float4 fx0 = float4(*(const device half4 *)(x_tok + k_off));
        const float4 fx1 = float4(*(const device half4 *)(x_tok + k_off + 4u));
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

            for (uint r = 0u; r < ROWS_PER_SG; r++) {
                const uint row = row0 + r;
                if (full_rows || row < p.intermediate_size) {
                    const ulong row_w_off = ulong(row) * p.row_bytes + w_off;
                    const ulong row_sb_off = ulong(row) * p.num_groups + g;
                    const uint32_t gw = *(const device uint32_t *)(gate_qw + row_w_off);
                    const half2 gsb = gate_sb[row_sb_off];
                    g_acc[r] = fma(qdot8_q4(gw, px0, px1), float(gsb.x), fma(x_sum, float(gsb.y), g_acc[r]));

                    const uint32_t uw = *(const device uint32_t *)(up_qw + row_w_off);
                    const half2 usb = up_sb[row_sb_off];
                    u_acc[r] = fma(qdot8_q4(uw, px0, px1), float(usb.x), fma(x_sum, float(usb.y), u_acc[r]));
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

            for (uint r = 0u; r < ROWS_PER_SG; r++) {
                const uint row = row0 + r;
                if (full_rows || row < p.intermediate_size) {
                    const ulong row_w_off = ulong(row) * p.row_bytes + w_off;
                    const ulong row_sb_off = ulong(row) * p.num_groups + g;
                    const device uchar *gwp = gate_qw + row_w_off;
                    const half2 gsb = gate_sb[row_sb_off];
                    g_acc[r] = fma(qdot8_q3(gwp[0], gwp[1], gwp[2], px0, px1, x2_hi, x5_hi),
                                   float(gsb.x), fma(x_sum, float(gsb.y), g_acc[r]));

                    const device uchar *uwp = up_qw + row_w_off;
                    const half2 usb = up_sb[row_sb_off];
                    u_acc[r] = fma(qdot8_q3(uwp[0], uwp[1], uwp[2], px0, px1, x2_hi, x5_hi),
                                   float(usb.x), fma(x_sum, float(usb.y), u_acc[r]));
                }
            }
        } else {
            const float4 px0 = float4(fx0.x,
                                      fx0.y * (1.0f / 4.0f),
                                      fx0.z * (1.0f / 16.0f),
                                      fx0.w * (1.0f / 64.0f));
            const float4 px1 = float4(fx1.x,
                                      fx1.y * (1.0f / 4.0f),
                                      fx1.z * (1.0f / 16.0f),
                                      fx1.w * (1.0f / 64.0f));
            const uint w_off = g * 16u + lane_in_grp * 2u;

            for (uint r = 0u; r < ROWS_PER_SG; r++) {
                const uint row = row0 + r;
                if (full_rows || row < p.intermediate_size) {
                    const ulong row_w_off = ulong(row) * p.row_bytes + w_off;
                    const ulong row_sb_off = ulong(row) * p.num_groups + g;
                    const uint16_t gw = *(const device uint16_t *)(gate_qw + row_w_off);
                    const half2 gsb = gate_sb[row_sb_off];
                    g_acc[r] = fma(qdot8_q2(gw, px0, px1), float(gsb.x), fma(x_sum, float(gsb.y), g_acc[r]));

                    const uint16_t uw = *(const device uint16_t *)(up_qw + row_w_off);
                    const half2 usb = up_sb[row_sb_off];
                    u_acc[r] = fma(qdot8_q2(uw, px0, px1), float(usb.x), fma(x_sum, float(usb.y), u_acc[r]));
                }
            }
        }
    }

    const ulong out_base = ulong(slot) * p.intermediate_size;
    for (uint r = 0u; r < ROWS_PER_SG; r++) {
        const float gt = simd_sum(g_acc[r]);
        const float ut = simd_sum(u_acc[r]);
        if (lane == 0u && (row0 + r) < p.intermediate_size) {
            inter[out_base + row0 + r] = half(fast_silu(gt) * ut);
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 16: Non-Deduplicated Per-Slot MoE Fused 3-bit / 4-bit Down Projection
// (Spike S5 comparison path vs CSR-deduplicated Kernel 4, ADR-0009).
//
// Grid: (ceil(hidden_size / 16), m * top_k, 1) threadgroups, 128 threads.
// ---------------------------------------------------------------------------
kernel void ami_moe_down_proj_perslot(const device half *inter [[buffer(0)]],
                                      const device uchar *weights_base [[buffer(1)]],
                                      const device ExpertWeights *expert_table [[buffer(2)]],
                                      const device uint *topk_experts [[buffer(3)]],
                                      const device float *topk_weights [[buffer(4)]],
                                      device float *down_out [[buffer(5)]],
                                      constant MoePerSlotParams &p [[buffer(6)]],
                                      uint2 tg_pos [[threadgroup_position_in_grid]],
                                      uint sg_id [[simdgroup_index_in_threadgroup]],
                                      uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint ROWS_PER_SG = 4u;
    const uint slot = tg_pos.y;
    const uint total_slots = p.m * p.top_k;
    if (slot >= total_slots) {
        return;
    }

    const uint row0 = tg_pos.x * 16u + sg_id * ROWS_PER_SG;
    if (row0 >= p.hidden_size) {
        return;
    }

    const uint expert_id = topk_experts[slot];
    const float rw = topk_weights[slot];
    const ExpertWeights ew = expert_table[expert_id];
    const device uchar *down_qw = weights_base + ew.down_qweight_offset;
    const device half2 *down_sb = (const device half2 *)(weights_base + ew.down_sb_offset);

    const uint sub_group = lane >> 3;
    const uint lane_in_grp = lane & 7u;
    const device half *in_row = inter + ulong(slot) * p.intermediate_size;
    const bool full_rows = (row0 + ROWS_PER_SG <= p.hidden_size);
    float acc[ROWS_PER_SG] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint g_base = 0u; g_base < p.num_groups; g_base += 4u) {
        const uint g = g_base + sub_group;
        if (g >= p.num_groups) {
            continue;
        }
        const uint k_off = g * Q_GROUP_SIZE + lane_in_grp * 8u;
        const float4 fx0 = float4(*(const device half4 *)(in_row + k_off));
        const float4 fx1 = float4(*(const device half4 *)(in_row + k_off + 4u));
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

            for (uint r = 0u; r < ROWS_PER_SG; r++) {
                const uint row = row0 + r;
                if (full_rows || row < p.hidden_size) {
                    const uint32_t dw = *(const device uint32_t *)(down_qw + ulong(row) * p.row_bytes + w_off);
                    const half2 sb = down_sb[ulong(row) * p.num_groups + g];
                    acc[r] = fma(qdot8_q4(dw, px0, px1), float(sb.x), fma(x_sum, float(sb.y), acc[r]));
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

            for (uint r = 0u; r < ROWS_PER_SG; r++) {
                const uint row = row0 + r;
                if (full_rows || row < p.hidden_size) {
                    const device uchar *wp = down_qw + ulong(row) * p.row_bytes + w_off;
                    const half2 sb = down_sb[ulong(row) * p.num_groups + g];
                    acc[r] = fma(qdot8_q3(wp[0], wp[1], wp[2], px0, px1, x2_hi, x5_hi),
                                 float(sb.x), fma(x_sum, float(sb.y), acc[r]));
                }
            }
        } else {
            const float4 px0 = float4(fx0.x,
                                      fx0.y * (1.0f / 4.0f),
                                      fx0.z * (1.0f / 16.0f),
                                      fx0.w * (1.0f / 64.0f));
            const float4 px1 = float4(fx1.x,
                                      fx1.y * (1.0f / 4.0f),
                                      fx1.z * (1.0f / 16.0f),
                                      fx1.w * (1.0f / 64.0f));
            const uint w_off = g * 16u + lane_in_grp * 2u;

            for (uint r = 0u; r < ROWS_PER_SG; r++) {
                const uint row = row0 + r;
                if (full_rows || row < p.hidden_size) {
                    const uint16_t dw = *(const device uint16_t *)(down_qw + ulong(row) * p.row_bytes + w_off);
                    const half2 sb = down_sb[ulong(row) * p.num_groups + g];
                    acc[r] = fma(qdot8_q2(dw, px0, px1), float(sb.x), fma(x_sum, float(sb.y), acc[r]));
                }
            }
        }
    }

    const ulong out_base = ulong(slot) * p.hidden_size;
    for (uint r = 0u; r < ROWS_PER_SG; r++) {
        const float total = simd_sum(acc[r]);
        if (lane == 0u && (row0 + r) < p.hidden_size) {
            down_out[out_base + row0 + r] = total * rw;
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 17: Strided 2D f16 Row Copy (feature-tap packing & [e_norm, h_norm]
// concatenation for EAGLE-3 / EAGLE-2 / MTP draft heads, M3).
//
// Copies `[num_rows, copy_cols]` from `src[row * src_stride + src_col_off]`
// to `dst[row * dst_stride + dst_col_off]`.
// Grid: (ceil(copy_cols / 256), num_rows, 1) threadgroups, 256 threads.
// ---------------------------------------------------------------------------
struct CopyStridedParams {
    uint num_rows;
    uint copy_cols;
    uint src_stride;
    uint src_col_off;
    uint dst_stride;
    uint dst_col_off;
    uint _pad0;
    uint _pad1;
};

static_assert(sizeof(CopyStridedParams) == 32, "CopyStridedParams must be 32 bytes");

kernel void ami_copy_strided_f16(const device half *src [[buffer(0)]],
                                 device half *dst [[buffer(1)]],
                                 constant CopyStridedParams &p [[buffer(2)]],
                                 uint2 tg_pos [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]]) {
    const uint row = tg_pos.y;
    const uint col = tg_pos.x * 256u + tid;
    if (row >= p.num_rows || col >= p.copy_cols) {
        return;
    }
    dst[ulong(row) * p.dst_stride + p.dst_col_off + col] =
        src[ulong(row) * p.src_stride + p.src_col_off + col];
}

// ---------------------------------------------------------------------------
// Kernel 18: Indexed f16 Row Gather (zero-CPU-copy accepted-node target
// feature gather and EAGLE-2 parent-state gather, M3).
//
// For `out_row in 0..count`: copies `src[indices[out_row] * row_elems + col]`
// into `dst[out_row * row_elems + col]`.
// Grid: (ceil(row_elems / 256), count, 1) threadgroups, 256 threads.
// ---------------------------------------------------------------------------
struct GatherRowsParams {
    uint count;
    uint row_elems;
    uint _pad0;
    uint _pad1;
};

static_assert(sizeof(GatherRowsParams) == 16, "GatherRowsParams must be 16 bytes");

kernel void ami_gather_rows_f16(const device half *src [[buffer(0)]],
                                const device uint *indices [[buffer(1)]],
                                device half *dst [[buffer(2)]],
                                constant GatherRowsParams &p [[buffer(3)]],
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

// ===========================================================================
// Milestone M5 (G13, ADR-0017): Scheme-Bucketed Heterogeneous MoE CSR Routing
//
// Single 256-thread threadgroup computes FP32 softmax + top-K (identical to
// `ami_moe_route_topk`, supporting up to 1024 experts and top_k <= 16), then
// partitions active unique experts by their static `WeightDType` scheme
// (`0=f16, 1=bf16, 2=q4, 3=q3, 4=q2`) into 5 disjoint preallocated CSR buckets
// sharing `token_indices`, `routing_weights`, `slot_k_indices`, `inter` and
// `down_out` with zero CPU-GPU roundtrips and zero dynamic allocation. Also
// fills the unified `sched_hdr`/`unique_experts`/`expert_token_offsets` arrays
// so telemetry and cold-expert pinning work transparently.
// ===========================================================================

struct BucketRouteParams {
    uint m;
    uint num_experts;
    uint top_k;
    uint norm_topk_prob;
    uint bucket_unique_stride;
    uint bucket_offsets_stride;
    uint _pad0;
    uint _pad1;
};

static_assert(sizeof(BucketRouteParams) == 32, "BucketRouteParams must be 32 bytes");

kernel void ami_moe_route_topk_bucketed(const device uchar *logits_raw [[buffer(0)]],
                                        device MoeScheduleHeader *sched_hdr [[buffer(1)]],
                                        device uint *unique_experts [[buffer(2)]],
                                        device uint *expert_token_offsets [[buffer(3)]],
                                        device uint *token_indices [[buffer(4)]],
                                        device float *routing_weights [[buffer(5)]],
                                        device uint *slot_k_indices [[buffer(6)]],
                                        device uint *scratch_topk_experts [[buffer(7)]],
                                        device float *scratch_topk_weights [[buffer(8)]],
                                        const device uchar *expert_schemes [[buffer(9)]],
                                        device uchar *bucket_headers [[buffer(10)]],
                                        device uint *bucket_unique_experts [[buffer(11)]],
                                        device uint *bucket_expert_token_offsets [[buffer(12)]],
                                        constant BucketRouteParams &p [[buffer(13)]],
                                        uint tid [[thread_index_in_threadgroup]],
                                        uint sg_id [[simdgroup_index_in_threadgroup]],
                                        uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint NUM_SCHEMES = 5u;
    threadgroup atomic_uint s_counts[1024];
    threadgroup atomic_uint s_cursors[1024];
    threadgroup uint s_prefix_tmp[8];
    threadgroup uint s_u_cnt;
    threadgroup uint s_a_cnt;

    const uint top_k = (MOE_TOP_K > 0u) ? MOE_TOP_K : min(p.top_k, 16u);

    // Phase 0: clear per-expert token histograms.
    for (uint e = tid; e < p.num_experts; e += 256u) {
        atomic_store_explicit(&s_counts[e], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 1: parallel per-token fp32 softmax & top-k in registers (8 tokens/wave).
    for (uint tok = sg_id; tok < p.m; tok += 8u) {
        uint sel_e[16];
        float sel_w[16];
        float w_sum = 0.0f;

        if (p.num_experts <= 256u) {
            float vals[8];
            float local_max = -INFINITY;
            for (uint r = 0u; r < 8u; r++) {
                const uint e = lane + r * 32u;
                if (e < p.num_experts) {
                    const float v = LOGITS_F16
                        ? float(((const device half *)logits_raw)[ulong(tok) * p.num_experts + e])
                        : ((const device float *)logits_raw)[ulong(tok) * p.num_experts + e];
                    vals[r] = v;
                    local_max = max(local_max, v);
                } else {
                    vals[r] = -INFINITY;
                }
            }

            const float max_v = simd_max(local_max);
            float local_sum = 0.0f;
            for (uint r = 0u; r < 8u; r++) {
                const uint e = lane + r * 32u;
                if (e < p.num_experts) {
                    const float ex = exp(vals[r] - max_v);
                    vals[r] = ex;
                    local_sum += ex;
                } else {
                    vals[r] = -INFINITY;
                }
            }

            const float sum_exp = simd_sum(local_sum);
            const float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
            for (uint r = 0u; r < 8u; r++) {
                const uint e = lane + r * 32u;
                if (e < p.num_experts) {
                    vals[r] *= inv_sum;
                }
            }

            const uint k_limit = (MOE_TOP_K > 0u) ? MOE_TOP_K : top_k;
            for (uint k = 0u; k < k_limit; k++) {
                float best_v = -INFINITY;
                uint best_e = 0xFFFFFFFFu;
                for (uint r = 0u; r < 8u; r++) {
                    const uint e = lane + r * 32u;
                    if (e < p.num_experts) {
                        const float v = vals[r];
                        if (v > best_v || (v == best_v && e < best_e)) {
                            best_v = v;
                            best_e = e;
                        }
                    }
                }
                for (ushort offset = 16u; offset > 0u; offset >>= 1u) {
                    const float ov = simd_shuffle_down(best_v, offset);
                    const uint oe = simd_shuffle_down(best_e, offset);
                    if (ov > best_v || (ov == best_v && oe < best_e)) {
                        best_v = ov;
                        best_e = oe;
                    }
                }
                const uint raw_win_e = simd_broadcast_first(best_e);
                const uint win_e = (raw_win_e < p.num_experts) ? raw_win_e : 0u;
                const float win_v = simd_broadcast_first(best_v);
                if ((win_e & 31u) == lane) {
                    vals[win_e >> 5] = -INFINITY;
                }
                if (lane == 0u) {
                    sel_e[k] = win_e;
                    sel_w[k] = win_v;
                    w_sum += win_v;
                }
            }
        } else {
            const uint chunks = (p.num_experts + 31u) >> 5;
            const ulong row_base = ulong(tok) * p.num_experts;
            float local_max = -INFINITY;
            for (uint r = 0u; r < chunks; r++) {
                const uint e = lane + r * 32u;
                if (e < p.num_experts) {
                    const float v = LOGITS_F16
                        ? float(((const device half *)logits_raw)[row_base + e])
                        : ((const device float *)logits_raw)[row_base + e];
                    local_max = max(local_max, v);
                }
            }
            const float max_v = simd_max(local_max);

            float local_sum = 0.0f;
            for (uint r = 0u; r < chunks; r++) {
                const uint e = lane + r * 32u;
                if (e < p.num_experts) {
                    const float v = LOGITS_F16
                        ? float(((const device half *)logits_raw)[row_base + e])
                        : ((const device float *)logits_raw)[row_base + e];
                    local_sum += exp(v - max_v);
                }
            }
            const float sum_exp = simd_sum(local_sum);
            const float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;

            uint picked_mask = 0u;
            const uint k_limit = (MOE_TOP_K > 0u) ? MOE_TOP_K : top_k;
            for (uint k = 0u; k < k_limit; k++) {
                float best_v = -INFINITY;
                uint best_e = 0xFFFFFFFFu;
                for (uint r = 0u; r < chunks; r++) {
                    const uint e = lane + r * 32u;
                    if (e < p.num_experts && ((picked_mask >> r) & 1u) == 0u) {
                        const float raw_v = LOGITS_F16
                            ? float(((const device half *)logits_raw)[row_base + e])
                            : ((const device float *)logits_raw)[row_base + e];
                        const float v = exp(raw_v - max_v) * inv_sum;
                        if (v > best_v || (v == best_v && e < best_e)) {
                            best_v = v;
                            best_e = e;
                        }
                    }
                }
                for (ushort offset = 16u; offset > 0u; offset >>= 1u) {
                    const float ov = simd_shuffle_down(best_v, offset);
                    const uint oe = simd_shuffle_down(best_e, offset);
                    if (ov > best_v || (ov == best_v && oe < best_e)) {
                        best_v = ov;
                        best_e = oe;
                    }
                }
                const uint raw_win_e = simd_broadcast_first(best_e);
                const uint win_e = (raw_win_e < p.num_experts) ? raw_win_e : 0u;
                const float win_v = simd_broadcast_first(best_v);
                if ((win_e & 31u) == lane) {
                    picked_mask |= (1u << (win_e >> 5));
                }
                if (lane == 0u) {
                    sel_e[k] = win_e;
                    sel_w[k] = win_v;
                    w_sum += win_v;
                }
            }
        }

        if (lane == 0u) {
            const float renorm = (p.norm_topk_prob != 0u && w_sum > 0.0f) ? (1.0f / w_sum) : 1.0f;
            const ulong base = ulong(tok) * top_k;
            for (uint k = 0u; k < top_k; k++) {
                const uint e = sel_e[k];
                scratch_topk_experts[base + k] = e;
                scratch_topk_weights[base + k] = sel_w[k] * renorm;
                atomic_fetch_add_explicit(&s_counts[e], 1u, memory_order_relaxed);
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    // Phase 2: partition active experts by static scheme into disjoint CSR slices.
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

            // 2a. Exclusive prefix sum of `in_s` -> block-local unique index `u_in_blk` and count `s_u_cnt`.
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

            // 2b. Exclusive prefix sum of `cnt_s` -> block-local assignment offset `a_in_blk` and count `s_a_cnt`.
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
            device MoeScheduleHeader *bhdr = (device MoeScheduleHeader *)(bucket_headers + ulong(s) * 64ul);
            bhdr->num_unique_experts = u_cnt_s;
            bhdr->total_assignments = a_cnt_s;
            bhdr->m = p.m;
            bhdr->top_k = top_k;
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
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: scatter token assignments into each active expert's contiguous CSR range.
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

// ===========================================================================
// M6 G7: Unconditional Shared Expert SwiGLU + Optional Scalar Sigmoid Gate
//
// Accumulates an unconditional shared expert branch into the routed MoE output
// on GPU with zero CPU-GPU syncs and zero heap allocations (ADR-0015).
// Supports q2/q3/q4 affine group-64 and dense f16/bf16 weights, plus optional
// per-token scalar sigmoid gate `g[m] = sigmoid(dot(x[m], w_gate))`.
// ===========================================================================

struct MoeSharedGateParams {
    uint m;
    uint hidden_size;
    uint _pad0;
    uint _pad1;
};

struct MoeSharedGateUpParams {
    ulong gate_qweight_offset;
    ulong gate_sb_offset;
    ulong up_qweight_offset;
    ulong up_sb_offset;
    uint m;
    uint hidden_size;
    uint intermediate_size;
    uint num_groups;
    uint row_bytes;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

struct MoeSharedDownParams {
    ulong down_qweight_offset;
    ulong down_sb_offset;
    uint m;
    uint hidden_size;
    uint intermediate_size;
    uint num_groups;
    uint row_bytes;
    uint use_gate;
    uint _pad0;
    uint _pad1;
};

static_assert(sizeof(MoeSharedGateParams) == 16, "MoeSharedGateParams must be 16 bytes");
static_assert(sizeof(MoeSharedGateUpParams) == 64, "MoeSharedGateUpParams must be 64 bytes");
static_assert(sizeof(MoeSharedDownParams) == 48, "MoeSharedDownParams must be 48 bytes");

// Scalar sigmoid gate for shared expert: `gate_out[tok] = sigmoid(dot(x[tok], gate_w))`.
// Specialized on ROUTER_W_F16 / ROUTER_W_BF16. Grid: (ceil(m / 4), 1, 1), 128 threads.
kernel void ami_moe_shared_gate(const device half *x [[buffer(0)]],
                                const device uchar *gate_w [[buffer(1)]],
                                device float *gate_out [[buffer(2)]],
                                constant MoeSharedGateParams &p [[buffer(3)]],
                                uint tg_x [[threadgroup_position_in_grid]],
                                uint sg_id [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    const uint tok = tg_x * 4u + sg_id;
    if (tok >= p.m) {
        return;
    }

    float acc = 0.0f;
    if ((p.hidden_size & 3u) == 0u) {
        const uint vec_k = p.hidden_size >> 2;
        const device half4 *x_row = (const device half4 *)(x + ulong(tok) * p.hidden_size);
        if (ROUTER_W_BF16) {
            const device bfloat4 *w_row = (const device bfloat4 *)gate_w;
            for (uint k = lane; k < vec_k; k += 32u) {
                acc += dot(float4(x_row[k]), float4(w_row[k]));
            }
        } else if (ROUTER_W_F16) {
            const device half4 *w_row = (const device half4 *)gate_w;
            for (uint k = lane; k < vec_k; k += 32u) {
                acc += dot(float4(x_row[k]), float4(w_row[k]));
            }
        } else {
            const device float4 *w_row = (const device float4 *)gate_w;
            for (uint k = lane; k < vec_k; k += 32u) {
                acc += dot(float4(x_row[k]), w_row[k]);
            }
        }
    } else {
        const device half *x_row = x + ulong(tok) * p.hidden_size;
        if (ROUTER_W_BF16) {
            const device bfloat *w_row = (const device bfloat *)gate_w;
            for (uint k = lane; k < p.hidden_size; k += 32u) {
                acc = fma(float(x_row[k]), float(w_row[k]), acc);
            }
        } else if (ROUTER_W_F16) {
            const device half *w_row = (const device half *)gate_w;
            for (uint k = lane; k < p.hidden_size; k += 32u) {
                acc = fma(float(x_row[k]), float(w_row[k]), acc);
            }
        } else {
            const device float *w_row = (const device float *)gate_w;
            for (uint k = lane; k < p.hidden_size; k += 32u) {
                acc = fma(float(x_row[k]), w_row[k], acc);
            }
        }
    }

    const float total = simd_sum(acc);
    if (lane == 0u) {
        gate_out[tok] = 1.0f / (1.0f + exp(-total));
    }
}

// Quantized (q2/q3/q4 group-64) unconditional shared expert Gate + Up + SwiGLU.
// Weight-stationary across up to 4 tokens per TG.
// Grid: (ceil(intermediate_size / 8), ceil(m / 4), 1), 128 threads (4 simdgroups).
kernel void ami_moe_shared_gate_up_swiglu(const device half *x [[buffer(0)]],
                                          const device uchar *weights_base [[buffer(1)]],
                                          device half *shared_inter [[buffer(2)]],
                                          constant MoeSharedGateUpParams &p [[buffer(3)]],
                                          uint2 tg_pos [[threadgroup_position_in_grid]],
                                          uint sg_id [[simdgroup_index_in_threadgroup]],
                                          uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint ROWS_PER_SG = 2u;
    const uint row0 = tg_pos.x * 8u + sg_id * ROWS_PER_SG;
    const uint tok_base = tg_pos.y * 4u;
    if (row0 >= p.intermediate_size || tok_base >= p.m) {
        return;
    }

    const uint cnt = min(4u, p.m - tok_base);
    const device uchar *gate_qw = weights_base + p.gate_qweight_offset;
    const device half2 *gate_sb = (const device half2 *)(weights_base + p.gate_sb_offset);
    const device uchar *up_qw = weights_base + p.up_qweight_offset;
    const device half2 *up_sb = (const device half2 *)(weights_base + p.up_sb_offset);

    const uint sub_group = lane >> 3;
    const uint lane_in_grp = lane & 7u;

    const device half *x0 = x + ulong(tok_base + 0u) * p.hidden_size;
    const device half *x1 = x + ulong(tok_base + min(1u, cnt - 1u)) * p.hidden_size;
    const device half *x2 = x + ulong(tok_base + min(2u, cnt - 1u)) * p.hidden_size;
    const device half *x3 = x + ulong(tok_base + min(3u, cnt - 1u)) * p.hidden_size;

    for (uint r = 0u; r < ROWS_PER_SG; r++) {
        const uint row = row0 + r;
        if (row >= p.intermediate_size) {
            break;
        }
        float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
        float u0 = 0.0f, u1 = 0.0f, u2 = 0.0f, u3 = 0.0f;
        const ulong row_w_base = ulong(row) * p.row_bytes;
        const ulong row_sb_base = ulong(row) * p.num_groups;

        for (uint g_base = 0u; g_base < p.num_groups; g_base += 4u) {
            const uint g = g_base + sub_group;
            if (g >= p.num_groups) {
                continue;
            }
            const half2 gsb = gate_sb[row_sb_base + g];
            const half2 usb = up_sb[row_sb_base + g];
            float4 gw0, gw1, uw0, uw1;
            if (Q_BITS == 4u) {
                const uint w_off = g * 32u + lane_in_grp * 4u;
                dequant8_q4(*(const device uint32_t *)(gate_qw + row_w_base + w_off), float(gsb.x), float(gsb.y), gw0, gw1);
                dequant8_q4(*(const device uint32_t *)(up_qw + row_w_base + w_off), float(usb.x), float(usb.y), uw0, uw1);
            } else if (Q_BITS == 3u) {
                const uint w_off = g * 24u + lane_in_grp * 3u;
                const device uchar *gwp = gate_qw + row_w_base + w_off;
                const device uchar *uwp = up_qw + row_w_base + w_off;
                dequant8_q3(gwp[0], gwp[1], gwp[2], float(gsb.x), float(gsb.y), gw0, gw1);
                dequant8_q3(uwp[0], uwp[1], uwp[2], float(usb.x), float(usb.y), uw0, uw1);
            } else {
                const uint w_off = g * 16u + lane_in_grp * 2u;
                dequant8_q2(*(const device uint16_t *)(gate_qw + row_w_base + w_off), float(gsb.x), float(gsb.y), gw0, gw1);
                dequant8_q2(*(const device uint16_t *)(up_qw + row_w_base + w_off), float(usb.x), float(usb.y), uw0, uw1);
            }

            const uint k_off = g * Q_GROUP_SIZE + lane_in_grp * 8u;
            { const float4 a = float4(*(const device half4 *)(x0 + k_off)); const float4 b = float4(*(const device half4 *)(x0 + k_off + 4u)); g0 += dot(a, gw0) + dot(b, gw1); u0 += dot(a, uw0) + dot(b, uw1); }
            if (cnt > 1u) { const float4 a = float4(*(const device half4 *)(x1 + k_off)); const float4 b = float4(*(const device half4 *)(x1 + k_off + 4u)); g1 += dot(a, gw0) + dot(b, gw1); u1 += dot(a, uw0) + dot(b, uw1); }
            if (cnt > 2u) { const float4 a = float4(*(const device half4 *)(x2 + k_off)); const float4 b = float4(*(const device half4 *)(x2 + k_off + 4u)); g2 += dot(a, gw0) + dot(b, gw1); u2 += dot(a, uw0) + dot(b, uw1); }
            if (cnt > 3u) { const float4 a = float4(*(const device half4 *)(x3 + k_off)); const float4 b = float4(*(const device half4 *)(x3 + k_off + 4u)); g3 += dot(a, gw0) + dot(b, gw1); u3 += dot(a, uw0) + dot(b, uw1); }
        }

        const float gt0 = simd_sum(g0), ut0 = simd_sum(u0);
        const float gt1 = (cnt > 1u) ? simd_sum(g1) : 0.0f;
        const float ut1 = (cnt > 1u) ? simd_sum(u1) : 0.0f;
        const float gt2 = (cnt > 2u) ? simd_sum(g2) : 0.0f;
        const float ut2 = (cnt > 2u) ? simd_sum(u2) : 0.0f;
        const float gt3 = (cnt > 3u) ? simd_sum(g3) : 0.0f;
        const float ut3 = (cnt > 3u) ? simd_sum(u3) : 0.0f;
        if (lane == 0u) {
            shared_inter[ulong(tok_base + 0u) * p.intermediate_size + row] = half(fast_silu(gt0) * ut0);
            if (cnt > 1u) shared_inter[ulong(tok_base + 1u) * p.intermediate_size + row] = half(fast_silu(gt1) * ut1);
            if (cnt > 2u) shared_inter[ulong(tok_base + 2u) * p.intermediate_size + row] = half(fast_silu(gt2) * ut2);
            if (cnt > 3u) shared_inter[ulong(tok_base + 3u) * p.intermediate_size + row] = half(fast_silu(gt3) * ut3);
        }
    }
}

// Dense FP16 / BF16 unconditional shared expert Gate + Up + SwiGLU.
// Grid: (ceil(intermediate_size / 4), m, 1), 128 threads (4 simdgroups).
kernel void ami_moe_shared_gate_up_swiglu_float(const device half *x [[buffer(0)]],
                                                const device uchar *weights_base [[buffer(1)]],
                                                device half *shared_inter [[buffer(2)]],
                                                constant MoeSharedGateUpParams &p [[buffer(3)]],
                                                uint2 tg_pos [[threadgroup_position_in_grid]],
                                                uint sg_id [[simdgroup_index_in_threadgroup]],
                                                uint lane [[thread_index_in_simdgroup]]) {
    const uint row = tg_pos.x * 4u + sg_id;
    const uint tok = tg_pos.y;
    if (row >= p.intermediate_size || tok >= p.m) {
        return;
    }

    const uint vec_k = p.hidden_size >> 2;
    const device half4 *x_row = (const device half4 *)(x + ulong(tok) * p.hidden_size);
    float g_acc = 0.0f;
    float u_acc = 0.0f;

    if (MOE_W_BF16) {
        const device bfloat4 *gw = (const device bfloat4 *)((const device bfloat *)(weights_base + p.gate_qweight_offset) + ulong(row) * p.hidden_size);
        const device bfloat4 *uw = (const device bfloat4 *)((const device bfloat *)(weights_base + p.up_qweight_offset) + ulong(row) * p.hidden_size);
        for (uint k = lane; k < vec_k; k += 32u) {
            const float4 xv = float4(x_row[k]);
            g_acc += dot(xv, float4(gw[k]));
            u_acc += dot(xv, float4(uw[k]));
        }
    } else {
        const device half4 *gw = (const device half4 *)((const device half *)(weights_base + p.gate_qweight_offset) + ulong(row) * p.hidden_size);
        const device half4 *uw = (const device half4 *)((const device half *)(weights_base + p.up_qweight_offset) + ulong(row) * p.hidden_size);
        for (uint k = lane; k < vec_k; k += 32u) {
            const float4 xv = float4(x_row[k]);
            g_acc += dot(xv, float4(gw[k]));
            u_acc += dot(xv, float4(uw[k]));
        }
    }

    const float gt = simd_sum(g_acc);
    const float ut = simd_sum(u_acc);
    if (lane == 0u) {
        shared_inter[ulong(tok) * p.intermediate_size + row] = half(fast_silu(gt) * ut);
    }
}

// Quantized (q2/q3/q4 group-64) shared expert DownProj with fused optional
// scalar gate and in-place accumulation into `out` (`out += gate * down(inter)`).
// Grid: (ceil(hidden_size / 8), ceil(m / 4), 1), 128 threads (4 simdgroups).
kernel void ami_moe_shared_down_accum(const device half *shared_inter [[buffer(0)]],
                                      const device uchar *weights_base [[buffer(1)]],
                                      const device float *shared_gate_scores [[buffer(2)]],
                                      device half *out [[buffer(3)]],
                                      constant MoeSharedDownParams &p [[buffer(4)]],
                                      uint2 tg_pos [[threadgroup_position_in_grid]],
                                      uint sg_id [[simdgroup_index_in_threadgroup]],
                                      uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint ROWS_PER_SG = 2u;
    const uint row0 = tg_pos.x * 8u + sg_id * ROWS_PER_SG;
    const uint tok_base = tg_pos.y * 4u;
    if (row0 >= p.hidden_size || tok_base >= p.m) {
        return;
    }

    const uint cnt = min(4u, p.m - tok_base);
    const device uchar *down_qw = weights_base + p.down_qweight_offset;
    const device half2 *down_sb = (const device half2 *)(weights_base + p.down_sb_offset);

    const uint sub_group = lane >> 3;
    const uint lane_in_grp = lane & 7u;

    const device half *i0 = shared_inter + ulong(tok_base + 0u) * p.intermediate_size;
    const device half *i1 = shared_inter + ulong(tok_base + min(1u, cnt - 1u)) * p.intermediate_size;
    const device half *i2 = shared_inter + ulong(tok_base + min(2u, cnt - 1u)) * p.intermediate_size;
    const device half *i3 = shared_inter + ulong(tok_base + min(3u, cnt - 1u)) * p.intermediate_size;

    const float sc0 = (p.use_gate != 0u) ? shared_gate_scores[tok_base + 0u] : 1.0f;
    const float sc1 = (p.use_gate != 0u && cnt > 1u) ? shared_gate_scores[tok_base + 1u] : 1.0f;
    const float sc2 = (p.use_gate != 0u && cnt > 2u) ? shared_gate_scores[tok_base + 2u] : 1.0f;
    const float sc3 = (p.use_gate != 0u && cnt > 3u) ? shared_gate_scores[tok_base + 3u] : 1.0f;

    for (uint r = 0u; r < ROWS_PER_SG; r++) {
        const uint row = row0 + r;
        if (row >= p.hidden_size) {
            break;
        }
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        const ulong row_w_base = ulong(row) * p.row_bytes;
        const ulong row_sb_base = ulong(row) * p.num_groups;

        for (uint g_base = 0u; g_base < p.num_groups; g_base += 4u) {
            const uint g = g_base + sub_group;
            if (g >= p.num_groups) {
                continue;
            }
            const half2 dsb = down_sb[row_sb_base + g];
            float4 dw0, dw1;
            if (Q_BITS == 4u) {
                const uint w_off = g * 32u + lane_in_grp * 4u;
                dequant8_q4(*(const device uint32_t *)(down_qw + row_w_base + w_off), float(dsb.x), float(dsb.y), dw0, dw1);
            } else if (Q_BITS == 3u) {
                const uint w_off = g * 24u + lane_in_grp * 3u;
                const device uchar *dwp = down_qw + row_w_base + w_off;
                dequant8_q3(dwp[0], dwp[1], dwp[2], float(dsb.x), float(dsb.y), dw0, dw1);
            } else {
                const uint w_off = g * 16u + lane_in_grp * 2u;
                dequant8_q2(*(const device uint16_t *)(down_qw + row_w_base + w_off), float(dsb.x), float(dsb.y), dw0, dw1);
            }

            const uint k_off = g * Q_GROUP_SIZE + lane_in_grp * 8u;
            { const float4 x0v = float4(*(const device half4 *)(i0 + k_off)); const float4 x1v = float4(*(const device half4 *)(i0 + k_off + 4u)); a0 += dot(x0v, dw0) + dot(x1v, dw1); }
            if (cnt > 1u) { const float4 x0v = float4(*(const device half4 *)(i1 + k_off)); const float4 x1v = float4(*(const device half4 *)(i1 + k_off + 4u)); a1 += dot(x0v, dw0) + dot(x1v, dw1); }
            if (cnt > 2u) { const float4 x0v = float4(*(const device half4 *)(i2 + k_off)); const float4 x1v = float4(*(const device half4 *)(i2 + k_off + 4u)); a2 += dot(x0v, dw0) + dot(x1v, dw1); }
            if (cnt > 3u) { const float4 x0v = float4(*(const device half4 *)(i3 + k_off)); const float4 x1v = float4(*(const device half4 *)(i3 + k_off + 4u)); a3 += dot(x0v, dw0) + dot(x1v, dw1); }
        }

        const float y0 = simd_sum(a0) * sc0;
        const float y1 = (cnt > 1u) ? simd_sum(a1) * sc1 : 0.0f;
        const float y2 = (cnt > 2u) ? simd_sum(a2) * sc2 : 0.0f;
        const float y3 = (cnt > 3u) ? simd_sum(a3) * sc3 : 0.0f;
        if (lane == 0u) {
            const ulong o0 = ulong(tok_base + 0u) * p.hidden_size + row;
            out[o0] = half(float(out[o0]) + y0);
            if (cnt > 1u) { const ulong o1 = ulong(tok_base + 1u) * p.hidden_size + row; out[o1] = half(float(out[o1]) + y1); }
            if (cnt > 2u) { const ulong o2 = ulong(tok_base + 2u) * p.hidden_size + row; out[o2] = half(float(out[o2]) + y2); }
            if (cnt > 3u) { const ulong o3 = ulong(tok_base + 3u) * p.hidden_size + row; out[o3] = half(float(out[o3]) + y3); }
        }
    }
}

// Dense FP16 / BF16 shared expert DownProj with fused optional scalar gate and
// in-place accumulation into `out` (`out += gate * down(inter)`).
// Grid: (ceil(hidden_size / 4), m, 1), 128 threads (4 simdgroups).
kernel void ami_moe_shared_down_accum_float(const device half *shared_inter [[buffer(0)]],
                                            const device uchar *weights_base [[buffer(1)]],
                                            const device float *shared_gate_scores [[buffer(2)]],
                                            device half *out [[buffer(3)]],
                                            constant MoeSharedDownParams &p [[buffer(4)]],
                                            uint2 tg_pos [[threadgroup_position_in_grid]],
                                            uint sg_id [[simdgroup_index_in_threadgroup]],
                                            uint lane [[thread_index_in_simdgroup]]) {
    const uint row = tg_pos.x * 4u + sg_id;
    const uint tok = tg_pos.y;
    if (row >= p.hidden_size || tok >= p.m) {
        return;
    }

    const uint vec_k = p.intermediate_size >> 2;
    const device half4 *in_row = (const device half4 *)(shared_inter + ulong(tok) * p.intermediate_size);
    float acc = 0.0f;

    if (MOE_W_BF16) {
        const device bfloat4 *dw = (const device bfloat4 *)((const device bfloat *)(weights_base + p.down_qweight_offset) + ulong(row) * p.intermediate_size);
        for (uint k = lane; k < vec_k; k += 32u) {
            acc += dot(float4(in_row[k]), float4(dw[k]));
        }
    } else {
        const device half4 *dw = (const device half4 *)((const device half *)(weights_base + p.down_qweight_offset) + ulong(row) * p.intermediate_size);
        for (uint k = lane; k < vec_k; k += 32u) {
            acc += dot(float4(in_row[k]), float4(dw[k]));
        }
    }

    const float sc = (p.use_gate != 0u) ? shared_gate_scores[tok] : 1.0f;
    const float y = simd_sum(acc) * sc;
    if (lane == 0u) {
        const ulong o_idx = ulong(tok) * p.hidden_size + row;
        out[o_idx] = half(float(out[o_idx]) + y);
    }
}
