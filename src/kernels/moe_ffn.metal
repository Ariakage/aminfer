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

// Weight-stationary 8-element affine dequantization for multi-token expert reuse.
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

// ---------------------------------------------------------------------------
// Kernel 1: MoE Router Projection (`[M, hidden] @ router_w^T -> [M, E] f32`)
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
        if (ROUTER_W_F16) {
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
        if (ROUTER_W_F16) {
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
// Single threadgroup of 256 threads (8 simdgroups). Supports num_experts <= 256
// and top_k <= 16. Each simdgroup handles one token fully in registers with
// zero barriers for softmax and top-k selection, then all 256 threads compact
// active experts into sorted unique_experts[0..U) and scatter CSR token slots.
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
    threadgroup atomic_uint s_counts[256];
    threadgroup atomic_uint s_cursors[256];
    threadgroup uint s_prefix_tmp[8];
    threadgroup uint s_total_unique;

    const uint top_k = (MOE_TOP_K > 0u) ? MOE_TOP_K : min(p.top_k, 16u);

    // Phase 0: clear per-expert token histograms.
    atomic_store_explicit(&s_counts[tid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 1: parallel per-token fp32 softmax & top-k in registers (8 tokens/wave).
    for (uint tok = sg_id; tok < p.m; tok += 8u) {
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

        uint sel_e[16];
        float sel_w[16];
        float w_sum = 0.0f;

        if (MOE_TOP_K > 0u) {
            for (uint k = 0u; k < MOE_TOP_K; k++) {
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
            for (uint k = 0u; k < top_k; k++) {
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

    // Phase 2: compact active experts [0..U) and compute CSR token offsets.
    const uint cnt = (tid < p.num_experts)
        ? atomic_load_explicit(&s_counts[tid], memory_order_relaxed)
        : 0u;
    const uint is_active = (cnt > 0u) ? 1u : 0u;

    // 2a. Exclusive prefix sum of `is_active` -> compacted unique index `u_idx` and total `U`.
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
            s_total_unique = ex + v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint u_idx = s_prefix_tmp[sg_id] + sg_act_excl;

    // 2b. Exclusive prefix sum of `cnt` -> CSR start offset `tok_off`.
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
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint tok_off = s_prefix_tmp[sg_id] + sg_cnt_excl;

    atomic_store_explicit(&s_cursors[tid], tok_off, memory_order_relaxed);
    if (is_active != 0u) {
        unique_experts[u_idx] = tid;
        expert_token_offsets[u_idx] = tok_off;
    }
    if (tid == 0u) {
        const uint u_total = s_total_unique;
        const uint total_assign = p.m * top_k;
        expert_token_offsets[u_total] = total_assign;
        sched_hdr->num_unique_experts = u_total;
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

    // Multi-token deduplicated path: dequantize each expert row ONCE across up to 16 tokens.
    constexpr uint T_BLK = 16u;
    for (uint t_base = 0u; t_base < count_u; t_base += T_BLK) {
        const uint t_cnt = min(T_BLK, count_u - t_base);
        uint toks[T_BLK];
        for (uint t = 0u; t < t_cnt; t++) {
            toks[t] = token_indices[s0 + t_base + t];
        }

        for (uint r = 0u; r < ROWS_PER_SG; r++) {
            const uint row = row0 + r;
            if (row >= p.intermediate_size) {
                break;
            }
            float g_acc[T_BLK];
            float u_acc[T_BLK];
            for (uint t = 0u; t < t_cnt; t++) {
                g_acc[t] = 0.0f;
                u_acc[t] = 0.0f;
            }

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
                    const uint32_t gw = *(const device uint32_t *)(gate_qw + row_w_base + w_off);
                    const uint32_t uw = *(const device uint32_t *)(up_qw + row_w_base + w_off);
                    dequant8_q4(gw, float(gsb.x), float(gsb.y), gw0, gw1);
                    dequant8_q4(uw, float(usb.x), float(usb.y), uw0, uw1);
                } else {
                    const uint w_off = g * 24u + lane_in_grp * 3u;
                    const device uchar *gwp = gate_qw + row_w_base + w_off;
                    const device uchar *uwp = up_qw + row_w_base + w_off;
                    dequant8_q3(gwp[0], gwp[1], gwp[2], float(gsb.x), float(gsb.y), gw0, gw1);
                    dequant8_q3(uwp[0], uwp[1], uwp[2], float(usb.x), float(usb.y), uw0, uw1);
                }

                const uint k_off = g * Q_GROUP_SIZE + lane_in_grp * 8u;
                for (uint t = 0u; t < t_cnt; t++) {
                    const device half *x_tok = x + ulong(toks[t]) * p.hidden_size + k_off;
                    const float4 fx0 = float4(*(const device half4 *)(x_tok));
                    const float4 fx1 = float4(*(const device half4 *)(x_tok + 4u));
                    g_acc[t] += dot(fx0, gw0) + dot(fx1, gw1);
                    u_acc[t] += dot(fx0, uw0) + dot(fx1, uw1);
                }
            }

            for (uint t = 0u; t < t_cnt; t++) {
                const float gt = simd_sum(g_acc[t]);
                const float ut = simd_sum(u_acc[t]);
                if (lane == 0u) {
                    const ulong slot = ulong(s0 + t_base + t);
                    inter[slot * p.intermediate_size + row] = half(fast_silu(gt) * ut);
                }
            }
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

    // Multi-token deduplicated path: dequantize each down_proj row ONCE across up to 16 tokens.
    constexpr uint T_BLK = 16u;
    for (uint t_base = 0u; t_base < count_u; t_base += T_BLK) {
        const uint t_cnt = min(T_BLK, count_u - t_base);

        for (uint r = 0u; r < ROWS_PER_SG; r++) {
            const uint row = row0 + r;
            if (row >= p.hidden_size) {
                break;
            }

            float acc[T_BLK];
            for (uint t = 0u; t < t_cnt; t++) {
                acc[t] = 0.0f;
            }

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
                    const uint32_t dw = *(const device uint32_t *)(down_qw + row_w_base + w_off);
                    dequant8_q4(dw, float(sb.x), float(sb.y), dw0, dw1);
                } else {
                    const uint w_off = g * 24u + lane_in_grp * 3u;
                    const device uchar *wp = down_qw + row_w_base + w_off;
                    dequant8_q3(wp[0], wp[1], wp[2], float(sb.x), float(sb.y), dw0, dw1);
                }

                const uint k_off = g * Q_GROUP_SIZE + lane_in_grp * 8u;
                for (uint t = 0u; t < t_cnt; t++) {
                    const device half *in_row = inter + ulong(s0 + t_base + t) * p.intermediate_size + k_off;
                    const float4 fx0 = float4(*(const device half4 *)(in_row));
                    const float4 fx1 = float4(*(const device half4 *)(in_row + 4u));
                    acc[t] += dot(fx0, dw0) + dot(fx1, dw1);
                }
            }

            for (uint t = 0u; t < t_cnt; t++) {
                const float total = simd_sum(acc[t]);
                if (lane == 0u) {
                    const uint s = s0 + t_base + t;
                    const uint tok = token_indices[s];
                    const uint k_idx = slot_k_indices[s];
                    const float rw = routing_weights[s];
                    down_out[(ulong(tok) * p.top_k + k_idx) * p.hidden_size + row] = total * rw;
                }
            }
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
