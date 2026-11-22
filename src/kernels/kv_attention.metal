// KV cache IOSurface texture kernels, Spike S1 microbenchmarks, RMSNorm,
// QK-Norm + NeoX RoPE + KV-Write, and GQA decode attention (R3, R4, S1, G4).
#include <metal_stdlib>
using namespace metal;

constant uint ATTN_HEAD_DIM [[function_constant(0)]];
constant uint S1_ACCESS_MODE [[function_constant(1)]];
// Norm weight storage for RMSNorm / QK-Norm: false = `half`, true = `bfloat`
// (ADR-0016). Variance and scaling accumulate in FP32 either way.
constant bool NORM_W_BF16 [[function_constant(2)]];
// Compile-time speculative tree width cap: 16, 32, or 64 (ADR-0014).
constant uint TREE_MAX_NODES [[function_constant(3)]];

// `bf16` is always a compile-time constant at every call site (a function
// constant or a literal), so the untaken branch is eliminated.
static inline float norm_w1(const device uchar *w, uint i, bool bf16) {
    if (bf16) {
        return float(((const device bfloat *)w)[i]);
    }
    return float(((const device half *)w)[i]);
}

static inline float4 norm_w4(const device uchar *w, uint i4, bool bf16) {
    if (bf16) {
        return float4(((const device bfloat4 *)w)[i4]);
    }
    return float4(((const device half4 *)w)[i4]);
}

// Bindless pair of resident IOSurface-backed Metal texture views for one KV chunk.
struct KvTilePair {
    texture2d<half, access::read> k_tex;
    texture2d<half, access::read> v_tex;
};

// Bindless pair of resident zero-copy alias MTLBuffer pointers (Spike S1 only).
struct KvBufPair {
    const device half4 *k_buf;
    const device half4 *v_buf;
};

static_assert(sizeof(KvTilePair) == 16, "KvTilePair must match two u64 MTLResourceIDs");
static_assert(sizeof(KvBufPair) == 16, "KvBufPair must match two u64 GPU addresses");

struct RmsNormParams {
    uint num_rows;
    uint hidden_dim;
    float eps;
    uint _pad;
};

struct QkNormRopeKvParams {
    uint num_q_heads;
    uint num_kv_heads;
    uint head_dim;
    uint pos;
    uint chunk_tokens;
    uint slot_in_chunk;
    float rms_eps;
    float rope_theta;
    uint apply_qk_norm;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

struct GqaDecodeParams {
    uint num_q_heads;
    uint num_kv_heads;
    uint seq_len;
    uint chunk_tokens;
    uint num_splits;
    uint q_groups_per_kv;
    float scale;
    uint _pad;
};

struct SpikeS1Params {
    uint num_q_heads;
    uint num_kv_heads;
    uint head_dim;
    uint chunk_tokens;
    uint num_chunks;
    float scale;
    uint _pad0;
    uint _pad1;
};

// Sum across each 16-lane half-simdgroup (0..15 and 16..31 independently).
static inline float half_simd_sum(float v) {
    v += simd_shuffle_xor(v, 8u);
    v += simd_shuffle_xor(v, 4u);
    v += simd_shuffle_xor(v, 2u);
    v += simd_shuffle_xor(v, 1u);
    return v;
}

// Min across each 16-lane half-simdgroup (0..15 and 16..31 independently).
static inline float half_simd_min(float v) {
    v = min(v, simd_shuffle_xor(v, 8u));
    v = min(v, simd_shuffle_xor(v, 4u));
    v = min(v, simd_shuffle_xor(v, 2u));
    v = min(v, simd_shuffle_xor(v, 1u));
    return v;
}

// Max across each 16-lane half-simdgroup (0..15 and 16..31 independently).
static inline float half_simd_max(float v) {
    v = max(v, simd_shuffle_xor(v, 8u));
    v = max(v, simd_shuffle_xor(v, 4u));
    v = max(v, simd_shuffle_xor(v, 2u));
    v = max(v, simd_shuffle_xor(v, 1u));
    return v;
}

// ---------------------------------------------------------------------------
// Kernel 1: RMSNorm (fp16 -> fp16 with `half` or `bfloat` weights selected by
// NORM_W_BF16, fp32 accumulation; supports in-place y == x)
//
// Grid: (num_rows, 1, 1) threadgroups, 256 threads (8 simdgroups) per row.
// ---------------------------------------------------------------------------
kernel void ami_rmsnorm(const device half *x [[buffer(0)]],
                        const device uchar *w [[buffer(1)]],
                        device half *y [[buffer(2)]],
                        constant RmsNormParams &p [[buffer(3)]],
                        uint row [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]],
                        uint sg_id [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    if (row >= p.num_rows) {
        return;
    }

    const device half *x_row = x + ulong(row) * p.hidden_dim;
    device half *y_row = y + ulong(row) * p.hidden_dim;

    float sum_sq = 0.0f;
    if ((p.hidden_dim & 3u) == 0u) {
        const uint vec_len = p.hidden_dim >> 2;
        const device half4 *x4 = (const device half4 *)x_row;
        for (uint i = tid; i < vec_len; i += 256u) {
            const float4 v = float4(x4[i]);
            sum_sq += dot(v, v);
        }
    } else {
        for (uint i = tid; i < p.hidden_dim; i += 256u) {
            const float v = float(x_row[i]);
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
        float total = 0.0f;
        for (uint s = 0u; s < 8u; s++) {
            total += sg_sums[s];
        }
        sg_sums[0] = rsqrt(total / float(p.hidden_dim) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inv_rms = sg_sums[0];
    if ((p.hidden_dim & 3u) == 0u) {
        const uint vec_len = p.hidden_dim >> 2;
        const device half4 *x4 = (const device half4 *)x_row;
        device half4 *y4 = (device half4 *)y_row;
        for (uint i = tid; i < vec_len; i += 256u) {
            const float4 v = (float4(x4[i]) * inv_rms) * norm_w4(w, i, NORM_W_BF16);
            y4[i] = half4(v);
        }
    } else {
        for (uint i = tid; i < p.hidden_dim; i += 256u) {
            const float v = (float(x_row[i]) * inv_rms) * norm_w1(w, i, NORM_W_BF16);
            y_row[i] = half(v);
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: Optional Qwen3 QK-Norm (`half` or `bfloat` norm weights selected by
// NORM_W_BF16) + NeoX half-split RoPE + KV-Tile Write
//
// Grid: (num_q_heads + num_kv_heads, 1, 1) threadgroups, 32 threads (1 simdgroup).
// Updates Q in-place and writes rotated K and raw V directly into the current
// 512-token chunk's RGBA16Float IOSurface KV tiles (token-major rows).
// ---------------------------------------------------------------------------
static inline void qknorm_rope_kv_write_impl(device half *q,
                                             const device half *k,
                                             const device half *v,
                                             const device uchar *q_norm,
                                             const device uchar *k_norm,
                                             constant QkNormRopeKvParams &p,
                                             texture2d<half, access::write> k_tex,
                                             texture2d<half, access::write> v_tex,
                                             threadgroup float *vec,
                                             uint tg_id,
                                             uint lane,
                                             bool norm_bf16) {
    const uint total_heads = p.num_q_heads + p.num_kv_heads;
    if (tg_id >= total_heads) {
        return;
    }

    const bool is_q = (tg_id < p.num_q_heads);
    const uint head_idx = is_q ? tg_id : (tg_id - p.num_q_heads);
    const uint d_dim = p.head_dim;
    const uint half_dim = d_dim >> 1;

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
            vec[d] = (vec[d] * inv_rms) * norm_w1(nw, d, norm_bf16);
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // NeoX half-split RoPE: [0..d/2) paired with [d/2..d).
    const float log_theta = log(p.rope_theta);
    const float inv_d = 2.0f / float(d_dim);
    const float fpos = float(p.pos);
    for (uint j = lane; j < half_dim; j += 32u) {
        const float inv_freq = exp(-log_theta * (float(j) * inv_d));
        const float angle = fpos * inv_freq;
        const float c = cos(angle);
        const float s = sin(angle);
        const float x0 = vec[j];
        const float x1 = vec[j + half_dim];
        vec[j] = fma(x0, c, -x1 * s);
        vec[j + half_dim] = fma(x1, c, x0 * s);
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


// Specialized entry point used by `AttentionPipelines` (NORM_W_BF16 selects
// `half` or `bfloat` QK-norm weights).
kernel void ami_qknorm_rope_kv_write_norm(device half *q [[buffer(0)]],
                                          const device half *k [[buffer(1)]],
                                          const device half *v [[buffer(2)]],
                                          const device uchar *q_norm [[buffer(3)]],
                                          const device uchar *k_norm [[buffer(4)]],
                                          constant QkNormRopeKvParams &p [[buffer(5)]],
                                          texture2d<half, access::write> k_tex [[texture(0)]],
                                          texture2d<half, access::write> v_tex [[texture(1)]],
                                          uint tg_id [[threadgroup_position_in_grid]],
                                          uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float vec[256];
    qknorm_rope_kv_write_impl(q, k, v, q_norm, k_norm, p, k_tex, v_tex, vec, tg_id, lane, NORM_W_BF16);
}

// Unspecialized `half` QK-norm entry point (kept for bridge-level tests and
// callers that build the pipeline without function constants).
kernel void ami_qknorm_rope_kv_write(device half *q [[buffer(0)]],
                                     const device half *k [[buffer(1)]],
                                     const device half *v [[buffer(2)]],
                                     const device uchar *q_norm [[buffer(3)]],
                                     const device uchar *k_norm [[buffer(4)]],
                                     constant QkNormRopeKvParams &p [[buffer(5)]],
                                     texture2d<half, access::write> k_tex [[texture(0)]],
                                     texture2d<half, access::write> v_tex [[texture(1)]],
                                     uint tg_id [[threadgroup_position_in_grid]],
                                     uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float vec[256];
    qknorm_rope_kv_write_impl(q, k, v, q_norm, k_norm, p, k_tex, v_tex, vec, tg_id, lane, false);
}

// ---------------------------------------------------------------------------
// Kernel 3a: GQA Decode Attention (M = 1) over multi-chunk RGBA16Float KvTiles
//
// Specialized on ATTN_HEAD_DIM (64 or 128). Uses 256 threads (8 simdgroups) per
// threadgroup: all 256 threads cooperatively stage a tile of K and V into 4 KiB
// threadgroup SRAM (8 tokens for d=128, 16 tokens for d=64), then each simdgroup
// independently updates one query head in scalar/vector registers (zero array
// spills) using tiled FlashAttention-2 base-2 online softmax (`exp2`).
//
// Grid: (num_kv_heads * q_groups_per_kv, num_splits, 1) threadgroups, 256 threads.
// ---------------------------------------------------------------------------
kernel void ami_gqa_decode_attn_split(const device half *q [[buffer(0)]],
                                      const device KvTilePair *kv_tiles [[buffer(1)]],
                                      device half *out [[buffer(2)]],
                                      device float *partial_m [[buffer(3)]],
                                      device float *partial_l [[buffer(4)]],
                                      device float *partial_o [[buffer(5)]],
                                      constant GqaDecodeParams &p [[buffer(6)]],
                                      uint2 tg_pos [[threadgroup_position_in_grid]],
                                      uint tid [[thread_index_in_threadgroup]],
                                      uint sg_id [[simdgroup_index_in_threadgroup]],
                                      uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint Q_TILE = 8u;
    constexpr float LOG2E = 1.4426950408889634f;

    const uint kv_head = tg_pos.x / p.q_groups_per_kv;
    const uint q_grp = tg_pos.x % p.q_groups_per_kv;
    const uint split_idx = tg_pos.y;
    if (kv_head >= p.num_kv_heads || split_idx >= p.num_splits) {
        return;
    }

    const uint gqa_ratio = p.num_q_heads / p.num_kv_heads;
    const uint q_start = kv_head * gqa_ratio + q_grp * Q_TILE;
    const uint q_end = min(q_start + Q_TILE, (kv_head + 1u) * gqa_ratio);
    const uint q_cnt = q_end - q_start;

    const uint tokens_per_split = (p.seq_len + p.num_splits - 1u) / p.num_splits;
    const uint t_start = split_idx * tokens_per_split;
    const uint t_end = min(t_start + tokens_per_split, p.seq_len);

    if (t_start >= t_end) {
        if (p.num_splits > 1u && tid < q_cnt) {
            const ulong ps = ulong(q_start + tid) * p.num_splits + split_idx;
            partial_m[ps] = -INFINITY;
            partial_l[ps] = 0.0f;
        }
        return;
    }

    const bool active_q = (sg_id < q_cnt);
    const uint qh = q_start + min(sg_id, q_cnt - 1u);
    const float scale_log2e = p.scale * LOG2E;

    threadgroup half4 tg_k[2][256];
    threadgroup half4 tg_v[2][256];

    const uint first_chunk = t_start / p.chunk_tokens;
    const uint last_chunk = (t_end - 1u) / p.chunk_tokens;
    const uint head_row_base = kv_head * p.chunk_tokens;

    if (ATTN_HEAD_DIM == 128u) {
        const float4 q_vec = active_q
            ? float4(*(const device half4 *)(q + ulong(qh) * 128u + lane * 4u))
            : float4(0.0f);
        float m_val = -INFINITY;
        float l_val = 0.0f;
        float4 acc = float4(0.0f);

        const uint load_tok = tid >> 5;  // 0..7
        const uint load_col = tid & 31u; // 0..31

        for (uint chunk = first_chunk; chunk <= last_chunk; chunk++) {
            const uint chunk_base = chunk * p.chunk_tokens;
            const uint slot_lo = (t_start > chunk_base) ? (t_start - chunk_base) : 0u;
            const uint slot_hi = min(t_end - chunk_base, p.chunk_tokens);
            const texture2d<half, access::read> k_tex = kv_tiles[chunk].k_tex;
            const texture2d<half, access::read> v_tex = kv_tiles[chunk].v_tex;

            uint buf = 0u;
            // Prologue: stage first 8-token tile into double buffer 0.
            if (slot_lo + load_tok < slot_hi) {
                const uint row = head_row_base + slot_lo + load_tok;
                tg_k[0][tid] = k_tex.read(uint2(load_col, row));
                tg_v[0][tid] = v_tex.read(uint2(load_col, row));
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint base_slot = slot_lo; base_slot < slot_hi; base_slot += 8u) {
                const uint n_tok = min(8u, slot_hi - base_slot);
                const uint next_base = base_slot + 8u;
                const uint next_buf = buf ^ 1u;

                // Overlap DRAM fetch of next 8-token tile with softmax/V compute on current tile.
                half4 next_k = half4(0.0h);
                half4 next_v = half4(0.0h);
                const bool has_next = (next_base + load_tok < slot_hi);
                if (has_next) {
                    const uint next_row = head_row_base + next_base + load_tok;
                    next_k = k_tex.read(uint2(load_col, next_row));
                    next_v = v_tex.read(uint2(load_col, next_row));
                }

                if (active_q) {
                    if (p.num_splits > 1u && n_tok == 8u) {
                        float s0 = simd_sum(dot(q_vec, float4(tg_k[buf][0u * 32u + lane]))) * scale_log2e;
                        float s1 = simd_sum(dot(q_vec, float4(tg_k[buf][1u * 32u + lane]))) * scale_log2e;
                        float s2 = simd_sum(dot(q_vec, float4(tg_k[buf][2u * 32u + lane]))) * scale_log2e;
                        float s3 = simd_sum(dot(q_vec, float4(tg_k[buf][3u * 32u + lane]))) * scale_log2e;
                        float s4 = simd_sum(dot(q_vec, float4(tg_k[buf][4u * 32u + lane]))) * scale_log2e;
                        float s5 = simd_sum(dot(q_vec, float4(tg_k[buf][5u * 32u + lane]))) * scale_log2e;
                        float s6 = simd_sum(dot(q_vec, float4(tg_k[buf][6u * 32u + lane]))) * scale_log2e;
                        float s7 = simd_sum(dot(q_vec, float4(tg_k[buf][7u * 32u + lane]))) * scale_log2e;

                        float tile_m = max(max(max(s0, s1), max(s2, s3)),
                                           max(max(s4, s5), max(s6, s7)));
                        float m_new = max(m_val, tile_m);
                        float alpha = exp2(m_val - m_new);
                        float p0 = exp2(s0 - m_new);
                        float p1 = exp2(s1 - m_new);
                        float p2 = exp2(s2 - m_new);
                        float p3 = exp2(s3 - m_new);
                        float p4 = exp2(s4 - m_new);
                        float p5 = exp2(s5 - m_new);
                        float p6 = exp2(s6 - m_new);
                        float p7 = exp2(s7 - m_new);

                        l_val = fma(l_val, alpha, ((p0 + p1) + (p2 + p3)) + ((p4 + p5) + (p6 + p7)));
                        acc = acc * alpha;
                        acc = fma(float4(p0), float4(tg_v[buf][0u * 32u + lane]), acc);
                        acc = fma(float4(p1), float4(tg_v[buf][1u * 32u + lane]), acc);
                        acc = fma(float4(p2), float4(tg_v[buf][2u * 32u + lane]), acc);
                        acc = fma(float4(p3), float4(tg_v[buf][3u * 32u + lane]), acc);
                        acc = fma(float4(p4), float4(tg_v[buf][4u * 32u + lane]), acc);
                        acc = fma(float4(p5), float4(tg_v[buf][5u * 32u + lane]), acc);
                        acc = fma(float4(p6), float4(tg_v[buf][6u * 32u + lane]), acc);
                        acc = fma(float4(p7), float4(tg_v[buf][7u * 32u + lane]), acc);
                        m_val = m_new;
                    } else {
                        for (uint k = 0u; k < n_tok; k++) {
                            const float score = simd_sum(dot(q_vec, float4(tg_k[buf][k * 32u + lane]))) * scale_log2e;
                            const float m_new = max(m_val, score);
                            const float alpha = exp2(m_val - m_new);
                            const float prob = exp2(score - m_new);
                            l_val = fma(l_val, alpha, prob);
                            acc = fma(acc, float4(alpha), prob * float4(tg_v[buf][k * 32u + lane]));
                            m_val = m_new;
                        }
                    }
                }

                if (has_next) {
                    tg_k[next_buf][tid] = next_k;
                    tg_v[next_buf][tid] = next_v;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                buf = next_buf;
            }
        }

        if (active_q) {
            if (p.num_splits == 1u) {
                const float inv_l = (l_val > 0.0f) ? (1.0f / l_val) : 0.0f;
                *(device half4 *)(out + ulong(qh) * 128u + lane * 4u) = half4(acc * inv_l);
            } else {
                const ulong ps = ulong(qh) * p.num_splits + split_idx;
                if (lane == 0u) {
                    partial_m[ps] = m_val;
                    partial_l[ps] = l_val;
                }
                *(device float4 *)(partial_o + ps * 128u + lane * 4u) = acc;
            }
        }
    } else {
        // ATTN_HEAD_DIM == 64: 16 texels/token; 256 threads load 16 tokens/step.
        // Half-simdgroup 0 (lanes 0..15) handles tokens 0..7; half 1 (16..31) handles 8..15.
        const uint col = lane & 15u;
        const uint half_id = lane >> 4;
        const float4 q_vec = active_q
            ? float4(*(const device half4 *)(q + ulong(qh) * 64u + col * 4u))
            : float4(0.0f);
        float m_val = -INFINITY;
        float l_val = 0.0f;
        float4 acc = float4(0.0f);

        const uint load_tok = tid >> 4;  // 0..15
        const uint load_col = tid & 15u; // 0..15

        for (uint chunk = first_chunk; chunk <= last_chunk; chunk++) {
            const uint chunk_base = chunk * p.chunk_tokens;
            const uint slot_lo = (t_start > chunk_base) ? (t_start - chunk_base) : 0u;
            const uint slot_hi = min(t_end - chunk_base, p.chunk_tokens);
            const texture2d<half, access::read> k_tex = kv_tiles[chunk].k_tex;
            const texture2d<half, access::read> v_tex = kv_tiles[chunk].v_tex;

            uint buf = 0u;
            if (slot_lo + load_tok < slot_hi) {
                const uint row = head_row_base + slot_lo + load_tok;
                tg_k[0][tid] = k_tex.read(uint2(load_col, row));
                tg_v[0][tid] = v_tex.read(uint2(load_col, row));
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint base_slot = slot_lo; base_slot < slot_hi; base_slot += 16u) {
                const uint n_tok = min(16u, slot_hi - base_slot);
                const uint next_base = base_slot + 16u;
                const uint next_buf = buf ^ 1u;

                half4 next_k = half4(0.0h);
                half4 next_v = half4(0.0h);
                const bool has_next = (next_base + load_tok < slot_hi);
                if (has_next) {
                    const uint next_row = head_row_base + next_base + load_tok;
                    next_k = k_tex.read(uint2(load_col, next_row));
                    next_v = v_tex.read(uint2(load_col, next_row));
                }

                if (active_q) {
                    if (n_tok == 16u) {
                        const uint k_off = half_id * 8u;
                        float s0 = half_simd_sum(dot(q_vec, float4(tg_k[buf][(k_off + 0u) * 16u + col]))) * scale_log2e;
                        float s1 = half_simd_sum(dot(q_vec, float4(tg_k[buf][(k_off + 1u) * 16u + col]))) * scale_log2e;
                        float s2 = half_simd_sum(dot(q_vec, float4(tg_k[buf][(k_off + 2u) * 16u + col]))) * scale_log2e;
                        float s3 = half_simd_sum(dot(q_vec, float4(tg_k[buf][(k_off + 3u) * 16u + col]))) * scale_log2e;
                        float s4 = half_simd_sum(dot(q_vec, float4(tg_k[buf][(k_off + 4u) * 16u + col]))) * scale_log2e;
                        float s5 = half_simd_sum(dot(q_vec, float4(tg_k[buf][(k_off + 5u) * 16u + col]))) * scale_log2e;
                        float s6 = half_simd_sum(dot(q_vec, float4(tg_k[buf][(k_off + 6u) * 16u + col]))) * scale_log2e;
                        float s7 = half_simd_sum(dot(q_vec, float4(tg_k[buf][(k_off + 7u) * 16u + col]))) * scale_log2e;

                        float tile_m = max(max(max(s0, s1), max(s2, s3)),
                                           max(max(s4, s5), max(s6, s7)));
                        float m_new = max(m_val, tile_m);
                        float alpha = exp2(m_val - m_new);
                        float p0 = exp2(s0 - m_new);
                        float p1 = exp2(s1 - m_new);
                        float p2 = exp2(s2 - m_new);
                        float p3 = exp2(s3 - m_new);
                        float p4 = exp2(s4 - m_new);
                        float p5 = exp2(s5 - m_new);
                        float p6 = exp2(s6 - m_new);
                        float p7 = exp2(s7 - m_new);

                        l_val = fma(l_val, alpha, ((p0 + p1) + (p2 + p3)) + ((p4 + p5) + (p6 + p7)));
                        acc = acc * alpha;
                        acc = fma(float4(p0), float4(tg_v[buf][(k_off + 0u) * 16u + col]), acc);
                        acc = fma(float4(p1), float4(tg_v[buf][(k_off + 1u) * 16u + col]), acc);
                        acc = fma(float4(p2), float4(tg_v[buf][(k_off + 2u) * 16u + col]), acc);
                        acc = fma(float4(p3), float4(tg_v[buf][(k_off + 3u) * 16u + col]), acc);
                        acc = fma(float4(p4), float4(tg_v[buf][(k_off + 4u) * 16u + col]), acc);
                        acc = fma(float4(p5), float4(tg_v[buf][(k_off + 5u) * 16u + col]), acc);
                        acc = fma(float4(p6), float4(tg_v[buf][(k_off + 6u) * 16u + col]), acc);
                        acc = fma(float4(p7), float4(tg_v[buf][(k_off + 7u) * 16u + col]), acc);
                        m_val = m_new;
                    } else {
                        for (uint k_base = 0u; k_base < n_tok; k_base += 2u) {
                            const uint k = k_base + half_id;
                            const bool valid = (k < n_tok);
                            const float4 k4 = valid ? float4(tg_k[buf][k * 16u + col]) : float4(0.0f);
                            const float score = half_simd_sum(dot(q_vec, k4)) * scale_log2e;
                            if (valid) {
                                const float m_new = max(m_val, score);
                                const float alpha = exp2(m_val - m_new);
                                const float prob = exp2(score - m_new);
                                l_val = fma(l_val, alpha, prob);
                                acc = fma(acc, float4(alpha), prob * float4(tg_v[buf][k * 16u + col]));
                                m_val = m_new;
                            }
                        }
                    }
                }

                if (has_next) {
                    tg_k[next_buf][tid] = next_k;
                    tg_v[next_buf][tid] = next_v;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                buf = next_buf;
            }
        }

        if (active_q) {
            // Merge the two 16-lane half-simdgroups with zero TG memory.
            const float other_m = simd_shuffle_xor(m_val, 16u);
            const float other_l = simd_shuffle_xor(l_val, 16u);
            const float4 other_acc = float4(simd_shuffle_xor(acc.x, 16u),
                                            simd_shuffle_xor(acc.y, 16u),
                                            simd_shuffle_xor(acc.z, 16u),
                                            simd_shuffle_xor(acc.w, 16u));
            const float m_final = max(m_val, other_m);
            const float w_self = (l_val > 0.0f) ? exp2(m_val - m_final) : 0.0f;
            const float w_other = (other_l > 0.0f) ? exp2(other_m - m_final) : 0.0f;
            const float l_final = fma(w_self, l_val, w_other * other_l);
            const float4 acc_final = fma(float4(w_self), acc, float4(w_other) * other_acc);

            if (lane < 16u) {
                if (p.num_splits == 1u) {
                    const float inv_l = (l_final > 0.0f) ? (1.0f / l_final) : 0.0f;
                    *(device half4 *)(out + ulong(qh) * 64u + col * 4u) = half4(acc_final * inv_l);
                } else {
                    const ulong ps = ulong(qh) * p.num_splits + split_idx;
                    if (lane == 0u) {
                        partial_m[ps] = m_final;
                        partial_l[ps] = l_final;
                    }
                    *(device float4 *)(partial_o + ps * 64u + col * 4u) = acc_final;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 3b: FlashDecoding Split-KV Reduction (merges num_splits > 1 partials)
//
// Grid: (num_q_heads, 1, 1) threadgroups, 32 threads.
// ---------------------------------------------------------------------------
kernel void ami_gqa_decode_attn_reduce(const device float *partial_m [[buffer(0)]],
                                       const device float *partial_l [[buffer(1)]],
                                       const device float *partial_o [[buffer(2)]],
                                       device half *out [[buffer(3)]],
                                       constant GqaDecodeParams &p [[buffer(4)]],
                                       uint qh [[threadgroup_position_in_grid]],
                                       uint lane [[thread_index_in_simdgroup]]) {
    const uint num_texels = ATTN_HEAD_DIM >> 2;
    if (qh >= p.num_q_heads || lane >= num_texels) {
        return;
    }

    const ulong head_base = ulong(qh) * p.num_splits;
    float m_max = -INFINITY;
    for (uint s = 0u; s < p.num_splits; s++) {
        if (partial_l[head_base + s] > 0.0f) {
            m_max = max(m_max, partial_m[head_base + s]);
        }
    }

    float l_tot = 0.0f;
    float4 o_tot = float4(0.0f);
    for (uint s = 0u; s < p.num_splits; s++) {
        const float ls = partial_l[head_base + s];
        if (ls > 0.0f) {
            const float w = exp2(partial_m[head_base + s] - m_max);
            l_tot = fma(ls, w, l_tot);
            const float4 po = *(const device float4 *)(partial_o + (head_base + s) * ATTN_HEAD_DIM + lane * 4u);
            o_tot = fma(po, float4(w), o_tot);
        }
    }

    const float inv_l = (l_tot > 0.0f) ? (1.0f / l_tot) : 0.0f;
    *(device half4 *)(out + ulong(qh) * ATTN_HEAD_DIM + lane * 4u) = half4(o_tot * inv_l);
}

// ---------------------------------------------------------------------------
// Spike S1 Microbenchmark Kernels (R4 / ADR-0004)
//
// Compares M4 decode-attention KV read bandwidth across:
//   S1_ACCESS_MODE == 0: RGBA16Float tex.read(uint2(x, y))
//   S1_ACCESS_MODE == 1: R16Float    4x tex.read(uint2(4*x+i, y)).x
//   S1_ACCESS_MODE == 2: Aliasing MTLBuffer half4 loads
// on Qwen3-30B-A3B geometry (32 Q heads, 4 KV heads, head_dim = 128,
// chunk_tokens = 512) for both token-major and channel-major tile layouts.
// Both layouts use zero-spill register-resident accumulators so DRAM / texture
// read bandwidth is measured cleanly without stack traffic.
// ---------------------------------------------------------------------------
kernel void ami_spike_s1_token_major(const device half *q [[buffer(0)]],
                                     const device KvTilePair *kv_tiles [[buffer(1)]],
                                     const device KvBufPair *kv_bufs [[buffer(2)]],
                                     device half *out [[buffer(3)]],
                                     constant SpikeS1Params &p [[buffer(4)]],
                                     uint2 tg_pos [[threadgroup_position_in_grid]],
                                     uint tid [[thread_index_in_threadgroup]],
                                     uint sg_id [[simdgroup_index_in_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]]) {
    constexpr float LOG2E = 1.4426950408889634f;
    const uint kv_head = tg_pos.x;
    const uint chunk = tg_pos.y;
    if (kv_head >= p.num_kv_heads || chunk >= p.num_chunks) {
        return;
    }

    const uint gqa_ratio = p.num_q_heads / p.num_kv_heads;
    const bool active_q = (sg_id < gqa_ratio);
    const uint qh = kv_head * gqa_ratio + min(sg_id, gqa_ratio - 1u);
    const float4 q_vec = active_q
        ? float4(*(const device half4 *)(q + ulong(qh) * 128u + lane * 4u))
        : float4(0.0f);
    const float scale_log2e = p.scale * LOG2E;

    float m_val = -INFINITY;
    float l_val = 0.0f;
    float4 acc = float4(0.0f);

    threadgroup half4 tg_k[2][256];
    threadgroup half4 tg_v[2][256];

    const uint head_row_base = kv_head * p.chunk_tokens;
    const uint load_tok = tid >> 5;  // 0..7
    const uint load_col = tid & 31u; // 0..31
    const texture2d<half, access::read> k_tex = kv_tiles[chunk].k_tex;
    const texture2d<half, access::read> v_tex = kv_tiles[chunk].v_tex;
    const device half4 *k_buf = kv_bufs[chunk].k_buf;
    const device half4 *v_buf = kv_bufs[chunk].v_buf;

    // Prologue: load first 8-token tile into buffer 0.
    {
        const uint row = head_row_base + load_tok;
        if (S1_ACCESS_MODE == 0u) {
            tg_k[0][tid] = k_tex.read(uint2(load_col, row));
            tg_v[0][tid] = v_tex.read(uint2(load_col, row));
        } else if (S1_ACCESS_MODE == 1u) {
            const uint x0 = load_col * 4u;
            tg_k[0][tid] = half4(k_tex.read(uint2(x0 + 0u, row)).x,
                                 k_tex.read(uint2(x0 + 1u, row)).x,
                                 k_tex.read(uint2(x0 + 2u, row)).x,
                                 k_tex.read(uint2(x0 + 3u, row)).x);
            tg_v[0][tid] = half4(v_tex.read(uint2(x0 + 0u, row)).x,
                                 v_tex.read(uint2(x0 + 1u, row)).x,
                                 v_tex.read(uint2(x0 + 2u, row)).x,
                                 v_tex.read(uint2(x0 + 3u, row)).x);
        } else {
            const ulong idx = ulong(row) * 32u + load_col;
            tg_k[0][tid] = k_buf[idx];
            tg_v[0][tid] = v_buf[idx];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint buf = 0u;
    for (uint base_slot = 0u; base_slot < p.chunk_tokens; base_slot += 8u) {
        const uint next_base = base_slot + 8u;
        const uint next_buf = buf ^ 1u;
        const bool has_next = (next_base < p.chunk_tokens);

        half4 next_k = half4(0.0h);
        half4 next_v = half4(0.0h);
        if (has_next) {
            const uint next_row = head_row_base + next_base + load_tok;
            if (S1_ACCESS_MODE == 0u) {
                next_k = k_tex.read(uint2(load_col, next_row));
                next_v = v_tex.read(uint2(load_col, next_row));
            } else if (S1_ACCESS_MODE == 1u) {
                const uint x0 = load_col * 4u;
                next_k = half4(k_tex.read(uint2(x0 + 0u, next_row)).x,
                               k_tex.read(uint2(x0 + 1u, next_row)).x,
                               k_tex.read(uint2(x0 + 2u, next_row)).x,
                               k_tex.read(uint2(x0 + 3u, next_row)).x);
                next_v = half4(v_tex.read(uint2(x0 + 0u, next_row)).x,
                               v_tex.read(uint2(x0 + 1u, next_row)).x,
                               v_tex.read(uint2(x0 + 2u, next_row)).x,
                               v_tex.read(uint2(x0 + 3u, next_row)).x);
            } else {
                const ulong idx = ulong(next_row) * 32u + load_col;
                next_k = k_buf[idx];
                next_v = v_buf[idx];
            }
        }

        if (active_q) {
            float s0 = simd_sum(dot(q_vec, float4(tg_k[buf][0u * 32u + lane]))) * scale_log2e;
            float s1 = simd_sum(dot(q_vec, float4(tg_k[buf][1u * 32u + lane]))) * scale_log2e;
            float s2 = simd_sum(dot(q_vec, float4(tg_k[buf][2u * 32u + lane]))) * scale_log2e;
            float s3 = simd_sum(dot(q_vec, float4(tg_k[buf][3u * 32u + lane]))) * scale_log2e;
            float s4 = simd_sum(dot(q_vec, float4(tg_k[buf][4u * 32u + lane]))) * scale_log2e;
            float s5 = simd_sum(dot(q_vec, float4(tg_k[buf][5u * 32u + lane]))) * scale_log2e;
            float s6 = simd_sum(dot(q_vec, float4(tg_k[buf][6u * 32u + lane]))) * scale_log2e;
            float s7 = simd_sum(dot(q_vec, float4(tg_k[buf][7u * 32u + lane]))) * scale_log2e;

            float tile_m = max(max(max(s0, s1), max(s2, s3)),
                               max(max(s4, s5), max(s6, s7)));
            float m_new = max(m_val, tile_m);
            float alpha = exp2(m_val - m_new);
            float p0 = exp2(s0 - m_new);
            float p1 = exp2(s1 - m_new);
            float p2 = exp2(s2 - m_new);
            float p3 = exp2(s3 - m_new);
            float p4 = exp2(s4 - m_new);
            float p5 = exp2(s5 - m_new);
            float p6 = exp2(s6 - m_new);
            float p7 = exp2(s7 - m_new);

            l_val = fma(l_val, alpha, ((p0 + p1) + (p2 + p3)) + ((p4 + p5) + (p6 + p7)));
            acc = acc * alpha;
            acc = fma(float4(p0), float4(tg_v[buf][0u * 32u + lane]), acc);
            acc = fma(float4(p1), float4(tg_v[buf][1u * 32u + lane]), acc);
            acc = fma(float4(p2), float4(tg_v[buf][2u * 32u + lane]), acc);
            acc = fma(float4(p3), float4(tg_v[buf][3u * 32u + lane]), acc);
            acc = fma(float4(p4), float4(tg_v[buf][4u * 32u + lane]), acc);
            acc = fma(float4(p5), float4(tg_v[buf][5u * 32u + lane]), acc);
            acc = fma(float4(p6), float4(tg_v[buf][6u * 32u + lane]), acc);
            acc = fma(float4(p7), float4(tg_v[buf][7u * 32u + lane]), acc);
            m_val = m_new;
        }

        if (has_next) {
            tg_k[next_buf][tid] = next_k;
            tg_v[next_buf][tid] = next_v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf = next_buf;
    }

    if (active_q) {
        const ulong out_idx = (ulong(qh) * p.num_chunks + chunk) * 128u + lane * 4u;
        *(device half4 *)(out + out_idx) = half4(acc / l_val);
    }
}

// Channel-major layout: row = kv_head * 128 + d, col = token (512 f16/row).
// 128 threads/threadgroup: each thread owns 4 contiguous tokens [4*tid..4*tid+3],
// reading 100% coalesced rows along the token axis for each channel d = 0..127.
// Explicitly unrolled 8 Q-head registers (zero dynamic array spills).
kernel void ami_spike_s1_channel_major(const device half *q [[buffer(0)]],
                                       const device KvTilePair *kv_tiles [[buffer(1)]],
                                       const device KvBufPair *kv_bufs [[buffer(2)]],
                                       device half *out [[buffer(3)]],
                                       constant SpikeS1Params &p [[buffer(4)]],
                                       uint2 tg_pos [[threadgroup_position_in_grid]],
                                       uint tid [[thread_index_in_threadgroup]],
                                       uint sg_id [[simdgroup_index_in_threadgroup]],
                                       uint lane [[thread_index_in_simdgroup]]) {
    constexpr float LOG2E = 1.4426950408889634f;
    const uint kv_head = tg_pos.x;
    const uint chunk = tg_pos.y;
    if (kv_head >= p.num_kv_heads || chunk >= p.num_chunks) {
        return;
    }

    const uint gqa_ratio = p.num_q_heads / p.num_kv_heads;
    const uint q_start = kv_head * gqa_ratio;
    const float scale_log2e = p.scale * LOG2E;

    threadgroup half tg_q[8u * 128u];
    for (uint g = 0u; g < 8u; g++) {
        const uint qh = q_start + min(g, gqa_ratio - 1u);
        tg_q[g * 128u + tid] = q[ulong(qh) * 128u + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 s0 = float4(0.0f), s1 = float4(0.0f), s2 = float4(0.0f), s3 = float4(0.0f);
    float4 s4 = float4(0.0f), s5 = float4(0.0f), s6 = float4(0.0f), s7 = float4(0.0f);

    const uint head_row_base = kv_head * 128u;
    const uint row_vec_stride = p.chunk_tokens >> 2; // 128 half4s/row for 512 tokens
    const texture2d<half, access::read> k_tex = kv_tiles[chunk].k_tex;
    const texture2d<half, access::read> v_tex = kv_tiles[chunk].v_tex;
    const device half4 *k_buf = kv_bufs[chunk].k_buf;
    const device half4 *v_buf = kv_bufs[chunk].v_buf;

    // Pass 1: Read all 128 K rows once, accumulating Q*K across all 8 Q heads in registers.
    for (uint d = 0u; d < 128u; d++) {
        const uint row = head_row_base + d;
        half4 k_raw;
        if (S1_ACCESS_MODE == 0u) {
            k_raw = k_tex.read(uint2(tid, row));
        } else if (S1_ACCESS_MODE == 1u) {
            const uint x0 = tid * 4u;
            k_raw = half4(k_tex.read(uint2(x0 + 0u, row)).x,
                          k_tex.read(uint2(x0 + 1u, row)).x,
                          k_tex.read(uint2(x0 + 2u, row)).x,
                          k_tex.read(uint2(x0 + 3u, row)).x);
        } else {
            k_raw = k_buf[ulong(row) * row_vec_stride + tid];
        }
        const float4 k4 = float4(k_raw);
        s0 = fma(k4, float4(float(tg_q[0u * 128u + d])), s0);
        s1 = fma(k4, float4(float(tg_q[1u * 128u + d])), s1);
        s2 = fma(k4, float4(float(tg_q[2u * 128u + d])), s2);
        s3 = fma(k4, float4(float(tg_q[3u * 128u + d])), s3);
        s4 = fma(k4, float4(float(tg_q[4u * 128u + d])), s4);
        s5 = fma(k4, float4(float(tg_q[5u * 128u + d])), s5);
        s6 = fma(k4, float4(float(tg_q[6u * 128u + d])), s6);
        s7 = fma(k4, float4(float(tg_q[7u * 128u + d])), s7);
    }

    s0 *= scale_log2e; s1 *= scale_log2e; s2 *= scale_log2e; s3 *= scale_log2e;
    s4 *= scale_log2e; s5 *= scale_log2e; s6 *= scale_log2e; s7 *= scale_log2e;

    threadgroup float tg_red[4u * 8u];
    {
        float m0 = simd_max(max(max(s0.x, s0.y), max(s0.z, s0.w)));
        float m1 = simd_max(max(max(s1.x, s1.y), max(s1.z, s1.w)));
        float m2 = simd_max(max(max(s2.x, s2.y), max(s2.z, s2.w)));
        float m3 = simd_max(max(max(s3.x, s3.y), max(s3.z, s3.w)));
        float m4 = simd_max(max(max(s4.x, s4.y), max(s4.z, s4.w)));
        float m5 = simd_max(max(max(s5.x, s5.y), max(s5.z, s5.w)));
        float m6 = simd_max(max(max(s6.x, s6.y), max(s6.z, s6.w)));
        float m7 = simd_max(max(max(s7.x, s7.y), max(s7.z, s7.w)));
        if (lane == 0u) {
            tg_red[sg_id * 8u + 0u] = m0; tg_red[sg_id * 8u + 1u] = m1;
            tg_red[sg_id * 8u + 2u] = m2; tg_red[sg_id * 8u + 3u] = m3;
            tg_red[sg_id * 8u + 4u] = m4; tg_red[sg_id * 8u + 5u] = m5;
            tg_red[sg_id * 8u + 6u] = m6; tg_red[sg_id * 8u + 7u] = m7;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 p0 = exp2(s0 - float4(max(max(tg_red[0], tg_red[8]),  max(tg_red[16], tg_red[24]))));
    float4 p1 = exp2(s1 - float4(max(max(tg_red[1], tg_red[9]),  max(tg_red[17], tg_red[25]))));
    float4 p2 = exp2(s2 - float4(max(max(tg_red[2], tg_red[10]), max(tg_red[18], tg_red[26]))));
    float4 p3 = exp2(s3 - float4(max(max(tg_red[3], tg_red[11]), max(tg_red[19], tg_red[27]))));
    float4 p4 = exp2(s4 - float4(max(max(tg_red[4], tg_red[12]), max(tg_red[20], tg_red[28]))));
    float4 p5 = exp2(s5 - float4(max(max(tg_red[5], tg_red[13]), max(tg_red[21], tg_red[29]))));
    float4 p6 = exp2(s6 - float4(max(max(tg_red[6], tg_red[14]), max(tg_red[22], tg_red[30]))));
    float4 p7 = exp2(s7 - float4(max(max(tg_red[7], tg_red[15]), max(tg_red[23], tg_red[31]))));

    {
        float l0 = simd_sum((p0.x + p0.y) + (p0.z + p0.w));
        float l1 = simd_sum((p1.x + p1.y) + (p1.z + p1.w));
        float l2 = simd_sum((p2.x + p2.y) + (p2.z + p2.w));
        float l3 = simd_sum((p3.x + p3.y) + (p3.z + p3.w));
        float l4 = simd_sum((p4.x + p4.y) + (p4.z + p4.w));
        float l5 = simd_sum((p5.x + p5.y) + (p5.z + p5.w));
        float l6 = simd_sum((p6.x + p6.y) + (p6.z + p6.w));
        float l7 = simd_sum((p7.x + p7.y) + (p7.z + p7.w));
        if (lane == 0u) {
            tg_red[sg_id * 8u + 0u] = l0; tg_red[sg_id * 8u + 1u] = l1;
            tg_red[sg_id * 8u + 2u] = l2; tg_red[sg_id * 8u + 3u] = l3;
            tg_red[sg_id * 8u + 4u] = l4; tg_red[sg_id * 8u + 5u] = l5;
            tg_red[sg_id * 8u + 6u] = l6; tg_red[sg_id * 8u + 7u] = l7;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    p0 /= (tg_red[0] + tg_red[8]  + tg_red[16] + tg_red[24]);
    p1 /= (tg_red[1] + tg_red[9]  + tg_red[17] + tg_red[25]);
    p2 /= (tg_red[2] + tg_red[10] + tg_red[18] + tg_red[26]);
    p3 /= (tg_red[3] + tg_red[11] + tg_red[19] + tg_red[27]);
    p4 /= (tg_red[4] + tg_red[12] + tg_red[20] + tg_red[28]);
    p5 /= (tg_red[5] + tg_red[13] + tg_red[21] + tg_red[29]);
    p6 /= (tg_red[6] + tg_red[14] + tg_red[22] + tg_red[30]);
    p7 /= (tg_red[7] + tg_red[15] + tg_red[23] + tg_red[31]);

    // Pass 2: Read all 128 V rows once in batches of 4 channels to minimize barriers.
    threadgroup float tg_vred[4u * 4u * 8u]; // [channel_in_batch][sg_id][g]
    for (uint d_base = 0u; d_base < 128u; d_base += 4u) {
        for (uint dd = 0u; dd < 4u; dd++) {
            const uint row = head_row_base + d_base + dd;
            half4 v_raw;
            if (S1_ACCESS_MODE == 0u) {
                v_raw = v_tex.read(uint2(tid, row));
            } else if (S1_ACCESS_MODE == 1u) {
                const uint x0 = tid * 4u;
                v_raw = half4(v_tex.read(uint2(x0 + 0u, row)).x,
                              v_tex.read(uint2(x0 + 1u, row)).x,
                              v_tex.read(uint2(x0 + 2u, row)).x,
                              v_tex.read(uint2(x0 + 3u, row)).x);
            } else {
                v_raw = v_buf[ulong(row) * row_vec_stride + tid];
            }
            const float4 v4 = float4(v_raw);
            float r0 = simd_sum(dot(p0, v4));
            float r1 = simd_sum(dot(p1, v4));
            float r2 = simd_sum(dot(p2, v4));
            float r3 = simd_sum(dot(p3, v4));
            float r4 = simd_sum(dot(p4, v4));
            float r5 = simd_sum(dot(p5, v4));
            float r6 = simd_sum(dot(p6, v4));
            float r7 = simd_sum(dot(p7, v4));
            if (lane == 0u) {
                const uint base = (dd * 4u + sg_id) * 8u;
                tg_vred[base + 0u] = r0; tg_vred[base + 1u] = r1;
                tg_vred[base + 2u] = r2; tg_vred[base + 3u] = r3;
                tg_vred[base + 4u] = r4; tg_vred[base + 5u] = r5;
                tg_vred[base + 6u] = r6; tg_vred[base + 7u] = r7;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 32u) {
            const uint dd = tid >> 3;  // 0..3
            const uint g = tid & 7u;   // 0..7
            if (g < gqa_ratio) {
                const uint b = dd * 32u + g;
                const float tot = tg_vred[b + 0u] + tg_vred[b + 8u] +
                                  tg_vred[b + 16u] + tg_vred[b + 24u];
                const ulong out_idx = (ulong(q_start + g) * p.num_chunks + chunk) * 128u + d_base + dd;
                out[out_idx] = half(tot);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ===========================================================================
// Milestone M3 (G5): Speculative Tree-Attention Verification & KV Commit
// ===========================================================================

struct TreeQkNormRopeParams {
    uint num_nodes;
    uint num_q_heads;
    uint num_kv_heads;
    uint head_dim;
    float rms_eps;
    float rope_theta;
    uint apply_qk_norm;
    uint _pad;
};

struct GqaTreeVerifyParams {
    uint num_nodes;
    uint num_q_heads;
    uint num_kv_heads;
    uint base_pos;
    uint chunk_tokens;
    uint num_splits;
    uint q_groups_per_kv;
    float scale;
};

struct KvTreeCommitParams {
    uint num_kv_heads;
    uint head_dim;
    uint base_pos;
    uint chunk_tokens;
    uint accepted_count;
    uint use_node_buffer;
    uint has_second_chunk;
    uint _pad;
    ushort inline_nodes[64];
};

static_assert(sizeof(TreeQkNormRopeParams) == 32, "TreeQkNormRopeParams must be 32 bytes");
static_assert(sizeof(GqaTreeVerifyParams) == 32, "GqaTreeVerifyParams must be 32 bytes");
static_assert(sizeof(KvTreeCommitParams) == 160, "KvTreeCommitParams must be 160 bytes");

// ---------------------------------------------------------------------------
// Kernel 4: Batched Tree QK-Norm + NeoX Half-Split RoPE (does NOT write
// committed IOSurface KvTiles yet; writes rotated K and V to scratch tree KV).
//
// Specialized on NORM_W_BF16 (function_constant(2)).
// Grid: (num_q_heads + num_kv_heads, num_nodes, 1) threadgroups, 32 threads.
// ---------------------------------------------------------------------------
kernel void ami_tree_qknorm_rope(device half *q [[buffer(0)]],
                                 const device half *k [[buffer(1)]],
                                 const device half *v [[buffer(2)]],
                                 const device uchar *q_norm [[buffer(3)]],
                                 const device uchar *k_norm [[buffer(4)]],
                                 const device uint *positions [[buffer(5)]],
                                 device half *k_tree_out [[buffer(6)]],
                                 device half *v_tree_out [[buffer(7)]],
                                 constant TreeQkNormRopeParams &p [[buffer(8)]],
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
    const uint half_dim = d_dim >> 1;
    const uint pos = positions[node];

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
            vec[d] = (vec[d] * inv_rms) * norm_w1(nw, d, NORM_W_BF16);
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // NeoX half-split RoPE: exact match with ami_qknorm_rope_kv_write_norm.
    const float log_theta = log(p.rope_theta);
    const float inv_d = 2.0f / float(d_dim);
    const float fpos = float(pos);
    for (uint j = lane; j < half_dim; j += 32u) {
        const float inv_freq = exp(-log_theta * (float(j) * inv_d));
        const float angle = fpos * inv_freq;
        const float c = cos(angle);
        const float s = sin(angle);
        const float x0 = vec[j];
        const float x1 = vec[j + half_dim];
        vec[j] = fma(x0, c, -x1 * s);
        vec[j + half_dim] = fma(x1, c, x0 * s);
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

// ---------------------------------------------------------------------------
// Kernel 5a: Speculative GQA Tree-Attention Verification (G5)
//
// Specialized on ATTN_HEAD_DIM ({64, 128}) and TREE_MAX_NODES ({16, 32, 64})
// (ADR-0014). Combines committed prefix IOSurface RGBA16Float KV tiles
// (`0..base_pos`, with parallel split-KV when `num_splits > 1`) and masked
// tree-local KV (`j in 0..num_nodes` with `(ancestor_masks[i] & (1ul<<j)) != 0`)
// using exact FP32 base-2 online softmax.
//
// Grid: (num_kv_heads * q_groups_per_kv, num_splits, 1) threadgroups, 256 threads.
// ---------------------------------------------------------------------------
kernel void ami_gqa_tree_verify_attn_split(const device half *q [[buffer(0)]],
                                           const device KvTilePair *kv_tiles [[buffer(1)]],
                                           const device half *k_tree [[buffer(2)]],
                                           const device half *v_tree [[buffer(3)]],
                                           const device ulong *ancestor_masks [[buffer(4)]],
                                           device half *out [[buffer(5)]],
                                           device float *partial_m [[buffer(6)]],
                                           device float *partial_l [[buffer(7)]],
                                           device float *partial_o [[buffer(8)]],
                                           constant GqaTreeVerifyParams &p [[buffer(9)]],
                                           uint2 tg_pos [[threadgroup_position_in_grid]],
                                           uint tid [[thread_index_in_threadgroup]],
                                           uint sg_id [[simdgroup_index_in_threadgroup]],
                                           uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint Q_TILE = 8u;
    constexpr float LOG2E = 1.4426950408889634f;

    const uint kv_head = tg_pos.x / p.q_groups_per_kv;
    const uint q_grp = tg_pos.x % p.q_groups_per_kv;
    const uint split_idx = tg_pos.y;
    if (kv_head >= p.num_kv_heads || split_idx >= p.num_splits) {
        return;
    }

    const uint gqa_ratio = p.num_q_heads / p.num_kv_heads;
    const uint total_q = p.num_nodes * gqa_ratio;
    const uint q_start_flat = q_grp * Q_TILE;
    if (q_start_flat >= total_q) {
        return;
    }
    const uint q_cnt = min(Q_TILE, total_q - q_start_flat);

    uint t_start = 0u;
    uint t_end = 0u;
    if (p.base_pos > 0u) {
        const uint tokens_per_split = (p.base_pos + p.num_splits - 1u) / p.num_splits;
        t_start = split_idx * tokens_per_split;
        t_end = min(t_start + tokens_per_split, p.base_pos);
    }

    const bool do_prefix = (t_start < t_end);
    const bool do_tree = (split_idx == 0u);
    if (!do_prefix && !do_tree) {
        if (p.num_splits > 1u && tid < q_cnt) {
            const uint flat_q = q_start_flat + tid;
            const uint node = flat_q / gqa_ratio;
            const uint qh = kv_head * gqa_ratio + (flat_q - node * gqa_ratio);
            const ulong ps = (ulong(node) * p.num_q_heads + qh) * p.num_splits + split_idx;
            partial_m[ps] = -INFINITY;
            partial_l[ps] = 0.0f;
        }
        return;
    }

    const bool active_q = (sg_id < q_cnt);
    const uint safe_flat = q_start_flat + min(sg_id, q_cnt - 1u);
    const uint node = safe_flat / gqa_ratio;
    const uint qh = kv_head * gqa_ratio + (safe_flat - node * gqa_ratio);
    const ulong mask = active_q ? ancestor_masks[node] : 0ul;
    const float scale_log2e = p.scale * LOG2E;
    const uint head_row_base = kv_head * p.chunk_tokens;

    threadgroup half4 tg_k[2][256];
    threadgroup half4 tg_v[2][256];

    if (ATTN_HEAD_DIM == 128u) {
        const ulong q_off = (ulong(node) * p.num_q_heads + qh) * 128u + lane * 4u;
        const float4 q_vec = active_q ? float4(*(const device half4 *)(q + q_off)) : float4(0.0f);
        float m_val = -INFINITY;
        float l_val = 0.0f;
        float4 acc = float4(0.0f);

        const uint load_tok = tid >> 5;  // 0..7
        const uint load_col = tid & 31u; // 0..31

        if (do_prefix) {
            const uint first_chunk = t_start / p.chunk_tokens;
            const uint last_chunk = (t_end - 1u) / p.chunk_tokens;

            for (uint chunk = first_chunk; chunk <= last_chunk; chunk++) {
                const uint chunk_base = chunk * p.chunk_tokens;
                const uint slot_lo = (t_start > chunk_base) ? (t_start - chunk_base) : 0u;
                const uint slot_hi = min(t_end - chunk_base, p.chunk_tokens);
                const texture2d<half, access::read> k_tex = kv_tiles[chunk].k_tex;
                const texture2d<half, access::read> v_tex = kv_tiles[chunk].v_tex;

                uint buf = 0u;
                if (slot_lo + load_tok < slot_hi) {
                    const uint row = head_row_base + slot_lo + load_tok;
                    tg_k[0][tid] = k_tex.read(uint2(load_col, row));
                    tg_v[0][tid] = v_tex.read(uint2(load_col, row));
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint base_slot = slot_lo; base_slot < slot_hi; base_slot += 8u) {
                    const uint n_tok = min(8u, slot_hi - base_slot);
                    const uint next_base = base_slot + 8u;
                    const uint next_buf = buf ^ 1u;

                    half4 next_k = half4(0.0h);
                    half4 next_v = half4(0.0h);
                    const bool has_next = (next_base + load_tok < slot_hi);
                    if (has_next) {
                        const uint next_row = head_row_base + next_base + load_tok;
                        next_k = k_tex.read(uint2(load_col, next_row));
                        next_v = v_tex.read(uint2(load_col, next_row));
                    }

                    if (active_q) {
                        if (p.num_splits > 1u && n_tok == 8u) {
                            float s0 = simd_sum(dot(q_vec, float4(tg_k[buf][0u * 32u + lane]))) * scale_log2e;
                            float s1 = simd_sum(dot(q_vec, float4(tg_k[buf][1u * 32u + lane]))) * scale_log2e;
                            float s2 = simd_sum(dot(q_vec, float4(tg_k[buf][2u * 32u + lane]))) * scale_log2e;
                            float s3 = simd_sum(dot(q_vec, float4(tg_k[buf][3u * 32u + lane]))) * scale_log2e;
                            float s4 = simd_sum(dot(q_vec, float4(tg_k[buf][4u * 32u + lane]))) * scale_log2e;
                            float s5 = simd_sum(dot(q_vec, float4(tg_k[buf][5u * 32u + lane]))) * scale_log2e;
                            float s6 = simd_sum(dot(q_vec, float4(tg_k[buf][6u * 32u + lane]))) * scale_log2e;
                            float s7 = simd_sum(dot(q_vec, float4(tg_k[buf][7u * 32u + lane]))) * scale_log2e;

                            float tile_m = max(max(max(s0, s1), max(s2, s3)),
                                               max(max(s4, s5), max(s6, s7)));
                            float m_new = max(m_val, tile_m);
                            float alpha = (l_val > 0.0f) ? exp2(m_val - m_new) : 0.0f;
                            float p0 = exp2(s0 - m_new);
                            float p1 = exp2(s1 - m_new);
                            float p2 = exp2(s2 - m_new);
                            float p3 = exp2(s3 - m_new);
                            float p4 = exp2(s4 - m_new);
                            float p5 = exp2(s5 - m_new);
                            float p6 = exp2(s6 - m_new);
                            float p7 = exp2(s7 - m_new);

                            l_val = fma(l_val, alpha, ((p0 + p1) + (p2 + p3)) + ((p4 + p5) + (p6 + p7)));
                            acc = acc * alpha;
                            acc = fma(float4(p0), float4(tg_v[buf][0u * 32u + lane]), acc);
                            acc = fma(float4(p1), float4(tg_v[buf][1u * 32u + lane]), acc);
                            acc = fma(float4(p2), float4(tg_v[buf][2u * 32u + lane]), acc);
                            acc = fma(float4(p3), float4(tg_v[buf][3u * 32u + lane]), acc);
                            acc = fma(float4(p4), float4(tg_v[buf][4u * 32u + lane]), acc);
                            acc = fma(float4(p5), float4(tg_v[buf][5u * 32u + lane]), acc);
                            acc = fma(float4(p6), float4(tg_v[buf][6u * 32u + lane]), acc);
                            acc = fma(float4(p7), float4(tg_v[buf][7u * 32u + lane]), acc);
                            m_val = m_new;
                        } else {
                            for (uint k = 0u; k < n_tok; k++) {
                                const float score = simd_sum(dot(q_vec, float4(tg_k[buf][k * 32u + lane]))) * scale_log2e;
                                const float m_new = max(m_val, score);
                                const float alpha = (l_val > 0.0f) ? exp2(m_val - m_new) : 0.0f;
                                const float prob = exp2(score - m_new);
                                l_val = fma(l_val, alpha, prob);
                                acc = fma(acc, float4(alpha), prob * float4(tg_v[buf][k * 32u + lane]));
                                m_val = m_new;
                            }
                        }
                    }

                    if (has_next) {
                        tg_k[next_buf][tid] = next_k;
                        tg_v[next_buf][tid] = next_v;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    buf = next_buf;
                }
            }
        }

        if (do_tree) {
            for (uint t_base = 0u; t_base < TREE_MAX_NODES; t_base += 8u) {
                if (t_base >= p.num_nodes) {
                    break;
                }
                const uint n_tok = min(8u, p.num_nodes - t_base);
                if (load_tok < n_tok) {
                    const uint node_j = t_base + load_tok;
                    const ulong kv_off = (ulong(node_j) * p.num_kv_heads + kv_head) * 128u + load_col * 4u;
                    tg_k[0][tid] = *(const device half4 *)(k_tree + kv_off);
                    tg_v[0][tid] = *(const device half4 *)(v_tree + kv_off);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (active_q) {
                    const uint tile_mask = uint((mask >> t_base) & 0xFFul);
                    if (tile_mask != 0u) {
                        for (uint k = 0u; k < n_tok; k++) {
                            if (((tile_mask >> k) & 1u) != 0u) {
                                const float score = simd_sum(dot(q_vec, float4(tg_k[0][k * 32u + lane]))) * scale_log2e;
                                const float m_new = max(m_val, score);
                                const float alpha = (l_val > 0.0f) ? exp2(m_val - m_new) : 0.0f;
                                const float prob = exp2(score - m_new);
                                l_val = fma(l_val, alpha, prob);
                                acc = fma(acc, float4(alpha), prob * float4(tg_v[0][k * 32u + lane]));
                                m_val = m_new;
                            }
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        if (active_q) {
            if (p.num_splits == 1u) {
                const float inv_l = (l_val > 0.0f) ? (1.0f / l_val) : 0.0f;
                *(device half4 *)(out + (ulong(node) * p.num_q_heads + qh) * 128u + lane * 4u) = half4(acc * inv_l);
            } else {
                const ulong ps = (ulong(node) * p.num_q_heads + qh) * p.num_splits + split_idx;
                if (lane == 0u) {
                    partial_m[ps] = m_val;
                    partial_l[ps] = l_val;
                }
                *(device float4 *)(partial_o + ps * 128u + lane * 4u) = acc;
            }
        }
    } else {
        // ATTN_HEAD_DIM == 64: 16 texels/token; 256 threads load 16 tokens/step.
        const uint col = lane & 15u;
        const uint half_id = lane >> 4;
        const ulong q_off = (ulong(node) * p.num_q_heads + qh) * 64u + col * 4u;
        const float4 q_vec = active_q ? float4(*(const device half4 *)(q + q_off)) : float4(0.0f);
        float m_val = -INFINITY;
        float l_val = 0.0f;
        float4 acc = float4(0.0f);

        const uint load_tok = tid >> 4;  // 0..15
        const uint load_col = tid & 15u; // 0..15

        if (do_prefix) {
            const uint first_chunk = t_start / p.chunk_tokens;
            const uint last_chunk = (t_end - 1u) / p.chunk_tokens;

            for (uint chunk = first_chunk; chunk <= last_chunk; chunk++) {
                const uint chunk_base = chunk * p.chunk_tokens;
                const uint slot_lo = (t_start > chunk_base) ? (t_start - chunk_base) : 0u;
                const uint slot_hi = min(t_end - chunk_base, p.chunk_tokens);
                const texture2d<half, access::read> k_tex = kv_tiles[chunk].k_tex;
                const texture2d<half, access::read> v_tex = kv_tiles[chunk].v_tex;

                uint buf = 0u;
                if (slot_lo + load_tok < slot_hi) {
                    const uint row = head_row_base + slot_lo + load_tok;
                    tg_k[0][tid] = k_tex.read(uint2(load_col, row));
                    tg_v[0][tid] = v_tex.read(uint2(load_col, row));
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint base_slot = slot_lo; base_slot < slot_hi; base_slot += 16u) {
                    const uint n_tok = min(16u, slot_hi - base_slot);
                    const uint next_base = base_slot + 16u;
                    const uint next_buf = buf ^ 1u;

                    half4 next_k = half4(0.0h);
                    half4 next_v = half4(0.0h);
                    const bool has_next = (next_base + load_tok < slot_hi);
                    if (has_next) {
                        const uint next_row = head_row_base + next_base + load_tok;
                        next_k = k_tex.read(uint2(load_col, next_row));
                        next_v = v_tex.read(uint2(load_col, next_row));
                    }

                    if (active_q) {
                        if (n_tok == 16u) {
                            const uint k_off = half_id * 8u;
                            float s0 = half_simd_sum(dot(q_vec, float4(tg_k[buf][(k_off + 0u) * 16u + col]))) * scale_log2e;
                            float s1 = half_simd_sum(dot(q_vec, float4(tg_k[buf][(k_off + 1u) * 16u + col]))) * scale_log2e;
                            float s2 = half_simd_sum(dot(q_vec, float4(tg_k[buf][(k_off + 2u) * 16u + col]))) * scale_log2e;
                            float s3 = half_simd_sum(dot(q_vec, float4(tg_k[buf][(k_off + 3u) * 16u + col]))) * scale_log2e;
                            float s4 = half_simd_sum(dot(q_vec, float4(tg_k[buf][(k_off + 4u) * 16u + col]))) * scale_log2e;
                            float s5 = half_simd_sum(dot(q_vec, float4(tg_k[buf][(k_off + 5u) * 16u + col]))) * scale_log2e;
                            float s6 = half_simd_sum(dot(q_vec, float4(tg_k[buf][(k_off + 6u) * 16u + col]))) * scale_log2e;
                            float s7 = half_simd_sum(dot(q_vec, float4(tg_k[buf][(k_off + 7u) * 16u + col]))) * scale_log2e;

                            float tile_m = max(max(max(s0, s1), max(s2, s3)),
                                               max(max(s4, s5), max(s6, s7)));
                            float m_new = max(m_val, tile_m);
                            float alpha = (l_val > 0.0f) ? exp2(m_val - m_new) : 0.0f;
                            float p0 = exp2(s0 - m_new);
                            float p1 = exp2(s1 - m_new);
                            float p2 = exp2(s2 - m_new);
                            float p3 = exp2(s3 - m_new);
                            float p4 = exp2(s4 - m_new);
                            float p5 = exp2(s5 - m_new);
                            float p6 = exp2(s6 - m_new);
                            float p7 = exp2(s7 - m_new);

                            l_val = fma(l_val, alpha, ((p0 + p1) + (p2 + p3)) + ((p4 + p5) + (p6 + p7)));
                            acc = acc * alpha;
                            acc = fma(float4(p0), float4(tg_v[buf][(k_off + 0u) * 16u + col]), acc);
                            acc = fma(float4(p1), float4(tg_v[buf][(k_off + 1u) * 16u + col]), acc);
                            acc = fma(float4(p2), float4(tg_v[buf][(k_off + 2u) * 16u + col]), acc);
                            acc = fma(float4(p3), float4(tg_v[buf][(k_off + 3u) * 16u + col]), acc);
                            acc = fma(float4(p4), float4(tg_v[buf][(k_off + 4u) * 16u + col]), acc);
                            acc = fma(float4(p5), float4(tg_v[buf][(k_off + 5u) * 16u + col]), acc);
                            acc = fma(float4(p6), float4(tg_v[buf][(k_off + 6u) * 16u + col]), acc);
                            acc = fma(float4(p7), float4(tg_v[buf][(k_off + 7u) * 16u + col]), acc);
                            m_val = m_new;
                        } else {
                            for (uint k_base = 0u; k_base < n_tok; k_base += 2u) {
                                const uint k = k_base + half_id;
                                const bool valid = (k < n_tok);
                                const float4 k4 = valid ? float4(tg_k[buf][k * 16u + col]) : float4(0.0f);
                                const float score = half_simd_sum(dot(q_vec, k4)) * scale_log2e;
                                if (valid) {
                                    const float m_new = max(m_val, score);
                                    const float alpha = (l_val > 0.0f) ? exp2(m_val - m_new) : 0.0f;
                                    const float prob = exp2(score - m_new);
                                    l_val = fma(l_val, alpha, prob);
                                    acc = fma(acc, float4(alpha), prob * float4(tg_v[buf][k * 16u + col]));
                                    m_val = m_new;
                                }
                            }
                        }
                    }

                    if (has_next) {
                        tg_k[next_buf][tid] = next_k;
                        tg_v[next_buf][tid] = next_v;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    buf = next_buf;
                }
            }
        }

        if (do_tree) {
            for (uint t_base = 0u; t_base < TREE_MAX_NODES; t_base += 16u) {
                if (t_base >= p.num_nodes) {
                    break;
                }
                const uint n_tok = min(16u, p.num_nodes - t_base);
                if (load_tok < n_tok) {
                    const uint node_j = t_base + load_tok;
                    const ulong kv_off = (ulong(node_j) * p.num_kv_heads + kv_head) * 64u + load_col * 4u;
                    tg_k[0][tid] = *(const device half4 *)(k_tree + kv_off);
                    tg_v[0][tid] = *(const device half4 *)(v_tree + kv_off);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (active_q) {
                    const uint tile_mask = uint((mask >> t_base) & 0xFFFFul);
                    if (tile_mask != 0u) {
                        for (uint k_base = 0u; k_base < n_tok; k_base += 2u) {
                            const uint k = k_base + half_id;
                            const bool valid = (k < n_tok) && (((tile_mask >> k) & 1u) != 0u);
                            const float4 k4 = valid ? float4(tg_k[0][k * 16u + col]) : float4(0.0f);
                            const float score = half_simd_sum(dot(q_vec, k4)) * scale_log2e;
                            if (valid) {
                                const float m_new = max(m_val, score);
                                const float alpha = (l_val > 0.0f) ? exp2(m_val - m_new) : 0.0f;
                                const float prob = exp2(score - m_new);
                                l_val = fma(l_val, alpha, prob);
                                acc = fma(acc, float4(alpha), prob * float4(tg_v[0][k * 16u + col]));
                                m_val = m_new;
                            }
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        if (active_q) {
            const float other_m = simd_shuffle_xor(m_val, 16u);
            const float other_l = simd_shuffle_xor(l_val, 16u);
            const float4 other_acc = float4(simd_shuffle_xor(acc.x, 16u),
                                            simd_shuffle_xor(acc.y, 16u),
                                            simd_shuffle_xor(acc.z, 16u),
                                            simd_shuffle_xor(acc.w, 16u));
            const float m_final = max(m_val, other_m);
            const float w_self = (l_val > 0.0f) ? exp2(m_val - m_final) : 0.0f;
            const float w_other = (other_l > 0.0f) ? exp2(other_m - m_final) : 0.0f;
            const float l_final = fma(w_self, l_val, w_other * other_l);
            const float4 acc_final = fma(float4(w_self), acc, float4(w_other) * other_acc);

            if (lane < 16u) {
                if (p.num_splits == 1u) {
                    const float inv_l = (l_final > 0.0f) ? (1.0f / l_final) : 0.0f;
                    *(device half4 *)(out + (ulong(node) * p.num_q_heads + qh) * 64u + col * 4u) = half4(acc_final * inv_l);
                } else {
                    const ulong ps = (ulong(node) * p.num_q_heads + qh) * p.num_splits + split_idx;
                    if (lane == 0u) {
                        partial_m[ps] = m_final;
                        partial_l[ps] = l_final;
                    }
                    *(device float4 *)(partial_o + ps * 64u + col * 4u) = acc_final;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 5b: Tree-Attention Split-KV Reduction (merges num_splits > 1 partials)
//
// Specialized on ATTN_HEAD_DIM ({64, 128}).
// Grid: (num_q_heads, num_nodes, 1) threadgroups, 32 threads.
// ---------------------------------------------------------------------------
kernel void ami_gqa_tree_verify_attn_reduce(const device float *partial_m [[buffer(0)]],
                                            const device float *partial_l [[buffer(1)]],
                                            const device float *partial_o [[buffer(2)]],
                                            device half *out [[buffer(3)]],
                                            constant GqaTreeVerifyParams &p [[buffer(4)]],
                                            uint2 tg_pos [[threadgroup_position_in_grid]],
                                            uint lane [[thread_index_in_simdgroup]]) {
    const uint qh = tg_pos.x;
    const uint node = tg_pos.y;
    const uint num_texels = ATTN_HEAD_DIM >> 2;
    if (qh >= p.num_q_heads || node >= p.num_nodes || lane >= num_texels) {
        return;
    }

    const ulong head_base = (ulong(node) * p.num_q_heads + qh) * p.num_splits;
    float m_max = -INFINITY;
    for (uint s = 0u; s < p.num_splits; s++) {
        if (partial_l[head_base + s] > 0.0f) {
            m_max = max(m_max, partial_m[head_base + s]);
        }
    }

    float l_tot = 0.0f;
    float4 o_tot = float4(0.0f);
    for (uint s = 0u; s < p.num_splits; s++) {
        const float ls = partial_l[head_base + s];
        if (ls > 0.0f) {
            const float w = exp2(partial_m[head_base + s] - m_max);
            l_tot = fma(ls, w, l_tot);
            const float4 po = *(const device float4 *)(partial_o + (head_base + s) * ATTN_HEAD_DIM + lane * 4u);
            o_tot = fma(po, float4(w), o_tot);
        }
    }

    const float inv_l = (l_tot > 0.0f) ? (1.0f / l_tot) : 0.0f;
    *(device half4 *)(out + (ulong(node) * p.num_q_heads + qh) * ATTN_HEAD_DIM + lane * 4u) = half4(o_tot * inv_l);
}

// ---------------------------------------------------------------------------
// Kernel 6: Accepted-Path KV Tree Commit into IOSurface RGBA16Float KvTiles
//
// Copies k_tree[node_idx] and v_tree[node_idx] for `step in 0..accepted_count`
// into committed RGBA16Float KvTile rows at positions `base_pos + step`.
// Cleanly handles crossing a `chunk_tokens` boundary across tile0 and tile1
// in ONE GPU dispatch.
//
// Grid: (num_kv_heads, accepted_count, 1) threadgroups, 32 threads.
// ---------------------------------------------------------------------------
kernel void ami_kv_tree_commit(const device half *k_tree [[buffer(0)]],
                               const device half *v_tree [[buffer(1)]],
                               const device uint *accepted_nodes [[buffer(2)]],
                               constant KvTreeCommitParams &p [[buffer(3)]],
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
    const uint pos = p.base_pos + step;
    const uint chunk0 = p.base_pos / p.chunk_tokens;
    const uint pos_chunk = pos / p.chunk_tokens;
    const uint slot_in_chunk = pos - pos_chunk * p.chunk_tokens;
    const uint row = kvh * p.chunk_tokens + slot_in_chunk;
    const uint num_cols = p.head_dim >> 2;
    const ulong src_off = (ulong(node_idx) * p.num_kv_heads + kvh) * p.head_dim;

    const device half4 *k_src = (const device half4 *)(k_tree + src_off);
    const device half4 *v_src = (const device half4 *)(v_tree + src_off);
    const bool use_tile1 = (p.has_second_chunk != 0u) && (pos_chunk > chunk0);

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

// ===========================================================================
// Milestone M6.5: R5 Group-64 q8 Retired KV Tier, R8 CoW Prefix Tail Copy,
// and C14 Multi-Request Batched Decode (ADR-0004, ADR-0019)
//
// Group-64 q8 layout per 64-element group (`68 B`, 4-byte aligned):
//   bytes [0..63]: 64 signed int8 codes (`char`)
//   bytes [64..67]: `half2(scale, bias)` -> dequant: `float(q) * scale + bias`
// ===========================================================================

struct KvRetireQ8Params {
    uint num_kv_heads;
    uint head_dim;
    uint chunk_tokens;
    uint dst_slot;
    uint plane_stride_bytes;
    uint chunk_stride_bytes;
    uint _pad0;
    uint _pad1;
};

struct KvTileCowCopyParams {
    uint num_kv_heads;
    uint head_dim;
    uint chunk_tokens;
    uint valid_tokens;
};

struct TieredGqaDecodeParams {
    uint num_q_heads;
    uint num_kv_heads;
    uint seq_len;
    uint chunk_tokens;
    uint num_splits;
    uint q_groups_per_kv;
    float scale;
    uint num_retired_chunks;
    uint retired_plane_stride_bytes;
    uint retired_chunk_stride_bytes;
    uint _pad0;
    uint _pad1;
};

struct TieredGqaTreeVerifyParams {
    uint num_nodes;
    uint num_q_heads;
    uint num_kv_heads;
    uint base_pos;
    uint chunk_tokens;
    uint num_splits;
    uint q_groups_per_kv;
    float scale;
    uint num_retired_chunks;
    uint retired_plane_stride_bytes;
    uint retired_chunk_stride_bytes;
    uint _pad0;
};

struct BatchKvWriteSlot {
    texture2d<half, access::write> k_tex;
    texture2d<half, access::write> v_tex;
    uint pos;
    uint slot_in_chunk;
    uint _pad0;
    uint _pad1;
};

struct BatchQkNormRopeKvParams {
    uint batch_size;
    uint num_q_heads;
    uint num_kv_heads;
    uint head_dim;
    uint chunk_tokens;
    float rms_eps;
    float rope_theta;
    uint apply_qk_norm;
};

struct BatchGqaSlotDesc {
    uint seq_len;
    uint tile_table_offset;
    uint retired_offset;
    uint num_retired_chunks;
};

struct BatchGqaDecodeParams {
    uint batch_size;
    uint num_q_heads;
    uint num_kv_heads;
    uint chunk_tokens;
    uint num_splits;
    uint q_groups_per_kv;
    float scale;
    uint retired_plane_stride_bytes;
    uint retired_chunk_stride_bytes;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

static_assert(sizeof(KvRetireQ8Params) == 32, "KvRetireQ8Params must be 32 bytes");
static_assert(sizeof(KvTileCowCopyParams) == 16, "KvTileCowCopyParams must be 16 bytes");
static_assert(sizeof(TieredGqaDecodeParams) == 48, "TieredGqaDecodeParams must be 48 bytes");
static_assert(sizeof(TieredGqaTreeVerifyParams) == 48, "TieredGqaTreeVerifyParams must be 48 bytes");
static_assert(sizeof(BatchKvWriteSlot) == 32, "BatchKvWriteSlot must be 32 bytes");
static_assert(sizeof(BatchQkNormRopeKvParams) == 32, "BatchQkNormRopeKvParams must be 32 bytes");
static_assert(sizeof(BatchGqaSlotDesc) == 16, "BatchGqaSlotDesc must be 16 bytes");
static_assert(sizeof(BatchGqaDecodeParams) == 48, "BatchGqaDecodeParams must be 48 bytes");

static inline half4 load_q8_half4(const device uchar *plane_base,
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

static inline void gqa_tile_update_d128(const threadgroup half4 *tg_k_buf,
                                        const threadgroup half4 *tg_v_buf,
                                        uint n_tok,
                                        float4 q_vec,
                                        float scale_log2e,
                                        uint lane,
                                        thread float &m_val,
                                        thread float &l_val,
                                        thread float4 &acc) {
    if (n_tok == 8u) {
        const float s0 = simd_sum(dot(q_vec, float4(tg_k_buf[0u * 32u + lane]))) * scale_log2e;
        const float s1 = simd_sum(dot(q_vec, float4(tg_k_buf[1u * 32u + lane]))) * scale_log2e;
        const float s2 = simd_sum(dot(q_vec, float4(tg_k_buf[2u * 32u + lane]))) * scale_log2e;
        const float s3 = simd_sum(dot(q_vec, float4(tg_k_buf[3u * 32u + lane]))) * scale_log2e;
        const float s4 = simd_sum(dot(q_vec, float4(tg_k_buf[4u * 32u + lane]))) * scale_log2e;
        const float s5 = simd_sum(dot(q_vec, float4(tg_k_buf[5u * 32u + lane]))) * scale_log2e;
        const float s6 = simd_sum(dot(q_vec, float4(tg_k_buf[6u * 32u + lane]))) * scale_log2e;
        const float s7 = simd_sum(dot(q_vec, float4(tg_k_buf[7u * 32u + lane]))) * scale_log2e;

        const float tile_m = max(max(max(s0, s1), max(s2, s3)),
                                 max(max(s4, s5), max(s6, s7)));
        const float m_new = max(m_val, tile_m);
        const float alpha = (l_val > 0.0f) ? exp2(m_val - m_new) : 0.0f;
        const float p0 = exp2(s0 - m_new);
        const float p1 = exp2(s1 - m_new);
        const float p2 = exp2(s2 - m_new);
        const float p3 = exp2(s3 - m_new);
        const float p4 = exp2(s4 - m_new);
        const float p5 = exp2(s5 - m_new);
        const float p6 = exp2(s6 - m_new);
        const float p7 = exp2(s7 - m_new);

        l_val = fma(l_val, alpha, ((p0 + p1) + (p2 + p3)) + ((p4 + p5) + (p6 + p7)));
        acc = acc * alpha;
        acc = fma(float4(p0), float4(tg_v_buf[0u * 32u + lane]), acc);
        acc = fma(float4(p1), float4(tg_v_buf[1u * 32u + lane]), acc);
        acc = fma(float4(p2), float4(tg_v_buf[2u * 32u + lane]), acc);
        acc = fma(float4(p3), float4(tg_v_buf[3u * 32u + lane]), acc);
        acc = fma(float4(p4), float4(tg_v_buf[4u * 32u + lane]), acc);
        acc = fma(float4(p5), float4(tg_v_buf[5u * 32u + lane]), acc);
        acc = fma(float4(p6), float4(tg_v_buf[6u * 32u + lane]), acc);
        acc = fma(float4(p7), float4(tg_v_buf[7u * 32u + lane]), acc);
        m_val = m_new;
    } else {
        for (uint k = 0u; k < n_tok; k++) {
            const float score = simd_sum(dot(q_vec, float4(tg_k_buf[k * 32u + lane]))) * scale_log2e;
            const float m_new = max(m_val, score);
            const float alpha = (l_val > 0.0f) ? exp2(m_val - m_new) : 0.0f;
            const float prob = exp2(score - m_new);
            l_val = fma(l_val, alpha, prob);
            acc = fma(acc, float4(alpha), prob * float4(tg_v_buf[k * 32u + lane]));
            m_val = m_new;
        }
    }
}

static inline void gqa_tile_update_d64(const threadgroup half4 *tg_k_buf,
                                       const threadgroup half4 *tg_v_buf,
                                       uint n_tok,
                                       float4 q_vec,
                                       float scale_log2e,
                                       uint col,
                                       uint half_id,
                                       thread float &m_val,
                                       thread float &l_val,
                                       thread float4 &acc) {
    if (n_tok == 16u) {
        const uint k_off = half_id * 8u;
        const float s0 = half_simd_sum(dot(q_vec, float4(tg_k_buf[(k_off + 0u) * 16u + col]))) * scale_log2e;
        const float s1 = half_simd_sum(dot(q_vec, float4(tg_k_buf[(k_off + 1u) * 16u + col]))) * scale_log2e;
        const float s2 = half_simd_sum(dot(q_vec, float4(tg_k_buf[(k_off + 2u) * 16u + col]))) * scale_log2e;
        const float s3 = half_simd_sum(dot(q_vec, float4(tg_k_buf[(k_off + 3u) * 16u + col]))) * scale_log2e;
        const float s4 = half_simd_sum(dot(q_vec, float4(tg_k_buf[(k_off + 4u) * 16u + col]))) * scale_log2e;
        const float s5 = half_simd_sum(dot(q_vec, float4(tg_k_buf[(k_off + 5u) * 16u + col]))) * scale_log2e;
        const float s6 = half_simd_sum(dot(q_vec, float4(tg_k_buf[(k_off + 6u) * 16u + col]))) * scale_log2e;
        const float s7 = half_simd_sum(dot(q_vec, float4(tg_k_buf[(k_off + 7u) * 16u + col]))) * scale_log2e;

        const float tile_m = max(max(max(s0, s1), max(s2, s3)),
                                 max(max(s4, s5), max(s6, s7)));
        const float m_new = max(m_val, tile_m);
        const float alpha = (l_val > 0.0f) ? exp2(m_val - m_new) : 0.0f;
        const float p0 = exp2(s0 - m_new);
        const float p1 = exp2(s1 - m_new);
        const float p2 = exp2(s2 - m_new);
        const float p3 = exp2(s3 - m_new);
        const float p4 = exp2(s4 - m_new);
        const float p5 = exp2(s5 - m_new);
        const float p6 = exp2(s6 - m_new);
        const float p7 = exp2(s7 - m_new);

        l_val = fma(l_val, alpha, ((p0 + p1) + (p2 + p3)) + ((p4 + p5) + (p6 + p7)));
        acc = acc * alpha;
        acc = fma(float4(p0), float4(tg_v_buf[(k_off + 0u) * 16u + col]), acc);
        acc = fma(float4(p1), float4(tg_v_buf[(k_off + 1u) * 16u + col]), acc);
        acc = fma(float4(p2), float4(tg_v_buf[(k_off + 2u) * 16u + col]), acc);
        acc = fma(float4(p3), float4(tg_v_buf[(k_off + 3u) * 16u + col]), acc);
        acc = fma(float4(p4), float4(tg_v_buf[(k_off + 4u) * 16u + col]), acc);
        acc = fma(float4(p5), float4(tg_v_buf[(k_off + 5u) * 16u + col]), acc);
        acc = fma(float4(p6), float4(tg_v_buf[(k_off + 6u) * 16u + col]), acc);
        acc = fma(float4(p7), float4(tg_v_buf[(k_off + 7u) * 16u + col]), acc);
        m_val = m_new;
    } else {
        for (uint k_base = 0u; k_base < n_tok; k_base += 2u) {
            const uint k = k_base + half_id;
            const bool valid = (k < n_tok);
            const float4 k4 = valid ? float4(tg_k_buf[k * 16u + col]) : float4(0.0f);
            const float score = half_simd_sum(dot(q_vec, k4)) * scale_log2e;
            if (valid) {
                const float m_new = max(m_val, score);
                const float alpha = (l_val > 0.0f) ? exp2(m_val - m_new) : 0.0f;
                const float prob = exp2(score - m_new);
                l_val = fma(l_val, alpha, prob);
                acc = fma(acc, float4(alpha), prob * float4(tg_v_buf[k * 16u + col]));
                m_val = m_new;
            }
        }
    }
}

static inline void gqa_tiered_attend_range_d128(uint t_start,
                                                uint t_end,
                                                uint chunk_tokens,
                                                uint head_row_base,
                                                uint num_retired_chunks,
                                                uint tile_table_offset,
                                                uint retired_offset,
                                                uint retired_plane_stride_bytes,
                                                uint retired_chunk_stride_bytes,
                                                float scale_log2e,
                                                const device KvTilePair *kv_tiles,
                                                const device uchar *retired_pool,
                                                const device uint *retired_chunk_ids,
                                                float4 q_vec,
                                                bool active_q,
                                                uint tid,
                                                uint lane,
                                                threadgroup half4 *tg_k0,
                                                threadgroup half4 *tg_k1,
                                                threadgroup half4 *tg_v0,
                                                threadgroup half4 *tg_v1,
                                                thread float &m_val,
                                                thread float &l_val,
                                                thread float4 &acc) {
    const uint load_tok = tid >> 5;  // 0..7
    const uint load_col = tid & 31u; // 0..31
    const uint first_chunk = t_start / chunk_tokens;
    const uint last_chunk = (t_end - 1u) / chunk_tokens;

    for (uint chunk = first_chunk; chunk <= last_chunk; chunk++) {
        const uint chunk_base = chunk * chunk_tokens;
        const uint slot_lo = (t_start > chunk_base) ? (t_start - chunk_base) : 0u;
        const uint slot_hi = min(t_end - chunk_base, chunk_tokens);

        if (chunk < num_retired_chunks) {
            const uint phys_slot = retired_chunk_ids[retired_offset + chunk];
            const device uchar *k_plane = retired_pool + ulong(phys_slot) * retired_chunk_stride_bytes;
            const device uchar *v_plane = k_plane + retired_plane_stride_bytes;

            uint buf = 0u;
            if (slot_lo + load_tok < slot_hi) {
                const uint row = head_row_base + slot_lo + load_tok;
                tg_k0[tid] = load_q8_half4(k_plane, row, 2u, load_col);
                tg_v0[tid] = load_q8_half4(v_plane, row, 2u, load_col);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint base_slot = slot_lo; base_slot < slot_hi; base_slot += 8u) {
                const uint n_tok = min(8u, slot_hi - base_slot);
                const uint next_base = base_slot + 8u;
                const uint next_buf = buf ^ 1u;

                half4 next_k = half4(0.0h);
                half4 next_v = half4(0.0h);
                const bool has_next = (next_base + load_tok < slot_hi);
                if (has_next) {
                    const uint next_row = head_row_base + next_base + load_tok;
                    next_k = load_q8_half4(k_plane, next_row, 2u, load_col);
                    next_v = load_q8_half4(v_plane, next_row, 2u, load_col);
                }

                if (active_q) {
                    const threadgroup half4 *cur_k = (buf == 0u) ? tg_k0 : tg_k1;
                    const threadgroup half4 *cur_v = (buf == 0u) ? tg_v0 : tg_v1;
                    gqa_tile_update_d128(cur_k, cur_v, n_tok, q_vec, scale_log2e, lane, m_val, l_val, acc);
                }

                if (has_next) {
                    threadgroup half4 *dst_k = (next_buf == 0u) ? tg_k0 : tg_k1;
                    threadgroup half4 *dst_v = (next_buf == 0u) ? tg_v0 : tg_v1;
                    dst_k[tid] = next_k;
                    dst_v[tid] = next_v;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                buf = next_buf;
            }
        } else {
            const uint active_idx = tile_table_offset + (chunk - num_retired_chunks);
            const texture2d<half, access::read> k_tex = kv_tiles[active_idx].k_tex;
            const texture2d<half, access::read> v_tex = kv_tiles[active_idx].v_tex;

            uint buf = 0u;
            if (slot_lo + load_tok < slot_hi) {
                const uint row = head_row_base + slot_lo + load_tok;
                tg_k0[tid] = k_tex.read(uint2(load_col, row));
                tg_v0[tid] = v_tex.read(uint2(load_col, row));
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint base_slot = slot_lo; base_slot < slot_hi; base_slot += 8u) {
                const uint n_tok = min(8u, slot_hi - base_slot);
                const uint next_base = base_slot + 8u;
                const uint next_buf = buf ^ 1u;

                half4 next_k = half4(0.0h);
                half4 next_v = half4(0.0h);
                const bool has_next = (next_base + load_tok < slot_hi);
                if (has_next) {
                    const uint next_row = head_row_base + next_base + load_tok;
                    next_k = k_tex.read(uint2(load_col, next_row));
                    next_v = v_tex.read(uint2(load_col, next_row));
                }

                if (active_q) {
                    const threadgroup half4 *cur_k = (buf == 0u) ? tg_k0 : tg_k1;
                    const threadgroup half4 *cur_v = (buf == 0u) ? tg_v0 : tg_v1;
                    gqa_tile_update_d128(cur_k, cur_v, n_tok, q_vec, scale_log2e, lane, m_val, l_val, acc);
                }

                if (has_next) {
                    threadgroup half4 *dst_k = (next_buf == 0u) ? tg_k0 : tg_k1;
                    threadgroup half4 *dst_v = (next_buf == 0u) ? tg_v0 : tg_v1;
                    dst_k[tid] = next_k;
                    dst_v[tid] = next_v;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                buf = next_buf;
            }
        }
    }
}

static inline void gqa_tiered_attend_range_d64(uint t_start,
                                               uint t_end,
                                               uint chunk_tokens,
                                               uint head_row_base,
                                               uint num_retired_chunks,
                                               uint tile_table_offset,
                                               uint retired_offset,
                                               uint retired_plane_stride_bytes,
                                               uint retired_chunk_stride_bytes,
                                               float scale_log2e,
                                               const device KvTilePair *kv_tiles,
                                               const device uchar *retired_pool,
                                               const device uint *retired_chunk_ids,
                                               float4 q_vec,
                                               bool active_q,
                                               uint tid,
                                               uint col,
                                               uint half_id,
                                               threadgroup half4 *tg_k0,
                                               threadgroup half4 *tg_k1,
                                               threadgroup half4 *tg_v0,
                                               threadgroup half4 *tg_v1,
                                               thread float &m_val,
                                               thread float &l_val,
                                               thread float4 &acc) {
    const uint load_tok = tid >> 4;  // 0..15
    const uint load_col = tid & 15u; // 0..15
    const uint first_chunk = t_start / chunk_tokens;
    const uint last_chunk = (t_end - 1u) / chunk_tokens;

    for (uint chunk = first_chunk; chunk <= last_chunk; chunk++) {
        const uint chunk_base = chunk * chunk_tokens;
        const uint slot_lo = (t_start > chunk_base) ? (t_start - chunk_base) : 0u;
        const uint slot_hi = min(t_end - chunk_base, chunk_tokens);

        if (chunk < num_retired_chunks) {
            const uint phys_slot = retired_chunk_ids[retired_offset + chunk];
            const device uchar *k_plane = retired_pool + ulong(phys_slot) * retired_chunk_stride_bytes;
            const device uchar *v_plane = k_plane + retired_plane_stride_bytes;

            uint buf = 0u;
            if (slot_lo + load_tok < slot_hi) {
                const uint row = head_row_base + slot_lo + load_tok;
                tg_k0[tid] = load_q8_half4(k_plane, row, 1u, load_col);
                tg_v0[tid] = load_q8_half4(v_plane, row, 1u, load_col);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint base_slot = slot_lo; base_slot < slot_hi; base_slot += 16u) {
                const uint n_tok = min(16u, slot_hi - base_slot);
                const uint next_base = base_slot + 16u;
                const uint next_buf = buf ^ 1u;

                half4 next_k = half4(0.0h);
                half4 next_v = half4(0.0h);
                const bool has_next = (next_base + load_tok < slot_hi);
                if (has_next) {
                    const uint next_row = head_row_base + next_base + load_tok;
                    next_k = load_q8_half4(k_plane, next_row, 1u, load_col);
                    next_v = load_q8_half4(v_plane, next_row, 1u, load_col);
                }

                if (active_q) {
                    const threadgroup half4 *cur_k = (buf == 0u) ? tg_k0 : tg_k1;
                    const threadgroup half4 *cur_v = (buf == 0u) ? tg_v0 : tg_v1;
                    gqa_tile_update_d64(cur_k, cur_v, n_tok, q_vec, scale_log2e, col, half_id, m_val, l_val, acc);
                }

                if (has_next) {
                    threadgroup half4 *dst_k = (next_buf == 0u) ? tg_k0 : tg_k1;
                    threadgroup half4 *dst_v = (next_buf == 0u) ? tg_v0 : tg_v1;
                    dst_k[tid] = next_k;
                    dst_v[tid] = next_v;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                buf = next_buf;
            }
        } else {
            const uint active_idx = tile_table_offset + (chunk - num_retired_chunks);
            const texture2d<half, access::read> k_tex = kv_tiles[active_idx].k_tex;
            const texture2d<half, access::read> v_tex = kv_tiles[active_idx].v_tex;

            uint buf = 0u;
            if (slot_lo + load_tok < slot_hi) {
                const uint row = head_row_base + slot_lo + load_tok;
                tg_k0[tid] = k_tex.read(uint2(load_col, row));
                tg_v0[tid] = v_tex.read(uint2(load_col, row));
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint base_slot = slot_lo; base_slot < slot_hi; base_slot += 16u) {
                const uint n_tok = min(16u, slot_hi - base_slot);
                const uint next_base = base_slot + 16u;
                const uint next_buf = buf ^ 1u;

                half4 next_k = half4(0.0h);
                half4 next_v = half4(0.0h);
                const bool has_next = (next_base + load_tok < slot_hi);
                if (has_next) {
                    const uint next_row = head_row_base + next_base + load_tok;
                    next_k = k_tex.read(uint2(load_col, next_row));
                    next_v = v_tex.read(uint2(load_col, next_row));
                }

                if (active_q) {
                    const threadgroup half4 *cur_k = (buf == 0u) ? tg_k0 : tg_k1;
                    const threadgroup half4 *cur_v = (buf == 0u) ? tg_v0 : tg_v1;
                    gqa_tile_update_d64(cur_k, cur_v, n_tok, q_vec, scale_log2e, col, half_id, m_val, l_val, acc);
                }

                if (has_next) {
                    threadgroup half4 *dst_k = (next_buf == 0u) ? tg_k0 : tg_k1;
                    threadgroup half4 *dst_v = (next_buf == 0u) ? tg_v0 : tg_v1;
                    dst_k[tid] = next_k;
                    dst_v[tid] = next_v;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                buf = next_buf;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 7 (R5): Retire a completed 512-token RGBA16Float IOSurface KvTilePair
// into a group-64 q8 slot (`64 int8 + f16 scale + f16 bias = 68 B / 64 dims`)
// inside `RetiredKvQ8Pool`.
//
// Grid: (num_kv_heads * chunk_tokens, head_dim / 64, 1) threadgroups, 32 threads.
// Lanes 0..15 quantize K; lanes 16..31 quantize V independently with zero TG mem.
// ---------------------------------------------------------------------------
kernel void ami_kv_retire_tile_to_q8(device uchar *retired_pool [[buffer(0)]],
                                     constant KvRetireQ8Params &p [[buffer(1)]],
                                     texture2d<half, access::read> k_tex [[texture(0)]],
                                     texture2d<half, access::read> v_tex [[texture(1)]],
                                     uint2 tg_pos [[threadgroup_position_in_grid]],
                                     uint lane [[thread_index_in_simdgroup]]) {
    const uint total_rows = p.num_kv_heads * p.chunk_tokens;
    const uint groups_per_head = p.head_dim >> 6;
    const uint row = tg_pos.x;
    const uint group_idx = tg_pos.y;
    if (row >= total_rows || group_idx >= groups_per_head) {
        return;
    }

    const uint is_v = lane >> 4;        // 0 = K, 1 = V
    const uint col_in_grp = lane & 15u; // 0..15 (4 dims per lane = 64 dims)
    const uint tex_col = group_idx * 16u + col_in_grp;

    const half4 h4 = (is_v == 0u) ? k_tex.read(uint2(tex_col, row))
                                  : v_tex.read(uint2(tex_col, row));
    const float4 x = float4(h4);

    const float lo = half_simd_min(min(min(x.x, x.y), min(x.z, x.w)));
    const float hi = half_simd_max(max(max(x.x, x.y), max(x.z, x.w)));

    const float range = hi - lo;
    const half scale_h = half(range / 255.0f);
    const float scale_f = float(scale_h);
    const float mid = 0.5f * (lo + hi);
    const half bias_h = (scale_f > 0.0f) ? half(fma(0.5f, scale_f, mid)) : half(mid);
    const float bias_f = float(bias_h);
    const float inv_scale = (scale_f > 0.0f) ? (1.0f / scale_f) : 0.0f;

    const int4 qi = clamp(int4(rint((x - bias_f) * inv_scale)), int4(-128), int4(127));
    const char4 q4 = char4(qi);

    const ulong chunk_base = ulong(p.dst_slot) * p.chunk_stride_bytes;
    const ulong plane_base = chunk_base + ((is_v != 0u) ? ulong(p.plane_stride_bytes) : 0ul);
    const ulong grp_off = (ulong(row) * groups_per_head + group_idx) * 68ul;
    device uchar *dst = retired_pool + plane_base + grp_off;

    *(device char4 *)(dst + col_in_grp * 4u) = q4;
    if (col_in_grp == 0u) {
        *(device half2 *)(dst + 64u) = half2(scale_h, bias_h);
    }
}

// ---------------------------------------------------------------------------
// Kernel 8 (R8): Copy-on-Write partial tail KvTile copy on GPU (0 CPU memcpy).
// Copies `valid_tokens` (`1..chunk_tokens`) token-major RGBA16Float rows across
// `num_kv_heads` from cached `(src_k, src_v)` into private `(dst_k, dst_v)`.
//
// Grid: (num_kv_heads, valid_tokens, 1) threadgroups, 32 threads.
// ---------------------------------------------------------------------------
kernel void ami_kv_tile_cow_copy(constant KvTileCowCopyParams &p [[buffer(0)]],
                                 texture2d<half, access::read> src_k [[texture(0)]],
                                 texture2d<half, access::read> src_v [[texture(1)]],
                                 texture2d<half, access::write> dst_k [[texture(2)]],
                                 texture2d<half, access::write> dst_v [[texture(3)]],
                                 uint2 tg_pos [[threadgroup_position_in_grid]],
                                 uint lane [[thread_index_in_simdgroup]]) {
    const uint kvh = tg_pos.x;
    const uint tok = tg_pos.y;
    if (kvh >= p.num_kv_heads || tok >= p.valid_tokens) {
        return;
    }
    const uint row = kvh * p.chunk_tokens + tok;
    const uint num_cols = p.head_dim >> 2;
    for (uint col = lane; col < num_cols; col += 32u) {
        dst_k.write(src_k.read(uint2(col, row)), uint2(col, row));
        dst_v.write(src_v.read(uint2(col, row)), uint2(col, row));
    }
}

// ---------------------------------------------------------------------------
// Kernel 9 (R5): Tiered GQA Decode Attention (M = 1) seamlessly spanning
// `num_retired_chunks` group-64 q8 chunks in `RetiredKvQ8Pool` AND active
// `RGBA16Float` IOSurface `KvTilePair`s in ONE split-KV online-softmax pass.
//
// Specialized on ATTN_HEAD_DIM ({64, 128}).
// Grid: (num_kv_heads * q_groups_per_kv, num_splits, 1) threadgroups, 256 threads.
// ---------------------------------------------------------------------------
kernel void ami_gqa_tiered_decode_attn_split(const device half *q [[buffer(0)]],
                                             const device KvTilePair *kv_tiles [[buffer(1)]],
                                             const device uchar *retired_pool [[buffer(2)]],
                                             const device uint *retired_chunk_ids [[buffer(3)]],
                                             device half *out [[buffer(4)]],
                                             device float *partial_m [[buffer(5)]],
                                             device float *partial_l [[buffer(6)]],
                                             device float *partial_o [[buffer(7)]],
                                             constant TieredGqaDecodeParams &p [[buffer(8)]],
                                             uint2 tg_pos [[threadgroup_position_in_grid]],
                                             uint tid [[thread_index_in_threadgroup]],
                                             uint sg_id [[simdgroup_index_in_threadgroup]],
                                             uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint Q_TILE = 8u;
    constexpr float LOG2E = 1.4426950408889634f;

    const uint kv_head = tg_pos.x / p.q_groups_per_kv;
    const uint q_grp = tg_pos.x % p.q_groups_per_kv;
    const uint split_idx = tg_pos.y;
    if (kv_head >= p.num_kv_heads || split_idx >= p.num_splits) {
        return;
    }

    const uint gqa_ratio = p.num_q_heads / p.num_kv_heads;
    const uint q_start = kv_head * gqa_ratio + q_grp * Q_TILE;
    const uint q_end = min(q_start + Q_TILE, (kv_head + 1u) * gqa_ratio);
    const uint q_cnt = q_end - q_start;

    const uint tokens_per_split = (p.seq_len + p.num_splits - 1u) / p.num_splits;
    const uint t_start = split_idx * tokens_per_split;
    const uint t_end = min(t_start + tokens_per_split, p.seq_len);

    if (t_start >= t_end) {
        if (p.num_splits > 1u && tid < q_cnt) {
            const ulong ps = ulong(q_start + tid) * p.num_splits + split_idx;
            partial_m[ps] = -INFINITY;
            partial_l[ps] = 0.0f;
        }
        return;
    }

    const bool active_q = (sg_id < q_cnt);
    const uint qh = q_start + min(sg_id, q_cnt - 1u);
    const float scale_log2e = p.scale * LOG2E;
    const uint head_row_base = kv_head * p.chunk_tokens;

    threadgroup half4 tg_k0[256];
    threadgroup half4 tg_k1[256];
    threadgroup half4 tg_v0[256];
    threadgroup half4 tg_v1[256];

    if (ATTN_HEAD_DIM == 128u) {
        const float4 q_vec = active_q
            ? float4(*(const device half4 *)(q + ulong(qh) * 128u + lane * 4u))
            : float4(0.0f);
        float m_val = -INFINITY;
        float l_val = 0.0f;
        float4 acc = float4(0.0f);

        gqa_tiered_attend_range_d128(t_start, t_end, p.chunk_tokens, head_row_base,
                                     p.num_retired_chunks, 0u, 0u,
                                     p.retired_plane_stride_bytes, p.retired_chunk_stride_bytes,
                                     scale_log2e, kv_tiles, retired_pool, retired_chunk_ids,
                                     q_vec, active_q, tid, lane,
                                     tg_k0, tg_k1, tg_v0, tg_v1, m_val, l_val, acc);

        if (active_q) {
            if (p.num_splits == 1u) {
                const float inv_l = (l_val > 0.0f) ? (1.0f / l_val) : 0.0f;
                *(device half4 *)(out + ulong(qh) * 128u + lane * 4u) = half4(acc * inv_l);
            } else {
                const ulong ps = ulong(qh) * p.num_splits + split_idx;
                if (lane == 0u) {
                    partial_m[ps] = m_val;
                    partial_l[ps] = l_val;
                }
                *(device float4 *)(partial_o + ps * 128u + lane * 4u) = acc;
            }
        }
    } else {
        const uint col = lane & 15u;
        const uint half_id = lane >> 4;
        const float4 q_vec = active_q
            ? float4(*(const device half4 *)(q + ulong(qh) * 64u + col * 4u))
            : float4(0.0f);
        float m_val = -INFINITY;
        float l_val = 0.0f;
        float4 acc = float4(0.0f);

        gqa_tiered_attend_range_d64(t_start, t_end, p.chunk_tokens, head_row_base,
                                    p.num_retired_chunks, 0u, 0u,
                                    p.retired_plane_stride_bytes, p.retired_chunk_stride_bytes,
                                    scale_log2e, kv_tiles, retired_pool, retired_chunk_ids,
                                    q_vec, active_q, tid, col, half_id,
                                    tg_k0, tg_k1, tg_v0, tg_v1, m_val, l_val, acc);

        if (active_q) {
            const float other_m = simd_shuffle_xor(m_val, 16u);
            const float other_l = simd_shuffle_xor(l_val, 16u);
            const float4 other_acc = float4(simd_shuffle_xor(acc.x, 16u),
                                            simd_shuffle_xor(acc.y, 16u),
                                            simd_shuffle_xor(acc.z, 16u),
                                            simd_shuffle_xor(acc.w, 16u));
            const float m_final = max(m_val, other_m);
            const float w_self = (l_val > 0.0f) ? exp2(m_val - m_final) : 0.0f;
            const float w_other = (other_l > 0.0f) ? exp2(other_m - m_final) : 0.0f;
            const float l_final = fma(w_self, l_val, w_other * other_l);
            const float4 acc_final = fma(float4(w_self), acc, float4(w_other) * other_acc);

            if (lane < 16u) {
                if (p.num_splits == 1u) {
                    const float inv_l = (l_final > 0.0f) ? (1.0f / l_final) : 0.0f;
                    *(device half4 *)(out + ulong(qh) * 64u + col * 4u) = half4(acc_final * inv_l);
                } else {
                    const ulong ps = ulong(qh) * p.num_splits + split_idx;
                    if (lane == 0u) {
                        partial_m[ps] = m_final;
                        partial_l[ps] = l_final;
                    }
                    *(device float4 *)(partial_o + ps * 64u + col * 4u) = acc_final;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 10 (R5 + G5): Tiered Speculative Tree-Attention Verification
// spanning retired group-64 q8 chunks, active RGBA16Float KvTiles, AND
// 64-bit ancestor-masked tree KV in ONE split-KV pass.
//
// Specialized on ATTN_HEAD_DIM ({64, 128}) and TREE_MAX_NODES ({16, 32, 64}).
// Grid: (num_kv_heads * q_groups_per_kv, num_splits, 1) threadgroups, 256 threads.
// ---------------------------------------------------------------------------
kernel void ami_gqa_tiered_tree_verify_attn_split(const device half *q [[buffer(0)]],
                                                  const device KvTilePair *kv_tiles [[buffer(1)]],
                                                  const device uchar *retired_pool [[buffer(2)]],
                                                  const device uint *retired_chunk_ids [[buffer(3)]],
                                                  const device half *k_tree [[buffer(4)]],
                                                  const device half *v_tree [[buffer(5)]],
                                                  const device ulong *ancestor_masks [[buffer(6)]],
                                                  device half *out [[buffer(7)]],
                                                  device float *partial_m [[buffer(8)]],
                                                  device float *partial_l [[buffer(9)]],
                                                  device float *partial_o [[buffer(10)]],
                                                  constant TieredGqaTreeVerifyParams &p [[buffer(11)]],
                                                  uint2 tg_pos [[threadgroup_position_in_grid]],
                                                  uint tid [[thread_index_in_threadgroup]],
                                                  uint sg_id [[simdgroup_index_in_threadgroup]],
                                                  uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint Q_TILE = 8u;
    constexpr float LOG2E = 1.4426950408889634f;

    const uint kv_head = tg_pos.x / p.q_groups_per_kv;
    const uint q_grp = tg_pos.x % p.q_groups_per_kv;
    const uint split_idx = tg_pos.y;
    if (kv_head >= p.num_kv_heads || split_idx >= p.num_splits) {
        return;
    }

    const uint gqa_ratio = p.num_q_heads / p.num_kv_heads;
    const uint total_q = p.num_nodes * gqa_ratio;
    const uint q_start_flat = q_grp * Q_TILE;
    if (q_start_flat >= total_q) {
        return;
    }
    const uint q_cnt = min(Q_TILE, total_q - q_start_flat);

    uint t_start = 0u;
    uint t_end = 0u;
    if (p.base_pos > 0u) {
        const uint tokens_per_split = (p.base_pos + p.num_splits - 1u) / p.num_splits;
        t_start = split_idx * tokens_per_split;
        t_end = min(t_start + tokens_per_split, p.base_pos);
    }

    const bool do_prefix = (t_start < t_end);
    const bool do_tree = (split_idx == 0u);
    if (!do_prefix && !do_tree) {
        if (p.num_splits > 1u && tid < q_cnt) {
            const uint flat_q = q_start_flat + tid;
            const uint node = flat_q / gqa_ratio;
            const uint qh = kv_head * gqa_ratio + (flat_q - node * gqa_ratio);
            const ulong ps = (ulong(node) * p.num_q_heads + qh) * p.num_splits + split_idx;
            partial_m[ps] = -INFINITY;
            partial_l[ps] = 0.0f;
        }
        return;
    }

    const bool active_q = (sg_id < q_cnt);
    const uint safe_flat = q_start_flat + min(sg_id, q_cnt - 1u);
    const uint node = safe_flat / gqa_ratio;
    const uint qh = kv_head * gqa_ratio + (safe_flat - node * gqa_ratio);
    const ulong mask = active_q ? ancestor_masks[node] : 0ul;
    const float scale_log2e = p.scale * LOG2E;
    const uint head_row_base = kv_head * p.chunk_tokens;

    threadgroup half4 tg_k0[256];
    threadgroup half4 tg_k1[256];
    threadgroup half4 tg_v0[256];
    threadgroup half4 tg_v1[256];

    if (ATTN_HEAD_DIM == 128u) {
        const ulong q_off = (ulong(node) * p.num_q_heads + qh) * 128u + lane * 4u;
        const float4 q_vec = active_q ? float4(*(const device half4 *)(q + q_off)) : float4(0.0f);
        float m_val = -INFINITY;
        float l_val = 0.0f;
        float4 acc = float4(0.0f);

        if (do_prefix) {
            gqa_tiered_attend_range_d128(t_start, t_end, p.chunk_tokens, head_row_base,
                                         p.num_retired_chunks, 0u, 0u,
                                         p.retired_plane_stride_bytes, p.retired_chunk_stride_bytes,
                                         scale_log2e, kv_tiles, retired_pool, retired_chunk_ids,
                                         q_vec, active_q, tid, lane,
                                         tg_k0, tg_k1, tg_v0, tg_v1, m_val, l_val, acc);
        }

        if (do_tree) {
            const uint load_tok = tid >> 5;
            const uint load_col = tid & 31u;
            for (uint t_base = 0u; t_base < TREE_MAX_NODES; t_base += 8u) {
                if (t_base >= p.num_nodes) {
                    break;
                }
                const uint n_tok = min(8u, p.num_nodes - t_base);
                if (load_tok < n_tok) {
                    const uint node_j = t_base + load_tok;
                    const ulong kv_off = (ulong(node_j) * p.num_kv_heads + kv_head) * 128u + load_col * 4u;
                    tg_k0[tid] = *(const device half4 *)(k_tree + kv_off);
                    tg_v0[tid] = *(const device half4 *)(v_tree + kv_off);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (active_q) {
                    const uint tile_mask = uint((mask >> t_base) & 0xFFul);
                    if (tile_mask != 0u) {
                        for (uint k = 0u; k < n_tok; k++) {
                            if (((tile_mask >> k) & 1u) != 0u) {
                                const float score = simd_sum(dot(q_vec, float4(tg_k0[k * 32u + lane]))) * scale_log2e;
                                const float m_new = max(m_val, score);
                                const float alpha = (l_val > 0.0f) ? exp2(m_val - m_new) : 0.0f;
                                const float prob = exp2(score - m_new);
                                l_val = fma(l_val, alpha, prob);
                                acc = fma(acc, float4(alpha), prob * float4(tg_v0[k * 32u + lane]));
                                m_val = m_new;
                            }
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        if (active_q) {
            if (p.num_splits == 1u) {
                const float inv_l = (l_val > 0.0f) ? (1.0f / l_val) : 0.0f;
                *(device half4 *)(out + (ulong(node) * p.num_q_heads + qh) * 128u + lane * 4u) = half4(acc * inv_l);
            } else {
                const ulong ps = (ulong(node) * p.num_q_heads + qh) * p.num_splits + split_idx;
                if (lane == 0u) {
                    partial_m[ps] = m_val;
                    partial_l[ps] = l_val;
                }
                *(device float4 *)(partial_o + ps * 128u + lane * 4u) = acc;
            }
        }
    } else {
        const uint col = lane & 15u;
        const uint half_id = lane >> 4;
        const ulong q_off = (ulong(node) * p.num_q_heads + qh) * 64u + col * 4u;
        const float4 q_vec = active_q ? float4(*(const device half4 *)(q + q_off)) : float4(0.0f);
        float m_val = -INFINITY;
        float l_val = 0.0f;
        float4 acc = float4(0.0f);

        if (do_prefix) {
            gqa_tiered_attend_range_d64(t_start, t_end, p.chunk_tokens, head_row_base,
                                        p.num_retired_chunks, 0u, 0u,
                                        p.retired_plane_stride_bytes, p.retired_chunk_stride_bytes,
                                        scale_log2e, kv_tiles, retired_pool, retired_chunk_ids,
                                        q_vec, active_q, tid, col, half_id,
                                        tg_k0, tg_k1, tg_v0, tg_v1, m_val, l_val, acc);
        }

        if (do_tree) {
            const uint load_tok = tid >> 4;
            const uint load_col = tid & 15u;
            for (uint t_base = 0u; t_base < TREE_MAX_NODES; t_base += 16u) {
                if (t_base >= p.num_nodes) {
                    break;
                }
                const uint n_tok = min(16u, p.num_nodes - t_base);
                if (load_tok < n_tok) {
                    const uint node_j = t_base + load_tok;
                    const ulong kv_off = (ulong(node_j) * p.num_kv_heads + kv_head) * 64u + load_col * 4u;
                    tg_k0[tid] = *(const device half4 *)(k_tree + kv_off);
                    tg_v0[tid] = *(const device half4 *)(v_tree + kv_off);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (active_q) {
                    const uint tile_mask = uint((mask >> t_base) & 0xFFFFul);
                    if (tile_mask != 0u) {
                        for (uint k_base = 0u; k_base < n_tok; k_base += 2u) {
                            const uint k = k_base + half_id;
                            const bool valid = (k < n_tok) && (((tile_mask >> k) & 1u) != 0u);
                            const float4 k4 = valid ? float4(tg_k0[k * 16u + col]) : float4(0.0f);
                            const float score = half_simd_sum(dot(q_vec, k4)) * scale_log2e;
                            if (valid) {
                                const float m_new = max(m_val, score);
                                const float alpha = (l_val > 0.0f) ? exp2(m_val - m_new) : 0.0f;
                                const float prob = exp2(score - m_new);
                                l_val = fma(l_val, alpha, prob);
                                acc = fma(acc, float4(alpha), prob * float4(tg_v0[k * 16u + col]));
                                m_val = m_new;
                            }
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        if (active_q) {
            const float other_m = simd_shuffle_xor(m_val, 16u);
            const float other_l = simd_shuffle_xor(l_val, 16u);
            const float4 other_acc = float4(simd_shuffle_xor(acc.x, 16u),
                                            simd_shuffle_xor(acc.y, 16u),
                                            simd_shuffle_xor(acc.z, 16u),
                                            simd_shuffle_xor(acc.w, 16u));
            const float m_final = max(m_val, other_m);
            const float w_self = (l_val > 0.0f) ? exp2(m_val - m_final) : 0.0f;
            const float w_other = (other_l > 0.0f) ? exp2(other_m - m_final) : 0.0f;
            const float l_final = fma(w_self, l_val, w_other * other_l);
            const float4 acc_final = fma(float4(w_self), acc, float4(w_other) * other_acc);

            if (lane < 16u) {
                if (p.num_splits == 1u) {
                    const float inv_l = (l_final > 0.0f) ? (1.0f / l_final) : 0.0f;
                    *(device half4 *)(out + (ulong(node) * p.num_q_heads + qh) * 64u + col * 4u) = half4(acc_final * inv_l);
                } else {
                    const ulong ps = (ulong(node) * p.num_q_heads + qh) * p.num_splits + split_idx;
                    if (lane == 0u) {
                        partial_m[ps] = m_final;
                        partial_l[ps] = l_final;
                    }
                    *(device float4 *)(partial_o + ps * 64u + col * 4u) = acc_final;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 11 (C14): Multi-Request Batched QK-Norm + NeoX RoPE + Bindless KV Write
// Executes `batch_size` (`1..16+`) independent decode slots in ONE GPU pass.
//
// Specialized on NORM_W_BF16 (function_constant(2)).
// Grid: (num_q_heads + num_kv_heads, batch_size, 1) threadgroups, 32 threads.
// ---------------------------------------------------------------------------
kernel void ami_batch_qknorm_rope_kv_write(device half *q [[buffer(0)]],
                                           const device half *k [[buffer(1)]],
                                           const device half *v [[buffer(2)]],
                                           const device uchar *q_norm [[buffer(3)]],
                                           const device uchar *k_norm [[buffer(4)]],
                                           const device BatchKvWriteSlot *slots [[buffer(5)]],
                                           constant BatchQkNormRopeKvParams &p [[buffer(6)]],
                                           uint2 tg_pos [[threadgroup_position_in_grid]],
                                           uint lane [[thread_index_in_simdgroup]]) {
    const uint total_heads = p.num_q_heads + p.num_kv_heads;
    const uint b = tg_pos.y;
    if (tg_pos.x >= total_heads || b >= p.batch_size) {
        return;
    }

    threadgroup float vec[256];
    const bool is_q = (tg_pos.x < p.num_q_heads);
    const uint head_idx = is_q ? tg_pos.x : (tg_pos.x - p.num_q_heads);
    const uint d_dim = p.head_dim;
    const uint half_dim = d_dim >> 1;

    const ulong q_off = (ulong(b) * p.num_q_heads + head_idx) * d_dim;
    const ulong kv_off = (ulong(b) * p.num_kv_heads + head_idx) * d_dim;
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
            vec[d] = (vec[d] * inv_rms) * norm_w1(nw, d, NORM_W_BF16);
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    const BatchKvWriteSlot slot = slots[b];
    const float log_theta = log(p.rope_theta);
    const float inv_d = 2.0f / float(d_dim);
    const float fpos = float(slot.pos);
    for (uint j = lane; j < half_dim; j += 32u) {
        const float inv_freq = exp(-log_theta * (float(j) * inv_d));
        const float angle = fpos * inv_freq;
        const float c = cos(angle);
        const float s = sin(angle);
        const float x0 = vec[j];
        const float x1 = vec[j + half_dim];
        vec[j] = fma(x0, c, -x1 * s);
        vec[j + half_dim] = fma(x1, c, x0 * s);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    if (is_q) {
        device half *dst = q + q_off;
        for (uint d = lane; d < d_dim; d += 32u) {
            dst[d] = half(vec[d]);
        }
    } else {
        const uint row = head_idx * p.chunk_tokens + slot.slot_in_chunk;
        const uint num_cols = d_dim >> 2;
        const device half4 *v_src = (const device half4 *)(v + kv_off);
        for (uint col = lane; col < num_cols; col += 32u) {
            const uint base = col * 4u;
            const half4 k4 = half4(half(vec[base + 0u]),
                                   half(vec[base + 1u]),
                                   half(vec[base + 2u]),
                                   half(vec[base + 3u]));
            slot.k_tex.write(k4, uint2(col, row));
            slot.v_tex.write(v_src[col], uint2(col, row));
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 12a (C14 + R5): Multi-Request Batched Tiered GQA Decode Split-KV
// Executes `batch_size` (`1..16+`) independent decode requests with per-slot
// `(seq_len[b], tile_table_offset[b], retired_offset[b], num_retired_chunks[b])`
// in ONE GPU dispatch.
//
// Specialized on ATTN_HEAD_DIM ({64, 128}).
// Grid: (num_kv_heads * q_groups_per_kv, num_splits, batch_size) threadgroups,
//       256 threads.
// ---------------------------------------------------------------------------
kernel void ami_batch_gqa_decode_attn_split(const device half *q [[buffer(0)]],
                                            const device KvTilePair *kv_tiles [[buffer(1)]],
                                            const device uchar *retired_pool [[buffer(2)]],
                                            const device uint *retired_chunk_ids [[buffer(3)]],
                                            const device BatchGqaSlotDesc *slots [[buffer(4)]],
                                            device half *out [[buffer(5)]],
                                            device float *partial_m [[buffer(6)]],
                                            device float *partial_l [[buffer(7)]],
                                            device float *partial_o [[buffer(8)]],
                                            constant BatchGqaDecodeParams &p [[buffer(9)]],
                                            uint3 tg_pos [[threadgroup_position_in_grid]],
                                            uint tid [[thread_index_in_threadgroup]],
                                            uint sg_id [[simdgroup_index_in_threadgroup]],
                                            uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint Q_TILE = 8u;
    constexpr float LOG2E = 1.4426950408889634f;

    const uint kv_head = tg_pos.x / p.q_groups_per_kv;
    const uint q_grp = tg_pos.x % p.q_groups_per_kv;
    const uint split_idx = tg_pos.y;
    const uint b = tg_pos.z;
    if (kv_head >= p.num_kv_heads || split_idx >= p.num_splits || b >= p.batch_size) {
        return;
    }

    const uint gqa_ratio = p.num_q_heads / p.num_kv_heads;
    const uint q_start = kv_head * gqa_ratio + q_grp * Q_TILE;
    const uint q_end = min(q_start + Q_TILE, (kv_head + 1u) * gqa_ratio);
    const uint q_cnt = q_end - q_start;

    const BatchGqaSlotDesc slot = slots[b];
    const uint seq_len = slot.seq_len;
    const uint tokens_per_split = (seq_len > 0u) ? ((seq_len + p.num_splits - 1u) / p.num_splits) : 0u;
    const uint t_start = split_idx * tokens_per_split;
    const uint t_end = min(t_start + tokens_per_split, seq_len);

    if (t_start >= t_end) {
        if (p.num_splits > 1u && tid < q_cnt) {
            const ulong ps = (ulong(b) * p.num_q_heads + (q_start + tid)) * p.num_splits + split_idx;
            partial_m[ps] = -INFINITY;
            partial_l[ps] = 0.0f;
        } else if (p.num_splits == 1u && sg_id < q_cnt) {
            const uint qh = q_start + sg_id;
            const uint num_texels = ATTN_HEAD_DIM >> 2;
            if (lane < num_texels) {
                *(device half4 *)(out + (ulong(b) * p.num_q_heads + qh) * ATTN_HEAD_DIM + lane * 4u) = half4(0.0h);
            }
        }
        return;
    }

    const bool active_q = (sg_id < q_cnt);
    const uint qh = q_start + min(sg_id, q_cnt - 1u);
    const float scale_log2e = p.scale * LOG2E;
    const uint head_row_base = kv_head * p.chunk_tokens;

    threadgroup half4 tg_k0[256];
    threadgroup half4 tg_k1[256];
    threadgroup half4 tg_v0[256];
    threadgroup half4 tg_v1[256];

    if (ATTN_HEAD_DIM == 128u) {
        const ulong q_off = (ulong(b) * p.num_q_heads + qh) * 128u + lane * 4u;
        const float4 q_vec = active_q ? float4(*(const device half4 *)(q + q_off)) : float4(0.0f);
        float m_val = -INFINITY;
        float l_val = 0.0f;
        float4 acc = float4(0.0f);

        gqa_tiered_attend_range_d128(t_start, t_end, p.chunk_tokens, head_row_base,
                                     slot.num_retired_chunks, slot.tile_table_offset, slot.retired_offset,
                                     p.retired_plane_stride_bytes, p.retired_chunk_stride_bytes,
                                     scale_log2e, kv_tiles, retired_pool, retired_chunk_ids,
                                     q_vec, active_q, tid, lane,
                                     tg_k0, tg_k1, tg_v0, tg_v1, m_val, l_val, acc);

        if (active_q) {
            if (p.num_splits == 1u) {
                const float inv_l = (l_val > 0.0f) ? (1.0f / l_val) : 0.0f;
                *(device half4 *)(out + (ulong(b) * p.num_q_heads + qh) * 128u + lane * 4u) = half4(acc * inv_l);
            } else {
                const ulong ps = (ulong(b) * p.num_q_heads + qh) * p.num_splits + split_idx;
                if (lane == 0u) {
                    partial_m[ps] = m_val;
                    partial_l[ps] = l_val;
                }
                *(device float4 *)(partial_o + ps * 128u + lane * 4u) = acc;
            }
        }
    } else {
        const uint col = lane & 15u;
        const uint half_id = lane >> 4;
        const ulong q_off = (ulong(b) * p.num_q_heads + qh) * 64u + col * 4u;
        const float4 q_vec = active_q ? float4(*(const device half4 *)(q + q_off)) : float4(0.0f);
        float m_val = -INFINITY;
        float l_val = 0.0f;
        float4 acc = float4(0.0f);

        gqa_tiered_attend_range_d64(t_start, t_end, p.chunk_tokens, head_row_base,
                                    slot.num_retired_chunks, slot.tile_table_offset, slot.retired_offset,
                                    p.retired_plane_stride_bytes, p.retired_chunk_stride_bytes,
                                    scale_log2e, kv_tiles, retired_pool, retired_chunk_ids,
                                    q_vec, active_q, tid, col, half_id,
                                    tg_k0, tg_k1, tg_v0, tg_v1, m_val, l_val, acc);

        if (active_q) {
            const float other_m = simd_shuffle_xor(m_val, 16u);
            const float other_l = simd_shuffle_xor(l_val, 16u);
            const float4 other_acc = float4(simd_shuffle_xor(acc.x, 16u),
                                            simd_shuffle_xor(acc.y, 16u),
                                            simd_shuffle_xor(acc.z, 16u),
                                            simd_shuffle_xor(acc.w, 16u));
            const float m_final = max(m_val, other_m);
            const float w_self = (l_val > 0.0f) ? exp2(m_val - m_final) : 0.0f;
            const float w_other = (other_l > 0.0f) ? exp2(other_m - m_final) : 0.0f;
            const float l_final = fma(w_self, l_val, w_other * other_l);
            const float4 acc_final = fma(float4(w_self), acc, float4(w_other) * other_acc);

            if (lane < 16u) {
                if (p.num_splits == 1u) {
                    const float inv_l = (l_final > 0.0f) ? (1.0f / l_final) : 0.0f;
                    *(device half4 *)(out + (ulong(b) * p.num_q_heads + qh) * 64u + col * 4u) = half4(acc_final * inv_l);
                } else {
                    const ulong ps = (ulong(b) * p.num_q_heads + qh) * p.num_splits + split_idx;
                    if (lane == 0u) {
                        partial_m[ps] = m_final;
                        partial_l[ps] = l_final;
                    }
                    *(device float4 *)(partial_o + ps * 64u + col * 4u) = acc_final;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 12b (C14): Multi-Request Batched Split-KV Reduction
//
// Specialized on ATTN_HEAD_DIM ({64, 128}).
// Grid: (num_q_heads, batch_size, 1) threadgroups, 32 threads.
// ---------------------------------------------------------------------------
kernel void ami_batch_gqa_decode_attn_reduce(const device float *partial_m [[buffer(0)]],
                                             const device float *partial_l [[buffer(1)]],
                                             const device float *partial_o [[buffer(2)]],
                                             device half *out [[buffer(3)]],
                                             constant BatchGqaDecodeParams &p [[buffer(4)]],
                                             uint2 tg_pos [[threadgroup_position_in_grid]],
                                             uint lane [[thread_index_in_simdgroup]]) {
    const uint qh = tg_pos.x;
    const uint b = tg_pos.y;
    const uint num_texels = ATTN_HEAD_DIM >> 2;
    if (qh >= p.num_q_heads || b >= p.batch_size || lane >= num_texels) {
        return;
    }

    const ulong head_base = (ulong(b) * p.num_q_heads + qh) * p.num_splits;
    float m_max = -INFINITY;
    for (uint s = 0u; s < p.num_splits; s++) {
        if (partial_l[head_base + s] > 0.0f) {
            m_max = max(m_max, partial_m[head_base + s]);
        }
    }

    float l_tot = 0.0f;
    float4 o_tot = float4(0.0f);
    for (uint s = 0u; s < p.num_splits; s++) {
        const float ls = partial_l[head_base + s];
        if (ls > 0.0f) {
            const float w = exp2(partial_m[head_base + s] - m_max);
            l_tot = fma(ls, w, l_tot);
            const float4 po = *(const device float4 *)(partial_o + (head_base + s) * ATTN_HEAD_DIM + lane * 4u);
            o_tot = fma(po, float4(w), o_tot);
        }
    }

    const float inv_l = (l_tot > 0.0f) ? (1.0f / l_tot) : 0.0f;
    *(device half4 *)(out + (ulong(b) * p.num_q_heads + qh) * ATTN_HEAD_DIM + lane * 4u) = half4(o_tot * inv_l);
}
