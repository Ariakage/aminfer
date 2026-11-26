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
// G26 asymmetric K/V head dimensions (`0` = dynamic <= 256, `192`/`128`, `256`/`256`).
constant uint G26_GQA_K_DIM [[function_constant(4)]];
constant uint G26_GQA_V_DIM [[function_constant(5)]];

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
    uint unit_offset;
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
    const float w_add1 = (p.unit_offset != 0u) ? 1.0f : 0.0f;
    if ((p.hidden_dim & 3u) == 0u) {
        const float4 w_add4 = float4(w_add1);
        const uint vec_len = p.hidden_dim >> 2;
        const device half4 *x4 = (const device half4 *)x_row;
        device half4 *y4 = (device half4 *)y_row;
        for (uint i = tid; i < vec_len; i += 256u) {
            const float4 v = (float4(x4[i]) * inv_rms) * (norm_w4(w, i, NORM_W_BF16) + w_add4);
            y4[i] = half4(v);
        }
    } else {
        for (uint i = tid; i < p.hidden_dim; i += 256u) {
            const float v = (float(x_row[i]) * inv_rms) * (norm_w1(w, i, NORM_W_BF16) + w_add1);
            y_row[i] = half(v);
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 1b (`G18`, ADR-0023): Fused ResidualAdd + RMSNorm
//
// Reads `x` and `delta`, updates `x` in place (`x_residual = half(x + delta)`),
// caches up to 8 `half4` chunks per thread (hidden_dim <= 8192) directly in
// registers with compile-time unrolled constant indexing (0 `alloca` / 0 stack
// spills, 0 intermediate DRAM re-read, 0 intermediate GPU dispatch barrier),
// and emits normalized `x_norm` in the same single dispatch.
// Specialized on NORM_W_BF16 (`function_constant(2)`).
// Grid: (num_rows, 1, 1) threadgroups, 256 threads (8 simdgroups) per row.
// ---------------------------------------------------------------------------
kernel void ami_fused_residual_rmsnorm(device half *x [[buffer(0)]],
                                       const device half *delta [[buffer(1)]],
                                       const device uchar *w [[buffer(2)]],
                                       device half *x_norm [[buffer(3)]],
                                       constant RmsNormParams &p [[buffer(4)]],
                                       uint row [[threadgroup_position_in_grid]],
                                       uint tid [[thread_index_in_threadgroup]],
                                       uint sg_id [[simdgroup_index_in_threadgroup]],
                                       uint lane [[thread_index_in_simdgroup]]) {
    if (row >= p.num_rows) {
        return;
    }

    device half *x_row = x + ulong(row) * p.hidden_dim;
    const device half *d_row = delta + ulong(row) * p.hidden_dim;
    device half *y_row = x_norm + ulong(row) * p.hidden_dim;

    threadgroup float sg_sums[8];
    float sum_sq = 0.0f;

    if ((p.hidden_dim & 3u) == 0u) {
        const uint vec_len = p.hidden_dim >> 2;
        device half4 *x4 = (device half4 *)x_row;
        const device half4 *d4 = (const device half4 *)d_row;
        device half4 *y4 = (device half4 *)y_row;

        half4 reg_x[8];
        #pragma clang loop unroll(full)
        for (uint c = 0u; c < 8u; c++) {
            const uint i = tid + c * 256u;
            if (i < vec_len) {
                const half4 s = half4(float4(x4[i]) + float4(d4[i]));
                x4[i] = s;
                reg_x[c] = s;
                const float4 fv = float4(s);
                sum_sq += dot(fv, fv);
            } else {
                reg_x[c] = half4(0.0h);
            }
        }
        for (uint i = tid + 8u * 256u; i < vec_len; i += 256u) {
            const half4 s = half4(float4(x4[i]) + float4(d4[i]));
            x4[i] = s;
            const float4 fv = float4(s);
            sum_sq += dot(fv, fv);
        }

        const float sg_tot = simd_sum(sum_sq);
        if (lane == 0u) {
            sg_sums[sg_id] = sg_tot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0u) {
            float total = 0.0f;
            #pragma clang loop unroll(full)
            for (uint s = 0u; s < 8u; s++) {
                total += sg_sums[s];
            }
            sg_sums[0] = rsqrt(total / float(p.hidden_dim) + p.eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float inv_rms = sg_sums[0];
        const float4 w_add4 = float4((p.unit_offset != 0u) ? 1.0f : 0.0f);
        #pragma clang loop unroll(full)
        for (uint c = 0u; c < 8u; c++) {
            const uint i = tid + c * 256u;
            if (i < vec_len) {
                const float4 v = (float4(reg_x[c]) * inv_rms) * (norm_w4(w, i, NORM_W_BF16) + w_add4);
                y4[i] = half4(v);
            }
        }
        for (uint i = tid + 8u * 256u; i < vec_len; i += 256u) {
            const float4 v = (float4(x4[i]) * inv_rms) * (norm_w4(w, i, NORM_W_BF16) + w_add4);
            y4[i] = half4(v);
        }
    } else {
        half reg_s[8];
        #pragma clang loop unroll(full)
        for (uint c = 0u; c < 8u; c++) {
            const uint i = tid + c * 256u;
            if (i < p.hidden_dim) {
                const half s = half(float(x_row[i]) + float(d_row[i]));
                x_row[i] = s;
                reg_s[c] = s;
                const float fv = float(s);
                sum_sq = fma(fv, fv, sum_sq);
            } else {
                reg_s[c] = half(0.0h);
            }
        }
        for (uint i = tid + 8u * 256u; i < p.hidden_dim; i += 256u) {
            const half s = half(float(x_row[i]) + float(d_row[i]));
            x_row[i] = s;
            const float fv = float(s);
            sum_sq = fma(fv, fv, sum_sq);
        }

        const float sg_tot = simd_sum(sum_sq);
        if (lane == 0u) {
            sg_sums[sg_id] = sg_tot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0u) {
            float total = 0.0f;
            #pragma clang loop unroll(full)
            for (uint s = 0u; s < 8u; s++) {
                total += sg_sums[s];
            }
            sg_sums[0] = rsqrt(total / float(p.hidden_dim) + p.eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float inv_rms = sg_sums[0];
        const float w_add1 = (p.unit_offset != 0u) ? 1.0f : 0.0f;
        #pragma clang loop unroll(full)
        for (uint c = 0u; c < 8u; c++) {
            const uint i = tid + c * 256u;
            if (i < p.hidden_dim) {
                const float v = (float(reg_s[c]) * inv_rms) * (norm_w1(w, i, NORM_W_BF16) + w_add1);
                y_row[i] = half(v);
            }
        }
        for (uint i = tid + 8u * 256u; i < p.hidden_dim; i += 256u) {
            const float v = (float(x_row[i]) * inv_rms) * (norm_w1(w, i, NORM_W_BF16) + w_add1);
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

// ===========================================================================
// Milestone M6.8 (G15, ADR-0020): Wide-Chunk Direct-KvTile Prefill Kernels
//   - `ami_prefill_qknorm_rope_kv_write`: QK-Norm + RoPE/YaRN/3D-MRoPE +
//     direct bindless RGBA16Float KvTile writes for `1 .. 1024+` tokens
//     across 512-token tile boundaries (zero `tree_k`/`tree_v` staging).
//   - `ami_gqa_causal_prefill_flash_attn`: Block-tiled (`BQ = 32, BK = 32`)
//     hardware `simdgroup_matrix` causal FlashAttention-2 reading seamlessly
//     across R5 retired group-64 `q8` chunks and active `RGBA16Float` KvTiles,
//     strictly skipping upper-triangular KV blocks (`kv_block_start > max_q_pos`)
//     and out-of-window blocks (`sliding_window > 0`).
// ===========================================================================

struct KvTileWritePair {
    texture2d<half, access::write> k_tex;
    texture2d<half, access::write> v_tex;
};

static_assert(sizeof(KvTileWritePair) == 16, "KvTileWritePair must match two u64 MTLResourceIDs");

struct PrefillQkNormRopeKvParams {
    uint num_tokens;
    uint num_q_heads;
    uint num_kv_heads;
    uint head_dim;
    uint rotary_dim;
    uint start_pos;
    uint chunk_tokens;
    uint num_retired_chunks;
    uint tile_table_offset;
    uint ring_capacity;
    uint apply_qk_norm;
    uint kv_only;
    uint use_mrope;
    uint mrope_s0;
    uint mrope_s1;
    uint mrope_s2;
    float rms_eps;
    float rope_theta;
    float rope_yarn_scale;
    float yarn_mscale;
    float yarn_beta_fast;
    float yarn_beta_slow;
    uint original_max_pos;
    uint _pad0;
};

struct CausalPrefillAttnParams {
    uint num_q;
    uint q_start_pos;
    uint kv_len;
    uint num_q_heads;
    uint num_kv_heads;
    uint head_dim;
    uint chunk_tokens;
    uint sliding_window;
    uint ring_capacity;
    float scale;
    uint num_retired_chunks;
    uint tile_table_offset;
    uint retired_offset;
    uint retired_plane_stride_bytes;
    uint retired_chunk_stride_bytes;
    uint _pad0;
};

static_assert(sizeof(PrefillQkNormRopeKvParams) == 96, "PrefillQkNormRopeKvParams must be 96 bytes");
static_assert(sizeof(CausalPrefillAttnParams) == 64, "CausalPrefillAttnParams must be 64 bytes");

static inline uint prefill_mrope_axis(uint j, uint s0, uint s1, uint s2) {
    const uint m1 = min(s0, min(s1, s2));
    const uint three_m1 = m1 * 3u;
    if (j < three_m1) {
        return j % 3u;
    }
    const uint rem = j - three_m1;
    const uint a0 = s0 - m1;
    const uint a1 = s1 - m1;
    const uint a2 = s2 - m1;
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

static inline float prefill_yarn_inv_freq(uint j,
                                          uint rotary_dim,
                                          float rope_theta,
                                          float yarn_scale,
                                          float beta_fast,
                                          float beta_slow,
                                          uint original_max_pos) {
    const float log_theta = log(rope_theta);
    const float inv_freq_extrap = exp(-log_theta * (2.0f * float(j) / float(rotary_dim)));
    if (yarn_scale <= 1.0f || original_max_pos == 0u) {
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
    const float inv_freq_interp = inv_freq_extrap / yarn_scale;
    return fma(inv_freq_extrap, 1.0f - ramp, inv_freq_interp * ramp);
}

// Direct-KvTile wide prefill QK-Norm + RoPE (with optional YaRN & 3D MRoPE).
// Specialized on NORM_W_BF16 (`function_constant(2)`).
// Grid: ((kv_only ? num_kv_heads : num_q_heads + num_kv_heads), num_tokens, 1), 32 threads.
kernel void ami_prefill_qknorm_rope_kv_write(device half *q [[buffer(0)]],
                                             const device half *k [[buffer(1)]],
                                             const device half *v [[buffer(2)]],
                                             const device uchar *q_norm [[buffer(3)]],
                                             const device uchar *k_norm [[buffer(4)]],
                                             const device KvTileWritePair *kv_tiles [[buffer(5)]],
                                             const device uint *mrope_pos [[buffer(6)]],
                                             constant PrefillQkNormRopeKvParams &p [[buffer(7)]],
                                             uint2 tg_pos [[threadgroup_position_in_grid]],
                                             uint lane [[thread_index_in_simdgroup]]) {
    const uint total_heads = (p.kv_only != 0u) ? p.num_kv_heads : (p.num_q_heads + p.num_kv_heads);
    const uint tok = tg_pos.y;
    if (tg_pos.x >= total_heads || tok >= p.num_tokens) {
        return;
    }

    const bool is_q = (p.kv_only == 0u) && (tg_pos.x < p.num_q_heads);
    const uint head_idx = is_q ? tg_pos.x : (tg_pos.x - ((p.kv_only != 0u) ? 0u : p.num_q_heads));

    // When writing into a circular ring smaller than num_tokens, skip earlier
    // overwritten KV tokens so no two threadgroups race on the same ring slot.
    if (!is_q && p.ring_capacity > 0u && (tok + p.ring_capacity < p.num_tokens)) {
        return;
    }

    threadgroup float vec[256];
    const uint d_dim = p.head_dim;
    const uint rot_dim = (p.rotary_dim > 0u) ? p.rotary_dim : d_dim;
    const uint half_rot = rot_dim >> 1;

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
            vec[d] = (vec[d] * inv_rms) * norm_w1(nw, d, NORM_W_BF16);
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    const uint lin_pos = p.start_pos + tok;
    const float mscale = (p.yarn_mscale > 0.0f) ? p.yarn_mscale : 1.0f;
    for (uint j = lane; j < half_rot; j += 32u) {
        uint axis_pos = lin_pos;
        if (p.use_mrope != 0u) {
            const uint axis = prefill_mrope_axis(j, p.mrope_s0, p.mrope_s1, p.mrope_s2);
            axis_pos = mrope_pos[ulong(tok) * 3ul + axis];
        }
        const float inv_f = prefill_yarn_inv_freq(
            j, rot_dim, p.rope_theta, p.rope_yarn_scale,
            p.yarn_beta_fast, p.yarn_beta_slow, p.original_max_pos);
        const float angle = float(axis_pos) * inv_f;
        const float c = cos(angle) * mscale;
        const float s = sin(angle) * mscale;
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
        const uint phys_pos = (p.ring_capacity > 0u) ? (lin_pos % p.ring_capacity) : lin_pos;
        const uint chunk = phys_pos / p.chunk_tokens;
        const uint slot_in_chunk = phys_pos - chunk * p.chunk_tokens;
        const uint tile_idx = p.tile_table_offset +
            ((p.ring_capacity == 0u) ? (chunk - p.num_retired_chunks) : chunk);
        const uint row = head_idx * p.chunk_tokens + slot_in_chunk;
        const KvTileWritePair tile = kv_tiles[tile_idx];
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
    }
}

template <uint D, uint D_PAD, uint BK, bool IS_V>
static inline void flash_prefill_load_kv(threadgroup half4 *KVs4,
                                         const device KvTilePair *kv_tiles,
                                         const device uchar *retired_pool,
                                         const device uint *retired_chunk_ids,
                                         constant CausalPrefillAttnParams &p,
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
        const uint kv_pos = kv_block_start + row;
        half4 val = half4(0.0h);
        if (kv_pos < kv_block_end) {
            const uint phys_pos = (p.ring_capacity > 0u) ? (kv_pos % p.ring_capacity) : kv_pos;
            const uint chunk = phys_pos / p.chunk_tokens;
            const uint slot = phys_pos - chunk * p.chunk_tokens;
            const uint tex_row = kv_head * p.chunk_tokens + slot;
            if (p.ring_capacity == 0u && chunk < p.num_retired_chunks) {
                const uint phys_slot = retired_chunk_ids[p.retired_offset + chunk];
                const device uchar *plane = retired_pool +
                    ulong(phys_slot) * p.retired_chunk_stride_bytes +
                    (IS_V ? p.retired_plane_stride_bytes : 0u);
                val = load_q8_half4(plane, tex_row, GROUPS_PER_HEAD, load_col4);
            } else {
                const uint active_chunk = (p.ring_capacity == 0u) ? (chunk - p.num_retired_chunks) : chunk;
                const KvTilePair pair = kv_tiles[p.tile_table_offset + active_chunk];
                val = IS_V ? pair.v_tex.read(uint2(load_col4, tex_row))
                           : pair.k_tex.read(uint2(load_col4, tex_row));
            }
        }
        KVs4[row * D_PAD4 + load_col4] = val;
    }
}

// ---------------------------------------------------------------------------
// Scalar/vector causal FlashAttention-2 prefill (`G19` short-sequence guard,
// ADR-0023 §2): used strictly when `Q_len < 32` or `KV_len < 64` (avoiding
// `8x8` MMA tile padding waste and threadgroup barriers on short prompts and
// narrow draft trees), and as the parity baseline for `G19` MMA/NAX tests.
// Specialized on ATTN_HEAD_DIM ({64, 128}).
// Grid: (num_q_heads, num_q, 1) threadgroups, 32 threads (1 simdgroup).
// ---------------------------------------------------------------------------
kernel void ami_gqa_causal_prefill_flash_attn_vec(const device half *q [[buffer(0)]],
                                                  const device KvTilePair *kv_tiles [[buffer(1)]],
                                                  const device uchar *retired_pool [[buffer(2)]],
                                                  const device uint *retired_chunk_ids [[buffer(3)]],
                                                  device half *out [[buffer(4)]],
                                                  constant CausalPrefillAttnParams &p [[buffer(5)]],
                                                  uint2 tg_pos [[threadgroup_position_in_grid]],
                                                  uint lane [[thread_index_in_simdgroup]]) {
    constexpr float LOG2E = 1.4426950408889634f;
    const uint qh = tg_pos.x;
    const uint qi = tg_pos.y;
    if (qh >= p.num_q_heads || qi >= p.num_q) {
        return;
    }

    const uint gqa_ratio = p.num_q_heads / p.num_kv_heads;
    const uint kv_head = qh / gqa_ratio;
    const uint qp = p.q_start_pos + qi;
    const uint kv_end = min(p.kv_len, qp + 1u);
    const uint kv_begin = (p.sliding_window > 0u && (qp + 1u) > p.sliding_window)
        ? (qp + 1u - p.sliding_window)
        : 0u;
    const float scale_log2e = p.scale * LOG2E;

    const uint d_dim = ATTN_HEAD_DIM;
    const uint d_vec4 = d_dim >> 2;
    const uint groups_per_head = d_dim >> 6;
    const bool active_lane = (lane < d_vec4);
    const ulong q_row_off = (ulong(qi) * p.num_q_heads + qh) * d_dim;

    const float4 q_vec = active_lane
        ? float4(*(const device half4 *)(q + q_row_off + lane * 4u))
        : float4(0.0f);
    float m_val = -INFINITY;
    float l_val = 0.0f;
    float4 acc = float4(0.0f);

    for (uint kv_pos = kv_begin; kv_pos < kv_end; kv_pos++) {
        half4 k4 = half4(0.0h);
        half4 v4 = half4(0.0h);
        if (active_lane) {
            const uint phys_pos = (p.ring_capacity > 0u) ? (kv_pos % p.ring_capacity) : kv_pos;
            const uint chunk = phys_pos / p.chunk_tokens;
            const uint slot = phys_pos - chunk * p.chunk_tokens;
            const uint tex_row = kv_head * p.chunk_tokens + slot;
            if (p.ring_capacity == 0u && chunk < p.num_retired_chunks) {
                const uint phys_slot = retired_chunk_ids[p.retired_offset + chunk];
                const device uchar *k_plane = retired_pool + ulong(phys_slot) * p.retired_chunk_stride_bytes;
                const device uchar *v_plane = k_plane + p.retired_plane_stride_bytes;
                k4 = load_q8_half4(k_plane, tex_row, groups_per_head, lane);
                v4 = load_q8_half4(v_plane, tex_row, groups_per_head, lane);
            } else {
                const uint active_chunk = (p.ring_capacity == 0u) ? (chunk - p.num_retired_chunks) : chunk;
                const KvTilePair pair = kv_tiles[p.tile_table_offset + active_chunk];
                k4 = pair.k_tex.read(uint2(lane, tex_row));
                v4 = pair.v_tex.read(uint2(lane, tex_row));
            }
        }
        const float partial = active_lane ? dot(q_vec, float4(k4)) : 0.0f;
        const float score = simd_sum(partial) * scale_log2e;
        const float new_m = max(m_val, score);
        const float corr = (l_val > 0.0f) ? exp2(m_val - new_m) : 0.0f;
        const float p_w = exp2(score - new_m);
        l_val = fma(l_val, corr, p_w);
        if (active_lane) {
            acc = fma(acc, float4(corr), float4(v4) * p_w);
        }
        m_val = new_m;
    }

    if (active_lane) {
        const float inv_l = (l_val > 0.0f) ? (1.0f / l_val) : 0.0f;
        *(device half4 *)(out + q_row_off + lane * 4u) = half4(acc * inv_l);
    }
}

template <uint D, uint D_PAD>
static inline void flash_prefill_exec_bk32(const device half *q,
                                           const device KvTilePair *kv_tiles,
                                           const device uchar *retired_pool,
                                           const device uint *retired_chunk_ids,
                                           device half *out,
                                           constant CausalPrefillAttnParams &p,
                                           threadgroup half4 *KVs4,
                                           threadgroup float *Ss,
                                           threadgroup half *Ps,
                                           uint2 tg_pos,
                                           uint tid,
                                           uint sg_id,
                                           uint lane) {
    constexpr uint BQ = 32u;
    constexpr uint BK = 32u;
    constexpr uint D_VEC4 = D >> 2;
    constexpr uint D_PAD4 = D_PAD >> 2;
    constexpr uint ROW_STEP = 128u / D_VEC4;
    constexpr uint D_COLS32 = D >> 5;
    constexpr uint D_TILES8 = D >> 3;
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
    const uint kv_begin = (p.sliding_window > 0u && (min_q_pos + 1u) > p.sliding_window)
        ? (min_q_pos + 1u - p.sliding_window)
        : 0u;

    // 1. Stage Q tile [BQ=32, D] into KVs4 once, hoist each simdgroup's 8xD
    // Q slice into simdgroup_half8x8 fragments, then reuse KVs4 for K/V.
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
    float o_reg[8][D_COLS32];

    #pragma clang loop unroll(full)
    for (uint r = 0u; r < 8u; r++) {
        m_val[r] = -INFINITY;
        l_val[r] = 0.0f;
        #pragma clang loop unroll(full)
        for (uint c = 0u; c < D_COLS32; c++) {
            o_reg[r][c] = 0.0f;
        }
    }

    const float scale_log2e = p.scale * LOG2E;

    // 2. Stream BK = 32 KV blocks across [kv_begin, kv_end).
    for (uint kv_block_start = kv_begin; kv_block_start < kv_end; kv_block_start += BK) {
        const uint kv_block_end = min(kv_block_start + BK, kv_end);

        flash_prefill_load_kv<D, D_PAD, BK, false>(
            KVs4, kv_tiles, retired_pool, retired_chunk_ids, p,
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
                               (p.sliding_window == 0u || (qp - kv_pos) < p.sliding_window);
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

        // Single threadgroup barrier synchronizes both Ps writes and K reads before V load.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        flash_prefill_load_kv<D, D_PAD, BK, true>(
            KVs4, kv_tiles, retired_pool, retired_chunk_ids, p,
            kv_head, kv_block_start, kv_block_end, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (uint c = 0u; c < D_COLS32; c++) {
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
            if (c + 1u < D_COLS32) {
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
        for (uint c = 0u; c < D_COLS32; c++) {
            out[out_base + c * 32u + lane] = half(o_reg[r][c] * inv_l);
        }
    }
}

// Wide-KV (`BQ = 32, BK = 64`) Apple9 `simdgroup_half8x8` FlashAttention-2
// for `head_dim = 64`: fits in 15,872 B (`<= 16 KiB`) bank-conflict-free
// padded SRAM (`D_PAD = 72`, `BK_PAD = 72`), halving KV outer-loop iterations.
union FlashPrefillD64Union {
    half4 KVs4[64u * 18u];  // 9,216 B
    float Ss[32u * 64u];    // 8,192 B
};

static inline void flash_prefill_exec_d64_bk64(const device half *q,
                                               const device KvTilePair *kv_tiles,
                                               const device uchar *retired_pool,
                                               const device uint *retired_chunk_ids,
                                               device half *out,
                                               constant CausalPrefillAttnParams &p,
                                               threadgroup FlashPrefillD64Union &u_smem,
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
    const uint kv_begin = (p.sliding_window > 0u && (min_q_pos + 1u) > p.sliding_window)
        ? (min_q_pos + 1u - p.sliding_window)
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

        // 2a. Load K [64, 64] into u_smem.KVs4.
        flash_prefill_load_kv<D, D_PAD, BK, false>(
            u_smem.KVs4, kv_tiles, retired_pool, retired_chunk_ids, p,
            kv_head, kv_block_start, kv_block_end, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2b. S = Q[8, 64] @ K[64, 64]^T across 8 8x8 MMA tiles.
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
        // All simdgroups finish reading K before aliasing u_smem as Ss.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (uint bj = 0u; bj < 8u; bj++) {
            simdgroup_store(s_frag[bj], u_smem.Ss + r0 * BK + bj * 8u, BK);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // 2c. Causal + sliding-window mask & FP32 base-2 online softmax (2 KV cols/lane).
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
                            (p.sliding_window == 0u || (qp - kv_pos0) < p.sliding_window);
            const bool v1 = act && in_blk1 && (kv_pos1 <= qp) &&
                            (p.sliding_window == 0u || (qp - kv_pos1) < p.sliding_window);
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

        // 2d. Load V [64, 64] into u_smem.KVs4 after Ss reads & Ps writes complete.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        flash_prefill_load_kv<D, D_PAD, BK, true>(
            u_smem.KVs4, kv_tiles, retired_pool, retired_chunk_ids, p,
            kv_head, kv_block_start, kv_block_end, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2e. dO = P[8, 64] @ V[64, 64] in 4 x 16-col passes through Os[32 * 16].
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

// Block-tiled Apple9 `simdgroup_half8x8` causal FlashAttention-2 prefill (`G19`,
// ADR-0023 §2): `BQ = 32, BK = 64` for `D = 64` (15,872 B <= 16 KiB SRAM) and
// `BQ = 32, BK = 32` for `D = 128` (15,360 B <= 16 KiB SRAM).
// Gated strictly on `Q_len >= 32` and `KV_len >= 64` by `src/gpu/attention.zig`.
// Grid: (num_kv_heads, ceil(num_q * gqa_ratio / 32), 1) threadgroups, 128 threads.
kernel void ami_gqa_causal_prefill_flash_attn(const device half *q [[buffer(0)]],
                                              const device KvTilePair *kv_tiles [[buffer(1)]],
                                              const device uchar *retired_pool [[buffer(2)]],
                                              const device uint *retired_chunk_ids [[buffer(3)]],
                                              device half *out [[buffer(4)]],
                                              constant CausalPrefillAttnParams &p [[buffer(5)]],
                                              uint2 tg_pos [[threadgroup_position_in_grid]],
                                              uint tid [[thread_index_in_threadgroup]],
                                              uint sg_id [[simdgroup_index_in_threadgroup]],
                                              uint lane [[thread_index_in_simdgroup]]) {
    // Unified <= 16 KiB threadgroup SRAM union across D=64 (15,872 B) and D=128 (15,360 B).
    union FlashPrefillSmem {
        struct {
            half4 KVs4[32u * 34u];  // 8,704 B
            float Ss[32u * 32u];    // 4,096 B
            half Ps[32u * 40u];     // 2,560 B (+8 half pad)
        } d128;                     // 15,360 B total
        struct {
            FlashPrefillD64Union u; // 9,216 B (KVs4[64*18] | Ss[32*64])
            half Ps[32u * 72u];     // 4,608 B (+8 half pad)
            float Os[32u * 16u];    // 2,048 B
        } d64;                      // 15,872 B total (<= 16 KiB)
    };
    threadgroup FlashPrefillSmem smem;

    if (ATTN_HEAD_DIM == 128u) {
        flash_prefill_exec_bk32<128u, 136u>(
            q, kv_tiles, retired_pool, retired_chunk_ids, out, p,
            smem.d128.KVs4, smem.d128.Ss, smem.d128.Ps, tg_pos, tid, sg_id, lane);
    } else {
        flash_prefill_exec_d64_bk64(
            q, kv_tiles, retired_pool, retired_chunk_ids, out, p,
            smem.d64.u, smem.d64.Ps, smem.d64.Os, tg_pos, tid, sg_id, lane);
    }
}

// ===========================================================================
// G26: Attention Output Sigmoid Gate (`qwen3_5` `q_gate_proj`) & Asymmetric
// `K/V <= 256` (`192/128`, `256/256`, dynamic) Global GQA Split-KV & Prefill
// with FP32 Online Softmax Sink Bias (`mimo_v2` & `qwen3_5`).
// ===========================================================================

static inline float g26_load_scalar_f32(const device uchar *buf, uint idx, uint dtype) {
    if (dtype == 2u) {
        return ((const device float *)buf)[idx];
    }
    if (dtype == 1u) {
        return float(((const device bfloat *)buf)[idx]);
    }
    return float(((const device half *)buf)[idx]);
}

struct MulSigmoidParams {
    uint count;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

static_assert(sizeof(MulSigmoidParams) == 16, "MulSigmoidParams must be 16 bytes");

// Elementwise `out[i] *= sigmoid(gate[i])` in FP32 (`qwen3_5` `attn_output_gate`).
// Grid: (ceil(count / 1024), 1, 1) threadgroups, 256 threads.
kernel void ami_mul_sigmoid_f16(device half *out [[buffer(0)]],
                                const device half *gate [[buffer(1)]],
                                constant MulSigmoidParams &p [[buffer(2)]],
                                uint gid [[thread_position_in_grid]]) {
    const uint base = gid * 4u;
    if (base + 4u <= p.count) {
        const float4 x = float4(*(const device half4 *)(out + base));
        const float4 g = float4(*(const device half4 *)(gate + base));
        const float4 sig = 1.0f / (1.0f + exp(-g));
        *(device half4 *)(out + base) = half4(x * sig);
        return;
    }
    for (uint i = base; i < p.count; i++) {
        const float x = float(out[i]);
        const float g = float(gate[i]);
        out[i] = half(x / (1.0f + exp(-g)));
    }
}

struct G26GqaAttnParams {
    uint num_nodes;
    uint num_q_heads;
    uint num_kv_heads;
    uint k_head_dim;
    uint v_head_dim;
    uint base_pos;
    uint chunk_tokens;
    uint num_splits;
    uint q_groups_per_kv;
    uint num_retired_chunks;
    uint retired_plane_stride_bytes;
    uint retired_chunk_stride_bytes;
    uint tile_table_offset;
    uint retired_offset;
    uint has_tree;
    uint has_sink;
    uint sink_dtype;     // 0 = f16, 1 = bf16, 2 = f32
    uint sliding_window; // 0 = global full attention, >0 = sliding window
    float scale;
    uint _pad0;
};

static_assert(sizeof(G26GqaAttnParams) == 80, "G26GqaAttnParams must be 80 bytes");

// Asymmetric K/V (`K <= 256, V <= 256`, specialized on `(192,128)`, `(256,256)`,
// or dynamic `(0,0)`) Split-KV Global GQA Decode (`M = 1`) and Speculative Tree
// Verification (`M = 1..64`) with optional FP32 online-softmax sink bias.
// Grid: (num_kv_heads * q_groups_per_kv, num_splits, 1) threadgroups, 256 threads.
kernel void ami_g26_gqa_split(const device half *q [[buffer(0)]],
                              const device KvTilePair *kv_tiles [[buffer(1)]],
                              const device half *k_tree [[buffer(2)]],
                              const device half *v_tree [[buffer(3)]],
                              const device ulong *ancestor_masks [[buffer(4)]],
                              const device uchar *sink_bias [[buffer(5)]],
                              const device uchar *retired_pool [[buffer(6)]],
                              const device uint *retired_chunk_ids [[buffer(7)]],
                              device half *out [[buffer(8)]],
                              device float *partial_m [[buffer(9)]],
                              device float *partial_l [[buffer(10)]],
                              device float *partial_o [[buffer(11)]],
                              constant G26GqaAttnParams &p [[buffer(12)]],
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

    const uint win_lo = (p.sliding_window > 0u && p.base_pos > p.sliding_window)
        ? (p.base_pos - p.sliding_window)
        : 0u;
    const uint span_len = p.base_pos - win_lo;
    uint t_start = win_lo;
    uint t_end = win_lo;
    if (span_len > 0u) {
        const uint tokens_per_split = (span_len + p.num_splits - 1u) / p.num_splits;
        t_start = min(win_lo + split_idx * tokens_per_split, p.base_pos);
        t_end = min(t_start + tokens_per_split, p.base_pos);
    }

    const bool do_prefix = (t_start < t_end);
    const bool do_tree = (split_idx == 0u && p.has_tree != 0u);
    const bool do_sink = (split_idx == 0u && p.has_sink != 0u);
    if (!do_prefix && !do_tree && !do_sink) {
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

    const uint k_dim = (G26_GQA_K_DIM > 0u) ? G26_GQA_K_DIM : p.k_head_dim;
    const uint v_dim = (G26_GQA_V_DIM > 0u) ? G26_GQA_V_DIM : p.v_head_dim;
    const uint k_cols = k_dim >> 2;
    const uint v_cols = v_dim >> 2;
    const bool can_retire_q8 = (p.num_retired_chunks > 0u && k_dim == v_dim && (k_dim & 63u) == 0u);
    const uint q8_groups = k_dim >> 6;

    const bool active_q = (sg_id < q_cnt);
    const uint safe_flat = q_start_flat + min(sg_id, q_cnt - 1u);
    const uint node = safe_flat / gqa_ratio;
    const uint qh = kv_head * gqa_ratio + (safe_flat - node * gqa_ratio);
    const ulong q_row_off = (ulong(node) * p.num_q_heads + qh) * k_dim;
    const float scale_log2e = p.scale * LOG2E;

    const float4 q0 = (active_q && lane < k_cols)
        ? float4(*(const device half4 *)(q + q_row_off + lane * 4u))
        : float4(0.0f);
    const float4 q1 = (active_q && (lane + 32u) < k_cols)
        ? float4(*(const device half4 *)(q + q_row_off + (lane + 32u) * 4u))
        : float4(0.0f);

    float m_val = -INFINITY;
    float l_val = 0.0f;
    if (do_sink && active_q) {
        m_val = g26_load_scalar_f32(sink_bias, qh, p.sink_dtype) * LOG2E;
        l_val = 1.0f;
    }
    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);

    // 4-token cooperative staging (4 x 64 half4 = 256 half4 each = 4 KiB total TG SRAM).
    threadgroup half4 tg_k[256];
    threadgroup half4 tg_v[256];

    if (do_prefix) {
        const uint load_tok = tid >> 6;  // 0..3
        const uint load_col = tid & 63u; // 0..63
        const uint head_row_base = kv_head * p.chunk_tokens;

        for (uint tile_t = t_start; tile_t < t_end; tile_t += 4u) {
            const uint tile_len = min(4u, t_end - tile_t);
            if (load_tok < tile_len) {
                const uint t = tile_t + load_tok;
                const uint chunk = t / p.chunk_tokens;
                const uint slot = t - chunk * p.chunk_tokens;
                const uint row = head_row_base + slot;
                if (can_retire_q8 && chunk < p.num_retired_chunks) {
                    const uint phys_slot = retired_chunk_ids[p.retired_offset + chunk];
                    const device uchar *k_plane = retired_pool + ulong(phys_slot) * p.retired_chunk_stride_bytes;
                    const device uchar *v_plane = k_plane + p.retired_plane_stride_bytes;
                    if (load_col < k_cols) {
                        tg_k[tid] = load_q8_half4(k_plane, row, q8_groups, load_col);
                    }
                    if (load_col < v_cols) {
                        tg_v[tid] = load_q8_half4(v_plane, row, q8_groups, load_col);
                    }
                } else {
                    const uint active_chunk = chunk - p.num_retired_chunks;
                    const KvTilePair pair = kv_tiles[p.tile_table_offset + active_chunk];
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
    }

    if (do_tree && active_q) {
        ulong rem = ancestor_masks[node];
        while (rem != 0ul) {
            const uint j = uint(ctz(rem));
            rem &= (rem - 1ul);
            if (j < p.num_nodes) {
                const ulong k_off = (ulong(j) * p.num_kv_heads + kv_head) * k_dim;
                const ulong v_off = (ulong(j) * p.num_kv_heads + kv_head) * v_dim;
                float partial = 0.0f;
                if (lane < k_cols) {
                    partial = dot(q0, float4(*(const device half4 *)(k_tree + k_off + lane * 4u)));
                }
                if ((lane + 32u) < k_cols) {
                    partial += dot(q1, float4(*(const device half4 *)(k_tree + k_off + (lane + 32u) * 4u)));
                }
                const float score = simd_sum(partial) * scale_log2e;
                const float new_m = max(m_val, score);
                const float corr = (l_val > 0.0f) ? exp2(m_val - new_m) : 0.0f;
                const float p_w = exp2(score - new_m);
                l_val = fma(l_val, corr, p_w);
                if (lane < v_cols) {
                    const float4 v0 = float4(*(const device half4 *)(v_tree + v_off + lane * 4u));
                    acc0 = fma(acc0, float4(corr), v0 * p_w);
                }
                if ((lane + 32u) < v_cols) {
                    const float4 v1 = float4(*(const device half4 *)(v_tree + v_off + (lane + 32u) * 4u));
                    acc1 = fma(acc1, float4(corr), v1 * p_w);
                }
                m_val = new_m;
            }
        }
    }

    if (!active_q) {
        return;
    }

    if (p.num_splits == 1u) {
        const float inv_l = (l_val > 0.0f) ? (1.0f / l_val) : 0.0f;
        const ulong out_off = (ulong(node) * p.num_q_heads + qh) * v_dim;
        if (lane < v_cols) {
            *(device half4 *)(out + out_off + lane * 4u) = half4(acc0 * inv_l);
        }
        if ((lane + 32u) < v_cols) {
            *(device half4 *)(out + out_off + (lane + 32u) * 4u) = half4(acc1 * inv_l);
        }
    } else {
        const ulong ps = (ulong(node) * p.num_q_heads + qh) * p.num_splits + split_idx;
        if (lane == 0u) {
            partial_m[ps] = m_val;
            partial_l[ps] = l_val;
        }
        device float4 *po4 = (device float4 *)(partial_o + ps * v_dim);
        if (lane < v_cols) {
            po4[lane] = acc0;
        }
        if ((lane + 32u) < v_cols) {
            po4[lane + 32u] = acc1;
        }
    }
}

// Split-KV reduction for `ami_g26_gqa_split` (`num_splits > 1`, `v_head_dim <= 256`).
// Grid: (num_q_heads, num_nodes, 1) threadgroups, 32 threads (1 simdgroup).
kernel void ami_g26_gqa_reduce(const device float *partial_m [[buffer(0)]],
                               const device float *partial_l [[buffer(1)]],
                               const device float *partial_o [[buffer(2)]],
                               device half *out [[buffer(3)]],
                               constant G26GqaAttnParams &p [[buffer(4)]],
                               uint2 tg_pos [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_simdgroup]]) {
    const uint qh = tg_pos.x;
    const uint node = tg_pos.y;
    if (qh >= p.num_q_heads || node >= p.num_nodes) {
        return;
    }

    const uint v_dim = (G26_GQA_V_DIM > 0u) ? G26_GQA_V_DIM : p.v_head_dim;
    const uint v_cols = v_dim >> 2;
    const ulong base_s = (ulong(node) * p.num_q_heads + qh) * p.num_splits;

    float global_m = -INFINITY;
    for (uint s = 0u; s < p.num_splits; s++) {
        const float pl = partial_l[base_s + s];
        if (pl > 0.0f) {
            global_m = max(global_m, partial_m[base_s + s]);
        }
    }

    float global_l = 0.0f;
    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);
    for (uint s = 0u; s < p.num_splits; s++) {
        const float pl = partial_l[base_s + s];
        if (pl <= 0.0f) {
            continue;
        }
        const float scale_s = exp2(partial_m[base_s + s] - global_m);
        global_l = fma(pl, scale_s, global_l);
        const device float4 *po4 = (const device float4 *)(partial_o + (base_s + s) * v_dim);
        if (lane < v_cols) {
            acc0 = fma(po4[lane], float4(scale_s), acc0);
        }
        if ((lane + 32u) < v_cols) {
            acc1 = fma(po4[lane + 32u], float4(scale_s), acc1);
        }
    }

    const float inv_l = (global_l > 0.0f) ? (1.0f / global_l) : 0.0f;
    const ulong out_off = (ulong(node) * p.num_q_heads + qh) * v_dim;
    if (lane < v_cols) {
        *(device half4 *)(out + out_off + lane * 4u) = half4(acc0 * inv_l);
    }
    if ((lane + 32u) < v_cols) {
        *(device half4 *)(out + out_off + (lane + 32u) * 4u) = half4(acc1 * inv_l);
    }
}

struct G26GqaPrefillParams {
    uint num_q;
    uint q_start_pos;
    uint kv_len;
    uint num_q_heads;
    uint num_kv_heads;
    uint k_head_dim;
    uint v_head_dim;
    uint chunk_tokens;
    uint sliding_window;
    uint ring_capacity;
    uint tile_table_offset;
    uint num_retired_chunks;
    uint retired_offset;
    uint retired_plane_stride_bytes;
    uint retired_chunk_stride_bytes;
    uint has_sink;
    uint sink_dtype;
    uint _pad0;
    float scale;
    uint _pad1;
};

static_assert(sizeof(G26GqaPrefillParams) == 80, "G26GqaPrefillParams must be 80 bytes");

// Asymmetric K/V (`K <= 256, V <= 256`) Causal Prefill Attention over committed
// `KvTilePair` chunks with optional FP32 online-softmax sink bias.
// Grid: (num_kv_heads * q_groups_per_kv, num_q, 1) threadgroups, 256 threads.
kernel void ami_g26_gqa_prefill(const device half *q [[buffer(0)]],
                                const device KvTilePair *kv_tiles [[buffer(1)]],
                                const device uchar *sink_bias [[buffer(2)]],
                                const device uchar *retired_pool [[buffer(3)]],
                                const device uint *retired_chunk_ids [[buffer(4)]],
                                device half *out [[buffer(5)]],
                                constant G26GqaPrefillParams &p [[buffer(6)]],
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

    const uint k_dim = (G26_GQA_K_DIM > 0u) ? G26_GQA_K_DIM : p.k_head_dim;
    const uint v_dim = (G26_GQA_V_DIM > 0u) ? G26_GQA_V_DIM : p.v_head_dim;
    const uint k_cols = k_dim >> 2;
    const uint v_cols = v_dim >> 2;
    const bool can_retire_q8 = (p.ring_capacity == 0u && p.num_retired_chunks > 0u &&
                                k_dim == v_dim && (k_dim & 63u) == 0u);
    const uint q8_groups = k_dim >> 6;
    const float scale_log2e = p.scale * LOG2E;

    const uint q_pos = p.q_start_pos + qi;
    const uint t_hi = min(p.kv_len, q_pos + 1u);
    const uint t_lo = (p.sliding_window > 0u && t_hi > p.sliding_window)
        ? (t_hi - p.sliding_window)
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
        m_val = g26_load_scalar_f32(sink_bias, qh, p.sink_dtype) * LOG2E;
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
            const uint phys_t = (p.ring_capacity > 0u) ? (t % p.ring_capacity) : t;
            const uint chunk = phys_t / p.chunk_tokens;
            const uint slot = phys_t - chunk * p.chunk_tokens;
            const uint row = head_row_base + slot;
            if (can_retire_q8 && chunk < p.num_retired_chunks) {
                const uint phys_slot = retired_chunk_ids[p.retired_offset + chunk];
                const device uchar *k_plane = retired_pool + ulong(phys_slot) * p.retired_chunk_stride_bytes;
                const device uchar *v_plane = k_plane + p.retired_plane_stride_bytes;
                if (load_col < k_cols) {
                    tg_k[tid] = load_q8_half4(k_plane, row, q8_groups, load_col);
                }
                if (load_col < v_cols) {
                    tg_v[tid] = load_q8_half4(v_plane, row, q8_groups, load_col);
                }
            } else {
                const uint active_chunk = (p.ring_capacity == 0u) ? (chunk - p.num_retired_chunks) : chunk;
                const KvTilePair pair = kv_tiles[p.tile_table_offset + active_chunk];
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
