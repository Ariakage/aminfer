// Fused Metal 4 248K-vocab radix/bucket constrained sampler (G21, ADR-0023 §4).
//
// Supports `vocab_size` up to `262,144` (covering Qwen3 151,936, DeepSeek-V4.1
// 129,280, Qwen3.8 248,077) over `f16` or `f32` device logits, single-row
// decode (`M = 1`) and batched/speculative-tree multi-row verification:
//   - Fast-path unconstrained greedy argmax (`ami_sample_greedy_*`): vectorized
//     `half4`/`float4` multi-block GPU argmax with zero reads of bitmask, bias,
//     penalty, or RNG buffers.
//   - Two-pass multi-block constrained reduction + 1024-bucket radix/Gumbel-max
//     sampler fusing C22 allowed-token bitmasks (~31 KiB for 248K vocab), sparse
//     `logit_bias`, per-slot `presence_penalty` / `frequency_penalty` /
//     `repetition_penalty` (via a 4 KiB threadgroup Bloom filter + O(1) device
//     hash tables), temperature scaling, `min_p`, `top_k`, `top_p`, parallel
//     multi-block G20 indirect boundary refinement (0 threadgroups dispatched
//     when clean), optional `logprobs` / `top_logprobs` (up to 20), and
//     deterministic slot-local D11-seeded SplitMix64 Gumbel-max sampling.
#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

constant bool SAMP_LOGITS_F16 [[function_constant(0)]];
constant bool SAMP_USE_BITMASK [[function_constant(1)]];
constant bool SAMP_HAS_OVERRIDES [[function_constant(2)]];

constant uint SAMP_MAX_BLOCKS = 32u;
constant uint SAMP_NUM_BUCKETS = 1024u;
constant uint SAMP_BIAS_CAP = 128u;
constant uint SAMP_BIAS_MASK = 127u;
constant uint SAMP_FREQ_CAP = 4096u;
constant uint SAMP_FREQ_MASK = 4095u;
constant uint SAMP_MAX_TOP_LOGPROBS = 20u;
constant uint SAMP_CUT_CAP = 2048u;
constant uint SAMP_BLOOM_WORDS = 1024u; // 4 KiB bitset (32,768 bits, <0.4% FP at 128 overrides)
constant uint SAMP_EMPTY_ID = 0xFFFFFFFFu;

constant uint SAMP_FLAG_USE_BITMASK = 1u << 0;
constant uint SAMP_FLAG_HAS_BIAS = 1u << 1;
constant uint SAMP_FLAG_HAS_PENALTIES = 1u << 2;
constant uint SAMP_FLAG_NEED_LOGPROBS = 1u << 3;

constant uint SAMP_STATUS_OK = 0u;
constant uint SAMP_STATUS_EMPTY_ALLOWED = 1u;
constant uint SAMP_STATUS_INVALID_LOGITS = 2u;

struct GpuBiasEntry {
    uint token_id;
    float bias;
};

struct GpuFreqEntry {
    uint token_id;
    uint count;
};

struct RowSampleConfig {
    float temperature;
    float inv_temp;
    float top_p;
    float min_p;
    float presence_penalty;
    float frequency_penalty;
    float repetition_penalty;
    uint top_k;
    uint seed_lo;
    uint seed_hi;
    uint flags;
    uint top_logprobs;
    uint bias_count;
    uint freq_count;
    uint bitmask_row_index;
    uint _pad0;
};

struct SampleDispatchParams {
    uint num_rows;
    uint vocab_size;
    uint num_blocks;
    uint aligned_vec4; // bit 0: all rows aligned, bit 1: row0/div4-rows aligned
    uint bitmask_stride_u32;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

struct BlockStats {
    float max_logit;
    uint argmax_id;
    float logsum_m;
    float logsum_exp;
};

struct RowState {
    float global_max;
    uint global_argmax_id;
    float global_log_z;
    uint status;
    uint cut_bucket;
    uint rem_k;
    float rem_mass;
    float total_mass;
    uint prefix_best_tok;
    uint prefix_best_gumbel;
    float cut_s_lo;
    float cut_s_hi;
    uint need_refine; // 0 = clean, 1 = parallel gather+sort (<=2048), 2 = tie fallback (>2048)
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

struct CutCandidate {
    float s;
    uint token_id;
};

struct GpuTokenLogprob {
    uint token_id;
    float logprob;
};

struct GpuSampleOutput {
    uint token_id;
    float logprob;
    uint status;
    uint top_logprobs_len;
    GpuTokenLogprob top_logprobs[20];
    uint _pad[4];
};

struct SampleIndirectArgs {
    uint threadgroups_per_grid[3];
};

static_assert(sizeof(GpuBiasEntry) == 8, "GpuBiasEntry must be 8 bytes");
static_assert(sizeof(GpuFreqEntry) == 8, "GpuFreqEntry must be 8 bytes");
static_assert(sizeof(RowSampleConfig) == 64, "RowSampleConfig must be 64 bytes");
static_assert(sizeof(SampleDispatchParams) == 32, "SampleDispatchParams must be 32 bytes");
static_assert(sizeof(BlockStats) == 16, "BlockStats must be 16 bytes");
static_assert(sizeof(RowState) == 64, "RowState must be 64 bytes");
static_assert(sizeof(CutCandidate) == 8, "CutCandidate must be 8 bytes");
static_assert(sizeof(GpuTokenLogprob) == 8, "GpuTokenLogprob must be 8 bytes");
static_assert(sizeof(GpuSampleOutput) == 192, "GpuSampleOutput must be 192 bytes");
static_assert(sizeof(SampleIndirectArgs) == 12, "SampleIndirectArgs must be 12 bytes");

// ---------------------------------------------------------------------------
// Helpers: bit/hash/radix/Gumbel utilities
// ---------------------------------------------------------------------------

static inline uint samp_hash_u32(uint token_id) {
    return token_id * 0x9E3779B1u;
}

static inline uint samp_bloom_bit1(uint token_id) {
    return (token_id * 0x9E3779B1u) >> 17; // 0..32767 (4 KiB bitset)
}

static inline uint samp_bloom_bit2(uint token_id) {
    return (token_id * 0x85EBCA77u) >> 17; // 0..32767 (2nd independent Bloom probe)
}

static inline uint float_to_sortable_u32(float v) {
    const uint bits = as_type<uint>(v);
    const uint mask = ((bits & 0x80000000u) != 0u) ? 0xFFFFFFFFu : 0x80000000u;
    return bits ^ mask;
}

static inline ulong pack_gumbel_u64(float g, uint token_id) {
    return (ulong(float_to_sortable_u32(g)) << 32) | ulong(0xFFFFFFFFu - token_id);
}

static inline uint unpack_gumbel_token(ulong packed) {
    return 0xFFFFFFFFu - uint(packed & 0xFFFFFFFFul);
}

static inline uint unpack_gumbel_key(ulong packed) {
    return uint(packed >> 32);
}

static inline float token_uniform01(ulong step_seed, uint token_id) {
    ulong z = step_seed ^ (ulong(token_id) * 0x9E3779B97F4A7C15ul);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ul;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBul;
    z = z ^ (z >> 31);
    const uint r24 = uint(z >> 40);
    return (float(r24) + 0.5f) * (1.0f / 16777216.0f);
}

static inline float gumbel_noise(ulong step_seed, uint token_id) {
    const float u = token_uniform01(step_seed, token_id);
    return -log(-log(u));
}

static inline uint bucket_of_s(float s) {
    const float d = max(0.0f, -s);
    if (d < 8.0f) {
        return min(uint(d * 96.0f), 767u);
    }
    if (d < 24.0f) {
        return 768u + min(uint((d - 8.0f) * 12.0f), 191u);
    }
    if (d < 80.0f) {
        return 960u + min(uint((d - 24.0f) * (63.0f / 56.0f)), 62u);
    }
    return 1023u;
}

static inline float bucket_s_hi(uint b) {
    if (b < 768u) {
        return -float(b) * (1.0f / 96.0f);
    }
    if (b < 960u) {
        return -8.0f - float(b - 768u) * (1.0f / 12.0f);
    }
    if (b < 1023u) {
        return -24.0f - float(b - 960u) * (56.0f / 63.0f);
    }
    return -80.0f;
}

static inline float bucket_s_lo(uint b) {
    if (b < 768u) {
        return -float(b + 1u) * (1.0f / 96.0f);
    }
    if (b < 960u) {
        return -8.0f - float(b - 767u) * (1.0f / 12.0f);
    }
    if (b < 1023u) {
        return -24.0f - float(b - 959u) * (56.0f / 63.0f);
    }
    return -1e30f;
}

static inline float load_raw_logit(const device uchar *logits_raw, ulong row_off, uint idx) {
    if (SAMP_LOGITS_F16) {
        return float(((const device half *)logits_raw)[row_off + idx]);
    } else {
        return ((const device float *)logits_raw)[row_off + idx];
    }
}

static inline float4 load_raw_logit4(const device uchar *logits_raw, ulong row_off, uint vi) {
    if (SAMP_LOGITS_F16) {
        const device half4 *v = (const device half4 *)(((const device half *)logits_raw) + row_off);
        return float4(v[vi]);
    } else {
        const device float4 *v = (const device float4 *)(((const device float *)logits_raw) + row_off);
        return v[vi];
    }
}

static inline float lookup_bias(const device GpuBiasEntry *table, uint token_id, thread bool &found) {
    uint slot = samp_hash_u32(token_id) & SAMP_BIAS_MASK;
    for (uint probe = 0u; probe < SAMP_BIAS_CAP; probe++) {
        const GpuBiasEntry e = table[slot];
        if (e.token_id == token_id) {
            found = true;
            return e.bias;
        }
        if (e.token_id == SAMP_EMPTY_ID) {
            break;
        }
        slot = (slot + 1u) & SAMP_BIAS_MASK;
    }
    found = false;
    return 0.0f;
}

static inline uint lookup_freq(const device GpuFreqEntry *table, uint token_id) {
    uint slot = samp_hash_u32(token_id) & SAMP_FREQ_MASK;
    for (uint probe = 0u; probe < SAMP_FREQ_CAP; probe++) {
        const GpuFreqEntry e = table[slot];
        if (e.token_id == token_id) {
            return e.count;
        }
        if (e.token_id == SAMP_EMPTY_ID) {
            break;
        }
        slot = (slot + 1u) & SAMP_FREQ_MASK;
    }
    return 0u;
}

static inline void build_override_bloom(threadgroup atomic_uint *s_bloom,
                                        const device uint *bias_keys,
                                        uint bias_count,
                                        const device uint *freq_keys,
                                        uint freq_count,
                                        uint tid) {
    for (uint w = tid; w < SAMP_BLOOM_WORDS; w += 256u) {
        atomic_store_explicit(&s_bloom[w], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < bias_count; i += 256u) {
        const uint k = bias_keys[i];
        const uint h1 = samp_bloom_bit1(k);
        const uint h2 = samp_bloom_bit2(k);
        atomic_fetch_or_explicit(&s_bloom[h1 >> 5], 1u << (h1 & 31u), memory_order_relaxed);
        atomic_fetch_or_explicit(&s_bloom[h2 >> 5], 1u << (h2 & 31u), memory_order_relaxed);
    }
    for (uint i = tid; i < freq_count; i += 256u) {
        const uint k = freq_keys[i];
        const uint h1 = samp_bloom_bit1(k);
        const uint h2 = samp_bloom_bit2(k);
        atomic_fetch_or_explicit(&s_bloom[h1 >> 5], 1u << (h1 & 31u), memory_order_relaxed);
        atomic_fetch_or_explicit(&s_bloom[h2 >> 5], 1u << (h2 & 31u), memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

static inline float apply_overrides_one(float raw_v,
                                        uint idx,
                                        const device GpuBiasEntry *bias_table,
                                        const device GpuFreqEntry *freq_table,
                                        threadgroup const atomic_uint *s_bloom,
                                        RowSampleConfig cfg,
                                        bool use_bloom,
                                        thread bool &saw_invalid) {
    if (isnan(raw_v) || (isinf(raw_v) && raw_v > 0.0f)) {
        saw_invalid = true;
        return -INFINITY;
    }
    float v = raw_v;
    if (SAMP_HAS_OVERRIDES && isfinite(v)) {
        const bool do_bias = (cfg.flags & SAMP_FLAG_HAS_BIAS) != 0u && cfg.bias_count > 0u;
        const bool do_pen = (cfg.flags & SAMP_FLAG_HAS_PENALTIES) != 0u && cfg.freq_count > 0u;
        if (do_bias || do_pen) {
            bool maybe_hit = true;
            if (use_bloom) {
                const uint h1 = samp_bloom_bit1(idx);
                const uint bw1 = atomic_load_explicit(&s_bloom[h1 >> 5], memory_order_relaxed);
                maybe_hit = (((bw1 >> (h1 & 31u)) & 1u) != 0u);
                if (maybe_hit) {
                    const uint h2 = samp_bloom_bit2(idx);
                    const uint bw2 = atomic_load_explicit(&s_bloom[h2 >> 5], memory_order_relaxed);
                    maybe_hit = (((bw2 >> (h2 & 31u)) & 1u) != 0u);
                }
            }
            if (maybe_hit) {
                if (do_bias) {
                    bool found = false;
                    const float b = lookup_bias(bias_table, idx, found);
                    if (found) {
                        if (isnan(b) || (isinf(b) && b > 0.0f)) {
                            saw_invalid = true;
                            return -INFINITY;
                        }
                        if (isinf(b) && b < 0.0f) {
                            return -INFINITY;
                        }
                        v += b;
                    }
                }
                if (do_pen && isfinite(v)) {
                    const uint cnt = lookup_freq(freq_table, idx);
                    if (cnt > 0u) {
                        if (cfg.repetition_penalty != 1.0f) {
                            v = (v > 0.0f) ? (v / cfg.repetition_penalty) : (v * cfg.repetition_penalty);
                        }
                        v -= (cfg.frequency_penalty * float(cnt) + cfg.presence_penalty);
                    }
                }
            }
        }
    }
    return v;
}

static inline float eval_constrained_logit(const device uchar *logits_raw,
                                           ulong row_off,
                                           uint idx,
                                           const device uint *bitmask_row,
                                           const device GpuBiasEntry *bias_table,
                                           const device GpuFreqEntry *freq_table,
                                           threadgroup const atomic_uint *s_bloom,
                                           RowSampleConfig cfg,
                                           bool use_bloom,
                                           thread bool &saw_invalid) {
    const float raw_v = load_raw_logit(logits_raw, row_off, idx);
    if (isnan(raw_v) || (isinf(raw_v) && raw_v > 0.0f)) {
        saw_invalid = true;
        return -INFINITY;
    }
    if (SAMP_USE_BITMASK && (cfg.flags & SAMP_FLAG_USE_BITMASK) != 0u) {
        const uint word = bitmask_row[idx >> 5];
        if (((word >> (idx & 31u)) & 1u) == 0u) {
            return -INFINITY;
        }
    }
    return apply_overrides_one(raw_v, idx, bias_table, freq_table, s_bloom, cfg, use_bloom, saw_invalid);
}

static inline void block_range(uint blk, uint num_blocks, uint vocab_size, thread uint &start_id, thread uint &end_id) {
    const uint raw_chunk = (vocab_size + num_blocks - 1u) / num_blocks;
    const uint chunk = (raw_chunk + 31u) & ~31u;
    const uint s = blk * chunk;
    start_id = min(s, vocab_size);
    end_id = min(s + chunk, vocab_size);
}

static inline bool is_row_vec4(SampleDispatchParams p, ulong row_off) {
    return ((p.aligned_vec4 & 1u) != 0u) || (((p.aligned_vec4 & 2u) != 0u) && ((row_off & 3ul) == 0ul));
}

// ===========================================================================
// Mode 0: Unconstrained Greedy Fast-Path Bypass (Zero mask/bias/penalty reads)
// ===========================================================================

kernel void ami_sample_greedy_pass1(const device uchar *logits_raw [[buffer(0)]],
                                    device BlockStats *partials [[buffer(1)]],
                                    constant SampleDispatchParams &p [[buffer(2)]],
                                    uint2 tg_pos [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]],
                                    uint sg_id [[simdgroup_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]]) {
    const uint blk = tg_pos.x;
    const uint row = tg_pos.y;
    if (blk >= p.num_blocks || row >= p.num_rows) {
        return;
    }

    uint start_id = 0u;
    uint end_id = 0u;
    block_range(blk, p.num_blocks, p.vocab_size, start_id, end_id);

    float best_v = -INFINITY;
    uint best_i = SAMP_EMPTY_ID;
    uint invalid_flag = 0u;

    const ulong row_off = ulong(row) * p.vocab_size;
    const bool use_vec4 = is_row_vec4(p, row_off);

    if (use_vec4) {
        const uint vec_start = start_id >> 2;
        const uint vec_end = end_id >> 2;
        for (uint vi = vec_start + tid; vi < vec_end; vi += 256u) {
            const float4 fv = load_raw_logit4(logits_raw, row_off, vi);
            const uint base = vi << 2;
            if (isnan(fv.x) || (isinf(fv.x) && fv.x > 0.0f) ||
                isnan(fv.y) || (isinf(fv.y) && fv.y > 0.0f) ||
                isnan(fv.z) || (isinf(fv.z) && fv.z > 0.0f) ||
                isnan(fv.w) || (isinf(fv.w) && fv.w > 0.0f)) {
                invalid_flag = 1u;
            }
            if (fv.x > best_v || (fv.x == best_v && (base + 0u) < best_i)) { best_v = fv.x; best_i = base + 0u; }
            if (fv.y > best_v || (fv.y == best_v && (base + 1u) < best_i)) { best_v = fv.y; best_i = base + 1u; }
            if (fv.z > best_v || (fv.z == best_v && (base + 2u) < best_i)) { best_v = fv.z; best_i = base + 2u; }
            if (fv.w > best_v || (fv.w == best_v && (base + 3u) < best_i)) { best_v = fv.w; best_i = base + 3u; }
        }
        const uint tail = vec_end << 2;
        for (uint i = tail + tid; i < end_id; i += 256u) {
            const float v = load_raw_logit(logits_raw, row_off, i);
            if (isnan(v) || (isinf(v) && v > 0.0f)) { invalid_flag = 1u; }
            if (v > best_v || (v == best_v && i < best_i)) { best_v = v; best_i = i; }
        }
    } else {
        for (uint i = start_id + tid; i < end_id; i += 256u) {
            const float v = load_raw_logit(logits_raw, row_off, i);
            if (isnan(v) || (isinf(v) && v > 0.0f)) { invalid_flag = 1u; }
            if (v > best_v || (v == best_v && i < best_i)) { best_v = v; best_i = i; }
        }
    }

    const uint sg_inv = simd_max(invalid_flag);
    for (ushort off = 16u; off > 0u; off >>= 1u) {
        const float ov = simd_shuffle_down(best_v, off);
        const uint oi = simd_shuffle_down(best_i, off);
        if (ov > best_v || (ov == best_v && oi < best_i)) {
            best_v = ov;
            best_i = oi;
        }
    }

    threadgroup float s_v[8];
    threadgroup uint s_i[8];
    threadgroup uint s_inv[8];
    if (lane == 0u) {
        s_v[sg_id] = best_v;
        s_i[sg_id] = best_i;
        s_inv[sg_id] = sg_inv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_id == 0u) {
        float fv = (lane < 8u) ? s_v[lane] : -INFINITY;
        uint fi = (lane < 8u) ? s_i[lane] : SAMP_EMPTY_ID;
        const uint any_inv = simd_max((lane < 8u) ? s_inv[lane] : 0u);
        for (ushort off = 4u; off > 0u; off >>= 1u) {
            const float ov = simd_shuffle_down(fv, off);
            const uint oi = simd_shuffle_down(fi, off);
            if (ov > fv || (ov == fv && oi < fi)) {
                fv = ov;
                fi = oi;
            }
        }
        if (lane == 0u) {
            partials[row * SAMP_MAX_BLOCKS + blk] = {
                fv,
                fi,
                0.0f,
                (any_inv != 0u) ? -1.0f : 0.0f,
            };
        }
    }
}

kernel void ami_sample_greedy_pass2(const device BlockStats *partials [[buffer(0)]],
                                    device GpuSampleOutput *outputs [[buffer(1)]],
                                    constant SampleDispatchParams &p [[buffer(2)]],
                                    uint row [[threadgroup_position_in_grid]],
                                    uint lane [[thread_index_in_simdgroup]]) {
    if (row >= p.num_rows) {
        return;
    }
    const bool valid = lane < p.num_blocks;
    const BlockStats bs = valid ? partials[row * SAMP_MAX_BLOCKS + lane] : BlockStats{ -INFINITY, SAMP_EMPTY_ID, 0.0f, 0.0f };
    float best_v = bs.max_logit;
    uint best_i = bs.argmax_id;
    const uint any_inv = simd_max((bs.logsum_exp < 0.0f) ? 1u : 0u);

    for (ushort off = 16u; off > 0u; off >>= 1u) {
        const float ov = simd_shuffle_down(best_v, off);
        const uint oi = simd_shuffle_down(best_i, off);
        if (ov > best_v || (ov == best_v && oi < best_i)) {
            best_v = ov;
            best_i = oi;
        }
    }

    if (lane == 0u) {
        outputs[row].token_id = (best_i == SAMP_EMPTY_ID) ? 0u : best_i;
        outputs[row].logprob = 0.0f;
        outputs[row].top_logprobs_len = 0u;
        if (any_inv != 0u || best_v == -INFINITY || best_i == SAMP_EMPTY_ID) {
            outputs[row].status = SAMP_STATUS_INVALID_LOGITS;
        } else {
            outputs[row].status = SAMP_STATUS_OK;
        }
    }
}

// ===========================================================================
// Mode 1 & 2 Pass 1: Fused Constrained Reduction (Vectorized half4/float4)
// ===========================================================================

static inline void pass1_consider(float v,
                                  uint idx,
                                  bool need_lp,
                                  float inv_t_lp,
                                  thread float &best_v,
                                  thread uint &best_i,
                                  thread float &local_m,
                                  thread float &local_sum) {
    if (v > best_v || (v == best_v && idx < best_i)) {
        best_v = v;
        best_i = idx;
    }
    if (need_lp && isfinite(v)) {
        if (v > local_m) {
            const float shift = (local_m - v) * inv_t_lp;
            local_sum = (local_sum > 0.0f && shift >= -80.0f) ? fma(local_sum, exp(shift), 1.0f) : 1.0f;
            local_m = v;
        } else {
            const float diff = (v - local_m) * inv_t_lp;
            if (diff >= -80.0f) {
                local_sum += exp(diff);
            }
        }
    }
}

kernel void ami_sample_constrained_pass1(const device uchar *logits_raw [[buffer(0)]],
                                         const device RowSampleConfig *row_cfgs [[buffer(1)]],
                                         const device uint *bitmasks [[buffer(2)]],
                                         const device GpuBiasEntry *bias_tables [[buffer(3)]],
                                         const device uint *bias_keys [[buffer(4)]],
                                         const device GpuFreqEntry *freq_tables [[buffer(5)]],
                                         const device uint *freq_keys [[buffer(6)]],
                                         device BlockStats *partials [[buffer(7)]],
                                         constant SampleDispatchParams &p [[buffer(8)]],
                                         uint2 tg_pos [[threadgroup_position_in_grid]],
                                         uint tid [[thread_index_in_threadgroup]],
                                         uint sg_id [[simdgroup_index_in_threadgroup]],
                                         uint lane [[thread_index_in_simdgroup]]) {
    const uint blk = tg_pos.x;
    const uint row = tg_pos.y;
    if (blk >= p.num_blocks || row >= p.num_rows) {
        return;
    }

    const RowSampleConfig cfg = row_cfgs[row];
    const ulong row_off = ulong(row) * p.vocab_size;
    const device uint *bm_row = bitmasks + ulong(cfg.bitmask_row_index) * p.bitmask_stride_u32;
    const device GpuBiasEntry *b_tbl = bias_tables + ulong(row) * SAMP_BIAS_CAP;
    const device uint *b_keys = bias_keys + ulong(row) * 64u;
    const device GpuFreqEntry *f_tbl = freq_tables + ulong(row) * SAMP_FREQ_CAP;
    const device uint *f_keys = freq_keys + ulong(row) * 2048u;

    threadgroup atomic_uint s_bloom[SAMP_BLOOM_WORDS];
    const bool use_bloom = SAMP_HAS_OVERRIDES &&
        (((cfg.flags & SAMP_FLAG_HAS_BIAS) != 0u && cfg.bias_count > 0u) ||
         ((cfg.flags & SAMP_FLAG_HAS_PENALTIES) != 0u && cfg.freq_count > 0u));
    if (use_bloom) {
        build_override_bloom(s_bloom, b_keys, cfg.bias_count, f_keys, cfg.freq_count, tid);
    }

    uint start_id = 0u;
    uint end_id = 0u;
    block_range(blk, p.num_blocks, p.vocab_size, start_id, end_id);

    const bool need_lp = ((cfg.flags & SAMP_FLAG_NEED_LOGPROBS) != 0u);
    const float inv_t_lp = (cfg.temperature > 0.0f) ? cfg.inv_temp : 1.0f;
    const bool bm_active = SAMP_USE_BITMASK && ((cfg.flags & SAMP_FLAG_USE_BITMASK) != 0u);
    const bool use_vec4 = is_row_vec4(p, row_off);

    float best_v = -INFINITY;
    uint best_i = SAMP_EMPTY_ID;
    float local_m = -INFINITY;
    float local_sum = 0.0f;
    bool saw_invalid = false;

    if (use_vec4) {
        const uint vec_start = start_id >> 2;
        const uint vec_end = end_id >> 2;
        for (uint vi = vec_start + tid; vi < vec_end; vi += 256u) {
            const float4 fv = load_raw_logit4(logits_raw, row_off, vi);
            if (isnan(fv.x) || (isinf(fv.x) && fv.x > 0.0f) ||
                isnan(fv.y) || (isinf(fv.y) && fv.y > 0.0f) ||
                isnan(fv.z) || (isinf(fv.z) && fv.z > 0.0f) ||
                isnan(fv.w) || (isinf(fv.w) && fv.w > 0.0f)) {
                saw_invalid = true;
            }
            const uint base = vi << 2;
            uint bm4 = 0xFu;
            if (bm_active) {
                bm4 = (bm_row[base >> 5] >> (base & 31u)) & 0xFu;
                if (bm4 == 0u) {
                    continue;
                }
            }
            if ((bm4 & 1u) != 0u) {
                const float v0 = apply_overrides_one(fv.x, base + 0u, b_tbl, f_tbl, s_bloom, cfg, use_bloom, saw_invalid);
                pass1_consider(v0, base + 0u, need_lp, inv_t_lp, best_v, best_i, local_m, local_sum);
            }
            if ((bm4 & 2u) != 0u) {
                const float v1 = apply_overrides_one(fv.y, base + 1u, b_tbl, f_tbl, s_bloom, cfg, use_bloom, saw_invalid);
                pass1_consider(v1, base + 1u, need_lp, inv_t_lp, best_v, best_i, local_m, local_sum);
            }
            if ((bm4 & 4u) != 0u) {
                const float v2 = apply_overrides_one(fv.z, base + 2u, b_tbl, f_tbl, s_bloom, cfg, use_bloom, saw_invalid);
                pass1_consider(v2, base + 2u, need_lp, inv_t_lp, best_v, best_i, local_m, local_sum);
            }
            if ((bm4 & 8u) != 0u) {
                const float v3 = apply_overrides_one(fv.w, base + 3u, b_tbl, f_tbl, s_bloom, cfg, use_bloom, saw_invalid);
                pass1_consider(v3, base + 3u, need_lp, inv_t_lp, best_v, best_i, local_m, local_sum);
            }
        }
        const uint tail = vec_end << 2;
        for (uint i = tail + tid; i < end_id; i += 256u) {
            const float v = eval_constrained_logit(
                logits_raw, row_off, i, bm_row, b_tbl, f_tbl, s_bloom, cfg, use_bloom, saw_invalid
            );
            pass1_consider(v, i, need_lp, inv_t_lp, best_v, best_i, local_m, local_sum);
        }
    } else {
        for (uint i = start_id + tid; i < end_id; i += 256u) {
            const float v = eval_constrained_logit(
                logits_raw, row_off, i, bm_row, b_tbl, f_tbl, s_bloom, cfg, use_bloom, saw_invalid
            );
            pass1_consider(v, i, need_lp, inv_t_lp, best_v, best_i, local_m, local_sum);
        }
    }

    const uint sg_inv = simd_max(saw_invalid ? 1u : 0u);
    for (ushort off = 16u; off > 0u; off >>= 1u) {
        const float ov = simd_shuffle_down(best_v, off);
        const uint oi = simd_shuffle_down(best_i, off);
        if (ov > best_v || (ov == best_v && oi < best_i)) {
            best_v = ov;
            best_i = oi;
        }
    }

    float sg_m = -INFINITY;
    float sg_sum = 0.0f;
    if (need_lp) {
        sg_m = simd_max(local_m);
        const float sdiff = (local_m - sg_m) * inv_t_lp;
        const float contrib = (local_sum > 0.0f && isfinite(sg_m) && sdiff >= -80.0f) ? (local_sum * exp(sdiff)) : 0.0f;
        sg_sum = simd_sum(contrib);
    }

    threadgroup float s_v[8];
    threadgroup uint s_i[8];
    threadgroup float s_m[8];
    threadgroup float s_sum[8];
    threadgroup uint s_inv[8];

    if (lane == 0u) {
        s_v[sg_id] = best_v;
        s_i[sg_id] = best_i;
        s_m[sg_id] = sg_m;
        s_sum[sg_id] = sg_sum;
        s_inv[sg_id] = sg_inv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_id == 0u) {
        float fv = (lane < 8u) ? s_v[lane] : -INFINITY;
        uint fi = (lane < 8u) ? s_i[lane] : SAMP_EMPTY_ID;
        const uint any_inv = simd_max((lane < 8u) ? s_inv[lane] : 0u);
        for (ushort off = 4u; off > 0u; off >>= 1u) {
            const float ov = simd_shuffle_down(fv, off);
            const uint oi = simd_shuffle_down(fi, off);
            if (ov > fv || (ov == fv && oi < fi)) {
                fv = ov;
                fi = oi;
            }
        }
        float bm = -INFINITY;
        float bsum = 0.0f;
        if (need_lp) {
            const float lm = (lane < 8u) ? s_m[lane] : -INFINITY;
            const float ls = (lane < 8u) ? s_sum[lane] : 0.0f;
            bm = simd_max(lm);
            const float bdiff = (lm - bm) * inv_t_lp;
            bsum = simd_sum((ls > 0.0f && isfinite(bm) && bdiff >= -80.0f) ? (ls * exp(bdiff)) : 0.0f);
        }
        if (lane == 0u) {
            partials[row * SAMP_MAX_BLOCKS + blk] = {
                fv,
                fi,
                bm,
                (any_inv != 0u) ? -1.0f : bsum,
            };
        }
    }
}

// Merges per-block Pass 1 stats into `RowState[row]`, completes greedy rows,
// zeroes this row's 1024-bucket histogram, and resets the 3 indirect grids:
//   refine_indirect[0] = gather, [1] = sort, [2] = tie fallback.
kernel void ami_sample_merge_stats(const device BlockStats *partials [[buffer(0)]],
                                   const device RowSampleConfig *row_cfgs [[buffer(1)]],
                                   device RowState *row_states [[buffer(2)]],
                                   device uint *hist_count [[buffer(3)]],
                                   device float *hist_mass [[buffer(4)]],
                                   device ulong *hist_gumbel [[buffer(5)]],
                                   device SampleIndirectArgs *refine_indirect [[buffer(6)]],
                                   device uint *cut_cand_counts [[buffer(7)]],
                                   device GpuSampleOutput *outputs [[buffer(8)]],
                                   constant SampleDispatchParams &p [[buffer(9)]],
                                   uint row [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]],
                                   uint sg_id [[simdgroup_index_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]]) {
    if (row >= p.num_rows) {
        return;
    }

    const ulong h_base = ulong(row) * SAMP_NUM_BUCKETS;
    for (uint b = tid; b < SAMP_NUM_BUCKETS; b += 256u) {
        hist_count[h_base + b] = 0u;
        hist_mass[h_base + b] = 0.0f;
        hist_gumbel[h_base + b] = 0ul;
    }
    if (tid == 0u) {
        cut_cand_counts[row] = 0u;
    }
    if (row == 0u && tid < 3u) {
        refine_indirect[tid].threadgroups_per_grid[0] = 0u;
        refine_indirect[tid].threadgroups_per_grid[1] = 1u;
        refine_indirect[tid].threadgroups_per_grid[2] = 1u;
    }

    if (sg_id == 0u) {
        const RowSampleConfig cfg = row_cfgs[row];
        const bool valid = lane < p.num_blocks;
        const BlockStats bs = valid
            ? partials[row * SAMP_MAX_BLOCKS + lane]
            : BlockStats{ -INFINITY, SAMP_EMPTY_ID, -INFINITY, 0.0f };

        float best_v = bs.max_logit;
        uint best_i = bs.argmax_id;
        const uint any_inv = simd_max((bs.logsum_exp < 0.0f) ? 1u : 0u);

        for (ushort off = 16u; off > 0u; off >>= 1u) {
            const float ov = simd_shuffle_down(best_v, off);
            const uint oi = simd_shuffle_down(best_i, off);
            if (ov > best_v || (ov == best_v && oi < best_i)) {
                best_v = ov;
                best_i = oi;
            }
        }

        const bool need_lp = ((cfg.flags & SAMP_FLAG_NEED_LOGPROBS) != 0u);
        const float inv_t_lp = (cfg.temperature > 0.0f) ? cfg.inv_temp : 1.0f;
        float gm = -INFINITY;
        float gsum = 0.0f;
        if (need_lp) {
            const float lm = (bs.logsum_exp > 0.0f) ? bs.logsum_m : -INFINITY;
            const float ls = (bs.logsum_exp > 0.0f) ? bs.logsum_exp : 0.0f;
            gm = simd_max(lm);
            const float gdiff = (lm - gm) * inv_t_lp;
            gsum = simd_sum((ls > 0.0f && isfinite(gm) && gdiff >= -80.0f) ? (ls * exp(gdiff)) : 0.0f);
        }

        if (lane == 0u) {
            uint status = SAMP_STATUS_OK;
            if (any_inv != 0u) {
                status = SAMP_STATUS_INVALID_LOGITS;
            } else if (best_v == -INFINITY || best_i == SAMP_EMPTY_ID) {
                status = SAMP_STATUS_EMPTY_ALLOWED;
            }
            const float log_z = (need_lp && gsum > 0.0f) ? log(gsum) : 0.0f;

            RowState st;
            st.global_max = best_v;
            st.global_argmax_id = (best_i == SAMP_EMPTY_ID) ? 0u : best_i;
            st.global_log_z = log_z;
            st.status = status;
            st.cut_bucket = 0u;
            st.rem_k = 0u;
            st.rem_mass = 0.0f;
            st.total_mass = 0.0f;
            st.prefix_best_tok = SAMP_EMPTY_ID;
            st.prefix_best_gumbel = 0u;
            st.cut_s_lo = 0.0f;
            st.cut_s_hi = 0.0f;
            st.need_refine = 0u;
            st._pad0 = 0u;
            st._pad1 = 0u;
            st._pad2 = 0u;
            row_states[row] = st;

            outputs[row].token_id = st.global_argmax_id;
            outputs[row].logprob = (need_lp && status == SAMP_STATUS_OK) ? (-log_z) : 0.0f;
            outputs[row].status = status;
            outputs[row].top_logprobs_len = 0u;
        }
    }
}

// ===========================================================================
// Mode 2 Pass 2: Vectorized 1024-Bucket Radix + Per-Bucket Gumbel-Max
// ===========================================================================

static inline void pass2_consider(float v,
                                  uint idx,
                                  uint start_id,
                                  float gmax,
                                  uint argmax_id,
                                  float inv_t,
                                  float s_cutoff,
                                  ulong step_seed,
                                  threadgroup atomic_uint *s_b_count,
                                  threadgroup atomic_uint *s_b_mass,
                                  threadgroup atomic_uint *s_b_best) {
    if (!isfinite(v)) {
        return;
    }
    const float s = (v - gmax) * inv_t;
    if (s < s_cutoff && idx != argmax_id) {
        return;
    }
    const uint b = bucket_of_s(s);
    const float s_hi = bucket_s_hi(b);
    const float rel_s = min(s - s_hi, 0.0f);
    const float rel_w = (s < -80.0f) ? 0.0f : exp(rel_s);
    const uint q_mass = uint(rel_w * 262144.0f + 0.5f);

    const float g_rel = rel_s + gumbel_noise(step_seed, idx);
    const float norm_g = clamp((g_rel + 6.0f) * (1.0f / 26.0f), 0.0f, 1.0f);
    const uint q19 = 1u + uint(norm_g * 524286.0f);
    const uint local_idx = idx - start_id;
    const uint packed = (q19 << 13) | (8191u - (local_idx & 8191u));

    atomic_fetch_add_explicit(&s_b_count[b], 1u, memory_order_relaxed);
    atomic_fetch_add_explicit(&s_b_mass[b], q_mass, memory_order_relaxed);
    atomic_fetch_max_explicit(&s_b_best[b], packed, memory_order_relaxed);
}

kernel void ami_sample_radix_pass2(const device uchar *logits_raw [[buffer(0)]],
                                   const device RowSampleConfig *row_cfgs [[buffer(1)]],
                                   const device RowState *row_states [[buffer(2)]],
                                   const device uint *bitmasks [[buffer(3)]],
                                   const device GpuBiasEntry *bias_tables [[buffer(4)]],
                                   const device uint *bias_keys [[buffer(5)]],
                                   const device GpuFreqEntry *freq_tables [[buffer(6)]],
                                   const device uint *freq_keys [[buffer(7)]],
                                   device atomic_uint *d_hist_count [[buffer(8)]],
                                   device atomic_float *d_hist_mass [[buffer(9)]],
                                   device atomic_ulong *d_hist_gumbel [[buffer(10)]],
                                   constant SampleDispatchParams &p [[buffer(11)]],
                                   uint2 tg_pos [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]]) {
    const uint blk = tg_pos.x;
    const uint row = tg_pos.y;
    if (blk >= p.num_blocks || row >= p.num_rows) {
        return;
    }

    const RowSampleConfig cfg = row_cfgs[row];
    const RowState st = row_states[row];
    if (st.status != SAMP_STATUS_OK || cfg.temperature <= 0.0f) {
        return;
    }

    const ulong row_off = ulong(row) * p.vocab_size;
    const device uint *bm_row = bitmasks + ulong(cfg.bitmask_row_index) * p.bitmask_stride_u32;
    const device GpuBiasEntry *b_tbl = bias_tables + ulong(row) * SAMP_BIAS_CAP;
    const device uint *b_keys = bias_keys + ulong(row) * 64u;
    const device GpuFreqEntry *f_tbl = freq_tables + ulong(row) * SAMP_FREQ_CAP;
    const device uint *f_keys = freq_keys + ulong(row) * 2048u;

    // 16 KiB total threadgroup SRAM.
    threadgroup atomic_uint s_b_count[SAMP_NUM_BUCKETS]; // 4 KiB
    threadgroup atomic_uint s_b_mass[SAMP_NUM_BUCKETS];  // 4 KiB
    threadgroup atomic_uint s_b_best[SAMP_NUM_BUCKETS];  // 4 KiB
    threadgroup atomic_uint s_bloom[SAMP_BLOOM_WORDS];   // 4 KiB

    for (uint b = tid; b < SAMP_NUM_BUCKETS; b += 256u) {
        atomic_store_explicit(&s_b_count[b], 0u, memory_order_relaxed);
        atomic_store_explicit(&s_b_mass[b], 0u, memory_order_relaxed);
        atomic_store_explicit(&s_b_best[b], 0u, memory_order_relaxed);
    }

    const bool use_bloom = SAMP_HAS_OVERRIDES &&
        (((cfg.flags & SAMP_FLAG_HAS_BIAS) != 0u && cfg.bias_count > 0u) ||
         ((cfg.flags & SAMP_FLAG_HAS_PENALTIES) != 0u && cfg.freq_count > 0u));
    if (use_bloom) {
        build_override_bloom(s_bloom, b_keys, cfg.bias_count, f_keys, cfg.freq_count, tid);
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint start_id = 0u;
    uint end_id = 0u;
    block_range(blk, p.num_blocks, p.vocab_size, start_id, end_id);

    const float gmax = st.global_max;
    const uint argmax_id = st.global_argmax_id;
    const float inv_t = cfg.inv_temp;
    const float s_cutoff = (cfg.min_p > 0.0f) ? max(-80.0f, log(cfg.min_p)) : -80.0f;
    const ulong step_seed = (ulong(cfg.seed_hi) << 32) | ulong(cfg.seed_lo);
    const bool bm_active = SAMP_USE_BITMASK && ((cfg.flags & SAMP_FLAG_USE_BITMASK) != 0u);
    const bool has_ov = SAMP_HAS_OVERRIDES && use_bloom;
    const bool use_vec4 = is_row_vec4(p, row_off);
    bool dummy_inv = false;

    if (use_vec4) {
        const uint vec_start = start_id >> 2;
        const uint vec_end = end_id >> 2;
        for (uint vi = vec_start + tid; vi < vec_end; vi += 256u) {
            const float4 fv = load_raw_logit4(logits_raw, row_off, vi);
            const uint base = vi << 2;
            uint bm4 = 0xFu;
            if (bm_active) {
                bm4 = (bm_row[base >> 5] >> (base & 31u)) & 0xFu;
                if (bm4 == 0u) {
                    continue;
                }
            }
            // Fast 4-element pre-check when no per-token overrides apply:
            if (!has_ov) {
                const float4 sv = (fv - gmax) * inv_t;
                if (sv.x < s_cutoff && sv.y < s_cutoff && sv.z < s_cutoff && sv.w < s_cutoff) {
                    if (argmax_id < base || argmax_id >= base + 4u) {
                        continue;
                    }
                }
                if ((bm4 & 1u) != 0u) {
                    pass2_consider(fv.x, base + 0u, start_id, gmax, argmax_id, inv_t, s_cutoff, step_seed, s_b_count, s_b_mass, s_b_best);
                }
                if ((bm4 & 2u) != 0u) {
                    pass2_consider(fv.y, base + 1u, start_id, gmax, argmax_id, inv_t, s_cutoff, step_seed, s_b_count, s_b_mass, s_b_best);
                }
                if ((bm4 & 4u) != 0u) {
                    pass2_consider(fv.z, base + 2u, start_id, gmax, argmax_id, inv_t, s_cutoff, step_seed, s_b_count, s_b_mass, s_b_best);
                }
                if ((bm4 & 8u) != 0u) {
                    pass2_consider(fv.w, base + 3u, start_id, gmax, argmax_id, inv_t, s_cutoff, step_seed, s_b_count, s_b_mass, s_b_best);
                }
            } else {
                if ((bm4 & 1u) != 0u) {
                    const float v0 = apply_overrides_one(fv.x, base + 0u, b_tbl, f_tbl, s_bloom, cfg, true, dummy_inv);
                    pass2_consider(v0, base + 0u, start_id, gmax, argmax_id, inv_t, s_cutoff, step_seed, s_b_count, s_b_mass, s_b_best);
                }
                if ((bm4 & 2u) != 0u) {
                    const float v1 = apply_overrides_one(fv.y, base + 1u, b_tbl, f_tbl, s_bloom, cfg, true, dummy_inv);
                    pass2_consider(v1, base + 1u, start_id, gmax, argmax_id, inv_t, s_cutoff, step_seed, s_b_count, s_b_mass, s_b_best);
                }
                if ((bm4 & 4u) != 0u) {
                    const float v2 = apply_overrides_one(fv.z, base + 2u, b_tbl, f_tbl, s_bloom, cfg, true, dummy_inv);
                    pass2_consider(v2, base + 2u, start_id, gmax, argmax_id, inv_t, s_cutoff, step_seed, s_b_count, s_b_mass, s_b_best);
                }
                if ((bm4 & 8u) != 0u) {
                    const float v3 = apply_overrides_one(fv.w, base + 3u, b_tbl, f_tbl, s_bloom, cfg, true, dummy_inv);
                    pass2_consider(v3, base + 3u, start_id, gmax, argmax_id, inv_t, s_cutoff, step_seed, s_b_count, s_b_mass, s_b_best);
                }
            }
        }
        const uint tail = vec_end << 2;
        for (uint i = tail + tid; i < end_id; i += 256u) {
            const float v = eval_constrained_logit(
                logits_raw, row_off, i, bm_row, b_tbl, f_tbl, s_bloom, cfg, use_bloom, dummy_inv
            );
            pass2_consider(v, i, start_id, gmax, argmax_id, inv_t, s_cutoff, step_seed, s_b_count, s_b_mass, s_b_best);
        }
    } else {
        for (uint i = start_id + tid; i < end_id; i += 256u) {
            const float v = eval_constrained_logit(
                logits_raw, row_off, i, bm_row, b_tbl, f_tbl, s_bloom, cfg, use_bloom, dummy_inv
            );
            pass2_consider(v, i, start_id, gmax, argmax_id, inv_t, s_cutoff, step_seed, s_b_count, s_b_mass, s_b_best);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Flush ONLY non-empty buckets into the row's 16 KiB global histogram.
    const ulong h_base = ulong(row) * SAMP_NUM_BUCKETS;
    for (uint b = tid; b < SAMP_NUM_BUCKETS; b += 256u) {
        const uint cnt = atomic_load_explicit(&s_b_count[b], memory_order_relaxed);
        if (cnt == 0u) {
            continue;
        }
        const uint q_sum = atomic_load_explicit(&s_b_mass[b], memory_order_relaxed);
        const float block_mass = (float(q_sum) * (1.0f / 262144.0f)) * exp(bucket_s_hi(b));
        const uint best_p = atomic_load_explicit(&s_b_best[b], memory_order_relaxed);
        const uint win_local = 8191u - (best_p & 8191u);
        const uint win_tok = start_id + win_local;

        const float v_win = eval_constrained_logit(
            logits_raw, row_off, win_tok, bm_row, b_tbl, f_tbl, s_bloom, cfg, use_bloom, dummy_inv
        );
        const float s_win = (v_win - gmax) * inv_t;
        const float g_exact = s_win + gumbel_noise(step_seed, win_tok);

        atomic_fetch_add_explicit(&d_hist_count[h_base + b], cnt, memory_order_relaxed);
        atomic_fetch_add_explicit(&d_hist_mass[h_base + b], block_mass, memory_order_relaxed);
        atomic_max_explicit(&d_hist_gumbel[h_base + b], pack_gumbel_u64(g_exact, win_tok), memory_order_relaxed);
    }
}

// ===========================================================================
// Mode 2 Select: Parallel 256-Thread Prefix Scan Over 1024 Buckets
// Triggers parallel 32-block indirect refinement ONLY when a bucket straddles.
// ===========================================================================

kernel void ami_sample_radix_select(const device uchar *logits_raw [[buffer(0)]],
                                    const device RowSampleConfig *row_cfgs [[buffer(1)]],
                                    device RowState *row_states [[buffer(2)]],
                                    const device uint *bitmasks [[buffer(3)]],
                                    const device GpuBiasEntry *bias_tables [[buffer(4)]],
                                    const device GpuFreqEntry *freq_tables [[buffer(5)]],
                                    const device uint *hist_count [[buffer(6)]],
                                    const device float *hist_mass [[buffer(7)]],
                                    const device ulong *hist_gumbel [[buffer(8)]],
                                    device atomic_uint *refine_indirect_x [[buffer(9)]], // stride 3 u32s/struct -> idx 0, 3, 6
                                    device GpuSampleOutput *outputs [[buffer(10)]],
                                    constant SampleDispatchParams &p [[buffer(11)]],
                                    uint row [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]],
                                    uint sg_id [[simdgroup_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]]) {
    if (row >= p.num_rows) {
        return;
    }

    const RowSampleConfig cfg = row_cfgs[row];
    const RowState st = row_states[row];
    if (st.status != SAMP_STATUS_OK || cfg.temperature <= 0.0f) {
        return;
    }

    threadgroup uint s_cnt[SAMP_NUM_BUCKETS];
    threadgroup float s_mass[SAMP_NUM_BUCKETS];
    threadgroup ulong s_gumbel[SAMP_NUM_BUCKETS];
    threadgroup uint s_sg_cnt[9];
    threadgroup float s_sg_mass[9];
    threadgroup uint s_cuts[3]; // [0]=topk, [1]=topp, [2]=min

    // Each thread loads 4 consecutive buckets (`b0 = tid*4`).
    const ulong h_base = ulong(row) * SAMP_NUM_BUCKETS;
    const uint b0 = tid << 2;
    const uint c0 = hist_count[h_base + b0 + 0u];
    const uint c1 = hist_count[h_base + b0 + 1u];
    const uint c2 = hist_count[h_base + b0 + 2u];
    const uint c3 = hist_count[h_base + b0 + 3u];
    const float m0 = hist_mass[h_base + b0 + 0u];
    const float m1 = hist_mass[h_base + b0 + 1u];
    const float m2 = hist_mass[h_base + b0 + 2u];
    const float m3 = hist_mass[h_base + b0 + 3u];
    s_cnt[b0 + 0u] = c0; s_cnt[b0 + 1u] = c1; s_cnt[b0 + 2u] = c2; s_cnt[b0 + 3u] = c3;
    s_mass[b0 + 0u] = m0; s_mass[b0 + 1u] = m1; s_mass[b0 + 2u] = m2; s_mass[b0 + 3u] = m3;
    s_gumbel[b0 + 0u] = hist_gumbel[h_base + b0 + 0u];
    s_gumbel[b0 + 1u] = hist_gumbel[h_base + b0 + 1u];
    s_gumbel[b0 + 2u] = hist_gumbel[h_base + b0 + 2u];
    s_gumbel[b0 + 3u] = hist_gumbel[h_base + b0 + 3u];

    const uint my_c = c0 + c1 + c2 + c3;
    const float my_m = m0 + m1 + m2 + m3;

    // 2-level inclusive prefix sum of `my_c` and `my_m` across 256 threads (~0.5 us).
    const uint sg_c_incl = simd_prefix_inclusive_sum(my_c);
    const float sg_m_incl = simd_prefix_inclusive_sum(my_m);
    if (lane == 31u) {
        s_sg_cnt[sg_id] = sg_c_incl;
        s_sg_mass[sg_id] = sg_m_incl;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_id == 0u) {
        const uint vc = (lane < 8u) ? s_sg_cnt[lane] : 0u;
        const float vm = (lane < 8u) ? s_sg_mass[lane] : 0.0f;
        const uint ex_c = simd_prefix_exclusive_sum(vc);
        const float ex_m = simd_prefix_exclusive_sum(vm);
        if (lane < 8u) {
            s_sg_cnt[lane] = ex_c;
            s_sg_mass[lane] = ex_m;
        }
        if (lane == 7u) {
            s_sg_cnt[8] = ex_c + vc;   // total_cnt
            s_sg_mass[8] = ex_m + vm;  // total_mass
            s_cuts[0] = SAMP_NUM_BUCKETS - 1u;
            s_cuts[1] = SAMP_NUM_BUCKETS - 1u;
            s_cuts[2] = SAMP_NUM_BUCKETS - 1u;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint total_cnt = s_sg_cnt[8];
    const float total_mass = s_sg_mass[8];
    if (total_cnt == 0u || !(total_mass > 0.0f)) {
        if (tid == 0u) {
            outputs[row].token_id = st.global_argmax_id;
            outputs[row].logprob = ((cfg.flags & SAMP_FLAG_NEED_LOGPROBS) != 0u) ? (-st.global_log_z) : 0.0f;
            outputs[row].status = SAMP_STATUS_OK;
        }
        return;
    }

    const uint incl_c4 = s_sg_cnt[sg_id] + sg_c_incl;
    const uint excl_c4 = incl_c4 - my_c;
    const float incl_m4 = s_sg_mass[sg_id] + sg_m_incl;
    const float excl_m4 = incl_m4 - my_m;

    const bool has_topk = (cfg.top_k > 0u && cfg.top_k < total_cnt);
    const bool has_topp = (cfg.top_p < 1.0f && total_cnt > 1u);
    const float target_mass = cfg.top_p * total_mass;

    if (has_topk && excl_c4 < cfg.top_k && incl_c4 >= cfg.top_k) {
        uint cur = excl_c4;
        if (cur + c0 >= cfg.top_k) s_cuts[0] = b0 + 0u;
        else if (cur + c0 + c1 >= cfg.top_k) s_cuts[0] = b0 + 1u;
        else if (cur + c0 + c1 + c2 >= cfg.top_k) s_cuts[0] = b0 + 2u;
        else s_cuts[0] = b0 + 3u;
    }
    if (has_topp && excl_m4 < target_mass && incl_m4 >= target_mass) {
        float cur = excl_m4;
        if (cur + m0 >= target_mass) s_cuts[1] = b0 + 0u;
        else if (cur + m0 + m1 >= target_mass) s_cuts[1] = b0 + 1u;
        else if (cur + m0 + m1 + m2 >= target_mass) s_cuts[1] = b0 + 2u;
        else s_cuts[1] = b0 + 3u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        s_cuts[2] = min(s_cuts[0], s_cuts[1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint b_cut = s_cuts[2];
    uint p_c = 0u;
    float p_m = 0.0f;
    ulong p_best = 0ul;
    for (uint k = 0u; k < 4u; k++) {
        const uint b = b0 + k;
        if (b < b_cut) {
            p_c += s_cnt[b];
            p_m += s_mass[b];
            if (s_cnt[b] > 0u && s_gumbel[b] > p_best) {
                p_best = s_gumbel[b];
            }
        }
    }
    const uint sg_pc = simd_sum(p_c);
    const float sg_pm = simd_sum(p_m);
    uint pk_hi = uint(p_best >> 32);
    uint pk_lo = uint(p_best & 0xFFFFFFFFul);
    for (ushort off = 16u; off > 0u; off >>= 1u) {
        const uint oh = simd_shuffle_down(pk_hi, off);
        const uint ol = simd_shuffle_down(pk_lo, off);
        if (oh > pk_hi || (oh == pk_hi && ol > pk_lo)) {
            pk_hi = oh;
            pk_lo = ol;
        }
    }
    threadgroup ulong s_sg_best[8];
    if (lane == 0u) {
        s_sg_cnt[sg_id] = sg_pc;
        s_sg_mass[sg_id] = sg_pm;
        s_sg_best[sg_id] = (ulong(pk_hi) << 32) | ulong(pk_lo);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        uint prefix_cnt = 0u;
        float prefix_mass = 0.0f;
        ulong prefix_best = 0ul;
        for (uint s = 0u; s < 8u; s++) {
            prefix_cnt += s_sg_cnt[s];
            prefix_mass += s_sg_mass[s];
            if (s_sg_best[s] > prefix_best) {
                prefix_best = s_sg_best[s];
            }
        }

        const uint cut_cnt = s_cnt[b_cut];
        const float cut_mass = s_mass[b_cut];
        const bool splits_topk = has_topk && (prefix_cnt + cut_cnt > cfg.top_k);
        const bool splits_topp = has_topp && (prefix_mass + cut_mass > target_mass) &&
            ((prefix_mass + cut_mass - target_mass) > 0.005f * total_mass);
        const bool need_refine = (cut_cnt > 1u) && (splits_topk || splits_topp);

        if (need_refine) {
            RowState next_st = st;
            next_st.cut_bucket = b_cut;
            next_st.rem_k = has_topk ? (cfg.top_k - prefix_cnt) : 0xFFFFFFFFu;
            next_st.rem_mass = has_topp ? max(target_mass - prefix_mass, 0.0f) : INFINITY;
            next_st.total_mass = total_mass;
            next_st.prefix_best_tok = (prefix_best != 0ul) ? unpack_gumbel_token(prefix_best) : SAMP_EMPTY_ID;
            next_st.prefix_best_gumbel = (prefix_best != 0ul) ? unpack_gumbel_key(prefix_best) : 0u;
            next_st.cut_s_lo = bucket_s_lo(b_cut);
            next_st.cut_s_hi = bucket_s_hi(b_cut);
            if (cut_cnt <= SAMP_CUT_CAP) {
                next_st.need_refine = 1u;
                row_states[row] = next_st;
                // Indirect[0] = gather (num_blocks, num_rows, 1); Indirect[1] = sort (num_rows, 1, 1).
                atomic_store_explicit(&refine_indirect_x[0], p.num_blocks, memory_order_relaxed);
                atomic_store_explicit(&refine_indirect_x[1], p.num_rows, memory_order_relaxed);
                atomic_store_explicit(&refine_indirect_x[3], p.num_rows, memory_order_relaxed);
            } else {
                next_st.need_refine = 2u;
                row_states[row] = next_st;
                // Indirect[2] = massive-tie fallback (num_rows, 1, 1).
                atomic_store_explicit(&refine_indirect_x[6], p.num_rows, memory_order_relaxed);
            }
            return;
        }

        ulong best_all = prefix_best;
        if (cut_cnt > 0u && s_gumbel[b_cut] > best_all) {
            best_all = s_gumbel[b_cut];
        }
        const uint chosen = (best_all != 0ul) ? unpack_gumbel_token(best_all) : st.global_argmax_id;

        float out_lp = 0.0f;
        if ((cfg.flags & SAMP_FLAG_NEED_LOGPROBS) != 0u) {
            const ulong row_off = ulong(row) * p.vocab_size;
            const device uint *bm_row = bitmasks + ulong(cfg.bitmask_row_index) * p.bitmask_stride_u32;
            const device GpuBiasEntry *b_tbl = bias_tables + ulong(row) * SAMP_BIAS_CAP;
            const device GpuFreqEntry *f_tbl = freq_tables + ulong(row) * SAMP_FREQ_CAP;
            threadgroup atomic_uint dummy_bloom[1];
            bool dummy_inv = false;
            const float v_chosen = eval_constrained_logit(
                logits_raw, row_off, chosen, bm_row, b_tbl, f_tbl, dummy_bloom, cfg, false, dummy_inv
            );
            out_lp = isfinite(v_chosen) ? ((v_chosen - st.global_max) * cfg.inv_temp - st.global_log_z) : -INFINITY;
        }

        outputs[row].token_id = chosen;
        outputs[row].logprob = out_lp;
        outputs[row].status = SAMP_STATUS_OK;
    }
}

// ===========================================================================
// Parallel 32-Block Indirect Boundary Refinement (`need_refine == 1`, <=2048)
// ===========================================================================

static inline void refine_gather_one(float v,
                                     uint idx,
                                     float gmax,
                                     uint argmax_id,
                                     float inv_t,
                                     float s_cutoff,
                                     uint cut_b,
                                     device atomic_uint *row_cand_cnt,
                                     device CutCandidate *row_cands) {
    if (!isfinite(v)) {
        return;
    }
    const float s = (v - gmax) * inv_t;
    if (s < s_cutoff && idx != argmax_id) {
        return;
    }
    if (bucket_of_s(s) == cut_b) {
        const uint pos = atomic_fetch_add_explicit(row_cand_cnt, 1u, memory_order_relaxed);
        if (pos < SAMP_CUT_CAP) {
            row_cands[pos] = { s, idx };
        }
    }
}

kernel void ami_sample_refine_gather(const device uchar *logits_raw [[buffer(0)]],
                                     const device RowSampleConfig *row_cfgs [[buffer(1)]],
                                     const device RowState *row_states [[buffer(2)]],
                                     const device uint *bitmasks [[buffer(3)]],
                                     const device GpuBiasEntry *bias_tables [[buffer(4)]],
                                     const device uint *bias_keys [[buffer(5)]],
                                     const device GpuFreqEntry *freq_tables [[buffer(6)]],
                                     const device uint *freq_keys [[buffer(7)]],
                                     device atomic_uint *cut_cand_counts [[buffer(8)]],
                                     device CutCandidate *cut_cands [[buffer(9)]],
                                     constant SampleDispatchParams &p [[buffer(10)]],
                                     uint2 tg_pos [[threadgroup_position_in_grid]],
                                     uint tid [[thread_index_in_threadgroup]]) {
    const uint blk = tg_pos.x;
    const uint row = tg_pos.y;
    if (blk >= p.num_blocks || row >= p.num_rows) {
        return;
    }
    const RowState st = row_states[row];
    if (st.need_refine != 1u || st.status != SAMP_STATUS_OK) {
        return;
    }

    const RowSampleConfig cfg = row_cfgs[row];
    const ulong row_off = ulong(row) * p.vocab_size;
    const device uint *bm_row = bitmasks + ulong(cfg.bitmask_row_index) * p.bitmask_stride_u32;
    const device GpuBiasEntry *b_tbl = bias_tables + ulong(row) * SAMP_BIAS_CAP;
    const device uint *b_keys = bias_keys + ulong(row) * 64u;
    const device GpuFreqEntry *f_tbl = freq_tables + ulong(row) * SAMP_FREQ_CAP;
    const device uint *f_keys = freq_keys + ulong(row) * 2048u;

    threadgroup atomic_uint s_bloom[SAMP_BLOOM_WORDS];
    const bool use_bloom = SAMP_HAS_OVERRIDES &&
        (((cfg.flags & SAMP_FLAG_HAS_BIAS) != 0u && cfg.bias_count > 0u) ||
         ((cfg.flags & SAMP_FLAG_HAS_PENALTIES) != 0u && cfg.freq_count > 0u));
    if (use_bloom) {
        build_override_bloom(s_bloom, b_keys, cfg.bias_count, f_keys, cfg.freq_count, tid);
    }

    uint start_id = 0u;
    uint end_id = 0u;
    block_range(blk, p.num_blocks, p.vocab_size, start_id, end_id);

    const float gmax = st.global_max;
    const uint argmax_id = st.global_argmax_id;
    const float inv_t = cfg.inv_temp;
    const float s_cutoff = (cfg.min_p > 0.0f) ? max(-80.0f, log(cfg.min_p)) : -80.0f;
    const uint cut_b = st.cut_bucket;
    const float b_hi = st.cut_s_hi;
    const float b_lo = st.cut_s_lo;
    const bool bm_active = SAMP_USE_BITMASK && ((cfg.flags & SAMP_FLAG_USE_BITMASK) != 0u);
    const bool has_ov = SAMP_HAS_OVERRIDES && use_bloom;
    const bool use_vec4 = is_row_vec4(p, row_off);
    device atomic_uint *row_cnt = &cut_cand_counts[row];
    device CutCandidate *row_cands = cut_cands + ulong(row) * SAMP_CUT_CAP;
    bool dummy_inv = false;

    if (use_vec4) {
        const uint vec_start = start_id >> 2;
        const uint vec_end = end_id >> 2;
        for (uint vi = vec_start + tid; vi < vec_end; vi += 256u) {
            const float4 fv = load_raw_logit4(logits_raw, row_off, vi);
            const uint base = vi << 2;
            uint bm4 = 0xFu;
            if (bm_active) {
                bm4 = (bm_row[base >> 5] >> (base & 31u)) & 0xFu;
                if (bm4 == 0u) {
                    continue;
                }
            }
            if (!has_ov) {
                const float4 sv = (fv - gmax) * inv_t;
                if (all(sv < b_lo) || all(sv > b_hi)) {
                    continue;
                }
                if ((bm4 & 1u) != 0u) refine_gather_one(fv.x, base + 0u, gmax, argmax_id, inv_t, s_cutoff, cut_b, row_cnt, row_cands);
                if ((bm4 & 2u) != 0u) refine_gather_one(fv.y, base + 1u, gmax, argmax_id, inv_t, s_cutoff, cut_b, row_cnt, row_cands);
                if ((bm4 & 4u) != 0u) refine_gather_one(fv.z, base + 2u, gmax, argmax_id, inv_t, s_cutoff, cut_b, row_cnt, row_cands);
                if ((bm4 & 8u) != 0u) refine_gather_one(fv.w, base + 3u, gmax, argmax_id, inv_t, s_cutoff, cut_b, row_cnt, row_cands);
            } else {
                if ((bm4 & 1u) != 0u) {
                    const float v0 = apply_overrides_one(fv.x, base + 0u, b_tbl, f_tbl, s_bloom, cfg, true, dummy_inv);
                    refine_gather_one(v0, base + 0u, gmax, argmax_id, inv_t, s_cutoff, cut_b, row_cnt, row_cands);
                }
                if ((bm4 & 2u) != 0u) {
                    const float v1 = apply_overrides_one(fv.y, base + 1u, b_tbl, f_tbl, s_bloom, cfg, true, dummy_inv);
                    refine_gather_one(v1, base + 1u, gmax, argmax_id, inv_t, s_cutoff, cut_b, row_cnt, row_cands);
                }
                if ((bm4 & 4u) != 0u) {
                    const float v2 = apply_overrides_one(fv.z, base + 2u, b_tbl, f_tbl, s_bloom, cfg, true, dummy_inv);
                    refine_gather_one(v2, base + 2u, gmax, argmax_id, inv_t, s_cutoff, cut_b, row_cnt, row_cands);
                }
                if ((bm4 & 8u) != 0u) {
                    const float v3 = apply_overrides_one(fv.w, base + 3u, b_tbl, f_tbl, s_bloom, cfg, true, dummy_inv);
                    refine_gather_one(v3, base + 3u, gmax, argmax_id, inv_t, s_cutoff, cut_b, row_cnt, row_cands);
                }
            }
        }
        const uint tail = vec_end << 2;
        for (uint i = tail + tid; i < end_id; i += 256u) {
            const float v = eval_constrained_logit(
                logits_raw, row_off, i, bm_row, b_tbl, f_tbl, s_bloom, cfg, use_bloom, dummy_inv
            );
            refine_gather_one(v, i, gmax, argmax_id, inv_t, s_cutoff, cut_b, row_cnt, row_cands);
        }
    } else {
        for (uint i = start_id + tid; i < end_id; i += 256u) {
            const float v = eval_constrained_logit(
                logits_raw, row_off, i, bm_row, b_tbl, f_tbl, s_bloom, cfg, use_bloom, dummy_inv
            );
            refine_gather_one(v, i, gmax, argmax_id, inv_t, s_cutoff, cut_b, row_cnt, row_cands);
        }
    }
}

kernel void ami_sample_refine_sort(const device uchar *logits_raw [[buffer(0)]],
                                   const device RowSampleConfig *row_cfgs [[buffer(1)]],
                                   const device RowState *row_states [[buffer(2)]],
                                   const device uint *bitmasks [[buffer(3)]],
                                   const device GpuBiasEntry *bias_tables [[buffer(4)]],
                                   const device GpuFreqEntry *freq_tables [[buffer(5)]],
                                   const device uint *cut_cand_counts [[buffer(6)]],
                                   const device CutCandidate *cut_cands [[buffer(7)]],
                                   device GpuSampleOutput *outputs [[buffer(8)]],
                                   constant SampleDispatchParams &p [[buffer(9)]],
                                   uint row [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]]) {
    if (row >= p.num_rows) {
        return;
    }
    const RowState st = row_states[row];
    if (st.need_refine != 1u || st.status != SAMP_STATUS_OK) {
        return;
    }
    const RowSampleConfig cfg = row_cfgs[row];

    // 16 KiB SRAM bitonic sort over up to 2048 candidates.
    threadgroup float s_cand_s[SAMP_CUT_CAP];
    threadgroup uint s_cand_id[SAMP_CUT_CAP];

    const uint n = min(cut_cand_counts[row], SAMP_CUT_CAP);
    uint pow2 = 1u;
    while (pow2 < n) {
        pow2 <<= 1u;
    }
    const device CutCandidate *row_cands = cut_cands + ulong(row) * SAMP_CUT_CAP;
    for (uint j = tid; j < pow2; j += 256u) {
        if (j < n) {
            const CutCandidate c = row_cands[j];
            s_cand_s[j] = c.s;
            s_cand_id[j] = c.token_id;
        } else {
            s_cand_s[j] = -INFINITY;
            s_cand_id[j] = SAMP_EMPTY_ID;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k_sz = 2u; k_sz <= pow2; k_sz <<= 1u) {
        for (uint j_step = k_sz >> 1u; j_step > 0u; j_step >>= 1u) {
            for (uint idx = tid; idx < pow2; idx += 256u) {
                const uint ixj = idx ^ j_step;
                if (ixj > idx) {
                    const float sa = s_cand_s[idx];
                    const uint ida = s_cand_id[idx];
                    const float sb = s_cand_s[ixj];
                    const uint idb = s_cand_id[ixj];
                    const bool a_better = (sa > sb) || (sa == sb && ida < idb);
                    const bool want_desc = ((idx & k_sz) == 0u);
                    if ((want_desc && !a_better) || (!want_desc && a_better)) {
                        s_cand_s[idx] = sb;
                        s_cand_id[idx] = idb;
                        s_cand_s[ixj] = sa;
                        s_cand_id[ixj] = ida;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (tid == 0u) {
        const ulong step_seed = (ulong(cfg.seed_hi) << 32) | ulong(cfg.seed_lo);
        const uint limit_k = min(n, st.rem_k);
        uint keep_n = 0u;
        if (isfinite(st.rem_mass)) {
            float cum = 0.0f;
            for (uint j = 0u; j < limit_k; j++) {
                const float s = s_cand_s[j];
                cum += (s < -80.0f) ? 0.0f : exp(s);
                keep_n = j + 1u;
                if (cum >= st.rem_mass) {
                    break;
                }
            }
        } else {
            keep_n = limit_k;
        }
        if (keep_n == 0u && st.prefix_best_tok == SAMP_EMPTY_ID && n > 0u) {
            keep_n = 1u;
        }
        ulong best_in_cut = 0ul;
        for (uint j = 0u; j < keep_n; j++) {
            const uint id = s_cand_id[j];
            const float g = s_cand_s[j] + gumbel_noise(step_seed, id);
            const ulong pk = pack_gumbel_u64(g, id);
            if (pk > best_in_cut) {
                best_in_cut = pk;
            }
        }
        const ulong prefix_pk = (st.prefix_best_tok != SAMP_EMPTY_ID)
            ? ((ulong(st.prefix_best_gumbel) << 32) | ulong(0xFFFFFFFFu - st.prefix_best_tok))
            : 0ul;
        const ulong best_all = max(prefix_pk, best_in_cut);
        const uint chosen = (best_all != 0ul) ? unpack_gumbel_token(best_all) : st.global_argmax_id;

        float out_lp = 0.0f;
        if ((cfg.flags & SAMP_FLAG_NEED_LOGPROBS) != 0u) {
            const ulong row_off = ulong(row) * p.vocab_size;
            const device uint *bm_row = bitmasks + ulong(cfg.bitmask_row_index) * p.bitmask_stride_u32;
            const device GpuBiasEntry *b_tbl = bias_tables + ulong(row) * SAMP_BIAS_CAP;
            const device GpuFreqEntry *f_tbl = freq_tables + ulong(row) * SAMP_FREQ_CAP;
            threadgroup atomic_uint dummy_bloom[1];
            bool dummy_inv = false;
            const float v_chosen = eval_constrained_logit(
                logits_raw, row_off, chosen, bm_row, b_tbl, f_tbl, dummy_bloom, cfg, false, dummy_inv
            );
            out_lp = isfinite(v_chosen) ? ((v_chosen - st.global_max) * cfg.inv_temp - st.global_log_z) : -INFINITY;
        }
        outputs[row].token_id = chosen;
        outputs[row].logprob = out_lp;
        outputs[row].status = SAMP_STATUS_OK;
    }
}

// ===========================================================================
// Massive-Tie Indirect Fallback (`need_refine == 2`, >2048 tokens in cut_b)
// ===========================================================================

kernel void ami_sample_refine_tie(const device uchar *logits_raw [[buffer(0)]],
                                  const device RowSampleConfig *row_cfgs [[buffer(1)]],
                                  const device RowState *row_states [[buffer(2)]],
                                  const device uint *bitmasks [[buffer(3)]],
                                  const device GpuBiasEntry *bias_tables [[buffer(4)]],
                                  const device uint *bias_keys [[buffer(5)]],
                                  const device GpuFreqEntry *freq_tables [[buffer(6)]],
                                  const device uint *freq_keys [[buffer(7)]],
                                  device GpuSampleOutput *outputs [[buffer(8)]],
                                  constant SampleDispatchParams &p [[buffer(9)]],
                                  uint row [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]],
                                  uint sg_id [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
    if (row >= p.num_rows) {
        return;
    }
    const RowState st = row_states[row];
    if (st.need_refine != 2u || st.status != SAMP_STATUS_OK) {
        return;
    }

    const RowSampleConfig cfg = row_cfgs[row];
    const ulong row_off = ulong(row) * p.vocab_size;
    const device uint *bm_row = bitmasks + ulong(cfg.bitmask_row_index) * p.bitmask_stride_u32;
    const device GpuBiasEntry *b_tbl = bias_tables + ulong(row) * SAMP_BIAS_CAP;
    const device uint *b_keys = bias_keys + ulong(row) * 64u;
    const device GpuFreqEntry *f_tbl = freq_tables + ulong(row) * SAMP_FREQ_CAP;
    const device uint *f_keys = freq_keys + ulong(row) * 2048u;

    threadgroup atomic_uint s_bloom[SAMP_BLOOM_WORDS];
    threadgroup atomic_uint s_sub_cnt[256];
    threadgroup float s_sub_mass[256];
    threadgroup uint s_sg_tok[8];
    threadgroup uint s_sg_key[8];
    threadgroup float s_lo_cur;
    threadgroup float s_hi_cur;
    threadgroup uint s_rem_k_cur;
    threadgroup float s_rem_mass_cur;
    threadgroup uint s_id_lo;
    threadgroup uint s_id_hi;
    threadgroup uint s_best_key_hi;
    threadgroup uint s_best_tok_hi;
    threadgroup uint s_sub_cut;
    threadgroup uint s_ib_cut;

    const bool use_bloom = SAMP_HAS_OVERRIDES &&
        (((cfg.flags & SAMP_FLAG_HAS_BIAS) != 0u && cfg.bias_count > 0u) ||
         ((cfg.flags & SAMP_FLAG_HAS_PENALTIES) != 0u && cfg.freq_count > 0u));
    if (use_bloom) {
        build_override_bloom(s_bloom, b_keys, cfg.bias_count, f_keys, cfg.freq_count, tid);
    }

    const uint cut_b = st.cut_bucket;
    const float gmax = st.global_max;
    const uint argmax_id = st.global_argmax_id;
    const float inv_t = cfg.inv_temp;
    const float s_cutoff = (cfg.min_p > 0.0f) ? max(-80.0f, log(cfg.min_p)) : -80.0f;
    const ulong step_seed = (ulong(cfg.seed_hi) << 32) | ulong(cfg.seed_lo);
    bool dummy_inv = false;

    if (tid == 0u) {
        s_lo_cur = st.cut_s_lo;
        s_hi_cur = st.cut_s_hi;
        s_rem_k_cur = st.rem_k;
        s_rem_mass_cur = st.rem_mass;
        s_best_key_hi = st.prefix_best_gumbel;
        s_best_tok_hi = st.prefix_best_tok;
        s_sub_cut = 0u;
        s_ib_cut = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint round = 0u; round < 2u; round++) {
        const float lo = s_lo_cur;
        const float hi = s_hi_cur;
        const float span = hi - lo;
        if (span <= 1e-6f || s_rem_k_cur == 0u) {
            break;
        }
        atomic_store_explicit(&s_sub_cnt[tid], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float inv_span = 256.0f / span;
        for (uint i = tid; i < p.vocab_size; i += 256u) {
            const float v = eval_constrained_logit(
                logits_raw, row_off, i, bm_row, b_tbl, f_tbl, s_bloom, cfg, use_bloom, dummy_inv
            );
            if (!isfinite(v)) { continue; }
            const float s = (v - gmax) * inv_t;
            if (s < s_cutoff && i != argmax_id) { continue; }
            if (bucket_of_s(s) != cut_b || s < lo || s > hi) { continue; }
            const uint sb = min(uint((hi - s) * inv_span), 255u);
            atomic_fetch_add_explicit(&s_sub_cnt[sb], 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float sub_w = span * (1.0f / 256.0f);
        const float sub_mid = hi - (float(tid) + 0.5f) * sub_w;
        s_sub_mass[tid] = float(atomic_load_explicit(&s_sub_cnt[tid], memory_order_relaxed)) * exp(sub_mid);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0u) {
            uint c = 0u;
            float m = 0.0f;
            uint sc = 255u;
            for (uint sb = 0u; sb < 256u; sb++) {
                const uint scnt = atomic_load_explicit(&s_sub_cnt[sb], memory_order_relaxed);
                const float smass = s_sub_mass[sb];
                const bool hit_k = (s_rem_k_cur != 0xFFFFFFFFu) && (c + scnt >= s_rem_k_cur);
                const bool hit_m = isfinite(s_rem_mass_cur) && (m + smass >= s_rem_mass_cur);
                if (hit_k || hit_m) {
                    sc = sb;
                    break;
                }
                c += scnt;
                m += smass;
            }
            s_sub_cut = sc;
            if (s_rem_k_cur != 0xFFFFFFFFu) {
                s_rem_k_cur = (s_rem_k_cur > c) ? (s_rem_k_cur - c) : 0u;
            }
            if (isfinite(s_rem_mass_cur)) {
                s_rem_mass_cur = max(s_rem_mass_cur - m, 0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint sc = s_sub_cut;
        const float new_hi = hi - float(sc) * sub_w;
        const float new_lo = (sc == 255u) ? lo : (hi - float(sc + 1u) * sub_w);

        if (sc > 0u) {
            ulong loc_best = 0ul;
            for (uint i = tid; i < p.vocab_size; i += 256u) {
                const float v = eval_constrained_logit(
                    logits_raw, row_off, i, bm_row, b_tbl, f_tbl, s_bloom, cfg, use_bloom, dummy_inv
                );
                if (!isfinite(v)) { continue; }
                const float s = (v - gmax) * inv_t;
                if (s < s_cutoff && i != argmax_id) { continue; }
                if (bucket_of_s(s) != cut_b) { continue; }
                if (s > new_hi && s <= hi) {
                    const ulong pk = pack_gumbel_u64(s + gumbel_noise(step_seed, i), i);
                    if (pk > loc_best) { loc_best = pk; }
                }
            }
            uint bk = (loc_best != 0ul) ? unpack_gumbel_key(loc_best) : 0u;
            uint bt = (loc_best != 0ul) ? unpack_gumbel_token(loc_best) : SAMP_EMPTY_ID;
            for (ushort off = 16u; off > 0u; off >>= 1u) {
                const uint ok = simd_shuffle_down(bk, off);
                const uint ot = simd_shuffle_down(bt, off);
                if (ok > bk || (ok == bk && ot < bt)) { bk = ok; bt = ot; }
            }
            if (lane == 0u) { s_sg_key[sg_id] = bk; s_sg_tok[sg_id] = bt; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u) {
                for (uint s_i = 0u; s_i < 8u; s_i++) {
                    if (s_sg_key[s_i] > s_best_key_hi || (s_sg_key[s_i] == s_best_key_hi && s_sg_tok[s_i] < s_best_tok_hi)) {
                        s_best_key_hi = s_sg_key[s_i];
                        s_best_tok_hi = s_sg_tok[s_i];
                    }
                }
            }
        }
        if (tid == 0u) {
            s_hi_cur = new_hi;
            s_lo_cur = new_lo;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        const float w_tie = max(exp(s_hi_cur), 1e-30f);
        uint keep_m = isfinite(s_rem_mass_cur) ? uint(ceil(s_rem_mass_cur / w_tie)) : 0xFFFFFFFFu;
        if (isfinite(s_rem_mass_cur) && keep_m == 0u && s_best_tok_hi == SAMP_EMPTY_ID) {
            keep_m = 1u;
        }
        s_rem_k_cur = min(s_rem_k_cur, keep_m);
        s_id_lo = 0u;
        s_id_hi = p.vocab_size;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint id_r = 0u; id_r < 2u; id_r++) {
        const uint ilo = s_id_lo;
        const uint ihi = s_id_hi;
        const uint ispan = ihi - ilo;
        if (ispan <= 1u || s_rem_k_cur == 0u) {
            break;
        }
        const uint bin_w = (ispan + 255u) / 256u;
        atomic_store_explicit(&s_sub_cnt[tid], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float lo = s_lo_cur;
        const float hi = s_hi_cur;
        for (uint i = ilo + tid; i < ihi; i += 256u) {
            const float v = eval_constrained_logit(
                logits_raw, row_off, i, bm_row, b_tbl, f_tbl, s_bloom, cfg, use_bloom, dummy_inv
            );
            if (!isfinite(v)) { continue; }
            const float s = (v - gmax) * inv_t;
            if (s < s_cutoff && i != argmax_id) { continue; }
            if (bucket_of_s(s) != cut_b || s < lo || s > hi) { continue; }
            const uint ib = min((i - ilo) / bin_w, 255u);
            atomic_fetch_add_explicit(&s_sub_cnt[ib], 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0u) {
            uint c = 0u;
            uint ic = 255u;
            for (uint ib = 0u; ib < 256u; ib++) {
                const uint cnt_ib = atomic_load_explicit(&s_sub_cnt[ib], memory_order_relaxed);
                if (c + cnt_ib >= s_rem_k_cur) {
                    ic = ib;
                    break;
                }
                c += cnt_ib;
            }
            s_ib_cut = ic;
            s_rem_k_cur = (s_rem_k_cur > c) ? (s_rem_k_cur - c) : 0u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint ic = s_ib_cut;
        const uint split_lo = min(ilo + ic * bin_w, ihi);
        const uint split_hi = min(split_lo + bin_w, ihi);

        if (split_lo > ilo) {
            ulong loc_best = 0ul;
            for (uint i = ilo + tid; i < split_lo; i += 256u) {
                const float v = eval_constrained_logit(
                    logits_raw, row_off, i, bm_row, b_tbl, f_tbl, s_bloom, cfg, use_bloom, dummy_inv
                );
                if (!isfinite(v)) { continue; }
                const float s = (v - gmax) * inv_t;
                if (s < s_cutoff && i != argmax_id) { continue; }
                if (bucket_of_s(s) != cut_b || s < lo || s > hi) { continue; }
                const ulong pk = pack_gumbel_u64(s + gumbel_noise(step_seed, i), i);
                if (pk > loc_best) { loc_best = pk; }
            }
            uint bk = (loc_best != 0ul) ? unpack_gumbel_key(loc_best) : 0u;
            uint bt = (loc_best != 0ul) ? unpack_gumbel_token(loc_best) : SAMP_EMPTY_ID;
            for (ushort off = 16u; off > 0u; off >>= 1u) {
                const uint ok = simd_shuffle_down(bk, off);
                const uint ot = simd_shuffle_down(bt, off);
                if (ok > bk || (ok == bk && ot < bt)) { bk = ok; bt = ot; }
            }
            if (lane == 0u) { s_sg_key[sg_id] = bk; s_sg_tok[sg_id] = bt; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u) {
                for (uint s_i = 0u; s_i < 8u; s_i++) {
                    if (s_sg_key[s_i] > s_best_key_hi || (s_sg_key[s_i] == s_best_key_hi && s_sg_tok[s_i] < s_best_tok_hi)) {
                        s_best_key_hi = s_sg_key[s_i];
                        s_best_tok_hi = s_sg_tok[s_i];
                    }
                }
            }
        }
        if (tid == 0u) {
            s_id_lo = split_lo;
            s_id_hi = split_hi;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        uint rem = s_rem_k_cur;
        const float lo = s_lo_cur;
        const float hi = s_hi_cur;
        for (uint i = s_id_lo; i < s_id_hi && rem > 0u; i++) {
            const float v = eval_constrained_logit(
                logits_raw, row_off, i, bm_row, b_tbl, f_tbl, s_bloom, cfg, use_bloom, dummy_inv
            );
            if (!isfinite(v)) { continue; }
            const float s = (v - gmax) * inv_t;
            if (s < s_cutoff && i != argmax_id) { continue; }
            if (bucket_of_s(s) != cut_b || s < lo || s > hi) { continue; }
            const ulong pk = pack_gumbel_u64(s + gumbel_noise(step_seed, i), i);
            const uint k_pk = unpack_gumbel_key(pk);
            if (k_pk > s_best_key_hi || (k_pk == s_best_key_hi && i < s_best_tok_hi)) {
                s_best_key_hi = k_pk;
                s_best_tok_hi = i;
            }
            rem--;
        }
        const uint chosen = (s_best_tok_hi != SAMP_EMPTY_ID) ? s_best_tok_hi : argmax_id;

        float out_lp = 0.0f;
        if ((cfg.flags & SAMP_FLAG_NEED_LOGPROBS) != 0u) {
            const float v_chosen = eval_constrained_logit(
                logits_raw, row_off, chosen, bm_row, b_tbl, f_tbl, s_bloom, cfg, use_bloom, dummy_inv
            );
            out_lp = isfinite(v_chosen) ? ((v_chosen - gmax) * inv_t - st.global_log_z) : -INFINITY;
        }
        outputs[row].token_id = chosen;
        outputs[row].logprob = out_lp;
        outputs[row].status = SAMP_STATUS_OK;
    }
}

// ===========================================================================
// Optional Exact Top-Logprobs Extraction (`top_logprobs` 1..20, ZERO spill!)
// Dispatched ONLY when `any_row_wants_top_logprobs == true`.
// ===========================================================================

kernel void ami_sample_toplogprobs_pass1(const device uchar *logits_raw [[buffer(0)]],
                                         const device RowSampleConfig *row_cfgs [[buffer(1)]],
                                         const device RowState *row_states [[buffer(2)]],
                                         const device uint *bitmasks [[buffer(3)]],
                                         const device GpuBiasEntry *bias_tables [[buffer(4)]],
                                         const device uint *bias_keys [[buffer(5)]],
                                         const device GpuFreqEntry *freq_tables [[buffer(6)]],
                                         const device uint *freq_keys [[buffer(7)]],
                                         device GpuTokenLogprob *block_top [[buffer(8)]],
                                         constant SampleDispatchParams &p [[buffer(9)]],
                                         uint2 tg_pos [[threadgroup_position_in_grid]],
                                         uint tid [[thread_index_in_threadgroup]],
                                         uint sg_id [[simdgroup_index_in_threadgroup]],
                                         uint lane [[thread_index_in_simdgroup]]) {
    const uint blk = tg_pos.x;
    const uint row = tg_pos.y;
    if (blk >= p.num_blocks || row >= p.num_rows) {
        return;
    }

    const RowSampleConfig cfg = row_cfgs[row];
    const RowState st = row_states[row];
    const uint k = min(cfg.top_logprobs, SAMP_MAX_TOP_LOGPROBS);
    if (st.status != SAMP_STATUS_OK || k == 0u) {
        return;
    }

    const ulong row_off = ulong(row) * p.vocab_size;
    const device uint *bm_row = bitmasks + ulong(cfg.bitmask_row_index) * p.bitmask_stride_u32;
    const device GpuBiasEntry *b_tbl = bias_tables + ulong(row) * SAMP_BIAS_CAP;
    const device uint *b_keys = bias_keys + ulong(row) * 64u;
    const device GpuFreqEntry *f_tbl = freq_tables + ulong(row) * SAMP_FREQ_CAP;
    const device uint *f_keys = freq_keys + ulong(row) * 2048u;

    threadgroup atomic_uint s_bloom[SAMP_BLOOM_WORDS];
    threadgroup float s_sg_v[8][20];
    threadgroup uint s_sg_i[8][20];

    const bool use_bloom = SAMP_HAS_OVERRIDES &&
        (((cfg.flags & SAMP_FLAG_HAS_BIAS) != 0u && cfg.bias_count > 0u) ||
         ((cfg.flags & SAMP_FLAG_HAS_PENALTIES) != 0u && cfg.freq_count > 0u));
    if (use_bloom) {
        build_override_bloom(s_bloom, b_keys, cfg.bias_count, f_keys, cfg.freq_count, tid);
    }

    if (lane < 20u) {
        s_sg_v[sg_id][lane] = -INFINITY;
        s_sg_i[sg_id][lane] = SAMP_EMPTY_ID;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    uint start_id = 0u;
    uint end_id = 0u;
    block_range(blk, p.num_blocks, p.vocab_size, start_id, end_id);

    bool dummy_inv = false;
    const uint span = end_id - start_id;
    const uint iters = (span + 255u) >> 8;

    for (uint it = 0u; it < iters; it++) {
        const uint i = start_id + (it << 8) + tid;
        float v = -INFINITY;
        if (i < end_id) {
            v = eval_constrained_logit(
                logits_raw, row_off, i, bm_row, b_tbl, f_tbl, s_bloom, cfg, use_bloom, dummy_inv
            );
        }
        const float cur_min_v = s_sg_v[sg_id][k - 1u];
        const uint cur_min_i = s_sg_i[sg_id][k - 1u];
        const bool cand = isfinite(v) && (v > cur_min_v || (v == cur_min_v && i < cur_min_i));
        uint mask = uint(simd_vote::vote_t(simd_ballot(cand)));
        while (mask != 0u) {
            const uint src_lane = ctz(mask);
            mask &= (mask - 1u);
            const float cv = simd_broadcast(v, ushort(src_lane));
            const uint ci = simd_broadcast(i, ushort(src_lane));
            if (lane == 0u) {
                const float mv = s_sg_v[sg_id][k - 1u];
                const uint mi = s_sg_i[sg_id][k - 1u];
                if (cv > mv || (cv == mv && ci < mi)) {
                    uint pos = k - 1u;
                    while (pos > 0u) {
                        const float pv = s_sg_v[sg_id][pos - 1u];
                        const uint pi = s_sg_i[sg_id][pos - 1u];
                        if (cv > pv || (cv == pv && ci < pi)) {
                            s_sg_v[sg_id][pos] = pv;
                            s_sg_i[sg_id][pos] = pi;
                            pos--;
                        } else {
                            break;
                        }
                    }
                    s_sg_v[sg_id][pos] = cv;
                    s_sg_i[sg_id][pos] = ci;
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        uint ptr[8] = { 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u };
        const ulong out_base = (ulong(row) * SAMP_MAX_BLOCKS + blk) * SAMP_MAX_TOP_LOGPROBS;
        for (uint rank = 0u; rank < k; rank++) {
            float bv = -INFINITY;
            uint bi = SAMP_EMPTY_ID;
            uint win_sg = 0u;
            for (uint s = 0u; s < 8u; s++) {
                const uint p_s = ptr[s];
                if (p_s < k) {
                    const float cv = s_sg_v[s][p_s];
                    const uint ci = s_sg_i[s][p_s];
                    if (cv > bv || (cv == bv && ci < bi)) {
                        bv = cv;
                        bi = ci;
                        win_sg = s;
                    }
                }
            }
            if (bi != SAMP_EMPTY_ID) {
                ptr[win_sg]++;
            }
            block_top[out_base + rank] = { bi, bv };
        }
    }
}

kernel void ami_sample_toplogprobs_pass2(const device RowSampleConfig *row_cfgs [[buffer(0)]],
                                         const device RowState *row_states [[buffer(1)]],
                                         const device GpuTokenLogprob *block_top [[buffer(2)]],
                                         device GpuSampleOutput *outputs [[buffer(3)]],
                                         constant SampleDispatchParams &p [[buffer(4)]],
                                         uint row [[threadgroup_position_in_grid]],
                                         uint lane [[thread_index_in_simdgroup]]) {
    if (row >= p.num_rows) {
        return;
    }
    const RowSampleConfig cfg = row_cfgs[row];
    const RowState st = row_states[row];
    const uint k = min(cfg.top_logprobs, SAMP_MAX_TOP_LOGPROBS);
    if (st.status != SAMP_STATUS_OK || k == 0u) {
        return;
    }

    const float inv_t_lp = (cfg.temperature > 0.0f) ? cfg.inv_temp : 1.0f;
    const ulong row_base = ulong(row) * SAMP_MAX_BLOCKS * SAMP_MAX_TOP_LOGPROBS;

    uint cursor = 0u;
    const bool valid_blk = lane < p.num_blocks;
    uint out_len = 0u;

    for (uint rank = 0u; rank < k; rank++) {
        float cv = -INFINITY;
        uint ci = SAMP_EMPTY_ID;
        if (valid_blk && cursor < k) {
            const GpuTokenLogprob e = block_top[row_base + ulong(lane) * SAMP_MAX_TOP_LOGPROBS + cursor];
            cv = e.logprob;
            ci = e.token_id;
        }

        float bv = cv;
        uint bi = ci;
        uint bl = lane;
        for (ushort off = 16u; off > 0u; off >>= 1u) {
            const float ov = simd_shuffle_down(bv, off);
            const uint oi = simd_shuffle_down(bi, off);
            const uint ol = simd_shuffle_down(bl, off);
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov;
                bi = oi;
                bl = ol;
            }
        }
        const float win_v = simd_broadcast_first(bv);
        const uint win_i = simd_broadcast_first(bi);
        const uint win_lane = simd_broadcast_first(bl);
        if (win_i == SAMP_EMPTY_ID || !isfinite(win_v)) {
            break;
        }
        if (lane == win_lane) {
            cursor++;
        }
        if (lane == 0u) {
            const float lp = (win_v - st.global_max) * inv_t_lp - st.global_log_z;
            outputs[row].top_logprobs[out_len] = { win_i, lp };
            out_len++;
        }
    }
    if (lane == 0u) {
        outputs[row].top_logprobs_len = out_len;
    }
}
