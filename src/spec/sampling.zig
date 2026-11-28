//! Unified zero-allocation constrained token sampler and streaming `StopHoldback`
//! prefix ring (capabilities C18, C22, D11, Milestone M7.5, ADR-0022).
//!
//! Features:
//! - Unconstrained greedy (`T = 0`) fast-path bypass: zero scratch writes, zero
//!   softmax, zero RNG draws when penalties, biases, logprobs, and grammar masks
//!   are inactive.
//! - Full constrained & stochastic pipeline over `f16` or `f32` logits:
//!   `C22` `u32` token bitmask filtering, sparse `logit_bias`, O(unique-seen)
//!   `presence_penalty`, `frequency_penalty`, and `repetition_penalty`,
//!   `temperature`, `top_k`, `min_p`, `top_p`, `logprobs` / `top_logprobs`, and
//!   `D11` `Xoshiro256` RNG draws with zero heap allocation.
//! - Zero-allocation `StopHoldback` ring for `stop` / `stop_sequences`: emits
//!   non-matching spans with zero delay, holds only ambiguous proper prefixes,
//!   and never leaks matched stop strings to streaming clients.

const std = @import("std");
const grammar = @import("grammar.zig");

pub const MAX_LOGIT_BIAS_ENTRIES: usize = 64;
pub const MAX_TOP_LOGPROBS: usize = 20;
pub const MAX_STOP_SEQUENCES: usize = 8;
pub const MAX_STOP_BYTES: usize = 64;

pub const SampleError = error{
    EmptyVocab,
    InvalidTemperature,
    InvalidTopP,
    InvalidMinP,
    InvalidRepetitionPenalty,
    InvalidTopLogprobs,
    InvalidLogits,
    ScratchTooSmall,
    BitmaskTooSmall,
    EmptyAllowedSet,
};

pub const StopConfigError = error{
    EmptyStopSequence,
    StopSequenceTooLong,
    TooManyStopSequences,
};

pub const LogitBiasEntry = struct {
    token_id: u32,
    bias: f32,
};

pub const TokenLogprob = struct {
    token_id: u32,
    logprob: f32,
};

pub const SampleOutput = struct {
    token_id: u32,
    logprob: f32 = 0.0,
    top_logprobs: [MAX_TOP_LOGPROBS]TokenLogprob = undefined,
    top_logprobs_len: u8 = 0,

    pub fn topLogprobsSlice(self: *const SampleOutput) []const TokenLogprob {
        return self.top_logprobs[0..self.top_logprobs_len];
    }
};

pub const SamplingParams = struct {
    temperature: f32 = 0.0,
    top_p: f32 = 1.0,
    top_k: u32 = 0,
    min_p: f32 = 0.0,
    presence_penalty: f32 = 0.0,
    frequency_penalty: f32 = 0.0,
    repetition_penalty: f32 = 1.0,
    logit_bias: []const LogitBiasEntry = &.{},
    logprobs: bool = false,
    top_logprobs: u8 = 0,
    allowed_token_mask: ?[]const u32 = null,

    pub inline fn hasPenalties(self: SamplingParams) bool {
        return self.presence_penalty != 0.0 or
            self.frequency_penalty != 0.0 or
            self.repetition_penalty != 1.0;
    }

    pub inline fn isUnconstrainedGreedy(self: SamplingParams) bool {
        return self.temperature == 0.0 and
            !self.hasPenalties() and
            self.logit_bias.len == 0 and
            !self.logprobs and
            self.top_logprobs == 0 and
            self.allowed_token_mask == null;
    }

    pub fn validate(self: SamplingParams) SampleError!void {
        if (!std.math.isFinite(self.temperature) or self.temperature < 0.0) {
            return error.InvalidTemperature;
        }
        if (!std.math.isFinite(self.top_p) or !(self.top_p > 0.0) or self.top_p > 1.0) {
            return error.InvalidTopP;
        }
        if (!std.math.isFinite(self.min_p) or self.min_p < 0.0 or self.min_p > 1.0) {
            return error.InvalidMinP;
        }
        if (!std.math.isFinite(self.repetition_penalty) or self.repetition_penalty < 1e-4) {
            return error.InvalidRepetitionPenalty;
        }
        if (!std.math.isFinite(self.presence_penalty) or !std.math.isFinite(self.frequency_penalty)) {
            return error.InvalidLogits;
        }
        if (self.top_logprobs > MAX_TOP_LOGPROBS) {
            return error.InvalidTopLogprobs;
        }
    }
};

// ---------------------------------------------------------------------------
// Sparse Zero-Allocation Penalty Tracker (O(unique_seen) apply & reset)
// ---------------------------------------------------------------------------

pub const TokenFreqEntry = struct {
    token_id: u32,
    count: u32,
};

/// Preallocated open-addressed frequency counter over unique seen token IDs.
/// Resets and applies penalties in `O(unique_tokens)` (`0` heap allocations).
pub fn FixedPenaltyTracker(comptime max_unique: usize) type {
    const table_cap = std.math.ceilPowerOfTwo(usize, max_unique * 2) catch @compileError("max_unique too large");
    const EMPTY: u32 = std.math.maxInt(u32);

    return struct {
        const Self = @This();

        buckets: [table_cap]u32 = @splat(EMPTY),
        used_bucket_indices: [max_unique]u32 = undefined,
        entries: [max_unique]TokenFreqEntry = undefined,
        entry_count: usize = 0,

        pub fn init() Self {
            return .{};
        }

        pub fn reset(self: *Self) void {
            for (self.used_bucket_indices[0..self.entry_count]) |b_idx| {
                self.buckets[b_idx] = EMPTY;
            }
            self.entry_count = 0;
        }

        pub fn recordToken(self: *Self, token_id: u32) void {
            const mask: usize = table_cap - 1;
            var slot: usize = (@as(usize, token_id) *% 0x9E37_79B1) & mask;
            var probes: usize = 0;
            while (probes < table_cap) : (probes += 1) {
                const idx = self.buckets[slot];
                if (idx == EMPTY) {
                    if (self.entry_count >= max_unique) return;
                    const e_idx: u32 = @intCast(self.entry_count);
                    self.buckets[slot] = e_idx;
                    self.used_bucket_indices[self.entry_count] = @intCast(slot);
                    self.entries[self.entry_count] = .{ .token_id = token_id, .count = 1 };
                    self.entry_count += 1;
                    return;
                }
                if (self.entries[idx].token_id == token_id) {
                    self.entries[idx].count +|= 1;
                    return;
                }
                slot = (slot + 1) & mask;
            }
        }

        pub fn recordSlice(self: *Self, ids: []const u32) void {
            for (ids) |id| self.recordToken(id);
        }

        pub fn activeEntries(self: *const Self) []const TokenFreqEntry {
            return self.entries[0..self.entry_count];
        }
    };
}

pub const DefaultPenaltyTracker = FixedPenaltyTracker(2048);

/// Caller-provided scratch slices for non-greedy or constrained sampling (`0` heap allocs).
pub const SamplerScratch = struct {
    logits_f32: []f32 = &.{},
    cand_ids: []u32 = &.{},
};

fn extractRandom(rng_or_random: anytype) std.Random {
    const T = @TypeOf(rng_or_random);
    if (T == std.Random) return rng_or_random;
    return rng_or_random.random();
}

inline fn clampFiniteLogit(v: f64) f32 {
    const max_f32: f64 = @floatCast(std.math.floatMax(f32));
    if (v >= max_f32) return std.math.floatMax(f32);
    if (v <= -max_f32) return -std.math.floatMax(f32);
    return @floatCast(v);
}

/// Applies sparse `logit_bias` and `seen_frequencies` penalties + `allowed_token_mask`
/// into `out_logits[0..logits.len]`, returning the maximum finite modified logit.
pub fn prepareConstrainedLogits(
    comptime LogitT: type,
    logits: []const LogitT,
    params: SamplingParams,
    seen_frequencies: []const TokenFreqEntry,
    out_logits: []f32,
) SampleError!f32 {
    comptime {
        if (LogitT != f16 and LogitT != f32) {
            @compileError("LogitT must be f16 or f32");
        }
    }
    const vocab_size = logits.len;
    if (vocab_size == 0) return error.EmptyVocab;
    if (out_logits.len < vocab_size) return error.ScratchTooSmall;

    if (params.allowed_token_mask) |mask| {
        if (mask.len < grammar.bitmaskWordCount(vocab_size)) return error.BitmaskTooSmall;
        for (logits, 0..) |raw, idx| {
            const v: f32 = @floatCast(raw);
            if (std.math.isNan(v) or (std.math.isInf(v) and v > 0.0)) {
                return error.InvalidLogits;
            }
            if (grammar.isTokenAllowed(mask, @intCast(idx))) {
                out_logits[idx] = v;
            } else {
                out_logits[idx] = -std.math.inf(f32);
            }
        }
    } else {
        for (logits, 0..) |raw, idx| {
            const v: f32 = @floatCast(raw);
            if (std.math.isNan(v) or (std.math.isInf(v) and v > 0.0)) {
                return error.InvalidLogits;
            }
            out_logits[idx] = v;
        }
    }

    // Sparse logit_bias (`O(logit_bias.len)`).
    for (params.logit_bias) |lb| {
        if (lb.token_id >= vocab_size) continue;
        if (!std.math.isFinite(lb.bias) and !(std.math.isInf(lb.bias) and lb.bias < 0.0)) {
            return error.InvalidLogits;
        }
        const cur = out_logits[lb.token_id];
        if (cur == -std.math.inf(f32)) continue;
        if (std.math.isInf(lb.bias)) {
            // Explicit `-Inf` bias bans the token.
            out_logits[lb.token_id] = -std.math.inf(f32);
        } else {
            out_logits[lb.token_id] = clampFiniteLogit(@as(f64, cur) + @as(f64, lb.bias));
        }
    }

    // Sparse repetition, frequency, and presence penalties (`O(seen_frequencies.len)`).
    if (params.hasPenalties()) {
        if (!std.math.isFinite(params.presence_penalty) or !std.math.isFinite(params.frequency_penalty)) {
            return error.InvalidLogits;
        }
        if (!std.math.isFinite(params.repetition_penalty) or !(params.repetition_penalty > 0.0)) {
            return error.InvalidRepetitionPenalty;
        }
        const rep: f64 = @as(f64, params.repetition_penalty);
        for (seen_frequencies) |entry| {
            if (entry.count == 0 or entry.token_id >= vocab_size) continue;
            const cur_f32 = out_logits[entry.token_id];
            if (!std.math.isFinite(cur_f32)) continue;

            var cur: f64 = @as(f64, cur_f32);
            if (params.repetition_penalty != 1.0) {
                if (cur > 0.0) {
                    cur /= rep;
                } else {
                    cur *= rep;
                }
            }
            const freq_term = @as(f64, params.frequency_penalty) * @as(f64, @floatFromInt(entry.count));
            cur -= (freq_term + @as(f64, params.presence_penalty));
            out_logits[entry.token_id] = clampFiniteLogit(cur);
        }
    }

    var max_val: f32 = -std.math.inf(f32);
    for (out_logits[0..vocab_size]) |v| {
        if (v > max_val) max_val = v;
    }
    if (max_val == -std.math.inf(f32)) return error.EmptyAllowedSet;
    return max_val;
}

fn computeLogprobsInto(
    scaled_logits: []const f32,
    max_logit: f32,
    inv_temp: f64,
    chosen_id: u32,
    want_top_k: u8,
    out: *SampleOutput,
) void {
    const max_f64: f64 = @as(f64, max_logit);
    var sum_exp: f64 = 0.0;

    var top_ids: [MAX_TOP_LOGPROBS]u32 = undefined;
    var top_vals: [MAX_TOP_LOGPROBS]f32 = undefined;
    var top_len: usize = 0;
    const cap_k: usize = @min(@as(usize, want_top_k), MAX_TOP_LOGPROBS);

    for (scaled_logits, 0..) |l, idx| {
        if (l == -std.math.inf(f32)) continue;
        const shifted = (@as(f64, l) - max_f64) * inv_temp;
        if (shifted >= -80.0) {
            sum_exp += @exp(shifted);
        }

        if (cap_k > 0) {
            if (top_len < cap_k or l > top_vals[top_len - 1]) {
                var pos = if (top_len < cap_k) top_len else cap_k - 1;
                if (top_len < cap_k) top_len += 1;
                while (pos > 0 and l > top_vals[pos - 1]) : (pos -= 1) {
                    top_vals[pos] = top_vals[pos - 1];
                    top_ids[pos] = top_ids[pos - 1];
                }
                top_vals[pos] = l;
                top_ids[pos] = @intCast(idx);
            }
        }
    }

    const log_z: f64 = if (sum_exp > 0.0) @log(sum_exp) else 0.0;
    const chosen_l = scaled_logits[chosen_id];
    out.logprob = if (chosen_l == -std.math.inf(f32))
        -std.math.inf(f32)
    else
        @floatCast((@as(f64, chosen_l) - max_f64) * inv_temp - log_z);

    out.top_logprobs_len = @intCast(top_len);
    for (0..top_len) |i| {
        out.top_logprobs[i] = .{
            .token_id = top_ids[i],
            .logprob = @floatCast((@as(f64, top_vals[i]) - max_f64) * inv_temp - log_z),
        };
    }
}

/// Partition helper for zero-allocation O(N) average quickselect descending by `scores[id]`.
fn quickselectTopK(cand_ids: []u32, scores: []const f32, k: usize) void {
    if (k == 0 or k >= cand_ids.len) return;
    var left: usize = 0;
    var right: usize = cand_ids.len - 1;

    while (left < right) {
        const mid = left + (right - left) / 2;
        const pivot_id = cand_ids[mid];
        const pivot_val = scores[pivot_id];
        std.mem.swap(u32, &cand_ids[mid], &cand_ids[right]);

        var store = left;
        for (left..right) |i| {
            const id = cand_ids[i];
            const v = scores[id];
            if (v > pivot_val or (v == pivot_val and id < pivot_id)) {
                std.mem.swap(u32, &cand_ids[store], &cand_ids[i]);
                store += 1;
            }
        }
        std.mem.swap(u32, &cand_ids[store], &cand_ids[right]);

        if (store == k - 1) {
            return;
        } else if (store < k - 1) {
            left = store + 1;
        } else {
            right = store - 1;
        }
    }
}

fn sortCandidatesDesc(cand_ids: []u32, scores: []const f32) void {
    const Ctx = struct {
        s: []const f32,
        pub fn lessThan(ctx: @This(), a: u32, b: u32) bool {
            const va = ctx.s[a];
            const vb = ctx.s[b];
            if (va != vb) return va > vb;
            return a < b;
        }
    };
    std.mem.sort(u32, cand_ids, Ctx{ .s = scores }, Ctx.lessThan);
}

/// Unified zero-allocation token sampler supporting `f16` and `f32` logits,
/// `C22` bitmasks, penalties, `logit_bias`, `top_k`, `min_p`, `top_p`, `logprobs`,
/// and `D11` `Xoshiro256` RNGs (`*RequestRng` or `std.Random`).
pub fn sampleToken(
    comptime LogitT: type,
    logits: []const LogitT,
    params: SamplingParams,
    seen_frequencies: []const TokenFreqEntry,
    rng_source: anytype,
    scratch: SamplerScratch,
) SampleError!SampleOutput {
    comptime {
        if (LogitT != f16 and LogitT != f32) {
            @compileError("LogitT must be f16 or f32");
        }
    }
    const vocab_size = logits.len;
    if (vocab_size == 0) return error.EmptyVocab;
    try params.validate();

    // 1. Strict zero-overhead greedy fast-path bypass!
    if (params.isUnconstrainedGreedy()) {
        var best_val: f32 = -std.math.inf(f32);
        var best_id: u32 = 0;
        for (logits, 0..) |raw, idx| {
            const v: f32 = @floatCast(raw);
            if (std.math.isNan(v) or (std.math.isInf(v) and v > 0.0)) {
                return error.InvalidLogits;
            }
            if (v > best_val) {
                best_val = v;
                best_id = @intCast(idx);
            }
        }
        if (best_val == -std.math.inf(f32)) return error.InvalidLogits;
        return .{ .token_id = best_id };
    }

    // 2. Constrained / penalized / stochastic / logprob path.
    const work_logits = scratch.logits_f32;
    const max_val = try prepareConstrainedLogits(LogitT, logits, params, seen_frequencies, work_logits);

    var argmax_id: u32 = 0;
    for (work_logits[0..vocab_size], 0..) |v, idx| {
        if (v == max_val) {
            argmax_id = @intCast(idx);
            break;
        }
    }

    const need_logprobs = params.logprobs or params.top_logprobs > 0;
    const inv_temp_for_lp: f64 = if (params.temperature > 0.0) 1.0 / @as(f64, params.temperature) else 1.0;

    if (params.temperature == 0.0) {
        var out = SampleOutput{ .token_id = argmax_id };
        if (need_logprobs) {
            computeLogprobsInto(
                work_logits[0..vocab_size],
                max_val,
                inv_temp_for_lp,
                argmax_id,
                params.top_logprobs,
                &out,
            );
        }
        return out;
    }

    // 3. Stochastic (`temperature > 0.0`) filtering (`top_k` -> `min_p` -> `top_p` -> `Xoshiro256`).
    var out = SampleOutput{ .token_id = argmax_id };
    if (need_logprobs) {
        // Compute logprobs BEFORE `work_logits` is overwritten with post-filter probabilities!
        computeLogprobsInto(
            work_logits[0..vocab_size],
            max_val,
            inv_temp_for_lp,
            argmax_id,
            params.top_logprobs,
            &out,
        );
    }

    const inv_t: f64 = 1.0 / @as(f64, params.temperature);
    const max_f64: f64 = @as(f64, max_val);

    // Fast small-`top_k` path without needing `scratch.cand_ids` when `1 <= top_k <= 64`!
    var stack_cands: [64]u32 = undefined;
    var active_ids: []u32 = undefined;

    if (params.top_k > 0 and params.top_k <= stack_cands.len and params.top_k < vocab_size) {
        const k_lim: usize = @intCast(params.top_k);
        var heap_len: usize = 0;
        for (work_logits[0..vocab_size], 0..) |v, idx| {
            if (v == -std.math.inf(f32)) continue;
            const shifted = (@as(f64, v) - max_f64) * inv_t;
            if (shifted < -80.0 and idx != argmax_id) continue;

            if (heap_len < k_lim or v > work_logits[stack_cands[heap_len - 1]]) {
                var pos = if (heap_len < k_lim) heap_len else k_lim - 1;
                if (heap_len < k_lim) heap_len += 1;
                while (pos > 0 and v > work_logits[stack_cands[pos - 1]]) : (pos -= 1) {
                    stack_cands[pos] = stack_cands[pos - 1];
                }
                stack_cands[pos] = @intCast(idx);
            }
        }
        active_ids = stack_cands[0..heap_len];
    } else {
        if (scratch.cand_ids.len < vocab_size) return error.ScratchTooSmall;
        var count: usize = 0;
        for (work_logits[0..vocab_size], 0..) |v, idx| {
            if (v == -std.math.inf(f32)) continue;
            const shifted = (@as(f64, v) - max_f64) * inv_t;
            if (shifted < -80.0 and idx != argmax_id) continue;
            scratch.cand_ids[count] = @intCast(idx);
            count += 1;
        }
        if (params.top_k > 0 and params.top_k < count) {
            const k_lim: usize = @intCast(params.top_k);
            quickselectTopK(scratch.cand_ids[0..count], work_logits[0..vocab_size], k_lim);
            count = k_lim;
        }
        active_ids = scratch.cand_ids[0..count];
    }

    if (active_ids.len == 0) {
        stack_cands[0] = argmax_id;
        active_ids = stack_cands[0..1];
    }

    // Recompute chosen `logprob` once final token is drawn if `need_logprobs` is true:
    // keep a copy of `(chosen_logit - max) * inv_t - log_z` by storing `log_z` before mutating `work_logits`!
    var saved_log_z: f64 = 0.0;
    if (need_logprobs) {
        // Recover `log_z` directly from `argmax_id`: `argmax` had `shifted == 0`, so
        // if `top_logprobs_len > 0`, `top_logprobs[0].logprob == -log_z`; otherwise from `out.logprob == 0 - log_z`!
        saved_log_z = -@as(f64, out.logprob);
    }

    // Convert `active_ids` in-place inside `work_logits[id]` from raw logits to unnormalized `exp(shifted)`
    // (saving each candidate's shifted value if `need_logprobs` is active via `log(e)`!).
    var max_exp: f64 = 0.0;
    for (active_ids) |id| {
        const shifted = (@as(f64, work_logits[id]) - max_f64) * inv_t;
        const e: f64 = if (shifted < -80.0) 0.0 else @exp(shifted);
        work_logits[id] = @floatCast(e);
        if (e > max_exp) max_exp = e;
    }

    // Apply `min_p` relative to `max_exp`.
    if (params.min_p > 0.0 and max_exp > 0.0) {
        const thresh: f32 = @floatCast(@as(f64, params.min_p) * max_exp);
        var kept: usize = 0;
        for (active_ids) |id| {
            if (work_logits[id] >= thresh or id == argmax_id) {
                active_ids[kept] = id;
                kept += 1;
            }
        }
        active_ids = active_ids[0..kept];
    }

    // Normalize candidate probabilities and apply `top_p` nucleus truncation.
    var sum_p: f64 = 0.0;
    for (active_ids) |id| sum_p += @as(f64, work_logits[id]);

    if (!(sum_p > 0.0)) {
        out.token_id = argmax_id;
        return out;
    }

    if (params.top_p < 1.0 and active_ids.len > 1) {
        sortCandidatesDesc(active_ids, work_logits[0..vocab_size]);
        const target_mass: f64 = @as(f64, params.top_p) * sum_p;
        var cum: f64 = 0.0;
        var keep_n: usize = 0;
        for (active_ids) |id| {
            cum += @as(f64, work_logits[id]);
            keep_n += 1;
            if (cum >= target_mass) break;
        }
        active_ids = active_ids[0..@max(keep_n, 1)];
        sum_p = cum;
    }

    const rng = extractRandom(rng_source);
    const u: f64 = rng.float(f64) * sum_p;
    var acc: f64 = 0.0;
    var chosen: u32 = active_ids[active_ids.len - 1];
    for (active_ids) |id| {
        const p = @as(f64, work_logits[id]);
        if (p > 0.0) {
            acc += p;
            if (u < acc) {
                chosen = id;
                break;
            }
        }
    }

    out.token_id = chosen;
    if (need_logprobs) {
        // Since `work_logits[chosen]` now holds `exp((l_chosen - max_val) * inv_t)`,
        // `log(work_logits[chosen]) - saved_log_z` recovers exact `log_softmax[chosen]`!
        const e_chosen: f64 = @as(f64, work_logits[chosen]);
        out.logprob = if (e_chosen > 0.0) @floatCast(@log(e_chosen) - saved_log_z) else -std.math.inf(f32);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Zero-Allocation Streaming `StopHoldback` Ring
// ---------------------------------------------------------------------------

pub const StopConfig = struct {
    seqs: [MAX_STOP_SEQUENCES][MAX_STOP_BYTES]u8 = undefined,
    lens: [MAX_STOP_SEQUENCES]u8 = @splat(0),
    count: u8 = 0,
    /// Precomputed bitset of first bytes across all registered stop sequences for
    /// zero-delay O(1) span pass-through when `hold_len == 0`.
    first_bytes: grammar.ByteSet = grammar.ByteSet.empty,

    pub fn init() StopConfig {
        return .{};
    }

    pub fn reset(self: *StopConfig) void {
        self.count = 0;
        self.lens = @splat(0);
        self.first_bytes = grammar.ByteSet.empty;
    }

    pub fn add(self: *StopConfig, seq: []const u8) StopConfigError!void {
        if (seq.len == 0) return error.EmptyStopSequence;
        if (seq.len > MAX_STOP_BYTES) return error.StopSequenceTooLong;
        // Deduplicate identical stop strings silently.
        for (0..self.count) |i| {
            if (std.mem.eql(u8, self.sliceAt(i), seq)) return;
        }
        if (self.count >= MAX_STOP_SEQUENCES) return error.TooManyStopSequences;
        const idx = self.count;
        @memcpy(self.seqs[idx][0..seq.len], seq);
        self.lens[idx] = @intCast(seq.len);
        self.first_bytes.insert(seq[0]);
        self.count += 1;
    }

    pub inline fn sliceAt(self: *const StopConfig, idx: usize) []const u8 {
        return self.seqs[idx][0..self.lens[idx]];
    }
};

/// Streaming stop-sequence prefix holdback buffer:
/// - `0` delay on non-matching spans (`cfg.count == 0` or bytes outside `first_bytes` when `hold_len == 0`).
/// - Buffers ONLY the longest suffix of recent bytes that is a proper prefix of some stop sequence.
/// - Never leaks a matched stop sequence or any bytes following it.
pub const StopHoldback = struct {
    hold: [MAX_STOP_BYTES]u8 = undefined,
    hold_len: u8 = 0,
    stopped: bool = false,
    matched_index: ?u8 = null,

    pub fn reset(self: *StopHoldback) void {
        self.hold_len = 0;
        self.stopped = false;
        self.matched_index = null;
    }

    pub fn matchedSequence(self: *const StopHoldback, cfg: *const StopConfig) ?[]const u8 {
        const idx = self.matched_index orelse return null;
        if (idx >= cfg.count) return null;
        return cfg.sliceAt(idx);
    }

    /// Feeds `chunk` into the holdback state machine, invoking `emit_fn(ctx, safe_bytes)`
    /// for all disambiguated non-stop bytes.
    /// Returns `true` to continue decoding, or `false` as soon as a stop sequence matches.
    pub fn feed(
        self: *StopHoldback,
        cfg: *const StopConfig,
        chunk: []const u8,
        ctx: anytype,
        comptime emit_fn: anytype,
    ) !bool {
        if (self.stopped) return false;
        if (chunk.len == 0) return true;

        // Strict zero-overhead bypass when no stop sequences are configured!
        if (cfg.count == 0) {
            try emit_fn(ctx, chunk);
            return true;
        }

        var i: usize = 0;
        while (i < chunk.len) {
            // Zero-delay fast span pass-through when `hold_len == 0`!
            if (self.hold_len == 0) {
                const span_start = i;
                while (i < chunk.len and !cfg.first_bytes.contains(chunk[i])) : (i += 1) {}
                if (i > span_start) {
                    try emit_fn(ctx, chunk[span_start..i]);
                }
                if (i >= chunk.len) return true;
            }

            // Append `chunk[i]` to `cand` (`MAX_STOP_BYTES + 1` stack buffer so `hold` can reach max len).
            var cand_buf: [MAX_STOP_BYTES + 1]u8 = undefined;
            const cur_len: usize = @as(usize, self.hold_len) + 1;
            if (self.hold_len > 0) {
                @memcpy(cand_buf[0..self.hold_len], self.hold[0..self.hold_len]);
            }
            cand_buf[self.hold_len] = chunk[i];
            i += 1;
            const cand = cand_buf[0..cur_len];

            // Scan from earliest offset `p = 0 .. cur_len` for:
            //   (a) an exact stop sequence match `cand[p..] == S_k` (earliest `p` wins!), or
            //   (b) the earliest `p` where `cand[p..]` is a proper prefix of at least one `S_k`.
            var keep_from: usize = cur_len;
            for (0..cur_len) |p| {
                const suf = cand[p..];
                var exact_match: ?u8 = null;
                var is_proper_prefix = false;

                for (0..cfg.count) |k| {
                    const s = cfg.sliceAt(k);
                    if (suf.len == s.len) {
                        if (std.mem.eql(u8, suf, s)) {
                            exact_match = @intCast(k);
                            break;
                        }
                    } else if (suf.len < s.len) {
                        if (std.mem.eql(u8, suf, s[0..suf.len])) {
                            is_proper_prefix = true;
                        }
                    }
                }

                if (exact_match) |matched_k| {
                    if (p > 0) {
                        try emit_fn(ctx, cand[0..p]);
                    }
                    self.hold_len = 0;
                    self.stopped = true;
                    self.matched_index = matched_k;
                    return false;
                }
                if (is_proper_prefix) {
                    keep_from = p;
                    break;
                }
            }

            if (keep_from > 0) {
                try emit_fn(ctx, cand[0..keep_from]);
            }
            const rem = cur_len - keep_from;
            if (rem > 0) {
                @memcpy(self.hold[0..rem], cand[keep_from..cur_len]);
            }
            self.hold_len = @intCast(rem);
        }

        return true;
    }

    /// Flushes any trailing partial prefix at normal end-of-stream (EOS / `max_tokens`).
    pub fn flush(
        self: *StopHoldback,
        ctx: anytype,
        comptime emit_fn: anytype,
    ) !void {
        if (self.stopped or self.hold_len == 0) return;
        const n = self.hold_len;
        self.hold_len = 0;
        try emit_fn(ctx, self.hold[0..n]);
    }
};

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

const MallocStatistics = extern struct {
    blocks_in_use: c_uint,
    size_in_use: usize,
    max_size_in_use: usize,
    size_allocated: usize,
};
extern "c" fn malloc_zone_statistics(zone: ?*anyopaque, stats: *MallocStatistics) void;

test "sampler unconstrained greedy fast-path bypasses scratch and RNG with zero allocations" {
    const logits_f16 = [_]f16{ -1.5, 3.25, 7.5, 2.0, -4.0 };
    var prng = std.Random.DefaultPrng.init(12345);
    const before_state = prng.s;

    var m_before: MallocStatistics = undefined;
    var m_after: MallocStatistics = undefined;
    malloc_zone_statistics(null, &m_before);

    // Empty scratch proves fast-path touches zero scratch buffers!
    const out = try sampleToken(f16, &logits_f16, .{}, &.{}, &prng, .{});
    malloc_zone_statistics(null, &m_after);

    try std.testing.expectEqual(@as(u32, 2), out.token_id);
    try std.testing.expectEqual(before_state, prng.s); // Zero RNG draws!
    try std.testing.expectEqual(m_before.size_allocated, m_after.size_allocated);
}

test "sampler enforces C22 bitmask, penalties, logit_bias, top_k, min_p, top_p, and exact logprobs" {
    const logits = [_]f32{ 5.0, 4.0, 3.0, 2.0, 1.0, 0.0 };
    var scratch_logits: [6]f32 = undefined;
    var scratch_cands: [6]u32 = undefined;
    const scratch = SamplerScratch{
        .logits_f32 = &scratch_logits,
        .cand_ids = &scratch_cands,
    };

    // 1. C22 bitmask masks out tokens 0 and 1 -> greedy selects token 2!
    var mask: [1]u32 = .{0};
    grammar.setTokenAllowed(&mask, 2);
    grammar.setTokenAllowed(&mask, 3);
    grammar.setTokenAllowed(&mask, 5);

    var prng = std.Random.DefaultPrng.init(42);
    const masked_out = try sampleToken(f32, &logits, .{
        .temperature = 0.0,
        .allowed_token_mask = &mask,
        .logprobs = true,
        .top_logprobs = 2,
    }, &.{}, &prng, scratch);
    try std.testing.expectEqual(@as(u32, 2), masked_out.token_id);
    try std.testing.expectEqual(@as(u8, 2), masked_out.top_logprobs_len);
    try std.testing.expectEqual(@as(u32, 2), masked_out.top_logprobs[0].token_id);
    try std.testing.expectEqual(@as(u32, 3), masked_out.top_logprobs[1].token_id);
    try std.testing.expect(masked_out.logprob < 0.0 and masked_out.logprob > -0.4);

    // 2. Sparse PenaltyTracker + logit_bias suppress token 0 and boost token 4!
    var tracker = DefaultPenaltyTracker.init();
    tracker.recordSlice(&[_]u32{ 0, 0, 0 }); // token 0 seen 3x
    const biased_out = try sampleToken(f32, &logits, .{
        .temperature = 0.0,
        .presence_penalty = 1.5,
        .frequency_penalty = 1.0, // token 0 drops from 5.0 -> 5.0 - 4.5 = 0.5
        .logit_bias = &[_]LogitBiasEntry{.{ .token_id = 4, .bias = 4.5 }}, // token 4 rises 1.0 -> 5.5
    }, tracker.activeEntries(), &prng, scratch);
    try std.testing.expectEqual(@as(u32, 4), biased_out.token_id);

    // 3. Stochastic top_k = 2 & top_p = 0.60 never draws tail tokens 2..5!
    for (0..500) |_| {
        const s_out = try sampleToken(f32, &logits, .{
            .temperature = 1.0,
            .top_k = 2,
            .top_p = 0.80,
            .min_p = 0.10,
            .logprobs = true,
        }, &.{}, &prng, scratch);
        try std.testing.expect(s_out.token_id == 0 or s_out.token_id == 1);
        try std.testing.expect(std.math.isFinite(s_out.logprob) and s_out.logprob <= 0.0);
    }
}

test "StopHoldback emits non-matching bytes with zero delay, resolves overlaps, and never leaks matched stop" {
    const Collector = struct {
        buf: [256]u8 = undefined,
        len: usize = 0,
        calls: usize = 0,

        fn reset(self: *@This()) void {
            self.len = 0;
            self.calls = 0;
        }
        fn emit(self: *@This(), bytes: []const u8) error{Overflow}!void {
            if (self.len + bytes.len > self.buf.len) return error.Overflow;
            @memcpy(self.buf[self.len .. self.len + bytes.len], bytes);
            self.len += bytes.len;
            self.calls += 1;
        }
        fn slice(self: *const @This()) []const u8 {
            return self.buf[0..self.len];
        }
    };

    var cfg = StopConfig.init();
    try cfg.add("</tool>");
    try cfg.add("aab");

    var hold = StopHoldback{};
    var col = Collector{};

    // 1. Non-matching bytes emit immediately on chunk 1 (`0` delay!), while partial prefix `</t` is held!
    try std.testing.expect(try hold.feed(&cfg, "hello world</t", &col, Collector.emit));
    try std.testing.expectEqualStrings("hello world", col.slice());
    try std.testing.expectEqual(@as(u8, 3), hold.hold_len);

    // 2. Completing `ool>TRAILING` triggers stop, never leaks `</tool>` or `TRAILING`!
    try std.testing.expect(!try hold.feed(&cfg, "ool>TRAILING_GARBAGE", &col, Collector.emit));
    try std.testing.expect(hold.stopped);
    try std.testing.expectEqualStrings("</tool>", hold.matchedSequence(&cfg).?);
    try std.testing.expectEqualStrings("hello world", col.slice());

    // 3. Overlapping prefix `"aaab"` with stop `"aab"` emits `"a"` and suppresses `"aab"`!
    hold.reset();
    col.reset();
    for ("aaabEXTRA") |ch| {
        const one = [1]u8{ch};
        if (!try hold.feed(&cfg, &one, &col, Collector.emit)) break;
    }
    try std.testing.expect(hold.stopped);
    try std.testing.expectEqualStrings("aab", hold.matchedSequence(&cfg).?);
    try std.testing.expectEqualStrings("a", col.slice());

    // 4. False prefix at EOF flushes cleanly!
    hold.reset();
    col.reset();
    try std.testing.expect(try hold.feed(&cfg, "ok</to", &col, Collector.emit));
    try std.testing.expectEqualStrings("ok", col.slice());
    try hold.flush(&col, Collector.emit);
    try std.testing.expectEqualStrings("ok</to", col.slice());
}

test "F20.2 Appendix A.6 repetition_penalty validation and finite logit clamping prevent NaN logprobs" {
    // 1. Appendix A.6 repro: repetition_penalty < 1e-4 rejected explicitly by validate().
    const bad_params = SamplingParams{
        .temperature = 1.0,
        .repetition_penalty = 1e-38,
        .logprobs = true,
    };
    try std.testing.expectError(error.InvalidRepetitionPenalty, bad_params.validate());

    var logits = [_]f32{ 4.0, 1.0, 0.5, -1.0 };
    var work: [4]f32 = undefined;
    var cands: [4]u32 = undefined;
    var tracker = DefaultPenaltyTracker.init();
    tracker.recordToken(0);
    var prng = std.Random.DefaultPrng.init(99);

    try std.testing.expectError(
        error.InvalidRepetitionPenalty,
        sampleToken(f32, &logits, bad_params, tracker.activeEntries(), &prng, .{
            .logits_f32 = &work,
            .cand_ids = &cands,
        }),
    );

    // 2. Minimum valid repetition_penalty = 1e-4 + extreme positive logits + huge logit_bias clamp cleanly to finite f32 (never +Inf / NaN).
    const extreme_logits = [_]f32{ 3.0e38, 1.0, 0.5, -3.0e38 };
    const out = try sampleToken(f32, &extreme_logits, .{
        .temperature = 1.0,
        .repetition_penalty = 1e-4,
        .logit_bias = &.{.{ .token_id = 0, .bias = 2.0e38 }},
        .logprobs = true,
        .top_logprobs = 2,
    }, tracker.activeEntries(), &prng, .{
        .logits_f32 = &work,
        .cand_ids = &cands,
    });
    try std.testing.expect(!std.math.isNan(out.logprob));
    try std.testing.expect(std.math.isFinite(out.logprob));
    for (out.top_logprobs[0..out.top_logprobs_len]) |tp| {
        try std.testing.expect(!std.math.isNan(tp.logprob));
        try std.testing.expect(std.math.isFinite(tp.logprob));
    }
}
