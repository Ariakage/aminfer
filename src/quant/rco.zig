//! Native multi-threaded Weighted OLS Residual Compensation Optimization (RCO)
//! affine quantizer (`ADR-0017`, `gsq-convert` acceleration).
//!
//! Mirrors the exact math and IEEE-754 float16 rounding of
//! `tools/src/aminfer_tools/gsq_rco.py`:
//! - Exact `f64` accumulators and IEEE-754 binary16 clamp/round (`[-65504, 65504]`).
//! - Clipped range initializers (`2..6` bits) and high-activation bounding box
//!   (`w_thresh = 0.05 * max(w_norm)`) for outlier-resistant weighted quantization.
//! - Alternating closed-form ridge-stabilized OLS (`lam = 1e-12 * max(1, sum_q2)`)
//!   with final bias-residual compensation and unweighted fallback guard.
//! - Zero dynamic heap allocation in the hot path (stack-only group buffers,
//!   up to `MAX_GROUP_SIZE = 128`, and up to `MAX_THREADS = 16` row workers).

const std = @import("std");
const spec = @import("spec.zig");
const formats = @import("formats.zig");

pub const AffineQuantBits = formats.AffineQuantBits;
pub const MAX_GROUP_SIZE: usize = 128;
pub const MAX_THREADS: usize = 16;
pub const MIN_ROWS_FOR_PARALLEL: usize = 32;
const F16_MAX: f64 = 65504.0;

pub const SrcDType = enum(u32) {
    f32 = 0,
    f16 = 1,
    bf16 = 2,

    pub fn fromU32(raw: u32) Error!SrcDType {
        return switch (raw) {
            0 => .f32,
            1 => .f16,
            2 => .bf16,
            else => error.InvalidSrcDType,
        };
    }

    pub fn elementBytes(self: SrcDType) usize {
        return switch (self) {
            .f32 => 4,
            .f16, .bf16 => 2,
        };
    }
};

pub const Error = error{
    InvalidSrcDType,
    InvalidBits,
    InvalidGroupSize,
    ZeroDimension,
    UnalignedK,
    NullBuffer,
    Overflow,
    ThreadSpawnFailed,
};

pub fn bitsFromU32(raw: u32) Error!AffineQuantBits {
    return switch (raw) {
        2 => .q2,
        3 => .q3,
        4 => .q4,
        5 => .q5,
        6 => .q6,
        8 => .q8,
        else => error.InvalidBits,
    };
}

pub const Params = struct {
    src_dtype: SrcDType,
    n_rows: usize,
    k: usize,
    bits: AffineQuantBits,
    group_size: u16,
    use_rco: bool,
    rco_iters: usize,
    col_weights: ?[]align(1) const f32,
    num_threads: usize,
};

const SB = struct {
    s: f64,
    b: f64,
};

const ClippedStarts = struct {
    items: [2]SB = undefined,
    len: usize = 0,

    fn append(self: *ClippedStarts, sb: SB) void {
        if (self.len < self.items.len) {
            self.items[self.len] = sb;
            self.len += 1;
        }
    }
};

const WeightedStarts = struct {
    items: [7]SB = undefined,
    len: usize = 0,

    fn addUnique(self: *WeightedStarts, sb: SB) void {
        if (!(sb.s > 0.0)) return;
        for (self.items[0..self.len]) |existing| {
            if (existing.s == sb.s and existing.b == sb.b) return;
        }
        if (self.len < self.items.len) {
            self.items[self.len] = sb;
            self.len += 1;
        }
    }
};

const EvalUnweighted = struct {
    mse: f64,
    sum_q: u64,
    sum_q2: u64,
    sum_wq: f64,
};

const EvalWeighted = struct {
    wmse: f64,
    sum_wq: f64,
    sum_wq2: f64,
    sum_wxq: f64,
};

const GroupResult = struct {
    s: f64,
    b: f64,
    mse: f64,
};

/// Clamps `val` to finite IEEE-754 binary16 range `[-65504, 65504]` (flushing
/// non-finite values to `0.0`) and rounds to nearest-even `f16` value.
pub inline fn clampF16(val: f64) f64 {
    if (!std.math.isFinite(val)) return 0.0;
    const clamped = std.math.clamp(val, -F16_MAX, F16_MAX);
    const h: f16 = @floatCast(clamped);
    return @floatCast(h);
}

/// Clamps and rounds `(scale, bias)` to exact IEEE-754 binary16 values,
/// forcing negative or non-finite scale to `0.0` and non-finite bias to `0.0`.
pub inline fn roundPairF16(s_in: f64, b_in: f64) SB {
    var s = s_in;
    if (!std.math.isFinite(s) or s < 0.0) {
        s = 0.0;
    } else if (s > F16_MAX) {
        s = F16_MAX;
    }
    var b = b_in;
    if (!std.math.isFinite(b)) {
        b = 0.0;
    } else if (b > F16_MAX) {
        b = F16_MAX;
    } else if (b < -F16_MAX) {
        b = -F16_MAX;
    }
    const sh: f16 = @floatCast(s);
    const bh: f16 = @floatCast(b);
    return .{
        .s = @floatCast(sh),
        .b = @floatCast(bh),
    };
}

inline fn quantizeCode(x: f64, b_h: f64, inv_s: f64, qmax_f: f64) u8 {
    const v = (x - b_h) * inv_s;
    // Clamp into [0.0, qmax_f] (NaN falls into else -> 0.0) so `v + 0.5` is
    // strictly in [0.5, 255.5] and `@intFromFloat` matches Python `int(v + 0.5)`.
    const clamped_v = if (v > 0.0) (if (v < qmax_f) v else qmax_f) else 0.0;
    return @intFromFloat(clamped_v + 0.5);
}

fn evalGroupWithSb(
    comptime gs: usize,
    values: *const [gs]f64,
    s_h: f64,
    b_h: f64,
    qmax: u8,
    codes: *[gs]u8,
) EvalUnweighted {
    const inv_n: f64 = 1.0 / @as(f64, @floatFromInt(gs));
    if (s_h <= 0.0) {
        @memset(codes, 0);
        var err2: f64 = 0.0;
        for (0..gs) |i| {
            const d = values[i] - b_h;
            err2 += d * d;
        }
        return .{ .mse = err2 * inv_n, .sum_q = 0, .sum_q2 = 0, .sum_wq = 0.0 };
    }

    const inv_s = 1.0 / s_h;
    const qmax_f: f64 = @floatFromInt(qmax);
    var sum_q: u64 = 0;
    var sum_q2: u64 = 0;
    var sum_wq: f64 = 0.0;
    var err2: f64 = 0.0;

    for (0..gs) |i| {
        const x = values[i];
        const q = quantizeCode(x, b_h, inv_s, qmax_f);
        codes[i] = q;
        const q_u64: u64 = q;
        const q_f: f64 = @floatFromInt(q);
        sum_q += q_u64;
        sum_q2 += q_u64 * q_u64;
        sum_wq += x * q_f;
        const d = x - (b_h + q_f * s_h);
        err2 += d * d;
    }

    return .{
        .mse = err2 * inv_n,
        .sum_q = sum_q,
        .sum_q2 = sum_q2,
        .sum_wq = sum_wq,
    };
}

fn evalGroupWithSbWeighted(
    comptime gs: usize,
    values: *const [gs]f64,
    w_norm: *const [gs]f64,
    s_h: f64,
    b_h: f64,
    qmax: u8,
    codes: *[gs]u8,
) EvalWeighted {
    const inv_n: f64 = 1.0 / @as(f64, @floatFromInt(gs));
    if (s_h <= 0.0) {
        @memset(codes, 0);
        var w_err2: f64 = 0.0;
        for (0..gs) |i| {
            const d = values[i] - b_h;
            w_err2 += w_norm[i] * d * d;
        }
        return .{ .wmse = w_err2 * inv_n, .sum_wq = 0.0, .sum_wq2 = 0.0, .sum_wxq = 0.0 };
    }

    const inv_s = 1.0 / s_h;
    const qmax_f: f64 = @floatFromInt(qmax);
    var sum_wq: f64 = 0.0;
    var sum_wq2: f64 = 0.0;
    var sum_wxq: f64 = 0.0;
    var w_err2: f64 = 0.0;

    for (0..gs) |i| {
        const x = values[i];
        const wi = w_norm[i];
        const q = quantizeCode(x, b_h, inv_s, qmax_f);
        codes[i] = q;
        const q_f: f64 = @floatFromInt(q);
        const wq = wi * q_f;
        sum_wq += wq;
        sum_wq2 += wq * q_f;
        sum_wxq += wq * x;
        const d = x - (b_h + q_f * s_h);
        w_err2 += wi * d * d;
    }

    return .{
        .wmse = w_err2 * inv_n,
        .sum_wq = sum_wq,
        .sum_wq2 = sum_wq2,
        .sum_wxq = sum_wxq,
    };
}

/// Normalizes a group's activation-variance weights so `sum(w_norm) == gs`.
/// Returns `false` when weights are uniform (`w_max <= w_min * (1 + 1e-12)`),
/// signaling fallback to unweighted quantization.
fn normalizeGroupWeights(
    comptime gs: usize,
    weights: [*]align(1) const f32,
    out_w_norm: *[gs]f64,
) bool {
    var w_raw: [gs]f64 = undefined;
    var w_min: f64 = std.math.inf(f64);
    var w_max: f64 = 0.0;

    for (0..gs) |i| {
        const wf: f64 = @floatCast(weights[i]);
        const w = if (std.math.isFinite(wf) and wf > 1e-12) wf else 1e-12;
        w_raw[i] = w;
        if (w < w_min) w_min = w;
        if (w > w_max) w_max = w;
    }
    if (w_max <= w_min * (1.0 + 1e-12)) return false;

    var sum_scaled: f64 = 0.0;
    for (0..gs) |i| {
        const ws = w_raw[i] / w_max;
        out_w_norm[i] = ws;
        sum_scaled += ws;
    }
    const scale = @as(f64, @floatFromInt(gs)) / sum_scaled;
    for (0..gs) |i| {
        out_w_norm[i] *= scale;
    }
    return true;
}

fn clippedInitializers(
    bits: AffineQuantBits,
    qmax_f: f64,
    lo: f64,
    hi: f64,
    mean: f64,
    std_val: f64,
    s_mm: f64,
    b_mm: f64,
) ClippedStarts {
    var out: ClippedStarts = .{};
    if (!(std_val > 0.0) or !(hi > lo)) return out;
    const span = hi - lo;

    switch (bits) {
        .q2 => {
            for ([_]f64{ 1.75, 2.25 }) |mult| {
                const c_lo = @max(lo, mean - mult * std_val);
                const c_hi = @min(hi, mean + mult * std_val);
                if (c_hi > c_lo) {
                    const sb = roundPairF16((c_hi - c_lo) / qmax_f, c_lo);
                    if (sb.s > 0.0 and (sb.s != s_mm or sb.b != b_mm)) out.append(sb);
                }
            }
        },
        .q3 => {
            if (span > 4.2 * std_val) {
                for ([_]f64{ 2.35, 2.85 }) |mult| {
                    const c_lo = @max(lo, mean - mult * std_val);
                    const c_hi = @min(hi, mean + mult * std_val);
                    if (c_hi > c_lo) {
                        const sb = roundPairF16((c_hi - c_lo) / qmax_f, c_lo);
                        if (sb.s > 0.0 and (sb.s != s_mm or sb.b != b_mm)) out.append(sb);
                    }
                }
            }
        },
        .q4 => {
            if (span > 4.8 * std_val) {
                const mid = 0.5 * (lo + hi);
                for ([_]f64{ 0.84, 0.92 }) |alpha| {
                    const half = 0.5 * span * alpha;
                    const sb = roundPairF16((2.0 * half) / qmax_f, mid - half);
                    if (sb.s > 0.0 and (sb.s != s_mm or sb.b != b_mm)) out.append(sb);
                }
            }
        },
        .q5, .q6 => {
            if (span > 5.5 * std_val) {
                const mid = 0.5 * (lo + hi);
                for ([_]f64{ 0.90, 0.96 }) |alpha| {
                    const half = 0.5 * span * alpha;
                    const sb = roundPairF16((2.0 * half) / qmax_f, mid - half);
                    if (sb.s > 0.0 and (sb.s != s_mm or sb.b != b_mm)) out.append(sb);
                }
            }
        },
        .q8 => {},
    }
    return out;
}

const GroupStats = struct {
    lo: f64,
    hi: f64,
    sum_w: f64,
    sum_w2: f64,
};

fn computeGroupStats(comptime gs: usize, values: *const [gs]f64) GroupStats {
    var lo: f64 = values[0];
    var hi: f64 = values[0];
    var sum_w: f64 = values[0];
    var sum_w2: f64 = values[0] * values[0];
    for (1..gs) |i| {
        const x = values[i];
        if (x < lo) lo = x;
        if (x > hi) hi = x;
        sum_w += x;
        sum_w2 += x * x;
    }
    return .{ .lo = lo, .hi = hi, .sum_w = sum_w, .sum_w2 = sum_w2 };
}

fn quantizeGroupMinmaxFixed(
    comptime gs: usize,
    values: *const [gs]f64,
    qmax: u8,
    codes: *[gs]u8,
) GroupResult {
    const stats = computeGroupStats(gs, values);
    const inv_n: f64 = 1.0 / @as(f64, @floatFromInt(gs));
    const qmax_f: f64 = @floatFromInt(qmax);

    if (stats.hi <= stats.lo) {
        const b_h = clampF16(stats.lo);
        const ev = evalGroupWithSb(gs, values, 0.0, b_h, qmax, codes);
        return .{ .s = 0.0, .b = b_h, .mse = ev.mse };
    }

    const sb = roundPairF16((stats.hi - stats.lo) / qmax_f, stats.lo);
    if (sb.s <= 0.0) {
        const b_h = clampF16(stats.sum_w * inv_n);
        const ev = evalGroupWithSb(gs, values, 0.0, b_h, qmax, codes);
        return .{ .s = 0.0, .b = b_h, .mse = ev.mse };
    }

    const ev = evalGroupWithSb(gs, values, sb.s, sb.b, qmax, codes);
    return .{ .s = sb.s, .b = sb.b, .mse = ev.mse };
}

fn quantizeGroupRcoUnweightedFixed(
    comptime gs: usize,
    values: *const [gs]f64,
    bits: AffineQuantBits,
    qmax: u8,
    iters: usize,
    best_codes: *[gs]u8,
) GroupResult {
    const stats = computeGroupStats(gs, values);
    const n_f: f64 = @floatFromInt(gs);
    const inv_n: f64 = 1.0 / n_f;
    const qmax_f: f64 = @floatFromInt(qmax);

    if (stats.hi <= stats.lo) {
        const b_h = clampF16(stats.lo);
        const ev = evalGroupWithSb(gs, values, 0.0, b_h, qmax, best_codes);
        return .{ .s = 0.0, .b = b_h, .mse = ev.mse };
    }

    const mm = roundPairF16((stats.hi - stats.lo) / qmax_f, stats.lo);
    if (mm.s <= 0.0) {
        const b_mean = clampF16(stats.sum_w * inv_n);
        const ev = evalGroupWithSb(gs, values, 0.0, b_mean, qmax, best_codes);
        return .{ .s = 0.0, .b = b_mean, .mse = ev.mse };
    }

    const mm_eval = evalGroupWithSb(gs, values, mm.s, mm.b, qmax, best_codes);
    var best_s = mm.s;
    var best_b = mm.b;
    var best_mse = mm_eval.mse;
    if (best_mse == 0.0 or iters == 0) {
        return .{ .s = best_s, .b = best_b, .mse = best_mse };
    }

    const mean = stats.sum_w * inv_n;
    const std_val = @sqrt(@max(0.0, stats.sum_w2 * inv_n - mean * mean));
    const clipped = clippedInitializers(bits, qmax_f, stats.lo, stats.hi, mean, std_val, mm.s, mm.b);

    var cur_codes: [gs]u8 = undefined;
    var new_codes: [gs]u8 = undefined;

    const total_starts: usize = 1 + clipped.len;
    for (0..total_starts) |start_idx| {
        var s_cur: f64 = undefined;
        var b_cur: f64 = undefined;
        var sum_q: u64 = undefined;
        var sum_q2: u64 = undefined;
        var sum_wq: f64 = undefined;

        if (start_idx == 0) {
            s_cur = mm.s;
            b_cur = mm.b;
            sum_q = mm_eval.sum_q;
            sum_q2 = mm_eval.sum_q2;
            sum_wq = mm_eval.sum_wq;
            @memcpy(&cur_codes, best_codes);
        } else {
            const init_sb = clipped.items[start_idx - 1];
            s_cur = init_sb.s;
            b_cur = init_sb.b;
            const ev = evalGroupWithSb(gs, values, s_cur, b_cur, qmax, &cur_codes);
            sum_q = ev.sum_q;
            sum_q2 = ev.sum_q2;
            sum_wq = ev.sum_wq;
            if (ev.mse < best_mse) {
                best_mse = ev.mse;
                best_s = s_cur;
                best_b = b_cur;
                @memcpy(best_codes, &cur_codes);
            }
        }

        for (0..iters) |_| {
            const denom_i: i64 = @as(i64, @intCast(gs)) * @as(i64, @intCast(sum_q2)) -
                @as(i64, @intCast(sum_q)) * @as(i64, @intCast(sum_q));
            if (denom_i <= 0) break;
            const denom: f64 = @floatFromInt(denom_i);
            const lam = 1e-12 * @max(1.0, @as(f64, @floatFromInt(sum_q2)));
            const sum_q_f: f64 = @floatFromInt(sum_q);
            const s_new = (n_f * sum_wq - sum_q_f * stats.sum_w + lam * s_cur) / (denom + lam);
            if (!(s_new > 0.0)) break;
            const b_new = (stats.sum_w - s_new * sum_q_f) * inv_n;
            const sb_h = roundPairF16(s_new, b_new);
            if (sb_h.s <= 0.0 or (sb_h.s == s_cur and sb_h.b == b_cur)) break;

            const new_ev = evalGroupWithSb(gs, values, sb_h.s, sb_h.b, qmax, &new_codes);
            if (new_ev.mse < best_mse) {
                best_mse = new_ev.mse;
                best_s = sb_h.s;
                best_b = sb_h.b;
                @memcpy(best_codes, &new_codes);
            }
            const same_codes = std.mem.eql(u8, &new_codes, &cur_codes);
            s_cur = sb_h.s;
            b_cur = sb_h.b;
            sum_q = new_ev.sum_q;
            sum_q2 = new_ev.sum_q2;
            sum_wq = new_ev.sum_wq;
            @memcpy(&cur_codes, &new_codes);
            if (same_codes) break;
        }
    }

    if (best_s > 0.0) {
        var sum_q_best: u64 = 0;
        for (best_codes) |q| sum_q_best += q;
        const b_opt = clampF16((stats.sum_w - best_s * @as(f64, @floatFromInt(sum_q_best))) * inv_n);
        if (b_opt != best_b) {
            const opt_ev = evalGroupWithSb(gs, values, best_s, b_opt, qmax, &cur_codes);
            if (opt_ev.mse < best_mse) {
                best_mse = opt_ev.mse;
                best_b = b_opt;
                @memcpy(best_codes, &cur_codes);
            }
        }
    }

    return .{ .s = best_s, .b = best_b, .mse = best_mse };
}

fn quantizeGroupRcoWeightedFixed(
    comptime gs: usize,
    values: *const [gs]f64,
    w_norm: *const [gs]f64,
    bits: AffineQuantBits,
    qmax: u8,
    iters: usize,
    best_codes: *[gs]u8,
) GroupResult {
    const stats = computeGroupStats(gs, values);
    const n_f: f64 = @floatFromInt(gs);
    const inv_n: f64 = 1.0 / n_f;
    const qmax_f: f64 = @floatFromInt(qmax);

    var sum_wx: f64 = 0.0;
    var sum_wx2: f64 = 0.0;
    var w_norm_max: f64 = 0.0;
    for (0..gs) |i| {
        const x = values[i];
        const wi = w_norm[i];
        sum_wx += wi * x;
        sum_wx2 += wi * x * x;
        if (wi > w_norm_max) w_norm_max = wi;
    }

    if (stats.hi <= stats.lo) {
        const b_h = clampF16(stats.lo);
        const ev = evalGroupWithSbWeighted(gs, values, w_norm, 0.0, b_h, qmax, best_codes);
        return .{ .s = 0.0, .b = b_h, .mse = ev.wmse };
    }

    const mm = roundPairF16((stats.hi - stats.lo) / qmax_f, stats.lo);
    if (mm.s <= 0.0) {
        const b_mean = clampF16(sum_wx * inv_n);
        const ev = evalGroupWithSbWeighted(gs, values, w_norm, 0.0, b_mean, qmax, best_codes);
        return .{ .s = 0.0, .b = b_mean, .mse = ev.wmse };
    }

    const mm_eval = evalGroupWithSbWeighted(gs, values, w_norm, mm.s, mm.b, qmax, best_codes);
    const minmax_wmse = mm_eval.wmse;
    var best_wmse = mm_eval.wmse;
    var best_s = mm.s;
    var best_b = mm.b;
    if (best_wmse == 0.0 or iters == 0) {
        return .{ .s = best_s, .b = best_b, .mse = best_wmse };
    }

    var starts_w: WeightedStarts = .{};
    starts_w.addUnique(mm);

    // Tight bounding box over high-activation channels so low-weight outliers never hijack range.
    const w_thresh = 0.05 * w_norm_max;
    var has_imp = false;
    var lo_imp: f64 = 0.0;
    var hi_imp: f64 = 0.0;
    for (0..gs) |i| {
        if (w_norm[i] >= w_thresh) {
            const x = values[i];
            if (!has_imp) {
                lo_imp = x;
                hi_imp = x;
                has_imp = true;
            } else {
                if (x < lo_imp) lo_imp = x;
                if (x > hi_imp) hi_imp = x;
            }
        }
    }
    if (has_imp and hi_imp > lo_imp) {
        starts_w.addUnique(roundPairF16((hi_imp - lo_imp) / qmax_f, lo_imp));
    }

    const w_mean = sum_wx * inv_n;
    const w_std = @sqrt(@max(0.0, sum_wx2 * inv_n - w_mean * w_mean));
    if (w_std > 0.0) {
        const mults: []const f64 = switch (bits) {
            .q2 => &.{ 1.5, 2.0, 2.5 },
            .q3 => &.{ 2.0, 2.6 },
            else => &.{ 2.5, 3.2 },
        };
        for (mults) |mult| {
            const c_lo = @max(stats.lo, w_mean - mult * w_std);
            const c_hi = @min(stats.hi, w_mean + mult * w_std);
            if (c_hi > c_lo) {
                starts_w.addUnique(roundPairF16((c_hi - c_lo) / qmax_f, c_lo));
            }
        }
    }

    const u_mean = stats.sum_w * inv_n;
    const u_std = @sqrt(@max(0.0, stats.sum_w2 * inv_n - u_mean * u_mean));
    const clipped = clippedInitializers(bits, qmax_f, stats.lo, stats.hi, u_mean, u_std, mm.s, mm.b);
    for (clipped.items[0..clipped.len]) |sb| starts_w.addUnique(sb);

    var cur_codes: [gs]u8 = undefined;
    var new_codes: [gs]u8 = undefined;

    for (0..starts_w.len) |start_idx| {
        var s_cur: f64 = undefined;
        var b_cur: f64 = undefined;
        var sum_wq: f64 = undefined;
        var sum_wq2: f64 = undefined;
        var sum_wxq: f64 = undefined;

        if (start_idx == 0) {
            s_cur = mm.s;
            b_cur = mm.b;
            sum_wq = mm_eval.sum_wq;
            sum_wq2 = mm_eval.sum_wq2;
            sum_wxq = mm_eval.sum_wxq;
            @memcpy(&cur_codes, best_codes);
        } else {
            const init_sb = starts_w.items[start_idx];
            s_cur = init_sb.s;
            b_cur = init_sb.b;
            const ev = evalGroupWithSbWeighted(gs, values, w_norm, s_cur, b_cur, qmax, &cur_codes);
            sum_wq = ev.sum_wq;
            sum_wq2 = ev.sum_wq2;
            sum_wxq = ev.sum_wxq;
            if (ev.wmse < best_wmse) {
                best_wmse = ev.wmse;
                best_s = s_cur;
                best_b = b_cur;
                @memcpy(best_codes, &cur_codes);
            }
        }

        for (0..iters) |_| {
            const denom = n_f * sum_wq2 - sum_wq * sum_wq;
            if (!(denom > 0.0)) break;
            const lam = 1e-12 * @max(1.0, sum_wq2);
            const s_new = (n_f * sum_wxq - sum_wq * sum_wx + lam * s_cur) / (denom + lam);
            if (!(s_new > 0.0)) break;
            const b_new = (sum_wx - s_new * sum_wq) * inv_n;
            const sb_h = roundPairF16(s_new, b_new);
            if (sb_h.s <= 0.0 or (sb_h.s == s_cur and sb_h.b == b_cur)) break;

            const new_ev = evalGroupWithSbWeighted(gs, values, w_norm, sb_h.s, sb_h.b, qmax, &new_codes);
            if (new_ev.wmse < best_wmse) {
                best_wmse = new_ev.wmse;
                best_s = sb_h.s;
                best_b = sb_h.b;
                @memcpy(best_codes, &new_codes);
            }
            const same_codes = std.mem.eql(u8, &new_codes, &cur_codes);
            s_cur = sb_h.s;
            b_cur = sb_h.b;
            sum_wq = new_ev.sum_wq;
            sum_wq2 = new_ev.sum_wq2;
            sum_wxq = new_ev.sum_wxq;
            @memcpy(&cur_codes, &new_codes);
            if (same_codes) break;
        }
    }

    if (best_s > 0.0) {
        var sum_wq_best: f64 = 0.0;
        for (0..gs) |i| {
            sum_wq_best += w_norm[i] * @as(f64, @floatFromInt(best_codes[i]));
        }
        const b_opt = clampF16((sum_wx - best_s * sum_wq_best) * inv_n);
        if (b_opt != best_b) {
            const opt_ev = evalGroupWithSbWeighted(gs, values, w_norm, best_s, b_opt, qmax, &cur_codes);
            if (opt_ev.wmse < best_wmse) {
                best_wmse = opt_ev.wmse;
                best_b = b_opt;
                @memcpy(best_codes, &cur_codes);
            }
        }
    }

    // Fallback check against unweighted RCO if weighted OLS failed to improve over min-max.
    if (best_wmse >= minmax_wmse) {
        var uw_codes: [gs]u8 = undefined;
        const uw = quantizeGroupRcoUnweightedFixed(gs, values, bits, qmax, iters, &uw_codes);
        const uw_eval = evalGroupWithSbWeighted(gs, values, w_norm, uw.s, uw.b, qmax, &uw_codes);
        if (uw_eval.wmse < best_wmse) {
            best_s = uw.s;
            best_b = uw.b;
            best_wmse = uw_eval.wmse;
            @memcpy(best_codes, &uw_codes);
        }
    }

    return .{ .s = best_s, .b = best_b, .mse = best_wmse };
}

inline fn loadGroupValues(
    comptime src_dtype: SrcDType,
    comptime gs: usize,
    src_ptr: [*]const u8,
    values: *[gs]f64,
) void {
    switch (src_dtype) {
        .f32 => {
            for (0..gs) |i| {
                const bits_u32 = std.mem.readInt(u32, src_ptr[i * 4 ..][0..4], .little);
                const v: f32 = @bitCast(bits_u32);
                values[i] = @floatCast(v);
            }
        },
        .f16 => {
            for (0..gs) |i| {
                const bits_u16 = std.mem.readInt(u16, src_ptr[i * 2 ..][0..2], .little);
                const v: f16 = @bitCast(bits_u16);
                values[i] = @floatCast(v);
            }
        },
        .bf16 => {
            for (0..gs) |i| {
                const bits_u16 = std.mem.readInt(u16, src_ptr[i * 2 ..][0..2], .little);
                values[i] = @floatCast(spec.bf16ToF32(bits_u16));
            }
        },
    }
}

fn processRowRangeFixed(
    comptime src_dtype: SrcDType,
    comptime bits: AffineQuantBits,
    comptime gs: u16,
    src_bytes: [*]const u8,
    dst_qweight: [*]u8,
    dst_params: [*]u8,
    col_weights: ?[*]align(1) const f32,
    row_start: usize,
    row_end: usize,
    k: usize,
    use_rco: bool,
    rco_iters: usize,
    out_mse_sum: *f64,
) void {
    const Ops = formats.AffineQuantOps(.{ .bits = bits, .group_size = gs });
    const gs_usize: usize = gs;
    const grp_q_bytes = Ops.groupWeightBytes();
    const grp_src_bytes = gs_usize * src_dtype.elementBytes();
    const groups = k / gs_usize;
    const row_src_bytes = groups * grp_src_bytes;
    const row_q_bytes = groups * grp_q_bytes;
    const row_p_bytes = groups * 4;
    const qmax: u8 = @intCast((@as(u16, 1) << @intCast(bits.bitWidth())) - 1);

    var values: [gs_usize]f64 = undefined;
    var w_norm: [gs_usize]f64 = undefined;
    var codes: [gs_usize]u8 = undefined;
    var local_mse_sum: f64 = 0.0;

    for (row_start..row_end) |r| {
        const s_row = src_bytes + r * row_src_bytes;
        const q_row = dst_qweight + r * row_q_bytes;
        const p_row = dst_params + r * row_p_bytes;

        for (0..groups) |g| {
            loadGroupValues(src_dtype, gs_usize, s_row + g * grp_src_bytes, &values);

            const has_non_uniform_w = if (col_weights) |cw|
                normalizeGroupWeights(gs_usize, cw + g * gs_usize, &w_norm)
            else
                false;

            const res: GroupResult = if (use_rco) blk: {
                if (has_non_uniform_w) {
                    break :blk quantizeGroupRcoWeightedFixed(gs_usize, &values, &w_norm, bits, qmax, rco_iters, &codes);
                } else {
                    break :blk quantizeGroupRcoUnweightedFixed(gs_usize, &values, bits, qmax, rco_iters, &codes);
                }
            } else blk: {
                var mm_res = quantizeGroupMinmaxFixed(gs_usize, &values, qmax, &codes);
                if (has_non_uniform_w) {
                    const w_ev = evalGroupWithSbWeighted(gs_usize, &values, &w_norm, mm_res.s, mm_res.b, qmax, &codes);
                    mm_res.mse = w_ev.wmse;
                }
                break :blk mm_res;
            };

            local_mse_sum += res.mse;

            const dst_q_ptr: *[grp_q_bytes]u8 = @ptrCast(q_row + g * grp_q_bytes);
            Ops.packGroup(&codes, dst_q_ptr);

            const sh: f16 = @floatCast(res.s);
            const bh: f16 = @floatCast(res.b);
            const dst_p = p_row + g * 4;
            std.mem.writeInt(u16, dst_p[0..2], @bitCast(sh), .little);
            std.mem.writeInt(u16, dst_p[2..4], @bitCast(bh), .little);
        }
    }

    out_mse_sum.* = local_mse_sum;
}

const WorkQueue = struct {
    next_row: std.atomic.Value(usize),
    n_rows: usize,
    chunk_rows: usize,
};

const WorkerCtx = struct {
    src_dtype: SrcDType,
    bits: AffineQuantBits,
    group_size: u16,
    src_bytes: [*]const u8,
    dst_qweight: [*]u8,
    dst_params: [*]u8,
    col_weights: ?[*]align(1) const f32,
    row_start: usize,
    row_end: usize,
    k: usize,
    use_rco: bool,
    rco_iters: usize,
    queue: ?*WorkQueue = null,
    mse_sum: f64 = 0.0,
};

fn dispatchRowRange(ctx: *WorkerCtx) void {
    switch (ctx.src_dtype) {
        .f32 => dispatchBits(.f32, ctx),
        .f16 => dispatchBits(.f16, ctx),
        .bf16 => dispatchBits(.bf16, ctx),
    }
}

fn dispatchBits(comptime src_dtype: SrcDType, ctx: *WorkerCtx) void {
    switch (ctx.bits) {
        .q2 => dispatchGroupSize(src_dtype, .q2, ctx),
        .q3 => dispatchGroupSize(src_dtype, .q3, ctx),
        .q4 => dispatchGroupSize(src_dtype, .q4, ctx),
        .q5 => dispatchGroupSize(src_dtype, .q5, ctx),
        .q6 => dispatchGroupSize(src_dtype, .q6, ctx),
        .q8 => dispatchGroupSize(src_dtype, .q8, ctx),
    }
}

fn dispatchGroupSize(comptime src_dtype: SrcDType, comptime bits: AffineQuantBits, ctx: *WorkerCtx) void {
    switch (ctx.group_size) {
        32 => runWorkerFixed(src_dtype, bits, 32, ctx),
        64 => runWorkerFixed(src_dtype, bits, 64, ctx),
        128 => runWorkerFixed(src_dtype, bits, 128, ctx),
        else => unreachable,
    }
}

fn runWorkerFixed(
    comptime src_dtype: SrcDType,
    comptime bits: AffineQuantBits,
    comptime gs: u16,
    ctx: *WorkerCtx,
) void {
    if (ctx.queue) |q| {
        var total_mse: f64 = 0.0;
        while (true) {
            const start = q.next_row.fetchAdd(q.chunk_rows, .monotonic);
            if (start >= q.n_rows) break;
            const end = @min(start + q.chunk_rows, q.n_rows);
            var chunk_mse: f64 = 0.0;
            processRowRangeFixed(
                src_dtype,
                bits,
                gs,
                ctx.src_bytes,
                ctx.dst_qweight,
                ctx.dst_params,
                ctx.col_weights,
                start,
                end,
                ctx.k,
                ctx.use_rco,
                ctx.rco_iters,
                &chunk_mse,
            );
            total_mse += chunk_mse;
        }
        ctx.mse_sum = total_mse;
    } else {
        processRowRangeFixed(
            src_dtype,
            bits,
            gs,
            ctx.src_bytes,
            ctx.dst_qweight,
            ctx.dst_params,
            ctx.col_weights,
            ctx.row_start,
            ctx.row_end,
            ctx.k,
            ctx.use_rco,
            ctx.rco_iters,
            &ctx.mse_sum,
        );
    }
}

fn workerEntry(ctx: *WorkerCtx) void {
    dispatchRowRange(ctx);
}

fn resolveThreadCount(req_threads: usize, n_rows: usize) usize {
    if (n_rows < MIN_ROWS_FOR_PARALLEL) return 1;
    const auto_cap: usize = if (req_threads == 0) blk: {
        const cpu_cnt = std.Thread.getCpuCount() catch |err| switch (err) {
            error.PermissionDenied,
            error.SystemResources,
            error.Unsupported,
            error.Unexpected,
            => 1,
        };
        break :blk @max(@as(usize, 1), cpu_cnt);
    } else req_threads;
    return @min(@min(auto_cap, MAX_THREADS), n_rows);
}

/// Quantizes `n_rows x k` raw float rows (`F32`, `F16`, or `BF16`) into row-major
/// MLX affine packed codes and little-endian `ScaleBias` (`[f16 scale, f16 bias]`).
/// Returns the mean per-group (weighted or unweighted) MSE across all groups.
pub fn quantizeRowsAffineRaw(
    params: Params,
    src_bytes: ?[*]const u8,
    dst_qweight: ?[*]u8,
    dst_params: ?[*]u8,
) Error!f64 {
    if (params.group_size != 32 and params.group_size != 64 and params.group_size != 128) {
        return error.InvalidGroupSize;
    }
    const gs: usize = params.group_size;
    if (params.k == 0) return error.ZeroDimension;
    if (params.k % gs != 0) return error.UnalignedK;
    if (params.col_weights) |cw| {
        if (cw.len != params.k) return error.InvalidGroupSize;
    }

    const groups = params.k / gs;
    const grp_q_bytes = (gs * @as(usize, params.bits.bitWidth())) / 8;
    const row_src_bytes = try std.math.mul(usize, params.k, params.src_dtype.elementBytes());
    const row_q_bytes = try std.math.mul(usize, groups, grp_q_bytes);
    const row_p_bytes = try std.math.mul(usize, groups, 4);

    if (params.n_rows == 0) return 0.0;

    _ = try std.math.mul(usize, params.n_rows, row_src_bytes);
    _ = try std.math.mul(usize, params.n_rows, row_q_bytes);
    _ = try std.math.mul(usize, params.n_rows, row_p_bytes);
    const total_groups = try std.math.mul(usize, params.n_rows, groups);

    const src = src_bytes orelse return error.NullBuffer;
    const dst_q = dst_qweight orelse return error.NullBuffer;
    const dst_p = dst_params orelse return error.NullBuffer;
    const cw_ptr: ?[*]align(1) const f32 = if (params.col_weights) |cw| cw.ptr else null;

    const eff_threads = resolveThreadCount(params.num_threads, params.n_rows);
    if (eff_threads <= 1) {
        var ctx: WorkerCtx = .{
            .src_dtype = params.src_dtype,
            .bits = params.bits,
            .group_size = params.group_size,
            .src_bytes = src,
            .dst_qweight = dst_q,
            .dst_params = dst_p,
            .col_weights = cw_ptr,
            .row_start = 0,
            .row_end = params.n_rows,
            .k = params.k,
            .use_rco = params.use_rco,
            .rco_iters = params.rco_iters,
        };
        dispatchRowRange(&ctx);
        return ctx.mse_sum / @as(f64, @floatFromInt(total_groups));
    }

    var queue: WorkQueue = .{
        .next_row = std.atomic.Value(usize).init(0),
        .n_rows = params.n_rows,
        .chunk_rows = @max(@as(usize, 1), @min(@as(usize, 16), params.n_rows / (eff_threads * 4))),
    };
    var ctxs: [MAX_THREADS]WorkerCtx = undefined;
    var threads: [MAX_THREADS]std.Thread = undefined;

    for (0..eff_threads) |t| {
        ctxs[t] = .{
            .src_dtype = params.src_dtype,
            .bits = params.bits,
            .group_size = params.group_size,
            .src_bytes = src,
            .dst_qweight = dst_q,
            .dst_params = dst_p,
            .col_weights = cw_ptr,
            .row_start = 0,
            .row_end = params.n_rows,
            .k = params.k,
            .use_rco = params.use_rco,
            .rco_iters = params.rco_iters,
            .queue = &queue,
        };
    }

    // Spawn `eff_threads - 1` worker threads and participate on the caller thread.
    var spawned: usize = 0;
    errdefer {
        queue.next_row.store(params.n_rows, .release);
        for (0..spawned) |t| threads[t].join();
    }
    for (0..eff_threads - 1) |t| {
        threads[t] = std.Thread.spawn(.{}, workerEntry, .{&ctxs[t]}) catch return error.ThreadSpawnFailed;
        spawned += 1;
    }
    dispatchRowRange(&ctxs[eff_threads - 1]);
    for (0..spawned) |t| threads[t].join();
    spawned = 0;

    var total_mse_sum: f64 = 0.0;
    for (0..eff_threads) |t| {
        total_mse_sum += ctxs[t].mse_sum;
    }
    return total_mse_sum / @as(f64, @floatFromInt(total_groups));
}

// ============================================================================
// Unit Tests
// ============================================================================

fn encodeSrcElement(dst: []u8, src_dtype: SrcDType, val: f32) void {
    switch (src_dtype) {
        .f32 => std.mem.writeInt(u32, dst[0..4], @bitCast(val), .little),
        .f16 => {
            const h: f16 = @floatCast(val);
            std.mem.writeInt(u16, dst[0..2], @bitCast(h), .little);
        },
        .bf16 => std.mem.writeInt(u16, dst[0..2], spec.f32ToBf16(val), .little),
    }
}

fn decodeSrcElement(src: []const u8, src_dtype: SrcDType) f64 {
    return switch (src_dtype) {
        .f32 => @as(f64, @floatCast(@as(f32, @bitCast(std.mem.readInt(u32, src[0..4], .little))))),
        .f16 => @as(f64, @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, src[0..2], .little))))),
        .bf16 => @as(f64, @floatCast(spec.bf16ToF32(std.mem.readInt(u16, src[0..2], .little)))),
    };
}

fn amfAffineDType(bits: AffineQuantBits) formats.DType {
    return switch (bits) {
        .q2 => .q2_affine,
        .q3 => .q3_affine,
        .q4 => .q4_affine,
        .q5 => .q5_affine,
        .q6 => .q6_affine,
        .q8 => .q8_affine,
    };
}

fn fillDeterministicMatrix(
    src_buf: []u8,
    src_dtype: SrcDType,
    n_rows: usize,
    k: usize,
    seed: u64,
) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    const elem_bytes = src_dtype.elementBytes();

    for (0..n_rows) |r| {
        for (0..k) |c| {
            // Mix Gaussian-like core with occasional outliers so RCO clipping activates.
            const u1_v = rng.float(f64) * 2.0 - 1.0;
            const u2_v = rng.float(f64) * 2.0 - 1.0;
            var val: f32 = @floatCast(0.18 * (u1_v + u2_v));
            if (c % 31 == 3) {
                val = if ((r + c) % 2 == 0) 1.75 else -1.65;
            }
            const off = (r * k + c) * elem_bytes;
            encodeSrcElement(src_buf[off .. off + elem_bytes], src_dtype, val);
        }
    }
}

fn computeActualMse(
    src_bytes: []const u8,
    src_dtype: SrcDType,
    qweight: []const u8,
    params: []const u8,
    bits: AffineQuantBits,
    gs: u16,
    n_rows: usize,
    k: usize,
    col_weights: ?[]const f32,
) !f64 {
    const allocator = std.testing.allocator;
    const deq_row = try allocator.alloc(f32, k);
    defer allocator.free(deq_row);

    const groups = k / gs;
    const elem_bytes = src_dtype.elementBytes();
    const dtype = amfAffineDType(bits);
    var mse_sum: f64 = 0.0;
    var w_norm: [MAX_GROUP_SIZE]f64 = undefined;

    for (0..n_rows) |r| {
        try formats.dequantRowCpuRef(dtype, qweight, params, n_rows, k, r, gs, deq_row);
        for (0..groups) |g| {
            const c_start = g * gs;
            const has_w = if (col_weights) |cw| blk: {
                break :blk switch (gs) {
                    32 => normalizeGroupWeights(32, cw[c_start..].ptr, w_norm[0..32]),
                    64 => normalizeGroupWeights(64, cw[c_start..].ptr, w_norm[0..64]),
                    128 => normalizeGroupWeights(128, cw[c_start..].ptr, w_norm[0..128]),
                    else => unreachable,
                };
            } else false;

            var err2: f64 = 0.0;
            for (0..gs) |j| {
                const col = c_start + j;
                const off = (r * k + col) * elem_bytes;
                const orig = decodeSrcElement(src_bytes[off .. off + elem_bytes], src_dtype);
                const diff = orig - @as(f64, @floatCast(deq_row[col]));
                const w = if (has_w) w_norm[j] else 1.0;
                err2 += w * diff * diff;
            }
            mse_sum += err2 / @as(f64, @floatFromInt(gs));
        }
    }
    return mse_sum / @as(f64, @floatFromInt(n_rows * groups));
}

test "quant.rco: all bits x group_size x src_dtype x weighted/unweighted satisfy mse_rco <= mse_minmax and bit-exact dequant" {
    const allocator = std.testing.allocator;
    const all_bits = [_]AffineQuantBits{ .q2, .q3, .q4, .q5, .q6, .q8 };
    const all_gs = [_]u16{ 32, 64, 128 };
    const all_dtypes = [_]SrcDType{ .f32, .f16, .bf16 };

    const n_rows: usize = 3;
    const k: usize = 256;

    var col_weights: [k]f32 = undefined;
    for (0..k) |i| {
        col_weights[i] = if ((i % 32) < 8) 120.0 else 1e-3;
    }

    for (all_dtypes) |sdt| {
        const src_buf = try allocator.alloc(u8, n_rows * k * sdt.elementBytes());
        defer allocator.free(src_buf);
        fillDeterministicMatrix(src_buf, sdt, n_rows, k, 0x51A7_1001 + @backingInt(sdt));

        for (all_gs) |gs| {
            const groups = k / gs;
            for (all_bits) |bits| {
                const q_bytes = n_rows * groups * ((@as(usize, gs) * bits.bitWidth()) / 8);
                const p_bytes = n_rows * groups * 4;

                const q_mm = try allocator.alloc(u8, q_bytes);
                defer allocator.free(q_mm);
                const p_mm = try allocator.alloc(u8, p_bytes);
                defer allocator.free(p_mm);
                const q_rco = try allocator.alloc(u8, q_bytes);
                defer allocator.free(q_rco);
                const p_rco = try allocator.alloc(u8, p_bytes);
                defer allocator.free(p_rco);

                for ([_]?[]const f32{ null, &col_weights }) |maybe_w| {
                    const mse_mm = try quantizeRowsAffineRaw(.{
                        .src_dtype = sdt,
                        .n_rows = n_rows,
                        .k = k,
                        .bits = bits,
                        .group_size = gs,
                        .use_rco = false,
                        .rco_iters = 3,
                        .col_weights = maybe_w,
                        .num_threads = 1,
                    }, src_buf.ptr, q_mm.ptr, p_mm.ptr);

                    const mse_rco = try quantizeRowsAffineRaw(.{
                        .src_dtype = sdt,
                        .n_rows = n_rows,
                        .k = k,
                        .bits = bits,
                        .group_size = gs,
                        .use_rco = true,
                        .rco_iters = 3,
                        .col_weights = maybe_w,
                        .num_threads = 1,
                    }, src_buf.ptr, q_rco.ptr, p_rco.ptr);

                    try std.testing.expect(std.math.isFinite(mse_mm) and std.math.isFinite(mse_rco));
                    try std.testing.expect(mse_rco <= mse_mm + 1e-15);

                    const actual_mm = try computeActualMse(src_buf, sdt, q_mm, p_mm, bits, gs, n_rows, k, maybe_w);
                    const actual_rco = try computeActualMse(src_buf, sdt, q_rco, p_rco, bits, gs, n_rows, k, maybe_w);
                    try std.testing.expectApproxEqAbs(actual_mm, mse_mm, 1e-6);
                    try std.testing.expectApproxEqAbs(actual_rco, mse_rco, 1e-6);
                }
            }
        }
    }
}

test "quant.rco: weighted RCO focuses precision on high-weight channels over low-weight outliers" {
    const allocator = std.testing.allocator;
    const gs: usize = 64;
    const n_rows: usize = 2;
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    var src_f32: [n_rows * gs]f32 = undefined;
    var weights: [gs]f32 = undefined;
    for (0..n_rows) |r| {
        for (0..16) |c| {
            src_f32[r * gs + c] = @floatCast((rng.float(f64) * 2.0 - 1.0) * 0.14);
        }
        for (16..60) |c| {
            src_f32[r * gs + c] = @floatCast((rng.float(f64) * 2.0 - 1.0) * 0.08);
        }
        src_f32[r * gs + 60] = 2.2;
        src_f32[r * gs + 61] = -2.1;
        src_f32[r * gs + 62] = 1.9;
        src_f32[r * gs + 63] = -1.8;
    }
    for (0..16) |c| weights[c] = 100.0;
    for (16..gs) |c| weights[c] = 1e-4;

    var src_bytes: [n_rows * gs * 4]u8 = undefined;
    for (src_f32, 0..) |v, idx| {
        std.mem.writeInt(u32, src_bytes[idx * 4 ..][0..4], @bitCast(v), .little);
    }

    const q_uw = try allocator.alloc(u8, n_rows * 16);
    defer allocator.free(q_uw);
    const p_uw = try allocator.alloc(u8, n_rows * 4);
    defer allocator.free(p_uw);
    const q_w = try allocator.alloc(u8, n_rows * 16);
    defer allocator.free(q_w);
    const p_w = try allocator.alloc(u8, n_rows * 4);
    defer allocator.free(p_w);

    _ = try quantizeRowsAffineRaw(.{
        .src_dtype = .f32,
        .n_rows = n_rows,
        .k = gs,
        .bits = .q2,
        .group_size = 64,
        .use_rco = true,
        .rco_iters = 3,
        .col_weights = null,
        .num_threads = 1,
    }, &src_bytes, q_uw.ptr, p_uw.ptr);

    const reported_wmse = try quantizeRowsAffineRaw(.{
        .src_dtype = .f32,
        .n_rows = n_rows,
        .k = gs,
        .bits = .q2,
        .group_size = 64,
        .use_rco = true,
        .rco_iters = 3,
        .col_weights = &weights,
        .num_threads = 1,
    }, &src_bytes, q_w.ptr, p_w.ptr);

    const uw_under_weights = try computeActualMse(&src_bytes, .f32, q_uw, p_uw, .q2, 64, n_rows, gs, &weights);
    const w_under_weights = try computeActualMse(&src_bytes, .f32, q_w, p_w, .q2, 64, n_rows, gs, &weights);
    try std.testing.expectApproxEqAbs(w_under_weights, reported_wmse, 1e-6);
    try std.testing.expect(w_under_weights < uw_under_weights * 0.5);
}

test "quant.rco: degenerate weights fall back to unweighted RCO and multi-threaded matches single-threaded bit-for-bit" {
    const allocator = std.testing.allocator;
    const n_rows: usize = 67; // Not divisible by thread count and >= MIN_ROWS_FOR_PARALLEL
    const k: usize = 128;
    const gs: u16 = 64;

    const src_buf = try allocator.alloc(u8, n_rows * k * 2);
    defer allocator.free(src_buf);
    fillDeterministicMatrix(src_buf, .bf16, n_rows, k, 9001);

    const q_len = n_rows * ((k * 3) / 8);
    const p_len = n_rows * (k / gs) * 4;
    const q1 = try allocator.alloc(u8, q_len);
    defer allocator.free(q1);
    const p1 = try allocator.alloc(u8, p_len);
    defer allocator.free(p1);
    const q4 = try allocator.alloc(u8, q_len);
    defer allocator.free(q4);
    const p4 = try allocator.alloc(u8, p_len);
    defer allocator.free(p4);
    const q_deg = try allocator.alloc(u8, q_len);
    defer allocator.free(q_deg);
    const p_deg = try allocator.alloc(u8, p_len);
    defer allocator.free(p_deg);

    var bad_weights: [k]f32 = undefined;
    for (0..k) |i| {
        bad_weights[i] = switch (i % 4) {
            0 => 0.0,
            1 => -5.0,
            2 => std.math.nan(f32),
            else => std.math.inf(f32),
        };
    }

    const mse_1t = try quantizeRowsAffineRaw(.{
        .src_dtype = .bf16,
        .n_rows = n_rows,
        .k = k,
        .bits = .q3,
        .group_size = gs,
        .use_rco = true,
        .rco_iters = 3,
        .col_weights = null,
        .num_threads = 1,
    }, src_buf.ptr, q1.ptr, p1.ptr);

    const mse_4t = try quantizeRowsAffineRaw(.{
        .src_dtype = .bf16,
        .n_rows = n_rows,
        .k = k,
        .bits = .q3,
        .group_size = gs,
        .use_rco = true,
        .rco_iters = 3,
        .col_weights = null,
        .num_threads = 4,
    }, src_buf.ptr, q4.ptr, p4.ptr);

    const mse_deg = try quantizeRowsAffineRaw(.{
        .src_dtype = .bf16,
        .n_rows = n_rows,
        .k = k,
        .bits = .q3,
        .group_size = gs,
        .use_rco = true,
        .rco_iters = 3,
        .col_weights = &bad_weights,
        .num_threads = 0,
    }, src_buf.ptr, q_deg.ptr, p_deg.ptr);

    try std.testing.expectEqualSlices(u8, q1, q4);
    try std.testing.expectEqualSlices(u8, p1, p4);
    try std.testing.expectEqualSlices(u8, q1, q_deg);
    try std.testing.expectEqualSlices(u8, p1, p_deg);
    try std.testing.expectApproxEqAbs(mse_1t, mse_4t, 1e-14);
    try std.testing.expectApproxEqAbs(mse_1t, mse_deg, 1e-14);
}

test "quant.rco: handles constant groups, sub-f16 dynamic range, unaligned buffers, and iters=0" {
    const gs: usize = 32;
    // Unaligned offsets (+1 byte) for F32/q4/params buffers.
    var raw_storage: [2 * gs * 4 + 3]u8 = undefined;
    var q_storage: [2 * 16 + 3]u8 = undefined;
    var p_storage: [2 * 4 + 3]u8 = undefined;
    const src = raw_storage[1..][0 .. 2 * gs * 4];
    const q_out = q_storage[1..][0 .. 2 * 16];
    const p_out = p_storage[1..][0 .. 2 * 4];

    // Row 0: constant group (hi == lo); Row 1: sub-f16 dynamic range (s_mm == 0).
    for (0..gs) |c| {
        encodeSrcElement(src[c * 4 ..][0..4], .f32, 0.25);
        encodeSrcElement(src[(gs + c) * 4 ..][0..4], .f32, 0.5 + @as(f32, @floatFromInt(c)) * 1e-12);
    }

    const mse = try quantizeRowsAffineRaw(.{
        .src_dtype = .f32,
        .n_rows = 2,
        .k = gs,
        .bits = .q4,
        .group_size = 32,
        .use_rco = true,
        .rco_iters = 3,
        .col_weights = null,
        .num_threads = 1,
    }, src.ptr, q_out.ptr, p_out.ptr);
    try std.testing.expect(std.math.isFinite(mse));
    try std.testing.expect(mse < 1e-6);

    // Row 0 and Row 1 must both emit zero scale (constant group representation).
    try std.testing.expectEqual(@as(u16, 0), std.mem.readInt(u16, p_out[0..2], .little));
    try std.testing.expectEqual(@as(u16, 0), std.mem.readInt(u16, p_out[4..6], .little));
    try std.testing.expectEqual(@as(f16, 0.25), @as(f16, @bitCast(std.mem.readInt(u16, p_out[2..4], .little))));
    try std.testing.expectEqual(@as(f16, 0.5), @as(f16, @bitCast(std.mem.readInt(u16, p_out[6..8], .little))));

    // iters=0 with use_rco=true must match use_rco=false (exact minmax).
    var q_mm: [2 * 16]u8 = undefined;
    var p_mm: [2 * 4]u8 = undefined;
    var q_it0: [2 * 16]u8 = undefined;
    var p_it0: [2 * 4]u8 = undefined;
    fillDeterministicMatrix(src, .f32, 2, gs, 12345);
    const mse_mm = try quantizeRowsAffineRaw(.{
        .src_dtype = .f32,
        .n_rows = 2,
        .k = gs,
        .bits = .q4,
        .group_size = 32,
        .use_rco = false,
        .rco_iters = 3,
        .col_weights = null,
        .num_threads = 1,
    }, src.ptr, &q_mm, &p_mm);
    const mse_it0 = try quantizeRowsAffineRaw(.{
        .src_dtype = .f32,
        .n_rows = 2,
        .k = gs,
        .bits = .q4,
        .group_size = 32,
        .use_rco = true,
        .rco_iters = 0,
        .col_weights = null,
        .num_threads = 1,
    }, src.ptr, &q_it0, &p_it0);
    try std.testing.expectEqualSlices(u8, &q_mm, &q_it0);
    try std.testing.expectEqualSlices(u8, &p_mm, &p_it0);
    try std.testing.expectApproxEqAbs(mse_mm, mse_it0, 1e-15);

    // n_rows=0 returns 0.0 cleanly even with null pointers.
    const empty_mse = try quantizeRowsAffineRaw(.{
        .src_dtype = .bf16,
        .n_rows = 0,
        .k = 64,
        .bits = .q4,
        .group_size = 64,
        .use_rco = true,
        .rco_iters = 3,
        .col_weights = null,
        .num_threads = 0,
    }, null, null, null);
    try std.testing.expectEqual(@as(f64, 0.0), empty_mse);
}
