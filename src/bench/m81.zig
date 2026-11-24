//! Milestone M8.1 (`ADR-0024`) `W24` `KL`/top-1 capability quality-gate harness
//! and `aminfer bench-m81` benchmark suite (`G23`, `G24`, `C25`, `W21`/`W22`,
//! `W24`/`C18`/`C17`).
//!
//! All measurements run on real Apple Silicon hardware over 16 KiB-aligned
//! wired memory with `swapouts_delta == 0` strictly enforced (`ADR-0002`,
//! `ADR-0011`, `ADR-0012`).

const std = @import("std");
const metal = @import("../bridge/metal.zig");
const page = @import("../mem/page.zig");
const budget_mod = @import("../mem/budget.zig");
const sysinfo = @import("../platform/sysinfo.zig");
const chip_profile_mod = @import("../platform/chip_profile.zig");
const vmstat = @import("../platform/vmstat.zig");
const quant = @import("../quant/spec.zig");
const amf = @import("../format/amf.zig");
const dequant = @import("../gpu/dequant.zig");
const quant_formats = @import("../gpu/quant_formats.zig");
const sme = @import("../cpu/sme.zig");
const qwen3 = @import("../model/qwen3.zig");
const draft_mod = @import("../spec/draft.zig");
const tokenizer_mod = @import("../model/tokenizer.zig");
const chat_template_mod = @import("../model/chat_template.zig");
const server_mod = @import("../server/http_sse.zig");

const Io = std.Io;

// ============================================================================
// W24 Logits KL Divergence & Top-1 Capability Quality Gate
// ============================================================================

pub const KlStepMetrics = struct {
    kl_divergence: f64,
    ref_top1: u32,
    cand_top1: u32,
    top1_match: bool,
    all_finite: bool,
};

/// Computes numerically stabilized `f64` softmax `KL(P_ref || P_cand)` and
/// greedy top-1 token agreement from raw `f16` (or `f32`) logits slices.
pub fn computeLogitsKlAndTop1F32(ref_logits: []const f32, cand_logits: []const f32) KlStepMetrics {
    std.debug.assert(ref_logits.len == cand_logits.len and ref_logits.len > 0);
    var ref_max: f64 = -std.math.inf(f64);
    var cand_max: f64 = -std.math.inf(f64);
    var ref_best_v: f32 = -std.math.inf(f32);
    var cand_best_v: f32 = -std.math.inf(f32);
    var ref_top1: u32 = 0;
    var cand_top1: u32 = 0;
    var all_finite = true;

    for (ref_logits, cand_logits, 0..) |rv, cv, idx| {
        if (!std.math.isFinite(rv) or !std.math.isFinite(cv)) {
            all_finite = false;
            continue;
        }
        const r64: f64 = @floatCast(rv);
        const c64: f64 = @floatCast(cv);
        if (r64 > ref_max) ref_max = r64;
        if (c64 > cand_max) cand_max = c64;
        if (rv > ref_best_v) {
            ref_best_v = rv;
            ref_top1 = @intCast(idx);
        }
        if (cv > cand_best_v) {
            cand_best_v = cv;
            cand_top1 = @intCast(idx);
        }
    }

    if (!all_finite or !std.math.isFinite(ref_max) or !std.math.isFinite(cand_max)) {
        return .{
            .kl_divergence = std.math.inf(f64),
            .ref_top1 = ref_top1,
            .cand_top1 = cand_top1,
            .top1_match = false,
            .all_finite = false,
        };
    }

    var ref_sum_exp: f64 = 0.0;
    var cand_sum_exp: f64 = 0.0;
    for (ref_logits, cand_logits) |rv, cv| {
        ref_sum_exp += @exp(@as(f64, @floatCast(rv)) - ref_max);
        cand_sum_exp += @exp(@as(f64, @floatCast(cv)) - cand_max);
    }
    const ref_lse = ref_max + @log(ref_sum_exp);
    const cand_lse = cand_max + @log(cand_sum_exp);

    var kl: f64 = 0.0;
    for (ref_logits, cand_logits) |rv, cv| {
        const log_p = @as(f64, @floatCast(rv)) - ref_lse;
        const log_q = @as(f64, @floatCast(cv)) - cand_lse;
        const p = @exp(log_p);
        if (p > 0.0) {
            kl += p * (log_p - log_q);
        }
    }
    if (kl < 0.0 and kl > -1e-9) kl = 0.0;

    return .{
        .kl_divergence = kl,
        .ref_top1 = ref_top1,
        .cand_top1 = cand_top1,
        .top1_match = (ref_top1 == cand_top1),
        .all_finite = true,
    };
}

pub fn computeLogitsKlAndTop1F16(ref_logits: []const f16, cand_logits: []const f16) KlStepMetrics {
    std.debug.assert(ref_logits.len == cand_logits.len and ref_logits.len > 0);
    var ref_max: f64 = -std.math.inf(f64);
    var cand_max: f64 = -std.math.inf(f64);
    var ref_best_v: f32 = -std.math.inf(f32);
    var cand_best_v: f32 = -std.math.inf(f32);
    var ref_top1: u32 = 0;
    var cand_top1: u32 = 0;
    var all_finite = true;

    for (ref_logits, cand_logits, 0..) |rv16, cv16, idx| {
        const rv: f32 = @floatCast(rv16);
        const cv: f32 = @floatCast(cv16);
        if (!std.math.isFinite(rv) or !std.math.isFinite(cv)) {
            all_finite = false;
            continue;
        }
        const r64: f64 = @floatCast(rv);
        const c64: f64 = @floatCast(cv);
        if (r64 > ref_max) ref_max = r64;
        if (c64 > cand_max) cand_max = c64;
        if (rv > ref_best_v) {
            ref_best_v = rv;
            ref_top1 = @intCast(idx);
        }
        if (cv > cand_best_v) {
            cand_best_v = cv;
            cand_top1 = @intCast(idx);
        }
    }

    if (!all_finite or !std.math.isFinite(ref_max) or !std.math.isFinite(cand_max)) {
        return .{
            .kl_divergence = std.math.inf(f64),
            .ref_top1 = ref_top1,
            .cand_top1 = cand_top1,
            .top1_match = false,
            .all_finite = false,
        };
    }

    var ref_sum_exp: f64 = 0.0;
    var cand_sum_exp: f64 = 0.0;
    for (ref_logits, cand_logits) |rv16, cv16| {
        ref_sum_exp += @exp(@as(f64, @floatCast(rv16)) - ref_max);
        cand_sum_exp += @exp(@as(f64, @floatCast(cv16)) - cand_max);
    }
    const ref_lse = ref_max + @log(ref_sum_exp);
    const cand_lse = cand_max + @log(cand_sum_exp);

    var kl: f64 = 0.0;
    for (ref_logits, cand_logits) |rv16, cv16| {
        const log_p = @as(f64, @floatCast(rv16)) - ref_lse;
        const log_q = @as(f64, @floatCast(cv16)) - cand_lse;
        const p = @exp(log_p);
        if (p > 0.0) {
            kl += p * (log_p - log_q);
        }
    }
    if (kl < 0.0 and kl > -1e-9) kl = 0.0;

    return .{
        .kl_divergence = kl,
        .ref_top1 = ref_top1,
        .cand_top1 = cand_top1,
        .top1_match = (ref_top1 == cand_top1),
        .all_finite = true,
    };
}

pub const QualityGatePairResult = struct {
    label_buf: [64]u8 = @splat(0),
    label_len: usize = 0,
    dtype: amf.DType = .bf16,
    group_size: u16 = 0,
    act_quant: quant_formats.ActQuantPolicy = .none,
    steps_evaluated: u32 = 0,
    top1_matches: u32 = 0,
    top1_agreement_pct: f64 = 0.0,
    mean_kl: f64 = 0.0,
    max_kl: f64 = 0.0,
    greedy_rollout_steps: u32 = 0,
    greedy_rollout_matches: u32 = 0,
    math_greedy_steps: u32 = 0,
    math_greedy_matches: u32 = 0,
    code_greedy_steps: u32 = 0,
    code_greedy_matches: u32 = 0,
    tokenizer_resolved: bool = false,
    all_finite: bool = true,
    executed: bool = false,
    status: []const u8 = "ok",

    pub fn label(self: *const QualityGatePairResult) []const u8 {
        return self.label_buf[0..self.label_len];
    }

    pub fn setLabel(self: *QualityGatePairResult, s: []const u8) void {
        const n = @min(s.len, self.label_buf.len);
        @memcpy(self.label_buf[0..n], s[0..n]);
        self.label_len = n;
    }
};

pub const QualityGateConfig = struct {
    ref_amf_path: ?[:0]const u8 = null,
    cand_amf_path: ?[:0]const u8 = null,
    config_path: ?[:0]const u8 = null,
    tokenizer_path: ?[:0]const u8 = null,
    act_quant_policy: quant_formats.ActQuantPolicy = .none,
    steps: u32 = 8,
    max_seq_len: u32 = 256,
    prompt_tokens: ?[]const u32 = null,
    include_synthetic_sweep: bool = true,
};

pub const MAX_SYNTH_QUALITY_PAIRS: usize = 12;

pub const QualityGateReport = struct {
    synth_count: usize = 0,
    synth_pairs: [MAX_SYNTH_QUALITY_PAIRS]QualityGatePairResult = @splat(.{}),
    file_pair: ?QualityGatePairResult = null,
    vm_before: vmstat.Snapshot = undefined,
    vm_after: vmstat.Snapshot = undefined,

    pub fn write(self: *const QualityGateReport, w: *Io.Writer) !void {
        try w.print("quality_gate.swapouts_delta      {d}\n", .{
            vmstat.Snapshot.swapoutDelta(self.vm_before, self.vm_after),
        });
        for (self.synth_pairs[0..self.synth_count]) |*p| {
            try w.print(
                "quality_gate.synth.{s:<20} mean_kl={d:.6} max_kl={d:.6} top1={d:.1}% ({d}/{d}) greedy={d}/{d} act_quant={s} finite={s}\n",
                .{
                    p.label(),
                    p.mean_kl,
                    p.max_kl,
                    p.top1_agreement_pct,
                    p.top1_matches,
                    p.steps_evaluated,
                    p.greedy_rollout_matches,
                    p.greedy_rollout_steps,
                    p.act_quant.label(),
                    if (p.all_finite) "yes" else "no",
                },
            );
        }
        if (self.file_pair) |*fp| {
            try w.print(
                "quality_gate.real.{s:<21} status={s} mean_kl={d:.6} max_kl={d:.6} top1={d:.1}% ({d}/{d}) greedy={d}/{d} math={d}/{d} code={d}/{d} tok={s} act_quant={s} finite={s}\n",
                .{
                    fp.label(),
                    fp.status,
                    fp.mean_kl,
                    fp.max_kl,
                    fp.top1_agreement_pct,
                    fp.top1_matches,
                    fp.steps_evaluated,
                    fp.greedy_rollout_matches,
                    fp.greedy_rollout_steps,
                    fp.math_greedy_matches,
                    fp.math_greedy_steps,
                    fp.code_greedy_matches,
                    fp.code_greedy_steps,
                    if (fp.tokenizer_resolved) "resolved" else "fallback",
                    fp.act_quant.label(),
                    if (fp.all_finite) "yes" else "no",
                },
            );
        }
    }
};

// ============================================================================
// Deterministic Multi-Format Synthetic `.amf` Builder (Zero Double-Quant)
// ============================================================================

const QuantizedBlob = struct {
    data: []u8,
    params: []u8,
    dtype: amf.DType,
    quant_bits: u8,
    group_size: u16,

    fn deinit(self: QuantizedBlob, allocator: std.mem.Allocator) void {
        allocator.free(self.data);
        if (self.params.len > 0) allocator.free(self.params);
    }
};

fn fillDeterministicMatrixF32(seed: u64, n: usize, k: usize, scale: f32, out: []f32) void {
    var st: u64 = seed ^ 0x9E3779B97F4A7C15;
    for (0..n * k) |i| {
        st ^= st >> 12;
        st ^= st << 25;
        st ^= st >> 27;
        const u = (st *% 0x2545F4914F6CDD1D) >> 40;
        const f = (@as(f32, @floatFromInt(u)) / 16777216.0) * 2.0 - 1.0;
        out[i] = f * scale;
    }
}

fn writeU16Le(dst: []u8, v: u16) void {
    std.mem.writeInt(u16, dst[0..2], v, .little);
}

fn writeF16Le(dst: []u8, v: f32) void {
    const h: f16 = @floatCast(v);
    writeU16Le(dst, @bitCast(h));
}

fn writeF32Le(dst: []u8, v: f32) void {
    std.mem.writeInt(u32, dst[0..4], @bitCast(v), .little);
}

fn encodeUe8m0FromAmax(amax: f32) u8 {
    if (amax <= 0.0) return 0;
    const raw_scale = amax / 448.0;
    const u: u32 = @bitCast(raw_scale);
    const exp_u: i32 = @as(i32, @intCast((u >> 23) & 0xFF));
    const mant = u & 0x7FFFFF;
    const biased = exp_u + (if (mant != 0) @as(i32, 1) else @as(i32, 0));
    return @intCast(std.math.clamp(biased, 0, 254));
}

fn packScaleMinK4(j: usize, sc: u8, mn: u8, dst: *[12]u8) void {
    const s6 = sc & 63;
    const m6 = mn & 63;
    if (j < 4) {
        dst[j] = (dst[j] & 0xC0) | s6;
        dst[j + 4] = (dst[j + 4] & 0xC0) | m6;
    } else {
        dst[j + 4] = (s6 & 0xF) | ((m6 & 0xF) << 4);
        dst[j - 4] = (dst[j - 4] & 0x3F) | ((s6 >> 4) << 6);
        dst[j] = (dst[j] & 0x3F) | ((m6 >> 4) << 6);
    }
}

/// Losslessly encodes an `[n, k]` `f32` matrix into canonical `.amf` `data` and
/// `params` buffers for any `M8.1` target format (`bf16`, `f16`, MLX affine
/// `q2..q8`, `fp8_e4m3_block/row/tensor`, `mxfp8_g32`, `nvfp4_g16`,
/// `gguf_q8_0`, `gguf_q4_k`, `gguf_q2_0`).
pub fn quantizeMatrixToAmf(
    allocator: std.mem.Allocator,
    src_f32: []const f32,
    n: usize,
    k: usize,
    dtype: amf.DType,
    group_size: u16,
) !QuantizedBlob {
    switch (dtype) {
        .bf16 => {
            const data = try allocator.alloc(u8, n * k * 2);
            for (0..n * k) |i| writeU16Le(data[i * 2 ..][0..2], quant.f32ToBf16(src_f32[i]));
            return .{ .data = data, .params = &.{}, .dtype = .bf16, .quant_bits = 0, .group_size = 0 };
        },
        .f16 => {
            const data = try allocator.alloc(u8, n * k * 2);
            for (0..n * k) |i| writeF16Le(data[i * 2 ..][0..2], src_f32[i]);
            return .{ .data = data, .params = &.{}, .dtype = .f16, .quant_bits = 0, .group_size = 0 };
        },
        .q2_affine, .q3_affine, .q4_affine, .q5_affine, .q6_affine, .q8_affine => {
            const qf_dt = quant_formats.DType.fromAmf(dtype).?;
            const bits = qf_dt.affineBits().?;
            const gs: usize = qf_dt.effectiveGroupSize(group_size);
            const grp_bytes = (gs * bits) / 8;
            const grps = k / gs;
            const data = try allocator.alloc(u8, n * grps * grp_bytes);
            errdefer allocator.free(data);
            const params = try allocator.alloc(u8, n * grps * 4);
            errdefer allocator.free(params);
            const max_q: f32 = @floatFromInt((@as(u32, 1) << @intCast(bits)) - 1);
            var codes_buf: [128]u8 = undefined;

            for (0..n) |r| {
                for (0..grps) |g| {
                    const row_g = src_f32[r * k + g * gs ..][0..gs];
                    var lo: f32 = row_g[0];
                    var hi: f32 = row_g[0];
                    for (row_g[1..]) |v| {
                        if (v < lo) lo = v;
                        if (v > hi) hi = v;
                    }
                    const sc: f32 = if (hi > lo) (hi - lo) / max_q else 1.0;
                    const bi: f32 = lo;
                    const sc_f16: f16 = @floatCast(sc);
                    const bi_f16: f16 = @floatCast(bi);
                    const sc_eff: f32 = @floatCast(sc_f16);
                    const bi_eff: f32 = @floatCast(bi_f16);
                    const p_off = (r * grps + g) * 4;
                    writeU16Le(params[p_off .. p_off + 2], @bitCast(sc_f16));
                    writeU16Le(params[p_off + 2 .. p_off + 4], @bitCast(bi_f16));
                    for (0..gs) |j| {
                        const qf = if (sc_eff > 0.0) @round((row_g[j] - bi_eff) / sc_eff) else 0.0;
                        const qi: i32 = @intFromFloat(std.math.clamp(qf, 0.0, max_q));
                        codes_buf[j] = @intCast(qi);
                    }
                    const d_off = (r * grps + g) * grp_bytes;
                    quant_formats.packAffineGroup(bits, @intCast(gs), codes_buf[0..gs], data[d_off..][0..grp_bytes]);
                }
            }
            return .{ .data = data, .params = params, .dtype = dtype, .quant_bits = bits, .group_size = @intCast(gs) };
        },
        .fp8_e4m3_block => {
            const data = try allocator.alloc(u8, n * k);
            errdefer allocator.free(data);
            const bn = (n + 127) / 128;
            const bk = (k + 127) / 128;
            const params = try allocator.alloc(u8, bn * bk * 4);
            errdefer allocator.free(params);
            for (0..bn) |br| {
                const r0 = br * 128;
                const r1 = @min(n, r0 + 128);
                for (0..bk) |bc| {
                    const c0 = bc * 128;
                    const c1 = @min(k, c0 + 128);
                    var amax: f32 = 0.0;
                    for (r0..r1) |r| {
                        for (c0..c1) |c| amax = @max(amax, @abs(src_f32[r * k + c]));
                    }
                    const sc: f32 = if (amax > 0.0) amax / 448.0 else 1.0;
                    writeF32Le(params[(br * bk + bc) * 4 ..][0..4], sc);
                    for (r0..r1) |r| {
                        for (c0..c1) |c| {
                            data[r * k + c] = quant_formats.f32ToE4m3(src_f32[r * k + c] / sc);
                        }
                    }
                }
            }
            return .{ .data = data, .params = params, .dtype = .fp8_e4m3_block, .quant_bits = 8, .group_size = 128 };
        },
        .mxfp8_g32 => {
            const grps = k / 32;
            const data = try allocator.alloc(u8, n * k);
            errdefer allocator.free(data);
            const params = try allocator.alloc(u8, n * grps);
            errdefer allocator.free(params);
            for (0..n) |r| {
                for (0..grps) |g| {
                    const slice = src_f32[r * k + g * 32 ..][0..32];
                    var amax: f32 = 0.0;
                    for (slice) |v| amax = @max(amax, @abs(v));
                    const ue = encodeUe8m0FromAmax(amax);
                    params[r * grps + g] = ue;
                    const sc = quant_formats.ue8m0ToF32(ue);
                    for (0..32) |j| {
                        data[r * k + g * 32 + j] = if (sc > 0.0) quant_formats.f32ToE4m3(slice[j] / sc) else 0;
                    }
                }
            }
            return .{ .data = data, .params = params, .dtype = .mxfp8_g32, .quant_bits = 8, .group_size = 32 };
        },
        .nvfp4_g16 => {
            const grps = k / 16;
            const data = try allocator.alloc(u8, n * (k / 2));
            errdefer allocator.free(data);
            const micro_raw = n * grps;
            const outer_off = ((micro_raw + 3) / 4) * 4;
            const params = try allocator.alloc(u8, outer_off + 4);
            errdefer allocator.free(params);
            @memset(params, 0);

            var global_amax: f32 = 0.0;
            for (src_f32) |v| global_amax = @max(global_amax, @abs(v));
            const outer: f32 = if (global_amax > 0.0) global_amax / 2688.0 else 1.0;
            writeF32Le(params[outer_off .. outer_off + 4], outer);

            for (0..n) |r| {
                for (0..grps) |g| {
                    const slice = src_f32[r * k + g * 16 ..][0..16];
                    var bmax: f32 = 0.0;
                    for (slice) |v| bmax = @max(bmax, @abs(v));
                    const sc_e4 = if (bmax > 0.0) quant_formats.f32ToE4m3(bmax / (6.0 * outer)) else 0;
                    params[r * grps + g] = sc_e4;
                    const sc = quant_formats.e4m3ScaleToF32(sc_e4) * outer;
                    const row_off = r * (k / 2) + g * 8;
                    for (0..8) |b| {
                        const n0: u8 = if (sc > 0.0) quant_formats.f32ToE2m1(slice[2 * b + 0] / sc) else 0;
                        const n1: u8 = if (sc > 0.0) quant_formats.f32ToE2m1(slice[2 * b + 1] / sc) else 0;
                        data[row_off + b] = n0 | (n1 << 4);
                    }
                }
            }
            return .{ .data = data, .params = params, .dtype = .nvfp4_g16, .quant_bits = 4, .group_size = 16 };
        },
        .gguf_q8_0 => {
            const blks = k / 32;
            const data = try allocator.alloc(u8, n * blks * 34);
            errdefer allocator.free(data);
            for (0..n) |r| {
                for (0..blks) |b| {
                    const slice = src_f32[r * k + b * 32 ..][0..32];
                    var amax: f32 = 0.0;
                    for (slice) |v| amax = @max(amax, @abs(v));
                    const d: f32 = if (amax > 0.0) amax / 127.0 else 0.0;
                    const d_f16: f16 = @floatCast(d);
                    const d_eff: f32 = @floatCast(d_f16);
                    const blk = data[(r * blks + b) * 34 ..][0..34];
                    writeU16Le(blk[0..2], @bitCast(d_f16));
                    for (0..32) |j| {
                        const qf = if (d_eff > 0.0) @round(slice[j] / d_eff) else 0.0;
                        const qi: i8 = @intFromFloat(std.math.clamp(qf, -127.0, 127.0));
                        blk[2 + j] = @bitCast(qi);
                    }
                }
            }
            return .{ .data = data, .params = &.{}, .dtype = .gguf_q8_0, .quant_bits = 8, .group_size = 32 };
        },
        .gguf_q4_k => {
            const sbs = k / 256;
            const data = try allocator.alloc(u8, n * sbs * 144);
            errdefer allocator.free(data);
            @memset(data, 0);
            for (0..n) |r| {
                for (0..sbs) |sb| {
                    const blk = data[(r * sbs + sb) * 144 ..][0..144];
                    var sub_sc: [8]f32 = undefined;
                    var sub_mn: [8]f32 = undefined;
                    var max_sc: f32 = 0.0;
                    var max_mn: f32 = 0.0;
                    for (0..8) |j| {
                        const s = src_f32[r * k + sb * 256 + j * 32 ..][0..32];
                        var lo: f32 = s[0];
                        var hi: f32 = s[0];
                        for (s[1..]) |v| {
                            if (v < lo) lo = v;
                            if (v > hi) hi = v;
                        }
                        const mn_pos = @max(0.0, -lo);
                        const span = hi + mn_pos;
                        sub_sc[j] = if (span > 0.0) span / 15.0 else 0.0;
                        sub_mn[j] = mn_pos;
                        max_sc = @max(max_sc, sub_sc[j]);
                        max_mn = @max(max_mn, sub_mn[j]);
                    }
                    const d_f16: f16 = @floatCast(if (max_sc > 0.0) max_sc / 63.0 else 0.0);
                    const dmin_f16: f16 = @floatCast(if (max_mn > 0.0) max_mn / 63.0 else 0.0);
                    const d_eff: f32 = @floatCast(d_f16);
                    const dmin_eff: f32 = @floatCast(dmin_f16);
                    writeU16Le(blk[0..2], @bitCast(d_f16));
                    writeU16Le(blk[2..4], @bitCast(dmin_f16));
                    var scales_raw: [12]u8 = @splat(0);
                    var q_sc: [8]u8 = undefined;
                    var q_mn: [8]u8 = undefined;
                    for (0..8) |j| {
                        q_sc[j] = if (d_eff > 0.0) @intFromFloat(std.math.clamp(@round(sub_sc[j] / d_eff), 0.0, 63.0)) else 0;
                        q_mn[j] = if (dmin_eff > 0.0) @intFromFloat(std.math.clamp(@round(sub_mn[j] / dmin_eff), 0.0, 63.0)) else 0;
                        packScaleMinK4(j, q_sc[j], q_mn[j], &scales_raw);
                    }
                    @memcpy(blk[4..16], &scales_raw);
                    const qs = blk[16..144];
                    for (0..4) |pair| {
                        const d1 = d_eff * @as(f32, @floatFromInt(q_sc[pair * 2]));
                        const m1 = dmin_eff * @as(f32, @floatFromInt(q_mn[pair * 2]));
                        const d2 = d_eff * @as(f32, @floatFromInt(q_sc[pair * 2 + 1]));
                        const m2 = dmin_eff * @as(f32, @floatFromInt(q_mn[pair * 2 + 1]));
                        const s1 = src_f32[r * k + sb * 256 + (pair * 2) * 32 ..][0..32];
                        const s2 = src_f32[r * k + sb * 256 + (pair * 2 + 1) * 32 ..][0..32];
                        for (0..32) |l| {
                            const lo_n: u8 = if (d1 > 0.0) @intFromFloat(std.math.clamp(@round((s1[l] + m1) / d1), 0.0, 15.0)) else 0;
                            const hi_n: u8 = if (d2 > 0.0) @intFromFloat(std.math.clamp(@round((s2[l] + m2) / d2), 0.0, 15.0)) else 0;
                            qs[pair * 32 + l] = lo_n | (hi_n << 4);
                        }
                    }
                }
            }
            return .{ .data = data, .params = &.{}, .dtype = .gguf_q4_k, .quant_bits = 4, .group_size = 256 };
        },
        .gguf_q2_0 => {
            const blks = k / 64;
            const data = try allocator.alloc(u8, n * blks * 18);
            errdefer allocator.free(data);
            @memset(data, 0);
            for (0..n) |r| {
                for (0..blks) |b| {
                    const slice = src_f32[r * k + b * 64 ..][0..64];
                    var amax: f32 = 0.0;
                    for (slice) |v| amax = @max(amax, @abs(v));
                    const d_f16: f16 = @floatCast(if (amax > 0.0) amax / 1.5 else 0.0);
                    const d_eff: f32 = @floatCast(d_f16);
                    const blk = data[(r * blks + b) * 18 ..][0..18];
                    writeU16Le(blk[0..2], @bitCast(d_f16));
                    for (0..64) |j| {
                        // Decode formula: `(q - 1) * d` with `q in {0, 1, 2, 3}` -> `{-1, 0, +1, +2} * d`.
                        const qf = if (d_eff > 0.0) @round(slice[j] / d_eff) + 1.0 else 1.0;
                        const q2: u8 = @intFromFloat(std.math.clamp(qf, 0.0, 3.0));
                        blk[2 + j / 4] |= (q2 << @intCast((j % 4) * 2));
                    }
                }
            }
            return .{ .data = data, .params = &.{}, .dtype = .gguf_q2_0, .quant_bits = 2, .group_size = 64 };
        },
        else => return error.InvalidArgument,
    }
}

const BuiltEntry = struct {
    entry: amf.TensorEntry,
    blob: []u8,
};

fn buildAmfImageFromEntries(
    allocator: std.mem.Allocator,
    model_id: []const u8,
    items: []BuiltEntry,
) ![]u8 {
    const raw_dir_bytes = items.len * @sizeOf(amf.TensorEntry);
    const dir_bytes: u64 = if (raw_dir_bytes == 0) 0 else try page.alignUp(raw_dir_bytes);
    const dir_off: u64 = page.APPLE_PAGE_SIZE;
    var cur_off: u64 = dir_off + dir_bytes;

    for (items) |*it| {
        it.entry.blob_offset = cur_off;
        it.entry.blob_bytes = it.blob.len;
        it.entry.blob_crc32 = std.hash.Crc32.hash(it.blob);
        cur_off = try page.alignUp(cur_off + it.blob.len);
    }

    const img = try allocator.alloc(u8, @intCast(cur_off));
    errdefer allocator.free(img);
    @memset(img, 0);

    for (items, 0..) |it, idx| {
        const dst_e = img[@intCast(dir_off + idx * @sizeOf(amf.TensorEntry))..][0..@sizeOf(amf.TensorEntry)];
        @memcpy(dst_e, std.mem.asBytes(&it.entry));
        @memcpy(img[@intCast(it.entry.blob_offset)..][0..it.blob.len], it.blob);
    }

    const dir_slice = img[@intCast(dir_off)..][0..raw_dir_bytes];
    var hdr: amf.Header = std.mem.zeroes(amf.Header);
    hdr.magic = amf.MAGIC;
    hdr.version = amf.FORMAT_VERSION;
    hdr.flags = 0;
    hdr.page_size = page.APPLE_PAGE_SIZE;
    hdr.checksum_algo = amf.CHECKSUM_CRC32_IEEE;
    hdr.tensor_count = @intCast(items.len);
    hdr.directory_offset = dir_off;
    hdr.directory_bytes = dir_bytes;
    hdr.directory_crc32 = std.hash.Crc32.hash(dir_slice);
    const id_n = @min(model_id.len, hdr.model_id.len - 1);
    @memcpy(hdr.model_id[0..id_n], model_id[0..id_n]);
    hdr.header_crc32 = amf.computeHeaderCrc32(&hdr);
    @memcpy(img[0..@sizeOf(amf.Header)], std.mem.asBytes(&hdr));
    return img;
}

fn appendLinearEntry(
    allocator: std.mem.Allocator,
    list: *std.ArrayList(BuiltEntry),
    name: []const u8,
    kind: amf.TensorKind,
    layer: u16,
    expert: u16,
    n: usize,
    k: usize,
    seed: u64,
    scale: f32,
    dtype: amf.DType,
    group_size: u16,
) !void {
    const mat = try allocator.alloc(f32, n * k);
    defer allocator.free(mat);
    fillDeterministicMatrixF32(seed, n, k, scale, mat);

    const qb = try quantizeMatrixToAmf(allocator, mat, n, k, dtype, group_size);
    defer qb.deinit(allocator);

    const params_off: u64 = if (qb.params.len > 0) amf.align64(qb.data.len) else 0;
    const raw_len: usize = if (qb.params.len > 0) @intCast(params_off + qb.params.len) else qb.data.len;
    const blob_len: usize = @intCast(try page.alignUp(raw_len));
    const blob = try allocator.alloc(u8, blob_len);
    errdefer allocator.free(blob);
    @memset(blob, 0);
    @memcpy(blob[0..qb.data.len], qb.data);
    if (qb.params.len > 0) {
        @memcpy(blob[@intCast(params_off)..][0..qb.params.len], qb.params);
    }

    var entry = amf.TensorEntry.zeroed(name, kind, qb.dtype);
    entry.quant_bits = qb.quant_bits;
    entry.rank = 2;
    entry.group_size = qb.group_size;
    entry.layer = layer;
    entry.expert = expert;
    entry.shape = .{ @intCast(n), @intCast(k), 0, 0 };
    entry.data_bytes = qb.data.len;
    entry.params_offset = params_off;
    entry.params_bytes = qb.params.len;

    try list.append(allocator, .{ .entry = entry, .blob = blob });
}

fn appendNormEntry(
    allocator: std.mem.Allocator,
    list: *std.ArrayList(BuiltEntry),
    name: []const u8,
    layer: u16,
    dim: usize,
    seed: u64,
) !void {
    const raw_bytes = dim * 2;
    const blob_len: usize = @intCast(try page.alignUp(raw_bytes));
    const blob = try allocator.alloc(u8, blob_len);
    errdefer allocator.free(blob);
    @memset(blob, 0);
    var st = seed ^ 0x517CC1B727220A95;
    for (0..dim) |i| {
        st ^= st >> 12;
        st ^= st << 25;
        st ^= st >> 27;
        const u = (st *% 0x2545F4914F6CDD1D) >> 40;
        const w = 0.85 + (@as(f32, @floatFromInt(u)) / 16777216.0) * 0.30;
        writeF16Le(blob[i * 2 ..][0..2], w);
    }
    var entry = amf.TensorEntry.zeroed(name, .norm, .f16);
    entry.rank = 1;
    entry.layer = layer;
    entry.shape = .{ @intCast(dim), 0, 0, 0 };
    entry.data_bytes = raw_bytes;
    try list.append(allocator, .{ .entry = entry, .blob = blob });
}

pub fn syntheticMultiFormatDenseConfig() qwen3.ModelConfig {
    return .{
        .arch = .qwen3_dense,
        .vocab_size = 512,
        .hidden_size = 256,
        .num_hidden_layers = 2,
        .num_attention_heads = 4,
        .num_key_value_heads = 2,
        .head_dim = 64,
        .intermediate_size = 512,
        .moe_intermediate_size = 0,
        .num_experts = 0,
        .num_experts_per_tok = 0,
        .norm_topk_prob = false,
        .rms_norm_eps = 1e-6,
        .rope_theta = 10000.0,
        .max_position_embeddings = 256,
        .tie_word_embeddings = false,
    };
}

/// Builds a synthetic 2-layer `Qwen3-Dense` `.amf` image where all transformer
/// linear projections (`q/k/v/o`, `gate/up/down`) use `(linear_dtype, group_size)`
/// and `embed_tokens`/`lm_head` stay in `bf16` (matching `W21` high-precision
/// boundary protection) so KL divergence isolates the projection kernel cleanly.
pub fn buildSyntheticMultiFormatDenseAmf(
    allocator: std.mem.Allocator,
    cfg: qwen3.ModelConfig,
    linear_dtype: amf.DType,
    group_size: u16,
    embed_dtype_override: ?amf.DType,
    seed: u64,
) ![]u8 {
    var list: std.ArrayList(BuiltEntry) = .empty;
    defer {
        for (list.items) |it| allocator.free(it.blob);
        list.deinit(allocator);
    }

    const h: usize = cfg.hidden_size;
    const q_dim: usize = cfg.qDim();
    const kv_dim: usize = cfg.kvDim();
    const inter: usize = cfg.intermediate_size;
    const emb_dt = embed_dtype_override orelse .bf16;
    const emb_gs: u16 = switch (emb_dt) {
        .bf16, .f16, .f32 => 0,
        .q2_affine, .q3_affine, .q4_affine, .q5_affine, .q6_affine, .q8_affine => if (group_size == 32 or group_size == 64 or group_size == 128) group_size else 64,
        .nvfp4_g16 => 16,
        .mxfp8_g32, .gguf_q8_0, .gguf_q4_0 => 32,
        .gguf_q2_0 => 64,
        .fp8_e4m3_block => 128,
        .gguf_q4_k => 256,
        else => group_size,
    };

    try appendLinearEntry(allocator, &list, "embed_tokens.weight", .embedding, amf.NO_LAYER, amf.NO_EXPERT, cfg.vocab_size, h, seed + 1, 0.25, emb_dt, emb_gs);
    try appendNormEntry(allocator, &list, "norm.weight", amf.NO_LAYER, h, seed + 2);
    try appendLinearEntry(allocator, &list, "lm_head.weight", .lm_head, amf.NO_LAYER, amf.NO_EXPERT, cfg.vocab_size, h, seed + 3, 0.22, .bf16, 0);

    var nbuf: [64]u8 = undefined;
    for (0..cfg.num_hidden_layers) |l| {
        const ls = seed + 100 * (l + 1);
        const l16: u16 = @intCast(l);
        try appendNormEntry(allocator, &list, try std.fmt.bufPrint(&nbuf, "layers.{d}.input_layernorm.weight", .{l}), l16, h, ls + 1);
        try appendLinearEntry(allocator, &list, try std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.q_proj", .{l}), .dense, l16, amf.NO_EXPERT, q_dim, h, ls + 2, 0.14, linear_dtype, group_size);
        try appendLinearEntry(allocator, &list, try std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.k_proj", .{l}), .dense, l16, amf.NO_EXPERT, kv_dim, h, ls + 3, 0.14, linear_dtype, group_size);
        try appendLinearEntry(allocator, &list, try std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.v_proj", .{l}), .dense, l16, amf.NO_EXPERT, kv_dim, h, ls + 4, 0.14, linear_dtype, group_size);
        try appendLinearEntry(allocator, &list, try std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.o_proj", .{l}), .dense, l16, amf.NO_EXPERT, h, q_dim, ls + 5, 0.14, linear_dtype, group_size);
        try appendNormEntry(allocator, &list, try std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.q_norm.weight", .{l}), l16, cfg.head_dim, ls + 6);
        try appendNormEntry(allocator, &list, try std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.k_norm.weight", .{l}), l16, cfg.head_dim, ls + 7);
        try appendNormEntry(allocator, &list, try std.fmt.bufPrint(&nbuf, "layers.{d}.post_attention_layernorm.weight", .{l}), l16, h, ls + 8);
        try appendLinearEntry(allocator, &list, try std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.gate_proj", .{l}), .dense, l16, amf.NO_EXPERT, inter, h, ls + 9, 0.12, linear_dtype, group_size);
        try appendLinearEntry(allocator, &list, try std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.up_proj", .{l}), .dense, l16, amf.NO_EXPERT, inter, h, ls + 10, 0.12, linear_dtype, group_size);
        try appendLinearEntry(allocator, &list, try std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.down_proj", .{l}), .dense, l16, amf.NO_EXPERT, h, inter, ls + 11, 0.12, linear_dtype, group_size);
    }

    return buildAmfImageFromEntries(allocator, "synth-m81-dense", list.items);
}

pub fn syntheticMultiFormatMoeConfig() qwen3.ModelConfig {
    return .{
        .arch = .qwen3_moe,
        .vocab_size = 512,
        .hidden_size = 256,
        .num_hidden_layers = 1,
        .num_attention_heads = 4,
        .num_key_value_heads = 2,
        .head_dim = 64,
        .intermediate_size = 0,
        .moe_intermediate_size = 256,
        .num_experts = 4,
        .num_experts_per_tok = 2,
        .norm_topk_prob = true,
        .rms_norm_eps = 1e-6,
        .rope_theta = 10000.0,
        .max_position_embeddings = 128,
        .tie_word_embeddings = false,
    };
}

/// Builds a synthetic 1-layer `Qwen3-MoE` `.amf` exercising mixed-format dynamic
/// GGUF/NVFP4 experts (`e in {0,1}`: `nvfp4_g16` gate/up + `gguf_q2_0` down;
/// `e in {2,3}`: `gguf_q4_k` gate/up/down) + `fp8_e4m3_block` attention.
pub fn buildSyntheticMultiFormatMoeAmf(
    allocator: std.mem.Allocator,
    cfg: qwen3.ModelConfig,
    seed: u64,
) ![]u8 {
    var list: std.ArrayList(BuiltEntry) = .empty;
    defer {
        for (list.items) |it| allocator.free(it.blob);
        list.deinit(allocator);
    }

    const h: usize = cfg.hidden_size;
    const q_dim: usize = cfg.qDim();
    const kv_dim: usize = cfg.kvDim();
    const moe_i: usize = cfg.moe_intermediate_size;

    try appendLinearEntry(allocator, &list, "embed_tokens.weight", .embedding, amf.NO_LAYER, amf.NO_EXPERT, cfg.vocab_size, h, seed + 1, 0.25, .gguf_q8_0, 32);
    try appendNormEntry(allocator, &list, "norm.weight", amf.NO_LAYER, h, seed + 2);
    try appendLinearEntry(allocator, &list, "lm_head.weight", .lm_head, amf.NO_LAYER, amf.NO_EXPERT, cfg.vocab_size, h, seed + 3, 0.22, .mxfp8_g32, 32);

    var nbuf: [64]u8 = undefined;
    for (0..cfg.num_hidden_layers) |l| {
        const ls = seed + 200 * (l + 1);
        const l16: u16 = @intCast(l);
        try appendNormEntry(allocator, &list, try std.fmt.bufPrint(&nbuf, "layers.{d}.input_layernorm.weight", .{l}), l16, h, ls + 1);
        try appendLinearEntry(allocator, &list, try std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.q_proj", .{l}), .dense, l16, amf.NO_EXPERT, q_dim, h, ls + 2, 0.14, .fp8_e4m3_block, 128);
        try appendLinearEntry(allocator, &list, try std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.k_proj", .{l}), .dense, l16, amf.NO_EXPERT, kv_dim, h, ls + 3, 0.14, .fp8_e4m3_block, 128);
        try appendLinearEntry(allocator, &list, try std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.v_proj", .{l}), .dense, l16, amf.NO_EXPERT, kv_dim, h, ls + 4, 0.14, .fp8_e4m3_block, 128);
        try appendLinearEntry(allocator, &list, try std.fmt.bufPrint(&nbuf, "layers.{d}.self_attn.o_proj", .{l}), .dense, l16, amf.NO_EXPERT, h, q_dim, ls + 5, 0.14, .mxfp8_g32, 32);
        try appendNormEntry(allocator, &list, try std.fmt.bufPrint(&nbuf, "layers.{d}.post_attention_layernorm.weight", .{l}), l16, h, ls + 8);
        try appendLinearEntry(allocator, &list, try std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.gate", .{l}), .router, l16, amf.NO_EXPERT, cfg.num_experts, h, ls + 9, 0.20, .bf16, 0);

        for (0..cfg.num_experts) |e| {
            const e16: u16 = @intCast(e);
            const es = ls + 20 + e * 7;
            const gu_dt: amf.DType = if (e < 2) .nvfp4_g16 else .gguf_q4_k;
            const gu_gs: u16 = if (e < 2) 16 else 256;
            const dn_dt: amf.DType = if (e < 2) .gguf_q2_0 else .gguf_q4_k;
            const dn_gs: u16 = if (e < 2) 64 else 256;
            try appendLinearEntry(allocator, &list, try std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.experts.{d}.gate_proj", .{ l, e }), .hot_expert, l16, e16, moe_i, h, es + 1, 0.12, gu_dt, gu_gs);
            try appendLinearEntry(allocator, &list, try std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.experts.{d}.up_proj", .{ l, e }), .hot_expert, l16, e16, moe_i, h, es + 2, 0.12, gu_dt, gu_gs);
            try appendLinearEntry(allocator, &list, try std.fmt.bufPrint(&nbuf, "layers.{d}.mlp.experts.{d}.down_proj", .{ l, e }), .hot_expert, l16, e16, h, moe_i, es + 3, 0.12, dn_dt, dn_gs);
        }
    }

    return buildAmfImageFromEntries(allocator, "synth-m81-moe-hetero", list.items);
}

fn evaluateModelPairSequential(
    allocator: std.mem.Allocator,
    ref_model: *qwen3.Qwen3Model,
    cand_model: *qwen3.Qwen3Model,
    prompt_tokens: []const u32,
    steps: u32,
    res: *QualityGatePairResult,
) !void {
    const vocab: usize = ref_model.config.vocab_size;
    const ref_logits_all = try allocator.alloc(f32, @as(usize, steps) * vocab);
    defer allocator.free(ref_logits_all);
    const ref_greedy = try allocator.alloc(u32, steps);
    defer allocator.free(ref_greedy);

    // 1. Reference teacher-forced + greedy rollout.
    ref_model.resetKvCache();
    var pos: u32 = 0;
    var last_tok: u32 = prompt_tokens[0];
    for (prompt_tokens) |tid| {
        last_tok = try ref_model.stepDecode(tid, pos);
        pos += 1;
    }
    for (0..steps) |s| {
        const lg = ref_model.readLastLogitsF16();
        for (0..vocab) |v| ref_logits_all[s * vocab + v] = @floatCast(lg[v]);
        ref_greedy[s] = last_tok;
        if (s + 1 < steps) {
            last_tok = try ref_model.stepDecode(last_tok, pos);
            pos += 1;
        }
    }

    // 2. Candidate teacher-forced KL & top-1 agreement.
    cand_model.resetKvCache();
    pos = 0;
    var cand_tok: u32 = prompt_tokens[0];
    for (prompt_tokens) |tid| {
        cand_tok = try cand_model.stepDecode(tid, pos);
        pos += 1;
    }
    var sum_kl: f64 = 0.0;
    var max_kl: f64 = 0.0;
    var top1_matches: u32 = 0;
    var all_finite = true;
    const cand_tmp = try allocator.alloc(f32, vocab);
    defer allocator.free(cand_tmp);

    for (0..steps) |s| {
        const lg = cand_model.readLastLogitsF16();
        for (0..vocab) |v| cand_tmp[v] = @floatCast(lg[v]);
        const m = computeLogitsKlAndTop1F32(ref_logits_all[s * vocab ..][0..vocab], cand_tmp);
        if (!m.all_finite) all_finite = false;
        sum_kl += m.kl_divergence;
        if (m.kl_divergence > max_kl) max_kl = m.kl_divergence;
        if (m.top1_match) top1_matches += 1;
        if (s + 1 < steps) {
            cand_tok = try cand_model.stepDecode(ref_greedy[s], pos);
            pos += 1;
        }
    }

    // 3. Candidate free-running deterministic greedy rollout vs reference.
    cand_model.resetKvCache();
    pos = 0;
    var free_tok: u32 = prompt_tokens[0];
    for (prompt_tokens) |tid| {
        free_tok = try cand_model.stepDecode(tid, pos);
        pos += 1;
    }
    var greedy_matches: u32 = 0;
    for (0..steps) |s| {
        if (free_tok == ref_greedy[s]) greedy_matches += 1;
        if (s + 1 < steps) {
            free_tok = try cand_model.stepDecode(free_tok, pos);
            pos += 1;
        }
    }

    res.steps_evaluated = steps;
    res.top1_matches = top1_matches;
    res.top1_agreement_pct = 100.0 * @as(f64, @floatFromInt(top1_matches)) / @as(f64, @floatFromInt(steps));
    res.mean_kl = sum_kl / @as(f64, @floatFromInt(steps));
    res.max_kl = max_kl;
    res.greedy_rollout_steps = steps;
    res.greedy_rollout_matches = greedy_matches;
    res.all_finite = all_finite;
    res.executed = true;
    res.status = "ok";
}

/// Runs `W24` quality gate comparing two on-disk `.amf` files sequentially so
/// peak wired RAM never exceeds `max(ref, cand)` (respecting `R9` live headroom).
pub fn runFileQualityGate(
    allocator: std.mem.Allocator,
    ref_amf_path: [:0]const u8,
    cand_amf_path: [:0]const u8,
    config_path: ?[:0]const u8,
    tokenizer_path: ?[:0]const u8,
    act_quant_policy: quant_formats.ActQuantPolicy,
    steps_in: u32,
    max_seq_len: u32,
    explicit_tokens: ?[]const u32,
) !QualityGatePairResult {
    var res: QualityGatePairResult = .{
        .act_quant = act_quant_policy,
        .status = "skipped_missing",
    };
    var dyn_label_buf: [64]u8 = undefined;
    const ref_stem = std.fs.path.stem(ref_amf_path);
    const cand_stem = std.fs.path.stem(cand_amf_path);
    const dyn_label = std.fmt.bufPrint(&dyn_label_buf, "{s}_vs_{s}", .{ ref_stem, cand_stem }) catch ref_stem;
    res.setLabel(dyn_label);

    // Check file existence cleanly before touching Metal or wired memory.
    var ref_file = amf.AmfFile.open(allocator, ref_amf_path) catch {
        res.status = "skipped_ref_missing";
        return res;
    };
    const ref_file_bytes = ref_file.parsed.file_size;
    var cfg_meta = amf.resolveConfigJsonAlloc(
        allocator,
        &ref_file,
        ref_amf_path,
        if (config_path) |cp| @as([]const u8, cp) else null,
    ) catch {
        ref_file.close(allocator);
        res.status = "skipped_config_missing";
        return res;
    };
    defer cfg_meta.deinit(allocator);

    const tok_path_opt: ?[]const u8 = if (tokenizer_path) |tp| @as([]const u8, tp) else null;
    var tok_meta_opt: ?amf.ResolvedMetadata = if (amf.resolveTokenizerJsonAlloc(
        allocator,
        &ref_file,
        ref_amf_path,
        tok_path_opt,
    )) |tm| tm else |err| switch (err) {
        error.OutOfMemory => {
            ref_file.close(allocator);
            return error.OutOfMemory;
        },
        else => null,
    };
    defer if (tok_meta_opt) |*tm| tm.deinit(allocator);
    ref_file.close(allocator);

    var cand_file = amf.AmfFile.open(allocator, cand_amf_path) catch {
        res.status = "skipped_cand_missing";
        return res;
    };
    const cand_file_bytes = cand_file.parsed.file_size;
    if (tok_meta_opt == null) {
        tok_meta_opt = if (amf.resolveTokenizerJsonAlloc(
            allocator,
            &cand_file,
            cand_amf_path,
            tok_path_opt,
        )) |tm| tm else |err| switch (err) {
            error.OutOfMemory => {
                cand_file.close(allocator);
                return error.OutOfMemory;
            },
            else => null,
        };
    }
    cand_file.close(allocator);

    // R9 live zero-swap RAM headroom check (`>= max(ref, cand) + 1.0 GiB`).
    const vm_now = try vmstat.Snapshot.capture();
    const live_avail = vm_now.liveZeroSwapReclaimableBytes();
    const peak_need = @max(ref_file_bytes, cand_file_bytes) + (1024 * 1024 * 1024);
    if (live_avail < peak_need) {
        res.status = "skipped_low_live_ram";
        return res;
    }

    const cfg = try qwen3.ModelConfig.fromJson(allocator, cfg_meta.bytes);
    const steps: u32 = @max(@as(u32, 1), @min(steps_in, 16));
    const seq_cap: u32 = @min(cfg.max_position_embeddings, @max(max_seq_len, 64));

    // Resolve deterministic multi-prompt math + code eval tokens (via explicit `-k`,
    // embedded `__meta.tokenizer_json`, or sidecar `tokenizer.json` before fallback).
    var owned_math_prompt: ?[]u32 = null;
    defer if (owned_math_prompt) |op| allocator.free(op);
    var owned_code_prompt: ?[]u32 = null;
    defer if (owned_code_prompt) |op| allocator.free(op);

    const fallback_math = [_]u32{ 1, 17, 23, 42, 100 };
    const fallback_code = [_]u32{ 2, 11, 29, 53, 89 };
    var math_slice: []const u32 = explicit_tokens orelse &fallback_math;
    var code_slice: []const u32 = explicit_tokens orelse &fallback_code;

    if (explicit_tokens == null) {
        if (tok_meta_opt) |tm| {
            if (tokenizer_mod.Tokenizer.fromHuggingFaceJson(allocator, tm.bytes)) |tok_val| {
                var tok = tok_val;
                defer tok.deinit(allocator);
                res.tokenizer_resolved = true;
                var scratch = try tokenizer_mod.EncodeScratch.init(allocator, 4096);
                defer scratch.deinit(allocator);
                var ids_buf: [64]u32 = undefined;
                const math_text = "Compute 17 * 23 = ";
                if (tok.encodeWithScratch(&scratch, math_text, &ids_buf)) |encoded| {
                    if (encoded.len > 0) {
                        const copy = try allocator.alloc(u32, encoded.len);
                        @memcpy(copy, encoded);
                        owned_math_prompt = copy;
                        math_slice = copy;
                    }
                } else |err| switch (err) {
                    error.OutOfMemory => return error.OutOfMemory,
                    else => {},
                }
                const code_text = "def fib(n):\n    if n <= 1:\n        return n\n    return ";
                if (tok.encodeWithScratch(&scratch, code_text, &ids_buf)) |encoded| {
                    if (encoded.len > 0) {
                        const copy = try allocator.alloc(u32, encoded.len);
                        @memcpy(copy, encoded);
                        owned_code_prompt = copy;
                        code_slice = copy;
                    }
                } else |err| switch (err) {
                    error.OutOfMemory => return error.OutOfMemory,
                    else => {},
                }
            } else |err| switch (err) {
                error.OutOfMemory => return error.OutOfMemory,
                else => {},
            }
        }
    }

    const math_steps: u32 = if (explicit_tokens != null or steps == 1) steps else (steps + 1) / 2;
    const code_steps: u32 = steps - math_steps;
    const prompt_slices = [_][]const u32{ math_slice, code_slice };
    const prompt_steps = [_]u32{ math_steps, code_steps };
    const prompt_count: usize = if (code_steps > 0) 2 else 1;

    const vocab: usize = cfg.vocab_size;
    const ref_logits_all = try allocator.alloc(f32, @as(usize, steps) * vocab);
    defer allocator.free(ref_logits_all);
    const ref_greedy = try allocator.alloc(u32, steps);
    defer allocator.free(ref_greedy);

    // Phase 1: Load reference, record teacher-forced logits & greedy rollouts across
    // math + code prompts, then DEINIT immediately!
    {
        var ref_model = try qwen3.Qwen3Model.loadFromFile(allocator, ref_amf_path, cfg, .{
            .max_seq_len = seq_cap,
            .max_tree_nodes = 1,
            .verify_crcs = false,
            .startup_canary = false,
            .gpu_prefill_mode = .tree64,
        });
        defer ref_model.deinit(allocator);

        var step_off: usize = 0;
        for (0..prompt_count) |pi| {
            ref_model.resetKvCache();
            const pslice = prompt_slices[pi];
            const psteps = prompt_steps[pi];
            var pos: u32 = 0;
            var last_tok: u32 = pslice[0];
            for (pslice) |tid| {
                last_tok = try ref_model.stepDecode(tid, pos);
                pos += 1;
            }
            for (0..psteps) |s| {
                const lg = ref_model.readLastLogitsF16();
                for (0..vocab) |v| ref_logits_all[(step_off + s) * vocab + v] = @floatCast(lg[v]);
                ref_greedy[step_off + s] = last_tok;
                if (s + 1 < psteps) {
                    last_tok = try ref_model.stepDecode(last_tok, pos);
                    pos += 1;
                }
            }
            step_off += psteps;
        }
    }

    // Phase 2: Load candidate sequentially (zero overlap with ref wired memory!).
    {
        var cand_model = try qwen3.Qwen3Model.loadFromFile(allocator, cand_amf_path, cfg, .{
            .max_seq_len = seq_cap,
            .max_tree_nodes = 1,
            .verify_crcs = false,
            .startup_canary = false,
            .gpu_prefill_mode = .tree64,
            .act_quant_policy = act_quant_policy,
        });
        defer cand_model.deinit(allocator);

        var sum_kl: f64 = 0.0;
        var max_kl: f64 = 0.0;
        var top1_matches: u32 = 0;
        var greedy_matches: u32 = 0;
        var math_matches: u32 = 0;
        var code_matches: u32 = 0;
        var all_finite = true;
        const cand_tmp = try allocator.alloc(f32, vocab);
        defer allocator.free(cand_tmp);

        var step_off: usize = 0;
        for (0..prompt_count) |pi| {
            const pslice = prompt_slices[pi];
            const psteps = prompt_steps[pi];

            // Teacher-forced KL divergence & top-1 agreement on this prompt.
            cand_model.resetKvCache();
            var pos: u32 = 0;
            var cand_tok: u32 = pslice[0];
            for (pslice) |tid| {
                cand_tok = try cand_model.stepDecode(tid, pos);
                pos += 1;
            }
            for (0..psteps) |s| {
                const lg = cand_model.readLastLogitsF16();
                for (0..vocab) |v| cand_tmp[v] = @floatCast(lg[v]);
                const m = computeLogitsKlAndTop1F32(ref_logits_all[(step_off + s) * vocab ..][0..vocab], cand_tmp);
                if (!m.all_finite) all_finite = false;
                sum_kl += m.kl_divergence;
                if (m.kl_divergence > max_kl) max_kl = m.kl_divergence;
                if (m.top1_match) top1_matches += 1;
                if (s + 1 < psteps) {
                    cand_tok = try cand_model.stepDecode(ref_greedy[step_off + s], pos);
                    pos += 1;
                }
            }

            // Free-running greedy rollout parity on this prompt.
            cand_model.resetKvCache();
            pos = 0;
            var free_tok: u32 = pslice[0];
            for (pslice) |tid| {
                free_tok = try cand_model.stepDecode(tid, pos);
                pos += 1;
            }
            var p_greedy: u32 = 0;
            for (0..psteps) |s| {
                if (free_tok == ref_greedy[step_off + s]) p_greedy += 1;
                if (s + 1 < psteps) {
                    free_tok = try cand_model.stepDecode(free_tok, pos);
                    pos += 1;
                }
            }
            greedy_matches += p_greedy;
            if (pi == 0) {
                math_matches = p_greedy;
            } else {
                code_matches = p_greedy;
            }
            step_off += psteps;
        }

        res.steps_evaluated = steps;
        res.top1_matches = top1_matches;
        res.top1_agreement_pct = 100.0 * @as(f64, @floatFromInt(top1_matches)) / @as(f64, @floatFromInt(steps));
        res.mean_kl = sum_kl / @as(f64, @floatFromInt(steps));
        res.max_kl = max_kl;
        res.greedy_rollout_steps = steps;
        res.greedy_rollout_matches = greedy_matches;
        res.math_greedy_steps = math_steps;
        res.math_greedy_matches = math_matches;
        res.code_greedy_steps = code_steps;
        res.code_greedy_matches = code_matches;
        res.all_finite = all_finite;
        res.executed = true;
        res.status = "ok";
    }

    return res;
}

const SynthGateSpec = struct {
    name: []const u8,
    dtype: amf.DType,
    group_size: u16,
    act_quant: quant_formats.ActQuantPolicy = .none,
};

const DEFAULT_SYNTH_SPECS = [_]SynthGateSpec{
    .{ .name = "fp8_e4m3_block", .dtype = .fp8_e4m3_block, .group_size = 128 },
    .{ .name = "mxfp8_g32", .dtype = .mxfp8_g32, .group_size = 32 },
    .{ .name = "q8_affine_g64", .dtype = .q8_affine, .group_size = 64 },
    .{ .name = "q6_affine_g64", .dtype = .q6_affine, .group_size = 64 },
    .{ .name = "q5_affine_g64", .dtype = .q5_affine, .group_size = 64 },
    .{ .name = "q4_affine_g32", .dtype = .q4_affine, .group_size = 32 },
    .{ .name = "nvfp4_g16", .dtype = .nvfp4_g16, .group_size = 16 },
    .{ .name = "nvfp4_g16_aq46", .dtype = .nvfp4_g16, .group_size = 16, .act_quant = .fouroversix_mae },
    .{ .name = "gguf_q8_0", .dtype = .gguf_q8_0, .group_size = 32 },
    .{ .name = "gguf_q4_k", .dtype = .gguf_q4_k, .group_size = 256 },
    .{ .name = "gguf_q2_0", .dtype = .gguf_q2_0, .group_size = 64 },
};

pub fn runQualityGate(allocator: std.mem.Allocator, cfg: QualityGateConfig) !QualityGateReport {
    var report: QualityGateReport = .{};
    report.vm_before = try vmstat.Snapshot.capture();

    if (cfg.include_synthetic_sweep) {
        const scfg = syntheticMultiFormatDenseConfig();
        const seed: u64 = 0x81_2026_0001;
        const ref_bytes = try buildSyntheticMultiFormatDenseAmf(allocator, scfg, .bf16, 0, null, seed);
        defer allocator.free(ref_bytes);

        var ref_model = try qwen3.Qwen3Model.loadFromBytes(allocator, ref_bytes, scfg, .{
            .max_seq_len = 64,
            .max_tree_nodes = 1,
            .verify_crcs = true,
            .startup_canary = false,
            .gpu_prefill_mode = .tree64,
        });
        defer ref_model.deinit(allocator);

        const prompt = [_]u32{ 7, 19, 42, 83, 129, 211 };
        const steps: u32 = @max(@as(u32, 1), @min(cfg.steps, 8));

        for (DEFAULT_SYNTH_SPECS) |sp| {
            const cand_bytes = try buildSyntheticMultiFormatDenseAmf(allocator, scfg, sp.dtype, sp.group_size, null, seed);
            defer allocator.free(cand_bytes);
            var cand_model = try qwen3.Qwen3Model.loadFromBytes(allocator, cand_bytes, scfg, .{
                .max_seq_len = 64,
                .max_tree_nodes = 1,
                .verify_crcs = true,
                .startup_canary = false,
                .gpu_prefill_mode = .tree64,
                .act_quant_policy = sp.act_quant,
            });
            defer cand_model.deinit(allocator);

            var pair: QualityGatePairResult = .{
                .dtype = sp.dtype,
                .group_size = sp.group_size,
                .act_quant = sp.act_quant,
            };
            pair.setLabel(sp.name);
            try evaluateModelPairSequential(allocator, &ref_model, &cand_model, &prompt, steps, &pair);
            report.synth_pairs[report.synth_count] = pair;
            report.synth_count += 1;
        }
    }

    if (cfg.ref_amf_path != null and cfg.cand_amf_path != null) {
        report.file_pair = try runFileQualityGate(
            allocator,
            cfg.ref_amf_path.?,
            cfg.cand_amf_path.?,
            cfg.config_path,
            cfg.tokenizer_path,
            cfg.act_quant_policy,
            cfg.steps,
            cfg.max_seq_len,
            cfg.prompt_tokens,
        );
    }

    report.vm_after = try vmstat.Snapshot.capture();
    try vmstat.checkNoSwap(report.vm_before, report.vm_after);
    return report;
}

// ============================================================================
// Milestone M8.1 Full Benchmark Suite (`aminfer bench-m81`)
// ============================================================================

pub const M81Config = struct {
    gemv_n: u32 = 4096,
    gemv_k: u32 = 4096,
    gemv_min_working_set_bytes: usize = 64 * 1024 * 1024,
    gemv_warmup_batches: u32 = 3,
    gemv_timed_batches: u32 = 6,
    gemv_dispatches_per_batch: u32 = 24,
    prefill_m: u32 = 64,
    prefill_n: u32 = 2048,
    prefill_k: u32 = 2048,
    prefill_iters: u32 = 6,
    act_quant_k: u32 = 4096,
    act_quant_iters: u32 = 16,
    sme_iters: u32 = 8,
    quality_steps: u32 = 8,
    skip_real_checkpoints: bool = false,
    bus_bandwidth_gbs_override: ?f64 = null,
};

pub const G23GemvRow = struct {
    label: []const u8 = "",
    dtype: quant_formats.DType = .q4_affine,
    group_size: u16 = 64,
    gbs: f64 = 0.0,
    bus_pct: f64 = 0.0,
    avg_us: f64 = 0.0,
};

pub const G23PrefillRow = struct {
    label: []const u8 = "",
    dtype: quant_formats.DType = .nvfp4_g16,
    group_size: u16 = 16,
    avg_ms: f64 = 0.0,
    tflops: f64 = 0.0,
};

pub const G24ActRow = struct {
    mode: quant_formats.ActQuantMode = .none,
    m1_us: f64 = 0.0,
    m16_us: f64 = 0.0,
    mae_vs_fp32: f64 = 0.0,
};

pub const C25SmeRow = struct {
    label: []const u8 = "",
    dtype: sme.PanelDType = .nvfp4_g16,
    m4_gops: f64 = 0.0,
    m16_gops: f64 = 0.0,
    parity_max_err: f32 = 0.0,
    parity_ok: bool = false,
};

pub const MAX_G23_GEMV_ROWS: usize = 16;
pub const MAX_G23_PREFILL_ROWS: usize = 4;
pub const MAX_C25_ROWS: usize = 6;

pub const M81Report = struct {
    chip_profile: chip_profile_mod.ChipProfile = undefined,
    gpu_name_buf: [128]u8 = @splat(0),
    gpu_name_len: usize = 0,

    // Section 1: G23 M=1 GEMV GB/s + M=64 tiled fp16 prefill
    baseline_q4_gbs: f64 = 0.0,
    baseline_q4_bus_pct: f64 = 0.0,
    g23_gemv_count: usize = 0,
    g23_gemv: [MAX_G23_GEMV_ROWS]G23GemvRow = @splat(.{}),
    g23_prefill_count: usize = 0,
    g23_prefill: [MAX_G23_PREFILL_ROWS]G23PrefillRow = @splat(.{}),

    // Section 2: G24 activation quantization emulation
    g24_none_dispatches: u32 = 0,
    g24_rows: [5]G24ActRow = @splat(.{}),
    g24_fouroversix_win_pct: f64 = 0.0,
    g24_fouroversix_mae_reduction_pct: f64 = 0.0,

    // Section 3: C25 SME2 microkernels + C13 bus contention gate
    c25_has_sme2: bool = false,
    c25_rows: [MAX_C25_ROWS]C25SmeRow = @splat(.{}),
    c25_c13_bus_gate_ok: bool = false,

    // Section 4: W21/W22 end-to-end model forward/decode/tree/prefill parity + fork guard
    w21_dense_decode_tree_prefill_ok: bool = false,
    w22_moe_asymmetric_hetero_ok: bool = false,
    w22_draft_quant_ok: bool = false,
    w22_fork_collision_rejected: bool = false,

    // Section 5: W24 + C18 + C17 tokenizer MB/s, ChatTemplate render us, HTTP/SSE checks, KL gate
    w24_inkling2_live: bool = false,
    w24_inkling2_mbs: f64 = 0.0,
    w24_hy4_live: bool = false,
    w24_hy4_mbs: f64 = 0.0,
    w24_chat_tpl_chatml_us: f64 = 0.0,
    w24_chat_tpl_inkling2_us: f64 = 0.0,
    w24_chat_tpl_hy4_us: f64 = 0.0,
    c18_keepalive_reuse_ok: bool = false,
    c18_hmac_sig_and_tamper_400_ok: bool = false,
    c18_response_completed_output_ok: bool = false,
    quality_gate: QualityGateReport = .{},

    vm_before: vmstat.Snapshot = undefined,
    vm_after: vmstat.Snapshot = undefined,

    pub fn gpuName(self: *const M81Report) []const u8 {
        return self.gpu_name_buf[0..self.gpu_name_len];
    }

    pub fn write(self: *const M81Report, w: *Io.Writer) !void {
        try w.print(
            \\bench.m81.chip                   {s} ({s}, spec_bus={d:.1} GB/s)
            \\bench.m81.swapouts_delta         {d}
            \\bench.m81.g23.baseline.q4_g64    {d:.2} GB/s ({d:.1}% bus)
            \\
        , .{
            self.gpuName(),
            @tagName(self.chip_profile.generation),
            self.chip_profile.spec_bus_gbs,
            vmstat.Snapshot.swapoutDelta(self.vm_before, self.vm_after),
            self.baseline_q4_gbs,
            self.baseline_q4_bus_pct,
        });
        for (self.g23_gemv[0..self.g23_gemv_count]) |row| {
            try w.print(
                "bench.m81.g23.gemv.{s:<16} {d:.2} GB/s ({d:.1}% bus, {d:.2} us)\n",
                .{ row.label, row.gbs, row.bus_pct, row.avg_us },
            );
        }
        for (self.g23_prefill[0..self.g23_prefill_count]) |row| {
            try w.print(
                "bench.m81.g23.prefill64.{s:<11} {d:.3} ms ({d:.2} TFLOPS)\n",
                .{ row.label, row.avg_ms, row.tflops },
            );
        }
        try w.print("bench.m81.g24.none_dispatches    {d}\n", .{self.g24_none_dispatches});
        for (self.g24_rows) |ar| {
            try w.print(
                "bench.m81.g24.act.{s:<17} m1={d:.2} us m16={d:.2} us mae={d:.6}\n",
                .{ ar.mode.label(), ar.m1_us, ar.m16_us, ar.mae_vs_fp32 },
            );
        }
        try w.print(
            "bench.m81.g24.fouroversix        win_rate={d:.1}% mae_reduction={d:.2}%\n",
            .{ self.g24_fouroversix_win_pct, self.g24_fouroversix_mae_reduction_pct },
        );
        try w.print(
            "bench.m81.c25.sme2_hw            {s} (c13_bus_gate={s})\n",
            .{ if (self.c25_has_sme2) "active" else "scalar_fallback", if (self.c25_c13_bus_gate_ok) "pass" else "fail" },
        );
        for (self.c25_rows) |sr| {
            try w.print(
                "bench.m81.c25.sme2.{s:<16} m4={d:.2} GOPS m16={d:.2} GOPS max_err={e:.2} parity={s}\n",
                .{ sr.label, sr.m4_gops, sr.m16_gops, sr.parity_max_err, if (sr.parity_ok) "pass" else "fail" },
            );
        }
        try w.print(
            \\bench.m81.w21.dense_e2e_parity   {s}
            \\bench.m81.w22.moe_hetero_parity  {s}
            \\bench.m81.w22.draft_quant_parity {s}
            \\bench.m81.w22.fork_stq1_reject   {s}
            \\bench.m81.w24.tok.inkling2       {d:.2} MB/s (source={s})
            \\bench.m81.w24.tok.hy4            {d:.2} MB/s (source={s})
            \\bench.m81.w24.tpl_render_us      chatml={d:.2} inkling2={d:.2} hy4={d:.2}
            \\bench.m81.c18.http_hardening     keepalive={s} hmac_sig_400={s} resp_completed_output={s}
            \\
        , .{
            if (self.w21_dense_decode_tree_prefill_ok) "pass" else "fail",
            if (self.w22_moe_asymmetric_hetero_ok) "pass" else "fail",
            if (self.w22_draft_quant_ok) "pass" else "fail",
            if (self.w22_fork_collision_rejected) "pass" else "fail",
            self.w24_inkling2_mbs,
            if (self.w24_inkling2_live) "live_metadata" else "inline_fixture",
            self.w24_hy4_mbs,
            if (self.w24_hy4_live) "live_metadata" else "inline_fixture",
            self.w24_chat_tpl_chatml_us,
            self.w24_chat_tpl_inkling2_us,
            self.w24_chat_tpl_hy4_us,
            if (self.c18_keepalive_reuse_ok) "pass" else "fail",
            if (self.c18_hmac_sig_and_tamper_400_ok) "pass" else "fail",
            if (self.c18_response_completed_output_ok) "pass" else "fail",
        });
        try self.quality_gate.write(w);
    }
};

fn readSmallFileAlloc(allocator: std.mem.Allocator, path: [:0]const u8, max_bytes: usize) ![]u8 {
    const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.FileNotFound;
    defer _ = std.c.close(fd);

    var st: std.c.Stat = undefined;
    if (std.c.fstat(fd, &st) != 0) return error.StatFailed;
    if (st.size <= 0 or @as(u64, @intCast(st.size)) > max_bytes) return error.FileTooLarge;
    const len: usize = @intCast(st.size);
    const buf = try allocator.alloc(u8, len);
    errdefer allocator.free(buf);
    var off: usize = 0;
    while (off < len) {
        const n = std.c.read(fd, buf.ptr + off, len - off);
        if (n <= 0) return error.ReadFailed;
        off += @intCast(n);
    }
    return buf;
}

// --- Section 1: G23 M=1 GEMV GB/s + M=64 Tiled fp16 Prefill ---
fn benchM81G23(
    device: metal.Device,
    library: metal.Library,
    compute: *metal.Compute,
    dequant_pipes: dequant.DequantPipelines,
    spec_bus_gbs: f64,
    cfg: M81Config,
    report: *M81Report,
) !void {
    const G23Entry = struct { label: []const u8, dtype: quant_formats.DType, gs: u16 };
    const entries = [_]G23Entry{
        .{ .label = "q4_affine_g64", .dtype = .q4_affine, .gs = 64 },
        .{ .label = "q4_affine_g32", .dtype = .q4_affine, .gs = 32 },
        .{ .label = "q5_affine_g64", .dtype = .q5_affine, .gs = 64 },
        .{ .label = "q6_affine_g64", .dtype = .q6_affine, .gs = 64 },
        .{ .label = "q8_affine_g64", .dtype = .q8_affine, .gs = 64 },
        .{ .label = "fp8_e4m3_block", .dtype = .fp8_e4m3_block, .gs = 128 },
        .{ .label = "mxfp8_g32", .dtype = .mxfp8_g32, .gs = 32 },
        .{ .label = "nvfp4_g16", .dtype = .nvfp4_g16, .gs = 16 },
        .{ .label = "gguf_q8_0", .dtype = .gguf_q8_0, .gs = 32 },
        .{ .label = "gguf_q4_0", .dtype = .gguf_q4_0, .gs = 32 },
        .{ .label = "gguf_q4_k", .dtype = .gguf_q4_k, .gs = 256 },
        .{ .label = "gguf_q2_0", .dtype = .gguf_q2_0, .gs = 64 },
        .{ .label = "gguf_iq2_xxs", .dtype = .gguf_iq2_xxs, .gs = 256 },
        .{ .label = "gguf_tq1_0", .dtype = .gguf_tq1_0, .gs = 256 },
    };

    var specs: [entries.len]quant_formats.FormatSpec = undefined;
    for (entries, 0..) |e, i| specs[i] = quant_formats.FormatSpec.normalized(e.dtype, e.gs);
    var qf = try quant_formats.QuantFormatPipelines.initWithOptions(device, library, .{
        .specs = &specs,
        .include_moe = false,
        .include_act_quant = false,
    });
    defer qf.close();
    try qf.makeResident(compute.*);
    defer compute.removeResident(&.{qf.codebook_buf}) catch |err| {
        std.debug.panic("removeResident failed in benchM81G23: {}", .{err});
    };

    // Baseline: legacy `dequant.q4_affine` (`g64`) on exact same `(n, k)` working set.
    {
        const Q4 = quant.QuantOps(.{ .bits = .q4, .group_size = 64 });
        const w_bytes = try Q4.packedWeightBytes(cfg.gemv_n, cfg.gemv_k);
        const sb_bytes = try Q4.scaleBiasBytes(cfg.gemv_n, cfg.gemv_k);
        const slot_stride = amf.align64(w_bytes) + amf.align64(sb_bytes);
        const slots = @max(@as(usize, 2), (cfg.gemv_min_working_set_bytes + slot_stride - 1) / slot_stride);
        const x_off = amf.align64(slots * slot_stride);
        const y_off = x_off + amf.align64(@as(usize, cfg.gemv_k) * 2);
        const total = y_off + amf.align64(@as(usize, cfg.gemv_n) * 2);
        const region = try page.WiredRegion.init(total);
        defer region.deinit();
        @memset(region.bytes, 0x11);
        const buf = try metal.Buffer.wrap(device, region.bytes);
        defer buf.close();
        try compute.makeResident(&.{buf});
        defer compute.removeResident(&.{buf}) catch |err| {
            std.debug.panic("removeResident failed in benchM81G23 baseline: {}", .{err});
        };

        for (0..cfg.gemv_warmup_batches) |_| {
            var frame = try compute.beginFrame(30_000);
            errdefer frame.abort();
            for (0..cfg.gemv_dispatches_per_batch) |d| {
                const s = d % slots;
                try dequant_pipes.encodeGemv(&frame, .q4, .{
                    .x = .{ .buffer = buf, .offset = x_off },
                    .qweight = .{ .buffer = buf, .offset = s * slot_stride },
                    .scale_bias = .{ .buffer = buf, .offset = s * slot_stride + amf.align64(w_bytes) },
                    .y = .{ .buffer = buf, .offset = y_off },
                    .n = cfg.gemv_n,
                    .k = cfg.gemv_k,
                }, d > 0);
            }
            try compute.wait(try frame.commit(), 30_000);
        }
        const t0 = try vmstat.nowNs();
        for (0..cfg.gemv_timed_batches) |_| {
            var frame = try compute.beginFrame(30_000);
            errdefer frame.abort();
            for (0..cfg.gemv_dispatches_per_batch) |d| {
                const s = d % slots;
                try dequant_pipes.encodeGemv(&frame, .q4, .{
                    .x = .{ .buffer = buf, .offset = x_off },
                    .qweight = .{ .buffer = buf, .offset = s * slot_stride },
                    .scale_bias = .{ .buffer = buf, .offset = s * slot_stride + amf.align64(w_bytes) },
                    .y = .{ .buffer = buf, .offset = y_off },
                    .n = cfg.gemv_n,
                    .k = cfg.gemv_k,
                }, d > 0);
            }
            try compute.wait(try frame.commit(), 30_000);
        }
        const elapsed = (try vmstat.nowNs()) -| t0;
        const bpd: u64 = w_bytes + sb_bytes + @as(u64, cfg.gemv_k) * 2 + @as(u64, cfg.gemv_n) * 2;
        const disps: u64 = @as(u64, cfg.gemv_timed_batches) * cfg.gemv_dispatches_per_batch;
        report.baseline_q4_gbs = if (elapsed > 0) (@as(f64, @floatFromInt(bpd * disps)) / @as(f64, @floatFromInt(elapsed))) else 0.0;
        report.baseline_q4_bus_pct = (report.baseline_q4_gbs / spec_bus_gbs) * 100.0;
    }

    for (entries) |e| {
        const res = try quant_formats.measureGemvBandwidthGbs(device, library, &qf, .{
            .dtype = e.dtype,
            .group_size = e.gs,
            .n = cfg.gemv_n,
            .k = cfg.gemv_k,
            .min_working_set_bytes = cfg.gemv_min_working_set_bytes,
            .warmup_batches = cfg.gemv_warmup_batches,
            .timed_batches = cfg.gemv_timed_batches,
            .dispatches_per_batch = cfg.gemv_dispatches_per_batch,
            .spec_bus_gbs = spec_bus_gbs,
        });
        report.g23_gemv[report.g23_gemv_count] = .{
            .label = e.label,
            .dtype = e.dtype,
            .group_size = e.gs,
            .gbs = res.bandwidthGbs(),
            .bus_pct = res.busUtilizationPct(),
            .avg_us = res.avgLatencyUs(),
        };
        report.g23_gemv_count += 1;
    }

    // M = 64 tiled fp16 wide prefill (`encodeDequantTileF16` + `encodeDenseGemm(.f16, wide_tile=true)`).
    const prefill_specs = [_]G23Entry{
        .{ .label = "nvfp4_g16", .dtype = .nvfp4_g16, .gs = 16 },
        .{ .label = "mxfp8_g32", .dtype = .mxfp8_g32, .gs = 32 },
        .{ .label = "fp8_block", .dtype = .fp8_e4m3_block, .gs = 128 },
        .{ .label = "gguf_q4_k", .dtype = .gguf_q4_k, .gs = 256 },
    };
    const pm = cfg.prefill_m;
    const pn = cfg.prefill_n;
    const pk = cfg.prefill_k;
    const tile_bytes = amf.align64(@as(usize, pn) * pk * 2);
    const x_bytes = amf.align64(@as(usize, pm) * pk * 2);
    const y_bytes = amf.align64(@as(usize, pm) * pn * 2);
    const max_w = amf.align64(@as(usize, pn) * pk * 2);
    const max_p = amf.align64(@as(usize, pn) * pk);
    const pf_region = try page.WiredRegion.init(max_w + max_p + tile_bytes + x_bytes + y_bytes);
    defer pf_region.deinit();
    @memset(pf_region.bytes, 0);
    const pf_buf = try metal.Buffer.wrap(device, pf_region.bytes);
    defer pf_buf.close();
    try compute.makeResident(&.{pf_buf});
    defer compute.removeResident(&.{pf_buf}) catch |err| {
        std.debug.panic("removeResident failed in benchM81G23 prefill: {}", .{err});
    };

    const w_off: u64 = 0;
    const p_off: u64 = max_w;
    const tile_off: u64 = p_off + max_p;
    const x_off: u64 = tile_off + tile_bytes;
    const y_off: u64 = x_off + x_bytes;
    const iters: usize = @max(@as(usize, 1), cfg.prefill_iters);

    for (prefill_specs) |pe| {
        for (0..2) |_| {
            var frame = try compute.beginFrame(30_000);
            errdefer frame.abort();
            try dequant_pipes.encodeLinearUnified(&frame, .{
                .amf_dtype = pe.dtype.toAmf(),
                .group_size = pe.gs,
                .weight = .{ .buffer = pf_buf, .offset = w_off },
                .params = .{ .buffer = pf_buf, .offset = p_off },
                .x = .{ .buffer = pf_buf, .offset = x_off },
                .y = .{ .buffer = pf_buf, .offset = y_off },
                .m = pm,
                .n = pn,
                .k = pk,
                .mode = .gemm_wide,
                .qf_pipes = &qf,
                .tile_f16 = .{ .buffer = pf_buf, .offset = tile_off },
                .tile_f16_bytes = tile_bytes,
            }, false);
            try compute.wait(try frame.commit(), 30_000);
        }
        const t0 = try vmstat.nowNs();
        for (0..iters) |_| {
            var frame = try compute.beginFrame(30_000);
            errdefer frame.abort();
            try dequant_pipes.encodeLinearUnified(&frame, .{
                .amf_dtype = pe.dtype.toAmf(),
                .group_size = pe.gs,
                .weight = .{ .buffer = pf_buf, .offset = w_off },
                .params = .{ .buffer = pf_buf, .offset = p_off },
                .x = .{ .buffer = pf_buf, .offset = x_off },
                .y = .{ .buffer = pf_buf, .offset = y_off },
                .m = pm,
                .n = pn,
                .k = pk,
                .mode = .gemm_wide,
                .qf_pipes = &qf,
                .tile_f16 = .{ .buffer = pf_buf, .offset = tile_off },
                .tile_f16_bytes = tile_bytes,
            }, false);
            try compute.wait(try frame.commit(), 30_000);
        }
        const elapsed = (try vmstat.nowNs()) -| t0;
        const avg_ms = @as(f64, @floatFromInt(elapsed)) / (@as(f64, @floatFromInt(iters)) * 1_000_000.0);
        const flops: f64 = 2.0 * @as(f64, @floatFromInt(pm)) * @as(f64, @floatFromInt(pn)) * @as(f64, @floatFromInt(pk));
        const tflops = if (avg_ms > 0.0) (flops / (avg_ms * 1e-3)) / 1e12 else 0.0;
        report.g23_prefill[report.g23_prefill_count] = .{
            .label = pe.label,
            .dtype = pe.dtype,
            .group_size = pe.gs,
            .avg_ms = avg_ms,
            .tflops = tflops,
        };
        report.g23_prefill_count += 1;
    }
}

// --- Section 2: G24 Activation Quantization Emulation & 4/6 MAE Win Rate ---
fn benchM81G24(
    allocator: std.mem.Allocator,
    device: metal.Device,
    library: metal.Library,
    compute: *metal.Compute,
    cfg: M81Config,
    report: *M81Report,
) !void {
    var qf = try quant_formats.QuantFormatPipelines.initWithOptions(device, library, .{
        .specs = &.{},
        .include_moe = false,
        .include_act_quant = true,
    });
    defer qf.close();
    try qf.makeResident(compute.*);
    defer compute.removeResident(&.{qf.codebook_buf}) catch |err| {
        std.debug.panic("removeResident failed in benchM81G24: {}", .{err});
    };

    const k: u32 = cfg.act_quant_k;
    const max_m: u32 = 16;
    const io_bytes = amf.align64(@as(usize, max_m) * k * 2);
    const region = try page.WiredRegion.init(io_bytes * 2);
    defer region.deinit();
    const x_in: []f16 = @alignCast(std.mem.bytesAsSlice(f16, region.bytes[0 .. @as(usize, max_m) * k * 2]));
    const x_out: []f16 = @alignCast(std.mem.bytesAsSlice(f16, region.bytes[io_bytes..][0 .. @as(usize, max_m) * k * 2]));

    // Populate with a realistic mix of heavy-tailed + compact blocks so 4/6 selection is exercised.
    var st: u64 = 0x24_2026_CAFEBABE;
    for (0..max_m) |row| {
        for (0..k / 16) |blk| {
            const amp: f32 = if ((blk & 3) == 0) 4.1 else 2.2;
            for (0..16) |j| {
                st ^= st >> 12;
                st ^= st << 25;
                st ^= st >> 27;
                const u = (st *% 0x2545F4914F6CDD1D) >> 40;
                const v = ((@as(f32, @floatFromInt(u)) / 16777216.0) * 2.0 - 1.0) * amp;
                x_in[row * k + blk * 16 + j] = @floatCast(v);
            }
        }
    }

    const buf = try metal.Buffer.wrap(device, region.bytes);
    defer buf.close();
    try compute.makeResident(&.{buf});
    defer compute.removeResident(&.{buf}) catch |err| {
        std.debug.panic("removeResident failed in benchM81G24 buf: {}", .{err});
    };

    const modes = [_]quant_formats.ActQuantMode{ .none, .mxfp8_g32, .fp8_g128, .nvfp4_g16, .nvfp4_fouroversix_mae };
    const iters: usize = @max(@as(usize, 1), cfg.act_quant_iters);
    report.g24_none_dispatches = 0;

    // Save GPU outputs of nvfp4_g16 vs nvfp4_fouroversix_mae on row 0..15 to compute exact block win rate.
    const nv4_out = try allocator.alloc(f16, @as(usize, max_m) * k);
    defer allocator.free(nv4_out);
    const f46_out = try allocator.alloc(f16, @as(usize, max_m) * k);
    defer allocator.free(f46_out);

    for (modes, 0..) |mode, idx| {
        if (mode == .none) {
            report.g24_rows[idx] = .{ .mode = .none, .m1_us = 0.0, .m16_us = 0.0, .mae_vs_fp32 = 0.0 };
            continue;
        }
        var m1_us: f64 = 0.0;
        var m16_us: f64 = 0.0;
        for ([_]u32{ 1, max_m }) |m_val| {
            for (0..3) |_| {
                var frame = try compute.beginFrame(30_000);
                errdefer frame.abort();
                try qf.encodeActQuantEmulate(&frame, mode, .{
                    .x_in = .{ .buffer = buf, .offset = 0 },
                    .x_out = .{ .buffer = buf, .offset = io_bytes },
                    .m = m_val,
                    .k = k,
                }, false);
                try compute.wait(try frame.commit(), 30_000);
            }
            const t0 = try vmstat.nowNs();
            var frame = try compute.beginFrame(30_000);
            errdefer frame.abort();
            for (0..iters) |d| {
                try qf.encodeActQuantEmulate(&frame, mode, .{
                    .x_in = .{ .buffer = buf, .offset = 0 },
                    .x_out = .{ .buffer = buf, .offset = io_bytes },
                    .m = m_val,
                    .k = k,
                }, d > 0);
            }
            try compute.wait(try frame.commit(), 30_000);
            const elapsed = (try vmstat.nowNs()) -| t0;
            const avg_us = @as(f64, @floatFromInt(elapsed)) / (@as(f64, @floatFromInt(iters)) * 1000.0);
            if (m_val == 1) m1_us = avg_us else m16_us = avg_us;
        }

        var sum_abs: f64 = 0.0;
        for (0..@as(usize, max_m) * k) |i| {
            sum_abs += @abs(@as(f64, @floatCast(x_out[i])) - @as(f64, @floatCast(x_in[i])));
        }
        const mae = sum_abs / @as(f64, @floatFromInt(@as(usize, max_m) * k));
        if (mode == .nvfp4_g16) @memcpy(nv4_out, x_out);
        if (mode == .nvfp4_fouroversix_mae) @memcpy(f46_out, x_out);

        report.g24_rows[idx] = .{ .mode = mode, .m1_us = m1_us, .m16_us = m16_us, .mae_vs_fp32 = mae };
    }

    var wins: usize = 0;
    const total_blocks = @as(usize, max_m) * (k / 16);
    var mae6_sum: f64 = 0.0;
    var mae46_sum: f64 = 0.0;
    for (0..total_blocks) |b| {
        var e6: f64 = 0.0;
        var e46: f64 = 0.0;
        for (0..16) |j| {
            const ii = b * 16 + j;
            const ref_v: f64 = @floatCast(x_in[ii]);
            e6 += @abs(@as(f64, @floatCast(nv4_out[ii])) - ref_v);
            e46 += @abs(@as(f64, @floatCast(f46_out[ii])) - ref_v);
        }
        mae6_sum += e6;
        mae46_sum += e46;
        if (e46 <= e6 + 1e-12) wins += 1;
    }
    report.g24_fouroversix_win_pct = 100.0 * @as(f64, @floatFromInt(wins)) / @as(f64, @floatFromInt(total_blocks));
    report.g24_fouroversix_mae_reduction_pct = if (mae6_sum > 0.0) ((mae6_sum - mae46_sum) / mae6_sum) * 100.0 else 0.0;
}

// --- Section 3: C25 SME2 Multi-Format Microkernels & C13 Bus-Contention Gate ---
fn benchM81C25(
    allocator: std.mem.Allocator,
    chip: chip_profile_mod.ChipProfile,
    cfg: M81Config,
    report: *M81Report,
) !void {
    const caps = try sme.SmeCaps.probe();
    report.c25_has_sme2 = caps.supportsSme2();
    var disp = try sme.SmeDispatcher.init(allocator, .{ .caps_override = caps, .total_threads = 2 });
    defer disp.deinit();

    const n: usize = 512;
    const k: usize = 512;
    const m4: usize = 4;
    const m16: usize = 16;

    const a_u16 = try allocator.alignedAlloc(u16, .fromByteUnits(sme.SVL_BYTES), m16 * k);
    defer allocator.free(a_u16);
    for (a_u16, 0..) |*v, i| v.* = quant.f32ToFloatBits(.f16, @as(f32, @floatFromInt((i % 17))) * 0.04 - 0.3);

    const out_sme = try allocator.alignedAlloc(f32, .fromByteUnits(sme.SVL_BYTES), m16 * n);
    defer allocator.free(out_sme);
    const out_ref = try allocator.alignedAlloc(f32, .fromByteUnits(sme.SVL_BYTES), m16 * n);
    defer allocator.free(out_ref);

    const C25Spec = struct { label: []const u8, dt: sme.PanelDType };
    const c25_specs = [_]C25Spec{
        .{ .label = "nvfp4_g16", .dt = .nvfp4_g16 },
        .{ .label = "mxfp4_g32", .dt = .mxfp4_g32 },
        .{ .label = "q2_0", .dt = .q2_0 },
        .{ .label = "tq1_0", .dt = .tq1_0 },
        .{ .label = "tq2_0", .dt = .tq2_0 },
        .{ .label = "iq2_xxs", .dt = .iq2_xxs },
    };

    const use_be: ?sme.ExecutionBackend = if (caps.supportsSme2()) .sme2 else .scalar_fallback;
    const iters: usize = @max(@as(usize, 1), cfg.sme_iters);

    for (c25_specs, 0..) |sp, idx| {
        const w_bytes = try sme.packedBlobBytes(sp.dt, n, k);
        const w_packed = try allocator.alignedAlloc(u8, .fromByteUnits(sme.SVL_BYTES), w_bytes);
        defer allocator.free(w_packed);

        switch (sp.dt) {
            .nvfp4_g16 => {
                const nib = try allocator.alloc(u8, n * (k / 2));
                defer allocator.free(nib);
                const sc = try allocator.alloc(u8, n * (k / 16));
                defer allocator.free(sc);
                for (nib, 0..) |*b, i| b.* = @truncate((i * 37 + 11) & 0xFF);
                for (sc, 0..) |*s, i| s.* = quant_formats.f32ToE4m3(0.25 + @as(f32, @floatFromInt(i % 7)) * 0.1);
                try sme.packNvfp4Panel64(nib, sc, n, k, w_packed);
            },
            .mxfp4_g32 => {
                const codes = try allocator.alloc(u8, n * k);
                defer allocator.free(codes);
                const sc = try allocator.alloc(u8, n * (k / 32));
                defer allocator.free(sc);
                for (codes, 0..) |*c, i| c.* = @truncate((i * 13 + 5) & 0xF);
                for (sc, 0..) |*s, i| s.* = @intCast(124 + (i % 5));
                try sme.packMxfp4Panel64(codes, sc, n, k, w_packed);
            },
            .q2_0 => {
                const codes = try allocator.alloc(u8, n * k);
                defer allocator.free(codes);
                const sc = try allocator.alloc(f16, n * (k / 64));
                defer allocator.free(sc);
                for (codes, 0..) |*c, i| c.* = @truncate((i * 7 + 1) & 3);
                for (sc, 0..) |*s, i| s.* = @floatCast(0.15 + @as(f32, @floatFromInt(i % 5)) * 0.05);
                try sme.packQ20Panel64(codes, sc, n, k, w_packed);
            },
            .tq1_0 => {
                const raw = try allocator.alloc(u8, n * (k / 256) * 54);
                defer allocator.free(raw);
                @memset(raw, 0);
                for (0..n * (k / 256)) |bi| {
                    const blk = raw[bi * 54 ..][0..54];
                    for (0..52) |j| blk[j] = @truncate((bi * 19 + j * 11) % 243);
                    writeF16Le(blk[52..54], 0.25);
                }
                try sme.packTq10Panel64(raw, n, k, w_packed);
            },
            .tq2_0 => {
                const raw = try allocator.alloc(u8, n * (k / 256) * 66);
                defer allocator.free(raw);
                @memset(raw, 0);
                for (0..n * (k / 256)) |bi| {
                    const blk = raw[bi * 66 ..][0..66];
                    for (0..64) |j| blk[j] = @truncate((bi * 29 + j * 7) & 0xFF);
                    writeF16Le(blk[64..66], 0.25);
                }
                try sme.packTq20Panel64(raw, n, k, w_packed);
            },
            .iq2_xxs => {
                const raw = try allocator.alloc(u8, n * (k / 256) * 66);
                defer allocator.free(raw);
                @memset(raw, 0);
                for (0..n * (k / 256)) |bi| {
                    const blk = raw[bi * 66 ..][0..66];
                    writeF16Le(blk[0..2], 0.25);
                    for (2..66) |j| blk[j] = @truncate((bi * 41 + j * 13) & 0xFF);
                }
                try sme.packIq2XxsPanel64(raw, n, k, w_packed);
            },
            else => unreachable,
        }

        const w_obj: sme.SmeLinearWeight = .{
            .dtype = sp.dt,
            .param_kind = .affine_scale_bias,
            .cache_hint = .temporal,
            .n = n,
            .k = k,
            .w_packed = w_packed,
            .outer_scale = if (sp.dt == .nvfp4_g16) 0.5 else 1.0,
        };
        const op_m4 = [_]sme.BatchLinearOp{.{
            .weight = w_obj,
            .a_u16 = a_u16[0 .. m4 * k],
            .out_bytes = std.mem.sliceAsBytes(out_sme[0 .. m4 * n]),
            .m = m4,
        }};
        const op_m16 = [_]sme.BatchLinearOp{.{
            .weight = w_obj,
            .a_u16 = a_u16,
            .out_bytes = std.mem.sliceAsBytes(out_sme),
            .m = m16,
        }};
        const op_ref = [_]sme.BatchLinearOp{.{
            .weight = w_obj,
            .a_u16 = a_u16,
            .out_bytes = std.mem.sliceAsBytes(out_ref),
            .m = m16,
        }};

        // Warmup + parity check against scalar fallback.
        try disp.evalLinearBatchHoisted(&op_m16, use_be);
        try disp.evalLinearBatchHoisted(&op_ref, .scalar_fallback);
        var max_err: f32 = 0.0;
        for (out_sme, out_ref) |a, b| {
            const d = @abs(a - b);
            if (d > max_err) max_err = d;
        }

        for (0..2) |_| try disp.evalLinearBatchHoisted(&op_m4, use_be);
        const t4_0 = try vmstat.nowNs();
        for (0..iters) |_| try disp.evalLinearBatchHoisted(&op_m4, use_be);
        const e4 = (try vmstat.nowNs()) -| t4_0;

        for (0..2) |_| try disp.evalLinearBatchHoisted(&op_m16, use_be);
        const t16_0 = try vmstat.nowNs();
        for (0..iters) |_| try disp.evalLinearBatchHoisted(&op_m16, use_be);
        const e16 = (try vmstat.nowNs()) -| t16_0;

        const ops4 = 2.0 * @as(f64, @floatFromInt(m4 * n * k * iters));
        const ops16 = 2.0 * @as(f64, @floatFromInt(m16 * n * k * iters));
        report.c25_rows[idx] = .{
            .label = sp.label,
            .dtype = sp.dt,
            .m4_gops = if (e4 > 0) ops4 / @as(f64, @floatFromInt(e4)) else 0.0,
            .m16_gops = if (e16 > 0) ops16 / @as(f64, @floatFromInt(e16)) else 0.0,
            .parity_max_err = max_err,
            .parity_ok = std.math.isFinite(max_err) and max_err < 1e-2,
        };
    }

    const l2_cap = caps.hotAdapterL2BudgetBytes();
    report.c25_c13_bus_gate_ok = (l2_cap <= 4 * 1024 * 1024) and
        (!caps.supportsSme2() or
            (sme.evaluateC13BusGate(caps, chip, l2_cap, true, null, 0) == .sme2_l2_temporal and
                sme.evaluateC13BusGate(caps, chip, l2_cap + 4096, true, 1.02, 0) == .defer_gpu_bus_saturated));
}

// --- Section 4: W21/W22 End-to-End Multi-Format Model + Draft + Fork Rejection ---
fn benchM81W21W22(allocator: std.mem.Allocator, report: *M81Report) !void {
    // 1. Dense multi-format (`nvfp4_g16` weights + `q8_affine` embed) across M=1 decode, M=4 verifyTree, M=16 wide prefill.
    {
        const scfg = syntheticMultiFormatDenseConfig();
        const amf_bytes = try buildSyntheticMultiFormatDenseAmf(allocator, scfg, .nvfp4_g16, 16, .q8_affine, 0x4141);
        defer allocator.free(amf_bytes);

        var model = try qwen3.Qwen3Model.loadFromBytes(allocator, amf_bytes, scfg, .{
            .max_seq_len = 64,
            .max_tree_nodes = 8,
            .enable_feature_taps = true,
            .verify_crcs = true,
            .startup_canary = false,
            .gpu_prefill_mode = .wide,
            .wide_prefill_chunk_tokens = 32,
        });
        defer model.deinit(allocator);

        // Wide prefill (M = 16 -> exercises encodeDequantTileF16 + encodeDenseGemm(.f16)!)
        const pre_toks = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
        const pf_tok = try model.prefillChunk(&pre_toks, 0, true);
        const dec_tok = try model.stepDecode(pf_tok, 16);

        // VerifyTree (M = 4 -> exercises encodeGemvBatch!)
        const tree_toks = [_]u32{ dec_tok, 21, 22, 23 };
        const tree_pos = [_]u32{ 17, 18, 18, 19 };
        const tree_masks = [_]u64{ 0b0001, 0b0011, 0b0101, 0b1011 };
        const vt = try model.verifyTree(&tree_toks, &tree_pos, &tree_masks, 17, true);
        var all_fin = true;
        for (vt.logits_f16[0..scfg.vocab_size]) |v| {
            if (!std.math.isFinite(@as(f32, @floatCast(v)))) all_fin = false;
        }
        report.w21_dense_decode_tree_prefill_ok = all_fin and pf_tok < scfg.vocab_size and dec_tok < scfg.vocab_size and vt.num_nodes == 4;

        // 2. Quantized `DraftHead` (MTP `q8_affine` + `nvfp4_g16` against quantized target).
        const dcfg: draft_mod.DraftConfig = .{
            .format = .mtp,
            .num_hidden_layers = 1,
            .hidden_size = scfg.hidden_size,
            .target_hidden_size = scfg.hidden_size,
            .intermediate_size = scfg.intermediate_size,
            .num_attention_heads = scfg.num_attention_heads,
            .num_key_value_heads = scfg.num_key_value_heads,
            .head_dim = scfg.head_dim,
            .vocab_size = scfg.vocab_size,
            .draft_vocab_size = scfg.vocab_size,
            .rms_norm_eps = scfg.rms_norm_eps,
            .rope_theta = scfg.rope_theta,
            .max_position_embeddings = scfg.max_position_embeddings,
            .share_embed_tokens = true,
            .share_lm_head = true,
        };
        const draft_bytes = try buildSyntheticQuantMtpDraftAmf(allocator, dcfg, .nvfp4_g16, 16, 0xD2A7);
        defer allocator.free(draft_bytes);

        var dh = try draft_mod.DraftHead.loadFromBytes(allocator, &model, draft_bytes, dcfg, .{
            .placement = .gpu,
            .max_seq_len = 64,
            .max_tree_slots = 8,
            .verify_crcs = true,
        });
        defer dh.deinit(allocator, &model);
        const h_fake = try allocator.alloc(f16, scfg.hidden_size);
        defer allocator.free(h_fake);
        for (h_fake, 0..) |*v, i| v.* = @floatCast(@as(f32, @floatFromInt(i % 11)) * 0.05);
        @memcpy(model.accepted_top_host[0..scfg.hidden_size], h_fake);
        try dh.syncAccepted(&model, &.{dec_tok}, model.accepted_top_bind, 0, 4);
        report.w22_draft_quant_ok = (dh.rootCandidates().len == 4);
    }

    // 3. Heterogeneous asymmetric MoE (`nvfp4_g16` gate/up + `gguf_q2_0` down AND `gguf_q4_k` experts + `fp8_e4m3_block` attn).
    {
        const mcfg = syntheticMultiFormatMoeConfig();
        const moe_bytes = try buildSyntheticMultiFormatMoeAmf(allocator, mcfg, 0x5252);
        defer allocator.free(moe_bytes);

        var moe_m = try qwen3.Qwen3Model.loadFromBytes(allocator, moe_bytes, mcfg, .{
            .max_seq_len = 64,
            .max_tree_nodes = 8,
            .verify_crcs = true,
            .startup_canary = false,
            .gpu_prefill_mode = .wide,
            .wide_prefill_chunk_tokens = 32,
        });
        defer moe_m.deinit(allocator);

        const pf_toks = [_]u32{ 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32 };
        const t0 = try moe_m.prefillChunk(&pf_toks, 0, true);
        const t1 = try moe_m.stepDecode(t0, 16);
        const vt = try moe_m.verifyTree(&.{ t1, 3, 5, 7 }, &.{ 17, 18, 18, 19 }, &.{ 0b1, 0b11, 0b101, 0b1011 }, 17, true);
        var fin = true;
        for (vt.logits_f16[0..mcfg.vocab_size]) |v| {
            if (!std.math.isFinite(@as(f32, @floatCast(v)))) fin = false;
        }
        report.w22_moe_asymmetric_hetero_ok = fin and t0 < mcfg.vocab_size and t1 < mcfg.vocab_size;
    }

    // 4. Fork-collision guard (`STQ1_0` 1.312 bpw payload colliding with `F8_E4M3` 8.0 bpw on type 43).
    {
        // Construct an `.amf` entry claiming `fp8_e4m3_block` `[256, 256]` (65,536 B weights)
        // but carrying `STQ1_0`-sized `10,752 B` data_bytes: `amf.Parsed.parseBytes` MUST reject it!
        var e = amf.TensorEntry.zeroed("layers.0.mlp.down_proj", .dense, .fp8_e4m3_block);
        e.quant_bits = 8;
        e.rank = 2;
        e.group_size = 128;
        e.shape = .{ 256, 256, 0, 0 };
        e.data_bytes = 10_752; // 1.3125 bpw STQ1_0 payload size vs 65,536 B expected!
        e.params_offset = amf.align64(e.data_bytes);
        e.params_bytes = 16;
        const blob = try allocator.alloc(u8, page.APPLE_PAGE_SIZE);
        defer allocator.free(blob);
        @memset(blob, 0);
        var items = [_]BuiltEntry{.{ .entry = e, .blob = blob }};
        const bad_img = try buildAmfImageFromEntries(allocator, "fork-stq1-collision", &items);
        defer allocator.free(bad_img);

        if (amf.Parsed.parseBytes(allocator, bad_img)) |parsed_ok| {
            var p = parsed_ok;
            p.deinit(allocator);
            report.w22_fork_collision_rejected = false;
        } else |err| {
            report.w22_fork_collision_rejected = (err == error.InvalidShape or err == error.InvalidQuantization or err == error.BlobOutOfBounds);
        }
    }
}

fn buildSyntheticQuantMtpDraftAmf(
    allocator: std.mem.Allocator,
    cfg: draft_mod.DraftConfig,
    linear_dtype: amf.DType,
    group_size: u16,
    seed: u64,
) ![]u8 {
    var list: std.ArrayList(BuiltEntry) = .empty;
    defer {
        for (list.items) |it| allocator.free(it.blob);
        list.deinit(allocator);
    }
    const h: usize = cfg.hidden_size;
    const q_dim: usize = cfg.num_attention_heads * cfg.head_dim;
    const kv_dim: usize = cfg.num_key_value_heads * cfg.head_dim;
    const inter: usize = cfg.intermediate_size;

    try appendNormEntry(allocator, &list, "enorm.weight", amf.NO_LAYER, h, seed + 1);
    try appendNormEntry(allocator, &list, "hnorm.weight", amf.NO_LAYER, h, seed + 2);
    try appendLinearEntry(allocator, &list, "eh_proj.weight", .dense, amf.NO_LAYER, amf.NO_EXPERT, h, 2 * h, seed + 3, 0.14, linear_dtype, group_size);
    try appendNormEntry(allocator, &list, "norm.weight", amf.NO_LAYER, h, seed + 4);

    try appendNormEntry(allocator, &list, "layers.0.input_layernorm.weight", 0, h, seed + 10);
    try appendLinearEntry(allocator, &list, "layers.0.self_attn.q_proj", .dense, 0, amf.NO_EXPERT, q_dim, h, seed + 11, 0.14, linear_dtype, group_size);
    try appendLinearEntry(allocator, &list, "layers.0.self_attn.k_proj", .dense, 0, amf.NO_EXPERT, kv_dim, h, seed + 12, 0.14, linear_dtype, group_size);
    try appendLinearEntry(allocator, &list, "layers.0.self_attn.v_proj", .dense, 0, amf.NO_EXPERT, kv_dim, h, seed + 13, 0.14, linear_dtype, group_size);
    try appendLinearEntry(allocator, &list, "layers.0.self_attn.o_proj", .dense, 0, amf.NO_EXPERT, h, q_dim, seed + 14, 0.14, linear_dtype, group_size);
    try appendNormEntry(allocator, &list, "layers.0.post_attention_layernorm.weight", 0, h, seed + 15);
    try appendLinearEntry(allocator, &list, "layers.0.mlp.gate_proj", .dense, 0, amf.NO_EXPERT, inter, h, seed + 16, 0.12, linear_dtype, group_size);
    try appendLinearEntry(allocator, &list, "layers.0.mlp.up_proj", .dense, 0, amf.NO_EXPERT, inter, h, seed + 17, 0.12, linear_dtype, group_size);
    try appendLinearEntry(allocator, &list, "layers.0.mlp.down_proj", .dense, 0, amf.NO_EXPERT, h, inter, seed + 18, 0.12, linear_dtype, group_size);

    return buildAmfImageFromEntries(allocator, "synth-m81-mtp-draft", list.items);
}

// --- Section 5: W24 Tokenizers, ChatTemplate Render Latency, C18 HTTP/SSE, KL Gate ---
fn benchTokenizerThroughput(allocator: std.mem.Allocator, tok: *const tokenizer_mod.Tokenizer) !f64 {
    const sample =
        "The quick brown fox jumps over 12345 lazy dogs in bare-metal Zig 0.17.1! " ++
        "def compute_kl(p_ref, p_cand): return sum(p * math.log(p / q) for p, q in zip(p_ref, p_cand))\n" ++
        "苹果芯片 Apple Silicon M4 统一内存零拷贝推理引擎测试 2026.\n";
    var corpus: [8192]u8 = undefined;
    var filled: usize = 0;
    while (filled + sample.len <= corpus.len) : (filled += sample.len) {
        @memcpy(corpus[filled..][0..sample.len], sample);
    }
    const text = corpus[0..filled];

    var scratch = try tokenizer_mod.EncodeScratch.init(allocator, 8192);
    defer scratch.deinit(allocator);
    const out_ids = try allocator.alloc(u32, text.len + 64);
    defer allocator.free(out_ids);

    // Warmup
    _ = try tok.encodeWithScratch(&scratch, text, out_ids);

    const iters: usize = 16;
    const t0 = try vmstat.nowNs();
    for (0..iters) |_| {
        _ = try tok.encodeWithScratch(&scratch, text, out_ids);
    }
    const elapsed = (try vmstat.nowNs()) -| t0;
    const total_bytes: f64 = @floatFromInt(text.len * iters);
    return if (elapsed > 0) (total_bytes / @as(f64, @floatFromInt(elapsed))) * 1000.0 else 0.0;
}

fn sendAllFd(fd: std.c.fd_t, data: []const u8) !void {
    var off: usize = 0;
    while (off < data.len) {
        const n = std.c.write(fd, data.ptr + off, data.len - off);
        if (n <= 0) return error.WriteFailed;
        off += @intCast(n);
    }
}

fn readHttpResponseOne(fd: std.c.fd_t, out_buf: []u8) ![]const u8 {
    var used: usize = 0;
    var header_end: ?usize = null;
    var content_len: ?usize = null;
    var is_chunked = false;
    var is_sse = false;

    while (used < out_buf.len) {
        const n = std.c.read(fd, out_buf.ptr + used, out_buf.len - used);
        if (n < 0) return error.ReadFailed;
        if (n == 0) break;
        used += @intCast(n);
        if (header_end == null) {
            if (std.mem.indexOf(u8, out_buf[0..used], "\r\n\r\n")) |hend| {
                header_end = hend + 4;
                const hdrs = out_buf[0..hend];
                if (std.mem.indexOf(u8, hdrs, "Content-Length: ")) |cl_pos| {
                    const rest = hdrs[cl_pos + "Content-Length: ".len ..];
                    const eol = std.mem.indexOf(u8, rest, "\r\n") orelse rest.len;
                    content_len = std.fmt.parseInt(usize, rest[0..eol], 10) catch null;
                }
                if (std.mem.indexOf(u8, hdrs, "Transfer-Encoding: chunked") != null) is_chunked = true;
                if (std.mem.indexOf(u8, hdrs, "text/event-stream") != null) is_sse = true;
            }
        }
        if (header_end) |hend| {
            if (content_len) |cl| {
                if (used >= hend + cl) break;
            } else if (is_chunked and std.mem.indexOf(u8, out_buf[hend..used], "0\r\n\r\n") != null) {
                break;
            } else if (is_sse and std.mem.indexOf(u8, out_buf[hend..used], "event: response.completed\n") != null and
                std.mem.endsWith(u8, out_buf[hend..used], "\n\n"))
            {
                break;
            }
        }
    }
    return out_buf[0..used];
}

fn connectLoopback(port: u16) !std.c.fd_t {
    const fd = std.c.socket(std.c.AF.INET, std.c.SOCK.STREAM, 0);
    if (fd < 0) return error.SocketFailed;
    errdefer _ = std.c.close(fd);
    var tv: std.c.timeval = .{ .sec = 5, .usec = 0 };
    _ = std.c.setsockopt(fd, std.c.SOL.SOCKET, std.c.SO.RCVTIMEO, std.mem.asBytes(&tv), @sizeOf(std.c.timeval));
    var addr: std.c.sockaddr.in = .{
        .family = std.c.AF.INET,
        .port = std.mem.nativeToBig(u16, port),
        .addr = std.mem.nativeToBig(u32, 0x7f000001),
        .zero = @splat(0),
    };
    if (std.c.connect(fd, @ptrCast(&addr), @sizeOf(std.c.sockaddr.in)) != 0) return error.ConnectFailed;
    return fd;
}

fn benchM81W24AndC18(allocator: std.mem.Allocator, cfg: M81Config, report: *M81Report) !void {
    // 1. Inkling 2 (`o200k` + `ignore_merges`) & Hy4 tokenizers (live metadata when present).
    const ink_path: [:0]const u8 = "/Volumes/Akari/models/metadata/thinkingmachines__Inkling-2-Mini/tokenizer.json";
    if (readSmallFileAlloc(allocator, ink_path, 48 * 1024 * 1024)) |bytes| {
        defer allocator.free(bytes);
        var tok = try tokenizer_mod.Tokenizer.fromHuggingFaceJson(allocator, bytes);
        defer tok.deinit(allocator);
        report.w24_inkling2_live = true;
        report.w24_inkling2_mbs = try benchTokenizerThroughput(allocator, &tok);
    } else |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => {
            const inline_ink =
                \\{"model":{"type":"BPE","ignore_merges":true,"vocab":{"a":0,"b":1,"c":2," ":3,"e":4,"t":5,"h":6,"i":7,"n":8,"o":9,"r":10,"s":11,"l":12,"d":13,"u":14,"m":15,"p":16,"f":17,"g":18,"y":19,"w":20,"k":21,"v":22,"x":23,"z":24,"q":25,"j":26,"0":27,"1":28,"2":29,"3":30,"4":31,"5":32,"6":33,"7":34,"8":35,"9":36,".":37,",":38,"!":39,":":40,"(":41,")":42,"_":43,"-":44,"/":45,"\n":46,"the":47,"quick":48,"brown":49,"fox":50,"<|message_user|>":51,"<|content_model_end_sampling|>":52},"merges":["t h","th e"]},"pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":"[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+"},"behavior":"Isolated"},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}]},"added_tokens":[{"id":51,"content":"<|message_user|>","special":true},{"id":52,"content":"<|content_model_end_sampling|>","special":true}]}
            ;
            var tok = try tokenizer_mod.Tokenizer.fromHuggingFaceJson(allocator, inline_ink);
            defer tok.deinit(allocator);
            report.w24_inkling2_live = false;
            report.w24_inkling2_mbs = try benchTokenizerThroughput(allocator, &tok);
        },
    }

    const hy4_path: [:0]const u8 = "/Volumes/Akari/models/metadata/tencent__Hy4-flash/tokenizer.json";
    if (readSmallFileAlloc(allocator, hy4_path, 32 * 1024 * 1024)) |bytes| {
        defer allocator.free(bytes);
        var tok = try tokenizer_mod.Tokenizer.fromHuggingFaceJson(allocator, bytes);
        defer tok.deinit(allocator);
        report.w24_hy4_live = true;
        report.w24_hy4_mbs = try benchTokenizerThroughput(allocator, &tok);
    } else |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => {
            const inline_hy4 =
                \\{"model":{"type":"BPE","vocab":{"a":0,"b":1,"c":2," ":3,"e":4,"t":5,"h":6,"i":7,"n":8,"o":9,"r":10,"s":11,"<｜hy_start:opensource｜>":12,"<｜hy_end:opensource｜>":13},"merges":["t h","th e"]},"added_tokens":[{"id":12,"content":"<｜hy_start:opensource｜>","special":true},{"id":13,"content":"<｜hy_end:opensource｜>","special":true}]}
            ;
            var tok = try tokenizer_mod.Tokenizer.fromHuggingFaceJson(allocator, inline_hy4);
            defer tok.deinit(allocator);
            report.w24_hy4_live = false;
            report.w24_hy4_mbs = try benchTokenizerThroughput(allocator, &tok);
        },
    }

    // 2. Zero-alloc `ChatTemplate` render latency (`us`) across ChatML, Inkling 2, Hy4.
    var render_buf: [4096]u8 = undefined;
    const tpl_iters: usize = 200;
    const dialects = [_]chat_template_mod.Dialect{ .chatml, .inkling2, .hy4 };
    var tpl_us: [3]f64 = .{ 0.0, 0.0, 0.0 };
    for (dialects, 0..) |d, di| {
        const tpl = chat_template_mod.ChatTemplate.preset(d);
        const t0 = try vmstat.nowNs();
        for (0..tpl_iters) |_| {
            var fw = chat_template_mod.FixedWriter.init(&render_buf);
            var r = chat_template_mod.PromptRenderer.init(&tpl, &fw, .{ .reasoning_effort = .medium }, true, 1);
            try r.begin();
            try r.pushMessage(.{ .role = .system, .content = "You are a concise assistant." }, 0);
            try r.pushMessage(.{ .role = .user, .content = "Compute 17 * 23 and explain briefly." }, 1);
            try r.finish();
            std.mem.doNotOptimizeAway(fw.written().len);
        }
        const elapsed = (try vmstat.nowNs()) -| t0;
        tpl_us[di] = @as(f64, @floatFromInt(elapsed)) / (@as(f64, @floatFromInt(tpl_iters)) * 1000.0);
    }
    report.w24_chat_tpl_chatml_us = tpl_us[0];
    report.w24_chat_tpl_inkling2_us = tpl_us[1];
    report.w24_chat_tpl_hy4_us = tpl_us[2];

    // 3. C18 live loopback HTTP/SSE check: keep-alive reuse + HMAC-SHA256 signature + tamper 400 + response.completed.output.
    {
        const Mock = struct {
            fn generate(_: *anyopaque, _: *const server_mod.InferenceJob, stream: *server_mod.TokenStream) void {
                _ = stream.setPromptTokens(8);
                _ = stream.emitTokenBytes("<think>Step 1: multiply 17 by 23. Step 2: 391.</think>391");
                stream.finish(.stop, 8);
            }
        };
        var dummy: u8 = 0;
        const srv = try server_mod.Server.init(allocator, .{
            .port = 0,
            .model_id = "m81-verify",
            .api_key = "m81-bench-secret",
            .apply_qos = false,
            .http_read_timeout_ms = 5_000,
        });
        defer srv.deinit(allocator);
        try srv.start(.{ .ctx = &dummy, .generateFn = Mock.generate });
        defer srv.stop();

        const port = srv.boundPort();
        var resp1_buf: [16384]u8 = undefined;
        var resp2_buf: [16384]u8 = undefined;

        // (a) HTTP/1.1 keep-alive reuse across 2 sequential requests on ONE socket!
        {
            const fd = try connectLoopback(port);
            defer _ = std.c.close(fd);
            const req1 = "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n";
            try sendAllFd(fd, req1);
            const r1 = try readHttpResponseOne(fd, &resp1_buf);
            const req2 = "GET /v1/models HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer m81-bench-secret\r\nConnection: close\r\n\r\n";
            try sendAllFd(fd, req2);
            const r2 = try readHttpResponseOne(fd, &resp2_buf);
            report.c18_keepalive_reuse_ok =
                std.mem.startsWith(u8, r1, "HTTP/1.1 200 OK\r\n") and
                std.mem.indexOf(u8, r1, "Connection: keep-alive") != null and
                std.mem.startsWith(u8, r2, "HTTP/1.1 200 OK\r\n");
        }

        // (b) Anthropic `/v1/messages` HMAC-SHA256 `signature` emit + tampered `signature` 400 rejection!
        {
            const body1 = "{\"model\":\"m81-verify\",\"max_tokens\":16,\"thinking\":{\"type\":\"enabled\",\"budget_tokens\":32},\"messages\":[{\"role\":\"user\",\"content\":\"17*23\"}]}";
            var req_buf: [2048]u8 = undefined;
            const req = try std.fmt.bufPrint(
                &req_buf,
                "POST /v1/messages HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer m81-bench-secret\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}",
                .{ body1.len, body1 },
            );
            const fd1 = try connectLoopback(port);
            defer _ = std.c.close(fd1);
            try sendAllFd(fd1, req);
            const r1 = try readHttpResponseOne(fd1, &resp1_buf);
            const has_sig = std.mem.startsWith(u8, r1, "HTTP/1.1 200 OK\r\n") and
                std.mem.indexOf(u8, r1, "\"signature\":\"") != null and
                std.mem.indexOf(u8, r1, "aminfer_sig_") == null;

            const bad_body = "{\"model\":\"m81-verify\",\"max_tokens\":16,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"},{\"role\":\"assistant\",\"content\":[{\"type\":\"thinking\",\"thinking\":\"tampered\",\"signature\":\"0000000000000000000000000000000000000000000000000000000000000000\"},{\"type\":\"text\",\"text\":\"ok\"}]},{\"role\":\"user\",\"content\":\"next\"}]}";
            const bad_req = try std.fmt.bufPrint(
                &req_buf,
                "POST /v1/messages HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer m81-bench-secret\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}",
                .{ bad_body.len, bad_body },
            );
            const fd2 = try connectLoopback(port);
            defer _ = std.c.close(fd2);
            try sendAllFd(fd2, bad_req);
            const r2 = try readHttpResponseOne(fd2, &resp2_buf);
            report.c18_hmac_sig_and_tamper_400_ok = has_sig and std.mem.startsWith(u8, r2, "HTTP/1.1 400 Bad Request\r\n");
        }

        // (c) OpenAI `/v1/responses` streaming `response.completed.response.output` check!
        {
            const body = "{\"model\":\"m81-verify\",\"input\":\"17*23\",\"reasoning\":{\"effort\":\"high\",\"summary\":\"concise\"},\"stream\":true}";
            var req_buf: [2048]u8 = undefined;
            const req = try std.fmt.bufPrint(
                &req_buf,
                "POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer m81-bench-secret\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}",
                .{ body.len, body },
            );
            const fd = try connectLoopback(port);
            defer _ = std.c.close(fd);
            try sendAllFd(fd, req);
            const r = try readHttpResponseOne(fd, &resp1_buf);
            const comp_pos = std.mem.indexOf(u8, r, "event: response.completed\n");
            report.c18_response_completed_output_ok = (comp_pos != null) and
                std.mem.indexOf(u8, r[comp_pos.?..], "\"output\":[") != null and
                std.mem.indexOf(u8, r[comp_pos.?..], "\"type\":\"reasoning_text\"") != null and
                std.mem.indexOf(u8, r[comp_pos.?..], "\"output_text\":\"391\"") != null;
        }
    }

    // 4. W24 Quality Gate (synthetic multi-format sweep + optional real on-disk Qwen3-0.6B BF16 vs GSQ-RCO).
    report.quality_gate = try runQualityGate(allocator, .{
        .ref_amf_path = if (cfg.skip_real_checkpoints) null else "/Volumes/Akari/models/amf/qwen3-0.6b-bf16.amf",
        .cand_amf_path = if (cfg.skip_real_checkpoints) null else "/Volumes/Akari/models/amf/qwen3-0.6b-gsq-rco.amf",
        .config_path = if (cfg.skip_real_checkpoints) null else "/Volumes/Akari/models/Qwen3-0.6B/config.json",
        .tokenizer_path = if (cfg.skip_real_checkpoints) null else "/Volumes/Akari/models/Qwen3-0.6B/tokenizer.json",
        .steps = cfg.quality_steps,
        .include_synthetic_sweep = true,
    });
}

pub fn runM81WithConfig(allocator: std.mem.Allocator, cfg: M81Config) !M81Report {
    const vm_before = try vmstat.Snapshot.capture();
    const facts = try sysinfo.Facts.read();
    var dev = try metal.Device.open();
    defer dev.close();
    const dev_info = try dev.info();
    const lib = try metal.Library.openEmbedded(dev);
    defer lib.close();
    var compute = try metal.Compute.initWithFrames(dev, 2);
    defer compute.close();
    var dequant_pipes = try dequant.DequantPipelines.init(dev, lib);
    defer dequant_pipes.close();

    const profile = chip_profile_mod.ChipProfile.resolve(
        facts,
        dev_info.name(),
        dev_info.appleGpuFamily(),
        cfg.bus_bandwidth_gbs_override,
    );

    var report: M81Report = .{};
    report.chip_profile = profile;
    const gname = dev_info.name();
    report.gpu_name_len = @min(gname.len, report.gpu_name_buf.len);
    @memcpy(report.gpu_name_buf[0..report.gpu_name_len], gname[0..report.gpu_name_len]);

    try benchM81G23(dev, lib, &compute, dequant_pipes, profile.spec_bus_gbs, cfg, &report);
    try benchM81G24(allocator, dev, lib, &compute, cfg, &report);
    try benchM81C25(allocator, profile, cfg, &report);
    try benchM81W21W22(allocator, &report);
    try benchM81W24AndC18(allocator, cfg, &report);

    report.vm_before = vm_before;
    report.vm_after = try vmstat.Snapshot.capture();
    try vmstat.checkNoSwap(report.vm_before, report.vm_after);
    return report;
}

pub fn runM81(allocator: std.mem.Allocator) !M81Report {
    return runM81WithConfig(allocator, .{});
}

test "m81: runM81WithConfig and W24 KL quality gate verify all 5 M8.1 tracks with zero swapouts" {
    const gpa = std.testing.allocator;
    const report = try runM81WithConfig(gpa, .{
        .gemv_n = 1024,
        .gemv_k = 1024,
        .gemv_min_working_set_bytes = 4 * 1024 * 1024,
        .gemv_warmup_batches = 1,
        .gemv_timed_batches = 2,
        .gemv_dispatches_per_batch = 4,
        .prefill_m = 64,
        .prefill_n = 512,
        .prefill_k = 512,
        .prefill_iters = 2,
        .act_quant_k = 1024,
        .act_quant_iters = 2,
        .sme_iters = 2,
        .quality_steps = 4,
        .skip_real_checkpoints = true,
    });

    try std.testing.expect(report.baseline_q4_gbs > 0.0);
    try std.testing.expect(report.g23_gemv_count >= 14);
    try std.testing.expect(report.g23_prefill_count == 4);
    try std.testing.expectEqual(@as(u32, 0), report.g24_none_dispatches);
    try std.testing.expect(report.g24_fouroversix_win_pct >= 99.0);
    try std.testing.expect(report.c25_c13_bus_gate_ok);
    for (report.c25_rows) |sr| try std.testing.expect(sr.parity_ok);
    try std.testing.expect(report.w21_dense_decode_tree_prefill_ok);
    try std.testing.expect(report.w22_moe_asymmetric_hetero_ok);
    try std.testing.expect(report.w22_draft_quant_ok);
    try std.testing.expect(report.w22_fork_collision_rejected);
    try std.testing.expect(report.w24_inkling2_mbs > 0.0);
    try std.testing.expect(report.w24_hy4_mbs > 0.0);
    try std.testing.expect(report.c18_keepalive_reuse_ok);
    try std.testing.expect(report.c18_hmac_sig_and_tamper_400_ok);
    try std.testing.expect(report.c18_response_completed_output_ok);
    try std.testing.expectEqual(@as(usize, DEFAULT_SYNTH_SPECS.len), report.quality_gate.synth_count);
    for (report.quality_gate.synth_pairs[0..report.quality_gate.synth_count]) |pair| {
        try std.testing.expect(pair.executed);
        try std.testing.expect(pair.all_finite);
        try std.testing.expect(pair.mean_kl >= 0.0 and std.math.isFinite(pair.mean_kl));
    }
    try std.testing.expectEqual(@as(u64, 0), vmstat.Snapshot.swapoutDelta(report.vm_before, report.vm_after));

    var aw: std.Io.Writer.Allocating = .init(gpa);
    defer aw.deinit();
    try report.write(&aw.writer);
    const out = aw.written();
    try std.testing.expect(std.mem.indexOf(u8, out, "bench.m81.g23.gemv.nvfp4_g16") != null);
    try std.testing.expect(std.mem.indexOf(u8, out, "quality_gate.synth.fp8_e4m3_block") != null);
}
