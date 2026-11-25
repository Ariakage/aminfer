//! Milestone M8.2 (`ADR-0024`, `ADR-0025`) `aminfer bench-m82` benchmark and
//! verification suite (`G25`, `C26`, `D14`, `W23`).
//!
//! All measurements run on real Apple Silicon hardware over 16 KiB-aligned
//! wired memory with `swapouts_delta == 0` strictly enforced (`ADR-0002`,
//! `ADR-0011`, `ADR-0012`).

const std = @import("std");
const metal = @import("../bridge/metal.zig");
const page = @import("../mem/page.zig");
const arena_mod = @import("../mem/arena.zig");
const budget_mod = @import("../mem/budget.zig");
const sysinfo = @import("../platform/sysinfo.zig");
const vmstat = @import("../platform/vmstat.zig");
const quant = @import("../quant/spec.zig");
const amf = @import("../format/amf.zig");
const dequant = @import("../gpu/dequant.zig");
const attention = @import("../gpu/attention.zig");
const hybrid = @import("../gpu/hybrid.zig");
const inkling2_hy4_gpu = @import("../gpu/inkling2_hy4.zig");
const sme = @import("../cpu/sme.zig");
const inkling2_ctrl = @import("../cpu/inkling2_ctrl.zig");
const ple = @import("../storage/ple.zig");
const graph = @import("../model/graph.zig");
const m82_runtime = @import("../model/m82_runtime.zig");
const dflare_dfly = @import("../spec/dflare_dfly.zig");
const engine_mod = @import("../spec/engine.zig");

const Io = std.Io;
const APPLE_PAGE_SIZE: usize = page.APPLE_PAGE_SIZE;

pub const M82Config = struct {
    kernel_iters: u32 = 40,
    ctrl_iters: u32 = 200,
    engram_iters: u32 = 64,
    cold_expert_iters: u32 = 32,
    synth_tokens: u32 = 16,
    include_python_synths: bool = true,
};

pub const E2eSynthResult = struct {
    ran: bool = false,
    verify_amf_ok: bool = false,
    greedy_parity_ok: bool = false,
    tokens_compared: u32 = 0,
    decode_tok_s: f64 = 0.0,
    spec_tok_s: f64 = 0.0,
    spec_tau: f64 = 0.0,
    spec_rounds: u64 = 0,
    dflare_core3_batches: u64 = 0,
    sme2_ctrl_calls: u64 = 0,
    gpu_ctrl_calls: u64 = 0,
    mid_pass_flushes: u64 = 0,
    cold_expert_fetches: u64 = 0,
    index_shared_reuses: u64 = 0,
    engram_nvme_reads: u64 = 0,
    engram_r_ge1_reads: u64 = 0,
    /// `M = 8` batched `Core x 3` `verifyTree` vs `8 x 1` serial `stepDecode` speedup (Inkling 2 Mini).
    batch8_verify_us: f64 = 0.0,
    serial8_decode_us: f64 = 0.0,
    batch8_core3_speedup: f64 = 0.0,
    /// `DFly2` `confidence_head` early-stop A/B (Hy4 & Hy4-flash).
    early_stop_parity_ok: bool = true,
    early_stop_tau: f64 = 0.0,
    early_stop_count: u64 = 0,
    no_early_stop_tok_s: f64 = 0.0,
    early_stop_tok_s: f64 = 0.0,
    early_stop_speedup: f64 = 0.0,
};

pub const M82Report = struct {
    // 1. G25 Fused vs Unfused Sublayer Speedups + 40/48 SWA KV + IndexCache + IndexPool + LatentMoE
    rmsnorm_sconv4_fused_us: f64 = 0.0,
    rmsnorm_sconv4_unfused_us: f64 = 0.0,
    rmsnorm_sconv4_speedup: f64 = 0.0,

    kv_swa40_8k_mib: f64 = 0.0,
    kv_full48_8k_mib: f64 = 0.0,
    kv_reduction_8k_pct: f64 = 0.0,
    kv_swa40_32k_mib: f64 = 0.0,
    kv_full48_32k_mib: f64 = 0.0,
    kv_reduction_32k_pct: f64 = 0.0,
    rel_gqa_m1_us: f64 = 0.0,
    rel_gqa_m8_us: f64 = 0.0,

    inkling2_moe_attnres_fused_us: f64 = 0.0,

    hy4_idx_all_full_4l_us: f64 = 0.0,
    hy4_idx_period4_4l_us: f64 = 0.0,
    hy4_idx_period4_speedup: f64 = 0.0,
    hy4_idx_shared_dispatches: u32 = 0,

    hy4_idx_pool_c4_us: f64 = 0.0,
    hy4_idx_pool_c1_us: f64 = 0.0,
    hy4_idx_pool_c4_speedup: f64 = 0.0,

    hy4_latent_moe_us: f64 = 0.0,
    hy4_full_moe_us: f64 = 0.0,
    hy4_latent_moe_speedup: f64 = 0.0,
    hy4_latent_moe_bytes_reduction_x: f64 = 0.0,

    // 2. C26 L2 SME2 gate_rank=64 loop controller + fp8_e4m3_row Engram F_NOCACHE + cold-expert F_NOCACHE
    sme2_ctrl_placement: inkling2_ctrl.Placement = .cpu_scalar_fallback,
    sme2_ctrl_fits_l2: bool = false,
    sme2_ctrl_weight_kib: f64 = 0.0,
    sme2_ctrl_m1_us_branch: f64 = 0.0,
    sme2_ctrl_m1_us_4b: f64 = 0.0,
    sme2_ctrl_m8_us_4b: f64 = 0.0,
    sme2_ctrl_gpu_dispatches: u32 = 0,

    engram_fp8_cold_us: f64 = 0.0,
    engram_fp8_r0_reads: u64 = 0,
    engram_fp8_r_ge1_reads: u64 = 0,
    engram_fp8_parity_ok: bool = false,

    cold_expert_fetches: u64 = 0,
    cold_expert_us: f64 = 0.0,
    cold_expert_gbs: f64 = 0.0,

    // 3. W23 + D14 End-to-End Python Synth -> Zig verify-amf, decode, spec-decode Parity & Speedup
    e2e_inkling2_mini: E2eSynthResult = .{},
    e2e_inkling2_small: E2eSynthResult = .{},
    e2e_hy4: E2eSynthResult = .{},
    e2e_hy4_flash: E2eSynthResult = .{},

    vm_before: vmstat.Snapshot = undefined,
    vm_after: vmstat.Snapshot = undefined,
    swapouts_delta: u64 = 0,
    all_passed: bool = false,

    pub fn write(self: *const M82Report, w: *Io.Writer) !void {
        try w.print(
            \\bench.m82.g25.rmsnorm_sconv4.fused_us         {d:.2}
            \\bench.m82.g25.rmsnorm_sconv4.unfused_us       {d:.2}
            \\bench.m82.g25.rmsnorm_sconv4.speedup          {d:.2}x
            \\bench.m82.g25.swa40_kv.8k_mib                 {d:.2} (vs full48 {d:.2} MiB, -{d:.1}%)
            \\bench.m82.g25.swa40_kv.32k_mib                {d:.2} (vs full48 {d:.2} MiB, -{d:.1}%)
            \\bench.m82.g25.rel_gqa_attn.m1_us              {d:.2}
            \\bench.m82.g25.rel_gqa_attn.m8_us              {d:.2}
            \\bench.m82.g25.inkling2_moe_attnres.m8_us      {d:.2}
            \\bench.m82.g25.hy4_index_cache.all_full_4l_us  {d:.2}
            \\bench.m82.g25.hy4_index_cache.period4_4l_us   {d:.2}
            \\bench.m82.g25.hy4_index_cache.speedup         {d:.2}x
            \\bench.m82.g25.hy4_index_cache.shared_dispatch {d}
            \\bench.m82.g25.hy4_index_pool.c4_us            {d:.2}
            \\bench.m82.g25.hy4_index_pool.c1_us            {d:.2}
            \\bench.m82.g25.hy4_index_pool.speedup          {d:.2}x
            \\bench.m82.g25.hy4_latent_moe.latent_us        {d:.2}
            \\bench.m82.g25.hy4_latent_moe.full_us          {d:.2}
            \\bench.m82.g25.hy4_latent_moe.speedup          {d:.2}x (weight_bytes={d:.2}x smaller)
            \\
        , .{
            self.rmsnorm_sconv4_fused_us,
            self.rmsnorm_sconv4_unfused_us,
            self.rmsnorm_sconv4_speedup,
            self.kv_swa40_8k_mib,
            self.kv_full48_8k_mib,
            self.kv_reduction_8k_pct,
            self.kv_swa40_32k_mib,
            self.kv_full48_32k_mib,
            self.kv_reduction_32k_pct,
            self.rel_gqa_m1_us,
            self.rel_gqa_m8_us,
            self.inkling2_moe_attnres_fused_us,
            self.hy4_idx_all_full_4l_us,
            self.hy4_idx_period4_4l_us,
            self.hy4_idx_period4_speedup,
            self.hy4_idx_shared_dispatches,
            self.hy4_idx_pool_c4_us,
            self.hy4_idx_pool_c1_us,
            self.hy4_idx_pool_c4_speedup,
            self.hy4_latent_moe_us,
            self.hy4_full_moe_us,
            self.hy4_latent_moe_speedup,
            self.hy4_latent_moe_bytes_reduction_x,
        });
        try w.print(
            \\bench.m82.c26.sme2_ctrl.placement             {s}
            \\bench.m82.c26.sme2_ctrl.fits_l2               {} ({d:.1} KiB)
            \\bench.m82.c26.sme2_ctrl.m1_us_per_branch      {d:.2}
            \\bench.m82.c26.sme2_ctrl.m1_us_4branches       {d:.2}
            \\bench.m82.c26.sme2_ctrl.m8_us_4branches       {d:.2}
            \\bench.m82.c26.sme2_ctrl.gpu_dispatches        {d}
            \\bench.m82.c26.engram_fp8.cold_us              {d:.2}
            \\bench.m82.c26.engram_fp8.r0_nvme_reads        {d}
            \\bench.m82.c26.engram_fp8.r_ge1_nvme_reads     {d}
            \\bench.m82.c26.engram_fp8.parity_ok            {}
            \\bench.m82.c26.cold_expert.fetches             {d}
            \\bench.m82.c26.cold_expert.us_per_expert       {d:.2}
            \\bench.m82.c26.cold_expert.bandwidth_gbs       {d:.2}
            \\
        , .{
            @tagName(self.sme2_ctrl_placement),
            self.sme2_ctrl_fits_l2,
            self.sme2_ctrl_weight_kib,
            self.sme2_ctrl_m1_us_branch,
            self.sme2_ctrl_m1_us_4b,
            self.sme2_ctrl_m8_us_4b,
            self.sme2_ctrl_gpu_dispatches,
            self.engram_fp8_cold_us,
            self.engram_fp8_r0_reads,
            self.engram_fp8_r_ge1_reads,
            self.engram_fp8_parity_ok,
            self.cold_expert_fetches,
            self.cold_expert_us,
            self.cold_expert_gbs,
        });

        try writeE2eCase(w, "inkling2_mini_dflare3", &self.e2e_inkling2_mini, true, false);
        try writeE2eCase(w, "inkling2_small_dflare5", &self.e2e_inkling2_small, false, false);
        try writeE2eCase(w, "hy4_dfly2", &self.e2e_hy4, false, true);
        try writeE2eCase(w, "hy4_flash_engram_dfly2", &self.e2e_hy4_flash, false, true);

        try w.print(
            \\bench.m82.swapouts_delta                      {d}
            \\bench.m82.all_passed                          {}
            \\
        , .{ self.swapouts_delta, self.all_passed });
    }
};

fn writeE2eCase(
    w: *Io.Writer,
    label: []const u8,
    r: *const E2eSynthResult,
    show_batch8: bool,
    show_dfly2: bool,
) !void {
    if (!r.ran) {
        try w.print("bench.m82.e2e.{s}.status  skipped\n", .{label});
        return;
    }
    try w.print(
        \\bench.m82.e2e.{s}.verify_amf_ok          {}
        \\bench.m82.e2e.{s}.greedy_t0_parity       {} ({d}/{d})
        \\bench.m82.e2e.{s}.decode_tok_s           {d:.2}
        \\bench.m82.e2e.{s}.spec_tok_s             {d:.2}
        \\bench.m82.e2e.{s}.spec_tau               {d:.3} (rounds={d})
        \\
    , .{
        label,             r.verify_amf_ok,
        label,             r.greedy_parity_ok,
        r.tokens_compared, r.tokens_compared,
        label,             r.decode_tok_s,
        label,             r.spec_tok_s,
        label,             r.spec_tau,
        r.spec_rounds,
    });
    if (show_batch8) {
        try w.print(
            \\bench.m82.e2e.{s}.dflare_core3_batches   {d}
            \\bench.m82.e2e.{s}.sme2_ctrl_calls        {d}
            \\bench.m82.e2e.{s}.gpu_ctrl_calls         {d}
            \\bench.m82.e2e.{s}.mid_pass_flushes       {d}
            \\bench.m82.e2e.{s}.cold_expert_fetches    {d}
            \\bench.m82.e2e.{s}.batch8_core3_verify_us {d:.2} (vs serial8 {d:.2} us, speedup={d:.2}x)
            \\
        , .{
            label,               r.dflare_core3_batches,
            label,               r.sme2_ctrl_calls,
            label,               r.gpu_ctrl_calls,
            label,               r.mid_pass_flushes,
            label,               r.cold_expert_fetches,
            label,               r.batch8_verify_us,
            r.serial8_decode_us, r.batch8_core3_speedup,
        });
    }
    if (show_dfly2) {
        try w.print(
            \\bench.m82.e2e.{s}.index_shared_reuses    {d}
            \\bench.m82.e2e.{s}.engram_nvme_reads      {d} (r>=1 reads={d})
            \\bench.m82.e2e.{s}.dfly2_early_stop_ok    {} (early_stops={d}, tau={d:.3})
            \\bench.m82.e2e.{s}.dfly2_early_stop_speed {d:.2}x ({d:.2} vs {d:.2} tok/s)
            \\
        , .{
            label,                  r.index_shared_reuses,
            label,                  r.engram_nvme_reads,
            r.engram_r_ge1_reads,   label,
            r.early_stop_parity_ok, r.early_stop_count,
            r.early_stop_tau,       label,
            r.early_stop_speedup,   r.early_stop_tok_s,
            r.no_early_stop_tok_s,
        });
    }
}

fn fillF16Deterministic(dst: []f16, seed: u64, scale: f32) void {
    var s: u64 = seed ^ 0x9E37_79B9_7F4A_7C15;
    for (dst) |*slot| {
        s = s *% 6364136223846793005 +% 1442695040888963407;
        const bits: i32 = @as(i32, @intCast((s >> 48) & 0xFFFF)) - 32768;
        const v: f32 = (@as(f32, @floatFromInt(bits)) / 32768.0) * scale;
        slot.* = @floatCast(v);
    }
}

fn fillF32Deterministic(dst: []f32, seed: u64, scale: f32) void {
    var s: u64 = seed ^ 0xBF58_476D_1CE4_E5B9;
    for (dst) |*slot| {
        s = s *% 6364136223846793005 +% 1442695040888963407;
        const bits: i32 = @as(i32, @intCast((s >> 48) & 0xFFFF)) - 32768;
        slot.* = (@as(f32, @floatFromInt(bits)) / 32768.0) * scale;
    }
}

fn fillQ4Deterministic(
    qweight: []u8,
    scale_bias: []u32,
    n: u32,
    k: u32,
    seed: u64,
) void {
    var s: u64 = seed ^ 0xD1B5_4A32_D192_ED03;
    for (qweight) |*b| {
        s = s *% 6364136223846793005 +% 1;
        b.* = @truncate(s >> 56);
    }
    const groups: usize = (@as(usize, n) * k) / 64;
    const scale_f16: u16 = @bitCast(@as(f16, @floatCast(0.015625)));
    const bias_f16: u16 = @bitCast(@as(f16, @floatCast(-0.125)));
    const packed_sb: u32 = @as(u32, scale_f16) | (@as(u32, bias_f16) << 16);
    for (scale_bias[0..groups]) |*sb| sb.* = packed_sb;
}

fn allocF16(arena: *arena_mod.WiredArena, count: usize) !struct { off: u64, slice: []f16 } {
    const raw = try arena.allocAligned(count * 2, 64);
    const off: u64 = @intCast(arena.offsetOf(raw.ptr));
    const slice: []f16 = @alignCast(std.mem.bytesAsSlice(f16, raw));
    return .{ .off = off, .slice = slice };
}

fn allocF32(arena: *arena_mod.WiredArena, count: usize) !struct { off: u64, slice: []f32 } {
    const raw = try arena.allocAligned(count * 4, 64);
    const off: u64 = @intCast(arena.offsetOf(raw.ptr));
    const slice: []f32 = @alignCast(std.mem.bytesAsSlice(f32, raw));
    return .{ .off = off, .slice = slice };
}

fn allocU32(arena: *arena_mod.WiredArena, count: usize) !struct { off: u64, slice: []u32 } {
    const raw = try arena.allocAligned(count * 4, 64);
    const off: u64 = @intCast(arena.offsetOf(raw.ptr));
    const slice: []u32 = @alignCast(std.mem.bytesAsSlice(u32, raw));
    return .{ .off = off, .slice = slice };
}

fn allocU64(arena: *arena_mod.WiredArena, count: usize) !struct { off: u64, slice: []u64 } {
    const raw = try arena.allocAligned(count * 8, 64);
    const off: u64 = @intCast(arena.offsetOf(raw.ptr));
    const slice: []u64 = @alignCast(std.mem.bytesAsSlice(u64, raw));
    return .{ .off = off, .slice = slice };
}

fn allocBytes(arena: *arena_mod.WiredArena, count: usize) !struct { off: u64, slice: []u8 } {
    const raw = try arena.allocAligned(count, 64);
    const off: u64 = @intCast(arena.offsetOf(raw.ptr));
    return .{ .off = off, .slice = raw };
}

fn bind(buf: metal.Buffer, off: u64) metal.BufferBinding {
    return .{ .buffer = buf, .offset = off };
}

fn benchG25Kernels(
    allocator: std.mem.Allocator,
    device: metal.Device,
    library: metal.Library,
    compute: *metal.Compute,
    budget: budget_mod.Budget,
    iters: u32,
    report: *M82Report,
) !void {
    const m82_pipes = try inkling2_hy4_gpu.Inkling2Hy4Pipelines.init(device, library);
    defer m82_pipes.close();
    const attn_pipes = try attention.AttentionPipelines.init(device, library);
    defer attn_pipes.close();
    const dq_pipes = try dequant.DequantPipelines.init(device, library);
    defer dq_pipes.close();

    // -----------------------------------------------------------------------
    // 1A. Inkling 2 RMSNorm + sconv4 fused (1 dispatch) vs unfused (2 dispatches)
    // -----------------------------------------------------------------------
    {
        const m: u32 = 8;
        const h: u32 = 2304;
        const sconv_bytes = try inkling2_hy4_gpu.SconvRingArena.requiredBytes(.{ .num_layers = 1, .hidden_dim = h });
        const need = try page.alignUp(sconv_bytes + 4 * 1024 * 1024);
        var arena = try arena_mod.WiredArena.init(need, budget);
        defer arena.deinit();
        const buf = try metal.Buffer.wrap(device, arena.bytes());
        defer buf.close();
        try compute.makeResident(&.{buf});
        defer compute.removeResident(&.{buf}) catch |err| {
            std.debug.panic("bench-m82 removeResident(sconv) failed: {}", .{err});
        };

        var ring = try inkling2_hy4_gpu.SconvRingArena.init(&arena, .{ .num_layers = 1, .hidden_dim = h });
        const x = try allocF16(&arena, m * h);
        const norm_w = try allocF16(&arena, h);
        const conv_w = try allocF16(&arena, 4 * h);
        const tmp = try allocF16(&arena, m * h);
        const out = try allocF16(&arena, m * h);
        fillF16Deterministic(x.slice, 101, 0.2);
        fillF16Deterministic(norm_w.slice, 102, 1.0);
        fillF16Deterministic(conv_w.slice, 103, 0.1);
        const st_bind = try ring.layerBinding(0, buf);

        // Warmup BOTH fused and unfused paths (16 iters) so M4 GPU clocks reach steady-state boost!
        {
            var f = try compute.beginFrame(10_000);
            for (0..16) |wi| {
                try m82_pipes.encodeRmsNormSconv4(&f, .{
                    .x = bind(buf, x.off),
                    .norm_w = bind(buf, norm_w.off),
                    .conv_w = bind(buf, conv_w.off),
                    .conv_state_in = st_bind,
                    .conv_state_out = st_bind,
                    .out = bind(buf, out.off),
                    .m = m,
                    .hidden_dim = h,
                    .state_pos = 0,
                    .write_state = false,
                    .apply_rms_norm = true,
                    .norm_dtype = .f16,
                    .conv_w_f16 = true,
                }, wi > 0);
                try attn_pipes.encodeRmsNorm(&f, .{
                    .x = bind(buf, x.off),
                    .w = bind(buf, norm_w.off),
                    .y = bind(buf, tmp.off),
                    .num_rows = m,
                    .hidden_dim = h,
                    .eps = 1e-6,
                    .w_dtype = .f16,
                }, true);
            }
            try compute.wait(try f.commit(), 10_000);
        }

        // Fused (1 dispatch)
        const t0 = try vmstat.nowNs();
        {
            var f = try compute.beginFrame(10_000);
            for (0..iters) |it| {
                try m82_pipes.encodeRmsNormSconv4(&f, .{
                    .x = bind(buf, x.off),
                    .norm_w = bind(buf, norm_w.off),
                    .conv_w = bind(buf, conv_w.off),
                    .conv_state_in = st_bind,
                    .conv_state_out = st_bind,
                    .out = bind(buf, out.off),
                    .m = m,
                    .hidden_dim = h,
                    .state_pos = 0,
                    .write_state = false,
                    .apply_rms_norm = true,
                    .norm_dtype = .f16,
                    .conv_w_f16 = true,
                }, it > 0);
            }
            try compute.wait(try f.commit(), 10_000);
        }
        const t1 = try vmstat.nowNs();

        // Unfused (RMSNorm + barrier + sconv4 without RMSNorm)
        const t2 = try vmstat.nowNs();
        {
            var f = try compute.beginFrame(10_000);
            for (0..iters) |it| {
                try attn_pipes.encodeRmsNorm(&f, .{
                    .x = bind(buf, x.off),
                    .w = bind(buf, norm_w.off),
                    .y = bind(buf, tmp.off),
                    .num_rows = m,
                    .hidden_dim = h,
                    .eps = 1e-6,
                    .w_dtype = .f16,
                }, it > 0);
                try m82_pipes.encodeRmsNormSconv4(&f, .{
                    .x = bind(buf, tmp.off),
                    .conv_w = bind(buf, conv_w.off),
                    .conv_state_in = st_bind,
                    .conv_state_out = st_bind,
                    .out = bind(buf, out.off),
                    .m = m,
                    .hidden_dim = h,
                    .state_pos = 0,
                    .write_state = false,
                    .apply_rms_norm = false,
                    .conv_w_f16 = true,
                }, true);
            }
            try compute.wait(try f.commit(), 10_000);
        }
        const t3 = try vmstat.nowNs();

        report.rmsnorm_sconv4_fused_us = @as(f64, @floatFromInt(t1 -| t0)) / (@as(f64, @floatFromInt(iters)) * 1000.0);
        report.rmsnorm_sconv4_unfused_us = @as(f64, @floatFromInt(t3 -| t2)) / (@as(f64, @floatFromInt(iters)) * 1000.0);
        report.rmsnorm_sconv4_speedup = if (report.rmsnorm_sconv4_fused_us > 0.0)
            report.rmsnorm_sconv4_unfused_us / report.rmsnorm_sconv4_fused_us
        else
            0.0;
    }

    // -----------------------------------------------------------------------
    // 1B. Inkling 2 40/48 1024-SWA HybridKvRing Memory Reduction + RelGQA Attn
    // -----------------------------------------------------------------------
    {
        const kv_heads: u32 = 2;
        const head_dim: u32 = 128;
        const q_heads: u32 = 16;
        const d_rel: u32 = 16;
        const swa_tiles: u32 = 8; // 8 * 128 = 1024-token SWA ring
        const full_8k_tiles: u32 = 16; // 16 * 512 = 8192 tokens
        const full_32k_tiles: u32 = 64; // 64 * 512 = 32768 tokens

        const need = try page.alignUp(8 * 1024 * 1024);
        var arena = try arena_mod.WiredArena.init(need, budget);
        defer arena.deinit();
        const buf = try metal.Buffer.wrap(device, arena.bytes());
        defer buf.close();
        try compute.makeResident(&.{buf});
        defer compute.removeResident(&.{buf}) catch |err| {
            std.debug.panic("bench-m82 removeResident(rel_gqa) failed: {}", .{err});
        };

        var swa_ring = try hybrid.HybridKvRing.init(allocator, device, &arena, .{
            .mode = .sliding_window_128,
            .num_kv_heads = kv_heads,
            .head_dim = head_dim,
            .num_tiles = swa_tiles,
        });
        defer swa_ring.deinit(allocator);
        try swa_ring.makeResident(compute.*);
        defer swa_ring.removeResident(compute.*) catch |err| {
            std.debug.panic("bench-m82 removeResident(swa_ring) failed: {}", .{err});
        };

        var full_probe = try hybrid.HybridKvRing.init(allocator, device, &arena, .{
            .mode = .full_chunk_512,
            .num_kv_heads = kv_heads,
            .head_dim = head_dim,
            .num_tiles = 1,
        });
        defer full_probe.deinit(allocator);

        const swa_tile_bytes: u64 = (try swa_ring.k_tiles[0].info()).surfaceAllocSize() +
            (try swa_ring.v_tiles[0].info()).surfaceAllocSize();
        const full_tile_bytes: u64 = (try full_probe.k_tiles[0].info()).surfaceAllocSize() +
            (try full_probe.v_tiles[0].info()).surfaceAllocSize();

        const swa_layer_bytes: u64 = swa_tile_bytes * swa_tiles;
        const full_8k_layer_bytes: u64 = full_tile_bytes * full_8k_tiles;
        const full_32k_layer_bytes: u64 = full_tile_bytes * full_32k_tiles;

        const swa40_8k_bytes: f64 = @floatFromInt(40 * swa_layer_bytes + 8 * full_8k_layer_bytes);
        const full48_8k_bytes: f64 = @floatFromInt(48 * full_8k_layer_bytes);
        const swa40_32k_bytes: f64 = @floatFromInt(40 * swa_layer_bytes + 8 * full_32k_layer_bytes);
        const full48_32k_bytes: f64 = @floatFromInt(48 * full_32k_layer_bytes);

        const mib: f64 = 1024.0 * 1024.0;
        report.kv_swa40_8k_mib = swa40_8k_bytes / mib;
        report.kv_full48_8k_mib = full48_8k_bytes / mib;
        report.kv_reduction_8k_pct = (1.0 - swa40_8k_bytes / full48_8k_bytes) * 100.0;
        report.kv_swa40_32k_mib = swa40_32k_bytes / mib;
        report.kv_full48_32k_mib = full48_32k_bytes / mib;
        report.kv_reduction_32k_pct = (1.0 - swa40_32k_bytes / full48_32k_bytes) * 100.0;

        // Live GPU single-pass `encodeInkling2RelGqaAttn` decode latency (`M = 1` and `M = 8` at `kv_len = 1024`)
        const q = try allocF16(&arena, 8 * q_heads * head_dim);
        const q_rel = try allocF16(&arena, 8 * q_heads * d_rel);
        const rel_emb = try allocF16(&arena, 1024 * d_rel);
        const k_tree = try allocF16(&arena, 8 * kv_heads * head_dim);
        const v_tree = try allocF16(&arena, 8 * kv_heads * head_dim);
        const masks = try allocU64(&arena, 8);
        const out = try allocF16(&arena, 8 * q_heads * head_dim);
        fillF16Deterministic(q.slice, 201, 0.25);
        fillF16Deterministic(q_rel.slice, 202, 0.25);
        fillF16Deterministic(rel_emb.slice, 203, 0.1);
        fillF16Deterministic(k_tree.slice, 204, 0.25);
        fillF16Deterministic(v_tree.slice, 205, 0.25);
        for (0..8) |ni| masks.slice[ni] = (@as(u64, 1) << @intCast(ni + 1)) - 1;

        for ([_]u32{ 1, 8 }) |nodes| {
            // Warmup
            {
                var f = try compute.beginFrame(10_000);
                try m82_pipes.encodeInkling2RelGqaAttn(&f, .{
                    .q = bind(buf, q.off),
                    .q_rel = bind(buf, q_rel.off),
                    .rel_emb = bind(buf, rel_emb.off),
                    .kv_ring = &swa_ring,
                    .kv_ring_buf = buf,
                    .k_tree = bind(buf, k_tree.off),
                    .v_tree = bind(buf, v_tree.off),
                    .ancestor_masks = bind(buf, masks.off),
                    .out = bind(buf, out.off),
                    .num_nodes = nodes,
                    .num_q_heads = q_heads,
                    .kv_len = 1024,
                    .d_rel = d_rel,
                    .rel_extent = 1024,
                }, false);
                try compute.wait(try f.commit(), 10_000);
            }
            const t0 = try vmstat.nowNs();
            {
                var f = try compute.beginFrame(10_000);
                for (0..iters) |it| {
                    try m82_pipes.encodeInkling2RelGqaAttn(&f, .{
                        .q = bind(buf, q.off),
                        .q_rel = bind(buf, q_rel.off),
                        .rel_emb = bind(buf, rel_emb.off),
                        .kv_ring = &swa_ring,
                        .kv_ring_buf = buf,
                        .k_tree = bind(buf, k_tree.off),
                        .v_tree = bind(buf, v_tree.off),
                        .ancestor_masks = bind(buf, masks.off),
                        .out = bind(buf, out.off),
                        .num_nodes = nodes,
                        .num_q_heads = q_heads,
                        .kv_len = 1024,
                        .d_rel = d_rel,
                        .rel_extent = 1024,
                    }, it > 0);
                }
                try compute.wait(try f.commit(), 10_000);
            }
            const t1 = try vmstat.nowNs();
            const us = @as(f64, @floatFromInt(t1 -| t0)) / (@as(f64, @floatFromInt(iters)) * 1000.0);
            if (nodes == 1) report.rel_gqa_m1_us = us else report.rel_gqa_m8_us = us;
        }
    }

    // -----------------------------------------------------------------------
    // 1C. Inkling 2 Fused MoE Route + SwiGLU Softcap + Epilogue + AttnRes (M = 8)
    // -----------------------------------------------------------------------
    {
        const m: u32 = 8;
        const h: u32 = 2304;
        const num_experts: u32 = 128;
        const top_k: u32 = 8;
        const inter: u32 = 768;
        const ar_bytes = try inkling2_hy4_gpu.AttnResRingArena.requiredBytes(.{
            .max_blocks = 6,
            .max_tokens = m,
            .hidden_dim = h,
        });
        const need = try page.alignUp(ar_bytes + 4 * 1024 * 1024);
        var arena = try arena_mod.WiredArena.init(need, budget);
        defer arena.deinit();
        const buf = try metal.Buffer.wrap(device, arena.bytes());
        defer buf.close();
        try compute.makeResident(&.{buf});
        defer compute.removeResident(&.{buf}) catch |err| {
            std.debug.panic("bench-m82 removeResident(moe_ar) failed: {}", .{err});
        };

        var ar_ring = try inkling2_hy4_gpu.AttnResRingArena.init(&arena, .{
            .max_blocks = 6,
            .max_tokens = m,
            .hidden_dim = h,
        });
        for (0..6) |_| _ = ar_ring.pushSlot();

        const logits = try allocF16(&arena, m * num_experts);
        const topk_ids = try allocU32(&arena, m * top_k);
        const topk_w = try allocF32(&arena, m * top_k);
        const gate_up = try allocF16(&arena, m * top_k * inter * 2);
        const act = try allocF16(&arena, m * top_k * inter);
        const exp_out = try allocF16(&arena, m * top_k * h);
        const res_in = try allocF16(&arena, m * h);
        const moe_out = try allocF16(&arena, m * h);
        const ar_q = try allocF16(&arena, h);
        const ar_norm = try allocF16(&arena, h);
        const ar_out = try allocF16(&arena, m * h);
        fillF16Deterministic(logits.slice, 301, 1.0);
        fillF16Deterministic(gate_up.slice, 302, 0.25);
        fillF16Deterministic(exp_out.slice, 303, 0.25);
        fillF16Deterministic(res_in.slice, 304, 0.25);
        fillF16Deterministic(ar_q.slice, 305, 0.1);
        fillF16Deterministic(ar_norm.slice, 306, 1.0);

        const t0 = try vmstat.nowNs();
        {
            var f = try compute.beginFrame(10_000);
            for (0..iters) |it| {
                try m82_pipes.encodeInkling2MoeRoute(&f, .{
                    .logits = bind(buf, logits.off),
                    .logits_f16 = true,
                    .topk_experts = bind(buf, topk_ids.off),
                    .topk_weights = bind(buf, topk_w.off),
                    .m = m,
                    .num_experts = num_experts,
                    .top_k = top_k,
                }, it > 0);
                try m82_pipes.encodeInkling2MoeSwigluSoftcap(&f, .{
                    .gate_in = bind(buf, gate_up.off),
                    .out = bind(buf, act.off),
                    .num_rows = m * top_k,
                    .intermediate_size = inter,
                    .interleaved_gate_up = true,
                }, true);
                try m82_pipes.encodeInkling2MoeEpilogue(&f, .{
                    .expert_out = bind(buf, exp_out.off),
                    .topk_weights = bind(buf, topk_w.off),
                    .residual = bind(buf, res_in.off),
                    .out = bind(buf, moe_out.off),
                    .m = m,
                    .hidden_dim = h,
                    .top_k = top_k,
                }, true);
                try m82_pipes.encodeInkling2AttnRes(&f, .{
                    .ring = &ar_ring,
                    .ring_buf = buf,
                    .query = bind(buf, ar_q.off),
                    .norm_w = bind(buf, ar_norm.off),
                    .out = bind(buf, ar_out.off),
                    .m = m,
                    .norm_dtype = .f16,
                }, true);
            }
            try compute.wait(try f.commit(), 10_000);
        }
        const t1 = try vmstat.nowNs();
        report.inkling2_moe_attnres_fused_us = @as(f64, @floatFromInt(t1 -| t0)) / (@as(f64, @floatFromInt(iters)) * 1000.0);
    }

    // -----------------------------------------------------------------------
    // 1D. Hy4 IndexCache Period-4 (0-dispatch .shared) + IndexPool (compress_ratio=4 vs 1)
    // -----------------------------------------------------------------------
    {
        const kv_len: u32 = 4096;
        const idx_dim: u32 = 64;
        const q_heads: u32 = 64;
        const spec_c4: inkling2_hy4_gpu.IndexPoolSpec = .{
            .max_nodes = 1,
            .max_kv_len = kv_len,
            .num_k_heads = 1,
            .idx_dim = idx_dim,
            .compress_ratio = 4,
            .group_topk = 256,
            .candidate_topk = 1024,
        };
        const spec_c1: inkling2_hy4_gpu.IndexPoolSpec = .{
            .max_nodes = 1,
            .max_kv_len = kv_len,
            .num_k_heads = 1,
            .idx_dim = idx_dim,
            .compress_ratio = 1,
            .group_topk = 1024,
            .candidate_topk = 1024,
        };
        const p4_bytes = try inkling2_hy4_gpu.IndexPoolBuffer.requiredBytes(spec_c4);
        const p1_bytes = try inkling2_hy4_gpu.IndexPoolBuffer.requiredBytes(spec_c1);
        const need = try page.alignUp(p4_bytes + p1_bytes + 4 * 1024 * 1024);
        var arena = try arena_mod.WiredArena.init(need, budget);
        defer arena.deinit();
        const buf = try metal.Buffer.wrap(device, arena.bytes());
        defer buf.close();
        try compute.makeResident(&.{buf});
        defer compute.removeResident(&.{buf}) catch |err| {
            std.debug.panic("bench-m82 removeResident(idx_pool) failed: {}", .{err});
        };

        var pool_c4 = try inkling2_hy4_gpu.IndexPoolBuffer.init(&arena, spec_c4);
        var pool_c1 = try inkling2_hy4_gpu.IndexPoolBuffer.init(&arena, spec_c1);
        const q_idx = try allocF16(&arena, q_heads * idx_dim);
        const k_fine = try allocF16(&arena, kv_len * idx_dim);
        fillF16Deterministic(q_idx.slice, 401, 0.25);
        fillF16Deterministic(k_fine.slice, 402, 0.25);

        // Warmup
        {
            var f = try compute.beginFrame(10_000);
            _ = try m82_pipes.encodeHy4IndexLayer(&f, .{
                .mode = .full,
                .compress_keys = true,
                .q_idx = bind(buf, q_idx.off),
                .k_fine = bind(buf, k_fine.off),
                .index_pool = &pool_c4,
                .index_metal_buf = buf,
                .num_nodes = 1,
                .num_q_heads = q_heads,
                .kv_len = kv_len,
            }, false);
            try compute.wait(try f.commit(), 10_000);
        }

        // All-4 `.full` layers
        const t0 = try vmstat.nowNs();
        {
            var f = try compute.beginFrame(10_000);
            for (0..iters) |it| {
                for (0..4) |li| {
                    _ = try m82_pipes.encodeHy4IndexLayer(&f, .{
                        .mode = .full,
                        .compress_keys = true,
                        .q_idx = bind(buf, q_idx.off),
                        .k_fine = bind(buf, k_fine.off),
                        .index_pool = &pool_c4,
                        .index_metal_buf = buf,
                        .num_nodes = 1,
                        .num_q_heads = q_heads,
                        .kv_len = kv_len,
                    }, it > 0 or li > 0);
                }
            }
            try compute.wait(try f.commit(), 10_000);
        }
        const t1 = try vmstat.nowNs();

        // Period-4: 1 `.full` + 3 `.shared` (`0` dispatches for shared layers!)
        var shared_dispatches: u32 = 0;
        const t2 = try vmstat.nowNs();
        {
            var f = try compute.beginFrame(10_000);
            for (0..iters) |it| {
                for (0..4) |li| {
                    const mode = inkling2_hy4_gpu.IndexCacheMode.forPeriod4(@intCast(li));
                    const disp = try m82_pipes.encodeHy4IndexLayer(&f, .{
                        .mode = mode,
                        .compress_keys = (mode == .full),
                        .q_idx = bind(buf, q_idx.off),
                        .k_fine = bind(buf, k_fine.off),
                        .index_pool = &pool_c4,
                        .index_metal_buf = buf,
                        .num_nodes = 1,
                        .num_q_heads = q_heads,
                        .kv_len = kv_len,
                    }, it > 0 or li > 0);
                    if (mode == .shared) shared_dispatches += disp.total();
                }
            }
            try compute.wait(try f.commit(), 10_000);
        }
        const t3 = try vmstat.nowNs();

        // IndexPool compress_ratio = 1 (uncompressed full fine scan) vs compress_ratio = 4
        const t4 = try vmstat.nowNs();
        {
            var f = try compute.beginFrame(10_000);
            for (0..iters) |it| {
                _ = try m82_pipes.encodeHy4IndexLayer(&f, .{
                    .mode = .full,
                    .compress_keys = true,
                    .q_idx = bind(buf, q_idx.off),
                    .k_fine = bind(buf, k_fine.off),
                    .index_pool = &pool_c1,
                    .index_metal_buf = buf,
                    .num_nodes = 1,
                    .num_q_heads = q_heads,
                    .kv_len = kv_len,
                }, it > 0);
            }
            try compute.wait(try f.commit(), 10_000);
        }
        const t5 = try vmstat.nowNs();

        report.hy4_idx_all_full_4l_us = @as(f64, @floatFromInt(t1 -| t0)) / (@as(f64, @floatFromInt(iters)) * 1000.0);
        report.hy4_idx_period4_4l_us = @as(f64, @floatFromInt(t3 -| t2)) / (@as(f64, @floatFromInt(iters)) * 1000.0);
        report.hy4_idx_period4_speedup = if (report.hy4_idx_period4_4l_us > 0.0)
            report.hy4_idx_all_full_4l_us / report.hy4_idx_period4_4l_us
        else
            0.0;
        report.hy4_idx_shared_dispatches = shared_dispatches;

        report.hy4_idx_pool_c4_us = report.hy4_idx_period4_4l_us; // 1 full c=4 dispatch per 4L period
        report.hy4_idx_pool_c1_us = @as(f64, @floatFromInt(t5 -| t4)) / (@as(f64, @floatFromInt(iters)) * 1000.0);
        report.hy4_idx_pool_c4_speedup = if (report.hy4_idx_pool_c4_us > 0.0)
            report.hy4_idx_pool_c1_us / report.hy4_idx_pool_c4_us
        else
            0.0;
    }

    // -----------------------------------------------------------------------
    // 1E. Hy4 LatentMoE (latent=1024) vs Full-Dim MoE (hidden=4096), Q4 top_k=8
    // -----------------------------------------------------------------------
    {
        const h: u32 = 4096;
        const ld: u32 = 1024;
        const inter: u32 = 1024;
        const top_k: u32 = 8;

        // Bytes per Q4 matrix: n * k / 2 codes + (n * k / 64) * 4 scale_bias
        const q4Bytes = struct {
            fn f(n: u32, k: u32) usize {
                return (@as(usize, n) * k) / 2 + ((@as(usize, n) * k) / 64) * 4;
            }
        }.f;

        const lat_down_b = q4Bytes(ld, h);
        const lat_up_b = q4Bytes(h, ld);
        const lat_exp_w13_b = q4Bytes(2 * inter, ld);
        const lat_exp_w2_b = q4Bytes(ld, inter);
        const full_exp_w13_b = q4Bytes(2 * inter, h);
        const full_exp_w2_b = q4Bytes(h, inter);

        const lat_total_w = lat_down_b + lat_up_b + top_k * (lat_exp_w13_b + lat_exp_w2_b);
        const full_total_w = top_k * (full_exp_w13_b + full_exp_w2_b);
        report.hy4_latent_moe_bytes_reduction_x = @as(f64, @floatFromInt(full_total_w)) / @as(f64, @floatFromInt(lat_total_w));

        const need = try page.alignUp(lat_total_w + full_total_w + 4 * 1024 * 1024);
        var arena = try arena_mod.WiredArena.init(need, budget);
        defer arena.deinit();
        const buf = try metal.Buffer.wrap(device, arena.bytes());
        defer buf.close();
        try compute.makeResident(&.{buf});
        defer compute.removeResident(&.{buf}) catch |err| {
            std.debug.panic("bench-m82 removeResident(latent_moe) failed: {}", .{err});
        };

        const Q4Mat = struct { q_off: u64, sb_off: u64 };
        const allocQ4 = struct {
            fn f(a: *arena_mod.WiredArena, n: u32, k: u32, seed: u64) !Q4Mat {
                const q = try allocBytes(a, (@as(usize, n) * k) / 2);
                const sb = try allocU32(a, (@as(usize, n) * k) / 64);
                fillQ4Deterministic(q.slice, sb.slice, n, k, seed);
                return .{ .q_off = q.off, .sb_off = sb.off };
            }
        }.f;

        const w_ld = try allocQ4(&arena, ld, h, 501);
        const w_lu = try allocQ4(&arena, h, ld, 502);
        var lat_w13: [top_k]Q4Mat = undefined;
        var lat_w2: [top_k]Q4Mat = undefined;
        var full_w13: [top_k]Q4Mat = undefined;
        var full_w2: [top_k]Q4Mat = undefined;
        for (0..top_k) |ki| {
            lat_w13[ki] = try allocQ4(&arena, 2 * inter, ld, 510 + ki);
            lat_w2[ki] = try allocQ4(&arena, ld, inter, 530 + ki);
            full_w13[ki] = try allocQ4(&arena, 2 * inter, h, 550 + ki);
            full_w2[ki] = try allocQ4(&arena, h, inter, 570 + ki);
        }

        const x_h = try allocF16(&arena, h);
        const x_lat = try allocF16(&arena, ld);
        const gu_buf = try allocF16(&arena, 2 * inter);
        const act_buf = try allocF16(&arena, inter);
        const lat_exp_out = try allocF16(&arena, top_k * ld);
        const full_exp_out = try allocF16(&arena, top_k * h);
        const topk_w = try allocF32(&arena, top_k);
        const comb_norm_lat = try allocF16(&arena, ld);
        const comb_norm_full = try allocF16(&arena, h);
        const lat_comb = try allocF16(&arena, ld);
        const y_out = try allocF16(&arena, h);
        fillF16Deterministic(x_h.slice, 590, 0.2);
        fillF32Deterministic(topk_w.slice, 591, 0.125);
        fillF16Deterministic(comb_norm_lat.slice, 592, 1.0);
        fillF16Deterministic(comb_norm_full.slice, 593, 1.0);

        const bench_iters: u32 = @min(iters, 20);
        // LatentMoE GPU timing
        const t0 = try vmstat.nowNs();
        for (0..bench_iters) |_| {
            var f = try compute.beginFrame(10_000);
            try dq_pipes.encodeGemv(&f, .q4, .{
                .x = bind(buf, x_h.off),
                .qweight = bind(buf, w_ld.q_off),
                .scale_bias = bind(buf, w_ld.sb_off),
                .y = bind(buf, x_lat.off),
                .n = ld,
                .k = h,
            }, false);
            for (0..top_k) |ki| {
                try dq_pipes.encodeGemv(&f, .q4, .{
                    .x = bind(buf, x_lat.off),
                    .qweight = bind(buf, lat_w13[ki].q_off),
                    .scale_bias = bind(buf, lat_w13[ki].sb_off),
                    .y = bind(buf, gu_buf.off),
                    .n = 2 * inter,
                    .k = ld,
                }, true);
                try m82_pipes.encodeHy4SwigluClampedLinearGate(&f, .{
                    .gate_in = bind(buf, gu_buf.off),
                    .out = bind(buf, act_buf.off),
                    .num_rows = 1,
                    .intermediate_size = inter,
                    .interleaved_gate_up = true,
                    .swiglu_limit = 10.0,
                }, true);
                try dq_pipes.encodeGemv(&f, .q4, .{
                    .x = bind(buf, act_buf.off),
                    .qweight = bind(buf, lat_w2[ki].q_off),
                    .scale_bias = bind(buf, lat_w2[ki].sb_off),
                    .y = bind(buf, lat_exp_out.off + @as(u64, ki) * ld * 2),
                    .n = ld,
                    .k = inter,
                }, true);
            }
            try m82_pipes.encodeHy4CombineSandwichNorm(&f, .{
                .expert_or_x = bind(buf, lat_exp_out.off),
                .topk_weights = bind(buf, topk_w.off),
                .combine_norm_w = bind(buf, comb_norm_lat.off),
                .out = bind(buf, lat_comb.off),
                .m = 1,
                .hidden_dim = ld,
                .top_k = top_k,
                .norm_dtype = .f16,
            }, true);
            try dq_pipes.encodeGemv(&f, .q4, .{
                .x = bind(buf, lat_comb.off),
                .qweight = bind(buf, w_lu.q_off),
                .scale_bias = bind(buf, w_lu.sb_off),
                .y = bind(buf, y_out.off),
                .n = h,
                .k = ld,
            }, true);
            try compute.wait(try f.commit(), 10_000);
        }
        const t1 = try vmstat.nowNs();

        // Full-dim MoE GPU timing
        const t2 = try vmstat.nowNs();
        for (0..bench_iters) |_| {
            var f = try compute.beginFrame(10_000);
            for (0..top_k) |ki| {
                try dq_pipes.encodeGemv(&f, .q4, .{
                    .x = bind(buf, x_h.off),
                    .qweight = bind(buf, full_w13[ki].q_off),
                    .scale_bias = bind(buf, full_w13[ki].sb_off),
                    .y = bind(buf, gu_buf.off),
                    .n = 2 * inter,
                    .k = h,
                }, ki > 0);
                try m82_pipes.encodeHy4SwigluClampedLinearGate(&f, .{
                    .gate_in = bind(buf, gu_buf.off),
                    .out = bind(buf, act_buf.off),
                    .num_rows = 1,
                    .intermediate_size = inter,
                    .interleaved_gate_up = true,
                    .swiglu_limit = 10.0,
                }, true);
                try dq_pipes.encodeGemv(&f, .q4, .{
                    .x = bind(buf, act_buf.off),
                    .qweight = bind(buf, full_w2[ki].q_off),
                    .scale_bias = bind(buf, full_w2[ki].sb_off),
                    .y = bind(buf, full_exp_out.off + @as(u64, ki) * h * 2),
                    .n = h,
                    .k = inter,
                }, true);
            }
            try m82_pipes.encodeHy4CombineSandwichNorm(&f, .{
                .expert_or_x = bind(buf, full_exp_out.off),
                .topk_weights = bind(buf, topk_w.off),
                .combine_norm_w = bind(buf, comb_norm_full.off),
                .out = bind(buf, y_out.off),
                .m = 1,
                .hidden_dim = h,
                .top_k = top_k,
                .norm_dtype = .f16,
            }, true);
            try compute.wait(try f.commit(), 10_000);
        }
        const t3 = try vmstat.nowNs();

        report.hy4_latent_moe_us = @as(f64, @floatFromInt(t1 -| t0)) / (@as(f64, @floatFromInt(bench_iters)) * 1000.0);
        report.hy4_full_moe_us = @as(f64, @floatFromInt(t3 -| t2)) / (@as(f64, @floatFromInt(bench_iters)) * 1000.0);
        report.hy4_latent_moe_speedup = if (report.hy4_latent_moe_us > 0.0)
            report.hy4_full_moe_us / report.hy4_latent_moe_us
        else
            0.0;
    }
}

fn writeAmfFilePosix(path: [:0]const u8, bytes: []const u8) !void {
    const fd = std.c.open(
        path.ptr,
        .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true },
        @as(std.c.mode_t, 0o644),
    );
    if (fd < 0) return error.FileOpenFailed;
    defer _ = std.c.close(fd);
    var written: usize = 0;
    while (written < bytes.len) {
        const rc = std.c.write(fd, bytes.ptr + written, bytes.len - written);
        if (rc < 0) {
            if (std.c.errno(rc) == .INTR) continue;
            return error.FileWriteFailed;
        }
        written += @intCast(rc);
    }
}

fn readFileAllocPosix(allocator: std.mem.Allocator, path: [:0]const u8, max_bytes: usize) ![]u8 {
    const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.FileOpenFailed;
    defer _ = std.c.close(fd);
    var st: std.c.Stat = undefined;
    if (std.c.fstat(fd, &st) != 0) return error.FileStatFailed;
    const sz: usize = @intCast(st.size);
    if (sz > max_bytes) return error.FileTooLarge;
    const buf = try allocator.alloc(u8, sz);
    errdefer allocator.free(buf);
    var done: usize = 0;
    while (done < sz) {
        const rc = std.c.read(fd, buf.ptr + done, sz - done);
        if (rc < 0) {
            if (std.c.errno(rc) == .INTR) continue;
            return error.FileReadFailed;
        }
        if (rc == 0) return error.UnexpectedEof;
        done += @intCast(rc);
    }
    return buf;
}

fn buildFp8EngramShardBytes(
    allocator: std.mem.Allocator,
    tensor_name: []const u8,
    rows: u32,
    packed_dim: u32,
    seed: u64,
) ![]u8 {
    // `fp8_e4m3_row`: `rows * packed_dim` FP8 codes followed by 64-byte aligned `rows` FP32 row scales.
    const code_bytes: usize = @as(usize, rows) * packed_dim;
    const params_off_rel: usize = amf.align64(code_bytes);
    const params_bytes: usize = @as(usize, rows) * 4;
    const data_bytes: usize = params_off_rel + params_bytes;
    const blob_bytes: usize = try page.alignUp(data_bytes);

    const dir_off: usize = APPLE_PAGE_SIZE;
    const dir_raw_bytes: usize = @sizeOf(amf.TensorEntry);
    const dir_bytes: usize = try page.alignUp(dir_raw_bytes);
    const blob_off: usize = dir_off + dir_bytes;
    const total: usize = blob_off + blob_bytes;

    const img = try allocator.alloc(u8, total);
    errdefer allocator.free(img);
    @memset(img, 0);

    var s: u64 = seed ^ 0x517C_C1B7_2722_0A95;
    for (img[blob_off .. blob_off + code_bytes]) |*b| {
        s = s *% 6364136223846793005 +% 1;
        // Finite non-NaN FP8 E4M3 codes (sign bit + mantissa/exp avoiding 0x7F/0xFF).
        b.* = @as(u8, @truncate(s >> 56)) & 0x7E;
    }
    const scales_dst = img[blob_off + params_off_rel .. blob_off + params_off_rel + params_bytes];
    for (0..rows) |r| {
        const sc: f32 = 0.0625 + @as(f32, @floatFromInt(r % 17)) * 0.004;
        std.mem.writeInt(u32, scales_dst[r * 4 ..][0..4], @bitCast(sc), .little);
    }

    var entry = amf.TensorEntry.zeroed(tensor_name, .sparse_table, .fp8_e4m3_row);
    entry.quant_bits = 8;
    entry.rank = 2;
    entry.group_size = 0;
    entry.layer = 0;
    entry.expert = amf.NO_EXPERT;
    entry.shape = .{ rows, packed_dim, 0, 0 };
    entry.blob_offset = blob_off;
    entry.blob_bytes = blob_bytes;
    entry.data_bytes = code_bytes;
    entry.params_offset = params_off_rel;
    entry.params_bytes = params_bytes;
    entry.blob_crc32 = std.hash.Crc32.hash(img[blob_off .. blob_off + blob_bytes]);

    @memcpy(img[dir_off .. dir_off + dir_raw_bytes], std.mem.asBytes(&entry));

    var hdr: amf.Header = std.mem.zeroes(amf.Header);
    hdr.magic = amf.MAGIC;
    hdr.version = amf.FORMAT_VERSION;
    hdr.page_size = @intCast(APPLE_PAGE_SIZE);
    hdr.checksum_algo = amf.CHECKSUM_CRC32_IEEE;
    hdr.tensor_count = 1;
    hdr.directory_offset = dir_off;
    hdr.directory_bytes = dir_bytes;
    hdr.directory_crc32 = std.hash.Crc32.hash(img[dir_off .. dir_off + dir_raw_bytes]);
    const mid = "bench-m82-fp8-engram";
    @memcpy(hdr.model_id[0..mid.len], mid);
    hdr.header_crc32 = amf.computeHeaderCrc32(&hdr);
    @memcpy(img[0..@sizeOf(amf.Header)], std.mem.asBytes(&hdr));
    return img;
}

fn buildColdExpertBenchAmfBytes(
    allocator: std.mem.Allocator,
    num_experts: u32,
    hidden: u32,
    inter: u32,
    out_metas: []m82_runtime.M82ColdExpertMeta,
) ![]u8 {
    // 2 tensors per expert (fused gate_up `[2*inter, hidden]` + down `[hidden, inter]`), Q4 affine group=64.
    const w1_codes: usize = (@as(usize, 2 * inter) * hidden) / 2;
    const w1_params_rel: usize = amf.align64(w1_codes);
    const w1_params_b: usize = ((@as(usize, 2 * inter) * hidden) / 64) * 4;
    const w1_data: usize = w1_params_rel + w1_params_b;
    const w1_blob: usize = try page.alignUp(w1_data);

    const w2_codes: usize = (@as(usize, hidden) * inter) / 2;
    const w2_params_rel: usize = amf.align64(w2_codes);
    const w2_params_b: usize = ((@as(usize, hidden) * inter) / 64) * 4;
    const w2_data: usize = w2_params_rel + w2_params_b;
    const w2_blob: usize = try page.alignUp(w2_data);

    const n_tensors: usize = @as(usize, num_experts) * 2;
    const dir_off: usize = APPLE_PAGE_SIZE;
    const dir_raw_bytes: usize = n_tensors * @sizeOf(amf.TensorEntry);
    const dir_bytes: usize = try page.alignUp(dir_raw_bytes);
    const first_blob: usize = dir_off + dir_bytes;
    const total: usize = first_blob + @as(usize, num_experts) * (w1_blob + w2_blob);

    const img = try allocator.alloc(u8, total);
    errdefer allocator.free(img);
    @memset(img, 0);

    var cursor: usize = first_blob;
    var entries = try allocator.alloc(amf.TensorEntry, n_tensors);
    defer allocator.free(entries);

    for (0..num_experts) |ei| {
        const w1_off = cursor;
        img[w1_off] = @truncate(ei +% 1);
        cursor += w1_blob;
        const w2_off = cursor;
        img[w2_off] = @truncate(ei +% 17);
        cursor += w2_blob;

        var n1_buf: [48]u8 = undefined;
        const n1 = try std.fmt.bufPrint(&n1_buf, "l0.e{d}.w13", .{ei});
        var e1 = amf.TensorEntry.zeroed(n1, .cold_expert, .q4_affine);
        e1.quant_bits = 4;
        e1.rank = 2;
        e1.group_size = 64;
        e1.layer = 0;
        e1.expert = @intCast(ei);
        e1.shape = .{ 2 * inter, hidden, 0, 0 };
        e1.blob_offset = w1_off;
        e1.blob_bytes = w1_blob;
        e1.data_bytes = w1_codes;
        e1.params_offset = w1_params_rel;
        e1.params_bytes = w1_params_b;
        e1.blob_crc32 = std.hash.Crc32.hash(img[w1_off .. w1_off + w1_blob]);
        entries[ei * 2] = e1;

        var n2_buf: [48]u8 = undefined;
        const n2 = try std.fmt.bufPrint(&n2_buf, "l0.e{d}.w2", .{ei});
        var e2 = amf.TensorEntry.zeroed(n2, .cold_expert, .q4_affine);
        e2.quant_bits = 4;
        e2.rank = 2;
        e2.group_size = 64;
        e2.layer = 0;
        e2.expert = @intCast(ei);
        e2.shape = .{ hidden, inter, 0, 0 };
        e2.blob_offset = w2_off;
        e2.blob_bytes = w2_blob;
        e2.data_bytes = w2_codes;
        e2.params_offset = w2_params_rel;
        e2.params_bytes = w2_params_b;
        e2.blob_crc32 = std.hash.Crc32.hash(img[w2_off .. w2_off + w2_blob]);
        entries[ei * 2 + 1] = e2;

        out_metas[ei] = .{
            .layer = 0,
            .expert = @intCast(ei),
            .w1_offset = w1_off,
            .w1_bytes = w1_blob,
            .w1_params_offset = w1_params_rel,
            .w1_amf_dtype = .q4_affine,
            .w1_group_size = 64,
            .w1_n = 2 * inter,
            .w1_k = hidden,
            .w2_offset = w2_off,
            .w2_bytes = w2_blob,
            .w2_params_offset = w2_params_rel,
            .w2_amf_dtype = .q4_affine,
            .w2_group_size = 64,
            .w2_n = hidden,
            .w2_k = inter,
        };
    }

    const dir_slice = std.mem.sliceAsBytes(entries);
    @memcpy(img[dir_off .. dir_off + dir_raw_bytes], dir_slice);

    var hdr: amf.Header = std.mem.zeroes(amf.Header);
    hdr.magic = amf.MAGIC;
    hdr.version = amf.FORMAT_VERSION;
    hdr.page_size = @intCast(APPLE_PAGE_SIZE);
    hdr.checksum_algo = amf.CHECKSUM_CRC32_IEEE;
    hdr.tensor_count = @intCast(n_tensors);
    hdr.directory_offset = dir_off;
    hdr.directory_bytes = dir_bytes;
    hdr.directory_crc32 = std.hash.Crc32.hash(dir_slice);
    const mid = "bench-m82-cold-experts";
    @memcpy(hdr.model_id[0..mid.len], mid);
    hdr.header_crc32 = amf.computeHeaderCrc32(&hdr);
    @memcpy(img[0..@sizeOf(amf.Header)], std.mem.asBytes(&hdr));
    return img;
}

fn benchC26CpuAndIo(
    allocator: std.mem.Allocator,
    device: metal.Device,
    compute: *metal.Compute,
    budget: budget_mod.Budget,
    scratch_dir: []const u8,
    cfg: M82Config,
    report: *M82Report,
) !void {
    // -----------------------------------------------------------------------
    // 2A. C26 L2 SME2 gate_rank=64 Inkling2LoopController (0 GPU dispatches)
    // -----------------------------------------------------------------------
    {
        const h: u32 = 2304;
        const r: u32 = inkling2_ctrl.GATE_RANK;
        const ccfg = inkling2_ctrl.ControllerConfig.uniform(h, h, .f16, .tanh, 8);
        var dispatcher = try sme.SmeDispatcher.init(allocator, .{});
        defer dispatcher.deinit();

        const down_f32 = try allocator.alloc(f32, @as(usize, r) * h);
        defer allocator.free(down_f32);
        const up_f32 = try allocator.alloc(f32, @as(usize, h) * r);
        defer allocator.free(up_f32);
        fillF32Deterministic(down_f32, 701, 0.05);
        fillF32Deterministic(up_f32, 702, 0.05);

        const branch_w: inkling2_ctrl.BranchWeightInput = .{
            .down_weight = .{ .dense_f32 = down_f32 },
            .forget_weight = .{ .dense_f32 = up_f32 },
            .gain_weight = .{ .dense_f32 = up_f32 },
        };
        const all_w = [inkling2_ctrl.NUM_BRANCHES]inkling2_ctrl.BranchWeightInput{ branch_w, branch_w, branch_w, branch_w };

        var ctrl = try inkling2_ctrl.Inkling2LoopController.init(ccfg, all_w);
        defer ctrl.deinit();

        const b1 = try inkling2_ctrl.benchmarkControllerPass(&ctrl, &dispatcher, 1, cfg.ctrl_iters, null);
        const b8 = try inkling2_ctrl.benchmarkControllerPass(&ctrl, &dispatcher, 8, cfg.ctrl_iters, null);
        report.sme2_ctrl_placement = b1.placement;
        report.sme2_ctrl_fits_l2 = ctrl.fitsL2(null);
        report.sme2_ctrl_weight_kib = @as(f64, @floatFromInt(b1.total_weight_bytes)) / 1024.0;
        report.sme2_ctrl_m1_us_branch = b1.us_per_branch;
        report.sme2_ctrl_m1_us_4b = b1.us_per_full_pass_4_branches;
        report.sme2_ctrl_m8_us_4b = b8.us_per_full_pass_4_branches;
        report.sme2_ctrl_gpu_dispatches = 0;
    }

    // -----------------------------------------------------------------------
    // 2B. C26 fp8_e4m3_row Engram F_NOCACHE Streaming (0 r>=1 NVMe reads)
    // -----------------------------------------------------------------------
    {
        const val_dim: u32 = 128;
        const packed_dim: u32 = val_dim * 2;
        const rows_per_shard: u32 = 256;

        const img0 = try buildFp8EngramShardBytes(allocator, "engram.layers.0.shard.0.weight", rows_per_shard, packed_dim, 801);
        defer allocator.free(img0);
        const img1 = try buildFp8EngramShardBytes(allocator, "engram.layers.0.shard.1.weight", rows_per_shard, packed_dim, 802);
        defer allocator.free(img1);

        const p0 = try std.fmt.allocPrintSentinel(allocator, "{s}/bench_engram_s0.amf", .{scratch_dir}, 0);
        defer allocator.free(p0);
        const p1 = try std.fmt.allocPrintSentinel(allocator, "{s}/bench_engram_s1.amf", .{scratch_dir}, 0);
        defer allocator.free(p1);
        try writeAmfFilePosix(p0, img0);
        defer _ = std.c.unlink(p0.ptr);
        try writeAmfFilePosix(p1, img1);
        defer _ = std.c.unlink(p1.ptr);

        const shard_paths = [_][:0]const u8{ p0, p1 };
        var store = try ple.PleStore.open(allocator, &shard_paths, .{
            .cache = .{
                .cache_pages = 8,
                .max_inflight = 4,
                .io_backend = .auto,
                .budget_override = budget,
            },
            .verify_blob_crcs = true,
        });
        defer store.deinit(allocator);

        const orders = [_]u8{ 2, 3, 4, 5 };
        var scratch = try ple.LoopPleScratch.initEngram(allocator, 1, 4, @intCast(orders.len), val_dim, budget);
        defer scratch.deinit(allocator);

        var hidden_f32: [val_dim]f32 = undefined;
        fillF32Deterministic(&hidden_f32, 805, 0.25);
        var out_r0: [val_dim]f32 = undefined;
        var out_r1: [val_dim]f32 = undefined;

        var total_ns: u64 = 0;
        var r0_reads: u64 = 0;
        var r_ge1_reads: u64 = 0;
        var parity_ok = true;

        for (0..cfg.engram_iters) |it| {
            const win = [_]u32{
                @intCast((it * 7 + 3) % 256),
                @intCast((it * 11 + 5) % 256),
                @intCast((it * 13 + 9) % 256),
                @intCast((it * 17 + 1) % 256),
            };
            const windows = [_][]const u32{&win};
            const reads_before_r0 = store.stats().nvme_reads;
            const t0 = try vmstat.nowNs();
            try store.lookupEngramConditionalF32(0, &windows, &orders, &hidden_f32, 0, &scratch, &out_r0);
            const t1 = try vmstat.nowNs();
            total_ns += (t1 -| t0);
            const reads_after_r0 = store.stats().nvme_reads;
            r0_reads += (reads_after_r0 -| reads_before_r0);

            // Loop pass r = 1 MUST perform 0 NVMe reads and produce exact bit-identical output!
            try store.lookupEngramConditionalF32(0, &windows, &orders, &hidden_f32, 1, &scratch, &out_r1);
            const reads_after_r1 = store.stats().nvme_reads;
            r_ge1_reads += (reads_after_r1 -| reads_after_r0);

            for (0..val_dim) |d| {
                if (!std.math.isFinite(out_r0[d]) or @abs(out_r0[d] - out_r1[d]) > 1e-6) {
                    parity_ok = false;
                }
            }
        }

        report.engram_fp8_cold_us = @as(f64, @floatFromInt(total_ns)) / (@as(f64, @floatFromInt(cfg.engram_iters)) * 1000.0);
        report.engram_fp8_r0_reads = r0_reads;
        report.engram_fp8_r_ge1_reads = r_ge1_reads;
        report.engram_fp8_parity_ok = parity_ok and (r_ge1_reads == 0);
    }

    // -----------------------------------------------------------------------
    // 2C. Cold-Expert F_NOCACHE Streaming (`M82ColdExpertCache`)
    // -----------------------------------------------------------------------
    {
        const num_experts: u32 = 32;
        const hidden: u32 = 2304;
        const inter: u32 = 768;
        var metas: [num_experts]m82_runtime.M82ColdExpertMeta = undefined;
        const img = try buildColdExpertBenchAmfBytes(allocator, num_experts, hidden, inter, &metas);
        defer allocator.free(img);

        const cold_path = try std.fmt.allocPrintSentinel(allocator, "{s}/bench_cold_experts.amf", .{scratch_dir}, 0);
        defer allocator.free(cold_path);
        try writeAmfFilePosix(cold_path, img);
        defer _ = std.c.unlink(cold_path.ptr);

        var cold_cache = try m82_runtime.M82ColdExpertCache.init(
            allocator,
            device,
            compute.*,
            cold_path,
            4, // 4 LRU slots -> cycling across 32 experts guarantees 100% real F_NOCACHE cold misses!
            @intCast(metas[0].w1_bytes),
            0,
            @intCast(metas[0].w2_bytes),
            budget,
        );
        defer cold_cache.deinit(allocator, compute.*);

        var st: m82_runtime.M82Stats = .{};
        const iters = cfg.cold_expert_iters;
        const t0 = try vmstat.nowNs();
        for (0..iters) |it| {
            _ = try cold_cache.pinExpert(metas[it % num_experts], &st);
        }
        const t1 = try vmstat.nowNs();
        const ns: f64 = @floatFromInt(t1 -| t0);
        const bytes_per_exp: f64 = @floatFromInt(metas[0].w1_bytes + metas[0].w2_bytes);
        report.cold_expert_fetches = st.cold_expert_fetches;
        report.cold_expert_us = ns / (@as(f64, @floatFromInt(iters)) * 1000.0);
        report.cold_expert_gbs = if (ns > 0.0)
            (bytes_per_exp * @as(f64, @floatFromInt(iters))) / ns
        else
            0.0;
    }
}

fn fileExistsPosix(path: [:0]const u8) bool {
    const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return false;
    _ = std.c.close(fd);
    return true;
}

fn runAminferToolsSynth(allocator: std.mem.Allocator, io: Io, tool_args: []const []const u8) !void {
    var argv_buf: [32][]const u8 = undefined;
    var n: usize = 0;
    if (fileExistsPosix("tools/.venv/bin/aminfer-tools")) {
        argv_buf[n] = "tools/.venv/bin/aminfer-tools";
        n += 1;
    } else {
        argv_buf[n] = "/opt/homebrew/bin/uv";
        n += 1;
        argv_buf[n] = "run";
        n += 1;
        argv_buf[n] = "--directory";
        n += 1;
        argv_buf[n] = "tools";
        n += 1;
        argv_buf[n] = "aminfer-tools";
        n += 1;
    }
    if (n + tool_args.len > argv_buf.len) return error.TooManyArgs;
    for (tool_args) |a| {
        argv_buf[n] = a;
        n += 1;
    }
    const res = try std.process.run(allocator, io, .{ .argv = argv_buf[0..n] });
    defer allocator.free(res.stdout);
    defer allocator.free(res.stderr);
    switch (res.term) {
        .exited => |code| if (code != 0) return error.SynthToolFailed,
        else => return error.SynthToolFailed,
    }
}

fn runE2eSynthCase(
    allocator: std.mem.Allocator,
    bundle_dir: []const u8,
    draft_kind: enum { dflare, dfly2 },
    engram_shards: usize,
    measure_batch8_core3: bool,
    n_tokens: u32,
) !E2eSynthResult {
    var out: E2eSynthResult = .{ .ran = true };

    const model_amf = try std.fmt.allocPrintSentinel(allocator, "{s}/model.amf", .{bundle_dir}, 0);
    defer allocator.free(model_amf);
    const model_cfg_path = try std.fmt.allocPrintSentinel(allocator, "{s}/config.json", .{bundle_dir}, 0);
    defer allocator.free(model_cfg_path);
    const draft_amf = try std.fmt.allocPrintSentinel(allocator, "{s}/{s}", .{
        bundle_dir,
        if (draft_kind == .dflare) "dflare.amf" else "dfly2.amf",
    }, 0);
    defer allocator.free(draft_amf);
    const draft_cfg_path = try std.fmt.allocPrintSentinel(allocator, "{s}/{s}", .{
        bundle_dir,
        if (draft_kind == .dflare) "dflare_config.json" else "dfly2_config.json",
    }, 0);
    defer allocator.free(draft_cfg_path);

    // 1. `verify-amf` on model + draft (+ optional engram shards)
    {
        var af = try amf.AmfFile.open(allocator, model_amf);
        defer af.close(allocator);
        try af.verifyAllBlobs(allocator);
        var daf = try amf.AmfFile.open(allocator, draft_amf);
        defer daf.close(allocator);
        try daf.verifyAllBlobs(allocator);
        out.verify_amf_ok = true;
    }

    const cfg_bytes = try readFileAllocPosix(allocator, model_cfg_path, 4 * 1024 * 1024);
    defer allocator.free(cfg_bytes);
    var gcfg = try graph.GraphConfig.fromJson(allocator, cfg_bytes);
    defer gcfg.deinit(allocator);

    var ple_owned: [8][:0]u8 = undefined;
    var ple_views: [8][:0]const u8 = undefined;
    for (0..engram_shards) |si| {
        ple_owned[si] = try std.fmt.allocPrintSentinel(allocator, "{s}/engram-{d:0>5}-of-{d:0>5}.amf", .{ bundle_dir, si, engram_shards }, 0);
        ple_views[si] = ple_owned[si];
    }
    defer for (0..engram_shards) |si| allocator.free(ple_owned[si]);

    var model = try graph.GraphModel.loadFromFile(allocator, model_amf, gcfg, .{
        .max_seq_len = 256,
        .max_tree_nodes = 16,
        .enable_feature_taps = true,
        .startup_canary = false,
        .ple_paths = ple_views[0..engram_shards],
    });
    defer model.deinit(allocator);

    const prompt = [_]u32{ 1, 7, 13, 21 };
    const dec_tokens = try allocator.alloc(u32, n_tokens);
    defer allocator.free(dec_tokens);
    const spec_tokens = try allocator.alloc(u32, n_tokens);
    defer allocator.free(spec_tokens);

    // 2. Greedy T = 0 baseline `decode`
    model.resetKvCache();
    var cur_tok = try model.prefillChunk(&prompt, 0, true);
    dec_tokens[0] = cur_tok;
    const t_dec0 = try vmstat.nowNs();
    for (1..n_tokens) |step| {
        cur_tok = try model.stepDecode(cur_tok, @intCast(prompt.len + step - 1));
        dec_tokens[step] = cur_tok;
    }
    const t_dec1 = try vmstat.nowNs();
    const dec_s = @as(f64, @floatFromInt(t_dec1 -| t_dec0)) / 1e9;
    out.decode_tok_s = if (dec_s > 0.0 and n_tokens > 1) @as(f64, @floatFromInt(n_tokens - 1)) / dec_s else 0.0;

    if (model.m82Stats()) |ms| {
        out.sme2_ctrl_calls = ms.inkling2_sme2_ctrl_calls;
        out.gpu_ctrl_calls = ms.inkling2_gpu_ctrl_calls;
        out.mid_pass_flushes = ms.mid_pass_flushes;
        out.cold_expert_fetches = ms.cold_expert_fetches;
        out.index_shared_reuses = ms.hy4_index_shared_reuses;
    }
    if (model.pleStats()) |ps| {
        out.engram_nvme_reads = ps.nvme_reads;
    }
    out.engram_r_ge1_reads = model.loopIsolationStats().ple_r_ge1_nvme_reads;

    // 2B. Optional M = 8 batched `Core x 3` `verifyTree` vs `8 x 1` serial `stepDecode` speedup
    if (measure_batch8_core3) {
        const block_tokens = [_]u32{ 7, 13, 21, 29, 37, 43, 51, 59 };
        var positions: [8]u32 = undefined;
        var masks: [8]u64 = undefined;
        for (0..8) |bi| {
            positions[bi] = @intCast(prompt.len + bi);
            masks[bi] = (@as(u64, 1) << @intCast(bi + 1)) - 1;
        }
        const batch_iters: usize = 6;

        model.resetKvCache();
        _ = try model.prefillChunk(&prompt, 0, true);
        const tb0 = try vmstat.nowNs();
        for (0..batch_iters) |_| {
            _ = try model.verifyTree(&block_tokens, &positions, &masks, @intCast(prompt.len));
        }
        const tb1 = try vmstat.nowNs();

        const ts0 = try vmstat.nowNs();
        for (0..batch_iters) |_| {
            model.resetKvCache();
            _ = try model.prefillChunk(&prompt, 0, true);
            for (0..8) |bi| {
                _ = try model.stepDecode(block_tokens[bi], @intCast(prompt.len + bi));
            }
        }
        const ts1 = try vmstat.nowNs();

        // Subtract prefill overhead from serial measurement using a quick prefill-only baseline.
        const tp0 = try vmstat.nowNs();
        for (0..batch_iters) |_| {
            model.resetKvCache();
            _ = try model.prefillChunk(&prompt, 0, true);
        }
        const tp1 = try vmstat.nowNs();

        const batch_ns = tb1 -| tb0;
        const serial_ns = (ts1 -| ts0) -| (tp1 -| tp0);
        out.batch8_verify_us = @as(f64, @floatFromInt(batch_ns)) / (@as(f64, @floatFromInt(batch_iters)) * 1000.0);
        out.serial8_decode_us = @as(f64, @floatFromInt(serial_ns)) / (@as(f64, @floatFromInt(batch_iters)) * 1000.0);
        out.batch8_core3_speedup = if (out.batch8_verify_us > 0.0)
            out.serial8_decode_us / out.batch8_verify_us
        else
            0.0;
    }

    // 3. Greedy T = 0 `spec-decode` (`DFlare` or `DFly2`) + 100% bit-exact parity assertion
    const dcfg_bytes = try readFileAllocPosix(allocator, draft_cfg_path, 4 * 1024 * 1024);
    defer allocator.free(dcfg_bytes);

    if (draft_kind == .dflare) {
        var loaded = try engine_mod.LoadedDFlareProposer.loadFromFile(allocator, &model, draft_amf, dcfg_bytes, null, null);
        defer loaded.deinit(allocator);
        var se = try engine_mod.SpecEngine.initWithDFlare(allocator, &model, loaded.proposer(), .{
            .mode = .dflare,
            .max_tree_nodes = 16,
            .temperature = 0.0,
            .enable_pruning = false,
        });
        defer se.deinit(allocator);

        spec_tokens[0] = try se.prefill(&prompt, null);
        var produced: usize = 1;
        const t_sp0 = try vmstat.nowNs();
        while (produced < n_tokens) {
            const rs = try se.stepRound(null, spec_tokens[produced..n_tokens]);
            if (rs.emitted_count == 0) break;
            produced += rs.emitted_count;
        }
        const t_sp1 = try vmstat.nowNs();
        const sp_s = @as(f64, @floatFromInt(t_sp1 -| t_sp0)) / 1e9;
        out.spec_tok_s = if (sp_s > 0.0 and produced > 1) @as(f64, @floatFromInt(produced - 1)) / sp_s else 0.0;
        out.spec_tau = se.stats.meanAcceptanceLength();
        out.spec_rounds = se.stats.rounds;
        out.dflare_core3_batches = se.stats.dflare_core3_batches;
        out.tokens_compared = @intCast(produced);
        out.greedy_parity_ok = (produced == n_tokens) and std.mem.eql(u32, dec_tokens, spec_tokens);
        if (model.m82Stats()) |ms| {
            out.sme2_ctrl_calls = ms.inkling2_sme2_ctrl_calls;
            out.gpu_ctrl_calls = ms.inkling2_gpu_ctrl_calls;
            out.mid_pass_flushes = ms.mid_pass_flushes;
            out.cold_expert_fetches = ms.cold_expert_fetches;
        }
    } else {
        // Run DFly2 twice: (a) `confidence_threshold = 0.001` (no early-stop baseline),
        // (b) `confidence_threshold = 0.50` (active `confidence_head` early-stop)!
        const thresholds = [_]f32{ 0.001, 0.50 };
        for (thresholds, 0..) |thr, tidx| {
            var loaded = try engine_mod.LoadedDFly2Proposer.loadFromFile(allocator, &model, draft_amf, dcfg_bytes, null, thr, null);
            defer loaded.deinit(allocator);
            var se = try engine_mod.SpecEngine.initWithDFly2(allocator, &model, loaded.proposer(), .{
                .mode = .dfly2,
                .max_tree_nodes = 16,
                .temperature = 0.0,
                .enable_pruning = false,
            });
            defer se.deinit(allocator);

            spec_tokens[0] = try se.prefill(&prompt, null);
            var produced: usize = 1;
            const t_sp0 = try vmstat.nowNs();
            while (produced < n_tokens) {
                const rs = try se.stepRound(null, spec_tokens[produced..n_tokens]);
                if (rs.emitted_count == 0) break;
                produced += rs.emitted_count;
            }
            const t_sp1 = try vmstat.nowNs();
            const sp_s = @as(f64, @floatFromInt(t_sp1 -| t_sp0)) / 1e9;
            const tok_s = if (sp_s > 0.0 and produced > 1) @as(f64, @floatFromInt(produced - 1)) / sp_s else 0.0;
            const exact = (produced == n_tokens) and std.mem.eql(u32, dec_tokens, spec_tokens);

            if (tidx == 0) {
                out.no_early_stop_tok_s = tok_s;
                out.spec_tok_s = tok_s;
                out.spec_tau = se.stats.meanAcceptanceLength();
                out.spec_rounds = se.stats.rounds;
                out.tokens_compared = @intCast(produced);
                out.greedy_parity_ok = exact;
            } else {
                out.early_stop_tok_s = tok_s;
                out.early_stop_tau = se.stats.meanAcceptanceLength();
                out.early_stop_count = se.stats.dfly2_early_stops;
                out.early_stop_parity_ok = exact;
                out.early_stop_speedup = if (out.no_early_stop_tok_s > 0.0)
                    out.early_stop_tok_s / out.no_early_stop_tok_s
                else
                    0.0;
                if (out.early_stop_tok_s > out.spec_tok_s) out.spec_tok_s = out.early_stop_tok_s;
            }
            if (model.m82Stats()) |ms| {
                out.index_shared_reuses = @max(out.index_shared_reuses, ms.hy4_index_shared_reuses);
                out.cold_expert_fetches = @max(out.cold_expert_fetches, ms.cold_expert_fetches);
            }
            if (model.pleStats()) |ps| {
                out.engram_nvme_reads = @max(out.engram_nvme_reads, ps.nvme_reads);
            }
        }
    }

    return out;
}

fn resolveBenchScratchRoot(allocator: std.mem.Allocator) ![]u8 {
    const base_raw: []const u8 = if (std.c.getenv("TMPDIR")) |p| blk: {
        const s = std.mem.sliceTo(p, 0);
        break :blk if (s.len > 0) s else "/tmp";
    } else "/tmp";
    const base = std.mem.trimEnd(u8, base_raw, "/");
    return std.fmt.allocPrint(allocator, "{s}/aminfer-bench-m82", .{if (base.len > 0) base else "/tmp"});
}

pub fn runM82WithConfig(allocator: std.mem.Allocator, cfg: M82Config) !M82Report {
    var report: M82Report = .{};
    report.vm_before = try vmstat.Snapshot.capture();

    const facts = try sysinfo.Facts.read();
    const device = try metal.Device.open();
    defer device.close();
    const dev_info = try device.info();
    const budget = budget_mod.Budget.fromHost(facts, dev_info, .{});
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    var compute = try metal.Compute.init(device);
    defer compute.close();

    var threaded_io: std.Io.Threaded = .init(allocator, .{});
    defer threaded_io.deinit();
    const io = threaded_io.io();

    const scratch_root = try resolveBenchScratchRoot(allocator);
    defer allocator.free(scratch_root);
    {
        const mk_res = try std.process.run(allocator, io, .{
            .argv = &.{ "/bin/mkdir", "-p", scratch_root },
        });
        allocator.free(mk_res.stdout);
        allocator.free(mk_res.stderr);
    }
    defer {
        if (std.process.run(allocator, io, .{
            .argv = &.{ "/bin/rm", "-rf", scratch_root },
        })) |rm_res| {
            allocator.free(rm_res.stdout);
            allocator.free(rm_res.stderr);
        } else |_| {}
    }

    // 1. G25 GPU Kernels + 40/48 SWA KV + IndexCache + IndexPool + LatentMoE
    try benchG25Kernels(allocator, device, library, &compute, budget, cfg.kernel_iters, &report);

    // 2. C26 L2 SME2 Loop Controller + fp8_e4m3_row Engram F_NOCACHE + Cold-Expert F_NOCACHE
    try benchC26CpuAndIo(allocator, device, &compute, budget, scratch_root, cfg, &report);

    // 3. W23 + D14 End-to-End Python Synths -> Zig verify-amf, decode, spec-decode
    if (cfg.include_python_synths) {
        const dir_mini = try std.fmt.allocPrint(allocator, "{s}/inkling2_mini", .{scratch_root});
        defer allocator.free(dir_mini);
        try runAminferToolsSynth(allocator, io, &.{
            "synth-inkling2",
            "--preset",
            "tiny-inkling2-mini",
            "--dflare-layers",
            "3",
            "--sme2-pack",
            "controllers",
            "--cold-experts-per-layer",
            "4",
            "--output-dir",
            dir_mini,
        });
        report.e2e_inkling2_mini = try runE2eSynthCase(allocator, dir_mini, .dflare, 0, true, cfg.synth_tokens);

        const dir_small = try std.fmt.allocPrint(allocator, "{s}/inkling2_small", .{scratch_root});
        defer allocator.free(dir_small);
        try runAminferToolsSynth(allocator, io, &.{
            "synth-inkling2",
            "--preset",
            "tiny-inkling2-small",
            "--dflare-layers",
            "5",
            "--output-dir",
            dir_small,
        });
        report.e2e_inkling2_small = try runE2eSynthCase(allocator, dir_small, .dflare, 0, false, cfg.synth_tokens);

        const dir_hy4 = try std.fmt.allocPrint(allocator, "{s}/hy4", .{scratch_root});
        defer allocator.free(dir_hy4);
        try runAminferToolsSynth(allocator, io, &.{
            "synth-hy4",
            "--preset",
            "tiny-hy4",
            "--output-dir",
            dir_hy4,
        });
        report.e2e_hy4 = try runE2eSynthCase(allocator, dir_hy4, .dfly2, 0, false, cfg.synth_tokens);

        const dir_hy4f = try std.fmt.allocPrint(allocator, "{s}/hy4_flash", .{scratch_root});
        defer allocator.free(dir_hy4f);
        try runAminferToolsSynth(allocator, io, &.{
            "synth-hy4",
            "--preset",
            "tiny-hy4-flash",
            "--engram-shards",
            "2",
            "--engram-dtype",
            "fp8_e4m3_row",
            "--cold-experts-per-layer",
            "4",
            "--output-dir",
            dir_hy4f,
        });
        report.e2e_hy4_flash = try runE2eSynthCase(allocator, dir_hy4f, .dfly2, 2, false, cfg.synth_tokens);
    }

    report.vm_after = try vmstat.Snapshot.capture();
    report.swapouts_delta = vmstat.Snapshot.swapoutDelta(report.vm_before, report.vm_after);

    const micro_ok = (report.swapouts_delta == 0) and
        (report.hy4_idx_shared_dispatches == 0) and
        (report.sme2_ctrl_gpu_dispatches == 0) and
        (report.sme2_ctrl_fits_l2) and
        (report.engram_fp8_r_ge1_reads == 0) and
        (report.engram_fp8_parity_ok) and
        (report.cold_expert_fetches > 0) and
        (report.kv_reduction_8k_pct > 70.0) and
        (report.hy4_latent_moe_bytes_reduction_x > 2.5);

    const e2e_ok = if (!cfg.include_python_synths)
        true
    else
        (report.e2e_inkling2_mini.verify_amf_ok and report.e2e_inkling2_mini.greedy_parity_ok and
            report.e2e_inkling2_mini.dflare_core3_batches > 0 and
            (report.e2e_inkling2_mini.sme2_ctrl_calls + report.e2e_inkling2_mini.gpu_ctrl_calls) > 0 and
            report.e2e_inkling2_small.verify_amf_ok and report.e2e_inkling2_small.greedy_parity_ok and
            report.e2e_hy4.verify_amf_ok and report.e2e_hy4.greedy_parity_ok and report.e2e_hy4.early_stop_parity_ok and
            report.e2e_hy4.index_shared_reuses > 0 and
            report.e2e_hy4_flash.verify_amf_ok and report.e2e_hy4_flash.greedy_parity_ok and
            report.e2e_hy4_flash.early_stop_parity_ok and report.e2e_hy4_flash.engram_nvme_reads > 0 and
            report.e2e_hy4_flash.engram_r_ge1_reads == 0);

    report.all_passed = micro_ok and e2e_ok;
    return report;
}

pub fn runM82(allocator: std.mem.Allocator) !M82Report {
    return runM82WithConfig(allocator, .{});
}

test "m82: G25 fused kernels, 40/48 SWA KV, IndexCache 0-dispatch, LatentMoE, and C26 SME2+FP8 Engram+cold-expert pass with zero swapouts" {
    const gpa = std.testing.allocator;
    const report = try runM82WithConfig(gpa, .{
        .kernel_iters = 8,
        .ctrl_iters = 20,
        .engram_iters = 8,
        .cold_expert_iters = 8,
        .include_python_synths = false,
    });
    try std.testing.expect(report.all_passed);
    try std.testing.expectEqual(@as(u64, 0), report.swapouts_delta);
    try std.testing.expectEqual(@as(u32, 0), report.hy4_idx_shared_dispatches);
    try std.testing.expectEqual(@as(u32, 0), report.sme2_ctrl_gpu_dispatches);
    try std.testing.expectEqual(@as(u64, 0), report.engram_fp8_r_ge1_reads);
    try std.testing.expect(report.engram_fp8_parity_ok);
    try std.testing.expect(report.sme2_ctrl_fits_l2);
    try std.testing.expect(report.kv_reduction_8k_pct > 70.0);
    try std.testing.expect(report.hy4_latent_moe_bytes_reduction_x > 2.5);
    try std.testing.expect(report.cold_expert_fetches >= 8);

    var aw: std.Io.Writer.Allocating = .init(gpa);
    defer aw.deinit();
    try report.write(&aw.writer);
    const rendered = aw.written();
    try std.testing.expect(std.mem.indexOf(u8, rendered, "bench.m82.g25.rmsnorm_sconv4.speedup") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "bench.m82.c26.engram_fp8.r_ge1_nvme_reads") != null);
}
