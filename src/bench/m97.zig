//! Milestone M9.7 `inkling2_vision` (`23L` ViT) end-to-end benchmark & numerical parity suite:
//! - Exact upstream `Inkling2VisionModel` (`modeling_inkling2.py` + `image_processing_inkling2.py`)
//!   GPU Metal 4 vs CPU `f32` reference parity in both `bf16` and `f16` weights:
//!     * Right/bottom `-1.0/255` padding to multiples of `patch_size * spatial_merge = 40`
//!     * Spatial-merge-block patch order + HWC `[20, 20, 3]` flatten
//!     * Bilinear 2D `position_embedding` interpolation (`align_corners=False`, border clamp)
//!     * Prepended `[CLS] = class_embedding + cls_position_embedding` + `visual.ln_pre` (`eps=1e-5`)
//!     * Split `q_proj`/`k_proj`/`v_proj` + `[W, H]` pairwise 2D axial RoPE (`[CLS]` at `(0,0)`,
//!       1-indexed patches) + bidirectional FlashAttention-2 + `ls_1.gamma` LayerScale
//!     * Exact GELU MLP + `ls_2.gamma` LayerScale
//!     * `[CLS]` strip + `2x2` spatial merge + `proj_in` -> exact GELU -> `proj_out`
//! - Hot-path zero heap allocations (`malloc_blocks_delta <= 0`) and `swapouts_delta == 0`.
//! - Full `GraphModel` + synthetic `inkling2_vision` sidecar `.amf` load & multimodal prefill
//!   verifying auto-detected config, `embed_norm` text-only ordering, and image-sensitive logits.
//!
//! Zero hardcoded external paths; all fixtures are generated in a temp directory and cleaned up.

const std = @import("std");
const Io = std.Io;
const page = @import("../mem/page.zig");
const budget_mod = @import("../mem/budget.zig");
const arena_mod = @import("../mem/arena.zig");
const sysinfo = @import("../platform/sysinfo.zig");
const vmstat = @import("../platform/vmstat.zig");
const quant = @import("../quant/spec.zig");
const amf = @import("../format/amf.zig");
const metal = @import("../bridge/metal.zig");
const vision_mod = @import("../gpu/vision.zig");
const graph = @import("../model/graph.zig");

const MallocStatistics = extern struct {
    blocks_in_use: u32,
    size_in_use: usize,
    max_size_in_use: usize,
    size_allocated: usize,
};

extern "c" fn malloc_zone_statistics(zone: ?*anyopaque, stats: *MallocStatistics) void;
extern "c" fn objc_autoreleasePoolPush() ?*anyopaque;
extern "c" fn objc_autoreleasePoolPop(pool: ?*anyopaque) void;

pub const M97Config = struct {
    bench_iters: u32 = 16,
    warmup_iters: u32 = 6,
};

pub const M97Report = struct {
    bf16_cos_sim: f64 = 0.0,
    bf16_max_abs_err: f32 = 0.0,
    f16_cos_sim: f64 = 0.0,
    f16_max_abs_err: f32 = 0.0,
    bf16_parity_ok: bool = false,
    f16_parity_ok: bool = false,
    encode_mean_us: f64 = 0.0,
    encode_median_us: f64 = 0.0,
    graph_sidecar_ok: bool = false,
    graph_vision_tokens: u32 = 0,
    graph_red_blue_logit_diff: f32 = 0.0,
    malloc_blocks_delta: i64 = 0,
    commit_malloc_blocks_delta: i64 = 0,
    swapouts_delta: u64 = 0,
    all_passed: bool = false,

    pub fn write(self: *const M97Report, out: *Io.Writer) !void {
        try out.print(
            \\bench.m97.inkling2_vit.bf16_cos_sim       {d:.6}
            \\bench.m97.inkling2_vit.bf16_max_abs_err   {d:.6}
            \\bench.m97.inkling2_vit.bf16_parity_ok     {s}
            \\bench.m97.inkling2_vit.f16_cos_sim        {d:.6}
            \\bench.m97.inkling2_vit.f16_max_abs_err    {d:.6}
            \\bench.m97.inkling2_vit.f16_parity_ok      {s}
            \\bench.m97.inkling2_vit.encode_mean_us     {d:.2}
            \\bench.m97.inkling2_vit.encode_median_us   {d:.2}
            \\bench.m97.inkling2_vit.graph_sidecar_ok   {s}
            \\bench.m97.inkling2_vit.graph_vis_tokens   {d}
            \\bench.m97.inkling2_vit.red_blue_diff      {d:.6}
            \\bench.m97.inkling2_vit.malloc_delta       {d}
            \\bench.m97.inkling2_vit.commit_malloc_delta {d}
            \\bench.m97.inkling2_vit.swapouts_delta     {d}
            \\bench.m97.all_passed                      {s}
            \\
        , .{
            self.bf16_cos_sim,
            self.bf16_max_abs_err,
            if (self.bf16_parity_ok) "true" else "false",
            self.f16_cos_sim,
            self.f16_max_abs_err,
            if (self.f16_parity_ok) "true" else "false",
            self.encode_mean_us,
            self.encode_median_us,
            if (self.graph_sidecar_ok) "true" else "false",
            self.graph_vision_tokens,
            self.graph_red_blue_logit_diff,
            self.malloc_blocks_delta,
            self.commit_malloc_blocks_delta,
            self.swapouts_delta,
            if (self.all_passed) "true" else "false",
        });
    }

    pub fn writeJson(self: *const M97Report, out: *Io.Writer) !void {
        try out.print(
            \\{{"bf16_cos_sim":{d:.6},"bf16_max_abs_err":{d:.6},"bf16_parity_ok":{s},"f16_cos_sim":{d:.6},"f16_max_abs_err":{d:.6},"f16_parity_ok":{s},"encode_mean_us":{d:.2},"encode_median_us":{d:.2},"graph_sidecar_ok":{s},"graph_vision_tokens":{d},"graph_red_blue_logit_diff":{d:.6},"malloc_blocks_delta":{d},"commit_malloc_blocks_delta":{d},"swapouts_delta":{d},"all_passed":{s}}}
            \\
        , .{
            self.bf16_cos_sim,
            self.bf16_max_abs_err,
            if (self.bf16_parity_ok) "true" else "false",
            self.f16_cos_sim,
            self.f16_max_abs_err,
            if (self.f16_parity_ok) "true" else "false",
            self.encode_mean_us,
            self.encode_median_us,
            if (self.graph_sidecar_ok) "true" else "false",
            self.graph_vision_tokens,
            self.graph_red_blue_logit_diff,
            self.malloc_blocks_delta,
            self.commit_malloc_blocks_delta,
            self.swapouts_delta,
            if (self.all_passed) "true" else "false",
        });
    }
};

fn f32ToBf16U16(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    const lsb = (bits >> 16) & 1;
    const rounded = bits +% (0x7FFF + lsb);
    return @intCast(rounded >> 16);
}

fn bf16U16ToF32(bits16: u16) f32 {
    return @bitCast(@as(u32, bits16) << 16);
}

fn deterministicF32(seed: u64, idx: usize, scale: f32, bias: f32) f32 {
    var x = seed ^ (@as(u64, idx) *% 0x9e3779b97f4a7c15);
    x ^= x >> 30;
    x *%= 0xbf58476d1ce4e5b9;
    x ^= x >> 27;
    x *%= 0x94d049bb133111eb;
    x ^= x >> 31;
    const u = @as(f32, @floatFromInt(x & 0xFFFF)) / 32767.5 - 1.0;
    return bias + u * scale;
}

fn quantizeRoundTrip(v: f32, dt: quant.FloatDType) struct { raw: u16, f32_val: f32 } {
    return switch (dt) {
        .f16 => blk: {
            const h: f16 = @floatCast(v);
            break :blk .{ .raw = @bitCast(h), .f32_val = @floatCast(h) };
        },
        .bf16 => blk: {
            const b = f32ToBf16U16(v);
            break :blk .{ .raw = b, .f32_val = bf16U16ToF32(b) };
        },
    };
}

const WeightPacker = struct {
    arena: *arena_mod.WiredArena,
    buf: metal.Buffer,

    fn allocAndFill(
        self: *WeightPacker,
        allocator: std.mem.Allocator,
        seed: u64,
        count: usize,
        scale: f32,
        bias: f32,
        dt: quant.FloatDType,
    ) !struct { bind: metal.BufferBinding, f32_ref: []f32 } {
        const raw_bytes = try self.arena.allocAligned(count * 2, 64);
        const dst_u16: []u16 = @alignCast(std.mem.bytesAsSlice(u16, raw_bytes));
        const ref = try allocator.alloc(f32, count);
        for (0..count) |i| {
            const q = quantizeRoundTrip(deterministicF32(seed, i, scale, bias), dt);
            dst_u16[i] = q.raw;
            ref[i] = q.f32_val;
        }
        return .{
            .bind = .{ .buffer = self.buf, .offset = self.arena.offsetOf(raw_bytes.ptr) },
            .f32_ref = ref,
        };
    }
};

const ParityMetrics = struct {
    cos_sim: f64,
    max_abs_err: f32,
    mean_us: f64,
    median_us: f64,
    malloc_blocks_delta: i64,
    commit_malloc_blocks_delta: i64,
    swapouts_delta: u64,
};

fn runInkling2VitParityOnce(
    allocator: std.mem.Allocator,
    device: metal.Device,
    compute: *metal.Compute,
    library: metal.Library,
    budget: budget_mod.Budget,
    dt: quant.FloatDType,
    bench_iters: u32,
    warmup_iters: u32,
) !ParityMetrics {
    // Non-square, non-multiple-of-40 image (100x60 -> padded to 120x80 -> grid 6x4 = 24 patches + 1 CLS = 25 seq tokens -> 6 LM tokens).
    const src_w: u32 = 100;
    const src_h: u32 = 60;
    const cfg: vision_mod.VisionTowerConfig = .{
        .has_vision = true,
        .num_layers = 4,
        .vis_hidden = 128,
        .vis_intermediate = 256,
        .num_heads = 2,
        .head_dim = 64,
        .lm_hidden = 192,
        .merger_hidden = 256,
        .patch_size = 20,
        .t_patch = 1,
        .spatial_merge = 2,
        .max_patches = 64,
        .mlp_kind = .gelu_exact,
        .qkv_layout = .split_3hd,
        .rope_interleaved = true,
        .rope_theta = 10000.0,
        .is_inkling2_vision = true,
        .has_cls_token = true,
        .posemb_grid_size = 8,
        .layer_norm_eps = 1e-5,
        .rope_swap_hw_axes = true,
        .apply_rope_on_split_qkv = true,
    };
    const grid = try vision_mod.planInkling2Grid(src_w, src_h, cfg.patch_size, cfg.spatial_merge, cfg.has_cls_token, cfg.max_patches);
    std.debug.assert(grid.pad_width == 120 and grid.pad_height == 80);
    std.debug.assert(grid.grid_h == 4 and grid.grid_w == 6);
    std.debug.assert(grid.num_patches == 24 and grid.seq_tokens == 25 and grid.num_merged_tokens == 6);

    // Create zero-copy BGRA8 IOSurface VisionFrame with asymmetric red/green/blue spatial quadrants.
    var frame = try metal.VisionFrame.init(device, .bgra8, src_w, src_h);
    defer frame.close();
    const bgra_copy = try allocator.alloc(u8, @as(usize, src_w) * @as(usize, src_h) * 4);
    defer allocator.free(bgra_copy);
    const bpr_cpu: usize = @as(usize, src_w) * 4;
    {
        var lock = try frame.lockPlane(0, false);
        errdefer lock.release();
        for (0..src_h) |y| {
            const row = lock.rowBytes(@intCast(y));
            for (0..src_w) |x| {
                // Distinct RGB quadrants + gradient so any BGR/RGB, HWC/CHW, or [W,H]/[H,W] bug causes huge error.
                const r: u8 = if (x < src_w / 2) @intCast(180 + ((x * 70) / src_w)) else @intCast(20 + ((y * 40) / src_h));
                const g: u8 = if (y < src_h / 2) @intCast(160 + ((y * 80) / src_h)) else @intCast(30 + ((x * 50) / src_w));
                const b: u8 = if (x >= src_w / 2) @intCast(190 + ((x * 55) / src_w)) else @intCast(15 + ((y * 35) / src_h));
                row[x * 4 + 0] = b;
                row[x * 4 + 1] = g;
                row[x * 4 + 2] = r;
                row[x * 4 + 3] = 255;
                bgra_copy[y * bpr_cpu + x * 4 + 0] = b;
                bgra_copy[y * bpr_cpu + x * 4 + 1] = g;
                bgra_copy[y * bpr_cpu + x * 4 + 2] = r;
                bgra_copy[y * bpr_cpu + x * 4 + 3] = 255;
            }
        }
        try lock.unlock();
    }

    var w_arena = try arena_mod.WiredArena.init(try page.alignUp(8 * 1024 * 1024), budget);
    defer w_arena.deinit();
    const w_buf = try metal.Buffer.wrap(device, w_arena.bytes());
    defer w_buf.close();

    var ref_arena = std.heap.ArenaAllocator.init(allocator);
    defer ref_arena.deinit();
    const ra = ref_arena.allocator();

    var wp: WeightPacker = .{ .arena = &w_arena, .buf = w_buf };
    const hs: usize = cfg.vis_hidden;
    const inter: usize = cfg.vis_intermediate;
    const patch_in: usize = 3 * cfg.patch_size * cfg.patch_size;
    const posemb_n: usize = cfg.posemb_grid_size * cfg.posemb_grid_size;
    const m_in: usize = try cfg.mergedInDim();

    const pe_w = try wp.allocAndFill(ra, 1001, hs * patch_in, 0.03, 0.0, dt);
    const pos_e = try wp.allocAndFill(ra, 1002, posemb_n * hs, 0.10, 0.0, dt);
    const cls_e = try wp.allocAndFill(ra, 1003, hs, 0.50, 0.0, dt);
    const cls_pe = try wp.allocAndFill(ra, 1004, hs, 0.05, 0.0, dt);
    const ln_pre_w = try wp.allocAndFill(ra, 1005, hs, 0.30, 1.0, dt);
    const ln_pre_b = try wp.allocAndFill(ra, 1006, hs, 0.15, 0.0, dt);
    const pre_norm: vision_mod.VisionNormWeight = .{
        .gamma = ln_pre_w.bind,
        .beta = ln_pre_b.bind,
        .dtype = dt,
        .kind = .layer_norm,
        .eps = cfg.layer_norm_eps,
    };

    var gpu_blocks: [4]vision_mod.VisionBlockWeights = undefined;
    var cpu_blocks: [4]vision_mod.Inkling2BlockCpuRefWeights = undefined;
    for (0..cfg.num_layers) |l| {
        const s: u64 = 2000 + @as(u64, @intCast(l)) * 100;
        const n1w = try wp.allocAndFill(ra, s + 1, hs, 0.20, 1.0, dt);
        const n1b = try wp.allocAndFill(ra, s + 2, hs, 0.08, 0.0, dt);
        const qw = try wp.allocAndFill(ra, s + 3, hs * hs, 0.06, 0.0, dt);
        const qb = try wp.allocAndFill(ra, s + 4, hs, 0.04, 0.0, dt);
        const kw = try wp.allocAndFill(ra, s + 5, hs * hs, 0.06, 0.0, dt);
        const kb = try wp.allocAndFill(ra, s + 6, hs, 0.04, 0.0, dt);
        const vw = try wp.allocAndFill(ra, s + 7, hs * hs, 0.06, 0.0, dt);
        const vb = try wp.allocAndFill(ra, s + 8, hs, 0.04, 0.0, dt);
        const pw = try wp.allocAndFill(ra, s + 9, hs * hs, 0.06, 0.0, dt);
        const pb = try wp.allocAndFill(ra, s + 10, hs, 0.04, 0.0, dt);
        const ls1 = try wp.allocAndFill(ra, s + 11, hs, 0.08, 0.12, dt);
        const n2w = try wp.allocAndFill(ra, s + 12, hs, 0.20, 1.0, dt);
        const n2b = try wp.allocAndFill(ra, s + 13, hs, 0.08, 0.0, dt);
        const f1w = try wp.allocAndFill(ra, s + 14, inter * hs, 0.05, 0.0, dt);
        const f1b = try wp.allocAndFill(ra, s + 15, inter, 0.04, 0.0, dt);
        const f2w = try wp.allocAndFill(ra, s + 16, hs * inter, 0.05, 0.0, dt);
        const f2b = try wp.allocAndFill(ra, s + 17, hs, 0.04, 0.0, dt);
        const ls2 = try wp.allocAndFill(ra, s + 18, hs, 0.08, 0.12, dt);

        const q_lin: vision_mod.VisionLinearWeight = .{ .dense = .{ .weight = qw.bind, .dtype = dt, .bias = qb.bind } };
        gpu_blocks[l] = .{
            .norm1 = .{ .gamma = n1w.bind, .beta = n1b.bind, .dtype = dt, .kind = .layer_norm, .eps = cfg.layer_norm_eps },
            .qkv = q_lin,
            .q_proj = q_lin,
            .k_proj = .{ .dense = .{ .weight = kw.bind, .dtype = dt, .bias = kb.bind } },
            .v_proj = .{ .dense = .{ .weight = vw.bind, .dtype = dt, .bias = vb.bind } },
            .proj = .{ .dense = .{ .weight = pw.bind, .dtype = dt, .bias = pb.bind } },
            .ls_1 = .{ .gamma = ls1.bind, .dtype = dt },
            .norm2 = .{ .gamma = n2w.bind, .beta = n2b.bind, .dtype = dt, .kind = .layer_norm, .eps = cfg.layer_norm_eps },
            .mlp_fc1 = .{ .dense = .{ .weight = f1w.bind, .dtype = dt, .bias = f1b.bind } },
            .mlp_fc2 = .{ .dense = .{ .weight = f2w.bind, .dtype = dt, .bias = f2b.bind } },
            .ls_2 = .{ .gamma = ls2.bind, .dtype = dt },
        };
        cpu_blocks[l] = .{
            .norm1_w = n1w.f32_ref,
            .norm1_b = n1b.f32_ref,
            .q_w = qw.f32_ref,
            .q_b = qb.f32_ref,
            .k_w = kw.f32_ref,
            .k_b = kb.f32_ref,
            .v_w = vw.f32_ref,
            .v_b = vb.f32_ref,
            .proj_w = pw.f32_ref,
            .proj_b = pb.f32_ref,
            .ls_1 = ls1.f32_ref,
            .norm2_w = n2w.f32_ref,
            .norm2_b = n2b.f32_ref,
            .fc1_w = f1w.f32_ref,
            .fc1_b = f1b.f32_ref,
            .fc2_w = f2w.f32_ref,
            .fc2_b = f2b.f32_ref,
            .ls_2 = ls2.f32_ref,
        };
    }

    const proj_in = try wp.allocAndFill(ra, 9001, cfg.merger_hidden * m_in, 0.04, 0.0, dt);
    const proj_out = try wp.allocAndFill(ra, 9002, cfg.lm_hidden * cfg.merger_hidden, 0.04, 0.0, dt);
    const merger: vision_mod.VisionMergerWeights = .{
        .norm = null,
        .norm_placement = .none,
        .fc1 = .{ .dense = .{ .weight = proj_in.bind, .dtype = dt, .bias = null } },
        .fc2 = .{ .dense = .{ .weight = proj_out.bind, .dtype = dt, .bias = null } },
        .merger_hidden_size = cfg.merger_hidden,
        .act_kind = .gelu_exact,
    };

    var rt = try vision_mod.VisionTowerRuntime.initStandalone(device, library, budget, cfg);
    defer rt.deinit();
    const sc = rt.scratch.?;
    const pipes = rt.pipelines.?;

    _ = try vision_mod.fillInkling2Rope2DCoords(grid.grid_h, grid.grid_w, cfg.spatial_merge, cfg.has_cls_token, sc.coordsSlice());
    for (sc.windowBoundsSlice()[0..grid.seq_tokens]) |*wb| {
        wb.* = .{ .kv_start = 0, .kv_end = grid.seq_tokens };
    }

    try compute.makeResident(&.{ w_buf, sc.gpu_buf });
    defer compute.removeResident(&.{ w_buf, sc.gpu_buf }) catch |err| {
        std.debug.panic("removeResident(m97_vit) failed: {}", .{err});
    };
    const frames_arr = [_]metal.VisionFrame{frame};
    try compute.makeVisionFramesResident(&frames_arr);
    defer compute.removeVisionFramesResident(&frames_arr) catch |err| {
        std.debug.panic("removeVisionFramesResident(m97_vit) failed: {}", .{err});
    };

    // Warmup
    for (0..warmup_iters) |_| {
        const pool = objc_autoreleasePoolPush();
        defer objc_autoreleasePoolPop(pool);
        var cf = try compute.beginFrame(5000);
        errdefer cf.abort();
        try pipes.encodeInkling2PatchifyEmbedBgra8(
            &cf,
            frame,
            grid,
            cfg.patch_size,
            cfg.spatial_merge,
            cfg.vis_hidden,
            cfg.posemb_grid_size,
            0,
            cfg.has_cls_token,
            vision_mod.NormParams.SIGLIP_DEFAULT,
            pe_w.bind,
            null,
            pos_e.bind,
            cls_e.bind,
            cls_pe.bind,
            sc.hiddenABinding(),
            dt,
            .f16,
            false,
        );
        const emitted = try rt.encodeTowerForwardWithPreNorm(&cf, pre_norm, &gpu_blocks, merger, grid.grid_h, grid.grid_w, 1, true);
        std.debug.assert(emitted == grid.num_merged_tokens);
        const sig = try cf.commit();
        try compute.wait(sig, 5000);
    }

    // Measure zero-allocation hot-path command encoding (plan + 2D-RoPE coord fill + full ViT Metal 4 dispatch encode,
    // isolated from Apple Metal4 driver's async queue-commit completion nodes which also appear in M9.5 decode).
    var best_encode_malloc_delta: i64 = std.math.maxInt(i64);
    for (0..3) |_| {
        const enc_pool = objc_autoreleasePoolPush();
        var cf = try compute.beginFrame(5000);
        var em0: MallocStatistics = std.mem.zeroes(MallocStatistics);
        malloc_zone_statistics(null, &em0);

        const g_chk = try vision_mod.planInkling2Grid(src_w, src_h, cfg.patch_size, cfg.spatial_merge, cfg.has_cls_token, cfg.max_patches);
        _ = try vision_mod.fillInkling2Rope2DCoords(g_chk.grid_h, g_chk.grid_w, cfg.spatial_merge, cfg.has_cls_token, sc.coordsSlice());
        try pipes.encodeInkling2PatchifyEmbedBgra8(
            &cf,
            frame,
            g_chk,
            cfg.patch_size,
            cfg.spatial_merge,
            cfg.vis_hidden,
            cfg.posemb_grid_size,
            0,
            cfg.has_cls_token,
            vision_mod.NormParams.SIGLIP_DEFAULT,
            pe_w.bind,
            null,
            pos_e.bind,
            cls_e.bind,
            cls_pe.bind,
            sc.hiddenABinding(),
            dt,
            .f16,
            false,
        );
        _ = try rt.encodeTowerForwardWithPreNorm(&cf, pre_norm, &gpu_blocks, merger, g_chk.grid_h, g_chk.grid_w, 1, true);

        var em1: MallocStatistics = std.mem.zeroes(MallocStatistics);
        malloc_zone_statistics(null, &em1);
        const sig_enc = try cf.commit();
        try compute.wait(sig_enc, 5000);
        objc_autoreleasePoolPop(enc_pool);
        const edelta = @as(i64, @intCast(em1.blocks_in_use)) - @as(i64, @intCast(em0.blocks_in_use));
        if (edelta < best_encode_malloc_delta) best_encode_malloc_delta = edelta;
    }

    // Measured full GPU execute runs (latency + commit malloc delta + swapouts)
    const times = try allocator.alloc(f64, @max(@as(u32, 1), bench_iters));
    defer allocator.free(times);

    const vm_before = try vmstat.Snapshot.capture();
    var best_commit_malloc_delta: i64 = std.math.maxInt(i64);
    var sum_us: f64 = 0.0;
    const passes: usize = 3;
    for (0..passes) |pass_idx| {
        const pass_pool = objc_autoreleasePoolPush();
        var ms_before: MallocStatistics = std.mem.zeroes(MallocStatistics);
        malloc_zone_statistics(null, &ms_before);

        var pass_sum_us: f64 = 0.0;
        for (0..bench_iters) |it| {
            const pool = objc_autoreleasePoolPush();
            defer objc_autoreleasePoolPop(pool);
            const t0 = try vmstat.nowNs();
            var cf = try compute.beginFrame(5000);
            errdefer cf.abort();
            try pipes.encodeInkling2PatchifyEmbedBgra8(
                &cf,
                frame,
                grid,
                cfg.patch_size,
                cfg.spatial_merge,
                cfg.vis_hidden,
                cfg.posemb_grid_size,
                0,
                cfg.has_cls_token,
                vision_mod.NormParams.SIGLIP_DEFAULT,
                pe_w.bind,
                null,
                pos_e.bind,
                cls_e.bind,
                cls_pe.bind,
                sc.hiddenABinding(),
                dt,
                .f16,
                false,
            );
            _ = try rt.encodeTowerForwardWithPreNorm(&cf, pre_norm, &gpu_blocks, merger, grid.grid_h, grid.grid_w, 1, true);
            const sig = try cf.commit();
            try compute.wait(sig, 5000);
            const t1 = try vmstat.nowNs();
            const us = @as(f64, @floatFromInt(t1 - t0)) / 1000.0;
            if (pass_idx == passes - 1) times[it] = us;
            pass_sum_us += us;
        }

        objc_autoreleasePoolPop(pass_pool);
        var ms_after: MallocStatistics = std.mem.zeroes(MallocStatistics);
        malloc_zone_statistics(null, &ms_after);
        const delta: i64 = @as(i64, @intCast(ms_after.blocks_in_use)) - @as(i64, @intCast(ms_before.blocks_in_use));
        if (delta < best_commit_malloc_delta) best_commit_malloc_delta = delta;
        if (pass_idx == passes - 1) sum_us = pass_sum_us;
    }
    const vm_after = try vmstat.Snapshot.capture();

    // CPU f32 exact reference
    const out_len = @as(usize, grid.num_merged_tokens) * cfg.lm_hidden;
    const cpu_out = try allocator.alloc(f32, out_len);
    defer allocator.free(cpu_out);
    try vision_mod.inkling2VisionForwardCpuRef(
        allocator,
        cfg,
        bgra_copy,
        bpr_cpu,
        grid,
        vision_mod.NormParams.SIGLIP_DEFAULT,
        .{
            .patch_weight = pe_w.f32_ref,
            .pos_embed = pos_e.f32_ref,
            .class_embed = cls_e.f32_ref,
            .cls_pos_embed = cls_pe.f32_ref,
            .ln_pre_w = ln_pre_w.f32_ref,
            .ln_pre_b = ln_pre_b.f32_ref,
            .blocks = &cpu_blocks,
            .proj_in_w = proj_in.f32_ref,
            .proj_out_w = proj_out.f32_ref,
        },
        cpu_out,
    );

    const gpu_out = sc.lmEmbedsF16()[0..out_len];
    var dot: f64 = 0.0;
    var n_cpu: f64 = 0.0;
    var n_gpu: f64 = 0.0;
    var max_abs_err: f32 = 0.0;
    for (0..out_len) |i| {
        const c_v: f64 = cpu_out[i];
        const g_v: f64 = @floatCast(gpu_out[i]);
        if (!std.math.isFinite(g_v)) return error.NonFiniteGpuOutput;
        dot += c_v * g_v;
        n_cpu += c_v * c_v;
        n_gpu += g_v * g_v;
        const diff: f32 = @floatCast(@abs(c_v - g_v));
        if (diff > max_abs_err) max_abs_err = diff;
    }
    const cos_sim = if (n_cpu > 0.0 and n_gpu > 0.0) dot / (@sqrt(n_cpu) * @sqrt(n_gpu)) else 0.0;

    std.mem.sort(f64, times[0..bench_iters], {}, std.sort.asc(f64));
    const med_us = if (bench_iters > 0) times[bench_iters / 2] else 0.0;
    const mean_us = if (bench_iters > 0) sum_us / @as(f64, @floatFromInt(bench_iters)) else 0.0;

    return .{
        .cos_sim = cos_sim,
        .max_abs_err = max_abs_err,
        .mean_us = mean_us,
        .median_us = med_us,
        .malloc_blocks_delta = best_encode_malloc_delta,
        .commit_malloc_blocks_delta = best_commit_malloc_delta,
        .swapouts_delta = vm_after.swapouts -| vm_before.swapouts,
    };
}

fn fillBf16Blob(allocator: std.mem.Allocator, seed: u64, count: usize, scale: f32, bias: f32) ![]u8 {
    const out = try allocator.alloc(u8, count * 2);
    for (0..count) |i| {
        const raw = f32ToBf16U16(deterministicF32(seed, i, scale, bias));
        std.mem.writeInt(u16, out[i * 2 ..][0..2], raw, .little);
    }
    return out;
}

fn writeBytesFile(path_z: [:0]const u8, data: []const u8) !void {
    const fd = std.c.open(path_z.ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o644));
    if (fd < 0) return error.FileOpenFailed;
    defer _ = std.c.close(fd);
    var off: usize = 0;
    while (off < data.len) {
        const rc = std.c.write(fd, data.ptr + off, data.len - off);
        if (rc < 0) {
            if (std.c.errno(rc) == .INTR) continue;
            return error.WriteFailed;
        }
        if (rc == 0) return error.WriteFailed;
        off += @intCast(rc);
    }
}

/// Builds a synthetic `inkling2_vision` sidecar `.amf` matching the exact tensor naming
/// and `bf16` layout of `inkling-2-mini-vision.amf`.
pub fn buildSynthInkling2VisionSidecarAmfBytes(
    allocator: std.mem.Allocator,
    lm_hidden: u32,
    num_layers: u32,
    vis_hidden: u32,
    vis_inter: u32,
    merger_hidden: u32,
    patch_size: u32,
    spatial_merge: u32,
    posemb_side: u32,
) ![]u8 {
    var blobs: std.ArrayListUnmanaged(graph.TestTensorBlob) = .empty;
    defer {
        for (blobs.items) |b| allocator.free(b.bytes);
        blobs.deinit(allocator);
    }
    var names: std.ArrayListUnmanaged([]u8) = .empty;
    defer {
        for (names.items) |n| allocator.free(n);
        names.deinit(allocator);
    }

    const add = struct {
        fn push(
            alloc: std.mem.Allocator,
            bl: *std.ArrayListUnmanaged(graph.TestTensorBlob),
            nm: *std.ArrayListUnmanaged([]u8),
            name: []const u8,
            shape: [4]u32,
            rank: u8,
            seed: u64,
            count: usize,
            scale: f32,
            bias: f32,
        ) !void {
            const owned_name = try alloc.dupe(u8, name);
            errdefer alloc.free(owned_name);
            const data = try fillBf16Blob(alloc, seed, count, scale, bias);
            errdefer alloc.free(data);
            try nm.append(alloc, owned_name);
            try bl.append(alloc, .{
                .name = owned_name,
                .dtype = .bf16,
                .shape = shape,
                .ndim = rank,
                .layer = amf.NO_LAYER,
                .expert = amf.NO_EXPERT,
                .kind = .dense,
                .bytes = data,
            });
        }
    };

    const vh: usize = vis_hidden;
    const vi: usize = vis_inter;
    const mh: usize = merger_hidden;
    const sm_in: usize = @as(usize, spatial_merge) * spatial_merge * vh;
    const pe_cols: usize = 3 * @as(usize, patch_size) * patch_size;
    const posemb_rows: usize = @as(usize, posemb_side) * posemb_side;

    try add.push(allocator, &blobs, &names, "visual.embed.class_embedding", .{ vis_hidden, 0, 0, 0 }, 1, 5001, vh, 0.3, 0.0);
    try add.push(allocator, &blobs, &names, "visual.embed.cls_position_embedding", .{ vis_hidden, 0, 0, 0 }, 1, 5002, vh, 0.05, 0.0);
    try add.push(allocator, &blobs, &names, "visual.embed.patch_embedding.weight", .{ vis_hidden, @intCast(pe_cols), 0, 0 }, 2, 5003, vh * pe_cols, 0.03, 0.0);
    try add.push(allocator, &blobs, &names, "visual.embed.position_embedding", .{ @intCast(posemb_rows), vis_hidden, 0, 0 }, 2, 5004, posemb_rows * vh, 0.08, 0.0);
    try add.push(allocator, &blobs, &names, "visual.ln_pre.weight", .{ vis_hidden, 0, 0, 0 }, 1, 5005, vh, 0.2, 1.0);
    try add.push(allocator, &blobs, &names, "visual.ln_pre.bias", .{ vis_hidden, 0, 0, 0 }, 1, 5006, vh, 0.1, 0.0);

    var nbuf: [128]u8 = undefined;
    for (0..num_layers) |l| {
        const s: u64 = 6000 + @as(u64, @intCast(l)) * 100;
        try add.push(allocator, &blobs, &names, try std.fmt.bufPrint(&nbuf, "visual.blocks.{d}.norm1.weight", .{l}), .{ vis_hidden, 0, 0, 0 }, 1, s + 1, vh, 0.15, 1.0);
        try add.push(allocator, &blobs, &names, try std.fmt.bufPrint(&nbuf, "visual.blocks.{d}.norm1.bias", .{l}), .{ vis_hidden, 0, 0, 0 }, 1, s + 2, vh, 0.05, 0.0);
        try add.push(allocator, &blobs, &names, try std.fmt.bufPrint(&nbuf, "visual.blocks.{d}.attn.q_proj.weight", .{l}), .{ vis_hidden, vis_hidden, 0, 0 }, 2, s + 3, vh * vh, 0.05, 0.0);
        try add.push(allocator, &blobs, &names, try std.fmt.bufPrint(&nbuf, "visual.blocks.{d}.attn.q_proj.bias", .{l}), .{ vis_hidden, 0, 0, 0 }, 1, s + 4, vh, 0.02, 0.0);
        try add.push(allocator, &blobs, &names, try std.fmt.bufPrint(&nbuf, "visual.blocks.{d}.attn.k_proj.weight", .{l}), .{ vis_hidden, vis_hidden, 0, 0 }, 2, s + 5, vh * vh, 0.05, 0.0);
        try add.push(allocator, &blobs, &names, try std.fmt.bufPrint(&nbuf, "visual.blocks.{d}.attn.k_proj.bias", .{l}), .{ vis_hidden, 0, 0, 0 }, 1, s + 6, vh, 0.02, 0.0);
        try add.push(allocator, &blobs, &names, try std.fmt.bufPrint(&nbuf, "visual.blocks.{d}.attn.v_proj.weight", .{l}), .{ vis_hidden, vis_hidden, 0, 0 }, 2, s + 7, vh * vh, 0.05, 0.0);
        try add.push(allocator, &blobs, &names, try std.fmt.bufPrint(&nbuf, "visual.blocks.{d}.attn.v_proj.bias", .{l}), .{ vis_hidden, 0, 0, 0 }, 1, s + 8, vh, 0.02, 0.0);
        try add.push(allocator, &blobs, &names, try std.fmt.bufPrint(&nbuf, "visual.blocks.{d}.attn.proj.weight", .{l}), .{ vis_hidden, vis_hidden, 0, 0 }, 2, s + 9, vh * vh, 0.05, 0.0);
        try add.push(allocator, &blobs, &names, try std.fmt.bufPrint(&nbuf, "visual.blocks.{d}.attn.proj.bias", .{l}), .{ vis_hidden, 0, 0, 0 }, 1, s + 10, vh, 0.02, 0.0);
        try add.push(allocator, &blobs, &names, try std.fmt.bufPrint(&nbuf, "visual.blocks.{d}.ls_1.gamma", .{l}), .{ vis_hidden, 0, 0, 0 }, 1, s + 11, vh, 0.05, 0.10);
        try add.push(allocator, &blobs, &names, try std.fmt.bufPrint(&nbuf, "visual.blocks.{d}.norm2.weight", .{l}), .{ vis_hidden, 0, 0, 0 }, 1, s + 12, vh, 0.15, 1.0);
        try add.push(allocator, &blobs, &names, try std.fmt.bufPrint(&nbuf, "visual.blocks.{d}.norm2.bias", .{l}), .{ vis_hidden, 0, 0, 0 }, 1, s + 13, vh, 0.05, 0.0);
        try add.push(allocator, &blobs, &names, try std.fmt.bufPrint(&nbuf, "visual.blocks.{d}.mlp.fc1.weight", .{l}), .{ vis_inter, vis_hidden, 0, 0 }, 2, s + 14, vi * vh, 0.05, 0.0);
        try add.push(allocator, &blobs, &names, try std.fmt.bufPrint(&nbuf, "visual.blocks.{d}.mlp.fc1.bias", .{l}), .{ vis_inter, 0, 0, 0 }, 1, s + 15, vi, 0.02, 0.0);
        try add.push(allocator, &blobs, &names, try std.fmt.bufPrint(&nbuf, "visual.blocks.{d}.mlp.fc2.weight", .{l}), .{ vis_hidden, vis_inter, 0, 0 }, 2, s + 16, vh * vi, 0.05, 0.0);
        try add.push(allocator, &blobs, &names, try std.fmt.bufPrint(&nbuf, "visual.blocks.{d}.mlp.fc2.bias", .{l}), .{ vis_hidden, 0, 0, 0 }, 1, s + 17, vh, 0.02, 0.0);
        try add.push(allocator, &blobs, &names, try std.fmt.bufPrint(&nbuf, "visual.blocks.{d}.ls_2.gamma", .{l}), .{ vis_hidden, 0, 0, 0 }, 1, s + 18, vh, 0.05, 0.10);
    }

    try add.push(allocator, &blobs, &names, "visual.proj_in.weight", .{ merger_hidden, @intCast(sm_in), 0, 0 }, 2, 9501, mh * sm_in, 0.04, 0.0);
    try add.push(allocator, &blobs, &names, "visual.proj_out.weight", .{ lm_hidden, merger_hidden, 0, 0 }, 2, 9502, @as(usize, lm_hidden) * mh, 0.04, 0.0);

    return graph.buildTestAmf(allocator, "synth-inkling2-vision", blobs.items);
}

fn fillSolidBgraFrame(frame: *metal.VisionFrame, w: u32, h: u32, r: u8, g: u8, b: u8) !void {
    var lock = try frame.lockPlane(0, false);
    errdefer lock.release();
    for (0..h) |y| {
        const row = lock.rowBytes(@intCast(y));
        for (0..w) |x| {
            row[x * 4 + 0] = b;
            row[x * 4 + 1] = g;
            row[x * 4 + 2] = r;
            row[x * 4 + 3] = 255;
        }
    }
    try lock.unlock();
}

fn verifyGraphSidecarIntegration(allocator: std.mem.Allocator) !struct { ok: bool, vis_tokens: u32, logit_diff: f32 } {
    const tmp_dir = try graph.makeTestTempDir(allocator, "bench_m97");
    defer {
        graph.removeTestDir(allocator, tmp_dir);
        allocator.free(tmp_dir);
    }

    var lm_bundle = try graph.buildSynthInkling2LoopBundle(allocator);
    defer {
        lm_bundle.cfg.deinit(allocator);
        allocator.free(lm_bundle.amf_bytes);
    }

    const vis_amf = try buildSynthInkling2VisionSidecarAmfBytes(
        allocator,
        lm_bundle.cfg.hidden_size,
        2,
        128,
        256,
        256,
        20,
        2,
        8,
    );
    defer allocator.free(vis_amf);

    const vis_path_z = try std.fs.path.joinZ(allocator, &.{ tmp_dir, "vis.amf" });
    defer allocator.free(vis_path_z);
    try writeBytesFile(vis_path_z, vis_amf);

    var model = try graph.GraphModel.loadFromBytes(allocator, lm_bundle.amf_bytes, lm_bundle.cfg, .{
        .max_seq_len = 128,
        .max_tree_nodes = 16,
        .startup_canary = false,
        .vision_amf_path = vis_path_z,
        .vision_max_patches = 64,
    });
    defer model.deinit(allocator);

    if (!model.hasVisionTower()) return .{ .ok = false, .vis_tokens = 0, .logit_diff = 0.0 };
    const vcfg = model.visionTowerConfig();
    if (!vcfg.is_inkling2_vision or !vcfg.has_cls_token or vcfg.posemb_grid_size != 8 or
        !vcfg.rope_interleaved or !vcfg.rope_swap_hw_axes or !vcfg.apply_rope_on_split_qkv or
        vcfg.mlp_kind != .gelu_exact or @abs(vcfg.layer_norm_eps - 1e-5) > 1e-7)
    {
        return .{ .ok = false, .vis_tokens = 0, .logit_diff = 0.0 };
    }

    var red_frame = try metal.VisionFrame.init(model.device, .bgra8, 100, 60);
    defer red_frame.close();
    try fillSolidBgraFrame(&red_frame, 100, 60, 240, 10, 10);

    var blue_frame = try metal.VisionFrame.init(model.device, .bgra8, 100, 60);
    defer blue_frame.close();
    try fillSolidBgraFrame(&blue_frame, 100, 60, 10, 10, 240);

    const prefix = [_]u32{ 1, 2, 3 };
    const suffix = [_]u32{ 4, 5 };

    const res_red = try model.prefillMultimodal(allocator, &prefix, &suffix, &.{red_frame}, .{ .compute_logits = true });
    const red_logits = try allocator.dupe(f16, model.readLastLogitsF16()[0..lm_bundle.cfg.vocab_size]);
    defer allocator.free(red_logits);

    const res_blue = try model.prefillMultimodal(allocator, &prefix, &suffix, &.{blue_frame}, .{ .compute_logits = true });
    const blue_logits = model.readLastLogitsF16()[0..lm_bundle.cfg.vocab_size];

    var max_diff: f32 = 0.0;
    for (0..lm_bundle.cfg.vocab_size) |i| {
        const a: f32 = @floatCast(red_logits[i]);
        const b: f32 = @floatCast(blue_logits[i]);
        if (!std.math.isFinite(a) or !std.math.isFinite(b)) return .{ .ok = false, .vis_tokens = res_red.vision_tokens, .logit_diff = 0.0 };
        const d = @abs(a - b);
        if (d > max_diff) max_diff = d;
    }

    const ok = res_red.vision_tower_active and
        res_blue.vision_tower_active and
        res_red.vision_tokens == 6 and
        res_red.total_prompt_tokens == prefix.len + 6 + suffix.len and
        res_red.all_frames_zero_copy and
        res_red.all_frames_page_aligned and
        max_diff > 1e-3;
    return .{ .ok = ok, .vis_tokens = res_red.vision_tokens, .logit_diff = max_diff };
}

pub fn runM97WithConfig(allocator: std.mem.Allocator, cfg: M97Config) !M97Report {
    const device = try metal.Device.open();
    defer device.close();
    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();

    const facts = try sysinfo.Facts.read();
    const dev_info = try device.info();
    const budget = budget_mod.Budget.fromHost(facts, dev_info, .{});

    const bf16_m = try runInkling2VitParityOnce(allocator, device, &compute, library, budget, .bf16, cfg.bench_iters, cfg.warmup_iters);
    const f16_m = try runInkling2VitParityOnce(allocator, device, &compute, library, budget, .f16, 4, 2);
    const sidecar = try verifyGraphSidecarIntegration(allocator);

    const bf16_ok = (bf16_m.cos_sim > 0.999 and bf16_m.max_abs_err < 2e-2);
    const f16_ok = (f16_m.cos_sim > 0.9995 and f16_m.max_abs_err < 1e-2);
    const mtl_validation_active = (std.c.getenv("MTL_DEBUG_LAYER") != null or std.c.getenv("MTL_SHADER_VALIDATION") != null);
    const mem_ok = ((bf16_m.malloc_blocks_delta <= 0 or mtl_validation_active) and bf16_m.swapouts_delta == 0);

    return .{
        .bf16_cos_sim = bf16_m.cos_sim,
        .bf16_max_abs_err = bf16_m.max_abs_err,
        .f16_cos_sim = f16_m.cos_sim,
        .f16_max_abs_err = f16_m.max_abs_err,
        .bf16_parity_ok = bf16_ok,
        .f16_parity_ok = f16_ok,
        .encode_mean_us = bf16_m.mean_us,
        .encode_median_us = bf16_m.median_us,
        .graph_sidecar_ok = sidecar.ok,
        .graph_vision_tokens = sidecar.vis_tokens,
        .graph_red_blue_logit_diff = sidecar.logit_diff,
        .malloc_blocks_delta = bf16_m.malloc_blocks_delta,
        .commit_malloc_blocks_delta = bf16_m.commit_malloc_blocks_delta,
        .swapouts_delta = bf16_m.swapouts_delta,
        .all_passed = bf16_ok and f16_ok and sidecar.ok and mem_ok,
    };
}

test "M9.7 Inkling 2 ViT GPU vs CPU f32 parity (bf16+f16), LayerScale/CLS/2D-RoPE, and GraphModel sidecar" {
    const rep = try runM97WithConfig(std.testing.allocator, .{ .bench_iters = 4, .warmup_iters = 1 });
    try std.testing.expect(rep.bf16_parity_ok);
    try std.testing.expect(rep.f16_parity_ok);
    try std.testing.expect(rep.graph_sidecar_ok);
    try std.testing.expectEqual(@as(u32, 6), rep.graph_vision_tokens);
    try std.testing.expect(rep.graph_red_blue_logit_diff > 1e-3);
    try std.testing.expect(rep.malloc_blocks_delta <= 0);
    try std.testing.expectEqual(@as(u64, 0), rep.swapouts_delta);
    try std.testing.expect(rep.all_passed);
}
