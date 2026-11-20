//! C9 kernel bandwidth calibration and Spike S2 Metal 4 steady-state hot-path
//! benchmark harness (ADR-0011, ADR-0012, docs/status.md C9).

const std = @import("std");
const quant = @import("../quant/spec.zig");
const amf = @import("../format/amf.zig");
const vmstat = @import("../platform/vmstat.zig");
const sysinfo = @import("../platform/sysinfo.zig");
const metal = @import("../bridge/metal.zig");
const page = @import("../mem/page.zig");
const arena_mod = @import("../mem/arena.zig");
const budget_mod = @import("../mem/budget.zig");
const dequant = @import("../gpu/dequant.zig");
const attention = @import("../gpu/attention.zig");
const moe = @import("../gpu/moe.zig");
const qwen3 = @import("../model/qwen3.zig");
const sam_mod = @import("../spec/sam.zig");
const tree_mod = @import("../spec/tree.zig");
const draft_mod = @import("../spec/draft.zig");
const engine_mod = @import("../spec/engine.zig");

/// Apple M4 unified memory bus peak bandwidth in decimal GB/s
/// (docs/architecture/hardware.md).
pub const M4_BUS_BANDWIDTH_GBS: f64 = 120.0;

/// Maximum uniform-bearing dispatches per Metal 4 frame (`AMI_UNIFORM_SLOTS_PER_FRAME`).
const MAX_DISPATCHES_PER_FRAME: u32 = 256;

const Q4 = quant.QuantOps(.{ .bits = .q4, .group_size = 64 });
const Q3 = quant.QuantOps(.{ .bits = .q3, .group_size = 64 });

pub const KernelKind = enum {
    gemv_q4,
    gemv_q3,
    gemm_q4,
    gemm_q3,

    pub fn bits(self: KernelKind) quant.QuantBits {
        return switch (self) {
            .gemv_q4, .gemm_q4 => .q4,
            .gemv_q3, .gemm_q3 => .q3,
        };
    }

    pub fn isGemv(self: KernelKind) bool {
        return switch (self) {
            .gemv_q4, .gemv_q3 => true,
            .gemm_q4, .gemm_q3 => false,
        };
    }

    pub fn label(self: KernelKind) []const u8 {
        return switch (self) {
            .gemv_q4 => "gemv_q4",
            .gemv_q3 => "gemv_q3",
            .gemm_q4 => "gemm_q4",
            .gemm_q3 => "gemm_q3",
        };
    }
};

pub const Case = struct {
    kind: KernelKind,
    m: u32,
    n: u32,
    k: u32,
};

pub const DEFAULT_CASES = [_]Case{
    // M=1 decode GEMV calibration (q4)
    .{ .kind = .gemv_q4, .m = 1, .n = 2048, .k = 2048 },
    .{ .kind = .gemv_q4, .m = 1, .n = 6144, .k = 2048 },
    .{ .kind = .gemv_q4, .m = 1, .n = 2048, .k = 768 },
    .{ .kind = .gemv_q4, .m = 1, .n = 4096, .k = 4096 },
    // M=1 decode GEMV calibration (q3)
    .{ .kind = .gemv_q3, .m = 1, .n = 2048, .k = 2048 },
    .{ .kind = .gemv_q3, .m = 1, .n = 6144, .k = 2048 },
    .{ .kind = .gemv_q3, .m = 1, .n = 2048, .k = 768 },
    .{ .kind = .gemv_q3, .m = 1, .n = 4096, .k = 4096 },
    // M=16 and M=64 speculative/prefill GEMM calibration (q4)
    .{ .kind = .gemm_q4, .m = 16, .n = 2048, .k = 2048 },
    .{ .kind = .gemm_q4, .m = 16, .n = 4096, .k = 4096 },
    .{ .kind = .gemm_q4, .m = 64, .n = 2048, .k = 2048 },
    .{ .kind = .gemm_q4, .m = 64, .n = 4096, .k = 4096 },
    // M=16 and M=64 speculative/prefill GEMM calibration (q3)
    .{ .kind = .gemm_q3, .m = 16, .n = 2048, .k = 2048 },
    .{ .kind = .gemm_q3, .m = 16, .n = 4096, .k = 4096 },
    .{ .kind = .gemm_q3, .m = 64, .n = 2048, .k = 2048 },
    .{ .kind = .gemm_q3, .m = 64, .n = 4096, .k = 4096 },
};

pub const Config = struct {
    cases: []const Case = &DEFAULT_CASES,
    /// Minimum weight working-set size in bytes so repeated dispatches stream
    /// past L2/SLC from unified DRAM. Set to 0 in small unit tests.
    min_working_set_bytes: usize = 64 * 1024 * 1024,
    /// Extra GPU spin-up batches before C9 case 0 so M4 DVFS reaches peak
    /// frequency before the first timed measurement.
    dvfs_warmup_batches: u32 = 16,
    warmup_batches: u32 = 4,
    gemv_timed_batches: u32 = 8,
    gemv_dispatches_per_batch: u32 = 32,
    gemm_timed_batches: u32 = 4,
    gemm_dispatches_per_batch: u32 = 16,
    /// Warms Metal 4's internal driver pool (~2000 entries, ~128 KiB) before
    /// measuring steady-state hot-path allocations in Spike S2.
    s2_warmup_rounds: u32 = 2048,
    s2_rounds: u32 = 1000,
    s2_dispatches_per_round: u32 = 32,
    timeout_ms: u64 = 30_000,
};

pub const CaseResult = struct {
    case: Case,
    bytes_per_dispatch: u64,
    dispatches: u64,
    weight_slots: usize,
    elapsed_ns: u64,

    pub fn latencyUs(self: CaseResult) f64 {
        if (self.dispatches == 0) return 0.0;
        return (@as(f64, @floatFromInt(self.elapsed_ns)) / @as(f64, @floatFromInt(self.dispatches))) / 1000.0;
    }

    /// Effective memory bandwidth `beta` in decimal GB/s (`bytes_moved / elapsed_ns`).
    pub fn effectiveBandwidthGbs(self: CaseResult) f64 {
        if (self.elapsed_ns == 0) return 0.0;
        const total_bytes = @as(f64, @floatFromInt(self.bytes_per_dispatch)) * @as(f64, @floatFromInt(self.dispatches));
        return total_bytes / @as(f64, @floatFromInt(self.elapsed_ns));
    }

    pub fn busUtilizationPct(self: CaseResult) f64 {
        return (self.effectiveBandwidthGbs() / M4_BUS_BANDWIDTH_GBS) * 100.0;
    }
};

pub const S2Result = struct {
    rounds: u32,
    dispatches_per_round: u32,
    encode_ns: u64,
    total_ns: u64,
    before: vmstat.Snapshot,
    after: vmstat.Snapshot,
    heap_delta: vmstat.HeapDelta,

    pub fn cpuEncodeUsPerRound(self: S2Result) f64 {
        if (self.rounds == 0) return 0.0;
        return (@as(f64, @floatFromInt(self.encode_ns)) / @as(f64, @floatFromInt(self.rounds))) / 1000.0;
    }

    pub fn cpuEncodeNsPerDispatch(self: S2Result) f64 {
        const total_dispatches: u64 = @as(u64, self.rounds) * self.dispatches_per_round;
        if (total_dispatches == 0) return 0.0;
        return @as(f64, @floatFromInt(self.encode_ns)) / @as(f64, @floatFromInt(total_dispatches));
    }

    pub fn fullRoundUs(self: S2Result) f64 {
        if (self.rounds == 0) return 0.0;
        return (@as(f64, @floatFromInt(self.total_ns)) / @as(f64, @floatFromInt(self.rounds))) / 1000.0;
    }
};

pub const Report = struct {
    gpu_name: [64]u8,
    gpu_name_len: usize,
    wired_budget_bytes: u64,
    arena_bytes: u64,
    results: []CaseResult,
    s2: S2Result,
    vm_before: vmstat.Snapshot,
    vm_after: vmstat.Snapshot,

    pub fn deinit(self: *Report, allocator: std.mem.Allocator) void {
        allocator.free(self.results);
        self.results = &.{};
    }

    pub fn gpuName(self: *const Report) []const u8 {
        return self.gpu_name[0..self.gpu_name_len];
    }

    pub fn write(self: *const Report, out: *std.Io.Writer) std.Io.Writer.Error!void {
        try out.print(
            \\bench.gpu                        {s}
            \\bench.wired_budget_mib           {d}
            \\bench.arena_mib                  {d}
            \\bench.bus_peak_gbs               {d:.1}
            \\
            \\C9 Kernel Bandwidth Calibration
            \\kernel    shape              bytes/disp   slots   latency(us)   beta(GB/s)   bus(%)
            \\
        , .{
            self.gpuName(),
            self.wired_budget_bytes >> 20,
            self.arena_bytes >> 20,
            M4_BUS_BANDWIDTH_GBS,
        });

        for (self.results) |res| {
            var shape_buf: [40]u8 = undefined;
            const shape = std.fmt.bufPrint(&shape_buf, "{d}x{d}x{d}", .{
                res.case.m,
                res.case.n,
                res.case.k,
            }) catch unreachable;
            try out.print(
                \\{s: <9} {s: <18} {d: >10}   {d: >5}   {d: >11.2}   {d: >10.2}   {d: >6.1}
                \\
            , .{
                res.case.kind.label(),
                shape,
                res.bytes_per_dispatch,
                res.weight_slots,
                res.latencyUs(),
                res.effectiveBandwidthGbs(),
                res.busUtilizationPct(),
            });
        }

        try out.print(
            \\
            \\Spike S2 Metal 4 Steady-State Hot Path
            \\s2.rounds                        {d}
            \\s2.dispatches_per_round          {d}
            \\s2.cpu_encode_us_per_round       {d:.2}
            \\s2.cpu_encode_ns_per_dispatch    {d:.1}
            \\s2.full_round_us                 {d:.2}
            \\s2.malloc_blocks_in_use_delta    {d}
            \\s2.malloc_size_in_use_delta      {d}
            \\s2.malloc_size_allocated_delta   {d}
            \\
            \\Swap Check
            \\swap.status                      ok
            \\swap.swapouts_delta              {d}
            \\swap.pageouts_delta              {d}
            \\swap.used_mib                    {d}
            \\
        , .{
            self.s2.rounds,
            self.s2.dispatches_per_round,
            self.s2.cpuEncodeUsPerRound(),
            self.s2.cpuEncodeNsPerDispatch(),
            self.s2.fullRoundUs(),
            self.s2.heap_delta.blocks_in_use,
            self.s2.heap_delta.size_in_use,
            self.s2.heap_delta.size_allocated,
            vmstat.Snapshot.swapoutDelta(self.vm_before, self.vm_after),
            vmstat.Snapshot.pageoutDelta(self.vm_before, self.vm_after),
            self.vm_after.swap_used_bytes >> 20,
        });
    }
};

pub const Error = error{
    InvalidConfig,
    NonFiniteOutput,
    OutOfMemory,
} || quant.ShapeError || vmstat.Error || vmstat.SwapError || sysinfo.Error || metal.WrapError ||
    arena_mod.WiredArena.InitError || arena_mod.WiredArena.AllocError;

const CaseLayout = struct {
    w_bytes: usize,
    sb_bytes: usize,
    x_bytes: usize,
    y_bytes: usize,
    bytes_per_dispatch: u64,
};

fn computeCaseLayout(case: Case) (quant.ShapeError || error{InvalidConfig})!CaseLayout {
    if (case.m == 0 or case.n == 0) return error.ZeroDimension;
    if (case.kind.isGemv() and case.m != 1) return error.InvalidConfig;

    const w_bytes = switch (case.kind.bits()) {
        .q4 => try Q4.packedWeightBytes(case.n, case.k),
        .q3 => try Q3.packedWeightBytes(case.n, case.k),
    };
    const sb_bytes = switch (case.kind.bits()) {
        .q4 => try Q4.scaleBiasBytes(case.n, case.k),
        .q3 => try Q3.scaleBiasBytes(case.n, case.k),
    };
    const x_elems = std.math.mul(usize, case.m, case.k) catch return error.Overflow;
    const x_bytes = std.math.mul(usize, x_elems, @sizeOf(f16)) catch return error.Overflow;
    const y_elems = std.math.mul(usize, case.m, case.n) catch return error.Overflow;
    const y_bytes = std.math.mul(usize, y_elems, @sizeOf(f16)) catch return error.Overflow;

    const w_sb = std.math.add(u64, w_bytes, sb_bytes) catch return error.Overflow;
    const xy = std.math.add(u64, x_bytes, y_bytes) catch return error.Overflow;
    const total = std.math.add(u64, w_sb, xy) catch return error.Overflow;

    return .{
        .w_bytes = w_bytes,
        .sb_bytes = sb_bytes,
        .x_bytes = x_bytes,
        .y_bytes = y_bytes,
        .bytes_per_dispatch = total,
    };
}

/// Computes the algorithmic minimum bytes transferred per dispatch
/// (`qweight + scale_bias + x + y`).
pub fn bytesMovedPerDispatch(case: Case) (quant.ShapeError || error{InvalidConfig})!u64 {
    return (try computeCaseLayout(case)).bytes_per_dispatch;
}

fn fillBenchBuffers(
    qweight_pool: []u8,
    sb_pool_raw: []u8,
    x_raw: []u8,
    y_raw: []u8,
) void {
    // Non-zero deterministic weight bitstream (prevents zero-page shortcuts).
    var state: u64 = 0x9E37_79B9_7F4A_7C15;
    const w_words: []u64 = @alignCast(std.mem.bytesAsSlice(u64, qweight_pool));
    for (w_words) |*word| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        word.* = state;
    }

    // Small finite affine parameters so dot products stay well inside fp16 range.
    const sb_table = [_]quant.ScaleBias{
        .{ .scale = @floatCast(0.015625), .bias = @floatCast(-0.125) },
        .{ .scale = @floatCast(0.0234375), .bias = @floatCast(-0.15625) },
        .{ .scale = @floatCast(0.01953125), .bias = @floatCast(-0.09375) },
        .{ .scale = @floatCast(0.01171875), .bias = @floatCast(-0.0625) },
    };
    const sb_slice: []quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, sb_pool_raw));
    for (sb_slice, 0..) |*sb, i| {
        sb.* = sb_table[i & 3];
    }

    const x_table = [_]f16{
        @floatCast(0.03125),
        @floatCast(-0.015625),
        @floatCast(0.046875),
        @floatCast(-0.03125),
        @floatCast(0.015625),
        @floatCast(-0.046875),
        @floatCast(0.0625),
        @floatCast(-0.0625),
    };
    const x_slice: []f16 = @alignCast(std.mem.bytesAsSlice(f16, x_raw));
    for (x_slice, 0..) |*v, i| {
        v.* = x_table[i & 7];
    }

    @memset(y_raw, 0);
}

fn encodeCase(
    pipes: dequant.DequantPipelines,
    frame: *metal.Compute.Frame,
    case: Case,
    gpu_buf: metal.Buffer,
    x_off: u64,
    w_off: u64,
    sb_off: u64,
    y_off: u64,
    insert_barrier_before: bool,
) dequant.EncodeError!void {
    if (case.kind.isGemv()) {
        try pipes.encodeGemv(frame, case.kind.bits(), .{
            .x = .{ .buffer = gpu_buf, .offset = x_off },
            .qweight = .{ .buffer = gpu_buf, .offset = w_off },
            .scale_bias = .{ .buffer = gpu_buf, .offset = sb_off },
            .y = .{ .buffer = gpu_buf, .offset = y_off },
            .n = case.n,
            .k = case.k,
        }, insert_barrier_before);
    } else {
        try pipes.encodeGemm(frame, case.kind.bits(), .{
            .x = .{ .buffer = gpu_buf, .offset = x_off },
            .qweight = .{ .buffer = gpu_buf, .offset = w_off },
            .scale_bias = .{ .buffer = gpu_buf, .offset = sb_off },
            .y = .{ .buffer = gpu_buf, .offset = y_off },
            .m = case.m,
            .n = case.n,
            .k = case.k,
        }, insert_barrier_before);
    }
}

/// Executes C9 kernel bandwidth calibration and Spike S2 steady-state hot-path
/// measurement over one `Budget`-checked, zero-copy `WiredArena`, enforcing
/// zero system swapouts over the entire run.
pub fn run(allocator: std.mem.Allocator, config: Config) Error!Report {
    if (config.gemv_timed_batches == 0 or
        config.gemv_dispatches_per_batch == 0 or
        config.gemv_dispatches_per_batch > MAX_DISPATCHES_PER_FRAME or
        config.gemm_timed_batches == 0 or
        config.gemm_dispatches_per_batch == 0 or
        config.gemm_dispatches_per_batch > MAX_DISPATCHES_PER_FRAME or
        config.s2_rounds == 0 or
        config.s2_dispatches_per_round == 0 or
        config.s2_dispatches_per_round > MAX_DISPATCHES_PER_FRAME)
    {
        return error.InvalidConfig;
    }

    // Minimum footprints required by Spike S2's (N=16, K=64) q4 GEMV dispatch.
    const s2_case: Case = .{ .kind = .gemv_q4, .m = 1, .n = 16, .k = 64 };
    const s2_layout = try computeCaseLayout(s2_case);

    var max_w_stride: usize = amf.align64(s2_layout.w_bytes);
    var max_sb_stride: usize = amf.align64(s2_layout.sb_bytes);
    var max_x_bytes: usize = amf.align64(s2_layout.x_bytes);
    var max_y_bytes: usize = amf.align64(s2_layout.y_bytes);

    for (config.cases) |case| {
        const layout = try computeCaseLayout(case);
        max_w_stride = @max(max_w_stride, amf.align64(layout.w_bytes));
        max_sb_stride = @max(max_sb_stride, amf.align64(layout.sb_bytes));
        max_x_bytes = @max(max_x_bytes, amf.align64(layout.x_bytes));
        max_y_bytes = @max(max_y_bytes, amf.align64(layout.y_bytes));
    }

    const qweight_pool_bytes = amf.align64(@max(max_w_stride, config.min_working_set_bytes));
    const sb_pool_bytes = amf.align64(@max(max_sb_stride, config.min_working_set_bytes / 4));
    const raw_arena_bytes = qweight_pool_bytes + sb_pool_bytes + max_x_bytes + max_y_bytes + 4 * page.APPLE_PAGE_SIZE;
    const arena_bytes = try page.alignUp(raw_arena_bytes);

    const facts = try sysinfo.Facts.read();
    const device = try metal.Device.open();
    defer device.close();
    const info = try device.info();
    const wired_budget = budget_mod.Budget.fromHost(facts, info, .{});

    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try dequant.DequantPipelines.init(device, library);
    defer pipes.close();

    const results = try allocator.alloc(CaseResult, config.cases.len);
    errdefer allocator.free(results);

    const vm_before = try vmstat.Snapshot.capture();

    var arena = try arena_mod.WiredArena.init(arena_bytes, wired_budget);
    defer arena.deinit();

    const qweight_pool = try arena.allocAligned(qweight_pool_bytes, 64);
    const sb_pool_raw = try arena.allocAligned(sb_pool_bytes, 64);
    const x_raw = try arena.allocAligned(max_x_bytes, 64);
    const y_raw = try arena.allocAligned(max_y_bytes, 64);
    fillBenchBuffers(qweight_pool, sb_pool_raw, x_raw, y_raw);

    const qweight_base_off: u64 = arena.offsetOf(qweight_pool.ptr);
    const sb_base_off: u64 = arena.offsetOf(sb_pool_raw.ptr);
    const x_off: u64 = arena.offsetOf(x_raw.ptr);
    const y_off: u64 = arena.offsetOf(y_raw.ptr);
    const y_f16: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, y_raw));

    const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
    defer gpu_buf.close();

    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();
    try compute.makeResident(&.{gpu_buf});

    // --- C9 Kernel Bandwidth Calibration ---
    for (config.cases, 0..) |case, case_idx| {
        const layout = try computeCaseLayout(case);
        const w_stride = amf.align64(layout.w_bytes);
        const sb_stride = amf.align64(layout.sb_bytes);
        const num_slots: usize = @max(
            @as(usize, 1),
            @min(qweight_pool_bytes / w_stride, sb_pool_bytes / sb_stride),
        );

        const timed_batches = if (case.kind.isGemv()) config.gemv_timed_batches else config.gemm_timed_batches;
        const dispatches_per_batch = if (case.kind.isGemv()) config.gemv_dispatches_per_batch else config.gemm_dispatches_per_batch;

        var slot_cursor: usize = 0;
        const warmups = config.warmup_batches + if (case_idx == 0) config.dvfs_warmup_batches else 0;
        var warmup_sig: u64 = 0;
        for (0..warmups) |_| {
            var frame = try compute.beginFrame(config.timeout_ms);
            errdefer frame.abort();
            for (0..dispatches_per_batch) |d| {
                const slot = slot_cursor % num_slots;
                slot_cursor += 1;
                try encodeCase(
                    pipes,
                    &frame,
                    case,
                    gpu_buf,
                    x_off,
                    qweight_base_off + @as(u64, slot) * w_stride,
                    sb_base_off + @as(u64, slot) * sb_stride,
                    y_off,
                    d > 0,
                );
            }
            warmup_sig = try frame.commit();
        }
        try compute.wait(warmup_sig, config.timeout_ms);

        var last_sig: u64 = 0;
        const t0 = try vmstat.nowNs();
        for (0..timed_batches) |_| {
            var frame = try compute.beginFrame(config.timeout_ms);
            errdefer frame.abort();
            for (0..dispatches_per_batch) |d| {
                const slot = slot_cursor % num_slots;
                slot_cursor += 1;
                try encodeCase(
                    pipes,
                    &frame,
                    case,
                    gpu_buf,
                    x_off,
                    qweight_base_off + @as(u64, slot) * w_stride,
                    sb_base_off + @as(u64, slot) * sb_stride,
                    y_off,
                    d > 0,
                );
            }
            last_sig = try frame.commit();
        }
        try compute.wait(last_sig, config.timeout_ms);
        const t1 = try vmstat.nowNs();

        const y_count: usize = @as(usize, case.m) * case.n;
        if (!std.math.isFinite(y_f16[0]) or !std.math.isFinite(y_f16[y_count - 1])) {
            return error.NonFiniteOutput;
        }

        results[case_idx] = .{
            .case = case,
            .bytes_per_dispatch = layout.bytes_per_dispatch,
            .dispatches = @as(u64, timed_batches) * dispatches_per_batch,
            .weight_slots = num_slots,
            .elapsed_ns = t1 - t0,
        };
    }

    // --- Spike S2: Steady-State Multi-Dispatch Metal 4 Hot Path ---
    var s2_warmup_sig: u64 = 0;
    for (0..config.s2_warmup_rounds) |_| {
        var frame = try compute.beginFrame(config.timeout_ms);
        errdefer frame.abort();
        for (0..config.s2_dispatches_per_round) |d| {
            try pipes.encodeGemv(&frame, .q4, .{
                .x = .{ .buffer = gpu_buf, .offset = x_off },
                .qweight = .{ .buffer = gpu_buf, .offset = qweight_base_off },
                .scale_bias = .{ .buffer = gpu_buf, .offset = sb_base_off },
                .y = .{ .buffer = gpu_buf, .offset = y_off },
                .n = s2_case.n,
                .k = s2_case.k,
            }, d > 0);
        }
        s2_warmup_sig = try frame.commit();
    }
    try compute.wait(s2_warmup_sig, config.timeout_ms);

    const s2_before = try vmstat.Snapshot.capture();
    var slot_sig: [2]u64 = .{ 0, 0 };
    var s2_last_sig: u64 = 0;
    var s2_encode_ns: u64 = 0;

    const s2_t0 = try vmstat.nowNs();
    for (0..config.s2_rounds) |round| {
        const slot = round & 1;
        // Drain only this ring slot's prior work outside the CPU encode timer
        // while keeping the sibling slot in flight on the GPU.
        if (slot_sig[slot] != 0) {
            try compute.wait(slot_sig[slot], config.timeout_ms);
        }
        const enc0 = try vmstat.nowNs();
        var frame = try compute.beginFrame(config.timeout_ms);
        errdefer frame.abort();
        for (0..config.s2_dispatches_per_round) |d| {
            try pipes.encodeGemv(&frame, .q4, .{
                .x = .{ .buffer = gpu_buf, .offset = x_off },
                .qweight = .{ .buffer = gpu_buf, .offset = qweight_base_off },
                .scale_bias = .{ .buffer = gpu_buf, .offset = sb_base_off },
                .y = .{ .buffer = gpu_buf, .offset = y_off },
                .n = s2_case.n,
                .k = s2_case.k,
            }, d > 0);
        }
        const sig = try frame.commit();
        const enc1 = try vmstat.nowNs();
        s2_encode_ns += enc1 - enc0;
        slot_sig[slot] = sig;
        s2_last_sig = sig;
    }
    try compute.wait(s2_last_sig, config.timeout_ms);
    const s2_t1 = try vmstat.nowNs();
    const s2_after = try vmstat.Snapshot.capture();

    const vm_after = try vmstat.Snapshot.capture();
    try vmstat.checkNoSwap(vm_before, vm_after);

    var gpu_name_buf: [64]u8 = @splat(0);
    const name_slice = info.name();
    const copy_len = @min(name_slice.len, gpu_name_buf.len);
    @memcpy(gpu_name_buf[0..copy_len], name_slice[0..copy_len]);

    return .{
        .gpu_name = gpu_name_buf,
        .gpu_name_len = copy_len,
        .wired_budget_bytes = wired_budget.limit_bytes,
        .arena_bytes = arena.capacity(),
        .results = results,
        .s2 = .{
            .rounds = config.s2_rounds,
            .dispatches_per_round = config.s2_dispatches_per_round,
            .encode_ns = s2_encode_ns,
            .total_ns = s2_t1 - s2_t0,
            .before = s2_before,
            .after = s2_after,
            .heap_delta = vmstat.heapDelta(s2_before, s2_after),
        },
        .vm_before = vm_before,
        .vm_after = vm_after,
    };
}

// ---------------------------------------------------------------------------
// G3 MoE Deduplicated Gather / SwiGLU Benchmark (M2)
// ---------------------------------------------------------------------------

pub const MoeBenchCase = struct {
    m: u32,
    bits: quant.QuantBits = .q3,
    hidden_size: u32 = 2048,
    intermediate_size: u32 = 768,
    num_experts: u32 = 128,
    top_k: u32 = 8,
    /// Distinct active experts shared across the `m` tokens in each dispatch.
    shared_pool: u32 = 8,
};

pub const DEFAULT_MOE_BENCH_CASES = [_]MoeBenchCase{
    // M=1 decode (8 unique experts / 8 token-expert slots)
    .{ .m = 1, .bits = .q3, .hidden_size = 2048, .intermediate_size = 768, .num_experts = 128, .top_k = 8, .shared_pool = 8 },
    // M=8 batched/speculative with high sharing (24 unique experts / 64 slots = 62.5% dedup savings)
    .{ .m = 8, .bits = .q3, .hidden_size = 2048, .intermediate_size = 768, .num_experts = 128, .top_k = 8, .shared_pool = 24 },
    // M=8 batched/speculative with moderate sharing (40 unique experts / 64 slots = 37.5% dedup savings)
    .{ .m = 8, .bits = .q3, .hidden_size = 2048, .intermediate_size = 768, .num_experts = 128, .top_k = 8, .shared_pool = 40 },
};

pub const MoeBenchConfig = struct {
    cases: []const MoeBenchCase = &DEFAULT_MOE_BENCH_CASES,
    dvfs_warmup_batches: u32 = 8,
    warmup_batches: u32 = 4,
    timed_batches: u32 = 8,
    dispatches_per_batch: u32 = 8,
    timeout_ms: u64 = 30_000,
};

pub const MoeBenchCaseResult = struct {
    case: MoeBenchCase,
    unique_experts: u32,
    naive_slots: u32,
    dedup_bytes: u64,
    naive_bytes: u64,
    dispatches: u64,
    elapsed_ns: u64,

    pub fn latencyUs(self: MoeBenchCaseResult) f64 {
        if (self.dispatches == 0) return 0.0;
        return (@as(f64, @floatFromInt(self.elapsed_ns)) / @as(f64, @floatFromInt(self.dispatches))) / 1000.0;
    }

    pub fn effectiveBandwidthGbs(self: MoeBenchCaseResult) f64 {
        if (self.elapsed_ns == 0) return 0.0;
        const total = @as(f64, @floatFromInt(self.dedup_bytes)) * @as(f64, @floatFromInt(self.dispatches));
        return total / @as(f64, @floatFromInt(self.elapsed_ns));
    }

    pub fn naiveEquivalentGbs(self: MoeBenchCaseResult) f64 {
        if (self.elapsed_ns == 0) return 0.0;
        const total = @as(f64, @floatFromInt(self.naive_bytes)) * @as(f64, @floatFromInt(self.dispatches));
        return total / @as(f64, @floatFromInt(self.elapsed_ns));
    }

    pub fn busUtilizationPct(self: MoeBenchCaseResult) f64 {
        return (self.effectiveBandwidthGbs() / M4_BUS_BANDWIDTH_GBS) * 100.0;
    }

    pub fn byteSavingsPct(self: MoeBenchCaseResult) f64 {
        if (self.naive_bytes == 0) return 0.0;
        return (1.0 - (@as(f64, @floatFromInt(self.dedup_bytes)) / @as(f64, @floatFromInt(self.naive_bytes)))) * 100.0;
    }
};

pub const MoeBenchReport = struct {
    results: []MoeBenchCaseResult,
    vm_before: vmstat.Snapshot,
    vm_after: vmstat.Snapshot,

    pub fn deinit(self: *MoeBenchReport, allocator: std.mem.Allocator) void {
        allocator.free(self.results);
        self.results = &.{};
    }

    pub fn write(self: *const MoeBenchReport, out: *std.Io.Writer) std.Io.Writer.Error!void {
        try out.print(
            \\G3 MoE Deduplicated Gather / SwiGLU FFN (3-bit Qwen3-30B-A3B Geometry)
            \\m    E    k   U/slots   dedup_bytes   naive_bytes   saved(%)   latency(us)   beta(GB/s)   bus(%)   naive_eq(GB/s)
            \\
        , .{});
        for (self.results) |res| {
            var u_buf: [24]u8 = undefined;
            const u_str = std.fmt.bufPrint(&u_buf, "{d}/{d}", .{ res.unique_experts, res.naive_slots }) catch unreachable;
            try out.print(
                \\{d: <4} {d: <4} {d: <3} {s: <9} {d: >11}   {d: >11}   {d: >8.1}   {d: >11.2}   {d: >10.2}   {d: >6.1}   {d: >14.2}
                \\
            , .{
                res.case.m,
                res.case.num_experts,
                res.case.top_k,
                u_str,
                res.dedup_bytes,
                res.naive_bytes,
                res.byteSavingsPct(),
                res.latencyUs(),
                res.effectiveBandwidthGbs(),
                res.busUtilizationPct(),
                res.naiveEquivalentGbs(),
            });
        }
        try out.print(
            \\g3.swapouts_delta                {d}
            \\g3.pageouts_delta                {d}
            \\
        , .{
            vmstat.Snapshot.swapoutDelta(self.vm_before, self.vm_after),
            vmstat.Snapshot.pageoutDelta(self.vm_before, self.vm_after),
        });
    }
};

pub const MoeBenchError = Error || moe.WorkspaceError || moe.EncodeError;

pub fn runMoeBench(allocator: std.mem.Allocator, config: MoeBenchConfig) MoeBenchError!MoeBenchReport {
    if (config.timed_batches == 0 or config.dispatches_per_batch == 0 or
        config.dispatches_per_batch * 5 > metal.UNIFORM_SLOTS_PER_FRAME)
    {
        return error.InvalidConfig;
    }

    const facts = try sysinfo.Facts.read();
    const device = try metal.Device.open();
    defer device.close();
    const dev_info = try device.info();
    const wired_budget = budget_mod.Budget.fromHost(facts, dev_info, .{});

    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try moe.MoePipelines.init(device, library);
    defer pipes.close();

    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();

    const results = try allocator.alloc(MoeBenchCaseResult, config.cases.len);
    errdefer allocator.free(results);

    const vm_before = try vmstat.Snapshot.capture();

    for (config.cases, 0..) |case, case_idx| {
        if (case.m == 0 or case.top_k == 0 or case.shared_pool < case.top_k or
            case.shared_pool > case.num_experts or case.num_experts > moe.MAX_EXPERTS)
        {
            return error.InvalidConfig;
        }
        const gu_w_bytes, const gu_sb_bytes, const dn_w_bytes, const dn_sb_bytes = switch (case.bits) {
            .q3 => .{
                try Q3.packedWeightBytes(case.intermediate_size, case.hidden_size),
                try Q3.scaleBiasBytes(case.intermediate_size, case.hidden_size),
                try Q3.packedWeightBytes(case.hidden_size, case.intermediate_size),
                try Q3.scaleBiasBytes(case.hidden_size, case.intermediate_size),
            },
            .q4 => .{
                try Q4.packedWeightBytes(case.intermediate_size, case.hidden_size),
                try Q4.scaleBiasBytes(case.intermediate_size, case.hidden_size),
                try Q4.packedWeightBytes(case.hidden_size, case.intermediate_size),
                try Q4.scaleBiasBytes(case.hidden_size, case.intermediate_size),
            },
        };

        const per_expert_bytes = amf.align64(gu_w_bytes) + amf.align64(gu_sb_bytes) +
            amf.align64(gu_w_bytes) + amf.align64(gu_sb_bytes) +
            amf.align64(dn_w_bytes) + amf.align64(dn_sb_bytes);
        const all_experts_bytes = @as(usize, case.num_experts) * per_expert_bytes;

        const num_banks: usize = @max(@as(usize, 1), @as(usize, case.num_experts) / case.shared_pool);
        const bank_logits_bytes = amf.align64(@as(usize, case.m) * case.num_experts * @sizeOf(f32));
        const all_logits_bytes = num_banks * bank_logits_bytes;
        const x_bytes = amf.align64(@as(usize, case.m) * case.hidden_size * @sizeOf(f16));
        const out_bytes = x_bytes;
        const tbl_bytes = amf.align64(@as(usize, case.num_experts) * @sizeOf(moe.ExpertWeights));
        const ws_bytes = try moe.MoeWorkspace.requiredBytes(.{
            .max_tokens = case.m,
            .num_experts = case.num_experts,
            .top_k = case.top_k,
            .hidden_size = case.hidden_size,
            .intermediate_size = case.intermediate_size,
        });

        const arena_cap = try page.alignUp(
            all_experts_bytes + all_logits_bytes + x_bytes + out_bytes + tbl_bytes + ws_bytes + 8 * page.APPLE_PAGE_SIZE,
        );
        var arena = try arena_mod.WiredArena.init(arena_cap, wired_budget);
        defer arena.deinit();

        const weights_raw = try arena.allocAligned(all_experts_bytes, 64);
        const logits_raw = try arena.allocAligned(all_logits_bytes, 64);
        const x_raw = try arena.allocAligned(x_bytes, 64);
        const out_raw = try arena.allocAligned(out_bytes, 64);
        const tbl_raw = try arena.allocAligned(tbl_bytes, 64);
        const ws = try moe.MoeWorkspace.init(&arena, .{
            .max_tokens = case.m,
            .num_experts = case.num_experts,
            .top_k = case.top_k,
            .hidden_size = case.hidden_size,
            .intermediate_size = case.intermediate_size,
        });

        // Fill expert weights with non-zero deterministic data & finite small scales.
        const exp_tbl: []moe.ExpertWeights = @alignCast(std.mem.bytesAsSlice(moe.ExpertWeights, tbl_raw));
        var rng: u64 = 0x517C_C1B7_2722_0A95 +% case_idx;
        const sb_val: quant.ScaleBias = .{ .scale = @floatCast(0.015625), .bias = @floatCast(-0.0625) };
        var cursor: usize = 0;
        for (0..case.num_experts) |e_idx| {
            const g_w = cursor;
            cursor += amf.align64(gu_w_bytes);
            const g_sb = cursor;
            cursor += amf.align64(gu_sb_bytes);
            const u_w = cursor;
            cursor += amf.align64(gu_w_bytes);
            const u_sb = cursor;
            cursor += amf.align64(gu_sb_bytes);
            const d_w = cursor;
            cursor += amf.align64(dn_w_bytes);
            const d_sb = cursor;
            cursor += amf.align64(dn_sb_bytes);

            for ([_]struct { off: usize, len: usize }{
                .{ .off = g_w, .len = gu_w_bytes },
                .{ .off = u_w, .len = gu_w_bytes },
                .{ .off = d_w, .len = dn_w_bytes },
            }) |seg| {
                const words: []u64 = @alignCast(std.mem.bytesAsSlice(u64, weights_raw[seg.off .. seg.off + (seg.len & ~@as(usize, 7))]));
                for (words) |*w| {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    w.* = rng;
                }
            }
            for ([_]struct { off: usize, len: usize }{
                .{ .off = g_sb, .len = gu_sb_bytes },
                .{ .off = u_sb, .len = gu_sb_bytes },
                .{ .off = d_sb, .len = dn_sb_bytes },
            }) |seg| {
                const sbs: []quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, weights_raw[seg.off .. seg.off + seg.len]));
                for (sbs) |*sb| sb.* = sb_val;
            }

            exp_tbl[e_idx] = .{
                .gate_qweight_offset = g_w,
                .gate_sb_offset = g_sb,
                .up_qweight_offset = u_w,
                .up_sb_offset = u_sb,
                .down_qweight_offset = d_w,
                .down_sb_offset = d_sb,
            };
        }

        const x_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, x_raw));
        for (x_f16, 0..) |*v, i| {
            const c: i32 = @as(i32, @intCast((i * 7 + 3) % 15)) - 7;
            v.* = @floatCast(@as(f32, @floatFromInt(c)) * 0.03125);
        }
        @memset(out_raw, 0);

        // Build `num_banks` rotating router-logit banks where each bank activates
        // exactly `case.shared_pool` distinct experts distributed across `case.m` tokens.
        for (0..num_banks) |bank| {
            const bank_slice = logits_raw[bank * bank_logits_bytes ..][0 .. @as(usize, case.m) * case.num_experts * @sizeOf(f32)];
            const logits_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, bank_slice));
            for (logits_f32) |*v| v.* = -8.0;

            const base_e: u32 = @intCast((bank * case.shared_pool) % case.num_experts);
            const stride: u32 = if (case.m > 1 and case.shared_pool > case.top_k)
                @max(@as(u32, 1), (case.shared_pool - case.top_k + case.m - 2) / (case.m - 1))
            else
                0;
            for (0..case.m) |tok| {
                const tok_row = logits_f32[tok * case.num_experts ..][0..case.num_experts];
                const start_in_pool: u32 = @min(
                    @as(u32, @intCast(tok)) * stride,
                    case.shared_pool - case.top_k,
                );
                for (0..case.top_k) |k_idx| {
                    const eid: usize = @intCast((base_e + start_in_pool + @as(u32, @intCast(k_idx))) % case.num_experts);
                    tok_row[eid] = 6.0 - @as(f32, @floatFromInt(k_idx)) * 0.1;
                }
            }
        }

        const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
        defer gpu_buf.close();
        try compute.makeResident(&.{gpu_buf});
        defer compute.removeResident(&.{gpu_buf}) catch |err| {
            std.debug.panic("removeResident failed during G3 bench cleanup: {}", .{err});
        };

        const weights_base: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(weights_raw.ptr) };
        const x_bind: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(x_raw.ptr) };
        const out_bind: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out_raw.ptr) };
        const tbl_bind: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(tbl_raw.ptr) };
        const logits_base: u64 = arena.offsetOf(logits_raw.ptr);
        const out_f16: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, out_raw));

        var bank_cursor: usize = 0;
        const warmups = config.warmup_batches + if (case_idx == 0) config.dvfs_warmup_batches else 0;
        var w_sig: u64 = 0;
        for (0..warmups) |_| {
            var frame = try compute.beginFrame(config.timeout_ms);
            errdefer frame.abort();
            for (0..config.dispatches_per_batch) |d| {
                const b = bank_cursor % num_banks;
                bank_cursor += 1;
                try pipes.encodeMoELayer(&frame, case.bits, .{
                    .x = x_bind,
                    .router = .{ .logits = .{ .binding = .{ .buffer = gpu_buf, .offset = logits_base + @as(u64, b) * bank_logits_bytes }, .dtype = .f32 } },
                    .weights_base = weights_base,
                    .expert_table = tbl_bind,
                    .workspace = &ws,
                    .workspace_buf = gpu_buf,
                    .out = out_bind,
                    .m = case.m,
                    .num_experts = case.num_experts,
                    .top_k = case.top_k,
                    .hidden_size = case.hidden_size,
                    .intermediate_size = case.intermediate_size,
                    .norm_topk_prob = true,
                }, d > 0);
            }
            w_sig = try frame.commit();
        }
        if (w_sig != 0) try compute.wait(w_sig, config.timeout_ms);

        const observed_unique: u32 = ws.schedule_header.num_unique_experts;

        var last_sig: u64 = 0;
        const t0 = try vmstat.nowNs();
        for (0..config.timed_batches) |_| {
            var frame = try compute.beginFrame(config.timeout_ms);
            errdefer frame.abort();
            for (0..config.dispatches_per_batch) |d| {
                const b = bank_cursor % num_banks;
                bank_cursor += 1;
                try pipes.encodeMoELayer(&frame, case.bits, .{
                    .x = x_bind,
                    .router = .{ .logits = .{ .binding = .{ .buffer = gpu_buf, .offset = logits_base + @as(u64, b) * bank_logits_bytes }, .dtype = .f32 } },
                    .weights_base = weights_base,
                    .expert_table = tbl_bind,
                    .workspace = &ws,
                    .workspace_buf = gpu_buf,
                    .out = out_bind,
                    .m = case.m,
                    .num_experts = case.num_experts,
                    .top_k = case.top_k,
                    .hidden_size = case.hidden_size,
                    .intermediate_size = case.intermediate_size,
                    .norm_topk_prob = true,
                }, d > 0);
            }
            last_sig = try frame.commit();
        }
        try compute.wait(last_sig, config.timeout_ms);
        const t1 = try vmstat.nowNs();

        const out_elems: usize = @as(usize, case.m) * case.hidden_size;
        if (!std.math.isFinite(out_f16[0]) or !std.math.isFinite(out_f16[out_elems - 1])) {
            return error.NonFiniteOutput;
        }

        const naive_slots: u32 = case.m * case.top_k;
        results[case_idx] = .{
            .case = case,
            .unique_experts = observed_unique,
            .naive_slots = naive_slots,
            .dedup_bytes = try moe.expertLayerBytes(case.bits, observed_unique, case.hidden_size, case.intermediate_size),
            .naive_bytes = try moe.expertLayerBytes(case.bits, naive_slots, case.hidden_size, case.intermediate_size),
            .dispatches = @as(u64, config.timed_batches) * config.dispatches_per_batch,
            .elapsed_ns = t1 - t0,
        };
    }

    const vm_after = try vmstat.Snapshot.capture();
    try vmstat.checkNoSwap(vm_before, vm_after);

    return .{
        .results = results,
        .vm_before = vm_before,
        .vm_after = vm_after,
    };
}

// ---------------------------------------------------------------------------
// M2 End-to-End .amf Greedy Decode Benchmark & Fast Streaming .amf Writer
// ---------------------------------------------------------------------------

pub const SynthWriteError = error{
    OpenFailed,
    WriteFailed,
    SeekFailed,
    OutOfMemory,
} || qwen3.ModelConfig.ConfigError || quant.ShapeError;

fn writeAllFd(fd: std.c.fd_t, data: []const u8) SynthWriteError!void {
    var written: usize = 0;
    while (written < data.len) {
        const rc = std.c.write(fd, data.ptr + written, data.len - written);
        if (rc < 0) {
            if (std.c.errno(rc) == .INTR) continue;
            return error.WriteFailed;
        }
        if (rc == 0) return error.WriteFailed;
        written += @intCast(rc);
    }
}

const StreamTensorPlan = struct {
    name: [48]u8,
    kind: amf.TensorKind,
    dtype: amf.DType,
    quant_bits: u8,
    group_size: u16,
    layer: u16,
    expert: u16,
    shape: [4]u32,
    rank: u8,
    scale_hint: f32,
};

fn makePlan(
    name: []const u8,
    kind: amf.TensorKind,
    dtype: amf.DType,
    quant_bits: u8,
    group_size: u16,
    layer: u16,
    expert: u16,
    shape: [4]u32,
    rank: u8,
    scale_hint: f32,
) StreamTensorPlan {
    var p: StreamTensorPlan = .{
        .name = @splat(0),
        .kind = kind,
        .dtype = dtype,
        .quant_bits = quant_bits,
        .group_size = group_size,
        .layer = layer,
        .expert = expert,
        .shape = shape,
        .rank = rank,
        .scale_hint = scale_hint,
    };
    const n = @min(name.len, p.name.len - 1);
    @memcpy(p.name[0..n], name[0..n]);
    return p;
}

/// Streams a deterministic, valid, 16 KiB-aligned `.amf` v0 file for `cfg`
/// straight to disk with bounded (~1 MiB) scratch memory.
/// Supports both affine-quantized (`q3`/`q4`) and native unquantized (`f16`/`bf16`)
/// configs.
pub fn writeSyntheticQwen3AmfFile(
    allocator: std.mem.Allocator,
    path: [:0]const u8,
    cfg: qwen3.ModelConfig,
    seed: u64,
) SynthWriteError!void {
    try cfg.validate();

    const h = cfg.hidden_size;
    const q_dim = cfg.qDim();
    const kv_dim = cfg.kvDim();
    const is_quant = cfg.quant_bits != null;
    const lin_dtype: amf.DType = if (cfg.quant_bits) |qb|
        (if (qb == .q3) .q3_affine else .q4_affine)
    else
        (if (cfg.default_float_dtype == .f16) .f16 else .bf16);
    const norm_dtype: amf.DType = if (is_quant) .f16 else lin_dtype;
    const bits_u8: u8 = if (cfg.quant_bits) |qb| qb.bitWidth() else 0;
    const gs_u16: u16 = if (is_quant) 64 else 0;
    const inv_sqrt_h = 1.0 / @sqrt(@as(f32, @floatFromInt(h)));
    const inv_sqrt_q = 1.0 / @sqrt(@as(f32, @floatFromInt(q_dim)));
    const base_mul: f32 = switch (lin_dtype) {
        .q3_affine => 0.35,
        .q4_affine => 0.17,
        .f16, .bf16 => 0.5,
        .f32, .u32 => unreachable,
    };

    var plans: std.ArrayList(StreamTensorPlan) = .empty;
    defer plans.deinit(allocator);

    try plans.append(allocator, makePlan(
        if (is_quant) "embed_tokens" else "embed_tokens.weight",
        .embedding,
        lin_dtype,
        bits_u8,
        gs_u16,
        amf.NO_LAYER,
        amf.NO_EXPERT,
        .{ cfg.vocab_size, h, 0, 0 },
        2,
        switch (lin_dtype) {
            .q3_affine => 0.06,
            .q4_affine => 0.03,
            .f16, .bf16 => 0.15,
            .f32, .u32 => unreachable,
        },
    ));

    const suf: []const u8 = if (is_quant) "" else ".weight";
    var name_buf: [64]u8 = undefined;
    for (0..cfg.num_hidden_layers) |l_usize| {
        const l: u16 = @intCast(l_usize);
        try plans.append(allocator, makePlan(std.fmt.bufPrint(&name_buf, "layers.{d}.input_layernorm.weight", .{l}) catch unreachable, .norm, norm_dtype, 0, 0, l, amf.NO_EXPERT, .{ h, 0, 0, 0 }, 1, 1.0));
        try plans.append(allocator, makePlan(std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.q_proj{s}", .{ l, suf }) catch unreachable, .dense, lin_dtype, bits_u8, gs_u16, l, amf.NO_EXPERT, .{ q_dim, h, 0, 0 }, 2, base_mul * inv_sqrt_h));
        try plans.append(allocator, makePlan(std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.k_proj{s}", .{ l, suf }) catch unreachable, .dense, lin_dtype, bits_u8, gs_u16, l, amf.NO_EXPERT, .{ kv_dim, h, 0, 0 }, 2, base_mul * inv_sqrt_h));
        try plans.append(allocator, makePlan(std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.v_proj{s}", .{ l, suf }) catch unreachable, .dense, lin_dtype, bits_u8, gs_u16, l, amf.NO_EXPERT, .{ kv_dim, h, 0, 0 }, 2, base_mul * inv_sqrt_h));
        try plans.append(allocator, makePlan(std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.o_proj{s}", .{ l, suf }) catch unreachable, .dense, lin_dtype, bits_u8, gs_u16, l, amf.NO_EXPERT, .{ h, q_dim, 0, 0 }, 2, base_mul * inv_sqrt_q));
        try plans.append(allocator, makePlan(std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.q_norm.weight", .{l}) catch unreachable, .norm, norm_dtype, 0, 0, l, amf.NO_EXPERT, .{ cfg.head_dim, 0, 0, 0 }, 1, 1.0));
        try plans.append(allocator, makePlan(std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.k_norm.weight", .{l}) catch unreachable, .norm, norm_dtype, 0, 0, l, amf.NO_EXPERT, .{ cfg.head_dim, 0, 0, 0 }, 1, 1.0));
        try plans.append(allocator, makePlan(std.fmt.bufPrint(&name_buf, "layers.{d}.post_attention_layernorm.weight", .{l}) catch unreachable, .norm, norm_dtype, 0, 0, l, amf.NO_EXPERT, .{ h, 0, 0, 0 }, 1, 1.0));

        if (cfg.arch == .qwen3_moe) {
            try plans.append(allocator, makePlan(std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.gate.weight", .{l}) catch unreachable, .router, norm_dtype, 0, 0, l, amf.NO_EXPERT, .{ cfg.num_experts, h, 0, 0 }, 2, 0.5 * inv_sqrt_h));
            const inv_sqrt_mi = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.moe_intermediate_size)));
            for (0..cfg.num_experts) |e_usize| {
                const e: u16 = @intCast(e_usize);
                try plans.append(allocator, makePlan(std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.experts.{d}.gate_proj{s}", .{ l, e, suf }) catch unreachable, .hot_expert, lin_dtype, bits_u8, gs_u16, l, e, .{ cfg.moe_intermediate_size, h, 0, 0 }, 2, base_mul * inv_sqrt_h));
                try plans.append(allocator, makePlan(std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.experts.{d}.up_proj{s}", .{ l, e, suf }) catch unreachable, .hot_expert, lin_dtype, bits_u8, gs_u16, l, e, .{ cfg.moe_intermediate_size, h, 0, 0 }, 2, base_mul * inv_sqrt_h));
                try plans.append(allocator, makePlan(std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.experts.{d}.down_proj{s}", .{ l, e, suf }) catch unreachable, .hot_expert, lin_dtype, bits_u8, gs_u16, l, e, .{ h, cfg.moe_intermediate_size, 0, 0 }, 2, base_mul * inv_sqrt_mi));
            }
        } else {
            const inv_sqrt_di = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.intermediate_size)));
            try plans.append(allocator, makePlan(std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.gate_proj{s}", .{ l, suf }) catch unreachable, .dense, lin_dtype, bits_u8, gs_u16, l, amf.NO_EXPERT, .{ cfg.intermediate_size, h, 0, 0 }, 2, base_mul * inv_sqrt_h));
            try plans.append(allocator, makePlan(std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.up_proj{s}", .{ l, suf }) catch unreachable, .dense, lin_dtype, bits_u8, gs_u16, l, amf.NO_EXPERT, .{ cfg.intermediate_size, h, 0, 0 }, 2, base_mul * inv_sqrt_h));
            try plans.append(allocator, makePlan(std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.down_proj{s}", .{ l, suf }) catch unreachable, .dense, lin_dtype, bits_u8, gs_u16, l, amf.NO_EXPERT, .{ h, cfg.intermediate_size, 0, 0 }, 2, base_mul * inv_sqrt_di));
        }
    }

    try plans.append(allocator, makePlan("norm.weight", .norm, norm_dtype, 0, 0, amf.NO_LAYER, amf.NO_EXPERT, .{ h, 0, 0, 0 }, 1, 1.0));
    if (!cfg.tie_word_embeddings) {
        try plans.append(allocator, makePlan(if (is_quant) "lm_head" else "lm_head.weight", .lm_head, lin_dtype, bits_u8, gs_u16, amf.NO_LAYER, amf.NO_EXPERT, .{ cfg.vocab_size, h, 0, 0 }, 2, 0.45 * base_mul * inv_sqrt_h));
    }

    const fd = std.c.open(path.ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o644));
    if (fd < 0) return error.OpenFailed;
    defer _ = std.c.close(fd);
    // Bypass macOS Unified Buffer Cache so writing multi-GiB benchmark .amf files
    // never pollutes host page cache or triggers background swapouts.
    _ = std.c.fcntl(fd, std.c.F.NOCACHE, @as(c_int, 1));

    const dir_offset: usize = page.APPLE_PAGE_SIZE;
    const dir_raw_len: usize = plans.items.len * @sizeOf(amf.TensorEntry);
    const dir_bytes: usize = try page.alignUp(dir_raw_len);

    const chunk_buf = try allocator.alignedAlloc(u8, .fromByteUnits(64), 1024 * 1024);
    defer allocator.free(chunk_buf);
    var zero_page: [page.APPLE_PAGE_SIZE]u8 = @splat(0);

    // Reserve header + directory pages.
    var reserved: usize = dir_offset + dir_bytes;
    while (reserved > 0) {
        const take = @min(reserved, zero_page.len);
        try writeAllFd(fd, zero_page[0..take]);
        reserved -= take;
    }

    const entries = try allocator.alloc(amf.TensorEntry, plans.items.len);
    defer allocator.free(entries);

    var cursor: u64 = dir_offset + dir_bytes;
    for (plans.items, 0..) |p, idx| {
        var rng: u64 = seed ^ (@as(u64, idx + 1) *% 0x9E37_79B9_7F4A_7C15);
        var hasher = std.hash.Crc32.init();
        var data_bytes: u64 = 0;
        var params_offset: u64 = 0;
        var params_bytes: u64 = 0;
        var blob_bytes: u64 = 0;

        if (p.dtype == .q3_affine or p.dtype == .q4_affine) {
            const n: usize = p.shape[0];
            const k: usize = p.shape[1];
            data_bytes = switch (p.dtype) {
                .q3_affine => try Q3.packedWeightBytes(n, k),
                .q4_affine => try Q4.packedWeightBytes(n, k),
                else => unreachable,
            };
            params_bytes = switch (p.dtype) {
                .q3_affine => try Q3.scaleBiasBytes(n, k),
                .q4_affine => try Q4.scaleBiasBytes(n, k),
                else => unreachable,
            };
            params_offset = amf.align64(@intCast(data_bytes));
            const content_len: usize = @intCast(params_offset + params_bytes);
            blob_bytes = try page.alignUp(content_len);

            // Stream random qweight bytes.
            var rem: usize = @intCast(data_bytes);
            while (rem > 0) {
                const take = @min(rem, chunk_buf.len);
                const words: []u64 = @alignCast(std.mem.bytesAsSlice(u64, chunk_buf[0 .. take & ~@as(usize, 7)]));
                for (words) |*w| {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    w.* = rng;
                }
                for (chunk_buf[take & ~@as(usize, 7) .. take]) |*b| {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    b.* = @truncate(rng);
                }
                try writeAllFd(fd, chunk_buf[0..take]);
                hasher.update(chunk_buf[0..take]);
                rem -= take;
            }

            // Mid zero padding to params_offset.
            if (params_offset > data_bytes) {
                const pad: usize = @intCast(params_offset - data_bytes);
                try writeAllFd(fd, zero_page[0..pad]);
                hasher.update(zero_page[0..pad]);
            }

            // Stream finite well-scaled ScaleBias pairs.
            const mid: f32 = if (p.dtype == .q3_affine) 3.5 else 7.5;
            var sb_table: [257]quant.ScaleBias = undefined;
            for (&sb_table) |*sb| {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                const u = @as(f32, @floatFromInt(rng & 0xFF)) / 255.0;
                const s = @max(@as(f32, 1e-4), p.scale_hint * (0.75 + 0.5 * u));
                const j = (@as(f32, @floatFromInt((rng >> 8) & 0xFF)) / 255.0 - 0.5) * 0.25 * s;
                sb.* = .{ .scale = @floatCast(s), .bias = @floatCast(-mid * s + j) };
            }
            const sb_chunk: []quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, chunk_buf));
            var sb_rem: usize = @intCast(params_bytes / @sizeOf(quant.ScaleBias));
            var phase: usize = 0;
            while (sb_rem > 0) {
                const take_elems = @min(sb_rem, sb_chunk.len);
                for (0..take_elems) |i| {
                    sb_chunk[i] = sb_table[phase];
                    phase += 1;
                    if (phase == sb_table.len) phase = 0;
                }
                const bytes_slice = std.mem.sliceAsBytes(sb_chunk[0..take_elems]);
                try writeAllFd(fd, bytes_slice);
                hasher.update(bytes_slice);
                sb_rem -= take_elems;
            }

            // Tail zero padding to 16 KiB boundary.
            if (blob_bytes > content_len) {
                var pad_rem: usize = @intCast(blob_bytes - content_len);
                while (pad_rem > 0) {
                    const take = @min(pad_rem, zero_page.len);
                    try writeAllFd(fd, zero_page[0..take]);
                    hasher.update(zero_page[0..take]);
                    pad_rem -= take;
                }
            }
        } else {
            var elems: usize = 1;
            for (0..p.rank) |d| elems *= p.shape[d];
            data_bytes = elems * @sizeOf(u16);
            blob_bytes = try page.alignUp(@intCast(data_bytes));

            const float_dt: quant.FloatDType = if (p.dtype == .bf16) .bf16 else .f16;
            var u16_table: [257]u16 = undefined;
            for (&u16_table) |*v| {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                const u = @as(f32, @floatFromInt(rng & 0xFFFF)) / 65535.0;
                const f = if (p.kind == .norm) (0.95 + 0.1 * u) else ((u * 2.0 - 1.0) * p.scale_hint);
                v.* = quant.f32ToFloatBits(float_dt, f);
            }
            const u16_chunk: []u16 = @alignCast(std.mem.bytesAsSlice(u16, chunk_buf));
            var rem_elems = elems;
            var phase: usize = 0;
            while (rem_elems > 0) {
                const take_elems = @min(rem_elems, u16_chunk.len);
                for (0..take_elems) |i| {
                    u16_chunk[i] = u16_table[phase];
                    phase += 1;
                    if (phase == u16_table.len) phase = 0;
                }
                const bytes_slice = std.mem.sliceAsBytes(u16_chunk[0..take_elems]);
                try writeAllFd(fd, bytes_slice);
                hasher.update(bytes_slice);
                rem_elems -= take_elems;
            }
            if (blob_bytes > data_bytes) {
                var pad_rem: usize = @intCast(blob_bytes - data_bytes);
                while (pad_rem > 0) {
                    const take = @min(pad_rem, zero_page.len);
                    try writeAllFd(fd, zero_page[0..take]);
                    hasher.update(zero_page[0..take]);
                    pad_rem -= take;
                }
            }
        }

        var entry = amf.TensorEntry.zeroed(std.mem.sliceTo(&p.name, 0), p.kind, p.dtype);
        entry.quant_bits = p.quant_bits;
        entry.rank = p.rank;
        entry.group_size = p.group_size;
        entry.layer = p.layer;
        entry.expert = p.expert;
        entry.shape = p.shape;
        entry.blob_offset = cursor;
        entry.blob_bytes = blob_bytes;
        entry.data_bytes = data_bytes;
        entry.params_offset = params_offset;
        entry.params_bytes = params_bytes;
        entry.blob_crc32 = hasher.final();
        entries[idx] = entry;
        cursor += blob_bytes;
    }

    const dir_slice = std.mem.sliceAsBytes(entries);
    const dir_crc = std.hash.Crc32.hash(dir_slice);

    var hdr: amf.Header = .{
        .magic = amf.MAGIC,
        .version = amf.FORMAT_VERSION,
        .flags = 0,
        .page_size = page.APPLE_PAGE_SIZE,
        .checksum_algo = amf.CHECKSUM_CRC32_IEEE,
        .tensor_count = @intCast(entries.len),
        .reserved0 = 0,
        .directory_offset = dir_offset,
        .directory_bytes = dir_bytes,
        .directory_crc32 = dir_crc,
        .header_crc32 = 0,
        .model_id = @splat(0),
        .reserved1 = 0,
    };
    const mid = cfg.arch.label();
    @memcpy(hdr.model_id[0..mid.len], mid);
    hdr.header_crc32 = amf.computeHeaderCrc32(&hdr);

    if (std.c.lseek(fd, 0, std.c.SEEK.SET) < 0) return error.SeekFailed;
    var hdr_page: [page.APPLE_PAGE_SIZE]u8 = @splat(0);
    @memcpy(hdr_page[0..@sizeOf(amf.Header)], std.mem.asBytes(&hdr));
    try writeAllFd(fd, &hdr_page);

    if (std.c.lseek(fd, @intCast(dir_offset), std.c.SEEK.SET) < 0) return error.SeekFailed;
    try writeAllFd(fd, dir_slice);
    if (dir_bytes > dir_slice.len) {
        var pad_rem = dir_bytes - dir_slice.len;
        while (pad_rem > 0) {
            const take = @min(pad_rem, zero_page.len);
            try writeAllFd(fd, zero_page[0..take]);
            pad_rem -= take;
        }
    }
}

pub const DecodeBenchOptions = struct {
    label: []const u8,
    /// Full target layer count for linear layer-body scaling (e.g. 48 for Qwen3-30B-A3B, 40 for Qwen3-14B).
    scale_layers: u32,
    prefill_tokens: u32 = 511,
    warmup_steps: u32 = 4,
    decode_steps: u32 = 16,
    max_seq_len: u32 = 576,
    verify_crcs: bool = true,
};

pub const DecodeBenchResult = struct {
    label_buf: [64]u8,
    label_len: usize,
    arch: qwen3.ModelArch,
    quant_bits: ?quant.QuantBits,
    default_float_dtype: quant.FloatDType = .bf16,
    measured_layers: u32,
    scale_layers: u32,
    seq_len: u32,
    decode_steps: u32,
    shard_count: usize,
    wired_bytes: u64,
    load_ms: f64,
    full_step_ns: u64,
    layers_only_ns: u64,
    measured_bytes_per_tok: u64,
    extrapolated_bytes_per_tok: u64,
    heap_delta: vmstat.HeapDelta,
    swapouts_delta: u64,
    pageouts_delta: u64,

    pub fn label(self: *const DecodeBenchResult) []const u8 {
        return self.label_buf[0..self.label_len];
    }

    pub fn precisionLabel(self: DecodeBenchResult) []const u8 {
        if (self.quant_bits) |qb| {
            return switch (qb) {
                .q3 => "q3",
                .q4 => "q4",
            };
        }
        return switch (self.default_float_dtype) {
            .f16 => "f16",
            .bf16 => "bf16",
        };
    }

    pub fn measuredMsPerTok(self: DecodeBenchResult) f64 {
        if (self.decode_steps == 0) return 0.0;
        return (@as(f64, @floatFromInt(self.full_step_ns)) / @as(f64, @floatFromInt(self.decode_steps))) / 1_000_000.0;
    }

    pub fn measuredPerLayerMs(self: DecodeBenchResult) f64 {
        if (self.decode_steps == 0 or self.measured_layers == 0) return 0.0;
        const body_ms = (@as(f64, @floatFromInt(self.layers_only_ns)) / @as(f64, @floatFromInt(self.decode_steps))) / 1_000_000.0;
        return body_ms / @as(f64, @floatFromInt(self.measured_layers));
    }

    pub fn measuredHeadMs(self: DecodeBenchResult) f64 {
        const body_ms = self.measuredPerLayerMs() * @as(f64, @floatFromInt(self.measured_layers));
        const full_ms = self.measuredMsPerTok();
        return if (full_ms > body_ms) full_ms - body_ms else 0.0;
    }

    pub fn measuredTokPerS(self: DecodeBenchResult) f64 {
        const ms = self.measuredMsPerTok();
        return if (ms > 0.0) 1000.0 / ms else 0.0;
    }

    pub fn measuredEffectiveGbs(self: DecodeBenchResult) f64 {
        const ns_per_tok = self.measuredMsPerTok() * 1_000_000.0;
        return if (ns_per_tok > 0.0) @as(f64, @floatFromInt(self.measured_bytes_per_tok)) / ns_per_tok else 0.0;
    }

    /// Linear layer scaling: `scale_layers * measured_per_layer_ms + measured_head_ms`.
    pub fn extrapolatedMsPerTok(self: DecodeBenchResult) f64 {
        return @as(f64, @floatFromInt(self.scale_layers)) * self.measuredPerLayerMs() + self.measuredHeadMs();
    }

    pub fn extrapolatedTokPerS(self: DecodeBenchResult) f64 {
        const ms = self.extrapolatedMsPerTok();
        return if (ms > 0.0) 1000.0 / ms else 0.0;
    }

    pub fn extrapolatedEffectiveGbs(self: DecodeBenchResult) f64 {
        const ns_per_tok = self.extrapolatedMsPerTok() * 1_000_000.0;
        return if (ns_per_tok > 0.0) @as(f64, @floatFromInt(self.extrapolated_bytes_per_tok)) / ns_per_tok else 0.0;
    }

    pub fn write(self: *const DecodeBenchResult, out: *std.Io.Writer) std.Io.Writer.Error!void {
        try out.print(
            \\decode_bench.target              {s}
            \\decode_bench.arch                {s} ({s})
            \\decode_bench.seq_len             {d}
            \\decode_bench.shards              {d}
            \\decode_bench.wired_mib           {d}
            \\decode_bench.load_ms             {d:.1}
            \\decode_bench.measured_layers     {d}
            \\decode_bench.measured_ms_per_tok {d:.3}
            \\decode_bench.measured_tok_per_s  {d:.2}
            \\decode_bench.measured_layer_ms   {d:.3}
            \\decode_bench.measured_head_ms    {d:.3}
            \\decode_bench.measured_bytes_tok  {d}
            \\decode_bench.measured_beta_gbs   {d:.2} ({d:.1}% bus)
            \\decode_bench.scaled_layers       {d} (linear: {d} * layer_ms + head_ms)
            \\decode_bench.scaled_ms_per_tok   {d:.3}
            \\decode_bench.scaled_tok_per_s    {d:.2}
            \\decode_bench.scaled_bytes_tok    {d}
            \\decode_bench.scaled_beta_gbs     {d:.2} ({d:.1}% bus)
            \\decode_bench.malloc_blocks_delta {d}
            \\decode_bench.malloc_bytes_delta  {d}
            \\decode_bench.swapouts_delta      {d}
            \\decode_bench.pageouts_delta      {d}
            \\
        , .{
            self.label(),
            self.arch.label(),
            self.precisionLabel(),
            self.seq_len,
            self.shard_count,
            self.wired_bytes >> 20,
            self.load_ms,
            self.measured_layers,
            self.measuredMsPerTok(),
            self.measuredTokPerS(),
            self.measuredPerLayerMs(),
            self.measuredHeadMs(),
            self.measured_bytes_per_tok,
            self.measuredEffectiveGbs(),
            (self.measuredEffectiveGbs() / M4_BUS_BANDWIDTH_GBS) * 100.0,
            self.scale_layers,
            self.scale_layers,
            self.extrapolatedMsPerTok(),
            self.extrapolatedTokPerS(),
            self.extrapolated_bytes_per_tok,
            self.extrapolatedEffectiveGbs(),
            (self.extrapolatedEffectiveGbs() / M4_BUS_BANDWIDTH_GBS) * 100.0,
            self.heap_delta.blocks_in_use,
            self.heap_delta.size_in_use,
            self.swapouts_delta,
            self.pageouts_delta,
        });
    }
};

pub const DecodeBenchError = qwen3.LoadError || qwen3.StepError || vmstat.Error || vmstat.SwapError || SynthWriteError || error{InvalidConfig};

pub fn runDecodeBenchFromFile(
    allocator: std.mem.Allocator,
    amf_path: [:0]const u8,
    cfg: qwen3.ModelConfig,
    opts: DecodeBenchOptions,
) DecodeBenchError!DecodeBenchResult {
    if (opts.decode_steps == 0 or opts.prefill_tokens + 1 >= opts.max_seq_len) return error.InvalidConfig;

    const load_t0 = try vmstat.nowNs();
    var model = try qwen3.Qwen3Model.loadFromFile(allocator, amf_path, cfg, .{
        .max_seq_len = opts.max_seq_len,
        .verify_crcs = opts.verify_crcs,
    });
    defer model.deinit(allocator);
    const load_t1 = try vmstat.nowNs();

    const vm_before = try vmstat.Snapshot.capture();

    // Populate real KV cache entries up to `prefill_tokens`.
    for (0..opts.prefill_tokens) |p| {
        const tok: u32 = @intCast((p * 17 + 5) % cfg.vocab_size);
        try model.prefillToken(tok, @intCast(p));
    }

    // Warmup full decode steps at `pos = prefill_tokens` (seq_len = prefill_tokens + 1).
    const eval_pos: u32 = opts.prefill_tokens;
    var cur_tok: u32 = 17 % cfg.vocab_size;
    for (0..opts.warmup_steps) |_| {
        cur_tok = try model.stepDecode(cur_tok, eval_pos);
    }

    // Timed steady-state full decode steps (`measured_layers` + final norm + lm_head + argmax).
    const heap_before = try vmstat.Snapshot.capture();
    const t_full_0 = try vmstat.nowNs();
    for (0..opts.decode_steps) |_| {
        cur_tok = try model.stepDecode(cur_tok, eval_pos);
    }
    const t_full_1 = try vmstat.nowNs();
    const heap_after = try vmstat.Snapshot.capture();

    // Timed transformer-body-only steps at identical `seq_len` to separate `T_layer` from `T_head`.
    const t_body_0 = try vmstat.nowNs();
    for (0..opts.decode_steps) |i| {
        const tok: u32 = @intCast((cur_tok + @as(u32, @intCast(i * 13))) % cfg.vocab_size);
        try model.prefillToken(tok, eval_pos);
    }
    const t_body_1 = try vmstat.nowNs();

    const vm_after = try vmstat.Snapshot.capture();
    try vmstat.checkNoSwap(vm_before, vm_after);

    const seq_len: u32 = eval_pos + 1;
    const measured_bytes = model.bytesPerDecodeToken(seq_len);
    const extrapolated_bytes = @as(u64, opts.scale_layers) * model.bytesPerLayerToken(seq_len) + model.bytesGlobalHead();

    var label_buf: [64]u8 = @splat(0);
    const copy_len = @min(opts.label.len, label_buf.len);
    @memcpy(label_buf[0..copy_len], opts.label[0..copy_len]);

    return .{
        .label_buf = label_buf,
        .label_len = copy_len,
        .arch = cfg.arch,
        .quant_bits = cfg.quant_bits,
        .default_float_dtype = cfg.default_float_dtype,
        .measured_layers = cfg.num_hidden_layers,
        .scale_layers = opts.scale_layers,
        .seq_len = seq_len,
        .decode_steps = opts.decode_steps,
        .shard_count = model.shardCount(),
        .wired_bytes = model.totalWiredBytes(),
        .load_ms = @as(f64, @floatFromInt(load_t1 - load_t0)) / 1_000_000.0,
        .full_step_ns = t_full_1 - t_full_0,
        .layers_only_ns = t_body_1 - t_body_0,
        .measured_bytes_per_tok = measured_bytes,
        .extrapolated_bytes_per_tok = extrapolated_bytes,
        .heap_delta = vmstat.heapDelta(heap_before, heap_after),
        .swapouts_delta = vmstat.Snapshot.swapoutDelta(vm_before, vm_after),
        .pageouts_delta = vmstat.Snapshot.pageoutDelta(vm_before, vm_after),
    };
}

pub const M2Report = struct {
    s1: attention.SpikeS1Report,
    g3: MoeBenchReport,
    g4: attention.GqaBenchReport,
    decode_moe: DecodeBenchResult,
    decode_dense: DecodeBenchResult,
    vm_before: vmstat.Snapshot,
    vm_after: vmstat.Snapshot,

    pub fn deinit(self: *M2Report, allocator: std.mem.Allocator) void {
        self.g3.deinit(allocator);
        self.g4.deinit(allocator);
    }

    pub fn write(self: *const M2Report, out: *std.Io.Writer) std.Io.Writer.Error!void {
        try out.writeAll("\n");
        try self.s1.write(out);
        try out.writeAll("\n");
        try self.g3.write(out);
        try out.writeAll("\n");
        try self.g4.write(out);
        try out.writeAll("\nM2 End-to-End Greedy Decode from .amf (Primary & Secondary Targets)\n");
        try self.decode_moe.write(out);
        try out.writeAll("\n");
        try self.decode_dense.write(out);
        try out.print(
            \\
            \\M2 Overall Swap Check
            \\m2.swap.status                   ok
            \\m2.swap.swapouts_delta           {d}
            \\m2.swap.pageouts_delta           {d}
            \\m2.swap.used_mib                 {d}
            \\
        , .{
            vmstat.Snapshot.swapoutDelta(self.vm_before, self.vm_after),
            vmstat.Snapshot.pageoutDelta(self.vm_before, self.vm_after),
            self.vm_after.swap_used_bytes >> 20,
        });
    }
};

pub const M2BenchError = attention.BenchError || MoeBenchError || DecodeBenchError;

fn tempDirPrefix() []const u8 {
    if (std.c.getenv("TMPDIR")) |raw| {
        var slice: []const u8 = std.mem.sliceTo(raw, 0);
        while (slice.len > 1 and slice[slice.len - 1] == '/') {
            slice = slice[0 .. slice.len - 1];
        }
        if (slice.len > 0) return slice;
    }
    return "/tmp";
}

/// Runs the complete M2 benchmark suite: Spike S1, G3 MoE deduplicated FFN,
/// G4 GQA KV-tile decode attention, and end-to-end `.amf` greedy decode on
/// real 4-layer Qwen3-30B-A3B-3bit (scaled to 48L) and Qwen3-14B-4bit (scaled to 40L).
pub fn runM2(allocator: std.mem.Allocator) M2BenchError!M2Report {
    const vm_before = try vmstat.Snapshot.capture();

    const s1 = try attention.runSpikeS1(allocator, .{});
    var g3 = try runMoeBench(allocator, .{});
    errdefer g3.deinit(allocator);
    var g4 = try attention.runGqaBench(allocator, .{});
    errdefer g4.deinit(allocator);

    const pid = std.c.getpid();
    const tmp_dir = tempDirPrefix();
    var moe_path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const moe_raw = std.fmt.bufPrint(moe_path_buf[0 .. moe_path_buf.len - 1], "{s}/aminfer-m2-moe-{d}.amf", .{ tmp_dir, pid }) catch unreachable;
    moe_path_buf[moe_raw.len] = 0;
    const moe_path: [:0]const u8 = moe_path_buf[0..moe_raw.len :0];

    const moe_cfg: qwen3.ModelConfig = .{
        .arch = .qwen3_moe,
        .quant_bits = .q3,
        .vocab_size = 151936,
        .hidden_size = 2048,
        .num_hidden_layers = 4,
        .num_attention_heads = 32,
        .num_key_value_heads = 4,
        .head_dim = 128,
        .moe_intermediate_size = 768,
        .num_experts = 128,
        .num_experts_per_tok = 8,
        .norm_topk_prob = true,
    };
    try writeSyntheticQwen3AmfFile(allocator, moe_path, moe_cfg, 30);
    defer _ = std.c.unlink(moe_path.ptr);
    const decode_moe = try runDecodeBenchFromFile(allocator, moe_path, moe_cfg, .{
        .label = "qwen3-30b-a3b-3bit",
        .scale_layers = 48,
        .prefill_tokens = 511,
        .warmup_steps = 4,
        .decode_steps = 16,
        .max_seq_len = 576,
        .verify_crcs = true,
    });

    var dense_path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dense_raw = std.fmt.bufPrint(dense_path_buf[0 .. dense_path_buf.len - 1], "{s}/aminfer-m2-dense-{d}.amf", .{ tmp_dir, pid }) catch unreachable;
    dense_path_buf[dense_raw.len] = 0;
    const dense_path: [:0]const u8 = dense_path_buf[0..dense_raw.len :0];

    const dense_cfg: qwen3.ModelConfig = .{
        .arch = .qwen3_dense,
        .quant_bits = .q4,
        .vocab_size = 151936,
        .hidden_size = 5120,
        .num_hidden_layers = 4,
        .num_attention_heads = 40,
        .num_key_value_heads = 8,
        .head_dim = 128,
        .intermediate_size = 17408,
    };
    try writeSyntheticQwen3AmfFile(allocator, dense_path, dense_cfg, 14);
    defer _ = std.c.unlink(dense_path.ptr);
    const decode_dense = try runDecodeBenchFromFile(allocator, dense_path, dense_cfg, .{
        .label = "qwen3-14b-4bit",
        .scale_layers = 40,
        .prefill_tokens = 511,
        .warmup_steps = 4,
        .decode_steps = 16,
        .max_seq_len = 576,
        .verify_crcs = true,
    });

    const vm_after = try vmstat.Snapshot.capture();
    try vmstat.checkNoSwap(vm_before, vm_after);

    return .{
        .s1 = s1,
        .g3 = g3,
        .g4 = g4,
        .decode_moe = decode_moe,
        .decode_dense = decode_dense,
        .vm_before = vm_before,
        .vm_after = vm_after,
    };
}

// ---------------------------------------------------------------------------
// M2.5 Native FP16 / BF16 GEMV & GEMM Bandwidth Calibration + BF16 Decode
// ---------------------------------------------------------------------------

pub const DenseKernelKind = enum {
    gemv_f16,
    gemv_bf16,
    gemm_f16,
    gemm_bf16,

    pub fn floatDType(self: DenseKernelKind) quant.FloatDType {
        return switch (self) {
            .gemv_f16, .gemm_f16 => .f16,
            .gemv_bf16, .gemm_bf16 => .bf16,
        };
    }

    pub fn isGemv(self: DenseKernelKind) bool {
        return switch (self) {
            .gemv_f16, .gemv_bf16 => true,
            .gemm_f16, .gemm_bf16 => false,
        };
    }

    pub fn label(self: DenseKernelKind) []const u8 {
        return switch (self) {
            .gemv_f16 => "gemv_f16",
            .gemv_bf16 => "gemv_bf16",
            .gemm_f16 => "gemm_f16",
            .gemm_bf16 => "gemm_bf16",
        };
    }
};

pub const DenseCase = struct {
    kind: DenseKernelKind,
    m: u32,
    n: u32,
    k: u32,
};

pub const DEFAULT_M25_CASES = [_]DenseCase{
    // M=1 decode GEMV calibration (f16)
    .{ .kind = .gemv_f16, .m = 1, .n = 2048, .k = 2048 },
    .{ .kind = .gemv_f16, .m = 1, .n = 6144, .k = 2048 },
    .{ .kind = .gemv_f16, .m = 1, .n = 2048, .k = 768 },
    .{ .kind = .gemv_f16, .m = 1, .n = 4096, .k = 4096 },
    // M=1 decode GEMV calibration (bf16)
    .{ .kind = .gemv_bf16, .m = 1, .n = 2048, .k = 2048 },
    .{ .kind = .gemv_bf16, .m = 1, .n = 6144, .k = 2048 },
    .{ .kind = .gemv_bf16, .m = 1, .n = 2048, .k = 768 },
    .{ .kind = .gemv_bf16, .m = 1, .n = 4096, .k = 4096 },
    // M=16 and M=64 speculative/prefill GEMM calibration (f16)
    .{ .kind = .gemm_f16, .m = 16, .n = 2048, .k = 2048 },
    .{ .kind = .gemm_f16, .m = 16, .n = 4096, .k = 4096 },
    .{ .kind = .gemm_f16, .m = 64, .n = 2048, .k = 2048 },
    .{ .kind = .gemm_f16, .m = 64, .n = 4096, .k = 4096 },
    // M=16 and M=64 speculative/prefill GEMM calibration (bf16)
    .{ .kind = .gemm_bf16, .m = 16, .n = 2048, .k = 2048 },
    .{ .kind = .gemm_bf16, .m = 16, .n = 4096, .k = 4096 },
    .{ .kind = .gemm_bf16, .m = 64, .n = 2048, .k = 2048 },
    .{ .kind = .gemm_bf16, .m = 64, .n = 4096, .k = 4096 },
};

pub const DenseBenchConfig = struct {
    cases: []const DenseCase = &DEFAULT_M25_CASES,
    /// Rotating weight pool size (>= 64 MiB per shape so even 32 MiB 4096x4096
    /// matrices rotate past M4 L2/SLC). Set to 0 in unit smoke tests.
    min_working_set_bytes: usize = 96 * 1024 * 1024,
    dvfs_warmup_batches: u32 = 16,
    warmup_batches: u32 = 4,
    gemv_timed_batches: u32 = 8,
    gemv_dispatches_per_batch: u32 = 16,
    gemm_timed_batches: u32 = 4,
    gemm_dispatches_per_batch: u32 = 16,
    timeout_ms: u64 = 30_000,
};

pub const DenseCaseResult = struct {
    case: DenseCase,
    bytes_per_dispatch: u64,
    dispatches: u64,
    weight_slots: usize,
    elapsed_ns: u64,

    pub fn latencyUs(self: DenseCaseResult) f64 {
        if (self.dispatches == 0) return 0.0;
        return (@as(f64, @floatFromInt(self.elapsed_ns)) / @as(f64, @floatFromInt(self.dispatches))) / 1000.0;
    }

    pub fn effectiveBandwidthGbs(self: DenseCaseResult) f64 {
        if (self.elapsed_ns == 0) return 0.0;
        const total_bytes = @as(f64, @floatFromInt(self.bytes_per_dispatch)) * @as(f64, @floatFromInt(self.dispatches));
        return total_bytes / @as(f64, @floatFromInt(self.elapsed_ns));
    }

    pub fn busUtilizationPct(self: DenseCaseResult) f64 {
        return (self.effectiveBandwidthGbs() / M4_BUS_BANDWIDTH_GBS) * 100.0;
    }

    pub fn effectiveTflops(self: DenseCaseResult) f64 {
        if (self.elapsed_ns == 0) return 0.0;
        const flops = 2.0 * @as(f64, @floatFromInt(self.case.m)) *
            @as(f64, @floatFromInt(self.case.n)) *
            @as(f64, @floatFromInt(self.case.k)) *
            @as(f64, @floatFromInt(self.dispatches));
        // flops / elapsed_ns = GFLOP/s; divide by 1000 for TFLOP/s.
        return (flops / @as(f64, @floatFromInt(self.elapsed_ns))) / 1000.0;
    }
};

pub const DenseBenchReport = struct {
    wired_budget_bytes: u64,
    arena_bytes: u64,
    results: []DenseCaseResult,
    vm_before: vmstat.Snapshot,
    vm_after: vmstat.Snapshot,

    pub fn deinit(self: *DenseBenchReport, allocator: std.mem.Allocator) void {
        allocator.free(self.results);
        self.results = &.{};
    }

    pub fn write(self: *const DenseBenchReport, out: *std.Io.Writer) std.Io.Writer.Error!void {
        try out.print(
            \\M2.5 Native FP16 / BF16 GEMV & GEMM Bandwidth Calibration
            \\m25.wired_budget_mib             {d}
            \\m25.arena_mib                    {d}
            \\kernel     shape              bytes/disp   slots   latency(us)   beta(GB/s)   bus(%)   tflops
            \\
        , .{
            self.wired_budget_bytes >> 20,
            self.arena_bytes >> 20,
        });
        for (self.results) |res| {
            var shape_buf: [40]u8 = undefined;
            const shape = std.fmt.bufPrint(&shape_buf, "{d}x{d}x{d}", .{
                res.case.m,
                res.case.n,
                res.case.k,
            }) catch unreachable;
            try out.print(
                \\{s: <10} {s: <18} {d: >10}   {d: >5}   {d: >11.2}   {d: >10.2}   {d: >6.1}   {d: >6.3}
                \\
            , .{
                res.case.kind.label(),
                shape,
                res.bytes_per_dispatch,
                res.weight_slots,
                res.latencyUs(),
                res.effectiveBandwidthGbs(),
                res.busUtilizationPct(),
                res.effectiveTflops(),
            });
        }
        try out.print(
            \\m25.dense.swapouts_delta         {d}
            \\m25.dense.pageouts_delta         {d}
            \\
        , .{
            vmstat.Snapshot.swapoutDelta(self.vm_before, self.vm_after),
            vmstat.Snapshot.pageoutDelta(self.vm_before, self.vm_after),
        });
    }
};

const DenseCaseLayout = struct {
    w_bytes: usize,
    x_bytes: usize,
    y_bytes: usize,
    bytes_per_dispatch: u64,
};

fn computeDenseCaseLayout(case: DenseCase) (quant.ShapeError || error{InvalidConfig})!DenseCaseLayout {
    if (case.m == 0 or case.n == 0 or case.k == 0) return error.ZeroDimension;
    if (case.kind.isGemv() and case.m != 1) return error.InvalidConfig;

    const w_elems = std.math.mul(usize, case.n, case.k) catch return error.Overflow;
    const w_bytes = std.math.mul(usize, w_elems, @sizeOf(u16)) catch return error.Overflow;
    const x_elems = std.math.mul(usize, case.m, case.k) catch return error.Overflow;
    const x_bytes = std.math.mul(usize, x_elems, @sizeOf(f16)) catch return error.Overflow;
    const y_elems = std.math.mul(usize, case.m, case.n) catch return error.Overflow;
    const y_bytes = std.math.mul(usize, y_elems, @sizeOf(f16)) catch return error.Overflow;

    const wx = std.math.add(u64, w_bytes, x_bytes) catch return error.Overflow;
    const total = std.math.add(u64, wx, y_bytes) catch return error.Overflow;
    return .{
        .w_bytes = w_bytes,
        .x_bytes = x_bytes,
        .y_bytes = y_bytes,
        .bytes_per_dispatch = total,
    };
}

/// Computes the algorithmic minimum bytes transferred per native FP16/BF16 dispatch
/// (`weight + x + y`).
pub fn denseBytesMovedPerDispatch(case: DenseCase) (quant.ShapeError || error{InvalidConfig})!u64 {
    return (try computeDenseCaseLayout(case)).bytes_per_dispatch;
}

fn fillDenseWeightPool(weight_pool: []u8, dtype: quant.FloatDType) void {
    var table: [257]u16 = undefined;
    var state: u64 = 0xD1B5_4A32_D192_ED03 ^ @as(u64, @backingInt(dtype));
    for (&table) |*v| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        const u = @as(f32, @floatFromInt(state & 0xFFFF)) / 65535.0;
        const f = (u * 2.0 - 1.0) * 0.03125;
        v.* = quant.f32ToFloatBits(dtype, f);
    }
    const words: []u16 = @alignCast(std.mem.bytesAsSlice(u16, weight_pool));
    var phase: usize = 0;
    for (words) |*w| {
        w.* = table[phase];
        phase += 1;
        if (phase == table.len) phase = 0;
    }
}

/// Executes M2.5 native FP16 / BF16 GEMV and GEMM bandwidth calibration over a
/// single `Budget`-checked, zero-copy `WiredArena`, enforcing zero swapouts.
pub fn runDenseBench(allocator: std.mem.Allocator, config: DenseBenchConfig) Error!DenseBenchReport {
    if (config.gemv_timed_batches == 0 or
        config.gemv_dispatches_per_batch == 0 or
        config.gemv_dispatches_per_batch > MAX_DISPATCHES_PER_FRAME or
        config.gemm_timed_batches == 0 or
        config.gemm_dispatches_per_batch == 0 or
        config.gemm_dispatches_per_batch > MAX_DISPATCHES_PER_FRAME)
    {
        return error.InvalidConfig;
    }

    var max_w_stride: usize = 64;
    var max_x_bytes: usize = 64;
    var max_y_bytes: usize = 64;
    for (config.cases) |case| {
        const layout = try computeDenseCaseLayout(case);
        max_w_stride = @max(max_w_stride, amf.align64(layout.w_bytes));
        max_x_bytes = @max(max_x_bytes, amf.align64(layout.x_bytes));
        max_y_bytes = @max(max_y_bytes, amf.align64(layout.y_bytes));
    }

    const weight_pool_bytes = amf.align64(@max(max_w_stride, config.min_working_set_bytes));
    const raw_arena_bytes = weight_pool_bytes + max_x_bytes + max_y_bytes + 4 * page.APPLE_PAGE_SIZE;
    const arena_bytes = try page.alignUp(raw_arena_bytes);

    const facts = try sysinfo.Facts.read();
    const device = try metal.Device.open();
    defer device.close();
    const info = try device.info();
    const wired_budget = budget_mod.Budget.fromHost(facts, info, .{});

    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try dequant.DequantPipelines.init(device, library);
    defer pipes.close();

    const results = try allocator.alloc(DenseCaseResult, config.cases.len);
    errdefer allocator.free(results);

    const vm_before = try vmstat.Snapshot.capture();

    var arena = try arena_mod.WiredArena.init(arena_bytes, wired_budget);
    defer arena.deinit();

    const weight_pool = try arena.allocAligned(weight_pool_bytes, 64);
    const x_raw = try arena.allocAligned(max_x_bytes, 64);
    const y_raw = try arena.allocAligned(max_y_bytes, 64);

    const x_table = [_]f16{
        @floatCast(0.03125),
        @floatCast(-0.015625),
        @floatCast(0.046875),
        @floatCast(-0.03125),
        @floatCast(0.015625),
        @floatCast(-0.046875),
        @floatCast(0.0625),
        @floatCast(-0.0625),
    };
    const x_slice: []f16 = @alignCast(std.mem.bytesAsSlice(f16, x_raw));
    for (x_slice, 0..) |*v, i| v.* = x_table[i & 7];
    @memset(y_raw, 0);

    const w_base_off: u64 = arena.offsetOf(weight_pool.ptr);
    const x_off: u64 = arena.offsetOf(x_raw.ptr);
    const y_off: u64 = arena.offsetOf(y_raw.ptr);
    const y_f16: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, y_raw));

    const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
    defer gpu_buf.close();

    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();
    try compute.makeResident(&.{gpu_buf});

    var filled_dtype: ?quant.FloatDType = null;
    for (config.cases, 0..) |case, case_idx| {
        const dt = case.kind.floatDType();
        if (filled_dtype != dt) {
            fillDenseWeightPool(weight_pool, dt);
            filled_dtype = dt;
        }
        const layout = try computeDenseCaseLayout(case);
        const w_stride = amf.align64(layout.w_bytes);
        const num_slots: usize = @max(@as(usize, 1), weight_pool_bytes / w_stride);

        const timed_batches = if (case.kind.isGemv()) config.gemv_timed_batches else config.gemm_timed_batches;
        const dispatches_per_batch = if (case.kind.isGemv()) config.gemv_dispatches_per_batch else config.gemm_dispatches_per_batch;

        var slot_cursor: usize = 0;
        const warmups = config.warmup_batches + if (case_idx == 0) config.dvfs_warmup_batches else 0;
        var warmup_sig: u64 = 0;
        for (0..warmups) |_| {
            var frame = try compute.beginFrame(config.timeout_ms);
            errdefer frame.abort();
            for (0..dispatches_per_batch) |d| {
                const slot = slot_cursor % num_slots;
                slot_cursor += 1;
                const w_bind: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = w_base_off + @as(u64, slot) * w_stride };
                if (case.kind.isGemv()) {
                    try pipes.encodeDenseGemv(&frame, dt, .{
                        .x = .{ .buffer = gpu_buf, .offset = x_off },
                        .weight = w_bind,
                        .y = .{ .buffer = gpu_buf, .offset = y_off },
                        .n = case.n,
                        .k = case.k,
                    }, d > 0);
                } else {
                    try pipes.encodeDenseGemm(&frame, dt, .{
                        .x = .{ .buffer = gpu_buf, .offset = x_off },
                        .weight = w_bind,
                        .y = .{ .buffer = gpu_buf, .offset = y_off },
                        .m = case.m,
                        .n = case.n,
                        .k = case.k,
                    }, d > 0);
                }
            }
            warmup_sig = try frame.commit();
        }
        if (warmup_sig != 0) try compute.wait(warmup_sig, config.timeout_ms);

        var last_sig: u64 = 0;
        const t0 = try vmstat.nowNs();
        for (0..timed_batches) |_| {
            var frame = try compute.beginFrame(config.timeout_ms);
            errdefer frame.abort();
            for (0..dispatches_per_batch) |d| {
                const slot = slot_cursor % num_slots;
                slot_cursor += 1;
                const w_bind: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = w_base_off + @as(u64, slot) * w_stride };
                if (case.kind.isGemv()) {
                    try pipes.encodeDenseGemv(&frame, dt, .{
                        .x = .{ .buffer = gpu_buf, .offset = x_off },
                        .weight = w_bind,
                        .y = .{ .buffer = gpu_buf, .offset = y_off },
                        .n = case.n,
                        .k = case.k,
                    }, d > 0);
                } else {
                    try pipes.encodeDenseGemm(&frame, dt, .{
                        .x = .{ .buffer = gpu_buf, .offset = x_off },
                        .weight = w_bind,
                        .y = .{ .buffer = gpu_buf, .offset = y_off },
                        .m = case.m,
                        .n = case.n,
                        .k = case.k,
                    }, d > 0);
                }
            }
            last_sig = try frame.commit();
        }
        try compute.wait(last_sig, config.timeout_ms);
        const t1 = try vmstat.nowNs();

        const y_count: usize = @as(usize, case.m) * case.n;
        if (!std.math.isFinite(y_f16[0]) or !std.math.isFinite(y_f16[y_count - 1])) {
            return error.NonFiniteOutput;
        }

        results[case_idx] = .{
            .case = case,
            .bytes_per_dispatch = layout.bytes_per_dispatch,
            .dispatches = @as(u64, timed_batches) * dispatches_per_batch,
            .weight_slots = num_slots,
            .elapsed_ns = t1 - t0,
        };
    }

    const vm_after = try vmstat.Snapshot.capture();
    try vmstat.checkNoSwap(vm_before, vm_after);

    return .{
        .wired_budget_bytes = wired_budget.limit_bytes,
        .arena_bytes = arena.capacity(),
        .results = results,
        .vm_before = vm_before,
        .vm_after = vm_after,
    };
}

pub const M25Report = struct {
    dense: DenseBenchReport,
    decode_bf16: DecodeBenchResult,
    vm_before: vmstat.Snapshot,
    vm_after: vmstat.Snapshot,

    pub fn deinit(self: *M25Report, allocator: std.mem.Allocator) void {
        self.dense.deinit(allocator);
    }

    pub fn write(self: *const M25Report, out: *std.Io.Writer) std.Io.Writer.Error!void {
        try out.writeAll("\n");
        try self.dense.write(out);
        try out.writeAll("\nM2.5 Synthetic BF16 End-to-End Greedy Decode from .amf (Qwen3-0.6B Geometry)\n");
        try self.decode_bf16.write(out);
        try out.print(
            \\
            \\M2.5 Overall Swap Check
            \\m25.swap.status                  ok
            \\m25.swap.swapouts_delta          {d}
            \\m25.swap.pageouts_delta          {d}
            \\m25.swap.used_mib                {d}
            \\
        , .{
            vmstat.Snapshot.swapoutDelta(self.vm_before, self.vm_after),
            vmstat.Snapshot.pageoutDelta(self.vm_before, self.vm_after),
            self.vm_after.swap_used_bytes >> 20,
        });
    }
};

pub const M25BenchError = Error || DecodeBenchError;

/// Runs the M2.5 benchmark suite: native FP16 / BF16 GEMV & GEMM bandwidth
/// calibration across C9 shapes + synthetic BF16 end-to-end `.amf` greedy decode
/// (Qwen3-0.6B geometry, `tie_word_embeddings = true`, 4L scaled to 28L).
pub fn runM25(allocator: std.mem.Allocator) M25BenchError!M25Report {
    const vm_before = try vmstat.Snapshot.capture();

    var dense = try runDenseBench(allocator, .{});
    errdefer dense.deinit(allocator);

    const pid = std.c.getpid();
    const tmp_dir = tempDirPrefix();
    var bf16_path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const bf16_raw = std.fmt.bufPrint(bf16_path_buf[0 .. bf16_path_buf.len - 1], "{s}/aminfer-m25-bf16-{d}.amf", .{ tmp_dir, pid }) catch unreachable;
    bf16_path_buf[bf16_raw.len] = 0;
    const bf16_path: [:0]const u8 = bf16_path_buf[0..bf16_raw.len :0];

    const bf16_cfg: qwen3.ModelConfig = .{
        .arch = .qwen3_dense,
        .quant_bits = null,
        .default_float_dtype = .bf16,
        .vocab_size = 151936,
        .hidden_size = 1024,
        .num_hidden_layers = 4,
        .num_attention_heads = 16,
        .num_key_value_heads = 8,
        .head_dim = 128,
        .intermediate_size = 3072,
        .tie_word_embeddings = true,
    };
    try writeSyntheticQwen3AmfFile(allocator, bf16_path, bf16_cfg, 6);
    defer _ = std.c.unlink(bf16_path.ptr);
    const decode_bf16 = try runDecodeBenchFromFile(allocator, bf16_path, bf16_cfg, .{
        .label = "qwen3-0.6b-bf16-synth",
        .scale_layers = 28,
        .prefill_tokens = 511,
        .warmup_steps = 4,
        .decode_steps = 16,
        .max_seq_len = 576,
        .verify_crcs = true,
    });

    const vm_after = try vmstat.Snapshot.capture();
    try vmstat.checkNoSwap(vm_before, vm_after);

    return .{
        .dense = dense,
        .decode_bf16 = decode_bf16,
        .vm_before = vm_before,
        .vm_after = vm_after,
    };
}

// ============================================================================
// M3 Speculative Decoding Benchmark Suite (Spikes S4/S5, L1 SAM, Synthetic E2E)
// ============================================================================

pub const SamLatencyResult = struct {
    context_tokens: u32,
    queries: u32,
    sam_arena_bytes: u64,
    append_us_per_tok: f64,
    query_active_us: f64,
    query_suffix_us: f64,
    max_query_us: f64,
    seed_hits: u32,
    mean_draft_len: f64,
    heap_delta: vmstat.HeapDelta,

    pub fn write(self: *const SamLatencyResult, out: *std.Io.Writer) std.Io.Writer.Error!void {
        try out.print(
            \\L1 Suffix Automaton Latency (D1, target < 50.00 us)
            \\sam.context_tokens               {d}
            \\sam.arena_mib                    {d:.2}
            \\sam.append_us_per_tok            {d:.3}
            \\sam.query_active_us              {d:.3}
            \\sam.query_suffix_us              {d:.3}
            \\sam.max_query_us                 {d:.3}
            \\sam.queries                      {d}
            \\sam.seed_hits                    {d}
            \\sam.mean_draft_len               {d:.2}
            \\sam.malloc_blocks_delta          {d}
            \\sam.malloc_bytes_delta           {d}
            \\
        , .{
            self.context_tokens,
            @as(f64, @floatFromInt(self.sam_arena_bytes)) / (1024.0 * 1024.0),
            self.append_us_per_tok,
            self.query_active_us,
            self.query_suffix_us,
            self.max_query_us,
            self.queries,
            self.seed_hits,
            self.mean_draft_len,
            self.heap_delta.blocks_in_use,
            self.heap_delta.size_in_use,
        });
    }
};

pub const SpecSynthResult = struct {
    mode: draft_mod.DraftMode,
    effective_placement: draft_mod.Placement,
    lossless_match: bool,
    generated_tokens: u32,
    rounds: u64,
    mean_accept_len: f64,
    mean_drafted_nodes: f64,
    mean_verified_nodes: f64,
    mean_unique_experts_per_layer: f64,
    bytes_per_accepted_tok: f64,
    heap_delta: vmstat.HeapDelta,
    swapouts_delta: u64,

    pub fn write(self: *const SpecSynthResult, out: *std.Io.Writer) std.Io.Writer.Error!void {
        try out.print(
            \\Synthetic End-to-End Speculative Decode Check (MoE q3 + EAGLE-3 + L1 + D7)
            \\spec_synth.mode                  {s}
            \\spec_synth.placement             {s}
            \\spec_synth.lossless_match        {s}
            \\spec_synth.generated_tokens      {d}
            \\spec_synth.rounds                {d}
            \\spec_synth.mean_accept_len       {d:.3}
            \\spec_synth.drafted_nodes         {d:.2}
            \\spec_synth.verified_nodes        {d:.2}
            \\spec_synth.unique_experts_layer  {d:.2}
            \\spec_synth.bytes_per_acc_tok     {d:.0}
            \\spec_synth.malloc_blocks_delta   {d}
            \\spec_synth.malloc_bytes_delta    {d}
            \\spec_synth.swapouts_delta        {d}
            \\
        , .{
            self.mode.label(),
            self.effective_placement.label(),
            if (self.lossless_match) "yes" else "NO",
            self.generated_tokens,
            self.rounds,
            self.mean_accept_len,
            self.mean_drafted_nodes,
            self.mean_verified_nodes,
            self.mean_unique_experts_per_layer,
            self.bytes_per_accepted_tok,
            self.heap_delta.blocks_in_use,
            self.heap_delta.size_in_use,
            self.swapouts_delta,
        });
    }
};

pub const M3Config = struct {
    s4: attention.SpikeS4Config = .{},
    s5: moe.SpikeS5Config = .{},
    sam_context_tokens: u32 = 32_768,
    sam_queries: u32 = 10_000,
    synth_gen_tokens: u32 = 12,
};

pub const M3Report = struct {
    s4: attention.SpikeS4Report,
    s5: moe.SpikeS5Report,
    sam: SamLatencyResult,
    synth_spec: SpecSynthResult,
    vm_before: vmstat.Snapshot,
    vm_after: vmstat.Snapshot,

    pub fn deinit(self: *M3Report, allocator: std.mem.Allocator) void {
        self.s5.deinit(allocator);
    }

    pub fn write(self: *const M3Report, out: *std.Io.Writer) std.Io.Writer.Error!void {
        try out.writeAll("\n=== M3 Speculative Decoding & Placement Gate Benchmarks ===\n\n");
        try self.s4.write(out);
        try out.writeAll("\n");
        try self.s5.write(out);
        try out.writeAll("\n");
        try self.sam.write(out);
        try out.writeAll("\n");
        try self.synth_spec.write(out);
        try out.print(
            \\
            \\M3 Swap Check
            \\m3.swap.status                   ok
            \\m3.swap.swapouts_delta           {d}
            \\m3.swap.pageouts_delta           {d}
            \\
        , .{
            vmstat.Snapshot.swapoutDelta(self.vm_before, self.vm_after),
            vmstat.Snapshot.pageoutDelta(self.vm_before, self.vm_after),
        });
    }
};

pub const M3Error = error{
    LosslessMismatch,
} || attention.SpikeS4Error || moe.SpikeS5Error || sam_mod.InitError || sam_mod.AppendError ||
    qwen3.LoadError || qwen3.StepError || draft_mod.DraftLoadError ||
    engine_mod.InitError || engine_mod.RunError || vmstat.Error;

/// Benchmarks zero-allocation L1 `SuffixAutomaton` build and query latency at
/// `context_tokens` (default 32,768 tokens, target `< 50 us` per query).
pub fn runSamLatencyBench(context_tokens: u32, queries: u32) M3Error!SamLatencyResult {
    const facts = try sysinfo.Facts.read();
    const device = try metal.Device.open();
    defer device.close();
    const dev_info = try device.info();
    const host_budget = budget_mod.Budget.fromHost(facts, dev_info, .{});

    const sam_bytes = try sam_mod.SuffixAutomaton.requiredBytes(context_tokens);
    const arena_bytes = try page.alignUp(sam_bytes);
    var sam_arena = try arena_mod.WiredArena.init(arena_bytes, host_budget);
    defer sam_arena.deinit();

    const backing = try sam_arena.allocAligned(sam_bytes, sam_mod.SuffixAutomaton.backing_alignment);
    var sam = try sam_mod.SuffixAutomaton.initFromBuffer(backing, context_tokens);

    // Build realistic code-like token stream: repeating 32-token blocks interspersed with unique tokens.
    const t_app0 = try vmstat.nowNs();
    var state: u64 = 0xD15E_A5E5_1234_5678;
    for (0..context_tokens) |idx| {
        const block_pos = idx & 63;
        const tok: u32 = if (block_pos < 24)
            100 + @as(u32, @intCast(block_pos)) + @as(u32, @intCast(((idx >> 6) & 3) * 32))
        else blk: {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            break :blk 1000 + @as(u32, @intCast(state % 50_000));
        };
        try sam.append(tok);
    }
    const t_app1 = try vmstat.nowNs();

    var out_buf: [16]u32 = undefined;
    var tail: [16]u32 = undefined;
    for (0..16) |k| tail[k] = 100 + @as(u32, @intCast(k));

    const heap_before = try vmstat.Snapshot.capture();
    var seed_hits: u32 = 0;
    var draft_len_sum: u64 = 0;
    var max_q_ns: u64 = 0;

    const t_act0 = try vmstat.nowNs();
    for (0..queries) |q_idx| {
        const q0 = try vmstat.nowNs();
        const r = sam.queryActive(16, &out_buf);
        const q1 = try vmstat.nowNs();
        if (q1 - q0 > max_q_ns) max_q_ns = q1 - q0;
        if (r.is_seed) seed_hits += 1;
        draft_len_sum += r.draft_len;
        std.mem.doNotOptimizeAway(out_buf[q_idx & 15]);
    }
    const t_act1 = try vmstat.nowNs();

    const t_suf0 = try vmstat.nowNs();
    for (0..queries) |q_idx| {
        tail[0] = 100 + @as(u32, @intCast((q_idx & 3) * 32));
        for (1..16) |k| tail[k] = tail[0] + @as(u32, @intCast(k));
        const q0 = try vmstat.nowNs();
        const r = sam.querySuffix(&tail, 16, &out_buf);
        const q1 = try vmstat.nowNs();
        if (q1 - q0 > max_q_ns) max_q_ns = q1 - q0;
        if (r.is_seed) seed_hits += 1;
        draft_len_sum += r.draft_len;
        std.mem.doNotOptimizeAway(out_buf[q_idx & 15]);
    }
    const t_suf1 = try vmstat.nowNs();
    const heap_after = try vmstat.Snapshot.capture();

    const total_q: f64 = @floatFromInt(@as(u64, queries) * 2);
    return .{
        .context_tokens = context_tokens,
        .queries = queries,
        .sam_arena_bytes = arena_bytes,
        .append_us_per_tok = (@as(f64, @floatFromInt(t_app1 - t_app0)) / @as(f64, @floatFromInt(context_tokens))) / 1000.0,
        .query_active_us = (@as(f64, @floatFromInt(t_act1 - t_act0)) / @as(f64, @floatFromInt(queries))) / 1000.0,
        .query_suffix_us = (@as(f64, @floatFromInt(t_suf1 - t_suf0)) / @as(f64, @floatFromInt(queries))) / 1000.0,
        .max_query_us = @as(f64, @floatFromInt(max_q_ns)) / 1000.0,
        .seed_hits = seed_hits,
        .mean_draft_len = @as(f64, @floatFromInt(draft_len_sum)) / total_q,
        .heap_delta = vmstat.heapDelta(heap_before, heap_after),
    };
}

/// Runs an in-memory synthetic MoE 3-bit + EAGLE-3 + L1 + D7 speculative decode
/// benchmark verifying 100% token identity vs `Qwen3Model.stepDecode` and zero swap.
pub fn runSpecSynthBench(allocator: std.mem.Allocator, gen_tokens: u32) M3Error!SpecSynthResult {
    const cfg: qwen3.ModelConfig = .{
        .arch = .qwen3_moe,
        .quant_bits = .q3,
        .vocab_size = 64,
        .hidden_size = 128,
        .num_hidden_layers = 2,
        .num_attention_heads = 2,
        .num_key_value_heads = 1,
        .head_dim = 64,
        .intermediate_size = 128,
        .moe_intermediate_size = 128,
        .num_experts = 4,
        .num_experts_per_tok = 2,
        .rms_norm_eps = 1e-6,
        .rope_theta = 1_000_000.0,
        .max_position_embeddings = 64,
        .tie_word_embeddings = false,
        .norm_topk_prob = true,
    };
    const model_bytes = try qwen3.buildSyntheticAmfImageWithSpec(allocator, cfg, .{
        .linear_dtype = .q3_affine,
        .embed_dtype = .f16,
        .norm_dtype = .f16,
        .router_dtype = .f16,
        .expert_dtype = .q3_affine,
    }, 0x0003_5EC0);
    defer allocator.free(model_bytes);

    const dcfg: draft_mod.DraftConfig = .{
        .format = .eagle3,
        .vocab_size = cfg.vocab_size,
        .draft_vocab_size = 32,
        .hidden_size = cfg.hidden_size,
        .target_hidden_size = cfg.hidden_size,
        .num_hidden_layers = 1,
        .num_attention_heads = cfg.num_attention_heads,
        .num_key_value_heads = cfg.num_key_value_heads,
        .head_dim = cfg.head_dim,
        .intermediate_size = 128,
        .rms_norm_eps = cfg.rms_norm_eps,
        .rope_theta = 10_000.0,
        .max_position_embeddings = 64,
        .norm_before_residual = true,
        .aux_layer_ids = qwen3.defaultEagle3AuxLayerIds(cfg.num_hidden_layers),
    };
    const draft_bytes = try draft_mod.buildSyntheticDraftAmfImage(allocator, dcfg, 0x0003_E300);
    defer allocator.free(draft_bytes);

    const prompt = [_]u32{ 3, 7, 11, 15, 19, 3, 7, 11, 15 };
    const n_gen: usize = @min(@as(usize, gen_tokens), 32);

    // Reference sequential greedy decode.
    var ref_model = try qwen3.Qwen3Model.loadFromBytes(allocator, model_bytes, cfg, .{
        .max_seq_len = 64,
        .chunk_tokens = 16,
        .verify_crcs = true,
    });
    defer ref_model.deinit(allocator);

    var ref_out: [32]u32 = undefined;
    var pos: u32 = 0;
    for (prompt[0 .. prompt.len - 1]) |tok| {
        try ref_model.prefillToken(tok, pos);
        pos += 1;
    }
    var cur = try ref_model.stepDecode(prompt[prompt.len - 1], pos);
    pos += 1;
    ref_out[0] = cur;
    for (1..n_gen) |step| {
        cur = try ref_model.stepDecode(cur, pos);
        pos += 1;
        ref_out[step] = cur;
    }

    // Speculative decode (`EAGLE-3` + `L1` + `D7` + `.auto` placement).
    var spec_model = try qwen3.Qwen3Model.loadFromBytes(allocator, model_bytes, cfg, .{
        .max_seq_len = 64,
        .chunk_tokens = 16,
        .max_tree_nodes = 16,
        .enable_feature_taps = true,
        .verify_crcs = true,
    });
    defer spec_model.deinit(allocator);

    var dh = try draft_mod.DraftHead.loadFromBytes(allocator, &spec_model, draft_bytes, dcfg, .{
        .placement = .auto,
        .max_seq_len = 64,
        .chunk_tokens = 16,
        .max_tree_slots = 16,
    });
    defer dh.deinit(allocator, &spec_model);

    var eng = try engine_mod.SpecEngine.init(allocator, &spec_model, &dh, .{
        .mode = .eagle3,
        .placement = .auto,
        .max_tree_nodes = 12,
        .max_draft_depth = 4,
        .branch_width = 2,
        .top_k_per_node = 3,
        .temperature = 0.0,
        .enable_pruning = true,
    });
    defer eng.deinit(allocator);

    const vm_before = try vmstat.Snapshot.capture();
    var spec_out: [32]u32 = undefined;
    spec_out[0] = try eng.prefill(&prompt, null);

    const heap_before = try vmstat.Snapshot.capture();
    var produced: usize = 1;
    while (produced < n_gen) {
        const rs = try eng.stepRound(null, spec_out[produced..n_gen]);
        produced += rs.emitted_count;
    }
    const heap_after = try vmstat.Snapshot.capture();
    const vm_after = try vmstat.Snapshot.capture();
    try vmstat.checkNoSwap(vm_before, vm_after);

    const match = std.mem.eql(u32, ref_out[0..n_gen], spec_out[0..n_gen]);
    if (!match) return error.LosslessMismatch;

    return .{
        .mode = .eagle3,
        .effective_placement = eng.effectivePlacement(),
        .lossless_match = match,
        .generated_tokens = @intCast(n_gen),
        .rounds = eng.stats.rounds,
        .mean_accept_len = eng.stats.meanAcceptanceLength(),
        .mean_drafted_nodes = eng.stats.meanDraftedNodes(),
        .mean_verified_nodes = eng.stats.meanVerifiedNodes(),
        .mean_unique_experts_per_layer = eng.stats.meanUniqueExpertsPerLayer(),
        .bytes_per_accepted_tok = eng.stats.bytesPerAcceptedToken(),
        .heap_delta = vmstat.heapDelta(heap_before, heap_after),
        .swapouts_delta = vmstat.Snapshot.swapoutDelta(vm_before, vm_after),
    };
}

pub fn runM3WithConfig(allocator: std.mem.Allocator, config: M3Config) M3Error!M3Report {
    const vm_before = try vmstat.Snapshot.capture();
    const s4 = try attention.runSpikeS4(allocator, config.s4);
    var s5 = try moe.runSpikeS5(allocator, config.s5);
    errdefer s5.deinit(allocator);
    const sam_res = try runSamLatencyBench(config.sam_context_tokens, config.sam_queries);
    const synth_res = try runSpecSynthBench(allocator, config.synth_gen_tokens);
    const vm_after = try vmstat.Snapshot.capture();
    try vmstat.checkNoSwap(vm_before, vm_after);

    return .{
        .s4 = s4,
        .s5 = s5,
        .sam = sam_res,
        .synth_spec = synth_res,
        .vm_before = vm_before,
        .vm_after = vm_after,
    };
}

pub fn runM3(allocator: std.mem.Allocator) M3Error!M3Report {
    return runM3WithConfig(allocator, .{});
}

test "bytesMovedPerDispatch matches exact algorithmic traffic and validates shapes" {
    // q4 (2048, 2048), M=1: w = 2048*1024 = 2097152, sb = 2048*32*4 = 262144, x = 4096, y = 4096
    const q4_gemv = try bytesMovedPerDispatch(.{ .kind = .gemv_q4, .m = 1, .n = 2048, .k = 2048 });
    try std.testing.expectEqual(@as(u64, 2097152 + 262144 + 4096 + 4096), q4_gemv);

    // q3 (2048, 768), M=1: w = 2048*288 = 589824, sb = 2048*12*4 = 98304, x = 1536, y = 4096
    const q3_gemv = try bytesMovedPerDispatch(.{ .kind = .gemv_q3, .m = 1, .n = 2048, .k = 768 });
    try std.testing.expectEqual(@as(u64, 589824 + 98304 + 1536 + 4096), q3_gemv);

    // q4 (4096, 4096), M=16: w = 8388608, sb = 1048576, x = 16*4096*2 = 131072, y = 16*4096*2 = 131072
    const q4_gemm = try bytesMovedPerDispatch(.{ .kind = .gemm_q4, .m = 16, .n = 4096, .k = 4096 });
    try std.testing.expectEqual(@as(u64, 8388608 + 1048576 + 131072 + 131072), q4_gemm);

    try std.testing.expectError(error.InvalidConfig, bytesMovedPerDispatch(.{ .kind = .gemv_q4, .m = 2, .n = 64, .k = 64 }));
    try std.testing.expectError(error.ZeroDimension, bytesMovedPerDispatch(.{ .kind = .gemm_q3, .m = 0, .n = 64, .k = 64 }));
    try std.testing.expectError(error.UnalignedK, bytesMovedPerDispatch(.{ .kind = .gemv_q3, .m = 1, .n = 64, .k = 65 }));
}

test "CaseResult and S2Result compute decimal GB/s, bus %, and encode latencies accurately" {
    const r = CaseResult{
        .case = .{ .kind = .gemv_q4, .m = 1, .n = 2048, .k = 2048 },
        .bytes_per_dispatch = 2_400_000,
        .dispatches = 100,
        .weight_slots = 8,
        // 240,000,000 bytes in 2,500,000 ns => 96.0 GB/s (80.0% of 120 GB/s), 25.0 us/dispatch
        .elapsed_ns = 2_500_000,
    };
    try std.testing.expectApproxEqAbs(@as(f64, 25.0), r.latencyUs(), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 96.0), r.effectiveBandwidthGbs(), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 80.0), r.busUtilizationPct(), 1e-6);

    const zero_snap = std.mem.zeroes(vmstat.Snapshot);
    const s2 = S2Result{
        .rounds = 1000,
        .dispatches_per_round = 32,
        .encode_ns = 16_000_000, // 16 us/round, 500 ns/dispatch
        .total_ns = 80_000_000, // 80 us/round
        .before = zero_snap,
        .after = zero_snap,
        .heap_delta = std.mem.zeroes(vmstat.HeapDelta),
    };
    try std.testing.expectApproxEqAbs(@as(f64, 16.0), s2.cpuEncodeUsPerRound(), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 500.0), s2.cpuEncodeNsPerDispatch(), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 80.0), s2.fullRoundUs(), 1e-6);
}

test "DEFAULT_CASES covers all 16 required M1 GEMV and GEMM calibration shapes" {
    try std.testing.expectEqual(@as(usize, 16), DEFAULT_CASES.len);
    for (DEFAULT_CASES) |case| {
        const bytes = try bytesMovedPerDispatch(case);
        try std.testing.expect(bytes > 0);
    }
}

test "runner smoke test exercises C9 kernels, Spike S2 ring, WiredArena and swap guard" {
    const gpa = std.testing.allocator;
    const smoke_cases = [_]Case{
        .{ .kind = .gemv_q4, .m = 1, .n = 64, .k = 128 },
        .{ .kind = .gemv_q3, .m = 1, .n = 64, .k = 128 },
        .{ .kind = .gemm_q4, .m = 16, .n = 64, .k = 128 },
        .{ .kind = .gemm_q3, .m = 16, .n = 64, .k = 128 },
    };

    var report = try run(gpa, .{
        .cases = &smoke_cases,
        .min_working_set_bytes = 0,
        .dvfs_warmup_batches = 0,
        .warmup_batches = 1,
        .gemv_timed_batches = 1,
        .gemv_dispatches_per_batch = 4,
        .gemm_timed_batches = 1,
        .gemm_dispatches_per_batch = 2,
        .s2_warmup_rounds = 4,
        .s2_rounds = 16,
        .s2_dispatches_per_round = 8,
        .timeout_ms = 5000,
    });
    defer report.deinit(gpa);

    try std.testing.expect(report.gpuName().len > 0);
    try std.testing.expect(report.arena_bytes > 0);
    try std.testing.expect(report.arena_bytes <= report.wired_budget_bytes);
    try std.testing.expectEqual(@as(usize, smoke_cases.len), report.results.len);
    for (report.results) |res| {
        try std.testing.expect(res.elapsed_ns > 0);
        try std.testing.expect(res.latencyUs() > 0.0);
        try std.testing.expect(res.effectiveBandwidthGbs() > 0.0);
        try std.testing.expect(res.weight_slots >= 1);
    }

    try std.testing.expectEqual(@as(u32, 16), report.s2.rounds);
    try std.testing.expectEqual(@as(u32, 8), report.s2.dispatches_per_round);
    try std.testing.expect(report.s2.encode_ns > 0);
    try std.testing.expect(report.s2.total_ns >= report.s2.encode_ns);
    try vmstat.checkNoSwap(report.vm_before, report.vm_after);

    var aw: std.Io.Writer.Allocating = .init(gpa);
    defer aw.deinit();
    try report.write(&aw.writer);
    const rendered = aw.written();
    try std.testing.expect(std.mem.indexOf(u8, rendered, "gemv_q4") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "gemm_q3") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "s2.rounds") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "swap.status") != null);
}

test "runMoeBench verifies G3 M=1 and M=8 deduplicated expert bytes and bandwidth" {
    const gpa = std.testing.allocator;
    const cases = [_]MoeBenchCase{
        .{ .m = 1, .bits = .q3, .hidden_size = 128, .intermediate_size = 128, .num_experts = 16, .top_k = 4, .shared_pool = 4 },
        .{ .m = 8, .bits = .q3, .hidden_size = 128, .intermediate_size = 128, .num_experts = 16, .top_k = 4, .shared_pool = 6 },
    };
    var report = try runMoeBench(gpa, .{
        .cases = &cases,
        .dvfs_warmup_batches = 0,
        .warmup_batches = 1,
        .timed_batches = 2,
        .dispatches_per_batch = 2,
        .timeout_ms = 5000,
    });
    defer report.deinit(gpa);

    try std.testing.expectEqual(@as(usize, 2), report.results.len);
    const m1 = report.results[0];
    const m8 = report.results[1];
    try std.testing.expectEqual(@as(u32, 4), m1.unique_experts);
    try std.testing.expect(m8.unique_experts <= 6);
    try std.testing.expect(m8.unique_experts < m8.case.m * m8.case.top_k);
    try std.testing.expect(m8.dedup_bytes < m8.naive_bytes);
    try std.testing.expect(m1.effectiveBandwidthGbs() > 0.0);
    try std.testing.expect(m8.effectiveBandwidthGbs() > 0.0);
    try vmstat.checkNoSwap(report.vm_before, report.vm_after);

    var aw: std.Io.Writer.Allocating = .init(gpa);
    defer aw.deinit();
    try report.write(&aw.writer);
    try std.testing.expect(std.mem.indexOf(u8, aw.written(), "G3 MoE Deduplicated") != null);
}

test "runDecodeBenchFromFile measures real .amf greedy decode and linear layer scaling" {
    const gpa = std.testing.allocator;
    const cfg: qwen3.ModelConfig = .{
        .arch = .qwen3_moe,
        .quant_bits = .q3,
        .vocab_size = 256,
        .hidden_size = 128,
        .num_hidden_layers = 2,
        .num_attention_heads = 4,
        .num_key_value_heads = 2,
        .head_dim = 64,
        .moe_intermediate_size = 128,
        .num_experts = 8,
        .num_experts_per_tok = 2,
        .norm_topk_prob = true,
    };

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const raw_path = try std.fmt.bufPrint(path_buf[0 .. path_buf.len - 1], ".zig-cache/tmp/{s}/tiny_moe.amf", .{tmp.sub_path});
    path_buf[raw_path.len] = 0;
    const amf_path: [:0]const u8 = path_buf[0..raw_path.len :0];

    try writeSyntheticQwen3AmfFile(gpa, amf_path, cfg, 42);

    const res = try runDecodeBenchFromFile(gpa, amf_path, cfg, .{
        .label = "tiny-moe-3bit",
        .scale_layers = 48,
        .prefill_tokens = 4,
        .warmup_steps = 1,
        .decode_steps = 4,
        .max_seq_len = 32,
        .verify_crcs = true,
    });
    try std.testing.expectEqual(@as(u32, 2), res.measured_layers);
    try std.testing.expectEqual(@as(u32, 48), res.scale_layers);
    try std.testing.expect(res.measuredMsPerTok() > 0.0);
    try std.testing.expect(res.extrapolatedMsPerTok() > res.measuredMsPerTok());
    try std.testing.expect(res.measured_bytes_per_tok > 0);
    try std.testing.expect(res.extrapolated_bytes_per_tok > res.measured_bytes_per_tok);
    try std.testing.expectEqual(@as(u64, 0), res.swapouts_delta);
}

test "denseBytesMovedPerDispatch and DEFAULT_M25_CASES cover all 16 FP16/BF16 shapes" {
    try std.testing.expectEqual(@as(usize, 16), DEFAULT_M25_CASES.len);
    // f16 (2048, 2048), M=1: w=2048*2048*2=8388608, x=4096, y=4096
    const f16_gemv = try denseBytesMovedPerDispatch(.{ .kind = .gemv_f16, .m = 1, .n = 2048, .k = 2048 });
    try std.testing.expectEqual(@as(u64, 8_388_608 + 4096 + 4096), f16_gemv);

    // bf16 (4096, 4096), M=16: w=33554432, x=131072, y=131072
    const bf16_gemm = try denseBytesMovedPerDispatch(.{ .kind = .gemm_bf16, .m = 16, .n = 4096, .k = 4096 });
    try std.testing.expectEqual(@as(u64, 33_554_432 + 131_072 + 131_072), bf16_gemm);

    for (DEFAULT_M25_CASES) |case| {
        const bytes = try denseBytesMovedPerDispatch(case);
        try std.testing.expect(bytes > 0);
    }
    try std.testing.expectError(error.InvalidConfig, denseBytesMovedPerDispatch(.{ .kind = .gemv_bf16, .m = 2, .n = 64, .k = 64 }));
    try std.testing.expectError(error.ZeroDimension, denseBytesMovedPerDispatch(.{ .kind = .gemm_f16, .m = 0, .n = 64, .k = 64 }));
}

test "runDenseBench and synthetic BF16 decode smoke test exercise M2.5 pipelines without swap" {
    const gpa = std.testing.allocator;
    const smoke_cases = [_]DenseCase{
        .{ .kind = .gemv_f16, .m = 1, .n = 64, .k = 128 },
        .{ .kind = .gemv_bf16, .m = 1, .n = 64, .k = 128 },
        .{ .kind = .gemm_f16, .m = 16, .n = 64, .k = 128 },
        .{ .kind = .gemm_bf16, .m = 16, .n = 64, .k = 128 },
    };
    var report = try runDenseBench(gpa, .{
        .cases = &smoke_cases,
        .min_working_set_bytes = 0,
        .dvfs_warmup_batches = 0,
        .warmup_batches = 1,
        .gemv_timed_batches = 1,
        .gemv_dispatches_per_batch = 4,
        .gemm_timed_batches = 1,
        .gemm_dispatches_per_batch = 2,
        .timeout_ms = 5000,
    });
    defer report.deinit(gpa);

    try std.testing.expectEqual(@as(usize, smoke_cases.len), report.results.len);
    for (report.results) |res| {
        try std.testing.expect(res.elapsed_ns > 0);
        try std.testing.expect(res.effectiveBandwidthGbs() > 0.0);
        try std.testing.expect(res.effectiveTflops() > 0.0);
    }
    try vmstat.checkNoSwap(report.vm_before, report.vm_after);

    var aw: std.Io.Writer.Allocating = .init(gpa);
    defer aw.deinit();
    try report.write(&aw.writer);
    try std.testing.expect(std.mem.indexOf(u8, aw.written(), "gemv_bf16") != null);
    try std.testing.expect(std.mem.indexOf(u8, aw.written(), "gemm_f16") != null);

    // Synthetic BF16 tied-embedding Qwen3 decode bench round-trip
    const bf16_cfg: qwen3.ModelConfig = .{
        .arch = .qwen3_dense,
        .quant_bits = null,
        .default_float_dtype = .bf16,
        .vocab_size = 256,
        .hidden_size = 128,
        .num_hidden_layers = 2,
        .num_attention_heads = 4,
        .num_key_value_heads = 2,
        .head_dim = 64,
        .intermediate_size = 384,
        .tie_word_embeddings = true,
    };
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const raw_path = try std.fmt.bufPrint(path_buf[0 .. path_buf.len - 1], ".zig-cache/tmp/{s}/tiny_bf16.amf", .{tmp.sub_path});
    path_buf[raw_path.len] = 0;
    const amf_path: [:0]const u8 = path_buf[0..raw_path.len :0];

    try writeSyntheticQwen3AmfFile(gpa, amf_path, bf16_cfg, 600);
    const dec = try runDecodeBenchFromFile(gpa, amf_path, bf16_cfg, .{
        .label = "tiny-bf16",
        .scale_layers = 28,
        .prefill_tokens = 4,
        .warmup_steps = 1,
        .decode_steps = 4,
        .max_seq_len = 32,
        .verify_crcs = true,
    });
    try std.testing.expectEqualStrings("bf16", dec.precisionLabel());
    try std.testing.expect(dec.measuredMsPerTok() > 0.0);
    try std.testing.expectEqual(@as(u64, 0), dec.swapouts_delta);
}

test "runM3WithConfig exercises Spike S4, Spike S5, L1 SAM latency, and synthetic E2E spec decode" {
    const gpa = std.testing.allocator;
    const s5_ms = [_]u32{ 1, 4 };
    var report = try runM3WithConfig(gpa, .{
        .s4 = .{
            .handoff_c = 64,
            .handoff_s = 32,
            .handoff_warmup_iters = 1,
            .handoff_timed_iters = 3,
            .draft_hidden = 128,
            .draft_layers = 1,
            .draft_batch = 32,
            .draft_warmup_iters = 1,
            .draft_timed_iters = 3,
            .gpu_dispatches_per_batch = 2,
            .stream_hidden = 128,
            .stream_layers = 1,
            .stream_spatial = 32,
            .gpu_gemv_n = 128,
            .gpu_gemv_k = 128,
            .min_working_set_bytes = 0,
            .contention_warmup_batches = 1,
            .contention_timed_batches = 2,
            .contention_dispatches_per_batch = 2,
            .timeout_ms = 10_000,
        },
        .s5 = .{
            .bits = .q3,
            .num_experts = 16,
            .top_k = 4,
            .hidden_size = 128,
            .intermediate_size = 128,
            .m_values = &s5_ms,
            .num_routing_banks = 4,
            .min_working_set_bytes = 0,
            .dvfs_warmup_batches = 0,
            .warmup_batches = 1,
            .timed_batches = 2,
            .dispatches_per_batch = 2,
            .timeout_ms = 10_000,
        },
        .sam_context_tokens = 2048,
        .sam_queries = 500,
        .synth_gen_tokens = 6,
    });
    defer report.deinit(gpa);

    try std.testing.expect(report.sam.query_active_us < 50.0);
    try std.testing.expect(report.sam.seed_hits > 0);
    try std.testing.expect(report.synth_spec.lossless_match);
    try std.testing.expectEqual(@as(u64, 0), report.synth_spec.swapouts_delta);

    var aw: std.Io.Writer.Allocating = .init(gpa);
    defer aw.deinit();
    try report.write(&aw.writer);
    const rendered = aw.written();
    try std.testing.expect(std.mem.indexOf(u8, rendered, "Spike S4 CoreML ANE") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "Spike S5 Unique Active Experts") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "L1 Suffix Automaton Latency") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "spec_synth.lossless_match") != null);
}
