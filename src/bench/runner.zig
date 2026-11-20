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
    const q_dtype: amf.DType = if (cfg.quant_bits == .q3) .q3_affine else .q4_affine;
    const bits_u8: u8 = cfg.quant_bits.bitWidth();
    const inv_sqrt_h = 1.0 / @sqrt(@as(f32, @floatFromInt(h)));
    const inv_sqrt_q = 1.0 / @sqrt(@as(f32, @floatFromInt(q_dim)));
    const base_mul: f32 = if (bits_u8 == 3) 0.35 else 0.17;

    var plans: std.ArrayList(StreamTensorPlan) = .empty;
    defer plans.deinit(allocator);

    try plans.append(allocator, makePlan(
        "embed_tokens",
        .embedding,
        q_dtype,
        bits_u8,
        64,
        amf.NO_LAYER,
        amf.NO_EXPERT,
        .{ cfg.vocab_size, h, 0, 0 },
        2,
        if (bits_u8 == 3) 0.06 else 0.03,
    ));

    var name_buf: [64]u8 = undefined;
    for (0..cfg.num_hidden_layers) |l_usize| {
        const l: u16 = @intCast(l_usize);
        try plans.append(allocator, makePlan(std.fmt.bufPrint(&name_buf, "layers.{d}.input_layernorm.weight", .{l}) catch unreachable, .norm, .f16, 0, 0, l, amf.NO_EXPERT, .{ h, 0, 0, 0 }, 1, 1.0));
        try plans.append(allocator, makePlan(std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.q_proj", .{l}) catch unreachable, .dense, q_dtype, bits_u8, 64, l, amf.NO_EXPERT, .{ q_dim, h, 0, 0 }, 2, base_mul * inv_sqrt_h));
        try plans.append(allocator, makePlan(std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.k_proj", .{l}) catch unreachable, .dense, q_dtype, bits_u8, 64, l, amf.NO_EXPERT, .{ kv_dim, h, 0, 0 }, 2, base_mul * inv_sqrt_h));
        try plans.append(allocator, makePlan(std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.v_proj", .{l}) catch unreachable, .dense, q_dtype, bits_u8, 64, l, amf.NO_EXPERT, .{ kv_dim, h, 0, 0 }, 2, base_mul * inv_sqrt_h));
        try plans.append(allocator, makePlan(std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.o_proj", .{l}) catch unreachable, .dense, q_dtype, bits_u8, 64, l, amf.NO_EXPERT, .{ h, q_dim, 0, 0 }, 2, base_mul * inv_sqrt_q));
        try plans.append(allocator, makePlan(std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.q_norm.weight", .{l}) catch unreachable, .norm, .f16, 0, 0, l, amf.NO_EXPERT, .{ cfg.head_dim, 0, 0, 0 }, 1, 1.0));
        try plans.append(allocator, makePlan(std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.k_norm.weight", .{l}) catch unreachable, .norm, .f16, 0, 0, l, amf.NO_EXPERT, .{ cfg.head_dim, 0, 0, 0 }, 1, 1.0));
        try plans.append(allocator, makePlan(std.fmt.bufPrint(&name_buf, "layers.{d}.post_attention_layernorm.weight", .{l}) catch unreachable, .norm, .f16, 0, 0, l, amf.NO_EXPERT, .{ h, 0, 0, 0 }, 1, 1.0));

        if (cfg.arch == .qwen3_moe) {
            try plans.append(allocator, makePlan(std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.gate.weight", .{l}) catch unreachable, .router, .f16, 0, 0, l, amf.NO_EXPERT, .{ cfg.num_experts, h, 0, 0 }, 2, 0.5 * inv_sqrt_h));
            const inv_sqrt_mi = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.moe_intermediate_size)));
            for (0..cfg.num_experts) |e_usize| {
                const e: u16 = @intCast(e_usize);
                try plans.append(allocator, makePlan(std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.experts.{d}.gate_proj", .{ l, e }) catch unreachable, .hot_expert, q_dtype, bits_u8, 64, l, e, .{ cfg.moe_intermediate_size, h, 0, 0 }, 2, base_mul * inv_sqrt_h));
                try plans.append(allocator, makePlan(std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.experts.{d}.up_proj", .{ l, e }) catch unreachable, .hot_expert, q_dtype, bits_u8, 64, l, e, .{ cfg.moe_intermediate_size, h, 0, 0 }, 2, base_mul * inv_sqrt_h));
                try plans.append(allocator, makePlan(std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.experts.{d}.down_proj", .{ l, e }) catch unreachable, .hot_expert, q_dtype, bits_u8, 64, l, e, .{ h, cfg.moe_intermediate_size, 0, 0 }, 2, base_mul * inv_sqrt_mi));
            }
        } else {
            const inv_sqrt_di = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.intermediate_size)));
            try plans.append(allocator, makePlan(std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.gate_proj", .{l}) catch unreachable, .dense, q_dtype, bits_u8, 64, l, amf.NO_EXPERT, .{ cfg.intermediate_size, h, 0, 0 }, 2, base_mul * inv_sqrt_h));
            try plans.append(allocator, makePlan(std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.up_proj", .{l}) catch unreachable, .dense, q_dtype, bits_u8, 64, l, amf.NO_EXPERT, .{ cfg.intermediate_size, h, 0, 0 }, 2, base_mul * inv_sqrt_h));
            try plans.append(allocator, makePlan(std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.down_proj", .{l}) catch unreachable, .dense, q_dtype, bits_u8, 64, l, amf.NO_EXPERT, .{ h, cfg.intermediate_size, 0, 0 }, 2, base_mul * inv_sqrt_di));
        }
    }

    try plans.append(allocator, makePlan("norm.weight", .norm, .f16, 0, 0, amf.NO_LAYER, amf.NO_EXPERT, .{ h, 0, 0, 0 }, 1, 1.0));
    if (!cfg.tie_word_embeddings) {
        try plans.append(allocator, makePlan("lm_head", .lm_head, q_dtype, bits_u8, 64, amf.NO_LAYER, amf.NO_EXPERT, .{ cfg.vocab_size, h, 0, 0 }, 2, 0.45 * base_mul * inv_sqrt_h));
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
            data_bytes = elems * @sizeOf(f16);
            blob_bytes = try page.alignUp(@intCast(data_bytes));

            var f16_table: [257]f16 = undefined;
            for (&f16_table) |*v| {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                const u = @as(f32, @floatFromInt(rng & 0xFFFF)) / 65535.0;
                const f = if (p.kind == .norm) (0.95 + 0.1 * u) else ((u * 2.0 - 1.0) * p.scale_hint);
                v.* = @floatCast(f);
            }
            const f16_chunk: []f16 = @alignCast(std.mem.bytesAsSlice(f16, chunk_buf));
            var rem_elems = elems;
            var phase: usize = 0;
            while (rem_elems > 0) {
                const take_elems = @min(rem_elems, f16_chunk.len);
                for (0..take_elems) |i| {
                    f16_chunk[i] = f16_table[phase];
                    phase += 1;
                    if (phase == f16_table.len) phase = 0;
                }
                const bytes_slice = std.mem.sliceAsBytes(f16_chunk[0..take_elems]);
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
    quant_bits: quant.QuantBits,
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
            \\decode_bench.arch                {s} (q{d})
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
            self.quant_bits.bitWidth(),
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
    var moe_path_buf: [128]u8 = undefined;
    const moe_raw = std.fmt.bufPrint(moe_path_buf[0 .. moe_path_buf.len - 1], "/tmp/aminfer-m2-moe-{d}.amf", .{pid}) catch unreachable;
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

    var dense_path_buf: [128]u8 = undefined;
    const dense_raw = std.fmt.bufPrint(dense_path_buf[0 .. dense_path_buf.len - 1], "/tmp/aminfer-m2-dense-{d}.amf", .{pid}) catch unreachable;
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
