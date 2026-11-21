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
const aio = @import("../storage/aio.zig");
const wavefront = @import("../ane/wavefront.zig");
const predictor_mod = @import("../ane/predictor.zig");
const sme = @import("../cpu/sme.zig");
const qos = @import("../platform/qos.zig");
const ngram_nvme = @import("../spec/ngram_nvme.zig");
const server_mod = @import("../server/http_sse.zig");
const hybrid = @import("../gpu/hybrid.zig");
const ple = @import("../storage/ple.zig");
const graph = @import("../model/graph.zig");

/// Apple M4 unified memory bus peak bandwidth in decimal GB/s
/// (docs/architecture/hardware.md).
pub const M4_BUS_BANDWIDTH_GBS: f64 = 120.0;

/// Maximum uniform-bearing dispatches per Metal 4 frame (`AMI_UNIFORM_SLOTS_PER_FRAME`).
const MAX_DISPATCHES_PER_FRAME: u32 = 256;

const Q4 = quant.QuantOps(.{ .bits = .q4, .group_size = 64 });
const Q3 = quant.QuantOps(.{ .bits = .q3, .group_size = 64 });
const Q2 = quant.QuantOps(.{ .bits = .q2, .group_size = 64 });

pub const KernelKind = enum {
    gemv_q4,
    gemv_q3,
    gemv_q2,
    gemm_q4,
    gemm_q3,
    gemm_q2,

    pub fn bits(self: KernelKind) quant.QuantBits {
        return switch (self) {
            .gemv_q4, .gemm_q4 => .q4,
            .gemv_q3, .gemm_q3 => .q3,
            .gemv_q2, .gemm_q2 => .q2,
        };
    }

    pub fn isGemv(self: KernelKind) bool {
        return switch (self) {
            .gemv_q4, .gemv_q3, .gemv_q2 => true,
            .gemm_q4, .gemm_q3, .gemm_q2 => false,
        };
    }

    pub fn label(self: KernelKind) []const u8 {
        return switch (self) {
            .gemv_q4 => "gemv_q4",
            .gemv_q3 => "gemv_q3",
            .gemv_q2 => "gemv_q2",
            .gemm_q4 => "gemm_q4",
            .gemm_q3 => "gemm_q3",
            .gemm_q2 => "gemm_q2",
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
        .q2 => try Q2.packedWeightBytes(case.n, case.k),
    };
    const sb_bytes = switch (case.kind.bits()) {
        .q4 => try Q4.scaleBiasBytes(case.n, case.k),
        .q3 => try Q3.scaleBiasBytes(case.n, case.k),
        .q2 => try Q2.scaleBiasBytes(case.n, case.k),
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
            .q2 => .{
                try Q2.packedWeightBytes(case.intermediate_size, case.hidden_size),
                try Q2.scaleBiasBytes(case.intermediate_size, case.hidden_size),
                try Q2.packedWeightBytes(case.hidden_size, case.intermediate_size),
                try Q2.scaleBiasBytes(case.hidden_size, case.intermediate_size),
            },
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
        switch (qb) {
            .q2 => .q2_affine,
            .q3 => .q3_affine,
            .q4 => .q4_affine,
        }
    else
        (if (cfg.default_float_dtype == .f16) .f16 else .bf16);
    const norm_dtype: amf.DType = if (is_quant) .f16 else lin_dtype;
    const bits_u8: u8 = if (cfg.quant_bits) |qb| qb.bitWidth() else 0;
    const gs_u16: u16 = if (is_quant) 64 else 0;
    const inv_sqrt_h = 1.0 / @sqrt(@as(f32, @floatFromInt(h)));
    const inv_sqrt_q = 1.0 / @sqrt(@as(f32, @floatFromInt(q_dim)));
    const base_mul: f32 = switch (lin_dtype) {
        .q2_affine => 0.65,
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
            .q2_affine => 0.12,
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

        if (p.dtype == .q2_affine or p.dtype == .q3_affine or p.dtype == .q4_affine) {
            const n: usize = p.shape[0];
            const k: usize = p.shape[1];
            data_bytes = switch (p.dtype) {
                .q2_affine => try Q2.packedWeightBytes(n, k),
                .q3_affine => try Q3.packedWeightBytes(n, k),
                .q4_affine => try Q4.packedWeightBytes(n, k),
                else => unreachable,
            };
            params_bytes = switch (p.dtype) {
                .q2_affine => try Q2.scaleBiasBytes(n, k),
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
            const mid: f32 = switch (p.dtype) {
                .q2_affine => 1.5,
                .q3_affine => 3.5,
                .q4_affine => 7.5,
                else => unreachable,
            };
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
    budget_override: ?budget_mod.Budget = null,
    cold_experts_per_layer: u32 = 0,
    heat_profile_json: ?[]const u8 = null,
    cold_slot_count: u32 = 64,
    io_backend: aio.IoBackend = .auto,
    enable_ane_predictor: bool = true,
    ane_predictor_path: ?[:0]const u8 = null,
    prefill_mode: qwen3.PrefillMode = .auto,
    enable_hybrid_prefill: bool = false,
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
    cold_stats: ?qwen3.ColdExpertStats = null,
    heap_delta: vmstat.HeapDelta,
    swapouts_delta: u64,
    pageouts_delta: u64,

    pub fn label(self: *const DecodeBenchResult) []const u8 {
        return self.label_buf[0..self.label_len];
    }

    pub fn precisionLabel(self: DecodeBenchResult) []const u8 {
        if (self.quant_bits) |qb| {
            return switch (qb) {
                .q2 => "q2",
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
        if (self.cold_stats) |cs| {
            const ct = cs.cache_telemetry;
            try out.print(
                \\decode_bench.cold_experts        {d} (slots={d}, backend={s}, ane_pred={s})
                \\decode_bench.cold_hits           hot={d} cache={d} prefetch={d} sync_miss={d}
                \\decode_bench.cold_hit_rates      prefetch={d:.3} cold_total={d:.3} bytes_read={d} evictions={d}
                \\
            , .{
                cs.total_cold_experts,
                cs.slot_count,
                cs.io_backend.label(),
                if (cs.ane_predictor_active) "on" else "off",
                ct.hot_hits,
                ct.cache_hits,
                ct.prefetch_hits,
                ct.sync_misses,
                ct.prefetchHitRate(),
                ct.coldHitRate(),
                ct.bytes_read,
                ct.evictions,
            });
        }
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
        .budget_override = opts.budget_override,
        .cold_experts_per_layer = opts.cold_experts_per_layer,
        .heat_profile_json = opts.heat_profile_json,
        .cold_slot_count = opts.cold_slot_count,
        .io_backend = opts.io_backend,
        .enable_ane_predictor = opts.enable_ane_predictor,
        .ane_predictor_path = opts.ane_predictor_path,
        .prefill_mode = opts.prefill_mode,
        .enable_hybrid_prefill = opts.enable_hybrid_prefill,
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

    model.resetColdTelemetry();

    // Timed steady-state full decode steps (`measured_layers` + final norm + lm_head + argmax).
    const heap_before = try vmstat.Snapshot.capture();
    const t_full_0 = try vmstat.nowNs();
    for (0..opts.decode_steps) |_| {
        cur_tok = try model.stepDecode(cur_tok, eval_pos);
    }
    const t_full_1 = try vmstat.nowNs();
    const heap_after = try vmstat.Snapshot.capture();
    const timed_cold_stats = model.coldStats();

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
        .cold_stats = timed_cold_stats,
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

// ---------------------------------------------------------------------------
// M4 Hybrid ANE/GPU Wavefront Prefill, Cold-Expert NVMe Streaming, A2 & D4
// ---------------------------------------------------------------------------

pub const DEFAULT_HYBRID_PROMPT_LENS = [_]u32{ 1024, 2048 };

pub const HybridPrefillCaseResult = struct {
    prompt_tokens: u32,
    chunk_tokens: u32,
    num_layers: u32,
    hybrid_ms: f64,
    gpu_ms: f64,
    ane_total_ms: f64,
    gpu_ffn_total_ms: f64,
    overlap_efficiency: f64,
    all_outputs_zero_copy: bool,
    all_buckets_use_ane: bool,

    pub fn hybridTokPerS(self: HybridPrefillCaseResult) f64 {
        return if (self.hybrid_ms > 0.0)
            (@as(f64, @floatFromInt(self.prompt_tokens)) * 1000.0) / self.hybrid_ms
        else
            0.0;
    }

    pub fn gpuTokPerS(self: HybridPrefillCaseResult) f64 {
        return if (self.gpu_ms > 0.0)
            (@as(f64, @floatFromInt(self.prompt_tokens)) * 1000.0) / self.gpu_ms
        else
            0.0;
    }

    pub fn speedupVsGpu(self: HybridPrefillCaseResult) f64 {
        return if (self.hybrid_ms > 0.0) self.gpu_ms / self.hybrid_ms else 0.0;
    }
};

pub const HybridPrefillBenchReport = struct {
    hidden_size: u32,
    num_layers: u32,
    num_experts: u32,
    top_k: u32,
    chunk_tokens: u32,
    results: []HybridPrefillCaseResult,
    swapouts_delta: u64,

    pub fn deinit(self: *HybridPrefillBenchReport, allocator: std.mem.Allocator) void {
        allocator.free(self.results);
        self.results = &.{};
    }

    pub fn write(self: *const HybridPrefillBenchReport, out: *std.Io.Writer) std.Io.Writer.Error!void {
        try out.print(
            \\Synthetic End-to-End A1/G6 Hybrid Wavefront Prefill vs GPU-Only Prefill
            \\hybrid_prefill.geometry          layers={d} hidden={d} experts={d} top_k={d} chunk={d}
            \\prompt_toks   hybrid(ms)   hybrid(tok/s)   gpu_only(ms)   gpu(tok/s)   speedup   overlap   ane   zc
            \\
        , .{
            self.num_layers,
            self.hidden_size,
            self.num_experts,
            self.top_k,
            self.chunk_tokens,
        });
        for (self.results) |r| {
            try out.print(
                \\{d: >11}   {d: >10.3}   {d: >13.1}   {d: >12.3}   {d: >10.1}   {d: >6.2}x   {d: >6.2}x   {s: <3}   {s}
                \\
            , .{
                r.prompt_tokens,
                r.hybrid_ms,
                r.hybridTokPerS(),
                r.gpu_ms,
                r.gpuTokPerS(),
                r.speedupVsGpu(),
                r.overlap_efficiency,
                if (r.all_buckets_use_ane) "yes" else "no",
                if (r.all_outputs_zero_copy) "yes" else "no",
            });
        }
        try out.print("hybrid_prefill.swapouts_delta    {d}\n", .{self.swapouts_delta});
    }
};

pub const ColdSynthResult = struct {
    num_layers: u32,
    num_experts: u32,
    cold_experts_per_layer: u32,
    hot_wired_bytes: u64,
    cold_wired_bytes: u64,
    gen_tokens: u32,
    bit_exact: bool,
    cold_stats: qwen3.ColdExpertStats,
    heap_delta: vmstat.HeapDelta,
    swapouts_delta: u64,

    pub fn write(self: *const ColdSynthResult, out: *std.Io.Writer) std.Io.Writer.Error!void {
        const ct = self.cold_stats.cache_telemetry;
        try out.print(
            \\Synthetic End-to-End Cold-Expert NVMe Streaming Decode (C3-C5 + A2)
            \\cold_synth.geometry              layers={d} experts={d} cold_per_layer={d} slots={d}
            \\cold_synth.io_backend            {s} (ane_predictor={s}, uses_ne={s})
            \\cold_synth.wired_bytes           all_hot={d} cold_mode={d}
            \\cold_synth.bit_exact_vs_hot      {s} (gen_tokens={d})
            \\cold_synth.hits                  hot={d} cache={d} prefetch={d} sync_miss={d}
            \\cold_synth.hit_rates             prefetch={d:.3} cold_total={d:.3} bytes_read={d} evictions={d}
            \\cold_synth.malloc_blocks_delta   {d}
            \\cold_synth.malloc_bytes_delta    {d}
            \\cold_synth.swapouts_delta        {d}
            \\
        , .{
            self.num_layers,
            self.num_experts,
            self.cold_experts_per_layer,
            self.cold_stats.slot_count,
            self.cold_stats.io_backend.label(),
            if (self.cold_stats.ane_predictor_active) "on" else "off",
            if (self.cold_stats.ane_predictor_uses_ne) "yes" else "no",
            self.hot_wired_bytes,
            self.cold_wired_bytes,
            if (self.bit_exact) "yes" else "NO",
            self.gen_tokens,
            ct.hot_hits,
            ct.cache_hits,
            ct.prefetch_hits,
            ct.sync_misses,
            ct.prefetchHitRate(),
            ct.coldHitRate(),
            ct.bytes_read,
            ct.evictions,
            self.heap_delta.blocks_in_use,
            self.heap_delta.size_in_use,
            self.swapouts_delta,
        });
    }
};

pub const D4SynthResult = struct {
    effective_placement: draft_mod.Placement,
    placement_gate_ane_us: f64,
    placement_gate_gpu_us: f64,
    lossless_match: bool,
    generated_tokens: u32,
    rounds: u64,
    mean_accept_len: f64,
    mean_drafted_nodes: f64,
    mean_verified_nodes: f64,
    heap_delta: vmstat.HeapDelta,
    swapouts_delta: u64,

    pub fn write(self: *const D4SynthResult, out: *std.Io.Writer) std.Io.Writer.Error!void {
        try out.print(
            \\Synthetic End-to-End D4 Same-Family Small Draft Lossless Check
            \\d4_synth.placement               {s} (gate_ane_us={d:.1}, gate_gpu_us={d:.1})
            \\d4_synth.lossless_match          {s}
            \\d4_synth.generated_tokens        {d}
            \\d4_synth.rounds                  {d}
            \\d4_synth.mean_accept_len         {d:.3}
            \\d4_synth.drafted_nodes           {d:.2}
            \\d4_synth.verified_nodes          {d:.2}
            \\d4_synth.malloc_blocks_delta     {d}
            \\d4_synth.malloc_bytes_delta      {d}
            \\d4_synth.swapouts_delta          {d}
            \\
        , .{
            self.effective_placement.label(),
            self.placement_gate_ane_us,
            self.placement_gate_gpu_us,
            if (self.lossless_match) "yes" else "NO",
            self.generated_tokens,
            self.rounds,
            self.mean_accept_len,
            self.mean_drafted_nodes,
            self.mean_verified_nodes,
            self.heap_delta.blocks_in_use,
            self.heap_delta.size_in_use,
            self.swapouts_delta,
        });
    }
};

pub const M4Config = struct {
    s3: aio.SpikeS3Config = .{},
    s6: wavefront.SpikeS6Config = .{},
    a2: predictor_mod.PredictorEvalConfig = .{},
    hybrid_prompt_lens: []const u32 = &DEFAULT_HYBRID_PROMPT_LENS,
    hybrid_chunk_tokens: u32 = 512,
    hybrid_hidden_size: u32 = 512,
    hybrid_num_layers: u32 = 2,
    hybrid_num_q_heads: u32 = 8,
    hybrid_num_kv_heads: u32 = 4,
    hybrid_head_dim: u32 = 64,
    hybrid_moe_intermediate: u32 = 256,
    hybrid_num_experts: u32 = 16,
    hybrid_top_k: u32 = 4,
    hybrid_timed_iters: u32 = 2,
    cold_temp_dir: []const u8 = "/Volumes/Akari/.tmp",
    cold_gen_tokens: u32 = 12,
    d4_gen_tokens: u32 = 12,
};

pub const M4Report = struct {
    s3: aio.SpikeS3Report,
    s6: wavefront.SpikeS6Report,
    a2: predictor_mod.PredictorEvalReport,
    hybrid_prefill: HybridPrefillBenchReport,
    cold_synth: ColdSynthResult,
    d4_synth: D4SynthResult,
    vm_before: vmstat.Snapshot,
    vm_after: vmstat.Snapshot,

    pub fn deinit(self: *M4Report, allocator: std.mem.Allocator) void {
        self.s3.deinit(allocator);
        self.hybrid_prefill.deinit(allocator);
    }

    pub fn write(self: *const M4Report, out: *std.Io.Writer) std.Io.Writer.Error!void {
        try out.writeAll("\n=== M4 Hybrid ANE/GPU Wavefront, Cold-Expert NVMe Streaming, A2 & D4 Benchmarks ===\n\n");
        try self.s3.write(out);
        try out.writeAll("\n");
        try self.s6.write(out);
        try out.writeAll("\n");
        try self.a2.write(out);
        try out.writeAll("\n");
        try self.hybrid_prefill.write(out);
        try out.writeAll("\n");
        try self.cold_synth.write(out);
        try out.writeAll("\n");
        try self.d4_synth.write(out);
        try out.print(
            \\
            \\M4 Swap Check
            \\m4.swap.status                   ok
            \\m4.swap.swapouts_delta           {d}
            \\m4.swap.pageouts_delta           {d}
            \\
        , .{
            vmstat.Snapshot.swapoutDelta(self.vm_before, self.vm_after),
            vmstat.Snapshot.pageoutDelta(self.vm_before, self.vm_after),
        });
    }
};

pub const M4Error = error{
    LosslessMismatch,
    InvalidConfig,
} || aio.SpikeS3Error || wavefront.SpikeS6Error || predictor_mod.AneExpertPredictor.Error ||
    qwen3.LoadError || qwen3.StepError || draft_mod.SmallDraftLoadError ||
    engine_mod.InitError || engine_mod.RunError || SynthWriteError || vmstat.Error || vmstat.SwapError;

pub fn runHybridPrefillVsGpuBench(allocator: std.mem.Allocator, config: M4Config) M4Error!HybridPrefillBenchReport {
    if (config.hybrid_prompt_lens.len == 0 or config.hybrid_timed_iters == 0) return error.InvalidConfig;
    var max_prompt: u32 = config.hybrid_chunk_tokens * 2;
    for (config.hybrid_prompt_lens) |pl| {
        if (pl <= config.hybrid_chunk_tokens) return error.InvalidConfig;
        if (pl > max_prompt) max_prompt = pl;
    }

    const cfg: qwen3.ModelConfig = .{
        .arch = .qwen3_moe,
        .quant_bits = .q3,
        .vocab_size = 256,
        .hidden_size = config.hybrid_hidden_size,
        .num_hidden_layers = config.hybrid_num_layers,
        .num_attention_heads = config.hybrid_num_q_heads,
        .num_key_value_heads = config.hybrid_num_kv_heads,
        .head_dim = config.hybrid_head_dim,
        .moe_intermediate_size = config.hybrid_moe_intermediate,
        .num_experts = config.hybrid_num_experts,
        .num_experts_per_tok = config.hybrid_top_k,
        .rms_norm_eps = 1e-6,
        .rope_theta = 1_000_000.0,
        .max_position_embeddings = max_prompt + 64,
        .tie_word_embeddings = false,
        .norm_topk_prob = true,
    };

    const image = try qwen3.buildSyntheticAmfImageWithSpec(allocator, cfg, .{
        .linear_dtype = .q3_affine,
        .embed_dtype = .f16,
        .norm_dtype = .f16,
        .router_dtype = .f16,
        .expert_dtype = .q3_affine,
    }, 0x0004_A166);
    defer allocator.free(image);

    var model = try qwen3.Qwen3Model.loadFromBytes(allocator, image, cfg, .{
        .max_seq_len = max_prompt,
        .chunk_tokens = config.hybrid_chunk_tokens,
        .max_tree_nodes = 64,
        .verify_crcs = true,
        .prefill_mode = .hybrid,
        .enable_hybrid_prefill = true,
        .hybrid_max_bucket = max_prompt,
    });
    defer model.deinit(allocator);

    const tokens = try allocator.alloc(u32, max_prompt);
    defer allocator.free(tokens);
    for (tokens, 0..) |*t, i| t.* = @intCast((i * 19 + 7) % cfg.vocab_size);

    const results = try allocator.alloc(HybridPrefillCaseResult, config.hybrid_prompt_lens.len);
    errdefer allocator.free(results);

    const vm_before = try vmstat.Snapshot.capture();
    for (config.hybrid_prompt_lens, 0..) |plen, ci| {
        const prompt = tokens[0..plen];

        // Warmup both paths once so ANE & Metal 4 caches are hot.
        model.resetKvCache();
        _ = try model.prefillHybridWavefront(prompt, true);
        model.resetKvCache();
        {
            var pos: usize = 0;
            while (pos < prompt.len) {
                const take = @min(@as(usize, 64), prompt.len - pos);
                _ = try model.prefillChunk(prompt[pos .. pos + take], @intCast(pos), pos + take == prompt.len);
                pos += take;
            }
        }

        // Timed A1/G6 Hybrid Wavefront Prefill.
        var hybrid_ns_sum: u64 = 0;
        var ane_ns_sum: u64 = 0;
        var gpu_ffn_ns_sum: u64 = 0;
        var all_zc = true;
        var all_ane = true;
        for (0..config.hybrid_timed_iters) |_| {
            model.resetKvCache();
            const t0 = try vmstat.nowNs();
            const hres = try model.prefillHybridWavefront(prompt, true);
            const t1 = try vmstat.nowNs();
            hybrid_ns_sum += t1 -| t0;
            ane_ns_sum += hres.wavefront_stats.ane_total_ns;
            gpu_ffn_ns_sum += hres.wavefront_stats.gpu_total_ns;
            all_zc = all_zc and hres.all_outputs_zero_copy;
            all_ane = all_ane and hres.all_buckets_use_ane;
        }

        // Timed GPU-only batched chunked prefill (`prefillChunk`, 64 tokens/batch).
        var gpu_ns_sum: u64 = 0;
        for (0..config.hybrid_timed_iters) |_| {
            model.resetKvCache();
            const t0 = try vmstat.nowNs();
            var pos: usize = 0;
            while (pos < prompt.len) {
                const take = @min(@as(usize, 64), prompt.len - pos);
                _ = try model.prefillChunk(prompt[pos .. pos + take], @intCast(pos), pos + take == prompt.len);
                pos += take;
            }
            const t1 = try vmstat.nowNs();
            gpu_ns_sum += t1 -| t0;
        }

        const iters_f: f64 = @floatFromInt(config.hybrid_timed_iters);
        const h_ms = (@as(f64, @floatFromInt(hybrid_ns_sum)) / iters_f) / 1_000_000.0;
        const g_ms = (@as(f64, @floatFromInt(gpu_ns_sum)) / iters_f) / 1_000_000.0;
        const ane_ms = (@as(f64, @floatFromInt(ane_ns_sum)) / iters_f) / 1_000_000.0;
        const gffn_ms = (@as(f64, @floatFromInt(gpu_ffn_ns_sum)) / iters_f) / 1_000_000.0;

        results[ci] = .{
            .prompt_tokens = plen,
            .chunk_tokens = config.hybrid_chunk_tokens,
            .num_layers = config.hybrid_num_layers,
            .hybrid_ms = h_ms,
            .gpu_ms = g_ms,
            .ane_total_ms = ane_ms,
            .gpu_ffn_total_ms = gffn_ms,
            .overlap_efficiency = if (h_ms > 0.0) (ane_ms + gffn_ms) / h_ms else 0.0,
            .all_outputs_zero_copy = all_zc,
            .all_buckets_use_ane = all_ane,
        };
    }
    const vm_after = try vmstat.Snapshot.capture();
    try vmstat.checkNoSwap(vm_before, vm_after);

    return .{
        .hidden_size = config.hybrid_hidden_size,
        .num_layers = config.hybrid_num_layers,
        .num_experts = config.hybrid_num_experts,
        .top_k = config.hybrid_top_k,
        .chunk_tokens = config.hybrid_chunk_tokens,
        .results = results,
        .swapouts_delta = vmstat.Snapshot.swapoutDelta(vm_before, vm_after),
    };
}

pub fn runColdSynthBench(
    allocator: std.mem.Allocator,
    temp_dir: []const u8,
    gen_tokens: u32,
) M4Error!ColdSynthResult {
    const cfg: qwen3.ModelConfig = .{
        .arch = .qwen3_moe,
        .quant_bits = .q3,
        .vocab_size = 128,
        .hidden_size = 256,
        .num_hidden_layers = 4,
        .num_attention_heads = 4,
        .num_key_value_heads = 2,
        .head_dim = 64,
        .moe_intermediate_size = 256,
        .num_experts = 16,
        .num_experts_per_tok = 4,
        .rms_norm_eps = 1e-6,
        .rope_theta = 1_000_000.0,
        .max_position_embeddings = 128,
        .tie_word_embeddings = false,
        .norm_topk_prob = true,
    };

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const pid = std.c.getpid();
    const stamp = try vmstat.nowNs();
    const path_slice = std.fmt.bufPrint(
        path_buf[0 .. path_buf.len - 1],
        "{s}/aminfer_m4_cold_{d}_{d}.amf",
        .{ temp_dir, pid, stamp },
    ) catch return error.InvalidConfig;
    path_buf[path_slice.len] = 0;
    const amf_path: [:0]const u8 = path_buf[0..path_slice.len :0];

    try writeSyntheticQwen3AmfFile(allocator, amf_path, cfg, 0x0004_C01D);
    defer _ = std.c.unlink(amf_path.ptr);

    const prompt = [_]u32{ 5, 11, 23, 37, 41 };
    const n_gen: usize = @min(@as(usize, @max(gen_tokens, 2)), 32);
    var ref_out: [32]u32 = undefined;
    var cold_out: [32]u32 = undefined;

    const vm_before = try vmstat.Snapshot.capture();

    var hot_wired_bytes: u64 = 0;
    {
        var hot_model = try qwen3.Qwen3Model.loadFromFile(allocator, amf_path, cfg, .{
            .max_seq_len = 64,
            .chunk_tokens = 32,
            .verify_crcs = true,
        });
        defer hot_model.deinit(allocator);
        hot_wired_bytes = hot_model.totalWiredBytes();

        var cur = try hot_model.prefillPrompt(&prompt, true);
        ref_out[0] = cur;
        var pos: u32 = @intCast(prompt.len);
        for (1..n_gen) |step| {
            cur = try hot_model.stepDecode(cur, pos);
            pos += 1;
            ref_out[step] = cur;
        }
    }

    var cold_model = try qwen3.Qwen3Model.loadFromFile(allocator, amf_path, cfg, .{
        .max_seq_len = 64,
        .chunk_tokens = 32,
        .verify_crcs = true,
        .cold_experts_per_layer = 8,
        .cold_slot_count = 16,
        .io_backend = .auto,
        .enable_ane_predictor = true,
    });
    defer cold_model.deinit(allocator);

    var cur = try cold_model.prefillPrompt(&prompt, true);
    cold_out[0] = cur;
    cold_model.resetColdTelemetry();

    const heap_before = try vmstat.Snapshot.capture();
    var pos: u32 = @intCast(prompt.len);
    for (1..n_gen) |step| {
        cur = try cold_model.stepDecode(cur, pos);
        pos += 1;
        cold_out[step] = cur;
    }
    const heap_after = try vmstat.Snapshot.capture();
    const vm_after = try vmstat.Snapshot.capture();
    try vmstat.checkNoSwap(vm_before, vm_after);

    const exact = std.mem.eql(u32, ref_out[0..n_gen], cold_out[0..n_gen]);
    if (!exact) return error.LosslessMismatch;

    return .{
        .num_layers = cfg.num_hidden_layers,
        .num_experts = cfg.num_experts,
        .cold_experts_per_layer = 8,
        .hot_wired_bytes = hot_wired_bytes,
        .cold_wired_bytes = cold_model.totalWiredBytes(),
        .gen_tokens = @intCast(n_gen),
        .bit_exact = exact,
        .cold_stats = cold_model.coldStats().?,
        .heap_delta = vmstat.heapDelta(heap_before, heap_after),
        .swapouts_delta = vmstat.Snapshot.swapoutDelta(vm_before, vm_after),
    };
}

pub fn runD4SynthBench(allocator: std.mem.Allocator, gen_tokens: u32) M4Error!D4SynthResult {
    const target_cfg: qwen3.ModelConfig = .{
        .arch = .qwen3_dense,
        .quant_bits = null,
        .default_float_dtype = .bf16,
        .vocab_size = 128,
        .hidden_size = 128,
        .num_hidden_layers = 4,
        .num_attention_heads = 4,
        .num_key_value_heads = 2,
        .head_dim = 64,
        .intermediate_size = 256,
        .rms_norm_eps = 1e-6,
        .rope_theta = 1_000_000.0,
        .max_position_embeddings = 128,
        .tie_word_embeddings = true,
    };
    const draft_cfg: qwen3.ModelConfig = .{
        .arch = .qwen3_dense,
        .quant_bits = null,
        .default_float_dtype = .bf16,
        .vocab_size = 128,
        .hidden_size = 128,
        .num_hidden_layers = 2,
        .num_attention_heads = 4,
        .num_key_value_heads = 2,
        .head_dim = 64,
        .intermediate_size = 256,
        .rms_norm_eps = 1e-6,
        .rope_theta = 1_000_000.0,
        .max_position_embeddings = 128,
        .tie_word_embeddings = true,
    };

    const target_bytes = try qwen3.buildSyntheticAmfImageWithSpec(allocator, target_cfg, .{
        .linear_dtype = .bf16,
        .embed_dtype = .bf16,
        .norm_dtype = .bf16,
        .use_weight_suffix = true,
    }, 0x0004_D400);
    defer allocator.free(target_bytes);

    const draft_bytes = try qwen3.buildSyntheticAmfImageWithSpec(allocator, draft_cfg, .{
        .linear_dtype = .bf16,
        .embed_dtype = .bf16,
        .norm_dtype = .bf16,
        .use_weight_suffix = true,
    }, 0x0004_D401);
    defer allocator.free(draft_bytes);

    const prompt = [_]u32{ 3, 9, 17, 25, 31, 43 };
    const n_gen: usize = @min(@as(usize, @max(gen_tokens, 2)), 32);
    var ref_out: [32]u32 = undefined;
    var spec_out: [32]u32 = undefined;

    const vm_before = try vmstat.Snapshot.capture();
    {
        var ref_model = try qwen3.Qwen3Model.loadFromBytes(allocator, target_bytes, target_cfg, .{
            .max_seq_len = 64,
            .chunk_tokens = 32,
            .verify_crcs = true,
        });
        defer ref_model.deinit(allocator);
        var cur = try ref_model.prefillPrompt(&prompt, true);
        ref_out[0] = cur;
        var pos: u32 = @intCast(prompt.len);
        for (1..n_gen) |step| {
            cur = try ref_model.stepDecode(cur, pos);
            pos += 1;
            ref_out[step] = cur;
        }
    }

    var target_model = try qwen3.Qwen3Model.loadFromBytes(allocator, target_bytes, target_cfg, .{
        .max_seq_len = 64,
        .chunk_tokens = 32,
        .max_tree_nodes = 16,
        .enable_feature_taps = false,
        .verify_crcs = true,
    });
    defer target_model.deinit(allocator);

    var sd = try draft_mod.SmallDraftModel.loadFromBytes(allocator, &target_model, draft_bytes, draft_cfg, .{
        .placement = .auto,
        .max_seq_len = 64,
        .chunk_tokens = 32,
        .max_tree_slots = 16,
        .verify_crcs = true,
    });
    defer sd.deinit(allocator);

    var eng = try engine_mod.SpecEngine.initWithSmallDraft(allocator, &target_model, &sd, .{
        .mode = .small_draft,
        .placement = .auto,
        .max_tree_nodes = 12,
        .max_draft_depth = 4,
        .branch_width = 2,
        .top_k_per_node = 3,
        .temperature = 0.0,
        .enable_pruning = true,
    });
    defer eng.deinit(allocator);

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

    const exact = std.mem.eql(u32, ref_out[0..n_gen], spec_out[0..n_gen]);
    if (!exact) return error.LosslessMismatch;

    return .{
        .effective_placement = eng.effectivePlacement(),
        .placement_gate_ane_us = sd.placement_gate_ane_us,
        .placement_gate_gpu_us = sd.placement_gate_gpu_us,
        .lossless_match = exact,
        .generated_tokens = @intCast(n_gen),
        .rounds = eng.stats.rounds,
        .mean_accept_len = eng.stats.meanAcceptanceLength(),
        .mean_drafted_nodes = eng.stats.meanDraftedNodes(),
        .mean_verified_nodes = eng.stats.meanVerifiedNodes(),
        .heap_delta = vmstat.heapDelta(heap_before, heap_after),
        .swapouts_delta = vmstat.Snapshot.swapoutDelta(vm_before, vm_after),
    };
}

pub fn runM4WithConfig(allocator: std.mem.Allocator, config: M4Config) M4Error!M4Report {
    const vm_before = try vmstat.Snapshot.capture();
    var s3 = try aio.runSpikeS3(allocator, config.s3);
    errdefer s3.deinit(allocator);
    const s6 = try wavefront.runSpikeS6(allocator, config.s6);
    const a2 = try predictor_mod.runPredictorBenchmark(allocator, config.a2);
    var hybrid_prefill = try runHybridPrefillVsGpuBench(allocator, config);
    errdefer hybrid_prefill.deinit(allocator);
    const cold_synth = try runColdSynthBench(allocator, config.cold_temp_dir, config.cold_gen_tokens);
    const d4_synth = try runD4SynthBench(allocator, config.d4_gen_tokens);
    const vm_after = try vmstat.Snapshot.capture();
    try vmstat.checkNoSwap(vm_before, vm_after);

    return .{
        .s3 = s3,
        .s6 = s6,
        .a2 = a2,
        .hybrid_prefill = hybrid_prefill,
        .cold_synth = cold_synth,
        .d4_synth = d4_synth,
        .vm_before = vm_before,
        .vm_after = vm_after,
    };
}

pub fn runM4(allocator: std.mem.Allocator) M4Error!M4Report {
    return runM4WithConfig(allocator, .{});
}

// ---------------------------------------------------------------------------
// Real-Checkpoint A2 Cross-Layer MoE Expert Predictor Evaluation (`bench-a2`)
// ---------------------------------------------------------------------------

pub const RealA2DeltaMetrics = struct {
    delta: u32 = 0,
    samples: u64 = 0,
    ane_inmem_r8: f64 = 0.0,
    ane_inmem_r12: f64 = 0.0,
    ane_inmem_r16: f64 = 0.0,
    ane_offline_r8: f64 = 0.0,
    ane_offline_r12: f64 = 0.0,
    ane_offline_r16: f64 = 0.0,
    gpu_baseline_r8: f64 = 0.0,
    gpu_baseline_r12: f64 = 0.0,
    gpu_baseline_r16: f64 = 0.0,
};

pub const RealA2Options = struct {
    capture_steps: u32 = 16,
    max_lookahead: u32 = 3,
    timed_iters: u32 = 32,
    ane_predictor_path: ?[:0]const u8 = null,
    verify_crcs: bool = true,
    max_seq_len: u32 = 512,
};

pub const RealA2Report = struct {
    num_layers: u32,
    num_experts: u32,
    top_k: u32,
    hidden_size: u32,
    capture_steps: u32,
    max_lookahead: u32,
    measured_gpu_layer_us: f64,
    inmem_compile_ms: f64,
    inmem_ane_us: f64,
    inmem_uses_ne: bool,
    inmem_zero_copy: bool,
    has_offline_mlpackage: bool,
    offline_compile_ms: f64,
    offline_ane_us: f64,
    offline_uses_ne: bool,
    offline_zero_copy: bool,
    gpu_lookahead_us: f64,
    deltas: [predictor_mod.MAX_LOOKAHEAD]RealA2DeltaMetrics,
    swapouts_delta: u64,
    pageouts_delta: u64,

    pub fn write(self: *const RealA2Report, out: *std.Io.Writer) std.Io.Writer.Error!void {
        try out.print(
            \\Real-Checkpoint A2 ANE Cross-Layer MoE Expert Predictor Evaluation
            \\real_a2.shape                    layers={d} experts={d} top_k={d} hidden={d} steps={d}
            \\real_a2.gpu_layer_time_us        {d:.2}
            \\real_a2.inmem_ane                compile_ms={d:.1} latency_us={d:.2} uses_ne={s} zero_copy={s}
            \\real_a2.gpu_lookahead_us         {d:.2} (for {d} lookahead layers)
            \\
        , .{
            self.num_layers,
            self.num_experts,
            self.top_k,
            self.hidden_size,
            self.capture_steps,
            self.measured_gpu_layer_us,
            self.inmem_compile_ms,
            self.inmem_ane_us,
            if (self.inmem_uses_ne) "yes" else "no",
            if (self.inmem_zero_copy) "yes" else "no",
            self.gpu_lookahead_us,
            self.max_lookahead,
        });
        if (self.has_offline_mlpackage) {
            try out.print(
                \\real_a2.offline_mlpackage        compile_ms={d:.1} latency_us={d:.2} uses_ne={s} zero_copy={s}
                \\
            , .{
                self.offline_compile_ms,
                self.offline_ane_us,
                if (self.offline_uses_ne) "yes" else "no",
                if (self.offline_zero_copy) "yes" else "no",
            });
        }

        try out.writeAll(
            \\
            \\delta  lead_us   inmem_ANE(R@8/12/16)       gpu_base(R@8/12/16)        w7_mlpkg(R@8/12/16)
            \\
        );
        for (0..self.max_lookahead) |di| {
            const d = self.deltas[di];
            const lead_us = @as(f64, @floatFromInt(d.delta)) * self.measured_gpu_layer_us - self.inmem_ane_us;
            if (self.has_offline_mlpackage) {
                try out.print(
                    \\{d: >5}  {d: >7.1}   {d:.3} / {d:.3} / {d:.3}      {d:.3} / {d:.3} / {d:.3}      {d:.3} / {d:.3} / {d:.3}
                    \\
                , .{
                    d.delta,
                    lead_us,
                    d.ane_inmem_r8,
                    d.ane_inmem_r12,
                    d.ane_inmem_r16,
                    d.gpu_baseline_r8,
                    d.gpu_baseline_r12,
                    d.gpu_baseline_r16,
                    d.ane_offline_r8,
                    d.ane_offline_r12,
                    d.ane_offline_r16,
                });
            } else {
                try out.print(
                    \\{d: >5}  {d: >7.1}   {d:.3} / {d:.3} / {d:.3}      {d:.3} / {d:.3} / {d:.3}      n/a
                    \\
                , .{
                    d.delta,
                    lead_us,
                    d.ane_inmem_r8,
                    d.ane_inmem_r12,
                    d.ane_inmem_r16,
                    d.gpu_baseline_r8,
                    d.gpu_baseline_r12,
                    d.gpu_baseline_r16,
                });
            }
        }
        try out.print(
            \\real_a2.swapouts_delta           {d}
            \\real_a2.pageouts_delta           {d}
            \\
        , .{ self.swapouts_delta, self.pageouts_delta });
    }
};

pub const RealA2Error = error{
    InvalidConfig,
} || qwen3.LoadError || qwen3.StepError || predictor_mod.AneExpertPredictor.Error || vmstat.Error || vmstat.SwapError;

pub fn runRealA2Eval(
    allocator: std.mem.Allocator,
    amf_path: [:0]const u8,
    cfg: qwen3.ModelConfig,
    prompt_ids: []const u32,
    opts: RealA2Options,
) RealA2Error!RealA2Report {
    if (cfg.arch != .qwen3_moe or cfg.num_hidden_layers < 2 or prompt_ids.len == 0 or
        opts.capture_steps == 0 or opts.max_lookahead == 0 or opts.max_lookahead > predictor_mod.MAX_LOOKAHEAD or
        opts.timed_iters == 0)
    {
        return error.InvalidConfig;
    }

    const max_la: u32 = @min(opts.max_lookahead, cfg.num_hidden_layers - 1);
    const needed_seq: u32 = @intCast(prompt_ids.len + opts.capture_steps + 32);
    const max_seq_len: u32 = @min(cfg.max_position_embeddings, @max(opts.max_seq_len, needed_seq));

    const n_layers: usize = cfg.num_hidden_layers;
    const h: usize = cfg.hidden_size;
    const top_k: usize = cfg.num_experts_per_tok;
    const n_exp: usize = cfg.num_experts;
    const steps: usize = opts.capture_steps;

    const hidden_trace = try allocator.alloc(f16, steps * n_layers * h);
    defer allocator.free(hidden_trace);
    const topk_trace = try allocator.alloc(u32, steps * n_layers * top_k);
    defer allocator.free(topk_trace);

    var deltas: [predictor_mod.MAX_LOOKAHEAD]RealA2DeltaMetrics = @splat(.{});
    for (0..max_la) |di| deltas[di].delta = @intCast(di + 1);

    var gpu_logits_storage: [predictor_mod.MAX_LOOKAHEAD][predictor_mod.MAX_EXPERTS]f32 = undefined;
    var gpu_logits_slices: [predictor_mod.MAX_LOOKAHEAD][]f32 = undefined;
    for (0..max_la) |di| gpu_logits_slices[di] = gpu_logits_storage[di][0..n_exp];

    var measured_gpu_layer_us: f64 = 0.0;
    var gpu_lookahead_ns_sum: u64 = 0;
    var gpu_lookahead_full_calls: u64 = 0;
    var routers: [][]f16 = &.{};
    defer {
        for (routers) |r| allocator.free(r);
        allocator.free(routers);
    }

    var swapouts_total: u64 = 0;
    var pageouts_total: u64 = 0;

    // Phase 1: load target checkpoint, capture real hidden states & top-K routing,
    // benchmark GPU lookahead baseline, extract folded router weights (~24 MiB),
    // then release the multi-GiB target weights before CoreML ANE compilation.
    {
        var model = try qwen3.Qwen3Model.loadFromFile(allocator, amf_path, cfg, .{
            .max_seq_len = max_seq_len,
            .verify_crcs = opts.verify_crcs,
        });
        defer model.deinit(allocator);

        const vm_gpu_before = try vmstat.Snapshot.capture();

        var pos: u32 = 0;
        for (prompt_ids[0 .. prompt_ids.len - 1]) |pid| {
            try model.prefillToken(pid, pos);
            pos += 1;
        }

        // Measure true GPU per-layer decode latency at current KV position (`prefillToken` = body only).
        for (0..2) |_| try model.prefillToken(prompt_ids[prompt_ids.len - 1], pos);
        const gl_t0 = try vmstat.nowNs();
        const body_iters: u32 = 6;
        for (0..body_iters) |_| try model.prefillToken(prompt_ids[prompt_ids.len - 1], pos);
        const gl_t1 = try vmstat.nowNs();
        measured_gpu_layer_us = (@as(f64, @floatFromInt(gl_t1 -| gl_t0)) /
            (@as(f64, @floatFromInt(body_iters)) * @as(f64, @floatFromInt(cfg.num_hidden_layers)))) / 1000.0;

        var cur_tok = prompt_ids[prompt_ids.len - 1];
        for (0..steps) |s_idx| {
            const h_dst = hidden_trace[s_idx * n_layers * h .. (s_idx + 1) * n_layers * h];
            const k_dst = topk_trace[s_idx * n_layers * top_k .. (s_idx + 1) * n_layers * top_k];
            cur_tok = try model.stepDecodeCaptureLayerRouting(cur_tok, pos, h_dst, k_dst);
            pos += 1;
        }

        for (0..steps) |s_idx| {
            for (0..n_layers - 1) |cur_l| {
                const x_cur = hidden_trace[(s_idx * n_layers + cur_l) * h ..][0..h];
                const pred_layers: u32 = @min(max_la, @as(u32, @intCast(n_layers - 1 - cur_l)));
                const gpu_ns = try model.evalGpuRouterLookahead(x_cur, @intCast(cur_l), max_la, gpu_logits_slices[0..pred_layers]);
                if (pred_layers == max_la) {
                    gpu_lookahead_ns_sum += gpu_ns;
                    gpu_lookahead_full_calls += 1;
                }
                for (0..pred_layers) |di| {
                    const target_l = cur_l + 1 + di;
                    const gt_topk = topk_trace[(s_idx * n_layers + target_l) * top_k ..][0..top_k];
                    deltas[di].gpu_baseline_r8 += predictor_mod.computeRecallAtK(gpu_logits_slices[di], gt_topk, 8);
                    deltas[di].gpu_baseline_r12 += predictor_mod.computeRecallAtK(gpu_logits_slices[di], gt_topk, 12);
                    deltas[di].gpu_baseline_r16 += predictor_mod.computeRecallAtK(gpu_logits_slices[di], gt_topk, 16);
                    deltas[di].samples += 1;
                }
            }
        }

        routers = try model.extractRouterWeightsF16(allocator, true);

        const vm_gpu_after = try vmstat.Snapshot.capture();
        try vmstat.checkNoSwap(vm_gpu_before, vm_gpu_after);
        swapouts_total += vmstat.Snapshot.swapoutDelta(vm_gpu_before, vm_gpu_after);
        pageouts_total += vmstat.Snapshot.pageoutDelta(vm_gpu_before, vm_gpu_after);
    }

    // Phase 2: compile & evaluate in-memory folded ANE predictor + optional W7 `.mlpackage`
    // with full host RAM headroom freed.
    const device = try metal.Device.open();
    defer device.close();

    const ro_slices = try allocator.alloc([]const f16, routers.len);
    defer allocator.free(ro_slices);
    for (routers, 0..) |r, i| ro_slices[i] = r;

    const inmem_pcfg: predictor_mod.PredictorConfig = .{
        .num_layers = cfg.num_hidden_layers,
        .num_experts = cfg.num_experts,
        .hidden_size = cfg.hidden_size,
        .lookahead_layers = max_la,
        .max_tokens = 1,
        .top_k = cfg.num_experts_per_tok,
        .compute_units = .cpu_and_neural_engine,
        .enable_async_thread = false,
    };
    var inmem_pred = try predictor_mod.AneExpertPredictor.initFromRouterWeights(allocator, device, inmem_pcfg, ro_slices);
    defer inmem_pred.deinit(allocator);

    var offline_pred: ?*predictor_mod.AneExpertPredictor = null;
    defer if (offline_pred) |p| p.deinit(allocator);
    if (opts.ane_predictor_path) |p| {
        const off_pcfg: predictor_mod.PredictorConfig = .{
            .num_layers = cfg.num_hidden_layers,
            .num_experts = cfg.num_experts,
            .hidden_size = cfg.hidden_size,
            .lookahead_layers = max_la,
            .max_tokens = 32,
            .top_k = cfg.num_experts_per_tok,
            .compute_units = .cpu_and_neural_engine,
            .enable_async_thread = false,
        };
        offline_pred = try predictor_mod.AneExpertPredictor.loadAllLayersFromPath(allocator, device, off_pcfg, p);
    }

    const vm_ane_before = try vmstat.Snapshot.capture();

    var bitsets: [predictor_mod.MAX_LOOKAHEAD]tree_mod.ExpertBitset = undefined;
    var prefetches: [32]predictor_mod.PrefetchCandidate = undefined;
    var logits_buf: [predictor_mod.MAX_EXPERTS]f32 = undefined;

    var inmem_zc = true;
    var offline_zc = true;

    for (0..steps) |s_idx| {
        for (0..n_layers - 1) |cur_l| {
            const x_cur = hidden_trace[(s_idx * n_layers + cur_l) * h ..][0..h];

            try inmem_pred.writeInputHiddenF16(x_cur, 1);
            const pr_in = try inmem_pred.predictAhead(@intCast(cur_l), 1, 16, null, &bitsets, &prefetches);
            inmem_zc = inmem_zc and pr_in.all_outputs_zero_copy;

            if (offline_pred) |op| {
                try op.writeInputHiddenF16(x_cur, 1);
                const pr_off = try op.predictAhead(@intCast(cur_l), 1, 16, null, &bitsets, &prefetches);
                offline_zc = offline_zc and pr_off.all_outputs_zero_copy;
            }

            for (0..pr_in.predicted_layers) |di| {
                const target_l = cur_l + 1 + di;
                const gt_topk = topk_trace[(s_idx * n_layers + target_l) * top_k ..][0..top_k];

                try inmem_pred.copyLastLogitsF32(@intCast(cur_l), @intCast(di), 0, logits_buf[0..n_exp]);
                deltas[di].ane_inmem_r8 += predictor_mod.computeRecallAtK(logits_buf[0..n_exp], gt_topk, 8);
                deltas[di].ane_inmem_r12 += predictor_mod.computeRecallAtK(logits_buf[0..n_exp], gt_topk, 12);
                deltas[di].ane_inmem_r16 += predictor_mod.computeRecallAtK(logits_buf[0..n_exp], gt_topk, 16);

                if (offline_pred) |op| {
                    try op.copyLastLogitsF32(@intCast(cur_l), @intCast(di), 0, logits_buf[0..n_exp]);
                    deltas[di].ane_offline_r8 += predictor_mod.computeRecallAtK(logits_buf[0..n_exp], gt_topk, 8);
                    deltas[di].ane_offline_r12 += predictor_mod.computeRecallAtK(logits_buf[0..n_exp], gt_topk, 12);
                    deltas[di].ane_offline_r16 += predictor_mod.computeRecallAtK(logits_buf[0..n_exp], gt_topk, 16);
                }
            }
        }
    }

    for (0..max_la) |di| {
        const denom: f64 = @floatFromInt(@max(deltas[di].samples, 1));
        deltas[di].ane_inmem_r8 /= denom;
        deltas[di].ane_inmem_r12 /= denom;
        deltas[di].ane_inmem_r16 /= denom;
        deltas[di].ane_offline_r8 /= denom;
        deltas[di].ane_offline_r12 /= denom;
        deltas[di].ane_offline_r16 /= denom;
        deltas[di].gpu_baseline_r8 /= denom;
        deltas[di].gpu_baseline_r12 /= denom;
        deltas[di].gpu_baseline_r16 /= denom;
    }

    // Time steady-state ANE prediction latency (`inmem` & optional `offline`).
    try inmem_pred.writeInputHiddenF16(hidden_trace[0..h], 1);
    for (0..4) |_| _ = try inmem_pred.predictAhead(0, 1, 16, null, &bitsets, &prefetches);
    const it0 = try vmstat.nowNs();
    for (0..opts.timed_iters) |_| _ = try inmem_pred.predictAhead(0, 1, 16, null, &bitsets, &prefetches);
    const it1 = try vmstat.nowNs();
    const inmem_ane_us = (@as(f64, @floatFromInt(it1 -| it0)) / @as(f64, @floatFromInt(opts.timed_iters))) / 1000.0;

    var offline_ane_us: f64 = 0.0;
    if (offline_pred) |op| {
        try op.writeInputHiddenF16(hidden_trace[0..h], 1);
        for (0..4) |_| _ = try op.predictAhead(0, 1, 16, null, &bitsets, &prefetches);
        const ot0 = try vmstat.nowNs();
        for (0..opts.timed_iters) |_| _ = try op.predictAhead(0, 1, 16, null, &bitsets, &prefetches);
        const ot1 = try vmstat.nowNs();
        offline_ane_us = (@as(f64, @floatFromInt(ot1 -| ot0)) / @as(f64, @floatFromInt(opts.timed_iters))) / 1000.0;
    }

    const vm_ane_after = try vmstat.Snapshot.capture();
    try vmstat.checkNoSwap(vm_ane_before, vm_ane_after);
    swapouts_total += vmstat.Snapshot.swapoutDelta(vm_ane_before, vm_ane_after);
    pageouts_total += vmstat.Snapshot.pageoutDelta(vm_ane_before, vm_ane_after);

    const gpu_la_us = if (gpu_lookahead_full_calls > 0)
        (@as(f64, @floatFromInt(gpu_lookahead_ns_sum)) / @as(f64, @floatFromInt(gpu_lookahead_full_calls))) / 1000.0
    else
        0.0;

    return .{
        .num_layers = cfg.num_hidden_layers,
        .num_experts = cfg.num_experts,
        .top_k = cfg.num_experts_per_tok,
        .hidden_size = cfg.hidden_size,
        .capture_steps = opts.capture_steps,
        .max_lookahead = max_la,
        .measured_gpu_layer_us = measured_gpu_layer_us,
        .inmem_compile_ms = inmem_pred.totalCompileMs(),
        .inmem_ane_us = inmem_ane_us,
        .inmem_uses_ne = inmem_pred.usesNeuralEngine(),
        .inmem_zero_copy = inmem_zc,
        .has_offline_mlpackage = offline_pred != null,
        .offline_compile_ms = if (offline_pred) |op| op.totalCompileMs() else 0.0,
        .offline_ane_us = offline_ane_us,
        .offline_uses_ne = if (offline_pred) |op| op.usesNeuralEngine() else false,
        .offline_zero_copy = offline_zc,
        .gpu_lookahead_us = gpu_la_us,
        .deltas = deltas,
        .swapouts_delta = swapouts_total,
        .pageouts_delta = pageouts_total,
    };
}

// ============================================================================
// M4.5 Bare-Metal ARMv9 SME2 Suite: Spike S7 + C13 Concurrent Bus Gate +
// ldnt1b L2 Preservation + L2-Resident Tree Router Calibration (`bench-m45`)
// ============================================================================

pub const BusContentionRow = struct {
    scenario: []const u8,
    sme_dtype: sme.PanelDType,
    sme_hint: sme.CacheHint,
    sme_m: u32,
    sme_n: u32,
    sme_k: u32,
    gpu_solo_gbs: f64,
    gpu_solo_us: f64,
    gpu_conc_gbs: f64,
    gpu_conc_us: f64,
    gpu_delta_pct: f64,
    sme_solo_us: f64,
    sme_solo_gbs: f64,
    sme_solo_gflops: f64,
    sme_conc_iters: u64,
    sme_conc_us: f64,
    sme_conc_gbs: f64,
    sme_conc_gflops: f64,
    aggregate_gbs: f64,
    net_speedup: f64,
    gate_pass: bool,
};

pub const LdntL2PreservationResult = struct {
    hot_bytes: usize,
    cold_total_bytes: usize,
    cold_experts: u32,
    hot_baseline_us: f64,
    hot_after_ld1_us: f64,
    hot_after_ldnt1_us: f64,
    cold_swiglu_ld1_us: f64,
    cold_swiglu_ldnt1_us: f64,
    eviction_overhead_ld1_pct: f64,
    eviction_overhead_ldnt1_pct: f64,
};

pub const RouterPredCalRow = struct {
    m: u32,
    sampled_layers: u32,
    num_experts: u32,
    hidden_size: u32,
    gpu_us: f64,
    sme2_us: f64,
    speedup: f64,
    topk_agreement: f64,
};

pub const M45Config = struct {
    run_full_s7: bool = true,
    // C13 Concurrent GPU + SME2 Bus Contention Sweep
    gpu_n: u32 = 4096,
    gpu_k: u32 = 4096,
    gpu_min_working_bytes: usize = 64 * 1024 * 1024,
    gpu_warmup_batches: u32 = 4,
    gpu_timed_batches: u32 = 24,
    gpu_dispatches_per_batch: u32 = 16,
    sme_l2_m: u32 = 16,
    sme_l2_n: u32 = 1024,
    sme_l2_k: u32 = 2048,
    sme_dram_m: u32 = 1,
    sme_dram_n: u32 = 4096,
    sme_dram_k: u32 = 4096,
    sme_dram_min_working_bytes: usize = 48 * 1024 * 1024,
    // C12/C13 ldnt1b L2 Preservation
    l2_pres_hot_m: u32 = 16,
    l2_pres_hot_n: u32 = 1024,
    l2_pres_hot_k: u32 = 2048,
    l2_pres_cold_h: u32 = 2048,
    l2_pres_cold_inter: u32 = 768,
    l2_pres_cold_experts: u32 = 6,
    l2_pres_reps: u32 = 5,
    // C13 L2-Resident Speculative Tree Router Calibration
    router_layers: u32 = 8,
    router_experts: u32 = 128,
    router_top_k: u32 = 8,
    router_hidden: u32 = 2048,
    router_moe_inter: u32 = 64,
    router_vocab: u32 = 256,
    router_reps: u32 = 9,
    timeout_ms: u64 = 30_000,
};

pub const M45Report = struct {
    caps: sme.SmeCaps,
    s7: ?sme.SpikeS7Report,
    bus_rows: [3]BusContentionRow,
    ldnt_l2: LdntL2PreservationResult,
    router_cal: [3]RouterPredCalRow,
    router_bank_bytes: u64,
    vm_before: vmstat.Snapshot,
    vm_after: vmstat.Snapshot,

    pub fn write(self: *const M45Report, out: *std.Io.Writer) std.Io.Writer.Error!void {
        try out.writeAll("\n=== M4.5 Bare-Metal ARMv9 SME2 Engine & C13 Concurrent Bus Suite ===\n\n");
        if (self.s7) |*s7| {
            try s7.write(out);
            try out.writeAll("\n");
        }

        try out.writeAll(
            \\--- C13 Concurrent Metal 4 GPU + CPU SME2 Unified-Bus Contention Gate (M4 120 GB/s) ---
            \\scenario              sme_shape         gpu_solo   gpu_conc  gpu_d% | sme_solo   sme_conc   agg_GB/s  net_spd  gate
            \\
        );
        for (self.bus_rows) |r| {
            var shp: [32]u8 = undefined;
            const s_str = std.fmt.bufPrint(&shp, "{s}:{d}x{d}x{d}", .{
                r.sme_dtype.label(),
                r.sme_m,
                r.sme_n,
                r.sme_k,
            }) catch unreachable;
            try out.print(
                \\{s: <21} {s: <17} {d: >6.2}GB/s {d: >6.2}GB/s {d: >6.1}% | {d: >7.1}us {d: >7.1}us {d: >8.2} {d: >7.2}x  {s}
                \\
            , .{
                r.scenario,
                s_str,
                r.gpu_solo_gbs,
                r.gpu_conc_gbs,
                r.gpu_delta_pct,
                r.sme_solo_us,
                r.sme_conc_us,
                r.aggregate_gbs,
                r.net_speedup,
                if (r.gate_pass) "PASS" else "FAIL",
            });
        }

        try out.print(
            \\
            \\--- C12/C13 ldnt1b Non-Temporal Cold-Expert L2 Cache Preservation ---
            \\ldnt_l2.hot_router_kib           {d:.1}
            \\ldnt_l2.cold_stream_mib          {d:.2} ({d} q4 SwiGLU experts)
            \\ldnt_l2.hot_baseline_us          {d:.2}
            \\ldnt_l2.hot_after_ld1_us         {d:.2} ({d:.1}% vs baseline)
            \\ldnt_l2.hot_after_ldnt1_us       {d:.2} ({d:.1}% vs baseline)
            \\ldnt_l2.cold_swiglu_ld1_us       {d:.2} /expert
            \\ldnt_l2.cold_swiglu_ldnt1_us     {d:.2} /expert
            \\
            \\--- C13 L2-Resident SME2 vs Metal 4 GPU Speculative Tree Router Prediction ---
            \\router_cal.bank_kib              {d:.1}
            \\M    shape (L x E x H)       gpu_us    sme2_us   speedup   topk_agree
            \\
        , .{
            @as(f64, @floatFromInt(self.ldnt_l2.hot_bytes)) / 1024.0,
            @as(f64, @floatFromInt(self.ldnt_l2.cold_total_bytes)) / (1024.0 * 1024.0),
            self.ldnt_l2.cold_experts,
            self.ldnt_l2.hot_baseline_us,
            self.ldnt_l2.hot_after_ld1_us,
            self.ldnt_l2.eviction_overhead_ld1_pct,
            self.ldnt_l2.hot_after_ldnt1_us,
            self.ldnt_l2.eviction_overhead_ldnt1_pct,
            self.ldnt_l2.cold_swiglu_ld1_us,
            self.ldnt_l2.cold_swiglu_ldnt1_us,
            @as(f64, @floatFromInt(self.router_bank_bytes)) / 1024.0,
        });
        for (self.router_cal) |rc| {
            var rshp: [28]u8 = undefined;
            const rs = std.fmt.bufPrint(&rshp, "{d}x{d}x{d}", .{
                rc.sampled_layers,
                rc.num_experts,
                rc.hidden_size,
            }) catch unreachable;
            try out.print(
                \\{d: <4} {s: <22} {d: >7.2}    {d: >7.2}   {d: >6.2}x   {d: >6.2}%
                \\
            , .{
                rc.m,
                rs,
                rc.gpu_us,
                rc.sme2_us,
                rc.speedup,
                rc.topk_agreement * 100.0,
            });
        }

        const swap_d = vmstat.Snapshot.swapoutDelta(self.vm_before, self.vm_after);
        const page_d = vmstat.Snapshot.pageoutDelta(self.vm_before, self.vm_after);
        try out.print(
            \\
            \\M4.5 Swap Check
            \\m45.swap.status                  ok
            \\m45.swap.swapouts_delta          {d}
            \\m45.swap.pageouts_delta          {d}
            \\
        , .{ swap_d, page_d });
    }
};

pub const M45Error = Error || sme.SpikeS7Error || qwen3.LoadError || qwen3.StepError || qos.Error;

fn fillPackedBenchBlob(
    allocator: std.mem.Allocator,
    dtype: sme.PanelDType,
    n: usize,
    k: usize,
    dst: []u8,
    seed: u64,
) M45Error!void {
    var st = seed ^ 0xA076_1D64_78BD_642F;
    if (!dtype.isQuantized()) {
        const tmp = try allocator.alloc(u16, n * k);
        defer allocator.free(tmp);
        const fdt: quant.FloatDType = if (dtype == .bf16) .bf16 else .f16;
        for (tmp) |*v| {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            const f: f32 = (@as(f32, @floatFromInt(st & 0xFFFF)) / 32768.0 - 1.0) * 0.125;
            v.* = quant.f32ToFloatBits(fdt, f);
        }
        try sme.packDensePanel64(dtype, tmp, n, k, dst);
    } else {
        const groups = k / sme.GROUP_SIZE;
        const mask: u8 = switch (dtype) {
            .q4_affine => 0x0F,
            .q3_affine => 0x07,
            .q2_affine => 0x03,
            else => unreachable,
        };
        const codes = try allocator.alloc(u8, n * k);
        defer allocator.free(codes);
        const sbs = try allocator.alloc(quant.ScaleBias, n * groups);
        defer allocator.free(sbs);
        for (codes) |*c| {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            c.* = @as(u8, @truncate(st)) & mask;
        }
        for (sbs) |*sb| {
            sb.* = .{ .scale = 0.03125, .bias = -0.125 };
        }
        try sme.packQuantAffinePanel64(dtype, codes, sbs, n, k, dst);
    }
}

fn medianF64Slice(samples: []f64) f64 {
    std.sort.pdq(f64, samples, {}, std.sort.asc(f64));
    return samples[samples.len / 2];
}

fn runConcurrentBusSweep(
    allocator: std.mem.Allocator,
    config: M45Config,
) M45Error![3]BusContentionRow {
    const facts = try sysinfo.Facts.read();
    const device = try metal.Device.open();
    defer device.close();
    const dev_info = try device.info();
    const wired_budget = budget_mod.Budget.fromHost(facts, dev_info, .{});

    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try dequant.DequantPipelines.init(device, library);
    defer pipes.close();

    const gpu_case: Case = .{ .kind = .gemv_q3, .m = 1, .n = config.gpu_n, .k = config.gpu_k };
    const gpu_layout = try computeCaseLayout(gpu_case);
    const w_stride = amf.align64(gpu_layout.w_bytes);
    const sb_stride = amf.align64(gpu_layout.sb_bytes);
    const weight_pool_bytes = amf.align64(@max(w_stride, config.gpu_min_working_bytes));
    const num_slots: usize = @max(@as(usize, 1), weight_pool_bytes / w_stride);
    const sb_pool_bytes = num_slots * sb_stride;
    const raw_arena = weight_pool_bytes + sb_pool_bytes + amf.align64(gpu_layout.x_bytes) + amf.align64(gpu_layout.y_bytes) + 4 * page.APPLE_PAGE_SIZE;
    const arena_bytes = try page.alignUp(raw_arena);

    var arena = try arena_mod.WiredArena.init(arena_bytes, wired_budget);
    defer arena.deinit();
    const w_pool = try arena.allocAligned(weight_pool_bytes, 64);
    const sb_pool = try arena.allocAligned(sb_pool_bytes, 64);
    const x_raw = try arena.allocAligned(gpu_layout.x_bytes, 64);
    const y_raw = try arena.allocAligned(gpu_layout.y_bytes, 64);
    fillBenchBuffers(w_pool, sb_pool, x_raw, y_raw);

    const w_base_off = arena.offsetOf(w_pool.ptr);
    const sb_base_off = arena.offsetOf(sb_pool.ptr);
    const x_off = arena.offsetOf(x_raw.ptr);
    const y_off = arena.offsetOf(y_raw.ptr);

    const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
    defer gpu_buf.close();
    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();
    try compute.makeResident(&.{gpu_buf});

    const runGpuWindow = struct {
        fn call(
            comp: *metal.Compute,
            dp: dequant.DequantPipelines,
            buf: metal.Buffer,
            wb: u64,
            sbb: u64,
            ws: usize,
            sbs: usize,
            ns: usize,
            xo: u64,
            yo: u64,
            gn: u32,
            gk: u32,
            batches: u32,
            disp_per: u32,
            timeout_ms: u64,
        ) M45Error!u64 {
            var cur: usize = 0;
            var last_sig: u64 = 0;
            const t0 = try vmstat.nowNs();
            for (0..batches) |_| {
                var frame = try comp.beginFrame(timeout_ms);
                errdefer frame.abort();
                for (0..disp_per) |d| {
                    const slot = cur % ns;
                    cur += 1;
                    try dp.encodeGemv(&frame, .q3, .{
                        .x = .{ .buffer = buf, .offset = xo },
                        .qweight = .{ .buffer = buf, .offset = wb + @as(u64, slot) * ws },
                        .scale_bias = .{ .buffer = buf, .offset = sbb + @as(u64, slot) * sbs },
                        .y = .{ .buffer = buf, .offset = yo },
                        .n = gn,
                        .k = gk,
                    }, d > 0);
                }
                last_sig = try frame.commit();
            }
            if (last_sig != 0) try comp.wait(last_sig, timeout_ms);
            const t1 = try vmstat.nowNs();
            return t1 -| t0;
        }
    }.call;

    // Warmup GPU DVFS
    _ = try runGpuWindow(&compute, pipes, gpu_buf, w_base_off, sb_base_off, w_stride, sb_stride, num_slots, x_off, y_off, config.gpu_n, config.gpu_k, config.gpu_warmup_batches, config.gpu_dispatches_per_batch, config.timeout_ms);

    const gpu_disps: u64 = @as(u64, config.gpu_timed_batches) * config.gpu_dispatches_per_batch;
    const gpu_total_bytes: f64 = @as(f64, @floatFromInt(gpu_layout.bytes_per_dispatch)) * @as(f64, @floatFromInt(gpu_disps));

    // Measure GPU solo baseline (median of 3 windows)
    var gpu_solo_ns_arr: [3]f64 = undefined;
    for (0..3) |rep| {
        const ns = try runGpuWindow(&compute, pipes, gpu_buf, w_base_off, sb_base_off, w_stride, sb_stride, num_slots, x_off, y_off, config.gpu_n, config.gpu_k, config.gpu_timed_batches, config.gpu_dispatches_per_batch, config.timeout_ms);
        gpu_solo_ns_arr[rep] = @floatFromInt(@max(ns, 1));
    }
    const gpu_solo_ns = medianF64Slice(&gpu_solo_ns_arr);
    const gpu_solo_gbs = gpu_total_bytes / gpu_solo_ns;
    const gpu_solo_us = (gpu_solo_ns / @as(f64, @floatFromInt(gpu_disps))) / 1000.0;

    // Prepare persistent SME2 dispatcher
    const max_sme_k: usize = @max(@as(usize, config.sme_l2_k), @as(usize, config.sme_dram_k));
    var disp = try sme.SmeDispatcher.init(allocator, .{
        .a_pack_u16_cap = 32 * @max(max_sme_k, sme.GROUP_SIZE),
        .x_sums_f32_cap = @max(@as(usize, 64), 4 * (@max(max_sme_k, sme.GROUP_SIZE) / sme.GROUP_SIZE)),
    });
    defer disp.deinit();

    const ScenarioSpec = struct {
        name: []const u8,
        dtype: sme.PanelDType,
        hint: sme.CacheHint,
        m: u32,
        n: u32,
        k: u32,
        min_working_bytes: usize,
    };
    const scenarios = [_]ScenarioSpec{
        .{
            .name = "gpu+sme2_l2_router",
            .dtype = .bf16,
            .hint = .temporal,
            .m = config.sme_l2_m,
            .n = config.sme_l2_n,
            .k = config.sme_l2_k,
            .min_working_bytes = 0,
        },
        .{
            .name = "gpu+sme2_dram_ld1",
            .dtype = .q4_affine,
            .hint = .temporal,
            .m = config.sme_dram_m,
            .n = config.sme_dram_n,
            .k = config.sme_dram_k,
            .min_working_bytes = config.sme_dram_min_working_bytes,
        },
        .{
            .name = "gpu+sme2_dram_ldnt1",
            .dtype = .q4_affine,
            .hint = .non_temporal,
            .m = config.sme_dram_m,
            .n = config.sme_dram_n,
            .k = config.sme_dram_k,
            .min_working_bytes = config.sme_dram_min_working_bytes,
        },
    };

    const WorkerCtx = struct {
        disp: *sme.SmeDispatcher,
        cfg: sme.GemmConfig,
        a: []const u16,
        w_ring: []const u8,
        w_stride: usize,
        slabs: usize,
        out: []f32,
        m: usize,
        n: usize,
        k: usize,
        ready: std.atomic.Value(bool) = .init(false),
        stop: std.atomic.Value(bool) = .init(false),
        iters: std.atomic.Value(u64) = .init(0),
        err: ?sme.Error = null,
        qos_err: ?qos.Error = null,

        fn mainFn(ctx: *@This()) void {
            qos.steerPerformanceCore(.user_interactive) catch |e| {
                ctx.qos_err = e;
                ctx.ready.store(true, .release);
                return;
            };
            // Warmup one call so thread is active on a P-core before signalling ready.
            ctx.disp.gemmF32(ctx.cfg, ctx.a, ctx.w_ring[0..ctx.w_stride], null, ctx.out, ctx.m, ctx.n, ctx.k) catch |e| {
                ctx.err = e;
                ctx.ready.store(true, .release);
                return;
            };
            ctx.ready.store(true, .release);

            var idx: usize = 1;
            var local_iters: u64 = 0;
            while (!ctx.stop.load(.acquire)) {
                const slab = idx % ctx.slabs;
                ctx.disp.gemmF32(ctx.cfg, ctx.a, ctx.w_ring[slab * ctx.w_stride ..][0..ctx.w_stride], null, ctx.out, ctx.m, ctx.n, ctx.k) catch |e| {
                    ctx.err = e;
                    break;
                };
                idx += 1;
                local_iters += 1;
            }
            ctx.iters.store(local_iters, .release);
        }
    };

    var rows: [3]BusContentionRow = undefined;
    for (scenarios, 0..) |sc, si| {
        const m: usize = sc.m;
        const n: usize = sc.n;
        const k: usize = sc.k;
        const w_bytes = try sme.packedBlobBytes(sc.dtype, n, k);
        const slabs: usize = if (sc.min_working_bytes == 0)
            1
        else
            @max(@as(usize, 2), (sc.min_working_bytes + w_bytes - 1) / w_bytes);
        const w_ring = try allocator.alignedAlloc(u8, .fromByteUnits(sme.SVL_BYTES), slabs * w_bytes);
        defer allocator.free(w_ring);
        try fillPackedBenchBlob(allocator, sc.dtype, n, k, w_ring[0..w_bytes], 0x45B0 + si);
        for (1..slabs) |s_idx| {
            @memcpy(w_ring[s_idx * w_bytes ..][0..w_bytes], w_ring[0..w_bytes]);
        }

        const a_u16 = try allocator.alignedAlloc(u16, .fromByteUnits(sme.SVL_BYTES), m * k);
        defer allocator.free(a_u16);
        const act_dt: quant.FloatDType = if (sc.dtype.activationIsBf16()) .bf16 else .f16;
        for (a_u16, 0..) |*v, i| {
            v.* = quant.f32ToFloatBits(act_dt, (@as(f32, @floatFromInt(i % 13)) - 6.0) * 0.0625);
        }
        const out_f32 = try allocator.alignedAlloc(f32, .fromByteUnits(sme.SVL_BYTES), m * n);
        defer allocator.free(out_f32);

        const gcfg: sme.GemmConfig = .{ .dtype = sc.dtype, .cache_hint = sc.hint };
        // Measure SME2 solo baseline
        for (0..3) |_| {
            try disp.gemmF32(gcfg, a_u16, w_ring[0..w_bytes], null, out_f32, m, n, k);
        }
        const solo_reps: usize = if (slabs == 1) 15 else 7;
        var solo_samples: [15]f64 = undefined;
        for (0..solo_reps) |r| {
            const slab = r % slabs;
            const t0 = try vmstat.nowNs();
            try disp.gemmF32(gcfg, a_u16, w_ring[slab * w_bytes ..][0..w_bytes], null, out_f32, m, n, k);
            const t1 = try vmstat.nowNs();
            solo_samples[r] = @as(f64, @floatFromInt(@max(t1 -| t0, 1))) / 1000.0;
        }
        const sme_solo_us = medianF64Slice(solo_samples[0..solo_reps]);
        const sme_bytes: f64 = @floatFromInt(w_bytes + m * k * 2 + m * n * 4);
        const sme_flops: f64 = 2.0 * @as(f64, @floatFromInt(m)) * @as(f64, @floatFromInt(n)) * @as(f64, @floatFromInt(k));
        const sme_solo_gbs = sme_bytes / (sme_solo_us * 1000.0);
        const sme_solo_gflops = sme_flops / (sme_solo_us * 1000.0);

        // Concurrent window: SME2 thread runs continuously while GPU executes timed window.
        var wctx: WorkerCtx = .{
            .disp = &disp,
            .cfg = gcfg,
            .a = a_u16,
            .w_ring = w_ring,
            .w_stride = w_bytes,
            .slabs = slabs,
            .out = out_f32,
            .m = m,
            .n = n,
            .k = k,
        };
        const th = try std.Thread.spawn(.{}, WorkerCtx.mainFn, .{&wctx});
        while (!wctx.ready.load(.acquire)) {
            std.atomic.spinLoopHint();
        }
        const conc_ns = runGpuWindow(&compute, pipes, gpu_buf, w_base_off, sb_base_off, w_stride, sb_stride, num_slots, x_off, y_off, config.gpu_n, config.gpu_k, config.gpu_timed_batches, config.gpu_dispatches_per_batch, config.timeout_ms) catch |e| {
            wctx.stop.store(true, .release);
            th.join();
            return e;
        };
        wctx.stop.store(true, .release);
        th.join();
        if (wctx.qos_err) |qe| return qe;
        if (wctx.err) |we| return we;

        const conc_ns_f: f64 = @floatFromInt(@max(conc_ns, 1));
        const gpu_conc_gbs = gpu_total_bytes / conc_ns_f;
        const gpu_conc_us = (conc_ns_f / @as(f64, @floatFromInt(gpu_disps))) / 1000.0;
        const gpu_delta_pct = (gpu_conc_gbs / gpu_solo_gbs - 1.0) * 100.0;

        const sme_iters = wctx.iters.load(.acquire);
        const sme_conc_us: f64 = if (sme_iters > 0) (conc_ns_f / @as(f64, @floatFromInt(sme_iters))) / 1000.0 else 0.0;
        const sme_conc_gbs: f64 = if (sme_iters > 0) (sme_bytes * @as(f64, @floatFromInt(sme_iters))) / conc_ns_f else 0.0;
        const sme_conc_gflops: f64 = if (sme_iters > 0) (sme_flops * @as(f64, @floatFromInt(sme_iters))) / conc_ns_f else 0.0;

        // Combined heterogeneous work-rate speedup:
        // (T_seq = T_gpu_solo + N_sme * T_sme_solo) / T_conc
        // = (gpu_conc_gbs / gpu_solo_gbs) + (sme_conc_gflops / sme_solo_gflops)
        const net_speedup = (gpu_conc_gbs / gpu_solo_gbs) +
            (if (sme_solo_gflops > 0.0) sme_conc_gflops / sme_solo_gflops else 0.0);

        rows[si] = .{
            .scenario = sc.name,
            .sme_dtype = sc.dtype,
            .sme_hint = sc.hint,
            .sme_m = sc.m,
            .sme_n = sc.n,
            .sme_k = sc.k,
            .gpu_solo_gbs = gpu_solo_gbs,
            .gpu_solo_us = gpu_solo_us,
            .gpu_conc_gbs = gpu_conc_gbs,
            .gpu_conc_us = gpu_conc_us,
            .gpu_delta_pct = gpu_delta_pct,
            .sme_solo_us = sme_solo_us,
            .sme_solo_gbs = sme_solo_gbs,
            .sme_solo_gflops = sme_solo_gflops,
            .sme_conc_iters = sme_iters,
            .sme_conc_us = sme_conc_us,
            .sme_conc_gbs = sme_conc_gbs,
            .sme_conc_gflops = sme_conc_gflops,
            .aggregate_gbs = gpu_conc_gbs + sme_conc_gbs,
            .net_speedup = net_speedup,
            .gate_pass = (net_speedup > 1.05),
        };
    }

    return rows;
}

fn runLdntL2PreservationBench(
    allocator: std.mem.Allocator,
    config: M45Config,
) M45Error!LdntL2PreservationResult {
    const hot_m: usize = config.l2_pres_hot_m;
    const hot_n: usize = config.l2_pres_hot_n;
    const hot_k: usize = config.l2_pres_hot_k;
    const cold_h: usize = config.l2_pres_cold_h;
    const cold_inter: usize = config.l2_pres_cold_inter;
    const cold_cnt: usize = @max(@as(usize, 1), config.l2_pres_cold_experts);

    const max_k = @max(hot_k, @max(cold_h, cold_inter));
    var disp = try sme.SmeDispatcher.init(allocator, .{
        .a_pack_u16_cap = 32 * @max(max_k, sme.GROUP_SIZE),
        .x_sums_f32_cap = @max(@as(usize, 64), 4 * (@max(max_k, sme.GROUP_SIZE) / sme.GROUP_SIZE)),
    });
    defer disp.deinit();

    const hot_w_bytes = try sme.packedBlobBytes(.bf16, hot_n, hot_k);
    const hot_w = try allocator.alignedAlloc(u8, .fromByteUnits(sme.SVL_BYTES), hot_w_bytes);
    defer allocator.free(hot_w);
    try fillPackedBenchBlob(allocator, .bf16, hot_n, hot_k, hot_w, 0x4512);

    const hot_a = try allocator.alignedAlloc(u16, .fromByteUnits(sme.SVL_BYTES), hot_m * hot_k);
    defer allocator.free(hot_a);
    for (hot_a, 0..) |*v, i| v.* = quant.f32ToBf16((@as(f32, @floatFromInt(i % 11)) - 5.0) * 0.0625);
    const hot_out = try allocator.alignedAlloc(f32, .fromByteUnits(sme.SVL_BYTES), hot_m * hot_n);
    defer allocator.free(hot_out);

    const gu_bytes = try sme.packedBlobBytes(.q4_affine, cold_inter, cold_h);
    const dn_bytes = try sme.packedBlobBytes(.q4_affine, cold_h, cold_inter);
    const one_exp_bytes = 2 * gu_bytes + dn_bytes;
    const cold_slab = try allocator.alignedAlloc(u8, .fromByteUnits(sme.SVL_BYTES), cold_cnt * one_exp_bytes);
    defer allocator.free(cold_slab);
    try fillPackedBenchBlob(allocator, .q4_affine, cold_inter, cold_h, cold_slab[0..gu_bytes], 0x4513);
    try fillPackedBenchBlob(allocator, .q4_affine, cold_inter, cold_h, cold_slab[gu_bytes .. 2 * gu_bytes], 0x4514);
    try fillPackedBenchBlob(allocator, .q4_affine, cold_h, cold_inter, cold_slab[2 * gu_bytes .. one_exp_bytes], 0x4515);
    for (1..cold_cnt) |ei| {
        @memcpy(cold_slab[ei * one_exp_bytes ..][0..one_exp_bytes], cold_slab[0..one_exp_bytes]);
    }

    const cold_x = try allocator.alignedAlloc(u16, .fromByteUnits(sme.SVL_BYTES), cold_h);
    defer allocator.free(cold_x);
    for (cold_x, 0..) |*v, i| v.* = quant.f32ToFloatBits(.f16, (@as(f32, @floatFromInt(i % 9)) - 4.0) * 0.0625);
    const up_scr = try allocator.alignedAlloc(f32, .fromByteUnits(sme.SVL_BYTES), cold_inter);
    defer allocator.free(up_scr);
    const inter_scr = try allocator.alignedAlloc(u16, .fromByteUnits(sme.SVL_BYTES), cold_inter);
    defer allocator.free(inter_scr);
    const dn_out = try allocator.alignedAlloc(f32, .fromByteUnits(sme.SVL_BYTES), cold_h);
    defer allocator.free(dn_out);

    const reps: usize = @min(@max(@as(usize, 1), config.l2_pres_reps), 9);
    var base_samples: [9]f64 = undefined;
    var ld1_samples: [9]f64 = undefined;
    var ldnt1_samples: [9]f64 = undefined;
    var swiglu_ld1_samples: [9]f64 = undefined;
    var swiglu_ldnt1_samples: [9]f64 = undefined;

    const hot_cfg: sme.GemmConfig = .{ .dtype = .bf16, .cache_hint = .temporal };
    for (0..3) |_| {
        try disp.gemmF32(hot_cfg, hot_a, hot_w, null, hot_out, hot_m, hot_n, hot_k);
    }

    for (0..reps) |r| {
        // Warm hot router into L2, measure L2-hot baseline.
        try disp.gemmF32(hot_cfg, hot_a, hot_w, null, hot_out, hot_m, hot_n, hot_k);
        const tb0 = try vmstat.nowNs();
        try disp.gemmF32(hot_cfg, hot_a, hot_w, null, hot_out, hot_m, hot_n, hot_k);
        const tb1 = try vmstat.nowNs();
        base_samples[r] = @as(f64, @floatFromInt(@max(tb1 -| tb0, 1))) / 1000.0;

        // Sweep cold experts with temporal `ld1b`, then immediately measure hot router.
        const tc0 = try vmstat.nowNs();
        for (0..cold_cnt) |ei| {
            const base = ei * one_exp_bytes;
            try qwen3.Qwen3Model.evalSwiGluExpertSme2(
                &disp,
                .q4_affine,
                .affine_scale_bias,
                .temporal,
                cold_slab[base .. base + gu_bytes],
                cold_slab[base + gu_bytes .. base + 2 * gu_bytes],
                cold_slab[base + 2 * gu_bytes .. base + one_exp_bytes],
                cold_x,
                up_scr,
                inter_scr,
                dn_out,
                null,
                1,
                cold_h,
                cold_inter,
            );
        }
        const tc1 = try vmstat.nowNs();
        swiglu_ld1_samples[r] = (@as(f64, @floatFromInt(@max(tc1 -| tc0, 1))) / @as(f64, @floatFromInt(cold_cnt))) / 1000.0;

        const ta0 = try vmstat.nowNs();
        try disp.gemmF32(hot_cfg, hot_a, hot_w, null, hot_out, hot_m, hot_n, hot_k);
        const ta1 = try vmstat.nowNs();
        ld1_samples[r] = @as(f64, @floatFromInt(@max(ta1 -| ta0, 1))) / 1000.0;

        // Re-warm hot router into L2, then sweep cold experts with non-temporal `ldnt1b`.
        try disp.gemmF32(hot_cfg, hot_a, hot_w, null, hot_out, hot_m, hot_n, hot_k);
        const tnt0 = try vmstat.nowNs();
        for (0..cold_cnt) |ei| {
            const base = ei * one_exp_bytes;
            try qwen3.Qwen3Model.evalSwiGluExpertSme2(
                &disp,
                .q4_affine,
                .affine_scale_bias,
                .non_temporal,
                cold_slab[base .. base + gu_bytes],
                cold_slab[base + gu_bytes .. base + 2 * gu_bytes],
                cold_slab[base + 2 * gu_bytes .. base + one_exp_bytes],
                cold_x,
                up_scr,
                inter_scr,
                dn_out,
                null,
                1,
                cold_h,
                cold_inter,
            );
        }
        const tnt1 = try vmstat.nowNs();
        swiglu_ldnt1_samples[r] = (@as(f64, @floatFromInt(@max(tnt1 -| tnt0, 1))) / @as(f64, @floatFromInt(cold_cnt))) / 1000.0;

        const tn0 = try vmstat.nowNs();
        try disp.gemmF32(hot_cfg, hot_a, hot_w, null, hot_out, hot_m, hot_n, hot_k);
        const tn1 = try vmstat.nowNs();
        ldnt1_samples[r] = @as(f64, @floatFromInt(@max(tn1 -| tn0, 1))) / 1000.0;
    }

    const base_us = medianF64Slice(base_samples[0..reps]);
    const after_ld1 = medianF64Slice(ld1_samples[0..reps]);
    const after_ldnt1 = medianF64Slice(ldnt1_samples[0..reps]);

    return .{
        .hot_bytes = hot_w_bytes,
        .cold_total_bytes = cold_cnt * one_exp_bytes,
        .cold_experts = @intCast(cold_cnt),
        .hot_baseline_us = base_us,
        .hot_after_ld1_us = after_ld1,
        .hot_after_ldnt1_us = after_ldnt1,
        .cold_swiglu_ld1_us = medianF64Slice(swiglu_ld1_samples[0..reps]),
        .cold_swiglu_ldnt1_us = medianF64Slice(swiglu_ldnt1_samples[0..reps]),
        .eviction_overhead_ld1_pct = (after_ld1 / base_us - 1.0) * 100.0,
        .eviction_overhead_ldnt1_pct = (after_ldnt1 / base_us - 1.0) * 100.0,
    };
}

fn runRouterPredCalibrationBench(
    allocator: std.mem.Allocator,
    config: M45Config,
) M45Error!struct { rows: [3]RouterPredCalRow, bank_bytes: u64 } {
    const cfg: qwen3.ModelConfig = .{
        .arch = .qwen3_moe,
        .quant_bits = .q3,
        .default_float_dtype = .bf16,
        .vocab_size = config.router_vocab,
        .hidden_size = config.router_hidden,
        .num_hidden_layers = config.router_layers,
        .num_attention_heads = 4,
        .num_key_value_heads = 2,
        .head_dim = 64,
        .moe_intermediate_size = config.router_moe_inter,
        .num_experts = config.router_experts,
        .num_experts_per_tok = config.router_top_k,
        .norm_topk_prob = true,
    };
    const img = try qwen3.buildSyntheticAmfImageWithSpec(allocator, cfg, .{
        .linear_dtype = .q3_affine,
        .embed_dtype = .q3_affine,
        .norm_dtype = .bf16,
        .router_dtype = .q3_affine,
        .expert_dtype = .q3_affine,
    }, 0x45C13A);
    defer allocator.free(img);

    var model = try qwen3.Qwen3Model.loadFromBytes(allocator, img, cfg, .{
        .max_seq_len = 32,
        .chunk_tokens = 16,
        .max_tree_nodes = 16,
        .verify_crcs = false,
        .sme_router_mode = .auto,
    });
    defer model.deinit(allocator);

    const cal = model.routerPredCalibration();
    const ms = [_]u32{ 1, 8, 16 };
    const reps: usize = @min(@max(@as(usize, 1), config.router_reps), 15);
    var gpu_bs: [16 * qwen3.MAX_EXPERT_PRED_LAYERS]tree_mod.ExpertBitset = undefined;
    var sme_bs: [16 * qwen3.MAX_EXPERT_PRED_LAYERS]tree_mod.ExpertBitset = undefined;
    var rows: [3]RouterPredCalRow = undefined;

    for (ms, 0..) |m_u32, mi| {
        const m: usize = m_u32;
        var toks: [16]u32 = undefined;
        for (0..m) |i| toks[i] = @intCast((i * 17 + 3) % cfg.vocab_size);

        // Warmup + correctness overlap check
        const sg = try model.predictMoEExpertsForTokensGpu(toks[0..m], &gpu_bs);
        const ss = try model.predictMoEExpertsForTokensSme2(toks[0..m], &sme_bs);
        var overlap: usize = 0;
        const total = m * @as(usize, ss) * cfg.num_experts_per_tok;
        for (0..m * ss) |i| {
            overlap += gpu_bs[i].intersectWith(sme_bs[i]).count();
        }
        const agree = if (total > 0) @as(f64, @floatFromInt(overlap)) / @as(f64, @floatFromInt(total)) else 1.0;

        var gpu_samples: [15]f64 = undefined;
        var sme_samples: [15]f64 = undefined;
        for (0..reps) |r| {
            const tg0 = try vmstat.nowNs();
            _ = try model.predictMoEExpertsForTokensGpu(toks[0..m], &gpu_bs);
            const tg1 = try vmstat.nowNs();
            gpu_samples[r] = @as(f64, @floatFromInt(@max(tg1 -| tg0, 1))) / 1000.0;

            const ts0 = try vmstat.nowNs();
            _ = try model.predictMoEExpertsForTokensSme2(toks[0..m], &sme_bs);
            const ts1 = try vmstat.nowNs();
            sme_samples[r] = @as(f64, @floatFromInt(@max(ts1 -| ts0, 1))) / 1000.0;
        }
        const g_us = medianF64Slice(gpu_samples[0..reps]);
        const s_us = medianF64Slice(sme_samples[0..reps]);
        rows[mi] = .{
            .m = m_u32,
            .sampled_layers = sg,
            .num_experts = cfg.num_experts,
            .hidden_size = cfg.hidden_size,
            .gpu_us = g_us,
            .sme2_us = s_us,
            .speedup = g_us / s_us,
            .topk_agreement = agree,
        };
    }
    return .{ .rows = rows, .bank_bytes = cal.bank_bytes };
}

pub fn runM45WithConfig(allocator: std.mem.Allocator, config: M45Config) M45Error!M45Report {
    const caps = try sme.SmeCaps.probe();
    const vm_before = try vmstat.Snapshot.capture();

    const s7: ?sme.SpikeS7Report = if (config.run_full_s7)
        try sme.runSpikeS7Standalone(allocator)
    else
        null;

    const bus_rows = try runConcurrentBusSweep(allocator, config);
    const ldnt_l2 = try runLdntL2PreservationBench(allocator, config);
    const rcal = try runRouterPredCalibrationBench(allocator, config);

    const vm_after = try vmstat.Snapshot.capture();
    try vmstat.checkNoSwap(vm_before, vm_after);

    return .{
        .caps = caps,
        .s7 = s7,
        .bus_rows = bus_rows,
        .ldnt_l2 = ldnt_l2,
        .router_cal = rcal.rows,
        .router_bank_bytes = rcal.bank_bytes,
        .vm_before = vm_before,
        .vm_after = vm_after,
    };
}

pub fn runM45(allocator: std.mem.Allocator) M45Error!M45Report {
    return runM45WithConfig(allocator, .{});
}

// ---------------------------------------------------------------------------
// M5 Serving, Pipelining, C6 N-Gram & ADR-0017 GSQ+RCO Hybrid Quant Suite
// ---------------------------------------------------------------------------

pub const DEFAULT_M5_Q2_CASES = [_]Case{
    .{ .kind = .gemv_q2, .m = 1, .n = 2048, .k = 2048 },
    .{ .kind = .gemv_q2, .m = 1, .n = 6144, .k = 2048 },
    .{ .kind = .gemv_q2, .m = 1, .n = 2048, .k = 768 },
    .{ .kind = .gemv_q2, .m = 1, .n = 4096, .k = 4096 },
    .{ .kind = .gemm_q2, .m = 16, .n = 2048, .k = 2048 },
    .{ .kind = .gemm_q2, .m = 16, .n = 4096, .k = 4096 },
    .{ .kind = .gemm_q2, .m = 64, .n = 2048, .k = 2048 },
    .{ .kind = .gemm_q2, .m = 64, .n = 4096, .k = 4096 },
};

pub const M5Config = struct {
    q2_cases: []const Case = &DEFAULT_M5_Q2_CASES,
    q2_min_working_set_bytes: usize = 64 * 1024 * 1024,
    q2_warmup_batches: u32 = 4,
    q2_gemv_timed_batches: u32 = 8,
    q2_gemv_dispatches_per_batch: u32 = 32,
    q2_gemm_timed_batches: u32 = 4,
    q2_gemm_dispatches_per_batch: u32 = 16,

    hetero_moe_hidden: u32 = 2048,
    hetero_moe_inter: u32 = 768,
    hetero_moe_experts: u32 = 128,
    hetero_moe_top_k: u32 = 8,
    hetero_moe_shared_m8: u32 = 24,
    hetero_moe_warmup_batches: u32 = 4,
    hetero_moe_timed_batches: u32 = 8,
    hetero_moe_dispatches_per_batch: u32 = 8,

    ngram_temp_dir: []const u8 = "/Volumes/Akari/.tmp",
    ngram_buckets: u32 = 256,
    ngram_entries_per_bucket: u32 = 4,
    ngram_cache_pages: usize = 32,
    ngram_cold_queries: usize = 64,
    ngram_cache_queries: usize = 256,

    pearl_layers: u32 = 4,
    pearl_hidden: u32 = 512,
    pearl_inter: u32 = 1536,
    pearl_vocab: u32 = 256,
    pearl_gen_tokens: usize = 20,

    server_health_iters: usize = 64,
    server_steady_iters: usize = 16,
    server_concurrent_clients: usize = 8,
    server_tokens_per_req: u32 = 16,

    timeout_ms: u64 = 30_000,
};

pub const HeteroMoeRow = struct {
    label: []const u8,
    m: u32,
    num_experts: u32,
    top_k: u32,
    unique_experts: u32,
    active_buckets: u32,
    bytes_per_disp: u64,
    latency_us: f64,
    effective_gbs: f64,
    bus_pct: f64,
    vs_q4_bytes_pct: f64,
};

pub const NgramBenchResult = struct {
    file_bytes: u64,
    num_buckets: u32,
    total_entries: u64,
    wired_kib: u64,
    cold_nvme_p50_us: f64,
    cold_nvme_p99_us: f64,
    cache_hit_p50_us: f64,
    prefetch_hit_p50_us: f64,
    queries: u64,
    hits: u64,
    seed_hits: u64,
    nvme_reads: u64,
};

pub const PearlSynthBenchResult = struct {
    has_own_queue: bool,
    bit_exact_off: bool,
    bit_exact_on: bool,
    base_tok_per_s: f64,
    off_tok_per_s: f64,
    on_tok_per_s: f64,
    mean_accept_len: f64,
    pearl_rounds: u64,
    pearl_hits: u64,
    pearl_rollbacks: u64,
    pearl_saved_ms: f64,
};

pub const ServerBenchResult = struct {
    health_p50_us: f64,
    single_ttft_p50_us: f64,
    single_e2e_p50_us: f64,
    concurrent_clients: u32,
    concurrent_ttft_p50_us: f64,
    concurrent_ttft_p99_us: f64,
    concurrent_tok_per_s: f64,
    steady_malloc_blocks_delta: i64,
    steady_malloc_bytes_delta: i64,
    all_ok: bool,
};

pub const M5Report = struct {
    q2_cases: []CaseResult,
    hetero_moe_rows: [8]HeteroMoeRow,
    ngram: NgramBenchResult,
    pearl: PearlSynthBenchResult,
    server: ServerBenchResult,
    vm_before: vmstat.Snapshot,
    vm_after: vmstat.Snapshot,

    pub fn deinit(self: *M5Report, allocator: std.mem.Allocator) void {
        allocator.free(self.q2_cases);
        self.q2_cases = &.{};
    }

    pub fn write(self: *const M5Report, out: *std.Io.Writer) std.Io.Writer.Error!void {
        try out.print(
            \\=== M5 Serving, Pipelining, C6 NVMe N-Gram & ADR-0017 GSQ+RCO Hybrid Quant Suite ===
            \\
            \\--- G13 Metal 4 q2_affine Kernel Bandwidth Calibration ---
            \\kernel    shape              bytes/disp   slots   latency(us)   beta(GB/s)   bus(%)
            \\
        , .{});
        for (self.q2_cases) |res| {
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
            \\--- G13 Scheme-Bucketed Heterogeneous MoE ({{q2, q3, q4}}) vs Uniform CSR Gather ---
            \\profile          m    E    k   U    bkts    bytes/disp   vs_q4(%)   latency(us)   beta(GB/s)   bus(%)
            \\
        , .{});
        for (self.hetero_moe_rows) |row| {
            try out.print(
                \\{s: <16} {d: <4} {d: <4} {d: <3} {d: <4} {d: <7} {d: >10}   {d: >8.1}   {d: >11.2}   {d: >10.2}   {d: >6.1}
                \\
            , .{
                row.label,
                row.m,
                row.num_experts,
                row.top_k,
                row.unique_experts,
                row.active_buckets,
                row.bytes_per_disp,
                row.vs_q4_bytes_pct,
                row.latency_us,
                row.effective_gbs,
                row.bus_pct,
            });
        }

        try out.print(
            \\
            \\--- C6 F_NOCACHE NVMe .ang N-Gram Lookup Latency ---
            \\ngram.file_bytes                 {d}
            \\ngram.buckets                    {d}
            \\ngram.total_entries              {d}
            \\ngram.wired_cache_kib            {d}
            \\ngram.cold_nvme_p50_us           {d:.2}
            \\ngram.cold_nvme_p99_us           {d:.2}
            \\ngram.cache_hit_p50_us           {d:.2}
            \\ngram.prefetch_hit_p50_us        {d:.2}
            \\ngram.queries                    {d}
            \\ngram.hits                       {d}
            \\ngram.seed_hits                  {d}
            \\ngram.nvme_reads                 {d}
            \\
            \\--- D8 PEARL Async Pipelined Small-Draft Synthetic Parity & Overlap ---
            \\pearl_synth.own_compute_queue    {s}
            \\pearl_synth.bit_exact_off        {s}
            \\pearl_synth.bit_exact_on         {s}
            \\pearl_synth.base_tok_per_s       {d:.2}
            \\pearl_synth.off_tok_per_s        {d:.2}
            \\pearl_synth.on_tok_per_s         {d:.2}
            \\pearl_synth.mean_accept_len      {d:.3}
            \\pearl_synth.pearl_rounds         {d}
            \\pearl_synth.pearl_hits           {d}
            \\pearl_synth.pearl_rollbacks      {d}
            \\pearl_synth.pearl_saved_ms       {d:.3}
            \\
        , .{
            self.ngram.file_bytes,
            self.ngram.num_buckets,
            self.ngram.total_entries,
            self.ngram.wired_kib,
            self.ngram.cold_nvme_p50_us,
            self.ngram.cold_nvme_p99_us,
            self.ngram.cache_hit_p50_us,
            self.ngram.prefetch_hit_p50_us,
            self.ngram.queries,
            self.ngram.hits,
            self.ngram.seed_hits,
            self.ngram.nvme_reads,
            if (self.pearl.has_own_queue) "yes" else "no",
            if (self.pearl.bit_exact_off) "yes" else "no",
            if (self.pearl.bit_exact_on) "yes" else "no",
            self.pearl.base_tok_per_s,
            self.pearl.off_tok_per_s,
            self.pearl.on_tok_per_s,
            self.pearl.mean_accept_len,
            self.pearl.pearl_rounds,
            self.pearl.pearl_hits,
            self.pearl.pearl_rollbacks,
            self.pearl.pearl_saved_ms,
        });
        try out.print(
            \\
            \\--- C7 + C8 kqueue HTTP/1.1 + SSE Server Loopback Microbenchmark ---
            \\server.health_p50_us             {d:.2}
            \\server.single_ttft_p50_us        {d:.2}
            \\server.single_e2e_p50_us         {d:.2}
            \\server.concurrent_clients        {d}
            \\server.concurrent_ttft_p50_us    {d:.2}
            \\server.concurrent_ttft_p99_us    {d:.2}
            \\server.concurrent_tok_per_s      {d:.1}
            \\server.steady_malloc_blocks      {d}
            \\server.steady_malloc_bytes       {d}
            \\server.all_ok                    {s}
            \\
            \\M5 Swap Check
            \\m5.swap.status                   ok
            \\m5.swap.swapouts_delta           {d}
            \\m5.swap.pageouts_delta           {d}
            \\
        , .{
            self.server.health_p50_us,
            self.server.single_ttft_p50_us,
            self.server.single_e2e_p50_us,
            self.server.concurrent_clients,
            self.server.concurrent_ttft_p50_us,
            self.server.concurrent_ttft_p99_us,
            self.server.concurrent_tok_per_s,
            self.server.steady_malloc_blocks_delta,
            self.server.steady_malloc_bytes_delta,
            if (self.server.all_ok) "yes" else "no",
            vmstat.Snapshot.swapoutDelta(self.vm_before, self.vm_after),
            vmstat.Snapshot.pageoutDelta(self.vm_before, self.vm_after),
        });
    }
};

pub const M5Error = Error || moe.WorkspaceError || moe.EncodeError ||
    ngram_nvme.BuildError || ngram_nvme.OpenError || ngram_nvme.QueryError ||
    qwen3.LoadError || qwen3.StepError || draft_mod.DraftLoadError ||
    engine_mod.InitError || engine_mod.RunError || server_mod.Error ||
    error{ LoopbackFailed, PathTooLong, BenchParityFailed };

fn percentileF64(samples: []f64, pct: f64) f64 {
    if (samples.len == 0) return 0.0;
    std.sort.pdq(f64, samples, {}, std.sort.asc(f64));
    const clamped = std.math.clamp(pct, 0.0, 1.0);
    const idx: usize = @intFromFloat(@floor(clamped * @as(f64, @floatFromInt(samples.len - 1))));
    return samples[@min(idx, samples.len - 1)];
}

fn runM5Q2Calibration(allocator: std.mem.Allocator, config: M5Config) M5Error![]CaseResult {
    var max_w_stride: usize = 64;
    var max_sb_stride: usize = 64;
    var max_x_bytes: usize = 64;
    var max_y_bytes: usize = 64;
    for (config.q2_cases) |case| {
        const layout = try computeCaseLayout(case);
        max_w_stride = @max(max_w_stride, amf.align64(layout.w_bytes));
        max_sb_stride = @max(max_sb_stride, amf.align64(layout.sb_bytes));
        max_x_bytes = @max(max_x_bytes, amf.align64(layout.x_bytes));
        max_y_bytes = @max(max_y_bytes, amf.align64(layout.y_bytes));
    }

    const qweight_pool_bytes = amf.align64(@max(max_w_stride, config.q2_min_working_set_bytes));
    const sb_pool_bytes = amf.align64(@max(max_sb_stride, config.q2_min_working_set_bytes / 4));
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

    const results = try allocator.alloc(CaseResult, config.q2_cases.len);
    errdefer allocator.free(results);

    for (config.q2_cases, 0..) |case, case_idx| {
        const layout = try computeCaseLayout(case);
        const w_stride = amf.align64(layout.w_bytes);
        const sb_stride = amf.align64(layout.sb_bytes);
        const num_slots: usize = @max(
            @as(usize, 1),
            @min(qweight_pool_bytes / w_stride, sb_pool_bytes / sb_stride),
        );
        const timed_batches = if (case.kind.isGemv()) config.q2_gemv_timed_batches else config.q2_gemm_timed_batches;
        const disps_per_batch = if (case.kind.isGemv()) config.q2_gemv_dispatches_per_batch else config.q2_gemm_dispatches_per_batch;

        var slot_cursor: usize = 0;
        var warmup_sig: u64 = 0;
        for (0..config.q2_warmup_batches) |_| {
            var frame = try compute.beginFrame(config.timeout_ms);
            errdefer frame.abort();
            for (0..disps_per_batch) |d| {
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
            for (0..disps_per_batch) |d| {
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
            .dispatches = @as(u64, timed_batches) * disps_per_batch,
            .weight_slots = num_slots,
            .elapsed_ns = t1 - t0,
        };
    }

    return results;
}

fn runM5HeteroMoeBench(config: M5Config) M5Error![8]HeteroMoeRow {
    const facts = try sysinfo.Facts.read();
    const device = try metal.Device.open();
    defer device.close();
    const info = try device.info();
    const wired_budget = budget_mod.Budget.fromHost(facts, info, .{});

    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try moe.MoePipelines.init(device, library);
    defer pipes.close();
    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();

    const Profile = enum { uniform_q4, uniform_q3, uniform_q2, hetero_q2_q3_q4 };
    const profiles = [_]Profile{ .uniform_q4, .uniform_q3, .uniform_q2, .hetero_q2_q3_q4 };
    const m_vals = [_]u32{ 1, 8 };

    const hidden = config.hetero_moe_hidden;
    const inter = config.hetero_moe_inter;
    const n_exp = config.hetero_moe_experts;
    const top_k = config.hetero_moe_top_k;

    var rows: [8]HeteroMoeRow = undefined;
    var row_idx: usize = 0;

    for (m_vals) |m| {
        const shared_pool: u32 = if (m == 1) top_k else @min(n_exp, @max(top_k, config.hetero_moe_shared_m8));
        var q4_ref_bytes: u64 = 1;

        for (profiles) |prof| {
            var schemes: [moe.MAX_EXPERTS]quant.WeightDType = @splat(.q4_affine);
            var scheme_counts: [moe.NUM_SCHEME_BUCKETS]u32 = @splat(0);
            for (0..n_exp) |e| {
                const dt: quant.WeightDType = switch (prof) {
                    .uniform_q4 => .q4_affine,
                    .uniform_q3 => .q3_affine,
                    .uniform_q2 => .q2_affine,
                    // 25% q4_affine (hot), 25% q3_affine (mid), 50% q2_affine (cold/low-sens)
                    .hetero_q2_q3_q4 => switch (e & 3) {
                        0 => .q4_affine,
                        1 => .q3_affine,
                        else => .q2_affine,
                    },
                };
                schemes[e] = dt;
                scheme_counts[@backingInt(dt)] += 1;
            }
            var active_buckets: u32 = 0;
            for (scheme_counts) |c| {
                if (c > 0) active_buckets += 1;
            }

            // Compute exact arena layout per expert.
            var all_experts_bytes: usize = 0;
            for (0..n_exp) |e| {
                const bits: quant.QuantBits = switch (schemes[e]) {
                    .q4_affine => .q4,
                    .q3_affine => .q3,
                    .q2_affine => .q2,
                    else => unreachable,
                };
                const gu_w, const gu_sb, const dn_w, const dn_sb = switch (bits) {
                    .q4 => .{
                        try Q4.packedWeightBytes(inter, hidden),
                        try Q4.scaleBiasBytes(inter, hidden),
                        try Q4.packedWeightBytes(hidden, inter),
                        try Q4.scaleBiasBytes(hidden, inter),
                    },
                    .q3 => .{
                        try Q3.packedWeightBytes(inter, hidden),
                        try Q3.scaleBiasBytes(inter, hidden),
                        try Q3.packedWeightBytes(hidden, inter),
                        try Q3.scaleBiasBytes(hidden, inter),
                    },
                    .q2 => .{
                        try Q2.packedWeightBytes(inter, hidden),
                        try Q2.scaleBiasBytes(inter, hidden),
                        try Q2.packedWeightBytes(hidden, inter),
                        try Q2.scaleBiasBytes(hidden, inter),
                    },
                };
                all_experts_bytes += 2 * (amf.align64(gu_w) + amf.align64(gu_sb)) +
                    amf.align64(dn_w) + amf.align64(dn_sb);
            }

            const num_banks: usize = @max(@as(usize, 1), @as(usize, n_exp) / shared_pool);
            const bank_logits_bytes = amf.align64(@as(usize, m) * n_exp * @sizeOf(f32));
            const all_logits_bytes = num_banks * bank_logits_bytes;
            const x_bytes = amf.align64(@as(usize, m) * hidden * @sizeOf(f16));
            const out_bytes = x_bytes;
            const tbl_bytes = amf.align64(@as(usize, n_exp) * @sizeOf(moe.ExpertWeights));
            const schemes_bytes = amf.align64(n_exp);
            const ws_bytes = try moe.MoeWorkspace.requiredBytes(.{
                .max_tokens = m,
                .num_experts = n_exp,
                .top_k = top_k,
                .hidden_size = hidden,
                .intermediate_size = inter,
            });
            const arena_cap = try page.alignUp(
                all_experts_bytes + all_logits_bytes + x_bytes + out_bytes + tbl_bytes + schemes_bytes + ws_bytes + 8 * page.APPLE_PAGE_SIZE,
            );

            var arena = try arena_mod.WiredArena.init(arena_cap, wired_budget);
            defer arena.deinit();

            const weights_raw = try arena.allocAligned(all_experts_bytes, 64);
            const logits_raw = try arena.allocAligned(all_logits_bytes, 64);
            const x_raw = try arena.allocAligned(x_bytes, 64);
            const out_raw = try arena.allocAligned(out_bytes, 64);
            const tbl_raw = try arena.allocAligned(tbl_bytes, 64);
            const schemes_raw = try arena.allocAligned(schemes_bytes, 64);
            const ws = try moe.MoeWorkspace.init(&arena, .{
                .max_tokens = m,
                .num_experts = n_exp,
                .top_k = top_k,
                .hidden_size = hidden,
                .intermediate_size = inter,
            });

            for (0..n_exp) |e| {
                schemes_raw[e] = @backingInt(schemes[e]);
            }

            const exp_tbl: []moe.ExpertWeights = @alignCast(std.mem.bytesAsSlice(moe.ExpertWeights, tbl_raw));
            var rng: u64 = 0x517C_C1B7_2722_0A95 +% row_idx;
            const sb_val: quant.ScaleBias = .{ .scale = @floatCast(0.015625), .bias = @floatCast(-0.0625) };
            var cursor: usize = 0;
            for (0..n_exp) |e| {
                const bits: quant.QuantBits = switch (schemes[e]) {
                    .q4_affine => .q4,
                    .q3_affine => .q3,
                    .q2_affine => .q2,
                    else => unreachable,
                };
                const gu_w, const gu_sb, const dn_w, const dn_sb = switch (bits) {
                    .q4 => .{
                        try Q4.packedWeightBytes(inter, hidden),
                        try Q4.scaleBiasBytes(inter, hidden),
                        try Q4.packedWeightBytes(hidden, inter),
                        try Q4.scaleBiasBytes(hidden, inter),
                    },
                    .q3 => .{
                        try Q3.packedWeightBytes(inter, hidden),
                        try Q3.scaleBiasBytes(inter, hidden),
                        try Q3.packedWeightBytes(hidden, inter),
                        try Q3.scaleBiasBytes(hidden, inter),
                    },
                    .q2 => .{
                        try Q2.packedWeightBytes(inter, hidden),
                        try Q2.scaleBiasBytes(inter, hidden),
                        try Q2.packedWeightBytes(hidden, inter),
                        try Q2.scaleBiasBytes(hidden, inter),
                    },
                };
                const g_w = cursor;
                cursor += amf.align64(gu_w);
                const g_sb = cursor;
                cursor += amf.align64(gu_sb);
                const u_w = cursor;
                cursor += amf.align64(gu_w);
                const u_sb = cursor;
                cursor += amf.align64(gu_sb);
                const d_w = cursor;
                cursor += amf.align64(dn_w);
                const d_sb = cursor;
                cursor += amf.align64(dn_sb);

                for ([_]struct { off: usize, len: usize }{
                    .{ .off = g_w, .len = gu_w },
                    .{ .off = u_w, .len = gu_w },
                    .{ .off = d_w, .len = dn_w },
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
                    .{ .off = g_sb, .len = gu_sb },
                    .{ .off = u_sb, .len = gu_sb },
                    .{ .off = d_sb, .len = dn_sb },
                }) |seg| {
                    const sbs: []quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, weights_raw[seg.off .. seg.off + seg.len]));
                    for (sbs) |*sb| sb.* = sb_val;
                }
                exp_tbl[e] = .{
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

            // Build rotating router-logit banks activating `shared_pool` unique experts per bank.
            var bank0_unique_bytes: u64 = 0;
            var bank0_unique_count: u32 = 0;
            for (0..num_banks) |bank| {
                const bank_slice = logits_raw[bank * bank_logits_bytes ..][0 .. @as(usize, m) * n_exp * @sizeOf(f32)];
                const logits_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, bank_slice));
                for (logits_f32) |*v| v.* = -8.0;

                const base_e: u32 = @intCast((bank * shared_pool) % n_exp);
                const stride: u32 = if (m > 1 and shared_pool > top_k)
                    @max(@as(u32, 1), (shared_pool - top_k + m - 2) / (m - 1))
                else
                    0;

                var seen: [moe.MAX_EXPERTS]bool = @splat(false);
                for (0..m) |tok| {
                    const start = @min(@as(u32, @intCast(tok)) * stride, shared_pool - top_k);
                    for (0..top_k) |j| {
                        const e_id: usize = (base_e + start + @as(u32, @intCast(j))) % n_exp;
                        logits_f32[tok * n_exp + e_id] = 4.0 - @as(f32, @floatFromInt(j)) * 0.1;
                        seen[e_id] = true;
                    }
                }
                if (bank == 0) {
                    for (0..n_exp) |e| {
                        if (seen[e]) {
                            bank0_unique_count += 1;
                            bank0_unique_bytes += try moe.singleExpertBytesForDType(schemes[e], hidden, inter);
                        }
                    }
                }
            }

            const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
            defer gpu_buf.close();
            try compute.makeResident(&.{gpu_buf});

            const weights_base: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(weights_raw.ptr) };
            const x_bind: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(x_raw.ptr) };
            const out_bind: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out_raw.ptr) };
            const tbl_bind: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(tbl_raw.ptr) };
            const schemes_bind: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(schemes_raw.ptr) };
            const logits_base_off = arena.offsetOf(logits_raw.ptr);

            var bank_cursor: usize = 0;
            var warmup_sig: u64 = 0;
            for (0..config.hetero_moe_warmup_batches) |_| {
                var frame = try compute.beginFrame(config.timeout_ms);
                errdefer frame.abort();
                for (0..config.hetero_moe_dispatches_per_batch) |d| {
                    const b = bank_cursor % num_banks;
                    bank_cursor += 1;
                    const base_args: moe.MoELayerArgs = .{
                        .x = x_bind,
                        .router = .{ .logits = .{ .binding = .{ .buffer = gpu_buf, .offset = logits_base_off + @as(u64, b) * bank_logits_bytes }, .dtype = .f32 } },
                        .weights_base = weights_base,
                        .expert_table = tbl_bind,
                        .workspace = &ws,
                        .workspace_buf = gpu_buf,
                        .residual = null,
                        .out = out_bind,
                        .m = m,
                        .num_experts = n_exp,
                        .top_k = top_k,
                        .hidden_size = hidden,
                        .intermediate_size = inter,
                        .norm_topk_prob = true,
                    };
                    switch (prof) {
                        .uniform_q4 => try pipes.encodeMoELayer(&frame, .q4, base_args, d > 0),
                        .uniform_q3 => try pipes.encodeMoELayer(&frame, .q3, base_args, d > 0),
                        .uniform_q2 => try pipes.encodeMoELayer(&frame, .q2, base_args, d > 0),
                        .hetero_q2_q3_q4 => try pipes.encodeMoELayerHetero(&frame, .{
                            .base = base_args,
                            .expert_schemes = schemes_bind,
                            .scheme_counts = scheme_counts,
                        }, d > 0),
                    }
                }
                warmup_sig = try frame.commit();
            }
            try compute.wait(warmup_sig, config.timeout_ms);

            var last_sig: u64 = 0;
            const t0 = try vmstat.nowNs();
            for (0..config.hetero_moe_timed_batches) |_| {
                var frame = try compute.beginFrame(config.timeout_ms);
                errdefer frame.abort();
                for (0..config.hetero_moe_dispatches_per_batch) |d| {
                    const b = bank_cursor % num_banks;
                    bank_cursor += 1;
                    const base_args: moe.MoELayerArgs = .{
                        .x = x_bind,
                        .router = .{ .logits = .{ .binding = .{ .buffer = gpu_buf, .offset = logits_base_off + @as(u64, b) * bank_logits_bytes }, .dtype = .f32 } },
                        .weights_base = weights_base,
                        .expert_table = tbl_bind,
                        .workspace = &ws,
                        .workspace_buf = gpu_buf,
                        .residual = null,
                        .out = out_bind,
                        .m = m,
                        .num_experts = n_exp,
                        .top_k = top_k,
                        .hidden_size = hidden,
                        .intermediate_size = inter,
                        .norm_topk_prob = true,
                    };
                    switch (prof) {
                        .uniform_q4 => try pipes.encodeMoELayer(&frame, .q4, base_args, d > 0),
                        .uniform_q3 => try pipes.encodeMoELayer(&frame, .q3, base_args, d > 0),
                        .uniform_q2 => try pipes.encodeMoELayer(&frame, .q2, base_args, d > 0),
                        .hetero_q2_q3_q4 => try pipes.encodeMoELayerHetero(&frame, .{
                            .base = base_args,
                            .expert_schemes = schemes_bind,
                            .scheme_counts = scheme_counts,
                        }, d > 0),
                    }
                }
                last_sig = try frame.commit();
            }
            try compute.wait(last_sig, config.timeout_ms);
            const t1 = try vmstat.nowNs();

            const out_f16: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, out_raw));
            if (!std.math.isFinite(out_f16[0])) return error.NonFiniteOutput;

            const act_bytes: u64 = @as(u64, m) * hidden * @sizeOf(f16) * 2 + @as(u64, m) * n_exp * @sizeOf(f32);
            const bytes_per_disp: u64 = bank0_unique_bytes + act_bytes;
            if (prof == .uniform_q4) q4_ref_bytes = bytes_per_disp;

            const total_disps: u64 = @as(u64, config.hetero_moe_timed_batches) * config.hetero_moe_dispatches_per_batch;
            const elapsed_ns: u64 = @max(t1 -| t0, 1);
            const lat_us: f64 = (@as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(total_disps))) / 1000.0;
            const eff_gbs: f64 = (@as(f64, @floatFromInt(bytes_per_disp)) * @as(f64, @floatFromInt(total_disps))) / @as(f64, @floatFromInt(elapsed_ns));

            rows[row_idx] = .{
                .label = switch (prof) {
                    .uniform_q4 => "uniform_q4",
                    .uniform_q3 => "uniform_q3",
                    .uniform_q2 => "uniform_q2",
                    .hetero_q2_q3_q4 => "hetero_q2_q3_q4",
                },
                .m = m,
                .num_experts = n_exp,
                .top_k = top_k,
                .unique_experts = bank0_unique_count,
                .active_buckets = active_buckets,
                .bytes_per_disp = bytes_per_disp,
                .latency_us = lat_us,
                .effective_gbs = eff_gbs,
                .bus_pct = (eff_gbs / M4_BUS_BANDWIDTH_GBS) * 100.0,
                .vs_q4_bytes_pct = (@as(f64, @floatFromInt(bytes_per_disp)) / @as(f64, @floatFromInt(q4_ref_bytes)) - 1.0) * 100.0,
            };
            row_idx += 1;
        }
    }

    return rows;
}

fn runM5NgramBench(allocator: std.mem.Allocator, config: M5Config) M5Error!NgramBenchResult {
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const raw_path = std.fmt.bufPrint(
        path_buf[0 .. path_buf.len - 1],
        "{s}/aminfer_m5_ngram_bench_{d}.ang",
        .{ config.ngram_temp_dir, @as(u64, @intCast(try vmstat.nowNs())) & 0xFFFF_FFFF },
    ) catch return error.PathTooLong;
    path_buf[raw_path.len] = 0;
    const ang_path: [:0]const u8 = path_buf[0..raw_path.len :0];
    defer _ = std.c.unlink(ang_path.ptr);

    const total_n: usize = @as(usize, config.ngram_buckets) * config.ngram_entries_per_bucket;
    const ctx_storage = try allocator.alloc([4]u32, total_n);
    defer allocator.free(ctx_storage);
    const cont_storage = try allocator.alloc([8]u32, total_n);
    defer allocator.free(cont_storage);
    const entries = try allocator.alloc(ngram_nvme.BuildEntry, total_n);
    defer allocator.free(entries);

    for (0..total_n) |i| {
        const base: u32 = @intCast(1000 + i * 13);
        ctx_storage[i] = .{ base, base + 1, base + 2, base + 3 };
        cont_storage[i] = .{ base + 4, base + 5, base + 6, base + 7, base + 8, base + 9, base + 10, base + 11 };
        entries[i] = .{
            .ctx = &ctx_storage[i],
            .cont = &cont_storage[i],
            .hit_count = @intCast(100 - (i % 64)),
            .step_prob = 0.92,
        };
    }

    try ngram_nvme.writeTableFile(allocator, ang_path, entries, .{
        .num_buckets = config.ngram_buckets,
        .min_ngram = 2,
        .max_ngram = 4,
        .max_cont_len = 8,
        .vocab_size = 151936,
        .corpus_id = "m5-bench-ngram",
        .include_table_crc = true,
    });

    const hdr = try ngram_nvme.verifyFile(allocator, ang_path);
    var tbl = try ngram_nvme.NvmeNgramTable.open(allocator, ang_path, .{
        .cache_pages = config.ngram_cache_pages,
        .io_backend = .kqueue_aio,
    });
    defer tbl.deinit(allocator);

    var out_cont: [ngram_nvme.MAX_CONT_LEN]u32 = undefined;

    // 1. Cold F_NOCACHE NVMe queries (clearCache between bursts exceeding cache_pages).
    const cold_n = @min(config.ngram_cold_queries, total_n);
    const cold_samples = try allocator.alloc(f64, cold_n);
    defer allocator.free(cold_samples);
    try tbl.clearCache();
    tbl.resetStats();

    var cold_seen: usize = 0;
    var scan_idx: usize = 0;
    while (cold_seen < cold_n and scan_idx < total_n) : (scan_idx += 1) {
        if ((cold_seen % @max(@as(usize, 1), config.ngram_cache_pages / 2)) == 0) {
            try tbl.clearCache();
        }
        const t0 = try vmstat.nowNs();
        const res = try tbl.query(&ctx_storage[scan_idx], &out_cont);
        const t1 = try vmstat.nowNs();
        if (!res.cache_hit and res.is_seed) {
            cold_samples[cold_seen] = @as(f64, @floatFromInt(@max(t1 -| t0, 1))) / 1000.0;
            cold_seen += 1;
        }
    }
    const cold_p50 = percentileF64(cold_samples[0..cold_seen], 0.50);
    const cold_p99 = percentileF64(cold_samples[0..cold_seen], 0.99);

    // 2. Wired LRU cache-hit queries (warm 8 hot contexts, then query repeatedly).
    const hot_n = @min(@as(usize, 8), total_n);
    for (0..hot_n) |i| {
        _ = try tbl.query(&ctx_storage[i], &out_cont);
    }
    const cache_samples = try allocator.alloc(f64, config.ngram_cache_queries);
    defer allocator.free(cache_samples);
    for (0..config.ngram_cache_queries) |i| {
        const idx = i % hot_n;
        const t0 = try vmstat.nowNs();
        const res = try tbl.query(&ctx_storage[idx], &out_cont);
        const t1 = try vmstat.nowNs();
        if (!res.is_seed) return error.BenchParityFailed;
        cache_samples[i] = @as(f64, @floatFromInt(@max(t1 -| t0, 1))) / 1000.0;
    }
    const cache_p50 = percentileF64(cache_samples, 0.50);

    // 3. Async `prefetch()` followed by `query()`.
    const pref_n = @min(@as(usize, 16), total_n);
    const pref_samples = try allocator.alloc(f64, pref_n);
    defer allocator.free(pref_samples);
    try tbl.clearCache();
    for (0..pref_n) |i| {
        _ = try tbl.prefetch(&ctx_storage[i]);
        // Brief poll so the background F_NOCACHE AIO finishes before foreground query.
        var spin: usize = 0;
        while (spin < 200) : (spin += 1) {
            if ((try tbl.reapPrefetches()) > 0) break;
            var ts: std.c.timespec = .{ .sec = 0, .nsec = 10_000 };
            _ = std.c.nanosleep(&ts, null);
        }
        const t0 = try vmstat.nowNs();
        const res = try tbl.query(&ctx_storage[i], &out_cont);
        const t1 = try vmstat.nowNs();
        if (!res.is_seed) return error.BenchParityFailed;
        pref_samples[i] = @as(f64, @floatFromInt(@max(t1 -| t0, 1))) / 1000.0;
    }
    const pref_p50 = percentileF64(pref_samples, 0.50);

    return .{
        .file_bytes = hdr.table_offset + hdr.table_bytes,
        .num_buckets = hdr.num_buckets,
        .total_entries = hdr.total_entries,
        .wired_kib = tbl.totalWiredBytes() >> 10,
        .cold_nvme_p50_us = cold_p50,
        .cold_nvme_p99_us = cold_p99,
        .cache_hit_p50_us = cache_p50,
        .prefetch_hit_p50_us = pref_p50,
        .queries = tbl.stats.queries,
        .hits = tbl.stats.hits,
        .seed_hits = tbl.stats.seed_hits,
        .nvme_reads = tbl.stats.nvme_reads,
    };
}

fn runM5PearlSynthBench(allocator: std.mem.Allocator, config: M5Config) M5Error!PearlSynthBenchResult {
    const dense_cfg: qwen3.ModelConfig = .{
        .arch = .qwen3_dense,
        .quant_bits = .q4,
        .group_size = 64,
        .vocab_size = config.pearl_vocab,
        .hidden_size = config.pearl_hidden,
        .num_hidden_layers = config.pearl_layers,
        .num_attention_heads = @max(@as(u32, 2), config.pearl_hidden / 64),
        .num_key_value_heads = @max(@as(u32, 1), (config.pearl_hidden / 64) / 2),
        .head_dim = 64,
        .intermediate_size = config.pearl_inter,
        .tie_word_embeddings = true,
    };
    const tgt_bytes = try qwen3.buildSyntheticAmfImageWithSpec(allocator, dense_cfg, .{
        .linear_dtype = .q4_affine,
        .embed_dtype = .bf16,
        .norm_dtype = .bf16,
    }, 0xD8_B301);
    defer allocator.free(tgt_bytes);

    // Same-family small draft (shares same synthetic weights so PEARL gets high spec-hit rate).
    const prompt = [_]u32{ 3, 11, 29, 47, 5, 13, 31, 53 };
    const gen_len = config.pearl_gen_tokens;
    const expected = try allocator.alloc(u32, gen_len);
    defer allocator.free(expected);
    const out_off = try allocator.alloc(u32, gen_len);
    defer allocator.free(out_off);
    const out_on = try allocator.alloc(u32, gen_len);
    defer allocator.free(out_on);

    // 1. Baseline non-speculative greedy decode.
    var base_ns: u64 = 1;
    {
        var base_model = try qwen3.Qwen3Model.loadFromBytes(allocator, tgt_bytes, dense_cfg, .{
            .max_seq_len = 128,
            .chunk_tokens = 16,
            .max_tree_nodes = 1,
        });
        defer base_model.deinit(allocator);

        var tok = try base_model.prefillPrompt(&prompt, true);
        expected[0] = tok;
        var pos: u32 = @intCast(prompt.len);
        const t0 = try vmstat.nowNs();
        for (1..gen_len) |i| {
            tok = try base_model.stepDecode(tok, pos);
            expected[i] = tok;
            pos += 1;
        }
        const t1 = try vmstat.nowNs();
        base_ns = @max(t1 -| t0, 1);
    }

    // 2. Small draft PEARL `off` vs PEARL `on`.
    var spec_tgt = try qwen3.Qwen3Model.loadFromBytes(allocator, tgt_bytes, dense_cfg, .{
        .max_seq_len = 128,
        .chunk_tokens = 16,
        .max_tree_nodes = 16,
    });
    defer spec_tgt.deinit(allocator);

    var sd = try draft_mod.SmallDraftModel.loadFromBytes(allocator, &spec_tgt, tgt_bytes, dense_cfg, .{
        .placement = .gpu,
        .max_seq_len = 128,
        .chunk_tokens = 16,
        .max_tree_slots = 16,
        .own_compute_queue = true,
    });
    defer sd.deinit(allocator);

    var off_ns: u64 = 1;
    {
        var eng_off = try engine_mod.SpecEngine.initWithSmallDraft(allocator, &spec_tgt, &sd, .{
            .mode = .small_draft,
            .max_tree_nodes = 12,
            .max_draft_depth = 3,
            .branch_width = 2,
            .top_k_per_node = 3,
            .temperature = 0.0,
            .enable_pruning = false,
            .pearl_mode = .off,
        });
        defer eng_off.deinit(allocator);

        out_off[0] = try eng_off.prefill(&prompt, null);
        var p: usize = 1;
        const t0 = try vmstat.nowNs();
        while (p < gen_len) {
            const rs = try eng_off.stepRound(null, out_off[p..]);
            p += rs.emitted_count;
        }
        const t1 = try vmstat.nowNs();
        off_ns = @max(t1 -| t0, 1);
    }

    var on_ns: u64 = 1;
    var on_stats: engine_mod.SpecStats = .{};
    {
        var eng_on = try engine_mod.SpecEngine.initWithSmallDraft(allocator, &spec_tgt, &sd, .{
            .mode = .small_draft,
            .max_tree_nodes = 12,
            .max_draft_depth = 3,
            .branch_width = 2,
            .top_k_per_node = 3,
            .temperature = 0.0,
            .enable_pruning = false,
            .pearl_mode = .on,
        });
        defer eng_on.deinit(allocator);

        out_on[0] = try eng_on.prefill(&prompt, null);
        var p: usize = 1;
        const t0 = try vmstat.nowNs();
        while (p < gen_len) {
            const rs = try eng_on.stepRound(null, out_on[p..]);
            p += rs.emitted_count;
        }
        const t1 = try vmstat.nowNs();
        on_ns = @max(t1 -| t0, 1);
        on_stats = eng_on.stats;
    }

    const bit_exact_off = std.mem.eql(u32, expected, out_off);
    const bit_exact_on = std.mem.eql(u32, expected, out_on);
    if (!bit_exact_off or !bit_exact_on) return error.BenchParityFailed;

    const dec_toks: f64 = @floatFromInt(@max(gen_len -| 1, 1));
    return .{
        .has_own_queue = sd.hasOwnComputeQueue(),
        .bit_exact_off = bit_exact_off,
        .bit_exact_on = bit_exact_on,
        .base_tok_per_s = dec_toks * 1_000_000_000.0 / @as(f64, @floatFromInt(base_ns)),
        .off_tok_per_s = dec_toks * 1_000_000_000.0 / @as(f64, @floatFromInt(off_ns)),
        .on_tok_per_s = dec_toks * 1_000_000_000.0 / @as(f64, @floatFromInt(on_ns)),
        .mean_accept_len = on_stats.meanAcceptanceLength(),
        .pearl_rounds = on_stats.pearl_rounds,
        .pearl_hits = on_stats.pearl_hits,
        .pearl_rollbacks = on_stats.pearl_rollbacks,
        .pearl_saved_ms = @as(f64, @floatFromInt(on_stats.pearl_saved_ns)) / 1_000_000.0,
    };
}

fn loopbackHttpExchange(
    port: u16,
    req: []const u8,
    rx_buf: []u8,
) error{LoopbackFailed}!struct { total_bytes: usize, ttft_ns: u64, e2e_ns: u64 } {
    const fd = std.c.socket(std.c.AF.INET, std.c.SOCK.STREAM, 0);
    if (fd < 0) return error.LoopbackFailed;
    defer _ = std.c.close(fd);

    var one: c_int = 1;
    _ = std.c.setsockopt(fd, std.c.SOL.SOCKET, std.c.SO.NOSIGPIPE, &one, @sizeOf(c_int));

    var addr = std.c.sockaddr.in{
        .port = std.mem.nativeToBig(u16, port),
        .addr = @bitCast([4]u8{ 127, 0, 0, 1 }),
    };
    const t0 = vmstat.nowNs() catch return error.LoopbackFailed;
    if (std.c.connect(fd, @ptrCast(&addr), @sizeOf(std.c.sockaddr.in)) != 0) {
        return error.LoopbackFailed;
    }

    var sent: usize = 0;
    while (sent < req.len) {
        const n = std.c.send(fd, req.ptr + sent, req.len - sent, 0);
        if (n > 0) {
            sent += @intCast(n);
        } else if (n < 0 and std.c.errno(n) == .INTR) {
            continue;
        } else {
            return error.LoopbackFailed;
        }
    }

    var got: usize = 0;
    var first_data_ns: u64 = 0;
    while (got < rx_buf.len) {
        const n = std.c.recv(fd, rx_buf.ptr + got, rx_buf.len - got, 0);
        if (n > 0) {
            got += @intCast(n);
            if (first_data_ns == 0 and std.mem.indexOf(u8, rx_buf[0..got], "data: ") != null) {
                first_data_ns = vmstat.nowNs() catch t0;
            }
        } else if (n == 0) {
            break;
        } else if (std.c.errno(n) == .INTR) {
            continue;
        } else {
            return error.LoopbackFailed;
        }
    }
    const t1 = vmstat.nowNs() catch t0;
    return .{
        .total_bytes = got,
        .ttft_ns = if (first_data_ns > t0) first_data_ns - t0 else (t1 -| t0),
        .e2e_ns = @max(t1 -| t0, 1),
    };
}

fn runM5ServerBench(allocator: std.mem.Allocator, config: M5Config) M5Error!ServerBenchResult {
    const MockCtx = struct {
        tokens_per_req: u32,
        fn generate(ctx: *anyopaque, job: *const server_mod.InferenceJob, stream: *server_mod.TokenStream) void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            if (!stream.setPromptTokens(12)) return;
            const total = @min(job.max_tokens, self.tokens_per_req);
            const prefix = [_][]const u8{ "<think>", "ok", "</think>" };
            var emitted: u32 = 0;
            for (prefix) |p| {
                if (emitted >= total) break;
                if (!stream.emitTokenBytes(p)) return;
                emitted += 1;
            }
            while (emitted < total) : (emitted += 1) {
                if (!stream.emitTokenBytes(" tok")) return;
            }
            stream.finish(.stop, 12);
        }
    };

    var mock = MockCtx{ .tokens_per_req = config.server_tokens_per_req };
    const srv = try server_mod.Server.init(allocator, .{
        .host = .{ 127, 0, 0, 1 },
        .port = 0,
        .model_id = "m5-bench-model",
        .wired_mib = 128,
        .max_conns = 32,
        .max_active_requests = 16,
    });
    defer srv.deinit(allocator);

    try srv.start(.{ .ctx = &mock, .generateFn = MockCtx.generate });
    defer srv.stop();
    const port = srv.boundPort();

    const health_req =
        "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n";
    const chat_body =
        "{\"model\":\"m5-bench-model\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":true,\"max_tokens\":16,\"temperature\":0.0}";
    var chat_req_buf: [512]u8 = undefined;
    const chat_req = std.fmt.bufPrint(
        &chat_req_buf,
        "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n{s}",
        .{ chat_body.len, chat_body },
    ) catch unreachable;

    var rx_buf: [32 * 1024]u8 = undefined;
    var all_ok = true;

    // Warmup both endpoints before measuring allocations or latency.
    for (0..4) |_| {
        _ = try loopbackHttpExchange(port, health_req, &rx_buf);
        _ = try loopbackHttpExchange(port, chat_req, &rx_buf);
    }

    // 1. GET /health latency.
    const health_samples = try allocator.alloc(f64, config.server_health_iters);
    defer allocator.free(health_samples);
    for (0..config.server_health_iters) |i| {
        const ex = try loopbackHttpExchange(port, health_req, &rx_buf);
        if (std.mem.indexOf(u8, rx_buf[0..ex.total_bytes], "200 OK") == null) all_ok = false;
        health_samples[i] = @as(f64, @floatFromInt(ex.e2e_ns)) / 1000.0;
    }

    // 2. Single-client steady-state SSE latency + zero-allocation check.
    const single_ttft = try allocator.alloc(f64, config.server_steady_iters);
    defer allocator.free(single_ttft);
    const single_e2e = try allocator.alloc(f64, config.server_steady_iters);
    defer allocator.free(single_e2e);

    const heap_before = try vmstat.Snapshot.capture();
    for (0..config.server_steady_iters) |i| {
        const ex = try loopbackHttpExchange(port, chat_req, &rx_buf);
        const body = rx_buf[0..ex.total_bytes];
        if (std.mem.indexOf(u8, body, "200 OK") == null or
            std.mem.indexOf(u8, body, "reasoning_content") == null or
            std.mem.indexOf(u8, body, "data: [DONE]") == null)
        {
            all_ok = false;
        }
        single_ttft[i] = @as(f64, @floatFromInt(ex.ttft_ns)) / 1000.0;
        single_e2e[i] = @as(f64, @floatFromInt(ex.e2e_ns)) / 1000.0;
    }
    const heap_after = try vmstat.Snapshot.capture();
    const hdelta = vmstat.heapDelta(heap_before, heap_after);

    // 3. Concurrent multi-client SSE fan-out (`server_concurrent_clients` simultaneous sockets).
    const n_clients: usize = @min(@max(config.server_concurrent_clients, 1), 16);
    const ClientTask = struct {
        port: u16,
        req: []const u8,
        ttft_us: f64 = 0.0,
        ok: bool = false,

        fn run(self: *@This()) void {
            var local_rx: [32 * 1024]u8 = undefined;
            const ex = loopbackHttpExchange(self.port, self.req, &local_rx) catch {
                self.ok = false;
                return;
            };
            const body = local_rx[0..ex.total_bytes];
            self.ok = std.mem.indexOf(u8, body, "200 OK") != null and
                std.mem.indexOf(u8, body, "data: [DONE]") != null;
            self.ttft_us = @as(f64, @floatFromInt(ex.ttft_ns)) / 1000.0;
        }
    };

    var tasks: [16]ClientTask = undefined;
    var threads: [16]std.Thread = undefined;
    const conc_t0 = try vmstat.nowNs();
    for (0..n_clients) |ci| {
        tasks[ci] = .{ .port = port, .req = chat_req };
        threads[ci] = std.Thread.spawn(.{}, ClientTask.run, .{&tasks[ci]}) catch return error.LoopbackFailed;
    }
    for (0..n_clients) |ci| {
        threads[ci].join();
        if (!tasks[ci].ok) all_ok = false;
    }
    const conc_t1 = try vmstat.nowNs();
    const conc_ns = @max(conc_t1 -| conc_t0, 1);

    var conc_ttft: [16]f64 = undefined;
    for (0..n_clients) |ci| conc_ttft[ci] = tasks[ci].ttft_us;
    const total_toks: f64 = @floatFromInt(n_clients * @as(usize, config.server_tokens_per_req));

    return .{
        .health_p50_us = percentileF64(health_samples, 0.50),
        .single_ttft_p50_us = percentileF64(single_ttft, 0.50),
        .single_e2e_p50_us = percentileF64(single_e2e, 0.50),
        .concurrent_clients = @intCast(n_clients),
        .concurrent_ttft_p50_us = percentileF64(conc_ttft[0..n_clients], 0.50),
        .concurrent_ttft_p99_us = percentileF64(conc_ttft[0..n_clients], 0.99),
        .concurrent_tok_per_s = total_toks * 1_000_000_000.0 / @as(f64, @floatFromInt(conc_ns)),
        .steady_malloc_blocks_delta = hdelta.blocks_in_use,
        .steady_malloc_bytes_delta = hdelta.size_in_use,
        .all_ok = all_ok,
    };
}

pub fn runM5WithConfig(allocator: std.mem.Allocator, config: M5Config) M5Error!M5Report {
    const vm_before = try vmstat.Snapshot.capture();

    const q2_cases = try runM5Q2Calibration(allocator, config);
    errdefer allocator.free(q2_cases);

    const hetero_rows = try runM5HeteroMoeBench(config);
    const ngram_res = try runM5NgramBench(allocator, config);
    const pearl_res = try runM5PearlSynthBench(allocator, config);
    const srv_res = try runM5ServerBench(allocator, config);

    const vm_after = try vmstat.Snapshot.capture();
    try vmstat.checkNoSwap(vm_before, vm_after);

    return .{
        .q2_cases = q2_cases,
        .hetero_moe_rows = hetero_rows,
        .ngram = ngram_res,
        .pearl = pearl_res,
        .server = srv_res,
        .vm_before = vm_before,
        .vm_after = vm_after,
    };
}

pub fn runM5(allocator: std.mem.Allocator) M5Error!M5Report {
    return runM5WithConfig(allocator, .{});
}

// ===========================================================================
// Milestone M6: GraphModel Decode Bench & `aminfer bench-m6` Suite
// (W9/W10/W16, G7/G8/G9/G10, C10 PLE, D12 SelfSpecRecurrent, D13 cpu_sme2)
// ===========================================================================

pub const GraphDecodeBenchOptions = struct {
    prefill_tokens: u32 = 511,
    decode_steps: u32 = 16,
    warmup_steps: u32 = 4,
    max_seq_len: u32 = 576,
    budget_override: ?budget_mod.Budget = null,
    ple_paths: []const [:0]const u8 = &.{},
    loop_adapter_amf_path: ?[:0]const u8 = null,
    max_loops_override: ?u32 = null,
    enable_loop_early_exit: bool = false,
    cold_slot_count: u32 = 0,
};

pub const GraphDecodeBenchResult = struct {
    attention_summary: []const u8,
    ffn_label: []const u8,
    residual_label: []const u8,
    loop_label: []const u8,
    measured_layers: u32,
    effective_loops: u32,
    seq_len: u32,
    decode_steps: u32,
    shard_count: usize,
    wired_bytes: u64,
    load_ms: f64,
    prefill_ms: f64,
    decode_ns: u64,
    bytes_per_tok: u64,
    loop_isolation: graph.LoopIsolationStats,
    ple_stats: ?ple.Stats,
    cold_stats: ?qwen3.ColdExpertStats,
    heap_delta: vmstat.HeapDelta,
    swapouts_delta: u64,
    pageouts_delta: u64,

    pub fn msPerToken(self: GraphDecodeBenchResult) f64 {
        return (@as(f64, @floatFromInt(self.decode_ns)) / @as(f64, @floatFromInt(self.decode_steps))) / 1_000_000.0;
    }

    pub fn tokensPerSec(self: GraphDecodeBenchResult) f64 {
        const ms = self.msPerToken();
        return if (ms > 0.0) 1000.0 / ms else 0.0;
    }

    pub fn effectiveGbs(self: GraphDecodeBenchResult) f64 {
        const ms = self.msPerToken();
        return if (ms > 0.0) @as(f64, @floatFromInt(self.bytes_per_tok)) / (ms * 1_000_000.0) else 0.0;
    }

    pub fn write(self: GraphDecodeBenchResult, out: *std.Io.Writer) std.Io.Writer.Error!void {
        try out.print(
            \\graph_bench.attention            {s}
            \\graph_bench.ffn                  {s}
            \\graph_bench.residual             {s}
            \\graph_bench.loop                 {s} (R={d})
            \\graph_bench.layers               {d}
            \\graph_bench.shards               {d}
            \\graph_bench.wired_mib            {d}
            \\graph_bench.load_ms              {d:.2}
            \\graph_bench.seq_len              {d}
            \\graph_bench.prefill_ms           {d:.3}
            \\graph_bench.decode_steps         {d}
            \\graph_bench.ms_per_tok           {d:.3}
            \\graph_bench.tok_per_s            {d:.2}
            \\graph_bench.bytes_per_tok        {d}
            \\graph_bench.effective_gbs        {d:.2}
            \\graph_bench.gdn_scratch_evals    {d}
            \\graph_bench.gdn_final_commits    {d}
            \\graph_bench.ple_r0_fills         {d}
            \\graph_bench.ple_r_ge1_nvme_reads {d}
            \\graph_bench.qsa_index_reuses     {d}
            \\graph_bench.moe_r_ge1_cold_stall {d}
            \\graph_bench.malloc_blocks_delta  {d}
            \\graph_bench.malloc_bytes_delta   {d}
            \\graph_bench.swapouts_delta       {d}
            \\graph_bench.pageouts_delta       {d}
            \\
        , .{
            self.attention_summary,
            self.ffn_label,
            self.residual_label,
            self.loop_label,
            self.effective_loops,
            self.measured_layers,
            self.shard_count,
            self.wired_bytes >> 20,
            self.load_ms,
            self.seq_len,
            self.prefill_ms,
            self.decode_steps,
            self.msPerToken(),
            self.tokensPerSec(),
            self.bytes_per_tok,
            self.effectiveGbs(),
            self.loop_isolation.gdn_scratch_evals,
            self.loop_isolation.gdn_final_commits,
            self.loop_isolation.ple_r0_fills,
            self.loop_isolation.ple_r_ge1_nvme_reads,
            self.loop_isolation.qsa_index_reuses,
            self.loop_isolation.moe_r_ge1_cold_stalls,
            self.heap_delta.blocks_in_use,
            self.heap_delta.size_in_use,
            self.swapouts_delta,
            self.pageouts_delta,
        });
    }
};

pub const GraphDecodeBenchError = graph.GraphLoadError || graph.GraphStepError || vmstat.Error || vmstat.SwapError || error{InvalidConfig};

pub fn runGraphDecodeBenchFromFile(
    allocator: std.mem.Allocator,
    amf_path: [:0]const u8,
    cfg: graph.GraphConfig,
    opts: GraphDecodeBenchOptions,
) GraphDecodeBenchError!GraphDecodeBenchResult {
    if (opts.decode_steps == 0 or opts.prefill_tokens + 1 >= opts.max_seq_len) return error.InvalidConfig;

    const load_t0 = try vmstat.nowNs();
    var model = try graph.GraphModel.loadFromFile(allocator, amf_path, cfg, .{
        .max_seq_len = opts.max_seq_len,
        .max_tree_nodes = 16,
        .wired_budget_override = opts.budget_override,
        .ple_paths = opts.ple_paths,
        .loop_adapter_amf_path = opts.loop_adapter_amf_path,
        .max_loops_override = opts.max_loops_override,
        .enable_loop_early_exit = opts.enable_loop_early_exit,
        .cold_slot_count = opts.cold_slot_count,
    });
    defer model.deinit(allocator);
    const load_t1 = try vmstat.nowNs();

    const prompt_toks = try allocator.alloc(u32, opts.prefill_tokens);
    defer allocator.free(prompt_toks);
    for (0..opts.prefill_tokens) |p| {
        prompt_toks[p] = @intCast((p * 17 + 5) % cfg.vocab_size);
    }

    const vm_before = try vmstat.Snapshot.capture();
    const pre_t0 = try vmstat.nowNs();
    var cur_tok = try model.prefillChunk(prompt_toks, 0, true);
    const pre_t1 = try vmstat.nowNs();

    const eval_pos: u32 = opts.prefill_tokens;
    for (0..opts.warmup_steps) |_| {
        cur_tok = try model.stepDecode(cur_tok, eval_pos);
    }

    model.resetLoopIsolationStats();
    model.resetColdTelemetry();

    const heap_before = try vmstat.Snapshot.capture();
    const dec_t0 = try vmstat.nowNs();
    for (0..opts.decode_steps) |_| {
        cur_tok = try model.stepDecode(cur_tok, eval_pos);
    }
    const dec_t1 = try vmstat.nowNs();
    const heap_after = try vmstat.Snapshot.capture();
    const vm_after = try vmstat.Snapshot.capture();
    try vmstat.checkNoSwap(vm_before, vm_after);

    const seq_len: u32 = eval_pos + 1;
    return .{
        .attention_summary = cfg.axes.attentionSummaryLabel(),
        .ffn_label = cfg.axes.ffn_kind.label(),
        .residual_label = cfg.axes.residual_kind.label(),
        .loop_label = cfg.axes.loop_schedule.mode.label(),
        .measured_layers = cfg.num_hidden_layers,
        .effective_loops = model.effective_max_loops,
        .seq_len = seq_len,
        .decode_steps = opts.decode_steps,
        .shard_count = model.shardCount(),
        .wired_bytes = model.totalWiredBytes(),
        .load_ms = @as(f64, @floatFromInt(load_t1 - load_t0)) / 1_000_000.0,
        .prefill_ms = @as(f64, @floatFromInt(pre_t1 - pre_t0)) / 1_000_000.0,
        .decode_ns = dec_t1 - dec_t0,
        .bytes_per_tok = model.bytesPerDecodeToken(seq_len),
        .loop_isolation = model.loopIsolationStats(),
        .ple_stats = model.pleStats(),
        .cold_stats = model.coldExpertStats(),
        .heap_delta = vmstat.heapDelta(heap_before, heap_after),
        .swapouts_delta = vmstat.Snapshot.swapoutDelta(vm_before, vm_after),
        .pageouts_delta = vmstat.Snapshot.pageoutDelta(vm_before, vm_after),
    };
}

pub const M6Config = struct {
    // Section 1: G8 GDN, G9 QSA, G10 GR, G7 512E MoE
    gdn_k_heads: u32 = 16,
    gdn_v_heads: u32 = 32,
    gdn_head_k: u32 = 128,
    gdn_head_v: u32 = 128,
    gdn_prefill_seq: u32 = 256,
    gdn_reps: u32 = 32,

    qsa_q_heads: u32 = 16,
    qsa_kv_heads: u32 = 4,
    qsa_head_dim: u32 = 128,
    qsa_block_size: u32 = 16,
    qsa_top_k_blocks: u32 = 8,
    qsa_full_kv_len: u32 = 512,
    qsa_swa_kv_len: u32 = 128,
    qsa_reps: u32 = 32,

    gr_hidden: u32 = 2048,
    gr_reps: u32 = 48,

    moe512_num_experts: u32 = 512,
    moe512_top_k: u32 = 10,
    moe512_hidden: u32 = 1024,
    moe512_inter: u32 = 256,
    moe512_shared_inter: u32 = 512,
    moe512_reps: u32 = 16,

    // Section 2: C10 PLE F_NOCACHE + r>=1 zero-NVMe isolation
    ple_num_shards: u32 = 4,
    ple_rows_per_shard: u32 = 2048,
    ple_embed_dim: u32 = 256,
    ple_cache_pages: usize = 8,
    ple_cold_queries: u32 = 48,
    ple_hit_queries: u32 = 64,
    ple_loop_steps: u32 = 16,

    // Section 3: End-to-end scale-qwen4-512e + MTP spec-decode
    qwen4_hidden: u32 = 128,
    qwen4_vocab: u32 = 256,
    qwen4_moe_inter: u32 = 128,
    qwen4_shared_inter: u32 = 128,
    qwen4_num_experts: u32 = 512,
    qwen4_top_k: u32 = 10,
    qwen4_cold_per_layer: u32 = 32,
    qwen4_ple_shards: u32 = 4,
    qwen4_ple_rows: u64 = 256,
    qwen4_gen_tokens: u32 = 16,

    // Section 4: D12 SelfSpecRecurrent sweep (R = 1, 2, 3, 4)
    loop_hidden: u32 = 1024,
    loop_inter: u32 = 3584,
    loop_vocab: u32 = 1024,
    loop_num_layers: u32 = 8, // 1 Prelude + 6 Core + 1 Coda (~50 MiB q4 Core > L2/SLC)
    loop_num_heads: u32 = 8,
    loop_num_kv_heads: u32 = 2,
    loop_head_dim: u32 = 128,
    loop_use_q4: bool = true,
    loop_serial_steps: u32 = 12,
    loop_batch_reps: u32 = 8,
    loop_spec_tokens: u32 = 16,

    // Section 5: D13 Placement.cpu_sme2 vs .gpu
    d13_scale_down: u32 = 1, // 1 for production shapes; 8 for tiny unit test
    d13_iters: u32 = 15,
    d13_warmups: u32 = 3,

    timeout_ms: u64 = 30_000,
};

pub const M6OpsReport = struct {
    // G8 GDN
    gdn_prefill_seq: u32,
    gdn_prefill_us: f64,
    gdn_prefill_tok_per_s: f64,
    gdn_prefill_gbs: f64,
    gdn_decode_us: f64,
    gdn_decode_gbs: f64,
    gdn_stmp_us: f64,
    gdn_commit_us: f64,
    gdn_stmp_isolated: bool,

    // G9 QSA
    qsa_mrope_us: f64,
    qsa_full512_r0_us: f64,
    qsa_full512_reuse_us: f64,
    qsa_swa128_r0_us: f64,
    qsa_swa128_reuse_us: f64,

    // G10 4-branch GR
    gr_hidden: u32,
    gr_m1_us: f64,
    gr_m1_gbs: f64,
    gr_m16_us: f64,
    gr_m16_gbs: f64,

    // G7 512-expert top-10 + shared_expert MoE
    moe512_num_experts: u32,
    moe512_top_k: u32,
    moe512_wired_mib: u64,
    moe512_m1_us: f64,
    moe512_m1_gbs: f64,
    moe512_m1_unique: u32,
    moe512_m16_us: f64,
    moe512_m16_gbs: f64,
    moe512_m16_unique: u32,
    moe512_dedup_ratio: f64,
};

pub const M6PleReport = struct {
    num_shards: u32,
    total_rows: u64,
    embed_dim: u32,
    wired_kib: u64,
    cold_p50_us: f64,
    cold_p99_us: f64,
    hit_p50_us: f64,
    r0_nvme_reads: u64,
    r_ge1_nvme_reads: u64,
    r_ge1_scratch_reuses: u64,
};

pub const M6ScaleQwen4Report = struct {
    num_experts: u32,
    top_k: u32,
    cold_per_layer: u32,
    ple_shards: u32,
    wired_mib: u64,
    gen_tokens: u32,
    serial_tok_per_s: f64,
    mtp_tok_per_s: f64,
    mtp_tau: f64,
    mtp_placement: draft_mod.Placement,
    bit_exact_mtp: bool,
    cold_hits: u64,
    cold_misses: u64,
    ple_nvme_reads: u64,
    ple_r_ge1_nvme_reads: u64,
};

pub const M6SelfSpecRow = struct {
    loops: u32,
    serial_ms_per_tok: f64,
    serial_tok_per_s: f64,
    ratio_vs_r1: f64,
    verify_m4_ms: f64,
    verify_m8_ms: f64,
    verify_m12_ms: f64,
    verify_m16_ms: f64,
    breakeven_m: u32, // 0 if > 16
    spec_m8_tok_per_s: f64,
    spec_m8_tau: f64,
    spec_m16_tok_per_s: f64,
    spec_m16_tau: f64,
    bit_exact: bool,
};

pub const M6SelfSpecReport = struct {
    core_layers: u32,
    core_weight_mib: f64,
    total_wired_mib: u64,
    rows: [4]M6SelfSpecRow,
};

pub const M6SmeDraftRow = struct {
    label: []const u8,
    dtype_label: []const u8,
    n: u32,
    k: u32,
    m: u32,
    weight_bytes: usize,
    fits_l2: bool,
    sme2_us: f64,
    gpu_us: f64,
    prefer_sme2: bool,
};

pub const M6Report = struct {
    ops: M6OpsReport,
    ple: M6PleReport,
    scale_qwen4: M6ScaleQwen4Report,
    self_spec: M6SelfSpecReport,
    d13_rows: [12]M6SmeDraftRow,
    vm_before: vmstat.Snapshot,
    vm_after: vmstat.Snapshot,

    pub fn write(self: *const M6Report, out: *std.Io.Writer) std.Io.Writer.Error!void {
        try out.writeAll("=== Milestone M6 Benchmark Suite (Qwen3.8-Flash-Next + LoopSchedule + D12/D13) ===\n\n");

        // 1. Operators (G8, G9, G10, G7)
        try out.writeAll("--- 1. Hybrid-Sparse Operators (G8 GDN, G9 QSA, G10 4-Branch GR, G7 512E MoE) ---\n");
        try out.print(
            \\g8.gdn.prefill_seq               {d}
            \\g8.gdn.prefill_us                {d:.1}
            \\g8.gdn.prefill_tok_per_s         {d:.1}
            \\g8.gdn.prefill_gbs               {d:.2}
            \\g8.gdn.decode_m1_us              {d:.1}
            \\g8.gdn.decode_m1_gbs             {d:.2}
            \\g8.gdn.stmp_eval_us              {d:.1}
            \\g8.gdn.stmp_commit_us            {d:.1}
            \\g8.gdn.stmp_isolated             {}
            \\g9.qsa.mrope_yarn_us             {d:.1}
            \\g9.qsa.full512_r0_us             {d:.1}
            \\g9.qsa.full512_reuse_us          {d:.1}
            \\g9.qsa.full512_reuse_speedup     {d:.2}x
            \\g9.qsa.swa128_r0_us              {d:.1}
            \\g9.qsa.swa128_reuse_us           {d:.1}
            \\g9.qsa.swa128_reuse_speedup      {d:.2}x
            \\g10.gr.hidden                    {d}
            \\g10.gr.m1_us                     {d:.1}
            \\g10.gr.m1_gbs                    {d:.2}
            \\g10.gr.m16_us                    {d:.1}
            \\g10.gr.m16_gbs                   {d:.2}
            \\g7.moe512.experts                {d} (top-{d} + shared_expert)
            \\g7.moe512.wired_mib              {d}
            \\g7.moe512.m1_us                  {d:.1}
            \\g7.moe512.m1_gbs                 {d:.2}
            \\g7.moe512.m1_unique_experts      {d}
            \\g7.moe512.m16_us                 {d:.1}
            \\g7.moe512.m16_gbs                {d:.2}
            \\g7.moe512.m16_unique_experts     {d}
            \\g7.moe512.m16_dedup_ratio        {d:.2}x
            \\
        , .{
            self.ops.gdn_prefill_seq,
            self.ops.gdn_prefill_us,
            self.ops.gdn_prefill_tok_per_s,
            self.ops.gdn_prefill_gbs,
            self.ops.gdn_decode_us,
            self.ops.gdn_decode_gbs,
            self.ops.gdn_stmp_us,
            self.ops.gdn_commit_us,
            self.ops.gdn_stmp_isolated,
            self.ops.qsa_mrope_us,
            self.ops.qsa_full512_r0_us,
            self.ops.qsa_full512_reuse_us,
            if (self.ops.qsa_full512_reuse_us > 0.0) self.ops.qsa_full512_r0_us / self.ops.qsa_full512_reuse_us else 0.0,
            self.ops.qsa_swa128_r0_us,
            self.ops.qsa_swa128_reuse_us,
            if (self.ops.qsa_swa128_reuse_us > 0.0) self.ops.qsa_swa128_r0_us / self.ops.qsa_swa128_reuse_us else 0.0,
            self.ops.gr_hidden,
            self.ops.gr_m1_us,
            self.ops.gr_m1_gbs,
            self.ops.gr_m16_us,
            self.ops.gr_m16_gbs,
            self.ops.moe512_num_experts,
            self.ops.moe512_top_k,
            self.ops.moe512_wired_mib,
            self.ops.moe512_m1_us,
            self.ops.moe512_m1_gbs,
            self.ops.moe512_m1_unique,
            self.ops.moe512_m16_us,
            self.ops.moe512_m16_gbs,
            self.ops.moe512_m16_unique,
            self.ops.moe512_dedup_ratio,
        });

        // 2. C10 PLE
        try out.writeAll("\n--- 2. C10 F_NOCACHE NVMe PLE Store & Loop Isolation (r >= 1) ---\n");
        try out.print(
            \\c10.ple.shards                   {d}
            \\c10.ple.total_rows               {d}
            \\c10.ple.embed_dim                {d}
            \\c10.ple.wired_kib                {d}
            \\c10.ple.cold_nvme_p50_us         {d:.2}
            \\c10.ple.cold_nvme_p99_us         {d:.2}
            \\c10.ple.wired_lru_hit_p50_us     {d:.2}
            \\c10.ple.loop_r0_nvme_reads       {d}
            \\c10.ple.loop_r_ge1_nvme_reads    {d}
            \\c10.ple.loop_r_ge1_reuses        {d}
            \\
        , .{
            self.ple.num_shards,
            self.ple.total_rows,
            self.ple.embed_dim,
            self.ple.wired_kib,
            self.ple.cold_p50_us,
            self.ple.cold_p99_us,
            self.ple.hit_p50_us,
            self.ple.r0_nvme_reads,
            self.ple.r_ge1_nvme_reads,
            self.ple.r_ge1_scratch_reuses,
        });

        // 3. scale-qwen4-512e + MTP
        try out.writeAll("\n--- 3. End-to-End scale-qwen4-512e (3:1 GDN:QSA, 4-Branch GR, 512E+Shared, PLE, Cold, MTP) ---\n");
        try out.print(
            \\qwen4_512e.experts               {d} (top-{d} + shared, {d} cold/layer)
            \\qwen4_512e.ple_shards            {d}
            \\qwen4_512e.wired_mib             {d}
            \\qwen4_512e.gen_tokens            {d}
            \\qwen4_512e.serial_tok_per_s      {d:.2}
            \\qwen4_512e.mtp_tok_per_s         {d:.2}
            \\qwen4_512e.mtp_mean_accept_tau   {d:.3}
            \\qwen4_512e.mtp_placement         {s}
            \\qwen4_512e.mtp_bit_exact_t0      {}
            \\qwen4_512e.cold_hits             {d}
            \\qwen4_512e.cold_misses           {d}
            \\qwen4_512e.ple_nvme_reads        {d}
            \\qwen4_512e.ple_r_ge1_nvme_reads  {d}
            \\
        , .{
            self.scale_qwen4.num_experts,
            self.scale_qwen4.top_k,
            self.scale_qwen4.cold_per_layer,
            self.scale_qwen4.ple_shards,
            self.scale_qwen4.wired_mib,
            self.scale_qwen4.gen_tokens,
            self.scale_qwen4.serial_tok_per_s,
            self.scale_qwen4.mtp_tok_per_s,
            self.scale_qwen4.mtp_tau,
            self.scale_qwen4.mtp_placement.label(),
            self.scale_qwen4.bit_exact_mtp,
            self.scale_qwen4.cold_hits,
            self.scale_qwen4.cold_misses,
            self.scale_qwen4.ple_nvme_reads,
            self.scale_qwen4.ple_r_ge1_nvme_reads,
        });

        // 4. D12 SelfSpecRecurrent sweep
        try out.writeAll("\n--- 4. D12 SelfSpecRecurrent Sweep (R = 1, 2, 3, 4; Core > L2/SLC) ---\n");
        try out.print(
            \\d12.core_layers                  {d} ({d:.2} MiB q4 Core, {d} MiB total wired)
            \\R   serial_ms   serial_tok/s  vs_R1   vM4_ms  vM8_ms  vM12_ms vM16_ms break_M  specM8_tok/s (tau)   specM16_tok/s (tau)  t0_exact
            \\
        , .{
            self.self_spec.core_layers,
            self.self_spec.core_weight_mib,
            self.self_spec.total_wired_mib,
        });
        for (self.self_spec.rows) |row| {
            try out.print(
                "{d:<3} {d:>8.3}    {d:>10.2}  {d:>5.2}x  {d:>6.2}  {d:>6.2}  {d:>6.2}  {d:>6.2}  {d:>5}    {d:>9.2} ({d:.2})    {d:>9.2} ({d:.2})   {}\n",
                .{
                    row.loops,
                    row.serial_ms_per_tok,
                    row.serial_tok_per_s,
                    row.ratio_vs_r1,
                    row.verify_m4_ms,
                    row.verify_m8_ms,
                    row.verify_m12_ms,
                    row.verify_m16_ms,
                    row.breakeven_m,
                    row.spec_m8_tok_per_s,
                    row.spec_m8_tau,
                    row.spec_m16_tok_per_s,
                    row.spec_m16_tau,
                    row.bit_exact,
                },
            );
        }

        // 5. D13 cpu_sme2 vs gpu
        try out.writeAll("\n--- 5. D13 Placement.cpu_sme2 vs .gpu (L2-Resident Draft Projections) ---\n");
        try out.writeAll("subgraph                    dtype  N     K     M   w_kib    l2    sme2_us   gpu_us  gpu/sme  winner\n");
        for (self.d13_rows) |row| {
            const ratio = if (row.sme2_us > 0.0) row.gpu_us / row.sme2_us else 0.0;
            try out.print(
                "{s:<27} {s:<5} {d:>5} {d:>5} {d:>2} {d:>7.1}  {s:<4}  {d:>8.1} {d:>8.1}  {d:>6.2}x  {s}\n",
                .{
                    row.label,
                    row.dtype_label,
                    row.n,
                    row.k,
                    row.m,
                    @as(f64, @floatFromInt(row.weight_bytes)) / 1024.0,
                    if (row.fits_l2) "yes" else "no",
                    row.sme2_us,
                    row.gpu_us,
                    ratio,
                    if (row.prefer_sme2) "cpu_sme2" else "gpu",
                },
            );
        }

        const sw_delta = vmstat.Snapshot.swapoutDelta(self.vm_before, self.vm_after);
        const pg_delta = vmstat.Snapshot.pageoutDelta(self.vm_before, self.vm_after);
        try out.print(
            \\\nm6.swap.swapouts_delta           {d}
            \\m6.swap.pageouts_delta           {d}
            \\m6.swap.status                   {s}
            \\
        , .{
            sw_delta,
            pg_delta,
            if (sw_delta == 0) "PASS (no swap)" else "FAIL (swap detected)",
        });
    }
};

fn removeDirRecursiveBestEffort(allocator: std.mem.Allocator, dir_path: []const u8) void {
    const dir_z = std.fmt.allocPrintSentinel(allocator, "{s}", .{dir_path}, 0) catch return;
    defer allocator.free(dir_z);
    if (std.c.opendir(dir_z.ptr)) |dp| {
        defer _ = std.c.closedir(dp);
        while (std.c.readdir(dp)) |ent| {
            const name = std.mem.sliceTo(&ent.name, 0);
            if (std.mem.eql(u8, name, ".") or std.mem.eql(u8, name, "..")) continue;
            const child = std.fmt.allocPrintSentinel(allocator, "{s}/{s}", .{ dir_path, name }, 0) catch continue;
            defer allocator.free(child);
            _ = std.c.unlink(child.ptr);
        }
    }
    _ = std.c.rmdir(dir_z.ptr);
}

fn makeM6TempDir(allocator: std.mem.Allocator, tag: []const u8) ![]u8 {
    const base_env = std.c.getenv("TMPDIR");
    const root: []const u8 = if (base_env) |p| std.mem.span(p) else "/Volumes/Akari/.tmp";
    const root_z = try std.fmt.allocPrintSentinel(allocator, "{s}", .{root}, 0);
    defer allocator.free(root_z);
    _ = std.c.mkdir(root_z.ptr, 0o755);
    const dir = try std.fmt.allocPrint(allocator, "{s}/aminfer_m6_{s}_{d}", .{ root, tag, std.c.getpid() });
    errdefer allocator.free(dir);
    const dir_z = try std.fmt.allocPrintSentinel(allocator, "{s}", .{dir}, 0);
    defer allocator.free(dir_z);
    _ = std.c.mkdir(dir_z.ptr, 0o755);
    return dir;
}

fn runM6OpsBench(allocator: std.mem.Allocator, cfg: M6Config) !M6OpsReport {
    var device = try metal.Device.open();
    defer device.close();
    const facts = try sysinfo.Facts.read();
    const dev_info = try device.info();
    const wired_budget = budget_mod.Budget.fromHost(facts, dev_info, .{});
    var library = try metal.Library.openEmbedded(device);
    defer library.close();
    var compute = try metal.Compute.init(device);
    defer compute.close();

    var hp = try hybrid.HybridPipelines.init(device, library);
    defer hp.close();
    var mp = try moe.MoePipelines.init(device, library);
    defer mp.close();

    // --- 1A. G8 GDN: prefill (seq), decode (M=1), S_tmp isolation + commit ---
    const gdn_spec: hybrid.GdnStateSpec = .{
        .num_gdn_layers = 1,
        .num_v_heads = cfg.gdn_v_heads,
        .head_k_dim = cfg.gdn_head_k,
        .head_v_dim = cfg.gdn_head_v,
    };
    const gdn_state_bytes = try hybrid.GdnStateArena.requiredBytes(gdn_spec);
    const qk_dim: usize = @as(usize, cfg.gdn_k_heads) * cfg.gdn_head_k;
    const v_dim: usize = @as(usize, cfg.gdn_v_heads) * cfg.gdn_head_v;
    const seq_u: usize = cfg.gdn_prefill_seq;

    var gdn_arena = try arena_mod.WiredArena.init(
        try page.alignUp(
            gdn_state_bytes +
                amf.align64(seq_u * qk_dim * 2) * 2 +
                amf.align64(seq_u * v_dim * 2) * 2 +
                amf.align64(seq_u * cfg.gdn_v_heads * 4) * 2 +
                65536,
        ),
        wired_budget,
    );
    defer gdn_arena.deinit();
    var gdn_buf = try metal.Buffer.wrap(device, gdn_arena.bytes());
    defer gdn_buf.close();
    try compute.makeResident(&.{gdn_buf});
    defer compute.removeResident(&.{gdn_buf}) catch |err| std.debug.panic("removeResident failed: {}", .{err});

    var gdn_states = try hybrid.GdnStateArena.init(&gdn_arena, gdn_spec);
    gdn_states.resetAll();
    const gdn_q = try gdn_arena.allocAligned(seq_u * qk_dim * 2, 64);
    const gdn_k = try gdn_arena.allocAligned(seq_u * qk_dim * 2, 64);
    const gdn_v = try gdn_arena.allocAligned(seq_u * v_dim * 2, 64);
    const gdn_y = try gdn_arena.allocAligned(seq_u * v_dim * 2, 64);
    const gdn_g = try gdn_arena.allocAligned(seq_u * cfg.gdn_v_heads * 4, 64);
    const gdn_beta = try gdn_arena.allocAligned(seq_u * cfg.gdn_v_heads * 4, 64);
    fillDeterministicF16Slice(gdn_q, 1001, 0.15);
    fillDeterministicF16Slice(gdn_k, 1002, 0.15);
    fillDeterministicF16Slice(gdn_v, 1003, 0.15);
    {
        const gs = std.mem.bytesAsSlice(f32, gdn_g);
        const bs = std.mem.bytesAsSlice(f32, gdn_beta);
        for (gs) |*v| v.* = -0.05;
        for (bs) |*v| v.* = -1.0;
    }

    const gdn_q_b: metal.BufferBinding = .{ .buffer = gdn_buf, .offset = gdn_arena.offsetOf(gdn_q.ptr) };
    const gdn_k_b: metal.BufferBinding = .{ .buffer = gdn_buf, .offset = gdn_arena.offsetOf(gdn_k.ptr) };
    const gdn_v_b: metal.BufferBinding = .{ .buffer = gdn_buf, .offset = gdn_arena.offsetOf(gdn_v.ptr) };
    const gdn_y_b: metal.BufferBinding = .{ .buffer = gdn_buf, .offset = gdn_arena.offsetOf(gdn_y.ptr) };
    const gdn_g_b: metal.BufferBinding = .{ .buffer = gdn_buf, .offset = gdn_arena.offsetOf(gdn_g.ptr) };
    const gdn_beta_b: metal.BufferBinding = .{ .buffer = gdn_buf, .offset = gdn_arena.offsetOf(gdn_beta.ptr) };
    const st_comm = try gdn_states.committedBinding(gdn_buf, 0);
    const st_tmp = try gdn_states.scratchBinding(gdn_buf, 0);

    // Warmup + timed GDN prefill
    const prefill_reps: u32 = @max(@as(u32, 2), cfg.gdn_reps / 4);
    for (0..2) |_| {
        var frame = try compute.beginFrame(cfg.timeout_ms);
        errdefer frame.abort();
        try hp.encodeGdnPrefill(&frame, .{
            .q = gdn_q_b,
            .k = gdn_k_b,
            .v = gdn_v_b,
            .g = gdn_g_b,
            .beta = gdn_beta_b,
            .state_in = st_comm,
            .state_out = st_comm,
            .out = gdn_y_b,
            .seq_len = cfg.gdn_prefill_seq,
            .num_k_heads = cfg.gdn_k_heads,
            .num_v_heads = cfg.gdn_v_heads,
            .head_k_dim = cfg.gdn_head_k,
            .head_v_dim = cfg.gdn_head_v,
        }, false);
        try compute.wait(try frame.commit(), cfg.timeout_ms);
    }
    const gdn_pre_t0 = try vmstat.nowNs();
    for (0..prefill_reps) |_| {
        var frame = try compute.beginFrame(cfg.timeout_ms);
        errdefer frame.abort();
        try hp.encodeGdnPrefill(&frame, .{
            .q = gdn_q_b,
            .k = gdn_k_b,
            .v = gdn_v_b,
            .g = gdn_g_b,
            .beta = gdn_beta_b,
            .state_in = st_comm,
            .state_out = st_comm,
            .out = gdn_y_b,
            .seq_len = cfg.gdn_prefill_seq,
            .num_k_heads = cfg.gdn_k_heads,
            .num_v_heads = cfg.gdn_v_heads,
            .head_k_dim = cfg.gdn_head_k,
            .head_v_dim = cfg.gdn_head_v,
        }, false);
        try compute.wait(try frame.commit(), cfg.timeout_ms);
    }
    const gdn_pre_t1 = try vmstat.nowNs();
    const gdn_prefill_us = @as(f64, @floatFromInt(gdn_pre_t1 - gdn_pre_t0)) / (@as(f64, @floatFromInt(prefill_reps)) * 1000.0);
    const single_state_bytes: u64 = @as(u64, gdn_states.layer_state_elems) * 4;
    const prefill_alg_bytes: u64 = 2 * single_state_bytes + @as(u64, seq_u) * (2 * qk_dim * 2 + 2 * v_dim * 2 + cfg.gdn_v_heads * 8);
    const gdn_prefill_gbs = @as(f64, @floatFromInt(prefill_alg_bytes)) / (gdn_prefill_us * 1000.0);
    const gdn_prefill_tok_per_s = @as(f64, @floatFromInt(cfg.gdn_prefill_seq)) * 1_000_000.0 / gdn_prefill_us;

    // Timed GDN decode (M=1 in-place S_t -> S_t)
    const gdn_dec_t0 = try vmstat.nowNs();
    {
        var frame = try compute.beginFrame(cfg.timeout_ms);
        errdefer frame.abort();
        for (0..cfg.gdn_reps) |rep| {
            try hp.encodeGdnDecode(&frame, .{
                .q = gdn_q_b,
                .k = gdn_k_b,
                .v = gdn_v_b,
                .g = gdn_g_b,
                .beta = gdn_beta_b,
                .state_in = st_comm,
                .state_out = st_comm,
                .out = gdn_y_b,
                .num_k_heads = cfg.gdn_k_heads,
                .num_v_heads = cfg.gdn_v_heads,
                .head_k_dim = cfg.gdn_head_k,
                .head_v_dim = cfg.gdn_head_v,
            }, rep > 0);
        }
        try compute.wait(try frame.commit(), cfg.timeout_ms);
    }
    const gdn_dec_t1 = try vmstat.nowNs();
    const gdn_decode_us = @as(f64, @floatFromInt(gdn_dec_t1 - gdn_dec_t0)) / (@as(f64, @floatFromInt(cfg.gdn_reps)) * 1000.0);
    const gdn_dec_bytes: u64 = 2 * single_state_bytes + (2 * qk_dim * 2 + 2 * v_dim * 2 + cfg.gdn_v_heads * 8);
    const gdn_decode_gbs = @as(f64, @floatFromInt(gdn_dec_bytes)) / (gdn_decode_us * 1000.0);

    // Loop-isolated S_tmp evaluation + verification + encodeCommitScratchState
    const snap_before = try allocator.dupe(f32, try gdn_states.committedLayerSlice(0));
    defer allocator.free(snap_before);
    const gdn_stmp_t0 = try vmstat.nowNs();
    {
        var frame = try compute.beginFrame(cfg.timeout_ms);
        errdefer frame.abort();
        for (0..cfg.gdn_reps) |rep| {
            try hp.encodeGdnDecode(&frame, .{
                .q = gdn_q_b,
                .k = gdn_k_b,
                .v = gdn_v_b,
                .g = gdn_g_b,
                .beta = gdn_beta_b,
                .state_in = st_comm,
                .state_out = st_tmp,
                .out = gdn_y_b,
                .num_k_heads = cfg.gdn_k_heads,
                .num_v_heads = cfg.gdn_v_heads,
                .head_k_dim = cfg.gdn_head_k,
                .head_v_dim = cfg.gdn_head_v,
            }, rep > 0);
        }
        try compute.wait(try frame.commit(), cfg.timeout_ms);
    }
    const gdn_stmp_t1 = try vmstat.nowNs();
    const gdn_stmp_us = @as(f64, @floatFromInt(gdn_stmp_t1 - gdn_stmp_t0)) / (@as(f64, @floatFromInt(cfg.gdn_reps)) * 1000.0);
    const committed_after_stmp = try gdn_states.committedLayerSlice(0);
    const gdn_stmp_isolated = std.mem.eql(u8, std.mem.sliceAsBytes(snap_before), std.mem.sliceAsBytes(committed_after_stmp));

    const gdn_cm_t0 = try vmstat.nowNs();
    {
        var frame = try compute.beginFrame(cfg.timeout_ms);
        errdefer frame.abort();
        for (0..cfg.gdn_reps) |rep| {
            try hp.encodeCommitScratchState(&frame, &gdn_states, gdn_buf, 0, rep > 0);
        }
        try compute.wait(try frame.commit(), cfg.timeout_ms);
    }
    const gdn_cm_t1 = try vmstat.nowNs();
    const gdn_commit_us = @as(f64, @floatFromInt(gdn_cm_t1 - gdn_cm_t0)) / (@as(f64, @floatFromInt(cfg.gdn_reps)) * 1000.0);

    // --- 1B. G9 QSA: [11,11,10] MRoPE + YaRN + 512/128 HybridKvRing + r>=1 index reuse ---
    const qsa_qd: usize = @as(usize, cfg.qsa_q_heads) * cfg.qsa_head_dim;
    const qsa_kvd: usize = @as(usize, cfg.qsa_kv_heads) * cfg.qsa_head_dim;
    const full_blocks: u32 = (cfg.qsa_full_kv_len + cfg.qsa_block_size - 1) / cfg.qsa_block_size;
    const swa_blocks: u32 = (cfg.qsa_swa_kv_len + cfg.qsa_block_size - 1) / cfg.qsa_block_size;
    const full_tiles: u32 = (cfg.qsa_full_kv_len + hybrid.QSA_FULL_CHUNK_TOKENS - 1) / hybrid.QSA_FULL_CHUNK_TOKENS;

    const full_idx_spec: hybrid.QsaIndexBufferSpec = .{
        .max_nodes = 16,
        .num_kv_heads = cfg.qsa_kv_heads,
        .top_k_blocks = cfg.qsa_top_k_blocks,
    };
    const qsa_arena_bytes = try page.alignUp(
        (try hybrid.HybridKvRing.tableRequiredBytes(full_tiles)) +
            (try hybrid.HybridKvRing.tableRequiredBytes(1)) +
            (try hybrid.QsaIndexBuffer.requiredBytes(full_idx_spec)) * 2 +
            amf.align64(@as(usize, full_blocks) * qsa_kvd * 2) +
            amf.align64(@as(usize, swa_blocks) * qsa_kvd * 2) +
            amf.align64(16 * qsa_qd * 2) * 2 +
            amf.align64(16 * qsa_kvd * 2) * 4 +
            amf.align64(@as(usize, cfg.qsa_head_dim) * 2) * 2 +
            65536,
    );
    var qsa_arena = try arena_mod.WiredArena.init(qsa_arena_bytes, wired_budget);
    defer qsa_arena.deinit();
    var qsa_buf = try metal.Buffer.wrap(device, qsa_arena.bytes());
    defer qsa_buf.close();
    try compute.makeResident(&.{qsa_buf});
    defer compute.removeResident(&.{qsa_buf}) catch |err| std.debug.panic("removeResident failed: {}", .{err});

    var full_ring = try hybrid.HybridKvRing.init(allocator, device, &qsa_arena, .{
        .mode = .full_chunk_512,
        .num_kv_heads = cfg.qsa_kv_heads,
        .head_dim = cfg.qsa_head_dim,
        .num_tiles = full_tiles,
    });
    defer full_ring.deinit(allocator);
    try full_ring.makeResident(compute);
    defer full_ring.removeResident(compute) catch |err| std.debug.panic("removeResident failed: {}", .{err});

    var swa_ring = try hybrid.HybridKvRing.init(allocator, device, &qsa_arena, .{
        .mode = .sliding_window_128,
        .num_kv_heads = cfg.qsa_kv_heads,
        .head_dim = cfg.qsa_head_dim,
        .num_tiles = 1,
    });
    defer swa_ring.deinit(allocator);
    try swa_ring.makeResident(compute);
    defer swa_ring.removeResident(compute) catch |err| std.debug.panic("removeResident failed: {}", .{err});

    const full_idx_buf = try hybrid.QsaIndexBuffer.init(&qsa_arena, full_idx_spec);
    const swa_idx_buf = try hybrid.QsaIndexBuffer.init(&qsa_arena, full_idx_spec);
    const full_bkeys = try qsa_arena.allocAligned(@as(usize, full_blocks) * qsa_kvd * 2, 64);
    const swa_bkeys = try qsa_arena.allocAligned(@as(usize, swa_blocks) * qsa_kvd * 2, 64);
    const qsa_q = try qsa_arena.allocAligned(16 * qsa_qd * 2, 64);
    const qsa_out = try qsa_arena.allocAligned(16 * qsa_qd * 2, 64);
    const qsa_k = try qsa_arena.allocAligned(16 * qsa_kvd * 2, 64);
    const qsa_v = try qsa_arena.allocAligned(16 * qsa_kvd * 2, 64);
    const qsa_tk = try qsa_arena.allocAligned(16 * qsa_kvd * 2, 64);
    const qsa_tv = try qsa_arena.allocAligned(16 * qsa_kvd * 2, 64);
    const qsa_qn = try qsa_arena.allocAligned(@as(usize, cfg.qsa_head_dim) * 2, 64);
    const qsa_kn = try qsa_arena.allocAligned(@as(usize, cfg.qsa_head_dim) * 2, 64);
    const qsa_pos3d = try qsa_arena.allocAligned(16 * @sizeOf(hybrid.Pos3D), 64);
    const qsa_masks = try qsa_arena.allocAligned(16 * @sizeOf(u64), 64);
    fillDeterministicF16Slice(full_bkeys, 2001, 0.2);
    fillDeterministicF16Slice(swa_bkeys, 2002, 0.2);
    fillDeterministicF16Slice(qsa_q, 2003, 0.2);
    fillDeterministicF16Slice(qsa_k, 2004, 0.2);
    fillDeterministicF16Slice(qsa_v, 2005, 0.2);
    fillDeterministicF16Slice(qsa_tk, 2006, 0.2);
    fillDeterministicF16Slice(qsa_tv, 2007, 0.2);
    fillConstantF16Slice(qsa_qn, 1.0);
    fillConstantF16Slice(qsa_kn, 1.0);
    {
        const p3 = std.mem.bytesAsSlice(hybrid.Pos3D, qsa_pos3d);
        const ms = std.mem.bytesAsSlice(u64, qsa_masks);
        for (0..16) |i| {
            p3[i] = hybrid.Pos3D.text(cfg.qsa_full_kv_len + @as(u32, @intCast(i)));
            ms[i] = (@as(u64, 1) << @intCast(i + 1)) - 1;
        }
    }

    const qsa_q_b: metal.BufferBinding = .{ .buffer = qsa_buf, .offset = qsa_arena.offsetOf(qsa_q.ptr) };
    const qsa_out_b: metal.BufferBinding = .{ .buffer = qsa_buf, .offset = qsa_arena.offsetOf(qsa_out.ptr) };
    const qsa_k_b: metal.BufferBinding = .{ .buffer = qsa_buf, .offset = qsa_arena.offsetOf(qsa_k.ptr) };
    const qsa_v_b: metal.BufferBinding = .{ .buffer = qsa_buf, .offset = qsa_arena.offsetOf(qsa_v.ptr) };
    const qsa_tk_b: metal.BufferBinding = .{ .buffer = qsa_buf, .offset = qsa_arena.offsetOf(qsa_tk.ptr) };
    const qsa_tv_b: metal.BufferBinding = .{ .buffer = qsa_buf, .offset = qsa_arena.offsetOf(qsa_tv.ptr) };
    const qsa_qn_b: metal.BufferBinding = .{ .buffer = qsa_buf, .offset = qsa_arena.offsetOf(qsa_qn.ptr) };
    const qsa_kn_b: metal.BufferBinding = .{ .buffer = qsa_buf, .offset = qsa_arena.offsetOf(qsa_kn.ptr) };
    const qsa_p3_b: metal.BufferBinding = .{ .buffer = qsa_buf, .offset = qsa_arena.offsetOf(qsa_pos3d.ptr) };
    const qsa_mk_b: metal.BufferBinding = .{ .buffer = qsa_buf, .offset = qsa_arena.offsetOf(qsa_masks.ptr) };
    const full_bk_b: metal.BufferBinding = .{ .buffer = qsa_buf, .offset = qsa_arena.offsetOf(full_bkeys.ptr) };
    const swa_bk_b: metal.BufferBinding = .{ .buffer = qsa_buf, .offset = qsa_arena.offsetOf(swa_bkeys.ptr) };

    // Time [11,11,10] MRoPE + YaRN (`encodeQsaTreeRope`, M=1)
    const mr_t0 = try vmstat.nowNs();
    {
        var frame = try compute.beginFrame(cfg.timeout_ms);
        errdefer frame.abort();
        for (0..cfg.qsa_reps) |rep| {
            try hp.encodeQsaTreeRope(&frame, .{
                .q = qsa_q_b,
                .k = qsa_k_b,
                .v = qsa_v_b,
                .q_norm = qsa_qn_b,
                .k_norm = qsa_kn_b,
                .positions_3d = qsa_p3_b,
                .k_tree_out = qsa_tk_b,
                .v_tree_out = qsa_tv_b,
                .num_nodes = 1,
                .num_q_heads = cfg.qsa_q_heads,
                .num_kv_heads = cfg.qsa_kv_heads,
                .head_dim = cfg.qsa_head_dim,
                .rotary_dim = cfg.qsa_head_dim,
                .mrope = hybrid.MropeSection.QWEN38_DEFAULT,
                .rms_eps = 1e-6,
                .rope_theta = 1_000_000.0,
                .yarn = .{ .factor = 4.0, .original_max_pos = 32768 },
                .norm_dtype = .f16,
            }, rep > 0);
        }
        try compute.wait(try frame.commit(), cfg.timeout_ms);
    }
    const mr_t1 = try vmstat.nowNs();
    const qsa_mrope_us = @as(f64, @floatFromInt(mr_t1 - mr_t0)) / (@as(f64, @floatFromInt(cfg.qsa_reps)) * 1000.0);

    // Time Full-512 ring r=0 vs r>=1 reuse
    const timeQsaRing = struct {
        fn call(
            comp: *metal.Compute,
            pipes: *const hybrid.HybridPipelines,
            ring: *const hybrid.HybridKvRing,
            idx_b: *const hybrid.QsaIndexBuffer,
            buf: metal.Buffer,
            q_b: metal.BufferBinding,
            bk_b: metal.BufferBinding,
            tk_b: metal.BufferBinding,
            tv_b: metal.BufferBinding,
            mk_b: metal.BufferBinding,
            out_b: metal.BufferBinding,
            c: M6Config,
            kv_len: u32,
            reuse: bool,
        ) !f64 {
            // Prime indices once before reuse benchmark so index buffer is valid.
            if (reuse) {
                var pf = try comp.beginFrame(c.timeout_ms);
                errdefer pf.abort();
                try pipes.encodeQsaAttention(&pf, .{
                    .q = q_b,
                    .kv_tiles = ring.tileTableBinding(buf),
                    .block_keys = bk_b,
                    .index_buf = idx_b,
                    .index_metal_buf = buf,
                    .k_tree = tk_b,
                    .v_tree = tv_b,
                    .ancestor_masks = mk_b,
                    .out = out_b,
                    .num_nodes = 1,
                    .num_q_heads = c.qsa_q_heads,
                    .num_kv_heads = c.qsa_kv_heads,
                    .head_dim = c.qsa_head_dim,
                    .kv_len = kv_len,
                    .window_size = ring.window_size,
                    .ring_capacity = ring.ring_capacity,
                    .chunk_tokens = ring.chunk_tokens,
                    .block_tokens = c.qsa_block_size,
                    .top_k_blocks = c.qsa_top_k_blocks,
                    .reuse_block_indices = false,
                }, false);
                try comp.wait(try pf.commit(), c.timeout_ms);
            }
            const t0 = try vmstat.nowNs();
            var frame = try comp.beginFrame(c.timeout_ms);
            errdefer frame.abort();
            for (0..c.qsa_reps) |rep| {
                try pipes.encodeQsaAttention(&frame, .{
                    .q = q_b,
                    .kv_tiles = ring.tileTableBinding(buf),
                    .block_keys = bk_b,
                    .index_buf = idx_b,
                    .index_metal_buf = buf,
                    .k_tree = tk_b,
                    .v_tree = tv_b,
                    .ancestor_masks = mk_b,
                    .out = out_b,
                    .num_nodes = 1,
                    .num_q_heads = c.qsa_q_heads,
                    .num_kv_heads = c.qsa_kv_heads,
                    .head_dim = c.qsa_head_dim,
                    .kv_len = kv_len,
                    .window_size = ring.window_size,
                    .ring_capacity = ring.ring_capacity,
                    .chunk_tokens = ring.chunk_tokens,
                    .block_tokens = c.qsa_block_size,
                    .top_k_blocks = c.qsa_top_k_blocks,
                    .reuse_block_indices = reuse,
                }, rep > 0);
            }
            try comp.wait(try frame.commit(), c.timeout_ms);
            const t1 = try vmstat.nowNs();
            return @as(f64, @floatFromInt(t1 - t0)) / (@as(f64, @floatFromInt(c.qsa_reps)) * 1000.0);
        }
    }.call;

    const qsa_full512_r0_us = try timeQsaRing(&compute, &hp, &full_ring, &full_idx_buf, qsa_buf, qsa_q_b, full_bk_b, qsa_tk_b, qsa_tv_b, qsa_mk_b, qsa_out_b, cfg, cfg.qsa_full_kv_len, false);
    const qsa_full512_reuse_us = try timeQsaRing(&compute, &hp, &full_ring, &full_idx_buf, qsa_buf, qsa_q_b, full_bk_b, qsa_tk_b, qsa_tv_b, qsa_mk_b, qsa_out_b, cfg, cfg.qsa_full_kv_len, true);
    const qsa_swa128_r0_us = try timeQsaRing(&compute, &hp, &swa_ring, &swa_idx_buf, qsa_buf, qsa_q_b, swa_bk_b, qsa_tk_b, qsa_tv_b, qsa_mk_b, qsa_out_b, cfg, cfg.qsa_swa_kv_len, false);
    const qsa_swa128_reuse_us = try timeQsaRing(&compute, &hp, &swa_ring, &swa_idx_buf, qsa_buf, qsa_q_b, swa_bk_b, qsa_tk_b, qsa_tv_b, qsa_mk_b, qsa_out_b, cfg, cfg.qsa_swa_kv_len, true);

    // --- 1C. G10 4-branch GR (M=1 and M=16) ---
    const gr_spec: hybrid.GrStreamSpec = .{
        .max_tokens = 16,
        .hidden_dim = cfg.gr_hidden,
        .dtype = .f16,
    };
    const gr_arena_bytes = try page.alignUp(
        (try hybrid.GrStreamArena.requiredBytes(gr_spec)) +
            amf.align64(16 * @as(usize, cfg.gr_hidden) * 2) * 3 +
            amf.align64(4 * @as(usize, cfg.gr_hidden) * 2) * 3 +
            amf.align64(16 * 16) * 3 +
            65536,
    );
    var gr_arena = try arena_mod.WiredArena.init(gr_arena_bytes, wired_budget);
    defer gr_arena.deinit();
    var gr_buf = try metal.Buffer.wrap(device, gr_arena.bytes());
    defer gr_buf.close();
    try compute.makeResident(&.{gr_buf});
    defer compute.removeResident(&.{gr_buf}) catch |err| std.debug.panic("removeResident failed: {}", .{err});

    const gr_stream = try hybrid.GrStreamArena.init(&gr_arena, gr_spec);
    const gr_x0 = try gr_arena.allocAligned(16 * @as(usize, cfg.gr_hidden) * 2, 64);
    const gr_xin = try gr_arena.allocAligned(16 * @as(usize, cfg.gr_hidden) * 2, 64);
    const gr_yd = try gr_arena.allocAligned(16 * @as(usize, cfg.gr_hidden) * 2, 64);
    const gr_wr = try gr_arena.allocAligned(4 * @as(usize, cfg.gr_hidden) * 2, 64);
    const gr_ww = try gr_arena.allocAligned(4 * @as(usize, cfg.gr_hidden) * 2, 64);
    const gr_wd = try gr_arena.allocAligned(4 * @as(usize, cfg.gr_hidden) * 2, 64);
    const gr_gr = try gr_arena.allocAligned(16 * 16, 64);
    const gr_gw = try gr_arena.allocAligned(16 * 16, 64);
    const gr_gd = try gr_arena.allocAligned(16 * 16, 64);
    fillDeterministicF16Slice(gr_x0, 3001, 0.2);
    fillDeterministicF16Slice(gr_yd, 3002, 0.2);
    fillDeterministicF16Slice(gr_wr, 3003, 0.1);
    fillDeterministicF16Slice(gr_ww, 3004, 0.1);
    fillDeterministicF16Slice(gr_wd, 3005, 0.1);

    const timeGr = struct {
        fn call(
            comp: *metal.Compute,
            pipes: *const hybrid.HybridPipelines,
            stream: *const hybrid.GrStreamArena,
            buf: metal.Buffer,
            arena_p: *const arena_mod.WiredArena,
            x0: []u8,
            xin: []u8,
            yd: []u8,
            wr: []u8,
            ww: []u8,
            wd: []u8,
            gr: []u8,
            gw: []u8,
            gd: []u8,
            m: u32,
            c: M6Config,
        ) !struct { us: f64, gbs: f64 } {
            const x0_b: metal.BufferBinding = .{ .buffer = buf, .offset = arena_p.offsetOf(x0.ptr) };
            const xin_b: metal.BufferBinding = .{ .buffer = buf, .offset = arena_p.offsetOf(xin.ptr) };
            const yd_b: metal.BufferBinding = .{ .buffer = buf, .offset = arena_p.offsetOf(yd.ptr) };
            const wr_b: metal.BufferBinding = .{ .buffer = buf, .offset = arena_p.offsetOf(wr.ptr) };
            const ww_b: metal.BufferBinding = .{ .buffer = buf, .offset = arena_p.offsetOf(ww.ptr) };
            const wd_b: metal.BufferBinding = .{ .buffer = buf, .offset = arena_p.offsetOf(wd.ptr) };
            const gr_b: metal.BufferBinding = .{ .buffer = buf, .offset = arena_p.offsetOf(gr.ptr) };
            const gw_b: metal.BufferBinding = .{ .buffer = buf, .offset = arena_p.offsetOf(gw.ptr) };
            const gd_b: metal.BufferBinding = .{ .buffer = buf, .offset = arena_p.offsetOf(gd.ptr) };
            const sb = stream.binding(buf);

            const t0 = try vmstat.nowNs();
            var frame = try comp.beginFrame(c.timeout_ms);
            errdefer frame.abort();
            for (0..c.gr_reps) |rep| {
                try pipes.encodeGrComputeGates(&frame, .{
                    .stream = sb,
                    .w_read = wr_b,
                    .w_write = ww_b,
                    .w_decay = wd_b,
                    .g_read_out = gr_b,
                    .g_write_out = gw_b,
                    .decay_out = gd_b,
                    .m = m,
                    .hidden_dim = c.gr_hidden,
                    .branch_token_stride = 16,
                    .weight_dtype = .f16,
                }, rep > 0);
                try pipes.encodeGrReadGate(&frame, .{
                    .stream = sb,
                    .g_read = gr_b,
                    .x_in = xin_b,
                    .m = m,
                    .hidden_dim = c.gr_hidden,
                    .branch_token_stride = 16,
                }, true);
                try pipes.encodeGrWriteGate(&frame, .{
                    .stream = sb,
                    .y_delta = yd_b,
                    .decay = gd_b,
                    .g_write = gw_b,
                    .m = m,
                    .hidden_dim = c.gr_hidden,
                    .branch_token_stride = 16,
                }, true);
                try pipes.encodeGrCollapse(&frame, .{
                    .stream = sb,
                    .out = x0_b,
                    .mode = .mean,
                    .m = m,
                    .hidden_dim = c.gr_hidden,
                    .branch_token_stride = 16,
                }, true);
            }
            try comp.wait(try frame.commit(), c.timeout_ms);
            const t1 = try vmstat.nowNs();
            const us = @as(f64, @floatFromInt(t1 - t0)) / (@as(f64, @floatFromInt(c.gr_reps)) * 1000.0);
            // Stream traffic across compute_gates + read + write + collapse:
            // read 4*m*H (gates) + read 4*m*H + write m*H (read_gate) + read/write 4*m*H + read m*H (write_gate) + read 4*m*H + write m*H (collapse) + 3*4*H weights
            const bytes: u64 = (@as(u64, m) * c.gr_hidden * 2) * (4 + 5 + 9 + 5) + 3 * 4 * @as(u64, c.gr_hidden) * 2;
            return .{ .us = us, .gbs = @as(f64, @floatFromInt(bytes)) / (us * 1000.0) };
        }
    }.call;

    const gr_m1 = try timeGr(&compute, &hp, &gr_stream, gr_buf, &gr_arena, gr_x0, gr_xin, gr_yd, gr_wr, gr_ww, gr_wd, gr_gr, gr_gw, gr_gd, 1, cfg);
    const gr_m16 = try timeGr(&compute, &hp, &gr_stream, gr_buf, &gr_arena, gr_x0, gr_xin, gr_yd, gr_wr, gr_ww, gr_wd, gr_gr, gr_gw, gr_gd, 16, cfg);

    // --- 1D. G7 512-expert top-10 + shared_expert MoE (M=1 and M=16 CSR dedup) ---
    const e_cnt = cfg.moe512_num_experts;
    const h512 = cfg.moe512_hidden;
    const inter512 = cfg.moe512_inter;
    const sh_inter512 = cfg.moe512_shared_inter;
    const q4_gu_qbytes: usize = @as(usize, inter512) * (h512 / 2);
    const q4_gu_sbbytes: usize = @as(usize, inter512) * (h512 / 64) * 4;
    const q4_dn_qbytes: usize = @as(usize, h512) * (inter512 / 2);
    const q4_dn_sbbytes: usize = @as(usize, h512) * (inter512 / 64) * 4;
    const single_exp_bytes: usize = amf.align64(q4_gu_qbytes) * 2 + amf.align64(q4_gu_sbbytes) * 2 + amf.align64(q4_dn_qbytes) + amf.align64(q4_dn_sbbytes);
    const sh_gu_qbytes: usize = @as(usize, sh_inter512) * (h512 / 2);
    const sh_gu_sbbytes: usize = @as(usize, sh_inter512) * (h512 / 64) * 4;
    const sh_dn_qbytes: usize = @as(usize, h512) * (sh_inter512 / 2);
    const sh_dn_sbbytes: usize = @as(usize, h512) * (sh_inter512 / 64) * 4;
    const sh_total_bytes: usize = amf.align64(sh_gu_qbytes) * 2 + amf.align64(sh_gu_sbbytes) * 2 + amf.align64(sh_dn_qbytes) + amf.align64(sh_dn_sbbytes);

    const ws_spec: moe.WorkspaceSpec = .{
        .max_tokens = 16,
        .num_experts = e_cnt,
        .top_k = cfg.moe512_top_k,
        .hidden_size = h512,
        .intermediate_size = inter512,
        .shared_intermediate_size = sh_inter512,
    };
    const ws_bytes = try moe.MoeWorkspace.requiredBytes(ws_spec);
    const moe_arena_bytes = try page.alignUp(
        @as(usize, e_cnt) * single_exp_bytes +
            sh_total_bytes +
            ws_bytes +
            amf.align64(@as(usize, e_cnt) * @sizeOf(moe.ExpertWeights)) +
            amf.align64(16 * @as(usize, e_cnt) * 4) +
            amf.align64(16 * @as(usize, h512) * 2) * 2 +
            amf.align64(@as(usize, h512) * 2) +
            131072,
    );
    var moe_arena = try arena_mod.WiredArena.init(moe_arena_bytes, wired_budget);
    defer moe_arena.deinit();
    var moe_buf = try metal.Buffer.wrap(device, moe_arena.bytes());
    defer moe_buf.close();
    try compute.makeResident(&.{moe_buf});
    defer compute.removeResident(&.{moe_buf}) catch |err| std.debug.panic("removeResident failed: {}", .{err});

    const moe_ws = try moe.MoeWorkspace.init(&moe_arena, ws_spec);
    const exp_tbl_bytes = try moe_arena.allocAligned(@as(usize, e_cnt) * @sizeOf(moe.ExpertWeights), 64);
    const exp_tbl = std.mem.bytesAsSlice(moe.ExpertWeights, exp_tbl_bytes);
    const weights_region = try moe_arena.allocAligned(@as(usize, e_cnt) * single_exp_bytes + sh_total_bytes, 64);
    // Deterministic fast fill for q4 weights + valid scale/bias
    @memset(weights_region, 0x37);
    const weights_base: metal.BufferBinding = .{ .buffer = moe_buf, .offset = moe_arena.offsetOf(weights_region.ptr) };
    const sb_s: u16 = @bitCast(@as(f16, 0.008));
    const sb_b: u16 = @bitCast(@as(f16, -0.06));
    const writeSb = struct {
        fn call(slice: []u8, s: u16, b: u16) void {
            var idx: usize = 0;
            while (idx + 4 <= slice.len) : (idx += 4) {
                std.mem.writeInt(u16, slice[idx..][0..2], s, .little);
                std.mem.writeInt(u16, slice[idx + 2 ..][0..2], b, .little);
            }
        }
    }.call;

    var cur_off: usize = 0;
    for (0..e_cnt) |e| {
        const gq = cur_off;
        cur_off += amf.align64(q4_gu_qbytes);
        const gsb = cur_off;
        writeSb(weights_region[gsb .. gsb + q4_gu_sbbytes], sb_s, sb_b);
        cur_off += amf.align64(q4_gu_sbbytes);
        const uq = cur_off;
        cur_off += amf.align64(q4_gu_qbytes);
        const usb = cur_off;
        writeSb(weights_region[usb .. usb + q4_gu_sbbytes], sb_s, sb_b);
        cur_off += amf.align64(q4_gu_sbbytes);
        const dq = cur_off;
        cur_off += amf.align64(q4_dn_qbytes);
        const dsb = cur_off;
        writeSb(weights_region[dsb .. dsb + q4_dn_sbbytes], sb_s, sb_b);
        cur_off += amf.align64(q4_dn_sbbytes);
        exp_tbl[e] = .{
            .gate_qweight_offset = @intCast(gq),
            .gate_sb_offset = @intCast(gsb),
            .up_qweight_offset = @intCast(uq),
            .up_sb_offset = @intCast(usb),
            .down_qweight_offset = @intCast(dq),
            .down_sb_offset = @intCast(dsb),
        };
    }
    const sh_gq = cur_off;
    cur_off += amf.align64(sh_gu_qbytes);
    const sh_gsb = cur_off;
    writeSb(weights_region[sh_gsb .. sh_gsb + sh_gu_sbbytes], sb_s, sb_b);
    cur_off += amf.align64(sh_gu_sbbytes);
    const sh_uq = cur_off;
    cur_off += amf.align64(sh_gu_qbytes);
    const sh_usb = cur_off;
    writeSb(weights_region[sh_usb .. sh_usb + sh_gu_sbbytes], sb_s, sb_b);
    cur_off += amf.align64(sh_gu_sbbytes);
    const sh_dq = cur_off;
    cur_off += amf.align64(sh_dn_qbytes);
    const sh_dsb = cur_off;
    writeSb(weights_region[sh_dsb .. sh_dsb + sh_dn_sbbytes], sb_s, sb_b);
    const shared_ew: moe.ExpertWeights = .{
        .gate_qweight_offset = @intCast(sh_gq),
        .gate_sb_offset = @intCast(sh_gsb),
        .up_qweight_offset = @intCast(sh_uq),
        .up_sb_offset = @intCast(sh_usb),
        .down_qweight_offset = @intCast(sh_dq),
        .down_sb_offset = @intCast(sh_dsb),
    };

    const moe_x = try moe_arena.allocAligned(16 * @as(usize, h512) * 2, 64);
    const moe_out = try moe_arena.allocAligned(16 * @as(usize, h512) * 2, 64);
    const moe_sgate = try moe_arena.allocAligned(@as(usize, h512) * 2, 64);
    const moe_logits = try moe_arena.allocAligned(16 * @as(usize, e_cnt) * 4, 64);
    fillDeterministicF16Slice(moe_x, 4001, 0.2);
    fillDeterministicF16Slice(moe_sgate, 4002, 0.1);
    // Realistic Zipf-clustered routing logits for 16 tree nodes (shared hot core + per-node diversity).
    {
        const ls = std.mem.bytesAsSlice(f32, moe_logits);
        var prng = std.Random.DefaultPrng.init(4042);
        const rng = prng.random();
        for (0..16) |node| {
            for (0..e_cnt) |e| {
                // First ~32 experts carry realistic hot-expert bias + per-node variation.
                const hot_bias: f32 = if (e < 24) 3.5 - @as(f32, @floatFromInt(e)) * 0.08 else 0.0;
                ls[node * e_cnt + e] = hot_bias + rng.float(f32) * 2.2 + @as(f32, @floatFromInt((node * 7 + e) % 13)) * 0.05;
            }
        }
    }

    const moe_x_b: metal.BufferBinding = .{ .buffer = moe_buf, .offset = moe_arena.offsetOf(moe_x.ptr) };
    const moe_out_b: metal.BufferBinding = .{ .buffer = moe_buf, .offset = moe_arena.offsetOf(moe_out.ptr) };
    const moe_sg_b: metal.BufferBinding = .{ .buffer = moe_buf, .offset = moe_arena.offsetOf(moe_sgate.ptr) };
    const moe_lg_b: metal.BufferBinding = .{ .buffer = moe_buf, .offset = moe_arena.offsetOf(moe_logits.ptr) };
    const moe_et_b: metal.BufferBinding = .{ .buffer = moe_buf, .offset = moe_arena.offsetOf(exp_tbl_bytes.ptr) };

    const timeMoe512 = struct {
        fn call(
            comp: *metal.Compute,
            pipes: *const moe.MoePipelines,
            ws: *const moe.MoeWorkspace,
            buf: metal.Buffer,
            x_b: metal.BufferBinding,
            out_b: metal.BufferBinding,
            lg_b: metal.BufferBinding,
            wb: metal.BufferBinding,
            et_b: metal.BufferBinding,
            sh: moe.ExpertWeights,
            sg_b: metal.BufferBinding,
            m: u32,
            c: M6Config,
            exp_b: usize,
            sh_b: usize,
        ) !struct { us: f64, gbs: f64, unique: u32 } {
            const args: moe.MoELayerArgs = .{
                .x = x_b,
                .router = .{ .logits = .{ .binding = lg_b, .dtype = .f32 } },
                .weights_base = wb,
                .expert_table = et_b,
                .workspace = ws,
                .workspace_buf = buf,
                .out = out_b,
                .m = m,
                .num_experts = c.moe512_num_experts,
                .top_k = c.moe512_top_k,
                .hidden_size = c.moe512_hidden,
                .intermediate_size = c.moe512_inter,
                .shared_expert = sh,
                .shared_weights_base = wb,
                .shared_expert_dtype = .q4_affine,
                .shared_intermediate_size = c.moe512_shared_inter,
                .shared_gate_weight = sg_b,
                .shared_gate_dtype = .f16,
            };
            // Warmup
            {
                var wf = try comp.beginFrame(c.timeout_ms);
                errdefer wf.abort();
                try pipes.encodeMoELayer(&wf, .q4, args, false);
                try comp.wait(try wf.commit(), c.timeout_ms);
            }
            const unique = ws.schedule_header.num_unique_experts;
            const t0 = try vmstat.nowNs();
            for (0..c.moe512_reps) |_| {
                var frame = try comp.beginFrame(c.timeout_ms);
                errdefer frame.abort();
                try pipes.encodeMoELayer(&frame, .q4, args, false);
                try comp.wait(try frame.commit(), c.timeout_ms);
            }
            const t1 = try vmstat.nowNs();
            const us = @as(f64, @floatFromInt(t1 - t0)) / (@as(f64, @floatFromInt(c.moe512_reps)) * 1000.0);
            const moved: u64 = @as(u64, unique) * exp_b + sh_b + @as(u64, m) * c.moe512_hidden * 4;
            return .{
                .us = us,
                .gbs = @as(f64, @floatFromInt(moved)) / (us * 1000.0),
                .unique = unique,
            };
        }
    }.call;

    const m512_1 = try timeMoe512(&compute, &mp, &moe_ws, moe_buf, moe_x_b, moe_out_b, moe_lg_b, weights_base, moe_et_b, shared_ew, moe_sg_b, 1, cfg, single_exp_bytes, sh_total_bytes);
    const m512_16 = try timeMoe512(&compute, &mp, &moe_ws, moe_buf, moe_x_b, moe_out_b, moe_lg_b, weights_base, moe_et_b, shared_ew, moe_sg_b, 16, cfg, single_exp_bytes, sh_total_bytes);
    const naive_16: f64 = @floatFromInt(@as(u32, 16) * cfg.moe512_top_k);
    const dedup_ratio: f64 = if (m512_16.unique > 0) naive_16 / @as(f64, @floatFromInt(m512_16.unique)) else 1.0;

    return .{
        .gdn_prefill_seq = cfg.gdn_prefill_seq,
        .gdn_prefill_us = gdn_prefill_us,
        .gdn_prefill_tok_per_s = gdn_prefill_tok_per_s,
        .gdn_prefill_gbs = gdn_prefill_gbs,
        .gdn_decode_us = gdn_decode_us,
        .gdn_decode_gbs = gdn_decode_gbs,
        .gdn_stmp_us = gdn_stmp_us,
        .gdn_commit_us = gdn_commit_us,
        .gdn_stmp_isolated = gdn_stmp_isolated,
        .qsa_mrope_us = qsa_mrope_us,
        .qsa_full512_r0_us = qsa_full512_r0_us,
        .qsa_full512_reuse_us = qsa_full512_reuse_us,
        .qsa_swa128_r0_us = qsa_swa128_r0_us,
        .qsa_swa128_reuse_us = qsa_swa128_reuse_us,
        .gr_hidden = cfg.gr_hidden,
        .gr_m1_us = gr_m1.us,
        .gr_m1_gbs = gr_m1.gbs,
        .gr_m16_us = gr_m16.us,
        .gr_m16_gbs = gr_m16.gbs,
        .moe512_num_experts = e_cnt,
        .moe512_top_k = cfg.moe512_top_k,
        .moe512_wired_mib = moe_arena.bytes().len >> 20,
        .moe512_m1_us = m512_1.us,
        .moe512_m1_gbs = m512_1.gbs,
        .moe512_m1_unique = m512_1.unique,
        .moe512_m16_us = m512_16.us,
        .moe512_m16_gbs = m512_16.gbs,
        .moe512_m16_unique = m512_16.unique,
        .moe512_dedup_ratio = dedup_ratio,
    };
}

fn fillDeterministicF16Slice(dst: []u8, seed: u64, amp: f32) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const count = dst.len / 2;
    for (0..count) |i| {
        const v: f32 = (r.float(f32) * 2.0 - 1.0) * amp;
        const bits: u16 = @bitCast(@as(f16, @floatCast(v)));
        std.mem.writeInt(u16, dst[i * 2 ..][0..2], bits, .little);
    }
}

fn fillConstantF16Slice(dst: []u8, val: f32) void {
    const bits: u16 = @bitCast(@as(f16, @floatCast(val)));
    const count = dst.len / 2;
    for (0..count) |i| {
        std.mem.writeInt(u16, dst[i * 2 ..][0..2], bits, .little);
    }
}

fn runM6PleBench(allocator: std.mem.Allocator, cfg: M6Config) !M6PleReport {
    const tmp_dir = try makeM6TempDir(allocator, "ple");
    defer {
        removeDirRecursiveBestEffort(allocator, tmp_dir);
        allocator.free(tmp_dir);
    }

    // Build 4-shard PLE `.amf` files via graph.buildSynthQwen4Bundle (using minimal experts for speed).
    const total_rows: u64 = @as(u64, cfg.ple_num_shards) * cfg.ple_rows_per_shard;
    var bundle = try graph.buildSynthQwen4Bundle(allocator, .{
        .hidden_size = cfg.ple_embed_dim,
        .vocab_size = 128,
        .num_attention_heads = 2,
        .num_key_value_heads = 1,
        .head_dim = 64,
        .gdn_kh = 2,
        .gdn_vh = 2,
        .gdn_kd = 32,
        .gdn_vd = 32,
        .moe_inter = 64,
        .shared_inter = 64,
        .num_experts = 4,
        .top_k = 2,
        .ple_num_shards = cfg.ple_num_shards,
        .ple_rows = total_rows,
        .tmp_dir = tmp_dir,
    });
    defer bundle.deinit(allocator);

    var store = try ple.PleStore.open(allocator, bundle.ple_paths, .{
        .cache = .{ .cache_pages = cfg.ple_cache_pages },
    });
    defer store.deinit(allocator);

    var scratch = try ple.LoopPleScratch.init(allocator, 4, 1, cfg.ple_embed_dim, null);
    defer scratch.deinit(allocator);

    const acc_buf = try allocator.alloc(f32, cfg.ple_embed_dim);
    defer allocator.free(acc_buf);

    // 1. Cold F_NOCACHE misses (clearCache before each distinct query to guarantee true SSD pread).
    const cold_samples = try allocator.alloc(f64, cfg.ple_cold_queries);
    defer allocator.free(cold_samples);
    for (0..cfg.ple_cold_queries) |q| {
        try store.clearCache();
        @memset(acc_buf, 0.0);
        const toks = [_]u32{ @intCast(q * 31 + 3), @intCast(q * 17 + 11) };
        const loc = try store.mapNgram(0, &toks, 2);
        const rows = [_]u64{loc.global_row};
        const t0 = try vmstat.nowNs();
        try store.lookupAndAccumulate(0, &rows, 1.0, acc_buf);
        const t1 = try vmstat.nowNs();
        cold_samples[q] = @as(f64, @floatFromInt(t1 - t0)) / 1000.0;
    }

    // 2. Wired LRU cache hits (same row repeated).
    const hit_samples = try allocator.alloc(f64, cfg.ple_hit_queries);
    defer allocator.free(hit_samples);
    const hot_toks = [_]u32{ 7, 19 };
    const hot_loc = try store.mapNgram(0, &hot_toks, 2);
    const hot_rows = [_]u64{hot_loc.global_row};
    @memset(acc_buf, 0.0);
    try store.lookupAndAccumulate(0, &hot_rows, 1.0, acc_buf); // warm into LRU
    for (0..cfg.ple_hit_queries) |q| {
        @memset(acc_buf, 0.0);
        const t0 = try vmstat.nowNs();
        try store.lookupAndAccumulate(0, &hot_rows, 1.0, acc_buf);
        const t1 = try vmstat.nowNs();
        hit_samples[q] = @as(f64, @floatFromInt(t1 - t0)) / 1000.0;
    }

    // 3. ADR-0019 Loop Isolation #2: `lookupForLoopStep` across R=3 (`r = 0, 1, 2`).
    try store.clearCache();
    store.resetStats();
    var r0_nvme: u64 = 0;
    var r_ge1_nvme: u64 = 0;
    var r_ge1_reuses: u64 = 0;
    for (0..cfg.ple_loop_steps) |step| {
        scratch.resetStep();
        const toks = [_]u32{ @intCast(step * 23 + 5), @intCast(step * 41 + 9) };
        const loc = try store.mapNgram(0, &toks, 2);
        const rows = [_]u64{loc.global_row};
        const before_r0 = store.stats().nvme_reads;
        @memset(acc_buf, 0.0);
        try store.lookupForLoopStep(0, &rows, 1.0, 0, &scratch, acc_buf);
        r0_nvme += store.stats().nvme_reads - before_r0;
        for ([_]u32{ 1, 2 }) |r| {
            const before_r = store.stats().nvme_reads;
            @memset(acc_buf, 0.0);
            try store.lookupForLoopStep(0, &rows, 1.0, r, &scratch, acc_buf);
            r_ge1_nvme += store.stats().nvme_reads - before_r;
            r_ge1_reuses += 1;
        }
    }

    return .{
        .num_shards = cfg.ple_num_shards,
        .total_rows = total_rows,
        .embed_dim = cfg.ple_embed_dim,
        .wired_kib = (store.totalWiredBytes() + scratch.totalWiredBytes()) >> 10,
        .cold_p50_us = percentileF64(cold_samples, 0.50),
        .cold_p99_us = percentileF64(cold_samples, 0.99),
        .hit_p50_us = percentileF64(hit_samples, 0.50),
        .r0_nvme_reads = r0_nvme,
        .r_ge1_nvme_reads = r_ge1_nvme,
        .r_ge1_scratch_reuses = r_ge1_reuses,
    };
}

fn runM6ScaleQwen4Bench(allocator: std.mem.Allocator, cfg: M6Config) !M6ScaleQwen4Report {
    const tmp_dir = try makeM6TempDir(allocator, "qwen4_512e");
    defer {
        removeDirRecursiveBestEffort(allocator, tmp_dir);
        allocator.free(tmp_dir);
    }

    var bundle = try graph.buildSynthQwen4Bundle(allocator, .{
        .hidden_size = cfg.qwen4_hidden,
        .vocab_size = cfg.qwen4_vocab,
        .num_attention_heads = 2,
        .num_key_value_heads = 1,
        .head_dim = 64,
        .gdn_kh = 2,
        .gdn_vh = 2,
        .gdn_kd = 32,
        .gdn_vd = 32,
        .moe_inter = cfg.qwen4_moe_inter,
        .shared_inter = cfg.qwen4_shared_inter,
        .num_experts = cfg.qwen4_num_experts,
        .top_k = cfg.qwen4_top_k,
        .cold_experts_per_layer = cfg.qwen4_cold_per_layer,
        .max_loops = 1,
        .ple_num_shards = cfg.qwen4_ple_shards,
        .ple_rows = cfg.qwen4_ple_rows,
        .use_q4_experts = true,
        .tmp_dir = tmp_dir,
    });
    defer bundle.deinit(allocator);

    const mtp_cfg: draft_mod.DraftConfig = .{
        .format = .mtp,
        .vocab_size = cfg.qwen4_vocab,
        .draft_vocab_size = cfg.qwen4_vocab,
        .hidden_size = cfg.qwen4_hidden,
        .target_hidden_size = cfg.qwen4_hidden,
        .num_hidden_layers = 1,
        .num_attention_heads = 2,
        .num_key_value_heads = 1,
        .head_dim = 64,
        .intermediate_size = cfg.qwen4_hidden * 2,
        .rope_theta = 10_000.0,
        .share_embed_tokens = true,
        .share_lm_head = true,
    };
    const mtp_img = try draft_mod.buildSyntheticDraftAmfImage(allocator, mtp_cfg, 91827);
    defer allocator.free(mtp_img);

    const prompt = [_]u32{ 5, 19, 43, 87 };
    const gen_n: usize = cfg.qwen4_gen_tokens;
    const expected = try allocator.alloc(u32, gen_n);
    defer allocator.free(expected);
    const got = try allocator.alloc(u32, gen_n);
    defer allocator.free(got);

    // 1. Serial baseline on GraphModel (with 4-shard PLE + 32 cold experts/layer)
    var ref_model = try graph.GraphModel.loadFromFile(allocator, bundle.amf_file_path.?, bundle.cfg, .{
        .max_seq_len = 256,
        .max_tree_nodes = 16,
        .ple_paths = bundle.ple_paths,
        .cold_slot_count = 32,
        .timeout_ms = cfg.timeout_ms,
    });
    defer ref_model.deinit(allocator);

    expected[0] = try ref_model.prefillChunk(&prompt, 0, true);
    var pos: u32 = prompt.len;
    const ser_t0 = try vmstat.nowNs();
    for (1..gen_n) |i| {
        expected[i] = try ref_model.stepDecode(expected[i - 1], pos);
        pos += 1;
    }
    const ser_t1 = try vmstat.nowNs();
    const serial_tok_per_s: f64 = if (gen_n > 1 and ser_t1 > ser_t0)
        @as(f64, @floatFromInt(gen_n - 1)) * 1_000_000_000.0 / @as(f64, @floatFromInt(ser_t1 - ser_t0))
    else
        0.0;

    // 2. SpecEngine with MTP DraftHead (Placement.auto -> cpu_sme2 / gpu calibrated)
    var spec_model = try graph.GraphModel.loadFromFile(allocator, bundle.amf_file_path.?, bundle.cfg, .{
        .max_seq_len = 256,
        .max_tree_nodes = 16,
        .enable_feature_taps = true,
        .ple_paths = bundle.ple_paths,
        .cold_slot_count = 32,
        .timeout_ms = cfg.timeout_ms,
    });
    defer spec_model.deinit(allocator);

    var dh = try draft_mod.DraftHead.loadFromBytesForGraph(allocator, &spec_model, mtp_img, mtp_cfg, .{
        .placement = .auto,
        .max_seq_len = 256,
        .max_tree_slots = 16,
    });
    defer dh.deinitForGraph(allocator, &spec_model);

    var eng = try engine_mod.SpecEngine.initWithGraph(allocator, &spec_model, &dh, .{
        .mode = .mtp,
        .placement = .auto,
        .max_tree_nodes = 8,
        .max_draft_depth = 3,
        .branch_width = 2,
        .top_k_per_node = 2,
        .enable_pruning = false,
        .temperature = 0.0,
    });
    defer eng.deinit(allocator);

    got[0] = try eng.prefill(&prompt, null);
    var p: usize = 1;
    var rbuf: [engine_mod.MAX_TREE_NODES]u32 = undefined;
    const spec_t0 = try vmstat.nowNs();
    while (p < gen_n) {
        const cap = @min(gen_n - p, rbuf.len);
        const rs = try eng.stepRound(null, rbuf[0..cap]);
        @memcpy(got[p .. p + rs.emitted_count], rbuf[0..rs.emitted_count]);
        p += rs.emitted_count;
    }
    const spec_t1 = try vmstat.nowNs();
    const mtp_tok_per_s: f64 = if (gen_n > 1 and spec_t1 > spec_t0)
        @as(f64, @floatFromInt(gen_n - 1)) * 1_000_000_000.0 / @as(f64, @floatFromInt(spec_t1 - spec_t0))
    else
        0.0;

    const cs = spec_model.coldExpertStats();
    const ps = spec_model.pleStats();
    const iso = spec_model.loopIsolationStats();

    return .{
        .num_experts = cfg.qwen4_num_experts,
        .top_k = cfg.qwen4_top_k,
        .cold_per_layer = cfg.qwen4_cold_per_layer,
        .ple_shards = cfg.qwen4_ple_shards,
        .wired_mib = eng.totalWiredBytes() >> 20,
        .gen_tokens = cfg.qwen4_gen_tokens,
        .serial_tok_per_s = serial_tok_per_s,
        .mtp_tok_per_s = mtp_tok_per_s,
        .mtp_tau = eng.stats.meanAcceptanceLength(),
        .mtp_placement = dh.effective_placement,
        .bit_exact_mtp = std.mem.eql(u32, expected, got),
        .cold_hits = if (cs) |c| c.cache_telemetry.cache_hits + c.cache_telemetry.prefetch_hits else 0,
        .cold_misses = if (cs) |c| c.cache_telemetry.sync_misses else 0,
        .ple_nvme_reads = if (ps) |s| s.nvme_reads else 0,
        .ple_r_ge1_nvme_reads = iso.ple_r_ge1_nvme_reads,
    };
}

fn runM6SelfSpecBench(allocator: std.mem.Allocator, cfg: M6Config) !M6SelfSpecReport {
    const start_l: u32 = 1;
    const end_l: u32 = cfg.loop_num_layers - 1;
    const core_layers: u32 = end_l - start_l;
    var bundle = try graph.buildSynthLoopDenseBundle(allocator, .{
        .hidden_size = cfg.loop_hidden,
        .vocab_size = cfg.loop_vocab,
        .num_layers = cfg.loop_num_layers,
        .num_attention_heads = cfg.loop_num_heads,
        .num_key_value_heads = cfg.loop_num_kv_heads,
        .head_dim = cfg.loop_head_dim,
        .intermediate_size = cfg.loop_inter,
        .start_layer = start_l,
        .end_layer = end_l,
        .mode = .native_recurrent,
        .max_loops = 4,
        .use_q4 = cfg.loop_use_q4,
    });
    defer bundle.deinit(allocator);

    var model = try graph.GraphModel.loadFromBytes(allocator, bundle.amf_bytes, bundle.cfg, .{
        .max_seq_len = 256,
        .max_tree_nodes = 16,
        .max_loops_override = 4,
        .timeout_ms = cfg.timeout_ms,
    });
    defer model.deinit(allocator);

    // Estimate Core weight bytes directly from layer shapes.
    const h64: f64 = @floatFromInt(cfg.loop_hidden);
    const i64_f: f64 = @floatFromInt(cfg.loop_inter);
    const qd64: f64 = @floatFromInt(cfg.loop_num_heads * cfg.loop_head_dim);
    const kvd64: f64 = @floatFromInt(cfg.loop_num_kv_heads * cfg.loop_head_dim);
    const bpe: f64 = if (cfg.loop_use_q4) 0.5625 else 2.0;
    const per_layer_bytes = (qd64 * h64 + 2.0 * kvd64 * h64 + h64 * qd64 + 3.0 * i64_f * h64) * bpe;
    const core_weight_mib = (per_layer_bytes * @as(f64, @floatFromInt(core_layers))) / (1024.0 * 1024.0);
    const total_wired_mib = model.totalWiredBytes() >> 20;

    const prompt = [_]u32{ 3, 17, 41, 63 };
    const spec_n: usize = cfg.loop_spec_tokens;
    const expected = try allocator.alloc(u32, spec_n);
    defer allocator.free(expected);
    const got8 = try allocator.alloc(u32, spec_n);
    defer allocator.free(got8);
    const got16 = try allocator.alloc(u32, spec_n);
    defer allocator.free(got16);

    var rows: [4]M6SelfSpecRow = undefined;
    var r1_serial_ms: f64 = 0.0;

    const tree_toks = [_]u32{ 11, 22, 33, 44, 55, 66, 77, 88, 12, 23, 34, 45, 56, 67, 78, 89 };
    var tree_pos: [16]u32 = undefined;
    var tree_masks: [16]u64 = undefined;
    for (0..16) |i| {
        tree_pos[i] = 16 + @as(u32, @intCast(i));
        tree_masks[i] = (@as(u64, 1) << @intCast(i + 1)) - 1;
    }

    for ([_]u32{ 1, 2, 3, 4 }, 0..) |R, ri| {
        model.effective_max_loops = R;
        model.resetKvCache();
        _ = try model.prefillChunk(&prompt, 0, true);
        // Warmup
        var cur: u32 = 17;
        for (0..2) |w| cur = try model.stepDecode(cur, @intCast(prompt.len + w));

        // 1. Serial M=1 decode latency
        const st0 = try vmstat.nowNs();
        for (0..cfg.loop_serial_steps) |s| {
            cur = try model.stepDecode(cur, @intCast(prompt.len + 2 + s));
        }
        const st1 = try vmstat.nowNs();
        const serial_ms = @as(f64, @floatFromInt(st1 - st0)) / (@as(f64, @floatFromInt(cfg.loop_serial_steps)) * 1_000_000.0);
        const serial_tps = if (serial_ms > 0.0) 1000.0 / serial_ms else 0.0;
        if (R == 1) r1_serial_ms = serial_ms;

        // 2. Batched verifyTree latency across M in {4, 8, 12, 16} (fixed base_pos = 16)
        model.resetKvCache();
        var pre16: [16]u32 = undefined;
        for (0..16) |i| pre16[i] = @intCast((i * 7 + 1) % cfg.loop_vocab);
        _ = try model.prefillChunk(&pre16, 0, false);

        var v_ms: [4]f64 = .{ 0.0, 0.0, 0.0, 0.0 };
        const ms_list = [_]u32{ 4, 8, 12, 16 };
        var breakeven_m: u32 = 0;
        for (ms_list, 0..) |m_val, mi| {
            _ = try model.verifyTree(tree_toks[0..m_val], tree_pos[0..m_val], tree_masks[0..m_val], 16);
            const vt0 = try vmstat.nowNs();
            for (0..cfg.loop_batch_reps) |_| {
                _ = try model.verifyTree(tree_toks[0..m_val], tree_pos[0..m_val], tree_masks[0..m_val], 16);
            }
            const vt1 = try vmstat.nowNs();
            v_ms[mi] = @as(f64, @floatFromInt(vt1 - vt0)) / (@as(f64, @floatFromInt(cfg.loop_batch_reps)) * 1_000_000.0);
            if (breakeven_m == 0 and r1_serial_ms > 0.0 and (v_ms[mi] / @as(f64, @floatFromInt(m_val))) <= r1_serial_ms) {
                breakeven_m = m_val;
            }
        }
        if (R == 1) breakeven_m = 1;

        // 3. End-to-end lossless SelfSpecRecurrent (T=0) at M=8 and M=16 vs serial stepDecode
        model.resetKvCache();
        expected[0] = try model.prefillChunk(&prompt, 0, true);
        for (1..spec_n) |i| {
            expected[i] = try model.stepDecode(expected[i - 1], @intCast(prompt.len + i - 1));
        }

        const runSpecM = struct {
            fn call(
                alloc: std.mem.Allocator,
                gm: *graph.GraphModel,
                pr: []const u32,
                out_toks: []u32,
                m_cap: u16,
            ) !struct { tps: f64, tau: f64 } {
                gm.resetKvCache();
                var eng = try engine_mod.SpecEngine.initWithGraph(alloc, gm, null, .{
                    .mode = .self_recurrent,
                    .max_tree_nodes = m_cap,
                    .max_draft_depth = if (m_cap >= 16) 5 else 4,
                    .branch_width = 2,
                    .top_k_per_node = 2,
                    .enable_pruning = false,
                    .temperature = 0.0,
                });
                defer eng.deinit(alloc);
                out_toks[0] = try eng.prefill(pr, null);
                var p: usize = 1;
                var rb: [engine_mod.MAX_TREE_NODES]u32 = undefined;
                const t0 = try vmstat.nowNs();
                while (p < out_toks.len) {
                    const cap = @min(out_toks.len - p, rb.len);
                    const rs = try eng.stepRound(null, rb[0..cap]);
                    @memcpy(out_toks[p .. p + rs.emitted_count], rb[0..rs.emitted_count]);
                    p += rs.emitted_count;
                }
                const t1 = try vmstat.nowNs();
                const tps: f64 = if (out_toks.len > 1 and t1 > t0)
                    @as(f64, @floatFromInt(out_toks.len - 1)) * 1_000_000_000.0 / @as(f64, @floatFromInt(t1 - t0))
                else
                    0.0;
                return .{ .tps = tps, .tau = eng.stats.meanAcceptanceLength() };
            }
        }.call;

        const sp8 = try runSpecM(allocator, &model, &prompt, got8, 8);
        const sp16 = try runSpecM(allocator, &model, &prompt, got16, 16);
        const exact = std.mem.eql(u32, expected, got8) and std.mem.eql(u32, expected, got16);

        rows[ri] = .{
            .loops = R,
            .serial_ms_per_tok = serial_ms,
            .serial_tok_per_s = serial_tps,
            .ratio_vs_r1 = if (r1_serial_ms > 0.0) r1_serial_ms / serial_ms else 1.0,
            .verify_m4_ms = v_ms[0],
            .verify_m8_ms = v_ms[1],
            .verify_m12_ms = v_ms[2],
            .verify_m16_ms = v_ms[3],
            .breakeven_m = breakeven_m,
            .spec_m8_tok_per_s = sp8.tps,
            .spec_m8_tau = sp8.tau,
            .spec_m16_tok_per_s = sp16.tps,
            .spec_m16_tau = sp16.tau,
            .bit_exact = exact,
        };
    }

    return .{
        .core_layers = core_layers,
        .core_weight_mib = core_weight_mib,
        .total_wired_mib = total_wired_mib,
        .rows = rows,
    };
}

fn runM6D13SmeDraftBench(allocator: std.mem.Allocator, cfg: M6Config) ![12]M6SmeDraftRow {
    var device = try metal.Device.open();
    defer device.close();
    const facts = try sysinfo.Facts.read();
    const dev_info = try device.info();
    const wired_budget = budget_mod.Budget.fromHost(facts, dev_info, .{});
    var library = try metal.Library.openEmbedded(device);
    defer library.close();
    var compute = try metal.Compute.init(device);
    defer compute.close();
    var dp = try dequant.DequantPipelines.init(device, library);
    defer dp.close();
    var dispatcher = try sme.SmeDispatcher.init(allocator, .{});
    defer dispatcher.deinit();

    const ShapeSpec = struct {
        label: []const u8,
        is_q4: bool,
        n: u32,
        k: u32,
    };
    const sd = @max(@as(u32, 1), cfg.d13_scale_down);
    const shapes = [4]ShapeSpec{
        .{ .label = "eagle3_fc_q4_2048x6144", .is_q4 = true, .n = @max(@as(u32, 64), 2048 / sd), .k = @max(@as(u32, 64), 6144 / sd) },
        .{ .label = "eagle3_fc_f16_1024x3072", .is_q4 = false, .n = @max(@as(u32, 64), 1024 / sd), .k = @max(@as(u32, 64), 3072 / sd) },
        .{ .label = "mtp_eh_proj_q4_2048x4096", .is_q4 = true, .n = @max(@as(u32, 64), 2048 / sd), .k = @max(@as(u32, 64), 4096 / sd) },
        .{ .label = "mtp_eh_proj_f16_1024x2048", .is_q4 = false, .n = @max(@as(u32, 64), 1024 / sd), .k = @max(@as(u32, 64), 2048 / sd) },
    };
    const ms = [3]u32{ 1, 4, 8 };

    var rows: [12]M6SmeDraftRow = undefined;
    var row_idx: usize = 0;

    for (shapes) |sh| {
        const n: usize = sh.n;
        const k: usize = sh.k;
        const pdt: sme.PanelDType = if (sh.is_q4) .q4_affine else .f16;
        const sme_packed_bytes = try sme.packedBlobBytes(pdt, n, k);
        const sme_w = try allocator.alignedAlloc(u8, .@"64", sme_packed_bytes);
        defer allocator.free(sme_w);

        const gpu_w_bytes: usize = if (sh.is_q4) n * (k / 2) else n * k * 2;
        const gpu_sb_bytes: usize = if (sh.is_q4) n * (k / 64) * 4 else 64;
        const arena_bytes = try page.alignUp(
            amf.align64(gpu_w_bytes) +
                amf.align64(gpu_sb_bytes) +
                amf.align64(8 * k * 2) +
                amf.align64(8 * n * 2) +
                65536,
        );
        var arena = try arena_mod.WiredArena.init(arena_bytes, wired_budget);
        defer arena.deinit();
        var buf = try metal.Buffer.wrap(device, arena.bytes());
        defer buf.close();
        try compute.makeResident(&.{buf});
        defer compute.removeResident(&.{buf}) catch |err| std.debug.panic("removeResident failed: {}", .{err});

        const gw = try arena.allocAligned(gpu_w_bytes, 64);
        const gsb = try arena.allocAligned(gpu_sb_bytes, 64);
        const gx = try arena.allocAligned(8 * k * 2, 64);
        const gy = try arena.allocAligned(8 * n * 2, 64);
        fillDeterministicF16Slice(gx, 6001 + row_idx, 0.2);

        if (sh.is_q4) {
            @memset(gw, 0x45);
            const sb_aligned: []align(@alignOf(quant.ScaleBias)) u8 = @alignCast(gsb[0 .. n * (k / 64) * 4]);
            const sb_slice = std.mem.bytesAsSlice(quant.ScaleBias, sb_aligned);
            for (sb_slice) |*sb| sb.* = .{ .scale = 0.01, .bias = -0.07 };
            try sme.packMlxQuantPanel64(.q4, gw, sb_slice, n, k, sme_w);
        } else {
            fillDeterministicF16Slice(gw, 6100 + row_idx, 0.1);
            const gw_aligned: []align(2) u8 = @alignCast(gw);
            const row_major = std.mem.bytesAsSlice(u16, gw_aligned);
            try sme.packDensePanel64(.f16, row_major, n, k, sme_w);
        }

        const w_lin: sme.SmeLinearWeight = .{
            .dtype = pdt,
            .param_kind = .affine_scale_bias,
            .cache_hint = .temporal,
            .n = n,
            .k = k,
            .w_packed = sme_w,
        };
        var proj = try sme.SmeDraftProjection.init(w_lin, .{
            .max_m = 8,
        });
        defer proj.deinit();

        const out_sme = try allocator.alloc(u8, 8 * n * 2);
        defer allocator.free(out_sme);

        const GpuCtx = struct {
            comp: *metal.Compute,
            pipes: *const dequant.DequantPipelines,
            is_q4: bool,
            w_b: metal.BufferBinding,
            sb_b: metal.BufferBinding,
            x_b: metal.BufferBinding,
            y_b: metal.BufferBinding,
            n: u32,
            k: u32,
            m: u32,
            timeout_ms: u64,
        };
        const gpuStep = struct {
            fn call(raw: *anyopaque) anyerror!void {
                const c: *const GpuCtx = @ptrCast(@alignCast(raw));
                var frame = try c.comp.beginFrame(c.timeout_ms);
                errdefer frame.abort();
                if (c.is_q4) {
                    if (c.m == 1) {
                        try c.pipes.encodeGemv(&frame, .q4, .{
                            .x = c.x_b,
                            .qweight = c.w_b,
                            .scale_bias = c.sb_b,
                            .y = c.y_b,
                            .n = c.n,
                            .k = c.k,
                        }, false);
                    } else {
                        try c.pipes.encodeGemm(&frame, .q4, .{
                            .x = c.x_b,
                            .qweight = c.w_b,
                            .scale_bias = c.sb_b,
                            .y = c.y_b,
                            .m = c.m,
                            .n = c.n,
                            .k = c.k,
                        }, false);
                    }
                } else {
                    if (c.m == 1) {
                        try c.pipes.encodeDenseGemv(&frame, .f16, .{
                            .x = c.x_b,
                            .weight = c.w_b,
                            .y = c.y_b,
                            .n = c.n,
                            .k = c.k,
                        }, false);
                    } else {
                        try c.pipes.encodeDenseGemm(&frame, .f16, .{
                            .x = c.x_b,
                            .weight = c.w_b,
                            .y = c.y_b,
                            .m = c.m,
                            .n = c.n,
                            .k = c.k,
                        }, false);
                    }
                }
                try c.comp.wait(try frame.commit(), c.timeout_ms);
            }
        }.call;

        for (ms) |m_val| {
            var gctx: GpuCtx = .{
                .comp = &compute,
                .pipes = &dp,
                .is_q4 = sh.is_q4,
                .w_b = .{ .buffer = buf, .offset = arena.offsetOf(gw.ptr) },
                .sb_b = .{ .buffer = buf, .offset = arena.offsetOf(gsb.ptr) },
                .x_b = .{ .buffer = buf, .offset = arena.offsetOf(gx.ptr) },
                .y_b = .{ .buffer = buf, .offset = arena.offsetOf(gy.ptr) },
                .n = sh.n,
                .k = sh.k,
                .m = m_val,
                .timeout_ms = cfg.timeout_ms,
            };
            const cal = try sme.calibrateSmeVsGpuFn(
                &dispatcher,
                &proj,
                m_val,
                out_sme[0 .. @as(usize, m_val) * n * 2],
                &gctx,
                &gpuStep,
                .{ .warmups = cfg.d13_warmups, .iters = cfg.d13_iters, .out_dtype = .f16 },
            );
            rows[row_idx] = .{
                .label = sh.label,
                .dtype_label = if (sh.is_q4) "q4" else "f16",
                .n = sh.n,
                .k = sh.k,
                .m = m_val,
                .weight_bytes = cal.weight_bytes,
                .fits_l2 = cal.fits_l2,
                .sme2_us = cal.sme_us,
                .gpu_us = cal.gpu_us,
                .prefer_sme2 = cal.prefer_sme2,
            };
            row_idx += 1;
        }
    }
    return rows;
}

pub fn runM6WithConfig(allocator: std.mem.Allocator, config: M6Config) !M6Report {
    const vm_before = try vmstat.Snapshot.capture();
    const ops = try runM6OpsBench(allocator, config);
    const ple_res = try runM6PleBench(allocator, config);
    const qwen4_res = try runM6ScaleQwen4Bench(allocator, config);
    const self_spec_res = try runM6SelfSpecBench(allocator, config);
    const d13_rows = try runM6D13SmeDraftBench(allocator, config);
    const vm_after = try vmstat.Snapshot.capture();
    try vmstat.checkNoSwap(vm_before, vm_after);

    return .{
        .ops = ops,
        .ple = ple_res,
        .scale_qwen4 = qwen4_res,
        .self_spec = self_spec_res,
        .d13_rows = d13_rows,
        .vm_before = vm_before,
        .vm_after = vm_after,
    };
}

pub fn runM6(allocator: std.mem.Allocator) !M6Report {
    return runM6WithConfig(allocator, .{});
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

test "runM4WithConfig exercises Spike S3, Spike S6, A2, A1/G6 hybrid prefill, cold NVMe decode and D4 small draft" {
    const gpa = std.testing.allocator;
    const s3_blobs = [_]usize{64 * 1024};
    const s3_qds = [_]usize{ 1, 2 };
    const s3_workers = [_]usize{2};
    const hyb_lens = [_]u32{128};

    var report = try runM4WithConfig(gpa, .{
        .s3 = .{
            .temp_dir = "/Volumes/Akari/.tmp",
            .file_bytes = 512 * 1024,
            .blob_sizes = &s3_blobs,
            .queue_depths = &s3_qds,
            .pread_workers = &s3_workers,
            .io_policies = &.{.important},
            .warmup_reads = 2,
            .timed_reads = 4,
            .require_zero_swap = true,
            .timeout_ms = 10_000,
        },
        .s6 = .{
            .hidden_size = 128,
            .num_q_heads = 4,
            .num_kv_heads = 2,
            .head_dim = 64,
            .chunk_tokens = 64,
            .buckets = .{ 64, 128, 256, 512 },
            .moe_intermediate_size = 128,
            .num_experts = 8,
            .top_k = 2,
            .moe_bits = .q3,
            .wavefront_chunks = 2,
            .wavefront_layers = 2,
            .gpu_attn_tile = 32,
            .precision_sample_rows = 4,
            .warmup_iters = 1,
            .timed_iters = 2,
            .timeout_ms = 10_000,
        },
        .a2 = .{
            .num_layers = 4,
            .num_experts = 16,
            .hidden_size = 128,
            .top_k = 4,
            .lookahead_layers = 2,
            .batch_tokens = 1,
            .warmup_iters = 1,
            .timed_iters = 4,
        },
        .hybrid_prompt_lens = &hyb_lens,
        .hybrid_chunk_tokens = 64,
        .hybrid_hidden_size = 128,
        .hybrid_num_layers = 2,
        .hybrid_num_q_heads = 4,
        .hybrid_num_kv_heads = 2,
        .hybrid_head_dim = 64,
        .hybrid_moe_intermediate = 128,
        .hybrid_num_experts = 8,
        .hybrid_top_k = 2,
        .hybrid_timed_iters = 1,
        .cold_temp_dir = "/Volumes/Akari/.tmp",
        .cold_gen_tokens = 4,
        .d4_gen_tokens = 4,
    });
    defer report.deinit(gpa);

    try std.testing.expectEqual(@as(u64, 0), report.s3.swapouts_delta);
    try std.testing.expect(report.cold_synth.bit_exact);
    try std.testing.expect(report.cold_synth.cold_wired_bytes < report.cold_synth.hot_wired_bytes);
    try std.testing.expect(report.d4_synth.lossless_match);
    try std.testing.expectEqual(@as(usize, 1), report.hybrid_prefill.results.len);
    try std.testing.expect(report.hybrid_prefill.results[0].all_outputs_zero_copy);
    try vmstat.checkNoSwap(report.vm_before, report.vm_after);

    var aw: std.Io.Writer.Allocating = .init(gpa);
    defer aw.deinit();
    try report.write(&aw.writer);
    const rendered = aw.written();
    try std.testing.expect(std.mem.indexOf(u8, rendered, "Spike S3 NVMe Direct-I/O Sweep") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "Spike S6 Hybrid ANE/GPU Wavefront Prefill") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "A2 ANE Cross-Layer MoE Expert Predictor") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "cold_synth.bit_exact_vs_hot") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "d4_synth.lossless_match") != null);
}

test "runM45WithConfig exercises C13 concurrent GPU+SME2 bus gate, ldnt1b L2 preservation and router calibration" {
    const gpa = std.testing.allocator;
    const report = try runM45WithConfig(gpa, .{
        .run_full_s7 = false,
        .gpu_n = 256,
        .gpu_k = 256,
        .gpu_min_working_bytes = 0,
        .gpu_warmup_batches = 1,
        .gpu_timed_batches = 3,
        .gpu_dispatches_per_batch = 4,
        .sme_l2_m = 8,
        .sme_l2_n = 128,
        .sme_l2_k = 128,
        .sme_dram_m = 1,
        .sme_dram_n = 128,
        .sme_dram_k = 128,
        .sme_dram_min_working_bytes = 0,
        .l2_pres_hot_m = 8,
        .l2_pres_hot_n = 64,
        .l2_pres_hot_k = 128,
        .l2_pres_cold_h = 128,
        .l2_pres_cold_inter = 64,
        .l2_pres_cold_experts = 2,
        .l2_pres_reps = 2,
        .router_layers = 4,
        .router_experts = 16,
        .router_top_k = 2,
        .router_hidden = 128,
        .router_moe_inter = 64,
        .router_vocab = 64,
        .router_reps = 3,
        .timeout_ms = 10_000,
    });

    for (report.bus_rows) |row| {
        try std.testing.expect(row.gpu_solo_gbs > 0.0);
        try std.testing.expect(row.gpu_conc_gbs > 0.0);
        try std.testing.expect(row.sme_solo_us > 0.0);
    }
    try std.testing.expect(report.ldnt_l2.hot_baseline_us > 0.0);
    try std.testing.expect(report.ldnt_l2.cold_swiglu_ldnt1_us > 0.0);
    for (report.router_cal) |rc| {
        try std.testing.expect(rc.gpu_us > 0.0);
        try std.testing.expect(rc.sme2_us > 0.0);
        try std.testing.expect(rc.topk_agreement >= 0.90);
    }
    try vmstat.checkNoSwap(report.vm_before, report.vm_after);

    var aw: std.Io.Writer.Allocating = .init(gpa);
    defer aw.deinit();
    try report.write(&aw.writer);
    const rendered = aw.written();
    try std.testing.expect(std.mem.indexOf(u8, rendered, "C13 Concurrent Metal 4 GPU + CPU SME2") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "ldnt_l2.hot_after_ldnt1_us") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "router_cal.bank_kib") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "m45.swap.status") != null);
}

test "runM5WithConfig exercises G13 q2/hetero-MoE, C6 F_NOCACHE ngram, D8 PEARL and C7/C8 kqueue server" {
    const gpa = std.testing.allocator;
    const smoke_q2 = [_]Case{
        .{ .kind = .gemv_q2, .m = 1, .n = 64, .k = 128 },
        .{ .kind = .gemm_q2, .m = 16, .n = 64, .k = 128 },
    };
    var report = try runM5WithConfig(gpa, .{
        .q2_cases = &smoke_q2,
        .q2_min_working_set_bytes = 0,
        .q2_warmup_batches = 1,
        .q2_gemv_timed_batches = 1,
        .q2_gemv_dispatches_per_batch = 4,
        .q2_gemm_timed_batches = 1,
        .q2_gemm_dispatches_per_batch = 2,
        .hetero_moe_hidden = 128,
        .hetero_moe_inter = 128,
        .hetero_moe_experts = 16,
        .hetero_moe_top_k = 4,
        .hetero_moe_shared_m8 = 8,
        .hetero_moe_warmup_batches = 1,
        .hetero_moe_timed_batches = 2,
        .hetero_moe_dispatches_per_batch = 2,
        .ngram_temp_dir = "/Volumes/Akari/.tmp",
        .ngram_buckets = 16,
        .ngram_entries_per_bucket = 2,
        .ngram_cache_pages = 4,
        .ngram_cold_queries = 8,
        .ngram_cache_queries = 16,
        .pearl_layers = 2,
        .pearl_hidden = 128,
        .pearl_inter = 256,
        .pearl_vocab = 64,
        .pearl_gen_tokens = 10,
        .server_health_iters = 8,
        .server_steady_iters = 4,
        .server_concurrent_clients = 4,
        .server_tokens_per_req = 8,
        .timeout_ms = 10_000,
    });
    defer report.deinit(gpa);

    try std.testing.expectEqual(@as(usize, 2), report.q2_cases.len);
    for (report.q2_cases) |c| try std.testing.expect(c.effectiveBandwidthGbs() > 0.0);
    // Heterogeneous {q2,q3,q4} MoE uses strictly fewer bytes than uniform q4 at both M=1 and M=8.
    try std.testing.expect(report.hetero_moe_rows[3].vs_q4_bytes_pct < -20.0);
    try std.testing.expectEqual(@as(u32, 3), report.hetero_moe_rows[3].active_buckets);
    try std.testing.expect(report.hetero_moe_rows[7].vs_q4_bytes_pct < -20.0);
    try std.testing.expect(report.ngram.seed_hits > 0);
    try std.testing.expect(report.ngram.nvme_reads > 0);
    try std.testing.expect(report.pearl.has_own_queue);
    try std.testing.expect(report.pearl.bit_exact_off);
    try std.testing.expect(report.pearl.bit_exact_on);
    try std.testing.expect(report.pearl.pearl_rounds > 0);
    try std.testing.expect(report.server.all_ok);
    try std.testing.expectEqual(@as(i64, 0), report.server.steady_malloc_blocks_delta);
    try vmstat.checkNoSwap(report.vm_before, report.vm_after);

    var aw: std.Io.Writer.Allocating = .init(gpa);
    defer aw.deinit();
    try report.write(&aw.writer);
    const rendered = aw.written();
    try std.testing.expect(std.mem.indexOf(u8, rendered, "gemv_q2") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "hetero_q2_q3_q4") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "ngram.cold_nvme_p50_us") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "pearl_synth.bit_exact_on") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "server.steady_malloc_blocks") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "m5.swap.status") != null);
}

test "runM6WithConfig exercises G8/G9/G10/G7, C10 PLE, scale-qwen4-512e+MTP, D12 SelfSpecRecurrent and D13 cpu_sme2" {
    const gpa = std.testing.allocator;
    const report = try runM6WithConfig(gpa, .{
        .gdn_k_heads = 2,
        .gdn_v_heads = 4,
        .gdn_head_k = 64,
        .gdn_head_v = 64,
        .gdn_prefill_seq = 32,
        .gdn_reps = 4,
        .qsa_q_heads = 4,
        .qsa_kv_heads = 2,
        .qsa_head_dim = 64,
        .qsa_block_size = 16,
        .qsa_top_k_blocks = 2,
        .qsa_full_kv_len = 64,
        .qsa_swa_kv_len = 64,
        .qsa_reps = 4,
        .gr_hidden = 128,
        .gr_reps = 4,
        .moe512_num_experts = 512,
        .moe512_top_k = 10,
        .moe512_hidden = 128,
        .moe512_inter = 64,
        .moe512_shared_inter = 64,
        .moe512_reps = 2,
        .ple_num_shards = 4,
        .ple_rows_per_shard = 64,
        .ple_embed_dim = 64,
        .ple_cache_pages = 4,
        .ple_cold_queries = 6,
        .ple_hit_queries = 8,
        .ple_loop_steps = 4,
        .qwen4_hidden = 64,
        .qwen4_vocab = 128,
        .qwen4_moe_inter = 64,
        .qwen4_shared_inter = 64,
        .qwen4_num_experts = 512,
        .qwen4_top_k = 10,
        .qwen4_cold_per_layer = 16,
        .qwen4_ple_shards = 4,
        .qwen4_ple_rows = 64,
        .qwen4_gen_tokens = 6,
        .loop_hidden = 64,
        .loop_inter = 128,
        .loop_vocab = 128,
        .loop_num_layers = 4,
        .loop_num_heads = 2,
        .loop_num_kv_heads = 1,
        .loop_head_dim = 64,
        .loop_use_q4 = true,
        .loop_serial_steps = 3,
        .loop_batch_reps = 2,
        .loop_spec_tokens = 6,
        .d13_scale_down = 8,
        .d13_iters = 3,
        .d13_warmups = 1,
        .timeout_ms = 10_000,
    });

    try std.testing.expect(report.ops.gdn_stmp_isolated);
    try std.testing.expect(report.ops.gdn_decode_us > 0.0);
    try std.testing.expect(report.ops.qsa_full512_r0_us > 0.0);
    try std.testing.expect(report.ops.qsa_full512_reuse_us > 0.0);
    try std.testing.expect(report.ops.moe512_m1_unique == 10);
    try std.testing.expect(report.ple.r0_nvme_reads > 0);
    try std.testing.expectEqual(@as(u64, 0), report.ple.r_ge1_nvme_reads);
    try std.testing.expect(report.scale_qwen4.bit_exact_mtp);
    try std.testing.expectEqual(@as(u64, 0), report.scale_qwen4.ple_r_ge1_nvme_reads);
    for (report.self_spec.rows) |row| {
        try std.testing.expect(row.bit_exact);
        try std.testing.expect(row.serial_tok_per_s > 0.0);
    }
    for (report.d13_rows) |row| {
        try std.testing.expect(row.sme2_us > 0.0);
        try std.testing.expect(row.gpu_us > 0.0);
    }
    try vmstat.checkNoSwap(report.vm_before, report.vm_after);

    var aw: std.Io.Writer.Allocating = .init(gpa);
    defer aw.deinit();
    try report.write(&aw.writer);
    const rendered = aw.written();
    try std.testing.expect(std.mem.indexOf(u8, rendered, "g8.gdn.stmp_isolated") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "c10.ple.loop_r_ge1_nvme_reads") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "qwen4_512e.mtp_bit_exact_t0") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "D12 SelfSpecRecurrent Sweep") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "D13 Placement.cpu_sme2 vs .gpu") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "m6.swap.status") != null);
}
