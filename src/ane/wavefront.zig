//! Hybrid ANE / GPU wavefront prefill engine (`A1`), static-shape KV-bucket
//! strategy (`A3`), and Spike `S6` benchmark harness (`ADR-0004`, `ADR-0010`).
//!
//! Architecture summary:
//! - Prompts longer than one chunk (`512` tokens) are split into static 512-token
//!   chunks processed with two chunks in flight (`slot = chunk & 1`): while the
//!   `USER_INITIATED` ANE driver thread runs attention block `A(l, c_next)`, the
//!   main GPU thread runs FFN/MoE `F(l, c_cur)` (or `A(l + 1, c0)` overlaps
//!   `F(l, c1)`), hiding attention latency behind GPU FFN/MoE execution.
//! - Zero-copy IOSurface `KvTile`s (`OneComponent16Half`, 64 B row stride, 16 KiB
//!   aligned) bridge ANE outputs (`attn_out`, `k_tile_out`, `v_tile_out`) directly
//!   to Metal 4 GPU textures and aliasing `MTLBuffer`s without `memcpy`.
//! - Hot-path `WavefrontPrefill.runWavefront` performs zero dynamic heap
//!   allocation, synchronizing through preallocated lock-free `SpscRing`s and
//!   Darwin's native `os_sync_wait_on_address` futex doorbell.

const std = @import("std");
const metal = @import("../bridge/metal.zig");
const page = @import("../mem/page.zig");
const arena_mod = @import("../mem/arena.zig");
const budget_mod = @import("../mem/budget.zig");
const amf = @import("../format/amf.zig");
const quant = @import("../quant/spec.zig");
const dequant = @import("../gpu/dequant.zig");
const attention = @import("../gpu/attention.zig");
const moe = @import("../gpu/moe.zig");
const ring = @import("../runtime/ring.zig");
const qos = @import("../platform/qos.zig");
const sysinfo = @import("../platform/sysinfo.zig");
const vmstat = @import("../platform/vmstat.zig");

/// Default static prefill chunk size (ADR-0004, ADR-0010).
pub const DEFAULT_CHUNK_TOKENS: u32 = 512;

/// Default power-of-two static KV bucket lengths evaluated in Spike S6 (A3).
pub const DEFAULT_KV_BUCKETS = [4]u32{ 512, 1024, 2048, 4096 };

/// Safe fp16 causal/padding mask sentinel (well within fp16 finite range `[-65504, 65504]`).
pub const MASK_NEG_INF_F16: f16 = @floatCast(-10000.0);

/// Maximum number of static KV buckets held by one `AneBucketedAttention`.
pub const MAX_STATIC_BUCKETS: usize = 8;

/// Darwin `<os/os_sync_wait_on_address.h>` (macOS 14.4+, guaranteed on macOS 26.0+).
const OS_SYNC_WAIT_ON_ADDRESS_NONE: u32 = 0;
const OS_SYNC_WAKE_BY_ADDRESS_NONE: u32 = 0;

extern "c" fn os_sync_wait_on_address(addr: *anyopaque, value: u64, size: usize, flags: u32) c_int;
extern "c" fn os_sync_wake_by_address_any(addr: *anyopaque, size: usize, flags: u32) c_int;

inline fn futexWaitU32(atom: *std.atomic.Value(u32), expected: u32) void {
    _ = os_sync_wait_on_address(@ptrCast(&atom.raw), expected, @sizeOf(u32), OS_SYNC_WAIT_ON_ADDRESS_NONE);
}

inline fn futexWakeOneU32(atom: *std.atomic.Value(u32)) void {
    _ = os_sync_wake_by_address_any(@ptrCast(&atom.raw), @sizeOf(u32), OS_SYNC_WAKE_BY_ADDRESS_NONE);
}

/// Static-shape execution strategy for growing KV cache lengths during chunked
/// hybrid prefill (`A3`, `ADR-0010`, Spike `S6`).
pub const StaticShapeStrategy = enum {
    /// Precompiled power-of-two static KV bucket CoreML graphs
    /// (`B in {512, 1024, 2048, 4096}`) executing QKV + causal/pad-masked GQA
    /// attention + O-projection on ANE and writing K/V straight into `KvTile`s.
    bucketed_full_ane,
    /// Single fixed-shape 512-token ANE QKV projection writing `q_out`,
    /// `k_tile_out`, and `v_tile_out` zero-copy into IOSurface `KvTile`s,
    /// paired with Metal 4 GPU split-KV attention + O-projection (scales to
    /// arbitrary KV lengths with constant compile cost and zero padding waste).
    split_ane_qkv_gpu_attn,
    /// GPU-only chunked attention baseline and short-prompt (`<= 512`) fallback.
    gpu_only,

    pub fn label(self: StaticShapeStrategy) []const u8 {
        return switch (self) {
            .bucketed_full_ane => "bucketed_full_ane",
            .split_ane_qkv_gpu_attn => "split_ane_qkv_gpu_attn",
            .gpu_only => "gpu_only",
        };
    }
};

/// Returns the index of the smallest bucket `B >= total_tokens` in sorted
/// ascending `buckets`, or `null` if `total_tokens` exceeds the largest bucket.
pub fn selectBucketIndex(total_tokens: u32, buckets: []const u32) ?usize {
    for (buckets, 0..) |b, i| {
        if (b >= total_tokens) return i;
    }
    return null;
}

/// Returns the smallest bucket length `B >= total_tokens`, or `null` if exceeded.
pub fn selectBucketLen(total_tokens: u32, buckets: []const u32) ?u32 {
    const idx = selectBucketIndex(total_tokens, buckets) orelse return null;
    return buckets[idx];
}

/// Populates a `[chunk_tokens, bucket_tokens]` fp16 causal + padding attention
/// mask for static KV bucket length `bucket_tokens >= chunk_tokens`.
///
/// Layout of keys along columns `0..bucket_tokens`:
/// - `[0 .. valid_past_tokens)`: valid past KV tokens (`0.0`).
/// - `[valid_past_tokens .. bucket_tokens - chunk_tokens)`: padded unused past
///   bucket slots (`-10000.0`).
/// - `[past_cap + j]` for `0 <= j < chunk_tokens`: current chunk token `j`,
///   visible to query row `i` iff `j <= i` and `j < valid_chunk_tokens`.
///   Padded query rows (`i >= valid_chunk_tokens`) self-attend to column
///   `past_cap + i` so softmax never produces `NaN`.
pub fn fillCausalBucketMask(
    dst: []f16,
    chunk_tokens: u32,
    bucket_tokens: u32,
    valid_past_tokens: u32,
    valid_chunk_tokens: u32,
) void {
    std.debug.assert(bucket_tokens >= chunk_tokens);
    std.debug.assert(dst.len >= @as(usize, chunk_tokens) * bucket_tokens);
    const past_cap: u32 = bucket_tokens - chunk_tokens;
    const clamped_past: u32 = @min(valid_past_tokens, past_cap);
    const clamped_cur: u32 = @min(valid_chunk_tokens, chunk_tokens);
    const zero_h: f16 = @floatCast(0.0);

    for (0..chunk_tokens) |row_u| {
        const row: u32 = @intCast(row_u);
        const row_slice = dst[row_u * bucket_tokens ..][0..bucket_tokens];
        if (row < clamped_cur) {
            for (0..past_cap) |col_u| {
                row_slice[col_u] = if (col_u < clamped_past) zero_h else MASK_NEG_INF_F16;
            }
            for (0..chunk_tokens) |j_u| {
                const j: u32 = @intCast(j_u);
                row_slice[past_cap + j_u] = if (j <= row) zero_h else MASK_NEG_INF_F16;
            }
        } else {
            @memset(row_slice, MASK_NEG_INF_F16);
            row_slice[past_cap + row] = zero_h;
        }
    }
}

// ---------------------------------------------------------------------------
// Reusable 2-Chunk Hybrid Wavefront Prefill Engine (A1, ADR-0010)
// ---------------------------------------------------------------------------

/// One scheduled `(layer, chunk)` step handed to ANE or GPU callbacks.
pub const StepTask = struct {
    layer: u32,
    chunk: u32,
    /// Double-buffered activation slot index (`chunk & 1`, always disjoint
    /// between the concurrently running ANE and GPU steps).
    slot: u1,
};

/// Callback signature invoked on the `USER_INITIATED` ANE driver thread
/// (`ane_step_fn`) or caller GPU thread (`gpu_ffn_step_fn`).
pub const StepFn = *const fn (ctx: *anyopaque, task: StepTask) anyerror!void;

/// Timing and concurrency telemetry from one `WavefrontPrefill.runWavefront` call.
pub const WavefrontStats = struct {
    num_chunks: u32,
    num_layers: u32,
    total_steps: u32,
    overlapped_steps: u32,
    total_ns: u64,
    ane_total_ns: u64,
    gpu_total_ns: u64,
    ane_thread_qos: qos.QosClass,

    pub fn totalMs(self: WavefrontStats) f64 {
        return @as(f64, @floatFromInt(self.total_ns)) / 1_000_000.0;
    }

    pub fn msPerChunk(self: WavefrontStats) f64 {
        if (self.num_chunks == 0) return 0.0;
        return self.totalMs() / @as(f64, @floatFromInt(self.num_chunks));
    }

    pub fn aneTotalMs(self: WavefrontStats) f64 {
        return @as(f64, @floatFromInt(self.ane_total_ns)) / 1_000_000.0;
    }

    pub fn gpuTotalMs(self: WavefrontStats) f64 {
        return @as(f64, @floatFromInt(self.gpu_total_ns)) / 1_000_000.0;
    }

    pub fn sequentialSumMs(self: WavefrontStats) f64 {
        return self.aneTotalMs() + self.gpuTotalMs();
    }

    /// Measured overlap efficiency `(t_A + t_F) / t_wavefront` (`> 1.0` when
    /// ANE and GPU genuinely overlap; approaches `2.0` when `t_A == t_F`).
    pub fn overlapEfficiency(self: WavefrontStats) f64 {
        if (self.total_ns == 0) return 0.0;
        return @as(f64, @floatFromInt(self.ane_total_ns + self.gpu_total_ns)) /
            @as(f64, @floatFromInt(self.total_ns));
    }

    pub fn tokensPerSec(self: WavefrontStats, chunk_tokens: u32) f64 {
        if (self.total_ns == 0 or chunk_tokens == 0 or self.num_chunks == 0) return 0.0;
        const total_toks: f64 = @floatFromInt(@as(u64, self.num_chunks) * chunk_tokens);
        return total_toks / (@as(f64, @floatFromInt(self.total_ns)) / 1_000_000_000.0);
    }
};

const AneWorkCmd = struct {
    kind: enum(u8) { run_step, shutdown },
    task: StepTask,
    ctx: ?*anyopaque,
    step_fn: ?StepFn,
};

const AneWorkDone = struct {
    elapsed_ns: u64,
    err: ?anyerror,
};

/// Returns the `k`-th `(layer, chunk)` step in the dependency-safe 2-chunk
/// wavefront traversal order (`0 <= k < num_chunks * num_layers`).
///
/// Chunks are paired `(2p, 2p + 1)` across all layers:
/// `(0, 2p), (0, 2p+1), (1, 2p), (1, 2p+1), ..., (L-1, 2p), (L-1, 2p+1)`
/// followed immediately by `(0, 2p+2)` (whose `slot = 0` is free once `(L-1, 2p)`
/// completes!), so every consecutive pair with `next.chunk != cur.chunk` has
/// zero data dependency and disjoint double-buffer activation slots (`chunk & 1`).
pub fn wavefrontScheduleItem(k: u32, num_chunks: u32, num_layers: u32) StepTask {
    const pair_steps: u32 = 2 * num_layers;
    const full_pairs: u32 = num_chunks / 2;
    const full_pair_total: u32 = full_pairs * pair_steps;
    if (k < full_pair_total) {
        const pair_idx = k / pair_steps;
        const rem = k % pair_steps;
        const layer = rem / 2;
        const chunk = pair_idx * 2 + (rem & 1);
        return .{ .layer = layer, .chunk = chunk, .slot = @intCast(chunk & 1) };
    }
    const chunk = num_chunks - 1;
    const layer = k - full_pair_total;
    return .{ .layer = layer, .chunk = chunk, .slot = @intCast(chunk & 1) };
}

/// Persistent 2-chunk hybrid ANE/GPU wavefront prefill scheduler (`A1`, `ADR-0010`).
///
/// Spawns one dedicated `USER_INITIATED` ANE driver thread once at startup and
/// drives it through preallocated lock-free `SpscRing`s and a native Darwin
/// `os_sync_wait_on_address` doorbell — zero thread spawns and zero heap
/// allocations on every `runWavefront` call.
pub const WavefrontPrefill = struct {
    const RING_CAP: usize = 8;

    cmd_slots: [RING_CAP]AneWorkCmd,
    done_slots: [RING_CAP]AneWorkDone,
    cmd_ring: ring.SpscRing(AneWorkCmd),
    done_ring: ring.SpscRing(AneWorkDone),

    cmd_seq: std.atomic.Value(u32) align(ring.CACHE_LINE_BYTES),
    done_seq: std.atomic.Value(u32) align(ring.CACHE_LINE_BYTES),
    worker_ready: std.atomic.Value(u32) align(ring.CACHE_LINE_BYTES),
    worker_qos_raw: std.atomic.Value(u32),

    worker_thread: ?std.Thread,

    pub const InitError = std.Thread.SpawnError || ring.InitError || error{OutOfMemory};

    /// Heap-allocates and starts a pinned `WavefrontPrefill` engine (startup only).
    pub fn init(allocator: std.mem.Allocator) InitError!*WavefrontPrefill {
        const self = try allocator.create(WavefrontPrefill);
        errdefer allocator.destroy(self);
        try self.initInPlace();
        return self;
    }

    /// Stops and joins the ANE driver thread and frees the engine.
    pub fn deinit(self: *WavefrontPrefill, allocator: std.mem.Allocator) void {
        self.deinitInPlace();
        allocator.destroy(self);
    }

    /// Initializes a caller-pinned `WavefrontPrefill` in place (must not be moved
    /// in memory until `deinitInPlace` returns).
    pub fn initInPlace(self: *WavefrontPrefill) (std.Thread.SpawnError || ring.InitError)!void {
        self.cmd_ring = try ring.SpscRing(AneWorkCmd).init(&self.cmd_slots);
        self.done_ring = try ring.SpscRing(AneWorkDone).init(&self.done_slots);
        self.cmd_seq = .init(0);
        self.done_seq = .init(0);
        self.worker_ready = .init(0);
        self.worker_qos_raw = .init(@backingInt(qos.QosClass.unspecified));
        self.worker_thread = null;

        const t = try std.Thread.spawn(.{}, workerMain, .{self});
        self.worker_thread = t;

        while (self.worker_ready.load(.acquire) == 0) {
            futexWaitU32(&self.worker_ready, 0);
        }
    }

    /// Gracefully shuts down and joins the ANE driver thread.
    pub fn deinitInPlace(self: *WavefrontPrefill) void {
        if (self.worker_thread) |t| {
            self.pushCmd(.{
                .kind = .shutdown,
                .task = .{ .layer = 0, .chunk = 0, .slot = 0 },
                .ctx = null,
                .step_fn = null,
            });
            t.join();
            self.worker_thread = null;
        }
    }

    /// Returns the observed macOS pthread QoS class of the ANE driver thread.
    pub fn aneThreadQos(self: *const WavefrontPrefill) qos.QosClass {
        const raw = self.worker_qos_raw.load(.acquire);
        return std.enums.fromInt(qos.QosClass, raw) orelse .unspecified;
    }

    fn pushCmd(self: *WavefrontPrefill, cmd: AneWorkCmd) void {
        while (!self.cmd_ring.tryPush(cmd)) {
            std.atomic.spinLoopHint();
        }
        _ = self.cmd_seq.fetchAdd(1, .release);
        futexWakeOneU32(&self.cmd_seq);
    }

    fn waitDone(self: *WavefrontPrefill) AneWorkDone {
        while (true) {
            if (self.done_ring.tryPop()) |res| return res;
            const seen = self.done_seq.load(.acquire);
            if (self.done_ring.tryPop()) |res| return res;
            futexWaitU32(&self.done_seq, seen);
        }
    }

    fn workerMain(self: *WavefrontPrefill) void {
        // Steer ANE driver thread to P-core USER_INITIATED QoS per runtime-pipeline.md.
        if (qos.applyRole(.ane_driver)) |_| {
            if (qos.getCurrentThreadQos()) |tq| {
                self.worker_qos_raw.store(@backingInt(tq.class), .release);
            } else |_| {
                self.worker_qos_raw.store(@backingInt(qos.QosClass.user_initiated), .release);
            }
        } else |_| {
            self.worker_qos_raw.store(@backingInt(qos.QosClass.unspecified), .release);
        }
        self.worker_ready.store(1, .release);
        futexWakeOneU32(&self.worker_ready);

        while (true) {
            const cmd = blk: {
                while (true) {
                    if (self.cmd_ring.tryPop()) |item| break :blk item;
                    const seen = self.cmd_seq.load(.acquire);
                    if (self.cmd_ring.tryPop()) |item| break :blk item;
                    futexWaitU32(&self.cmd_seq, seen);
                }
            };
            if (cmd.kind == .shutdown) return;

            const t0 = vmstat.nowNs() catch 0;
            const err_opt: ?anyerror = if (cmd.step_fn) |fn_ptr|
                if (fn_ptr(cmd.ctx.?, cmd.task)) |_| null else |err| err
            else
                error.InvalidArgument;
            const t1 = vmstat.nowNs() catch t0;

            const done: AneWorkDone = .{
                .elapsed_ns = t1 -| t0,
                .err = err_opt,
            };
            while (!self.done_ring.tryPush(done)) {
                std.atomic.spinLoopHint();
            }
            _ = self.done_seq.fetchAdd(1, .release);
            futexWakeOneU32(&self.done_seq);
        }
    }

    /// Executes `num_chunks * num_layers` `(ane_step_fn, gpu_ffn_step_fn)` steps
    /// using the 2-chunk overlapping wavefront schedule with zero dynamic heap
    /// allocation. Guarantees that any in-flight ANE step is drained before
    /// returning even on error.
    pub fn runWavefront(
        self: *WavefrontPrefill,
        num_chunks: u32,
        num_layers: u32,
        ctx: *anyopaque,
        ane_step_fn: StepFn,
        gpu_ffn_step_fn: StepFn,
    ) anyerror!WavefrontStats {
        if (num_chunks == 0 or num_layers == 0) return error.InvalidArgument;
        if (self.worker_thread == null) return error.InvalidArgument;

        const total_steps = std.math.mul(u32, num_chunks, num_layers) catch return error.InvalidArgument;
        var ane_total_ns: u64 = 0;
        var gpu_total_ns: u64 = 0;
        var overlapped_steps: u32 = 0;

        const wall_t0 = try vmstat.nowNs();

        // Prologue: execute A(0, 0) on ANE driver thread.
        const first_task = wavefrontScheduleItem(0, num_chunks, num_layers);
        self.pushCmd(.{
            .kind = .run_step,
            .task = first_task,
            .ctx = ctx,
            .step_fn = ane_step_fn,
        });
        const first_done = self.waitDone();
        ane_total_ns += first_done.elapsed_ns;
        if (first_done.err) |e| return e;

        // Steady state + epilogue.
        var step_idx: u32 = 0;
        while (step_idx < total_steps) : (step_idx += 1) {
            const cur = wavefrontScheduleItem(step_idx, num_chunks, num_layers);
            if (step_idx + 1 < total_steps) {
                const nxt = wavefrontScheduleItem(step_idx + 1, num_chunks, num_layers);
                if (nxt.chunk != cur.chunk) {
                    // Dependencies of `nxt` are satisfied and `nxt.slot != cur.slot`:
                    // launch ANE(nxt) asynchronously while running GPU(cur) concurrently!
                    self.pushCmd(.{
                        .kind = .run_step,
                        .task = nxt,
                        .ctx = ctx,
                        .step_fn = ane_step_fn,
                    });
                    const gt0 = try vmstat.nowNs();
                    const gpu_err: ?anyerror = if (gpu_ffn_step_fn(ctx, cur)) |_| null else |e| e;
                    const gt1 = try vmstat.nowNs();
                    gpu_total_ns += gt1 -| gt0;

                    // Always drain in-flight ANE command before propagating any error.
                    const ane_done = self.waitDone();
                    ane_total_ns += ane_done.elapsed_ns;
                    overlapped_steps += 1;

                    if (gpu_err) |e| return e;
                    if (ane_done.err) |e| return e;
                } else {
                    // Same chunk (only occurs on an odd final chunk): sequential dependency.
                    const gt0 = try vmstat.nowNs();
                    try gpu_ffn_step_fn(ctx, cur);
                    const gt1 = try vmstat.nowNs();
                    gpu_total_ns += gt1 -| gt0;

                    self.pushCmd(.{
                        .kind = .run_step,
                        .task = nxt,
                        .ctx = ctx,
                        .step_fn = ane_step_fn,
                    });
                    const ane_done = self.waitDone();
                    ane_total_ns += ane_done.elapsed_ns;
                    if (ane_done.err) |e| return e;
                }
            } else {
                // Epilogue: finish GPU FFN/MoE for the last step.
                const gt0 = try vmstat.nowNs();
                try gpu_ffn_step_fn(ctx, cur);
                const gt1 = try vmstat.nowNs();
                gpu_total_ns += gt1 -| gt0;
            }
        }

        const wall_t1 = try vmstat.nowNs();
        return .{
            .num_chunks = num_chunks,
            .num_layers = num_layers,
            .total_steps = total_steps,
            .overlapped_steps = overlapped_steps,
            .total_ns = wall_t1 -| wall_t0,
            .ane_total_ns = ane_total_ns,
            .gpu_total_ns = gpu_total_ns,
            .ane_thread_qos = self.aneThreadQos(),
        };
    }

    /// Type-safe generic wrapper around `runWavefront` for `Qwen3Model` integration.
    pub fn runWavefrontTyped(
        self: *WavefrontPrefill,
        comptime Ctx: type,
        num_chunks: u32,
        num_layers: u32,
        ctx: *Ctx,
        comptime ane_fn: fn (*Ctx, StepTask) anyerror!void,
        comptime gpu_fn: fn (*Ctx, StepTask) anyerror!void,
    ) anyerror!WavefrontStats {
        const Trampoline = struct {
            fn callAne(raw: *anyopaque, task: StepTask) anyerror!void {
                return ane_fn(@ptrCast(@alignCast(raw)), task);
            }
            fn callGpu(raw: *anyopaque, task: StepTask) anyerror!void {
                return gpu_fn(@ptrCast(@alignCast(raw)), task);
            }
        };
        return self.runWavefront(num_chunks, num_layers, @ptrCast(ctx), Trampoline.callAne, Trampoline.callGpu);
    }
};

// ---------------------------------------------------------------------------
// Reusable ANE Chunked Attention / QKV Blocks (for Qwen3Model & Spike S6)
// ---------------------------------------------------------------------------

pub const AneAttentionGeometry = struct {
    hidden_size: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    chunk_tokens: u32 = DEFAULT_CHUNK_TOKENS,

    pub fn qDim(self: AneAttentionGeometry) u32 {
        return self.num_q_heads * self.head_dim;
    }
    pub fn kvDim(self: AneAttentionGeometry) u32 {
        return self.num_kv_heads * self.head_dim;
    }
    pub fn validate(self: AneAttentionGeometry) bool {
        return self.hidden_size > 0 and self.num_q_heads > 0 and self.num_kv_heads > 0 and
            self.head_dim > 0 and self.chunk_tokens > 0 and
            (self.num_q_heads % self.num_kv_heads) == 0 and
            (self.hidden_size % metal.KV_TILE_ROW_ALIGN_F16) == 0 and
            (self.head_dim % metal.KV_TILE_ROW_ALIGN_F16) == 0;
    }
};

/// Single fixed-shape 512-token ANE QKV projection block writing `q_out`,
/// `k_tile_out`, and `v_tile_out` zero-copy into IOSurface `KvTile`s.
pub const AneQkvBlock = struct {
    geom: AneAttentionGeometry,
    model: metal.AneModel,
    model_info: metal.AneModelInfo,
    compile_ns: u64,
    in_name: [:0]const u8 = "x_in",
    q_name: [:0]const u8 = "q_out",
    k_name: [:0]const u8 = "k_tile_out",
    v_name: [:0]const u8 = "v_tile_out",
    flat_2d_io: bool = false,

    pub const InitError = metal.Error || metal.ProtoEmitError || vmstat.Error || error{InvalidConfig};

    pub fn initFromWeights(
        allocator: std.mem.Allocator,
        geom: AneAttentionGeometry,
        w_q: []const f16,
        w_k: []const f16,
        w_v: []const f16,
        units: metal.AneComputeUnits,
    ) InitError!AneQkvBlock {
        if (!geom.validate()) return error.InvalidConfig;
        const spec_bytes = try metal.emitCoreMlQkvProjectionBlock(allocator, .{
            .hidden_size = geom.hidden_size,
            .num_q_heads = geom.num_q_heads,
            .num_kv_heads = geom.num_kv_heads,
            .head_dim = geom.head_dim,
            .chunk_tokens = geom.chunk_tokens,
            .w_q = w_q,
            .w_k = w_k,
            .w_v = w_v,
        });
        defer allocator.free(spec_bytes);

        const t0 = try vmstat.nowNs();
        const model = try metal.AneModel.loadSpecData(spec_bytes, units);
        errdefer model.close();
        const t1 = try vmstat.nowNs();
        const minfo = try model.info();
        return .{
            .geom = geom,
            .model = model,
            .model_info = minfo,
            .compile_ns = t1 -| t0,
        };
    }

    pub fn initFromPath(
        geom: AneAttentionGeometry,
        path: [:0]const u8,
        units: metal.AneComputeUnits,
    ) InitError!AneQkvBlock {
        if (!geom.validate()) return error.InvalidConfig;
        const t0 = try vmstat.nowNs();
        const model = try metal.AneModel.loadPath(path, units);
        errdefer model.close();
        const t1 = try vmstat.nowNs();
        const minfo = try model.info();

        // Auto-detect offline W7 (`hidden_in` -> `q_out`, `k_out`, `v_out`, flat 2D)
        // vs in-memory 4D ADR-0004 (`x_in` -> `q_out`, `k_tile_out`, `v_tile_out`).
        var name_buf: [64]u8 = undefined;
        const first_in = try model.ioName(false, 0, &name_buf);
        const is_w7 = std.mem.eql(u8, first_in, "hidden_in");
        return .{
            .geom = geom,
            .model = model,
            .model_info = minfo,
            .compile_ns = t1 -| t0,
            .in_name = if (is_w7) "hidden_in" else "x_in",
            .q_name = "q_out",
            .k_name = if (is_w7) "k_out" else "k_tile_out",
            .v_name = if (is_w7) "v_out" else "v_tile_out",
            .flat_2d_io = is_w7,
        };
    }

    pub fn close(self: AneQkvBlock) void {
        self.model.close();
    }

    pub fn compileMs(self: AneQkvBlock) f64 {
        return @as(f64, @floatFromInt(self.compile_ns)) / 1_000_000.0;
    }

    /// Prebuilds a zero-copy `AnePlan` binding `x_tile` -> `(q_tile, k_tile, v_tile)`.
    pub const BoundPlan = struct {
        x_t: metal.AneTensor,
        q_t: metal.AneTensor,
        k_t: metal.AneTensor,
        v_t: metal.AneTensor,
        plan: metal.AnePlan,

        pub fn close(self: BoundPlan) void {
            self.plan.close();
            self.v_t.close();
            self.k_t.close();
            self.q_t.close();
            self.x_t.close();
        }
    };

    pub fn bindTiles(
        self: AneQkvBlock,
        x_tile: metal.KvTile,
        q_tile: metal.KvTile,
        k_tile_out: metal.KvTile,
        v_tile_out: metal.KvTile,
    ) metal.Error!BoundPlan {
        const g = self.geom;
        const x_t = if (self.flat_2d_io)
            try metal.AneTensor.fromKvTile(x_tile, &.{ g.chunk_tokens, g.hidden_size })
        else
            try metal.AneTensor.fromKvTile(x_tile, &.{ 1, g.chunk_tokens, g.hidden_size });
        errdefer x_t.close();
        const q_t = if (self.flat_2d_io)
            try metal.AneTensor.fromKvTile(q_tile, &.{ g.chunk_tokens, g.qDim() })
        else
            try metal.AneTensor.fromKvTile(q_tile, &.{ 1, g.chunk_tokens, g.qDim() });
        errdefer q_t.close();
        const k_t = if (self.flat_2d_io)
            try metal.AneTensor.fromKvTile(k_tile_out, &.{ g.chunk_tokens, g.kvDim() })
        else
            try metal.AneTensor.fromKvTile(k_tile_out, &.{ 1, g.num_kv_heads, g.chunk_tokens, g.head_dim });
        errdefer k_t.close();
        const v_t = if (self.flat_2d_io)
            try metal.AneTensor.fromKvTile(v_tile_out, &.{ g.chunk_tokens, g.kvDim() })
        else
            try metal.AneTensor.fromKvTile(v_tile_out, &.{ 1, g.num_kv_heads, g.chunk_tokens, g.head_dim });
        errdefer v_t.close();

        const plan = try metal.AnePlan.init(
            &.{.{ .name = self.in_name, .tensor = x_t }},
            &.{
                .{ .name = self.q_name, .tensor = q_t },
                .{ .name = self.k_name, .tensor = k_t },
                .{ .name = self.v_name, .tensor = v_t },
            },
        );
        return .{ .x_t = x_t, .q_t = q_t, .k_t = k_t, .v_t = v_t, .plan = plan };
    }

    pub fn run(self: AneQkvBlock, bound: BoundPlan) metal.Error!metal.AnePredictInfo {
        return self.model.predict(bound.plan);
    }
};

/// Single compiled static KV bucket attention model (`B >= chunk_tokens`).
pub const AneAttentionBucket = struct {
    geom: AneAttentionGeometry,
    bucket_tokens: u32,
    model: metal.AneModel,
    model_info: metal.AneModelInfo,
    compile_ns: u64,

    pub const InitError = metal.Error || metal.ProtoEmitError || vmstat.Error || error{InvalidConfig};

    pub fn initFromWeights(
        allocator: std.mem.Allocator,
        geom: AneAttentionGeometry,
        bucket_tokens: u32,
        w_q: []const f16,
        w_k: []const f16,
        w_v: []const f16,
        w_o: []const f16,
        units: metal.AneComputeUnits,
    ) InitError!AneAttentionBucket {
        if (!geom.validate() or bucket_tokens < geom.chunk_tokens or
            (bucket_tokens % metal.KV_TILE_ROW_ALIGN_F16) != 0)
        {
            return error.InvalidConfig;
        }
        const spec_bytes = try metal.emitCoreMlAttentionBlock(allocator, .{
            .hidden_size = geom.hidden_size,
            .num_q_heads = geom.num_q_heads,
            .num_kv_heads = geom.num_kv_heads,
            .head_dim = geom.head_dim,
            .chunk_tokens = geom.chunk_tokens,
            .kv_bucket_tokens = bucket_tokens,
            .use_conv_proj = true,
            .w_q = w_q,
            .w_k = w_k,
            .w_v = w_v,
            .w_o = w_o,
        });
        defer allocator.free(spec_bytes);

        const t0 = try vmstat.nowNs();
        const model = try metal.AneModel.loadSpecData(spec_bytes, units);
        errdefer model.close();
        const t1 = try vmstat.nowNs();
        const minfo = try model.info();
        return .{
            .geom = geom,
            .bucket_tokens = bucket_tokens,
            .model = model,
            .model_info = minfo,
            .compile_ns = t1 -| t0,
        };
    }

    pub fn initFromPath(
        geom: AneAttentionGeometry,
        bucket_tokens: u32,
        path: [:0]const u8,
        units: metal.AneComputeUnits,
    ) InitError!AneAttentionBucket {
        if (!geom.validate() or bucket_tokens < geom.chunk_tokens) return error.InvalidConfig;
        const t0 = try vmstat.nowNs();
        const model = try metal.AneModel.loadPath(path, units);
        errdefer model.close();
        const t1 = try vmstat.nowNs();
        const minfo = try model.info();
        return .{
            .geom = geom,
            .bucket_tokens = bucket_tokens,
            .model = model,
            .model_info = minfo,
            .compile_ns = t1 -| t0,
        };
    }

    pub fn close(self: AneAttentionBucket) void {
        self.model.close();
    }

    pub fn compileMs(self: AneAttentionBucket) f64 {
        return @as(f64, @floatFromInt(self.compile_ns)) / 1_000_000.0;
    }

    pub const BoundPlan = struct {
        x_t: metal.AneTensor,
        kp_t: ?metal.AneTensor,
        vp_t: ?metal.AneTensor,
        mask_t: metal.AneTensor,
        out_t: metal.AneTensor,
        ko_t: metal.AneTensor,
        vo_t: metal.AneTensor,
        plan: metal.AnePlan,

        pub fn close(self: BoundPlan) void {
            self.plan.close();
            self.vo_t.close();
            self.ko_t.close();
            self.out_t.close();
            self.mask_t.close();
            if (self.vp_t) |t| t.close();
            if (self.kp_t) |t| t.close();
            self.x_t.close();
        }
    };

    pub fn bindTiles(
        self: AneAttentionBucket,
        x_tile: metal.KvTile,
        k_past_tile: ?metal.KvTile,
        v_past_tile: ?metal.KvTile,
        mask_tile: metal.KvTile,
        attn_out_tile: metal.KvTile,
        k_tile_out: metal.KvTile,
        v_tile_out: metal.KvTile,
    ) metal.Error!BoundPlan {
        const g = self.geom;
        const past = self.bucket_tokens - g.chunk_tokens;

        const x_t = try metal.AneTensor.fromKvTile(x_tile, &.{ 1, g.chunk_tokens, g.hidden_size });
        errdefer x_t.close();
        const mask_t = try metal.AneTensor.fromKvTile(mask_tile, &.{ 1, 1, g.chunk_tokens, self.bucket_tokens });
        errdefer mask_t.close();
        const out_t = try metal.AneTensor.fromKvTile(attn_out_tile, &.{ 1, g.chunk_tokens, g.hidden_size });
        errdefer out_t.close();
        const ko_t = try metal.AneTensor.fromKvTile(k_tile_out, &.{ 1, g.num_kv_heads, g.chunk_tokens, g.head_dim });
        errdefer ko_t.close();
        const vo_t = try metal.AneTensor.fromKvTile(v_tile_out, &.{ 1, g.num_kv_heads, g.chunk_tokens, g.head_dim });
        errdefer vo_t.close();

        var kp_t: ?metal.AneTensor = null;
        var vp_t: ?metal.AneTensor = null;
        errdefer if (vp_t) |t| t.close();
        errdefer if (kp_t) |t| t.close();

        var in_binds: [4]metal.AneBinding = undefined;
        var in_count: usize = 0;
        in_binds[in_count] = .{ .name = "x_in", .tensor = x_t };
        in_count += 1;

        if (past > 0) {
            const kpt = k_past_tile orelse return error.InvalidArgument;
            const vpt = v_past_tile orelse return error.InvalidArgument;
            kp_t = try metal.AneTensor.fromKvTile(kpt, &.{ 1, g.num_kv_heads, past, g.head_dim });
            vp_t = try metal.AneTensor.fromKvTile(vpt, &.{ 1, g.num_kv_heads, past, g.head_dim });
            in_binds[in_count] = .{ .name = "k_past", .tensor = kp_t.? };
            in_count += 1;
            in_binds[in_count] = .{ .name = "v_past", .tensor = vp_t.? };
            in_count += 1;
        }
        in_binds[in_count] = .{ .name = "mask", .tensor = mask_t };
        in_count += 1;

        const out_binds = [_]metal.AneBinding{
            .{ .name = "attn_out", .tensor = out_t },
            .{ .name = "k_tile_out", .tensor = ko_t },
            .{ .name = "v_tile_out", .tensor = vo_t },
        };
        const plan = try metal.AnePlan.init(in_binds[0..in_count], &out_binds);
        return .{
            .x_t = x_t,
            .kp_t = kp_t,
            .vp_t = vp_t,
            .mask_t = mask_t,
            .out_t = out_t,
            .ko_t = ko_t,
            .vo_t = vo_t,
            .plan = plan,
        };
    }

    pub fn run(self: AneAttentionBucket, bound: BoundPlan) metal.Error!metal.AnePredictInfo {
        return self.model.predict(bound.plan);
    }
};

// ---------------------------------------------------------------------------
// Spike S6 Benchmark: Static-Shape KV Buckets, fp16 Numerics, 2-Chunk Overlap
// ---------------------------------------------------------------------------

pub const SpikeS6Config = struct {
    /// Default geometry matches Qwen3-30B-A3B (`ADR-0010`).
    hidden_size: u32 = 2048,
    num_q_heads: u32 = 32,
    num_kv_heads: u32 = 4,
    head_dim: u32 = 128,
    chunk_tokens: u32 = DEFAULT_CHUNK_TOKENS,
    buckets: [4]u32 = DEFAULT_KV_BUCKETS,

    /// GPU FFN/MoE block geometry for the 2-chunk wavefront overlap test.
    moe_intermediate_size: u32 = 768,
    num_experts: u32 = 128,
    top_k: u32 = 8,
    moe_bits: quant.QuantBits = .q3,

    /// Wavefront overlap dimensions (`num_chunks >= 2`).
    wavefront_chunks: u32 = 2,
    wavefront_layers: u32 = 6,

    /// Sub-tile size for GPU causal chunked attention (`<= attention.MAX_TREE_NODES`).
    gpu_attn_tile: u32 = 64,
    /// Number of query rows sampled for CPU fp64 reference numerical check.
    precision_sample_rows: u32 = 32,

    warmup_iters: u32 = 2,
    timed_iters: u32 = 5,
    timeout_ms: u64 = 30_000,
};

pub const BucketBenchRow = struct {
    kv_len: u32,
    compile_ms: f64,
    total_layers: u32,
    ane_preferred_layers: u32,
    uses_ane: bool,
    all_outputs_zero_copy: bool,
    bucketed_ane_ms: f64,
    split_ane_qkv_ms: f64,
    split_gpu_attn_ms: f64,
    split_total_ms: f64,
    gpu_only_ms: f64,

    pub fn bucketedTokPerS(self: BucketBenchRow, chunk_tokens: u32) f64 {
        return if (self.bucketed_ane_ms > 0.0)
            (@as(f64, @floatFromInt(chunk_tokens)) * 1000.0) / self.bucketed_ane_ms
        else
            0.0;
    }

    pub fn splitTokPerS(self: BucketBenchRow, chunk_tokens: u32) f64 {
        return if (self.split_total_ms > 0.0)
            (@as(f64, @floatFromInt(chunk_tokens)) * 1000.0) / self.split_total_ms
        else
            0.0;
    }

    pub fn gpuOnlyTokPerS(self: BucketBenchRow, chunk_tokens: u32) f64 {
        return if (self.gpu_only_ms > 0.0)
            (@as(f64, @floatFromInt(chunk_tokens)) * 1000.0) / self.gpu_only_ms
        else
            0.0;
    }

    pub fn fastestStrategy(self: BucketBenchRow) StaticShapeStrategy {
        if (self.bucketed_ane_ms <= self.split_total_ms and self.bucketed_ane_ms <= self.gpu_only_ms) {
            return .bucketed_full_ane;
        }
        if (self.split_total_ms <= self.gpu_only_ms) {
            return .split_ane_qkv_gpu_attn;
        }
        return .gpu_only;
    }
};

pub const PrecisionSummary = struct {
    eval_kv_len: u32,
    sampled_rows: u32,
    attn_max_abs_err: f32,
    attn_rmse: f32,
    kv_max_abs_err: f32,
    kv_rmse: f32,
};

pub const WavefrontBenchSummary = struct {
    num_chunks: u32,
    num_layers: u32,
    chunk_tokens: u32,
    ane_strategy: StaticShapeStrategy,
    ane_thread_qos: qos.QosClass,
    overlapped_steps: u32,
    total_steps: u32,
    solo_ane_ms_per_layer_chunk: f64,
    solo_gpu_ffn_ms_per_layer_chunk: f64,
    solo_gpu_attn_ms_per_layer_chunk: f64,
    sequential_ms_per_chunk: f64,
    wavefront_ms_per_chunk: f64,
    gpu_only_ms_per_chunk: f64,
    overlap_efficiency: f64,
    wavefront_tok_per_s: f64,
    sequential_tok_per_s: f64,
    gpu_only_tok_per_s: f64,
    speedup_vs_sequential: f64,
    speedup_vs_gpu_only: f64,
    all_outputs_zero_copy: bool,
};

pub const SpikeS6Report = struct {
    config: SpikeS6Config,
    ane_core_count: u32,
    qkv_compile_ms: f64,
    qkv_total_layers: u32,
    qkv_ane_preferred_layers: u32,
    qkv_uses_ane: bool,
    qkv_zero_copy: bool,
    buckets: [4]BucketBenchRow,
    precision: PrecisionSummary,
    wavefront: WavefrontBenchSummary,
    recommended_strategy: StaticShapeStrategy,
    recommended_max_bucket: u32,
    vm_before: vmstat.Snapshot,
    vm_after: vmstat.Snapshot,

    pub fn allBucketsUseAne(self: *const SpikeS6Report) bool {
        if (!self.qkv_uses_ane) return false;
        for (self.buckets) |b| {
            if (!b.uses_ane) return false;
        }
        return true;
    }

    pub fn allBucketsZeroCopy(self: *const SpikeS6Report) bool {
        if (!self.qkv_zero_copy or !self.wavefront.all_outputs_zero_copy) return false;
        for (self.buckets) |b| {
            if (!b.all_outputs_zero_copy) return false;
        }
        return true;
    }

    pub fn write(self: *const SpikeS6Report, out: *std.Io.Writer) std.Io.Writer.Error!void {
        try out.print(
            \\Spike S6 Hybrid ANE/GPU Wavefront Prefill & Static KV Bucket Study (A1, A3, ADR-0010)
            \\s6.ane_cores                     {d}
            \\s6.geometry                      hidden={d} q_heads={d} kv_heads={d} head_dim={d} chunk={d}
            \\s6.qkv_only.compile_ms           {d:.1} (ane={d}/{d}, uses_ane={s}, zero_copy={s})
            \\
            \\Static KV Bucket Comparison (per 1 layer, {d}-token chunk)
            \\kv_len  compile(ms)  ane_layers  zc   ane_bucket(ms)  split_qkv+gpu(ms)  gpu_only(ms)  ane_tok/s  gpu_tok/s  fastest
            \\
        , .{
            self.ane_core_count,
            self.config.hidden_size,
            self.config.num_q_heads,
            self.config.num_kv_heads,
            self.config.head_dim,
            self.config.chunk_tokens,
            self.qkv_compile_ms,
            self.qkv_ane_preferred_layers,
            self.qkv_total_layers,
            if (self.qkv_uses_ane) "yes" else "no",
            if (self.qkv_zero_copy) "yes" else "no",
            self.config.chunk_tokens,
        });

        for (self.buckets) |row| {
            try out.print(
                \\{d: >6}  {d: >11.1}  {d: >4}/{d: <5}  {s: <3}  {d: >14.3}  {d: >17.3}  {d: >12.3}  {d: >9.0}  {d: >9.0}  {s}
                \\
            , .{
                row.kv_len,
                row.compile_ms,
                row.ane_preferred_layers,
                row.total_layers,
                if (row.all_outputs_zero_copy) "yes" else "no",
                row.bucketed_ane_ms,
                row.split_total_ms,
                row.gpu_only_ms,
                row.bucketedTokPerS(self.config.chunk_tokens),
                row.gpuOnlyTokPerS(self.config.chunk_tokens),
                row.fastestStrategy().label(),
            });
        }

        const wf = self.wavefront;
        try out.print(
            \\
            \\FP16 Numerical Precision vs CPU FP64 Reference (kv_len={d}, sampled_rows={d})
            \\s6.precision.attn_max_abs_err    {d:.6}
            \\s6.precision.attn_rmse           {d:.6}
            \\s6.precision.kv_max_abs_err      {d:.6}
            \\s6.precision.kv_rmse             {d:.6}
            \\
            \\2-Chunk Hybrid Wavefront Overlap (A1: {d} chunks x {d} layers, chunk={d} tokens)
            \\s6.wf.ane_thread_qos             {s}
            \\s6.wf.overlapped_steps           {d}/{d}
            \\s6.wf.zero_copy_backing          {s}
            \\s6.wf.t_ane_layer_chunk_ms       {d:.3}
            \\s6.wf.t_gpu_ffn_layer_chunk_ms   {d:.3}
            \\s6.wf.t_gpu_attn_layer_chunk_ms  {d:.3}
            \\s6.wf.sequential_ms_per_chunk    {d:.3} ({d:.0} tok/s)
            \\s6.wf.wavefront_ms_per_chunk     {d:.3} ({d:.0} tok/s)
            \\s6.wf.gpu_only_ms_per_chunk      {d:.3} ({d:.0} tok/s)
            \\s6.wf.overlap_efficiency         {d:.3}x
            \\s6.wf.speedup_vs_sequential      {d:.3}x
            \\s6.wf.speedup_vs_gpu_only        {d:.3}x
            \\s6.a3.recommended_strategy       {s} (max_bucket={d})
            \\s6.swapouts_delta                {d}
            \\s6.pageouts_delta                {d}
            \\
        , .{
            self.precision.eval_kv_len,
            self.precision.sampled_rows,
            self.precision.attn_max_abs_err,
            self.precision.attn_rmse,
            self.precision.kv_max_abs_err,
            self.precision.kv_rmse,
            wf.num_chunks,
            wf.num_layers,
            wf.chunk_tokens,
            @tagName(wf.ane_thread_qos),
            wf.overlapped_steps,
            wf.total_steps,
            if (wf.all_outputs_zero_copy) "yes" else "no",
            wf.solo_ane_ms_per_layer_chunk,
            wf.solo_gpu_ffn_ms_per_layer_chunk,
            wf.solo_gpu_attn_ms_per_layer_chunk,
            wf.sequential_ms_per_chunk,
            wf.sequential_tok_per_s,
            wf.wavefront_ms_per_chunk,
            wf.wavefront_tok_per_s,
            wf.gpu_only_ms_per_chunk,
            wf.gpu_only_tok_per_s,
            wf.overlap_efficiency,
            wf.speedup_vs_sequential,
            wf.speedup_vs_gpu_only,
            self.recommended_strategy.label(),
            self.recommended_max_bucket,
            vmstat.Snapshot.swapoutDelta(self.vm_before, self.vm_after),
            vmstat.Snapshot.pageoutDelta(self.vm_before, self.vm_after),
        });
    }
};

pub const SpikeS6Error = error{
    InvalidConfig,
    NonFiniteOutput,
    OutOfMemory,
} || metal.WrapError || metal.ProtoEmitError || quant.ShapeError ||
    dequant.EncodeError || moe.EncodeError || moe.WorkspaceError ||
    sysinfo.Error || vmstat.Error || vmstat.SwapError ||
    arena_mod.WiredArena.InitError || arena_mod.WiredArena.AllocError ||
    WavefrontPrefill.InitError;

fn fillDeterministicF16(dst: []f16, seed: usize, scale: f32) void {
    var state: u64 = (@as(u64, seed) +% 1) *% 0x9E37_79B9_7F4A_7C15;
    for (dst) |*v| {
        state +%= 0x9E37_79B9_7F4A_7C15;
        var z = state;
        z = (z ^ (z >> 30)) *% 0xBF58_476D_1CE4_E5B9;
        z = (z ^ (z >> 27)) *% 0x94D0_49BB_1331_11EB;
        z = z ^ (z >> 31);
        // Zero-mean symmetric value in `[-1.0, +1.0]` scaled by `scale`.
        const centered: f32 = (@as(f32, @floatFromInt(z & 0xFFFF)) / 32767.5) - 1.0;
        v.* = @floatCast(centered * scale);
    }
}

const Q2Ops = quant.QuantOps(.{ .bits = .q2, .group_size = 64 });
const Q3Ops = quant.QuantOps(.{ .bits = .q3, .group_size = 64 });
const Q4Ops = quant.QuantOps(.{ .bits = .q4, .group_size = 64 });

fn packedWeightBytesBits(bits: quant.QuantBits, n: usize, k: usize) quant.ShapeError!usize {
    return switch (bits) {
        .q2 => Q2Ops.packedWeightBytes(n, k),
        .q3 => Q3Ops.packedWeightBytes(n, k),
        .q4 => Q4Ops.packedWeightBytes(n, k),
    };
}

fn scaleBiasBytesBits(bits: quant.QuantBits, n: usize, k: usize) quant.ShapeError!usize {
    return switch (bits) {
        .q2 => Q2Ops.scaleBiasBytes(n, k),
        .q3 => Q3Ops.scaleBiasBytes(n, k),
        .q4 => Q4Ops.scaleBiasBytes(n, k),
    };
}

fn fillQuantMatrix(
    comptime spec: quant.QuantSpec,
    qweight: []u8,
    scale_bias: []quant.ScaleBias,
    n: usize,
    k: usize,
    seed: usize,
) quant.ShapeError!void {
    const Ops = quant.QuantOps(spec);
    const groups = try Ops.groupsPerRow(k);
    const row_bytes = try Ops.rowWeightBytes(k);
    const q_mask: u8 = (@as(u8, 1) << @intCast(spec.bits.bitWidth())) - 1;
    var codes: [spec.group_size]u8 = undefined;
    for (0..n) |row| {
        for (0..groups) |g| {
            for (0..spec.group_size) |j| {
                codes[j] = @intCast((row * 7 + g * 5 + j * 3 + seed) & q_mask);
            }
            const g_off = row * row_bytes + g * Ops.groupWeightBytes();
            Ops.packGroup(&codes, qweight[g_off..][0..Ops.groupWeightBytes()]);
            const midpoint: f32 = @as(f32, @floatFromInt(q_mask)) * 0.5;
            const sb_scale: f32 = 0.00390625;
            scale_bias[row * groups + g] = .{
                .scale = @floatCast(sb_scale),
                .bias = @floatCast(-midpoint * sb_scale),
            };
        }
    }
}

fn fillQuantMatrixBits(
    bits: quant.QuantBits,
    qweight: []u8,
    scale_bias: []quant.ScaleBias,
    n: usize,
    k: usize,
    seed: usize,
) quant.ShapeError!void {
    return switch (bits) {
        .q2 => fillQuantMatrix(.{ .bits = .q2, .group_size = 64 }, qweight, scale_bias, n, k, seed),
        .q3 => fillQuantMatrix(.{ .bits = .q3, .group_size = 64 }, qweight, scale_bias, n, k, seed),
        .q4 => fillQuantMatrix(.{ .bits = .q4, .group_size = 64 }, qweight, scale_bias, n, k, seed),
    };
}

fn evalBucketPrecisionCpu(
    allocator: std.mem.Allocator,
    geom: AneAttentionGeometry,
    bucket_len: u32,
    valid_past: u32,
    sample_rows: u32,
    x_host: []const f16,
    kp_host: []const f16,
    vp_host: []const f16,
    mask_host: []const f16,
    w_q: []const f16,
    w_k: []const f16,
    w_v: []const f16,
    w_o: []const f16,
    got_attn: []const f16,
    got_k_tile: []const f16,
    got_v_tile: []const f16,
) !PrecisionSummary {
    const h: usize = geom.hidden_size;
    const qh: usize = geom.num_q_heads;
    const kvh: usize = geom.num_kv_heads;
    const d: usize = geom.head_dim;
    const s: usize = geom.chunk_tokens;
    const b: usize = bucket_len;
    const past: usize = b - s;
    const gqa: usize = qh / kvh;
    const q_dim: usize = qh * d;
    const scale: f64 = 1.0 / @sqrt(@as(f64, @floatFromInt(d)));

    // Compute exact K_cur and V_cur on CPU for all `s` tokens (fast: kvh*d is small)
    // and check `k_tile_out` / `v_tile_out` precision across all `kvh * s * d` elements.
    const k_cur_ref = try allocator.alloc(f64, kvh * s * d);
    defer allocator.free(k_cur_ref);
    const v_cur_ref = try allocator.alloc(f64, kvh * s * d);
    defer allocator.free(v_cur_ref);

    var kv_max_err: f64 = 0.0;
    var kv_sq_sum: f64 = 0.0;
    const kv_count: usize = kvh * s * d * 2;

    for (0..s) |ti| {
        const x_row = x_host[ti * h ..][0..h];
        for (0..kvh) |kh| {
            for (0..d) |di| {
                const row_w = kh * d + di;
                var ak: f64 = 0.0;
                var av: f64 = 0.0;
                for (0..h) |ci| {
                    const xv: f64 = @floatCast(x_row[ci]);
                    ak += xv * @as(f64, @floatCast(w_k[row_w * h + ci]));
                    av += xv * @as(f64, @floatCast(w_v[row_w * h + ci]));
                }
                const idx = (kh * s + ti) * d + di;
                k_cur_ref[idx] = ak;
                v_cur_ref[idx] = av;

                const ek = @abs(ak - @as(f64, @floatCast(got_k_tile[idx])));
                const ev = @abs(av - @as(f64, @floatCast(got_v_tile[idx])));
                if (ek > kv_max_err) kv_max_err = ek;
                if (ev > kv_max_err) kv_max_err = ev;
                kv_sq_sum += ek * ek + ev * ev;
            }
        }
    }

    const n_samp: usize = @min(@as(usize, @max(sample_rows, 1)), s);
    const q_row_ref = try allocator.alloc(f64, q_dim);
    defer allocator.free(q_row_ref);
    const ctx_row_ref = try allocator.alloc(f64, q_dim);
    defer allocator.free(ctx_row_ref);
    const scores = try allocator.alloc(f64, b);
    defer allocator.free(scores);

    var attn_max_err: f64 = 0.0;
    var attn_sq_sum: f64 = 0.0;

    for (0..n_samp) |si| {
        // Uniformly span 0..s-1 including causal boundaries `0` and `s-1`.
        const ti: usize = if (n_samp == 1) 0 else (si * (s - 1)) / (n_samp - 1);
        const x_row = x_host[ti * h ..][0..h];

        for (0..q_dim) |qi| {
            var aq: f64 = 0.0;
            const wq_row = w_q[qi * h ..][0..h];
            for (0..h) |ci| {
                aq += @as(f64, @floatCast(x_row[ci])) * @as(f64, @floatCast(wq_row[ci]));
            }
            q_row_ref[qi] = aq;
        }

        const mask_row = mask_host[ti * b ..][0..b];
        for (0..qh) |head| {
            const kvh_i = head / gqa;
            const q_h = q_row_ref[head * d ..][0..d];
            var max_s: f64 = -std.math.inf(f64);

            for (0..b) |bi| {
                const m_val: f64 = @floatCast(mask_row[bi]);
                if (bi < past and bi >= valid_past) {
                    scores[bi] = m_val;
                    continue;
                }
                var dot_v: f64 = 0.0;
                if (bi < past) {
                    const kp_row = kp_host[(kvh_i * past + bi) * d ..][0..d];
                    for (0..d) |di| dot_v += q_h[di] * @as(f64, @floatCast(kp_row[di]));
                } else {
                    const kc_row = k_cur_ref[(kvh_i * s + (bi - past)) * d ..][0..d];
                    for (0..d) |di| dot_v += q_h[di] * kc_row[di];
                }
                const sc = dot_v * scale + m_val;
                scores[bi] = sc;
                if (sc > max_s) max_s = sc;
            }

            var sum_e: f64 = 0.0;
            for (0..b) |bi| {
                if (bi < past and bi >= valid_past) {
                    scores[bi] = 0.0;
                } else {
                    const p = @exp(scores[bi] - max_s);
                    scores[bi] = p;
                    sum_e += p;
                }
            }

            for (0..d) |di| {
                var acc: f64 = 0.0;
                for (0..b) |bi| {
                    const p = scores[bi];
                    if (p == 0.0) continue;
                    const vv: f64 = if (bi < past)
                        @as(f64, @floatCast(vp_host[(kvh_i * past + bi) * d + di]))
                    else
                        v_cur_ref[(kvh_i * s + (bi - past)) * d + di];
                    acc += (p / sum_e) * vv;
                }
                ctx_row_ref[head * d + di] = acc;
            }
        }

        const got_row = got_attn[ti * h ..][0..h];
        for (0..h) |ci| {
            var acc: f64 = 0.0;
            const wo_row = w_o[ci * q_dim ..][0..q_dim];
            for (0..q_dim) |qi| {
                acc += ctx_row_ref[qi] * @as(f64, @floatCast(wo_row[qi]));
            }
            const err = @abs(acc - @as(f64, @floatCast(got_row[ci])));
            if (err > attn_max_err) attn_max_err = err;
            attn_sq_sum += err * err;
        }
    }

    return .{
        .eval_kv_len = bucket_len,
        .sampled_rows = @intCast(n_samp),
        .attn_max_abs_err = @floatCast(attn_max_err),
        .attn_rmse = @floatCast(@sqrt(attn_sq_sum / @as(f64, @floatFromInt(n_samp * h)))),
        .kv_max_abs_err = @floatCast(kv_max_err),
        .kv_rmse = @floatCast(@sqrt(kv_sq_sum / @as(f64, @floatFromInt(kv_count)))),
    };
}

/// Runs Spike S6 on real Apple Silicon ANE + Metal 4 GPU hardware (ADR-0010, A1, A3):
/// 1) Measures CoreML compile latency & `MLComputePlan` ANE layer placement for
///    `512`-token QKV-only and static KV bucket (`512, 1024, 2048, 4096`) graphs,
/// 2) Benchmarks Bucketed Full ANE Attention vs Split ANE QKV + GPU Split-KV
///    Attention vs GPU-only chunked attention at every bucket length,
/// 3) Verifies fp16 numerical accuracy (`max_abs_err`, `rmse`) against CPU fp64,
/// 4) Benchmarks real 2-chunk hybrid wavefront overlap on a `USER_INITIATED` ANE
///    driver thread overlapped with Metal 4 GPU MoE FFN (`WavefrontPrefill`),
///    enforcing `swapouts_delta == 0`.
pub fn runSpikeS6(allocator: std.mem.Allocator, config: SpikeS6Config) SpikeS6Error!SpikeS6Report {
    const geom: AneAttentionGeometry = .{
        .hidden_size = config.hidden_size,
        .num_q_heads = config.num_q_heads,
        .num_kv_heads = config.num_kv_heads,
        .head_dim = config.head_dim,
        .chunk_tokens = config.chunk_tokens,
    };
    if (!geom.validate() or
        config.head_dim != 64 and config.head_dim != 128 or
        (config.hidden_size % 64) != 0 or
        (geom.qDim() % 64) != 0 or
        (geom.kvDim() % 64) != 0 or
        (config.moe_intermediate_size % 64) != 0 or
        config.num_experts == 0 or config.num_experts > moe.MAX_EXPERTS or
        config.top_k == 0 or config.top_k > moe.MAX_TOP_K or config.top_k > config.num_experts or
        config.wavefront_chunks < 2 or config.wavefront_layers == 0 or
        config.timed_iters == 0 or
        config.gpu_attn_tile == 0 or config.gpu_attn_tile > attention.MAX_TREE_NODES or
        (config.chunk_tokens % config.gpu_attn_tile) != 0)
    {
        return error.InvalidConfig;
    }
    for (config.buckets, 0..) |b_len, bi| {
        if (b_len < config.chunk_tokens or
            (b_len % config.chunk_tokens) != 0 or
            (b_len % metal.KV_TILE_ROW_ALIGN_F16) != 0 or
            @as(u64, config.num_kv_heads) * b_len > metal.MAX_TEXTURE_DIM)
        {
            return error.InvalidConfig;
        }
        if (bi > 0 and b_len < config.buckets[bi - 1]) return error.InvalidConfig;
    }

    const facts = try sysinfo.Facts.read();
    const device = try metal.Device.open();
    defer device.close();
    const dev_info = try device.info();
    const wired_budget = budget_mod.Budget.fromHost(facts, dev_info, .{});

    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const deq_pipes = try dequant.DequantPipelines.init(device, library);
    defer deq_pipes.close();
    const attn_pipes = try attention.AttentionPipelines.init(device, library);
    defer attn_pipes.close();
    const moe_pipes = try moe.MoePipelines.init(device, library);
    defer moe_pipes.close();

    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();

    const vm_before = try vmstat.Snapshot.capture();

    const h = config.hidden_size;
    const qh = config.num_q_heads;
    const kvh = config.num_kv_heads;
    const d = config.head_dim;
    const s = config.chunk_tokens;
    const q_dim = geom.qDim();
    const kv_dim = geom.kvDim();
    const max_bucket = config.buckets[config.buckets.len - 1];
    const max_chunks = max_bucket / s;
    const tile_m = config.gpu_attn_tile;
    const sub_tiles = s / tile_m;

    // -----------------------------------------------------------------------
    // 1. Build deterministic fp16 attention weights & shared IOSurface tiles
    // -----------------------------------------------------------------------
    const w_q = try allocator.alloc(f16, @as(usize, q_dim) * h);
    defer allocator.free(w_q);
    const w_k = try allocator.alloc(f16, @as(usize, kv_dim) * h);
    defer allocator.free(w_k);
    const w_v = try allocator.alloc(f16, @as(usize, kv_dim) * h);
    defer allocator.free(w_v);
    const w_o = try allocator.alloc(f16, @as(usize, h) * q_dim);
    defer allocator.free(w_o);

    const w_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(h)));
    fillDeterministicF16(w_q, 101, w_scale);
    fillDeterministicF16(w_k, 102, w_scale);
    fillDeterministicF16(w_v, 103, w_scale);
    fillDeterministicF16(w_o, 104, 1.0 / @sqrt(@as(f32, @floatFromInt(q_dim))));

    const x_host = try allocator.alloc(f16, @as(usize, s) * h);
    defer allocator.free(x_host);
    fillDeterministicF16(x_host, 201, 0.5);

    // Double-buffered activation IOSurface KvTiles (`slot = 0, 1`) shared zero-copy with GPU!
    var x_tiles: [2]metal.KvTile = undefined;
    var out_tiles: [2]metal.KvTile = undefined;
    var created_act: usize = 0;
    defer {
        for (0..created_act) |i| {
            out_tiles[i].close();
            x_tiles[i].close();
        }
    }
    for (0..2) |slot| {
        x_tiles[slot] = try metal.KvTile.init(device, h, s, .{ .alias_buffer = true });
        errdefer x_tiles[slot].close();
        out_tiles[slot] = try metal.KvTile.init(device, h, s, .{ .alias_buffer = true });
        created_act += 1;

        var g = try x_tiles[slot].lock(false);
        errdefer g.release();
        @memcpy(g.f16Slice()[0..x_host.len], x_host);
        try g.unlock();
    }

    const q_out_tile = try metal.KvTile.init(device, q_dim, s, .{ .alias_buffer = true });
    defer q_out_tile.close();

    // Per-chunk IOSurface KV tiles for GPU split-KV attention and ANE KV output backings.
    const k_chunk_tiles = try allocator.alloc(metal.KvTile, max_chunks);
    defer allocator.free(k_chunk_tiles);
    const v_chunk_tiles = try allocator.alloc(metal.KvTile, max_chunks);
    defer allocator.free(v_chunk_tiles);
    var created_kv: usize = 0;
    defer {
        for (0..created_kv) |i| {
            v_chunk_tiles[i].close();
            k_chunk_tiles[i].close();
        }
    }
    for (0..max_chunks) |ci| {
        k_chunk_tiles[ci] = try metal.KvTile.init(device, d, kvh * s, .{});
        errdefer k_chunk_tiles[ci].close();
        v_chunk_tiles[ci] = try metal.KvTile.init(device, d, kvh * s, .{});
        created_kv += 1;

        var kg = try k_chunk_tiles[ci].lock(false);
        errdefer kg.release();
        fillDeterministicF16(kg.f16Slice()[0 .. @as(usize, kvh) * s * d], 300 + ci, 0.5);
        try kg.unlock();

        var vg = try v_chunk_tiles[ci].lock(false);
        errdefer vg.release();
        fillDeterministicF16(vg.f16Slice()[0 .. @as(usize, kvh) * s * d], 400 + ci, 0.5);
        try vg.unlock();
    }

    try compute.makeTilesResident(&x_tiles);
    defer compute.removeTilesResident(&x_tiles) catch |err| {
        std.debug.panic("removeTilesResident failed: {}", .{err});
    };
    try compute.makeTilesResident(&out_tiles);
    defer compute.removeTilesResident(&out_tiles) catch |err| {
        std.debug.panic("removeTilesResident failed: {}", .{err});
    };
    try compute.makeTilesResident(&.{q_out_tile});
    defer compute.removeTilesResident(&.{q_out_tile}) catch |err| {
        std.debug.panic("removeTilesResident failed: {}", .{err});
    };
    try compute.makeTilesResident(k_chunk_tiles);
    defer compute.removeTilesResident(k_chunk_tiles) catch |err| {
        std.debug.panic("removeTilesResident failed: {}", .{err});
    };
    try compute.makeTilesResident(v_chunk_tiles);
    defer compute.removeTilesResident(v_chunk_tiles) catch |err| {
        std.debug.panic("removeTilesResident failed: {}", .{err});
    };

    // -----------------------------------------------------------------------
    // 2. Compile Fixed-Shape ANE QKV-Only Block & Measure Latency
    // -----------------------------------------------------------------------
    const qkv_block = try AneQkvBlock.initFromWeights(allocator, geom, w_q, w_k, w_v, .cpu_and_neural_engine);
    defer qkv_block.close();

    const qkv_bound = try qkv_block.bindTiles(x_tiles[0], q_out_tile, k_chunk_tiles[0], v_chunk_tiles[0]);
    defer qkv_bound.close();

    var qkv_zero_copy = false;
    for (0..config.warmup_iters) |_| {
        qkv_zero_copy = (try qkv_block.run(qkv_bound)).allOutputsZeroCopy();
    }
    const qkv_t0 = try vmstat.nowNs();
    for (0..config.timed_iters) |_| {
        qkv_zero_copy = (try qkv_block.run(qkv_bound)).allOutputsZeroCopy() and qkv_zero_copy;
    }
    const qkv_t1 = try vmstat.nowNs();
    const qkv_ane_ms = (@as(f64, @floatFromInt(qkv_t1 -| qkv_t0)) / @as(f64, @floatFromInt(config.timed_iters))) / 1_000_000.0;

    // -----------------------------------------------------------------------
    // 3. Allocate GPU WiredArena for GPU Attention + MoE FFN Wavefront Workloads
    // -----------------------------------------------------------------------
    const wq_raw_len = try packedWeightBytesBits(config.moe_bits, q_dim, h);
    const sbq_raw_len = try scaleBiasBytesBits(config.moe_bits, q_dim, h);
    const wkv_raw_len = try packedWeightBytesBits(config.moe_bits, kv_dim, h);
    const sbkv_raw_len = try scaleBiasBytesBits(config.moe_bits, kv_dim, h);
    const wo_raw_len = try packedWeightBytesBits(config.moe_bits, h, q_dim);
    const sbo_raw_len = try scaleBiasBytesBits(config.moe_bits, h, q_dim);

    const wq_bytes = amf.align64(wq_raw_len);
    const sbq_bytes = amf.align64(sbq_raw_len);
    const wkv_bytes = amf.align64(wkv_raw_len);
    const sbkv_bytes = amf.align64(sbkv_raw_len);
    const wo_bytes = amf.align64(wo_raw_len);
    const sbo_bytes = amf.align64(sbo_raw_len);

    const gu_w_bytes = try packedWeightBytesBits(config.moe_bits, config.moe_intermediate_size, h);
    const gu_sb_bytes = try scaleBiasBytesBits(config.moe_bits, config.moe_intermediate_size, h);
    const dn_w_bytes = try packedWeightBytesBits(config.moe_bits, h, config.moe_intermediate_size);
    const dn_sb_bytes = try scaleBiasBytesBits(config.moe_bits, h, config.moe_intermediate_size);
    const one_expert_bytes = amf.align64(gu_w_bytes) * 2 + amf.align64(gu_sb_bytes) * 2 +
        amf.align64(dn_w_bytes) + amf.align64(dn_sb_bytes);
    const all_experts_bytes = @as(usize, config.num_experts) * one_expert_bytes;
    const exp_tbl_bytes = amf.align64(@as(usize, config.num_experts) * @sizeOf(moe.ExpertWeights));
    const router_logits_bytes = amf.align64(@as(usize, s) * config.num_experts * @sizeOf(f32));
    const norm_w_bytes = amf.align64(@as(usize, h) * @sizeOf(f16));

    const q_act_bytes = amf.align64(@as(usize, s) * q_dim * @sizeOf(f16));
    const kv_act_bytes = amf.align64(@as(usize, s) * kv_dim * @sizeOf(f16));
    const x_act_bytes = amf.align64(@as(usize, s) * h * @sizeOf(f16));
    const masks_bytes = amf.align64(@as(usize, tile_m) * @sizeOf(u64));
    const kv_tbl_bytes = amf.align64(@as(usize, max_chunks) * @sizeOf(attention.KvTilePair));
    const gqa_tree_layout = attention.gqaTreeScratchLayout(tile_m, qh, d, attention.MAX_SPLITS);
    const moe_ws_spec: moe.WorkspaceSpec = .{
        .max_tokens = s,
        .num_experts = config.num_experts,
        .top_k = config.top_k,
        .hidden_size = h,
        .intermediate_size = config.moe_intermediate_size,
    };
    const moe_ws_bytes = try moe.MoeWorkspace.requiredBytes(moe_ws_spec);

    const total_arena_raw = wq_bytes + sbq_bytes + (wkv_bytes + sbkv_bytes) * 2 + wo_bytes + sbo_bytes +
        all_experts_bytes + exp_tbl_bytes + router_logits_bytes + norm_w_bytes +
        q_act_bytes * 2 + kv_act_bytes * 2 + x_act_bytes * 2 + masks_bytes + kv_tbl_bytes +
        amf.align64(gqa_tree_layout.total_bytes) + moe_ws_bytes + 16 * page.APPLE_PAGE_SIZE;
    const arena_cap = try page.alignUp(total_arena_raw);
    var gpu_arena = try arena_mod.WiredArena.init(arena_cap, wired_budget);
    defer gpu_arena.deinit();

    // Carve GPU attention weight & scratch slices.
    const wq_raw = try gpu_arena.allocAligned(wq_bytes, 64);
    const sbq_raw = try gpu_arena.allocAligned(sbq_bytes, 64);
    const wk_raw = try gpu_arena.allocAligned(wkv_bytes, 64);
    const sbk_raw = try gpu_arena.allocAligned(sbkv_bytes, 64);
    const wv_raw = try gpu_arena.allocAligned(wkv_bytes, 64);
    const sbv_raw = try gpu_arena.allocAligned(sbkv_bytes, 64);
    const wo_raw = try gpu_arena.allocAligned(wo_bytes, 64);
    const sbo_raw = try gpu_arena.allocAligned(sbo_bytes, 64);

    try fillQuantMatrixBits(config.moe_bits, wq_raw[0..wq_raw_len], @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, sbq_raw[0..sbq_raw_len])), q_dim, h, 1);
    try fillQuantMatrixBits(config.moe_bits, wk_raw[0..wkv_raw_len], @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, sbk_raw[0..sbkv_raw_len])), kv_dim, h, 2);
    try fillQuantMatrixBits(config.moe_bits, wv_raw[0..wkv_raw_len], @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, sbv_raw[0..sbkv_raw_len])), kv_dim, h, 3);
    try fillQuantMatrixBits(config.moe_bits, wo_raw[0..wo_raw_len], @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, sbo_raw[0..sbo_raw_len])), h, q_dim, 4);

    const gpu_q_raw = try gpu_arena.allocAligned(q_act_bytes, 64);
    const gpu_ctx_raw = try gpu_arena.allocAligned(q_act_bytes, 64);
    const gpu_k_raw = try gpu_arena.allocAligned(kv_act_bytes, 64);
    const gpu_v_raw = try gpu_arena.allocAligned(kv_act_bytes, 64);
    const gpu_norm_tmp_raw = try gpu_arena.allocAligned(x_act_bytes, 64);
    const gpu_out_tmp_raw = try gpu_arena.allocAligned(x_act_bytes, 64);
    const masks_raw = try gpu_arena.allocAligned(masks_bytes, 64);
    const kv_tbl_raw = try gpu_arena.allocAligned(kv_tbl_bytes, 64);
    const gqa_scratch_raw = try gpu_arena.allocAligned(amf.align64(gqa_tree_layout.total_bytes), 64);
    const norm_w_raw = try gpu_arena.allocAligned(norm_w_bytes, 64);
    @memset(gpu_q_raw, 0);
    @memset(gpu_ctx_raw, 0);
    @memset(gpu_k_raw, 0);
    @memset(gpu_v_raw, 0);
    @memset(gpu_norm_tmp_raw, 0);
    @memset(gpu_out_tmp_raw, 0);
    @memset(gqa_scratch_raw, 0);

    const norm_w_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, norm_w_raw[0 .. @as(usize, h) * @sizeOf(f16)]));
    for (norm_w_f16) |*v| v.* = @floatCast(1.0);

    // Causal lower-triangular ancestor masks for a `tile_m`-token sub-tile.
    const masks_u64: []u64 = @alignCast(std.mem.bytesAsSlice(u64, masks_raw[0 .. @as(usize, tile_m) * @sizeOf(u64)]));
    for (0..tile_m) |i| {
        masks_u64[i] = if (i == 63) ~@as(u64, 0) else (@as(u64, 1) << @as(u6, @intCast(i + 1))) - 1;
    }

    const kv_tbl_slice: []attention.KvTilePair = @alignCast(std.mem.bytesAsSlice(
        attention.KvTilePair,
        kv_tbl_raw[0 .. @as(usize, max_chunks) * @sizeOf(attention.KvTilePair)],
    ));
    for (0..max_chunks) |ci| {
        kv_tbl_slice[ci] = attention.KvTilePair.fromTiles(k_chunk_tiles[ci], v_chunk_tiles[ci], .rgba16f);
    }

    // Carve MoE experts + router logits + workspace for 512-token GPU FFN/MoE stage.
    const experts_raw = try gpu_arena.allocAligned(all_experts_bytes, 64);
    const exp_tbl_raw = try gpu_arena.allocAligned(exp_tbl_bytes, 64);
    const router_logits_raw = try gpu_arena.allocAligned(router_logits_bytes, 64);
    const moe_ws = try moe.MoeWorkspace.init(&gpu_arena, moe_ws_spec);

    const exp_tbl: []moe.ExpertWeights = @alignCast(std.mem.bytesAsSlice(moe.ExpertWeights, exp_tbl_raw[0 .. @as(usize, config.num_experts) * @sizeOf(moe.ExpertWeights)]));
    var rng: u64 = 0x517C_C1B7_2722_0A95;
    const moe_mid: f32 = switch (config.moe_bits) {
        .q2 => 1.5,
        .q3 => 3.5,
        .q4 => 7.5,
    };
    const moe_scale: f32 = 0.00390625;
    const sb_val: quant.ScaleBias = .{
        .scale = @floatCast(moe_scale),
        .bias = @floatCast(-moe_mid * moe_scale),
    };
    var e_cursor: usize = 0;
    for (0..config.num_experts) |e_idx| {
        const gw = e_cursor;
        e_cursor += amf.align64(gu_w_bytes);
        const gsb = e_cursor;
        e_cursor += amf.align64(gu_sb_bytes);
        const uw = e_cursor;
        e_cursor += amf.align64(gu_w_bytes);
        const usb = e_cursor;
        e_cursor += amf.align64(gu_sb_bytes);
        const dw = e_cursor;
        e_cursor += amf.align64(dn_w_bytes);
        const dsb = e_cursor;
        e_cursor += amf.align64(dn_sb_bytes);

        for ([_]struct { off: usize, len: usize }{
            .{ .off = gw, .len = gu_w_bytes },
            .{ .off = uw, .len = gu_w_bytes },
            .{ .off = dw, .len = dn_w_bytes },
        }) |seg| {
            const words: []u64 = @alignCast(std.mem.bytesAsSlice(u64, experts_raw[seg.off .. seg.off + (seg.len & ~@as(usize, 7))]));
            for (words) |*w| {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                w.* = rng;
            }
        }
        for ([_]struct { off: usize, len: usize }{
            .{ .off = gsb, .len = gu_sb_bytes },
            .{ .off = usb, .len = gu_sb_bytes },
            .{ .off = dsb, .len = dn_sb_bytes },
        }) |seg| {
            const sbs: []quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, experts_raw[seg.off .. seg.off + seg.len]));
            for (sbs) |*sb| sb.* = sb_val;
        }
        exp_tbl[e_idx] = .{
            .gate_qweight_offset = gw,
            .gate_sb_offset = gsb,
            .up_qweight_offset = uw,
            .up_sb_offset = usb,
            .down_qweight_offset = dw,
            .down_sb_offset = dsb,
        };
    }

    // Realistic routing across 512 tokens activating all experts with Zipf-like skew.
    const r_logits: []f32 = @alignCast(std.mem.bytesAsSlice(f32, router_logits_raw[0 .. @as(usize, s) * config.num_experts * @sizeOf(f32)]));
    for (r_logits) |*v| v.* = -8.0;
    for (0..s) |ti| {
        const row = r_logits[ti * config.num_experts ..][0..config.num_experts];
        for (0..config.top_k) |ki| {
            const eid = (ti * 7 + ki * 13) % config.num_experts;
            row[eid] = 6.0 - @as(f32, @floatFromInt(ki)) * 0.2;
        }
    }

    const gpu_buf = try metal.Buffer.wrap(device, gpu_arena.bytes());
    defer gpu_buf.close();
    try compute.makeResident(&.{gpu_buf});
    defer compute.removeResident(&.{gpu_buf}) catch |err| {
        std.debug.panic("removeResident failed: {}", .{err});
    };

    const x_alias0 = x_tiles[0].aliasBuffer() orelse return error.TileCreationFailed;
    const x_alias1 = x_tiles[1].aliasBuffer() orelse return error.TileCreationFailed;
    const out_alias0 = out_tiles[0].aliasBuffer() orelse return error.TileCreationFailed;
    const out_alias1 = out_tiles[1].aliasBuffer() orelse return error.TileCreationFailed;
    const q_alias = q_out_tile.aliasBuffer() orelse return error.TileCreationFailed;

    const wq_bind = metal.BufferBinding{ .buffer = gpu_buf, .offset = gpu_arena.offsetOf(wq_raw.ptr) };
    const sbq_bind = metal.BufferBinding{ .buffer = gpu_buf, .offset = gpu_arena.offsetOf(sbq_raw.ptr) };
    const wk_bind = metal.BufferBinding{ .buffer = gpu_buf, .offset = gpu_arena.offsetOf(wk_raw.ptr) };
    const sbk_bind = metal.BufferBinding{ .buffer = gpu_buf, .offset = gpu_arena.offsetOf(sbk_raw.ptr) };
    const wv_bind = metal.BufferBinding{ .buffer = gpu_buf, .offset = gpu_arena.offsetOf(wv_raw.ptr) };
    const sbv_bind = metal.BufferBinding{ .buffer = gpu_buf, .offset = gpu_arena.offsetOf(sbv_raw.ptr) };
    const wo_bind = metal.BufferBinding{ .buffer = gpu_buf, .offset = gpu_arena.offsetOf(wo_raw.ptr) };
    const sbo_bind = metal.BufferBinding{ .buffer = gpu_buf, .offset = gpu_arena.offsetOf(sbo_raw.ptr) };
    const gpu_q_bind = metal.BufferBinding{ .buffer = gpu_buf, .offset = gpu_arena.offsetOf(gpu_q_raw.ptr) };
    const gpu_ctx_bind = metal.BufferBinding{ .buffer = gpu_buf, .offset = gpu_arena.offsetOf(gpu_ctx_raw.ptr) };
    const gpu_k_bind = metal.BufferBinding{ .buffer = gpu_buf, .offset = gpu_arena.offsetOf(gpu_k_raw.ptr) };
    const gpu_v_bind = metal.BufferBinding{ .buffer = gpu_buf, .offset = gpu_arena.offsetOf(gpu_v_raw.ptr) };
    const norm_tmp_bind = metal.BufferBinding{ .buffer = gpu_buf, .offset = gpu_arena.offsetOf(gpu_norm_tmp_raw.ptr) };
    const out_tmp_bind = metal.BufferBinding{ .buffer = gpu_buf, .offset = gpu_arena.offsetOf(gpu_out_tmp_raw.ptr) };
    const masks_bind = metal.BufferBinding{ .buffer = gpu_buf, .offset = gpu_arena.offsetOf(masks_raw.ptr) };
    const kv_tbl_bind = metal.BufferBinding{ .buffer = gpu_buf, .offset = gpu_arena.offsetOf(kv_tbl_raw.ptr) };
    const gqa_scratch_bind = metal.BufferBinding{ .buffer = gpu_buf, .offset = gpu_arena.offsetOf(gqa_scratch_raw.ptr) };
    const norm_w_bind = metal.BufferBinding{ .buffer = gpu_buf, .offset = gpu_arena.offsetOf(norm_w_raw.ptr) };
    const experts_bind = metal.BufferBinding{ .buffer = gpu_buf, .offset = gpu_arena.offsetOf(experts_raw.ptr) };
    const exp_tbl_bind = metal.BufferBinding{ .buffer = gpu_buf, .offset = gpu_arena.offsetOf(exp_tbl_raw.ptr) };
    const r_logits_bind = metal.BufferBinding{ .buffer = gpu_buf, .offset = gpu_arena.offsetOf(router_logits_raw.ptr) };

    var accepted_tile_nodes: [attention.MAX_TREE_NODES]u16 = undefined;
    for (0..tile_m) |i| accepted_tile_nodes[i] = @intCast(i);

    // Helper closure to run GPU attention + O-proj (`include_qkv_and_commit == true` for GPU-only,
    // `false` for Split ANE QKV + GPU Split-KV Attention where Q/K/V already live in IOSurface tiles!).
    const GpuAttnRunner = struct {
        fn runOnce(
            comp: *metal.Compute,
            dq: dequant.DequantPipelines,
            ap: attention.AttentionPipelines,
            bits: quant.QuantBits,
            x_in_b: metal.BufferBinding,
            q_in_b: metal.BufferBinding,
            k_in_b: metal.BufferBinding,
            v_in_b: metal.BufferBinding,
            ctx_b: metal.BufferBinding,
            out_b: metal.BufferBinding,
            wq_b: metal.BufferBinding,
            sbq_b: metal.BufferBinding,
            wk_b: metal.BufferBinding,
            sbk_b: metal.BufferBinding,
            wv_b: metal.BufferBinding,
            sbv_b: metal.BufferBinding,
            wo_b: metal.BufferBinding,
            sbo_b: metal.BufferBinding,
            kv_tbl_b: metal.BufferBinding,
            masks_b: metal.BufferBinding,
            gqa_scr_b: metal.BufferBinding,
            k_tile_cur: metal.KvTile,
            v_tile_cur: metal.KvTile,
            acc_nodes: []const u16,
            g: AneAttentionGeometry,
            past_tokens: u32,
            tm: u32,
            ntiles: u32,
            include_qkv_and_commit: bool,
            timeout: u64,
        ) SpikeS6Error!void {
            var frame = try comp.beginFrame(timeout);
            errdefer frame.abort();
            var need_barrier = false;
            if (include_qkv_and_commit) {
                try dq.encodeGemm(&frame, bits, .{
                    .x = x_in_b,
                    .qweight = wq_b,
                    .scale_bias = sbq_b,
                    .y = q_in_b,
                    .m = g.chunk_tokens,
                    .n = g.qDim(),
                    .k = g.hidden_size,
                }, false);
                try dq.encodeGemm(&frame, bits, .{
                    .x = x_in_b,
                    .qweight = wk_b,
                    .scale_bias = sbk_b,
                    .y = k_in_b,
                    .m = g.chunk_tokens,
                    .n = g.kvDim(),
                    .k = g.hidden_size,
                }, false);
                try dq.encodeGemm(&frame, bits, .{
                    .x = x_in_b,
                    .qweight = wv_b,
                    .scale_bias = sbv_b,
                    .y = v_in_b,
                    .m = g.chunk_tokens,
                    .n = g.kvDim(),
                    .k = g.hidden_size,
                }, false);
                need_barrier = true;
            }

            const q_tile_bytes: u64 = @as(u64, tm) * g.qDim() * @sizeOf(f16);
            const kv_tile_bytes: u64 = @as(u64, tm) * g.kvDim() * @sizeOf(f16);

            for (0..ntiles) |st| {
                const base_p = past_tokens + @as(u32, @intCast(st)) * tm;
                const q_sub = metal.BufferBinding{ .buffer = q_in_b.buffer, .offset = q_in_b.offset + @as(u64, st) * q_tile_bytes };
                const ctx_sub = metal.BufferBinding{ .buffer = ctx_b.buffer, .offset = ctx_b.offset + @as(u64, st) * q_tile_bytes };
                const k_sub = metal.BufferBinding{ .buffer = k_in_b.buffer, .offset = k_in_b.offset + @as(u64, st) * kv_tile_bytes };
                const v_sub = metal.BufferBinding{ .buffer = v_in_b.buffer, .offset = v_in_b.offset + @as(u64, st) * kv_tile_bytes };

                try ap.encodeGqaTreeVerifyAttn(&frame, .{
                    .q = q_sub,
                    .kv_tiles = if (base_p > 0) kv_tbl_b else null,
                    .base_pos = base_p,
                    .k_tree = k_sub,
                    .v_tree = v_sub,
                    .ancestor_masks = masks_b,
                    .out = ctx_sub,
                    .scratch = if (base_p > 128) gqa_scr_b else null,
                    .max_splits = attention.MAX_SPLITS,
                    .num_nodes = tm,
                    .num_q_heads = g.num_q_heads,
                    .num_kv_heads = g.num_kv_heads,
                    .head_dim = g.head_dim,
                    .chunk_tokens = g.chunk_tokens,
                }, need_barrier or st > 0);

                if (include_qkv_and_commit) {
                    try ap.encodeKvTreeCommit(&frame, .{
                        .k_tree = k_sub,
                        .v_tree = v_sub,
                        .accepted_nodes = .{ .inline_u16 = acc_nodes },
                        .accepted_count = tm,
                        .k_tile0 = k_tile_cur,
                        .v_tile0 = v_tile_cur,
                        .num_kv_heads = g.num_kv_heads,
                        .head_dim = g.head_dim,
                        .base_pos = base_p,
                        .chunk_tokens = g.chunk_tokens,
                    }, true);
                }
            }

            try dq.encodeGemm(&frame, bits, .{
                .x = ctx_b,
                .qweight = wo_b,
                .scale_bias = sbo_b,
                .y = out_b,
                .m = g.chunk_tokens,
                .n = g.hidden_size,
                .k = g.qDim(),
            }, true);

            const sig = try frame.commit();
            try comp.wait(sig, timeout);
        }
    };

    // -----------------------------------------------------------------------
    // 4. Benchmark Static KV Buckets (B in {512, 1024, 2048, 4096}) & Precision
    // -----------------------------------------------------------------------
    var bucket_rows: [4]BucketBenchRow = undefined;
    var precision: PrecisionSummary = .{
        .eval_kv_len = 0,
        .sampled_rows = 0,
        .attn_max_abs_err = 0,
        .attn_rmse = 0,
        .kv_max_abs_err = 0,
        .kv_rmse = 0,
    };

    // Keep bucket 0 (`B = 512`) and bucket 1 (`B = 1024`) alive for the 2-chunk wavefront test!
    var wf_bucket0: ?AneAttentionBucket = null;
    var wf_bucket1: ?AneAttentionBucket = null;
    defer if (wf_bucket1) |b| b.close();
    defer if (wf_bucket0) |b| b.close();

    var wf_mask0_tile: ?metal.KvTile = null;
    var wf_mask1_tile: ?metal.KvTile = null;
    var wf_kp1_tile: ?metal.KvTile = null;
    var wf_vp1_tile: ?metal.KvTile = null;
    defer if (wf_vp1_tile) |t| t.close();
    defer if (wf_kp1_tile) |t| t.close();
    defer if (wf_mask1_tile) |t| t.close();
    defer if (wf_mask0_tile) |t| t.close();

    for (config.buckets, 0..) |b_len, bi| {
        const past_tokens = b_len - s;
        const cur_chunk_idx: usize = past_tokens / s;

        var bucket = try AneAttentionBucket.initFromWeights(
            allocator,
            geom,
            b_len,
            w_q,
            w_k,
            w_v,
            w_o,
            .cpu_and_neural_engine,
        );
        var keep_bucket = false;
        defer if (!keep_bucket) bucket.close();

        var mask_tile = try metal.KvTile.init(device, b_len, s, .{});
        var keep_mask = false;
        defer if (!keep_mask) mask_tile.close();

        const mask_host = try allocator.alloc(f16, @as(usize, s) * b_len);
        defer allocator.free(mask_host);
        fillCausalBucketMask(mask_host, s, b_len, past_tokens, s);
        {
            var mg = try mask_tile.lock(false);
            errdefer mg.release();
            @memcpy(mg.f16Slice()[0..mask_host.len], mask_host);
            try mg.unlock();
        }

        var kp_tile_opt: ?metal.KvTile = null;
        var vp_tile_opt: ?metal.KvTile = null;
        var keep_kv_past = false;
        defer if (!keep_kv_past) {
            if (vp_tile_opt) |t| t.close();
            if (kp_tile_opt) |t| t.close();
        };

        const kp_host = try allocator.alloc(f16, @as(usize, kvh) * past_tokens * d);
        defer allocator.free(kp_host);
        const vp_host = try allocator.alloc(f16, @as(usize, kvh) * past_tokens * d);
        defer allocator.free(vp_host);

        if (past_tokens > 0) {
            fillDeterministicF16(kp_host, 500 + bi, 0.5);
            fillDeterministicF16(vp_host, 600 + bi, 0.5);
            kp_tile_opt = try metal.KvTile.init(device, d, kvh * past_tokens, .{});
            vp_tile_opt = try metal.KvTile.init(device, d, kvh * past_tokens, .{});
            {
                var g = try kp_tile_opt.?.lock(false);
                errdefer g.release();
                @memcpy(g.f16Slice()[0..kp_host.len], kp_host);
                try g.unlock();
            }
            {
                var g = try vp_tile_opt.?.lock(false);
                errdefer g.release();
                @memcpy(g.f16Slice()[0..vp_host.len], vp_host);
                try g.unlock();
            }
        }

        const bound = try bucket.bindTiles(
            x_tiles[0],
            kp_tile_opt,
            vp_tile_opt,
            mask_tile,
            out_tiles[0],
            k_chunk_tiles[cur_chunk_idx],
            v_chunk_tiles[cur_chunk_idx],
        );
        defer bound.close();

        var zc = false;
        for (0..config.warmup_iters) |_| {
            zc = (try bucket.run(bound)).allOutputsZeroCopy();
        }
        const bt0 = try vmstat.nowNs();
        for (0..config.timed_iters) |_| {
            zc = (try bucket.run(bound)).allOutputsZeroCopy() and zc;
        }
        const bt1 = try vmstat.nowNs();
        const bucketed_ms = (@as(f64, @floatFromInt(bt1 -| bt0)) / @as(f64, @floatFromInt(config.timed_iters))) / 1_000_000.0;

        // Evaluate fp16 numerical precision against CPU fp64 on bucket 1 (`B = 1024`,
        // exercising BOTH `k_past`/`v_past` concat AND causal mask!).
        if (bi == 1) {
            var og = try out_tiles[0].lock(true);
            errdefer og.release();
            var kg = try k_chunk_tiles[cur_chunk_idx].lock(true);
            errdefer kg.release();
            var vg = try v_chunk_tiles[cur_chunk_idx].lock(true);
            errdefer vg.release();
            precision = try evalBucketPrecisionCpu(
                allocator,
                geom,
                b_len,
                past_tokens,
                config.precision_sample_rows,
                x_host,
                kp_host,
                vp_host,
                mask_host,
                w_q,
                w_k,
                w_v,
                w_o,
                og.f16Slice(),
                kg.f16Slice(),
                vg.f16Slice(),
            );
            try vg.unlock();
            try kg.unlock();
            try og.unlock();
        }

        // Measure Split-GPU-Attention tail (paired with ANE QKV projection).
        for (0..config.warmup_iters) |_| {
            try GpuAttnRunner.runOnce(
                &compute,
                deq_pipes,
                attn_pipes,
                config.moe_bits,
                .{ .buffer = x_alias0, .offset = 0 },
                .{ .buffer = q_alias, .offset = 0 },
                gpu_k_bind,
                gpu_v_bind,
                gpu_ctx_bind,
                out_tmp_bind,
                wq_bind,
                sbq_bind,
                wk_bind,
                sbk_bind,
                wv_bind,
                sbv_bind,
                wo_bind,
                sbo_bind,
                kv_tbl_bind,
                masks_bind,
                gqa_scratch_bind,
                k_chunk_tiles[cur_chunk_idx],
                v_chunk_tiles[cur_chunk_idx],
                accepted_tile_nodes[0..tile_m],
                geom,
                past_tokens,
                tile_m,
                sub_tiles,
                false,
                config.timeout_ms,
            );
        }
        const sg0 = try vmstat.nowNs();
        for (0..config.timed_iters) |_| {
            try GpuAttnRunner.runOnce(
                &compute,
                deq_pipes,
                attn_pipes,
                config.moe_bits,
                .{ .buffer = x_alias0, .offset = 0 },
                .{ .buffer = q_alias, .offset = 0 },
                gpu_k_bind,
                gpu_v_bind,
                gpu_ctx_bind,
                out_tmp_bind,
                wq_bind,
                sbq_bind,
                wk_bind,
                sbk_bind,
                wv_bind,
                sbv_bind,
                wo_bind,
                sbo_bind,
                kv_tbl_bind,
                masks_bind,
                gqa_scratch_bind,
                k_chunk_tiles[cur_chunk_idx],
                v_chunk_tiles[cur_chunk_idx],
                accepted_tile_nodes[0..tile_m],
                geom,
                past_tokens,
                tile_m,
                sub_tiles,
                false,
                config.timeout_ms,
            );
        }
        const sg1 = try vmstat.nowNs();
        const split_gpu_ms = (@as(f64, @floatFromInt(sg1 -| sg0)) / @as(f64, @floatFromInt(config.timed_iters))) / 1_000_000.0;

        // Measure GPU-only full 512-token chunked attention (QKV + KV-commit + GQA + O-proj).
        for (0..config.warmup_iters) |_| {
            try GpuAttnRunner.runOnce(
                &compute,
                deq_pipes,
                attn_pipes,
                config.moe_bits,
                .{ .buffer = x_alias0, .offset = 0 },
                gpu_q_bind,
                gpu_k_bind,
                gpu_v_bind,
                gpu_ctx_bind,
                out_tmp_bind,
                wq_bind,
                sbq_bind,
                wk_bind,
                sbk_bind,
                wv_bind,
                sbv_bind,
                wo_bind,
                sbo_bind,
                kv_tbl_bind,
                masks_bind,
                gqa_scratch_bind,
                k_chunk_tiles[cur_chunk_idx],
                v_chunk_tiles[cur_chunk_idx],
                accepted_tile_nodes[0..tile_m],
                geom,
                past_tokens,
                tile_m,
                sub_tiles,
                true,
                config.timeout_ms,
            );
        }
        const go0 = try vmstat.nowNs();
        for (0..config.timed_iters) |_| {
            try GpuAttnRunner.runOnce(
                &compute,
                deq_pipes,
                attn_pipes,
                config.moe_bits,
                .{ .buffer = x_alias0, .offset = 0 },
                gpu_q_bind,
                gpu_k_bind,
                gpu_v_bind,
                gpu_ctx_bind,
                out_tmp_bind,
                wq_bind,
                sbq_bind,
                wk_bind,
                sbk_bind,
                wv_bind,
                sbv_bind,
                wo_bind,
                sbo_bind,
                kv_tbl_bind,
                masks_bind,
                gqa_scratch_bind,
                k_chunk_tiles[cur_chunk_idx],
                v_chunk_tiles[cur_chunk_idx],
                accepted_tile_nodes[0..tile_m],
                geom,
                past_tokens,
                tile_m,
                sub_tiles,
                true,
                config.timeout_ms,
            );
        }
        const go1 = try vmstat.nowNs();
        const gpu_only_ms = (@as(f64, @floatFromInt(go1 -| go0)) / @as(f64, @floatFromInt(config.timed_iters))) / 1_000_000.0;

        bucket_rows[bi] = .{
            .kv_len = b_len,
            .compile_ms = bucket.compileMs(),
            .total_layers = bucket.model_info.totalLayers(),
            .ane_preferred_layers = bucket.model_info.anePreferredLayers(),
            .uses_ane = bucket.model_info.usesNeuralEngine(),
            .all_outputs_zero_copy = zc,
            .bucketed_ane_ms = bucketed_ms,
            .split_ane_qkv_ms = qkv_ane_ms,
            .split_gpu_attn_ms = split_gpu_ms,
            .split_total_ms = qkv_ane_ms + split_gpu_ms,
            .gpu_only_ms = gpu_only_ms,
        };

        if (bi == 0) {
            wf_bucket0 = bucket;
            wf_mask0_tile = mask_tile;
            keep_bucket = true;
            keep_mask = true;
        } else if (bi == 1) {
            wf_bucket1 = bucket;
            wf_mask1_tile = mask_tile;
            wf_kp1_tile = kp_tile_opt;
            wf_vp1_tile = vp_tile_opt;
            keep_bucket = true;
            keep_mask = true;
            keep_kv_past = true;
        }
    }

    // -----------------------------------------------------------------------
    // 5. Real 2-Chunk Hybrid Wavefront Overlap Benchmark (A1)
    // -----------------------------------------------------------------------
    const wf_bound0 = try wf_bucket0.?.bindTiles(
        x_tiles[0],
        null,
        null,
        wf_mask0_tile.?,
        out_tiles[0],
        k_chunk_tiles[0],
        v_chunk_tiles[0],
    );
    defer wf_bound0.close();

    const wf_bound1 = try wf_bucket1.?.bindTiles(
        x_tiles[1],
        wf_kp1_tile.?,
        wf_vp1_tile.?,
        wf_mask1_tile.?,
        out_tiles[1],
        k_chunk_tiles[1],
        v_chunk_tiles[1],
    );
    defer wf_bound1.close();

    const WfCtx = struct {
        b0: AneAttentionBucket,
        b1: AneAttentionBucket,
        p0: AneAttentionBucket.BoundPlan,
        p1: AneAttentionBucket.BoundPlan,
        comp: *metal.Compute,
        ap: attention.AttentionPipelines,
        mp: moe.MoePipelines,
        bits: quant.QuantBits,
        x_binds: [2]metal.BufferBinding,
        out_binds: [2]metal.BufferBinding,
        norm_w: metal.BufferBinding,
        norm_tmp: metal.BufferBinding,
        r_logits: metal.BufferBinding,
        exp_base: metal.BufferBinding,
        exp_tbl: metal.BufferBinding,
        ws: *const moe.MoeWorkspace,
        ws_buf: metal.Buffer,
        cfg: SpikeS6Config,
        all_zc: std.atomic.Value(bool) = .init(true),

        fn aneStep(self: *@This(), task: StepTask) anyerror!void {
            const info = if (task.slot == 0)
                try self.b0.run(self.p0)
            else
                try self.b1.run(self.p1);
            if (!info.allOutputsZeroCopy()) {
                self.all_zc.store(false, .release);
            }
        }

        fn gpuFfnStepDirect(self: *@This(), task: StepTask) SpikeS6Error!void {
            // GPU half (G6): post-attention RMSNorm + 512-token MoE SwiGLU + residual add
            // reading ANE's zero-copy IOSurface `attn_out` and writing into `x_in[slot]`!
            const attn_in = self.out_binds[task.slot];
            const x_dst = self.x_binds[task.slot];
            var frame = try self.comp.beginFrame(self.cfg.timeout_ms);
            errdefer frame.abort();
            try self.ap.encodeRmsNorm(&frame, .{
                .x = attn_in,
                .w = self.norm_w,
                .y = self.norm_tmp,
                .num_rows = self.cfg.chunk_tokens,
                .hidden_dim = self.cfg.hidden_size,
                .eps = 1e-6,
            }, false);
            try self.mp.encodeMoELayer(&frame, self.bits, .{
                .x = self.norm_tmp,
                .router = .{ .logits = .{ .binding = self.r_logits, .dtype = .f32 } },
                .weights_base = self.exp_base,
                .expert_table = self.exp_tbl,
                .workspace = self.ws,
                .workspace_buf = self.ws_buf,
                .residual = attn_in,
                .out = x_dst,
                .m = self.cfg.chunk_tokens,
                .num_experts = self.cfg.num_experts,
                .top_k = self.cfg.top_k,
                .hidden_size = self.cfg.hidden_size,
                .intermediate_size = self.cfg.moe_intermediate_size,
                .norm_topk_prob = true,
            }, true);
            const sig = try frame.commit();
            try self.comp.wait(sig, self.cfg.timeout_ms);
        }

        fn gpuFfnStep(self: *@This(), task: StepTask) anyerror!void {
            return self.gpuFfnStepDirect(task);
        }
    };

    var wf_ctx: WfCtx = .{
        .b0 = wf_bucket0.?,
        .b1 = wf_bucket1.?,
        .p0 = wf_bound0,
        .p1 = wf_bound1,
        .comp = &compute,
        .ap = attn_pipes,
        .mp = moe_pipes,
        .bits = config.moe_bits,
        .x_binds = .{
            .{ .buffer = x_alias0, .offset = 0 },
            .{ .buffer = x_alias1, .offset = 0 },
        },
        .out_binds = .{
            .{ .buffer = out_alias0, .offset = 0 },
            .{ .buffer = out_alias1, .offset = 0 },
        },
        .norm_w = norm_w_bind,
        .norm_tmp = norm_tmp_bind,
        .r_logits = r_logits_bind,
        .exp_base = experts_bind,
        .exp_tbl = exp_tbl_bind,
        .ws = &moe_ws,
        .ws_buf = gpu_buf,
        .cfg = config,
    };

    // Solo GPU FFN/MoE calibration (for exact `t_F` and GPU-only full-layer baseline).
    for (0..config.warmup_iters) |_| {
        try wf_ctx.gpuFfnStepDirect(.{ .layer = 0, .chunk = 0, .slot = 0 });
    }
    const ffn_t0 = try vmstat.nowNs();
    for (0..config.timed_iters) |_| {
        try wf_ctx.gpuFfnStepDirect(.{ .layer = 0, .chunk = 0, .slot = 0 });
    }
    const ffn_t1 = try vmstat.nowNs();
    const solo_ffn_ms = (@as(f64, @floatFromInt(ffn_t1 -| ffn_t0)) / @as(f64, @floatFromInt(config.timed_iters))) / 1_000_000.0;

    // Reset x_tiles to clean finite inputs before timed wavefront.
    for (0..2) |slot| {
        var g = try x_tiles[slot].lock(false);
        errdefer g.release();
        @memcpy(g.f16Slice()[0..x_host.len], x_host);
        try g.unlock();
    }

    var wf_engine = try WavefrontPrefill.init(allocator);
    defer wf_engine.deinit(allocator);

    // Warmup wavefront (1 layer x 2 chunks).
    _ = wf_engine.runWavefrontTyped(WfCtx, config.wavefront_chunks, 1, &wf_ctx, WfCtx.aneStep, WfCtx.gpuFfnStep) catch return error.ComputeFailed;

    // Reset x_tiles again so timed wavefront starts from well-scaled activations.
    for (0..2) |slot| {
        var g = try x_tiles[slot].lock(false);
        errdefer g.release();
        @memcpy(g.f16Slice()[0..x_host.len], x_host);
        try g.unlock();
    }
    wf_ctx.all_zc.store(true, .release);

    const wf_stats = wf_engine.runWavefrontTyped(
        WfCtx,
        config.wavefront_chunks,
        config.wavefront_layers,
        &wf_ctx,
        WfCtx.aneStep,
        WfCtx.gpuFfnStep,
    ) catch return error.ComputeFailed;

    // Verify outputs on both double-buffered tiles remain finite.
    for (0..2) |slot| {
        var g = try x_tiles[slot].lock(true);
        errdefer g.release();
        const sl = g.f16Slice();
        if (!std.math.isFinite(sl[0]) or !std.math.isFinite(sl[x_host.len - 1])) {
            try g.unlock();
            return error.NonFiniteOutput;
        }
        try g.unlock();
    }

    // Derive exact A3 recommendation from measured bucket & overlap telemetry:
    // Count how many buckets favor bucketed full ANE (particularly with wavefront overlap
    // where bucketed ANE hides completely behind GPU FFN whenever t_A <= t_F!).
    var max_bucket_worth_ane: u32 = config.buckets[0];
    var bucketed_wins_count: u32 = 0;
    for (bucket_rows) |row| {
        // Under wavefront overlap, full ANE wins whenever max(t_A, t_F) < t_gpu_attn + t_F,
        // and strictly beats split when bucketed_ane_ms <= split_total_ms or t_A <= solo_ffn_ms.
        const wf_bucket_cost = @max(row.bucketed_ane_ms, solo_ffn_ms);
        const gpu_full_cost = row.gpu_only_ms + solo_ffn_ms;
        if (wf_bucket_cost < gpu_full_cost) {
            max_bucket_worth_ane = row.kv_len;
        }
        if (row.bucketed_ane_ms <= row.split_total_ms) {
            bucketed_wins_count += 1;
        }
    }

    const avg_gpu_attn_ms = (bucket_rows[0].gpu_only_ms + bucket_rows[1].gpu_only_ms) * 0.5;
    const avg_ane_ms = wf_stats.aneTotalMs() / @as(f64, @floatFromInt(wf_stats.total_steps));
    const avg_ffn_ms = wf_stats.gpuTotalMs() / @as(f64, @floatFromInt(wf_stats.total_steps));
    const seq_ms_per_chunk = wf_stats.sequentialSumMs() / @as(f64, @floatFromInt(config.wavefront_chunks));
    const wf_ms_per_chunk = wf_stats.msPerChunk();
    const gpu_only_ms_per_chunk = @as(f64, @floatFromInt(config.wavefront_layers)) * (avg_gpu_attn_ms + solo_ffn_ms);

    const wf_tok_s = wf_stats.tokensPerSec(s);
    const seq_tok_s = if (seq_ms_per_chunk > 0.0) (@as(f64, @floatFromInt(s)) * 1000.0) / seq_ms_per_chunk else 0.0;
    const gpu_tok_s = if (gpu_only_ms_per_chunk > 0.0) (@as(f64, @floatFromInt(s)) * 1000.0) / gpu_only_ms_per_chunk else 0.0;

    const rec_strategy: StaticShapeStrategy = if (wf_ms_per_chunk < gpu_only_ms_per_chunk)
        (if (bucketed_wins_count >= 2 or max_bucket_worth_ane >= config.buckets[1])
            .bucketed_full_ane
        else
            .split_ane_qkv_gpu_attn)
    else if (bucket_rows[0].split_total_ms < bucket_rows[0].gpu_only_ms)
        .split_ane_qkv_gpu_attn
    else
        .gpu_only;

    const vm_after = try vmstat.Snapshot.capture();
    try vmstat.checkNoSwap(vm_before, vm_after);

    return .{
        .config = config,
        .ane_core_count = dev_info.aneCoreCount(),
        .qkv_compile_ms = qkv_block.compileMs(),
        .qkv_total_layers = qkv_block.model_info.totalLayers(),
        .qkv_ane_preferred_layers = qkv_block.model_info.anePreferredLayers(),
        .qkv_uses_ane = qkv_block.model_info.usesNeuralEngine(),
        .qkv_zero_copy = qkv_zero_copy,
        .buckets = bucket_rows,
        .precision = precision,
        .wavefront = .{
            .num_chunks = config.wavefront_chunks,
            .num_layers = config.wavefront_layers,
            .chunk_tokens = s,
            .ane_strategy = .bucketed_full_ane,
            .ane_thread_qos = wf_stats.ane_thread_qos,
            .overlapped_steps = wf_stats.overlapped_steps,
            .total_steps = wf_stats.total_steps,
            .solo_ane_ms_per_layer_chunk = avg_ane_ms,
            .solo_gpu_ffn_ms_per_layer_chunk = avg_ffn_ms,
            .solo_gpu_attn_ms_per_layer_chunk = avg_gpu_attn_ms,
            .sequential_ms_per_chunk = seq_ms_per_chunk,
            .wavefront_ms_per_chunk = wf_ms_per_chunk,
            .gpu_only_ms_per_chunk = gpu_only_ms_per_chunk,
            .overlap_efficiency = wf_stats.overlapEfficiency(),
            .wavefront_tok_per_s = wf_tok_s,
            .sequential_tok_per_s = seq_tok_s,
            .gpu_only_tok_per_s = gpu_tok_s,
            .speedup_vs_sequential = if (wf_ms_per_chunk > 0.0) seq_ms_per_chunk / wf_ms_per_chunk else 0.0,
            .speedup_vs_gpu_only = if (wf_ms_per_chunk > 0.0) gpu_only_ms_per_chunk / wf_ms_per_chunk else 0.0,
            .all_outputs_zero_copy = wf_ctx.all_zc.load(.acquire),
        },
        .recommended_strategy = rec_strategy,
        .recommended_max_bucket = max_bucket_worth_ane,
        .vm_before = vm_before,
        .vm_after = vm_after,
    };
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

test "selectBucketLen and fillCausalBucketMask enforce exact causal and padding boundaries" {
    try std.testing.expectEqual(@as(?u32, 512), selectBucketLen(1, &DEFAULT_KV_BUCKETS));
    try std.testing.expectEqual(@as(?u32, 512), selectBucketLen(512, &DEFAULT_KV_BUCKETS));
    try std.testing.expectEqual(@as(?u32, 1024), selectBucketLen(513, &DEFAULT_KV_BUCKETS));
    try std.testing.expectEqual(@as(?u32, 2048), selectBucketLen(1536, &DEFAULT_KV_BUCKETS));
    try std.testing.expectEqual(@as(?u32, 4096), selectBucketLen(4096, &DEFAULT_KV_BUCKETS));
    try std.testing.expectEqual(@as(?u32, null), selectBucketLen(4097, &DEFAULT_KV_BUCKETS));

    // chunk=4, bucket=8 (past_cap=4), valid_past=2, valid_chunk=3
    var mask: [4 * 8]f16 = undefined;
    fillCausalBucketMask(&mask, 4, 8, 2, 3);
    // Row 0 (valid): cols 0,1 = 0; cols 2,3 = -10000; col 4 = 0; cols 5,6,7 = -10000
    try std.testing.expectEqual(@as(f16, @floatCast(0.0)), mask[0]);
    try std.testing.expectEqual(@as(f16, @floatCast(0.0)), mask[1]);
    try std.testing.expectEqual(MASK_NEG_INF_F16, mask[2]);
    try std.testing.expectEqual(MASK_NEG_INF_F16, mask[3]);
    try std.testing.expectEqual(@as(f16, @floatCast(0.0)), mask[4]);
    try std.testing.expectEqual(MASK_NEG_INF_F16, mask[5]);
    // Row 2 (valid): cols 4,5,6 = 0; col 7 = -10000
    try std.testing.expectEqual(@as(f16, @floatCast(0.0)), mask[2 * 8 + 6]);
    try std.testing.expectEqual(MASK_NEG_INF_F16, mask[2 * 8 + 7]);
    // Row 3 (padded query): only self-slot col 7 is 0 so softmax stays finite
    try std.testing.expectEqual(MASK_NEG_INF_F16, mask[3 * 8 + 0]);
    try std.testing.expectEqual(@as(f16, @floatCast(0.0)), mask[3 * 8 + 7]);
}

test "WavefrontPrefill enforces every (layer, chunk) dependency, disjoint double-buffer slots, USER_INITIATED QoS, and zero hot-path allocs" {
    const gpa = std.testing.allocator;
    var wf = try WavefrontPrefill.init(gpa);
    defer wf.deinit(gpa);

    try std.testing.expectEqual(qos.QosClass.user_initiated, wf.aneThreadQos());

    // Verify all chunk counts (1, 2, 3, 4) x 4 layers!
    for ([_]u32{ 1, 2, 3, 4 }) |n_chunks| {
        const n_layers: u32 = 4;
        const TestCtx = struct {
            ane_done: [4][4]std.atomic.Value(u32),
            gpu_done: [4][4]std.atomic.Value(u32),
            slot_in_use: [2]std.atomic.Value(u32),
            concurrent_hits: std.atomic.Value(u32),
            slot_collisions: std.atomic.Value(u32),

            fn aneStep(self: *@This(), task: StepTask) anyerror!void {
                // Verify dependencies BEFORE running A(l, c):
                // 1) F(l-1, c) must be done if l > 0.
                if (task.layer > 0) {
                    if (self.gpu_done[task.layer - 1][task.chunk].load(.acquire) != 1) {
                        return error.DependencyViolation;
                    }
                }
                // 2) A(l, c-1) must be done if c > 0.
                if (task.chunk > 0) {
                    if (self.ane_done[task.layer][task.chunk - 1].load(.acquire) != 1) {
                        return error.DependencyViolation;
                    }
                }
                // Verify disjoint slot ownership.
                if (self.slot_in_use[task.slot].fetchAdd(1, .acq_rel) != 0) {
                    _ = self.slot_collisions.fetchAdd(1, .monotonic);
                }
                // Check if sibling slot is concurrently active on GPU!
                if (self.slot_in_use[1 - task.slot].load(.acquire) != 0) {
                    _ = self.concurrent_hits.fetchAdd(1, .monotonic);
                }
                // Brief deterministic CPU work so overlap is observable.
                var spin: u64 = 0;
                for (0..2000) |i| spin +%= i *% 7;
                std.mem.doNotOptimizeAway(spin);

                _ = self.slot_in_use[task.slot].fetchSub(1, .acq_rel);
                self.ane_done[task.layer][task.chunk].store(1, .release);
            }

            fn gpuStep(self: *@This(), task: StepTask) anyerror!void {
                // Verify dependency: A(l, c) must already be done!
                if (self.ane_done[task.layer][task.chunk].load(.acquire) != 1) {
                    return error.DependencyViolation;
                }
                if (self.slot_in_use[task.slot].fetchAdd(1, .acq_rel) != 0) {
                    _ = self.slot_collisions.fetchAdd(1, .monotonic);
                }
                if (self.slot_in_use[1 - task.slot].load(.acquire) != 0) {
                    _ = self.concurrent_hits.fetchAdd(1, .monotonic);
                }
                var spin: u64 = 0;
                for (0..2000) |i| spin +%= i *% 11;
                std.mem.doNotOptimizeAway(spin);

                _ = self.slot_in_use[task.slot].fetchSub(1, .acq_rel);
                self.gpu_done[task.layer][task.chunk].store(1, .release);
            }
        };

        var ctx: TestCtx = .{
            .ane_done = undefined,
            .gpu_done = undefined,
            .slot_in_use = .{ .init(0), .init(0) },
            .concurrent_hits = .init(0),
            .slot_collisions = .init(0),
        };
        for (0..4) |l| {
            for (0..4) |c| {
                ctx.ane_done[l][c] = .init(0);
                ctx.gpu_done[l][c] = .init(0);
            }
        }

        const stats = try wf.runWavefrontTyped(TestCtx, n_chunks, n_layers, &ctx, TestCtx.aneStep, TestCtx.gpuStep);
        try std.testing.expectEqual(n_chunks * n_layers, stats.total_steps);
        try std.testing.expectEqual(@as(u32, 0), ctx.slot_collisions.load(.acquire));

        for (0..n_layers) |l| {
            for (0..n_chunks) |c| {
                try std.testing.expectEqual(@as(u32, 1), ctx.ane_done[l][c].load(.acquire));
                try std.testing.expectEqual(@as(u32, 1), ctx.gpu_done[l][c].load(.acquire));
            }
        }
        if (n_chunks >= 2) {
            // Even chunks: every step except prologue/epilogue overlaps (`total_steps - 1`).
            if ((n_chunks & 1) == 0) {
                try std.testing.expectEqual(n_chunks * n_layers - 1, stats.overlapped_steps);
            } else {
                try std.testing.expect(stats.overlapped_steps >= 2 * n_layers);
            }
        } else {
            try std.testing.expectEqual(@as(u32, 0), stats.overlapped_steps);
        }
    }

    // Verify clean error propagation & draining without leaking or hanging.
    const ErrCtx = struct {
        fn okAne(_: *@This(), _: StepTask) anyerror!void {}
        fn failGpu(_: *@This(), task: StepTask) anyerror!void {
            if (task.layer == 1 and task.chunk == 0) return error.ComputeFailed;
        }
    };
    var ectx: ErrCtx = .{};
    try std.testing.expectError(
        error.ComputeFailed,
        wf.runWavefrontTyped(ErrCtx, 2, 3, &ectx, ErrCtx.okAne, ErrCtx.failGpu),
    );
    // Engine remains healthy after an error!
    var ok_ctx: struct {
        count: std.atomic.Value(u32) = .init(0),
        fn step(self: *@This(), _: StepTask) anyerror!void {
            _ = self.count.fetchAdd(1, .monotonic);
        }
    } = .{};
    _ = try wf.runWavefrontTyped(@TypeOf(ok_ctx), 2, 2, &ok_ctx, @TypeOf(ok_ctx).step, @TypeOf(ok_ctx).step);
    try std.testing.expectEqual(@as(u32, 8), ok_ctx.count.load(.acquire));
}

test "Spike S6 smoke benchmark verifies ANE buckets, fp16 precision, 2-chunk wavefront overlap, and zero swap" {
    const gpa = std.testing.allocator;
    const report = try runSpikeS6(gpa, .{
        .hidden_size = 128,
        .num_q_heads = 4,
        .num_kv_heads = 2,
        .head_dim = 64,
        .chunk_tokens = 32,
        .buckets = .{ 32, 64, 128, 256 },
        .moe_intermediate_size = 128,
        .num_experts = 8,
        .top_k = 2,
        .moe_bits = .q3,
        .wavefront_chunks = 2,
        .wavefront_layers = 2,
        .gpu_attn_tile = 32,
        .precision_sample_rows = 8,
        .warmup_iters = 1,
        .timed_iters = 2,
        .timeout_ms = 10_000,
    });

    if (report.ane_core_count > 0) {
        try std.testing.expect(report.allBucketsUseAne());
    }
    try std.testing.expect(report.allBucketsZeroCopy());
    for (report.buckets) |row| {
        try std.testing.expect(row.compile_ms > 0.0);
        try std.testing.expect(row.bucketed_ane_ms > 0.0);
        try std.testing.expect(row.split_total_ms > 0.0);
        try std.testing.expect(row.gpu_only_ms > 0.0);
    }
    try std.testing.expect(report.precision.attn_max_abs_err < 0.05);
    try std.testing.expect(report.precision.attn_rmse < 0.02);
    try std.testing.expect(report.precision.kv_max_abs_err < 0.05);
    try std.testing.expectEqual(qos.QosClass.user_initiated, report.wavefront.ane_thread_qos);
    try std.testing.expectEqual(@as(u32, 3), report.wavefront.overlapped_steps);
    try std.testing.expect(report.wavefront.wavefront_ms_per_chunk > 0.0);
    try vmstat.checkNoSwap(report.vm_before, report.vm_after);

    var aw: std.Io.Writer.Allocating = .init(gpa);
    defer aw.deinit();
    try report.write(&aw.writer);
    const rendered = aw.written();
    try std.testing.expect(std.mem.indexOf(u8, rendered, "Spike S6") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "s6.a3.recommended_strategy") != null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "s6.wf.overlap_efficiency") != null);
}
