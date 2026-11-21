//! ANE cross-layer MoE expert predictor (`A2`, `docs/architecture/experts-and-storage.md`).
//!
//! Predicts upcoming layers' (`l + 1 .. l + lookahead`) MoE expert activations
//! on the Apple Neural Engine over zero-copy IOSurface `KvTile`s (`ADR-0004`),
//! feeding both:
//! - asynchronous NVMe cold-expert prefetch (`C3`–`C5`, hiding SSD latency with
//!   cross-layer lead time), and
//! - expert-byte-budget speculative tree pruning (`D7`, `ADR-0009`).
//!
//! Supports three startup construction modes (zero Python at runtime):
//! 1. In-memory compilation from target model router gate weights (`fp16`)
//!    either for a single lookahead window or across all `num_layers` via
//!    overlapped layer blocks (so every `current_layer` executes in one ANE call),
//! 2. In-memory compilation of a 2-layer bottleneck MLP (`hidden -> bn -> act -> out`),
//! 3. Loading an offline `.mlpackage` / `.mlmodelc` bundle exported by `W7`.
//!
//! Steady-state `predictAhead`, `predictPerTokenBitsets`, and optional async
//! `USER_INITIATED` background dispatch over lock-free `SpscRing`s perform
//! zero dynamic heap allocation in our Zig code.

const std = @import("std");
const metal = @import("../bridge/metal.zig");
const ring = @import("../runtime/ring.zig");
const qos = @import("../platform/qos.zig");
const vmstat = @import("../platform/vmstat.zig");
const tree_mod = @import("../spec/tree.zig");

/// Maximum MoE experts per layer supported by `ExpertBitset` (`[4]u64`).
pub const MAX_EXPERTS: u32 = tree_mod.ExpertBitset.MAX_EXPERTS;

/// Maximum cross-layer lookahead horizon supported on the stack.
pub const MAX_LOOKAHEAD: u32 = 8;

/// Maximum tokens per batch supported by stack unpacking helpers.
pub const MAX_BATCH_TOKENS: u32 = 64;

/// Default block stride for full-model overlapped router banks (48 layers -> 6 ANE blocks).
pub const DEFAULT_BLOCK_STRIDE: u32 = 8;

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

fn alignUp32(v: u32) u32 {
    return ((v + metal.KV_TILE_ROW_ALIGN_F16 - 1) / metal.KV_TILE_ROW_ALIGN_F16) * metal.KV_TILE_ROW_ALIGN_F16;
}

/// Ranked cold-expert prefetch candidate emitted by `AneExpertPredictor.predictAhead`.
pub const PrefetchCandidate = struct {
    layer: u16,
    expert: u16,
    /// Aggregated cross-token routing confidence score (higher = prefetch sooner).
    score: f32,
};

pub const PredictorConfig = struct {
    num_layers: u32,
    num_experts: u32,
    hidden_size: u32,
    /// Number of upcoming layers predicted ahead (`l + 1 .. l + lookahead_layers`).
    lookahead_layers: u32 = 2,
    /// Static ANE input batch rows `M_padded >= 1` (`1` for decode, `32`/`64` for tree verify).
    max_tokens: u32 = 32,
    /// Per-token top-K experts marked active in each predicted `ExpertBitset`.
    top_k: u32 = 8,
    /// Optional bottleneck width for 2-layer MLP predictors (`0` = 1-layer linear).
    bottleneck_dim: u32 = 0,
    activation: metal.CoreMlRouterActivation = .silu,
    compute_units: metal.AneComputeUnits = .cpu_and_neural_engine,
    /// Spawn a dedicated `USER_INITIATED` ANE worker thread communicating through
    /// preallocated lock-free `SpscRing`s so prediction overlaps GPU layer work.
    enable_async_thread: bool = false,
    /// Layer stride for multi-block full-model router banks.
    block_stride: u32 = DEFAULT_BLOCK_STRIDE,
    /// Optional explicit CoreML input/output feature names for `loadFromPath`.
    /// When `null`, discovered automatically via `AneModel.ioName`.
    input_name: ?[:0]const u8 = null,
    output_name: ?[:0]const u8 = null,

    pub fn validate(self: PredictorConfig) bool {
        return self.num_layers > 0 and
            self.num_experts > 0 and self.num_experts <= MAX_EXPERTS and
            self.hidden_size > 0 and (self.hidden_size % metal.KV_TILE_ROW_ALIGN_F16) == 0 and
            self.lookahead_layers > 0 and self.lookahead_layers <= MAX_LOOKAHEAD and
            self.max_tokens > 0 and self.max_tokens <= MAX_BATCH_TOKENS and
            self.top_k > 0 and self.top_k <= self.num_experts and self.top_k <= 16 and
            self.block_stride > 0;
    }
};

pub const PredictResult = struct {
    current_layer: u32,
    predicted_layers: u32,
    prefetch_count: u32,
    all_outputs_zero_copy: bool,
    predict_ns: u64,
};

pub const AsyncPredictResult = struct {
    req_id: u64,
    result: PredictResult,
};

const ModelBlock = struct {
    /// First target layer represented at output slice 0 of this block.
    start_target_layer: u32,
    /// Number of consecutive target layers covered by this block.
    window_layers: u32,
    /// Padded output column count (`alignUp32(window_layers * num_experts)`).
    out_cols: u32,
    model: metal.AneModel,
    model_info: metal.AneModelInfo,
    logits_tile: metal.KvTile,
    logits_tensor: metal.AneTensor,
    plan: metal.AnePlan,
    compile_ns: u64,

    fn deinit(self: *ModelBlock) void {
        self.plan.close();
        self.logits_tensor.close();
        self.logits_tile.close();
        self.model.close();
    }
};

const AsyncReq = struct {
    kind: enum(u8) { predict, shutdown },
    req_id: u64,
    current_layer: u32,
    num_tokens: u32,
    block_idx: usize,
};

const AsyncResp = struct {
    req_id: u64,
    current_layer: u32,
    num_tokens: u32,
    block_idx: usize,
    all_zero_copy: bool,
    elapsed_ns: u64,
    err: ?metal.Error,
};

const PredictorMode = enum {
    single_window,
    full_layer_bank,
};

/// Zero-copy IOSurface ANE cross-layer MoE expert predictor (`A2`).
pub const AneExpertPredictor = struct {
    const RING_CAP: usize = 8;

    config: PredictorConfig,
    mode: PredictorMode,
    x_tile: metal.KvTile,
    x_tensor: metal.AneTensor,
    blocks: []ModelBlock,
    last_predict_ns: u64,
    last_all_zero_copy: bool,

    // Optional persistent background ANE worker state (zero hot-path alloc).
    req_slots: [RING_CAP]AsyncReq,
    resp_slots: [RING_CAP]AsyncResp,
    req_ring: ring.SpscRing(AsyncReq),
    resp_ring: ring.SpscRing(AsyncResp),
    req_seq: std.atomic.Value(u32) align(ring.CACHE_LINE_BYTES),
    resp_seq: std.atomic.Value(u32) align(ring.CACHE_LINE_BYTES),
    worker_ready: std.atomic.Value(u32) align(ring.CACHE_LINE_BYTES),
    worker_qos_raw: std.atomic.Value(u32),
    worker_thread: ?std.Thread,

    pub const Error = error{
        InvalidConfig,
        AsyncNotEnabled,
        InvalidLayer,
        InvalidTokenCount,
        BufferTooSmall,
        OutOfMemory,
    } || metal.Error || metal.ProtoEmitError || vmstat.Error || ring.InitError || std.Thread.SpawnError;

    fn allocateBase(
        allocator: std.mem.Allocator,
        device: metal.Device,
        config: PredictorConfig,
        mode: PredictorMode,
        num_blocks: usize,
    ) Error!*AneExpertPredictor {
        if (!config.validate() or num_blocks == 0) return error.InvalidConfig;

        const self = try allocator.create(AneExpertPredictor);
        errdefer allocator.destroy(self);

        const x_tile = try metal.KvTile.init(device, config.hidden_size, config.max_tokens, .{ .alias_buffer = true });
        errdefer x_tile.close();
        const x_tensor = try metal.AneTensor.fromKvTile(x_tile, &.{ config.max_tokens, config.hidden_size });
        errdefer x_tensor.close();

        const blocks = try allocator.alloc(ModelBlock, num_blocks);
        errdefer allocator.free(blocks);

        self.* = .{
            .config = config,
            .mode = mode,
            .x_tile = x_tile,
            .x_tensor = x_tensor,
            .blocks = blocks,
            .last_predict_ns = 0,
            .last_all_zero_copy = false,
            .req_slots = undefined,
            .resp_slots = undefined,
            .req_ring = undefined,
            .resp_ring = undefined,
            .req_seq = .init(0),
            .resp_seq = .init(0),
            .worker_ready = .init(0),
            .worker_qos_raw = .init(@backingInt(qos.QosClass.unspecified)),
            .worker_thread = null,
        };
        return self;
    }

    fn startAsyncWorkerIfEnabled(self: *AneExpertPredictor) Error!void {
        if (!self.config.enable_async_thread) return;
        self.req_ring = try ring.SpscRing(AsyncReq).init(&self.req_slots);
        self.resp_ring = try ring.SpscRing(AsyncResp).init(&self.resp_slots);
        self.worker_thread = try std.Thread.spawn(.{}, asyncWorkerMain, .{self});
        while (self.worker_ready.load(.acquire) == 0) {
            futexWaitU32(&self.worker_ready, 0);
        }
    }

    fn buildLinearBlock(
        allocator: std.mem.Allocator,
        device: metal.Device,
        config: PredictorConfig,
        x_tensor: metal.AneTensor,
        start_target_layer: u32,
        window_layers: u32,
        layer_weights: []const []const f16,
    ) Error!ModelBlock {
        const raw_out: u32 = window_layers * config.num_experts;
        const out_cols: u32 = alignUp32(raw_out);
        const h: usize = config.hidden_size;
        const e: usize = config.num_experts;

        const stacked = try allocator.alloc(f16, @as(usize, out_cols) * h);
        defer allocator.free(stacked);
        @memset(stacked, @as(f16, @floatCast(0.0)));

        for (0..window_layers) |wl| {
            const src = layer_weights[wl];
            if (src.len != e * h) return error.InvalidConfig;
            @memcpy(stacked[wl * e * h ..][0 .. e * h], src);
        }

        const spec = try metal.emitCoreMlRouterPredictor(allocator, .{
            .batch_size = config.max_tokens,
            .hidden_size = config.hidden_size,
            .out_experts = out_cols,
            .weights_layer1 = stacked,
        });
        defer allocator.free(spec);

        const t0 = try vmstat.nowNs();
        const model = try metal.AneModel.loadSpecData(spec, config.compute_units);
        errdefer model.close();
        const t1 = try vmstat.nowNs();
        const minfo = try model.info();

        const logits_tile = try metal.KvTile.init(device, out_cols, config.max_tokens, .{});
        errdefer logits_tile.close();
        const logits_tensor = try metal.AneTensor.fromKvTile(logits_tile, &.{ config.max_tokens, out_cols });
        errdefer logits_tensor.close();

        const plan = try metal.AnePlan.init(
            &.{.{ .name = "x", .tensor = x_tensor }},
            &.{.{ .name = "logits", .tensor = logits_tensor }},
        );
        return .{
            .start_target_layer = start_target_layer,
            .window_layers = window_layers,
            .out_cols = out_cols,
            .model = model,
            .model_info = minfo,
            .logits_tile = logits_tile,
            .logits_tensor = logits_tensor,
            .plan = plan,
            .compile_ns = t1 -| t0,
        };
    }

    /// Compiles an ANE expert predictor directly in memory from target router
    /// weights (`[num_experts, hidden_size]` row-major fp16 per layer).
    ///
    /// - Pass `router_weights_per_layer.len == config.num_layers` to compile a
    ///   full-model overlapped router bank (covering every `current_layer` in
    ///   exactly one ANE call with `ceil((num_layers - 1) / block_stride)` blocks).
    /// - Or pass `router_weights_per_layer.len == config.lookahead_layers` to
    ///   compile a single lookahead window model.
    pub fn initFromRouterWeights(
        allocator: std.mem.Allocator,
        device: metal.Device,
        config: PredictorConfig,
        router_weights_per_layer: []const []const f16,
    ) Error!*AneExpertPredictor {
        if (!config.validate()) return error.InvalidConfig;

        const is_full_bank = router_weights_per_layer.len == config.num_layers and config.num_layers > 1;
        const is_single_window = router_weights_per_layer.len == config.lookahead_layers;
        if (!is_full_bank and !is_single_window) return error.InvalidConfig;

        if (is_single_window) {
            const self = try allocateBase(allocator, device, config, .single_window, 1);
            errdefer {
                allocator.free(self.blocks);
                self.x_tensor.close();
                self.x_tile.close();
                allocator.destroy(self);
            }
            self.blocks[0] = try buildLinearBlock(
                allocator,
                device,
                config,
                self.x_tensor,
                1,
                config.lookahead_layers,
                router_weights_per_layer,
            );
            errdefer self.blocks[0].deinit();
            try self.startAsyncWorkerIfEnabled();
            return self;
        }

        // Full-model overlapped block bank:
        // Block `b` covers current layers `[b * stride .. (b + 1) * stride - 1]`,
        // whose lookahead targets span `[b * stride + 1 .. min(num_layers, (b + 1) * stride + lookahead - 1)]`.
        const stride = config.block_stride;
        const max_cur_layers = config.num_layers - 1;
        const num_blocks: usize = (max_cur_layers + stride - 1) / stride;
        const self = try allocateBase(allocator, device, config, .full_layer_bank, num_blocks);
        var built: usize = 0;
        errdefer {
            for (0..built) |i| self.blocks[i].deinit();
            allocator.free(self.blocks);
            self.x_tensor.close();
            self.x_tile.close();
            allocator.destroy(self);
        }

        for (0..num_blocks) |b| {
            const first_cur: u32 = @intCast(b * stride);
            const last_cur: u32 = @min(first_cur + stride - 1, max_cur_layers - 1);
            const start_target: u32 = first_cur + 1;
            const end_target_excl: u32 = @min(config.num_layers, last_cur + 1 + config.lookahead_layers);
            const win_layers: u32 = end_target_excl - start_target;

            self.blocks[b] = try buildLinearBlock(
                allocator,
                device,
                config,
                self.x_tensor,
                start_target,
                win_layers,
                router_weights_per_layer[start_target..end_target_excl],
            );
            built += 1;
        }

        try self.startAsyncWorkerIfEnabled();
        return self;
    }

    /// Compiles a 2-layer bottleneck MLP predictor (`hidden_size -> bottleneck_dim
    /// -> activation -> lookahead_layers * num_experts`) directly onto ANE.
    pub fn initFromBottleneckMlp(
        allocator: std.mem.Allocator,
        device: metal.Device,
        config: PredictorConfig,
        w1_f16: []const f16,
        w2_f16: []const f16,
    ) Error!*AneExpertPredictor {
        if (!config.validate() or config.bottleneck_dim == 0) return error.InvalidConfig;
        const raw_out: u32 = config.lookahead_layers * config.num_experts;
        const out_cols: u32 = alignUp32(raw_out);
        if (w1_f16.len != @as(usize, config.bottleneck_dim) * config.hidden_size) return error.InvalidConfig;
        if (w2_f16.len != @as(usize, raw_out) * config.bottleneck_dim) return error.InvalidConfig;

        const w2_padded = try allocator.alloc(f16, @as(usize, out_cols) * config.bottleneck_dim);
        defer allocator.free(w2_padded);
        @memset(w2_padded, @as(f16, @floatCast(0.0)));
        @memcpy(w2_padded[0..w2_f16.len], w2_f16);

        const self = try allocateBase(allocator, device, config, .single_window, 1);
        errdefer {
            allocator.free(self.blocks);
            self.x_tensor.close();
            self.x_tile.close();
            allocator.destroy(self);
        }

        const spec = try metal.emitCoreMlRouterPredictor(allocator, .{
            .batch_size = config.max_tokens,
            .hidden_size = config.hidden_size,
            .out_experts = out_cols,
            .bottleneck_dim = config.bottleneck_dim,
            .activation = config.activation,
            .weights_layer1 = w1_f16,
            .weights_layer2 = w2_padded,
        });
        defer allocator.free(spec);

        const t0 = try vmstat.nowNs();
        const model = try metal.AneModel.loadSpecData(spec, config.compute_units);
        errdefer model.close();
        const t1 = try vmstat.nowNs();
        const minfo = try model.info();

        const logits_tile = try metal.KvTile.init(device, out_cols, config.max_tokens, .{});
        errdefer logits_tile.close();
        const logits_tensor = try metal.AneTensor.fromKvTile(logits_tile, &.{ config.max_tokens, out_cols });
        errdefer logits_tensor.close();
        const plan = try metal.AnePlan.init(
            &.{.{ .name = "x", .tensor = self.x_tensor }},
            &.{.{ .name = "logits", .tensor = logits_tensor }},
        );
        self.blocks[0] = .{
            .start_target_layer = 1,
            .window_layers = config.lookahead_layers,
            .out_cols = out_cols,
            .model = model,
            .model_info = minfo,
            .logits_tile = logits_tile,
            .logits_tensor = logits_tensor,
            .plan = plan,
            .compile_ns = t1 -| t0,
        };
        errdefer self.blocks[0].deinit();

        try self.startAsyncWorkerIfEnabled();
        return self;
    }

    /// Loads an offline-exported CoreML `.mlpackage` / `.mlmodelc` / `.mlmodel`
    /// predictor (`W7`) from disk onto ANE with zero-copy IOSurface `KvTile`s.
    /// Automatically discovers input and output feature names if not explicitly
    /// configured in `config.input_name` / `config.output_name`.
    pub fn loadFromPath(
        allocator: std.mem.Allocator,
        device: metal.Device,
        config: PredictorConfig,
        path: [:0]const u8,
    ) Error!*AneExpertPredictor {
        return loadFromPathImpl(allocator, device, config, path, false);
    }

    /// Loads an all-layers packed CoreML predictor (`W7 --pack-all-layers`,
    /// output shape `[max_tokens, num_layers * num_experts]`, e.g.
    /// `qwen3-30b-a3b-expert-predictor-all48.mlpackage`) as a single ANE block
    /// supporting arbitrary `current_layer` lookahead slicing (`l+1 .. l+L`).
    pub fn loadAllLayersFromPath(
        allocator: std.mem.Allocator,
        device: metal.Device,
        config: PredictorConfig,
        path: [:0]const u8,
    ) Error!*AneExpertPredictor {
        return loadFromPathImpl(allocator, device, config, path, true);
    }

    fn loadFromPathImpl(
        allocator: std.mem.Allocator,
        device: metal.Device,
        config: PredictorConfig,
        path: [:0]const u8,
        packed_all_layers: bool,
    ) Error!*AneExpertPredictor {
        if (!config.validate() or path.len == 0) return error.InvalidConfig;

        const out_layers: u32 = if (packed_all_layers) config.num_layers else config.lookahead_layers;
        const out_cols_raw = std.math.mul(u32, out_layers, config.num_experts) catch return error.InvalidConfig;
        const out_cols: u32 = alignUp32(out_cols_raw);
        if (out_cols > metal.MAX_TEXTURE_DIM) return error.InvalidConfig;

        const mode: PredictorMode = if (packed_all_layers) .full_layer_bank else .single_window;
        const self = try allocateBase(allocator, device, config, mode, 1);
        errdefer {
            allocator.free(self.blocks);
            self.x_tensor.close();
            self.x_tile.close();
            allocator.destroy(self);
        }

        const t0 = try vmstat.nowNs();
        const model = try metal.AneModel.loadPath(path, config.compute_units);
        errdefer model.close();
        const t1 = try vmstat.nowNs();
        const minfo = try model.info();

        var in_name_buf: [64]u8 = undefined;
        var out_name_buf: [64]u8 = undefined;
        const in_name: [:0]const u8 = config.input_name orelse try model.ioName(false, 0, &in_name_buf);
        const out_name: [:0]const u8 = config.output_name orelse try model.ioName(true, 0, &out_name_buf);

        const logits_tile = try metal.KvTile.init(device, out_cols, config.max_tokens, .{});
        errdefer logits_tile.close();
        const logits_tensor = try metal.AneTensor.fromKvTile(logits_tile, &.{ config.max_tokens, out_cols });
        errdefer logits_tensor.close();
        const plan = try metal.AnePlan.init(
            &.{.{ .name = in_name, .tensor = self.x_tensor }},
            &.{.{ .name = out_name, .tensor = logits_tensor }},
        );
        self.blocks[0] = .{
            .start_target_layer = if (packed_all_layers) 0 else 1,
            .window_layers = out_layers,
            .out_cols = out_cols,
            .model = model,
            .model_info = minfo,
            .logits_tile = logits_tile,
            .logits_tensor = logits_tensor,
            .plan = plan,
            .compile_ns = t1 -| t0,
        };
        errdefer self.blocks[0].deinit();

        try self.startAsyncWorkerIfEnabled();
        return self;
    }

    pub fn deinit(self: *AneExpertPredictor, allocator: std.mem.Allocator) void {
        if (self.worker_thread) |t| {
            const stop_req: AsyncReq = .{
                .kind = .shutdown,
                .req_id = 0,
                .current_layer = 0,
                .num_tokens = 0,
                .block_idx = 0,
            };
            while (!self.req_ring.tryPush(stop_req)) {
                std.atomic.spinLoopHint();
            }
            _ = self.req_seq.fetchAdd(1, .release);
            futexWakeOneU32(&self.req_seq);
            t.join();
            self.worker_thread = null;
        }
        for (self.blocks) |*b| b.deinit();
        allocator.free(self.blocks);
        self.x_tensor.close();
        self.x_tile.close();
        allocator.destroy(self);
    }

    /// Zero-copy IOSurface input tile (`[max_tokens, hidden_size]` fp16, with alias buffer).
    pub fn inputTile(self: *const AneExpertPredictor) metal.KvTile {
        return self.x_tile;
    }

    /// Zero-copy `MTLBuffer` alias over `x_tile` so GPU kernels can write hidden
    /// states straight into the ANE predictor's input pages with zero `memcpy`.
    pub fn inputAliasBuffer(self: *const AneExpertPredictor) metal.Buffer {
        return self.x_tile.aliasBuffer() orelse unreachable;
    }

    /// Copies `num_tokens` hidden rows `[num_tokens, hidden_size]` into `x_tile`
    /// and zeroes any unused padded rows up to `max_tokens`.
    pub fn writeInputHiddenF16(
        self: *AneExpertPredictor,
        hidden_rows: []const f16,
        num_tokens: u32,
    ) Error!void {
        if (num_tokens == 0 or num_tokens > self.config.max_tokens) return error.InvalidTokenCount;
        const needed = @as(usize, num_tokens) * self.config.hidden_size;
        if (hidden_rows.len < needed) return error.BufferTooSmall;

        var guard = try self.x_tile.lock(false);
        errdefer guard.release();
        const dst = guard.f16Slice();
        @memcpy(dst[0..needed], hidden_rows[0..needed]);
        const total = @as(usize, self.config.max_tokens) * self.config.hidden_size;
        if (total > needed) {
            @memset(dst[needed..total], @as(f16, @floatCast(0.0)));
        }
        try guard.unlock();
    }

    pub fn usesNeuralEngine(self: *const AneExpertPredictor) bool {
        if (self.blocks.len == 0) return false;
        for (self.blocks) |b| {
            if (!b.model_info.usesNeuralEngine()) return false;
        }
        return true;
    }

    pub fn allOutputsZeroCopy(self: *const AneExpertPredictor) bool {
        return self.last_all_zero_copy;
    }

    pub fn blockCount(self: *const AneExpertPredictor) usize {
        return self.blocks.len;
    }

    pub fn totalCompileMs(self: *const AneExpertPredictor) f64 {
        var sum: u64 = 0;
        for (self.blocks) |b| sum += b.compile_ns;
        return @as(f64, @floatFromInt(sum)) / 1_000_000.0;
    }

    pub fn lastPredictNs(self: *const AneExpertPredictor) u64 {
        return self.last_predict_ns;
    }

    pub fn aneThreadQos(self: *const AneExpertPredictor) qos.QosClass {
        const raw = self.worker_qos_raw.load(.acquire);
        return std.enums.fromInt(qos.QosClass, raw) orelse .unspecified;
    }

    fn resolveBlockIndex(self: *const AneExpertPredictor, current_layer: u32) Error!usize {
        if (current_layer >= self.config.num_layers) return error.InvalidLayer;
        return switch (self.mode) {
            .single_window => 0,
            .full_layer_bank => blk: {
                const idx = current_layer / self.config.block_stride;
                break :blk @min(idx, self.blocks.len - 1);
            },
        };
    }

    fn predictedLayerCount(self: *const AneExpertPredictor, current_layer: u32) u32 {
        if (current_layer + 1 >= self.config.num_layers) return 0;
        return @min(self.config.lookahead_layers, self.config.num_layers - 1 - current_layer);
    }

    fn blockOffsetForTarget(self: *const AneExpertPredictor, block: *const ModelBlock, target_layer: u32, delta: u32) u32 {
        return switch (self.mode) {
            .single_window => delta,
            .full_layer_bank => target_layer - block.start_target_layer,
        };
    }

    fn unpackBlockOutputs(
        self: *AneExpertPredictor,
        block: *ModelBlock,
        current_layer: u32,
        num_tokens: u32,
        top_m_prefetch: u32,
        cold_masks_by_layer: ?[]const tree_mod.ExpertBitset,
        out_bitsets_per_layer: []tree_mod.ExpertBitset,
        out_prefetch_list: []PrefetchCandidate,
    ) Error!struct { predicted_layers: u32, prefetch_count: u32 } {
        const pred_layers = self.predictedLayerCount(current_layer);
        if (pred_layers == 0) return .{ .predicted_layers = 0, .prefetch_count = 0 };
        if (out_bitsets_per_layer.len < pred_layers) return error.BufferTooSmall;
        if (cold_masks_by_layer) |cm| {
            if (cm.len < self.config.num_layers) return error.BufferTooSmall;
        }

        const n_exp = self.config.num_experts;
        const top_k = self.config.top_k;
        const out_cols = block.out_cols;

        var guard = try block.logits_tile.lock(true);
        errdefer guard.release();
        const logits_f16 = guard.f16Slice();

        var cand_buf: [MAX_LOOKAHEAD * MAX_EXPERTS]PrefetchCandidate = undefined;
        var cand_count: usize = 0;

        for (0..pred_layers) |delta_u| {
            const delta: u32 = @intCast(delta_u);
            const target_layer: u32 = current_layer + 1 + delta;
            const win_slot = self.blockOffsetForTarget(block, target_layer, delta);
            const col_base: usize = @as(usize, win_slot) * n_exp;

            var layer_union = tree_mod.ExpertBitset.empty();
            var agg_scores: [MAX_EXPERTS]f32 = @splat(0.0);

            for (0..num_tokens) |tok| {
                const row = logits_f16[tok * out_cols + col_base ..][0..n_exp];

                // Softmax normalization in fp32 for stable cross-layer score ranking.
                var max_v: f32 = -std.math.inf(f32);
                for (row) |h| {
                    const v: f32 = @floatCast(h);
                    if (v > max_v) max_v = v;
                }
                var sum_exp: f32 = 0.0;
                var probs: [MAX_EXPERTS]f32 = undefined;
                for (row, 0..) |h, ei| {
                    const p = @exp(@as(f32, @floatCast(h)) - max_v);
                    probs[ei] = p;
                    sum_exp += p;
                }
                const inv_sum: f32 = if (sum_exp > 0.0) 1.0 / sum_exp else 0.0;

                // Select top-K experts into `layer_union` (zero heap allocation).
                var top_ids: [16]u32 = @splat(0);
                var top_vals: [16]f32 = @splat(-std.math.inf(f32));
                for (0..n_exp) |ei| {
                    const p = probs[ei] * inv_sum;
                    agg_scores[ei] += p;

                    if (p > top_vals[top_k - 1]) {
                        var ins: usize = top_k - 1;
                        while (ins > 0 and p > top_vals[ins - 1]) : (ins -= 1) {
                            top_vals[ins] = top_vals[ins - 1];
                            top_ids[ins] = top_ids[ins - 1];
                        }
                        top_vals[ins] = p;
                        top_ids[ins] = @intCast(ei);
                    }
                }
                for (0..top_k) |ki| {
                    layer_union.set(top_ids[ki]);
                }
            }

            out_bitsets_per_layer[delta_u] = layer_union;

            // Collect cold-expert (or all predicted) candidates for this target layer.
            for (0..n_exp) |ei| {
                const eid: u32 = @intCast(ei);
                if (cold_masks_by_layer) |cm| {
                    if (!cm[target_layer].contains(eid)) continue;
                } else {
                    if (!layer_union.contains(eid)) continue;
                }
                cand_buf[cand_count] = .{
                    .layer = @intCast(target_layer),
                    .expert = @intCast(eid),
                    .score = agg_scores[ei],
                };
                cand_count += 1;
            }
        }
        try guard.unlock();

        // Sort candidates descending by score (deterministic tie-break by layer, then expert).
        const Cmp = struct {
            fn lessThan(_: void, a: PrefetchCandidate, b: PrefetchCandidate) bool {
                if (a.score != b.score) return a.score > b.score;
                if (a.layer != b.layer) return a.layer < b.layer;
                return a.expert < b.expert;
            }
        };
        std.mem.sortUnstable(PrefetchCandidate, cand_buf[0..cand_count], {}, Cmp.lessThan);

        const emit_count: usize = @min(cand_count, @min(@as(usize, top_m_prefetch), out_prefetch_list.len));
        @memcpy(out_prefetch_list[0..emit_count], cand_buf[0..emit_count]);

        return .{
            .predicted_layers = pred_layers,
            .prefetch_count = @intCast(emit_count),
        };
    }

    /// Runs ANE cross-layer prediction synchronously over the hidden states
    /// currently in `x_tile` (written either zero-copy by GPU via `inputAliasBuffer`
    /// or via `writeInputHiddenF16`).
    ///
    /// Writes:
    /// - `out_bitsets_per_layer[0..predicted_layers]`: predicted active `ExpertBitset`
    ///   union across `0..num_tokens` for layers `current_layer + 1 .. + predicted_layers`.
    /// - `out_prefetch_list[0..prefetch_count]`: ranked `(layer, expert, score)`
    ///   cold-expert prefetch candidates (filtered by `cold_masks_by_layer` if non-null).
    ///
    /// Zero dynamic heap allocation in our Zig code.
    pub fn predictAhead(
        self: *AneExpertPredictor,
        current_layer: u32,
        num_tokens: u32,
        top_m_prefetch: u32,
        cold_masks_by_layer: ?[]const tree_mod.ExpertBitset,
        out_bitsets_per_layer: []tree_mod.ExpertBitset,
        out_prefetch_list: []PrefetchCandidate,
    ) Error!PredictResult {
        if (num_tokens == 0 or num_tokens > self.config.max_tokens) return error.InvalidTokenCount;
        const pred_layers = self.predictedLayerCount(current_layer);
        if (pred_layers == 0) {
            return .{
                .current_layer = current_layer,
                .predicted_layers = 0,
                .prefetch_count = 0,
                .all_outputs_zero_copy = true,
                .predict_ns = 0,
            };
        }

        const block_idx = try self.resolveBlockIndex(current_layer);
        const block = &self.blocks[block_idx];

        const t0 = try vmstat.nowNs();
        const pinfo = try block.model.predict(block.plan);
        const t1 = try vmstat.nowNs();

        self.last_predict_ns = t1 -| t0;
        self.last_all_zero_copy = pinfo.allOutputsZeroCopy();

        const unpacked = try self.unpackBlockOutputs(
            block,
            current_layer,
            num_tokens,
            top_m_prefetch,
            cold_masks_by_layer,
            out_bitsets_per_layer,
            out_prefetch_list,
        );
        return .{
            .current_layer = current_layer,
            .predicted_layers = unpacked.predicted_layers,
            .prefetch_count = unpacked.prefetch_count,
            .all_outputs_zero_copy = pinfo.allOutputsZeroCopy(),
            .predict_ns = self.last_predict_ns,
        };
    }

    /// Writes per-node per-layer `ExpertBitset`s in `[token * predicted_layers + delta]`
    /// layout directly consumable by `DraftTree.pruneToOptimalSubtree` (D7, ADR-0009).
    /// Zero dynamic heap allocation.
    pub fn predictPerTokenBitsets(
        self: *AneExpertPredictor,
        current_layer: u32,
        num_tokens: u32,
        out_node_layer_bitsets: []tree_mod.ExpertBitset,
    ) Error!PredictResult {
        if (num_tokens == 0 or num_tokens > self.config.max_tokens) return error.InvalidTokenCount;
        const pred_layers = self.predictedLayerCount(current_layer);
        if (pred_layers == 0) {
            return .{
                .current_layer = current_layer,
                .predicted_layers = 0,
                .prefetch_count = 0,
                .all_outputs_zero_copy = true,
                .predict_ns = 0,
            };
        }
        if (out_node_layer_bitsets.len < @as(usize, num_tokens) * pred_layers) return error.BufferTooSmall;

        const block_idx = try self.resolveBlockIndex(current_layer);
        const block = &self.blocks[block_idx];

        const t0 = try vmstat.nowNs();
        const pinfo = try block.model.predict(block.plan);
        const t1 = try vmstat.nowNs();

        self.last_predict_ns = t1 -| t0;
        self.last_all_zero_copy = pinfo.allOutputsZeroCopy();

        const n_exp = self.config.num_experts;
        const top_k = self.config.top_k;
        const out_cols = block.out_cols;

        var guard = try block.logits_tile.lock(true);
        errdefer guard.release();
        const logits_f16 = guard.f16Slice();

        for (0..num_tokens) |tok| {
            for (0..pred_layers) |delta_u| {
                const delta: u32 = @intCast(delta_u);
                const target_layer: u32 = current_layer + 1 + delta;
                const win_slot = self.blockOffsetForTarget(block, target_layer, delta);
                const row = logits_f16[tok * out_cols + @as(usize, win_slot) * n_exp ..][0..n_exp];

                var top_ids: [16]u32 = @splat(0);
                var top_vals: [16]f32 = @splat(-std.math.inf(f32));
                for (0..n_exp) |ei| {
                    const v: f32 = @floatCast(row[ei]);
                    if (v > top_vals[top_k - 1]) {
                        var ins: usize = top_k - 1;
                        while (ins > 0 and v > top_vals[ins - 1]) : (ins -= 1) {
                            top_vals[ins] = top_vals[ins - 1];
                            top_ids[ins] = top_ids[ins - 1];
                        }
                        top_vals[ins] = v;
                        top_ids[ins] = @intCast(ei);
                    }
                }
                var bs = tree_mod.ExpertBitset.empty();
                for (0..top_k) |ki| bs.set(top_ids[ki]);
                out_node_layer_bitsets[tok * pred_layers + delta_u] = bs;
            }
        }
        try guard.unlock();

        return .{
            .current_layer = current_layer,
            .predicted_layers = pred_layers,
            .prefetch_count = 0,
            .all_outputs_zero_copy = pinfo.allOutputsZeroCopy(),
            .predict_ns = self.last_predict_ns,
        };
    }

    /// Reads raw fp16 logits produced by the most recent prediction for
    /// `(token_idx, target_layer)` into `dst[0..num_experts]`.
    pub fn copyLastLogitsF32(
        self: *AneExpertPredictor,
        current_layer: u32,
        delta: u32,
        token_idx: u32,
        dst: []f32,
    ) Error!void {
        if (token_idx >= self.config.max_tokens) return error.InvalidTokenCount;
        if (delta >= self.predictedLayerCount(current_layer)) return error.InvalidLayer;
        if (dst.len < self.config.num_experts) return error.BufferTooSmall;

        const block_idx = try self.resolveBlockIndex(current_layer);
        const block = &self.blocks[block_idx];
        const target_layer = current_layer + 1 + delta;
        const win_slot = self.blockOffsetForTarget(block, target_layer, delta);

        var guard = try block.logits_tile.lock(true);
        errdefer guard.release();
        const row = guard.f16Slice()[@as(usize, token_idx) * block.out_cols + @as(usize, win_slot) * self.config.num_experts ..][0..self.config.num_experts];
        for (row, 0..) |h, i| dst[i] = @floatCast(h);
        try guard.unlock();
    }

    // --- Asynchronous USER_INITIATED background ANE dispatch via SpscRing ---

    /// Enqueues a non-blocking ANE prediction request onto the lock-free `SpscRing`
    /// so the `USER_INITIATED` ANE driver thread runs concurrently with GPU layers.
    /// Returns `false` if the request ring is temporarily full.
    pub fn submitAsync(
        self: *AneExpertPredictor,
        req_id: u64,
        current_layer: u32,
        num_tokens: u32,
    ) Error!bool {
        if (self.worker_thread == null) return error.AsyncNotEnabled;
        if (num_tokens == 0 or num_tokens > self.config.max_tokens) return error.InvalidTokenCount;
        const block_idx = try self.resolveBlockIndex(current_layer);

        const pushed = self.req_ring.tryPush(.{
            .kind = .predict,
            .req_id = req_id,
            .current_layer = current_layer,
            .num_tokens = num_tokens,
            .block_idx = block_idx,
        });
        if (pushed) {
            _ = self.req_seq.fetchAdd(1, .release);
            futexWakeOneU32(&self.req_seq);
        }
        return pushed;
    }

    /// Polls the completion `SpscRing` without blocking. When ready, unpacks
    /// predicted bitsets and ranked cold-expert prefetch candidates with zero
    /// dynamic heap allocation.
    pub fn tryPollAsync(
        self: *AneExpertPredictor,
        top_m_prefetch: u32,
        cold_masks_by_layer: ?[]const tree_mod.ExpertBitset,
        out_bitsets_per_layer: []tree_mod.ExpertBitset,
        out_prefetch_list: []PrefetchCandidate,
    ) Error!?AsyncPredictResult {
        if (self.worker_thread == null) return error.AsyncNotEnabled;
        const resp = self.resp_ring.tryPop() orelse return null;
        return try self.finishAsyncResp(resp, top_m_prefetch, cold_masks_by_layer, out_bitsets_per_layer, out_prefetch_list);
    }

    /// Blocks on the native Darwin futex doorbell until the next async ANE
    /// prediction finishes, then unpacks its bitsets and prefetch candidates.
    pub fn waitAsync(
        self: *AneExpertPredictor,
        top_m_prefetch: u32,
        cold_masks_by_layer: ?[]const tree_mod.ExpertBitset,
        out_bitsets_per_layer: []tree_mod.ExpertBitset,
        out_prefetch_list: []PrefetchCandidate,
    ) Error!AsyncPredictResult {
        if (self.worker_thread == null) return error.AsyncNotEnabled;
        const resp = blk: {
            while (true) {
                if (self.resp_ring.tryPop()) |r| break :blk r;
                const seen = self.resp_seq.load(.acquire);
                if (self.resp_ring.tryPop()) |r| break :blk r;
                futexWaitU32(&self.resp_seq, seen);
            }
        };
        return self.finishAsyncResp(resp, top_m_prefetch, cold_masks_by_layer, out_bitsets_per_layer, out_prefetch_list);
    }

    fn finishAsyncResp(
        self: *AneExpertPredictor,
        resp: AsyncResp,
        top_m_prefetch: u32,
        cold_masks_by_layer: ?[]const tree_mod.ExpertBitset,
        out_bitsets_per_layer: []tree_mod.ExpertBitset,
        out_prefetch_list: []PrefetchCandidate,
    ) Error!AsyncPredictResult {
        if (resp.err) |e| return e;
        self.last_predict_ns = resp.elapsed_ns;
        self.last_all_zero_copy = resp.all_zero_copy;
        const block = &self.blocks[resp.block_idx];
        const unpacked = try self.unpackBlockOutputs(
            block,
            resp.current_layer,
            resp.num_tokens,
            top_m_prefetch,
            cold_masks_by_layer,
            out_bitsets_per_layer,
            out_prefetch_list,
        );
        return .{
            .req_id = resp.req_id,
            .result = .{
                .current_layer = resp.current_layer,
                .predicted_layers = unpacked.predicted_layers,
                .prefetch_count = unpacked.prefetch_count,
                .all_outputs_zero_copy = resp.all_zero_copy,
                .predict_ns = resp.elapsed_ns,
            },
        };
    }

    fn asyncWorkerMain(self: *AneExpertPredictor) void {
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
            const req = blk: {
                while (true) {
                    if (self.req_ring.tryPop()) |item| break :blk item;
                    const seen = self.req_seq.load(.acquire);
                    if (self.req_ring.tryPop()) |item| break :blk item;
                    futexWaitU32(&self.req_seq, seen);
                }
            };
            if (req.kind == .shutdown) return;

            const block = &self.blocks[req.block_idx];
            const t0 = vmstat.nowNs() catch 0;
            var zc = false;
            const err_opt: ?metal.Error = if (block.model.predict(block.plan)) |pinfo| blk: {
                zc = pinfo.allOutputsZeroCopy();
                break :blk null;
            } else |e| e;
            const t1 = vmstat.nowNs() catch t0;

            const resp: AsyncResp = .{
                .req_id = req.req_id,
                .current_layer = req.current_layer,
                .num_tokens = req.num_tokens,
                .block_idx = req.block_idx,
                .all_zero_copy = zc,
                .elapsed_ns = t1 -| t0,
                .err = err_opt,
            };
            while (!self.resp_ring.tryPush(resp)) {
                std.atomic.spinLoopHint();
            }
            _ = self.resp_seq.fetchAdd(1, .release);
            futexWakeOneU32(&self.resp_seq);
        }
    }
};

// ---------------------------------------------------------------------------
// Recall@K (K in {8, 12, 16}) & Lead-Time Baseline Evaluator (A2)
// ---------------------------------------------------------------------------

/// Computes exact `Recall@K = |predicted_top_K ∩ actual_experts| / actual_experts.len`
/// for one token's predicted router logits (`[num_experts]`) against ground-truth
/// `actual_experts` (e.g. top-8 actual routed experts). Zero dynamic heap allocation.
pub fn computeRecallAtK(
    predicted_logits: []const f32,
    actual_experts: []const u32,
    k: u32,
) f64 {
    if (actual_experts.len == 0 or predicted_logits.len == 0 or k == 0) return 0.0;
    const eff_k: usize = @min(@as(usize, k), @min(predicted_logits.len, 64));

    var top_ids: [64]u32 = @splat(0);
    var top_vals: [64]f32 = @splat(-std.math.inf(f32));
    for (predicted_logits, 0..) |v, ei| {
        if (v > top_vals[eff_k - 1]) {
            var ins: usize = eff_k - 1;
            while (ins > 0 and v > top_vals[ins - 1]) : (ins -= 1) {
                top_vals[ins] = top_vals[ins - 1];
                top_ids[ins] = top_ids[ins - 1];
            }
            top_vals[ins] = v;
            top_ids[ins] = @intCast(ei);
        }
    }

    var pred_set = tree_mod.ExpertBitset.empty();
    for (0..eff_k) |i| {
        if (top_ids[i] < MAX_EXPERTS) pred_set.set(top_ids[i]);
    }

    var hits: u32 = 0;
    for (actual_experts) |eid| {
        if (pred_set.contains(eid)) hits += 1;
    }
    return @as(f64, @floatFromInt(hits)) / @as(f64, @floatFromInt(actual_experts.len));
}

/// Convenience wrapper for fp16 router logits.
pub fn computeRecallAtKFromF16(
    predicted_logits_f16: []const f16,
    actual_experts: []const u32,
    k: u32,
) f64 {
    var buf: [MAX_EXPERTS]f32 = undefined;
    const n = @min(predicted_logits_f16.len, MAX_EXPERTS);
    for (0..n) |i| buf[i] = @floatCast(predicted_logits_f16[i]);
    return computeRecallAtK(buf[0..n], actual_experts, k);
}

/// CPU fp32 reference router matvec `logits[e] = dot(x, W[e])` returning exact
/// top-`top_k` expert IDs for ground-truth or CPU baseline evaluation.
pub fn routeCpuTopK(
    x_f16: []const f16,
    router_w_f16: []const f16,
    num_experts: u32,
    hidden_size: u32,
    top_k: u32,
    out_logits: []f32,
    out_topk: []u32,
) void {
    std.debug.assert(out_logits.len >= num_experts);
    std.debug.assert(out_topk.len >= top_k);

    var top_vals: [16]f32 = @splat(-std.math.inf(f32));
    for (0..num_experts) |ei| {
        const w_row = router_w_f16[ei * hidden_size ..][0..hidden_size];
        var acc: f32 = 0.0;
        for (0..hidden_size) |ci| {
            acc += @as(f32, @floatCast(x_f16[ci])) * @as(f32, @floatCast(w_row[ci]));
        }
        out_logits[ei] = acc;
        if (acc > top_vals[top_k - 1]) {
            var ins: usize = top_k - 1;
            while (ins > 0 and acc > top_vals[ins - 1]) : (ins -= 1) {
                top_vals[ins] = top_vals[ins - 1];
                out_topk[ins] = out_topk[ins - 1];
            }
            top_vals[ins] = acc;
            out_topk[ins] = @intCast(ei);
        }
    }
}

pub const PredictorEvalConfig = struct {
    /// Default geometry matches Qwen3-30B-A3B (`128` experts, `top-8`, `hidden=2048`).
    num_layers: u32 = 16,
    num_experts: u32 = 128,
    hidden_size: u32 = 2048,
    top_k: u32 = 8,
    lookahead_layers: u32 = 2,
    batch_tokens: u32 = 1,
    /// Per-layer residual perturbation amplitude simulating transformer layer drift
    /// (`0.22` corresponds to ~0.95 adjacent-layer cosine similarity).
    residual_drift_scale: f32 = 0.22,
    /// Reference GPU execution time per MoE layer (`~460 us` for Qwen3-30B-A3B at 44 tok/s).
    gpu_layer_time_us: f64 = 460.0,
    /// Target NVMe cold-expert read latency to hide (`~400 us` for 2 MiB at 5 GB/s).
    nvme_expert_read_us: f64 = 400.0,
    warmup_iters: u32 = 4,
    timed_iters: u32 = 30,
};

pub const PredictorEvalReport = struct {
    config: PredictorEvalConfig,
    ane_core_count: u32,
    uses_ane: bool,
    all_outputs_zero_copy: bool,
    block_count: usize,
    compile_ms: f64,
    ane_predict_us: f64,
    cpu_baseline_us: f64,
    /// Available lead time (`lookahead_layers * gpu_layer_time_us - ane_predict_us`)
    /// to issue and complete async NVMe cold-expert reads before layer `l + lookahead`.
    lead_time_us: f64,
    hides_nvme_latency: bool,
    recall_at_8: f64,
    recall_at_12: f64,
    recall_at_16: f64,
    baseline_recall_at_8: f64,
    baseline_recall_at_12: f64,
    baseline_recall_at_16: f64,
    unused_prefetch_ratio_at_12: f64,
    ane_thread_qos: qos.QosClass,
    vm_before: vmstat.Snapshot,
    vm_after: vmstat.Snapshot,

    pub fn write(self: *const PredictorEvalReport, out: *std.Io.Writer) std.Io.Writer.Error!void {
        try out.print(
            \\A2 ANE Cross-Layer MoE Expert Predictor Evaluation
            \\a2.ane_cores                     {d}
            \\a2.shape                         layers={d} experts={d} top_k={d} hidden={d} lookahead={d} M={d}
            \\a2.blocks_compiled               {d} (compile_ms={d:.1}, uses_ane={s}, zero_copy={s})
            \\a2.ane_thread_qos                {s}
            \\a2.ane_predict_us                {d:.2}
            \\a2.cpu_baseline_us               {d:.2}
            \\a2.lead_time_us                  {d:.2} (gpu_layer={d:.1}us x {d}, nvme_target={d:.1}us, hides_nvme={s})
            \\a2.ane_recall                    R@8={d:.3}  R@12={d:.3}  R@16={d:.3}
            \\a2.baseline_recall               R@8={d:.3}  R@12={d:.3}  R@16={d:.3}
            \\a2.unused_prefetch_ratio_k12     {d:.3}
            \\a2.swapouts_delta                {d}
            \\a2.pageouts_delta                {d}
            \\
        , .{
            self.ane_core_count,
            self.config.num_layers,
            self.config.num_experts,
            self.config.top_k,
            self.config.hidden_size,
            self.config.lookahead_layers,
            self.config.batch_tokens,
            self.block_count,
            self.compile_ms,
            if (self.uses_ane) "yes" else "no",
            if (self.all_outputs_zero_copy) "yes" else "no",
            @tagName(self.ane_thread_qos),
            self.ane_predict_us,
            self.cpu_baseline_us,
            self.lead_time_us,
            self.config.gpu_layer_time_us,
            self.config.lookahead_layers,
            self.config.nvme_expert_read_us,
            if (self.hides_nvme_latency) "yes" else "no",
            self.recall_at_8,
            self.recall_at_12,
            self.recall_at_16,
            self.baseline_recall_at_8,
            self.baseline_recall_at_12,
            self.baseline_recall_at_16,
            self.unused_prefetch_ratio_at_12,
            vmstat.Snapshot.swapoutDelta(self.vm_before, self.vm_after),
            vmstat.Snapshot.pageoutDelta(self.vm_before, self.vm_after),
        });
    }
};

fn lcgNextF32(state: *u64) f32 {
    state.* = state.* *% 6364136223846793005 +% 1442695040888963407;
    const u: u32 = @truncate(state.* >> 40);
    return (@as(f32, @floatFromInt(u)) / 16777216.0) * 2.0 - 1.0;
}

/// Runs a complete end-to-end A2 benchmark on real Apple Silicon ANE hardware:
/// compiles a full-model overlapped router bank onto ANE (`USER_INITIATED` async
/// thread enabled), evaluates Recall@8 / Recall@12 / Recall@16 across a residual-
/// stream hidden-state trajectory against ground-truth target router activations
/// and a CPU router-lookahead baseline, measures ANE prediction latency & lead
/// time (`us`), and verifies `swapouts_delta == 0`.
pub fn runPredictorBenchmark(
    allocator: std.mem.Allocator,
    config: PredictorEvalConfig,
) (AneExpertPredictor.Error || vmstat.SwapError)!PredictorEvalReport {
    if (config.num_layers < config.lookahead_layers + 1 or
        config.timed_iters == 0 or
        config.num_experts > MAX_EXPERTS or
        config.batch_tokens == 0 or config.batch_tokens > MAX_BATCH_TOKENS)
    {
        return error.InvalidConfig;
    }

    const device = try metal.Device.open();
    defer device.close();
    const dev_info = try device.info();
    const vm_before = try vmstat.Snapshot.capture();

    const n_layers: usize = config.num_layers;
    const n_exp: usize = config.num_experts;
    const h: usize = config.hidden_size;
    const inv_sqrt_h: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(h)));

    // Generate deterministic per-layer router matrices `[n_exp, h]` fp16.
    const router_storage = try allocator.alloc(f16, n_layers * n_exp * h);
    defer allocator.free(router_storage);
    const router_slices = try allocator.alloc([]const f16, n_layers);
    defer allocator.free(router_slices);

    var rng: u64 = 0xA2A2_2026_1121_0001;
    for (0..n_layers) |l| {
        const dst = router_storage[l * n_exp * h ..][0 .. n_exp * h];
        for (dst) |*w| w.* = @floatCast(lcgNextF32(&rng) * inv_sqrt_h);
        router_slices[l] = dst;
    }

    // Build realistic normalized residual-stream hidden states across all layers:
    // `h_{l+1} = rms_norm(h_l + drift * delta_l)`.
    const hidden_trace = try allocator.alloc(f16, n_layers * config.batch_tokens * h);
    defer allocator.free(hidden_trace);

    for (0..config.batch_tokens) |tok| {
        var cur = try allocator.alloc(f32, h);
        defer allocator.free(cur);
        var sq: f64 = 0.0;
        for (0..h) |ci| {
            const v = lcgNextF32(&rng);
            cur[ci] = v;
            sq += @as(f64, v) * @as(f64, v);
        }
        const inv0: f32 = @floatCast(1.0 / @sqrt(sq / @as(f64, @floatFromInt(h)) + 1e-6));
        for (0..h) |ci| {
            cur[ci] *= inv0;
            hidden_trace[(0 * config.batch_tokens + tok) * h + ci] = @floatCast(cur[ci]);
        }

        for (1..n_layers) |l| {
            var sq_l: f64 = 0.0;
            for (0..h) |ci| {
                cur[ci] += config.residual_drift_scale * lcgNextF32(&rng);
                sq_l += @as(f64, cur[ci]) * @as(f64, cur[ci]);
            }
            const inv_l: f32 = @floatCast(1.0 / @sqrt(sq_l / @as(f64, @floatFromInt(h)) + 1e-6));
            for (0..h) |ci| {
                cur[ci] *= inv_l;
                hidden_trace[(l * config.batch_tokens + tok) * h + ci] = @floatCast(cur[ci]);
            }
        }
    }

    var pred = try AneExpertPredictor.initFromRouterWeights(allocator, device, .{
        .num_layers = config.num_layers,
        .num_experts = config.num_experts,
        .hidden_size = config.hidden_size,
        .lookahead_layers = config.lookahead_layers,
        .max_tokens = config.batch_tokens,
        .top_k = config.top_k,
        .compute_units = .cpu_and_neural_engine,
        .enable_async_thread = true,
    }, router_slices);
    defer pred.deinit(allocator);

    // Evaluate Recall@8/12/16 across every valid (current_layer, lookahead_delta, token)
    // comparing ANE fp16 predictor vs exact CPU fp32 router-lookahead baseline.
    var bitsets: [MAX_LOOKAHEAD]tree_mod.ExpertBitset = undefined;
    var prefetches: [32]PrefetchCandidate = undefined;
    var gt_logits: [MAX_EXPERTS]f32 = undefined;
    var gt_topk: [16]u32 = undefined;
    var base_logits: [MAX_EXPERTS]f32 = undefined;
    var base_topk: [16]u32 = undefined;
    var ane_logits: [MAX_EXPERTS]f32 = undefined;

    var ane_r8_sum: f64 = 0.0;
    var ane_r12_sum: f64 = 0.0;
    var ane_r16_sum: f64 = 0.0;
    var base_r8_sum: f64 = 0.0;
    var base_r12_sum: f64 = 0.0;
    var base_r16_sum: f64 = 0.0;
    var eval_count: u64 = 0;
    var all_zc = true;

    const max_cur = config.num_layers - 1;
    for (0..max_cur) |l_u| {
        const cur_l: u32 = @intCast(l_u);
        const x_cur = hidden_trace[l_u * config.batch_tokens * h ..][0 .. config.batch_tokens * h];
        try pred.writeInputHiddenF16(x_cur, config.batch_tokens);

        // Exercise async USER_INITIATED path on even layers and sync path on odd layers.
        const pr = if ((l_u & 1) == 0) blk: {
            const pushed = try pred.submitAsync(l_u + 1, cur_l, config.batch_tokens);
            std.debug.assert(pushed);
            const ar = try pred.waitAsync(16, null, &bitsets, &prefetches);
            break :blk ar.result;
        } else try pred.predictAhead(cur_l, config.batch_tokens, 16, null, &bitsets, &prefetches);

        all_zc = all_zc and pr.all_outputs_zero_copy;

        for (0..pr.predicted_layers) |d_u| {
            const delta: u32 = @intCast(d_u);
            const target_l = cur_l + 1 + delta;
            for (0..config.batch_tokens) |tok| {
                const x_target = hidden_trace[(target_l * config.batch_tokens + tok) * h ..][0..h];
                const x_now = hidden_trace[(cur_l * config.batch_tokens + tok) * h ..][0..h];

                // Ground truth at `target_l` using `h_{target_l}`.
                routeCpuTopK(x_target, router_slices[target_l], config.num_experts, config.hidden_size, config.top_k, &gt_logits, &gt_topk);
                // CPU router-lookahead baseline using `h_{cur_l}` against `W_{target_l}`.
                routeCpuTopK(x_now, router_slices[target_l], config.num_experts, config.hidden_size, config.top_k, &base_logits, &base_topk);
                // ANE predicted logits for `(tok, delta)`.
                try pred.copyLastLogitsF32(cur_l, delta, @intCast(tok), ane_logits[0..config.num_experts]);

                ane_r8_sum += computeRecallAtK(ane_logits[0..n_exp], gt_topk[0..config.top_k], 8);
                ane_r12_sum += computeRecallAtK(ane_logits[0..n_exp], gt_topk[0..config.top_k], 12);
                ane_r16_sum += computeRecallAtK(ane_logits[0..n_exp], gt_topk[0..config.top_k], 16);

                base_r8_sum += computeRecallAtK(base_logits[0..n_exp], gt_topk[0..config.top_k], 8);
                base_r12_sum += computeRecallAtK(base_logits[0..n_exp], gt_topk[0..config.top_k], 12);
                base_r16_sum += computeRecallAtK(base_logits[0..n_exp], gt_topk[0..config.top_k], 16);
                eval_count += 1;
            }
        }
    }

    // Timed ANE prediction latency vs CPU lookahead baseline latency.
    try pred.writeInputHiddenF16(hidden_trace[0 .. config.batch_tokens * h], config.batch_tokens);
    for (0..config.warmup_iters) |_| {
        _ = try pred.predictAhead(0, config.batch_tokens, 16, null, &bitsets, &prefetches);
    }
    const at0 = try vmstat.nowNs();
    for (0..config.timed_iters) |_| {
        _ = try pred.predictAhead(0, config.batch_tokens, 16, null, &bitsets, &prefetches);
    }
    const at1 = try vmstat.nowNs();
    const ane_us = (@as(f64, @floatFromInt(at1 -| at0)) / @as(f64, @floatFromInt(config.timed_iters))) / 1000.0;

    const ct0 = try vmstat.nowNs();
    for (0..config.timed_iters) |_| {
        for (0..config.lookahead_layers) |d_u| {
            for (0..config.batch_tokens) |tok| {
                routeCpuTopK(
                    hidden_trace[tok * h ..][0..h],
                    router_slices[1 + d_u],
                    config.num_experts,
                    config.hidden_size,
                    config.top_k,
                    &base_logits,
                    &base_topk,
                );
                std.mem.doNotOptimizeAway(base_topk[0]);
            }
        }
    }
    const ct1 = try vmstat.nowNs();
    const cpu_us = (@as(f64, @floatFromInt(ct1 -| ct0)) / @as(f64, @floatFromInt(config.timed_iters))) / 1000.0;

    const vm_after = try vmstat.Snapshot.capture();
    try vmstat.checkNoSwap(vm_before, vm_after);

    const denom: f64 = @floatFromInt(@max(eval_count, 1));
    const r8 = ane_r8_sum / denom;
    const r12 = ane_r12_sum / denom;
    const r16 = ane_r16_sum / denom;
    const lead_us = @as(f64, @floatFromInt(config.lookahead_layers)) * config.gpu_layer_time_us - ane_us;
    const unused_k12 = 1.0 - (r12 * @as(f64, @floatFromInt(config.top_k)) / 12.0);

    return .{
        .config = config,
        .ane_core_count = dev_info.aneCoreCount(),
        .uses_ane = pred.usesNeuralEngine(),
        .all_outputs_zero_copy = all_zc,
        .block_count = pred.blockCount(),
        .compile_ms = pred.totalCompileMs(),
        .ane_predict_us = ane_us,
        .cpu_baseline_us = cpu_us,
        .lead_time_us = lead_us,
        .hides_nvme_latency = lead_us >= config.nvme_expert_read_us,
        .recall_at_8 = r8,
        .recall_at_12 = r12,
        .recall_at_16 = r16,
        .baseline_recall_at_8 = base_r8_sum / denom,
        .baseline_recall_at_12 = base_r12_sum / denom,
        .baseline_recall_at_16 = base_r16_sum / denom,
        .unused_prefetch_ratio_at_12 = std.math.clamp(unused_k12, 0.0, 1.0),
        .ane_thread_qos = pred.aneThreadQos(),
        .vm_before = vm_before,
        .vm_after = vm_after,
    };
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

test "computeRecallAtK computes exact Recall@8, Recall@12, and Recall@16" {
    var logits: [32]f32 = @splat(-5.0);
    // Top-16 predicted: experts 0..15 with strictly decreasing logits.
    for (0..16) |i| {
        logits[i] = 16.0 - @as(f32, @floatFromInt(i));
    }
    // Actual top-8 contains: 4 inside 0..7, 2 inside 8..11, 1 inside 12..15, 1 outside (20).
    const actual = [_]u32{ 0, 2, 4, 6, 8, 10, 13, 20 };
    try std.testing.expectApproxEqAbs(@as(f64, 4.0 / 8.0), computeRecallAtK(&logits, &actual, 8), 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 6.0 / 8.0), computeRecallAtK(&logits, &actual, 12), 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 7.0 / 8.0), computeRecallAtK(&logits, &actual, 16), 1e-9);
}

test "AneExpertPredictor in-memory full-layer bank, cold filtering, async SpscRing thread, and per-token bitsets match CPU reference" {
    const gpa = std.testing.allocator;
    const device = try metal.Device.open();
    defer device.close();
    const dev_info = try device.info();

    const num_layers: u32 = 10;
    const num_experts: u32 = 32;
    const hidden_size: u32 = 64;
    const top_k: u32 = 4;
    const lookahead: u32 = 2;
    const max_tokens: u32 = 4;

    const router_storage = try gpa.alloc(f16, num_layers * num_experts * hidden_size);
    defer gpa.free(router_storage);
    const router_slices = try gpa.alloc([]const f16, num_layers);
    defer gpa.free(router_slices);

    var rng: u64 = 0x1234_5678_9ABC_DEF0;
    for (0..num_layers) |l| {
        const dst = router_storage[l * num_experts * hidden_size ..][0 .. num_experts * hidden_size];
        for (dst) |*w| w.* = @floatCast(lcgNextF32(&rng) * 0.125);
        router_slices[l] = dst;
    }

    var pred = try AneExpertPredictor.initFromRouterWeights(gpa, device, .{
        .num_layers = num_layers,
        .num_experts = num_experts,
        .hidden_size = hidden_size,
        .lookahead_layers = lookahead,
        .max_tokens = max_tokens,
        .top_k = top_k,
        .block_stride = 4, // 9 current layers -> 3 overlapped ANE blocks!
        .enable_async_thread = true,
    }, router_slices);
    defer pred.deinit(gpa);

    try std.testing.expectEqual(@as(usize, 3), pred.blockCount());
    try std.testing.expectEqual(qos.QosClass.user_initiated, pred.aneThreadQos());
    if (dev_info.aneCoreCount() > 0) try std.testing.expect(pred.usesNeuralEngine());
    try std.testing.expect(pred.inputAliasBuffer().gpuAddress() != 0);

    // Write 3 tokens into zero-copy IOSurface input tile (row 3 auto-zeroed).
    const num_tokens: u32 = 3;
    const x_rows = try gpa.alloc(f16, num_tokens * hidden_size);
    defer gpa.free(x_rows);
    for (x_rows) |*v| v.* = @floatCast(lcgNextF32(&rng) * 0.5);
    try pred.writeInputHiddenF16(x_rows, num_tokens);

    // Mark odd experts as cold on every layer.
    var cold_masks: [num_layers]tree_mod.ExpertBitset = @splat(tree_mod.ExpertBitset.empty());
    for (&cold_masks) |*cm| {
        var e: u32 = 1;
        while (e < num_experts) : (e += 2) cm.set(e);
    }

    var bitsets: [lookahead]tree_mod.ExpertBitset = undefined;
    var prefetches: [12]PrefetchCandidate = undefined;

    // Test sync prediction crossing block boundary (current_layer = 3 predicts layers 4, 5).
    const res_sync = try pred.predictAhead(3, num_tokens, 12, &cold_masks, &bitsets, &prefetches);
    try std.testing.expectEqual(@as(u32, 2), res_sync.predicted_layers);
    try std.testing.expect(res_sync.all_outputs_zero_copy);
    try std.testing.expect(res_sync.prefetch_count > 0);
    // Every emitted prefetch candidate must be cold (odd expert) and sorted descending!
    for (prefetches[0..res_sync.prefetch_count], 0..) |cand, i| {
        try std.testing.expect(cand.layer == 4 or cand.layer == 5);
        try std.testing.expect((cand.expert & 1) == 1);
        if (i > 0) try std.testing.expect(prefetches[i - 1].score >= cand.score);
    }

    // Verify per-token logits match CPU fp32 reference closely -> Recall@8 == 1.0!
    var cpu_logits: [num_experts]f32 = undefined;
    var cpu_topk: [top_k]u32 = undefined;
    var ane_logits: [num_experts]f32 = undefined;
    for (0..2) |d_u| {
        const target_l: usize = 4 + d_u;
        for (0..num_tokens) |tok| {
            routeCpuTopK(x_rows[tok * hidden_size ..][0..hidden_size], router_slices[target_l], num_experts, hidden_size, top_k, &cpu_logits, &cpu_topk);
            try pred.copyLastLogitsF32(3, @intCast(d_u), @intCast(tok), &ane_logits);
            for (0..num_experts) |ei| {
                try std.testing.expectApproxEqAbs(cpu_logits[ei], ane_logits[ei], 0.03);
            }
            const r4 = computeRecallAtK(&ane_logits, &cpu_topk, 4);
            try std.testing.expect(r4 >= 0.75);
        }
    }

    // Verify async SpscRing dispatch produces identical bitsets.
    var async_bitsets: [lookahead]tree_mod.ExpertBitset = undefined;
    var async_prefetches: [12]PrefetchCandidate = undefined;
    try std.testing.expect(try pred.submitAsync(42, 3, num_tokens));
    const ar = try pred.waitAsync(12, &cold_masks, &async_bitsets, &async_prefetches);
    try std.testing.expectEqual(@as(u64, 42), ar.req_id);
    try std.testing.expect(ar.result.all_outputs_zero_copy);
    for (0..lookahead) |d_u| {
        try std.testing.expectEqualSlices(u64, &bitsets[d_u].words, &async_bitsets[d_u].words);
    }

    // Verify per-token bitsets for D7 tree pruning.
    var per_tok: [num_tokens * lookahead]tree_mod.ExpertBitset = undefined;
    const pt_res = try pred.predictPerTokenBitsets(3, num_tokens, &per_tok);
    try std.testing.expectEqual(@as(u32, 2), pt_res.predicted_layers);
    for (per_tok) |bs| {
        try std.testing.expectEqual(top_k, bs.count());
    }
}

test "AneExpertPredictor supports 2-layer bottleneck MLP and .mlpackage loadFromPath with auto feature discovery" {
    const gpa = std.testing.allocator;
    const device = try metal.Device.open();
    defer device.close();

    const h: u32 = 64;
    const bn: u32 = 32;
    const n_exp: u32 = 32;
    const lookahead: u32 = 2;
    const m: u32 = 2;

    const w1 = try gpa.alloc(f16, bn * h);
    defer gpa.free(w1);
    const w2 = try gpa.alloc(f16, lookahead * n_exp * bn);
    defer gpa.free(w2);
    var rng: u64 = 0xCAFE_BABE_0001;
    for (w1) |*v| v.* = @floatCast(lcgNextF32(&rng) * 0.125);
    for (w2) |*v| v.* = @floatCast(lcgNextF32(&rng) * 0.125);

    var mlp_pred = try AneExpertPredictor.initFromBottleneckMlp(gpa, device, .{
        .num_layers = 8,
        .num_experts = n_exp,
        .hidden_size = h,
        .lookahead_layers = lookahead,
        .max_tokens = m,
        .top_k = 4,
        .bottleneck_dim = bn,
        .activation = .silu,
    }, w1, w2);
    defer mlp_pred.deinit(gpa);

    var x: [m * h]f16 = undefined;
    for (&x) |*v| v.* = @floatCast(lcgNextF32(&rng) * 0.5);
    try mlp_pred.writeInputHiddenF16(&x, m);
    var bitsets: [lookahead]tree_mod.ExpertBitset = undefined;
    var prefetches: [8]PrefetchCandidate = undefined;
    const r = try mlp_pred.predictAhead(0, m, 8, null, &bitsets, &prefetches);
    try std.testing.expectEqual(@as(u32, 2), r.predicted_layers);
    try std.testing.expect(r.all_outputs_zero_copy);
    try std.testing.expect(bitsets[0].count() >= 4);
}

test "AneExpertPredictor loadFromPath auto-discovers feature names from .mlpackage and runPredictorBenchmark verifies Recall@K and lead time" {
    const gpa = std.testing.allocator;
    const device = try metal.Device.open();
    defer device.close();

    // 1. External .mlpackage with custom tensor names ("hidden_in" -> "router_out").
    {
        const h: u32 = 64;
        const n_exp: u32 = 32;
        const lookahead: u32 = 2;
        const m: u32 = 2;
        const w = try gpa.alloc(f16, lookahead * n_exp * h);
        defer gpa.free(w);
        var rng: u64 = 0x7777_8888;
        for (w) |*v| v.* = @floatCast(lcgNextF32(&rng) * 0.125);

        const spec = try metal.emitCoreMlRouterPredictor(gpa, .{
            .batch_size = m,
            .hidden_size = h,
            .out_experts = lookahead * n_exp,
            .input_name = "hidden_in",
            .output_name = "router_out",
            .weights_layer1 = w,
        });
        defer gpa.free(spec);

        const tmp_root = std.mem.sliceTo(std.c.getenv("TMPDIR") orelse "/Volumes/Akari/.tmp", 0);
        const pid = std.c.getpid();
        var path_buf: [512]u8 = undefined;
        const p_str = try std.fmt.bufPrint(path_buf[0 .. path_buf.len - 1], "{s}/aminfer_a2_ext_{d}.mlmodel", .{ tmp_root, pid });
        path_buf[p_str.len] = 0;
        const p_z: [:0]const u8 = path_buf[0..p_str.len :0];
        const fd = std.c.open(p_z.ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o644));
        try std.testing.expect(fd >= 0);
        defer _ = std.c.unlink(p_z.ptr);
        {
            defer _ = std.c.close(fd);
            var off: usize = 0;
            while (off < spec.len) {
                const n = std.c.write(fd, spec.ptr + off, spec.len - off);
                try std.testing.expect(n > 0);
                off += @intCast(n);
            }
        }

        var ext_pred = try AneExpertPredictor.loadFromPath(gpa, device, .{
            .num_layers = 6,
            .num_experts = n_exp,
            .hidden_size = h,
            .lookahead_layers = lookahead,
            .max_tokens = m,
            .top_k = 4,
        }, p_z);
        defer ext_pred.deinit(gpa);

        var x: [m * h]f16 = undefined;
        for (&x) |*v| v.* = @floatCast(lcgNextF32(&rng) * 0.5);
        try ext_pred.writeInputHiddenF16(&x, m);
        var bitsets: [lookahead]tree_mod.ExpertBitset = undefined;
        var prefetches: [8]PrefetchCandidate = undefined;
        const pr = try ext_pred.predictAhead(1, m, 8, null, &bitsets, &prefetches);
        try std.testing.expectEqual(@as(u32, 2), pr.predicted_layers);
        try std.testing.expect(pr.all_outputs_zero_copy);
    }

    // 2. Smoke run of `runPredictorBenchmark` (Recall@8/12/16, lead time, zero swap).
    const rep = try runPredictorBenchmark(gpa, .{
        .num_layers = 6,
        .num_experts = 32,
        .hidden_size = 128,
        .top_k = 4,
        .lookahead_layers = 2,
        .batch_tokens = 2,
        .warmup_iters = 1,
        .timed_iters = 4,
    });
    if (rep.ane_core_count > 0) try std.testing.expect(rep.uses_ane);
    try std.testing.expect(rep.all_outputs_zero_copy);
    try std.testing.expect(rep.recall_at_8 > 0.3);
    try std.testing.expect(rep.recall_at_12 >= rep.recall_at_8);
    try std.testing.expect(rep.recall_at_16 >= rep.recall_at_12);
    try std.testing.expect(rep.ane_predict_us > 0.0);
    try std.testing.expect(rep.lead_time_us > 0.0);
    try vmstat.checkNoSwap(rep.vm_before, rep.vm_after);

    var aw: std.Io.Writer.Allocating = .init(gpa);
    defer aw.deinit();
    try rep.write(&aw.writer);
    try std.testing.expect(std.mem.indexOf(u8, aw.written(), "a2.ane_recall") != null);
    try std.testing.expect(std.mem.indexOf(u8, aw.written(), "a2.lead_time_us") != null);
}
