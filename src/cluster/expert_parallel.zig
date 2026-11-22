//! Multi-Mac Expert-Parallel Coordinator & Peer Worker Runtime (Milestone M7,
//! capabilities X2 & X3, ADR-0003, ADR-0015).
//!
//! Architecture:
//! - `ExpertPartitionPlan`: partitions up to `1024` experts (covering `512+`
//!   `deepseek41` 552B MoE) across `1..16` ranks via `.contiguous_by_rank` or
//!   `.round_robin`, generating `ExpertSource` entries (`.local_wired_ram` for
//!   locally owned experts, `.thunderbolt_peer` for remote experts).
//! - `ExpertEvalBackend`: pluggable evaluation callback (`evalLocalSubsetFn`)
//!   supporting both the built-in zero-allocation exact CPU SwiGLU evaluator
//!   (`f16` / `bf16` / `q4_affine` / `q3_affine` / `q2_affine`) and future
//!   Phase 2 GPU `MoePipelines` adapters without touching the cluster runtime.
//! - `PeerWorkerServer` (X3): non-blocking `kqueue` server over `16 KiB`-aligned
//!   wired slabs serving `hello`, `eval_experts_req`, `ping`, and `shutdown`
//!   with zero steady-state heap allocations and zero payload copies.
//! - `ExpertParallelCoordinator` (X2): connects to `N - 1` `PeerWorkerServer`
//!   endpoints at startup (`hello` handshake validating `hidden_size` and exact
//!   expert ownership bitmap), splits routed assignments by owner rank into
//!   preallocated wired buckets, dispatches non-blocking remote RPCs BEFORE
//!   local evaluation so network + peer compute overlap with local compute, and
//!   collects partial `f32` outputs via a persistent `kqueue` loop.

const std = @import("std");
const page = @import("../mem/page.zig");
const source_mod = @import("../expert/source.zig");
const quant = @import("../quant/spec.zig");
const vmstat = @import("../platform/vmstat.zig");
const transport = @import("transport.zig");

pub const MAX_WORLD_SIZE: usize = 16;
pub const MAX_EXPERTS: usize = transport.MAX_EXPERTS_BITMAP_BITS; // 1024
pub const MAX_SERVER_CLIENTS: usize = 8;

/// Expert partitioning strategy across cluster ranks (`X2`).
pub const PartitionStrategy = enum(u8) {
    contiguous_by_rank = 0,
    round_robin = 1,
};

pub const PlanError = error{
    InvalidWorldSize,
    InvalidRank,
    InvalidLayerCount,
    InvalidExpertCount,
    ExpertOutOfRange,
    LayerOutOfRange,
    MissingLocalBytes,
    SliceLengthMismatch,
};

pub const EvalError = error{
    InvalidLayer,
    InvalidExpert,
    InvalidTokenIndex,
    InvalidShape,
    BufferTooSmall,
    UnsupportedDType,
    BackendFailed,
    ClockFailed,
};

pub const ClusterError = PlanError ||
    EvalError ||
    transport.TransportError ||
    transport.ProtocolError ||
    transport.SlabError ||
    error{
        InvalidConfig,
        PeerCountMismatch,
        DuplicatePeerRank,
        SelfInPeerList,
        PeerRankNotFound,
        HandshakeRejected,
        HandshakeMismatch,
        ExpertOwnershipMismatch,
        UnexpectedMessage,
        SequenceMismatch,
        RemoteEvalFailed,
        AlreadyRunning,
        ThreadSpawnFailed,
    };

pub const ExpertRange = struct {
    start: u16,
    end: u16,
};

/// Immutable expert-to-rank partitioning plan (`1..16` ranks, up to `1024` experts).
pub const ExpertPartitionPlan = struct {
    world_size: u16,
    local_rank: u16,
    num_layers: u16,
    num_experts: u16,
    strategy: PartitionStrategy,

    pub fn init(
        world_size: u16,
        local_rank: u16,
        num_layers: u16,
        num_experts: u16,
        strategy: PartitionStrategy,
    ) PlanError!ExpertPartitionPlan {
        if (world_size == 0 or world_size > MAX_WORLD_SIZE) return error.InvalidWorldSize;
        if (local_rank >= world_size) return error.InvalidRank;
        if (num_layers == 0) return error.InvalidLayerCount;
        if (num_experts == 0 or num_experts > MAX_EXPERTS) return error.InvalidExpertCount;
        return .{
            .world_size = world_size,
            .local_rank = local_rank,
            .num_layers = num_layers,
            .num_experts = num_experts,
            .strategy = strategy,
        };
    }

    /// Returns the rank owning `expert_id`.
    pub fn ownerRank(self: *const ExpertPartitionPlan, expert_id: u16) PlanError!u16 {
        if (expert_id >= self.num_experts) return error.ExpertOutOfRange;
        return switch (self.strategy) {
            .round_robin => expert_id % self.world_size,
            .contiguous_by_rank => blk: {
                const base = self.num_experts / self.world_size;
                const rem = self.num_experts % self.world_size;
                const split = rem * (base + 1);
                if (expert_id < split) {
                    break :blk expert_id / (base + 1);
                }
                break :blk rem + (expert_id - split) / base;
            },
        };
    }

    pub fn isLocal(self: *const ExpertPartitionPlan, expert_id: u16) bool {
        const owner = self.ownerRank(expert_id) catch return false;
        return owner == self.local_rank;
    }

    /// Exact `[start, end)` range for `.contiguous_by_rank`, or bounding range for `.round_robin`.
    pub fn rankExpertRange(self: *const ExpertPartitionPlan, rank: u16) PlanError!ExpertRange {
        if (rank >= self.world_size) return error.InvalidRank;
        return switch (self.strategy) {
            .contiguous_by_rank => blk: {
                const base = self.num_experts / self.world_size;
                const rem = self.num_experts % self.world_size;
                if (rank < rem) {
                    const start = rank * (base + 1);
                    break :blk .{ .start = start, .end = start + base + 1 };
                }
                const start = rem * (base + 1) + (rank - rem) * base;
                break :blk .{ .start = start, .end = start + base };
            },
            .round_robin => blk: {
                if (rank >= self.num_experts) break :blk .{ .start = 0, .end = 0 };
                var last: u16 = rank;
                while (last + self.world_size < self.num_experts) {
                    last += self.world_size;
                }
                break :blk .{ .start = rank, .end = last + 1 };
            },
        };
    }

    pub fn rankExpertCount(self: *const ExpertPartitionPlan, rank: u16) PlanError!u16 {
        if (rank >= self.world_size) return error.InvalidRank;
        var count: u16 = 0;
        for (0..self.num_experts) |e| {
            if ((try self.ownerRank(@intCast(e))) == rank) count += 1;
        }
        return count;
    }

    pub fn localExpertCount(self: *const ExpertPartitionPlan) u16 {
        return self.rankExpertCount(self.local_rank) catch unreachable;
    }

    /// Returns the 1024-bit ownership bitmap for `rank`.
    pub fn fillExpertBitmap(self: *const ExpertPartitionPlan, rank: u16) PlanError![transport.MAX_EXPERTS_BITMAP_WORDS]u64 {
        if (rank >= self.world_size) return error.InvalidRank;
        var payload: transport.HelloPayload = .{};
        for (0..self.num_experts) |e| {
            const eid: u16 = @intCast(e);
            if ((try self.ownerRank(eid)) == rank) {
                payload.setExpert(eid);
            }
        }
        return payload.expert_bitmap;
    }

    /// Constructs the ADR-0003 `ExpertSource` entry for `(layer, expert_id)`:
    /// `.local_wired_ram` when owned by `self.local_rank`, `.thunderbolt_peer` otherwise.
    pub fn buildExpertSource(
        self: *const ExpertPartitionPlan,
        layer: u16,
        expert_id: u16,
        local_bytes: ?[]align(page.APPLE_PAGE_SIZE) const u8,
        remote_offset: u64,
        remote_len: u64,
    ) (PlanError || page.ShareError || source_mod.PeerExpert.InitError)!source_mod.ExpertSource {
        if (layer >= self.num_layers) return error.LayerOutOfRange;
        const owner = try self.ownerRank(expert_id);
        if (owner == self.local_rank) {
            const bytes = local_bytes orelse return error.MissingLocalBytes;
            return .{ .local_wired_ram = try source_mod.WiredExpert.init(bytes) };
        }
        return .{
            .thunderbolt_peer = try source_mod.PeerExpert.init(
                owner,
                layer,
                expert_id,
                remote_offset,
                remote_len,
            ),
        };
    }

    /// Populates all `num_experts` `ExpertSource` entries for `layer` into `out_sources`.
    pub fn populateLayerSources(
        self: *const ExpertPartitionPlan,
        layer: u16,
        out_sources: []source_mod.ExpertSource,
        local_blobs: []const ?[]align(page.APPLE_PAGE_SIZE) const u8,
        remote_offsets: []const u64,
        remote_lens: []const u64,
    ) (PlanError || page.ShareError || source_mod.PeerExpert.InitError)!void {
        if (layer >= self.num_layers) return error.LayerOutOfRange;
        if (out_sources.len != self.num_experts or
            local_blobs.len != self.num_experts or
            remote_offsets.len != self.num_experts or
            remote_lens.len != self.num_experts)
        {
            return error.SliceLengthMismatch;
        }
        for (0..self.num_experts) |e| {
            out_sources[e] = try self.buildExpertSource(
                layer,
                @intCast(e),
                local_blobs[e],
                remote_offsets[e],
                remote_lens[e],
            );
        }
    }
};

// ---------------------------------------------------------------------------
// ExpertEvalBackend Interface & Built-in Zero-Alloc CPU SwiGLU Evaluator
// ---------------------------------------------------------------------------

/// Pluggable local-expert evaluation callback used by both `PeerWorkerServer`
/// and `ExpertParallelCoordinator`. Implementations must accumulate
/// `+= assignment.weight * SwiGLU_expert(hidden_f16[token_idx])` into
/// `accum_out_f32[token_idx * hidden_size ..]` and return compute elapsed `ns`.
pub const ExpertEvalBackend = struct {
    ctx: *anyopaque,
    evalLocalSubsetFn: *const fn (
        ctx: *anyopaque,
        layer: u16,
        num_tokens: u16,
        hidden_f16: []const f16,
        assignments: []const transport.Assignment,
        accum_out_f32: []f32,
    ) EvalError!u64,

    pub fn evalLocalSubset(
        self: ExpertEvalBackend,
        layer: u16,
        num_tokens: u16,
        hidden_f16: []const f16,
        assignments: []const transport.Assignment,
        accum_out_f32: []f32,
    ) EvalError!u64 {
        return self.evalLocalSubsetFn(
            self.ctx,
            layer,
            num_tokens,
            hidden_f16,
            assignments,
            accum_out_f32,
        );
    }
};

/// Byte offsets of one expert's `gate`, `up`, and `down` projections inside a
/// wired weight base buffer (matches `src/gpu/moe.zig:ExpertWeights`).
pub const CpuExpertWeights = extern struct {
    gate_weight_offset: u64 = 0,
    gate_sb_offset: u64 = 0,
    up_weight_offset: u64 = 0,
    up_sb_offset: u64 = 0,
    down_weight_offset: u64 = 0,
    down_sb_offset: u64 = 0,
};

pub const CpuExpertEntry = struct {
    dtype: quant.WeightDType = .q4_affine,
    weights: CpuExpertWeights = .{},
    present: bool = false,
};

/// Zero-allocation exact CPU SwiGLU MoE evaluator supporting all 5 aminfer
/// weight dtypes (`f16`, `bf16`, `q4_affine`, `q3_affine`, `q2_affine`) over a
/// single wired weight buffer and a preallocated `scratch_act_f32` vector.
pub const CpuMoeExpertStore = struct {
    num_layers: u16,
    num_experts: u16,
    hidden_size: u32,
    intermediate_size: u32,
    weights_base: []const u8,
    /// Flat `[num_layers * num_experts]` descriptor table.
    experts: []const CpuExpertEntry,
    /// Preallocated `[intermediate_size]` scratch buffer for `silu(gate) * up`.
    scratch_act_f32: []f32,

    pub fn init(
        num_layers: u16,
        num_experts: u16,
        hidden_size: u32,
        intermediate_size: u32,
        weights_base: []const u8,
        experts: []const CpuExpertEntry,
        scratch_act_f32: []f32,
    ) EvalError!CpuMoeExpertStore {
        if (num_layers == 0 or num_experts == 0 or hidden_size == 0 or intermediate_size == 0) {
            return error.InvalidShape;
        }
        const total_entries = std.math.mul(usize, num_layers, num_experts) catch return error.InvalidShape;
        if (experts.len < total_entries) return error.BufferTooSmall;
        if (scratch_act_f32.len < intermediate_size) return error.BufferTooSmall;
        return .{
            .num_layers = num_layers,
            .num_experts = num_experts,
            .hidden_size = hidden_size,
            .intermediate_size = intermediate_size,
            .weights_base = weights_base,
            .experts = experts,
            .scratch_act_f32 = scratch_act_f32,
        };
    }

    pub fn backend(self: *CpuMoeExpertStore) ExpertEvalBackend {
        return .{
            .ctx = self,
            .evalLocalSubsetFn = evalCallback,
        };
    }

    fn evalCallback(
        ctx: *anyopaque,
        layer: u16,
        num_tokens: u16,
        hidden_f16: []const f16,
        assignments: []const transport.Assignment,
        accum_out_f32: []f32,
    ) EvalError!u64 {
        const self: *CpuMoeExpertStore = @ptrCast(@alignCast(ctx));
        return self.evalLocalSubset(layer, num_tokens, hidden_f16, assignments, accum_out_f32);
    }

    pub fn evalLocalSubset(
        self: *CpuMoeExpertStore,
        layer: u16,
        num_tokens: u16,
        hidden_f16: []const f16,
        assignments: []const transport.Assignment,
        accum_out_f32: []f32,
    ) EvalError!u64 {
        if (layer >= self.num_layers) return error.InvalidLayer;
        const h: usize = self.hidden_size;
        const inter: usize = self.intermediate_size;
        const total_elems = std.math.mul(usize, num_tokens, h) catch return error.InvalidShape;
        if (hidden_f16.len < total_elems or accum_out_f32.len < total_elems) {
            return error.BufferTooSmall;
        }

        const t0 = vmstat.nowNs() catch return error.ClockFailed;
        const layer_base: usize = @as(usize, layer) * @as(usize, self.num_experts);
        const act = self.scratch_act_f32[0..inter];

        for (assignments) |a| {
            if (a.token_idx >= num_tokens) return error.InvalidTokenIndex;
            if (a.expert_id >= self.num_experts) return error.InvalidExpert;
            const entry = self.experts[layer_base + a.expert_id];
            if (!entry.present) return error.InvalidExpert;

            const x_row = hidden_f16[@as(usize, a.token_idx) * h ..][0..h];
            const out_row = accum_out_f32[@as(usize, a.token_idx) * h ..][0..h];

            switch (entry.dtype) {
                .f16 => try self.evalOneFloat(.f16, entry.weights, x_row, act, out_row, a.weight),
                .bf16 => try self.evalOneFloat(.bf16, entry.weights, x_row, act, out_row, a.weight),
                .q4_affine => try self.evalOneQuant(.{ .bits = .q4, .group_size = 64 }, entry.weights, x_row, act, out_row, a.weight),
                .q3_affine => try self.evalOneQuant(.{ .bits = .q3, .group_size = 64 }, entry.weights, x_row, act, out_row, a.weight),
                .q2_affine => try self.evalOneQuant(.{ .bits = .q2, .group_size = 64 }, entry.weights, x_row, act, out_row, a.weight),
            }
        }

        const t1 = vmstat.nowNs() catch return error.ClockFailed;
        return t1 -| t0;
    }

    fn sliceFloatMatrix(self: *const CpuMoeExpertStore, offset: u64, rows: usize, cols: usize) EvalError![]const u16 {
        const elems = std.math.mul(usize, rows, cols) catch return error.InvalidShape;
        const byte_len = std.math.mul(usize, elems, @sizeOf(u16)) catch return error.InvalidShape;
        const off = std.math.cast(usize, offset) orelse return error.BufferTooSmall;
        const end = std.math.add(usize, off, byte_len) catch return error.BufferTooSmall;
        if (end > self.weights_base.len) return error.BufferTooSmall;
        const raw = self.weights_base[off..end];
        if (!std.mem.isAligned(@intFromPtr(raw.ptr), @alignOf(u16))) return error.InvalidShape;
        return std.mem.bytesAsSlice(u16, @as([]align(@alignOf(u16)) const u8, @alignCast(raw)));
    }

    fn evalOneFloat(
        self: *const CpuMoeExpertStore,
        comptime dtype: quant.FloatDType,
        w: CpuExpertWeights,
        x_row: []const f16,
        act: []f32,
        out_row: []f32,
        weight: f32,
    ) EvalError!void {
        const h: usize = self.hidden_size;
        const inter: usize = self.intermediate_size;
        const w_gate = try self.sliceFloatMatrix(w.gate_weight_offset, inter, h);
        const w_up = try self.sliceFloatMatrix(w.up_weight_offset, inter, h);
        const w_down = try self.sliceFloatMatrix(w.down_weight_offset, h, inter);

        for (0..inter) |i| {
            const g_row = w_gate[i * h ..][0..h];
            const u_row = w_up[i * h ..][0..h];
            var g_acc: f32 = 0.0;
            var u_acc: f32 = 0.0;
            for (0..h) |k| {
                const xv: f32 = @floatCast(x_row[k]);
                g_acc += xv * quant.floatBitsToF32(dtype, g_row[k]);
                u_acc += xv * quant.floatBitsToF32(dtype, u_row[k]);
            }
            const silu_g = g_acc / (1.0 + @exp(-g_acc));
            act[i] = silu_g * u_acc;
        }

        for (0..h) |hi| {
            const d_row = w_down[hi * inter ..][0..inter];
            var acc: f32 = 0.0;
            for (0..inter) |i| {
                acc += act[i] * quant.floatBitsToF32(dtype, d_row[i]);
            }
            out_row[hi] += weight * acc;
        }
    }

    fn sliceQuantMatrix(
        self: *const CpuMoeExpertStore,
        comptime Ops: type,
        q_offset: u64,
        sb_offset: u64,
        rows: usize,
        cols: usize,
    ) EvalError!struct { q: []const u8, sb: []const quant.ScaleBias, row_bytes: usize, groups: usize } {
        const groups = Ops.groupsPerRow(cols) catch return error.InvalidShape;
        const row_bytes = Ops.rowWeightBytes(cols) catch return error.InvalidShape;
        const q_len = Ops.packedWeightBytes(rows, cols) catch return error.InvalidShape;
        const sb_len = Ops.scaleBiasBytes(rows, cols) catch return error.InvalidShape;

        const q_off = std.math.cast(usize, q_offset) orelse return error.BufferTooSmall;
        const q_end = std.math.add(usize, q_off, q_len) catch return error.BufferTooSmall;
        if (q_end > self.weights_base.len) return error.BufferTooSmall;

        const sb_off = std.math.cast(usize, sb_offset) orelse return error.BufferTooSmall;
        const sb_end = std.math.add(usize, sb_off, sb_len) catch return error.BufferTooSmall;
        if (sb_end > self.weights_base.len) return error.BufferTooSmall;

        const sb_raw = self.weights_base[sb_off..sb_end];
        if (!std.mem.isAligned(@intFromPtr(sb_raw.ptr), @alignOf(quant.ScaleBias))) {
            return error.InvalidShape;
        }
        const sb = std.mem.bytesAsSlice(
            quant.ScaleBias,
            @as([]align(@alignOf(quant.ScaleBias)) const u8, @alignCast(sb_raw)),
        );
        return .{
            .q = self.weights_base[q_off..q_end],
            .sb = sb,
            .row_bytes = row_bytes,
            .groups = groups,
        };
    }

    fn evalOneQuant(
        self: *const CpuMoeExpertStore,
        comptime spec: quant.QuantSpec,
        w: CpuExpertWeights,
        x_row: []const f16,
        act: []f32,
        out_row: []f32,
        weight: f32,
    ) EvalError!void {
        const Ops = quant.QuantOps(spec);
        const gs: usize = spec.group_size;
        const gb: usize = Ops.groupWeightBytes();
        const h: usize = self.hidden_size;
        const inter: usize = self.intermediate_size;

        const gate_m = try self.sliceQuantMatrix(Ops, w.gate_weight_offset, w.gate_sb_offset, inter, h);
        const up_m = try self.sliceQuantMatrix(Ops, w.up_weight_offset, w.up_sb_offset, inter, h);
        const down_m = try self.sliceQuantMatrix(Ops, w.down_weight_offset, w.down_sb_offset, h, inter);

        var deq_g: [gs]f32 = undefined;
        var deq_u: [gs]f32 = undefined;
        for (0..inter) |i| {
            const g_qrow = gate_m.q[i * gate_m.row_bytes ..][0..gate_m.row_bytes];
            const g_sbrow = gate_m.sb[i * gate_m.groups ..][0..gate_m.groups];
            const u_qrow = up_m.q[i * up_m.row_bytes ..][0..up_m.row_bytes];
            const u_sbrow = up_m.sb[i * up_m.groups ..][0..up_m.groups];
            var g_acc: f32 = 0.0;
            var u_acc: f32 = 0.0;
            for (0..gate_m.groups) |g| {
                Ops.dequantizeGroup(g_qrow[g * gb ..][0..gb], g_sbrow[g], &deq_g);
                Ops.dequantizeGroup(u_qrow[g * gb ..][0..gb], u_sbrow[g], &deq_u);
                const x_grp = x_row[g * gs ..][0..gs];
                for (0..gs) |j| {
                    const xv: f32 = @floatCast(x_grp[j]);
                    g_acc += xv * deq_g[j];
                    u_acc += xv * deq_u[j];
                }
            }
            const silu_g = g_acc / (1.0 + @exp(-g_acc));
            act[i] = silu_g * u_acc;
        }

        var deq_d: [gs]f32 = undefined;
        for (0..h) |hi| {
            const d_qrow = down_m.q[hi * down_m.row_bytes ..][0..down_m.row_bytes];
            const d_sbrow = down_m.sb[hi * down_m.groups ..][0..down_m.groups];
            var acc: f32 = 0.0;
            for (0..down_m.groups) |g| {
                Ops.dequantizeGroup(d_qrow[g * gb ..][0..gb], d_sbrow[g], &deq_d);
                const a_grp = act[g * gs ..][0..gs];
                for (0..gs) |j| {
                    acc += a_grp[j] * deq_d[j];
                }
            }
            out_row[hi] += weight * acc;
        }
    }
};

// ---------------------------------------------------------------------------
// PeerWorkerServer (X3) — zero-allocation kqueue RPC server
// ---------------------------------------------------------------------------

pub const PeerWorkerConfig = struct {
    host: [4]u8 = .{ 127, 0, 0, 1 },
    port: u16 = 0,
    backlog: u31 = 16,
    transport_kind: transport.TransportKind = .auto,
    plan: ExpertPartitionPlan,
    hidden_size: u32,
    max_tokens: u16 = 64,
    max_assignments: u32 = 1024,
    max_clients: u16 = 4,
    timeout_ms: u32 = transport.DEFAULT_TIMEOUT_MS,
    backend: ExpertEvalBackend,
    /// Optional `16 KiB`-aligned slice carved from an external `WiredArena`.
    /// When `null`, `PeerWorkerServer` maps and locks its own `WiredRegion`.
    external_slab: ?[]align(page.APPLE_PAGE_SIZE) u8 = null,
};

pub const WorkerStats = struct {
    requests_served: u64 = 0,
    assignments_evaluated: u64 = 0,
    bytes_rx: u64 = 0,
    bytes_tx: u64 = 0,
    compute_ns: u64 = 0,
};

const WorkerClientSlot = struct {
    active: bool = false,
    chan: transport.FrameChannel = undefined,
};

const WORKER_UDATA_LISTEN: u64 = 0xFFFF_0001;
const WORKER_UDATA_STOP: u64 = 0xFFFF_0002;
const WORKER_IDENT_STOP: u64 = 0xCAFE_0001;

/// Non-blocking `kqueue` peer worker serving `hello`, `eval_experts_req`,
/// `ping`, and `shutdown` over `16 KiB`-aligned wired slabs (`0` steady-state
/// heap allocations).
pub const PeerWorkerServer = struct {
    config: PeerWorkerConfig,
    resolved_transport: transport.ResolvedTransport,
    listener: transport.TcpListener,
    kq_fd: std.c.fd_t,
    slab: transport.WiredSlab,
    slots: [MAX_SERVER_CLIENTS]WorkerClientSlot,
    slot_count: usize,
    stop_requested: std.atomic.Value(bool),
    running: std.atomic.Value(bool),
    thread: ?std.Thread,

    // Thread-safe atomic stats counters.
    stat_requests_served: std.atomic.Value(u64),
    stat_assignments_evaluated: std.atomic.Value(u64),
    stat_bytes_rx: std.atomic.Value(u64),
    stat_bytes_tx: std.atomic.Value(u64),
    stat_compute_ns: std.atomic.Value(u64),

    pub fn perSlotTxBytes(max_tokens: u16, hidden_size: u32) ClusterError!usize {
        const resp_payload = try transport.evalRespPayloadBytes(max_tokens, hidden_size);
        const min_payload = @max(resp_payload, transport.HelloPayload.WIRE_BYTES);
        const frame = try std.math.add(usize, transport.HEADER_BYTES, min_payload);
        return try page.alignUp(frame);
    }

    pub fn perSlotRxBytes(max_tokens: u16, hidden_size: u32, max_assignments: u32) ClusterError!usize {
        const req_payload = try transport.evalReqPayloadBytes(max_tokens, hidden_size, max_assignments);
        const min_payload = @max(req_payload, transport.HelloPayload.WIRE_BYTES);
        const frame = try std.math.add(usize, transport.HEADER_BYTES, min_payload);
        return try page.alignUp(frame);
    }

    pub fn requiredSlabBytes(config: PeerWorkerConfig) ClusterError!usize {
        if (config.max_clients == 0 or config.max_clients > MAX_SERVER_CLIENTS) return error.InvalidConfig;
        if (config.hidden_size == 0 or config.max_tokens == 0 or config.max_assignments == 0) {
            return error.InvalidConfig;
        }
        const tx_each = try perSlotTxBytes(config.max_tokens, config.hidden_size);
        const rx_each = try perSlotRxBytes(config.max_tokens, config.hidden_size, config.max_assignments);
        const per_client = try std.math.add(usize, tx_each, rx_each);
        return try std.math.mul(usize, per_client, config.max_clients);
    }

    pub fn init(config: PeerWorkerConfig) ClusterError!PeerWorkerServer {
        const resolved = try transport.selectTransport(config.transport_kind);
        const total_slab = try requiredSlabBytes(config);
        const tx_each = try perSlotTxBytes(config.max_tokens, config.hidden_size);
        const rx_each = try perSlotRxBytes(config.max_tokens, config.hidden_size, config.max_assignments);

        var slab = if (config.external_slab) |ext|
            try transport.WiredSlab.initBorrowed(ext)
        else
            try transport.WiredSlab.initOwned(total_slab);
        errdefer slab.deinit();

        var slots: [MAX_SERVER_CLIENTS]WorkerClientSlot = @splat(.{});
        for (0..config.max_clients) |i| {
            const tx_buf = try slab.carvePages(tx_each);
            const rx_buf = try slab.carvePages(rx_each);
            slots[i] = .{
                .active = false,
                .chan = try transport.FrameChannel.init(-1, tx_buf, rx_buf),
            };
        }

        var listener = try transport.TcpListener.bind(.{
            .host = config.host,
            .port = config.port,
        }, config.backlog);
        errdefer listener.close();

        const kq_fd = try transport.kqueueCreate();
        errdefer _ = std.c.close(kq_fd);
        try transport.kqueueRegisterUser(kq_fd, WORKER_IDENT_STOP, WORKER_UDATA_STOP);
        try transport.kqueueSetFilter(
            kq_fd,
            listener.fd,
            std.c.EVFILT.READ,
            std.c.EV.ADD | std.c.EV.ENABLE,
            WORKER_UDATA_LISTEN,
        );

        return .{
            .config = config,
            .resolved_transport = resolved,
            .listener = listener,
            .kq_fd = kq_fd,
            .slab = slab,
            .slots = slots,
            .slot_count = config.max_clients,
            .stop_requested = .init(false),
            .running = .init(false),
            .thread = null,
            .stat_requests_served = .init(0),
            .stat_assignments_evaluated = .init(0),
            .stat_bytes_rx = .init(0),
            .stat_bytes_tx = .init(0),
            .stat_compute_ns = .init(0),
        };
    }

    pub fn boundPort(self: *const PeerWorkerServer) u16 {
        return self.listener.bound_port;
    }

    pub fn stats(self: *const PeerWorkerServer) WorkerStats {
        return .{
            .requests_served = self.stat_requests_served.load(.acquire),
            .assignments_evaluated = self.stat_assignments_evaluated.load(.acquire),
            .bytes_rx = self.stat_bytes_rx.load(.acquire),
            .bytes_tx = self.stat_bytes_tx.load(.acquire),
            .compute_ns = self.stat_compute_ns.load(.acquire),
        };
    }

    /// Spawns a background thread driving the `kqueue` worker loop.
    /// `self` must remain at a stable memory address until `stop()` / `deinit()`.
    pub fn start(self: *PeerWorkerServer) ClusterError!void {
        if (self.thread != null) return error.AlreadyRunning;
        self.stop_requested.store(false, .release);
        const Runner = struct {
            fn entry(srv: *PeerWorkerServer) void {
                srv.run();
            }
        };
        self.thread = std.Thread.spawn(.{}, Runner.entry, .{self}) catch return error.ThreadSpawnFailed;
    }

    /// Signals the `kqueue` loop to terminate and joins the background thread.
    pub fn stop(self: *PeerWorkerServer) void {
        self.stop_requested.store(true, .release);
        if (self.kq_fd >= 0) {
            transport.kqueueTriggerUser(self.kq_fd, WORKER_IDENT_STOP, WORKER_UDATA_STOP);
        }
        if (self.thread) |t| {
            t.join();
            self.thread = null;
        }
    }

    pub fn deinit(self: *PeerWorkerServer) void {
        self.stop();
        for (0..self.slot_count) |i| {
            self.closeSlot(i);
        }
        self.listener.close();
        if (self.kq_fd >= 0) {
            _ = std.c.close(self.kq_fd);
            self.kq_fd = -1;
        }
        self.slab.deinit();
    }

    /// Runs the zero-allocation `kqueue` server loop until stopped or a `shutdown`
    /// control frame is received.
    pub fn run(self: *PeerWorkerServer) void {
        self.running.store(true, .release);
        defer self.running.store(false, .release);

        var events: [MAX_SERVER_CLIENTS * 2 + 4]std.c.kevent64_s = undefined;
        while (!self.stop_requested.load(.acquire)) {
            const nev = transport.kqueueWait(self.kq_fd, &events, null) catch break;
            for (events[0..nev]) |ev| {
                if (ev.udata == WORKER_UDATA_STOP) {
                    if (self.stop_requested.load(.acquire)) break;
                    continue;
                }
                if (ev.udata == WORKER_UDATA_LISTEN) {
                    self.acceptAllPending();
                    continue;
                }
                const idx: usize = @intCast(ev.udata);
                if (idx >= self.slot_count or !self.slots[idx].active) continue;

                if (ev.filter == std.c.EVFILT.WRITE) {
                    self.driveClientWrite(idx);
                    if (self.slots[idx].active and !self.slots[idx].chan.hasPendingTx()) {
                        self.driveClientRead(idx);
                    }
                } else if (ev.filter == std.c.EVFILT.READ) {
                    if (!self.slots[idx].chan.hasPendingTx()) {
                        self.driveClientRead(idx);
                    }
                }
                if (self.stop_requested.load(.acquire)) break;
            }
        }
    }

    fn acceptAllPending(self: *PeerWorkerServer) void {
        while (true) {
            const client_fd_opt = self.listener.acceptNonBlocking() catch return;
            const client_fd = client_fd_opt orelse return;
            const free_idx = self.findFreeSlot() orelse {
                _ = std.c.close(client_fd);
                continue;
            };
            transport.kqueueSetFilter(
                self.kq_fd,
                client_fd,
                std.c.EVFILT.READ,
                std.c.EV.ADD | std.c.EV.ENABLE,
                @intCast(free_idx),
            ) catch {
                _ = std.c.close(client_fd);
                continue;
            };
            self.slots[free_idx].active = true;
            self.slots[free_idx].chan.fd = client_fd;
            self.slots[free_idx].chan.resetState();
        }
    }

    fn findFreeSlot(self: *const PeerWorkerServer) ?usize {
        for (0..self.slot_count) |i| {
            if (!self.slots[i].active) return i;
        }
        return null;
    }

    fn closeSlot(self: *PeerWorkerServer, idx: usize) void {
        if (self.slots[idx].chan.fd >= 0) {
            self.slots[idx].chan.close();
        }
        self.slots[idx].active = false;
    }

    fn armWrite(self: *PeerWorkerServer, idx: usize) void {
        if (self.slots[idx].chan.write_armed) return;
        transport.kqueueSetFilter(
            self.kq_fd,
            self.slots[idx].chan.fd,
            std.c.EVFILT.WRITE,
            std.c.EV.ADD | std.c.EV.ENABLE,
            @intCast(idx),
        ) catch {
            self.closeSlot(idx);
            return;
        };
        self.slots[idx].chan.write_armed = true;
    }

    fn disarmWrite(self: *PeerWorkerServer, idx: usize) void {
        if (!self.slots[idx].chan.write_armed) return;
        transport.kqueueSetFilter(
            self.kq_fd,
            self.slots[idx].chan.fd,
            std.c.EVFILT.WRITE,
            std.c.EV.DELETE,
            @intCast(idx),
        ) catch {
            // If socket already closed or filter absent, force clear flag.
            self.slots[idx].chan.write_armed = false;
            return;
        };
        self.slots[idx].chan.write_armed = false;
    }

    fn driveClientWrite(self: *PeerWorkerServer, idx: usize) void {
        const step = self.slots[idx].chan.stepSend() catch {
            self.closeSlot(idx);
            return;
        };
        switch (step) {
            .complete => |n| {
                if (n > 0) _ = self.stat_bytes_tx.fetchAdd(n, .monotonic);
                self.disarmWrite(idx);
            },
            .would_block => |n| {
                if (n > 0) _ = self.stat_bytes_tx.fetchAdd(n, .monotonic);
                self.armWrite(idx);
            },
        }
    }

    fn driveClientRead(self: *PeerWorkerServer, idx: usize) void {
        while (self.slots[idx].active and !self.slots[idx].chan.hasPendingTx()) {
            const step = self.slots[idx].chan.stepRecv() catch {
                self.closeSlot(idx);
                return;
            };
            switch (step) {
                .would_block => |n| {
                    if (n > 0) _ = self.stat_bytes_rx.fetchAdd(n, .monotonic);
                    return;
                },
                .peer_closed => {
                    self.closeSlot(idx);
                    return;
                },
                .frame_ready => |frame| {
                    if (frame.bytes_read > 0) {
                        _ = self.stat_bytes_rx.fetchAdd(frame.bytes_read, .monotonic);
                    }
                    self.handleFrame(idx, frame);
                    if (!self.slots[idx].active or self.stop_requested.load(.acquire)) return;
                    self.slots[idx].chan.resetRx();
                    if (self.slots[idx].chan.hasPendingTx()) {
                        self.driveClientWrite(idx);
                    }
                },
            }
        }
    }

    fn handleFrame(self: *PeerWorkerServer, idx: usize, frame: transport.ReceivedFrame) void {
        const slot = &self.slots[idx];
        switch (frame.header.msg_type) {
            .hello_req => {
                const range = self.config.plan.rankExpertRange(self.config.plan.local_rank) catch {
                    self.closeSlot(idx);
                    return;
                };
                const bitmap = self.config.plan.fillExpertBitmap(self.config.plan.local_rank) catch {
                    self.closeSlot(idx);
                    return;
                };
                var resp: transport.HelloPayload = .{
                    .expert_bitmap = bitmap,
                    .hidden_size = self.config.hidden_size,
                    .rank = self.config.plan.local_rank,
                    .world_size = self.config.plan.world_size,
                    .num_layers = self.config.plan.num_layers,
                    .num_experts = self.config.plan.num_experts,
                    .expert_range_start = range.start,
                    .expert_range_end = range.end,
                    .transport_kind = @backingInt(self.resolved_transport.kind),
                    .status = 0,
                };
                const dst = slot.chan.preparePayload(transport.HelloPayload.WIRE_BYTES) catch {
                    self.closeSlot(idx);
                    return;
                };
                resp.encode(dst[0..transport.HelloPayload.WIRE_BYTES]);
                _ = slot.chan.commitFrame(.hello_resp, 0, frame.header.seq_id, transport.HelloPayload.WIRE_BYTES);
            },
            .ping => {
                if (frame.payload.len != 8) {
                    self.closeSlot(idx);
                    return;
                }
                const dst = slot.chan.preparePayload(8) catch {
                    self.closeSlot(idx);
                    return;
                };
                @memcpy(dst[0..8], frame.payload[0..8]);
                _ = slot.chan.commitFrame(.pong, 0, frame.header.seq_id, 8);
            },
            .shutdown => {
                self.stop_requested.store(true, .release);
                self.closeSlot(idx);
            },
            .eval_experts_req => {
                self.handleEvalExperts(idx, frame);
            },
            .hello_resp, .eval_experts_resp, .pong => {
                self.closeSlot(idx);
            },
        }
    }

    fn queueEvalError(
        self: *PeerWorkerServer,
        idx: usize,
        seq_id: u64,
        status: transport.EvalStatus,
    ) void {
        const slot = &self.slots[idx];
        const dst = slot.chan.preparePayload(transport.EvalExpertsRespHeader.WIRE_BYTES) catch {
            self.closeSlot(idx);
            return;
        };
        const hdr: transport.EvalExpertsRespHeader = .{
            .status = @backingInt(status),
            .num_tokens = 0,
            .hidden_size = self.config.hidden_size,
            .assignments_evaluated = 0,
            .peer_compute_ns = 0,
        };
        hdr.encode(dst[0..transport.EvalExpertsRespHeader.WIRE_BYTES]);
        _ = slot.chan.commitFrame(.eval_experts_resp, 0, seq_id, transport.EvalExpertsRespHeader.WIRE_BYTES);
    }

    fn handleEvalExperts(self: *PeerWorkerServer, idx: usize, frame: transport.ReceivedFrame) void {
        const view = transport.decodeEvalReqView(frame.payload) catch {
            self.queueEvalError(idx, frame.header.seq_id, .invalid_shape);
            return;
        };
        const req = view.header;
        if (req.layer_idx >= self.config.plan.num_layers) {
            self.queueEvalError(idx, frame.header.seq_id, .invalid_layer);
            return;
        }
        if (req.num_tokens == 0 or req.num_tokens > self.config.max_tokens or
            req.hidden_size != self.config.hidden_size or
            req.num_assignments > self.config.max_assignments)
        {
            self.queueEvalError(idx, frame.header.seq_id, .invalid_shape);
            return;
        }
        for (view.assignments) |a| {
            if (a.token_idx >= req.num_tokens) {
                self.queueEvalError(idx, frame.header.seq_id, .invalid_shape);
                return;
            }
            if (!self.config.plan.isLocal(a.expert_id)) {
                self.queueEvalError(idx, frame.header.seq_id, .invalid_expert);
                return;
            }
        }

        const resp_bytes = transport.evalRespPayloadBytes(req.num_tokens, req.hidden_size) catch {
            self.queueEvalError(idx, frame.header.seq_id, .buffer_too_small);
            return;
        };
        const slot = &self.slots[idx];
        const dst = slot.chan.preparePayload(resp_bytes) catch {
            self.queueEvalError(idx, frame.header.seq_id, .buffer_too_small);
            return;
        };

        // Zero-copy output slice directly inside `tx_buf` after the 24-byte response header.
        const out_raw: []align(4) u8 = @alignCast(dst[transport.EvalExpertsRespHeader.WIRE_BYTES..resp_bytes]);
        const out_f32 = std.mem.bytesAsSlice(f32, out_raw);
        @memset(out_f32, 0.0);

        const compute_ns = self.config.backend.evalLocalSubset(
            req.layer_idx,
            req.num_tokens,
            view.hidden_f16,
            view.assignments,
            out_f32,
        ) catch |err| {
            const status: transport.EvalStatus = switch (err) {
                error.InvalidLayer => .invalid_layer,
                error.InvalidExpert => .invalid_expert,
                error.InvalidTokenIndex, error.InvalidShape => .invalid_shape,
                error.BufferTooSmall => .buffer_too_small,
                error.UnsupportedDType, error.BackendFailed, error.ClockFailed => .backend_failed,
            };
            self.queueEvalError(idx, frame.header.seq_id, status);
            return;
        };

        const resp_hdr: transport.EvalExpertsRespHeader = .{
            .status = @backingInt(transport.EvalStatus.ok),
            .num_tokens = req.num_tokens,
            .hidden_size = req.hidden_size,
            .assignments_evaluated = req.num_assignments,
            .peer_compute_ns = compute_ns,
        };
        resp_hdr.encode(dst[0..transport.EvalExpertsRespHeader.WIRE_BYTES]);
        _ = slot.chan.commitFrame(.eval_experts_resp, 0, frame.header.seq_id, resp_bytes);

        _ = self.stat_requests_served.fetchAdd(1, .monotonic);
        _ = self.stat_assignments_evaluated.fetchAdd(req.num_assignments, .monotonic);
        _ = self.stat_compute_ns.fetchAdd(compute_ns, .monotonic);
    }
};

// ---------------------------------------------------------------------------
// ExpertParallelCoordinator (X2) — overlapped multi-peer dispatch
// ---------------------------------------------------------------------------

pub const PeerEndpoint = struct {
    rank: u16,
    host: [4]u8 = .{ 127, 0, 0, 1 },
    port: u16,
};

pub const CoordinatorConfig = struct {
    plan: ExpertPartitionPlan,
    hidden_size: u32,
    max_tokens: u16 = 64,
    max_assignments: u32 = 1024,
    transport_kind: transport.TransportKind = .auto,
    timeout_ms: u32 = transport.DEFAULT_TIMEOUT_MS,
    /// Must list all `plan.world_size - 1` remote peer endpoints.
    peers: []const PeerEndpoint,
    local_backend: ExpertEvalBackend,
    /// Optional `16 KiB`-aligned slice carved from an external `WiredArena`.
    /// When `null`, `ExpertParallelCoordinator` maps and locks its own `WiredRegion`.
    external_slab: ?[]align(page.APPLE_PAGE_SIZE) u8 = null,
};

pub const DispatchStats = struct {
    remote_rpc_ns: u64 = 0,
    local_compute_ns: u64 = 0,
    max_peer_compute_ns: u64 = 0,
    local_assignments: u32 = 0,
    remote_assignments: u32 = 0,
    active_remote_peers: u16 = 0,
    bytes_tx: u64 = 0,
    bytes_rx: u64 = 0,
    bytes_transferred: u64 = 0,
};

const RemotePeerSlot = struct {
    rank: u16 = 0,
    connected: bool = false,
    chan: transport.FrameChannel = undefined,
    remote_bitmap: [transport.MAX_EXPERTS_BITMAP_WORDS]u64 = @splat(0),
};

/// Multi-Mac expert-parallel coordinator (`X2`).
/// Preallocates all per-peer `16 KiB`-aligned TX/RX slabs and per-rank
/// assignment split buckets once at startup so steady-state `dispatchLayerExperts`
/// performs zero heap allocations while overlapping remote RPCs with local compute.
pub const ExpertParallelCoordinator = struct {
    config: CoordinatorConfig,
    resolved_transport: transport.ResolvedTransport,
    slab: transport.WiredSlab,
    kq_fd: std.c.fd_t,
    peer_slots: [MAX_WORLD_SIZE]RemotePeerSlot,
    peer_count: usize,
    /// Maps global rank `0..world_size-1` -> index in `peer_slots`, or `null` for `local_rank`.
    rank_to_peer_idx: [MAX_WORLD_SIZE]?u8,
    /// Preallocated per-rank assignment buckets (`[world_size]` slices of length `max_assignments`).
    rank_buckets: [MAX_WORLD_SIZE][]transport.Assignment,
    /// Preallocated `f32` accumulation buffer for `dispatchLayerExpertsToF16`.
    scratch_out_f32: []f32,
    next_seq_id: u64,
    total_dispatches: u64,
    total_remote_rpc_ns: u64,
    total_local_compute_ns: u64,
    total_remote_assignments: u64,
    total_bytes_transferred: u64,

    pub fn perPeerTxBytes(max_tokens: u16, hidden_size: u32, max_assignments: u32) ClusterError!usize {
        const req_payload = try transport.evalReqPayloadBytes(max_tokens, hidden_size, max_assignments);
        const min_payload = @max(req_payload, transport.HelloPayload.WIRE_BYTES);
        const frame = try std.math.add(usize, transport.HEADER_BYTES, min_payload);
        return try page.alignUp(frame);
    }

    pub fn perPeerRxBytes(max_tokens: u16, hidden_size: u32) ClusterError!usize {
        const resp_payload = try transport.evalRespPayloadBytes(max_tokens, hidden_size);
        const min_payload = @max(resp_payload, transport.HelloPayload.WIRE_BYTES);
        const frame = try std.math.add(usize, transport.HEADER_BYTES, min_payload);
        return try page.alignUp(frame);
    }

    pub fn requiredSlabBytes(config: CoordinatorConfig) ClusterError!usize {
        if (config.hidden_size == 0 or config.max_tokens == 0 or config.max_assignments == 0) {
            return error.InvalidConfig;
        }
        const remote_count: usize = config.plan.world_size - 1;
        const tx_each = try perPeerTxBytes(config.max_tokens, config.hidden_size, config.max_assignments);
        const rx_each = try perPeerRxBytes(config.max_tokens, config.hidden_size);
        const peer_pair = try std.math.add(usize, tx_each, rx_each);
        const peers_total = try std.math.mul(usize, peer_pair, remote_count);

        const bucket_each = try std.math.mul(usize, config.max_assignments, @sizeOf(transport.Assignment));
        const buckets_total = try page.alignUp(try std.math.mul(usize, bucket_each, config.plan.world_size));

        const scratch_elems = try std.math.mul(usize, config.max_tokens, config.hidden_size);
        const scratch_bytes = try page.alignUp(try std.math.mul(usize, scratch_elems, @sizeOf(f32)));

        const sum1 = try std.math.add(usize, peers_total, buckets_total);
        return try std.math.add(usize, sum1, scratch_bytes);
    }

    pub fn init(config: CoordinatorConfig) ClusterError!ExpertParallelCoordinator {
        const resolved = try transport.selectTransport(config.transport_kind);
        const expected_peers: usize = config.plan.world_size - 1;
        if (config.peers.len != expected_peers) return error.PeerCountMismatch;
        if (config.hidden_size == 0 or config.max_tokens == 0 or config.max_assignments == 0) {
            return error.InvalidConfig;
        }

        // Validate peer ranks before allocating or opening sockets.
        var seen_rank: [MAX_WORLD_SIZE]bool = @splat(false);
        seen_rank[config.plan.local_rank] = true;
        for (config.peers) |ep| {
            if (ep.rank >= config.plan.world_size) return error.InvalidRank;
            if (ep.rank == config.plan.local_rank) return error.SelfInPeerList;
            if (seen_rank[ep.rank]) return error.DuplicatePeerRank;
            seen_rank[ep.rank] = true;
        }

        const total_slab = try requiredSlabBytes(config);
        const tx_each = try perPeerTxBytes(config.max_tokens, config.hidden_size, config.max_assignments);
        const rx_each = try perPeerRxBytes(config.max_tokens, config.hidden_size);

        var slab = if (config.external_slab) |ext|
            try transport.WiredSlab.initBorrowed(ext)
        else
            try transport.WiredSlab.initOwned(total_slab);
        errdefer slab.deinit();

        var peer_slots: [MAX_WORLD_SIZE]RemotePeerSlot = @splat(.{});
        var rank_to_peer_idx: [MAX_WORLD_SIZE]?u8 = @splat(null);
        for (config.peers, 0..) |ep, i| {
            const tx_buf = try slab.carvePages(tx_each);
            const rx_buf = try slab.carvePages(rx_each);
            peer_slots[i] = .{
                .rank = ep.rank,
                .connected = false,
                .chan = try transport.FrameChannel.init(-1, tx_buf, rx_buf),
            };
            rank_to_peer_idx[ep.rank] = @intCast(i);
        }

        // Carve per-rank assignment split buckets + scratch f32 output buffer.
        var rank_buckets: [MAX_WORLD_SIZE][]transport.Assignment = @splat(&.{});
        const bucket_bytes = try std.math.mul(usize, config.max_assignments, @sizeOf(transport.Assignment));
        for (0..config.plan.world_size) |r| {
            const raw = try slab.carveAligned(bucket_bytes, @alignOf(transport.Assignment));
            rank_buckets[r] = std.mem.bytesAsSlice(
                transport.Assignment,
                @as([]align(@alignOf(transport.Assignment)) u8, @alignCast(raw)),
            );
        }

        const scratch_elems = try std.math.mul(usize, config.max_tokens, config.hidden_size);
        const scratch_raw = try slab.carvePages(try std.math.mul(usize, scratch_elems, @sizeOf(f32)));
        const scratch_out_f32 = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(scratch_raw)))[0..scratch_elems];

        const kq_fd = try transport.kqueueCreate();
        errdefer _ = std.c.close(kq_fd);

        errdefer {
            for (0..expected_peers) |i| {
                if (peer_slots[i].connected) peer_slots[i].chan.close();
            }
        }

        // Use a dedicated handshake kqueue so `kq_fd` only carries persistent EVFILT_READ filters.
        const handshake_kq = try transport.kqueueCreate();
        defer _ = std.c.close(handshake_kq);

        var next_seq: u64 = 1;
        const local_bitmap = try config.plan.fillExpertBitmap(config.plan.local_rank);
        const local_range = try config.plan.rankExpertRange(config.plan.local_rank);

        for (config.peers, 0..) |ep, i| {
            const fd = try transport.connectNonBlocking(.{
                .host = ep.host,
                .port = ep.port,
            }, config.timeout_ms);
            peer_slots[i].chan.fd = fd;
            peer_slots[i].chan.resetState();
            peer_slots[i].connected = true;

            // Send `hello_req` and validate `hello_resp`.
            const seq = next_seq;
            next_seq += 1;
            var hello_req: transport.HelloPayload = .{
                .expert_bitmap = local_bitmap,
                .hidden_size = config.hidden_size,
                .rank = config.plan.local_rank,
                .world_size = config.plan.world_size,
                .num_layers = config.plan.num_layers,
                .num_experts = config.plan.num_experts,
                .expert_range_start = local_range.start,
                .expert_range_end = local_range.end,
                .transport_kind = @backingInt(resolved.kind),
                .status = 0,
            };
            const dst = try peer_slots[i].chan.preparePayload(transport.HelloPayload.WIRE_BYTES);
            hello_req.encode(dst[0..transport.HelloPayload.WIRE_BYTES]);
            _ = peer_slots[i].chan.commitFrame(.hello_req, 0, seq, transport.HelloPayload.WIRE_BYTES);

            _ = try peer_slots[i].chan.sendFrameSync(handshake_kq, config.timeout_ms);
            const resp_frame = try peer_slots[i].chan.recvFrameSync(handshake_kq, config.timeout_ms);
            defer peer_slots[i].chan.resetRx();

            if (resp_frame.header.msg_type != .hello_resp) return error.UnexpectedMessage;
            if (resp_frame.header.seq_id != seq) return error.SequenceMismatch;
            const hello_resp = try transport.HelloPayload.decode(resp_frame.payload);
            if (hello_resp.status != 0) return error.HandshakeRejected;
            if (hello_resp.rank != ep.rank or
                hello_resp.world_size != config.plan.world_size or
                hello_resp.num_layers != config.plan.num_layers or
                hello_resp.num_experts != config.plan.num_experts or
                hello_resp.hidden_size != config.hidden_size)
            {
                return error.HandshakeMismatch;
            }
            const expected_bitmap = try config.plan.fillExpertBitmap(ep.rank);
            if (!std.mem.eql(u64, &expected_bitmap, &hello_resp.expert_bitmap)) {
                return error.ExpertOwnershipMismatch;
            }
            peer_slots[i].remote_bitmap = hello_resp.expert_bitmap;

            // Register persistent level-triggered EVFILT_READ on `kq_fd`.
            try transport.kqueueSetFilter(
                kq_fd,
                fd,
                std.c.EVFILT.READ,
                std.c.EV.ADD | std.c.EV.ENABLE,
                @intCast(i),
            );
        }

        return .{
            .config = config,
            .resolved_transport = resolved,
            .slab = slab,
            .kq_fd = kq_fd,
            .peer_slots = peer_slots,
            .peer_count = expected_peers,
            .rank_to_peer_idx = rank_to_peer_idx,
            .rank_buckets = rank_buckets,
            .scratch_out_f32 = scratch_out_f32,
            .next_seq_id = next_seq,
            .total_dispatches = 0,
            .total_remote_rpc_ns = 0,
            .total_local_compute_ns = 0,
            .total_remote_assignments = 0,
            .total_bytes_transferred = 0,
        };
    }

    pub fn deinit(self: *ExpertParallelCoordinator) void {
        for (0..self.peer_count) |i| {
            if (self.peer_slots[i].connected) {
                self.peer_slots[i].chan.close();
                self.peer_slots[i].connected = false;
            }
        }
        if (self.kq_fd >= 0) {
            _ = std.c.close(self.kq_fd);
            self.kq_fd = -1;
        }
        self.slab.deinit();
    }

    /// Measures round-trip ping latency (`ns`) to a connected remote peer rank.
    pub fn pingPeer(self: *ExpertParallelCoordinator, rank: u16) ClusterError!u64 {
        if (rank >= self.config.plan.world_size) return error.InvalidRank;
        const p_idx = self.rank_to_peer_idx[rank] orelse return error.PeerRankNotFound;
        const peer = &self.peer_slots[p_idx];

        const t0 = vmstat.nowNs() catch return error.ClockFailed;
        const seq = self.next_seq_id;
        self.next_seq_id += 1;

        const dst = try peer.chan.preparePayload(8);
        std.mem.writeInt(u64, dst[0..8], t0, .little);
        _ = peer.chan.commitFrame(.ping, 0, seq, 8);

        const deadline_ns = t0 + @as(u64, self.config.timeout_ms) * std.time.ns_per_ms;
        try self.drivePeerTxToCompletion(p_idx, deadline_ns);

        var events: [MAX_WORLD_SIZE]std.c.kevent64_s = undefined;
        while (true) {
            switch (try peer.chan.stepRecv()) {
                .frame_ready => |frame| {
                    defer peer.chan.resetRx();
                    if (frame.header.msg_type != .pong) return error.UnexpectedMessage;
                    if (frame.header.seq_id != seq) return error.SequenceMismatch;
                    if (frame.payload.len != 8 or std.mem.readInt(u64, frame.payload[0..8], .little) != t0) {
                        return error.MalformedPayload;
                    }
                    const t1 = vmstat.nowNs() catch return error.ClockFailed;
                    return t1 -| t0;
                },
                .peer_closed => return error.UnexpectedEof,
                .would_block => {
                    const now = vmstat.nowNs() catch return error.ClockFailed;
                    if (now >= deadline_ns) return error.Timeout;
                    const nev = try transport.kqueueWait(self.kq_fd, &events, deadline_ns - now);
                    if (nev == 0) return error.Timeout;
                },
            }
        }
    }

    /// Sends graceful `shutdown` control frames to all connected remote peers.
    pub fn shutdownPeers(self: *ExpertParallelCoordinator) ClusterError!void {
        const now = vmstat.nowNs() catch return error.ClockFailed;
        const deadline_ns = now + @as(u64, self.config.timeout_ms) * std.time.ns_per_ms;
        for (0..self.peer_count) |i| {
            if (!self.peer_slots[i].connected) continue;
            const seq = self.next_seq_id;
            self.next_seq_id += 1;
            _ = try self.peer_slots[i].chan.preparePayload(0);
            _ = self.peer_slots[i].chan.commitFrame(.shutdown, 0, seq, 0);
            try self.drivePeerTxToCompletion(i, deadline_ns);
        }
    }

    /// Dispatches one MoE layer's routed assignments across local + remote ranks:
    /// 1) partitions assignments by owner rank into preallocated wired buckets (`0` heap allocs);
    /// 2) initiates non-blocking TCP sends to all active remote peers FIRST;
    /// 3) executes local experts while remote peers compute over Thunderbolt/TCP;
    /// 4) collects remote `f32` partial outputs via `kqueue` and sums into `accum_out_f32`.
    pub fn dispatchLayerExperts(
        self: *ExpertParallelCoordinator,
        layer_idx: u16,
        num_tokens: u16,
        hidden_f16: []const f16,
        assignments: []const transport.Assignment,
        accum_out_f32: []f32,
    ) ClusterError!DispatchStats {
        if (layer_idx >= self.config.plan.num_layers) return error.InvalidLayer;
        if (num_tokens == 0 or num_tokens > self.config.max_tokens) return error.InvalidShape;
        if (assignments.len > self.config.max_assignments) return error.BufferTooSmall;
        const h: usize = self.config.hidden_size;
        const total_elems = try std.math.mul(usize, num_tokens, h);
        if (hidden_f16.len < total_elems or accum_out_f32.len < total_elems) {
            return error.BufferTooSmall;
        }

        const out_slice = accum_out_f32[0..total_elems];
        @memset(out_slice, 0.0);

        // 1) Split routed assignments by owner rank into preallocated wired buckets.
        var rank_counts: [MAX_WORLD_SIZE]usize = @splat(0);
        for (assignments) |a| {
            if (a.token_idx >= num_tokens) return error.InvalidTokenIndex;
            const owner = try self.config.plan.ownerRank(a.expert_id);
            const idx = rank_counts[owner];
            if (idx >= self.config.max_assignments) return error.BufferTooSmall;
            self.rank_buckets[owner][idx] = a;
            rank_counts[owner] = idx + 1;
        }

        const seq_id = self.next_seq_id;
        self.next_seq_id += 1;

        var stats: DispatchStats = .{};
        var peer_pending_recv: [MAX_WORLD_SIZE]bool = @splat(false);
        var pending_recv_count: usize = 0;

        // 2) Fire non-blocking remote `eval_experts_req` frames FIRST to overlap with local compute.
        const rpc_t0 = vmstat.nowNs() catch return error.ClockFailed;
        const deadline_ns = rpc_t0 + @as(u64, self.config.timeout_ms) * std.time.ns_per_ms;

        for (0..self.peer_count) |p_idx| {
            const peer = &self.peer_slots[p_idx];
            const count = rank_counts[peer.rank];
            if (count == 0) continue;

            const peer_assignments = self.rank_buckets[peer.rank][0..count];
            const payload_len = try transport.evalReqPayloadBytes(
                num_tokens,
                self.config.hidden_size,
                @intCast(count),
            );
            const payload_buf = try peer.chan.preparePayload(payload_len);
            const written = try transport.encodeEvalReqInto(
                payload_buf,
                layer_idx,
                num_tokens,
                self.config.hidden_size,
                peer_assignments,
                hidden_f16[0..total_elems],
            );
            std.debug.assert(written == payload_len);
            _ = peer.chan.commitFrame(.eval_experts_req, 0, seq_id, payload_len);

            switch (try peer.chan.stepSend()) {
                .complete => |n| {
                    stats.bytes_tx += n;
                },
                .would_block => |n| {
                    stats.bytes_tx += n;
                    try self.armPeerWrite(p_idx);
                },
            }

            peer_pending_recv[p_idx] = true;
            pending_recv_count += 1;
            stats.active_remote_peers += 1;
            stats.remote_assignments += @intCast(count);
        }

        // 3) Run local rank's experts concurrently while remote peers are executing.
        const local_count = rank_counts[self.config.plan.local_rank];
        if (local_count > 0) {
            const local_assignments = self.rank_buckets[self.config.plan.local_rank][0..local_count];
            stats.local_compute_ns = try self.config.local_backend.evalLocalSubset(
                layer_idx,
                num_tokens,
                hidden_f16[0..total_elems],
                local_assignments,
                out_slice,
            );
            stats.local_assignments = @intCast(local_count);
        }

        // 4) Collect all remote `eval_experts_resp` frames via persistent `kqueue` and sum into `out_slice`.
        var events: [MAX_WORLD_SIZE * 2]std.c.kevent64_s = undefined;
        while (pending_recv_count > 0) {
            const now = vmstat.nowNs() catch return error.ClockFailed;
            if (now >= deadline_ns) return error.Timeout;

            const nev = try transport.kqueueWait(self.kq_fd, &events, deadline_ns - now);
            if (nev == 0) return error.Timeout;

            for (events[0..nev]) |ev| {
                const p_idx: usize = @intCast(ev.udata);
                if (p_idx >= self.peer_count or !peer_pending_recv[p_idx]) continue;
                const peer = &self.peer_slots[p_idx];

                if (ev.filter == std.c.EVFILT.WRITE and peer.chan.hasPendingTx()) {
                    switch (try peer.chan.stepSend()) {
                        .complete => |n| {
                            stats.bytes_tx += n;
                            try self.disarmPeerWrite(p_idx);
                        },
                        .would_block => |n| {
                            stats.bytes_tx += n;
                        },
                    }
                    continue;
                }

                if (ev.filter == std.c.EVFILT.READ) {
                    switch (try peer.chan.stepRecv()) {
                        .would_block => |n| {
                            stats.bytes_rx += n;
                        },
                        .peer_closed => return error.UnexpectedEof,
                        .frame_ready => |frame| {
                            defer peer.chan.resetRx();
                            stats.bytes_rx += frame.bytes_read;
                            if (frame.header.msg_type != .eval_experts_resp) return error.UnexpectedMessage;
                            if (frame.header.seq_id != seq_id) return error.SequenceMismatch;

                            const view = try transport.decodeEvalRespView(frame.payload);
                            if (view.header.status != @backingInt(transport.EvalStatus.ok)) {
                                return error.RemoteEvalFailed;
                            }
                            if (view.header.num_tokens != num_tokens or
                                view.header.hidden_size != self.config.hidden_size or
                                view.accum_out_f32.len != total_elems)
                            {
                                return error.MalformedPayload;
                            }

                            for (out_slice, view.accum_out_f32) |*dst_v, src_v| {
                                dst_v.* += src_v;
                            }
                            if (view.header.peer_compute_ns > stats.max_peer_compute_ns) {
                                stats.max_peer_compute_ns = view.header.peer_compute_ns;
                            }

                            peer_pending_recv[p_idx] = false;
                            pending_recv_count -= 1;
                        },
                    }
                }
            }
        }

        const rpc_t1 = vmstat.nowNs() catch return error.ClockFailed;
        stats.remote_rpc_ns = if (stats.active_remote_peers > 0) (rpc_t1 -| rpc_t0) else 0;
        stats.bytes_transferred = stats.bytes_tx + stats.bytes_rx;

        self.total_dispatches += 1;
        self.total_remote_rpc_ns += stats.remote_rpc_ns;
        self.total_local_compute_ns += stats.local_compute_ns;
        self.total_remote_assignments += stats.remote_assignments;
        self.total_bytes_transferred += stats.bytes_transferred;
        return stats;
    }

    /// Convenience wrapper accumulating in `f32` inside the coordinator's
    /// preallocated wired scratch buffer and narrowing to `accum_out_f16` (`0` heap allocs).
    pub fn dispatchLayerExpertsToF16(
        self: *ExpertParallelCoordinator,
        layer_idx: u16,
        num_tokens: u16,
        hidden_f16: []const f16,
        assignments: []const transport.Assignment,
        accum_out_f16: []f16,
    ) ClusterError!DispatchStats {
        const total_elems = try std.math.mul(usize, num_tokens, self.config.hidden_size);
        if (accum_out_f16.len < total_elems) return error.BufferTooSmall;
        const stats = try self.dispatchLayerExperts(
            layer_idx,
            num_tokens,
            hidden_f16,
            assignments,
            self.scratch_out_f32[0..total_elems],
        );
        for (accum_out_f16[0..total_elems], self.scratch_out_f32[0..total_elems]) |*dst_h, src_f| {
            dst_h.* = @floatCast(src_f);
        }
        return stats;
    }

    fn armPeerWrite(self: *ExpertParallelCoordinator, p_idx: usize) ClusterError!void {
        if (self.peer_slots[p_idx].chan.write_armed) return;
        try transport.kqueueSetFilter(
            self.kq_fd,
            self.peer_slots[p_idx].chan.fd,
            std.c.EVFILT.WRITE,
            std.c.EV.ADD | std.c.EV.ENABLE,
            @intCast(p_idx),
        );
        self.peer_slots[p_idx].chan.write_armed = true;
    }

    fn disarmPeerWrite(self: *ExpertParallelCoordinator, p_idx: usize) ClusterError!void {
        if (!self.peer_slots[p_idx].chan.write_armed) return;
        try transport.kqueueSetFilter(
            self.kq_fd,
            self.peer_slots[p_idx].chan.fd,
            std.c.EVFILT.WRITE,
            std.c.EV.DELETE,
            @intCast(p_idx),
        );
        self.peer_slots[p_idx].chan.write_armed = false;
    }

    fn drivePeerTxToCompletion(self: *ExpertParallelCoordinator, p_idx: usize, deadline_ns: u64) ClusterError!void {
        const peer = &self.peer_slots[p_idx];
        var events: [MAX_WORLD_SIZE]std.c.kevent64_s = undefined;
        while (true) {
            switch (try peer.chan.stepSend()) {
                .complete => {
                    try self.disarmPeerWrite(p_idx);
                    return;
                },
                .would_block => {
                    try self.armPeerWrite(p_idx);
                    const now = vmstat.nowNs() catch return error.ClockFailed;
                    if (now >= deadline_ns) return error.Timeout;
                    const nev = try transport.kqueueWait(self.kq_fd, &events, deadline_ns - now);
                    if (nev == 0) return error.Timeout;
                },
            }
        }
    }
};

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

/// Deterministic synthetic MoE weight fixture generator for tests (`q4_affine`
/// and `bf16` / `f16` / `q3_affine` / `q2_affine`).
const TestMoeFixture = struct {
    num_layers: u16,
    num_experts: u16,
    hidden_size: u32,
    intermediate_size: u32,
    dtype: quant.WeightDType,
    weight_region: page.WiredRegion,
    entries: []CpuExpertEntry,

    fn create(
        allocator: std.mem.Allocator,
        num_layers: u16,
        num_experts: u16,
        hidden_size: u32,
        intermediate_size: u32,
        dtype: quant.WeightDType,
        seed: u64,
    ) !TestMoeFixture {
        const total = @as(usize, num_layers) * @as(usize, num_experts);
        const entries = try allocator.alloc(CpuExpertEntry, total);
        errdefer allocator.free(entries);

        const per_expert_bytes = try computeExpertBytes(dtype, hidden_size, intermediate_size);
        const total_bytes = per_expert_bytes * total;
        const region = try page.WiredRegion.init(total_bytes);
        errdefer region.deinit();
        @memset(region.bytes, 0);

        var prng = std.Random.DefaultPrng.init(seed);
        const rand = prng.random();

        for (0..total) |idx| {
            const base_off: usize = idx * per_expert_bytes;
            entries[idx] = .{
                .dtype = dtype,
                .weights = try fillExpertWeights(
                    region.bytes,
                    base_off,
                    dtype,
                    hidden_size,
                    intermediate_size,
                    rand,
                ),
                .present = true,
            };
        }

        return .{
            .num_layers = num_layers,
            .num_experts = num_experts,
            .hidden_size = hidden_size,
            .intermediate_size = intermediate_size,
            .dtype = dtype,
            .weight_region = region,
            .entries = entries,
        };
    }

    fn deinit(self: *TestMoeFixture, allocator: std.mem.Allocator) void {
        allocator.free(self.entries);
        self.weight_region.deinit();
    }

    fn computeExpertBytes(dtype: quant.WeightDType, h: u32, inter: u32) !usize {
        const align8 = struct {
            fn f(v: usize) usize {
                return (v + 7) & ~@as(usize, 7);
            }
        }.f;
        return switch (dtype) {
            .f16, .bf16 => blk: {
                const gu = align8(@as(usize, inter) * @as(usize, h) * 2);
                const dn = align8(@as(usize, h) * @as(usize, inter) * 2);
                break :blk gu * 2 + dn;
            },
            .q4_affine => try quantExpertBytes(.{ .bits = .q4, .group_size = 64 }, h, inter),
            .q3_affine => try quantExpertBytes(.{ .bits = .q3, .group_size = 64 }, h, inter),
            .q2_affine => try quantExpertBytes(.{ .bits = .q2, .group_size = 64 }, h, inter),
        };
    }

    fn quantExpertBytes(comptime spec: quant.QuantSpec, h: u32, inter: u32) !usize {
        const Ops = quant.QuantOps(spec);
        const align8 = struct {
            fn f(v: usize) usize {
                return (v + 7) & ~@as(usize, 7);
            }
        }.f;
        const gu_q = align8(try Ops.packedWeightBytes(inter, h));
        const gu_sb = align8(try Ops.scaleBiasBytes(inter, h));
        const dn_q = align8(try Ops.packedWeightBytes(h, inter));
        const dn_sb = align8(try Ops.scaleBiasBytes(h, inter));
        return (gu_q + gu_sb) * 2 + dn_q + dn_sb;
    }

    fn fillExpertWeights(
        buf: []u8,
        base_off: usize,
        dtype: quant.WeightDType,
        h: u32,
        inter: u32,
        rand: std.Random,
    ) !CpuExpertWeights {
        const align8 = struct {
            fn f(v: usize) usize {
                return (v + 7) & ~@as(usize, 7);
            }
        }.f;
        switch (dtype) {
            .f16, .bf16 => {
                const fdt: quant.FloatDType = if (dtype == .f16) .f16 else .bf16;
                const gu_elems = @as(usize, inter) * @as(usize, h);
                const gu_bytes = align8(gu_elems * 2);
                const dn_bytes = align8(gu_elems * 2);

                const gate_off = base_off;
                const up_off = gate_off + gu_bytes;
                const down_off = up_off + gu_bytes;

                fillFloatMat(buf[gate_off .. gate_off + gu_elems * 2], fdt, rand, 0.12);
                fillFloatMat(buf[up_off .. up_off + gu_elems * 2], fdt, rand, 0.12);
                fillFloatMat(buf[down_off .. down_off + gu_elems * 2], fdt, rand, 0.10);
                _ = dn_bytes;
                return .{
                    .gate_weight_offset = gate_off,
                    .up_weight_offset = up_off,
                    .down_weight_offset = down_off,
                };
            },
            .q4_affine => return try fillQuantExpert(.{ .bits = .q4, .group_size = 64 }, buf, base_off, h, inter, rand),
            .q3_affine => return try fillQuantExpert(.{ .bits = .q3, .group_size = 64 }, buf, base_off, h, inter, rand),
            .q2_affine => return try fillQuantExpert(.{ .bits = .q2, .group_size = 64 }, buf, base_off, h, inter, rand),
        }
    }

    fn fillFloatMat(dst: []u8, fdt: quant.FloatDType, rand: std.Random, scale: f32) void {
        const out = std.mem.bytesAsSlice(u16, @as([]align(2) u8, @alignCast(dst)));
        for (out) |*slot| {
            const v: f32 = (rand.float(f32) * 2.0 - 1.0) * scale;
            slot.* = quant.f32ToFloatBits(fdt, v);
        }
    }

    fn fillQuantExpert(
        comptime spec: quant.QuantSpec,
        buf: []u8,
        base_off: usize,
        h: u32,
        inter: u32,
        rand: std.Random,
    ) !CpuExpertWeights {
        const Ops = quant.QuantOps(spec);
        const align8 = struct {
            fn f(v: usize) usize {
                return (v + 7) & ~@as(usize, 7);
            }
        }.f;
        const gu_q_len = try Ops.packedWeightBytes(inter, h);
        const gu_sb_len = try Ops.scaleBiasBytes(inter, h);
        const dn_q_len = try Ops.packedWeightBytes(h, inter);
        const dn_sb_len = try Ops.scaleBiasBytes(h, inter);

        var off = base_off;
        const gate_q = off;
        off += align8(gu_q_len);
        const gate_sb = off;
        off += align8(gu_sb_len);
        const up_q = off;
        off += align8(gu_q_len);
        const up_sb = off;
        off += align8(gu_sb_len);
        const down_q = off;
        off += align8(dn_q_len);
        const down_sb = off;

        fillQuantMat(Ops, buf[gate_q .. gate_q + gu_q_len], buf[gate_sb .. gate_sb + gu_sb_len], rand, 0.015);
        fillQuantMat(Ops, buf[up_q .. up_q + gu_q_len], buf[up_sb .. up_sb + gu_sb_len], rand, 0.015);
        fillQuantMat(Ops, buf[down_q .. down_q + dn_q_len], buf[down_sb .. down_sb + dn_sb_len], rand, 0.012);

        return .{
            .gate_weight_offset = gate_q,
            .gate_sb_offset = gate_sb,
            .up_weight_offset = up_q,
            .up_sb_offset = up_sb,
            .down_weight_offset = down_q,
            .down_sb_offset = down_sb,
        };
    }

    fn fillQuantMat(comptime Ops: type, q_dst: []u8, sb_dst: []u8, rand: std.Random, max_scale: f32) void {
        const bits: u5 = @intCast(Ops.quant_spec.bits.bitWidth());
        const max_code: u8 = (@as(u8, 1) << @intCast(bits)) - 1;
        const gs: usize = Ops.quant_spec.group_size;
        const gb: usize = Ops.groupWeightBytes();
        const num_groups = q_dst.len / gb;
        const sb_slice = std.mem.bytesAsSlice(
            quant.ScaleBias,
            @as([]align(@alignOf(quant.ScaleBias)) u8, @alignCast(sb_dst)),
        );
        var codes: [gs]u8 = undefined;
        for (0..num_groups) |g| {
            for (&codes) |*c| c.* = rand.intRangeAtMost(u8, 0, max_code);
            Ops.packGroup(&codes, q_dst[g * gb ..][0..gb]);
            const scale: f32 = (0.25 + 0.75 * rand.float(f32)) * max_scale;
            const bias: f32 = -0.5 * @as(f32, @floatFromInt(max_code)) * scale;
            sb_slice[g] = .{ .scale = @floatCast(scale), .bias = @floatCast(bias) };
        }
    }

    /// Builds a masked expert descriptor slice exposing ONLY experts owned by `plan.local_rank`.
    fn makeRankEntries(
        self: *const TestMoeFixture,
        allocator: std.mem.Allocator,
        plan: ExpertPartitionPlan,
    ) ![]CpuExpertEntry {
        const out = try allocator.alloc(CpuExpertEntry, self.entries.len);
        for (0..self.num_layers) |l| {
            for (0..self.num_experts) |e| {
                const idx = l * self.num_experts + e;
                if (plan.isLocal(@intCast(e))) {
                    out[idx] = self.entries[idx];
                } else {
                    out[idx] = .{ .present = false };
                }
            }
        }
        return out;
    }
};

fn buildTestRouting(
    allocator: std.mem.Allocator,
    num_tokens: u16,
    num_experts: u16,
    top_k: u16,
    seed: u64,
) ![]transport.Assignment {
    const out = try allocator.alloc(transport.Assignment, @as(usize, num_tokens) * @as(usize, top_k));
    var prng = std.Random.DefaultPrng.init(seed);
    const rand = prng.random();
    for (0..num_tokens) |t| {
        var sum_w: f32 = 0.0;
        const base = t * top_k;
        // Spread top-k experts deterministically across all ranks while ensuring uniqueness per token.
        const step: u16 = @max(@as(u16, 1), num_experts / top_k);
        for (0..top_k) |k| {
            const offset: u16 = rand.intRangeLessThan(u16, 0, step);
            const eid: u16 = @intCast((k * step + offset + t * 3) % num_experts);
            const w: f32 = 0.2 + rand.float(f32);
            sum_w += w;
            out[base + k] = .{
                .token_idx = @intCast(t),
                .expert_id = eid,
                .weight = w,
            };
        }
        for (0..top_k) |k| {
            out[base + k].weight /= sum_w;
        }
    }
    return out;
}

fn fillTestHidden(buf: []f16, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const rand = prng.random();
    for (buf) |*v| {
        v.* = @floatCast((rand.float(f32) * 2.0 - 1.0) * 0.75);
    }
}

/// Runs a full live loopback TCP multi-rank expert-parallel cluster test
/// (`world_size` = 2 or 4) comparing distributed output against a single-node
/// CPU reference over `64` experts `top-8` for `M = 1` and `M = 16`, plus
/// optional zero steady-state heap allocation verification via `vmstat`.
fn runClusterParityScenario(
    world_size: u16,
    strategy: PartitionStrategy,
    dtype: quant.WeightDType,
    check_zero_malloc: bool,
) !void {
    const gpa = std.testing.allocator;
    const num_layers: u16 = 2;
    const num_experts: u16 = 64;
    const hidden_size: u32 = 64;
    const intermediate_size: u32 = 128;
    const top_k: u16 = 8;

    var fixture = try TestMoeFixture.create(
        gpa,
        num_layers,
        num_experts,
        hidden_size,
        intermediate_size,
        dtype,
        0xA11CE_0000 + @as(u64, world_size) * 17 + @backingInt(dtype),
    );
    defer fixture.deinit(gpa);

    // Single-node reference backend owning all 64 experts.
    var ref_scratch: [intermediate_size]f32 = undefined;
    var ref_store = try CpuMoeExpertStore.init(
        num_layers,
        num_experts,
        hidden_size,
        intermediate_size,
        fixture.weight_region.bytes,
        fixture.entries,
        &ref_scratch,
    );

    // Prepare per-rank masked stores and start `world_size - 1` PeerWorkerServer threads.
    var rank_entries: [MAX_WORLD_SIZE][]CpuExpertEntry = @splat(&.{});
    var rank_scratches: [MAX_WORLD_SIZE][intermediate_size]f32 = undefined;
    var rank_stores: [MAX_WORLD_SIZE]CpuMoeExpertStore = undefined;
    var servers: [MAX_WORLD_SIZE]PeerWorkerServer = undefined;
    var servers_inited: usize = 0;

    defer {
        for (0..servers_inited) |i| servers[i].deinit();
        for (0..world_size) |r| {
            if (rank_entries[r].len > 0) gpa.free(rank_entries[r]);
        }
    }

    for (0..world_size) |r| {
        const plan_r = try ExpertPartitionPlan.init(
            world_size,
            @intCast(r),
            num_layers,
            num_experts,
            strategy,
        );
        rank_entries[r] = try fixture.makeRankEntries(gpa, plan_r);
        rank_stores[r] = try CpuMoeExpertStore.init(
            num_layers,
            num_experts,
            hidden_size,
            intermediate_size,
            fixture.weight_region.bytes,
            rank_entries[r],
            &rank_scratches[r],
        );
    }

    var peer_endpoints: [MAX_WORLD_SIZE]PeerEndpoint = undefined;
    for (1..world_size) |r| {
        const slot = r - 1;
        const plan_r = try ExpertPartitionPlan.init(
            world_size,
            @intCast(r),
            num_layers,
            num_experts,
            strategy,
        );
        servers[slot] = try PeerWorkerServer.init(.{
            .host = .{ 127, 0, 0, 1 },
            .port = 0,
            .transport_kind = .auto,
            .plan = plan_r,
            .hidden_size = hidden_size,
            .max_tokens = 16,
            .max_assignments = 256,
            .max_clients = 2,
            .backend = rank_stores[r].backend(),
        });
        servers_inited += 1;
        try servers[slot].start();
        peer_endpoints[slot] = .{
            .rank = @intCast(r),
            .host = .{ 127, 0, 0, 1 },
            .port = servers[slot].boundPort(),
        };
    }

    const coord_plan = try ExpertPartitionPlan.init(world_size, 0, num_layers, num_experts, strategy);
    var coord = try ExpertParallelCoordinator.init(.{
        .plan = coord_plan,
        .hidden_size = hidden_size,
        .max_tokens = 16,
        .max_assignments = 256,
        .transport_kind = .auto,
        .timeout_ms = 5000,
        .peers = peer_endpoints[0 .. world_size - 1],
        .local_backend = rank_stores[0].backend(),
    });
    defer coord.deinit();

    // Ping every remote rank.
    for (1..world_size) |r| {
        const rtt_ns = try coord.pingPeer(@intCast(r));
        try std.testing.expect(rtt_ns < 5 * std.time.ns_per_s);
    }

    // Verify M = 1 (decode) and M = 16 (tree/prefill) across both layers.
    const batch_sizes = [_]u16{ 1, 16 };
    for (batch_sizes) |m| {
        const elems = @as(usize, m) * hidden_size;
        const hidden = try gpa.alloc(f16, elems);
        defer gpa.free(hidden);
        fillTestHidden(hidden, 0xBEEF + m);

        const assignments = try buildTestRouting(gpa, m, num_experts, top_k, 0xCAFE + m);
        defer gpa.free(assignments);

        const ref_out = try gpa.alloc(f32, elems);
        defer gpa.free(ref_out);
        @memset(ref_out, 0.0);

        const dist_out = try gpa.alloc(f32, elems);
        defer gpa.free(dist_out);
        const dist_f16 = try gpa.alloc(f16, elems);
        defer gpa.free(dist_f16);

        const layer_idx: u16 = if (m == 1) 0 else 1;
        _ = try ref_store.evalLocalSubset(layer_idx, m, hidden, assignments, ref_out);
        const dstats = try coord.dispatchLayerExperts(layer_idx, m, hidden, assignments, dist_out);

        try std.testing.expectEqual(@as(u32, @intCast(assignments.len)), dstats.local_assignments + dstats.remote_assignments);
        try std.testing.expect(dstats.remote_assignments > 0);
        try std.testing.expect(dstats.bytes_transferred > 0);
        try std.testing.expect(dstats.active_remote_peers > 0);

        var max_err: f32 = 0.0;
        for (ref_out, dist_out) |rv, dv| {
            const diff = @abs(rv - dv);
            if (diff > max_err) max_err = diff;
        }
        try std.testing.expect(max_err < 1e-4);

        // Also validate zero-alloc f16 output path.
        _ = try coord.dispatchLayerExpertsToF16(layer_idx, m, hidden, assignments, dist_f16);
        for (ref_out, dist_f16) |rv, dh| {
            const expected_h: f16 = @floatCast(rv);
            try std.testing.expectApproxEqAbs(@as(f32, @floatCast(expected_h)), @as(f32, @floatCast(dh)), 1e-3);
        }

        // Steady-state zero-allocation check via native Mach malloc zone stats.
        if (check_zero_malloc and m == 16) {
            // Warm one pass, then measure 8 steady-state distributed RPC dispatches.
            _ = try coord.dispatchLayerExperts(layer_idx, m, hidden, assignments, dist_out);
            const snap_before = try vmstat.Snapshot.capture();
            for (0..8) |_| {
                _ = try coord.dispatchLayerExperts(layer_idx, m, hidden, assignments, dist_out);
            }
            const snap_after = try vmstat.Snapshot.capture();
            const delta = vmstat.heapDelta(snap_before, snap_after);
            try std.testing.expect(delta.blocks_in_use <= 0);
        }
    }

    // Verify peer workers recorded non-zero stats before graceful network shutdown.
    for (0..world_size - 1) |slot| {
        const st = servers[slot].stats();
        try std.testing.expect(st.requests_served >= 2);
        try std.testing.expect(st.assignments_evaluated > 0);
        try std.testing.expect(st.bytes_rx > 0);
        try std.testing.expect(st.bytes_tx > 0);
    }

    try coord.shutdownPeers();
}

test "ExpertPartitionPlan supports 1..16 ranks and 512..1024 experts with ADR-0003 ExpertSource" {
    // 512 experts (deepseek41 552B MoE scale) across 4 ranks, contiguous_by_rank.
    const plan4 = try ExpertPartitionPlan.init(4, 1, 61, 512, .contiguous_by_rank);
    try std.testing.expectEqual(@as(u16, 128), plan4.localExpertCount());
    const r1 = try plan4.rankExpertRange(1);
    try std.testing.expectEqual(@as(u16, 128), r1.start);
    try std.testing.expectEqual(@as(u16, 256), r1.end);
    try std.testing.expectEqual(@as(u16, 0), try plan4.ownerRank(0));
    try std.testing.expectEqual(@as(u16, 1), try plan4.ownerRank(128));
    try std.testing.expectEqual(@as(u16, 3), try plan4.ownerRank(511));

    // 1024 experts across 16 ranks, round_robin.
    const plan16 = try ExpertPartitionPlan.init(16, 5, 4, 1024, .round_robin);
    try std.testing.expectEqual(@as(u16, 64), plan16.localExpertCount());
    try std.testing.expectEqual(@as(u16, 5), try plan16.ownerRank(5));
    try std.testing.expectEqual(@as(u16, 5), try plan16.ownerRank(21));
    try std.testing.expect(plan16.isLocal(1013)); // 1013 % 16 == 5
    try std.testing.expect(!plan16.isLocal(1014));

    // Build ExpertSource entries: owned -> .local_wired_ram, remote -> .thunderbolt_peer.
    const wired = try page.WiredRegion.init(page.APPLE_PAGE_SIZE);
    defer wired.deinit();

    const local_src = try plan4.buildExpertSource(3, 130, wired.bytes, 0, page.APPLE_PAGE_SIZE);
    switch (local_src) {
        .local_wired_ram => |w| try std.testing.expectEqual(wired.bytes.ptr, w.bytes.ptr),
        else => return error.TestUnexpectedResult,
    }
    const remote_src = try plan4.buildExpertSource(3, 300, null, 2 * page.APPLE_PAGE_SIZE, 4 * page.APPLE_PAGE_SIZE);
    switch (remote_src) {
        .thunderbolt_peer => |p| {
            try std.testing.expectEqual(@as(u16, 2), p.peer);
            try std.testing.expectEqual(@as(u16, 3), p.layer);
            try std.testing.expectEqual(@as(u16, 300), p.expert_id);
            try std.testing.expectEqual(@as(u64, 2 * page.APPLE_PAGE_SIZE), p.remote_offset);
            try std.testing.expectEqual(@as(u64, 4 * page.APPLE_PAGE_SIZE), p.len);
        },
        else => return error.TestUnexpectedResult,
    }
    try std.testing.expectError(error.TransportUnavailable, remote_src.residentBytes());

    // Invalid bounds rejected.
    try std.testing.expectError(error.InvalidWorldSize, ExpertPartitionPlan.init(0, 0, 1, 64, .contiguous_by_rank));
    try std.testing.expectError(error.InvalidWorldSize, ExpertPartitionPlan.init(17, 0, 1, 64, .contiguous_by_rank));
    try std.testing.expectError(error.InvalidExpertCount, ExpertPartitionPlan.init(4, 0, 1, 1025, .contiguous_by_rank));
    try std.testing.expectError(error.InvalidRank, ExpertPartitionPlan.init(4, 4, 1, 64, .contiguous_by_rank));
}

test "strict TB5 RDMA transport request fails cleanly on TB4 hardware" {
    const live = transport.probeTb5Rdma();
    if (live.available) return; // Only applies when no TB5 RDMA HCA is present.

    const plan = try ExpertPartitionPlan.init(2, 1, 1, 8, .contiguous_by_rank);
    var dummy_ctx: u8 = 0;
    const dummy_backend: ExpertEvalBackend = .{
        .ctx = &dummy_ctx,
        .evalLocalSubsetFn = struct {
            fn f(_: *anyopaque, _: u16, _: u16, _: []const f16, _: []const transport.Assignment, _: []f32) EvalError!u64 {
                return 0;
            }
        }.f,
    };

    try std.testing.expectError(
        error.Tb5RdmaUnavailable,
        PeerWorkerServer.init(.{
            .transport_kind = .tb5_rdma,
            .plan = plan,
            .hidden_size = 64,
            .backend = dummy_backend,
        }),
    );

    const coord_plan = try ExpertPartitionPlan.init(2, 0, 1, 8, .contiguous_by_rank);
    const dummy_peers = [_]PeerEndpoint{.{ .rank = 1, .host = .{ 127, 0, 0, 1 }, .port = 9999 }};
    try std.testing.expectError(
        error.Tb5RdmaUnavailable,
        ExpertParallelCoordinator.init(.{
            .transport_kind = .tb5_rdma,
            .plan = coord_plan,
            .hidden_size = 64,
            .peers = &dummy_peers,
            .local_backend = dummy_backend,
        }),
    );
}

test "2-rank loopback ExpertParallelCoordinator matches single-node reference on q4_affine SwiGLU MoE" {
    try runClusterParityScenario(2, .contiguous_by_rank, .q4_affine, false);
}

test "4-rank loopback ExpertParallelCoordinator matches single-node reference on q4_affine SwiGLU MoE with zero heap allocs" {
    try runClusterParityScenario(4, .contiguous_by_rank, .q4_affine, true);
}

test "2-rank and 4-rank loopback ExpertParallelCoordinator match single-node reference on bf16 SwiGLU MoE" {
    try runClusterParityScenario(2, .round_robin, .bf16, false);
    try runClusterParityScenario(4, .round_robin, .bf16, true);
}

test "Coordinator handshake rejects hidden_size and expert-ownership mismatches" {
    const plan_srv = try ExpertPartitionPlan.init(2, 1, 1, 16, .contiguous_by_rank);
    var dummy_ctx: u8 = 0;
    const dummy_backend: ExpertEvalBackend = .{
        .ctx = &dummy_ctx,
        .evalLocalSubsetFn = struct {
            fn f(_: *anyopaque, _: u16, _: u16, _: []const f16, _: []const transport.Assignment, _: []f32) EvalError!u64 {
                return 0;
            }
        }.f,
    };

    var srv = try PeerWorkerServer.init(.{
        .host = .{ 127, 0, 0, 1 },
        .port = 0,
        .plan = plan_srv,
        .hidden_size = 128,
        .backend = dummy_backend,
    });
    defer srv.deinit();
    try srv.start();

    const peers = [_]PeerEndpoint{.{ .rank = 1, .host = .{ 127, 0, 0, 1 }, .port = srv.boundPort() }};

    // Wrong hidden_size -> HandshakeMismatch.
    const plan_ok = try ExpertPartitionPlan.init(2, 0, 1, 16, .contiguous_by_rank);
    try std.testing.expectError(
        error.HandshakeMismatch,
        ExpertParallelCoordinator.init(.{
            .plan = plan_ok,
            .hidden_size = 64,
            .peers = &peers,
            .local_backend = dummy_backend,
        }),
    );

    // Different partition strategy -> ExpertOwnershipMismatch.
    const plan_rr = try ExpertPartitionPlan.init(2, 0, 1, 16, .round_robin);
    try std.testing.expectError(
        error.ExpertOwnershipMismatch,
        ExpertParallelCoordinator.init(.{
            .plan = plan_rr,
            .hidden_size = 128,
            .peers = &peers,
            .local_backend = dummy_backend,
        }),
    );
}

test "CpuMoeExpertStore evaluates f16, q3_affine, and q2_affine SwiGLU experts without allocation" {
    const gpa = std.testing.allocator;
    const dtypes = [_]quant.WeightDType{ .f16, .q3_affine, .q2_affine };
    for (dtypes) |dt| {
        var fixture = try TestMoeFixture.create(gpa, 1, 4, 64, 64, dt, 0x5555 + @as(u64, @backingInt(dt)));
        defer fixture.deinit(gpa);

        var scratch: [64]f32 = undefined;
        var store = try CpuMoeExpertStore.init(
            1,
            4,
            64,
            64,
            fixture.weight_region.bytes,
            fixture.entries,
            &scratch,
        );

        var hidden: [64]f16 = undefined;
        fillTestHidden(&hidden, 0x1111);
        const assignments = [_]transport.Assignment{
            .{ .token_idx = 0, .expert_id = 0, .weight = 0.6 },
            .{ .token_idx = 0, .expert_id = 3, .weight = 0.4 },
        };
        var out: [64]f32 = @splat(0.0);
        const ns = try store.evalLocalSubset(0, 1, &hidden, &assignments, &out);
        try std.testing.expect(ns > 0);
        var l2: f32 = 0.0;
        for (out) |v| {
            try std.testing.expect(std.math.isFinite(v));
            l2 += v * v;
        }
        try std.testing.expect(l2 > 0.0);
    }
}
