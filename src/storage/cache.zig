//! Wired cold-expert slot pool and LFU-decay residency cache (status row C5,
//! ADR-0003, ADR-0006, docs/architecture/experts-and-storage.md).
//!
//! - `ExpertSlotPool`: carved once at startup from a `WiredArena` so all
//!   `num_slots` 16 KiB-aligned cold-expert slots live contiguously inside a
//!   single resident `metal.Buffer` (zero `MTLResidencySet` churn on the hot
//!   path). Each slot holds a complete `[gate_proj, up_proj, down_proj]` triple
//!   at deterministic 16 KiB-aligned sub-offsets.
//! - `ExpertCache`: O(1) flat `(layer, expert)` residency table with zero
//!   dynamic heap allocation on the steady-state `prefetch` / `acquirePinned` /
//!   `pollCompletions` / `releasePins` hot path. Hot experts
//!   (`local_wired_ram`) resolve directly with zero slot overhead; cold experts
//!   (`local_nvme_ssd`) stream asynchronously via `AsyncExpertReader`, coalesce
//!   in-flight prefetches, pin active verification-pass experts against
//!   eviction, and evict unpinned slots using LFU with periodic halving decay.

const std = @import("std");
const page = @import("../mem/page.zig");
const arena_mod = @import("../mem/arena.zig");
const budget_mod = @import("../mem/budget.zig");
const metal = @import("../bridge/metal.zig");
const expert = @import("../expert/source.zig");
const amf = @import("../format/amf.zig");
const moe = @import("../gpu/moe.zig");
const vmstat = @import("../platform/vmstat.zig");
const paths = @import("../platform/paths.zig");
const aio = @import("aio.zig");

const INVALID_SLOT: u32 = std.math.maxInt(u32);
const BLOBS_PER_EXPERT: u8 = 3;
const MAX_LFU_SCORE: u32 = std.math.maxInt(u32) / 2;

/// Unique `(layer, expert)` identifier inside an MoE model.
pub const ExpertKey = struct {
    layer: u16,
    expert: u16,

    pub fn eql(a: ExpertKey, b: ExpertKey) bool {
        return a.layer == b.layer and a.expert == b.expert;
    }
};

/// A complete MoE expert's three projection blobs (`gate`, `up`, `down`) on NVMe.
pub const ColdExpertTriple = struct {
    /// Open `F_NOCACHE` file descriptor backing the `.amf` pack.
    fd: std.c.fd_t,
    gate: expert.NvmeExpert,
    up: expert.NvmeExpert,
    down: expert.NvmeExpert,
    /// Byte offset of `scale_bias` within each blob (`0` for unquantized FP16/BF16).
    gate_params_offset: u64 = 0,
    up_params_offset: u64 = 0,
    down_params_offset: u64 = 0,

    pub fn totalBlobBytes(self: ColdExpertTriple) u64 {
        return self.gate.len + self.up.len + self.down.len;
    }

    /// Constructs a validated `ColdExpertTriple` directly from `.amf` directory entries.
    pub fn fromAmfEntries(
        fd: std.c.fd_t,
        file_index: u16,
        gate_entry: *const amf.TensorEntry,
        up_entry: *const amf.TensorEntry,
        down_entry: *const amf.TensorEntry,
    ) expert.NvmeExpert.InitError!ColdExpertTriple {
        return .{
            .fd = fd,
            .gate = try expert.NvmeExpert.init(file_index, gate_entry.blob_offset, gate_entry.blob_bytes),
            .up = try expert.NvmeExpert.init(file_index, up_entry.blob_offset, up_entry.blob_bytes),
            .down = try expert.NvmeExpert.init(file_index, down_entry.blob_offset, down_entry.blob_bytes),
            .gate_params_offset = gate_entry.params_offset,
            .up_params_offset = up_entry.params_offset,
            .down_params_offset = down_entry.params_offset,
        };
    }
};

/// A complete MoE expert permanently resident in wired RAM (`local_wired_ram`).
pub const HotExpertTriple = struct {
    gate: expert.WiredExpert,
    up: expert.WiredExpert,
    down: expert.WiredExpert,
    /// Precomputed 64-bit byte offsets inside the owning hot weight `metal.Buffer`.
    weights: moe.ExpertWeights,
    /// Optional owning `metal.Buffer` for zero-copy Metal 4 argument-table binding.
    metal_buffer: ?metal.Buffer = null,
};

/// Static startup placement for one `(layer, expert)` mirroring `ExpertSource` (ADR-0003).
pub const ExpertPlacement = union(enum) {
    local_wired_ram: HotExpertTriple,
    local_nvme_ssd: ColdExpertTriple,
    thunderbolt_peer: expert.PeerExpert,

    pub fn sourceKind(self: ExpertPlacement) std.meta.Tag(expert.ExpertSource) {
        return switch (self) {
            .local_wired_ram => .local_wired_ram,
            .local_nvme_ssd => .local_nvme_ssd,
            .thunderbolt_peer => .thunderbolt_peer,
        };
    }
};

// ---------------------------------------------------------------------------
// ExpertSlotPool: 16 KiB-aligned wired slots over a single resident MTLBuffer
// ---------------------------------------------------------------------------

pub const PoolError = error{
    ZeroSlots,
    ZeroLength,
    MisalignedLength,
    Overflow,
} || arena_mod.WiredArena.AllocError;

pub const PoolSpec = struct {
    num_slots: u32,
    /// 16 KiB-aligned byte capacity for `gate_proj` in each slot.
    gate_blob_bytes: usize,
    /// 16 KiB-aligned byte capacity for `up_proj` in each slot.
    up_blob_bytes: usize,
    /// 16 KiB-aligned byte capacity for `down_proj` in each slot.
    down_blob_bytes: usize,

    pub fn validate(self: PoolSpec) PoolError!void {
        if (self.num_slots == 0) return error.ZeroSlots;
        if (self.gate_blob_bytes == 0 or self.up_blob_bytes == 0 or self.down_blob_bytes == 0) {
            return error.ZeroLength;
        }
        if (self.gate_blob_bytes % page.APPLE_PAGE_SIZE != 0 or
            self.up_blob_bytes % page.APPLE_PAGE_SIZE != 0 or
            self.down_blob_bytes % page.APPLE_PAGE_SIZE != 0)
        {
            return error.MisalignedLength;
        }
    }

    pub fn slotBytes(self: PoolSpec) PoolError!usize {
        const gu = std.math.add(usize, self.gate_blob_bytes, self.up_blob_bytes) catch return error.Overflow;
        return std.math.add(usize, gu, self.down_blob_bytes) catch return error.Overflow;
    }

    pub fn totalBytes(self: PoolSpec) PoolError!usize {
        const sb = try self.slotBytes();
        return std.math.mul(usize, self.num_slots, sb) catch return error.Overflow;
    }

    pub fn gateSubOffset(self: PoolSpec) usize {
        _ = self;
        return 0;
    }

    pub fn upSubOffset(self: PoolSpec) usize {
        return self.gate_blob_bytes;
    }

    pub fn downSubOffset(self: PoolSpec) usize {
        return self.gate_blob_bytes + self.up_blob_bytes;
    }
};

/// Contiguous pool of `num_slots` 16 KiB-aligned cold-expert slots carved once
/// at startup from a `WiredArena`. When paired with a `metal.Buffer`, all slots
/// share that single resident buffer with zero `MTLResidencySet` churn.
pub const ExpertSlotPool = struct {
    spec: PoolSpec,
    slot_bytes: usize,
    owned_arena: ?arena_mod.WiredArena,
    owns_metal_buffer: bool,
    metal_buffer: ?metal.Buffer,
    backing: []align(page.APPLE_PAGE_SIZE) u8,
    /// Byte offset of `backing.ptr` inside the owning `WiredArena` / `metal_buffer`.
    arena_base_offset: u64,

    /// Carves `spec.num_slots` contiguous 16 KiB-aligned slots from an existing
    /// `WiredArena` (e.g. a model weight shard so hot experts and cold slots can
    /// live inside one `metal.Buffer` and run in a single `encodeMoELayer` dispatch).
    pub fn initFromArena(
        arena: *arena_mod.WiredArena,
        metal_buffer: ?metal.Buffer,
        spec: PoolSpec,
    ) PoolError!ExpertSlotPool {
        try spec.validate();
        const sb = try spec.slotBytes();
        const total = try spec.totalBytes();
        const backing = try arena.allocPages(total);
        return .{
            .spec = spec,
            .slot_bytes = sb,
            .owned_arena = null,
            .owns_metal_buffer = false,
            .metal_buffer = metal_buffer,
            .backing = backing,
            .arena_base_offset = arena.offsetOf(backing.ptr),
        };
    }

    /// Allocates a dedicated `Budget`-checked `WiredArena` for the cold-expert
    /// slot pool and optionally wraps it once into a zero-copy `metal.Buffer`.
    pub fn initOwned(
        budget: budget_mod.Budget,
        device: ?metal.Device,
        spec: PoolSpec,
    ) (PoolError || arena_mod.WiredArena.InitError || metal.WrapError)!ExpertSlotPool {
        try spec.validate();
        const sb = try spec.slotBytes();
        const total = try spec.totalBytes();

        var arena = try arena_mod.WiredArena.init(total, budget);
        errdefer arena.deinit();

        const backing = try arena.allocPages(total);
        const metal_buf: ?metal.Buffer = if (device) |dev|
            try metal.Buffer.wrap(dev, arena.bytes())
        else
            null;

        return .{
            .spec = spec,
            .slot_bytes = sb,
            .owned_arena = arena,
            .owns_metal_buffer = metal_buf != null,
            .metal_buffer = metal_buf,
            .backing = backing,
            .arena_base_offset = 0,
        };
    }

    pub fn deinit(self: *ExpertSlotPool) void {
        if (self.owns_metal_buffer) {
            if (self.metal_buffer) |buf| buf.close();
        }
        self.metal_buffer = null;
        self.owns_metal_buffer = false;
        if (self.owned_arena) |*a| {
            a.deinit();
            self.owned_arena = null;
        }
        self.backing = &.{};
    }

    pub fn numSlots(self: *const ExpertSlotPool) u32 {
        return self.spec.num_slots;
    }

    pub fn totalBytes(self: *const ExpertSlotPool) usize {
        return self.backing.len;
    }

    /// Full 16 KiB-aligned byte slice for slot `slot_idx`.
    pub fn slotSlice(self: *const ExpertSlotPool, slot_idx: u32) []align(page.APPLE_PAGE_SIZE) u8 {
        std.debug.assert(slot_idx < self.spec.num_slots);
        const start = @as(usize, slot_idx) * self.slot_bytes;
        return @alignCast(self.backing[start .. start + self.slot_bytes]);
    }

    pub fn gateSlice(self: *const ExpertSlotPool, slot_idx: u32) []align(page.APPLE_PAGE_SIZE) u8 {
        const s = self.slotSlice(slot_idx);
        const off = self.spec.gateSubOffset();
        return @alignCast(s[off .. off + self.spec.gate_blob_bytes]);
    }

    pub fn upSlice(self: *const ExpertSlotPool, slot_idx: u32) []align(page.APPLE_PAGE_SIZE) u8 {
        const s = self.slotSlice(slot_idx);
        const off = self.spec.upSubOffset();
        return @alignCast(s[off .. off + self.spec.up_blob_bytes]);
    }

    pub fn downSlice(self: *const ExpertSlotPool, slot_idx: u32) []align(page.APPLE_PAGE_SIZE) u8 {
        const s = self.slotSlice(slot_idx);
        const off = self.spec.downSubOffset();
        return @alignCast(s[off .. off + self.spec.down_blob_bytes]);
    }

    /// Byte offset of `slot_idx` inside the owning `WiredArena` / `metal_buffer`.
    pub fn slotArenaOffset(self: *const ExpertSlotPool, slot_idx: u32) u64 {
        std.debug.assert(slot_idx < self.spec.num_slots);
        return self.arena_base_offset + @as(u64, slot_idx) * @as(u64, self.slot_bytes);
    }

    /// Computes ready-to-bind 64-bit `moe.ExpertWeights` offsets into the pool's
    /// `metal_buffer` for `slot_idx`. Unquantized FP16/BF16 experts have
    /// `*_params_offset == 0`, which maps `*_sb_offset = 0` matching `qwen3.zig`.
    pub fn slotExpertWeights(
        self: *const ExpertSlotPool,
        slot_idx: u32,
        gate_params_offset: u64,
        up_params_offset: u64,
        down_params_offset: u64,
    ) moe.ExpertWeights {
        const base = self.slotArenaOffset(slot_idx);
        const g_off = base + self.spec.gateSubOffset();
        const u_off = base + self.spec.upSubOffset();
        const d_off = base + self.spec.downSubOffset();
        return .{
            .gate_qweight_offset = g_off,
            .gate_sb_offset = if (gate_params_offset == 0) 0 else g_off + gate_params_offset,
            .up_qweight_offset = u_off,
            .up_sb_offset = if (up_params_offset == 0) 0 else u_off + up_params_offset,
            .down_qweight_offset = d_off,
            .down_sb_offset = if (down_params_offset == 0) 0 else d_off + down_params_offset,
        };
    }

    /// Returns 16 KiB-aligned `WiredExpert` views over `[gate, up, down]` for `slot_idx`.
    pub fn wiredSources(self: *const ExpertSlotPool, slot_idx: u32) page.ShareError![3]expert.WiredExpert {
        return .{
            try expert.WiredExpert.init(self.gateSlice(slot_idx)),
            try expert.WiredExpert.init(self.upSlice(slot_idx)),
            try expert.WiredExpert.init(self.downSlice(slot_idx)),
        };
    }
};

// ---------------------------------------------------------------------------
// ExpertCache: LFU-decay residency manager, async prefetch, pass pinning
// ---------------------------------------------------------------------------

pub const CacheError = error{
    InvalidDimensions,
    InvalidConfig,
    ExpertOutOfRange,
    ExpertNotRegistered,
    BlobExceedsSlotCapacity,
    InvalidParamsOffset,
    NoEvictableSlot,
    TransportUnavailable,
    OutOfMemory,
} || aio.EngineError;

pub const CacheConfig = struct {
    num_layers: u16,
    num_experts: u16,
    /// Number of cache accesses between periodic LFU halvings (`0` disables decay).
    decay_interval: u64 = 1024,
    /// Fixed-point LFU score increment per access (`LFU_SCALE = 256` for smooth halving).
    lfu_increment: u32 = 256,
    /// Maximum time to wait for an in-flight or synchronous cold-expert read.
    io_timeout_ms: u64 = 10_000,
};

pub const PrefetchOutcome = enum {
    /// Expert is permanently wired (`local_wired_ram`); zero cache or I/O work.
    already_hot,
    /// Cold expert is already resident (`.ready`) in the slot pool; LFU score updated.
    already_resident,
    /// Cold expert already has an in-flight read; coalesced with zero duplicate I/O.
    coalesced_inflight,
    /// Cold expert was assigned a slot and its 3 async `.amf` reads were submitted.
    submitted,
};

pub const ResidencyState = enum {
    unregistered,
    hot,
    cached_ready,
    cached_inflight,
    cold_nvme,
    remote_peer,
};

/// Pinned resident view returned by `ExpertCache.acquirePinned`. Remains pinned
/// and guaranteed immune to eviction until `ExpertCache.releasePins()` (or
/// `ExpertCache.releasePin`).
pub const PinnedSlot = struct {
    key: ExpertKey,
    /// `null` for hot experts (`local_wired_ram`); `0..num_slots-1` for cold slots.
    slot_index: ?u32,
    /// Ready-to-write `moe.ExpertWeights` for `layer.expert_table[e]`.
    weights: moe.ExpertWeights,
    /// Owning `metal.Buffer` (hot weight shard or cold slot pool buffer).
    metal_buffer: ?metal.Buffer,
    gate_bytes: []align(page.APPLE_PAGE_SIZE) const u8,
    up_bytes: []align(page.APPLE_PAGE_SIZE) const u8,
    down_bytes: []align(page.APPLE_PAGE_SIZE) const u8,
    is_hot: bool,
};

pub const Telemetry = struct {
    hot_hits: u64 = 0,
    cache_hits: u64 = 0,
    prefetch_hits: u64 = 0,
    sync_misses: u64 = 0,
    evictions: u64 = 0,
    bytes_read: u64 = 0,
    unused_prefetched_experts: u64 = 0,
    prefetch_submitted: u64 = 0,
    prefetch_coalesced: u64 = 0,
    prefetch_wait_ns: u64 = 0,
    sync_miss_wait_ns: u64 = 0,
    io_errors: u64 = 0,

    pub fn prefetchWaitUs(self: Telemetry) f64 {
        return @as(f64, @floatFromInt(self.prefetch_wait_ns)) / 1000.0;
    }

    pub fn syncMissWaitUs(self: Telemetry) f64 {
        return @as(f64, @floatFromInt(self.sync_miss_wait_ns)) / 1000.0;
    }

    pub fn totalColdAcquires(self: Telemetry) u64 {
        return self.cache_hits + self.prefetch_hits + self.sync_misses;
    }

    pub fn coldHitRate(self: Telemetry) f64 {
        const total = self.totalColdAcquires();
        if (total == 0) return 0.0;
        return @as(f64, @floatFromInt(self.cache_hits + self.prefetch_hits)) / @as(f64, @floatFromInt(total));
    }

    pub fn prefetchHitRate(self: Telemetry) f64 {
        const total = self.totalColdAcquires();
        if (total == 0) return 0.0;
        return @as(f64, @floatFromInt(self.prefetch_hits)) / @as(f64, @floatFromInt(total));
    }
};

const SlotState = enum(u8) {
    empty,
    inflight,
    ready,
    failed,
};

const SlotMeta = struct {
    state: SlotState = .empty,
    key: ExpertKey = .{ .layer = 0, .expert = 0 },
    pin_count: u32 = 0,
    lfu_score: u32 = 0,
    last_access_tick: u64 = 0,
    pending_blobs: u8 = 0,
    io_err_code: i32 = 0,
    io_short_read: bool = false,
    prefetched_unused: bool = false,
    generation: u32 = 1,
};

const ExpertRecord = struct {
    placement: ?ExpertPlacement = null,
    /// Slot index in `ExpertSlotPool`, or `INVALID_SLOT` when not resident/in-flight.
    slot: u32 = INVALID_SLOT,
};

/// Wired cold-expert cache and async prefetch coordinator.
///
/// Startup allocates the flat `[num_layers * num_experts]` residency table and
/// per-slot metadata once; every hot-path method (`prefetch`, `acquirePinned`,
/// `pollCompletions`, `releasePins`) performs zero dynamic heap allocation.
pub const ExpertCache = struct {
    num_layers: u16,
    num_experts: u16,
    decay_interval: u64,
    lfu_increment: u32,
    io_timeout_ms: u64,

    pool: *ExpertSlotPool,
    reader: *aio.AsyncExpertReader,

    /// Flat `[num_layers * num_experts]` table indexed by `layer * num_experts + expert`.
    records: []ExpertRecord,
    /// Per-slot metadata indexed by `0..pool.numSlots()-1`.
    slots: []SlotMeta,
    /// Preallocated list of currently pinned slot indices for O(P) `releasePins()`.
    pinned_slots: []u32,
    pinned_slot_count: usize,

    access_tick: u64,
    stats: Telemetry,

    pub fn init(
        allocator: std.mem.Allocator,
        pool: *ExpertSlotPool,
        reader: *aio.AsyncExpertReader,
        config: CacheConfig,
    ) CacheError!ExpertCache {
        if (config.num_layers == 0 or config.num_experts == 0) return error.InvalidDimensions;
        if (config.lfu_increment == 0 or config.lfu_increment > MAX_LFU_SCORE) return error.InvalidConfig;
        if (reader.maxInflight() < BLOBS_PER_EXPERT) return error.InvalidConfig;

        const total_experts = std.math.mul(usize, config.num_layers, config.num_experts) catch return error.InvalidDimensions;
        const num_slots: usize = pool.numSlots();

        const records = try allocator.alloc(ExpertRecord, total_experts);
        errdefer allocator.free(records);
        @memset(records, .{});

        const slots = try allocator.alloc(SlotMeta, num_slots);
        errdefer allocator.free(slots);
        @memset(slots, .{});

        const pinned_slots = try allocator.alloc(u32, num_slots);
        errdefer allocator.free(pinned_slots);

        return .{
            .num_layers = config.num_layers,
            .num_experts = config.num_experts,
            .decay_interval = config.decay_interval,
            .lfu_increment = config.lfu_increment,
            .io_timeout_ms = config.io_timeout_ms,
            .pool = pool,
            .reader = reader,
            .records = records,
            .slots = slots,
            .pinned_slots = pinned_slots,
            .pinned_slot_count = 0,
            .access_tick = 0,
            .stats = .{},
        };
    }

    /// Drains any in-flight reads targeting `pool` before releasing startup
    /// metadata so no background DMA can outlive the slot pool.
    pub fn deinit(self: *ExpertCache, allocator: std.mem.Allocator) void {
        self.drainInflightBestEffort();
        allocator.free(self.pinned_slots);
        allocator.free(self.slots);
        allocator.free(self.records);
        self.pinned_slots = &.{};
        self.slots = &.{};
        self.records = &.{};
        self.pinned_slot_count = 0;
    }

    /// Registers a hot expert (`local_wired_ram`) at startup.
    pub fn registerHot(self: *ExpertCache, layer: u16, expert_idx: u16, hot: HotExpertTriple) CacheError!void {
        const idx = try self.flatIndex(layer, expert_idx);
        self.records[idx] = .{
            .placement = .{ .local_wired_ram = hot },
            .slot = INVALID_SLOT,
        };
    }

    /// Registers a cold expert (`local_nvme_ssd`) at startup, validating its
    /// three blob sizes fit inside the `ExpertSlotPool` sub-regions.
    pub fn registerCold(self: *ExpertCache, layer: u16, expert_idx: u16, cold: ColdExpertTriple) CacheError!void {
        const idx = try self.flatIndex(layer, expert_idx);
        if (cold.fd < 0) return error.InvalidArgument;
        if (cold.gate.len > self.pool.spec.gate_blob_bytes or
            cold.up.len > self.pool.spec.up_blob_bytes or
            cold.down.len > self.pool.spec.down_blob_bytes)
        {
            return error.BlobExceedsSlotCapacity;
        }
        if ((cold.gate_params_offset != 0 and cold.gate_params_offset >= cold.gate.len) or
            (cold.up_params_offset != 0 and cold.up_params_offset >= cold.up.len) or
            (cold.down_params_offset != 0 and cold.down_params_offset >= cold.down.len))
        {
            return error.InvalidParamsOffset;
        }
        self.records[idx] = .{
            .placement = .{ .local_nvme_ssd = cold },
            .slot = INVALID_SLOT,
        };
    }

    /// Registers a reserved Thunderbolt peer expert (`thunderbolt_peer`).
    pub fn registerPeer(self: *ExpertCache, layer: u16, expert_idx: u16, peer: expert.PeerExpert) CacheError!void {
        const idx = try self.flatIndex(layer, expert_idx);
        self.records[idx] = .{
            .placement = .{ .thunderbolt_peer = peer },
            .slot = INVALID_SLOT,
        };
    }

    /// Queries the current residency state of `(layer, expert_idx)` without mutating state.
    pub fn residencyState(self: *const ExpertCache, layer: u16, expert_idx: u16) ResidencyState {
        if (layer >= self.num_layers or expert_idx >= self.num_experts) return .unregistered;
        const rec = self.records[@as(usize, layer) * self.num_experts + expert_idx];
        const placement = rec.placement orelse return .unregistered;
        return switch (placement) {
            .local_wired_ram => .hot,
            .thunderbolt_peer => .remote_peer,
            .local_nvme_ssd => blk: {
                if (rec.slot == INVALID_SLOT) break :blk .cold_nvme;
                break :blk switch (self.slots[rec.slot].state) {
                    .ready => .cached_ready,
                    .inflight => .cached_inflight,
                    .empty, .failed => .cold_nvme,
                };
            },
        };
    }

    /// Returns `true` if `(layer, expert_idx)` is immediately resident (hot or cached `.ready`).
    pub fn isResident(self: *const ExpertCache, layer: u16, expert_idx: u16) bool {
        return switch (self.residencyState(layer, expert_idx)) {
            .hot, .cached_ready => true,
            else => false,
        };
    }

    /// Returns `true` if `(layer, expert_idx)` is hot, cached `.ready`, or `.inflight`.
    pub fn isResidentOrInflight(self: *const ExpertCache, layer: u16, expert_idx: u16) bool {
        return switch (self.residencyState(layer, expert_idx)) {
            .hot, .cached_ready, .cached_inflight => true,
            else => false,
        };
    }

    pub fn telemetry(self: *const ExpertCache) Telemetry {
        return self.stats;
    }

    pub fn resetTelemetry(self: *ExpertCache) void {
        self.stats = .{};
    }

    pub fn pinnedCount(self: *const ExpertCache) usize {
        return self.pinned_slot_count;
    }

    pub fn slotCount(self: *const ExpertCache) u32 {
        return @intCast(self.slots.len);
    }

    pub fn residentColdCount(self: *const ExpertCache) usize {
        var count: usize = 0;
        for (self.slots) |slot| {
            if (slot.state == .ready) count += 1;
        }
        return count;
    }

    pub fn inflightColdCount(self: *const ExpertCache) usize {
        var count: usize = 0;
        for (self.slots) |slot| {
            if (slot.state == .inflight) count += 1;
        }
        return count;
    }

    /// Returns the current decayed LFU score of a resident/in-flight cold expert,
    /// or `null` if it is hot, unregistered, or not currently in the slot pool.
    pub fn slotLfuScore(self: *const ExpertCache, layer: u16, expert_idx: u16) ?u32 {
        if (layer >= self.num_layers or expert_idx >= self.num_experts) return null;
        const rec = self.records[@as(usize, layer) * self.num_experts + expert_idx];
        if (rec.slot == INVALID_SLOT) return null;
        const slot = self.slots[rec.slot];
        if (slot.state != .ready and slot.state != .inflight) return null;
        return slot.lfu_score;
    }

    /// Populates `out_cold_nonresident[0..num_experts]` for `layer`: `true` if
    /// expert `e` is cold on NVMe and not currently `.ready` in the wired pool
    /// (used by D7 byte-budget tree pruning `cold_penalty_bytes`). Zero alloc.
    pub fn layerColdMask(self: *const ExpertCache, layer: u16, out_cold_nonresident: []bool) CacheError!void {
        if (layer >= self.num_layers) return error.ExpertOutOfRange;
        if (out_cold_nonresident.len < self.num_experts) return error.InvalidArgument;
        const base = @as(usize, layer) * @as(usize, self.num_experts);
        for (0..self.num_experts) |e| {
            const rec = self.records[base + e];
            out_cold_nonresident[e] = if (rec.placement) |p| switch (p) {
                .local_wired_ram => false,
                .thunderbolt_peer => true,
                .local_nvme_ssd => rec.slot == INVALID_SLOT or self.slots[rec.slot].state != .ready,
            } else false;
        }
    }

    /// Counts slots currently resident in the pool that were prefetched and have
    /// not yet been consumed by `acquirePinned`.
    pub fn countCurrentlyUnusedPrefetched(self: *const ExpertCache) u64 {
        var count: u64 = 0;
        for (self.slots) |slot| {
            if (slot.state == .ready and slot.prefetched_unused) count += 1;
        }
        return count;
    }

    /// Explicitly halves every non-empty slot's LFU score (also triggered
    /// automatically every `decay_interval` accesses).
    pub fn decayAll(self: *ExpertCache) void {
        for (self.slots) |*slot| {
            if (slot.state != .empty) {
                slot.lfu_score >>= 1;
            }
        }
    }

    /// Non-blocking poll absorbing completed async I/O sub-blobs from `reader`.
    /// Zero dynamic heap allocation.
    pub fn pollCompletions(self: *ExpertCache) aio.EngineError!usize {
        var comps: [aio.MAX_INFLIGHT]aio.Completion = undefined;
        var total: usize = 0;
        while (self.reader.inflightCount() > 0) {
            const n = try self.reader.poll(&comps);
            if (n == 0) break;
            for (comps[0..n]) |c| {
                self.handleCompletion(c);
            }
            total += n;
        }
        return total;
    }

    /// Waits for at least one async I/O completion from `reader` (up to `timeout_ms`).
    pub fn waitCompletions(self: *ExpertCache, timeout_ms: ?u64) aio.EngineError!usize {
        if (self.reader.inflightCount() == 0) return 0;
        var comps: [aio.MAX_INFLIGHT]aio.Completion = undefined;
        const n = try self.reader.wait(&comps, timeout_ms);
        for (comps[0..n]) |c| {
            self.handleCompletion(c);
        }
        return n;
    }

    /// Waits until all in-flight cold-expert reads have completed.
    pub fn drainInflight(self: *ExpertCache) CacheError!void {
        while (self.reader.inflightCount() > 0) {
            const n = try self.waitCompletions(self.io_timeout_ms);
            if (n == 0 and self.reader.inflightCount() > 0) return error.Timeout;
        }
    }

    /// Asynchronously prefetches `(layer, expert_idx)` into the wired slot pool.
    ///
    /// - Hot experts return `.already_hot` immediately.
    /// - Already resident cold experts update LFU + recency and return `.already_resident`.
    /// - Already in-flight cold experts coalesce without duplicate I/O (`.coalesced_inflight`).
    /// - Missing cold experts evict the lowest-score unpinned `.ready` slot and
    ///   submit 3 async reads (`gate`, `up`, `down`) with zero heap allocation.
    pub fn prefetch(self: *ExpertCache, layer: u16, expert_idx: u16) CacheError!PrefetchOutcome {
        const idx = try self.flatIndex(layer, expert_idx);
        _ = try self.pollCompletions();

        const rec = &self.records[idx];
        const placement = rec.placement orelse return error.ExpertNotRegistered;
        const cold = switch (placement) {
            .local_wired_ram => return .already_hot,
            .thunderbolt_peer => return error.TransportUnavailable,
            .local_nvme_ssd => |c| c,
        };

        if (rec.slot != INVALID_SLOT) {
            const s = rec.slot;
            const slot = &self.slots[s];
            switch (slot.state) {
                .ready => {
                    self.touchSlot(s);
                    return .already_resident;
                },
                .inflight => {
                    self.stats.prefetch_coalesced += 1;
                    return .coalesced_inflight;
                },
                .empty, .failed => {
                    rec.slot = INVALID_SLOT;
                },
            }
        }

        // Check I/O queue headroom before evicting any resident victim slot.
        if (self.reader.availableSlots() < BLOBS_PER_EXPERT) {
            return error.QueueFull;
        }

        const victim_idx = self.selectEvictionVictim() orelse return error.NoEvictableSlot;
        self.evictIfReady(victim_idx);

        try self.startSlotLoad(victim_idx, .{ .layer = layer, .expert = expert_idx }, idx, cold, true, false);
        self.stats.prefetch_submitted += 1;
        return .submitted;
    }

    /// Acquires `(layer, expert_idx)` and pins its slot (`pin_count += 1`) for
    /// the current verification pass so it can never be evicted mid-pass.
    ///
    /// - Hot experts increment `hot_hits` with zero slot consumption.
    /// - Ready cold slots record `prefetch_hits` (first use after prefetch) or `cache_hits`.
    /// - In-flight cold slots wait only for that slot's completion (`prefetch_wait_ns`).
    /// - Missing cold slots evict an unpinned victim (or return `error.NoEvictableSlot`
    ///   if all slots are pinned) and issue a high-priority synchronous read (`sync_misses`).
    pub fn acquirePinned(self: *ExpertCache, layer: u16, expert_idx: u16) CacheError!PinnedSlot {
        const idx = try self.flatIndex(layer, expert_idx);
        _ = try self.pollCompletions();

        const rec = &self.records[idx];
        const placement = rec.placement orelse return error.ExpertNotRegistered;
        const key: ExpertKey = .{ .layer = layer, .expert = expert_idx };

        switch (placement) {
            .local_wired_ram => |hot| {
                self.stats.hot_hits += 1;
                return .{
                    .key = key,
                    .slot_index = null,
                    .weights = hot.weights,
                    .metal_buffer = hot.metal_buffer,
                    .gate_bytes = hot.gate.bytes,
                    .up_bytes = hot.up.bytes,
                    .down_bytes = hot.down.bytes,
                    .is_hot = true,
                };
            },
            .thunderbolt_peer => return error.TransportUnavailable,
            .local_nvme_ssd => |cold| {
                if (rec.slot != INVALID_SLOT) {
                    const s = rec.slot;
                    const slot = &self.slots[s];
                    switch (slot.state) {
                        .ready => {
                            self.pinSlot(s);
                            self.touchSlot(s);
                            if (slot.prefetched_unused) {
                                slot.prefetched_unused = false;
                                self.stats.prefetch_hits += 1;
                            } else {
                                self.stats.cache_hits += 1;
                            }
                            return self.buildColdPinnedSlot(key, s, cold);
                        },
                        .inflight => {
                            // Pin immediately so completion cannot be evicted before we return.
                            self.pinSlot(s);
                            const t0 = monotonicNowNs();
                            self.waitSlotDone(s) catch |err| {
                                self.unpinSlot(s);
                                return err;
                            };
                            const t1 = monotonicNowNs();
                            self.stats.prefetch_wait_ns += (t1 -| t0);

                            if (slot.state == .failed) {
                                const err = mapSlotFailure(slot);
                                self.clearFailedPinnedSlot(s, idx);
                                return err;
                            }
                            self.touchSlot(s);
                            if (slot.prefetched_unused) {
                                slot.prefetched_unused = false;
                                self.stats.prefetch_hits += 1;
                            } else {
                                self.stats.cache_hits += 1;
                            }
                            return self.buildColdPinnedSlot(key, s, cold);
                        },
                        .failed => {
                            self.clearFailedUnpinnedSlot(s, idx);
                        },
                        .empty => {
                            rec.slot = INVALID_SLOT;
                        },
                    }
                }

                // Cold miss: if every slot is actively pinned, refuse eviction immediately.
                if (self.pinned_slot_count >= self.slots.len) {
                    return error.NoEvictableSlot;
                }

                const miss_t0 = monotonicNowNs();

                // Acquire an empty or unpinned ready slot; if all unpinned slots are
                // currently in-flight prefetches, wait cleanly for one DMA to finish.
                var victim_idx: ?u32 = self.selectEvictionVictim();
                while (victim_idx == null) {
                    if (self.pinned_slot_count >= self.slots.len) return error.NoEvictableSlot;
                    if (self.reader.inflightCount() == 0) return error.NoEvictableSlot;
                    const got = try self.waitCompletions(self.io_timeout_ms);
                    if (got == 0) return error.Timeout;
                    victim_idx = self.selectEvictionVictim();
                }

                // Ensure I/O engine has room for all 3 sub-blobs before evicting victim.
                while (self.reader.availableSlots() < BLOBS_PER_EXPERT) {
                    const got = try self.waitCompletions(self.io_timeout_ms);
                    if (got == 0 and self.reader.availableSlots() < BLOBS_PER_EXPERT) {
                        return error.Timeout;
                    }
                }

                const s = victim_idx.?;
                self.evictIfReady(s);

                try self.startSlotLoad(s, key, idx, cold, false, true);
                self.waitSlotDone(s) catch |err| {
                    self.unpinSlot(s);
                    return err;
                };

                const miss_t1 = monotonicNowNs();
                self.stats.sync_miss_wait_ns += (miss_t1 -| miss_t0);

                const slot = &self.slots[s];
                if (slot.state == .failed) {
                    const err = mapSlotFailure(slot);
                    self.clearFailedPinnedSlot(s, idx);
                    return err;
                }

                self.touchSlot(s);
                self.stats.sync_misses += 1;
                return self.buildColdPinnedSlot(key, s, cold);
            },
        }
    }

    /// Releases all active verification-pass pins (`pin_count = 0`). Zero heap allocation.
    pub fn releasePins(self: *ExpertCache) void {
        for (self.pinned_slots[0..self.pinned_slot_count]) |slot_idx| {
            self.slots[slot_idx].pin_count = 0;
        }
        self.pinned_slot_count = 0;
    }

    /// Decrements a single slot's pin count (no-op if `slot_index` is invalid or unpinned).
    pub fn releasePin(self: *ExpertCache, slot_index: u32) void {
        if (slot_index >= self.slots.len) return;
        self.unpinSlot(slot_index);
    }

    // -----------------------------------------------------------------------
    // Internal helpers (zero heap allocation)
    // -----------------------------------------------------------------------

    fn flatIndex(self: *const ExpertCache, layer: u16, expert_idx: u16) CacheError!usize {
        if (layer >= self.num_layers or expert_idx >= self.num_experts) {
            return error.ExpertOutOfRange;
        }
        return @as(usize, layer) * @as(usize, self.num_experts) + @as(usize, expert_idx);
    }

    fn nextTick(self: *ExpertCache) u64 {
        self.access_tick += 1;
        if (self.decay_interval > 0 and (self.access_tick % self.decay_interval) == 0) {
            self.decayAll();
        }
        return self.access_tick;
    }

    fn touchSlot(self: *ExpertCache, slot_idx: u32) void {
        const tick = self.nextTick();
        const slot = &self.slots[slot_idx];
        slot.lfu_score = @min(slot.lfu_score +| self.lfu_increment, MAX_LFU_SCORE);
        slot.last_access_tick = tick;
    }

    fn pinSlot(self: *ExpertCache, slot_idx: u32) void {
        const slot = &self.slots[slot_idx];
        if (slot.pin_count == 0) {
            std.debug.assert(self.pinned_slot_count < self.pinned_slots.len);
            self.pinned_slots[self.pinned_slot_count] = slot_idx;
            self.pinned_slot_count += 1;
        }
        slot.pin_count += 1;
    }

    fn unpinSlot(self: *ExpertCache, slot_idx: u32) void {
        const slot = &self.slots[slot_idx];
        if (slot.pin_count == 0) return;
        slot.pin_count -= 1;
        if (slot.pin_count == 0) {
            for (0..self.pinned_slot_count) |i| {
                if (self.pinned_slots[i] == slot_idx) {
                    self.pinned_slot_count -= 1;
                    self.pinned_slots[i] = self.pinned_slots[self.pinned_slot_count];
                    break;
                }
            }
        }
    }

    fn selectEvictionVictim(self: *const ExpertCache) ?u32 {
        var best: ?u32 = null;
        var best_score: u32 = std.math.maxInt(u32);
        var best_tick: u64 = std.math.maxInt(u64);

        for (self.slots, 0..) |slot, i| {
            if (slot.state == .empty) return @intCast(i);
            if (slot.state != .ready or slot.pin_count > 0) continue;
            if (best == null or
                slot.lfu_score < best_score or
                (slot.lfu_score == best_score and slot.last_access_tick < best_tick))
            {
                best = @intCast(i);
                best_score = slot.lfu_score;
                best_tick = slot.last_access_tick;
            }
        }
        return best;
    }

    fn evictIfReady(self: *ExpertCache, slot_idx: u32) void {
        const slot = &self.slots[slot_idx];
        std.debug.assert(slot.pin_count == 0);
        if (slot.state == .ready) {
            const old_idx = @as(usize, slot.key.layer) * @as(usize, self.num_experts) + @as(usize, slot.key.expert);
            if (self.records[old_idx].slot == slot_idx) {
                self.records[old_idx].slot = INVALID_SLOT;
            }
            self.stats.evictions += 1;
            if (slot.prefetched_unused) {
                self.stats.unused_prefetched_experts += 1;
            }
            slot.state = .empty;
            slot.prefetched_unused = false;
        }
    }

    fn packTag(generation: u32, slot_idx: u32, blob_kind: u8) u64 {
        return (@as(u64, generation) << 32) | (@as(u64, slot_idx) << 8) | @as(u64, blob_kind);
    }

    fn startSlotLoad(
        self: *ExpertCache,
        slot_idx: u32,
        key: ExpertKey,
        record_idx: usize,
        cold: ColdExpertTriple,
        from_prefetch: bool,
        pin_immediately: bool,
    ) CacheError!void {
        const slot = &self.slots[slot_idx];
        std.debug.assert(slot.state == .empty and slot.pin_count == 0);

        slot.generation +%= 1;
        if (slot.generation == 0) slot.generation = 1;
        const gen = slot.generation;

        slot.state = .inflight;
        slot.key = key;
        slot.pending_blobs = BLOBS_PER_EXPERT;
        slot.io_err_code = 0;
        slot.io_short_read = false;
        slot.prefetched_unused = from_prefetch;
        slot.lfu_score = if (from_prefetch) self.lfu_increment else 0;
        slot.last_access_tick = self.access_tick;

        if (pin_immediately) {
            self.pinSlot(slot_idx);
        }
        self.records[record_idx].slot = slot_idx;

        const gate_len: usize = @intCast(cold.gate.len);
        const up_len: usize = @intCast(cold.up.len);
        const down_len: usize = @intCast(cold.down.len);

        const reqs = [BLOBS_PER_EXPERT]aio.ReadRequest{
            .{
                .fd = cold.fd,
                .offset = cold.gate.offset,
                .dst = @alignCast(self.pool.gateSlice(slot_idx)[0..gate_len]),
                .user_data = packTag(gen, slot_idx, 0),
            },
            .{
                .fd = cold.fd,
                .offset = cold.up.offset,
                .dst = @alignCast(self.pool.upSlice(slot_idx)[0..up_len]),
                .user_data = packTag(gen, slot_idx, 1),
            },
            .{
                .fd = cold.fd,
                .offset = cold.down.offset,
                .dst = @alignCast(self.pool.downSlice(slot_idx)[0..down_len]),
                .user_data = packTag(gen, slot_idx, 2),
            },
        };

        var submitted: u8 = 0;
        for (reqs) |req| {
            self.reader.submit(req) catch |err| {
                // Roll back safely: if prior sub-blobs were submitted to kernel/workers,
                // wait for those DMA operations to complete before clearing the slot.
                if (submitted == 0) {
                    if (pin_immediately) self.unpinSlot(slot_idx);
                    self.records[record_idx].slot = INVALID_SLOT;
                    slot.state = .empty;
                    slot.pending_blobs = 0;
                    slot.prefetched_unused = false;
                } else {
                    slot.pending_blobs = submitted;
                    slot.io_err_code = @backingInt(std.c.E.IO);
                    self.waitSlotDoneBestEffort(slot_idx);
                    if (pin_immediately) self.unpinSlot(slot_idx);
                    self.records[record_idx].slot = INVALID_SLOT;
                    slot.state = .empty;
                    slot.pending_blobs = 0;
                    slot.prefetched_unused = false;
                }
                return err;
            };
            submitted += 1;
        }
    }

    fn handleCompletion(self: *ExpertCache, comp: aio.Completion) void {
        const gen: u32 = @intCast(comp.user_data >> 32);
        const slot_idx: u32 = @intCast((comp.user_data >> 8) & 0xFF_FFFF);
        if (slot_idx >= self.slots.len) return;

        const slot = &self.slots[slot_idx];
        if (slot.state != .inflight or slot.generation != gen or slot.pending_blobs == 0) return;

        self.stats.bytes_read += comp.bytes_read;
        if (comp.err_code != 0 and slot.io_err_code == 0) {
            slot.io_err_code = comp.err_code;
        }
        if (comp.short_read) {
            slot.io_short_read = true;
        }

        slot.pending_blobs -= 1;
        if (slot.pending_blobs == 0) {
            if (slot.io_err_code == 0 and !slot.io_short_read) {
                slot.state = .ready;
            } else {
                self.stats.io_errors += 1;
                if (slot.pin_count == 0) {
                    const rec_idx = @as(usize, slot.key.layer) * @as(usize, self.num_experts) + @as(usize, slot.key.expert);
                    if (self.records[rec_idx].slot == slot_idx) {
                        self.records[rec_idx].slot = INVALID_SLOT;
                    }
                    slot.state = .empty;
                    slot.prefetched_unused = false;
                } else {
                    slot.state = .failed;
                }
            }
        }
    }

    fn waitSlotDone(self: *ExpertCache, slot_idx: u32) CacheError!void {
        const deadline_ns = monotonicNowNs() +| (self.io_timeout_ms *| std.time.ns_per_ms);
        while (self.slots[slot_idx].state == .inflight) {
            const now = monotonicNowNs();
            if (now >= deadline_ns) return error.Timeout;
            const rem_ms = @max(@as(u64, 1), (deadline_ns - now) / std.time.ns_per_ms);
            _ = try self.waitCompletions(rem_ms);
        }
    }

    fn waitSlotDoneBestEffort(self: *ExpertCache, slot_idx: u32) void {
        while (self.slots[slot_idx].state == .inflight) {
            const n = self.waitCompletions(self.io_timeout_ms) catch break;
            if (n == 0) break;
        }
    }

    fn drainInflightBestEffort(self: *ExpertCache) void {
        while (self.reader.inflightCount() > 0) {
            const n = self.waitCompletions(self.io_timeout_ms) catch break;
            if (n == 0) break;
        }
    }

    fn mapSlotFailure(slot: *const SlotMeta) CacheError {
        if (slot.io_err_code != 0) {
            if (slot.io_err_code == @backingInt(std.c.E.CANCELED)) return error.Canceled;
            return error.ReadFailed;
        }
        if (slot.io_short_read) return error.ShortRead;
        return error.ReadFailed;
    }

    fn clearFailedPinnedSlot(self: *ExpertCache, slot_idx: u32, record_idx: usize) void {
        self.unpinSlot(slot_idx);
        if (self.records[record_idx].slot == slot_idx) {
            self.records[record_idx].slot = INVALID_SLOT;
        }
        const slot = &self.slots[slot_idx];
        slot.state = .empty;
        slot.prefetched_unused = false;
        slot.io_err_code = 0;
        slot.io_short_read = false;
    }

    fn clearFailedUnpinnedSlot(self: *ExpertCache, slot_idx: u32, record_idx: usize) void {
        if (self.records[record_idx].slot == slot_idx) {
            self.records[record_idx].slot = INVALID_SLOT;
        }
        const slot = &self.slots[slot_idx];
        slot.state = .empty;
        slot.prefetched_unused = false;
        slot.io_err_code = 0;
        slot.io_short_read = false;
    }

    fn buildColdPinnedSlot(
        self: *const ExpertCache,
        key: ExpertKey,
        slot_idx: u32,
        cold: ColdExpertTriple,
    ) PinnedSlot {
        const gate_len: usize = @intCast(cold.gate.len);
        const up_len: usize = @intCast(cold.up.len);
        const down_len: usize = @intCast(cold.down.len);
        return .{
            .key = key,
            .slot_index = slot_idx,
            .weights = self.pool.slotExpertWeights(
                slot_idx,
                cold.gate_params_offset,
                cold.up_params_offset,
                cold.down_params_offset,
            ),
            .metal_buffer = self.pool.metal_buffer,
            .gate_bytes = @alignCast(self.pool.gateSlice(slot_idx)[0..gate_len]),
            .up_bytes = @alignCast(self.pool.upSlice(slot_idx)[0..up_len]),
            .down_bytes = @alignCast(self.pool.downSlice(slot_idx)[0..down_len]),
            .is_hot = false,
        };
    }
};

fn monotonicNowNs() u64 {
    return vmstat.nowNs() catch 0;
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

fn testBudget(bytes: usize) budget_mod.Budget {
    return budget_mod.Budget.compute(.{
        .memory_bytes = @max(bytes * 4, 64 * page.APPLE_PAGE_SIZE),
        .user_wire_limit = bytes * 2,
        .global_user_wire_limit = bytes * 2,
        .gpu_working_set_bytes = bytes * 2,
        .configured_cap = null,
        .headroom_bytes = page.APPLE_PAGE_SIZE,
    });
}

fn expertFillByte(layer: u16, expert_idx: u16, blob_kind: u8, page_in_blob: usize) u8 {
    const h: u32 = (@as(u32, layer) *% 73) ^ (@as(u32, expert_idx) *% 19) ^
        (@as(u32, blob_kind) *% 41) ^ (@as(u32, @truncate(page_in_blob)) *% 11) ^ 0x3D;
    return @truncate(h);
}

const SyntheticColdPack = struct {
    path_buf: [std.fs.max_path_bytes]u8,
    path_len: usize,
    file: aio.DirectFile,
    num_layers: u16,
    num_experts: u16,
    blob_bytes: usize,

    fn create(tag: []const u8, num_layers: u16, num_experts: u16, blob_bytes: usize) !SyntheticColdPack {
        var pack: SyntheticColdPack = .{
            .path_buf = undefined,
            .path_len = 0,
            .file = .{ .fd = -1, .size = 0 },
            .num_layers = num_layers,
            .num_experts = num_experts,
            .blob_bytes = blob_bytes,
        };
        const pid = std.c.getpid();
        const stamp = monotonicNowNs();
        const path_z = try paths.tmpPathFmtZ(
            &pack.path_buf,
            "aminfer_cache_test_{s}_{d}_{d}.bin",
            .{ tag, pid, stamp },
        );
        pack.path_len = path_z.len;

        const fd = std.c.open(
            path_z.ptr,
            .{ .ACCMODE = .RDWR, .CREAT = true, .TRUNC = true },
            @as(std.c.mode_t, 0o600),
        );
        if (fd < 0) return error.OpenFailed;
        errdefer _ = std.c.unlink(path_z.ptr);
        {
            defer _ = std.c.close(fd);
            try aio.enableNoCache(fd);

            const expert_bytes = blob_bytes * 3;
            const scratch = try page.WiredRegion.init(expert_bytes);
            defer scratch.deinit();

            var offset: u64 = 0;
            for (0..num_layers) |l| {
                for (0..num_experts) |e| {
                    for (0..3) |b| {
                        const dst = scratch.bytes[b * blob_bytes .. (b + 1) * blob_bytes];
                        const pages = blob_bytes / page.APPLE_PAGE_SIZE;
                        for (0..pages) |p| {
                            const val = expertFillByte(@intCast(l), @intCast(e), @intCast(b), p);
                            @memset(dst[p * page.APPLE_PAGE_SIZE .. (p + 1) * page.APPLE_PAGE_SIZE], val);
                        }
                    }
                    var done: usize = 0;
                    while (done < scratch.bytes.len) {
                        const rc = std.c.pwrite(fd, scratch.bytes.ptr + done, scratch.bytes.len - done, @intCast(offset + done));
                        if (rc < 0) {
                            if (std.c.errno(rc) == .INTR) continue;
                            return error.ReadFailed;
                        }
                        done += @intCast(rc);
                    }
                    offset += expert_bytes;
                }
            }
            if (std.c.fsync(fd) != 0) return error.ReadFailed;
        }

        pack.file = try aio.DirectFile.openReadOnly(path_z);
        return pack;
    }

    fn deinit(self: *SyntheticColdPack) void {
        self.file.close();
        const path_z: [:0]const u8 = self.path_buf[0..self.path_len :0];
        _ = std.c.unlink(path_z.ptr);
    }

    fn tripleFor(self: *const SyntheticColdPack, layer: u16, expert_idx: u16) !ColdExpertTriple {
        const flat = @as(u64, layer) * @as(u64, self.num_experts) + @as(u64, expert_idx);
        const base = flat * @as(u64, self.blob_bytes) * 3;
        return .{
            .fd = self.file.fd,
            .gate = try expert.NvmeExpert.init(0, base, self.blob_bytes),
            .up = try expert.NvmeExpert.init(0, base + self.blob_bytes, self.blob_bytes),
            .down = try expert.NvmeExpert.init(0, base + 2 * self.blob_bytes, self.blob_bytes),
            .gate_params_offset = 1024,
            .up_params_offset = 1024,
            .down_params_offset = 1024,
        };
    }

    fn expectSlotMatches(self: *const SyntheticColdPack, pinned: PinnedSlot) !void {
        const slices = [3][]const u8{ pinned.gate_bytes, pinned.up_bytes, pinned.down_bytes };
        const pages = self.blob_bytes / page.APPLE_PAGE_SIZE;
        for (slices, 0..) |sl, b| {
            try std.testing.expectEqual(self.blob_bytes, sl.len);
            for (0..pages) |p| {
                const expected = expertFillByte(pinned.key.layer, pinned.key.expert, @intCast(b), p);
                const page_sl = sl[p * page.APPLE_PAGE_SIZE .. (p + 1) * page.APPLE_PAGE_SIZE];
                try std.testing.expectEqual(expected, page_sl[0]);
                try std.testing.expectEqual(expected, page_sl[page_sl.len / 2]);
                try std.testing.expectEqual(expected, page_sl[page_sl.len - 1]);
            }
        }
    }
};

test "ExpertSlotPool carves 16 KiB-aligned slots from WiredArena with deterministic ExpertWeights offsets" {
    const gate_bytes = 2 * page.APPLE_PAGE_SIZE;
    const up_bytes = 2 * page.APPLE_PAGE_SIZE;
    const down_bytes = 3 * page.APPLE_PAGE_SIZE;
    const spec: PoolSpec = .{
        .num_slots = 4,
        .gate_blob_bytes = gate_bytes,
        .up_blob_bytes = up_bytes,
        .down_blob_bytes = down_bytes,
    };

    const prefix_bytes = page.APPLE_PAGE_SIZE;
    const total = prefix_bytes + try spec.totalBytes();
    var arena = try arena_mod.WiredArena.init(total, testBudget(total));
    defer arena.deinit();

    // Carve one leading page first to verify non-zero `arena_base_offset` math.
    _ = try arena.allocPages(prefix_bytes);

    var device = try metal.Device.open();
    defer device.close();
    const shared_buf = try metal.Buffer.wrap(device, arena.bytes());
    defer shared_buf.close();

    var pool = try ExpertSlotPool.initFromArena(&arena, shared_buf, spec);
    defer pool.deinit();

    try std.testing.expectEqual(@as(u64, prefix_bytes), pool.arena_base_offset);
    try std.testing.expectEqual(@as(usize, 7 * page.APPLE_PAGE_SIZE), pool.slot_bytes);

    const w2 = pool.slotExpertWeights(2, 512, 1024, 2048);
    const base2: u64 = prefix_bytes + 2 * 7 * page.APPLE_PAGE_SIZE;
    try std.testing.expectEqual(base2, w2.gate_qweight_offset);
    try std.testing.expectEqual(base2 + 512, w2.gate_sb_offset);
    try std.testing.expectEqual(base2 + gate_bytes, w2.up_qweight_offset);
    try std.testing.expectEqual(base2 + gate_bytes + 1024, w2.up_sb_offset);
    try std.testing.expectEqual(base2 + gate_bytes + up_bytes, w2.down_qweight_offset);
    try std.testing.expectEqual(base2 + gate_bytes + up_bytes + 2048, w2.down_sb_offset);

    // Float experts (`params_offset == 0`) keep `*_sb_offset == 0`.
    const wf = pool.slotExpertWeights(1, 0, 0, 0);
    try std.testing.expectEqual(@as(u64, 0), wf.gate_sb_offset);
    try std.testing.expectEqual(@as(u64, 0), wf.up_sb_offset);
    try std.testing.expectEqual(@as(u64, 0), wf.down_sb_offset);

    const wired = try pool.wiredSources(0);
    try std.testing.expectEqual(gate_bytes, wired[0].bytes.len);
    try std.testing.expectEqual(up_bytes, wired[1].bytes.len);
    try std.testing.expectEqual(down_bytes, wired[2].bytes.len);
}

test "ExpertCache resolves hot experts with zero slot overhead and rejects thunderbolt_peer" {
    const gpa = std.testing.allocator;
    const blob_bytes = page.APPLE_PAGE_SIZE;
    const spec: PoolSpec = .{
        .num_slots = 1,
        .gate_blob_bytes = blob_bytes,
        .up_blob_bytes = blob_bytes,
        .down_blob_bytes = blob_bytes,
    };
    const total = try spec.totalBytes();
    var pool = try ExpertSlotPool.initOwned(testBudget(total), null, spec);
    defer pool.deinit();

    var reader = try aio.AsyncExpertReader.init(gpa, .{ .backend = .kqueue_aio, .max_inflight = 8 });
    defer reader.deinit(gpa);

    var cache = try ExpertCache.init(gpa, &pool, &reader, .{ .num_layers = 1, .num_experts = 4 });
    defer cache.deinit(gpa);

    const hot_region = try page.WiredRegion.init(3 * page.APPLE_PAGE_SIZE);
    defer hot_region.deinit();
    @memset(hot_region.bytes, 0x7E);

    try cache.registerHot(0, 0, .{
        .gate = try expert.WiredExpert.init(hot_region.bytes[0..page.APPLE_PAGE_SIZE]),
        .up = try expert.WiredExpert.init(hot_region.bytes[page.APPLE_PAGE_SIZE .. 2 * page.APPLE_PAGE_SIZE]),
        .down = try expert.WiredExpert.init(hot_region.bytes[2 * page.APPLE_PAGE_SIZE .. 3 * page.APPLE_PAGE_SIZE]),
        .weights = .{
            .gate_qweight_offset = 0,
            .gate_sb_offset = 64,
            .up_qweight_offset = page.APPLE_PAGE_SIZE,
            .up_sb_offset = page.APPLE_PAGE_SIZE + 64,
            .down_qweight_offset = 2 * page.APPLE_PAGE_SIZE,
            .down_sb_offset = 2 * page.APPLE_PAGE_SIZE + 64,
        },
    });
    try cache.registerPeer(0, 1, .{ .peer = 1, .remote_offset = 0, .len = page.APPLE_PAGE_SIZE });

    try std.testing.expectEqual(PrefetchOutcome.already_hot, try cache.prefetch(0, 0));
    const pinned = try cache.acquirePinned(0, 0);
    try std.testing.expect(pinned.is_hot);
    try std.testing.expectEqual(@as(?u32, null), pinned.slot_index);
    try std.testing.expectEqual(@as(u8, 0x7E), pinned.gate_bytes[0]);
    try std.testing.expectEqual(@as(usize, 0), cache.pinnedCount());
    try std.testing.expectEqual(@as(u64, 1), cache.telemetry().hot_hits);

    try std.testing.expectError(error.TransportUnavailable, cache.prefetch(0, 1));
    try std.testing.expectError(error.TransportUnavailable, cache.acquirePinned(0, 1));
}

test "ExpertCache pin-safety prevents mid-pass eviction and returns NoEvictableSlot when full" {
    const gpa = std.testing.allocator;
    const blob_bytes = 2 * page.APPLE_PAGE_SIZE;
    var pack = try SyntheticColdPack.create("pinsafe", 1, 4, blob_bytes);
    defer pack.deinit();

    const spec: PoolSpec = .{
        .num_slots = 2,
        .gate_blob_bytes = blob_bytes,
        .up_blob_bytes = blob_bytes,
        .down_blob_bytes = blob_bytes,
    };
    var pool = try ExpertSlotPool.initOwned(testBudget(try spec.totalBytes()), null, spec);
    defer pool.deinit();

    var reader = try aio.AsyncExpertReader.init(gpa, .{ .backend = .kqueue_aio, .max_inflight = 16 });
    defer reader.deinit(gpa);

    var cache = try ExpertCache.init(gpa, &pool, &reader, .{ .num_layers = 1, .num_experts = 4 });
    defer cache.deinit(gpa);

    for (0..4) |e| {
        try cache.registerCold(0, @intCast(e), try pack.tripleFor(0, @intCast(e)));
    }

    // Pin 2 cold experts filling the 2-slot pool completely.
    const p0 = try cache.acquirePinned(0, 0);
    const p1 = try cache.acquirePinned(0, 1);
    try pack.expectSlotMatches(p0);
    try pack.expectSlotMatches(p1);
    try std.testing.expectEqual(@as(usize, 2), cache.pinnedCount());

    // Re-acquiring an already pinned expert succeeds without consuming another slot.
    const p0_again = try cache.acquirePinned(0, 0);
    try std.testing.expectEqual(p0.slot_index, p0_again.slot_index);
    try std.testing.expectEqual(@as(usize, 2), cache.pinnedCount());

    // Under pressure: neither `prefetch` nor `acquirePinned` may evict pinned slots.
    try std.testing.expectError(error.NoEvictableSlot, cache.prefetch(0, 2));
    try std.testing.expectError(error.NoEvictableSlot, cache.acquirePinned(0, 2));
    try std.testing.expectEqual(@as(u64, 0), cache.telemetry().evictions);
    try std.testing.expect(cache.isResident(0, 0));
    try std.testing.expect(cache.isResident(0, 1));
    try std.testing.expect(!cache.isResident(0, 2));

    // Releasing verification-pass pins restores evictability immediately.
    cache.releasePins();
    try std.testing.expectEqual(@as(usize, 0), cache.pinnedCount());

    const p2 = try cache.acquirePinned(0, 2);
    try pack.expectSlotMatches(p2);
    try std.testing.expectEqual(@as(u64, 1), cache.telemetry().evictions);
    cache.releasePins();
}

test "ExpertCache LFU with periodic recency decay evicts in exact expected order" {
    const gpa = std.testing.allocator;
    const blob_bytes = page.APPLE_PAGE_SIZE;
    var pack = try SyntheticColdPack.create("lfudecay", 1, 4, blob_bytes);
    defer pack.deinit();

    const spec: PoolSpec = .{
        .num_slots = 2,
        .gate_blob_bytes = blob_bytes,
        .up_blob_bytes = blob_bytes,
        .down_blob_bytes = blob_bytes,
    };
    var pool = try ExpertSlotPool.initOwned(testBudget(try spec.totalBytes()), null, spec);
    defer pool.deinit();

    var reader = try aio.AsyncExpertReader.init(gpa, .{
        .backend = .ecore_pread,
        .max_inflight = 16,
        .worker_count = 2,
    });
    defer reader.deinit(gpa);

    // Disable automatic decay for Part A (pure LFU check).
    var cache = try ExpertCache.init(gpa, &pool, &reader, .{
        .num_layers = 1,
        .num_experts = 4,
        .decay_interval = 0,
    });
    defer cache.deinit(gpa);

    for (0..4) |e| {
        try cache.registerCold(0, @intCast(e), try pack.tripleFor(0, @intCast(e)));
    }

    // Part A: E0 gets 5 hits, then E1 gets 1 more recent hit. Loading E2 must evict E1 (lower LFU).
    for (0..5) |_| {
        _ = try cache.acquirePinned(0, 0);
        cache.releasePins();
    }
    _ = try cache.acquirePinned(0, 1);
    cache.releasePins();

    const p2 = try cache.acquirePinned(0, 2);
    try pack.expectSlotMatches(p2);
    cache.releasePins();

    try std.testing.expect(cache.isResident(0, 0));
    try std.testing.expect(!cache.isResident(0, 1));
    try std.testing.expect(cache.isResident(0, 2));

    // Part B: Recency decay — enable periodic halving every 2 accesses. E0 goes
    // cold while E2 gets fresh hits across several halvings, aging E0 below E2.
    cache.decay_interval = 2;
    for (0..8) |_| {
        _ = try cache.acquirePinned(0, 2);
        cache.releasePins();
    }

    // Now E3 must evict the stale E0 (decayed score) and keep recently-hot E2!
    const p3 = try cache.acquirePinned(0, 3);
    try pack.expectSlotMatches(p3);
    cache.releasePins();

    try std.testing.expect(!cache.isResident(0, 0));
    try std.testing.expect(cache.isResident(0, 2));
    try std.testing.expect(cache.isResident(0, 3));
}

test "ExpertCache deduplicates in-flight prefetches and tracks unused prefetched evictions" {
    const gpa = std.testing.allocator;
    const blob_bytes = 32 * page.APPLE_PAGE_SIZE; // 512 KiB per blob, 1.5 MiB per expert
    var pack = try SyntheticColdPack.create("dedup", 1, 6, blob_bytes);
    defer pack.deinit();

    const spec: PoolSpec = .{
        .num_slots = 4,
        .gate_blob_bytes = blob_bytes,
        .up_blob_bytes = blob_bytes,
        .down_blob_bytes = blob_bytes,
    };
    var pool = try ExpertSlotPool.initOwned(testBudget(try spec.totalBytes()), null, spec);
    defer pool.deinit();

    var reader = try aio.AsyncExpertReader.init(gpa, .{
        .backend = .ecore_pread,
        .max_inflight = 32,
        .worker_count = 1,
    });
    defer reader.deinit(gpa);

    var cache = try ExpertCache.init(gpa, &pool, &reader, .{ .num_layers = 1, .num_experts = 6 });
    defer cache.deinit(gpa);

    for (0..6) |e| {
        try cache.registerCold(0, @intCast(e), try pack.tripleFor(0, @intCast(e)));
    }

    // Queue 4 cold experts back-to-back on a single worker so E3 is deterministically
    // queued in-flight behind E0..E2; re-prefetching E3 must coalesce without duplicate I/O.
    for (0..4) |e| {
        try std.testing.expectEqual(PrefetchOutcome.submitted, try cache.prefetch(0, @intCast(e)));
    }
    const dup = try cache.prefetch(0, 3);
    try std.testing.expectEqual(PrefetchOutcome.coalesced_inflight, dup);
    try std.testing.expectEqual(@as(u64, 4), cache.telemetry().prefetch_submitted);
    try std.testing.expectEqual(@as(u64, 1), cache.telemetry().prefetch_coalesced);

    // Acquiring E3 waits only for its in-flight read and records a prefetch hit (0 sync misses).
    const p3 = try cache.acquirePinned(0, 3);
    try pack.expectSlotMatches(p3);
    try std.testing.expectEqual(@as(u64, 1), cache.telemetry().prefetch_hits);
    try std.testing.expectEqual(@as(u64, 0), cache.telemetry().sync_misses);
    cache.releasePins();

    try cache.drainInflight();
    // E0, E1, E2 were prefetched and never acquired yet -> 3 currently unused.
    try std.testing.expectEqual(@as(u64, 3), cache.countCurrentlyUnusedPrefetched());

    // Boost E3 so one of {E0, E1, E2} is evicted when E4 loads -> unused_prefetched_experts == 1.
    _ = try cache.acquirePinned(0, 3);
    cache.releasePins();

    const p4 = try cache.acquirePinned(0, 4);
    try pack.expectSlotMatches(p4);
    cache.releasePins();

    try std.testing.expectEqual(@as(u64, 1), cache.telemetry().unused_prefetched_experts);
    try std.testing.expectEqual(@as(u64, 2), cache.countCurrentlyUnusedPrefetched());

    // Verify layerColdMask matches exact current residency.
    var cold_mask: [6]bool = undefined;
    try cache.layerColdMask(0, &cold_mask);
    try std.testing.expect(!cold_mask[3]); // resident
    try std.testing.expect(!cold_mask[4]); // resident
    try std.testing.expect(cold_mask[5]); // cold non-resident
}

test "ExpertCache recovers cleanly from ShortRead without leaking pins or slots" {
    const gpa = std.testing.allocator;
    const blob_bytes = page.APPLE_PAGE_SIZE;
    var pack = try SyntheticColdPack.create("shortread", 1, 1, blob_bytes);
    defer pack.deinit();

    const spec: PoolSpec = .{
        .num_slots = 1,
        .gate_blob_bytes = 2 * blob_bytes,
        .up_blob_bytes = blob_bytes,
        .down_blob_bytes = blob_bytes,
    };
    var pool = try ExpertSlotPool.initOwned(testBudget(try spec.totalBytes()), null, spec);
    defer pool.deinit();

    var reader = try aio.AsyncExpertReader.init(gpa, .{ .backend = .kqueue_aio, .max_inflight = 8 });
    defer reader.deinit(gpa);

    var cache = try ExpertCache.init(gpa, &pool, &reader, .{ .num_layers = 1, .num_experts = 2 });
    defer cache.deinit(gpa);

    // Register a valid expert 0 and an out-of-EOF expert 1 (offset past file end).
    try cache.registerCold(0, 0, try pack.tripleFor(0, 0));
    try cache.registerCold(0, 1, .{
        .fd = pack.file.fd,
        .gate = try expert.NvmeExpert.init(0, pack.file.size, 2 * blob_bytes),
        .up = try expert.NvmeExpert.init(0, 0, blob_bytes),
        .down = try expert.NvmeExpert.init(0, blob_bytes, blob_bytes),
    });

    // Sync miss on out-of-EOF expert must fail with ShortRead and leave 0 pins leaked.
    try std.testing.expectError(error.ShortRead, cache.acquirePinned(0, 1));
    try std.testing.expectEqual(@as(usize, 0), cache.pinnedCount());
    try std.testing.expectEqual(@as(usize, 0), cache.residentColdCount());
    try std.testing.expectEqual(@as(u64, 1), cache.telemetry().io_errors);

    // The single slot remains usable for valid expert 0 afterwards.
    const p0 = try cache.acquirePinned(0, 0);
    try pack.expectSlotMatches(p0);
    try std.testing.expectEqual(@as(usize, 1), cache.pinnedCount());
    cache.releasePins();
}

test "ExpertCache async prefetch delivers zero-miss hits on both kqueue_aio and ecore_pread" {
    const gpa = std.testing.allocator;
    const blob_bytes = 4 * page.APPLE_PAGE_SIZE; // 64 KiB per blob, 192 KiB per expert
    var pack = try SyntheticColdPack.create("zeromiss", 2, 4, blob_bytes);
    defer pack.deinit();

    const spec: PoolSpec = .{
        .num_slots = 8,
        .gate_blob_bytes = blob_bytes,
        .up_blob_bytes = blob_bytes,
        .down_blob_bytes = blob_bytes,
    };

    for ([_]aio.IoBackend{ .kqueue_aio, .ecore_pread }) |backend| {
        var pool = try ExpertSlotPool.initOwned(testBudget(try spec.totalBytes()), null, spec);
        defer pool.deinit();

        var reader = try aio.AsyncExpertReader.init(gpa, .{
            .backend = backend,
            .max_inflight = 32,
            .worker_count = 2,
        });
        defer reader.deinit(gpa);

        var cache = try ExpertCache.init(gpa, &pool, &reader, .{ .num_layers = 2, .num_experts = 4 });
        defer cache.deinit(gpa);

        for (0..2) |l| {
            for (0..4) |e| {
                try cache.registerCold(@intCast(l), @intCast(e), try pack.tripleFor(@intCast(l), @intCast(e)));
            }
        }

        // Prefetch all 8 experts ahead of verification pass.
        for (0..2) |l| {
            for (0..4) |e| {
                try std.testing.expectEqual(PrefetchOutcome.submitted, try cache.prefetch(@intCast(l), @intCast(e)));
            }
        }
        try cache.drainInflight();

        // Acquire all 8 experts in one verification pass: 100% prefetch hits, 0 sync misses.
        for (0..2) |l| {
            for (0..4) |e| {
                const pinned = try cache.acquirePinned(@intCast(l), @intCast(e));
                try pack.expectSlotMatches(pinned);
            }
        }
        const stats = cache.telemetry();
        try std.testing.expectEqual(@as(u64, 8), stats.prefetch_hits);
        try std.testing.expectEqual(@as(u64, 0), stats.sync_misses);
        try std.testing.expectEqual(@as(u64, 0), stats.cache_hits);
        try std.testing.expectEqual(@as(u64, 8 * 3 * blob_bytes), stats.bytes_read);
        try std.testing.expectEqual(@as(f64, 1.0), stats.prefetchHitRate());
        try std.testing.expectEqual(@as(f64, 1.0), stats.coldHitRate());

        // Second pass over the same resident experts records pure `cache_hits`.
        cache.releasePins();
        for (0..2) |l| {
            for (0..4) |e| {
                const pinned = try cache.acquirePinned(@intCast(l), @intCast(e));
                try pack.expectSlotMatches(pinned);
            }
        }
        cache.releasePins();
        try std.testing.expectEqual(@as(u64, 8), cache.telemetry().cache_hits);
        try std.testing.expectEqual(@as(u64, 0), cache.telemetry().sync_misses);
    }
}
