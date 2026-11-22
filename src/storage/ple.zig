//! C10 NVMe `F_NOCACHE` Per-Layer Embedding / Engram Store (ADR-0015, ADR-0019).
//!
//! Streams cold `.amf` `TensorKind.sparse_table` row tables (`f16`, `bf16`,
//! `q4_affine`, `q3_affine`, `q2_affine`) across single-file or multi-shard
//! `.amf` containers opened with `fcntl(fd, F_NOCACHE, 1)`:
//!
//! - Deterministic bigram/trigram token-tuple hashing (`hashPleNgram`) mapped
//!   to `(shard, local_row)` across arbitrarily sharded layer tables.
//! - 16 KiB page-aligned `WiredArena` LRU page cache (`PlePageCache`) with
//!   O(1) hash lookup, O(1) intrusive LRU eviction, and async `KqueueAioEngine`
//!   / `EcorePreadPool` batch prefetch (`prefetchBatch`).
//! - Zero-allocation batched lookup + FP32 accumulation (`lookupAndAccumulate`)
//!   across `M = 1..64` tokens or speculative tree nodes.
//! - Mandatory `ADR-0019` `r = 0` Single-Lookup Loop Isolation (`LoopPleScratch`
//!   + `lookupForLoopStep`): fetches rows once on `loop_step == 0` into wired
//!   RAM and serves `loop_step >= 1` strictly from `LoopPleScratch` with `0`
//!   extra NVMe reads (`nvme_reads_delta == 0`) and `0` heap allocations.

const std = @import("std");
const page = @import("../mem/page.zig");
const arena_mod = @import("../mem/arena.zig");
const budget_mod = @import("../mem/budget.zig");
const sysinfo = @import("../platform/sysinfo.zig");
const qos = @import("../platform/qos.zig");
const quant = @import("../quant/spec.zig");
const amf = @import("../format/amf.zig");
const aio = @import("aio.zig");

pub const MAX_BATCH_TOKENS: usize = 64;
pub const MAX_LAYERS: usize = 512;
pub const DEFAULT_CACHE_PAGES: usize = 128; // 2 MiB wired LRU page cache
pub const DEFAULT_MAX_INFLIGHT: usize = 16;
pub const DEFAULT_TIMEOUT_MS: u64 = 5_000;

/// Multi-head Engram conditional memory bounds (M7, ADR-0015).
pub const ENGRAM_MIN_HEADS: u8 = 2;
pub const ENGRAM_MAX_HEADS: u8 = 8;
pub const ENGRAM_MIN_ORDER: u8 = 2;
pub const ENGRAM_MAX_ORDER: u8 = 5;

/// Sparse external memory table semantics stored in `.amf` `TensorKind.sparse_table`.
///
/// - `ple_ngram`: M6 Per-Layer Embedding (single `[embed_dim]` vector per row).
/// - `engram_conditional`: M7 196B Engram conditional memory (each row packs
///   contiguous `[key_vec[0..D], value_vec[0..D]]` of total width `2 * D` so one
///   `F_NOCACHE` page fetch retrieves both key and value for context gating).
pub const MemoryTableKind = enum(u8) {
    ple_ngram = 0,
    engram_conditional = 1,
};

const Q4 = quant.QuantOps(.{ .bits = .q4, .group_size = 64 });
const Q3 = quant.QuantOps(.{ .bits = .q3, .group_size = 64 });
const Q2 = quant.QuantOps(.{ .bits = .q2, .group_size = 64 });

const EMPTY_HASH_SLOT: u32 = std.math.maxInt(u32);
const INVALID_LRU_NODE: u32 = std.math.maxInt(u32);

pub const Error = amf.ParseError || amf.IoError || aio.EcorePoolError ||
    arena_mod.WiredArena.InitError || arena_mod.WiredArena.AllocError ||
    sysinfo.Error || error{
    NoFilesProvided,
    NoSparseTablesFound,
    InvalidCacheConfig,
    UnsupportedDType,
    InconsistentEmbedDim,
    LayerOutOfRange,
    LayerNotFound,
    RowOutOfBounds,
    InvalidNgramOrder,
    InvalidEngramHeads,
    InvalidEngramOrder,
    InvalidEngramLayout,
    TableKindMismatch,
    InvalidBatchSize,
    BufferTooSmall,
    LoopScratchMismatch,
    LoopScratchUnpopulated,
    Overflow,
};

/// Deterministic FNV-1a + SplitMix64 avalanche hash mapping a token tuple of
/// length `order` (`1` = unigram, `2` = bigram, `3` = trigram, ...) to a global
/// row index in `[0, total_rows)`.
///
/// Consumes the trailing `min(tokens.len, order)` tokens when `tokens.len >= order`,
/// and left-pads with deterministic BOS sentinels when `tokens.len < order`.
pub fn hashPleNgram(tokens: []const u32, order: u8, total_rows: u64) u64 {
    std.debug.assert(order >= 1);
    std.debug.assert(total_rows > 0);
    // Domain-separate by order so unigram/bigram/trigram hashes remain independent.
    var h: u64 = 0xcbf29ce484222325 ^ (@as(u64, order) *% 0x9e3779b97f4a7c15);
    const ord: usize = order;
    const start: usize = if (tokens.len >= ord) tokens.len - ord else 0;
    const pad: usize = ord - (tokens.len - start);
    for (0..pad) |p| {
        h ^= 0xFFFF_FFFF_0000_0000 | @as(u64, @intCast(p));
        h *%= 0x100000001b3;
    }
    for (tokens[start..]) |tok| {
        h ^= @as(u64, tok);
        h *%= 0x100000001b3;
    }
    // SplitMix64 finalizer for uniform distribution modulo arbitrary `total_rows`.
    h ^= h >> 30;
    h *%= 0xbf58476d1ce4e5b9;
    h ^= h >> 27;
    h *%= 0x94d049bb133111eb;
    h ^= h >> 31;
    return h % total_rows;
}

/// Resolved `(shard_index, local_row)` target for a global PLE row index.
pub const ShardRowLocation = struct {
    shard_index: u32,
    local_row: u64,
    global_row: u64,
};

/// Deterministically maps a token tuple (`order >= 1`) across sharded row counts
/// into `(shard_index, local_row, global_row)`.
pub fn mapNgramToShardRow(
    tokens: []const u32,
    order: u8,
    shard_row_counts: []const u64,
) error{ InvalidNgramOrder, RowOutOfBounds, Overflow }!ShardRowLocation {
    if (order == 0) return error.InvalidNgramOrder;
    if (shard_row_counts.len == 0) return error.RowOutOfBounds;

    var total_rows: u64 = 0;
    for (shard_row_counts) |cnt| {
        if (cnt == 0) return error.RowOutOfBounds;
        total_rows = std.math.add(u64, total_rows, cnt) catch return error.Overflow;
    }
    const global_row = hashPleNgram(tokens, order, total_rows);
    var prefix: u64 = 0;
    for (shard_row_counts, 0..) |cnt, s| {
        if (global_row < prefix + cnt) {
            return .{
                .shard_index = @intCast(s),
                .local_row = global_row - prefix,
                .global_row = global_row,
            };
        }
        prefix += cnt;
    }
    return error.RowOutOfBounds;
}

/// Multi-head deterministic hash for M7 196B Engram conditional memory (`C10`).
///
/// Maps the trailing `ngram_order` (`n in 2..5`, left-padded with BOS sentinels
/// when `tokens.len < ngram_order`) and `head_idx` (`0..H-1`, `H in 2..8`) to a
/// global row index in `[0, total_rows)` across sharded 16 KiB-aligned `.amf`
/// `TensorKind.sparse_table` tables.
pub fn hashEngramHead(tokens: []const u32, head_idx: u8, ngram_order: u8, total_rows: u64) u64 {
    std.debug.assert(ngram_order >= 1);
    std.debug.assert(total_rows > 0);
    // Domain-separate Engram from PLE and separate every (head_idx, ngram_order) pair.
    const head_mix: u64 = (@as(u64, head_idx) +% 1) *% 0x517cc1b727220a95;
    const order_mix: u64 = @as(u64, ngram_order) *% 0x9e3779b97f4a7c15;
    var h: u64 = 0x84222325cbf29ce4 ^ head_mix ^ std.math.rotl(u64, order_mix, 17);
    const ord: usize = ngram_order;
    const start: usize = if (tokens.len >= ord) tokens.len - ord else 0;
    const pad: usize = ord - (tokens.len - start);
    for (0..pad) |p| {
        h ^= 0xFFFF_FFFE_0000_0000 | (@as(u64, head_idx) << 16) | @as(u64, @intCast(p));
        h *%= 0x100000001b3;
    }
    for (tokens[start..]) |tok| {
        h ^= @as(u64, tok) ^ head_mix;
        h *%= 0x100000001b3;
    }
    // SplitMix64 avalanche finalizer for uniform distribution modulo `total_rows`.
    h ^= h >> 30;
    h *%= 0xbf58476d1ce4e5b9;
    h ^= h >> 27;
    h *%= 0x94d049bb133111eb;
    h ^= h >> 31;
    return h % total_rows;
}

/// Deterministically maps an Engram `(head_idx, ngram_order)` suffix lookup
/// across sharded row counts into `(shard_index, local_row, global_row)`.
pub fn mapEngramHeadToShardRow(
    tokens: []const u32,
    head_idx: u8,
    ngram_order: u8,
    shard_row_counts: []const u64,
) error{ InvalidEngramHeads, InvalidEngramOrder, RowOutOfBounds, Overflow }!ShardRowLocation {
    if (head_idx >= ENGRAM_MAX_HEADS) return error.InvalidEngramHeads;
    if (ngram_order < ENGRAM_MIN_ORDER or ngram_order > ENGRAM_MAX_ORDER) {
        return error.InvalidEngramOrder;
    }
    if (shard_row_counts.len == 0) return error.RowOutOfBounds;

    var total_rows: u64 = 0;
    for (shard_row_counts) |cnt| {
        if (cnt == 0) return error.RowOutOfBounds;
        total_rows = std.math.add(u64, total_rows, cnt) catch return error.Overflow;
    }
    const global_row = hashEngramHead(tokens, head_idx, ngram_order, total_rows);
    var prefix: u64 = 0;
    for (shard_row_counts, 0..) |cnt, s| {
        if (global_row < prefix + cnt) {
            return .{
                .shard_index = @intCast(s),
                .local_row = global_row - prefix,
                .global_row = global_row,
            };
        }
        prefix += cnt;
    }
    return error.RowOutOfBounds;
}

inline fn stableSigmoidF32(x: f32) f32 {
    const clamped = std.math.clamp(x, -80.0, 80.0);
    return 1.0 / (1.0 + @exp(-clamped));
}

/// Computes context-gated Engram conditional memory injection in FP16 with
/// FP32 inner-product and multi-head accumulation (zero heap allocations):
///   `out[d] = sum_{h=0..H-1} sigmoid(dot(hidden_norm, k_h) * inv_sqrt_d) * v_h[d]`
///
/// `kv_heads_f32` holds `num_heads` contiguous rows of width `2 * dim`, where
/// head `h` occupies `[k_h[0..dim], v_h[0..dim]]`.
pub fn applyEngramContextGateF16(
    hidden_norm: []const f16,
    kv_heads_f32: []const f32,
    num_heads: u8,
    dim: u32,
    out: []f16,
) Error!void {
    if (num_heads < ENGRAM_MIN_HEADS or num_heads > ENGRAM_MAX_HEADS) {
        return error.InvalidEngramHeads;
    }
    if (dim == 0) return error.InvalidEngramLayout;
    const d_usize: usize = dim;
    const row_stride = std.math.mul(usize, d_usize, 2) catch return error.Overflow;
    const kv_need = std.math.mul(usize, @as(usize, num_heads), row_stride) catch return error.Overflow;
    if (hidden_norm.len < d_usize or out.len < d_usize or kv_heads_f32.len < kv_need) {
        return error.BufferTooSmall;
    }

    const inv_sqrt_d: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(dim)));
    var gates: [ENGRAM_MAX_HEADS]f32 = @splat(0.0);
    const heads: usize = num_heads;

    for (0..heads) |h| {
        const k_h = kv_heads_f32[h * row_stride ..][0..d_usize];
        var dot: f32 = 0.0;
        for (0..d_usize) |d| {
            dot += @as(f32, @floatCast(hidden_norm[d])) * k_h[d];
        }
        gates[h] = stableSigmoidF32(dot * inv_sqrt_d);
    }

    for (0..d_usize) |d| {
        var acc: f32 = 0.0;
        for (0..heads) |h| {
            const v_h = kv_heads_f32[h * row_stride + d_usize ..][0..d_usize];
            acc += gates[h] * v_h[d];
        }
        out[d] = @floatCast(acc);
    }
}

/// FP32 counterpart of `applyEngramContextGateF16`:
///   `out[d] = sum_{h=0..H-1} sigmoid(dot(hidden_norm, k_h) * inv_sqrt_d) * v_h[d]`
pub fn applyEngramContextGateF32(
    hidden_norm: []const f32,
    kv_heads_f32: []const f32,
    num_heads: u8,
    dim: u32,
    out: []f32,
) Error!void {
    if (num_heads < ENGRAM_MIN_HEADS or num_heads > ENGRAM_MAX_HEADS) {
        return error.InvalidEngramHeads;
    }
    if (dim == 0) return error.InvalidEngramLayout;
    const d_usize: usize = dim;
    const row_stride = std.math.mul(usize, d_usize, 2) catch return error.Overflow;
    const kv_need = std.math.mul(usize, @as(usize, num_heads), row_stride) catch return error.Overflow;
    if (hidden_norm.len < d_usize or out.len < d_usize or kv_heads_f32.len < kv_need) {
        return error.BufferTooSmall;
    }

    const inv_sqrt_d: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(dim)));
    var gates: [ENGRAM_MAX_HEADS]f32 = @splat(0.0);
    const heads: usize = num_heads;

    for (0..heads) |h| {
        const k_h = kv_heads_f32[h * row_stride ..][0..d_usize];
        var dot: f32 = 0.0;
        for (0..d_usize) |d| {
            dot += hidden_norm[d] * k_h[d];
        }
        gates[h] = stableSigmoidF32(dot * inv_sqrt_d);
    }

    for (0..d_usize) |d| {
        var acc: f32 = 0.0;
        for (0..heads) |h| {
            const v_h = kv_heads_f32[h * row_stride + d_usize ..][0..d_usize];
            acc += gates[h] * v_h[d];
        }
        out[d] = acc;
    }
}

/// Unique identifier for one 16 KiB page inside an opened `.amf` shard file.
pub const PageKey = struct {
    file_idx: u16,
    page_idx: u64,

    pub inline fn eql(a: PageKey, b: PageKey) bool {
        return a.file_idx == b.file_idx and a.page_idx == b.page_idx;
    }
};

fn hashPageKey(key: PageKey) u64 {
    var h: u64 = key.page_idx ^ (@as(u64, key.file_idx) *% 0x9e3779b97f4a7c15);
    h ^= h >> 30;
    h *%= 0xbf58476d1ce4e5b9;
    h ^= h >> 27;
    h *%= 0x94d049bb133111eb;
    h ^= h >> 31;
    return h;
}

/// Runtime telemetry for `PlePageCache` and `PleStore`.
pub const Stats = struct {
    cache_hits: u64 = 0,
    cache_misses: u64 = 0,
    prefetch_submitted: u64 = 0,
    prefetch_hits: u64 = 0,
    nvme_reads: u64 = 0,
    bytes_read: u64 = 0,
    evictions: u64 = 0,
    rows_decoded: u64 = 0,
    loop_scratch_fills: u64 = 0,
    loop_scratch_hits: u64 = 0,
    engram_lookups: u64 = 0,
};

pub const PageCacheConfig = struct {
    cache_pages: usize = DEFAULT_CACHE_PAGES,
    max_inflight: usize = DEFAULT_MAX_INFLIGHT,
    io_backend: aio.IoBackend = .auto,
    worker_count: usize = 2,
    worker_qos: qos.QosClass = .utility,
    io_policy: qos.DiskIoPolicy = .important,
    timeout_ms: u64 = DEFAULT_TIMEOUT_MS,
    budget_override: ?budget_mod.Budget = null,
};

/// 16 KiB page-aligned `WiredArena` LRU page cache backed by `KqueueAioEngine`
/// / `EcorePreadPool` (`AsyncExpertReader`).
///
/// All wired pages, open-addressing hash buckets, LRU slot metadata, and AIO
/// slots are preallocated once in `init()`. Steady-state `prefetchPage`,
/// `reapPrefetches`, and `fetchPage` perform zero dynamic heap allocations.
pub const PlePageCache = struct {
    const SlotState = enum(u8) {
        empty = 0,
        inflight = 1,
        ready = 2,
    };

    const CacheSlot = struct {
        state: SlotState = .empty,
        key: PageKey = .{ .file_idx = 0, .page_idx = 0 },
        prev: u32 = INVALID_LRU_NODE,
        next: u32 = INVALID_LRU_NODE,
        pin_count: u16 = 0,
    };

    arena: arena_mod.WiredArena,
    slots: []CacheSlot,
    hash_table: []u32,
    hash_mask: usize,
    free_stack: []u32,
    free_top: usize,
    lru_head: u32,
    lru_tail: u32,
    async_reader: aio.AsyncExpertReader,
    timeout_ms: u64,
    stats: Stats,

    pub fn init(allocator: std.mem.Allocator, config: PageCacheConfig) Error!PlePageCache {
        if (config.cache_pages == 0 or config.cache_pages > 16384) return error.InvalidCacheConfig;
        if (config.max_inflight == 0 or config.max_inflight > aio.MAX_INFLIGHT) return error.InvalidCacheConfig;

        const arena_bytes = std.math.mul(usize, config.cache_pages, page.APPLE_PAGE_SIZE) catch return error.Overflow;
        const wired_budget = config.budget_override orelse try defaultWiredBudget();
        try wired_budget.checkPlan(arena_bytes);

        var mem_arena = try arena_mod.WiredArena.init(arena_bytes, wired_budget);
        errdefer mem_arena.deinit();

        const slots = try allocator.alloc(CacheSlot, config.cache_pages);
        errdefer allocator.free(slots);
        @memset(slots, .{});

        const min_hash_cap = std.math.mul(usize, config.cache_pages, 2) catch return error.Overflow;
        const hash_cap = std.math.ceilPowerOfTwo(usize, @max(@as(usize, 8), min_hash_cap)) catch return error.Overflow;
        const hash_table = try allocator.alloc(u32, hash_cap);
        errdefer allocator.free(hash_table);
        @memset(hash_table, EMPTY_HASH_SLOT);

        const free_stack = try allocator.alloc(u32, config.cache_pages);
        errdefer allocator.free(free_stack);
        for (0..config.cache_pages) |i| {
            // Pop slot 0 first.
            free_stack[i] = @intCast(config.cache_pages - 1 - i);
        }

        const eff_inflight: usize = if (config.cache_pages >= 2)
            @min(config.max_inflight, config.cache_pages / 2)
        else
            1;

        var reader = try aio.AsyncExpertReader.init(allocator, .{
            .backend = config.io_backend,
            .max_inflight = eff_inflight,
            .worker_count = @max(@as(usize, 1), config.worker_count),
            .worker_qos = config.worker_qos,
            .io_policy = config.io_policy,
        });
        errdefer reader.deinit(allocator);

        return .{
            .arena = mem_arena,
            .slots = slots,
            .hash_table = hash_table,
            .hash_mask = hash_cap - 1,
            .free_stack = free_stack,
            .free_top = config.cache_pages,
            .lru_head = INVALID_LRU_NODE,
            .lru_tail = INVALID_LRU_NODE,
            .async_reader = reader,
            .timeout_ms = config.timeout_ms,
            .stats = .{},
        };
    }

    pub fn deinit(self: *PlePageCache, allocator: std.mem.Allocator) void {
        // Drain all in-flight AIO / E-core pread operations BEFORE freeing the wired arena!
        self.async_reader.deinit(allocator);
        allocator.free(self.free_stack);
        self.free_stack = &.{};
        allocator.free(self.hash_table);
        self.hash_table = &.{};
        allocator.free(self.slots);
        self.slots = &.{};
        self.arena.deinit();
    }

    pub fn totalWiredBytes(self: *const PlePageCache) u64 {
        return self.arena.bytes().len;
    }

    pub fn pageCount(self: *const PlePageCache) usize {
        return self.slots.len;
    }

    pub fn activeBackend(self: *const PlePageCache) aio.IoBackend {
        return self.async_reader.activeBackend();
    }

    pub inline fn pageSliceMut(self: *PlePageCache, slot_idx: usize) []align(page.APPLE_PAGE_SIZE) u8 {
        const start = slot_idx * page.APPLE_PAGE_SIZE;
        return @alignCast(self.arena.bytes()[start .. start + page.APPLE_PAGE_SIZE]);
    }

    pub inline fn pageSliceConst(self: *const PlePageCache, slot_idx: usize) []align(page.APPLE_PAGE_SIZE) const u8 {
        const start = slot_idx * page.APPLE_PAGE_SIZE;
        return @alignCast(self.arena.bytes()[start .. start + page.APPLE_PAGE_SIZE]);
    }

    fn hashFind(self: *const PlePageCache, key: PageKey) ?usize {
        var b: usize = @intCast(hashPageKey(key) & self.hash_mask);
        while (true) {
            const s = self.hash_table[b];
            if (s == EMPTY_HASH_SLOT) return null;
            if (self.slots[s].key.eql(key)) return s;
            b = (b + 1) & self.hash_mask;
        }
    }

    fn hashInsert(self: *PlePageCache, key: PageKey, slot_idx: u32) void {
        var b: usize = @intCast(hashPageKey(key) & self.hash_mask);
        while (self.hash_table[b] != EMPTY_HASH_SLOT) {
            std.debug.assert(self.hash_table[b] != slot_idx);
            b = (b + 1) & self.hash_mask;
        }
        self.hash_table[b] = slot_idx;
    }

    fn hashRemove(self: *PlePageCache, key: PageKey, slot_idx: u32) void {
        var hole: usize = @intCast(hashPageKey(key) & self.hash_mask);
        while (self.hash_table[hole] != slot_idx) {
            std.debug.assert(self.hash_table[hole] != EMPTY_HASH_SLOT);
            hole = (hole + 1) & self.hash_mask;
        }
        var cur = (hole + 1) & self.hash_mask;
        while (self.hash_table[cur] != EMPTY_HASH_SLOT) {
            const elem = self.hash_table[cur];
            const home: usize = @intCast(hashPageKey(self.slots[elem].key) & self.hash_mask);
            const d_cur = (cur -% home) & self.hash_mask;
            const d_hole = (hole -% home) & self.hash_mask;
            if (d_hole < d_cur) {
                self.hash_table[hole] = elem;
                hole = cur;
            }
            cur = (cur + 1) & self.hash_mask;
        }
        self.hash_table[hole] = EMPTY_HASH_SLOT;
    }

    fn lruPushFront(self: *PlePageCache, slot_idx: u32) void {
        const node = &self.slots[slot_idx];
        node.prev = INVALID_LRU_NODE;
        node.next = self.lru_head;
        if (self.lru_head != INVALID_LRU_NODE) {
            self.slots[self.lru_head].prev = slot_idx;
        } else {
            self.lru_tail = slot_idx;
        }
        self.lru_head = slot_idx;
    }

    fn lruRemove(self: *PlePageCache, slot_idx: u32) void {
        const node = &self.slots[slot_idx];
        if (node.prev != INVALID_LRU_NODE) {
            self.slots[node.prev].next = node.next;
        } else {
            self.lru_head = node.next;
        }
        if (node.next != INVALID_LRU_NODE) {
            self.slots[node.next].prev = node.prev;
        } else {
            self.lru_tail = node.prev;
        }
        node.prev = INVALID_LRU_NODE;
        node.next = INVALID_LRU_NODE;
    }

    fn lruMoveToFront(self: *PlePageCache, slot_idx: u32) void {
        if (self.lru_head == slot_idx) return;
        self.lruRemove(slot_idx);
        self.lruPushFront(slot_idx);
    }

    fn acquireVictimSlot(self: *PlePageCache) ?u32 {
        if (self.free_top > 0) {
            self.free_top -= 1;
            return self.free_stack[self.free_top];
        }
        var cur = self.lru_tail;
        while (cur != INVALID_LRU_NODE) {
            const slot = &self.slots[cur];
            if (slot.state == .ready and slot.pin_count == 0) {
                self.lruRemove(cur);
                self.hashRemove(slot.key, cur);
                slot.state = .empty;
                self.stats.evictions += 1;
                return cur;
            }
            cur = slot.prev;
        }
        return null;
    }

    fn releaseEmptySlot(self: *PlePageCache, slot_idx: u32) void {
        std.debug.assert(self.slots[slot_idx].state == .empty);
        std.debug.assert(self.free_top < self.free_stack.len);
        self.free_stack[self.free_top] = slot_idx;
        self.free_top += 1;
    }

    fn applyCompletion(self: *PlePageCache, comp: aio.Completion) aio.EngineError!void {
        const slot_idx: u32 = @intCast(comp.user_data);
        std.debug.assert(slot_idx < self.slots.len);
        const slot = &self.slots[slot_idx];
        std.debug.assert(slot.state == .inflight);

        const bytes_read = comp.check() catch |err| {
            self.hashRemove(slot.key, slot_idx);
            slot.state = .empty;
            self.releaseEmptySlot(slot_idx);
            return err;
        };
        if (bytes_read != page.APPLE_PAGE_SIZE) {
            self.hashRemove(slot.key, slot_idx);
            slot.state = .empty;
            self.releaseEmptySlot(slot_idx);
            return error.ShortRead;
        }
        slot.state = .ready;
        self.lruPushFront(slot_idx);
        self.stats.nvme_reads += 1;
        self.stats.bytes_read += bytes_read;
    }

    /// Reaps any already-completed asynchronous prefetches without blocking.
    pub fn reapPrefetches(self: *PlePageCache) aio.EngineError!usize {
        if (self.async_reader.inflightCount() == 0) return 0;
        var comps: [aio.MAX_INFLIGHT]aio.Completion = undefined;
        var total: usize = 0;
        while (self.async_reader.inflightCount() > 0) {
            const got = try self.async_reader.poll(&comps);
            if (got == 0) break;
            for (comps[0..got]) |c| {
                try self.applyCompletion(c);
            }
            total += got;
        }
        return total;
    }

    pub fn drainPrefetches(self: *PlePageCache) aio.EngineError!void {
        var comps: [aio.MAX_INFLIGHT]aio.Completion = undefined;
        while (self.async_reader.inflightCount() > 0) {
            const got = try self.async_reader.wait(&comps, self.timeout_ms);
            if (got == 0) return error.Timeout;
            for (comps[0..got]) |c| {
                try self.applyCompletion(c);
            }
        }
    }

    /// Clears all cached pages after draining any in-flight prefetches.
    pub fn clear(self: *PlePageCache) aio.EngineError!void {
        try self.drainPrefetches();
        @memset(self.slots, .{});
        @memset(self.hash_table, EMPTY_HASH_SLOT);
        for (0..self.slots.len) |i| {
            self.free_stack[i] = @intCast(self.slots.len - 1 - i);
        }
        self.free_top = self.slots.len;
        self.lru_head = INVALID_LRU_NODE;
        self.lru_tail = INVALID_LRU_NODE;
    }

    /// Submits a non-blocking 16 KiB direct-I/O page prefetch if `key` is not
    /// already cached or in flight. Returns `true` iff a new NVMe read was queued.
    pub fn prefetchPage(self: *PlePageCache, fd: std.c.fd_t, key: PageKey) aio.EngineError!bool {
        _ = try self.reapPrefetches();
        if (self.hashFind(key)) |existing| {
            if (self.slots[existing].state == .ready) {
                self.lruMoveToFront(@intCast(existing));
            }
            return false;
        }
        if (self.async_reader.availableSlots() == 0) return false;
        const victim = self.acquireVictimSlot() orelse return false;

        self.slots[victim] = .{
            .state = .inflight,
            .key = key,
            .prev = INVALID_LRU_NODE,
            .next = INVALID_LRU_NODE,
            .pin_count = 0,
        };
        self.hashInsert(key, victim);

        const file_off: u64 = key.page_idx * page.APPLE_PAGE_SIZE;
        self.async_reader.submit(.{
            .fd = fd,
            .offset = file_off,
            .dst = self.pageSliceMut(victim),
            .user_data = victim,
        }) catch |err| {
            self.hashRemove(key, victim);
            self.slots[victim].state = .empty;
            self.releaseEmptySlot(victim);
            if (err == error.QueueFull) return false;
            return err;
        };
        self.stats.prefetch_submitted += 1;
        return true;
    }

    fn waitForInflightSlot(self: *PlePageCache, target_slot: usize) aio.EngineError!void {
        var comps: [aio.MAX_INFLIGHT]aio.Completion = undefined;
        while (self.slots[target_slot].state == .inflight) {
            const got = try self.async_reader.wait(&comps, self.timeout_ms);
            if (got == 0) return error.Timeout;
            for (comps[0..got]) |c| {
                try self.applyCompletion(c);
            }
        }
    }

    /// Returns the ready cache slot index holding `key`, completing an in-flight
    /// prefetch or issuing a direct `F_NOCACHE` `pread` on miss.
    pub fn fetchPage(
        self: *PlePageCache,
        fd: std.c.fd_t,
        key: PageKey,
    ) aio.EngineError!struct { slot_idx: usize, cache_hit: bool } {
        if (self.hashFind(key)) |idx| {
            if (self.slots[idx].state == .inflight) {
                try self.waitForInflightSlot(idx);
                if (self.slots[idx].state == .ready and self.slots[idx].key.eql(key)) {
                    self.lruMoveToFront(@intCast(idx));
                    self.stats.prefetch_hits += 1;
                    self.stats.cache_hits += 1;
                    return .{ .slot_idx = idx, .cache_hit = true };
                }
            } else if (self.slots[idx].state == .ready) {
                self.lruMoveToFront(@intCast(idx));
                self.stats.cache_hits += 1;
                return .{ .slot_idx = idx, .cache_hit = true };
            }
        }

        var victim_opt = self.acquireVictimSlot();
        while (victim_opt == null) {
            if (self.async_reader.inflightCount() == 0) return error.QueueFull;
            var comps: [aio.MAX_INFLIGHT]aio.Completion = undefined;
            const got = try self.async_reader.wait(&comps, self.timeout_ms);
            if (got == 0) return error.Timeout;
            for (comps[0..got]) |c| {
                try self.applyCompletion(c);
            }
            victim_opt = self.acquireVictimSlot();
        }
        const victim = victim_opt.?;
        const file_off: u64 = key.page_idx * page.APPLE_PAGE_SIZE;
        preadExact(fd, self.pageSliceMut(victim), file_off) catch |err| {
            self.releaseEmptySlot(victim);
            return err;
        };

        self.slots[victim] = .{
            .state = .ready,
            .key = key,
            .prev = INVALID_LRU_NODE,
            .next = INVALID_LRU_NODE,
            .pin_count = 0,
        };
        self.hashInsert(key, victim);
        self.lruPushFront(victim);
        self.stats.cache_misses += 1;
        self.stats.nvme_reads += 1;
        self.stats.bytes_read += page.APPLE_PAGE_SIZE;
        return .{ .slot_idx = victim, .cache_hit = false };
    }
};

/// Preallocated 16 KiB-aligned wired RAM scratch holding `r = 0` dequantized
/// PLE (`[max_layers, max_tokens, embed_dim]f32`) or multi-head Engram
/// (`[max_layers, max_tokens, max_heads, 2 * value_dim]f32`) rows for mandatory
/// `ADR-0019` `r = 0` single-lookup loop isolation (`nvme_reads_delta == 0` for
/// `r >= 1`).
pub const LoopPleScratch = struct {
    arena: arena_mod.WiredArena,
    data_f32: []f32,
    valid_tokens: []u16,
    valid_heads: []u8,
    max_layers: u16,
    max_tokens: u16,
    max_heads: u8,
    embed_dim: u32,
    table_kind: MemoryTableKind,
    loop_reuse_hits: u64 = 0,

    pub fn init(
        allocator: std.mem.Allocator,
        max_layers: u16,
        max_tokens: u16,
        embed_dim: u32,
        budget_override: ?budget_mod.Budget,
    ) Error!LoopPleScratch {
        return initInternal(allocator, max_layers, max_tokens, 1, embed_dim, .ple_ngram, budget_override);
    }

    /// Allocates wired scratch for multi-head Engram `(key_vec, value_vec)` rows
    /// of dimension `value_dim` (`embed_dim == 2 * value_dim`, `max_heads in 2..8`).
    pub fn initEngram(
        allocator: std.mem.Allocator,
        max_layers: u16,
        max_tokens: u16,
        max_heads: u8,
        value_dim: u32,
        budget_override: ?budget_mod.Budget,
    ) Error!LoopPleScratch {
        if (max_heads < ENGRAM_MIN_HEADS or max_heads > ENGRAM_MAX_HEADS) {
            return error.InvalidEngramHeads;
        }
        if (value_dim == 0) return error.InvalidEngramLayout;
        const packed_dim = std.math.mul(u32, value_dim, 2) catch return error.Overflow;
        return initInternal(allocator, max_layers, max_tokens, max_heads, packed_dim, .engram_conditional, budget_override);
    }

    fn initInternal(
        allocator: std.mem.Allocator,
        max_layers: u16,
        max_tokens: u16,
        max_heads: u8,
        embed_dim: u32,
        table_kind: MemoryTableKind,
        budget_override: ?budget_mod.Budget,
    ) Error!LoopPleScratch {
        if (max_layers == 0 or max_tokens == 0 or max_tokens > MAX_BATCH_TOKENS or max_heads == 0 or embed_dim == 0) {
            return error.InvalidBatchSize;
        }
        const lt = std.math.mul(usize, max_layers, max_tokens) catch return error.Overflow;
        const lth = std.math.mul(usize, lt, max_heads) catch return error.Overflow;
        const elems = std.math.mul(usize, lth, embed_dim) catch return error.Overflow;
        const raw_bytes = std.math.mul(usize, elems, @sizeOf(f32)) catch return error.Overflow;
        const aligned_bytes = page.alignUp(raw_bytes) catch return error.Overflow;

        const wired_budget = budget_override orelse try defaultWiredBudget();
        try wired_budget.checkPlan(aligned_bytes);

        var mem_arena = try arena_mod.WiredArena.init(aligned_bytes, wired_budget);
        errdefer mem_arena.deinit();

        const buf = try mem_arena.allocPages(raw_bytes);
        const data_f32: []f32 = @as([]f32, @alignCast(std.mem.bytesAsSlice(f32, buf)))[0..elems];
        @memset(data_f32, 0.0);

        const valid_tokens = try allocator.alloc(u16, max_layers);
        errdefer allocator.free(valid_tokens);
        @memset(valid_tokens, 0);

        const valid_heads = try allocator.alloc(u8, max_layers);
        errdefer allocator.free(valid_heads);
        @memset(valid_heads, 0);

        return .{
            .arena = mem_arena,
            .data_f32 = data_f32,
            .valid_tokens = valid_tokens,
            .valid_heads = valid_heads,
            .max_layers = max_layers,
            .max_tokens = max_tokens,
            .max_heads = max_heads,
            .embed_dim = embed_dim,
            .table_kind = table_kind,
            .loop_reuse_hits = 0,
        };
    }

    pub fn deinit(self: *LoopPleScratch, allocator: std.mem.Allocator) void {
        allocator.free(self.valid_heads);
        allocator.free(self.valid_tokens);
        self.valid_heads = &.{};
        self.valid_tokens = &.{};
        self.data_f32 = &.{};
        self.arena.deinit();
    }

    /// Resets per-layer valid-token/head markers at the beginning of a new decode step.
    pub fn resetStep(self: *LoopPleScratch) void {
        @memset(self.valid_tokens, 0);
        @memset(self.valid_heads, 0);
    }

    pub fn totalWiredBytes(self: *const LoopPleScratch) u64 {
        return self.arena.bytes().len;
    }

    pub inline fn engramValueDim(self: *const LoopPleScratch) u32 {
        return self.embed_dim / 2;
    }

    pub inline fn rowSliceMut(self: *LoopPleScratch, layer: u16, token_idx: usize) []f32 {
        const stride_token = @as(usize, self.max_heads) * @as(usize, self.embed_dim);
        const stride_layer = @as(usize, self.max_tokens) * stride_token;
        const start = @as(usize, layer) * stride_layer + token_idx * stride_token;
        return self.data_f32[start .. start + self.embed_dim];
    }

    pub inline fn rowSliceConst(self: *const LoopPleScratch, layer: u16, token_idx: usize) []const f32 {
        const stride_token = @as(usize, self.max_heads) * @as(usize, self.embed_dim);
        const stride_layer = @as(usize, self.max_tokens) * stride_token;
        const start = @as(usize, layer) * stride_layer + token_idx * stride_token;
        return self.data_f32[start .. start + self.embed_dim];
    }

    pub inline fn engramRowSliceMut(self: *LoopPleScratch, layer: u16, token_idx: usize, head_idx: usize) []f32 {
        const stride_token = @as(usize, self.max_heads) * @as(usize, self.embed_dim);
        const stride_layer = @as(usize, self.max_tokens) * stride_token;
        const start = @as(usize, layer) * stride_layer + token_idx * stride_token + head_idx * @as(usize, self.embed_dim);
        return self.data_f32[start .. start + self.embed_dim];
    }

    pub inline fn engramRowSliceConst(self: *const LoopPleScratch, layer: u16, token_idx: usize, head_idx: usize) []const f32 {
        const stride_token = @as(usize, self.max_heads) * @as(usize, self.embed_dim);
        const stride_layer = @as(usize, self.max_tokens) * stride_token;
        const start = @as(usize, layer) * stride_layer + token_idx * stride_token + head_idx * @as(usize, self.embed_dim);
        return self.data_f32[start .. start + self.embed_dim];
    }

    pub inline fn engramKeySliceConst(self: *const LoopPleScratch, layer: u16, token_idx: usize, head_idx: usize) []const f32 {
        const d = self.engramValueDim();
        return self.engramRowSliceConst(layer, token_idx, head_idx)[0..d];
    }

    pub inline fn engramValSliceConst(self: *const LoopPleScratch, layer: u16, token_idx: usize, head_idx: usize) []const f32 {
        const d = self.engramValueDim();
        return self.engramRowSliceConst(layer, token_idx, head_idx)[d .. d * 2];
    }

    pub inline fn engramTokenHeadsConst(self: *const LoopPleScratch, layer: u16, token_idx: usize, num_heads: usize) []const f32 {
        const stride_token = @as(usize, self.max_heads) * @as(usize, self.embed_dim);
        const stride_layer = @as(usize, self.max_tokens) * stride_token;
        const start = @as(usize, layer) * stride_layer + token_idx * stride_token;
        return self.data_f32[start .. start + num_heads * @as(usize, self.embed_dim)];
    }
};

pub const LookupRequest = struct {
    layer: u16,
    row_index: u64,
};

pub const Config = struct {
    cache: PageCacheConfig = .{},
    /// When `true`, verifies CRC-32 of every `.amf` blob at startup.
    verify_blob_crcs: bool = false,
    /// Optional explicit table kind override (`null` auto-detects `.engram_conditional`
    /// when any `sparse_table` tensor name contains `"engram."`, else `.ple_ngram`).
    table_kind: ?MemoryTableKind = null,
};

const ShardDesc = struct {
    file_idx: u16,
    shard_id: u32,
    entry_idx: u32,
    dtype: amf.DType,
    blob_offset: u64,
    blob_bytes: u64,
    params_offset: u64,
    row_start: u64,
    num_rows: u64,
    row_code_bytes: usize,
    row_param_bytes: usize,

    fn lessThan(_: void, a: ShardDesc, b: ShardDesc) bool {
        if (a.shard_id != b.shard_id) return a.shard_id < b.shard_id;
        if (a.file_idx != b.file_idx) return a.file_idx < b.file_idx;
        return a.entry_idx < b.entry_idx;
    }
};

const LayerTable = struct {
    shard_start: usize = 0,
    shard_count: usize = 0,
    total_rows: u64 = 0,
};

/// Single-file or multi-shard NVMe `F_NOCACHE` Per-Layer Embedding / Engram store (C10).
pub const PleStore = struct {
    files: []amf.AmfFile,
    shards: []ShardDesc,
    layers: []LayerTable,
    embed_dim: u32,
    table_kind: MemoryTableKind,
    cache: PlePageCache,
    /// Preallocated scratch for page-straddling row assembly & FP32 dequantization
    /// (zero hot-path heap allocations).
    row_code_scratch: []u8,
    row_param_scratch: []u8,
    row_f32_scratch: []f32,

    pub fn openSingle(
        allocator: std.mem.Allocator,
        path: [:0]const u8,
        config: Config,
    ) Error!PleStore {
        const single = [_][:0]const u8{path};
        return open(allocator, &single, config);
    }

    /// Opens one or more `.amf` files with `F_NOCACHE`, indexes every
    /// `TensorKind.sparse_table` tensor across all files, and preallocates the
    /// wired 16 KiB `PlePageCache` and row dequantization scratch buffers.
    pub fn open(
        allocator: std.mem.Allocator,
        paths: []const [:0]const u8,
        config: Config,
    ) Error!PleStore {
        if (paths.len == 0) return error.NoFilesProvided;

        const files = try allocator.alloc(amf.AmfFile, paths.len);
        var opened_count: usize = 0;
        errdefer {
            for (0..opened_count) |i| files[i].close(allocator);
            allocator.free(files);
        }

        for (paths, 0..) |p, i| {
            files[i] = try amf.AmfFile.open(allocator, p);
            opened_count += 1;
            if (config.verify_blob_crcs) {
                try files[i].verifyAllBlobs(allocator);
            }
        }

        // Pass 1: discover all `sparse_table` entries and determine layer count & embed_dim.
        var max_layer: usize = 0;
        var total_shards: usize = 0;
        var embed_dim: u32 = 0;
        var max_row_code_bytes: usize = 0;
        var max_row_param_bytes: usize = 0;
        var saw_engram_name = false;

        for (files) |*f| {
            for (f.parsed.entries) |*entry| {
                if ((try entry.tensorKind()) != .sparse_table) continue;
                const dtype = try entry.dataType();
                const geom = try rowGeometry(dtype, entry.shape[1]);
                const layer_idx = try resolveLayerIndex(entry);
                if (layer_idx >= MAX_LAYERS) return error.LayerOutOfRange;

                const name = entry.nameSlice();
                if (std.mem.startsWith(u8, name, "engram.") or std.mem.indexOf(u8, name, ".engram.") != null) {
                    saw_engram_name = true;
                }

                if (embed_dim == 0) {
                    embed_dim = entry.shape[1];
                } else if (embed_dim != entry.shape[1]) {
                    return error.InconsistentEmbedDim;
                }
                max_layer = @max(max_layer, layer_idx);
                max_row_code_bytes = @max(max_row_code_bytes, geom.code_bytes);
                max_row_param_bytes = @max(max_row_param_bytes, geom.param_bytes);
                total_shards += 1;
            }
        }
        if (total_shards == 0) return error.NoSparseTablesFound;

        const detected_kind: MemoryTableKind = config.table_kind orelse
            (if (saw_engram_name) .engram_conditional else .ple_ngram);
        if (detected_kind == .engram_conditional and (embed_dim < 2 or (embed_dim % 2) != 0)) {
            return error.InvalidEngramLayout;
        }

        const num_layers = max_layer + 1;
        const layers = try allocator.alloc(LayerTable, num_layers);
        errdefer allocator.free(layers);
        @memset(layers, .{});

        // Count shards per layer.
        for (files) |*f| {
            for (f.parsed.entries) |*entry| {
                if ((try entry.tensorKind()) != .sparse_table) continue;
                const layer_idx = try resolveLayerIndex(entry);
                layers[layer_idx].shard_count += 1;
            }
        }

        // Prefix-sum shard ranges per layer.
        var cursor: usize = 0;
        for (layers) |*lt| {
            lt.shard_start = cursor;
            cursor += lt.shard_count;
            lt.shard_count = 0; // reused as write cursor below
        }

        const shards = try allocator.alloc(ShardDesc, total_shards);
        errdefer allocator.free(shards);

        for (files, 0..) |*f, file_idx| {
            for (f.parsed.entries, 0..) |*entry, entry_idx| {
                if ((try entry.tensorKind()) != .sparse_table) continue;
                const dtype = try entry.dataType();
                const geom = try rowGeometry(dtype, entry.shape[1]);
                const layer_idx = try resolveLayerIndex(entry);
                const shard_id = resolveShardIndex(entry);

                const lt = &layers[layer_idx];
                const dst_idx = lt.shard_start + lt.shard_count;
                shards[dst_idx] = .{
                    .file_idx = @intCast(file_idx),
                    .shard_id = shard_id,
                    .entry_idx = @intCast(entry_idx),
                    .dtype = dtype,
                    .blob_offset = entry.blob_offset,
                    .blob_bytes = entry.blob_bytes,
                    .params_offset = entry.params_offset,
                    .row_start = 0,
                    .num_rows = entry.shape[0],
                    .row_code_bytes = geom.code_bytes,
                    .row_param_bytes = geom.param_bytes,
                };
                lt.shard_count += 1;
            }
        }

        // Sort each layer's shards deterministically and compute cumulative global row offsets.
        for (layers) |*lt| {
            if (lt.shard_count == 0) continue;
            const layer_shards = shards[lt.shard_start .. lt.shard_start + lt.shard_count];
            std.mem.sort(ShardDesc, layer_shards, {}, ShardDesc.lessThan);
            var row_acc: u64 = 0;
            for (layer_shards) |*sh| {
                sh.row_start = row_acc;
                row_acc = std.math.add(u64, row_acc, sh.num_rows) catch return error.Overflow;
            }
            lt.total_rows = row_acc;
        }

        var cache = try PlePageCache.init(allocator, config.cache);
        errdefer cache.deinit(allocator);

        const row_code_scratch = try allocator.alloc(u8, @max(@as(usize, 1), max_row_code_bytes));
        errdefer allocator.free(row_code_scratch);
        const row_param_scratch = try allocator.alloc(u8, @max(@as(usize, 1), max_row_param_bytes));
        errdefer allocator.free(row_param_scratch);
        const row_f32_scratch = try allocator.alloc(f32, embed_dim);
        errdefer allocator.free(row_f32_scratch);

        return .{
            .files = files,
            .shards = shards,
            .layers = layers,
            .embed_dim = embed_dim,
            .table_kind = detected_kind,
            .cache = cache,
            .row_code_scratch = row_code_scratch,
            .row_param_scratch = row_param_scratch,
            .row_f32_scratch = row_f32_scratch,
        };
    }

    pub fn deinit(self: *PleStore, allocator: std.mem.Allocator) void {
        // Drain async readers BEFORE closing underlying F_NOCACHE file descriptors.
        self.cache.deinit(allocator);
        allocator.free(self.row_f32_scratch);
        allocator.free(self.row_param_scratch);
        allocator.free(self.row_code_scratch);
        allocator.free(self.shards);
        allocator.free(self.layers);
        for (self.files) |*f| f.close(allocator);
        allocator.free(self.files);
        self.files = &.{};
        self.shards = &.{};
        self.layers = &.{};
    }

    pub fn totalWiredBytes(self: *const PleStore) u64 {
        return self.cache.totalWiredBytes();
    }

    pub fn stats(self: *const PleStore) Stats {
        return self.cache.stats;
    }

    pub fn resetStats(self: *PleStore) void {
        self.cache.stats = .{};
    }

    pub fn clearCache(self: *PleStore) aio.EngineError!void {
        try self.cache.clear();
    }

    pub fn layerTotalRows(self: *const PleStore, layer: u16) Error!u64 {
        const lt = try self.layerTable(layer);
        return lt.total_rows;
    }

    pub fn layerShardCount(self: *const PleStore, layer: u16) Error!usize {
        const lt = try self.layerTable(layer);
        return lt.shard_count;
    }

    fn layerTable(self: *const PleStore, layer: u16) Error!*const LayerTable {
        if (layer >= self.layers.len) return error.LayerNotFound;
        const lt = &self.layers[layer];
        if (lt.total_rows == 0 or lt.shard_count == 0) return error.LayerNotFound;
        return lt;
    }

    fn resolveRowShard(self: *const PleStore, layer: u16, global_row: u64) Error!struct {
        shard: *const ShardDesc,
        shard_index: u32,
        local_row: u64,
    } {
        const lt = try self.layerTable(layer);
        if (global_row >= lt.total_rows) return error.RowOutOfBounds;
        const layer_shards = self.shards[lt.shard_start .. lt.shard_start + lt.shard_count];
        if (layer_shards.len == 1) {
            return .{
                .shard = &layer_shards[0],
                .shard_index = 0,
                .local_row = global_row,
            };
        }
        // Binary search across sorted cumulative `[row_start, row_start + num_rows)`.
        var lo: usize = 0;
        var hi: usize = layer_shards.len;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            const sh = &layer_shards[mid];
            if (global_row < sh.row_start) {
                hi = mid;
            } else if (global_row >= sh.row_start + sh.num_rows) {
                lo = mid + 1;
            } else {
                return .{
                    .shard = sh,
                    .shard_index = @intCast(mid),
                    .local_row = global_row - sh.row_start,
                };
            }
        }
        return error.RowOutOfBounds;
    }

    /// Deterministically maps a token tuple (`order >= 1`, e.g. bigram/trigram)
    /// to `(shard_index, local_row, global_row)` for `layer`.
    pub fn mapNgram(self: *const PleStore, layer: u16, tokens: []const u32, order: u8) Error!ShardRowLocation {
        if (order == 0) return error.InvalidNgramOrder;
        const lt = try self.layerTable(layer);
        const global_row = hashPleNgram(tokens, order, lt.total_rows);
        const resolved = try self.resolveRowShard(layer, global_row);
        return .{
            .shard_index = resolved.shard_index,
            .local_row = resolved.local_row,
            .global_row = global_row,
        };
    }

    fn prefetchSpan(self: *PleStore, file_idx: u16, file_offset: u64, len: usize) aio.EngineError!usize {
        if (len == 0) return 0;
        const fd = self.files[file_idx].fd;
        const first_page = file_offset / page.APPLE_PAGE_SIZE;
        const last_page = (file_offset + len - 1) / page.APPLE_PAGE_SIZE;
        var submitted: usize = 0;
        var p = first_page;
        while (p <= last_page) : (p += 1) {
            if (try self.cache.prefetchPage(fd, .{ .file_idx = file_idx, .page_idx = p })) {
                submitted += 1;
            }
        }
        return submitted;
    }

    fn prefetchRowPages(self: *PleStore, layer: u16, global_row: u64) Error!usize {
        const res = try self.resolveRowShard(layer, global_row);
        const sh = res.shard;
        var submitted: usize = 0;
        const code_off = sh.blob_offset + res.local_row * sh.row_code_bytes;
        submitted += try self.prefetchSpan(sh.file_idx, code_off, sh.row_code_bytes);
        if (sh.row_param_bytes > 0) {
            const param_off = sh.blob_offset + sh.params_offset + res.local_row * sh.row_param_bytes;
            submitted += try self.prefetchSpan(sh.file_idx, param_off, sh.row_param_bytes);
        }
        return submitted;
    }

    /// Reaps any already-completed background prefetches without blocking.
    pub fn reapPrefetches(self: *PleStore) aio.EngineError!usize {
        return self.cache.reapPrefetches();
    }

    /// Submits non-blocking asynchronous `F_NOCACHE` 16 KiB page prefetches for
    /// a batch of `(layer, row_index)` lookups via `KqueueAioEngine` /
    /// `EcorePreadPool`. Returns total newly queued NVMe page reads.
    pub fn prefetchBatch(self: *PleStore, requests: []const LookupRequest) Error!usize {
        _ = try self.cache.reapPrefetches();
        var total: usize = 0;
        for (requests) |req| {
            if (self.cache.async_reader.availableSlots() == 0) break;
            total += try self.prefetchRowPages(req.layer, req.row_index);
        }
        return total;
    }

    /// Convenience helper: hashes each token window (`order = 2` bigram / `3`
    /// trigram) and submits an async `F_NOCACHE` batch prefetch for `layer`.
    pub fn prefetchNgramBatch(
        self: *PleStore,
        layer: u16,
        token_windows: []const []const u32,
        order: u8,
    ) Error!usize {
        if (order == 0) return error.InvalidNgramOrder;
        const lt = try self.layerTable(layer);
        _ = try self.cache.reapPrefetches();
        var total: usize = 0;
        for (token_windows) |win| {
            if (self.cache.async_reader.availableSlots() == 0) break;
            const row_idx = hashPleNgram(win, order, lt.total_rows);
            total += try self.prefetchRowPages(layer, row_idx);
        }
        return total;
    }

    fn readSpanInto(self: *PleStore, file_idx: u16, file_offset: u64, dst: []u8) aio.EngineError!void {
        const fd = self.files[file_idx].fd;
        var pos: usize = 0;
        while (pos < dst.len) {
            const cur_off = file_offset + pos;
            const page_idx = cur_off / page.APPLE_PAGE_SIZE;
            const in_page: usize = @intCast(cur_off % page.APPLE_PAGE_SIZE);
            const take = @min(dst.len - pos, page.APPLE_PAGE_SIZE - in_page);
            const fetched = try self.cache.fetchPage(fd, .{ .file_idx = file_idx, .page_idx = page_idx });
            const pg = self.cache.pageSliceConst(fetched.slot_idx);
            @memcpy(dst[pos .. pos + take], pg[in_page .. in_page + take]);
            pos += take;
        }
    }

    /// Fetches one row from `PlePageCache` / NVMe and decodes it to FP32 into `out_row`.
    fn decodeRowF32(self: *PleStore, layer: u16, global_row: u64, out_row: []f32) Error!void {
        std.debug.assert(out_row.len == self.embed_dim);
        const res = try self.resolveRowShard(layer, global_row);
        const sh = res.shard;
        const fd = self.files[sh.file_idx].fd;
        const code_off = sh.blob_offset + res.local_row * sh.row_code_bytes;

        switch (sh.dtype) {
            .f16, .bf16 => {
                const in_page: usize = @intCast(code_off % page.APPLE_PAGE_SIZE);
                const raw_bytes: []const u8 = if (in_page + sh.row_code_bytes <= page.APPLE_PAGE_SIZE) blk: {
                    // Single-page fast path: decode directly from wired page with zero copy.
                    const fetched = try self.cache.fetchPage(fd, .{
                        .file_idx = sh.file_idx,
                        .page_idx = code_off / page.APPLE_PAGE_SIZE,
                    });
                    break :blk self.cache.pageSliceConst(fetched.slot_idx)[in_page .. in_page + sh.row_code_bytes];
                } else blk: {
                    const dst = self.row_code_scratch[0..sh.row_code_bytes];
                    try self.readSpanInto(sh.file_idx, code_off, dst);
                    break :blk dst;
                };

                if (sh.dtype == .f16) {
                    for (0..self.embed_dim) |d| {
                        const bits = std.mem.readInt(u16, raw_bytes[d * 2 ..][0..2], .little);
                        out_row[d] = @floatCast(@as(f16, @bitCast(bits)));
                    }
                } else {
                    for (0..self.embed_dim) |d| {
                        const bits = std.mem.readInt(u16, raw_bytes[d * 2 ..][0..2], .little);
                        out_row[d] = quant.bf16ToF32(bits);
                    }
                }
            },
            .q4_affine, .q3_affine, .q2_affine => {
                // Copy small per-row ScaleBias parameters into scratch first so
                // code page fetches can never evict the param page even if cache_pages == 1.
                const param_off = sh.blob_offset + sh.params_offset + res.local_row * sh.row_param_bytes;
                const param_bytes = self.row_param_scratch[0..sh.row_param_bytes];
                try self.readSpanInto(sh.file_idx, param_off, param_bytes);

                const in_page: usize = @intCast(code_off % page.APPLE_PAGE_SIZE);
                const code_bytes: []const u8 = if (in_page + sh.row_code_bytes <= page.APPLE_PAGE_SIZE) blk: {
                    const fetched = try self.cache.fetchPage(fd, .{
                        .file_idx = sh.file_idx,
                        .page_idx = code_off / page.APPLE_PAGE_SIZE,
                    });
                    break :blk self.cache.pageSliceConst(fetched.slot_idx)[in_page .. in_page + sh.row_code_bytes];
                } else blk: {
                    const dst = self.row_code_scratch[0..sh.row_code_bytes];
                    try self.readSpanInto(sh.file_idx, code_off, dst);
                    break :blk dst;
                };

                const groups = @as(usize, self.embed_dim) / 64;
                for (0..groups) |g| {
                    const scale_bits = std.mem.readInt(u16, param_bytes[g * 4 + 0 ..][0..2], .little);
                    const bias_bits = std.mem.readInt(u16, param_bytes[g * 4 + 2 ..][0..2], .little);
                    const sb: quant.ScaleBias = .{
                        .scale = @bitCast(scale_bits),
                        .bias = @bitCast(bias_bits),
                    };
                    const out_grp: *[64]f32 = out_row[g * 64 ..][0..64];
                    switch (sh.dtype) {
                        .q4_affine => Q4.dequantizeGroup(code_bytes[g * 32 ..][0..32], sb, out_grp),
                        .q3_affine => Q3.dequantizeGroup(code_bytes[g * 24 ..][0..24], sb, out_grp),
                        .q2_affine => Q2.dequantizeGroup(code_bytes[g * 16 ..][0..16], sb, out_grp),
                        else => unreachable,
                    }
                }
            },
            else => return error.UnsupportedDType,
        }
        self.cache.stats.rows_decoded += 1;
    }

    /// Looks up `row_indices[0..m]` (`m` in `1..64`) for `layer`, dequantizing
    /// rows from `PlePageCache` / NVMe and accumulating:
    ///   `out_hidden[m_i, d] += scale * row_dequant[d]`
    /// Zero dynamic heap allocations on the hot path.
    pub fn lookupAndAccumulate(
        self: *PleStore,
        layer: u16,
        row_indices: []const u64,
        scale: f32,
        out_hidden: []f32,
    ) Error!void {
        if (row_indices.len == 0 or row_indices.len > MAX_BATCH_TOKENS) {
            return error.InvalidBatchSize;
        }
        const dim: usize = self.embed_dim;
        const need = std.math.mul(usize, row_indices.len, dim) catch return error.Overflow;
        if (out_hidden.len < need) return error.BufferTooSmall;
        _ = try self.layerTable(layer);

        // Pipeline NVMe reads across the batch via non-blocking async prefetch.
        if (row_indices.len > 1) {
            for (row_indices) |r| {
                if (self.cache.async_reader.availableSlots() == 0) break;
                _ = try self.prefetchRowPages(layer, r);
            }
        }

        const row_buf = self.row_f32_scratch[0..dim];
        for (row_indices, 0..) |row_idx, m_i| {
            try self.decodeRowF32(layer, row_idx, row_buf);
            const dst = out_hidden[m_i * dim ..][0..dim];
            for (0..dim) |d| {
                dst[d] += scale * row_buf[d];
            }
        }
    }

    /// Mandatory `ADR-0019` `r = 0` Single-Lookup Loop Isolation:
    ///
    /// - `loop_step == 0`: fetches `row_indices[0..m]` from `PlePageCache`/NVMe,
    ///   stores the exact FP32 rows into `scratch[layer, 0..m, :]`, and accumulates
    ///   `out_hidden[m_i, d] += scale * row[d]`.
    /// - `loop_step >= 1` (`r < R`): reads strictly from wired `LoopPleScratch`
    ///   without touching `PlePageCache` or NVMe (`nvme_reads_delta == 0` across
    ///   all `r >= 1`, zero heap allocations, bit-exact parity with `r = 0`).
    pub fn lookupForLoopStep(
        self: *PleStore,
        layer: u16,
        row_indices: []const u64,
        scale: f32,
        loop_step: u32,
        scratch: *LoopPleScratch,
        out_hidden: []f32,
    ) Error!void {
        if (row_indices.len == 0 or row_indices.len > MAX_BATCH_TOKENS) {
            return error.InvalidBatchSize;
        }
        if (scratch.embed_dim != self.embed_dim or
            layer >= scratch.max_layers or
            row_indices.len > scratch.max_tokens)
        {
            return error.LoopScratchMismatch;
        }
        const dim: usize = self.embed_dim;
        const need = std.math.mul(usize, row_indices.len, dim) catch return error.Overflow;
        if (out_hidden.len < need) return error.BufferTooSmall;

        if (loop_step == 0) {
            scratch.valid_tokens[layer] = 0;
            _ = try self.layerTable(layer);
            if (row_indices.len > 1) {
                for (row_indices) |r| {
                    if (self.cache.async_reader.availableSlots() == 0) break;
                    _ = try self.prefetchRowPages(layer, r);
                }
            }
            for (row_indices, 0..) |row_idx, m_i| {
                const cached_row = scratch.rowSliceMut(layer, m_i);
                try self.decodeRowF32(layer, row_idx, cached_row);
                const dst = out_hidden[m_i * dim ..][0..dim];
                for (0..dim) |d| {
                    dst[d] += scale * cached_row[d];
                }
            }
            scratch.valid_tokens[layer] = @intCast(row_indices.len);
            self.cache.stats.loop_scratch_fills += 1;
            return;
        }

        // loop_step >= 1: strictly wired RAM reuse — zero cache / NVMe interaction!
        if (scratch.valid_tokens[layer] != row_indices.len) {
            return error.LoopScratchUnpopulated;
        }
        for (0..row_indices.len) |m_i| {
            const cached_row = scratch.rowSliceConst(layer, m_i);
            const dst = out_hidden[m_i * dim ..][0..dim];
            for (0..dim) |d| {
                dst[d] += scale * cached_row[d];
            }
        }
        scratch.loop_reuse_hits += 1;
        self.cache.stats.loop_scratch_hits += 1;
    }

    /// Returns the logical Engram key/value vector dimension `D` (`embed_dim / 2`)
    /// when this store is configured as `.engram_conditional`.
    pub fn engramValueDim(self: *const PleStore) Error!u32 {
        if (self.table_kind != .engram_conditional) return error.TableKindMismatch;
        if (self.embed_dim < 2 or (self.embed_dim % 2) != 0) return error.InvalidEngramLayout;
        return self.embed_dim / 2;
    }

    /// Deterministically maps an Engram `(head_idx, ngram_order)` suffix lookup
    /// to `(shard_index, local_row, global_row)` for `layer`.
    pub fn mapEngramHead(
        self: *const PleStore,
        layer: u16,
        tokens: []const u32,
        head_idx: u8,
        ngram_order: u8,
    ) Error!ShardRowLocation {
        if (head_idx >= ENGRAM_MAX_HEADS) return error.InvalidEngramHeads;
        if (ngram_order < ENGRAM_MIN_ORDER or ngram_order > ENGRAM_MAX_ORDER) {
            return error.InvalidEngramOrder;
        }
        const lt = try self.layerTable(layer);
        const global_row = hashEngramHead(tokens, head_idx, ngram_order, lt.total_rows);
        const resolved = try self.resolveRowShard(layer, global_row);
        return .{
            .shard_index = resolved.shard_index,
            .local_row = resolved.local_row,
            .global_row = global_row,
        };
    }

    /// Submits non-blocking asynchronous `F_NOCACHE` page prefetches for a batch
    /// of multi-head Engram token windows (`orders.len` in `2..8`, each order in `2..5`).
    pub fn prefetchEngramBatch(
        self: *PleStore,
        layer: u16,
        token_windows: []const []const u32,
        orders: []const u8,
    ) Error!usize {
        if (self.table_kind != .engram_conditional) return error.TableKindMismatch;
        if (token_windows.len == 0 or token_windows.len > MAX_BATCH_TOKENS) {
            return error.InvalidBatchSize;
        }
        if (orders.len < ENGRAM_MIN_HEADS or orders.len > ENGRAM_MAX_HEADS) {
            return error.InvalidEngramHeads;
        }
        for (orders) |ord| {
            if (ord < ENGRAM_MIN_ORDER or ord > ENGRAM_MAX_ORDER) return error.InvalidEngramOrder;
        }
        const lt = try self.layerTable(layer);
        _ = try self.cache.reapPrefetches();
        var total: usize = 0;
        for (token_windows) |win| {
            for (orders, 0..) |ord, h| {
                if (self.cache.async_reader.availableSlots() == 0) return total;
                const row_idx = hashEngramHead(win, @intCast(h), ord, lt.total_rows);
                total += try self.prefetchRowPages(layer, row_idx);
            }
        }
        return total;
    }

    /// Ensures `scratch` holds all `m * num_heads` packed `(k_h, v_h)` Engram
    /// rows for `layer` under mandatory `ADR-0019` `r = 0` Single-Lookup Isolation:
    /// - `loop_step == 0`: hashes all heads, submits async prefetches, decodes
    ///   rows once via `F_NOCACHE` + `PlePageCache` into `scratch`.
    /// - `loop_step >= 1`: validates `scratch` and reuses cached `(k_h, v_h)` rows
    ///   with strictly `0` cache/NVMe reads.
    fn ensureEngramScratchLoaded(
        self: *PleStore,
        layer: u16,
        token_windows: []const []const u32,
        orders: []const u8,
        loop_step: u32,
        scratch: *LoopPleScratch,
    ) Error!u32 {
        if (self.table_kind != .engram_conditional) return error.TableKindMismatch;
        const dim = try self.engramValueDim();
        if (token_windows.len == 0 or token_windows.len > MAX_BATCH_TOKENS) {
            return error.InvalidBatchSize;
        }
        if (orders.len < ENGRAM_MIN_HEADS or orders.len > ENGRAM_MAX_HEADS) {
            return error.InvalidEngramHeads;
        }
        for (orders) |ord| {
            if (ord < ENGRAM_MIN_ORDER or ord > ENGRAM_MAX_ORDER) return error.InvalidEngramOrder;
        }
        if (scratch.table_kind != .engram_conditional or
            scratch.embed_dim != self.embed_dim or
            layer >= scratch.max_layers or
            token_windows.len > scratch.max_tokens or
            orders.len > scratch.max_heads)
        {
            return error.LoopScratchMismatch;
        }

        const m = token_windows.len;
        const num_heads = orders.len;

        if (loop_step == 0) {
            scratch.valid_tokens[layer] = 0;
            scratch.valid_heads[layer] = 0;
            const lt = try self.layerTable(layer);

            var row_ids: [MAX_BATCH_TOKENS * ENGRAM_MAX_HEADS]u64 = undefined;
            for (token_windows, 0..) |win, m_i| {
                for (orders, 0..) |ord, h| {
                    row_ids[m_i * num_heads + h] = hashEngramHead(win, @intCast(h), ord, lt.total_rows);
                }
            }

            // Pipeline async F_NOCACHE page prefetches across all token x head rows.
            const total_lookups = m * num_heads;
            if (total_lookups > 1) {
                for (0..total_lookups) |idx| {
                    if (self.cache.async_reader.availableSlots() == 0) break;
                    _ = try self.prefetchRowPages(layer, row_ids[idx]);
                }
            }

            for (0..m) |m_i| {
                for (0..num_heads) |h| {
                    const dst_kv = scratch.engramRowSliceMut(layer, m_i, h);
                    try self.decodeRowF32(layer, row_ids[m_i * num_heads + h], dst_kv);
                }
            }
            scratch.valid_tokens[layer] = @intCast(m);
            scratch.valid_heads[layer] = @intCast(num_heads);
            self.cache.stats.loop_scratch_fills += 1;
            return dim;
        }

        // loop_step >= 1: strictly wired LoopPleScratch reuse (0 NVMe reads).
        if (scratch.valid_tokens[layer] != m or scratch.valid_heads[layer] != num_heads) {
            return error.LoopScratchUnpopulated;
        }
        scratch.loop_reuse_hits += 1;
        self.cache.stats.loop_scratch_hits += 1;
        return dim;
    }

    /// M7 `C10` 196B Engram conditional memory lookup + context-gated FP16 injection:
    ///
    /// - Fetches multi-head `(k_h, v_h)` rows via `F_NOCACHE` + `PlePageCache`
    ///   strictly once at `loop_step == 0` into `scratch` (`ADR-0019` Isolation #2);
    ///   `loop_step >= 1` reuses `scratch` with `0` extra SSD reads while re-gating
    ///   against the current `hidden_norm_f16`.
    /// - Writes `out_f16[m_i, d] = sum_h sigmoid(dot(hidden_norm_f16[m_i], k_h) * inv_sqrt_d) * v_h[d]`.
    /// - Zero dynamic heap allocations on the hot path.
    pub fn lookupEngramConditional(
        self: *PleStore,
        layer: u16,
        token_windows: []const []const u32,
        orders: []const u8,
        hidden_norm_f16: []const f16,
        loop_step: u32,
        scratch: *LoopPleScratch,
        out_f16: []f16,
    ) Error!void {
        const dim = try self.ensureEngramScratchLoaded(layer, token_windows, orders, loop_step, scratch);
        const d_usize: usize = dim;
        const need = std.math.mul(usize, token_windows.len, d_usize) catch return error.Overflow;
        if (hidden_norm_f16.len < need or out_f16.len < need) return error.BufferTooSmall;

        const num_heads: u8 = @intCast(orders.len);
        for (0..token_windows.len) |m_i| {
            const h_norm = hidden_norm_f16[m_i * d_usize ..][0..d_usize];
            const kv_heads = scratch.engramTokenHeadsConst(layer, m_i, orders.len);
            const out_row = out_f16[m_i * d_usize ..][0..d_usize];
            try applyEngramContextGateF16(h_norm, kv_heads, num_heads, dim, out_row);
        }
        self.cache.stats.engram_lookups += token_windows.len;
    }

    /// FP32 variant of `lookupEngramConditional` for FP32 residual accumulators.
    pub fn lookupEngramConditionalF32(
        self: *PleStore,
        layer: u16,
        token_windows: []const []const u32,
        orders: []const u8,
        hidden_norm_f32: []const f32,
        loop_step: u32,
        scratch: *LoopPleScratch,
        out_f32: []f32,
    ) Error!void {
        const dim = try self.ensureEngramScratchLoaded(layer, token_windows, orders, loop_step, scratch);
        const d_usize: usize = dim;
        const need = std.math.mul(usize, token_windows.len, d_usize) catch return error.Overflow;
        if (hidden_norm_f32.len < need or out_f32.len < need) return error.BufferTooSmall;

        const num_heads: u8 = @intCast(orders.len);
        for (0..token_windows.len) |m_i| {
            const h_norm = hidden_norm_f32[m_i * d_usize ..][0..d_usize];
            const kv_heads = scratch.engramTokenHeadsConst(layer, m_i, orders.len);
            const out_row = out_f32[m_i * d_usize ..][0..d_usize];
            try applyEngramContextGateF32(h_norm, kv_heads, num_heads, dim, out_row);
        }
        self.cache.stats.engram_lookups += token_windows.len;
    }
};

fn defaultWiredBudget() sysinfo.Error!budget_mod.Budget {
    const facts = try sysinfo.Facts.read();
    return budget_mod.Budget.compute(.{
        .memory_bytes = facts.memory_bytes,
        .user_wire_limit = facts.user_wire_limit,
        .global_user_wire_limit = facts.global_user_wire_limit,
        .gpu_working_set_bytes = facts.memory_bytes,
        .configured_cap = null,
        .headroom_bytes = budget_mod.DEFAULT_HEADROOM_BYTES,
    });
}

fn rowGeometry(dtype: amf.DType, embed_dim: u32) Error!struct { code_bytes: usize, param_bytes: usize } {
    if (embed_dim == 0) return error.InvalidShape;
    const dim: usize = embed_dim;
    return switch (dtype) {
        .f16, .bf16 => .{ .code_bytes = dim * 2, .param_bytes = 0 },
        .q4_affine => .{
            .code_bytes = Q4.rowWeightBytes(dim) catch return error.InvalidShape,
            .param_bytes = Q4.scaleBiasBytes(1, dim) catch return error.InvalidShape,
        },
        .q3_affine => .{
            .code_bytes = Q3.rowWeightBytes(dim) catch return error.InvalidShape,
            .param_bytes = Q3.scaleBiasBytes(1, dim) catch return error.InvalidShape,
        },
        .q2_affine => .{
            .code_bytes = Q2.rowWeightBytes(dim) catch return error.InvalidShape,
            .param_bytes = Q2.scaleBiasBytes(1, dim) catch return error.InvalidShape,
        },
        .f32, .u32 => error.UnsupportedDType,
    };
}

fn parseDecimalAfter(haystack: []const u8, needle: []const u8) ?u32 {
    const idx = std.mem.indexOf(u8, haystack, needle) orelse return null;
    const rest = haystack[idx + needle.len ..];
    var end: usize = 0;
    while (end < rest.len and rest[end] >= '0' and rest[end] <= '9') : (end += 1) {}
    if (end == 0) return null;
    return std.fmt.parseInt(u32, rest[0..end], 10) catch return null;
}

fn resolveLayerIndex(entry: *const amf.TensorEntry) Error!u16 {
    if (entry.layer != amf.NO_LAYER) {
        if (entry.layer >= MAX_LAYERS) return error.LayerOutOfRange;
        return entry.layer;
    }
    const name = entry.nameSlice();
    if (std.mem.indexOf(u8, name, "layers.") != null) {
        const parsed = parseDecimalAfter(name, "layers.") orelse return error.LayerOutOfRange;
        if (parsed >= MAX_LAYERS) return error.LayerOutOfRange;
        return @intCast(parsed);
    }
    return 0;
}

fn resolveShardIndex(entry: *const amf.TensorEntry) u32 {
    if (parseDecimalAfter(entry.nameSlice(), ".shard.")) |s| return s;
    if (entry.expert != amf.NO_EXPERT) return entry.expert;
    return 0;
}

fn preadExact(fd: std.c.fd_t, dst: []u8, offset: u64) aio.EngineError!void {
    var done: usize = 0;
    while (done < dst.len) {
        const rc = std.c.pread(fd, dst.ptr + done, dst.len - done, @intCast(offset + done));
        if (rc < 0) {
            const err = std.c.errno(rc);
            if (err == .INTR) continue;
            return error.ReadFailed;
        }
        if (rc == 0) return error.ShortRead;
        done += @intCast(rc);
    }
}

// ============================================================================
// Tests
// ============================================================================

const SyntheticTableSpec = struct {
    name: []const u8,
    layer: u16,
    dtype: amf.DType,
    num_rows: u32,
    embed_dim: u32,
    seed: u64,
};

fn buildSyntheticPleAmf(
    allocator: std.mem.Allocator,
    specs: []const SyntheticTableSpec,
    out_ref_rows: [][]f32,
) ![]u8 {
    std.debug.assert(specs.len == out_ref_rows.len);
    const count = specs.len;
    const dir_offset: usize = page.APPLE_PAGE_SIZE;
    const dir_raw = count * @sizeOf(amf.TensorEntry);
    const dir_bytes: usize = try page.alignUp(dir_raw);

    const entries = try allocator.alloc(amf.TensorEntry, count);
    defer allocator.free(entries);
    const blobs = try allocator.alloc([]u8, count);
    var built_blobs: usize = 0;
    defer {
        for (0..built_blobs) |i| allocator.free(blobs[i]);
        allocator.free(blobs);
    }

    var next_off: usize = dir_offset + dir_bytes;
    for (specs, 0..) |sp, i| {
        const geom = try rowGeometry(sp.dtype, sp.embed_dim);
        const data_bytes: usize = @as(usize, sp.num_rows) * geom.code_bytes;
        const params_offset: usize = if (geom.param_bytes > 0) amf.align64(data_bytes) else 0;
        const params_bytes: usize = @as(usize, sp.num_rows) * geom.param_bytes;
        const raw_blob_len = if (geom.param_bytes > 0) params_offset + params_bytes else data_bytes;
        const blob_bytes = try page.alignUp(raw_blob_len);

        const blob = try allocator.alloc(u8, blob_bytes);
        errdefer allocator.free(blob);
        @memset(blob, 0);

        const ref = out_ref_rows[i];
        std.debug.assert(ref.len == @as(usize, sp.num_rows) * @as(usize, sp.embed_dim));

        switch (sp.dtype) {
            .f16 => {
                for (0..sp.num_rows) |r| {
                    for (0..sp.embed_dim) |d| {
                        const raw_v = (@as(i32, @intCast((sp.seed +% r * 17 +% d * 13) % 31)) - 15);
                        const h: f16 = @floatCast(@as(f32, @floatFromInt(raw_v)) * 0.0625);
                        const bits: u16 = @bitCast(h);
                        const idx = r * sp.embed_dim + d;
                        std.mem.writeInt(u16, blob[idx * 2 ..][0..2], bits, .little);
                        ref[idx] = @floatCast(h);
                    }
                }
            },
            .bf16 => {
                for (0..sp.num_rows) |r| {
                    for (0..sp.embed_dim) |d| {
                        const raw_v = (@as(i32, @intCast((sp.seed +% r * 23 +% d * 7) % 29)) - 14);
                        const f: f32 = @as(f32, @floatFromInt(raw_v)) * 0.125;
                        const bits: u16 = quant.f32ToBf16(f);
                        const idx = r * sp.embed_dim + d;
                        std.mem.writeInt(u16, blob[idx * 2 ..][0..2], bits, .little);
                        ref[idx] = quant.bf16ToF32(bits);
                    }
                }
            },
            .q4_affine, .q2_affine => {
                const groups = sp.embed_dim / 64;
                const max_q: u64 = if (sp.dtype == .q4_affine) 15 else 3;
                const grp_code_bytes: usize = if (sp.dtype == .q4_affine) 32 else 16;
                var codes: [64]u8 = undefined;
                for (0..sp.num_rows) |r| {
                    for (0..groups) |g| {
                        const sb: quant.ScaleBias = .{
                            .scale = @floatCast(0.0625 * @as(f32, @floatFromInt(((sp.seed +% r +% g) % 4) + 1))),
                            .bias = @floatCast(-0.25 + 0.125 * @as(f32, @floatFromInt((sp.seed +% r * 3 +% g) % 5))),
                        };
                        const p_off = params_offset + (r * groups + g) * 4;
                        std.mem.writeInt(u16, blob[p_off + 0 ..][0..2], @bitCast(sb.scale), .little);
                        std.mem.writeInt(u16, blob[p_off + 2 ..][0..2], @bitCast(sb.bias), .little);

                        for (0..64) |j| {
                            const q: u8 = @intCast((sp.seed +% r * 11 +% g * 5 +% j * 3) % (max_q + 1));
                            codes[j] = q;
                            ref[r * sp.embed_dim + g * 64 + j] =
                                @as(f32, @floatFromInt(q)) * @as(f32, @floatCast(sb.scale)) +
                                @as(f32, @floatCast(sb.bias));
                        }
                        const c_off = r * geom.code_bytes + g * grp_code_bytes;
                        if (sp.dtype == .q4_affine) {
                            Q4.packGroup(&codes, blob[c_off..][0..32]);
                        } else {
                            Q2.packGroup(&codes, blob[c_off..][0..16]);
                        }
                    }
                }
            },
            else => return error.UnsupportedDType,
        }

        var e = amf.TensorEntry.zeroed(sp.name, .sparse_table, sp.dtype);
        e.rank = 2;
        e.layer = sp.layer;
        e.shape = .{ sp.num_rows, sp.embed_dim, 0, 0 };
        e.data_bytes = data_bytes;
        if (geom.param_bytes > 0) {
            e.quant_bits = if (sp.dtype == .q4_affine) 4 else 2;
            e.group_size = 64;
            e.params_offset = params_offset;
            e.params_bytes = params_bytes;
        }
        e.blob_offset = next_off;
        e.blob_bytes = blob_bytes;
        e.blob_crc32 = std.hash.Crc32.hash(blob);

        entries[i] = e;
        blobs[i] = blob;
        built_blobs += 1;
        next_off += blob_bytes;
    }

    const image = try allocator.alloc(u8, next_off);
    errdefer allocator.free(image);
    @memset(image, 0);

    for (entries, 0..) |e, i| {
        @memcpy(image[dir_offset + i * @sizeOf(amf.TensorEntry) ..][0..@sizeOf(amf.TensorEntry)], std.mem.asBytes(&e));
        @memcpy(image[@intCast(e.blob_offset)..][0..@intCast(e.blob_bytes)], blobs[i]);
    }

    var hdr: amf.Header = .{
        .magic = amf.MAGIC,
        .version = amf.FORMAT_VERSION,
        .flags = 0,
        .page_size = page.APPLE_PAGE_SIZE,
        .checksum_algo = amf.CHECKSUM_CRC32_IEEE,
        .tensor_count = @intCast(count),
        .reserved0 = 0,
        .directory_offset = dir_offset,
        .directory_bytes = dir_bytes,
        .directory_crc32 = std.hash.Crc32.hash(image[dir_offset..][0..dir_raw]),
        .header_crc32 = 0,
        .model_id = @splat(0),
        .reserved1 = 0,
    };
    const model_name = "synthetic-ple-c10";
    @memcpy(hdr.model_id[0..model_name.len], model_name);
    hdr.header_crc32 = amf.computeHeaderCrc32(&hdr);
    @memcpy(image[0..@sizeOf(amf.Header)], std.mem.asBytes(&hdr));
    return image;
}

fn testBudget(cap_bytes: usize) budget_mod.Budget {
    return budget_mod.Budget.compute(.{
        .memory_bytes = cap_bytes * 8,
        .user_wire_limit = cap_bytes * 2,
        .global_user_wire_limit = cap_bytes * 2,
        .gpu_working_set_bytes = cap_bytes * 2,
        .configured_cap = null,
        .headroom_bytes = page.APPLE_PAGE_SIZE,
    });
}

test "hashPleNgram is deterministic across bigrams, trigrams, BOS padding and multi-shard mapping" {
    const seq = [_]u32{ 101, 202, 303, 404 };
    const total_rows: u64 = 10_007; // prime non-power-of-two table size

    const bi_a = hashPleNgram(&seq, 2, total_rows);
    const bi_b = hashPleNgram(seq[2..4], 2, total_rows);
    try std.testing.expectEqual(bi_a, bi_b);
    try std.testing.expect(bi_a < total_rows);

    const tri_a = hashPleNgram(&seq, 3, total_rows);
    const tri_b = hashPleNgram(seq[1..4], 3, total_rows);
    try std.testing.expectEqual(tri_a, tri_b);
    try std.testing.expect(tri_a < total_rows);

    // Order domain separation: bigram vs trigram on same trailing window differ.
    const pair = [_]u32{ 303, 404 };
    try std.testing.expect(hashPleNgram(&pair, 2, total_rows) != hashPleNgram(&pair, 3, total_rows));

    // BOS left-padding for short token slices stays deterministic.
    const single = [_]u32{42};
    try std.testing.expectEqual(hashPleNgram(&single, 3, total_rows), hashPleNgram(&single, 3, total_rows));

    // Multi-shard mapping maps global_row accurately into shard boundaries.
    const shards = [_]u64{ 3000, 4000, 3007 };
    const loc = try mapNgramToShardRow(&seq, 3, &shards);
    try std.testing.expectEqual(tri_a, loc.global_row);
    const expected_prefix: u64 = switch (loc.shard_index) {
        0 => 0,
        1 => 3000,
        2 => 7000,
        else => unreachable,
    };
    try std.testing.expectEqual(loc.global_row - expected_prefix, loc.local_row);
    try std.testing.expect(loc.local_row < shards[loc.shard_index]);
}

test "PleStore F_NOCACHE disk round-trip: f16, bf16, q4_affine, q2_affine, multi-shard, straddling rows, async prefetch and LRU" {
    const gpa = std.testing.allocator;
    // Embed dim 192 (multiple of 64, 384 B in f16/bf16, 96 B in q4, 48 B in q2)
    // does not evenly divide 16384 B -> exercises 16 KiB page-straddling rows!
    const dim: u32 = 192;

    // Layer 0 split across TWO .amf shard files (f16 shard 0 + f16 shard 1).
    // Layer 1: bf16; Layer 2: q4_affine; Layer 3: q2_affine.
    const shard0_specs = [_]SyntheticTableSpec{
        .{ .name = "ple.layers.0.shard.0.weight", .layer = 0, .dtype = .f16, .num_rows = 70, .embed_dim = dim, .seed = 11 },
        .{ .name = "ple.layers.1.weight", .layer = 1, .dtype = .bf16, .num_rows = 60, .embed_dim = dim, .seed = 29 },
    };
    const shard1_specs = [_]SyntheticTableSpec{
        .{ .name = "ple.layers.0.shard.1.weight", .layer = 0, .dtype = .f16, .num_rows = 55, .embed_dim = dim, .seed = 43 },
        .{ .name = "ple.layers.2.weight", .layer = 2, .dtype = .q4_affine, .num_rows = 240, .embed_dim = dim, .seed = 67 },
        .{ .name = "ple.layers.3.weight", .layer = 3, .dtype = .q2_affine, .num_rows = 420, .embed_dim = dim, .seed = 89 },
    };

    const ref_l0_s0 = try gpa.alloc(f32, 70 * dim);
    defer gpa.free(ref_l0_s0);
    const ref_l1 = try gpa.alloc(f32, 60 * dim);
    defer gpa.free(ref_l1);
    const ref_l0_s1 = try gpa.alloc(f32, 55 * dim);
    defer gpa.free(ref_l0_s1);
    const ref_l2 = try gpa.alloc(f32, 240 * dim);
    defer gpa.free(ref_l2);
    const ref_l3 = try gpa.alloc(f32, 420 * dim);
    defer gpa.free(ref_l3);

    var refs0 = [_][]f32{ ref_l0_s0, ref_l1 };
    var refs1 = [_][]f32{ ref_l0_s1, ref_l2, ref_l3 };

    const img0 = try buildSyntheticPleAmf(gpa, &shard0_specs, &refs0);
    defer gpa.free(img0);
    const img1 = try buildSyntheticPleAmf(gpa, &shard1_specs, &refs1);
    defer gpa.free(img1);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "ple_shard0.amf", .data = img0 });
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "ple_shard1.amf", .data = img1 });

    var p0_buf: [std.fs.max_path_bytes]u8 = undefined;
    const p0_raw = try std.fmt.bufPrint(p0_buf[0 .. p0_buf.len - 1], ".zig-cache/tmp/{s}/ple_shard0.amf", .{tmp.sub_path});
    p0_buf[p0_raw.len] = 0;
    const path0: [:0]const u8 = p0_buf[0..p0_raw.len :0];

    var p1_buf: [std.fs.max_path_bytes]u8 = undefined;
    const p1_raw = try std.fmt.bufPrint(p1_buf[0 .. p1_buf.len - 1], ".zig-cache/tmp/{s}/ple_shard1.amf", .{tmp.sub_path});
    p1_buf[p1_raw.len] = 0;
    const path1: [:0]const u8 = p1_buf[0..p1_raw.len :0];

    const paths = [_][:0]const u8{ path0, path1 };

    // Exercise BOTH async backends (`kqueue_aio` and `ecore_pread`).
    for ([_]aio.IoBackend{ .kqueue_aio, .ecore_pread }) |backend| {
        const cache_pages: usize = 4; // small enough to force real LRU evictions across multi-page tables!
        var store = try PleStore.open(gpa, &paths, .{
            .verify_blob_crcs = true,
            .cache = .{
                .cache_pages = cache_pages,
                .max_inflight = 2,
                .io_backend = backend,
                .worker_count = 1,
                .budget_override = testBudget(cache_pages * page.APPLE_PAGE_SIZE),
            },
        });
        defer store.deinit(gpa);

        try std.testing.expectEqual(@as(u64, cache_pages * page.APPLE_PAGE_SIZE), store.totalWiredBytes());
        try std.testing.expectEqual(@as(u64, 125), try store.layerTotalRows(0));
        try std.testing.expectEqual(@as(usize, 2), try store.layerShardCount(0));
        try std.testing.expectEqual(@as(u64, 60), try store.layerTotalRows(1));
        try std.testing.expectEqual(@as(u64, 240), try store.layerTotalRows(2));
        try std.testing.expectEqual(@as(u64, 420), try store.layerTotalRows(3));

        // 1. Multi-shard Layer 0 (f16) with M = 64 batch spanning shard 0, straddling row 42 (42*384=16128..16512), and shard 1!
        var rows_l0: [64]u64 = undefined;
        var pf_reqs: [4]LookupRequest = undefined;
        for (0..64) |m_i| {
            rows_l0[m_i] = (m_i * 7 + 42) % 125;
        }
        for (0..4) |j| pf_reqs[j] = .{ .layer = 0, .row_index = rows_l0[j] };
        const prefetched = try store.prefetchBatch(&pf_reqs);
        try std.testing.expect(prefetched >= 1);

        const out_l0 = try gpa.alloc(f32, 64 * dim);
        defer gpa.free(out_l0);
        for (out_l0, 0..) |*v, idx| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(idx % 9)) - 4)) * 0.25;
        const base_l0 = try gpa.dupe(f32, out_l0);
        defer gpa.free(base_l0);

        const scale: f32 = 0.75;
        try store.lookupAndAccumulate(0, &rows_l0, scale, out_l0);

        for (0..64) |m_i| {
            const g_row = rows_l0[m_i];
            const ref_src = if (g_row < 70)
                ref_l0_s0[@intCast(g_row * dim)..][0..dim]
            else
                ref_l0_s1[@intCast((g_row - 70) * dim)..][0..dim];
            for (0..dim) |d| {
                const expected = base_l0[m_i * dim + d] + scale * ref_src[d];
                try std.testing.expectApproxEqAbs(expected, out_l0[m_i * dim + d], 1e-6);
            }
        }

        // 2. Verify bf16 (Layer 1), q4_affine (Layer 2), and q2_affine (Layer 3) across M = 1, 5, 16.
        const layer_checks = [_]struct { layer: u16, ref: []const f32, total_rows: u64, m: usize }{
            .{ .layer = 1, .ref = ref_l1, .total_rows = 60, .m = 1 },
            .{ .layer = 2, .ref = ref_l2, .total_rows = 240, .m = 5 },
            .{ .layer = 3, .ref = ref_l3, .total_rows = 420, .m = 16 },
        };
        for (layer_checks) |chk| {
            var row_ids: [16]u64 = undefined;
            for (0..chk.m) |m_i| {
                const toks = [_]u32{ @intCast(100 + m_i), @intCast(500 + chk.layer), @intCast(900 + m_i * 3) };
                const loc = try store.mapNgram(chk.layer, &toks, 3);
                row_ids[m_i] = loc.global_row;
            }
            const out_buf = try gpa.alloc(f32, chk.m * dim);
            defer gpa.free(out_buf);
            @memset(out_buf, 1.5);
            try store.lookupAndAccumulate(chk.layer, row_ids[0..chk.m], -0.5, out_buf);
            for (0..chk.m) |m_i| {
                const r: usize = @intCast(row_ids[m_i]);
                for (0..dim) |d| {
                    const expected = 1.5 + (-0.5) * chk.ref[r * dim + d];
                    try std.testing.expectApproxEqAbs(expected, out_buf[m_i * dim + d], 1e-5);
                }
            }
        }

        const st = store.stats();
        try std.testing.expect(st.nvme_reads > 0);
        try std.testing.expect(st.cache_hits > 0);
        try std.testing.expect(st.evictions > 0);
        try std.testing.expectEqual(@as(u64, 64 + 1 + 5 + 16), st.rows_decoded);
    }
}

test "ADR-0019 r=0 Single-Lookup Loop Isolation: nvme_reads_delta == 0 across r >= 1 with bit-exact parity" {
    const gpa = std.testing.allocator;
    const dim: u32 = 128;
    const specs = [_]SyntheticTableSpec{
        .{ .name = "ple.layers.0.weight", .layer = 0, .dtype = .q4_affine, .num_rows = 128, .embed_dim = dim, .seed = 101 },
        .{ .name = "ple.layers.1.weight", .layer = 1, .dtype = .q2_affine, .num_rows = 128, .embed_dim = dim, .seed = 202 },
    };
    const ref0 = try gpa.alloc(f32, 128 * dim);
    defer gpa.free(ref0);
    const ref1 = try gpa.alloc(f32, 128 * dim);
    defer gpa.free(ref1);
    var refs = [_][]f32{ ref0, ref1 };

    const img = try buildSyntheticPleAmf(gpa, &specs, &refs);
    defer gpa.free(img);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "ple_loop.amf", .data = img });

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const raw = try std.fmt.bufPrint(path_buf[0 .. path_buf.len - 1], ".zig-cache/tmp/{s}/ple_loop.amf", .{tmp.sub_path});
    path_buf[raw.len] = 0;
    const path: [:0]const u8 = path_buf[0..raw.len :0];

    var store = try PleStore.openSingle(gpa, path, .{
        .cache = .{
            .cache_pages = 8,
            .max_inflight = 4,
            .io_backend = .auto,
            .budget_override = testBudget(8 * page.APPLE_PAGE_SIZE),
        },
    });
    defer store.deinit(gpa);

    var loop_scratch = try LoopPleScratch.init(
        gpa,
        2,
        16,
        dim,
        testBudget(4 * page.APPLE_PAGE_SIZE),
    );
    defer loop_scratch.deinit(gpa);
    try std.testing.expect(loop_scratch.totalWiredBytes() >= 2 * 16 * dim * @sizeOf(f32));

    const m: usize = 8;
    const rows = [_]u64{ 3, 19, 44, 65, 88, 102, 111, 127 };

    // Calling loop_step >= 1 before loop_step == 0 must fail cleanly rather than reading NVMe.
    var unpop_out: [m * dim]f32 = @splat(0.0);
    try std.testing.expectError(
        error.LoopScratchUnpopulated,
        store.lookupForLoopStep(0, &rows, 1.0, 1, &loop_scratch, &unpop_out),
    );
    try std.testing.expectEqual(@as(u64, 0), store.stats().nvme_reads);

    // Loop step r = 0: fetches from NVMe into PlePageCache AND caches exact FP32 rows in LoopPleScratch.
    var out_r0_l0: [m * dim]f32 = @splat(0.0);
    var out_r0_l1: [m * dim]f32 = @splat(0.0);
    try store.lookupForLoopStep(0, &rows, 0.5, 0, &loop_scratch, &out_r0_l0);
    try store.lookupForLoopStep(1, &rows, 0.25, 0, &loop_scratch, &out_r0_l1);

    const nvme_reads_after_r0 = store.stats().nvme_reads;
    try std.testing.expect(nvme_reads_after_r0 > 0);
    try std.testing.expectEqual(@as(u64, 2), store.stats().loop_scratch_fills);

    // Explicitly flush the entire PlePageCache! Even with a cold page cache,
    // loop_step >= 1 MUST serve strictly from LoopPleScratch with ZERO NVMe reads.
    try store.clearCache();

    for ([_]u32{ 1, 2, 3 }) |r| {
        var out_l0: [m * dim]f32 = @splat(0.0);
        var out_l1: [m * dim]f32 = @splat(0.0);
        try store.lookupForLoopStep(0, &rows, 0.5, r, &loop_scratch, &out_l0);
        try store.lookupForLoopStep(1, &rows, 0.25, r, &loop_scratch, &out_l1);

        // Bit-exact output parity with r = 0!
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&out_r0_l0), std.mem.sliceAsBytes(&out_l0));
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&out_r0_l1), std.mem.sliceAsBytes(&out_l1));
    }

    // Mandatory ADR-0019 invariant: nvme_reads_delta == 0 across all r >= 1!
    try std.testing.expectEqual(nvme_reads_after_r0, store.stats().nvme_reads);
    try std.testing.expectEqual(@as(u64, 6), loop_scratch.loop_reuse_hits);
    try std.testing.expectEqual(@as(u64, 6), store.stats().loop_scratch_hits);

    // Next decode step resets LoopPleScratch validity cleanly.
    loop_scratch.resetStep();
    try std.testing.expectError(
        error.LoopScratchUnpopulated,
        store.lookupForLoopStep(0, &rows, 0.5, 1, &loop_scratch, &unpop_out),
    );
}

test "C10 Engram hashEngramHead determinism, multi-head independence and sharded mapping" {
    const seq_a = [_]u32{ 11, 22, 33, 44, 55 };
    const seq_b = [_]u32{ 99, 22, 33, 44, 55 };
    const total_rows: u64 = 1_000_003;

    // Order 4 depends only on the trailing 4 tokens (22, 33, 44, 55).
    for (0..ENGRAM_MAX_HEADS) |h_usize| {
        const h: u8 = @intCast(h_usize);
        const r_a = hashEngramHead(&seq_a, h, 4, total_rows);
        const r_b = hashEngramHead(&seq_b, h, 4, total_rows);
        try std.testing.expectEqual(r_a, r_b);
        try std.testing.expect(r_a < total_rows);
        // Order 5 distinguishes prefix 11 vs 99.
        const r5_a = hashEngramHead(&seq_a, h, 5, total_rows);
        const r5_b = hashEngramHead(&seq_b, h, 5, total_rows);
        try std.testing.expect(r5_a != r5_b);
    }

    // Different heads for the same (tokens, order) map to distinct rows.
    const h0 = hashEngramHead(&seq_a, 0, 3, total_rows);
    const h1 = hashEngramHead(&seq_a, 1, 3, total_rows);
    const h2 = hashEngramHead(&seq_a, 2, 3, total_rows);
    const h3 = hashEngramHead(&seq_a, 3, 3, total_rows);
    try std.testing.expect(h0 != h1 and h1 != h2 and h2 != h3 and h0 != h3);

    // BOS left-padding for short prefixes.
    const short = [_]u32{42};
    const r_short_2 = hashEngramHead(&short, 0, 2, total_rows);
    const r_short_4 = hashEngramHead(&short, 0, 4, total_rows);
    try std.testing.expect(r_short_2 != r_short_4);

    // Sharded mapping across uneven shard counts.
    const shards = [_]u64{ 100, 250, 150 };
    const loc = try mapEngramHeadToShardRow(&seq_a, 3, 4, &shards);
    try std.testing.expectEqual(hashEngramHead(&seq_a, 3, 4, 500), loc.global_row);
    try std.testing.expect(loc.shard_index < shards.len);
    try std.testing.expect(loc.local_row < shards[loc.shard_index]);

    try std.testing.expectError(error.InvalidEngramOrder, mapEngramHeadToShardRow(&seq_a, 0, 1, &shards));
    try std.testing.expectError(error.InvalidEngramOrder, mapEngramHeadToShardRow(&seq_a, 0, 6, &shards));
    try std.testing.expectError(error.InvalidEngramHeads, mapEngramHeadToShardRow(&seq_a, ENGRAM_MAX_HEADS, 3, &shards));
}

test "C10 Engram multi-head context-gated lookup, multi-shard F_NOCACHE, and ADR-0019 r=0 loop isolation" {
    const gpa = std.testing.allocator;
    const val_dim: u32 = 64;
    const packed_dim: u32 = val_dim * 2; // [k_h (64) || v_h (64)] per row

    // Layer 0 spans two separate .amf shard files (q4_affine + f16) using engram.* naming.
    const specs_s0 = [_]SyntheticTableSpec{
        .{ .name = "engram.layers.0.shard.0.weight", .layer = 0, .dtype = .q4_affine, .num_rows = 64, .embed_dim = packed_dim, .seed = 701 },
    };
    const specs_s1 = [_]SyntheticTableSpec{
        .{ .name = "engram.layers.0.shard.1.weight", .layer = 0, .dtype = .f16, .num_rows = 64, .embed_dim = packed_dim, .seed = 702 },
    };

    const ref_s0 = try gpa.alloc(f32, 64 * packed_dim);
    defer gpa.free(ref_s0);
    const ref_s1 = try gpa.alloc(f32, 64 * packed_dim);
    defer gpa.free(ref_s1);
    var refs_s0 = [_][]f32{ref_s0};
    var refs_s1 = [_][]f32{ref_s1};

    const img_s0 = try buildSyntheticPleAmf(gpa, &specs_s0, &refs_s0);
    defer gpa.free(img_s0);
    const img_s1 = try buildSyntheticPleAmf(gpa, &specs_s1, &refs_s1);
    defer gpa.free(img_s1);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "engram-s0.amf", .data = img_s0 });
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "engram-s1.amf", .data = img_s1 });

    var p0_buf: [std.fs.max_path_bytes]u8 = undefined;
    var p1_buf: [std.fs.max_path_bytes]u8 = undefined;
    const p0_raw = try std.fmt.bufPrint(p0_buf[0 .. p0_buf.len - 1], ".zig-cache/tmp/{s}/engram-s0.amf", .{tmp.sub_path});
    p0_buf[p0_raw.len] = 0;
    const p1_raw = try std.fmt.bufPrint(p1_buf[0 .. p1_buf.len - 1], ".zig-cache/tmp/{s}/engram-s1.amf", .{tmp.sub_path});
    p1_buf[p1_raw.len] = 0;
    const paths = [_][:0]const u8{ p0_buf[0..p0_raw.len :0], p1_buf[0..p1_raw.len :0] };

    var store = try PleStore.open(gpa, &paths, .{
        .cache = .{
            .cache_pages = 8,
            .max_inflight = 4,
            .io_backend = .auto,
            .budget_override = testBudget(8 * page.APPLE_PAGE_SIZE),
        },
        .verify_blob_crcs = true,
    });
    defer store.deinit(gpa);

    try std.testing.expectEqual(MemoryTableKind.engram_conditional, store.table_kind);
    try std.testing.expectEqual(val_dim, try store.engramValueDim());
    try std.testing.expectEqual(@as(u64, 128), try store.layerTotalRows(0));
    try std.testing.expectEqual(@as(usize, 2), try store.layerShardCount(0));

    const orders = [_]u8{ 2, 3, 4, 5 }; // 4 Engram heads spanning orders 2..5
    var scratch = try LoopPleScratch.initEngram(
        gpa,
        1,
        8,
        @intCast(orders.len),
        val_dim,
        testBudget(8 * page.APPLE_PAGE_SIZE),
    );
    defer scratch.deinit(gpa);

    const tok0 = [_]u32{ 10, 20, 30, 40, 50 };
    const tok1 = [_]u32{ 5, 15, 25, 35, 45 };
    const tok2 = [_]u32{ 7, 14 };
    const windows = [_][]const u32{ &tok0, &tok1, &tok2 };
    const m = windows.len;

    var hidden_r0: [m * val_dim]f16 = undefined;
    var hidden_r2: [m * val_dim]f16 = undefined;
    for (0..m) |mi| {
        for (0..val_dim) |d| {
            const v0: f32 = @sin(@as(f32, @floatFromInt(mi * 37 + d + 1))) * 0.5;
            const v2: f32 = @cos(@as(f32, @floatFromInt(mi * 19 + d + 3))) * 0.5;
            hidden_r0[mi * val_dim + d] = @floatCast(v0);
            hidden_r2[mi * val_dim + d] = @floatCast(v2);
        }
    }

    // Calling loop_step >= 1 before loop_step == 0 must fail cleanly without touching NVMe.
    var out_unpop: [m * val_dim]f16 = @splat(0.0);
    try std.testing.expectError(
        error.LoopScratchUnpopulated,
        store.lookupEngramConditional(0, &windows, &orders, &hidden_r0, 1, &scratch, &out_unpop),
    );
    try std.testing.expectEqual(@as(u64, 0), store.stats().nvme_reads);

    // Optional async prefetch + r = 0 lookup.
    _ = try store.prefetchEngramBatch(0, &windows, &orders);
    var out_r0: [m * val_dim]f16 = @splat(0.0);
    try store.lookupEngramConditional(0, &windows, &orders, &hidden_r0, 0, &scratch, &out_r0);

    const nvme_reads_after_r0 = store.stats().nvme_reads;
    try std.testing.expect(nvme_reads_after_r0 > 0);
    try std.testing.expectEqual(@as(u64, m), store.stats().engram_lookups);

    // Reference check for r = 0 directly against exact packed (k_h, v_h) rows.
    const inv_sqrt_d: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(val_dim)));
    for (0..m) |mi| {
        var exp_row: [val_dim]f32 = @splat(0.0);
        for (orders, 0..) |ord, h| {
            const g_row = hashEngramHead(windows[mi], @intCast(h), ord, 128);
            const row_ref: []const f32 = if (g_row < 64)
                ref_s0[@as(usize, @intCast(g_row)) * packed_dim ..][0..packed_dim]
            else
                ref_s1[@as(usize, @intCast(g_row - 64)) * packed_dim ..][0..packed_dim];
            const k_h = row_ref[0..val_dim];
            const v_h = row_ref[val_dim..packed_dim];
            var dot: f32 = 0.0;
            for (0..val_dim) |d| {
                dot += @as(f32, @floatCast(hidden_r0[mi * val_dim + d])) * k_h[d];
            }
            const gate = 1.0 / (1.0 + @exp(-(dot * inv_sqrt_d)));
            for (0..val_dim) |d| {
                exp_row[d] += gate * v_h[d];
            }
        }
        for (0..val_dim) |d| {
            const got: f32 = @floatCast(out_r0[mi * val_dim + d]);
            try std.testing.expectApproxEqAbs(exp_row[d], got, 5e-3);
        }
    }

    // Flush PlePageCache completely — loop_step >= 1 MUST issue 0 NVMe reads!
    try store.clearCache();

    // r = 1 with identical hidden_norm_f16 -> bit-exact parity with r = 0!
    var out_r1: [m * val_dim]f16 = @splat(0.0);
    try store.lookupEngramConditional(0, &windows, &orders, &hidden_r0, 1, &scratch, &out_r1);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&out_r0), std.mem.sliceAsBytes(&out_r1));

    // r = 2 with updated hidden_norm_f16 -> dynamically re-gates cached (k_h, v_h) with 0 NVMe reads!
    var out_r2: [m * val_dim]f16 = @splat(0.0);
    try store.lookupEngramConditional(0, &windows, &orders, &hidden_r2, 2, &scratch, &out_r2);
    for (0..m) |mi| {
        var exp_row: [val_dim]f32 = @splat(0.0);
        for (orders, 0..) |ord, h| {
            const g_row = hashEngramHead(windows[mi], @intCast(h), ord, 128);
            const row_ref: []const f32 = if (g_row < 64)
                ref_s0[@as(usize, @intCast(g_row)) * packed_dim ..][0..packed_dim]
            else
                ref_s1[@as(usize, @intCast(g_row - 64)) * packed_dim ..][0..packed_dim];
            const k_h = row_ref[0..val_dim];
            const v_h = row_ref[val_dim..packed_dim];
            var dot: f32 = 0.0;
            for (0..val_dim) |d| {
                dot += @as(f32, @floatCast(hidden_r2[mi * val_dim + d])) * k_h[d];
            }
            const gate = 1.0 / (1.0 + @exp(-(dot * inv_sqrt_d)));
            for (0..val_dim) |d| {
                exp_row[d] += gate * v_h[d];
            }
        }
        for (0..val_dim) |d| {
            const got: f32 = @floatCast(out_r2[mi * val_dim + d]);
            try std.testing.expectApproxEqAbs(exp_row[d], got, 5e-3);
        }
    }

    // ADR-0019 invariant: strictly ZERO NVMe reads across r = 1 and r = 2!
    try std.testing.expectEqual(nvme_reads_after_r0, store.stats().nvme_reads);
    try std.testing.expectEqual(@as(u64, 2), scratch.loop_reuse_hits);
}
