//! Preallocated zero-allocation shared prefix `KvTile` & `GDN` snapshot cache
//! (`R8`, ADR-0004, ADR-0019).
//!
//! Indexes token sequences in fixed `block_tokens` (`512`-token) blocks using
//! a rolling 64-bit chained FNV-1a/avalanche hash paired with exact token-slice
//! verification in wired memory so hash collisions are impossible.
//!
//! - Full `512`-token prefix blocks hold reference-counted (`pin_count`)
//!   handles to shared read-only `KvTilePair`s (`.active_tile`) or
//!   `RetiredKvQ8Pool` slots (`.retired_q8`): concurrent requests bind the
//!   exact same physical `KvTilePair` / retired slot IDs (`0` duplicate KV RAM,
//!   `0` copy for full chunks).
//! - Partial tail prefixes (`1 <= tail_tokens < block_tokens`) copy only the
//!   valid tail rows into each request slot's private tail `KvTilePair` on GPU
//!   via `ami_kv_tile_cow_copy` (copy-on-write, `0` CPU `memcpy`).
//! - Hybrid `GDN` checkpoints attach a `GdnSnapshotPool` slot ID to cached
//!   prefix boundaries and save/restore `[num_gdn_layers, num_v_heads, d_k, d_v]`
//!   + `conv1d` history on GPU via `ami_gdn_clone_state`.
//! - Unreferenced leaf entries (`pin_count == 0 && child_count == 0`) are
//!   evicted in deterministic O(1) LRU order with `0` heap allocations after
//!   `init` (all metadata and token storage are carved from `WiredArena`).

const std = @import("std");
const page = @import("page.zig");
const arena_mod = @import("arena.zig");
const budget_mod = @import("budget.zig");
const amf = @import("../format/amf.zig");
const metal = @import("../bridge/metal.zig");
const attention = @import("../gpu/attention.zig");
const hybrid = @import("../gpu/hybrid.zig");
const qos = @import("../platform/qos.zig");

/// Sentinel index for empty intrusive-list links / root parent.
pub const NONE_ID: u32 = std.math.maxInt(u32);

/// Sentinel for entries that do not carry a `GDN` recurrent state snapshot.
pub const NO_SNAPSHOT: u32 = std.math.maxInt(u32);

/// Default tokens per full cached prefix block (matches `attention.DEFAULT_CHUNK_TOKENS`).
pub const DEFAULT_BLOCK_TOKENS: u32 = attention.DEFAULT_CHUNK_TOKENS;

pub const Error = arena_mod.WiredArena.AllocError || error{
    InvalidConfig,
    InvalidEntry,
    InvalidParent,
    InvalidTokenLength,
    InvalidLayerCount,
    InvalidLayerIndex,
    BufferTooSmall,
    CacheFull,
    EntryPinned,
    EntryHasChildren,
    UnbalancedUnpin,
    AlreadyRetired,
    InvalidTierOrdering,
};

pub const KvBlockStorage = enum(u8) {
    active_tile = 0,
    retired_q8 = 1,
};

/// Per-layer physical KV storage descriptor for one cached prefix block.
pub const KvBlockHandle = extern struct {
    /// For `.active_tile`: `KvTile.rgbaResourceId()` for K (`0` for `.retired_q8`).
    k_tex_id: u64 = 0,
    /// For `.active_tile`: `KvTile.rgbaResourceId()` for V (`0` for `.retired_q8`).
    v_tex_id: u64 = 0,
    /// Caller-managed pool slot ID: active tile pool index (`.active_tile`) or
    /// `RetiredKvQ8Pool` slot ID (`.retired_q8`).
    pool_slot_id: u32 = 0,
    storage: KvBlockStorage = .active_tile,
    _pad: [3]u8 = .{ 0, 0, 0 },

    pub fn activeFromTiles(k: metal.KvTile, v: metal.KvTile, active_pool_slot_id: u32) KvBlockHandle {
        return .{
            .k_tex_id = k.rgbaResourceId(),
            .v_tex_id = v.rgbaResourceId(),
            .pool_slot_id = active_pool_slot_id,
            .storage = .active_tile,
        };
    }

    pub fn retiredQ8(retired_slot_id: u32) KvBlockHandle {
        return .{
            .k_tex_id = 0,
            .v_tex_id = 0,
            .pool_slot_id = retired_slot_id,
            .storage = .retired_q8,
        };
    }

    pub fn toTilePair(self: KvBlockHandle) attention.KvTilePair {
        std.debug.assert(self.storage == .active_tile);
        return .{ .k_tex_id = self.k_tex_id, .v_tex_id = self.v_tex_id };
    }
};

comptime {
    if (@sizeOf(KvBlockHandle) != 24) @compileError("KvBlockHandle must be 24 bytes");
}

pub const PrefixCacheSpec = struct {
    /// Maximum cached blocks (full blocks + partial tail blocks combined).
    max_entries: u32,
    /// Number of tokens in one full prefix block (default `512`).
    block_tokens: u32 = DEFAULT_BLOCK_TOKENS,
    /// Number of KV-bearing attention layers stored per cached block (`>= 1`).
    num_kv_layers: u32,
};

pub const Entry = struct {
    hash: u64,
    last_access_tick: u64,
    parent_id: u32,
    first_child: u32,
    next_sibling: u32,
    prev_sibling: u32,
    lru_prev: u32,
    lru_next: u32,
    free_next: u32,
    valid_tokens: u32,
    prefix_len: u32,
    pin_count: u32,
    child_count: u32,
    gdn_snapshot_id: u32,
    in_use: bool,
    in_evict_list: bool,
    in_hash_table: bool,
    is_root_tail: bool,
    /// When `true` (e.g., ChatML intra-tile message-boundary checkpoints, ADR-0020),
    /// this unreferenced leaf is always evicted (in LRU order) before any
    /// canonical (`evict_first == false`) entry.
    evict_first: bool = false,
};

/// Metadata returned when an unreferenced entry is evicted so the caller can
/// immediately reclaim its active `KvTilePair`s, `RetiredKvQ8Pool` slots, and
/// `GdnSnapshotPool` slot without callbacks or heap allocations.
pub const EvictedEntry = struct {
    entry_id: u32,
    valid_tokens: u32,
    prefix_len: u32,
    gdn_snapshot_id: u32,
    num_kv_layers: u32,
    evict_first: bool = false,
};

pub const InsertOptions = struct {
    gdn_snapshot_id: u32 = NO_SNAPSHOT,
    /// When `true` and the cache has no free entry slots, evicts one unpinned
    /// LRU leaf first and returns it in `InsertResult.evicted`.
    evict_if_full: bool = false,
    /// Optional caller buffer (`len >= num_kv_layers`) receiving the evicted
    /// entry's per-layer `KvBlockHandle`s when eviction occurs.
    evicted_handles_out: ?[]KvBlockHandle = null,
    /// Pin the newly inserted (or already-existing) entry immediately (`pin_count += 1`).
    pin_on_insert: bool = false,
    /// Mark this entry in the lower-priority `evict_first` tier (`C15`, ADR-0020)
    /// so it is evicted before canonical full-tile (`evict_first == false`) entries.
    evict_first: bool = false,
};

pub const InsertResult = struct {
    entry_id: u32,
    already_existed: bool,
    evicted: ?EvictedEntry = null,
};

pub const MatchOptions = struct {
    /// Optional cap on matched tokens (e.g., `prompt.len - 1` when the engine
    /// must still evaluate the final prompt token to emit logits).
    max_match_tokens: ?usize = null,
    /// Allow matching a partial tail block (`1 .. block_tokens - 1` tokens)
    /// for GPU Copy-on-Write (`ami_kv_tile_cow_copy`).
    allow_partial_tail: bool = true,
    /// For hybrid `GDN + GQA` models: clamp the match to the deepest boundary
    /// that carries a valid `gdn_snapshot_id != NO_SNAPSHOT` so the recurrent
    /// state can be cloned identically via `ami_gdn_clone_state`.
    require_gdn_snapshot: bool = false,
};

pub const PrefixMatch = struct {
    /// Total matched prefix tokens (`full_block_count * block_tokens + tail_tokens`).
    matched_tokens: u32,
    /// Number of full `block_tokens` shared read-only blocks written to `full_ids_out[0..full_block_count]`.
    /// Each returned entry has its `pin_count` incremented by 1.
    full_block_count: u32,
    /// Cached entry supplying the partial tail KV (`NONE_ID` when `tail_tokens == 0`).
    /// When `!= NONE_ID`, its `pin_count` has been incremented by 1 until
    /// `releaseTailPin` or `releaseMatch` is called.
    tail_entry_id: u32,
    /// Number of valid prefix tokens to GPU CoW-copy from `tail_entry_id` (`0 .. block_tokens - 1`).
    tail_tokens: u32,
    /// `GdnSnapshotPool` slot ID valid at `matched_tokens` (`NO_SNAPSHOT` if none).
    gdn_snapshot_id: u32,
};

pub const TieredKvCounts = struct {
    num_retired_chunks: u32,
    num_active_chunks: u32,
};

/// Rolling 64-bit chained hash (FNV-1a + Murmur3/xxHash64 avalanche finalizer)
/// mixing `parent_hash`, block length, and exact token IDs. Always paired with
/// exact wired token-slice equality verification in `PrefixCache`.
pub fn hashBlock(parent_hash: u64, block_tokens: []const u32) u64 {
    var h: u64 = 0xcbf29ce484222325 ^ parent_hash;
    const prime: u64 = 0x100000001b3;
    h ^= @as(u64, block_tokens.len);
    h *%= prime;
    for (block_tokens) |tok| {
        h ^= @as(u64, tok);
        h *%= prime;
    }
    h ^= h >> 33;
    h *%= 0xff51afd7ed558ccd;
    h ^= h >> 33;
    h *%= 0xc4ceb9fe1a85ec53;
    h ^= h >> 33;
    return h;
}

/// Computes the chained 64-bit prefix hash across `tokens` in `block_tokens`
/// chunks (matching `PrefixCache`'s chained `hashBlock` tree).
pub fn hashPrefix(tokens: []const u32, block_tokens: u32) u64 {
    if (tokens.len == 0 or block_tokens == 0) return 0;
    var h: u64 = 0;
    var off: usize = 0;
    const step: usize = block_tokens;
    while (off < tokens.len) {
        const take: usize = @min(step, tokens.len - off);
        h = hashBlock(h, tokens[off .. off + take]);
        off += take;
    }
    return h;
}

/// Zero-allocation shared prefix `KvTile` & `GDN` snapshot cache (`R8`).
/// All arrays live inside a startup `WiredArena`; zero heap allocations occur
/// during `init` or on the hot path.
pub const PrefixCache = struct {
    spec: PrefixCacheSpec,
    entries: []Entry,
    kv_handles: []KvBlockHandle,
    tokens: []u32,
    hash_buckets: []u32,
    hash_next: []u32,
    bucket_mask: u32,
    free_head: u32,
    evict_head: u32, // MRU evictable canonical leaf (`evict_first == false`)
    evict_tail: u32, // LRU evictable canonical leaf (`evict_first == false`)
    evict_first_head: u32, // MRU evictable priority-victim leaf (`evict_first == true`)
    evict_first_tail: u32, // LRU evictable priority-victim leaf (`evict_first == true`)
    root_tail_head: u32, // Root-level partial tail entries (parent == NONE_ID, valid_tokens < block_tokens)
    live_count: u32,
    pinned_count: u32,
    evictable_count: u32,
    tick: u64,
    /// Number of `matchAndPin` hits resolved at an `evict_first` message-boundary checkpoint (`C15`).
    boundary_hits: u64 = 0,
    /// Number of `evict_first == true` entries evicted ahead of canonical entries (`C15`).
    evict_first_evictions: u64 = 0,

    pub fn validateSpec(spec: PrefixCacheSpec) Error!void {
        if (spec.max_entries == 0 or spec.block_tokens == 0 or spec.num_kv_layers == 0) {
            return error.InvalidConfig;
        }
        if (spec.max_entries >= NONE_ID) {
            return error.InvalidConfig;
        }
    }

    fn bucketCountFor(max_entries: u32) Error!usize {
        const doubled = std.math.mul(usize, max_entries, 2) catch return error.Overflow;
        return std.math.ceilPowerOfTwo(usize, @max(doubled, 8)) catch return error.Overflow;
    }

    /// Exact `16 KiB`-aligned `WiredArena` bytes needed to back `PrefixCache`.
    pub fn requiredBytes(spec: PrefixCacheSpec) Error!usize {
        try validateSpec(spec);
        const n: usize = spec.max_entries;
        const buckets = try bucketCountFor(spec.max_entries);

        const entries_bytes = amf.align64(std.math.mul(usize, n, @sizeOf(Entry)) catch return error.Overflow);
        const handle_count = std.math.mul(usize, n, spec.num_kv_layers) catch return error.Overflow;
        const handles_bytes = amf.align64(std.math.mul(usize, handle_count, @sizeOf(KvBlockHandle)) catch return error.Overflow);
        const token_count = std.math.mul(usize, n, spec.block_tokens) catch return error.Overflow;
        const tokens_bytes = amf.align64(std.math.mul(usize, token_count, @sizeOf(u32)) catch return error.Overflow);
        const buckets_bytes = amf.align64(std.math.mul(usize, buckets, @sizeOf(u32)) catch return error.Overflow);
        const next_bytes = amf.align64(std.math.mul(usize, n, @sizeOf(u32)) catch return error.Overflow);

        var total = std.math.add(usize, entries_bytes, handles_bytes) catch return error.Overflow;
        total = std.math.add(usize, total, tokens_bytes) catch return error.Overflow;
        total = std.math.add(usize, total, buckets_bytes) catch return error.Overflow;
        total = std.math.add(usize, total, next_bytes) catch return error.Overflow;
        return page.alignUp(total) catch return error.Overflow;
    }

    pub fn init(arena: *arena_mod.WiredArena, spec: PrefixCacheSpec) Error!PrefixCache {
        try validateSpec(spec);
        const n: usize = spec.max_entries;
        const buckets = try bucketCountFor(spec.max_entries);

        const entries_raw = try arena.allocAligned(amf.align64(n * @sizeOf(Entry)), 64);
        const entries: []Entry = @alignCast(std.mem.bytesAsSlice(Entry, entries_raw[0 .. n * @sizeOf(Entry)]));

        const handle_count = n * spec.num_kv_layers;
        const handles_raw = try arena.allocAligned(amf.align64(handle_count * @sizeOf(KvBlockHandle)), 64);
        const kv_handles: []KvBlockHandle = @alignCast(std.mem.bytesAsSlice(
            KvBlockHandle,
            handles_raw[0 .. handle_count * @sizeOf(KvBlockHandle)],
        ));

        const token_count = n * spec.block_tokens;
        const tokens_raw = try arena.allocAligned(amf.align64(token_count * @sizeOf(u32)), 64);
        const tokens: []u32 = @alignCast(std.mem.bytesAsSlice(u32, tokens_raw[0 .. token_count * @sizeOf(u32)]));

        const buckets_raw = try arena.allocAligned(amf.align64(buckets * @sizeOf(u32)), 64);
        const hash_buckets: []u32 = @alignCast(std.mem.bytesAsSlice(u32, buckets_raw[0 .. buckets * @sizeOf(u32)]));

        const next_raw = try arena.allocAligned(amf.align64(n * @sizeOf(u32)), 64);
        const hash_next: []u32 = @alignCast(std.mem.bytesAsSlice(u32, next_raw[0 .. n * @sizeOf(u32)]));

        var cache: PrefixCache = .{
            .spec = spec,
            .entries = entries,
            .kv_handles = kv_handles,
            .tokens = tokens,
            .hash_buckets = hash_buckets,
            .hash_next = hash_next,
            .bucket_mask = @intCast(buckets - 1),
            .free_head = 0,
            .evict_head = NONE_ID,
            .evict_tail = NONE_ID,
            .evict_first_head = NONE_ID,
            .evict_first_tail = NONE_ID,
            .root_tail_head = NONE_ID,
            .live_count = 0,
            .pinned_count = 0,
            .evictable_count = 0,
            .tick = 0,
            .boundary_hits = 0,
            .evict_first_evictions = 0,
        };
        cache.reset();
        return cache;
    }

    /// Resets all entries and indices without releasing or reallocating memory.
    pub fn reset(self: *PrefixCache) void {
        const n = self.spec.max_entries;
        for (0..n) |i| {
            self.entries[i] = .{
                .hash = 0,
                .last_access_tick = 0,
                .parent_id = NONE_ID,
                .first_child = NONE_ID,
                .next_sibling = NONE_ID,
                .prev_sibling = NONE_ID,
                .lru_prev = NONE_ID,
                .lru_next = NONE_ID,
                .free_next = if (i + 1 < n) @as(u32, @intCast(i + 1)) else NONE_ID,
                .valid_tokens = 0,
                .prefix_len = 0,
                .pin_count = 0,
                .child_count = 0,
                .gdn_snapshot_id = NO_SNAPSHOT,
                .in_use = false,
                .in_evict_list = false,
                .in_hash_table = false,
                .is_root_tail = false,
                .evict_first = false,
            };
            self.hash_next[i] = NONE_ID;
        }
        @memset(self.hash_buckets, NONE_ID);
        @memset(self.kv_handles, .{});
        @memset(self.tokens, 0);
        self.free_head = 0;
        self.evict_head = NONE_ID;
        self.evict_tail = NONE_ID;
        self.evict_first_head = NONE_ID;
        self.evict_first_tail = NONE_ID;
        self.root_tail_head = NONE_ID;
        self.live_count = 0;
        self.pinned_count = 0;
        self.evictable_count = 0;
        self.tick = 0;
        self.boundary_hits = 0;
        self.evict_first_evictions = 0;
    }

    /// Returns the stored token slice for live entry `entry_id`.
    pub fn entryTokens(self: *const PrefixCache, entry_id: u32) Error![]const u32 {
        if (entry_id >= self.spec.max_entries or !self.entries[entry_id].in_use) {
            return error.InvalidEntry;
        }
        const start = @as(usize, entry_id) * self.spec.block_tokens;
        return self.tokens[start .. start + self.entries[entry_id].valid_tokens];
    }

    /// Returns the per-layer `KvBlockHandle` slice (`len == num_kv_layers`) for live entry `entry_id`.
    pub fn entryHandles(self: *const PrefixCache, entry_id: u32) Error![]const KvBlockHandle {
        if (entry_id >= self.spec.max_entries or !self.entries[entry_id].in_use) {
            return error.InvalidEntry;
        }
        const start = @as(usize, entry_id) * self.spec.num_kv_layers;
        return self.kv_handles[start .. start + self.spec.num_kv_layers];
    }

    pub fn entryHandle(self: *const PrefixCache, entry_id: u32, layer_idx: u32) Error!KvBlockHandle {
        if (layer_idx >= self.spec.num_kv_layers) return error.InvalidLayerIndex;
        const slice = try self.entryHandles(entry_id);
        return slice[layer_idx];
    }

    pub fn entryInfo(self: *const PrefixCache, entry_id: u32) Error!Entry {
        if (entry_id >= self.spec.max_entries or !self.entries[entry_id].in_use) {
            return error.InvalidEntry;
        }
        return self.entries[entry_id];
    }

    /// Inserts a full `block_tokens` (`512`-token) prefix block chained to
    /// `parent_id` (`NONE_ID` for block 0). If an identical `(parent_id, block_tokens)`
    /// entry already exists (verified by exact token equality), returns its ID
    /// with `already_existed = true` without duplicating storage.
    pub fn insertFullBlock(
        self: *PrefixCache,
        parent_id: u32,
        block_tokens: []const u32,
        layer_handles: []const KvBlockHandle,
        opts: InsertOptions,
    ) Error!InsertResult {
        if (block_tokens.len != self.spec.block_tokens) return error.InvalidTokenLength;
        if (layer_handles.len != self.spec.num_kv_layers) return error.InvalidLayerCount;

        var parent_hash: u64 = 0;
        var parent_prefix_len: u32 = 0;
        if (parent_id != NONE_ID) {
            if (parent_id >= self.spec.max_entries or !self.entries[parent_id].in_use) {
                return error.InvalidParent;
            }
            if (self.entries[parent_id].valid_tokens != self.spec.block_tokens) {
                return error.InvalidParent;
            }
            parent_hash = self.entries[parent_id].hash;
            parent_prefix_len = self.entries[parent_id].prefix_len;
        }

        const h = hashBlock(parent_hash, block_tokens);
        if (self.findFullBlockByHash(parent_id, h, block_tokens)) |existing| {
            if (self.entries[existing].evict_first and !opts.evict_first) {
                self.promoteToCanonical(existing);
            }
            self.touchEntry(existing);
            if (opts.pin_on_insert) {
                self.pinUnchecked(existing);
            }
            return .{ .entry_id = existing, .already_existed = true, .evicted = null };
        }

        var parent_temp_pinned = false;
        if (parent_id != NONE_ID) {
            self.pinUnchecked(parent_id);
            parent_temp_pinned = true;
        }
        errdefer if (parent_temp_pinned) self.unpinRestoringEvictNoTick(parent_id);

        const alloc_res = try self.allocateEntrySlot(opts.evict_if_full, opts.evicted_handles_out);
        const id = alloc_res.entry_id;
        if (parent_id != NONE_ID and id == parent_id) {
            self.returnFreeSlot(id);
            return error.InvalidParent;
        }

        self.tick += 1;
        const prefix_len = std.math.add(u32, parent_prefix_len, self.spec.block_tokens) catch {
            self.returnFreeSlot(id);
            return error.Overflow;
        };

        self.entries[id] = .{
            .hash = h,
            .last_access_tick = self.tick,
            .parent_id = parent_id,
            .first_child = NONE_ID,
            .next_sibling = NONE_ID,
            .prev_sibling = NONE_ID,
            .lru_prev = NONE_ID,
            .lru_next = NONE_ID,
            .free_next = NONE_ID,
            .valid_tokens = self.spec.block_tokens,
            .prefix_len = prefix_len,
            .pin_count = 0,
            .child_count = 0,
            .gdn_snapshot_id = opts.gdn_snapshot_id,
            .in_use = true,
            .in_evict_list = false,
            .in_hash_table = false,
            .is_root_tail = false,
            .evict_first = opts.evict_first,
        };

        const tok_start = @as(usize, id) * self.spec.block_tokens;
        @memcpy(self.tokens[tok_start .. tok_start + self.spec.block_tokens], block_tokens);
        const h_start = @as(usize, id) * self.spec.num_kv_layers;
        @memcpy(self.kv_handles[h_start .. h_start + self.spec.num_kv_layers], layer_handles);

        self.live_count += 1;
        self.hashInsert(id, h);
        if (parent_id != NONE_ID) {
            self.linkChild(parent_id, id);
            if (parent_temp_pinned) {
                self.unpinRestoringEvictNoTick(parent_id);
                parent_temp_pinned = false;
            }
        }
        if (opts.pin_on_insert) {
            self.pinUnchecked(id);
        } else {
            self.evictPushHead(id);
        }

        return .{
            .entry_id = id,
            .already_existed = false,
            .evicted = alloc_res.evicted,
        };
    }

    /// Inserts a partial tail prefix block (`1 <= tail_tokens.len < block_tokens`)
    /// under `parent_id` (`NONE_ID` for prompts shorter than one full block).
    /// Used as the read-only source for GPU Copy-on-Write tail cloning
    /// (`ami_kv_tile_cow_copy`) and optional `GDN` checkpoint restoration.
    pub fn insertTailBlock(
        self: *PrefixCache,
        parent_id: u32,
        tail_tokens: []const u32,
        layer_handles: []const KvBlockHandle,
        opts: InsertOptions,
    ) Error!InsertResult {
        if (tail_tokens.len == 0 or tail_tokens.len >= self.spec.block_tokens) {
            return error.InvalidTokenLength;
        }
        if (layer_handles.len != self.spec.num_kv_layers) return error.InvalidLayerCount;

        var parent_hash: u64 = 0;
        var parent_prefix_len: u32 = 0;
        if (parent_id != NONE_ID) {
            if (parent_id >= self.spec.max_entries or !self.entries[parent_id].in_use) {
                return error.InvalidParent;
            }
            if (self.entries[parent_id].valid_tokens != self.spec.block_tokens) {
                return error.InvalidParent;
            }
            parent_hash = self.entries[parent_id].hash;
            parent_prefix_len = self.entries[parent_id].prefix_len;
        }

        // Check existing children under `parent_id` for an exact tail match.
        const list_head = if (parent_id != NONE_ID) self.entries[parent_id].first_child else self.root_tail_head;
        var cur = list_head;
        while (cur != NONE_ID) : (cur = self.entries[cur].next_sibling) {
            const e = &self.entries[cur];
            if (e.valid_tokens == tail_tokens.len) {
                const start = @as(usize, cur) * self.spec.block_tokens;
                if (std.mem.eql(u32, self.tokens[start .. start + e.valid_tokens], tail_tokens)) {
                    if (e.evict_first and !opts.evict_first) {
                        self.promoteToCanonical(cur);
                    }
                    self.touchEntry(cur);
                    if (opts.pin_on_insert) {
                        self.pinUnchecked(cur);
                    }
                    return .{ .entry_id = cur, .already_existed = true, .evicted = null };
                }
            }
        }

        var parent_temp_pinned = false;
        if (parent_id != NONE_ID) {
            self.pinUnchecked(parent_id);
            parent_temp_pinned = true;
        }
        errdefer if (parent_temp_pinned) self.unpinRestoringEvictNoTick(parent_id);

        const alloc_res = try self.allocateEntrySlot(opts.evict_if_full, opts.evicted_handles_out);
        const id = alloc_res.entry_id;
        if (parent_id != NONE_ID and id == parent_id) {
            self.returnFreeSlot(id);
            return error.InvalidParent;
        }

        self.tick += 1;
        const valid_u32: u32 = @intCast(tail_tokens.len);
        const prefix_len = std.math.add(u32, parent_prefix_len, valid_u32) catch {
            self.returnFreeSlot(id);
            return error.Overflow;
        };
        const h = hashBlock(parent_hash, tail_tokens);

        self.entries[id] = .{
            .hash = h,
            .last_access_tick = self.tick,
            .parent_id = parent_id,
            .first_child = NONE_ID,
            .next_sibling = NONE_ID,
            .prev_sibling = NONE_ID,
            .lru_prev = NONE_ID,
            .lru_next = NONE_ID,
            .free_next = NONE_ID,
            .valid_tokens = valid_u32,
            .prefix_len = prefix_len,
            .pin_count = 0,
            .child_count = 0,
            .gdn_snapshot_id = opts.gdn_snapshot_id,
            .in_use = true,
            .in_evict_list = false,
            .in_hash_table = false,
            .is_root_tail = (parent_id == NONE_ID),
            .evict_first = opts.evict_first,
        };

        const tok_start = @as(usize, id) * self.spec.block_tokens;
        @memcpy(self.tokens[tok_start .. tok_start + tail_tokens.len], tail_tokens);
        const h_start = @as(usize, id) * self.spec.num_kv_layers;
        @memcpy(self.kv_handles[h_start .. h_start + self.spec.num_kv_layers], layer_handles);

        self.live_count += 1;
        if (parent_id != NONE_ID) {
            self.linkChild(parent_id, id);
            if (parent_temp_pinned) {
                self.unpinRestoringEvictNoTick(parent_id);
                parent_temp_pinned = false;
            }
        } else {
            self.linkRootTail(id);
        }
        if (opts.pin_on_insert) {
            self.pinUnchecked(id);
        } else {
            self.evictPushHead(id);
        }

        return .{
            .entry_id = id,
            .already_existed = false,
            .evicted = alloc_res.evicted,
        };
    }

    /// Inserts a ChatML message-boundary checkpoint (`C15`, ADR-0020) for an
    /// arbitrary boundary block (`1 <= boundary_block_tokens.len <= block_tokens`)
    /// chained to `parent_id`:
    /// - Full-tile boundaries (`len == block_tokens`) insert as a full block
    ///   (retaining `opts.evict_first`).
    /// - Intra-tile boundaries (`1 <= len < block_tokens`) insert as a CoW tail
    ///   block marked `evict_first = true` so canonical full-tile blocks retain
    ///   higher eviction priority.
    pub fn insertBoundaryCheckpoint(
        self: *PrefixCache,
        parent_id: u32,
        boundary_block_tokens: []const u32,
        layer_handles: []const KvBlockHandle,
        opts: InsertOptions,
    ) Error!InsertResult {
        if (boundary_block_tokens.len == 0 or boundary_block_tokens.len > self.spec.block_tokens) {
            return error.InvalidTokenLength;
        }
        if (boundary_block_tokens.len == self.spec.block_tokens) {
            return self.insertFullBlock(parent_id, boundary_block_tokens, layer_handles, opts);
        }
        var tail_opts = opts;
        tail_opts.evict_first = true;
        return self.insertTailBlock(parent_id, boundary_block_tokens, layer_handles, tail_opts);
    }

    /// Matches the longest cached prefix of `prompt_tokens` with exact token
    /// verification, pinning every returned full block (`full_ids_out[0..full_block_count]`)
    /// and the partial tail source (`tail_entry_id`, if any) in O(blocks + branching).
    /// Zero heap allocations.
    pub fn matchAndPin(
        self: *PrefixCache,
        prompt_tokens: []const u32,
        full_ids_out: []u32,
        opts: MatchOptions,
    ) Error!PrefixMatch {
        const limit_len = if (opts.max_match_tokens) |m| @min(prompt_tokens.len, m) else prompt_tokens.len;
        if (limit_len == 0) {
            return .{
                .matched_tokens = 0,
                .full_block_count = 0,
                .tail_entry_id = NONE_ID,
                .tail_tokens = 0,
                .gdn_snapshot_id = NO_SNAPSHOT,
            };
        }

        const bs: usize = self.spec.block_tokens;
        var matched_full: u32 = 0;
        var parent_id: u32 = NONE_ID;
        var parent_hash: u64 = 0;
        var offset: usize = 0;

        while (offset + bs <= limit_len) {
            if (matched_full >= full_ids_out.len) {
                break;
            }
            const blk_slice = prompt_tokens[offset .. offset + bs];
            const h = hashBlock(parent_hash, blk_slice);
            const found = self.findFullBlockByHash(parent_id, h, blk_slice) orelse break;
            full_ids_out[matched_full] = found;
            matched_full += 1;
            parent_id = found;
            parent_hash = h;
            offset += bs;
        }

        var tail_entry: u32 = NONE_ID;
        var tail_len: u32 = 0;
        var tail_snap: u32 = NO_SNAPSHOT;
        var tail_exact_boundary: bool = false;

        const rem_len = limit_len - offset;
        if (opts.allow_partial_tail and rem_len > 0) {
            const rem_slice = prompt_tokens[offset..limit_len];
            const list_head = if (parent_id != NONE_ID) self.entries[parent_id].first_child else self.root_tail_head;
            var cur = list_head;
            while (cur != NONE_ID) : (cur = self.entries[cur].next_sibling) {
                const e = &self.entries[cur];
                // Only entries whose attention layers are ALL `.active_tile` can be CoW-copied via `ami_kv_tile_cow_copy`.
                const h_base = @as(usize, cur) * self.spec.num_kv_layers;
                var all_active = true;
                for (self.kv_handles[h_base .. h_base + self.spec.num_kv_layers]) |h| {
                    if (h.storage != .active_tile) {
                        all_active = false;
                        break;
                    }
                }
                if (!all_active) continue;
                const tok_start = @as(usize, cur) * bs;
                const cand_tokens = self.tokens[tok_start .. tok_start + e.valid_tokens];
                const max_cmp = @min(cand_tokens.len, rem_slice.len);
                var lcp: usize = 0;
                while (lcp < max_cmp and cand_tokens[lcp] == rem_slice[lcp]) : (lcp += 1) {}
                if (lcp == 0) continue;
                // A full block match would already have been caught above; cap tail strictly < block_tokens.
                if (lcp >= bs) lcp = bs - 1;
                // R8 invariant: exact token-slice verification with zero hash-only hits.
                std.debug.assert(std.mem.eql(u32, cand_tokens[0..lcp], rem_slice[0..lcp]));

                if (opts.require_gdn_snapshot) {
                    // For GDN models, a recurrent snapshot is exact only when the full
                    // checkpoint boundary (`lcp == e.valid_tokens`) matches.
                    if (lcp == e.valid_tokens and e.gdn_snapshot_id != NO_SNAPSHOT) {
                        if (@as(u32, @intCast(lcp)) > tail_len) {
                            tail_len = @intCast(lcp);
                            tail_entry = cur;
                            tail_snap = e.gdn_snapshot_id;
                            tail_exact_boundary = true;
                        }
                    }
                } else {
                    const cand_exact = (lcp == e.valid_tokens);
                    const cand_snap = if (cand_exact) e.gdn_snapshot_id else NO_SNAPSHOT;
                    const lcp_u32: u32 = @intCast(lcp);
                    const better_len = lcp_u32 > tail_len;
                    const better_snap = (lcp_u32 == tail_len and tail_snap == NO_SNAPSHOT and cand_snap != NO_SNAPSHOT);
                    const better_exact = (lcp_u32 == tail_len and (tail_snap == NO_SNAPSHOT) == (cand_snap == NO_SNAPSHOT) and !tail_exact_boundary and cand_exact);
                    if (better_len or better_snap or better_exact) {
                        tail_len = lcp_u32;
                        tail_entry = cur;
                        tail_snap = cand_snap;
                        tail_exact_boundary = cand_exact;
                    }
                }
            }
        }

        // Resolve GDN snapshot & clamp if `require_gdn_snapshot` is active.
        var final_snap: u32 = NO_SNAPSHOT;
        if (tail_entry != NONE_ID) {
            final_snap = tail_snap;
        } else if (matched_full > 0) {
            final_snap = self.entries[full_ids_out[matched_full - 1]].gdn_snapshot_id;
        }

        if (opts.require_gdn_snapshot and final_snap == NO_SNAPSHOT) {
            tail_entry = NONE_ID;
            tail_len = 0;
            while (matched_full > 0) {
                const top_id = full_ids_out[matched_full - 1];
                if (self.entries[top_id].gdn_snapshot_id != NO_SNAPSHOT) {
                    final_snap = self.entries[top_id].gdn_snapshot_id;
                    break;
                }
                matched_full -= 1;
            }
        }

        // Track message-boundary checkpoint hits (`evict_first == true`).
        if (tail_entry != NONE_ID and self.entries[tail_entry].evict_first) {
            self.boundary_hits += 1;
        } else if (tail_entry == NONE_ID and matched_full > 0 and self.entries[full_ids_out[matched_full - 1]].evict_first) {
            self.boundary_hits += 1;
        }

        // Pin all matched full blocks + tail source and bump LRU recency.
        for (full_ids_out[0..matched_full]) |id| {
            self.touchEntry(id);
            self.pinUnchecked(id);
        }
        if (tail_entry != NONE_ID) {
            self.touchEntry(tail_entry);
            self.pinUnchecked(tail_entry);
        }

        return .{
            .matched_tokens = matched_full * self.spec.block_tokens + tail_len,
            .full_block_count = matched_full,
            .tail_entry_id = tail_entry,
            .tail_tokens = tail_len,
            .gdn_snapshot_id = final_snap,
        };
    }

    pub fn pin(self: *PrefixCache, entry_id: u32) Error!void {
        if (entry_id >= self.spec.max_entries or !self.entries[entry_id].in_use) {
            return error.InvalidEntry;
        }
        self.touchEntry(entry_id);
        self.pinUnchecked(entry_id);
    }

    pub fn unpin(self: *PrefixCache, entry_id: u32) Error!void {
        if (entry_id >= self.spec.max_entries or !self.entries[entry_id].in_use) {
            return error.InvalidEntry;
        }
        const e = &self.entries[entry_id];
        if (e.pin_count == 0) return error.UnbalancedUnpin;
        e.pin_count -= 1;
        if (e.pin_count == 0) {
            self.pinned_count -= 1;
            if (e.child_count == 0 and !e.in_evict_list) {
                // Just finished active use -> push to MRU of evictable list.
                self.tick += 1;
                e.last_access_tick = self.tick;
                self.evictPushHead(entry_id);
            }
        }
    }

    /// Releases the pin on a partial tail entry once the GPU `ami_kv_tile_cow_copy`
    /// has finished and the request owns its private CoW tail tile.
    pub fn releaseTailPin(self: *PrefixCache, match: *PrefixMatch) Error!void {
        if (match.tail_entry_id == NONE_ID) return;
        try self.unpin(match.tail_entry_id);
        match.tail_entry_id = NONE_ID;
    }

    /// Releases all pins acquired by `matchAndPin` (full blocks + tail pin if still held).
    pub fn releaseMatch(self: *PrefixCache, match: *PrefixMatch, full_ids: []const u32) Error!void {
        if (full_ids.len < match.full_block_count) return error.BufferTooSmall;
        if (match.tail_entry_id != NONE_ID) {
            try self.unpin(match.tail_entry_id);
            match.tail_entry_id = NONE_ID;
        }
        var i: usize = match.full_block_count;
        while (i > 0) {
            i -= 1;
            try self.unpin(full_ids[i]);
        }
        match.full_block_count = 0;
    }

    /// Upgrades a cached full block's per-layer handle from an active IOSurface
    /// `KvTilePair` to a `RetiredKvQ8Pool` slot (`R5 + R8`), returning the
    /// previous active `KvBlockHandle` so the caller can return the IOSurface
    /// pair to the free active tile pool.
    pub fn retireEntryLayerToQ8(
        self: *PrefixCache,
        entry_id: u32,
        layer_idx: u32,
        retired_slot_id: u32,
    ) Error!KvBlockHandle {
        if (entry_id >= self.spec.max_entries or !self.entries[entry_id].in_use) {
            return error.InvalidEntry;
        }
        if (self.entries[entry_id].valid_tokens != self.spec.block_tokens) {
            return error.InvalidTokenLength;
        }
        if (layer_idx >= self.spec.num_kv_layers) return error.InvalidLayerIndex;
        const idx = @as(usize, entry_id) * self.spec.num_kv_layers + layer_idx;
        const old = self.kv_handles[idx];
        if (old.storage == .retired_q8) return error.AlreadyRetired;
        self.kv_handles[idx] = KvBlockHandle.retiredQ8(retired_slot_id);
        return old;
    }

    /// Populates a request slot's retired `q8` chunk IDs table and active
    /// `KvTilePair` table for `layer_idx` directly from `full_ids` with `0` copy
    /// and `0` heap allocation. Enforces that retired chunks precede active chunks.
    pub fn populateLayerKvTables(
        self: *const PrefixCache,
        full_ids: []const u32,
        layer_idx: u32,
        retired_ids_out: []u32,
        active_tiles_out: []attention.KvTilePair,
    ) Error!TieredKvCounts {
        if (layer_idx >= self.spec.num_kv_layers) return error.InvalidLayerIndex;
        var n_retired: u32 = 0;
        var n_active: u32 = 0;
        for (full_ids) |id| {
            const h = try self.entryHandle(id, layer_idx);
            switch (h.storage) {
                .retired_q8 => {
                    if (n_active > 0) return error.InvalidTierOrdering;
                    if (n_retired >= retired_ids_out.len) return error.BufferTooSmall;
                    retired_ids_out[n_retired] = h.pool_slot_id;
                    n_retired += 1;
                },
                .active_tile => {
                    if (n_active >= active_tiles_out.len) return error.BufferTooSmall;
                    active_tiles_out[n_active] = h.toTilePair();
                    n_active += 1;
                },
            }
        }
        return .{
            .num_retired_chunks = n_retired,
            .num_active_chunks = n_active,
        };
    }

    /// Evicts the least-recently-used unreferenced leaf entry (`pin_count == 0`
    /// and `child_count == 0`) in strict O(1) time and `0` heap allocations.
    /// Always drains `evict_first == true` leaves (LRU within tier) before any
    /// canonical (`evict_first == false`) entry (`C15`, ADR-0020).
    /// Copies the evicted entry's per-layer `KvBlockHandle`s into
    /// `evicted_handles_out` when non-null. Returns `error.CacheFull` if all
    /// cached entries are pinned (or have pinned descendants).
    pub fn evictLru(self: *PrefixCache, evicted_handles_out: ?[]KvBlockHandle) Error!EvictedEntry {
        const victim = if (self.evict_first_tail != NONE_ID) self.evict_first_tail else self.evict_tail;
        if (victim == NONE_ID) return error.CacheFull;
        if (evicted_handles_out) |out_buf| {
            if (out_buf.len < self.spec.num_kv_layers) return error.BufferTooSmall;
            const h_start = @as(usize, victim) * self.spec.num_kv_layers;
            @memcpy(out_buf[0..self.spec.num_kv_layers], self.kv_handles[h_start .. h_start + self.spec.num_kv_layers]);
        }

        const e = self.entries[victim];
        std.debug.assert(e.in_use and e.pin_count == 0 and e.child_count == 0);

        self.evictRemove(victim);
        if (e.evict_first) {
            self.evict_first_evictions += 1;
        }
        if (e.in_hash_table) {
            self.hashRemove(victim, e.hash);
        }
        if (e.parent_id != NONE_ID) {
            self.unlinkChild(e.parent_id, victim);
        } else if (e.is_root_tail) {
            self.unlinkRootTail(victim);
        }

        const info: EvictedEntry = .{
            .entry_id = victim,
            .valid_tokens = e.valid_tokens,
            .prefix_len = e.prefix_len,
            .gdn_snapshot_id = e.gdn_snapshot_id,
            .num_kv_layers = self.spec.num_kv_layers,
            .evict_first = e.evict_first,
        };

        self.entries[victim].in_use = false;
        self.entries[victim].in_hash_table = false;
        self.entries[victim].is_root_tail = false;
        self.entries[victim].evict_first = false;
        self.entries[victim].parent_id = NONE_ID;
        self.entries[victim].gdn_snapshot_id = NO_SNAPSHOT;
        self.entries[victim].valid_tokens = 0;
        self.entries[victim].prefix_len = 0;
        self.live_count -= 1;
        self.returnFreeSlot(victim);
        return info;
    }

    pub fn liveEntries(self: *const PrefixCache) u32 {
        return self.live_count;
    }

    pub fn freeEntries(self: *const PrefixCache) u32 {
        return self.spec.max_entries - self.live_count;
    }

    pub fn pinnedEntries(self: *const PrefixCache) u32 {
        return self.pinned_count;
    }

    pub fn evictableEntries(self: *const PrefixCache) u32 {
        return self.evictable_count;
    }

    // -----------------------------------------------------------------------
    // GPU Copy-on-Write Tail & GDN Snapshot Helpers (`R8`)
    // -----------------------------------------------------------------------

    /// Schedules `ami_kv_tile_cow_copy` to copy `valid_tokens` (`1..block_tokens`)
    /// from a cached tail `KvTilePair` into a request slot's private tail
    /// `KvTilePair` on GPU (`0` CPU `memcpy`, `0` heap allocs).
    pub fn encodeTailCowCopy(
        attn_pipes: attention.AttentionPipelines,
        frame: *metal.Compute.Frame,
        src_k: metal.KvTile,
        src_v: metal.KvTile,
        dst_k: metal.KvTile,
        dst_v: metal.KvTile,
        num_kv_heads: u32,
        head_dim: u32,
        chunk_tokens: u32,
        valid_tokens: u32,
        insert_barrier_before: bool,
    ) metal.Error!void {
        return attn_pipes.encodeKvTileCowCopy(frame, .{
            .src_k = src_k,
            .src_v = src_v,
            .dst_k = dst_k,
            .dst_v = dst_v,
            .num_kv_heads = num_kv_heads,
            .head_dim = head_dim,
            .chunk_tokens = chunk_tokens,
            .valid_tokens = valid_tokens,
        }, insert_barrier_before);
    }

    /// Schedules `ami_gdn_clone_state` to snapshot a slot's committed `GdnStateArena`
    /// (plus optional `conv1d` history) into `snapshot_pool[snapshot_slot]` on GPU.
    pub fn encodeSaveGdnSnapshot(
        hybrid_pipes: hybrid.HybridPipelines,
        frame: *metal.Compute.Frame,
        state_arena: *const hybrid.GdnStateArena,
        state_buf: metal.Buffer,
        conv_src: ?metal.BufferBinding,
        conv_src_layer_stride_f16: u32,
        snapshot_pool: *const hybrid.GdnSnapshotPool,
        snapshot_buf: metal.Buffer,
        snapshot_slot: u32,
        insert_barrier_before: bool,
    ) (hybrid.EncodeError || hybrid.GdnSnapshotPoolError)!void {
        const dst_state = try snapshot_pool.stateBinding(snapshot_buf, snapshot_slot);
        const dst_conv = try snapshot_pool.convBinding(snapshot_buf, snapshot_slot);
        try hybrid_pipes.encodeGdnCloneState(frame, .{
            .src_state = try state_arena.committedBinding(state_buf, 0),
            .dst_state = dst_state,
            .src_conv = conv_src,
            .dst_conv = dst_conv,
            .num_gdn_layers = state_arena.spec.num_gdn_layers,
            .state_f32_per_layer = snapshot_pool.layer_state_elems,
            .src_state_layer_stride_f32 = @intCast(state_arena.layer_stride_bytes / @sizeOf(f32)),
            .dst_state_layer_stride_f32 = snapshot_pool.stateLayerStrideF32(),
            .conv_f16_per_layer = snapshot_pool.spec.conv_f16_per_layer,
            .src_conv_layer_stride_f16 = conv_src_layer_stride_f16,
            .dst_conv_layer_stride_f16 = snapshot_pool.convLayerStrideF16(),
        }, insert_barrier_before);
    }

    /// Schedules `ami_gdn_clone_state` to restore `snapshot_pool[snapshot_slot]`
    /// into a new request slot's committed `GdnStateArena` (plus optional `conv1d`
    /// history) on GPU (`0` CPU `memcpy`, `0` heap allocs).
    pub fn encodeRestoreGdnSnapshot(
        hybrid_pipes: hybrid.HybridPipelines,
        frame: *metal.Compute.Frame,
        snapshot_pool: *const hybrid.GdnSnapshotPool,
        snapshot_buf: metal.Buffer,
        snapshot_slot: u32,
        state_arena: *const hybrid.GdnStateArena,
        state_buf: metal.Buffer,
        conv_dst: ?metal.BufferBinding,
        conv_dst_layer_stride_f16: u32,
        insert_barrier_before: bool,
    ) (hybrid.EncodeError || hybrid.GdnSnapshotPoolError)!void {
        const src_state = try snapshot_pool.stateBinding(snapshot_buf, snapshot_slot);
        const src_conv = try snapshot_pool.convBinding(snapshot_buf, snapshot_slot);
        try hybrid_pipes.encodeGdnCloneState(frame, .{
            .src_state = src_state,
            .dst_state = try state_arena.committedBinding(state_buf, 0),
            .src_conv = src_conv,
            .dst_conv = conv_dst,
            .num_gdn_layers = state_arena.spec.num_gdn_layers,
            .state_f32_per_layer = snapshot_pool.layer_state_elems,
            .src_state_layer_stride_f32 = snapshot_pool.stateLayerStrideF32(),
            .dst_state_layer_stride_f32 = @intCast(state_arena.layer_stride_bytes / @sizeOf(f32)),
            .conv_f16_per_layer = snapshot_pool.spec.conv_f16_per_layer,
            .src_conv_layer_stride_f16 = snapshot_pool.convLayerStrideF16(),
            .dst_conv_layer_stride_f16 = conv_dst_layer_stride_f16,
        }, insert_barrier_before);
    }

    // -----------------------------------------------------------------------
    // Internal Intrusive Hash / Tree / O(1) LRU Operations
    // -----------------------------------------------------------------------

    fn allocateEntrySlot(
        self: *PrefixCache,
        evict_if_full: bool,
        evicted_handles_out: ?[]KvBlockHandle,
    ) Error!struct { entry_id: u32, evicted: ?EvictedEntry } {
        var evicted: ?EvictedEntry = null;
        if (self.free_head == NONE_ID) {
            if (!evict_if_full) return error.CacheFull;
            evicted = try self.evictLru(evicted_handles_out);
        }
        const id = self.free_head;
        std.debug.assert(id != NONE_ID);
        self.free_head = self.entries[id].free_next;
        self.entries[id].free_next = NONE_ID;
        return .{ .entry_id = id, .evicted = evicted };
    }

    fn returnFreeSlot(self: *PrefixCache, id: u32) void {
        self.entries[id].free_next = self.free_head;
        self.free_head = id;
    }

    fn findFullBlockByHash(self: *const PrefixCache, parent_id: u32, h: u64, block_tokens: []const u32) ?u32 {
        const bucket: usize = @intCast(h & self.bucket_mask);
        var cur = self.hash_buckets[bucket];
        while (cur != NONE_ID) : (cur = self.hash_next[cur]) {
            const e = &self.entries[cur];
            if (e.hash == h and e.parent_id == parent_id and e.valid_tokens == self.spec.block_tokens) {
                const start = @as(usize, cur) * self.spec.block_tokens;
                if (std.mem.eql(u32, self.tokens[start .. start + self.spec.block_tokens], block_tokens)) {
                    return cur;
                }
            }
        }
        return null;
    }

    fn hashInsert(self: *PrefixCache, id: u32, h: u64) void {
        const bucket: usize = @intCast(h & self.bucket_mask);
        self.hash_next[id] = self.hash_buckets[bucket];
        self.hash_buckets[bucket] = id;
        self.entries[id].in_hash_table = true;
    }

    fn hashRemove(self: *PrefixCache, id: u32, h: u64) void {
        const bucket: usize = @intCast(h & self.bucket_mask);
        var prev: u32 = NONE_ID;
        var cur = self.hash_buckets[bucket];
        while (cur != NONE_ID) {
            if (cur == id) {
                if (prev == NONE_ID) {
                    self.hash_buckets[bucket] = self.hash_next[cur];
                } else {
                    self.hash_next[prev] = self.hash_next[cur];
                }
                self.hash_next[id] = NONE_ID;
                self.entries[id].in_hash_table = false;
                return;
            }
            prev = cur;
            cur = self.hash_next[cur];
        }
        unreachable;
    }

    fn linkChild(self: *PrefixCache, parent_id: u32, child_id: u32) void {
        const p = &self.entries[parent_id];
        const old_head = p.first_child;
        self.entries[child_id].prev_sibling = NONE_ID;
        self.entries[child_id].next_sibling = old_head;
        if (old_head != NONE_ID) {
            self.entries[old_head].prev_sibling = child_id;
        }
        p.first_child = child_id;
        if (p.child_count == 0 and p.in_evict_list) {
            self.evictRemove(parent_id);
        }
        p.child_count += 1;
    }

    fn unlinkChild(self: *PrefixCache, parent_id: u32, child_id: u32) void {
        const p = &self.entries[parent_id];
        const child_evict_first = self.entries[child_id].evict_first;
        const prev = self.entries[child_id].prev_sibling;
        const next = self.entries[child_id].next_sibling;
        if (prev != NONE_ID) {
            self.entries[prev].next_sibling = next;
        } else {
            p.first_child = next;
        }
        if (next != NONE_ID) {
            self.entries[next].prev_sibling = prev;
        }
        self.entries[child_id].prev_sibling = NONE_ID;
        self.entries[child_id].next_sibling = NONE_ID;

        std.debug.assert(p.child_count > 0);
        p.child_count -= 1;
        if (p.child_count == 0 and p.pin_count == 0 and !p.in_evict_list) {
            if (child_evict_first and !p.evict_first) {
                // Evicting a transient `evict_first` boundary child should not demote
                // a fresh canonical parent behind older canonical entries.
                self.evictPushHead(parent_id);
            } else {
                // Parent was exposed because its LRU child was evicted: place at LRU
                // tail so the evicted prefix chain reclaims before fresher branches.
                self.evictPushTail(parent_id);
            }
        }
    }

    fn linkRootTail(self: *PrefixCache, id: u32) void {
        const old_head = self.root_tail_head;
        self.entries[id].prev_sibling = NONE_ID;
        self.entries[id].next_sibling = old_head;
        if (old_head != NONE_ID) {
            self.entries[old_head].prev_sibling = id;
        }
        self.root_tail_head = id;
    }

    fn unlinkRootTail(self: *PrefixCache, id: u32) void {
        const prev = self.entries[id].prev_sibling;
        const next = self.entries[id].next_sibling;
        if (prev != NONE_ID) {
            self.entries[prev].next_sibling = next;
        } else {
            self.root_tail_head = next;
        }
        if (next != NONE_ID) {
            self.entries[next].prev_sibling = prev;
        }
        self.entries[id].prev_sibling = NONE_ID;
        self.entries[id].next_sibling = NONE_ID;
    }

    fn pinUnchecked(self: *PrefixCache, id: u32) void {
        const e = &self.entries[id];
        if (e.pin_count == 0) {
            self.pinned_count += 1;
            if (e.in_evict_list) {
                self.evictRemove(id);
            }
        }
        e.pin_count += 1;
    }

    fn unpinRestoringEvictNoTick(self: *PrefixCache, id: u32) void {
        const e = &self.entries[id];
        std.debug.assert(e.in_use and e.pin_count > 0);
        e.pin_count -= 1;
        if (e.pin_count == 0) {
            self.pinned_count -= 1;
            if (e.child_count == 0 and !e.in_evict_list) {
                self.evictPushHead(id);
            }
        }
    }

    fn promoteToCanonical(self: *PrefixCache, id: u32) void {
        if (!self.entries[id].evict_first) return;
        const was_in_list = self.entries[id].in_evict_list;
        if (was_in_list) {
            self.evictRemove(id);
        }
        self.entries[id].evict_first = false;
        if (was_in_list) {
            self.evictPushHead(id);
        }
    }

    fn touchEntry(self: *PrefixCache, id: u32) void {
        self.tick += 1;
        self.entries[id].last_access_tick = self.tick;
        if (self.entries[id].in_evict_list) {
            const head = if (self.entries[id].evict_first) self.evict_first_head else self.evict_head;
            if (head != id) {
                self.evictRemove(id);
                self.evictPushHead(id);
            }
        }
    }

    fn evictPushHead(self: *PrefixCache, id: u32) void {
        std.debug.assert(!self.entries[id].in_evict_list);
        const head_ptr: *u32 = if (self.entries[id].evict_first) &self.evict_first_head else &self.evict_head;
        const tail_ptr: *u32 = if (self.entries[id].evict_first) &self.evict_first_tail else &self.evict_tail;
        self.entries[id].lru_prev = NONE_ID;
        self.entries[id].lru_next = head_ptr.*;
        if (head_ptr.* != NONE_ID) {
            self.entries[head_ptr.*].lru_prev = id;
        } else {
            tail_ptr.* = id;
        }
        head_ptr.* = id;
        self.entries[id].in_evict_list = true;
        self.evictable_count += 1;
    }

    fn evictPushTail(self: *PrefixCache, id: u32) void {
        std.debug.assert(!self.entries[id].in_evict_list);
        const head_ptr: *u32 = if (self.entries[id].evict_first) &self.evict_first_head else &self.evict_head;
        const tail_ptr: *u32 = if (self.entries[id].evict_first) &self.evict_first_tail else &self.evict_tail;
        self.entries[id].lru_next = NONE_ID;
        self.entries[id].lru_prev = tail_ptr.*;
        if (tail_ptr.* != NONE_ID) {
            self.entries[tail_ptr.*].lru_next = id;
        } else {
            head_ptr.* = id;
        }
        tail_ptr.* = id;
        self.entries[id].in_evict_list = true;
        self.evictable_count += 1;
    }

    fn evictRemove(self: *PrefixCache, id: u32) void {
        std.debug.assert(self.entries[id].in_evict_list);
        const head_ptr: *u32 = if (self.entries[id].evict_first) &self.evict_first_head else &self.evict_head;
        const tail_ptr: *u32 = if (self.entries[id].evict_first) &self.evict_first_tail else &self.evict_tail;
        const prev = self.entries[id].lru_prev;
        const next = self.entries[id].lru_next;
        if (prev != NONE_ID) {
            self.entries[prev].lru_next = next;
        } else {
            head_ptr.* = next;
        }
        if (next != NONE_ID) {
            self.entries[next].lru_prev = prev;
        } else {
            tail_ptr.* = prev;
        }
        self.entries[id].lru_prev = NONE_ID;
        self.entries[id].lru_next = NONE_ID;
        self.entries[id].in_evict_list = false;
        self.evictable_count -= 1;
    }
};

// ===========================================================================
// R10: Opt-In NVMe Persistent Prefix KV (`q8` / `FP4`) & `GDN` Cache
// ===========================================================================

pub const DISK_PREFIX_MAGIC: [8]u8 = .{ 'A', 'M', 'K', 'V', 'P', 'C', '0', '1' };
pub const DISK_PREFIX_VERSION: u32 = 1;

pub const DiskKvFormat = enum(u8) {
    none = 0,
    retired_q8 = 1,
    mla_fp4 = 2,

    pub fn label(self: DiskKvFormat) []const u8 {
        return switch (self) {
            .none => "none",
            .retired_q8 => "retired_q8",
            .mla_fp4 => "mla_fp4",
        };
    }
};

/// 144-byte page-0 header for 16 KiB-aligned `F_NOCACHE` persisted prefix artifacts
/// (`R10`, `ADR-0022`). Zero implicit padding; verified at compile time.
pub const DiskPrefixHeader = extern struct {
    magic: [8]u8,
    version: u32,
    page_size: u32,
    amf_header_crc32: u32,
    amf_dir_crc32: u32,
    prefix_hash: u64,
    parent_hash: u64,
    token_count: u32,
    num_kv_layers: u32,
    kv_format: u8,
    has_gdn_snapshot: u8,
    quant_profile_tag: u16,
    kv_bytes_per_layer: u32,
    tokens_offset: u64,
    tokens_blob_bytes: u64,
    kv_offset: u64,
    kv_data_bytes: u64,
    kv_blob_bytes: u64,
    gdn_offset: u64,
    gdn_state_bytes: u64,
    gdn_blob_bytes: u64,
    tokens_crc32: u32,
    kv_crc32: u32,
    gdn_crc32: u32,
    header_crc32: u32,
    reserved: u64,

    pub fn computeCrc32(self: *const DiskPrefixHeader) u32 {
        var copy = self.*;
        copy.header_crc32 = 0;
        return std.hash.Crc32.hash(std.mem.asBytes(&copy));
    }

    pub fn kvFormat(self: *const DiskPrefixHeader) ?DiskKvFormat {
        return std.enums.fromInt(DiskKvFormat, self.kv_format);
    }
};

comptime {
    if (@sizeOf(DiskPrefixHeader) != 144) @compileError("DiskPrefixHeader must be 144 bytes");
}

pub const DiskPrefixError = qos.Error || std.Thread.SpawnError || error{
    Overflow,
    InvalidConfig,
    PathTooLong,
    DirectoryFailed,
    OpenFailed,
    FcntlFailed,
    StatFailed,
    ReadFailed,
    ShortRead,
    WriteFailed,
    RenameFailed,
    BufferTooSmall,
    PayloadTooLarge,
    CorruptHeader,
    ChecksumMismatch,
    ModelMismatch,
    FormatMismatch,
    OutOfMemory,
};

pub const DiskPrefixCacheConfig = struct {
    /// Directory where 16 KiB-aligned `.amkvc` prefix artifacts are persisted.
    cache_dir: []const u8,
    /// Owning model's `.amf` `header_crc32` (binds artifacts to exact checkpoint).
    amf_header_crc32: u32,
    /// Owning model's `.amf` `directory_crc32` (binds artifacts to exact quant layout).
    amf_dir_crc32: u32,
    /// Maximum tokens stored per persisted artifact (for preallocated staging & reload scratch).
    max_tokens_per_artifact: u32 = DEFAULT_BLOCK_TOKENS,
    /// Maximum compressed KV bytes (`R5` `q8` or `R7` `FP4`) per artifact.
    max_kv_bytes_per_artifact: usize,
    /// Maximum `GDN` recurrent state + `conv1d` snapshot bytes per artifact (`0` if non-GDN).
    max_gdn_bytes_per_artifact: usize = 0,
    /// Preallocated 16 KiB-aligned staging slots for async background spills (`1..32`).
    max_pending_spills: u32 = 4,
    /// Chunk granularity (multiple of 16 KiB) between cooperative yields to decode-critical cold reads.
    write_chunk_bytes: usize = 64 * 1024,
    /// Spawn a low-priority utility/throttled background spill worker thread.
    enable_async_worker: bool = true,
    worker_qos: qos.QosClass = .utility,
    worker_io_policy: qos.DiskIoPolicy = .throttle,
    /// Optional external atomic counter incremented by `C3`–`C5` decode-critical cold-expert reads.
    external_cold_read_inflight: ?*const std.atomic.Value(u32) = null,
};

pub const DiskSpillPayload = struct {
    prefix_hash: u64,
    parent_hash: u64 = 0,
    tokens: []const u32,
    num_kv_layers: u32,
    kv_format: DiskKvFormat,
    quant_profile_tag: u16 = 0,
    kv_bytes_per_layer: u32,
    kv_bytes: []const u8,
    gdn_bytes: ?[]const u8 = null,
};

pub const DiskReloadResult = struct {
    header: DiskPrefixHeader,
    kv_bytes_loaded: usize,
    gdn_bytes_loaded: usize,
    has_gdn_snapshot: bool,
};

pub const DiskPrefixStats = struct {
    spills_enqueued: u64 = 0,
    spills_completed: u64 = 0,
    spills_failed: u64 = 0,
    spills_dropped_full: u64 = 0,
    reloads_hit: u64 = 0,
    reloads_miss: u64 = 0,
    reloads_rejected_collision: u64 = 0,
    reloads_rejected_corrupt: u64 = 0,
    bytes_written: u64 = 0,
    bytes_read: u64 = 0,
    io_yields_for_cold_reads: u64 = 0,
};

/// Opt-in NVMe persistent prefix KV (`R5` `q8` / `R7` `FP4`) & `GDN` snapshot
/// cache (`R10`, `--disk-prefix-cache <dir>`, off by default).
///
/// All staging and reload scratch buffers are preallocated once in `init` with
/// 16 KiB page alignment (`0` steady-state heap allocations). Background writes
/// execute with `F_NOCACHE` on a utility/throttled I/O thread and cooperatively
/// yield before every write chunk whenever decode-critical cold-expert reads are
/// active.
pub const DiskPrefixCache = struct {
    const MAX_STAGING_SLOTS: usize = 32;

    const StagingSlot = struct {
        buf: []align(page.APPLE_PAGE_SIZE) u8,
        total_bytes: usize = 0,
        prefix_hash: u64 = 0,
        in_queue: bool = false,
    };

    const SharedState = struct {
        cache_dir: [std.fs.max_path_bytes]u8,
        cache_dir_len: usize,
        amf_header_crc32: u32,
        amf_dir_crc32: u32,
        write_chunk_bytes: usize,
        worker_qos: qos.QosClass,
        worker_io_policy: qos.DiskIoPolicy,
        external_cold_read_inflight: ?*const std.atomic.Value(u32),
        internal_cold_reads: std.atomic.Value(u32),

        slots: []StagingSlot,
        queue: [MAX_STAGING_SLOTS]u32,
        q_head: usize,
        q_tail: usize,
        q_count: usize,
        in_flight_writes: usize,
        temp_seq: std.atomic.Value(u64),

        shutdown: std.atomic.Value(bool),
        worker_ready: std.atomic.Value(bool),
        worker_init_err: ?qos.Error,

        mutex: std.c.pthread_mutex_t,
        work_cond: std.c.pthread_cond_t,
        done_cond: std.c.pthread_cond_t,
        init_cond: std.c.pthread_cond_t,

        spills_enqueued: std.atomic.Value(u64),
        spills_completed: std.atomic.Value(u64),
        spills_failed: std.atomic.Value(u64),
        spills_dropped_full: std.atomic.Value(u64),
        bytes_written: std.atomic.Value(u64),
        io_yields_for_cold_reads: std.atomic.Value(u64),

        fn cacheDirSlice(self: *const SharedState) []const u8 {
            return self.cache_dir[0..self.cache_dir_len];
        }

        fn shouldYieldToColdReads(self: *const SharedState) bool {
            if (self.internal_cold_reads.load(.acquire) > 0) return true;
            if (self.external_cold_read_inflight) |ext| {
                if (ext.load(.acquire) > 0) return true;
            }
            return false;
        }

        fn waitWhileColdReadsActive(self: *SharedState) void {
            while (self.shouldYieldToColdReads()) {
                _ = self.io_yields_for_cold_reads.fetchAdd(1, .monotonic);
                var ts: std.c.timespec = .{ .sec = 0, .nsec = 250_000 }; // 250 us
                _ = std.c.nanosleep(&ts, null);
            }
        }
    };

    state: *SharedState,
    worker_thread: ?std.Thread,
    max_tokens: u32,
    max_kv_bytes: usize,
    max_gdn_bytes: usize,
    slot_capacity_bytes: usize,
    reload_scratch: []align(page.APPLE_PAGE_SIZE) u8,
    reloads_hit: u64,
    reloads_miss: u64,
    reloads_rejected_collision: u64,
    reloads_rejected_corrupt: u64,
    bytes_read: u64,

    pub fn init(allocator: std.mem.Allocator, cfg: DiskPrefixCacheConfig) DiskPrefixError!DiskPrefixCache {
        if (cfg.cache_dir.len == 0 or cfg.cache_dir.len + 96 >= std.fs.max_path_bytes) return error.PathTooLong;
        if (cfg.max_tokens_per_artifact == 0) return error.InvalidConfig;
        if (cfg.max_pending_spills == 0 or cfg.max_pending_spills > MAX_STAGING_SLOTS) return error.InvalidConfig;
        if (cfg.write_chunk_bytes == 0 or (cfg.write_chunk_bytes % page.APPLE_PAGE_SIZE) != 0) return error.InvalidConfig;

        try ensureDirExists(cfg.cache_dir);

        const max_tok_bytes = try page.alignUp(
            std.math.mul(usize, cfg.max_tokens_per_artifact, @sizeOf(u32)) catch return error.InvalidConfig,
        );
        const max_kv_aligned = if (cfg.max_kv_bytes_per_artifact > 0)
            try page.alignUp(cfg.max_kv_bytes_per_artifact)
        else
            0;
        const max_gdn_aligned = if (cfg.max_gdn_bytes_per_artifact > 0)
            try page.alignUp(cfg.max_gdn_bytes_per_artifact)
        else
            0;
        const slot_cap = page.APPLE_PAGE_SIZE + max_tok_bytes + max_kv_aligned + max_gdn_aligned;
        const reload_scratch_len = @max(
            @max(page.APPLE_PAGE_SIZE, max_tok_bytes),
            @min(@max(max_kv_aligned, max_gdn_aligned), @as(usize, 256 * 1024)),
        );

        const state = try allocator.create(SharedState);
        errdefer allocator.destroy(state);

        const slots = try allocator.alloc(StagingSlot, cfg.max_pending_spills);
        errdefer allocator.free(slots);
        var allocated_slots: usize = 0;
        errdefer for (0..allocated_slots) |i| allocator.free(slots[i].buf);

        for (0..cfg.max_pending_spills) |i| {
            const buf = try allocator.alignedAlloc(u8, .fromByteUnits(page.APPLE_PAGE_SIZE), slot_cap);
            slots[i] = .{ .buf = buf };
            allocated_slots += 1;
        }

        const reload_scratch = try allocator.alignedAlloc(u8, .fromByteUnits(page.APPLE_PAGE_SIZE), reload_scratch_len);
        errdefer allocator.free(reload_scratch);

        state.* = .{
            .cache_dir = @splat(0),
            .cache_dir_len = cfg.cache_dir.len,
            .amf_header_crc32 = cfg.amf_header_crc32,
            .amf_dir_crc32 = cfg.amf_dir_crc32,
            .write_chunk_bytes = cfg.write_chunk_bytes,
            .worker_qos = cfg.worker_qos,
            .worker_io_policy = cfg.worker_io_policy,
            .external_cold_read_inflight = cfg.external_cold_read_inflight,
            .internal_cold_reads = .init(0),
            .slots = slots,
            .queue = @splat(0),
            .q_head = 0,
            .q_tail = 0,
            .q_count = 0,
            .in_flight_writes = 0,
            .temp_seq = .init(1),
            .shutdown = .init(false),
            .worker_ready = .init(false),
            .worker_init_err = null,
            .mutex = std.c.PTHREAD_MUTEX_INITIALIZER,
            .work_cond = std.c.PTHREAD_COND_INITIALIZER,
            .done_cond = std.c.PTHREAD_COND_INITIALIZER,
            .init_cond = std.c.PTHREAD_COND_INITIALIZER,
            .spills_enqueued = .init(0),
            .spills_completed = .init(0),
            .spills_failed = .init(0),
            .spills_dropped_full = .init(0),
            .bytes_written = .init(0),
            .io_yields_for_cold_reads = .init(0),
        };
        @memcpy(state.cache_dir[0..cfg.cache_dir.len], cfg.cache_dir);

        var cache: DiskPrefixCache = .{
            .state = state,
            .worker_thread = null,
            .max_tokens = cfg.max_tokens_per_artifact,
            .max_kv_bytes = cfg.max_kv_bytes_per_artifact,
            .max_gdn_bytes = cfg.max_gdn_bytes_per_artifact,
            .slot_capacity_bytes = slot_cap,
            .reload_scratch = reload_scratch,
            .reloads_hit = 0,
            .reloads_miss = 0,
            .reloads_rejected_collision = 0,
            .reloads_rejected_corrupt = 0,
            .bytes_read = 0,
        };
        errdefer cache.deinit(allocator);

        if (cfg.enable_async_worker) {
            cache.worker_thread = try std.Thread.spawn(.{}, workerMain, .{state});
            _ = std.c.pthread_mutex_lock(&state.mutex);
            while (!state.worker_ready.load(.acquire)) {
                _ = std.c.pthread_cond_wait(&state.init_cond, &state.mutex);
            }
            _ = std.c.pthread_mutex_unlock(&state.mutex);
            if (state.worker_init_err) |err| return err;
        }

        return cache;
    }

    pub fn deinit(self: *DiskPrefixCache, allocator: std.mem.Allocator) void {
        const st = self.state;
        if (self.worker_thread) |th| {
            _ = std.c.pthread_mutex_lock(&st.mutex);
            st.shutdown.store(true, .release);
            _ = std.c.pthread_cond_broadcast(&st.work_cond);
            _ = std.c.pthread_mutex_unlock(&st.mutex);
            th.join();
            self.worker_thread = null;
        }
        _ = std.c.pthread_cond_destroy(&st.init_cond);
        _ = std.c.pthread_cond_destroy(&st.done_cond);
        _ = std.c.pthread_cond_destroy(&st.work_cond);
        _ = std.c.pthread_mutex_destroy(&st.mutex);

        allocator.free(self.reload_scratch);
        for (st.slots) |slot| allocator.free(slot.buf);
        allocator.free(st.slots);
        allocator.destroy(st);
    }

    /// Marks the start of a decode-critical `C3`–`C5` cold-expert NVMe read.
    /// Background `DiskPrefixCache` writes immediately yield until all critical
    /// reads call `exitDecodeCriticalRead()`.
    pub fn enterDecodeCriticalRead(self: *DiskPrefixCache) void {
        _ = self.state.internal_cold_reads.fetchAdd(1, .acq_rel);
    }

    pub fn exitDecodeCriticalRead(self: *DiskPrefixCache) void {
        const prev = self.state.internal_cold_reads.fetchSub(1, .acq_rel);
        std.debug.assert(prev > 0);
    }

    pub fn stats(self: *const DiskPrefixCache) DiskPrefixStats {
        const st = self.state;
        return .{
            .spills_enqueued = st.spills_enqueued.load(.monotonic),
            .spills_completed = st.spills_completed.load(.monotonic),
            .spills_failed = st.spills_failed.load(.monotonic),
            .spills_dropped_full = st.spills_dropped_full.load(.monotonic),
            .reloads_hit = self.reloads_hit,
            .reloads_miss = self.reloads_miss,
            .reloads_rejected_collision = self.reloads_rejected_collision,
            .reloads_rejected_corrupt = self.reloads_rejected_corrupt,
            .bytes_written = st.bytes_written.load(.monotonic),
            .bytes_read = self.bytes_read,
            .io_yields_for_cold_reads = st.io_yields_for_cold_reads.load(.monotonic),
        };
    }

    /// Packs `payload` into a preallocated 16 KiB-aligned staging slot with ZERO
    /// heap allocation and queues it for asynchronous `F_NOCACHE` disk spill.
    /// Returns `false` (without blocking) if all staging slots are currently busy.
    pub fn spillAsync(self: *DiskPrefixCache, payload: DiskSpillPayload) DiskPrefixError!bool {
        if (self.worker_thread == null) {
            try self.spillSync(payload);
            return true;
        }
        try self.validatePayload(payload);

        const st = self.state;
        _ = std.c.pthread_mutex_lock(&st.mutex);
        var free_idx: ?usize = null;
        for (st.slots, 0..) |slot, i| {
            if (!slot.in_queue) {
                free_idx = i;
                break;
            }
        }
        if (free_idx == null) {
            _ = std.c.pthread_mutex_unlock(&st.mutex);
            _ = st.spills_dropped_full.fetchAdd(1, .monotonic);
            return false;
        }
        const slot_idx = free_idx.?;
        st.slots[slot_idx].in_queue = true;
        _ = std.c.pthread_mutex_unlock(&st.mutex);

        const total_bytes = packArtifactIntoBuffer(
            st.amf_header_crc32,
            st.amf_dir_crc32,
            payload,
            st.slots[slot_idx].buf,
        ) catch |err| {
            _ = std.c.pthread_mutex_lock(&st.mutex);
            st.slots[slot_idx].in_queue = false;
            _ = std.c.pthread_mutex_unlock(&st.mutex);
            return err;
        };

        _ = std.c.pthread_mutex_lock(&st.mutex);
        st.slots[slot_idx].total_bytes = total_bytes;
        st.slots[slot_idx].prefix_hash = payload.prefix_hash;
        st.queue[st.q_tail] = @intCast(slot_idx);
        st.q_tail = (st.q_tail + 1) % MAX_STAGING_SLOTS;
        st.q_count += 1;
        _ = st.spills_enqueued.fetchAdd(1, .monotonic);
        _ = std.c.pthread_cond_signal(&st.work_cond);
        _ = std.c.pthread_mutex_unlock(&st.mutex);
        return true;
    }

    /// Synchronously writes a 16 KiB-aligned `F_NOCACHE` prefix artifact using
    /// a preallocated staging slot (`0` heap allocs), yielding to active
    /// decode-critical cold-expert reads between chunks.
    pub fn spillSync(self: *DiskPrefixCache, payload: DiskSpillPayload) DiskPrefixError!void {
        try self.validatePayload(payload);
        self.flush();

        const st = self.state;
        const total_bytes = try packArtifactIntoBuffer(
            st.amf_header_crc32,
            st.amf_dir_crc32,
            payload,
            st.slots[0].buf,
        );
        _ = st.spills_enqueued.fetchAdd(1, .monotonic);
        writeArtifactAtomic(st, payload.prefix_hash, st.slots[0].buf[0..total_bytes]) catch |err| {
            _ = st.spills_failed.fetchAdd(1, .monotonic);
            return err;
        };
        _ = st.spills_completed.fetchAdd(1, .monotonic);
    }

    /// Blocks until all queued asynchronous spill writes finish.
    pub fn flush(self: *DiskPrefixCache) void {
        if (self.worker_thread == null) return;
        const st = self.state;
        _ = std.c.pthread_mutex_lock(&st.mutex);
        while (st.q_count > 0 or st.in_flight_writes > 0) {
            _ = std.c.pthread_cond_wait(&st.done_cond, &st.mutex);
        }
        _ = std.c.pthread_mutex_unlock(&st.mutex);
    }

    /// Reloads a persisted prefix artifact from disk with `F_NOCACHE` and zero
    /// heap allocation. Verifies `.amf` header/directory CRC, format, quant tag,
    /// full header/tokens/KV/GDN CRC-32s, and **exact token-slice equality**
    /// (`0` hash-collision risk). Returns `null` on miss or collision rejection.
    pub fn reload(
        self: *DiskPrefixCache,
        prefix_hash: u64,
        expected_tokens: []const u32,
        expected_format: DiskKvFormat,
        expected_quant_profile: u16,
        kv_dst: []u8,
        gdn_dst: ?[]u8,
    ) DiskPrefixError!?DiskReloadResult {
        if (expected_tokens.len == 0 or expected_tokens.len > self.max_tokens) return error.InvalidConfig;

        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const path_z = try formatArtifactPath(
            self.state.cacheDirSlice(),
            self.state.amf_header_crc32,
            self.state.amf_dir_crc32,
            prefix_hash,
            &path_buf,
        );

        const fd = std.c.open(path_z.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
        if (fd < 0) {
            const err = std.c.errno(fd);
            if (err == .NOENT) {
                self.reloads_miss += 1;
                return null;
            }
            return error.OpenFailed;
        }
        defer _ = std.c.close(fd);

        if (std.c.fcntl(fd, std.c.F.NOCACHE, @as(c_int, 1)) != 0) return error.FcntlFailed;

        var st: std.c.Stat = undefined;
        if (std.c.fstat(fd, &st) != 0) return error.StatFailed;
        if (st.size < page.APPLE_PAGE_SIZE) {
            self.reloads_rejected_corrupt += 1;
            return error.CorruptHeader;
        }
        const file_size: u64 = @intCast(st.size);
        if ((file_size % page.APPLE_PAGE_SIZE) != 0) {
            self.reloads_rejected_corrupt += 1;
            return error.CorruptHeader;
        }

        // 1. Read and validate page 0 header.
        const page0 = self.reload_scratch[0..page.APPLE_PAGE_SIZE];
        try diskPreadExact(fd, page0, 0);
        self.bytes_read += page.APPLE_PAGE_SIZE;

        var hdr: DiskPrefixHeader = undefined;
        @memcpy(std.mem.asBytes(&hdr), page0[0..@sizeOf(DiskPrefixHeader)]);

        if (!std.mem.eql(u8, &hdr.magic, &DISK_PREFIX_MAGIC) or
            hdr.version != DISK_PREFIX_VERSION or
            hdr.page_size != page.APPLE_PAGE_SIZE)
        {
            self.reloads_rejected_corrupt += 1;
            return error.CorruptHeader;
        }
        if (hdr.computeCrc32() != hdr.header_crc32) {
            self.reloads_rejected_corrupt += 1;
            return error.ChecksumMismatch;
        }
        if (hdr.amf_header_crc32 != self.state.amf_header_crc32 or
            hdr.amf_dir_crc32 != self.state.amf_dir_crc32)
        {
            self.reloads_miss += 1;
            return error.ModelMismatch;
        }
        if (hdr.prefix_hash != prefix_hash) {
            self.reloads_rejected_collision += 1;
            return null;
        }
        const fmt = hdr.kvFormat() orelse {
            self.reloads_rejected_corrupt += 1;
            return error.CorruptHeader;
        };
        if (fmt != expected_format or hdr.quant_profile_tag != expected_quant_profile) {
            self.reloads_miss += 1;
            return error.FormatMismatch;
        }
        if (hdr.token_count != expected_tokens.len) {
            self.reloads_rejected_collision += 1;
            return null;
        }

        // Validate aligned section geometry without u64 wrap-around.
        const tok_raw_bytes: u64 = @as(u64, hdr.token_count) * @sizeOf(u32);
        const expected_tok_blob: u64 = page.alignUp(@intCast(tok_raw_bytes)) catch {
            self.reloads_rejected_corrupt += 1;
            return error.CorruptHeader;
        };
        const expected_kv_blob: u64 = if (hdr.kv_data_bytes > 0)
            page.alignUp(@intCast(hdr.kv_data_bytes)) catch {
                self.reloads_rejected_corrupt += 1;
                return error.CorruptHeader;
            }
        else
            0;
        const expected_gdn_blob: u64 = if (hdr.has_gdn_snapshot != 0 and hdr.gdn_state_bytes > 0)
            page.alignUp(@intCast(hdr.gdn_state_bytes)) catch {
                self.reloads_rejected_corrupt += 1;
                return error.CorruptHeader;
            }
        else
            0;
        const expected_kv_offset = std.math.add(u64, hdr.tokens_offset, hdr.tokens_blob_bytes) catch {
            self.reloads_rejected_corrupt += 1;
            return error.CorruptHeader;
        };
        const expected_gdn_offset = std.math.add(u64, hdr.kv_offset, hdr.kv_blob_bytes) catch {
            self.reloads_rejected_corrupt += 1;
            return error.CorruptHeader;
        };
        const sum_tok = std.math.add(u64, page.APPLE_PAGE_SIZE, hdr.tokens_blob_bytes) catch {
            self.reloads_rejected_corrupt += 1;
            return error.CorruptHeader;
        };
        const sum_kv = std.math.add(u64, sum_tok, hdr.kv_blob_bytes) catch {
            self.reloads_rejected_corrupt += 1;
            return error.CorruptHeader;
        };
        const expected_file_size = std.math.add(u64, sum_kv, hdr.gdn_blob_bytes) catch {
            self.reloads_rejected_corrupt += 1;
            return error.CorruptHeader;
        };
        if (hdr.tokens_offset != page.APPLE_PAGE_SIZE or
            hdr.tokens_blob_bytes != expected_tok_blob or
            hdr.kv_offset != expected_kv_offset or
            hdr.kv_blob_bytes != expected_kv_blob or
            (hdr.has_gdn_snapshot != 0 and hdr.gdn_offset != expected_gdn_offset) or
            hdr.gdn_blob_bytes != expected_gdn_blob or
            expected_file_size != file_size)
        {
            self.reloads_rejected_corrupt += 1;
            return error.CorruptHeader;
        }

        if (hdr.kv_data_bytes > kv_dst.len) return error.BufferTooSmall;
        if (hdr.has_gdn_snapshot != 0 and hdr.gdn_state_bytes > 0) {
            const gd = gdn_dst orelse return error.BufferTooSmall;
            if (hdr.gdn_state_bytes > gd.len) return error.BufferTooSmall;
        }

        // 2. Read full token slice blob, verify CRC32, and verify exact token equality.
        const tok_blob_usize: usize = @intCast(hdr.tokens_blob_bytes);
        if (tok_blob_usize > self.reload_scratch.len) return error.BufferTooSmall;
        const tok_buf = self.reload_scratch[0..tok_blob_usize];
        try diskPreadExact(fd, tok_buf, hdr.tokens_offset);
        self.bytes_read += tok_blob_usize;

        if (std.hash.Crc32.hash(tok_buf) != hdr.tokens_crc32) {
            self.reloads_rejected_corrupt += 1;
            return error.ChecksumMismatch;
        }
        const stored_tokens: []const u32 = @alignCast(std.mem.bytesAsSlice(u32, tok_buf[0..@intCast(tok_raw_bytes)]));
        if (!std.mem.eql(u32, stored_tokens, expected_tokens)) {
            self.reloads_rejected_collision += 1;
            return null;
        }

        // 3. Stream compressed KV tiles (`q8` or `FP4`) via F_NOCACHE pread and verify CRC32.
        const kv_len: usize = @intCast(hdr.kv_data_bytes);
        if (hdr.kv_blob_bytes > 0) {
            const kv_crc = try self.streamSectionFromDisk(fd, hdr.kv_offset, hdr.kv_blob_bytes, kv_dst[0..kv_len]);
            if (kv_crc != hdr.kv_crc32) {
                self.reloads_rejected_corrupt += 1;
                return error.ChecksumMismatch;
            }
        }

        // 4. Stream optional GDN recurrent state snapshot and verify CRC32.
        var gdn_len: usize = 0;
        if (hdr.has_gdn_snapshot != 0 and hdr.gdn_blob_bytes > 0) {
            gdn_len = @intCast(hdr.gdn_state_bytes);
            const gdn_crc = try self.streamSectionFromDisk(fd, hdr.gdn_offset, hdr.gdn_blob_bytes, gdn_dst.?[0..gdn_len]);
            if (gdn_crc != hdr.gdn_crc32) {
                self.reloads_rejected_corrupt += 1;
                return error.ChecksumMismatch;
            }
        }

        self.reloads_hit += 1;
        return .{
            .header = hdr,
            .kv_bytes_loaded = kv_len,
            .gdn_bytes_loaded = gdn_len,
            .has_gdn_snapshot = hdr.has_gdn_snapshot != 0,
        };
    }

    /// Counts persisted `.amkvc` artifacts in `cache_dir` matching this model's
    /// `.amf` header + directory CRC pair (used on cold restart).
    pub fn countModelArtifacts(self: *const DiskPrefixCache) DiskPrefixError!u32 {
        var dir_buf: [std.fs.max_path_bytes]u8 = undefined;
        const dir_slice = self.state.cacheDirSlice();
        @memcpy(dir_buf[0..dir_slice.len], dir_slice);
        dir_buf[dir_slice.len] = 0;
        const dp = std.c.opendir(dir_buf[0..dir_slice.len :0].ptr) orelse return error.DirectoryFailed;
        defer _ = std.c.closedir(dp);

        var prefix_buf: [32]u8 = undefined;
        const prefix = std.fmt.bufPrint(&prefix_buf, "pc_{x:0>8}_{x:0>8}_", .{
            self.state.amf_header_crc32,
            self.state.amf_dir_crc32,
        }) catch return error.PathTooLong;

        var count: u32 = 0;
        while (std.c.readdir(dp)) |ent| {
            const name = ent.name[0..ent.namlen];
            if (std.mem.startsWith(u8, name, prefix) and std.mem.endsWith(u8, name, ".amkvc")) {
                count += 1;
            }
        }
        return count;
    }

    fn validatePayload(self: *const DiskPrefixCache, payload: DiskSpillPayload) DiskPrefixError!void {
        if (payload.tokens.len == 0 or payload.tokens.len > self.max_tokens) return error.PayloadTooLarge;
        const expected_kv = std.math.mul(usize, payload.num_kv_layers, payload.kv_bytes_per_layer) catch return error.PayloadTooLarge;
        if (payload.kv_bytes.len != expected_kv) return error.InvalidConfig;
        if (payload.kv_bytes.len > self.max_kv_bytes) return error.PayloadTooLarge;
        if (payload.gdn_bytes) |gb| {
            if (gb.len > self.max_gdn_bytes) return error.PayloadTooLarge;
        }
    }

    fn streamSectionFromDisk(
        self: *DiskPrefixCache,
        fd: std.c.fd_t,
        file_offset: u64,
        blob_bytes: u64,
        dst: []u8,
    ) DiskPrefixError!u32 {
        var hasher = std.hash.Crc32.init();
        var done: u64 = 0;
        while (done < blob_bytes) {
            const take: usize = @intCast(@min(blob_bytes - done, @as(u64, self.reload_scratch.len)));
            const chunk = self.reload_scratch[0..take];
            try diskPreadExact(fd, chunk, file_offset + done);
            self.bytes_read += take;
            hasher.update(chunk);
            if (done < dst.len) {
                const copy_n: usize = @min(dst.len - @as(usize, @intCast(done)), take);
                @memcpy(dst[@intCast(done) .. @as(usize, @intCast(done)) + copy_n], chunk[0..copy_n]);
            }
            done += take;
        }
        return hasher.final();
    }

    fn packArtifactIntoBuffer(
        amf_header_crc32: u32,
        amf_dir_crc32: u32,
        payload: DiskSpillPayload,
        buf: []align(page.APPLE_PAGE_SIZE) u8,
    ) DiskPrefixError!usize {
        const tok_raw_bytes = std.math.mul(usize, payload.tokens.len, @sizeOf(u32)) catch return error.PayloadTooLarge;
        const tok_blob_bytes = try page.alignUp(tok_raw_bytes);
        const kv_blob_bytes = if (payload.kv_bytes.len > 0) try page.alignUp(payload.kv_bytes.len) else 0;
        const gdn_slice = payload.gdn_bytes orelse &.{};
        const has_gdn: u8 = if (gdn_slice.len > 0) 1 else 0;
        const gdn_blob_bytes = if (gdn_slice.len > 0) try page.alignUp(gdn_slice.len) else 0;

        const tokens_offset: usize = page.APPLE_PAGE_SIZE;
        const kv_offset: usize = tokens_offset + tok_blob_bytes;
        const gdn_offset: usize = if (has_gdn != 0) kv_offset + kv_blob_bytes else 0;
        const total_bytes: usize = kv_offset + kv_blob_bytes + gdn_blob_bytes;
        if (total_bytes > buf.len) return error.PayloadTooLarge;

        @memset(buf[0..total_bytes], 0);

        @memcpy(buf[tokens_offset .. tokens_offset + tok_raw_bytes], std.mem.sliceAsBytes(payload.tokens));
        const tok_crc = std.hash.Crc32.hash(buf[tokens_offset .. tokens_offset + tok_blob_bytes]);

        var kv_crc: u32 = 0;
        if (kv_blob_bytes > 0) {
            @memcpy(buf[kv_offset .. kv_offset + payload.kv_bytes.len], payload.kv_bytes);
            kv_crc = std.hash.Crc32.hash(buf[kv_offset .. kv_offset + kv_blob_bytes]);
        }

        var gdn_crc: u32 = 0;
        if (gdn_blob_bytes > 0) {
            @memcpy(buf[gdn_offset .. gdn_offset + gdn_slice.len], gdn_slice);
            gdn_crc = std.hash.Crc32.hash(buf[gdn_offset .. gdn_offset + gdn_blob_bytes]);
        }

        var hdr: DiskPrefixHeader = .{
            .magic = DISK_PREFIX_MAGIC,
            .version = DISK_PREFIX_VERSION,
            .page_size = page.APPLE_PAGE_SIZE,
            .amf_header_crc32 = amf_header_crc32,
            .amf_dir_crc32 = amf_dir_crc32,
            .prefix_hash = payload.prefix_hash,
            .parent_hash = payload.parent_hash,
            .token_count = @intCast(payload.tokens.len),
            .num_kv_layers = payload.num_kv_layers,
            .kv_format = @backingInt(payload.kv_format),
            .has_gdn_snapshot = has_gdn,
            .quant_profile_tag = payload.quant_profile_tag,
            .kv_bytes_per_layer = payload.kv_bytes_per_layer,
            .tokens_offset = tokens_offset,
            .tokens_blob_bytes = tok_blob_bytes,
            .kv_offset = kv_offset,
            .kv_data_bytes = payload.kv_bytes.len,
            .kv_blob_bytes = kv_blob_bytes,
            .gdn_offset = gdn_offset,
            .gdn_state_bytes = gdn_slice.len,
            .gdn_blob_bytes = gdn_blob_bytes,
            .tokens_crc32 = tok_crc,
            .kv_crc32 = kv_crc,
            .gdn_crc32 = gdn_crc,
            .header_crc32 = 0,
            .reserved = 0,
        };
        hdr.header_crc32 = hdr.computeCrc32();
        @memcpy(buf[0..@sizeOf(DiskPrefixHeader)], std.mem.asBytes(&hdr));
        return total_bytes;
    }

    fn writeArtifactAtomic(st: *SharedState, prefix_hash: u64, data: []const u8) DiskPrefixError!void {
        std.debug.assert((data.len % page.APPLE_PAGE_SIZE) == 0);
        var final_buf: [std.fs.max_path_bytes]u8 = undefined;
        const final_z = try formatArtifactPath(
            st.cacheDirSlice(),
            st.amf_header_crc32,
            st.amf_dir_crc32,
            prefix_hash,
            &final_buf,
        );

        const seq = st.temp_seq.fetchAdd(1, .monotonic);
        var tmp_buf: [std.fs.max_path_bytes]u8 = undefined;
        const tmp_raw = std.fmt.bufPrint(tmp_buf[0 .. tmp_buf.len - 1], "{s}.tmp.{d}", .{
            final_z,
            seq,
        }) catch return error.PathTooLong;
        tmp_buf[tmp_raw.len] = 0;
        const tmp_z: [:0]const u8 = tmp_buf[0..tmp_raw.len :0];

        const fd = std.c.open(
            tmp_z.ptr,
            .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true },
            @as(std.c.mode_t, 0o644),
        );
        if (fd < 0) return error.OpenFailed;
        var closed = false;
        errdefer {
            if (!closed) _ = std.c.close(fd);
            _ = std.c.unlink(tmp_z.ptr);
        }

        if (std.c.fcntl(fd, std.c.F.NOCACHE, @as(c_int, 1)) != 0) return error.FcntlFailed;

        var written: usize = 0;
        while (written < data.len) {
            st.waitWhileColdReadsActive();
            const take = @min(st.write_chunk_bytes, data.len - written);
            try diskPwriteExact(fd, data[written .. written + take], written);
            written += take;
            _ = st.bytes_written.fetchAdd(take, .monotonic);
        }
        st.waitWhileColdReadsActive();
        if (std.c.fsync(fd) != 0) return error.WriteFailed;
        _ = std.c.close(fd);
        closed = true;

        if (std.c.rename(tmp_z.ptr, final_z.ptr) != 0) return error.RenameFailed;
    }

    fn workerMain(st: *SharedState) void {
        var init_err: ?qos.Error = null;
        if (qos.setCurrentThreadQos(st.worker_qos, 0)) {
            if (qos.setThreadDiskIoPolicy(st.worker_io_policy)) {} else |err| {
                init_err = err;
            }
        } else |err| {
            init_err = err;
        }

        _ = std.c.pthread_mutex_lock(&st.mutex);
        st.worker_init_err = init_err;
        st.worker_ready.store(true, .release);
        _ = std.c.pthread_cond_signal(&st.init_cond);
        _ = std.c.pthread_mutex_unlock(&st.mutex);
        if (init_err != null) return;

        while (true) {
            _ = std.c.pthread_mutex_lock(&st.mutex);
            while (st.q_count == 0 and !st.shutdown.load(.acquire)) {
                _ = std.c.pthread_cond_wait(&st.work_cond, &st.mutex);
            }
            if (st.q_count == 0 and st.shutdown.load(.acquire)) {
                _ = std.c.pthread_mutex_unlock(&st.mutex);
                return;
            }
            const slot_idx: usize = st.queue[st.q_head];
            st.q_head = (st.q_head + 1) % MAX_STAGING_SLOTS;
            st.q_count -= 1;
            st.in_flight_writes += 1;
            const total_bytes = st.slots[slot_idx].total_bytes;
            const prefix_hash = st.slots[slot_idx].prefix_hash;
            _ = std.c.pthread_mutex_unlock(&st.mutex);

            if (writeArtifactAtomic(st, prefix_hash, st.slots[slot_idx].buf[0..total_bytes])) {
                _ = st.spills_completed.fetchAdd(1, .monotonic);
            } else |_| {
                _ = st.spills_failed.fetchAdd(1, .monotonic);
            }

            _ = std.c.pthread_mutex_lock(&st.mutex);
            st.slots[slot_idx].in_queue = false;
            st.in_flight_writes -= 1;
            if (st.q_count == 0 and st.in_flight_writes == 0) {
                _ = std.c.pthread_cond_broadcast(&st.done_cond);
            }
            _ = std.c.pthread_mutex_unlock(&st.mutex);
        }
    }

    fn formatArtifactPath(
        cache_dir: []const u8,
        amf_header_crc32: u32,
        amf_dir_crc32: u32,
        prefix_hash: u64,
        buf: *[std.fs.max_path_bytes]u8,
    ) DiskPrefixError![:0]const u8 {
        const sep = if (std.mem.endsWith(u8, cache_dir, "/")) "" else "/";
        const raw = std.fmt.bufPrint(buf[0 .. buf.len - 1], "{s}{s}pc_{x:0>8}_{x:0>8}_{x:0>16}.amkvc", .{
            cache_dir,
            sep,
            amf_header_crc32,
            amf_dir_crc32,
            prefix_hash,
        }) catch return error.PathTooLong;
        buf[raw.len] = 0;
        return buf[0..raw.len :0];
    }

    fn ensureDirExists(dir_path: []const u8) DiskPrefixError!void {
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        if (dir_path.len == 0 or dir_path.len >= buf.len) return error.PathTooLong;
        var i: usize = 1;
        while (i <= dir_path.len) : (i += 1) {
            if (i < dir_path.len and dir_path[i] != '/') continue;
            @memcpy(buf[0..i], dir_path[0..i]);
            buf[i] = 0;
            const rc = std.c.mkdir(buf[0..i :0].ptr, 0o755);
            if (rc != 0) {
                const err = std.c.errno(rc);
                if (err != .EXIST) return error.DirectoryFailed;
            }
        }
    }

    fn diskPreadExact(fd: std.c.fd_t, dst: []u8, offset: u64) DiskPrefixError!void {
        const MAX_PREAD_CHUNK: usize = 0x7fff_c000;
        var done: usize = 0;
        while (done < dst.len) {
            const chunk = @min(dst.len - done, MAX_PREAD_CHUNK);
            const rc = std.c.pread(fd, dst.ptr + done, chunk, @intCast(offset + done));
            if (rc < 0) {
                const err = std.c.errno(rc);
                if (err == .INTR) continue;
                return error.ReadFailed;
            }
            if (rc == 0) return error.ShortRead;
            done += @intCast(rc);
        }
    }

    fn diskPwriteExact(fd: std.c.fd_t, src: []const u8, offset: u64) DiskPrefixError!void {
        var done: usize = 0;
        while (done < src.len) {
            const rc = std.c.pwrite(fd, src.ptr + done, src.len - done, @intCast(offset + done));
            if (rc < 0) {
                const err = std.c.errno(rc);
                if (err == .INTR) continue;
                return error.WriteFailed;
            }
            if (rc == 0) return error.WriteFailed;
            done += @intCast(rc);
        }
    }
};

// ===========================================================================
// Tests
// ===========================================================================

fn makeTestBudget(bytes: usize) budget_mod.Budget {
    return budget_mod.Budget.compute(.{
        .memory_bytes = bytes * 8,
        .user_wire_limit = bytes * 4,
        .global_user_wire_limit = bytes * 4,
        .gpu_working_set_bytes = bytes * 4,
        .configured_cap = null,
        .headroom_bytes = page.APPLE_PAGE_SIZE,
    });
}

test "PrefixCache exact token verification prevents hash bucket collisions and enforces O(1) LRU + pinning" {
    const spec: PrefixCacheSpec = .{
        .max_entries = 6,
        .block_tokens = 16,
        .num_kv_layers = 2,
    };
    const need = try PrefixCache.requiredBytes(spec);
    var arena = try arena_mod.WiredArena.init(need, makeTestBudget(need));
    defer arena.deinit();

    var cache = try PrefixCache.init(&arena, spec);
    const rem_after_init = arena.remaining();

    // Build two distinct prefixes A (32 tok = 2 full blocks) and B (24 tok = 1 full + 8 tail).
    var tok_a: [32]u32 = undefined;
    var tok_b: [24]u32 = undefined;
    for (0..32) |i| tok_a[i] = @intCast(100 + i);
    for (0..24) |i| tok_b[i] = @intCast(200 + i);

    const h_a0 = [2]KvBlockHandle{
        .{ .k_tex_id = 10, .v_tex_id = 11, .pool_slot_id = 0, .storage = .active_tile },
        .{ .k_tex_id = 12, .v_tex_id = 13, .pool_slot_id = 1, .storage = .active_tile },
    };
    const h_a1 = [2]KvBlockHandle{
        .{ .k_tex_id = 20, .v_tex_id = 21, .pool_slot_id = 2, .storage = .active_tile },
        .{ .k_tex_id = 22, .v_tex_id = 23, .pool_slot_id = 3, .storage = .active_tile },
    };
    const h_b0 = [2]KvBlockHandle{
        .{ .k_tex_id = 30, .v_tex_id = 31, .pool_slot_id = 4, .storage = .active_tile },
        .{ .k_tex_id = 32, .v_tex_id = 33, .pool_slot_id = 5, .storage = .active_tile },
    };
    const h_b1 = [2]KvBlockHandle{
        .{ .k_tex_id = 40, .v_tex_id = 41, .pool_slot_id = 6, .storage = .active_tile },
        .{ .k_tex_id = 42, .v_tex_id = 43, .pool_slot_id = 7, .storage = .active_tile },
    };

    const r_a0 = try cache.insertFullBlock(NONE_ID, tok_a[0..16], &h_a0, .{ .gdn_snapshot_id = 0 });
    const r_a1 = try cache.insertFullBlock(r_a0.entry_id, tok_a[16..32], &h_a1, .{ .gdn_snapshot_id = 1 });
    const r_b0 = try cache.insertFullBlock(NONE_ID, tok_b[0..16], &h_b0, .{ .gdn_snapshot_id = 2 });
    const r_b1 = try cache.insertTailBlock(r_b0.entry_id, tok_b[16..24], &h_b1, .{ .gdn_snapshot_id = 3 });
    try std.testing.expect(!r_a0.already_existed);
    try std.testing.expect(!r_b1.already_existed);
    try std.testing.expectEqual(@as(u32, 4), cache.liveEntries());

    // Duplicate insert returns existing entry without consuming a new slot.
    const dup_a0 = try cache.insertFullBlock(NONE_ID, tok_a[0..16], &h_a0, .{});
    try std.testing.expect(dup_a0.already_existed);
    try std.testing.expectEqual(r_a0.entry_id, dup_a0.entry_id);
    try std.testing.expectEqual(@as(u32, 4), cache.liveEntries());

    // Simulate a forced hash collision in the bucket of `r_a0`: exact token verification MUST reject it.
    var tok_collide = tok_a[0..16].*;
    tok_collide[15] ^= 1;
    // Force `r_a0` to share the hash of `tok_collide` temporarily to verify exact token compare.
    const saved_hash = cache.entries[r_a0.entry_id].hash;
    const fake_h = hashBlock(0, &tok_collide);
    cache.hashRemove(r_a0.entry_id, saved_hash);
    cache.entries[r_a0.entry_id].hash = fake_h;
    cache.hashInsert(r_a0.entry_id, fake_h);
    var match_buf: [4]u32 = undefined;
    const miss = try cache.matchAndPin(&tok_collide, &match_buf, .{});
    try std.testing.expectEqual(@as(u32, 0), miss.matched_tokens);
    // Restore real hash.
    cache.hashRemove(r_a0.entry_id, fake_h);
    cache.entries[r_a0.entry_id].hash = saved_hash;
    cache.hashInsert(r_a0.entry_id, saved_hash);

    // Hybrid GDN partial-tail fallback: matching only 4 of 8 tokens in `r_b1`
    // - without `require_gdn_snapshot`: returns 16 full + 4 CoW tail (no GDN snapshot at 20).
    // - with `require_gdn_snapshot`: clamps cleanly back to 16 full (`r_b0`, snapshot 2).
    var tok_b_short: [20]u32 = undefined;
    @memcpy(tok_b_short[0..20], tok_b[0..20]);
    var m_gqa = try cache.matchAndPin(&tok_b_short, &match_buf, .{ .require_gdn_snapshot = false });
    try std.testing.expectEqual(@as(u32, 20), m_gqa.matched_tokens);
    try std.testing.expectEqual(@as(u32, 1), m_gqa.full_block_count);
    try std.testing.expectEqual(@as(u32, 4), m_gqa.tail_tokens);
    try std.testing.expectEqual(r_b1.entry_id, m_gqa.tail_entry_id);
    try std.testing.expectEqual(NO_SNAPSHOT, m_gqa.gdn_snapshot_id);
    try cache.releaseMatch(&m_gqa, &match_buf);

    var m_gdn = try cache.matchAndPin(&tok_b_short, &match_buf, .{ .require_gdn_snapshot = true });
    try std.testing.expectEqual(@as(u32, 16), m_gdn.matched_tokens);
    try std.testing.expectEqual(@as(u32, 1), m_gdn.full_block_count);
    try std.testing.expectEqual(@as(u32, 0), m_gdn.tail_tokens);
    try std.testing.expectEqual(@as(u32, 2), m_gdn.gdn_snapshot_id);
    try cache.releaseMatch(&m_gdn, &match_buf);

    // Pin prefix A (r_a0 + r_a1) and verify eviction skips pinned A and evicts B1 then B0.
    var m_a = try cache.matchAndPin(&tok_a, &match_buf, .{});
    try std.testing.expectEqual(@as(u32, 32), m_a.matched_tokens);
    try std.testing.expectEqual(@as(u32, 2), cache.pinnedEntries());

    var evicted_h: [2]KvBlockHandle = undefined;
    const ev1 = try cache.evictLru(&evicted_h);
    try std.testing.expectEqual(r_b1.entry_id, ev1.entry_id);
    try std.testing.expectEqual(@as(u32, 3), ev1.gdn_snapshot_id);
    try std.testing.expectEqual(@as(u32, 6), evicted_h[0].pool_slot_id);

    const ev2 = try cache.evictLru(&evicted_h);
    try std.testing.expectEqual(r_b0.entry_id, ev2.entry_id);
    try std.testing.expectEqual(@as(u32, 2), ev2.gdn_snapshot_id);

    // Only pinned prefix A remains -> eviction MUST fail with `error.CacheFull`.
    try std.testing.expectError(error.CacheFull, cache.evictLru(&evicted_h));

    // Retire r_a0 to q8 in place and verify `populateLayerKvTables` returns 1 retired + 1 active.
    const prev_h0 = try cache.retireEntryLayerToQ8(r_a0.entry_id, 0, 77);
    try std.testing.expectEqual(@as(u32, 0), prev_h0.pool_slot_id);
    try std.testing.expectEqual(KvBlockStorage.active_tile, prev_h0.storage);
    var ret_ids: [4]u32 = undefined;
    var act_tiles: [4]attention.KvTilePair = undefined;
    const counts = try cache.populateLayerKvTables(match_buf[0..m_a.full_block_count], 0, &ret_ids, &act_tiles);
    try std.testing.expectEqual(@as(u32, 1), counts.num_retired_chunks);
    try std.testing.expectEqual(@as(u32, 1), counts.num_active_chunks);
    try std.testing.expectEqual(@as(u32, 77), ret_ids[0]);
    try std.testing.expectEqual(@as(u64, 20), act_tiles[0].k_tex_id);

    // Release prefix A pins -> now A1 then A0 evict cleanly in O(1).
    try cache.releaseMatch(&m_a, &match_buf);
    try std.testing.expectEqual(@as(u32, 0), cache.pinnedEntries());
    const ev3 = try cache.evictLru(&evicted_h);
    try std.testing.expectEqual(r_a1.entry_id, ev3.entry_id);
    const ev4 = try cache.evictLru(&evicted_h);
    try std.testing.expectEqual(r_a0.entry_id, ev4.entry_id);
    try std.testing.expectEqual(@as(u32, 0), cache.liveEntries());
    // Zero arena or heap allocations occurred after init.
    try std.testing.expectEqual(rem_after_init, arena.remaining());
}

test "R8 PrefixCache GPU end-to-end: 4 concurrent requests share 1,200-token prefix (2 full tiles + 176-tok CoW tail + GDN snapshot)" {
    const gpa = std.testing.allocator;
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const attn_pipes = try attention.AttentionPipelines.init(device, library);
    defer attn_pipes.close();
    const hybrid_pipes = try hybrid.HybridPipelines.init(device, library);
    defer hybrid_pipes.close();

    // Shared prefix geometry: 1,200 tokens = 2 full 512-tok tiles + 176-tok partial tail.
    const num_reqs: u32 = 4;
    const prefix_len: u32 = 1200;
    const chunk_tokens: u32 = DEFAULT_BLOCK_TOKENS; // 512
    const full_chunks: u32 = prefix_len / chunk_tokens; // 2
    const tail_tokens: u32 = prefix_len % chunk_tokens; // 176
    const num_q_heads: u32 = 8;
    const num_kv_heads: u32 = 2;
    const head_dim: u32 = 128;
    const rows: u32 = num_kv_heads * chunk_tokens;

    // Hybrid GDN geometry (2 GDN layers, conv_k = 4 -> 3 history tokens).
    const gdn_spec: hybrid.GdnStateSpec = .{
        .num_gdn_layers = 2,
        .num_v_heads = 4,
        .head_k_dim = 64,
        .head_v_dim = 64,
    };
    const gdn_num_k_heads: u32 = 2;
    const conv_k: u32 = 4;
    const gdn_qk: u32 = gdn_num_k_heads * gdn_spec.head_k_dim; // 128
    const gdn_v: u32 = gdn_spec.num_v_heads * gdn_spec.head_v_dim; // 256
    const conv_dim: u32 = 2 * gdn_qk + gdn_v; // 512
    const conv_f16_per_layer: u32 = (conv_k - 1) * conv_dim; // 1536
    const conv_layer_stride_bytes: usize = amf.align64(@as(usize, conv_f16_per_layer) * @sizeOf(f16));
    const conv_layer_stride_f16: u32 = @intCast(conv_layer_stride_bytes / @sizeOf(f16));

    // Preallocate ONE wired arena for PrefixCache, GdnSnapshotPool, 4 per-slot GdnStateArenas,
    // conv1d histories, and batched decode scratch/IO buffers (0 hot-path heap allocs!).
    const cache_spec: PrefixCacheSpec = .{
        .max_entries = 8,
        .block_tokens = chunk_tokens,
        .num_kv_layers = 1,
    };
    const snap_spec: hybrid.GdnSnapshotPoolSpec = .{
        .state = gdn_spec,
        .conv_f16_per_layer = conv_f16_per_layer,
        .max_snapshots = 4,
    };
    const scratch_layout = attention.gqaBatchScratchLayout(num_reqs, num_q_heads, head_dim, attention.MAX_SPLITS);
    const arena_bytes: usize =
        (try PrefixCache.requiredBytes(cache_spec)) +
        (try hybrid.GdnSnapshotPool.requiredArenaBytes(snap_spec)) +
        (try hybrid.GdnStateArena.requiredBytes(gdn_spec)) * num_reqs +
        conv_layer_stride_bytes * gdn_spec.num_gdn_layers * num_reqs +
        scratch_layout.total_bytes +
        page.APPLE_PAGE_SIZE * 64;
    var arena = try arena_mod.WiredArena.init(arena_bytes, makeTestBudget(arena_bytes));
    defer arena.deinit();

    var prefix_cache = try PrefixCache.init(&arena, cache_spec);
    var gdn_snap_pool = try hybrid.GdnSnapshotPool.init(&arena, snap_spec);
    var slot_gdn_arenas: [num_reqs]hybrid.GdnStateArena = undefined;
    for (0..num_reqs) |r| {
        slot_gdn_arenas[r] = try hybrid.GdnStateArena.init(&arena, gdn_spec);
    }
    var slot_conv_raw: [num_reqs][]u8 = undefined;
    for (0..num_reqs) |r| {
        slot_conv_raw[r] = try arena.allocAligned(conv_layer_stride_bytes * gdn_spec.num_gdn_layers, 64);
        @memset(slot_conv_raw[r], 0);
    }

    // Allocate IOSurface KvTiles:
    // - 2 shared read-only full chunks (`[0..512)`, `[512..1024)`)
    // - 1 cached read-only partial tail tile (`[1024..1200)`)
    // - 4 private per-request CoW tail tiles (one per concurrent request!)
    var shared_k: [full_chunks]metal.KvTile = undefined;
    var shared_v: [full_chunks]metal.KvTile = undefined;
    var shared_created: usize = 0;
    defer {
        for (0..shared_created) |i| {
            shared_v[i].close();
            shared_k[i].close();
        }
    }
    for (0..full_chunks) |c| {
        shared_k[c] = try metal.KvTile.init(device, head_dim, rows, .{});
        errdefer shared_k[c].close();
        shared_v[c] = try metal.KvTile.init(device, head_dim, rows, .{});
        shared_created += 1;
    }

    const cached_tail_k = try metal.KvTile.init(device, head_dim, rows, .{});
    defer cached_tail_k.close();
    const cached_tail_v = try metal.KvTile.init(device, head_dim, rows, .{});
    defer cached_tail_v.close();

    var priv_tail_k: [num_reqs]metal.KvTile = undefined;
    var priv_tail_v: [num_reqs]metal.KvTile = undefined;
    var priv_created: usize = 0;
    defer {
        for (0..priv_created) |i| {
            priv_tail_v[i].close();
            priv_tail_k[i].close();
        }
    }
    for (0..num_reqs) |r| {
        priv_tail_k[r] = try metal.KvTile.init(device, head_dim, rows, .{});
        errdefer priv_tail_k[r].close();
        priv_tail_v[r] = try metal.KvTile.init(device, head_dim, rows, .{});
        priv_created += 1;
    }

    // Populate deterministic prefix K/V into the 2 shared full tiles + cached 176-tok tail tile.
    var rng = std.Random.DefaultPrng.init(0x65_00_08_01);
    const rand = rng.random();
    const k_prefix_ref = try gpa.alloc(f16, @as(usize, prefix_len) * num_kv_heads * head_dim);
    defer gpa.free(k_prefix_ref);
    const v_prefix_ref = try gpa.alloc(f16, @as(usize, prefix_len) * num_kv_heads * head_dim);
    defer gpa.free(v_prefix_ref);
    for (k_prefix_ref, v_prefix_ref) |*kp, *vp| {
        kp.* = @floatCast((rand.float(f32) - 0.5) * 1.2);
        vp.* = @floatCast((rand.float(f32) - 0.5) * 1.0);
    }

    for (0..full_chunks) |c| {
        var kg = try shared_k[c].lock(false);
        errdefer kg.release();
        var vg = try shared_v[c].lock(false);
        errdefer vg.release();
        const stride = kg.bytes_per_row / @sizeOf(f16);
        const km: []f16 = @alignCast(std.mem.bytesAsSlice(f16, kg.bytes));
        const vm: []f16 = @alignCast(std.mem.bytesAsSlice(f16, vg.bytes));
        for (0..chunk_tokens) |s| {
            const pos = c * chunk_tokens + s;
            for (0..num_kv_heads) |kvh| {
                const src = (pos * num_kv_heads + kvh) * head_dim;
                const dst = (kvh * chunk_tokens + s) * stride;
                @memcpy(km[dst .. dst + head_dim], k_prefix_ref[src .. src + head_dim]);
                @memcpy(vm[dst .. dst + head_dim], v_prefix_ref[src .. src + head_dim]);
            }
        }
        try vg.unlock();
        try kg.unlock();
    }
    {
        var kg = try cached_tail_k.lock(false);
        errdefer kg.release();
        var vg = try cached_tail_v.lock(false);
        errdefer vg.release();
        const stride = kg.bytes_per_row / @sizeOf(f16);
        const km: []f16 = @alignCast(std.mem.bytesAsSlice(f16, kg.bytes));
        const vm: []f16 = @alignCast(std.mem.bytesAsSlice(f16, vg.bytes));
        @memset(km, @floatCast(0.0));
        @memset(vm, @floatCast(0.0));
        for (0..tail_tokens) |s| {
            const pos = full_chunks * chunk_tokens + s;
            for (0..num_kv_heads) |kvh| {
                const src = (pos * num_kv_heads + kvh) * head_dim;
                const dst = (kvh * chunk_tokens + s) * stride;
                @memcpy(km[dst .. dst + head_dim], k_prefix_ref[src .. src + head_dim]);
                @memcpy(vm[dst .. dst + head_dim], v_prefix_ref[src .. src + head_dim]);
            }
        }
        try vg.unlock();
        try kg.unlock();
    }

    // Seed Request 0's committed GDN recurrent state + conv1d history at token 1,200.
    for (0..gdn_spec.num_gdn_layers) |l| {
        const s_slice = try slot_gdn_arenas[0].committedLayerSlice(@intCast(l));
        for (s_slice, 0..) |*x, idx| {
            x.* = @as(f32, @floatFromInt(@as(i32, @intCast((idx + l * 31) % 19)) - 9)) * 0.05;
        }
        const c_start = l * conv_layer_stride_bytes;
        const c_slice: []f16 = @alignCast(std.mem.bytesAsSlice(
            f16,
            slot_conv_raw[0][c_start .. c_start + @as(usize, conv_f16_per_layer) * @sizeOf(f16)],
        ));
        for (c_slice, 0..) |*x, idx| {
            x.* = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast((idx + l * 13) % 17)) - 8)) * 0.0625);
        }
    }

    // Carve batched decode buffers from `WiredArena` before hot-path prefix ops.
    const batch_q_raw = try arena.allocAligned(amf.align64(@as(usize, num_reqs) * num_q_heads * head_dim * @sizeOf(f16)), 64);
    const batch_k_raw = try arena.allocAligned(amf.align64(@as(usize, num_reqs) * num_kv_heads * head_dim * @sizeOf(f16)), 64);
    const batch_v_raw = try arena.allocAligned(amf.align64(@as(usize, num_reqs) * num_kv_heads * head_dim * @sizeOf(f16)), 64);
    const batch_out_raw = try arena.allocAligned(amf.align64(@as(usize, num_reqs) * num_q_heads * head_dim * @sizeOf(f16)), 64);
    const batch_write_slots_raw = try arena.allocAligned(amf.align64(@as(usize, num_reqs) * @sizeOf(attention.BatchKvWriteSlot)), 64);
    const batch_attn_slots_raw = try arena.allocAligned(amf.align64(@as(usize, num_reqs) * @sizeOf(attention.BatchGqaSlotDesc)), 64);
    const batch_tile_table_raw = try arena.allocAligned(amf.align64(@as(usize, num_reqs) * (full_chunks + 1) * @sizeOf(attention.KvTilePair)), 64);
    const batch_scratch_raw = try arena.allocAligned(scratch_layout.total_bytes, 64);

    // GDN batched decode buffers (1 step on layer 0 for all 4 concurrent slots).
    const gdn_q_raw = try arena.allocAligned(amf.align64(@as(usize, num_reqs) * gdn_num_k_heads * gdn_spec.head_k_dim * @sizeOf(f16)), 64);
    const gdn_k_raw = try arena.allocAligned(amf.align64(@as(usize, num_reqs) * gdn_num_k_heads * gdn_spec.head_k_dim * @sizeOf(f16)), 64);
    const gdn_v_raw = try arena.allocAligned(amf.align64(@as(usize, num_reqs) * gdn_spec.num_v_heads * gdn_spec.head_v_dim * @sizeOf(f16)), 64);
    const gdn_g_raw = try arena.allocAligned(amf.align64(@as(usize, num_reqs) * gdn_spec.num_v_heads * @sizeOf(f32)), 64);
    const gdn_b_raw = try arena.allocAligned(amf.align64(@as(usize, num_reqs) * gdn_spec.num_v_heads * @sizeOf(f32)), 64);
    const gdn_out_raw = try arena.allocAligned(amf.align64(@as(usize, num_reqs) * gdn_spec.num_v_heads * gdn_spec.head_v_dim * @sizeOf(f16)), 64);
    const gdn_slots_raw = try arena.allocAligned(amf.align64(@as(usize, num_reqs) * @sizeOf(u32)), 64);

    const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
    defer gpu_buf.close();
    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();
    try compute.makeResident(&.{gpu_buf});
    try compute.makeTilesResident(&shared_k);
    try compute.makeTilesResident(&shared_v);
    try compute.makeTilesResident(&.{ cached_tail_k, cached_tail_v });
    try compute.makeTilesResident(&priv_tail_k);
    try compute.makeTilesResident(&priv_tail_v);

    // Record remaining arena bytes: EVERYTHING from here on MUST perform ZERO allocations!
    const arena_rem_hot_start = arena.remaining();

    // 1. Insert the 1,200-token system prefix into `PrefixCache` and save the GDN snapshot on GPU.
    var prompt_tokens: [prefix_len + 4]u32 = undefined;
    for (0..prefix_len) |i| prompt_tokens[i] = @intCast(5000 + i);

    const snap_slot = try gdn_snap_pool.acquireSlot();
    const b0_res = try prefix_cache.insertFullBlock(
        NONE_ID,
        prompt_tokens[0..chunk_tokens],
        &.{KvBlockHandle.activeFromTiles(shared_k[0], shared_v[0], 0)},
        .{},
    );
    const b1_res = try prefix_cache.insertFullBlock(
        b0_res.entry_id,
        prompt_tokens[chunk_tokens .. 2 * chunk_tokens],
        &.{KvBlockHandle.activeFromTiles(shared_k[1], shared_v[1], 1)},
        .{},
    );
    const tail_res = try prefix_cache.insertTailBlock(
        b1_res.entry_id,
        prompt_tokens[2 * chunk_tokens .. prefix_len],
        &.{KvBlockHandle.activeFromTiles(cached_tail_k, cached_tail_v, 2)},
        .{ .gdn_snapshot_id = snap_slot },
    );

    // 2. Match all 4 concurrent requests sharing the 1,200-token prefix (with diverging suffixes).
    var req_full_ids: [num_reqs][4]u32 = undefined;
    var req_matches: [num_reqs]PrefixMatch = undefined;
    for (0..num_reqs) |r| {
        for (0..4) |s| prompt_tokens[prefix_len + s] = @intCast(90_000 + r * 100 + s);
        req_matches[r] = try prefix_cache.matchAndPin(&prompt_tokens, &req_full_ids[r], .{
            .require_gdn_snapshot = true,
        });
        try std.testing.expectEqual(prefix_len, req_matches[r].matched_tokens);
        try std.testing.expectEqual(full_chunks, req_matches[r].full_block_count);
        try std.testing.expectEqual(tail_tokens, req_matches[r].tail_tokens);
        try std.testing.expectEqual(tail_res.entry_id, req_matches[r].tail_entry_id);
        try std.testing.expectEqual(snap_slot, req_matches[r].gdn_snapshot_id);
        // Every request shares the exact same physical full-block IDs!
        try std.testing.expectEqual(b0_res.entry_id, req_full_ids[r][0]);
        try std.testing.expectEqual(b1_res.entry_id, req_full_ids[r][1]);
    }
    // Verify refcounts: all 4 requests pin block 0, block 1, and tail source.
    try std.testing.expectEqual(@as(u32, num_reqs), (try prefix_cache.entryInfo(b0_res.entry_id)).pin_count);
    try std.testing.expectEqual(@as(u32, num_reqs), (try prefix_cache.entryInfo(b1_res.entry_id)).pin_count);
    try std.testing.expectEqual(@as(u32, num_reqs), (try prefix_cache.entryInfo(tail_res.entry_id)).pin_count);
    // While pinned, eviction MUST fail cleanly.
    try std.testing.expectError(error.CacheFull, prefix_cache.evictLru(null));

    // Populate shared full-tile resource IDs + private tail tile into the batched KV tile table.
    const tile_table: []attention.KvTilePair = @alignCast(std.mem.bytesAsSlice(
        attention.KvTilePair,
        batch_tile_table_raw[0 .. @as(usize, num_reqs) * (full_chunks + 1) * @sizeOf(attention.KvTilePair)],
    ));
    const write_slots: []attention.BatchKvWriteSlot = @alignCast(std.mem.bytesAsSlice(
        attention.BatchKvWriteSlot,
        batch_write_slots_raw[0 .. @as(usize, num_reqs) * @sizeOf(attention.BatchKvWriteSlot)],
    ));
    const attn_slots: []attention.BatchGqaSlotDesc = @alignCast(std.mem.bytesAsSlice(
        attention.BatchGqaSlotDesc,
        batch_attn_slots_raw[0 .. @as(usize, num_reqs) * @sizeOf(attention.BatchGqaSlotDesc)],
    ));

    for (0..num_reqs) |r| {
        const base = r * (full_chunks + 1);
        var dummy_ret: [1]u32 = undefined;
        const counts = try prefix_cache.populateLayerKvTables(
            req_full_ids[r][0..req_matches[r].full_block_count],
            0,
            &dummy_ret,
            tile_table[base .. base + full_chunks],
        );
        try std.testing.expectEqual(@as(u32, 0), counts.num_retired_chunks);
        try std.testing.expectEqual(full_chunks, counts.num_active_chunks);
        // Zero duplicate KV RAM: exact same physical KvTilePair resource IDs across all 4 slots!
        try std.testing.expectEqual(shared_k[0].rgbaResourceId(), tile_table[base + 0].k_tex_id);
        try std.testing.expectEqual(shared_k[1].rgbaResourceId(), tile_table[base + 1].k_tex_id);
        // Private CoW tail tile for slot r.
        tile_table[base + 2] = attention.KvTilePair.fromTiles(priv_tail_k[r], priv_tail_v[r], .rgba16f);
        // Each slot will write its own diverging token at position 1,200 (slot_in_chunk = 176).
        write_slots[r] = attention.BatchKvWriteSlot.fromTiles(priv_tail_k[r], priv_tail_v[r], prefix_len, chunk_tokens);
        attn_slots[r] = .{
            .seq_len = prefix_len + 1,
            .tile_table_offset = @intCast(base),
            .retired_offset = 0,
            .num_retired_chunks = 0,
        };
    }

    // Fill unique diverging Q/K/V and GDN inputs per request slot.
    const batch_q: []f16 = @alignCast(std.mem.bytesAsSlice(f16, batch_q_raw[0 .. @as(usize, num_reqs) * num_q_heads * head_dim * @sizeOf(f16)]));
    const batch_k: []f16 = @alignCast(std.mem.bytesAsSlice(f16, batch_k_raw[0 .. @as(usize, num_reqs) * num_kv_heads * head_dim * @sizeOf(f16)]));
    const batch_v: []f16 = @alignCast(std.mem.bytesAsSlice(f16, batch_v_raw[0 .. @as(usize, num_reqs) * num_kv_heads * head_dim * @sizeOf(f16)]));
    const batch_out: []f16 = @alignCast(std.mem.bytesAsSlice(f16, batch_out_raw[0 .. @as(usize, num_reqs) * num_q_heads * head_dim * @sizeOf(f16)]));
    for (0..num_reqs) |r| {
        const sign: f32 = if ((r & 1) == 0) 1.0 else -1.0;
        const mag: f32 = 0.25 * @as(f32, @floatFromInt(r + 1));
        for (0..num_q_heads * head_dim) |i| {
            batch_q[r * num_q_heads * head_dim + i] = @floatCast(sign * mag * (rand.float(f32) - 0.5));
        }
        for (0..num_kv_heads * head_dim) |i| {
            batch_k[r * num_kv_heads * head_dim + i] = @floatCast(sign * mag * (rand.float(f32) - 0.5));
            batch_v[r * num_kv_heads * head_dim + i] = @floatCast(sign * mag * (rand.float(f32) - 0.5) * 4.0);
        }
    }

    const gdn_q: []f16 = @alignCast(std.mem.bytesAsSlice(f16, gdn_q_raw[0 .. @as(usize, num_reqs) * gdn_num_k_heads * gdn_spec.head_k_dim * @sizeOf(f16)]));
    const gdn_k: []f16 = @alignCast(std.mem.bytesAsSlice(f16, gdn_k_raw[0 .. @as(usize, num_reqs) * gdn_num_k_heads * gdn_spec.head_k_dim * @sizeOf(f16)]));
    const gdn_v_arr: []f16 = @alignCast(std.mem.bytesAsSlice(f16, gdn_v_raw[0 .. @as(usize, num_reqs) * gdn_spec.num_v_heads * gdn_spec.head_v_dim * @sizeOf(f16)]));
    const gdn_g: []f32 = @alignCast(std.mem.bytesAsSlice(f32, gdn_g_raw[0 .. @as(usize, num_reqs) * gdn_spec.num_v_heads * @sizeOf(f32)]));
    const gdn_b: []f32 = @alignCast(std.mem.bytesAsSlice(f32, gdn_b_raw[0 .. @as(usize, num_reqs) * gdn_spec.num_v_heads * @sizeOf(f32)]));
    const gdn_out: []f16 = @alignCast(std.mem.bytesAsSlice(f16, gdn_out_raw[0 .. @as(usize, num_reqs) * gdn_spec.num_v_heads * gdn_spec.head_v_dim * @sizeOf(f16)]));
    const gdn_slots: []u32 = @alignCast(std.mem.bytesAsSlice(u32, gdn_slots_raw[0 .. @as(usize, num_reqs) * @sizeOf(u32)]));
    for (0..num_reqs) |r| {
        gdn_slots[r] = @intCast(r);
        const mult: f32 = @floatFromInt(r + 1);
        for (0..gdn_num_k_heads * gdn_spec.head_k_dim) |i| {
            gdn_q[r * gdn_num_k_heads * gdn_spec.head_k_dim + i] = @floatCast((rand.float(f32) - 0.5) * 0.5 * mult);
            gdn_k[r * gdn_num_k_heads * gdn_spec.head_k_dim + i] = @floatCast((rand.float(f32) - 0.5) * 0.5 * mult);
        }
        for (0..gdn_spec.num_v_heads * gdn_spec.head_v_dim) |i| {
            gdn_v_arr[r * gdn_spec.num_v_heads * gdn_spec.head_v_dim + i] = @floatCast((rand.float(f32) - 0.5) * 0.8 * mult);
        }
        for (0..gdn_spec.num_v_heads) |vh| {
            gdn_g[r * gdn_spec.num_v_heads + vh] = -0.1 * mult;
            gdn_b[r * gdn_spec.num_v_heads + vh] = 0.25 * mult - 0.5;
        }
    }

    // 3. Execute on GPU in ONE frame:
    //    (a) Save Req 0's GDN+conv1d checkpoint -> snapshot pool
    //    (b) Restore snapshot -> Req 1, 2, 3's private GdnStateArenas + conv1d buffers
    //    (c) GPU CoW copy 176-tok partial tail from cached_tail -> priv_tail[0..3]
    //    (d) Batched QK-Norm+RoPE+KV-Write for all 4 diverging tokens at pos=1,200
    //    (e) Batched GQA decode attention over [2 shared full tiles + private CoW tail]
    //    (f) Batched GDN decode on layer 0 across all 4 independent state arenas
    {
        var frame = try compute.beginFrame(5000);
        errdefer frame.abort();

        try PrefixCache.encodeSaveGdnSnapshot(
            hybrid_pipes,
            &frame,
            &slot_gdn_arenas[0],
            gpu_buf,
            .{ .buffer = gpu_buf, .offset = arena.offsetOf(slot_conv_raw[0].ptr) },
            conv_layer_stride_f16,
            &gdn_snap_pool,
            gpu_buf,
            snap_slot,
            false,
        );
        for (1..num_reqs) |r| {
            try PrefixCache.encodeRestoreGdnSnapshot(
                hybrid_pipes,
                &frame,
                &gdn_snap_pool,
                gpu_buf,
                snap_slot,
                &slot_gdn_arenas[r],
                gpu_buf,
                .{ .buffer = gpu_buf, .offset = arena.offsetOf(slot_conv_raw[r].ptr) },
                conv_layer_stride_f16,
                r == 1, // barrier after save before first restore
            );
        }
        for (0..num_reqs) |r| {
            try PrefixCache.encodeTailCowCopy(
                attn_pipes,
                &frame,
                cached_tail_k,
                cached_tail_v,
                priv_tail_k[r],
                priv_tail_v[r],
                num_kv_heads,
                head_dim,
                chunk_tokens,
                tail_tokens,
                false,
            );
        }
        try attn_pipes.encodeBatchQkNormRopeKvWrite(&frame, .{
            .q = .{ .buffer = gpu_buf, .offset = arena.offsetOf(batch_q_raw.ptr) },
            .k = .{ .buffer = gpu_buf, .offset = arena.offsetOf(batch_k_raw.ptr) },
            .v = .{ .buffer = gpu_buf, .offset = arena.offsetOf(batch_v_raw.ptr) },
            .slots = .{ .buffer = gpu_buf, .offset = arena.offsetOf(batch_write_slots_raw.ptr) },
            .batch_size = num_reqs,
            .num_q_heads = num_q_heads,
            .num_kv_heads = num_kv_heads,
            .head_dim = head_dim,
            .chunk_tokens = chunk_tokens,
        }, true);
        try attn_pipes.encodeBatchGqaDecodeAttn(&frame, .{
            .q = .{ .buffer = gpu_buf, .offset = arena.offsetOf(batch_q_raw.ptr) },
            .kv_tiles = .{ .buffer = gpu_buf, .offset = arena.offsetOf(batch_tile_table_raw.ptr) },
            .slots = .{ .buffer = gpu_buf, .offset = arena.offsetOf(batch_attn_slots_raw.ptr) },
            .out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(batch_out_raw.ptr) },
            .scratch = .{ .buffer = gpu_buf, .offset = arena.offsetOf(batch_scratch_raw.ptr) },
            .batch_size = num_reqs,
            .num_q_heads = num_q_heads,
            .num_kv_heads = num_kv_heads,
            .head_dim = head_dim,
            .max_seq_len = prefix_len + 1,
            .chunk_tokens = chunk_tokens,
        }, true);

        const gdn_slot_stride_bytes = slot_gdn_arenas[1].base_offset - slot_gdn_arenas[0].base_offset;
        try hybrid_pipes.encodeGdnBatchDecode(&frame, .{
            .q = .{ .buffer = gpu_buf, .offset = arena.offsetOf(gdn_q_raw.ptr) },
            .k = .{ .buffer = gpu_buf, .offset = arena.offsetOf(gdn_k_raw.ptr) },
            .v = .{ .buffer = gpu_buf, .offset = arena.offsetOf(gdn_v_raw.ptr) },
            .g = .{ .buffer = gpu_buf, .offset = arena.offsetOf(gdn_g_raw.ptr) },
            .beta = .{ .buffer = gpu_buf, .offset = arena.offsetOf(gdn_b_raw.ptr) },
            .state_in_pool = try slot_gdn_arenas[0].committedBinding(gpu_buf, 0),
            .state_out_pool = try slot_gdn_arenas[0].committedBinding(gpu_buf, 0),
            .out = .{ .buffer = gpu_buf, .offset = arena.offsetOf(gdn_out_raw.ptr) },
            .state_in_slots = .{ .buffer = gpu_buf, .offset = arena.offsetOf(gdn_slots_raw.ptr) },
            .batch_size = num_reqs,
            .num_k_heads = gdn_num_k_heads,
            .num_v_heads = gdn_spec.num_v_heads,
            .head_k_dim = gdn_spec.head_k_dim,
            .head_v_dim = gdn_spec.head_v_dim,
            .slot_stride_f32 = @intCast(gdn_slot_stride_bytes / @sizeOf(f32)),
        }, true);

        const sig = try frame.commit();
        try compute.wait(sig, 5000);
    }

    // 4. Verify GPU CoW & GDN mutation isolation:
    //    (a) Cached tail tile row 176 (written by all 4 slots into priv_tail) MUST still be 0.0!
    {
        var kg = try cached_tail_k.lock(true);
        errdefer kg.release();
        const stride = kg.bytes_per_row / @sizeOf(f16);
        const km: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, kg.bytes));
        for (0..num_kv_heads) |kvh| {
            const untouched = km[(kvh * chunk_tokens + tail_tokens) * stride ..][0..head_dim];
            for (untouched) |x| try std.testing.expectEqual(@as(f16, @floatCast(0.0)), x);
        }
        try kg.unlock();
    }
    //    (b) Each private tail tile matches cached_tail[0..176) bit-for-bit AND has unique row 176!
    for (0..num_reqs) |r| {
        var kg = try priv_tail_k[r].lock(true);
        errdefer kg.release();
        var vg = try priv_tail_v[r].lock(true);
        errdefer vg.release();
        const stride = kg.bytes_per_row / @sizeOf(f16);
        const km: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, kg.bytes));
        const vm: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, vg.bytes));
        for (0..tail_tokens) |s| {
            const pos = full_chunks * chunk_tokens + s;
            for (0..num_kv_heads) |kvh| {
                const ref_off = (pos * num_kv_heads + kvh) * head_dim;
                const dst_off = (kvh * chunk_tokens + s) * stride;
                try std.testing.expectEqualSlices(f16, k_prefix_ref[ref_off .. ref_off + head_dim], km[dst_off .. dst_off + head_dim]);
                try std.testing.expectEqualSlices(f16, v_prefix_ref[ref_off .. ref_off + head_dim], vm[dst_off .. dst_off + head_dim]);
            }
        }
        // Row 176 V must match this request's unique V token!
        for (0..num_kv_heads) |kvh| {
            const dst_off = (kvh * chunk_tokens + tail_tokens) * stride;
            const exp_v = batch_v[(r * num_kv_heads + kvh) * head_dim ..][0..head_dim];
            try std.testing.expectEqualSlices(f16, exp_v, vm[dst_off .. dst_off + head_dim]);
        }
        try vg.unlock();
        try kg.unlock();
    }
    //    (c) GDN layer 1 (not stepped) and conv1d histories across Req 1..3 match Req 0 bit-for-bit,
    //        while GDN layer 0 (stepped with distinct tokens) diverged cleanly across all 4 slots!
    const l1_ref = try slot_gdn_arenas[0].committedLayerSlice(1);
    for (1..num_reqs) |r| {
        try std.testing.expectEqualSlices(f32, l1_ref, try slot_gdn_arenas[r].committedLayerSlice(1));
        try std.testing.expectEqualSlices(u8, slot_conv_raw[0], slot_conv_raw[r]);
        const l0_0 = try slot_gdn_arenas[0].committedLayerSlice(0);
        const l0_r = try slot_gdn_arenas[r].committedLayerSlice(0);
        var diff_count: usize = 0;
        for (l0_0, l0_r) |a, b| {
            if (@abs(a - b) > 1e-4) diff_count += 1;
        }
        try std.testing.expect(diff_count > 100);
    }
    //    (d) Verify GQA attention outputs match CPU reference for all 4 requests!
    const k_req_ref = try gpa.alloc(f16, @as(usize, prefix_len + 1) * num_kv_heads * head_dim);
    defer gpa.free(k_req_ref);
    const v_req_ref = try gpa.alloc(f16, @as(usize, prefix_len + 1) * num_kv_heads * head_dim);
    defer gpa.free(v_req_ref);
    @memcpy(k_req_ref[0..k_prefix_ref.len], k_prefix_ref);
    @memcpy(v_req_ref[0..v_prefix_ref.len], v_prefix_ref);
    const out_ref = try gpa.alloc(f32, num_q_heads * head_dim);
    defer gpa.free(out_ref);
    for (0..num_reqs) |r| {
        // Read rotated K at pos=1,200 directly from priv_tail_k[r] for exact attention ref check.
        var kg = try priv_tail_k[r].lock(true);
        errdefer kg.release();
        const stride = kg.bytes_per_row / @sizeOf(f16);
        const km: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, kg.bytes));
        for (0..num_kv_heads) |kvh| {
            const dst_off = (@as(usize, prefix_len) * num_kv_heads + kvh) * head_dim;
            @memcpy(k_req_ref[dst_off .. dst_off + head_dim], km[(kvh * chunk_tokens + tail_tokens) * stride ..][0..head_dim]);
            const v_src = (r * num_kv_heads + kvh) * head_dim;
            @memcpy(v_req_ref[dst_off .. dst_off + head_dim], batch_v[v_src .. v_src + head_dim]);
        }
        try kg.unlock();

        try attention.gqaDecodeCpuRef(
            gpa,
            batch_q[r * num_q_heads * head_dim ..][0 .. num_q_heads * head_dim],
            k_req_ref,
            v_req_ref,
            out_ref,
            num_q_heads,
            num_kv_heads,
            head_dim,
            prefix_len + 1,
            1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))),
        );
        for (out_ref, 0..) |exp, i| {
            const got: f32 = @floatCast(batch_out[r * num_q_heads * head_dim + i]);
            try std.testing.expectApproxEqAbs(exp, got, 2e-3);
        }
    }
    // Check that GDN outputs are finite and distinct across the 4 diverging slots.
    for (gdn_out) |v_h| try std.testing.expect(std.math.isFinite(@as(f32, @floatCast(v_h))));
    var gdn_diff: f32 = 0.0;
    const gdn_stride_out = gdn_spec.num_v_heads * gdn_spec.head_v_dim;
    for (0..gdn_stride_out) |i| {
        gdn_diff = @max(gdn_diff, @abs(@as(f32, @floatCast(gdn_out[i])) - @as(f32, @floatCast(gdn_out[gdn_stride_out + i]))));
    }
    try std.testing.expect(gdn_diff > 1e-2);

    // 5. Release tail pins first (safe immediately after GPU CoW copy finished),
    //    then release all 4 request leases, then evict in O(1) LRU order and reclaim GDN snapshot.
    for (0..num_reqs) |r| {
        try prefix_cache.releaseTailPin(&req_matches[r]);
    }
    try std.testing.expectEqual(@as(u32, 0), (try prefix_cache.entryInfo(tail_res.entry_id)).pin_count);
    // Tail is now evictable even while blocks 0 & 1 are still pinned by all 4 requests!
    var ev_h: [1]KvBlockHandle = undefined;
    const ev_tail = try prefix_cache.evictLru(&ev_h);
    try std.testing.expectEqual(tail_res.entry_id, ev_tail.entry_id);
    try std.testing.expectEqual(snap_slot, ev_tail.gdn_snapshot_id);
    try gdn_snap_pool.releaseSlot(ev_tail.gdn_snapshot_id);
    try std.testing.expectEqual(@as(u32, 0), gdn_snap_pool.activeCount());

    // Blocks 0 & 1 remain protected while all 4 requests still hold leases.
    try std.testing.expectError(error.CacheFull, prefix_cache.evictLru(&ev_h));
    for (0..num_reqs) |r| {
        try prefix_cache.releaseMatch(&req_matches[r], &req_full_ids[r]);
    }
    try std.testing.expectEqual(@as(u32, 0), prefix_cache.pinnedEntries());
    const ev_b1 = try prefix_cache.evictLru(&ev_h);
    try std.testing.expectEqual(b1_res.entry_id, ev_b1.entry_id);
    const ev_b0 = try prefix_cache.evictLru(&ev_h);
    try std.testing.expectEqual(b0_res.entry_id, ev_b0.entry_id);
    try std.testing.expectEqual(@as(u32, 0), prefix_cache.liveEntries());

    // Zero hot-path arena or heap allocations occurred in PrefixCache / GdnSnapshotPool!
    try std.testing.expectEqual(arena_rem_hot_start, arena.remaining());
}

test "C15 PrefixCache evict_first priority and intra-tile ChatML message-boundary CoW hits" {
    const spec: PrefixCacheSpec = .{
        .max_entries = 6,
        .block_tokens = 16,
        .num_kv_layers = 2,
    };
    const need = try PrefixCache.requiredBytes(spec);
    var arena = try arena_mod.WiredArena.init(need, makeTestBudget(need));
    defer arena.deinit();

    var cache = try PrefixCache.init(&arena, spec);

    const makeHandles = struct {
        fn build(base: u32) [2]KvBlockHandle {
            return .{
                .{ .k_tex_id = base * 10 + 1, .v_tex_id = base * 10 + 2, .pool_slot_id = base, .storage = .active_tile },
                .{ .k_tex_id = base * 10 + 3, .v_tex_id = base * 10 + 4, .pool_slot_id = base + 100, .storage = .active_tile },
            };
        }
    }.build;

    // 1. Insert two older canonical full-tile blocks (`evict_first = false`).
    var blk_old_a: [16]u32 = undefined;
    var blk_old_b: [16]u32 = undefined;
    for (0..16) |i| {
        blk_old_a[i] = @intCast(100 + i);
        blk_old_b[i] = @intCast(200 + i);
    }
    const old_a = try cache.insertFullBlock(NONE_ID, &blk_old_a, &makeHandles(1), .{ .gdn_snapshot_id = 1 });
    const old_b = try cache.insertFullBlock(NONE_ID, &blk_old_b, &makeHandles(2), .{ .gdn_snapshot_id = 2 });

    // 2. Insert a multi-turn ChatML prefix: 2 canonical full blocks (32 tok) +
    //    an intra-tile message-boundary checkpoint at token 39 (+7 tokens, `evict_first = true`).
    var chat_b0: [16]u32 = undefined;
    var chat_b1: [16]u32 = undefined;
    for (0..16) |i| {
        chat_b0[i] = @intCast(1000 + i);
        chat_b1[i] = @intCast(1100 + i);
    }
    const boundary_tail = [_]u32{ 1200, 1201, 1202, 1203, 1204, 1205, 1206 };
    const root_boundary = [_]u32{ 500, 501, 502, 503, 504 };

    const cb0 = try cache.insertFullBlock(NONE_ID, &chat_b0, &makeHandles(3), .{ .gdn_snapshot_id = 3 });
    const cb1 = try cache.insertFullBlock(cb0.entry_id, &chat_b1, &makeHandles(4), .{ .gdn_snapshot_id = 4 });
    const msg_ckpt = try cache.insertBoundaryCheckpoint(cb1.entry_id, &boundary_tail, &makeHandles(5), .{
        .gdn_snapshot_id = 55,
    });
    try std.testing.expect((try cache.entryInfo(msg_ckpt.entry_id)).evict_first);

    // Also insert an arbitrary `< block_tokens` root message-boundary checkpoint (`evict_first = true`).
    const root_ckpt = try cache.insertTailBlock(NONE_ID, &root_boundary, &makeHandles(6), .{
        .gdn_snapshot_id = 66,
        .evict_first = true,
    });
    try std.testing.expect((try cache.entryInfo(root_ckpt.entry_id)).evict_first);
    try std.testing.expectEqual(@as(u32, 6), cache.liveEntries());

    // 3. Edit the final user message after token 39: prompt shares [0..39] and diverges at [39..46].
    var edited_prompt: [46]u32 = undefined;
    @memcpy(edited_prompt[0..16], &chat_b0);
    @memcpy(edited_prompt[16..32], &chat_b1);
    @memcpy(edited_prompt[32..39], &boundary_tail);
    for (39..46) |i| edited_prompt[i] = @intCast(9000 + i);

    var full_ids: [4]u32 = undefined;
    var m1 = try cache.matchAndPin(&edited_prompt, &full_ids, .{
        .max_match_tokens = edited_prompt.len - 1,
        .allow_partial_tail = true,
        .require_gdn_snapshot = true,
    });
    try std.testing.expectEqual(@as(u32, 39), m1.matched_tokens);
    try std.testing.expectEqual(@as(u32, 2), m1.full_block_count);
    try std.testing.expectEqual(msg_ckpt.entry_id, m1.tail_entry_id);
    try std.testing.expectEqual(@as(u32, 7), m1.tail_tokens);
    try std.testing.expectEqual(@as(u32, 55), m1.gdn_snapshot_id);
    try std.testing.expectEqual(@as(u64, 1), cache.boundary_hits);

    // Verify pinning protects `msg_ckpt` while `m1` holds its ref, so `evictLru`
    // evicts the OTHER unpinned `evict_first` entry (`root_ckpt`) BEFORE older canonical tiles!
    var ev_buf: [2]KvBlockHandle = undefined;
    const ev1 = try cache.evictLru(&ev_buf);
    try std.testing.expectEqual(root_ckpt.entry_id, ev1.entry_id);
    try std.testing.expect(ev1.evict_first);
    try std.testing.expectEqual(@as(u64, 1), cache.evict_first_evictions);

    // Re-insert `root_ckpt` and test exact token-slice verification on a 1-token pre-boundary mismatch.
    const root_ckpt2 = try cache.insertTailBlock(NONE_ID, &root_boundary, &makeHandles(7), .{
        .gdn_snapshot_id = 77,
        .evict_first = true,
    });
    const mismatch_prompt = [_]u32{ 500, 501, 9999, 503, 504, 505 };
    var m_mis = try cache.matchAndPin(&mismatch_prompt, &full_ids, .{
        .max_match_tokens = mismatch_prompt.len - 1,
        .allow_partial_tail = true,
        .require_gdn_snapshot = true,
    });
    try std.testing.expectEqual(@as(u32, 0), m_mis.matched_tokens);
    try cache.releaseMatch(&m_mis, &full_ids);

    // Exact root boundary hit increments `boundary_hits` again and restores GDN snapshot 77.
    const root_edited = [_]u32{ 500, 501, 502, 503, 504, 8888, 8889 };
    var m_root = try cache.matchAndPin(&root_edited, &full_ids, .{
        .max_match_tokens = root_edited.len - 1,
        .allow_partial_tail = true,
        .require_gdn_snapshot = true,
    });
    try std.testing.expectEqual(@as(u32, 5), m_root.matched_tokens);
    try std.testing.expectEqual(root_ckpt2.entry_id, m_root.tail_entry_id);
    try std.testing.expectEqual(@as(u32, 77), m_root.gdn_snapshot_id);
    try std.testing.expectEqual(@as(u64, 2), cache.boundary_hits);

    // Release `m1` first (becomes LRU in `evict_first` tier), then `m_root` (MRU in `evict_first` tier).
    try cache.releaseMatch(&m1, &full_ids);
    try cache.releaseMatch(&m_root, &full_ids);
    try std.testing.expectEqual(@as(u32, 0), cache.pinnedEntries());

    // Even though `old_a` and `old_b` are much older canonical entries, BOTH `evict_first`
    // entries (`msg_ckpt` LRU, then `root_ckpt2` MRU) MUST evict BEFORE `old_a` / `old_b`!
    const ev2 = try cache.evictLru(&ev_buf);
    try std.testing.expectEqual(msg_ckpt.entry_id, ev2.entry_id);
    try std.testing.expect(ev2.evict_first);

    const ev3 = try cache.evictLru(&ev_buf);
    try std.testing.expectEqual(root_ckpt2.entry_id, ev3.entry_id);
    try std.testing.expect(ev3.evict_first);
    try std.testing.expectEqual(@as(u64, 3), cache.evict_first_evictions);

    // Now the `evict_first` tier is empty: canonical LRU evicts `old_a` first, NOT fresh `cb1`!
    const ev4 = try cache.evictLru(&ev_buf);
    try std.testing.expectEqual(old_a.entry_id, ev4.entry_id);
    try std.testing.expect(!ev4.evict_first);

    const ev5 = try cache.evictLru(&ev_buf);
    try std.testing.expectEqual(old_b.entry_id, ev5.entry_id);
    try std.testing.expect(!ev5.evict_first);

    // Promoting an `evict_first` entry to canonical via re-insert with `evict_first = false`.
    const promo = try cache.insertTailBlock(cb1.entry_id, &boundary_tail, &makeHandles(8), .{
        .evict_first = true,
    });
    try std.testing.expect((try cache.entryInfo(promo.entry_id)).evict_first);
    const promo2 = try cache.insertTailBlock(cb1.entry_id, &boundary_tail, &makeHandles(8), .{
        .evict_first = false,
    });
    try std.testing.expect(promo2.already_existed);
    try std.testing.expectEqual(promo.entry_id, promo2.entry_id);
    try std.testing.expect(!(try cache.entryInfo(promo.entry_id)).evict_first);
}

test "R10 DiskPrefixCache spills/reloads q8 + FP4 KV tiles and GDN snapshots across cold restart with 0 collision risk" {
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var dir_buf: [std.fs.max_path_bytes]u8 = undefined;
    const cache_dir = try std.fmt.bufPrint(&dir_buf, ".zig-cache/tmp/{s}/disk_pc", .{tmp.sub_path});

    const amf_hdr_crc: u32 = 0x1A2B_3C4D;
    const amf_dir_crc: u32 = 0x5E6F_7788;
    var tokens: [32]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(1000 + i * 7);
    const p_hash = hashPrefix(&tokens, 16);

    const num_layers: u32 = 2;
    const kv_per_layer: u32 = 4096;
    const kv_total = num_layers * kv_per_layer;
    const gdn_total: usize = 2048;

    const kv_q8_src = try gpa.alloc(u8, kv_total);
    defer gpa.free(kv_q8_src);
    for (kv_q8_src, 0..) |*b, i| b.* = @truncate((i * 31 + 17) & 0xFF);

    const gdn_src = try gpa.alloc(u8, gdn_total);
    defer gpa.free(gdn_src);
    for (gdn_src, 0..) |*b, i| b.* = @truncate((i * 13 + 5) & 0xFF);

    // 1. Spill q8 KV tiles + GDN snapshot asynchronously, exercising cooperative yield to cold reads!
    {
        var disk = try DiskPrefixCache.init(gpa, .{
            .cache_dir = cache_dir,
            .amf_header_crc32 = amf_hdr_crc,
            .amf_dir_crc32 = amf_dir_crc,
            .max_tokens_per_artifact = 64,
            .max_kv_bytes_per_artifact = kv_total,
            .max_gdn_bytes_per_artifact = gdn_total,
            .max_pending_spills = 4,
            .write_chunk_bytes = page.APPLE_PAGE_SIZE,
            .enable_async_worker = true,
        });
        defer disk.deinit(gpa);

        // Hold a decode-critical cold-expert read while enqueuing the spill so the worker yields.
        disk.enterDecodeCriticalRead();
        const queued = try disk.spillAsync(.{
            .prefix_hash = p_hash,
            .tokens = &tokens,
            .num_kv_layers = num_layers,
            .kv_format = .retired_q8,
            .quant_profile_tag = 0x0840,
            .kv_bytes_per_layer = kv_per_layer,
            .kv_bytes = kv_q8_src,
            .gdn_bytes = gdn_src,
        });
        try std.testing.expect(queued);
        var wait_ts: std.c.timespec = .{ .sec = 0, .nsec = 5 * std.time.ns_per_ms };
        _ = std.c.nanosleep(&wait_ts, null);
        // Must not have finished while decode-critical read was active!
        try std.testing.expectEqual(@as(u64, 0), disk.stats().spills_completed);
        try std.testing.expect(disk.stats().io_yields_for_cold_reads >= 1);

        disk.exitDecodeCriticalRead();
        disk.flush();

        const st = disk.stats();
        try std.testing.expectEqual(@as(u64, 1), st.spills_completed);
        try std.testing.expectEqual(@as(u64, 0), st.spills_failed);
        try std.testing.expectEqual(@as(u32, 1), try disk.countModelArtifacts());

        // Also spill an R7 MLA FP4 artifact under a different prefix hash.
        var fp4_tokens = tokens;
        fp4_tokens[31] = 99999;
        const fp4_hash = hashPrefix(&fp4_tokens, 16);
        try disk.spillSync(.{
            .prefix_hash = fp4_hash,
            .tokens = &fp4_tokens,
            .num_kv_layers = num_layers,
            .kv_format = .mla_fp4,
            .quant_profile_tag = 0x0410,
            .kv_bytes_per_layer = kv_per_layer,
            .kv_bytes = kv_q8_src,
            .gdn_bytes = null,
        });
        try std.testing.expectEqual(@as(u32, 2), try disk.countModelArtifacts());
    }

    // 2. Cold restart: brand-new DiskPrefixCache instance reloads exact q8 + GDN and FP4 bytes.
    {
        var cold_disk = try DiskPrefixCache.init(gpa, .{
            .cache_dir = cache_dir,
            .amf_header_crc32 = amf_hdr_crc,
            .amf_dir_crc32 = amf_dir_crc,
            .max_tokens_per_artifact = 64,
            .max_kv_bytes_per_artifact = kv_total,
            .max_gdn_bytes_per_artifact = gdn_total,
            .enable_async_worker = false,
        });
        defer cold_disk.deinit(gpa);

        const kv_out = try gpa.alloc(u8, kv_total);
        defer gpa.free(kv_out);
        const gdn_out = try gpa.alloc(u8, gdn_total);
        defer gpa.free(gdn_out);

        const hit = (try cold_disk.reload(p_hash, &tokens, .retired_q8, 0x0840, kv_out, gdn_out)) orelse {
            return error.TestUnexpectedResult;
        };
        try std.testing.expectEqual(kv_total, hit.kv_bytes_loaded);
        try std.testing.expectEqual(gdn_total, hit.gdn_bytes_loaded);
        try std.testing.expect(hit.has_gdn_snapshot);
        try std.testing.expectEqualSlices(u8, kv_q8_src, kv_out);
        try std.testing.expectEqualSlices(u8, gdn_src, gdn_out);

        // Reload FP4 artifact.
        var fp4_tokens = tokens;
        fp4_tokens[31] = 99999;
        const fp4_hash = hashPrefix(&fp4_tokens, 16);
        @memset(kv_out, 0);
        const fp4_hit = (try cold_disk.reload(fp4_hash, &fp4_tokens, .mla_fp4, 0x0410, kv_out, null)) orelse {
            return error.TestUnexpectedResult;
        };
        try std.testing.expectEqual(kv_total, fp4_hit.kv_bytes_loaded);
        try std.testing.expect(!fp4_hit.has_gdn_snapshot);
        try std.testing.expectEqualSlices(u8, kv_q8_src, kv_out);

        // 3. Simulated hash collision: SAME prefix_hash, 1 mutated token -> MUST reject with null!
        var colliding_tokens = tokens;
        colliding_tokens[12] ^= 1;
        const coll = try cold_disk.reload(p_hash, &colliding_tokens, .retired_q8, 0x0840, kv_out, gdn_out);
        try std.testing.expect(coll == null);
        try std.testing.expectEqual(@as(u64, 1), cold_disk.stats().reloads_rejected_collision);

        // 4. Quantization profile / format mismatch is rejected cleanly.
        try std.testing.expectError(
            error.FormatMismatch,
            cold_disk.reload(p_hash, &tokens, .mla_fp4, 0x0840, kv_out, gdn_out),
        );
        try std.testing.expectError(
            error.FormatMismatch,
            cold_disk.reload(p_hash, &tokens, .retired_q8, 0x9999, kv_out, gdn_out),
        );
    }

    // 5. Cross-model isolation: different `.amf` CRC never hits another model's persisted prefixes.
    {
        var other_model = try DiskPrefixCache.init(gpa, .{
            .cache_dir = cache_dir,
            .amf_header_crc32 = amf_hdr_crc ^ 1,
            .amf_dir_crc32 = amf_dir_crc,
            .max_tokens_per_artifact = 64,
            .max_kv_bytes_per_artifact = kv_total,
            .max_gdn_bytes_per_artifact = gdn_total,
            .enable_async_worker = false,
        });
        defer other_model.deinit(gpa);
        const kv_out = try gpa.alloc(u8, kv_total);
        defer gpa.free(kv_out);
        const miss = try other_model.reload(p_hash, &tokens, .retired_q8, 0x0840, kv_out, null);
        try std.testing.expect(miss == null);
        try std.testing.expectEqual(@as(u32, 0), try other_model.countModelArtifacts());
    }
}

test "F10 PrefixCache protects unpinned parent_id across evictLru without self-cycle" {
    const spec: PrefixCacheSpec = .{
        .max_entries = 2,
        .block_tokens = 4,
        .num_kv_layers = 2,
    };
    const need = try PrefixCache.requiredBytes(spec);
    var arena = try arena_mod.WiredArena.init(need, makeTestBudget(need));
    defer arena.deinit();

    var cache = try PrefixCache.init(&arena, spec);

    const h0 = [2]KvBlockHandle{
        .{ .k_tex_id = 1, .v_tex_id = 2, .pool_slot_id = 0, .storage = .active_tile },
        .{ .k_tex_id = 3, .v_tex_id = 4, .pool_slot_id = 1, .storage = .active_tile },
    };
    const h1 = [2]KvBlockHandle{
        .{ .k_tex_id = 5, .v_tex_id = 6, .pool_slot_id = 2, .storage = .active_tile },
        .{ .k_tex_id = 7, .v_tex_id = 8, .pool_slot_id = 3, .storage = .active_tile },
    };

    const blk0 = [_]u32{ 10, 11, 12, 13 };
    const blk1 = [_]u32{ 20, 21, 22, 23 };
    const blk2 = [_]u32{ 30, 31, 32, 33 };
    const tail1 = [_]u32{ 40, 41 };

    // Case 1: Cache full (max_entries = 2). Entry 0 is canonical unpinned leaf;
    // Entry 1 is an unpinned `evict_first` parent sitting at `evict_first_tail`.
    // Attaching a child to Entry 1 must evict Entry 0 and NEVER evict Entry 1 itself!
    const r0 = try cache.insertFullBlock(NONE_ID, &blk0, &h0, .{ .pin_on_insert = false });
    const r1 = try cache.insertFullBlock(NONE_ID, &blk1, &h1, .{ .pin_on_insert = false, .evict_first = true });
    try std.testing.expectEqual(@as(u32, 2), cache.liveEntries());
    try std.testing.expectEqual(@as(u32, 0), cache.entries[r1.entry_id].pin_count);

    const child_res = try cache.insertTailBlock(r1.entry_id, &tail1, &h1, .{
        .evict_if_full = true,
        .evict_first = true,
    });
    try std.testing.expect(child_res.entry_id != r1.entry_id);
    try std.testing.expectEqual(r0.entry_id, child_res.evicted.?.entry_id);
    try std.testing.expectEqual(r1.entry_id, cache.entries[child_res.entry_id].parent_id);
    try std.testing.expectEqual(@as(u32, 0), cache.entries[r1.entry_id].pin_count);
    try std.testing.expectEqual(@as(u32, 1), cache.entries[r1.entry_id].child_count);

    // Case 2: Pin the child so `r1` is the only ancestor and there are NO other
    // unpinned leaves; inserting another full block under a fresh unpinned leaf
    // returns `CacheFull` without creating a self-parented cycle.
    cache.reset();
    const p_pinned = try cache.insertFullBlock(NONE_ID, &blk0, &h0, .{ .pin_on_insert = true });
    const p_parent = try cache.insertFullBlock(NONE_ID, &blk1, &h1, .{ .pin_on_insert = false, .evict_first = true });
    try std.testing.expectError(
        error.CacheFull,
        cache.insertFullBlock(p_parent.entry_id, &blk2, &h0, .{ .evict_if_full = true }),
    );
    try std.testing.expectEqual(@as(u32, 0), cache.entries[p_parent.entry_id].pin_count);
    try std.testing.expectEqual(@as(u32, 0), cache.entries[p_parent.entry_id].child_count);
    try std.testing.expect(cache.entries[p_parent.entry_id].parent_id != p_parent.entry_id);
    try cache.unpin(p_pinned.entry_id);
}

test "L9 PrefixCache matchAndPin skips CoW tail source if any attention layer is retired" {
    const spec: PrefixCacheSpec = .{
        .max_entries = 4,
        .block_tokens = 4,
        .num_kv_layers = 2,
    };
    const need = try PrefixCache.requiredBytes(spec);
    var arena = try arena_mod.WiredArena.init(need, makeTestBudget(need));
    defer arena.deinit();

    var cache = try PrefixCache.init(&arena, spec);
    const full_toks = [_]u32{ 1, 2, 3, 4 };
    const tail_toks = [_]u32{ 5, 6, 7 };
    const handles = [2]KvBlockHandle{
        .{ .k_tex_id = 10, .v_tex_id = 11, .pool_slot_id = 0, .storage = .active_tile },
        .{ .k_tex_id = 12, .v_tex_id = 13, .pool_slot_id = 1, .storage = .active_tile },
    };

    const f_res = try cache.insertFullBlock(NONE_ID, &full_toks, &handles, .{});
    const t_res = try cache.insertTailBlock(f_res.entry_id, &tail_toks, &handles, .{});
    // Retire ONLY layer 1 on the tail entry (layer 0 stays `.active_tile`).
    cache.kv_handles[@as(usize, t_res.entry_id) * 2 + 1] = KvBlockHandle.retiredQ8(9);

    const prompt = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var full_ids: [4]u32 = undefined;
    var m = try cache.matchAndPin(&prompt, &full_ids, .{ .allow_partial_tail = true });

    try std.testing.expectEqual(@as(usize, 1), m.full_block_count);
    try std.testing.expectEqual(NONE_ID, m.tail_entry_id);
    try std.testing.expectEqual(@as(u32, 0), m.tail_tokens);
    try cache.releaseMatch(&m, &full_ids);
}

test "F19.4 DiskPrefixCache.reload rejects u64-overflowing header offsets cleanly" {
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var dir_buf: [std.fs.max_path_bytes]u8 = undefined;
    const cache_dir = try std.fmt.bufPrint(&dir_buf, ".zig-cache/tmp/{s}/overflow_pc", .{tmp.sub_path});

    const amf_hdr_crc: u32 = 0x1122_3344;
    const amf_dir_crc: u32 = 0x5566_7788;
    const tokens = [_]u32{ 11, 22, 33, 44 };
    const p_hash = hashPrefix(&tokens, 4);

    var disk = try DiskPrefixCache.init(gpa, .{
        .cache_dir = cache_dir,
        .amf_header_crc32 = amf_hdr_crc,
        .amf_dir_crc32 = amf_dir_crc,
        .max_tokens_per_artifact = 16,
        .max_kv_bytes_per_artifact = 4096,
        .enable_async_worker = false,
    });
    defer disk.deinit(gpa);

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_z = try DiskPrefixCache.formatArtifactPath(cache_dir, amf_hdr_crc, amf_dir_crc, p_hash, &path_buf);

    // Craft a valid-CRC header whose `kv_data_bytes = 0x8000_0000_0000_0000` (16 KiB aligned,
    // passes `page.alignUp`, overflows `u64` when summed with `APPLE_PAGE_SIZE + tokens_blob_bytes`).
    var hdr = std.mem.zeroes(DiskPrefixHeader);
    hdr.magic = DISK_PREFIX_MAGIC;
    hdr.version = DISK_PREFIX_VERSION;
    hdr.page_size = page.APPLE_PAGE_SIZE;
    hdr.amf_header_crc32 = amf_hdr_crc;
    hdr.amf_dir_crc32 = amf_dir_crc;
    hdr.prefix_hash = p_hash;
    hdr.token_count = tokens.len;
    hdr.num_kv_layers = 1;
    hdr.kv_format = @backingInt(DiskKvFormat.retired_q8);
    hdr.quant_profile_tag = 1;
    hdr.tokens_offset = page.APPLE_PAGE_SIZE;
    hdr.tokens_blob_bytes = page.APPLE_PAGE_SIZE;
    hdr.kv_offset = 2 * page.APPLE_PAGE_SIZE;
    hdr.kv_data_bytes = 0x8000_0000_0000_0000;
    hdr.kv_blob_bytes = 0x8000_0000_0000_0000;
    hdr.header_crc32 = hdr.computeCrc32();

    var page0: [page.APPLE_PAGE_SIZE]u8 = @splat(0);
    @memcpy(page0[0..@sizeOf(DiskPrefixHeader)], std.mem.asBytes(&hdr));
    const fd = std.c.open(path_z.ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o644));
    try std.testing.expect(fd >= 0);
    defer _ = std.c.close(fd);
    _ = std.c.pwrite(fd, &page0, page0.len, 0);

    var kv_out: [4096]u8 = undefined;
    try std.testing.expectError(
        error.CorruptHeader,
        disk.reload(p_hash, &tokens, .retired_q8, 1, &kv_out, null),
    );
}
