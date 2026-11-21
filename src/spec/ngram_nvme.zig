//! C6 16 KiB-Aligned NVMe Direct-I/O N-Gram / Suffix Table (`.ang` v1,
//! ADR-0006, ADR-0005, docs/architecture/experts-and-storage.md).
//!
//! Stores corpus- or repository-scale `2..4`-gram contexts mapped to up to
//! 8-token continuations in power-of-two 16 KiB bucket pages starting at byte
//! `16384`. At runtime, `NvmeNgramTable` opens the `.ang` file read-only with
//! `fcntl(fd, F_NOCACHE, 1)` (bypassing the macOS unified buffer cache) and
//! maintains a small preallocated 16 KiB-aligned `WiredArena` LRU page cache
//! with zero-copy asynchronous `aio.AsyncExpertReader` background prefetching
//! and zero dynamic heap allocation on the query/prefetch hot path.

const std = @import("std");
const page = @import("../mem/page.zig");
const budget_mod = @import("../mem/budget.zig");
const arena_mod = @import("../mem/arena.zig");
const sysinfo = @import("../platform/sysinfo.zig");
const vmstat = @import("../platform/vmstat.zig");
const aio = @import("../storage/aio.zig");
const sam_mod = @import("sam.zig");

/// Magic bytes for `.ang` v1 files.
pub const MAGIC: [8]u8 = .{ 'A', 'N', 'G', '0', 0, 0, 0, 0 };
pub const FORMAT_VERSION: u32 = 1;
pub const CHECKSUM_CRC32_IEEE: u32 = 1;

/// Header flag indicating `table_crc32` holds the IEEE CRC-32 of all bucket pages.
pub const FLAG_HAS_TABLE_CRC32: u32 = 1 << 0;

pub const MAX_CTX_LEN: usize = 4;
pub const MAX_CONT_LEN: usize = 8;
pub const SLOTS_PER_BUCKET: usize = page.APPLE_PAGE_SIZE / @sizeOf(NgramSlot);

const INVALID_BUCKET: u32 = std.math.maxInt(u32);
const DEFAULT_VERIFY_CHUNK_BYTES: usize = 256 * 1024; // 16 * 16 KiB

/// 128-byte `.ang` v1 header at byte 0 of page 0 (little-endian, zero implicit padding).
pub const Header = extern struct {
    magic: [8]u8,
    version: u32,
    flags: u32,
    page_size: u32,
    checksum_algo: u32,
    num_buckets: u32,
    slots_per_bucket: u32,
    min_ngram: u8,
    max_ngram: u8,
    max_cont_len: u8,
    reserved_u8: u8,
    vocab_size: u32,
    total_entries: u64,
    table_offset: u64,
    table_bytes: u64,
    table_crc32: u32,
    header_crc32: u32,
    corpus_id: [48]u8,
    reserved1: u64,

    pub fn corpusIdSlice(self: *const Header) []const u8 {
        return std.mem.sliceTo(&self.corpus_id, 0);
    }

    pub fn hasTableCrc(self: *const Header) bool {
        return (self.flags & FLAG_HAS_TABLE_CRC32) != 0;
    }
};

/// 64-byte fixed N-gram record packed 256 per 16 KiB bucket page
/// (little-endian, zero implicit padding).
pub const NgramSlot = extern struct {
    ctx_hash: u64,
    ctx_tail: [MAX_CTX_LEN]u32,
    cont_tokens: [MAX_CONT_LEN]u32,
    cont_len: u8,
    ctx_len: u8,
    hit_count: u16,
    step_prob_f16: u16,
    reserved: u16,

    pub inline fn isEmpty(self: *const NgramSlot) bool {
        return self.cont_len == 0;
    }

    pub inline fn stepProb(self: *const NgramSlot) f32 {
        const f: f16 = @bitCast(self.step_prob_f16);
        const v: f32 = @floatCast(f);
        if (!std.math.isFinite(v)) return 0.0;
        return std.math.clamp(v, 0.0, 1.0);
    }

    pub inline fn matchesContext(self: *const NgramSlot, hash: u64, ctx: []const u32) bool {
        if (self.ctx_hash != hash or self.ctx_len != ctx.len) return false;
        for (ctx, 0..) |tok, i| {
            if (self.ctx_tail[i] != tok) return false;
        }
        return true;
    }
};

comptime {
    if (@sizeOf(Header) != 128) @compileError("ANG Header must be 128 bytes");
    if (@offsetOf(Header, "header_crc32") != 68) @compileError("ANG Header.header_crc32 offset mismatch");
    if (@sizeOf(NgramSlot) != 64) @compileError("ANG NgramSlot must be 64 bytes");
    if (SLOTS_PER_BUCKET != 256) @compileError("ANG bucket page must hold exactly 256 NgramSlots");
    if (SLOTS_PER_BUCKET * @sizeOf(NgramSlot) != page.APPLE_PAGE_SIZE) {
        @compileError("ANG bucket page must equal APPLE_PAGE_SIZE");
    }
    if (DEFAULT_VERIFY_CHUNK_BYTES % page.APPLE_PAGE_SIZE != 0) {
        @compileError("DEFAULT_VERIFY_CHUNK_BYTES must be 16 KiB aligned");
    }
}

/// Deterministic 64-bit SplitMix64 context hash shared bit-for-bit with
/// `tools/src/aminfer_tools/ngram.py`.
pub fn hashContext(ctx: []const u32) u64 {
    std.debug.assert(ctx.len >= 1 and ctx.len <= MAX_CTX_LEN);
    var h: u64 = 0x9E37_79B9_7F4A_7C15 ^ (@as(u64, ctx.len) *% 0xBF58_476D_1CE4_E5B9);
    for (ctx) |tok| {
        var x: u64 = @as(u64, tok) +% 0x9E37_79B9_7F4A_7C15;
        x = (x ^ (x >> 30)) *% 0xBF58_476D_1CE4_E5B9;
        x = (x ^ (x >> 27)) *% 0x94D0_49BB_1331_11EB;
        x = x ^ (x >> 31);
        h = (h ^ x) *% 0xD1B5_4A32_D192_ED03 +% 0x9E37_79B9_7F4A_7C15;
    }
    h = (h ^ (h >> 30)) *% 0xBF58_476D_1CE4_E5B9;
    h = (h ^ (h >> 27)) *% 0x94D0_49BB_1331_11EB;
    h = h ^ (h >> 31);
    return h;
}

/// Computes the IEEE CRC-32 over the 128-byte `Header` with `header_crc32 = 0`.
pub fn computeHeaderCrc32(hdr: *const Header) u32 {
    var copy = hdr.*;
    copy.header_crc32 = 0;
    return std.hash.Crc32.hash(std.mem.asBytes(&copy));
}

pub inline fn f32ToF16Bits(v: f32) u16 {
    const clamped: f32 = if (!std.math.isFinite(v)) 0.0 else std.math.clamp(v, 0.0, 1.0);
    const h: f16 = @floatCast(clamped);
    return @bitCast(h);
}

pub const FormatError = error{
    FileTooSmall,
    MisalignedFileSize,
    InvalidMagic,
    UnsupportedVersion,
    UnsupportedPageSize,
    UnsupportedChecksum,
    HeaderChecksumMismatch,
    InvalidBucketCount,
    InvalidSlotsPerBucket,
    InvalidNgramBounds,
    InvalidTableLayout,
    TableChecksumMismatch,
    InvalidSlotRecord,
    UnsortedBucketPage,
};

pub const OpenError = FormatError || aio.EcorePoolError ||
    arena_mod.WiredArena.InitError || budget_mod.PlanError || sysinfo.Error ||
    error{InvalidCacheConfig};

pub const QueryError = aio.EngineError;

/// Result of querying `NvmeNgramTable` for a context tail continuation.
pub const QueryResult = struct {
    /// Matched context n-gram length (`1..4`, or `0` on miss).
    match_len: u32,
    /// Number of continuation tokens written to `out_cont` (`0..8`).
    draft_len: u32,
    /// Corpus frequency count of the matched record (`0` on miss).
    hit_count: u16,
    /// Conditional step probability in `[0, 1]` from the matched record.
    step_prob: f32,
    /// True when `match_len >= SEED_MIN_MATCH_LEN (4)` (ADR-0005).
    is_seed: bool,
    /// True when served from the preallocated wired LRU cache or a completed prefetch.
    cache_hit: bool,
};

/// Cumulative runtime telemetry for `NvmeNgramTable`.
pub const Stats = struct {
    queries: u64 = 0,
    hits: u64 = 0,
    seed_hits: u64 = 0,
    branch_hits: u64 = 0,
    cache_hits: u64 = 0,
    prefetch_submitted: u64 = 0,
    prefetch_hits: u64 = 0,
    nvme_reads: u64 = 0,
    bytes_read: u64 = 0,
    total_query_ns: u64 = 0,

    pub fn hitRate(self: Stats) f64 {
        if (self.queries == 0) return 0.0;
        return @as(f64, @floatFromInt(self.hits)) / @as(f64, @floatFromInt(self.queries));
    }

    pub fn meanQueryUs(self: Stats) f64 {
        if (self.queries == 0) return 0.0;
        return (@as(f64, @floatFromInt(self.total_query_ns)) / @as(f64, @floatFromInt(self.queries))) / 1000.0;
    }
};

pub const Config = struct {
    /// Number of 16 KiB wired LRU page cache slots (`1..=1024`; default `32` = `512 KiB`).
    cache_pages: usize = 32,
    /// Maximum concurrent async prefetch reads (`1..=16`; default `4`).
    max_inflight: usize = 4,
    /// Async I/O backend for background `prefetch()` (`kqueue_aio` spawns 0 extra threads at QD<=4).
    io_backend: aio.IoBackend = .kqueue_aio,
    /// Verify the optional full-table CRC-32 at open time when present in the header.
    verify_table_crc: bool = false,
    /// Timeout waiting for an in-flight prefetch page during `query()`.
    timeout_ms: u64 = 5_000,
    /// Optional explicit wired budget (null computes from host sysctl facts).
    budget_override: ?budget_mod.Budget = null,
};

pub fn validateHeader(hdr: *const Header, file_size: u64) FormatError!void {
    if (file_size < 2 * page.APPLE_PAGE_SIZE) return error.FileTooSmall;
    if (file_size % page.APPLE_PAGE_SIZE != 0) return error.MisalignedFileSize;
    if (!std.mem.eql(u8, &hdr.magic, &MAGIC)) return error.InvalidMagic;
    if (hdr.version != FORMAT_VERSION) return error.UnsupportedVersion;
    if (hdr.page_size != page.APPLE_PAGE_SIZE) return error.UnsupportedPageSize;
    if (hdr.checksum_algo != CHECKSUM_CRC32_IEEE) return error.UnsupportedChecksum;
    if (computeHeaderCrc32(hdr) != hdr.header_crc32) return error.HeaderChecksumMismatch;
    if (hdr.num_buckets == 0 or !std.math.isPowerOfTwo(hdr.num_buckets)) return error.InvalidBucketCount;
    if (hdr.slots_per_bucket != SLOTS_PER_BUCKET) return error.InvalidSlotsPerBucket;
    if (hdr.min_ngram == 0 or hdr.max_ngram == 0 or
        hdr.min_ngram > hdr.max_ngram or hdr.max_ngram > MAX_CTX_LEN or
        hdr.max_cont_len == 0 or hdr.max_cont_len > MAX_CONT_LEN)
    {
        return error.InvalidNgramBounds;
    }
    const expected_table_bytes = @as(u64, hdr.num_buckets) * page.APPLE_PAGE_SIZE;
    if (hdr.table_offset != page.APPLE_PAGE_SIZE or
        hdr.table_bytes != expected_table_bytes or
        hdr.table_offset + hdr.table_bytes != file_size)
    {
        return error.InvalidTableLayout;
    }
    const max_possible_entries = @as(u64, hdr.num_buckets) * SLOTS_PER_BUCKET;
    if (hdr.total_entries > max_possible_entries) return error.InvalidTableLayout;
}

fn preadExact(fd: std.c.fd_t, dst: []u8, offset: u64) aio.IoResultError!void {
    var done: usize = 0;
    while (done < dst.len) {
        const rc = std.c.pread(fd, dst.ptr + done, dst.len - done, @intCast(offset + done));
        if (rc < 0) {
            if (std.c.errno(rc) == .INTR) continue;
            return error.ReadFailed;
        }
        if (rc == 0) return error.ShortRead;
        done += @intCast(rc);
    }
}

/// Validates an `.ang` file on disk via `F_NOCACHE` direct reads: verifies
/// page-0 header CRC, optional full-table CRC-32, slot invariants, and
/// descending `hit_count` ordering on every bucket page.
pub fn verifyFile(allocator: std.mem.Allocator, path: [:0]const u8) OpenError!Header {
    var file = try aio.DirectFile.openReadOnly(path);
    defer file.close();

    const buf = try allocator.alignedAlloc(u8, .fromByteUnits(page.APPLE_PAGE_SIZE), DEFAULT_VERIFY_CHUNK_BYTES);
    defer allocator.free(buf);

    try preadExact(file.fd, buf[0..page.APPLE_PAGE_SIZE], 0);
    var hdr: Header = undefined;
    @memcpy(std.mem.asBytes(&hdr), buf[0..@sizeOf(Header)]);
    try validateHeader(&hdr, file.size);

    var crc = std.hash.Crc32.init();
    var counted_entries: u64 = 0;
    var bucket: u32 = 0;
    const buckets_per_chunk: u32 = DEFAULT_VERIFY_CHUNK_BYTES / page.APPLE_PAGE_SIZE;

    while (bucket < hdr.num_buckets) {
        const take_buckets: u32 = @min(buckets_per_chunk, hdr.num_buckets - bucket);
        const take_bytes: usize = @as(usize, take_buckets) * page.APPLE_PAGE_SIZE;
        const off: u64 = hdr.table_offset + @as(u64, bucket) * page.APPLE_PAGE_SIZE;
        try preadExact(file.fd, buf[0..take_bytes], off);

        if (hdr.hasTableCrc()) {
            crc.update(buf[0..take_bytes]);
        }

        for (0..take_buckets) |b_rel| {
            const page_bytes = buf[b_rel * page.APPLE_PAGE_SIZE .. (b_rel + 1) * page.APPLE_PAGE_SIZE];
            const slots: []const NgramSlot = @alignCast(std.mem.bytesAsSlice(NgramSlot, page_bytes));
            var saw_empty = false;
            var prev_hits: u16 = std.math.maxInt(u16);
            for (slots) |slot| {
                if (slot.isEmpty()) {
                    saw_empty = true;
                    continue;
                }
                if (saw_empty) return error.UnsortedBucketPage;
                if (slot.cont_len > hdr.max_cont_len or
                    slot.ctx_len < hdr.min_ngram or slot.ctx_len > hdr.max_ngram)
                {
                    return error.InvalidSlotRecord;
                }
                if (slot.hit_count > prev_hits) return error.UnsortedBucketPage;
                prev_hits = slot.hit_count;
                const expected_hash = hashContext(slot.ctx_tail[0..slot.ctx_len]);
                if (slot.ctx_hash != expected_hash) return error.InvalidSlotRecord;
                const expected_bucket: u32 = @intCast(expected_hash & @as(u64, hdr.num_buckets - 1));
                if (expected_bucket != bucket + @as(u32, @intCast(b_rel))) return error.InvalidSlotRecord;
                counted_entries += 1;
            }
        }
        bucket += take_buckets;
    }

    if (counted_entries != hdr.total_entries) return error.InvalidTableLayout;
    if (hdr.hasTableCrc() and crc.final() != hdr.table_crc32) return error.TableChecksumMismatch;
    return hdr;
}

/// Runtime direct-I/O NVMe N-gram lookup table (`C6`, ADR-0006).
///
/// All wired cache pages (`cache_pages * 16 KiB`), LRU slot metadata, and
/// asynchronous I/O handles are allocated once in `open()`. Hot-path `query()`
/// and `prefetch()` perform zero dynamic heap allocations and zero OS unified
/// buffer cache pollution (`F_NOCACHE`).
pub const NvmeNgramTable = struct {
    const SlotState = enum(u8) {
        empty = 0,
        inflight = 1,
        ready = 2,
    };

    const CacheSlot = struct {
        state: SlotState = .empty,
        bucket_idx: u32 = INVALID_BUCKET,
        last_used_tick: u64 = 0,
    };

    file: aio.DirectFile,
    header: Header,
    bucket_mask: u64,
    arena: arena_mod.WiredArena,
    slots: []CacheSlot,
    async_reader: aio.AsyncExpertReader,
    timeout_ms: u64,
    tick: u64,
    stats: Stats,

    pub fn open(
        allocator: std.mem.Allocator,
        path: [:0]const u8,
        config: Config,
    ) OpenError!NvmeNgramTable {
        if (config.cache_pages == 0 or config.cache_pages > 1024) return error.InvalidCacheConfig;
        if (config.max_inflight == 0 or config.max_inflight > 16) return error.InvalidCacheConfig;

        var file = try aio.DirectFile.openReadOnly(path);
        errdefer file.close();

        const arena_bytes = config.cache_pages * page.APPLE_PAGE_SIZE;
        const wired_budget = config.budget_override orelse blk: {
            const facts = try sysinfo.Facts.read();
            break :blk budget_mod.Budget.compute(.{
                .memory_bytes = facts.memory_bytes,
                .user_wire_limit = facts.user_wire_limit,
                .global_user_wire_limit = facts.global_user_wire_limit,
                .gpu_working_set_bytes = facts.memory_bytes,
                .configured_cap = null,
                .headroom_bytes = budget_mod.DEFAULT_HEADROOM_BYTES,
            });
        };
        try wired_budget.checkPlan(arena_bytes);

        var arena = try arena_mod.WiredArena.init(arena_bytes, wired_budget);
        errdefer arena.deinit();

        // Read page 0 header directly into wired page 0 via F_NOCACHE pread.
        const page0 = arena.bytes()[0..page.APPLE_PAGE_SIZE];
        try preadExact(file.fd, page0, 0);
        var hdr: Header = undefined;
        @memcpy(std.mem.asBytes(&hdr), page0[0..@sizeOf(Header)]);
        try validateHeader(&hdr, file.size);

        if (config.verify_table_crc and hdr.hasTableCrc()) {
            var crc = std.hash.Crc32.init();
            var off: u64 = hdr.table_offset;
            const end_off: u64 = hdr.table_offset + hdr.table_bytes;
            while (off < end_off) {
                const take_pages = @min(config.cache_pages, @as(usize, @intCast((end_off - off) / page.APPLE_PAGE_SIZE)));
                const take_bytes = take_pages * page.APPLE_PAGE_SIZE;
                try preadExact(file.fd, arena.bytes()[0..take_bytes], off);
                crc.update(arena.bytes()[0..take_bytes]);
                off += take_bytes;
            }
            if (crc.final() != hdr.table_crc32) return error.TableChecksumMismatch;
        }
        @memset(arena.bytes()[0..page.APPLE_PAGE_SIZE], 0);

        const slots = try allocator.alloc(CacheSlot, config.cache_pages);
        errdefer allocator.free(slots);
        @memset(slots, .{});

        const eff_inflight: usize = if (config.cache_pages >= 2)
            @min(config.max_inflight, config.cache_pages / 2)
        else
            1;
        var reader = try aio.AsyncExpertReader.init(allocator, .{
            .backend = config.io_backend,
            .max_inflight = eff_inflight,
            .worker_count = 1,
            .worker_qos = .utility,
            .io_policy = .important,
        });
        errdefer reader.deinit(allocator);

        return .{
            .file = file,
            .header = hdr,
            .bucket_mask = @as(u64, hdr.num_buckets - 1),
            .arena = arena,
            .slots = slots,
            .async_reader = reader,
            .timeout_ms = config.timeout_ms,
            .tick = 0,
            .stats = .{},
        };
    }

    pub fn deinit(self: *NvmeNgramTable, allocator: std.mem.Allocator) void {
        // Drain all in-flight kernel AIO / E-core pread requests BEFORE releasing
        // the wired arena or closing the F_NOCACHE fd.
        self.async_reader.deinit(allocator);
        allocator.free(self.slots);
        self.slots = &.{};
        self.arena.deinit();
        self.file.close();
    }

    pub fn totalWiredBytes(self: *const NvmeNgramTable) u64 {
        return self.arena.bytes().len;
    }

    pub fn cachePageCount(self: *const NvmeNgramTable) usize {
        return self.slots.len;
    }

    pub fn resetStats(self: *NvmeNgramTable) void {
        self.stats = .{};
    }

    pub fn clearCache(self: *NvmeNgramTable) QueryError!void {
        try self.drainPrefetches();
        for (self.slots) |*s| s.* = .{};
    }

    inline fn nextTick(self: *NvmeNgramTable) u64 {
        self.tick +%= 1;
        if (self.tick == 0) self.tick = 1;
        return self.tick;
    }

    inline fn bucketForHash(self: *const NvmeNgramTable, h: u64) u32 {
        return @intCast(h & self.bucket_mask);
    }

    inline fn pageSlice(self: *NvmeNgramTable, slot_idx: usize) []align(page.APPLE_PAGE_SIZE) u8 {
        const start = slot_idx * page.APPLE_PAGE_SIZE;
        return @alignCast(self.arena.bytes()[start .. start + page.APPLE_PAGE_SIZE]);
    }

    inline fn pageSlotsConst(self: *const NvmeNgramTable, slot_idx: usize) []const NgramSlot {
        const start = slot_idx * page.APPLE_PAGE_SIZE;
        const raw = self.arena.bytes()[start .. start + page.APPLE_PAGE_SIZE];
        return @alignCast(std.mem.bytesAsSlice(NgramSlot, raw));
    }

    fn findSlotForBucket(self: *const NvmeNgramTable, bucket_idx: u32) ?usize {
        for (self.slots, 0..) |s, i| {
            if (s.state != .empty and s.bucket_idx == bucket_idx) return i;
        }
        return null;
    }

    fn pickVictimSlot(self: *const NvmeNgramTable) ?usize {
        var best_lru: ?usize = null;
        var best_tick: u64 = std.math.maxInt(u64);
        for (self.slots, 0..) |s, i| {
            switch (s.state) {
                .empty => return i,
                .inflight => continue,
                .ready => {
                    if (s.last_used_tick < best_tick) {
                        best_tick = s.last_used_tick;
                        best_lru = i;
                    }
                },
            }
        }
        return best_lru;
    }

    fn applyCompletion(self: *NvmeNgramTable, comp: aio.Completion) QueryError!void {
        const slot_idx: usize = @intCast(comp.user_data);
        std.debug.assert(slot_idx < self.slots.len);
        const slot = &self.slots[slot_idx];
        const bytes_read = comp.check() catch |err| {
            slot.state = .empty;
            slot.bucket_idx = INVALID_BUCKET;
            return err;
        };
        if (bytes_read != page.APPLE_PAGE_SIZE) {
            slot.state = .empty;
            slot.bucket_idx = INVALID_BUCKET;
            return error.ShortRead;
        }
        slot.state = .ready;
        slot.last_used_tick = self.nextTick();
        self.stats.nvme_reads += 1;
        self.stats.bytes_read += bytes_read;
    }

    /// Reaps any already-completed background prefetches without blocking.
    pub fn reapPrefetches(self: *NvmeNgramTable) QueryError!usize {
        if (self.async_reader.inflightCount() == 0) return 0;
        var comps: [16]aio.Completion = undefined;
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

    fn drainPrefetches(self: *NvmeNgramTable) QueryError!void {
        var comps: [16]aio.Completion = undefined;
        while (self.async_reader.inflightCount() > 0) {
            const got = try self.async_reader.wait(&comps, self.timeout_ms);
            if (got == 0) return error.Timeout;
            for (comps[0..got]) |c| {
                try self.applyCompletion(c);
            }
        }
    }

    fn waitForInflightSlot(self: *NvmeNgramTable, target_slot: usize) QueryError!void {
        var comps: [16]aio.Completion = undefined;
        while (self.slots[target_slot].state == .inflight) {
            const got = try self.async_reader.wait(&comps, self.timeout_ms);
            if (got == 0) return error.Timeout;
            for (comps[0..got]) |c| {
                try self.applyCompletion(c);
            }
        }
    }

    /// Submits an asynchronous `F_NOCACHE` 16 KiB bucket page prefetch for the
    /// longest valid suffix of `tail_tokens` directly into a wired LRU slot.
    /// Zero hot-path heap allocation; returns `true` if a new NVMe read was queued.
    pub fn prefetch(self: *NvmeNgramTable, tail_tokens: []const u32) QueryError!bool {
        _ = try self.reapPrefetches();
        if (tail_tokens.len < self.header.min_ngram) return false;
        if (self.async_reader.availableSlots() == 0) return false;

        const ctx_len: usize = @min(tail_tokens.len, @as(usize, self.header.max_ngram));
        const ctx = tail_tokens[tail_tokens.len - ctx_len .. tail_tokens.len];
        const h = hashContext(ctx);
        const bucket_idx = self.bucketForHash(h);

        if (self.findSlotForBucket(bucket_idx)) |existing| {
            if (self.slots[existing].state == .ready) {
                self.slots[existing].last_used_tick = self.nextTick();
            }
            return false;
        }

        const victim = self.pickVictimSlot() orelse return false;
        const file_off: u64 = self.header.table_offset + @as(u64, bucket_idx) * page.APPLE_PAGE_SIZE;
        self.slots[victim] = .{
            .state = .inflight,
            .bucket_idx = bucket_idx,
            .last_used_tick = self.nextTick(),
        };
        self.async_reader.submit(.{
            .fd = self.file.fd,
            .offset = file_off,
            .dst = self.pageSlice(victim),
            .user_data = victim,
        }) catch |err| {
            self.slots[victim] = .{};
            if (err == error.QueueFull) return false;
            return err;
        };
        self.stats.prefetch_submitted += 1;
        return true;
    }

    fn fetchBucketSlot(self: *NvmeNgramTable, bucket_idx: u32) QueryError!struct { slot_idx: usize, cache_hit: bool } {
        if (self.findSlotForBucket(bucket_idx)) |idx| {
            if (self.slots[idx].state == .inflight) {
                try self.waitForInflightSlot(idx);
                if (self.slots[idx].state == .ready and self.slots[idx].bucket_idx == bucket_idx) {
                    self.slots[idx].last_used_tick = self.nextTick();
                    self.stats.prefetch_hits += 1;
                    self.stats.cache_hits += 1;
                    return .{ .slot_idx = idx, .cache_hit = true };
                }
            } else if (self.slots[idx].state == .ready) {
                self.slots[idx].last_used_tick = self.nextTick();
                self.stats.cache_hits += 1;
                return .{ .slot_idx = idx, .cache_hit = true };
            }
        }

        // Cache miss: obtain an empty or LRU ready slot (waiting for one in-flight
        // prefetch only if every single cache slot is currently in flight).
        var victim_opt = self.pickVictimSlot();
        while (victim_opt == null) {
            var comps: [16]aio.Completion = undefined;
            const got = try self.async_reader.wait(&comps, self.timeout_ms);
            if (got == 0) return error.Timeout;
            for (comps[0..got]) |c| {
                try self.applyCompletion(c);
            }
            victim_opt = self.pickVictimSlot();
        }
        const victim = victim_opt.?;
        const file_off: u64 = self.header.table_offset + @as(u64, bucket_idx) * page.APPLE_PAGE_SIZE;
        try preadExact(self.file.fd, self.pageSlice(victim), file_off);
        self.slots[victim] = .{
            .state = .ready,
            .bucket_idx = bucket_idx,
            .last_used_tick = self.nextTick(),
        };
        self.stats.nvme_reads += 1;
        self.stats.bytes_read += page.APPLE_PAGE_SIZE;
        return .{ .slot_idx = victim, .cache_hit = false };
    }

    /// Queries the `.ang` table for the longest matching suffix of `tail_tokens`
    /// (`min(tail_tokens.len, max_ngram)` down to `min_ngram`), writing up to
    /// `out_cont.len` continuation tokens into `out_cont`. Zero heap allocation.
    pub fn query(
        self: *NvmeNgramTable,
        tail_tokens: []const u32,
        out_cont: []u32,
    ) QueryError!QueryResult {
        const t0 = vmstat.nowNs() catch 0;
        defer {
            const t1 = vmstat.nowNs() catch t0;
            self.stats.total_query_ns += t1 -| t0;
        }

        _ = try self.reapPrefetches();
        self.stats.queries += 1;

        if (tail_tokens.len < self.header.min_ngram or out_cont.len == 0) {
            return .{
                .match_len = 0,
                .draft_len = 0,
                .hit_count = 0,
                .step_prob = 0.0,
                .is_seed = false,
                .cache_hit = false,
            };
        }

        const start_len: usize = @min(tail_tokens.len, @as(usize, self.header.max_ngram));
        const min_len: usize = self.header.min_ngram;
        var ctx_len: usize = start_len;
        while (ctx_len >= min_len) {
            const ctx = tail_tokens[tail_tokens.len - ctx_len .. tail_tokens.len];
            const h = hashContext(ctx);
            const bucket_idx = self.bucketForHash(h);
            const fetched = try self.fetchBucketSlot(bucket_idx);
            const page_slots = self.pageSlotsConst(fetched.slot_idx);

            for (page_slots) |*slot| {
                if (slot.isEmpty()) break;
                if (slot.matchesContext(h, ctx)) {
                    const n: usize = @min(@as(usize, slot.cont_len), out_cont.len);
                    @memcpy(out_cont[0..n], slot.cont_tokens[0..n]);
                    const is_seed = ctx_len >= sam_mod.SEED_MIN_MATCH_LEN;
                    self.stats.hits += 1;
                    if (is_seed) {
                        self.stats.seed_hits += 1;
                    } else {
                        self.stats.branch_hits += 1;
                    }
                    return .{
                        .match_len = @intCast(ctx_len),
                        .draft_len = @intCast(n),
                        .hit_count = slot.hit_count,
                        .step_prob = slot.stepProb(),
                        .is_seed = is_seed,
                        .cache_hit = fetched.cache_hit,
                    };
                }
            }

            if (ctx_len == min_len) break;
            ctx_len -= 1;
        }

        return .{
            .match_len = 0,
            .draft_len = 0,
            .hit_count = 0,
            .step_prob = 0.0,
            .is_seed = false,
            .cache_hit = false,
        };
    }
};

// ---------------------------------------------------------------------------
// In-Zig `.ang` v1 Table Builder (for tests and synthetic benchmarks)
// ---------------------------------------------------------------------------

pub const BuildEntry = struct {
    ctx: []const u32,
    cont: []const u32,
    hit_count: u16 = 1,
    step_prob: f32 = 0.9,
};

pub const BuildOptions = struct {
    num_buckets: u32 = 16,
    min_ngram: u8 = 2,
    max_ngram: u8 = 4,
    max_cont_len: u8 = 8,
    vocab_size: u32 = 0,
    corpus_id: []const u8 = "zig-test",
    include_table_crc: bool = true,
};

pub const BuildError = error{
    InvalidArgument,
    BucketOverflow,
    WriteFailed,
    OutOfMemory,
};

fn slotSortDesc(_: void, a: NgramSlot, b: NgramSlot) bool {
    if (a.hit_count != b.hit_count) return a.hit_count > b.hit_count;
    if (a.ctx_len != b.ctx_len) return a.ctx_len > b.ctx_len;
    return a.ctx_hash < b.ctx_hash;
}

/// Builds a 16 KiB-aligned `.ang` v1 image in memory from `entries`.
pub fn buildTableBytes(
    allocator: std.mem.Allocator,
    entries: []const BuildEntry,
    options: BuildOptions,
) BuildError![]u8 {
    if (options.num_buckets == 0 or !std.math.isPowerOfTwo(options.num_buckets)) return error.InvalidArgument;
    if (options.min_ngram == 0 or options.max_ngram == 0 or
        options.min_ngram > options.max_ngram or options.max_ngram > MAX_CTX_LEN or
        options.max_cont_len == 0 or options.max_cont_len > MAX_CONT_LEN)
    {
        return error.InvalidArgument;
    }

    const table_bytes: usize = @as(usize, options.num_buckets) * page.APPLE_PAGE_SIZE;
    const total_bytes: usize = page.APPLE_PAGE_SIZE + table_bytes;
    const img = try allocator.alloc(u8, total_bytes);
    errdefer allocator.free(img);
    @memset(img, 0);

    const counts = try allocator.alloc(usize, options.num_buckets);
    defer allocator.free(counts);
    @memset(counts, 0);

    const mask: u64 = @as(u64, options.num_buckets - 1);
    var total_inserted: u64 = 0;

    for (entries) |e| {
        if (e.ctx.len < options.min_ngram or e.ctx.len > options.max_ngram) return error.InvalidArgument;
        if (e.cont.len == 0 or e.cont.len > options.max_cont_len) return error.InvalidArgument;
        const h = hashContext(e.ctx);
        const b: usize = @intCast(h & mask);
        const page_off = page.APPLE_PAGE_SIZE + b * page.APPLE_PAGE_SIZE;
        const slots: []NgramSlot = @alignCast(std.mem.bytesAsSlice(NgramSlot, img[page_off .. page_off + page.APPLE_PAGE_SIZE]));

        var slot: NgramSlot = std.mem.zeroes(NgramSlot);
        slot.ctx_hash = h;
        @memcpy(slot.ctx_tail[0..e.ctx.len], e.ctx);
        @memcpy(slot.cont_tokens[0..e.cont.len], e.cont);
        slot.cont_len = @intCast(e.cont.len);
        slot.ctx_len = @intCast(e.ctx.len);
        slot.hit_count = e.hit_count;
        slot.step_prob_f16 = f32ToF16Bits(e.step_prob);

        if (counts[b] >= SLOTS_PER_BUCKET) return error.BucketOverflow;
        slots[counts[b]] = slot;
        counts[b] += 1;
        total_inserted += 1;
    }

    // Sort occupied slots within each bucket page by descending `hit_count`.
    for (0..options.num_buckets) |b| {
        if (counts[b] > 1) {
            const page_off = page.APPLE_PAGE_SIZE + b * page.APPLE_PAGE_SIZE;
            const slots: []NgramSlot = @alignCast(std.mem.bytesAsSlice(NgramSlot, img[page_off .. page_off + page.APPLE_PAGE_SIZE]));
            std.mem.sort(NgramSlot, slots[0..counts[b]], {}, slotSortDesc);
        }
    }

    const table_crc: u32 = if (options.include_table_crc)
        std.hash.Crc32.hash(img[page.APPLE_PAGE_SIZE..])
    else
        0;

    var hdr: Header = .{
        .magic = MAGIC,
        .version = FORMAT_VERSION,
        .flags = if (options.include_table_crc) FLAG_HAS_TABLE_CRC32 else 0,
        .page_size = page.APPLE_PAGE_SIZE,
        .checksum_algo = CHECKSUM_CRC32_IEEE,
        .num_buckets = options.num_buckets,
        .slots_per_bucket = SLOTS_PER_BUCKET,
        .min_ngram = options.min_ngram,
        .max_ngram = options.max_ngram,
        .max_cont_len = options.max_cont_len,
        .reserved_u8 = 0,
        .vocab_size = options.vocab_size,
        .total_entries = total_inserted,
        .table_offset = page.APPLE_PAGE_SIZE,
        .table_bytes = table_bytes,
        .table_crc32 = table_crc,
        .header_crc32 = 0,
        .corpus_id = @splat(0),
        .reserved1 = 0,
    };
    const cid_len = @min(options.corpus_id.len, hdr.corpus_id.len - 1);
    @memcpy(hdr.corpus_id[0..cid_len], options.corpus_id[0..cid_len]);
    hdr.header_crc32 = computeHeaderCrc32(&hdr);
    @memcpy(img[0..@sizeOf(Header)], std.mem.asBytes(&hdr));
    return img;
}

/// Builds and writes a 16 KiB-aligned `.ang` v1 file to `path`.
pub fn writeTableFile(
    allocator: std.mem.Allocator,
    path: [:0]const u8,
    entries: []const BuildEntry,
    options: BuildOptions,
) BuildError!void {
    const img = try buildTableBytes(allocator, entries, options);
    defer allocator.free(img);

    const fd = std.c.open(path.ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o644));
    if (fd < 0) return error.WriteFailed;
    defer _ = std.c.close(fd);

    var written: usize = 0;
    while (written < img.len) {
        const rc = std.c.write(fd, img.ptr + written, img.len - written);
        if (rc < 0) {
            if (std.c.errno(rc) == .INTR) continue;
            return error.WriteFailed;
        }
        if (rc == 0) return error.WriteFailed;
        written += @intCast(rc);
    }
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------

test "ANG v1 hashContext matches cross-language golden vectors" {
    // Verified identically in `tools/tests/test_m5_ngram.py`.
    try std.testing.expectEqual(@as(u64, 0x1E03_65D9_32B4_0171), hashContext(&[_]u32{ 1, 2 }));
    try std.testing.expectEqual(@as(u64, 0x4845_8ECD_6D74_8F49), hashContext(&[_]u32{ 10, 20, 30, 40 }));
    try std.testing.expectEqual(@as(u64, 0xC3E6_F058_F572_B82B), hashContext(&[_]u32{0}));
    try std.testing.expectEqual(@as(u64, 0xF824_27FE_34E2_5F3B), hashContext(&[_]u32{ 151643, 151644, 151645 }));
    try std.testing.expect(hashContext(&[_]u32{ 1, 2 }) != hashContext(&[_]u32{ 2, 1 }));
    try std.testing.expect(hashContext(&[_]u32{ 1, 2, 0 }) != hashContext(&[_]u32{ 1, 2 }));
}

test "NvmeNgramTable F_NOCACHE open, exact verification, 4->3->2 fallback, LRU cache, and async prefetch" {
    const gpa = std.testing.allocator;
    const tmp_path: [:0]const u8 = "/Volumes/Akari/.tmp/aminfer-test-c6-ngram.ang";
    defer _ = std.c.unlink(tmp_path.ptr);

    const entries = [_]BuildEntry{
        .{
            .ctx = &[_]u32{ 10, 20, 30, 40 },
            .cont = &[_]u32{ 50, 60, 70, 80, 90, 100 },
            .hit_count = 42,
            .step_prob = 0.92,
        },
        .{
            .ctx = &[_]u32{ 20, 30, 40 },
            .cont = &[_]u32{ 51, 61, 71 },
            .hit_count = 15,
            .step_prob = 0.65,
        },
        .{
            .ctx = &[_]u32{ 30, 40 },
            .cont = &[_]u32{ 52, 62 },
            .hit_count = 8,
            .step_prob = 0.50,
        },
        .{
            .ctx = &[_]u32{ 200, 201, 202, 203 },
            .cont = &[_]u32{ 204, 205, 206, 207 },
            .hit_count = 99,
            .step_prob = 0.95,
        },
    };

    try writeTableFile(gpa, tmp_path, &entries, .{
        .num_buckets = 16,
        .vocab_size = 1024,
        .corpus_id = "unit-test-corpus",
        .include_table_crc = true,
    });

    // Standalone file validator verifies header CRC, table CRC, slot sorting & hashes.
    const verified_hdr = try verifyFile(gpa, tmp_path);
    try std.testing.expectEqual(@as(u32, 16), verified_hdr.num_buckets);
    try std.testing.expectEqual(@as(u64, 4), verified_hdr.total_entries);
    try std.testing.expectEqualStrings("unit-test-corpus", verified_hdr.corpusIdSlice());

    // Open with a tiny 2-page wired LRU cache to exercise misses, hits, eviction, and async prefetch.
    var tbl = try NvmeNgramTable.open(gpa, tmp_path, .{
        .cache_pages = 2,
        .max_inflight = 1,
        .io_backend = .kqueue_aio,
        .verify_table_crc = true,
    });
    defer tbl.deinit(gpa);
    try std.testing.expectEqual(@as(u64, 2 * page.APPLE_PAGE_SIZE), tbl.totalWiredBytes());

    var out: [8]u32 = undefined;

    // 1. Exact 4-gram match -> seed (`is_seed = true`), cold NVMe read (`cache_hit = false`).
    const q4 = try tbl.query(&[_]u32{ 999, 10, 20, 30, 40 }, &out);
    try std.testing.expectEqual(@as(u32, 4), q4.match_len);
    try std.testing.expectEqual(@as(u32, 6), q4.draft_len);
    try std.testing.expect(q4.is_seed);
    try std.testing.expect(!q4.cache_hit);
    try std.testing.expectEqual(@as(u16, 42), q4.hit_count);
    try std.testing.expectApproxEqAbs(@as(f32, 0.92), q4.step_prob, 2e-3);
    try std.testing.expectEqualSlices(u32, &[_]u32{ 50, 60, 70, 80, 90, 100 }, out[0..q4.draft_len]);

    // 2. Second query of the same 4-gram serves straight from wired LRU cache (`cache_hit = true`)!
    const q4_cached = try tbl.query(&[_]u32{ 10, 20, 30, 40 }, &out);
    try std.testing.expectEqual(@as(u32, 4), q4_cached.match_len);
    try std.testing.expect(q4_cached.cache_hit);
    try std.testing.expectEqualSlices(u32, &[_]u32{ 50, 60, 70, 80, 90, 100 }, out[0..q4_cached.draft_len]);

    // 3. Fallback from missing 4-gram `[777, 20, 30, 40]` -> 3-gram `[20, 30, 40]` (branch).
    const q3 = try tbl.query(&[_]u32{ 777, 20, 30, 40 }, &out);
    try std.testing.expectEqual(@as(u32, 3), q3.match_len);
    try std.testing.expectEqual(@as(u32, 3), q3.draft_len);
    try std.testing.expect(!q3.is_seed);
    try std.testing.expectEqualSlices(u32, &[_]u32{ 51, 61, 71 }, out[0..q3.draft_len]);

    // 4. Fallback to 2-gram `[30, 40]`.
    const q2 = try tbl.query(&[_]u32{ 888, 999, 30, 40 }, &out);
    try std.testing.expectEqual(@as(u32, 2), q2.match_len);
    try std.testing.expectEqual(@as(u32, 2), q2.draft_len);
    try std.testing.expect(!q2.is_seed);
    try std.testing.expectEqualSlices(u32, &[_]u32{ 52, 62 }, out[0..q2.draft_len]);

    // 5. Async background `prefetch()` -> subsequent `query()` hits prefetched wired slot!
    try tbl.clearCache();
    const submitted = try tbl.prefetch(&[_]u32{ 200, 201, 202, 203 });
    try std.testing.expect(submitted);
    const q_pref = try tbl.query(&[_]u32{ 200, 201, 202, 203 }, &out);
    try std.testing.expectEqual(@as(u32, 4), q_pref.match_len);
    try std.testing.expect(q_pref.is_seed);
    try std.testing.expect(q_pref.cache_hit);
    try std.testing.expectEqualSlices(u32, &[_]u32{ 204, 205, 206, 207 }, out[0..q_pref.draft_len]);

    // 6. Miss returns zero length cleanly.
    const q_miss = try tbl.query(&[_]u32{ 555, 556, 557, 558 }, &out);
    try std.testing.expectEqual(@as(u32, 0), q_miss.match_len);
    try std.testing.expectEqual(@as(u32, 0), q_miss.draft_len);
    try std.testing.expect(!q_miss.is_seed);

    try std.testing.expect(tbl.stats.hits >= 5);
    try std.testing.expect(tbl.stats.seed_hits >= 3);
    try std.testing.expect(tbl.stats.cache_hits >= 2);
    try std.testing.expect(tbl.stats.nvme_reads >= 2);
}

test "NvmeNgramTable rejects corrupted header CRC and corrupted table CRC" {
    const gpa = std.testing.allocator;
    const tmp_path: [:0]const u8 = "/Volumes/Akari/.tmp/aminfer-test-c6-corrupt.ang";
    defer _ = std.c.unlink(tmp_path.ptr);

    const entries = [_]BuildEntry{
        .{ .ctx = &[_]u32{ 1, 2, 3, 4 }, .cont = &[_]u32{ 5, 6, 7, 8 }, .hit_count = 10 },
    };
    const img = try buildTableBytes(gpa, &entries, .{ .num_buckets = 4, .include_table_crc = true });
    defer gpa.free(img);

    // Corrupt a byte in bucket page 0 -> table CRC mismatch!
    img[page.APPLE_PAGE_SIZE + 100] ^= 0xFF;
    {
        const fd = std.c.open(tmp_path.ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o644));
        try std.testing.expect(fd >= 0);
        defer _ = std.c.close(fd);
        var w: usize = 0;
        while (w < img.len) {
            const rc = std.c.write(fd, img.ptr + w, img.len - w);
            try std.testing.expect(rc > 0);
            w += @intCast(rc);
        }
    }
    try std.testing.expectError(error.TableChecksumMismatch, NvmeNgramTable.open(gpa, tmp_path, .{
        .cache_pages = 4,
        .verify_table_crc = true,
    }));

    // Corrupt header magic -> header error!
    img[0] = 'X';
    {
        const fd = std.c.open(tmp_path.ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o644));
        try std.testing.expect(fd >= 0);
        defer _ = std.c.close(fd);
        var w: usize = 0;
        while (w < img.len) {
            const rc = std.c.write(fd, img.ptr + w, img.len - w);
            try std.testing.expect(rc > 0);
            w += @intCast(rc);
        }
    }
    try std.testing.expectError(error.InvalidMagic, NvmeNgramTable.open(gpa, tmp_path, .{
        .cache_pages = 4,
        .verify_table_crc = false,
    }));
}
