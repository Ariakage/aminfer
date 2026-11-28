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
const paths = @import("../platform/paths.zig");
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
pub const MAX_CONT_TOKENS: usize = MAX_CONT_LEN;
pub const SLOTS_PER_BUCKET: usize = page.APPLE_PAGE_SIZE / @sizeOf(NgramSlot);

/// Maximum simultaneously registered domain `.ang` corpus packs (`C14`).
pub const MAX_DOMAIN_PACKS: usize = 8;
/// Maximum byte length of a domain pack identifier (`domain_id`).
pub const MAX_DOMAIN_ID_BYTES: usize = 64;

const INVALID_BUCKET: u32 = std.math.maxInt(u32);
const INVALID_BANK: u8 = 0xFF;
const INVALID_PACK_SLOT: u8 = 0xFF;
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

inline fn validateSlotContinuation(slot: *const NgramSlot, vocab_size: u32) ?usize {
    if (slot.cont_len == 0 or slot.cont_len > MAX_CONT_TOKENS) return null;
    const clen: usize = @min(@as(usize, slot.cont_len), MAX_CONT_TOKENS);
    if (vocab_size > 0) {
        for (slot.cont_tokens[0..clen]) |tok| {
            if (tok >= vocab_size) return null;
        }
    }
    return clen;
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

    fn applyCompletionsAll(self: *NvmeNgramTable, comps: []const aio.Completion) QueryError!void {
        var first_err: ?QueryError = null;
        for (comps) |c| {
            self.applyCompletion(c) catch |err| {
                if (first_err == null) first_err = err;
            };
        }
        if (first_err) |err| return err;
    }

    /// Reaps any already-completed background prefetches without blocking.
    pub fn reapPrefetches(self: *NvmeNgramTable) QueryError!usize {
        if (self.async_reader.inflightCount() == 0) return 0;
        var comps: [16]aio.Completion = undefined;
        var total: usize = 0;
        var first_err: ?QueryError = null;
        while (self.async_reader.inflightCount() > 0) {
            const got = self.async_reader.poll(&comps) catch |err| {
                if (first_err == null) first_err = err;
                break;
            };
            if (got == 0) break;
            self.applyCompletionsAll(comps[0..got]) catch |err| {
                if (first_err == null) first_err = err;
            };
            total += got;
        }
        if (first_err) |err| return err;
        return total;
    }

    fn drainPrefetches(self: *NvmeNgramTable) QueryError!void {
        var comps: [16]aio.Completion = undefined;
        var first_err: ?QueryError = null;
        while (self.async_reader.inflightCount() > 0) {
            const got = self.async_reader.wait(&comps, self.timeout_ms) catch |err| {
                if (first_err == null) first_err = err;
                break;
            };
            if (got == 0) return first_err orelse error.Timeout;
            self.applyCompletionsAll(comps[0..got]) catch |err| {
                if (first_err == null) first_err = err;
            };
        }
        if (first_err) |err| return err;
    }

    fn waitForInflightSlot(self: *NvmeNgramTable, target_slot: usize) QueryError!void {
        var comps: [16]aio.Completion = undefined;
        while (self.slots[target_slot].state == .inflight) {
            const got = try self.async_reader.wait(&comps, self.timeout_ms);
            if (got == 0) return error.Timeout;
            try self.applyCompletionsAll(comps[0..got]);
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
            try self.applyCompletionsAll(comps[0..got]);
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
                    const valid_len = validateSlotContinuation(slot, self.header.vocab_size) orelse continue;
                    const n: usize = @min(out_cont.len, valid_len);
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
// C14 Hot-Swappable Multi-Pack Domain `.ang` Registry
// ---------------------------------------------------------------------------

pub const RegistryError = OpenError || error{
    InvalidDomainId,
    RegistryFull,
    DomainAlreadyExists,
    DomainNotFound,
};

pub const RegistryQueryError = QueryError || error{
    DomainNotFound,
};

pub const DomainSelectError = error{
    InvalidDomainId,
    DomainNotFound,
};

pub const RegistryConfig = struct {
    /// Number of 16 KiB wired LRU cache pages shared across all registered domain packs (`1..=1024`).
    /// One additional internal 16 KiB wired scratch page is reserved for zero-allocation header/CRC
    /// validation during `registerPack` and `swapDomainPack`.
    cache_pages: usize = 64,
    /// Maximum concurrent async prefetch reads (`1..=16`; default `4`).
    max_inflight: usize = 4,
    /// Async I/O backend for background `prefetchDomain()`.
    io_backend: aio.IoBackend = .kqueue_aio,
    /// Default setting for verifying full-table CRC-32 when registering or swapping packs.
    verify_table_crc: bool = false,
    /// Timeout waiting for an in-flight prefetch page during `queryDomain()`.
    timeout_ms: u64 = 5_000,
    /// Optional explicit wired budget (`null` computes from host sysctl facts).
    budget_override: ?budget_mod.Budget = null,
};

/// Stack-allocated immutable snapshot of a registered domain `.ang` pack for HTTP/CLI telemetry.
pub const DomainPackInfo = struct {
    domain_id: [MAX_DOMAIN_ID_BYTES]u8 = @splat(0),
    domain_id_len: u8 = 0,
    corpus_id: [48]u8 = @splat(0),
    corpus_id_len: u8 = 0,
    num_buckets: u32 = 0,
    total_entries: u64 = 0,
    min_ngram: u8 = 0,
    max_ngram: u8 = 0,
    max_cont_len: u8 = 0,
    generation: u64 = 0,
    swaps: u64 = 0,
    is_default: bool = false,
    stats: Stats = .{},

    pub fn domainIdSlice(self: *const DomainPackInfo) []const u8 {
        return self.domain_id[0..self.domain_id_len];
    }

    pub fn corpusIdSlice(self: *const DomainPackInfo) []const u8 {
        return self.corpus_id[0..self.corpus_id_len];
    }
};

pub fn isValidDomainId(id: []const u8) bool {
    if (id.len == 0 or id.len > MAX_DOMAIN_ID_BYTES) return false;
    for (id) |c| {
        if (c < 0x20 or c == 0x7f) return false;
    }
    return true;
}

/// Multi-pack hot-swappable `.ang` v1 NVMe N-gram registry (`C14`).
///
/// Hosts up to `MAX_DOMAIN_PACKS` (`8`) independent `.ang` files keyed by
/// `domain_id` (`[64]u8`, defaulting to the `.ang` header's `corpus_id` or a
/// caller-provided alias such as `"repo"` / `"docs"`), backed by a single
/// preallocated 16 KiB-aligned `WiredArena` LRU page pool.
///
/// Hot-swapping (`swapDomainPack`) uses double-buffered descriptors guarded by
/// atomic generation counters (`swap_gen`) and per-bank active reader/AIO pin
/// counters (`pins`), ensuring in-flight `queryDomain` and `prefetchDomain`
/// calls never race against file descriptor closure or stale cache pages, with
/// zero steady-state heap allocations.
pub const DomainNgramRegistry = struct {
    const PackDescriptor = struct {
        file: aio.DirectFile = .{ .fd = -1, .size = 0 },
        header: Header = std.mem.zeroes(Header),
        bucket_mask: u64 = 0,
        generation: u64 = 0,
        pins: std.atomic.Value(u32) = .init(0),
    };

    const AtomicStats = struct {
        queries: std.atomic.Value(u64) = .init(0),
        hits: std.atomic.Value(u64) = .init(0),
        seed_hits: std.atomic.Value(u64) = .init(0),
        branch_hits: std.atomic.Value(u64) = .init(0),
        cache_hits: std.atomic.Value(u64) = .init(0),
        prefetch_submitted: std.atomic.Value(u64) = .init(0),
        prefetch_hits: std.atomic.Value(u64) = .init(0),
        nvme_reads: std.atomic.Value(u64) = .init(0),
        bytes_read: std.atomic.Value(u64) = .init(0),
        total_query_ns: std.atomic.Value(u64) = .init(0),
        swaps: std.atomic.Value(u64) = .init(0),

        fn snapshot(self: *const AtomicStats) Stats {
            return .{
                .queries = self.queries.load(.monotonic),
                .hits = self.hits.load(.monotonic),
                .seed_hits = self.seed_hits.load(.monotonic),
                .branch_hits = self.branch_hits.load(.monotonic),
                .cache_hits = self.cache_hits.load(.monotonic),
                .prefetch_submitted = self.prefetch_submitted.load(.monotonic),
                .prefetch_hits = self.prefetch_hits.load(.monotonic),
                .nvme_reads = self.nvme_reads.load(.monotonic),
                .bytes_read = self.bytes_read.load(.monotonic),
                .total_query_ns = self.total_query_ns.load(.monotonic),
            };
        }
    };

    const PackSlot = struct {
        occupied: std.atomic.Value(bool) = .init(false),
        domain_id: [MAX_DOMAIN_ID_BYTES]u8 = @splat(0),
        domain_id_len: u8 = 0,
        active_bank: std.atomic.Value(u8) = .init(INVALID_BANK),
        /// Even = stable; odd = active_bank flip in progress.
        swap_gen: std.atomic.Value(u64) = .init(0),
        banks: [2]PackDescriptor = .{ .{}, .{} },
        stats: AtomicStats = .{},

        fn domainIdSlice(self: *const PackSlot) []const u8 {
            return self.domain_id[0..self.domain_id_len];
        }
    };

    const SharedSlotState = enum(u8) {
        empty = 0,
        inflight = 1,
        loading = 2,
        ready = 3,
    };

    const SharedCacheSlot = struct {
        state: SharedSlotState = .empty,
        pack_idx: u8 = INVALID_PACK_SLOT,
        bank_idx: u8 = INVALID_BANK,
        desc_gen: u64 = 0,
        bucket_idx: u32 = INVALID_BUCKET,
        last_used_tick: u64 = 0,
        readers: u32 = 0,
    };

    const PinnedBank = struct {
        pack_idx: u8,
        bank_idx: u8,
        desc: *const PackDescriptor,
    };

    arena: arena_mod.WiredArena,
    cache_pages: usize,
    cache_slots: []SharedCacheSlot,
    async_reader: aio.AsyncExpertReader,
    verify_table_crc_default: bool,
    timeout_ms: u64,
    tick: u64,

    packs: [MAX_DOMAIN_PACKS]PackSlot,
    pack_count: std.atomic.Value(usize),
    default_slot: std.atomic.Value(u8),
    next_generation: std.atomic.Value(u64),

    // Lightweight atomic locks: `cache_lock` guards LRU slot metadata and `async_reader`
    // (never held across synchronous `preadExact`); `writer_lock` serializes cold-path
    // `registerPack` / `swapDomainPack` operations.
    cache_lock: std.atomic.Value(bool),
    writer_lock: std.atomic.Value(bool),

    pub fn init(allocator: std.mem.Allocator, config: RegistryConfig) OpenError!DomainNgramRegistry {
        if (config.cache_pages == 0 or config.cache_pages > 1024) return error.InvalidCacheConfig;
        if (config.max_inflight == 0 or config.max_inflight > 16) return error.InvalidCacheConfig;

        // Allocate `cache_pages` LRU pages + 1 dedicated 16 KiB writer scratch page.
        const arena_bytes = (config.cache_pages + 1) * page.APPLE_PAGE_SIZE;
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

        const cache_slots = try allocator.alloc(SharedCacheSlot, config.cache_pages);
        errdefer allocator.free(cache_slots);
        @memset(cache_slots, .{});

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

        var packs: [MAX_DOMAIN_PACKS]PackSlot = undefined;
        for (&packs) |*p| p.* = .{};

        return .{
            .arena = arena,
            .cache_pages = config.cache_pages,
            .cache_slots = cache_slots,
            .async_reader = reader,
            .verify_table_crc_default = config.verify_table_crc,
            .timeout_ms = config.timeout_ms,
            .tick = 0,
            .packs = packs,
            .pack_count = .init(0),
            .default_slot = .init(INVALID_PACK_SLOT),
            .next_generation = .init(1),
            .cache_lock = .init(false),
            .writer_lock = .init(false),
        };
    }

    pub fn deinit(self: *DomainNgramRegistry, allocator: std.mem.Allocator) void {
        // Drain all in-flight kernel AIO / E-core pread requests BEFORE releasing
        // the wired arena or closing any F_NOCACHE file descriptors.
        self.async_reader.deinit(allocator);
        for (&self.packs) |*p| {
            for (&p.banks) |*b| {
                if (b.file.fd >= 0) {
                    b.file.close();
                }
            }
            p.occupied.store(false, .release);
            p.active_bank.store(INVALID_BANK, .release);
        }
        allocator.free(self.cache_slots);
        self.cache_slots = &.{};
        self.arena.deinit();
    }

    pub fn totalWiredBytes(self: *const DomainNgramRegistry) u64 {
        return self.arena.bytes().len;
    }

    pub fn cachePageCount(self: *const DomainNgramRegistry) usize {
        return self.cache_pages;
    }

    pub fn packCount(self: *const DomainNgramRegistry) usize {
        return self.pack_count.load(.acquire);
    }

    inline fn acquireCacheLock(self: *DomainNgramRegistry) void {
        while (self.cache_lock.cmpxchgWeak(false, true, .acquire, .monotonic) != null) {
            std.atomic.spinLoopHint();
        }
    }

    inline fn releaseCacheLock(self: *DomainNgramRegistry) void {
        self.cache_lock.store(false, .release);
    }

    inline fn acquireWriterLock(self: *DomainNgramRegistry) void {
        while (self.writer_lock.cmpxchgWeak(false, true, .acquire, .monotonic) != null) {
            std.atomic.spinLoopHint();
        }
    }

    inline fn releaseWriterLock(self: *DomainNgramRegistry) void {
        self.writer_lock.store(false, .release);
    }

    inline fn nextTickLocked(self: *DomainNgramRegistry) u64 {
        self.tick +%= 1;
        if (self.tick == 0) self.tick = 1;
        return self.tick;
    }

    inline fn pageSlice(self: *DomainNgramRegistry, slot_idx: usize) []align(page.APPLE_PAGE_SIZE) u8 {
        const start = slot_idx * page.APPLE_PAGE_SIZE;
        return @alignCast(self.arena.bytes()[start .. start + page.APPLE_PAGE_SIZE]);
    }

    inline fn writerScratchPage(self: *DomainNgramRegistry) []align(page.APPLE_PAGE_SIZE) u8 {
        const start = self.cache_pages * page.APPLE_PAGE_SIZE;
        return @alignCast(self.arena.bytes()[start .. start + page.APPLE_PAGE_SIZE]);
    }

    inline fn pageSlotsConst(self: *const DomainNgramRegistry, slot_idx: usize) []const NgramSlot {
        const start = slot_idx * page.APPLE_PAGE_SIZE;
        const raw = self.arena.bytes()[start .. start + page.APPLE_PAGE_SIZE];
        return @alignCast(std.mem.bytesAsSlice(NgramSlot, raw));
    }

    fn findPackIndex(self: *const DomainNgramRegistry, domain_id: []const u8) ?u8 {
        for (&self.packs, 0..) |*p, i| {
            if (p.occupied.load(.acquire) and std.mem.eql(u8, p.domainIdSlice(), domain_id)) {
                return @intCast(i);
            }
        }
        return null;
    }

    /// Lock-free check for whether `domain_id` is currently registered.
    pub fn hasDomain(self: *const DomainNgramRegistry, domain_id: []const u8) bool {
        if (!isValidDomainId(domain_id)) return false;
        return self.findPackIndex(domain_id) != null;
    }

    fn pinPackBank(self: *DomainNgramRegistry, pack_idx: u8) ?PinnedBank {
        const pack = &self.packs[pack_idx];
        while (true) {
            if (!pack.occupied.load(.acquire)) return null;
            const gen_before = pack.swap_gen.load(.acquire);
            if ((gen_before & 1) != 0) {
                std.atomic.spinLoopHint();
                continue;
            }
            const bank_idx = pack.active_bank.load(.acquire);
            if (bank_idx >= 2) return null;
            const desc = &pack.banks[bank_idx];
            _ = desc.pins.fetchAdd(1, .acq_rel);
            const gen_after = pack.swap_gen.load(.acquire);
            if (gen_after == gen_before and pack.active_bank.load(.acquire) == bank_idx) {
                return .{
                    .pack_idx = pack_idx,
                    .bank_idx = bank_idx,
                    .desc = desc,
                };
            }
            _ = desc.pins.fetchSub(1, .acq_rel);
        }
    }

    inline fn unpinPackBank(self: *DomainNgramRegistry, pack_idx: u8, bank_idx: u8) void {
        _ = self.packs[pack_idx].banks[bank_idx].pins.fetchSub(1, .acq_rel);
    }

    fn openAndValidateFileLocked(
        self: *DomainNgramRegistry,
        path: [:0]const u8,
        verify_table_crc: bool,
    ) OpenError!struct { file: aio.DirectFile, header: Header } {
        var file = try aio.DirectFile.openReadOnly(path);
        errdefer file.close();

        const scratch = self.writerScratchPage();
        try preadExact(file.fd, scratch, 0);
        var hdr: Header = undefined;
        @memcpy(std.mem.asBytes(&hdr), scratch[0..@sizeOf(Header)]);
        try validateHeader(&hdr, file.size);

        if (verify_table_crc and hdr.hasTableCrc()) {
            var crc = std.hash.Crc32.init();
            var off: u64 = hdr.table_offset;
            const end_off: u64 = hdr.table_offset + hdr.table_bytes;
            while (off < end_off) {
                try preadExact(file.fd, scratch, off);
                crc.update(scratch);
                off += page.APPLE_PAGE_SIZE;
            }
            if (crc.final() != hdr.table_crc32) return error.TableChecksumMismatch;
        }
        @memset(scratch, 0);
        return .{ .file = file, .header = hdr };
    }

    /// Registers a new `.ang` v1 NVMe corpus pack under `domain_id_alias`
    /// (defaulting to the `.ang` header's `corpus_id` when `null`).
    /// Performs zero heap allocations after `DomainNgramRegistry.init`.
    pub fn registerPack(
        self: *DomainNgramRegistry,
        domain_id_alias: ?[]const u8,
        path: [:0]const u8,
        verify_table_crc: ?bool,
    ) RegistryError!usize {
        if (domain_id_alias) |alias| {
            if (!isValidDomainId(alias)) return error.InvalidDomainId;
        }

        self.acquireWriterLock();
        defer self.releaseWriterLock();

        const verify_crc = verify_table_crc orelse self.verify_table_crc_default;
        var opened = try self.openAndValidateFileLocked(path, verify_crc);
        errdefer opened.file.close();

        const resolved_id: []const u8 = domain_id_alias orelse opened.header.corpusIdSlice();
        if (!isValidDomainId(resolved_id)) return error.InvalidDomainId;

        if (self.findPackIndex(resolved_id) != null) return error.DomainAlreadyExists;

        var free_idx: ?usize = null;
        for (&self.packs, 0..) |*p, i| {
            if (!p.occupied.load(.acquire)) {
                free_idx = i;
                break;
            }
        }
        const slot_idx = free_idx orelse return error.RegistryFull;
        const pack = &self.packs[slot_idx];

        const gen = self.next_generation.fetchAdd(1, .acq_rel);
        @memset(&pack.domain_id, 0);
        @memcpy(pack.domain_id[0..resolved_id.len], resolved_id);
        pack.domain_id_len = @intCast(resolved_id.len);
        pack.banks[0] = .{
            .file = opened.file,
            .header = opened.header,
            .bucket_mask = @as(u64, opened.header.num_buckets - 1),
            .generation = gen,
            .pins = .init(0),
        };
        pack.banks[1] = .{};
        pack.stats = .{};
        pack.active_bank.store(0, .seq_cst);
        pack.swap_gen.store(0, .seq_cst);
        pack.occupied.store(true, .seq_cst);
        _ = self.pack_count.fetchAdd(1, .acq_rel);

        // If no default pack was active yet, atomically select this pack as default.
        _ = self.default_slot.cmpxchgStrong(INVALID_PACK_SLOT, @intCast(slot_idx), .seq_cst, .acquire);
        return slot_idx;
    }

    /// Atomically hot-swaps the `.ang` file backing an existing `domain_id`.
    ///
    /// Opens and validates `new_path` into the inactive descriptor bank with
    /// zero heap allocation, flips `active_bank` under `swap_gen`, waits for
    /// all pinned in-flight queries and async prefetches on the old bank to
    /// drain, and then closes the old `F_NOCACHE` descriptor. Returns the new
    /// monotonic descriptor generation number.
    pub fn swapDomainPack(
        self: *DomainNgramRegistry,
        domain_id: []const u8,
        new_path: [:0]const u8,
        verify_table_crc: ?bool,
    ) RegistryError!u64 {
        if (!isValidDomainId(domain_id)) return error.InvalidDomainId;

        self.acquireWriterLock();
        defer self.releaseWriterLock();

        const pack_idx = self.findPackIndex(domain_id) orelse return error.DomainNotFound;
        const pack = &self.packs[pack_idx];
        const old_bank = pack.active_bank.load(.acquire);
        if (old_bank >= 2) return error.DomainNotFound;
        const new_bank: u8 = 1 - old_bank;

        // Ensure the target inactive bank has zero residual pins before reuse.
        while (pack.banks[new_bank].pins.load(.acquire) > 0) {
            try self.reapPendingForDrain();
            std.atomic.spinLoopHint();
        }
        if (pack.banks[new_bank].file.fd >= 0) {
            pack.banks[new_bank].file.close();
            pack.banks[new_bank] = .{};
        }

        const verify_crc = verify_table_crc orelse self.verify_table_crc_default;
        var opened = try self.openAndValidateFileLocked(new_path, verify_crc);
        var close_new_on_err = true;
        errdefer if (close_new_on_err) opened.file.close();

        const new_gen = self.next_generation.fetchAdd(1, .acq_rel);
        pack.banks[new_bank] = .{
            .file = opened.file,
            .header = opened.header,
            .bucket_mask = @as(u64, opened.header.num_buckets - 1),
            .generation = new_gen,
            .pins = .init(0),
        };
        // Ownership of `opened.file` transferred to `pack.banks[new_bank]` before publication.
        close_new_on_err = false;

        // Atomic generation-guarded descriptor bank publication.
        const prev_swap = pack.swap_gen.load(.acquire);
        pack.swap_gen.store(prev_swap +% 1, .seq_cst);
        pack.active_bank.store(new_bank, .seq_cst);
        pack.swap_gen.store(prev_swap +% 2, .seq_cst);
        _ = pack.stats.swaps.fetchAdd(1, .monotonic);

        // Drain all in-flight queries and async prefetches pinned to `old_bank`
        // BEFORE closing `old_bank.file`.
        var drain_err: ?QueryError = null;
        while (pack.banks[old_bank].pins.load(.acquire) > 0) {
            self.reapPendingForDrain() catch |err| {
                if (drain_err == null) drain_err = err;
            };
            std.atomic.spinLoopHint();
        }
        if (pack.banks[old_bank].file.fd >= 0) {
            pack.banks[old_bank].file.close();
        }
        pack.banks[old_bank] = .{};
        if (drain_err) |err| return err;

        return new_gen;
    }

    fn reapPendingForDrain(self: *DomainNgramRegistry) QueryError!void {
        self.acquireCacheLock();
        defer self.releaseCacheLock();
        if (self.async_reader.inflightCount() == 0) return;
        const reaped = try self.reapPrefetchesLocked();
        if (reaped == 0 and self.async_reader.inflightCount() > 0) {
            var comps: [16]aio.Completion = undefined;
            const got = try self.async_reader.wait(&comps, 2);
            var first_err: ?QueryError = null;
            for (comps[0..got]) |c| {
                self.applyCompletionLocked(c) catch |err| {
                    if (first_err == null) first_err = err;
                };
            }
            if (first_err) |err| return err;
        }
    }

    /// Atomically switches the active default domain pack (`0` heap allocs).
    pub fn selectDefaultDomain(self: *DomainNgramRegistry, domain_id: []const u8) DomainSelectError!void {
        if (!isValidDomainId(domain_id)) return error.InvalidDomainId;
        const idx = self.findPackIndex(domain_id) orelse return error.DomainNotFound;
        self.default_slot.store(idx, .seq_cst);
    }

    /// Copies the active default domain ID into `out` and returns the populated slice
    /// (empty when no domain packs are registered). Lock-free, zero heap allocations.
    pub fn defaultDomainId(self: *const DomainNgramRegistry, out: *[MAX_DOMAIN_ID_BYTES]u8) []const u8 {
        const idx = self.default_slot.load(.acquire);
        if (idx >= MAX_DOMAIN_PACKS) return out[0..0];
        const pack = &self.packs[idx];
        if (!pack.occupied.load(.acquire)) return out[0..0];
        const s = pack.domainIdSlice();
        @memcpy(out[0..s.len], s);
        return out[0..s.len];
    }

    /// Captures a consistent lock-free snapshot of all registered domain packs into `out`.
    /// Returns the number of entries written (`0..=MAX_DOMAIN_PACKS`). Zero heap allocations.
    pub fn snapshotPacks(self: *DomainNgramRegistry, out: *[MAX_DOMAIN_PACKS]DomainPackInfo) usize {
        const def_idx = self.default_slot.load(.acquire);
        var count: usize = 0;
        for (0..MAX_DOMAIN_PACKS) |i| {
            const pinned = self.pinPackBank(@intCast(i)) orelse continue;
            defer self.unpinPackBank(pinned.pack_idx, pinned.bank_idx);

            const pack = &self.packs[i];
            var info: DomainPackInfo = .{};
            const dom = pack.domainIdSlice();
            @memcpy(info.domain_id[0..dom.len], dom);
            info.domain_id_len = @intCast(dom.len);

            const cid = pinned.desc.header.corpusIdSlice();
            @memcpy(info.corpus_id[0..cid.len], cid);
            info.corpus_id_len = @intCast(cid.len);

            info.num_buckets = pinned.desc.header.num_buckets;
            info.total_entries = pinned.desc.header.total_entries;
            info.min_ngram = pinned.desc.header.min_ngram;
            info.max_ngram = pinned.desc.header.max_ngram;
            info.max_cont_len = pinned.desc.header.max_cont_len;
            info.generation = pinned.desc.generation;
            info.swaps = pack.stats.swaps.load(.monotonic);
            info.is_default = (def_idx == @as(u8, @intCast(i)));
            info.stats = pack.stats.snapshot();

            out[count] = info;
            count += 1;
        }
        return count;
    }

    fn resolveTargetPack(self: *DomainNgramRegistry, domain_id: ?[]const u8) error{DomainNotFound}!?u8 {
        if (domain_id) |id| {
            if (id.len > 0) {
                return self.findPackIndex(id) orelse return error.DomainNotFound;
            }
        }
        const def_idx = self.default_slot.load(.acquire);
        if (def_idx >= MAX_DOMAIN_PACKS) return null;
        if (!self.packs[def_idx].occupied.load(.acquire)) return null;
        return def_idx;
    }

    fn findCacheSlotLocked(
        self: *const DomainNgramRegistry,
        pack_idx: u8,
        desc_gen: u64,
        bucket_idx: u32,
    ) ?usize {
        for (self.cache_slots, 0..) |s, i| {
            if (s.state != .empty and
                s.pack_idx == pack_idx and
                s.desc_gen == desc_gen and
                s.bucket_idx == bucket_idx)
            {
                return i;
            }
        }
        return null;
    }

    fn pickVictimSlotLocked(self: *const DomainNgramRegistry) ?usize {
        var best_lru: ?usize = null;
        var best_tick: u64 = std.math.maxInt(u64);
        for (self.cache_slots, 0..) |s, i| {
            switch (s.state) {
                .empty => return i,
                .inflight, .loading => continue,
                .ready => {
                    if (s.readers > 0) continue;
                    if (s.last_used_tick < best_tick) {
                        best_tick = s.last_used_tick;
                        best_lru = i;
                    }
                },
            }
        }
        return best_lru;
    }

    fn applyCompletionLocked(self: *DomainNgramRegistry, comp: aio.Completion) QueryError!void {
        const slot_idx: usize = @intCast(comp.user_data);
        std.debug.assert(slot_idx < self.cache_slots.len);
        const slot = &self.cache_slots[slot_idx];
        const p_idx = slot.pack_idx;
        const b_idx = slot.bank_idx;
        // Always release the bank pin acquired for this async prefetch!
        defer {
            if (p_idx < MAX_DOMAIN_PACKS and b_idx < 2) {
                self.unpinPackBank(p_idx, b_idx);
            }
        }

        const bytes_read = comp.check() catch |err| {
            slot.* = .{};
            return err;
        };
        if (bytes_read != page.APPLE_PAGE_SIZE) {
            slot.* = .{};
            return error.ShortRead;
        }
        slot.state = .ready;
        slot.last_used_tick = self.nextTickLocked();
        if (p_idx < MAX_DOMAIN_PACKS) {
            _ = self.packs[p_idx].stats.nvme_reads.fetchAdd(1, .monotonic);
            _ = self.packs[p_idx].stats.bytes_read.fetchAdd(bytes_read, .monotonic);
        }
    }

    fn reapPrefetchesLocked(self: *DomainNgramRegistry) QueryError!usize {
        if (self.async_reader.inflightCount() == 0) return 0;
        var comps: [16]aio.Completion = undefined;
        var total: usize = 0;
        var first_err: ?QueryError = null;
        while (self.async_reader.inflightCount() > 0) {
            const got = try self.async_reader.poll(&comps);
            if (got == 0) break;
            for (comps[0..got]) |c| {
                self.applyCompletionLocked(c) catch |err| {
                    if (first_err == null) first_err = err;
                };
            }
            total += got;
        }
        if (first_err) |err| return err;
        return total;
    }

    /// Reaps any already-completed background prefetches across all packs.
    pub fn reapPrefetches(self: *DomainNgramRegistry) QueryError!usize {
        self.acquireCacheLock();
        defer self.releaseCacheLock();
        return self.reapPrefetchesLocked();
    }

    /// Clears the shared wired LRU cache once all in-flight prefetches and active
    /// page scans complete.
    pub fn clearCache(self: *DomainNgramRegistry) QueryError!void {
        while (true) {
            self.acquireCacheLock();
            if (self.async_reader.inflightCount() > 0) {
                var comps: [16]aio.Completion = undefined;
                const got = self.async_reader.wait(&comps, self.timeout_ms) catch |err| {
                    self.releaseCacheLock();
                    return err;
                };
                if (got == 0) {
                    self.releaseCacheLock();
                    return error.Timeout;
                }
                var first_err: ?QueryError = null;
                for (comps[0..got]) |c| {
                    self.applyCompletionLocked(c) catch |err| {
                        if (first_err == null) first_err = err;
                    };
                }
                self.releaseCacheLock();
                if (first_err) |err| return err;
                continue;
            }
            var busy = false;
            for (self.cache_slots) |s| {
                if (s.state == .loading or s.readers > 0) {
                    busy = true;
                    break;
                }
            }
            if (busy) {
                self.releaseCacheLock();
                std.atomic.spinLoopHint();
                continue;
            }
            for (self.cache_slots) |*s| s.* = .{};
            self.releaseCacheLock();
            return;
        }
    }

    inline fn releaseCacheSlotReader(self: *DomainNgramRegistry, slot_idx: usize) void {
        self.acquireCacheLock();
        std.debug.assert(self.cache_slots[slot_idx].readers > 0);
        self.cache_slots[slot_idx].readers -= 1;
        self.releaseCacheLock();
    }

    /// Submits an asynchronous `F_NOCACHE` 16 KiB bucket page prefetch for the
    /// resolved domain (`domain_id` or active default when `null`/empty).
    /// Zero heap allocation; safely pins the active descriptor bank across the AIO.
    pub fn prefetchDomain(
        self: *DomainNgramRegistry,
        domain_id: ?[]const u8,
        tail_tokens: []const u32,
    ) RegistryQueryError!bool {
        const pack_idx = (try self.resolveTargetPack(domain_id)) orelse return false;
        const pinned = self.pinPackBank(pack_idx) orelse return false;
        defer self.unpinPackBank(pinned.pack_idx, pinned.bank_idx);

        if (tail_tokens.len < pinned.desc.header.min_ngram) return false;

        const ctx_len: usize = @min(tail_tokens.len, @as(usize, pinned.desc.header.max_ngram));
        const ctx = tail_tokens[tail_tokens.len - ctx_len .. tail_tokens.len];
        const h = hashContext(ctx);
        const bucket_idx: u32 = @intCast(h & pinned.desc.bucket_mask);

        self.acquireCacheLock();
        defer self.releaseCacheLock();

        _ = try self.reapPrefetchesLocked();
        if (self.async_reader.availableSlots() == 0) return false;

        if (self.findCacheSlotLocked(pinned.pack_idx, pinned.desc.generation, bucket_idx)) |existing| {
            if (self.cache_slots[existing].state == .ready) {
                self.cache_slots[existing].last_used_tick = self.nextTickLocked();
            }
            return false;
        }

        const victim = self.pickVictimSlotLocked() orelse return false;
        const file_off: u64 = pinned.desc.header.table_offset + @as(u64, bucket_idx) * page.APPLE_PAGE_SIZE;

        // Transfer an independent pin on `pinned.desc` to the in-flight AIO request
        // (released unconditionally in `applyCompletionLocked` or on submit error).
        _ = self.packs[pinned.pack_idx].banks[pinned.bank_idx].pins.fetchAdd(1, .acq_rel);
        self.cache_slots[victim] = .{
            .state = .inflight,
            .pack_idx = pinned.pack_idx,
            .bank_idx = pinned.bank_idx,
            .desc_gen = pinned.desc.generation,
            .bucket_idx = bucket_idx,
            .last_used_tick = self.nextTickLocked(),
            .readers = 0,
        };
        self.async_reader.submit(.{
            .fd = pinned.desc.file.fd,
            .offset = file_off,
            .dst = self.pageSlice(victim),
            .user_data = victim,
        }) catch |err| {
            self.cache_slots[victim] = .{};
            self.unpinPackBank(pinned.pack_idx, pinned.bank_idx);
            if (err == error.QueueFull) return false;
            return err;
        };
        _ = self.packs[pinned.pack_idx].stats.prefetch_submitted.fetchAdd(1, .monotonic);
        return true;
    }

    fn fetchBucketSlot(
        self: *DomainNgramRegistry,
        pinned: PinnedBank,
        bucket_idx: u32,
    ) QueryError!struct { slot_idx: usize, cache_hit: bool } {
        while (true) {
            self.acquireCacheLock();
            _ = self.reapPrefetchesLocked() catch |err| {
                self.releaseCacheLock();
                return err;
            };

            if (self.findCacheSlotLocked(pinned.pack_idx, pinned.desc.generation, bucket_idx)) |idx| {
                switch (self.cache_slots[idx].state) {
                    .ready => {
                        self.cache_slots[idx].readers += 1;
                        self.cache_slots[idx].last_used_tick = self.nextTickLocked();
                        self.releaseCacheLock();
                        _ = self.packs[pinned.pack_idx].stats.cache_hits.fetchAdd(1, .monotonic);
                        return .{ .slot_idx = idx, .cache_hit = true };
                    },
                    .inflight => {
                        var comps: [16]aio.Completion = undefined;
                        const got = self.async_reader.wait(&comps, self.timeout_ms) catch |err| {
                            self.releaseCacheLock();
                            return err;
                        };
                        if (got == 0) {
                            self.releaseCacheLock();
                            return error.Timeout;
                        }
                        var first_err: ?QueryError = null;
                        for (comps[0..got]) |c| {
                            self.applyCompletionLocked(c) catch |err| {
                                if (first_err == null) first_err = err;
                            };
                        }
                        if (self.cache_slots[idx].state == .ready and
                            self.cache_slots[idx].pack_idx == pinned.pack_idx and
                            self.cache_slots[idx].desc_gen == pinned.desc.generation and
                            self.cache_slots[idx].bucket_idx == bucket_idx)
                        {
                            self.cache_slots[idx].readers += 1;
                            self.cache_slots[idx].last_used_tick = self.nextTickLocked();
                            self.releaseCacheLock();
                            _ = self.packs[pinned.pack_idx].stats.prefetch_hits.fetchAdd(1, .monotonic);
                            _ = self.packs[pinned.pack_idx].stats.cache_hits.fetchAdd(1, .monotonic);
                            return .{ .slot_idx = idx, .cache_hit = true };
                        }
                        self.releaseCacheLock();
                        if (first_err) |err| return err;
                        continue;
                    },
                    .loading => {
                        self.releaseCacheLock();
                        std.atomic.spinLoopHint();
                        continue;
                    },
                    .empty => {},
                }
            }

            const victim_opt = self.pickVictimSlotLocked();
            if (victim_opt == null) {
                if (self.async_reader.inflightCount() > 0) {
                    var comps: [16]aio.Completion = undefined;
                    const got = self.async_reader.wait(&comps, self.timeout_ms) catch |err| {
                        self.releaseCacheLock();
                        return err;
                    };
                    if (got == 0) {
                        self.releaseCacheLock();
                        return error.Timeout;
                    }
                    var first_err: ?QueryError = null;
                    for (comps[0..got]) |c| {
                        self.applyCompletionLocked(c) catch |err| {
                            if (first_err == null) first_err = err;
                        };
                    }
                    self.releaseCacheLock();
                    if (first_err) |err| return err;
                } else {
                    self.releaseCacheLock();
                    std.atomic.spinLoopHint();
                }
                continue;
            }

            const victim = victim_opt.?;
            self.cache_slots[victim] = .{
                .state = .loading,
                .pack_idx = pinned.pack_idx,
                .bank_idx = pinned.bank_idx,
                .desc_gen = pinned.desc.generation,
                .bucket_idx = bucket_idx,
                .last_used_tick = self.nextTickLocked(),
                .readers = 1,
            };
            self.releaseCacheLock();

            const file_off: u64 = pinned.desc.header.table_offset + @as(u64, bucket_idx) * page.APPLE_PAGE_SIZE;
            preadExact(pinned.desc.file.fd, self.pageSlice(victim), file_off) catch |err| {
                self.acquireCacheLock();
                self.cache_slots[victim] = .{};
                self.releaseCacheLock();
                return err;
            };

            self.acquireCacheLock();
            self.cache_slots[victim].state = .ready;
            self.cache_slots[victim].last_used_tick = self.nextTickLocked();
            self.releaseCacheLock();

            _ = self.packs[pinned.pack_idx].stats.nvme_reads.fetchAdd(1, .monotonic);
            _ = self.packs[pinned.pack_idx].stats.bytes_read.fetchAdd(page.APPLE_PAGE_SIZE, .monotonic);
            return .{ .slot_idx = victim, .cache_hit = false };
        }
    }

    /// Queries the resolved domain pack (`domain_id` or active default when `null`/empty)
    /// for the longest matching suffix of `tail_tokens`. Zero steady-state heap allocations;
    /// safe concurrently with `selectDefaultDomain` and `swapDomainPack`.
    pub fn queryDomain(
        self: *DomainNgramRegistry,
        domain_id: ?[]const u8,
        tail_tokens: []const u32,
        out_cont: []u32,
    ) RegistryQueryError!QueryResult {
        const pack_idx = (try self.resolveTargetPack(domain_id)) orelse {
            return .{
                .match_len = 0,
                .draft_len = 0,
                .hit_count = 0,
                .step_prob = 0.0,
                .is_seed = false,
                .cache_hit = false,
            };
        };
        const pinned = self.pinPackBank(pack_idx) orelse {
            return .{
                .match_len = 0,
                .draft_len = 0,
                .hit_count = 0,
                .step_prob = 0.0,
                .is_seed = false,
                .cache_hit = false,
            };
        };
        defer self.unpinPackBank(pinned.pack_idx, pinned.bank_idx);

        const t0 = vmstat.nowNs() catch 0;
        defer {
            const t1 = vmstat.nowNs() catch t0;
            _ = self.packs[pinned.pack_idx].stats.total_query_ns.fetchAdd(t1 -| t0, .monotonic);
        }

        _ = self.packs[pinned.pack_idx].stats.queries.fetchAdd(1, .monotonic);

        if (tail_tokens.len < pinned.desc.header.min_ngram or out_cont.len == 0) {
            return .{
                .match_len = 0,
                .draft_len = 0,
                .hit_count = 0,
                .step_prob = 0.0,
                .is_seed = false,
                .cache_hit = false,
            };
        }

        const start_len: usize = @min(tail_tokens.len, @as(usize, pinned.desc.header.max_ngram));
        const min_len: usize = pinned.desc.header.min_ngram;
        var ctx_len: usize = start_len;
        while (ctx_len >= min_len) {
            const ctx = tail_tokens[tail_tokens.len - ctx_len .. tail_tokens.len];
            const h = hashContext(ctx);
            const bucket_idx: u32 = @intCast(h & pinned.desc.bucket_mask);
            const fetched = try self.fetchBucketSlot(pinned, bucket_idx);
            const page_slots = self.pageSlotsConst(fetched.slot_idx);

            var matched: ?QueryResult = null;
            for (page_slots) |*slot| {
                if (slot.isEmpty()) break;
                if (slot.matchesContext(h, ctx)) {
                    const valid_len = validateSlotContinuation(slot, pinned.desc.header.vocab_size) orelse continue;
                    const n: usize = @min(out_cont.len, valid_len);
                    @memcpy(out_cont[0..n], slot.cont_tokens[0..n]);
                    const is_seed = ctx_len >= sam_mod.SEED_MIN_MATCH_LEN;
                    _ = self.packs[pinned.pack_idx].stats.hits.fetchAdd(1, .monotonic);
                    if (is_seed) {
                        _ = self.packs[pinned.pack_idx].stats.seed_hits.fetchAdd(1, .monotonic);
                    } else {
                        _ = self.packs[pinned.pack_idx].stats.branch_hits.fetchAdd(1, .monotonic);
                    }
                    matched = .{
                        .match_len = @intCast(ctx_len),
                        .draft_len = @intCast(n),
                        .hit_count = slot.hit_count,
                        .step_prob = slot.stepProb(),
                        .is_seed = is_seed,
                        .cache_hit = fetched.cache_hit,
                    };
                    break;
                }
            }
            self.releaseCacheSlotReader(fetched.slot_idx);
            if (matched) |res| return res;

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
    var tmp_buf: [std.fs.max_path_bytes]u8 = undefined;
    const tmp_path = try paths.tmpPathFmtZ(&tmp_buf, "aminfer-test-c6-ngram-{d}.ang", .{std.c.getpid()});
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
    var tmp_buf: [std.fs.max_path_bytes]u8 = undefined;
    const tmp_path = try paths.tmpPathFmtZ(&tmp_buf, "aminfer-test-c6-corrupt-{d}.ang", .{std.c.getpid()});
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

const TestMallocStatistics = extern struct {
    blocks_in_use: c_uint,
    size_in_use: usize,
    max_size_in_use: usize,
    size_allocated: usize,
};

extern "c" fn malloc_zone_statistics(zone: ?*anyopaque, stats: *TestMallocStatistics) void;

fn testTmpAngPath(buf: *[256]u8, name: []const u8) ![:0]const u8 {
    return paths.tmpPathZ(buf, name);
}

test "DomainNgramRegistry multi-pack F_NOCACHE routing, default switching, async prefetch, and zero steady-state malloc" {
    const gpa = std.testing.allocator;
    var p_repo_buf: [256]u8 = undefined;
    var p_docs_buf: [256]u8 = undefined;
    const p_repo = try testTmpAngPath(&p_repo_buf, "aminfer-test-c14-repo.ang");
    const p_docs = try testTmpAngPath(&p_docs_buf, "aminfer-test-c14-docs.ang");
    defer _ = std.c.unlink(p_repo.ptr);
    defer _ = std.c.unlink(p_docs.ptr);

    const repo_entries = [_]BuildEntry{
        .{ .ctx = &[_]u32{ 10, 20, 30, 40 }, .cont = &[_]u32{ 101, 102, 103, 104 }, .hit_count = 50, .step_prob = 0.94 },
        .{ .ctx = &[_]u32{ 30, 40 }, .cont = &[_]u32{ 105, 106 }, .hit_count = 12, .step_prob = 0.60 },
        .{ .ctx = &[_]u32{ 500, 501, 502, 503 }, .cont = &[_]u32{ 504, 505, 506 }, .hit_count = 80, .step_prob = 0.91 },
    };
    const docs_entries = [_]BuildEntry{
        .{ .ctx = &[_]u32{ 10, 20, 30, 40 }, .cont = &[_]u32{ 201, 202, 203 }, .hit_count = 75, .step_prob = 0.88 },
        .{ .ctx = &[_]u32{ 70, 80, 90, 100 }, .cont = &[_]u32{ 210, 211, 212, 213 }, .hit_count = 40, .step_prob = 0.85 },
    };

    try writeTableFile(gpa, p_repo, &repo_entries, .{
        .num_buckets = 8,
        .vocab_size = 4096,
        .corpus_id = "aminfer-repo-src",
        .include_table_crc = true,
    });
    try writeTableFile(gpa, p_docs, &docs_entries, .{
        .num_buckets = 8,
        .vocab_size = 4096,
        .corpus_id = "docs",
        .include_table_crc = true,
    });

    var reg = try DomainNgramRegistry.init(gpa, .{
        .cache_pages = 8,
        .max_inflight = 2,
        .io_backend = .kqueue_aio,
        .verify_table_crc = true,
    });
    defer reg.deinit(gpa);
    try std.testing.expectEqual(@as(u64, 9 * page.APPLE_PAGE_SIZE), reg.totalWiredBytes());

    // Register "repo" via caller alias, and "docs" defaulting to header corpus_id.
    const idx_repo = try reg.registerPack("repo", p_repo, true);
    const idx_docs = try reg.registerPack(null, p_docs, true);
    try std.testing.expectEqual(@as(usize, 0), idx_repo);
    try std.testing.expectEqual(@as(usize, 1), idx_docs);
    try std.testing.expectEqual(@as(usize, 2), reg.packCount());
    try std.testing.expect(reg.hasDomain("repo"));
    try std.testing.expect(reg.hasDomain("docs"));
    try std.testing.expect(!reg.hasDomain("unknown"));

    // Duplicate domain and invalid IDs rejected explicitly.
    try std.testing.expectError(error.DomainAlreadyExists, reg.registerPack("repo", p_repo, false));
    try std.testing.expectError(error.InvalidDomainId, reg.registerPack("", p_repo, false));
    try std.testing.expectError(error.DomainNotFound, reg.selectDefaultDomain("missing"));

    // Default domain starts as first registered ("repo").
    var def_buf: [MAX_DOMAIN_ID_BYTES]u8 = undefined;
    try std.testing.expectEqualStrings("repo", reg.defaultDomainId(&def_buf));

    var out: [8]u32 = undefined;
    const q_def_repo = try reg.queryDomain(null, &[_]u32{ 10, 20, 30, 40 }, &out);
    try std.testing.expectEqual(@as(u32, 4), q_def_repo.match_len);
    try std.testing.expectEqualSlices(u32, &[_]u32{ 101, 102, 103, 104 }, out[0..q_def_repo.draft_len]);

    // Explicit per-request routing to "docs" hits the other pack on the exact same context!
    const q_docs = try reg.queryDomain("docs", &[_]u32{ 10, 20, 30, 40 }, &out);
    try std.testing.expectEqual(@as(u32, 4), q_docs.match_len);
    try std.testing.expectEqualSlices(u32, &[_]u32{ 201, 202, 203 }, out[0..q_docs.draft_len]);

    // Async prefetch + query hit on "repo".
    const pref_ok = try reg.prefetchDomain("repo", &[_]u32{ 500, 501, 502, 503 });
    try std.testing.expect(pref_ok);
    const q_pref = try reg.queryDomain("repo", &[_]u32{ 500, 501, 502, 503 }, &out);
    try std.testing.expect(q_pref.cache_hit);
    try std.testing.expectEqualSlices(u32, &[_]u32{ 504, 505, 506 }, out[0..q_pref.draft_len]);

    // Switch default domain live to "docs" and query with `null`.
    try reg.selectDefaultDomain("docs");
    try std.testing.expectEqualStrings("docs", reg.defaultDomainId(&def_buf));
    const q_def_docs = try reg.queryDomain(null, &[_]u32{ 10, 20, 30, 40 }, &out);
    try std.testing.expect(q_def_docs.cache_hit);
    try std.testing.expectEqualSlices(u32, &[_]u32{ 201, 202, 203 }, out[0..q_def_docs.draft_len]);

    // Unknown explicit domain returns DomainNotFound.
    try std.testing.expectError(error.DomainNotFound, reg.queryDomain("missing", &[_]u32{ 10, 20, 30, 40 }, &out));

    // Snapshot packs metadata and stats cleanly.
    var infos: [MAX_DOMAIN_PACKS]DomainPackInfo = undefined;
    const n_infos = reg.snapshotPacks(&infos);
    try std.testing.expectEqual(@as(usize, 2), n_infos);
    try std.testing.expectEqualStrings("repo", infos[0].domainIdSlice());
    try std.testing.expectEqualStrings("aminfer-repo-src", infos[0].corpusIdSlice());
    try std.testing.expect(!infos[0].is_default);
    try std.testing.expectEqual(@as(u64, 3), infos[0].total_entries);
    try std.testing.expectEqualStrings("docs", infos[1].domainIdSlice());
    try std.testing.expect(infos[1].is_default);
    try std.testing.expect(infos[1].stats.hits >= 2);

    // Steady-state zero-malloc verification across query, prefetch, select, snapshot.
    var m_warm: TestMallocStatistics = std.mem.zeroes(TestMallocStatistics);
    malloc_zone_statistics(null, &m_warm);
    var m_before: TestMallocStatistics = std.mem.zeroes(TestMallocStatistics);
    malloc_zone_statistics(null, &m_before);

    for (0..32) |iter| {
        const dom = if ((iter & 1) == 0) "repo" else "docs";
        try reg.selectDefaultDomain(dom);
        _ = try reg.queryDomain(null, &[_]u32{ 10, 20, 30, 40 }, &out);
        _ = try reg.prefetchDomain("docs", &[_]u32{ 70, 80, 90, 100 });
        _ = try reg.queryDomain("docs", &[_]u32{ 70, 80, 90, 100 }, &out);
        _ = reg.snapshotPacks(&infos);
    }

    var m_after: TestMallocStatistics = std.mem.zeroes(TestMallocStatistics);
    malloc_zone_statistics(null, &m_after);
    try std.testing.expect(m_after.blocks_in_use <= m_before.blocks_in_use);
    try std.testing.expect(m_after.size_in_use <= m_before.size_in_use);
}

test "DomainNgramRegistry atomic hot-swap (swapDomainPack) under concurrent multi-threaded queries has zero races and zero malloc" {
    const gpa = std.testing.allocator;
    var p_v1_buf: [256]u8 = undefined;
    var p_v2_buf: [256]u8 = undefined;
    const p_v1 = try testTmpAngPath(&p_v1_buf, "aminfer-test-c14-swap-v1.ang");
    const p_v2 = try testTmpAngPath(&p_v2_buf, "aminfer-test-c14-swap-v2.ang");
    defer _ = std.c.unlink(p_v1.ptr);
    defer _ = std.c.unlink(p_v2.ptr);

    const v1_entries = [_]BuildEntry{
        .{ .ctx = &[_]u32{ 1, 2, 3, 4 }, .cont = &[_]u32{ 111, 112, 113, 114 }, .hit_count = 30, .step_prob = 0.90 },
    };
    const v2_entries = [_]BuildEntry{
        .{ .ctx = &[_]u32{ 1, 2, 3, 4 }, .cont = &[_]u32{ 221, 222, 223, 224 }, .hit_count = 90, .step_prob = 0.95 },
    };
    try writeTableFile(gpa, p_v1, &v1_entries, .{ .num_buckets = 4, .corpus_id = "hot-v1", .include_table_crc = true });
    try writeTableFile(gpa, p_v2, &v2_entries, .{ .num_buckets = 4, .corpus_id = "hot-v2", .include_table_crc = true });

    var reg = try DomainNgramRegistry.init(gpa, .{
        .cache_pages = 4,
        .max_inflight = 2,
        .io_backend = .kqueue_aio,
        .verify_table_crc = true,
    });
    defer reg.deinit(gpa);
    _ = try reg.registerPack("live", p_v1, true);

    // Warm up once before concurrent swap stress.
    var warm_out: [8]u32 = undefined;
    const warm_q = try reg.queryDomain("live", &[_]u32{ 1, 2, 3, 4 }, &warm_out);
    try std.testing.expectEqualSlices(u32, &[_]u32{ 111, 112, 113, 114 }, warm_out[0..warm_q.draft_len]);

    const ReaderState = struct {
        reg: *DomainNgramRegistry,
        stop: std.atomic.Value(bool) = .init(false),
        v1_seen: std.atomic.Value(u64) = .init(0),
        v2_seen: std.atomic.Value(u64) = .init(0),
        errors: std.atomic.Value(u64) = .init(0),
        bad_payloads: std.atomic.Value(u64) = .init(0),

        fn run(st: *@This()) void {
            var out: [8]u32 = undefined;
            var iters: u64 = 0;
            while (!st.stop.load(.acquire)) : (iters += 1) {
                if ((iters & 7) == 0) {
                    _ = st.reg.prefetchDomain("live", &[_]u32{ 1, 2, 3, 4 }) catch {
                        _ = st.errors.fetchAdd(1, .monotonic);
                        continue;
                    };
                }
                const res = st.reg.queryDomain("live", &[_]u32{ 1, 2, 3, 4 }, &out) catch {
                    _ = st.errors.fetchAdd(1, .monotonic);
                    continue;
                };
                if (res.draft_len == 4 and std.mem.eql(u32, out[0..4], &[_]u32{ 111, 112, 113, 114 })) {
                    _ = st.v1_seen.fetchAdd(1, .monotonic);
                } else if (res.draft_len == 4 and std.mem.eql(u32, out[0..4], &[_]u32{ 221, 222, 223, 224 })) {
                    _ = st.v2_seen.fetchAdd(1, .monotonic);
                } else {
                    _ = st.bad_payloads.fetchAdd(1, .monotonic);
                }
            }
        }
    };

    var st = ReaderState{ .reg = &reg };
    const t1 = try std.Thread.spawn(.{}, ReaderState.run, .{&st});
    const t2 = try std.Thread.spawn(.{}, ReaderState.run, .{&st});

    var last_gen: u64 = 1;
    for (0..16) |round| {
        const target = if ((round & 1) == 0) p_v2 else p_v1;
        const gen = try reg.swapDomainPack("live", target, true);
        try std.testing.expect(gen > last_gen);
        last_gen = gen;
    }

    st.stop.store(true, .release);
    t1.join();
    t2.join();

    try std.testing.expectEqual(@as(u64, 0), st.errors.load(.acquire));
    try std.testing.expectEqual(@as(u64, 0), st.bad_payloads.load(.acquire));
    try std.testing.expect(st.v1_seen.load(.acquire) > 0);
    try std.testing.expect(st.v2_seen.load(.acquire) > 0);

    // After final swap (round 15 = odd -> p_v1), swap once more to p_v2 and confirm
    // stale v1 cache entries are immediately bypassed with zero heap allocations!
    var m_warm: TestMallocStatistics = std.mem.zeroes(TestMallocStatistics);
    malloc_zone_statistics(null, &m_warm);
    var m_before: TestMallocStatistics = std.mem.zeroes(TestMallocStatistics);
    malloc_zone_statistics(null, &m_before);

    _ = try reg.swapDomainPack("live", p_v2, true);
    const after_q = try reg.queryDomain("live", &[_]u32{ 1, 2, 3, 4 }, &warm_out);
    try std.testing.expectEqual(@as(u32, 4), after_q.draft_len);
    try std.testing.expect(!after_q.cache_hit);
    try std.testing.expectEqualSlices(u32, &[_]u32{ 221, 222, 223, 224 }, warm_out[0..4]);

    var m_after: TestMallocStatistics = std.mem.zeroes(TestMallocStatistics);
    malloc_zone_statistics(null, &m_after);
    try std.testing.expect(m_after.blocks_in_use <= m_before.blocks_in_use);
    try std.testing.expect(m_after.size_in_use <= m_before.size_in_use);

    var infos: [MAX_DOMAIN_PACKS]DomainPackInfo = undefined;
    try std.testing.expectEqual(@as(usize, 1), reg.snapshotPacks(&infos));
    try std.testing.expectEqual(@as(u64, 17), infos[0].swaps);
    try std.testing.expectEqualStrings("hot-v2", infos[0].corpusIdSlice());
}

test "F15.3 swapDomainPack preserves live new_bank fd on drain error, F19.1 skips corrupt slots, F19.3 applies all batch completions" {
    const gpa = std.testing.allocator;

    // 1. F19.1: corrupt slot with cont_len > MAX_CONT_LEN and OOB cont_tokens >= vocab_size are skipped cleanly.
    var p_corrupt_buf: [256]u8 = undefined;
    const p_corrupt = try testTmpAngPath(&p_corrupt_buf, "aminfer-test-f19-corrupt.ang");
    defer _ = std.c.unlink(p_corrupt.ptr);

    const ctx = [_]u32{ 1, 2, 3, 4 };
    const entries = [_]BuildEntry{
        .{ .ctx = &ctx, .cont = &[_]u32{ 5, 6 }, .hit_count = 100, .step_prob = 0.9 },
        .{ .ctx = &ctx, .cont = &[_]u32{ 7, 8 }, .hit_count = 50, .step_prob = 0.8 },
    };
    var img = try buildTableBytes(gpa, &entries, .{
        .num_buckets = 1,
        .vocab_size = 100,
        .include_table_crc = false,
    });
    defer gpa.free(img);

    // Corrupt slot 0 in bucket page 0: set cont_len = 25 (> MAX_CONT_LEN=8), and slot 1 cont_tokens[0] = 999 (>= vocab_size=100).
    const slots: []NgramSlot = @alignCast(std.mem.bytesAsSlice(NgramSlot, img[page.APPLE_PAGE_SIZE .. 2 * page.APPLE_PAGE_SIZE]));
    slots[0].cont_len = 25;
    slots[1].cont_tokens[0] = 999;

    {
        const fd = std.c.open(p_corrupt.ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o644));
        try std.testing.expect(fd >= 0);
        defer _ = std.c.close(fd);
        var woff: usize = 0;
        while (woff < img.len) {
            const rc = std.c.write(fd, img[woff..].ptr, img.len - woff);
            try std.testing.expect(rc > 0);
            woff += @intCast(rc);
        }
    }

    var tbl = try NvmeNgramTable.open(gpa, p_corrupt, .{
        .cache_pages = 4,
        .max_inflight = 2,
        .verify_table_crc = false,
    });
    defer tbl.deinit(gpa);

    var out_cont: [16]u32 = undefined;
    const qres = try tbl.query(&ctx, &out_cont);
    try std.testing.expectEqual(@as(u32, 0), qres.draft_len);

    // 2. F19.3: applyCompletionsAll updates ALL completions in a batch even when comps[0] fails.
    tbl.slots[0] = .{ .state = .inflight, .bucket_idx = 0, .last_used_tick = 1 };
    tbl.slots[1] = .{ .state = .inflight, .bucket_idx = 0, .last_used_tick = 2 };
    const comps = [_]aio.Completion{
        .{ .user_data = 0, .bytes_read = 0, .err_code = 5, .short_read = false },
        .{ .user_data = 1, .bytes_read = page.APPLE_PAGE_SIZE, .err_code = 0, .short_read = false },
    };
    try std.testing.expectError(error.ReadFailed, tbl.applyCompletionsAll(&comps));
    try std.testing.expectEqual(NvmeNgramTable.SlotState.empty, tbl.slots[0].state);
    try std.testing.expectEqual(NvmeNgramTable.SlotState.ready, tbl.slots[1].state);

    // 3. F15.3: DomainNgramRegistry.swapDomainPack keeps new_bank.file.fd open if drain hits an AIO error.
    var p1_buf: [256]u8 = undefined;
    var p2_buf: [256]u8 = undefined;
    const p1 = try testTmpAngPath(&p1_buf, "aminfer-test-f15-v1.ang");
    const p2 = try testTmpAngPath(&p2_buf, "aminfer-test-f15-v2.ang");
    defer _ = std.c.unlink(p1.ptr);
    defer _ = std.c.unlink(p2.ptr);

    try writeTableFile(gpa, p1, &[_]BuildEntry{.{ .ctx = &ctx, .cont = &[_]u32{ 11, 12 } }}, .{
        .num_buckets = 2,
        .vocab_size = 100,
        .corpus_id = "v1",
        .include_table_crc = true,
    });
    try writeTableFile(gpa, p2, &[_]BuildEntry{.{ .ctx = &ctx, .cont = &[_]u32{ 21, 22 } }}, .{
        .num_buckets = 2,
        .vocab_size = 100,
        .corpus_id = "v2",
        .include_table_crc = true,
    });

    var reg = try DomainNgramRegistry.init(gpa, .{
        .cache_pages = 4,
        .max_inflight = 2,
        .io_backend = .kqueue_aio,
        .verify_table_crc = true,
    });
    defer reg.deinit(gpa);
    const pidx = try reg.registerPack("live", p1, true);

    // Simulate an in-flight short-read prefetch pinned to old bank 0 by queueing a past-EOF AIO read on p1's fd!
    const short_fd = std.c.open(p1.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    try std.testing.expect(short_fd >= 0);
    defer _ = std.c.close(short_fd);
    _ = reg.packs[pidx].banks[0].pins.fetchAdd(1, .acq_rel);
    reg.cache_slots[0] = .{
        .state = .inflight,
        .pack_idx = @intCast(pidx),
        .bank_idx = 0,
        .desc_gen = reg.packs[pidx].banks[0].generation,
        .bucket_idx = 0,
        .last_used_tick = 1,
        .readers = 0,
    };
    try reg.async_reader.submit(.{
        .fd = short_fd,
        .offset = 16 * page.APPLE_PAGE_SIZE,
        .dst = reg.pageSlice(0),
        .user_data = 0,
    });

    const swap_res = reg.swapDomainPack("live", p2, true);
    try std.testing.expectError(error.ShortRead, swap_res);
    // New live bank MUST still have a valid open fd and serve v2 queries cleanly!
    const live_bank = reg.packs[pidx].active_bank.load(.acquire);
    try std.testing.expect(reg.packs[pidx].banks[live_bank].file.fd >= 0);
    try std.testing.expectEqual(@as(std.c.fd_t, -1), reg.packs[pidx].banks[1 - live_bank].file.fd);
    const after_q = try reg.queryDomain("live", &ctx, &out_cont);
    try std.testing.expectEqualSlices(u32, &[_]u32{ 21, 22 }, out_cont[0..after_q.draft_len]);
}
