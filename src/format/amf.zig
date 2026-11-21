//! aminfer Model Format (`.amf` v0) binary specification, startup validator,
//! and wired-arena loader (ADR-0006). Every section and blob is 16 KiB page
//! aligned and protected by CRC-32 IEEE checksums.

const std = @import("std");
const page = @import("../mem/page.zig");
const arena_mod = @import("../mem/arena.zig");
const budget_mod = @import("../mem/budget.zig");
const expert = @import("../expert/source.zig");
const quant = @import("../quant/spec.zig");

pub const MAGIC: [8]u8 = .{ 'A', 'M', 'F', '0', 0, 0, 0, 0 };
pub const FORMAT_VERSION: u32 = 1;
pub const CHECKSUM_CRC32_IEEE: u32 = 1;
pub const NO_LAYER: u16 = 0xFFFF;
pub const NO_EXPERT: u16 = 0xFFFF;

/// ARMv9 SME2 `sme2_panel64` geometry constants (ADR-0018, W15).
pub const SME2_PANEL_COLS: u64 = 16;
pub const SME2_GROUP_K: u64 = 64;
pub const SME2_QUANT_HEADER_BYTES: u64 = 128;
pub const SME2_F16_GROUP_BYTES: u64 = 2048; // 32 kp chunks * 32 u16 lanes
pub const SME2_Q4_GROUP_BYTES: u64 = 640; // 128 B inline header + 512 B packed codes
pub const SME2_Q3_GROUP_BYTES: u64 = 512; // 128 B inline header + 384 B packed codes
pub const SME2_Q2_GROUP_BYTES: u64 = 384; // 128 B inline header + 256 B packed codes

/// Single-owner tensor byte layout stored in the low byte of `TensorEntry.reserved0`
/// (ADR-0018, W15). Existing `.amf` v0 files carry `reserved0 == 0` (`row_major`).
pub const TensorLayout = enum(u8) {
    row_major = 0,
    sme2_panel64 = 1,
};

/// Quantization parameter encoding stored in the high byte of `TensorEntry.reserved0`
/// (ADR-0017, ADR-0018). Existing `.amf` v0 files carry `reserved0 == 0` (`affine_scale_bias`).
pub const QuantParamKind = enum(u8) {
    affine_scale_bias = 0,
    lut16_codebook = 1,
};

pub const TensorKind = enum(u8) {
    dense = 0,
    hot_expert = 1,
    cold_expert = 2,
    embedding = 3,
    norm = 4,
    router = 5,
    lm_head = 6,
    /// Wired shared MoE expert (ADR-0015, Qwen3.8 / DeepSeek-V4.1).
    shared_expert = 7,
    /// Cold 16 KiB-aligned NVMe `F_NOCACHE` PLE / Engram row table (ADR-0015, C10).
    /// Must remain on disk (excluded from `totalWiredBytes()` and `loadWired()`).
    sparse_table = 8,
    /// Wired recurrent `Step-LoRA` adapter (`lora_a`, `lora_b`, `alpha`, ADR-0019, W16).
    loop_adapter = 9,

    /// Returns `true` for tensor kinds loaded into startup wired RAM and `false`
    /// for cold NVMe-streamed kinds (`cold_expert`, `sparse_table`).
    pub fn isWired(self: TensorKind) bool {
        return switch (self) {
            .dense, .hot_expert, .embedding, .norm, .router, .lm_head, .shared_expert, .loop_adapter => true,
            .cold_expert, .sparse_table => false,
        };
    }

    pub fn isCold(self: TensorKind) bool {
        return !self.isWired();
    }
};

pub const DType = enum(u8) {
    f16 = 0,
    f32 = 1,
    q4_affine = 2,
    q3_affine = 3,
    /// Unquantized bfloat16 (ADR-0016): `quant_bits = 0`, 2 bytes per element.
    bf16 = 4,
    /// Unquantized 32-bit unsigned integer (e.g. EAGLE-3 `d2t` vocab maps,
    /// ADR-0013): `quant_bits = 0`, 4 bytes per element.
    u32 = 5,
    /// MLX-compatible 2-bit affine group-64 quantization (ADR-0017, W14/G13):
    /// `quant_bits = 2`, 16 B packed codes + 4 B `half2(scale, bias)` per 64 weights.
    q2_affine = 6,

    /// Bytes per element for unquantized dtypes; `null` for affine-quantized ones.
    pub fn unquantizedElemSize(self: DType) ?u64 {
        return switch (self) {
            .f16, .bf16 => 2,
            .f32, .u32 => 4,
            .q4_affine, .q3_affine, .q2_affine => null,
        };
    }
};

/// Chunk size used to stream blob CRC verification from disk (bounded memory).
pub const VERIFY_CHUNK_BYTES: usize = 8 * 1024 * 1024;

comptime {
    if (VERIFY_CHUNK_BYTES % page.APPLE_PAGE_SIZE != 0) @compileError("VERIFY_CHUNK_BYTES must be 16 KiB aligned");
}

/// 128-byte wire header at byte 0 of page 0 (little-endian, zero implicit padding).
pub const Header = extern struct {
    magic: [8]u8,
    version: u32,
    flags: u32,
    page_size: u32,
    checksum_algo: u32,
    tensor_count: u32,
    reserved0: u32,
    directory_offset: u64,
    directory_bytes: u64,
    directory_crc32: u32,
    header_crc32: u32,
    model_id: [64]u8,
    reserved1: u64,
};

/// 128-byte fixed directory record (little-endian, zero implicit padding).
pub const TensorEntry = extern struct {
    name: [48]u8,
    kind: u8,
    dtype: u8,
    quant_bits: u8,
    rank: u8,
    group_size: u16,
    layer: u16,
    expert: u16,
    reserved0: u16,
    blob_crc32: u32,
    shape: [4]u32,
    blob_offset: u64,
    blob_bytes: u64,
    data_bytes: u64,
    params_offset: u64,
    params_bytes: u64,
    reserved1: u64,

    pub fn zeroed(name_str: []const u8, kind: TensorKind, dtype: DType) TensorEntry {
        var entry: TensorEntry = std.mem.zeroes(TensorEntry);
        const copy_len = @min(name_str.len, entry.name.len - 1);
        @memcpy(entry.name[0..copy_len], name_str[0..copy_len]);
        entry.kind = @backingInt(kind);
        entry.dtype = @backingInt(dtype);
        entry.layer = NO_LAYER;
        entry.expert = NO_EXPERT;
        return entry;
    }

    pub fn nameSlice(self: *const TensorEntry) []const u8 {
        return std.mem.sliceTo(&self.name, 0);
    }

    pub fn tensorKind(self: *const TensorEntry) ParseError!TensorKind {
        return std.enums.fromInt(TensorKind, self.kind) orelse error.InvalidTensorKind;
    }

    pub fn dataType(self: *const TensorEntry) ParseError!DType {
        return std.enums.fromInt(DType, self.dtype) orelse error.InvalidDType;
    }

    pub fn tensorLayout(self: *const TensorEntry) ParseError!TensorLayout {
        const raw: u8 = @truncate(self.reserved0 & 0x00FF);
        return std.enums.fromInt(TensorLayout, raw) orelse error.InvalidTensorLayout;
    }

    pub fn quantParamKind(self: *const TensorEntry) ParseError!QuantParamKind {
        const raw: u8 = @truncate(self.reserved0 >> 8);
        return std.enums.fromInt(QuantParamKind, raw) orelse error.InvalidQuantParamKind;
    }

    pub fn setLayoutAndParamKind(self: *TensorEntry, layout: TensorLayout, param_kind: QuantParamKind) void {
        self.reserved0 = @as(u16, @backingInt(layout)) | (@as(u16, @backingInt(param_kind)) << 8);
    }

    /// Returns `true` when this tensor is wired in RAM at startup (`false` for
    /// cold NVMe tensors such as `cold_expert` and `sparse_table`).
    pub fn isWired(self: *const TensorEntry) ParseError!bool {
        return (try self.tensorKind()).isWired();
    }

    /// Exposes a cold-expert or sparse-table directory record as an `ExpertSource.local_nvme_ssd`.
    pub fn asNvmeSource(self: *const TensorEntry, file_index: u16) expert.NvmeExpert.InitError!expert.ExpertSource {
        return .{ .local_nvme_ssd = try expert.NvmeExpert.init(file_index, self.blob_offset, self.blob_bytes) };
    }
};

comptime {
    if (@sizeOf(Header) != 128) @compileError("AMF Header must be 128 bytes");
    if (@sizeOf(TensorEntry) != 128) @compileError("AMF TensorEntry must be 128 bytes");
    if (@offsetOf(TensorEntry, "reserved0") != 58) @compileError("AMF TensorEntry.reserved0 must remain at byte offset 58");
}

/// Returns the exact packed payload byte length (`data_bytes`) of a 2D `[n, k]`
/// tensor in `TensorLayout.sme2_panel64` (ADR-0018, W15).
///
/// Rows `n .. ((n + 15) & ~15)` are zero-padded; `k` must be a non-zero multiple
/// of `64`. Quantized dtypes embed their 128 B per-(panel, group) header inline
/// inside `data_bytes` (`params_offset = 0`, `params_bytes = 0`).
pub fn sme2Panel64BlobBytes(dtype: DType, n: u64, k: u64, group_size: u16) ParseError!u64 {
    if (n == 0 or k == 0 or k % SME2_GROUP_K != 0) return error.InvalidShape;
    const group_bytes: u64 = switch (dtype) {
        .f16, .bf16 => blk: {
            if (group_size != 0) return error.InvalidQuantization;
            break :blk SME2_F16_GROUP_BYTES;
        },
        .q4_affine => blk: {
            if (group_size != SME2_GROUP_K) return error.InvalidQuantization;
            break :blk SME2_Q4_GROUP_BYTES;
        },
        .q3_affine => blk: {
            if (group_size != SME2_GROUP_K) return error.InvalidQuantization;
            break :blk SME2_Q3_GROUP_BYTES;
        },
        .q2_affine => blk: {
            if (group_size != SME2_GROUP_K) return error.InvalidQuantization;
            break :blk SME2_Q2_GROUP_BYTES;
        },
        .f32, .u32 => return error.InvalidTensorLayout,
    };
    const n_plus_pad = std.math.add(u64, n, SME2_PANEL_COLS - 1) catch return error.InvalidShape;
    const panels = n_plus_pad / SME2_PANEL_COLS;
    const groups = k / SME2_GROUP_K;
    const panel_groups = std.math.mul(u64, panels, groups) catch return error.InvalidShape;
    return std.math.mul(u64, panel_groups, group_bytes) catch return error.InvalidShape;
}

pub inline fn align64(len: usize) usize {
    return (len + 63) & ~@as(usize, 63);
}

/// Computes the CRC-32 IEEE of `hdr` with `header_crc32` treated as zero.
pub fn computeHeaderCrc32(hdr: *const Header) u32 {
    var copy = hdr.*;
    copy.header_crc32 = 0;
    return std.hash.Crc32.hash(std.mem.asBytes(&copy));
}

pub const ParseError = error{
    FileTooSmall,
    MisalignedFileSize,
    InvalidMagic,
    UnsupportedVersion,
    UnsupportedPageSize,
    UnsupportedChecksum,
    HeaderChecksumMismatch,
    MisalignedDirectory,
    DirectoryOutOfBounds,
    DirectoryChecksumMismatch,
    EmptyTensorName,
    InvalidTensorKind,
    InvalidDType,
    InvalidTensorLayout,
    InvalidQuantParamKind,
    InvalidRank,
    InvalidQuantization,
    InvalidShape,
    MisalignedBlob,
    MisalignedParams,
    BlobOutOfBounds,
    OverlappingBlobs,
    BlobChecksumMismatch,
    DuplicateTensorName,
    OutOfMemory,
};

pub const IoError = std.posix.UnexpectedError || error{
    OpenFailed,
    StatFailed,
    FcntlFailed,
    ShortRead,
    ReadFailed,
};

/// Name -> directory index map. Keys borrow the `name` bytes of the owning
/// `Parsed.entries` slice, which stays at a fixed heap address for the lifetime
/// of the `Parsed` value.
pub const NameIndex = std.StringHashMapUnmanaged(u32);

pub const Parsed = struct {
    header: Header,
    entries: []TensorEntry,
    file_size: u64,
    /// O(1) tensor lookup built during validation; freed by `deinit`.
    index: NameIndex = .empty,

    pub fn deinit(self: *Parsed, allocator: std.mem.Allocator) void {
        self.index.deinit(allocator);
        self.index = .empty;
        allocator.free(self.entries);
        self.entries = &.{};
    }

    pub fn modelId(self: *const Parsed) []const u8 {
        return std.mem.sliceTo(&self.header.model_id, 0);
    }

    /// O(1) average lookup of a tensor entry by name.
    pub fn find(self: *const Parsed, name: []const u8) ?*const TensorEntry {
        if (self.index.count() == self.entries.len) {
            const idx = self.index.get(name) orelse return null;
            return &self.entries[idx];
        }
        // Fallback for a hand-assembled `Parsed` whose index was never built.
        for (self.entries) |*entry| {
            if (std.mem.eql(u8, entry.nameSlice(), name)) return entry;
        }
        return null;
    }

    /// Parses and validates an `.amf` image held in memory (header, directory
    /// CRC, alignment, bounds and non-overlapping blob layout).
    pub fn parseBytes(allocator: std.mem.Allocator, bytes: []const u8) ParseError!Parsed {
        if (bytes.len < page.APPLE_PAGE_SIZE) return error.FileTooSmall;
        if (bytes.len % page.APPLE_PAGE_SIZE != 0) return error.MisalignedFileSize;

        var hdr: Header = undefined;
        @memcpy(std.mem.asBytes(&hdr), bytes[0..@sizeOf(Header)]);
        try validateHeader(&hdr, bytes.len);

        const count: usize = hdr.tensor_count;
        const entries = try allocator.alloc(TensorEntry, count);
        errdefer allocator.free(entries);

        const dir_raw_len = count * @sizeOf(TensorEntry);
        const dir_off: usize = @intCast(hdr.directory_offset);
        if (count > 0) {
            const dir_slice = bytes[dir_off .. dir_off + dir_raw_len];
            if (std.hash.Crc32.hash(dir_slice) != hdr.directory_crc32) {
                return error.DirectoryChecksumMismatch;
            }
            @memcpy(std.mem.sliceAsBytes(entries), dir_slice);
        } else if (hdr.directory_crc32 != std.hash.Crc32.hash(&.{})) {
            return error.DirectoryChecksumMismatch;
        }

        var index = try validateEntries(allocator, &hdr, entries, bytes.len);
        errdefer index.deinit(allocator);
        return .{ .header = hdr, .entries = entries, .file_size = bytes.len, .index = index };
    }

    /// Verifies the CRC-32 of every blob in an in-memory `.amf` image.
    pub fn verifyAllBlobsFromBytes(self: *const Parsed, bytes: []const u8) ParseError!void {
        if (bytes.len < self.file_size) return error.BlobOutOfBounds;
        for (self.entries) |*entry| {
            const start: usize = @intCast(entry.blob_offset);
            const len: usize = @intCast(entry.blob_bytes);
            if (std.hash.Crc32.hash(bytes[start .. start + len]) != entry.blob_crc32) {
                return error.BlobChecksumMismatch;
            }
        }
    }

    /// Returns total 16 KiB-aligned blob bytes across all startup-wired tensors.
    /// Cold NVMe tensors (`cold_expert` and `sparse_table`) are excluded.
    pub fn totalWiredBytes(self: *const Parsed) u64 {
        var total: u64 = 0;
        for (self.entries) |*entry| {
            const kind = std.enums.fromInt(TensorKind, entry.kind) orelse continue;
            if (kind.isWired()) {
                total += entry.blob_bytes;
            }
        }
        return total;
    }

    /// Returns total 16 KiB-aligned blob bytes across all cold NVMe tensors
    /// (`cold_expert` and `sparse_table`).
    pub fn totalColdBytes(self: *const Parsed) u64 {
        var total: u64 = 0;
        for (self.entries) |*entry| {
            const kind = std.enums.fromInt(TensorKind, entry.kind) orelse continue;
            if (kind.isCold()) {
                total += entry.blob_bytes;
            }
        }
        return total;
    }

    /// Loads every startup-wired blob from an in-memory `.amf` image into `arena`
    /// with CRC-32 verification. Cold entries (`cold_expert`, `sparse_table`) stay
    /// `null` and consume zero wired bytes. Caller owns the returned pointer slice.
    pub fn loadWiredFromBytes(
        self: *const Parsed,
        allocator: std.mem.Allocator,
        bytes: []const u8,
        arena: *arena_mod.WiredArena,
    ) (ParseError || arena_mod.WiredArena.AllocError)![]?[]align(page.APPLE_PAGE_SIZE) u8 {
        const out = try allocator.alloc(?[]align(page.APPLE_PAGE_SIZE) u8, self.entries.len);
        errdefer allocator.free(out);
        @memset(out, null);

        for (self.entries, 0..) |*entry, i| {
            if (try entry.isWired()) {
                out[i] = try self.loadBlobFromBytes(bytes, entry, arena);
            }
        }
        return out;
    }

    /// Copies one validated blob from an in-memory `.amf` image into a 16 KiB
    /// page slice carved from `arena` and verifies its CRC-32.
    pub fn loadBlobFromBytes(
        self: *const Parsed,
        bytes: []const u8,
        entry: *const TensorEntry,
        arena: *arena_mod.WiredArena,
    ) (ParseError || arena_mod.WiredArena.AllocError)![]align(page.APPLE_PAGE_SIZE) u8 {
        _ = self;
        const start: usize = @intCast(entry.blob_offset);
        const len: usize = @intCast(entry.blob_bytes);
        if (start + len > bytes.len) return error.BlobOutOfBounds;
        const dst = try arena.allocPages(len);
        @memcpy(dst, bytes[start .. start + len]);
        if (std.hash.Crc32.hash(dst) != entry.blob_crc32) {
            return error.BlobChecksumMismatch;
        }
        return dst;
    }
};

/// Direct-I/O (`F_NOCACHE`) `.amf` file handle validated at startup.
pub const AmfFile = struct {
    fd: std.c.fd_t,
    parsed: Parsed,

    pub const OpenError = ParseError || IoError;

    /// Opens an `.amf` file with `F_NOCACHE`, reads and validates page 0 and
    /// the directory, and checks all alignment and overlap invariants.
    pub fn open(allocator: std.mem.Allocator, path: [:0]const u8) OpenError!AmfFile {
        const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
        if (fd < 0) return error.OpenFailed;
        errdefer _ = std.c.close(fd);

        if (std.c.fcntl(fd, std.c.F.NOCACHE, @as(c_int, 1)) != 0) {
            return error.FcntlFailed;
        }

        var st: std.c.Stat = undefined;
        if (std.c.fstat(fd, &st) != 0) return error.StatFailed;
        if (st.size < page.APPLE_PAGE_SIZE) return error.FileTooSmall;
        const file_size: u64 = @intCast(st.size);
        if (file_size % page.APPLE_PAGE_SIZE != 0) return error.MisalignedFileSize;

        var hdr_page: [page.APPLE_PAGE_SIZE]u8 align(page.APPLE_PAGE_SIZE) = undefined;
        try preadExact(fd, &hdr_page, 0);

        var hdr: Header = undefined;
        @memcpy(std.mem.asBytes(&hdr), hdr_page[0..@sizeOf(Header)]);
        try validateHeader(&hdr, file_size);

        const count: usize = hdr.tensor_count;
        const entries = try allocator.alloc(TensorEntry, count);
        errdefer allocator.free(entries);

        if (count > 0) {
            const dir_len: usize = @intCast(hdr.directory_bytes);
            const dir_buf = try allocator.alignedAlloc(u8, .fromByteUnits(page.APPLE_PAGE_SIZE), dir_len);
            defer allocator.free(dir_buf);
            try preadExact(fd, dir_buf, hdr.directory_offset);

            const dir_raw_len = count * @sizeOf(TensorEntry);
            if (std.hash.Crc32.hash(dir_buf[0..dir_raw_len]) != hdr.directory_crc32) {
                return error.DirectoryChecksumMismatch;
            }
            @memcpy(std.mem.sliceAsBytes(entries), dir_buf[0..dir_raw_len]);
        } else if (hdr.directory_crc32 != std.hash.Crc32.hash(&.{})) {
            return error.DirectoryChecksumMismatch;
        }

        var index = try validateEntries(allocator, &hdr, entries, file_size);
        errdefer index.deinit(allocator);
        return .{
            .fd = fd,
            .parsed = .{
                .header = hdr,
                .entries = entries,
                .file_size = file_size,
                .index = index,
            },
        };
    }

    pub fn close(self: *AmfFile, allocator: std.mem.Allocator) void {
        self.parsed.deinit(allocator);
        _ = std.c.close(self.fd);
    }

    pub fn totalWiredBytes(self: *const AmfFile) u64 {
        return self.parsed.totalWiredBytes();
    }

    pub fn totalColdBytes(self: *const AmfFile) u64 {
        return self.parsed.totalColdBytes();
    }

    /// Loads every startup-wired blob from disk via `F_NOCACHE` `pread` into
    /// `arena` with CRC-32 verification. Cold entries (`cold_expert`, `sparse_table`)
    /// stay `null`, perform zero disk reads, and consume zero wired bytes.
    /// Caller owns the returned slice of optional blob pointers.
    pub fn loadWired(
        self: *const AmfFile,
        allocator: std.mem.Allocator,
        arena: *arena_mod.WiredArena,
    ) (ParseError || IoError || arena_mod.WiredArena.AllocError)![]?[]align(page.APPLE_PAGE_SIZE) u8 {
        const out = try allocator.alloc(?[]align(page.APPLE_PAGE_SIZE) u8, self.parsed.entries.len);
        errdefer allocator.free(out);
        @memset(out, null);

        for (self.parsed.entries, 0..) |*entry, i| {
            if (try entry.isWired()) {
                out[i] = try self.loadBlob(entry, arena);
            }
        }
        return out;
    }

    /// Reads one blob directly into a 16 KiB-aligned `WiredArena` slice via
    /// `F_NOCACHE` `pread` and verifies its CRC-32.
    pub fn loadBlob(
        self: *const AmfFile,
        entry: *const TensorEntry,
        arena: *arena_mod.WiredArena,
    ) (ParseError || IoError || arena_mod.WiredArena.AllocError)![]align(page.APPLE_PAGE_SIZE) u8 {
        const len: usize = @intCast(entry.blob_bytes);
        const dst = try arena.allocPages(len);
        try preadExact(self.fd, dst, entry.blob_offset);
        if (std.hash.Crc32.hash(dst) != entry.blob_crc32) {
            return error.BlobChecksumMismatch;
        }
        return dst;
    }

    /// Streams every blob through one reusable, bounded scratch buffer
    /// (at most `VERIFY_CHUNK_BYTES`) to verify all blob CRCs on disk, so
    /// multi-GiB `.amf` files verify with constant memory.
    pub fn verifyAllBlobs(self: *const AmfFile, allocator: std.mem.Allocator) (ParseError || IoError)!void {
        return self.verifyAllBlobsChunked(allocator, VERIFY_CHUNK_BYTES);
    }

    /// `verifyAllBlobs` with an explicit 16 KiB-multiple chunk size (tests use
    /// one page to exercise multi-chunk streaming on tiny files).
    pub fn verifyAllBlobsChunked(
        self: *const AmfFile,
        allocator: std.mem.Allocator,
        chunk_bytes: usize,
    ) (ParseError || IoError)!void {
        std.debug.assert(chunk_bytes > 0 and chunk_bytes % page.APPLE_PAGE_SIZE == 0);
        var max_blob: u64 = 0;
        for (self.parsed.entries) |*entry| max_blob = @max(max_blob, entry.blob_bytes);
        if (max_blob == 0) return;
        const scratch_len: usize = @intCast(@min(max_blob, @as(u64, chunk_bytes)));
        const buf = try allocator.alignedAlloc(u8, .fromByteUnits(page.APPLE_PAGE_SIZE), scratch_len);
        defer allocator.free(buf);
        for (self.parsed.entries) |*entry| {
            var hasher = std.hash.Crc32.init();
            var done: u64 = 0;
            while (done < entry.blob_bytes) {
                const take: usize = @intCast(@min(entry.blob_bytes - done, @as(u64, scratch_len)));
                try preadExact(self.fd, buf[0..take], entry.blob_offset + done);
                hasher.update(buf[0..take]);
                done += take;
            }
            if (hasher.final() != entry.blob_crc32) return error.BlobChecksumMismatch;
        }
    }
};

fn preadExact(fd: std.c.fd_t, dst: []u8, offset: u64) IoError!void {
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

fn validateHeader(hdr: *const Header, file_size: u64) ParseError!void {
    if (!std.mem.eql(u8, &hdr.magic, &MAGIC)) return error.InvalidMagic;
    if (hdr.version != FORMAT_VERSION) return error.UnsupportedVersion;
    if (hdr.page_size != page.APPLE_PAGE_SIZE) return error.UnsupportedPageSize;
    if (hdr.checksum_algo != CHECKSUM_CRC32_IEEE) return error.UnsupportedChecksum;
    if (computeHeaderCrc32(hdr) != hdr.header_crc32) return error.HeaderChecksumMismatch;

    if (hdr.directory_offset != page.APPLE_PAGE_SIZE) return error.MisalignedDirectory;
    if (hdr.directory_bytes % page.APPLE_PAGE_SIZE != 0) return error.MisalignedDirectory;
    const raw_dir = @as(u64, hdr.tensor_count) * @sizeOf(TensorEntry);
    const expected_dir_bytes: u64 = if (raw_dir == 0)
        0
    else
        ((raw_dir + page.APPLE_PAGE_SIZE - 1) / page.APPLE_PAGE_SIZE) * page.APPLE_PAGE_SIZE;
    if (hdr.directory_bytes != expected_dir_bytes) return error.MisalignedDirectory;
    const dir_end = std.math.add(u64, hdr.directory_offset, hdr.directory_bytes) catch return error.DirectoryOutOfBounds;
    if (dir_end > file_size) return error.DirectoryOutOfBounds;
}

const Range = struct {
    start: u64,
    end: u64,

    fn lessThan(_: void, a: Range, b: Range) bool {
        return a.start < b.start;
    }
};

/// Validates every directory record and returns the O(1) name index (caller
/// owns it). Duplicate names are detected through the same hash map, so the
/// whole pass is O(N log N) (dominated by the overlap sort) instead of O(N^2).
fn validateEntries(
    allocator: std.mem.Allocator,
    hdr: *const Header,
    entries: []const TensorEntry,
    file_size: u64,
) ParseError!NameIndex {
    const data_start = hdr.directory_offset + hdr.directory_bytes;
    const ranges = try allocator.alloc(Range, entries.len);
    defer allocator.free(ranges);

    var index: NameIndex = .empty;
    errdefer index.deinit(allocator);
    try index.ensureTotalCapacity(allocator, hdr.tensor_count);

    for (entries, 0..) |*entry, i| {
        const name = entry.nameSlice();
        if (name.len == 0) return error.EmptyTensorName;
        const gop = index.getOrPutAssumeCapacity(name);
        if (gop.found_existing) return error.DuplicateTensorName;
        gop.value_ptr.* = @intCast(i);
        const kind = try entry.tensorKind();
        const dtype = try entry.dataType();
        const layout = try entry.tensorLayout();
        const param_kind = try entry.quantParamKind();
        if (entry.rank == 0 or entry.rank > 4) return error.InvalidRank;

        if (kind == .sparse_table) {
            // C10 NVMe PLE / Engram row tables must be 2D `[num_rows, embed_dim, 0, 0]`
            // in row-major layout so individual rows stream cleanly via 16 KiB F_NOCACHE reads.
            if (entry.rank != 2) return error.InvalidRank;
            if (layout != .row_major) return error.InvalidTensorLayout;
            if (entry.shape[2] != 0 or entry.shape[3] != 0) return error.InvalidShape;
        }

        if (entry.blob_offset < data_start or
            entry.blob_offset % page.APPLE_PAGE_SIZE != 0 or
            entry.blob_bytes == 0 or
            entry.blob_bytes % page.APPLE_PAGE_SIZE != 0)
        {
            return error.MisalignedBlob;
        }
        const blob_end = std.math.add(u64, entry.blob_offset, entry.blob_bytes) catch return error.BlobOutOfBounds;
        if (blob_end > file_size) return error.BlobOutOfBounds;

        switch (layout) {
            .row_major => {
                if (param_kind != .affine_scale_bias) return error.InvalidQuantParamKind;
                switch (dtype) {
                    .f16, .bf16, .f32, .u32 => {
                        if (entry.quant_bits != 0 or entry.group_size != 0) return error.InvalidQuantization;
                        if (entry.params_offset != 0 or entry.params_bytes != 0) return error.MisalignedParams;
                        var elems: u64 = 1;
                        for (0..entry.rank) |d| {
                            if (entry.shape[d] == 0) return error.InvalidShape;
                            elems = std.math.mul(u64, elems, entry.shape[d]) catch return error.InvalidShape;
                        }
                        const elem_size: u64 = dtype.unquantizedElemSize() orelse unreachable;
                        const expected_bytes = std.math.mul(u64, elems, elem_size) catch return error.InvalidShape;
                        if (entry.data_bytes != expected_bytes or entry.data_bytes > entry.blob_bytes) {
                            return error.InvalidShape;
                        }
                    },
                    .q4_affine, .q3_affine, .q2_affine => {
                        const expected_bits: u8 = switch (dtype) {
                            .q4_affine => 4,
                            .q3_affine => 3,
                            .q2_affine => 2,
                            else => unreachable,
                        };
                        if (entry.rank != 2) return error.InvalidRank;
                        if (entry.quant_bits != expected_bits or entry.group_size != 64) {
                            return error.InvalidQuantization;
                        }
                        const n: u64 = entry.shape[0];
                        const k: u64 = entry.shape[1];
                        if (n == 0 or k == 0 or k % entry.group_size != 0) return error.InvalidShape;
                        const expected_data = n * ((k * expected_bits) / 8);
                        const expected_params = n * (k / entry.group_size) * @sizeOf(quant.ScaleBias);
                        if (entry.data_bytes != expected_data or entry.params_bytes != expected_params) {
                            return error.InvalidShape;
                        }
                        if (entry.params_offset < entry.data_bytes or entry.params_offset % 64 != 0) {
                            return error.MisalignedParams;
                        }
                        const end_in_blob = std.math.add(u64, entry.params_offset, entry.params_bytes) catch return error.BlobOutOfBounds;
                        if (end_in_blob > entry.blob_bytes) return error.BlobOutOfBounds;
                    },
                }
            },
            .sme2_panel64 => {
                if (entry.rank != 2) return error.InvalidRank;
                if (entry.params_offset != 0 or entry.params_bytes != 0) return error.MisalignedParams;
                switch (dtype) {
                    .f16, .bf16 => {
                        if (param_kind != .affine_scale_bias) return error.InvalidQuantParamKind;
                        if (entry.quant_bits != 0 or entry.group_size != 0) return error.InvalidQuantization;
                    },
                    .q4_affine => {
                        if (entry.quant_bits != 4 or entry.group_size != SME2_GROUP_K) {
                            return error.InvalidQuantization;
                        }
                    },
                    .q3_affine => {
                        if (entry.quant_bits != 3 or entry.group_size != SME2_GROUP_K) {
                            return error.InvalidQuantization;
                        }
                    },
                    .q2_affine => {
                        if (entry.quant_bits != 2 or entry.group_size != SME2_GROUP_K) {
                            return error.InvalidQuantization;
                        }
                    },
                    .f32, .u32 => return error.InvalidTensorLayout,
                }
                const expected_data = try sme2Panel64BlobBytes(dtype, entry.shape[0], entry.shape[1], entry.group_size);
                if (entry.data_bytes != expected_data) return error.InvalidShape;
                if (entry.data_bytes > entry.blob_bytes) return error.BlobOutOfBounds;
            },
        }
        ranges[i] = .{ .start = entry.blob_offset, .end = blob_end };
    }

    std.mem.sort(Range, ranges, {}, Range.lessThan);
    for (1..ranges.len) |i| {
        if (ranges[i].start < ranges[i - 1].end) return error.OverlappingBlobs;
    }
    return index;
}

fn buildTestAmf(allocator: std.mem.Allocator, entry: TensorEntry, blob_data: []const u8) ![]u8 {
    const dir_offset: usize = page.APPLE_PAGE_SIZE;
    const dir_bytes: usize = page.APPLE_PAGE_SIZE;
    const blob_offset: usize = dir_offset + dir_bytes;
    const blob_bytes: usize = try page.alignUp(blob_data.len);
    const total = blob_offset + blob_bytes;

    const image = try allocator.alloc(u8, total);
    errdefer allocator.free(image);
    @memset(image, 0);

    @memcpy(image[blob_offset..][0..blob_data.len], blob_data);
    var e = entry;
    e.blob_offset = blob_offset;
    e.blob_bytes = blob_bytes;
    e.blob_crc32 = std.hash.Crc32.hash(image[blob_offset .. blob_offset + blob_bytes]);

    @memcpy(image[dir_offset..][0..@sizeOf(TensorEntry)], std.mem.asBytes(&e));
    const dir_crc = std.hash.Crc32.hash(image[dir_offset..][0..@sizeOf(TensorEntry)]);

    var hdr: Header = .{
        .magic = MAGIC,
        .version = FORMAT_VERSION,
        .flags = 0,
        .page_size = page.APPLE_PAGE_SIZE,
        .checksum_algo = CHECKSUM_CRC32_IEEE,
        .tensor_count = 1,
        .reserved0 = 0,
        .directory_offset = dir_offset,
        .directory_bytes = dir_bytes,
        .directory_crc32 = dir_crc,
        .header_crc32 = 0,
        .model_id = @splat(0),
        .reserved1 = 0,
    };
    const model = "test-model";
    @memcpy(hdr.model_id[0..model.len], model);
    hdr.header_crc32 = computeHeaderCrc32(&hdr);
    @memcpy(image[0..@sizeOf(Header)], std.mem.asBytes(&hdr));
    return image;
}

test "Header and TensorEntry have exact 128-byte zero-padding wire layouts" {
    try std.testing.expectEqual(@as(usize, 128), @sizeOf(Header));
    try std.testing.expectEqual(@as(usize, 128), @sizeOf(TensorEntry));
}

test "parseAndValidateBytes accepts a well-formed quantized .amf image" {
    const gpa = std.testing.allocator;
    const Q4 = quant.QuantOps(.{ .bits = .q4, .group_size = 64 });
    const n: u32 = 4;
    const k: u32 = 64;
    const data_bytes = try Q4.packedWeightBytes(n, k);
    const params_offset = align64(data_bytes);
    const params_bytes = try Q4.scaleBiasBytes(n, k);

    const blob = try gpa.alloc(u8, params_offset + params_bytes);
    defer gpa.free(blob);
    @memset(blob, 0x3C);

    var entry = TensorEntry.zeroed("layers.0.mlp.experts.2.gate_proj", .cold_expert, .q4_affine);
    entry.quant_bits = 4;
    entry.rank = 2;
    entry.group_size = 64;
    entry.layer = 0;
    entry.expert = 2;
    entry.shape = .{ n, k, 0, 0 };
    entry.data_bytes = data_bytes;
    entry.params_offset = params_offset;
    entry.params_bytes = params_bytes;

    const image = try buildTestAmf(gpa, entry, blob);
    defer gpa.free(image);

    var parsed = try Parsed.parseBytes(gpa, image);
    defer parsed.deinit(gpa);

    try std.testing.expectEqualStrings("test-model", parsed.modelId());
    try std.testing.expectEqual(@as(usize, 1), parsed.entries.len);
    const found = parsed.find("layers.0.mlp.experts.2.gate_proj") orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(u64, 2 * page.APPLE_PAGE_SIZE), found.blob_offset);
    try std.testing.expectEqual(@as(u64, page.APPLE_PAGE_SIZE), found.blob_bytes);

    // Cold expert entries map straight into ExpertSource.local_nvme_ssd.
    const nvme_src = try found.asNvmeSource(0);
    try std.testing.expectError(error.NotResident, nvme_src.residentBytes());

    // Loading a blob into a WiredArena validates CRC-32 and aligns to 16 KiB.
    const cap = 4 * page.APPLE_PAGE_SIZE;
    const budget = budget_mod.Budget.compute(.{
        .memory_bytes = 16 * page.APPLE_PAGE_SIZE,
        .user_wire_limit = cap,
        .global_user_wire_limit = cap,
        .gpu_working_set_bytes = cap,
        .configured_cap = null,
        .headroom_bytes = page.APPLE_PAGE_SIZE,
    });
    var arena = try arena_mod.WiredArena.init(cap, budget);
    defer arena.deinit();

    const loaded = try parsed.loadBlobFromBytes(image, found, &arena);
    try std.testing.expectEqual(page.APPLE_PAGE_SIZE, loaded.len);
    try std.testing.expectEqualSlices(u8, blob, loaded[0..blob.len]);
}

test "validator rejects bad magic, corrupted CRCs and misaligned blobs" {
    const gpa = std.testing.allocator;
    const payload: [256]u8 = @splat(0xAB);
    var entry = TensorEntry.zeroed("norm.weight", .norm, .f16);
    entry.rank = 1;
    entry.shape = .{ 128, 0, 0, 0 };
    entry.data_bytes = 256;

    const good = try buildTestAmf(gpa, entry, &payload);
    defer gpa.free(good);

    // 1. Bad magic
    const bad_magic = try gpa.dupe(u8, good);
    defer gpa.free(bad_magic);
    bad_magic[0] = 'X';
    try std.testing.expectError(error.InvalidMagic, Parsed.parseBytes(gpa, bad_magic));

    // 2. Corrupted header CRC
    const bad_hcrc = try gpa.dupe(u8, good);
    defer gpa.free(bad_hcrc);
    bad_hcrc[56] ^= 0xFF;
    try std.testing.expectError(error.HeaderChecksumMismatch, Parsed.parseBytes(gpa, bad_hcrc));

    // 3. Corrupted directory CRC
    const bad_dcrc = try gpa.dupe(u8, good);
    defer gpa.free(bad_dcrc);
    bad_dcrc[page.APPLE_PAGE_SIZE] ^= 0xFF;
    try std.testing.expectError(error.DirectoryChecksumMismatch, Parsed.parseBytes(gpa, bad_dcrc));

    // 4. Corrupted blob CRC (verified on full validate / load)
    const bad_bcrc = try gpa.dupe(u8, good);
    defer gpa.free(bad_bcrc);
    bad_bcrc[2 * page.APPLE_PAGE_SIZE] ^= 0xFF;
    var parsed = try Parsed.parseBytes(gpa, bad_bcrc);
    defer parsed.deinit(gpa);
    try std.testing.expectError(error.BlobChecksumMismatch, parsed.verifyAllBlobsFromBytes(bad_bcrc));

    // 5. Misaligned blob offset (recompute dir + header CRCs so alignment check trips)
    const bad_align = try gpa.dupe(u8, good);
    defer gpa.free(bad_align);
    const entry_ptr: *TensorEntry = @ptrCast(@alignCast(bad_align.ptr + page.APPLE_PAGE_SIZE));
    entry_ptr.blob_offset += 64;
    const hdr_ptr: *Header = @ptrCast(@alignCast(bad_align.ptr));
    hdr_ptr.directory_crc32 = std.hash.Crc32.hash(bad_align[page.APPLE_PAGE_SIZE..][0..@sizeOf(TensorEntry)]);
    hdr_ptr.header_crc32 = computeHeaderCrc32(hdr_ptr);
    try std.testing.expectError(error.MisalignedBlob, Parsed.parseBytes(gpa, bad_align));
}

test "validator rejects overlapping blobs" {
    const gpa = std.testing.allocator;
    const dir_offset: usize = page.APPLE_PAGE_SIZE;
    const dir_bytes: usize = page.APPLE_PAGE_SIZE;
    const blob_offset: usize = dir_offset + dir_bytes;
    const total = blob_offset + page.APPLE_PAGE_SIZE;

    const image = try gpa.alloc(u8, total);
    defer gpa.free(image);
    @memset(image, 0);

    var e0 = TensorEntry.zeroed("a", .norm, .f16);
    e0.rank = 1;
    e0.shape = .{ 64, 0, 0, 0 };
    e0.blob_offset = blob_offset;
    e0.blob_bytes = page.APPLE_PAGE_SIZE;
    e0.data_bytes = 128;

    var e1 = TensorEntry.zeroed("b", .norm, .f16);
    e1.rank = 1;
    e1.shape = .{ 64, 0, 0, 0 };
    e1.blob_offset = blob_offset; // overlaps e0!
    e1.blob_bytes = page.APPLE_PAGE_SIZE;
    e1.data_bytes = 128;

    @memcpy(image[dir_offset..][0..@sizeOf(TensorEntry)], std.mem.asBytes(&e0));
    @memcpy(image[dir_offset + @sizeOf(TensorEntry) ..][0..@sizeOf(TensorEntry)], std.mem.asBytes(&e1));

    var hdr: Header = .{
        .magic = MAGIC,
        .version = FORMAT_VERSION,
        .flags = 0,
        .page_size = page.APPLE_PAGE_SIZE,
        .checksum_algo = CHECKSUM_CRC32_IEEE,
        .tensor_count = 2,
        .reserved0 = 0,
        .directory_offset = dir_offset,
        .directory_bytes = dir_bytes,
        .directory_crc32 = std.hash.Crc32.hash(image[dir_offset..][0 .. 2 * @sizeOf(TensorEntry)]),
        .header_crc32 = 0,
        .model_id = @splat(0),
        .reserved1 = 0,
    };
    hdr.header_crc32 = computeHeaderCrc32(&hdr);
    @memcpy(image[0..@sizeOf(Header)], std.mem.asBytes(&hdr));

    try std.testing.expectError(error.OverlappingBlobs, Parsed.parseBytes(gpa, image));
}

test "AmfFile opens with F_NOCACHE, verifies CRCs and loads into WiredArena" {
    const gpa = std.testing.allocator;
    const payload: [256]u8 = @splat(0x5E);
    var entry = TensorEntry.zeroed("layers.0.input_layernorm.weight", .norm, .f16);
    entry.rank = 1;
    entry.layer = 0;
    entry.shape = .{ 128, 0, 0, 0 };
    entry.data_bytes = 256;

    const image = try buildTestAmf(gpa, entry, &payload);
    defer gpa.free(image);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "sample.amf", .data = image });

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const raw_path = try std.fmt.bufPrint(path_buf[0 .. path_buf.len - 1], ".zig-cache/tmp/{s}/sample.amf", .{tmp.sub_path});
    path_buf[raw_path.len] = 0;
    const rel_path: [:0]const u8 = path_buf[0..raw_path.len :0];

    var amf_file = try AmfFile.open(gpa, rel_path);
    defer amf_file.close(gpa);

    try amf_file.verifyAllBlobs(gpa);
    const found = amf_file.parsed.find("layers.0.input_layernorm.weight") orelse return error.TestUnexpectedResult;

    const cap = 4 * page.APPLE_PAGE_SIZE;
    const budget = budget_mod.Budget.compute(.{
        .memory_bytes = 16 * page.APPLE_PAGE_SIZE,
        .user_wire_limit = cap,
        .global_user_wire_limit = cap,
        .gpu_working_set_bytes = cap,
        .configured_cap = null,
        .headroom_bytes = page.APPLE_PAGE_SIZE,
    });
    var arena = try arena_mod.WiredArena.init(cap, budget);
    defer arena.deinit();

    const loaded = try amf_file.loadBlob(found, &arena);
    try std.testing.expectEqual(page.APPLE_PAGE_SIZE, loaded.len);
    try std.testing.expectEqualSlices(u8, &payload, loaded[0..payload.len]);
}

/// Builds an `.amf` image with one 16 KiB blob per entry (each blob filled
/// with a per-entry byte pattern). Entry blob offsets, sizes and CRCs plus the
/// directory and header CRCs are computed here.
fn buildTestAmfMulti(allocator: std.mem.Allocator, entries_in: []const TensorEntry) ![]u8 {
    const count = entries_in.len;
    const dir_offset: usize = page.APPLE_PAGE_SIZE;
    const dir_raw = count * @sizeOf(TensorEntry);
    const dir_bytes: usize = try page.alignUp(dir_raw);
    const data_start = dir_offset + dir_bytes;
    const total = data_start + count * page.APPLE_PAGE_SIZE;

    const image = try allocator.alloc(u8, total);
    errdefer allocator.free(image);
    @memset(image, 0);

    for (entries_in, 0..) |entry, i| {
        const blob_offset = data_start + i * page.APPLE_PAGE_SIZE;
        const blob = image[blob_offset..][0..page.APPLE_PAGE_SIZE];
        @memset(blob[0..@intCast(entry.data_bytes)], @truncate(i *% 31 +% 7));
        var e = entry;
        e.blob_offset = blob_offset;
        e.blob_bytes = page.APPLE_PAGE_SIZE;
        e.blob_crc32 = std.hash.Crc32.hash(blob);
        @memcpy(image[dir_offset + i * @sizeOf(TensorEntry) ..][0..@sizeOf(TensorEntry)], std.mem.asBytes(&e));
    }

    var hdr: Header = .{
        .magic = MAGIC,
        .version = FORMAT_VERSION,
        .flags = 0,
        .page_size = page.APPLE_PAGE_SIZE,
        .checksum_algo = CHECKSUM_CRC32_IEEE,
        .tensor_count = @intCast(count),
        .reserved0 = 0,
        .directory_offset = dir_offset,
        .directory_bytes = dir_bytes,
        .directory_crc32 = std.hash.Crc32.hash(image[dir_offset..][0..dir_raw]),
        .header_crc32 = 0,
        .model_id = @splat(0),
        .reserved1 = 0,
    };
    const model = "multi-model";
    @memcpy(hdr.model_id[0..model.len], model);
    hdr.header_crc32 = computeHeaderCrc32(&hdr);
    @memcpy(image[0..@sizeOf(Header)], std.mem.asBytes(&hdr));
    return image;
}

fn bf16TestEntry() TensorEntry {
    var entry = TensorEntry.zeroed("layers.0.self_attn.q_proj.weight", .dense, .bf16);
    entry.rank = 2;
    entry.layer = 0;
    entry.shape = .{ 4, 64, 0, 0 };
    entry.data_bytes = 4 * 64 * 2;
    return entry;
}

test "DType.bf16 is wire value 4 and parses as an unquantized 2-byte tensor" {
    const gpa = std.testing.allocator;
    try std.testing.expectEqual(@as(u8, 4), @backingInt(DType.bf16));
    try std.testing.expectEqual(@as(?u64, 2), DType.bf16.unquantizedElemSize());
    try std.testing.expectEqual(@as(?u64, 2), DType.f16.unquantizedElemSize());
    try std.testing.expectEqual(@as(?u64, 4), DType.f32.unquantizedElemSize());
    try std.testing.expectEqual(@as(?u64, null), DType.q3_affine.unquantizedElemSize());

    var payload: [512]u8 = undefined;
    for (&payload, 0..) |*b, i| b.* = @truncate(i);
    const image = try buildTestAmf(gpa, bf16TestEntry(), &payload);
    defer gpa.free(image);

    var parsed = try Parsed.parseBytes(gpa, image);
    defer parsed.deinit(gpa);
    const found = parsed.find("layers.0.self_attn.q_proj.weight") orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(DType.bf16, try found.dataType());
    try std.testing.expectEqual(@as(u8, 0), found.quant_bits);
    try std.testing.expectEqual(@as(u64, 512), found.data_bytes);
    try parsed.verifyAllBlobsFromBytes(image);
    try std.testing.expect(parsed.find("layers.0.self_attn.q_proj") == null);
}

test "validator rejects invalid bf16 metadata" {
    const gpa = std.testing.allocator;
    const payload: [2048]u8 = @splat(0x3F);

    const Case = struct { err: ParseError, entry: TensorEntry };
    var cases: [9]Case = undefined;
    var n: usize = 0;

    var e = bf16TestEntry();
    e.quant_bits = 16; // ADR-0016 contract: unquantized bf16 carries quant_bits = 0.
    cases[n] = .{ .err = error.InvalidQuantization, .entry = e };
    n += 1;

    e = bf16TestEntry();
    e.group_size = 64;
    cases[n] = .{ .err = error.InvalidQuantization, .entry = e };
    n += 1;

    e = bf16TestEntry();
    e.params_offset = 512;
    cases[n] = .{ .err = error.MisalignedParams, .entry = e };
    n += 1;

    e = bf16TestEntry();
    e.params_bytes = 64;
    cases[n] = .{ .err = error.MisalignedParams, .entry = e };
    n += 1;

    e = bf16TestEntry();
    e.data_bytes = 4 * 64 * 4; // f32-sized payload
    cases[n] = .{ .err = error.InvalidShape, .entry = e };
    n += 1;

    e = bf16TestEntry();
    e.shape = .{ 4, 0, 0, 0 };
    cases[n] = .{ .err = error.InvalidShape, .entry = e };
    n += 1;

    e = bf16TestEntry();
    e.rank = 0;
    cases[n] = .{ .err = error.InvalidRank, .entry = e };
    n += 1;

    e = bf16TestEntry();
    e.rank = 5;
    cases[n] = .{ .err = error.InvalidRank, .entry = e };
    n += 1;

    e = bf16TestEntry();
    e.dtype = 7; // unknown dtype past q2_affine
    cases[n] = .{ .err = error.InvalidDType, .entry = e };
    n += 1;

    for (cases[0..n]) |c| {
        const image = try buildTestAmf(gpa, c.entry, &payload);
        defer gpa.free(image);
        try std.testing.expectError(c.err, Parsed.parseBytes(gpa, image));
    }

    // data_bytes larger than the blob is rejected too.
    e = bf16TestEntry();
    e.rank = 1;
    e.shape = .{ page.APPLE_PAGE_SIZE, 0, 0, 0 }; // 32 KiB of bf16 > one 16 KiB blob
    e.data_bytes = 2 * page.APPLE_PAGE_SIZE;
    const image = try buildTestAmf(gpa, e, &payload);
    defer gpa.free(image);
    try std.testing.expectError(error.InvalidShape, Parsed.parseBytes(gpa, image));
}

test "hash-indexed lookup and duplicate detection over thousands of entries" {
    const gpa = std.testing.allocator;
    const count: usize = 4096;
    const entries = try gpa.alloc(TensorEntry, count);
    defer gpa.free(entries);

    var name_buf: [48]u8 = undefined;
    for (entries, 0..) |*entry, i| {
        const name = try std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.experts.{d}.gate_proj.weight", .{ i / 128, i % 128 });
        entry.* = TensorEntry.zeroed(name, .hot_expert, if (i % 2 == 0) .bf16 else .f16);
        entry.rank = 1;
        entry.layer = @intCast(i / 128);
        entry.expert = @intCast(i % 128);
        entry.shape = .{ 64, 0, 0, 0 };
        entry.data_bytes = 128;
    }

    const image = try buildTestAmfMulti(gpa, entries);
    defer gpa.free(image);

    var parsed = try Parsed.parseBytes(gpa, image);
    defer parsed.deinit(gpa);
    try std.testing.expectEqual(count, parsed.entries.len);
    try std.testing.expectEqual(@as(u32, count), parsed.index.count());
    try parsed.verifyAllBlobsFromBytes(image);

    for (0..count) |i| {
        const name = try std.fmt.bufPrint(&name_buf, "layers.{d}.mlp.experts.{d}.gate_proj.weight", .{ i / 128, i % 128 });
        const found = parsed.find(name) orelse return error.TestUnexpectedResult;
        try std.testing.expectEqual(&parsed.entries[i], found);
        try std.testing.expectEqual(@as(u16, @intCast(i % 128)), found.expert);
    }
    try std.testing.expect(parsed.find("layers.99.mlp.experts.0.gate_proj.weight") == null);
    try std.testing.expect(parsed.find("") == null);

    // The same hash map rejects a duplicate placed at the very end.
    entries[count - 1] = entries[0];
    const dup_image = try buildTestAmfMulti(gpa, entries);
    defer gpa.free(dup_image);
    try std.testing.expectError(error.DuplicateTensorName, Parsed.parseBytes(gpa, dup_image));
}

test "AmfFile.verifyAllBlobsChunked streams multi-chunk blobs and detects corruption" {
    const gpa = std.testing.allocator;
    const payload_len = 3 * page.APPLE_PAGE_SIZE - 100;
    const payload = try gpa.alloc(u8, payload_len);
    defer gpa.free(payload);
    for (payload, 0..) |*b, i| b.* = @truncate(i *% 13);

    var entry = TensorEntry.zeroed("embed_tokens.weight", .embedding, .bf16);
    entry.rank = 2;
    entry.shape = .{ 6, (payload_len / 2) / 6, 0, 0 };
    entry.data_bytes = @as(u64, entry.shape[0]) * entry.shape[1] * 2;

    const image = try buildTestAmf(gpa, entry, payload);
    defer gpa.free(image);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "chunked.amf", .data = image });
    // Corrupt the last page of the 3-page blob in a second file.
    const bad = try gpa.dupe(u8, image);
    defer gpa.free(bad);
    bad[bad.len - 1] ^= 0x5A;
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "chunked_bad.amf", .data = bad });

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const raw_path = try std.fmt.bufPrint(path_buf[0 .. path_buf.len - 1], ".zig-cache/tmp/{s}/chunked.amf", .{tmp.sub_path});
    path_buf[raw_path.len] = 0;
    var good_file = try AmfFile.open(gpa, path_buf[0..raw_path.len :0]);
    defer good_file.close(gpa);
    try good_file.verifyAllBlobsChunked(gpa, page.APPLE_PAGE_SIZE);
    try good_file.verifyAllBlobs(gpa);
    const found = good_file.parsed.find("embed_tokens.weight") orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(DType.bf16, try found.dataType());

    var bad_buf: [std.fs.max_path_bytes]u8 = undefined;
    const bad_path = try std.fmt.bufPrint(bad_buf[0 .. bad_buf.len - 1], ".zig-cache/tmp/{s}/chunked_bad.amf", .{tmp.sub_path});
    bad_buf[bad_path.len] = 0;
    var bad_file = try AmfFile.open(gpa, bad_buf[0..bad_path.len :0]);
    defer bad_file.close(gpa);
    try std.testing.expectError(error.BlobChecksumMismatch, bad_file.verifyAllBlobsChunked(gpa, page.APPLE_PAGE_SIZE));
    try std.testing.expectError(error.BlobChecksumMismatch, bad_file.verifyAllBlobs(gpa));
}

fn u32TestEntry() TensorEntry {
    var entry = TensorEntry.zeroed("d2t", .dense, .u32);
    entry.rank = 1;
    entry.shape = .{ 64, 0, 0, 0 };
    entry.data_bytes = 64 * 4;
    return entry;
}

test "DType.u32 is wire value 5 and parses as an unquantized 4-byte tensor" {
    const gpa = std.testing.allocator;
    try std.testing.expectEqual(@as(u8, 5), @backingInt(DType.u32));
    try std.testing.expectEqual(@as(?u64, 4), DType.u32.unquantizedElemSize());

    var ids: [64]u32 = undefined;
    for (&ids, 0..) |*slot, i| slot.* = @intCast(i * 3 + 10);
    const image = try buildTestAmf(gpa, u32TestEntry(), std.mem.sliceAsBytes(&ids));
    defer gpa.free(image);

    var parsed = try Parsed.parseBytes(gpa, image);
    defer parsed.deinit(gpa);
    const found = parsed.find("d2t") orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(DType.u32, try found.dataType());
    try std.testing.expectEqual(TensorKind.dense, try found.tensorKind());
    try std.testing.expectEqual(@as(u8, 0), found.quant_bits);
    try std.testing.expectEqual(@as(u16, 0), found.group_size);
    try std.testing.expectEqual(@as(u64, 64 * 4), found.data_bytes);
    try parsed.verifyAllBlobsFromBytes(image);

    const cap = 4 * page.APPLE_PAGE_SIZE;
    const budget = budget_mod.Budget.compute(.{
        .memory_bytes = 16 * page.APPLE_PAGE_SIZE,
        .user_wire_limit = cap,
        .global_user_wire_limit = cap,
        .gpu_working_set_bytes = cap,
        .configured_cap = null,
        .headroom_bytes = page.APPLE_PAGE_SIZE,
    });
    var arena = try arena_mod.WiredArena.init(cap, budget);
    defer arena.deinit();

    const loaded = try parsed.loadBlobFromBytes(image, found, &arena);
    const got_ids: []const u32 = @alignCast(std.mem.bytesAsSlice(u32, loaded[0 .. 64 * 4]));
    try std.testing.expectEqualSlices(u32, &ids, got_ids);
}

test "validator rejects invalid u32 metadata" {
    const gpa = std.testing.allocator;
    const payload: [256]u8 = @splat(0);

    const Case = struct { err: ParseError, entry: TensorEntry };
    var cases: [7]Case = undefined;
    var n: usize = 0;

    var e = u32TestEntry();
    e.quant_bits = 32;
    cases[n] = .{ .err = error.InvalidQuantization, .entry = e };
    n += 1;

    e = u32TestEntry();
    e.group_size = 64;
    cases[n] = .{ .err = error.InvalidQuantization, .entry = e };
    n += 1;

    e = u32TestEntry();
    e.params_offset = 256;
    cases[n] = .{ .err = error.MisalignedParams, .entry = e };
    n += 1;

    e = u32TestEntry();
    e.params_bytes = 64;
    cases[n] = .{ .err = error.MisalignedParams, .entry = e };
    n += 1;

    e = u32TestEntry();
    e.data_bytes = 64 * 2; // wrong element size (2 instead of 4)
    cases[n] = .{ .err = error.InvalidShape, .entry = e };
    n += 1;

    e = u32TestEntry();
    e.rank = 0;
    cases[n] = .{ .err = error.InvalidRank, .entry = e };
    n += 1;

    e = u32TestEntry();
    e.rank = 5;
    cases[n] = .{ .err = error.InvalidRank, .entry = e };
    n += 1;

    for (cases[0..n]) |c| {
        const image = try buildTestAmf(gpa, c.entry, &payload);
        defer gpa.free(image);
        try std.testing.expectError(c.err, Parsed.parseBytes(gpa, image));
    }
}

test "TensorEntry reserved0 encodes TensorLayout and QuantParamKind with 100% v0 compatibility" {
    try std.testing.expectEqual(@as(usize, 58), @offsetOf(TensorEntry, "reserved0"));

    // Existing v0 zero-initialized entries decode to row_major + affine_scale_bias.
    var entry = TensorEntry.zeroed("layers.0.mlp.gate.weight", .router, .bf16);
    try std.testing.expectEqual(@as(u16, 0), entry.reserved0);
    try std.testing.expectEqual(TensorLayout.row_major, try entry.tensorLayout());
    try std.testing.expectEqual(QuantParamKind.affine_scale_bias, try entry.quantParamKind());

    entry.setLayoutAndParamKind(.sme2_panel64, .affine_scale_bias);
    try std.testing.expectEqual(@as(u16, 0x0001), entry.reserved0);
    try std.testing.expectEqual(TensorLayout.sme2_panel64, try entry.tensorLayout());
    try std.testing.expectEqual(QuantParamKind.affine_scale_bias, try entry.quantParamKind());

    entry.setLayoutAndParamKind(.sme2_panel64, .lut16_codebook);
    try std.testing.expectEqual(@as(u16, 0x0101), entry.reserved0);
    try std.testing.expectEqual(TensorLayout.sme2_panel64, try entry.tensorLayout());
    try std.testing.expectEqual(QuantParamKind.lut16_codebook, try entry.quantParamKind());

    entry.reserved0 = 0x0002;
    try std.testing.expectError(error.InvalidTensorLayout, entry.tensorLayout());

    entry.reserved0 = 0x0201;
    try std.testing.expectEqual(TensorLayout.sme2_panel64, try entry.tensorLayout());
    try std.testing.expectError(error.InvalidQuantParamKind, entry.quantParamKind());
}

test "sme2Panel64BlobBytes computes exact panel sizes including N%16!=0 padding" {
    // Unquantized f16 / bf16: 2048 B per (panel, group), group_size == 0.
    try std.testing.expectEqual(@as(u64, 2048), try sme2Panel64BlobBytes(.bf16, 16, 64, 0));
    try std.testing.expectEqual(@as(u64, 2048), try sme2Panel64BlobBytes(.f16, 1, 64, 0));
    try std.testing.expectEqual(@as(u64, 2 * 2 * 2048), try sme2Panel64BlobBytes(.bf16, 17, 128, 0));
    try std.testing.expectEqual(@as(u64, 3 * 2 * 2048), try sme2Panel64BlobBytes(.f16, 33, 128, 0));

    // Quantized q4_affine: 640 B per (panel, group), group_size == 64.
    try std.testing.expectEqual(@as(u64, 640), try sme2Panel64BlobBytes(.q4_affine, 16, 64, 64));
    try std.testing.expectEqual(@as(u64, 2 * 3 * 640), try sme2Panel64BlobBytes(.q4_affine, 17, 192, 64));

    // Quantized q3_affine: 512 B per (panel, group), group_size == 64.
    try std.testing.expectEqual(@as(u64, 512), try sme2Panel64BlobBytes(.q3_affine, 16, 64, 64));
    try std.testing.expectEqual(@as(u64, 3 * 2 * 512), try sme2Panel64BlobBytes(.q3_affine, 33, 128, 64));

    // Reject invalid shapes, group sizes, and unsupported dtypes.
    try std.testing.expectError(error.InvalidShape, sme2Panel64BlobBytes(.bf16, 0, 64, 0));
    try std.testing.expectError(error.InvalidShape, sme2Panel64BlobBytes(.bf16, 16, 0, 0));
    try std.testing.expectError(error.InvalidShape, sme2Panel64BlobBytes(.bf16, 16, 32, 0));
    try std.testing.expectError(error.InvalidQuantization, sme2Panel64BlobBytes(.bf16, 16, 64, 64));
    try std.testing.expectError(error.InvalidQuantization, sme2Panel64BlobBytes(.q4_affine, 16, 64, 0));
    try std.testing.expectError(error.InvalidQuantization, sme2Panel64BlobBytes(.q3_affine, 16, 64, 32));
    try std.testing.expectError(error.InvalidTensorLayout, sme2Panel64BlobBytes(.f32, 16, 64, 0));
    try std.testing.expectError(error.InvalidTensorLayout, sme2Panel64BlobBytes(.u32, 16, 64, 0));
}

test "validator accepts valid sme2_panel64 blobs and rejects malformed reserved0 or metadata" {
    const gpa = std.testing.allocator;

    // 1. Valid bf16 sme2_panel64 with N = 17 (padded to 32 -> 2 panels), K = 128 (2 groups) => 8192 B.
    const bf16_len = try sme2Panel64BlobBytes(.bf16, 17, 128, 0);
    const bf16_buf = try gpa.alloc(u8, @intCast(bf16_len));
    defer gpa.free(bf16_buf);
    @memset(bf16_buf, 0x2A);

    var bf16_entry = TensorEntry.zeroed("layers.0.mlp.gate.weight", .router, .bf16);
    bf16_entry.rank = 2;
    bf16_entry.layer = 0;
    bf16_entry.shape = .{ 17, 128, 0, 0 };
    bf16_entry.data_bytes = bf16_len;
    bf16_entry.setLayoutAndParamKind(.sme2_panel64, .affine_scale_bias);

    const bf16_img = try buildTestAmf(gpa, bf16_entry, bf16_buf);
    defer gpa.free(bf16_img);
    var parsed_bf16 = try Parsed.parseBytes(gpa, bf16_img);
    defer parsed_bf16.deinit(gpa);
    try parsed_bf16.verifyAllBlobsFromBytes(bf16_img);
    const found_bf16 = parsed_bf16.find("layers.0.mlp.gate.weight") orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(TensorLayout.sme2_panel64, try found_bf16.tensorLayout());
    try std.testing.expectEqual(QuantParamKind.affine_scale_bias, try found_bf16.quantParamKind());

    // 2. Valid q4_affine sme2_panel64 (both affine_scale_bias and lut16_codebook) with N = 33, K = 128.
    const q4_len = try sme2Panel64BlobBytes(.q4_affine, 33, 128, 64);
    const q4_buf = try gpa.alloc(u8, @intCast(q4_len));
    defer gpa.free(q4_buf);
    @memset(q4_buf, 0x51);

    for ([_]QuantParamKind{ .affine_scale_bias, .lut16_codebook }) |pk| {
        var q4_entry = TensorEntry.zeroed("layers.0.self_attn.q_proj", .dense, .q4_affine);
        q4_entry.quant_bits = 4;
        q4_entry.rank = 2;
        q4_entry.group_size = 64;
        q4_entry.layer = 0;
        q4_entry.shape = .{ 33, 128, 0, 0 };
        q4_entry.data_bytes = q4_len;
        q4_entry.params_offset = 0;
        q4_entry.params_bytes = 0;
        q4_entry.setLayoutAndParamKind(.sme2_panel64, pk);

        const q4_img = try buildTestAmf(gpa, q4_entry, q4_buf);
        defer gpa.free(q4_img);
        var parsed_q4 = try Parsed.parseBytes(gpa, q4_img);
        defer parsed_q4.deinit(gpa);
        try parsed_q4.verifyAllBlobsFromBytes(q4_img);
        const found_q4 = parsed_q4.find("layers.0.self_attn.q_proj") orelse return error.TestUnexpectedResult;
        try std.testing.expectEqual(TensorLayout.sme2_panel64, try found_q4.tensorLayout());
        try std.testing.expectEqual(pk, try found_q4.quantParamKind());
    }

    // 3. Valid q3_affine sme2_panel64 with N = 19, K = 64 (2 panels * 1 group * 512 B = 1024 B).
    const q3_len = try sme2Panel64BlobBytes(.q3_affine, 19, 64, 64);
    const q3_buf = try gpa.alloc(u8, @intCast(q3_len));
    defer gpa.free(q3_buf);
    @memset(q3_buf, 0x73);

    var q3_entry = TensorEntry.zeroed("layers.0.mlp.experts.0.gate_proj", .cold_expert, .q3_affine);
    q3_entry.quant_bits = 3;
    q3_entry.rank = 2;
    q3_entry.group_size = 64;
    q3_entry.layer = 0;
    q3_entry.expert = 0;
    q3_entry.shape = .{ 19, 64, 0, 0 };
    q3_entry.data_bytes = q3_len;
    q3_entry.setLayoutAndParamKind(.sme2_panel64, .affine_scale_bias);

    const q3_img = try buildTestAmf(gpa, q3_entry, q3_buf);
    defer gpa.free(q3_img);
    var parsed_q3 = try Parsed.parseBytes(gpa, q3_img);
    defer parsed_q3.deinit(gpa);
    try parsed_q3.verifyAllBlobsFromBytes(q3_img);

    // 4. Reject invalid reserved0 and invalid sme2_panel64 metadata combinations.
    const Case = struct { err: ParseError, entry: TensorEntry };
    var cases: [11]Case = undefined;
    var n_cases: usize = 0;

    var bad = bf16_entry;
    bad.reserved0 = 0x0002; // unknown TensorLayout
    cases[n_cases] = .{ .err = error.InvalidTensorLayout, .entry = bad };
    n_cases += 1;

    bad = bf16_entry;
    bad.reserved0 = 0x0201; // unknown QuantParamKind
    cases[n_cases] = .{ .err = error.InvalidQuantParamKind, .entry = bad };
    n_cases += 1;

    bad = bf16_entry;
    bad.setLayoutAndParamKind(.sme2_panel64, .lut16_codebook); // lut16 invalid on unquantized bf16
    cases[n_cases] = .{ .err = error.InvalidQuantParamKind, .entry = bad };
    n_cases += 1;

    bad = bf16TestEntry();
    bad.setLayoutAndParamKind(.row_major, .lut16_codebook); // lut16 invalid on row_major
    cases[n_cases] = .{ .err = error.InvalidQuantParamKind, .entry = bad };
    n_cases += 1;

    bad = bf16_entry;
    bad.params_offset = 64; // sme2_panel64 requires inline metadata (params_offset == 0)
    cases[n_cases] = .{ .err = error.MisalignedParams, .entry = bad };
    n_cases += 1;

    bad = bf16_entry;
    bad.params_bytes = 64; // sme2_panel64 requires inline metadata (params_bytes == 0)
    cases[n_cases] = .{ .err = error.MisalignedParams, .entry = bad };
    n_cases += 1;

    bad = bf16_entry;
    bad.rank = 1;
    cases[n_cases] = .{ .err = error.InvalidRank, .entry = bad };
    n_cases += 1;

    bad = bf16_entry;
    bad.data_bytes = 17 * 128 * 2; // row-major unpadded byte size instead of panel-padded
    cases[n_cases] = .{ .err = error.InvalidShape, .entry = bad };
    n_cases += 1;

    bad = bf16_entry;
    bad.shape = .{ 17, 96, 0, 0 }; // K not multiple of 64
    cases[n_cases] = .{ .err = error.InvalidShape, .entry = bad };
    n_cases += 1;

    bad = TensorEntry.zeroed("bad.f32.panel", .dense, .f32);
    bad.rank = 2;
    bad.shape = .{ 16, 64, 0, 0 };
    bad.data_bytes = 4096;
    bad.setLayoutAndParamKind(.sme2_panel64, .affine_scale_bias);
    cases[n_cases] = .{ .err = error.InvalidTensorLayout, .entry = bad };
    n_cases += 1;

    bad = q3_entry;
    bad.quant_bits = 4; // mismatched quant_bits for q3_affine
    cases[n_cases] = .{ .err = error.InvalidQuantization, .entry = bad };
    n_cases += 1;

    for (cases[0..n_cases]) |c| {
        const img = try buildTestAmf(gpa, c.entry, bf16_buf);
        defer gpa.free(img);
        try std.testing.expectError(c.err, Parsed.parseBytes(gpa, img));
    }

    // 5. Reject sme2_panel64 whose valid data_bytes exceeds blob_bytes.
    const big_len = try sme2Panel64BlobBytes(.bf16, 16, 1024, 0); // 16 * 2048 = 32768 B > 16384 B blob
    var big_entry = bf16_entry;
    big_entry.shape = .{ 16, 1024, 0, 0 };
    big_entry.data_bytes = big_len;
    const big_img = try buildTestAmf(gpa, big_entry, bf16_buf);
    defer gpa.free(big_img);
    try std.testing.expectError(error.BlobOutOfBounds, Parsed.parseBytes(gpa, big_img));
}

test "DType.q2_affine is wire value 6 and validates both row_major and sme2_panel64 layouts" {
    const gpa = std.testing.allocator;
    try std.testing.expectEqual(@as(u8, 6), @backingInt(DType.q2_affine));
    try std.testing.expectEqual(@as(?u64, null), DType.q2_affine.unquantizedElemSize());
    try std.testing.expectEqual(@as(u64, 384), try sme2Panel64BlobBytes(.q2_affine, 16, 64, 64));
    try std.testing.expectEqual(@as(u64, 2 * 3 * 384), try sme2Panel64BlobBytes(.q2_affine, 17, 192, 64));
    try std.testing.expectError(error.InvalidQuantization, sme2Panel64BlobBytes(.q2_affine, 16, 64, 32));

    // 1. row_major q2_affine (ADR-0017: 16 B codes + 4 B ScaleBias per group-64)
    const Q2 = quant.QuantOps(.{ .bits = .q2, .group_size = 64 });
    const n: u32 = 8;
    const k: u32 = 128;
    const data_bytes = try Q2.packedWeightBytes(n, k);
    const params_offset = align64(data_bytes);
    const params_bytes = try Q2.scaleBiasBytes(n, k);
    try std.testing.expectEqual(@as(usize, 256), data_bytes);
    try std.testing.expectEqual(@as(usize, 64), params_bytes);

    const blob = try gpa.alloc(u8, params_offset + params_bytes);
    defer gpa.free(blob);
    @memset(blob, 0xE4);

    var entry = TensorEntry.zeroed("layers.0.mlp.experts.5.gate_proj", .hot_expert, .q2_affine);
    entry.quant_bits = 2;
    entry.rank = 2;
    entry.group_size = 64;
    entry.layer = 0;
    entry.expert = 5;
    entry.shape = .{ n, k, 0, 0 };
    entry.data_bytes = data_bytes;
    entry.params_offset = params_offset;
    entry.params_bytes = params_bytes;

    const img = try buildTestAmf(gpa, entry, blob);
    defer gpa.free(img);
    var parsed = try Parsed.parseBytes(gpa, img);
    defer parsed.deinit(gpa);
    try parsed.verifyAllBlobsFromBytes(img);
    const found = parsed.find("layers.0.mlp.experts.5.gate_proj") orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(DType.q2_affine, try found.dataType());
    try std.testing.expectEqual(@as(u8, 2), found.quant_bits);
    try std.testing.expectEqual(TensorLayout.row_major, try found.tensorLayout());

    // Reject wrong quant_bits for q2_affine
    var bad_bits = entry;
    bad_bits.quant_bits = 3;
    const bad_bits_img = try buildTestAmf(gpa, bad_bits, blob);
    defer gpa.free(bad_bits_img);
    try std.testing.expectError(error.InvalidQuantization, Parsed.parseBytes(gpa, bad_bits_img));

    // 2. sme2_panel64 q2_affine (384 B per (panel, group))
    const q2_sme_len = try sme2Panel64BlobBytes(.q2_affine, 19, 128, 64);
    try std.testing.expectEqual(@as(u64, 2 * 2 * 384), q2_sme_len);
    const q2_sme_buf = try gpa.alloc(u8, @intCast(q2_sme_len));
    defer gpa.free(q2_sme_buf);
    @memset(q2_sme_buf, 0x1B);

    var sme_entry = TensorEntry.zeroed("layers.0.mlp.experts.5.down_proj", .cold_expert, .q2_affine);
    sme_entry.quant_bits = 2;
    sme_entry.rank = 2;
    sme_entry.group_size = 64;
    sme_entry.layer = 0;
    sme_entry.expert = 5;
    sme_entry.shape = .{ 19, 128, 0, 0 };
    sme_entry.data_bytes = q2_sme_len;
    sme_entry.setLayoutAndParamKind(.sme2_panel64, .affine_scale_bias);

    const sme_img = try buildTestAmf(gpa, sme_entry, q2_sme_buf);
    defer gpa.free(sme_img);
    var parsed_sme = try Parsed.parseBytes(gpa, sme_img);
    defer parsed_sme.deinit(gpa);
    try parsed_sme.verifyAllBlobsFromBytes(sme_img);
    const found_sme = parsed_sme.find("layers.0.mlp.experts.5.down_proj") orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(DType.q2_affine, try found_sme.dataType());
    try std.testing.expectEqual(TensorLayout.sme2_panel64, try found_sme.tensorLayout());
}

test "M6 TensorKind extensions: shared_expert, sparse_table, loop_adapter and wired/cold isolation" {
    const gpa = std.testing.allocator;

    // 1. Exact wire values and wired/cold classification.
    try std.testing.expectEqual(@as(u8, 7), @backingInt(TensorKind.shared_expert));
    try std.testing.expectEqual(@as(u8, 8), @backingInt(TensorKind.sparse_table));
    try std.testing.expectEqual(@as(u8, 9), @backingInt(TensorKind.loop_adapter));

    try std.testing.expect(TensorKind.dense.isWired());
    try std.testing.expect(TensorKind.hot_expert.isWired());
    try std.testing.expect(TensorKind.shared_expert.isWired());
    try std.testing.expect(TensorKind.loop_adapter.isWired());
    try std.testing.expect(!TensorKind.cold_expert.isWired());
    try std.testing.expect(TensorKind.cold_expert.isCold());
    try std.testing.expect(!TensorKind.sparse_table.isWired());
    try std.testing.expect(TensorKind.sparse_table.isCold());

    // 2. Build mixed multi-tensor .amf image with wired + cold entries (one 16 KiB page each).
    var entries: [5]TensorEntry = undefined;

    entries[0] = TensorEntry.zeroed("layers.0.mlp.shared_expert.gate_proj", .shared_expert, .f16);
    entries[0].rank = 2;
    entries[0].layer = 0;
    entries[0].shape = .{ 16, 64, 0, 0 };
    entries[0].data_bytes = 16 * 64 * 2;

    entries[1] = TensorEntry.zeroed("layers.0.loop.1.lora_a", .loop_adapter, .bf16);
    entries[1].rank = 2;
    entries[1].layer = 0;
    entries[1].shape = .{ 16, 64, 0, 0 };
    entries[1].data_bytes = try sme2Panel64BlobBytes(.bf16, 16, 64, 0);
    entries[1].setLayoutAndParamKind(.sme2_panel64, .affine_scale_bias);

    entries[2] = TensorEntry.zeroed("layers.0.loop.alpha", .loop_adapter, .f32);
    entries[2].rank = 1;
    entries[2].layer = 0;
    entries[2].shape = .{ 4, 0, 0, 0 };
    entries[2].data_bytes = 4 * 4;

    entries[3] = TensorEntry.zeroed("layers.0.mlp.experts.9.gate_proj", .cold_expert, .f16);
    entries[3].rank = 2;
    entries[3].layer = 0;
    entries[3].expert = 9;
    entries[3].shape = .{ 16, 64, 0, 0 };
    entries[3].data_bytes = 16 * 64 * 2;

    entries[4] = TensorEntry.zeroed("ple.layers.0.shard.0.weight", .sparse_table, .bf16);
    entries[4].rank = 2;
    entries[4].layer = 0;
    entries[4].shape = .{ 32, 64, 0, 0 };
    entries[4].data_bytes = 32 * 64 * 2;

    const image = try buildTestAmfMulti(gpa, &entries);
    defer gpa.free(image);

    var parsed = try Parsed.parseBytes(gpa, image);
    defer parsed.deinit(gpa);
    try parsed.verifyAllBlobsFromBytes(image);

    // Wired total must include shared_expert + 2 loop_adapter entries (3 pages),
    // and strictly exclude cold_expert + sparse_table (2 pages).
    try std.testing.expectEqual(@as(u64, 3 * page.APPLE_PAGE_SIZE), parsed.totalWiredBytes());
    try std.testing.expectEqual(@as(u64, 2 * page.APPLE_PAGE_SIZE), parsed.totalColdBytes());

    const ple_entry = parsed.find("ple.layers.0.shard.0.weight") orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(TensorKind.sparse_table, try ple_entry.tensorKind());
    try std.testing.expect(!(try ple_entry.isWired()));
    // Sparse tables also expose their 16 KiB direct-I/O range via asNvmeSource.
    const nvme_src = try ple_entry.asNvmeSource(0);
    try std.testing.expectError(error.NotResident, nvme_src.residentBytes());

    // Load wired blobs into a WiredArena sized strictly to totalWiredBytes() (3 pages):
    // if sparse_table or cold_expert were mistakenly loaded, allocation would fail OutOfSpace!
    const exact_cap: usize = @intCast(parsed.totalWiredBytes());
    const budget = budget_mod.Budget.compute(.{
        .memory_bytes = 16 * page.APPLE_PAGE_SIZE,
        .user_wire_limit = exact_cap,
        .global_user_wire_limit = exact_cap,
        .gpu_working_set_bytes = exact_cap,
        .configured_cap = null,
        .headroom_bytes = page.APPLE_PAGE_SIZE,
    });
    var mem_arena = try arena_mod.WiredArena.init(exact_cap, budget);
    defer mem_arena.deinit();

    const wired_from_bytes = try parsed.loadWiredFromBytes(gpa, image, &mem_arena);
    defer gpa.free(wired_from_bytes);
    try std.testing.expectEqual(@as(usize, 0), mem_arena.remaining());
    try std.testing.expect(wired_from_bytes[0] != null); // shared_expert
    try std.testing.expect(wired_from_bytes[1] != null); // loop_adapter lora_a
    try std.testing.expect(wired_from_bytes[2] != null); // loop_adapter alpha
    try std.testing.expect(wired_from_bytes[3] == null); // cold_expert excluded
    try std.testing.expect(wired_from_bytes[4] == null); // sparse_table excluded

    // Disk round-trip via AmfFile.open (F_NOCACHE) + AmfFile.loadWired().
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "m6_mixed.amf", .data = image });

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const raw_path = try std.fmt.bufPrint(path_buf[0 .. path_buf.len - 1], ".zig-cache/tmp/{s}/m6_mixed.amf", .{tmp.sub_path});
    path_buf[raw_path.len] = 0;
    const rel_path: [:0]const u8 = path_buf[0..raw_path.len :0];

    var amf_file = try AmfFile.open(gpa, rel_path);
    defer amf_file.close(gpa);
    try amf_file.verifyAllBlobs(gpa);
    try std.testing.expectEqual(@as(u64, 3 * page.APPLE_PAGE_SIZE), amf_file.totalWiredBytes());
    try std.testing.expectEqual(@as(u64, 2 * page.APPLE_PAGE_SIZE), amf_file.totalColdBytes());

    var disk_arena = try arena_mod.WiredArena.init(exact_cap, budget);
    defer disk_arena.deinit();
    const wired_from_disk = try amf_file.loadWired(gpa, &disk_arena);
    defer gpa.free(wired_from_disk);
    try std.testing.expectEqual(@as(usize, 0), disk_arena.remaining());
    try std.testing.expect(wired_from_disk[0] != null);
    try std.testing.expect(wired_from_disk[1] != null);
    try std.testing.expect(wired_from_disk[2] != null);
    try std.testing.expect(wired_from_disk[3] == null);
    try std.testing.expect(wired_from_disk[4] == null);
    try std.testing.expectEqualSlices(u8, wired_from_bytes[0].?, wired_from_disk[0].?);
    try std.testing.expectEqualSlices(u8, wired_from_bytes[1].?, wired_from_disk[1].?);

    // 3. Reject malformed sparse_table metadata (wrong rank, non-zero trailing shape, sme2_panel64).
    const small_buf: [2048]u8 = @splat(0x42);

    var bad_rank = TensorEntry.zeroed("ple.layers.0.weight", .sparse_table, .f16);
    bad_rank.rank = 1;
    bad_rank.shape = .{ 64, 0, 0, 0 };
    bad_rank.data_bytes = 64 * 2;
    const bad_rank_img = try buildTestAmf(gpa, bad_rank, &small_buf);
    defer gpa.free(bad_rank_img);
    try std.testing.expectError(error.InvalidRank, Parsed.parseBytes(gpa, bad_rank_img));

    var bad_tail = TensorEntry.zeroed("ple.layers.0.weight", .sparse_table, .f16);
    bad_tail.rank = 2;
    bad_tail.shape = .{ 8, 64, 1, 0 };
    bad_tail.data_bytes = 8 * 64 * 2;
    const bad_tail_img = try buildTestAmf(gpa, bad_tail, &small_buf);
    defer gpa.free(bad_tail_img);
    try std.testing.expectError(error.InvalidShape, Parsed.parseBytes(gpa, bad_tail_img));

    var bad_layout = TensorEntry.zeroed("ple.layers.0.weight", .sparse_table, .f16);
    bad_layout.rank = 2;
    bad_layout.shape = .{ 16, 64, 0, 0 };
    bad_layout.data_bytes = try sme2Panel64BlobBytes(.f16, 16, 64, 0);
    bad_layout.setLayoutAndParamKind(.sme2_panel64, .affine_scale_bias);
    const bad_layout_img = try buildTestAmf(gpa, bad_layout, &small_buf);
    defer gpa.free(bad_layout_img);
    try std.testing.expectError(error.InvalidTensorLayout, Parsed.parseBytes(gpa, bad_layout_img));
}

test "QuantParamKind.lut16_codebook (reserved0=0x0101) validates across q4/q3/q2 sme2_panel64 and rejects row_major" {
    const gpa = std.testing.allocator;

    const specs = [_]struct { dtype: DType, bits: u8 }{
        .{ .dtype = .q4_affine, .bits = 4 },
        .{ .dtype = .q3_affine, .bits = 3 },
        .{ .dtype = .q2_affine, .bits = 2 },
    };

    for (specs) |sp| {
        const blob_len = try sme2Panel64BlobBytes(sp.dtype, 25, 128, 64);
        const buf = try gpa.alloc(u8, @intCast(blob_len));
        defer gpa.free(buf);
        @memset(buf, 0x6D);

        var entry = TensorEntry.zeroed("layers.0.mlp.shared_expert.down_proj", .shared_expert, sp.dtype);
        entry.quant_bits = sp.bits;
        entry.rank = 2;
        entry.group_size = 64;
        entry.layer = 0;
        entry.shape = .{ 25, 128, 0, 0 };
        entry.data_bytes = blob_len;
        entry.setLayoutAndParamKind(.sme2_panel64, .lut16_codebook);
        try std.testing.expectEqual(@as(u16, 0x0101), entry.reserved0);

        const img = try buildTestAmf(gpa, entry, buf);
        defer gpa.free(img);
        var parsed = try Parsed.parseBytes(gpa, img);
        defer parsed.deinit(gpa);
        try parsed.verifyAllBlobsFromBytes(img);

        const found = parsed.find("layers.0.mlp.shared_expert.down_proj") orelse return error.TestUnexpectedResult;
        try std.testing.expectEqual(@as(u16, 0x0101), found.reserved0);
        try std.testing.expectEqual(TensorLayout.sme2_panel64, try found.tensorLayout());
        try std.testing.expectEqual(QuantParamKind.lut16_codebook, try found.quantParamKind());
        try std.testing.expectEqual(TensorKind.shared_expert, try found.tensorKind());

        // Reject lut16_codebook on row_major quantized tensor (`reserved0 == 0x0100`).
        const rm_data: u64 = 16 * ((128 * @as(u64, sp.bits)) / 8);
        const rm_params_off: u64 = align64(@intCast(rm_data));
        const rm_params: u64 = 16 * (128 / 64) * @sizeOf(quant.ScaleBias);
        const rm_buf = try gpa.alloc(u8, @intCast(rm_params_off + rm_params));
        defer gpa.free(rm_buf);
        @memset(rm_buf, 0x22);

        var bad_rm = TensorEntry.zeroed("bad.rm.lut16", .dense, sp.dtype);
        bad_rm.quant_bits = sp.bits;
        bad_rm.rank = 2;
        bad_rm.group_size = 64;
        bad_rm.shape = .{ 16, 128, 0, 0 };
        bad_rm.data_bytes = rm_data;
        bad_rm.params_offset = rm_params_off;
        bad_rm.params_bytes = rm_params;
        bad_rm.setLayoutAndParamKind(.row_major, .lut16_codebook);
        try std.testing.expectEqual(@as(u16, 0x0100), bad_rm.reserved0);

        const bad_img = try buildTestAmf(gpa, bad_rm, rm_buf);
        defer gpa.free(bad_img);
        try std.testing.expectError(error.InvalidQuantParamKind, Parsed.parseBytes(gpa, bad_img));
    }
}
