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

pub const TensorKind = enum(u8) {
    dense = 0,
    hot_expert = 1,
    cold_expert = 2,
    embedding = 3,
    norm = 4,
    router = 5,
    lm_head = 6,
};

pub const DType = enum(u8) {
    f16 = 0,
    f32 = 1,
    q4_affine = 2,
    q3_affine = 3,
};

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

    /// Exposes a cold-expert directory record as an `ExpertSource.local_nvme_ssd`.
    pub fn asNvmeSource(self: *const TensorEntry, file_index: u16) expert.NvmeExpert.InitError!expert.ExpertSource {
        return .{ .local_nvme_ssd = try expert.NvmeExpert.init(file_index, self.blob_offset, self.blob_bytes) };
    }
};

comptime {
    if (@sizeOf(Header) != 128) @compileError("AMF Header must be 128 bytes");
    if (@sizeOf(TensorEntry) != 128) @compileError("AMF TensorEntry must be 128 bytes");
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

pub const Parsed = struct {
    header: Header,
    entries: []TensorEntry,
    file_size: u64,

    pub fn deinit(self: *Parsed, allocator: std.mem.Allocator) void {
        allocator.free(self.entries);
        self.entries = &.{};
    }

    pub fn modelId(self: *const Parsed) []const u8 {
        return std.mem.sliceTo(&self.header.model_id, 0);
    }

    pub fn find(self: *const Parsed, name: []const u8) ?*const TensorEntry {
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

        try validateEntries(allocator, &hdr, entries, bytes.len);
        return .{ .header = hdr, .entries = entries, .file_size = bytes.len };
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

        try validateEntries(allocator, &hdr, entries, file_size);
        return .{
            .fd = fd,
            .parsed = .{
                .header = hdr,
                .entries = entries,
                .file_size = file_size,
            },
        };
    }

    pub fn close(self: *AmfFile, allocator: std.mem.Allocator) void {
        self.parsed.deinit(allocator);
        _ = std.c.close(self.fd);
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

    /// Streams every blob through a scratch buffer to verify all blob CRCs on disk.
    pub fn verifyAllBlobs(self: *const AmfFile, allocator: std.mem.Allocator) (ParseError || IoError)!void {
        var scratch_len: usize = 0;
        for (self.parsed.entries) |*entry| {
            scratch_len = @max(scratch_len, @as(usize, @intCast(entry.blob_bytes)));
        }
        if (scratch_len == 0) return;
        const buf = try allocator.alignedAlloc(u8, .fromByteUnits(page.APPLE_PAGE_SIZE), scratch_len);
        defer allocator.free(buf);
        for (self.parsed.entries) |*entry| {
            const len: usize = @intCast(entry.blob_bytes);
            try preadExact(self.fd, buf[0..len], entry.blob_offset);
            if (std.hash.Crc32.hash(buf[0..len]) != entry.blob_crc32) {
                return error.BlobChecksumMismatch;
            }
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

fn validateEntries(
    allocator: std.mem.Allocator,
    hdr: *const Header,
    entries: []const TensorEntry,
    file_size: u64,
) ParseError!void {
    const data_start = hdr.directory_offset + hdr.directory_bytes;
    const ranges = try allocator.alloc(Range, entries.len);
    defer allocator.free(ranges);

    for (entries, 0..) |*entry, i| {
        const name = entry.nameSlice();
        if (name.len == 0) return error.EmptyTensorName;
        for (entries[0..i]) |*prev| {
            if (std.mem.eql(u8, prev.nameSlice(), name)) return error.DuplicateTensorName;
        }
        _ = try entry.tensorKind();
        const dtype = try entry.dataType();
        if (entry.rank == 0 or entry.rank > 4) return error.InvalidRank;

        if (entry.blob_offset < data_start or
            entry.blob_offset % page.APPLE_PAGE_SIZE != 0 or
            entry.blob_bytes == 0 or
            entry.blob_bytes % page.APPLE_PAGE_SIZE != 0)
        {
            return error.MisalignedBlob;
        }
        const blob_end = std.math.add(u64, entry.blob_offset, entry.blob_bytes) catch return error.BlobOutOfBounds;
        if (blob_end > file_size) return error.BlobOutOfBounds;

        switch (dtype) {
            .f16, .f32 => {
                if (entry.quant_bits != 0 or entry.group_size != 0) return error.InvalidQuantization;
                if (entry.params_offset != 0 or entry.params_bytes != 0) return error.MisalignedParams;
                var elems: u64 = 1;
                for (0..entry.rank) |d| {
                    if (entry.shape[d] == 0) return error.InvalidShape;
                    elems = std.math.mul(u64, elems, entry.shape[d]) catch return error.InvalidShape;
                }
                const elem_size: u64 = if (dtype == .f16) 2 else 4;
                const expected_bytes = std.math.mul(u64, elems, elem_size) catch return error.InvalidShape;
                if (entry.data_bytes != expected_bytes or entry.data_bytes > entry.blob_bytes) {
                    return error.InvalidShape;
                }
            },
            .q4_affine, .q3_affine => {
                const expected_bits: u8 = if (dtype == .q4_affine) 4 else 3;
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
        ranges[i] = .{ .start = entry.blob_offset, .end = blob_end };
    }

    std.mem.sort(Range, ranges, {}, Range.lessThan);
    for (1..ranges.len) |i| {
        if (ranges[i].start < ranges[i - 1].end) return error.OverlappingBlobs;
    }
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
