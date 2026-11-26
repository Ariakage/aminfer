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

/// `Header.flags` bit indicating cold embedded metadata blobs (`__meta.*`, ADR-0022, W18).
pub const FLAG_EMBEDDED_METADATA: u32 = 1 << 0;
pub const KNOWN_HEADER_FLAGS: u32 = FLAG_EMBEDDED_METADATA;

/// Reserved directory entry prefix & canonical names for single-file `.amf` / `.amlora`
/// cold metadata payloads (`ADR-0022`, `W18`, `W19`).
pub const META_PREFIX: []const u8 = "__meta.";
pub const META_CONFIG_JSON: []const u8 = "__meta.config_json";
pub const META_TOKENIZER_JSON: []const u8 = "__meta.tokenizer_json";
pub const META_ADAPTER_CONFIG_JSON: []const u8 = "__meta.adapter_config_json";
pub const META_CHAT_TEMPLATE_JSON: []const u8 = "__meta.chat_template_json";
/// Canonical W24 `.amf` embedded chat-template spec metadata entry name.
pub const CHAT_TEMPLATE_METADATA_NAME: []const u8 = META_CHAT_TEMPLATE_JSON;
/// Tolerated read alias for `.amf` chat-template spec payloads.
pub const LEGACY_CHAT_TEMPLATE_METADATA_NAME: []const u8 = "__aminfer_chat_template__";

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
    /// Cold 16 KiB-aligned UTF-8 JSON metadata (`__meta.config_json`,
    /// `__meta.tokenizer_json`, `__meta.adapter_config_json`, ADR-0022, W18/W19).
    /// Never `mlock`ed into wired weight shards (`0` runtime decode overhead).
    metadata = 10,

    /// Returns `true` for tensor kinds loaded into startup wired RAM and `false`
    /// for cold NVMe-streamed or metadata kinds (`cold_expert`, `sparse_table`, `metadata`).
    pub fn isWired(self: TensorKind) bool {
        return switch (self) {
            .dense, .hot_expert, .embedding, .norm, .router, .lm_head, .shared_expert, .loop_adapter => true,
            .cold_expert, .sparse_table, .metadata => false,
        };
    }

    pub fn isCold(self: TensorKind) bool {
        return switch (self) {
            .cold_expert, .sparse_table => true,
            .dense, .hot_expert, .embedding, .norm, .router, .lm_head, .shared_expert, .loop_adapter, .metadata => false,
        };
    }

    pub fn isMetadata(self: TensorKind) bool {
        return self == .metadata;
    }
};

pub fn isMetaName(name: []const u8) bool {
    return std.mem.startsWith(u8, name, META_PREFIX) or
        std.mem.eql(u8, name, LEGACY_CHAT_TEMPLATE_METADATA_NAME);
}

pub fn isMetaEntry(entry: *const TensorEntry) bool {
    return entry.kind == @backingInt(TensorKind.metadata) or isMetaName(entry.nameSlice());
}

/// Builds a canonical `TensorKind.metadata` directory record for a non-empty
/// UTF-8 metadata blob (`__meta.*` or `LEGACY_CHAT_TEMPLATE_METADATA_NAME`) of
/// exact unpadded length `payload_len`.
pub fn makeMetadataEntry(name_str: []const u8, payload_len: u64) ParseError!TensorEntry {
    if (!isMetaName(name_str) or name_str.len >= 48) return error.InvalidTensorKind;
    if (payload_len == 0 or payload_len > std.math.maxInt(u32)) return error.InvalidShape;
    var entry = TensorEntry.zeroed(name_str, .metadata, .u32);
    entry.rank = 1;
    entry.shape = .{ @intCast(payload_len), 0, 0, 0 };
    entry.data_bytes = payload_len;
    return entry;
}

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
    /// MLX-compatible 2-bit affine group quantization (ADR-0017, W14/G13):
    /// `quant_bits = 2`, packed codes + `half2(scale, bias)` per group.
    q2_affine = 6,
    /// MLX-compatible 5/6/8-bit affine group quantization (`group_size in {32, 64, 128}`).
    q5_affine = 7,
    q6_affine = 8,
    q8_affine = 9,
    /// `128x128` block-scaled FP8 (`E4M3FN` 1 B/el + `[ceil(N/128), ceil(K/128)]` `f32` scales).
    fp8_e4m3_block = 10,
    /// Per-row FP8 (`E4M3FN` 1 B/el + `f32[N]` scales, e.g. Engram tables).
    fp8_e4m3_row = 11,
    /// Per-tensor/per-channel FP8 (`E4M3FN` 1 B/el + `f32[1]` or `f32[N]` scale).
    fp8_e4m3_tensor = 12,
    /// OCP `MXFP8` (`E4M3FN` 1 B/el + `u8` `UE8M0` scale per 32 weights).
    mxfp8_g32 = 13,
    /// NVIDIA `NVFP4` (`E2M1` low-nibble-first `0.5 B/el` + `[N, K/16]` `u8` `E4M3`
    /// micro-scales padded to 4B + trailing `f32[1]` multiplicative outer scale).
    nvfp4_g16 = 14,
    /// Standard upstream `GGUF` raw block dtypes (`15..37`, zero double-quantization).
    gguf_q4_0 = 15,
    gguf_q4_1 = 16,
    gguf_q5_0 = 17,
    gguf_q5_1 = 18,
    gguf_q8_0 = 19,
    gguf_q2_k = 20,
    gguf_q3_k = 21,
    gguf_q4_k = 22,
    gguf_q5_k = 23,
    gguf_q6_k = 24,
    gguf_iq1_s = 25,
    gguf_iq1_m = 26,
    gguf_iq2_xxs = 27,
    gguf_iq2_xs = 28,
    gguf_iq2_s = 29,
    gguf_iq3_xxs = 30,
    gguf_iq3_s = 31,
    gguf_iq4_nl = 32,
    gguf_iq4_xs = 33,
    gguf_tq1_0 = 34,
    gguf_tq2_0 = 35,
    gguf_mxfp4 = 36,
    gguf_q2_0 = 37,

    /// Bytes per element for unquantized dtypes; `null` for quantized/block ones.
    pub fn unquantizedElemSize(self: DType) ?u64 {
        return switch (self) {
            .f16, .bf16 => 2,
            .f32, .u32 => 4,
            else => null,
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
        else => return error.InvalidTensorLayout,
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

        const count = std.math.cast(usize, hdr.tensor_count) orelse return error.DirectoryOutOfBounds;
        const dir_raw_len = std.math.mul(usize, count, @sizeOf(TensorEntry)) catch return error.DirectoryOutOfBounds;
        const dir_off = std.math.cast(usize, hdr.directory_offset) orelse return error.DirectoryOutOfBounds;
        const dir_end = std.math.add(usize, dir_off, dir_raw_len) catch return error.DirectoryOutOfBounds;
        if (dir_end > bytes.len) return error.DirectoryOutOfBounds;

        const entries = try allocator.alloc(TensorEntry, count);
        errdefer allocator.free(entries);

        if (count > 0) {
            const dir_slice = bytes[dir_off..dir_end];
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
            const start = std.math.cast(usize, entry.blob_offset) orelse return error.BlobOutOfBounds;
            const len = std.math.cast(usize, entry.blob_bytes) orelse return error.BlobOutOfBounds;
            const end = std.math.add(usize, start, len) catch return error.BlobOutOfBounds;
            if (end > bytes.len) return error.BlobOutOfBounds;
            if (std.hash.Crc32.hash(bytes[start..end]) != entry.blob_crc32) {
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

    /// Returns total 16 KiB-aligned blob bytes across all embedded cold metadata
    /// payloads (`TensorKind.metadata`, `__meta.*`).
    pub fn totalMetadataBytes(self: *const Parsed) u64 {
        var total: u64 = 0;
        for (self.entries) |*entry| {
            const kind = std.enums.fromInt(TensorKind, entry.kind) orelse continue;
            if (kind.isMetadata()) {
                total += entry.blob_bytes;
            }
        }
        return total;
    }

    pub fn hasEmbeddedMetadata(self: *const Parsed, meta_name: []const u8) bool {
        const entry = self.find(meta_name) orelse return false;
        return entry.kind == @backingInt(TensorKind.metadata);
    }

    pub fn hasEmbeddedConfig(self: *const Parsed) bool {
        return self.hasEmbeddedMetadata(META_CONFIG_JSON);
    }

    pub fn hasEmbeddedTokenizer(self: *const Parsed) bool {
        return self.hasEmbeddedMetadata(META_TOKENIZER_JSON);
    }

    pub fn hasEmbeddedChatTemplate(self: *const Parsed) bool {
        return self.hasEmbeddedMetadata(META_CHAT_TEMPLATE_JSON) or
            self.hasEmbeddedMetadata(LEGACY_CHAT_TEMPLATE_METADATA_NAME);
    }

    /// Returns a zero-copy slice of an embedded metadata payload (`__meta.*` or
    /// `LEGACY_CHAT_TEMPLATE_METADATA_NAME`) from an in-memory `.amf` image after
    /// verifying its 16 KiB-aligned blob CRC-32.
    pub fn extractMetadataFromBytes(
        self: *const Parsed,
        bytes: []const u8,
        meta_name: []const u8,
    ) ParseError!?[]const u8 {
        const entry = self.find(meta_name) orelse return null;
        if ((try entry.tensorKind()) != .metadata) return error.InvalidTensorKind;
        const start = std.math.cast(usize, entry.blob_offset) orelse return error.BlobOutOfBounds;
        const blob_len = std.math.cast(usize, entry.blob_bytes) orelse return error.BlobOutOfBounds;
        const data_len = std.math.cast(usize, entry.data_bytes) orelse return error.BlobOutOfBounds;
        const blob_end = std.math.add(usize, start, blob_len) catch return error.BlobOutOfBounds;
        if (blob_end > bytes.len or data_len > blob_len) return error.BlobOutOfBounds;
        if (std.hash.Crc32.hash(bytes[start..blob_end]) != entry.blob_crc32) {
            return error.BlobChecksumMismatch;
        }
        return bytes[start .. start + data_len];
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
        const start = std.math.cast(usize, entry.blob_offset) orelse return error.BlobOutOfBounds;
        const len = std.math.cast(usize, entry.blob_bytes) orelse return error.BlobOutOfBounds;
        const end = std.math.add(usize, start, len) catch return error.BlobOutOfBounds;
        if (end > bytes.len) return error.BlobOutOfBounds;
        const dst = try arena.allocPages(len);
        @memcpy(dst, bytes[start..end]);
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

    pub fn totalMetadataBytes(self: *const AmfFile) u64 {
        return self.parsed.totalMetadataBytes();
    }

    /// Reads an embedded cold metadata payload (`__meta.*`) from disk via
    /// `F_NOCACHE` `pread` (zero `mlock`, zero wired-arena allocation), verifies
    /// its full 16 KiB-aligned blob CRC-32, and returns an owned slice of exact
    /// unpadded length `entry.data_bytes` (`null` when not embedded).
    pub fn readMetadataAlloc(
        self: *const AmfFile,
        allocator: std.mem.Allocator,
        meta_name: []const u8,
    ) (ParseError || IoError)!?[]u8 {
        const entry = self.parsed.find(meta_name) orelse return null;
        if ((try entry.tensorKind()) != .metadata) return error.InvalidTensorKind;
        const blob_len: usize = @intCast(entry.blob_bytes);
        const data_len: usize = @intCast(entry.data_bytes);
        if (data_len == 0 or data_len > blob_len) return error.InvalidShape;

        const out = try allocator.alloc(u8, data_len);
        errdefer allocator.free(out);

        const chunk_len: usize = @min(blob_len, VERIFY_CHUNK_BYTES);
        const scratch = try allocator.alignedAlloc(u8, .fromByteUnits(page.APPLE_PAGE_SIZE), chunk_len);
        defer allocator.free(scratch);

        var hasher = std.hash.Crc32.init();
        var done: usize = 0;
        while (done < blob_len) {
            const take: usize = @min(blob_len - done, chunk_len);
            try preadExact(self.fd, scratch[0..take], entry.blob_offset + done);
            hasher.update(scratch[0..take]);
            if (done < data_len) {
                const copy_n = @min(data_len - done, take);
                @memcpy(out[done .. done + copy_n], scratch[0..copy_n]);
            }
            done += take;
        }
        if (hasher.final() != entry.blob_crc32) return error.BlobChecksumMismatch;
        return out;
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
    if ((hdr.flags & ~KNOWN_HEADER_FLAGS) != 0) return error.UnsupportedVersion;
    if (hdr.page_size != page.APPLE_PAGE_SIZE) return error.UnsupportedPageSize;
    if (hdr.checksum_algo != CHECKSUM_CRC32_IEEE) return error.UnsupportedChecksum;
    if (computeHeaderCrc32(hdr) != hdr.header_crc32) return error.HeaderChecksumMismatch;

    if (hdr.directory_offset != page.APPLE_PAGE_SIZE) return error.MisalignedDirectory;
    if (hdr.directory_bytes % page.APPLE_PAGE_SIZE != 0) return error.MisalignedDirectory;
    const raw_dir = std.math.mul(u64, @as(u64, hdr.tensor_count), @sizeOf(TensorEntry)) catch return error.DirectoryOutOfBounds;
    const expected_dir_bytes: u64 = if (raw_dir == 0)
        0
    else blk: {
        const padded = std.math.add(u64, raw_dir, page.APPLE_PAGE_SIZE - 1) catch return error.DirectoryOutOfBounds;
        break :blk (padded / page.APPLE_PAGE_SIZE) * page.APPLE_PAGE_SIZE;
    };
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

        const name_is_meta = isMetaName(name);
        if (kind == .metadata or name_is_meta) {
            if (kind != .metadata or !name_is_meta) return error.InvalidTensorKind;
            if (layout != .row_major) return error.InvalidTensorLayout;
            if (param_kind != .affine_scale_bias) return error.InvalidQuantParamKind;
            if (dtype != .u32) return error.InvalidDType;
            if (entry.quant_bits != 0 or entry.group_size != 0) return error.InvalidQuantization;
            if (entry.params_offset != 0 or entry.params_bytes != 0) return error.MisalignedParams;
            if (entry.rank != 1) return error.InvalidRank;
            if (entry.layer != NO_LAYER or entry.expert != NO_EXPERT) return error.InvalidShape;
            if (entry.shape[0] == 0 or entry.shape[1] != 0 or entry.shape[2] != 0 or entry.shape[3] != 0) {
                return error.InvalidShape;
            }
            if (entry.data_bytes != entry.shape[0] or entry.data_bytes > entry.blob_bytes) {
                return error.InvalidShape;
            }
            ranges[i] = .{ .start = entry.blob_offset, .end = blob_end };
            continue;
        }

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
                    .q2_affine, .q3_affine, .q4_affine, .q5_affine, .q6_affine, .q8_affine => {
                        const expected_bits: u8 = switch (dtype) {
                            .q2_affine => 2,
                            .q3_affine => 3,
                            .q4_affine => 4,
                            .q5_affine => 5,
                            .q6_affine => 6,
                            .q8_affine => 8,
                            else => unreachable,
                        };
                        if (entry.rank != 2) return error.InvalidRank;
                        if (entry.quant_bits != expected_bits or
                            (entry.group_size != 32 and entry.group_size != 64 and entry.group_size != 128))
                        {
                            return error.InvalidQuantization;
                        }
                        const n: u64 = entry.shape[0];
                        const k: u64 = entry.shape[1];
                        if (n == 0 or k == 0 or k % entry.group_size != 0) return error.InvalidShape;
                        const k_bits = std.math.mul(u64, k, expected_bits) catch return error.InvalidShape;
                        const row_data = k_bits / 8;
                        const expected_data = std.math.mul(u64, n, row_data) catch return error.InvalidShape;
                        const groups = std.math.mul(u64, n, k / entry.group_size) catch return error.InvalidShape;
                        const expected_params = std.math.mul(u64, groups, @sizeOf(quant.ScaleBias)) catch return error.InvalidShape;
                        if (entry.data_bytes != expected_data or entry.params_bytes != expected_params) {
                            return error.InvalidShape;
                        }
                        if (entry.params_offset < entry.data_bytes or entry.params_offset % 64 != 0) {
                            return error.MisalignedParams;
                        }
                        const end_in_blob = std.math.add(u64, entry.params_offset, entry.params_bytes) catch return error.BlobOutOfBounds;
                        if (end_in_blob > entry.blob_bytes) return error.BlobOutOfBounds;
                    },
                    .fp8_e4m3_block => {
                        if (entry.rank != 2) return error.InvalidRank;
                        if (entry.quant_bits != 8 or entry.group_size != 128) return error.InvalidQuantization;
                        const n: u64 = entry.shape[0];
                        const k: u64 = entry.shape[1];
                        if (n == 0 or k == 0) return error.InvalidShape;
                        const expected_data = std.math.mul(u64, n, k) catch return error.InvalidShape;
                        const nb_cnt = (n + 127) / 128;
                        const kb_cnt = (k + 127) / 128;
                        const scale_elems = std.math.mul(u64, nb_cnt, kb_cnt) catch return error.InvalidShape;
                        const expected_params = std.math.mul(u64, scale_elems, 4) catch return error.InvalidShape;
                        if (entry.data_bytes != expected_data or entry.params_bytes != expected_params) {
                            return error.InvalidShape;
                        }
                        if (entry.params_offset < entry.data_bytes or entry.params_offset % 64 != 0) {
                            return error.MisalignedParams;
                        }
                        const end_in_blob = std.math.add(u64, entry.params_offset, entry.params_bytes) catch return error.BlobOutOfBounds;
                        if (end_in_blob > entry.blob_bytes) return error.BlobOutOfBounds;
                    },
                    .fp8_e4m3_row => {
                        if (entry.rank != 2) return error.InvalidRank;
                        if (entry.quant_bits != 8 or entry.group_size != 0) return error.InvalidQuantization;
                        const n: u64 = entry.shape[0];
                        const k: u64 = entry.shape[1];
                        if (n == 0 or k == 0) return error.InvalidShape;
                        const expected_data = std.math.mul(u64, n, k) catch return error.InvalidShape;
                        const expected_params = std.math.mul(u64, n, 4) catch return error.InvalidShape;
                        if (entry.data_bytes != expected_data or entry.params_bytes != expected_params) {
                            return error.InvalidShape;
                        }
                        if (entry.params_offset < entry.data_bytes or entry.params_offset % 64 != 0) {
                            return error.MisalignedParams;
                        }
                        const end_in_blob = std.math.add(u64, entry.params_offset, entry.params_bytes) catch return error.BlobOutOfBounds;
                        if (end_in_blob > entry.blob_bytes) return error.BlobOutOfBounds;
                    },
                    .fp8_e4m3_tensor => {
                        if (entry.rank != 1 and entry.rank != 2) return error.InvalidRank;
                        if (entry.quant_bits != 8 or entry.group_size != 0) return error.InvalidQuantization;
                        var elems: u64 = 1;
                        for (0..entry.rank) |d| {
                            if (entry.shape[d] == 0) return error.InvalidShape;
                            elems = std.math.mul(u64, elems, entry.shape[d]) catch return error.InvalidShape;
                        }
                        const row_params = std.math.mul(u64, entry.shape[0], 4) catch return error.InvalidShape;
                        const params_ok = entry.params_bytes == 4 or (entry.rank == 2 and entry.params_bytes == row_params);
                        if (entry.data_bytes != elems or !params_ok) return error.InvalidShape;
                        if (entry.params_offset < entry.data_bytes or entry.params_offset % 64 != 0) {
                            return error.MisalignedParams;
                        }
                        const end_in_blob = std.math.add(u64, entry.params_offset, entry.params_bytes) catch return error.BlobOutOfBounds;
                        if (end_in_blob > entry.blob_bytes) return error.BlobOutOfBounds;
                    },
                    .mxfp8_g32 => {
                        if (entry.rank != 2) return error.InvalidRank;
                        if (entry.quant_bits != 8 or entry.group_size != 32) return error.InvalidQuantization;
                        const n: u64 = entry.shape[0];
                        const k: u64 = entry.shape[1];
                        if (n == 0 or k == 0 or k % 32 != 0) return error.InvalidShape;
                        const expected_data = std.math.mul(u64, n, k) catch return error.InvalidShape;
                        const expected_params = std.math.mul(u64, n, k / 32) catch return error.InvalidShape;
                        if (entry.data_bytes != expected_data or entry.params_bytes != expected_params) {
                            return error.InvalidShape;
                        }
                        if (entry.params_offset < entry.data_bytes or entry.params_offset % 64 != 0) {
                            return error.MisalignedParams;
                        }
                        const end_in_blob = std.math.add(u64, entry.params_offset, entry.params_bytes) catch return error.BlobOutOfBounds;
                        if (end_in_blob > entry.blob_bytes) return error.BlobOutOfBounds;
                    },
                    .nvfp4_g16 => {
                        if (entry.rank != 2) return error.InvalidRank;
                        if (entry.quant_bits != 4 or entry.group_size != 16) return error.InvalidQuantization;
                        const n: u64 = entry.shape[0];
                        const k: u64 = entry.shape[1];
                        if (n == 0 or k == 0 or k % 16 != 0) return error.InvalidShape;
                        const expected_data = std.math.mul(u64, n, k / 2) catch return error.InvalidShape;
                        const micro_bytes = std.math.mul(u64, n, k / 16) catch return error.InvalidShape;
                        const padded_micro = (std.math.add(u64, micro_bytes, 3) catch return error.InvalidShape) & ~@as(u64, 3);
                        const expected_params = std.math.add(u64, padded_micro, 4) catch return error.InvalidShape;
                        if (entry.data_bytes != expected_data or entry.params_bytes != expected_params) {
                            return error.InvalidShape;
                        }
                        if (entry.params_offset < entry.data_bytes or entry.params_offset % 64 != 0) {
                            return error.MisalignedParams;
                        }
                        const end_in_blob = std.math.add(u64, entry.params_offset, entry.params_bytes) catch return error.BlobOutOfBounds;
                        if (end_in_blob > entry.blob_bytes) return error.BlobOutOfBounds;
                    },
                    else => {
                        const info = quant.formats.ggufBlockInfo(dtype) orelse return error.InvalidDType;
                        if (entry.rank != 1 and entry.rank != 2) return error.InvalidRank;
                        if (entry.quant_bits != info.quant_bits or entry.group_size != info.block_elems) {
                            return error.InvalidQuantization;
                        }
                        if (entry.params_offset != 0 or entry.params_bytes != 0) return error.MisalignedParams;
                        const n: u64 = if (entry.rank == 2) entry.shape[0] else 1;
                        const k: u64 = if (entry.rank == 2) entry.shape[1] else entry.shape[0];
                        if (n == 0 or k == 0 or k % info.block_elems != 0) return error.InvalidShape;
                        const blocks_per_row = k / info.block_elems;
                        const row_bytes = std.math.mul(u64, blocks_per_row, info.block_bytes) catch return error.InvalidShape;
                        const expected_data = std.math.mul(u64, n, row_bytes) catch return error.InvalidShape;
                        if (entry.data_bytes != expected_data or entry.data_bytes > entry.blob_bytes) {
                            return error.InvalidShape;
                        }
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
                    else => return error.InvalidTensorLayout,
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

// ============================================================================
// W18: Self-Contained Single-File `.amf` Metadata & Zero-`mlock` Inspection
// ============================================================================

pub const MetadataSource = enum(u8) {
    none = 0,
    explicit_override = 1,
    embedded = 2,
    sidecar = 3,

    pub fn label(self: MetadataSource) []const u8 {
        return switch (self) {
            .none => "none",
            .explicit_override => "explicit_override",
            .embedded => "embedded",
            .sidecar => "sidecar",
        };
    }
};

pub const ResolvedMetadata = struct {
    bytes: []u8,
    source: MetadataSource,
    crc32: u32,

    pub fn deinit(self: *ResolvedMetadata, allocator: std.mem.Allocator) void {
        if (self.bytes.len > 0) allocator.free(self.bytes);
        self.* = .{ .bytes = &.{}, .source = .none, .crc32 = 0 };
    }
};

pub const ResolveMetadataError = ParseError || IoError || error{
    MetadataNotFound,
    PathTooLong,
    MetadataTooLarge,
};

const MAX_METADATA_FILE_BYTES: u64 = 128 * 1024 * 1024;

fn toPosixPathZ(path: []const u8, buf: *[std.fs.max_path_bytes]u8) error{PathTooLong}![:0]const u8 {
    if (path.len == 0 or path.len >= buf.len) return error.PathTooLong;
    @memcpy(buf[0..path.len], path);
    buf[path.len] = 0;
    return buf[0..path.len :0];
}

fn readRegularFileAlloc(allocator: std.mem.Allocator, path: []const u8) ResolveMetadataError![]u8 {
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_z = try toPosixPathZ(path, &path_buf);
    const fd = std.c.open(path_z.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.OpenFailed;
    defer _ = std.c.close(fd);

    var st: std.c.Stat = undefined;
    if (std.c.fstat(fd, &st) != 0) return error.StatFailed;
    if (st.size <= 0) return error.FileTooSmall;
    const file_len_u64: u64 = @intCast(st.size);
    if (file_len_u64 > MAX_METADATA_FILE_BYTES) return error.MetadataTooLarge;
    const file_len: usize = @intCast(file_len_u64);

    const out = try allocator.alloc(u8, file_len);
    errdefer allocator.free(out);
    try preadExact(fd, out, 0);
    return out;
}

/// Resolves a model metadata JSON payload (`config.json` or `tokenizer.json`) with
/// `ADR-0022` `W18` precedence:
///   1. `explicit_override_path` (`--config` / `--tokenizer` CLI flag)
///   2. Embedded cold metadata (`__meta.config_json` / `__meta.tokenizer_json`) inside `amf_file`
///   3. Sidecar `<dirname(amf_path)>/<sidecar_filename>`
pub fn resolveMetadataAlloc(
    allocator: std.mem.Allocator,
    amf_file: *const AmfFile,
    amf_path: []const u8,
    meta_name: []const u8,
    sidecar_filename: []const u8,
    explicit_override_path: ?[]const u8,
) ResolveMetadataError!ResolvedMetadata {
    if (explicit_override_path) |ov_path| {
        const bytes = try readRegularFileAlloc(allocator, ov_path);
        return .{
            .bytes = bytes,
            .source = .explicit_override,
            .crc32 = std.hash.Crc32.hash(bytes),
        };
    }

    if (try amf_file.readMetadataAlloc(allocator, meta_name)) |embedded_bytes| {
        const entry = amf_file.parsed.find(meta_name).?;
        return .{
            .bytes = embedded_bytes,
            .source = .embedded,
            .crc32 = entry.blob_crc32,
        };
    }

    const dir = std.fs.path.dirname(amf_path) orelse ".";
    var sidecar_buf: [std.fs.max_path_bytes]u8 = undefined;
    const sidecar_path = std.fmt.bufPrint(&sidecar_buf, "{s}/{s}", .{ dir, sidecar_filename }) catch {
        return error.PathTooLong;
    };
    const bytes = readRegularFileAlloc(allocator, sidecar_path) catch |err| switch (err) {
        error.OpenFailed => return error.MetadataNotFound,
        else => return err,
    };
    return .{
        .bytes = bytes,
        .source = .sidecar,
        .crc32 = std.hash.Crc32.hash(bytes),
    };
}

pub fn resolveConfigJsonAlloc(
    allocator: std.mem.Allocator,
    amf_file: *const AmfFile,
    amf_path: []const u8,
    explicit_config_path: ?[]const u8,
) ResolveMetadataError!ResolvedMetadata {
    return resolveMetadataAlloc(
        allocator,
        amf_file,
        amf_path,
        META_CONFIG_JSON,
        "config.json",
        explicit_config_path,
    );
}

pub fn resolveTokenizerJsonAlloc(
    allocator: std.mem.Allocator,
    amf_file: *const AmfFile,
    amf_path: []const u8,
    explicit_tokenizer_path: ?[]const u8,
) ResolveMetadataError!ResolvedMetadata {
    return resolveMetadataAlloc(
        allocator,
        amf_file,
        amf_path,
        META_TOKENIZER_JSON,
        "tokenizer.json",
        explicit_tokenizer_path,
    );
}

pub fn resolveChatTemplateJsonAlloc(
    allocator: std.mem.Allocator,
    amf_file: *const AmfFile,
    amf_path: []const u8,
    explicit_chat_template_path: ?[]const u8,
) ResolveMetadataError!ResolvedMetadata {
    if (explicit_chat_template_path) |ov_path| {
        const bytes = try readRegularFileAlloc(allocator, ov_path);
        return .{
            .bytes = bytes,
            .source = .explicit_override,
            .crc32 = std.hash.Crc32.hash(bytes),
        };
    }
    if (try amf_file.readMetadataAlloc(allocator, META_CHAT_TEMPLATE_JSON)) |embedded_bytes| {
        const entry = amf_file.parsed.find(META_CHAT_TEMPLATE_JSON).?;
        return .{
            .bytes = embedded_bytes,
            .source = .embedded,
            .crc32 = entry.blob_crc32,
        };
    }
    return resolveMetadataAlloc(
        allocator,
        amf_file,
        amf_path,
        LEGACY_CHAT_TEMPLATE_METADATA_NAME,
        "chat_template.json",
        null,
    );
}

pub const MetadataAttachment = struct {
    name: []const u8,
    payload: []const u8,
};

/// Returns a new 16 KiB-aligned `.amf` image embedding/replacing `__meta.*`
/// payloads inside `base_amf_bytes` while keeping all weight blobs bit-exact.
pub fn embedMetadataIntoAmfBytes(
    allocator: std.mem.Allocator,
    base_amf_bytes: []const u8,
    attachments: []const MetadataAttachment,
) (ParseError || error{Overflow})![]u8 {
    var parsed = try Parsed.parseBytes(allocator, base_amf_bytes);
    defer parsed.deinit(allocator);
    try parsed.verifyAllBlobsFromBytes(base_amf_bytes);

    for (attachments) |att| {
        _ = try makeMetadataEntry(att.name, att.payload.len);
    }

    var kept_indices = try allocator.alloc(usize, parsed.entries.len);
    defer allocator.free(kept_indices);
    var kept_count: usize = 0;
    var any_remaining_meta = false;

    for (parsed.entries, 0..) |*e, idx| {
        var replaced = false;
        for (attachments) |att| {
            if (std.mem.eql(u8, e.nameSlice(), att.name)) {
                replaced = true;
                break;
            }
        }
        if (!replaced) {
            kept_indices[kept_count] = idx;
            kept_count += 1;
            if (e.kind == @backingInt(TensorKind.metadata)) any_remaining_meta = true;
        }
    }

    const total_entries = kept_count + attachments.len;
    const new_entries = try allocator.alloc(TensorEntry, total_entries);
    defer allocator.free(new_entries);

    const dir_offset: usize = page.APPLE_PAGE_SIZE;
    const dir_raw_len: usize = total_entries * @sizeOf(TensorEntry);
    const dir_bytes: usize = if (total_entries == 0) 0 else try page.alignUp(dir_raw_len);
    var cursor: usize = dir_offset + dir_bytes;

    for (0..kept_count) |k| {
        const src_e = parsed.entries[kept_indices[k]];
        var dst_e = src_e;
        dst_e.blob_offset = cursor;
        new_entries[k] = dst_e;
        cursor = std.math.add(usize, cursor, @intCast(src_e.blob_bytes)) catch return error.BlobOutOfBounds;
    }

    for (attachments, 0..) |att, a_idx| {
        var meta_e = try makeMetadataEntry(att.name, att.payload.len);
        const blob_len = try page.alignUp(att.payload.len);
        meta_e.blob_offset = cursor;
        meta_e.blob_bytes = blob_len;
        new_entries[kept_count + a_idx] = meta_e;
        cursor = std.math.add(usize, cursor, blob_len) catch return error.BlobOutOfBounds;
    }

    const out = try allocator.alloc(u8, cursor);
    errdefer allocator.free(out);
    @memset(out, 0);

    for (0..kept_count) |k| {
        const src_e = parsed.entries[kept_indices[k]];
        const dst_e = &new_entries[k];
        const src_off: usize = @intCast(src_e.blob_offset);
        const dst_off: usize = @intCast(dst_e.blob_offset);
        const blob_len: usize = @intCast(src_e.blob_bytes);
        @memcpy(out[dst_off .. dst_off + blob_len], base_amf_bytes[src_off .. src_off + blob_len]);
    }

    for (attachments, 0..) |att, a_idx| {
        const dst_e = &new_entries[kept_count + a_idx];
        const dst_off: usize = @intCast(dst_e.blob_offset);
        const blob_len: usize = @intCast(dst_e.blob_bytes);
        @memcpy(out[dst_off .. dst_off + att.payload.len], att.payload);
        dst_e.blob_crc32 = std.hash.Crc32.hash(out[dst_off .. dst_off + blob_len]);
    }

    const dir_slice = std.mem.sliceAsBytes(new_entries);
    if (dir_slice.len > 0) {
        @memcpy(out[dir_offset .. dir_offset + dir_slice.len], dir_slice);
    }
    const dir_crc = std.hash.Crc32.hash(dir_slice);

    var hdr = parsed.header;
    hdr.tensor_count = @intCast(total_entries);
    hdr.directory_offset = dir_offset;
    hdr.directory_bytes = dir_bytes;
    hdr.directory_crc32 = dir_crc;
    if (attachments.len > 0 or any_remaining_meta) {
        hdr.flags |= FLAG_EMBEDDED_METADATA;
    } else {
        hdr.flags &= ~FLAG_EMBEDDED_METADATA;
    }
    hdr.header_crc32 = computeHeaderCrc32(&hdr);
    @memcpy(out[0..@sizeOf(Header)], std.mem.asBytes(&hdr));
    return out;
}

pub const DTYPE_COUNT: usize = 38;

pub const DTypeStats = struct {
    tensor_count: u32 = 0,
    wired_bytes: u64 = 0,
    cold_bytes: u64 = 0,
    data_bytes: u64 = 0,
};

pub const MetaBlobReport = struct {
    present: bool = false,
    data_bytes: u64 = 0,
    blob_bytes: u64 = 0,
    crc32: u32 = 0,
    crc_valid: bool = false,
};

pub const HardwareTier = enum(u8) {
    m4_16gb = 0,
    m4_24gb = 1,
    pro_48gb = 2,
    max_64gb = 3,
    max_128gb = 4,
    ultra_192gb = 5,
    ultra_512gb = 6,
    cluster_required = 7,

    pub fn label(self: HardwareTier) []const u8 {
        return switch (self) {
            .m4_16gb => "M4 16GB",
            .m4_24gb => "M4 24GB",
            .pro_48gb => "Pro 48GB",
            .max_64gb => "Max 64GB",
            .max_128gb => "Max 128GB",
            .ultra_192gb => "Ultra 192GB",
            .ultra_512gb => "Ultra 512GB",
            .cluster_required => "Cluster (>512GB)",
        };
    }

    /// Usable wired budget (`physical_ram - DEFAULT_HEADROOM_BYTES`, ADR-0012).
    pub fn usableWiredBytes(self: HardwareTier) u64 {
        const gib: u64 = 1024 * 1024 * 1024;
        const ram: u64 = switch (self) {
            .m4_16gb => 16 * gib,
            .m4_24gb => 24 * gib,
            .pro_48gb => 48 * gib,
            .max_64gb => 64 * gib,
            .max_128gb => 128 * gib,
            .ultra_192gb => 192 * gib,
            .ultra_512gb => 512 * gib,
            .cluster_required => 0,
        };
        return if (ram > budget_mod.DEFAULT_HEADROOM_BYTES)
            ram - budget_mod.DEFAULT_HEADROOM_BYTES
        else
            0;
    }
};

pub const TierFitSummary = struct {
    fits_m4_16gb: bool,
    fits_m4_24gb: bool,
    fits_pro_48gb: bool,
    fits_max_64gb: bool,
    fits_max_128gb: bool,
    fits_ultra_192gb: bool,
    fits_ultra_512gb: bool,
    recommended_min_tier: HardwareTier,

    pub fn evaluate(required_wired_bytes: u64) TierFitSummary {
        const fit16 = required_wired_bytes <= HardwareTier.m4_16gb.usableWiredBytes();
        const fit24 = required_wired_bytes <= HardwareTier.m4_24gb.usableWiredBytes();
        const fit48 = required_wired_bytes <= HardwareTier.pro_48gb.usableWiredBytes();
        const fit64 = required_wired_bytes <= HardwareTier.max_64gb.usableWiredBytes();
        const fit128 = required_wired_bytes <= HardwareTier.max_128gb.usableWiredBytes();
        const fit192 = required_wired_bytes <= HardwareTier.ultra_192gb.usableWiredBytes();
        const fit512 = required_wired_bytes <= HardwareTier.ultra_512gb.usableWiredBytes();
        const min_tier: HardwareTier = if (fit16)
            .m4_16gb
        else if (fit24)
            .m4_24gb
        else if (fit48)
            .pro_48gb
        else if (fit64)
            .max_64gb
        else if (fit128)
            .max_128gb
        else if (fit192)
            .ultra_192gb
        else if (fit512)
            .ultra_512gb
        else
            .cluster_required;
        return .{
            .fits_m4_16gb = fit16,
            .fits_m4_24gb = fit24,
            .fits_pro_48gb = fit48,
            .fits_max_64gb = fit64,
            .fits_max_128gb = fit128,
            .fits_ultra_192gb = fit192,
            .fits_ultra_512gb = fit512,
            .recommended_min_tier = min_tier,
        };
    }
};

pub const ArchAxesSummary = struct {
    arch_name: [48]u8 = @splat(0),
    arch_len: u8 = 0,
    attention_axis: [24]u8 = @splat(0),
    attention_len: u8 = 0,
    ffn_axis: [24]u8 = @splat(0),
    ffn_len: u8 = 0,
    kv_axis: [24]u8 = @splat(0),
    kv_len: u8 = 0,
    sparse_axis: [24]u8 = @splat(0),
    sparse_len: u8 = 0,
    hidden_size: u32 = 0,
    num_hidden_layers: u32 = 0,
    num_attention_heads: u32 = 0,
    num_key_value_heads: u32 = 0,
    head_dim: u32 = 0,
    vocab_size: u32 = 0,
    max_position_embeddings: u32 = 0,
    intermediate_size: u32 = 0,
    moe_intermediate_size: u32 = 0,
    num_experts: u32 = 0,
    num_experts_per_tok: u32 = 0,
    tie_word_embeddings: bool = false,
    has_moe: bool = false,
    has_shared_expert: bool = false,
    has_gdn_hybrid: bool = false,
    has_mla: bool = false,
    has_sparse_ple_engram: bool = false,
    has_loop_adapter: bool = false,
    has_vision: bool = false,
    lora_rank: u32 = 0,

    pub fn setArch(self: *ArchAxesSummary, s: []const u8) void {
        const n: usize = @min(s.len, self.arch_name.len);
        @memset(&self.arch_name, 0);
        @memcpy(self.arch_name[0..n], s[0..n]);
        self.arch_len = @intCast(n);
    }

    pub fn archSlice(self: *const ArchAxesSummary) []const u8 {
        return self.arch_name[0..self.arch_len];
    }

    pub fn attentionSlice(self: *const ArchAxesSummary) []const u8 {
        return self.attention_axis[0..self.attention_len];
    }

    pub fn ffnSlice(self: *const ArchAxesSummary) []const u8 {
        return self.ffn_axis[0..self.ffn_len];
    }

    pub fn kvSlice(self: *const ArchAxesSummary) []const u8 {
        return self.kv_axis[0..self.kv_len];
    }

    pub fn sparseSlice(self: *const ArchAxesSummary) []const u8 {
        return self.sparse_axis[0..self.sparse_len];
    }
};

fn setFixedField(dst: []u8, len_out: *u8, src: []const u8) void {
    const n: usize = @min(src.len, dst.len);
    @memset(dst, 0);
    @memcpy(dst[0..n], src[0..n]);
    len_out.* = @intCast(n);
}

pub const AmfInspection = struct {
    path: []u8,
    model_id: [64]u8,
    model_id_len: u8,
    file_bytes: u64,
    header_crc32: u32,
    directory_crc32: u32,
    header_flags: u32,
    total_entries: u32,
    weight_tensors: u32,
    metadata_entries: u32,
    wired_weight_bytes: u64,
    cold_weight_bytes: u64,
    sparse_table_bytes: u64,
    metadata_blob_bytes: u64,
    estimated_runtime_kv_scratch_bytes: u64,
    estimated_total_wired_bytes: u64,
    effective_bits_per_weight: f64,
    sme2_panel64_tensors: u32,
    lut16_tensors: u32,
    hot_expert_tensors: u32,
    cold_expert_tensors: u32,
    sparse_table_tensors: u32,
    loop_adapter_tensors: u32,
    per_dtype: [DTYPE_COUNT]DTypeStats,
    config_meta: MetaBlobReport,
    tokenizer_meta: MetaBlobReport,
    adapter_config_meta: MetaBlobReport,
    config_source: MetadataSource,
    arch: ArchAxesSummary,
    tier_fit: TierFitSummary,

    pub fn deinit(self: *AmfInspection, allocator: std.mem.Allocator) void {
        if (self.path.len > 0) allocator.free(self.path);
        self.path = &.{};
    }

    pub fn modelIdSlice(self: *const AmfInspection) []const u8 {
        return self.model_id[0..self.model_id_len];
    }

    pub fn archSlice(self: *const AmfInspection) []const u8 {
        return self.arch.archSlice();
    }
};

pub const InspectError = ParseError || IoError || error{
    PathTooLong,
    DirectoryOpenFailed,
};

fn jsonU32Opt(obj: std.json.ObjectMap, key: []const u8) ?u32 {
    const val = obj.get(key) orelse return null;
    return switch (val) {
        .integer => |v| if (v >= 0 and v <= std.math.maxInt(u32)) @intCast(v) else null,
        else => null,
    };
}

fn jsonBoolOpt(obj: std.json.ObjectMap, key: []const u8) ?bool {
    const val = obj.get(key) orelse return null;
    return switch (val) {
        .bool => |b| b,
        else => null,
    };
}

fn jsonStrOpt(obj: std.json.ObjectMap, key: []const u8) ?[]const u8 {
    const val = obj.get(key) orelse return null;
    return switch (val) {
        .string => |s| s,
        else => null,
    };
}

fn populateArchFromJson(allocator: std.mem.Allocator, json_bytes: []const u8, arch: *ArchAxesSummary) error{OutOfMemory}!void {
    var parsed_json = std.json.parseFromSlice(std.json.Value, allocator, json_bytes, .{
        .ignore_unknown_fields = true,
    }) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return,
    };
    defer parsed_json.deinit();
    if (parsed_json.value != .object) return;
    const root = parsed_json.value.object;
    const obj = if (root.get("text_config")) |tc|
        (if (tc == .object) tc.object else root)
    else
        root;

    if (jsonStrOpt(obj, "model_type") orelse jsonStrOpt(root, "model_type") orelse jsonStrOpt(root, "peft_type")) |mt| {
        arch.setArch(mt);
    }
    if (jsonU32Opt(obj, "hidden_size")) |v| arch.hidden_size = v;
    if (jsonU32Opt(obj, "num_hidden_layers")) |v| arch.num_hidden_layers = v;
    if (jsonU32Opt(obj, "num_attention_heads")) |v| arch.num_attention_heads = v;
    if (jsonU32Opt(obj, "num_key_value_heads")) |v| arch.num_key_value_heads = v;
    if (jsonU32Opt(obj, "head_dim")) |v| arch.head_dim = v;
    if (jsonU32Opt(obj, "vocab_size")) |v| arch.vocab_size = v;
    if (jsonU32Opt(obj, "max_position_embeddings") orelse
        jsonU32Opt(obj, "max_seq_len") orelse
        jsonU32Opt(obj, "seq_length") orelse
        jsonU32Opt(obj, "n_ctx") orelse
        jsonU32Opt(root, "max_position_embeddings")) |v|
    {
        arch.max_position_embeddings = v;
    }
    if (root.get("vision_config")) |vc| {
        if (vc != .null) arch.has_vision = true;
    }
    if (jsonU32Opt(obj, "intermediate_size")) |v| arch.intermediate_size = v;
    if (jsonU32Opt(obj, "moe_intermediate_size")) |v| arch.moe_intermediate_size = v;
    if (jsonU32Opt(obj, "num_experts") orelse jsonU32Opt(obj, "n_routed_experts")) |v| arch.num_experts = v;
    if (jsonU32Opt(obj, "num_experts_per_tok")) |v| arch.num_experts_per_tok = v;
    if (jsonBoolOpt(obj, "tie_word_embeddings") orelse jsonBoolOpt(root, "tie_word_embeddings")) |b| {
        arch.tie_word_embeddings = b;
    }
    if (jsonU32Opt(root, "r")) |r_val| arch.lora_rank = r_val;

    if (arch.head_dim == 0 and arch.hidden_size > 0 and arch.num_attention_heads > 0) {
        arch.head_dim = arch.hidden_size / arch.num_attention_heads;
    }
    if (arch.num_experts > 0) arch.has_moe = true;
    if ((jsonU32Opt(obj, "shared_expert_intermediate_size") orelse 0) > 0 or
        (jsonU32Opt(obj, "n_shared_experts") orelse 0) > 0)
    {
        arch.has_shared_expert = true;
    }
    if ((jsonU32Opt(obj, "kv_lora_rank") orelse 0) > 0) arch.has_mla = true;
    if ((jsonU32Opt(obj, "linear_key_head_dim") orelse 0) > 0 or
        (jsonU32Opt(obj, "linear_num_value_heads") orelse 0) > 0)
    {
        arch.has_gdn_hybrid = true;
    }

    if (root.get("operator_axes") orelse obj.get("operator_axes")) |axes_val| {
        if (axes_val == .object) {
            const ax = axes_val.object;
            if (jsonStrOpt(ax, "attention")) |s| {
                setFixedField(&arch.attention_axis, &arch.attention_len, s);
                if (std.mem.indexOf(u8, s, "mla") != null) arch.has_mla = true;
                if (std.mem.indexOf(u8, s, "gdn") != null or std.mem.indexOf(u8, s, "hybrid") != null) {
                    arch.has_gdn_hybrid = true;
                }
            }
            if (jsonStrOpt(ax, "ffn")) |s| {
                setFixedField(&arch.ffn_axis, &arch.ffn_len, s);
                if (std.mem.indexOf(u8, s, "moe") != null) arch.has_moe = true;
                if (std.mem.indexOf(u8, s, "shared") != null) arch.has_shared_expert = true;
            }
            if (jsonStrOpt(ax, "kv_layout") orelse jsonStrOpt(ax, "kv")) |s| {
                setFixedField(&arch.kv_axis, &arch.kv_len, s);
            }
            if (jsonStrOpt(ax, "sparse_memory") orelse jsonStrOpt(ax, "sparse")) |s| {
                setFixedField(&arch.sparse_axis, &arch.sparse_len, s);
                if (!std.mem.eql(u8, s, "none")) arch.has_sparse_ple_engram = true;
            }
        }
    }
}

fn inferMissingArchAxesFromEntries(entries: []const TensorEntry, arch: *ArchAxesSummary) void {
    var max_layer_plus_one: u32 = 0;
    var max_expert_plus_one: u32 = 0;
    var q_out_dim: u32 = 0;
    var k_out_dim: u32 = 0;
    var seen_lora_pair = false;

    for (entries) |*e| {
        const kind = std.enums.fromInt(TensorKind, e.kind) orelse continue;
        if (kind == .metadata) continue;
        const nm = e.nameSlice();

        if (e.layer != NO_LAYER) {
            max_layer_plus_one = @max(max_layer_plus_one, @as(u32, e.layer) + 1);
        }
        if (e.expert != NO_EXPERT) {
            max_expert_plus_one = @max(max_expert_plus_one, @as(u32, e.expert) + 1);
        }
        if (kind == .hot_expert or kind == .cold_expert or kind == .router) arch.has_moe = true;
        if (kind == .shared_expert) arch.has_shared_expert = true;
        if (kind == .sparse_table) arch.has_sparse_ple_engram = true;
        if (kind == .loop_adapter) {
            if (std.mem.endsWith(u8, nm, ".lora_a") and e.rank == 2) {
                seen_lora_pair = true;
                if (arch.lora_rank == 0) arch.lora_rank = e.shape[0];
                if (arch.hidden_size == 0) arch.hidden_size = e.shape[1];
            }
            arch.has_loop_adapter = true;
        }
        if (std.mem.indexOf(u8, nm, "linear_attn") != null or std.mem.indexOf(u8, nm, "gdn") != null) {
            arch.has_gdn_hybrid = true;
        }
        if (std.mem.indexOf(u8, nm, "kv_a_proj") != null or std.mem.indexOf(u8, nm, "q_a_proj") != null) {
            arch.has_mla = true;
        }
        if (std.mem.startsWith(u8, nm, "vision.") or
            std.mem.startsWith(u8, nm, "visual.") or
            std.mem.startsWith(u8, nm, "vision_tower."))
        {
            arch.has_vision = true;
        }

        if ((std.mem.eql(u8, nm, "embed_tokens") or std.mem.eql(u8, nm, "embed_tokens.weight")) and e.rank == 2) {
            if (arch.vocab_size == 0) arch.vocab_size = e.shape[0];
            if (arch.hidden_size == 0) arch.hidden_size = e.shape[1];
        } else if (std.mem.eql(u8, nm, "norm.weight") and e.rank == 1) {
            if (arch.hidden_size == 0) arch.hidden_size = e.shape[0];
        } else if (std.mem.endsWith(u8, nm, "self_attn.q_proj") or std.mem.endsWith(u8, nm, "self_attn.q_proj.weight")) {
            if (e.rank == 2) {
                q_out_dim = @max(q_out_dim, e.shape[0]);
                if (arch.hidden_size == 0) arch.hidden_size = e.shape[1];
            }
        } else if (std.mem.endsWith(u8, nm, "self_attn.k_proj") or std.mem.endsWith(u8, nm, "self_attn.k_proj.weight")) {
            if (e.rank == 2) k_out_dim = @max(k_out_dim, e.shape[0]);
        } else if (std.mem.endsWith(u8, nm, "q_norm.weight") and e.rank == 1) {
            if (arch.head_dim == 0) arch.head_dim = e.shape[0];
        } else if (std.mem.endsWith(u8, nm, "mlp.gate_proj") or std.mem.endsWith(u8, nm, "mlp.gate_proj.weight")) {
            if (e.rank == 2 and arch.intermediate_size == 0) arch.intermediate_size = e.shape[0];
        } else if (std.mem.endsWith(u8, nm, ".gate_proj") and (kind == .hot_expert or kind == .cold_expert)) {
            if (e.rank == 2 and arch.moe_intermediate_size == 0) arch.moe_intermediate_size = e.shape[0];
        }
    }

    if (arch.num_hidden_layers == 0 and max_layer_plus_one > 0) arch.num_hidden_layers = max_layer_plus_one;
    if (arch.num_experts == 0 and max_expert_plus_one > 0) arch.num_experts = max_expert_plus_one;
    if (arch.head_dim > 0) {
        if (arch.num_attention_heads == 0 and q_out_dim > 0) arch.num_attention_heads = q_out_dim / arch.head_dim;
        if (arch.num_key_value_heads == 0 and k_out_dim > 0) arch.num_key_value_heads = k_out_dim / arch.head_dim;
    }

    if (arch.attention_len == 0) {
        const attn_lbl = if (arch.has_mla and arch.has_gdn_hybrid)
            "mla_gdn_hybrid"
        else if (arch.has_mla)
            "mla"
        else if (arch.has_gdn_hybrid)
            "gqa_gdn_hybrid"
        else
            "gqa";
        setFixedField(&arch.attention_axis, &arch.attention_len, attn_lbl);
    }
    if (arch.ffn_len == 0) {
        const ffn_lbl = if (arch.has_moe and arch.has_shared_expert)
            "moe_shared"
        else if (arch.has_moe)
            "moe"
        else
            "dense";
        setFixedField(&arch.ffn_axis, &arch.ffn_len, ffn_lbl);
    }
    if (arch.kv_len == 0) {
        const kv_lbl = if (arch.has_mla) "fp4_latent" else "rgba16f_q8";
        setFixedField(&arch.kv_axis, &arch.kv_len, kv_lbl);
    }
    if (arch.sparse_len == 0) {
        const sp_lbl = if (arch.has_sparse_ple_engram) "nvme_sparse_table" else "none";
        setFixedField(&arch.sparse_axis, &arch.sparse_len, sp_lbl);
    }
    if (arch.arch_len == 0) {
        const inferred = if (seen_lora_pair and arch.vocab_size == 0)
            "amlora"
        else if (arch.has_mla)
            "deepseek_v4_1"
        else if (arch.has_gdn_hybrid and arch.has_sparse_ple_engram)
            "qwen3_8"
        else if (arch.has_gdn_hybrid)
            "qwen3_next"
        else if (arch.has_moe)
            "qwen3_moe"
        else
            "qwen3_dense";
        arch.setArch(inferred);
    }
}

fn verifyMetaEntryReport(
    amf_file: *const AmfFile,
    allocator: std.mem.Allocator,
    meta_name: []const u8,
) (ParseError || IoError)!MetaBlobReport {
    const entry = amf_file.parsed.find(meta_name) orelse return .{};
    if ((try entry.tensorKind()) != .metadata) return error.InvalidTensorKind;

    const blob_len: usize = @intCast(entry.blob_bytes);
    const chunk_len: usize = @min(blob_len, VERIFY_CHUNK_BYTES);
    const scratch = try allocator.alignedAlloc(u8, .fromByteUnits(page.APPLE_PAGE_SIZE), chunk_len);
    defer allocator.free(scratch);

    var hasher = std.hash.Crc32.init();
    var done: usize = 0;
    while (done < blob_len) {
        const take: usize = @min(blob_len - done, chunk_len);
        try preadExact(amf_file.fd, scratch[0..take], entry.blob_offset + done);
        hasher.update(scratch[0..take]);
        done += take;
    }
    const valid = (hasher.final() == entry.blob_crc32);
    return .{
        .present = true,
        .data_bytes = entry.data_bytes,
        .blob_bytes = entry.blob_bytes,
        .crc32 = entry.blob_crc32,
        .crc_valid = valid,
    };
}

fn estimateRuntimeKvAndScratchBytes(arch: *const ArchAxesSummary) u64 {
    const default_ctx_tokens: u64 = 2048;
    const min_floor: u64 = 64 * 1024 * 1024;
    if (arch.num_hidden_layers == 0 or arch.hidden_size == 0) return min_floor;

    const layers: u64 = arch.num_hidden_layers;
    const h: u64 = arch.hidden_size;
    const kv_bytes: u64 = if (arch.has_mla)
        layers * default_ctx_tokens * 324
    else blk: {
        const kv_heads: u64 = if (arch.num_key_value_heads > 0) arch.num_key_value_heads else 4;
        const hd: u64 = if (arch.head_dim > 0) arch.head_dim else 128;
        // K + V in FP16 (`2 * 2 = 4` bytes per channel per token).
        break :blk layers * default_ctx_tokens * kv_heads * hd * 4;
    };
    const wide_chunk_tokens: u64 = 512;
    const scratch_bytes: u64 = wide_chunk_tokens * h * 16;
    const raw = kv_bytes + scratch_bytes;
    const aligned = ((raw + page.APPLE_PAGE_SIZE - 1) / page.APPLE_PAGE_SIZE) * page.APPLE_PAGE_SIZE;
    return @max(aligned, min_floor);
}

/// Header-only `.amf` inspector (`ADR-0022`, `W18`): reads page 0, the tensor
/// directory, and any tiny cold metadata JSON (`__meta.*` or sidecar `config.json`)
/// via `F_NOCACHE` `pread` without allocating a `WiredArena` or `mlock`ing any
/// model weights (`0` wired weight bytes).
pub fn inspectAmfPath(allocator: std.mem.Allocator, path: []const u8) InspectError!AmfInspection {
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_z = try toPosixPathZ(path, &path_buf);

    var file = try AmfFile.open(allocator, path_z);
    defer file.close(allocator);

    const owned_path = try allocator.dupe(u8, path);
    errdefer allocator.free(owned_path);

    var per_dtype: [DTYPE_COUNT]DTypeStats = @splat(.{});
    var weight_tensors: u32 = 0;
    var metadata_entries: u32 = 0;
    var wired_weight_bytes: u64 = 0;
    var cold_weight_bytes: u64 = 0;
    var sparse_table_bytes: u64 = 0;
    var metadata_blob_bytes: u64 = 0;
    var sme2_panel64_tensors: u32 = 0;
    var lut16_tensors: u32 = 0;
    var hot_expert_tensors: u32 = 0;
    var cold_expert_tensors: u32 = 0;
    var sparse_table_tensors: u32 = 0;
    var loop_adapter_tensors: u32 = 0;

    var weighted_bits_sum: f64 = 0.0;
    var total_weight_elems: f64 = 0.0;

    for (file.parsed.entries) |*entry| {
        const kind = try entry.tensorKind();
        const dt = try entry.dataType();
        const layout = try entry.tensorLayout();
        const pk = try entry.quantParamKind();

        if (kind == .metadata) {
            metadata_entries += 1;
            metadata_blob_bytes += entry.blob_bytes;
            continue;
        }

        weight_tensors += 1;
        if (layout == .sme2_panel64) sme2_panel64_tensors += 1;
        if (pk == .lut16_codebook) lut16_tensors += 1;
        switch (kind) {
            .hot_expert => hot_expert_tensors += 1,
            .cold_expert => cold_expert_tensors += 1,
            .sparse_table => {
                sparse_table_tensors += 1;
                sparse_table_bytes += entry.blob_bytes;
            },
            .loop_adapter => loop_adapter_tensors += 1,
            else => {},
        }

        const dt_idx: usize = @backingInt(dt);
        per_dtype[dt_idx].tensor_count += 1;
        per_dtype[dt_idx].data_bytes += entry.data_bytes;
        if (kind.isWired()) {
            wired_weight_bytes += entry.blob_bytes;
            per_dtype[dt_idx].wired_bytes += entry.blob_bytes;
        } else {
            cold_weight_bytes += entry.blob_bytes;
            per_dtype[dt_idx].cold_bytes += entry.blob_bytes;
        }

        var elems: u64 = 1;
        for (0..entry.rank) |d| {
            elems = std.math.mul(u64, elems, entry.shape[d]) catch 0;
        }
        if (elems > 0) {
            const elems_f: f64 = @floatFromInt(elems);
            const eff_bytes: f64 = @floatFromInt(entry.data_bytes + entry.params_bytes);
            weighted_bits_sum += eff_bytes * 8.0;
            total_weight_elems += elems_f;
        }
    }

    const config_meta = try verifyMetaEntryReport(&file, allocator, META_CONFIG_JSON);
    const tokenizer_meta = try verifyMetaEntryReport(&file, allocator, META_TOKENIZER_JSON);
    const adapter_config_meta = try verifyMetaEntryReport(&file, allocator, META_ADAPTER_CONFIG_JSON);

    var arch: ArchAxesSummary = .{};
    var config_source: MetadataSource = .none;

    if (config_meta.present and config_meta.crc_valid) {
        if (try file.readMetadataAlloc(allocator, META_CONFIG_JSON)) |cfg_bytes| {
            defer allocator.free(cfg_bytes);
            try populateArchFromJson(allocator, cfg_bytes, &arch);
            config_source = .embedded;
        }
    } else if (adapter_config_meta.present and adapter_config_meta.crc_valid) {
        if (try file.readMetadataAlloc(allocator, META_ADAPTER_CONFIG_JSON)) |acfg_bytes| {
            defer allocator.free(acfg_bytes);
            try populateArchFromJson(allocator, acfg_bytes, &arch);
            config_source = .embedded;
        }
    } else {
        const dir = std.fs.path.dirname(path) orelse ".";
        var sidecar_buf: [std.fs.max_path_bytes]u8 = undefined;
        const sc_path = std.fmt.bufPrint(&sidecar_buf, "{s}/config.json", .{dir}) catch return error.PathTooLong;
        const sc_opt: ?[]u8 = readRegularFileAlloc(allocator, sc_path) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.PathTooLong => return error.PathTooLong,
            error.OpenFailed, error.StatFailed, error.FileTooSmall, error.MetadataTooLarge, error.ReadFailed, error.ShortRead => null,
            else => return error.ReadFailed,
        };
        if (sc_opt) |sc_bytes| {
            defer allocator.free(sc_bytes);
            try populateArchFromJson(allocator, sc_bytes, &arch);
            config_source = .sidecar;
        }
    }

    inferMissingArchAxesFromEntries(file.parsed.entries, &arch);

    const est_kv_scratch = if (std.mem.eql(u8, arch.archSlice(), "amlora"))
        @as(u64, 0)
    else
        estimateRuntimeKvAndScratchBytes(&arch);
    const est_total_wired = wired_weight_bytes + est_kv_scratch;
    const eff_bpw: f64 = if (total_weight_elems > 0.0) weighted_bits_sum / total_weight_elems else 0.0;

    const mid_slice = file.parsed.modelId();
    var mid_buf: [64]u8 = @splat(0);
    @memcpy(mid_buf[0..mid_slice.len], mid_slice);

    return .{
        .path = owned_path,
        .model_id = mid_buf,
        .model_id_len = @intCast(mid_slice.len),
        .file_bytes = file.parsed.file_size,
        .header_crc32 = file.parsed.header.header_crc32,
        .directory_crc32 = file.parsed.header.directory_crc32,
        .header_flags = file.parsed.header.flags,
        .total_entries = @intCast(file.parsed.entries.len),
        .weight_tensors = weight_tensors,
        .metadata_entries = metadata_entries,
        .wired_weight_bytes = wired_weight_bytes,
        .cold_weight_bytes = cold_weight_bytes,
        .sparse_table_bytes = sparse_table_bytes,
        .metadata_blob_bytes = metadata_blob_bytes,
        .estimated_runtime_kv_scratch_bytes = est_kv_scratch,
        .estimated_total_wired_bytes = est_total_wired,
        .effective_bits_per_weight = eff_bpw,
        .sme2_panel64_tensors = sme2_panel64_tensors,
        .lut16_tensors = lut16_tensors,
        .hot_expert_tensors = hot_expert_tensors,
        .cold_expert_tensors = cold_expert_tensors,
        .sparse_table_tensors = sparse_table_tensors,
        .loop_adapter_tensors = loop_adapter_tensors,
        .per_dtype = per_dtype,
        .config_meta = config_meta,
        .tokenizer_meta = tokenizer_meta,
        .adapter_config_meta = adapter_config_meta,
        .config_source = config_source,
        .arch = arch,
        .tier_fit = TierFitSummary.evaluate(est_total_wired),
    };
}

pub const ScanOptions = struct {
    /// Also scan immediate 1-level bundle subdirectories (`<dir>/<bundle>/*.amf`).
    scan_bundle_subdirs: bool = true,
    /// Include `.amlora` adapter bundles alongside `.amf` model files.
    include_amlora: bool = false,
};

fn hasMatchingAmfExtension(name: []const u8, include_amlora: bool) bool {
    if (std.mem.endsWith(u8, name, ".partial")) return false;
    if (std.mem.endsWith(u8, name, ".amf")) return true;
    if (include_amlora and std.mem.endsWith(u8, name, ".amlora")) return true;
    return false;
}

fn collectAmfPathsInDir(
    allocator: std.mem.Allocator,
    dir_path: []const u8,
    allow_subdirs: bool,
    include_amlora: bool,
    out_paths: *std.ArrayList([]u8),
) InspectError!void {
    var dir_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir_z = try toPosixPathZ(dir_path, &dir_buf);
    const dp = std.c.opendir(dir_z.ptr) orelse return error.DirectoryOpenFailed;
    defer _ = std.c.closedir(dp);

    while (std.c.readdir(dp)) |ent| {
        const name = ent.name[0..ent.namlen];
        if (name.len == 0 or name[0] == '.') continue;

        var child_buf: [std.fs.max_path_bytes]u8 = undefined;
        const sep = if (std.mem.endsWith(u8, dir_path, "/")) "" else "/";
        const child_path = std.fmt.bufPrint(&child_buf, "{s}{s}{s}", .{ dir_path, sep, name }) catch {
            return error.PathTooLong;
        };

        var is_reg = (ent.type == std.c.DT.REG);
        var is_dir = (ent.type == std.c.DT.DIR);
        if (ent.type == std.c.DT.UNKNOWN or ent.type == std.c.DT.LNK) {
            var st_path_buf: [std.fs.max_path_bytes]u8 = undefined;
            const st_z = try toPosixPathZ(child_path, &st_path_buf);
            var st: std.c.Stat = undefined;
            if (std.c.fstatat(std.c.AT.FDCWD, st_z.ptr, &st, 0) == 0) {
                const ftype = st.mode & std.c.S.IFMT;
                is_reg = (ftype == std.c.S.IFREG);
                is_dir = (ftype == std.c.S.IFDIR);
            }
        }

        if (is_reg and hasMatchingAmfExtension(name, include_amlora)) {
            const duped = try allocator.dupe(u8, child_path);
            errdefer allocator.free(duped);
            try out_paths.append(allocator, duped);
        } else if (is_dir and allow_subdirs) {
            try collectAmfPathsInDir(allocator, child_path, false, include_amlora, out_paths);
        }
    }
}

fn pathLessThan(_: void, a: []u8, b: []u8) bool {
    return std.mem.lessThan(u8, a, b);
}

/// Scans `dir_path` (and immediate bundle subdirectories by default) for `.amf`
/// models, returning deterministic path-sorted header-only `AmfInspection` records
/// with zero weight `mlock` (`ADR-0022`, `W18`).
pub fn scanAmfDirectoryWithOptions(
    allocator: std.mem.Allocator,
    dir_path: []const u8,
    opts: ScanOptions,
) InspectError![]AmfInspection {
    var paths: std.ArrayList([]u8) = .empty;
    defer {
        for (paths.items) |p| allocator.free(p);
        paths.deinit(allocator);
    }

    try collectAmfPathsInDir(allocator, dir_path, opts.scan_bundle_subdirs, opts.include_amlora, &paths);
    std.mem.sort([]u8, paths.items, {}, pathLessThan);

    var results: std.ArrayList(AmfInspection) = .empty;
    errdefer {
        for (results.items) |*item| item.deinit(allocator);
        results.deinit(allocator);
    }

    for (paths.items) |p| {
        const info = try inspectAmfPath(allocator, p);
        try results.append(allocator, info);
    }
    return results.toOwnedSlice(allocator);
}

pub fn scanAmfDirectory(allocator: std.mem.Allocator, dir_path: []const u8) InspectError![]AmfInspection {
    return scanAmfDirectoryWithOptions(allocator, dir_path, .{});
}

pub fn freeAmfInspections(allocator: std.mem.Allocator, items: []AmfInspection) void {
    for (items) |*item| item.deinit(allocator);
    allocator.free(items);
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
    e.dtype = DTYPE_COUNT; // unknown dtype past gguf_q2_0
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

test "W18 single-file embedded metadata, sidecar override precedence, and zero-mlock inspect/ls" {
    const gpa = std.testing.allocator;

    // Build a base .amf image with embed_tokens + q_proj + norm.weight (no embedded metadata).
    var base_entries: [3]TensorEntry = undefined;
    base_entries[0] = TensorEntry.zeroed("embed_tokens", .embedding, .f16);
    base_entries[0].rank = 2;
    base_entries[0].shape = .{ 128, 64, 0, 0 };
    base_entries[0].data_bytes = 128 * 64 * 2;

    const Q4 = quant.QuantOps(.{ .bits = .q4, .group_size = 64 });
    base_entries[1] = TensorEntry.zeroed("layers.0.self_attn.q_proj", .dense, .q4_affine);
    base_entries[1].quant_bits = 4;
    base_entries[1].rank = 2;
    base_entries[1].group_size = 64;
    base_entries[1].layer = 0;
    base_entries[1].shape = .{ 64, 64, 0, 0 };
    base_entries[1].data_bytes = try Q4.packedWeightBytes(64, 64);
    base_entries[1].params_offset = align64(@intCast(base_entries[1].data_bytes));
    base_entries[1].params_bytes = try Q4.scaleBiasBytes(64, 64);

    base_entries[2] = TensorEntry.zeroed("norm.weight", .norm, .f16);
    base_entries[2].rank = 1;
    base_entries[2].shape = .{ 64, 0, 0, 0 };
    base_entries[2].data_bytes = 64 * 2;

    const base_image = try buildTestAmfMulti(gpa, &base_entries);
    defer gpa.free(base_image);

    // Odd-length UTF-8 JSON payloads verify exact unpadded `data_bytes` preservation.
    const embedded_cfg_json =
        \\{"model_type":"qwen3","hidden_size":64,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":2,"head_dim":32,"vocab_size":128,"intermediate_size":128,"tie_word_embeddings":true}
    ;
    const embedded_tok_json =
        \\{"version":"1.0","model":{"type":"BPE","vocab":{"a":0},"merges":[]}}
    ;
    try std.testing.expect((embedded_cfg_json.len % 4) != 0 or (embedded_tok_json.len % 4) == 0);

    const single_file_image = try embedMetadataIntoAmfBytes(gpa, base_image, &.{
        .{ .name = META_CONFIG_JSON, .payload = embedded_cfg_json },
        .{ .name = META_TOKENIZER_JSON, .payload = embedded_tok_json },
    });
    defer gpa.free(single_file_image);

    var parsed_sf = try Parsed.parseBytes(gpa, single_file_image);
    defer parsed_sf.deinit(gpa);
    try parsed_sf.verifyAllBlobsFromBytes(single_file_image);

    try std.testing.expect((parsed_sf.header.flags & FLAG_EMBEDDED_METADATA) != 0);
    try std.testing.expect(parsed_sf.hasEmbeddedConfig());
    try std.testing.expect(parsed_sf.hasEmbeddedTokenizer());
    // Embedded metadata must consume ZERO wired bytes and ZERO cold weight bytes!
    try std.testing.expectEqual(@as(u64, 3 * page.APPLE_PAGE_SIZE), parsed_sf.totalWiredBytes());
    try std.testing.expectEqual(@as(u64, 0), parsed_sf.totalColdBytes());
    try std.testing.expectEqual(@as(u64, 2 * page.APPLE_PAGE_SIZE), parsed_sf.totalMetadataBytes());

    const got_cfg = (try parsed_sf.extractMetadataFromBytes(single_file_image, META_CONFIG_JSON)) orelse return error.TestUnexpectedResult;
    const got_tok = (try parsed_sf.extractMetadataFromBytes(single_file_image, META_TOKENIZER_JSON)) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings(embedded_cfg_json, got_cfg);
    try std.testing.expectEqualStrings(embedded_tok_json, got_tok);

    // Verify `loadWiredFromBytes` skips `__meta.*` entries cleanly.
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
    const wired_ptrs = try parsed_sf.loadWiredFromBytes(gpa, single_file_image, &arena);
    defer gpa.free(wired_ptrs);
    try std.testing.expectEqual(@as(usize, 5), wired_ptrs.len);
    try std.testing.expect(wired_ptrs[0] != null and wired_ptrs[1] != null and wired_ptrs[2] != null);
    try std.testing.expect(wired_ptrs[3] == null and wired_ptrs[4] == null);

    // Write both single-file model and legacy sidecar bundle to disk to test resolution & ls/inspect.
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "single.amf", .data = single_file_image });
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "override_cfg.json", .data = "{\"model_type\":\"override\"}" });
    try tmp.dir.createDirPath(std.testing.io, "legacy_bundle");
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "legacy_bundle/model.amf", .data = base_image });
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "legacy_bundle/config.json", .data = embedded_cfg_json });
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "legacy_bundle/tokenizer.json", .data = embedded_tok_json });

    var sf_path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const sf_raw = try std.fmt.bufPrint(sf_path_buf[0 .. sf_path_buf.len - 1], ".zig-cache/tmp/{s}/single.amf", .{tmp.sub_path});
    sf_path_buf[sf_raw.len] = 0;
    const sf_path_z: [:0]const u8 = sf_path_buf[0..sf_raw.len :0];

    var ov_path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const ov_path = try std.fmt.bufPrint(&ov_path_buf, ".zig-cache/tmp/{s}/override_cfg.json", .{tmp.sub_path});

    var sf_file = try AmfFile.open(gpa, sf_path_z);
    defer sf_file.close(gpa);

    // 1. Default resolution on single-file .amf uses embedded metadata.
    var res_emb = try resolveConfigJsonAlloc(gpa, &sf_file, sf_raw, null);
    defer res_emb.deinit(gpa);
    try std.testing.expectEqual(MetadataSource.embedded, res_emb.source);
    try std.testing.expectEqualStrings(embedded_cfg_json, res_emb.bytes);

    // 2. Explicit `--config` override beats embedded metadata.
    var res_ov = try resolveConfigJsonAlloc(gpa, &sf_file, sf_raw, ov_path);
    defer res_ov.deinit(gpa);
    try std.testing.expectEqual(MetadataSource.explicit_override, res_ov.source);
    try std.testing.expectEqualStrings("{\"model_type\":\"override\"}", res_ov.bytes);

    // 3. Legacy `.amf` without embedded metadata resolves sidecar `config.json` & `tokenizer.json`.
    var leg_path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const leg_raw = try std.fmt.bufPrint(leg_path_buf[0 .. leg_path_buf.len - 1], ".zig-cache/tmp/{s}/legacy_bundle/model.amf", .{tmp.sub_path});
    leg_path_buf[leg_raw.len] = 0;
    var leg_file = try AmfFile.open(gpa, leg_path_buf[0..leg_raw.len :0]);
    defer leg_file.close(gpa);
    var res_sc = try resolveTokenizerJsonAlloc(gpa, &leg_file, leg_raw, null);
    defer res_sc.deinit(gpa);
    try std.testing.expectEqual(MetadataSource.sidecar, res_sc.source);
    try std.testing.expectEqualStrings(embedded_tok_json, res_sc.bytes);

    // 4. Header-only `inspectAmfPath` & `scanAmfDirectory` (zero weight mlock).
    var dir_root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir_root = try std.fmt.bufPrint(&dir_root_buf, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const list = try scanAmfDirectory(gpa, dir_root);
    defer freeAmfInspections(gpa, list);
    try std.testing.expectEqual(@as(usize, 2), list.len);

    // Sorted lexicographically: `legacy_bundle/model.amf` then `single.amf`.
    const leg_info = &list[0];
    const sf_info = &list[1];
    try std.testing.expect(std.mem.endsWith(u8, leg_info.path, "legacy_bundle/model.amf"));
    try std.testing.expect(!leg_info.config_meta.present);
    try std.testing.expectEqual(MetadataSource.sidecar, leg_info.config_source);

    try std.testing.expect(std.mem.endsWith(u8, sf_info.path, "single.amf"));
    try std.testing.expect(sf_info.config_meta.present and sf_info.config_meta.crc_valid);
    try std.testing.expect(sf_info.tokenizer_meta.present and sf_info.tokenizer_meta.crc_valid);
    try std.testing.expectEqual(MetadataSource.embedded, sf_info.config_source);
    try std.testing.expectEqual(@as(u32, 3), sf_info.weight_tensors);
    try std.testing.expectEqual(@as(u32, 2), sf_info.metadata_entries);
    try std.testing.expectEqual(@as(u32, 2), sf_info.per_dtype[@backingInt(DType.f16)].tensor_count);
    try std.testing.expectEqual(@as(u32, 1), sf_info.per_dtype[@backingInt(DType.q4_affine)].tensor_count);
    try std.testing.expectEqual(@as(u32, 64), sf_info.arch.hidden_size);
    try std.testing.expectEqual(@as(u32, 128), sf_info.arch.vocab_size);
    try std.testing.expect(sf_info.tier_fit.fits_m4_24gb);
    try std.testing.expectEqual(HardwareTier.m4_16gb, sf_info.tier_fit.recommended_min_tier);

    // 5. Corrupting embedded metadata byte triggers `BlobChecksumMismatch` on extraction.
    const corrupted = try gpa.dupe(u8, single_file_image);
    defer gpa.free(corrupted);
    const meta_e = parsed_sf.find(META_CONFIG_JSON).?;
    corrupted[@intCast(meta_e.blob_offset)] ^= 0xFF;
    try std.testing.expectError(error.BlobChecksumMismatch, parsed_sf.extractMetadataFromBytes(corrupted, META_CONFIG_JSON));
}

test "M8.1 DType enum discriminants 0..37, generalized affine group_size, FP8/MXFP8/NVFP4, GGUF blocks, and chat_template metadata" {
    const gpa = std.testing.allocator;
    try std.testing.expectEqual(@as(usize, 38), DTYPE_COUNT);
    try std.testing.expectEqual(@as(u8, 7), @backingInt(DType.q5_affine));
    try std.testing.expectEqual(@as(u8, 10), @backingInt(DType.fp8_e4m3_block));
    try std.testing.expectEqual(@as(u8, 13), @backingInt(DType.mxfp8_g32));
    try std.testing.expectEqual(@as(u8, 14), @backingInt(DType.nvfp4_g16));
    try std.testing.expectEqual(@as(u8, 15), @backingInt(DType.gguf_q4_0));
    try std.testing.expectEqual(@as(u8, 36), @backingInt(DType.gguf_mxfp4));
    try std.testing.expectEqual(@as(u8, 37), @backingInt(DType.gguf_q2_0));

    // Affine q2..q8 across group_size in {32, 64, 128}
    inline for (.{
        .{ .dt = DType.q2_affine, .bits = @as(u8, 2) },
        .{ .dt = DType.q5_affine, .bits = @as(u8, 5) },
        .{ .dt = DType.q6_affine, .bits = @as(u8, 6) },
        .{ .dt = DType.q8_affine, .bits = @as(u8, 8) },
    }) |case| {
        inline for (.{ @as(u16, 32), @as(u16, 64), @as(u16, 128) }) |gs| {
            const n: u64 = 4;
            const k: u64 = 128;
            const data_len: usize = @intCast(n * ((k * case.bits) / 8));
            const params_off: usize = align64(data_len);
            const params_len: usize = @intCast(n * (k / gs) * @sizeOf(quant.ScaleBias));
            const blob = try gpa.alloc(u8, params_off + params_len);
            defer gpa.free(blob);
            @memset(blob, 0x11);

            var e = TensorEntry.zeroed("layers.0.self_attn.q_proj", .dense, case.dt);
            e.quant_bits = case.bits;
            e.rank = 2;
            e.group_size = gs;
            e.shape = .{ @intCast(n), @intCast(k), 0, 0 };
            e.data_bytes = data_len;
            e.params_offset = params_off;
            e.params_bytes = params_len;
            const img = try buildTestAmf(gpa, e, blob);
            defer gpa.free(img);
            var parsed = try Parsed.parseBytes(gpa, img);
            defer parsed.deinit(gpa);
            try std.testing.expectEqual(case.dt, try parsed.find("layers.0.self_attn.q_proj").?.dataType());
        }
    }

    // NVFP4 (N=1, K=32 -> 2 micro-scale bytes padded to 4 + 4B outer f32 = 8B params)
    {
        var e = TensorEntry.zeroed("layers.0.mlp.experts.0.down_proj", .cold_expert, .nvfp4_g16);
        e.quant_bits = 4;
        e.rank = 2;
        e.group_size = 16;
        e.shape = .{ 1, 32, 0, 0 };
        e.data_bytes = 16;
        e.params_offset = 64;
        e.params_bytes = 8;
        const blob = try gpa.alloc(u8, 72);
        defer gpa.free(blob);
        @memset(blob, 0);
        const img = try buildTestAmf(gpa, e, blob);
        defer gpa.free(img);
        var parsed = try Parsed.parseBytes(gpa, img);
        defer parsed.deinit(gpa);
        try std.testing.expectEqual(DType.nvfp4_g16, try parsed.find("layers.0.mlp.experts.0.down_proj").?.dataType());
    }

    // GGUF raw block entry validation (`gguf_q2_0`)
    {
        var e = TensorEntry.zeroed("layers.0.mlp.experts.0.down_proj", .hot_expert, .gguf_q2_0);
        e.quant_bits = 2;
        e.rank = 2;
        e.group_size = 64;
        e.shape = .{ 2, 128, 0, 0 };
        e.data_bytes = 2 * 2 * 18;
        const blob = try gpa.alloc(u8, 72);
        defer gpa.free(blob);
        @memset(blob, 0);
        const img = try buildTestAmf(gpa, e, blob);
        defer gpa.free(img);
        var parsed = try Parsed.parseBytes(gpa, img);
        defer parsed.deinit(gpa);
        try std.testing.expectEqual(DType.gguf_q2_0, try parsed.find("layers.0.mlp.experts.0.down_proj").?.dataType());
    }

    // Chat template metadata (`CHAT_TEMPLATE_METADATA_NAME` + legacy alias)
    {
        var norm_e = TensorEntry.zeroed("norm.weight", .norm, .f16);
        norm_e.rank = 1;
        norm_e.shape = .{ 8, 0, 0, 0 };
        norm_e.data_bytes = 16;
        var norm_bytes: [16]u8 = @splat(0);
        const base = try buildTestAmf(gpa, norm_e, &norm_bytes);
        defer gpa.free(base);

        const spec_json = "{\"schema_version\":1,\"dialect\":\"inkling2\"}";
        const with_ct = try embedMetadataIntoAmfBytes(gpa, base, &.{
            .{ .name = CHAT_TEMPLATE_METADATA_NAME, .payload = spec_json },
            .{ .name = LEGACY_CHAT_TEMPLATE_METADATA_NAME, .payload = spec_json },
        });
        defer gpa.free(with_ct);
        var parsed = try Parsed.parseBytes(gpa, with_ct);
        defer parsed.deinit(gpa);
        try std.testing.expect(parsed.hasEmbeddedChatTemplate());
        const got = (try parsed.extractMetadataFromBytes(with_ct, CHAT_TEMPLATE_METADATA_NAME)).?;
        try std.testing.expectEqualStrings(spec_json, got);
        const got_legacy = (try parsed.extractMetadataFromBytes(with_ct, LEGACY_CHAT_TEMPLATE_METADATA_NAME)).?;
        try std.testing.expectEqualStrings(spec_json, got_legacy);
    }
}

test "C28 populateArchFromJson and inspectAmfPath extract max_position_embeddings and has_vision" {
    const gpa = std.testing.allocator;

    // 1. Top-level max_position_embeddings + vision_config.
    {
        var arch: ArchAxesSummary = .{};
        try populateArchFromJson(
            gpa,
            "{\"model_type\":\"qwen3_vl\",\"vocab_size\":151936,\"max_position_embeddings\":131072,\"vision_config\":{\"hidden_size\":1152}}",
            &arch,
        );
        try std.testing.expectEqualStrings("qwen3_vl", arch.archSlice());
        try std.testing.expectEqual(@as(u32, 131072), arch.max_position_embeddings);
        try std.testing.expect(arch.has_vision);
    }

    // 2. Nested text_config with fallback keys (n_ctx / max_seq_len) and top-level max_position_embeddings fallback.
    {
        var arch_nested: ArchAxesSummary = .{};
        try populateArchFromJson(
            gpa,
            "{\"model_type\":\"gemma3\",\"max_position_embeddings\":65536,\"text_config\":{\"model_type\":\"gemma3_text\",\"vocab_size\":262144,\"max_position_embeddings\":32768}}",
            &arch_nested,
        );
        try std.testing.expectEqual(@as(u32, 32768), arch_nested.max_position_embeddings);
        try std.testing.expect(!arch_nested.has_vision);

        var arch_fallback: ArchAxesSummary = .{};
        try populateArchFromJson(
            gpa,
            "{\"max_position_embeddings\":40960,\"text_config\":{\"model_type\":\"llama\",\"vocab_size\":128256},\"vision_config\":null}",
            &arch_fallback,
        );
        try std.testing.expectEqual(@as(u32, 40960), arch_fallback.max_position_embeddings);
        try std.testing.expect(!arch_fallback.has_vision);

        var arch_nctx: ArchAxesSummary = .{};
        try populateArchFromJson(gpa, "{\"text_config\":{\"n_ctx\":8192}}", &arch_nctx);
        try std.testing.expectEqual(@as(u32, 8192), arch_nctx.max_position_embeddings);
    }

    // 3. inspectAmfPath with embedded config.json + vision.* tensor prefix inference.
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var e = TensorEntry.zeroed("vision.blocks.0.attn.qkv.weight", .dense, .f16);
    e.rank = 2;
    e.shape = .{ 4, 4, 0, 0 };
    e.data_bytes = 32;
    var wbytes: [32]u8 = @splat(0);
    const base = try buildTestAmf(gpa, e, &wbytes);
    defer gpa.free(base);

    const cfg_json = "{\"model_type\":\"qwen2_5_vl\",\"text_config\":{\"vocab_size\":152064,\"max_seq_len\":65536}}";
    const img = try embedMetadataIntoAmfBytes(gpa, base, &.{
        .{ .name = META_CONFIG_JSON, .payload = cfg_json },
    });
    defer gpa.free(img);

    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "vision_model.amf", .data = img });

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const amf_path = try std.fmt.bufPrint(&path_buf, ".zig-cache/tmp/{s}/vision_model.amf", .{tmp.sub_path});
    var insp = try inspectAmfPath(gpa, amf_path);
    defer insp.deinit(gpa);

    try std.testing.expectEqualStrings("qwen2_5_vl", insp.archSlice());
    try std.testing.expectEqual(@as(u32, 152064), insp.arch.vocab_size);
    try std.testing.expectEqual(@as(u32, 65536), insp.arch.max_position_embeddings);
    try std.testing.expect(insp.arch.has_vision);
}
