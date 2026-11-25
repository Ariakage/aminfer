//! Native Zig Qwen2 / Qwen3 / Qwen3.5 byte-level BPE tokenizer
//! (status row W5, docs/architecture/runtime-pipeline.md).
//!
//! Startup parses HuggingFace `tokenizer.json` into flat compact tables;
//! `decode`, `tokenBytes`, and `encodeWithScratch` perform zero dynamic heap
//! allocation on the inference hot path. Zero Python at runtime.

const std = @import("std");
pub const chat_template = @import("chat_template.zig");
const paths = @import("../platform/paths.zig");

pub const UNICODE_TO_BYTE_LEN: usize = 0x144;

/// Upper bound for O(1) dense token-ID array indexing; IDs >= DENSE_LIMIT
/// spill into a sparse map so large synthetic IDs never blow up memory.
const DENSE_LIMIT: usize = 512_000;

/// Stack fast-path size for BPE chunks so normal `encode` calls perform zero
/// heap allocations even without an explicit `EncodeScratch`.
const STACK_CHUNK_CAP: usize = 128;

const NO_NODE: u32 = std.math.maxInt(u32);

pub const ByteLevelError = error{
    InvalidUtf8,
    InvalidByteLevelCodepoint,
    BufferTooSmall,
};

pub const LoadError = error{
    InvalidJson,
    MissingVocab,
    InvalidVocabEntry,
    InvalidMerge,
    InvalidAddedToken,
    TokenTooLong,
    OutOfMemory,
};

pub const EncodeError = error{
    UnknownByte,
    BufferTooSmall,
    ScratchTooSmall,
    OutOfMemory,
};

pub const DecodeError = error{
    UnknownTokenId,
    BufferTooSmall,
};

/// Bijective GPT-2 / Qwen byte-to-Unicode codepoint table.
/// Printable bytes (`'!'..'~'`, `0xA1..0xAC`, `0xAE..0xFF`) map to themselves;
/// the remaining 68 bytes map in ascending order to `U+0100..U+0143`.
pub const BYTE_TO_UNICODE: [256]u21 = blk: {
    var table: [256]u21 = undefined;
    var next_cp: u21 = 256;
    for (0..256) |b| {
        const direct = (b >= '!' and b <= '~') or (b >= 0xA1 and b <= 0xAC) or (b >= 0xAE and b <= 0xFF);
        if (direct) {
            table[b] = @intCast(b);
        } else {
            table[b] = next_cp;
            next_cp += 1;
        }
    }
    if (next_cp != UNICODE_TO_BYTE_LEN) @compileError("unexpected byte-to-unicode table size");
    break :blk table;
};

/// Inverse of `BYTE_TO_UNICODE`, mapping `U+0021..U+0143` back to raw bytes.
pub const UNICODE_TO_BYTE: [UNICODE_TO_BYTE_LEN]?u8 = blk: {
    var inv: [UNICODE_TO_BYTE_LEN]?u8 = @splat(null);
    for (BYTE_TO_UNICODE, 0..) |cp, b| {
        inv[cp] = @intCast(b);
    }
    break :blk inv;
};

pub inline fn byteToUnicode(b: u8) u21 {
    return BYTE_TO_UNICODE[b];
}

pub inline fn unicodeToByte(cp: u21) ?u8 {
    if (cp >= UNICODE_TO_BYTE_LEN) return null;
    return UNICODE_TO_BYTE[cp];
}

/// Encodes raw bytes into GPT-2/Qwen byte-level UTF-8 representation.
pub fn encodeByteLevelString(src: []const u8, dst: []u8) ByteLevelError![]u8 {
    var pos: usize = 0;
    for (src) |b| {
        const cp = byteToUnicode(b);
        var tmp: [4]u8 = undefined;
        const n = std.unicode.utf8Encode(cp, &tmp) catch unreachable;
        if (pos + n > dst.len) return error.BufferTooSmall;
        @memcpy(dst[pos .. pos + n], tmp[0..n]);
        pos += n;
    }
    return dst[0..pos];
}

/// Decodes a GPT-2/Qwen byte-level UTF-8 vocab string back into its raw bytes.
pub fn decodeByteLevelString(src: []const u8, dst: []u8) ByteLevelError![]u8 {
    var in_pos: usize = 0;
    var out_pos: usize = 0;
    while (in_pos < src.len) {
        const seq_len = std.unicode.utf8ByteSequenceLength(src[in_pos]) catch return error.InvalidUtf8;
        if (in_pos + seq_len > src.len) return error.InvalidUtf8;
        const cp = std.unicode.utf8Decode(src[in_pos .. in_pos + seq_len]) catch return error.InvalidUtf8;
        const b = unicodeToByte(cp) orelse return error.InvalidByteLevelCodepoint;
        if (out_pos >= dst.len) return error.BufferTooSmall;
        dst[out_pos] = b;
        out_pos += 1;
        in_pos += seq_len;
    }
    return dst[0..out_pos];
}

const CodepointRange = struct { start: u21, end: u21 };

fn inRanges(comptime ranges: []const CodepointRange, cp: u21) bool {
    var lo: usize = 0;
    var hi: usize = ranges.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        const r = ranges[mid];
        if (cp < r.start) {
            hi = mid;
        } else if (cp > r.end) {
            lo = mid + 1;
        } else {
            return true;
        }
    }
    return false;
}

fn assertSortedRanges(comptime ranges: []const CodepointRange) void {
    for (ranges, 0..) |r, i| {
        if (r.start > r.end) @compileError("invalid Unicode range");
        if (i > 0 and ranges[i - 1].end >= r.start) @compileError("unsorted or overlapping Unicode ranges");
    }
}

/// Non-ASCII Unicode `\p{L}` (Letter) ranges covering Latin, Greek, Cyrillic,
/// Armenian, Hebrew, Arabic, Indic, Thai, Georgian, Hangul, Kana, Bopomofo,
/// CJK Unified Ideographs (BMP + Ext A–H), Compatibility, and Fullwidth letters.
const NON_ASCII_LETTER_RANGES = [_]CodepointRange{
    .{ .start = 0x00AA, .end = 0x00AA },
    .{ .start = 0x00B5, .end = 0x00B5 },
    .{ .start = 0x00BA, .end = 0x00BA },
    .{ .start = 0x00C0, .end = 0x00D6 },
    .{ .start = 0x00D8, .end = 0x00F6 },
    .{ .start = 0x00F8, .end = 0x02C1 },
    .{ .start = 0x02C6, .end = 0x02D1 },
    .{ .start = 0x02E0, .end = 0x02E4 },
    .{ .start = 0x02EC, .end = 0x02EC },
    .{ .start = 0x02EE, .end = 0x02EE },
    .{ .start = 0x0370, .end = 0x0374 },
    .{ .start = 0x0376, .end = 0x0377 },
    .{ .start = 0x037A, .end = 0x037D },
    .{ .start = 0x037F, .end = 0x037F },
    .{ .start = 0x0386, .end = 0x0386 },
    .{ .start = 0x0388, .end = 0x038A },
    .{ .start = 0x038C, .end = 0x038C },
    .{ .start = 0x038E, .end = 0x03A1 },
    .{ .start = 0x03A3, .end = 0x03F5 },
    .{ .start = 0x03F7, .end = 0x0481 },
    .{ .start = 0x048A, .end = 0x052F },
    .{ .start = 0x0531, .end = 0x0556 },
    .{ .start = 0x0559, .end = 0x0559 },
    .{ .start = 0x0560, .end = 0x0588 },
    .{ .start = 0x05D0, .end = 0x05EA },
    .{ .start = 0x05EF, .end = 0x05F2 },
    .{ .start = 0x0620, .end = 0x064A },
    .{ .start = 0x066E, .end = 0x066F },
    .{ .start = 0x0671, .end = 0x06D3 },
    .{ .start = 0x06D5, .end = 0x06D5 },
    .{ .start = 0x06E5, .end = 0x06E6 },
    .{ .start = 0x06EE, .end = 0x06EF },
    .{ .start = 0x06FA, .end = 0x06FC },
    .{ .start = 0x06FF, .end = 0x06FF },
    .{ .start = 0x0904, .end = 0x0939 },
    .{ .start = 0x093D, .end = 0x093D },
    .{ .start = 0x0950, .end = 0x0950 },
    .{ .start = 0x0958, .end = 0x0961 },
    .{ .start = 0x0971, .end = 0x0980 },
    .{ .start = 0x0985, .end = 0x09B9 },
    .{ .start = 0x0E01, .end = 0x0E30 },
    .{ .start = 0x0E32, .end = 0x0E33 },
    .{ .start = 0x0E40, .end = 0x0E46 },
    .{ .start = 0x10A0, .end = 0x10FA },
    .{ .start = 0x1100, .end = 0x1248 },
    .{ .start = 0x1E00, .end = 0x1EFF },
    .{ .start = 0x1F00, .end = 0x1FBC },
    .{ .start = 0x1FC2, .end = 0x1FCC },
    .{ .start = 0x1FD0, .end = 0x1FDB },
    .{ .start = 0x1FE0, .end = 0x1FEC },
    .{ .start = 0x1FF2, .end = 0x1FFC },
    .{ .start = 0x2102, .end = 0x2102 },
    .{ .start = 0x2107, .end = 0x2107 },
    .{ .start = 0x210A, .end = 0x2113 },
    .{ .start = 0x2115, .end = 0x2115 },
    .{ .start = 0x2119, .end = 0x211D },
    .{ .start = 0x2124, .end = 0x2124 },
    .{ .start = 0x2126, .end = 0x2126 },
    .{ .start = 0x2128, .end = 0x2128 },
    .{ .start = 0x212A, .end = 0x212D },
    .{ .start = 0x212F, .end = 0x2139 },
    .{ .start = 0x213C, .end = 0x213F },
    .{ .start = 0x2145, .end = 0x2149 },
    .{ .start = 0x214E, .end = 0x214E },
    .{ .start = 0x3005, .end = 0x3006 },
    .{ .start = 0x3031, .end = 0x3035 },
    .{ .start = 0x303B, .end = 0x303C },
    .{ .start = 0x3041, .end = 0x3096 },
    .{ .start = 0x309D, .end = 0x309F },
    .{ .start = 0x30A1, .end = 0x30FA },
    .{ .start = 0x30FC, .end = 0x30FF },
    .{ .start = 0x3105, .end = 0x312F },
    .{ .start = 0x3131, .end = 0x318E },
    .{ .start = 0x31A0, .end = 0x31BF },
    .{ .start = 0x31F0, .end = 0x31FF },
    .{ .start = 0x3400, .end = 0x4DBF },
    .{ .start = 0x4E00, .end = 0xA48C },
    .{ .start = 0xAC00, .end = 0xD7A3 },
    .{ .start = 0xF900, .end = 0xFA6D },
    .{ .start = 0xFA70, .end = 0xFAD9 },
    .{ .start = 0xFB00, .end = 0xFB06 },
    .{ .start = 0xFF21, .end = 0xFF3A },
    .{ .start = 0xFF41, .end = 0xFF5A },
    .{ .start = 0xFF66, .end = 0xFFBE },
    .{ .start = 0xFFC2, .end = 0xFFC7 },
    .{ .start = 0xFFCA, .end = 0xFFCF },
    .{ .start = 0xFFD2, .end = 0xFFD7 },
    .{ .start = 0xFFDA, .end = 0xFFDC },
    .{ .start = 0x20000, .end = 0x2A6DF },
    .{ .start = 0x2A700, .end = 0x2B739 },
    .{ .start = 0x2B740, .end = 0x2B81D },
    .{ .start = 0x2B820, .end = 0x2CEA1 },
    .{ .start = 0x2CEB0, .end = 0x2EBE0 },
    .{ .start = 0x2EBF0, .end = 0x2EE5D },
    .{ .start = 0x2F800, .end = 0x2FA1D },
    .{ .start = 0x30000, .end = 0x3134A },
    .{ .start = 0x31350, .end = 0x323AF },
};

/// Non-ASCII Unicode `\p{M}` (Mark) ranges.
const NON_ASCII_MARK_RANGES = [_]CodepointRange{
    .{ .start = 0x0300, .end = 0x036F },
    .{ .start = 0x0483, .end = 0x0489 },
    .{ .start = 0x0591, .end = 0x05BD },
    .{ .start = 0x05BF, .end = 0x05BF },
    .{ .start = 0x05C1, .end = 0x05C2 },
    .{ .start = 0x05C4, .end = 0x05C5 },
    .{ .start = 0x05C7, .end = 0x05C7 },
    .{ .start = 0x0610, .end = 0x061A },
    .{ .start = 0x064B, .end = 0x065F },
    .{ .start = 0x0670, .end = 0x0670 },
    .{ .start = 0x06D6, .end = 0x06DC },
    .{ .start = 0x06DF, .end = 0x06E4 },
    .{ .start = 0x06E7, .end = 0x06E8 },
    .{ .start = 0x06EA, .end = 0x06ED },
    .{ .start = 0x0900, .end = 0x0903 },
    .{ .start = 0x093A, .end = 0x093C },
    .{ .start = 0x093E, .end = 0x094F },
    .{ .start = 0x0951, .end = 0x0957 },
    .{ .start = 0x0962, .end = 0x0963 },
    .{ .start = 0x0E31, .end = 0x0E31 },
    .{ .start = 0x0E34, .end = 0x0E3A },
    .{ .start = 0x0E47, .end = 0x0E4E },
    .{ .start = 0x1AB0, .end = 0x1AFF },
    .{ .start = 0x1DC0, .end = 0x1DFF },
    .{ .start = 0x20D0, .end = 0x20F0 },
    .{ .start = 0x302A, .end = 0x302F },
    .{ .start = 0x3099, .end = 0x309A },
    .{ .start = 0xFE00, .end = 0xFE0F },
    .{ .start = 0xFE20, .end = 0xFE2F },
};

/// Non-ASCII Unicode `\p{N}` (Number) ranges.
const NON_ASCII_NUMBER_RANGES = [_]CodepointRange{
    .{ .start = 0x00B2, .end = 0x00B3 },
    .{ .start = 0x00B9, .end = 0x00B9 },
    .{ .start = 0x00BC, .end = 0x00BE },
    .{ .start = 0x0660, .end = 0x0669 },
    .{ .start = 0x06F0, .end = 0x06F9 },
    .{ .start = 0x0966, .end = 0x096F },
    .{ .start = 0x09E6, .end = 0x09EF },
    .{ .start = 0x0E50, .end = 0x0E59 },
    .{ .start = 0x2070, .end = 0x2070 },
    .{ .start = 0x2074, .end = 0x2079 },
    .{ .start = 0x2080, .end = 0x2089 },
    .{ .start = 0x2150, .end = 0x2182 },
    .{ .start = 0x2185, .end = 0x2189 },
    .{ .start = 0x2460, .end = 0x249B },
    .{ .start = 0x24EA, .end = 0x24FF },
    .{ .start = 0x2776, .end = 0x2793 },
    .{ .start = 0x3007, .end = 0x3007 },
    .{ .start = 0x3021, .end = 0x3029 },
    .{ .start = 0x3038, .end = 0x303A },
    .{ .start = 0xFF10, .end = 0xFF19 },
};

const O200kLetterCat = enum(u2) { upper_title, lower, other };
const LetterCatRange = struct { start: u21, end: u21, cat: O200kLetterCat };
const O200K_LETTER_CAT_RANGES = [_]LetterCatRange{
    .{ .start = 0x00AA, .end = 0x00AA, .cat = .other },         .{ .start = 0x00B5, .end = 0x00B5, .cat = .lower },         .{ .start = 0x00BA, .end = 0x00BA, .cat = .other },
    .{ .start = 0x00C0, .end = 0x00D6, .cat = .upper_title },   .{ .start = 0x00D8, .end = 0x00DE, .cat = .upper_title },   .{ .start = 0x00DF, .end = 0x00F6, .cat = .lower },
    .{ .start = 0x00F8, .end = 0x00FF, .cat = .lower },         .{ .start = 0x0100, .end = 0x0100, .cat = .upper_title },   .{ .start = 0x0101, .end = 0x0101, .cat = .lower },
    .{ .start = 0x0102, .end = 0x0102, .cat = .upper_title },   .{ .start = 0x0103, .end = 0x0103, .cat = .lower },         .{ .start = 0x0104, .end = 0x0104, .cat = .upper_title },
    .{ .start = 0x0105, .end = 0x0105, .cat = .lower },         .{ .start = 0x0106, .end = 0x0106, .cat = .upper_title },   .{ .start = 0x0107, .end = 0x0107, .cat = .lower },
    .{ .start = 0x0108, .end = 0x0108, .cat = .upper_title },   .{ .start = 0x0109, .end = 0x0109, .cat = .lower },         .{ .start = 0x010A, .end = 0x010A, .cat = .upper_title },
    .{ .start = 0x010B, .end = 0x010B, .cat = .lower },         .{ .start = 0x010C, .end = 0x010C, .cat = .upper_title },   .{ .start = 0x010D, .end = 0x010D, .cat = .lower },
    .{ .start = 0x010E, .end = 0x010E, .cat = .upper_title },   .{ .start = 0x010F, .end = 0x010F, .cat = .lower },         .{ .start = 0x0110, .end = 0x0110, .cat = .upper_title },
    .{ .start = 0x0111, .end = 0x0111, .cat = .lower },         .{ .start = 0x0112, .end = 0x0112, .cat = .upper_title },   .{ .start = 0x0113, .end = 0x0113, .cat = .lower },
    .{ .start = 0x0114, .end = 0x0114, .cat = .upper_title },   .{ .start = 0x0115, .end = 0x0115, .cat = .lower },         .{ .start = 0x0116, .end = 0x0116, .cat = .upper_title },
    .{ .start = 0x0117, .end = 0x0117, .cat = .lower },         .{ .start = 0x0118, .end = 0x0118, .cat = .upper_title },   .{ .start = 0x0119, .end = 0x0119, .cat = .lower },
    .{ .start = 0x011A, .end = 0x011A, .cat = .upper_title },   .{ .start = 0x011B, .end = 0x011B, .cat = .lower },         .{ .start = 0x011C, .end = 0x011C, .cat = .upper_title },
    .{ .start = 0x011D, .end = 0x011D, .cat = .lower },         .{ .start = 0x011E, .end = 0x011E, .cat = .upper_title },   .{ .start = 0x011F, .end = 0x011F, .cat = .lower },
    .{ .start = 0x0120, .end = 0x0120, .cat = .upper_title },   .{ .start = 0x0121, .end = 0x0121, .cat = .lower },         .{ .start = 0x0122, .end = 0x0122, .cat = .upper_title },
    .{ .start = 0x0123, .end = 0x0123, .cat = .lower },         .{ .start = 0x0124, .end = 0x0124, .cat = .upper_title },   .{ .start = 0x0125, .end = 0x0125, .cat = .lower },
    .{ .start = 0x0126, .end = 0x0126, .cat = .upper_title },   .{ .start = 0x0127, .end = 0x0127, .cat = .lower },         .{ .start = 0x0128, .end = 0x0128, .cat = .upper_title },
    .{ .start = 0x0129, .end = 0x0129, .cat = .lower },         .{ .start = 0x012A, .end = 0x012A, .cat = .upper_title },   .{ .start = 0x012B, .end = 0x012B, .cat = .lower },
    .{ .start = 0x012C, .end = 0x012C, .cat = .upper_title },   .{ .start = 0x012D, .end = 0x012D, .cat = .lower },         .{ .start = 0x012E, .end = 0x012E, .cat = .upper_title },
    .{ .start = 0x012F, .end = 0x012F, .cat = .lower },         .{ .start = 0x0130, .end = 0x0130, .cat = .upper_title },   .{ .start = 0x0131, .end = 0x0131, .cat = .lower },
    .{ .start = 0x0132, .end = 0x0132, .cat = .upper_title },   .{ .start = 0x0133, .end = 0x0133, .cat = .lower },         .{ .start = 0x0134, .end = 0x0134, .cat = .upper_title },
    .{ .start = 0x0135, .end = 0x0135, .cat = .lower },         .{ .start = 0x0136, .end = 0x0136, .cat = .upper_title },   .{ .start = 0x0137, .end = 0x0138, .cat = .lower },
    .{ .start = 0x0139, .end = 0x0139, .cat = .upper_title },   .{ .start = 0x013A, .end = 0x013A, .cat = .lower },         .{ .start = 0x013B, .end = 0x013B, .cat = .upper_title },
    .{ .start = 0x013C, .end = 0x013C, .cat = .lower },         .{ .start = 0x013D, .end = 0x013D, .cat = .upper_title },   .{ .start = 0x013E, .end = 0x013E, .cat = .lower },
    .{ .start = 0x013F, .end = 0x013F, .cat = .upper_title },   .{ .start = 0x0140, .end = 0x0140, .cat = .lower },         .{ .start = 0x0141, .end = 0x0141, .cat = .upper_title },
    .{ .start = 0x0142, .end = 0x0142, .cat = .lower },         .{ .start = 0x0143, .end = 0x0143, .cat = .upper_title },   .{ .start = 0x0144, .end = 0x0144, .cat = .lower },
    .{ .start = 0x0145, .end = 0x0145, .cat = .upper_title },   .{ .start = 0x0146, .end = 0x0146, .cat = .lower },         .{ .start = 0x0147, .end = 0x0147, .cat = .upper_title },
    .{ .start = 0x0148, .end = 0x0149, .cat = .lower },         .{ .start = 0x014A, .end = 0x014A, .cat = .upper_title },   .{ .start = 0x014B, .end = 0x014B, .cat = .lower },
    .{ .start = 0x014C, .end = 0x014C, .cat = .upper_title },   .{ .start = 0x014D, .end = 0x014D, .cat = .lower },         .{ .start = 0x014E, .end = 0x014E, .cat = .upper_title },
    .{ .start = 0x014F, .end = 0x014F, .cat = .lower },         .{ .start = 0x0150, .end = 0x0150, .cat = .upper_title },   .{ .start = 0x0151, .end = 0x0151, .cat = .lower },
    .{ .start = 0x0152, .end = 0x0152, .cat = .upper_title },   .{ .start = 0x0153, .end = 0x0153, .cat = .lower },         .{ .start = 0x0154, .end = 0x0154, .cat = .upper_title },
    .{ .start = 0x0155, .end = 0x0155, .cat = .lower },         .{ .start = 0x0156, .end = 0x0156, .cat = .upper_title },   .{ .start = 0x0157, .end = 0x0157, .cat = .lower },
    .{ .start = 0x0158, .end = 0x0158, .cat = .upper_title },   .{ .start = 0x0159, .end = 0x0159, .cat = .lower },         .{ .start = 0x015A, .end = 0x015A, .cat = .upper_title },
    .{ .start = 0x015B, .end = 0x015B, .cat = .lower },         .{ .start = 0x015C, .end = 0x015C, .cat = .upper_title },   .{ .start = 0x015D, .end = 0x015D, .cat = .lower },
    .{ .start = 0x015E, .end = 0x015E, .cat = .upper_title },   .{ .start = 0x015F, .end = 0x015F, .cat = .lower },         .{ .start = 0x0160, .end = 0x0160, .cat = .upper_title },
    .{ .start = 0x0161, .end = 0x0161, .cat = .lower },         .{ .start = 0x0162, .end = 0x0162, .cat = .upper_title },   .{ .start = 0x0163, .end = 0x0163, .cat = .lower },
    .{ .start = 0x0164, .end = 0x0164, .cat = .upper_title },   .{ .start = 0x0165, .end = 0x0165, .cat = .lower },         .{ .start = 0x0166, .end = 0x0166, .cat = .upper_title },
    .{ .start = 0x0167, .end = 0x0167, .cat = .lower },         .{ .start = 0x0168, .end = 0x0168, .cat = .upper_title },   .{ .start = 0x0169, .end = 0x0169, .cat = .lower },
    .{ .start = 0x016A, .end = 0x016A, .cat = .upper_title },   .{ .start = 0x016B, .end = 0x016B, .cat = .lower },         .{ .start = 0x016C, .end = 0x016C, .cat = .upper_title },
    .{ .start = 0x016D, .end = 0x016D, .cat = .lower },         .{ .start = 0x016E, .end = 0x016E, .cat = .upper_title },   .{ .start = 0x016F, .end = 0x016F, .cat = .lower },
    .{ .start = 0x0170, .end = 0x0170, .cat = .upper_title },   .{ .start = 0x0171, .end = 0x0171, .cat = .lower },         .{ .start = 0x0172, .end = 0x0172, .cat = .upper_title },
    .{ .start = 0x0173, .end = 0x0173, .cat = .lower },         .{ .start = 0x0174, .end = 0x0174, .cat = .upper_title },   .{ .start = 0x0175, .end = 0x0175, .cat = .lower },
    .{ .start = 0x0176, .end = 0x0176, .cat = .upper_title },   .{ .start = 0x0177, .end = 0x0177, .cat = .lower },         .{ .start = 0x0178, .end = 0x0179, .cat = .upper_title },
    .{ .start = 0x017A, .end = 0x017A, .cat = .lower },         .{ .start = 0x017B, .end = 0x017B, .cat = .upper_title },   .{ .start = 0x017C, .end = 0x017C, .cat = .lower },
    .{ .start = 0x017D, .end = 0x017D, .cat = .upper_title },   .{ .start = 0x017E, .end = 0x0180, .cat = .lower },         .{ .start = 0x0181, .end = 0x0182, .cat = .upper_title },
    .{ .start = 0x0183, .end = 0x0183, .cat = .lower },         .{ .start = 0x0184, .end = 0x0184, .cat = .upper_title },   .{ .start = 0x0185, .end = 0x0185, .cat = .lower },
    .{ .start = 0x0186, .end = 0x0187, .cat = .upper_title },   .{ .start = 0x0188, .end = 0x0188, .cat = .lower },         .{ .start = 0x0189, .end = 0x018B, .cat = .upper_title },
    .{ .start = 0x018C, .end = 0x018D, .cat = .lower },         .{ .start = 0x018E, .end = 0x0191, .cat = .upper_title },   .{ .start = 0x0192, .end = 0x0192, .cat = .lower },
    .{ .start = 0x0193, .end = 0x0194, .cat = .upper_title },   .{ .start = 0x0195, .end = 0x0195, .cat = .lower },         .{ .start = 0x0196, .end = 0x0198, .cat = .upper_title },
    .{ .start = 0x0199, .end = 0x019B, .cat = .lower },         .{ .start = 0x019C, .end = 0x019D, .cat = .upper_title },   .{ .start = 0x019E, .end = 0x019E, .cat = .lower },
    .{ .start = 0x019F, .end = 0x01A0, .cat = .upper_title },   .{ .start = 0x01A1, .end = 0x01A1, .cat = .lower },         .{ .start = 0x01A2, .end = 0x01A2, .cat = .upper_title },
    .{ .start = 0x01A3, .end = 0x01A3, .cat = .lower },         .{ .start = 0x01A4, .end = 0x01A4, .cat = .upper_title },   .{ .start = 0x01A5, .end = 0x01A5, .cat = .lower },
    .{ .start = 0x01A6, .end = 0x01A7, .cat = .upper_title },   .{ .start = 0x01A8, .end = 0x01A8, .cat = .lower },         .{ .start = 0x01A9, .end = 0x01A9, .cat = .upper_title },
    .{ .start = 0x01AA, .end = 0x01AB, .cat = .lower },         .{ .start = 0x01AC, .end = 0x01AC, .cat = .upper_title },   .{ .start = 0x01AD, .end = 0x01AD, .cat = .lower },
    .{ .start = 0x01AE, .end = 0x01AF, .cat = .upper_title },   .{ .start = 0x01B0, .end = 0x01B0, .cat = .lower },         .{ .start = 0x01B1, .end = 0x01B3, .cat = .upper_title },
    .{ .start = 0x01B4, .end = 0x01B4, .cat = .lower },         .{ .start = 0x01B5, .end = 0x01B5, .cat = .upper_title },   .{ .start = 0x01B6, .end = 0x01B6, .cat = .lower },
    .{ .start = 0x01B7, .end = 0x01B8, .cat = .upper_title },   .{ .start = 0x01B9, .end = 0x01BA, .cat = .lower },         .{ .start = 0x01BB, .end = 0x01BB, .cat = .other },
    .{ .start = 0x01BC, .end = 0x01BC, .cat = .upper_title },   .{ .start = 0x01BD, .end = 0x01BF, .cat = .lower },         .{ .start = 0x01C0, .end = 0x01C3, .cat = .other },
    .{ .start = 0x01C4, .end = 0x01C5, .cat = .upper_title },   .{ .start = 0x01C6, .end = 0x01C6, .cat = .lower },         .{ .start = 0x01C7, .end = 0x01C8, .cat = .upper_title },
    .{ .start = 0x01C9, .end = 0x01C9, .cat = .lower },         .{ .start = 0x01CA, .end = 0x01CB, .cat = .upper_title },   .{ .start = 0x01CC, .end = 0x01CC, .cat = .lower },
    .{ .start = 0x01CD, .end = 0x01CD, .cat = .upper_title },   .{ .start = 0x01CE, .end = 0x01CE, .cat = .lower },         .{ .start = 0x01CF, .end = 0x01CF, .cat = .upper_title },
    .{ .start = 0x01D0, .end = 0x01D0, .cat = .lower },         .{ .start = 0x01D1, .end = 0x01D1, .cat = .upper_title },   .{ .start = 0x01D2, .end = 0x01D2, .cat = .lower },
    .{ .start = 0x01D3, .end = 0x01D3, .cat = .upper_title },   .{ .start = 0x01D4, .end = 0x01D4, .cat = .lower },         .{ .start = 0x01D5, .end = 0x01D5, .cat = .upper_title },
    .{ .start = 0x01D6, .end = 0x01D6, .cat = .lower },         .{ .start = 0x01D7, .end = 0x01D7, .cat = .upper_title },   .{ .start = 0x01D8, .end = 0x01D8, .cat = .lower },
    .{ .start = 0x01D9, .end = 0x01D9, .cat = .upper_title },   .{ .start = 0x01DA, .end = 0x01DA, .cat = .lower },         .{ .start = 0x01DB, .end = 0x01DB, .cat = .upper_title },
    .{ .start = 0x01DC, .end = 0x01DD, .cat = .lower },         .{ .start = 0x01DE, .end = 0x01DE, .cat = .upper_title },   .{ .start = 0x01DF, .end = 0x01DF, .cat = .lower },
    .{ .start = 0x01E0, .end = 0x01E0, .cat = .upper_title },   .{ .start = 0x01E1, .end = 0x01E1, .cat = .lower },         .{ .start = 0x01E2, .end = 0x01E2, .cat = .upper_title },
    .{ .start = 0x01E3, .end = 0x01E3, .cat = .lower },         .{ .start = 0x01E4, .end = 0x01E4, .cat = .upper_title },   .{ .start = 0x01E5, .end = 0x01E5, .cat = .lower },
    .{ .start = 0x01E6, .end = 0x01E6, .cat = .upper_title },   .{ .start = 0x01E7, .end = 0x01E7, .cat = .lower },         .{ .start = 0x01E8, .end = 0x01E8, .cat = .upper_title },
    .{ .start = 0x01E9, .end = 0x01E9, .cat = .lower },         .{ .start = 0x01EA, .end = 0x01EA, .cat = .upper_title },   .{ .start = 0x01EB, .end = 0x01EB, .cat = .lower },
    .{ .start = 0x01EC, .end = 0x01EC, .cat = .upper_title },   .{ .start = 0x01ED, .end = 0x01ED, .cat = .lower },         .{ .start = 0x01EE, .end = 0x01EE, .cat = .upper_title },
    .{ .start = 0x01EF, .end = 0x01F0, .cat = .lower },         .{ .start = 0x01F1, .end = 0x01F2, .cat = .upper_title },   .{ .start = 0x01F3, .end = 0x01F3, .cat = .lower },
    .{ .start = 0x01F4, .end = 0x01F4, .cat = .upper_title },   .{ .start = 0x01F5, .end = 0x01F5, .cat = .lower },         .{ .start = 0x01F6, .end = 0x01F8, .cat = .upper_title },
    .{ .start = 0x01F9, .end = 0x01F9, .cat = .lower },         .{ .start = 0x01FA, .end = 0x01FA, .cat = .upper_title },   .{ .start = 0x01FB, .end = 0x01FB, .cat = .lower },
    .{ .start = 0x01FC, .end = 0x01FC, .cat = .upper_title },   .{ .start = 0x01FD, .end = 0x01FD, .cat = .lower },         .{ .start = 0x01FE, .end = 0x01FE, .cat = .upper_title },
    .{ .start = 0x01FF, .end = 0x01FF, .cat = .lower },         .{ .start = 0x0200, .end = 0x0200, .cat = .upper_title },   .{ .start = 0x0201, .end = 0x0201, .cat = .lower },
    .{ .start = 0x0202, .end = 0x0202, .cat = .upper_title },   .{ .start = 0x0203, .end = 0x0203, .cat = .lower },         .{ .start = 0x0204, .end = 0x0204, .cat = .upper_title },
    .{ .start = 0x0205, .end = 0x0205, .cat = .lower },         .{ .start = 0x0206, .end = 0x0206, .cat = .upper_title },   .{ .start = 0x0207, .end = 0x0207, .cat = .lower },
    .{ .start = 0x0208, .end = 0x0208, .cat = .upper_title },   .{ .start = 0x0209, .end = 0x0209, .cat = .lower },         .{ .start = 0x020A, .end = 0x020A, .cat = .upper_title },
    .{ .start = 0x020B, .end = 0x020B, .cat = .lower },         .{ .start = 0x020C, .end = 0x020C, .cat = .upper_title },   .{ .start = 0x020D, .end = 0x020D, .cat = .lower },
    .{ .start = 0x020E, .end = 0x020E, .cat = .upper_title },   .{ .start = 0x020F, .end = 0x020F, .cat = .lower },         .{ .start = 0x0210, .end = 0x0210, .cat = .upper_title },
    .{ .start = 0x0211, .end = 0x0211, .cat = .lower },         .{ .start = 0x0212, .end = 0x0212, .cat = .upper_title },   .{ .start = 0x0213, .end = 0x0213, .cat = .lower },
    .{ .start = 0x0214, .end = 0x0214, .cat = .upper_title },   .{ .start = 0x0215, .end = 0x0215, .cat = .lower },         .{ .start = 0x0216, .end = 0x0216, .cat = .upper_title },
    .{ .start = 0x0217, .end = 0x0217, .cat = .lower },         .{ .start = 0x0218, .end = 0x0218, .cat = .upper_title },   .{ .start = 0x0219, .end = 0x0219, .cat = .lower },
    .{ .start = 0x021A, .end = 0x021A, .cat = .upper_title },   .{ .start = 0x021B, .end = 0x021B, .cat = .lower },         .{ .start = 0x021C, .end = 0x021C, .cat = .upper_title },
    .{ .start = 0x021D, .end = 0x021D, .cat = .lower },         .{ .start = 0x021E, .end = 0x021E, .cat = .upper_title },   .{ .start = 0x021F, .end = 0x021F, .cat = .lower },
    .{ .start = 0x0220, .end = 0x0220, .cat = .upper_title },   .{ .start = 0x0221, .end = 0x0221, .cat = .lower },         .{ .start = 0x0222, .end = 0x0222, .cat = .upper_title },
    .{ .start = 0x0223, .end = 0x0223, .cat = .lower },         .{ .start = 0x0224, .end = 0x0224, .cat = .upper_title },   .{ .start = 0x0225, .end = 0x0225, .cat = .lower },
    .{ .start = 0x0226, .end = 0x0226, .cat = .upper_title },   .{ .start = 0x0227, .end = 0x0227, .cat = .lower },         .{ .start = 0x0228, .end = 0x0228, .cat = .upper_title },
    .{ .start = 0x0229, .end = 0x0229, .cat = .lower },         .{ .start = 0x022A, .end = 0x022A, .cat = .upper_title },   .{ .start = 0x022B, .end = 0x022B, .cat = .lower },
    .{ .start = 0x022C, .end = 0x022C, .cat = .upper_title },   .{ .start = 0x022D, .end = 0x022D, .cat = .lower },         .{ .start = 0x022E, .end = 0x022E, .cat = .upper_title },
    .{ .start = 0x022F, .end = 0x022F, .cat = .lower },         .{ .start = 0x0230, .end = 0x0230, .cat = .upper_title },   .{ .start = 0x0231, .end = 0x0231, .cat = .lower },
    .{ .start = 0x0232, .end = 0x0232, .cat = .upper_title },   .{ .start = 0x0233, .end = 0x0239, .cat = .lower },         .{ .start = 0x023A, .end = 0x023B, .cat = .upper_title },
    .{ .start = 0x023C, .end = 0x023C, .cat = .lower },         .{ .start = 0x023D, .end = 0x023E, .cat = .upper_title },   .{ .start = 0x023F, .end = 0x0240, .cat = .lower },
    .{ .start = 0x0241, .end = 0x0241, .cat = .upper_title },   .{ .start = 0x0242, .end = 0x0242, .cat = .lower },         .{ .start = 0x0243, .end = 0x0246, .cat = .upper_title },
    .{ .start = 0x0247, .end = 0x0247, .cat = .lower },         .{ .start = 0x0248, .end = 0x0248, .cat = .upper_title },   .{ .start = 0x0249, .end = 0x0249, .cat = .lower },
    .{ .start = 0x024A, .end = 0x024A, .cat = .upper_title },   .{ .start = 0x024B, .end = 0x024B, .cat = .lower },         .{ .start = 0x024C, .end = 0x024C, .cat = .upper_title },
    .{ .start = 0x024D, .end = 0x024D, .cat = .lower },         .{ .start = 0x024E, .end = 0x024E, .cat = .upper_title },   .{ .start = 0x024F, .end = 0x0293, .cat = .lower },
    .{ .start = 0x0294, .end = 0x0295, .cat = .other },         .{ .start = 0x0296, .end = 0x02AF, .cat = .lower },         .{ .start = 0x02B0, .end = 0x02C1, .cat = .other },
    .{ .start = 0x02C6, .end = 0x02D1, .cat = .other },         .{ .start = 0x02E0, .end = 0x02E4, .cat = .other },         .{ .start = 0x02EC, .end = 0x02EC, .cat = .other },
    .{ .start = 0x02EE, .end = 0x02EE, .cat = .other },         .{ .start = 0x0370, .end = 0x0370, .cat = .upper_title },   .{ .start = 0x0371, .end = 0x0371, .cat = .lower },
    .{ .start = 0x0372, .end = 0x0372, .cat = .upper_title },   .{ .start = 0x0373, .end = 0x0373, .cat = .lower },         .{ .start = 0x0374, .end = 0x0374, .cat = .other },
    .{ .start = 0x0376, .end = 0x0376, .cat = .upper_title },   .{ .start = 0x0377, .end = 0x0377, .cat = .lower },         .{ .start = 0x037A, .end = 0x037A, .cat = .other },
    .{ .start = 0x037B, .end = 0x037D, .cat = .lower },         .{ .start = 0x037F, .end = 0x037F, .cat = .upper_title },   .{ .start = 0x0386, .end = 0x0386, .cat = .upper_title },
    .{ .start = 0x0388, .end = 0x038A, .cat = .upper_title },   .{ .start = 0x038C, .end = 0x038C, .cat = .upper_title },   .{ .start = 0x038E, .end = 0x038F, .cat = .upper_title },
    .{ .start = 0x0390, .end = 0x0390, .cat = .lower },         .{ .start = 0x0391, .end = 0x03A1, .cat = .upper_title },   .{ .start = 0x03A3, .end = 0x03AB, .cat = .upper_title },
    .{ .start = 0x03AC, .end = 0x03CE, .cat = .lower },         .{ .start = 0x03CF, .end = 0x03CF, .cat = .upper_title },   .{ .start = 0x03D0, .end = 0x03D1, .cat = .lower },
    .{ .start = 0x03D2, .end = 0x03D4, .cat = .upper_title },   .{ .start = 0x03D5, .end = 0x03D7, .cat = .lower },         .{ .start = 0x03D8, .end = 0x03D8, .cat = .upper_title },
    .{ .start = 0x03D9, .end = 0x03D9, .cat = .lower },         .{ .start = 0x03DA, .end = 0x03DA, .cat = .upper_title },   .{ .start = 0x03DB, .end = 0x03DB, .cat = .lower },
    .{ .start = 0x03DC, .end = 0x03DC, .cat = .upper_title },   .{ .start = 0x03DD, .end = 0x03DD, .cat = .lower },         .{ .start = 0x03DE, .end = 0x03DE, .cat = .upper_title },
    .{ .start = 0x03DF, .end = 0x03DF, .cat = .lower },         .{ .start = 0x03E0, .end = 0x03E0, .cat = .upper_title },   .{ .start = 0x03E1, .end = 0x03E1, .cat = .lower },
    .{ .start = 0x03E2, .end = 0x03E2, .cat = .upper_title },   .{ .start = 0x03E3, .end = 0x03E3, .cat = .lower },         .{ .start = 0x03E4, .end = 0x03E4, .cat = .upper_title },
    .{ .start = 0x03E5, .end = 0x03E5, .cat = .lower },         .{ .start = 0x03E6, .end = 0x03E6, .cat = .upper_title },   .{ .start = 0x03E7, .end = 0x03E7, .cat = .lower },
    .{ .start = 0x03E8, .end = 0x03E8, .cat = .upper_title },   .{ .start = 0x03E9, .end = 0x03E9, .cat = .lower },         .{ .start = 0x03EA, .end = 0x03EA, .cat = .upper_title },
    .{ .start = 0x03EB, .end = 0x03EB, .cat = .lower },         .{ .start = 0x03EC, .end = 0x03EC, .cat = .upper_title },   .{ .start = 0x03ED, .end = 0x03ED, .cat = .lower },
    .{ .start = 0x03EE, .end = 0x03EE, .cat = .upper_title },   .{ .start = 0x03EF, .end = 0x03F3, .cat = .lower },         .{ .start = 0x03F4, .end = 0x03F4, .cat = .upper_title },
    .{ .start = 0x03F5, .end = 0x03F5, .cat = .lower },         .{ .start = 0x03F7, .end = 0x03F7, .cat = .upper_title },   .{ .start = 0x03F8, .end = 0x03F8, .cat = .lower },
    .{ .start = 0x03F9, .end = 0x03FA, .cat = .upper_title },   .{ .start = 0x03FB, .end = 0x03FC, .cat = .lower },         .{ .start = 0x03FD, .end = 0x042F, .cat = .upper_title },
    .{ .start = 0x0430, .end = 0x045F, .cat = .lower },         .{ .start = 0x0460, .end = 0x0460, .cat = .upper_title },   .{ .start = 0x0461, .end = 0x0461, .cat = .lower },
    .{ .start = 0x0462, .end = 0x0462, .cat = .upper_title },   .{ .start = 0x0463, .end = 0x0463, .cat = .lower },         .{ .start = 0x0464, .end = 0x0464, .cat = .upper_title },
    .{ .start = 0x0465, .end = 0x0465, .cat = .lower },         .{ .start = 0x0466, .end = 0x0466, .cat = .upper_title },   .{ .start = 0x0467, .end = 0x0467, .cat = .lower },
    .{ .start = 0x0468, .end = 0x0468, .cat = .upper_title },   .{ .start = 0x0469, .end = 0x0469, .cat = .lower },         .{ .start = 0x046A, .end = 0x046A, .cat = .upper_title },
    .{ .start = 0x046B, .end = 0x046B, .cat = .lower },         .{ .start = 0x046C, .end = 0x046C, .cat = .upper_title },   .{ .start = 0x046D, .end = 0x046D, .cat = .lower },
    .{ .start = 0x046E, .end = 0x046E, .cat = .upper_title },   .{ .start = 0x046F, .end = 0x046F, .cat = .lower },         .{ .start = 0x0470, .end = 0x0470, .cat = .upper_title },
    .{ .start = 0x0471, .end = 0x0471, .cat = .lower },         .{ .start = 0x0472, .end = 0x0472, .cat = .upper_title },   .{ .start = 0x0473, .end = 0x0473, .cat = .lower },
    .{ .start = 0x0474, .end = 0x0474, .cat = .upper_title },   .{ .start = 0x0475, .end = 0x0475, .cat = .lower },         .{ .start = 0x0476, .end = 0x0476, .cat = .upper_title },
    .{ .start = 0x0477, .end = 0x0477, .cat = .lower },         .{ .start = 0x0478, .end = 0x0478, .cat = .upper_title },   .{ .start = 0x0479, .end = 0x0479, .cat = .lower },
    .{ .start = 0x047A, .end = 0x047A, .cat = .upper_title },   .{ .start = 0x047B, .end = 0x047B, .cat = .lower },         .{ .start = 0x047C, .end = 0x047C, .cat = .upper_title },
    .{ .start = 0x047D, .end = 0x047D, .cat = .lower },         .{ .start = 0x047E, .end = 0x047E, .cat = .upper_title },   .{ .start = 0x047F, .end = 0x047F, .cat = .lower },
    .{ .start = 0x0480, .end = 0x0480, .cat = .upper_title },   .{ .start = 0x0481, .end = 0x0481, .cat = .lower },         .{ .start = 0x048A, .end = 0x048A, .cat = .upper_title },
    .{ .start = 0x048B, .end = 0x048B, .cat = .lower },         .{ .start = 0x048C, .end = 0x048C, .cat = .upper_title },   .{ .start = 0x048D, .end = 0x048D, .cat = .lower },
    .{ .start = 0x048E, .end = 0x048E, .cat = .upper_title },   .{ .start = 0x048F, .end = 0x048F, .cat = .lower },         .{ .start = 0x0490, .end = 0x0490, .cat = .upper_title },
    .{ .start = 0x0491, .end = 0x0491, .cat = .lower },         .{ .start = 0x0492, .end = 0x0492, .cat = .upper_title },   .{ .start = 0x0493, .end = 0x0493, .cat = .lower },
    .{ .start = 0x0494, .end = 0x0494, .cat = .upper_title },   .{ .start = 0x0495, .end = 0x0495, .cat = .lower },         .{ .start = 0x0496, .end = 0x0496, .cat = .upper_title },
    .{ .start = 0x0497, .end = 0x0497, .cat = .lower },         .{ .start = 0x0498, .end = 0x0498, .cat = .upper_title },   .{ .start = 0x0499, .end = 0x0499, .cat = .lower },
    .{ .start = 0x049A, .end = 0x049A, .cat = .upper_title },   .{ .start = 0x049B, .end = 0x049B, .cat = .lower },         .{ .start = 0x049C, .end = 0x049C, .cat = .upper_title },
    .{ .start = 0x049D, .end = 0x049D, .cat = .lower },         .{ .start = 0x049E, .end = 0x049E, .cat = .upper_title },   .{ .start = 0x049F, .end = 0x049F, .cat = .lower },
    .{ .start = 0x04A0, .end = 0x04A0, .cat = .upper_title },   .{ .start = 0x04A1, .end = 0x04A1, .cat = .lower },         .{ .start = 0x04A2, .end = 0x04A2, .cat = .upper_title },
    .{ .start = 0x04A3, .end = 0x04A3, .cat = .lower },         .{ .start = 0x04A4, .end = 0x04A4, .cat = .upper_title },   .{ .start = 0x04A5, .end = 0x04A5, .cat = .lower },
    .{ .start = 0x04A6, .end = 0x04A6, .cat = .upper_title },   .{ .start = 0x04A7, .end = 0x04A7, .cat = .lower },         .{ .start = 0x04A8, .end = 0x04A8, .cat = .upper_title },
    .{ .start = 0x04A9, .end = 0x04A9, .cat = .lower },         .{ .start = 0x04AA, .end = 0x04AA, .cat = .upper_title },   .{ .start = 0x04AB, .end = 0x04AB, .cat = .lower },
    .{ .start = 0x04AC, .end = 0x04AC, .cat = .upper_title },   .{ .start = 0x04AD, .end = 0x04AD, .cat = .lower },         .{ .start = 0x04AE, .end = 0x04AE, .cat = .upper_title },
    .{ .start = 0x04AF, .end = 0x04AF, .cat = .lower },         .{ .start = 0x04B0, .end = 0x04B0, .cat = .upper_title },   .{ .start = 0x04B1, .end = 0x04B1, .cat = .lower },
    .{ .start = 0x04B2, .end = 0x04B2, .cat = .upper_title },   .{ .start = 0x04B3, .end = 0x04B3, .cat = .lower },         .{ .start = 0x04B4, .end = 0x04B4, .cat = .upper_title },
    .{ .start = 0x04B5, .end = 0x04B5, .cat = .lower },         .{ .start = 0x04B6, .end = 0x04B6, .cat = .upper_title },   .{ .start = 0x04B7, .end = 0x04B7, .cat = .lower },
    .{ .start = 0x04B8, .end = 0x04B8, .cat = .upper_title },   .{ .start = 0x04B9, .end = 0x04B9, .cat = .lower },         .{ .start = 0x04BA, .end = 0x04BA, .cat = .upper_title },
    .{ .start = 0x04BB, .end = 0x04BB, .cat = .lower },         .{ .start = 0x04BC, .end = 0x04BC, .cat = .upper_title },   .{ .start = 0x04BD, .end = 0x04BD, .cat = .lower },
    .{ .start = 0x04BE, .end = 0x04BE, .cat = .upper_title },   .{ .start = 0x04BF, .end = 0x04BF, .cat = .lower },         .{ .start = 0x04C0, .end = 0x04C1, .cat = .upper_title },
    .{ .start = 0x04C2, .end = 0x04C2, .cat = .lower },         .{ .start = 0x04C3, .end = 0x04C3, .cat = .upper_title },   .{ .start = 0x04C4, .end = 0x04C4, .cat = .lower },
    .{ .start = 0x04C5, .end = 0x04C5, .cat = .upper_title },   .{ .start = 0x04C6, .end = 0x04C6, .cat = .lower },         .{ .start = 0x04C7, .end = 0x04C7, .cat = .upper_title },
    .{ .start = 0x04C8, .end = 0x04C8, .cat = .lower },         .{ .start = 0x04C9, .end = 0x04C9, .cat = .upper_title },   .{ .start = 0x04CA, .end = 0x04CA, .cat = .lower },
    .{ .start = 0x04CB, .end = 0x04CB, .cat = .upper_title },   .{ .start = 0x04CC, .end = 0x04CC, .cat = .lower },         .{ .start = 0x04CD, .end = 0x04CD, .cat = .upper_title },
    .{ .start = 0x04CE, .end = 0x04CF, .cat = .lower },         .{ .start = 0x04D0, .end = 0x04D0, .cat = .upper_title },   .{ .start = 0x04D1, .end = 0x04D1, .cat = .lower },
    .{ .start = 0x04D2, .end = 0x04D2, .cat = .upper_title },   .{ .start = 0x04D3, .end = 0x04D3, .cat = .lower },         .{ .start = 0x04D4, .end = 0x04D4, .cat = .upper_title },
    .{ .start = 0x04D5, .end = 0x04D5, .cat = .lower },         .{ .start = 0x04D6, .end = 0x04D6, .cat = .upper_title },   .{ .start = 0x04D7, .end = 0x04D7, .cat = .lower },
    .{ .start = 0x04D8, .end = 0x04D8, .cat = .upper_title },   .{ .start = 0x04D9, .end = 0x04D9, .cat = .lower },         .{ .start = 0x04DA, .end = 0x04DA, .cat = .upper_title },
    .{ .start = 0x04DB, .end = 0x04DB, .cat = .lower },         .{ .start = 0x04DC, .end = 0x04DC, .cat = .upper_title },   .{ .start = 0x04DD, .end = 0x04DD, .cat = .lower },
    .{ .start = 0x04DE, .end = 0x04DE, .cat = .upper_title },   .{ .start = 0x04DF, .end = 0x04DF, .cat = .lower },         .{ .start = 0x04E0, .end = 0x04E0, .cat = .upper_title },
    .{ .start = 0x04E1, .end = 0x04E1, .cat = .lower },         .{ .start = 0x04E2, .end = 0x04E2, .cat = .upper_title },   .{ .start = 0x04E3, .end = 0x04E3, .cat = .lower },
    .{ .start = 0x04E4, .end = 0x04E4, .cat = .upper_title },   .{ .start = 0x04E5, .end = 0x04E5, .cat = .lower },         .{ .start = 0x04E6, .end = 0x04E6, .cat = .upper_title },
    .{ .start = 0x04E7, .end = 0x04E7, .cat = .lower },         .{ .start = 0x04E8, .end = 0x04E8, .cat = .upper_title },   .{ .start = 0x04E9, .end = 0x04E9, .cat = .lower },
    .{ .start = 0x04EA, .end = 0x04EA, .cat = .upper_title },   .{ .start = 0x04EB, .end = 0x04EB, .cat = .lower },         .{ .start = 0x04EC, .end = 0x04EC, .cat = .upper_title },
    .{ .start = 0x04ED, .end = 0x04ED, .cat = .lower },         .{ .start = 0x04EE, .end = 0x04EE, .cat = .upper_title },   .{ .start = 0x04EF, .end = 0x04EF, .cat = .lower },
    .{ .start = 0x04F0, .end = 0x04F0, .cat = .upper_title },   .{ .start = 0x04F1, .end = 0x04F1, .cat = .lower },         .{ .start = 0x04F2, .end = 0x04F2, .cat = .upper_title },
    .{ .start = 0x04F3, .end = 0x04F3, .cat = .lower },         .{ .start = 0x04F4, .end = 0x04F4, .cat = .upper_title },   .{ .start = 0x04F5, .end = 0x04F5, .cat = .lower },
    .{ .start = 0x04F6, .end = 0x04F6, .cat = .upper_title },   .{ .start = 0x04F7, .end = 0x04F7, .cat = .lower },         .{ .start = 0x04F8, .end = 0x04F8, .cat = .upper_title },
    .{ .start = 0x04F9, .end = 0x04F9, .cat = .lower },         .{ .start = 0x04FA, .end = 0x04FA, .cat = .upper_title },   .{ .start = 0x04FB, .end = 0x04FB, .cat = .lower },
    .{ .start = 0x04FC, .end = 0x04FC, .cat = .upper_title },   .{ .start = 0x04FD, .end = 0x04FD, .cat = .lower },         .{ .start = 0x04FE, .end = 0x04FE, .cat = .upper_title },
    .{ .start = 0x04FF, .end = 0x04FF, .cat = .lower },         .{ .start = 0x0500, .end = 0x0500, .cat = .upper_title },   .{ .start = 0x0501, .end = 0x0501, .cat = .lower },
    .{ .start = 0x0502, .end = 0x0502, .cat = .upper_title },   .{ .start = 0x0503, .end = 0x0503, .cat = .lower },         .{ .start = 0x0504, .end = 0x0504, .cat = .upper_title },
    .{ .start = 0x0505, .end = 0x0505, .cat = .lower },         .{ .start = 0x0506, .end = 0x0506, .cat = .upper_title },   .{ .start = 0x0507, .end = 0x0507, .cat = .lower },
    .{ .start = 0x0508, .end = 0x0508, .cat = .upper_title },   .{ .start = 0x0509, .end = 0x0509, .cat = .lower },         .{ .start = 0x050A, .end = 0x050A, .cat = .upper_title },
    .{ .start = 0x050B, .end = 0x050B, .cat = .lower },         .{ .start = 0x050C, .end = 0x050C, .cat = .upper_title },   .{ .start = 0x050D, .end = 0x050D, .cat = .lower },
    .{ .start = 0x050E, .end = 0x050E, .cat = .upper_title },   .{ .start = 0x050F, .end = 0x050F, .cat = .lower },         .{ .start = 0x0510, .end = 0x0510, .cat = .upper_title },
    .{ .start = 0x0511, .end = 0x0511, .cat = .lower },         .{ .start = 0x0512, .end = 0x0512, .cat = .upper_title },   .{ .start = 0x0513, .end = 0x0513, .cat = .lower },
    .{ .start = 0x0514, .end = 0x0514, .cat = .upper_title },   .{ .start = 0x0515, .end = 0x0515, .cat = .lower },         .{ .start = 0x0516, .end = 0x0516, .cat = .upper_title },
    .{ .start = 0x0517, .end = 0x0517, .cat = .lower },         .{ .start = 0x0518, .end = 0x0518, .cat = .upper_title },   .{ .start = 0x0519, .end = 0x0519, .cat = .lower },
    .{ .start = 0x051A, .end = 0x051A, .cat = .upper_title },   .{ .start = 0x051B, .end = 0x051B, .cat = .lower },         .{ .start = 0x051C, .end = 0x051C, .cat = .upper_title },
    .{ .start = 0x051D, .end = 0x051D, .cat = .lower },         .{ .start = 0x051E, .end = 0x051E, .cat = .upper_title },   .{ .start = 0x051F, .end = 0x051F, .cat = .lower },
    .{ .start = 0x0520, .end = 0x0520, .cat = .upper_title },   .{ .start = 0x0521, .end = 0x0521, .cat = .lower },         .{ .start = 0x0522, .end = 0x0522, .cat = .upper_title },
    .{ .start = 0x0523, .end = 0x0523, .cat = .lower },         .{ .start = 0x0524, .end = 0x0524, .cat = .upper_title },   .{ .start = 0x0525, .end = 0x0525, .cat = .lower },
    .{ .start = 0x0526, .end = 0x0526, .cat = .upper_title },   .{ .start = 0x0527, .end = 0x0527, .cat = .lower },         .{ .start = 0x0528, .end = 0x0528, .cat = .upper_title },
    .{ .start = 0x0529, .end = 0x0529, .cat = .lower },         .{ .start = 0x052A, .end = 0x052A, .cat = .upper_title },   .{ .start = 0x052B, .end = 0x052B, .cat = .lower },
    .{ .start = 0x052C, .end = 0x052C, .cat = .upper_title },   .{ .start = 0x052D, .end = 0x052D, .cat = .lower },         .{ .start = 0x052E, .end = 0x052E, .cat = .upper_title },
    .{ .start = 0x052F, .end = 0x052F, .cat = .lower },         .{ .start = 0x0531, .end = 0x0556, .cat = .upper_title },   .{ .start = 0x0559, .end = 0x0559, .cat = .other },
    .{ .start = 0x0560, .end = 0x0588, .cat = .lower },         .{ .start = 0x05D0, .end = 0x05EA, .cat = .other },         .{ .start = 0x05EF, .end = 0x05F2, .cat = .other },
    .{ .start = 0x0620, .end = 0x064A, .cat = .other },         .{ .start = 0x066E, .end = 0x066F, .cat = .other },         .{ .start = 0x0671, .end = 0x06D3, .cat = .other },
    .{ .start = 0x06D5, .end = 0x06D5, .cat = .other },         .{ .start = 0x06E5, .end = 0x06E6, .cat = .other },         .{ .start = 0x06EE, .end = 0x06EF, .cat = .other },
    .{ .start = 0x06FA, .end = 0x06FC, .cat = .other },         .{ .start = 0x06FF, .end = 0x06FF, .cat = .other },         .{ .start = 0x0710, .end = 0x0710, .cat = .other },
    .{ .start = 0x0712, .end = 0x072F, .cat = .other },         .{ .start = 0x074D, .end = 0x07A5, .cat = .other },         .{ .start = 0x07B1, .end = 0x07B1, .cat = .other },
    .{ .start = 0x07CA, .end = 0x07EA, .cat = .other },         .{ .start = 0x07F4, .end = 0x07F5, .cat = .other },         .{ .start = 0x07FA, .end = 0x07FA, .cat = .other },
    .{ .start = 0x0800, .end = 0x0815, .cat = .other },         .{ .start = 0x081A, .end = 0x081A, .cat = .other },         .{ .start = 0x0824, .end = 0x0824, .cat = .other },
    .{ .start = 0x0828, .end = 0x0828, .cat = .other },         .{ .start = 0x0840, .end = 0x0858, .cat = .other },         .{ .start = 0x0860, .end = 0x086A, .cat = .other },
    .{ .start = 0x0870, .end = 0x0887, .cat = .other },         .{ .start = 0x0889, .end = 0x088F, .cat = .other },         .{ .start = 0x08A0, .end = 0x08C9, .cat = .other },
    .{ .start = 0x0904, .end = 0x0939, .cat = .other },         .{ .start = 0x093D, .end = 0x093D, .cat = .other },         .{ .start = 0x0950, .end = 0x0950, .cat = .other },
    .{ .start = 0x0958, .end = 0x0961, .cat = .other },         .{ .start = 0x0971, .end = 0x0980, .cat = .other },         .{ .start = 0x0985, .end = 0x098C, .cat = .other },
    .{ .start = 0x098F, .end = 0x0990, .cat = .other },         .{ .start = 0x0993, .end = 0x09A8, .cat = .other },         .{ .start = 0x09AA, .end = 0x09B0, .cat = .other },
    .{ .start = 0x09B2, .end = 0x09B2, .cat = .other },         .{ .start = 0x09B6, .end = 0x09B9, .cat = .other },         .{ .start = 0x09BD, .end = 0x09BD, .cat = .other },
    .{ .start = 0x09CE, .end = 0x09CE, .cat = .other },         .{ .start = 0x09DC, .end = 0x09DD, .cat = .other },         .{ .start = 0x09DF, .end = 0x09E1, .cat = .other },
    .{ .start = 0x09F0, .end = 0x09F1, .cat = .other },         .{ .start = 0x09FC, .end = 0x09FC, .cat = .other },         .{ .start = 0x0A05, .end = 0x0A0A, .cat = .other },
    .{ .start = 0x0A0F, .end = 0x0A10, .cat = .other },         .{ .start = 0x0A13, .end = 0x0A28, .cat = .other },         .{ .start = 0x0A2A, .end = 0x0A30, .cat = .other },
    .{ .start = 0x0A32, .end = 0x0A33, .cat = .other },         .{ .start = 0x0A35, .end = 0x0A36, .cat = .other },         .{ .start = 0x0A38, .end = 0x0A39, .cat = .other },
    .{ .start = 0x0A59, .end = 0x0A5C, .cat = .other },         .{ .start = 0x0A5E, .end = 0x0A5E, .cat = .other },         .{ .start = 0x0A72, .end = 0x0A74, .cat = .other },
    .{ .start = 0x0A85, .end = 0x0A8D, .cat = .other },         .{ .start = 0x0A8F, .end = 0x0A91, .cat = .other },         .{ .start = 0x0A93, .end = 0x0AA8, .cat = .other },
    .{ .start = 0x0AAA, .end = 0x0AB0, .cat = .other },         .{ .start = 0x0AB2, .end = 0x0AB3, .cat = .other },         .{ .start = 0x0AB5, .end = 0x0AB9, .cat = .other },
    .{ .start = 0x0ABD, .end = 0x0ABD, .cat = .other },         .{ .start = 0x0AD0, .end = 0x0AD0, .cat = .other },         .{ .start = 0x0AE0, .end = 0x0AE1, .cat = .other },
    .{ .start = 0x0AF9, .end = 0x0AF9, .cat = .other },         .{ .start = 0x0B05, .end = 0x0B0C, .cat = .other },         .{ .start = 0x0B0F, .end = 0x0B10, .cat = .other },
    .{ .start = 0x0B13, .end = 0x0B28, .cat = .other },         .{ .start = 0x0B2A, .end = 0x0B30, .cat = .other },         .{ .start = 0x0B32, .end = 0x0B33, .cat = .other },
    .{ .start = 0x0B35, .end = 0x0B39, .cat = .other },         .{ .start = 0x0B3D, .end = 0x0B3D, .cat = .other },         .{ .start = 0x0B5C, .end = 0x0B5D, .cat = .other },
    .{ .start = 0x0B5F, .end = 0x0B61, .cat = .other },         .{ .start = 0x0B71, .end = 0x0B71, .cat = .other },         .{ .start = 0x0B83, .end = 0x0B83, .cat = .other },
    .{ .start = 0x0B85, .end = 0x0B8A, .cat = .other },         .{ .start = 0x0B8E, .end = 0x0B90, .cat = .other },         .{ .start = 0x0B92, .end = 0x0B95, .cat = .other },
    .{ .start = 0x0B99, .end = 0x0B9A, .cat = .other },         .{ .start = 0x0B9C, .end = 0x0B9C, .cat = .other },         .{ .start = 0x0B9E, .end = 0x0B9F, .cat = .other },
    .{ .start = 0x0BA3, .end = 0x0BA4, .cat = .other },         .{ .start = 0x0BA8, .end = 0x0BAA, .cat = .other },         .{ .start = 0x0BAE, .end = 0x0BB9, .cat = .other },
    .{ .start = 0x0BD0, .end = 0x0BD0, .cat = .other },         .{ .start = 0x0C05, .end = 0x0C0C, .cat = .other },         .{ .start = 0x0C0E, .end = 0x0C10, .cat = .other },
    .{ .start = 0x0C12, .end = 0x0C28, .cat = .other },         .{ .start = 0x0C2A, .end = 0x0C39, .cat = .other },         .{ .start = 0x0C3D, .end = 0x0C3D, .cat = .other },
    .{ .start = 0x0C58, .end = 0x0C5A, .cat = .other },         .{ .start = 0x0C5C, .end = 0x0C5D, .cat = .other },         .{ .start = 0x0C60, .end = 0x0C61, .cat = .other },
    .{ .start = 0x0C80, .end = 0x0C80, .cat = .other },         .{ .start = 0x0C85, .end = 0x0C8C, .cat = .other },         .{ .start = 0x0C8E, .end = 0x0C90, .cat = .other },
    .{ .start = 0x0C92, .end = 0x0CA8, .cat = .other },         .{ .start = 0x0CAA, .end = 0x0CB3, .cat = .other },         .{ .start = 0x0CB5, .end = 0x0CB9, .cat = .other },
    .{ .start = 0x0CBD, .end = 0x0CBD, .cat = .other },         .{ .start = 0x0CDC, .end = 0x0CDE, .cat = .other },         .{ .start = 0x0CE0, .end = 0x0CE1, .cat = .other },
    .{ .start = 0x0CF1, .end = 0x0CF2, .cat = .other },         .{ .start = 0x0D04, .end = 0x0D0C, .cat = .other },         .{ .start = 0x0D0E, .end = 0x0D10, .cat = .other },
    .{ .start = 0x0D12, .end = 0x0D3A, .cat = .other },         .{ .start = 0x0D3D, .end = 0x0D3D, .cat = .other },         .{ .start = 0x0D4E, .end = 0x0D4E, .cat = .other },
    .{ .start = 0x0D54, .end = 0x0D56, .cat = .other },         .{ .start = 0x0D5F, .end = 0x0D61, .cat = .other },         .{ .start = 0x0D7A, .end = 0x0D7F, .cat = .other },
    .{ .start = 0x0D85, .end = 0x0D96, .cat = .other },         .{ .start = 0x0D9A, .end = 0x0DB1, .cat = .other },         .{ .start = 0x0DB3, .end = 0x0DBB, .cat = .other },
    .{ .start = 0x0DBD, .end = 0x0DBD, .cat = .other },         .{ .start = 0x0DC0, .end = 0x0DC6, .cat = .other },         .{ .start = 0x0E01, .end = 0x0E30, .cat = .other },
    .{ .start = 0x0E32, .end = 0x0E33, .cat = .other },         .{ .start = 0x0E40, .end = 0x0E46, .cat = .other },         .{ .start = 0x0E81, .end = 0x0E82, .cat = .other },
    .{ .start = 0x0E84, .end = 0x0E84, .cat = .other },         .{ .start = 0x0E86, .end = 0x0E8A, .cat = .other },         .{ .start = 0x0E8C, .end = 0x0EA3, .cat = .other },
    .{ .start = 0x0EA5, .end = 0x0EA5, .cat = .other },         .{ .start = 0x0EA7, .end = 0x0EB0, .cat = .other },         .{ .start = 0x0EB2, .end = 0x0EB3, .cat = .other },
    .{ .start = 0x0EBD, .end = 0x0EBD, .cat = .other },         .{ .start = 0x0EC0, .end = 0x0EC4, .cat = .other },         .{ .start = 0x0EC6, .end = 0x0EC6, .cat = .other },
    .{ .start = 0x0EDC, .end = 0x0EDF, .cat = .other },         .{ .start = 0x0F00, .end = 0x0F00, .cat = .other },         .{ .start = 0x0F40, .end = 0x0F47, .cat = .other },
    .{ .start = 0x0F49, .end = 0x0F6C, .cat = .other },         .{ .start = 0x0F88, .end = 0x0F8C, .cat = .other },         .{ .start = 0x1000, .end = 0x102A, .cat = .other },
    .{ .start = 0x103F, .end = 0x103F, .cat = .other },         .{ .start = 0x1050, .end = 0x1055, .cat = .other },         .{ .start = 0x105A, .end = 0x105D, .cat = .other },
    .{ .start = 0x1061, .end = 0x1061, .cat = .other },         .{ .start = 0x1065, .end = 0x1066, .cat = .other },         .{ .start = 0x106E, .end = 0x1070, .cat = .other },
    .{ .start = 0x1075, .end = 0x1081, .cat = .other },         .{ .start = 0x108E, .end = 0x108E, .cat = .other },         .{ .start = 0x10A0, .end = 0x10C5, .cat = .upper_title },
    .{ .start = 0x10C7, .end = 0x10C7, .cat = .upper_title },   .{ .start = 0x10CD, .end = 0x10CD, .cat = .upper_title },   .{ .start = 0x10D0, .end = 0x10FA, .cat = .lower },
    .{ .start = 0x10FC, .end = 0x10FC, .cat = .other },         .{ .start = 0x10FD, .end = 0x10FF, .cat = .lower },         .{ .start = 0x1100, .end = 0x1248, .cat = .other },
    .{ .start = 0x124A, .end = 0x124D, .cat = .other },         .{ .start = 0x1250, .end = 0x1256, .cat = .other },         .{ .start = 0x1258, .end = 0x1258, .cat = .other },
    .{ .start = 0x125A, .end = 0x125D, .cat = .other },         .{ .start = 0x1260, .end = 0x1288, .cat = .other },         .{ .start = 0x128A, .end = 0x128D, .cat = .other },
    .{ .start = 0x1290, .end = 0x12B0, .cat = .other },         .{ .start = 0x12B2, .end = 0x12B5, .cat = .other },         .{ .start = 0x12B8, .end = 0x12BE, .cat = .other },
    .{ .start = 0x12C0, .end = 0x12C0, .cat = .other },         .{ .start = 0x12C2, .end = 0x12C5, .cat = .other },         .{ .start = 0x12C8, .end = 0x12D6, .cat = .other },
    .{ .start = 0x12D8, .end = 0x1310, .cat = .other },         .{ .start = 0x1312, .end = 0x1315, .cat = .other },         .{ .start = 0x1318, .end = 0x135A, .cat = .other },
    .{ .start = 0x1380, .end = 0x138F, .cat = .other },         .{ .start = 0x13A0, .end = 0x13F5, .cat = .upper_title },   .{ .start = 0x13F8, .end = 0x13FD, .cat = .lower },
    .{ .start = 0x1401, .end = 0x166C, .cat = .other },         .{ .start = 0x166F, .end = 0x167F, .cat = .other },         .{ .start = 0x1681, .end = 0x169A, .cat = .other },
    .{ .start = 0x16A0, .end = 0x16EA, .cat = .other },         .{ .start = 0x16F1, .end = 0x16F8, .cat = .other },         .{ .start = 0x1700, .end = 0x1711, .cat = .other },
    .{ .start = 0x171F, .end = 0x1731, .cat = .other },         .{ .start = 0x1740, .end = 0x1751, .cat = .other },         .{ .start = 0x1760, .end = 0x176C, .cat = .other },
    .{ .start = 0x176E, .end = 0x1770, .cat = .other },         .{ .start = 0x1780, .end = 0x17B3, .cat = .other },         .{ .start = 0x17D7, .end = 0x17D7, .cat = .other },
    .{ .start = 0x17DC, .end = 0x17DC, .cat = .other },         .{ .start = 0x1820, .end = 0x1878, .cat = .other },         .{ .start = 0x1880, .end = 0x1884, .cat = .other },
    .{ .start = 0x1887, .end = 0x18A8, .cat = .other },         .{ .start = 0x18AA, .end = 0x18AA, .cat = .other },         .{ .start = 0x18B0, .end = 0x18F5, .cat = .other },
    .{ .start = 0x1900, .end = 0x191E, .cat = .other },         .{ .start = 0x1950, .end = 0x196D, .cat = .other },         .{ .start = 0x1970, .end = 0x1974, .cat = .other },
    .{ .start = 0x1980, .end = 0x19AB, .cat = .other },         .{ .start = 0x19B0, .end = 0x19C9, .cat = .other },         .{ .start = 0x1A00, .end = 0x1A16, .cat = .other },
    .{ .start = 0x1A20, .end = 0x1A54, .cat = .other },         .{ .start = 0x1AA7, .end = 0x1AA7, .cat = .other },         .{ .start = 0x1B05, .end = 0x1B33, .cat = .other },
    .{ .start = 0x1B45, .end = 0x1B4C, .cat = .other },         .{ .start = 0x1B83, .end = 0x1BA0, .cat = .other },         .{ .start = 0x1BAE, .end = 0x1BAF, .cat = .other },
    .{ .start = 0x1BBA, .end = 0x1BE5, .cat = .other },         .{ .start = 0x1C00, .end = 0x1C23, .cat = .other },         .{ .start = 0x1C4D, .end = 0x1C4F, .cat = .other },
    .{ .start = 0x1C5A, .end = 0x1C7D, .cat = .other },         .{ .start = 0x1C80, .end = 0x1C88, .cat = .lower },         .{ .start = 0x1C89, .end = 0x1C89, .cat = .upper_title },
    .{ .start = 0x1C8A, .end = 0x1C8A, .cat = .lower },         .{ .start = 0x1C90, .end = 0x1CBA, .cat = .upper_title },   .{ .start = 0x1CBD, .end = 0x1CBF, .cat = .upper_title },
    .{ .start = 0x1CE9, .end = 0x1CEC, .cat = .other },         .{ .start = 0x1CEE, .end = 0x1CF3, .cat = .other },         .{ .start = 0x1CF5, .end = 0x1CF6, .cat = .other },
    .{ .start = 0x1CFA, .end = 0x1CFA, .cat = .other },         .{ .start = 0x1D00, .end = 0x1D2B, .cat = .lower },         .{ .start = 0x1D2C, .end = 0x1D6A, .cat = .other },
    .{ .start = 0x1D6B, .end = 0x1D77, .cat = .lower },         .{ .start = 0x1D78, .end = 0x1D78, .cat = .other },         .{ .start = 0x1D79, .end = 0x1D9A, .cat = .lower },
    .{ .start = 0x1D9B, .end = 0x1DBF, .cat = .other },         .{ .start = 0x1E00, .end = 0x1E00, .cat = .upper_title },   .{ .start = 0x1E01, .end = 0x1E01, .cat = .lower },
    .{ .start = 0x1E02, .end = 0x1E02, .cat = .upper_title },   .{ .start = 0x1E03, .end = 0x1E03, .cat = .lower },         .{ .start = 0x1E04, .end = 0x1E04, .cat = .upper_title },
    .{ .start = 0x1E05, .end = 0x1E05, .cat = .lower },         .{ .start = 0x1E06, .end = 0x1E06, .cat = .upper_title },   .{ .start = 0x1E07, .end = 0x1E07, .cat = .lower },
    .{ .start = 0x1E08, .end = 0x1E08, .cat = .upper_title },   .{ .start = 0x1E09, .end = 0x1E09, .cat = .lower },         .{ .start = 0x1E0A, .end = 0x1E0A, .cat = .upper_title },
    .{ .start = 0x1E0B, .end = 0x1E0B, .cat = .lower },         .{ .start = 0x1E0C, .end = 0x1E0C, .cat = .upper_title },   .{ .start = 0x1E0D, .end = 0x1E0D, .cat = .lower },
    .{ .start = 0x1E0E, .end = 0x1E0E, .cat = .upper_title },   .{ .start = 0x1E0F, .end = 0x1E0F, .cat = .lower },         .{ .start = 0x1E10, .end = 0x1E10, .cat = .upper_title },
    .{ .start = 0x1E11, .end = 0x1E11, .cat = .lower },         .{ .start = 0x1E12, .end = 0x1E12, .cat = .upper_title },   .{ .start = 0x1E13, .end = 0x1E13, .cat = .lower },
    .{ .start = 0x1E14, .end = 0x1E14, .cat = .upper_title },   .{ .start = 0x1E15, .end = 0x1E15, .cat = .lower },         .{ .start = 0x1E16, .end = 0x1E16, .cat = .upper_title },
    .{ .start = 0x1E17, .end = 0x1E17, .cat = .lower },         .{ .start = 0x1E18, .end = 0x1E18, .cat = .upper_title },   .{ .start = 0x1E19, .end = 0x1E19, .cat = .lower },
    .{ .start = 0x1E1A, .end = 0x1E1A, .cat = .upper_title },   .{ .start = 0x1E1B, .end = 0x1E1B, .cat = .lower },         .{ .start = 0x1E1C, .end = 0x1E1C, .cat = .upper_title },
    .{ .start = 0x1E1D, .end = 0x1E1D, .cat = .lower },         .{ .start = 0x1E1E, .end = 0x1E1E, .cat = .upper_title },   .{ .start = 0x1E1F, .end = 0x1E1F, .cat = .lower },
    .{ .start = 0x1E20, .end = 0x1E20, .cat = .upper_title },   .{ .start = 0x1E21, .end = 0x1E21, .cat = .lower },         .{ .start = 0x1E22, .end = 0x1E22, .cat = .upper_title },
    .{ .start = 0x1E23, .end = 0x1E23, .cat = .lower },         .{ .start = 0x1E24, .end = 0x1E24, .cat = .upper_title },   .{ .start = 0x1E25, .end = 0x1E25, .cat = .lower },
    .{ .start = 0x1E26, .end = 0x1E26, .cat = .upper_title },   .{ .start = 0x1E27, .end = 0x1E27, .cat = .lower },         .{ .start = 0x1E28, .end = 0x1E28, .cat = .upper_title },
    .{ .start = 0x1E29, .end = 0x1E29, .cat = .lower },         .{ .start = 0x1E2A, .end = 0x1E2A, .cat = .upper_title },   .{ .start = 0x1E2B, .end = 0x1E2B, .cat = .lower },
    .{ .start = 0x1E2C, .end = 0x1E2C, .cat = .upper_title },   .{ .start = 0x1E2D, .end = 0x1E2D, .cat = .lower },         .{ .start = 0x1E2E, .end = 0x1E2E, .cat = .upper_title },
    .{ .start = 0x1E2F, .end = 0x1E2F, .cat = .lower },         .{ .start = 0x1E30, .end = 0x1E30, .cat = .upper_title },   .{ .start = 0x1E31, .end = 0x1E31, .cat = .lower },
    .{ .start = 0x1E32, .end = 0x1E32, .cat = .upper_title },   .{ .start = 0x1E33, .end = 0x1E33, .cat = .lower },         .{ .start = 0x1E34, .end = 0x1E34, .cat = .upper_title },
    .{ .start = 0x1E35, .end = 0x1E35, .cat = .lower },         .{ .start = 0x1E36, .end = 0x1E36, .cat = .upper_title },   .{ .start = 0x1E37, .end = 0x1E37, .cat = .lower },
    .{ .start = 0x1E38, .end = 0x1E38, .cat = .upper_title },   .{ .start = 0x1E39, .end = 0x1E39, .cat = .lower },         .{ .start = 0x1E3A, .end = 0x1E3A, .cat = .upper_title },
    .{ .start = 0x1E3B, .end = 0x1E3B, .cat = .lower },         .{ .start = 0x1E3C, .end = 0x1E3C, .cat = .upper_title },   .{ .start = 0x1E3D, .end = 0x1E3D, .cat = .lower },
    .{ .start = 0x1E3E, .end = 0x1E3E, .cat = .upper_title },   .{ .start = 0x1E3F, .end = 0x1E3F, .cat = .lower },         .{ .start = 0x1E40, .end = 0x1E40, .cat = .upper_title },
    .{ .start = 0x1E41, .end = 0x1E41, .cat = .lower },         .{ .start = 0x1E42, .end = 0x1E42, .cat = .upper_title },   .{ .start = 0x1E43, .end = 0x1E43, .cat = .lower },
    .{ .start = 0x1E44, .end = 0x1E44, .cat = .upper_title },   .{ .start = 0x1E45, .end = 0x1E45, .cat = .lower },         .{ .start = 0x1E46, .end = 0x1E46, .cat = .upper_title },
    .{ .start = 0x1E47, .end = 0x1E47, .cat = .lower },         .{ .start = 0x1E48, .end = 0x1E48, .cat = .upper_title },   .{ .start = 0x1E49, .end = 0x1E49, .cat = .lower },
    .{ .start = 0x1E4A, .end = 0x1E4A, .cat = .upper_title },   .{ .start = 0x1E4B, .end = 0x1E4B, .cat = .lower },         .{ .start = 0x1E4C, .end = 0x1E4C, .cat = .upper_title },
    .{ .start = 0x1E4D, .end = 0x1E4D, .cat = .lower },         .{ .start = 0x1E4E, .end = 0x1E4E, .cat = .upper_title },   .{ .start = 0x1E4F, .end = 0x1E4F, .cat = .lower },
    .{ .start = 0x1E50, .end = 0x1E50, .cat = .upper_title },   .{ .start = 0x1E51, .end = 0x1E51, .cat = .lower },         .{ .start = 0x1E52, .end = 0x1E52, .cat = .upper_title },
    .{ .start = 0x1E53, .end = 0x1E53, .cat = .lower },         .{ .start = 0x1E54, .end = 0x1E54, .cat = .upper_title },   .{ .start = 0x1E55, .end = 0x1E55, .cat = .lower },
    .{ .start = 0x1E56, .end = 0x1E56, .cat = .upper_title },   .{ .start = 0x1E57, .end = 0x1E57, .cat = .lower },         .{ .start = 0x1E58, .end = 0x1E58, .cat = .upper_title },
    .{ .start = 0x1E59, .end = 0x1E59, .cat = .lower },         .{ .start = 0x1E5A, .end = 0x1E5A, .cat = .upper_title },   .{ .start = 0x1E5B, .end = 0x1E5B, .cat = .lower },
    .{ .start = 0x1E5C, .end = 0x1E5C, .cat = .upper_title },   .{ .start = 0x1E5D, .end = 0x1E5D, .cat = .lower },         .{ .start = 0x1E5E, .end = 0x1E5E, .cat = .upper_title },
    .{ .start = 0x1E5F, .end = 0x1E5F, .cat = .lower },         .{ .start = 0x1E60, .end = 0x1E60, .cat = .upper_title },   .{ .start = 0x1E61, .end = 0x1E61, .cat = .lower },
    .{ .start = 0x1E62, .end = 0x1E62, .cat = .upper_title },   .{ .start = 0x1E63, .end = 0x1E63, .cat = .lower },         .{ .start = 0x1E64, .end = 0x1E64, .cat = .upper_title },
    .{ .start = 0x1E65, .end = 0x1E65, .cat = .lower },         .{ .start = 0x1E66, .end = 0x1E66, .cat = .upper_title },   .{ .start = 0x1E67, .end = 0x1E67, .cat = .lower },
    .{ .start = 0x1E68, .end = 0x1E68, .cat = .upper_title },   .{ .start = 0x1E69, .end = 0x1E69, .cat = .lower },         .{ .start = 0x1E6A, .end = 0x1E6A, .cat = .upper_title },
    .{ .start = 0x1E6B, .end = 0x1E6B, .cat = .lower },         .{ .start = 0x1E6C, .end = 0x1E6C, .cat = .upper_title },   .{ .start = 0x1E6D, .end = 0x1E6D, .cat = .lower },
    .{ .start = 0x1E6E, .end = 0x1E6E, .cat = .upper_title },   .{ .start = 0x1E6F, .end = 0x1E6F, .cat = .lower },         .{ .start = 0x1E70, .end = 0x1E70, .cat = .upper_title },
    .{ .start = 0x1E71, .end = 0x1E71, .cat = .lower },         .{ .start = 0x1E72, .end = 0x1E72, .cat = .upper_title },   .{ .start = 0x1E73, .end = 0x1E73, .cat = .lower },
    .{ .start = 0x1E74, .end = 0x1E74, .cat = .upper_title },   .{ .start = 0x1E75, .end = 0x1E75, .cat = .lower },         .{ .start = 0x1E76, .end = 0x1E76, .cat = .upper_title },
    .{ .start = 0x1E77, .end = 0x1E77, .cat = .lower },         .{ .start = 0x1E78, .end = 0x1E78, .cat = .upper_title },   .{ .start = 0x1E79, .end = 0x1E79, .cat = .lower },
    .{ .start = 0x1E7A, .end = 0x1E7A, .cat = .upper_title },   .{ .start = 0x1E7B, .end = 0x1E7B, .cat = .lower },         .{ .start = 0x1E7C, .end = 0x1E7C, .cat = .upper_title },
    .{ .start = 0x1E7D, .end = 0x1E7D, .cat = .lower },         .{ .start = 0x1E7E, .end = 0x1E7E, .cat = .upper_title },   .{ .start = 0x1E7F, .end = 0x1E7F, .cat = .lower },
    .{ .start = 0x1E80, .end = 0x1E80, .cat = .upper_title },   .{ .start = 0x1E81, .end = 0x1E81, .cat = .lower },         .{ .start = 0x1E82, .end = 0x1E82, .cat = .upper_title },
    .{ .start = 0x1E83, .end = 0x1E83, .cat = .lower },         .{ .start = 0x1E84, .end = 0x1E84, .cat = .upper_title },   .{ .start = 0x1E85, .end = 0x1E85, .cat = .lower },
    .{ .start = 0x1E86, .end = 0x1E86, .cat = .upper_title },   .{ .start = 0x1E87, .end = 0x1E87, .cat = .lower },         .{ .start = 0x1E88, .end = 0x1E88, .cat = .upper_title },
    .{ .start = 0x1E89, .end = 0x1E89, .cat = .lower },         .{ .start = 0x1E8A, .end = 0x1E8A, .cat = .upper_title },   .{ .start = 0x1E8B, .end = 0x1E8B, .cat = .lower },
    .{ .start = 0x1E8C, .end = 0x1E8C, .cat = .upper_title },   .{ .start = 0x1E8D, .end = 0x1E8D, .cat = .lower },         .{ .start = 0x1E8E, .end = 0x1E8E, .cat = .upper_title },
    .{ .start = 0x1E8F, .end = 0x1E8F, .cat = .lower },         .{ .start = 0x1E90, .end = 0x1E90, .cat = .upper_title },   .{ .start = 0x1E91, .end = 0x1E91, .cat = .lower },
    .{ .start = 0x1E92, .end = 0x1E92, .cat = .upper_title },   .{ .start = 0x1E93, .end = 0x1E93, .cat = .lower },         .{ .start = 0x1E94, .end = 0x1E94, .cat = .upper_title },
    .{ .start = 0x1E95, .end = 0x1E9D, .cat = .lower },         .{ .start = 0x1E9E, .end = 0x1E9E, .cat = .upper_title },   .{ .start = 0x1E9F, .end = 0x1E9F, .cat = .lower },
    .{ .start = 0x1EA0, .end = 0x1EA0, .cat = .upper_title },   .{ .start = 0x1EA1, .end = 0x1EA1, .cat = .lower },         .{ .start = 0x1EA2, .end = 0x1EA2, .cat = .upper_title },
    .{ .start = 0x1EA3, .end = 0x1EA3, .cat = .lower },         .{ .start = 0x1EA4, .end = 0x1EA4, .cat = .upper_title },   .{ .start = 0x1EA5, .end = 0x1EA5, .cat = .lower },
    .{ .start = 0x1EA6, .end = 0x1EA6, .cat = .upper_title },   .{ .start = 0x1EA7, .end = 0x1EA7, .cat = .lower },         .{ .start = 0x1EA8, .end = 0x1EA8, .cat = .upper_title },
    .{ .start = 0x1EA9, .end = 0x1EA9, .cat = .lower },         .{ .start = 0x1EAA, .end = 0x1EAA, .cat = .upper_title },   .{ .start = 0x1EAB, .end = 0x1EAB, .cat = .lower },
    .{ .start = 0x1EAC, .end = 0x1EAC, .cat = .upper_title },   .{ .start = 0x1EAD, .end = 0x1EAD, .cat = .lower },         .{ .start = 0x1EAE, .end = 0x1EAE, .cat = .upper_title },
    .{ .start = 0x1EAF, .end = 0x1EAF, .cat = .lower },         .{ .start = 0x1EB0, .end = 0x1EB0, .cat = .upper_title },   .{ .start = 0x1EB1, .end = 0x1EB1, .cat = .lower },
    .{ .start = 0x1EB2, .end = 0x1EB2, .cat = .upper_title },   .{ .start = 0x1EB3, .end = 0x1EB3, .cat = .lower },         .{ .start = 0x1EB4, .end = 0x1EB4, .cat = .upper_title },
    .{ .start = 0x1EB5, .end = 0x1EB5, .cat = .lower },         .{ .start = 0x1EB6, .end = 0x1EB6, .cat = .upper_title },   .{ .start = 0x1EB7, .end = 0x1EB7, .cat = .lower },
    .{ .start = 0x1EB8, .end = 0x1EB8, .cat = .upper_title },   .{ .start = 0x1EB9, .end = 0x1EB9, .cat = .lower },         .{ .start = 0x1EBA, .end = 0x1EBA, .cat = .upper_title },
    .{ .start = 0x1EBB, .end = 0x1EBB, .cat = .lower },         .{ .start = 0x1EBC, .end = 0x1EBC, .cat = .upper_title },   .{ .start = 0x1EBD, .end = 0x1EBD, .cat = .lower },
    .{ .start = 0x1EBE, .end = 0x1EBE, .cat = .upper_title },   .{ .start = 0x1EBF, .end = 0x1EBF, .cat = .lower },         .{ .start = 0x1EC0, .end = 0x1EC0, .cat = .upper_title },
    .{ .start = 0x1EC1, .end = 0x1EC1, .cat = .lower },         .{ .start = 0x1EC2, .end = 0x1EC2, .cat = .upper_title },   .{ .start = 0x1EC3, .end = 0x1EC3, .cat = .lower },
    .{ .start = 0x1EC4, .end = 0x1EC4, .cat = .upper_title },   .{ .start = 0x1EC5, .end = 0x1EC5, .cat = .lower },         .{ .start = 0x1EC6, .end = 0x1EC6, .cat = .upper_title },
    .{ .start = 0x1EC7, .end = 0x1EC7, .cat = .lower },         .{ .start = 0x1EC8, .end = 0x1EC8, .cat = .upper_title },   .{ .start = 0x1EC9, .end = 0x1EC9, .cat = .lower },
    .{ .start = 0x1ECA, .end = 0x1ECA, .cat = .upper_title },   .{ .start = 0x1ECB, .end = 0x1ECB, .cat = .lower },         .{ .start = 0x1ECC, .end = 0x1ECC, .cat = .upper_title },
    .{ .start = 0x1ECD, .end = 0x1ECD, .cat = .lower },         .{ .start = 0x1ECE, .end = 0x1ECE, .cat = .upper_title },   .{ .start = 0x1ECF, .end = 0x1ECF, .cat = .lower },
    .{ .start = 0x1ED0, .end = 0x1ED0, .cat = .upper_title },   .{ .start = 0x1ED1, .end = 0x1ED1, .cat = .lower },         .{ .start = 0x1ED2, .end = 0x1ED2, .cat = .upper_title },
    .{ .start = 0x1ED3, .end = 0x1ED3, .cat = .lower },         .{ .start = 0x1ED4, .end = 0x1ED4, .cat = .upper_title },   .{ .start = 0x1ED5, .end = 0x1ED5, .cat = .lower },
    .{ .start = 0x1ED6, .end = 0x1ED6, .cat = .upper_title },   .{ .start = 0x1ED7, .end = 0x1ED7, .cat = .lower },         .{ .start = 0x1ED8, .end = 0x1ED8, .cat = .upper_title },
    .{ .start = 0x1ED9, .end = 0x1ED9, .cat = .lower },         .{ .start = 0x1EDA, .end = 0x1EDA, .cat = .upper_title },   .{ .start = 0x1EDB, .end = 0x1EDB, .cat = .lower },
    .{ .start = 0x1EDC, .end = 0x1EDC, .cat = .upper_title },   .{ .start = 0x1EDD, .end = 0x1EDD, .cat = .lower },         .{ .start = 0x1EDE, .end = 0x1EDE, .cat = .upper_title },
    .{ .start = 0x1EDF, .end = 0x1EDF, .cat = .lower },         .{ .start = 0x1EE0, .end = 0x1EE0, .cat = .upper_title },   .{ .start = 0x1EE1, .end = 0x1EE1, .cat = .lower },
    .{ .start = 0x1EE2, .end = 0x1EE2, .cat = .upper_title },   .{ .start = 0x1EE3, .end = 0x1EE3, .cat = .lower },         .{ .start = 0x1EE4, .end = 0x1EE4, .cat = .upper_title },
    .{ .start = 0x1EE5, .end = 0x1EE5, .cat = .lower },         .{ .start = 0x1EE6, .end = 0x1EE6, .cat = .upper_title },   .{ .start = 0x1EE7, .end = 0x1EE7, .cat = .lower },
    .{ .start = 0x1EE8, .end = 0x1EE8, .cat = .upper_title },   .{ .start = 0x1EE9, .end = 0x1EE9, .cat = .lower },         .{ .start = 0x1EEA, .end = 0x1EEA, .cat = .upper_title },
    .{ .start = 0x1EEB, .end = 0x1EEB, .cat = .lower },         .{ .start = 0x1EEC, .end = 0x1EEC, .cat = .upper_title },   .{ .start = 0x1EED, .end = 0x1EED, .cat = .lower },
    .{ .start = 0x1EEE, .end = 0x1EEE, .cat = .upper_title },   .{ .start = 0x1EEF, .end = 0x1EEF, .cat = .lower },         .{ .start = 0x1EF0, .end = 0x1EF0, .cat = .upper_title },
    .{ .start = 0x1EF1, .end = 0x1EF1, .cat = .lower },         .{ .start = 0x1EF2, .end = 0x1EF2, .cat = .upper_title },   .{ .start = 0x1EF3, .end = 0x1EF3, .cat = .lower },
    .{ .start = 0x1EF4, .end = 0x1EF4, .cat = .upper_title },   .{ .start = 0x1EF5, .end = 0x1EF5, .cat = .lower },         .{ .start = 0x1EF6, .end = 0x1EF6, .cat = .upper_title },
    .{ .start = 0x1EF7, .end = 0x1EF7, .cat = .lower },         .{ .start = 0x1EF8, .end = 0x1EF8, .cat = .upper_title },   .{ .start = 0x1EF9, .end = 0x1EF9, .cat = .lower },
    .{ .start = 0x1EFA, .end = 0x1EFA, .cat = .upper_title },   .{ .start = 0x1EFB, .end = 0x1EFB, .cat = .lower },         .{ .start = 0x1EFC, .end = 0x1EFC, .cat = .upper_title },
    .{ .start = 0x1EFD, .end = 0x1EFD, .cat = .lower },         .{ .start = 0x1EFE, .end = 0x1EFE, .cat = .upper_title },   .{ .start = 0x1EFF, .end = 0x1F07, .cat = .lower },
    .{ .start = 0x1F08, .end = 0x1F0F, .cat = .upper_title },   .{ .start = 0x1F10, .end = 0x1F15, .cat = .lower },         .{ .start = 0x1F18, .end = 0x1F1D, .cat = .upper_title },
    .{ .start = 0x1F20, .end = 0x1F27, .cat = .lower },         .{ .start = 0x1F28, .end = 0x1F2F, .cat = .upper_title },   .{ .start = 0x1F30, .end = 0x1F37, .cat = .lower },
    .{ .start = 0x1F38, .end = 0x1F3F, .cat = .upper_title },   .{ .start = 0x1F40, .end = 0x1F45, .cat = .lower },         .{ .start = 0x1F48, .end = 0x1F4D, .cat = .upper_title },
    .{ .start = 0x1F50, .end = 0x1F57, .cat = .lower },         .{ .start = 0x1F59, .end = 0x1F59, .cat = .upper_title },   .{ .start = 0x1F5B, .end = 0x1F5B, .cat = .upper_title },
    .{ .start = 0x1F5D, .end = 0x1F5D, .cat = .upper_title },   .{ .start = 0x1F5F, .end = 0x1F5F, .cat = .upper_title },   .{ .start = 0x1F60, .end = 0x1F67, .cat = .lower },
    .{ .start = 0x1F68, .end = 0x1F6F, .cat = .upper_title },   .{ .start = 0x1F70, .end = 0x1F7D, .cat = .lower },         .{ .start = 0x1F80, .end = 0x1F87, .cat = .lower },
    .{ .start = 0x1F88, .end = 0x1F8F, .cat = .upper_title },   .{ .start = 0x1F90, .end = 0x1F97, .cat = .lower },         .{ .start = 0x1F98, .end = 0x1F9F, .cat = .upper_title },
    .{ .start = 0x1FA0, .end = 0x1FA7, .cat = .lower },         .{ .start = 0x1FA8, .end = 0x1FAF, .cat = .upper_title },   .{ .start = 0x1FB0, .end = 0x1FB4, .cat = .lower },
    .{ .start = 0x1FB6, .end = 0x1FB7, .cat = .lower },         .{ .start = 0x1FB8, .end = 0x1FBC, .cat = .upper_title },   .{ .start = 0x1FBE, .end = 0x1FBE, .cat = .lower },
    .{ .start = 0x1FC2, .end = 0x1FC4, .cat = .lower },         .{ .start = 0x1FC6, .end = 0x1FC7, .cat = .lower },         .{ .start = 0x1FC8, .end = 0x1FCC, .cat = .upper_title },
    .{ .start = 0x1FD0, .end = 0x1FD3, .cat = .lower },         .{ .start = 0x1FD6, .end = 0x1FD7, .cat = .lower },         .{ .start = 0x1FD8, .end = 0x1FDB, .cat = .upper_title },
    .{ .start = 0x1FE0, .end = 0x1FE7, .cat = .lower },         .{ .start = 0x1FE8, .end = 0x1FEC, .cat = .upper_title },   .{ .start = 0x1FF2, .end = 0x1FF4, .cat = .lower },
    .{ .start = 0x1FF6, .end = 0x1FF7, .cat = .lower },         .{ .start = 0x1FF8, .end = 0x1FFC, .cat = .upper_title },   .{ .start = 0x2071, .end = 0x2071, .cat = .other },
    .{ .start = 0x207F, .end = 0x207F, .cat = .other },         .{ .start = 0x2090, .end = 0x209C, .cat = .other },         .{ .start = 0x2102, .end = 0x2102, .cat = .upper_title },
    .{ .start = 0x2107, .end = 0x2107, .cat = .upper_title },   .{ .start = 0x210A, .end = 0x210A, .cat = .lower },         .{ .start = 0x210B, .end = 0x210D, .cat = .upper_title },
    .{ .start = 0x210E, .end = 0x210F, .cat = .lower },         .{ .start = 0x2110, .end = 0x2112, .cat = .upper_title },   .{ .start = 0x2113, .end = 0x2113, .cat = .lower },
    .{ .start = 0x2115, .end = 0x2115, .cat = .upper_title },   .{ .start = 0x2119, .end = 0x211D, .cat = .upper_title },   .{ .start = 0x2124, .end = 0x2124, .cat = .upper_title },
    .{ .start = 0x2126, .end = 0x2126, .cat = .upper_title },   .{ .start = 0x2128, .end = 0x2128, .cat = .upper_title },   .{ .start = 0x212A, .end = 0x212D, .cat = .upper_title },
    .{ .start = 0x212F, .end = 0x212F, .cat = .lower },         .{ .start = 0x2130, .end = 0x2133, .cat = .upper_title },   .{ .start = 0x2134, .end = 0x2134, .cat = .lower },
    .{ .start = 0x2135, .end = 0x2138, .cat = .other },         .{ .start = 0x2139, .end = 0x2139, .cat = .lower },         .{ .start = 0x213C, .end = 0x213D, .cat = .lower },
    .{ .start = 0x213E, .end = 0x213F, .cat = .upper_title },   .{ .start = 0x2145, .end = 0x2145, .cat = .upper_title },   .{ .start = 0x2146, .end = 0x2149, .cat = .lower },
    .{ .start = 0x214E, .end = 0x214E, .cat = .lower },         .{ .start = 0x2183, .end = 0x2183, .cat = .upper_title },   .{ .start = 0x2184, .end = 0x2184, .cat = .lower },
    .{ .start = 0x2C00, .end = 0x2C2F, .cat = .upper_title },   .{ .start = 0x2C30, .end = 0x2C5F, .cat = .lower },         .{ .start = 0x2C60, .end = 0x2C60, .cat = .upper_title },
    .{ .start = 0x2C61, .end = 0x2C61, .cat = .lower },         .{ .start = 0x2C62, .end = 0x2C64, .cat = .upper_title },   .{ .start = 0x2C65, .end = 0x2C66, .cat = .lower },
    .{ .start = 0x2C67, .end = 0x2C67, .cat = .upper_title },   .{ .start = 0x2C68, .end = 0x2C68, .cat = .lower },         .{ .start = 0x2C69, .end = 0x2C69, .cat = .upper_title },
    .{ .start = 0x2C6A, .end = 0x2C6A, .cat = .lower },         .{ .start = 0x2C6B, .end = 0x2C6B, .cat = .upper_title },   .{ .start = 0x2C6C, .end = 0x2C6C, .cat = .lower },
    .{ .start = 0x2C6D, .end = 0x2C70, .cat = .upper_title },   .{ .start = 0x2C71, .end = 0x2C71, .cat = .lower },         .{ .start = 0x2C72, .end = 0x2C72, .cat = .upper_title },
    .{ .start = 0x2C73, .end = 0x2C74, .cat = .lower },         .{ .start = 0x2C75, .end = 0x2C75, .cat = .upper_title },   .{ .start = 0x2C76, .end = 0x2C7B, .cat = .lower },
    .{ .start = 0x2C7C, .end = 0x2C7D, .cat = .other },         .{ .start = 0x2C7E, .end = 0x2C80, .cat = .upper_title },   .{ .start = 0x2C81, .end = 0x2C81, .cat = .lower },
    .{ .start = 0x2C82, .end = 0x2C82, .cat = .upper_title },   .{ .start = 0x2C83, .end = 0x2C83, .cat = .lower },         .{ .start = 0x2C84, .end = 0x2C84, .cat = .upper_title },
    .{ .start = 0x2C85, .end = 0x2C85, .cat = .lower },         .{ .start = 0x2C86, .end = 0x2C86, .cat = .upper_title },   .{ .start = 0x2C87, .end = 0x2C87, .cat = .lower },
    .{ .start = 0x2C88, .end = 0x2C88, .cat = .upper_title },   .{ .start = 0x2C89, .end = 0x2C89, .cat = .lower },         .{ .start = 0x2C8A, .end = 0x2C8A, .cat = .upper_title },
    .{ .start = 0x2C8B, .end = 0x2C8B, .cat = .lower },         .{ .start = 0x2C8C, .end = 0x2C8C, .cat = .upper_title },   .{ .start = 0x2C8D, .end = 0x2C8D, .cat = .lower },
    .{ .start = 0x2C8E, .end = 0x2C8E, .cat = .upper_title },   .{ .start = 0x2C8F, .end = 0x2C8F, .cat = .lower },         .{ .start = 0x2C90, .end = 0x2C90, .cat = .upper_title },
    .{ .start = 0x2C91, .end = 0x2C91, .cat = .lower },         .{ .start = 0x2C92, .end = 0x2C92, .cat = .upper_title },   .{ .start = 0x2C93, .end = 0x2C93, .cat = .lower },
    .{ .start = 0x2C94, .end = 0x2C94, .cat = .upper_title },   .{ .start = 0x2C95, .end = 0x2C95, .cat = .lower },         .{ .start = 0x2C96, .end = 0x2C96, .cat = .upper_title },
    .{ .start = 0x2C97, .end = 0x2C97, .cat = .lower },         .{ .start = 0x2C98, .end = 0x2C98, .cat = .upper_title },   .{ .start = 0x2C99, .end = 0x2C99, .cat = .lower },
    .{ .start = 0x2C9A, .end = 0x2C9A, .cat = .upper_title },   .{ .start = 0x2C9B, .end = 0x2C9B, .cat = .lower },         .{ .start = 0x2C9C, .end = 0x2C9C, .cat = .upper_title },
    .{ .start = 0x2C9D, .end = 0x2C9D, .cat = .lower },         .{ .start = 0x2C9E, .end = 0x2C9E, .cat = .upper_title },   .{ .start = 0x2C9F, .end = 0x2C9F, .cat = .lower },
    .{ .start = 0x2CA0, .end = 0x2CA0, .cat = .upper_title },   .{ .start = 0x2CA1, .end = 0x2CA1, .cat = .lower },         .{ .start = 0x2CA2, .end = 0x2CA2, .cat = .upper_title },
    .{ .start = 0x2CA3, .end = 0x2CA3, .cat = .lower },         .{ .start = 0x2CA4, .end = 0x2CA4, .cat = .upper_title },   .{ .start = 0x2CA5, .end = 0x2CA5, .cat = .lower },
    .{ .start = 0x2CA6, .end = 0x2CA6, .cat = .upper_title },   .{ .start = 0x2CA7, .end = 0x2CA7, .cat = .lower },         .{ .start = 0x2CA8, .end = 0x2CA8, .cat = .upper_title },
    .{ .start = 0x2CA9, .end = 0x2CA9, .cat = .lower },         .{ .start = 0x2CAA, .end = 0x2CAA, .cat = .upper_title },   .{ .start = 0x2CAB, .end = 0x2CAB, .cat = .lower },
    .{ .start = 0x2CAC, .end = 0x2CAC, .cat = .upper_title },   .{ .start = 0x2CAD, .end = 0x2CAD, .cat = .lower },         .{ .start = 0x2CAE, .end = 0x2CAE, .cat = .upper_title },
    .{ .start = 0x2CAF, .end = 0x2CAF, .cat = .lower },         .{ .start = 0x2CB0, .end = 0x2CB0, .cat = .upper_title },   .{ .start = 0x2CB1, .end = 0x2CB1, .cat = .lower },
    .{ .start = 0x2CB2, .end = 0x2CB2, .cat = .upper_title },   .{ .start = 0x2CB3, .end = 0x2CB3, .cat = .lower },         .{ .start = 0x2CB4, .end = 0x2CB4, .cat = .upper_title },
    .{ .start = 0x2CB5, .end = 0x2CB5, .cat = .lower },         .{ .start = 0x2CB6, .end = 0x2CB6, .cat = .upper_title },   .{ .start = 0x2CB7, .end = 0x2CB7, .cat = .lower },
    .{ .start = 0x2CB8, .end = 0x2CB8, .cat = .upper_title },   .{ .start = 0x2CB9, .end = 0x2CB9, .cat = .lower },         .{ .start = 0x2CBA, .end = 0x2CBA, .cat = .upper_title },
    .{ .start = 0x2CBB, .end = 0x2CBB, .cat = .lower },         .{ .start = 0x2CBC, .end = 0x2CBC, .cat = .upper_title },   .{ .start = 0x2CBD, .end = 0x2CBD, .cat = .lower },
    .{ .start = 0x2CBE, .end = 0x2CBE, .cat = .upper_title },   .{ .start = 0x2CBF, .end = 0x2CBF, .cat = .lower },         .{ .start = 0x2CC0, .end = 0x2CC0, .cat = .upper_title },
    .{ .start = 0x2CC1, .end = 0x2CC1, .cat = .lower },         .{ .start = 0x2CC2, .end = 0x2CC2, .cat = .upper_title },   .{ .start = 0x2CC3, .end = 0x2CC3, .cat = .lower },
    .{ .start = 0x2CC4, .end = 0x2CC4, .cat = .upper_title },   .{ .start = 0x2CC5, .end = 0x2CC5, .cat = .lower },         .{ .start = 0x2CC6, .end = 0x2CC6, .cat = .upper_title },
    .{ .start = 0x2CC7, .end = 0x2CC7, .cat = .lower },         .{ .start = 0x2CC8, .end = 0x2CC8, .cat = .upper_title },   .{ .start = 0x2CC9, .end = 0x2CC9, .cat = .lower },
    .{ .start = 0x2CCA, .end = 0x2CCA, .cat = .upper_title },   .{ .start = 0x2CCB, .end = 0x2CCB, .cat = .lower },         .{ .start = 0x2CCC, .end = 0x2CCC, .cat = .upper_title },
    .{ .start = 0x2CCD, .end = 0x2CCD, .cat = .lower },         .{ .start = 0x2CCE, .end = 0x2CCE, .cat = .upper_title },   .{ .start = 0x2CCF, .end = 0x2CCF, .cat = .lower },
    .{ .start = 0x2CD0, .end = 0x2CD0, .cat = .upper_title },   .{ .start = 0x2CD1, .end = 0x2CD1, .cat = .lower },         .{ .start = 0x2CD2, .end = 0x2CD2, .cat = .upper_title },
    .{ .start = 0x2CD3, .end = 0x2CD3, .cat = .lower },         .{ .start = 0x2CD4, .end = 0x2CD4, .cat = .upper_title },   .{ .start = 0x2CD5, .end = 0x2CD5, .cat = .lower },
    .{ .start = 0x2CD6, .end = 0x2CD6, .cat = .upper_title },   .{ .start = 0x2CD7, .end = 0x2CD7, .cat = .lower },         .{ .start = 0x2CD8, .end = 0x2CD8, .cat = .upper_title },
    .{ .start = 0x2CD9, .end = 0x2CD9, .cat = .lower },         .{ .start = 0x2CDA, .end = 0x2CDA, .cat = .upper_title },   .{ .start = 0x2CDB, .end = 0x2CDB, .cat = .lower },
    .{ .start = 0x2CDC, .end = 0x2CDC, .cat = .upper_title },   .{ .start = 0x2CDD, .end = 0x2CDD, .cat = .lower },         .{ .start = 0x2CDE, .end = 0x2CDE, .cat = .upper_title },
    .{ .start = 0x2CDF, .end = 0x2CDF, .cat = .lower },         .{ .start = 0x2CE0, .end = 0x2CE0, .cat = .upper_title },   .{ .start = 0x2CE1, .end = 0x2CE1, .cat = .lower },
    .{ .start = 0x2CE2, .end = 0x2CE2, .cat = .upper_title },   .{ .start = 0x2CE3, .end = 0x2CE4, .cat = .lower },         .{ .start = 0x2CEB, .end = 0x2CEB, .cat = .upper_title },
    .{ .start = 0x2CEC, .end = 0x2CEC, .cat = .lower },         .{ .start = 0x2CED, .end = 0x2CED, .cat = .upper_title },   .{ .start = 0x2CEE, .end = 0x2CEE, .cat = .lower },
    .{ .start = 0x2CF2, .end = 0x2CF2, .cat = .upper_title },   .{ .start = 0x2CF3, .end = 0x2CF3, .cat = .lower },         .{ .start = 0x2D00, .end = 0x2D25, .cat = .lower },
    .{ .start = 0x2D27, .end = 0x2D27, .cat = .lower },         .{ .start = 0x2D2D, .end = 0x2D2D, .cat = .lower },         .{ .start = 0x2D30, .end = 0x2D67, .cat = .other },
    .{ .start = 0x2D6F, .end = 0x2D6F, .cat = .other },         .{ .start = 0x2D80, .end = 0x2D96, .cat = .other },         .{ .start = 0x2DA0, .end = 0x2DA6, .cat = .other },
    .{ .start = 0x2DA8, .end = 0x2DAE, .cat = .other },         .{ .start = 0x2DB0, .end = 0x2DB6, .cat = .other },         .{ .start = 0x2DB8, .end = 0x2DBE, .cat = .other },
    .{ .start = 0x2DC0, .end = 0x2DC6, .cat = .other },         .{ .start = 0x2DC8, .end = 0x2DCE, .cat = .other },         .{ .start = 0x2DD0, .end = 0x2DD6, .cat = .other },
    .{ .start = 0x2DD8, .end = 0x2DDE, .cat = .other },         .{ .start = 0x2E2F, .end = 0x2E2F, .cat = .other },         .{ .start = 0x3005, .end = 0x3006, .cat = .other },
    .{ .start = 0x3031, .end = 0x3035, .cat = .other },         .{ .start = 0x303B, .end = 0x303C, .cat = .other },         .{ .start = 0x3041, .end = 0x3096, .cat = .other },
    .{ .start = 0x309D, .end = 0x309F, .cat = .other },         .{ .start = 0x30A1, .end = 0x30FA, .cat = .other },         .{ .start = 0x30FC, .end = 0x30FF, .cat = .other },
    .{ .start = 0x3105, .end = 0x312F, .cat = .other },         .{ .start = 0x3131, .end = 0x318E, .cat = .other },         .{ .start = 0x31A0, .end = 0x31BF, .cat = .other },
    .{ .start = 0x31F0, .end = 0x31FF, .cat = .other },         .{ .start = 0x3400, .end = 0x4DBF, .cat = .other },         .{ .start = 0x4E00, .end = 0xA48C, .cat = .other },
    .{ .start = 0xA4D0, .end = 0xA4FD, .cat = .other },         .{ .start = 0xA500, .end = 0xA60C, .cat = .other },         .{ .start = 0xA610, .end = 0xA61F, .cat = .other },
    .{ .start = 0xA62A, .end = 0xA62B, .cat = .other },         .{ .start = 0xA640, .end = 0xA640, .cat = .upper_title },   .{ .start = 0xA641, .end = 0xA641, .cat = .lower },
    .{ .start = 0xA642, .end = 0xA642, .cat = .upper_title },   .{ .start = 0xA643, .end = 0xA643, .cat = .lower },         .{ .start = 0xA644, .end = 0xA644, .cat = .upper_title },
    .{ .start = 0xA645, .end = 0xA645, .cat = .lower },         .{ .start = 0xA646, .end = 0xA646, .cat = .upper_title },   .{ .start = 0xA647, .end = 0xA647, .cat = .lower },
    .{ .start = 0xA648, .end = 0xA648, .cat = .upper_title },   .{ .start = 0xA649, .end = 0xA649, .cat = .lower },         .{ .start = 0xA64A, .end = 0xA64A, .cat = .upper_title },
    .{ .start = 0xA64B, .end = 0xA64B, .cat = .lower },         .{ .start = 0xA64C, .end = 0xA64C, .cat = .upper_title },   .{ .start = 0xA64D, .end = 0xA64D, .cat = .lower },
    .{ .start = 0xA64E, .end = 0xA64E, .cat = .upper_title },   .{ .start = 0xA64F, .end = 0xA64F, .cat = .lower },         .{ .start = 0xA650, .end = 0xA650, .cat = .upper_title },
    .{ .start = 0xA651, .end = 0xA651, .cat = .lower },         .{ .start = 0xA652, .end = 0xA652, .cat = .upper_title },   .{ .start = 0xA653, .end = 0xA653, .cat = .lower },
    .{ .start = 0xA654, .end = 0xA654, .cat = .upper_title },   .{ .start = 0xA655, .end = 0xA655, .cat = .lower },         .{ .start = 0xA656, .end = 0xA656, .cat = .upper_title },
    .{ .start = 0xA657, .end = 0xA657, .cat = .lower },         .{ .start = 0xA658, .end = 0xA658, .cat = .upper_title },   .{ .start = 0xA659, .end = 0xA659, .cat = .lower },
    .{ .start = 0xA65A, .end = 0xA65A, .cat = .upper_title },   .{ .start = 0xA65B, .end = 0xA65B, .cat = .lower },         .{ .start = 0xA65C, .end = 0xA65C, .cat = .upper_title },
    .{ .start = 0xA65D, .end = 0xA65D, .cat = .lower },         .{ .start = 0xA65E, .end = 0xA65E, .cat = .upper_title },   .{ .start = 0xA65F, .end = 0xA65F, .cat = .lower },
    .{ .start = 0xA660, .end = 0xA660, .cat = .upper_title },   .{ .start = 0xA661, .end = 0xA661, .cat = .lower },         .{ .start = 0xA662, .end = 0xA662, .cat = .upper_title },
    .{ .start = 0xA663, .end = 0xA663, .cat = .lower },         .{ .start = 0xA664, .end = 0xA664, .cat = .upper_title },   .{ .start = 0xA665, .end = 0xA665, .cat = .lower },
    .{ .start = 0xA666, .end = 0xA666, .cat = .upper_title },   .{ .start = 0xA667, .end = 0xA667, .cat = .lower },         .{ .start = 0xA668, .end = 0xA668, .cat = .upper_title },
    .{ .start = 0xA669, .end = 0xA669, .cat = .lower },         .{ .start = 0xA66A, .end = 0xA66A, .cat = .upper_title },   .{ .start = 0xA66B, .end = 0xA66B, .cat = .lower },
    .{ .start = 0xA66C, .end = 0xA66C, .cat = .upper_title },   .{ .start = 0xA66D, .end = 0xA66D, .cat = .lower },         .{ .start = 0xA66E, .end = 0xA66E, .cat = .other },
    .{ .start = 0xA67F, .end = 0xA67F, .cat = .other },         .{ .start = 0xA680, .end = 0xA680, .cat = .upper_title },   .{ .start = 0xA681, .end = 0xA681, .cat = .lower },
    .{ .start = 0xA682, .end = 0xA682, .cat = .upper_title },   .{ .start = 0xA683, .end = 0xA683, .cat = .lower },         .{ .start = 0xA684, .end = 0xA684, .cat = .upper_title },
    .{ .start = 0xA685, .end = 0xA685, .cat = .lower },         .{ .start = 0xA686, .end = 0xA686, .cat = .upper_title },   .{ .start = 0xA687, .end = 0xA687, .cat = .lower },
    .{ .start = 0xA688, .end = 0xA688, .cat = .upper_title },   .{ .start = 0xA689, .end = 0xA689, .cat = .lower },         .{ .start = 0xA68A, .end = 0xA68A, .cat = .upper_title },
    .{ .start = 0xA68B, .end = 0xA68B, .cat = .lower },         .{ .start = 0xA68C, .end = 0xA68C, .cat = .upper_title },   .{ .start = 0xA68D, .end = 0xA68D, .cat = .lower },
    .{ .start = 0xA68E, .end = 0xA68E, .cat = .upper_title },   .{ .start = 0xA68F, .end = 0xA68F, .cat = .lower },         .{ .start = 0xA690, .end = 0xA690, .cat = .upper_title },
    .{ .start = 0xA691, .end = 0xA691, .cat = .lower },         .{ .start = 0xA692, .end = 0xA692, .cat = .upper_title },   .{ .start = 0xA693, .end = 0xA693, .cat = .lower },
    .{ .start = 0xA694, .end = 0xA694, .cat = .upper_title },   .{ .start = 0xA695, .end = 0xA695, .cat = .lower },         .{ .start = 0xA696, .end = 0xA696, .cat = .upper_title },
    .{ .start = 0xA697, .end = 0xA697, .cat = .lower },         .{ .start = 0xA698, .end = 0xA698, .cat = .upper_title },   .{ .start = 0xA699, .end = 0xA699, .cat = .lower },
    .{ .start = 0xA69A, .end = 0xA69A, .cat = .upper_title },   .{ .start = 0xA69B, .end = 0xA69B, .cat = .lower },         .{ .start = 0xA69C, .end = 0xA69D, .cat = .other },
    .{ .start = 0xA6A0, .end = 0xA6E5, .cat = .other },         .{ .start = 0xA717, .end = 0xA71F, .cat = .other },         .{ .start = 0xA722, .end = 0xA722, .cat = .upper_title },
    .{ .start = 0xA723, .end = 0xA723, .cat = .lower },         .{ .start = 0xA724, .end = 0xA724, .cat = .upper_title },   .{ .start = 0xA725, .end = 0xA725, .cat = .lower },
    .{ .start = 0xA726, .end = 0xA726, .cat = .upper_title },   .{ .start = 0xA727, .end = 0xA727, .cat = .lower },         .{ .start = 0xA728, .end = 0xA728, .cat = .upper_title },
    .{ .start = 0xA729, .end = 0xA729, .cat = .lower },         .{ .start = 0xA72A, .end = 0xA72A, .cat = .upper_title },   .{ .start = 0xA72B, .end = 0xA72B, .cat = .lower },
    .{ .start = 0xA72C, .end = 0xA72C, .cat = .upper_title },   .{ .start = 0xA72D, .end = 0xA72D, .cat = .lower },         .{ .start = 0xA72E, .end = 0xA72E, .cat = .upper_title },
    .{ .start = 0xA72F, .end = 0xA731, .cat = .lower },         .{ .start = 0xA732, .end = 0xA732, .cat = .upper_title },   .{ .start = 0xA733, .end = 0xA733, .cat = .lower },
    .{ .start = 0xA734, .end = 0xA734, .cat = .upper_title },   .{ .start = 0xA735, .end = 0xA735, .cat = .lower },         .{ .start = 0xA736, .end = 0xA736, .cat = .upper_title },
    .{ .start = 0xA737, .end = 0xA737, .cat = .lower },         .{ .start = 0xA738, .end = 0xA738, .cat = .upper_title },   .{ .start = 0xA739, .end = 0xA739, .cat = .lower },
    .{ .start = 0xA73A, .end = 0xA73A, .cat = .upper_title },   .{ .start = 0xA73B, .end = 0xA73B, .cat = .lower },         .{ .start = 0xA73C, .end = 0xA73C, .cat = .upper_title },
    .{ .start = 0xA73D, .end = 0xA73D, .cat = .lower },         .{ .start = 0xA73E, .end = 0xA73E, .cat = .upper_title },   .{ .start = 0xA73F, .end = 0xA73F, .cat = .lower },
    .{ .start = 0xA740, .end = 0xA740, .cat = .upper_title },   .{ .start = 0xA741, .end = 0xA741, .cat = .lower },         .{ .start = 0xA742, .end = 0xA742, .cat = .upper_title },
    .{ .start = 0xA743, .end = 0xA743, .cat = .lower },         .{ .start = 0xA744, .end = 0xA744, .cat = .upper_title },   .{ .start = 0xA745, .end = 0xA745, .cat = .lower },
    .{ .start = 0xA746, .end = 0xA746, .cat = .upper_title },   .{ .start = 0xA747, .end = 0xA747, .cat = .lower },         .{ .start = 0xA748, .end = 0xA748, .cat = .upper_title },
    .{ .start = 0xA749, .end = 0xA749, .cat = .lower },         .{ .start = 0xA74A, .end = 0xA74A, .cat = .upper_title },   .{ .start = 0xA74B, .end = 0xA74B, .cat = .lower },
    .{ .start = 0xA74C, .end = 0xA74C, .cat = .upper_title },   .{ .start = 0xA74D, .end = 0xA74D, .cat = .lower },         .{ .start = 0xA74E, .end = 0xA74E, .cat = .upper_title },
    .{ .start = 0xA74F, .end = 0xA74F, .cat = .lower },         .{ .start = 0xA750, .end = 0xA750, .cat = .upper_title },   .{ .start = 0xA751, .end = 0xA751, .cat = .lower },
    .{ .start = 0xA752, .end = 0xA752, .cat = .upper_title },   .{ .start = 0xA753, .end = 0xA753, .cat = .lower },         .{ .start = 0xA754, .end = 0xA754, .cat = .upper_title },
    .{ .start = 0xA755, .end = 0xA755, .cat = .lower },         .{ .start = 0xA756, .end = 0xA756, .cat = .upper_title },   .{ .start = 0xA757, .end = 0xA757, .cat = .lower },
    .{ .start = 0xA758, .end = 0xA758, .cat = .upper_title },   .{ .start = 0xA759, .end = 0xA759, .cat = .lower },         .{ .start = 0xA75A, .end = 0xA75A, .cat = .upper_title },
    .{ .start = 0xA75B, .end = 0xA75B, .cat = .lower },         .{ .start = 0xA75C, .end = 0xA75C, .cat = .upper_title },   .{ .start = 0xA75D, .end = 0xA75D, .cat = .lower },
    .{ .start = 0xA75E, .end = 0xA75E, .cat = .upper_title },   .{ .start = 0xA75F, .end = 0xA75F, .cat = .lower },         .{ .start = 0xA760, .end = 0xA760, .cat = .upper_title },
    .{ .start = 0xA761, .end = 0xA761, .cat = .lower },         .{ .start = 0xA762, .end = 0xA762, .cat = .upper_title },   .{ .start = 0xA763, .end = 0xA763, .cat = .lower },
    .{ .start = 0xA764, .end = 0xA764, .cat = .upper_title },   .{ .start = 0xA765, .end = 0xA765, .cat = .lower },         .{ .start = 0xA766, .end = 0xA766, .cat = .upper_title },
    .{ .start = 0xA767, .end = 0xA767, .cat = .lower },         .{ .start = 0xA768, .end = 0xA768, .cat = .upper_title },   .{ .start = 0xA769, .end = 0xA769, .cat = .lower },
    .{ .start = 0xA76A, .end = 0xA76A, .cat = .upper_title },   .{ .start = 0xA76B, .end = 0xA76B, .cat = .lower },         .{ .start = 0xA76C, .end = 0xA76C, .cat = .upper_title },
    .{ .start = 0xA76D, .end = 0xA76D, .cat = .lower },         .{ .start = 0xA76E, .end = 0xA76E, .cat = .upper_title },   .{ .start = 0xA76F, .end = 0xA76F, .cat = .lower },
    .{ .start = 0xA770, .end = 0xA770, .cat = .other },         .{ .start = 0xA771, .end = 0xA778, .cat = .lower },         .{ .start = 0xA779, .end = 0xA779, .cat = .upper_title },
    .{ .start = 0xA77A, .end = 0xA77A, .cat = .lower },         .{ .start = 0xA77B, .end = 0xA77B, .cat = .upper_title },   .{ .start = 0xA77C, .end = 0xA77C, .cat = .lower },
    .{ .start = 0xA77D, .end = 0xA77E, .cat = .upper_title },   .{ .start = 0xA77F, .end = 0xA77F, .cat = .lower },         .{ .start = 0xA780, .end = 0xA780, .cat = .upper_title },
    .{ .start = 0xA781, .end = 0xA781, .cat = .lower },         .{ .start = 0xA782, .end = 0xA782, .cat = .upper_title },   .{ .start = 0xA783, .end = 0xA783, .cat = .lower },
    .{ .start = 0xA784, .end = 0xA784, .cat = .upper_title },   .{ .start = 0xA785, .end = 0xA785, .cat = .lower },         .{ .start = 0xA786, .end = 0xA786, .cat = .upper_title },
    .{ .start = 0xA787, .end = 0xA787, .cat = .lower },         .{ .start = 0xA788, .end = 0xA788, .cat = .other },         .{ .start = 0xA78B, .end = 0xA78B, .cat = .upper_title },
    .{ .start = 0xA78C, .end = 0xA78C, .cat = .lower },         .{ .start = 0xA78D, .end = 0xA78D, .cat = .upper_title },   .{ .start = 0xA78E, .end = 0xA78E, .cat = .lower },
    .{ .start = 0xA78F, .end = 0xA78F, .cat = .other },         .{ .start = 0xA790, .end = 0xA790, .cat = .upper_title },   .{ .start = 0xA791, .end = 0xA791, .cat = .lower },
    .{ .start = 0xA792, .end = 0xA792, .cat = .upper_title },   .{ .start = 0xA793, .end = 0xA795, .cat = .lower },         .{ .start = 0xA796, .end = 0xA796, .cat = .upper_title },
    .{ .start = 0xA797, .end = 0xA797, .cat = .lower },         .{ .start = 0xA798, .end = 0xA798, .cat = .upper_title },   .{ .start = 0xA799, .end = 0xA799, .cat = .lower },
    .{ .start = 0xA79A, .end = 0xA79A, .cat = .upper_title },   .{ .start = 0xA79B, .end = 0xA79B, .cat = .lower },         .{ .start = 0xA79C, .end = 0xA79C, .cat = .upper_title },
    .{ .start = 0xA79D, .end = 0xA79D, .cat = .lower },         .{ .start = 0xA79E, .end = 0xA79E, .cat = .upper_title },   .{ .start = 0xA79F, .end = 0xA79F, .cat = .lower },
    .{ .start = 0xA7A0, .end = 0xA7A0, .cat = .upper_title },   .{ .start = 0xA7A1, .end = 0xA7A1, .cat = .lower },         .{ .start = 0xA7A2, .end = 0xA7A2, .cat = .upper_title },
    .{ .start = 0xA7A3, .end = 0xA7A3, .cat = .lower },         .{ .start = 0xA7A4, .end = 0xA7A4, .cat = .upper_title },   .{ .start = 0xA7A5, .end = 0xA7A5, .cat = .lower },
    .{ .start = 0xA7A6, .end = 0xA7A6, .cat = .upper_title },   .{ .start = 0xA7A7, .end = 0xA7A7, .cat = .lower },         .{ .start = 0xA7A8, .end = 0xA7A8, .cat = .upper_title },
    .{ .start = 0xA7A9, .end = 0xA7A9, .cat = .lower },         .{ .start = 0xA7AA, .end = 0xA7AE, .cat = .upper_title },   .{ .start = 0xA7AF, .end = 0xA7AF, .cat = .lower },
    .{ .start = 0xA7B0, .end = 0xA7B4, .cat = .upper_title },   .{ .start = 0xA7B5, .end = 0xA7B5, .cat = .lower },         .{ .start = 0xA7B6, .end = 0xA7B6, .cat = .upper_title },
    .{ .start = 0xA7B7, .end = 0xA7B7, .cat = .lower },         .{ .start = 0xA7B8, .end = 0xA7B8, .cat = .upper_title },   .{ .start = 0xA7B9, .end = 0xA7B9, .cat = .lower },
    .{ .start = 0xA7BA, .end = 0xA7BA, .cat = .upper_title },   .{ .start = 0xA7BB, .end = 0xA7BB, .cat = .lower },         .{ .start = 0xA7BC, .end = 0xA7BC, .cat = .upper_title },
    .{ .start = 0xA7BD, .end = 0xA7BD, .cat = .lower },         .{ .start = 0xA7BE, .end = 0xA7BE, .cat = .upper_title },   .{ .start = 0xA7BF, .end = 0xA7BF, .cat = .lower },
    .{ .start = 0xA7C0, .end = 0xA7C0, .cat = .upper_title },   .{ .start = 0xA7C1, .end = 0xA7C1, .cat = .lower },         .{ .start = 0xA7C2, .end = 0xA7C2, .cat = .upper_title },
    .{ .start = 0xA7C3, .end = 0xA7C3, .cat = .lower },         .{ .start = 0xA7C4, .end = 0xA7C7, .cat = .upper_title },   .{ .start = 0xA7C8, .end = 0xA7C8, .cat = .lower },
    .{ .start = 0xA7C9, .end = 0xA7C9, .cat = .upper_title },   .{ .start = 0xA7CA, .end = 0xA7CA, .cat = .lower },         .{ .start = 0xA7CB, .end = 0xA7CC, .cat = .upper_title },
    .{ .start = 0xA7CD, .end = 0xA7CD, .cat = .lower },         .{ .start = 0xA7CE, .end = 0xA7CE, .cat = .upper_title },   .{ .start = 0xA7CF, .end = 0xA7CF, .cat = .lower },
    .{ .start = 0xA7D0, .end = 0xA7D0, .cat = .upper_title },   .{ .start = 0xA7D1, .end = 0xA7D1, .cat = .lower },         .{ .start = 0xA7D2, .end = 0xA7D2, .cat = .upper_title },
    .{ .start = 0xA7D3, .end = 0xA7D3, .cat = .lower },         .{ .start = 0xA7D4, .end = 0xA7D4, .cat = .upper_title },   .{ .start = 0xA7D5, .end = 0xA7D5, .cat = .lower },
    .{ .start = 0xA7D6, .end = 0xA7D6, .cat = .upper_title },   .{ .start = 0xA7D7, .end = 0xA7D7, .cat = .lower },         .{ .start = 0xA7D8, .end = 0xA7D8, .cat = .upper_title },
    .{ .start = 0xA7D9, .end = 0xA7D9, .cat = .lower },         .{ .start = 0xA7DA, .end = 0xA7DA, .cat = .upper_title },   .{ .start = 0xA7DB, .end = 0xA7DB, .cat = .lower },
    .{ .start = 0xA7DC, .end = 0xA7DC, .cat = .upper_title },   .{ .start = 0xA7F1, .end = 0xA7F4, .cat = .other },         .{ .start = 0xA7F5, .end = 0xA7F5, .cat = .upper_title },
    .{ .start = 0xA7F6, .end = 0xA7F6, .cat = .lower },         .{ .start = 0xA7F7, .end = 0xA7F9, .cat = .other },         .{ .start = 0xA7FA, .end = 0xA7FA, .cat = .lower },
    .{ .start = 0xA7FB, .end = 0xA801, .cat = .other },         .{ .start = 0xA803, .end = 0xA805, .cat = .other },         .{ .start = 0xA807, .end = 0xA80A, .cat = .other },
    .{ .start = 0xA80C, .end = 0xA822, .cat = .other },         .{ .start = 0xA840, .end = 0xA873, .cat = .other },         .{ .start = 0xA882, .end = 0xA8B3, .cat = .other },
    .{ .start = 0xA8F2, .end = 0xA8F7, .cat = .other },         .{ .start = 0xA8FB, .end = 0xA8FB, .cat = .other },         .{ .start = 0xA8FD, .end = 0xA8FE, .cat = .other },
    .{ .start = 0xA90A, .end = 0xA925, .cat = .other },         .{ .start = 0xA930, .end = 0xA946, .cat = .other },         .{ .start = 0xA960, .end = 0xA97C, .cat = .other },
    .{ .start = 0xA984, .end = 0xA9B2, .cat = .other },         .{ .start = 0xA9CF, .end = 0xA9CF, .cat = .other },         .{ .start = 0xA9E0, .end = 0xA9E4, .cat = .other },
    .{ .start = 0xA9E6, .end = 0xA9EF, .cat = .other },         .{ .start = 0xA9FA, .end = 0xA9FE, .cat = .other },         .{ .start = 0xAA00, .end = 0xAA28, .cat = .other },
    .{ .start = 0xAA40, .end = 0xAA42, .cat = .other },         .{ .start = 0xAA44, .end = 0xAA4B, .cat = .other },         .{ .start = 0xAA60, .end = 0xAA76, .cat = .other },
    .{ .start = 0xAA7A, .end = 0xAA7A, .cat = .other },         .{ .start = 0xAA7E, .end = 0xAAAF, .cat = .other },         .{ .start = 0xAAB1, .end = 0xAAB1, .cat = .other },
    .{ .start = 0xAAB5, .end = 0xAAB6, .cat = .other },         .{ .start = 0xAAB9, .end = 0xAABD, .cat = .other },         .{ .start = 0xAAC0, .end = 0xAAC0, .cat = .other },
    .{ .start = 0xAAC2, .end = 0xAAC2, .cat = .other },         .{ .start = 0xAADB, .end = 0xAADD, .cat = .other },         .{ .start = 0xAAE0, .end = 0xAAEA, .cat = .other },
    .{ .start = 0xAAF2, .end = 0xAAF4, .cat = .other },         .{ .start = 0xAB01, .end = 0xAB06, .cat = .other },         .{ .start = 0xAB09, .end = 0xAB0E, .cat = .other },
    .{ .start = 0xAB11, .end = 0xAB16, .cat = .other },         .{ .start = 0xAB20, .end = 0xAB26, .cat = .other },         .{ .start = 0xAB28, .end = 0xAB2E, .cat = .other },
    .{ .start = 0xAB30, .end = 0xAB5A, .cat = .lower },         .{ .start = 0xAB5C, .end = 0xAB5F, .cat = .other },         .{ .start = 0xAB60, .end = 0xAB68, .cat = .lower },
    .{ .start = 0xAB69, .end = 0xAB69, .cat = .other },         .{ .start = 0xAB70, .end = 0xABBF, .cat = .lower },         .{ .start = 0xABC0, .end = 0xABE2, .cat = .other },
    .{ .start = 0xAC00, .end = 0xD7A3, .cat = .other },         .{ .start = 0xD7B0, .end = 0xD7C6, .cat = .other },         .{ .start = 0xD7CB, .end = 0xD7FB, .cat = .other },
    .{ .start = 0xF900, .end = 0xFA6D, .cat = .other },         .{ .start = 0xFA70, .end = 0xFAD9, .cat = .other },         .{ .start = 0xFB00, .end = 0xFB06, .cat = .lower },
    .{ .start = 0xFB13, .end = 0xFB17, .cat = .lower },         .{ .start = 0xFB1D, .end = 0xFB1D, .cat = .other },         .{ .start = 0xFB1F, .end = 0xFB28, .cat = .other },
    .{ .start = 0xFB2A, .end = 0xFB36, .cat = .other },         .{ .start = 0xFB38, .end = 0xFB3C, .cat = .other },         .{ .start = 0xFB3E, .end = 0xFB3E, .cat = .other },
    .{ .start = 0xFB40, .end = 0xFB41, .cat = .other },         .{ .start = 0xFB43, .end = 0xFB44, .cat = .other },         .{ .start = 0xFB46, .end = 0xFBB1, .cat = .other },
    .{ .start = 0xFBD3, .end = 0xFD3D, .cat = .other },         .{ .start = 0xFD50, .end = 0xFD8F, .cat = .other },         .{ .start = 0xFD92, .end = 0xFDC7, .cat = .other },
    .{ .start = 0xFDF0, .end = 0xFDFB, .cat = .other },         .{ .start = 0xFE70, .end = 0xFE74, .cat = .other },         .{ .start = 0xFE76, .end = 0xFEFC, .cat = .other },
    .{ .start = 0xFF21, .end = 0xFF3A, .cat = .upper_title },   .{ .start = 0xFF41, .end = 0xFF5A, .cat = .lower },         .{ .start = 0xFF66, .end = 0xFFBE, .cat = .other },
    .{ .start = 0xFFC2, .end = 0xFFC7, .cat = .other },         .{ .start = 0xFFCA, .end = 0xFFCF, .cat = .other },         .{ .start = 0xFFD2, .end = 0xFFD7, .cat = .other },
    .{ .start = 0xFFDA, .end = 0xFFDC, .cat = .other },         .{ .start = 0x10000, .end = 0x1000B, .cat = .other },       .{ .start = 0x1000D, .end = 0x10026, .cat = .other },
    .{ .start = 0x10028, .end = 0x1003A, .cat = .other },       .{ .start = 0x1003C, .end = 0x1003D, .cat = .other },       .{ .start = 0x1003F, .end = 0x1004D, .cat = .other },
    .{ .start = 0x10050, .end = 0x1005D, .cat = .other },       .{ .start = 0x10080, .end = 0x100FA, .cat = .other },       .{ .start = 0x10280, .end = 0x1029C, .cat = .other },
    .{ .start = 0x102A0, .end = 0x102D0, .cat = .other },       .{ .start = 0x10300, .end = 0x1031F, .cat = .other },       .{ .start = 0x1032D, .end = 0x10340, .cat = .other },
    .{ .start = 0x10342, .end = 0x10349, .cat = .other },       .{ .start = 0x10350, .end = 0x10375, .cat = .other },       .{ .start = 0x10380, .end = 0x1039D, .cat = .other },
    .{ .start = 0x103A0, .end = 0x103C3, .cat = .other },       .{ .start = 0x103C8, .end = 0x103CF, .cat = .other },       .{ .start = 0x10400, .end = 0x10427, .cat = .upper_title },
    .{ .start = 0x10428, .end = 0x1044F, .cat = .lower },       .{ .start = 0x10450, .end = 0x1049D, .cat = .other },       .{ .start = 0x104B0, .end = 0x104D3, .cat = .upper_title },
    .{ .start = 0x104D8, .end = 0x104FB, .cat = .lower },       .{ .start = 0x10500, .end = 0x10527, .cat = .other },       .{ .start = 0x10530, .end = 0x10563, .cat = .other },
    .{ .start = 0x10570, .end = 0x1057A, .cat = .upper_title }, .{ .start = 0x1057C, .end = 0x1058A, .cat = .upper_title }, .{ .start = 0x1058C, .end = 0x10592, .cat = .upper_title },
    .{ .start = 0x10594, .end = 0x10595, .cat = .upper_title }, .{ .start = 0x10597, .end = 0x105A1, .cat = .lower },       .{ .start = 0x105A3, .end = 0x105B1, .cat = .lower },
    .{ .start = 0x105B3, .end = 0x105B9, .cat = .lower },       .{ .start = 0x105BB, .end = 0x105BC, .cat = .lower },       .{ .start = 0x105C0, .end = 0x105F3, .cat = .other },
    .{ .start = 0x10600, .end = 0x10736, .cat = .other },       .{ .start = 0x10740, .end = 0x10755, .cat = .other },       .{ .start = 0x10760, .end = 0x10767, .cat = .other },
    .{ .start = 0x10780, .end = 0x10785, .cat = .other },       .{ .start = 0x10787, .end = 0x107B0, .cat = .other },       .{ .start = 0x107B2, .end = 0x107BA, .cat = .other },
    .{ .start = 0x10800, .end = 0x10805, .cat = .other },       .{ .start = 0x10808, .end = 0x10808, .cat = .other },       .{ .start = 0x1080A, .end = 0x10835, .cat = .other },
    .{ .start = 0x10837, .end = 0x10838, .cat = .other },       .{ .start = 0x1083C, .end = 0x1083C, .cat = .other },       .{ .start = 0x1083F, .end = 0x10855, .cat = .other },
    .{ .start = 0x10860, .end = 0x10876, .cat = .other },       .{ .start = 0x10880, .end = 0x1089E, .cat = .other },       .{ .start = 0x108E0, .end = 0x108F2, .cat = .other },
    .{ .start = 0x108F4, .end = 0x108F5, .cat = .other },       .{ .start = 0x10900, .end = 0x10915, .cat = .other },       .{ .start = 0x10920, .end = 0x10939, .cat = .other },
    .{ .start = 0x10940, .end = 0x10959, .cat = .other },       .{ .start = 0x10980, .end = 0x109B7, .cat = .other },       .{ .start = 0x109BE, .end = 0x109BF, .cat = .other },
    .{ .start = 0x10A00, .end = 0x10A00, .cat = .other },       .{ .start = 0x10A10, .end = 0x10A13, .cat = .other },       .{ .start = 0x10A15, .end = 0x10A17, .cat = .other },
    .{ .start = 0x10A19, .end = 0x10A35, .cat = .other },       .{ .start = 0x10A60, .end = 0x10A7C, .cat = .other },       .{ .start = 0x10A80, .end = 0x10A9C, .cat = .other },
    .{ .start = 0x10AC0, .end = 0x10AC7, .cat = .other },       .{ .start = 0x10AC9, .end = 0x10AE4, .cat = .other },       .{ .start = 0x10B00, .end = 0x10B35, .cat = .other },
    .{ .start = 0x10B40, .end = 0x10B55, .cat = .other },       .{ .start = 0x10B60, .end = 0x10B72, .cat = .other },       .{ .start = 0x10B80, .end = 0x10B91, .cat = .other },
    .{ .start = 0x10C00, .end = 0x10C48, .cat = .other },       .{ .start = 0x10C80, .end = 0x10CB2, .cat = .upper_title }, .{ .start = 0x10CC0, .end = 0x10CF2, .cat = .lower },
    .{ .start = 0x10D00, .end = 0x10D23, .cat = .other },       .{ .start = 0x10D4A, .end = 0x10D4F, .cat = .other },       .{ .start = 0x10D50, .end = 0x10D65, .cat = .upper_title },
    .{ .start = 0x10D6F, .end = 0x10D6F, .cat = .other },       .{ .start = 0x10D70, .end = 0x10D85, .cat = .lower },       .{ .start = 0x10E80, .end = 0x10EA9, .cat = .other },
    .{ .start = 0x10EB0, .end = 0x10EB1, .cat = .other },       .{ .start = 0x10EC2, .end = 0x10EC7, .cat = .other },       .{ .start = 0x10F00, .end = 0x10F1C, .cat = .other },
    .{ .start = 0x10F27, .end = 0x10F27, .cat = .other },       .{ .start = 0x10F30, .end = 0x10F45, .cat = .other },       .{ .start = 0x10F70, .end = 0x10F81, .cat = .other },
    .{ .start = 0x10FB0, .end = 0x10FC4, .cat = .other },       .{ .start = 0x10FE0, .end = 0x10FF6, .cat = .other },       .{ .start = 0x11003, .end = 0x11037, .cat = .other },
    .{ .start = 0x11071, .end = 0x11072, .cat = .other },       .{ .start = 0x11075, .end = 0x11075, .cat = .other },       .{ .start = 0x11083, .end = 0x110AF, .cat = .other },
    .{ .start = 0x110D0, .end = 0x110E8, .cat = .other },       .{ .start = 0x11103, .end = 0x11126, .cat = .other },       .{ .start = 0x11144, .end = 0x11144, .cat = .other },
    .{ .start = 0x11147, .end = 0x11147, .cat = .other },       .{ .start = 0x11150, .end = 0x11172, .cat = .other },       .{ .start = 0x11176, .end = 0x11176, .cat = .other },
    .{ .start = 0x11183, .end = 0x111B2, .cat = .other },       .{ .start = 0x111C1, .end = 0x111C4, .cat = .other },       .{ .start = 0x111DA, .end = 0x111DA, .cat = .other },
    .{ .start = 0x111DC, .end = 0x111DC, .cat = .other },       .{ .start = 0x11200, .end = 0x11211, .cat = .other },       .{ .start = 0x11213, .end = 0x1122B, .cat = .other },
    .{ .start = 0x1123F, .end = 0x11240, .cat = .other },       .{ .start = 0x11280, .end = 0x11286, .cat = .other },       .{ .start = 0x11288, .end = 0x11288, .cat = .other },
    .{ .start = 0x1128A, .end = 0x1128D, .cat = .other },       .{ .start = 0x1128F, .end = 0x1129D, .cat = .other },       .{ .start = 0x1129F, .end = 0x112A8, .cat = .other },
    .{ .start = 0x112B0, .end = 0x112DE, .cat = .other },       .{ .start = 0x11305, .end = 0x1130C, .cat = .other },       .{ .start = 0x1130F, .end = 0x11310, .cat = .other },
    .{ .start = 0x11313, .end = 0x11328, .cat = .other },       .{ .start = 0x1132A, .end = 0x11330, .cat = .other },       .{ .start = 0x11332, .end = 0x11333, .cat = .other },
    .{ .start = 0x11335, .end = 0x11339, .cat = .other },       .{ .start = 0x1133D, .end = 0x1133D, .cat = .other },       .{ .start = 0x11350, .end = 0x11350, .cat = .other },
    .{ .start = 0x1135D, .end = 0x11361, .cat = .other },       .{ .start = 0x11380, .end = 0x11389, .cat = .other },       .{ .start = 0x1138B, .end = 0x1138B, .cat = .other },
    .{ .start = 0x1138E, .end = 0x1138E, .cat = .other },       .{ .start = 0x11390, .end = 0x113B5, .cat = .other },       .{ .start = 0x113B7, .end = 0x113B7, .cat = .other },
    .{ .start = 0x113D1, .end = 0x113D1, .cat = .other },       .{ .start = 0x113D3, .end = 0x113D3, .cat = .other },       .{ .start = 0x11400, .end = 0x11434, .cat = .other },
    .{ .start = 0x11447, .end = 0x1144A, .cat = .other },       .{ .start = 0x1145F, .end = 0x11461, .cat = .other },       .{ .start = 0x11480, .end = 0x114AF, .cat = .other },
    .{ .start = 0x114C4, .end = 0x114C5, .cat = .other },       .{ .start = 0x114C7, .end = 0x114C7, .cat = .other },       .{ .start = 0x11580, .end = 0x115AE, .cat = .other },
    .{ .start = 0x115D8, .end = 0x115DB, .cat = .other },       .{ .start = 0x11600, .end = 0x1162F, .cat = .other },       .{ .start = 0x11644, .end = 0x11644, .cat = .other },
    .{ .start = 0x11680, .end = 0x116AA, .cat = .other },       .{ .start = 0x116B8, .end = 0x116B8, .cat = .other },       .{ .start = 0x11700, .end = 0x1171A, .cat = .other },
    .{ .start = 0x11740, .end = 0x11746, .cat = .other },       .{ .start = 0x11800, .end = 0x1182B, .cat = .other },       .{ .start = 0x118A0, .end = 0x118BF, .cat = .upper_title },
    .{ .start = 0x118C0, .end = 0x118DF, .cat = .lower },       .{ .start = 0x118FF, .end = 0x11906, .cat = .other },       .{ .start = 0x11909, .end = 0x11909, .cat = .other },
    .{ .start = 0x1190C, .end = 0x11913, .cat = .other },       .{ .start = 0x11915, .end = 0x11916, .cat = .other },       .{ .start = 0x11918, .end = 0x1192F, .cat = .other },
    .{ .start = 0x1193F, .end = 0x1193F, .cat = .other },       .{ .start = 0x11941, .end = 0x11941, .cat = .other },       .{ .start = 0x119A0, .end = 0x119A7, .cat = .other },
    .{ .start = 0x119AA, .end = 0x119D0, .cat = .other },       .{ .start = 0x119E1, .end = 0x119E1, .cat = .other },       .{ .start = 0x119E3, .end = 0x119E3, .cat = .other },
    .{ .start = 0x11A00, .end = 0x11A00, .cat = .other },       .{ .start = 0x11A0B, .end = 0x11A32, .cat = .other },       .{ .start = 0x11A3A, .end = 0x11A3A, .cat = .other },
    .{ .start = 0x11A50, .end = 0x11A50, .cat = .other },       .{ .start = 0x11A5C, .end = 0x11A89, .cat = .other },       .{ .start = 0x11A9D, .end = 0x11A9D, .cat = .other },
    .{ .start = 0x11AB0, .end = 0x11AF8, .cat = .other },       .{ .start = 0x11BC0, .end = 0x11BE0, .cat = .other },       .{ .start = 0x11C00, .end = 0x11C08, .cat = .other },
    .{ .start = 0x11C0A, .end = 0x11C2E, .cat = .other },       .{ .start = 0x11C40, .end = 0x11C40, .cat = .other },       .{ .start = 0x11C72, .end = 0x11C8F, .cat = .other },
    .{ .start = 0x11D00, .end = 0x11D06, .cat = .other },       .{ .start = 0x11D08, .end = 0x11D09, .cat = .other },       .{ .start = 0x11D0B, .end = 0x11D30, .cat = .other },
    .{ .start = 0x11D46, .end = 0x11D46, .cat = .other },       .{ .start = 0x11D60, .end = 0x11D65, .cat = .other },       .{ .start = 0x11D67, .end = 0x11D68, .cat = .other },
    .{ .start = 0x11D6A, .end = 0x11D89, .cat = .other },       .{ .start = 0x11D98, .end = 0x11D98, .cat = .other },       .{ .start = 0x11DB0, .end = 0x11DDB, .cat = .other },
    .{ .start = 0x11EE0, .end = 0x11EF2, .cat = .other },       .{ .start = 0x11F02, .end = 0x11F02, .cat = .other },       .{ .start = 0x11F04, .end = 0x11F10, .cat = .other },
    .{ .start = 0x11F12, .end = 0x11F33, .cat = .other },       .{ .start = 0x11FB0, .end = 0x11FB0, .cat = .other },       .{ .start = 0x12000, .end = 0x12399, .cat = .other },
    .{ .start = 0x12480, .end = 0x12543, .cat = .other },       .{ .start = 0x12F90, .end = 0x12FF0, .cat = .other },       .{ .start = 0x13000, .end = 0x1342F, .cat = .other },
    .{ .start = 0x13441, .end = 0x13446, .cat = .other },       .{ .start = 0x13460, .end = 0x143FA, .cat = .other },       .{ .start = 0x14400, .end = 0x14646, .cat = .other },
    .{ .start = 0x16100, .end = 0x1611D, .cat = .other },       .{ .start = 0x16800, .end = 0x16A38, .cat = .other },       .{ .start = 0x16A40, .end = 0x16A5E, .cat = .other },
    .{ .start = 0x16A70, .end = 0x16ABE, .cat = .other },       .{ .start = 0x16AD0, .end = 0x16AED, .cat = .other },       .{ .start = 0x16B00, .end = 0x16B2F, .cat = .other },
    .{ .start = 0x16B40, .end = 0x16B43, .cat = .other },       .{ .start = 0x16B63, .end = 0x16B77, .cat = .other },       .{ .start = 0x16B7D, .end = 0x16B8F, .cat = .other },
    .{ .start = 0x16D40, .end = 0x16D6C, .cat = .other },       .{ .start = 0x16E40, .end = 0x16E5F, .cat = .upper_title }, .{ .start = 0x16E60, .end = 0x16E7F, .cat = .lower },
    .{ .start = 0x16EA0, .end = 0x16EB8, .cat = .upper_title }, .{ .start = 0x16EBB, .end = 0x16ED3, .cat = .lower },       .{ .start = 0x16F00, .end = 0x16F4A, .cat = .other },
    .{ .start = 0x16F50, .end = 0x16F50, .cat = .other },       .{ .start = 0x16F93, .end = 0x16F9F, .cat = .other },       .{ .start = 0x16FE0, .end = 0x16FE1, .cat = .other },
    .{ .start = 0x16FE3, .end = 0x16FE3, .cat = .other },       .{ .start = 0x16FF2, .end = 0x16FF3, .cat = .other },       .{ .start = 0x17000, .end = 0x18CD5, .cat = .other },
    .{ .start = 0x18CFF, .end = 0x18D1E, .cat = .other },       .{ .start = 0x18D80, .end = 0x18DF2, .cat = .other },       .{ .start = 0x1AFF0, .end = 0x1AFF3, .cat = .other },
    .{ .start = 0x1AFF5, .end = 0x1AFFB, .cat = .other },       .{ .start = 0x1AFFD, .end = 0x1AFFE, .cat = .other },       .{ .start = 0x1B000, .end = 0x1B122, .cat = .other },
    .{ .start = 0x1B132, .end = 0x1B132, .cat = .other },       .{ .start = 0x1B150, .end = 0x1B152, .cat = .other },       .{ .start = 0x1B155, .end = 0x1B155, .cat = .other },
    .{ .start = 0x1B164, .end = 0x1B167, .cat = .other },       .{ .start = 0x1B170, .end = 0x1B2FB, .cat = .other },       .{ .start = 0x1BC00, .end = 0x1BC6A, .cat = .other },
    .{ .start = 0x1BC70, .end = 0x1BC7C, .cat = .other },       .{ .start = 0x1BC80, .end = 0x1BC88, .cat = .other },       .{ .start = 0x1BC90, .end = 0x1BC99, .cat = .other },
    .{ .start = 0x1D400, .end = 0x1D419, .cat = .upper_title }, .{ .start = 0x1D41A, .end = 0x1D433, .cat = .lower },       .{ .start = 0x1D434, .end = 0x1D44D, .cat = .upper_title },
    .{ .start = 0x1D44E, .end = 0x1D454, .cat = .lower },       .{ .start = 0x1D456, .end = 0x1D467, .cat = .lower },       .{ .start = 0x1D468, .end = 0x1D481, .cat = .upper_title },
    .{ .start = 0x1D482, .end = 0x1D49B, .cat = .lower },       .{ .start = 0x1D49C, .end = 0x1D49C, .cat = .upper_title }, .{ .start = 0x1D49E, .end = 0x1D49F, .cat = .upper_title },
    .{ .start = 0x1D4A2, .end = 0x1D4A2, .cat = .upper_title }, .{ .start = 0x1D4A5, .end = 0x1D4A6, .cat = .upper_title }, .{ .start = 0x1D4A9, .end = 0x1D4AC, .cat = .upper_title },
    .{ .start = 0x1D4AE, .end = 0x1D4B5, .cat = .upper_title }, .{ .start = 0x1D4B6, .end = 0x1D4B9, .cat = .lower },       .{ .start = 0x1D4BB, .end = 0x1D4BB, .cat = .lower },
    .{ .start = 0x1D4BD, .end = 0x1D4C3, .cat = .lower },       .{ .start = 0x1D4C5, .end = 0x1D4CF, .cat = .lower },       .{ .start = 0x1D4D0, .end = 0x1D4E9, .cat = .upper_title },
    .{ .start = 0x1D4EA, .end = 0x1D503, .cat = .lower },       .{ .start = 0x1D504, .end = 0x1D505, .cat = .upper_title }, .{ .start = 0x1D507, .end = 0x1D50A, .cat = .upper_title },
    .{ .start = 0x1D50D, .end = 0x1D514, .cat = .upper_title }, .{ .start = 0x1D516, .end = 0x1D51C, .cat = .upper_title }, .{ .start = 0x1D51E, .end = 0x1D537, .cat = .lower },
    .{ .start = 0x1D538, .end = 0x1D539, .cat = .upper_title }, .{ .start = 0x1D53B, .end = 0x1D53E, .cat = .upper_title }, .{ .start = 0x1D540, .end = 0x1D544, .cat = .upper_title },
    .{ .start = 0x1D546, .end = 0x1D546, .cat = .upper_title }, .{ .start = 0x1D54A, .end = 0x1D550, .cat = .upper_title }, .{ .start = 0x1D552, .end = 0x1D56B, .cat = .lower },
    .{ .start = 0x1D56C, .end = 0x1D585, .cat = .upper_title }, .{ .start = 0x1D586, .end = 0x1D59F, .cat = .lower },       .{ .start = 0x1D5A0, .end = 0x1D5B9, .cat = .upper_title },
    .{ .start = 0x1D5BA, .end = 0x1D5D3, .cat = .lower },       .{ .start = 0x1D5D4, .end = 0x1D5ED, .cat = .upper_title }, .{ .start = 0x1D5EE, .end = 0x1D607, .cat = .lower },
    .{ .start = 0x1D608, .end = 0x1D621, .cat = .upper_title }, .{ .start = 0x1D622, .end = 0x1D63B, .cat = .lower },       .{ .start = 0x1D63C, .end = 0x1D655, .cat = .upper_title },
    .{ .start = 0x1D656, .end = 0x1D66F, .cat = .lower },       .{ .start = 0x1D670, .end = 0x1D689, .cat = .upper_title }, .{ .start = 0x1D68A, .end = 0x1D6A5, .cat = .lower },
    .{ .start = 0x1D6A8, .end = 0x1D6C0, .cat = .upper_title }, .{ .start = 0x1D6C2, .end = 0x1D6DA, .cat = .lower },       .{ .start = 0x1D6DC, .end = 0x1D6E1, .cat = .lower },
    .{ .start = 0x1D6E2, .end = 0x1D6FA, .cat = .upper_title }, .{ .start = 0x1D6FC, .end = 0x1D714, .cat = .lower },       .{ .start = 0x1D716, .end = 0x1D71B, .cat = .lower },
    .{ .start = 0x1D71C, .end = 0x1D734, .cat = .upper_title }, .{ .start = 0x1D736, .end = 0x1D74E, .cat = .lower },       .{ .start = 0x1D750, .end = 0x1D755, .cat = .lower },
    .{ .start = 0x1D756, .end = 0x1D76E, .cat = .upper_title }, .{ .start = 0x1D770, .end = 0x1D788, .cat = .lower },       .{ .start = 0x1D78A, .end = 0x1D78F, .cat = .lower },
    .{ .start = 0x1D790, .end = 0x1D7A8, .cat = .upper_title }, .{ .start = 0x1D7AA, .end = 0x1D7C2, .cat = .lower },       .{ .start = 0x1D7C4, .end = 0x1D7C9, .cat = .lower },
    .{ .start = 0x1D7CA, .end = 0x1D7CA, .cat = .upper_title }, .{ .start = 0x1D7CB, .end = 0x1D7CB, .cat = .lower },       .{ .start = 0x1DF00, .end = 0x1DF09, .cat = .lower },
    .{ .start = 0x1DF0A, .end = 0x1DF0A, .cat = .other },       .{ .start = 0x1DF0B, .end = 0x1DF1E, .cat = .lower },       .{ .start = 0x1DF25, .end = 0x1DF2A, .cat = .lower },
    .{ .start = 0x1E030, .end = 0x1E06D, .cat = .other },       .{ .start = 0x1E100, .end = 0x1E12C, .cat = .other },       .{ .start = 0x1E137, .end = 0x1E13D, .cat = .other },
    .{ .start = 0x1E14E, .end = 0x1E14E, .cat = .other },       .{ .start = 0x1E290, .end = 0x1E2AD, .cat = .other },       .{ .start = 0x1E2C0, .end = 0x1E2EB, .cat = .other },
    .{ .start = 0x1E4D0, .end = 0x1E4EB, .cat = .other },       .{ .start = 0x1E5D0, .end = 0x1E5ED, .cat = .other },       .{ .start = 0x1E5F0, .end = 0x1E5F0, .cat = .other },
    .{ .start = 0x1E6C0, .end = 0x1E6DE, .cat = .other },       .{ .start = 0x1E6E0, .end = 0x1E6E2, .cat = .other },       .{ .start = 0x1E6E4, .end = 0x1E6E5, .cat = .other },
    .{ .start = 0x1E6E7, .end = 0x1E6ED, .cat = .other },       .{ .start = 0x1E6F0, .end = 0x1E6F4, .cat = .other },       .{ .start = 0x1E6FE, .end = 0x1E6FF, .cat = .other },
    .{ .start = 0x1E7E0, .end = 0x1E7E6, .cat = .other },       .{ .start = 0x1E7E8, .end = 0x1E7EB, .cat = .other },       .{ .start = 0x1E7ED, .end = 0x1E7EE, .cat = .other },
    .{ .start = 0x1E7F0, .end = 0x1E7FE, .cat = .other },       .{ .start = 0x1E800, .end = 0x1E8C4, .cat = .other },       .{ .start = 0x1E900, .end = 0x1E921, .cat = .upper_title },
    .{ .start = 0x1E922, .end = 0x1E943, .cat = .lower },       .{ .start = 0x1E94B, .end = 0x1E94B, .cat = .other },       .{ .start = 0x1EE00, .end = 0x1EE03, .cat = .other },
    .{ .start = 0x1EE05, .end = 0x1EE1F, .cat = .other },       .{ .start = 0x1EE21, .end = 0x1EE22, .cat = .other },       .{ .start = 0x1EE24, .end = 0x1EE24, .cat = .other },
    .{ .start = 0x1EE27, .end = 0x1EE27, .cat = .other },       .{ .start = 0x1EE29, .end = 0x1EE32, .cat = .other },       .{ .start = 0x1EE34, .end = 0x1EE37, .cat = .other },
    .{ .start = 0x1EE39, .end = 0x1EE39, .cat = .other },       .{ .start = 0x1EE3B, .end = 0x1EE3B, .cat = .other },       .{ .start = 0x1EE42, .end = 0x1EE42, .cat = .other },
    .{ .start = 0x1EE47, .end = 0x1EE47, .cat = .other },       .{ .start = 0x1EE49, .end = 0x1EE49, .cat = .other },       .{ .start = 0x1EE4B, .end = 0x1EE4B, .cat = .other },
    .{ .start = 0x1EE4D, .end = 0x1EE4F, .cat = .other },       .{ .start = 0x1EE51, .end = 0x1EE52, .cat = .other },       .{ .start = 0x1EE54, .end = 0x1EE54, .cat = .other },
    .{ .start = 0x1EE57, .end = 0x1EE57, .cat = .other },       .{ .start = 0x1EE59, .end = 0x1EE59, .cat = .other },       .{ .start = 0x1EE5B, .end = 0x1EE5B, .cat = .other },
    .{ .start = 0x1EE5D, .end = 0x1EE5D, .cat = .other },       .{ .start = 0x1EE5F, .end = 0x1EE5F, .cat = .other },       .{ .start = 0x1EE61, .end = 0x1EE62, .cat = .other },
    .{ .start = 0x1EE64, .end = 0x1EE64, .cat = .other },       .{ .start = 0x1EE67, .end = 0x1EE6A, .cat = .other },       .{ .start = 0x1EE6C, .end = 0x1EE72, .cat = .other },
    .{ .start = 0x1EE74, .end = 0x1EE77, .cat = .other },       .{ .start = 0x1EE79, .end = 0x1EE7C, .cat = .other },       .{ .start = 0x1EE7E, .end = 0x1EE7E, .cat = .other },
    .{ .start = 0x1EE80, .end = 0x1EE89, .cat = .other },       .{ .start = 0x1EE8B, .end = 0x1EE9B, .cat = .other },       .{ .start = 0x1EEA1, .end = 0x1EEA3, .cat = .other },
    .{ .start = 0x1EEA5, .end = 0x1EEA9, .cat = .other },       .{ .start = 0x1EEAB, .end = 0x1EEBB, .cat = .other },       .{ .start = 0x20000, .end = 0x2A6DF, .cat = .other },
    .{ .start = 0x2A700, .end = 0x2B81D, .cat = .other },       .{ .start = 0x2B820, .end = 0x2CEAD, .cat = .other },       .{ .start = 0x2CEB0, .end = 0x2EBE0, .cat = .other },
    .{ .start = 0x2EBF0, .end = 0x2EE5D, .cat = .other },       .{ .start = 0x2F800, .end = 0x2FA1D, .cat = .other },       .{ .start = 0x30000, .end = 0x3134A, .cat = .other },
    .{ .start = 0x31350, .end = 0x33479, .cat = .other },
};

const O200K_MARK_RANGES = [_]CodepointRange{
    .{ .start = 0x0300, .end = 0x036F },   .{ .start = 0x0483, .end = 0x0489 },   .{ .start = 0x0591, .end = 0x05BD },   .{ .start = 0x05BF, .end = 0x05BF },
    .{ .start = 0x05C1, .end = 0x05C2 },   .{ .start = 0x05C4, .end = 0x05C5 },   .{ .start = 0x05C7, .end = 0x05C7 },   .{ .start = 0x0610, .end = 0x061A },
    .{ .start = 0x064B, .end = 0x065F },   .{ .start = 0x0670, .end = 0x0670 },   .{ .start = 0x06D6, .end = 0x06DC },   .{ .start = 0x06DF, .end = 0x06E4 },
    .{ .start = 0x06E7, .end = 0x06E8 },   .{ .start = 0x06EA, .end = 0x06ED },   .{ .start = 0x0711, .end = 0x0711 },   .{ .start = 0x0730, .end = 0x074A },
    .{ .start = 0x07A6, .end = 0x07B0 },   .{ .start = 0x07EB, .end = 0x07F3 },   .{ .start = 0x07FD, .end = 0x07FD },   .{ .start = 0x0816, .end = 0x0819 },
    .{ .start = 0x081B, .end = 0x0823 },   .{ .start = 0x0825, .end = 0x0827 },   .{ .start = 0x0829, .end = 0x082D },   .{ .start = 0x0859, .end = 0x085B },
    .{ .start = 0x0897, .end = 0x089F },   .{ .start = 0x08CA, .end = 0x08E1 },   .{ .start = 0x08E3, .end = 0x0903 },   .{ .start = 0x093A, .end = 0x093C },
    .{ .start = 0x093E, .end = 0x094F },   .{ .start = 0x0951, .end = 0x0957 },   .{ .start = 0x0962, .end = 0x0963 },   .{ .start = 0x0981, .end = 0x0983 },
    .{ .start = 0x09BC, .end = 0x09BC },   .{ .start = 0x09BE, .end = 0x09C4 },   .{ .start = 0x09C7, .end = 0x09C8 },   .{ .start = 0x09CB, .end = 0x09CD },
    .{ .start = 0x09D7, .end = 0x09D7 },   .{ .start = 0x09E2, .end = 0x09E3 },   .{ .start = 0x09FE, .end = 0x09FE },   .{ .start = 0x0A01, .end = 0x0A03 },
    .{ .start = 0x0A3C, .end = 0x0A3C },   .{ .start = 0x0A3E, .end = 0x0A42 },   .{ .start = 0x0A47, .end = 0x0A48 },   .{ .start = 0x0A4B, .end = 0x0A4D },
    .{ .start = 0x0A51, .end = 0x0A51 },   .{ .start = 0x0A70, .end = 0x0A71 },   .{ .start = 0x0A75, .end = 0x0A75 },   .{ .start = 0x0A81, .end = 0x0A83 },
    .{ .start = 0x0ABC, .end = 0x0ABC },   .{ .start = 0x0ABE, .end = 0x0AC5 },   .{ .start = 0x0AC7, .end = 0x0AC9 },   .{ .start = 0x0ACB, .end = 0x0ACD },
    .{ .start = 0x0AE2, .end = 0x0AE3 },   .{ .start = 0x0AFA, .end = 0x0AFF },   .{ .start = 0x0B01, .end = 0x0B03 },   .{ .start = 0x0B3C, .end = 0x0B3C },
    .{ .start = 0x0B3E, .end = 0x0B44 },   .{ .start = 0x0B47, .end = 0x0B48 },   .{ .start = 0x0B4B, .end = 0x0B4D },   .{ .start = 0x0B55, .end = 0x0B57 },
    .{ .start = 0x0B62, .end = 0x0B63 },   .{ .start = 0x0B82, .end = 0x0B82 },   .{ .start = 0x0BBE, .end = 0x0BC2 },   .{ .start = 0x0BC6, .end = 0x0BC8 },
    .{ .start = 0x0BCA, .end = 0x0BCD },   .{ .start = 0x0BD7, .end = 0x0BD7 },   .{ .start = 0x0C00, .end = 0x0C04 },   .{ .start = 0x0C3C, .end = 0x0C3C },
    .{ .start = 0x0C3E, .end = 0x0C44 },   .{ .start = 0x0C46, .end = 0x0C48 },   .{ .start = 0x0C4A, .end = 0x0C4D },   .{ .start = 0x0C55, .end = 0x0C56 },
    .{ .start = 0x0C62, .end = 0x0C63 },   .{ .start = 0x0C81, .end = 0x0C83 },   .{ .start = 0x0CBC, .end = 0x0CBC },   .{ .start = 0x0CBE, .end = 0x0CC4 },
    .{ .start = 0x0CC6, .end = 0x0CC8 },   .{ .start = 0x0CCA, .end = 0x0CCD },   .{ .start = 0x0CD5, .end = 0x0CD6 },   .{ .start = 0x0CE2, .end = 0x0CE3 },
    .{ .start = 0x0CF3, .end = 0x0CF3 },   .{ .start = 0x0D00, .end = 0x0D03 },   .{ .start = 0x0D3B, .end = 0x0D3C },   .{ .start = 0x0D3E, .end = 0x0D44 },
    .{ .start = 0x0D46, .end = 0x0D48 },   .{ .start = 0x0D4A, .end = 0x0D4D },   .{ .start = 0x0D57, .end = 0x0D57 },   .{ .start = 0x0D62, .end = 0x0D63 },
    .{ .start = 0x0D81, .end = 0x0D83 },   .{ .start = 0x0DCA, .end = 0x0DCA },   .{ .start = 0x0DCF, .end = 0x0DD4 },   .{ .start = 0x0DD6, .end = 0x0DD6 },
    .{ .start = 0x0DD8, .end = 0x0DDF },   .{ .start = 0x0DF2, .end = 0x0DF3 },   .{ .start = 0x0E31, .end = 0x0E31 },   .{ .start = 0x0E34, .end = 0x0E3A },
    .{ .start = 0x0E47, .end = 0x0E4E },   .{ .start = 0x0EB1, .end = 0x0EB1 },   .{ .start = 0x0EB4, .end = 0x0EBC },   .{ .start = 0x0EC8, .end = 0x0ECE },
    .{ .start = 0x0F18, .end = 0x0F19 },   .{ .start = 0x0F35, .end = 0x0F35 },   .{ .start = 0x0F37, .end = 0x0F37 },   .{ .start = 0x0F39, .end = 0x0F39 },
    .{ .start = 0x0F3E, .end = 0x0F3F },   .{ .start = 0x0F71, .end = 0x0F84 },   .{ .start = 0x0F86, .end = 0x0F87 },   .{ .start = 0x0F8D, .end = 0x0F97 },
    .{ .start = 0x0F99, .end = 0x0FBC },   .{ .start = 0x0FC6, .end = 0x0FC6 },   .{ .start = 0x102B, .end = 0x103E },   .{ .start = 0x1056, .end = 0x1059 },
    .{ .start = 0x105E, .end = 0x1060 },   .{ .start = 0x1062, .end = 0x1064 },   .{ .start = 0x1067, .end = 0x106D },   .{ .start = 0x1071, .end = 0x1074 },
    .{ .start = 0x1082, .end = 0x108D },   .{ .start = 0x108F, .end = 0x108F },   .{ .start = 0x109A, .end = 0x109D },   .{ .start = 0x135D, .end = 0x135F },
    .{ .start = 0x1712, .end = 0x1715 },   .{ .start = 0x1732, .end = 0x1734 },   .{ .start = 0x1752, .end = 0x1753 },   .{ .start = 0x1772, .end = 0x1773 },
    .{ .start = 0x17B4, .end = 0x17D3 },   .{ .start = 0x17DD, .end = 0x17DD },   .{ .start = 0x180B, .end = 0x180D },   .{ .start = 0x180F, .end = 0x180F },
    .{ .start = 0x1885, .end = 0x1886 },   .{ .start = 0x18A9, .end = 0x18A9 },   .{ .start = 0x1920, .end = 0x192B },   .{ .start = 0x1930, .end = 0x193B },
    .{ .start = 0x1A17, .end = 0x1A1B },   .{ .start = 0x1A55, .end = 0x1A5E },   .{ .start = 0x1A60, .end = 0x1A7C },   .{ .start = 0x1A7F, .end = 0x1A7F },
    .{ .start = 0x1AB0, .end = 0x1ADD },   .{ .start = 0x1AE0, .end = 0x1AEB },   .{ .start = 0x1B00, .end = 0x1B04 },   .{ .start = 0x1B34, .end = 0x1B44 },
    .{ .start = 0x1B6B, .end = 0x1B73 },   .{ .start = 0x1B80, .end = 0x1B82 },   .{ .start = 0x1BA1, .end = 0x1BAD },   .{ .start = 0x1BE6, .end = 0x1BF3 },
    .{ .start = 0x1C24, .end = 0x1C37 },   .{ .start = 0x1CD0, .end = 0x1CD2 },   .{ .start = 0x1CD4, .end = 0x1CE8 },   .{ .start = 0x1CED, .end = 0x1CED },
    .{ .start = 0x1CF4, .end = 0x1CF4 },   .{ .start = 0x1CF7, .end = 0x1CF9 },   .{ .start = 0x1DC0, .end = 0x1DFF },   .{ .start = 0x20D0, .end = 0x20F0 },
    .{ .start = 0x2CEF, .end = 0x2CF1 },   .{ .start = 0x2D7F, .end = 0x2D7F },   .{ .start = 0x2DE0, .end = 0x2DFF },   .{ .start = 0x302A, .end = 0x302F },
    .{ .start = 0x3099, .end = 0x309A },   .{ .start = 0xA66F, .end = 0xA672 },   .{ .start = 0xA674, .end = 0xA67D },   .{ .start = 0xA69E, .end = 0xA69F },
    .{ .start = 0xA6F0, .end = 0xA6F1 },   .{ .start = 0xA802, .end = 0xA802 },   .{ .start = 0xA806, .end = 0xA806 },   .{ .start = 0xA80B, .end = 0xA80B },
    .{ .start = 0xA823, .end = 0xA827 },   .{ .start = 0xA82C, .end = 0xA82C },   .{ .start = 0xA880, .end = 0xA881 },   .{ .start = 0xA8B4, .end = 0xA8C5 },
    .{ .start = 0xA8E0, .end = 0xA8F1 },   .{ .start = 0xA8FF, .end = 0xA8FF },   .{ .start = 0xA926, .end = 0xA92D },   .{ .start = 0xA947, .end = 0xA953 },
    .{ .start = 0xA980, .end = 0xA983 },   .{ .start = 0xA9B3, .end = 0xA9C0 },   .{ .start = 0xA9E5, .end = 0xA9E5 },   .{ .start = 0xAA29, .end = 0xAA36 },
    .{ .start = 0xAA43, .end = 0xAA43 },   .{ .start = 0xAA4C, .end = 0xAA4D },   .{ .start = 0xAA7B, .end = 0xAA7D },   .{ .start = 0xAAB0, .end = 0xAAB0 },
    .{ .start = 0xAAB2, .end = 0xAAB4 },   .{ .start = 0xAAB7, .end = 0xAAB8 },   .{ .start = 0xAABE, .end = 0xAABF },   .{ .start = 0xAAC1, .end = 0xAAC1 },
    .{ .start = 0xAAEB, .end = 0xAAEF },   .{ .start = 0xAAF5, .end = 0xAAF6 },   .{ .start = 0xABE3, .end = 0xABEA },   .{ .start = 0xABEC, .end = 0xABED },
    .{ .start = 0xFB1E, .end = 0xFB1E },   .{ .start = 0xFE00, .end = 0xFE0F },   .{ .start = 0xFE20, .end = 0xFE2F },   .{ .start = 0x101FD, .end = 0x101FD },
    .{ .start = 0x102E0, .end = 0x102E0 }, .{ .start = 0x10376, .end = 0x1037A }, .{ .start = 0x10A01, .end = 0x10A03 }, .{ .start = 0x10A05, .end = 0x10A06 },
    .{ .start = 0x10A0C, .end = 0x10A0F }, .{ .start = 0x10A38, .end = 0x10A3A }, .{ .start = 0x10A3F, .end = 0x10A3F }, .{ .start = 0x10AE5, .end = 0x10AE6 },
    .{ .start = 0x10D24, .end = 0x10D27 }, .{ .start = 0x10D69, .end = 0x10D6D }, .{ .start = 0x10EAB, .end = 0x10EAC }, .{ .start = 0x10EFA, .end = 0x10EFF },
    .{ .start = 0x10F46, .end = 0x10F50 }, .{ .start = 0x10F82, .end = 0x10F85 }, .{ .start = 0x11000, .end = 0x11002 }, .{ .start = 0x11038, .end = 0x11046 },
    .{ .start = 0x11070, .end = 0x11070 }, .{ .start = 0x11073, .end = 0x11074 }, .{ .start = 0x1107F, .end = 0x11082 }, .{ .start = 0x110B0, .end = 0x110BA },
    .{ .start = 0x110C2, .end = 0x110C2 }, .{ .start = 0x11100, .end = 0x11102 }, .{ .start = 0x11127, .end = 0x11134 }, .{ .start = 0x11145, .end = 0x11146 },
    .{ .start = 0x11173, .end = 0x11173 }, .{ .start = 0x11180, .end = 0x11182 }, .{ .start = 0x111B3, .end = 0x111C0 }, .{ .start = 0x111C9, .end = 0x111CC },
    .{ .start = 0x111CE, .end = 0x111CF }, .{ .start = 0x1122C, .end = 0x11237 }, .{ .start = 0x1123E, .end = 0x1123E }, .{ .start = 0x11241, .end = 0x11241 },
    .{ .start = 0x112DF, .end = 0x112EA }, .{ .start = 0x11300, .end = 0x11303 }, .{ .start = 0x1133B, .end = 0x1133C }, .{ .start = 0x1133E, .end = 0x11344 },
    .{ .start = 0x11347, .end = 0x11348 }, .{ .start = 0x1134B, .end = 0x1134D }, .{ .start = 0x11357, .end = 0x11357 }, .{ .start = 0x11362, .end = 0x11363 },
    .{ .start = 0x11366, .end = 0x1136C }, .{ .start = 0x11370, .end = 0x11374 }, .{ .start = 0x113B8, .end = 0x113C0 }, .{ .start = 0x113C2, .end = 0x113C2 },
    .{ .start = 0x113C5, .end = 0x113C5 }, .{ .start = 0x113C7, .end = 0x113CA }, .{ .start = 0x113CC, .end = 0x113D0 }, .{ .start = 0x113D2, .end = 0x113D2 },
    .{ .start = 0x113E1, .end = 0x113E2 }, .{ .start = 0x11435, .end = 0x11446 }, .{ .start = 0x1145E, .end = 0x1145E }, .{ .start = 0x114B0, .end = 0x114C3 },
    .{ .start = 0x115AF, .end = 0x115B5 }, .{ .start = 0x115B8, .end = 0x115C0 }, .{ .start = 0x115DC, .end = 0x115DD }, .{ .start = 0x11630, .end = 0x11640 },
    .{ .start = 0x116AB, .end = 0x116B7 }, .{ .start = 0x1171D, .end = 0x1172B }, .{ .start = 0x1182C, .end = 0x1183A }, .{ .start = 0x11930, .end = 0x11935 },
    .{ .start = 0x11937, .end = 0x11938 }, .{ .start = 0x1193B, .end = 0x1193E }, .{ .start = 0x11940, .end = 0x11940 }, .{ .start = 0x11942, .end = 0x11943 },
    .{ .start = 0x119D1, .end = 0x119D7 }, .{ .start = 0x119DA, .end = 0x119E0 }, .{ .start = 0x119E4, .end = 0x119E4 }, .{ .start = 0x11A01, .end = 0x11A0A },
    .{ .start = 0x11A33, .end = 0x11A39 }, .{ .start = 0x11A3B, .end = 0x11A3E }, .{ .start = 0x11A47, .end = 0x11A47 }, .{ .start = 0x11A51, .end = 0x11A5B },
    .{ .start = 0x11A8A, .end = 0x11A99 }, .{ .start = 0x11B60, .end = 0x11B67 }, .{ .start = 0x11C2F, .end = 0x11C36 }, .{ .start = 0x11C38, .end = 0x11C3F },
    .{ .start = 0x11C92, .end = 0x11CA7 }, .{ .start = 0x11CA9, .end = 0x11CB6 }, .{ .start = 0x11D31, .end = 0x11D36 }, .{ .start = 0x11D3A, .end = 0x11D3A },
    .{ .start = 0x11D3C, .end = 0x11D3D }, .{ .start = 0x11D3F, .end = 0x11D45 }, .{ .start = 0x11D47, .end = 0x11D47 }, .{ .start = 0x11D8A, .end = 0x11D8E },
    .{ .start = 0x11D90, .end = 0x11D91 }, .{ .start = 0x11D93, .end = 0x11D97 }, .{ .start = 0x11EF3, .end = 0x11EF6 }, .{ .start = 0x11F00, .end = 0x11F01 },
    .{ .start = 0x11F03, .end = 0x11F03 }, .{ .start = 0x11F34, .end = 0x11F3A }, .{ .start = 0x11F3E, .end = 0x11F42 }, .{ .start = 0x11F5A, .end = 0x11F5A },
    .{ .start = 0x13440, .end = 0x13440 }, .{ .start = 0x13447, .end = 0x13455 }, .{ .start = 0x1611E, .end = 0x1612F }, .{ .start = 0x16AF0, .end = 0x16AF4 },
    .{ .start = 0x16B30, .end = 0x16B36 }, .{ .start = 0x16F4F, .end = 0x16F4F }, .{ .start = 0x16F51, .end = 0x16F87 }, .{ .start = 0x16F8F, .end = 0x16F92 },
    .{ .start = 0x16FE4, .end = 0x16FE4 }, .{ .start = 0x16FF0, .end = 0x16FF1 }, .{ .start = 0x1BC9D, .end = 0x1BC9E }, .{ .start = 0x1CF00, .end = 0x1CF2D },
    .{ .start = 0x1CF30, .end = 0x1CF46 }, .{ .start = 0x1D165, .end = 0x1D169 }, .{ .start = 0x1D16D, .end = 0x1D172 }, .{ .start = 0x1D17B, .end = 0x1D182 },
    .{ .start = 0x1D185, .end = 0x1D18B }, .{ .start = 0x1D1AA, .end = 0x1D1AD }, .{ .start = 0x1D242, .end = 0x1D244 }, .{ .start = 0x1DA00, .end = 0x1DA36 },
    .{ .start = 0x1DA3B, .end = 0x1DA6C }, .{ .start = 0x1DA75, .end = 0x1DA75 }, .{ .start = 0x1DA84, .end = 0x1DA84 }, .{ .start = 0x1DA9B, .end = 0x1DA9F },
    .{ .start = 0x1DAA1, .end = 0x1DAAF }, .{ .start = 0x1E000, .end = 0x1E006 }, .{ .start = 0x1E008, .end = 0x1E018 }, .{ .start = 0x1E01B, .end = 0x1E021 },
    .{ .start = 0x1E023, .end = 0x1E024 }, .{ .start = 0x1E026, .end = 0x1E02A }, .{ .start = 0x1E08F, .end = 0x1E08F }, .{ .start = 0x1E130, .end = 0x1E136 },
    .{ .start = 0x1E2AE, .end = 0x1E2AE }, .{ .start = 0x1E2EC, .end = 0x1E2EF }, .{ .start = 0x1E4EC, .end = 0x1E4EF }, .{ .start = 0x1E5EE, .end = 0x1E5EF },
    .{ .start = 0x1E6E3, .end = 0x1E6E3 }, .{ .start = 0x1E6E6, .end = 0x1E6E6 }, .{ .start = 0x1E6EE, .end = 0x1E6EF }, .{ .start = 0x1E6F5, .end = 0x1E6F5 },
    .{ .start = 0x1E8D0, .end = 0x1E8D6 }, .{ .start = 0x1E944, .end = 0x1E94A }, .{ .start = 0xE0100, .end = 0xE01EF },
};

const O200K_NUMBER_RANGES = [_]CodepointRange{
    .{ .start = 0x00B2, .end = 0x00B3 },   .{ .start = 0x00B9, .end = 0x00B9 },   .{ .start = 0x00BC, .end = 0x00BE },   .{ .start = 0x0660, .end = 0x0669 },
    .{ .start = 0x06F0, .end = 0x06F9 },   .{ .start = 0x07C0, .end = 0x07C9 },   .{ .start = 0x0966, .end = 0x096F },   .{ .start = 0x09E6, .end = 0x09EF },
    .{ .start = 0x09F4, .end = 0x09F9 },   .{ .start = 0x0A66, .end = 0x0A6F },   .{ .start = 0x0AE6, .end = 0x0AEF },   .{ .start = 0x0B66, .end = 0x0B6F },
    .{ .start = 0x0B72, .end = 0x0B77 },   .{ .start = 0x0BE6, .end = 0x0BF2 },   .{ .start = 0x0C66, .end = 0x0C6F },   .{ .start = 0x0C78, .end = 0x0C7E },
    .{ .start = 0x0CE6, .end = 0x0CEF },   .{ .start = 0x0D58, .end = 0x0D5E },   .{ .start = 0x0D66, .end = 0x0D78 },   .{ .start = 0x0DE6, .end = 0x0DEF },
    .{ .start = 0x0E50, .end = 0x0E59 },   .{ .start = 0x0ED0, .end = 0x0ED9 },   .{ .start = 0x0F20, .end = 0x0F33 },   .{ .start = 0x1040, .end = 0x1049 },
    .{ .start = 0x1090, .end = 0x1099 },   .{ .start = 0x1369, .end = 0x137C },   .{ .start = 0x16EE, .end = 0x16F0 },   .{ .start = 0x17E0, .end = 0x17E9 },
    .{ .start = 0x17F0, .end = 0x17F9 },   .{ .start = 0x1810, .end = 0x1819 },   .{ .start = 0x1946, .end = 0x194F },   .{ .start = 0x19D0, .end = 0x19DA },
    .{ .start = 0x1A80, .end = 0x1A89 },   .{ .start = 0x1A90, .end = 0x1A99 },   .{ .start = 0x1B50, .end = 0x1B59 },   .{ .start = 0x1BB0, .end = 0x1BB9 },
    .{ .start = 0x1C40, .end = 0x1C49 },   .{ .start = 0x1C50, .end = 0x1C59 },   .{ .start = 0x2070, .end = 0x2070 },   .{ .start = 0x2074, .end = 0x2079 },
    .{ .start = 0x2080, .end = 0x2089 },   .{ .start = 0x2150, .end = 0x2182 },   .{ .start = 0x2185, .end = 0x2189 },   .{ .start = 0x2460, .end = 0x249B },
    .{ .start = 0x24EA, .end = 0x24FF },   .{ .start = 0x2776, .end = 0x2793 },   .{ .start = 0x2CFD, .end = 0x2CFD },   .{ .start = 0x3007, .end = 0x3007 },
    .{ .start = 0x3021, .end = 0x3029 },   .{ .start = 0x3038, .end = 0x303A },   .{ .start = 0x3192, .end = 0x3195 },   .{ .start = 0x3220, .end = 0x3229 },
    .{ .start = 0x3248, .end = 0x324F },   .{ .start = 0x3251, .end = 0x325F },   .{ .start = 0x3280, .end = 0x3289 },   .{ .start = 0x32B1, .end = 0x32BF },
    .{ .start = 0xA620, .end = 0xA629 },   .{ .start = 0xA6E6, .end = 0xA6EF },   .{ .start = 0xA830, .end = 0xA835 },   .{ .start = 0xA8D0, .end = 0xA8D9 },
    .{ .start = 0xA900, .end = 0xA909 },   .{ .start = 0xA9D0, .end = 0xA9D9 },   .{ .start = 0xA9F0, .end = 0xA9F9 },   .{ .start = 0xAA50, .end = 0xAA59 },
    .{ .start = 0xABF0, .end = 0xABF9 },   .{ .start = 0xFF10, .end = 0xFF19 },   .{ .start = 0x10107, .end = 0x10133 }, .{ .start = 0x10140, .end = 0x10178 },
    .{ .start = 0x1018A, .end = 0x1018B }, .{ .start = 0x102E1, .end = 0x102FB }, .{ .start = 0x10320, .end = 0x10323 }, .{ .start = 0x10341, .end = 0x10341 },
    .{ .start = 0x1034A, .end = 0x1034A }, .{ .start = 0x103D1, .end = 0x103D5 }, .{ .start = 0x104A0, .end = 0x104A9 }, .{ .start = 0x10858, .end = 0x1085F },
    .{ .start = 0x10879, .end = 0x1087F }, .{ .start = 0x108A7, .end = 0x108AF }, .{ .start = 0x108FB, .end = 0x108FF }, .{ .start = 0x10916, .end = 0x1091B },
    .{ .start = 0x109BC, .end = 0x109BD }, .{ .start = 0x109C0, .end = 0x109CF }, .{ .start = 0x109D2, .end = 0x109FF }, .{ .start = 0x10A40, .end = 0x10A48 },
    .{ .start = 0x10A7D, .end = 0x10A7E }, .{ .start = 0x10A9D, .end = 0x10A9F }, .{ .start = 0x10AEB, .end = 0x10AEF }, .{ .start = 0x10B58, .end = 0x10B5F },
    .{ .start = 0x10B78, .end = 0x10B7F }, .{ .start = 0x10BA9, .end = 0x10BAF }, .{ .start = 0x10CFA, .end = 0x10CFF }, .{ .start = 0x10D30, .end = 0x10D39 },
    .{ .start = 0x10D40, .end = 0x10D49 }, .{ .start = 0x10E60, .end = 0x10E7E }, .{ .start = 0x10F1D, .end = 0x10F26 }, .{ .start = 0x10F51, .end = 0x10F54 },
    .{ .start = 0x10FC5, .end = 0x10FCB }, .{ .start = 0x11052, .end = 0x1106F }, .{ .start = 0x110F0, .end = 0x110F9 }, .{ .start = 0x11136, .end = 0x1113F },
    .{ .start = 0x111D0, .end = 0x111D9 }, .{ .start = 0x111E1, .end = 0x111F4 }, .{ .start = 0x112F0, .end = 0x112F9 }, .{ .start = 0x11450, .end = 0x11459 },
    .{ .start = 0x114D0, .end = 0x114D9 }, .{ .start = 0x11650, .end = 0x11659 }, .{ .start = 0x116C0, .end = 0x116C9 }, .{ .start = 0x116D0, .end = 0x116E3 },
    .{ .start = 0x11730, .end = 0x1173B }, .{ .start = 0x118E0, .end = 0x118F2 }, .{ .start = 0x11950, .end = 0x11959 }, .{ .start = 0x11BF0, .end = 0x11BF9 },
    .{ .start = 0x11C50, .end = 0x11C6C }, .{ .start = 0x11D50, .end = 0x11D59 }, .{ .start = 0x11DA0, .end = 0x11DA9 }, .{ .start = 0x11DE0, .end = 0x11DE9 },
    .{ .start = 0x11F50, .end = 0x11F59 }, .{ .start = 0x11FC0, .end = 0x11FD4 }, .{ .start = 0x12400, .end = 0x1246E }, .{ .start = 0x16130, .end = 0x16139 },
    .{ .start = 0x16A60, .end = 0x16A69 }, .{ .start = 0x16AC0, .end = 0x16AC9 }, .{ .start = 0x16B50, .end = 0x16B59 }, .{ .start = 0x16B5B, .end = 0x16B61 },
    .{ .start = 0x16D70, .end = 0x16D79 }, .{ .start = 0x16E80, .end = 0x16E96 }, .{ .start = 0x16FF4, .end = 0x16FF6 }, .{ .start = 0x1CCF0, .end = 0x1CCF9 },
    .{ .start = 0x1D2C0, .end = 0x1D2D3 }, .{ .start = 0x1D2E0, .end = 0x1D2F3 }, .{ .start = 0x1D360, .end = 0x1D378 }, .{ .start = 0x1D7CE, .end = 0x1D7FF },
    .{ .start = 0x1E140, .end = 0x1E149 }, .{ .start = 0x1E2F0, .end = 0x1E2F9 }, .{ .start = 0x1E4F0, .end = 0x1E4F9 }, .{ .start = 0x1E5F1, .end = 0x1E5FA },
    .{ .start = 0x1E8C7, .end = 0x1E8CF }, .{ .start = 0x1E950, .end = 0x1E959 }, .{ .start = 0x1EC71, .end = 0x1ECAB }, .{ .start = 0x1ECAD, .end = 0x1ECAF },
    .{ .start = 0x1ECB1, .end = 0x1ECB4 }, .{ .start = 0x1ED01, .end = 0x1ED2D }, .{ .start = 0x1ED2F, .end = 0x1ED3D }, .{ .start = 0x1F100, .end = 0x1F10C },
    .{ .start = 0x1FBF0, .end = 0x1FBF9 },
};

fn assertSortedCatRanges(comptime ranges: []const LetterCatRange) void {
    @setEvalBranchQuota(10_000);
    for (ranges, 0..) |r, i| {
        if (r.start > r.end) @compileError("invalid Unicode letter category range");
        if (i > 0 and ranges[i - 1].end >= r.start) @compileError("unsorted or overlapping letter category ranges");
    }
}

comptime {
    assertSortedRanges(&NON_ASCII_LETTER_RANGES);
    assertSortedRanges(&NON_ASCII_MARK_RANGES);
    assertSortedRanges(&NON_ASCII_NUMBER_RANGES);
    assertSortedCatRanges(&O200K_LETTER_CAT_RANGES);
    assertSortedRanges(&O200K_MARK_RANGES);
    assertSortedRanges(&O200K_NUMBER_RANGES);
}

inline fn o200kLetterCat(cp: u21) ?O200kLetterCat {
    if (cp < 0x80) {
        if (cp >= 'A' and cp <= 'Z') return .upper_title;
        if (cp >= 'a' and cp <= 'z') return .lower;
        return null;
    }
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or (cp >= 0xAC00 and cp <= 0xD7A3)) return .other;
    var lo: usize = 0;
    var hi: usize = O200K_LETTER_CAT_RANGES.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        const r = O200K_LETTER_CAT_RANGES[mid];
        if (cp < r.start) {
            hi = mid;
        } else if (cp > r.end) {
            lo = mid + 1;
        } else {
            return r.cat;
        }
    }
    return null;
}

inline fn isO200kLetter(cp: u21) bool {
    return o200kLetterCat(cp) != null;
}

inline fn isO200kMark(cp: u21) bool {
    if (cp < 0x80) return false;
    return inRanges(&O200K_MARK_RANGES, cp);
}

inline fn isO200kNumber(cp: u21) bool {
    if (cp < 0x80) return cp >= '0' and cp <= '9';
    return inRanges(&O200K_NUMBER_RANGES, cp);
}

inline fn isO200kLetterOrMark(cp: u21) bool {
    return isO200kLetter(cp) or isO200kMark(cp);
}

inline fn isWhitespace(cp: u21) bool {
    if (cp < 0x80) return cp == ' ' or (cp >= 0x09 and cp <= 0x0D);
    return switch (cp) {
        0x0085, 0x00A0, 0x1680, 0x2000...0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000 => true,
        else => false,
    };
}

inline fn isLetter(cp: u21) bool {
    if (cp < 0x80) return (cp >= 'A' and cp <= 'Z') or (cp >= 'a' and cp <= 'z');
    return inRanges(&NON_ASCII_LETTER_RANGES, cp);
}

inline fn isMark(cp: u21) bool {
    if (cp < 0x80) return false;
    return inRanges(&NON_ASCII_MARK_RANGES, cp);
}

inline fn isNumber(cp: u21) bool {
    if (cp < 0x80) return cp >= '0' and cp <= '9';
    return inRanges(&NON_ASCII_NUMBER_RANGES, cp);
}

inline fn isLetterOrMark(cp: u21) bool {
    return isLetter(cp) or isMark(cp);
}

inline fn isPunctOrSymbol(cp: u21) bool {
    return !isWhitespace(cp) and !isLetter(cp) and !isMark(cp) and !isNumber(cp);
}

const DecodedCp = struct {
    cp: u21,
    len: usize,
};

inline fn decodeNextCp(bytes: []const u8) DecodedCp {
    std.debug.assert(bytes.len > 0);
    if (bytes[0] < 0x80) return .{ .cp = bytes[0], .len = 1 };
    const seq_len = std.unicode.utf8ByteSequenceLength(bytes[0]) catch return .{ .cp = 0xFFFD, .len = 1 };
    if (seq_len > bytes.len) return .{ .cp = 0xFFFD, .len = 1 };
    const cp = std.unicode.utf8Decode(bytes[0..seq_len]) catch return .{ .cp = 0xFFFD, .len = 1 };
    return .{ .cp = cp, .len = seq_len };
}

inline fn asciiLower(b: u8) u8 {
    return if (b >= 'A' and b <= 'Z') b + ('a' - 'A') else b;
}

inline fn isAsciiAlpha(b: u8) bool {
    return (b >= 'A' and b <= 'Z') or (b >= 'a' and b <= 'z');
}

/// ASCII punctuation class from DeepSeek's pre-tokenizer regex:
/// `[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~]`
inline fn isAsciiPunct(b: u8) bool {
    return (b >= '!' and b <= '/') or
        (b >= ':' and b <= '@') or
        (b >= '[' and b <= '`') or
        (b >= '{' and b <= '~');
}

/// CJK / Kana range isolated in stage 2 of DeepSeek's pre-tokenizer:
/// `[\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+`
inline fn isDeepSeekCjk(cp: u21) bool {
    return (cp >= 0x4E00 and cp <= 0x9FA5) or
        (cp >= 0x3040 and cp <= 0x309F) or
        (cp >= 0x30A0 and cp <= 0x30FF);
}

inline fn isDeepSeekStage1Or2(cp: u21) bool {
    return isNumber(cp) or isDeepSeekCjk(cp);
}

// ---------------------------------------------------------------------------
// C23: ARM64 NEON BPE pre-tokenizer class-boundary scanners (ADR-0023)
//   - Explicit 16-byte / 64-byte unrolled `@Vector(16, u8)` NEON fast paths
//   - `shrn`-equivalent nibble-mask reduction (compatible with `-Dcpu=apple_m1`)
//   - Clean `< 16 B` scalar tail fallback, zero dynamic heap allocation
// ---------------------------------------------------------------------------

const NeonVec16 = @Vector(16, u8);
const NeonVec8u16 = @Vector(8, u16);
const NeonVec8u8 = @Vector(8, u8);
const NeonShift4: @Vector(8, u4) = @splat(4);

inline fn neonBoolToMask(cond: @Vector(16, bool)) NeonVec16 {
    return @select(u8, cond, @as(NeonVec16, @splat(0xFF)), @as(NeonVec16, @splat(0)));
}

inline fn neonMaskToU64(mask: NeonVec16) u64 {
    const pairs: NeonVec8u16 = @bitCast(mask);
    const nibbles: NeonVec8u8 = @truncate(pairs >> NeonShift4);
    return @bitCast(nibbles);
}

inline fn neonNonAsciiAlphaMask(v: NeonVec16) NeonVec16 {
    const lower = v | @as(NeonVec16, @splat(0x20));
    const is_alpha = (lower -% @as(NeonVec16, @splat('a'))) <= @as(NeonVec16, @splat(25));
    return ~neonBoolToMask(is_alpha);
}

inline fn neonNonAsciiUpperMask(v: NeonVec16) NeonVec16 {
    const is_upper = (v -% @as(NeonVec16, @splat('A'))) <= @as(NeonVec16, @splat(25));
    return ~neonBoolToMask(is_upper);
}

inline fn neonNonAsciiLowerMask(v: NeonVec16) NeonVec16 {
    const is_lower = (v -% @as(NeonVec16, @splat('a'))) <= @as(NeonVec16, @splat(25));
    return ~neonBoolToMask(is_lower);
}

inline fn neonNonAsciiPunctMask(v: NeonVec16) NeonVec16 {
    const is_non_ascii = neonBoolToMask(v >= @as(NeonVec16, @splat(0x80)));
    const lower = v | @as(NeonVec16, @splat(0x20));
    const is_alpha = neonBoolToMask((lower -% @as(NeonVec16, @splat('a'))) <= @as(NeonVec16, @splat(25)));
    const is_digit = neonBoolToMask((v -% @as(NeonVec16, @splat('0'))) <= @as(NeonVec16, @splat(9)));
    const is_ws = neonBoolToMask(v == @as(NeonVec16, @splat(' '))) |
        neonBoolToMask((v -% @as(NeonVec16, @splat(0x09))) <= @as(NeonVec16, @splat(4)));
    return is_non_ascii | is_alpha | is_digit | is_ws;
}

inline fn neonNonAsciiWsMask(v: NeonVec16) NeonVec16 {
    const is_ws = neonBoolToMask(v == @as(NeonVec16, @splat(' '))) |
        neonBoolToMask((v -% @as(NeonVec16, @splat(0x09))) <= @as(NeonVec16, @splat(4)));
    return ~is_ws;
}

inline fn neonCrLfMask(v: NeonVec16) NeonVec16 {
    return neonBoolToMask(v == @as(NeonVec16, @splat('\r'))) |
        neonBoolToMask(v == @as(NeonVec16, @splat('\n')));
}

/// Scans the leading contiguous ASCII `[A-Za-z]` run using 64 B / 16 B NEON.
pub fn neonScanAsciiAlpha(bytes: []const u8) usize {
    var pos: usize = 0;
    const len = bytes.len;

    while (pos + 64 <= len) : (pos += 64) {
        const m0 = neonNonAsciiAlphaMask(bytes[pos..][0..16].*);
        const m1 = neonNonAsciiAlphaMask(bytes[pos + 16 ..][0..16].*);
        const m2 = neonNonAsciiAlphaMask(bytes[pos + 32 ..][0..16].*);
        const m3 = neonNonAsciiAlphaMask(bytes[pos + 48 ..][0..16].*);
        const any = (m0 | m1) | (m2 | m3);
        if (@reduce(.Max, any) != 0) {
            const b0 = neonMaskToU64(m0);
            if (b0 != 0) return pos + (@as(usize, @ctz(b0)) >> 2);
            const b1 = neonMaskToU64(m1);
            if (b1 != 0) return pos + 16 + (@as(usize, @ctz(b1)) >> 2);
            const b2 = neonMaskToU64(m2);
            if (b2 != 0) return pos + 32 + (@as(usize, @ctz(b2)) >> 2);
            const b3 = neonMaskToU64(m3);
            return pos + 48 + (@as(usize, @ctz(b3)) >> 2);
        }
    }

    while (pos + 16 <= len) : (pos += 16) {
        const m = neonNonAsciiAlphaMask(bytes[pos..][0..16].*);
        const u = neonMaskToU64(m);
        if (u != 0) return pos + (@as(usize, @ctz(u)) >> 2);
    }

    while (pos < len and isAsciiAlpha(bytes[pos])) : (pos += 1) {}
    return pos;
}

/// Scans the leading contiguous ASCII uppercase `[A-Z]` run using 64 B / 16 B NEON.
pub fn neonScanAsciiUpper(bytes: []const u8) usize {
    var pos: usize = 0;
    const len = bytes.len;

    while (pos + 64 <= len) : (pos += 64) {
        const m0 = neonNonAsciiUpperMask(bytes[pos..][0..16].*);
        const m1 = neonNonAsciiUpperMask(bytes[pos + 16 ..][0..16].*);
        const m2 = neonNonAsciiUpperMask(bytes[pos + 32 ..][0..16].*);
        const m3 = neonNonAsciiUpperMask(bytes[pos + 48 ..][0..16].*);
        const any = (m0 | m1) | (m2 | m3);
        if (@reduce(.Max, any) != 0) {
            const b0 = neonMaskToU64(m0);
            if (b0 != 0) return pos + (@as(usize, @ctz(b0)) >> 2);
            const b1 = neonMaskToU64(m1);
            if (b1 != 0) return pos + 16 + (@as(usize, @ctz(b1)) >> 2);
            const b2 = neonMaskToU64(m2);
            if (b2 != 0) return pos + 32 + (@as(usize, @ctz(b2)) >> 2);
            const b3 = neonMaskToU64(m3);
            return pos + 48 + (@as(usize, @ctz(b3)) >> 2);
        }
    }

    while (pos + 16 <= len) : (pos += 16) {
        const m = neonNonAsciiUpperMask(bytes[pos..][0..16].*);
        const u = neonMaskToU64(m);
        if (u != 0) return pos + (@as(usize, @ctz(u)) >> 2);
    }

    while (pos < len and bytes[pos] >= 'A' and bytes[pos] <= 'Z') : (pos += 1) {}
    return pos;
}

/// Scans the leading contiguous ASCII lowercase `[a-z]` run using 64 B / 16 B NEON.
pub fn neonScanAsciiLower(bytes: []const u8) usize {
    var pos: usize = 0;
    const len = bytes.len;

    while (pos + 64 <= len) : (pos += 64) {
        const m0 = neonNonAsciiLowerMask(bytes[pos..][0..16].*);
        const m1 = neonNonAsciiLowerMask(bytes[pos + 16 ..][0..16].*);
        const m2 = neonNonAsciiLowerMask(bytes[pos + 32 ..][0..16].*);
        const m3 = neonNonAsciiLowerMask(bytes[pos + 48 ..][0..16].*);
        const any = (m0 | m1) | (m2 | m3);
        if (@reduce(.Max, any) != 0) {
            const b0 = neonMaskToU64(m0);
            if (b0 != 0) return pos + (@as(usize, @ctz(b0)) >> 2);
            const b1 = neonMaskToU64(m1);
            if (b1 != 0) return pos + 16 + (@as(usize, @ctz(b1)) >> 2);
            const b2 = neonMaskToU64(m2);
            if (b2 != 0) return pos + 32 + (@as(usize, @ctz(b2)) >> 2);
            const b3 = neonMaskToU64(m3);
            return pos + 48 + (@as(usize, @ctz(b3)) >> 2);
        }
    }

    while (pos + 16 <= len) : (pos += 16) {
        const m = neonNonAsciiLowerMask(bytes[pos..][0..16].*);
        const u = neonMaskToU64(m);
        if (u != 0) return pos + (@as(usize, @ctz(u)) >> 2);
    }

    while (pos < len and bytes[pos] >= 'a' and bytes[pos] <= 'z') : (pos += 1) {}
    return pos;
}

/// Scans the leading contiguous ASCII `isPunctOrSymbol` run (`< 0x80` and not
/// whitespace, letter, or digit) using 64 B / 16 B NEON.
pub fn neonScanAsciiPunctOrSym(bytes: []const u8) usize {
    var pos: usize = 0;
    const len = bytes.len;

    while (pos + 64 <= len) : (pos += 64) {
        const m0 = neonNonAsciiPunctMask(bytes[pos..][0..16].*);
        const m1 = neonNonAsciiPunctMask(bytes[pos + 16 ..][0..16].*);
        const m2 = neonNonAsciiPunctMask(bytes[pos + 32 ..][0..16].*);
        const m3 = neonNonAsciiPunctMask(bytes[pos + 48 ..][0..16].*);
        const any = (m0 | m1) | (m2 | m3);
        if (@reduce(.Max, any) != 0) {
            const b0 = neonMaskToU64(m0);
            if (b0 != 0) return pos + (@as(usize, @ctz(b0)) >> 2);
            const b1 = neonMaskToU64(m1);
            if (b1 != 0) return pos + 16 + (@as(usize, @ctz(b1)) >> 2);
            const b2 = neonMaskToU64(m2);
            if (b2 != 0) return pos + 32 + (@as(usize, @ctz(b2)) >> 2);
            const b3 = neonMaskToU64(m3);
            return pos + 48 + (@as(usize, @ctz(b3)) >> 2);
        }
    }

    while (pos + 16 <= len) : (pos += 16) {
        const m = neonNonAsciiPunctMask(bytes[pos..][0..16].*);
        const u = neonMaskToU64(m);
        if (u != 0) return pos + (@as(usize, @ctz(u)) >> 2);
    }

    while (pos < len) : (pos += 1) {
        const b = bytes[pos];
        if (b >= 0x80 or !isPunctOrSymbol(b)) break;
    }
    return pos;
}

/// Scans the leading contiguous `[\r\n]` run using 64 B / 16 B NEON.
pub fn neonScanCrLf(bytes: []const u8) usize {
    var pos: usize = 0;
    const len = bytes.len;

    while (pos + 64 <= len) : (pos += 64) {
        const m0 = ~neonCrLfMask(bytes[pos..][0..16].*);
        const m1 = ~neonCrLfMask(bytes[pos + 16 ..][0..16].*);
        const m2 = ~neonCrLfMask(bytes[pos + 32 ..][0..16].*);
        const m3 = ~neonCrLfMask(bytes[pos + 48 ..][0..16].*);
        const any = (m0 | m1) | (m2 | m3);
        if (@reduce(.Max, any) != 0) {
            const b0 = neonMaskToU64(m0);
            if (b0 != 0) return pos + (@as(usize, @ctz(b0)) >> 2);
            const b1 = neonMaskToU64(m1);
            if (b1 != 0) return pos + 16 + (@as(usize, @ctz(b1)) >> 2);
            const b2 = neonMaskToU64(m2);
            if (b2 != 0) return pos + 32 + (@as(usize, @ctz(b2)) >> 2);
            const b3 = neonMaskToU64(m3);
            return pos + 48 + (@as(usize, @ctz(b3)) >> 2);
        }
    }

    while (pos + 16 <= len) : (pos += 16) {
        const m = ~neonCrLfMask(bytes[pos..][0..16].*);
        const u = neonMaskToU64(m);
        if (u != 0) return pos + (@as(usize, @ctz(u)) >> 2);
    }

    while (pos < len and (bytes[pos] == '\r' or bytes[pos] == '\n')) : (pos += 1) {}
    return pos;
}

const AsciiWsRun = struct {
    end: usize,
    after_last_nl: ?usize,
};

fn neonScanAsciiWhitespaceRun(bytes: []const u8) AsciiWsRun {
    var pos: usize = 0;
    const len = bytes.len;
    var after_last_nl: ?usize = null;

    while (pos + 64 <= len) : (pos += 64) {
        const v0: NeonVec16 = bytes[pos..][0..16].*;
        const v1: NeonVec16 = bytes[pos + 16 ..][0..16].*;
        const v2: NeonVec16 = bytes[pos + 32 ..][0..16].*;
        const v3: NeonVec16 = bytes[pos + 48 ..][0..16].*;
        const nw0 = neonNonAsciiWsMask(v0);
        const nw1 = neonNonAsciiWsMask(v1);
        const nw2 = neonNonAsciiWsMask(v2);
        const nw3 = neonNonAsciiWsMask(v3);
        const any_nw = (nw0 | nw1) | (nw2 | nw3);
        if (@reduce(.Max, any_nw) != 0) break;

        const nl0 = neonCrLfMask(v0);
        const nl1 = neonCrLfMask(v1);
        const nl2 = neonCrLfMask(v2);
        const nl3 = neonCrLfMask(v3);
        const any_nl = (nl0 | nl1) | (nl2 | nl3);
        if (@reduce(.Max, any_nl) != 0) {
            const b3 = neonMaskToU64(nl3);
            if (b3 != 0) {
                after_last_nl = pos + 48 + (15 - (@as(usize, @clz(b3)) >> 2)) + 1;
            } else {
                const b2 = neonMaskToU64(nl2);
                if (b2 != 0) {
                    after_last_nl = pos + 32 + (15 - (@as(usize, @clz(b2)) >> 2)) + 1;
                } else {
                    const b1 = neonMaskToU64(nl1);
                    if (b1 != 0) {
                        after_last_nl = pos + 16 + (15 - (@as(usize, @clz(b1)) >> 2)) + 1;
                    } else {
                        const b0 = neonMaskToU64(nl0);
                        after_last_nl = pos + (15 - (@as(usize, @clz(b0)) >> 2)) + 1;
                    }
                }
            }
        }
    }

    while (pos + 16 <= len) : (pos += 16) {
        const v: NeonVec16 = bytes[pos..][0..16].*;
        if (@reduce(.Max, neonNonAsciiWsMask(v)) != 0) break;
        const u_nl = neonMaskToU64(neonCrLfMask(v));
        if (u_nl != 0) {
            after_last_nl = pos + (15 - (@as(usize, @clz(u_nl)) >> 2)) + 1;
        }
    }

    while (pos < len) : (pos += 1) {
        const b = bytes[pos];
        if (b != ' ' and !(b >= 0x09 and b <= 0x0D)) break;
        if (b == '\r' or b == '\n') after_last_nl = pos + 1;
    }

    return .{ .end = pos, .after_last_nl = after_last_nl };
}

const CJK_LEAD_NIBBLES: u64 = 0x000F_00F0_0F00_F00F; // lanes {0, 3, 6, 9, 12}
const CJK_CONT_NIBBLES: u64 = 0x0FF0_FF0F_F0FF_0FF0; // lanes {1,2, 4,5, 7,8, 10,11, 13,14}

inline fn neonChunkIs5Utf8Cjk(v: NeonVec16, comptime include_kana: bool) bool {
    const v_next = @shuffle(u8, v, v, [16]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0 });
    const is_e5_e8 = neonBoolToMask((v >= @as(NeonVec16, @splat(0xE5))) & (v <= @as(NeonVec16, @splat(0xE8))));
    const is_e4 = neonBoolToMask((v == @as(NeonVec16, @splat(0xE4))) & (v_next >= @as(NeonVec16, @splat(0xB8))));
    const is_e9 = neonBoolToMask((v == @as(NeonVec16, @splat(0xE9))) & (v_next <= @as(NeonVec16, @splat(0xBD))));
    var lead = is_e5_e8 | is_e4 | is_e9;
    if (include_kana) {
        const is_kana = neonBoolToMask((v == @as(NeonVec16, @splat(0xE3))) &
            (v_next >= @as(NeonVec16, @splat(0x81))) &
            (v_next <= @as(NeonVec16, @splat(0x83))));
        lead = lead | is_kana;
    }
    const cont = neonBoolToMask((v & @as(NeonVec16, @splat(0xC0))) == @as(NeonVec16, @splat(0x80)));
    const lead_u64 = neonMaskToU64(lead);
    const cont_u64 = neonMaskToU64(cont);
    return ((lead_u64 & CJK_LEAD_NIBBLES) == CJK_LEAD_NIBBLES) and
        ((cont_u64 & CJK_CONT_NIBBLES) == CJK_CONT_NIBBLES);
}

/// Scans leading 3-byte UTF-8 CJK (`U+4E00..U+9F7F`, and optionally Kana
/// `U+3040..U+30FF`) in 60-byte (20-char) and 15-byte (5-char) NEON steps.
/// Any boundary codepoint (e.g. `U+9F80..U+9FA5`) exits cleanly to scalar UTF-8.
fn neonScanUtf8CjkFast(bytes: []const u8, comptime include_kana: bool) usize {
    var pos: usize = 0;
    const len = bytes.len;
    // 60-byte unrolled loop (4 x 15 B; requires pos + 61 <= len so the 4th 16 B load is in-bounds).
    while (pos + 61 <= len) : (pos += 60) {
        if (!neonChunkIs5Utf8Cjk(bytes[pos..][0..16].*, include_kana)) break;
        if (!neonChunkIs5Utf8Cjk(bytes[pos + 15 ..][0..16].*, include_kana)) {
            pos += 15;
            return pos;
        }
        if (!neonChunkIs5Utf8Cjk(bytes[pos + 30 ..][0..16].*, include_kana)) {
            pos += 30;
            return pos;
        }
        if (!neonChunkIs5Utf8Cjk(bytes[pos + 45 ..][0..16].*, include_kana)) {
            pos += 45;
            return pos;
        }
    }
    while (pos + 16 <= len) : (pos += 15) {
        if (!neonChunkIs5Utf8Cjk(bytes[pos..][0..16].*, include_kana)) break;
    }
    return pos;
}

const WsScanResult = struct {
    ws_end: usize,
    last_ws_start: usize,
    after_last_nl: ?usize,
};

fn scanWhitespaceRun(rem: []const u8) WsScanResult {
    var ws_end: usize = 0;
    var last_ws_start: usize = 0;
    var after_last_nl: ?usize = null;
    while (ws_end < rem.len) {
        if (rem[ws_end] < 0x80) {
            const run = neonScanAsciiWhitespaceRun(rem[ws_end..]);
            if (run.end == 0) break;
            if (run.after_last_nl) |rel_nl| after_last_nl = ws_end + rel_nl;
            last_ws_start = ws_end + run.end - 1;
            ws_end += run.end;
            continue;
        }
        const c = decodeNextCp(rem[ws_end..]);
        if (!isWhitespace(c.cp)) break;
        last_ws_start = ws_end;
        ws_end += c.len;
    }
    return .{
        .ws_end = ws_end,
        .last_ws_start = last_ws_start,
        .after_last_nl = after_last_nl,
    };
}

/// Pre-tokenization dialect auto-detected from HuggingFace `tokenizer.json`.
pub const PreTokenizerKind = enum(u8) {
    /// Qwen2 / Qwen3 / Qwen3.5 / Qwen3.8 single-digit `\p{N}` + contraction regex.
    qwen,
    /// DeepSeek V3 / V4 / V4.1 & Hy4 3-stage isolated split (`\p{N}{1,3}`, CJK/Kana, punct+ASCII word).
    deepseek,
};

/// High-level tokenizer model dialect auto-detected from `tokenizer.json`.
pub const TokenizerDialect = enum(u8) {
    qwen = 0,
    deepseek = 1,
    inkling2 = 2,
    hy4 = 3,
};

/// Deterministic zero-allocation scanner implementing Qwen (`\p{N}`),
/// DeepSeek (`\p{N}{1,3}` + CJK isolation + ASCII-punct-word), and
/// `o200k` / Inkling 2 (`Lu/Lt` vs `Ll` case splitting + `[\r\n/]*`) pre-tokenization.
pub const PreTokenIterator = struct {
    text: []const u8,
    pos: usize,
    kind: PreTokenizerKind = .qwen,
    use_o200k: bool = false,

    pub fn init(text: []const u8) PreTokenIterator {
        return .{ .text = text, .pos = 0, .kind = .qwen, .use_o200k = false };
    }

    pub fn initWithKind(text: []const u8, kind: PreTokenizerKind) PreTokenIterator {
        return .{ .text = text, .pos = 0, .kind = kind, .use_o200k = false };
    }

    pub fn initO200k(text: []const u8) PreTokenIterator {
        return .{ .text = text, .pos = 0, .kind = .qwen, .use_o200k = true };
    }

    pub fn next(self: *PreTokenIterator) ?[]const u8 {
        if (self.pos >= self.text.len) return null;
        const rem = self.text[self.pos..];
        const len = if (self.use_o200k)
            matchO200kPreToken(rem)
        else switch (self.kind) {
            .qwen => matchQwenPreToken(rem),
            .deepseek => matchDeepSeekPreToken(rem),
        };
        std.debug.assert(len > 0 and len <= rem.len);
        const chunk = rem[0..len];
        self.pos += len;
        return chunk;
    }
};

fn scanDeepSeekLetters(rem: []const u8, start: usize) usize {
    var end = start;
    while (end < rem.len) {
        if (rem[end] < 0x80) {
            const adv = neonScanAsciiAlpha(rem[end..]);
            if (adv == 0) break;
            end += adv;
            continue;
        }
        const c = decodeNextCp(rem[end..]);
        if (!isLetterOrMark(c.cp) or isDeepSeekCjk(c.cp)) break;
        end += c.len;
    }
    return end;
}

fn scanQwenLetters(rem: []const u8, start: usize) usize {
    var end = start;
    while (end < rem.len) {
        const b = rem[end];
        if (b < 0x80) {
            const adv = neonScanAsciiAlpha(rem[end..]);
            if (adv == 0) break;
            end += adv;
            continue;
        }
        if (b >= 0xE4 and b <= 0xE9) {
            const adv = neonScanUtf8CjkFast(rem[end..], false);
            if (adv > 0) {
                end += adv;
                continue;
            }
        }
        const c = decodeNextCp(rem[end..]);
        if (!isLetterOrMark(c.cp)) break;
        end += c.len;
    }
    return end;
}

fn matchDeepSeekPreToken(rem: []const u8) usize {
    const c0 = decodeNextCp(rem);

    // Stage 1: Isolated `\p{N}{1,3}`
    if (isNumber(c0.cp)) {
        var end = c0.len;
        var count: usize = 1;
        while (count < 3 and end < rem.len) {
            const c = decodeNextCp(rem[end..]);
            if (!isNumber(c.cp)) break;
            end += c.len;
            count += 1;
        }
        return end;
    }

    // Stage 2: Isolated `[\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+`
    if (isDeepSeekCjk(c0.cp)) {
        var end = c0.len;
        while (end < rem.len) {
            const b = rem[end];
            if (b >= 0xE3 and b <= 0xE9) {
                const adv = neonScanUtf8CjkFast(rem[end..], true);
                if (adv > 0) {
                    end += adv;
                    continue;
                }
            }
            const c = decodeNextCp(rem[end..]);
            if (!isDeepSeekCjk(c.cp)) break;
            end += c.len;
        }
        return end;
    }

    // Stage 3 runs strictly over the non-number, non-CJK prefix without scanning
    // past the current token (O(token_len) instead of O(rem.len), bit-exact).

    // Stage 3.1: `[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+`
    if (rem.len >= 2 and isAsciiPunct(rem[0]) and isAsciiAlpha(rem[1])) {
        return 2 + neonScanAsciiAlpha(rem[2..]);
    }

    // Stage 3.2: `[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+`
    if (isLetterOrMark(c0.cp)) {
        return scanDeepSeekLetters(rem, c0.len);
    }
    if (c0.cp != '\r' and c0.cp != '\n' and !isPunctOrSymbol(c0.cp) and c0.len < rem.len) {
        const c1 = decodeNextCp(rem[c0.len..]);
        if (!isDeepSeekStage1Or2(c1.cp) and isLetterOrMark(c1.cp)) {
            return scanDeepSeekLetters(rem, c0.len + c1.len);
        }
    }

    // Stage 3.3: ` ?[\p{P}\p{S}]+[\r\n]*`
    {
        var p_start: ?usize = null;
        if (isPunctOrSymbol(c0.cp)) {
            p_start = c0.len;
        } else if (c0.cp == ' ' and c0.len < rem.len) {
            const c1 = decodeNextCp(rem[c0.len..]);
            if (!isDeepSeekStage1Or2(c1.cp) and isPunctOrSymbol(c1.cp)) {
                p_start = c0.len + c1.len;
            }
        }
        if (p_start) |start| {
            var end = start;
            while (end < rem.len) {
                if (rem[end] < 0x80) {
                    const adv = neonScanAsciiPunctOrSym(rem[end..]);
                    if (adv == 0) break;
                    end += adv;
                    continue;
                }
                const c = decodeNextCp(rem[end..]);
                if (isDeepSeekStage1Or2(c.cp) or !isPunctOrSymbol(c.cp)) break;
                end += c.len;
            }
            end += neonScanCrLf(rem[end..]);
            return end;
        }
    }

    // Stage 3.4: `\s*[\r\n]+|\s+(?!\S)|\s+`
    if (isWhitespace(c0.cp)) {
        const ws = scanWhitespaceRun(rem);
        if (ws.after_last_nl) |nl_end| return nl_end;
        const at_sub_end = (ws.ws_end == rem.len) or isDeepSeekStage1Or2(decodeNextCp(rem[ws.ws_end..]).cp);
        if (at_sub_end) return ws.ws_end;
        if (ws.last_ws_start > 0) return ws.last_ws_start;
        return ws.ws_end;
    }

    return c0.len;
}

fn matchQwenPreToken(rem: []const u8) usize {
    // 1. (?i:'s|'t|'re|'ve|'m|'ll|'d)
    if (rem[0] == '\'') {
        if (rem.len >= 2) {
            switch (asciiLower(rem[1])) {
                's', 't', 'm', 'd' => return 2,
                'r' => if (rem.len >= 3 and asciiLower(rem[2]) == 'e') return 3,
                'v' => if (rem.len >= 3 and asciiLower(rem[2]) == 'e') return 3,
                'l' => if (rem.len >= 3 and asciiLower(rem[2]) == 'l') return 3,
                else => {},
            }
        }
    }

    const c0 = decodeNextCp(rem);

    // 2. [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
    if (isLetterOrMark(c0.cp)) {
        return scanQwenLetters(rem, c0.len);
    }
    if (c0.cp != '\r' and c0.cp != '\n' and !isNumber(c0.cp) and c0.len < rem.len) {
        const c1 = decodeNextCp(rem[c0.len..]);
        if (isLetterOrMark(c1.cp)) {
            return scanQwenLetters(rem, c0.len + c1.len);
        }
    }

    // 3. \p{N} (single digit/number codepoint in Qwen2/Qwen3/Qwen3.5)
    if (isNumber(c0.cp)) return c0.len;

    // 4. " ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*"
    {
        var p_start: ?usize = null;
        if (isPunctOrSymbol(c0.cp)) {
            p_start = c0.len;
        } else if (c0.cp == ' ' and c0.len < rem.len) {
            const c1 = decodeNextCp(rem[c0.len..]);
            if (isPunctOrSymbol(c1.cp)) {
                p_start = c0.len + c1.len;
            }
        }
        if (p_start) |start| {
            var end = start;
            while (end < rem.len) {
                if (rem[end] < 0x80) {
                    const adv = neonScanAsciiPunctOrSym(rem[end..]);
                    if (adv == 0) break;
                    end += adv;
                    continue;
                }
                const c = decodeNextCp(rem[end..]);
                if (!isPunctOrSymbol(c.cp)) break;
                end += c.len;
            }
            end += neonScanCrLf(rem[end..]);
            return end;
        }
    }

    // 5. \s*[\r\n]+ , 6. \s+(?!\S) , 7. \s+
    if (isWhitespace(c0.cp)) {
        const ws = scanWhitespaceRun(rem);
        // 5. \s*[\r\n]+
        if (ws.after_last_nl) |nl_end| return nl_end;
        // 6. \s+(?!\S)
        if (ws.ws_end == rem.len) return ws.ws_end;
        if (ws.last_ws_start > 0) return ws.last_ws_start;
        // 7. \s+
        return ws.ws_end;
    }

    return c0.len;
}

inline fn matchO200kContractionSuffix(rem: []const u8) usize {
    if (rem.len < 2 or rem[0] != '\'') return 0;
    switch (asciiLower(rem[1])) {
        's', 't', 'm', 'd' => return 2,
        'r' => if (rem.len >= 3 and asciiLower(rem[2]) == 'e') return 3,
        'v' => if (rem.len >= 3 and asciiLower(rem[2]) == 'e') return 3,
        'l' => if (rem.len >= 3 and asciiLower(rem[2]) == 'l') return 3,
        else => {},
    }
    return 0;
}

fn scanO200kWordAndContraction(rem: []const u8, word_start: usize, comptime use_neon: bool) usize {
    var pos = word_start;
    var seen_strict_lower = false;
    var last_ambig_end: ?usize = null;

    while (pos < rem.len) {
        if (rem[pos] < 0x80) {
            const b = rem[pos];
            if (b >= 'A' and b <= 'Z') {
                if (seen_strict_lower) break;
                pos += if (use_neon) (1 + neonScanAsciiUpper(rem[pos + 1 ..])) else 1;
                continue;
            }
            if (b >= 'a' and b <= 'z') {
                seen_strict_lower = true;
                pos += if (use_neon) (1 + neonScanAsciiLower(rem[pos + 1 ..])) else 1;
                continue;
            }
            break;
        }
        const c = decodeNextCp(rem[pos..]);
        if (o200kLetterCat(c.cp)) |cat| {
            switch (cat) {
                .upper_title => {
                    if (seen_strict_lower) break;
                    pos += c.len;
                },
                .lower => {
                    seen_strict_lower = true;
                    pos += c.len;
                },
                .other => {
                    pos += c.len;
                    if (!seen_strict_lower) last_ambig_end = pos;
                },
            }
        } else if (isO200kMark(c.cp)) {
            pos += c.len;
            if (!seen_strict_lower) last_ambig_end = pos;
        } else {
            break;
        }
    }

    if (!seen_strict_lower) {
        if (last_ambig_end) |ambig_end| {
            pos = ambig_end;
        }
    }
    pos += matchO200kContractionSuffix(rem[pos..]);
    return pos;
}

fn matchO200kPreToken(rem: []const u8) usize {
    const c0 = decodeNextCp(rem);

    // 1 & 2: [^\r\n\p{L}\p{N}]?[Lu|Lt|Lm|Lo|M]*[Ll|Lm|Lo|M]+(?i:'s|...)
    //        |[^\r\n\p{L}\p{N}]?[Lu|Lt|Lm|Lo|M]+[Ll|Lm|Lo|M]*(?i:'s|...)
    if (isO200kLetterOrMark(c0.cp)) {
        return scanO200kWordAndContraction(rem, 0, true);
    }
    if (c0.cp != '\r' and c0.cp != '\n' and !isO200kNumber(c0.cp) and c0.len < rem.len) {
        const c1 = decodeNextCp(rem[c0.len..]);
        if (isO200kLetterOrMark(c1.cp)) {
            return scanO200kWordAndContraction(rem, c0.len, true);
        }
    }

    // 3: \p{N}{1,3}
    if (isO200kNumber(c0.cp)) {
        var end = c0.len;
        var count: usize = 1;
        while (count < 3 and end < rem.len) {
            const c = decodeNextCp(rem[end..]);
            if (!isO200kNumber(c.cp)) break;
            end += c.len;
            count += 1;
        }
        return end;
    }

    // 4:  ?[^\s\p{L}\p{N}]+[\r\n/]*
    {
        var p_start: ?usize = null;
        if (!isWhitespace(c0.cp) and !isO200kLetter(c0.cp) and !isO200kNumber(c0.cp)) {
            p_start = c0.len;
        } else if (c0.cp == ' ' and c0.len < rem.len) {
            const c1 = decodeNextCp(rem[c0.len..]);
            if (!isWhitespace(c1.cp) and !isO200kLetter(c1.cp) and !isO200kNumber(c1.cp)) {
                p_start = c0.len + c1.len;
            }
        }
        if (p_start) |start| {
            var end = start;
            while (end < rem.len) {
                if (rem[end] < 0x80) {
                    const adv = neonScanAsciiPunctOrSym(rem[end..]);
                    if (adv == 0) break;
                    end += adv;
                    continue;
                }
                const c = decodeNextCp(rem[end..]);
                if (isWhitespace(c.cp) or isO200kLetter(c.cp) or isO200kNumber(c.cp)) break;
                end += c.len;
            }
            while (end < rem.len) {
                if (rem[end] == '\r' or rem[end] == '\n') {
                    end += neonScanCrLf(rem[end..]);
                } else if (rem[end] == '/') {
                    end += 1;
                } else {
                    break;
                }
            }
            return end;
        }
    }

    // 5. \s*[\r\n]+ , 6. \s+(?!\S) , 7. \s+
    if (isWhitespace(c0.cp)) {
        const ws = scanWhitespaceRun(rem);
        if (ws.after_last_nl) |nl_end| return nl_end;
        if (ws.ws_end == rem.len) return ws.ws_end;
        if (ws.last_ws_start > 0) return ws.last_ws_start;
        return ws.ws_end;
    }

    return c0.len;
}

pub const BpeNode = struct {
    id: u32,
    prev: u32,
    next: u32,
    alive: bool,
};

pub const HeapEntry = struct {
    rank: u32,
    left_idx: u32,
    right_idx: u32,
    left_id: u32,
    right_id: u32,
    merged_id: u32,
};

/// Preallocated BPE scratch workspace for zero-heap-allocation `encodeWithScratch`.
pub const EncodeScratch = struct {
    nodes: []BpeNode,
    heap: []HeapEntry,

    pub fn init(allocator: std.mem.Allocator, max_chunk_bytes: usize) error{OutOfMemory}!EncodeScratch {
        const nodes = try allocator.alloc(BpeNode, max_chunk_bytes);
        errdefer allocator.free(nodes);
        const heap = try allocator.alloc(HeapEntry, max_chunk_bytes * 3);
        return .{ .nodes = nodes, .heap = heap };
    }

    pub fn initFromSlices(nodes: []BpeNode, heap: []HeapEntry) EncodeScratch {
        return .{ .nodes = nodes, .heap = heap };
    }

    pub fn deinit(self: *EncodeScratch, allocator: std.mem.Allocator) void {
        allocator.free(self.nodes);
        allocator.free(self.heap);
        self.* = .{ .nodes = &.{}, .heap = &.{} };
    }
};

const BpeMinHeap = struct {
    buf: []HeapEntry,
    len: usize = 0,

    inline fn lessThan(a: HeapEntry, b: HeapEntry) bool {
        if (a.rank != b.rank) return a.rank < b.rank;
        return a.left_idx < b.left_idx;
    }

    fn push(self: *BpeMinHeap, entry: HeapEntry) EncodeError!void {
        if (self.len >= self.buf.len) return error.ScratchTooSmall;
        var idx = self.len;
        self.buf[idx] = entry;
        self.len += 1;
        while (idx > 0) {
            const parent = (idx - 1) / 2;
            if (!lessThan(self.buf[idx], self.buf[parent])) break;
            const tmp = self.buf[parent];
            self.buf[parent] = self.buf[idx];
            self.buf[idx] = tmp;
            idx = parent;
        }
    }

    fn pop(self: *BpeMinHeap) ?HeapEntry {
        if (self.len == 0) return null;
        const top = self.buf[0];
        self.len -= 1;
        if (self.len > 0) {
            self.buf[0] = self.buf[self.len];
            var idx: usize = 0;
            while (true) {
                const left = idx * 2 + 1;
                if (left >= self.len) break;
                const right = left + 1;
                var smallest = left;
                if (right < self.len and lessThan(self.buf[right], self.buf[left])) {
                    smallest = right;
                }
                if (!lessThan(self.buf[smallest], self.buf[idx])) break;
                const tmp = self.buf[idx];
                self.buf[idx] = self.buf[smallest];
                self.buf[smallest] = tmp;
                idx = smallest;
            }
        }
        return top;
    }
};

pub const MergeEntry = struct {
    rank: u32,
    merged_id: u32,
};

pub const AddedToken = struct {
    id: u32,
    content: []const u8,
    special: bool,

    fn longestFirst(_: void, a: AddedToken, b: AddedToken) bool {
        if (a.content.len != b.content.len) return a.content.len > b.content.len;
        return a.id < b.id;
    }
};

const TokenSpan = struct {
    offset: u32 = 0,
    len: u16 = 0,
    is_special: bool = false,
    present: bool = false,
};

const AddedTrieNode = struct {
    edge_start: u32 = 0,
    edge_len: u16 = 0,
    match_idx: u32 = NO_NODE,
};

const AddedTrieEdge = struct {
    byte: u8,
    child: u32,

    fn byByte(_: void, a: AddedTrieEdge, b: AddedTrieEdge) bool {
        return a.byte < b.byte;
    }
};

fn findTrieEdge(edges: []const AddedTrieEdge, b: u8) ?u32 {
    if (edges.len <= 8) {
        for (edges) |e| {
            if (e.byte == b) return e.child;
        }
        return null;
    }
    var lo: usize = 0;
    var hi: usize = edges.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        const eb = edges[mid].byte;
        if (b < eb) {
            hi = mid;
        } else if (b > eb) {
            lo = mid + 1;
        } else {
            return edges[mid].child;
        }
    }
    return null;
}

const AddedTrieBuilt = struct {
    root_next: [256]u32,
    nodes: []AddedTrieNode,
    edges: []AddedTrieEdge,
};

fn buildAddedTrie(
    allocator: std.mem.Allocator,
    temp: std.mem.Allocator,
    added_tokens: []const AddedToken,
) error{OutOfMemory}!AddedTrieBuilt {
    var root_next: [256]u32 = @splat(NO_NODE);
    if (added_tokens.len == 0) {
        return .{
            .root_next = root_next,
            .nodes = &.{},
            .edges = &.{},
        };
    }

    const BuilderNode = struct {
        edges: std.ArrayListUnmanaged(AddedTrieEdge) = .empty,
        match_idx: u32 = NO_NODE,
    };
    var bnodes: std.ArrayListUnmanaged(BuilderNode) = .empty;
    try bnodes.ensureTotalCapacity(temp, added_tokens.len * 4);

    var total_edges: usize = 0;
    for (added_tokens, 0..) |tok, tok_idx| {
        if (tok.content.len == 0) continue;
        const b0 = tok.content[0];
        var cur = root_next[b0];
        if (cur == NO_NODE) {
            cur = @intCast(bnodes.items.len);
            try bnodes.append(temp, .{});
            root_next[b0] = cur;
        }
        for (tok.content[1..]) |b| {
            var found: ?u32 = null;
            for (bnodes.items[cur].edges.items) |e| {
                if (e.byte == b) {
                    found = e.child;
                    break;
                }
            }
            if (found) |child| {
                cur = child;
            } else {
                const next_idx: u32 = @intCast(bnodes.items.len);
                try bnodes.append(temp, .{});
                try bnodes.items[cur].edges.append(temp, .{ .byte = b, .child = next_idx });
                total_edges += 1;
                cur = next_idx;
            }
        }
        if (bnodes.items[cur].match_idx == NO_NODE) {
            bnodes.items[cur].match_idx = @intCast(tok_idx);
        }
    }

    const out_nodes = try allocator.alloc(AddedTrieNode, bnodes.items.len);
    errdefer allocator.free(out_nodes);
    const out_edges = try allocator.alloc(AddedTrieEdge, total_edges);
    errdefer allocator.free(out_edges);

    var edge_cursor: usize = 0;
    for (bnodes.items, 0..) |*bn, idx| {
        const elen = bn.edges.items.len;
        const start = edge_cursor;
        if (elen > 0) {
            @memcpy(out_edges[start .. start + elen], bn.edges.items);
            std.mem.sort(AddedTrieEdge, out_edges[start .. start + elen], {}, AddedTrieEdge.byByte);
            edge_cursor += elen;
        }
        out_nodes[idx] = .{
            .edge_start = @intCast(start),
            .edge_len = @intCast(elen),
            .match_idx = bn.match_idx,
        };
    }

    return .{
        .root_next = root_next,
        .nodes = out_nodes,
        .edges = out_edges,
    };
}

inline fn pairKey(left: u32, right: u32) u64 {
    return (@as(u64, left) << 32) | right;
}

pub const Tokenizer = struct {
    token_bytes: []u8,
    dense_tokens: []TokenSpan,
    sparse_tokens: std.AutoHashMapUnmanaged(u32, TokenSpan),
    byte_to_id: [256]?u32,
    merges: std.AutoHashMapUnmanaged(u64, MergeEntry),
    full_piece_vocab: std.StringHashMapUnmanaged(u32) = .empty,
    added_tokens: []AddedToken,
    added_first_byte: [256]bool,
    added_root_next: [256]u32 = @splat(NO_NODE),
    added_trie_nodes: []AddedTrieNode = &.{},
    added_trie_edges: []AddedTrieEdge = &.{},
    pre_tokenizer_kind: PreTokenizerKind = .qwen,
    dialect: TokenizerDialect = .qwen,
    use_o200k_split: bool = false,
    ignore_merges: bool = false,
    token_count: usize,

    /// Parses a HuggingFace `tokenizer.json` payload at startup into compact
    /// flat lookup tables. Supports both `"a b"` string and `["a", "b"]` array
    /// merge formats, Qwen / DeepSeek / Hy4 / Inkling 2 (`o200k` + `ignore_merges`)
    /// dialects, and arbitrary JSON key ordering.
    pub fn fromHuggingFaceJson(allocator: std.mem.Allocator, json_bytes: []const u8) LoadError!Tokenizer {
        var temp_arena = std.heap.ArenaAllocator.init(allocator);
        defer temp_arena.deinit();
        const temp = temp_arena.allocator();

        var scanner = std.json.Scanner.initCompleteInput(temp, json_bytes);
        defer scanner.deinit();

        const first_tok = nextToken(&scanner, temp) catch |err| return err;
        if (first_tok != .object_begin) return error.InvalidJson;

        var saw_vocab = false;
        var pre_tok_info: PreTokenizerScanResult = .{};
        var ignore_merges = false;
        var temp_vocab: std.StringHashMapUnmanaged(u32) = .empty;
        var vocab_list: std.ArrayList(RawVocabEntry) = .empty;
        var raw_merges: std.ArrayList(RawMergePair) = .empty;
        var raw_added: std.ArrayList(RawAddedEntry) = .empty;

        while (true) {
            const key_tok = try nextToken(&scanner, temp);
            const key = switch (key_tok) {
                .object_end => break,
                .string => |s| s,
                .allocated_string => |s| s,
                else => return error.InvalidJson,
            };

            if (std.mem.eql(u8, key, "model")) {
                try parseModelSection(&scanner, temp, &saw_vocab, &ignore_merges, &temp_vocab, &vocab_list, &raw_merges);
            } else if (std.mem.eql(u8, key, "added_tokens")) {
                try parseAddedTokensSection(&scanner, temp, &raw_added);
            } else if (std.mem.eql(u8, key, "pre_tokenizer")) {
                pre_tok_info = try parsePreTokenizerSection(&scanner, temp);
            } else {
                try skipJsonValue(&scanner);
            }
        }

        const end_tok = try nextToken(&scanner, temp);
        if (end_tok != .end_of_document) return error.InvalidJson;
        if (!saw_vocab) return error.MissingVocab;

        // Compute dense table bound across both vocab and added_tokens.
        var max_dense_id: usize = 0;
        var has_dense = false;
        for (vocab_list.items) |v| {
            if (v.id < DENSE_LIMIT) {
                has_dense = true;
                max_dense_id = @max(max_dense_id, v.id);
            }
        }
        for (raw_added.items) |a| {
            if (a.id < DENSE_LIMIT) {
                has_dense = true;
                max_dense_id = @max(max_dense_id, a.id);
            }
        }

        const dense_len: usize = if (has_dense) max_dense_id + 1 else 0;
        const dense_tokens = try allocator.alloc(TokenSpan, dense_len);
        errdefer allocator.free(dense_tokens);
        @memset(dense_tokens, .{});

        var sparse_tokens: std.AutoHashMapUnmanaged(u32, TokenSpan) = .empty;
        errdefer sparse_tokens.deinit(allocator);

        var bytes_buf: std.ArrayList(u8) = .empty;
        errdefer bytes_buf.deinit(allocator);

        const VocabPending = struct {
            id: u32,
            offset: u32,
            len: u16,
        };
        var pending_vocab: std.ArrayList(VocabPending) = .empty;
        if (ignore_merges) {
            try pending_vocab.ensureTotalCapacity(temp, vocab_list.items.len);
        }

        var byte_to_id: [256]?u32 = @splat(null);
        var decode_scratch: std.ArrayList(u8) = .empty;
        var token_count: usize = 0;

        for (vocab_list.items) |v| {
            try decode_scratch.resize(temp, v.token.len);
            const raw_bytes: []const u8 = decodeByteLevelString(v.token, decode_scratch.items) catch |err| switch (err) {
                error.InvalidByteLevelCodepoint, error.InvalidUtf8 => v.token,
                error.BufferTooSmall => unreachable,
            };
            if (raw_bytes.len > std.math.maxInt(u16)) return error.TokenTooLong;
            if (bytes_buf.items.len + raw_bytes.len > std.math.maxInt(u32)) return error.TokenTooLong;

            const offset: u32 = @intCast(bytes_buf.items.len);
            try bytes_buf.appendSlice(allocator, raw_bytes);
            if (raw_bytes.len == 1) {
                byte_to_id[raw_bytes[0]] = v.id;
            }
            const span: TokenSpan = .{
                .offset = offset,
                .len = @intCast(raw_bytes.len),
                .is_special = false,
                .present = true,
            };
            const existed = try putTokenSpan(allocator, dense_tokens, &sparse_tokens, v.id, span);
            if (!existed) token_count += 1;
            if (ignore_merges) {
                pending_vocab.appendAssumeCapacity(.{
                    .id = v.id,
                    .offset = offset,
                    .len = @intCast(raw_bytes.len),
                });
            }
        }

        // Record added tokens (pointing into `bytes_buf` offsets, resolved to slices once finalized).
        const AddedPending = struct {
            id: u32,
            offset: u32,
            len: u16,
            special: bool,
        };
        var pending_added: std.ArrayList(AddedPending) = .empty;
        try pending_added.ensureTotalCapacity(temp, raw_added.items.len);

        for (raw_added.items) |a| {
            if (a.content.len == 0) return error.InvalidAddedToken;
            if (a.content.len > std.math.maxInt(u16)) return error.TokenTooLong;
            if (bytes_buf.items.len + a.content.len > std.math.maxInt(u32)) return error.TokenTooLong;

            const offset: u32 = @intCast(bytes_buf.items.len);
            try bytes_buf.appendSlice(allocator, a.content);
            const span: TokenSpan = .{
                .offset = offset,
                .len = @intCast(a.content.len),
                .is_special = a.special,
                .present = true,
            };
            const existed = try putTokenSpan(allocator, dense_tokens, &sparse_tokens, a.id, span);
            if (!existed) token_count += 1;
            pending_added.appendAssumeCapacity(.{
                .id = a.id,
                .offset = offset,
                .len = @intCast(a.content.len),
                .special = a.special,
            });
        }

        const token_bytes = try bytes_buf.toOwnedSlice(allocator);
        errdefer allocator.free(token_bytes);

        const added_tokens = try allocator.alloc(AddedToken, pending_added.items.len);
        errdefer allocator.free(added_tokens);
        var added_first_byte: [256]bool = @splat(false);
        var has_inkling_markers = false;
        var has_hy4_markers = false;
        for (pending_added.items, 0..) |p, i| {
            const slice = token_bytes[p.offset .. p.offset + p.len];
            added_tokens[i] = .{
                .id = p.id,
                .content = slice,
                .special = p.special,
            };
            added_first_byte[slice[0]] = true;
            if (std.mem.eql(u8, slice, "<|content_model_end_sampling|>") or
                std.mem.eql(u8, slice, "<|message_user|>") or
                std.mem.eql(u8, slice, "<|content_thinking|>"))
            {
                has_inkling_markers = true;
            } else if (std.mem.startsWith(u8, slice, "<\xef\xbd\x9chy_") or
                std.mem.eql(u8, slice, "<think:opensource>"))
            {
                has_hy4_markers = true;
            }
        }
        std.mem.sort(AddedToken, added_tokens, {}, AddedToken.longestFirst);

        const trie = try buildAddedTrie(allocator, temp, added_tokens);
        errdefer {
            allocator.free(trie.nodes);
            allocator.free(trie.edges);
        }

        var full_piece_vocab: std.StringHashMapUnmanaged(u32) = .empty;
        errdefer full_piece_vocab.deinit(allocator);
        if (ignore_merges and pending_vocab.items.len > 0) {
            try full_piece_vocab.ensureTotalCapacity(allocator, @intCast(pending_vocab.items.len));
            for (pending_vocab.items) |pv| {
                const slice = token_bytes[pv.offset .. pv.offset + pv.len];
                const gop = full_piece_vocab.getOrPutAssumeCapacity(slice);
                if (!gop.found_existing) {
                    gop.value_ptr.* = pv.id;
                }
            }
        }

        // Build BPE merge priority hashmap `(left_id, right_id) -> { rank, merged_id }`.
        var merges: std.AutoHashMapUnmanaged(u64, MergeEntry) = .empty;
        errdefer merges.deinit(allocator);
        if (raw_merges.items.len > 0) {
            try merges.ensureTotalCapacity(allocator, @intCast(raw_merges.items.len));
        }
        var concat_buf: std.ArrayList(u8) = .empty;
        for (raw_merges.items, 0..) |m, rank| {
            const left_id = temp_vocab.get(m.left) orelse return error.InvalidMerge;
            const right_id = temp_vocab.get(m.right) orelse return error.InvalidMerge;
            concat_buf.clearRetainingCapacity();
            try concat_buf.ensureTotalCapacity(temp, m.left.len + m.right.len);
            concat_buf.appendSliceAssumeCapacity(m.left);
            concat_buf.appendSliceAssumeCapacity(m.right);
            const merged_id = temp_vocab.get(concat_buf.items) orelse return error.InvalidMerge;

            const gop = merges.getOrPutAssumeCapacity(pairKey(left_id, right_id));
            if (!gop.found_existing) {
                gop.value_ptr.* = .{ .rank = @intCast(rank), .merged_id = merged_id };
            }
        }

        const dialect: TokenizerDialect = if (pre_tok_info.use_o200k or has_inkling_markers)
            .inkling2
        else if (has_hy4_markers)
            .hy4
        else if (pre_tok_info.kind == .deepseek)
            .deepseek
        else
            .qwen;

        return .{
            .token_bytes = token_bytes,
            .dense_tokens = dense_tokens,
            .sparse_tokens = sparse_tokens,
            .byte_to_id = byte_to_id,
            .merges = merges,
            .full_piece_vocab = full_piece_vocab,
            .added_tokens = added_tokens,
            .added_first_byte = added_first_byte,
            .added_root_next = trie.root_next,
            .added_trie_nodes = trie.nodes,
            .added_trie_edges = trie.edges,
            .pre_tokenizer_kind = pre_tok_info.kind,
            .dialect = dialect,
            .use_o200k_split = pre_tok_info.use_o200k,
            .ignore_merges = ignore_merges,
            .token_count = token_count,
        };
    }

    pub fn deinit(self: *Tokenizer, allocator: std.mem.Allocator) void {
        allocator.free(self.token_bytes);
        allocator.free(self.dense_tokens);
        self.sparse_tokens.deinit(allocator);
        self.merges.deinit(allocator);
        self.full_piece_vocab.deinit(allocator);
        allocator.free(self.added_tokens);
        if (self.added_trie_nodes.len > 0) allocator.free(self.added_trie_nodes);
        if (self.added_trie_edges.len > 0) allocator.free(self.added_trie_edges);
        self.* = undefined;
    }

    pub fn vocabSize(self: *const Tokenizer) usize {
        return self.token_count;
    }

    /// Looks up the token ID of an `added_tokens` entry (e.g. `<|im_start|>`).
    pub fn addedTokenId(self: *const Tokenizer, content: []const u8) ?u32 {
        if (self.added_tokens.len == 0 or content.len == 0) return null;
        if (self.added_trie_nodes.len > 0) {
            var node_idx = self.added_root_next[content[0]];
            if (node_idx == NO_NODE) return null;
            for (content[1..]) |b| {
                const node = self.added_trie_nodes[node_idx];
                const edges = self.added_trie_edges[node.edge_start .. node.edge_start + node.edge_len];
                node_idx = findTrieEdge(edges, b) orelse return null;
            }
            const match_idx = self.added_trie_nodes[node_idx].match_idx;
            return if (match_idx != NO_NODE) self.added_tokens[match_idx].id else null;
        }
        for (self.added_tokens) |tok| {
            if (std.mem.eql(u8, tok.content, content)) return tok.id;
        }
        return null;
    }

    /// Resolves canonical EOS token ID for this tokenizer's dialect.
    pub fn eosTokenId(self: *const Tokenizer) ?u32 {
        return switch (self.dialect) {
            .inkling2 => self.addedTokenId("<|content_model_end_sampling|>") orelse self.addedTokenId("<|endoftext|>"),
            .hy4 => self.addedTokenId("<\xef\xbd\x9chy_end:opensource\xef\xbd\x9c>"),
            .deepseek => self.addedTokenId("<\xef\xbd\x9cend\xe2\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c>") orelse self.addedTokenId("<|EOT|>"),
            .qwen => self.addedTokenId("<|im_end|>") orelse self.addedTokenId("<|endoftext|>"),
        };
    }

    /// Resolves canonical BOS token ID for this tokenizer's dialect, if present.
    pub fn bosTokenId(self: *const Tokenizer) ?u32 {
        return switch (self.dialect) {
            .inkling2 => self.addedTokenId("<|begin_of_text|>"),
            .hy4 => self.addedTokenId("<\xef\xbd\x9chy_start:opensource\xef\xbd\x9c>"),
            .deepseek => self.addedTokenId("<\xef\xbd\x9cbegin\xe2\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c>"),
            .qwen => null,
        };
    }

    pub fn isSpecialToken(self: *const Tokenizer, id: u32) bool {
        const span = self.lookupSpan(id) orelse return false;
        return span.is_special;
    }

    /// Zero-copy lookup of the raw bytes represented by `id`.
    pub fn tokenBytes(self: *const Tokenizer, id: u32) DecodeError![]const u8 {
        const span = self.lookupSpan(id) orelse return error.UnknownTokenId;
        return self.token_bytes[span.offset .. span.offset + span.len];
    }

    /// Encodes `text` into caller-provided `out_ids`. Uses stack scratch for chunks
    /// up to `STACK_CHUNK_CAP` bytes and falls back to `allocator` only for longer chunks.
    pub fn encode(
        self: *const Tokenizer,
        allocator: std.mem.Allocator,
        text: []const u8,
        out_ids: []u32,
    ) EncodeError![]u32 {
        var writer: SliceIdWriter = .{ .buf = out_ids };
        try self.encodeImpl(&writer, allocator, null, text);
        return writer.written();
    }

    /// Zero-heap-allocation encoding using caller-provided `EncodeScratch` and `out_ids`.
    pub fn encodeWithScratch(
        self: *const Tokenizer,
        scratch: *EncodeScratch,
        text: []const u8,
        out_ids: []u32,
    ) EncodeError![]u32 {
        var writer: SliceIdWriter = .{ .buf = out_ids };
        try self.encodeImpl(&writer, null, scratch, text);
        return writer.written();
    }

    /// Allocates and returns the encoded token IDs for `text`.
    pub fn encodeAlloc(
        self: *const Tokenizer,
        allocator: std.mem.Allocator,
        text: []const u8,
    ) EncodeError![]u32 {
        var list: std.ArrayList(u32) = .empty;
        errdefer list.deinit(allocator);
        var writer: ListIdWriter = .{ .allocator = allocator, .list = &list };
        try self.encodeImpl(&writer, allocator, null, text);
        return list.toOwnedSlice(allocator);
    }

    /// Zero-heap-allocation lossless decode of `ids` into `out_bytes`
    /// (preserves special tokens so `decode(encode(text)) == text`).
    pub fn decode(self: *const Tokenizer, ids: []const u32, out_bytes: []u8) DecodeError![]u8 {
        return self.decodeImpl(ids, out_bytes, false);
    }

    /// Zero-heap-allocation decode omitting tokens marked `special: true`.
    pub fn decodeSkippingSpecial(self: *const Tokenizer, ids: []const u32, out_bytes: []u8) DecodeError![]u8 {
        return self.decodeImpl(ids, out_bytes, true);
    }

    /// Allocates and returns the lossless decoded byte string for `ids`.
    pub fn decodeAlloc(
        self: *const Tokenizer,
        allocator: std.mem.Allocator,
        ids: []const u32,
    ) (DecodeError || error{OutOfMemory})![]u8 {
        var total: usize = 0;
        for (ids) |id| {
            const bytes = try self.tokenBytes(id);
            total += bytes.len;
        }
        const out = try allocator.alloc(u8, total);
        errdefer allocator.free(out);
        const written = try self.decode(ids, out);
        std.debug.assert(written.len == total);
        return out;
    }

    fn lookupSpan(self: *const Tokenizer, id: u32) ?TokenSpan {
        if (id < self.dense_tokens.len) {
            const s = self.dense_tokens[id];
            return if (s.present) s else null;
        }
        return self.sparse_tokens.get(id);
    }

    fn decodeImpl(self: *const Tokenizer, ids: []const u32, out_bytes: []u8, skip_special: bool) DecodeError![]u8 {
        var pos: usize = 0;
        for (ids) |id| {
            const span = self.lookupSpan(id) orelse return error.UnknownTokenId;
            if (skip_special and span.is_special) continue;
            const piece = self.token_bytes[span.offset .. span.offset + span.len];
            if (pos + piece.len > out_bytes.len) return error.BufferTooSmall;
            @memcpy(out_bytes[pos .. pos + piece.len], piece);
            pos += piece.len;
        }
        return out_bytes[0..pos];
    }

    fn matchAddedTokenAt(self: *const Tokenizer, text: []const u8, pos: usize) ?AddedToken {
        if (self.added_tokens.len == 0) return null;
        if (!self.added_first_byte[text[pos]]) return null;
        if (self.added_trie_nodes.len > 0) {
            var node_idx = self.added_root_next[text[pos]];
            if (node_idx == NO_NODE) return null;
            var best: ?AddedToken = null;
            var p = pos + 1;
            while (true) {
                const node = self.added_trie_nodes[node_idx];
                if (node.match_idx != NO_NODE) {
                    best = self.added_tokens[node.match_idx];
                }
                if (p >= text.len or node.edge_len == 0) break;
                const edges = self.added_trie_edges[node.edge_start .. node.edge_start + node.edge_len];
                node_idx = findTrieEdge(edges, text[p]) orelse break;
                p += 1;
            }
            return best;
        }
        const tail = text[pos..];
        for (self.added_tokens) |tok| {
            if (std.mem.startsWith(u8, tail, tok.content)) return tok;
        }
        return null;
    }

    fn encodeImpl(
        self: *const Tokenizer,
        writer: anytype,
        maybe_alloc: ?std.mem.Allocator,
        maybe_scratch: ?*EncodeScratch,
        text: []const u8,
    ) EncodeError!void {
        if (self.added_tokens.len == 0) {
            try self.encodePlainSegment(writer, maybe_alloc, maybe_scratch, text);
            return;
        }

        var plain_start: usize = 0;
        var pos: usize = 0;
        while (pos < text.len) {
            if (self.matchAddedTokenAt(text, pos)) |tok| {
                if (pos > plain_start) {
                    try self.encodePlainSegment(writer, maybe_alloc, maybe_scratch, text[plain_start..pos]);
                }
                try writer.push(tok.id);
                pos += tok.content.len;
                plain_start = pos;
            } else {
                pos += 1;
            }
        }
        if (plain_start < text.len) {
            try self.encodePlainSegment(writer, maybe_alloc, maybe_scratch, text[plain_start..]);
        }
    }

    fn encodePlainSegment(
        self: *const Tokenizer,
        writer: anytype,
        maybe_alloc: ?std.mem.Allocator,
        maybe_scratch: ?*EncodeScratch,
        segment: []const u8,
    ) EncodeError!void {
        var it = if (self.use_o200k_split)
            PreTokenIterator.initO200k(segment)
        else
            PreTokenIterator.initWithKind(segment, self.pre_tokenizer_kind);
        while (it.next()) |chunk| {
            try self.encodeChunk(writer, maybe_alloc, maybe_scratch, chunk);
        }
    }

    fn encodeChunk(
        self: *const Tokenizer,
        writer: anytype,
        maybe_alloc: ?std.mem.Allocator,
        maybe_scratch: ?*EncodeScratch,
        chunk: []const u8,
    ) EncodeError!void {
        if (chunk.len == 0) return;
        if (chunk.len == 1) {
            const id = self.byte_to_id[chunk[0]] orelse return error.UnknownByte;
            try writer.push(id);
            return;
        }
        if (self.ignore_merges) {
            if (self.full_piece_vocab.get(chunk)) |id| {
                try writer.push(id);
                return;
            }
        }

        if (maybe_scratch) |scratch| {
            if (chunk.len > scratch.nodes.len or chunk.len * 3 > scratch.heap.len) {
                return error.ScratchTooSmall;
            }
            try self.bpeMergeInto(writer, chunk, scratch.nodes[0..chunk.len], scratch.heap);
            return;
        }

        if (chunk.len <= STACK_CHUNK_CAP) {
            var stack_nodes: [STACK_CHUNK_CAP]BpeNode = undefined;
            var stack_heap: [STACK_CHUNK_CAP * 3]HeapEntry = undefined;
            try self.bpeMergeInto(writer, chunk, stack_nodes[0..chunk.len], &stack_heap);
            return;
        }

        const alloc = maybe_alloc orelse return error.ScratchTooSmall;
        const nodes = try alloc.alloc(BpeNode, chunk.len);
        defer alloc.free(nodes);
        const heap = try alloc.alloc(HeapEntry, chunk.len * 3);
        defer alloc.free(heap);
        try self.bpeMergeInto(writer, chunk, nodes, heap);
    }

    fn bpeMergeInto(
        self: *const Tokenizer,
        writer: anytype,
        chunk: []const u8,
        nodes: []BpeNode,
        heap_buf: []HeapEntry,
    ) EncodeError!void {
        const n: u32 = @intCast(chunk.len);
        for (chunk, 0..) |b, i| {
            const id = self.byte_to_id[b] orelse return error.UnknownByte;
            nodes[i] = .{
                .id = id,
                .prev = if (i == 0) NO_NODE else @intCast(i - 1),
                .next = if (i + 1 == chunk.len) NO_NODE else @intCast(i + 1),
                .alive = true,
            };
        }

        var heap: BpeMinHeap = .{ .buf = heap_buf };
        for (0..n - 1) |i| {
            const left_id = nodes[i].id;
            const right_id = nodes[i + 1].id;
            if (self.merges.get(pairKey(left_id, right_id))) |m| {
                try heap.push(.{
                    .rank = m.rank,
                    .left_idx = @intCast(i),
                    .right_idx = @intCast(i + 1),
                    .left_id = left_id,
                    .right_id = right_id,
                    .merged_id = m.merged_id,
                });
            }
        }

        while (heap.pop()) |top| {
            const left = &nodes[top.left_idx];
            const right = &nodes[top.right_idx];
            if (!left.alive or !right.alive) continue;
            if (left.next != top.right_idx) continue;
            if (left.id != top.left_id or right.id != top.right_id) continue;

            // Merge right into left.
            left.id = top.merged_id;
            left.next = right.next;
            right.alive = false;
            if (right.next != NO_NODE) {
                nodes[right.next].prev = top.left_idx;
            }

            // Enqueue newly formed left-neighbor and right-neighbor pairs.
            if (left.prev != NO_NODE) {
                const p_idx = left.prev;
                const p_id = nodes[p_idx].id;
                if (self.merges.get(pairKey(p_id, left.id))) |m| {
                    try heap.push(.{
                        .rank = m.rank,
                        .left_idx = p_idx,
                        .right_idx = top.left_idx,
                        .left_id = p_id,
                        .right_id = left.id,
                        .merged_id = m.merged_id,
                    });
                }
            }
            if (left.next != NO_NODE) {
                const n_idx = left.next;
                const n_id = nodes[n_idx].id;
                if (self.merges.get(pairKey(left.id, n_id))) |m| {
                    try heap.push(.{
                        .rank = m.rank,
                        .left_idx = top.left_idx,
                        .right_idx = n_idx,
                        .left_id = left.id,
                        .right_id = n_id,
                        .merged_id = m.merged_id,
                    });
                }
            }
        }

        var cur: u32 = 0;
        while (cur != NO_NODE) {
            try writer.push(nodes[cur].id);
            cur = nodes[cur].next;
        }
    }
};

const SliceIdWriter = struct {
    buf: []u32,
    len: usize = 0,

    fn push(self: *SliceIdWriter, id: u32) EncodeError!void {
        if (self.len >= self.buf.len) return error.BufferTooSmall;
        self.buf[self.len] = id;
        self.len += 1;
    }

    fn written(self: *const SliceIdWriter) []u32 {
        return self.buf[0..self.len];
    }
};

const ListIdWriter = struct {
    allocator: std.mem.Allocator,
    list: *std.ArrayList(u32),

    fn push(self: *ListIdWriter, id: u32) EncodeError!void {
        try self.list.append(self.allocator, id);
    }
};

const RawVocabEntry = struct {
    id: u32,
    token: []const u8,
};

const RawMergePair = struct {
    left: []const u8,
    right: []const u8,
};

const RawAddedEntry = struct {
    id: u32,
    content: []const u8,
    special: bool,
};

fn putTokenSpan(
    allocator: std.mem.Allocator,
    dense_tokens: []TokenSpan,
    sparse_tokens: *std.AutoHashMapUnmanaged(u32, TokenSpan),
    id: u32,
    span: TokenSpan,
) error{OutOfMemory}!bool {
    if (id < dense_tokens.len) {
        const existed = dense_tokens[id].present;
        dense_tokens[id] = span;
        return existed;
    }
    const gop = try sparse_tokens.getOrPut(allocator, id);
    const existed = gop.found_existing;
    gop.value_ptr.* = span;
    return existed;
}

fn mapJsonErr(err: anyerror) LoadError {
    return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.InvalidJson,
    };
}

fn nextToken(scanner: *std.json.Scanner, temp: std.mem.Allocator) LoadError!std.json.Token {
    return scanner.nextAlloc(temp, .alloc_if_needed) catch |err| mapJsonErr(err);
}

fn skipJsonValue(scanner: *std.json.Scanner) LoadError!void {
    scanner.skipValue() catch |err| return mapJsonErr(err);
}

fn parseU32Token(tok: std.json.Token) LoadError!u32 {
    const raw = switch (tok) {
        .number => |s| s,
        .allocated_number => |s| s,
        else => return error.InvalidJson,
    };
    return std.fmt.parseInt(u32, raw, 10) catch return error.InvalidJson;
}

const PreTokenizerScanResult = struct {
    kind: PreTokenizerKind = .qwen,
    use_o200k: bool = false,
};

fn parsePreTokenizerSection(
    scanner: *std.json.Scanner,
    temp: std.mem.Allocator,
) LoadError!PreTokenizerScanResult {
    var out: PreTokenizerScanResult = .{};
    var depth: usize = 0;
    var expect_regex_val = false;
    while (true) {
        const tok = try nextToken(scanner, temp);
        switch (tok) {
            .object_begin, .array_begin => {
                depth += 1;
                expect_regex_val = false;
            },
            .object_end, .array_end => {
                if (depth == 0) return error.InvalidJson;
                depth -= 1;
                expect_regex_val = false;
                if (depth == 0) break;
            },
            .string, .allocated_string => |s| {
                if (expect_regex_val) {
                    if (std.mem.indexOf(u8, s, "\\p{Lu}") != null or
                        std.mem.indexOf(u8, s, "\\p{Ll}") != null or
                        std.mem.indexOf(u8, s, "[\\r\\n/]*") != null)
                    {
                        out.use_o200k = true;
                        out.kind = .qwen;
                    } else if (!out.use_o200k and (std.mem.indexOf(u8, s, "\\p{N}{1,3}") != null or
                        std.mem.indexOf(u8, s, "\xe4\xb8\x80-\xe9\xbe\xa5") != null))
                    {
                        out.kind = .deepseek;
                    }
                    expect_regex_val = false;
                } else if (std.mem.eql(u8, s, "Regex")) {
                    expect_regex_val = true;
                }
                if (depth == 0) break;
            },
            .null, .true, .false, .number, .allocated_number => {
                expect_regex_val = false;
                if (depth == 0) break;
            },
            else => return error.InvalidJson,
        }
    }
    return out;
}

fn parseModelSection(
    scanner: *std.json.Scanner,
    temp: std.mem.Allocator,
    saw_vocab: *bool,
    ignore_merges: *bool,
    temp_vocab: *std.StringHashMapUnmanaged(u32),
    vocab_list: *std.ArrayList(RawVocabEntry),
    raw_merges: *std.ArrayList(RawMergePair),
) LoadError!void {
    const start_tok = try nextToken(scanner, temp);
    if (start_tok != .object_begin) return error.InvalidJson;

    while (true) {
        const key_tok = try nextToken(scanner, temp);
        const key = switch (key_tok) {
            .object_end => break,
            .string => |s| s,
            .allocated_string => |s| s,
            else => return error.InvalidJson,
        };

        if (std.mem.eql(u8, key, "vocab")) {
            saw_vocab.* = true;
            try parseVocabObject(scanner, temp, temp_vocab, vocab_list);
        } else if (std.mem.eql(u8, key, "merges")) {
            try parseMergesArray(scanner, temp, raw_merges);
        } else if (std.mem.eql(u8, key, "ignore_merges")) {
            const val_tok = try nextToken(scanner, temp);
            switch (val_tok) {
                .true => ignore_merges.* = true,
                .false, .null => ignore_merges.* = false,
                else => return error.InvalidJson,
            }
        } else {
            try skipJsonValue(scanner);
        }
    }
}

fn parseVocabObject(
    scanner: *std.json.Scanner,
    temp: std.mem.Allocator,
    temp_vocab: *std.StringHashMapUnmanaged(u32),
    vocab_list: *std.ArrayList(RawVocabEntry),
) LoadError!void {
    const start_tok = try nextToken(scanner, temp);
    if (start_tok != .object_begin) return error.InvalidVocabEntry;

    while (true) {
        const key_tok = try nextToken(scanner, temp);
        const tok_str = switch (key_tok) {
            .object_end => break,
            .string => |s| s,
            .allocated_string => |s| s,
            else => return error.InvalidVocabEntry,
        };
        const id_tok = try nextToken(scanner, temp);
        const id = parseU32Token(id_tok) catch return error.InvalidVocabEntry;

        try temp_vocab.put(temp, tok_str, id);
        try vocab_list.append(temp, .{ .id = id, .token = tok_str });
    }
}

fn parseMergesArray(
    scanner: *std.json.Scanner,
    temp: std.mem.Allocator,
    raw_merges: *std.ArrayList(RawMergePair),
) LoadError!void {
    const start_tok = try nextToken(scanner, temp);
    if (start_tok != .array_begin) return error.InvalidMerge;

    while (true) {
        const item_tok = try nextToken(scanner, temp);
        switch (item_tok) {
            .array_end => break,
            .string, .allocated_string => |s| {
                const sep = std.mem.indexOfScalar(u8, s, ' ') orelse return error.InvalidMerge;
                const left = s[0..sep];
                const right = s[sep + 1 ..];
                if (left.len == 0 or right.len == 0) return error.InvalidMerge;
                if (std.mem.indexOfScalar(u8, right, ' ') != null) return error.InvalidMerge;
                try raw_merges.append(temp, .{ .left = left, .right = right });
            },
            .array_begin => {
                const left_tok = try nextToken(scanner, temp);
                const left = switch (left_tok) {
                    .string => |s| s,
                    .allocated_string => |s| s,
                    else => return error.InvalidMerge,
                };
                const right_tok = try nextToken(scanner, temp);
                const right = switch (right_tok) {
                    .string => |s| s,
                    .allocated_string => |s| s,
                    else => return error.InvalidMerge,
                };
                const end_pair = try nextToken(scanner, temp);
                if (end_pair != .array_end) return error.InvalidMerge;
                if (left.len == 0 or right.len == 0) return error.InvalidMerge;
                try raw_merges.append(temp, .{ .left = left, .right = right });
            },
            else => return error.InvalidMerge,
        }
    }
}

fn parseAddedTokensSection(
    scanner: *std.json.Scanner,
    temp: std.mem.Allocator,
    raw_added: *std.ArrayList(RawAddedEntry),
) LoadError!void {
    const start_tok = try nextToken(scanner, temp);
    if (start_tok == .null) return;
    if (start_tok != .array_begin) return error.InvalidAddedToken;

    while (true) {
        const elem_tok = try nextToken(scanner, temp);
        switch (elem_tok) {
            .array_end => break,
            .object_begin => {
                var maybe_id: ?u32 = null;
                var maybe_content: ?[]const u8 = null;
                var special: bool = false;

                while (true) {
                    const field_tok = try nextToken(scanner, temp);
                    const field = switch (field_tok) {
                        .object_end => break,
                        .string => |s| s,
                        .allocated_string => |s| s,
                        else => return error.InvalidAddedToken,
                    };
                    if (std.mem.eql(u8, field, "id")) {
                        const v = try nextToken(scanner, temp);
                        maybe_id = parseU32Token(v) catch return error.InvalidAddedToken;
                    } else if (std.mem.eql(u8, field, "content")) {
                        const v = try nextToken(scanner, temp);
                        maybe_content = switch (v) {
                            .string => |s| s,
                            .allocated_string => |s| s,
                            else => return error.InvalidAddedToken,
                        };
                    } else if (std.mem.eql(u8, field, "special")) {
                        const v = try nextToken(scanner, temp);
                        special = switch (v) {
                            .true => true,
                            .false => false,
                            else => return error.InvalidAddedToken,
                        };
                    } else {
                        try skipJsonValue(scanner);
                    }
                }

                const id = maybe_id orelse return error.InvalidAddedToken;
                const content = maybe_content orelse return error.InvalidAddedToken;
                try raw_added.append(temp, .{ .id = id, .content = content, .special = special });
            },
            else => return error.InvalidAddedToken,
        }
    }
}

fn buildSyntheticTokenizerJson(allocator: std.mem.Allocator, array_merge_format: bool) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);

    try out.appendSlice(allocator, "{\"model\":{\"type\":\"BPE\",\"vocab\":{");
    var first = true;
    for (0..256) |b| {
        if (!first) try out.append(allocator, ',');
        first = false;
        const cp = byteToUnicode(@intCast(b));
        var utf8_buf: [4]u8 = undefined;
        const n = std.unicode.utf8Encode(cp, &utf8_buf) catch unreachable;
        try appendJsonString(allocator, &out, utf8_buf[0..n]);
        try out.print(allocator, ":{d}", .{b});
    }

    const extra_vocab = [_]struct { bytes: []const u8, id: u32 }{
        .{ .bytes = "aa", .id = 256 },
        .{ .bytes = "he", .id = 257 },
        .{ .bytes = "ll", .id = 258 },
        .{ .bytes = "lo", .id = 259 },
        .{ .bytes = "hel", .id = 260 },
        .{ .bytes = "hello", .id = 261 },
        .{ .bytes = " hello", .id = 262 },
        .{ .bytes = "  ", .id = 263 },
        .{ .bytes = "    ", .id = 264 },
        .{ .bytes = "\xE4\xBD", .id = 265 },
        .{ .bytes = "你", .id = 266 },
    };
    var bl_buf: [64]u8 = undefined;
    for (extra_vocab) |ev| {
        try out.append(allocator, ',');
        const bl = try encodeByteLevelString(ev.bytes, &bl_buf);
        try appendJsonString(allocator, &out, bl);
        try out.print(allocator, ":{d}", .{ev.id});
    }

    try out.appendSlice(allocator, "},\"merges\":[");
    const merges = [_][2][]const u8{
        .{ "a", "a" },
        .{ "h", "e" },
        .{ "he", "l" },
        .{ "l", "o" },
        .{ "hel", "lo" },
        .{ "l", "l" },
        .{ " ", "hello" },
        .{ " ", " " },
        .{ "  ", "  " },
        .{ "\xE4", "\xBD" },
        .{ "\xE4\xBD", "\xA0" },
    };
    var bl_a: [64]u8 = undefined;
    var bl_b: [64]u8 = undefined;
    for (merges, 0..) |m, idx| {
        if (idx > 0) try out.append(allocator, ',');
        const a = try encodeByteLevelString(m[0], &bl_a);
        const b = try encodeByteLevelString(m[1], &bl_b);
        if (array_merge_format) {
            try out.append(allocator, '[');
            try appendJsonString(allocator, &out, a);
            try out.append(allocator, ',');
            try appendJsonString(allocator, &out, b);
            try out.append(allocator, ']');
        } else {
            var joined: [130]u8 = undefined;
            @memcpy(joined[0..a.len], a);
            joined[a.len] = ' ';
            @memcpy(joined[a.len + 1 .. a.len + 1 + b.len], b);
            try appendJsonString(allocator, &out, joined[0 .. a.len + 1 + b.len]);
        }
    }

    try out.appendSlice(allocator,
        \\]},"added_tokens":[
        \\{"id":1000,"content":"<|endoftext|>","special":true},
        \\{"id":1001,"content":"<|im|>","special":true},
        \\{"id":1002,"content":"<|im_start|>","special":true},
        \\{"id":1003,"content":"<|im_end|>","special":true},
        \\{"id":1004,"content":"<think>","special":false},
        \\{"id":1005,"content":"</think>","special":false}
        \\]}
    );
    return out.toOwnedSlice(allocator);
}

fn appendJsonString(allocator: std.mem.Allocator, out: *std.ArrayList(u8), s: []const u8) !void {
    try out.append(allocator, '"');
    for (s) |c| {
        switch (c) {
            '"' => try out.appendSlice(allocator, "\\\""),
            '\\' => try out.appendSlice(allocator, "\\\\"),
            '\n' => try out.appendSlice(allocator, "\\n"),
            '\r' => try out.appendSlice(allocator, "\\r"),
            '\t' => try out.appendSlice(allocator, "\\t"),
            else => try out.append(allocator, c),
        }
    }
    try out.append(allocator, '"');
}

test "byte-level GPT-2/Qwen table is a 256-byte bijection with known anchor codepoints" {
    var seen: [UNICODE_TO_BYTE_LEN]bool = @splat(false);
    for (0..256) |b| {
        const cp = byteToUnicode(@intCast(b));
        try std.testing.expect(cp >= 0x21 and cp < UNICODE_TO_BYTE_LEN);
        try std.testing.expect(!seen[cp]);
        seen[cp] = true;
        try std.testing.expectEqual(@as(?u8, @intCast(b)), unicodeToByte(cp));
    }

    try std.testing.expectEqual(@as(u21, 0x0100), byteToUnicode(0x00)); // Ā
    try std.testing.expectEqual(@as(u21, 0x010A), byteToUnicode('\n')); // Ċ
    try std.testing.expectEqual(@as(u21, 0x0120), byteToUnicode(' ')); // Ġ
    try std.testing.expectEqual(@as(u21, '!'), byteToUnicode('!'));
    try std.testing.expectEqual(@as(u21, '~'), byteToUnicode('~'));
    try std.testing.expectEqual(@as(u21, 0x0143), byteToUnicode(0xAD)); // Ń
    try std.testing.expectEqual(@as(?u8, null), unicodeToByte(' ')); // ASCII space is never in table

    var all_bytes: [256]u8 = undefined;
    for (&all_bytes, 0..) |*slot, i| slot.* = @intCast(i);
    var encoded_buf: [512]u8 = undefined;
    const encoded = try encodeByteLevelString(&all_bytes, &encoded_buf);
    var decoded_buf: [256]u8 = undefined;
    const decoded = try decodeByteLevelString(encoded, &decoded_buf);
    try std.testing.expectEqualSlices(u8, &all_bytes, decoded);
}

test "Qwen pre-tokenizer splits contractions, single digits, Chinese, and code indentation" {
    // Contractions & ASCII words
    {
        var it = PreTokenIterator.init("don't I've WE'LL");
        const expected = [_][]const u8{ "don", "'t", " I", "'ve", " WE", "'LL" };
        for (expected) |want| {
            const got = it.next() orelse return error.TestUnexpectedResult;
            try std.testing.expectEqualStrings(want, got);
        }
        try std.testing.expectEqual(@as(?[]const u8, null), it.next());
    }

    // Code indentation + newlines + single digits
    {
        var it = PreTokenIterator.init("    def f(x):\n        return x + 42\n");
        const expected = [_][]const u8{
            "   ",     " def", " f", "(x", "):\n", "       ",
            " return", " x",   " +", " ",  "4",    "2",
            "\n",
        };
        for (expected) |want| {
            const got = it.next() orelse return error.TestUnexpectedResult;
            try std.testing.expectEqualStrings(want, got);
        }
        try std.testing.expectEqual(@as(?[]const u8, null), it.next());
    }

    // Chinese UTF-8 + fullwidth punctuation + digits
    // Notice `[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+` binds a leading punctuation mark
    // directly before letters ("，世界"), while space after punctuation separates it ("，", " 世界").
    {
        var it = PreTokenIterator.init("你好，世界！2026");
        const expected = [_][]const u8{ "你好", "，世界", "！", "2", "0", "2", "6" };
        for (expected) |want| {
            const got = it.next() orelse return error.TestUnexpectedResult;
            try std.testing.expectEqualStrings(want, got);
        }
        try std.testing.expectEqual(@as(?[]const u8, null), it.next());
    }
    {
        var it = PreTokenIterator.init("你好， 世界！2026");
        const expected = [_][]const u8{ "你好", "，", " 世界", "！", "2", "0", "2", "6" };
        for (expected) |want| {
            const got = it.next() orelse return error.TestUnexpectedResult;
            try std.testing.expectEqualStrings(want, got);
        }
        try std.testing.expectEqual(@as(?[]const u8, null), it.next());
    }
}

test "BPE obeys merge rank priority, handles merges-before-vocab JSON, and spills chunks >128 bytes" {
    const gpa = std.testing.allocator;
    // "merges" appears BEFORE "vocab"; merge ("l","l") has rank 1 < ("l","o") rank 2,
    // so "hello" must merge ("h","e")->he then ("l","l")->ll -> [he, ll, o], not [hel, lo].
    const json =
        \\{"model":{"merges":[["h","e"],["l","l"],["l","o"],["he","l"],["hel","lo"],["a","a"]],
        \\"vocab":{"a":0,"e":1,"h":2,"l":3,"o":4,"he":5,"ll":6,"lo":7,"hel":8,"hello":9,"aa":10}}}
    ;
    var tok = try Tokenizer.fromHuggingFaceJson(gpa, json);
    defer tok.deinit(gpa);

    var buf: [32]u32 = undefined;
    const ids = try tok.encode(gpa, "hello", &buf);
    try std.testing.expectEqualSlices(u32, &.{ 5, 6, 4 }, ids);

    // Long 200-byte single chunk exceeds STACK_CHUNK_CAP (128), exercising heap fallback.
    const long_a: [200]u8 = @splat('a');
    const long_ids = try tok.encodeAlloc(gpa, &long_a);
    defer gpa.free(long_ids);
    try std.testing.expectEqual(@as(usize, 100), long_ids.len);
    for (long_ids) |id| try std.testing.expectEqual(@as(u32, 10), id);

    // Tiny scratch cleanly reports ScratchTooSmall on long chunk.
    var small_scratch = try EncodeScratch.init(gpa, 16);
    defer small_scratch.deinit(gpa);
    var out_long: [200]u32 = undefined;
    try std.testing.expectError(error.ScratchTooSmall, tok.encodeWithScratch(&small_scratch, &long_a, &out_long));
}

test "synthetic BPE tokenizer supports string and array merges, tie-breaking, special tokens, and lossless round-trip" {
    const gpa = std.testing.allocator;

    for ([_]bool{ false, true }) |use_array_merges| {
        const json = try buildSyntheticTokenizerJson(gpa, use_array_merges);
        defer gpa.free(json);

        var tok = try Tokenizer.fromHuggingFaceJson(gpa, json);
        defer tok.deinit(gpa);

        try std.testing.expectEqual(@as(?u32, 1000), tok.addedTokenId("<|endoftext|>"));
        try std.testing.expectEqual(@as(?u32, 1002), tok.addedTokenId("<|im_start|>"));
        try std.testing.expect(tok.isSpecialToken(1002));
        try std.testing.expect(!tok.isSpecialToken(1004)); // <think> has special=false

        // Leftmost BPE tie-breaking on "aaa": (a,a)->256 then trailing 'a' (97).
        var id_buf: [64]u32 = undefined;
        const aaa_ids = try tok.encode(gpa, "aaa", &id_buf);
        try std.testing.expectEqualSlices(u32, &.{ 256, 'a' }, aaa_ids);

        // Chained merges on "hello hello": "hello" -> 261, " hello" -> 262.
        const hi_ids = try tok.encode(gpa, "hello hello", &id_buf);
        try std.testing.expectEqualSlices(u32, &.{ 261, 262 }, hi_ids);

        // Multi-byte UTF-8 Chinese merge: "你" -> 266.
        const zh_ids = try tok.encode(gpa, "你", &id_buf);
        try std.testing.expectEqualSlices(u32, &.{266}, zh_ids);

        // Longest-match special token splitting (<|im_start|> beats <|im|>).
        const sample = "<|im_start|>system\n    hello 你好世界!\n<|im_end|><|im|><think>aaa</think><|endoftext|>";
        const ids = try tok.encodeAlloc(gpa, sample);
        defer gpa.free(ids);
        try std.testing.expectEqual(@as(u32, 1002), ids[0]); // <|im_start|>
        try std.testing.expectEqual(@as(u32, 1000), ids[ids.len - 1]); // <|endoftext|>

        // Zero-heap-allocation encodeWithScratch produces identical IDs.
        var scratch = try EncodeScratch.init(gpa, 256);
        defer scratch.deinit(gpa);
        var scratch_ids_buf: [256]u32 = undefined;
        const scratch_ids = try tok.encodeWithScratch(&scratch, sample, &scratch_ids_buf);
        try std.testing.expectEqualSlices(u32, ids, scratch_ids);

        // Lossless decode round-trip.
        var out_buf: [256]u8 = undefined;
        const decoded = try tok.decode(ids, &out_buf);
        try std.testing.expectEqualStrings(sample, decoded);

        // decodeSkippingSpecial strips special=true tokens while keeping special=false (<think>).
        const skipped = try tok.decodeSkippingSpecial(ids, &out_buf);
        try std.testing.expectEqualStrings("system\n    hello 你好世界!\n<think>aaa</think>", skipped);

        // Empty input and raw non-UTF-8 bytes round-trip losslessly without crashing.
        const empty_ids = try tok.encode(gpa, "", &id_buf);
        try std.testing.expectEqual(@as(usize, 0), empty_ids.len);
        const empty_dec = try tok.decode(empty_ids, &out_buf);
        try std.testing.expectEqual(@as(usize, 0), empty_dec.len);

        const raw_bin = "\x00\xFF\xFE\x80\xAD";
        const bin_ids = try tok.encode(gpa, raw_bin, &id_buf);
        const bin_dec = try tok.decode(bin_ids, &out_buf);
        try std.testing.expectEqualSlices(u8, raw_bin, bin_dec);

        // Error checks: tiny buffer and unknown token ID.
        var tiny: [2]u8 = undefined;
        try std.testing.expectError(error.BufferTooSmall, tok.decode(ids, &tiny));
        try std.testing.expectError(error.UnknownTokenId, tok.tokenBytes(999_999));
    }
}

test "fromHuggingFaceJson rejects malformed JSON, missing vocab, and dangling merges" {
    const gpa = std.testing.allocator;
    try std.testing.expectError(error.InvalidJson, Tokenizer.fromHuggingFaceJson(gpa, "{not json"));
    try std.testing.expectError(error.MissingVocab, Tokenizer.fromHuggingFaceJson(gpa, "{\"model\":{\"type\":\"BPE\"}}"));
    try std.testing.expectError(
        error.InvalidMerge,
        Tokenizer.fromHuggingFaceJson(gpa, "{\"model\":{\"vocab\":{\"a\":0},\"merges\":[\"a b\"]}}"),
    );
}

test "opportunistic Qwen3.5-0.8B tokenizer.json round-trips Chinese, code, and special tokens" {
    const gpa = std.testing.allocator;
    const snapshots_path = "/Users/ariakage/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots";
    var snapshots_dir = std.Io.Dir.openDirAbsolute(std.testing.io, snapshots_path, .{ .iterate = true }) catch |err| switch (err) {
        error.FileNotFound, error.NotDir => return,
        else => return err,
    };
    defer snapshots_dir.close(std.testing.io);

    var json_bytes: ?[]u8 = null;
    defer if (json_bytes) |buf| gpa.free(buf);

    var it = snapshots_dir.iterate();
    while (try it.next(std.testing.io)) |entry| {
        if (entry.kind != .directory) continue;
        var snap_dir = snapshots_dir.openDir(std.testing.io, entry.name, .{}) catch |err| switch (err) {
            error.FileNotFound, error.NotDir => continue,
            else => return err,
        };
        defer snap_dir.close(std.testing.io);
        json_bytes = snap_dir.readFileAlloc(std.testing.io, "tokenizer.json", gpa, .limited(64 * 1024 * 1024)) catch |err| switch (err) {
            error.FileNotFound => continue,
            else => return err,
        };
        break;
    }

    const bytes = json_bytes orelse return;
    var tok = try Tokenizer.fromHuggingFaceJson(gpa, bytes);
    defer tok.deinit(gpa);

    try std.testing.expectEqual(@as(?u32, 248044), tok.addedTokenId("<|endoftext|>"));
    try std.testing.expectEqual(@as(?u32, 248045), tok.addedTokenId("<|im_start|>"));
    try std.testing.expectEqual(@as(?u32, 248046), tok.addedTokenId("<|im_end|>"));
    try std.testing.expectEqual(@as(?u32, 248068), tok.addedTokenId("<think>"));
    try std.testing.expectEqual(@as(?u32, 248069), tok.addedTokenId("</think>"));

    // Verify exact token IDs against Qwen3.5-0.8B BPE merges & pre-tokenization.
    var exact_buf: [64]u32 = undefined;
    const exact_ids = try tok.encode(gpa, "<|im_start|>user\nHello, world! 2026\n<|im_end|>", &exact_buf);
    try std.testing.expectEqualSlices(
        u32,
        &.{ 248045, 846, 198, 9419, 11, 1814, 0, 220, 17, 15, 17, 21, 198, 248046 },
        exact_ids,
    );

    const prompt =
        "<|im_start|>system\n" ++
        "You are aminfer, an Apple Silicon inference engine.\n" ++
        "<|im_end|>\n" ++
        "<|im_start|>user\n" ++
        "请用 Zig 0.17 写一个 16 KiB 对齐函数：\n" ++
        "```zig\n" ++
        "pub fn alignUp(n: usize) usize {\n" ++
        "    return (n + 16383) & ~@as(usize, 16383);\n" ++
        "}\n" ++
        "```\n" ++
        "<|im_end|>\n" ++
        "<|im_start|>assistant\n" ++
        "<think>检查 16 KiB 对齐边界与溢出保护。</think>\n" ++
        "你好，世界！🚀\n" ++
        "<|im_end|>";

    const ids = try tok.encodeAlloc(gpa, prompt);
    defer gpa.free(ids);

    // Special tokens at boundaries and real BPE compression occurred.
    try std.testing.expectEqual(@as(u32, 248045), ids[0]);
    try std.testing.expectEqual(@as(u32, 248046), ids[ids.len - 1]);
    try std.testing.expect(ids.len < prompt.len / 2);

    const round_tripped = try tok.decodeAlloc(gpa, ids);
    defer gpa.free(round_tripped);
    try std.testing.expectEqualStrings(prompt, round_tripped);
}

test "DeepSeek pre-tokenizer groups 1-3 digits, isolates CJK/Kana, and joins ASCII punct+word" {
    var it = PreTokenIterator.initWithKind("DeepSeek-V4.1 2026年1017号 .foo_bar", .deepseek);
    const expected = [_][]const u8{
        "DeepSeek",
        "-V",
        "4",
        ".",
        "1",
        " ",
        "202",
        "6",
        "年",
        "101",
        "7",
        "号",
        " .",
        "foo",
        "_bar",
    };
    for (expected) |exp| {
        const got = it.next() orelse return error.TestUnexpectedResult;
        try std.testing.expectEqualStrings(exp, got);
    }
    try std.testing.expect(it.next() == null);

    var it2 = PreTokenIterator.initWithKind(".foo_bar", .deepseek);
    try std.testing.expectEqualStrings(".foo", it2.next().?);
    try std.testing.expectEqualStrings("_bar", it2.next().?);
    try std.testing.expect(it2.next() == null);
}

test "fromHuggingFaceJson auto-detects DeepSeek pre_tokenizer dialect" {
    const gpa = std.testing.allocator;
    const base_json = try buildSyntheticTokenizerJson(gpa, false);
    defer gpa.free(base_json);
    const prefix =
        "{\"pre_tokenizer\":{\"type\":\"Sequence\",\"pretokenizers\":[" ++
        "{\"type\":\"Split\",\"pattern\":{\"Regex\":\"\\\\p{N}{1,3}\"},\"behavior\":\"Isolated\",\"invert\":false}," ++
        "{\"type\":\"Split\",\"pattern\":{\"Regex\":\"[\\u4e00-\\u9fa5\\u3040-\\u309f\\u30a0-\\u30ff]+\"},\"behavior\":\"Isolated\",\"invert\":false}," ++
        "{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"trim_offsets\":true,\"use_regex\":false}" ++
        "]},";
    const combined = try std.fmt.allocPrint(gpa, "{s}{s}", .{ prefix, base_json[1..] });
    defer gpa.free(combined);

    var tok = try Tokenizer.fromHuggingFaceJson(gpa, combined);
    defer tok.deinit(gpa);
    try std.testing.expectEqual(PreTokenizerKind.deepseek, tok.pre_tokenizer_kind);
}

fn readOptionalMetadataTokenizerAlloc(gpa: std.mem.Allocator, io: std.Io, subpath: []const u8) !?[]u8 {
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = (try paths.resolveMetadataFileZ(null, &path_buf, subpath)) orelse return null;
    return std.Io.Dir.readFileAlloc(
        std.Io.Dir.cwd(),
        io,
        path,
        gpa,
        .limited(64 * 1024 * 1024),
    ) catch |err| switch (err) {
        error.FileNotFound, error.NotDir => null,
        else => return err,
    };
}

test "opportunistic live Qwen3.8-Flash-Next and DeepSeek-V4.1-Flash-1017 tokenizer.json compatibility" {
    const gpa = std.testing.allocator;

    // 1. Qwen/Qwen3.8-Flash-Next
    if (try readOptionalMetadataTokenizerAlloc(gpa, std.testing.io, "Qwen__Qwen3.8-Flash-Next/tokenizer.json")) |bytes| {
        defer gpa.free(bytes);
        var tok = try Tokenizer.fromHuggingFaceJson(gpa, bytes);
        defer tok.deinit(gpa);

        try std.testing.expectEqual(PreTokenizerKind.qwen, tok.pre_tokenizer_kind);
        try std.testing.expectEqual(@as(?u32, 248044), tok.addedTokenId("<|endoftext|>"));
        try std.testing.expectEqual(@as(?u32, 248045), tok.addedTokenId("<|im_start|>"));
        try std.testing.expectEqual(@as(?u32, 248046), tok.addedTokenId("<|im_end|>"));
        try std.testing.expectEqual(@as(?u32, 248068), tok.addedTokenId("<think>"));
        try std.testing.expectEqual(@as(?u32, 248069), tok.addedTokenId("</think>"));

        const sample = "<|im_start|>user\nQwen3.8-Flash-Next 2026 16 KiB 对齐！\n<|im_end|>";
        var scratch = try EncodeScratch.init(gpa, 256);
        defer scratch.deinit(gpa);
        var id_buf: [256]u32 = undefined;
        const ids = try tok.encodeWithScratch(&scratch, sample, &id_buf);
        try std.testing.expectEqual(@as(u32, 248045), ids[0]);
        try std.testing.expectEqual(@as(u32, 248046), ids[ids.len - 1]);

        var out_buf: [512]u8 = undefined;
        const decoded = try tok.decode(ids, &out_buf);
        try std.testing.expectEqualStrings(sample, decoded);
        const plain = try tok.decodeSkippingSpecial(ids, &out_buf);
        try std.testing.expectEqualStrings("user\nQwen3.8-Flash-Next 2026 16 KiB 对齐！\n", plain);
    }

    // 2. deepseek-ai/DeepSeek-V4.1-Flash-1017
    if (try readOptionalMetadataTokenizerAlloc(gpa, std.testing.io, "deepseek-ai__DeepSeek-V4.1-Flash-1017/tokenizer.json")) |bytes| {
        defer gpa.free(bytes);
        var tok = try Tokenizer.fromHuggingFaceJson(gpa, bytes);
        defer tok.deinit(gpa);

        try std.testing.expectEqual(PreTokenizerKind.deepseek, tok.pre_tokenizer_kind);
        try std.testing.expectEqual(@as(usize, 129_280), tok.vocabSize());
        try std.testing.expectEqual(@as(?u32, 0), tok.addedTokenId("<｜begin▁of▁sentence｜>"));
        try std.testing.expectEqual(@as(?u32, 1), tok.addedTokenId("<｜end▁of▁sentence｜>"));
        try std.testing.expectEqual(@as(?u32, 2), tok.addedTokenId("<｜▁pad▁｜>"));
        try std.testing.expect(tok.isSpecialToken(0));
        try std.testing.expect(tok.isSpecialToken(1));

        const sample =
            "<｜begin▁of▁sentence｜>DeepSeek-V4.1-Flash-1017: 2026年10月17日，Apple Silicon 16 KiB zero-copy!\n" ++
            "pub fn alignUp(n: usize) usize { return (n + 16383) & ~@as(usize, 16383); }<｜end▁of▁sentence｜>";
        var scratch = try EncodeScratch.init(gpa, 256);
        defer scratch.deinit(gpa);
        var id_buf: [256]u32 = undefined;
        const ids = try tok.encodeWithScratch(&scratch, sample, &id_buf);
        try std.testing.expectEqual(@as(u32, 0), ids[0]);
        try std.testing.expectEqual(@as(u32, 1), ids[ids.len - 1]);
        try std.testing.expect(ids.len < sample.len / 2);

        var out_buf: [512]u8 = undefined;
        const decoded = try tok.decode(ids, &out_buf);
        try std.testing.expectEqualStrings(sample, decoded);
        const plain = try tok.decodeSkippingSpecial(ids, &out_buf);
        try std.testing.expect(!std.mem.containsAtLeast(u8, plain, 1, "begin▁of▁sentence"));
    }
}

pub fn matchQwenPreTokenScalarRef(rem: []const u8) usize {
    if (rem[0] == '\'') {
        if (rem.len >= 2) {
            switch (asciiLower(rem[1])) {
                's', 't', 'm', 'd' => return 2,
                'r' => if (rem.len >= 3 and asciiLower(rem[2]) == 'e') return 3,
                'v' => if (rem.len >= 3 and asciiLower(rem[2]) == 'e') return 3,
                'l' => if (rem.len >= 3 and asciiLower(rem[2]) == 'l') return 3,
                else => {},
            }
        }
    }
    const c0 = decodeNextCp(rem);
    if (isLetterOrMark(c0.cp)) {
        var end = c0.len;
        while (end < rem.len) {
            const c = decodeNextCp(rem[end..]);
            if (!isLetterOrMark(c.cp)) break;
            end += c.len;
        }
        return end;
    }
    if (c0.cp != '\r' and c0.cp != '\n' and !isNumber(c0.cp) and c0.len < rem.len) {
        const c1 = decodeNextCp(rem[c0.len..]);
        if (isLetterOrMark(c1.cp)) {
            var end = c0.len + c1.len;
            while (end < rem.len) {
                const c = decodeNextCp(rem[end..]);
                if (!isLetterOrMark(c.cp)) break;
                end += c.len;
            }
            return end;
        }
    }
    if (isNumber(c0.cp)) return c0.len;
    {
        var p_start: ?usize = null;
        if (isPunctOrSymbol(c0.cp)) {
            p_start = c0.len;
        } else if (c0.cp == ' ' and c0.len < rem.len) {
            const c1 = decodeNextCp(rem[c0.len..]);
            if (isPunctOrSymbol(c1.cp)) p_start = c0.len + c1.len;
        }
        if (p_start) |start| {
            var end = start;
            while (end < rem.len) {
                const c = decodeNextCp(rem[end..]);
                if (!isPunctOrSymbol(c.cp)) break;
                end += c.len;
            }
            while (end < rem.len and (rem[end] == '\r' or rem[end] == '\n')) end += 1;
            return end;
        }
    }
    if (isWhitespace(c0.cp)) {
        var ws_end: usize = 0;
        var last_ws_start: usize = 0;
        var after_last_nl: ?usize = null;
        while (ws_end < rem.len) {
            const c = decodeNextCp(rem[ws_end..]);
            if (!isWhitespace(c.cp)) break;
            last_ws_start = ws_end;
            ws_end += c.len;
            if (c.cp == '\r' or c.cp == '\n') after_last_nl = ws_end;
        }
        if (after_last_nl) |nl_end| return nl_end;
        if (ws_end == rem.len) return ws_end;
        if (last_ws_start > 0) return last_ws_start;
        return ws_end;
    }
    return c0.len;
}

fn matchDeepSeekPreTokenScalarRef(rem: []const u8) usize {
    const c0 = decodeNextCp(rem);
    if (isNumber(c0.cp)) {
        var end = c0.len;
        var count: usize = 1;
        while (count < 3 and end < rem.len) {
            const c = decodeNextCp(rem[end..]);
            if (!isNumber(c.cp)) break;
            end += c.len;
            count += 1;
        }
        return end;
    }
    if (isDeepSeekCjk(c0.cp)) {
        var end = c0.len;
        while (end < rem.len) {
            const c = decodeNextCp(rem[end..]);
            if (!isDeepSeekCjk(c.cp)) break;
            end += c.len;
        }
        return end;
    }
    var sub_end: usize = c0.len;
    while (sub_end < rem.len) {
        const c = decodeNextCp(rem[sub_end..]);
        if (isNumber(c.cp) or isDeepSeekCjk(c.cp)) break;
        sub_end += c.len;
    }
    const sub = rem[0..sub_end];
    if (sub.len >= 2 and isAsciiPunct(sub[0]) and isAsciiAlpha(sub[1])) {
        var end: usize = 2;
        while (end < sub.len and isAsciiAlpha(sub[end])) end += 1;
        return end;
    }
    if (isLetterOrMark(c0.cp)) {
        var end = c0.len;
        while (end < sub.len) {
            const c = decodeNextCp(sub[end..]);
            if (!isLetterOrMark(c.cp)) break;
            end += c.len;
        }
        return end;
    }
    if (c0.cp != '\r' and c0.cp != '\n' and !isPunctOrSymbol(c0.cp) and c0.len < sub.len) {
        const c1 = decodeNextCp(sub[c0.len..]);
        if (isLetterOrMark(c1.cp)) {
            var end = c0.len + c1.len;
            while (end < sub.len) {
                const c = decodeNextCp(sub[end..]);
                if (!isLetterOrMark(c.cp)) break;
                end += c.len;
            }
            return end;
        }
    }
    {
        var p_start: ?usize = null;
        if (isPunctOrSymbol(c0.cp)) {
            p_start = c0.len;
        } else if (c0.cp == ' ' and c0.len < sub.len) {
            const c1 = decodeNextCp(sub[c0.len..]);
            if (isPunctOrSymbol(c1.cp)) p_start = c0.len + c1.len;
        }
        if (p_start) |start| {
            var end = start;
            while (end < sub.len) {
                const c = decodeNextCp(sub[end..]);
                if (!isPunctOrSymbol(c.cp)) break;
                end += c.len;
            }
            while (end < sub.len and (sub[end] == '\r' or sub[end] == '\n')) end += 1;
            return end;
        }
    }
    if (isWhitespace(c0.cp)) {
        var ws_end: usize = 0;
        var last_ws_start: usize = 0;
        var after_last_nl: ?usize = null;
        while (ws_end < sub.len) {
            const c = decodeNextCp(sub[ws_end..]);
            if (!isWhitespace(c.cp)) break;
            last_ws_start = ws_end;
            ws_end += c.len;
            if (c.cp == '\r' or c.cp == '\n') after_last_nl = ws_end;
        }
        if (after_last_nl) |nl_end| return nl_end;
        if (ws_end == sub.len) return ws_end;
        if (last_ws_start > 0) return last_ws_start;
        return ws_end;
    }
    return c0.len;
}

fn verifyPreTokenParity(text: []const u8, kind: PreTokenizerKind) !void {
    var it = PreTokenIterator.initWithKind(text, kind);
    var pos: usize = 0;
    while (pos < text.len) {
        const expected_len = switch (kind) {
            .qwen => matchQwenPreTokenScalarRef(text[pos..]),
            .deepseek => matchDeepSeekPreTokenScalarRef(text[pos..]),
        };
        const got = it.next() orelse return error.TestUnexpectedResult;
        try std.testing.expectEqualStrings(text[pos .. pos + expected_len], got);
        pos += expected_len;
    }
    try std.testing.expect(it.next() == null);
}

test "C23 ARM64 NEON PreTokenIterator matches scalar bit-for-bit across prose, code, numbers, contractions, CJK/Kana, emoji, and 0..129 lengths" {
    // 1. Exhaustive 0..129 boundary sweep on NEON low-level class scanners.
    var buf: [160]u8 = undefined;
    for (0..130) |len| {
        for (0..len) |i| buf[i] = if ((i & 1) == 0) 'A' + @as(u8, @intCast(i % 26)) else 'a' + @as(u8, @intCast(i % 26));
        try std.testing.expectEqual(len, neonScanAsciiAlpha(buf[0..len]));
        if (len > 0) {
            const stops = [_]u8{ '0', ' ', '@', '[', '`', '{', 0x7F, 0x80, 0xE4, 0xFF };
            for (stops) |st| {
                const mid = len / 2;
                const saved = buf[mid];
                buf[mid] = st;
                try std.testing.expectEqual(mid, neonScanAsciiAlpha(buf[0..len]));
                buf[mid] = saved;
            }
        }

        // Punctuation/symbol scan
        const puncts = [_]u8{ '!', '-', '.', ':', '@', '[', '_', '{', '~', 0x01, 0x7F };
        for (0..len) |i| buf[i] = puncts[i % puncts.len];
        try std.testing.expectEqual(len, neonScanAsciiPunctOrSym(buf[0..len]));
        if (len > 0) {
            const non_puncts = [_]u8{ 'a', 'Z', '5', ' ', '\t', '\n', '\r', 0x0B, 0x0C, 0x80, 0xE4 };
            for (non_puncts) |np| {
                const pos = len - 1;
                const saved = buf[pos];
                buf[pos] = np;
                try std.testing.expectEqual(pos, neonScanAsciiPunctOrSym(buf[0..len]));
                buf[pos] = saved;
            }
        }

        // CRLF scan
        for (0..len) |i| buf[i] = if ((i & 1) == 0) '\r' else '\n';
        try std.testing.expectEqual(len, neonScanCrLf(buf[0..len]));
        if (len > 0) {
            buf[len - 1] = ' ';
            try std.testing.expectEqual(len - 1, neonScanCrLf(buf[0..len]));
        }
    }

    // 2. Diverse corpus covering both `.qwen` and `.deepseek`:
    //    - long ASCII words (> 64 B and > 128 B)
    //    - code indentation (spaces/tabs/newlines > 64 B), operators, punctuation runs > 64 B
    //    - numbers (ASCII + Unicode `\p{N}`), contractions
    //    - CJK & Kana including boundary codepoints (`\u{4DFF}`, `\u{4E00}`, `\u{9F7F}`, `\u{9F80}`, `\u{9FA5}`, `\u{9FA6}`, `\u{303F}`, `\u{3040}`, `\u{30A0}`, `\u{30FB}`, `\u{30FF}`)
    //    - emoji (4-byte UTF-8), combining diacritics (`e\u{0301}`), Unicode whitespace (`\u{3000}`, `\u{00A0}`), malformed UTF-8
    const corpora = [_][]const u8{
        "",
        "a",
        "don't I've WE'LL they're she'd it's you'll I'm CAN'T 'RE 'VE 'LL 'S 'T 'D 'M",
        "SupercalifragilisticexpialidociousSupercalifragilisticexpialidociousSupercalifragilisticexpialidocious end",
        "Supercalifragilisticexpialidocious\u{0301}CombiningMarkInMiddleOfVeryLongAsciiWord1234567890",
        "                                                                        def f(x):\r\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\treturn x + 42\r\n\r\n\n\n\r\r\n\n\n\n\n\n\n\n\n\n\n\n\n\n",
        "    \t   \n   \r\n   \u{3000}\u{00A0}    42   你好   \t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t   123",
        "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\r\n\r\n" ++
            "----====++++****////::::;;;;@@@@[[[[]]]]{{{{}}}}~~~~\n\n\n",
        "pub fn alignUp(n: usize) usize { return (n + 16383) & ~@as(usize, 16383); } // .foo_bar -> ::baz_qux\n",
        // Long CJK (> 60 bytes = > 20 chars) + Kana + exact boundary codepoints
        "苹果芯片统一内存零拷贝推理引擎高性能矢量扫描器测试样本苹果芯片统一内存零拷贝推理引擎高性能矢量扫描器测试样本！" ++
            "あいうえおかきくけこさしすせそたちつてとなにぬねのアイウエオカキクケコサシスセソタチツテト" ++
            "\u{4DFF}\u{4E00}\u{4FFF}\u{5000}\u{8FFF}\u{9000}\u{9F7F}\u{9F80}\u{9FA5}\u{9FA6}" ++
            "\u{303F}\u{3040}\u{3041}\u{309F}\u{30A0}\u{30A1}\u{30FB}\u{30FF}\u{3100}",
        // Unicode digits, emoji, ZWJ, flags, Cyrillic, Greek, Arabic, Korean, malformed UTF-8
        "2026年1017号 0123456789 ٢٣٤٥ ２０２６ ²³ 🚀🔥🧠👨‍👩‍👧‍👦🇺🇸 Привет мир Αθήνα مرحبا 안녕하세요 \xFF\xFE\xC0\xAF\xE4\x20",
        // Space right before CJK / number vs space right before ASCII word (critical DeepSeek Stage 3.4 sub_end boundary!)
        "   你好   123   abc   \u{30A0}   \u{30FB}   .veryLongAsciiIdentifierExceedingSixtyFourBytesForStage31NeonFastPath123",
    };

    for (corpora) |text| {
        try verifyPreTokenParity(text, .qwen);
        try verifyPreTokenParity(text, .deepseek);
    }

    // 3. Full BPE token-ID parity (encoding with NEON PreTokenIterator vs scalar-pretokenized BPE encode)
    //    on synthetic tokenizer across both `.qwen` and `.deepseek` dialects.
    const gpa = std.testing.allocator;
    const json_bytes = try buildSyntheticTokenizerJson(gpa, false);
    defer gpa.free(json_bytes);
    var tok = try Tokenizer.fromHuggingFaceJson(gpa, json_bytes);
    defer tok.deinit(gpa);

    const mixed_prompt =
        "don't hesitate! Here is a 72-byte identifier: " ++
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrst " ++
        "and code:\n                    return (x + 16383) & ~16383;\r\n" ++
        "中文测试样本：苹果芯片统一内存零拷贝推理引擎高性能矢量扫描器！あいうえおアイウエオ 2026 🚀\n";

    var scratch = try EncodeScratch.init(gpa, 4096);
    defer scratch.deinit(gpa);
    var neon_ids_buf: [2048]u32 = undefined;
    var scalar_ids_buf: [2048]u32 = undefined;

    const TestSliceWriter = struct {
        buf: []u32,
        len: usize = 0,
        fn push(self: *@This(), id: u32) EncodeError!void {
            if (self.len >= self.buf.len) return error.BufferTooSmall;
            self.buf[self.len] = id;
            self.len += 1;
        }
    };

    for ([_]PreTokenizerKind{ .qwen, .deepseek }) |kind| {
        tok.pre_tokenizer_kind = kind;
        const neon_ids = try tok.encodeWithScratch(&scratch, mixed_prompt, &neon_ids_buf);

        // Encode chunk-by-chunk using the scalar reference pre-tokenizer and compare every single token ID!
        var scalar_writer = TestSliceWriter{ .buf = &scalar_ids_buf };
        var pos: usize = 0;
        while (pos < mixed_prompt.len) {
            const clen = switch (kind) {
                .qwen => matchQwenPreTokenScalarRef(mixed_prompt[pos..]),
                .deepseek => matchDeepSeekPreTokenScalarRef(mixed_prompt[pos..]),
            };
            const chunk = mixed_prompt[pos .. pos + clen];
            try tok.encodeChunk(&scalar_writer, null, &scratch, chunk);
            pos += clen;
        }
        try std.testing.expectEqualSlices(u32, scalar_ids_buf[0..scalar_writer.len], neon_ids);
    }
}

test {
    _ = chat_template;
}

test "W24 o200k pre-tokenizer handles contractions, Unicode letter cases, digits 1-3, punctuation slashes, and whitespace" {
    // Contractions + uppercase/lowercase prefix rules (o200k attaches contractions to the word)
    {
        var it = PreTokenIterator.initO200k("don't I'VE We'll HTTPServer café\u{0301}");
        const expected = [_][]const u8{
            "don't",
            " I'VE",
            " We'll",
            " HTTPServer",
            " café\u{0301}",
        };
        for (expected) |exp| {
            const got = it.next() orelse return error.TestUnexpectedResult;
            try std.testing.expectEqualStrings(exp, got);
        }
        try std.testing.expect(it.next() == null);
    }

    // 1..3 numbers + punctuation with trailing CRLF/slash + standalone whitespace on `v12345!\r\n/next   \n   abc`
    var chunks: [16][]const u8 = undefined;
    var count: usize = 0;
    var it2 = PreTokenIterator.initO200k("v12345!\r\n/next   \n   abc");
    while (it2.next()) |chunk| {
        chunks[count] = chunk;
        count += 1;
    }
    try std.testing.expectEqual(@as(usize, 8), count);
    try std.testing.expectEqualStrings("v", chunks[0]);
    try std.testing.expectEqualStrings("123", chunks[1]);
    try std.testing.expectEqualStrings("45", chunks[2]);
    try std.testing.expectEqualStrings("!\r\n/", chunks[3]);
    try std.testing.expectEqualStrings("next", chunks[4]);
    try std.testing.expectEqualStrings("   \n", chunks[5]);
    try std.testing.expectEqualStrings("  ", chunks[6]);
    try std.testing.expectEqualStrings(" abc", chunks[7]);
}

test "W24 Inkling 2 synthetic tokenizer: o200k split + ignore_merges full-piece vocab override + special tokens" {
    const gpa = std.testing.allocator;
    const o200k_pat =
        "[^\\\\r\\\\n\\\\p{L}\\\\p{N}]?[\\\\p{Lu}\\\\p{Lt}\\\\p{Lm}\\\\p{Lo}\\\\p{M}]*[\\\\p{Ll}\\\\p{Lm}\\\\p{Lo}\\\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|" ++
        "[^\\\\r\\\\n\\\\p{L}\\\\p{N}]?[\\\\p{Lu}\\\\p{Lt}\\\\p{Lm}\\\\p{Lo}\\\\p{M}]+[\\\\p{Ll}\\\\p{Lm}\\\\p{Lo}\\\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|" ++
        "\\\\p{N}{1,3}| ?[^\\\\s\\\\p{L}\\\\p{N}]+[\\\\r\\\\n/]*|\\\\s*[\\\\r\\\\n]+|\\\\s+(?!\\\\S)|\\\\s+";

    const json_bytes = try std.fmt.allocPrint(
        gpa,
        "{{\"version\":\"1.0\",\"model\":{{\"type\":\"BPE\",\"ignore_merges\":true,\"vocab\":{{\"h\":0,\"e\":1,\"l\":2,\"o\":3,\"he\":4,\"ll\":5,\"hello\":6,\"Ġ\":7,\"!\":8,\"1\":9,\"2\":10,\"3\":11}},\"merges\":[[\"h\",\"e\"],[\"l\",\"l\"]]}},\"pre_tokenizer\":{{\"type\":\"Sequence\",\"pretokenizers\":[{{\"type\":\"Split\",\"pattern\":{{\"Regex\":\"{s}\"}},\"behavior\":\"Isolated\"}},{{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"trim_offsets\":true,\"use_regex\":false}}]}},\"added_tokens\":[{{\"id\":12,\"content\":\"<|message_system|>\",\"special\":true}},{{\"id\":13,\"content\":\"<|content_thinking|>\",\"special\":true}},{{\"id\":14,\"content\":\"<|end_message|>\",\"special\":true}},{{\"id\":15,\"content\":\"<|content_model_end_sampling|>\",\"special\":true}}]}}",
        .{o200k_pat},
    );
    defer gpa.free(json_bytes);

    var tok = try Tokenizer.fromHuggingFaceJson(gpa, json_bytes);
    defer tok.deinit(gpa);

    try std.testing.expectEqual(TokenizerDialect.inkling2, tok.dialect);
    try std.testing.expect(tok.use_o200k_split);
    try std.testing.expect(tok.ignore_merges);
    try std.testing.expectEqual(@as(?u32, 15), tok.eosTokenId());
    try std.testing.expectEqual(@as(?u32, 12), tok.addedTokenId("<|message_system|>"));

    var ids_buf: [64]u32 = undefined;
    // With ignore_merges = true, `"hello"` hits `full_piece_vocab` directly as token 6 even though
    // `merges` has no rule combining `"he" + "ll" + "o"` -> `"hello"`.
    const ids_ignore = try tok.encode(gpa, "<|message_system|>hello!", &ids_buf);
    try std.testing.expectEqualSlices(u32, &[_]u32{ 12, 6, 8 }, ids_ignore);

    const decoded = try tok.decodeAlloc(gpa, ids_ignore);
    defer gpa.free(decoded);
    try std.testing.expectEqualStrings("<|message_system|>hello!", decoded);

    // Flipping ignore_merges off reverts `"hello"` to the BPE merge loop (`he`=4, `ll`=5, `o`=3).
    tok.ignore_merges = false;
    const ids_bpe = try tok.encode(gpa, "hello", &ids_buf);
    try std.testing.expectEqualSlices(u32, &[_]u32{ 4, 5, 3 }, ids_bpe);

    // Detected ChatTemplate dialect matches Inkling 2.
    const detected = chat_template.ChatTemplate.fromTokenizer(&tok);
    try std.testing.expectEqual(chat_template.Dialect.inkling2, detected.dialect);
}

test "W24 Hy4 synthetic tokenizer: DeepSeek pre-tokenizer + added_tokens longest-match trie + C-API allow_special_tokens=0 guard" {
    const gpa = std.testing.allocator;
    const ds_pat = "\\\\p{N}{1,3}|[一-龥\\\\u3040-\\\\u309F\\\\u30A0-\\\\u30FF]+|[!\\\"#$%&'()*+,\\\\-./:;<=>?@\\\\[\\\\\\\\\\\\]^_`{|}~][A-Za-z]+";

    const json_bytes = try std.fmt.allocPrint(
        gpa,
        "{{\"version\":\"1.0\",\"model\":{{\"type\":\"BPE\",\"ignore_merges\":false,\"vocab\":{{\"a\":0,\"b\":1,\"r\":2,\"ab\":3}},\"merges\":[[\"a\",\"b\"]]}},\"pre_tokenizer\":{{\"type\":\"Sequence\",\"pretokenizers\":[{{\"type\":\"Split\",\"pattern\":{{\"Regex\":\"{s}\"}},\"behavior\":\"Isolated\"}},{{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"trim_offsets\":true,\"use_regex\":false}}]}},\"added_tokens\":[{{\"id\":100,\"content\":\"<｜hy_start:opensource｜>\",\"special\":true}},{{\"id\":101,\"content\":\"<｜hy_middle:opensource｜>\",\"special\":true}},{{\"id\":102,\"content\":\"<｜hy_end:opensource｜>\",\"special\":true}},{{\"id\":103,\"content\":\"<｜reasoning_mode:opensource｜>\",\"special\":true}},{{\"id\":104,\"content\":\"<think:opensource>\",\"special\":true}},{{\"id\":105,\"content\":\"</think:opensource>\",\"special\":true}},{{\"id\":106,\"content\":\"<｜hy_image▁pad:opensource｜>\",\"special\":true}},{{\"id\":107,\"content\":\"<|foo|>\",\"special\":true}},{{\"id\":108,\"content\":\"<|foo|>barbaz\",\"special\":true}},{{\"id\":109,\"content\":\"<|foo|>bar\",\"special\":true}}]}}",
        .{ds_pat},
    );
    defer gpa.free(json_bytes);

    var tok = try Tokenizer.fromHuggingFaceJson(gpa, json_bytes);
    defer tok.deinit(gpa);

    try std.testing.expectEqual(TokenizerDialect.hy4, tok.dialect);
    try std.testing.expectEqual(PreTokenizerKind.deepseek, tok.pre_tokenizer_kind);
    try std.testing.expect(!tok.ignore_merges);
    try std.testing.expectEqual(@as(?u32, 102), tok.eosTokenId());
    try std.testing.expectEqual(@as(?u32, 103), tok.addedTokenId("<｜reasoning_mode:opensource｜>"));
    try std.testing.expectEqual(@as(?u32, 106), tok.addedTokenId("<｜hy_image▁pad:opensource｜>"));

    var ids_buf: [64]u32 = undefined;
    // Trie longest-match: `<|foo|>barbaz` -> 108 (not `<|foo|>` 107 or `<|foo|>bar` 109), `<|foo|>bar` -> 109
    const ids = try tok.encode(gpa, "<｜hy_start:opensource｜><|foo|>barbaz<|foo|>bar<｜hy_end:opensource｜>", &ids_buf);
    try std.testing.expectEqualSlices(u32, &[_]u32{ 100, 108, 109, 102 }, ids);

    // When a view clears `added_tokens = &.{}` (C-API `allow_special_tokens == 0`), trie does NOT fire.
    var no_special_view = tok;
    no_special_view.added_tokens = &.{};
    try std.testing.expectEqual(@as(?u32, null), no_special_view.addedTokenId("<|foo|>"));
    const plain_ids = try no_special_view.encode(gpa, "ab", &ids_buf);
    try std.testing.expectEqualSlices(u32, &[_]u32{3}, plain_ids);

    // Detected ChatTemplate dialect matches Hy4.
    const detected = chat_template.ChatTemplate.fromTokenizer(&tok);
    try std.testing.expectEqual(chat_template.Dialect.hy4, detected.dialect);
}

test "W24 live tokenizer.json parity for Inkling-2-Mini and Hy4-flash when external metadata exists" {
    const gpa = std.testing.allocator;

    // 1. thinkingmachines/Inkling-2-Mini
    if (try readOptionalMetadataTokenizerAlloc(gpa, std.Options.debug_io, "thinkingmachines__Inkling-2-Mini/tokenizer.json")) |bytes| {
        defer gpa.free(bytes);

        var tok = try Tokenizer.fromHuggingFaceJson(gpa, bytes);
        defer tok.deinit(gpa);

        try std.testing.expectEqual(TokenizerDialect.inkling2, tok.dialect);
        try std.testing.expect(tok.use_o200k_split);
        try std.testing.expect(tok.ignore_merges);
        try std.testing.expectEqual(@as(usize, 60), tok.added_tokens.len);
        try std.testing.expectEqual(@as(usize, 199_998), tok.full_piece_vocab.count());
        try std.testing.expectEqual(
            tok.addedTokenId("<|content_model_end_sampling|>"),
            tok.eosTokenId(),
        );
        try std.testing.expectEqual(
            chat_template.Dialect.inkling2,
            chat_template.ChatTemplate.fromTokenizer(&tok).dialect,
        );

        const sample = "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Compute v12345!\r\n/next café\u{0301} 中文测试<|end_message|><|message_model|><|content_thinking|>";
        var scratch = try EncodeScratch.init(gpa, 4096);
        defer scratch.deinit(gpa);
        var ids_buf: [256]u32 = undefined;
        const ids = try tok.encodeWithScratch(&scratch, sample, &ids_buf);
        try std.testing.expect(ids.len > 10);
        const decoded = try tok.decodeAlloc(gpa, ids);
        defer gpa.free(decoded);
        try std.testing.expectEqualStrings(sample, decoded);
    }

    // 2. tencent/Hy4-flash
    if (try readOptionalMetadataTokenizerAlloc(gpa, std.Options.debug_io, "tencent__Hy4-flash/tokenizer.json")) |bytes| {
        defer gpa.free(bytes);

        var tok = try Tokenizer.fromHuggingFaceJson(gpa, bytes);
        defer tok.deinit(gpa);

        try std.testing.expectEqual(TokenizerDialect.hy4, tok.dialect);
        try std.testing.expectEqual(PreTokenizerKind.deepseek, tok.pre_tokenizer_kind);
        try std.testing.expect(!tok.ignore_merges);
        try std.testing.expectEqual(@as(usize, 832), tok.added_tokens.len);
        try std.testing.expect(tok.addedTokenId("<｜hy_start:opensource｜>") != null);
        try std.testing.expect(tok.addedTokenId("<｜reasoning_mode:opensource｜>") != null);
        try std.testing.expect(tok.addedTokenId("<think:opensource>") != null);
        try std.testing.expect(tok.addedTokenId("<｜hy_image▁pad:opensource｜>") != null);
        try std.testing.expectEqual(
            chat_template.Dialect.hy4,
            chat_template.ChatTemplate.fromTokenizer(&tok).dialect,
        );

        const sample = "<｜hy_start:opensource｜>user<｜hy_middle:opensource｜><｜reasoning_mode:opensource｜>reasoning_effort:highHello world 2026 中文<｜hy_end:opensource｜><｜hy_start:opensource｜>assistant<｜hy_middle:opensource｜><think:opensource>";
        var scratch = try EncodeScratch.init(gpa, 4096);
        defer scratch.deinit(gpa);
        var ids_buf: [256]u32 = undefined;
        const ids = try tok.encodeWithScratch(&scratch, sample, &ids_buf);
        try std.testing.expect(ids.len > 10);
        const decoded = try tok.decodeAlloc(gpa, ids);
        defer gpa.free(decoded);
        try std.testing.expectEqualStrings(sample, decoded);
    }
}
