//! Native Zig Qwen2 / Qwen3 / Qwen3.5 byte-level BPE tokenizer
//! (status row W5, docs/architecture/runtime-pipeline.md).
//!
//! Startup parses HuggingFace `tokenizer.json` into flat compact tables;
//! `decode`, `tokenBytes`, and `encodeWithScratch` perform zero dynamic heap
//! allocation on the inference hot path. Zero Python at runtime.

const std = @import("std");

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

comptime {
    assertSortedRanges(&NON_ASCII_LETTER_RANGES);
    assertSortedRanges(&NON_ASCII_MARK_RANGES);
    assertSortedRanges(&NON_ASCII_NUMBER_RANGES);
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
    /// DeepSeek V3 / V4 / V4.1 3-stage isolated split (`\p{N}{1,3}`, CJK/Kana, punct+ASCII word).
    deepseek,
};

/// Deterministic zero-allocation scanner implementing Qwen (`\p{N}`) and
/// DeepSeek (`\p{N}{1,3}` + CJK isolation + ASCII-punct-word) pre-tokenization.
pub const PreTokenIterator = struct {
    text: []const u8,
    pos: usize,
    kind: PreTokenizerKind = .qwen,

    pub fn init(text: []const u8) PreTokenIterator {
        return .{ .text = text, .pos = 0, .kind = .qwen };
    }

    pub fn initWithKind(text: []const u8, kind: PreTokenizerKind) PreTokenIterator {
        return .{ .text = text, .pos = 0, .kind = kind };
    }

    pub fn next(self: *PreTokenIterator) ?[]const u8 {
        if (self.pos >= self.text.len) return null;
        const rem = self.text[self.pos..];
        const len = switch (self.kind) {
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

inline fn pairKey(left: u32, right: u32) u64 {
    return (@as(u64, left) << 32) | right;
}

pub const Tokenizer = struct {
    token_bytes: []u8,
    dense_tokens: []TokenSpan,
    sparse_tokens: std.AutoHashMapUnmanaged(u32, TokenSpan),
    byte_to_id: [256]?u32,
    merges: std.AutoHashMapUnmanaged(u64, MergeEntry),
    added_tokens: []AddedToken,
    added_first_byte: [256]bool,
    pre_tokenizer_kind: PreTokenizerKind = .qwen,
    token_count: usize,

    /// Parses a HuggingFace `tokenizer.json` payload at startup into compact
    /// flat lookup tables. Supports both `"a b"` string and `["a", "b"]` array
    /// merge formats, Qwen and DeepSeek pre-tokenizers, and arbitrary JSON key ordering.
    pub fn fromHuggingFaceJson(allocator: std.mem.Allocator, json_bytes: []const u8) LoadError!Tokenizer {
        var temp_arena = std.heap.ArenaAllocator.init(allocator);
        defer temp_arena.deinit();
        const temp = temp_arena.allocator();

        var scanner = std.json.Scanner.initCompleteInput(temp, json_bytes);
        defer scanner.deinit();

        const first_tok = nextToken(&scanner, temp) catch |err| return err;
        if (first_tok != .object_begin) return error.InvalidJson;

        var saw_vocab = false;
        var pre_tokenizer_kind: PreTokenizerKind = .qwen;
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
                try parseModelSection(&scanner, temp, &saw_vocab, &temp_vocab, &vocab_list, &raw_merges);
            } else if (std.mem.eql(u8, key, "added_tokens")) {
                try parseAddedTokensSection(&scanner, temp, &raw_added);
            } else if (std.mem.eql(u8, key, "pre_tokenizer")) {
                pre_tokenizer_kind = try parsePreTokenizerSection(&scanner, temp);
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
        for (pending_added.items, 0..) |p, i| {
            const slice = token_bytes[p.offset .. p.offset + p.len];
            added_tokens[i] = .{
                .id = p.id,
                .content = slice,
                .special = p.special,
            };
            added_first_byte[slice[0]] = true;
        }
        std.mem.sort(AddedToken, added_tokens, {}, AddedToken.longestFirst);

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

        return .{
            .token_bytes = token_bytes,
            .dense_tokens = dense_tokens,
            .sparse_tokens = sparse_tokens,
            .byte_to_id = byte_to_id,
            .merges = merges,
            .added_tokens = added_tokens,
            .added_first_byte = added_first_byte,
            .pre_tokenizer_kind = pre_tokenizer_kind,
            .token_count = token_count,
        };
    }

    pub fn deinit(self: *Tokenizer, allocator: std.mem.Allocator) void {
        allocator.free(self.token_bytes);
        allocator.free(self.dense_tokens);
        self.sparse_tokens.deinit(allocator);
        self.merges.deinit(allocator);
        allocator.free(self.added_tokens);
        self.* = undefined;
    }

    pub fn vocabSize(self: *const Tokenizer) usize {
        return self.token_count;
    }

    /// Looks up the token ID of an `added_tokens` entry (e.g. `<|im_start|>`).
    pub fn addedTokenId(self: *const Tokenizer, content: []const u8) ?u32 {
        for (self.added_tokens) |tok| {
            if (std.mem.eql(u8, tok.content, content)) return tok.id;
        }
        return null;
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
        if (!self.added_first_byte[text[pos]]) return null;
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
        var it = PreTokenIterator.initWithKind(segment, self.pre_tokenizer_kind);
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

fn parsePreTokenizerSection(
    scanner: *std.json.Scanner,
    temp: std.mem.Allocator,
) LoadError!PreTokenizerKind {
    var kind: PreTokenizerKind = .qwen;
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
                    if (std.mem.indexOf(u8, s, "\\p{N}{1,3}") != null or
                        std.mem.indexOf(u8, s, "\xe4\xb8\x80-\xe9\xbe\xa5") != null)
                    {
                        kind = .deepseek;
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
    return kind;
}

fn parseModelSection(
    scanner: *std.json.Scanner,
    temp: std.mem.Allocator,
    saw_vocab: *bool,
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

test "opportunistic live Qwen3.8-Flash-Next and DeepSeek-V4.1-Flash-1017 tokenizer.json compatibility" {
    const gpa = std.testing.allocator;

    // 1. Qwen/Qwen3.8-Flash-Next
    if (std.Io.Dir.readFileAlloc(
        std.Io.Dir.cwd(),
        std.testing.io,
        "/Volumes/Akari/models/metadata/Qwen__Qwen3.8-Flash-Next/tokenizer.json",
        gpa,
        .limited(64 * 1024 * 1024),
    )) |bytes| {
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
    } else |err| switch (err) {
        error.FileNotFound, error.NotDir => {},
        else => return err,
    }

    // 2. deepseek-ai/DeepSeek-V4.1-Flash-1017
    if (std.Io.Dir.readFileAlloc(
        std.Io.Dir.cwd(),
        std.testing.io,
        "/Volumes/Akari/models/metadata/deepseek-ai__DeepSeek-V4.1-Flash-1017/tokenizer.json",
        gpa,
        .limited(64 * 1024 * 1024),
    )) |bytes| {
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
    } else |err| switch (err) {
        error.FileNotFound, error.NotDir => {},
        else => return err,
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
