//! Compile-time quantization specialization (ADR-0014): MLX-compatible affine
//! 3-bit and 4-bit group quantization, branch-free per-spec helpers, and CPU
//! fp32 reference dequantization / matrix multiplication used to verify GPU
//! kernels.

const std = @import("std");

/// Supported affine weight bit-widths (ADR-0014).
pub const QuantBits = enum(u8) {
    q3 = 3,
    q4 = 4,

    pub fn bitWidth(self: QuantBits) u8 {
        return @backingInt(self);
    }
};

/// Compile-time quantization configuration shared by host code and GPU
/// function constants (ADR-0014).
pub const QuantSpec = struct {
    bits: QuantBits,
    group_size: u16 = 64,
};

/// Interleaved affine dequantization parameters per group (`half2(scale, bias)`).
/// Value reconstruction: `w = q * scale + bias` (MLX affine convention).
pub const ScaleBias = extern struct {
    scale: f16,
    bias: f16,
};

comptime {
    if (@sizeOf(ScaleBias) != 4) @compileError("ScaleBias must be 4 bytes (half2)");
}

pub const ShapeError = error{
    ZeroDimension,
    UnalignedK,
    Overflow,
};

/// Generates branch-free, compile-time-specialized quantization helpers and
/// CPU fp32 reference math for `spec`. Unsupported configurations are compile
/// errors.
pub fn QuantOps(comptime spec: QuantSpec) type {
    if (spec.group_size != 64) {
        @compileError("aminfer v0 currently specializes group_size = 64 only");
    }
    const bits: usize = spec.bits.bitWidth();
    const gs: usize = spec.group_size;
    // 4-bit: 64 * 4 / 8 = 32 B; 3-bit: 64 * 3 / 8 = 24 B.
    const group_bytes: usize = (gs * bits) / 8;

    return struct {
        pub const quant_spec: QuantSpec = spec;

        pub inline fn groupWeightBytes() usize {
            return group_bytes;
        }

        pub inline fn groupsPerRow(k: usize) ShapeError!usize {
            if (k == 0) return error.ZeroDimension;
            if (k % gs != 0) return error.UnalignedK;
            return k / gs;
        }

        pub inline fn rowWeightBytes(k: usize) ShapeError!usize {
            const groups = try groupsPerRow(k);
            return std.math.mul(usize, groups, group_bytes) catch return error.Overflow;
        }

        pub inline fn packedWeightBytes(n: usize, k: usize) ShapeError!usize {
            if (n == 0) return error.ZeroDimension;
            const rb = try rowWeightBytes(k);
            return std.math.mul(usize, n, rb) catch return error.Overflow;
        }

        pub inline fn scaleBiasBytes(n: usize, k: usize) ShapeError!usize {
            if (n == 0) return error.ZeroDimension;
            const groups = try groupsPerRow(k);
            const total_groups = std.math.mul(usize, n, groups) catch return error.Overflow;
            return std.math.mul(usize, total_groups, @sizeOf(ScaleBias)) catch return error.Overflow;
        }

        /// Packs one group of `group_size` unsigned integer codes (`0..(1<<bits)-1`)
        /// into MLX-compatible LSB-first byte layout.
        pub fn packGroup(codes: *const [gs]u8, out: *[group_bytes]u8) void {
            if (comptime spec.bits == .q4) {
                for (0..gs / 2) |i| {
                    const lo: u8 = codes[2 * i] & 0x0F;
                    const hi: u8 = codes[2 * i + 1] & 0x0F;
                    out[i] = lo | (hi << 4);
                }
            } else if (comptime spec.bits == .q3) {
                for (0..gs / 8) |blk| {
                    const c = codes[blk * 8 ..][0..8];
                    const word: u24 =
                        @as(u24, c[0] & 0x07) |
                        (@as(u24, c[1] & 0x07) << 3) |
                        (@as(u24, c[2] & 0x07) << 6) |
                        (@as(u24, c[3] & 0x07) << 9) |
                        (@as(u24, c[4] & 0x07) << 12) |
                        (@as(u24, c[5] & 0x07) << 15) |
                        (@as(u24, c[6] & 0x07) << 18) |
                        (@as(u24, c[7] & 0x07) << 21);
                    out[blk * 3 + 0] = @truncate(word);
                    out[blk * 3 + 1] = @truncate(word >> 8);
                    out[blk * 3 + 2] = @truncate(word >> 16);
                }
            } else {
                @compileError("unsupported QuantBits");
            }
        }

        /// Dequantizes one group of `group_size` weights to fp32 (`q * scale + bias`).
        pub fn dequantizeGroup(packed_group: *const [group_bytes]u8, sb: ScaleBias, out: *[gs]f32) void {
            const scale: f32 = @floatCast(sb.scale);
            const bias: f32 = @floatCast(sb.bias);
            if (comptime spec.bits == .q4) {
                for (0..gs / 2) |i| {
                    const byte = packed_group[i];
                    out[2 * i] = @as(f32, @floatFromInt(byte & 0x0F)) * scale + bias;
                    out[2 * i + 1] = @as(f32, @floatFromInt(byte >> 4)) * scale + bias;
                }
            } else if (comptime spec.bits == .q3) {
                for (0..gs / 8) |blk| {
                    const b0: u24 = packed_group[blk * 3 + 0];
                    const b1: u24 = packed_group[blk * 3 + 1];
                    const b2: u24 = packed_group[blk * 3 + 2];
                    const word: u24 = b0 | (b1 << 8) | (b2 << 16);
                    inline for (0..8) |j| {
                        const q: u8 = @intCast((word >> @intCast(j * 3)) & 0x07);
                        out[blk * 8 + j] = @as(f32, @floatFromInt(q)) * scale + bias;
                    }
                }
            } else {
                @compileError("unsupported QuantBits");
            }
        }

        /// CPU fp32 reference for `Y[m, n] = sum_k X[m, k] * dequant(W[n, k])`.
        /// `x` is `[m, k]` fp16, `qweight` is `[n, rowWeightBytes(k)]`,
        /// `scale_bias` is `[n, k / group_size]`, `y` is `[m, n]` f32.
        pub fn matmulRef(
            x: []const f16,
            qweight: []const u8,
            scale_bias: []const ScaleBias,
            y: []f32,
            m: usize,
            n: usize,
            k: usize,
        ) (ShapeError || error{BufferTooSmall})!void {
            if (m == 0 or n == 0) return error.ZeroDimension;
            const groups = try groupsPerRow(k);
            const row_bytes = try rowWeightBytes(k);
            if (x.len < m * k or qweight.len < n * row_bytes or scale_bias.len < n * groups or y.len < m * n) {
                return error.BufferTooSmall;
            }
            var deq: [gs]f32 = undefined;
            for (0..m) |mi| {
                const x_row = x[mi * k ..][0..k];
                for (0..n) |ni| {
                    var acc: f32 = 0.0;
                    const w_row = qweight[ni * row_bytes ..][0..row_bytes];
                    const sb_row = scale_bias[ni * groups ..][0..groups];
                    for (0..groups) |g| {
                        dequantizeGroup(w_row[g * group_bytes ..][0..group_bytes], sb_row[g], &deq);
                        const x_grp = x_row[g * gs ..][0..gs];
                        for (0..gs) |j| {
                            acc += @as(f32, @floatCast(x_grp[j])) * deq[j];
                        }
                    }
                    y[mi * n + ni] = acc;
                }
            }
        }
    };
}

test "QuantOps computes exact byte sizes for 4-bit and 3-bit affine group-64" {
    const Q4 = QuantOps(.{ .bits = .q4, .group_size = 64 });
    const Q3 = QuantOps(.{ .bits = .q3, .group_size = 64 });

    try std.testing.expectEqual(@as(usize, 32), Q4.groupWeightBytes());
    try std.testing.expectEqual(@as(usize, 24), Q3.groupWeightBytes());
    try std.testing.expectEqual(@as(usize, 1024), try Q4.rowWeightBytes(2048));
    try std.testing.expectEqual(@as(usize, 768), try Q3.rowWeightBytes(2048));
    try std.testing.expectEqual(@as(usize, 2048 * 1024), try Q4.packedWeightBytes(2048, 2048));
    try std.testing.expectEqual(@as(usize, 2048 * 768), try Q3.packedWeightBytes(2048, 2048));
    try std.testing.expectEqual(@as(usize, 2048 * 32 * 4), try Q4.scaleBiasBytes(2048, 2048));
    try std.testing.expectError(error.UnalignedK, Q4.rowWeightBytes(65));
    try std.testing.expectError(error.ZeroDimension, Q3.rowWeightBytes(0));
}

test "4-bit pack and dequantize matches MLX affine formula" {
    const Q4 = QuantOps(.{ .bits = .q4, .group_size = 64 });
    var codes: [64]u8 = undefined;
    for (&codes, 0..) |*c, i| c.* = @intCast(i % 16);

    var packed_buf: [Q4.groupWeightBytes()]u8 = undefined;
    Q4.packGroup(&codes, &packed_buf);

    // Spot-check LSB-first nibble packing: byte 0 holds codes[0] (low) | codes[1] (high).
    try std.testing.expectEqual(@as(u8, 0x10), packed_buf[0]);
    try std.testing.expectEqual(@as(u8, 0x32), packed_buf[1]);

    const sb: ScaleBias = .{ .scale = @floatCast(0.25), .bias = @floatCast(-1.5) };
    var out: [64]f32 = undefined;
    Q4.dequantizeGroup(&packed_buf, sb, &out);

    for (out, 0..) |val, i| {
        const expected: f32 = @as(f32, @floatFromInt(i % 16)) * 0.25 - 1.5;
        try std.testing.expectApproxEqAbs(expected, val, 1e-6);
    }
}

test "3-bit pack and dequantize matches MLX 8-in-3-bytes bitstream" {
    const Q3 = QuantOps(.{ .bits = .q3, .group_size = 64 });
    var codes: [64]u8 = undefined;
    for (&codes, 0..) |*c, i| c.* = @intCast((i * 5 + 1) % 8);

    var packed_buf: [Q3.groupWeightBytes()]u8 = undefined;
    Q3.packGroup(&codes, &packed_buf);

    // Verify the first 8 3-bit values packed LSB-first into bytes [0..3].
    const word24: u24 =
        @as(u24, codes[0]) |
        (@as(u24, codes[1]) << 3) |
        (@as(u24, codes[2]) << 6) |
        (@as(u24, codes[3]) << 9) |
        (@as(u24, codes[4]) << 12) |
        (@as(u24, codes[5]) << 15) |
        (@as(u24, codes[6]) << 18) |
        (@as(u24, codes[7]) << 21);
    try std.testing.expectEqual(@as(u8, @truncate(word24)), packed_buf[0]);
    try std.testing.expectEqual(@as(u8, @truncate(word24 >> 8)), packed_buf[1]);
    try std.testing.expectEqual(@as(u8, @truncate(word24 >> 16)), packed_buf[2]);

    const sb: ScaleBias = .{ .scale = @floatCast(0.125), .bias = @floatCast(-0.5) };
    var out: [64]f32 = undefined;
    Q3.dequantizeGroup(&packed_buf, sb, &out);

    for (out, 0..) |val, i| {
        const expected: f32 = @as(f32, @floatFromInt(codes[i])) * 0.125 - 0.5;
        try std.testing.expectApproxEqAbs(expected, val, 1e-6);
    }
}

test "reference quantized matmul matches explicit dequantized dot product" {
    const Q3 = QuantOps(.{ .bits = .q3, .group_size = 64 });
    const m = 2;
    const n = 3;
    const k = 64;

    var qweight: [n * Q3.groupWeightBytes()]u8 = undefined;
    var sbs: [n]ScaleBias = undefined;
    var dense_w: [n][k]f32 = undefined;

    for (0..n) |row| {
        var codes: [64]u8 = undefined;
        for (&codes, 0..) |*c, col| c.* = @intCast((row * 7 + col * 3) % 8);
        Q3.packGroup(&codes, qweight[row * 24 ..][0..24]);
        sbs[row] = .{ .scale = @floatCast(0.25 * @as(f32, @floatFromInt(row + 1))), .bias = @floatCast(-0.5) };
        Q3.dequantizeGroup(qweight[row * 24 ..][0..24], sbs[row], &dense_w[row]);
    }

    var x: [m * k]f16 = undefined;
    for (&x, 0..) |*v, idx| v.* = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast(idx % 17)) - 8)) * 0.125);

    var y: [m * n]f32 = undefined;
    try Q3.matmulRef(&x, &qweight, &sbs, &y, m, n, k);

    for (0..m) |mi| {
        for (0..n) |ni| {
            var expected: f32 = 0.0;
            for (0..k) |ki| expected += @as(f32, @floatCast(x[mi * k + ki])) * dense_w[ni][ki];
            try std.testing.expectApproxEqAbs(expected, y[mi * n + ni], 1e-4);
        }
    }
}
