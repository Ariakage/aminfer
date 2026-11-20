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

/// Unquantized 16-bit floating-point weight storage (ADR-0016).
pub const FloatDType = enum(u8) {
    f16 = 0,
    bf16 = 1,
};

/// Per-tensor weight storage for GPU kernels that accept either unquantized
/// 16-bit floats or MLX affine group-64 quantized codes (ADR-0014, ADR-0016).
pub const WeightDType = enum(u8) {
    f16 = 0,
    bf16 = 1,
    q4_affine = 2,
    q3_affine = 3,

    pub fn fromFloat(dtype: FloatDType) WeightDType {
        return switch (dtype) {
            .f16 => .f16,
            .bf16 => .bf16,
        };
    }

    pub fn fromQuant(bits: QuantBits) WeightDType {
        return switch (bits) {
            .q4 => .q4_affine,
            .q3 => .q3_affine,
        };
    }
};

/// Widens raw `bfloat16` bits to `f32` exactly.
pub fn bf16ToF32(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

/// Narrows `f32` to raw `bfloat16` bits with IEEE round-to-nearest-even.
/// NaN stays NaN (quiet bit forced); infinities and signed zero are preserved.
pub fn f32ToBf16(val: f32) u16 {
    const u: u32 = @bitCast(val);
    if (std.math.isNan(val)) return @truncate((u >> 16) | 0x0040);
    const lsb: u32 = (u >> 16) & 1;
    const rounded: u32 = u +% (0x7FFF + lsb);
    return @truncate(rounded >> 16);
}

/// Widens one raw 16-bit weight (`f16` or `bf16` bit pattern) to `f32`.
pub inline fn floatBitsToF32(dtype: FloatDType, bits: u16) f32 {
    return switch (dtype) {
        .f16 => @floatCast(@as(f16, @bitCast(bits))),
        .bf16 => bf16ToF32(bits),
    };
}

/// Narrows `f32` to a raw 16-bit weight bit pattern of `dtype` (RNE).
pub inline fn f32ToFloatBits(dtype: FloatDType, val: f32) u16 {
    return switch (dtype) {
        .f16 => @bitCast(@as(f16, @floatCast(val))),
        .bf16 => f32ToBf16(val),
    };
}

/// CPU fp32 reference for dense `y[n] = sum_k x[k] * W[n, k]`.
/// `w` holds raw `f16` or `bf16` bits of a row-major `[n, k]` matrix.
pub fn denseGemvCpuRef(
    dtype: FloatDType,
    x: []const f16,
    w: []const u16,
    y: []f32,
    n: usize,
    k: usize,
) (ShapeError || error{BufferTooSmall})!void {
    return denseGemmCpuRef(dtype, x, w, y, 1, n, k);
}

/// CPU fp32 reference for dense `Y[m, n] = sum_k X[m, k] * W[n, k]`.
/// `x` is `[m, k]` fp16, `w` holds raw `f16` or `bf16` bits of `[n, k]`, and
/// `y` is `[m, n]` f32. Accumulates strictly in `f32`.
pub fn denseGemmCpuRef(
    dtype: FloatDType,
    x: []const f16,
    w: []const u16,
    y: []f32,
    m: usize,
    n: usize,
    k: usize,
) (ShapeError || error{BufferTooSmall})!void {
    if (m == 0 or n == 0 or k == 0) return error.ZeroDimension;
    const xk = std.math.mul(usize, m, k) catch return error.Overflow;
    const wk = std.math.mul(usize, n, k) catch return error.Overflow;
    const yn = std.math.mul(usize, m, n) catch return error.Overflow;
    if (x.len < xk or w.len < wk or y.len < yn) return error.BufferTooSmall;
    for (0..m) |mi| {
        const x_row = x[mi * k ..][0..k];
        for (0..n) |ni| {
            const w_row = w[ni * k ..][0..k];
            var acc: f32 = 0.0;
            for (0..k) |ki| {
                acc += @as(f32, @floatCast(x_row[ki])) * floatBitsToF32(dtype, w_row[ki]);
            }
            y[mi * n + ni] = acc;
        }
    }
}

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

test "bf16 conversion is exact on widening and round-to-nearest-even on narrowing" {
    try std.testing.expectEqual(@as(f32, 1.0), bf16ToF32(0x3F80));
    try std.testing.expectEqual(@as(f32, -2.0), bf16ToF32(0xC000));
    try std.testing.expectEqual(@as(u16, 0x3F80), f32ToBf16(1.0));
    try std.testing.expectEqual(@as(u16, 0x8000), f32ToBf16(-0.0));
    try std.testing.expectEqual(@as(u16, 0x7F80), f32ToBf16(std.math.inf(f32)));
    try std.testing.expect(std.math.isNan(bf16ToF32(f32ToBf16(std.math.nan(f32)))));
    // Exactly halfway between 0x3F80 and 0x3F81: ties to even (0x3F80).
    try std.testing.expectEqual(@as(u16, 0x3F80), f32ToBf16(@bitCast(@as(u32, 0x3F808000))));
    // Halfway between 0x3F81 and 0x3F82: ties to even (0x3F82).
    try std.testing.expectEqual(@as(u16, 0x3F82), f32ToBf16(@bitCast(@as(u32, 0x3F818000))));
    // Just above halfway rounds up.
    try std.testing.expectEqual(@as(u16, 0x3F81), f32ToBf16(@bitCast(@as(u32, 0x3F808001))));
    // Large magnitudes beyond the fp16 range survive bf16 round-trips.
    try std.testing.expectApproxEqRel(@as(f32, 1.0e6), bf16ToF32(f32ToBf16(1.0e6)), 1e-2);
    // Every bf16 bit pattern (non-NaN) round-trips exactly.
    var bits: u32 = 0;
    while (bits <= 0xFFFF) : (bits += 1) {
        const b: u16 = @intCast(bits);
        const f = bf16ToF32(b);
        if (std.math.isNan(f)) continue;
        try std.testing.expectEqual(b, f32ToBf16(f));
    }
}

test "dense CPU reference GEMM matches explicit f16 and bf16 dot products" {
    const m = 2;
    const n = 3;
    const k = 5;
    var x: [m * k]f16 = undefined;
    for (&x, 0..) |*v, i| v.* = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast(i % 7)) - 3)) * 0.25);
    inline for (.{ FloatDType.f16, FloatDType.bf16 }) |dt| {
        var w: [n * k]u16 = undefined;
        for (&w, 0..) |*v, i| v.* = f32ToFloatBits(dt, @as(f32, @floatFromInt(@as(i32, @intCast(i % 5)) - 2)) * 0.5);
        var y: [m * n]f32 = undefined;
        try denseGemmCpuRef(dt, &x, &w, &y, m, n, k);
        for (0..m) |mi| {
            for (0..n) |ni| {
                var expected: f32 = 0.0;
                for (0..k) |ki| expected += @as(f32, @floatCast(x[mi * k + ki])) * floatBitsToF32(dt, w[ni * k + ki]);
                try std.testing.expectEqual(expected, y[mi * n + ni]);
            }
        }
        var y1: [n]f32 = undefined;
        try denseGemvCpuRef(dt, x[0..k], &w, &y1, n, k);
        for (0..n) |ni| try std.testing.expectEqual(y[ni], y1[ni]);
    }
    var tiny: [1]f32 = undefined;
    try std.testing.expectError(error.BufferTooSmall, denseGemmCpuRef(.f16, &x, &[_]u16{0}, &tiny, 1, 1, 2));
    try std.testing.expectError(error.ZeroDimension, denseGemmCpuRef(.bf16, &x, &[_]u16{0}, &tiny, 0, 1, 1));
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
