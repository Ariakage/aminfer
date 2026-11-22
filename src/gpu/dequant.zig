//! GPU fused 3-bit and 4-bit affine dequantization GEMV and GEMM (G1, G2),
//! plus native unquantized FP16 / BF16 dense GEMV and GEMM (G12, ADR-0016).
//! Pipelines are compiled once at startup using Metal function constants
//! (`Q_BITS`, `Q_GROUP_SIZE`, `DENSE_W_BF16`) matching `src/quant/spec.zig`.

const std = @import("std");
const metal = @import("../bridge/metal.zig");
const page = @import("../mem/page.zig");
const arena_mod = @import("../mem/arena.zig");
const budget_mod = @import("../mem/budget.zig");
const quant = @import("../quant/spec.zig");
const amf = @import("../format/amf.zig");
const vmstat = @import("../platform/vmstat.zig");

/// Prefill matmul backend selection hook (G16, ADR-0020).
/// `.nax_apple10_reserved` reserves the dispatch interface for future M5+
/// `Apple10` Neural Accelerator (`mpp::tensor_ops::matmul2d`) shaders while
/// staying on verified `Apple9` `simdgroup_matrix` kernels until `Apple10`
/// hardware validation completes.
pub const PrefillMatmulBackend = enum(u8) {
    simdgroup_apple9 = 0,
    nax_apple10_reserved = 1,
};

pub fn selectPrefillMatmulBackend(info: metal.DeviceInfo) PrefillMatmulBackend {
    _ = info.supportsApple10();
    return .simdgroup_apple9;
}

pub const CANARY_TOL_F16: f32 = 2e-3;
pub const CANARY_TOL_BF16: f32 = 2e-2;
pub const CANARY_TOL_Q4: f32 = 2e-2;

pub const CanaryError = metal.Error || page.ShareError || page.WiredRegion.InitError || quant.ShapeError || vmstat.Error || error{
    BufferTooSmall,
    MetallibNumericalCanaryFailed,
};

pub const CanaryResult = struct {
    elapsed_us: f64,
    max_abs_err_f16: f32,
    max_abs_err_bf16: f32,
    max_abs_err_q4: f32,
};

/// Verifies `f16`, `bf16`, and `q4_affine` GPU canary outputs against `fp32`
/// references for finiteness and tight numerical tolerances (`ADR-0020` G16).
pub fn verifyCanarySlices(
    got_f16: []const f16,
    ref_f16: []const f32,
    got_bf16: []const f16,
    ref_bf16: []const f32,
    got_q4: []const f16,
    ref_q4: []const f32,
) error{MetallibNumericalCanaryFailed}!struct {
    max_abs_err_f16: f32,
    max_abs_err_bf16: f32,
    max_abs_err_q4: f32,
} {
    if (got_f16.len == 0 or got_f16.len != ref_f16.len or
        got_bf16.len == 0 or got_bf16.len != ref_bf16.len or
        got_q4.len == 0 or got_q4.len != ref_q4.len)
    {
        return error.MetallibNumericalCanaryFailed;
    }

    var max_f16: f32 = 0.0;
    for (got_f16, ref_f16) |g_h, r| {
        const g: f32 = @floatCast(g_h);
        if (std.math.isNan(g) or std.math.isInf(g) or std.math.isNan(r) or std.math.isInf(r)) {
            return error.MetallibNumericalCanaryFailed;
        }
        const err = @abs(g - r);
        if (err > CANARY_TOL_F16) return error.MetallibNumericalCanaryFailed;
        if (err > max_f16) max_f16 = err;
    }

    var max_bf16: f32 = 0.0;
    for (got_bf16, ref_bf16) |g_h, r| {
        const g: f32 = @floatCast(g_h);
        if (std.math.isNan(g) or std.math.isInf(g) or std.math.isNan(r) or std.math.isInf(r)) {
            return error.MetallibNumericalCanaryFailed;
        }
        const err = @abs(g - r);
        if (err > CANARY_TOL_BF16) return error.MetallibNumericalCanaryFailed;
        if (err > max_bf16) max_bf16 = err;
    }

    var max_q4: f32 = 0.0;
    for (got_q4, ref_q4) |g_h, r| {
        const g: f32 = @floatCast(g_h);
        if (std.math.isNan(g) or std.math.isInf(g) or std.math.isNan(r) or std.math.isInf(r)) {
            return error.MetallibNumericalCanaryFailed;
        }
        const err = @abs(g - r);
        if (err > CANARY_TOL_Q4) return error.MetallibNumericalCanaryFailed;
        if (err > max_q4) max_q4 = err;
    }

    return .{
        .max_abs_err_f16 = max_f16,
        .max_abs_err_bf16 = max_bf16,
        .max_abs_err_q4 = max_q4,
    };
}

/// Uniform payload matching `struct DequantParams` in `dequant_gemm.metal`.
pub const DequantParams = extern struct {
    m: u32,
    n: u32,
    k: u32,
    num_groups: u32,
    row_bytes: u32,
};

pub const GemvArgs = struct {
    x: metal.BufferBinding,
    qweight: metal.BufferBinding,
    scale_bias: metal.BufferBinding,
    y: metal.BufferBinding,
    n: u32,
    k: u32,
};

pub const GemmArgs = struct {
    x: metal.BufferBinding,
    qweight: metal.BufferBinding,
    scale_bias: metal.BufferBinding,
    y: metal.BufferBinding,
    m: u32,
    n: u32,
    k: u32,
    /// When `true` (default), `encodeGemm` dispatches the `64x64` wide prefill
    /// kernel once `m >= 64`. Set to `false` to force the legacy `32x32` path
    /// (e.g. `--gpu-prefill-mode tree64` A/B comparison).
    wide_tile: bool = true,
};

/// Uniform payload matching `struct DenseParams` in `dequant_gemm.metal`.
pub const DenseParams = extern struct {
    m: u32,
    n: u32,
    k: u32,
    _pad: u32 = 0,
};

comptime {
    if (@sizeOf(DequantParams) != 20) @compileError("DequantParams must be 20 bytes");
    if (@sizeOf(DenseParams) != 16) @compileError("DenseParams must be 16 bytes");
}

/// Dense FP16 / BF16 GEMV: `x` is `f16 [k]`, `weight` is `f16` or `bf16`
/// row-major `[n, k]`, `y` is `f16 [n]`. Bindings must be at least 8-byte
/// aligned (`.amf` blobs are 16 KiB aligned) for the `K % 4 == 0` fast path.
pub const DenseGemvArgs = struct {
    x: metal.BufferBinding,
    weight: metal.BufferBinding,
    y: metal.BufferBinding,
    n: u32,
    k: u32,
};

/// Dense FP16 / BF16 GEMM: `x` is `f16 [m, k]`, `weight` is `f16` or `bf16`
/// row-major `[n, k]`, `y` is `f16 [m, n]`.
pub const DenseGemmArgs = struct {
    x: metal.BufferBinding,
    weight: metal.BufferBinding,
    y: metal.BufferBinding,
    m: u32,
    n: u32,
    k: u32,
    /// When `true` (default), `encodeDenseGemm` dispatches the `64x64` wide
    /// prefill kernel once `m >= 64`. Set to `false` to force legacy `32x32`.
    wide_tile: bool = true,
};

pub const EncodeError = metal.Error || quant.ShapeError;

/// Startup-compiled specialized Metal 4 pipelines for 2-bit, 3-bit and 4-bit
/// group-64 affine dequantization GEMV and GEMM (G1, G2, G13).
pub const DequantPipelines = struct {
    gemv_q4: metal.Pipeline,
    gemv_q3: metal.Pipeline,
    gemv_q2: metal.Pipeline,
    gemm_q4: metal.Pipeline,
    gemm_q3: metal.Pipeline,
    gemm_q2: metal.Pipeline,
    gemm_wide_q4: metal.Pipeline,
    gemm_wide_q3: metal.Pipeline,
    gemm_wide_q2: metal.Pipeline,
    gemv_batch_q4: metal.Pipeline,
    gemv_batch_q3: metal.Pipeline,
    gemv_batch_q2: metal.Pipeline,
    gemv_f16: metal.Pipeline,
    gemv_bf16: metal.Pipeline,
    gemm_f16: metal.Pipeline,
    gemm_bf16: metal.Pipeline,
    gemm_wide_f16: metal.Pipeline,
    gemm_wide_bf16: metal.Pipeline,
    gemv_batch_f16: metal.Pipeline,
    gemv_batch_bf16: metal.Pipeline,

    pub fn init(device: metal.Device, library: metal.Library) metal.Error!DequantPipelines {
        const gemv_q4 = try metal.Pipeline.initSpecialized(device, library, "ami_dequant_gemv", &.{
            .{ .index = 0, .value = .{ .u32 = 4 } },
            .{ .index = 1, .value = .{ .u32 = 64 } },
        });
        errdefer gemv_q4.close();

        const gemv_q3 = try metal.Pipeline.initSpecialized(device, library, "ami_dequant_gemv", &.{
            .{ .index = 0, .value = .{ .u32 = 3 } },
            .{ .index = 1, .value = .{ .u32 = 64 } },
        });
        errdefer gemv_q3.close();

        const gemv_q2 = try metal.Pipeline.initSpecialized(device, library, "ami_dequant_gemv", &.{
            .{ .index = 0, .value = .{ .u32 = 2 } },
            .{ .index = 1, .value = .{ .u32 = 64 } },
        });
        errdefer gemv_q2.close();

        const gemm_q4 = try metal.Pipeline.initSpecialized(device, library, "ami_dequant_gemm", &.{
            .{ .index = 0, .value = .{ .u32 = 4 } },
            .{ .index = 1, .value = .{ .u32 = 64 } },
        });
        errdefer gemm_q4.close();

        const gemm_q3 = try metal.Pipeline.initSpecialized(device, library, "ami_dequant_gemm", &.{
            .{ .index = 0, .value = .{ .u32 = 3 } },
            .{ .index = 1, .value = .{ .u32 = 64 } },
        });
        errdefer gemm_q3.close();

        const gemm_q2 = try metal.Pipeline.initSpecialized(device, library, "ami_dequant_gemm", &.{
            .{ .index = 0, .value = .{ .u32 = 2 } },
            .{ .index = 1, .value = .{ .u32 = 64 } },
        });
        errdefer gemm_q2.close();

        const gemm_wide_q4 = try metal.Pipeline.initSpecialized(device, library, "ami_dequant_gemm_wide", &.{
            .{ .index = 0, .value = .{ .u32 = 4 } },
            .{ .index = 1, .value = .{ .u32 = 64 } },
        });
        errdefer gemm_wide_q4.close();

        const gemm_wide_q3 = try metal.Pipeline.initSpecialized(device, library, "ami_dequant_gemm_wide", &.{
            .{ .index = 0, .value = .{ .u32 = 3 } },
            .{ .index = 1, .value = .{ .u32 = 64 } },
        });
        errdefer gemm_wide_q3.close();

        const gemm_wide_q2 = try metal.Pipeline.initSpecialized(device, library, "ami_dequant_gemm_wide", &.{
            .{ .index = 0, .value = .{ .u32 = 2 } },
            .{ .index = 1, .value = .{ .u32 = 64 } },
        });
        errdefer gemm_wide_q2.close();

        const gemv_batch_q4 = try metal.Pipeline.initSpecialized(device, library, "ami_dequant_gemv_batch", &.{
            .{ .index = 0, .value = .{ .u32 = 4 } },
            .{ .index = 1, .value = .{ .u32 = 64 } },
        });
        errdefer gemv_batch_q4.close();

        const gemv_batch_q3 = try metal.Pipeline.initSpecialized(device, library, "ami_dequant_gemv_batch", &.{
            .{ .index = 0, .value = .{ .u32 = 3 } },
            .{ .index = 1, .value = .{ .u32 = 64 } },
        });
        errdefer gemv_batch_q3.close();

        const gemv_batch_q2 = try metal.Pipeline.initSpecialized(device, library, "ami_dequant_gemv_batch", &.{
            .{ .index = 0, .value = .{ .u32 = 2 } },
            .{ .index = 1, .value = .{ .u32 = 64 } },
        });
        errdefer gemv_batch_q2.close();

        const gemv_f16 = try metal.Pipeline.initSpecialized(device, library, "ami_dense_gemv", &.{
            .{ .index = 2, .value = .{ .bool = false } },
        });
        errdefer gemv_f16.close();

        const gemv_bf16 = try metal.Pipeline.initSpecialized(device, library, "ami_dense_gemv", &.{
            .{ .index = 2, .value = .{ .bool = true } },
        });
        errdefer gemv_bf16.close();

        const gemm_f16 = try metal.Pipeline.initSpecialized(device, library, "ami_dense_gemm", &.{
            .{ .index = 2, .value = .{ .bool = false } },
        });
        errdefer gemm_f16.close();

        const gemm_bf16 = try metal.Pipeline.initSpecialized(device, library, "ami_dense_gemm", &.{
            .{ .index = 2, .value = .{ .bool = true } },
        });
        errdefer gemm_bf16.close();

        const gemm_wide_f16 = try metal.Pipeline.initSpecialized(device, library, "ami_dense_gemm_wide", &.{
            .{ .index = 2, .value = .{ .bool = false } },
        });
        errdefer gemm_wide_f16.close();

        const gemm_wide_bf16 = try metal.Pipeline.initSpecialized(device, library, "ami_dense_gemm_wide", &.{
            .{ .index = 2, .value = .{ .bool = true } },
        });
        errdefer gemm_wide_bf16.close();

        const gemv_batch_f16 = try metal.Pipeline.initSpecialized(device, library, "ami_dense_gemv_batch", &.{
            .{ .index = 2, .value = .{ .bool = false } },
        });
        errdefer gemv_batch_f16.close();

        const gemv_batch_bf16 = try metal.Pipeline.initSpecialized(device, library, "ami_dense_gemv_batch", &.{
            .{ .index = 2, .value = .{ .bool = true } },
        });
        errdefer gemv_batch_bf16.close();

        return .{
            .gemv_q4 = gemv_q4,
            .gemv_q3 = gemv_q3,
            .gemv_q2 = gemv_q2,
            .gemm_q4 = gemm_q4,
            .gemm_q3 = gemm_q3,
            .gemm_q2 = gemm_q2,
            .gemm_wide_q4 = gemm_wide_q4,
            .gemm_wide_q3 = gemm_wide_q3,
            .gemm_wide_q2 = gemm_wide_q2,
            .gemv_batch_q4 = gemv_batch_q4,
            .gemv_batch_q3 = gemv_batch_q3,
            .gemv_batch_q2 = gemv_batch_q2,
            .gemv_f16 = gemv_f16,
            .gemv_bf16 = gemv_bf16,
            .gemm_f16 = gemm_f16,
            .gemm_bf16 = gemm_bf16,
            .gemm_wide_f16 = gemm_wide_f16,
            .gemm_wide_bf16 = gemm_wide_bf16,
            .gemv_batch_f16 = gemv_batch_f16,
            .gemv_batch_bf16 = gemv_batch_bf16,
        };
    }

    pub fn close(self: DequantPipelines) void {
        self.gemv_batch_bf16.close();
        self.gemv_batch_f16.close();
        self.gemm_wide_bf16.close();
        self.gemm_wide_f16.close();
        self.gemm_bf16.close();
        self.gemm_f16.close();
        self.gemv_bf16.close();
        self.gemv_f16.close();
        self.gemv_batch_q2.close();
        self.gemv_batch_q3.close();
        self.gemv_batch_q4.close();
        self.gemm_wide_q2.close();
        self.gemm_wide_q3.close();
        self.gemm_wide_q4.close();
        self.gemm_q2.close();
        self.gemm_q3.close();
        self.gemm_q4.close();
        self.gemv_q2.close();
        self.gemv_q3.close();
        self.gemv_q4.close();
    }

    pub fn encodeGemv(
        self: DequantPipelines,
        frame: *metal.Compute.Frame,
        bits: quant.QuantBits,
        args: GemvArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.n == 0) return error.ZeroDimension;
        const num_groups, const row_bytes, const pipe = switch (bits) {
            .q4 => blk: {
                const Ops = quant.QuantOps(.{ .bits = .q4, .group_size = 64 });
                break :blk .{ try Ops.groupsPerRow(args.k), try Ops.rowWeightBytes(args.k), self.gemv_q4 };
            },
            .q3 => blk: {
                const Ops = quant.QuantOps(.{ .bits = .q3, .group_size = 64 });
                break :blk .{ try Ops.groupsPerRow(args.k), try Ops.rowWeightBytes(args.k), self.gemv_q3 };
            },
            .q2 => blk: {
                const Ops = quant.QuantOps(.{ .bits = .q2, .group_size = 64 });
                break :blk .{ try Ops.groupsPerRow(args.k), try Ops.rowWeightBytes(args.k), self.gemv_q2 };
            },
        };
        const params: DequantParams = .{
            .m = 1,
            .n = args.n,
            .k = args.k,
            .num_groups = @intCast(num_groups),
            .row_bytes = @intCast(row_bytes),
        };
        const groups_x: u64 = (@as(u64, args.n) + 15) / 16;
        try frame.encodeThreadgroups(
            pipe,
            &.{ args.x, args.qweight, args.scale_bias, args.y },
            std.mem.asBytes(&params),
            .{ groups_x, 1, 1 },
            .{ 128, 1, 1 },
            insert_barrier_before,
        );
    }

    pub fn encodeGemm(
        self: DequantPipelines,
        frame: *metal.Compute.Frame,
        bits: quant.QuantBits,
        args: GemmArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.wide_tile and args.m >= 64) {
            return self.encodePrefillGemm(frame, bits, args, insert_barrier_before);
        }
        return self.encodeGemm32(frame, bits, args, insert_barrier_before);
    }

    fn encodeGemm32(
        self: DequantPipelines,
        frame: *metal.Compute.Frame,
        bits: quant.QuantBits,
        args: GemmArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.m == 0 or args.n == 0) return error.ZeroDimension;
        const num_groups, const row_bytes, const pipe = switch (bits) {
            .q4 => blk: {
                const Ops = quant.QuantOps(.{ .bits = .q4, .group_size = 64 });
                break :blk .{ try Ops.groupsPerRow(args.k), try Ops.rowWeightBytes(args.k), self.gemm_q4 };
            },
            .q3 => blk: {
                const Ops = quant.QuantOps(.{ .bits = .q3, .group_size = 64 });
                break :blk .{ try Ops.groupsPerRow(args.k), try Ops.rowWeightBytes(args.k), self.gemm_q3 };
            },
            .q2 => blk: {
                const Ops = quant.QuantOps(.{ .bits = .q2, .group_size = 64 });
                break :blk .{ try Ops.groupsPerRow(args.k), try Ops.rowWeightBytes(args.k), self.gemm_q2 };
            },
        };
        const params: DequantParams = .{
            .m = args.m,
            .n = args.n,
            .k = args.k,
            .num_groups = @intCast(num_groups),
            .row_bytes = @intCast(row_bytes),
        };
        const groups_x: u64 = (@as(u64, args.n) + 31) / 32;
        const groups_y: u64 = (@as(u64, args.m) + 31) / 32;
        try frame.encodeThreadgroups(
            pipe,
            &.{ args.x, args.qweight, args.scale_bias, args.y },
            std.mem.asBytes(&params),
            .{ groups_x, groups_y, 1 },
            .{ 128, 1, 1 },
            insert_barrier_before,
        );
    }

    /// Encodes a wide-tile (`BM = 64, BN = 64`, 8 simdgroups / 256 threads)
    /// quantized prefill GEMM (`q2`/`q3`/`q4` group-64 affine) for any `M >= 1`
    /// (`G15`, ADR-0020). Set `args.wide_tile = false` to fall back to `32x32`.
    pub fn encodePrefillGemm(
        self: DequantPipelines,
        frame: *metal.Compute.Frame,
        bits: quant.QuantBits,
        args: GemmArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (!args.wide_tile) {
            return self.encodeGemm32(frame, bits, args, insert_barrier_before);
        }
        if (args.m == 0 or args.n == 0) return error.ZeroDimension;
        const num_groups, const row_bytes, const pipe = switch (bits) {
            .q4 => blk: {
                const Ops = quant.QuantOps(.{ .bits = .q4, .group_size = 64 });
                break :blk .{ try Ops.groupsPerRow(args.k), try Ops.rowWeightBytes(args.k), self.gemm_wide_q4 };
            },
            .q3 => blk: {
                const Ops = quant.QuantOps(.{ .bits = .q3, .group_size = 64 });
                break :blk .{ try Ops.groupsPerRow(args.k), try Ops.rowWeightBytes(args.k), self.gemm_wide_q3 };
            },
            .q2 => blk: {
                const Ops = quant.QuantOps(.{ .bits = .q2, .group_size = 64 });
                break :blk .{ try Ops.groupsPerRow(args.k), try Ops.rowWeightBytes(args.k), self.gemm_wide_q2 };
            },
        };
        const params: DequantParams = .{
            .m = args.m,
            .n = args.n,
            .k = args.k,
            .num_groups = @intCast(num_groups),
            .row_bytes = @intCast(row_bytes),
        };
        const groups_x: u64 = (@as(u64, args.n) + 63) / 64;
        const groups_y: u64 = (@as(u64, args.m) + 63) / 64;
        try frame.encodeThreadgroups(
            pipe,
            &.{ args.x, args.qweight, args.scale_bias, args.y },
            std.mem.asBytes(&params),
            .{ groups_x, groups_y, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
    }

    /// Encodes a native FP16 / BF16 dense GEMV (`M = 1`), e.g. attention / FFN
    /// projections and the `lm_head` (`N = 151936`). Zero heap allocation.
    pub fn encodeDenseGemv(
        self: DequantPipelines,
        frame: *metal.Compute.Frame,
        dtype: quant.FloatDType,
        args: DenseGemvArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.n == 0 or args.k == 0) return error.ZeroDimension;
        const pipe = switch (dtype) {
            .f16 => self.gemv_f16,
            .bf16 => self.gemv_bf16,
        };
        const params: DenseParams = .{ .m = 1, .n = args.n, .k = args.k };
        const groups_x: u64 = (@as(u64, args.n) + 15) / 16;
        try frame.encodeThreadgroups(
            pipe,
            &.{ args.x, args.weight, args.y },
            std.mem.asBytes(&params),
            .{ groups_x, 1, 1 },
            .{ 128, 1, 1 },
            insert_barrier_before,
        );
    }

    /// Encodes a weight-stationary batched 3/4-bit dequant GEMV (`1 <= m <= 64`)
    /// matching `encodeGemv` bit-for-bit in FP32 accumulation order while reading
    /// each weight row once per 8-token block. Zero heap allocation.
    pub fn encodeGemvBatch(
        self: DequantPipelines,
        frame: *metal.Compute.Frame,
        bits: quant.QuantBits,
        args: GemmArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.m == 0 or args.n == 0) return error.ZeroDimension;
        const num_groups, const row_bytes, const pipe = switch (bits) {
            .q4 => blk: {
                const Ops = quant.QuantOps(.{ .bits = .q4, .group_size = 64 });
                break :blk .{ try Ops.groupsPerRow(args.k), try Ops.rowWeightBytes(args.k), self.gemv_batch_q4 };
            },
            .q3 => blk: {
                const Ops = quant.QuantOps(.{ .bits = .q3, .group_size = 64 });
                break :blk .{ try Ops.groupsPerRow(args.k), try Ops.rowWeightBytes(args.k), self.gemv_batch_q3 };
            },
            .q2 => blk: {
                const Ops = quant.QuantOps(.{ .bits = .q2, .group_size = 64 });
                break :blk .{ try Ops.groupsPerRow(args.k), try Ops.rowWeightBytes(args.k), self.gemv_batch_q2 };
            },
        };
        const params: DequantParams = .{
            .m = args.m,
            .n = args.n,
            .k = args.k,
            .num_groups = @intCast(num_groups),
            .row_bytes = @intCast(row_bytes),
        };
        const groups_x: u64 = (@as(u64, args.n) + 7) / 8;
        const groups_y: u64 = (@as(u64, args.m) + 7) / 8;
        try frame.encodeThreadgroups(
            pipe,
            &.{ args.x, args.qweight, args.scale_bias, args.y },
            std.mem.asBytes(&params),
            .{ groups_x, groups_y, 1 },
            .{ 128, 1, 1 },
            insert_barrier_before,
        );
    }

    /// Encodes a weight-stationary batched FP16 / BF16 dense GEMV (`1 <= m <= 64`)
    /// matching `encodeDenseGemv` bit-for-bit (no `bfloat` activation truncation)
    /// while reading each weight row once per 8-token block. Zero heap allocation.
    pub fn encodeDenseGemvBatch(
        self: DequantPipelines,
        frame: *metal.Compute.Frame,
        dtype: quant.FloatDType,
        args: DenseGemmArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.m == 0 or args.n == 0 or args.k == 0) return error.ZeroDimension;
        const pipe = switch (dtype) {
            .f16 => self.gemv_batch_f16,
            .bf16 => self.gemv_batch_bf16,
        };
        const params: DenseParams = .{ .m = args.m, .n = args.n, .k = args.k };
        const groups_x: u64 = (@as(u64, args.n) + 7) / 8;
        const groups_y: u64 = (@as(u64, args.m) + 7) / 8;
        try frame.encodeThreadgroups(
            pipe,
            &.{ args.x, args.weight, args.y },
            std.mem.asBytes(&params),
            .{ groups_x, groups_y, 1 },
            .{ 128, 1, 1 },
            insert_barrier_before,
        );
    }

    /// Encodes a native FP16 / BF16 dense GEMM (`M >= 1`) with FP32
    /// `simdgroup_float8x8` accumulation. Automatically uses the `64x64` wide
    /// prefill kernel when `args.wide_tile and args.m >= 64`. Zero heap allocation.
    pub fn encodeDenseGemm(
        self: DequantPipelines,
        frame: *metal.Compute.Frame,
        dtype: quant.FloatDType,
        args: DenseGemmArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.wide_tile and args.m >= 64) {
            return self.encodePrefillDenseGemm(frame, dtype, args, insert_barrier_before);
        }
        return self.encodeDenseGemm32(frame, dtype, args, insert_barrier_before);
    }

    fn encodeDenseGemm32(
        self: DequantPipelines,
        frame: *metal.Compute.Frame,
        dtype: quant.FloatDType,
        args: DenseGemmArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.m == 0 or args.n == 0 or args.k == 0) return error.ZeroDimension;
        const pipe = switch (dtype) {
            .f16 => self.gemm_f16,
            .bf16 => self.gemm_bf16,
        };
        const params: DenseParams = .{ .m = args.m, .n = args.n, .k = args.k };
        const groups_x: u64 = (@as(u64, args.n) + 31) / 32;
        const groups_y: u64 = (@as(u64, args.m) + 31) / 32;
        try frame.encodeThreadgroups(
            pipe,
            &.{ args.x, args.weight, args.y },
            std.mem.asBytes(&params),
            .{ groups_x, groups_y, 1 },
            .{ 128, 1, 1 },
            insert_barrier_before,
        );
    }

    /// Encodes a wide-tile (`BM = 64, BN = 64`, 8 simdgroups / 256 threads)
    /// dense FP16 / BF16 prefill GEMM for any `M >= 1` (`G15`, ADR-0020).
    /// Set `args.wide_tile = false` to fall back to `32x32`.
    pub fn encodePrefillDenseGemm(
        self: DequantPipelines,
        frame: *metal.Compute.Frame,
        dtype: quant.FloatDType,
        args: DenseGemmArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (!args.wide_tile) {
            return self.encodeDenseGemm32(frame, dtype, args, insert_barrier_before);
        }
        if (args.m == 0 or args.n == 0 or args.k == 0) return error.ZeroDimension;
        const pipe = switch (dtype) {
            .f16 => self.gemm_wide_f16,
            .bf16 => self.gemm_wide_bf16,
        };
        const params: DenseParams = .{ .m = args.m, .n = args.n, .k = args.k };
        const groups_x: u64 = (@as(u64, args.n) + 63) / 64;
        const groups_y: u64 = (@as(u64, args.m) + 63) / 64;
        try frame.encodeThreadgroups(
            pipe,
            &.{ args.x, args.weight, args.y },
            std.mem.asBytes(&params),
            .{ groups_x, groups_y, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
    }

    /// Runs the startup `.metallib` GPU numerical canary (`G16`, ADR-0020) over
    /// a temporary `16 KiB` wired page in one Metal 4 command buffer, verifying
    /// `f16`, `bf16` (including exponent paths exceeding `f16` dynamic range),
    /// and `q4_affine` across both `32x32` and `64x64` wide GEMM kernels.
    /// Note: a fresh `MTL4CommandQueue` / `MTLResidencySet` first submit takes
    /// ~1–4 ms cold wall time on Apple Silicon (including initial driver/AGX
    /// wake); see `runStartupNumericalCanaryProfile` for both cold first-submit
    /// and warm in-buffer GPU check latency.
    pub fn runStartupNumericalCanary(
        self: *const DequantPipelines,
        dev: *metal.Device,
    ) CanaryError!CanaryResult {
        const scratch = try page.WiredRegion.init(page.APPLE_PAGE_SIZE);
        defer scratch.deinit();

        const gpu_buf = try metal.Buffer.wrap(dev.*, scratch.bytes);
        defer gpu_buf.close();

        var compute = try metal.Compute.initWithFrames(dev.*, 1);
        defer compute.close();
        try compute.makeResident(&.{gpu_buf});

        return self.runStartupNumericalCanaryInBuffer(&compute, gpu_buf, scratch.bytes);
    }

    pub const CanaryProfile = struct {
        cold: CanaryResult,
        warm_elapsed_us: f64,
    };

    /// Runs the startup `.metallib` numerical canary twice over the same
    /// resident `16 KiB` wired buffer and `metal.Compute` queue, returning both
    /// the cold first-submit result (including initial Metal 4 queue/residency
    /// wake) and the warm in-buffer GPU check latency.
    pub fn runStartupNumericalCanaryProfile(
        self: *const DequantPipelines,
        dev: *metal.Device,
    ) CanaryError!CanaryProfile {
        const scratch = try page.WiredRegion.init(page.APPLE_PAGE_SIZE);
        defer scratch.deinit();

        const gpu_buf = try metal.Buffer.wrap(dev.*, scratch.bytes);
        defer gpu_buf.close();

        var compute = try metal.Compute.initWithFrames(dev.*, 1);
        defer compute.close();
        try compute.makeResident(&.{gpu_buf});

        const cold = try self.runStartupNumericalCanaryInBuffer(&compute, gpu_buf, scratch.bytes);
        const warm = try self.runStartupNumericalCanaryInBuffer(&compute, gpu_buf, scratch.bytes);
        return .{
            .cold = cold,
            .warm_elapsed_us = warm.elapsed_us,
        };
    }

    /// Runs the startup numerical canary reusing an existing `metal.Compute`
    /// context and a caller-supplied, already-resident `16 KiB`-aligned wired
    /// slice (minimum 16 KiB). Zero heap allocations.
    pub fn runStartupNumericalCanaryInBuffer(
        self: *const DequantPipelines,
        compute: *metal.Compute,
        gpu_buf: metal.Buffer,
        wired_bytes: []align(page.APPLE_PAGE_SIZE) u8,
    ) CanaryError!CanaryResult {
        if (wired_bytes.len < page.APPLE_PAGE_SIZE) return error.InvalidArgument;

        const cm: usize = 16;
        const cn: usize = 16;
        const ck: usize = 64;
        const OpsQ4 = quant.QuantOps(.{ .bits = .q4, .group_size = 64 });

        const x_off: usize = 0; // 16 * 64 * 2 = 2048 B
        const w_f16_off: usize = 2048; // 2048 B
        const w_bf16_off: usize = 4096; // 2048 B
        const w_q4_off: usize = 6144; // 16 * 32 = 512 B
        const sb_q4_off: usize = 6656; // 16 * 4 = 64 B
        const y_f16_off: usize = 6720; // 16 * 16 * 2 = 512 B
        const y_bf16_off: usize = 7232; // 512 B
        const y_q4_off: usize = 7744; // 512 B
        const y_f16_wide_off: usize = 8256; // 512 B
        const y_bf16_wide_off: usize = 8768; // 512 B
        const y_q4_wide_off: usize = 9280; // 512 B
        comptime {
            std.debug.assert(y_q4_wide_off + 512 <= page.APPLE_PAGE_SIZE);
        }

        @memset(wired_bytes[0 .. y_q4_wide_off + 512], 0);

        const x_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, wired_bytes[x_off .. x_off + cm * ck * 2]));
        const w_f16: []u16 = @alignCast(std.mem.bytesAsSlice(u16, wired_bytes[w_f16_off .. w_f16_off + cn * ck * 2]));
        const w_bf16: []u16 = @alignCast(std.mem.bytesAsSlice(u16, wired_bytes[w_bf16_off .. w_bf16_off + cn * ck * 2]));
        const w_q4: []u8 = wired_bytes[w_q4_off .. w_q4_off + cn * 32];
        const sb_q4: []quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, wired_bytes[sb_q4_off .. sb_q4_off + cn * 4]));

        const y_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, wired_bytes[y_f16_off .. y_f16_off + cm * cn * 2]));
        const y_bf16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, wired_bytes[y_bf16_off .. y_bf16_off + cm * cn * 2]));
        const y_q4: []f16 = @alignCast(std.mem.bytesAsSlice(f16, wired_bytes[y_q4_off .. y_q4_off + cm * cn * 2]));
        const y_f16_wide: []f16 = @alignCast(std.mem.bytesAsSlice(f16, wired_bytes[y_f16_wide_off .. y_f16_wide_off + cm * cn * 2]));
        const y_bf16_wide: []f16 = @alignCast(std.mem.bytesAsSlice(f16, wired_bytes[y_bf16_wide_off .. y_bf16_wide_off + cm * cn * 2]));
        const y_q4_wide: []f16 = @alignCast(std.mem.bytesAsSlice(f16, wired_bytes[y_q4_wide_off .. y_q4_wide_off + cm * cn * 2]));

        // Populate deterministic signed fractional inputs/weights exercising
        // f16, bf16 exponent+mantissa paths (including column-0 > f16 max paired
        // with 2^-14 activation), and q4_affine group-64 packing.
        for (0..cm) |mi| {
            for (0..ck) |ki| {
                const idx = mi * ck + ki;
                if (ki == 0) {
                    const sign: f32 = if ((mi & 1) == 0) 1.0 else -1.0;
                    // 2^-14 is exact normal in both f16 and bf16.
                    x_f16[idx] = @floatCast(sign * 0.00006103515625);
                } else {
                    const centered: i32 = @as(i32, @intCast((mi * 13 + ki * 7 + 5) % 15)) - 7;
                    x_f16[idx] = @floatCast(@as(f32, @floatFromInt(centered)) * 0.0625);
                }
            }
        }
        var codes: [64]u8 = undefined;
        for (0..cn) |ni| {
            for (0..ck) |ki| {
                const idx = ni * ck + ki;
                if (ki == 0) {
                    const sign_f: f32 = if ((ni & 1) == 0) 1.0 else -1.0;
                    // 73728.0 (= 9 * 2^13) is exact in bf16 and exceeds f16 max (65504),
                    // catching any compiler/driver path that truncates bf16 through f16.
                    w_bf16[idx] = quant.f32ToFloatBits(.bf16, sign_f * (73728.0 + @as(f32, @floatFromInt(ni % 4)) * 8192.0));
                    w_f16[idx] = quant.f32ToFloatBits(.f16, sign_f * 0.375);
                } else {
                    const centered: i32 = @as(i32, @intCast((ni * 19 + ki * 11 + 3) % 17)) - 8;
                    const val: f32 = @as(f32, @floatFromInt(centered)) * 0.03125;
                    w_f16[idx] = quant.f32ToFloatBits(.f16, val);
                    w_bf16[idx] = quant.f32ToFloatBits(.bf16, val * 1.25);
                }
                codes[ki] = @intCast((ni * 5 + ki * 3 + 1) & 0xF);
            }
            OpsQ4.packGroup(&codes, w_q4[ni * 32 ..][0..32]);
            const sc: f32 = 0.03125 + 0.0078125 * @as(f32, @floatFromInt(ni % 4));
            const bi: f32 = -0.25 + 0.0625 * @as(f32, @floatFromInt(ni % 3));
            sb_q4[ni] = .{ .scale = @floatCast(sc), .bias = @floatCast(bi) };
        }

        var ref_f16: [cm * cn]f32 = undefined;
        var ref_bf16: [cm * cn]f32 = undefined;
        var ref_q4: [cm * cn]f32 = undefined;
        try quant.denseGemmCpuRef(.f16, x_f16, w_f16, &ref_f16, cm, cn, ck);
        try quant.denseGemmCpuRef(.bf16, x_f16, w_bf16, &ref_bf16, cm, cn, ck);
        try OpsQ4.matmulRef(x_f16, w_q4, sb_q4, &ref_q4, cm, cn, ck);

        const x_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = x_off };
        const w_f16_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = w_f16_off };
        const w_bf16_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = w_bf16_off };
        const w_q4_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = w_q4_off };
        const sb_q4_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = sb_q4_off };

        const t0 = try vmstat.nowNs();
        var frame = try compute.beginFrame(5000);
        errdefer frame.abort();

        // Dispatch all 6 micro-checks in ONE single Metal 4 command buffer
        // with zero intra-buffer barriers (read-only inputs, disjoint outputs).
        try self.encodeDenseGemm32(&frame, .f16, .{
            .x = x_b,
            .weight = w_f16_b,
            .y = .{ .buffer = gpu_buf, .offset = y_f16_off },
            .m = cm,
            .n = cn,
            .k = ck,
        }, false);
        try self.encodeDenseGemm32(&frame, .bf16, .{
            .x = x_b,
            .weight = w_bf16_b,
            .y = .{ .buffer = gpu_buf, .offset = y_bf16_off },
            .m = cm,
            .n = cn,
            .k = ck,
        }, false);
        try self.encodeGemm32(&frame, .q4, .{
            .x = x_b,
            .qweight = w_q4_b,
            .scale_bias = sb_q4_b,
            .y = .{ .buffer = gpu_buf, .offset = y_q4_off },
            .m = cm,
            .n = cn,
            .k = ck,
        }, false);
        try self.encodePrefillDenseGemm(&frame, .f16, .{
            .x = x_b,
            .weight = w_f16_b,
            .y = .{ .buffer = gpu_buf, .offset = y_f16_wide_off },
            .m = cm,
            .n = cn,
            .k = ck,
        }, false);
        try self.encodePrefillDenseGemm(&frame, .bf16, .{
            .x = x_b,
            .weight = w_bf16_b,
            .y = .{ .buffer = gpu_buf, .offset = y_bf16_wide_off },
            .m = cm,
            .n = cn,
            .k = ck,
        }, false);
        try self.encodePrefillGemm(&frame, .q4, .{
            .x = x_b,
            .qweight = w_q4_b,
            .scale_bias = sb_q4_b,
            .y = .{ .buffer = gpu_buf, .offset = y_q4_wide_off },
            .m = cm,
            .n = cn,
            .k = ck,
        }, false);

        const sig = try frame.commit();
        try compute.wait(sig, 5000);
        const t1 = try vmstat.nowNs();

        const err32 = try verifyCanarySlices(y_f16, &ref_f16, y_bf16, &ref_bf16, y_q4, &ref_q4);
        const err64 = try verifyCanarySlices(y_f16_wide, &ref_f16, y_bf16_wide, &ref_bf16, y_q4_wide, &ref_q4);

        return .{
            .elapsed_us = @as(f64, @floatFromInt(t1 - t0)) / 1000.0,
            .max_abs_err_f16 = @max(err32.max_abs_err_f16, err64.max_abs_err_f16),
            .max_abs_err_bf16 = @max(err32.max_abs_err_bf16, err64.max_abs_err_bf16),
            .max_abs_err_q4 = @max(err32.max_abs_err_q4, err64.max_abs_err_q4),
        };
    }
};

fn verifySpec(comptime spec: quant.QuantSpec, m: usize, n: usize, k: usize) !void {
    const Ops = quant.QuantOps(spec);
    const gpa = std.testing.allocator;

    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try DequantPipelines.init(device, library);
    defer pipes.close();

    const w_bytes = try Ops.packedWeightBytes(n, k);
    const sb_offset = amf.align64(w_bytes);
    const sb_bytes = try Ops.scaleBiasBytes(n, k);
    const x_bytes = m * k * @sizeOf(f16);
    const y_bytes = m * n * @sizeOf(f16);

    const arena_cap = try page.alignUp(sb_offset + sb_bytes + 256) +
        try page.alignUp(x_bytes + 256) +
        try page.alignUp(y_bytes + 256) +
        page.APPLE_PAGE_SIZE;
    const budget = budget_mod.Budget.compute(.{
        .memory_bytes = arena_cap + 4 * page.APPLE_PAGE_SIZE,
        .user_wire_limit = arena_cap,
        .global_user_wire_limit = arena_cap,
        .gpu_working_set_bytes = arena_cap,
        .configured_cap = null,
        .headroom_bytes = page.APPLE_PAGE_SIZE,
    });
    var arena = try arena_mod.WiredArena.init(arena_cap, budget);
    defer arena.deinit();

    const w_blob = try arena.allocPages(sb_offset + sb_bytes);
    const x_slice_raw = try arena.allocAligned(x_bytes, 64);
    const y_slice_raw = try arena.allocAligned(y_bytes, 64);
    @memset(w_blob, 0);
    @memset(y_slice_raw, 0);

    const qweight = w_blob[0..w_bytes];
    const scale_bias: []quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, w_blob[sb_offset .. sb_offset + sb_bytes]));
    const x_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, x_slice_raw));
    const y_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, y_slice_raw));

    // Deterministic synthetic weights and activations covering signed ranges.
    const groups = try Ops.groupsPerRow(k);
    const row_bytes = try Ops.rowWeightBytes(k);
    const q_mask: u8 = (@as(u8, 1) << @intCast(spec.bits.bitWidth())) - 1;
    var codes: [spec.group_size]u8 = undefined;
    for (0..n) |row| {
        for (0..groups) |g| {
            for (0..spec.group_size) |j| {
                const v = (row * 13 + g * 7 + j * 5 + 3) & q_mask;
                codes[j] = @intCast(v);
            }
            const g_off = row * row_bytes + g * Ops.groupWeightBytes();
            Ops.packGroup(&codes, qweight[g_off..][0..Ops.groupWeightBytes()]);
            const s: f32 = 0.05 + 0.01 * @as(f32, @floatFromInt((row + g) % 9));
            const b: f32 = -0.25 - 0.05 * @as(f32, @floatFromInt((row * 3 + g) % 5));
            scale_bias[row * groups + g] = .{ .scale = @floatCast(s), .bias = @floatCast(b) };
        }
    }
    for (x_f16, 0..) |*v, idx| {
        const centered: i32 = @as(i32, @intCast((idx * 17 + 11) % 31)) - 15;
        v.* = @floatCast(@as(f32, @floatFromInt(centered)) * 0.0625);
    }

    const ref = try gpa.alloc(f32, m * n);
    defer gpa.free(ref);
    try Ops.matmulRef(x_f16, qweight, scale_bias, ref, m, n, k);

    const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
    defer gpu_buf.close();

    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();
    try compute.makeResident(&.{gpu_buf});

    const w_off: u64 = arena.offsetOf(w_blob.ptr);
    const sb_off: u64 = w_off + sb_offset;
    const x_off: u64 = arena.offsetOf(x_slice_raw.ptr);
    const y_off: u64 = arena.offsetOf(y_slice_raw.ptr);

    var frame = try compute.beginFrame(5000);
    errdefer frame.abort();
    if (m == 1) {
        try pipes.encodeGemv(&frame, spec.bits, .{
            .x = .{ .buffer = gpu_buf, .offset = x_off },
            .qweight = .{ .buffer = gpu_buf, .offset = w_off },
            .scale_bias = .{ .buffer = gpu_buf, .offset = sb_off },
            .y = .{ .buffer = gpu_buf, .offset = y_off },
            .n = @intCast(n),
            .k = @intCast(k),
        }, false);
    } else {
        try pipes.encodeGemm(&frame, spec.bits, .{
            .x = .{ .buffer = gpu_buf, .offset = x_off },
            .qweight = .{ .buffer = gpu_buf, .offset = w_off },
            .scale_bias = .{ .buffer = gpu_buf, .offset = sb_off },
            .y = .{ .buffer = gpu_buf, .offset = y_off },
            .m = @intCast(m),
            .n = @intCast(n),
            .k = @intCast(k),
        }, false);
    }
    const sig = try frame.commit();
    try compute.wait(sig, 5000);

    for (ref, 0..) |expected, idx| {
        const got: f32 = @floatCast(y_f16[idx]);
        // fp16 quantization error on output + intermediate half dequantization in GEMM.
        const tol: f32 = @max(0.02, @abs(expected) * 0.015);
        try std.testing.expectApproxEqAbs(expected, got, tol);
    }
}

test "4-bit fused dequant GEMV matches CPU fp32 reference on exact, odd and LLM shapes" {
    try verifySpec(.{ .bits = .q4, .group_size = 64 }, 1, 16, 64);
    try verifySpec(.{ .bits = .q4, .group_size = 64 }, 1, 37, 192);
    try verifySpec(.{ .bits = .q4, .group_size = 64 }, 1, 512, 1024);
}

test "3-bit fused dequant GEMV matches CPU fp32 reference on exact, odd and LLM shapes" {
    try verifySpec(.{ .bits = .q3, .group_size = 64 }, 1, 16, 64);
    try verifySpec(.{ .bits = .q3, .group_size = 64 }, 1, 37, 192);
    try verifySpec(.{ .bits = .q3, .group_size = 64 }, 1, 512, 1024);
}

test "2-bit fused dequant GEMV matches CPU fp32 reference on exact, odd and LLM shapes" {
    try verifySpec(.{ .bits = .q2, .group_size = 64 }, 1, 16, 64);
    try verifySpec(.{ .bits = .q2, .group_size = 64 }, 1, 37, 192);
    try verifySpec(.{ .bits = .q2, .group_size = 64 }, 1, 512, 1024);
}

test "4-bit fused dequant GEMM matches CPU fp32 reference across small, odd and tiled batches" {
    try verifySpec(.{ .bits = .q4, .group_size = 64 }, 7, 37, 192);
    try verifySpec(.{ .bits = .q4, .group_size = 64 }, 16, 64, 256);
    try verifySpec(.{ .bits = .q4, .group_size = 64 }, 37, 53, 192);
    try verifySpec(.{ .bits = .q4, .group_size = 64 }, 64, 128, 256);
}

test "3-bit fused dequant GEMM matches CPU fp32 reference across small, odd and tiled batches" {
    try verifySpec(.{ .bits = .q3, .group_size = 64 }, 7, 37, 192);
    try verifySpec(.{ .bits = .q3, .group_size = 64 }, 16, 64, 256);
    try verifySpec(.{ .bits = .q3, .group_size = 64 }, 37, 53, 192);
    try verifySpec(.{ .bits = .q3, .group_size = 64 }, 64, 128, 256);
}

test "2-bit fused dequant GEMM matches CPU fp32 reference across small, odd and tiled batches" {
    try verifySpec(.{ .bits = .q2, .group_size = 64 }, 7, 37, 192);
    try verifySpec(.{ .bits = .q2, .group_size = 64 }, 16, 64, 256);
    try verifySpec(.{ .bits = .q2, .group_size = 64 }, 37, 53, 192);
    try verifySpec(.{ .bits = .q2, .group_size = 64 }, 64, 128, 256);
}

test "end-to-end .amf blob loads into WiredArena and executes on GPU without copying" {
    const gpa = std.testing.allocator;
    const Q3 = quant.QuantOps(.{ .bits = .q3, .group_size = 64 });
    const n: u32 = 48;
    const k: u32 = 128;
    const w_bytes = try Q3.packedWeightBytes(n, k);
    const params_offset = amf.align64(w_bytes);
    const sb_bytes = try Q3.scaleBiasBytes(n, k);

    // Construct an in-memory .amf image containing a 3-bit projection blob.
    const dir_off: usize = page.APPLE_PAGE_SIZE;
    const dir_bytes: usize = page.APPLE_PAGE_SIZE;
    const blob_off: usize = dir_off + dir_bytes;
    const blob_bytes: usize = try page.alignUp(params_offset + sb_bytes);
    const image = try gpa.alloc(u8, blob_off + blob_bytes);
    defer gpa.free(image);
    @memset(image, 0);

    const blob_qweight = image[blob_off .. blob_off + w_bytes];
    const blob_sb: []quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(
        quant.ScaleBias,
        image[blob_off + params_offset .. blob_off + params_offset + sb_bytes],
    ));

    const groups = try Q3.groupsPerRow(k);
    const row_bytes = try Q3.rowWeightBytes(k);
    var codes: [64]u8 = undefined;
    for (0..n) |row| {
        for (0..groups) |g| {
            for (0..64) |j| codes[j] = @intCast((row * 5 + g * 3 + j) & 7);
            Q3.packGroup(&codes, blob_qweight[row * row_bytes + g * Q3.groupWeightBytes() ..][0..Q3.groupWeightBytes()]);
            blob_sb[row * groups + g] = .{ .scale = @floatCast(0.125), .bias = @floatCast(-0.5) };
        }
    }

    var entry = amf.TensorEntry.zeroed("layers.0.mlp.experts.0.gate_proj", .hot_expert, .q3_affine);
    entry.quant_bits = 3;
    entry.rank = 2;
    entry.group_size = 64;
    entry.layer = 0;
    entry.expert = 0;
    entry.shape = .{ n, k, 0, 0 };
    entry.blob_offset = blob_off;
    entry.blob_bytes = blob_bytes;
    entry.data_bytes = w_bytes;
    entry.params_offset = params_offset;
    entry.params_bytes = sb_bytes;
    entry.blob_crc32 = std.hash.Crc32.hash(image[blob_off .. blob_off + blob_bytes]);

    @memcpy(image[dir_off..][0..@sizeOf(amf.TensorEntry)], std.mem.asBytes(&entry));
    var hdr: amf.Header = .{
        .magic = amf.MAGIC,
        .version = amf.FORMAT_VERSION,
        .flags = 0,
        .page_size = page.APPLE_PAGE_SIZE,
        .checksum_algo = amf.CHECKSUM_CRC32_IEEE,
        .tensor_count = 1,
        .reserved0 = 0,
        .directory_offset = dir_off,
        .directory_bytes = dir_bytes,
        .directory_crc32 = std.hash.Crc32.hash(image[dir_off..][0..@sizeOf(amf.TensorEntry)]),
        .header_crc32 = 0,
        .model_id = @splat(0),
        .reserved1 = 0,
    };
    hdr.header_crc32 = amf.computeHeaderCrc32(&hdr);
    @memcpy(image[0..@sizeOf(amf.Header)], std.mem.asBytes(&hdr));

    var parsed = try amf.Parsed.parseBytes(gpa, image);
    defer parsed.deinit(gpa);
    const found = parsed.find("layers.0.mlp.experts.0.gate_proj") orelse return error.TestUnexpectedResult;

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

    const loaded_blob = try parsed.loadBlobFromBytes(image, found, &arena);
    const x_raw = try arena.allocAligned(k * @sizeOf(f16), 64);
    const y_raw = try arena.allocAligned(n * @sizeOf(f16), 64);
    const x_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, x_raw));
    const y_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, y_raw));
    for (x_f16, 0..) |*v, idx| v.* = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast(idx % 11)) - 5)) * 0.25);

    var ref: [n]f32 = undefined;
    try Q3.matmulRef(x_f16, blob_qweight, blob_sb, &ref, 1, n, k);

    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try DequantPipelines.init(device, library);
    defer pipes.close();

    const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
    defer gpu_buf.close();
    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();
    try compute.makeResident(&.{gpu_buf});

    const blob_arena_off: u64 = arena.offsetOf(loaded_blob.ptr);
    var frame = try compute.beginFrame(5000);
    errdefer frame.abort();
    try pipes.encodeGemv(&frame, .q3, .{
        .x = .{ .buffer = gpu_buf, .offset = arena.offsetOf(x_raw.ptr) },
        .qweight = .{ .buffer = gpu_buf, .offset = blob_arena_off },
        .scale_bias = .{ .buffer = gpu_buf, .offset = blob_arena_off + found.params_offset },
        .y = .{ .buffer = gpu_buf, .offset = arena.offsetOf(y_raw.ptr) },
        .n = found.shape[0],
        .k = found.shape[1],
    }, false);
    const sig = try frame.commit();
    try compute.wait(sig, 5000);

    for (ref, 0..) |expected, idx| {
        const got: f32 = @floatCast(y_f16[idx]);
        try std.testing.expectApproxEqAbs(expected, got, @max(0.02, @abs(expected) * 0.015));
    }
}

fn verifyDense(dtype: quant.FloatDType, m: usize, n: usize, k: usize, use_gemm: bool, big_weights: bool) !void {
    const gpa = std.testing.allocator;

    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try DequantPipelines.init(device, library);
    defer pipes.close();

    const x_bytes = m * k * @sizeOf(f16);
    const w_bytes = n * k * @sizeOf(u16);
    const y_bytes = m * n * @sizeOf(f16);
    const arena_cap = try page.alignUp(w_bytes) + try page.alignUp(x_bytes + 256) +
        try page.alignUp(y_bytes + 256) + 2 * page.APPLE_PAGE_SIZE;
    const budget = budget_mod.Budget.compute(.{
        .memory_bytes = arena_cap + 4 * page.APPLE_PAGE_SIZE,
        .user_wire_limit = arena_cap,
        .global_user_wire_limit = arena_cap,
        .gpu_working_set_bytes = arena_cap,
        .configured_cap = null,
        .headroom_bytes = page.APPLE_PAGE_SIZE,
    });
    var arena = try arena_mod.WiredArena.init(arena_cap, budget);
    defer arena.deinit();

    const w_raw = try arena.allocPages(try page.alignUp(w_bytes));
    const x_raw = try arena.allocAligned(x_bytes, 64);
    const y_raw = try arena.allocAligned(y_bytes, 64);
    @memset(y_raw, 0xFF);

    const w_bits: []u16 = @alignCast(std.mem.bytesAsSlice(u16, w_raw[0..w_bytes]));
    const x_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, x_raw));
    const y_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, y_raw));

    for (w_bits, 0..) |*v, idx| {
        const c: i32 = @as(i32, @intCast((idx * 29 + 7) % 37)) - 18;
        var val: f32 = @as(f32, @floatFromInt(c)) * 0.0078125;
        // BF16 exercise: magnitudes beyond the fp16 range (> 65504) stay finite.
        if (big_weights and (idx % k) == 0) val = 3.0e5 + 1024.0 * @as(f32, @floatFromInt((idx / k) % 5));
        v.* = quant.f32ToFloatBits(dtype, val);
    }
    for (x_f16, 0..) |*v, idx| {
        const c: i32 = @as(i32, @intCast((idx * 17 + 11) % 31)) - 15;
        v.* = @floatCast(@as(f32, @floatFromInt(c)) * 0.0625);
        if (big_weights and (idx % k) == 0) v.* = @floatCast(0.0078125);
    }

    const ref = try gpa.alloc(f32, m * n);
    defer gpa.free(ref);
    try quant.denseGemmCpuRef(dtype, x_f16, w_bits, ref, m, n, k);

    const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
    defer gpu_buf.close();
    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();
    try compute.makeResident(&.{gpu_buf});

    const x_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(x_raw.ptr) };
    const w_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(w_raw.ptr) };
    const y_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(y_raw.ptr) };

    var frame = try compute.beginFrame(5000);
    errdefer frame.abort();
    if (use_gemm) {
        try pipes.encodeDenseGemm(&frame, dtype, .{
            .x = x_b,
            .weight = w_b,
            .y = y_b,
            .m = @intCast(m),
            .n = @intCast(n),
            .k = @intCast(k),
        }, false);
    } else {
        std.debug.assert(m == 1);
        try pipes.encodeDenseGemv(&frame, dtype, .{
            .x = x_b,
            .weight = w_b,
            .y = y_b,
            .n = @intCast(n),
            .k = @intCast(k),
        }, false);
    }
    const sig = try frame.commit();
    try compute.wait(sig, 10_000);

    // GEMM BF16 stages X as bfloat (8-bit mantissa); allow its rounding error.
    const x_rel: f32 = if (use_gemm and dtype == .bf16) 1.0 / 256.0 else 1.0 / 2048.0;
    for (0..m) |mi| {
        for (0..n) |ni| {
            var abs_sum: f32 = 0.0;
            for (0..k) |ki| {
                abs_sum += @abs(@as(f32, @floatCast(x_f16[mi * k + ki])) * quant.floatBitsToF32(dtype, w_bits[ni * k + ki]));
            }
            const expected = ref[mi * n + ni];
            const got: f32 = @floatCast(y_f16[mi * n + ni]);
            const tol: f32 = 2e-3 + abs_sum * x_rel + @abs(expected) * (1.0 / 1024.0);
            try std.testing.expect(std.math.isFinite(got));
            try std.testing.expectApproxEqAbs(expected, got, tol);
        }
    }
}

test "dense FP16 GEMV matches CPU fp32 reference on aligned, tail, odd-K and lm_head-wide shapes" {
    try verifyDense(.f16, 1, 16, 64, false, false);
    try verifyDense(.f16, 1, 37, 100, false, false);
    try verifyDense(.f16, 1, 33, 67, false, false);
    try verifyDense(.f16, 1, 1024, 1024, false, false);
    try verifyDense(.f16, 1, 8, 5120, false, false);
    try verifyDense(.f16, 1, 151936, 64, false, false);
}

test "dense BF16 GEMV matches CPU fp32 reference including weights beyond the fp16 range" {
    try verifyDense(.bf16, 1, 16, 64, false, false);
    try verifyDense(.bf16, 1, 37, 100, false, false);
    try verifyDense(.bf16, 1, 33, 67, false, false);
    try verifyDense(.bf16, 1, 1024, 1024, false, false);
    try verifyDense(.bf16, 1, 151936, 64, false, false);
    try verifyDense(.bf16, 1, 64, 2048, false, true);
}

test "dense FP16 GEMM (simdgroup_half8x8) matches CPU fp32 reference on arbitrary M, N, K" {
    try verifyDense(.f16, 1, 40, 36, true, false);
    try verifyDense(.f16, 2, 37, 100, true, false);
    try verifyDense(.f16, 7, 33, 67, true, false);
    try verifyDense(.f16, 16, 64, 256, true, false);
    try verifyDense(.f16, 37, 53, 192, true, false);
    try verifyDense(.f16, 64, 128, 1024, true, false);
}

test "dense BF16 GEMM (simdgroup_bfloat8x8) matches CPU fp32 reference on arbitrary M, N, K" {
    try verifyDense(.bf16, 1, 40, 36, true, false);
    try verifyDense(.bf16, 2, 37, 100, true, false);
    try verifyDense(.bf16, 7, 33, 67, true, false);
    try verifyDense(.bf16, 16, 64, 256, true, false);
    try verifyDense(.bf16, 37, 53, 192, true, false);
    try verifyDense(.bf16, 64, 128, 1024, true, false);
    try verifyDense(.bf16, 8, 48, 2048, true, true);
}

test "dense encoders reject zero dimensions" {
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try DequantPipelines.init(device, library);
    defer pipes.close();

    const cap = 2 * page.APPLE_PAGE_SIZE;
    const budget = budget_mod.Budget.compute(.{
        .memory_bytes = 8 * page.APPLE_PAGE_SIZE,
        .user_wire_limit = cap,
        .global_user_wire_limit = cap,
        .gpu_working_set_bytes = cap,
        .configured_cap = null,
        .headroom_bytes = page.APPLE_PAGE_SIZE,
    });
    var arena = try arena_mod.WiredArena.init(cap, budget);
    defer arena.deinit();
    _ = try arena.allocPages(page.APPLE_PAGE_SIZE);
    const buf = try metal.Buffer.wrap(device, arena.bytes());
    defer buf.close();
    var compute = try metal.Compute.init(device);
    defer compute.close();
    var frame = try compute.beginFrame(5000);
    defer frame.abort();
    const b: metal.BufferBinding = .{ .buffer = buf };
    try std.testing.expectError(error.ZeroDimension, pipes.encodeDenseGemv(&frame, .f16, .{ .x = b, .weight = b, .y = b, .n = 0, .k = 4 }, false));
    try std.testing.expectError(error.ZeroDimension, pipes.encodeDenseGemm(&frame, .bf16, .{ .x = b, .weight = b, .y = b, .m = 0, .n = 4, .k = 4 }, false));
    try std.testing.expectError(error.ZeroDimension, pipes.encodeGemvBatch(&frame, .q4, .{ .x = b, .qweight = b, .scale_bias = b, .y = b, .m = 0, .n = 4, .k = 64 }, false));
    try std.testing.expectError(error.ZeroDimension, pipes.encodeDenseGemvBatch(&frame, .bf16, .{ .x = b, .weight = b, .y = b, .m = 0, .n = 4, .k = 4 }, false));
}

test "batched GEMV matches M=1 GEMV bit-for-bit across q2, q3, q4, f16 and bf16" {
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try DequantPipelines.init(device, library);
    defer pipes.close();

    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();

    const m: u32 = 13;
    const n: u32 = 37;
    const k: u32 = 128;

    const Q4 = quant.QuantOps(.{ .bits = .q4, .group_size = 64 });
    const Q3 = quant.QuantOps(.{ .bits = .q3, .group_size = 64 });
    const Q2 = quant.QuantOps(.{ .bits = .q2, .group_size = 64 });
    const w_q4_bytes = try Q4.packedWeightBytes(n, k);
    const sb_q4_bytes = try Q4.scaleBiasBytes(n, k);
    const w_q3_bytes = try Q3.packedWeightBytes(n, k);
    const sb_q3_bytes = try Q3.scaleBiasBytes(n, k);
    const w_q2_bytes = try Q2.packedWeightBytes(n, k);
    const sb_q2_bytes = try Q2.scaleBiasBytes(n, k);
    const w_f_bytes: usize = @as(usize, n) * k * @sizeOf(u16);
    const x_bytes: usize = @as(usize, m) * k * @sizeOf(f16);
    const y_bytes: usize = @as(usize, m) * n * @sizeOf(f16);

    const cap = try page.alignUp(
        amf.align64(w_q4_bytes) + amf.align64(sb_q4_bytes) +
            amf.align64(w_q3_bytes) + amf.align64(sb_q3_bytes) +
            amf.align64(w_q2_bytes) + amf.align64(sb_q2_bytes) +
            amf.align64(w_f_bytes) * 2 +
            amf.align64(x_bytes) + amf.align64(y_bytes) * 2 + 4 * page.APPLE_PAGE_SIZE,
    );
    const budget = budget_mod.Budget.compute(.{
        .memory_bytes = cap + 4 * page.APPLE_PAGE_SIZE,
        .user_wire_limit = cap,
        .global_user_wire_limit = cap,
        .gpu_working_set_bytes = cap,
        .configured_cap = null,
        .headroom_bytes = page.APPLE_PAGE_SIZE,
    });
    var arena = try arena_mod.WiredArena.init(cap, budget);
    defer arena.deinit();

    const w_q4 = try arena.allocAligned(w_q4_bytes, 64);
    const sb_q4_raw = try arena.allocAligned(sb_q4_bytes, 64);
    const w_q3 = try arena.allocAligned(w_q3_bytes, 64);
    const sb_q3_raw = try arena.allocAligned(sb_q3_bytes, 64);
    const w_q2 = try arena.allocAligned(w_q2_bytes, 64);
    const sb_q2_raw = try arena.allocAligned(sb_q2_bytes, 64);
    const w_f16_raw = try arena.allocAligned(w_f_bytes, 64);
    const w_bf16_raw = try arena.allocAligned(w_f_bytes, 64);
    const x_raw = try arena.allocAligned(x_bytes, 64);
    const y_ref_raw = try arena.allocAligned(y_bytes, 64);
    const y_bat_raw = try arena.allocAligned(y_bytes, 64);

    for (w_q4, 0..) |*b, idx| b.* = @truncate(idx * 37 + 13);
    for (w_q3, 0..) |*b, idx| b.* = @truncate(idx * 29 + 7);
    for (w_q2, 0..) |*b, idx| b.* = @truncate(idx * 43 + 11);
    const sb_q4: []quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, sb_q4_raw));
    for (sb_q4, 0..) |*sb, idx| sb.* = .{ .scale = @floatCast(0.03 + 0.005 * @as(f32, @floatFromInt(idx % 7))), .bias = @floatCast(-0.2) };
    const sb_q3: []quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, sb_q3_raw));
    for (sb_q3, 0..) |*sb, idx| sb.* = .{ .scale = @floatCast(0.04 + 0.005 * @as(f32, @floatFromInt(idx % 5))), .bias = @floatCast(-0.15) };
    const sb_q2: []quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, sb_q2_raw));
    for (sb_q2, 0..) |*sb, idx| sb.* = .{ .scale = @floatCast(0.06 + 0.008 * @as(f32, @floatFromInt(idx % 5))), .bias = @floatCast(-0.12) };
    const w_f16: []u16 = @alignCast(std.mem.bytesAsSlice(u16, w_f16_raw));
    const w_bf16: []u16 = @alignCast(std.mem.bytesAsSlice(u16, w_bf16_raw));
    for (0..@as(usize, n) * k) |idx| {
        const val = (@as(f32, @floatFromInt(@as(i32, @intCast((idx * 19 + 3) % 23)) - 11))) * 0.0625;
        w_f16[idx] = quant.f32ToFloatBits(.f16, val);
        w_bf16[idx] = quant.f32ToFloatBits(.bf16, val);
    }
    const x_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, x_raw));
    for (x_f16, 0..) |*v, idx| {
        const c: i32 = @as(i32, @intCast((idx * 17 + 5) % 29)) - 14;
        v.* = @floatCast(@as(f32, @floatFromInt(c)) * 0.0625);
    }

    const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
    defer gpu_buf.close();
    try compute.makeResident(&.{gpu_buf});

    const y_ref: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, y_ref_raw));
    const y_bat: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, y_bat_raw));
    const x_off = arena.offsetOf(x_raw.ptr);
    const y_ref_off = arena.offsetOf(y_ref_raw.ptr);
    const y_bat_off = arena.offsetOf(y_bat_raw.ptr);

    // 1. q4, q3 & q2 quantized bit-exact check
    for ([_]struct { bits: quant.QuantBits, w_off: u64, sb_off: u64 }{
        .{ .bits = .q4, .w_off = arena.offsetOf(w_q4.ptr), .sb_off = arena.offsetOf(sb_q4_raw.ptr) },
        .{ .bits = .q3, .w_off = arena.offsetOf(w_q3.ptr), .sb_off = arena.offsetOf(sb_q3_raw.ptr) },
        .{ .bits = .q2, .w_off = arena.offsetOf(w_q2.ptr), .sb_off = arena.offsetOf(sb_q2_raw.ptr) },
    }) |tc| {
        @memset(y_ref_raw, 0);
        @memset(y_bat_raw, 0);
        var frame = try compute.beginFrame(5000);
        errdefer frame.abort();
        for (0..m) |row| {
            try pipes.encodeGemv(&frame, tc.bits, .{
                .x = .{ .buffer = gpu_buf, .offset = x_off + @as(u64, row) * k * @sizeOf(f16) },
                .qweight = .{ .buffer = gpu_buf, .offset = tc.w_off },
                .scale_bias = .{ .buffer = gpu_buf, .offset = tc.sb_off },
                .y = .{ .buffer = gpu_buf, .offset = y_ref_off + @as(u64, row) * n * @sizeOf(f16) },
                .n = n,
                .k = k,
            }, row > 0);
        }
        try pipes.encodeGemvBatch(&frame, tc.bits, .{
            .x = .{ .buffer = gpu_buf, .offset = x_off },
            .qweight = .{ .buffer = gpu_buf, .offset = tc.w_off },
            .scale_bias = .{ .buffer = gpu_buf, .offset = tc.sb_off },
            .y = .{ .buffer = gpu_buf, .offset = y_bat_off },
            .m = m,
            .n = n,
            .k = k,
        }, true);
        const sig = try frame.commit();
        try compute.wait(sig, 5000);
        try std.testing.expectEqualSlices(u16, y_ref, y_bat);
    }

    // 2. f16 & bf16 dense bit-exact check
    for ([_]struct { dt: quant.FloatDType, w_off: u64 }{
        .{ .dt = .f16, .w_off = arena.offsetOf(w_f16_raw.ptr) },
        .{ .dt = .bf16, .w_off = arena.offsetOf(w_bf16_raw.ptr) },
    }) |tc| {
        @memset(y_ref_raw, 0);
        @memset(y_bat_raw, 0);
        var frame = try compute.beginFrame(5000);
        errdefer frame.abort();
        for (0..m) |row| {
            try pipes.encodeDenseGemv(&frame, tc.dt, .{
                .x = .{ .buffer = gpu_buf, .offset = x_off + @as(u64, row) * k * @sizeOf(f16) },
                .weight = .{ .buffer = gpu_buf, .offset = tc.w_off },
                .y = .{ .buffer = gpu_buf, .offset = y_ref_off + @as(u64, row) * n * @sizeOf(f16) },
                .n = n,
                .k = k,
            }, row > 0);
        }
        try pipes.encodeDenseGemvBatch(&frame, tc.dt, .{
            .x = .{ .buffer = gpu_buf, .offset = x_off },
            .weight = .{ .buffer = gpu_buf, .offset = tc.w_off },
            .y = .{ .buffer = gpu_buf, .offset = y_bat_off },
            .m = m,
            .n = n,
            .k = k,
        }, true);
        const sig = try frame.commit();
        try compute.wait(sig, 5000);
        try std.testing.expectEqualSlices(u16, y_ref, y_bat);
    }
}

test "G16: Apple10 NAX hook reserves backend while staying on Apple9 simdgroup_matrix" {
    const device = try metal.Device.open();
    defer device.close();
    const info = try device.info();
    try std.testing.expectEqual(PrefillMatmulBackend.simdgroup_apple9, selectPrefillMatmulBackend(info));

    var fake_apple10 = info;
    fake_apple10.raw.supports_apple10 = 1;
    fake_apple10.raw.apple_gpu_family = 10;
    try std.testing.expect(fake_apple10.supportsApple10());
    try std.testing.expectEqual(PrefillMatmulBackend.simdgroup_apple9, selectPrefillMatmulBackend(fake_apple10));
}

test "G16: startup metallib numerical canary passes quickly and rejects NaN / corrupted bf16" {
    var device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try DequantPipelines.init(device, library);
    defer pipes.close();

    // Warm up GPU command path on a reusable Compute context and verify steady-state latency.
    const res = try pipes.runStartupNumericalCanary(&device);
    try std.testing.expect(std.math.isFinite(res.elapsed_us));
    try std.testing.expect(res.elapsed_us > 0.0);
    try std.testing.expect(res.max_abs_err_f16 <= CANARY_TOL_F16);
    try std.testing.expect(res.max_abs_err_bf16 <= CANARY_TOL_BF16);
    try std.testing.expect(res.max_abs_err_q4 <= CANARY_TOL_Q4);

    const scratch = try page.WiredRegion.init(page.APPLE_PAGE_SIZE);
    defer scratch.deinit();
    const gpu_buf = try metal.Buffer.wrap(device, scratch.bytes);
    defer gpu_buf.close();
    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();
    try compute.makeResident(&.{gpu_buf});

    var best_us: f64 = std.math.inf(f64);
    for (0..8) |_| {
        const r = try pipes.runStartupNumericalCanaryInBuffer(&compute, gpu_buf, scratch.bytes);
        if (r.elapsed_us < best_us) best_us = r.elapsed_us;
        try std.testing.expect(r.max_abs_err_f16 <= CANARY_TOL_F16);
        try std.testing.expect(r.max_abs_err_bf16 <= CANARY_TOL_BF16);
        try std.testing.expect(r.max_abs_err_q4 <= CANARY_TOL_Q4);
    }
    const validation_on = (std.c.getenv("MTL_SHADER_VALIDATION") != null or std.c.getenv("MTL_DEBUG_LAYER") != null);
    if (!validation_on) {
        try std.testing.expect(best_us < 500.0);
    }

    // Explicit unit checks for verifyCanarySlices rejection of NaN, Inf, and bf16 corruption.
    const good_h = [_]f16{ @floatCast(1.25), @floatCast(-0.5) };
    const good_r = [_]f32{ 1.25, -0.5 };
    const ok = try verifyCanarySlices(&good_h, &good_r, &good_h, &good_r, &good_h, &good_r);
    try std.testing.expectEqual(@as(f32, 0.0), ok.max_abs_err_f16);

    const nan_h = [_]f16{ std.math.nan(f16), @floatCast(-0.5) };
    try std.testing.expectError(
        error.MetallibNumericalCanaryFailed,
        verifyCanarySlices(&nan_h, &good_r, &good_h, &good_r, &good_h, &good_r),
    );
    try std.testing.expectError(
        error.MetallibNumericalCanaryFailed,
        verifyCanarySlices(&good_h, &good_r, &nan_h, &good_r, &good_h, &good_r),
    );
    const inf_h = [_]f16{ std.math.inf(f16), @floatCast(-0.5) };
    try std.testing.expectError(
        error.MetallibNumericalCanaryFailed,
        verifyCanarySlices(&good_h, &good_r, &good_h, &good_r, &inf_h, &good_r),
    );
    const corrupted_bf16 = [_]f16{ @floatCast(1.25 + 0.05), @floatCast(-0.5) };
    try std.testing.expectError(
        error.MetallibNumericalCanaryFailed,
        verifyCanarySlices(&good_h, &good_r, &corrupted_bf16, &good_r, &good_h, &good_r),
    );
}

fn verifyWideQuantParity(comptime spec: quant.QuantSpec, m: usize, n: usize, k: usize) !void {
    const Ops = quant.QuantOps(spec);
    const gpa = std.testing.allocator;

    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try DequantPipelines.init(device, library);
    defer pipes.close();

    const w_bytes = try Ops.packedWeightBytes(n, k);
    const sb_offset = amf.align64(w_bytes);
    const sb_bytes = try Ops.scaleBiasBytes(n, k);
    const x_bytes = m * k * @sizeOf(f16);
    const y_bytes = m * n * @sizeOf(f16);

    const arena_cap = try page.alignUp(sb_offset + sb_bytes + 256) +
        try page.alignUp(x_bytes + 256) +
        2 * try page.alignUp(y_bytes + 256) +
        page.APPLE_PAGE_SIZE;
    const budget = budget_mod.Budget.compute(.{
        .memory_bytes = arena_cap + 4 * page.APPLE_PAGE_SIZE,
        .user_wire_limit = arena_cap,
        .global_user_wire_limit = arena_cap,
        .gpu_working_set_bytes = arena_cap,
        .configured_cap = null,
        .headroom_bytes = page.APPLE_PAGE_SIZE,
    });
    var arena = try arena_mod.WiredArena.init(arena_cap, budget);
    defer arena.deinit();

    const w_blob = try arena.allocPages(sb_offset + sb_bytes);
    const x_raw = try arena.allocAligned(x_bytes, 64);
    const y32_raw = try arena.allocAligned(y_bytes, 64);
    const y64_raw = try arena.allocAligned(y_bytes, 64);
    @memset(w_blob, 0);
    @memset(y32_raw, 0);
    @memset(y64_raw, 0);

    const qweight = w_blob[0..w_bytes];
    const scale_bias: []quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, w_blob[sb_offset .. sb_offset + sb_bytes]));
    const x_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, x_raw));
    const y32_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, y32_raw));
    const y64_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, y64_raw));

    const groups = try Ops.groupsPerRow(k);
    const row_bytes = try Ops.rowWeightBytes(k);
    const q_mask: u8 = (@as(u8, 1) << @intCast(spec.bits.bitWidth())) - 1;
    var codes: [spec.group_size]u8 = undefined;
    for (0..n) |row| {
        for (0..groups) |g| {
            for (0..spec.group_size) |j| {
                codes[j] = @intCast((row * 13 + g * 7 + j * 5 + 3) & q_mask);
            }
            const g_off = row * row_bytes + g * Ops.groupWeightBytes();
            Ops.packGroup(&codes, qweight[g_off..][0..Ops.groupWeightBytes()]);
            const s: f32 = 0.04 + 0.01 * @as(f32, @floatFromInt((row + g) % 7));
            const b: f32 = -0.20 - 0.04 * @as(f32, @floatFromInt((row * 3 + g) % 5));
            scale_bias[row * groups + g] = .{ .scale = @floatCast(s), .bias = @floatCast(b) };
        }
    }
    for (x_f16, 0..) |*v, idx| {
        const centered: i32 = @as(i32, @intCast((idx * 17 + 11) % 31)) - 15;
        v.* = @floatCast(@as(f32, @floatFromInt(centered)) * 0.0625);
    }

    const ref = try gpa.alloc(f32, m * n);
    defer gpa.free(ref);
    try Ops.matmulRef(x_f16, qweight, scale_bias, ref, m, n, k);

    const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
    defer gpu_buf.close();
    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();
    try compute.makeResident(&.{gpu_buf});

    const w_off = arena.offsetOf(w_blob.ptr);
    const sb_off = w_off + sb_offset;
    const x_off = arena.offsetOf(x_raw.ptr);
    const y32_off = arena.offsetOf(y32_raw.ptr);
    const y64_off = arena.offsetOf(y64_raw.ptr);

    var frame = try compute.beginFrame(5000);
    errdefer frame.abort();
    try pipes.encodeGemm(&frame, spec.bits, .{
        .x = .{ .buffer = gpu_buf, .offset = x_off },
        .qweight = .{ .buffer = gpu_buf, .offset = w_off },
        .scale_bias = .{ .buffer = gpu_buf, .offset = sb_off },
        .y = .{ .buffer = gpu_buf, .offset = y32_off },
        .m = @intCast(m),
        .n = @intCast(n),
        .k = @intCast(k),
        .wide_tile = false,
    }, false);
    try pipes.encodePrefillGemm(&frame, spec.bits, .{
        .x = .{ .buffer = gpu_buf, .offset = x_off },
        .qweight = .{ .buffer = gpu_buf, .offset = w_off },
        .scale_bias = .{ .buffer = gpu_buf, .offset = sb_off },
        .y = .{ .buffer = gpu_buf, .offset = y64_off },
        .m = @intCast(m),
        .n = @intCast(n),
        .k = @intCast(k),
        .wide_tile = true,
    }, false);
    const sig = try frame.commit();
    try compute.wait(sig, 10_000);

    for (ref, 0..) |expected, idx| {
        const g32: f32 = @floatCast(y32_f16[idx]);
        const g64: f32 = @floatCast(y64_f16[idx]);
        const tol: f32 = @max(0.02, @abs(expected) * 0.015);
        try std.testing.expect(std.math.isFinite(g64));
        try std.testing.expectApproxEqAbs(expected, g64, tol);
        try std.testing.expectApproxEqAbs(g32, g64, @max(0.01, @abs(g32) * 0.005));
    }
}

fn verifyWideDenseParity(dtype: quant.FloatDType, m: usize, n: usize, k: usize) !void {
    const gpa = std.testing.allocator;
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try DequantPipelines.init(device, library);
    defer pipes.close();

    const w_bytes = n * k * 2;
    const x_bytes = m * k * 2;
    const y_bytes = m * n * 2;
    const arena_cap = try page.alignUp(w_bytes + 256) +
        try page.alignUp(x_bytes + 256) +
        2 * try page.alignUp(y_bytes + 256) +
        page.APPLE_PAGE_SIZE;
    const budget = budget_mod.Budget.compute(.{
        .memory_bytes = arena_cap + 4 * page.APPLE_PAGE_SIZE,
        .user_wire_limit = arena_cap,
        .global_user_wire_limit = arena_cap,
        .gpu_working_set_bytes = arena_cap,
        .configured_cap = null,
        .headroom_bytes = page.APPLE_PAGE_SIZE,
    });
    var arena = try arena_mod.WiredArena.init(arena_cap, budget);
    defer arena.deinit();

    const w_raw = try arena.allocPages(w_bytes);
    const x_raw = try arena.allocAligned(x_bytes, 64);
    const y32_raw = try arena.allocAligned(y_bytes, 64);
    const y64_raw = try arena.allocAligned(y_bytes, 64);
    @memset(y32_raw, 0);
    @memset(y64_raw, 0);

    const w_bits: []u16 = @alignCast(std.mem.bytesAsSlice(u16, w_raw[0..w_bytes]));
    const x_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, x_raw));
    const y32_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, y32_raw));
    const y64_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, y64_raw));

    for (w_bits, 0..) |*v, idx| {
        const c: i32 = @as(i32, @intCast((idx * 29 + 7) % 37)) - 18;
        v.* = quant.f32ToFloatBits(dtype, @as(f32, @floatFromInt(c)) * 0.0078125);
    }
    for (x_f16, 0..) |*v, idx| {
        const c: i32 = @as(i32, @intCast((idx * 17 + 11) % 31)) - 15;
        v.* = @floatCast(@as(f32, @floatFromInt(c)) * 0.0625);
    }

    const ref = try gpa.alloc(f32, m * n);
    defer gpa.free(ref);
    try quant.denseGemmCpuRef(dtype, x_f16, w_bits, ref, m, n, k);

    const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
    defer gpu_buf.close();
    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();
    try compute.makeResident(&.{gpu_buf});

    const x_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(x_raw.ptr) };
    const w_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(w_raw.ptr) };

    var frame = try compute.beginFrame(5000);
    errdefer frame.abort();
    try pipes.encodeDenseGemm(&frame, dtype, .{
        .x = x_b,
        .weight = w_b,
        .y = .{ .buffer = gpu_buf, .offset = arena.offsetOf(y32_raw.ptr) },
        .m = @intCast(m),
        .n = @intCast(n),
        .k = @intCast(k),
        .wide_tile = false,
    }, false);
    try pipes.encodePrefillDenseGemm(&frame, dtype, .{
        .x = x_b,
        .weight = w_b,
        .y = .{ .buffer = gpu_buf, .offset = arena.offsetOf(y64_raw.ptr) },
        .m = @intCast(m),
        .n = @intCast(n),
        .k = @intCast(k),
        .wide_tile = true,
    }, false);
    const sig = try frame.commit();
    try compute.wait(sig, 10_000);

    const x_rel: f32 = if (dtype == .bf16) 1.0 / 256.0 else 1.0 / 2048.0;
    for (0..m) |mi| {
        for (0..n) |ni| {
            var abs_sum: f32 = 0.0;
            for (0..k) |ki| {
                abs_sum += @abs(@as(f32, @floatCast(x_f16[mi * k + ki])) * quant.floatBitsToF32(dtype, w_bits[ni * k + ki]));
            }
            const expected = ref[mi * n + ni];
            const g32: f32 = @floatCast(y32_f16[mi * n + ni]);
            const g64: f32 = @floatCast(y64_f16[mi * n + ni]);
            const tol: f32 = 2e-3 + abs_sum * x_rel + @abs(expected) * (1.0 / 1024.0);
            try std.testing.expect(std.math.isFinite(g64));
            try std.testing.expectApproxEqAbs(expected, g64, tol);
            try std.testing.expectApproxEqAbs(g32, g64, @max(5e-3, @abs(g32) * 0.005));
        }
    }
}

test "G15: 64x64 wide GEMM (q4, q3, q2) matches 32x32 and CPU fp32 across M=1,17,64,65,137,512,1024" {
    const shapes = [_]struct { m: usize, n: usize, k: usize }{
        .{ .m = 1, .n = 65, .k = 128 },
        .{ .m = 17, .n = 37, .k = 128 },
        .{ .m = 64, .n = 64, .k = 128 },
        .{ .m = 65, .n = 129, .k = 128 },
        .{ .m = 137, .n = 73, .k = 192 },
        .{ .m = 512, .n = 128, .k = 128 },
        .{ .m = 1024, .n = 96, .k = 128 },
    };
    for (shapes) |s| {
        try verifyWideQuantParity(.{ .bits = .q4, .group_size = 64 }, s.m, s.n, s.k);
    }
    try verifyWideQuantParity(.{ .bits = .q3, .group_size = 64 }, 65, 97, 128);
    try verifyWideQuantParity(.{ .bits = .q3, .group_size = 64 }, 512, 64, 128);
    try verifyWideQuantParity(.{ .bits = .q2, .group_size = 64 }, 137, 73, 128);
    try verifyWideQuantParity(.{ .bits = .q2, .group_size = 64 }, 1024, 64, 128);
}

test "G15: 64x64 wide dense GEMM (f16, bf16) matches 32x32 and CPU fp32 across M=1..1024 and odd K" {
    const shapes = [_]struct { m: usize, n: usize, k: usize }{
        .{ .m = 1, .n = 65, .k = 37 },
        .{ .m = 17, .n = 35, .k = 67 },
        .{ .m = 64, .n = 64, .k = 64 },
        .{ .m = 65, .n = 129, .k = 96 },
        .{ .m = 137, .n = 73, .k = 69 },
        .{ .m = 512, .n = 128, .k = 64 },
        .{ .m = 1024, .n = 67, .k = 64 },
    };
    for (shapes) |s| {
        try verifyWideDenseParity(.f16, s.m, s.n, s.k);
        try verifyWideDenseParity(.bf16, s.m, s.n, s.k);
    }
}
