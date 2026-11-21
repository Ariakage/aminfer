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
    gemv_batch_q4: metal.Pipeline,
    gemv_batch_q3: metal.Pipeline,
    gemv_batch_q2: metal.Pipeline,
    gemv_f16: metal.Pipeline,
    gemv_bf16: metal.Pipeline,
    gemm_f16: metal.Pipeline,
    gemm_bf16: metal.Pipeline,
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
            .gemv_batch_q4 = gemv_batch_q4,
            .gemv_batch_q3 = gemv_batch_q3,
            .gemv_batch_q2 = gemv_batch_q2,
            .gemv_f16 = gemv_f16,
            .gemv_bf16 = gemv_bf16,
            .gemm_f16 = gemm_f16,
            .gemm_bf16 = gemm_bf16,
            .gemv_batch_f16 = gemv_batch_f16,
            .gemv_batch_bf16 = gemv_batch_bf16,
        };
    }

    pub fn close(self: DequantPipelines) void {
        self.gemv_batch_bf16.close();
        self.gemv_batch_f16.close();
        self.gemm_bf16.close();
        self.gemm_f16.close();
        self.gemv_bf16.close();
        self.gemv_f16.close();
        self.gemv_batch_q2.close();
        self.gemv_batch_q3.close();
        self.gemv_batch_q4.close();
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
    /// `simdgroup_float8x8` accumulation. Zero heap allocation.
    pub fn encodeDenseGemm(
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
