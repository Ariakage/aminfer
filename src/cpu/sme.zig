//! Bare-metal ARMv9.2-A SME2 engine, `sme2_panel64` pre-packer, P-cluster
//! lock-free dispatcher, ABI register-clobber canary, and standalone Spike S7
//! benchmark (Milestone M4.5: C11, C12, S7, ADR-0018, ADR-0001).
//!
//! Pure Zig owns startup `sysctl` feature & P-cluster topology discovery,
//! `sme2_panel64` pre-packing, activation strip packing, L2 tile blocking,
//! M1–M3 scalar fallback, and lock-free multi-core work distribution.
//! Only the innermost fixed-ABI (`callconv(.c)`) panel loops execute in
//! `src/cpu/sme2_kernels.s`, entered once per macro-task (`smstart`..`smstop`).

const std = @import("std");
const sysinfo = @import("../platform/sysinfo.zig");
const qos = @import("../platform/qos.zig");
const vmstat = @import("../platform/vmstat.zig");
const page = @import("../mem/page.zig");
const arena_mod = @import("../mem/arena.zig");
const budget_mod = @import("../mem/budget.zig");
const quant = @import("../quant/spec.zig");
const amf = @import("../format/amf.zig");
const metal = @import("../bridge/metal.zig");

/// Apple M4+ Streaming Vector Length in bytes (512 bits).
pub const SVL_BYTES: usize = 64;
/// Output columns packed per `sme2_panel64` panel (`ZA` width in FP32 lanes).
pub const PANEL_COLS: usize = 16;
/// Quantization group size along `K` (`group_size = 64`).
pub const GROUP_SIZE: usize = 64;
/// Inline per-`(panel, group)` metadata header size in bytes (`scale`/`lut` + `bias`).
pub const QUANT_HEADER_BYTES: usize = 128;
/// Apple Silicon cacheline size for atomic contention isolation.
pub const CACHELINE_BYTES: usize = 128;
/// Maximum number of P-core worker threads supported by `SmeDispatcher`.
pub const MAX_WORKERS: usize = 32;

/// Constant 64-byte `ZT0` register codebook for affine integer codes `0..15`.
/// Hardware-verified on Apple M4: `luti4 .h` and `luti2 .h` index `ZT0` at
/// 32-bit stride (`zt0_u16[2 * code]`), so duplicating each entry across
/// `[2*i, 2*i+1]` works cleanly with a single `ldr zt0, [ptr]`.
pub const ZT0_AFFINE_TABLE: [32]f16 align(64) = blk: {
    var tbl: [32]f16 = undefined;
    for (0..16) |i| {
        const v: f16 = @floatFromInt(i);
        tbl[2 * i] = v;
        tbl[2 * i + 1] = v;
    }
    break :blk tbl;
};

/// Weight storage format inside an `sme2_panel64` matrix blob (ADR-0018, W15).
pub const PanelDType = enum(u8) {
    f16 = 0,
    bf16 = 1,
    q4_affine = 2,
    q3_affine = 3,
    q2_affine = 4,

    pub fn isQuantized(self: PanelDType) bool {
        return switch (self) {
            .f16, .bf16 => false,
            .q4_affine, .q3_affine, .q2_affine => true,
        };
    }

    pub fn activationIsBf16(self: PanelDType) bool {
        return self == .bf16;
    }

    pub fn label(self: PanelDType) []const u8 {
        return switch (self) {
            .f16 => "f16",
            .bf16 => "bf16",
            .q4_affine => "q4",
            .q3_affine => "q3",
            .q2_affine => "q2",
        };
    }
};

/// Inline metadata interpretation in bytes `0..64` of each quantized group header
/// (high byte of `TensorEntry.reserved0` in `.amf`, ADR-0017 / ADR-0018).
pub const QuantParamKind = enum(u8) {
    affine_scale_bias = 0,
    lut16_codebook = 1,
};

/// Cache locality hint selecting `ld1*` (L2-hot temporal) vs `ldnt1*`
/// (L2-bypassing non-temporal streaming for one-shot cold experts).
pub const CacheHint = enum(u32) {
    temporal = 0,
    non_temporal = 1,
};

/// Fused in-register `ZA` readout epilogue operation.
pub const EpilogueOp = enum(u32) {
    none = 0,
    add_residual = 1,
    /// `out = silu(this_matmul) * aux` (gate projection fused with `up` f32 tensor).
    swiglu_gate_up = 2,
};

/// Output element type emitted directly from streaming SVE registers.
pub const OutputDType = enum(u32) {
    f32 = 0,
    f16 = 1,
    bf16 = 2,

    pub fn elemBytes(self: OutputDType) usize {
        return switch (self) {
            .f32 => 4,
            .f16, .bf16 => 2,
        };
    }
};

/// Execution backend selected at startup (or overridden for testing M1–M3 fallback).
pub const ExecutionBackend = enum {
    sme2,
    scalar_fallback,
};

/// Three-way `M`-dimension microkernel geometry regime (ADR-0018, C12).
pub const KernelRegime = enum {
    /// `M = 1..4`: 4-vector burst load + `ZT0` LUT + streaming multi-vector dot.
    m1_4,
    /// `M = 5..16`: `1 x 4` wide-`N` `ZA0.S..ZA3.S` (`16 x 64`) outer product.
    m5_16,
    /// `M >= 32` (and `17..31` strips): `2 x 2` balanced `ZA0.S..ZA3.S` (`32 x 32`).
    m32,
};

pub const Error = error{
    ZeroDimension,
    UnalignedK,
    Overflow,
    BufferTooSmall,
    InvalidArgument,
    InvalidDType,
    MissingAuxBuffer,
    Sme2Unsupported,
    WorkerInitFailed,
    OutOfMemory,
};

/// Fixed C-ABI argument block passed in `x0` to every `.s` microkernel.
/// Exact byte offsets are pinned by `comptime` assertions below and matched in
/// `src/cpu/sme2_kernels.s`.
pub const KernelArgs = extern struct {
    a_ptr: [*]const u8,
    w_ptr: [*]const u8,
    out_ptr: [*]u8,
    aux_ptr: ?[*]const f32,
    x_group_sums_ptr: ?[*]const f32,
    zt0_table_ptr: [*]const u8,
    a_row_stride_bytes: usize,
    panel_stride_bytes: usize,
    out_row_stride_bytes: usize,
    aux_row_stride_elems: usize,
    m_rows: u32,
    n_valid_cols: u32,
    panel_count: u32,
    groups: u32,
    epilogue_op: u32,
    out_dtype: u32,
    param_kind: u32,
    _pad: u32 = 0,
};

comptime {
    if (@sizeOf(KernelArgs) != 112) @compileError("KernelArgs must be 112 bytes");
    if (@offsetOf(KernelArgs, "a_ptr") != 0) @compileError("a_ptr offset mismatch");
    if (@offsetOf(KernelArgs, "w_ptr") != 8) @compileError("w_ptr offset mismatch");
    if (@offsetOf(KernelArgs, "out_ptr") != 16) @compileError("out_ptr offset mismatch");
    if (@offsetOf(KernelArgs, "aux_ptr") != 24) @compileError("aux_ptr offset mismatch");
    if (@offsetOf(KernelArgs, "x_group_sums_ptr") != 32) @compileError("x_group_sums_ptr offset mismatch");
    if (@offsetOf(KernelArgs, "zt0_table_ptr") != 40) @compileError("zt0_table_ptr offset mismatch");
    if (@offsetOf(KernelArgs, "a_row_stride_bytes") != 48) @compileError("a_row_stride_bytes offset mismatch");
    if (@offsetOf(KernelArgs, "panel_stride_bytes") != 56) @compileError("panel_stride_bytes offset mismatch");
    if (@offsetOf(KernelArgs, "out_row_stride_bytes") != 64) @compileError("out_row_stride_bytes offset mismatch");
    if (@offsetOf(KernelArgs, "aux_row_stride_elems") != 72) @compileError("aux_row_stride_elems offset mismatch");
    if (@offsetOf(KernelArgs, "m_rows") != 80) @compileError("m_rows offset mismatch");
    if (@offsetOf(KernelArgs, "n_valid_cols") != 84) @compileError("n_valid_cols offset mismatch");
    if (@offsetOf(KernelArgs, "panel_count") != 88) @compileError("panel_count offset mismatch");
    if (@offsetOf(KernelArgs, "groups") != 92) @compileError("groups offset mismatch");
    if (@offsetOf(KernelArgs, "epilogue_op") != 96) @compileError("epilogue_op offset mismatch");
    if (@offsetOf(KernelArgs, "out_dtype") != 100) @compileError("out_dtype offset mismatch");
    if (@offsetOf(KernelArgs, "param_kind") != 104) @compileError("param_kind offset mismatch");
}

pub const KernelFn = *const fn (*const KernelArgs) callconv(.c) void;

// All 30 bare-metal ARMv9.2 SME2 assembly entrypoints from `src/cpu/sme2_kernels.s`.
extern fn ami_sme2_m1_4_bf16_t(args: *const KernelArgs) callconv(.c) void;
extern fn ami_sme2_m1_4_bf16_nt(args: *const KernelArgs) callconv(.c) void;
extern fn ami_sme2_m1_4_f16_t(args: *const KernelArgs) callconv(.c) void;
extern fn ami_sme2_m1_4_f16_nt(args: *const KernelArgs) callconv(.c) void;
extern fn ami_sme2_m1_4_q4_t(args: *const KernelArgs) callconv(.c) void;
extern fn ami_sme2_m1_4_q4_nt(args: *const KernelArgs) callconv(.c) void;
extern fn ami_sme2_m1_4_q3_t(args: *const KernelArgs) callconv(.c) void;
extern fn ami_sme2_m1_4_q3_nt(args: *const KernelArgs) callconv(.c) void;
extern fn ami_sme2_m1_4_q2_t(args: *const KernelArgs) callconv(.c) void;
extern fn ami_sme2_m1_4_q2_nt(args: *const KernelArgs) callconv(.c) void;

extern fn ami_sme2_m5_16_bf16_t(args: *const KernelArgs) callconv(.c) void;
extern fn ami_sme2_m5_16_bf16_nt(args: *const KernelArgs) callconv(.c) void;
extern fn ami_sme2_m5_16_f16_t(args: *const KernelArgs) callconv(.c) void;
extern fn ami_sme2_m5_16_f16_nt(args: *const KernelArgs) callconv(.c) void;
extern fn ami_sme2_m5_16_q4_t(args: *const KernelArgs) callconv(.c) void;
extern fn ami_sme2_m5_16_q4_nt(args: *const KernelArgs) callconv(.c) void;
extern fn ami_sme2_m5_16_q3_t(args: *const KernelArgs) callconv(.c) void;
extern fn ami_sme2_m5_16_q3_nt(args: *const KernelArgs) callconv(.c) void;
extern fn ami_sme2_m5_16_q2_t(args: *const KernelArgs) callconv(.c) void;
extern fn ami_sme2_m5_16_q2_nt(args: *const KernelArgs) callconv(.c) void;

extern fn ami_sme2_m32_bf16_t(args: *const KernelArgs) callconv(.c) void;
extern fn ami_sme2_m32_bf16_nt(args: *const KernelArgs) callconv(.c) void;
extern fn ami_sme2_m32_f16_t(args: *const KernelArgs) callconv(.c) void;
extern fn ami_sme2_m32_f16_nt(args: *const KernelArgs) callconv(.c) void;
extern fn ami_sme2_m32_q4_t(args: *const KernelArgs) callconv(.c) void;
extern fn ami_sme2_m32_q4_nt(args: *const KernelArgs) callconv(.c) void;
extern fn ami_sme2_m32_q3_t(args: *const KernelArgs) callconv(.c) void;
extern fn ami_sme2_m32_q3_nt(args: *const KernelArgs) callconv(.c) void;
extern fn ami_sme2_m32_q2_t(args: *const KernelArgs) callconv(.c) void;
extern fn ami_sme2_m32_q2_nt(args: *const KernelArgs) callconv(.c) void;

pub const KernelEntry = struct {
    name: []const u8,
    regime: KernelRegime,
    dtype: PanelDType,
    hint: CacheHint,
    func: KernelFn,
};

pub const ALL_KERNEL_ENTRIES = [_]KernelEntry{
    .{ .name = "ami_sme2_m1_4_bf16_t", .regime = .m1_4, .dtype = .bf16, .hint = .temporal, .func = &ami_sme2_m1_4_bf16_t },
    .{ .name = "ami_sme2_m1_4_bf16_nt", .regime = .m1_4, .dtype = .bf16, .hint = .non_temporal, .func = &ami_sme2_m1_4_bf16_nt },
    .{ .name = "ami_sme2_m1_4_f16_t", .regime = .m1_4, .dtype = .f16, .hint = .temporal, .func = &ami_sme2_m1_4_f16_t },
    .{ .name = "ami_sme2_m1_4_f16_nt", .regime = .m1_4, .dtype = .f16, .hint = .non_temporal, .func = &ami_sme2_m1_4_f16_nt },
    .{ .name = "ami_sme2_m1_4_q4_t", .regime = .m1_4, .dtype = .q4_affine, .hint = .temporal, .func = &ami_sme2_m1_4_q4_t },
    .{ .name = "ami_sme2_m1_4_q4_nt", .regime = .m1_4, .dtype = .q4_affine, .hint = .non_temporal, .func = &ami_sme2_m1_4_q4_nt },
    .{ .name = "ami_sme2_m1_4_q3_t", .regime = .m1_4, .dtype = .q3_affine, .hint = .temporal, .func = &ami_sme2_m1_4_q3_t },
    .{ .name = "ami_sme2_m1_4_q3_nt", .regime = .m1_4, .dtype = .q3_affine, .hint = .non_temporal, .func = &ami_sme2_m1_4_q3_nt },
    .{ .name = "ami_sme2_m1_4_q2_t", .regime = .m1_4, .dtype = .q2_affine, .hint = .temporal, .func = &ami_sme2_m1_4_q2_t },
    .{ .name = "ami_sme2_m1_4_q2_nt", .regime = .m1_4, .dtype = .q2_affine, .hint = .non_temporal, .func = &ami_sme2_m1_4_q2_nt },

    .{ .name = "ami_sme2_m5_16_bf16_t", .regime = .m5_16, .dtype = .bf16, .hint = .temporal, .func = &ami_sme2_m5_16_bf16_t },
    .{ .name = "ami_sme2_m5_16_bf16_nt", .regime = .m5_16, .dtype = .bf16, .hint = .non_temporal, .func = &ami_sme2_m5_16_bf16_nt },
    .{ .name = "ami_sme2_m5_16_f16_t", .regime = .m5_16, .dtype = .f16, .hint = .temporal, .func = &ami_sme2_m5_16_f16_t },
    .{ .name = "ami_sme2_m5_16_f16_nt", .regime = .m5_16, .dtype = .f16, .hint = .non_temporal, .func = &ami_sme2_m5_16_f16_nt },
    .{ .name = "ami_sme2_m5_16_q4_t", .regime = .m5_16, .dtype = .q4_affine, .hint = .temporal, .func = &ami_sme2_m5_16_q4_t },
    .{ .name = "ami_sme2_m5_16_q4_nt", .regime = .m5_16, .dtype = .q4_affine, .hint = .non_temporal, .func = &ami_sme2_m5_16_q4_nt },
    .{ .name = "ami_sme2_m5_16_q3_t", .regime = .m5_16, .dtype = .q3_affine, .hint = .temporal, .func = &ami_sme2_m5_16_q3_t },
    .{ .name = "ami_sme2_m5_16_q3_nt", .regime = .m5_16, .dtype = .q3_affine, .hint = .non_temporal, .func = &ami_sme2_m5_16_q3_nt },
    .{ .name = "ami_sme2_m5_16_q2_t", .regime = .m5_16, .dtype = .q2_affine, .hint = .temporal, .func = &ami_sme2_m5_16_q2_t },
    .{ .name = "ami_sme2_m5_16_q2_nt", .regime = .m5_16, .dtype = .q2_affine, .hint = .non_temporal, .func = &ami_sme2_m5_16_q2_nt },

    .{ .name = "ami_sme2_m32_bf16_t", .regime = .m32, .dtype = .bf16, .hint = .temporal, .func = &ami_sme2_m32_bf16_t },
    .{ .name = "ami_sme2_m32_bf16_nt", .regime = .m32, .dtype = .bf16, .hint = .non_temporal, .func = &ami_sme2_m32_bf16_nt },
    .{ .name = "ami_sme2_m32_f16_t", .regime = .m32, .dtype = .f16, .hint = .temporal, .func = &ami_sme2_m32_f16_t },
    .{ .name = "ami_sme2_m32_f16_nt", .regime = .m32, .dtype = .f16, .hint = .non_temporal, .func = &ami_sme2_m32_f16_nt },
    .{ .name = "ami_sme2_m32_q4_t", .regime = .m32, .dtype = .q4_affine, .hint = .temporal, .func = &ami_sme2_m32_q4_t },
    .{ .name = "ami_sme2_m32_q4_nt", .regime = .m32, .dtype = .q4_affine, .hint = .non_temporal, .func = &ami_sme2_m32_q4_nt },
    .{ .name = "ami_sme2_m32_q3_t", .regime = .m32, .dtype = .q3_affine, .hint = .temporal, .func = &ami_sme2_m32_q3_t },
    .{ .name = "ami_sme2_m32_q3_nt", .regime = .m32, .dtype = .q3_affine, .hint = .non_temporal, .func = &ami_sme2_m32_q3_nt },
    .{ .name = "ami_sme2_m32_q2_t", .regime = .m32, .dtype = .q2_affine, .hint = .temporal, .func = &ami_sme2_m32_q2_t },
    .{ .name = "ami_sme2_m32_q2_nt", .regime = .m32, .dtype = .q2_affine, .hint = .non_temporal, .func = &ami_sme2_m32_q2_nt },
};

pub fn selectKernelFn(regime: KernelRegime, dtype: PanelDType, hint: CacheHint) KernelFn {
    return switch (regime) {
        .m1_4 => switch (dtype) {
            .bf16 => if (hint == .temporal) &ami_sme2_m1_4_bf16_t else &ami_sme2_m1_4_bf16_nt,
            .f16 => if (hint == .temporal) &ami_sme2_m1_4_f16_t else &ami_sme2_m1_4_f16_nt,
            .q4_affine => if (hint == .temporal) &ami_sme2_m1_4_q4_t else &ami_sme2_m1_4_q4_nt,
            .q3_affine => if (hint == .temporal) &ami_sme2_m1_4_q3_t else &ami_sme2_m1_4_q3_nt,
            .q2_affine => if (hint == .temporal) &ami_sme2_m1_4_q2_t else &ami_sme2_m1_4_q2_nt,
        },
        .m5_16 => switch (dtype) {
            .bf16 => if (hint == .temporal) &ami_sme2_m5_16_bf16_t else &ami_sme2_m5_16_bf16_nt,
            .f16 => if (hint == .temporal) &ami_sme2_m5_16_f16_t else &ami_sme2_m5_16_f16_nt,
            .q4_affine => if (hint == .temporal) &ami_sme2_m5_16_q4_t else &ami_sme2_m5_16_q4_nt,
            .q3_affine => if (hint == .temporal) &ami_sme2_m5_16_q3_t else &ami_sme2_m5_16_q3_nt,
            .q2_affine => if (hint == .temporal) &ami_sme2_m5_16_q2_t else &ami_sme2_m5_16_q2_nt,
        },
        .m32 => switch (dtype) {
            .bf16 => if (hint == .temporal) &ami_sme2_m32_bf16_t else &ami_sme2_m32_bf16_nt,
            .f16 => if (hint == .temporal) &ami_sme2_m32_f16_t else &ami_sme2_m32_f16_nt,
            .q4_affine => if (hint == .temporal) &ami_sme2_m32_q4_t else &ami_sme2_m32_q4_nt,
            .q3_affine => if (hint == .temporal) &ami_sme2_m32_q3_t else &ami_sme2_m32_q3_nt,
            .q2_affine => if (hint == .temporal) &ami_sme2_m32_q2_t else &ami_sme2_m32_q2_nt,
        },
    };
}

// ============================================================================
// Runtime capability & P-cluster topology discovery (C11)
// ============================================================================

pub const SmeCaps = struct {
    has_sme: bool,
    has_sme2: bool,
    has_sme_b16f32: bool,
    has_sme_f16f32: bool,
    has_sme_i16i64: bool,
    svl_bytes: u64,
    pcore_physical_cpu: u64,
    pcore_cpus_per_l2: u64,
    pcore_clusters: u64,
    pcore_l2_bytes: u64,
    default_backend: ExecutionBackend,

    pub fn fromFacts(facts: sysinfo.Facts) SmeCaps {
        const supported = facts.supportsSme2Engine();
        return .{
            .has_sme = facts.has_sme,
            .has_sme2 = facts.has_sme2,
            .has_sme_b16f32 = facts.has_sme_b16f32,
            .has_sme_f16f32 = facts.has_sme_f16f32,
            .has_sme_i16i64 = facts.has_sme_i16i64,
            .svl_bytes = facts.sme_max_svl_bytes,
            .pcore_physical_cpu = if (facts.pcore_physical_cpu > 0) facts.pcore_physical_cpu else @max(@as(u64, 1), facts.performance_cores),
            .pcore_cpus_per_l2 = if (facts.pcore_cpus_per_l2 > 0) facts.pcore_cpus_per_l2 else @max(@as(u64, 1), facts.performance_cores),
            .pcore_clusters = @max(@as(u64, 1), facts.pcore_clusters),
            .pcore_l2_bytes = facts.pcore_l2_cache_bytes,
            .default_backend = if (supported) .sme2 else .scalar_fallback,
        };
    }

    pub fn probe() sysinfo.Error!SmeCaps {
        const facts = try sysinfo.Facts.read();
        return fromFacts(facts);
    }

    pub fn supportsSme2(self: SmeCaps) bool {
        return self.default_backend == .sme2;
    }

    pub fn resolveBackend(self: SmeCaps, override: ?ExecutionBackend) Error!ExecutionBackend {
        const chosen = override orelse self.default_backend;
        if (chosen == .sme2 and !self.supportsSme2()) return error.Sme2Unsupported;
        return chosen;
    }

    /// Recommended worker count across physical P-core clusters.
    pub fn recommendedThreads(self: SmeCaps, threads_per_cluster: usize) usize {
        const tpc = @max(@as(usize, 1), @min(threads_per_cluster, @as(usize, @intCast(self.pcore_cpus_per_l2))));
        const total = @as(usize, @intCast(self.pcore_clusters)) * tpc;
        return @max(@as(usize, 1), @min(total, MAX_WORKERS));
    }
};

// ============================================================================
// `sme2_panel64` geometry & pre-packing helpers (W15 binary contract)
// ============================================================================

pub inline fn paddedCols(n: usize) usize {
    return (n + PANEL_COLS - 1) & ~@as(usize, PANEL_COLS - 1);
}

pub inline fn panelCount(n: usize) usize {
    return paddedCols(n) / PANEL_COLS;
}

pub inline fn groupBytes(dtype: PanelDType) usize {
    return switch (dtype) {
        .f16, .bf16 => 2048,
        .q4_affine => QUANT_HEADER_BYTES + 512, // 640 B
        .q3_affine => QUANT_HEADER_BYTES + 384, // 512 B
        .q2_affine => QUANT_HEADER_BYTES + 256, // 384 B
    };
}

pub fn panelStrideBytes(dtype: PanelDType, k: usize) Error!usize {
    if (k == 0) return error.ZeroDimension;
    if ((k % GROUP_SIZE) != 0) return error.UnalignedK;
    const groups = k / GROUP_SIZE;
    return std.math.mul(usize, groups, groupBytes(dtype)) catch return error.Overflow;
}

pub fn packedBlobBytes(dtype: PanelDType, n: usize, k: usize) Error!usize {
    if (n == 0 or k == 0) return error.ZeroDimension;
    const ps = try panelStrideBytes(dtype, k);
    const n_pad = std.math.add(usize, n, PANEL_COLS - 1) catch return error.Overflow;
    const panels = (n_pad & ~@as(usize, PANEL_COLS - 1)) / PANEL_COLS;
    return std.math.mul(usize, panels, ps) catch return error.Overflow;
}

/// Packs a row-major `[n, k]` `f16` or `bf16` matrix (raw `u16` bits) into
/// `sme2_panel64` layout (`[panels, groups, 32 kp, 16 cols, 2 k]`).
/// Out-of-range rows `n..paddedCols(n)` are zero-padded.
pub fn packDensePanel64(
    dtype: PanelDType,
    src_row_major_u16: []const u16,
    n: usize,
    k: usize,
    dst: []u8,
) Error!void {
    if (dtype.isQuantized()) return error.InvalidDType;
    const total_bytes = try packedBlobBytes(dtype, n, k);
    const nk = std.math.mul(usize, n, k) catch return error.Overflow;
    if (src_row_major_u16.len < nk or dst.len < total_bytes) return error.BufferTooSmall;

    const panels = panelCount(n);
    const groups = k / GROUP_SIZE;
    const g_bytes = groupBytes(dtype);

    var out_off: usize = 0;
    for (0..panels) |p| {
        for (0..groups) |g| {
            const k_base = g * GROUP_SIZE;
            const grp_slice = dst[out_off .. out_off + g_bytes];
            for (0..32) |kp| {
                const k0 = k_base + 2 * kp;
                const k1 = k0 + 1;
                const chunk_off = kp * 64;
                for (0..PANEL_COLS) |c| {
                    const col = p * PANEL_COLS + c;
                    const v0: u16 = if (col < n) src_row_major_u16[col * k + k0] else 0;
                    const v1: u16 = if (col < n) src_row_major_u16[col * k + k1] else 0;
                    std.mem.writeInt(u16, grp_slice[chunk_off + c * 4 + 0 ..][0..2], v0, .little);
                    std.mem.writeInt(u16, grp_slice[chunk_off + c * 4 + 2 ..][0..2], v1, .little);
                }
            }
            out_off += g_bytes;
        }
    }
}

/// Packs a row-major `[n, k]` `f32` matrix into `f16` or `bf16` `sme2_panel64` layout.
pub fn packDenseFromF32Panel64(
    dtype: PanelDType,
    src_f32: []const f32,
    n: usize,
    k: usize,
    dst: []u8,
) Error!void {
    if (dtype.isQuantized()) return error.InvalidDType;
    const total_bytes = try packedBlobBytes(dtype, n, k);
    const nk = std.math.mul(usize, n, k) catch return error.Overflow;
    if (src_f32.len < nk or dst.len < total_bytes) return error.BufferTooSmall;

    const panels = panelCount(n);
    const groups = k / GROUP_SIZE;
    const g_bytes = groupBytes(dtype);
    const fdt: quant.FloatDType = if (dtype == .bf16) .bf16 else .f16;

    var out_off: usize = 0;
    for (0..panels) |p| {
        for (0..groups) |g| {
            const k_base = g * GROUP_SIZE;
            const grp_slice = dst[out_off .. out_off + g_bytes];
            for (0..32) |kp| {
                const k0 = k_base + 2 * kp;
                const k1 = k0 + 1;
                const chunk_off = kp * 64;
                for (0..PANEL_COLS) |c| {
                    const col = p * PANEL_COLS + c;
                    const v0: u16 = if (col < n) quant.f32ToFloatBits(fdt, src_f32[col * k + k0]) else 0;
                    const v1: u16 = if (col < n) quant.f32ToFloatBits(fdt, src_f32[col * k + k1]) else 0;
                    std.mem.writeInt(u16, grp_slice[chunk_off + c * 4 + 0 ..][0..2], v0, .little);
                    std.mem.writeInt(u16, grp_slice[chunk_off + c * 4 + 2 ..][0..2], v1, .little);
                }
            }
            out_off += g_bytes;
        }
    }
}

fn writeQuantCodesForGroup(
    dtype: PanelDType,
    codes_nk: []const u8,
    n: usize,
    k: usize,
    panel: usize,
    k_base: usize,
    code_dst: []u8,
) void {
    switch (dtype) {
        .q4_affine => {
            for (0..32) |kp| {
                const k0 = k_base + 2 * kp;
                const k1 = k0 + 1;
                for (0..PANEL_COLS) |c| {
                    const col = panel * PANEL_COLS + c;
                    const q0: u8 = if (col < n) (codes_nk[col * k + k0] & 0x0F) else 0;
                    const q1: u8 = if (col < n) (codes_nk[col * k + k1] & 0x0F) else 0;
                    code_dst[kp * 16 + c] = q0 | (q1 << 4);
                }
            }
        },
        .q3_affine => {
            // Two halves x 3 x 64 B (384 B total), unpacking in 11 SVE bit-ops/half.
            for (0..2) |half| {
                const kp_base = half * 16;
                var quads: [4][64]u8 = undefined;
                for (0..4) |q| {
                    for (0..64) |i| {
                        const kp = kp_base + q * 4 + (i / 16);
                        const c = i % 16;
                        const col = panel * PANEL_COLS + c;
                        const k0 = k_base + 2 * kp;
                        const k1 = k0 + 1;
                        const q0: u8 = if (col < n) (codes_nk[col * k + k0] & 0x07) else 0;
                        const q1: u8 = if (col < n) (codes_nk[col * k + k1] & 0x07) else 0;
                        quads[q][i] = q0 | (q1 << 4);
                    }
                }
                const h_off = half * 192;
                for (0..64) |i| {
                    const q3b = quads[3][i];
                    code_dst[h_off + 0 + i] = quads[0][i] | ((q3b & 0x01) << 3) | ((q3b & 0x10) << 3);
                    code_dst[h_off + 64 + i] = quads[1][i] | ((q3b & 0x02) << 2) | ((q3b & 0x20) << 2);
                    code_dst[h_off + 128 + i] = quads[2][i] | ((q3b & 0x04) << 1) | ((q3b & 0x40) << 1);
                }
            }
        },
        .q2_affine => {
            // 32 kp x 8 B = 256 B, exact M4 `luti2` 2-bit crumb order.
            for (0..32) |kp| {
                const k0 = k_base + 2 * kp;
                const k1 = k0 + 1;
                for (0..8) |b| {
                    const col0 = panel * PANEL_COLS + 2 * b;
                    const col1 = col0 + 1;
                    const c00: u8 = if (col0 < n) (codes_nk[col0 * k + k0] & 0x03) else 0;
                    const c01: u8 = if (col0 < n) (codes_nk[col0 * k + k1] & 0x03) else 0;
                    const c10: u8 = if (col1 < n) (codes_nk[col1 * k + k0] & 0x03) else 0;
                    const c11: u8 = if (col1 < n) (codes_nk[col1 * k + k1] & 0x03) else 0;
                    code_dst[kp * 8 + b] = c00 | (c01 << 2) | (c10 << 4) | (c11 << 6);
                }
            }
        },
        .f16, .bf16 => unreachable,
    }
}

/// Packs unpacked integer codes `[n, k]` (`0..15` for q4, `0..7` for q3, `0..3` for q2)
/// and per-group affine `ScaleBias` `[n, k / 64]` into `sme2_panel64` layout
/// (`QuantParamKind.affine_scale_bias`).
pub fn packQuantAffinePanel64(
    dtype: PanelDType,
    codes_nk: []const u8,
    scale_bias: []const quant.ScaleBias,
    n: usize,
    k: usize,
    dst: []u8,
) Error!void {
    if (!dtype.isQuantized()) return error.InvalidDType;
    const total_bytes = try packedBlobBytes(dtype, n, k);
    const nk = std.math.mul(usize, n, k) catch return error.Overflow;
    const groups = k / GROUP_SIZE;
    const ng = std.math.mul(usize, n, groups) catch return error.Overflow;
    if (codes_nk.len < nk or scale_bias.len < ng or dst.len < total_bytes) return error.BufferTooSmall;

    const panels = panelCount(n);
    const g_bytes = groupBytes(dtype);

    var out_off: usize = 0;
    for (0..panels) |p| {
        for (0..groups) |g| {
            const grp_slice = dst[out_off .. out_off + g_bytes];
            for (0..PANEL_COLS) |c| {
                const col = p * PANEL_COLS + c;
                const sb: quant.ScaleBias = if (col < n)
                    scale_bias[col * groups + g]
                else
                    .{ .scale = 0.0, .bias = 0.0 };
                const s_bits: u16 = @bitCast(sb.scale);
                const b_bits: u16 = @bitCast(sb.bias);
                std.mem.writeInt(u16, grp_slice[c * 4 + 0 ..][0..2], s_bits, .little);
                std.mem.writeInt(u16, grp_slice[c * 4 + 2 ..][0..2], s_bits, .little);
                std.mem.writeInt(u16, grp_slice[64 + c * 4 + 0 ..][0..2], b_bits, .little);
                std.mem.writeInt(u16, grp_slice[64 + c * 4 + 2 ..][0..2], b_bits, .little);
            }
            writeQuantCodesForGroup(dtype, codes_nk, n, k, p, g * GROUP_SIZE, grp_slice[QUANT_HEADER_BYTES..]);
            out_off += g_bytes;
        }
    }
}

/// Repacks MLX-format row-major `q4_affine`, `q3_affine`, or `q2_affine` weights directly into
/// `sme2_panel64` (`QuantParamKind.affine_scale_bias`) with zero heap allocations.
pub fn packMlxQuantPanel64(
    bits: quant.QuantBits,
    mlx_qweight: []const u8,
    scale_bias: []const quant.ScaleBias,
    n: usize,
    k: usize,
    dst: []u8,
) Error!void {
    const dtype: PanelDType = switch (bits) {
        .q4 => .q4_affine,
        .q3 => .q3_affine,
        .q2 => .q2_affine,
    };
    const total_bytes = try packedBlobBytes(dtype, n, k);
    const groups = k / GROUP_SIZE;
    const ng = std.math.mul(usize, n, groups) catch return error.Overflow;
    const src_grp_bytes: usize = switch (bits) {
        .q4 => 32,
        .q3 => 24,
        .q2 => 16,
    };
    const row_bytes = std.math.mul(usize, groups, src_grp_bytes) catch return error.Overflow;
    const src_total = std.math.mul(usize, n, row_bytes) catch return error.Overflow;
    if (mlx_qweight.len < src_total or scale_bias.len < ng or dst.len < total_bytes) {
        return error.BufferTooSmall;
    }

    const panels = panelCount(n);
    const g_bytes = groupBytes(dtype);
    var panel_codes: [PANEL_COLS][GROUP_SIZE]u8 = undefined;

    var out_off: usize = 0;
    for (0..panels) |p| {
        for (0..groups) |g| {
            const grp_slice = dst[out_off .. out_off + g_bytes];
            for (0..PANEL_COLS) |c| {
                const col = p * PANEL_COLS + c;
                const sb: quant.ScaleBias = if (col < n)
                    scale_bias[col * groups + g]
                else
                    .{ .scale = 0.0, .bias = 0.0 };
                const s_bits: u16 = @bitCast(sb.scale);
                const b_bits: u16 = @bitCast(sb.bias);
                std.mem.writeInt(u16, grp_slice[c * 4 + 0 ..][0..2], s_bits, .little);
                std.mem.writeInt(u16, grp_slice[c * 4 + 2 ..][0..2], s_bits, .little);
                std.mem.writeInt(u16, grp_slice[64 + c * 4 + 0 ..][0..2], b_bits, .little);
                std.mem.writeInt(u16, grp_slice[64 + c * 4 + 2 ..][0..2], b_bits, .little);

                if (col < n) {
                    const src = mlx_qweight[col * row_bytes + g * src_grp_bytes ..][0..src_grp_bytes];
                    switch (bits) {
                        .q4 => {
                            for (0..32) |i| {
                                panel_codes[c][2 * i] = src[i] & 0x0F;
                                panel_codes[c][2 * i + 1] = src[i] >> 4;
                            }
                        },
                        .q3 => {
                            for (0..8) |blk| {
                                const word: u24 =
                                    @as(u24, src[blk * 3 + 0]) |
                                    (@as(u24, src[blk * 3 + 1]) << 8) |
                                    (@as(u24, src[blk * 3 + 2]) << 16);
                                inline for (0..8) |j| {
                                    panel_codes[c][blk * 8 + j] = @intCast((word >> @intCast(j * 3)) & 0x07);
                                }
                            }
                        },
                        .q2 => {
                            for (0..16) |i| {
                                const b = src[i];
                                panel_codes[c][4 * i + 0] = b & 0x03;
                                panel_codes[c][4 * i + 1] = (b >> 2) & 0x03;
                                panel_codes[c][4 * i + 2] = (b >> 4) & 0x03;
                                panel_codes[c][4 * i + 3] = (b >> 6) & 0x03;
                            }
                        },
                    }
                } else {
                    @memset(&panel_codes[c], 0);
                }
            }

            const code_dst = grp_slice[QUANT_HEADER_BYTES..];
            switch (bits) {
                .q4 => {
                    for (0..32) |kp| {
                        for (0..PANEL_COLS) |c| {
                            code_dst[kp * 16 + c] = panel_codes[c][2 * kp] | (panel_codes[c][2 * kp + 1] << 4);
                        }
                    }
                },
                .q3 => {
                    for (0..2) |half| {
                        const kp_base = half * 16;
                        var quads: [4][64]u8 = undefined;
                        for (0..4) |q| {
                            for (0..64) |i| {
                                const kp = kp_base + q * 4 + (i / 16);
                                const c = i % 16;
                                quads[q][i] = (panel_codes[c][2 * kp] & 0x07) | ((panel_codes[c][2 * kp + 1] & 0x07) << 4);
                            }
                        }
                        const h_off = half * 192;
                        for (0..64) |i| {
                            const q3b = quads[3][i];
                            code_dst[h_off + 0 + i] = quads[0][i] | ((q3b & 0x01) << 3) | ((q3b & 0x10) << 3);
                            code_dst[h_off + 64 + i] = quads[1][i] | ((q3b & 0x02) << 2) | ((q3b & 0x20) << 2);
                            code_dst[h_off + 128 + i] = quads[2][i] | ((q3b & 0x04) << 1) | ((q3b & 0x40) << 1);
                        }
                    }
                },
                .q2 => {
                    for (0..32) |kp| {
                        for (0..8) |b| {
                            const c0 = 2 * b;
                            const c1 = c0 + 1;
                            const c00 = panel_codes[c0][2 * kp] & 0x03;
                            const c01 = panel_codes[c0][2 * kp + 1] & 0x03;
                            const c10 = panel_codes[c1][2 * kp] & 0x03;
                            const c11 = panel_codes[c1][2 * kp + 1] & 0x03;
                            code_dst[kp * 8 + b] = c00 | (c01 << 2) | (c10 << 4) | (c11 << 6);
                        }
                    }
                },
            }
            out_off += g_bytes;
        }
    }
}

/// Packs quantized codes `[n, k]` with per-`(panel, group)` 16-entry `ZT0`
/// codebooks (`QuantParamKind.lut16_codebook`, ADR-0017 / ADR-0018).
/// `codebooks_f16` has shape `[panelCount(n) * (k / 64)][16]f16`.
/// Optional `biases_f16` has shape `[n * (k / 64)]f16` (zeros when null).
pub fn packQuantLut16Panel64(
    dtype: PanelDType,
    codes_nk: []const u8,
    codebooks_f16: []const [16]f16,
    biases_f16: ?[]const f16,
    n: usize,
    k: usize,
    dst: []u8,
) Error!void {
    if (!dtype.isQuantized()) return error.InvalidDType;
    const total_bytes = try packedBlobBytes(dtype, n, k);
    const nk = std.math.mul(usize, n, k) catch return error.Overflow;
    const panels = panelCount(n);
    const groups = k / GROUP_SIZE;
    const pg = std.math.mul(usize, panels, groups) catch return error.Overflow;
    const ng = std.math.mul(usize, n, groups) catch return error.Overflow;
    if (codes_nk.len < nk or codebooks_f16.len < pg or dst.len < total_bytes) {
        return error.BufferTooSmall;
    }
    if (biases_f16) |bf| {
        if (bf.len < ng) return error.BufferTooSmall;
    }

    const g_bytes = groupBytes(dtype);
    var out_off: usize = 0;
    for (0..panels) |p| {
        for (0..groups) |g| {
            const grp_slice = dst[out_off .. out_off + g_bytes];
            const cb = codebooks_f16[p * groups + g];
            // Store [v0, v0, v1, v1, ..., v15, v15] for 32-bit-stride M4 `ldr zt0`.
            for (0..16) |idx| {
                const bits: u16 = @bitCast(cb[idx]);
                std.mem.writeInt(u16, grp_slice[idx * 4 + 0 ..][0..2], bits, .little);
                std.mem.writeInt(u16, grp_slice[idx * 4 + 2 ..][0..2], bits, .little);
            }
            for (0..PANEL_COLS) |c| {
                const col = p * PANEL_COLS + c;
                const b_val: f16 = if (col < n and biases_f16 != null)
                    biases_f16.?[col * groups + g]
                else
                    0.0;
                const b_bits: u16 = @bitCast(b_val);
                std.mem.writeInt(u16, grp_slice[64 + c * 4 + 0 ..][0..2], b_bits, .little);
                std.mem.writeInt(u16, grp_slice[64 + c * 4 + 2 ..][0..2], b_bits, .little);
            }
            writeQuantCodesForGroup(dtype, codes_nk, n, k, p, g * GROUP_SIZE, grp_slice[QUANT_HEADER_BYTES..]);
            out_off += g_bytes;
        }
    }
}

/// Packs up to 16 activation rows `m_start .. m_start + m_rows` from row-major
/// `[M, K]` (`u16` raw `f16`/`bf16` bits) into hardware-native `[K/2, 16, 2]`
/// `fmopa`/`bfmopa` layout (`dst_u16.len >= k * 16`). Rows `m_rows..16` are zeroed.
pub fn packActivationStrip(
    a_row_major_u16: []const u16,
    m_start: usize,
    m_rows: usize,
    k: usize,
    dst_u16: []u16,
) void {
    std.debug.assert(m_rows <= PANEL_COLS);
    std.debug.assert((k % 2) == 0);
    std.debug.assert(dst_u16.len >= k * PANEL_COLS);
    const kp_count = k / 2;
    for (0..kp_count) |kp| {
        const k0 = 2 * kp;
        const k1 = k0 + 1;
        const base = kp * 32;
        for (0..m_rows) |r| {
            const row_off = (m_start + r) * k;
            dst_u16[base + 2 * r] = a_row_major_u16[row_off + k0];
            dst_u16[base + 2 * r + 1] = a_row_major_u16[row_off + k1];
        }
        for (m_rows..PANEL_COLS) |r| {
            dst_u16[base + 2 * r] = 0;
            dst_u16[base + 2 * r + 1] = 0;
        }
    }
}

/// Computes per-group FP32 activation sums `sum_{j=0..63} f32(a[m, g*64 + j])`
/// for algebraic FP32 bias factoring in `M = 1..4` quantized streaming dot kernels.
pub fn computeGroupActivationSums(
    a_f16_u16: []const u16,
    m_start: usize,
    m_rows: usize,
    k: usize,
    out_sums: []f32,
) void {
    const groups = k / GROUP_SIZE;
    std.debug.assert(out_sums.len >= m_rows * groups);
    for (0..m_rows) |r| {
        const row = a_f16_u16[(m_start + r) * k ..][0..k];
        for (0..groups) |g| {
            var s: f32 = 0.0;
            const grp = row[g * GROUP_SIZE ..][0..GROUP_SIZE];
            for (grp) |bits| {
                s += @floatCast(@as(f16, @bitCast(bits)));
            }
            out_sums[r * groups + g] = s;
        }
    }
}

// ============================================================================
// Pure-Zig FP32 scalar reference & M1–M3 fallback (zero SME instructions!)
// ============================================================================

inline fn siluF32(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

fn applyEpilogueAndStoreScalar(
    val: f32,
    aux_val: f32,
    epilogue: EpilogueOp,
    out_dtype: OutputDType,
    out_bytes: []u8,
    elem_index: usize,
) void {
    const post: f32 = switch (epilogue) {
        .none => val,
        .add_residual => val + aux_val,
        .swiglu_gate_up => siluF32(val) * aux_val,
    };
    switch (out_dtype) {
        .f32 => {
            const bits: u32 = @bitCast(post);
            std.mem.writeInt(u32, out_bytes[elem_index * 4 ..][0..4], bits, .little);
        },
        .f16 => {
            const h: f16 = @floatCast(post);
            const bits: u16 = @bitCast(h);
            std.mem.writeInt(u16, out_bytes[elem_index * 2 ..][0..2], bits, .little);
        },
        .bf16 => {
            const bits: u16 = quant.f32ToBf16(post);
            std.mem.writeInt(u16, out_bytes[elem_index * 2 ..][0..2], bits, .little);
        },
    }
}

/// Pure-Zig FP32 scalar execution directly over `sme2_panel64` weights.
/// Used both as the M1–M3 `ExecutionBackend.scalar_fallback` (zero `SIGILL`)
/// and as a panel-format cross-check.
pub fn matmulPanel64ScalarRange(
    dtype: PanelDType,
    param_kind: QuantParamKind,
    epilogue: EpilogueOp,
    out_dtype: OutputDType,
    a_u16: []const u16,
    w_packed: []const u8,
    aux_f32: ?[]const f32,
    out_bytes: []u8,
    m_start: usize,
    m_rows: usize,
    panel_start: usize,
    panel_count: usize,
    n: usize,
    k: usize,
) void {
    const groups = k / GROUP_SIZE;
    const g_bytes = groupBytes(dtype);
    const ps = groups * g_bytes;
    const act_dt: quant.FloatDType = if (dtype.activationIsBf16()) .bf16 else .f16;

    for (panel_start..panel_start + panel_count) |p| {
        const p_base = p * ps;
        const col_base = p * PANEL_COLS;
        const valid_cols = @min(PANEL_COLS, n -| col_base);
        if (valid_cols == 0) continue;

        for (m_start..m_start + m_rows) |mi| {
            const a_row = a_u16[mi * k ..][0..k];
            var acc: [PANEL_COLS]f32 = @splat(0.0);

            for (0..groups) |g| {
                const grp = w_packed[p_base + g * g_bytes ..][0..g_bytes];
                const k_base = g * GROUP_SIZE;

                switch (dtype) {
                    .f16, .bf16 => {
                        const w_dt: quant.FloatDType = if (dtype == .bf16) .bf16 else .f16;
                        for (0..32) |kp| {
                            const x0 = quant.floatBitsToF32(act_dt, a_row[k_base + 2 * kp]);
                            const x1 = quant.floatBitsToF32(act_dt, a_row[k_base + 2 * kp + 1]);
                            const c_off = kp * 64;
                            for (0..valid_cols) |c| {
                                const w0 = quant.floatBitsToF32(w_dt, std.mem.readInt(u16, grp[c_off + c * 4 + 0 ..][0..2], .little));
                                const w1 = quant.floatBitsToF32(w_dt, std.mem.readInt(u16, grp[c_off + c * 4 + 2 ..][0..2], .little));
                                acc[c] += x0 * w0 + x1 * w1;
                            }
                        }
                    },
                    .q4_affine, .q3_affine, .q2_affine => {
                        var scales: [PANEL_COLS]f32 = undefined;
                        var biases: [PANEL_COLS]f32 = undefined;
                        var lut: [16]f32 = undefined;
                        for (0..16) |idx| {
                            const h_bits = std.mem.readInt(u16, grp[idx * 4 ..][0..2], .little);
                            const v: f32 = @floatCast(@as(f16, @bitCast(h_bits)));
                            scales[idx] = v;
                            lut[idx] = v;
                        }
                        for (0..PANEL_COLS) |c| {
                            const b_bits = std.mem.readInt(u16, grp[64 + c * 4 ..][0..2], .little);
                            biases[c] = @floatCast(@as(f16, @bitCast(b_bits)));
                        }
                        const codes = grp[QUANT_HEADER_BYTES..];

                        switch (dtype) {
                            .q4_affine => {
                                for (0..32) |kp| {
                                    const x0 = quant.floatBitsToF32(.f16, a_row[k_base + 2 * kp]);
                                    const x1 = quant.floatBitsToF32(.f16, a_row[k_base + 2 * kp + 1]);
                                    for (0..valid_cols) |c| {
                                        const byte = codes[kp * 16 + c];
                                        const q0 = byte & 0x0F;
                                        const q1 = byte >> 4;
                                        const w0 = if (param_kind == .affine_scale_bias)
                                            @as(f32, @floatFromInt(q0)) * scales[c] + biases[c]
                                        else
                                            lut[q0] + biases[c];
                                        const w1 = if (param_kind == .affine_scale_bias)
                                            @as(f32, @floatFromInt(q1)) * scales[c] + biases[c]
                                        else
                                            lut[q1] + biases[c];
                                        acc[c] += x0 * w0 + x1 * w1;
                                    }
                                }
                            },
                            .q3_affine => {
                                for (0..2) |half| {
                                    const h_off = half * 192;
                                    var quads: [4][64]u8 = undefined;
                                    for (0..64) |i| {
                                        const b0 = codes[h_off + 0 + i];
                                        const b1 = codes[h_off + 64 + i];
                                        const b2 = codes[h_off + 128 + i];
                                        quads[0][i] = b0 & 0x77;
                                        quads[1][i] = b1 & 0x77;
                                        quads[2][i] = b2 & 0x77;
                                        quads[3][i] = ((b0 & 0x88) >> 3) | ((b1 & 0x88) >> 2) | ((b2 & 0x88) >> 1);
                                    }
                                    for (0..4) |q| {
                                        for (0..4) |kp_sub| {
                                            const kp = half * 16 + q * 4 + kp_sub;
                                            const x0 = quant.floatBitsToF32(.f16, a_row[k_base + 2 * kp]);
                                            const x1 = quant.floatBitsToF32(.f16, a_row[k_base + 2 * kp + 1]);
                                            for (0..valid_cols) |c| {
                                                const byte = quads[q][kp_sub * 16 + c];
                                                const q0 = byte & 0x07;
                                                const q1 = (byte >> 4) & 0x07;
                                                const w0 = if (param_kind == .affine_scale_bias)
                                                    @as(f32, @floatFromInt(q0)) * scales[c] + biases[c]
                                                else
                                                    lut[q0] + biases[c];
                                                const w1 = if (param_kind == .affine_scale_bias)
                                                    @as(f32, @floatFromInt(q1)) * scales[c] + biases[c]
                                                else
                                                    lut[q1] + biases[c];
                                                acc[c] += x0 * w0 + x1 * w1;
                                            }
                                        }
                                    }
                                }
                            },
                            .q2_affine => {
                                for (0..32) |kp| {
                                    const x0 = quant.floatBitsToF32(.f16, a_row[k_base + 2 * kp]);
                                    const x1 = quant.floatBitsToF32(.f16, a_row[k_base + 2 * kp + 1]);
                                    for (0..valid_cols) |c| {
                                        const byte = codes[kp * 8 + (c / 2)];
                                        const shift: u3 = @intCast((c & 1) * 4);
                                        const q0 = (byte >> shift) & 0x03;
                                        const q1 = (byte >> (shift + 2)) & 0x03;
                                        const w0 = if (param_kind == .affine_scale_bias)
                                            @as(f32, @floatFromInt(q0)) * scales[c] + biases[c]
                                        else
                                            lut[q0] + biases[c];
                                        const w1 = if (param_kind == .affine_scale_bias)
                                            @as(f32, @floatFromInt(q1)) * scales[c] + biases[c]
                                        else
                                            lut[q1] + biases[c];
                                        acc[c] += x0 * w0 + x1 * w1;
                                    }
                                }
                            },
                            else => unreachable,
                        }
                    },
                }
            }

            for (0..valid_cols) |c| {
                const idx = mi * n + col_base + c;
                const aux_val: f32 = if (epilogue != .none) aux_f32.?[idx] else 0.0;
                applyEpilogueAndStoreScalar(acc[c], aux_val, epilogue, out_dtype, out_bytes, idx);
            }
        }
    }
}

/// Independent pure-Zig row-major FP32 reference matmul with epilogue, used to
/// verify both the `sme2_panel64` pre-packers and the `.s` microkernels.
pub fn matmulRowMajorRefF32(
    a_f32: []const f32,
    w_f32: []const f32,
    aux_f32: ?[]const f32,
    epilogue: EpilogueOp,
    out_f32: []f32,
    m: usize,
    n: usize,
    k: usize,
) void {
    for (0..m) |mi| {
        for (0..n) |ni| {
            var acc: f32 = 0.0;
            for (0..k) |ki| {
                acc += a_f32[mi * k + ki] * w_f32[ni * k + ki];
            }
            const idx = mi * n + ni;
            out_f32[idx] = switch (epilogue) {
                .none => acc,
                .add_residual => acc + aux_f32.?[idx],
                .swiglu_gate_up => siluF32(acc) * aux_f32.?[idx],
            };
        }
    }
}

// ============================================================================
// P-Cluster Lock-Free SmeDispatcher (C11 — zero hot-path heap allocations!)
// ============================================================================

pub const GemmConfig = struct {
    dtype: PanelDType,
    param_kind: QuantParamKind = .affine_scale_bias,
    cache_hint: CacheHint = .temporal,
    epilogue: EpilogueOp = .none,
    out_dtype: OutputDType = .f32,
    /// When `null`, uses `SmeCaps.default_backend` (`.sme2` on M4+, `.scalar_fallback` on M1–M3).
    backend: ?ExecutionBackend = null,
    /// Optional thread-count cap (`1..dispatcher.threadCount()`), e.g. for Spike S7 sweeps.
    max_threads: ?usize = null,
};

pub const DispatcherConfig = struct {
    /// Threads per physical P-core cluster (`0` = auto: `min(4, pcore_cpus_per_l2)`).
    threads_per_cluster: usize = 0,
    /// Optional explicit total thread count override (`1..MAX_WORKERS`).
    total_threads: ?usize = null,
    /// Preallocated packed activation scratch capacity in `u16` elements
    /// (default `32 * 16384 = 524,288 u16 = 1 MiB`, handles any `K <= 16384` per 32-row strip).
    a_pack_u16_cap: usize = 32 * 16384,
    /// Preallocated per-group activation sum scratch capacity in `f32` elements
    /// (default `4 * 1024 = 4,096 f32 = 16 KiB`, handles any `K <= 65536` for `M = 1..4`).
    x_sums_f32_cap: usize = 4 * 1024,
    /// Optional capability override (used to test M1–M3 `scalar_fallback` behavior on M4).
    caps_override: ?SmeCaps = null,
};

const DispatchJob = struct {
    backend: ExecutionBackend = .scalar_fallback,
    regime: KernelRegime = .m1_4,
    dtype: PanelDType = .f16,
    param_kind: QuantParamKind = .affine_scale_bias,
    epilogue: EpilogueOp = .none,
    out_dtype: OutputDType = .f32,
    kernel_fn: KernelFn = &ami_sme2_m1_4_f16_t,
    a_raw_ptr: [*]const u16 = undefined,
    a_packed_ptr: [*]const u8 = undefined,
    a_row_stride_bytes: usize = 0,
    w_ptr: [*]const u8 = undefined,
    panel_stride_bytes: usize = 0,
    out_ptr: [*]u8 = undefined,
    out_row_stride_bytes: usize = 0,
    aux_ptr: ?[*]const f32 = null,
    x_group_sums_ptr: ?[*]const f32 = null,
    m_start: usize = 0,
    m_rows: usize = 0,
    n: usize = 0,
    k: usize = 0,
    total_panels: usize = 0,
    panels_per_tile: usize = 1,
    total_tiles: u32 = 0,
    active_workers: usize = 1,
    custom_fn: ?*const fn (*anyopaque, u32) void = null,
    custom_ctx: ?*anyopaque = null,
};

const SharedPoolState = struct {
    next_tile: std.atomic.Value(u32) align(CACHELINE_BYTES),
    completed_tiles: std.atomic.Value(u32) align(CACHELINE_BYTES),
    workers_done: std.atomic.Value(u32) align(CACHELINE_BYTES),
    epoch: std.atomic.Value(u32) align(CACHELINE_BYTES),
    sleeping_workers: std.atomic.Value(u32) align(CACHELINE_BYTES),
    shutdown: std.atomic.Value(bool) align(CACHELINE_BYTES),
    ready_count: std.atomic.Value(u32) align(CACHELINE_BYTES),
    wake_mutex: std.c.pthread_mutex_t,
    wake_cond: std.c.pthread_cond_t,
    init_mutex: std.c.pthread_mutex_t,
    init_cond: std.c.pthread_cond_t,
    worker_errors: [MAX_WORKERS]?qos.Error,
    job: DispatchJob,
    bg_count: usize,
};

/// Persistent P-core `QOS_CLASS_USER_INTERACTIVE` worker pool with lock-free
/// 128B-padded atomic macro-tile claiming and zero hot-path heap allocations (C11).
pub const SmeDispatcher = struct {
    allocator: std.mem.Allocator,
    caps: SmeCaps,
    num_threads: usize,
    state: *SharedPoolState,
    threads: [MAX_WORKERS]std.Thread,
    bg_spawned: usize,
    a_pack_scratch: []align(SVL_BYTES) u16,
    x_sums_scratch: []align(SVL_BYTES) f32,

    pub fn init(allocator: std.mem.Allocator, config: DispatcherConfig) (Error || sysinfo.Error || std.Thread.SpawnError)!SmeDispatcher {
        const caps = config.caps_override orelse try SmeCaps.probe();
        const tpc: usize = if (config.threads_per_cluster > 0)
            config.threads_per_cluster
        else
            @min(@as(usize, 4), @as(usize, @intCast(@max(@as(u64, 1), caps.pcore_cpus_per_l2))));
        const num_threads: usize = if (config.total_threads) |tt|
            @max(@as(usize, 1), @min(tt, MAX_WORKERS))
        else
            caps.recommendedThreads(tpc);

        if (config.a_pack_u16_cap < 32 * GROUP_SIZE or config.x_sums_f32_cap < 4) {
            return error.InvalidArgument;
        }

        const state = try allocator.create(SharedPoolState);
        errdefer allocator.destroy(state);

        const a_pack_scratch = try allocator.alignedAlloc(u16, .fromByteUnits(SVL_BYTES), config.a_pack_u16_cap);
        errdefer allocator.free(a_pack_scratch);

        const x_sums_scratch = try allocator.alignedAlloc(f32, .fromByteUnits(SVL_BYTES), config.x_sums_f32_cap);
        errdefer allocator.free(x_sums_scratch);

        const bg_count = num_threads - 1;
        state.* = .{
            .next_tile = .init(0),
            .completed_tiles = .init(0),
            .workers_done = .init(0),
            .epoch = .init(0),
            .sleeping_workers = .init(0),
            .shutdown = .init(false),
            .ready_count = .init(0),
            .wake_mutex = std.c.PTHREAD_MUTEX_INITIALIZER,
            .wake_cond = std.c.PTHREAD_COND_INITIALIZER,
            .init_mutex = std.c.PTHREAD_MUTEX_INITIALIZER,
            .init_cond = std.c.PTHREAD_COND_INITIALIZER,
            .worker_errors = @splat(null),
            .job = .{},
            .bg_count = bg_count,
        };

        var self: SmeDispatcher = .{
            .allocator = allocator,
            .caps = caps,
            .num_threads = num_threads,
            .state = state,
            .threads = undefined,
            .bg_spawned = 0,
            .a_pack_scratch = a_pack_scratch,
            .x_sums_scratch = x_sums_scratch,
        };
        errdefer self.deinit();

        for (0..bg_count) |i| {
            self.threads[i] = try std.Thread.spawn(.{}, workerMain, .{ state, i });
            self.bg_spawned += 1;
        }

        if (bg_count > 0) {
            _ = std.c.pthread_mutex_lock(&state.init_mutex);
            while (state.ready_count.load(.acquire) < bg_count) {
                _ = std.c.pthread_cond_wait(&state.init_cond, &state.init_mutex);
            }
            _ = std.c.pthread_mutex_unlock(&state.init_mutex);

            for (0..bg_count) |i| {
                if (state.worker_errors[i] != null) return error.WorkerInitFailed;
            }
        }

        return self;
    }

    pub fn deinit(self: *SmeDispatcher) void {
        const state = self.state;
        _ = std.c.pthread_mutex_lock(&state.wake_mutex);
        state.shutdown.store(true, .release);
        _ = std.c.pthread_cond_broadcast(&state.wake_cond);
        _ = std.c.pthread_mutex_unlock(&state.wake_mutex);

        for (0..self.bg_spawned) |i| {
            self.threads[i].join();
        }
        _ = std.c.pthread_cond_destroy(&state.init_cond);
        _ = std.c.pthread_mutex_destroy(&state.init_mutex);
        _ = std.c.pthread_cond_destroy(&state.wake_cond);
        _ = std.c.pthread_mutex_destroy(&state.wake_mutex);

        self.allocator.free(self.x_sums_scratch);
        self.allocator.free(self.a_pack_scratch);
        self.allocator.destroy(state);
    }

    pub fn threadCount(self: *const SmeDispatcher) usize {
        return self.num_threads;
    }

    /// Returns the Spike-S7 calibrated thread count `SmeDispatcher` uses for
    /// `(dtype, m, n, k, backend)` when `cfg.max_threads` is `null`.
    /// Tiny L2-resident projections (`< 512 KiB` or `<= 16` panels, e.g. MoE routers)
    /// run single-threaded to avoid wakeup latency; larger L2/DRAM projections fan
    /// out across all P-cluster threads (`1.20x-1.72x` measured scaling on M4).
    pub fn autoThreadCount(
        self: *const SmeDispatcher,
        dtype: PanelDType,
        _: usize,
        n: usize,
        k: usize,
        _: ExecutionBackend,
    ) usize {
        const total_panels = panelCount(n);
        const groups = k / GROUP_SIZE;
        const w_total_bytes = total_panels * groups * groupBytes(dtype);
        if (w_total_bytes < 512 * 1024 or total_panels <= 16) return 1;
        return @max(@as(usize, 1), @min(self.num_threads, total_panels));
    }

    fn executeTile(job: *const DispatchJob, tile_idx: u32) void {
        if (job.custom_fn) |cf| {
            cf(job.custom_ctx.?, tile_idx);
            return;
        }
        const p_start = @as(usize, tile_idx) * job.panels_per_tile;
        if (p_start >= job.total_panels) return;
        const p_count = @min(job.panels_per_tile, job.total_panels - p_start);
        const col_start = p_start * PANEL_COLS;
        const valid_cols = job.n -| col_start;
        if (valid_cols == 0) return;

        if (job.backend == .scalar_fallback) {
            const total_out_elems = (job.m_start + job.m_rows) * job.n;
            const total_out_bytes = total_out_elems * job.out_dtype.elemBytes();
            const total_w_bytes = job.total_panels * job.panel_stride_bytes;
            const total_a_elems = (job.m_start + job.m_rows) * job.k;
            const aux_slice: ?[]const f32 = if (job.aux_ptr) |ap| ap[0..total_out_elems] else null;
            matmulPanel64ScalarRange(
                job.dtype,
                job.param_kind,
                job.epilogue,
                job.out_dtype,
                job.a_raw_ptr[0..total_a_elems],
                job.w_ptr[0..total_w_bytes],
                aux_slice,
                job.out_ptr[0..total_out_bytes],
                job.m_start,
                job.m_rows,
                p_start,
                p_count,
                job.n,
                job.k,
            );
            return;
        }

        const elem_bytes = job.out_dtype.elemBytes();
        const out_tile_off = job.m_start * job.out_row_stride_bytes + col_start * elem_bytes;
        const aux_tile_ptr: ?[*]const f32 = if (job.aux_ptr) |ap|
            ap + job.m_start * job.n + col_start
        else
            null;

        const args: KernelArgs = .{
            .a_ptr = job.a_packed_ptr,
            .w_ptr = job.w_ptr + p_start * job.panel_stride_bytes,
            .out_ptr = job.out_ptr + out_tile_off,
            .aux_ptr = aux_tile_ptr,
            .x_group_sums_ptr = job.x_group_sums_ptr,
            .zt0_table_ptr = std.mem.sliceAsBytes(&ZT0_AFFINE_TABLE).ptr,
            .a_row_stride_bytes = job.a_row_stride_bytes,
            .panel_stride_bytes = job.panel_stride_bytes,
            .out_row_stride_bytes = job.out_row_stride_bytes,
            .aux_row_stride_elems = job.n,
            .m_rows = @intCast(job.m_rows),
            .n_valid_cols = @intCast(valid_cols),
            .panel_count = @intCast(p_count),
            .groups = @intCast(job.k / GROUP_SIZE),
            .epilogue_op = @backingInt(job.epilogue),
            .out_dtype = @backingInt(job.out_dtype),
            .param_kind = @backingInt(job.param_kind),
        };
        job.kernel_fn(&args);
    }

    fn runClaimLoop(state: *SharedPoolState) void {
        const total_tiles = state.job.total_tiles;
        while (true) {
            const tile = state.next_tile.fetchAdd(1, .monotonic);
            if (tile >= total_tiles) break;
            executeTile(&state.job, tile);
            _ = state.completed_tiles.fetchAdd(1, .release);
        }
    }

    fn workerMain(state: *SharedPoolState, bg_idx: usize) void {
        qos.steerPerformanceCore(.user_interactive) catch |err| {
            state.worker_errors[bg_idx] = err;
        };
        _ = std.c.pthread_mutex_lock(&state.init_mutex);
        _ = state.ready_count.fetchAdd(1, .release);
        _ = std.c.pthread_cond_signal(&state.init_cond);
        _ = std.c.pthread_mutex_unlock(&state.init_mutex);

        var last_epoch: u32 = 0;
        while (true) {
            var spins: u32 = 0;
            while (state.epoch.load(.acquire) == last_epoch) {
                if (state.shutdown.load(.acquire)) return;
                if (spins < 1500) {
                    spins += 1;
                    std.atomic.spinLoopHint();
                } else {
                    _ = std.c.pthread_mutex_lock(&state.wake_mutex);
                    _ = state.sleeping_workers.fetchAdd(1, .seq_cst);
                    while (state.epoch.load(.seq_cst) == last_epoch and !state.shutdown.load(.seq_cst)) {
                        _ = std.c.pthread_cond_wait(&state.wake_cond, &state.wake_mutex);
                    }
                    _ = state.sleeping_workers.fetchSub(1, .seq_cst);
                    _ = std.c.pthread_mutex_unlock(&state.wake_mutex);
                    if (state.shutdown.load(.acquire)) return;
                }
            }
            last_epoch = state.epoch.load(.acquire);
            // Worker slot index is `bg_idx + 1` (slot 0 is the caller thread).
            if (bg_idx + 1 < state.job.active_workers) {
                runClaimLoop(state);
            }
            _ = state.workers_done.fetchAdd(1, .release);
        }
    }

    fn dispatchStrip(
        self: *SmeDispatcher,
        backend: ExecutionBackend,
        regime: KernelRegime,
        cfg: GemmConfig,
        a_u16: []const u16,
        w_packed: []const u8,
        aux_f32: ?[]const f32,
        out_bytes: []u8,
        m_start: usize,
        m_rows: usize,
        n: usize,
        k: usize,
        ps: usize,
    ) Error!void {
        const total_panels = panelCount(n);
        const groups = k / GROUP_SIZE;
        const elem_bytes = cfg.out_dtype.elemBytes();
        const auto_threads = self.autoThreadCount(cfg.dtype, m_rows, n, k, backend);
        const req_threads = @min(self.num_threads, cfg.max_threads orelse auto_threads);
        const active_workers = @max(@as(usize, 1), @min(req_threads, total_panels));

        // Pack activations or compute per-group activation sums into preallocated scratch.
        var a_packed_ptr: [*]const u8 = undefined;
        var a_row_stride_bytes: usize = 0;
        var x_sums_ptr: ?[*]const f32 = null;

        if (backend == .sme2) {
            switch (regime) {
                .m1_4 => {
                    a_packed_ptr = std.mem.sliceAsBytes(a_u16[m_start * k ..]).ptr;
                    a_row_stride_bytes = k * @sizeOf(u16);
                    if (cfg.dtype.isQuantized()) {
                        const need_sums = m_rows * groups;
                        if (need_sums > self.x_sums_scratch.len) return error.BufferTooSmall;
                        computeGroupActivationSums(a_u16, m_start, m_rows, k, self.x_sums_scratch[0..need_sums]);
                        x_sums_ptr = self.x_sums_scratch.ptr;
                    }
                },
                .m5_16 => {
                    const need_u16 = PANEL_COLS * k;
                    if (need_u16 > self.a_pack_scratch.len) return error.BufferTooSmall;
                    packActivationStrip(a_u16, m_start, m_rows, k, self.a_pack_scratch[0..need_u16]);
                    a_packed_ptr = std.mem.sliceAsBytes(self.a_pack_scratch[0..need_u16]).ptr;
                    a_row_stride_bytes = need_u16 * @sizeOf(u16);
                },
                .m32 => {
                    const strip_u16 = PANEL_COLS * k;
                    const need_u16 = 2 * strip_u16;
                    if (need_u16 > self.a_pack_scratch.len) return error.BufferTooSmall;
                    const top_rows = @min(m_rows, PANEL_COLS);
                    const bot_rows = m_rows - top_rows;
                    packActivationStrip(a_u16, m_start, top_rows, k, self.a_pack_scratch[0..strip_u16]);
                    packActivationStrip(a_u16, m_start + top_rows, bot_rows, k, self.a_pack_scratch[strip_u16..need_u16]);
                    a_packed_ptr = std.mem.sliceAsBytes(self.a_pack_scratch[0..need_u16]).ptr;
                    a_row_stride_bytes = strip_u16 * @sizeOf(u16);
                },
            }
        }

        // Macro-tile panel granularity: align to tile width (4 for 1x4, 2 for 2x2)
        // while keeping >= 1 macro-task per worker so `smstart` amortizes over many panels.
        const min_unit: usize = switch (regime) {
            .m1_4 => 1,
            .m5_16 => 4,
            .m32 => 2,
        };
        const target_tiles: usize = if (active_workers == 1) 1 else active_workers * 4;
        var panels_per_tile = @max(min_unit, (total_panels + target_tiles - 1) / target_tiles);
        if (min_unit > 1) {
            panels_per_tile = ((panels_per_tile + min_unit - 1) / min_unit) * min_unit;
        }
        const total_tiles_usize = (total_panels + panels_per_tile - 1) / panels_per_tile;
        const total_tiles: u32 = @intCast(total_tiles_usize);

        const job: DispatchJob = .{
            .backend = backend,
            .regime = regime,
            .dtype = cfg.dtype,
            .param_kind = cfg.param_kind,
            .epilogue = cfg.epilogue,
            .out_dtype = cfg.out_dtype,
            .kernel_fn = selectKernelFn(regime, cfg.dtype, cfg.cache_hint),
            .a_raw_ptr = a_u16.ptr,
            .a_packed_ptr = a_packed_ptr,
            .a_row_stride_bytes = a_row_stride_bytes,
            .w_ptr = w_packed.ptr,
            .panel_stride_bytes = ps,
            .out_ptr = out_bytes.ptr,
            .out_row_stride_bytes = n * elem_bytes,
            .aux_ptr = if (aux_f32) |a| a.ptr else null,
            .x_group_sums_ptr = x_sums_ptr,
            .m_start = m_start,
            .m_rows = m_rows,
            .n = n,
            .k = k,
            .total_panels = total_panels,
            .panels_per_tile = panels_per_tile,
            .total_tiles = total_tiles,
            .active_workers = active_workers,
        };

        // Single-thread fast path: zero synchronization overhead.
        if (active_workers <= 1 or total_tiles <= 1) {
            for (0..total_tiles) |t| {
                executeTile(&job, @intCast(t));
            }
            return;
        }

        const state = self.state;
        state.job = job;
        state.next_tile.store(0, .monotonic);
        state.completed_tiles.store(0, .monotonic);
        state.workers_done.store(0, .monotonic);
        _ = state.epoch.fetchAdd(1, .seq_cst);

        if (state.sleeping_workers.load(.seq_cst) > 0) {
            _ = std.c.pthread_mutex_lock(&state.wake_mutex);
            _ = std.c.pthread_cond_broadcast(&state.wake_cond);
            _ = std.c.pthread_mutex_unlock(&state.wake_mutex);
        }

        // Caller thread (worker 0) participates in claiming tiles lock-free.
        runClaimLoop(state);

        // Wait until all in-flight macro-tiles and every background worker have exited
        // this epoch before returning (prevents `state.job` overwrite race on next call).
        const expected_bg: u32 = @intCast(state.bg_count);
        while (state.completed_tiles.load(.acquire) < total_tiles or
            state.workers_done.load(.acquire) < expected_bg)
        {
            std.atomic.spinLoopHint();
        }
    }

    /// Executes `Out[M, N] = epilogue(A[M, K] * W_panel64[N, K]^T, aux[M, N])`.
    /// Zero dynamic heap allocations on the hot path.
    /// Automatically dispatches across the three geometry-matched SME2 microkernel
    /// regimes (`M=1..4` streaming dot, `M=5..16` `1x4` wide-N `ZA`, `M>=17` `2x2`
    /// balanced `ZA` with hybrid tail strip selection) or `scalar_fallback`.
    pub fn gemm(
        self: *SmeDispatcher,
        cfg: GemmConfig,
        a_u16: []const u16,
        w_packed: []const u8,
        aux_f32: ?[]const f32,
        out_bytes: []u8,
        m: usize,
        n: usize,
        k: usize,
    ) Error!void {
        if (m == 0 or n == 0 or k == 0) return error.ZeroDimension;
        if ((k % GROUP_SIZE) != 0) return error.UnalignedK;

        const backend = try self.caps.resolveBackend(cfg.backend);
        const ps = try panelStrideBytes(cfg.dtype, k);
        const w_need = try packedBlobBytes(cfg.dtype, n, k);
        const mk = std.math.mul(usize, m, k) catch return error.Overflow;
        const mn = std.math.mul(usize, m, n) catch return error.Overflow;
        const out_need = std.math.mul(usize, mn, cfg.out_dtype.elemBytes()) catch return error.Overflow;

        if (a_u16.len < mk or w_packed.len < w_need or out_bytes.len < out_need) {
            return error.BufferTooSmall;
        }
        if (cfg.epilogue != .none) {
            const aux = aux_f32 orelse return error.MissingAuxBuffer;
            if (aux.len < mn) return error.BufferTooSmall;
        }

        // Slice M into optimal geometry-matched strips:
        //   - remaining >= 17 -> 32-row strip on `2x2` (`m32`)
        //   - remaining in 5..16 -> `1x4` (`m5_16`)
        //   - remaining in 1..4 -> streaming burst dot (`m1_4`)
        var m_cur: usize = 0;
        while (m_cur < m) {
            const rem = m - m_cur;
            const regime: KernelRegime = if (rem <= 4)
                .m1_4
            else if (rem <= 16)
                .m5_16
            else
                .m32;
            const strip_rows: usize = switch (regime) {
                .m1_4 => rem,
                .m5_16 => rem,
                .m32 => @min(rem, 32),
            };
            try self.dispatchStrip(backend, regime, cfg, a_u16, w_packed, aux_f32, out_bytes, m_cur, strip_rows, n, k, ps);
            m_cur += strip_rows;
        }
    }

    /// Convenience typed wrapper for `OutputDType.f32`.
    pub fn gemmF32(
        self: *SmeDispatcher,
        cfg: GemmConfig,
        a_u16: []const u16,
        w_packed: []const u8,
        aux_f32: ?[]const f32,
        out_f32: []f32,
        m: usize,
        n: usize,
        k: usize,
    ) Error!void {
        var c = cfg;
        c.out_dtype = .f32;
        return self.gemm(c, a_u16, w_packed, aux_f32, std.mem.sliceAsBytes(out_f32), m, n, k);
    }

    /// Runs `task_fn(ctx, tile_idx)` across `[0, total_tiles)` using the persistent
    /// lock-free P-core worker pool with zero heap allocation on the hot path.
    pub fn parallelFor(
        self: *SmeDispatcher,
        total_tiles: u32,
        max_threads: ?usize,
        ctx: *anyopaque,
        task_fn: *const fn (*anyopaque, u32) void,
    ) void {
        if (total_tiles == 0) return;
        const req_threads = @min(self.num_threads, max_threads orelse self.num_threads);
        const active_workers = @max(@as(usize, 1), @min(req_threads, @as(usize, total_tiles)));
        if (active_workers <= 1 or total_tiles <= 1) {
            for (0..total_tiles) |t| {
                task_fn(ctx, @intCast(t));
            }
            return;
        }

        const state = self.state;
        state.job = .{
            .total_tiles = total_tiles,
            .active_workers = active_workers,
            .custom_fn = task_fn,
            .custom_ctx = ctx,
        };
        state.next_tile.store(0, .monotonic);
        state.completed_tiles.store(0, .monotonic);
        state.workers_done.store(0, .monotonic);
        _ = state.epoch.fetchAdd(1, .seq_cst);

        if (state.sleeping_workers.load(.seq_cst) > 0) {
            _ = std.c.pthread_mutex_lock(&state.wake_mutex);
            _ = std.c.pthread_cond_broadcast(&state.wake_cond);
            _ = std.c.pthread_mutex_unlock(&state.wake_mutex);
        }

        runClaimLoop(state);

        const expected_bg: u32 = @intCast(state.bg_count);
        while (state.completed_tiles.load(.acquire) < total_tiles or
            state.workers_done.load(.acquire) < expected_bg)
        {
            std.atomic.spinLoopHint();
        }
    }
};

// ============================================================================
// M6 W17: `.amf` `TensorEntry` Automatic `QuantParamKind` Dispatch
// ============================================================================

pub fn panelDTypeFromAmf(dtype: amf.DType) Error!PanelDType {
    return switch (dtype) {
        .f16 => .f16,
        .bf16 => .bf16,
        .q4_affine => .q4_affine,
        .q3_affine => .q3_affine,
        .q2_affine => .q2_affine,
        .f32, .u32 => error.InvalidDType,
    };
}

pub fn quantParamKindFromAmf(pk: amf.QuantParamKind) QuantParamKind {
    return switch (pk) {
        .affine_scale_bias => .affine_scale_bias,
        .lut16_codebook => .lut16_codebook,
    };
}

/// Validated zero-copy view of a wired `sme2_panel64` `.amf` tensor blob.
/// Automatically inspects `entry.quantParamKind()` (`reserved0 == 0x0001` vs
/// `0x0101` `lut16_codebook`) so quantized LUT16 tensors dispatch without
/// caller boilerplate (ADR-0017 / ADR-0018 / ADR-0019 W17).
pub const SmeLinearWeight = struct {
    dtype: PanelDType,
    param_kind: QuantParamKind,
    cache_hint: CacheHint,
    n: usize,
    k: usize,
    w_packed: []const u8,

    pub fn fromAmfEntry(
        entry: *const amf.TensorEntry,
        blob: []const u8,
        cache_hint_override: ?CacheHint,
    ) (Error || amf.ParseError)!SmeLinearWeight {
        const layout = try entry.tensorLayout();
        if (layout != .sme2_panel64) return error.InvalidTensorLayout;
        if (entry.rank != 2) return error.InvalidRank;

        const kind = try entry.tensorKind();
        const amf_dt = try entry.dataType();
        const dt = try panelDTypeFromAmf(amf_dt);
        const pk = quantParamKindFromAmf(try entry.quantParamKind());
        if (!dt.isQuantized() and pk != .affine_scale_bias) {
            return error.InvalidQuantParamKind;
        }

        const n: usize = entry.shape[0];
        const k: usize = entry.shape[1];
        const need = try packedBlobBytes(dt, n, k);
        if (entry.data_bytes != need or blob.len < need) return error.BufferTooSmall;

        const default_hint: CacheHint = if (kind == .cold_expert) .non_temporal else .temporal;
        return .{
            .dtype = dt,
            .param_kind = pk,
            .cache_hint = cache_hint_override orelse default_hint,
            .n = n,
            .k = k,
            .w_packed = blob[0..need],
        };
    }

    pub fn byteSize(self: SmeLinearWeight) usize {
        return self.w_packed.len;
    }
};

pub const LinearEvalOptions = struct {
    cache_hint: ?CacheHint = null,
    epilogue: EpilogueOp = .none,
    out_dtype: OutputDType = .f32,
    backend: ?ExecutionBackend = null,
    max_threads: ?usize = null,
};

/// Executes `Out[M, N] = epilogue(A[M, K] @ W^T, aux)` over a validated
/// `SmeLinearWeight`, automatically preserving `weight.param_kind`
/// (`affine_scale_bias` vs `lut16_codebook`) across `M = 1..64+` (`ld1b` and `ldnt1b`).
pub fn evalSmeLinear(
    dispatcher: *SmeDispatcher,
    weight: SmeLinearWeight,
    a_u16: []const u16,
    aux_f32: ?[]const f32,
    out_bytes: []u8,
    m: usize,
    opts: LinearEvalOptions,
) Error!void {
    return dispatcher.gemm(
        .{
            .dtype = weight.dtype,
            .param_kind = weight.param_kind,
            .cache_hint = opts.cache_hint orelse weight.cache_hint,
            .epilogue = opts.epilogue,
            .out_dtype = opts.out_dtype,
            .backend = opts.backend,
            .max_threads = opts.max_threads,
        },
        a_u16,
        weight.w_packed,
        aux_f32,
        out_bytes,
        m,
        weight.n,
        weight.k,
    );
}

/// One-shot `.amf` `TensorEntry` SME2 linear dispatch: inspects `entry.quantParamKind()`
/// (`0x0001` vs `0x0101`) and `entry.dataType()` automatically and evaluates across
/// `M = 1..64+` with zero hot-path heap allocations.
pub fn evalAmfLinear(
    dispatcher: *SmeDispatcher,
    entry: *const amf.TensorEntry,
    blob: []const u8,
    a_u16: []const u16,
    aux_f32: ?[]const f32,
    out_bytes: []u8,
    m: usize,
    opts: LinearEvalOptions,
) (Error || amf.ParseError)!void {
    const weight = try SmeLinearWeight.fromAmfEntry(entry, blob, opts.cache_hint);
    return evalSmeLinear(dispatcher, weight, a_u16, aux_f32, out_bytes, m, opts);
}

// ============================================================================
// M6 W16: Zero-Allocation L2-Resident `SmeStepLoraAdapter` (ADR-0019)
// ============================================================================

pub const MAX_STEP_LORA_RANK: usize = GROUP_SIZE; // 64
pub const MAX_STEP_LORA_M: usize = 64;
pub const MAX_L2_DRAFT_PROJ_BYTES: usize = 16 * 1024 * 1024; // 16 MiB P-cluster L2

pub const WiredScratchError = Error || arena_mod.WiredArena.InitError ||
    arena_mod.WiredArena.AllocError || sysinfo.Error || error{ExceedsL2Budget};

fn defaultWiredBudget() sysinfo.Error!budget_mod.Budget {
    const facts = try sysinfo.Facts.read();
    return budget_mod.Budget.compute(.{
        .memory_bytes = facts.memory_bytes,
        .user_wire_limit = facts.user_wire_limit,
        .global_user_wire_limit = facts.global_user_wire_limit,
        .gpu_working_set_bytes = facts.memory_bytes,
        .configured_cap = null,
        .headroom_bytes = budget_mod.DEFAULT_HEADROOM_BYTES,
    });
}

/// Preallocated 16 KiB-aligned wired RAM scratch for `SmeStepLoraAdapter`:
/// holds `[max_m * max_rank]f32` intermediate activations, `[max_m * 64]u16`
/// scaled & K-padded inputs for `B_r`, and an optional `[max_m * max_k]u16`
/// conversion buffer when evaluating from `f32` inputs. Zero hot-path allocs.
pub const StepLoraScratch = struct {
    arena: arena_mod.WiredArena,
    z_f32: []align(SVL_BYTES) f32,
    z_act_u16: []align(SVL_BYTES) u16,
    x_act_u16: []align(SVL_BYTES) u16,
    max_m: usize,
    max_rank: usize,
    max_k: usize,

    pub fn init(
        max_m: usize,
        max_rank: usize,
        max_k: usize,
        budget_override: ?budget_mod.Budget,
    ) WiredScratchError!StepLoraScratch {
        if (max_m == 0 or max_m > MAX_STEP_LORA_M) return error.InvalidArgument;
        if (max_rank == 0 or max_rank > MAX_STEP_LORA_RANK) return error.InvalidArgument;
        if (max_k > 0 and (max_k % GROUP_SIZE) != 0) return error.UnalignedK;

        const z_f32_elems = std.math.mul(usize, max_m, max_rank) catch return error.Overflow;
        const z_f32_bytes = std.math.mul(usize, z_f32_elems, @sizeOf(f32)) catch return error.Overflow;
        const z_u16_elems = std.math.mul(usize, max_m, GROUP_SIZE) catch return error.Overflow;
        const z_u16_bytes = std.math.mul(usize, z_u16_elems, @sizeOf(u16)) catch return error.Overflow;
        const x_u16_elems = std.math.mul(usize, max_m, max_k) catch return error.Overflow;
        const x_u16_bytes = std.math.mul(usize, x_u16_elems, @sizeOf(u16)) catch return error.Overflow;

        const seg0 = amf.align64(z_f32_bytes);
        const seg1 = amf.align64(z_u16_bytes);
        const seg2 = amf.align64(x_u16_bytes);
        const raw_total = std.math.add(usize, std.math.add(usize, seg0, seg1) catch return error.Overflow, seg2) catch return error.Overflow;
        const aligned_total = page.alignUp(raw_total) catch return error.Overflow;

        const wired_budget = budget_override orelse try defaultWiredBudget();
        try wired_budget.checkPlan(aligned_total);

        var mem_arena = try arena_mod.WiredArena.init(aligned_total, wired_budget);
        errdefer mem_arena.deinit();

        const z_f32_raw = try mem_arena.allocAligned(z_f32_bytes, SVL_BYTES);
        const z_u16_raw = try mem_arena.allocAligned(z_u16_bytes, SVL_BYTES);
        const x_u16_slice: []align(SVL_BYTES) u16 = if (x_u16_bytes > 0) blk: {
            const raw = try mem_arena.allocAligned(x_u16_bytes, SVL_BYTES);
            break :blk @alignCast(std.mem.bytesAsSlice(u16, raw));
        } else &.{};

        return .{
            .arena = mem_arena,
            .z_f32 = @alignCast(std.mem.bytesAsSlice(f32, z_f32_raw)),
            .z_act_u16 = @alignCast(std.mem.bytesAsSlice(u16, z_u16_raw)),
            .x_act_u16 = x_u16_slice,
            .max_m = max_m,
            .max_rank = max_rank,
            .max_k = max_k,
        };
    }

    pub fn deinit(self: *StepLoraScratch) void {
        self.z_f32 = &.{};
        self.z_act_u16 = &.{};
        self.x_act_u16 = &.{};
        self.arena.deinit();
    }

    pub fn totalWiredBytes(self: *const StepLoraScratch) u64 {
        return self.arena.bytes().len;
    }
};

/// Packs a logical `[n, rank]` (`1 <= rank <= 64`) dense `f16`/`bf16` Step-LoRA
/// `B_r` matrix into a valid `[n, 64]` `sme2_panel64` blob by zero-padding
/// columns `rank..64`.
pub fn packStepLoraBDensePanel64(
    dtype: PanelDType,
    src_row_major_u16: []const u16,
    n: usize,
    rank: usize,
    dst: []u8,
) Error!void {
    if (dtype.isQuantized()) return error.InvalidDType;
    if (n == 0 or rank == 0) return error.ZeroDimension;
    if (rank > MAX_STEP_LORA_RANK) return error.InvalidArgument;
    const total_bytes = try packedBlobBytes(dtype, n, GROUP_SIZE);
    const nr = std.math.mul(usize, n, rank) catch return error.Overflow;
    if (src_row_major_u16.len < nr or dst.len < total_bytes) return error.BufferTooSmall;

    const panels = panelCount(n);
    const g_bytes = groupBytes(dtype);
    var out_off: usize = 0;
    for (0..panels) |p| {
        const grp_slice = dst[out_off .. out_off + g_bytes];
        for (0..32) |kp| {
            const k0 = 2 * kp;
            const k1 = k0 + 1;
            const chunk_off = kp * 64;
            for (0..PANEL_COLS) |c| {
                const col = p * PANEL_COLS + c;
                const v0: u16 = if (col < n and k0 < rank) src_row_major_u16[col * rank + k0] else 0;
                const v1: u16 = if (col < n and k1 < rank) src_row_major_u16[col * rank + k1] else 0;
                std.mem.writeInt(u16, grp_slice[chunk_off + c * 4 + 0 ..][0..2], v0, .little);
                std.mem.writeInt(u16, grp_slice[chunk_off + c * 4 + 2 ..][0..2], v1, .little);
            }
        }
        out_off += g_bytes;
    }
}

/// Packs a logical `[n, rank]` (`1 <= rank <= 64`) `f32` Step-LoRA `B_r` matrix
/// into a valid `[n, 64]` `f16`/`bf16` `sme2_panel64` blob (cols `rank..64` zeroed).
pub fn packStepLoraBDenseFromF32Panel64(
    dtype: PanelDType,
    src_f32: []const f32,
    n: usize,
    rank: usize,
    dst: []u8,
) Error!void {
    if (dtype.isQuantized()) return error.InvalidDType;
    if (n == 0 or rank == 0) return error.ZeroDimension;
    if (rank > MAX_STEP_LORA_RANK) return error.InvalidArgument;
    const total_bytes = try packedBlobBytes(dtype, n, GROUP_SIZE);
    const nr = std.math.mul(usize, n, rank) catch return error.Overflow;
    if (src_f32.len < nr or dst.len < total_bytes) return error.BufferTooSmall;

    const panels = panelCount(n);
    const g_bytes = groupBytes(dtype);
    const fdt: quant.FloatDType = if (dtype == .bf16) .bf16 else .f16;
    var out_off: usize = 0;
    for (0..panels) |p| {
        const grp_slice = dst[out_off .. out_off + g_bytes];
        for (0..32) |kp| {
            const k0 = 2 * kp;
            const k1 = k0 + 1;
            const chunk_off = kp * 64;
            for (0..PANEL_COLS) |c| {
                const col = p * PANEL_COLS + c;
                const v0: u16 = if (col < n and k0 < rank) quant.f32ToFloatBits(fdt, src_f32[col * rank + k0]) else 0;
                const v1: u16 = if (col < n and k1 < rank) quant.f32ToFloatBits(fdt, src_f32[col * rank + k1]) else 0;
                std.mem.writeInt(u16, grp_slice[chunk_off + c * 4 + 0 ..][0..2], v0, .little);
                std.mem.writeInt(u16, grp_slice[chunk_off + c * 4 + 2 ..][0..2], v1, .little);
            }
        }
        out_off += g_bytes;
    }
}

/// Reads scalar or per-step vector `alpha_r` from a wired `.amf` `loop_adapter`
/// 1D tensor (`f32`, `f16`, or `bf16`, `shape = [1]` or `[max_loops]`).
pub fn readLoopAlphaFromAmf(
    entry: *const amf.TensorEntry,
    blob: []const u8,
    loop_step: usize,
) (Error || amf.ParseError)!f32 {
    if ((try entry.tensorKind()) != .loop_adapter) return error.InvalidTensorKind;
    if ((try entry.tensorLayout()) != .row_major) return error.InvalidTensorLayout;
    if (entry.rank != 1) return error.InvalidRank;
    const len: usize = entry.shape[0];
    if (len == 0) return error.InvalidShape;
    const idx: usize = if (len == 1) 0 else loop_step;
    if (idx >= len) return error.InvalidArgument;

    const dt = try entry.dataType();
    return switch (dt) {
        .f32 => blk: {
            if (blob.len < (idx + 1) * 4) return error.BufferTooSmall;
            const bits = std.mem.readInt(u32, blob[idx * 4 ..][0..4], .little);
            break :blk @bitCast(bits);
        },
        .f16 => blk: {
            if (blob.len < (idx + 1) * 2) return error.BufferTooSmall;
            const bits = std.mem.readInt(u16, blob[idx * 2 ..][0..2], .little);
            break :blk @floatCast(@as(f16, @bitCast(bits)));
        },
        .bf16 => blk: {
            if (blob.len < (idx + 1) * 2) return error.BufferTooSmall;
            const bits = std.mem.readInt(u16, blob[idx * 2 ..][0..2], .little);
            break :blk quant.bf16ToF32(bits);
        },
        else => error.InvalidDType,
    };
}

/// Zero-allocation L2-resident (`ld1b`) `Step-LoRA` adapter for recurrent depth
/// passes (`ADR-0019`, `W16`):
///   `y[m, :] += alpha_r * ((x[m, :] @ A_r^T) @ B_r^T)`
///
/// Enforces an exact fast-return no-op when `alpha == 0.0`, guaranteeing
/// bit-exact `r = 0` identity on frozen checkpoints.
pub const SmeStepLoraAdapter = struct {
    a_weight: SmeLinearWeight, // [rank, K] in sme2_panel64 (ld1b)
    b_weight: SmeLinearWeight, // [N, 64] in sme2_panel64 (cols rank..64 zeroed, ld1b)
    alpha: f32,
    rank: usize,
    n: usize,
    k: usize,

    pub fn fromWeights(
        a_weight: SmeLinearWeight,
        b_weight: SmeLinearWeight,
        alpha: f32,
    ) Error!SmeStepLoraAdapter {
        if (a_weight.n == 0 or a_weight.n > MAX_STEP_LORA_RANK) return error.InvalidArgument;
        if (a_weight.k == 0 or (a_weight.k % GROUP_SIZE) != 0) return error.UnalignedK;
        if (b_weight.n == 0) return error.ZeroDimension;
        // SME2 panel64 requires K % 64 == 0; for rank <= 64, B_r is padded to K = 64.
        if (b_weight.k != GROUP_SIZE) return error.InvalidArgument;

        var a = a_weight;
        var b = b_weight;
        a.cache_hint = .temporal;
        b.cache_hint = .temporal;
        return .{
            .a_weight = a,
            .b_weight = b,
            .alpha = alpha,
            .rank = a.n,
            .n = b.n,
            .k = a.k,
        };
    }

    pub fn fromAmfEntries(
        a_entry: *const amf.TensorEntry,
        a_blob: []const u8,
        b_entry: *const amf.TensorEntry,
        b_blob: []const u8,
        alpha: f32,
    ) (Error || amf.ParseError)!SmeStepLoraAdapter {
        if ((try a_entry.tensorKind()) != .loop_adapter) return error.InvalidTensorKind;
        if ((try b_entry.tensorKind()) != .loop_adapter) return error.InvalidTensorKind;
        const a_w = try SmeLinearWeight.fromAmfEntry(a_entry, a_blob, .temporal);
        const b_w = try SmeLinearWeight.fromAmfEntry(b_entry, b_blob, .temporal);
        return fromWeights(a_w, b_w, alpha);
    }

    pub fn withAlpha(self: SmeStepLoraAdapter, alpha: f32) SmeStepLoraAdapter {
        var copy = self;
        copy.alpha = alpha;
        return copy;
    }

    pub fn totalWeightBytes(self: SmeStepLoraAdapter) usize {
        return self.a_weight.byteSize() + self.b_weight.byteSize();
    }

    /// Evaluates `y_f32[m, :] += alpha * ((x_u16[m, :] @ A^T) @ B^T)` in-place
    /// using preallocated wired `scratch` (`0` heap allocs, L2 `ld1b`).
    /// Fast-returns with zero writes when `self.alpha == 0.0`.
    pub fn evalU16(
        self: SmeStepLoraAdapter,
        dispatcher: *SmeDispatcher,
        scratch: *StepLoraScratch,
        x_u16: []const u16,
        y_f32: []f32,
        m: usize,
        backend_override: ?ExecutionBackend,
        max_threads: ?usize,
    ) Error!void {
        if (m == 0 or m > MAX_STEP_LORA_M or m > scratch.max_m or self.rank > scratch.max_rank) {
            return error.InvalidArgument;
        }
        const mk = std.math.mul(usize, m, self.k) catch return error.Overflow;
        const mn = std.math.mul(usize, m, self.n) catch return error.Overflow;
        if (x_u16.len < mk or y_f32.len < mn) return error.BufferTooSmall;

        // Mandatory ADR-0019 r = 0 bit-exact identity fast path!
        if (self.alpha == 0.0) return;

        // 1. z_f32[M, rank] = x_u16[M, K] @ A^T (L2 temporal `ld1b`).
        const z_f32 = scratch.z_f32[0 .. m * self.rank];
        try evalSmeLinear(
            dispatcher,
            self.a_weight,
            x_u16[0..mk],
            null,
            std.mem.sliceAsBytes(z_f32),
            m,
            .{
                .cache_hint = .temporal,
                .epilogue = .none,
                .out_dtype = .f32,
                .backend = backend_override,
                .max_threads = max_threads,
            },
        );

        // 2. Scale by `alpha`, convert to B's activation dtype, and zero-pad [rank..64].
        const b_act_dt: quant.FloatDType = if (self.b_weight.dtype.activationIsBf16()) .bf16 else .f16;
        const z_u16 = scratch.z_act_u16[0 .. m * GROUP_SIZE];
        for (0..m) |row| {
            const src_row = z_f32[row * self.rank ..][0..self.rank];
            const dst_row = z_u16[row * GROUP_SIZE ..][0..GROUP_SIZE];
            for (0..self.rank) |j| {
                dst_row[j] = quant.f32ToFloatBits(b_act_dt, src_row[j] * self.alpha);
            }
            @memset(dst_row[self.rank..GROUP_SIZE], 0);
        }

        // 3. y_f32[M, N] += z_u16[M, 64] @ B^T via fused in-register `add_residual` epilogue.
        try evalSmeLinear(
            dispatcher,
            self.b_weight,
            z_u16,
            y_f32[0..mn],
            std.mem.sliceAsBytes(y_f32[0..mn]),
            m,
            .{
                .cache_hint = .temporal,
                .epilogue = .add_residual,
                .out_dtype = .f32,
                .backend = backend_override,
                .max_threads = max_threads,
            },
        );
    }

    /// Convenience FP32 input wrapper using `scratch.x_act_u16` (requires `scratch.max_k >= k`).
    pub fn evalF32(
        self: SmeStepLoraAdapter,
        dispatcher: *SmeDispatcher,
        scratch: *StepLoraScratch,
        x_f32: []const f32,
        y_f32: []f32,
        m: usize,
        backend_override: ?ExecutionBackend,
        max_threads: ?usize,
    ) Error!void {
        if (m == 0 or m > MAX_STEP_LORA_M or m > scratch.max_m or self.rank > scratch.max_rank) {
            return error.InvalidArgument;
        }
        const mk = std.math.mul(usize, m, self.k) catch return error.Overflow;
        const mn = std.math.mul(usize, m, self.n) catch return error.Overflow;
        if (x_f32.len < mk or y_f32.len < mn) return error.BufferTooSmall;
        if (self.alpha == 0.0) return;
        if (scratch.x_act_u16.len < mk) return error.BufferTooSmall;

        const a_act_dt: quant.FloatDType = if (self.a_weight.dtype.activationIsBf16()) .bf16 else .f16;
        for (0..mk) |i| {
            scratch.x_act_u16[i] = quant.f32ToFloatBits(a_act_dt, x_f32[i]);
        }
        return self.evalU16(dispatcher, scratch, scratch.x_act_u16[0..mk], y_f32, m, backend_override, max_threads);
    }
};

// ============================================================================
// M6 D13: L2-Resident `SmeDraftProjection` & Benchmark Gate (ADR-0008 / ADR-0019)
// ============================================================================

pub const NormWeightSlice = union(enum) {
    f32: []const f32,
    f16: []const f16,
    bf16: []const u16,

    pub fn len(self: NormWeightSlice) usize {
        return switch (self) {
            .f32 => |s| s.len,
            .f16 => |s| s.len,
            .bf16 => |s| s.len,
        };
    }

    pub inline fn atF32(self: NormWeightSlice, idx: usize) f32 {
        return switch (self) {
            .f32 => |s| s[idx],
            .f16 => |s| @floatCast(s[idx]),
            .bf16 => |s| quant.bf16ToF32(s[idx]),
        };
    }
};

pub const InputSlice = union(enum) {
    f32: []const f32,
    f16: []const f16,
    bf16: []const u16,

    pub fn len(self: InputSlice) usize {
        return switch (self) {
            .f32 => |s| s.len,
            .f16 => |s| s.len,
            .bf16 => |s| s.len,
        };
    }

    pub inline fn atF32(self: InputSlice, idx: usize) f32 {
        return switch (self) {
            .f32 => |s| s[idx],
            .f16 => |s| @floatCast(s[idx]),
            .bf16 => |s| quant.bf16ToF32(s[idx]),
        };
    }
};

fn rmsNormRowIntoU16(
    x: InputSlice,
    row_offset: usize,
    dim: usize,
    norm_w: ?NormWeightSlice,
    rms_eps: f32,
    out_dt: quant.FloatDType,
    dst: []u16,
) void {
    var sum_sq: f32 = 0.0;
    for (0..dim) |d| {
        const v = x.atF32(row_offset + d);
        sum_sq += v * v;
    }
    const inv_rms: f32 = 1.0 / @sqrt(sum_sq / @as(f32, @floatFromInt(dim)) + rms_eps);
    if (norm_w) |nw| {
        for (0..dim) |d| {
            const v = (x.atF32(row_offset + d) * inv_rms) * nw.atF32(d);
            dst[d] = quant.f32ToFloatBits(out_dt, v);
        }
    } else {
        for (0..dim) |d| {
            const v = x.atF32(row_offset + d) * inv_rms;
            dst[d] = quant.f32ToFloatBits(out_dt, v);
        }
    }
}

pub const DraftProjectionConfig = struct {
    max_m: usize = 64,
    l2_limit_bytes: usize = MAX_L2_DRAFT_PROJ_BYTES,
    budget_override: ?budget_mod.Budget = null,
};

/// Zero-allocation L2-resident (`<= 16 MiB`, `ld1b`) P-core SME2 runner for
/// compact `DraftHead` (`Placement.cpu_sme2`) projections: `EAGLE-3` `fc`
/// 3-tap fusion, `4B MTP` dual-norm `eh_proj`, and fused RMSNorm + Linear.
/// Executes with `0` Metal command buffers and `0` hot-path heap allocations.
pub const SmeDraftProjection = struct {
    weight: SmeLinearWeight,
    arena: arena_mod.WiredArena,
    x_act_u16: []align(SVL_BYTES) u16,
    max_m: usize,
    l2_limit_bytes: usize,

    pub fn init(weight_in: SmeLinearWeight, config: DraftProjectionConfig) WiredScratchError!SmeDraftProjection {
        if (config.max_m == 0 or config.max_m > MAX_STEP_LORA_M) return error.InvalidArgument;
        if (weight_in.byteSize() > config.l2_limit_bytes) return error.ExceedsL2Budget;

        var weight = weight_in;
        weight.cache_hint = .temporal; // L2-resident `ld1b`

        const elems = std.math.mul(usize, config.max_m, weight.k) catch return error.Overflow;
        const raw_bytes = std.math.mul(usize, elems, @sizeOf(u16)) catch return error.Overflow;
        const aligned_bytes = page.alignUp(raw_bytes) catch return error.Overflow;

        const wired_budget = config.budget_override orelse try defaultWiredBudget();
        try wired_budget.checkPlan(aligned_bytes);

        var mem_arena = try arena_mod.WiredArena.init(aligned_bytes, wired_budget);
        errdefer mem_arena.deinit();

        const raw_slice = try mem_arena.allocAligned(raw_bytes, SVL_BYTES);
        const x_act_u16: []align(SVL_BYTES) u16 = @alignCast(std.mem.bytesAsSlice(u16, raw_slice));
        @memset(x_act_u16, 0);

        return .{
            .weight = weight,
            .arena = mem_arena,
            .x_act_u16 = x_act_u16,
            .max_m = config.max_m,
            .l2_limit_bytes = config.l2_limit_bytes,
        };
    }

    pub fn fromAmfEntry(
        entry: *const amf.TensorEntry,
        blob: []const u8,
        config: DraftProjectionConfig,
    ) (WiredScratchError || amf.ParseError)!SmeDraftProjection {
        const w = try SmeLinearWeight.fromAmfEntry(entry, blob, .temporal);
        return init(w, config);
    }

    pub fn deinit(self: *SmeDraftProjection) void {
        self.x_act_u16 = &.{};
        self.arena.deinit();
    }

    pub fn fitsL2(self: *const SmeDraftProjection, l2_limit_override: ?usize) bool {
        return self.weight.byteSize() <= (l2_limit_override orelse self.l2_limit_bytes);
    }

    pub fn totalWiredBytes(self: *const SmeDraftProjection) u64 {
        return self.arena.bytes().len;
    }

    /// Evaluates already-packed `[m, k]` activations directly on SME2 (`ld1b`).
    pub fn evalU16(
        self: *const SmeDraftProjection,
        dispatcher: *SmeDispatcher,
        x_u16: []const u16,
        aux_f32: ?[]const f32,
        out_bytes: []u8,
        m: usize,
        opts: LinearEvalOptions,
    ) Error!void {
        if (m == 0 or m > self.max_m) return error.InvalidArgument;
        var o = opts;
        o.cache_hint = .temporal;
        return evalSmeLinear(dispatcher, self.weight, x_u16, aux_f32, out_bytes, m, o);
    }

    /// Fused `RMSNorm(x, norm_weight, rms_eps) @ W^T` on P-core SME2 (`ld1b`,
    /// `0` Metal CBs, `0` heap allocs).
    pub fn evalFusedNormLinear(
        self: *SmeDraftProjection,
        dispatcher: *SmeDispatcher,
        x: InputSlice,
        norm_weight: ?NormWeightSlice,
        rms_eps: f32,
        aux_f32: ?[]const f32,
        out_bytes: []u8,
        m: usize,
        opts: LinearEvalOptions,
    ) Error!void {
        if (m == 0 or m > self.max_m) return error.InvalidArgument;
        const k = self.weight.k;
        const mk = std.math.mul(usize, m, k) catch return error.Overflow;
        if (x.len() < mk) return error.BufferTooSmall;
        if (norm_weight) |nw| {
            if (nw.len() < k) return error.BufferTooSmall;
        }

        const act_dt: quant.FloatDType = if (self.weight.dtype.activationIsBf16()) .bf16 else .f16;
        for (0..m) |r| {
            rmsNormRowIntoU16(x, r * k, k, norm_weight, rms_eps, act_dt, self.x_act_u16[r * k ..][0..k]);
        }

        var o = opts;
        o.cache_hint = .temporal;
        return evalSmeLinear(dispatcher, self.weight, self.x_act_u16[0..mk], aux_f32, out_bytes, m, o);
    }

    /// 4B MTP dual-norm fusion: `concat(RMSNorm(e, enorm_w), RMSNorm(h, hnorm_w)) @ eh_proj^T`
    /// where `eh_proj` has shape `[H, 2 * H]` (`0` Metal CBs, `0` heap allocs).
    pub fn evalDualNormConcatLinear(
        self: *SmeDraftProjection,
        dispatcher: *SmeDispatcher,
        e_in: InputSlice,
        enorm_w: ?NormWeightSlice,
        h_in: InputSlice,
        hnorm_w: ?NormWeightSlice,
        rms_eps: f32,
        aux_f32: ?[]const f32,
        out_bytes: []u8,
        m: usize,
        opts: LinearEvalOptions,
    ) Error!void {
        if (m == 0 or m > self.max_m) return error.InvalidArgument;
        if ((self.weight.k % 2) != 0) return error.InvalidArgument;
        const k = self.weight.k;
        const half_k = k / 2;
        const m_half = std.math.mul(usize, m, half_k) catch return error.Overflow;
        if (e_in.len() < m_half or h_in.len() < m_half) return error.BufferTooSmall;
        if (enorm_w) |ew| {
            if (ew.len() < half_k) return error.BufferTooSmall;
        }
        if (hnorm_w) |hw| {
            if (hw.len() < half_k) return error.BufferTooSmall;
        }

        const act_dt: quant.FloatDType = if (self.weight.dtype.activationIsBf16()) .bf16 else .f16;
        for (0..m) |r| {
            const dst = self.x_act_u16[r * k ..][0..k];
            rmsNormRowIntoU16(e_in, r * half_k, half_k, enorm_w, rms_eps, act_dt, dst[0..half_k]);
            rmsNormRowIntoU16(h_in, r * half_k, half_k, hnorm_w, rms_eps, act_dt, dst[half_k..k]);
        }

        var o = opts;
        o.cache_hint = .temporal;
        return evalSmeLinear(dispatcher, self.weight, self.x_act_u16[0 .. m * k], aux_f32, out_bytes, m, o);
    }

    /// EAGLE-3 3-tap `fc` fusion: `concat(h_low, h_mid, h_high) @ fc^T` where
    /// `fc` has shape `[H, 3 * H_tgt]` (`0` Metal CBs, `0` heap allocs).
    pub fn evalEagle3FcFusion(
        self: *SmeDraftProjection,
        dispatcher: *SmeDispatcher,
        taps: [3]InputSlice,
        aux_f32: ?[]const f32,
        out_bytes: []u8,
        m: usize,
        opts: LinearEvalOptions,
    ) Error!void {
        if (m == 0 or m > self.max_m) return error.InvalidArgument;
        if ((self.weight.k % 3) != 0) return error.InvalidArgument;
        const k = self.weight.k;
        const h_tgt = k / 3;
        const m_tap = std.math.mul(usize, m, h_tgt) catch return error.Overflow;
        for (taps) |tap| {
            if (tap.len() < m_tap) return error.BufferTooSmall;
        }

        const act_dt: quant.FloatDType = if (self.weight.dtype.activationIsBf16()) .bf16 else .f16;
        for (0..m) |r| {
            const dst_row = self.x_act_u16[r * k ..][0..k];
            for (0..3) |t| {
                const src_off = r * h_tgt;
                const dst_seg = dst_row[t * h_tgt ..][0..h_tgt];
                for (0..h_tgt) |d| {
                    dst_seg[d] = quant.f32ToFloatBits(act_dt, taps[t].atF32(src_off + d));
                }
            }
        }

        var o = opts;
        o.cache_hint = .temporal;
        return evalSmeLinear(dispatcher, self.weight, self.x_act_u16[0 .. m * k], aux_f32, out_bytes, m, o);
    }
};

pub const MAX_CALIBRATE_ITERS: usize = 64;

pub const CalibrateConfig = struct {
    warmups: u32 = 4,
    iters: u32 = 16,
    out_dtype: OutputDType = .f32,
    backend: ?ExecutionBackend = null,
    max_threads: ?usize = null,
    l2_limit_bytes: ?usize = null,
    /// `prefer_sme2` is true iff `fits_l2` and `sme_us < gpu_us * sme_win_ratio`.
    sme_win_ratio: f64 = 1.0,
};

pub const SmeVsGpuCalibration = struct {
    sme_us: f64,
    gpu_us: f64,
    prefer_sme2: bool,
    weight_bytes: usize,
    fits_l2: bool,
    iters: u32,
};

fn medianNsToUs(samples: []u64) f64 {
    std.mem.sort(u64, samples, {}, std.sort.asc(u64));
    const mid = samples.len / 2;
    const med_ns: f64 = if ((samples.len & 1) == 1)
        @floatFromInt(samples[mid])
    else
        0.5 * (@as(f64, @floatFromInt(samples[mid - 1])) + @as(f64, @floatFromInt(samples[mid])));
    return med_ns / 1000.0;
}

/// D13 startup benchmark-gate helper (ADR-0008 / ADR-0019): measures median
/// P-core SME2 latency (`sme_us`) for `proj` at batch width `m` with zero heap
/// allocations and compares against a caller-measured `gpu_us` baseline.
pub fn calibrateSmeVsGpuUs(
    dispatcher: *SmeDispatcher,
    proj: *SmeDraftProjection,
    m: usize,
    out_bytes: []u8,
    gpu_us: f64,
    config: CalibrateConfig,
) (Error || vmstat.Error)!SmeVsGpuCalibration {
    if (m == 0 or m > proj.max_m) return error.InvalidArgument;
    if (config.iters == 0 or config.iters > MAX_CALIBRATE_ITERS) return error.InvalidArgument;

    const mk = std.math.mul(usize, m, proj.weight.k) catch return error.Overflow;
    const x_u16 = proj.x_act_u16[0..mk];
    const eval_opts: LinearEvalOptions = .{
        .cache_hint = .temporal,
        .epilogue = .none,
        .out_dtype = config.out_dtype,
        .backend = config.backend,
        .max_threads = config.max_threads,
    };

    for (0..config.warmups) |_| {
        try proj.evalU16(dispatcher, x_u16, null, out_bytes, m, eval_opts);
    }

    var samples_ns: [MAX_CALIBRATE_ITERS]u64 = undefined;
    const n_iters: usize = config.iters;
    for (0..n_iters) |i| {
        const t0 = try vmstat.nowNs();
        try proj.evalU16(dispatcher, x_u16, null, out_bytes, m, eval_opts);
        const t1 = try vmstat.nowNs();
        samples_ns[i] = t1 -| t0;
    }

    const sme_us = medianNsToUs(samples_ns[0..n_iters]);
    const fits_l2 = proj.fitsL2(config.l2_limit_bytes);
    return .{
        .sme_us = sme_us,
        .gpu_us = gpu_us,
        .prefer_sme2 = fits_l2 and (sme_us < gpu_us * config.sme_win_ratio),
        .weight_bytes = proj.weight.byteSize(),
        .fits_l2 = fits_l2,
        .iters = config.iters,
    };
}

/// D13 callback variant: times both P-core SME2 (`proj`) and a caller-supplied
/// GPU execution callback with identical warmup + median-of-N loops.
pub fn calibrateSmeVsGpuFn(
    dispatcher: *SmeDispatcher,
    proj: *SmeDraftProjection,
    m: usize,
    out_bytes: []u8,
    gpu_ctx: *anyopaque,
    gpu_step_fn: *const fn (*anyopaque) anyerror!void,
    config: CalibrateConfig,
) anyerror!SmeVsGpuCalibration {
    if (config.iters == 0 or config.iters > MAX_CALIBRATE_ITERS) return error.InvalidArgument;

    for (0..config.warmups) |_| {
        try gpu_step_fn(gpu_ctx);
    }
    var gpu_samples_ns: [MAX_CALIBRATE_ITERS]u64 = undefined;
    const n_iters: usize = config.iters;
    for (0..n_iters) |i| {
        const t0 = try vmstat.nowNs();
        try gpu_step_fn(gpu_ctx);
        const t1 = try vmstat.nowNs();
        gpu_samples_ns[i] = t1 -| t0;
    }
    const gpu_us = medianNsToUs(gpu_samples_ns[0..n_iters]);
    return calibrateSmeVsGpuUs(dispatcher, proj, m, out_bytes, gpu_us, config);
}

// ============================================================================
// Mandatory AAPCS64 Register-Clobber Canary Harness (ADR-0018 Safeguard B.1)
// ============================================================================

/// 18 distinct non-zero 64-bit sentinel bit patterns loaded into AAPCS64
/// callee-saved `x19..x28` (indices 0..9) and `d8..d15` (indices 10..17)
/// immediately before invoking a `.s` SME2 entrypoint.
pub const CANARY_SENTINELS = [18]u64{
    0xD15EA5E0_19191919, // x19
    0xD15EA5E0_20202020, // x20
    0xD15EA5E0_21212121, // x21
    0xD15EA5E0_22222222, // x22
    0xD15EA5E0_23232323, // x23
    0xD15EA5E0_24242424, // x24
    0xD15EA5E0_25252525, // x25
    0xD15EA5E0_26262626, // x26
    0xD15EA5E0_27272727, // x27
    0xD15EA5E0_28282828, // x28
    0xC0FFEE08_08080808, // d8
    0xC0FFEE09_09090909, // d9
    0xC0FFEE0A_0A0A0A0A, // d10
    0xC0FFEE0B_0B0B0B0B, // d11
    0xC0FFEE0C_0C0C0C0C, // d12
    0xC0FFEE0D_0D0D0D0D, // d13
    0xC0FFEE0E_0E0E0E0E, // d14
    0xC0FFEE0F_0F0F0F0F, // d15
};

pub const CANARY_REG_NAMES = [18][]const u8{
    "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28",
    "d8",  "d9",  "d10", "d11", "d12", "d13", "d14", "d15",
};

/// Invokes `kernel_fn(args)` under an inline-assembly AAPCS64 sentinel frame
/// that loads `CANARY_SENTINELS` into `x19..x28` and `d8..d15`, calls `kernel_fn`,
/// captures all 18 registers back into `out_after`, and restores the caller's
/// registers cleanly. Returns `error.RegisterClobbered` on any 1-bit mismatch.
pub noinline fn invokeWithRegisterCanary(
    kernel_fn: KernelFn,
    args: *const KernelArgs,
) error{RegisterClobbered}!void {
    var after: [18]u64 = @splat(0);
    var f_reg: usize = @intFromPtr(kernel_fn);
    var a_reg: usize = @intFromPtr(args);
    var s_reg: usize = @intFromPtr(&CANARY_SENTINELS);
    var o_reg: usize = @intFromPtr(&after);
    // Keep pointer to `after` in a stack-saved slot inside the inline asm block
    // so the callee cannot see or perturb it. Inputs are read-write (`+`) so
    // LLVM knows `x8..x11` are clobbered by `blr x8` and reloads them every call;
    // `v8..v15` are listed as clobbered so LLVM never caches 128-bit `q8..q15`
    // across the AAPCS64 call boundary (which only preserves low 64-bit `d8..d15`).
    asm volatile (
        \\sub sp, sp, #176
        \\stp x19, x20, [sp, #0]
        \\stp x21, x22, [sp, #16]
        \\stp x23, x24, [sp, #32]
        \\stp x25, x26, [sp, #48]
        \\stp x27, x28, [sp, #64]
        \\stp d8,  d9,  [sp, #80]
        \\stp d10, d11, [sp, #96]
        \\stp d12, d13, [sp, #112]
        \\stp d14, d15, [sp, #128]
        \\stp x29, x30, [sp, #144]
        \\str x11, [sp, #160]
        \\ldp x19, x20, [x10, #0]
        \\ldp x21, x22, [x10, #16]
        \\ldp x23, x24, [x10, #32]
        \\ldp x25, x26, [x10, #48]
        \\ldp x27, x28, [x10, #64]
        \\ldp d8,  d9,  [x10, #80]
        \\ldp d10, d11, [x10, #96]
        \\ldp d12, d13, [x10, #112]
        \\ldp d14, d15, [x10, #128]
        \\mov x0, x9
        \\blr x8
        \\ldr x11, [sp, #160]
        \\stp x19, x20, [x11, #0]
        \\stp x21, x22, [x11, #16]
        \\stp x23, x24, [x11, #32]
        \\stp x25, x26, [x11, #48]
        \\stp x27, x28, [x11, #64]
        \\stp d8,  d9,  [x11, #80]
        \\stp d10, d11, [x11, #96]
        \\stp d12, d13, [x11, #112]
        \\stp d14, d15, [x11, #128]
        \\ldp x29, x30, [sp, #144]
        \\ldp d14, d15, [sp, #128]
        \\ldp d12, d13, [sp, #112]
        \\ldp d10, d11, [sp, #96]
        \\ldp d8,  d9,  [sp, #80]
        \\ldp x27, x28, [sp, #64]
        \\ldp x25, x26, [sp, #48]
        \\ldp x23, x24, [sp, #32]
        \\ldp x21, x22, [sp, #16]
        \\ldp x19, x20, [sp, #0]
        \\add sp, sp, #176
        : [func] "+{x8}" (f_reg),
          [args] "+{x9}" (a_reg),
          [sent] "+{x10}" (s_reg),
          [out] "+{x11}" (o_reg),
        :
        : .{
          .x0 = true,
          .x1 = true,
          .x2 = true,
          .x3 = true,
          .x4 = true,
          .x5 = true,
          .x6 = true,
          .x7 = true,
          .x12 = true,
          .x13 = true,
          .x14 = true,
          .x15 = true,
          .x16 = true,
          .x17 = true,
          .x30 = true,
          .v0 = true,
          .v1 = true,
          .v2 = true,
          .v3 = true,
          .v4 = true,
          .v5 = true,
          .v6 = true,
          .v7 = true,
          .v8 = true,
          .v9 = true,
          .v10 = true,
          .v11 = true,
          .v12 = true,
          .v13 = true,
          .v14 = true,
          .v15 = true,
          .v16 = true,
          .v17 = true,
          .v18 = true,
          .v19 = true,
          .v20 = true,
          .v21 = true,
          .v22 = true,
          .v23 = true,
          .v24 = true,
          .v25 = true,
          .v26 = true,
          .v27 = true,
          .v28 = true,
          .v29 = true,
          .v30 = true,
          .v31 = true,
          .nzcv = true,
          .memory = true,
        });

    for (CANARY_SENTINELS, after) |exp, got| {
        if (exp != got) return error.RegisterClobbered;
    }
}

// ============================================================================
// Standalone Spike S7 Microkernel vs Apple Accelerate Benchmark (ADR-0018)
// ============================================================================

pub const SpikeS7Row = struct {
    scenario: []const u8,
    dtype: PanelDType,
    cache_hint: CacheHint,
    m: u32,
    n: u32,
    k: u32,
    threads: u32,
    sme2_us: f64,
    sme2_gbps: f64,
    sme2_gflops: f64,
    accel_sgemm_us: f64,
    accel_sgemm_gflops: f64,
    accel_dequant_plus_sgemm_us: f64,
    speedup_vs_sgemm: f64,
    speedup_vs_dequant_sgemm: f64,
};

pub const SpikeS7ThreadRow = struct {
    scenario: []const u8,
    dtype: PanelDType,
    m: u32,
    n: u32,
    k: u32,
    threads: u32,
    sme2_us: f64,
    sme2_gbps: f64,
    sme2_gflops: f64,
    scaling_vs_1t: f64,
};

pub const MAX_S7_ROWS: usize = 64;
pub const MAX_S7_THREAD_ROWS: usize = 24;

pub const SpikeS7Report = struct {
    caps: SmeCaps,
    swapouts_delta: u64,
    rows: [MAX_S7_ROWS]SpikeS7Row,
    row_count: usize,
    thread_rows: [MAX_S7_THREAD_ROWS]SpikeS7ThreadRow,
    thread_row_count: usize,

    pub fn write(self: *const SpikeS7Report, out: *std.Io.Writer) std.Io.Writer.Error!void {
        try out.print(
            "=== Spike S7 Standalone: ARMv9 SME2 vs Apple Accelerate (cblas_sgemm) ===\n",
            .{},
        );
        try out.print(
            "host.sme2={} sme_b16f32={} sme_f16f32={} svl_bytes={} pcore_clusters={} cpus_per_l2={} pcore_l2_kib={} swapouts_delta={}\n\n",
            .{
                self.caps.has_sme2,
                self.caps.has_sme_b16f32,
                self.caps.has_sme_f16f32,
                self.caps.svl_bytes,
                self.caps.pcore_clusters,
                self.caps.pcore_cpus_per_l2,
                self.caps.pcore_l2_bytes / 1024,
                self.swapouts_delta,
            },
        );
        try out.print(
            "{s:<16} {s:<5} {s:<4} {s:>4} {s:>5} {s:>5} {s:>2} | {s:>9} {s:>8} {s:>9} | {s:>9} {s:>9} | {s:>7} {s:>8}\n",
            .{
                "scenario", "dtype", "hint",    "M",      "N",      "K",      "T",
                "sme2_us",  "GB/s",  "GFLOP/s", "acc_us", "acc_GF", "vs_acc", "vs_dq+ac",
            },
        );
        for (self.rows[0..self.row_count]) |r| {
            const hint_str: []const u8 = if (r.cache_hint == .temporal) "ld1" else "ldnt";
            try out.print(
                "{s:<16} {s:<5} {s:<4} {d:>4} {d:>5} {d:>5} {d:>2} | {d:>9.2} {d:>8.2} {d:>9.2} | {d:>9.2} {d:>9.2} | {d:>6.2}x {d:>7.2}x\n",
                .{
                    r.scenario,         r.dtype.label(),            hint_str,
                    r.m,                r.n,                        r.k,
                    r.threads,          r.sme2_us,                  r.sme2_gbps,
                    r.sme2_gflops,      r.accel_sgemm_us,           r.accel_sgemm_gflops,
                    r.speedup_vs_sgemm, r.speedup_vs_dequant_sgemm,
                },
            );
        }
        try out.print("\n--- P-Cluster Thread Scaling Sweep (1 / 2 / 4 threads) ---\n", .{});
        try out.print(
            "{s:<16} {s:<5} {s:>4} {s:>5} {s:>5} {s:>2} | {s:>9} {s:>8} {s:>9} | {s:>7}\n",
            .{ "scenario", "dtype", "M", "N", "K", "T", "sme2_us", "GB/s", "GFLOP/s", "scale" },
        );
        for (self.thread_rows[0..self.thread_row_count]) |r| {
            try out.print(
                "{s:<16} {s:<5} {d:>4} {d:>5} {d:>5} {d:>2} | {d:>9.2} {d:>8.2} {d:>9.2} | {d:>6.2}x\n",
                .{ r.scenario, r.dtype.label(), r.m, r.n, r.k, r.threads, r.sme2_us, r.sme2_gbps, r.sme2_gflops, r.scaling_vs_1t },
            );
        }
    }
};

fn medianF64(samples: []f64) f64 {
    std.sort.pdq(f64, samples, {}, std.sort.asc(f64));
    return samples[samples.len / 2];
}

fn fillBenchWeightData(
    allocator: std.mem.Allocator,
    dtype: PanelDType,
    n: usize,
    k: usize,
    w_packed: []u8,
    w_f32_out: ?[]f32,
    q_codes_out: ?[]u8,
    sb_out: ?[]quant.ScaleBias,
) !void {
    const groups = k / GROUP_SIZE;
    if (!dtype.isQuantized()) {
        const tmp_u16 = try allocator.alloc(u16, n * k);
        defer allocator.free(tmp_u16);
        const fdt: quant.FloatDType = if (dtype == .bf16) .bf16 else .f16;
        for (0..n * k) |i| {
            const v: f32 = @as(f32, @floatFromInt(@as(i32, @intCast((i * 17 + 3) % 15)) - 7)) * 0.0625;
            tmp_u16[i] = quant.f32ToFloatBits(fdt, v);
            if (w_f32_out) |wf| wf[i] = quant.floatBitsToF32(fdt, tmp_u16[i]);
        }
        try packDensePanel64(dtype, tmp_u16, n, k, w_packed);
    } else {
        const max_q: u8 = switch (dtype) {
            .q4_affine => 15,
            .q3_affine => 7,
            .q2_affine => 3,
            else => unreachable,
        };
        const codes = q_codes_out orelse try allocator.alloc(u8, n * k);
        defer if (q_codes_out == null) allocator.free(codes);
        const sbs = sb_out orelse try allocator.alloc(quant.ScaleBias, n * groups);
        defer if (sb_out == null) allocator.free(sbs);

        for (0..n * groups) |gi| {
            sbs[gi] = .{
                .scale = @floatCast(0.0625 + @as(f32, @floatFromInt(gi % 5)) * 0.015625),
                .bias = @floatCast(-0.25 + @as(f32, @floatFromInt(gi % 3)) * 0.125),
            };
        }
        for (0..n * k) |i| {
            codes[i] = @intCast((i * 7 + 1) % (@as(usize, max_q) + 1));
            if (w_f32_out) |wf| {
                const sb = sbs[i / GROUP_SIZE];
                wf[i] = @as(f32, @floatFromInt(codes[i])) * @as(f32, @floatCast(sb.scale)) + @as(f32, @floatCast(sb.bias));
            }
        }
        try packQuantAffinePanel64(dtype, codes, sbs, n, k, w_packed);
    }
}

/// Pure-CPU scalar dequantization of `[n, k]` quantized codes into row-major `f32`,
/// modeling the pre-dequantization overhead required before calling `Accelerate` `cblas_sgemm`.
fn dequantizeToF32ForAccelerate(
    codes: []const u8,
    sbs: []const quant.ScaleBias,
    dst_f32: []f32,
    n: usize,
    k: usize,
) void {
    const groups = k / GROUP_SIZE;
    for (0..n) |ni| {
        const row_off = ni * k;
        for (0..groups) |g| {
            const sb = sbs[ni * groups + g];
            const s: f32 = @floatCast(sb.scale);
            const b: f32 = @floatCast(sb.bias);
            const base = row_off + g * GROUP_SIZE;
            for (0..GROUP_SIZE) |j| {
                dst_f32[base + j] = @as(f32, @floatFromInt(codes[base + j])) * s + b;
            }
        }
    }
}

pub const SpikeS7Error = Error || sysinfo.Error || vmstat.Error || vmstat.SwapError || metal.Error || std.Thread.SpawnError;

/// Runs the standalone Spike S7 benchmark on Apple Silicon:
/// - L2-hot (`1024 x 2048` and `128 x 2048` router) across `M = 1, 4, 8, 16, 32, 128, 512`
///   and all 5 dtypes (`bf16`, `f16`, `q4`, `q3`, `q2`) vs `Accelerate` `cblas_sgemm`
///   (and `dequant + cblas_sgemm` for quantized weights);
/// - SLC-busted DRAM (`4096 x 4096`, rotating slabs > 32 MiB) comparing `ld1` vs `ldnt1`
///   across `M = 1, 16, 128` for `bf16`, `q4`, `q3`, `q2`;
/// - P-cluster `1 / 2 / 4` thread scaling sweep (`swapouts_delta == 0`).
pub fn runSpikeS7Standalone(allocator: std.mem.Allocator) SpikeS7Error!SpikeS7Report {
    const caps = try SmeCaps.probe();
    const vm_before = try vmstat.Snapshot.capture();

    var disp = try SmeDispatcher.init(allocator, .{});
    defer disp.deinit();

    var report: SpikeS7Report = .{
        .caps = caps,
        .swapouts_delta = 0,
        .rows = undefined,
        .row_count = 0,
        .thread_rows = undefined,
        .thread_row_count = 0,
    };

    const max_t: usize = disp.threadCount();

    // Brief steady-state P-core DVFS warmup (~15 ms) so the first benchmark row
    // (`l2_router`) is not measured while P-cores are ramping from idle clock.
    {
        const wm_n: usize = 512;
        const wm_k: usize = 1024;
        const wm_w = try allocator.alignedAlloc(u8, .fromByteUnits(SVL_BYTES), try packedBlobBytes(.bf16, wm_n, wm_k));
        defer allocator.free(wm_w);
        @memset(wm_w, 0);
        const wm_a = try allocator.alignedAlloc(u16, .fromByteUnits(SVL_BYTES), 32 * wm_k);
        defer allocator.free(wm_a);
        @memset(wm_a, 0);
        const wm_y = try allocator.alignedAlloc(f32, .fromByteUnits(SVL_BYTES), 32 * wm_n);
        defer allocator.free(wm_y);
        for (0..64) |_| {
            try disp.gemmF32(.{ .dtype = .bf16, .cache_hint = .temporal }, wm_a, wm_w, null, wm_y, 32, wm_n, wm_k);
        }
    }

    // --- Part 1: L2-hot benchmark across M = 1, 4, 8, 16, 32, 128, 512 and all 5 dtypes ---
    const l2_shapes = [_]struct { label: []const u8, dtype: PanelDType, m: u32, n: u32, k: u32 }{
        // MoE router L2-hot (M=1 decode & M=16 spec tree)
        .{ .label = "l2_router", .dtype = .bf16, .m = 1, .n = 128, .k = 2048 },
        .{ .label = "l2_router", .dtype = .bf16, .m = 16, .n = 128, .k = 2048 },
        .{ .label = "l2_router", .dtype = .q4_affine, .m = 1, .n = 128, .k = 2048 },
        .{ .label = "l2_router", .dtype = .q4_affine, .m = 16, .n = 128, .k = 2048 },
        // L2-hot projection across full M sweep (1, 4, 8, 16, 32, 128, 512)
        .{ .label = "l2_proj", .dtype = .bf16, .m = 1, .n = 1024, .k = 2048 },
        .{ .label = "l2_proj", .dtype = .bf16, .m = 4, .n = 1024, .k = 2048 },
        .{ .label = "l2_proj", .dtype = .bf16, .m = 8, .n = 1024, .k = 2048 },
        .{ .label = "l2_proj", .dtype = .bf16, .m = 16, .n = 1024, .k = 2048 },
        .{ .label = "l2_proj", .dtype = .bf16, .m = 32, .n = 1024, .k = 2048 },
        .{ .label = "l2_proj", .dtype = .bf16, .m = 128, .n = 1024, .k = 2048 },
        .{ .label = "l2_proj", .dtype = .bf16, .m = 512, .n = 1024, .k = 2048 },
        .{ .label = "l2_proj", .dtype = .f16, .m = 1, .n = 1024, .k = 2048 },
        .{ .label = "l2_proj", .dtype = .f16, .m = 16, .n = 1024, .k = 2048 },
        .{ .label = "l2_proj", .dtype = .f16, .m = 128, .n = 1024, .k = 2048 },
        .{ .label = "l2_proj", .dtype = .f16, .m = 512, .n = 1024, .k = 2048 },
        .{ .label = "l2_proj", .dtype = .q4_affine, .m = 1, .n = 1024, .k = 2048 },
        .{ .label = "l2_proj", .dtype = .q4_affine, .m = 4, .n = 1024, .k = 2048 },
        .{ .label = "l2_proj", .dtype = .q4_affine, .m = 16, .n = 1024, .k = 2048 },
        .{ .label = "l2_proj", .dtype = .q4_affine, .m = 32, .n = 1024, .k = 2048 },
        .{ .label = "l2_proj", .dtype = .q4_affine, .m = 128, .n = 1024, .k = 2048 },
        .{ .label = "l2_proj", .dtype = .q3_affine, .m = 1, .n = 1024, .k = 2048 },
        .{ .label = "l2_proj", .dtype = .q3_affine, .m = 16, .n = 1024, .k = 2048 },
        .{ .label = "l2_proj", .dtype = .q3_affine, .m = 128, .n = 1024, .k = 2048 },
        .{ .label = "l2_proj", .dtype = .q2_affine, .m = 1, .n = 1024, .k = 2048 },
        .{ .label = "l2_proj", .dtype = .q2_affine, .m = 16, .n = 1024, .k = 2048 },
        .{ .label = "l2_proj", .dtype = .q2_affine, .m = 128, .n = 1024, .k = 2048 },
    };

    for (l2_shapes) |case| {
        const m: usize = case.m;
        const n: usize = case.n;
        const k: usize = case.k;
        const w_bytes = try packedBlobBytes(case.dtype, n, k);
        const w_packed = try allocator.alignedAlloc(u8, .fromByteUnits(SVL_BYTES), w_bytes);
        defer allocator.free(w_packed);
        const w_f32 = try allocator.alloc(f32, n * k);
        defer allocator.free(w_f32);
        const q_codes: ?[]u8 = if (case.dtype.isQuantized()) try allocator.alloc(u8, n * k) else null;
        defer if (q_codes) |qc| allocator.free(qc);
        const sbs: ?[]quant.ScaleBias = if (case.dtype.isQuantized()) try allocator.alloc(quant.ScaleBias, n * (k / GROUP_SIZE)) else null;
        defer if (sbs) |sb| allocator.free(sb);

        try fillBenchWeightData(allocator, case.dtype, n, k, w_packed, w_f32, q_codes, sbs);

        const a_u16 = try allocator.alignedAlloc(u16, .fromByteUnits(SVL_BYTES), m * k);
        defer allocator.free(a_u16);
        const a_f32 = try allocator.alloc(f32, m * k);
        defer allocator.free(a_f32);
        const act_dt: quant.FloatDType = if (case.dtype.activationIsBf16()) .bf16 else .f16;
        for (0..m * k) |i| {
            const v: f32 = @as(f32, @floatFromInt(@as(i32, @intCast((i * 11 + 5) % 13)) - 6)) * 0.0625;
            a_u16[i] = quant.f32ToFloatBits(act_dt, v);
            a_f32[i] = quant.floatBitsToF32(act_dt, a_u16[i]);
        }

        const out_sme = try allocator.alignedAlloc(f32, .fromByteUnits(SVL_BYTES), m * n);
        defer allocator.free(out_sme);
        const out_acc = try allocator.alloc(f32, m * n);
        defer allocator.free(out_acc);

        const cfg: GemmConfig = .{ .dtype = case.dtype, .cache_hint = .temporal };
        // Warmup
        for (0..3) |_| {
            try disp.gemmF32(cfg, a_u16, w_packed, null, out_sme, m, n, k);
            try metal.accelerateSgemm(case.m, case.n, case.k, 1.0, a_f32, w_f32, 0.0, out_acc, true);
        }

        const reps: usize = if (m <= 16 and n <= 1024) 15 else 7;
        var sme_samples: [15]f64 = undefined;
        var acc_samples: [15]f64 = undefined;
        var dq_samples: [15]f64 = undefined;

        for (0..reps) |r| {
            const t0 = try vmstat.nowNs();
            try disp.gemmF32(cfg, a_u16, w_packed, null, out_sme, m, n, k);
            const t1 = try vmstat.nowNs();
            sme_samples[r] = @as(f64, @floatFromInt(t1 - t0)) / 1000.0;

            const t2 = try vmstat.nowNs();
            try metal.accelerateSgemm(case.m, case.n, case.k, 1.0, a_f32, w_f32, 0.0, out_acc, true);
            const t3 = try vmstat.nowNs();
            acc_samples[r] = @as(f64, @floatFromInt(t3 - t2)) / 1000.0;

            if (case.dtype.isQuantized()) {
                const t4 = try vmstat.nowNs();
                dequantizeToF32ForAccelerate(q_codes.?, sbs.?, w_f32, n, k);
                try metal.accelerateSgemm(case.m, case.n, case.k, 1.0, a_f32, w_f32, 0.0, out_acc, true);
                const t5 = try vmstat.nowNs();
                dq_samples[r] = @as(f64, @floatFromInt(t5 - t4)) / 1000.0;
            } else {
                dq_samples[r] = acc_samples[r];
            }
        }

        const sme_us = medianF64(sme_samples[0..reps]);
        const acc_us = medianF64(acc_samples[0..reps]);
        const dq_us = medianF64(dq_samples[0..reps]);
        const flops = 2.0 * @as(f64, @floatFromInt(m)) * @as(f64, @floatFromInt(n)) * @as(f64, @floatFromInt(k));
        const bytes_moved = @as(f64, @floatFromInt(w_bytes + m * k * 2 + m * n * 4));

        report.rows[report.row_count] = .{
            .scenario = case.label,
            .dtype = case.dtype,
            .cache_hint = .temporal,
            .m = case.m,
            .n = case.n,
            .k = case.k,
            .threads = @intCast(disp.autoThreadCount(case.dtype, m, n, k, caps.default_backend)),
            .sme2_us = sme_us,
            .sme2_gbps = bytes_moved / (sme_us * 1000.0),
            .sme2_gflops = flops / (sme_us * 1000.0),
            .accel_sgemm_us = acc_us,
            .accel_sgemm_gflops = flops / (acc_us * 1000.0),
            .accel_dequant_plus_sgemm_us = dq_us,
            .speedup_vs_sgemm = acc_us / sme_us,
            .speedup_vs_dequant_sgemm = dq_us / sme_us,
        };
        report.row_count += 1;
    }

    // --- Part 2: SLC-busted DRAM streaming (`4096 x 4096`, rotating slabs > 32 MiB) ---
    // Compares temporal `ld1` vs non-temporal `ldnt1` across `bf16`, `q4`, `q3`, `q2`.
    const dram_cases = [_]struct { dtype: PanelDType, hint: CacheHint, m: u32 }{
        .{ .dtype = .bf16, .hint = .temporal, .m = 1 },
        .{ .dtype = .bf16, .hint = .non_temporal, .m = 1 },
        .{ .dtype = .bf16, .hint = .temporal, .m = 16 },
        .{ .dtype = .bf16, .hint = .temporal, .m = 128 },
        .{ .dtype = .q4_affine, .hint = .temporal, .m = 1 },
        .{ .dtype = .q4_affine, .hint = .non_temporal, .m = 1 },
        .{ .dtype = .q4_affine, .hint = .temporal, .m = 16 },
        .{ .dtype = .q4_affine, .hint = .non_temporal, .m = 16 },
        .{ .dtype = .q3_affine, .hint = .non_temporal, .m = 1 },
        .{ .dtype = .q2_affine, .hint = .non_temporal, .m = 1 },
    };

    const dram_n: usize = 4096;
    const dram_k: usize = 4096;
    for (dram_cases) |dc| {
        const m: usize = dc.m;
        const w_bytes = try packedBlobBytes(dc.dtype, dram_n, dram_k);
        // Rotate across enough independent slabs to exceed M4's 16 MiB L2 + 8 MiB SLC (> 40 MiB).
        const min_working_bytes: usize = 48 * 1024 * 1024;
        const slabs: usize = @max(@as(usize, 2), (min_working_bytes + w_bytes - 1) / w_bytes);
        const total_slab_bytes = slabs * w_bytes;
        const w_ring = try allocator.alignedAlloc(u8, .fromByteUnits(SVL_BYTES), total_slab_bytes);
        defer allocator.free(w_ring);

        // Fill slab 0 accurately and replicate into slabs 1..slabs-1 to keep setup fast.
        const w_f32 = try allocator.alloc(f32, dram_n * dram_k);
        defer allocator.free(w_f32);
        const q_codes: ?[]u8 = if (dc.dtype.isQuantized()) try allocator.alloc(u8, dram_n * dram_k) else null;
        defer if (q_codes) |qc| allocator.free(qc);
        const sbs: ?[]quant.ScaleBias = if (dc.dtype.isQuantized()) try allocator.alloc(quant.ScaleBias, dram_n * (dram_k / GROUP_SIZE)) else null;
        defer if (sbs) |sb| allocator.free(sb);

        try fillBenchWeightData(allocator, dc.dtype, dram_n, dram_k, w_ring[0..w_bytes], w_f32, q_codes, sbs);
        for (1..slabs) |s| {
            @memcpy(w_ring[s * w_bytes ..][0..w_bytes], w_ring[0..w_bytes]);
        }

        const a_u16 = try allocator.alignedAlloc(u16, .fromByteUnits(SVL_BYTES), m * dram_k);
        defer allocator.free(a_u16);
        const a_f32 = try allocator.alloc(f32, m * dram_k);
        defer allocator.free(a_f32);
        const act_dt: quant.FloatDType = if (dc.dtype.activationIsBf16()) .bf16 else .f16;
        for (0..m * dram_k) |i| {
            const v: f32 = @as(f32, @floatFromInt(@as(i32, @intCast((i * 13 + 7) % 15)) - 7)) * 0.0625;
            a_u16[i] = quant.f32ToFloatBits(act_dt, v);
            a_f32[i] = quant.floatBitsToF32(act_dt, a_u16[i]);
        }

        const out_sme = try allocator.alignedAlloc(f32, .fromByteUnits(SVL_BYTES), m * dram_n);
        defer allocator.free(out_sme);
        const out_acc = try allocator.alloc(f32, m * dram_n);
        defer allocator.free(out_acc);

        const cfg: GemmConfig = .{ .dtype = dc.dtype, .cache_hint = dc.hint };
        // Warmup 1 slab
        try disp.gemmF32(cfg, a_u16, w_ring[0..w_bytes], null, out_sme, m, dram_n, dram_k);
        try metal.accelerateSgemm(dc.m, dram_n, dram_k, 1.0, a_f32, w_f32, 0.0, out_acc, true);

        const reps: usize = 5;
        var sme_samples: [5]f64 = undefined;
        var acc_samples: [5]f64 = undefined;
        var dq_samples: [5]f64 = undefined;

        for (0..reps) |r| {
            const slab = (r + 1) % slabs;
            const t0 = try vmstat.nowNs();
            try disp.gemmF32(cfg, a_u16, w_ring[slab * w_bytes ..][0..w_bytes], null, out_sme, m, dram_n, dram_k);
            const t1 = try vmstat.nowNs();
            sme_samples[r] = @as(f64, @floatFromInt(t1 - t0)) / 1000.0;

            const t2 = try vmstat.nowNs();
            try metal.accelerateSgemm(dc.m, dram_n, dram_k, 1.0, a_f32, w_f32, 0.0, out_acc, true);
            const t3 = try vmstat.nowNs();
            acc_samples[r] = @as(f64, @floatFromInt(t3 - t2)) / 1000.0;

            if (dc.dtype.isQuantized()) {
                const t4 = try vmstat.nowNs();
                dequantizeToF32ForAccelerate(q_codes.?, sbs.?, w_f32, dram_n, dram_k);
                try metal.accelerateSgemm(dc.m, dram_n, dram_k, 1.0, a_f32, w_f32, 0.0, out_acc, true);
                const t5 = try vmstat.nowNs();
                dq_samples[r] = @as(f64, @floatFromInt(t5 - t4)) / 1000.0;
            } else {
                dq_samples[r] = acc_samples[r];
            }
        }

        const sme_us = medianF64(&sme_samples);
        const acc_us = medianF64(&acc_samples);
        const dq_us = medianF64(&dq_samples);
        const flops = 2.0 * @as(f64, @floatFromInt(m)) * @as(f64, @floatFromInt(dram_n)) * @as(f64, @floatFromInt(dram_k));
        const bytes_moved = @as(f64, @floatFromInt(w_bytes + m * dram_k * 2 + m * dram_n * 4));

        report.rows[report.row_count] = .{
            .scenario = "dram_slc_bust",
            .dtype = dc.dtype,
            .cache_hint = dc.hint,
            .m = dc.m,
            .n = dram_n,
            .k = dram_k,
            .threads = @intCast(disp.autoThreadCount(dc.dtype, m, dram_n, dram_k, caps.default_backend)),
            .sme2_us = sme_us,
            .sme2_gbps = bytes_moved / (sme_us * 1000.0),
            .sme2_gflops = flops / (sme_us * 1000.0),
            .accel_sgemm_us = acc_us,
            .accel_sgemm_gflops = flops / (acc_us * 1000.0),
            .accel_dequant_plus_sgemm_us = dq_us,
            .speedup_vs_sgemm = acc_us / sme_us,
            .speedup_vs_dequant_sgemm = dq_us / sme_us,
        };
        report.row_count += 1;
    }

    // --- Part 3: P-Cluster 1 / 2 / 4 Thread Sweep ---
    const sweep_cases = [_]struct { label: []const u8, dtype: PanelDType, m: u32, n: u32, k: u32 }{
        .{ .label = "sweep_l2_q4", .dtype = .q4_affine, .m = 1, .n = 2048, .k = 2048 },
        .{ .label = "sweep_l2_bf16", .dtype = .bf16, .m = 16, .n = 2048, .k = 2048 },
        .{ .label = "sweep_l2_bf16", .dtype = .bf16, .m = 128, .n = 2048, .k = 2048 },
        .{ .label = "sweep_dram_q4", .dtype = .q4_affine, .m = 1, .n = 4096, .k = 4096 },
    };
    const thread_counts = [_]usize{ 1, 2, 4 };

    for (sweep_cases) |sc| {
        const m: usize = sc.m;
        const n: usize = sc.n;
        const k: usize = sc.k;
        const w_bytes = try packedBlobBytes(sc.dtype, n, k);
        const w_packed = try allocator.alignedAlloc(u8, .fromByteUnits(SVL_BYTES), w_bytes);
        defer allocator.free(w_packed);
        try fillBenchWeightData(allocator, sc.dtype, n, k, w_packed, null, null, null);

        const a_u16 = try allocator.alignedAlloc(u16, .fromByteUnits(SVL_BYTES), m * k);
        defer allocator.free(a_u16);
        const act_dt: quant.FloatDType = if (sc.dtype.activationIsBf16()) .bf16 else .f16;
        for (0..m * k) |i| {
            a_u16[i] = quant.f32ToFloatBits(act_dt, @as(f32, @floatFromInt(@as(i32, @intCast(i % 9)) - 4)) * 0.125);
        }
        const out_sme = try allocator.alignedAlloc(f32, .fromByteUnits(SVL_BYTES), m * n);
        defer allocator.free(out_sme);

        var t1_us: f64 = 0.0;
        for (thread_counts) |tc| {
            if (tc > max_t) continue;
            const cfg: GemmConfig = .{ .dtype = sc.dtype, .cache_hint = .temporal, .max_threads = tc };
            try disp.gemmF32(cfg, a_u16, w_packed, null, out_sme, m, n, k);
            var samples: [7]f64 = undefined;
            for (&samples) |*s| {
                const t0 = try vmstat.nowNs();
                try disp.gemmF32(cfg, a_u16, w_packed, null, out_sme, m, n, k);
                const t1 = try vmstat.nowNs();
                s.* = @as(f64, @floatFromInt(t1 - t0)) / 1000.0;
            }
            const us = medianF64(&samples);
            if (tc == 1) t1_us = us;
            const flops = 2.0 * @as(f64, @floatFromInt(m)) * @as(f64, @floatFromInt(n)) * @as(f64, @floatFromInt(k));
            const bytes_moved = @as(f64, @floatFromInt(w_bytes + m * k * 2 + m * n * 4));
            report.thread_rows[report.thread_row_count] = .{
                .scenario = sc.label,
                .dtype = sc.dtype,
                .m = sc.m,
                .n = sc.n,
                .k = sc.k,
                .threads = @intCast(tc),
                .sme2_us = us,
                .sme2_gbps = bytes_moved / (us * 1000.0),
                .sme2_gflops = flops / (us * 1000.0),
                .scaling_vs_1t = if (t1_us > 0.0) t1_us / us else 1.0,
            };
            report.thread_row_count += 1;
        }
    }

    const vm_after = try vmstat.Snapshot.capture();
    report.swapouts_delta = vmstat.Snapshot.swapoutDelta(vm_before, vm_after);
    try vmstat.checkNoSwap(vm_before, vm_after);
    return report;
}

// ============================================================================
// Tests: C11 Probe, M1–M3 Fallback, Register-Clobber Canary, C12 Full Parity,
//        Multi-Thread Zero-Alloc Dispatcher, and MLX Repack Parity
// ============================================================================

fn readOutputAsF32(out_dtype: OutputDType, out_bytes: []const u8, idx: usize) f32 {
    return switch (out_dtype) {
        .f32 => @bitCast(std.mem.readInt(u32, out_bytes[idx * 4 ..][0..4], .little)),
        .f16 => @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, out_bytes[idx * 2 ..][0..2], .little)))),
        .bf16 => quant.bf16ToF32(std.mem.readInt(u16, out_bytes[idx * 2 ..][0..2], .little)),
    };
}

test "SmeCaps probes host SME2 features and supports simulated M1-M3 fallback cleanly" {
    const caps = try SmeCaps.probe();
    try std.testing.expect(caps.pcore_physical_cpu > 0);
    try std.testing.expect(caps.pcore_cpus_per_l2 > 0);
    try std.testing.expect(caps.pcore_clusters >= 1);
    if (caps.has_sme2) {
        try std.testing.expectEqual(@as(u64, 64), caps.svl_bytes);
        try std.testing.expectEqual(ExecutionBackend.sme2, caps.default_backend);
    }

    // Simulate an M1-M3 host lacking FEAT_SME2: auto resolves to scalar_fallback,
    // explicit `.sme2` is rejected with `error.Sme2Unsupported` (never SIGILL).
    const fake_m1 = SmeCaps.fromFacts(.{
        .page_size = page.APPLE_PAGE_SIZE,
        .memory_bytes = 16 * 1024 * 1024 * 1024,
        .performance_cores = 4,
        .efficiency_cores = 4,
        .user_wire_limit = 8 * 1024 * 1024 * 1024,
        .global_user_wire_limit = 12 * 1024 * 1024 * 1024,
        .gpu_wired_limit_mib = 0,
        .pcore_physical_cpu = 4,
        .pcore_cpus_per_l2 = 4,
        .pcore_l2_cache_bytes = 12 * 1024 * 1024,
        .pcore_clusters = 1,
        .has_sme = false,
        .has_sme2 = false,
        .has_sme_b16f32 = false,
        .has_sme_f16f32 = false,
        .sme_max_svl_bytes = 0,
    });
    try std.testing.expectEqual(ExecutionBackend.scalar_fallback, fake_m1.default_backend);
    try std.testing.expectEqual(ExecutionBackend.scalar_fallback, try fake_m1.resolveBackend(null));
    try std.testing.expectError(error.Sme2Unsupported, fake_m1.resolveBackend(.sme2));
}

test "register-clobber canary verifies 0-bit corruption of x19..x28 and d8..d15 across all 30 SME2 entrypoints" {
    const caps = try SmeCaps.probe();
    if (!caps.supportsSme2()) return;

    const gpa = std.testing.allocator;
    const k: usize = 64;
    const n: usize = 64; // 4 panels
    const m: usize = 32;

    const a_raw = try gpa.alignedAlloc(u16, .fromByteUnits(SVL_BYTES), m * k);
    defer gpa.free(a_raw);
    for (a_raw, 0..) |*v, i| v.* = @bitCast(@as(f16, @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast(i % 7)) - 3)) * 0.25)));

    const a_packed = try gpa.alignedAlloc(u16, .fromByteUnits(SVL_BYTES), 32 * k);
    defer gpa.free(a_packed);
    packActivationStrip(a_raw, 0, 16, k, a_packed[0 .. 16 * k]);
    packActivationStrip(a_raw, 16, 16, k, a_packed[16 * k .. 32 * k]);

    const x_sums = try gpa.alignedAlloc(f32, .fromByteUnits(SVL_BYTES), 4);
    defer gpa.free(x_sums);
    computeGroupActivationSums(a_raw, 0, 4, k, x_sums);

    const out_buf = try gpa.alignedAlloc(f32, .fromByteUnits(SVL_BYTES), m * n);
    defer gpa.free(out_buf);
    const aux_buf = try gpa.alignedAlloc(f32, .fromByteUnits(SVL_BYTES), m * n);
    defer gpa.free(aux_buf);
    @memset(aux_buf, 0.5);

    for (ALL_KERNEL_ENTRIES) |entry| {
        const w_bytes = try packedBlobBytes(entry.dtype, n, k);
        const w_buf = try gpa.alignedAlloc(u8, .fromByteUnits(SVL_BYTES), w_bytes);
        defer gpa.free(w_buf);
        @memset(w_buf, 0x11);

        const m_rows: u32 = switch (entry.regime) {
            .m1_4 => 4,
            .m5_16 => 16,
            .m32 => 32,
        };
        const a_ptr: [*]const u8 = if (entry.regime == .m1_4)
            std.mem.sliceAsBytes(a_raw).ptr
        else
            std.mem.sliceAsBytes(a_packed).ptr;
        const a_stride: usize = if (entry.regime == .m1_4) k * 2 else 16 * k * 2;

        // Exercise both affine (none) and SwiGLU epilogue under canary check!
        for ([_]EpilogueOp{ .none, .swiglu_gate_up }) |epi| {
            const args: KernelArgs = .{
                .a_ptr = a_ptr,
                .w_ptr = w_buf.ptr,
                .out_ptr = std.mem.sliceAsBytes(out_buf).ptr,
                .aux_ptr = aux_buf.ptr,
                .x_group_sums_ptr = x_sums.ptr,
                .zt0_table_ptr = std.mem.sliceAsBytes(&ZT0_AFFINE_TABLE).ptr,
                .a_row_stride_bytes = a_stride,
                .panel_stride_bytes = try panelStrideBytes(entry.dtype, k),
                .out_row_stride_bytes = n * 4,
                .aux_row_stride_elems = n,
                .m_rows = m_rows,
                .n_valid_cols = n,
                .panel_count = @intCast(panelCount(n)),
                .groups = 1,
                .epilogue_op = @backingInt(epi),
                .out_dtype = @backingInt(OutputDType.f32),
                .param_kind = @backingInt(QuantParamKind.affine_scale_bias),
            };
            try invokeWithRegisterCanary(entry.func, &args);
        }
    }

    // Negative canary self-check: prove that an intentional 1-bit `d8` clobber
    // is immediately caught as `error.RegisterClobbered` (not silently ignored).
    const BadClobber = struct {
        fn clobberD8(_: *const KernelArgs) callconv(.c) void {
            asm volatile ("fmov d8, xzr" ::: .{ .memory = true });
        }
    };
    const dummy_args: KernelArgs = .{
        .a_ptr = std.mem.sliceAsBytes(a_raw).ptr,
        .w_ptr = std.mem.sliceAsBytes(&ZT0_AFFINE_TABLE).ptr,
        .out_ptr = std.mem.sliceAsBytes(out_buf).ptr,
        .aux_ptr = null,
        .x_group_sums_ptr = null,
        .zt0_table_ptr = std.mem.sliceAsBytes(&ZT0_AFFINE_TABLE).ptr,
        .a_row_stride_bytes = 0,
        .panel_stride_bytes = 0,
        .out_row_stride_bytes = 0,
        .aux_row_stride_elems = 0,
        .m_rows = 0,
        .n_valid_cols = 0,
        .panel_count = 0,
        .groups = 0,
        .epilogue_op = 0,
        .out_dtype = 0,
        .param_kind = 0,
    };
    try std.testing.expectError(error.RegisterClobbered, invokeWithRegisterCanary(&BadClobber.clobberD8, &dummy_args));
}

fn verifyShapeParity(
    disp: *SmeDispatcher,
    allocator: std.mem.Allocator,
    dtype: PanelDType,
    param_kind: QuantParamKind,
    hint: CacheHint,
    epilogue: EpilogueOp,
    out_dtype: OutputDType,
    backend: ExecutionBackend,
    m: usize,
    n: usize,
    k: usize,
    max_threads: usize,
) !void {
    const groups = k / GROUP_SIZE;
    const act_dt: quant.FloatDType = if (dtype.activationIsBf16()) .bf16 else .f16;

    const a_u16 = try allocator.alignedAlloc(u16, .fromByteUnits(SVL_BYTES), m * k);
    defer allocator.free(a_u16);
    const a_f32 = try allocator.alloc(f32, m * k);
    defer allocator.free(a_f32);
    for (0..m * k) |i| {
        const v: f32 = @as(f32, @floatFromInt(@as(i32, @intCast((i * 19 + 7) % 17)) - 8)) * 0.0625;
        a_u16[i] = quant.f32ToFloatBits(act_dt, v);
        a_f32[i] = quant.floatBitsToF32(act_dt, a_u16[i]);
    }

    const w_bytes = try packedBlobBytes(dtype, n, k);
    const w_packed = try allocator.alignedAlloc(u8, .fromByteUnits(SVL_BYTES), w_bytes);
    defer allocator.free(w_packed);
    const w_f32 = try allocator.alloc(f32, n * k);
    defer allocator.free(w_f32);

    if (!dtype.isQuantized()) {
        const w_u16 = try allocator.alloc(u16, n * k);
        defer allocator.free(w_u16);
        const w_dt: quant.FloatDType = if (dtype == .bf16) .bf16 else .f16;
        for (0..n * k) |i| {
            const v: f32 = @as(f32, @floatFromInt(@as(i32, @intCast((i * 13 + 5) % 15)) - 7)) * 0.0625;
            w_u16[i] = quant.f32ToFloatBits(w_dt, v);
            w_f32[i] = quant.floatBitsToF32(w_dt, w_u16[i]);
        }
        try packDensePanel64(dtype, w_u16, n, k, w_packed);
    } else if (param_kind == .affine_scale_bias) {
        const max_q: usize = switch (dtype) {
            .q4_affine => 15,
            .q3_affine => 7,
            .q2_affine => 3,
            else => unreachable,
        };
        const codes = try allocator.alloc(u8, n * k);
        defer allocator.free(codes);
        const sbs = try allocator.alloc(quant.ScaleBias, n * groups);
        defer allocator.free(sbs);
        for (0..n * groups) |gi| {
            sbs[gi] = .{
                .scale = @floatCast(0.0625 * @as(f32, @floatFromInt((gi % 4) + 1))),
                .bias = @floatCast(-0.25 + 0.125 * @as(f32, @floatFromInt(gi % 5))),
            };
        }
        for (0..n) |ni| {
            for (0..k) |ki| {
                const q: u8 = @intCast((ni * 11 + ki * 7 + 3) % (max_q + 1));
                codes[ni * k + ki] = q;
                const sb = sbs[ni * groups + (ki / GROUP_SIZE)];
                w_f32[ni * k + ki] = @as(f32, @floatFromInt(q)) * @as(f32, @floatCast(sb.scale)) + @as(f32, @floatCast(sb.bias));
            }
        }
        try packQuantAffinePanel64(dtype, codes, sbs, n, k, w_packed);
    } else {
        const max_q: usize = switch (dtype) {
            .q4_affine => 15,
            .q3_affine => 7,
            .q2_affine => 3,
            else => unreachable,
        };
        const panels = panelCount(n);
        const codes = try allocator.alloc(u8, n * k);
        defer allocator.free(codes);
        const cbs = try allocator.alloc([16]f16, panels * groups);
        defer allocator.free(cbs);
        const biases = try allocator.alloc(f16, n * groups);
        defer allocator.free(biases);
        for (0..panels * groups) |pgi| {
            for (0..16) |idx| {
                cbs[pgi][idx] = @floatCast((@as(f32, @floatFromInt(@as(i32, @intCast(idx)) - 7)) * 0.0625) + @as(f32, @floatFromInt(pgi % 3)) * 0.03125);
            }
        }
        for (0..n * groups) |gi| {
            biases[gi] = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast(gi % 5)) - 2)) * 0.0625);
        }
        for (0..n) |ni| {
            const p = ni / PANEL_COLS;
            for (0..k) |ki| {
                const g = ki / GROUP_SIZE;
                const q: u8 = @intCast((ni * 5 + ki * 13 + 1) % (max_q + 1));
                codes[ni * k + ki] = q;
                w_f32[ni * k + ki] = @as(f32, @floatCast(cbs[p * groups + g][q])) + @as(f32, @floatCast(biases[ni * groups + g]));
            }
        }
        try packQuantLut16Panel64(dtype, codes, cbs, biases, n, k, w_packed);
    }

    const aux_f32 = try allocator.alloc(f32, m * n);
    defer allocator.free(aux_f32);
    for (aux_f32, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 7 + 2) % 11)) - 5)) * 0.125;
    }

    const ref_f32 = try allocator.alloc(f32, m * n);
    defer allocator.free(ref_f32);
    matmulRowMajorRefF32(a_f32, w_f32, aux_f32, epilogue, ref_f32, m, n, k);

    // Add canary guard bytes immediately after the output buffer to detect any OOB write on odd N!
    const out_bytes_len = m * n * out_dtype.elemBytes();
    const guard_len: usize = 64;
    const out_alloc = try allocator.alignedAlloc(u8, .fromByteUnits(SVL_BYTES), out_bytes_len + guard_len);
    defer allocator.free(out_alloc);
    @memset(out_alloc, 0xA5);

    try disp.gemm(
        .{
            .dtype = dtype,
            .param_kind = param_kind,
            .cache_hint = hint,
            .epilogue = epilogue,
            .out_dtype = out_dtype,
            .backend = backend,
            .max_threads = max_threads,
        },
        a_u16,
        w_packed,
        if (epilogue != .none) aux_f32 else null,
        out_alloc[0..out_bytes_len],
        m,
        n,
        k,
    );

    // Guard bytes past `m * n` must remain untouched!
    for (out_alloc[out_bytes_len..]) |b| {
        try std.testing.expectEqual(@as(u8, 0xA5), b);
    }

    const rel_tol: f32 = switch (out_dtype) {
        .f32 => if (dtype.isQuantized()) 2.5e-2 else 1.5e-2,
        .f16 => 3.5e-2,
        .bf16 => 5.0e-2,
    };
    const abs_tol: f32 = switch (out_dtype) {
        .f32 => 5.0e-3,
        .f16 => 2.5e-2,
        .bf16 => 4.0e-2,
    };

    for (0..m * n) |idx| {
        const exp = ref_f32[idx];
        const got = readOutputAsF32(out_dtype, out_alloc[0..out_bytes_len], idx);
        try std.testing.expectApproxEqAbs(exp, got, @max(abs_tol, @abs(exp) * rel_tol));
    }
}

test "C12 full numerical parity across odd/boundary M, N, K, all 5 dtypes, temporal/non_temporal, LUT16, epilogues and fallback" {
    const gpa = std.testing.allocator;
    const caps = try SmeCaps.probe();

    var disp = try SmeDispatcher.init(gpa, .{});
    defer disp.deinit();

    const backends: []const ExecutionBackend = if (caps.supportsSme2())
        &.{ .sme2, .scalar_fallback }
    else
        &.{.scalar_fallback};

    // 1. Odd and boundary M sweep across all 5 dtypes (covers M=1..4, 5..16, 17, 32, 64!)
    const m_values = [_]usize{ 1, 2, 3, 4, 5, 7, 13, 16, 17, 32, 64 };
    const all_dtypes = [_]PanelDType{ .bf16, .f16, .q4_affine, .q3_affine, .q2_affine };
    for (backends) |be| {
        for (all_dtypes) |dt| {
            for (m_values) |m| {
                // Use odd N = 33 (3 panels with tail mask!) and K = 128
                try verifyShapeParity(&disp, gpa, dt, .affine_scale_bias, .temporal, .none, .f32, be, m, 33, 128, disp.threadCount());
            }
        }
    }

    // 2. Odd & wide N sweep (`16, 33, 64, 128, 256`) and K sweep (`64, 128, 256`)
    //    covering both `temporal` (`ld1`) and `non_temporal` (`ldnt1`).
    const n_values = [_]usize{ 16, 33, 64, 128, 256 };
    const k_values = [_]usize{ 64, 128, 256 };
    for (backends) |be| {
        for ([_]CacheHint{ .temporal, .non_temporal }) |hint| {
            for (n_values, 0..) |n, ni| {
                const k = k_values[ni % k_values.len];
                const dt = all_dtypes[ni % all_dtypes.len];
                for ([_]usize{ 1, 13, 32 }) |m| {
                    try verifyShapeParity(&disp, gpa, dt, .affine_scale_bias, hint, .none, .f32, be, m, n, k, disp.threadCount());
                }
            }
        }
    }

    // 3. All 3 fused epilogues (`none`, `add_residual`, `swiglu_gate_up`) x all 3 output dtypes (`f32`, `f16`, `bf16`)
    //    across all 3 M regimes (`M = 3`, `M = 11`, `M = 35` [hybrid m32 + m1_4 tail!]).
    for (backends) |be| {
        for ([_]EpilogueOp{ .none, .add_residual, .swiglu_gate_up }) |epi| {
            for ([_]OutputDType{ .f32, .f16, .bf16 }) |odt| {
                for ([_]usize{ 3, 11, 35 }) |m| {
                    try verifyShapeParity(&disp, gpa, .q4_affine, .affine_scale_bias, .non_temporal, epi, odt, be, m, 67, 128, disp.threadCount());
                }
            }
        }
    }

    // 4. Custom ZT0 LUT16 codebook (`QuantParamKind.lut16_codebook`) across q4/q3/q2 and all 3 M regimes!
    for (backends) |be| {
        for ([_]PanelDType{ .q4_affine, .q3_affine, .q2_affine }) |dt| {
            for ([_]usize{ 2, 9, 32 }) |m| {
                try verifyShapeParity(&disp, gpa, dt, .lut16_codebook, .temporal, .add_residual, .f32, be, m, 33, 128, disp.threadCount());
            }
        }
    }
}

test "MLX q4/q3 direct repacking into sme2_panel64 matches QuantOps.matmulRef" {
    const gpa = std.testing.allocator;
    const caps = try SmeCaps.probe();
    var disp = try SmeDispatcher.init(gpa, .{});
    defer disp.deinit();

    const m: usize = 9;
    const n: usize = 37;
    const k: usize = 128;
    const groups = k / GROUP_SIZE;

    const a_f16 = try gpa.alignedAlloc(f16, .fromByteUnits(SVL_BYTES), m * k);
    defer gpa.free(a_f16);
    for (a_f16, 0..) |*v, i| v.* = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast((i * 11 + 3) % 13)) - 6)) * 0.125);

    inline for (.{ quant.QuantBits.q4, quant.QuantBits.q3 }) |bits| {
        const Ops = quant.QuantOps(.{ .bits = bits, .group_size = 64 });
        const dtype: PanelDType = if (bits == .q4) .q4_affine else .q3_affine;
        const max_q: u8 = if (bits == .q4) 15 else 7;

        const mlx_w = try gpa.alloc(u8, try Ops.packedWeightBytes(n, k));
        defer gpa.free(mlx_w);
        const sbs = try gpa.alloc(quant.ScaleBias, n * groups);
        defer gpa.free(sbs);

        const row_bytes = try Ops.rowWeightBytes(k);
        for (0..n) |ni| {
            for (0..groups) |g| {
                sbs[ni * groups + g] = .{
                    .scale = @floatCast(0.0625 * @as(f32, @floatFromInt((ni + g) % 4 + 1))),
                    .bias = @floatCast(-0.1875 + 0.0625 * @as(f32, @floatFromInt((ni * 3 + g) % 5))),
                };
                var codes: [64]u8 = undefined;
                for (&codes, 0..) |*c, j| c.* = @intCast((ni * 7 + g * 5 + j * 3) % (max_q + 1));
                Ops.packGroup(&codes, mlx_w[ni * row_bytes + g * Ops.groupWeightBytes() ..][0..Ops.groupWeightBytes()]);
            }
        }

        const ref_y = try gpa.alloc(f32, m * n);
        defer gpa.free(ref_y);
        try Ops.matmulRef(a_f16, mlx_w, sbs, ref_y, m, n, k);

        const packed_w = try gpa.alignedAlloc(u8, .fromByteUnits(SVL_BYTES), try packedBlobBytes(dtype, n, k));
        defer gpa.free(packed_w);
        try packMlxQuantPanel64(bits, mlx_w, sbs, n, k, packed_w);

        const sme_y = try gpa.alignedAlloc(f32, .fromByteUnits(SVL_BYTES), m * n);
        defer gpa.free(sme_y);
        const a_u16: []const u16 = @ptrCast(a_f16);
        try disp.gemmF32(.{ .dtype = dtype, .backend = caps.default_backend }, a_u16, packed_w, null, sme_y, m, n, k);

        for (0..m * n) |idx| {
            try std.testing.expectApproxEqAbs(ref_y[idx], sme_y[idx], @max(5.0e-3, @abs(ref_y[idx]) * 2.5e-2));
        }
    }
}

test "SmeDispatcher executes multi-threaded with zero hot-path malloc allocations and zero swapouts" {
    const gpa = std.testing.allocator;
    var disp = try SmeDispatcher.init(gpa, .{});
    defer disp.deinit();

    const m: usize = 16;
    const n: usize = 256;
    const k: usize = 256;
    const w_bytes = try packedBlobBytes(.q4_affine, n, k);
    const w_packed = try gpa.alignedAlloc(u8, .fromByteUnits(SVL_BYTES), w_bytes);
    defer gpa.free(w_packed);
    try fillBenchWeightData(gpa, .q4_affine, n, k, w_packed, null, null, null);

    const a_u16 = try gpa.alignedAlloc(u16, .fromByteUnits(SVL_BYTES), m * k);
    defer gpa.free(a_u16);
    for (a_u16, 0..) |*v, i| v.* = @bitCast(@as(f16, @floatCast(@as(f32, @floatFromInt(i % 7)) * 0.125)));

    const out_1t = try gpa.alignedAlloc(f32, .fromByteUnits(SVL_BYTES), m * n);
    defer gpa.free(out_1t);
    const out_mt = try gpa.alignedAlloc(f32, .fromByteUnits(SVL_BYTES), m * n);
    defer gpa.free(out_mt);

    // Warmup both 1-thread and multi-thread paths.
    try disp.gemmF32(.{ .dtype = .q4_affine, .max_threads = 1 }, a_u16, w_packed, null, out_1t, m, n, k);
    try disp.gemmF32(.{ .dtype = .q4_affine, .max_threads = disp.threadCount() }, a_u16, w_packed, null, out_mt, m, n, k);

    const snap_before = try vmstat.Snapshot.capture();
    for (0..20) |_| {
        try disp.gemmF32(.{ .dtype = .q4_affine, .max_threads = disp.threadCount() }, a_u16, w_packed, null, out_mt, m, n, k);
    }
    const snap_after = try vmstat.Snapshot.capture();

    try vmstat.checkNoSwap(snap_before, snap_after);
    const hdelta = vmstat.heapDelta(snap_before, snap_after);
    try std.testing.expectEqual(@as(i64, 0), hdelta.blocks_in_use);
    try std.testing.expectEqual(@as(i64, 0), hdelta.size_in_use);

    // 1-thread and multi-thread outputs must be bit-for-bit identical!
    for (out_1t, out_mt) |v1, vm| {
        try std.testing.expectEqual(v1, vm);
    }
}

fn buildAmfTestImageFromEntries(
    allocator: std.mem.Allocator,
    entries_in: []const amf.TensorEntry,
    blobs_in: []const []const u8,
) ![]u8 {
    std.debug.assert(entries_in.len == blobs_in.len);
    const count = entries_in.len;
    const dir_offset: usize = page.APPLE_PAGE_SIZE;
    const dir_raw = count * @sizeOf(amf.TensorEntry);
    const dir_bytes: usize = try page.alignUp(dir_raw);

    var next_off: usize = dir_offset + dir_bytes;
    const blob_offsets = try allocator.alloc(usize, count);
    defer allocator.free(blob_offsets);
    const blob_sizes = try allocator.alloc(usize, count);
    defer allocator.free(blob_sizes);
    for (blobs_in, 0..) |b, i| {
        const aligned_len = try page.alignUp(@max(@as(usize, 1), b.len));
        blob_offsets[i] = next_off;
        blob_sizes[i] = aligned_len;
        next_off += aligned_len;
    }

    const image = try allocator.alloc(u8, next_off);
    errdefer allocator.free(image);
    @memset(image, 0);

    for (entries_in, 0..) |entry, i| {
        const off = blob_offsets[i];
        const sz = blob_sizes[i];
        @memcpy(image[off..][0..blobs_in[i].len], blobs_in[i]);
        var e = entry;
        e.blob_offset = off;
        e.blob_bytes = sz;
        e.blob_crc32 = std.hash.Crc32.hash(image[off .. off + sz]);
        @memcpy(image[dir_offset + i * @sizeOf(amf.TensorEntry) ..][0..@sizeOf(amf.TensorEntry)], std.mem.asBytes(&e));
    }

    var hdr: amf.Header = .{
        .magic = amf.MAGIC,
        .version = amf.FORMAT_VERSION,
        .flags = 0,
        .page_size = page.APPLE_PAGE_SIZE,
        .checksum_algo = amf.CHECKSUM_CRC32_IEEE,
        .tensor_count = @intCast(count),
        .reserved0 = 0,
        .directory_offset = dir_offset,
        .directory_bytes = dir_bytes,
        .directory_crc32 = std.hash.Crc32.hash(image[dir_offset..][0..dir_raw]),
        .header_crc32 = 0,
        .model_id = @splat(0),
        .reserved1 = 0,
    };
    const model = "sme2-m6-test";
    @memcpy(hdr.model_id[0..model.len], model);
    hdr.header_crc32 = amf.computeHeaderCrc32(&hdr);
    @memcpy(image[0..@sizeOf(amf.Header)], std.mem.asBytes(&hdr));
    return image;
}

fn smeTestBudget(cap_bytes: usize) budget_mod.Budget {
    return budget_mod.Budget.compute(.{
        .memory_bytes = cap_bytes * 8,
        .user_wire_limit = cap_bytes * 2,
        .global_user_wire_limit = cap_bytes * 2,
        .gpu_working_set_bytes = cap_bytes * 2,
        .configured_cap = null,
        .headroom_bytes = page.APPLE_PAGE_SIZE,
    });
}

test "W17: parsed .amf reserved0=0x0101 lut16_codebook auto-dispatches across q4/q3/q2, M=1..64, ld1b & ldnt1b" {
    const gpa = std.testing.allocator;
    var disp = try SmeDispatcher.init(gpa, .{});
    defer disp.deinit();

    const backends: []const ExecutionBackend = if (disp.caps.supportsSme2())
        &.{ .sme2, .scalar_fallback }
    else
        &.{.scalar_fallback};

    const dtypes = [_]struct { amf_dt: amf.DType, panel_dt: PanelDType, bits: u8, max_q: usize }{
        .{ .amf_dt = .q4_affine, .panel_dt = .q4_affine, .bits = 4, .max_q = 15 },
        .{ .amf_dt = .q3_affine, .panel_dt = .q3_affine, .bits = 3, .max_q = 7 },
        .{ .amf_dt = .q2_affine, .panel_dt = .q2_affine, .bits = 2, .max_q = 3 },
    };

    const n: usize = 37;
    const k: usize = 128;
    const panels = panelCount(n);
    const groups = k / GROUP_SIZE;

    for (dtypes) |dt_spec| {
        const codes = try gpa.alloc(u8, n * k);
        defer gpa.free(codes);
        const cbs = try gpa.alloc([16]f16, panels * groups);
        defer gpa.free(cbs);
        const biases = try gpa.alloc(f16, n * groups);
        defer gpa.free(biases);
        const w_f32 = try gpa.alloc(f32, n * k);
        defer gpa.free(w_f32);

        for (0..panels * groups) |pgi| {
            for (0..16) |idx| {
                // Non-uniform codebook centroids (quadratic + sign asymmetry).
                const centered = @as(f32, @floatFromInt(@as(i32, @intCast(idx)) - 7));
                cbs[pgi][idx] = @floatCast(centered * 0.045 + (centered * centered) * 0.004 * @as(f32, @floatFromInt((pgi % 3) + 1)));
            }
        }
        for (0..n * groups) |gi| {
            biases[gi] = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast(gi % 5)) - 2)) * 0.03125);
        }
        for (0..n) |ni| {
            const p = ni / PANEL_COLS;
            for (0..k) |ki| {
                const g = ki / GROUP_SIZE;
                const q: u8 = @intCast((ni * 7 + ki * 11 + dt_spec.bits) % (dt_spec.max_q + 1));
                codes[ni * k + ki] = q;
                w_f32[ni * k + ki] = @as(f32, @floatCast(cbs[p * groups + g][q])) + @as(f32, @floatCast(biases[ni * groups + g]));
            }
        }

        const w_len = try packedBlobBytes(dt_spec.panel_dt, n, k);
        const w_packed = try gpa.alloc(u8, w_len);
        defer gpa.free(w_packed);
        try packQuantLut16Panel64(dt_spec.panel_dt, codes, cbs, biases, n, k, w_packed);

        var entry = amf.TensorEntry.zeroed("layers.0.mlp.shared_expert.gate_proj", .shared_expert, dt_spec.amf_dt);
        entry.quant_bits = dt_spec.bits;
        entry.rank = 2;
        entry.group_size = 64;
        entry.layer = 0;
        entry.shape = .{ @intCast(n), @intCast(k), 0, 0 };
        entry.data_bytes = w_len;
        entry.setLayoutAndParamKind(.sme2_panel64, .lut16_codebook);
        try std.testing.expectEqual(@as(u16, 0x0101), entry.reserved0);

        const entries = [_]amf.TensorEntry{entry};
        const blobs = [_][]const u8{w_packed};
        const img = try buildAmfTestImageFromEntries(gpa, &entries, &blobs);
        defer gpa.free(img);

        var parsed = try amf.Parsed.parseBytes(gpa, img);
        defer parsed.deinit(gpa);
        try parsed.verifyAllBlobsFromBytes(img);

        const cap: usize = @intCast(parsed.totalWiredBytes());
        var mem_arena = try arena_mod.WiredArena.init(cap, smeTestBudget(cap));
        defer mem_arena.deinit();
        const loaded = try parsed.loadWiredFromBytes(gpa, img, &mem_arena);
        defer gpa.free(loaded);

        const found = parsed.find("layers.0.mlp.shared_expert.gate_proj") orelse return error.TestUnexpectedResult;
        const blob = loaded[0] orelse return error.TestUnexpectedResult;
        const bound = try SmeLinearWeight.fromAmfEntry(found, blob, null);
        try std.testing.expectEqual(QuantParamKind.lut16_codebook, bound.param_kind);
        try std.testing.expectEqual(CacheHint.temporal, bound.cache_hint);

        for ([_]usize{ 1, 4, 8, 16, 32, 64 }) |m| {
            const a_u16 = try gpa.alignedAlloc(u16, .fromByteUnits(SVL_BYTES), m * k);
            defer gpa.free(a_u16);
            const a_f32 = try gpa.alloc(f32, m * k);
            defer gpa.free(a_f32);
            for (0..m * k) |i| {
                const v: f32 = @as(f32, @floatFromInt(@as(i32, @intCast((i * 13 + m) % 17)) - 8)) * 0.0625;
                a_u16[i] = quant.f32ToFloatBits(.f16, v);
                a_f32[i] = quant.floatBitsToF32(.f16, a_u16[i]);
            }

            const ref_y = try gpa.alloc(f32, m * n);
            defer gpa.free(ref_y);
            matmulRowMajorRefF32(a_f32, w_f32, null, .none, ref_y, m, n, k);

            const got_y = try gpa.alignedAlloc(f32, .fromByteUnits(SVL_BYTES), m * n);
            defer gpa.free(got_y);

            for (backends) |be| {
                for ([_]CacheHint{ .temporal, .non_temporal }) |hint| {
                    try evalAmfLinear(&disp, found, blob, a_u16, null, std.mem.sliceAsBytes(got_y), m, .{
                        .cache_hint = hint,
                        .epilogue = .none,
                        .out_dtype = .f32,
                        .backend = be,
                    });
                    for (0..m * n) |idx| {
                        const tol: f32 = @max(5.0e-3, @abs(ref_y[idx]) * 2.5e-2);
                        try std.testing.expectApproxEqAbs(ref_y[idx], got_y[idx], tol);
                    }
                }
            }
        }
    }
}

test "W16: SmeStepLoraAdapter enforces bit-exact alpha=0 identity and fp32 parity across rank=16..64 and M=1..64" {
    const gpa = std.testing.allocator;
    var disp = try SmeDispatcher.init(gpa, .{});
    defer disp.deinit();

    const backends: []const ExecutionBackend = if (disp.caps.supportsSme2())
        &.{ .sme2, .scalar_fallback }
    else
        &.{.scalar_fallback};

    const k: usize = 128;
    const n: usize = 96;

    var scratch = try StepLoraScratch.init(64, 64, k, smeTestBudget(256 * 1024));
    defer scratch.deinit();
    try std.testing.expect(scratch.totalWiredBytes() >= page.APPLE_PAGE_SIZE);

    for ([_]usize{ 16, 32, 64 }) |rank| {
        const a_f32 = try gpa.alloc(f32, rank * k);
        defer gpa.free(a_f32);
        const b_f32 = try gpa.alloc(f32, n * rank);
        defer gpa.free(b_f32);
        for (a_f32, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 11 + 3) % 15)) - 7)) * 0.0625;
        for (b_f32, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 17 + 5) % 13)) - 6)) * 0.0625;

        const a_bytes = try packedBlobBytes(.bf16, rank, k);
        const a_packed = try gpa.alloc(u8, a_bytes);
        defer gpa.free(a_packed);
        try packDenseFromF32Panel64(.bf16, a_f32, rank, k, a_packed);

        const b_bytes = try packedBlobBytes(.bf16, n, GROUP_SIZE);
        const b_packed = try gpa.alloc(u8, b_bytes);
        defer gpa.free(b_packed);
        try packStepLoraBDenseFromF32Panel64(.bf16, b_f32, n, rank, b_packed);

        const alphas = [_]f32{ 0.0, 0.375, -0.5 };
        var alpha_bytes: [alphas.len * 4]u8 = undefined;
        for (alphas, 0..) |av, idx| {
            std.mem.writeInt(u32, alpha_bytes[idx * 4 ..][0..4], @bitCast(av), .little);
        }

        var e_a = amf.TensorEntry.zeroed("layers.0.loop.lora_a", .loop_adapter, .bf16);
        e_a.rank = 2;
        e_a.layer = 0;
        e_a.shape = .{ @intCast(rank), @intCast(k), 0, 0 };
        e_a.data_bytes = a_bytes;
        e_a.setLayoutAndParamKind(.sme2_panel64, .affine_scale_bias);

        var e_b = amf.TensorEntry.zeroed("layers.0.loop.lora_b", .loop_adapter, .bf16);
        e_b.rank = 2;
        e_b.layer = 0;
        e_b.shape = .{ @intCast(n), @intCast(GROUP_SIZE), 0, 0 };
        e_b.data_bytes = b_bytes;
        e_b.setLayoutAndParamKind(.sme2_panel64, .affine_scale_bias);

        var e_alpha = amf.TensorEntry.zeroed("layers.0.loop.alpha", .loop_adapter, .f32);
        e_alpha.rank = 1;
        e_alpha.layer = 0;
        e_alpha.shape = .{ alphas.len, 0, 0, 0 };
        e_alpha.data_bytes = alpha_bytes.len;

        const entries = [_]amf.TensorEntry{ e_a, e_b, e_alpha };
        const blobs = [_][]const u8{ a_packed, b_packed, &alpha_bytes };
        const img = try buildAmfTestImageFromEntries(gpa, &entries, &blobs);
        defer gpa.free(img);

        var parsed = try amf.Parsed.parseBytes(gpa, img);
        defer parsed.deinit(gpa);
        try parsed.verifyAllBlobsFromBytes(img);

        const cap: usize = @intCast(parsed.totalWiredBytes());
        var mem_arena = try arena_mod.WiredArena.init(cap, smeTestBudget(cap));
        defer mem_arena.deinit();
        const loaded = try parsed.loadWiredFromBytes(gpa, img, &mem_arena);
        defer gpa.free(loaded);

        const alpha_r0 = try readLoopAlphaFromAmf(&parsed.entries[2], loaded[2].?, 0);
        const alpha_r1 = try readLoopAlphaFromAmf(&parsed.entries[2], loaded[2].?, 1);
        const alpha_r2 = try readLoopAlphaFromAmf(&parsed.entries[2], loaded[2].?, 2);
        try std.testing.expectEqual(@as(f32, 0.0), alpha_r0);
        try std.testing.expectEqual(@as(f32, 0.375), alpha_r1);
        try std.testing.expectEqual(@as(f32, -0.5), alpha_r2);

        const adapter_r0 = try SmeStepLoraAdapter.fromAmfEntries(
            &parsed.entries[0],
            loaded[0].?,
            &parsed.entries[1],
            loaded[1].?,
            alpha_r0,
        );
        try std.testing.expectEqual(a_bytes + b_bytes, adapter_r0.totalWeightBytes());

        for ([_]usize{ 1, 4, 8, 16, 32, 64 }) |m| {
            const x_f32 = try gpa.alloc(f32, m * k);
            defer gpa.free(x_f32);
            const x_u16 = try gpa.alignedAlloc(u16, .fromByteUnits(SVL_BYTES), m * k);
            defer gpa.free(x_u16);
            for (0..m * k) |i| {
                const v: f32 = @as(f32, @floatFromInt(@as(i32, @intCast((i * 19 + rank) % 17)) - 8)) * 0.0625;
                x_u16[i] = quant.f32ToFloatBits(.bf16, v);
                x_f32[i] = quant.floatBitsToF32(.bf16, x_u16[i]);
            }

            const y_base = try gpa.alloc(f32, m * n);
            defer gpa.free(y_base);
            for (y_base, 0..) |*v, i| {
                // Include non-trivial bit patterns so bit-exact r=0 identity is unmistakable.
                v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 23 + 1) % 37)) - 18)) * 0.03125 + 1.0e-4;
            }

            // 1. r = 0 (alpha == 0.0): strictly bit-exact identity!
            const y_r0 = try gpa.alignedAlloc(f32, .fromByteUnits(SVL_BYTES), m * n);
            defer gpa.free(y_r0);
            @memcpy(y_r0, y_base);
            try adapter_r0.evalU16(&disp, &scratch, x_u16, y_r0, m, null, null);
            try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(y_base), std.mem.sliceAsBytes(y_r0));

            // 2. r = 1, 2 (alpha != 0.0): verify fp32 reference parity across all backends.
            const z_ref = try gpa.alloc(f32, m * rank);
            defer gpa.free(z_ref);
            matmulRowMajorRefF32(x_f32, a_f32, null, .none, z_ref, m, rank, k);

            const delta_ref = try gpa.alloc(f32, m * n);
            defer gpa.free(delta_ref);
            matmulRowMajorRefF32(z_ref, b_f32, null, .none, delta_ref, m, n, rank);

            for ([_]f32{ alpha_r1, alpha_r2 }) |alpha_val| {
                const adapter_step = adapter_r0.withAlpha(alpha_val);
                for (backends) |be| {
                    const y_got = try gpa.alignedAlloc(f32, .fromByteUnits(SVL_BYTES), m * n);
                    defer gpa.free(y_got);
                    @memcpy(y_got, y_base);

                    try adapter_step.evalF32(&disp, &scratch, x_f32, y_got, m, be, null);
                    for (0..m * n) |idx| {
                        const expected = y_base[idx] + alpha_val * delta_ref[idx];
                        const tol: f32 = @max(1.5e-2, @abs(expected) * 3.5e-2);
                        try std.testing.expectApproxEqAbs(expected, y_got[idx], tol);
                    }
                }
            }
        }
    }

    // 3. Verify 0 hot-path heap allocations on steady-state Step-LoRA execution.
    const a_hot_bytes = try packedBlobBytes(.bf16, 16, k);
    const b_hot_bytes = try packedBlobBytes(.bf16, n, GROUP_SIZE);
    const a_hot = try gpa.alignedAlloc(u8, .fromByteUnits(SVL_BYTES), a_hot_bytes);
    defer gpa.free(a_hot);
    const b_hot = try gpa.alignedAlloc(u8, .fromByteUnits(SVL_BYTES), b_hot_bytes);
    defer gpa.free(b_hot);
    @memset(a_hot, 0);
    @memset(b_hot, 0);
    const adapter_hot = try SmeStepLoraAdapter.fromWeights(
        .{ .dtype = .bf16, .param_kind = .affine_scale_bias, .cache_hint = .temporal, .n = 16, .k = k, .w_packed = a_hot },
        .{ .dtype = .bf16, .param_kind = .affine_scale_bias, .cache_hint = .temporal, .n = n, .k = GROUP_SIZE, .w_packed = b_hot },
        0.5,
    );
    const x_hot = try gpa.alignedAlloc(u16, .fromByteUnits(SVL_BYTES), 16 * k);
    defer gpa.free(x_hot);
    @memset(x_hot, 0);
    const y_hot = try gpa.alignedAlloc(f32, .fromByteUnits(SVL_BYTES), 16 * n);
    defer gpa.free(y_hot);
    @memset(y_hot, 0);
    try adapter_hot.evalU16(&disp, &scratch, x_hot, y_hot, 16, null, null);
    const snap0 = try vmstat.Snapshot.capture();
    for (0..16) |_| {
        try adapter_hot.evalU16(&disp, &scratch, x_hot, y_hot, 16, null, null);
    }
    const snap1 = try vmstat.Snapshot.capture();
    const hdelta = vmstat.heapDelta(snap0, snap1);
    try std.testing.expectEqual(@as(i64, 0), hdelta.blocks_in_use);
    try std.testing.expectEqual(@as(i64, 0), hdelta.size_in_use);
}

test "D13: SmeDraftProjection evalFusedNormLinear, evalDualNormConcatLinear, evalEagle3FcFusion and calibrateSmeVsGpuUs" {
    const gpa = std.testing.allocator;
    var disp = try SmeDispatcher.init(gpa, .{});
    defer disp.deinit();

    const backends: []const ExecutionBackend = if (disp.caps.supportsSme2())
        &.{ .sme2, .scalar_fallback }
    else
        &.{.scalar_fallback};

    const h: usize = 64;
    const h_tgt: usize = 64;
    const eps: f32 = 1.0e-5;

    // 1. Fused RMSNorm + Linear (bf16 L2-resident projection) + L2 budget guard test.
    {
        const k = h;
        const n = h;
        const w_f32 = try gpa.alloc(f32, n * k);
        defer gpa.free(w_f32);
        for (w_f32, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 7 + 1) % 15)) - 7)) * 0.0625;
        const w_bytes = try packedBlobBytes(.bf16, n, k);
        const w_packed = try gpa.alignedAlloc(u8, .fromByteUnits(SVL_BYTES), w_bytes);
        defer gpa.free(w_packed);
        try packDenseFromF32Panel64(.bf16, w_f32, n, k, w_packed);

        const weight: SmeLinearWeight = .{
            .dtype = .bf16,
            .param_kind = .affine_scale_bias,
            .cache_hint = .temporal,
            .n = n,
            .k = k,
            .w_packed = w_packed,
        };

        // L2 budget guard rejects projections larger than `l2_limit_bytes`.
        try std.testing.expectError(
            error.ExceedsL2Budget,
            SmeDraftProjection.init(weight, .{
                .max_m = 16,
                .l2_limit_bytes = w_bytes - 1,
                .budget_override = smeTestBudget(64 * 1024),
            }),
        );

        var proj = try SmeDraftProjection.init(weight, .{
            .max_m = 64,
            .l2_limit_bytes = MAX_L2_DRAFT_PROJ_BYTES,
            .budget_override = smeTestBudget(64 * 1024),
        });
        defer proj.deinit();
        try std.testing.expect(proj.fitsL2(null));

        var norm_w: [h]f32 = undefined;
        for (&norm_w, 0..) |*v, i| v.* = 0.75 + 0.05 * @as(f32, @floatFromInt(i % 7));

        for ([_]usize{ 1, 8, 16, 64 }) |m| {
            const x_f32 = try gpa.alloc(f32, m * k);
            defer gpa.free(x_f32);
            const x_norm_ref = try gpa.alloc(f32, m * k);
            defer gpa.free(x_norm_ref);
            for (0..m) |r| {
                var ssq: f32 = 0.0;
                for (0..k) |d| {
                    const val: f32 = @as(f32, @floatFromInt(@as(i32, @intCast((r * 19 + d * 5 + 3) % 17)) - 8)) * 0.125;
                    x_f32[r * k + d] = val;
                    ssq += val * val;
                }
                const inv_rms = 1.0 / @sqrt(ssq / @as(f32, @floatFromInt(k)) + eps);
                for (0..k) |d| {
                    x_norm_ref[r * k + d] = (x_f32[r * k + d] * inv_rms) * norm_w[d];
                }
            }
            const ref_y = try gpa.alloc(f32, m * n);
            defer gpa.free(ref_y);
            matmulRowMajorRefF32(x_norm_ref, w_f32, null, .none, ref_y, m, n, k);

            const got_y = try gpa.alignedAlloc(f32, .fromByteUnits(SVL_BYTES), m * n);
            defer gpa.free(got_y);
            for (backends) |be| {
                try proj.evalFusedNormLinear(
                    &disp,
                    .{ .f32 = x_f32 },
                    .{ .f32 = &norm_w },
                    eps,
                    null,
                    std.mem.sliceAsBytes(got_y),
                    m,
                    .{ .backend = be, .out_dtype = .f32 },
                );
                for (0..m * n) |idx| {
                    const tol: f32 = @max(1.5e-2, @abs(ref_y[idx]) * 3.0e-2);
                    try std.testing.expectApproxEqAbs(ref_y[idx], got_y[idx], tol);
                }
            }
        }

        // Benchmark gate helper: calibrateSmeVsGpuUs + calibrateSmeVsGpuFn.
        const out_cal = try gpa.alignedAlloc(f32, .fromByteUnits(SVL_BYTES), 8 * n);
        defer gpa.free(out_cal);
        const fast_gpu = try calibrateSmeVsGpuUs(&disp, &proj, 8, std.mem.sliceAsBytes(out_cal), 0.001, .{
            .warmups = 2,
            .iters = 6,
        });
        try std.testing.expect(fast_gpu.sme_us > 0.0);
        try std.testing.expect(fast_gpu.fits_l2);
        try std.testing.expect(!fast_gpu.prefer_sme2);

        const slow_gpu = try calibrateSmeVsGpuUs(&disp, &proj, 8, std.mem.sliceAsBytes(out_cal), 100_000.0, .{
            .warmups = 2,
            .iters = 6,
        });
        try std.testing.expect(slow_gpu.prefer_sme2);

        const DummyGpu = struct {
            acc: u64 = 0,
            fn step(ctx: *anyopaque) anyerror!void {
                const self: *@This() = @ptrCast(@alignCast(ctx));
                var x: u64 = self.acc;
                for (0..2000) |i| x +%= i *% 0x9e3779b97f4a7c15;
                self.acc = x;
            }
        };
        var dummy_gpu: DummyGpu = .{};
        const fn_cal = try calibrateSmeVsGpuFn(
            &disp,
            &proj,
            8,
            std.mem.sliceAsBytes(out_cal),
            &dummy_gpu,
            DummyGpu.step,
            .{ .warmups = 2, .iters = 6 },
        );
        try std.testing.expect(fn_cal.sme_us > 0.0 and fn_cal.gpu_us >= 0.0);
        try std.testing.expect(dummy_gpu.acc != 0);
    }

    // 2. 4B MTP Dual-Norm `eh_proj: [H, 2*H]` + EAGLE-3 3-tap `fc: [H, 3*H_tgt]` (with LUT16 codebook!).
    {
        const k_fc = 3 * h_tgt;
        const n_fc = h;
        const panels = panelCount(n_fc);
        const groups = k_fc / GROUP_SIZE;

        const codes = try gpa.alloc(u8, n_fc * k_fc);
        defer gpa.free(codes);
        const cbs = try gpa.alloc([16]f16, panels * groups);
        defer gpa.free(cbs);
        const biases = try gpa.alloc(f16, n_fc * groups);
        defer gpa.free(biases);
        const w_f32 = try gpa.alloc(f32, n_fc * k_fc);
        defer gpa.free(w_f32);

        for (0..panels * groups) |pgi| {
            for (0..16) |idx| {
                cbs[pgi][idx] = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast(idx)) - 7)) * 0.05);
            }
        }
        for (0..n_fc * groups) |gi| biases[gi] = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast(gi % 5)) - 2)) * 0.025);
        for (0..n_fc) |ni| {
            const p = ni / PANEL_COLS;
            for (0..k_fc) |ki| {
                const g = ki / GROUP_SIZE;
                const q: u8 = @intCast((ni * 5 + ki * 9 + 1) % 16);
                codes[ni * k_fc + ki] = q;
                w_f32[ni * k_fc + ki] = @as(f32, @floatCast(cbs[p * groups + g][q])) + @as(f32, @floatCast(biases[ni * groups + g]));
            }
        }

        const w_bytes = try packedBlobBytes(.q4_affine, n_fc, k_fc);
        const w_packed = try gpa.alignedAlloc(u8, .fromByteUnits(SVL_BYTES), w_bytes);
        defer gpa.free(w_packed);
        try packQuantLut16Panel64(.q4_affine, codes, cbs, biases, n_fc, k_fc, w_packed);

        var fc_proj = try SmeDraftProjection.init(.{
            .dtype = .q4_affine,
            .param_kind = .lut16_codebook,
            .cache_hint = .temporal,
            .n = n_fc,
            .k = k_fc,
            .w_packed = w_packed,
        }, .{
            .max_m = 32,
            .budget_override = smeTestBudget(64 * 1024),
        });
        defer fc_proj.deinit();

        const m: usize = 16;
        const tap0 = try gpa.alloc(f32, m * h_tgt);
        defer gpa.free(tap0);
        const tap1 = try gpa.alloc(f32, m * h_tgt);
        defer gpa.free(tap1);
        const tap2 = try gpa.alloc(f32, m * h_tgt);
        defer gpa.free(tap2);
        const cat_ref = try gpa.alloc(f32, m * k_fc);
        defer gpa.free(cat_ref);
        for (0..m) |r| {
            for (0..h_tgt) |d| {
                const v0: f32 = @as(f32, @floatFromInt(@as(i32, @intCast((r * 7 + d * 3) % 13)) - 6)) * 0.0625;
                const v1: f32 = @as(f32, @floatFromInt(@as(i32, @intCast((r * 11 + d * 5) % 13)) - 6)) * 0.0625;
                const v2: f32 = @as(f32, @floatFromInt(@as(i32, @intCast((r * 13 + d * 7) % 13)) - 6)) * 0.0625;
                tap0[r * h_tgt + d] = v0;
                tap1[r * h_tgt + d] = v1;
                tap2[r * h_tgt + d] = v2;
                cat_ref[r * k_fc + d] = v0;
                cat_ref[r * k_fc + h_tgt + d] = v1;
                cat_ref[r * k_fc + 2 * h_tgt + d] = v2;
            }
        }
        const ref_y = try gpa.alloc(f32, m * n_fc);
        defer gpa.free(ref_y);
        matmulRowMajorRefF32(cat_ref, w_f32, null, .none, ref_y, m, n_fc, k_fc);

        const got_y = try gpa.alignedAlloc(f32, .fromByteUnits(SVL_BYTES), m * n_fc);
        defer gpa.free(got_y);
        for (backends) |be| {
            try fc_proj.evalEagle3FcFusion(
                &disp,
                .{ .{ .f32 = tap0 }, .{ .f32 = tap1 }, .{ .f32 = tap2 } },
                null,
                std.mem.sliceAsBytes(got_y),
                m,
                .{ .backend = be, .out_dtype = .f32 },
            );
            for (0..m * n_fc) |idx| {
                const tol: f32 = @max(1.5e-2, @abs(ref_y[idx]) * 3.0e-2);
                try std.testing.expectApproxEqAbs(ref_y[idx], got_y[idx], tol);
            }
        }

        // Zero hot-path heap allocations on EAGLE-3 fc fusion!
        const snap0 = try vmstat.Snapshot.capture();
        for (0..16) |_| {
            try fc_proj.evalEagle3FcFusion(
                &disp,
                .{ .{ .f32 = tap0 }, .{ .f32 = tap1 }, .{ .f32 = tap2 } },
                null,
                std.mem.sliceAsBytes(got_y),
                m,
                .{ .out_dtype = .f32 },
            );
        }
        const snap1 = try vmstat.Snapshot.capture();
        const hdelta = vmstat.heapDelta(snap0, snap1);
        try std.testing.expectEqual(@as(i64, 0), hdelta.blocks_in_use);
        try std.testing.expectEqual(@as(i64, 0), hdelta.size_in_use);

        // 4B MTP dual-norm concat (`eh_proj: [H, 2*H]`).
        const k_mtp = 2 * h;
        const w_mtp_f32 = try gpa.alloc(f32, h * k_mtp);
        defer gpa.free(w_mtp_f32);
        for (w_mtp_f32, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 13 + 2) % 15)) - 7)) * 0.0625;
        const w_mtp_bytes = try packedBlobBytes(.f16, h, k_mtp);
        const w_mtp_packed = try gpa.alignedAlloc(u8, .fromByteUnits(SVL_BYTES), w_mtp_bytes);
        defer gpa.free(w_mtp_packed);
        try packDenseFromF32Panel64(.f16, w_mtp_f32, h, k_mtp, w_mtp_packed);

        var mtp_proj = try SmeDraftProjection.init(.{
            .dtype = .f16,
            .param_kind = .affine_scale_bias,
            .cache_hint = .temporal,
            .n = h,
            .k = k_mtp,
            .w_packed = w_mtp_packed,
        }, .{
            .max_m = 32,
            .budget_override = smeTestBudget(64 * 1024),
        });
        defer mtp_proj.deinit();

        var ew: [h]f16 = undefined;
        var hw: [h]f16 = undefined;
        for (0..h) |d| {
            ew[d] = @floatCast(0.8 + 0.05 * @as(f32, @floatFromInt(d % 5)));
            hw[d] = @floatCast(1.1 - 0.05 * @as(f32, @floatFromInt(d % 7)));
        }
        const cat_mtp_ref = try gpa.alloc(f32, m * k_mtp);
        defer gpa.free(cat_mtp_ref);
        for (0..m) |r| {
            var se: f32 = 0.0;
            var sh: f32 = 0.0;
            for (0..h) |d| {
                const ve = tap0[r * h + d];
                const vh = tap1[r * h + d];
                se += ve * ve;
                sh += vh * vh;
            }
            const ie = 1.0 / @sqrt(se / @as(f32, @floatFromInt(h)) + eps);
            const ih = 1.0 / @sqrt(sh / @as(f32, @floatFromInt(h)) + eps);
            for (0..h) |d| {
                cat_mtp_ref[r * k_mtp + d] = (tap0[r * h + d] * ie) * @as(f32, @floatCast(ew[d]));
                cat_mtp_ref[r * k_mtp + h + d] = (tap1[r * h + d] * ih) * @as(f32, @floatCast(hw[d]));
            }
        }
        const ref_mtp = try gpa.alloc(f32, m * h);
        defer gpa.free(ref_mtp);
        matmulRowMajorRefF32(cat_mtp_ref, w_mtp_f32, null, .none, ref_mtp, m, h, k_mtp);

        for (backends) |be| {
            try mtp_proj.evalDualNormConcatLinear(
                &disp,
                .{ .f32 = tap0 },
                .{ .f16 = &ew },
                .{ .f32 = tap1 },
                .{ .f16 = &hw },
                eps,
                null,
                std.mem.sliceAsBytes(got_y),
                m,
                .{ .backend = be, .out_dtype = .f32 },
            );
            for (0..m * h) |idx| {
                const tol: f32 = @max(1.5e-2, @abs(ref_mtp[idx]) * 3.0e-2);
                try std.testing.expectApproxEqAbs(ref_mtp[idx], got_y[idx], tol);
            }
        }
    }
}
