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
const quant_formats = @import("quant_formats.zig");
const amf = @import("../format/amf.zig");
const vmstat = @import("../platform/vmstat.zig");

/// CLI / runtime policy for prefill matmul backend selection (`--prefill-matmul-backend`, ADR-0021).
pub const PrefillBackendMode = enum(u8) {
    auto = 0,
    simdgroup = 1,
    nax = 2,

    pub fn parse(text: []const u8) ?PrefillBackendMode {
        if (std.mem.eql(u8, text, "auto")) return .auto;
        if (std.mem.eql(u8, text, "simdgroup")) return .simdgroup;
        if (std.mem.eql(u8, text, "nax")) return .nax;
        return null;
    }

    pub fn label(self: PrefillBackendMode) []const u8 {
        return switch (self) {
            .auto => "auto",
            .simdgroup => "simdgroup",
            .nax => "nax",
        };
    }
};

/// Diagnostic reason why `NAX` TensorOps is inactive and prefill stays on
/// `simdgroup_apple9` (`G16` + `G17`, ADR-0021).
pub const NaxFallbackReason = enum(u8) {
    none = 0,
    forced_simdgroup = 1,
    unsupported_gpu_family = 2,
    nax_function_missing = 3,
    nax_pso_compile_failed = 4,
    nax_canary_failed = 5,

    pub fn label(self: NaxFallbackReason) []const u8 {
        return switch (self) {
            .none => "none",
            .forced_simdgroup => "forced_simdgroup",
            .unsupported_gpu_family => "unsupported_gpu_family",
            .nax_function_missing => "nax_function_missing",
            .nax_pso_compile_failed => "nax_pso_compile_failed",
            .nax_canary_failed => "nax_canary_failed",
        };
    }
};

/// Active prefill matmul backend (`G16` / `G17`, ADR-0020, ADR-0021).
/// Keeps `.nax_apple10_reserved` as the enum tag name so existing M6.8
/// exhaustive `switch` sites compile unchanged, while exposing `nax_apple10`
/// and `isNax()` for M6.9+ callers.
pub const PrefillMatmulBackend = enum(u8) {
    simdgroup_apple9 = 0,
    nax_apple10_reserved = 1,

    pub const nax_apple10: PrefillMatmulBackend = .nax_apple10_reserved;

    pub fn isNax(self: PrefillMatmulBackend) bool {
        return self == .nax_apple10_reserved;
    }

    pub fn label(self: PrefillMatmulBackend) []const u8 {
        return switch (self) {
            .simdgroup_apple9 => "simdgroup_apple9",
            .nax_apple10_reserved => "nax_apple10",
        };
    }
};

pub const BackendDecision = struct {
    backend: PrefillMatmulBackend,
    fallback_reason: NaxFallbackReason,
};

pub fn selectPrefillMatmulBackendWithMode(
    info: metal.DeviceInfo,
    mode: PrefillBackendMode,
) BackendDecision {
    return switch (mode) {
        .simdgroup => .{
            .backend = .simdgroup_apple9,
            .fallback_reason = .forced_simdgroup,
        },
        .auto, .nax => if (info.supportsNaxTensorOps())
            .{
                .backend = PrefillMatmulBackend.nax_apple10,
                .fallback_reason = .none,
            }
        else
            .{
                .backend = .simdgroup_apple9,
                .fallback_reason = .unsupported_gpu_family,
            },
    };
}

pub fn selectPrefillMatmulBackend(info: metal.DeviceInfo) PrefillMatmulBackend {
    return selectPrefillMatmulBackendWithMode(info, .auto).backend;
}

pub const DequantInitOptions = struct {
    norm_bf16_also: bool = false,
    backend_mode: PrefillBackendMode = .auto,
    strict_nax: bool = false,
    simulate_nax_canary_failure: bool = false,
};

pub const InitError = metal.Error || error{
    NaxRequiresApple10Gpu,
    NaxFunctionMissing,
    NaxPipelineCompileFailed,
};

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
    g18_max_abs_err: f32 = 0.0,
    nax_tested: bool = false,
    nax_demoted: bool = false,
    active_backend: PrefillMatmulBackend = .simdgroup_apple9,
    nax_fallback_reason: NaxFallbackReason = .none,
};

const NaxRuntimeState = struct {
    active_backend: std.atomic.Value(u8),
    fallback_reason: std.atomic.Value(u8),
    canary_demoted: std.atomic.Value(u8),

    fn init(backend: PrefillMatmulBackend, reason: NaxFallbackReason) NaxRuntimeState {
        return .{
            .active_backend = std.atomic.Value(u8).init(@backingInt(backend)),
            .fallback_reason = std.atomic.Value(u8).init(@backingInt(reason)),
            .canary_demoted = std.atomic.Value(u8).init(0),
        };
    }

    fn getBackend(self: *const NaxRuntimeState) PrefillMatmulBackend {
        return @fromBackingInt(@intCast(self.active_backend.load(.acquire)));
    }

    fn getFallbackReason(self: *const NaxRuntimeState) NaxFallbackReason {
        return @fromBackingInt(@intCast(self.fallback_reason.load(.acquire)));
    }

    fn isCanaryDemoted(self: *const NaxRuntimeState) bool {
        return self.canary_demoted.load(.acquire) != 0;
    }

    fn demoteCanaryFailed(self: *NaxRuntimeState) void {
        self.active_backend.store(@backingInt(PrefillMatmulBackend.simdgroup_apple9), .release);
        self.fallback_reason.store(@backingInt(NaxFallbackReason.nax_canary_failed), .release);
        self.canary_demoted.store(1, .release);
    }
};

pub const CanarySliceErrors = struct {
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
) error{MetallibNumericalCanaryFailed}!CanarySliceErrors {
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

/// Verifies one `f16` slice against an `fp32` reference for finiteness and
/// `max_abs_err <= tol` (`ADR-0020` / `ADR-0023` M7.8 G16 startup canary).
pub fn verifyCanaryPair(
    got: []const f16,
    ref: []const f32,
    tol: f32,
) error{MetallibNumericalCanaryFailed}!f32 {
    if (got.len == 0 or got.len != ref.len) return error.MetallibNumericalCanaryFailed;
    var max_err: f32 = 0.0;
    for (got, ref) |g_h, r| {
        const g: f32 = @floatCast(g_h);
        if (std.math.isNan(g) or std.math.isInf(g) or std.math.isNan(r) or std.math.isInf(r)) {
            return error.MetallibNumericalCanaryFailed;
        }
        const err = @abs(g - r);
        if (err > tol) return error.MetallibNumericalCanaryFailed;
        if (err > max_err) max_err = err;
    }
    return max_err;
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
    /// Optional fused residual input `[n]` (`G18`, ADR-0023): when non-null,
    /// computes `y[i] = dot(W_i, x) + residual[i]` in one pass (may alias `y`).
    residual: ?metal.BufferBinding = null,
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
    /// Optional fused residual input `[m, n]` (`G18`, ADR-0023): when non-null,
    /// computes `y[r, c] = dot(W_c, x_r) + residual[r, c]` in one pass (may alias `y`).
    residual: ?metal.BufferBinding = null,
};

/// Uniform payload matching `struct DenseParams` in `dequant_gemm.metal`.
pub const DenseParams = extern struct {
    m: u32,
    n: u32,
    k: u32,
    _pad: u32 = 0,
};

/// Uniform payload matching `struct FusedQkvParams` in `dequant_gemm.metal`.
pub const FusedQkvParams = extern struct {
    m: u32,
    n_q: u32,
    n_k: u32,
    n_v: u32,
    k: u32,
    num_groups: u32,
    row_bytes: u32,
    tgs_q: u32,
    tgs_k: u32,
    tgs_v: u32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
};

comptime {
    if (@sizeOf(DequantParams) != 20) @compileError("DequantParams must be 20 bytes");
    if (@sizeOf(DenseParams) != 16) @compileError("DenseParams must be 16 bytes");
    if (@sizeOf(FusedQkvParams) != 48) @compileError("FusedQkvParams must be 48 bytes");
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
    /// Optional fused residual input `[n]` (`G18`, ADR-0023): when non-null,
    /// computes `y[i] = dot(W_i, x) + residual[i]` in one pass (may alias `y`).
    residual: ?metal.BufferBinding = null,
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
    /// Optional fused residual input `[m, n]` (`G18`, ADR-0023): when non-null,
    /// computes `y[r, c] = dot(W_c, x_r) + residual[r, c]` in one pass (may alias `y`).
    residual: ?metal.BufferBinding = null,
};

/// Returns `true` if `dtype` is supported by the homogeneous G18 fused linear
/// kernels (`f16`, `bf16`, `q4_affine`, `q3_affine`, `q2_affine`).
pub fn isFusableLinearDType(dtype: amf.DType) bool {
    return switch (dtype) {
        .f16, .bf16, .q4_affine, .q3_affine, .q2_affine => true,
        else => false,
    };
}

pub fn weightDTypeToAmf(dtype: quant.WeightDType) amf.DType {
    return switch (dtype) {
        .f16 => .f16,
        .bf16 => .bf16,
        .q4_affine => .q4_affine,
        .q3_affine => .q3_affine,
        .q2_affine => .q2_affine,
    };
}

/// Heterogeneous-GSQ guard (`G18`, ADR-0023 §1): returns `true` strictly when
/// `q`, `k`, and `v` projections share one homogeneous supported `DType`.
pub fn canFuseQkv(q_dtype: amf.DType, k_dtype: amf.DType, v_dtype: amf.DType) bool {
    return q_dtype == k_dtype and k_dtype == v_dtype and isFusableLinearDType(q_dtype);
}

/// Heterogeneous-GSQ guard (`G18`, ADR-0023 §1): returns `true` strictly when
/// dense FFN `gate_proj` and `up_proj` share one homogeneous supported `DType`.
pub fn canFuseGateUp(gate_dtype: amf.DType, up_dtype: amf.DType) bool {
    return gate_dtype == up_dtype and isFusableLinearDType(gate_dtype);
}

/// Reports whether a G18 guarded fusion helper executed the fused single-pass
/// kernel or fell back to the unfused path for a mixed-DType GSQ layer.
pub const FusionPath = enum(u8) {
    fused = 0,
    unfused_heterogeneous = 1,
};

/// Single projection descriptor for `FusedQkvGemvArgs` / `FusedQkvGemmArgs`.
pub const QkvProjBinding = struct {
    dtype: amf.DType,
    weight: metal.BufferBinding,
    /// Required for `q2_affine`/`q3_affine`/`q4_affine`; ignored (`null`) for `f16`/`bf16`.
    scale_bias: ?metal.BufferBinding = null,
    y: metal.BufferBinding,
    n: u32,
};

/// Arguments for homogeneous-DType fused QKV GEMV (`M = 1`, `G18`, ADR-0023).
pub const FusedQkvGemvArgs = struct {
    x: metal.BufferBinding,
    q: QkvProjBinding,
    k_proj: QkvProjBinding,
    v: QkvProjBinding,
    k: u32,
    /// When `true` (default), mixed-DType GSQ layers automatically route to the
    /// 3 unfused projection dispatches instead of returning `error.HeterogeneousDType`.
    allow_unfused_fallback: bool = true,
};

/// Arguments for homogeneous-DType fused QKV GEMM (`M >= 1`, `G18`, ADR-0023).
pub const FusedQkvGemmArgs = struct {
    x: metal.BufferBinding,
    q: QkvProjBinding,
    k_proj: QkvProjBinding,
    v: QkvProjBinding,
    m: u32,
    k: u32,
    /// When `true` (default), mixed-DType GSQ layers automatically route to the
    /// 3 unfused projection dispatches instead of returning `error.HeterogeneousDType`.
    allow_unfused_fallback: bool = true,
};

/// Arguments for homogeneous-DType fused dense FFN `Gate + Up + SwiGLU` (`G18`,
/// ADR-0023): computes `y = SiLU(x @ W_gate^T) * (x @ W_up^T)` without writing
/// `2 * m * n` intermediate activations to unified DRAM.
pub const DenseGateUpSwigluArgs = struct {
    x: metal.BufferBinding,
    gate_dtype: amf.DType,
    w_gate: metal.BufferBinding,
    sb_gate: ?metal.BufferBinding = null,
    up_dtype: amf.DType,
    w_up: metal.BufferBinding,
    sb_up: ?metal.BufferBinding = null,
    y: metal.BufferBinding,
    m: u32,
    /// `intermediate_size`.
    n: u32,
    /// `hidden_size`.
    k: u32,
};

pub const EncodeError = metal.Error || quant.ShapeError;
pub const FusedEncodeError = EncodeError || error{
    HeterogeneousDType,
    UnsupportedDType,
};

fn mapQfEncodeError(err: quant_formats.EncodeError) EncodeError {
    return switch (err) {
        error.InvalidGroupSize, error.PipelineNotInitialized => error.InvalidArgument,
        else => |e| e,
    };
}

/// Returns `true` when `(amf_dtype, group_size)` routes through `QuantFormatPipelines`
/// (`G23`, ADR-0024) rather than the legacy dense FP16/BF16 or group-64 q2/q3/q4 kernels.
pub fn usesQuantFormats(amf_dtype: amf.DType, group_size: u16) bool {
    return switch (amf_dtype) {
        .f16, .bf16, .f32, .u32 => false,
        .q4_affine, .q3_affine, .q2_affine => group_size != 0 and group_size != 64,
        else => true,
    };
}

/// Returns `true` for every 2D linear / MoE weight `.amf` `DType` executable by
/// `DequantPipelines` + `QuantFormatPipelines` (`f16`, `bf16`, and all 34 `W21`/`W22`
/// quantized formats `2..37`).
pub fn isSupportedLinearAmfDType(dtype: amf.DType) bool {
    return switch (dtype) {
        .f16, .bf16 => true,
        .f32, .u32 => false,
        else => quant_formats.DType.fromAmf(dtype) != null,
    };
}

/// Returns `true` when an `fp8_e4m3_tensor` blob carries per-row `f32[N]` scales
/// (`params_bytes == N * 4`, `N > 1`) instead of a single scalar `f32[1]` scale.
pub fn isTensorScalePerRow(dtype: amf.DType, n: u32, params_bytes: u64) bool {
    return dtype == .fp8_e4m3_tensor and n > 1 and params_bytes == @as(u64, n) * 4;
}

/// Validates `[n, k]` shape alignment for any supported linear `amf.DType` and `group_size`.
pub fn validateLinearAmfShape(
    dtype: amf.DType,
    n: u32,
    k: u32,
    group_size: u16,
) (quant_formats.ShapeError || error{UnexpectedTensorDType})!void {
    if (n == 0 or k == 0) return error.ZeroDimension;
    switch (dtype) {
        .f16, .bf16 => return,
        .f32, .u32 => return error.UnexpectedTensorDType,
        else => {
            const qf_dt = quant_formats.DType.fromAmf(dtype) orelse return error.UnexpectedTensorDType;
            try qf_dt.validateShape(n, k, group_size);
        },
    }
}

/// Legacy `WeightDType` approximation for struct compatibility; authoritative dispatch
/// uses `amf_dtype` + `group_size` via `usesQuantFormats`.
pub fn legacyWeightDTypeForAmf(dtype: amf.DType) quant.WeightDType {
    return switch (dtype) {
        .f16 => .f16,
        .bf16 => .bf16,
        .q4_affine => .q4_affine,
        .q3_affine => .q3_affine,
        .q2_affine => .q2_affine,
        else => .q4_affine,
    };
}

pub const UnifiedLinearMode = enum(u8) {
    gemv,
    batched,
    gemm32,
    gemm_wide,
};

pub const UnifiedLinearArgs = struct {
    amf_dtype: amf.DType,
    group_size: u16 = 64,
    tensor_scale_per_row: bool = false,
    weight: metal.BufferBinding,
    params: ?metal.BufferBinding = null,
    x: metal.BufferBinding,
    y: metal.BufferBinding,
    residual: ?metal.BufferBinding = null,
    m: u32,
    n: u32,
    k: u32,
    mode: UnifiedLinearMode = .gemv,
    qf_pipes: ?*const quant_formats.QuantFormatPipelines = null,
    tile_f16: ?metal.BufferBinding = null,
    tile_f16_bytes: u64 = 0,
    act_quant_policy: quant_formats.ActQuantPolicy = .none,
    act_scratch: ?metal.BufferBinding = null,
    act_scratch_bytes: u64 = 0,
};

/// Startup-compiled specialized Metal 4 pipelines for 2-bit, 3-bit and 4-bit
/// group-64 affine dequantization GEMV and GEMM (G1, G2, G13), dense FP16/BF16
/// GEMV/GEMM (G12, G15), and lazily instantiated Apple10+ NAX TensorOps GEMM
/// pipelines (G17, ADR-0021).
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
    /// M7.8 (`G18`, ADR-0023): function-constant-specialized (`has_residual_epilogue = true`)
    /// variants computing `y[i] = dot(W_i, x) + residual[i]` in a single pass.
    gemv_q4_res: metal.Pipeline,
    gemv_q3_res: metal.Pipeline,
    gemv_q2_res: metal.Pipeline,
    gemm_q4_res: metal.Pipeline,
    gemm_q3_res: metal.Pipeline,
    gemm_q2_res: metal.Pipeline,
    gemm_wide_q4_res: metal.Pipeline,
    gemm_wide_q3_res: metal.Pipeline,
    gemm_wide_q2_res: metal.Pipeline,
    gemv_batch_q4_res: metal.Pipeline,
    gemv_batch_q3_res: metal.Pipeline,
    gemv_batch_q2_res: metal.Pipeline,
    gemv_f16_res: metal.Pipeline,
    gemv_bf16_res: metal.Pipeline,
    gemm_f16_res: metal.Pipeline,
    gemm_bf16_res: metal.Pipeline,
    gemm_wide_f16_res: metal.Pipeline,
    gemm_wide_bf16_res: metal.Pipeline,
    gemv_batch_f16_res: metal.Pipeline,
    gemv_batch_bf16_res: metal.Pipeline,
    /// M7.8 (`G18`, ADR-0023): homogeneous-DType fused QKV and dense Gate+Up+SwiGLU.
    fused_qkv_gemv: HomogeneousLinearSet,
    fused_qkv_gemm: HomogeneousLinearSet,
    dense_swiglu_gemv: HomogeneousLinearSet,
    dense_swiglu_gemm: HomogeneousLinearSet,
    /// Lazily instantiated Apple10+ NAX TensorOps pipelines (null on Apple9 / M4).
    gemm_nax_q4: ?metal.Pipeline = null,
    gemm_nax_q3: ?metal.Pipeline = null,
    gemm_nax_q2: ?metal.Pipeline = null,
    gemm_nax_f16: ?metal.Pipeline = null,
    gemm_nax_bf16: ?metal.Pipeline = null,
    gemm_nax_q4_res: ?metal.Pipeline = null,
    gemm_nax_q3_res: ?metal.Pipeline = null,
    gemm_nax_q2_res: ?metal.Pipeline = null,
    gemm_nax_f16_res: ?metal.Pipeline = null,
    gemm_nax_bf16_res: ?metal.Pipeline = null,
    nax_pipelines_compiled: bool = false,
    active_backend: PrefillMatmulBackend = .simdgroup_apple9,
    nax_fallback_reason: NaxFallbackReason = .unsupported_gpu_family,
    strict_nax: bool = false,
    simulate_nax_canary_failure: bool = false,
    nax_state: ?*NaxRuntimeState = null,

    pub const HomogeneousLinearSet = struct {
        f16: metal.Pipeline,
        bf16: metal.Pipeline,
        q4: metal.Pipeline,
        q3: metal.Pipeline,
        q2: metal.Pipeline,

        fn init(device: metal.Device, library: metal.Library, fn_name: [:0]const u8) metal.Error!HomogeneousLinearSet {
            const p_f16 = try metal.Pipeline.initSpecialized(device, library, fn_name, &.{
                .{ .index = 0, .value = .{ .u32 = 0 } },
                .{ .index = 1, .value = .{ .u32 = 64 } },
                .{ .index = 2, .value = .{ .bool = false } },
            });
            errdefer p_f16.close();

            const p_bf16 = try metal.Pipeline.initSpecialized(device, library, fn_name, &.{
                .{ .index = 0, .value = .{ .u32 = 0 } },
                .{ .index = 1, .value = .{ .u32 = 64 } },
                .{ .index = 2, .value = .{ .bool = true } },
            });
            errdefer p_bf16.close();

            const p_q4 = try metal.Pipeline.initSpecialized(device, library, fn_name, &.{
                .{ .index = 0, .value = .{ .u32 = 4 } },
                .{ .index = 1, .value = .{ .u32 = 64 } },
                .{ .index = 2, .value = .{ .bool = false } },
            });
            errdefer p_q4.close();

            const p_q3 = try metal.Pipeline.initSpecialized(device, library, fn_name, &.{
                .{ .index = 0, .value = .{ .u32 = 3 } },
                .{ .index = 1, .value = .{ .u32 = 64 } },
                .{ .index = 2, .value = .{ .bool = false } },
            });
            errdefer p_q3.close();

            const p_q2 = try metal.Pipeline.initSpecialized(device, library, fn_name, &.{
                .{ .index = 0, .value = .{ .u32 = 2 } },
                .{ .index = 1, .value = .{ .u32 = 64 } },
                .{ .index = 2, .value = .{ .bool = false } },
            });
            errdefer p_q2.close();

            return .{
                .f16 = p_f16,
                .bf16 = p_bf16,
                .q4 = p_q4,
                .q3 = p_q3,
                .q2 = p_q2,
            };
        }

        fn select(self: HomogeneousLinearSet, dtype: amf.DType) ?metal.Pipeline {
            return switch (dtype) {
                .f16 => self.f16,
                .bf16 => self.bf16,
                .q4_affine => self.q4,
                .q3_affine => self.q3,
                .q2_affine => self.q2,
                else => null,
            };
        }

        fn close(self: HomogeneousLinearSet) void {
            self.q2.close();
            self.q3.close();
            self.q4.close();
            self.bf16.close();
            self.f16.close();
        }
    };

    const NaxCompiledSet = struct {
        gemm_nax_q4: metal.Pipeline,
        gemm_nax_q3: metal.Pipeline,
        gemm_nax_q2: metal.Pipeline,
        gemm_nax_f16: metal.Pipeline,
        gemm_nax_bf16: metal.Pipeline,
        gemm_nax_q4_res: metal.Pipeline,
        gemm_nax_q3_res: metal.Pipeline,
        gemm_nax_q2_res: metal.Pipeline,
        gemm_nax_f16_res: metal.Pipeline,
        gemm_nax_bf16_res: metal.Pipeline,

        fn close(self: NaxCompiledSet) void {
            self.gemm_nax_bf16_res.close();
            self.gemm_nax_f16_res.close();
            self.gemm_nax_q2_res.close();
            self.gemm_nax_q3_res.close();
            self.gemm_nax_q4_res.close();
            self.gemm_nax_bf16.close();
            self.gemm_nax_f16.close();
            self.gemm_nax_q2.close();
            self.gemm_nax_q3.close();
            self.gemm_nax_q4.close();
        }
    };

    fn compileNaxPipelines(device: metal.Device, library: metal.Library) metal.Error!NaxCompiledSet {
        const gemm_nax_f16 = try metal.Pipeline.initSpecialized(device, library, "ami_dense_gemm_nax", &.{
            .{ .index = 2, .value = .{ .bool = false } },
            .{ .index = 3, .value = .{ .bool = false } },
        });
        errdefer gemm_nax_f16.close();

        const gemm_nax_bf16 = try metal.Pipeline.initSpecialized(device, library, "ami_dense_gemm_nax", &.{
            .{ .index = 2, .value = .{ .bool = true } },
            .{ .index = 3, .value = .{ .bool = false } },
        });
        errdefer gemm_nax_bf16.close();

        const gemm_nax_q4 = try metal.Pipeline.initSpecialized(device, library, "ami_dequant_gemm_nax", &.{
            .{ .index = 0, .value = .{ .u32 = 4 } },
            .{ .index = 1, .value = .{ .u32 = 64 } },
            .{ .index = 3, .value = .{ .bool = false } },
        });
        errdefer gemm_nax_q4.close();

        const gemm_nax_q3 = try metal.Pipeline.initSpecialized(device, library, "ami_dequant_gemm_nax", &.{
            .{ .index = 0, .value = .{ .u32 = 3 } },
            .{ .index = 1, .value = .{ .u32 = 64 } },
            .{ .index = 3, .value = .{ .bool = false } },
        });
        errdefer gemm_nax_q3.close();

        const gemm_nax_q2 = try metal.Pipeline.initSpecialized(device, library, "ami_dequant_gemm_nax", &.{
            .{ .index = 0, .value = .{ .u32 = 2 } },
            .{ .index = 1, .value = .{ .u32 = 64 } },
            .{ .index = 3, .value = .{ .bool = false } },
        });
        errdefer gemm_nax_q2.close();

        const gemm_nax_f16_res = try metal.Pipeline.initSpecialized(device, library, "ami_dense_gemm_nax", &.{
            .{ .index = 2, .value = .{ .bool = false } },
            .{ .index = 3, .value = .{ .bool = true } },
        });
        errdefer gemm_nax_f16_res.close();

        const gemm_nax_bf16_res = try metal.Pipeline.initSpecialized(device, library, "ami_dense_gemm_nax", &.{
            .{ .index = 2, .value = .{ .bool = true } },
            .{ .index = 3, .value = .{ .bool = true } },
        });
        errdefer gemm_nax_bf16_res.close();

        const gemm_nax_q4_res = try metal.Pipeline.initSpecialized(device, library, "ami_dequant_gemm_nax", &.{
            .{ .index = 0, .value = .{ .u32 = 4 } },
            .{ .index = 1, .value = .{ .u32 = 64 } },
            .{ .index = 3, .value = .{ .bool = true } },
        });
        errdefer gemm_nax_q4_res.close();

        const gemm_nax_q3_res = try metal.Pipeline.initSpecialized(device, library, "ami_dequant_gemm_nax", &.{
            .{ .index = 0, .value = .{ .u32 = 3 } },
            .{ .index = 1, .value = .{ .u32 = 64 } },
            .{ .index = 3, .value = .{ .bool = true } },
        });
        errdefer gemm_nax_q3_res.close();

        const gemm_nax_q2_res = try metal.Pipeline.initSpecialized(device, library, "ami_dequant_gemm_nax", &.{
            .{ .index = 0, .value = .{ .u32 = 2 } },
            .{ .index = 1, .value = .{ .u32 = 64 } },
            .{ .index = 3, .value = .{ .bool = true } },
        });
        errdefer gemm_nax_q2_res.close();

        return .{
            .gemm_nax_q4 = gemm_nax_q4,
            .gemm_nax_q3 = gemm_nax_q3,
            .gemm_nax_q2 = gemm_nax_q2,
            .gemm_nax_f16 = gemm_nax_f16,
            .gemm_nax_bf16 = gemm_nax_bf16,
            .gemm_nax_q4_res = gemm_nax_q4_res,
            .gemm_nax_q3_res = gemm_nax_q3_res,
            .gemm_nax_q2_res = gemm_nax_q2_res,
            .gemm_nax_f16_res = gemm_nax_f16_res,
            .gemm_nax_bf16_res = gemm_nax_bf16_res,
        };
    }

    pub fn init(device: metal.Device, library: metal.Library) metal.Error!DequantPipelines {
        return initWithOptions(device, library, .{}) catch |err| switch (err) {
            error.NaxRequiresApple10Gpu,
            error.NaxFunctionMissing,
            error.NaxPipelineCompileFailed,
            => unreachable,
            else => |e| return e,
        };
    }

    pub fn initWithOptions(
        device: metal.Device,
        library: metal.Library,
        opts: DequantInitOptions,
    ) InitError!DequantPipelines {
        _ = opts.norm_bf16_also;
        const dev_info = try device.info();
        const supports_nax = dev_info.supportsNaxTensorOps();

        // Fail fast before compiling 20 Metal PSOs when `--prefill-matmul-backend nax`
        // is combined with `--strict-nax` on an Apple9 or older GPU.
        if (!supports_nax and opts.backend_mode == .nax and opts.strict_nax) {
            return error.NaxRequiresApple10Gpu;
        }

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
            .{ .index = 3, .value = .{ .bool = false } },
        });
        errdefer gemv_batch_bf16.close();

        // M7.8 (G18, ADR-0023): fused residual-epilogue (`has_residual_epilogue = true`) PSOs.
        const gemv_q4_res = try metal.Pipeline.initSpecialized(device, library, "ami_dequant_gemv", &.{
            .{ .index = 0, .value = .{ .u32 = 4 } },
            .{ .index = 1, .value = .{ .u32 = 64 } },
            .{ .index = 3, .value = .{ .bool = true } },
        });
        errdefer gemv_q4_res.close();

        const gemv_q3_res = try metal.Pipeline.initSpecialized(device, library, "ami_dequant_gemv", &.{
            .{ .index = 0, .value = .{ .u32 = 3 } },
            .{ .index = 1, .value = .{ .u32 = 64 } },
            .{ .index = 3, .value = .{ .bool = true } },
        });
        errdefer gemv_q3_res.close();

        const gemv_q2_res = try metal.Pipeline.initSpecialized(device, library, "ami_dequant_gemv", &.{
            .{ .index = 0, .value = .{ .u32 = 2 } },
            .{ .index = 1, .value = .{ .u32 = 64 } },
            .{ .index = 3, .value = .{ .bool = true } },
        });
        errdefer gemv_q2_res.close();

        const gemm_q4_res = try metal.Pipeline.initSpecialized(device, library, "ami_dequant_gemm", &.{
            .{ .index = 0, .value = .{ .u32 = 4 } },
            .{ .index = 1, .value = .{ .u32 = 64 } },
            .{ .index = 3, .value = .{ .bool = true } },
        });
        errdefer gemm_q4_res.close();

        const gemm_q3_res = try metal.Pipeline.initSpecialized(device, library, "ami_dequant_gemm", &.{
            .{ .index = 0, .value = .{ .u32 = 3 } },
            .{ .index = 1, .value = .{ .u32 = 64 } },
            .{ .index = 3, .value = .{ .bool = true } },
        });
        errdefer gemm_q3_res.close();

        const gemm_q2_res = try metal.Pipeline.initSpecialized(device, library, "ami_dequant_gemm", &.{
            .{ .index = 0, .value = .{ .u32 = 2 } },
            .{ .index = 1, .value = .{ .u32 = 64 } },
            .{ .index = 3, .value = .{ .bool = true } },
        });
        errdefer gemm_q2_res.close();

        const gemm_wide_q4_res = try metal.Pipeline.initSpecialized(device, library, "ami_dequant_gemm_wide", &.{
            .{ .index = 0, .value = .{ .u32 = 4 } },
            .{ .index = 1, .value = .{ .u32 = 64 } },
            .{ .index = 3, .value = .{ .bool = true } },
        });
        errdefer gemm_wide_q4_res.close();

        const gemm_wide_q3_res = try metal.Pipeline.initSpecialized(device, library, "ami_dequant_gemm_wide", &.{
            .{ .index = 0, .value = .{ .u32 = 3 } },
            .{ .index = 1, .value = .{ .u32 = 64 } },
            .{ .index = 3, .value = .{ .bool = true } },
        });
        errdefer gemm_wide_q3_res.close();

        const gemm_wide_q2_res = try metal.Pipeline.initSpecialized(device, library, "ami_dequant_gemm_wide", &.{
            .{ .index = 0, .value = .{ .u32 = 2 } },
            .{ .index = 1, .value = .{ .u32 = 64 } },
            .{ .index = 3, .value = .{ .bool = true } },
        });
        errdefer gemm_wide_q2_res.close();

        const gemv_batch_q4_res = try metal.Pipeline.initSpecialized(device, library, "ami_dequant_gemv_batch", &.{
            .{ .index = 0, .value = .{ .u32 = 4 } },
            .{ .index = 1, .value = .{ .u32 = 64 } },
            .{ .index = 3, .value = .{ .bool = true } },
        });
        errdefer gemv_batch_q4_res.close();

        const gemv_batch_q3_res = try metal.Pipeline.initSpecialized(device, library, "ami_dequant_gemv_batch", &.{
            .{ .index = 0, .value = .{ .u32 = 3 } },
            .{ .index = 1, .value = .{ .u32 = 64 } },
            .{ .index = 3, .value = .{ .bool = true } },
        });
        errdefer gemv_batch_q3_res.close();

        const gemv_batch_q2_res = try metal.Pipeline.initSpecialized(device, library, "ami_dequant_gemv_batch", &.{
            .{ .index = 0, .value = .{ .u32 = 2 } },
            .{ .index = 1, .value = .{ .u32 = 64 } },
            .{ .index = 3, .value = .{ .bool = true } },
        });
        errdefer gemv_batch_q2_res.close();

        const gemv_f16_res = try metal.Pipeline.initSpecialized(device, library, "ami_dense_gemv", &.{
            .{ .index = 2, .value = .{ .bool = false } },
            .{ .index = 3, .value = .{ .bool = true } },
        });
        errdefer gemv_f16_res.close();

        const gemv_bf16_res = try metal.Pipeline.initSpecialized(device, library, "ami_dense_gemv", &.{
            .{ .index = 2, .value = .{ .bool = true } },
            .{ .index = 3, .value = .{ .bool = true } },
        });
        errdefer gemv_bf16_res.close();

        const gemm_f16_res = try metal.Pipeline.initSpecialized(device, library, "ami_dense_gemm", &.{
            .{ .index = 2, .value = .{ .bool = false } },
            .{ .index = 3, .value = .{ .bool = true } },
        });
        errdefer gemm_f16_res.close();

        const gemm_bf16_res = try metal.Pipeline.initSpecialized(device, library, "ami_dense_gemm", &.{
            .{ .index = 2, .value = .{ .bool = true } },
            .{ .index = 3, .value = .{ .bool = true } },
        });
        errdefer gemm_bf16_res.close();

        const gemm_wide_f16_res = try metal.Pipeline.initSpecialized(device, library, "ami_dense_gemm_wide", &.{
            .{ .index = 2, .value = .{ .bool = false } },
            .{ .index = 3, .value = .{ .bool = true } },
        });
        errdefer gemm_wide_f16_res.close();

        const gemm_wide_bf16_res = try metal.Pipeline.initSpecialized(device, library, "ami_dense_gemm_wide", &.{
            .{ .index = 2, .value = .{ .bool = true } },
            .{ .index = 3, .value = .{ .bool = true } },
        });
        errdefer gemm_wide_bf16_res.close();

        const gemv_batch_f16_res = try metal.Pipeline.initSpecialized(device, library, "ami_dense_gemv_batch", &.{
            .{ .index = 2, .value = .{ .bool = false } },
            .{ .index = 3, .value = .{ .bool = true } },
        });
        errdefer gemv_batch_f16_res.close();

        const gemv_batch_bf16_res = try metal.Pipeline.initSpecialized(device, library, "ami_dense_gemv_batch", &.{
            .{ .index = 2, .value = .{ .bool = true } },
            .{ .index = 3, .value = .{ .bool = true } },
        });
        errdefer gemv_batch_bf16_res.close();

        // M7.8 (G18, ADR-0023): homogeneous-DType fused QKV and dense Gate+Up+SwiGLU PSOs.
        const fused_qkv_gemv = try HomogeneousLinearSet.init(device, library, "ami_fused_qkv_gemv");
        errdefer fused_qkv_gemv.close();

        const fused_qkv_gemm = try HomogeneousLinearSet.init(device, library, "ami_fused_qkv_gemm");
        errdefer fused_qkv_gemm.close();

        const dense_swiglu_gemv = try HomogeneousLinearSet.init(device, library, "ami_dense_gate_up_swiglu");
        errdefer dense_swiglu_gemv.close();

        const dense_swiglu_gemm = try HomogeneousLinearSet.init(device, library, "ami_dense_gate_up_swiglu_gemm");
        errdefer dense_swiglu_gemm.close();

        // Lazy Apple10+ NAX PSO compilation (`G17`, ADR-0021): skipped completely
        // when forced to `simdgroup` or on `Apple9` (M4) so M4 startup pays zero
        // extra `MTL4Compiler` latency.
        var nax_set: ?NaxCompiledSet = null;
        errdefer if (nax_set) |ns| ns.close();
        var active_backend: PrefillMatmulBackend = .simdgroup_apple9;
        var fallback_reason: NaxFallbackReason = .unsupported_gpu_family;

        switch (opts.backend_mode) {
            .simdgroup => {
                active_backend = .simdgroup_apple9;
                fallback_reason = .forced_simdgroup;
            },
            .auto, .nax => {
                if (!supports_nax) {
                    active_backend = .simdgroup_apple9;
                    fallback_reason = .unsupported_gpu_family;
                } else {
                    const has_dense = try library.hasFunction("ami_dense_gemm_nax");
                    const has_dequant = try library.hasFunction("ami_dequant_gemm_nax");
                    if (!has_dense or !has_dequant) {
                        if (opts.strict_nax) return error.NaxFunctionMissing;
                        active_backend = .simdgroup_apple9;
                        fallback_reason = .nax_function_missing;
                    } else if (compileNaxPipelines(device, library)) |compiled| {
                        nax_set = compiled;
                        active_backend = PrefillMatmulBackend.nax_apple10;
                        fallback_reason = .none;
                    } else |err| {
                        // Explicitly classify PSO compilation failure vs OOM/invalid.
                        if (err == error.OutOfMemory) return error.OutOfMemory;
                        if (opts.strict_nax) return error.NaxPipelineCompileFailed;
                        active_backend = .simdgroup_apple9;
                        fallback_reason = .nax_pso_compile_failed;
                    }
                }
            },
        }

        const nax_compiled = (nax_set != null);
        var nax_state: ?*NaxRuntimeState = null;
        errdefer if (nax_state) |st| std.heap.c_allocator.destroy(st);
        if (nax_compiled or opts.simulate_nax_canary_failure) {
            const st = try std.heap.c_allocator.create(NaxRuntimeState);
            st.* = NaxRuntimeState.init(active_backend, fallback_reason);
            nax_state = st;
        }

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
            .gemv_q4_res = gemv_q4_res,
            .gemv_q3_res = gemv_q3_res,
            .gemv_q2_res = gemv_q2_res,
            .gemm_q4_res = gemm_q4_res,
            .gemm_q3_res = gemm_q3_res,
            .gemm_q2_res = gemm_q2_res,
            .gemm_wide_q4_res = gemm_wide_q4_res,
            .gemm_wide_q3_res = gemm_wide_q3_res,
            .gemm_wide_q2_res = gemm_wide_q2_res,
            .gemv_batch_q4_res = gemv_batch_q4_res,
            .gemv_batch_q3_res = gemv_batch_q3_res,
            .gemv_batch_q2_res = gemv_batch_q2_res,
            .gemv_f16_res = gemv_f16_res,
            .gemv_bf16_res = gemv_bf16_res,
            .gemm_f16_res = gemm_f16_res,
            .gemm_bf16_res = gemm_bf16_res,
            .gemm_wide_f16_res = gemm_wide_f16_res,
            .gemm_wide_bf16_res = gemm_wide_bf16_res,
            .gemv_batch_f16_res = gemv_batch_f16_res,
            .gemv_batch_bf16_res = gemv_batch_bf16_res,
            .fused_qkv_gemv = fused_qkv_gemv,
            .fused_qkv_gemm = fused_qkv_gemm,
            .dense_swiglu_gemv = dense_swiglu_gemv,
            .dense_swiglu_gemm = dense_swiglu_gemm,
            .gemm_nax_q4 = if (nax_set) |ns| ns.gemm_nax_q4 else null,
            .gemm_nax_q3 = if (nax_set) |ns| ns.gemm_nax_q3 else null,
            .gemm_nax_q2 = if (nax_set) |ns| ns.gemm_nax_q2 else null,
            .gemm_nax_f16 = if (nax_set) |ns| ns.gemm_nax_f16 else null,
            .gemm_nax_bf16 = if (nax_set) |ns| ns.gemm_nax_bf16 else null,
            .gemm_nax_q4_res = if (nax_set) |ns| ns.gemm_nax_q4_res else null,
            .gemm_nax_q3_res = if (nax_set) |ns| ns.gemm_nax_q3_res else null,
            .gemm_nax_q2_res = if (nax_set) |ns| ns.gemm_nax_q2_res else null,
            .gemm_nax_f16_res = if (nax_set) |ns| ns.gemm_nax_f16_res else null,
            .gemm_nax_bf16_res = if (nax_set) |ns| ns.gemm_nax_bf16_res else null,
            .nax_pipelines_compiled = nax_compiled,
            .active_backend = active_backend,
            .nax_fallback_reason = fallback_reason,
            .strict_nax = opts.strict_nax,
            .simulate_nax_canary_failure = opts.simulate_nax_canary_failure,
            .nax_state = nax_state,
        };
    }

    pub fn activeBackend(self: DequantPipelines) PrefillMatmulBackend {
        if (self.nax_state) |st| return st.getBackend();
        return self.active_backend;
    }

    pub fn naxFallbackReason(self: DequantPipelines) NaxFallbackReason {
        if (self.nax_state) |st| return st.getFallbackReason();
        return self.nax_fallback_reason;
    }

    pub fn useNaxPrefill(self: DequantPipelines) bool {
        return self.nax_pipelines_compiled and self.activeBackend().isNax();
    }

    /// Synchronizes the plain struct mirror fields (`active_backend`,
    /// `nax_fallback_reason`) with the live atomic canary state.
    pub fn syncNaxState(self: *DequantPipelines) void {
        self.active_backend = self.activeBackend();
        self.nax_fallback_reason = self.naxFallbackReason();
    }

    pub fn close(self: DequantPipelines) void {
        if (self.nax_state) |st| std.heap.c_allocator.destroy(st);
        if (self.gemm_nax_bf16_res) |p| p.close();
        if (self.gemm_nax_f16_res) |p| p.close();
        if (self.gemm_nax_q2_res) |p| p.close();
        if (self.gemm_nax_q3_res) |p| p.close();
        if (self.gemm_nax_q4_res) |p| p.close();
        if (self.gemm_nax_bf16) |p| p.close();
        if (self.gemm_nax_f16) |p| p.close();
        if (self.gemm_nax_q2) |p| p.close();
        if (self.gemm_nax_q3) |p| p.close();
        if (self.gemm_nax_q4) |p| p.close();
        self.dense_swiglu_gemm.close();
        self.dense_swiglu_gemv.close();
        self.fused_qkv_gemm.close();
        self.fused_qkv_gemv.close();
        self.gemv_batch_bf16_res.close();
        self.gemv_batch_f16_res.close();
        self.gemm_wide_bf16_res.close();
        self.gemm_wide_f16_res.close();
        self.gemm_bf16_res.close();
        self.gemm_f16_res.close();
        self.gemv_bf16_res.close();
        self.gemv_f16_res.close();
        self.gemv_batch_q2_res.close();
        self.gemv_batch_q3_res.close();
        self.gemv_batch_q4_res.close();
        self.gemm_wide_q2_res.close();
        self.gemm_wide_q3_res.close();
        self.gemm_wide_q4_res.close();
        self.gemm_q2_res.close();
        self.gemm_q3_res.close();
        self.gemm_q4_res.close();
        self.gemv_q2_res.close();
        self.gemv_q3_res.close();
        self.gemv_q4_res.close();
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
        const has_res = (args.residual != null);
        const num_groups, const row_bytes, const pipe = switch (bits) {
            .q4 => blk: {
                const Ops = quant.QuantOps(.{ .bits = .q4, .group_size = 64 });
                break :blk .{ try Ops.groupsPerRow(args.k), try Ops.rowWeightBytes(args.k), if (has_res) self.gemv_q4_res else self.gemv_q4 };
            },
            .q3 => blk: {
                const Ops = quant.QuantOps(.{ .bits = .q3, .group_size = 64 });
                break :blk .{ try Ops.groupsPerRow(args.k), try Ops.rowWeightBytes(args.k), if (has_res) self.gemv_q3_res else self.gemv_q3 };
            },
            .q2 => blk: {
                const Ops = quant.QuantOps(.{ .bits = .q2, .group_size = 64 });
                break :blk .{ try Ops.groupsPerRow(args.k), try Ops.rowWeightBytes(args.k), if (has_res) self.gemv_q2_res else self.gemv_q2 };
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
            &.{ args.x, args.qweight, args.scale_bias, args.y, args.residual orelse args.y },
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
        const has_res = (args.residual != null);
        const num_groups, const row_bytes, const pipe = switch (bits) {
            .q4 => blk: {
                const Ops = quant.QuantOps(.{ .bits = .q4, .group_size = 64 });
                break :blk .{ try Ops.groupsPerRow(args.k), try Ops.rowWeightBytes(args.k), if (has_res) self.gemm_q4_res else self.gemm_q4 };
            },
            .q3 => blk: {
                const Ops = quant.QuantOps(.{ .bits = .q3, .group_size = 64 });
                break :blk .{ try Ops.groupsPerRow(args.k), try Ops.rowWeightBytes(args.k), if (has_res) self.gemm_q3_res else self.gemm_q3 };
            },
            .q2 => blk: {
                const Ops = quant.QuantOps(.{ .bits = .q2, .group_size = 64 });
                break :blk .{ try Ops.groupsPerRow(args.k), try Ops.rowWeightBytes(args.k), if (has_res) self.gemm_q2_res else self.gemm_q2 };
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
            &.{ args.x, args.qweight, args.scale_bias, args.y, args.residual orelse args.y },
            std.mem.asBytes(&params),
            .{ groups_x, groups_y, 1 },
            .{ 128, 1, 1 },
            insert_barrier_before,
        );
    }

    /// Encodes a wide-tile (`BM = 64, BN = 64`) quantized prefill GEMM
    /// (`q2`/`q3`/`q4` group-64 affine) for any `M >= 1` (`G15`/`G17`, ADR-0020,
    /// ADR-0021). Dispatches `*_nax` (`128` threads / 4 simdgroups) whenever
    /// `self.useNaxPrefill()` is active, otherwise `*_wide` (`256` threads / 8
    /// simdgroups). Set `args.wide_tile = false` to fall back to `32x32`.
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
        if (self.useNaxPrefill()) {
            return self.encodeNaxGemm(frame, bits, args, insert_barrier_before);
        }
        return self.encodeWideSimdgroupGemm(frame, bits, args, insert_barrier_before);
    }

    fn encodeWideSimdgroupGemm(
        self: DequantPipelines,
        frame: *metal.Compute.Frame,
        bits: quant.QuantBits,
        args: GemmArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.m == 0 or args.n == 0) return error.ZeroDimension;
        const has_res = (args.residual != null);
        const num_groups, const row_bytes, const pipe = switch (bits) {
            .q4 => blk: {
                const Ops = quant.QuantOps(.{ .bits = .q4, .group_size = 64 });
                break :blk .{ try Ops.groupsPerRow(args.k), try Ops.rowWeightBytes(args.k), if (has_res) self.gemm_wide_q4_res else self.gemm_wide_q4 };
            },
            .q3 => blk: {
                const Ops = quant.QuantOps(.{ .bits = .q3, .group_size = 64 });
                break :blk .{ try Ops.groupsPerRow(args.k), try Ops.rowWeightBytes(args.k), if (has_res) self.gemm_wide_q3_res else self.gemm_wide_q3 };
            },
            .q2 => blk: {
                const Ops = quant.QuantOps(.{ .bits = .q2, .group_size = 64 });
                break :blk .{ try Ops.groupsPerRow(args.k), try Ops.rowWeightBytes(args.k), if (has_res) self.gemm_wide_q2_res else self.gemm_wide_q2 };
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
            &.{ args.x, args.qweight, args.scale_bias, args.y, args.residual orelse args.y },
            std.mem.asBytes(&params),
            .{ groups_x, groups_y, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
    }

    fn encodeNaxGemm(
        self: DequantPipelines,
        frame: *metal.Compute.Frame,
        bits: quant.QuantBits,
        args: GemmArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.m == 0 or args.n == 0) return error.ZeroDimension;
        const has_res = (args.residual != null);
        const num_groups, const row_bytes, const pipe = switch (bits) {
            .q4 => blk: {
                const Ops = quant.QuantOps(.{ .bits = .q4, .group_size = 64 });
                const p_opt = if (has_res) self.gemm_nax_q4_res else self.gemm_nax_q4;
                break :blk .{ try Ops.groupsPerRow(args.k), try Ops.rowWeightBytes(args.k), p_opt orelse return error.Unsupported };
            },
            .q3 => blk: {
                const Ops = quant.QuantOps(.{ .bits = .q3, .group_size = 64 });
                const p_opt = if (has_res) self.gemm_nax_q3_res else self.gemm_nax_q3;
                break :blk .{ try Ops.groupsPerRow(args.k), try Ops.rowWeightBytes(args.k), p_opt orelse return error.Unsupported };
            },
            .q2 => blk: {
                const Ops = quant.QuantOps(.{ .bits = .q2, .group_size = 64 });
                const p_opt = if (has_res) self.gemm_nax_q2_res else self.gemm_nax_q2;
                break :blk .{ try Ops.groupsPerRow(args.k), try Ops.rowWeightBytes(args.k), p_opt orelse return error.Unsupported };
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
            &.{ args.x, args.qweight, args.scale_bias, args.y, args.residual orelse args.y },
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
        const has_res = (args.residual != null);
        const pipe = switch (dtype) {
            .f16 => if (has_res) self.gemv_f16_res else self.gemv_f16,
            .bf16 => if (has_res) self.gemv_bf16_res else self.gemv_bf16,
        };
        const params: DenseParams = .{ .m = 1, .n = args.n, .k = args.k };
        const groups_x: u64 = (@as(u64, args.n) + 15) / 16;
        try frame.encodeThreadgroups(
            pipe,
            &.{ args.x, args.weight, args.y, args.residual orelse args.y },
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
        const has_res = (args.residual != null);
        const num_groups, const row_bytes, const pipe = switch (bits) {
            .q4 => blk: {
                const Ops = quant.QuantOps(.{ .bits = .q4, .group_size = 64 });
                break :blk .{ try Ops.groupsPerRow(args.k), try Ops.rowWeightBytes(args.k), if (has_res) self.gemv_batch_q4_res else self.gemv_batch_q4 };
            },
            .q3 => blk: {
                const Ops = quant.QuantOps(.{ .bits = .q3, .group_size = 64 });
                break :blk .{ try Ops.groupsPerRow(args.k), try Ops.rowWeightBytes(args.k), if (has_res) self.gemv_batch_q3_res else self.gemv_batch_q3 };
            },
            .q2 => blk: {
                const Ops = quant.QuantOps(.{ .bits = .q2, .group_size = 64 });
                break :blk .{ try Ops.groupsPerRow(args.k), try Ops.rowWeightBytes(args.k), if (has_res) self.gemv_batch_q2_res else self.gemv_batch_q2 };
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
            &.{ args.x, args.qweight, args.scale_bias, args.y, args.residual orelse args.y },
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
        const has_res = (args.residual != null);
        const pipe = switch (dtype) {
            .f16 => if (has_res) self.gemv_batch_f16_res else self.gemv_batch_f16,
            .bf16 => if (has_res) self.gemv_batch_bf16_res else self.gemv_batch_bf16,
        };
        const params: DenseParams = .{ .m = args.m, .n = args.n, .k = args.k };
        const groups_x: u64 = (@as(u64, args.n) + 7) / 8;
        const groups_y: u64 = (@as(u64, args.m) + 7) / 8;
        try frame.encodeThreadgroups(
            pipe,
            &.{ args.x, args.weight, args.y, args.residual orelse args.y },
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
        const has_res = (args.residual != null);
        const pipe = switch (dtype) {
            .f16 => if (has_res) self.gemm_f16_res else self.gemm_f16,
            .bf16 => if (has_res) self.gemm_bf16_res else self.gemm_bf16,
        };
        const params: DenseParams = .{ .m = args.m, .n = args.n, .k = args.k };
        const groups_x: u64 = (@as(u64, args.n) + 31) / 32;
        const groups_y: u64 = (@as(u64, args.m) + 31) / 32;
        try frame.encodeThreadgroups(
            pipe,
            &.{ args.x, args.weight, args.y, args.residual orelse args.y },
            std.mem.asBytes(&params),
            .{ groups_x, groups_y, 1 },
            .{ 128, 1, 1 },
            insert_barrier_before,
        );
    }

    /// Encodes a wide-tile (`BM = 64, BN = 64`) dense FP16 / BF16 prefill GEMM
    /// for any `M >= 1` (`G15`/`G17`, ADR-0020, ADR-0021). Dispatches `*_nax`
    /// (`128` threads / 4 simdgroups) whenever `self.useNaxPrefill()` is active,
    /// otherwise `*_wide` (`256` threads / 8 simdgroups). Set `args.wide_tile = false`
    /// to fall back to `32x32`.
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
        if (self.useNaxPrefill()) {
            return self.encodeNaxDenseGemm(frame, dtype, args, insert_barrier_before);
        }
        return self.encodeWideSimdgroupDenseGemm(frame, dtype, args, insert_barrier_before);
    }

    fn encodeWideSimdgroupDenseGemm(
        self: DequantPipelines,
        frame: *metal.Compute.Frame,
        dtype: quant.FloatDType,
        args: DenseGemmArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.m == 0 or args.n == 0 or args.k == 0) return error.ZeroDimension;
        const has_res = (args.residual != null);
        const pipe = switch (dtype) {
            .f16 => if (has_res) self.gemm_wide_f16_res else self.gemm_wide_f16,
            .bf16 => if (has_res) self.gemm_wide_bf16_res else self.gemm_wide_bf16,
        };
        const params: DenseParams = .{ .m = args.m, .n = args.n, .k = args.k };
        const groups_x: u64 = (@as(u64, args.n) + 63) / 64;
        const groups_y: u64 = (@as(u64, args.m) + 63) / 64;
        try frame.encodeThreadgroups(
            pipe,
            &.{ args.x, args.weight, args.y, args.residual orelse args.y },
            std.mem.asBytes(&params),
            .{ groups_x, groups_y, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
    }

    fn encodeNaxDenseGemm(
        self: DequantPipelines,
        frame: *metal.Compute.Frame,
        dtype: quant.FloatDType,
        args: DenseGemmArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.m == 0 or args.n == 0 or args.k == 0) return error.ZeroDimension;
        const has_res = (args.residual != null);
        const pipe = switch (dtype) {
            .f16 => (if (has_res) self.gemm_nax_f16_res else self.gemm_nax_f16) orelse return error.Unsupported,
            .bf16 => (if (has_res) self.gemm_nax_bf16_res else self.gemm_nax_bf16) orelse return error.Unsupported,
        };
        const params: DenseParams = .{ .m = args.m, .n = args.n, .k = args.k };
        const groups_x: u64 = (@as(u64, args.n) + 63) / 64;
        const groups_y: u64 = (@as(u64, args.m) + 63) / 64;
        try frame.encodeThreadgroups(
            pipe,
            &.{ args.x, args.weight, args.y, args.residual orelse args.y },
            std.mem.asBytes(&params),
            .{ groups_x, groups_y, 1 },
            .{ 128, 1, 1 },
            insert_barrier_before,
        );
    }

    /// Unified zero-alloc linear dispatch across all `.amf` `DType`s (`0..37`,
    /// `f16`/`bf16`, MLX affine `q2..q8` across `g32/g64/g128`, FP8/MXFP8/NVFP4,
    /// and all 23 GGUF block formats), with optional `G24` activation-quantization
    /// emulation (`0` dispatches when `act_quant_policy == .none`) and `M >= 16`
    /// tiled `fp16` wide prefill feeding `G15`/`G17`/`G19` GEMM paths.
    pub fn encodeLinearUnified(
        self: DequantPipelines,
        frame: *metal.Compute.Frame,
        args: UnifiedLinearArgs,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (args.m == 0 or args.n == 0 or args.k == 0) return error.ZeroDimension;

        var x_eff = args.x;
        var barrier = insert_barrier_before;

        const aq_mode = args.act_quant_policy.resolveForTensor(args.amf_dtype, args.k);
        if (aq_mode != .none) {
            const need_bytes = std.math.mul(u64, @as(u64, args.m) * args.k, @sizeOf(f16)) catch return error.Overflow;
            if (args.qf_pipes) |qf| {
                if (args.act_scratch) |scratch| {
                    if (args.act_scratch_bytes >= need_bytes) {
                        qf.encodeActQuantEmulate(frame, aq_mode, .{
                            .x_in = args.x,
                            .x_out = scratch,
                            .m = args.m,
                            .k = args.k,
                        }, barrier) catch |err| return mapQfEncodeError(err);
                        x_eff = scratch;
                        barrier = true;
                    }
                }
            }
        }

        if (!usesQuantFormats(args.amf_dtype, args.group_size)) {
            if (args.mode == .gemv or args.m == 1) {
                switch (args.amf_dtype) {
                    .f16 => try self.encodeDenseGemv(frame, .f16, .{ .x = x_eff, .weight = args.weight, .y = args.y, .n = args.n, .k = args.k, .residual = args.residual }, barrier),
                    .bf16 => try self.encodeDenseGemv(frame, .bf16, .{ .x = x_eff, .weight = args.weight, .y = args.y, .n = args.n, .k = args.k, .residual = args.residual }, barrier),
                    .q4_affine => try self.encodeGemv(frame, .q4, .{ .x = x_eff, .qweight = args.weight, .scale_bias = args.params orelse return error.InvalidArgument, .y = args.y, .n = args.n, .k = args.k, .residual = args.residual }, barrier),
                    .q3_affine => try self.encodeGemv(frame, .q3, .{ .x = x_eff, .qweight = args.weight, .scale_bias = args.params orelse return error.InvalidArgument, .y = args.y, .n = args.n, .k = args.k, .residual = args.residual }, barrier),
                    .q2_affine => try self.encodeGemv(frame, .q2, .{ .x = x_eff, .qweight = args.weight, .scale_bias = args.params orelse return error.InvalidArgument, .y = args.y, .n = args.n, .k = args.k, .residual = args.residual }, barrier),
                    else => return error.InvalidArgument,
                }
                return;
            }
            if (args.mode == .batched or args.m < 16) {
                switch (args.amf_dtype) {
                    .f16 => try self.encodeDenseGemvBatch(frame, .f16, .{ .x = x_eff, .weight = args.weight, .y = args.y, .m = args.m, .n = args.n, .k = args.k, .residual = args.residual }, barrier),
                    .bf16 => try self.encodeDenseGemvBatch(frame, .bf16, .{ .x = x_eff, .weight = args.weight, .y = args.y, .m = args.m, .n = args.n, .k = args.k, .residual = args.residual }, barrier),
                    .q4_affine => try self.encodeGemvBatch(frame, .q4, .{ .x = x_eff, .qweight = args.weight, .scale_bias = args.params orelse return error.InvalidArgument, .y = args.y, .m = args.m, .n = args.n, .k = args.k, .residual = args.residual }, barrier),
                    .q3_affine => try self.encodeGemvBatch(frame, .q3, .{ .x = x_eff, .qweight = args.weight, .scale_bias = args.params orelse return error.InvalidArgument, .y = args.y, .m = args.m, .n = args.n, .k = args.k, .residual = args.residual }, barrier),
                    .q2_affine => try self.encodeGemvBatch(frame, .q2, .{ .x = x_eff, .qweight = args.weight, .scale_bias = args.params orelse return error.InvalidArgument, .y = args.y, .m = args.m, .n = args.n, .k = args.k, .residual = args.residual }, barrier),
                    else => return error.InvalidArgument,
                }
                return;
            }
            const wide_tile = (args.mode == .gemm_wide);
            switch (args.amf_dtype) {
                .f16 => try self.encodeDenseGemm(frame, .f16, .{ .x = x_eff, .weight = args.weight, .y = args.y, .m = args.m, .n = args.n, .k = args.k, .wide_tile = wide_tile, .residual = args.residual }, barrier),
                .bf16 => try self.encodeDenseGemm(frame, .bf16, .{ .x = x_eff, .weight = args.weight, .y = args.y, .m = args.m, .n = args.n, .k = args.k, .wide_tile = wide_tile, .residual = args.residual }, barrier),
                .q4_affine => try self.encodeGemm(frame, .q4, .{ .x = x_eff, .qweight = args.weight, .scale_bias = args.params orelse return error.InvalidArgument, .y = args.y, .m = args.m, .n = args.n, .k = args.k, .wide_tile = wide_tile, .residual = args.residual }, barrier),
                .q3_affine => try self.encodeGemm(frame, .q3, .{ .x = x_eff, .qweight = args.weight, .scale_bias = args.params orelse return error.InvalidArgument, .y = args.y, .m = args.m, .n = args.n, .k = args.k, .wide_tile = wide_tile, .residual = args.residual }, barrier),
                .q2_affine => try self.encodeGemm(frame, .q2, .{ .x = x_eff, .qweight = args.weight, .scale_bias = args.params orelse return error.InvalidArgument, .y = args.y, .m = args.m, .n = args.n, .k = args.k, .wide_tile = wide_tile, .residual = args.residual }, barrier),
                else => return error.InvalidArgument,
            }
            return;
        }

        const qf = args.qf_pipes orelse return error.InvalidArgument;
        const qf_dt = quant_formats.DType.fromAmf(args.amf_dtype) orelse return error.InvalidArgument;

        if (args.mode == .gemv or args.m == 1) {
            qf.encodeGemv(frame, qf_dt, args.group_size, .{
                .x = x_eff,
                .qweight = args.weight,
                .params = args.params,
                .y = args.y,
                .residual = args.residual,
                .n = args.n,
                .k = args.k,
                .tensor_scale_per_row = args.tensor_scale_per_row,
            }, barrier) catch |err| return mapQfEncodeError(err);
            return;
        }

        if ((args.mode == .gemm32 or args.mode == .gemm_wide) and args.m >= 16) {
            const tile_bytes = std.math.mul(u64, @as(u64, args.n) * args.k, @sizeOf(f16)) catch 0;
            if (args.tile_f16) |tile_b| {
                if (tile_bytes > 0 and tile_bytes <= args.tile_f16_bytes) {
                    qf.encodeDequantTileF16(frame, qf_dt, args.group_size, .{
                        .qweight = args.weight,
                        .params = args.params,
                        .dst_f16 = tile_b,
                        .n_total = args.n,
                        .k_total = args.k,
                        .tile_n_start = 0,
                        .tile_k_start = 0,
                        .tile_n = args.n,
                        .tile_k = args.k,
                        .tensor_scale_per_row = args.tensor_scale_per_row,
                    }, barrier) catch |err| return mapQfEncodeError(err);
                    try self.encodeDenseGemm(frame, .f16, .{
                        .x = x_eff,
                        .weight = tile_b,
                        .y = args.y,
                        .m = args.m,
                        .n = args.n,
                        .k = args.k,
                        .wide_tile = (args.mode == .gemm_wide),
                        .residual = args.residual,
                    }, true);
                    return;
                }
            }
        }

        qf.encodeGemvBatch(frame, qf_dt, args.group_size, .{
            .x = x_eff,
            .qweight = args.weight,
            .params = args.params,
            .y = args.y,
            .residual = args.residual,
            .m = args.m,
            .n = args.n,
            .k = args.k,
            .tensor_scale_per_row = args.tensor_scale_per_row,
        }, barrier) catch |err| return mapQfEncodeError(err);
    }

    fn quantShapeForDType(dtype: amf.DType, k: u32) FusedEncodeError!struct { num_groups: u32, row_bytes: u32, is_quant: bool } {
        return switch (dtype) {
            .f16, .bf16 => .{ .num_groups = 0, .row_bytes = 0, .is_quant = false },
            .q4_affine => blk: {
                const Ops = quant.QuantOps(.{ .bits = .q4, .group_size = 64 });
                break :blk .{ .num_groups = @intCast(try Ops.groupsPerRow(k)), .row_bytes = @intCast(try Ops.rowWeightBytes(k)), .is_quant = true };
            },
            .q3_affine => blk: {
                const Ops = quant.QuantOps(.{ .bits = .q3, .group_size = 64 });
                break :blk .{ .num_groups = @intCast(try Ops.groupsPerRow(k)), .row_bytes = @intCast(try Ops.rowWeightBytes(k)), .is_quant = true };
            },
            .q2_affine => blk: {
                const Ops = quant.QuantOps(.{ .bits = .q2, .group_size = 64 });
                break :blk .{ .num_groups = @intCast(try Ops.groupsPerRow(k)), .row_bytes = @intCast(try Ops.rowWeightBytes(k)), .is_quant = true };
            },
            else => error.UnsupportedDType,
        };
    }

    fn encodeSingleProjGemv(
        self: DequantPipelines,
        frame: *metal.Compute.Frame,
        x: metal.BufferBinding,
        proj: QkvProjBinding,
        k: u32,
        insert_barrier_before: bool,
    ) FusedEncodeError!void {
        switch (proj.dtype) {
            .f16 => try self.encodeDenseGemv(frame, .f16, .{ .x = x, .weight = proj.weight, .y = proj.y, .n = proj.n, .k = k }, insert_barrier_before),
            .bf16 => try self.encodeDenseGemv(frame, .bf16, .{ .x = x, .weight = proj.weight, .y = proj.y, .n = proj.n, .k = k }, insert_barrier_before),
            .q4_affine => try self.encodeGemv(frame, .q4, .{ .x = x, .qweight = proj.weight, .scale_bias = proj.scale_bias orelse return error.InvalidArgument, .y = proj.y, .n = proj.n, .k = k }, insert_barrier_before),
            .q3_affine => try self.encodeGemv(frame, .q3, .{ .x = x, .qweight = proj.weight, .scale_bias = proj.scale_bias orelse return error.InvalidArgument, .y = proj.y, .n = proj.n, .k = k }, insert_barrier_before),
            .q2_affine => try self.encodeGemv(frame, .q2, .{ .x = x, .qweight = proj.weight, .scale_bias = proj.scale_bias orelse return error.InvalidArgument, .y = proj.y, .n = proj.n, .k = k }, insert_barrier_before),
            else => return error.UnsupportedDType,
        }
    }

    fn encodeSingleProjGemm(
        self: DequantPipelines,
        frame: *metal.Compute.Frame,
        x: metal.BufferBinding,
        proj: QkvProjBinding,
        m: u32,
        k: u32,
        insert_barrier_before: bool,
    ) FusedEncodeError!void {
        switch (proj.dtype) {
            .f16 => try self.encodeDenseGemm(frame, .f16, .{ .x = x, .weight = proj.weight, .y = proj.y, .m = m, .n = proj.n, .k = k }, insert_barrier_before),
            .bf16 => try self.encodeDenseGemm(frame, .bf16, .{ .x = x, .weight = proj.weight, .y = proj.y, .m = m, .n = proj.n, .k = k }, insert_barrier_before),
            .q4_affine => try self.encodeGemm(frame, .q4, .{ .x = x, .qweight = proj.weight, .scale_bias = proj.scale_bias orelse return error.InvalidArgument, .y = proj.y, .m = m, .n = proj.n, .k = k }, insert_barrier_before),
            .q3_affine => try self.encodeGemm(frame, .q3, .{ .x = x, .qweight = proj.weight, .scale_bias = proj.scale_bias orelse return error.InvalidArgument, .y = proj.y, .m = m, .n = proj.n, .k = k }, insert_barrier_before),
            .q2_affine => try self.encodeGemm(frame, .q2, .{ .x = x, .qweight = proj.weight, .scale_bias = proj.scale_bias orelse return error.InvalidArgument, .y = proj.y, .m = m, .n = proj.n, .k = k }, insert_barrier_before),
            else => return error.UnsupportedDType,
        }
    }

    /// Encodes `ami_fused_qkv_gemv` (`M = 1`, `G18`, ADR-0023 §1) for layers
    /// whose `q`, `k`, `v` projections share one homogeneous `DType`.
    /// Mixed-DType GSQ layers are strictly guarded: they either route cleanly
    /// to the 3 unfused dispatches (`args.allow_unfused_fallback == true`,
    /// returning `.unfused_heterogeneous`) or fail with `error.HeterogeneousDType`.
    pub fn encodeFusedQkvGemv(
        self: DequantPipelines,
        frame: *metal.Compute.Frame,
        args: FusedQkvGemvArgs,
        insert_barrier_before: bool,
    ) FusedEncodeError!FusionPath {
        if (args.k == 0 or args.q.n == 0 or args.k_proj.n == 0 or args.v.n == 0) {
            return error.ZeroDimension;
        }
        if (!canFuseQkv(args.q.dtype, args.k_proj.dtype, args.v.dtype)) {
            if (!args.allow_unfused_fallback) return error.HeterogeneousDType;
            try self.encodeSingleProjGemv(frame, args.x, args.q, args.k, insert_barrier_before);
            try self.encodeSingleProjGemv(frame, args.x, args.k_proj, args.k, false);
            try self.encodeSingleProjGemv(frame, args.x, args.v, args.k, false);
            return .unfused_heterogeneous;
        }
        const qshape = try quantShapeForDType(args.q.dtype, args.k);
        if (qshape.is_quant and (args.q.scale_bias == null or args.k_proj.scale_bias == null or args.v.scale_bias == null)) {
            return error.InvalidArgument;
        }
        const pipe = self.fused_qkv_gemv.select(args.q.dtype) orelse return error.UnsupportedDType;
        const tgs_q: u32 = @intCast((@as(u64, args.q.n) + 15) / 16);
        const tgs_k: u32 = @intCast((@as(u64, args.k_proj.n) + 15) / 16);
        const tgs_v: u32 = @intCast((@as(u64, args.v.n) + 15) / 16);
        const params: FusedQkvParams = .{
            .m = 1,
            .n_q = args.q.n,
            .n_k = args.k_proj.n,
            .n_v = args.v.n,
            .k = args.k,
            .num_groups = qshape.num_groups,
            .row_bytes = qshape.row_bytes,
            .tgs_q = tgs_q,
            .tgs_k = tgs_k,
            .tgs_v = tgs_v,
        };
        const total_tgs: u64 = @as(u64, tgs_q) + @as(u64, tgs_k) + @as(u64, tgs_v);
        try frame.encodeThreadgroups(
            pipe,
            &.{
                args.x,
                args.q.weight,
                args.q.scale_bias orelse args.q.weight,
                args.k_proj.weight,
                args.k_proj.scale_bias orelse args.k_proj.weight,
                args.v.weight,
                args.v.scale_bias orelse args.v.weight,
                args.q.y,
                args.k_proj.y,
                args.v.y,
            },
            std.mem.asBytes(&params),
            .{ total_tgs, 1, 1 },
            .{ 128, 1, 1 },
            insert_barrier_before,
        );
        return .fused;
    }

    /// Encodes `ami_fused_qkv_gemm` (`M >= 1`, `G18`, ADR-0023 §1) for layers
    /// whose `q`, `k`, `v` projections share one homogeneous `DType`.
    /// Dispatches `ami_fused_qkv_gemv` when `args.m == 1`, and strictly guards
    /// mixed-DType GSQ layers onto the 3 unfused projection dispatches.
    pub fn encodeFusedQkvGemm(
        self: DequantPipelines,
        frame: *metal.Compute.Frame,
        args: FusedQkvGemmArgs,
        insert_barrier_before: bool,
    ) FusedEncodeError!FusionPath {
        if (args.m == 0 or args.k == 0 or args.q.n == 0 or args.k_proj.n == 0 or args.v.n == 0) {
            return error.ZeroDimension;
        }
        if (args.m == 1) {
            return self.encodeFusedQkvGemv(frame, .{
                .x = args.x,
                .q = args.q,
                .k_proj = args.k_proj,
                .v = args.v,
                .k = args.k,
                .allow_unfused_fallback = args.allow_unfused_fallback,
            }, insert_barrier_before);
        }
        if (!canFuseQkv(args.q.dtype, args.k_proj.dtype, args.v.dtype)) {
            if (!args.allow_unfused_fallback) return error.HeterogeneousDType;
            try self.encodeSingleProjGemm(frame, args.x, args.q, args.m, args.k, insert_barrier_before);
            try self.encodeSingleProjGemm(frame, args.x, args.k_proj, args.m, args.k, false);
            try self.encodeSingleProjGemm(frame, args.x, args.v, args.m, args.k, false);
            return .unfused_heterogeneous;
        }
        const qshape = try quantShapeForDType(args.q.dtype, args.k);
        if (qshape.is_quant and (args.q.scale_bias == null or args.k_proj.scale_bias == null or args.v.scale_bias == null)) {
            return error.InvalidArgument;
        }
        const pipe = self.fused_qkv_gemm.select(args.q.dtype) orelse return error.UnsupportedDType;
        const tgs_q: u32 = @intCast((@as(u64, args.q.n) + 63) / 64);
        const tgs_k: u32 = @intCast((@as(u64, args.k_proj.n) + 63) / 64);
        const tgs_v: u32 = @intCast((@as(u64, args.v.n) + 63) / 64);
        const params: FusedQkvParams = .{
            .m = args.m,
            .n_q = args.q.n,
            .n_k = args.k_proj.n,
            .n_v = args.v.n,
            .k = args.k,
            .num_groups = qshape.num_groups,
            .row_bytes = qshape.row_bytes,
            .tgs_q = tgs_q,
            .tgs_k = tgs_k,
            .tgs_v = tgs_v,
        };
        const total_tgs_x: u64 = @as(u64, tgs_q) + @as(u64, tgs_k) + @as(u64, tgs_v);
        const groups_y: u64 = (@as(u64, args.m) + 63) / 64;
        try frame.encodeThreadgroups(
            pipe,
            &.{
                args.x,
                args.q.weight,
                args.q.scale_bias orelse args.q.weight,
                args.k_proj.weight,
                args.k_proj.scale_bias orelse args.k_proj.weight,
                args.v.weight,
                args.v.scale_bias orelse args.v.weight,
                args.q.y,
                args.k_proj.y,
                args.v.y,
            },
            std.mem.asBytes(&params),
            .{ total_tgs_x, groups_y, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
        return .fused;
    }

    /// Encodes fused dense FFN `Gate + Up + SwiGLU` (`G18`, ADR-0023 §1):
    /// computes `y = SiLU(x @ W_gate^T) * (x @ W_up^T)` in registers/SRAM
    /// without writing `2 x m x n` intermediate activations to unified DRAM.
    /// Strictly guarded: returns `error.HeterogeneousDType` when
    /// `gate_dtype != up_dtype` so mixed-DType GSQ layers use the unfused path.
    pub fn encodeDenseGateUpSwiglu(
        self: DequantPipelines,
        frame: *metal.Compute.Frame,
        args: DenseGateUpSwigluArgs,
        insert_barrier_before: bool,
    ) FusedEncodeError!void {
        if (args.m == 0 or args.n == 0 or args.k == 0) return error.ZeroDimension;
        if (!canFuseGateUp(args.gate_dtype, args.up_dtype)) {
            if (!isFusableLinearDType(args.gate_dtype) or !isFusableLinearDType(args.up_dtype)) {
                return error.UnsupportedDType;
            }
            return error.HeterogeneousDType;
        }
        const qshape = try quantShapeForDType(args.gate_dtype, args.k);
        if (qshape.is_quant and (args.sb_gate == null or args.sb_up == null)) {
            return error.InvalidArgument;
        }
        const params: DequantParams = .{
            .m = args.m,
            .n = args.n,
            .k = args.k,
            .num_groups = qshape.num_groups,
            .row_bytes = qshape.row_bytes,
        };
        const bindings = [_]metal.BufferBinding{
            args.x,
            args.w_gate,
            args.sb_gate orelse args.w_gate,
            args.w_up,
            args.sb_up orelse args.w_up,
            args.y,
        };
        if (args.m == 1) {
            const pipe = self.dense_swiglu_gemv.select(args.gate_dtype) orelse return error.UnsupportedDType;
            const groups_x: u64 = (@as(u64, args.n) + 7) / 8;
            try frame.encodeThreadgroups(
                pipe,
                &bindings,
                std.mem.asBytes(&params),
                .{ groups_x, 1, 1 },
                .{ 128, 1, 1 },
                insert_barrier_before,
            );
        } else {
            const pipe = self.dense_swiglu_gemm.select(args.gate_dtype) orelse return error.UnsupportedDType;
            const groups_x: u64 = (@as(u64, args.n) + 31) / 32;
            const groups_y: u64 = (@as(u64, args.m) + 31) / 32;
            try frame.encodeThreadgroups(
                pipe,
                &bindings,
                std.mem.asBytes(&params),
                .{ groups_x, groups_y, 1 },
                .{ 128, 1, 1 },
                insert_barrier_before,
            );
        }
    }

    /// Runs the startup `.metallib` GPU numerical canary (`G16`, ADR-0020,
    /// ADR-0021) over a temporary `16 KiB` wired page in one Metal 4 command
    /// buffer, verifying `f16`, `bf16` (including exponent paths exceeding `f16`
    /// dynamic range), and `q4_affine` across `32x32`, `64x64` `simdgroup_matrix`,
    /// and (when compiled on `Apple10+`) `64x64` `NAX` TensorOps kernels.
    /// When `NAX` fails finiteness or tolerance checks in non-strict mode,
    /// `self` automatically demotes to `.simdgroup_apple9` (`nax_canary_failed`)
    /// and verifies the `simdgroup_apple9` fallback kernels pass.
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

    /// Mutable convenience wrapper over `runStartupNumericalCanary` that also
    /// updates the plain `active_backend` and `nax_fallback_reason` struct fields.
    pub fn runStartupNumericalCanaryMut(
        self: *DequantPipelines,
        dev: *metal.Device,
    ) CanaryError!CanaryResult {
        const res = try self.runStartupNumericalCanary(dev);
        self.syncNaxState();
        return res;
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

    /// Mutable convenience wrapper over `runStartupNumericalCanaryProfile` that
    /// also synchronizes `active_backend` and `nax_fallback_reason`.
    pub fn runStartupNumericalCanaryProfileMut(
        self: *DequantPipelines,
        dev: *metal.Device,
    ) CanaryError!CanaryProfile {
        const prof = try self.runStartupNumericalCanaryProfile(dev);
        self.syncNaxState();
        return prof;
    }

    /// Mutable convenience wrapper over `runStartupNumericalCanaryInBuffer`
    /// that also synchronizes `active_backend` and `nax_fallback_reason`.
    pub fn runStartupNumericalCanaryInBufferMut(
        self: *DequantPipelines,
        compute: *metal.Compute,
        gpu_buf: metal.Buffer,
        wired_bytes: []align(page.APPLE_PAGE_SIZE) u8,
    ) CanaryError!CanaryResult {
        const res = try self.runStartupNumericalCanaryInBuffer(compute, gpu_buf, wired_bytes);
        self.syncNaxState();
        return res;
    }

    pub const COMBINED_CANARY_TAIL_OFFSET: usize = 13568;

    const canary_cm: usize = 16;
    const canary_cn: usize = 16;
    const canary_ck: usize = 64;
    const canary_x_off: usize = 0; // 16 * 64 * 2 = 2048 B
    const canary_w_f16_off: usize = 2048; // 2048 B
    const canary_w_bf16_off: usize = 4096; // 2048 B
    const canary_w_q4_off: usize = 6144; // 16 * 32 = 512 B
    const canary_sb_q4_off: usize = 6656; // 16 * 4 = 64 B
    const canary_y_f16_off: usize = 6720; // 16 * 16 * 2 = 512 B
    const canary_y_bf16_off: usize = 7232; // 512 B
    const canary_y_q4_off: usize = 7744; // 512 B
    const canary_y_f16_wide_off: usize = 8256; // 512 B
    const canary_y_bf16_wide_off: usize = 8768; // 512 B
    const canary_y_q4_wide_off: usize = 9280; // 512 B
    const canary_y_f16_nax_off: usize = 9792; // 512 B
    const canary_y_bf16_nax_off: usize = 10304; // 512 B
    const canary_y_q4_nax_off: usize = 10816; // 512 B
    // M7.8 G18 fused kernel startup canary slices (`11328..13504 <= COMBINED_CANARY_TAIL_OFFSET <= 16384 B`).
    const canary_res_seed_off: usize = 11328; // 16 * 16 * 2 = 512 B
    const canary_y_res_q4_wide_off: usize = 11840; // 512 B
    const canary_y_res_f16_gemv_off: usize = 12352; // 32 B padded to 64 B
    const canary_y_res_q4_gemv_off: usize = 12416; // 32 B padded to 64 B
    const canary_y_fqkv_gemv_q_off: usize = 12480; // 1 * 8 * 2 -> 64 B
    const canary_y_fqkv_gemv_k_off: usize = 12544; // 1 * 4 * 2 -> 64 B
    const canary_y_fqkv_gemv_v_off: usize = 12608; // 1 * 4 * 2 -> 64 B
    const canary_y_fqkv_gemm_q_off: usize = 12672; // 16 * 8 * 2 = 256 B
    const canary_y_fqkv_gemm_k_off: usize = 12928; // 16 * 4 * 2 = 128 B
    const canary_y_fqkv_gemm_v_off: usize = 13056; // 16 * 4 * 2 = 128 B
    const canary_y_swiglu_gemv_off: usize = 13184; // 1 * 8 * 2 -> 64 B
    const canary_y_swiglu_gemm_off: usize = 13248; // 16 * 8 * 2 = 256 B
    const canary_g18_end: usize = canary_y_swiglu_gemm_off + 256;

    comptime {
        std.debug.assert(canary_y_q4_nax_off + 512 <= page.APPLE_PAGE_SIZE);
        std.debug.assert(canary_g18_end <= COMBINED_CANARY_TAIL_OFFSET);
        std.debug.assert(COMBINED_CANARY_TAIL_OFFSET <= page.APPLE_PAGE_SIZE);
    }

    pub const StartupCanaryState = struct {
        run_real_nax: bool = false,
        run_simulated_nax_fail: bool = false,
        ref_f16: [canary_cm * canary_cn]f32 = undefined,
        ref_bf16: [canary_cm * canary_cn]f32 = undefined,
        ref_q4: [canary_cm * canary_cn]f32 = undefined,
        ref_res_q4: [canary_cm * canary_cn]f32 = undefined,
        ref_res_f16_gemv: [canary_cn]f32 = undefined,
        ref_fqkv_q: [canary_cm * 8]f32 = undefined,
        ref_fqkv_k: [canary_cm * 4]f32 = undefined,
        ref_fqkv_v: [canary_cm * 4]f32 = undefined,
        ref_swiglu: [canary_cm * 8]f32 = undefined,
    };

    /// Populates deterministic `f16`/`bf16`/`q4`/`G18` canary inputs and CPU
    /// fp32 reference buffers inside `wired_bytes[0..canary_g18_end]`.
    pub fn prepareStartupNumericalCanary(
        self: *const DequantPipelines,
        wired_bytes: []align(page.APPLE_PAGE_SIZE) u8,
        state: *StartupCanaryState,
    ) CanaryError!void {
        if (wired_bytes.len < page.APPLE_PAGE_SIZE) return error.InvalidArgument;

        const cm = canary_cm;
        const cn = canary_cn;
        const ck = canary_ck;
        const OpsQ4 = quant.QuantOps(.{ .bits = .q4, .group_size = 64 });

        const already_demoted = if (self.nax_state) |st| st.isCanaryDemoted() else false;
        state.run_real_nax = self.nax_pipelines_compiled and self.activeBackend().isNax();
        state.run_simulated_nax_fail = self.simulate_nax_canary_failure and !already_demoted;

        @memset(wired_bytes[0..canary_g18_end], 0);

        const x_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, wired_bytes[canary_x_off .. canary_x_off + cm * ck * 2]));
        const w_f16: []u16 = @alignCast(std.mem.bytesAsSlice(u16, wired_bytes[canary_w_f16_off .. canary_w_f16_off + cn * ck * 2]));
        const w_bf16: []u16 = @alignCast(std.mem.bytesAsSlice(u16, wired_bytes[canary_w_bf16_off .. canary_w_bf16_off + cn * ck * 2]));
        const w_q4: []u8 = wired_bytes[canary_w_q4_off .. canary_w_q4_off + cn * 32];
        const sb_q4: []quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, wired_bytes[canary_sb_q4_off .. canary_sb_q4_off + cn * 4]));
        const res_seed: []f16 = @alignCast(std.mem.bytesAsSlice(f16, wired_bytes[canary_res_seed_off .. canary_res_seed_off + cm * cn * 2]));

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

        try quant.denseGemmCpuRef(.f16, x_f16, w_f16, &state.ref_f16, cm, cn, ck);
        try quant.denseGemmCpuRef(.bf16, x_f16, w_bf16, &state.ref_bf16, cm, cn, ck);
        try OpsQ4.matmulRef(x_f16, w_q4, sb_q4, &state.ref_q4, cm, cn, ck);

        for (0..cm) |mi| {
            for (0..cn) |ni| {
                const idx = mi * cn + ni;
                const rs: f16 = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast((idx * 7 + 3) % 13)) - 6)) * 0.0625);
                res_seed[idx] = rs;
                const rsf: f32 = @floatCast(rs);
                state.ref_res_q4[idx] = state.ref_q4[idx] + rsf;
                if (mi == 0) state.ref_res_f16_gemv[ni] = state.ref_f16[ni] + rsf;
            }
            for (0..8) |ni| {
                const g = state.ref_q4[mi * cn + ni];
                const u = state.ref_q4[mi * cn + 8 + ni];
                const silu_g = g / (1.0 + @exp(-g));
                state.ref_fqkv_q[mi * 8 + ni] = g;
                state.ref_swiglu[mi * 8 + ni] = silu_g * u;
            }
            for (0..4) |ni| {
                state.ref_fqkv_k[mi * 4 + ni] = state.ref_q4[mi * cn + 8 + ni];
                state.ref_fqkv_v[mi * 4 + ni] = state.ref_q4[mi * cn + 12 + ni];
            }
        }
    }

    /// Encodes all G16 + M7.8 G18 dequant/fused kernel canary dispatches into
    /// `frame` with zero intra-frame barriers.
    pub fn encodeStartupNumericalCanary(
        self: *const DequantPipelines,
        frame: *metal.Compute.Frame,
        gpu_buf: metal.Buffer,
        state: *const StartupCanaryState,
    ) CanaryError!void {
        const cm = canary_cm;
        const cn = canary_cn;
        const ck = canary_ck;

        const x_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = canary_x_off };
        const w_f16_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = canary_w_f16_off };
        const w_bf16_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = canary_w_bf16_off };
        const w_q4_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = canary_w_q4_off };
        const sb_q4_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = canary_sb_q4_off };
        const res_seed_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = canary_res_seed_off };
        const w_q4_k_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = canary_w_q4_off + 8 * 32 };
        const sb_q4_k_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = canary_sb_q4_off + 8 * 4 };
        const w_q4_v_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = canary_w_q4_off + 12 * 32 };
        const sb_q4_v_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = canary_sb_q4_off + 12 * 4 };

        if (state.run_real_nax) {
            try self.encodeNaxDenseGemm(frame, .f16, .{
                .x = x_b,
                .weight = w_f16_b,
                .y = .{ .buffer = gpu_buf, .offset = canary_y_f16_nax_off },
                .m = cm,
                .n = cn,
                .k = ck,
            }, false);
            try self.encodeNaxDenseGemm(frame, .bf16, .{
                .x = x_b,
                .weight = w_bf16_b,
                .y = .{ .buffer = gpu_buf, .offset = canary_y_bf16_nax_off },
                .m = cm,
                .n = cn,
                .k = ck,
            }, false);
            try self.encodeNaxGemm(frame, .q4, .{
                .x = x_b,
                .qweight = w_q4_b,
                .scale_bias = sb_q4_b,
                .y = .{ .buffer = gpu_buf, .offset = canary_y_q4_nax_off },
                .m = cm,
                .n = cn,
                .k = ck,
            }, false);
        }

        try self.encodeDenseGemm32(frame, .f16, .{
            .x = x_b,
            .weight = w_f16_b,
            .y = .{ .buffer = gpu_buf, .offset = canary_y_f16_off },
            .m = cm,
            .n = cn,
            .k = ck,
        }, false);
        try self.encodeDenseGemm32(frame, .bf16, .{
            .x = x_b,
            .weight = w_bf16_b,
            .y = .{ .buffer = gpu_buf, .offset = canary_y_bf16_off },
            .m = cm,
            .n = cn,
            .k = ck,
        }, false);
        try self.encodeGemm32(frame, .q4, .{
            .x = x_b,
            .qweight = w_q4_b,
            .scale_bias = sb_q4_b,
            .y = .{ .buffer = gpu_buf, .offset = canary_y_q4_off },
            .m = cm,
            .n = cn,
            .k = ck,
        }, false);
        try self.encodeWideSimdgroupDenseGemm(frame, .f16, .{
            .x = x_b,
            .weight = w_f16_b,
            .y = .{ .buffer = gpu_buf, .offset = canary_y_f16_wide_off },
            .m = cm,
            .n = cn,
            .k = ck,
        }, false);
        try self.encodeWideSimdgroupDenseGemm(frame, .bf16, .{
            .x = x_b,
            .weight = w_bf16_b,
            .y = .{ .buffer = gpu_buf, .offset = canary_y_bf16_wide_off },
            .m = cm,
            .n = cn,
            .k = ck,
        }, false);
        try self.encodeWideSimdgroupGemm(frame, .q4, .{
            .x = x_b,
            .qweight = w_q4_b,
            .scale_bias = sb_q4_b,
            .y = .{ .buffer = gpu_buf, .offset = canary_y_q4_wide_off },
            .m = cm,
            .n = cn,
            .k = ck,
        }, false);

        try self.encodeWideSimdgroupGemm(frame, .q4, .{
            .x = x_b,
            .qweight = w_q4_b,
            .scale_bias = sb_q4_b,
            .y = .{ .buffer = gpu_buf, .offset = canary_y_res_q4_wide_off },
            .residual = res_seed_b,
            .m = cm,
            .n = cn,
            .k = ck,
        }, false);
        try self.encodeDenseGemv(frame, .f16, .{
            .x = x_b,
            .weight = w_f16_b,
            .y = .{ .buffer = gpu_buf, .offset = canary_y_res_f16_gemv_off },
            .residual = res_seed_b,
            .n = cn,
            .k = ck,
        }, false);
        try self.encodeGemv(frame, .q4, .{
            .x = x_b,
            .qweight = w_q4_b,
            .scale_bias = sb_q4_b,
            .y = .{ .buffer = gpu_buf, .offset = canary_y_res_q4_gemv_off },
            .residual = res_seed_b,
            .n = cn,
            .k = ck,
        }, false);
        _ = self.encodeFusedQkvGemv(frame, .{
            .x = x_b,
            .q = .{ .dtype = .q4_affine, .weight = w_q4_b, .scale_bias = sb_q4_b, .y = .{ .buffer = gpu_buf, .offset = canary_y_fqkv_gemv_q_off }, .n = 8 },
            .k_proj = .{ .dtype = .q4_affine, .weight = w_q4_k_b, .scale_bias = sb_q4_k_b, .y = .{ .buffer = gpu_buf, .offset = canary_y_fqkv_gemv_k_off }, .n = 4 },
            .v = .{ .dtype = .q4_affine, .weight = w_q4_v_b, .scale_bias = sb_q4_v_b, .y = .{ .buffer = gpu_buf, .offset = canary_y_fqkv_gemv_v_off }, .n = 4 },
            .k = ck,
            .allow_unfused_fallback = false,
        }, false) catch |err| switch (err) {
            error.HeterogeneousDType, error.UnsupportedDType => return error.MetallibNumericalCanaryFailed,
            else => |e| return e,
        };
        _ = self.encodeFusedQkvGemm(frame, .{
            .x = x_b,
            .q = .{ .dtype = .q4_affine, .weight = w_q4_b, .scale_bias = sb_q4_b, .y = .{ .buffer = gpu_buf, .offset = canary_y_fqkv_gemm_q_off }, .n = 8 },
            .k_proj = .{ .dtype = .q4_affine, .weight = w_q4_k_b, .scale_bias = sb_q4_k_b, .y = .{ .buffer = gpu_buf, .offset = canary_y_fqkv_gemm_k_off }, .n = 4 },
            .v = .{ .dtype = .q4_affine, .weight = w_q4_v_b, .scale_bias = sb_q4_v_b, .y = .{ .buffer = gpu_buf, .offset = canary_y_fqkv_gemm_v_off }, .n = 4 },
            .m = cm,
            .k = ck,
            .allow_unfused_fallback = false,
        }, false) catch |err| switch (err) {
            error.HeterogeneousDType, error.UnsupportedDType => return error.MetallibNumericalCanaryFailed,
            else => |e| return e,
        };
        self.encodeDenseGateUpSwiglu(frame, .{
            .x = x_b,
            .gate_dtype = .q4_affine,
            .w_gate = w_q4_b,
            .sb_gate = sb_q4_b,
            .up_dtype = .q4_affine,
            .w_up = w_q4_k_b,
            .sb_up = sb_q4_k_b,
            .y = .{ .buffer = gpu_buf, .offset = canary_y_swiglu_gemv_off },
            .m = 1,
            .n = 8,
            .k = ck,
        }, false) catch |err| switch (err) {
            error.HeterogeneousDType, error.UnsupportedDType => return error.MetallibNumericalCanaryFailed,
            else => |e| return e,
        };
        self.encodeDenseGateUpSwiglu(frame, .{
            .x = x_b,
            .gate_dtype = .q4_affine,
            .w_gate = w_q4_b,
            .sb_gate = sb_q4_b,
            .up_dtype = .q4_affine,
            .w_up = w_q4_k_b,
            .sb_up = sb_q4_k_b,
            .y = .{ .buffer = gpu_buf, .offset = canary_y_swiglu_gemm_off },
            .m = cm,
            .n = 8,
            .k = ck,
        }, false) catch |err| switch (err) {
            error.HeterogeneousDType, error.UnsupportedDType => return error.MetallibNumericalCanaryFailed,
            else => |e| return e,
        };
    }

    /// Validates completed GPU canary outputs in `wired_bytes` against `state`.
    pub fn verifyStartupNumericalCanary(
        self: *const DequantPipelines,
        wired_bytes: []align(page.APPLE_PAGE_SIZE) const u8,
        state: *const StartupCanaryState,
        elapsed_us: f64,
    ) CanaryError!CanaryResult {
        const cm = canary_cm;
        const cn = canary_cn;

        const y_f16: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, wired_bytes[canary_y_f16_off .. canary_y_f16_off + cm * cn * 2]));
        const y_bf16: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, wired_bytes[canary_y_bf16_off .. canary_y_bf16_off + cm * cn * 2]));
        const y_q4: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, wired_bytes[canary_y_q4_off .. canary_y_q4_off + cm * cn * 2]));
        const y_f16_wide: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, wired_bytes[canary_y_f16_wide_off .. canary_y_f16_wide_off + cm * cn * 2]));
        const y_bf16_wide: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, wired_bytes[canary_y_bf16_wide_off .. canary_y_bf16_wide_off + cm * cn * 2]));
        const y_q4_wide: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, wired_bytes[canary_y_q4_wide_off .. canary_y_q4_wide_off + cm * cn * 2]));
        const y_f16_nax: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, wired_bytes[canary_y_f16_nax_off .. canary_y_f16_nax_off + cm * cn * 2]));
        const y_bf16_nax: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, wired_bytes[canary_y_bf16_nax_off .. canary_y_bf16_nax_off + cm * cn * 2]));
        const y_q4_nax: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, wired_bytes[canary_y_q4_nax_off .. canary_y_q4_nax_off + cm * cn * 2]));

        const y_res_q4_wide: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, wired_bytes[canary_y_res_q4_wide_off .. canary_y_res_q4_wide_off + cm * cn * 2]));
        const y_res_f16_gemv: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, wired_bytes[canary_y_res_f16_gemv_off .. canary_y_res_f16_gemv_off + cn * 2]));
        const y_res_q4_gemv: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, wired_bytes[canary_y_res_q4_gemv_off .. canary_y_res_q4_gemv_off + cn * 2]));
        const y_fqkv_gemv_q: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, wired_bytes[canary_y_fqkv_gemv_q_off .. canary_y_fqkv_gemv_q_off + 8 * 2]));
        const y_fqkv_gemv_k: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, wired_bytes[canary_y_fqkv_gemv_k_off .. canary_y_fqkv_gemv_k_off + 4 * 2]));
        const y_fqkv_gemv_v: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, wired_bytes[canary_y_fqkv_gemv_v_off .. canary_y_fqkv_gemv_v_off + 4 * 2]));
        const y_fqkv_gemm_q: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, wired_bytes[canary_y_fqkv_gemm_q_off .. canary_y_fqkv_gemm_q_off + cm * 8 * 2]));
        const y_fqkv_gemm_k: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, wired_bytes[canary_y_fqkv_gemm_k_off .. canary_y_fqkv_gemm_k_off + cm * 4 * 2]));
        const y_fqkv_gemm_v: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, wired_bytes[canary_y_fqkv_gemm_v_off .. canary_y_fqkv_gemm_v_off + cm * 4 * 2]));
        const y_swiglu_gemv: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, wired_bytes[canary_y_swiglu_gemv_off .. canary_y_swiglu_gemv_off + 8 * 2]));
        const y_swiglu_gemm: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, wired_bytes[canary_y_swiglu_gemm_off .. canary_y_swiglu_gemm_off + cm * 8 * 2]));

        var nax_tested = false;
        var nax_demoted = false;
        var max_nax_f16: f32 = 0.0;
        var max_nax_bf16: f32 = 0.0;
        var max_nax_q4: f32 = 0.0;

        if (state.run_real_nax or state.run_simulated_nax_fail) {
            nax_tested = true;
            const nax_ok: error{MetallibNumericalCanaryFailed}!CanarySliceErrors = if (state.run_simulated_nax_fail)
                error.MetallibNumericalCanaryFailed
            else
                verifyCanarySlices(y_f16_nax, &state.ref_f16, y_bf16_nax, &state.ref_bf16, y_q4_nax, &state.ref_q4);

            if (nax_ok) |err_nax| {
                max_nax_f16 = err_nax.max_abs_err_f16;
                max_nax_bf16 = err_nax.max_abs_err_bf16;
                max_nax_q4 = err_nax.max_abs_err_q4;
            } else |err| {
                if (self.strict_nax) return err;
                if (self.nax_state) |st| {
                    st.demoteCanaryFailed();
                }
                nax_demoted = true;
            }
        }

        const err32 = try verifyCanarySlices(y_f16, &state.ref_f16, y_bf16, &state.ref_bf16, y_q4, &state.ref_q4);
        const err64 = try verifyCanarySlices(y_f16_wide, &state.ref_f16, y_bf16_wide, &state.ref_bf16, y_q4_wide, &state.ref_q4);

        var g18_err: f32 = 0.0;
        g18_err = @max(g18_err, try verifyCanaryPair(y_res_q4_wide, &state.ref_res_q4, CANARY_TOL_Q4));
        g18_err = @max(g18_err, try verifyCanaryPair(y_res_f16_gemv, &state.ref_res_f16_gemv, CANARY_TOL_F16));
        g18_err = @max(g18_err, try verifyCanaryPair(y_res_q4_gemv, state.ref_res_q4[0..cn], CANARY_TOL_Q4));
        g18_err = @max(g18_err, try verifyCanaryPair(y_fqkv_gemv_q, state.ref_fqkv_q[0..8], CANARY_TOL_Q4));
        g18_err = @max(g18_err, try verifyCanaryPair(y_fqkv_gemv_k, state.ref_fqkv_k[0..4], CANARY_TOL_Q4));
        g18_err = @max(g18_err, try verifyCanaryPair(y_fqkv_gemv_v, state.ref_fqkv_v[0..4], CANARY_TOL_Q4));
        g18_err = @max(g18_err, try verifyCanaryPair(y_fqkv_gemm_q, &state.ref_fqkv_q, CANARY_TOL_Q4));
        g18_err = @max(g18_err, try verifyCanaryPair(y_fqkv_gemm_k, &state.ref_fqkv_k, CANARY_TOL_Q4));
        g18_err = @max(g18_err, try verifyCanaryPair(y_fqkv_gemm_v, &state.ref_fqkv_v, CANARY_TOL_Q4));
        g18_err = @max(g18_err, try verifyCanaryPair(y_swiglu_gemv, state.ref_swiglu[0..8], CANARY_TOL_Q4));
        g18_err = @max(g18_err, try verifyCanaryPair(y_swiglu_gemm, &state.ref_swiglu, CANARY_TOL_Q4));

        return .{
            .elapsed_us = elapsed_us,
            .max_abs_err_f16 = @max(@max(err32.max_abs_err_f16, err64.max_abs_err_f16), max_nax_f16),
            .max_abs_err_bf16 = @max(@max(err32.max_abs_err_bf16, err64.max_abs_err_bf16), max_nax_bf16),
            .max_abs_err_q4 = @max(@max(err32.max_abs_err_q4, err64.max_abs_err_q4), max_nax_q4),
            .g18_max_abs_err = g18_err,
            .nax_tested = nax_tested,
            .nax_demoted = nax_demoted,
            .active_backend = self.activeBackend(),
            .nax_fallback_reason = self.naxFallbackReason(),
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
        var state: StartupCanaryState = .{};
        try self.prepareStartupNumericalCanary(wired_bytes, &state);

        const t0 = try vmstat.nowNs();
        var frame = try compute.beginFrame(5000);
        errdefer frame.abort();
        try self.encodeStartupNumericalCanary(&frame, gpu_buf, &state);
        const sig = try frame.commit();
        try compute.wait(sig, 5000);
        const t1 = try vmstat.nowNs();

        return self.verifyStartupNumericalCanary(wired_bytes, &state, @as(f64, @floatFromInt(t1 - t0)) / 1000.0);
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

test "G17: Apple9/Apple10/Apple11 backend policy and PrefillBackendMode parsing" {
    const device = try metal.Device.open();
    defer device.close();
    const info = try device.info();

    try std.testing.expectEqual(@as(?PrefillBackendMode, .auto), PrefillBackendMode.parse("auto"));
    try std.testing.expectEqual(@as(?PrefillBackendMode, .simdgroup), PrefillBackendMode.parse("simdgroup"));
    try std.testing.expectEqual(@as(?PrefillBackendMode, .nax), PrefillBackendMode.parse("nax"));
    try std.testing.expectEqual(@as(?PrefillBackendMode, null), PrefillBackendMode.parse("invalid"));

    if (!info.supportsNaxTensorOps()) {
        try std.testing.expectEqual(PrefillMatmulBackend.simdgroup_apple9, selectPrefillMatmulBackend(info));
        try std.testing.expect(!selectPrefillMatmulBackend(info).isNax());
    }

    var fake_apple9 = info;
    fake_apple9.raw.gpu_apple_family = 9;
    fake_apple9.raw.apple_gpu_family = 9;
    fake_apple9.raw.supports_apple9 = 1;
    fake_apple9.raw.supports_apple10 = 0;
    fake_apple9.raw.supports_apple11 = 0;
    try std.testing.expect(!fake_apple9.supportsNaxTensorOps());
    try std.testing.expectEqual(PrefillMatmulBackend.simdgroup_apple9, selectPrefillMatmulBackend(fake_apple9));
    const d9_auto = selectPrefillMatmulBackendWithMode(fake_apple9, .auto);
    try std.testing.expectEqual(PrefillMatmulBackend.simdgroup_apple9, d9_auto.backend);
    try std.testing.expectEqual(NaxFallbackReason.unsupported_gpu_family, d9_auto.fallback_reason);
    const d9_sg = selectPrefillMatmulBackendWithMode(fake_apple9, .simdgroup);
    try std.testing.expectEqual(PrefillMatmulBackend.simdgroup_apple9, d9_sg.backend);
    try std.testing.expectEqual(NaxFallbackReason.forced_simdgroup, d9_sg.fallback_reason);

    var fake_apple10 = fake_apple9;
    fake_apple10.raw.gpu_apple_family = 10;
    fake_apple10.raw.apple_gpu_family = 10;
    fake_apple10.raw.supports_apple10 = 1;
    try std.testing.expect(fake_apple10.supportsApple10());
    try std.testing.expect(fake_apple10.supportsNaxTensorOps());
    try std.testing.expectEqual(PrefillMatmulBackend.nax_apple10, selectPrefillMatmulBackend(fake_apple10));
    try std.testing.expect(selectPrefillMatmulBackend(fake_apple10).isNax());
    const d10_auto = selectPrefillMatmulBackendWithMode(fake_apple10, .auto);
    try std.testing.expectEqual(PrefillMatmulBackend.nax_apple10, d10_auto.backend);
    try std.testing.expectEqual(NaxFallbackReason.none, d10_auto.fallback_reason);
    const d10_sg = selectPrefillMatmulBackendWithMode(fake_apple10, .simdgroup);
    try std.testing.expectEqual(PrefillMatmulBackend.simdgroup_apple9, d10_sg.backend);
    try std.testing.expectEqual(NaxFallbackReason.forced_simdgroup, d10_sg.fallback_reason);

    var fake_apple11 = fake_apple10;
    fake_apple11.raw.gpu_apple_family = 11;
    fake_apple11.raw.apple_gpu_family = 11;
    fake_apple11.raw.supports_apple11 = 1;
    try std.testing.expect(fake_apple11.supportsApple11());
    try std.testing.expect(fake_apple11.supportsNaxTensorOps());
    try std.testing.expectEqual(PrefillMatmulBackend.nax_apple10, selectPrefillMatmulBackend(fake_apple11));
}

test "G17+G16: lazy NAX PSO selection, Apple9 fallback, strict-nax rejection, and canary auto-demotion" {
    var device = try metal.Device.open();
    defer device.close();
    const info = try device.info();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();

    // 1. Unified `.metallib` embeds all three NAX entry points without PSO creation.
    try std.testing.expect(try library.hasFunction("ami_dense_gemm_nax"));
    try std.testing.expect(try library.hasFunction("ami_dequant_gemm_nax"));
    try std.testing.expect(try library.hasFunction("ami_fp4_e2m1_gemm_nax"));
    try std.testing.expect(!try library.hasFunction("ami_nonexistent_kernel"));

    // 2. Forced `.simdgroup` mode always skips NAX PSO compilation.
    {
        const p_sg = try DequantPipelines.initWithOptions(device, library, .{
            .backend_mode = .simdgroup,
        });
        defer p_sg.close();
        try std.testing.expect(!p_sg.nax_pipelines_compiled);
        try std.testing.expectEqual(PrefillMatmulBackend.simdgroup_apple9, p_sg.activeBackend());
        try std.testing.expectEqual(NaxFallbackReason.forced_simdgroup, p_sg.naxFallbackReason());
        try std.testing.expect(!p_sg.useNaxPrefill());
    }

    // 3. On Apple9 (M4): `.auto` skips NAX PSOs, `.nax` falls back when non-strict,
    //    and `.nax` + `strict_nax` fails fast with `error.NaxRequiresApple10Gpu`.
    if (!info.supportsNaxTensorOps()) {
        if (std.mem.indexOf(u8, info.name(), "M4") != null) {
            try std.testing.expectEqual(@as(u8, 9), info.appleGpuFamily());
        }

        const p_auto = try DequantPipelines.initWithOptions(device, library, .{
            .backend_mode = .auto,
        });
        defer p_auto.close();
        try std.testing.expect(!p_auto.nax_pipelines_compiled);
        try std.testing.expectEqual(PrefillMatmulBackend.simdgroup_apple9, p_auto.active_backend);
        try std.testing.expectEqual(PrefillMatmulBackend.simdgroup_apple9, p_auto.activeBackend());
        try std.testing.expectEqual(NaxFallbackReason.unsupported_gpu_family, p_auto.nax_fallback_reason);
        try std.testing.expectEqual(NaxFallbackReason.unsupported_gpu_family, p_auto.naxFallbackReason());
        try std.testing.expect(!p_auto.useNaxPrefill());

        const p_nax_fallback = try DequantPipelines.initWithOptions(device, library, .{
            .backend_mode = .nax,
            .strict_nax = false,
        });
        defer p_nax_fallback.close();
        try std.testing.expect(!p_nax_fallback.nax_pipelines_compiled);
        try std.testing.expectEqual(PrefillMatmulBackend.simdgroup_apple9, p_nax_fallback.activeBackend());
        try std.testing.expectEqual(NaxFallbackReason.unsupported_gpu_family, p_nax_fallback.naxFallbackReason());

        try std.testing.expectError(
            error.NaxRequiresApple10Gpu,
            DequantPipelines.initWithOptions(device, library, .{
                .backend_mode = .nax,
                .strict_nax = true,
            }),
        );
    }

    // 4. Simulated NAX canary failure in non-strict mode auto-demotes cleanly to
    //    `simdgroup_apple9` (`nax_canary_failed`) and verifies the simdgroup canary passes.
    {
        var p_demote = try DequantPipelines.initWithOptions(device, library, .{
            .backend_mode = .auto,
            .strict_nax = false,
            .simulate_nax_canary_failure = true,
        });
        defer p_demote.close();

        const prof = try p_demote.runStartupNumericalCanaryProfileMut(&device);
        try std.testing.expect(prof.cold.nax_tested);
        try std.testing.expect(prof.cold.nax_demoted);
        try std.testing.expectEqual(PrefillMatmulBackend.simdgroup_apple9, prof.cold.active_backend);
        try std.testing.expectEqual(NaxFallbackReason.nax_canary_failed, prof.cold.nax_fallback_reason);
        try std.testing.expectEqual(PrefillMatmulBackend.simdgroup_apple9, p_demote.activeBackend());
        try std.testing.expectEqual(PrefillMatmulBackend.simdgroup_apple9, p_demote.active_backend);
        try std.testing.expectEqual(NaxFallbackReason.nax_canary_failed, p_demote.naxFallbackReason());
        try std.testing.expectEqual(NaxFallbackReason.nax_canary_failed, p_demote.nax_fallback_reason);
        try std.testing.expect(!p_demote.useNaxPrefill());
        try std.testing.expect(prof.cold.max_abs_err_f16 <= CANARY_TOL_F16);
        try std.testing.expect(prof.cold.max_abs_err_bf16 <= CANARY_TOL_BF16);
        try std.testing.expect(prof.cold.max_abs_err_q4 <= CANARY_TOL_Q4);
        try std.testing.expect(std.math.isFinite(prof.warm_elapsed_us));
    }

    // 5. Also verify auto-demotion works through a `const DequantPipelines` binding
    //    (matching existing M6.8 call sites in `qwen3.zig`, `graph.zig`, `main.zig`).
    {
        const p_const_demote = try DequantPipelines.initWithOptions(device, library, .{
            .backend_mode = .auto,
            .strict_nax = false,
            .simulate_nax_canary_failure = true,
        });
        defer p_const_demote.close();

        const res = try p_const_demote.runStartupNumericalCanary(&device);
        try std.testing.expect(res.nax_tested);
        try std.testing.expect(res.nax_demoted);
        try std.testing.expectEqual(PrefillMatmulBackend.simdgroup_apple9, p_const_demote.activeBackend());
        try std.testing.expectEqual(NaxFallbackReason.nax_canary_failed, p_const_demote.naxFallbackReason());
    }

    // 6. Simulated NAX canary failure under `strict_nax = true` returns a hard error.
    {
        const p_strict = try DequantPipelines.initWithOptions(device, library, .{
            .backend_mode = .auto,
            .strict_nax = true,
            .simulate_nax_canary_failure = true,
        });
        defer p_strict.close();

        try std.testing.expectError(
            error.MetallibNumericalCanaryFailed,
            p_strict.runStartupNumericalCanary(&device),
        );
        try std.testing.expectError(
            error.MetallibNumericalCanaryFailed,
            p_strict.runStartupNumericalCanaryProfile(&device),
        );
    }
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
        try std.testing.expect(r.g18_max_abs_err <= CANARY_TOL_Q4);
    }
    const validation_on = (std.c.getenv("MTL_SHADER_VALIDATION") != null or std.c.getenv("MTL_DEBUG_LAYER") != null);
    if (!validation_on) {
        try std.testing.expect(best_us < 350.0);
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

test "G18: fused residual epilogue matches unfused GEMV/GEMM + residual add across q4/q3/q2/f16/bf16" {
    const gpa = std.testing.allocator;
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try DequantPipelines.init(device, library);
    defer pipes.close();

    // Verify dense (f16, bf16) and quantized (q4, q3, q2) residual epilogues at M=1, M=17 (32x32), M=65 (64x64 wide).
    const ms = [_]usize{ 1, 17, 65 };
    const n: usize = 67;
    const k: usize = 128;

    const max_m: usize = 65;
    const w_dense_bytes = n * k * @sizeOf(u16);
    const Q4 = quant.QuantOps(.{ .bits = .q4, .group_size = 64 });
    const w_q4_bytes = n * try Q4.rowWeightBytes(k);
    const sb_q4_bytes = n * (try Q4.groupsPerRow(k)) * @sizeOf(u32);
    const x_bytes = max_m * k * @sizeOf(f16);
    const y_bytes = max_m * n * @sizeOf(f16);

    const arena_cap = try page.alignUp(w_dense_bytes + 256) +
        try page.alignUp(w_q4_bytes + 256) +
        try page.alignUp(sb_q4_bytes + 256) +
        try page.alignUp(x_bytes + 256) +
        5 * try page.alignUp(y_bytes + 256) +
        2 * page.APPLE_PAGE_SIZE;
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

    const w_dense_raw = try arena.allocPages(w_dense_bytes);
    const w_q_raw = try arena.allocPages(w_q4_bytes);
    const sb_q_raw = try arena.allocPages(sb_q4_bytes);
    const x_raw = try arena.allocAligned(x_bytes, 64);
    const res_raw = try arena.allocAligned(y_bytes, 64);
    const y_unfused_raw = try arena.allocAligned(y_bytes, 64);
    const y_fused_raw = try arena.allocAligned(y_bytes, 64);
    const y_q_unfused_raw = try arena.allocAligned(y_bytes, 64);
    const y_q_fused_raw = try arena.allocAligned(y_bytes, 64);

    const w_bits: []u16 = @alignCast(std.mem.bytesAsSlice(u16, w_dense_raw[0..w_dense_bytes]));
    const x_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, x_raw));
    const res_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, res_raw));
    const y_unfused: []f16 = @alignCast(std.mem.bytesAsSlice(f16, y_unfused_raw));
    const y_fused: []f16 = @alignCast(std.mem.bytesAsSlice(f16, y_fused_raw));
    const y_q_unfused: []f16 = @alignCast(std.mem.bytesAsSlice(f16, y_q_unfused_raw));
    const y_q_fused: []f16 = @alignCast(std.mem.bytesAsSlice(f16, y_q_fused_raw));

    for (x_f16, 0..) |*v, i| v.* = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast((i * 13 + 5) % 29)) - 14)) * 0.0625);
    for (res_f16, 0..) |*v, i| v.* = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast((i * 19 + 3) % 23)) - 11)) * 0.03125);
    for (w_q_raw[0..w_q4_bytes], 0..) |*b, i| b.* = @intCast((i * 37 + 11) & 0xFF);
    const sb_u16: []u16 = @alignCast(std.mem.bytesAsSlice(u16, sb_q_raw[0..sb_q4_bytes]));
    for (0..sb_u16.len / 2) |gi| {
        sb_u16[gi * 2 + 0] = @bitCast(@as(f16, 0.0625));
        sb_u16[gi * 2 + 1] = @bitCast(@as(f16, -0.375));
    }

    const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
    defer gpu_buf.close();
    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();
    try compute.makeResident(&.{gpu_buf});

    const x_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(x_raw.ptr) };
    const res_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(res_raw.ptr) };
    const wd_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(w_dense_raw.ptr) };
    const wq_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(w_q_raw.ptr) };
    const sbq_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(sb_q_raw.ptr) };
    const yu_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(y_unfused_raw.ptr) };
    const yf_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(y_fused_raw.ptr) };
    const yqu_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(y_q_unfused_raw.ptr) };
    const yqf_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(y_q_fused_raw.ptr) };

    const ref_q = try gpa.alloc(f32, max_m * n);
    defer gpa.free(ref_q);

    for ([_]quant.FloatDType{ .f16, .bf16 }) |dt| {
        for (w_bits, 0..) |*v, i| {
            v.* = quant.f32ToFloatBits(dt, @as(f32, @floatFromInt(@as(i32, @intCast((i * 23 + 7) % 31)) - 15)) * 0.015625);
        }
        for (ms) |m| {
            var frame = try compute.beginFrame(5000);
            errdefer frame.abort();
            if (m == 1) {
                try pipes.encodeDenseGemv(&frame, dt, .{ .x = x_b, .weight = wd_b, .y = yu_b, .n = @intCast(n), .k = @intCast(k) }, false);
                try pipes.encodeDenseGemv(&frame, dt, .{ .x = x_b, .weight = wd_b, .y = yf_b, .n = @intCast(n), .k = @intCast(k), .residual = res_b }, false);
            } else {
                try pipes.encodeDenseGemm(&frame, dt, .{ .x = x_b, .weight = wd_b, .y = yu_b, .m = @intCast(m), .n = @intCast(n), .k = @intCast(k) }, false);
                try pipes.encodeDenseGemm(&frame, dt, .{ .x = x_b, .weight = wd_b, .y = yf_b, .m = @intCast(m), .n = @intCast(n), .k = @intCast(k), .residual = res_b }, false);
            }
            const sig = try frame.commit();
            try compute.wait(sig, 5000);

            for (0..m * n) |idx| {
                const expected: f32 = @floatCast(@as(f16, @floatCast(@as(f32, @floatCast(y_unfused[idx])) + @as(f32, @floatCast(res_f16[idx])))));
                const got: f32 = @floatCast(y_fused[idx]);
                try std.testing.expectApproxEqAbs(expected, got, 2e-3);
            }
        }
    }

    for ([_]quant.QuantBits{ .q4, .q3, .q2 }) |bits| {
        const rb = switch (bits) {
            .q4 => try quant.QuantOps(.{ .bits = .q4, .group_size = 64 }).rowWeightBytes(k),
            .q3 => try quant.QuantOps(.{ .bits = .q3, .group_size = 64 }).rowWeightBytes(k),
            .q2 => try quant.QuantOps(.{ .bits = .q2, .group_size = 64 }).rowWeightBytes(k),
        };
        const ng = k / 64;
        for (ms) |m| {
            var frame = try compute.beginFrame(5000);
            errdefer frame.abort();
            if (m == 1) {
                try pipes.encodeGemv(&frame, bits, .{ .x = x_b, .qweight = wq_b, .scale_bias = sbq_b, .y = yqu_b, .n = @intCast(n), .k = @intCast(k) }, false);
                try pipes.encodeGemv(&frame, bits, .{ .x = x_b, .qweight = wq_b, .scale_bias = sbq_b, .y = yqf_b, .n = @intCast(n), .k = @intCast(k), .residual = res_b }, false);
            } else {
                try pipes.encodeGemm(&frame, bits, .{ .x = x_b, .qweight = wq_b, .scale_bias = sbq_b, .y = yqu_b, .m = @intCast(m), .n = @intCast(n), .k = @intCast(k) }, false);
                try pipes.encodeGemm(&frame, bits, .{ .x = x_b, .qweight = wq_b, .scale_bias = sbq_b, .y = yqf_b, .m = @intCast(m), .n = @intCast(n), .k = @intCast(k), .residual = res_b }, false);
            }
            const sig = try frame.commit();
            try compute.wait(sig, 5000);

            _ = rb;
            _ = ng;
            for (0..m * n) |idx| {
                const expected: f32 = @floatCast(@as(f16, @floatCast(@as(f32, @floatCast(y_q_unfused[idx])) + @as(f32, @floatCast(res_f16[idx])))));
                const got: f32 = @floatCast(y_q_fused[idx]);
                try std.testing.expectApproxEqAbs(expected, got, 4e-3);
            }
        }
    }
}

test "G18: homogeneous fused QKV and dense Gate+Up+SwiGLU match unfused paths and strictly guard mixed-DType GSQ" {
    const gpa = std.testing.allocator;
    const device = try metal.Device.open();
    defer device.close();
    const library = try metal.Library.openEmbedded(device);
    defer library.close();
    const pipes = try DequantPipelines.init(device, library);
    defer pipes.close();

    // Verify heterogeneous-GSQ guard helpers.
    try std.testing.expect(canFuseQkv(.q4_affine, .q4_affine, .q4_affine));
    try std.testing.expect(!canFuseQkv(.q4_affine, .q3_affine, .q4_affine));
    try std.testing.expect(!canFuseQkv(.f16, .f16, .bf16));
    try std.testing.expect(canFuseGateUp(.bf16, .bf16));
    try std.testing.expect(!canFuseGateUp(.q4_affine, .q2_affine));

    const max_m: usize = 65;
    const n_q: usize = 128;
    const n_k: usize = 64;
    const n_v: usize = 64;
    const n_ffn: usize = 132;
    const k: usize = 128;
    const max_n: usize = n_ffn;

    const w_dense_bytes = max_n * k * @sizeOf(u16);
    const Q4 = quant.QuantOps(.{ .bits = .q4, .group_size = 64 });
    const sb_bytes = max_n * (try Q4.groupsPerRow(k)) * @sizeOf(u32);
    const x_bytes = max_m * k * @sizeOf(f16);
    const y_bytes = max_m * max_n * @sizeOf(f16);

    const arena_cap = 3 * try page.alignUp(w_dense_bytes + 256) +
        3 * try page.alignUp(sb_bytes + 256) +
        try page.alignUp(x_bytes + 256) +
        8 * try page.alignUp(y_bytes + 256) +
        2 * page.APPLE_PAGE_SIZE;
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

    const w0_raw = try arena.allocPages(w_dense_bytes);
    const w1_raw = try arena.allocPages(w_dense_bytes);
    const w2_raw = try arena.allocPages(w_dense_bytes);
    const sb0_raw = try arena.allocPages(sb_bytes);
    const sb1_raw = try arena.allocPages(sb_bytes);
    const sb2_raw = try arena.allocPages(sb_bytes);
    const x_raw = try arena.allocAligned(x_bytes, 64);

    const y_q_u_raw = try arena.allocAligned(y_bytes, 64);
    const y_k_u_raw = try arena.allocAligned(y_bytes, 64);
    const y_v_u_raw = try arena.allocAligned(y_bytes, 64);
    const y_q_f_raw = try arena.allocAligned(y_bytes, 64);
    const y_k_f_raw = try arena.allocAligned(y_bytes, 64);
    const y_v_f_raw = try arena.allocAligned(y_bytes, 64);
    const y_gate_raw = try arena.allocAligned(y_bytes, 64);
    const y_up_raw = try arena.allocAligned(y_bytes, 64);

    const x_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, x_raw));
    for (x_f16, 0..) |*v, i| v.* = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast((i * 17 + 3) % 27)) - 13)) * 0.0625);

    const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
    defer gpu_buf.close();
    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();
    try compute.makeResident(&.{gpu_buf});

    const x_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(x_raw.ptr) };
    const w0_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(w0_raw.ptr) };
    const w1_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(w1_raw.ptr) };
    const w2_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(w2_raw.ptr) };
    const sb0_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(sb0_raw.ptr) };
    const sb1_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(sb1_raw.ptr) };
    const sb2_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(sb2_raw.ptr) };
    const yqu_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(y_q_u_raw.ptr) };
    const yku_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(y_k_u_raw.ptr) };
    const yvu_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(y_v_u_raw.ptr) };
    const yqf_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(y_q_f_raw.ptr) };
    const ykf_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(y_k_f_raw.ptr) };
    const yvf_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(y_v_f_raw.ptr) };
    const yg_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(y_gate_raw.ptr) };
    const yu_b: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(y_up_raw.ptr) };

    const yqu_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, y_q_u_raw));
    const yku_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, y_k_u_raw));
    const yvu_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, y_v_u_raw));
    const yqf_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, y_q_f_raw));
    const ykf_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, y_k_f_raw));
    const yvf_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, y_v_f_raw));
    const yg_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, y_gate_raw));
    const yu_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, y_up_raw));

    const cpu_gate = try gpa.alloc(f32, max_m * n_ffn);
    defer gpa.free(cpu_gate);
    const cpu_up = try gpa.alloc(f32, max_m * n_ffn);
    defer gpa.free(cpu_up);

    // 1. Homogeneous float (f16, bf16) and quantized (q4, q3, q2) QKV + Gate+Up+SwiGLU at M=1 and M=65.
    for ([_]amf.DType{ .f16, .bf16, .q4_affine, .q3_affine, .q2_affine }) |dt| {
        if (dt == .f16 or dt == .bf16) {
            const fdt: quant.FloatDType = if (dt == .bf16) .bf16 else .f16;
            for ([_][]u8{ w0_raw, w1_raw, w2_raw }, 0..) |wbuf, wi| {
                const ws: []u16 = @alignCast(std.mem.bytesAsSlice(u16, wbuf[0..w_dense_bytes]));
                for (ws, 0..) |*v, i| {
                    v.* = quant.f32ToFloatBits(fdt, @as(f32, @floatFromInt(@as(i32, @intCast((i * (11 + wi * 6) + 5) % 29)) - 14)) * 0.015625);
                }
            }
        } else {
            for ([_][]u8{ w0_raw, w1_raw, w2_raw }, [_][]u8{ sb0_raw, sb1_raw, sb2_raw }, 0..) |wbuf, sbuf, wi| {
                for (wbuf[0..w_dense_bytes], 0..) |*b, i| b.* = @intCast((i * (31 + wi * 12) + 7) & 0xFF);
                const sbu16: []u16 = @alignCast(std.mem.bytesAsSlice(u16, sbuf[0..sb_bytes]));
                for (0..sbu16.len / 2) |gi| {
                    sbu16[gi * 2 + 0] = @bitCast(@as(f16, 0.0625));
                    sbu16[gi * 2 + 1] = @bitCast(@as(f16, -0.375));
                }
            }
        }

        for ([_]usize{ 1, 65 }) |m| {
            var frame = try compute.beginFrame(5000);
            errdefer frame.abort();

            const q_unfused: QkvProjBinding = .{ .dtype = dt, .weight = w0_b, .scale_bias = sb0_b, .y = yqu_b, .n = @intCast(n_q) };
            const k_unfused: QkvProjBinding = .{ .dtype = dt, .weight = w1_b, .scale_bias = sb1_b, .y = yku_b, .n = @intCast(n_k) };
            const v_unfused: QkvProjBinding = .{ .dtype = dt, .weight = w2_b, .scale_bias = sb2_b, .y = yvu_b, .n = @intCast(n_v) };
            if (m == 1) {
                try pipes.encodeSingleProjGemv(&frame, x_b, q_unfused, @intCast(k), false);
                try pipes.encodeSingleProjGemv(&frame, x_b, k_unfused, @intCast(k), false);
                try pipes.encodeSingleProjGemv(&frame, x_b, v_unfused, @intCast(k), false);
            } else {
                try pipes.encodeSingleProjGemm(&frame, x_b, q_unfused, @intCast(m), @intCast(k), false);
                try pipes.encodeSingleProjGemm(&frame, x_b, k_unfused, @intCast(m), @intCast(k), false);
                try pipes.encodeSingleProjGemm(&frame, x_b, v_unfused, @intCast(m), @intCast(k), false);
            }

            const qkv_path = try pipes.encodeFusedQkvGemm(&frame, .{
                .x = x_b,
                .q = .{ .dtype = dt, .weight = w0_b, .scale_bias = sb0_b, .y = yqf_b, .n = @intCast(n_q) },
                .k_proj = .{ .dtype = dt, .weight = w1_b, .scale_bias = sb1_b, .y = ykf_b, .n = @intCast(n_k) },
                .v = .{ .dtype = dt, .weight = w2_b, .scale_bias = sb2_b, .y = yvf_b, .n = @intCast(n_v) },
                .m = @intCast(m),
                .k = @intCast(k),
            }, false);
            try std.testing.expectEqual(FusionPath.fused, qkv_path);

            // Unfused Gate + Up projections -> y_gate_raw, y_up_raw; fused SwiGLU -> yqu_b.
            const g_proj: QkvProjBinding = .{ .dtype = dt, .weight = w0_b, .scale_bias = sb0_b, .y = yg_b, .n = @intCast(n_ffn) };
            const u_proj: QkvProjBinding = .{ .dtype = dt, .weight = w1_b, .scale_bias = sb1_b, .y = yu_b, .n = @intCast(n_ffn) };
            if (m == 1) {
                try pipes.encodeSingleProjGemv(&frame, x_b, g_proj, @intCast(k), true);
                try pipes.encodeSingleProjGemv(&frame, x_b, u_proj, @intCast(k), false);
            } else {
                try pipes.encodeSingleProjGemm(&frame, x_b, g_proj, @intCast(m), @intCast(k), true);
                try pipes.encodeSingleProjGemm(&frame, x_b, u_proj, @intCast(m), @intCast(k), false);
            }
            const sig_qkv = try frame.commit();
            try compute.wait(sig_qkv, 5000);

            for (0..m * n_q) |i| try std.testing.expectApproxEqAbs(@as(f32, @floatCast(yqu_f16[i])), @as(f32, @floatCast(yqf_f16[i])), 4e-3);
            for (0..m * n_k) |i| try std.testing.expectApproxEqAbs(@as(f32, @floatCast(yku_f16[i])), @as(f32, @floatCast(ykf_f16[i])), 4e-3);
            for (0..m * n_v) |i| try std.testing.expectApproxEqAbs(@as(f32, @floatCast(yvu_f16[i])), @as(f32, @floatCast(yvf_f16[i])), 4e-3);

            var frame2 = try compute.beginFrame(5000);
            errdefer frame2.abort();
            try pipes.encodeDenseGateUpSwiglu(&frame2, .{
                .x = x_b,
                .gate_dtype = dt,
                .w_gate = w0_b,
                .sb_gate = sb0_b,
                .up_dtype = dt,
                .w_up = w1_b,
                .sb_up = sb1_b,
                .y = yqu_b,
                .m = @intCast(m),
                .n = @intCast(n_ffn),
                .k = @intCast(k),
            }, false);
            const sig = try frame2.commit();
            try compute.wait(sig, 5000);

            // SwiGLU parity vs unfused Gate/Up + fp32 SiLU(g)*u
            for (0..m * n_ffn) |i| {
                const gv: f32 = @floatCast(yg_f16[i]);
                const uv: f32 = @floatCast(yu_f16[i]);
                const silu_g = gv / (1.0 + @exp(-gv));
                const expected: f32 = @floatCast(@as(f16, @floatCast(silu_g * uv)));
                const got: f32 = @floatCast(yqu_f16[i]);
                try std.testing.expectApproxEqAbs(expected, got, 5e-3);
            }
        }
    }

    // 2. Verify heterogeneous-GSQ layers cleanly reject direct fusion and fall back to unfused QKV.
    {
        var frame = try compute.beginFrame(5000);
        errdefer frame.abort();
        try std.testing.expectError(error.HeterogeneousDType, pipes.encodeDenseGateUpSwiglu(&frame, .{
            .x = x_b,
            .gate_dtype = .q4_affine,
            .w_gate = w0_b,
            .sb_gate = sb0_b,
            .up_dtype = .q3_affine,
            .w_up = w1_b,
            .sb_up = sb1_b,
            .y = yg_b,
            .m = 32,
            .n = @intCast(n_ffn),
            .k = @intCast(k),
        }, false));
        try std.testing.expectError(error.HeterogeneousDType, pipes.encodeFusedQkvGemm(&frame, .{
            .x = x_b,
            .q = .{ .dtype = .q4_affine, .weight = w0_b, .scale_bias = sb0_b, .y = yqf_b, .n = @intCast(n_q) },
            .k_proj = .{ .dtype = .q3_affine, .weight = w1_b, .scale_bias = sb1_b, .y = ykf_b, .n = @intCast(n_k) },
            .v = .{ .dtype = .q4_affine, .weight = w2_b, .scale_bias = sb2_b, .y = yvf_b, .n = @intCast(n_v) },
            .m = 1,
            .k = @intCast(k),
            .allow_unfused_fallback = false,
        }, false));
        const fallback_path = try pipes.encodeFusedQkvGemm(&frame, .{
            .x = x_b,
            .q = .{ .dtype = .q4_affine, .weight = w0_b, .scale_bias = sb0_b, .y = yqf_b, .n = @intCast(n_q) },
            .k_proj = .{ .dtype = .q3_affine, .weight = w1_b, .scale_bias = sb1_b, .y = ykf_b, .n = @intCast(n_k) },
            .v = .{ .dtype = .q2_affine, .weight = w2_b, .scale_bias = sb2_b, .y = yvf_b, .n = @intCast(n_v) },
            .m = 8,
            .k = @intCast(k),
            .allow_unfused_fallback = true,
        }, false);
        try std.testing.expectEqual(FusionPath.unfused_heterogeneous, fallback_path);
        const sig = try frame.commit();
        try compute.wait(sig, 5000);
    }
}
