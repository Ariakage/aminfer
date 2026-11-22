//! M6.5 G14 zero-copy multimodal vision/video ingress (`ADR-0002`, `ADR-0004`,
//! `ADR-0011`, `ADR-0019`):
//! - Zero-copy `IOSurface`-backed `CVPixelBuffer` (`BGRA8Unorm`, bi-planar
//!   `NV12`, `OneComponent16Half`) bound directly as Metal 4 `MTLTexture` planes
//!   and CoreML / ANE inputs (`0` host `memcpy`).
//! - Preallocated `VisionIngress` workspace in a single `16 KiB`-aligned
//!   `WiredArena` (`0` dynamic heap allocation on the hot path).
//! - Metal 4 kernels (`vision_ingress.metal`) for `BGRA8` / `NV12` BT.709 &
//!   sRGB conversion, `(val - mean) * inv_std` normalization, `[t_patch, p, p]`
//!   spatiotemporal patchify (`t_patch = 1|2`, `patch_size = 14|16`), optional
//!   `2x2` spatial patch merge, and fused `f16`/`bf16` linear patch embedding.
//! - Deterministic 3D MRoPE position builder `buildMultimodalMropePositions`
//!   (`[3, total_tokens]u32` planar `[t_pos, h_pos, w_pos]`) plus zero-copy
//!   interleaved `[total_tokens]Pos3D` packing feeding M6 `G9` interleaved MRoPE
//!   (`[11, 11, 10]`).

const std = @import("std");
const aminfer = @import("aminfer");
const metal = aminfer.metal;
const page = aminfer.page;
const arena_mod = aminfer.arena;
const budget_mod = aminfer.budget;
const quant = aminfer.quant;
const hybrid = aminfer.hybrid;

pub const MAX_PATCH_SIZE: u32 = 32;
pub const PATCHIFY_TG_THREADS: u64 = 256;
pub const EMBED_TG_CHANNELS: u32 = 8;

pub const Error = error{
    InvalidGeometry,
    InvalidFormat,
    InvalidHiddenSize,
    InvalidTokenOffset,
    OutputBufferTooSmall,
    WeightBufferTooSmall,
    BiasBufferTooSmall,
    Overflow,
};

pub const EncodeError = Error || metal.Error;

pub const MropeError = error{
    InvalidGrid,
    EmptySequence,
    BufferTooSmall,
    Overflow,
};

/// Per-channel RGB normalization constants `(val - mean[c]) * inv_std[c]`.
pub const NormParams = struct {
    mean: [3]f32,
    inv_std: [3]f32,
    apply_srgb_to_linear: bool = false,

    /// OpenAI CLIP / Qwen2-VL / Qwen3-VL standard ImageNet-CLIP RGB statistics.
    pub const QWEN_VL_DEFAULT: NormParams = .{
        .mean = .{ 0.48145466, 0.4578275, 0.40821073 },
        .inv_std = .{ 1.0 / 0.26862954, 1.0 / 0.26130258, 1.0 / 0.27577711 },
        .apply_srgb_to_linear = false,
    };

    /// SigLIP / Gemma-style `[-1, 1]` normalization (`mean = 0.5, std = 0.5`).
    pub const SIGLIP_DEFAULT: NormParams = .{
        .mean = .{ 0.5, 0.5, 0.5 },
        .inv_std = .{ 2.0, 2.0, 2.0 },
        .apply_srgb_to_linear = false,
    };

    /// Identity normalization (`mean = 0, inv_std = 1`).
    pub const IDENTITY: NormParams = .{
        .mean = .{ 0.0, 0.0, 0.0 },
        .inv_std = .{ 1.0, 1.0, 1.0 },
        .apply_srgb_to_linear = false,
    };
};

/// Spatiotemporal patchify + spatial-merge geometry for one vision dispatch.
pub const PatchGeometry = struct {
    width: u32,
    height: u32,
    patch_size: u32 = 14,
    /// Temporal tube depth: `1` for still images, `2` for paired video frames.
    t_patch: u32 = 1,
    /// Spatial merge factor: `1` (unmerged) or `2` (`2x2` super-patch merge).
    spatial_merge: u32 = 1,

    pub fn validate(self: PatchGeometry) Error!void {
        if (self.width == 0 or self.height == 0 or
            self.width > metal.MAX_TEXTURE_DIM or self.height > metal.MAX_TEXTURE_DIM)
        {
            return error.InvalidGeometry;
        }
        if (self.patch_size == 0 or self.patch_size > MAX_PATCH_SIZE) {
            return error.InvalidGeometry;
        }
        if (self.t_patch != 1 and self.t_patch != 2) {
            return error.InvalidGeometry;
        }
        if (self.spatial_merge != 1 and self.spatial_merge != 2) {
            return error.InvalidGeometry;
        }
        const tile_div = std.math.mul(u32, self.patch_size, self.spatial_merge) catch return error.Overflow;
        if ((self.width % tile_div) != 0 or (self.height % tile_div) != 0) {
            return error.InvalidGeometry;
        }
    }

    pub fn gridH(self: PatchGeometry) u32 {
        return self.height / self.patch_size;
    }

    pub fn gridW(self: PatchGeometry) u32 {
        return self.width / self.patch_size;
    }

    pub fn mergedH(self: PatchGeometry) u32 {
        return self.gridH() / self.spatial_merge;
    }

    pub fn mergedW(self: PatchGeometry) u32 {
        return self.gridW() / self.spatial_merge;
    }

    pub fn numTokens(self: PatchGeometry) Error!u32 {
        return std.math.mul(u32, self.mergedH(), self.mergedW()) catch return error.Overflow;
    }

    /// Raw elements per single unmerged patch tube: `3 * t_patch * patch_size * patch_size`.
    pub fn rawPatchDim(self: PatchGeometry) Error!u32 {
        const area = std.math.mul(u32, self.patch_size, self.patch_size) catch return error.Overflow;
        const tube = std.math.mul(u32, self.t_patch, area) catch return error.Overflow;
        return std.math.mul(u32, 3, tube) catch return error.Overflow;
    }

    /// Total elements per output token: `spatial_merge^2 * rawPatchDim`.
    pub fn tokenInDim(self: PatchGeometry) Error!u32 {
        const raw_dim = try self.rawPatchDim();
        const merge_sq = self.spatial_merge * self.spatial_merge;
        return std.math.mul(u32, merge_sq, raw_dim) catch return error.Overflow;
    }
};

/// Metal 4 uniform block matching `VisionPatchifyParams` in `vision_ingress.metal`.
pub const VisionPatchifyParams = extern struct {
    width: u32,
    height: u32,
    patch_size: u32,
    t_patch: u32,
    spatial_merge: u32,
    hidden_size: u32,
    is_full_range: u32,
    apply_srgb_to_linear: u32,
    has_bias: u32,
    token_offset: u32,
    mean_r: f32,
    mean_g: f32,
    mean_b: f32,
    inv_std_r: f32,
    inv_std_g: f32,
    inv_std_b: f32,

    comptime {
        std.debug.assert(@sizeOf(VisionPatchifyParams) == 64);
    }
};

pub const PatchifyBgra8Args = struct {
    frame0: metal.VisionFrame,
    /// Second temporal frame when `geometry.t_patch == 2`; defaults to `frame0` when `null`.
    frame1: ?metal.VisionFrame = null,
    out: metal.BufferBinding,
    out_dtype: quant.FloatDType = .f16,
    geometry: PatchGeometry,
    norm: NormParams = NormParams.QWEN_VL_DEFAULT,
    token_offset: u32 = 0,
};

pub const PatchifyNv12Args = struct {
    frame0: metal.VisionFrame,
    /// Second temporal frame when `geometry.t_patch == 2`; defaults to `frame0` when `null`.
    frame1: ?metal.VisionFrame = null,
    out: metal.BufferBinding,
    out_dtype: quant.FloatDType = .f16,
    geometry: PatchGeometry,
    /// `true` for full-range `420f`, `false` for video-range `420v`.
    is_full_range: bool = true,
    norm: NormParams = NormParams.QWEN_VL_DEFAULT,
    token_offset: u32 = 0,
};

pub const PatchifyEmbedBgra8Args = struct {
    frame0: metal.VisionFrame,
    frame1: ?metal.VisionFrame = null,
    /// Row-major `[hidden_size, token_in_dim]` weights in `weight_dtype`.
    weight: metal.BufferBinding,
    /// Optional `[hidden_size]` bias vector in `weight_dtype`.
    bias: ?metal.BufferBinding = null,
    out: metal.BufferBinding,
    weight_dtype: quant.FloatDType = .f16,
    out_dtype: quant.FloatDType = .f16,
    hidden_size: u32,
    geometry: PatchGeometry,
    norm: NormParams = NormParams.QWEN_VL_DEFAULT,
    token_offset: u32 = 0,
};

pub const PatchifyEmbedNv12Args = struct {
    frame0: metal.VisionFrame,
    frame1: ?metal.VisionFrame = null,
    /// Row-major `[hidden_size, token_in_dim]` weights in `weight_dtype`.
    weight: metal.BufferBinding,
    /// Optional `[hidden_size]` bias vector in `weight_dtype`.
    bias: ?metal.BufferBinding = null,
    out: metal.BufferBinding,
    weight_dtype: quant.FloatDType = .f16,
    out_dtype: quant.FloatDType = .f16,
    hidden_size: u32,
    geometry: PatchGeometry,
    is_full_range: bool = true,
    norm: NormParams = NormParams.QWEN_VL_DEFAULT,
    token_offset: u32 = 0,
};

/// Startup-compiled Metal 4 compute pipelines for `vision_ingress.metal`.
pub const VisionPipelines = struct {
    patchify_bgra8: [2]metal.Pipeline,
    patchify_nv12: [2]metal.Pipeline,
    embed_bgra8: [2][2]metal.Pipeline,
    embed_nv12: [2][2]metal.Pipeline,

    pub fn init(device: metal.Device, library: metal.Library) metal.Error!VisionPipelines {
        const p_bgra_f16 = try metal.Pipeline.initSpecialized(device, library, "ami_vision_patchify_bgra8", &.{
            .{ .index = 0, .value = .{ .bool = false } },
            .{ .index = 1, .value = .{ .bool = false } },
        });
        errdefer p_bgra_f16.close();

        const p_bgra_bf16 = try metal.Pipeline.initSpecialized(device, library, "ami_vision_patchify_bgra8", &.{
            .{ .index = 0, .value = .{ .bool = false } },
            .{ .index = 1, .value = .{ .bool = true } },
        });
        errdefer p_bgra_bf16.close();

        const p_nv12_f16 = try metal.Pipeline.initSpecialized(device, library, "ami_vision_patchify_nv12", &.{
            .{ .index = 0, .value = .{ .bool = false } },
            .{ .index = 1, .value = .{ .bool = false } },
        });
        errdefer p_nv12_f16.close();

        const p_nv12_bf16 = try metal.Pipeline.initSpecialized(device, library, "ami_vision_patchify_nv12", &.{
            .{ .index = 0, .value = .{ .bool = false } },
            .{ .index = 1, .value = .{ .bool = true } },
        });
        errdefer p_nv12_bf16.close();

        const eb_ff = try metal.Pipeline.initSpecialized(device, library, "ami_vision_patchify_embed_bgra8", &.{
            .{ .index = 0, .value = .{ .bool = false } },
            .{ .index = 1, .value = .{ .bool = false } },
        });
        errdefer eb_ff.close();
        const eb_fb = try metal.Pipeline.initSpecialized(device, library, "ami_vision_patchify_embed_bgra8", &.{
            .{ .index = 0, .value = .{ .bool = false } },
            .{ .index = 1, .value = .{ .bool = true } },
        });
        errdefer eb_fb.close();
        const eb_bf = try metal.Pipeline.initSpecialized(device, library, "ami_vision_patchify_embed_bgra8", &.{
            .{ .index = 0, .value = .{ .bool = true } },
            .{ .index = 1, .value = .{ .bool = false } },
        });
        errdefer eb_bf.close();
        const eb_bb = try metal.Pipeline.initSpecialized(device, library, "ami_vision_patchify_embed_bgra8", &.{
            .{ .index = 0, .value = .{ .bool = true } },
            .{ .index = 1, .value = .{ .bool = true } },
        });
        errdefer eb_bb.close();

        const en_ff = try metal.Pipeline.initSpecialized(device, library, "ami_vision_patchify_embed_nv12", &.{
            .{ .index = 0, .value = .{ .bool = false } },
            .{ .index = 1, .value = .{ .bool = false } },
        });
        errdefer en_ff.close();
        const en_fb = try metal.Pipeline.initSpecialized(device, library, "ami_vision_patchify_embed_nv12", &.{
            .{ .index = 0, .value = .{ .bool = false } },
            .{ .index = 1, .value = .{ .bool = true } },
        });
        errdefer en_fb.close();
        const en_bf = try metal.Pipeline.initSpecialized(device, library, "ami_vision_patchify_embed_nv12", &.{
            .{ .index = 0, .value = .{ .bool = true } },
            .{ .index = 1, .value = .{ .bool = false } },
        });
        errdefer en_bf.close();
        const en_bb = try metal.Pipeline.initSpecialized(device, library, "ami_vision_patchify_embed_nv12", &.{
            .{ .index = 0, .value = .{ .bool = true } },
            .{ .index = 1, .value = .{ .bool = true } },
        });
        errdefer en_bb.close();

        return .{
            .patchify_bgra8 = .{ p_bgra_f16, p_bgra_bf16 },
            .patchify_nv12 = .{ p_nv12_f16, p_nv12_bf16 },
            .embed_bgra8 = .{ .{ eb_ff, eb_fb }, .{ eb_bf, eb_bb } },
            .embed_nv12 = .{ .{ en_ff, en_fb }, .{ en_bf, en_bb } },
        };
    }

    pub fn close(self: VisionPipelines) void {
        for (self.patchify_bgra8) |p| p.close();
        for (self.patchify_nv12) |p| p.close();
        for (self.embed_bgra8) |row| {
            for (row) |p| p.close();
        }
        for (self.embed_nv12) |row| {
            for (row) |p| p.close();
        }
    }

    pub fn encodePatchifyBgra8(
        self: *const VisionPipelines,
        frame: *metal.Compute.Frame,
        args: PatchifyBgra8Args,
        insert_barrier_before: bool,
    ) EncodeError!void {
        try args.geometry.validate();
        const num_tokens = try args.geometry.numTokens();
        _ = try args.geometry.tokenInDim();
        const f1 = args.frame1 orelse args.frame0;

        const params: VisionPatchifyParams = .{
            .width = args.geometry.width,
            .height = args.geometry.height,
            .patch_size = args.geometry.patch_size,
            .t_patch = args.geometry.t_patch,
            .spatial_merge = args.geometry.spatial_merge,
            .hidden_size = 0,
            .is_full_range = 1,
            .apply_srgb_to_linear = if (args.norm.apply_srgb_to_linear) 1 else 0,
            .has_bias = 0,
            .token_offset = args.token_offset,
            .mean_r = args.norm.mean[0],
            .mean_g = args.norm.mean[1],
            .mean_b = args.norm.mean[2],
            .inv_std_r = args.norm.inv_std[0],
            .inv_std_g = args.norm.inv_std[1],
            .inv_std_b = args.norm.inv_std[2],
        };

        try frame.encodeThreadgroupsWithVisionTextures(
            self.patchify_bgra8[@backingInt(args.out_dtype)],
            &.{args.out},
            &.{
                .{ .frame = args.frame0, .plane = 0 },
                .{ .frame = f1, .plane = 0 },
            },
            std.mem.asBytes(&params),
            .{ num_tokens, 1, 1 },
            .{ PATCHIFY_TG_THREADS, 1, 1 },
            insert_barrier_before,
        );
    }

    pub fn encodePatchifyNv12(
        self: *const VisionPipelines,
        frame: *metal.Compute.Frame,
        args: PatchifyNv12Args,
        insert_barrier_before: bool,
    ) EncodeError!void {
        try args.geometry.validate();
        if ((args.geometry.width & 1) != 0 or (args.geometry.height & 1) != 0) {
            return error.InvalidGeometry;
        }
        const num_tokens = try args.geometry.numTokens();
        _ = try args.geometry.tokenInDim();
        const f1 = args.frame1 orelse args.frame0;

        const params: VisionPatchifyParams = .{
            .width = args.geometry.width,
            .height = args.geometry.height,
            .patch_size = args.geometry.patch_size,
            .t_patch = args.geometry.t_patch,
            .spatial_merge = args.geometry.spatial_merge,
            .hidden_size = 0,
            .is_full_range = if (args.is_full_range) 1 else 0,
            .apply_srgb_to_linear = if (args.norm.apply_srgb_to_linear) 1 else 0,
            .has_bias = 0,
            .token_offset = args.token_offset,
            .mean_r = args.norm.mean[0],
            .mean_g = args.norm.mean[1],
            .mean_b = args.norm.mean[2],
            .inv_std_r = args.norm.inv_std[0],
            .inv_std_g = args.norm.inv_std[1],
            .inv_std_b = args.norm.inv_std[2],
        };

        try frame.encodeThreadgroupsWithVisionTextures(
            self.patchify_nv12[@backingInt(args.out_dtype)],
            &.{args.out},
            &.{
                .{ .frame = args.frame0, .plane = 0 },
                .{ .frame = args.frame0, .plane = 1 },
                .{ .frame = f1, .plane = 0 },
                .{ .frame = f1, .plane = 1 },
            },
            std.mem.asBytes(&params),
            .{ num_tokens, 1, 1 },
            .{ PATCHIFY_TG_THREADS, 1, 1 },
            insert_barrier_before,
        );
    }

    pub fn encodePatchifyEmbedBgra8(
        self: *const VisionPipelines,
        frame: *metal.Compute.Frame,
        args: PatchifyEmbedBgra8Args,
        insert_barrier_before: bool,
    ) EncodeError!void {
        try args.geometry.validate();
        if (args.hidden_size == 0) return error.InvalidHiddenSize;
        const num_tokens = try args.geometry.numTokens();
        _ = try args.geometry.tokenInDim();
        const f1 = args.frame1 orelse args.frame0;
        const bias_binding = args.bias orelse args.weight;

        const params: VisionPatchifyParams = .{
            .width = args.geometry.width,
            .height = args.geometry.height,
            .patch_size = args.geometry.patch_size,
            .t_patch = args.geometry.t_patch,
            .spatial_merge = args.geometry.spatial_merge,
            .hidden_size = args.hidden_size,
            .is_full_range = 1,
            .apply_srgb_to_linear = if (args.norm.apply_srgb_to_linear) 1 else 0,
            .has_bias = if (args.bias != null) 1 else 0,
            .token_offset = args.token_offset,
            .mean_r = args.norm.mean[0],
            .mean_g = args.norm.mean[1],
            .mean_b = args.norm.mean[2],
            .inv_std_r = args.norm.inv_std[0],
            .inv_std_g = args.norm.inv_std[1],
            .inv_std_b = args.norm.inv_std[2],
        };

        const groups_x: u64 = (@as(u64, args.hidden_size) + EMBED_TG_CHANNELS - 1) / EMBED_TG_CHANNELS;
        try frame.encodeThreadgroupsWithVisionTextures(
            self.embed_bgra8[@backingInt(args.weight_dtype)][@backingInt(args.out_dtype)],
            &.{ args.weight, bias_binding, args.out },
            &.{
                .{ .frame = args.frame0, .plane = 0 },
                .{ .frame = f1, .plane = 0 },
            },
            std.mem.asBytes(&params),
            .{ groups_x, num_tokens, 1 },
            .{ PATCHIFY_TG_THREADS, 1, 1 },
            insert_barrier_before,
        );
    }

    pub fn encodePatchifyEmbedNv12(
        self: *const VisionPipelines,
        frame: *metal.Compute.Frame,
        args: PatchifyEmbedNv12Args,
        insert_barrier_before: bool,
    ) EncodeError!void {
        try args.geometry.validate();
        if ((args.geometry.width & 1) != 0 or (args.geometry.height & 1) != 0) {
            return error.InvalidGeometry;
        }
        if (args.hidden_size == 0) return error.InvalidHiddenSize;
        const num_tokens = try args.geometry.numTokens();
        _ = try args.geometry.tokenInDim();
        const f1 = args.frame1 orelse args.frame0;
        const bias_binding = args.bias orelse args.weight;

        const params: VisionPatchifyParams = .{
            .width = args.geometry.width,
            .height = args.geometry.height,
            .patch_size = args.geometry.patch_size,
            .t_patch = args.geometry.t_patch,
            .spatial_merge = args.geometry.spatial_merge,
            .hidden_size = args.hidden_size,
            .is_full_range = if (args.is_full_range) 1 else 0,
            .apply_srgb_to_linear = if (args.norm.apply_srgb_to_linear) 1 else 0,
            .has_bias = if (args.bias != null) 1 else 0,
            .token_offset = args.token_offset,
            .mean_r = args.norm.mean[0],
            .mean_g = args.norm.mean[1],
            .mean_b = args.norm.mean[2],
            .inv_std_r = args.norm.inv_std[0],
            .inv_std_g = args.norm.inv_std[1],
            .inv_std_b = args.norm.inv_std[2],
        };

        const groups_x: u64 = (@as(u64, args.hidden_size) + EMBED_TG_CHANNELS - 1) / EMBED_TG_CHANNELS;
        try frame.encodeThreadgroupsWithVisionTextures(
            self.embed_nv12[@backingInt(args.weight_dtype)][@backingInt(args.out_dtype)],
            &.{ args.weight, bias_binding, args.out },
            &.{
                .{ .frame = args.frame0, .plane = 0 },
                .{ .frame = args.frame0, .plane = 1 },
                .{ .frame = f1, .plane = 0 },
                .{ .frame = f1, .plane = 1 },
            },
            std.mem.asBytes(&params),
            .{ groups_x, num_tokens, 1 },
            .{ PATCHIFY_TG_THREADS, 1, 1 },
            insert_barrier_before,
        );
    }
};

// ===========================================================================
// Deterministic 3D MRoPE Position Builder (`[3, total_tokens]u32` + `Pos3D`)
// ===========================================================================

/// Merged visual token grid geometry `(grid_t, grid_h, grid_w)` for 3D MRoPE.
pub const VisionGridSpec = struct {
    grid_t: u32,
    grid_h: u32,
    grid_w: u32,
    /// Temporal coordinate increment per temporal tube (default `1`).
    temporal_stride: u32 = 1,

    pub fn validate(self: VisionGridSpec) MropeError!void {
        if (self.grid_t == 0 or self.grid_h == 0 or self.grid_w == 0 or self.temporal_stride == 0) {
            return error.InvalidGrid;
        }
        _ = try self.numTokens();
    }

    pub fn numTokens(self: VisionGridSpec) MropeError!u32 {
        const hw = std.math.mul(u32, self.grid_h, self.grid_w) catch return error.Overflow;
        return std.math.mul(u32, self.grid_t, hw) catch return error.Overflow;
    }

    pub fn maxDelta(self: VisionGridSpec) MropeError!u32 {
        const dt_span = std.math.mul(u32, self.grid_t - 1, self.temporal_stride) catch return error.Overflow;
        return @max(dt_span, @max(self.grid_h - 1, self.grid_w - 1));
    }
};

/// One contiguous multimodal segment in a prompt sequence.
pub const MropeSegment = union(enum) {
    text: u32,
    vision: VisionGridSpec,
};

/// Summary returned by `buildMultimodalMropePositions*`.
pub const MropeBuildResult = struct {
    /// Total tokens across all text and visual segments.
    total_tokens: u32,
    /// Next 1D text position after the final token (`max_pos + 1`).
    next_text_pos: u32,
    /// Signed offset `next_text_pos - total_tokens` used to shift decode positions
    /// after multimodal prefill (Qwen2-VL / Qwen3-VL `rope_deltas`).
    mrope_delta: i64,
};

/// Builds deterministic planar `[3, total_tokens]u32` 3D MRoPE coordinates
/// `[t_pos, h_pos, w_pos]` with zero heap allocation:
/// - Text prefix `[0..prefix_text_tokens)`: `t = h = w = pos`.
/// - Visual grid tokens (`grid_t * grid_h * grid_w`, row-major `(dt, dh, dw)`):
///   `t = base + dt * temporal_stride`, `h = base + dh`, `w = base + dw`.
/// - Text suffix `[0..suffix_text_tokens)`:
///   `t = h = w = max_vis_pos + 1 + offset`.
///
/// Output layout in `out_planar[0 .. 3 * total_tokens]`:
/// - `out_planar[0 * total_tokens + i] = t_pos[i]`
/// - `out_planar[1 * total_tokens + i] = h_pos[i]`
/// - `out_planar[2 * total_tokens + i] = w_pos[i]`
pub fn buildMultimodalMropePositions(
    prefix_text_tokens: u32,
    grid: VisionGridSpec,
    suffix_text_tokens: u32,
    out_planar: []u32,
) MropeError!MropeBuildResult {
    const segments = [_]MropeSegment{
        .{ .text = prefix_text_tokens },
        .{ .vision = grid },
        .{ .text = suffix_text_tokens },
    };
    return buildMultimodalMropePositionsMulti(&segments, out_planar);
}

/// General multi-span 3D MRoPE position builder supporting arbitrary interleaved
/// text, image, and video segments into planar `[3, total_tokens]u32`.
pub fn buildMultimodalMropePositionsMulti(
    segments: []const MropeSegment,
    out_planar: []u32,
) MropeError!MropeBuildResult {
    var total_tokens: u32 = 0;
    for (segments) |seg| {
        switch (seg) {
            .text => |count| {
                total_tokens = std.math.add(u32, total_tokens, count) catch return error.Overflow;
            },
            .vision => |grid| {
                try grid.validate();
                const vis_tokens = try grid.numTokens();
                total_tokens = std.math.add(u32, total_tokens, vis_tokens) catch return error.Overflow;
            },
        }
    }
    if (total_tokens == 0) return error.EmptySequence;

    const required = std.math.mul(usize, 3, total_tokens) catch return error.Overflow;
    if (out_planar.len < required) return error.BufferTooSmall;

    const n: usize = total_tokens;
    const t_row = out_planar[0..n];
    const h_row = out_planar[n .. 2 * n];
    const w_row = out_planar[2 * n .. 3 * n];

    var cursor: usize = 0;
    var next_base: u32 = 0;

    for (segments) |seg| {
        switch (seg) {
            .text => |count| {
                for (0..count) |i| {
                    const pos = std.math.add(u32, next_base, @intCast(i)) catch return error.Overflow;
                    t_row[cursor] = pos;
                    h_row[cursor] = pos;
                    w_row[cursor] = pos;
                    cursor += 1;
                }
                next_base = std.math.add(u32, next_base, count) catch return error.Overflow;
            },
            .vision => |grid| {
                const base = next_base;
                for (0..grid.grid_t) |dt_usize| {
                    const dt: u32 = @intCast(dt_usize);
                    const t_off = std.math.mul(u32, dt, grid.temporal_stride) catch return error.Overflow;
                    const t_pos = std.math.add(u32, base, t_off) catch return error.Overflow;
                    for (0..grid.grid_h) |dh_usize| {
                        const dh: u32 = @intCast(dh_usize);
                        const h_pos = std.math.add(u32, base, dh) catch return error.Overflow;
                        for (0..grid.grid_w) |dw_usize| {
                            const dw: u32 = @intCast(dw_usize);
                            const w_pos = std.math.add(u32, base, dw) catch return error.Overflow;
                            t_row[cursor] = t_pos;
                            h_row[cursor] = h_pos;
                            w_row[cursor] = w_pos;
                            cursor += 1;
                        }
                    }
                }
                const max_delta = try grid.maxDelta();
                const max_vis_pos = std.math.add(u32, base, max_delta) catch return error.Overflow;
                next_base = std.math.add(u32, max_vis_pos, 1) catch return error.Overflow;
            },
        }
    }

    std.debug.assert(cursor == n);
    const delta: i64 = @as(i64, next_base) - @as(i64, total_tokens);
    return .{
        .total_tokens = total_tokens,
        .next_text_pos = next_base,
        .mrope_delta = delta,
    };
}

/// Packs planar `[3, total_tokens]u32` MRoPE coordinates into interleaved
/// `[total_tokens]hybrid.Pos3D` (`[total_tokens, 3]u32`) for direct submission to
/// M6 `G9` `HybridPipelines.encodeQsaTreeRope` (`MropeSection.QWEN38_DEFAULT = [11, 11, 10]`).
pub fn packMropePlanarToPos3D(
    planar: []const u32,
    total_tokens: usize,
    out_pos3d: []hybrid.Pos3D,
) MropeError!void {
    if (total_tokens == 0) return error.EmptySequence;
    const req_planar = std.math.mul(usize, 3, total_tokens) catch return error.Overflow;
    if (planar.len < req_planar or out_pos3d.len < total_tokens) return error.BufferTooSmall;
    for (0..total_tokens) |i| {
        out_pos3d[i] = .{
            .t = planar[i],
            .h = planar[total_tokens + i],
            .w = planar[2 * total_tokens + i],
        };
    }
}

// ===========================================================================
// Preallocated Wired `VisionIngress` Workspace (`0` hot-path heap allocs)
// ===========================================================================

pub const VisionIngressConfig = struct {
    max_tokens: u32,
    max_patch_dim: u32,
    max_hidden_size: u32,
    max_mrope_tokens: u32,
};

/// Zero-allocation multimodal vision/video ingress workspace backed by one
/// `16 KiB`-aligned `WiredArena` + `MTLBuffer` and startup-compiled pipelines.
pub const VisionIngress = struct {
    config: VisionIngressConfig,
    arena: arena_mod.WiredArena,
    gpu_buf: metal.Buffer,
    pipelines: VisionPipelines,
    patch_out_raw: []u8,
    embed_out_raw: []u8,
    mrope_planar_raw: []u8,
    mrope_pos3d_raw: []u8,

    pub const InitError = Error || arena_mod.WiredArena.InitError || arena_mod.WiredArena.AllocError || metal.WrapError;

    pub fn requiredArenaBytes(cfg: VisionIngressConfig) Error!usize {
        if (cfg.max_tokens == 0 or cfg.max_patch_dim == 0 or
            cfg.max_hidden_size == 0 or cfg.max_mrope_tokens == 0)
        {
            return error.InvalidGeometry;
        }
        const patch_elems = std.math.mul(usize, cfg.max_tokens, cfg.max_patch_dim) catch return error.Overflow;
        const patch_bytes = std.math.mul(usize, patch_elems, 2) catch return error.Overflow;
        const embed_elems = std.math.mul(usize, cfg.max_tokens, cfg.max_hidden_size) catch return error.Overflow;
        const embed_bytes = std.math.mul(usize, embed_elems, 2) catch return error.Overflow;
        const planar_elems = std.math.mul(usize, 3, cfg.max_mrope_tokens) catch return error.Overflow;
        const planar_bytes = std.math.mul(usize, planar_elems, @sizeOf(u32)) catch return error.Overflow;
        const pos3d_bytes = std.math.mul(usize, cfg.max_mrope_tokens, @sizeOf(hybrid.Pos3D)) catch return error.Overflow;

        var total: usize = 0;
        const chunks = [_]usize{ patch_bytes, embed_bytes, planar_bytes, pos3d_bytes };
        for (chunks) |c_len| {
            const aligned = (std.math.add(usize, c_len, 63) catch return error.Overflow) & ~@as(usize, 63);
            total = std.math.add(usize, total, aligned) catch return error.Overflow;
        }
        return page.alignUp(total) catch return error.Overflow;
    }

    pub fn init(
        device: metal.Device,
        library: metal.Library,
        budget: budget_mod.Budget,
        cfg: VisionIngressConfig,
    ) InitError!VisionIngress {
        const total_bytes = try requiredArenaBytes(cfg);
        var arena = try arena_mod.WiredArena.init(total_bytes, budget);
        errdefer arena.deinit();

        const patch_bytes = @as(usize, cfg.max_tokens) * @as(usize, cfg.max_patch_dim) * 2;
        const embed_bytes = @as(usize, cfg.max_tokens) * @as(usize, cfg.max_hidden_size) * 2;
        const planar_bytes = @as(usize, cfg.max_mrope_tokens) * 3 * @sizeOf(u32);
        const pos3d_bytes = @as(usize, cfg.max_mrope_tokens) * @sizeOf(hybrid.Pos3D);

        const patch_out_raw = try arena.allocAligned(patch_bytes, 64);
        const embed_out_raw = try arena.allocAligned(embed_bytes, 64);
        const mrope_planar_raw = try arena.allocAligned(planar_bytes, 64);
        const mrope_pos3d_raw = try arena.allocAligned(pos3d_bytes, 64);

        const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
        errdefer gpu_buf.close();

        const pipelines = try VisionPipelines.init(device, library);
        errdefer pipelines.close();

        return .{
            .config = cfg,
            .arena = arena,
            .gpu_buf = gpu_buf,
            .pipelines = pipelines,
            .patch_out_raw = patch_out_raw,
            .embed_out_raw = embed_out_raw,
            .mrope_planar_raw = mrope_planar_raw,
            .mrope_pos3d_raw = mrope_pos3d_raw,
        };
    }

    pub fn deinit(self: *VisionIngress) void {
        self.pipelines.close();
        self.gpu_buf.close();
        self.arena.deinit();
    }

    pub fn buffer(self: *const VisionIngress) metal.Buffer {
        return self.gpu_buf;
    }

    pub fn patchOutBinding(self: *const VisionIngress) metal.BufferBinding {
        return .{ .buffer = self.gpu_buf, .offset = self.arena.offsetOf(self.patch_out_raw.ptr) };
    }

    pub fn embedOutBinding(self: *const VisionIngress) metal.BufferBinding {
        return .{ .buffer = self.gpu_buf, .offset = self.arena.offsetOf(self.embed_out_raw.ptr) };
    }

    pub fn mropePos3dBinding(self: *const VisionIngress) metal.BufferBinding {
        return .{ .buffer = self.gpu_buf, .offset = self.arena.offsetOf(self.mrope_pos3d_raw.ptr) };
    }

    pub fn patchOutRawU16(self: *const VisionIngress) []u16 {
        return @alignCast(std.mem.bytesAsSlice(u16, self.patch_out_raw));
    }

    pub fn embedOutRawU16(self: *const VisionIngress) []u16 {
        return @alignCast(std.mem.bytesAsSlice(u16, self.embed_out_raw));
    }

    pub fn mropePlanar(self: *const VisionIngress) []u32 {
        return @alignCast(std.mem.bytesAsSlice(u32, self.mrope_planar_raw));
    }

    pub fn mropePos3d(self: *const VisionIngress) []hybrid.Pos3D {
        return @alignCast(std.mem.bytesAsSlice(hybrid.Pos3D, self.mrope_pos3d_raw));
    }

    /// Populates both planar `[3, total_tokens]u32` and GPU-readable wired
    /// `[total_tokens]hybrid.Pos3D` buffers with zero heap allocation.
    pub fn buildMropePositions(
        self: *VisionIngress,
        prefix_text_tokens: u32,
        grid: VisionGridSpec,
        suffix_text_tokens: u32,
    ) MropeError!MropeBuildResult {
        const res = try buildMultimodalMropePositions(
            prefix_text_tokens,
            grid,
            suffix_text_tokens,
            self.mropePlanar(),
        );
        try packMropePlanarToPos3D(self.mropePlanar(), res.total_tokens, self.mropePos3d());
        return res;
    }

    pub fn buildMropePositionsMulti(
        self: *VisionIngress,
        segments: []const MropeSegment,
    ) MropeError!MropeBuildResult {
        const res = try buildMultimodalMropePositionsMulti(segments, self.mropePlanar());
        try packMropePlanarToPos3D(self.mropePlanar(), res.total_tokens, self.mropePos3d());
        return res;
    }

    pub fn encodePatchifyBgra8(
        self: *const VisionIngress,
        frame: *metal.Compute.Frame,
        frame0: metal.VisionFrame,
        frame1: ?metal.VisionFrame,
        geometry: PatchGeometry,
        norm: NormParams,
        out_dtype: quant.FloatDType,
        token_offset: u32,
        insert_barrier_before: bool,
    ) EncodeError!void {
        try self.checkPatchCapacity(geometry, token_offset);
        return self.pipelines.encodePatchifyBgra8(frame, .{
            .frame0 = frame0,
            .frame1 = frame1,
            .out = self.patchOutBinding(),
            .out_dtype = out_dtype,
            .geometry = geometry,
            .norm = norm,
            .token_offset = token_offset,
        }, insert_barrier_before);
    }

    pub fn encodePatchifyNv12(
        self: *const VisionIngress,
        frame: *metal.Compute.Frame,
        frame0: metal.VisionFrame,
        frame1: ?metal.VisionFrame,
        geometry: PatchGeometry,
        is_full_range: bool,
        norm: NormParams,
        out_dtype: quant.FloatDType,
        token_offset: u32,
        insert_barrier_before: bool,
    ) EncodeError!void {
        try self.checkPatchCapacity(geometry, token_offset);
        return self.pipelines.encodePatchifyNv12(frame, .{
            .frame0 = frame0,
            .frame1 = frame1,
            .out = self.patchOutBinding(),
            .out_dtype = out_dtype,
            .geometry = geometry,
            .is_full_range = is_full_range,
            .norm = norm,
            .token_offset = token_offset,
        }, insert_barrier_before);
    }

    pub fn encodePatchifyEmbedBgra8(
        self: *const VisionIngress,
        frame: *metal.Compute.Frame,
        frame0: metal.VisionFrame,
        frame1: ?metal.VisionFrame,
        weight: metal.BufferBinding,
        bias: ?metal.BufferBinding,
        weight_dtype: quant.FloatDType,
        out_dtype: quant.FloatDType,
        hidden_size: u32,
        geometry: PatchGeometry,
        norm: NormParams,
        token_offset: u32,
        insert_barrier_before: bool,
    ) EncodeError!void {
        try self.checkEmbedCapacity(geometry, hidden_size, token_offset);
        return self.pipelines.encodePatchifyEmbedBgra8(frame, .{
            .frame0 = frame0,
            .frame1 = frame1,
            .weight = weight,
            .bias = bias,
            .out = self.embedOutBinding(),
            .weight_dtype = weight_dtype,
            .out_dtype = out_dtype,
            .hidden_size = hidden_size,
            .geometry = geometry,
            .norm = norm,
            .token_offset = token_offset,
        }, insert_barrier_before);
    }

    pub fn encodePatchifyEmbedNv12(
        self: *const VisionIngress,
        frame: *metal.Compute.Frame,
        frame0: metal.VisionFrame,
        frame1: ?metal.VisionFrame,
        weight: metal.BufferBinding,
        bias: ?metal.BufferBinding,
        weight_dtype: quant.FloatDType,
        out_dtype: quant.FloatDType,
        hidden_size: u32,
        geometry: PatchGeometry,
        is_full_range: bool,
        norm: NormParams,
        token_offset: u32,
        insert_barrier_before: bool,
    ) EncodeError!void {
        try self.checkEmbedCapacity(geometry, hidden_size, token_offset);
        return self.pipelines.encodePatchifyEmbedNv12(frame, .{
            .frame0 = frame0,
            .frame1 = frame1,
            .weight = weight,
            .bias = bias,
            .out = self.embedOutBinding(),
            .weight_dtype = weight_dtype,
            .out_dtype = out_dtype,
            .hidden_size = hidden_size,
            .geometry = geometry,
            .is_full_range = is_full_range,
            .norm = norm,
            .token_offset = token_offset,
        }, insert_barrier_before);
    }

    fn checkPatchCapacity(self: *const VisionIngress, geometry: PatchGeometry, token_offset: u32) Error!void {
        try geometry.validate();
        const num_tokens = try geometry.numTokens();
        const end_tokens = std.math.add(u32, token_offset, num_tokens) catch return error.Overflow;
        const dim = try geometry.tokenInDim();
        if (end_tokens > self.config.max_tokens or dim > self.config.max_patch_dim) {
            return error.OutputBufferTooSmall;
        }
    }

    fn checkEmbedCapacity(
        self: *const VisionIngress,
        geometry: PatchGeometry,
        hidden_size: u32,
        token_offset: u32,
    ) Error!void {
        try geometry.validate();
        if (hidden_size == 0) return error.InvalidHiddenSize;
        const num_tokens = try geometry.numTokens();
        const end_tokens = std.math.add(u32, token_offset, num_tokens) catch return error.Overflow;
        if (end_tokens > self.config.max_tokens or hidden_size > self.config.max_hidden_size) {
            return error.OutputBufferTooSmall;
        }
    }
};

// ===========================================================================
// CPU `fp32` Reference Implementations (for exact GPU numerical parity tests)
// ===========================================================================

pub fn srgbToLinearChanCpuRef(c: f32) f32 {
    const x = std.math.clamp(c, 0.0, 1.0);
    if (x <= 0.04045) return x / 12.92;
    return std.math.pow(f32, (x + 0.055) / 1.055, 2.4);
}

pub fn bt709Nv12ToRgbCpuRef(y_u8: u8, cb_u8: u8, cr_u8: u8, is_full_range: bool) [3]f32 {
    const y_raw = @as(f32, @floatFromInt(y_u8)) / 255.0;
    const cb_raw = @as(f32, @floatFromInt(cb_u8)) / 255.0;
    const cr_raw = @as(f32, @floatFromInt(cr_u8)) / 255.0;

    var yf = y_raw;
    var cb = cb_raw - (128.0 / 255.0);
    var cr = cr_raw - (128.0 / 255.0);
    if (!is_full_range) {
        yf = (y_raw - (16.0 / 255.0)) * (255.0 / 219.0);
        cb *= (255.0 / 224.0);
        cr *= (255.0 / 224.0);
    }
    return .{
        std.math.clamp(yf + 1.5748 * cr, 0.0, 1.0),
        std.math.clamp(yf - 0.187324 * cb - 0.468124 * cr, 0.0, 1.0),
        std.math.clamp(yf + 1.8556 * cb, 0.0, 1.0),
    };
}

fn normalizeRgbCpuRef(rgb_in: [3]f32, norm: NormParams) [3]f32 {
    var rgb = rgb_in;
    if (norm.apply_srgb_to_linear) {
        rgb[0] = srgbToLinearChanCpuRef(rgb[0]);
        rgb[1] = srgbToLinearChanCpuRef(rgb[1]);
        rgb[2] = srgbToLinearChanCpuRef(rgb[2]);
    }
    return .{
        (rgb[0] - norm.mean[0]) * norm.inv_std[0],
        (rgb[1] - norm.mean[1]) * norm.inv_std[1],
        (rgb[2] - norm.mean[2]) * norm.inv_std[2],
    };
}

pub fn patchifyBgra8CpuRef(
    frame0_bytes: []const u8,
    frame0_bpr: usize,
    frame1_bytes: ?[]const u8,
    frame1_bpr: usize,
    geometry: PatchGeometry,
    norm: NormParams,
    out_f32: []f32,
) Error!void {
    try geometry.validate();
    const num_tokens = try geometry.numTokens();
    const token_dim = try geometry.tokenInDim();
    const patch_dim = try geometry.rawPatchDim();
    const req = std.math.mul(usize, num_tokens, token_dim) catch return error.Overflow;
    if (out_f32.len < req) return error.OutputBufferTooSmall;

    const merged_h = geometry.mergedH();
    const merged_w = geometry.mergedW();
    const p = geometry.patch_size;
    const tp = geometry.t_patch;
    const sm = geometry.spatial_merge;
    const tube_pixels = tp * p * p;

    for (0..merged_h) |mh| {
        for (0..merged_w) |mw| {
            const tok = mh * merged_w + mw;
            const tok_base = tok * token_dim;
            for (0..sm) |sub_y| {
                for (0..sm) |sub_x| {
                    const sub_idx = sub_y * sm + sub_x;
                    const ph = mh * sm + sub_y;
                    const pw = mw * sm + sub_x;
                    for (0..tp) |dt| {
                        const src_bytes = if (dt == 0) frame0_bytes else (frame1_bytes orelse frame0_bytes);
                        const src_bpr = if (dt == 0) frame0_bpr else frame1_bpr;
                        for (0..p) |dy| {
                            const py = ph * p + dy;
                            for (0..p) |dx| {
                                const px = pw * p + dx;
                                const off = py * src_bpr + px * 4;
                                const b_u8 = src_bytes[off + 0];
                                const g_u8 = src_bytes[off + 1];
                                const r_u8 = src_bytes[off + 2];
                                const rgb_norm = normalizeRgbCpuRef(.{
                                    @as(f32, @floatFromInt(r_u8)) / 255.0,
                                    @as(f32, @floatFromInt(g_u8)) / 255.0,
                                    @as(f32, @floatFromInt(b_u8)) / 255.0,
                                }, norm);
                                const rem = (dt * p + dy) * p + dx;
                                const k_base = tok_base + sub_idx * patch_dim + rem;
                                out_f32[k_base] = rgb_norm[0];
                                out_f32[k_base + tube_pixels] = rgb_norm[1];
                                out_f32[k_base + 2 * tube_pixels] = rgb_norm[2];
                            }
                        }
                    }
                }
            }
        }
    }
}

pub fn patchifyNv12CpuRef(
    y0_bytes: []const u8,
    y0_bpr: usize,
    uv0_bytes: []const u8,
    uv0_bpr: usize,
    y1_bytes: ?[]const u8,
    y1_bpr: usize,
    uv1_bytes: ?[]const u8,
    uv1_bpr: usize,
    geometry: PatchGeometry,
    is_full_range: bool,
    norm: NormParams,
    out_f32: []f32,
) Error!void {
    try geometry.validate();
    const num_tokens = try geometry.numTokens();
    const token_dim = try geometry.tokenInDim();
    const patch_dim = try geometry.rawPatchDim();
    const req = std.math.mul(usize, num_tokens, token_dim) catch return error.Overflow;
    if (out_f32.len < req) return error.OutputBufferTooSmall;

    const merged_h = geometry.mergedH();
    const merged_w = geometry.mergedW();
    const p = geometry.patch_size;
    const tp = geometry.t_patch;
    const sm = geometry.spatial_merge;
    const tube_pixels = tp * p * p;

    for (0..merged_h) |mh| {
        for (0..merged_w) |mw| {
            const tok = mh * merged_w + mw;
            const tok_base = tok * token_dim;
            for (0..sm) |sub_y| {
                for (0..sm) |sub_x| {
                    const sub_idx = sub_y * sm + sub_x;
                    const ph = mh * sm + sub_y;
                    const pw = mw * sm + sub_x;
                    for (0..tp) |dt| {
                        const yb = if (dt == 0) y0_bytes else (y1_bytes orelse y0_bytes);
                        const ybpr = if (dt == 0) y0_bpr else y1_bpr;
                        const uvb = if (dt == 0) uv0_bytes else (uv1_bytes orelse uv0_bytes);
                        const uvbpr = if (dt == 0) uv0_bpr else uv1_bpr;
                        for (0..p) |dy| {
                            const py = ph * p + dy;
                            for (0..p) |dx| {
                                const px = pw * p + dx;
                                const y_u8 = yb[py * ybpr + px];
                                const uv_off = (py >> 1) * uvbpr + (px >> 1) * 2;
                                const cb_u8 = uvb[uv_off + 0];
                                const cr_u8 = uvb[uv_off + 1];
                                const rgb = bt709Nv12ToRgbCpuRef(y_u8, cb_u8, cr_u8, is_full_range);
                                const rgb_norm = normalizeRgbCpuRef(rgb, norm);
                                const rem = (dt * p + dy) * p + dx;
                                const k_base = tok_base + sub_idx * patch_dim + rem;
                                out_f32[k_base] = rgb_norm[0];
                                out_f32[k_base + tube_pixels] = rgb_norm[1];
                                out_f32[k_base + 2 * tube_pixels] = rgb_norm[2];
                            }
                        }
                    }
                }
            }
        }
    }
}

pub fn linearEmbedCpuRef(
    patches_f32: []const f32,
    weight_raw: []const u16,
    bias_raw: ?[]const u16,
    weight_dtype: quant.FloatDType,
    num_tokens: usize,
    token_in_dim: usize,
    hidden_size: usize,
    out_f32: []f32,
) Error!void {
    if (patches_f32.len < num_tokens * token_in_dim) return error.OutputBufferTooSmall;
    if (weight_raw.len < hidden_size * token_in_dim) return error.WeightBufferTooSmall;
    if (bias_raw) |b| {
        if (b.len < hidden_size) return error.BiasBufferTooSmall;
    }
    if (out_f32.len < num_tokens * hidden_size) return error.OutputBufferTooSmall;

    for (0..num_tokens) |tok| {
        const x = patches_f32[tok * token_in_dim ..][0..token_in_dim];
        for (0..hidden_size) |h| {
            const w = weight_raw[h * token_in_dim ..][0..token_in_dim];
            var acc: f64 = if (bias_raw) |b|
                @as(f64, quant.floatBitsToF32(weight_dtype, b[h]))
            else
                0.0;
            for (0..token_in_dim) |k| {
                acc += @as(f64, x[k]) * @as(f64, quant.floatBitsToF32(weight_dtype, w[k]));
            }
            out_f32[tok * hidden_size + h] = @floatCast(acc);
        }
    }
}

// ===========================================================================
// Tests (Zero-copy IOSurface, ImageIO, Hardware VideoToolbox, Metal 4 Parity,
//        3D MRoPE + M6 G9 Integration)
// ===========================================================================

fn testBudget(bytes: usize) budget_mod.Budget {
    const cap = @max(bytes, 64 * 1024 * 1024);
    return budget_mod.Budget.compute(.{
        .memory_bytes = cap * 4,
        .user_wire_limit = cap * 2,
        .global_user_wire_limit = cap * 2,
        .gpu_working_set_bytes = cap * 2,
        .configured_cap = null,
        .headroom_bytes = 4 * 1024 * 1024,
    });
}

fn fillSyntheticBgra8(frame: metal.VisionFrame, width: u32, height: u32, seed: u32) !void {
    var g = try frame.lockPlane(0, false);
    errdefer g.release();
    for (0..height) |y| {
        const row = g.rowBytes(@intCast(y));
        for (0..width) |x| {
            const xu: u32 = @intCast(x);
            const yu: u32 = @intCast(y);
            const r: u8 = @intCast((xu * 17 + yu * 7 + seed * 31) % 256);
            const gg: u8 = @intCast((xu * 5 + yu * 23 + seed * 47 + 40) % 256);
            const b: u8 = @intCast((xu * 11 + yu * 13 + seed * 59 + 90) % 256);
            row[x * 4 + 0] = b;
            row[x * 4 + 1] = gg;
            row[x * 4 + 2] = r;
            row[x * 4 + 3] = 255;
        }
    }
    try g.unlock();
}

fn fillSyntheticNv12(frame: metal.VisionFrame, width: u32, height: u32, seed: u32) !void {
    {
        var yg = try frame.lockPlane(0, false);
        errdefer yg.release();
        for (0..height) |y| {
            const row = yg.rowBytes(@intCast(y));
            for (0..width) |x| {
                const xu: u32 = @intCast(x);
                const yu: u32 = @intCast(y);
                row[x] = @intCast(20 + ((xu * 13 + yu * 9 + seed * 29) % 215));
            }
        }
        try yg.unlock();
    }
    {
        var uvg = try frame.lockPlane(1, false);
        errdefer uvg.release();
        const cw = width / 2;
        const ch = height / 2;
        for (0..ch) |y| {
            const row = uvg.rowBytes(@intCast(y));
            for (0..cw) |x| {
                const xu: u32 = @intCast(x);
                const yu: u32 = @intCast(y);
                row[x * 2 + 0] = @intCast(48 + ((xu * 19 + yu * 11 + seed * 37) % 160));
                row[x * 2 + 1] = @intCast(48 + ((xu * 7 + yu * 23 + seed * 41) % 160));
            }
        }
        try uvg.unlock();
    }
}

fn synthWeight(idx: usize, scale: f32) f32 {
    const v: i32 = @as(i32, @intCast((idx * 37 + 13) % 61)) - 30;
    return @as(f32, @floatFromInt(v)) * scale;
}

test "G14 zero-copy VisionFrame (BGRA8, NV12, F16) 16 KiB alignment, IOSurface identity and CoreML ANE probe" {
    const device = try metal.Device.open();
    defer device.close();

    // 1. BGRA8 frame
    const bgra = try metal.VisionFrame.init(device, .bgra8, 64, 48);
    defer bgra.close();
    const bi = try bgra.info();
    try std.testing.expectEqual(metal.VisionFormat.bgra8, bi.format());
    try std.testing.expectEqual(@as(u32, 64), bi.width());
    try std.testing.expectEqual(@as(u32, 48), bi.height());
    try std.testing.expectEqual(@as(u32, 1), bi.planeCount());
    try std.testing.expect(bi.zeroCopyVerified());
    try std.testing.expect(bi.isPageAligned());
    try std.testing.expect(bi.iosurfaceId() != 0);
    try std.testing.expectEqual(@as(u64, 0), bi.baseAddress() % page.APPLE_PAGE_SIZE);
    try std.testing.expectEqual(@as(u64, 0), bi.surfaceAllocSize() % page.APPLE_PAGE_SIZE);
    try std.testing.expect(bi.planeResourceId(0) != 0);
    try std.testing.expect(!bi.isVtDecoded());

    const b_probe = try bgra.probeAne();
    try std.testing.expect(b_probe.pixelBufferFeatureOk());
    try std.testing.expect(b_probe.iosurfaceIdentityMatched());
    try std.testing.expect(!b_probe.multiArrayZeroCopyOk());
    try std.testing.expectError(error.Unsupported, bgra.toAneTensor(&.{ 48, 64 }));

    // 2. Bi-planar NV12 frame
    const nv12 = try metal.VisionFrame.init(device, .nv12, 64, 48);
    defer nv12.close();
    const ni = try nv12.info();
    try std.testing.expectEqual(metal.VisionFormat.nv12, ni.format());
    try std.testing.expectEqual(@as(u32, 2), ni.planeCount());
    try std.testing.expect(ni.isFullRange());
    try std.testing.expect(ni.zeroCopyVerified());
    try std.testing.expect(ni.isPageAligned());
    try std.testing.expect(ni.iosurfaceId() != 0);
    try std.testing.expectEqual(@as(u64, 0), ni.baseAddress() % page.APPLE_PAGE_SIZE);
    try std.testing.expectEqual(@as(u64, 0), ni.surfaceAllocSize() % page.APPLE_PAGE_SIZE);
    try std.testing.expectEqual(@as(u32, 64), ni.planeWidth(0));
    try std.testing.expectEqual(@as(u32, 48), ni.planeHeight(0));
    try std.testing.expectEqual(@as(u32, 32), ni.planeWidth(1));
    try std.testing.expectEqual(@as(u32, 24), ni.planeHeight(1));
    try std.testing.expect(ni.planeResourceId(0) != 0);
    try std.testing.expect(ni.planeResourceId(1) != 0);
    try std.testing.expect(ni.planeResourceId(0) != ni.planeResourceId(1));

    const n_probe = try nv12.probeAne();
    try std.testing.expect(n_probe.pixelBufferFeatureOk());
    try std.testing.expect(n_probe.iosurfaceIdentityMatched());

    // 3. OneComponent16Half F16 frame -> zero-copy CoreML MLMultiArray
    const f16_frame = try metal.VisionFrame.init(device, .f16, 64, 32);
    defer f16_frame.close();
    const fi = try f16_frame.info();
    try std.testing.expectEqual(metal.VisionFormat.f16, fi.format());
    try std.testing.expect(fi.zeroCopyVerified());
    try std.testing.expect(fi.isPageAligned());
    try std.testing.expectEqual(@as(u32, 128), fi.planeBytesPerRow(0));
    try std.testing.expect(fi.planeResourceId(0) != 0);

    const f_probe = try f16_frame.probeAne();
    try std.testing.expect(f_probe.pixelBufferFeatureOk());
    try std.testing.expect(f_probe.multiArrayZeroCopyOk());
    try std.testing.expect(f_probe.iosurfaceIdentityMatched());

    const ane_tensor = try f16_frame.toAneTensor(&.{ 1, 32, 64 });
    defer ane_tensor.close();
}

test "G14 ImageIO PNG roundtrip loads directly into 16 KiB-aligned zero-copy IOSurface" {
    const device = try metal.Device.open();
    defer device.close();

    const w: u32 = 32;
    const h: u32 = 32;
    const src = try metal.VisionFrame.init(device, .bgra8, w, h);
    defer src.close();

    // Distinct corner colors to verify top-left origin is NOT vertically inverted:
    // top-left (0, 0) = pure red (R=240, G=16, B=24),
    // bottom-right (w-1, h-1) = pure blue (R=20, G=32, B=236).
    {
        var g = try src.lockPlane(0, false);
        errdefer g.release();
        for (0..h) |y| {
            const row = g.rowBytes(@intCast(y));
            for (0..w) |x| {
                row[x * 4 + 0] = @intCast((x * 7 + y * 5) % 256);
                row[x * 4 + 1] = @intCast((x * 3 + y * 11) % 256);
                row[x * 4 + 2] = @intCast((x * 9 + y * 2) % 256);
                row[x * 4 + 3] = 255;
            }
        }
        const top = g.rowBytes(0);
        top[0] = 24;
        top[1] = 16;
        top[2] = 240;
        top[3] = 255;
        const bot = g.rowBytes(h - 1);
        bot[(w - 1) * 4 + 0] = 236;
        bot[(w - 1) * 4 + 1] = 32;
        bot[(w - 1) * 4 + 2] = 20;
        bot[(w - 1) * 4 + 3] = 255;
        try g.unlock();
    }

    const tmp_dir: []const u8 = if (std.c.getenv("TMPDIR")) |p| std.mem.span(p) else "/tmp";
    var path_buf: [512]u8 = undefined;
    const raw_path = try std.fmt.bufPrint(&path_buf, "{s}/aminfer_g14_png_roundtrip.png", .{
        std.mem.trimEnd(u8, tmp_dir, "/"),
    });
    path_buf[raw_path.len] = 0;
    const path: [:0]const u8 = path_buf[0..raw_path.len :0];
    defer _ = std.c.unlink(path.ptr);

    try src.writePng(path);

    const loaded = try metal.VisionFrame.loadImage(device, path);
    defer loaded.close();

    const li = try loaded.info();
    try std.testing.expectEqual(metal.VisionFormat.bgra8, li.format());
    try std.testing.expectEqual(w, li.width());
    try std.testing.expectEqual(h, li.height());
    try std.testing.expect(li.zeroCopyVerified());
    try std.testing.expect(li.isPageAligned());
    try std.testing.expectEqual(@as(u64, 0), li.surfaceAllocSize() % page.APPLE_PAGE_SIZE);
    try std.testing.expect(li.planeResourceId(0) != 0);

    var lg = try loaded.lockPlane(0, true);
    errdefer lg.release();
    const ltop = lg.rowBytes(0);
    try std.testing.expectEqual(@as(u8, 24), ltop[0]);
    try std.testing.expectEqual(@as(u8, 16), ltop[1]);
    try std.testing.expectEqual(@as(u8, 240), ltop[2]);
    try std.testing.expectEqual(@as(u8, 255), ltop[3]);

    const lbot = lg.rowBytes(h - 1);
    try std.testing.expectEqual(@as(u8, 236), lbot[(w - 1) * 4 + 0]);
    try std.testing.expectEqual(@as(u8, 32), lbot[(w - 1) * 4 + 1]);
    try std.testing.expectEqual(@as(u8, 20), lbot[(w - 1) * 4 + 2]);
    try std.testing.expectEqual(@as(u8, 255), lbot[(w - 1) * 4 + 3]);
    try lg.unlock();
}

test "G14 real hardware VideoToolbox H.264 & HEVC decode -> zero-copy IOSurface -> MTLTexture + Metal 4 patchify" {
    const device = try metal.Device.open();
    defer device.close();
    const lib = try metal.Library.openEmbedded(device);
    defer lib.close();
    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();

    const width: u32 = 112;
    const height: u32 = 112;
    const geom: PatchGeometry = .{
        .width = width,
        .height = height,
        .patch_size = 14,
        .t_patch = 2,
        .spatial_merge = 2,
    };
    const num_tokens = try geom.numTokens(); // (112/14/2)^2 = 16
    const token_in_dim = try geom.tokenInDim(); // 4 * 3 * 2 * 14 * 14 = 4704
    const hidden_size: u32 = 32;

    var ingress = try VisionIngress.init(device, lib, testBudget(16 * 1024 * 1024), .{
        .max_tokens = num_tokens,
        .max_patch_dim = token_in_dim,
        .max_hidden_size = hidden_size,
        .max_mrope_tokens = 64,
    });
    defer ingress.deinit();
    try compute.makeResident(&.{ingress.buffer()});

    const codecs = [_]metal.VtCodec{ .h264, .hevc };
    const out_fmts = [_]metal.VisionFormat{ .nv12, .bgra8 };

    for (codecs) |codec| {
        const clip = try metal.VtClip.encodeTestClip(codec, width, height, 4, 30);
        defer clip.close();
        const ci = try clip.info();
        try std.testing.expectEqual(codec, ci.codec());
        try std.testing.expectEqual(width, ci.width());
        try std.testing.expectEqual(height, ci.height());
        try std.testing.expectEqual(@as(u32, 4), ci.frameCount());
        try std.testing.expect(ci.totalCompressedBytes() > 0);
        try std.testing.expect(ci.usingHardwareEncoder());

        for (out_fmts) |out_fmt| {
            const dec = try metal.VtDecodeSession.init(device, clip, out_fmt);
            defer dec.close();
            const di = try dec.info();
            try std.testing.expectEqual(codec, di.codec());
            try std.testing.expectEqual(out_fmt, di.outputFormat());
            try std.testing.expect(di.usingHardwareDecoder());

            const vf0 = try dec.decodeFrame(clip, 0);
            defer vf0.close();
            const vf1 = try dec.decodeFrame(clip, 1);
            defer vf1.close();

            const info0 = try vf0.info();
            const info1 = try vf1.info();
            try std.testing.expectEqual(out_fmt, info0.format());
            try std.testing.expectEqual(out_fmt, info1.format());
            try std.testing.expect(info0.isVtDecoded());
            try std.testing.expect(info1.isVtDecoded());
            // Hard zero-copy proof: Metal texture.iosurface == CVPixelBufferGetIOSurface(pb).
            try std.testing.expect(info0.zeroCopyVerified());
            try std.testing.expect(info1.zeroCopyVerified());
            try std.testing.expect(info0.iosurfaceId() != 0);
            try std.testing.expect(info1.iosurfaceId() != 0);
            try std.testing.expectEqual(@as(u64, 0), info0.baseAddress() % page.APPLE_PAGE_SIZE);
            try std.testing.expectEqual(@as(u64, 0), info1.baseAddress() % page.APPLE_PAGE_SIZE);
            try std.testing.expect(info0.surfaceAllocSize() > 0);
            try std.testing.expect(info0.planeResourceId(0) != 0);
            try std.testing.expect(info1.ptsValue() > info0.ptsValue());

            const p0 = try vf0.probeAne();
            try std.testing.expect(p0.pixelBufferFeatureOk());
            try std.testing.expect(p0.iosurfaceIdentityMatched());

            // Dispatch Metal 4 patchify directly over the VideoToolbox-decoded IOSurface textures.
            try compute.makeVisionFramesResident(&.{ vf0, vf1 });
            defer compute.removeVisionFramesResident(&.{ vf0, vf1 }) catch |err| {
                std.debug.panic("removeVisionFramesResident failed: {s}", .{@errorName(err)});
            };

            var frame = try compute.beginFrame(5000);
            errdefer frame.abort();
            if (out_fmt == .nv12) {
                try ingress.encodePatchifyNv12(
                    &frame,
                    vf0,
                    vf1,
                    geom,
                    info0.isFullRange(),
                    NormParams.QWEN_VL_DEFAULT,
                    .f16,
                    0,
                    false,
                );
            } else {
                try ingress.encodePatchifyBgra8(
                    &frame,
                    vf0,
                    vf1,
                    geom,
                    NormParams.QWEN_VL_DEFAULT,
                    .f16,
                    0,
                    false,
                );
            }
            const sig = try frame.commit();
            try compute.wait(sig, 5000);

            // Verify GPU output directly against CPU fp32 reference read from the same IOSurface.
            const ref = try std.testing.allocator.alloc(f32, @as(usize, num_tokens) * token_in_dim);
            defer std.testing.allocator.free(ref);

            if (out_fmt == .nv12) {
                var y0g = try vf0.lockPlane(0, true);
                defer y0g.release();
                var uv0g = try vf0.lockPlane(1, true);
                defer uv0g.release();
                var y1g = try vf1.lockPlane(0, true);
                defer y1g.release();
                var uv1g = try vf1.lockPlane(1, true);
                defer uv1g.release();
                try patchifyNv12CpuRef(
                    y0g.bytes,
                    y0g.bytes_per_row,
                    uv0g.bytes,
                    uv0g.bytes_per_row,
                    y1g.bytes,
                    y1g.bytes_per_row,
                    uv1g.bytes,
                    uv1g.bytes_per_row,
                    geom,
                    info0.isFullRange(),
                    NormParams.QWEN_VL_DEFAULT,
                    ref,
                );
            } else {
                var b0g = try vf0.lockPlane(0, true);
                defer b0g.release();
                var b1g = try vf1.lockPlane(0, true);
                defer b1g.release();
                try patchifyBgra8CpuRef(
                    b0g.bytes,
                    b0g.bytes_per_row,
                    b1g.bytes,
                    b1g.bytes_per_row,
                    geom,
                    NormParams.QWEN_VL_DEFAULT,
                    ref,
                );
            }

            const got_u16 = ingress.patchOutRawU16()[0..ref.len];
            var non_zero_count: usize = 0;
            for (ref, got_u16) |exp_v, got_bits| {
                const got_v = quant.floatBitsToF32(.f16, got_bits);
                if (@abs(got_v) > 0.01) non_zero_count += 1;
                try std.testing.expectApproxEqAbs(exp_v, got_v, @max(0.005, @abs(exp_v) * 0.005));
            }
            try std.testing.expect(non_zero_count > ref.len / 2);
        }
    }
}

test "G14 BGRA8 patchify (image t=1 p=14 merge=1, video t=2 p=16 merge=2, sRGB EOTF, f16 & bf16) matches CPU fp32 ref" {
    const device = try metal.Device.open();
    defer device.close();
    const lib = try metal.Library.openEmbedded(device);
    defer lib.close();
    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();

    const Case = struct {
        geom: PatchGeometry,
        norm: NormParams,
        out_dtype: quant.FloatDType,
        tol: f32,
    };
    const cases = [_]Case{
        .{
            .geom = .{ .width = 28, .height = 28, .patch_size = 14, .t_patch = 1, .spatial_merge = 1 },
            .norm = NormParams.QWEN_VL_DEFAULT,
            .out_dtype = .f16,
            .tol = 0.005,
        },
        .{
            .geom = .{ .width = 64, .height = 32, .patch_size = 16, .t_patch = 2, .spatial_merge = 2 },
            .norm = .{
                .mean = NormParams.SIGLIP_DEFAULT.mean,
                .inv_std = NormParams.SIGLIP_DEFAULT.inv_std,
                .apply_srgb_to_linear = true,
            },
            .out_dtype = .bf16,
            .tol = 0.025,
        },
    };

    for (cases, 0..) |tc, ci| {
        const f0 = try metal.VisionFrame.init(device, .bgra8, tc.geom.width, tc.geom.height);
        defer f0.close();
        const f1 = try metal.VisionFrame.init(device, .bgra8, tc.geom.width, tc.geom.height);
        defer f1.close();
        try fillSyntheticBgra8(f0, tc.geom.width, tc.geom.height, @intCast(ci * 2 + 1));
        try fillSyntheticBgra8(f1, tc.geom.width, tc.geom.height, @intCast(ci * 2 + 2));

        const num_tokens = try tc.geom.numTokens();
        const token_dim = try tc.geom.tokenInDim();

        var ingress = try VisionIngress.init(device, lib, testBudget(8 * 1024 * 1024), .{
            .max_tokens = num_tokens,
            .max_patch_dim = token_dim,
            .max_hidden_size = 16,
            .max_mrope_tokens = 32,
        });
        defer ingress.deinit();

        try compute.makeResident(&.{ingress.buffer()});
        try compute.makeVisionFramesResident(&.{ f0, f1 });
        defer {
            compute.clearResidency() catch |err| {
                std.debug.panic("clearResidency failed: {s}", .{@errorName(err)});
            };
        }

        var frame = try compute.beginFrame(5000);
        errdefer frame.abort();
        try ingress.encodePatchifyBgra8(
            &frame,
            f0,
            if (tc.geom.t_patch == 2) f1 else null,
            tc.geom,
            tc.norm,
            tc.out_dtype,
            0,
            false,
        );
        const sig = try frame.commit();
        try compute.wait(sig, 5000);

        const ref = try std.testing.allocator.alloc(f32, @as(usize, num_tokens) * token_dim);
        defer std.testing.allocator.free(ref);

        var g0 = try f0.lockPlane(0, true);
        defer g0.release();
        var g1 = try f1.lockPlane(0, true);
        defer g1.release();
        try patchifyBgra8CpuRef(
            g0.bytes,
            g0.bytes_per_row,
            if (tc.geom.t_patch == 2) g1.bytes else null,
            g1.bytes_per_row,
            tc.geom,
            tc.norm,
            ref,
        );

        const got_raw = ingress.patchOutRawU16()[0..ref.len];
        for (ref, got_raw) |exp_v, got_bits| {
            const got_v = quant.floatBitsToF32(tc.out_dtype, got_bits);
            try std.testing.expectApproxEqAbs(exp_v, got_v, @max(tc.tol, @abs(exp_v) * tc.tol));
        }
    }
}

test "G14 NV12 patchify (full-range 420f & video-range 420v, t=2 p=14 merge=2, f16 & bf16) matches CPU fp32 ref" {
    const device = try metal.Device.open();
    defer device.close();
    const lib = try metal.Library.openEmbedded(device);
    defer lib.close();
    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();

    const geom: PatchGeometry = .{
        .width = 56,
        .height = 56,
        .patch_size = 14,
        .t_patch = 2,
        .spatial_merge = 2,
    };
    const num_tokens = try geom.numTokens();
    const token_dim = try geom.tokenInDim();

    const f0 = try metal.VisionFrame.init(device, .nv12, geom.width, geom.height);
    defer f0.close();
    const f1 = try metal.VisionFrame.init(device, .nv12, geom.width, geom.height);
    defer f1.close();
    try fillSyntheticNv12(f0, geom.width, geom.height, 11);
    try fillSyntheticNv12(f1, geom.width, geom.height, 29);

    var ingress = try VisionIngress.init(device, lib, testBudget(8 * 1024 * 1024), .{
        .max_tokens = num_tokens,
        .max_patch_dim = token_dim,
        .max_hidden_size = 16,
        .max_mrope_tokens = 32,
    });
    defer ingress.deinit();

    try compute.makeResident(&.{ingress.buffer()});
    try compute.makeVisionFramesResident(&.{ f0, f1 });
    defer {
        compute.clearResidency() catch |err| {
            std.debug.panic("clearResidency failed: {s}", .{@errorName(err)});
        };
    }

    const modes = [_]struct { full_range: bool, dtype: quant.FloatDType, tol: f32 }{
        .{ .full_range = true, .dtype = .f16, .tol = 0.005 },
        .{ .full_range = false, .dtype = .bf16, .tol = 0.025 },
    };

    const ref = try std.testing.allocator.alloc(f32, @as(usize, num_tokens) * token_dim);
    defer std.testing.allocator.free(ref);

    for (modes) |m| {
        var frame = try compute.beginFrame(5000);
        errdefer frame.abort();
        try ingress.encodePatchifyNv12(
            &frame,
            f0,
            f1,
            geom,
            m.full_range,
            NormParams.QWEN_VL_DEFAULT,
            m.dtype,
            0,
            false,
        );
        const sig = try frame.commit();
        try compute.wait(sig, 5000);

        var y0g = try f0.lockPlane(0, true);
        defer y0g.release();
        var uv0g = try f0.lockPlane(1, true);
        defer uv0g.release();
        var y1g = try f1.lockPlane(0, true);
        defer y1g.release();
        var uv1g = try f1.lockPlane(1, true);
        defer uv1g.release();
        try patchifyNv12CpuRef(
            y0g.bytes,
            y0g.bytes_per_row,
            uv0g.bytes,
            uv0g.bytes_per_row,
            y1g.bytes,
            y1g.bytes_per_row,
            uv1g.bytes,
            uv1g.bytes_per_row,
            geom,
            m.full_range,
            NormParams.QWEN_VL_DEFAULT,
            ref,
        );

        const got_raw = ingress.patchOutRawU16()[0..ref.len];
        for (ref, got_raw) |exp_v, got_bits| {
            const got_v = quant.floatBitsToF32(m.dtype, got_bits);
            try std.testing.expectApproxEqAbs(exp_v, got_v, @max(m.tol, @abs(exp_v) * m.tol));
        }
    }
}

test "G14 fused patchify + 2x2 merge + linear patch embedding (BGRA8 & NV12, f16 & bf16) matches CPU fp32 ref" {
    const device = try metal.Device.open();
    defer device.close();
    const lib = try metal.Library.openEmbedded(device);
    defer lib.close();
    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();

    const geom: PatchGeometry = .{
        .width = 64,
        .height = 32,
        .patch_size = 16,
        .t_patch = 2,
        .spatial_merge = 2,
    };
    const num_tokens = try geom.numTokens(); // 2 * 1 = 2
    const token_in_dim = try geom.tokenInDim(); // 4 * 3 * 2 * 16 * 16 = 6144
    const hidden_size: u32 = 37; // intentionally non-multiple of 8 to test tail channels

    const bgra0 = try metal.VisionFrame.init(device, .bgra8, geom.width, geom.height);
    defer bgra0.close();
    const bgra1 = try metal.VisionFrame.init(device, .bgra8, geom.width, geom.height);
    defer bgra1.close();
    try fillSyntheticBgra8(bgra0, geom.width, geom.height, 3);
    try fillSyntheticBgra8(bgra1, geom.width, geom.height, 7);

    const nv0 = try metal.VisionFrame.init(device, .nv12, geom.width, geom.height);
    defer nv0.close();
    const nv1 = try metal.VisionFrame.init(device, .nv12, geom.width, geom.height);
    defer nv1.close();
    try fillSyntheticNv12(nv0, geom.width, geom.height, 5);
    try fillSyntheticNv12(nv1, geom.width, geom.height, 9);

    var ingress = try VisionIngress.init(device, lib, testBudget(16 * 1024 * 1024), .{
        .max_tokens = num_tokens + 2,
        .max_patch_dim = token_in_dim,
        .max_hidden_size = hidden_size,
        .max_mrope_tokens = 32,
    });
    defer ingress.deinit();

    // Separate wired arena holding linear embedding weights + optional bias.
    const w_bytes = @as(usize, hidden_size) * @as(usize, token_in_dim) * 2;
    const b_bytes = @as(usize, hidden_size) * 2;
    var w_arena = try arena_mod.WiredArena.init(try page.alignUp(w_bytes + b_bytes + 256), testBudget(16 * 1024 * 1024));
    defer w_arena.deinit();
    const w_raw_bytes = try w_arena.allocAligned(w_bytes, 64);
    const b_raw_bytes = try w_arena.allocAligned(b_bytes, 64);
    const w_u16: []u16 = @alignCast(std.mem.bytesAsSlice(u16, w_raw_bytes));
    const b_u16: []u16 = @alignCast(std.mem.bytesAsSlice(u16, b_raw_bytes));
    const w_buf = try metal.Buffer.wrap(device, w_arena.bytes());
    defer w_buf.close();

    try compute.makeResident(&.{ ingress.buffer(), w_buf });
    try compute.makeVisionFramesResident(&.{ bgra0, bgra1, nv0, nv1 });
    defer {
        compute.clearResidency() catch |err| {
            std.debug.panic("clearResidency failed: {s}", .{@errorName(err)});
        };
    }

    const patches_ref = try std.testing.allocator.alloc(f32, @as(usize, num_tokens) * token_in_dim);
    defer std.testing.allocator.free(patches_ref);
    const embed_ref = try std.testing.allocator.alloc(f32, @as(usize, num_tokens) * hidden_size);
    defer std.testing.allocator.free(embed_ref);

    const configs = [_]struct {
        is_nv12: bool,
        w_dtype: quant.FloatDType,
        o_dtype: quant.FloatDType,
        use_bias: bool,
        token_offset: u32,
        tol: f32,
    }{
        .{ .is_nv12 = false, .w_dtype = .f16, .o_dtype = .f16, .use_bias = true, .token_offset = 0, .tol = 0.02 },
        .{ .is_nv12 = false, .w_dtype = .bf16, .o_dtype = .bf16, .use_bias = false, .token_offset = 1, .tol = 0.05 },
        .{ .is_nv12 = true, .w_dtype = .bf16, .o_dtype = .f16, .use_bias = true, .token_offset = 0, .tol = 0.02 },
        .{ .is_nv12 = true, .w_dtype = .f16, .o_dtype = .bf16, .use_bias = true, .token_offset = 2, .tol = 0.05 },
    };

    for (configs, 0..) |cfg, ci| {
        for (w_u16, 0..) |*dst, idx| {
            dst.* = quant.f32ToFloatBits(cfg.w_dtype, synthWeight(idx + ci * 17, 0.001));
        }
        for (b_u16, 0..) |*dst, idx| {
            dst.* = quant.f32ToFloatBits(cfg.w_dtype, synthWeight(idx + ci * 5 + 3, 0.01));
        }

        const w_bind: metal.BufferBinding = .{ .buffer = w_buf, .offset = w_arena.offsetOf(w_raw_bytes.ptr) };
        const b_bind: ?metal.BufferBinding = if (cfg.use_bias)
            .{ .buffer = w_buf, .offset = w_arena.offsetOf(b_raw_bytes.ptr) }
        else
            null;

        var frame = try compute.beginFrame(5000);
        errdefer frame.abort();
        if (cfg.is_nv12) {
            try ingress.encodePatchifyEmbedNv12(
                &frame,
                nv0,
                nv1,
                w_bind,
                b_bind,
                cfg.w_dtype,
                cfg.o_dtype,
                hidden_size,
                geom,
                true,
                NormParams.QWEN_VL_DEFAULT,
                cfg.token_offset,
                false,
            );
        } else {
            try ingress.encodePatchifyEmbedBgra8(
                &frame,
                bgra0,
                bgra1,
                w_bind,
                b_bind,
                cfg.w_dtype,
                cfg.o_dtype,
                hidden_size,
                geom,
                NormParams.QWEN_VL_DEFAULT,
                cfg.token_offset,
                false,
            );
        }
        const sig = try frame.commit();
        try compute.wait(sig, 5000);

        if (cfg.is_nv12) {
            var y0g = try nv0.lockPlane(0, true);
            defer y0g.release();
            var uv0g = try nv0.lockPlane(1, true);
            defer uv0g.release();
            var y1g = try nv1.lockPlane(0, true);
            defer y1g.release();
            var uv1g = try nv1.lockPlane(1, true);
            defer uv1g.release();
            try patchifyNv12CpuRef(
                y0g.bytes,
                y0g.bytes_per_row,
                uv0g.bytes,
                uv0g.bytes_per_row,
                y1g.bytes,
                y1g.bytes_per_row,
                uv1g.bytes,
                uv1g.bytes_per_row,
                geom,
                true,
                NormParams.QWEN_VL_DEFAULT,
                patches_ref,
            );
        } else {
            var g0 = try bgra0.lockPlane(0, true);
            defer g0.release();
            var g1 = try bgra1.lockPlane(0, true);
            defer g1.release();
            try patchifyBgra8CpuRef(
                g0.bytes,
                g0.bytes_per_row,
                g1.bytes,
                g1.bytes_per_row,
                geom,
                NormParams.QWEN_VL_DEFAULT,
                patches_ref,
            );
        }

        try linearEmbedCpuRef(
            patches_ref,
            w_u16,
            if (cfg.use_bias) b_u16 else null,
            cfg.w_dtype,
            num_tokens,
            token_in_dim,
            hidden_size,
            embed_ref,
        );

        const base_out = @as(usize, cfg.token_offset) * hidden_size;
        const got_slice = ingress.embedOutRawU16()[base_out .. base_out + embed_ref.len];
        for (embed_ref, got_slice) |exp_v, got_bits| {
            const got_v = quant.floatBitsToF32(cfg.o_dtype, got_bits);
            try std.testing.expectApproxEqAbs(exp_v, got_v, @max(cfg.tol, @abs(exp_v) * cfg.tol));
        }
    }
}

test "G14 3D MRoPE buildMultimodalMropePositions layout and end-to-end M6 G9 interleaved MRoPE [11, 11, 10] GPU parity" {
    // 1. Exact coordinate layout verification:
    //    prefix=3 text tokens (0,1,2), video grid (t=2, h=2, w=3 -> 12 tokens),
    //    suffix=2 text tokens.
    //    base = 3, max_vis_pos = 3 + max(1, 1, 2) = 5 -> suffix starts at 6 (6, 7).
    const grid: VisionGridSpec = .{ .grid_t = 2, .grid_h = 2, .grid_w = 3 };
    const total: usize = 3 + 12 + 2;
    var planar: [3 * total]u32 = undefined;
    const res = try buildMultimodalMropePositions(3, grid, 2, &planar);
    try std.testing.expectEqual(@as(u32, 17), res.total_tokens);
    try std.testing.expectEqual(@as(u32, 8), res.next_text_pos);
    try std.testing.expectEqual(@as(i64, -9), res.mrope_delta);

    const t_row = planar[0..total];
    const h_row = planar[total .. 2 * total];
    const w_row = planar[2 * total .. 3 * total];

    // Text prefix: t = h = w = pos
    for (0..3) |i| {
        try std.testing.expectEqual(@as(u32, @intCast(i)), t_row[i]);
        try std.testing.expectEqual(@as(u32, @intCast(i)), h_row[i]);
        try std.testing.expectEqual(@as(u32, @intCast(i)), w_row[i]);
    }
    // Visual tokens: base = 3, row-major (dt, dh, dw)
    var idx: usize = 3;
    for (0..2) |dt| {
        for (0..2) |dh| {
            for (0..3) |dw| {
                try std.testing.expectEqual(@as(u32, @intCast(3 + dt)), t_row[idx]);
                try std.testing.expectEqual(@as(u32, @intCast(3 + dh)), h_row[idx]);
                try std.testing.expectEqual(@as(u32, @intCast(3 + dw)), w_row[idx]);
                idx += 1;
            }
        }
    }
    // Text suffix: max_vis_pos + 1 + offset = 6, 7
    try std.testing.expectEqual(@as(u32, 6), t_row[15]);
    try std.testing.expectEqual(@as(u32, 6), h_row[15]);
    try std.testing.expectEqual(@as(u32, 6), w_row[15]);
    try std.testing.expectEqual(@as(u32, 7), t_row[16]);
    try std.testing.expectEqual(@as(u32, 7), h_row[16]);
    try std.testing.expectEqual(@as(u32, 7), w_row[16]);

    // 2. End-to-end GPU integration: feed VisionIngress MRoPE positions directly into
    //    M6 G9 HybridPipelines.encodeQsaTreeRope with MropeSection.QWEN38_DEFAULT [11, 11, 10]!
    const device = try metal.Device.open();
    defer device.close();
    const lib = try metal.Library.openEmbedded(device);
    defer lib.close();
    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();

    var ingress = try VisionIngress.init(device, lib, testBudget(8 * 1024 * 1024), .{
        .max_tokens = 16,
        .max_patch_dim = 64,
        .max_hidden_size = 64,
        .max_mrope_tokens = 32,
    });
    defer ingress.deinit();

    const built = try ingress.buildMropePositions(3, grid, 2);
    try std.testing.expectEqual(@as(u32, 17), built.total_tokens);

    const num_nodes: u32 = built.total_tokens;
    const num_q_heads: u32 = 4;
    const num_kv_heads: u32 = 2;
    const head_dim: u32 = 128;
    const rotary_dim: u32 = 64; // [11, 11, 10] sums to 32 = rotary_dim / 2

    var qsa_pipes = try hybrid.HybridPipelines.init(device, lib);
    defer qsa_pipes.close();

    var qsa_arena = try arena_mod.WiredArena.init(512 * 1024, testBudget(8 * 1024 * 1024));
    defer qsa_arena.deinit();

    const q_raw = try qsa_arena.allocAligned(num_nodes * num_q_heads * head_dim * 2, 64);
    const k_raw = try qsa_arena.allocAligned(num_nodes * num_kv_heads * head_dim * 2, 64);
    const v_raw = try qsa_arena.allocAligned(num_nodes * num_kv_heads * head_dim * 2, 64);
    const qn_raw = try qsa_arena.allocAligned(head_dim * 2, 64);
    const kn_raw = try qsa_arena.allocAligned(head_dim * 2, 64);
    const k_out_raw = try qsa_arena.allocAligned(num_nodes * num_kv_heads * head_dim * 2, 64);
    const v_out_raw = try qsa_arena.allocAligned(num_nodes * num_kv_heads * head_dim * 2, 64);

    const q_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, q_raw));
    const k_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, k_raw));
    const v_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, v_raw));
    const qn_u16: []u16 = @alignCast(std.mem.bytesAsSlice(u16, qn_raw));
    const kn_u16: []u16 = @alignCast(std.mem.bytesAsSlice(u16, kn_raw));

    for (q_f16, 0..) |*x, i| x.* = @floatCast(synthWeight(i + 1, 0.04));
    for (k_f16, 0..) |*x, i| x.* = @floatCast(synthWeight(i + 7, 0.04));
    for (v_f16, 0..) |*x, i| x.* = @floatCast(synthWeight(i + 19, 0.04));
    for (qn_u16, 0..) |*x, i| x.* = quant.f32ToFloatBits(.f16, 0.9 + @abs(synthWeight(i + 3, 0.01)));
    for (kn_u16, 0..) |*x, i| x.* = quant.f32ToFloatBits(.f16, 0.85 + @abs(synthWeight(i + 5, 0.01)));

    const pos_slice = ingress.mropePos3d()[0..num_nodes];
    const q_ref = try std.testing.allocator.alloc(f32, q_f16.len);
    defer std.testing.allocator.free(q_ref);
    const k_ref = try std.testing.allocator.alloc(f32, k_f16.len);
    defer std.testing.allocator.free(k_ref);

    for (0..num_nodes) |n| {
        for (0..num_q_heads) |qh| {
            const off = (n * num_q_heads + qh) * head_dim;
            hybrid.qsaQkNormMropeHeadCpuRef(
                q_f16[off..][0..head_dim],
                qn_u16,
                .f16,
                q_ref[off..][0..head_dim],
                head_dim,
                rotary_dim,
                pos_slice[n],
                hybrid.MropeSection.QWEN38_DEFAULT,
                .{},
                1e-6,
                1_000_000.0,
            );
        }
        for (0..num_kv_heads) |kh| {
            const off = (n * num_kv_heads + kh) * head_dim;
            hybrid.qsaQkNormMropeHeadCpuRef(
                k_f16[off..][0..head_dim],
                kn_u16,
                .f16,
                k_ref[off..][0..head_dim],
                head_dim,
                rotary_dim,
                pos_slice[n],
                hybrid.MropeSection.QWEN38_DEFAULT,
                .{},
                1e-6,
                1_000_000.0,
            );
        }
    }

    const qsa_buf = try metal.Buffer.wrap(device, qsa_arena.bytes());
    defer qsa_buf.close();
    try compute.makeResident(&.{ ingress.buffer(), qsa_buf });

    var frame = try compute.beginFrame(5000);
    errdefer frame.abort();
    try qsa_pipes.encodeQsaTreeRope(&frame, .{
        .q = .{ .buffer = qsa_buf, .offset = qsa_arena.offsetOf(q_raw.ptr) },
        .k = .{ .buffer = qsa_buf, .offset = qsa_arena.offsetOf(k_raw.ptr) },
        .v = .{ .buffer = qsa_buf, .offset = qsa_arena.offsetOf(v_raw.ptr) },
        .q_norm = .{ .buffer = qsa_buf, .offset = qsa_arena.offsetOf(qn_raw.ptr) },
        .k_norm = .{ .buffer = qsa_buf, .offset = qsa_arena.offsetOf(kn_raw.ptr) },
        .norm_dtype = .f16,
        .positions_3d = ingress.mropePos3dBinding(),
        .k_tree_out = .{ .buffer = qsa_buf, .offset = qsa_arena.offsetOf(k_out_raw.ptr) },
        .v_tree_out = .{ .buffer = qsa_buf, .offset = qsa_arena.offsetOf(v_out_raw.ptr) },
        .num_nodes = num_nodes,
        .num_q_heads = num_q_heads,
        .num_kv_heads = num_kv_heads,
        .head_dim = head_dim,
        .rotary_dim = rotary_dim,
        .mrope = hybrid.MropeSection.QWEN38_DEFAULT,
    }, false);
    const sig = try frame.commit();
    try compute.wait(sig, 5000);

    const k_out: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, k_out_raw));
    for (q_ref, q_f16) |exp_v, act_h| {
        try std.testing.expectApproxEqAbs(exp_v, @as(f32, @floatCast(act_h)), @max(0.015, @abs(exp_v) * 0.015));
    }
    for (k_ref, k_out) |exp_v, act_h| {
        try std.testing.expectApproxEqAbs(exp_v, @as(f32, @floatCast(act_h)), @max(0.015, @abs(exp_v) * 0.015));
    }
}

test "G14 vision ingress validation and error handling" {
    const device = try metal.Device.open();
    defer device.close();

    // Reject odd NV12 dimensions and zero sizes explicitly.
    try std.testing.expectError(error.InvalidArgument, metal.VisionFrame.init(device, .nv12, 33, 32));
    try std.testing.expectError(error.InvalidArgument, metal.VisionFrame.init(device, .bgra8, 0, 32));
    try std.testing.expectError(error.InvalidArgument, metal.VtClip.encodeTestClip(.h264, 8, 8, 2, 30));

    // Reject invalid patch geometry and tiny MRoPE buffers.
    try std.testing.expectError(error.InvalidGeometry, (PatchGeometry{
        .width = 30,
        .height = 28,
        .patch_size = 14,
        .t_patch = 1,
        .spatial_merge = 1,
    }).validate());
    try std.testing.expectError(error.InvalidGeometry, (PatchGeometry{
        .width = 28,
        .height = 28,
        .patch_size = 14,
        .t_patch = 3,
        .spatial_merge = 1,
    }).validate());

    var small: [6]u32 = undefined;
    try std.testing.expectError(error.BufferTooSmall, buildMultimodalMropePositions(2, .{
        .grid_t = 1,
        .grid_h = 2,
        .grid_w = 2,
    }, 1, &small));
    try std.testing.expectError(error.InvalidGrid, buildMultimodalMropePositions(1, .{
        .grid_t = 0,
        .grid_h = 2,
        .grid_w = 2,
    }, 1, &small));
}

test "G14 VisionFrame(F16) zero-copy CoreML ANE prediction, VtClip.encodeFrames and multi-span MRoPE" {
    const device = try metal.Device.open();
    defer device.close();

    // 1. Real CoreML / ANE prediction directly over VisionFrame(.f16) IOSurfaces (0 copies).
    const c_in: u32 = 32;
    const c_out: u32 = 32;
    const s_dim: u32 = 32;
    var w: [c_out * c_in]f16 = undefined;
    for (0..c_out) |o| {
        for (0..c_in) |ci| {
            w[o * c_in + ci] = @floatCast(synthWeight(o * c_in + ci + 5, 0.02));
        }
    }
    const spec_bytes = try metal.emitCoreMlLinearModel(std.testing.allocator, .{
        .op = .conv1x1,
        .batch_or_spatial = s_dim,
        .layers = &.{.{ .c_in = c_in, .c_out = c_out, .weights_f16 = &w }},
    });
    defer std.testing.allocator.free(spec_bytes);

    const model = try metal.AneModel.loadSpecData(spec_bytes, .cpu_and_neural_engine);
    defer model.close();

    const in_frame = try metal.VisionFrame.init(device, .f16, s_dim, c_in);
    defer in_frame.close();
    const out_frame = try metal.VisionFrame.init(device, .f16, s_dim, c_out);
    defer out_frame.close();

    var x_host: [c_in * s_dim]f16 = undefined;
    {
        var ig = try in_frame.lockPlane(0, false);
        errdefer ig.release();
        const dst: []f16 = @alignCast(std.mem.bytesAsSlice(f16, ig.bytes));
        for (0..c_in) |ci| {
            for (0..s_dim) |si| {
                const v: f16 = @floatCast(synthWeight(ci * s_dim + si + 11, 0.05));
                x_host[ci * s_dim + si] = v;
                dst[ci * s_dim + si] = v;
            }
        }
        try ig.unlock();
    }

    const in_tensor = try in_frame.toAneTensor(&.{ c_in, 1, s_dim });
    defer in_tensor.close();
    const out_tensor = try out_frame.toAneTensor(&.{ c_out, 1, s_dim });
    defer out_tensor.close();

    const plan = try metal.AnePlan.init(
        &.{.{ .name = "x", .tensor = in_tensor }},
        &.{.{ .name = "y", .tensor = out_tensor }},
    );
    defer plan.close();

    const pinfo = try model.predict(plan);
    try std.testing.expectEqual(@as(u32, 1), pinfo.outputsTotal());
    try std.testing.expect(pinfo.allOutputsZeroCopy());

    {
        var og = try out_frame.lockPlane(0, true);
        errdefer og.release();
        const y_got: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, og.bytes));
        for (0..c_out) |o| {
            for (0..s_dim) |si| {
                var acc: f64 = 0.0;
                for (0..c_in) |ci| {
                    acc += @as(f64, @floatCast(w[o * c_in + ci])) *
                        @as(f64, @floatCast(x_host[ci * s_dim + si]));
                }
                const exp: f32 = @floatCast(acc);
                const got: f32 = @floatCast(y_got[o * s_dim + si]);
                try std.testing.expectApproxEqAbs(exp, got, @max(0.02, @abs(exp) * 0.02));
            }
        }
        try og.unlock();
    }

    // 2. VtClip.encodeFrames from caller-owned BGRA8 VisionFrames -> HEVC -> NV12 zero-copy decode.
    const f0 = try metal.VisionFrame.init(device, .bgra8, 64, 64);
    defer f0.close();
    const f1 = try metal.VisionFrame.init(device, .bgra8, 64, 64);
    defer f1.close();
    try fillSyntheticBgra8(f0, 64, 64, 1);
    try fillSyntheticBgra8(f1, 64, 64, 2);

    const clip = try metal.VtClip.encodeFrames(.hevc, &.{ f0, f1 }, 24);
    defer clip.close();
    const ci = try clip.info();
    try std.testing.expectEqual(@as(u32, 2), ci.frameCount());
    try std.testing.expect(ci.totalCompressedBytes() > 0);

    const dec = try metal.VtDecodeSession.init(device, clip, .nv12);
    defer dec.close();
    const d0 = try dec.decodeFrame(clip, 0);
    defer d0.close();
    const d1 = try dec.decodeFrame(clip, 1);
    defer d1.close();
    const di0 = try d0.info();
    const di1 = try d1.info();
    try std.testing.expect(di0.zeroCopyVerified() and di1.zeroCopyVerified());
    try std.testing.expect(di0.isVtDecoded() and di1.isVtDecoded());

    // 3. Multi-span 3D MRoPE interleaving text + image + text + video (temporal_stride=2) + text.
    const segs = [_]MropeSegment{
        .{ .text = 2 }, // pos 0, 1 -> next=2
        .{ .vision = .{ .grid_t = 1, .grid_h = 2, .grid_w = 2 } }, // base=2, max=3 -> next=4
        .{ .text = 1 }, // pos 4 -> next=5
        .{ .vision = .{ .grid_t = 2, .grid_h = 1, .grid_w = 2, .temporal_stride = 2 } }, // base=5, t in {5,7}, max=7 -> next=8
        .{ .text = 2 }, // pos 8, 9 -> next=10
    };
    const total: usize = 2 + 4 + 1 + 4 + 2; // 13
    var planar: [3 * total]u32 = undefined;
    const mres = try buildMultimodalMropePositionsMulti(&segs, &planar);
    try std.testing.expectEqual(@as(u32, 13), mres.total_tokens);
    try std.testing.expectEqual(@as(u32, 10), mres.next_text_pos);
    try std.testing.expectEqual(@as(i64, -3), mres.mrope_delta);

    var pos3d: [total]hybrid.Pos3D = undefined;
    try packMropePlanarToPos3D(&planar, total, &pos3d);
    try std.testing.expectEqual(hybrid.Pos3D.text(0), pos3d[0]);
    try std.testing.expectEqual(hybrid.Pos3D.text(1), pos3d[1]);
    try std.testing.expectEqual(hybrid.Pos3D{ .t = 2, .h = 2, .w = 2 }, pos3d[2]);
    try std.testing.expectEqual(hybrid.Pos3D{ .t = 2, .h = 3, .w = 3 }, pos3d[5]);
    try std.testing.expectEqual(hybrid.Pos3D.text(4), pos3d[6]);
    try std.testing.expectEqual(hybrid.Pos3D{ .t = 5, .h = 5, .w = 5 }, pos3d[7]);
    try std.testing.expectEqual(hybrid.Pos3D{ .t = 7, .h = 5, .w = 6 }, pos3d[10]);
    try std.testing.expectEqual(hybrid.Pos3D.text(8), pos3d[11]);
    try std.testing.expectEqual(hybrid.Pos3D.text(9), pos3d[12]);
}
