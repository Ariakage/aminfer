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
const chip_profile = aminfer.chip_profile;
const vmstat = aminfer.vmstat;

pub const MAX_PATCH_SIZE: u32 = 32;
pub const PATCHIFY_TG_THREADS: u64 = 256;
pub const EMBED_TG_CHANNELS: u32 = 8;

pub const Error = error{
    InvalidGeometry,
    InvalidFormat,
    InvalidHiddenSize,
    InvalidHeadDim,
    InvalidLayerCount,
    InvalidBlockWeights,
    InvalidTokenOffset,
    OutputBufferTooSmall,
    WeightBufferTooSmall,
    BiasBufferTooSmall,
    AliasedArenaTooSmall,
    AneShapeMismatch,
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
// M7.8 G22: Dynamic Aspect-Ratio AnyRes Tiling, Full Multi-Layer Bidirectional
// 2D-RoPE Vision Transformer Tower, PatchMerger / Aligner MLP, Aliased Wired
// Scratch, Zero-Overhead Text-Only Fast Path, & C16/D5 CoreML ANE Offload Hook
// ===========================================================================

pub const MAX_ANYRES_TILES: usize = 16;
pub const MAX_VIT_HEAD_DIM: u32 = 128;
pub const VIT_FA_BQ: u32 = 32;
pub const VIT_LINEAR_TILE: u32 = 32;
pub const VIT_LINEAR_NAX_TILE: u32 = 64;
pub const IOSURFACE_F16_ROW_ALIGN: u32 = 32;

/// 2D spatial patch coordinates `(pos_h, pos_w)` matching Metal `uint2`.
pub const RopeCoord2D = extern struct {
    h: u32,
    w: u32,

    comptime {
        std.debug.assert(@sizeOf(RopeCoord2D) == 8);
    }
};

/// Per-query bidirectional KV window `[kv_start, kv_end)` matching Metal `uint2`.
pub const AttnWindowBound = extern struct {
    kv_start: u32,
    kv_end: u32,

    comptime {
        std.debug.assert(@sizeOf(AttnWindowBound) == 8);
    }
};

pub const AnyResMode = enum(u8) {
    /// Qwen2-VL / Qwen3-VL / Qwen3.8 smart-resize snapping `(w, h)` to multiples
    /// of `patch_size * spatial_merge` inside `[min_pixels, max_pixels]`.
    smart_resize,
    /// DeepSeek-V4.1 / LLaVA-AnyRes dynamic aspect-ratio crop grid + optional
    /// full-frame global thumbnail tile.
    dynamic_grid,
};

pub const AnyResConfig = struct {
    mode: AnyResMode = .smart_resize,
    patch_size: u32 = 14,
    t_patch: u32 = 1,
    spatial_merge: u32 = 2,
    min_pixels: u32 = 256 * 14 * 14,
    max_pixels: u32 = 1024 * 14 * 14,
    /// Per-tile target resolution for `.dynamic_grid` mode (must be divisible by `patch_size * spatial_merge`).
    tile_size: u32 = 224,
    max_tiles: u32 = 6,
    include_thumbnail: bool = true,
};

pub const AnyResTileSpec = struct {
    src_x0: f32,
    src_y0: f32,
    src_w: f32,
    src_h: f32,
    dst_width: u32,
    dst_height: u32,
    grid_h: u32,
    grid_w: u32,
    token_offset: u32,
    is_thumbnail: bool,
};

pub const AnyResPlan = struct {
    src_width: u32,
    src_height: u32,
    resized_width: u32,
    resized_height: u32,
    grid_cols: u32,
    grid_rows: u32,
    patch_size: u32,
    spatial_merge: u32,
    total_patches: u32,
    total_merged_tokens: u32,
    num_tiles: u32,
    tiles: [MAX_ANYRES_TILES]AnyResTileSpec,

    pub fn activeTiles(self: *const AnyResPlan) []const AnyResTileSpec {
        return self.tiles[0..self.num_tiles];
    }

    /// Populates `out_coords[0..self.total_patches]` with deterministic per-tile
    /// `(pos_h, pos_w)` integer coordinates for 2D-RoPE (zero heap allocations).
    pub fn fillRope2DCoords(self: *const AnyResPlan, out_coords: []RopeCoord2D) Error!u32 {
        if (out_coords.len < self.total_patches) return error.OutputBufferTooSmall;
        var cursor: u32 = 0;
        for (self.activeTiles()) |tile| {
            for (0..tile.grid_h) |gh| {
                for (0..tile.grid_w) |gw| {
                    out_coords[cursor] = .{ .h = @intCast(gh), .w = @intCast(gw) };
                    cursor += 1;
                }
            }
        }
        return cursor;
    }

    /// Populates `out_bounds[0..self.total_patches]` with per-query bidirectional
    /// window bounds `[kv_start, kv_end)`. When `window_tokens == 0`, each tile
    /// attends bidirectionally across its full tile span without leaking across tiles;
    /// when `window_tokens > 0`, subdivides each tile into contiguous bidirectional
    /// windows of `window_tokens` patches.
    pub fn fillWindowBounds(
        self: *const AnyResPlan,
        window_tokens: u32,
        out_bounds: []AttnWindowBound,
    ) Error!u32 {
        if (out_bounds.len < self.total_patches) return error.OutputBufferTooSmall;
        var cursor: u32 = 0;
        for (self.activeTiles()) |tile| {
            const tile_patches = tile.grid_h * tile.grid_w;
            const tile_start = tile.token_offset;
            const tile_end = tile_start + tile_patches;
            if (window_tokens == 0 or window_tokens >= tile_patches) {
                for (0..tile_patches) |_| {
                    out_bounds[cursor] = .{ .kv_start = tile_start, .kv_end = tile_end };
                    cursor += 1;
                }
            } else {
                for (0..tile_patches) |local_i| {
                    const win_idx: u32 = @as(u32, @intCast(local_i)) / window_tokens;
                    const ws = tile_start + win_idx * window_tokens;
                    const we = @min(tile_end, ws + window_tokens);
                    out_bounds[cursor] = .{ .kv_start = ws, .kv_end = we };
                    cursor += 1;
                }
            }
        }
        return cursor;
    }

    /// Produces a `VisionGridSpec` for `G9` 3D interleaved MRoPE (`[11, 11, 10]`)
    /// after `spatial_merge x spatial_merge` PatchMerger pooling.
    pub fn toVisionGridSpec(self: *const AnyResPlan, temporal_frames: u32, temporal_stride: u32) VisionGridSpec {
        const merged_h = self.tiles[0].grid_h / self.spatial_merge;
        const merged_w = self.tiles[0].grid_w / self.spatial_merge;
        return .{
            .grid_t = temporal_frames * self.num_tiles,
            .grid_h = merged_h,
            .grid_w = merged_w,
            .temporal_stride = temporal_stride,
        };
    }
};

fn roundToFactor(val: f64, factor: u32) u32 {
    const f = @as(f64, @floatFromInt(factor));
    const rounded = @round(val / f) * f;
    const clamped = @max(f, rounded);
    return @intFromFloat(clamped);
}

/// Pure, zero-allocation AnyRes dynamic aspect-ratio grid planner (G22a).
pub fn planAnyResGrid(src_width: u32, src_height: u32, cfg: AnyResConfig) Error!AnyResPlan {
    if (src_width == 0 or src_height == 0 or
        src_width > metal.MAX_TEXTURE_DIM or src_height > metal.MAX_TEXTURE_DIM)
    {
        return error.InvalidGeometry;
    }
    if (cfg.patch_size == 0 or cfg.patch_size > MAX_PATCH_SIZE) return error.InvalidGeometry;
    if (cfg.t_patch != 1 and cfg.t_patch != 2) return error.InvalidGeometry;
    if (cfg.spatial_merge == 0 or cfg.spatial_merge > 4) return error.InvalidGeometry;

    const factor = std.math.mul(u32, cfg.patch_size, cfg.spatial_merge) catch return error.Overflow;
    const merge_sq = std.math.mul(u32, cfg.spatial_merge, cfg.spatial_merge) catch return error.Overflow;

    var plan: AnyResPlan = .{
        .src_width = src_width,
        .src_height = src_height,
        .resized_width = 0,
        .resized_height = 0,
        .grid_cols = 1,
        .grid_rows = 1,
        .patch_size = cfg.patch_size,
        .spatial_merge = cfg.spatial_merge,
        .total_patches = 0,
        .total_merged_tokens = 0,
        .num_tiles = 0,
        .tiles = undefined,
    };

    switch (cfg.mode) {
        .smart_resize => {
            if (cfg.min_pixels == 0 or cfg.max_pixels < cfg.min_pixels) return error.InvalidGeometry;
            var h_bar = roundToFactor(@floatFromInt(src_height), factor);
            var w_bar = roundToFactor(@floatFromInt(src_width), factor);
            const cur_pixels: u64 = @as(u64, h_bar) * @as(u64, w_bar);
            if (cur_pixels > cfg.max_pixels) {
                const beta = @sqrt(@as(f64, @floatFromInt(@as(u64, src_height) * @as(u64, src_width))) /
                    @as(f64, @floatFromInt(cfg.max_pixels)));
                h_bar = @max(factor, @as(u32, @intFromFloat(@floor(@as(f64, @floatFromInt(src_height)) / beta / @as(f64, @floatFromInt(factor))))) * factor);
                w_bar = @max(factor, @as(u32, @intFromFloat(@floor(@as(f64, @floatFromInt(src_width)) / beta / @as(f64, @floatFromInt(factor))))) * factor);
            } else if (cur_pixels < cfg.min_pixels) {
                const beta = @sqrt(@as(f64, @floatFromInt(cfg.min_pixels)) /
                    @as(f64, @floatFromInt(@as(u64, src_height) * @as(u64, src_width))));
                h_bar = @max(factor, @as(u32, @intFromFloat(@ceil(@as(f64, @floatFromInt(src_height)) * beta / @as(f64, @floatFromInt(factor))))) * factor);
                w_bar = @max(factor, @as(u32, @intFromFloat(@ceil(@as(f64, @floatFromInt(src_width)) * beta / @as(f64, @floatFromInt(factor))))) * factor);
            }
            const grid_h = h_bar / cfg.patch_size;
            const grid_w = w_bar / cfg.patch_size;
            const patches = std.math.mul(u32, grid_h, grid_w) catch return error.Overflow;
            plan.resized_width = w_bar;
            plan.resized_height = h_bar;
            plan.grid_cols = 1;
            plan.grid_rows = 1;
            plan.num_tiles = 1;
            plan.total_patches = patches;
            plan.total_merged_tokens = patches / merge_sq;
            plan.tiles[0] = .{
                .src_x0 = 0.0,
                .src_y0 = 0.0,
                .src_w = @floatFromInt(src_width),
                .src_h = @floatFromInt(src_height),
                .dst_width = w_bar,
                .dst_height = h_bar,
                .grid_h = grid_h,
                .grid_w = grid_w,
                .token_offset = 0,
                .is_thumbnail = false,
            };
        },
        .dynamic_grid => {
            if (cfg.tile_size == 0 or (cfg.tile_size % factor) != 0) return error.InvalidGeometry;
            if (cfg.max_tiles == 0) return error.InvalidGeometry;
            const budget_grid_tiles: u32 = if (cfg.include_thumbnail and cfg.max_tiles > 1)
                @min(cfg.max_tiles - 1, @as(u32, MAX_ANYRES_TILES - 1))
            else
                @min(cfg.max_tiles, @as(u32, MAX_ANYRES_TILES));

            const target_ratio = @as(f64, @floatFromInt(src_width)) / @as(f64, @floatFromInt(src_height));
            const src_area = @as(f64, @floatFromInt(@as(u64, src_width) * @as(u64, src_height)));
            const tile_area = @as(f64, @floatFromInt(@as(u64, cfg.tile_size) * @as(u64, cfg.tile_size)));

            var best_cols: u32 = 1;
            var best_rows: u32 = 1;
            var best_diff: f64 = std.math.inf(f64);
            var best_area: u32 = 1;

            var r: u32 = 1;
            while (r <= budget_grid_tiles) : (r += 1) {
                var c: u32 = 1;
                while (c <= budget_grid_tiles) : (c += 1) {
                    const area = r * c;
                    if (area > budget_grid_tiles) continue;
                    const ratio = @as(f64, @floatFromInt(c)) / @as(f64, @floatFromInt(r));
                    const diff = @abs(target_ratio - ratio);
                    if (diff < best_diff - 1e-6 or
                        (@abs(diff - best_diff) <= 1e-6 and
                            @as(f64, @floatFromInt(area)) * tile_area >= 0.5 * src_area and area < best_area))
                    {
                        best_diff = diff;
                        best_cols = c;
                        best_rows = r;
                        best_area = area;
                    }
                }
            }

            const grid_h = cfg.tile_size / cfg.patch_size;
            const grid_w = cfg.tile_size / cfg.patch_size;
            const per_tile_patches = std.math.mul(u32, grid_h, grid_w) catch return error.Overflow;

            var t_idx: u32 = 0;
            var tok_off: u32 = 0;
            const cell_w = @as(f32, @floatFromInt(src_width)) / @as(f32, @floatFromInt(best_cols));
            const cell_h = @as(f32, @floatFromInt(src_height)) / @as(f32, @floatFromInt(best_rows));

            for (0..best_rows) |ry| {
                for (0..best_cols) |cx| {
                    plan.tiles[t_idx] = .{
                        .src_x0 = @as(f32, @floatFromInt(cx)) * cell_w,
                        .src_y0 = @as(f32, @floatFromInt(ry)) * cell_h,
                        .src_w = cell_w,
                        .src_h = cell_h,
                        .dst_width = cfg.tile_size,
                        .dst_height = cfg.tile_size,
                        .grid_h = grid_h,
                        .grid_w = grid_w,
                        .token_offset = tok_off,
                        .is_thumbnail = false,
                    };
                    t_idx += 1;
                    tok_off += per_tile_patches;
                }
            }
            if (cfg.include_thumbnail and t_idx < MAX_ANYRES_TILES and (best_cols * best_rows > 1)) {
                plan.tiles[t_idx] = .{
                    .src_x0 = 0.0,
                    .src_y0 = 0.0,
                    .src_w = @floatFromInt(src_width),
                    .src_h = @floatFromInt(src_height),
                    .dst_width = cfg.tile_size,
                    .dst_height = cfg.tile_size,
                    .grid_h = grid_h,
                    .grid_w = grid_w,
                    .token_offset = tok_off,
                    .is_thumbnail = true,
                };
                t_idx += 1;
                tok_off += per_tile_patches;
            }
            plan.resized_width = best_cols * cfg.tile_size;
            plan.resized_height = best_rows * cfg.tile_size;
            plan.grid_cols = best_cols;
            plan.grid_rows = best_rows;
            plan.num_tiles = t_idx;
            plan.total_patches = tok_off;
            plan.total_merged_tokens = tok_off / merge_sq;
        },
    }
    return plan;
}

/// Metal 4 uniform structs matching `src/kernels/vision_ingress.metal` byte-for-byte.
pub const VisionAnyResParams = extern struct {
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    src_x0: f32,
    src_y0: f32,
    src_w: f32,
    src_h: f32,
    patch_size: u32,
    t_patch: u32,
    hidden_size: u32,
    is_full_range: u32,
    apply_srgb_to_linear: u32,
    has_bias: u32,
    has_pos_embed: u32,
    token_offset: u32,
    mean_r: f32,
    mean_g: f32,
    mean_b: f32,
    inv_std_r: f32,
    inv_std_g: f32,
    inv_std_b: f32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,

    comptime {
        std.debug.assert(@sizeOf(VisionAnyResParams) == 96);
    }
};

pub const VisionNormParams = extern struct {
    num_tokens: u32,
    dim: u32,
    eps: f32,
    is_rmsnorm: u32,
    has_bias: u32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
    _pad2: u32 = 0,

    comptime {
        std.debug.assert(@sizeOf(VisionNormParams) == 32);
    }
};

pub const VisionRope2DParams = extern struct {
    num_tokens: u32,
    num_heads: u32,
    head_dim: u32,
    qkv_layout: u32,
    rope_interleaved: u32,
    rope_theta: f32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,

    comptime {
        std.debug.assert(@sizeOf(VisionRope2DParams) == 32);
    }
};

pub const VisionAttnParams = extern struct {
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
    scale: f32,
    has_window_bounds: u32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
    _pad2: u32 = 0,

    comptime {
        std.debug.assert(@sizeOf(VisionAttnParams) == 32);
    }
};

pub const VisionLinearParams = extern struct {
    m: u32,
    n: u32,
    k: u32,
    num_groups: u32,
    row_bytes: u32,
    has_bias: u32,
    has_residual: u32,
    act_kind: u32,

    comptime {
        std.debug.assert(@sizeOf(VisionLinearParams) == 32);
    }
};

pub const VisionSwiGluParams = extern struct {
    count: u32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
    _pad2: u32 = 0,

    comptime {
        std.debug.assert(@sizeOf(VisionSwiGluParams) == 16);
    }
};

pub const VisionMergeParams = extern struct {
    grid_h: u32,
    grid_w: u32,
    spatial_merge: u32,
    vis_hidden: u32,
    tile_count: u32,
    _pad0: u32 = 0,
    _pad1: u32 = 0,
    _pad2: u32 = 0,

    comptime {
        std.debug.assert(@sizeOf(VisionMergeParams) == 32);
    }
};

pub const VisionBackend = enum(u8) {
    apple9_simdgroup_mma,
    apple10_nax_tensorops,
};

pub fn selectVisionBackend(info: metal.DeviceInfo) VisionBackend {
    return if (info.supportsNaxTensorOps()) .apple10_nax_tensorops else .apple9_simdgroup_mma;
}

pub const VisionNormKind = enum(u8) {
    layer_norm,
    rms_norm,
};

pub const VisionMlpKind = enum(u8) {
    gelu_tanh,
    gelu_exact,
    swiglu,
};

pub const VisionQkvLayout = enum(u8) {
    /// `[S, 3, H, D]` concatenated `[Q | K | V]` per token (Qwen-VL / Qwen3.8 / DeepSeek-V4.1).
    split_3hd,
    /// `[S, H, 3, D]` head-interleaved QKV per token.
    interleaved_h3d,
};

pub const MergerNormPlacement = enum(u8) {
    none,
    /// Apply `merger.norm` of width `vis_hidden` before spatial merge concat (Qwen3.8 / DeepSeek `vision.norm`).
    pre_merge,
    /// Apply `merger.norm` of width `spatial_merge^2 * vis_hidden` after `2x2` concat (ADR-0023 / Aligner).
    post_merge,
};

pub const VisionLinearAct = enum(u32) {
    none = 0,
    gelu_tanh = 1,
    gelu_exact = 2,
};

pub const VisionLinearWeight = union(enum) {
    dense: struct {
        weight: metal.BufferBinding,
        dtype: quant.FloatDType = .f16,
        bias: ?metal.BufferBinding = null,
    },
    q4_affine: struct {
        qweight: metal.BufferBinding,
        scale_bias: metal.BufferBinding,
        group_size: u32 = 64,
        bias: ?metal.BufferBinding = null,
        bias_dtype: quant.FloatDType = .f16,
    },
};

pub const VisionNormWeight = struct {
    gamma: metal.BufferBinding,
    beta: ?metal.BufferBinding = null,
    dtype: quant.FloatDType = .f16,
    kind: VisionNormKind = .layer_norm,
    eps: f32 = 1e-6,
};

pub const VisionBlockWeights = struct {
    norm1: VisionNormWeight,
    qkv: VisionLinearWeight,
    /// Optional split Q/K/V projections (Inkling 2 `23L` ViT & Hy4 `27L`/`40L` ViT).
    /// When set, `q_proj`/`k_proj`/`v_proj` run instead of fused `qkv` + `qkv_rope_2d`.
    q_proj: ?VisionLinearWeight = null,
    k_proj: ?VisionLinearWeight = null,
    v_proj: ?VisionLinearWeight = null,
    proj: VisionLinearWeight,
    norm2: VisionNormWeight,
    /// `linear_fc1` / `w1` for GELU MLP, or `gate_proj` / `w1` for SwiGLU MLP.
    mlp_fc1: VisionLinearWeight,
    /// Required when `mlp_kind == .swiglu` (`up_proj` / `w3`).
    mlp_up: ?VisionLinearWeight = null,
    /// `linear_fc2` / `w2` for GELU MLP, or `down_proj` / `w2` for SwiGLU MLP.
    mlp_fc2: VisionLinearWeight,
    use_window_attn: bool = false,
};

pub const VisionMergerWeights = struct {
    norm: ?VisionNormWeight = null,
    norm_placement: MergerNormPlacement = .post_merge,
    fc1: VisionLinearWeight,
    /// Optional second linear layer; `null` for single-layer projectors (`vision.merger.proj.0`).
    fc2: ?VisionLinearWeight = null,
    merger_hidden_size: u32,
    act_kind: VisionLinearAct = .gelu_tanh,
};

pub const VisionTowerConfig = struct {
    /// Set `false` for text-only models / turns: allocates `0` wired bytes,
    /// compiles `0` pipelines, and performs `0` GPU dispatches.
    has_vision: bool = true,
    num_layers: u32 = 27,
    vis_hidden: u32 = 1152,
    vis_intermediate: u32 = 4304,
    num_heads: u32 = 16,
    head_dim: u32 = 72,
    lm_hidden: u32 = 2560,
    merger_hidden: u32 = 4608,
    patch_size: u32 = 16,
    t_patch: u32 = 2,
    spatial_merge: u32 = 2,
    max_patches: u32 = 1024,
    mlp_kind: VisionMlpKind = .gelu_tanh,
    qkv_layout: VisionQkvLayout = .split_3hd,
    rope_interleaved: bool = false,
    rope_theta: f32 = 10000.0,

    pub fn validate(self: VisionTowerConfig) Error!void {
        if (!self.has_vision) return;
        if (self.num_layers == 0) return error.InvalidLayerCount;
        if (self.vis_hidden == 0 or self.vis_intermediate == 0 or
            self.lm_hidden == 0 or self.merger_hidden == 0 or self.max_patches == 0)
        {
            return error.InvalidHiddenSize;
        }
        if (self.num_heads == 0 or self.head_dim == 0 or self.head_dim > MAX_VIT_HEAD_DIM or (self.head_dim % 4) != 0) {
            return error.InvalidHeadDim;
        }
        if (self.num_heads * self.head_dim != self.vis_hidden) return error.InvalidHiddenSize;
        if (self.patch_size == 0 or self.patch_size > MAX_PATCH_SIZE) return error.InvalidGeometry;
        if (self.t_patch != 1 and self.t_patch != 2) return error.InvalidGeometry;
        if (self.spatial_merge == 0 or self.spatial_merge > 4) return error.InvalidGeometry;
    }

    pub fn mergedInDim(self: VisionTowerConfig) Error!u32 {
        const sm_sq = std.math.mul(u32, self.spatial_merge, self.spatial_merge) catch return error.Overflow;
        return std.math.mul(u32, sm_sq, self.vis_hidden) catch return error.Overflow;
    }

    pub fn maxMergedTokens(self: VisionTowerConfig) Error!u32 {
        const sm_sq = std.math.mul(u32, self.spatial_merge, self.spatial_merge) catch return error.Overflow;
        return @max(@as(u32, 1), (self.max_patches + sm_sq - 1) / sm_sq);
    }
};

/// Startup-compiled Metal 4 pipelines for the full M7.8 G22 ViT tower.
/// Lazily instantiates `Apple10+` `NAX` TensorOps pipelines only on `gpu_apple_family >= 10`.
pub const VisionTowerPipelines = struct {
    anyres_bgra8: [2][2]metal.Pipeline,
    anyres_nv12: [2][2]metal.Pipeline,
    norm: [2]metal.Pipeline,
    qkv_rope_2d: metal.Pipeline,
    bidir_flash_attn_mma: metal.Pipeline,
    bidir_flash_attn_nax: ?metal.Pipeline,
    linear_mma: [2]metal.Pipeline,
    linear_q4_mma: [2]metal.Pipeline,
    linear_nax: ?[2]metal.Pipeline,
    swiglu: metal.Pipeline,
    spatial_merge_concat: metal.Pipeline,
    backend: VisionBackend,

    pub fn init(device: metal.Device, library: metal.Library) metal.Error!VisionTowerPipelines {
        const ab_ff = try metal.Pipeline.initSpecialized(device, library, "ami_vision_anyres_patchify_embed_bgra8", &.{
            .{ .index = 0, .value = .{ .bool = false } },
            .{ .index = 1, .value = .{ .bool = false } },
        });
        errdefer ab_ff.close();
        const ab_fb = try metal.Pipeline.initSpecialized(device, library, "ami_vision_anyres_patchify_embed_bgra8", &.{
            .{ .index = 0, .value = .{ .bool = false } },
            .{ .index = 1, .value = .{ .bool = true } },
        });
        errdefer ab_fb.close();
        const ab_bf = try metal.Pipeline.initSpecialized(device, library, "ami_vision_anyres_patchify_embed_bgra8", &.{
            .{ .index = 0, .value = .{ .bool = true } },
            .{ .index = 1, .value = .{ .bool = false } },
        });
        errdefer ab_bf.close();
        const ab_bb = try metal.Pipeline.initSpecialized(device, library, "ami_vision_anyres_patchify_embed_bgra8", &.{
            .{ .index = 0, .value = .{ .bool = true } },
            .{ .index = 1, .value = .{ .bool = true } },
        });
        errdefer ab_bb.close();

        const an_ff = try metal.Pipeline.initSpecialized(device, library, "ami_vision_anyres_patchify_embed_nv12", &.{
            .{ .index = 0, .value = .{ .bool = false } },
            .{ .index = 1, .value = .{ .bool = false } },
        });
        errdefer an_ff.close();
        const an_fb = try metal.Pipeline.initSpecialized(device, library, "ami_vision_anyres_patchify_embed_nv12", &.{
            .{ .index = 0, .value = .{ .bool = false } },
            .{ .index = 1, .value = .{ .bool = true } },
        });
        errdefer an_fb.close();
        const an_bf = try metal.Pipeline.initSpecialized(device, library, "ami_vision_anyres_patchify_embed_nv12", &.{
            .{ .index = 0, .value = .{ .bool = true } },
            .{ .index = 1, .value = .{ .bool = false } },
        });
        errdefer an_bf.close();
        const an_bb = try metal.Pipeline.initSpecialized(device, library, "ami_vision_anyres_patchify_embed_nv12", &.{
            .{ .index = 0, .value = .{ .bool = true } },
            .{ .index = 1, .value = .{ .bool = true } },
        });
        errdefer an_bb.close();

        const norm_f16 = try metal.Pipeline.initSpecialized(device, library, "ami_vision_norm", &.{
            .{ .index = 0, .value = .{ .bool = false } },
        });
        errdefer norm_f16.close();
        const norm_bf16 = try metal.Pipeline.initSpecialized(device, library, "ami_vision_norm", &.{
            .{ .index = 0, .value = .{ .bool = true } },
        });
        errdefer norm_bf16.close();

        const qkv_rope = try metal.Pipeline.init(device, library, "ami_vision_qkv_rope_2d");
        errdefer qkv_rope.close();

        const fa_mma = try metal.Pipeline.init(device, library, "ami_vision_bidir_flash_attn_mma");
        errdefer fa_mma.close();

        const lin_f16 = try metal.Pipeline.initSpecialized(device, library, "ami_vision_linear_mma", &.{
            .{ .index = 0, .value = .{ .bool = false } },
        });
        errdefer lin_f16.close();
        const lin_bf16 = try metal.Pipeline.initSpecialized(device, library, "ami_vision_linear_mma", &.{
            .{ .index = 0, .value = .{ .bool = true } },
        });
        errdefer lin_bf16.close();

        const lin_q4_f16 = try metal.Pipeline.initSpecialized(device, library, "ami_vision_linear_q4_mma", &.{
            .{ .index = 0, .value = .{ .bool = false } },
        });
        errdefer lin_q4_f16.close();
        const lin_q4_bf16 = try metal.Pipeline.initSpecialized(device, library, "ami_vision_linear_q4_mma", &.{
            .{ .index = 0, .value = .{ .bool = true } },
        });
        errdefer lin_q4_bf16.close();

        const swiglu_p = try metal.Pipeline.init(device, library, "ami_vision_swiglu");
        errdefer swiglu_p.close();
        const merge_p = try metal.Pipeline.init(device, library, "ami_vision_spatial_merge_concat");
        errdefer merge_p.close();

        const dev_info = try device.info();
        const backend = selectVisionBackend(dev_info);
        var fa_nax: ?metal.Pipeline = null;
        var lin_nax: ?[2]metal.Pipeline = null;
        if (backend == .apple10_nax_tensorops) {
            fa_nax = try metal.Pipeline.init(device, library, "ami_vision_bidir_flash_attn_nax");
            errdefer if (fa_nax) |p| p.close();
            const nax_f16 = try metal.Pipeline.initSpecialized(device, library, "ami_vision_linear_nax", &.{
                .{ .index = 0, .value = .{ .bool = false } },
            });
            errdefer nax_f16.close();
            const nax_bf16 = try metal.Pipeline.initSpecialized(device, library, "ami_vision_linear_nax", &.{
                .{ .index = 0, .value = .{ .bool = true } },
            });
            errdefer nax_bf16.close();
            lin_nax = .{ nax_f16, nax_bf16 };
        }

        return .{
            .anyres_bgra8 = .{ .{ ab_ff, ab_fb }, .{ ab_bf, ab_bb } },
            .anyres_nv12 = .{ .{ an_ff, an_fb }, .{ an_bf, an_bb } },
            .norm = .{ norm_f16, norm_bf16 },
            .qkv_rope_2d = qkv_rope,
            .bidir_flash_attn_mma = fa_mma,
            .bidir_flash_attn_nax = fa_nax,
            .linear_mma = .{ lin_f16, lin_bf16 },
            .linear_q4_mma = .{ lin_q4_f16, lin_q4_bf16 },
            .linear_nax = lin_nax,
            .swiglu = swiglu_p,
            .spatial_merge_concat = merge_p,
            .backend = backend,
        };
    }

    pub fn close(self: VisionTowerPipelines) void {
        for (self.anyres_bgra8) |row| {
            for (row) |p| p.close();
        }
        for (self.anyres_nv12) |row| {
            for (row) |p| p.close();
        }
        for (self.norm) |p| p.close();
        self.qkv_rope_2d.close();
        self.bidir_flash_attn_mma.close();
        if (self.bidir_flash_attn_nax) |p| p.close();
        for (self.linear_mma) |p| p.close();
        for (self.linear_q4_mma) |p| p.close();
        if (self.linear_nax) |pair| {
            for (pair) |p| p.close();
        }
        self.swiglu.close();
        self.spatial_merge_concat.close();
    }

    pub fn encodeAnyResPatchifyEmbedBgra8(
        self: *const VisionTowerPipelines,
        frame: *metal.Compute.Frame,
        frame0: metal.VisionFrame,
        frame1: ?metal.VisionFrame,
        src_width: u32,
        src_height: u32,
        tile: AnyResTileSpec,
        patch_size: u32,
        t_patch: u32,
        hidden_size: u32,
        norm: NormParams,
        weight: metal.BufferBinding,
        bias: ?metal.BufferBinding,
        pos_embed: ?metal.BufferBinding,
        out: metal.BufferBinding,
        weight_dtype: quant.FloatDType,
        out_dtype: quant.FloatDType,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (hidden_size == 0 or tile.dst_width == 0 or tile.dst_height == 0 or
            (tile.dst_width % patch_size) != 0 or (tile.dst_height % patch_size) != 0)
        {
            return error.InvalidGeometry;
        }
        const num_tokens = tile.grid_h * tile.grid_w;
        const f1 = frame1 orelse frame0;
        const params: VisionAnyResParams = .{
            .src_width = src_width,
            .src_height = src_height,
            .dst_width = tile.dst_width,
            .dst_height = tile.dst_height,
            .src_x0 = tile.src_x0,
            .src_y0 = tile.src_y0,
            .src_w = tile.src_w,
            .src_h = tile.src_h,
            .patch_size = patch_size,
            .t_patch = t_patch,
            .hidden_size = hidden_size,
            .is_full_range = 1,
            .apply_srgb_to_linear = if (norm.apply_srgb_to_linear) 1 else 0,
            .has_bias = if (bias != null) 1 else 0,
            .has_pos_embed = if (pos_embed != null) 1 else 0,
            .token_offset = tile.token_offset,
            .mean_r = norm.mean[0],
            .mean_g = norm.mean[1],
            .mean_b = norm.mean[2],
            .inv_std_r = norm.inv_std[0],
            .inv_std_g = norm.inv_std[1],
            .inv_std_b = norm.inv_std[2],
        };
        const groups_x = (hidden_size + EMBED_TG_CHANNELS - 1) / EMBED_TG_CHANNELS;
        try frame.encodeThreadgroupsWithVisionTextures(
            self.anyres_bgra8[@backingInt(weight_dtype)][@backingInt(out_dtype)],
            &.{ weight, bias orelse weight, pos_embed orelse weight, out },
            &.{
                .{ .frame = frame0, .plane = 0 },
                .{ .frame = f1, .plane = 0 },
            },
            std.mem.asBytes(&params),
            .{ groups_x, num_tokens, 1 },
            .{ PATCHIFY_TG_THREADS, 1, 1 },
            insert_barrier_before,
        );
    }

    pub fn encodeAnyResPatchifyEmbedNv12(
        self: *const VisionTowerPipelines,
        frame: *metal.Compute.Frame,
        frame0: metal.VisionFrame,
        frame1: ?metal.VisionFrame,
        src_width: u32,
        src_height: u32,
        tile: AnyResTileSpec,
        patch_size: u32,
        t_patch: u32,
        hidden_size: u32,
        is_full_range: bool,
        norm: NormParams,
        weight: metal.BufferBinding,
        bias: ?metal.BufferBinding,
        pos_embed: ?metal.BufferBinding,
        out: metal.BufferBinding,
        weight_dtype: quant.FloatDType,
        out_dtype: quant.FloatDType,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (hidden_size == 0 or tile.dst_width == 0 or tile.dst_height == 0 or
            (tile.dst_width % patch_size) != 0 or (tile.dst_height % patch_size) != 0 or
            (src_width & 1) != 0 or (src_height & 1) != 0)
        {
            return error.InvalidGeometry;
        }
        const num_tokens = tile.grid_h * tile.grid_w;
        const f1 = frame1 orelse frame0;
        const params: VisionAnyResParams = .{
            .src_width = src_width,
            .src_height = src_height,
            .dst_width = tile.dst_width,
            .dst_height = tile.dst_height,
            .src_x0 = tile.src_x0,
            .src_y0 = tile.src_y0,
            .src_w = tile.src_w,
            .src_h = tile.src_h,
            .patch_size = patch_size,
            .t_patch = t_patch,
            .hidden_size = hidden_size,
            .is_full_range = if (is_full_range) 1 else 0,
            .apply_srgb_to_linear = if (norm.apply_srgb_to_linear) 1 else 0,
            .has_bias = if (bias != null) 1 else 0,
            .has_pos_embed = if (pos_embed != null) 1 else 0,
            .token_offset = tile.token_offset,
            .mean_r = norm.mean[0],
            .mean_g = norm.mean[1],
            .mean_b = norm.mean[2],
            .inv_std_r = norm.inv_std[0],
            .inv_std_g = norm.inv_std[1],
            .inv_std_b = norm.inv_std[2],
        };
        const groups_x = (hidden_size + EMBED_TG_CHANNELS - 1) / EMBED_TG_CHANNELS;
        try frame.encodeThreadgroupsWithVisionTextures(
            self.anyres_nv12[@backingInt(weight_dtype)][@backingInt(out_dtype)],
            &.{ weight, bias orelse weight, pos_embed orelse weight, out },
            &.{
                .{ .frame = frame0, .plane = 0 },
                .{ .frame = frame0, .plane = 1 },
                .{ .frame = f1, .plane = 0 },
                .{ .frame = f1, .plane = 1 },
            },
            std.mem.asBytes(&params),
            .{ groups_x, num_tokens, 1 },
            .{ PATCHIFY_TG_THREADS, 1, 1 },
            insert_barrier_before,
        );
    }

    pub fn encodeNorm(
        self: *const VisionTowerPipelines,
        frame: *metal.Compute.Frame,
        x: metal.BufferBinding,
        w: VisionNormWeight,
        out: metal.BufferBinding,
        num_tokens: u32,
        dim: u32,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (num_tokens == 0 or dim == 0) return error.InvalidGeometry;
        const params: VisionNormParams = .{
            .num_tokens = num_tokens,
            .dim = dim,
            .eps = w.eps,
            .is_rmsnorm = if (w.kind == .rms_norm) 1 else 0,
            .has_bias = if (w.beta != null) 1 else 0,
        };
        try frame.encodeThreadgroups(
            self.norm[@backingInt(w.dtype)],
            &.{ x, w.gamma, w.beta orelse w.gamma, out },
            std.mem.asBytes(&params),
            .{ num_tokens, 1, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
    }

    pub fn encodeQkvRope2D(
        self: *const VisionTowerPipelines,
        frame: *metal.Compute.Frame,
        qkv: metal.BufferBinding,
        coords: metal.BufferBinding,
        q_out: metal.BufferBinding,
        k_out: metal.BufferBinding,
        v_out: metal.BufferBinding,
        num_tokens: u32,
        num_heads: u32,
        head_dim: u32,
        qkv_layout: VisionQkvLayout,
        rope_interleaved: bool,
        rope_theta: f32,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (num_tokens == 0 or num_heads == 0) return error.InvalidGeometry;
        if (head_dim == 0 or head_dim > MAX_VIT_HEAD_DIM or (head_dim % 4) != 0) return error.InvalidHeadDim;
        const params: VisionRope2DParams = .{
            .num_tokens = num_tokens,
            .num_heads = num_heads,
            .head_dim = head_dim,
            .qkv_layout = if (qkv_layout == .split_3hd) 0 else 1,
            .rope_interleaved = if (rope_interleaved) 1 else 0,
            .rope_theta = rope_theta,
        };
        try frame.encodeThreadgroups(
            self.qkv_rope_2d,
            &.{ qkv, coords, q_out, k_out, v_out },
            std.mem.asBytes(&params),
            .{ num_heads, num_tokens, 1 },
            .{ 64, 1, 1 },
            insert_barrier_before,
        );
    }

    pub fn encodeBidirFlashAttn(
        self: *const VisionTowerPipelines,
        frame: *metal.Compute.Frame,
        q: metal.BufferBinding,
        k: metal.BufferBinding,
        v: metal.BufferBinding,
        window_bounds: ?metal.BufferBinding,
        out: metal.BufferBinding,
        seq_len: u32,
        num_heads: u32,
        head_dim: u32,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (seq_len == 0 or num_heads == 0) return error.InvalidGeometry;
        if (head_dim == 0 or head_dim > MAX_VIT_HEAD_DIM) return error.InvalidHeadDim;
        const params: VisionAttnParams = .{
            .seq_len = seq_len,
            .num_heads = num_heads,
            .head_dim = head_dim,
            .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))),
            .has_window_bounds = if (window_bounds != null) 1 else 0,
        };
        const pipe = if (self.backend == .apple10_nax_tensorops and self.bidir_flash_attn_nax != null)
            self.bidir_flash_attn_nax.?
        else
            self.bidir_flash_attn_mma;
        const q_blocks = (seq_len + VIT_FA_BQ - 1) / VIT_FA_BQ;
        try frame.encodeThreadgroups(
            pipe,
            &.{ q, k, v, window_bounds orelse q, out },
            std.mem.asBytes(&params),
            .{ num_heads, q_blocks, 1 },
            .{ 128, 1, 1 },
            insert_barrier_before,
        );
    }

    pub fn encodeLinear(
        self: *const VisionTowerPipelines,
        frame: *metal.Compute.Frame,
        x: metal.BufferBinding,
        w: VisionLinearWeight,
        residual: ?metal.BufferBinding,
        out: metal.BufferBinding,
        m: u32,
        n: u32,
        k: u32,
        act: VisionLinearAct,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (m == 0 or n == 0 or k == 0) return error.InvalidGeometry;
        switch (w) {
            .dense => |d| {
                const params: VisionLinearParams = .{
                    .m = m,
                    .n = n,
                    .k = k,
                    .num_groups = 0,
                    .row_bytes = 0,
                    .has_bias = if (d.bias != null) 1 else 0,
                    .has_residual = if (residual != null) 1 else 0,
                    .act_kind = @backingInt(act),
                };
                const use_nax = self.backend == .apple10_nax_tensorops and self.linear_nax != null and m >= 32 and n >= 32;
                const pipe = if (use_nax)
                    self.linear_nax.?[@backingInt(d.dtype)]
                else
                    self.linear_mma[@backingInt(d.dtype)];
                const tile: u32 = if (use_nax) VIT_LINEAR_NAX_TILE else VIT_LINEAR_TILE;
                try frame.encodeThreadgroups(
                    pipe,
                    &.{ x, d.weight, d.bias orelse d.weight, residual orelse x, out },
                    std.mem.asBytes(&params),
                    .{ (n + tile - 1) / tile, (m + tile - 1) / tile, 1 },
                    .{ 128, 1, 1 },
                    insert_barrier_before,
                );
            },
            .q4_affine => |q4| {
                if (q4.group_size != 64 or (k % 64) != 0) return error.InvalidGeometry;
                const num_groups = k / 64;
                const params: VisionLinearParams = .{
                    .m = m,
                    .n = n,
                    .k = k,
                    .num_groups = num_groups,
                    .row_bytes = num_groups * 32,
                    .has_bias = if (q4.bias != null) 1 else 0,
                    .has_residual = if (residual != null) 1 else 0,
                    .act_kind = @backingInt(act),
                };
                try frame.encodeThreadgroups(
                    self.linear_q4_mma[@backingInt(q4.bias_dtype)],
                    &.{ x, q4.qweight, q4.scale_bias, q4.bias orelse q4.qweight, residual orelse x, out },
                    std.mem.asBytes(&params),
                    .{ (n + VIT_LINEAR_TILE - 1) / VIT_LINEAR_TILE, (m + VIT_LINEAR_TILE - 1) / VIT_LINEAR_TILE, 1 },
                    .{ 128, 1, 1 },
                    insert_barrier_before,
                );
            },
        }
    }

    pub fn encodeSwiGlu(
        self: *const VisionTowerPipelines,
        frame: *metal.Compute.Frame,
        gate: metal.BufferBinding,
        up: metal.BufferBinding,
        out: metal.BufferBinding,
        count: u32,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (count == 0) return error.InvalidGeometry;
        const params: VisionSwiGluParams = .{ .count = count };
        try frame.encodeThreads(
            self.swiglu,
            &.{ gate, up, out },
            std.mem.asBytes(&params),
            .{ count, 1, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
    }

    pub fn encodeSpatialMergeConcat(
        self: *const VisionTowerPipelines,
        frame: *metal.Compute.Frame,
        in_tokens: metal.BufferBinding,
        out_merged: metal.BufferBinding,
        grid_h: u32,
        grid_w: u32,
        spatial_merge: u32,
        vis_hidden: u32,
        tile_count: u32,
        insert_barrier_before: bool,
    ) EncodeError!void {
        if (grid_h == 0 or grid_w == 0 or spatial_merge == 0 or vis_hidden == 0 or tile_count == 0 or
            (grid_h % spatial_merge) != 0 or (grid_w % spatial_merge) != 0)
        {
            return error.InvalidGeometry;
        }
        const total_merged = tile_count * (grid_h / spatial_merge) * (grid_w / spatial_merge);
        const params: VisionMergeParams = .{
            .grid_h = grid_h,
            .grid_w = grid_w,
            .spatial_merge = spatial_merge,
            .vis_hidden = vis_hidden,
            .tile_count = tile_count,
        };
        try frame.encodeThreadgroups(
            self.spatial_merge_concat,
            &.{ in_tokens, out_merged },
            std.mem.asBytes(&params),
            .{ total_merged, 1, 1 },
            .{ 256, 1, 1 },
            insert_barrier_before,
        );
    }
};

/// Preallocated wired scratch for executing the full `L_vis`-layer ViT tower +
/// PatchMerger / Aligner with zero hot-path heap allocations.
///
/// Designed for `WiredArena.allocAliasedPair(vision_bytes, prefill_bytes)` so
/// Phase-2 `m78-engine` aliases vision scratch with language prefill scratch for
/// `0` extra steady-state wired RAM during text decode.
pub const VisionTowerScratch = struct {
    config: VisionTowerConfig,
    owned_arena: ?arena_mod.WiredArena = null,
    owns_gpu_buf: bool = false,
    gpu_buf: metal.Buffer,
    base_offset: usize,
    total_bytes: usize,

    // Sub-buffer byte offsets relative to `gpu_buf` (all 64 B aligned).
    hidden_a_off: usize,
    hidden_b_off: usize,
    norm_off: usize,
    qkv_off: usize,
    q_off: usize,
    k_off: usize,
    v_off: usize,
    attn_out_off: usize,
    mlp_a_off: usize,
    mlp_b_off: usize,
    merger_concat_off: usize,
    merger_mid_off: usize,
    lm_embeds_off: usize,
    coords_off: usize,
    window_bounds_off: usize,

    // Host-visible slices for zero-copy CPU inspection and coordinate writes.
    hidden_a_raw: []u8,
    lm_embeds_raw: []u8,
    coords_raw: []u8,
    window_bounds_raw: []u8,

    const SliceSizes = struct {
        hidden_bytes: usize,
        qkv_bytes: usize,
        mlp_bytes: usize,
        merger_concat_bytes: usize,
        merger_mid_bytes: usize,
        lm_embeds_bytes: usize,
        coords_bytes: usize,
        window_bytes: usize,
        total_aligned: usize,
    };

    fn computeSizes(cfg: VisionTowerConfig) Error!SliceSizes {
        try cfg.validate();
        const s: usize = cfg.max_patches;
        const dv: usize = cfg.vis_hidden;
        const df: usize = cfg.vis_intermediate;
        const s_merged: usize = try cfg.maxMergedTokens();
        const m_in: usize = try cfg.mergedInDim();
        const m_hid: usize = cfg.merger_hidden;
        const dl: usize = cfg.lm_hidden;

        const hidden_bytes = std.math.mul(usize, std.math.mul(usize, s, dv) catch return error.Overflow, 2) catch return error.Overflow;
        const qkv_bytes = std.math.mul(usize, hidden_bytes, 3) catch return error.Overflow;
        const mlp_bytes = std.math.mul(usize, std.math.mul(usize, s, df) catch return error.Overflow, 2) catch return error.Overflow;
        const merger_concat_bytes = std.math.mul(usize, std.math.mul(usize, s_merged, m_in) catch return error.Overflow, 2) catch return error.Overflow;
        const merger_mid_bytes = std.math.mul(usize, std.math.mul(usize, s_merged, m_hid) catch return error.Overflow, 2) catch return error.Overflow;
        const lm_embeds_bytes = std.math.mul(usize, std.math.mul(usize, s_merged, dl) catch return error.Overflow, 2) catch return error.Overflow;
        const coords_bytes = std.math.mul(usize, s, @sizeOf(RopeCoord2D)) catch return error.Overflow;
        const window_bytes = std.math.mul(usize, s, @sizeOf(AttnWindowBound)) catch return error.Overflow;

        // 15 sub-buffers: hidden_a, hidden_b, norm, qkv, q, k, v, attn_out,
        // mlp_a, mlp_b, merger_concat, merger_mid, lm_embeds, coords, window_bounds.
        const chunks = [_]usize{
            hidden_bytes, // hidden_a
            hidden_bytes, // hidden_b
            hidden_bytes, // norm
            qkv_bytes, // qkv
            hidden_bytes, // q
            hidden_bytes, // k
            hidden_bytes, // v
            hidden_bytes, // attn_out
            mlp_bytes, // mlp_a
            mlp_bytes, // mlp_b
            merger_concat_bytes, // merger_concat
            merger_mid_bytes, // merger_mid
            lm_embeds_bytes, // lm_embeds
            coords_bytes, // coords
            window_bytes, // window_bounds
        };
        var total: usize = 0;
        for (chunks) |c| {
            const aligned = (std.math.add(usize, c, 63) catch return error.Overflow) & ~@as(usize, 63);
            total = std.math.add(usize, total, aligned) catch return error.Overflow;
        }
        return .{
            .hidden_bytes = hidden_bytes,
            .qkv_bytes = qkv_bytes,
            .mlp_bytes = mlp_bytes,
            .merger_concat_bytes = merger_concat_bytes,
            .merger_mid_bytes = merger_mid_bytes,
            .lm_embeds_bytes = lm_embeds_bytes,
            .coords_bytes = coords_bytes,
            .window_bytes = window_bytes,
            .total_aligned = page.alignUp(total) catch return error.Overflow,
        };
    }

    /// Returns exact 16 KiB-aligned scratch size in bytes (`0` when `!cfg.has_vision`).
    pub fn requiredBytes(cfg: VisionTowerConfig) Error!usize {
        if (!cfg.has_vision) return 0;
        return (try computeSizes(cfg)).total_aligned;
    }

    /// Carves all ViT scratch sub-buffers out of an existing `WiredArena` or
    /// `WiredArena.allocAliasedPair` sub-arena backed by `gpu_buf` at `base_offset`.
    /// Owns zero mappings (`0` extra wired RAM when aliased with prefill scratch).
    pub fn initFromSubArena(
        sub_arena: *arena_mod.WiredArena,
        gpu_buf: metal.Buffer,
        base_offset: usize,
        cfg: VisionTowerConfig,
    ) (Error || arena_mod.WiredArena.AllocError)!VisionTowerScratch {
        const sz = try computeSizes(cfg);
        if (sub_arena.remaining() < sz.total_aligned) return error.AliasedArenaTooSmall;

        const hidden_a = try sub_arena.allocAligned(sz.hidden_bytes, 64);
        const hidden_b = try sub_arena.allocAligned(sz.hidden_bytes, 64);
        const norm_s = try sub_arena.allocAligned(sz.hidden_bytes, 64);
        const qkv_s = try sub_arena.allocAligned(sz.qkv_bytes, 64);
        const q_s = try sub_arena.allocAligned(sz.hidden_bytes, 64);
        const k_s = try sub_arena.allocAligned(sz.hidden_bytes, 64);
        const v_s = try sub_arena.allocAligned(sz.hidden_bytes, 64);
        const attn_s = try sub_arena.allocAligned(sz.hidden_bytes, 64);
        const mlp_a = try sub_arena.allocAligned(sz.mlp_bytes, 64);
        const mlp_b = try sub_arena.allocAligned(sz.mlp_bytes, 64);
        const m_cat = try sub_arena.allocAligned(sz.merger_concat_bytes, 64);
        const m_mid = try sub_arena.allocAligned(sz.merger_mid_bytes, 64);
        const lm_emb = try sub_arena.allocAligned(sz.lm_embeds_bytes, 64);
        const coords_s = try sub_arena.allocAligned(sz.coords_bytes, 64);
        const win_s = try sub_arena.allocAligned(sz.window_bytes, 64);

        return .{
            .config = cfg,
            .owned_arena = null,
            .owns_gpu_buf = false,
            .gpu_buf = gpu_buf,
            .base_offset = base_offset,
            .total_bytes = sz.total_aligned,
            .hidden_a_off = base_offset + sub_arena.offsetOf(hidden_a.ptr),
            .hidden_b_off = base_offset + sub_arena.offsetOf(hidden_b.ptr),
            .norm_off = base_offset + sub_arena.offsetOf(norm_s.ptr),
            .qkv_off = base_offset + sub_arena.offsetOf(qkv_s.ptr),
            .q_off = base_offset + sub_arena.offsetOf(q_s.ptr),
            .k_off = base_offset + sub_arena.offsetOf(k_s.ptr),
            .v_off = base_offset + sub_arena.offsetOf(v_s.ptr),
            .attn_out_off = base_offset + sub_arena.offsetOf(attn_s.ptr),
            .mlp_a_off = base_offset + sub_arena.offsetOf(mlp_a.ptr),
            .mlp_b_off = base_offset + sub_arena.offsetOf(mlp_b.ptr),
            .merger_concat_off = base_offset + sub_arena.offsetOf(m_cat.ptr),
            .merger_mid_off = base_offset + sub_arena.offsetOf(m_mid.ptr),
            .lm_embeds_off = base_offset + sub_arena.offsetOf(lm_emb.ptr),
            .coords_off = base_offset + sub_arena.offsetOf(coords_s.ptr),
            .window_bounds_off = base_offset + sub_arena.offsetOf(win_s.ptr),
            .hidden_a_raw = hidden_a,
            .lm_embeds_raw = lm_emb,
            .coords_raw = coords_s,
            .window_bounds_raw = win_s,
        };
    }

    pub const StandaloneError = Error || arena_mod.WiredArena.InitError || arena_mod.WiredArena.AllocError || metal.WrapError;

    pub fn initStandalone(
        device: metal.Device,
        budget: budget_mod.Budget,
        cfg: VisionTowerConfig,
    ) StandaloneError!VisionTowerScratch {
        const total = try requiredBytes(cfg);
        var arena = try arena_mod.WiredArena.init(total, budget);
        errdefer arena.deinit();
        const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
        errdefer gpu_buf.close();
        var s = try initFromSubArena(&arena, gpu_buf, 0, cfg);
        s.owned_arena = arena;
        s.owns_gpu_buf = true;
        return s;
    }

    pub fn deinit(self: *VisionTowerScratch) void {
        if (self.owns_gpu_buf) {
            self.gpu_buf.close();
            self.owns_gpu_buf = false;
        }
        if (self.owned_arena) |*a| {
            a.deinit();
            self.owned_arena = null;
        }
    }

    pub fn hiddenABinding(self: *const VisionTowerScratch) metal.BufferBinding {
        return .{ .buffer = self.gpu_buf, .offset = self.hidden_a_off };
    }
    pub fn hiddenBBinding(self: *const VisionTowerScratch) metal.BufferBinding {
        return .{ .buffer = self.gpu_buf, .offset = self.hidden_b_off };
    }
    pub fn normBinding(self: *const VisionTowerScratch) metal.BufferBinding {
        return .{ .buffer = self.gpu_buf, .offset = self.norm_off };
    }
    pub fn qkvBinding(self: *const VisionTowerScratch) metal.BufferBinding {
        return .{ .buffer = self.gpu_buf, .offset = self.qkv_off };
    }
    pub fn qBinding(self: *const VisionTowerScratch) metal.BufferBinding {
        return .{ .buffer = self.gpu_buf, .offset = self.q_off };
    }
    pub fn kBinding(self: *const VisionTowerScratch) metal.BufferBinding {
        return .{ .buffer = self.gpu_buf, .offset = self.k_off };
    }
    pub fn vBinding(self: *const VisionTowerScratch) metal.BufferBinding {
        return .{ .buffer = self.gpu_buf, .offset = self.v_off };
    }
    pub fn attnOutBinding(self: *const VisionTowerScratch) metal.BufferBinding {
        return .{ .buffer = self.gpu_buf, .offset = self.attn_out_off };
    }
    pub fn mlpABinding(self: *const VisionTowerScratch) metal.BufferBinding {
        return .{ .buffer = self.gpu_buf, .offset = self.mlp_a_off };
    }
    pub fn mlpBBinding(self: *const VisionTowerScratch) metal.BufferBinding {
        return .{ .buffer = self.gpu_buf, .offset = self.mlp_b_off };
    }
    pub fn mergerConcatBinding(self: *const VisionTowerScratch) metal.BufferBinding {
        return .{ .buffer = self.gpu_buf, .offset = self.merger_concat_off };
    }
    pub fn mergerMidBinding(self: *const VisionTowerScratch) metal.BufferBinding {
        return .{ .buffer = self.gpu_buf, .offset = self.merger_mid_off };
    }
    pub fn lmEmbedsBinding(self: *const VisionTowerScratch) metal.BufferBinding {
        return .{ .buffer = self.gpu_buf, .offset = self.lm_embeds_off };
    }
    pub fn coordsBinding(self: *const VisionTowerScratch) metal.BufferBinding {
        return .{ .buffer = self.gpu_buf, .offset = self.coords_off };
    }
    pub fn windowBoundsBinding(self: *const VisionTowerScratch) metal.BufferBinding {
        return .{ .buffer = self.gpu_buf, .offset = self.window_bounds_off };
    }

    pub fn hiddenAF16(self: *const VisionTowerScratch) []f16 {
        return @alignCast(std.mem.bytesAsSlice(f16, self.hidden_a_raw));
    }
    pub fn lmEmbedsF16(self: *const VisionTowerScratch) []f16 {
        return @alignCast(std.mem.bytesAsSlice(f16, self.lm_embeds_raw));
    }
    pub fn coordsSlice(self: *const VisionTowerScratch) []RopeCoord2D {
        return @alignCast(std.mem.bytesAsSlice(RopeCoord2D, self.coords_raw));
    }
    pub fn windowBoundsSlice(self: *const VisionTowerScratch) []AttnWindowBound {
        return @alignCast(std.mem.bytesAsSlice(AttnWindowBound, self.window_bounds_raw));
    }
};

// ===========================================================================
// C16 / D5 CoreML ANE Vision-Tower Offload & Startup Placement Calibration
// ===========================================================================

pub const VisionPlacement = enum(u8) {
    auto,
    gpu,
    ane,
};

pub const VisionAneCalibration = struct {
    selected: VisionPlacement,
    ane_us: f64,
    gpu_us: f64,
    token_bucket: u32,
    chip_has_nax: bool,
    ane_available: bool,
};

/// Pure `C16` / `D5` startup placement selector choosing ANE vs Metal 4 GPU for a
/// vision-tower static token bucket. On `Apple10+` (`chip.has_nax == true`), ANE
/// must beat GPU by at least `10%` to override GPU `NAX` TensorOps (`ADR-0021`, `ADR-0023`).
pub fn selectVisionPlacement(
    chip: chip_profile.ChipProfile,
    has_coreml_vision: bool,
    num_tokens: u32,
    ane_us: f64,
    gpu_us: f64,
    policy: VisionPlacement,
) VisionPlacement {
    if (!has_coreml_vision or num_tokens == 0) return .gpu;
    return switch (policy) {
        .gpu => .gpu,
        .ane => .ane,
        .auto => blk: {
            if (ane_us <= 0.0 or gpu_us <= 0.0) break :blk .gpu;
            const margin: f64 = if (chip.has_nax) 0.90 else 0.98;
            break :blk if (ane_us < gpu_us * margin) .ane else .gpu;
        },
    };
}

/// Zero-copy CoreML ANE vision-tower runner bound over `G14` `OneComponent16Half`
/// `IOSurface` frames (`patch_tokens` -> `visual_embeds`), matching `export-coreml --component vision-tower`.
pub const VisionAneOffload = struct {
    model: metal.AneModel,
    in_frame: metal.VisionFrame,
    out_frame: metal.VisionFrame,
    in_tensor: metal.AneTensor,
    out_tensor: metal.AneTensor,
    plan: metal.AnePlan,
    token_bucket: u32,
    vis_hidden: u32,
    out_hidden: u32,
    is_conv1x1: bool,

    pub const InitError = Error || metal.Error;

    fn bindAndInit(
        device: metal.Device,
        model: metal.AneModel,
        token_bucket: u32,
        vis_hidden: u32,
        out_hidden: u32,
        is_conv1x1: bool,
        in_name: [:0]const u8,
        out_name: [:0]const u8,
    ) InitError!VisionAneOffload {
        if (token_bucket == 0 or vis_hidden == 0 or out_hidden == 0) return error.AneShapeMismatch;
        if (is_conv1x1) {
            if ((token_bucket % IOSURFACE_F16_ROW_ALIGN) != 0) return error.AneShapeMismatch;
        } else {
            if ((vis_hidden % IOSURFACE_F16_ROW_ALIGN) != 0 or (out_hidden % IOSURFACE_F16_ROW_ALIGN) != 0) {
                return error.AneShapeMismatch;
            }
        }
        // `VisionFrame(.f16, width, height)` has row stride = width fp16 elements (aligned to 32).
        const in_w = if (is_conv1x1) token_bucket else vis_hidden;
        const in_h = if (is_conv1x1) vis_hidden else token_bucket;
        const out_w = if (is_conv1x1) token_bucket else out_hidden;
        const out_h = if (is_conv1x1) out_hidden else token_bucket;

        const in_frame = try metal.VisionFrame.init(device, .f16, in_w, in_h);
        errdefer in_frame.close();
        const out_frame = try metal.VisionFrame.init(device, .f16, out_w, out_h);
        errdefer out_frame.close();

        const in_tensor = if (is_conv1x1)
            try in_frame.toAneTensor(&.{ vis_hidden, 1, token_bucket })
        else
            try in_frame.toAneTensor(&.{ token_bucket, vis_hidden });
        errdefer in_tensor.close();

        const out_tensor = if (is_conv1x1)
            try out_frame.toAneTensor(&.{ out_hidden, 1, token_bucket })
        else
            try out_frame.toAneTensor(&.{ token_bucket, out_hidden });
        errdefer out_tensor.close();

        const plan = try metal.AnePlan.init(
            &.{.{ .name = in_name, .tensor = in_tensor }},
            &.{.{ .name = out_name, .tensor = out_tensor }},
        );
        errdefer plan.close();

        return .{
            .model = model,
            .in_frame = in_frame,
            .out_frame = out_frame,
            .in_tensor = in_tensor,
            .out_tensor = out_tensor,
            .plan = plan,
            .token_bucket = token_bucket,
            .vis_hidden = vis_hidden,
            .out_hidden = out_hidden,
            .is_conv1x1 = is_conv1x1,
        };
    }

    pub fn initFromPath(
        device: metal.Device,
        mlpackage_path: [:0]const u8,
        token_bucket: u32,
        vis_hidden: u32,
        out_hidden: u32,
        is_conv1x1: bool,
    ) InitError!VisionAneOffload {
        const model = try metal.AneModel.loadPath(mlpackage_path, .cpu_and_neural_engine);
        errdefer model.close();
        return bindAndInit(device, model, token_bucket, vis_hidden, out_hidden, is_conv1x1, "patch_tokens", "visual_embeds");
    }

    pub fn initFromSpecBytes(
        device: metal.Device,
        spec_bytes: []const u8,
        token_bucket: u32,
        vis_hidden: u32,
        out_hidden: u32,
        is_conv1x1: bool,
        in_name: [:0]const u8,
        out_name: [:0]const u8,
    ) InitError!VisionAneOffload {
        const model = try metal.AneModel.loadSpecData(spec_bytes, .cpu_and_neural_engine);
        errdefer model.close();
        return bindAndInit(device, model, token_bucket, vis_hidden, out_hidden, is_conv1x1, in_name, out_name);
    }

    pub fn deinit(self: *VisionAneOffload) void {
        self.plan.close();
        self.out_tensor.close();
        self.in_tensor.close();
        self.out_frame.close();
        self.in_frame.close();
        self.model.close();
    }

    pub fn predictZeroCopy(self: *const VisionAneOffload) metal.Error!metal.AnePredictInfo {
        return self.model.predict(self.plan);
    }

    /// Runs startup `D5` / `C16` micro-calibration measuring median zero-copy ANE
    /// latency against a measured Metal 4 GPU vision latency (`gpu_us`).
    pub fn calibrateAgainstGpuUs(
        self: *const VisionAneOffload,
        chip: chip_profile.ChipProfile,
        gpu_us: f64,
        warmup_iters: u32,
        measure_iters: u32,
        policy: VisionPlacement,
    ) (metal.Error || vmstat.Error)!VisionAneCalibration {
        const eff_iters: u32 = @max(@as(u32, 1), @min(measure_iters, 32));
        for (0..warmup_iters) |_| {
            _ = try self.model.predict(self.plan);
        }
        var samples: [32]f64 = undefined;
        for (0..eff_iters) |i| {
            const t0 = try vmstat.nowNs();
            _ = try self.model.predict(self.plan);
            const t1 = try vmstat.nowNs();
            samples[i] = @as(f64, @floatFromInt(t1 - t0)) / 1000.0;
        }
        // Insertion sort small sample window for exact median (zero heap allocation).
        for (1..eff_iters) |i| {
            const key = samples[i];
            var j = i;
            while (j > 0 and samples[j - 1] > key) : (j -= 1) {
                samples[j] = samples[j - 1];
            }
            samples[j] = key;
        }
        const ane_us = samples[eff_iters / 2];
        const selected = selectVisionPlacement(chip, true, self.token_bucket, ane_us, gpu_us, policy);
        return .{
            .selected = selected,
            .ane_us = ane_us,
            .gpu_us = gpu_us,
            .token_bucket = self.token_bucket,
            .chip_has_nax = chip.has_nax,
            .ane_available = true,
        };
    }
};

// ===========================================================================
// High-Level Zero-Overhead `VisionTowerRuntime` (G22)
// ===========================================================================

pub const VisionTowerRuntime = struct {
    has_vision: bool,
    config: VisionTowerConfig,
    pipelines: ?VisionTowerPipelines,
    scratch: ?VisionTowerScratch,
    placement: VisionPlacement,
    ane_calibration: ?VisionAneCalibration,

    /// Zero-overhead text-only fast path: allocates `0` wired bytes, compiles `0`
    /// Metal pipelines, and dispatches `0` GPU commands.
    pub fn initTextOnly() VisionTowerRuntime {
        return .{
            .has_vision = false,
            .config = .{ .has_vision = false },
            .pipelines = null,
            .scratch = null,
            .placement = .gpu,
            .ane_calibration = null,
        };
    }

    pub const InitError = VisionTowerScratch.StandaloneError;

    /// Initializes the full ViT tower runtime with standalone wired scratch, or
    /// returns the zero-overhead text-only fast path immediately if `!cfg.has_vision`.
    pub fn initStandalone(
        device: metal.Device,
        library: metal.Library,
        budget: budget_mod.Budget,
        cfg: VisionTowerConfig,
    ) InitError!VisionTowerRuntime {
        if (!cfg.has_vision) return initTextOnly();
        try cfg.validate();
        const pipes = try VisionTowerPipelines.init(device, library);
        errdefer pipes.close();
        var scratch = try VisionTowerScratch.initStandalone(device, budget, cfg);
        errdefer scratch.deinit();
        return .{
            .has_vision = true,
            .config = cfg,
            .pipelines = pipes,
            .scratch = scratch,
            .placement = .gpu,
            .ane_calibration = null,
        };
    }

    /// Initializes the ViT tower runtime over a caller-provided aliased sub-arena
    /// (from `WiredArena.allocAliasedPair`), consuming `0` extra steady-state wired RAM.
    pub fn initAliased(
        device: metal.Device,
        library: metal.Library,
        sub_arena: *arena_mod.WiredArena,
        gpu_buf: metal.Buffer,
        base_offset: usize,
        cfg: VisionTowerConfig,
    ) (Error || metal.Error || arena_mod.WiredArena.AllocError)!VisionTowerRuntime {
        if (!cfg.has_vision) return initTextOnly();
        try cfg.validate();
        const pipes = try VisionTowerPipelines.init(device, library);
        errdefer pipes.close();
        const scratch = try VisionTowerScratch.initFromSubArena(sub_arena, gpu_buf, base_offset, cfg);
        return .{
            .has_vision = true,
            .config = cfg,
            .pipelines = pipes,
            .scratch = scratch,
            .placement = .gpu,
            .ane_calibration = null,
        };
    }

    pub fn deinit(self: *VisionTowerRuntime) void {
        if (self.scratch) |*s| {
            s.deinit();
            self.scratch = null;
        }
        if (self.pipelines) |p| {
            p.close();
            self.pipelines = null;
        }
        self.has_vision = false;
    }

    pub fn isActive(self: *const VisionTowerRuntime) bool {
        return self.has_vision and self.pipelines != null and self.scratch != null;
    }

    /// Returns steady-state extra wired RAM owned by this runtime (`0` for text-only
    /// or `WiredArena.allocAliasedPair`-aliased instances).
    pub fn ownedWiredBytes(self: *const VisionTowerRuntime) usize {
        if (!self.has_vision) return 0;
        const s = self.scratch orelse return 0;
        return if (s.owned_arena != null) s.total_bytes else 0;
    }

    /// Encodes one ViT encoder block:
    /// `norm1 -> qkv -> 2D-RoPE -> bidirectional FlashAttention-2 -> attn.proj + res -> norm2 -> GELU/SwiGLU MLP + res`.
    pub fn encodeBlock(
        self: *const VisionTowerRuntime,
        frame: *metal.Compute.Frame,
        block: VisionBlockWeights,
        in_hidden: metal.BufferBinding,
        out_hidden: metal.BufferBinding,
        num_patches: u32,
        insert_barrier_before: bool,
    ) EncodeError!void {
        const pipes = self.pipelines orelse return error.InvalidBlockWeights;
        const sc = self.scratch orelse return error.InvalidBlockWeights;
        const cfg = self.config;
        if (num_patches == 0 or num_patches > cfg.max_patches) return error.InvalidGeometry;

        // 1. Pre-attention LayerNorm / RMSNorm -> sc.norm
        try pipes.encodeNorm(frame, in_hidden, block.norm1, sc.normBinding(), num_patches, cfg.vis_hidden, insert_barrier_before);
        // 2-3. QKV projection (split Q/K/V for Inkling 2 / Hy4 ViTs, or fused QKV + 2D-RoPE).
        if (block.q_proj != null and block.k_proj != null and block.v_proj != null) {
            try pipes.encodeLinear(frame, sc.normBinding(), block.q_proj.?, null, sc.qBinding(), num_patches, cfg.vis_hidden, cfg.vis_hidden, .none, true);
            try pipes.encodeLinear(frame, sc.normBinding(), block.k_proj.?, null, sc.kBinding(), num_patches, cfg.vis_hidden, cfg.vis_hidden, .none, false);
            try pipes.encodeLinear(frame, sc.normBinding(), block.v_proj.?, null, sc.vBinding(), num_patches, cfg.vis_hidden, cfg.vis_hidden, .none, false);
        } else {
            // 2. QKV linear projection: [S, D] -> [S, 3*D]
            try pipes.encodeLinear(frame, sc.normBinding(), block.qkv, null, sc.qkvBinding(), num_patches, 3 * cfg.vis_hidden, cfg.vis_hidden, .none, true);
            // 3. Fused QKV split + 2D-RoPE -> Q, K, V
            try pipes.encodeQkvRope2D(
                frame,
                sc.qkvBinding(),
                sc.coordsBinding(),
                sc.qBinding(),
                sc.kBinding(),
                sc.vBinding(),
                num_patches,
                cfg.num_heads,
                cfg.head_dim,
                cfg.qkv_layout,
                cfg.rope_interleaved,
                cfg.rope_theta,
                true,
            );
        }
        // 4. Bidirectional full/windowed FlashAttention-2 (MMA / NAX) -> sc.attn_out
        const win_opt: ?metal.BufferBinding = if (block.use_window_attn) sc.windowBoundsBinding() else null;
        try pipes.encodeBidirFlashAttn(
            frame,
            sc.qBinding(),
            sc.kBinding(),
            sc.vBinding(),
            win_opt,
            sc.attnOutBinding(),
            num_patches,
            cfg.num_heads,
            cfg.head_dim,
            true,
        );
        // 5. Attention output projection + fused residual add -> out_hidden (intermediate residual state)
        try pipes.encodeLinear(frame, sc.attnOutBinding(), block.proj, in_hidden, out_hidden, num_patches, cfg.vis_hidden, cfg.vis_hidden, .none, true);
        // 6. Pre-MLP LayerNorm / RMSNorm -> sc.norm
        try pipes.encodeNorm(frame, out_hidden, block.norm2, sc.normBinding(), num_patches, cfg.vis_hidden, true);
        // 7. MLP (GELU or SwiGLU) + fused residual add -> out_hidden (writing back to out_hidden via in-place residual read)
        switch (cfg.mlp_kind) {
            .gelu_tanh, .gelu_exact => {
                const act: VisionLinearAct = if (cfg.mlp_kind == .gelu_tanh) .gelu_tanh else .gelu_exact;
                try pipes.encodeLinear(frame, sc.normBinding(), block.mlp_fc1, null, sc.mlpABinding(), num_patches, cfg.vis_intermediate, cfg.vis_hidden, act, true);
                try pipes.encodeLinear(frame, sc.mlpABinding(), block.mlp_fc2, out_hidden, out_hidden, num_patches, cfg.vis_hidden, cfg.vis_intermediate, .none, true);
            },
            .swiglu => {
                const up_w = block.mlp_up orelse return error.InvalidBlockWeights;
                try pipes.encodeLinear(frame, sc.normBinding(), block.mlp_fc1, null, sc.mlpABinding(), num_patches, cfg.vis_intermediate, cfg.vis_hidden, .none, true);
                try pipes.encodeLinear(frame, sc.normBinding(), up_w, null, sc.mlpBBinding(), num_patches, cfg.vis_intermediate, cfg.vis_hidden, .none, false);
                try pipes.encodeSwiGlu(frame, sc.mlpABinding(), sc.mlpBBinding(), sc.mlpABinding(), num_patches * cfg.vis_intermediate, true);
                try pipes.encodeLinear(frame, sc.mlpABinding(), block.mlp_fc2, out_hidden, out_hidden, num_patches, cfg.vis_hidden, cfg.vis_intermediate, .none, true);
            },
        }
    }

    /// Encodes the 2-layer `PatchMerger` / `Aligner` MLP:
    /// `[pre_merge norm] -> spatial_merge concat -> [post_merge norm] -> Linear + GELU -> Linear -> lm_embeds`.
    pub fn encodePatchMerger(
        self: *const VisionTowerRuntime,
        frame: *metal.Compute.Frame,
        merger: VisionMergerWeights,
        in_hidden: metal.BufferBinding,
        grid_h: u32,
        grid_w: u32,
        tile_count: u32,
        insert_barrier_before: bool,
    ) EncodeError!u32 {
        const pipes = self.pipelines orelse return error.InvalidBlockWeights;
        const sc = self.scratch orelse return error.InvalidBlockWeights;
        const cfg = self.config;
        const num_patches = tile_count * grid_h * grid_w;
        if (num_patches == 0 or num_patches > cfg.max_patches) return error.InvalidGeometry;
        const merged_tokens = tile_count * (grid_h / cfg.spatial_merge) * (grid_w / cfg.spatial_merge);
        const merged_in_dim = try cfg.mergedInDim();

        var pre_concat_src = in_hidden;
        var need_barrier = insert_barrier_before;
        if (merger.norm_placement == .pre_merge) {
            if (merger.norm) |nw| {
                try pipes.encodeNorm(frame, in_hidden, nw, sc.normBinding(), num_patches, cfg.vis_hidden, need_barrier);
                pre_concat_src = sc.normBinding();
                need_barrier = true;
            }
        }

        // Spatial `spatial_merge x spatial_merge` patch token concatenation.
        try pipes.encodeSpatialMergeConcat(
            frame,
            pre_concat_src,
            sc.mergerConcatBinding(),
            grid_h,
            grid_w,
            cfg.spatial_merge,
            cfg.vis_hidden,
            tile_count,
            need_barrier,
        );

        var fc1_in = sc.mergerConcatBinding();
        if (merger.norm_placement == .post_merge) {
            if (merger.norm) |nw| {
                // In-place post-merge LayerNorm over `[merged_tokens, spatial_merge^2 * vis_hidden]`.
                try pipes.encodeNorm(frame, sc.mergerConcatBinding(), nw, sc.mergerConcatBinding(), merged_tokens, merged_in_dim, true);
                fc1_in = sc.mergerConcatBinding();
            }
        }

        // 1-layer or 2-layer PatchMerger / Aligner MLP.
        if (merger.fc2) |fc2| {
            try pipes.encodeLinear(
                frame,
                fc1_in,
                merger.fc1,
                null,
                sc.mergerMidBinding(),
                merged_tokens,
                merger.merger_hidden_size,
                merged_in_dim,
                merger.act_kind,
                true,
            );
            try pipes.encodeLinear(
                frame,
                sc.mergerMidBinding(),
                fc2,
                null,
                sc.lmEmbedsBinding(),
                merged_tokens,
                cfg.lm_hidden,
                merger.merger_hidden_size,
                .none,
                true,
            );
        } else {
            try pipes.encodeLinear(
                frame,
                fc1_in,
                merger.fc1,
                null,
                sc.lmEmbedsBinding(),
                merged_tokens,
                cfg.lm_hidden,
                merged_in_dim,
                .none,
                true,
            );
        }
        return merged_tokens;
    }

    /// Executes the full `L_vis`-layer ViT encoder tower + `PatchMerger` / `Aligner`
    /// starting from patch embeddings staged in `scratch.hiddenABinding()` (or populated
    /// immediately beforehand by `encodeAnyResPatchifyEmbed*`).
    ///
    /// Returns the count of emitted language-model visual tokens (`0` immediately on the
    /// text-only fast path with `0` GPU dispatches).
    pub fn encodeTowerForward(
        self: *const VisionTowerRuntime,
        frame: *metal.Compute.Frame,
        blocks: []const VisionBlockWeights,
        merger: VisionMergerWeights,
        grid_h: u32,
        grid_w: u32,
        tile_count: u32,
        insert_barrier_before: bool,
    ) EncodeError!u32 {
        if (!self.has_vision) return 0;
        const sc = self.scratch orelse return error.InvalidBlockWeights;
        if (blocks.len != self.config.num_layers) return error.InvalidLayerCount;
        const num_patches = tile_count * grid_h * grid_w;
        if (num_patches == 0 or num_patches > self.config.max_patches) return error.InvalidGeometry;

        var cur_in = sc.hiddenABinding();
        var cur_out = sc.hiddenBBinding();
        for (blocks, 0..) |blk, i| {
            try self.encodeBlock(frame, blk, cur_in, cur_out, num_patches, if (i == 0) insert_barrier_before else true);
            const tmp = cur_in;
            cur_in = cur_out;
            cur_out = tmp;
        }
        return self.encodePatchMerger(frame, merger, cur_in, grid_h, grid_w, tile_count, true);
    }
};

// ===========================================================================
// M7.8 G22 CPU `fp32` Reference Implementations (for exact GPU parity tests)
// ===========================================================================

fn geluTanhF32(v: f32) f32 {
    const inner = 0.7978845608028654 * (v + 0.044715 * v * v * v);
    return 0.5 * v * (1.0 + std.math.tanh(inner));
}

fn geluExactF32(v: f32) f32 {
    const z = v * 0.7071067811865475;
    const az = @abs(z);
    const t = 1.0 / (1.0 + 0.3275911 * az);
    const poly = (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t) + 0.254829592) * t;
    const erf_a = 1.0 - poly * @exp(-az * az);
    const erf_z = if (z >= 0.0) erf_a else -erf_a;
    return 0.5 * v * (1.0 + erf_z);
}

pub fn anyresBilinearPatchifyEmbedBgra8CpuRef(
    frame0_bytes: []const u8,
    frame0_bpr: usize,
    frame1_bytes: ?[]const u8,
    frame1_bpr: usize,
    src_width: u32,
    src_height: u32,
    tile: AnyResTileSpec,
    patch_size: u32,
    t_patch: u32,
    norm: NormParams,
    weight_raw: []const u16,
    bias_raw: ?[]const u16,
    pos_embed_raw: ?[]const u16,
    weight_dtype: quant.FloatDType,
    hidden_size: u32,
    out_f32: []f32,
) Error!void {
    const num_tokens = tile.grid_h * tile.grid_w;
    const patch_area = patch_size * patch_size;
    const tube_pixels = t_patch * patch_area;
    const token_in_dim = 3 * tube_pixels;
    if (out_f32.len < (tile.token_offset + num_tokens) * hidden_size) return error.OutputBufferTooSmall;

    const readRgb = struct {
        fn get(bytes: []const u8, bpr: usize, x: u32, y: u32) [3]f32 {
            const off = @as(usize, y) * bpr + @as(usize, x) * 4;
            return .{
                @as(f32, @floatFromInt(bytes[off + 2])) / 255.0,
                @as(f32, @floatFromInt(bytes[off + 1])) / 255.0,
                @as(f32, @floatFromInt(bytes[off + 0])) / 255.0,
            };
        }
    }.get;

    for (0..tile.grid_h) |ph| {
        for (0..tile.grid_w) |pw| {
            const tok: u32 = @intCast(ph * tile.grid_w + pw);
            var h: u32 = 0;
            while (h < hidden_size) : (h += 1) {
                var acc: f64 = if (bias_raw) |b| @as(f64, quant.floatBitsToF32(weight_dtype, b[h])) else 0.0;
                const w_row = @as(usize, h) * token_in_dim;
                for (0..t_patch) |dt| {
                    const sb = if (dt == 0) frame0_bytes else (frame1_bytes orelse frame0_bytes);
                    const sbpr = if (dt == 0) frame0_bpr else frame1_bpr;
                    for (0..patch_size) |dy| {
                        const dst_y: u32 = @intCast(ph * patch_size + dy);
                        var fy = tile.src_y0 + (@as(f32, @floatFromInt(dst_y)) + 0.5) * (tile.src_h / @as(f32, @floatFromInt(tile.dst_height))) - 0.5;
                        const max_y = @max(tile.src_y0, @min(@as(f32, @floatFromInt(src_height - 1)), tile.src_y0 + tile.src_h - 1.0));
                        fy = std.math.clamp(fy, tile.src_y0, max_y);
                        const y0: u32 = @intFromFloat(@floor(fy));
                        const y1: u32 = @min(y0 + 1, src_height - 1);
                        const wy = fy - @as(f32, @floatFromInt(y0));
                        for (0..patch_size) |dx| {
                            const dst_x: u32 = @intCast(pw * patch_size + dx);
                            var fx = tile.src_x0 + (@as(f32, @floatFromInt(dst_x)) + 0.5) * (tile.src_w / @as(f32, @floatFromInt(tile.dst_width))) - 0.5;
                            const max_x = @max(tile.src_x0, @min(@as(f32, @floatFromInt(src_width - 1)), tile.src_x0 + tile.src_w - 1.0));
                            fx = std.math.clamp(fx, tile.src_x0, max_x);
                            const x0: u32 = @intFromFloat(@floor(fx));
                            const x1: u32 = @min(x0 + 1, src_width - 1);
                            const wx = fx - @as(f32, @floatFromInt(x0));

                            const c00 = readRgb(sb, sbpr, x0, y0);
                            const c10 = readRgb(sb, sbpr, x1, y0);
                            const c01 = readRgb(sb, sbpr, x0, y1);
                            const c11 = readRgb(sb, sbpr, x1, y1);
                            var rgb: [3]f32 = undefined;
                            for (0..3) |ch| {
                                const top = c00[ch] + (c10[ch] - c00[ch]) * wx;
                                const bot = c01[ch] + (c11[ch] - c01[ch]) * wx;
                                rgb[ch] = top + (bot - top) * wy;
                            }
                            const rn = normalizeRgbCpuRef(rgb, norm);
                            const pix = (dt * patch_size + dy) * patch_size + dx;
                            acc += @as(f64, rn[0]) * @as(f64, quant.floatBitsToF32(weight_dtype, weight_raw[w_row + pix]));
                            acc += @as(f64, rn[1]) * @as(f64, quant.floatBitsToF32(weight_dtype, weight_raw[w_row + pix + tube_pixels]));
                            acc += @as(f64, rn[2]) * @as(f64, quant.floatBitsToF32(weight_dtype, weight_raw[w_row + pix + 2 * tube_pixels]));
                        }
                    }
                }
                const out_idx = @as(usize, tile.token_offset + tok) * hidden_size + h;
                if (pos_embed_raw) |pe| {
                    acc += @as(f64, quant.floatBitsToF32(weight_dtype, pe[out_idx]));
                }
                out_f32[out_idx] = @floatCast(acc);
            }
        }
    }
}

pub fn anyresBilinearPatchifyEmbedNv12CpuRef(
    y0_bytes: []const u8,
    y0_bpr: usize,
    uv0_bytes: []const u8,
    uv0_bpr: usize,
    y1_bytes: ?[]const u8,
    y1_bpr: usize,
    uv1_bytes: ?[]const u8,
    uv1_bpr: usize,
    src_width: u32,
    src_height: u32,
    tile: AnyResTileSpec,
    patch_size: u32,
    t_patch: u32,
    is_full_range: bool,
    norm: NormParams,
    weight_raw: []const u16,
    bias_raw: ?[]const u16,
    pos_embed_raw: ?[]const u16,
    weight_dtype: quant.FloatDType,
    hidden_size: u32,
    out_f32: []f32,
) Error!void {
    const num_tokens = tile.grid_h * tile.grid_w;
    const patch_area = patch_size * patch_size;
    const tube_pixels = t_patch * patch_area;
    const token_in_dim = 3 * tube_pixels;
    if (out_f32.len < (tile.token_offset + num_tokens) * hidden_size) return error.OutputBufferTooSmall;

    const readNv12Rgb = struct {
        fn get(yb: []const u8, ybpr: usize, uvb: []const u8, uvbpr: usize, x: u32, y: u32, full: bool) [3]f32 {
            const y_u8 = yb[@as(usize, y) * ybpr + @as(usize, x)];
            const uv_off = @as(usize, y >> 1) * uvbpr + @as(usize, x >> 1) * 2;
            return bt709Nv12ToRgbCpuRef(y_u8, uvb[uv_off + 0], uvb[uv_off + 1], full);
        }
    }.get;

    for (0..tile.grid_h) |ph| {
        for (0..tile.grid_w) |pw| {
            const tok: u32 = @intCast(ph * tile.grid_w + pw);
            var h: u32 = 0;
            while (h < hidden_size) : (h += 1) {
                var acc: f64 = if (bias_raw) |b| @as(f64, quant.floatBitsToF32(weight_dtype, b[h])) else 0.0;
                const w_row = @as(usize, h) * token_in_dim;
                for (0..t_patch) |dt| {
                    const yb = if (dt == 0) y0_bytes else (y1_bytes orelse y0_bytes);
                    const ybpr = if (dt == 0) y0_bpr else y1_bpr;
                    const uvb = if (dt == 0) uv0_bytes else (uv1_bytes orelse uv0_bytes);
                    const uvbpr = if (dt == 0) uv0_bpr else uv1_bpr;
                    for (0..patch_size) |dy| {
                        const dst_y: u32 = @intCast(ph * patch_size + dy);
                        var fy = tile.src_y0 + (@as(f32, @floatFromInt(dst_y)) + 0.5) * (tile.src_h / @as(f32, @floatFromInt(tile.dst_height))) - 0.5;
                        const max_y = @max(tile.src_y0, @min(@as(f32, @floatFromInt(src_height - 1)), tile.src_y0 + tile.src_h - 1.0));
                        fy = std.math.clamp(fy, tile.src_y0, max_y);
                        const y0: u32 = @intFromFloat(@floor(fy));
                        const y1: u32 = @min(y0 + 1, src_height - 1);
                        const wy = fy - @as(f32, @floatFromInt(y0));
                        for (0..patch_size) |dx| {
                            const dst_x: u32 = @intCast(pw * patch_size + dx);
                            var fx = tile.src_x0 + (@as(f32, @floatFromInt(dst_x)) + 0.5) * (tile.src_w / @as(f32, @floatFromInt(tile.dst_width))) - 0.5;
                            const max_x = @max(tile.src_x0, @min(@as(f32, @floatFromInt(src_width - 1)), tile.src_x0 + tile.src_w - 1.0));
                            fx = std.math.clamp(fx, tile.src_x0, max_x);
                            const x0: u32 = @intFromFloat(@floor(fx));
                            const x1: u32 = @min(x0 + 1, src_width - 1);
                            const wx = fx - @as(f32, @floatFromInt(x0));

                            const c00 = readNv12Rgb(yb, ybpr, uvb, uvbpr, x0, y0, is_full_range);
                            const c10 = readNv12Rgb(yb, ybpr, uvb, uvbpr, x1, y0, is_full_range);
                            const c01 = readNv12Rgb(yb, ybpr, uvb, uvbpr, x0, y1, is_full_range);
                            const c11 = readNv12Rgb(yb, ybpr, uvb, uvbpr, x1, y1, is_full_range);
                            var rgb: [3]f32 = undefined;
                            for (0..3) |ch| {
                                const top = c00[ch] + (c10[ch] - c00[ch]) * wx;
                                const bot = c01[ch] + (c11[ch] - c01[ch]) * wx;
                                rgb[ch] = top + (bot - top) * wy;
                            }
                            const rn = normalizeRgbCpuRef(rgb, norm);
                            const pix = (dt * patch_size + dy) * patch_size + dx;
                            acc += @as(f64, rn[0]) * @as(f64, quant.floatBitsToF32(weight_dtype, weight_raw[w_row + pix]));
                            acc += @as(f64, rn[1]) * @as(f64, quant.floatBitsToF32(weight_dtype, weight_raw[w_row + pix + tube_pixels]));
                            acc += @as(f64, rn[2]) * @as(f64, quant.floatBitsToF32(weight_dtype, weight_raw[w_row + pix + 2 * tube_pixels]));
                        }
                    }
                }
                const out_idx = @as(usize, tile.token_offset + tok) * hidden_size + h;
                if (pos_embed_raw) |pe| {
                    acc += @as(f64, quant.floatBitsToF32(weight_dtype, pe[out_idx]));
                }
                out_f32[out_idx] = @floatCast(acc);
            }
        }
    }
}

pub fn visionNormCpuRef(
    x: []const f32,
    gamma: []const f32,
    beta: ?[]const f32,
    num_tokens: usize,
    dim: usize,
    kind: VisionNormKind,
    eps: f32,
    out: []f32,
) void {
    for (0..num_tokens) |t| {
        const row = x[t * dim ..][0..dim];
        var sum: f64 = 0.0;
        var sq: f64 = 0.0;
        for (row) |v| {
            sum += v;
            sq += @as(f64, v) * @as(f64, v);
        }
        const inv_n = 1.0 / @as(f64, @floatFromInt(dim));
        const mean: f64 = if (kind == .rms_norm) 0.0 else sum * inv_n;
        const var_v: f64 = if (kind == .rms_norm) sq * inv_n else @max(0.0, sq * inv_n - mean * mean);
        const inv_std: f64 = 1.0 / @sqrt(var_v + @as(f64, eps));
        for (0..dim) |d| {
            var y = (@as(f64, row[d]) - mean) * inv_std * @as(f64, gamma[d]);
            if (beta) |b| y += b[d];
            out[t * dim + d] = @floatCast(y);
        }
    }
}

pub fn visionQkvRope2DCpuRef(
    qkv: []const f32,
    coords: []const RopeCoord2D,
    num_tokens: usize,
    num_heads: usize,
    head_dim: usize,
    layout: VisionQkvLayout,
    interleaved: bool,
    theta: f32,
    q_out: []f32,
    k_out: []f32,
    v_out: []f32,
) void {
    const hidden = num_heads * head_dim;
    const axis_dim = head_dim / 2;
    const axis_pairs = axis_dim / 2;
    for (0..num_tokens) |t| {
        const hw = coords[t];
        for (0..num_heads) |h| {
            const q_in = if (layout == .split_3hd)
                t * 3 * hidden + h * head_dim
            else
                t * 3 * hidden + h * 3 * head_dim;
            const k_in = if (layout == .split_3hd)
                t * 3 * hidden + hidden + h * head_dim
            else
                t * 3 * hidden + h * 3 * head_dim + head_dim;
            const v_in = if (layout == .split_3hd)
                t * 3 * hidden + 2 * hidden + h * head_dim
            else
                t * 3 * hidden + h * 3 * head_dim + 2 * head_dim;
            const out_base = (t * num_heads + h) * head_dim;

            for (0..2) |axis| {
                const pos: f64 = @floatFromInt(if (axis == 0) hw.h else hw.w);
                const axis_base = axis * axis_dim;
                for (0..axis_pairs) |p_idx| {
                    const inv_freq = std.math.pow(f64, @as(f64, theta), -@as(f64, @floatFromInt(2 * p_idx)) / @as(f64, @floatFromInt(axis_dim)));
                    const angle = pos * inv_freq;
                    const c = @cos(angle);
                    const s = @sin(angle);
                    const d0 = if (!interleaved) axis_base + p_idx else axis_base + 2 * p_idx;
                    const d1 = if (!interleaved) axis_base + p_idx + axis_pairs else axis_base + 2 * p_idx + 1;

                    const q0: f64 = qkv[q_in + d0];
                    const q1: f64 = qkv[q_in + d1];
                    q_out[out_base + d0] = @floatCast(q0 * c - q1 * s);
                    q_out[out_base + d1] = @floatCast(q0 * s + q1 * c);

                    const k0: f64 = qkv[k_in + d0];
                    const k1: f64 = qkv[k_in + d1];
                    k_out[out_base + d0] = @floatCast(k0 * c - k1 * s);
                    k_out[out_base + d1] = @floatCast(k0 * s + k1 * c);

                    v_out[out_base + d0] = qkv[v_in + d0];
                    v_out[out_base + d1] = qkv[v_in + d1];
                }
            }
        }
    }
}

pub fn visionBidirAttnCpuRef(
    q: []const f32,
    k: []const f32,
    v: []const f32,
    window_bounds: ?[]const AttnWindowBound,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    out: []f32,
    scratch_scores: []f64,
) void {
    const scale: f64 = 1.0 / @sqrt(@as(f64, @floatFromInt(head_dim)));
    const stride = num_heads * head_dim;
    for (0..num_heads) |h| {
        for (0..seq_len) |qi| {
            const ks: usize = if (window_bounds) |wb| @min(wb[qi].kv_start, seq_len) else 0;
            const ke: usize = if (window_bounds) |wb| @min(wb[qi].kv_end, seq_len) else seq_len;
            var max_s: f64 = -std.math.inf(f64);
            for (ks..ke) |ki| {
                var dot: f64 = 0.0;
                for (0..head_dim) |d| {
                    dot += @as(f64, q[qi * stride + h * head_dim + d]) *
                        @as(f64, k[ki * stride + h * head_dim + d]);
                }
                const sc = dot * scale;
                scratch_scores[ki] = sc;
                if (sc > max_s) max_s = sc;
            }
            var sum_p: f64 = 0.0;
            for (ks..ke) |ki| {
                const p = @exp(scratch_scores[ki] - max_s);
                scratch_scores[ki] = p;
                sum_p += p;
            }
            const inv_l: f64 = if (sum_p > 0.0) 1.0 / sum_p else 0.0;
            for (0..head_dim) |d| {
                var acc: f64 = 0.0;
                for (ks..ke) |ki| {
                    acc += scratch_scores[ki] * @as(f64, v[ki * stride + h * head_dim + d]);
                }
                out[qi * stride + h * head_dim + d] = @floatCast(acc * inv_l);
            }
        }
    }
}

pub fn visionLinearCpuRef(
    x: []const f32,
    w: []const f32,
    bias: ?[]const f32,
    residual: ?[]const f32,
    m: usize,
    n: usize,
    k: usize,
    act: VisionLinearAct,
    out: []f32,
) void {
    for (0..m) |r| {
        for (0..n) |c| {
            var acc: f64 = if (bias) |b| b[c] else 0.0;
            for (0..k) |j| {
                acc += @as(f64, x[r * k + j]) * @as(f64, w[c * k + j]);
            }
            var val: f32 = @floatCast(acc);
            val = switch (act) {
                .none => val,
                .gelu_tanh => geluTanhF32(val),
                .gelu_exact => geluExactF32(val),
            };
            if (residual) |res| {
                val += res[r * n + c];
            }
            out[r * n + c] = val;
        }
    }
}

pub fn visionSwiGluCpuRef(gate: []const f32, up: []const f32, out: []f32) void {
    for (0..out.len) |i| {
        const g = gate[i];
        out[i] = (g / (1.0 + @exp(-g))) * up[i];
    }
}

pub fn visionSpatialMergeConcatCpuRef(
    in_tokens: []const f32,
    grid_h: usize,
    grid_w: usize,
    spatial_merge: usize,
    vis_hidden: usize,
    tile_count: usize,
    out_merged: []f32,
) void {
    const mh_max = grid_h / spatial_merge;
    const mw_max = grid_w / spatial_merge;
    const out_dim = spatial_merge * spatial_merge * vis_hidden;
    const in_per_tile = grid_h * grid_w;
    const out_per_tile = mh_max * mw_max;
    for (0..tile_count) |t| {
        for (0..mh_max) |mh| {
            for (0..mw_max) |mw| {
                const out_tok = t * out_per_tile + mh * mw_max + mw;
                for (0..spatial_merge) |sy| {
                    for (0..spatial_merge) |sx| {
                        const sub = sy * spatial_merge + sx;
                        const ph = mh * spatial_merge + sy;
                        const pw = mw * spatial_merge + sx;
                        const in_tok = t * in_per_tile + ph * grid_w + pw;
                        for (0..vis_hidden) |d| {
                            out_merged[out_tok * out_dim + sub * vis_hidden + d] =
                                in_tokens[in_tok * vis_hidden + d];
                        }
                    }
                }
            }
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

// ===========================================================================
// M7.8 G22 Tests (AnyRes Tiling, Bidirectional 2D-RoPE FlashAttention-2,
// ViT LayerNorm/RMSNorm, GELU/SwiGLU, Q4 Vision Linear, PatchMerger/Aligner,
// Aliased Scratch, Zero-Alloc Text Fast Path, and C16/D5 ANE Calibration)
// ===========================================================================

fn quantizeMatrixQ4(
    src_f32: []const f32,
    n: usize,
    k: usize,
    qweight_out: []u8,
    sb_out: []quant.ScaleBias,
    deq_f32_out: []f32,
) void {
    const Q4 = quant.QuantOps(.{ .bits = .q4, .group_size = 64 });
    const groups = k / 64;
    const row_bytes = groups * 32;
    for (0..n) |r| {
        for (0..groups) |g| {
            const src_grp = src_f32[r * k + g * 64 ..][0..64];
            var min_v: f32 = src_grp[0];
            var max_v: f32 = src_grp[0];
            for (src_grp) |v| {
                if (v < min_v) min_v = v;
                if (v > max_v) max_v = v;
            }
            const span = max_v - min_v;
            const raw_scale: f32 = if (span > 1e-8) span / 15.0 else 1e-4;
            const sb: quant.ScaleBias = .{
                .scale = @floatCast(raw_scale),
                .bias = @floatCast(min_v),
            };
            const eff_scale: f32 = @floatCast(sb.scale);
            const eff_bias: f32 = @floatCast(sb.bias);
            var codes: [64]u8 = undefined;
            for (0..64) |i| {
                const q_float = std.math.clamp(@round((src_grp[i] - eff_bias) / eff_scale), 0.0, 15.0);
                codes[i] = @intFromFloat(q_float);
            }
            const dst_grp = qweight_out[r * row_bytes + g * 32 ..][0..32];
            Q4.packGroup(&codes, dst_grp);
            sb_out[r * groups + g] = sb;
            var deq_grp: [64]f32 = undefined;
            Q4.dequantizeGroup(dst_grp, sb, &deq_grp);
            @memcpy(deq_f32_out[r * k + g * 64 ..][0..64], &deq_grp);
        }
    }
}

test "G22 AnyRes smart_resize and dynamic_grid tiling + Metal BGRA8/NV12 bilinear patchify embed parity" {
    // 1. Pure AnyRes planner checks (smart_resize + dynamic_grid + 2D-RoPE coords + window bounds + G9 MRoPE).
    const sr = try planAnyResGrid(450, 310, .{
        .mode = .smart_resize,
        .patch_size = 14,
        .t_patch = 1,
        .spatial_merge = 2,
        .min_pixels = 28 * 28,
        .max_pixels = 224 * 224,
    });
    try std.testing.expectEqual(@as(u32, 1), sr.num_tiles);
    try std.testing.expectEqual(@as(u32, 0), sr.resized_width % 28);
    try std.testing.expectEqual(@as(u32, 0), sr.resized_height % 28);
    try std.testing.expect(sr.resized_width * sr.resized_height <= 224 * 224);

    const dg = try planAnyResGrid(160, 80, .{
        .mode = .dynamic_grid,
        .patch_size = 14,
        .t_patch = 1,
        .spatial_merge = 2,
        .tile_size = 56,
        .max_tiles = 4,
        .include_thumbnail = true,
    });
    try std.testing.expectEqual(@as(u32, 2), dg.grid_cols);
    try std.testing.expectEqual(@as(u32, 1), dg.grid_rows);
    try std.testing.expectEqual(@as(u32, 3), dg.num_tiles); // 2x1 crops + 1 global thumbnail
    try std.testing.expect(dg.tiles[2].is_thumbnail);
    try std.testing.expectEqual(@as(u32, 48), dg.total_patches); // 3 * (4 * 4)
    try std.testing.expectEqual(@as(u32, 12), dg.total_merged_tokens);

    var coords: [48]RopeCoord2D = undefined;
    try std.testing.expectEqual(@as(u32, 48), try dg.fillRope2DCoords(&coords));
    try std.testing.expectEqual(RopeCoord2D{ .h = 0, .w = 0 }, coords[0]);
    try std.testing.expectEqual(RopeCoord2D{ .h = 3, .w = 3 }, coords[15]);
    try std.testing.expectEqual(RopeCoord2D{ .h = 0, .w = 0 }, coords[32]);

    var bounds: [48]AttnWindowBound = undefined;
    try std.testing.expectEqual(@as(u32, 48), try dg.fillWindowBounds(0, &bounds));
    try std.testing.expectEqual(AttnWindowBound{ .kv_start = 0, .kv_end = 16 }, bounds[5]);
    try std.testing.expectEqual(AttnWindowBound{ .kv_start = 16, .kv_end = 32 }, bounds[20]);
    try std.testing.expectEqual(AttnWindowBound{ .kv_start = 32, .kv_end = 48 }, bounds[47]);

    const gs = dg.toVisionGridSpec(1, 1);
    try std.testing.expectEqual(@as(u32, 12), try gs.numTokens());

    // 2. Metal GPU AnyRes bilinear resize + tube patchify + linear embed + pos_embed vs CPU fp32 ref.
    const device = try metal.Device.open();
    defer device.close();
    const lib = try metal.Library.openEmbedded(device);
    defer lib.close();
    const pipes = try VisionTowerPipelines.init(device, lib);
    defer pipes.close();
    var compute = try metal.Compute.init(device);
    defer compute.close();

    const src_w: u32 = 96;
    const src_h: u32 = 48;
    const p_sz: u32 = 14;
    const hidden: u32 = 64;
    const token_in_dim: u32 = 3 * 1 * p_sz * p_sz; // 588

    const bgra_frame = try metal.VisionFrame.init(device, .bgra8, src_w, src_h);
    defer bgra_frame.close();
    try fillSyntheticBgra8(bgra_frame, src_w, src_h, 7);

    const nv12_frame = try metal.VisionFrame.init(device, .nv12, src_w, src_h);
    defer nv12_frame.close();
    try fillSyntheticNv12(nv12_frame, src_w, src_h, 9);

    try compute.makeVisionFramesResident(&.{ bgra_frame, nv12_frame });
    defer compute.removeVisionFramesResident(&.{ bgra_frame, nv12_frame }) catch |e| {
        std.debug.print("removeVisionFramesResident err: {}\n", .{e});
    };

    const ar_plan = try planAnyResGrid(src_w, src_h, .{
        .mode = .dynamic_grid,
        .patch_size = p_sz,
        .t_patch = 1,
        .spatial_merge = 2,
        .tile_size = 28,
        .max_tiles = 4,
        .include_thumbnail = true,
    });
    try std.testing.expectEqual(@as(u32, 12), ar_plan.total_patches); // 3 tiles * 2x2 patches

    const w_elems = hidden * token_in_dim;
    const pe_elems = ar_plan.total_patches * hidden;
    const arena_bytes = try page.alignUp((w_elems + hidden + pe_elems + 2 * pe_elems) * 2 + 4096);
    var arena = try arena_mod.WiredArena.init(arena_bytes, testBudget(arena_bytes));
    defer arena.deinit();

    const w_raw = try arena.allocAligned(w_elems * 2, 64);
    const b_raw = try arena.allocAligned(hidden * 2, 64);
    const pe_raw = try arena.allocAligned(pe_elems * 2, 64);
    const out_bgra_raw = try arena.allocAligned(pe_elems * 2, 64);
    const out_nv12_raw = try arena.allocAligned(pe_elems * 2, 64);

    const w_u16: []u16 = @alignCast(std.mem.bytesAsSlice(u16, w_raw));
    const b_u16: []u16 = @alignCast(std.mem.bytesAsSlice(u16, b_raw));
    const pe_u16: []u16 = @alignCast(std.mem.bytesAsSlice(u16, pe_raw));
    for (0..w_elems) |i| w_u16[i] = quant.f32ToFloatBits(.bf16, synthWeight(i + 3, 0.015));
    for (0..hidden) |i| b_u16[i] = quant.f32ToFloatBits(.bf16, synthWeight(i + 91, 0.02));
    for (0..pe_elems) |i| pe_u16[i] = quant.f32ToFloatBits(.bf16, synthWeight(i + 19, 0.03));

    const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
    defer gpu_buf.close();
    try compute.makeResident(&.{gpu_buf});
    defer compute.removeResident(&.{gpu_buf}) catch |e| {
        std.debug.print("removeResident err: {}\n", .{e});
    };

    const w_bind: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(w_raw.ptr) };
    const b_bind: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(b_raw.ptr) };
    const pe_bind: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(pe_raw.ptr) };
    const out_bgra_bind: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out_bgra_raw.ptr) };
    const out_nv12_bind: metal.BufferBinding = .{ .buffer = gpu_buf, .offset = arena.offsetOf(out_nv12_raw.ptr) };

    var frame = try compute.beginFrame(5000);
    errdefer frame.abort();
    for (ar_plan.activeTiles(), 0..) |tile, ti| {
        try pipes.encodeAnyResPatchifyEmbedBgra8(
            &frame,
            bgra_frame,
            null,
            src_w,
            src_h,
            tile,
            p_sz,
            1,
            hidden,
            NormParams.QWEN_VL_DEFAULT,
            w_bind,
            b_bind,
            pe_bind,
            out_bgra_bind,
            .bf16,
            .f16,
            ti > 0,
        );
        try pipes.encodeAnyResPatchifyEmbedNv12(
            &frame,
            nv12_frame,
            null,
            src_w,
            src_h,
            tile,
            p_sz,
            1,
            hidden,
            true,
            NormParams.SIGLIP_DEFAULT,
            w_bind,
            b_bind,
            pe_bind,
            out_nv12_bind,
            .bf16,
            .bf16,
            true,
        );
    }
    const sig = try frame.commit();
    try compute.wait(sig, 5000);

    const ref_bgra = try std.testing.allocator.alloc(f32, pe_elems);
    defer std.testing.allocator.free(ref_bgra);
    const ref_nv12 = try std.testing.allocator.alloc(f32, pe_elems);
    defer std.testing.allocator.free(ref_nv12);

    {
        var bg = try bgra_frame.lockPlane(0, true);
        defer bg.release();
        for (ar_plan.activeTiles()) |tile| {
            try anyresBilinearPatchifyEmbedBgra8CpuRef(
                bg.bytes,
                bg.bytes_per_row,
                null,
                0,
                src_w,
                src_h,
                tile,
                p_sz,
                1,
                NormParams.QWEN_VL_DEFAULT,
                w_u16,
                b_u16,
                pe_u16,
                .bf16,
                hidden,
                ref_bgra,
            );
        }
    }
    {
        var yg = try nv12_frame.lockPlane(0, true);
        defer yg.release();
        var uvg = try nv12_frame.lockPlane(1, true);
        defer uvg.release();
        for (ar_plan.activeTiles()) |tile| {
            try anyresBilinearPatchifyEmbedNv12CpuRef(
                yg.bytes,
                yg.bytes_per_row,
                uvg.bytes,
                uvg.bytes_per_row,
                null,
                0,
                null,
                0,
                src_w,
                src_h,
                tile,
                p_sz,
                1,
                true,
                NormParams.SIGLIP_DEFAULT,
                w_u16,
                b_u16,
                pe_u16,
                .bf16,
                hidden,
                ref_nv12,
            );
        }
    }

    const got_bgra_u16: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, out_bgra_raw));
    const got_nv12_u16: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, out_nv12_raw));
    for (0..pe_elems) |i| {
        const g_bgra = quant.floatBitsToF32(.f16, got_bgra_u16[i]);
        try std.testing.expectApproxEqAbs(ref_bgra[i], g_bgra, @max(0.04, @abs(ref_bgra[i]) * 0.03));
        const g_nv12 = quant.floatBitsToF32(.bf16, got_nv12_u16[i]);
        try std.testing.expectApproxEqAbs(ref_nv12[i], g_nv12, @max(0.08, @abs(ref_nv12[i]) * 0.04));
    }
}

test "G22 LayerNorm/RMSNorm, fused QKV 2D-RoPE, and bidirectional full/windowed FlashAttention-2 MMA parity" {
    const device = try metal.Device.open();
    defer device.close();
    const lib = try metal.Library.openEmbedded(device);
    defer lib.close();
    const pipes = try VisionTowerPipelines.init(device, lib);
    defer pipes.close();
    var compute = try metal.Compute.init(device);
    defer compute.close();

    // Use seq_len = 40 (> BQ=32 to exercise multi-block Q and KV tiles), num_heads = 2, head_dim = 72
    // (real Qwen3.8 head_dim = 1152/16 = 72, testing non-power-of-two head_dim padding & 2D-RoPE).
    const seq_len: u32 = 40;
    const num_heads: u32 = 2;
    const head_dim: u32 = 72;
    const hidden: u32 = num_heads * head_dim; // 144

    const tok_elems = seq_len * hidden;
    const qkv_elems = seq_len * 3 * hidden;
    const arena_bytes = try page.alignUp((tok_elems * 10 + qkv_elems + hidden * 4) * 2 + seq_len * 16 + 4096);
    var arena = try arena_mod.WiredArena.init(arena_bytes, testBudget(arena_bytes));
    defer arena.deinit();

    const x_raw = try arena.allocAligned(tok_elems * 2, 64);
    const gamma_raw = try arena.allocAligned(hidden * 2, 64);
    const beta_raw = try arena.allocAligned(hidden * 2, 64);
    const ln_out_raw = try arena.allocAligned(tok_elems * 2, 64);
    const rms_out_raw = try arena.allocAligned(tok_elems * 2, 64);
    const qkv_raw = try arena.allocAligned(qkv_elems * 2, 64);
    const coords_raw = try arena.allocAligned(seq_len * @sizeOf(RopeCoord2D), 64);
    const bounds_raw = try arena.allocAligned(seq_len * @sizeOf(AttnWindowBound), 64);
    const q_raw = try arena.allocAligned(tok_elems * 2, 64);
    const k_raw = try arena.allocAligned(tok_elems * 2, 64);
    const v_raw = try arena.allocAligned(tok_elems * 2, 64);
    const attn_full_raw = try arena.allocAligned(tok_elems * 2, 64);
    const attn_win_raw = try arena.allocAligned(tok_elems * 2, 64);
    const attn_pert_raw = try arena.allocAligned(tok_elems * 2, 64);

    const x_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, x_raw));
    const gamma_u16: []u16 = @alignCast(std.mem.bytesAsSlice(u16, gamma_raw));
    const beta_u16: []u16 = @alignCast(std.mem.bytesAsSlice(u16, beta_raw));
    const qkv_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, qkv_raw));
    const coords: []RopeCoord2D = @alignCast(std.mem.bytesAsSlice(RopeCoord2D, coords_raw));
    const bounds: []AttnWindowBound = @alignCast(std.mem.bytesAsSlice(AttnWindowBound, bounds_raw));

    for (0..tok_elems) |i| x_f16[i] = @floatCast(synthWeight(i + 1, 0.04));
    for (0..hidden) |i| {
        gamma_u16[i] = quant.f32ToFloatBits(.f16, 0.8 + synthWeight(i + 7, 0.01));
        beta_u16[i] = quant.f32ToFloatBits(.f16, synthWeight(i + 23, 0.015));
    }
    for (0..qkv_elems) |i| qkv_f16[i] = @floatCast(synthWeight(i + 5, 0.03));
    for (0..seq_len) |i| {
        coords[i] = .{ .h = @intCast(i / 8), .w = @intCast(i % 8) };
        bounds[i] = if (i < 16) .{ .kv_start = 0, .kv_end = 16 } else .{ .kv_start = 16, .kv_end = seq_len };
    }

    const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
    defer gpu_buf.close();
    try compute.makeResident(&.{gpu_buf});
    defer compute.removeResident(&.{gpu_buf}) catch |e| {
        std.debug.print("removeResident err: {}\n", .{e});
    };

    const bind = struct {
        fn of(buf: metal.Buffer, a: *const arena_mod.WiredArena, slice: []u8) metal.BufferBinding {
            return .{ .buffer = buf, .offset = a.offsetOf(slice.ptr) };
        }
    }.of;

    var frame = try compute.beginFrame(5000);
    errdefer frame.abort();
    // LayerNorm with beta (f16)
    try pipes.encodeNorm(&frame, bind(gpu_buf, &arena, x_raw), .{
        .gamma = bind(gpu_buf, &arena, gamma_raw),
        .beta = bind(gpu_buf, &arena, beta_raw),
        .dtype = .f16,
        .kind = .layer_norm,
        .eps = 1e-6,
    }, bind(gpu_buf, &arena, ln_out_raw), seq_len, hidden, false);
    // RMSNorm without beta
    try pipes.encodeNorm(&frame, bind(gpu_buf, &arena, x_raw), .{
        .gamma = bind(gpu_buf, &arena, gamma_raw),
        .beta = null,
        .dtype = .f16,
        .kind = .rms_norm,
        .eps = 1e-6,
    }, bind(gpu_buf, &arena, rms_out_raw), seq_len, hidden, false);
    // Fused QKV split + 2D-RoPE
    try pipes.encodeQkvRope2D(
        &frame,
        bind(gpu_buf, &arena, qkv_raw),
        bind(gpu_buf, &arena, coords_raw),
        bind(gpu_buf, &arena, q_raw),
        bind(gpu_buf, &arena, k_raw),
        bind(gpu_buf, &arena, v_raw),
        seq_len,
        num_heads,
        head_dim,
        .split_3hd,
        false,
        10000.0,
        true,
    );
    // Bidirectional full attention (token 0 MUST attend to token 39!)
    try pipes.encodeBidirFlashAttn(
        &frame,
        bind(gpu_buf, &arena, q_raw),
        bind(gpu_buf, &arena, k_raw),
        bind(gpu_buf, &arena, v_raw),
        null,
        bind(gpu_buf, &arena, attn_full_raw),
        seq_len,
        num_heads,
        head_dim,
        true,
    );
    // Bidirectional windowed attention ([0..16) and [16..40))
    try pipes.encodeBidirFlashAttn(
        &frame,
        bind(gpu_buf, &arena, q_raw),
        bind(gpu_buf, &arena, k_raw),
        bind(gpu_buf, &arena, v_raw),
        bind(gpu_buf, &arena, bounds_raw),
        bind(gpu_buf, &arena, attn_win_raw),
        seq_len,
        num_heads,
        head_dim,
        true,
    );
    const sig1 = try frame.commit();
    try compute.wait(sig1, 5000);

    // CPU fp32 reference verification
    const x_f32 = try std.testing.allocator.alloc(f32, tok_elems);
    defer std.testing.allocator.free(x_f32);
    const g_f32 = try std.testing.allocator.alloc(f32, hidden);
    defer std.testing.allocator.free(g_f32);
    const b_f32 = try std.testing.allocator.alloc(f32, hidden);
    defer std.testing.allocator.free(b_f32);
    const ln_ref = try std.testing.allocator.alloc(f32, tok_elems);
    defer std.testing.allocator.free(ln_ref);
    const rms_ref = try std.testing.allocator.alloc(f32, tok_elems);
    defer std.testing.allocator.free(rms_ref);
    const qkv_f32 = try std.testing.allocator.alloc(f32, qkv_elems);
    defer std.testing.allocator.free(qkv_f32);
    const q_ref = try std.testing.allocator.alloc(f32, tok_elems);
    defer std.testing.allocator.free(q_ref);
    const k_ref = try std.testing.allocator.alloc(f32, tok_elems);
    defer std.testing.allocator.free(k_ref);
    const v_ref = try std.testing.allocator.alloc(f32, tok_elems);
    defer std.testing.allocator.free(v_ref);
    const attn_full_ref = try std.testing.allocator.alloc(f32, tok_elems);
    defer std.testing.allocator.free(attn_full_ref);
    const attn_win_ref = try std.testing.allocator.alloc(f32, tok_elems);
    defer std.testing.allocator.free(attn_win_ref);
    const score_scratch = try std.testing.allocator.alloc(f64, seq_len);
    defer std.testing.allocator.free(score_scratch);

    for (0..tok_elems) |i| x_f32[i] = @floatCast(x_f16[i]);
    for (0..hidden) |i| {
        g_f32[i] = quant.floatBitsToF32(.f16, gamma_u16[i]);
        b_f32[i] = quant.floatBitsToF32(.f16, beta_u16[i]);
    }
    for (0..qkv_elems) |i| qkv_f32[i] = @floatCast(qkv_f16[i]);

    visionNormCpuRef(x_f32, g_f32, b_f32, seq_len, hidden, .layer_norm, 1e-6, ln_ref);
    visionNormCpuRef(x_f32, g_f32, null, seq_len, hidden, .rms_norm, 1e-6, rms_ref);
    visionQkvRope2DCpuRef(qkv_f32, coords, seq_len, num_heads, head_dim, .split_3hd, false, 10000.0, q_ref, k_ref, v_ref);
    visionBidirAttnCpuRef(q_ref, k_ref, v_ref, null, seq_len, num_heads, head_dim, attn_full_ref, score_scratch);
    visionBidirAttnCpuRef(q_ref, k_ref, v_ref, bounds, seq_len, num_heads, head_dim, attn_win_ref, score_scratch);

    const got_ln: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, ln_out_raw));
    const got_rms: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, rms_out_raw));
    const got_q: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, q_raw));
    const got_k: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, k_raw));
    const got_v: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, v_raw));
    const got_af: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, attn_full_raw));
    const got_aw: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, attn_win_raw));

    for (0..tok_elems) |i| {
        try std.testing.expectApproxEqAbs(ln_ref[i], @as(f32, @floatCast(got_ln[i])), 8e-3);
        try std.testing.expectApproxEqAbs(rms_ref[i], @as(f32, @floatCast(got_rms[i])), 8e-3);
        try std.testing.expectApproxEqAbs(q_ref[i], @as(f32, @floatCast(got_q[i])), 5e-3);
        try std.testing.expectApproxEqAbs(k_ref[i], @as(f32, @floatCast(got_k[i])), 5e-3);
        try std.testing.expectApproxEqAbs(v_ref[i], @as(f32, @floatCast(got_v[i])), 2e-3);
        try std.testing.expectApproxEqAbs(attn_full_ref[i], @as(f32, @floatCast(got_af[i])), 6e-3);
        try std.testing.expectApproxEqAbs(attn_win_ref[i], @as(f32, @floatCast(got_aw[i])), 6e-3);
    }

    // Explicitly prove NON-CAUSALITY: perturbing the LAST token (index 39) MUST change token 0's output!
    const v_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, v_raw));
    for (0..hidden) |d| {
        v_f16[(seq_len - 1) * hidden + d] = @floatCast(@as(f32, @floatCast(v_f16[(seq_len - 1) * hidden + d])) + 4.0);
    }
    var frame2 = try compute.beginFrame(5000);
    errdefer frame2.abort();
    try pipes.encodeBidirFlashAttn(
        &frame2,
        bind(gpu_buf, &arena, q_raw),
        bind(gpu_buf, &arena, k_raw),
        bind(gpu_buf, &arena, v_raw),
        null,
        bind(gpu_buf, &arena, attn_pert_raw),
        seq_len,
        num_heads,
        head_dim,
        false,
    );
    const sig2 = try frame2.commit();
    try compute.wait(sig2, 5000);
    const got_ap: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, attn_pert_raw));
    var max_tok0_delta: f32 = 0.0;
    for (0..hidden) |d| {
        const diff = @abs(@as(f32, @floatCast(got_ap[d])) - @as(f32, @floatCast(got_af[d])));
        if (diff > max_tok0_delta) max_tok0_delta = diff;
    }
    try std.testing.expect(max_tok0_delta > 0.01);
}

test "G22 VisionLinear (f16/bf16/q4_affine + GELU/residual), SwiGLU, and SpatialMergeConcat parity" {
    const device = try metal.Device.open();
    defer device.close();
    const lib = try metal.Library.openEmbedded(device);
    defer lib.close();
    const pipes = try VisionTowerPipelines.init(device, lib);
    defer pipes.close();
    var compute = try metal.Compute.init(device);
    defer compute.close();

    const m: u32 = 36;
    const n: u32 = 40;
    const k: u32 = 128; // divisible by 64 for q4_affine
    const q4_row_bytes = (k / 64) * 32;
    const q4_groups = (n * k) / 64;

    const grid_h: u32 = 4;
    const grid_w: u32 = 4;
    const sm: u32 = 2;
    const vis_h: u32 = 16;
    const tiles: u32 = 2;
    const merge_in_elems = tiles * grid_h * grid_w * vis_h; // 512

    const arena_bytes = try page.alignUp((m * k + n * k * 2 + n * 2 + m * n * 6 + merge_in_elems * 2) * 2 + n * q4_row_bytes + q4_groups * 4 + 4096);
    var arena = try arena_mod.WiredArena.init(arena_bytes, testBudget(arena_bytes));
    defer arena.deinit();

    const x_raw = try arena.allocAligned(m * k * 2, 64);
    const w_f16_raw = try arena.allocAligned(n * k * 2, 64);
    const w_bf16_raw = try arena.allocAligned(n * k * 2, 64);
    const b_f16_raw = try arena.allocAligned(n * 2, 64);
    const b_bf16_raw = try arena.allocAligned(n * 2, 64);
    const res_raw = try arena.allocAligned(m * n * 2, 64);
    const y_f16_raw = try arena.allocAligned(m * n * 2, 64);
    const y_bf16_raw = try arena.allocAligned(m * n * 2, 64);
    const qw_raw = try arena.allocAligned(n * q4_row_bytes, 64);
    const qsb_raw = try arena.allocAligned(q4_groups * @sizeOf(quant.ScaleBias), 64);
    const y_q4_raw = try arena.allocAligned(m * n * 2, 64);
    const swiglu_out_raw = try arena.allocAligned(m * n * 2, 64);
    const merge_in_raw = try arena.allocAligned(merge_in_elems * 2, 64);
    const merge_out_raw = try arena.allocAligned(merge_in_elems * 2, 64);

    const x_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, x_raw));
    const w_f16_u16: []u16 = @alignCast(std.mem.bytesAsSlice(u16, w_f16_raw));
    const w_bf16_u16: []u16 = @alignCast(std.mem.bytesAsSlice(u16, w_bf16_raw));
    const b_f16_u16: []u16 = @alignCast(std.mem.bytesAsSlice(u16, b_f16_raw));
    const b_bf16_u16: []u16 = @alignCast(std.mem.bytesAsSlice(u16, b_bf16_raw));
    const res_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, res_raw));
    const qsb: []quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, qsb_raw));
    const merge_in_f16: []f16 = @alignCast(std.mem.bytesAsSlice(f16, merge_in_raw));

    const x_f32 = try std.testing.allocator.alloc(f32, m * k);
    defer std.testing.allocator.free(x_f32);
    const w_f32 = try std.testing.allocator.alloc(f32, n * k);
    defer std.testing.allocator.free(w_f32);
    const w_bf_f32 = try std.testing.allocator.alloc(f32, n * k);
    defer std.testing.allocator.free(w_bf_f32);
    const w_q4_f32 = try std.testing.allocator.alloc(f32, n * k);
    defer std.testing.allocator.free(w_q4_f32);
    const b_f32 = try std.testing.allocator.alloc(f32, n);
    defer std.testing.allocator.free(b_f32);
    const b_bf_f32 = try std.testing.allocator.alloc(f32, n);
    defer std.testing.allocator.free(b_bf_f32);
    const res_f32 = try std.testing.allocator.alloc(f32, m * n);
    defer std.testing.allocator.free(res_f32);

    for (0..m * k) |i| {
        const v = synthWeight(i + 3, 0.03);
        x_f16[i] = @floatCast(v);
        x_f32[i] = @floatCast(x_f16[i]);
    }
    for (0..n * k) |i| {
        const v = synthWeight(i + 17, 0.04);
        w_f16_u16[i] = quant.f32ToFloatBits(.f16, v);
        w_f32[i] = quant.floatBitsToF32(.f16, w_f16_u16[i]);
        w_bf16_u16[i] = quant.f32ToFloatBits(.bf16, v);
        w_bf_f32[i] = quant.floatBitsToF32(.bf16, w_bf16_u16[i]);
    }
    for (0..n) |i| {
        const v = synthWeight(i + 5, 0.02);
        b_f16_u16[i] = quant.f32ToFloatBits(.f16, v);
        b_f32[i] = quant.floatBitsToF32(.f16, b_f16_u16[i]);
        b_bf16_u16[i] = quant.f32ToFloatBits(.bf16, v);
        b_bf_f32[i] = quant.floatBitsToF32(.bf16, b_bf16_u16[i]);
    }
    for (0..m * n) |i| {
        const v = synthWeight(i + 29, 0.02);
        res_f16[i] = @floatCast(v);
        res_f32[i] = @floatCast(res_f16[i]);
    }
    quantizeMatrixQ4(w_f32, n, k, qw_raw, qsb, w_q4_f32);
    for (0..merge_in_elems) |i| merge_in_f16[i] = @floatCast(synthWeight(i + 11, 0.05));

    const gpu_buf = try metal.Buffer.wrap(device, arena.bytes());
    defer gpu_buf.close();
    try compute.makeResident(&.{gpu_buf});
    defer compute.removeResident(&.{gpu_buf}) catch |e| {
        std.debug.print("removeResident err: {}\n", .{e});
    };

    const bind = struct {
        fn of(buf: metal.Buffer, a: *const arena_mod.WiredArena, slice: []u8) metal.BufferBinding {
            return .{ .buffer = buf, .offset = a.offsetOf(slice.ptr) };
        }
    }.of;

    var frame = try compute.beginFrame(5000);
    errdefer frame.abort();
    // 1. Dense f16 + bias + gelu_tanh + residual
    try pipes.encodeLinear(&frame, bind(gpu_buf, &arena, x_raw), .{ .dense = .{
        .weight = bind(gpu_buf, &arena, w_f16_raw),
        .dtype = .f16,
        .bias = bind(gpu_buf, &arena, b_f16_raw),
    } }, bind(gpu_buf, &arena, res_raw), bind(gpu_buf, &arena, y_f16_raw), m, n, k, .gelu_tanh, false);
    // 2. Dense bf16 + bias + gelu_exact
    try pipes.encodeLinear(&frame, bind(gpu_buf, &arena, x_raw), .{ .dense = .{
        .weight = bind(gpu_buf, &arena, w_bf16_raw),
        .dtype = .bf16,
        .bias = bind(gpu_buf, &arena, b_bf16_raw),
    } }, null, bind(gpu_buf, &arena, y_bf16_raw), m, n, k, .gelu_exact, false);
    // 3. Q4_AFFINE + bias + residual
    try pipes.encodeLinear(&frame, bind(gpu_buf, &arena, x_raw), .{ .q4_affine = .{
        .qweight = bind(gpu_buf, &arena, qw_raw),
        .scale_bias = bind(gpu_buf, &arena, qsb_raw),
        .group_size = 64,
        .bias = bind(gpu_buf, &arena, b_f16_raw),
        .bias_dtype = .f16,
    } }, bind(gpu_buf, &arena, res_raw), bind(gpu_buf, &arena, y_q4_raw), m, n, k, .none, false);
    // 4. SwiGLU(y_f16, res)
    try pipes.encodeSwiGlu(&frame, bind(gpu_buf, &arena, y_f16_raw), bind(gpu_buf, &arena, res_raw), bind(gpu_buf, &arena, swiglu_out_raw), m * n, true);
    // 5. SpatialMergeConcat
    try pipes.encodeSpatialMergeConcat(&frame, bind(gpu_buf, &arena, merge_in_raw), bind(gpu_buf, &arena, merge_out_raw), grid_h, grid_w, sm, vis_h, tiles, false);

    const sig = try frame.commit();
    try compute.wait(sig, 5000);

    const ref_f16 = try std.testing.allocator.alloc(f32, m * n);
    defer std.testing.allocator.free(ref_f16);
    const ref_bf16 = try std.testing.allocator.alloc(f32, m * n);
    defer std.testing.allocator.free(ref_bf16);
    const ref_q4 = try std.testing.allocator.alloc(f32, m * n);
    defer std.testing.allocator.free(ref_q4);
    const ref_swi = try std.testing.allocator.alloc(f32, m * n);
    defer std.testing.allocator.free(ref_swi);
    const merge_in_f32 = try std.testing.allocator.alloc(f32, merge_in_elems);
    defer std.testing.allocator.free(merge_in_f32);
    const merge_ref = try std.testing.allocator.alloc(f32, merge_in_elems);
    defer std.testing.allocator.free(merge_ref);

    visionLinearCpuRef(x_f32, w_f32, b_f32, res_f32, m, n, k, .gelu_tanh, ref_f16);
    visionLinearCpuRef(x_f32, w_bf_f32, b_bf_f32, null, m, n, k, .gelu_exact, ref_bf16);
    visionLinearCpuRef(x_f32, w_q4_f32, b_f32, res_f32, m, n, k, .none, ref_q4);
    visionSwiGluCpuRef(ref_f16, res_f32, ref_swi);
    for (0..merge_in_elems) |i| merge_in_f32[i] = @floatCast(merge_in_f16[i]);
    visionSpatialMergeConcatCpuRef(merge_in_f32, grid_h, grid_w, sm, vis_h, tiles, merge_ref);

    const got_f16: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, y_f16_raw));
    const got_bf16: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, y_bf16_raw));
    const got_q4: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, y_q4_raw));
    const got_swi: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, swiglu_out_raw));
    const got_merge: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, merge_out_raw));

    for (0..m * n) |i| {
        try std.testing.expectApproxEqAbs(ref_f16[i], @as(f32, @floatCast(got_f16[i])), @max(5e-3, @abs(ref_f16[i]) * 2e-3));
        try std.testing.expectApproxEqAbs(ref_bf16[i], @as(f32, @floatCast(got_bf16[i])), @max(5e-3, @abs(ref_bf16[i]) * 2e-3));
        try std.testing.expectApproxEqAbs(ref_q4[i], @as(f32, @floatCast(got_q4[i])), @max(1e-2, @abs(ref_q4[i]) * 3e-3));
        try std.testing.expectApproxEqAbs(ref_swi[i], @as(f32, @floatCast(got_swi[i])), @max(5e-3, @abs(ref_swi[i]) * 2e-3));
    }
    for (0..merge_in_elems) |i| {
        try std.testing.expectApproxEqAbs(merge_ref[i], @as(f32, @floatCast(got_merge[i])), 1e-4);
    }
}

test "G22 end-to-end 3-layer ViT + PatchMerger/Aligner, aliased scratch, zero-alloc text fast path, and C16/D5 ANE calibration" {
    const device = try metal.Device.open();
    defer device.close();
    const lib = try metal.Library.openEmbedded(device);
    defer lib.close();
    var compute = try metal.Compute.init(device);
    defer compute.close();

    // 1. Zero-overhead text-only fast path (`has_vision == false`): 0 bytes, 0 pipelines, 0 GPU dispatches.
    try std.testing.expectEqual(@as(usize, 0), try VisionTowerScratch.requiredBytes(.{ .has_vision = false }));
    var text_rt = VisionTowerRuntime.initTextOnly();
    defer text_rt.deinit();
    try std.testing.expect(!text_rt.isActive());
    try std.testing.expectEqual(@as(usize, 0), text_rt.ownedWiredBytes());
    var dummy_frame = try compute.beginFrame(5000);
    const dummy_emitted = try text_rt.encodeTowerForward(&dummy_frame, &.{}, undefined, 4, 4, 1, false);
    dummy_frame.abort();
    try std.testing.expectEqual(@as(u32, 0), dummy_emitted);

    // 2. End-to-end 3-layer ViT (both Qwen-VL GELU + Q4 blocks + pre_merge norm,
    //    AND DeepSeek-V4.1 SwiGLU + RMSNorm + post_merge Aligner) using `WiredArena.allocAliasedPair`.
    const vis_h: u32 = 64;
    const vis_inter: u32 = 128;
    const num_heads: u32 = 2;
    const head_dim: u32 = 32;
    const lm_h: u32 = 96;
    const sm: u32 = 2;
    const m_in: u32 = vis_h * sm * sm; // 256
    const m_hid: u32 = 128;
    const grid_h: u32 = 4;
    const grid_w: u32 = 4;
    const num_patches: u32 = grid_h * grid_w; // 16
    const num_layers: u32 = 3;

    const tower_cfg: VisionTowerConfig = .{
        .has_vision = true,
        .num_layers = num_layers,
        .vis_hidden = vis_h,
        .vis_intermediate = vis_inter,
        .num_heads = num_heads,
        .head_dim = head_dim,
        .lm_hidden = lm_h,
        .merger_hidden = m_hid,
        .patch_size = 14,
        .t_patch = 1,
        .spatial_merge = sm,
        .max_patches = num_patches,
        .mlp_kind = .gelu_tanh,
        .qkv_layout = .split_3hd,
        .rope_interleaved = false,
        .rope_theta = 10000.0,
    };
    const vis_scratch_req = try VisionTowerScratch.requiredBytes(tower_cfg);
    const prefill_scratch_req: usize = vis_scratch_req; // simulated language prefill scratch

    // Allocate weight arena + ONE aliased scratch pair (vision scratch + prefill scratch share exact same pages).
    const qkv_qw_bytes = (3 * vis_h) * (vis_h / 64) * 32;
    const qkv_sb_bytes = (3 * vis_h) * (vis_h / 64) * @sizeOf(quant.ScaleBias);
    const per_layer_bytes = (vis_h * 4 + 3 * vis_h + vis_inter + vis_h) * 2 +
        qkv_qw_bytes + qkv_sb_bytes +
        (vis_h * vis_h + vis_inter * vis_h * 2 + vis_h * vis_inter) * 2 +
        vis_h * vis_h * 2;
    const merger_bytes = (vis_h * 2 + m_in * 2 + m_hid * m_in + m_hid + lm_h * m_hid + lm_h) * 2;
    const total_arena_bytes = try page.alignUp(per_layer_bytes * num_layers + merger_bytes + vis_scratch_req + 16384);

    var master_arena = try arena_mod.WiredArena.init(total_arena_bytes, testBudget(total_arena_bytes));
    defer master_arena.deinit();

    const pair = try master_arena.allocAliasedPair(vis_scratch_req, prefill_scratch_req);
    try std.testing.expectEqual(vis_scratch_req, pair.saved_bytes);
    try std.testing.expectEqual(@intFromPtr(pair.arena_a.bytes().ptr), @intFromPtr(pair.arena_b.bytes().ptr));

    const master_buf = try metal.Buffer.wrap(device, master_arena.bytes());
    defer master_buf.close();
    try compute.makeResident(&.{master_buf});
    defer compute.removeResident(&.{master_buf}) catch |e| {
        std.debug.print("removeResident err: {}\n", .{e});
    };

    const vis_base_off = master_arena.offsetOf(pair.backing.ptr);
    var vis_arena = pair.arena_a;
    var rt = try VisionTowerRuntime.initAliased(device, lib, &vis_arena, master_buf, vis_base_off, tower_cfg);
    defer rt.deinit();
    try std.testing.expect(rt.isActive());
    try std.testing.expectEqual(@as(usize, 0), rt.ownedWiredBytes());

    const CpuLayerRef = struct {
        n1_g: []f32,
        n1_b: []f32,
        qkv_w: []f32,
        qkv_bias: []f32,
        proj_w: []f32,
        proj_bias: []f32,
        n2_g: []f32,
        n2_b: []f32,
        fc1_w: []f32,
        fc1_bias: []f32,
        up_w: []f32,
        fc2_w: []f32,
        fc2_bias: []f32,
    };

    var blocks: [num_layers]VisionBlockWeights = undefined;
    var cpu_layers: [num_layers]CpuLayerRef = undefined;
    defer for (&cpu_layers) |*cl| {
        std.testing.allocator.free(cl.n1_g);
        std.testing.allocator.free(cl.n1_b);
        std.testing.allocator.free(cl.qkv_w);
        std.testing.allocator.free(cl.qkv_bias);
        std.testing.allocator.free(cl.proj_w);
        std.testing.allocator.free(cl.proj_bias);
        std.testing.allocator.free(cl.n2_g);
        std.testing.allocator.free(cl.n2_b);
        std.testing.allocator.free(cl.fc1_w);
        std.testing.allocator.free(cl.fc1_bias);
        std.testing.allocator.free(cl.up_w);
        std.testing.allocator.free(cl.fc2_w);
        std.testing.allocator.free(cl.fc2_bias);
    };

    const allocBind = struct {
        fn f16Vec(a: *arena_mod.WiredArena, buf: metal.Buffer, len: usize, seed_off: usize, scale: f32, bias_add: f32, cpu_out: *[]f32) !metal.BufferBinding {
            const raw = try a.allocAligned(len * 2, 64);
            const u16s: []u16 = @alignCast(std.mem.bytesAsSlice(u16, raw));
            cpu_out.* = try std.testing.allocator.alloc(f32, len);
            for (0..len) |idx| {
                const bits = quant.f32ToFloatBits(.f16, bias_add + synthWeight(idx + seed_off, scale));
                u16s[idx] = bits;
                cpu_out.*[idx] = quant.floatBitsToF32(.f16, bits);
            }
            return .{ .buffer = buf, .offset = a.offsetOf(raw.ptr) };
        }
    };

    for (0..num_layers) |l| {
        const s0 = (l + 1) * 1000;
        const n1_g_b = try allocBind.f16Vec(&master_arena, master_buf, vis_h, s0 + 1, 0.005, 1.0, &cpu_layers[l].n1_g);
        const n1_b_b = try allocBind.f16Vec(&master_arena, master_buf, vis_h, s0 + 2, 0.005, 0.0, &cpu_layers[l].n1_b);
        const n2_g_b = try allocBind.f16Vec(&master_arena, master_buf, vis_h, s0 + 3, 0.005, 1.0, &cpu_layers[l].n2_g);
        const n2_b_b = try allocBind.f16Vec(&master_arena, master_buf, vis_h, s0 + 4, 0.005, 0.0, &cpu_layers[l].n2_b);

        // Q4_AFFINE QKV projection + f16 bias
        const qkv_tmp = try std.testing.allocator.alloc(f32, 3 * vis_h * vis_h);
        defer std.testing.allocator.free(qkv_tmp);
        for (0..3 * vis_h * vis_h) |idx| qkv_tmp[idx] = synthWeight(idx + s0 + 5, 0.02);
        const qw_raw = try master_arena.allocAligned(qkv_qw_bytes, 64);
        const qsb_raw = try master_arena.allocAligned(qkv_sb_bytes, 64);
        const qsb: []quant.ScaleBias = @alignCast(std.mem.bytesAsSlice(quant.ScaleBias, qsb_raw));
        cpu_layers[l].qkv_w = try std.testing.allocator.alloc(f32, 3 * vis_h * vis_h);
        quantizeMatrixQ4(qkv_tmp, 3 * vis_h, vis_h, qw_raw, qsb, cpu_layers[l].qkv_w);
        const qkv_bias_b = try allocBind.f16Vec(&master_arena, master_buf, 3 * vis_h, s0 + 6, 0.005, 0.0, &cpu_layers[l].qkv_bias);

        const proj_w_b = try allocBind.f16Vec(&master_arena, master_buf, vis_h * vis_h, s0 + 7, 0.02, 0.0, &cpu_layers[l].proj_w);
        const proj_bias_b = try allocBind.f16Vec(&master_arena, master_buf, vis_h, s0 + 8, 0.005, 0.0, &cpu_layers[l].proj_bias);
        const fc1_w_b = try allocBind.f16Vec(&master_arena, master_buf, vis_inter * vis_h, s0 + 9, 0.02, 0.0, &cpu_layers[l].fc1_w);
        const fc1_bias_b = try allocBind.f16Vec(&master_arena, master_buf, vis_inter, s0 + 10, 0.005, 0.0, &cpu_layers[l].fc1_bias);
        _ = try allocBind.f16Vec(&master_arena, master_buf, vis_inter * vis_h, s0 + 11, 0.02, 0.0, &cpu_layers[l].up_w);
        const fc2_w_b = try allocBind.f16Vec(&master_arena, master_buf, vis_h * vis_inter, s0 + 12, 0.02, 0.0, &cpu_layers[l].fc2_w);
        const fc2_bias_b = try allocBind.f16Vec(&master_arena, master_buf, vis_h, s0 + 13, 0.005, 0.0, &cpu_layers[l].fc2_bias);

        blocks[l] = .{
            .norm1 = .{ .gamma = n1_g_b, .beta = n1_b_b, .dtype = .f16, .kind = .layer_norm, .eps = 1e-6 },
            .qkv = .{ .q4_affine = .{
                .qweight = .{ .buffer = master_buf, .offset = master_arena.offsetOf(qw_raw.ptr) },
                .scale_bias = .{ .buffer = master_buf, .offset = master_arena.offsetOf(qsb_raw.ptr) },
                .group_size = 64,
                .bias = qkv_bias_b,
                .bias_dtype = .f16,
            } },
            .proj = .{ .dense = .{ .weight = proj_w_b, .dtype = .f16, .bias = proj_bias_b } },
            .norm2 = .{ .gamma = n2_g_b, .beta = n2_b_b, .dtype = .f16, .kind = .layer_norm, .eps = 1e-6 },
            .mlp_fc1 = .{ .dense = .{ .weight = fc1_w_b, .dtype = .f16, .bias = fc1_bias_b } },
            .mlp_fc2 = .{ .dense = .{ .weight = fc2_w_b, .dtype = .f16, .bias = fc2_bias_b } },
            .use_window_attn = (l == 1), // layer 1 uses windowed bidirectional attn!
        };
    }

    var mn_g: []f32 = undefined;
    defer std.testing.allocator.free(mn_g);
    var mn_b: []f32 = undefined;
    defer std.testing.allocator.free(mn_b);
    var m1_w: []f32 = undefined;
    defer std.testing.allocator.free(m1_w);
    var m1_b: []f32 = undefined;
    defer std.testing.allocator.free(m1_b);
    var m2_w: []f32 = undefined;
    defer std.testing.allocator.free(m2_w);
    var m2_b: []f32 = undefined;
    defer std.testing.allocator.free(m2_b);

    const mn_g_b = try allocBind.f16Vec(&master_arena, master_buf, vis_h, 9001, 0.005, 1.0, &mn_g);
    const mn_b_b = try allocBind.f16Vec(&master_arena, master_buf, vis_h, 9002, 0.005, 0.0, &mn_b);
    const m1_w_b = try allocBind.f16Vec(&master_arena, master_buf, m_hid * m_in, 9003, 0.02, 0.0, &m1_w);
    const m1_b_b = try allocBind.f16Vec(&master_arena, master_buf, m_hid, 9004, 0.005, 0.0, &m1_b);
    const m2_w_b = try allocBind.f16Vec(&master_arena, master_buf, lm_h * m_hid, 9005, 0.02, 0.0, &m2_w);
    const m2_b_b = try allocBind.f16Vec(&master_arena, master_buf, lm_h, 9006, 0.005, 0.0, &m2_b);

    const merger: VisionMergerWeights = .{
        .norm = .{ .gamma = mn_g_b, .beta = mn_b_b, .dtype = .f16, .kind = .layer_norm, .eps = 1e-6 },
        .norm_placement = .pre_merge,
        .fc1 = .{ .dense = .{ .weight = m1_w_b, .dtype = .f16, .bias = m1_b_b } },
        .fc2 = .{ .dense = .{ .weight = m2_w_b, .dtype = .f16, .bias = m2_b_b } },
        .merger_hidden_size = m_hid,
        .act_kind = .gelu_tanh,
    };

    // Populate initial patch embeddings, 2D-RoPE coords, and window bounds in aliased scratch.
    const sc = rt.scratch.?;
    const x_init_f16 = sc.hiddenAF16();
    const x_init_f32 = try std.testing.allocator.alloc(f32, num_patches * vis_h);
    defer std.testing.allocator.free(x_init_f32);
    for (0..num_patches * vis_h) |idx| {
        x_init_f16[idx] = @floatCast(synthWeight(idx + 33, 0.05));
        x_init_f32[idx] = @floatCast(x_init_f16[idx]);
    }
    const coords_sl = sc.coordsSlice();
    const bounds_sl = sc.windowBoundsSlice();
    for (0..grid_h) |gh| {
        for (0..grid_w) |gw| {
            const p_i = gh * grid_w + gw;
            coords_sl[p_i] = .{ .h = @intCast(gh), .w = @intCast(gw) };
            bounds_sl[p_i] = if (p_i < 8) .{ .kv_start = 0, .kv_end = 8 } else .{ .kv_start = 8, .kv_end = num_patches };
        }
    }

    var frame = try compute.beginFrame(5000);
    errdefer frame.abort();
    const t0 = try vmstat.nowNs();
    const emitted = try rt.encodeTowerForward(&frame, &blocks, merger, grid_h, grid_w, 1, false);
    const sig = try frame.commit();
    try compute.wait(sig, 5000);
    const t1 = try vmstat.nowNs();
    const gpu_us = @as(f64, @floatFromInt(t1 - t0)) / 1000.0;
    try std.testing.expectEqual(@as(u32, 4), emitted);

    // Run full 3-layer ViT + PatchMerger CPU fp32 reference.
    const cur_ref = try std.testing.allocator.alloc(f32, num_patches * vis_h);
    defer std.testing.allocator.free(cur_ref);
    @memcpy(cur_ref, x_init_f32);
    const next_ref = try std.testing.allocator.alloc(f32, num_patches * vis_h);
    defer std.testing.allocator.free(next_ref);
    const norm_ref = try std.testing.allocator.alloc(f32, num_patches * vis_h);
    defer std.testing.allocator.free(norm_ref);
    const qkv_ref = try std.testing.allocator.alloc(f32, num_patches * 3 * vis_h);
    defer std.testing.allocator.free(qkv_ref);
    const q_ref = try std.testing.allocator.alloc(f32, num_patches * vis_h);
    defer std.testing.allocator.free(q_ref);
    const k_ref = try std.testing.allocator.alloc(f32, num_patches * vis_h);
    defer std.testing.allocator.free(k_ref);
    const v_ref = try std.testing.allocator.alloc(f32, num_patches * vis_h);
    defer std.testing.allocator.free(v_ref);
    const attn_ref = try std.testing.allocator.alloc(f32, num_patches * vis_h);
    defer std.testing.allocator.free(attn_ref);
    const mlp_ref = try std.testing.allocator.alloc(f32, num_patches * vis_inter);
    defer std.testing.allocator.free(mlp_ref);
    const score_scr = try std.testing.allocator.alloc(f64, num_patches);
    defer std.testing.allocator.free(score_scr);

    for (0..num_layers) |l| {
        const cl = cpu_layers[l];
        visionNormCpuRef(cur_ref, cl.n1_g, cl.n1_b, num_patches, vis_h, .layer_norm, 1e-6, norm_ref);
        visionLinearCpuRef(norm_ref, cl.qkv_w, cl.qkv_bias, null, num_patches, 3 * vis_h, vis_h, .none, qkv_ref);
        visionQkvRope2DCpuRef(qkv_ref, coords_sl[0..num_patches], num_patches, num_heads, head_dim, .split_3hd, false, 10000.0, q_ref, k_ref, v_ref);
        const wb_opt: ?[]const AttnWindowBound = if (l == 1) bounds_sl[0..num_patches] else null;
        visionBidirAttnCpuRef(q_ref, k_ref, v_ref, wb_opt, num_patches, num_heads, head_dim, attn_ref, score_scr);
        visionLinearCpuRef(attn_ref, cl.proj_w, cl.proj_bias, cur_ref, num_patches, vis_h, vis_h, .none, next_ref);
        visionNormCpuRef(next_ref, cl.n2_g, cl.n2_b, num_patches, vis_h, .layer_norm, 1e-6, norm_ref);
        visionLinearCpuRef(norm_ref, cl.fc1_w, cl.fc1_bias, null, num_patches, vis_inter, vis_h, .gelu_tanh, mlp_ref);
        visionLinearCpuRef(mlp_ref, cl.fc2_w, cl.fc2_bias, next_ref, num_patches, vis_h, vis_inter, .none, cur_ref);
    }
    visionNormCpuRef(cur_ref, mn_g, mn_b, num_patches, vis_h, .layer_norm, 1e-6, norm_ref);
    const mcat_ref = try std.testing.allocator.alloc(f32, emitted * m_in);
    defer std.testing.allocator.free(mcat_ref);
    visionSpatialMergeConcatCpuRef(norm_ref, grid_h, grid_w, sm, vis_h, 1, mcat_ref);
    const mmid_ref = try std.testing.allocator.alloc(f32, emitted * m_hid);
    defer std.testing.allocator.free(mmid_ref);
    visionLinearCpuRef(mcat_ref, m1_w, m1_b, null, emitted, m_hid, m_in, .gelu_tanh, mmid_ref);
    const lm_ref = try std.testing.allocator.alloc(f32, emitted * lm_h);
    defer std.testing.allocator.free(lm_ref);
    visionLinearCpuRef(mmid_ref, m2_w, m2_b, null, emitted, lm_h, m_hid, .none, lm_ref);

    const got_lm = sc.lmEmbedsF16();
    for (0..emitted * lm_h) |idx| {
        const g: f32 = @floatCast(got_lm[idx]);
        try std.testing.expectApproxEqAbs(lm_ref[idx], g, @max(0.05, @abs(lm_ref[idx]) * 0.05));
    }

    // 3. Verify `C16` / `D5` CoreML ANE vision-tower placement selection & live zero-copy ANE calibration!
    const base_facts = @import("aminfer").sysinfo.Facts{
        .page_size = page.APPLE_PAGE_SIZE,
        .memory_bytes = 24 * 1024 * 1024 * 1024,
        .performance_cores = 4,
        .efficiency_cores = 6,
        .user_wire_limit = 16 * 1024 * 1024 * 1024,
        .global_user_wire_limit = 16 * 1024 * 1024 * 1024,
        .gpu_wired_limit_mib = 16384,
        .has_sme2 = true,
    };
    const m4_profile = chip_profile.ChipProfile.resolve(base_facts, "Apple M4", 9, null);
    const m5_profile = chip_profile.ChipProfile.resolve(base_facts, "Apple M5", 10, null);
    try std.testing.expectEqual(VisionPlacement.ane, selectVisionPlacement(m4_profile, true, 64, 95.0, 100.0, .auto));
    try std.testing.expectEqual(VisionPlacement.gpu, selectVisionPlacement(m5_profile, true, 64, 95.0, 100.0, .auto));
    try std.testing.expectEqual(VisionPlacement.ane, selectVisionPlacement(m5_profile, true, 64, 80.0, 100.0, .auto));
    try std.testing.expectEqual(VisionPlacement.gpu, selectVisionPlacement(m4_profile, false, 64, 10.0, 100.0, .auto));

    var ane_w: [32 * 32]f16 = undefined;
    for (0..32 * 32) |idx| ane_w[idx] = @floatCast(synthWeight(idx + 11, 0.02));
    const spec_bytes = try metal.emitCoreMlLinearModel(std.testing.allocator, .{
        .op = .conv1x1,
        .batch_or_spatial = 32,
        .layers = &.{.{ .c_in = 32, .c_out = 32, .weights_f16 = &ane_w }},
    });
    defer std.testing.allocator.free(spec_bytes);

    var ane_offload = try VisionAneOffload.initFromSpecBytes(device, spec_bytes, 32, 32, 32, true, "x", "y");
    defer ane_offload.deinit();
    const cal = try ane_offload.calibrateAgainstGpuUs(m4_profile, gpu_us, 2, 5, .auto);
    try std.testing.expect(cal.ane_available);
    try std.testing.expect(cal.ane_us > 0.0);
    const pinfo = try ane_offload.predictZeroCopy();
    try std.testing.expect(pinfo.allOutputsZeroCopy());
}

test "G22 DeepSeek-style SwiGLU + RMSNorm + interleaved 2D-RoPE + post_merge Aligner end-to-end parity" {
    const device = try metal.Device.open();
    defer device.close();
    const lib = try metal.Library.openEmbedded(device);
    defer lib.close();
    var compute = try metal.Compute.init(device);
    defer compute.close();

    const vis_h: u32 = 64;
    const vis_inter: u32 = 96;
    const num_heads: u32 = 2;
    const head_dim: u32 = 32;
    const lm_h: u32 = 64;
    const sm: u32 = 2;
    const m_in: u32 = vis_h * sm * sm; // 256
    const m_hid: u32 = 64;
    const grid_h: u32 = 4;
    const grid_w: u32 = 4;
    const num_patches: u32 = grid_h * grid_w; // 16
    const num_layers: u32 = 2;

    const cfg: VisionTowerConfig = .{
        .has_vision = true,
        .num_layers = num_layers,
        .vis_hidden = vis_h,
        .vis_intermediate = vis_inter,
        .num_heads = num_heads,
        .head_dim = head_dim,
        .lm_hidden = lm_h,
        .merger_hidden = m_hid,
        .patch_size = 14,
        .t_patch = 1,
        .spatial_merge = sm,
        .max_patches = num_patches,
        .mlp_kind = .swiglu,
        .qkv_layout = .interleaved_h3d,
        .rope_interleaved = true,
        .rope_theta = 10000.0,
    };
    const vis_scratch_bytes = try VisionTowerScratch.requiredBytes(cfg);
    const per_layer_bytes = (vis_h * 2 + 3 * vis_h * vis_h + vis_h * vis_h + vis_inter * vis_h * 2 + vis_h * vis_inter) * 2;
    const aligner_bytes = (m_in + m_hid * m_in + m_hid + lm_h * m_hid + lm_h) * 2;
    const total_bytes = try page.alignUp(per_layer_bytes * num_layers + aligner_bytes + vis_scratch_bytes + 16384);

    var master_arena = try arena_mod.WiredArena.init(total_bytes, testBudget(total_bytes));
    defer master_arena.deinit();
    const pair = try master_arena.allocAliasedPair(vis_scratch_bytes, vis_scratch_bytes);
    const buf = try metal.Buffer.wrap(device, master_arena.bytes());
    defer buf.close();
    try compute.makeResident(&.{buf});
    defer compute.removeResident(&.{buf}) catch |e| {
        std.debug.print("removeResident err: {}\n", .{e});
    };

    var vis_arena = pair.arena_a;
    var rt = try VisionTowerRuntime.initAliased(device, lib, &vis_arena, buf, master_arena.offsetOf(pair.backing.ptr), cfg);
    defer rt.deinit();

    const fillMat = struct {
        fn run(a: *arena_mod.WiredArena, b: metal.Buffer, len: usize, seed_off: usize, scale: f32, bias_add: f32, out_f32: *[]f32) !metal.BufferBinding {
            const raw = try a.allocAligned(len * 2, 64);
            const u16s: []u16 = @alignCast(std.mem.bytesAsSlice(u16, raw));
            out_f32.* = try std.testing.allocator.alloc(f32, len);
            for (0..len) |idx| {
                const bits = quant.f32ToFloatBits(.f16, bias_add + synthWeight(idx + seed_off, scale));
                u16s[idx] = bits;
                out_f32.*[idx] = quant.floatBitsToF32(.f16, bits);
            }
            return .{ .buffer = b, .offset = a.offsetOf(raw.ptr) };
        }
    }.run;

    const LayerCpu = struct {
        n1_g: []f32,
        n2_g: []f32,
        qkv_w: []f32,
        proj_w: []f32,
        gate_w: []f32,
        up_w: []f32,
        down_w: []f32,
    };
    var blocks: [num_layers]VisionBlockWeights = undefined;
    var cpu_layers: [num_layers]LayerCpu = undefined;
    defer for (&cpu_layers) |*cl| {
        std.testing.allocator.free(cl.n1_g);
        std.testing.allocator.free(cl.n2_g);
        std.testing.allocator.free(cl.qkv_w);
        std.testing.allocator.free(cl.proj_w);
        std.testing.allocator.free(cl.gate_w);
        std.testing.allocator.free(cl.up_w);
        std.testing.allocator.free(cl.down_w);
    };

    for (0..num_layers) |l| {
        const s0 = (l + 1) * 2000;
        const n1_g_b = try fillMat(&master_arena, buf, vis_h, s0 + 1, 0.005, 1.0, &cpu_layers[l].n1_g);
        const n2_g_b = try fillMat(&master_arena, buf, vis_h, s0 + 2, 0.005, 1.0, &cpu_layers[l].n2_g);
        const qkv_w_b = try fillMat(&master_arena, buf, 3 * vis_h * vis_h, s0 + 3, 0.006, 0.0, &cpu_layers[l].qkv_w);
        const proj_w_b = try fillMat(&master_arena, buf, vis_h * vis_h, s0 + 4, 0.006, 0.0, &cpu_layers[l].proj_w);
        const gate_w_b = try fillMat(&master_arena, buf, vis_inter * vis_h, s0 + 5, 0.006, 0.0, &cpu_layers[l].gate_w);
        const up_w_b = try fillMat(&master_arena, buf, vis_inter * vis_h, s0 + 6, 0.006, 0.0, &cpu_layers[l].up_w);
        const down_w_b = try fillMat(&master_arena, buf, vis_h * vis_inter, s0 + 7, 0.006, 0.0, &cpu_layers[l].down_w);

        blocks[l] = .{
            .norm1 = .{ .gamma = n1_g_b, .beta = null, .dtype = .f16, .kind = .rms_norm, .eps = 1e-6 },
            .qkv = .{ .dense = .{ .weight = qkv_w_b, .dtype = .f16 } },
            .proj = .{ .dense = .{ .weight = proj_w_b, .dtype = .f16 } },
            .norm2 = .{ .gamma = n2_g_b, .beta = null, .dtype = .f16, .kind = .rms_norm, .eps = 1e-6 },
            .mlp_fc1 = .{ .dense = .{ .weight = gate_w_b, .dtype = .f16 } },
            .mlp_up = .{ .dense = .{ .weight = up_w_b, .dtype = .f16 } },
            .mlp_fc2 = .{ .dense = .{ .weight = down_w_b, .dtype = .f16 } },
            .use_window_attn = false,
        };
    }

    var an_g: []f32 = undefined;
    defer std.testing.allocator.free(an_g);
    var a1_w: []f32 = undefined;
    defer std.testing.allocator.free(a1_w);
    var a1_b: []f32 = undefined;
    defer std.testing.allocator.free(a1_b);
    var a2_w: []f32 = undefined;
    defer std.testing.allocator.free(a2_w);
    var a2_b: []f32 = undefined;
    defer std.testing.allocator.free(a2_b);

    const an_g_b = try fillMat(&master_arena, buf, m_in, 7001, 0.005, 1.0, &an_g);
    const a1_w_b = try fillMat(&master_arena, buf, m_hid * m_in, 7002, 0.006, 0.0, &a1_w);
    const a1_b_b = try fillMat(&master_arena, buf, m_hid, 7003, 0.005, 0.0, &a1_b);
    const a2_w_b = try fillMat(&master_arena, buf, lm_h * m_hid, 7004, 0.006, 0.0, &a2_w);
    const a2_b_b = try fillMat(&master_arena, buf, lm_h, 7005, 0.005, 0.0, &a2_b);

    const aligner: VisionMergerWeights = .{
        .norm = .{ .gamma = an_g_b, .beta = null, .dtype = .f16, .kind = .rms_norm, .eps = 1e-6 },
        .norm_placement = .post_merge,
        .fc1 = .{ .dense = .{ .weight = a1_w_b, .dtype = .f16, .bias = a1_b_b } },
        .fc2 = .{ .dense = .{ .weight = a2_w_b, .dtype = .f16, .bias = a2_b_b } },
        .merger_hidden_size = m_hid,
        .act_kind = .gelu_exact,
    };

    const sc = rt.scratch.?;
    const x_init_f16 = sc.hiddenAF16();
    const x_init_f32 = try std.testing.allocator.alloc(f32, num_patches * vis_h);
    defer std.testing.allocator.free(x_init_f32);
    for (0..num_patches * vis_h) |idx| {
        x_init_f16[idx] = @floatCast(synthWeight(idx + 55, 0.05));
        x_init_f32[idx] = @floatCast(x_init_f16[idx]);
    }
    const coords_sl = sc.coordsSlice();
    for (0..grid_h) |gh| {
        for (0..grid_w) |gw| {
            coords_sl[gh * grid_w + gw] = .{ .h = @intCast(gh), .w = @intCast(gw) };
        }
    }

    var frame = try compute.beginFrame(5000);
    errdefer frame.abort();
    const emitted = try rt.encodeTowerForward(&frame, &blocks, aligner, grid_h, grid_w, 1, false);
    const sig = try frame.commit();
    try compute.wait(sig, 5000);
    try std.testing.expectEqual(@as(u32, 4), emitted);

    const cur_ref = try std.testing.allocator.alloc(f32, num_patches * vis_h);
    defer std.testing.allocator.free(cur_ref);
    @memcpy(cur_ref, x_init_f32);
    const next_ref = try std.testing.allocator.alloc(f32, num_patches * vis_h);
    defer std.testing.allocator.free(next_ref);
    const norm_ref = try std.testing.allocator.alloc(f32, num_patches * vis_h);
    defer std.testing.allocator.free(norm_ref);
    const qkv_ref = try std.testing.allocator.alloc(f32, num_patches * 3 * vis_h);
    defer std.testing.allocator.free(qkv_ref);
    const q_ref = try std.testing.allocator.alloc(f32, num_patches * vis_h);
    defer std.testing.allocator.free(q_ref);
    const k_ref = try std.testing.allocator.alloc(f32, num_patches * vis_h);
    defer std.testing.allocator.free(k_ref);
    const v_ref = try std.testing.allocator.alloc(f32, num_patches * vis_h);
    defer std.testing.allocator.free(v_ref);
    const attn_ref = try std.testing.allocator.alloc(f32, num_patches * vis_h);
    defer std.testing.allocator.free(attn_ref);
    const gate_ref = try std.testing.allocator.alloc(f32, num_patches * vis_inter);
    defer std.testing.allocator.free(gate_ref);
    const up_ref = try std.testing.allocator.alloc(f32, num_patches * vis_inter);
    defer std.testing.allocator.free(up_ref);
    const swi_ref = try std.testing.allocator.alloc(f32, num_patches * vis_inter);
    defer std.testing.allocator.free(swi_ref);
    const score_scr = try std.testing.allocator.alloc(f64, num_patches);
    defer std.testing.allocator.free(score_scr);

    for (0..num_layers) |l| {
        const cl = cpu_layers[l];
        visionNormCpuRef(cur_ref, cl.n1_g, null, num_patches, vis_h, .rms_norm, 1e-6, norm_ref);
        visionLinearCpuRef(norm_ref, cl.qkv_w, null, null, num_patches, 3 * vis_h, vis_h, .none, qkv_ref);
        visionQkvRope2DCpuRef(qkv_ref, coords_sl[0..num_patches], num_patches, num_heads, head_dim, .interleaved_h3d, true, 10000.0, q_ref, k_ref, v_ref);
        visionBidirAttnCpuRef(q_ref, k_ref, v_ref, null, num_patches, num_heads, head_dim, attn_ref, score_scr);
        visionLinearCpuRef(attn_ref, cl.proj_w, null, cur_ref, num_patches, vis_h, vis_h, .none, next_ref);
        visionNormCpuRef(next_ref, cl.n2_g, null, num_patches, vis_h, .rms_norm, 1e-6, norm_ref);
        visionLinearCpuRef(norm_ref, cl.gate_w, null, null, num_patches, vis_inter, vis_h, .none, gate_ref);
        visionLinearCpuRef(norm_ref, cl.up_w, null, null, num_patches, vis_inter, vis_h, .none, up_ref);
        visionSwiGluCpuRef(gate_ref, up_ref, swi_ref);
        visionLinearCpuRef(swi_ref, cl.down_w, null, next_ref, num_patches, vis_h, vis_inter, .none, cur_ref);
    }
    const mcat_ref = try std.testing.allocator.alloc(f32, emitted * m_in);
    defer std.testing.allocator.free(mcat_ref);
    visionSpatialMergeConcatCpuRef(cur_ref, grid_h, grid_w, sm, vis_h, 1, mcat_ref);
    const mcat_norm_ref = try std.testing.allocator.alloc(f32, emitted * m_in);
    defer std.testing.allocator.free(mcat_norm_ref);
    visionNormCpuRef(mcat_ref, an_g, null, emitted, m_in, .rms_norm, 1e-6, mcat_norm_ref);
    const amid_ref = try std.testing.allocator.alloc(f32, emitted * m_hid);
    defer std.testing.allocator.free(amid_ref);
    visionLinearCpuRef(mcat_norm_ref, a1_w, a1_b, null, emitted, m_hid, m_in, .gelu_exact, amid_ref);
    const lm_ref = try std.testing.allocator.alloc(f32, emitted * lm_h);
    defer std.testing.allocator.free(lm_ref);
    visionLinearCpuRef(amid_ref, a2_w, a2_b, null, emitted, lm_h, m_hid, .none, lm_ref);

    const got_lm = sc.lmEmbedsF16();
    for (0..emitted * lm_h) |idx| {
        const g: f32 = @floatCast(got_lm[idx]);
        try std.testing.expectApproxEqAbs(lm_ref[idx], g, @max(0.01, @abs(lm_ref[idx]) * 0.02));
    }
}
