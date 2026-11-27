// M6.5 G14 zero-copy multimodal vision/video ingress Metal 4 kernels:
//   - Direct IOSurface MTLTexture reads (`BGRA8Unorm` and bi-planar `NV12`
//     `R8Unorm` + `RG8Unorm`, zero host memcpy)
//   - BT.709 YCbCr (full-range `420f` & video-range `420v`) / sRGB conversion
//     and per-channel `(x - mean) * inv_std` normalization
//   - Spatiotemporal tube patchify (`t_patch = 1` image, `t_patch = 2` video;
//     `patch_size = 14 / 16`) with optional `2x2` spatial patch merge
//   - Optional fused linear patch embedding (`f16` / `bf16` weights + bias ->
//     `hidden_size` visual tokens) using 4 KiB tiled threadgroup staging
//
// Compiled with `-std=metal4.0 -Wall -Werror -mmacosx-version-min=26.0`.
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

constant bool VISION_WEIGHT_BF16 [[function_constant(0)]];
constant bool VISION_OUT_BF16 [[function_constant(1)]];

constant bool VISION_HAS_W_BF16 = is_function_constant_defined(VISION_WEIGHT_BF16);
constant bool VISION_HAS_O_BF16 = is_function_constant_defined(VISION_OUT_BF16);
constant bool VISION_EFF_W_BF16 = VISION_HAS_W_BF16 ? VISION_WEIGHT_BF16 : false;
constant bool VISION_EFF_O_BF16 = VISION_HAS_O_BF16 ? VISION_OUT_BF16 : false;

struct VisionPatchifyParams {
    uint width;
    uint height;
    uint patch_size;
    uint t_patch;
    uint spatial_merge;
    uint hidden_size;
    uint is_full_range;
    uint apply_srgb_to_linear;
    uint has_bias;
    uint token_offset;
    float mean_r;
    float mean_g;
    float mean_b;
    float inv_std_r;
    float inv_std_g;
    float inv_std_b;
};

static_assert(sizeof(VisionPatchifyParams) == 64, "VisionPatchifyParams must be 64 bytes");

static inline float vision_load_f16_or_bf16(const device uchar *raw, ulong idx, bool is_bf16) {
    if (is_bf16) {
        return float(((const device bfloat *)raw)[idx]);
    }
    return float(((const device half *)raw)[idx]);
}

static inline void vision_store_f16_or_bf16(device uchar *raw, ulong idx, float val, bool is_bf16) {
    if (is_bf16) {
        ((device bfloat *)raw)[idx] = bfloat(val);
    } else {
        ((device half *)raw)[idx] = half(val);
    }
}

static inline float vision_srgb_to_linear_chan(float c) {
    c = clamp(c, 0.0f, 1.0f);
    if (c <= 0.04045f) {
        return c / 12.92f;
    }
    return pow((c + 0.055f) / 1.055f, 2.4f);
}

static inline float3 vision_bt709_nv12_to_rgb(float y, float2 cbcr, uint is_full_range) {
    float yf = y;
    float cb = cbcr.x - (128.0f / 255.0f);
    float cr = cbcr.y - (128.0f / 255.0f);
    if (is_full_range == 0u) {
        yf = (y - (16.0f / 255.0f)) * (255.0f / 219.0f);
        cb *= (255.0f / 224.0f);
        cr *= (255.0f / 224.0f);
    }
    const float r = yf + 1.5748f * cr;
    const float g = yf - 0.187324f * cb - 0.468124f * cr;
    const float b = yf + 1.8556f * cb;
    return clamp(float3(r, g, b), float3(0.0f), float3(1.0f));
}

static inline float3 vision_normalize_rgb(float3 rgb, constant VisionPatchifyParams &p) {
    if (p.apply_srgb_to_linear != 0u) {
        rgb = float3(
            vision_srgb_to_linear_chan(rgb.x),
            vision_srgb_to_linear_chan(rgb.y),
            vision_srgb_to_linear_chan(rgb.z)
        );
    }
    return float3(
        (rgb.x - p.mean_r) * p.inv_std_r,
        (rgb.y - p.mean_g) * p.inv_std_g,
        (rgb.z - p.mean_b) * p.inv_std_b
    );
}

struct VisionPixelCoord {
    uint x;
    uint y;
    uint dt;
    uint k_base;
};

static inline VisionPixelCoord vision_decode_pixel(uint tok,
                                                   uint pix,
                                                   uint merged_w,
                                                   uint patch_area,
                                                   uint tube_pixels,
                                                   constant VisionPatchifyParams &p) {
    const uint sub_idx = pix / tube_pixels;
    const uint rem = pix - sub_idx * tube_pixels;
    const uint dt = rem / patch_area;
    const uint rem_xy = rem - dt * patch_area;
    const uint dy = rem_xy / p.patch_size;
    const uint dx = rem_xy - dy * p.patch_size;

    const uint mh = tok / merged_w;
    const uint mw = tok - mh * merged_w;
    const uint sub_y = (p.spatial_merge == 2u) ? (sub_idx >> 1) : 0u;
    const uint sub_x = (p.spatial_merge == 2u) ? (sub_idx & 1u) : 0u;
    const uint ph = mh * p.spatial_merge + sub_y;
    const uint pw = mw * p.spatial_merge + sub_x;

    VisionPixelCoord out;
    out.x = pw * p.patch_size + dx;
    out.y = ph * p.patch_size + dy;
    out.dt = dt;
    out.k_base = sub_idx * (3u * tube_pixels) + rem;
    return out;
}

// ===========================================================================
// Raw Patchify (+ optional 2x2 spatial-temporal merge) directly from IOSurface
// Grid: (num_tokens, 1, 1) threadgroups, (256, 1, 1) threads.
// Output layout: `[token_offset + num_tokens, token_in_dim]` in f16 or bf16,
// where each sub-patch is channel-first `[3, t_patch, patch_size, patch_size]`.
// ===========================================================================

kernel void ami_vision_patchify_bgra8(device uchar *out [[buffer(0)]],
                                      constant VisionPatchifyParams &p [[buffer(1)]],
                                      texture2d<float, access::read> frame0 [[texture(0)]],
                                      texture2d<float, access::read> frame1 [[texture(1)]],
                                      uint tok [[threadgroup_position_in_grid]],
                                      uint tid [[thread_index_in_threadgroup]],
                                      uint tg_size [[threads_per_threadgroup]]) {
    const uint grid_h = p.height / p.patch_size;
    const uint grid_w = p.width / p.patch_size;
    const uint merged_h = grid_h / p.spatial_merge;
    const uint merged_w = grid_w / p.spatial_merge;
    const uint num_tokens = merged_h * merged_w;
    if (tok >= num_tokens) {
        return;
    }

    const uint patch_area = p.patch_size * p.patch_size;
    const uint tube_pixels = p.t_patch * patch_area;
    const uint merge_count = p.spatial_merge * p.spatial_merge;
    const uint num_pixels = merge_count * tube_pixels;
    const uint token_in_dim = merge_count * 3u * tube_pixels;
    const ulong tok_base = (ulong)(p.token_offset + tok) * (ulong)token_in_dim;

    for (uint pix = tid; pix < num_pixels; pix += tg_size) {
        const VisionPixelCoord c = vision_decode_pixel(tok, pix, merged_w, patch_area, tube_pixels, p);
        const float4 rgba = (c.dt == 0u)
            ? frame0.read(uint2(c.x, c.y))
            : frame1.read(uint2(c.x, c.y));
        const float3 norm = vision_normalize_rgb(rgba.rgb, p);

        vision_store_f16_or_bf16(out, tok_base + (ulong)c.k_base, norm.x, VISION_OUT_BF16);
        vision_store_f16_or_bf16(out, tok_base + (ulong)(c.k_base + tube_pixels), norm.y, VISION_OUT_BF16);
        vision_store_f16_or_bf16(out, tok_base + (ulong)(c.k_base + 2u * tube_pixels), norm.z, VISION_OUT_BF16);
    }
}

kernel void ami_vision_patchify_nv12(device uchar *out [[buffer(0)]],
                                     constant VisionPatchifyParams &p [[buffer(1)]],
                                     texture2d<float, access::read> y0 [[texture(0)]],
                                     texture2d<float, access::read> cbcr0 [[texture(1)]],
                                     texture2d<float, access::read> y1 [[texture(2)]],
                                     texture2d<float, access::read> cbcr1 [[texture(3)]],
                                     uint tok [[threadgroup_position_in_grid]],
                                     uint tid [[thread_index_in_threadgroup]],
                                     uint tg_size [[threads_per_threadgroup]]) {
    const uint grid_h = p.height / p.patch_size;
    const uint grid_w = p.width / p.patch_size;
    const uint merged_h = grid_h / p.spatial_merge;
    const uint merged_w = grid_w / p.spatial_merge;
    const uint num_tokens = merged_h * merged_w;
    if (tok >= num_tokens) {
        return;
    }

    const uint patch_area = p.patch_size * p.patch_size;
    const uint tube_pixels = p.t_patch * patch_area;
    const uint merge_count = p.spatial_merge * p.spatial_merge;
    const uint num_pixels = merge_count * tube_pixels;
    const uint token_in_dim = merge_count * 3u * tube_pixels;
    const ulong tok_base = (ulong)(p.token_offset + tok) * (ulong)token_in_dim;

    for (uint pix = tid; pix < num_pixels; pix += tg_size) {
        const VisionPixelCoord c = vision_decode_pixel(tok, pix, merged_w, patch_area, tube_pixels, p);
        const uint2 luma_xy = uint2(c.x, c.y);
        const uint2 chroma_xy = uint2(c.x >> 1, c.y >> 1);
        const float y_val = (c.dt == 0u) ? y0.read(luma_xy).r : y1.read(luma_xy).r;
        const float2 cbcr_val = (c.dt == 0u) ? cbcr0.read(chroma_xy).rg : cbcr1.read(chroma_xy).rg;
        const float3 rgb = vision_bt709_nv12_to_rgb(y_val, cbcr_val, p.is_full_range);
        const float3 norm = vision_normalize_rgb(rgb, p);

        vision_store_f16_or_bf16(out, tok_base + (ulong)c.k_base, norm.x, VISION_OUT_BF16);
        vision_store_f16_or_bf16(out, tok_base + (ulong)(c.k_base + tube_pixels), norm.y, VISION_OUT_BF16);
        vision_store_f16_or_bf16(out, tok_base + (ulong)(c.k_base + 2u * tube_pixels), norm.z, VISION_OUT_BF16);
    }
}

// ===========================================================================
// Fused Patchify + Optional 2x2 Spatial Merge + Linear Patch Embedding
// Directly transforms IOSurface MTLTexture planes -> `[num_tokens, hidden_size]`
// visual tokens with fp32 accumulation and zero intermediate global traffic.
// Grid: (ceil(hidden_size / 8), num_tokens, 1) threadgroups, (256, 1, 1) threads
// (8 simdgroups per threadgroup, 1 output channel per simdgroup).
// ===========================================================================

kernel void ami_vision_patchify_embed_bgra8(const device uchar *weight [[buffer(0)]],
                                            const device uchar *bias [[buffer(1)]],
                                            device uchar *out [[buffer(2)]],
                                            constant VisionPatchifyParams &p [[buffer(3)]],
                                            texture2d<float, access::read> frame0 [[texture(0)]],
                                            texture2d<float, access::read> frame1 [[texture(1)]],
                                            uint2 tg_pos [[threadgroup_position_in_grid]],
                                            uint tid [[thread_index_in_threadgroup]],
                                            uint sg_id [[simdgroup_index_in_threadgroup]],
                                            uint lane [[thread_index_in_simdgroup]]) {
    const uint grid_h = p.height / p.patch_size;
    const uint grid_w = p.width / p.patch_size;
    const uint merged_h = grid_h / p.spatial_merge;
    const uint merged_w = grid_w / p.spatial_merge;
    const uint num_tokens = merged_h * merged_w;
    const uint tok = tg_pos.y;
    if (tok >= num_tokens) {
        return;
    }

    const uint h = tg_pos.x * 8u + sg_id;
    const bool valid_h = (h < p.hidden_size);

    const uint patch_area = p.patch_size * p.patch_size;
    const uint tube_pixels = p.t_patch * patch_area;
    const uint merge_count = p.spatial_merge * p.spatial_merge;
    const uint num_pixels = merge_count * tube_pixels;
    const uint token_in_dim = merge_count * 3u * tube_pixels;
    const ulong w_row = (ulong)h * (ulong)token_in_dim;

    threadgroup float tg_rgb[256u * 3u];
    threadgroup uint tg_kbase[256u];

    float acc = 0.0f;

    for (uint tile_start = 0u; tile_start < num_pixels; tile_start += 256u) {
        const uint tile_count = min(256u, num_pixels - tile_start);
        if (tid < tile_count) {
            const uint pix = tile_start + tid;
            const VisionPixelCoord c = vision_decode_pixel(tok, pix, merged_w, patch_area, tube_pixels, p);
            const float4 rgba = (c.dt == 0u)
                ? frame0.read(uint2(c.x, c.y))
                : frame1.read(uint2(c.x, c.y));
            const float3 norm = vision_normalize_rgb(rgba.rgb, p);
            tg_rgb[tid * 3u + 0u] = norm.x;
            tg_rgb[tid * 3u + 1u] = norm.y;
            tg_rgb[tid * 3u + 2u] = norm.z;
            tg_kbase[tid] = c.k_base;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (valid_h) {
            for (uint i = lane; i < tile_count; i += 32u) {
                const ulong kb = w_row + (ulong)tg_kbase[i];
                const float wr = vision_load_f16_or_bf16(weight, kb, VISION_WEIGHT_BF16);
                const float wg = vision_load_f16_or_bf16(weight, kb + (ulong)tube_pixels, VISION_WEIGHT_BF16);
                const float wb = vision_load_f16_or_bf16(weight, kb + (ulong)(2u * tube_pixels), VISION_WEIGHT_BF16);
                acc = fma(tg_rgb[i * 3u + 0u], wr, acc);
                acc = fma(tg_rgb[i * 3u + 1u], wg, acc);
                acc = fma(tg_rgb[i * 3u + 2u], wb, acc);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (valid_h) {
        acc = simd_sum(acc);
        if (lane == 0u) {
            if (p.has_bias != 0u) {
                acc += vision_load_f16_or_bf16(bias, (ulong)h, VISION_WEIGHT_BF16);
            }
            const ulong out_idx = (ulong)(p.token_offset + tok) * (ulong)p.hidden_size + (ulong)h;
            vision_store_f16_or_bf16(out, out_idx, acc, VISION_OUT_BF16);
        }
    }
}

kernel void ami_vision_patchify_embed_nv12(const device uchar *weight [[buffer(0)]],
                                           const device uchar *bias [[buffer(1)]],
                                           device uchar *out [[buffer(2)]],
                                           constant VisionPatchifyParams &p [[buffer(3)]],
                                           texture2d<float, access::read> y0 [[texture(0)]],
                                           texture2d<float, access::read> cbcr0 [[texture(1)]],
                                           texture2d<float, access::read> y1 [[texture(2)]],
                                           texture2d<float, access::read> cbcr1 [[texture(3)]],
                                           uint2 tg_pos [[threadgroup_position_in_grid]],
                                           uint tid [[thread_index_in_threadgroup]],
                                           uint sg_id [[simdgroup_index_in_threadgroup]],
                                           uint lane [[thread_index_in_simdgroup]]) {
    const uint grid_h = p.height / p.patch_size;
    const uint grid_w = p.width / p.patch_size;
    const uint merged_h = grid_h / p.spatial_merge;
    const uint merged_w = grid_w / p.spatial_merge;
    const uint num_tokens = merged_h * merged_w;
    const uint tok = tg_pos.y;
    if (tok >= num_tokens) {
        return;
    }

    const uint h = tg_pos.x * 8u + sg_id;
    const bool valid_h = (h < p.hidden_size);

    const uint patch_area = p.patch_size * p.patch_size;
    const uint tube_pixels = p.t_patch * patch_area;
    const uint merge_count = p.spatial_merge * p.spatial_merge;
    const uint num_pixels = merge_count * tube_pixels;
    const uint token_in_dim = merge_count * 3u * tube_pixels;
    const ulong w_row = (ulong)h * (ulong)token_in_dim;

    threadgroup float tg_rgb[256u * 3u];
    threadgroup uint tg_kbase[256u];

    float acc = 0.0f;

    for (uint tile_start = 0u; tile_start < num_pixels; tile_start += 256u) {
        const uint tile_count = min(256u, num_pixels - tile_start);
        if (tid < tile_count) {
            const uint pix = tile_start + tid;
            const VisionPixelCoord c = vision_decode_pixel(tok, pix, merged_w, patch_area, tube_pixels, p);
            const uint2 luma_xy = uint2(c.x, c.y);
            const uint2 chroma_xy = uint2(c.x >> 1, c.y >> 1);
            const float y_val = (c.dt == 0u) ? y0.read(luma_xy).r : y1.read(luma_xy).r;
            const float2 cbcr_val = (c.dt == 0u) ? cbcr0.read(chroma_xy).rg : cbcr1.read(chroma_xy).rg;
            const float3 rgb = vision_bt709_nv12_to_rgb(y_val, cbcr_val, p.is_full_range);
            const float3 norm = vision_normalize_rgb(rgb, p);
            tg_rgb[tid * 3u + 0u] = norm.x;
            tg_rgb[tid * 3u + 1u] = norm.y;
            tg_rgb[tid * 3u + 2u] = norm.z;
            tg_kbase[tid] = c.k_base;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (valid_h) {
            for (uint i = lane; i < tile_count; i += 32u) {
                const ulong kb = w_row + (ulong)tg_kbase[i];
                const float wr = vision_load_f16_or_bf16(weight, kb, VISION_WEIGHT_BF16);
                const float wg = vision_load_f16_or_bf16(weight, kb + (ulong)tube_pixels, VISION_WEIGHT_BF16);
                const float wb = vision_load_f16_or_bf16(weight, kb + (ulong)(2u * tube_pixels), VISION_WEIGHT_BF16);
                acc = fma(tg_rgb[i * 3u + 0u], wr, acc);
                acc = fma(tg_rgb[i * 3u + 1u], wg, acc);
                acc = fma(tg_rgb[i * 3u + 2u], wb, acc);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (valid_h) {
        acc = simd_sum(acc);
        if (lane == 0u) {
            if (p.has_bias != 0u) {
                acc += vision_load_f16_or_bf16(bias, (ulong)h, VISION_WEIGHT_BF16);
            }
            const ulong out_idx = (ulong)(p.token_offset + tok) * (ulong)p.hidden_size + (ulong)h;
            vision_store_f16_or_bf16(out, out_idx, acc, VISION_OUT_BF16);
        }
    }
}

// ===========================================================================
// M7.8 G22: Dynamic Aspect-Ratio AnyRes Bilinear Resize + Tube Patchify +
// Fused Linear Patch Embedding directly over zero-copy IOSurface MTLTextures
// ===========================================================================

struct VisionAnyResParams {
    uint src_width;
    uint src_height;
    uint dst_width;
    uint dst_height;
    float src_x0;
    float src_y0;
    float src_w;
    float src_h;
    uint patch_size;
    uint t_patch;
    uint hidden_size;
    uint is_full_range;
    uint apply_srgb_to_linear;
    uint has_bias;
    uint has_pos_embed;
    uint token_offset;
    float mean_r;
    float mean_g;
    float mean_b;
    float inv_std_r;
    float inv_std_g;
    float inv_std_b;
    uint _pad0;
    uint _pad1;
};

static_assert(sizeof(VisionAnyResParams) == 96, "VisionAnyResParams must be 96 bytes");

static inline float3 vision_normalize_rgb_anyres(float3 rgb, constant VisionAnyResParams &p) {
    if (p.apply_srgb_to_linear != 0u) {
        rgb = float3(
            vision_srgb_to_linear_chan(rgb.x),
            vision_srgb_to_linear_chan(rgb.y),
            vision_srgb_to_linear_chan(rgb.z)
        );
    }
    return float3(
        (rgb.x - p.mean_r) * p.inv_std_r,
        (rgb.y - p.mean_g) * p.inv_std_g,
        (rgb.z - p.mean_b) * p.inv_std_b
    );
}

static inline float3 vision_bilinear_bgra8(texture2d<float, access::read> tex,
                                           uint dst_x,
                                           uint dst_y,
                                           constant VisionAnyResParams &p) {
    float fx = p.src_x0 + (float(dst_x) + 0.5f) * (p.src_w / float(p.dst_width)) - 0.5f;
    float fy = p.src_y0 + (float(dst_y) + 0.5f) * (p.src_h / float(p.dst_height)) - 0.5f;
    const float max_x = max(p.src_x0, min(float(p.src_width - 1u), p.src_x0 + p.src_w - 1.0f));
    const float max_y = max(p.src_y0, min(float(p.src_height - 1u), p.src_y0 + p.src_h - 1.0f));
    fx = clamp(fx, p.src_x0, max_x);
    fy = clamp(fy, p.src_y0, max_y);
    const uint x0 = uint(floor(fx));
    const uint y0 = uint(floor(fy));
    const uint x1 = min(x0 + 1u, p.src_width - 1u);
    const uint y1 = min(y0 + 1u, p.src_height - 1u);
    const float wx = fx - float(x0);
    const float wy = fy - float(y0);

    const float3 c00 = tex.read(uint2(x0, y0)).rgb;
    const float3 c10 = tex.read(uint2(x1, y0)).rgb;
    const float3 c01 = tex.read(uint2(x0, y1)).rgb;
    const float3 c11 = tex.read(uint2(x1, y1)).rgb;
    const float3 top = mix(c00, c10, wx);
    const float3 bot = mix(c01, c11, wx);
    return mix(top, bot, wy);
}

static inline float3 vision_bilinear_nv12(texture2d<float, access::read> y_tex,
                                          texture2d<float, access::read> uv_tex,
                                          uint dst_x,
                                          uint dst_y,
                                          constant VisionAnyResParams &p) {
    float fx = p.src_x0 + (float(dst_x) + 0.5f) * (p.src_w / float(p.dst_width)) - 0.5f;
    float fy = p.src_y0 + (float(dst_y) + 0.5f) * (p.src_h / float(p.dst_height)) - 0.5f;
    const float max_x = max(p.src_x0, min(float(p.src_width - 1u), p.src_x0 + p.src_w - 1.0f));
    const float max_y = max(p.src_y0, min(float(p.src_height - 1u), p.src_y0 + p.src_h - 1.0f));
    fx = clamp(fx, p.src_x0, max_x);
    fy = clamp(fy, p.src_y0, max_y);
    const uint x0 = uint(floor(fx));
    const uint y0 = uint(floor(fy));
    const uint x1 = min(x0 + 1u, p.src_width - 1u);
    const uint y1 = min(y0 + 1u, p.src_height - 1u);
    const float wx = fx - float(x0);
    const float wy = fy - float(y0);

    const float3 c00 = vision_bt709_nv12_to_rgb(y_tex.read(uint2(x0, y0)).r, uv_tex.read(uint2(x0 >> 1, y0 >> 1)).rg, p.is_full_range);
    const float3 c10 = vision_bt709_nv12_to_rgb(y_tex.read(uint2(x1, y0)).r, uv_tex.read(uint2(x1 >> 1, y0 >> 1)).rg, p.is_full_range);
    const float3 c01 = vision_bt709_nv12_to_rgb(y_tex.read(uint2(x0, y1)).r, uv_tex.read(uint2(x0 >> 1, y1 >> 1)).rg, p.is_full_range);
    const float3 c11 = vision_bt709_nv12_to_rgb(y_tex.read(uint2(x1, y1)).r, uv_tex.read(uint2(x1 >> 1, y1 >> 1)).rg, p.is_full_range);
    return mix(mix(c00, c10, wx), mix(c01, c11, wx), wy);
}

kernel void ami_vision_anyres_patchify_embed_bgra8(const device uchar *weight [[buffer(0)]],
                                                   const device uchar *bias [[buffer(1)]],
                                                   const device uchar *pos_embed [[buffer(2)]],
                                                   device uchar *out [[buffer(3)]],
                                                   constant VisionAnyResParams &p [[buffer(4)]],
                                                   texture2d<float, access::read> frame0 [[texture(0)]],
                                                   texture2d<float, access::read> frame1 [[texture(1)]],
                                                   uint2 tg_pos [[threadgroup_position_in_grid]],
                                                   uint tid [[thread_index_in_threadgroup]],
                                                   uint sg_id [[simdgroup_index_in_threadgroup]],
                                                   uint lane [[thread_index_in_simdgroup]]) {
    const uint grid_h = p.dst_height / p.patch_size;
    const uint grid_w = p.dst_width / p.patch_size;
    const uint num_tokens = grid_h * grid_w;
    const uint tok = tg_pos.y;
    if (tok >= num_tokens) {
        return;
    }

    const uint h = tg_pos.x * 8u + sg_id;
    const bool valid_h = (h < p.hidden_size);
    const uint ph = tok / grid_w;
    const uint pw = tok - ph * grid_w;
    const uint patch_area = p.patch_size * p.patch_size;
    const uint tube_pixels = p.t_patch * patch_area;
    const uint token_in_dim = 3u * tube_pixels;
    const ulong w_row = (ulong)h * (ulong)token_in_dim;

    threadgroup float tg_rgb[256u * 3u];
    float acc = 0.0f;

    for (uint tile_start = 0u; tile_start < tube_pixels; tile_start += 256u) {
        const uint tile_count = min(256u, tube_pixels - tile_start);
        if (tid < tile_count) {
            const uint pix = tile_start + tid;
            const uint dt = pix / patch_area;
            const uint rem_xy = pix - dt * patch_area;
            const uint dy = rem_xy / p.patch_size;
            const uint dx = rem_xy - dy * p.patch_size;
            const uint dst_x = pw * p.patch_size + dx;
            const uint dst_y = ph * p.patch_size + dy;
            const float3 rgb = (dt == 0u)
                ? vision_bilinear_bgra8(frame0, dst_x, dst_y, p)
                : vision_bilinear_bgra8(frame1, dst_x, dst_y, p);
            const float3 norm = vision_normalize_rgb_anyres(rgb, p);
            tg_rgb[tid * 3u + 0u] = norm.x;
            tg_rgb[tid * 3u + 1u] = norm.y;
            tg_rgb[tid * 3u + 2u] = norm.z;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (valid_h) {
            for (uint i = lane; i < tile_count; i += 32u) {
                const ulong kb = w_row + (ulong)(tile_start + i);
                const float wr = vision_load_f16_or_bf16(weight, kb, VISION_EFF_W_BF16);
                const float wg = vision_load_f16_or_bf16(weight, kb + (ulong)tube_pixels, VISION_EFF_W_BF16);
                const float wb = vision_load_f16_or_bf16(weight, kb + (ulong)(2u * tube_pixels), VISION_EFF_W_BF16);
                acc = fma(tg_rgb[i * 3u + 0u], wr, acc);
                acc = fma(tg_rgb[i * 3u + 1u], wg, acc);
                acc = fma(tg_rgb[i * 3u + 2u], wb, acc);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (valid_h) {
        acc = simd_sum(acc);
        if (lane == 0u) {
            if (p.has_bias != 0u) {
                acc += vision_load_f16_or_bf16(bias, (ulong)h, VISION_EFF_W_BF16);
            }
            const ulong out_idx = (ulong)(p.token_offset + tok) * (ulong)p.hidden_size + (ulong)h;
            if (p.has_pos_embed != 0u) {
                acc += vision_load_f16_or_bf16(pos_embed, out_idx, VISION_EFF_W_BF16);
            }
            vision_store_f16_or_bf16(out, out_idx, acc, VISION_EFF_O_BF16);
        }
    }
}

kernel void ami_vision_anyres_patchify_embed_nv12(const device uchar *weight [[buffer(0)]],
                                                  const device uchar *bias [[buffer(1)]],
                                                  const device uchar *pos_embed [[buffer(2)]],
                                                  device uchar *out [[buffer(3)]],
                                                  constant VisionAnyResParams &p [[buffer(4)]],
                                                  texture2d<float, access::read> y0 [[texture(0)]],
                                                  texture2d<float, access::read> cbcr0 [[texture(1)]],
                                                  texture2d<float, access::read> y1 [[texture(2)]],
                                                  texture2d<float, access::read> cbcr1 [[texture(3)]],
                                                  uint2 tg_pos [[threadgroup_position_in_grid]],
                                                  uint tid [[thread_index_in_threadgroup]],
                                                  uint sg_id [[simdgroup_index_in_threadgroup]],
                                                  uint lane [[thread_index_in_simdgroup]]) {
    const uint grid_h = p.dst_height / p.patch_size;
    const uint grid_w = p.dst_width / p.patch_size;
    const uint num_tokens = grid_h * grid_w;
    const uint tok = tg_pos.y;
    if (tok >= num_tokens) {
        return;
    }

    const uint h = tg_pos.x * 8u + sg_id;
    const bool valid_h = (h < p.hidden_size);
    const uint ph = tok / grid_w;
    const uint pw = tok - ph * grid_w;
    const uint patch_area = p.patch_size * p.patch_size;
    const uint tube_pixels = p.t_patch * patch_area;
    const uint token_in_dim = 3u * tube_pixels;
    const ulong w_row = (ulong)h * (ulong)token_in_dim;

    threadgroup float tg_rgb[256u * 3u];
    float acc = 0.0f;

    for (uint tile_start = 0u; tile_start < tube_pixels; tile_start += 256u) {
        const uint tile_count = min(256u, tube_pixels - tile_start);
        if (tid < tile_count) {
            const uint pix = tile_start + tid;
            const uint dt = pix / patch_area;
            const uint rem_xy = pix - dt * patch_area;
            const uint dy = rem_xy / p.patch_size;
            const uint dx = rem_xy - dy * p.patch_size;
            const uint dst_x = pw * p.patch_size + dx;
            const uint dst_y = ph * p.patch_size + dy;
            const float3 rgb = (dt == 0u)
                ? vision_bilinear_nv12(y0, cbcr0, dst_x, dst_y, p)
                : vision_bilinear_nv12(y1, cbcr1, dst_x, dst_y, p);
            const float3 norm = vision_normalize_rgb_anyres(rgb, p);
            tg_rgb[tid * 3u + 0u] = norm.x;
            tg_rgb[tid * 3u + 1u] = norm.y;
            tg_rgb[tid * 3u + 2u] = norm.z;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (valid_h) {
            for (uint i = lane; i < tile_count; i += 32u) {
                const ulong kb = w_row + (ulong)(tile_start + i);
                const float wr = vision_load_f16_or_bf16(weight, kb, VISION_EFF_W_BF16);
                const float wg = vision_load_f16_or_bf16(weight, kb + (ulong)tube_pixels, VISION_EFF_W_BF16);
                const float wb = vision_load_f16_or_bf16(weight, kb + (ulong)(2u * tube_pixels), VISION_EFF_W_BF16);
                acc = fma(tg_rgb[i * 3u + 0u], wr, acc);
                acc = fma(tg_rgb[i * 3u + 1u], wg, acc);
                acc = fma(tg_rgb[i * 3u + 2u], wb, acc);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (valid_h) {
        acc = simd_sum(acc);
        if (lane == 0u) {
            if (p.has_bias != 0u) {
                acc += vision_load_f16_or_bf16(bias, (ulong)h, VISION_EFF_W_BF16);
            }
            const ulong out_idx = (ulong)(p.token_offset + tok) * (ulong)p.hidden_size + (ulong)h;
            if (p.has_pos_embed != 0u) {
                acc += vision_load_f16_or_bf16(pos_embed, out_idx, VISION_EFF_W_BF16);
            }
            vision_store_f16_or_bf16(out, out_idx, acc, VISION_EFF_O_BF16);
        }
    }
}

static inline float vision_apply_act(float v, uint act_kind) {
    if (act_kind == 1u) {
        // PyTorch `gelu_pytorch_tanh`: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
        // Clamp `inner` to [-20, 20] so Metal fast-math `tanh` never evaluates `inf / inf = NaN`.
        const float inner = clamp(0.7978845608028654f * fma(0.044715f * v * v, v, v), -20.0f, 20.0f);
        return 0.5f * v * (1.0f + tanh(inner));
    }
    if (act_kind == 2u) {
        // Exact GELU via Abramowitz-Stegun erf (max error < 1.5e-7; matches PyTorch `nn.GELU()`).
        const float z = v * 0.7071067811865475f;
        const float az = fabs(z);
        const float t = 1.0f / fma(0.3275911f, az, 1.0f);
        const float poly = (((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t - 0.284496736f) * t) + 0.254829592f) * t;
        const float erf_a = 1.0f - poly * exp(-min(az * az, 80.0f));
        const float erf_z = (z >= 0.0f) ? erf_a : -erf_a;
        return 0.5f * v * (1.0f + erf_z);
    }
    return v;
}

// ===========================================================================
// M7.8 G22: ViT LayerNorm / RMSNorm (`ami_vision_norm`)
// Grid: (num_tokens, 1, 1), threadgroup: (256, 1, 1) (8 simdgroups).
// ===========================================================================

struct VisionNormParams {
    uint num_tokens;
    uint dim;
    float eps;
    uint is_rmsnorm;
    uint has_bias;
    uint act_kind; // 0 = none, 1 = gelu_tanh, 2 = gelu_exact (Glm5Next post_projection_norm + nn.GELU)
    uint _pad1;
    uint _pad2;
};

static_assert(sizeof(VisionNormParams) == 32, "VisionNormParams must be 32 bytes");

kernel void ami_vision_norm(const device half *x [[buffer(0)]],
                            const device uchar *gamma [[buffer(1)]],
                            const device uchar *beta [[buffer(2)]],
                            device half *out [[buffer(3)]],
                            constant VisionNormParams &p [[buffer(4)]],
                            uint tok [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint sg_id [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint tg_size [[threads_per_threadgroup]]) {
    if (tok >= p.num_tokens) {
        return;
    }
    const ulong row_base = (ulong)tok * (ulong)p.dim;

    float local_sum = 0.0f;
    float local_sq = 0.0f;
    for (uint d = tid; d < p.dim; d += tg_size) {
        const float v = float(x[row_base + d]);
        local_sum += v;
        local_sq = fma(v, v, local_sq);
    }

    local_sum = simd_sum(local_sum);
    local_sq = simd_sum(local_sq);

    threadgroup float tg_sum[8];
    threadgroup float tg_sq[8];
    if (lane == 0u) {
        tg_sum[sg_id] = local_sum;
        tg_sq[sg_id] = local_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float mean = 0.0f;
    float inv_std = 1.0f;
    if (sg_id == 0u) {
        const float s = (lane < 8u) ? tg_sum[lane] : 0.0f;
        const float q = (lane < 8u) ? tg_sq[lane] : 0.0f;
        const float tot_s = simd_sum(s);
        const float tot_q = simd_sum(q);
        if (lane == 0u) {
            const float inv_n = 1.0f / float(p.dim);
            if (p.is_rmsnorm != 0u) {
                tg_sum[0] = 0.0f;
                tg_sq[0] = rsqrt(max(0.0f, tot_q * inv_n) + p.eps);
            } else {
                const float mu = tot_s * inv_n;
                const float var = max(0.0f, tot_q * inv_n - mu * mu);
                tg_sum[0] = mu;
                tg_sq[0] = rsqrt(var + p.eps);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    mean = tg_sum[0];
    inv_std = tg_sq[0];

    for (uint d = tid; d < p.dim; d += tg_size) {
        const float v = (float(x[row_base + d]) - mean) * inv_std;
        const float g = vision_load_f16_or_bf16(gamma, (ulong)d, VISION_EFF_W_BF16);
        float y = v * g;
        if (p.has_bias != 0u) {
            y += vision_load_f16_or_bf16(beta, (ulong)d, VISION_EFF_W_BF16);
        }
        y = vision_apply_act(y, p.act_kind);
        out[row_base + d] = half(y);
    }
}

// ===========================================================================
// M7.8 G22: Fused QKV Split + Optional QK-RMSNorm + 2D-Rotary Positional Embeddings
// Grid: (num_heads, num_tokens, 1), threadgroup: (64, 1, 1) (head_dim <= 128).
// ===========================================================================

struct VisionRope2DParams {
    uint num_tokens;
    uint num_heads;
    uint head_dim;
    uint qkv_layout;       // 0 = [S, 3, H, D], 1 = [S, H, 3, D]
    uint rope_interleaved; // 0 = per-axis half-split (Qwen-VL), 1 = per-axis interleaved (Inkling2), 2 = global half-split cat([h,w,h,w]) (Glm5Next)
    float rope_theta;
    uint swap_hw_axes;     // 0 = [H, W] (Qwen-VL/Glm5Next), 1 = [W, H] (Inkling2: cat([freq_w, freq_h]))
    float qk_norm_eps;     // RMSNorm epsilon for `ami_vision_qkv_qknorm_rope_2d` (e.g. 1e-5 for Glm5Next)
};

static_assert(sizeof(VisionRope2DParams) == 32, "VisionRope2DParams must be 32 bytes");

kernel void ami_vision_qkv_rope_2d(const device half *qkv [[buffer(0)]],
                                   const device uint2 *coords [[buffer(1)]],
                                   device half *q_out [[buffer(2)]],
                                   device half *k_out [[buffer(3)]],
                                   device half *v_out [[buffer(4)]],
                                   constant VisionRope2DParams &p [[buffer(5)]],
                                   uint2 tg_pos [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]]) {
    const uint head = tg_pos.x;
    const uint tok = tg_pos.y;
    const uint half_dim = p.head_dim >> 1;
    if (head >= p.num_heads || tok >= p.num_tokens || tid >= half_dim) {
        return;
    }

    const uint axis_dim = half_dim;      // D / 2 per spatial axis (H and W)
    const uint axis_pairs = axis_dim >> 1; // D / 4 rotary pairs per axis
    const bool is_second_axis = (tid >= axis_pairs);
    const uint pair_idx = is_second_axis ? (tid - axis_pairs) : tid;

    const uint2 hw = coords[tok];
    const bool use_w = (p.swap_hw_axes != 0u) ? (!is_second_axis) : is_second_axis;
    const float pos = float(use_w ? hw.y : hw.x);
    const float inv_freq = pow(p.rope_theta, -float(2u * pair_idx) / float(axis_dim));
    const float angle = pos * inv_freq;
    const float cos_a = cos(angle);
    const float sin_a = sin(angle);

    uint d0;
    uint d1;
    if (p.rope_interleaved == 2u) {
        d0 = tid;
        d1 = tid + half_dim;
    } else {
        const uint axis_base = is_second_axis ? axis_dim : 0u;
        d0 = (p.rope_interleaved == 0u) ? (axis_base + pair_idx) : (axis_base + 2u * pair_idx);
        d1 = (p.rope_interleaved == 0u) ? (axis_base + pair_idx + axis_pairs) : (axis_base + 2u * pair_idx + 1u);
    }

    const ulong hidden = (ulong)p.num_heads * (ulong)p.head_dim;
    const ulong qkv_tok_base = (ulong)tok * 3ul * hidden;
    ulong q_in_base;
    ulong k_in_base;
    ulong v_in_base;
    if (p.qkv_layout == 0u) {
        // [S, 3, H, D]
        q_in_base = qkv_tok_base + (ulong)head * (ulong)p.head_dim;
        k_in_base = qkv_tok_base + hidden + (ulong)head * (ulong)p.head_dim;
        v_in_base = qkv_tok_base + 2ul * hidden + (ulong)head * (ulong)p.head_dim;
    } else {
        // [S, H, 3, D]
        const ulong h_base = qkv_tok_base + (ulong)head * 3ul * (ulong)p.head_dim;
        q_in_base = h_base;
        k_in_base = h_base + (ulong)p.head_dim;
        v_in_base = h_base + 2ul * (ulong)p.head_dim;
    }

    const ulong out_base = ((ulong)tok * (ulong)p.num_heads + (ulong)head) * (ulong)p.head_dim;

    const float q0 = float(qkv[q_in_base + d0]);
    const float q1 = float(qkv[q_in_base + d1]);
    q_out[out_base + d0] = half(fma(q0, cos_a, -q1 * sin_a));
    q_out[out_base + d1] = half(fma(q0, sin_a, q1 * cos_a));

    const float k0 = float(qkv[k_in_base + d0]);
    const float k1 = float(qkv[k_in_base + d1]);
    k_out[out_base + d0] = half(fma(k0, cos_a, -k1 * sin_a));
    k_out[out_base + d1] = half(fma(k0, sin_a, k1 * cos_a));

    v_out[out_base + d0] = qkv[v_in_base + d0];
    v_out[out_base + d1] = qkv[v_in_base + d1];
}

// M9.2: Fused QKV split + per-head Q/K RMSNorm + 2D axial RoPE (`Glm5NextVision`).
// Grid: (num_heads, num_tokens, 1), threadgroup: (64, 1, 1) (2 simdgroups, head_dim <= 128).
// Specialized on `VISION_WEIGHT_BF16` (`function_constant(0)`).
kernel void ami_vision_qkv_qknorm_rope_2d(const device half *qkv [[buffer(0)]],
                                          const device uint2 *coords [[buffer(1)]],
                                          const device uchar *q_norm_w [[buffer(2)]],
                                          const device uchar *k_norm_w [[buffer(3)]],
                                          device half *q_out [[buffer(4)]],
                                          device half *k_out [[buffer(5)]],
                                          device half *v_out [[buffer(6)]],
                                          constant VisionRope2DParams &p [[buffer(7)]],
                                          uint2 tg_pos [[threadgroup_position_in_grid]],
                                          uint tid [[thread_index_in_threadgroup]],
                                          uint sg_id [[simdgroup_index_in_threadgroup]],
                                          uint lane [[thread_index_in_simdgroup]]) {
    const uint head = tg_pos.x;
    const uint tok = tg_pos.y;
    if (head >= p.num_heads || tok >= p.num_tokens) {
        return;
    }

    const uint half_dim = p.head_dim >> 1;
    const bool active = (tid < half_dim);
    const uint axis_dim = half_dim;
    const uint axis_pairs = axis_dim >> 1;
    const bool is_second_axis = (tid >= axis_pairs);
    const uint pair_idx = is_second_axis ? (tid - axis_pairs) : tid;

    uint d0 = 0u;
    uint d1 = 0u;
    if (p.rope_interleaved == 2u) {
        d0 = tid;
        d1 = tid + half_dim;
    } else {
        const uint axis_base = is_second_axis ? axis_dim : 0u;
        d0 = (p.rope_interleaved == 0u) ? (axis_base + pair_idx) : (axis_base + 2u * pair_idx);
        d1 = (p.rope_interleaved == 0u) ? (axis_base + pair_idx + axis_pairs) : (axis_base + 2u * pair_idx + 1u);
    }

    const ulong hidden = (ulong)p.num_heads * (ulong)p.head_dim;
    const ulong qkv_tok_base = (ulong)tok * 3ul * hidden;
    ulong q_in_base;
    ulong k_in_base;
    ulong v_in_base;
    if (p.qkv_layout == 0u) {
        q_in_base = qkv_tok_base + (ulong)head * (ulong)p.head_dim;
        k_in_base = qkv_tok_base + hidden + (ulong)head * (ulong)p.head_dim;
        v_in_base = qkv_tok_base + 2ul * hidden + (ulong)head * (ulong)p.head_dim;
    } else {
        const ulong h_base = qkv_tok_base + (ulong)head * 3ul * (ulong)p.head_dim;
        q_in_base = h_base;
        k_in_base = h_base + (ulong)p.head_dim;
        v_in_base = h_base + 2ul * (ulong)p.head_dim;
    }

    float q0 = 0.0f;
    float q1 = 0.0f;
    float k0 = 0.0f;
    float k1 = 0.0f;
    if (active) {
        q0 = float(qkv[q_in_base + d0]);
        q1 = float(qkv[q_in_base + d1]);
        k0 = float(qkv[k_in_base + d0]);
        k1 = float(qkv[k_in_base + d1]);
    }

    float q_sq = active ? fma(q0, q0, q1 * q1) : 0.0f;
    float k_sq = active ? fma(k0, k0, k1 * k1) : 0.0f;
    q_sq = simd_sum(q_sq);
    k_sq = simd_sum(k_sq);

    threadgroup float tg_sq[4];
    if (lane == 0u) {
        tg_sq[sg_id] = q_sq;
        tg_sq[2u + sg_id] = k_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (!active) {
        return;
    }

    const float inv_d = 1.0f / float(p.head_dim);
    const float q_inv_rms = rsqrt(max(0.0f, (tg_sq[0] + tg_sq[1]) * inv_d) + p.qk_norm_eps);
    const float k_inv_rms = rsqrt(max(0.0f, (tg_sq[2] + tg_sq[3]) * inv_d) + p.qk_norm_eps);

    q0 *= q_inv_rms * vision_load_f16_or_bf16(q_norm_w, (ulong)d0, VISION_EFF_W_BF16);
    q1 *= q_inv_rms * vision_load_f16_or_bf16(q_norm_w, (ulong)d1, VISION_EFF_W_BF16);
    k0 *= k_inv_rms * vision_load_f16_or_bf16(k_norm_w, (ulong)d0, VISION_EFF_W_BF16);
    k1 *= k_inv_rms * vision_load_f16_or_bf16(k_norm_w, (ulong)d1, VISION_EFF_W_BF16);

    const uint2 hw = coords[tok];
    const bool use_w = (p.swap_hw_axes != 0u) ? (!is_second_axis) : is_second_axis;
    const float pos = float(use_w ? hw.y : hw.x);
    const float inv_freq = pow(p.rope_theta, -float(2u * pair_idx) / float(axis_dim));
    const float angle = pos * inv_freq;
    const float cos_a = cos(angle);
    const float sin_a = sin(angle);

    const ulong out_base = ((ulong)tok * (ulong)p.num_heads + (ulong)head) * (ulong)p.head_dim;

    q_out[out_base + d0] = half(fma(q0, cos_a, -q1 * sin_a));
    q_out[out_base + d1] = half(fma(q0, sin_a, q1 * cos_a));

    k_out[out_base + d0] = half(fma(k0, cos_a, -k1 * sin_a));
    k_out[out_base + d1] = half(fma(k0, sin_a, k1 * cos_a));

    v_out[out_base + d0] = qkv[v_in_base + d0];
    v_out[out_base + d1] = qkv[v_in_base + d1];
}

// M9.7: In-place 2D axial RoPE for split Q/K projections (Inkling 2 ViT).
// Grid: (num_heads, num_tokens, 1), threadgroup: (64, 1, 1) (head_dim <= 128).
kernel void ami_vision_qk_rope_2d_inplace(device half *q [[buffer(0)]],
                                          device half *k [[buffer(1)]],
                                          const device uint2 *coords [[buffer(2)]],
                                          constant VisionRope2DParams &p [[buffer(3)]],
                                          uint2 tg_pos [[threadgroup_position_in_grid]],
                                          uint tid [[thread_index_in_threadgroup]]) {
    const uint head = tg_pos.x;
    const uint tok = tg_pos.y;
    const uint half_dim = p.head_dim >> 1;
    if (head >= p.num_heads || tok >= p.num_tokens || tid >= half_dim) {
        return;
    }

    const uint axis_dim = half_dim;
    const uint axis_pairs = axis_dim >> 1;
    const bool is_second_axis = (tid >= axis_pairs);
    const uint pair_idx = is_second_axis ? (tid - axis_pairs) : tid;

    const uint2 hw = coords[tok];
    const bool use_w = (p.swap_hw_axes != 0u) ? (!is_second_axis) : is_second_axis;
    const float pos = float(use_w ? hw.y : hw.x);
    const float inv_freq = pow(p.rope_theta, -float(2u * pair_idx) / float(axis_dim));
    const float angle = pos * inv_freq;
    const float cos_a = cos(angle);
    const float sin_a = sin(angle);

    uint d0;
    uint d1;
    if (p.rope_interleaved == 2u) {
        d0 = tid;
        d1 = tid + half_dim;
    } else {
        const uint axis_base = is_second_axis ? axis_dim : 0u;
        d0 = (p.rope_interleaved == 0u) ? (axis_base + pair_idx) : (axis_base + 2u * pair_idx);
        d1 = (p.rope_interleaved == 0u) ? (axis_base + pair_idx + axis_pairs) : (axis_base + 2u * pair_idx + 1u);
    }

    const ulong base = ((ulong)tok * (ulong)p.num_heads + (ulong)head) * (ulong)p.head_dim;

    const float q0 = float(q[base + d0]);
    const float q1 = float(q[base + d1]);
    q[base + d0] = half(fma(q0, cos_a, -q1 * sin_a));
    q[base + d1] = half(fma(q0, sin_a, q1 * cos_a));

    const float k0 = float(k[base + d0]);
    const float k1 = float(k[base + d1]);
    k[base + d0] = half(fma(k0, cos_a, -k1 * sin_a));
    k[base + d1] = half(fma(k0, sin_a, k1 * cos_a));
}

// ===========================================================================
// M7.8 G22: Bidirectional (Non-Causal) Full / Windowed FlashAttention-2
//   - Apple9 `simdgroup_half8x8` MMA (`ami_vision_bidir_flash_attn_mma`)
//   - Apple10+ `NAX` TensorOps hook (`ami_vision_bidir_flash_attn_nax`)
// Grid: (num_heads, ceil(seq_len / 32), 1), threadgroup: (128, 1, 1).
// SRAM: 15,360 B <= 16 KiB (KVs 8,704 B + Ss 4,096 B + Ps 2,560 B).
// ===========================================================================

struct VisionAttnParams {
    uint seq_len;
    uint num_heads;
    uint head_dim;
    float scale;
    uint has_window_bounds; // 0 = full bidirectional [0, seq_len), 1 = per-query uint2(kv_start, kv_end)
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

static_assert(sizeof(VisionAttnParams) == 32, "VisionAttnParams must be 32 bytes");

constant constexpr float VISION_LOG2E = 1.4426950408889634f;
constant constexpr uint VIS_FA_BQ = 32u;
constant constexpr uint VIS_FA_BK = 32u;
constant constexpr uint VIS_FA_MAX_D_PAD = 136u; // ceil(128 / 8) * 8 + 8 bank-conflict pad

kernel void ami_vision_bidir_flash_attn_mma(const device half *q [[buffer(0)]],
                                            const device half *k [[buffer(1)]],
                                            const device half *v [[buffer(2)]],
                                            const device uint2 *window_bounds [[buffer(3)]],
                                            device half *out [[buffer(4)]],
                                            constant VisionAttnParams &p [[buffer(5)]],
                                            uint2 tg_pos [[threadgroup_position_in_grid]],
                                            uint tid [[thread_index_in_threadgroup]],
                                            uint sg_id [[simdgroup_index_in_threadgroup]],
                                            uint lane [[thread_index_in_simdgroup]]) {
    const uint head = tg_pos.x;
    const uint q_block_base = tg_pos.y * VIS_FA_BQ;
    if (head >= p.num_heads || q_block_base >= p.seq_len || p.head_dim == 0u || p.head_dim > 128u) {
        return;
    }

    const uint d_pad8 = (p.head_dim + 7u) & ~7u;
    const uint d_pad = d_pad8 + 8u;
    const uint d_tiles8 = d_pad8 >> 3;
    const uint d_cols32 = (p.head_dim + 31u) >> 5;
    const ulong tok_stride = (ulong)p.num_heads * (ulong)p.head_dim;
    const ulong head_off = (ulong)head * (ulong)p.head_dim;

    threadgroup half KVs[VIS_FA_BQ * VIS_FA_MAX_D_PAD];
    threadgroup float Ss[VIS_FA_BQ * VIS_FA_BK];
    threadgroup half Ps[VIS_FA_BQ * 40u];

    // Resolve per-query bidirectional KV window bounds & threadgroup [kv_begin, kv_end).
    const uint r0 = sg_id * 8u;
    uint q_kv_start[8];
    uint q_kv_end[8];
    uint local_min = p.seq_len;
    uint local_max = 0u;
    for (uint r = 0u; r < 8u; r++) {
        const uint qi = q_block_base + r0 + r;
        if (qi < p.seq_len) {
            if (p.has_window_bounds != 0u) {
                const uint2 wb = window_bounds[qi];
                q_kv_start[r] = min(wb.x, p.seq_len);
                q_kv_end[r] = min(wb.y, p.seq_len);
            } else {
                q_kv_start[r] = 0u;
                q_kv_end[r] = p.seq_len;
            }
            if (q_kv_start[r] < q_kv_end[r]) {
                local_min = min(local_min, q_kv_start[r]);
                local_max = max(local_max, q_kv_end[r]);
            }
        } else {
            q_kv_start[r] = 0u;
            q_kv_end[r] = 0u;
        }
    }
    if (lane == 0u) {
        Ss[sg_id] = as_type<float>(local_min);
        Ss[4u + sg_id] = as_type<float>(local_max);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint kv_begin = p.seq_len;
    uint kv_end = 0u;
    for (uint s = 0u; s < 4u; s++) {
        kv_begin = min(kv_begin, as_type<uint>(Ss[s]));
        kv_end = max(kv_end, as_type<uint>(Ss[4u + s]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 1. Cooperatively stage Q [32, d_pad8] into KVs, hoist into simdgroup_half8x8 register fragments.
    const uint total_q_elems = VIS_FA_BQ * d_pad8;
    for (uint idx = tid; idx < total_q_elems; idx += 128u) {
        const uint row = idx / d_pad8;
        const uint col = idx - row * d_pad8;
        const uint qi = q_block_base + row;
        if (qi < p.seq_len && col < p.head_dim) {
            KVs[row * d_pad + col] = q[(ulong)qi * tok_stride + head_off + col];
        } else {
            KVs[row * d_pad + col] = half(0.0h);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_half8x8 q_frag[16];
    for (uint kt = 0u; kt < d_tiles8; kt++) {
        simdgroup_load(q_frag[kt], KVs + r0 * d_pad + kt * 8u, d_pad);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_val[8];
    float l_val[8];
    float o_reg[8][4];
    for (uint r = 0u; r < 8u; r++) {
        m_val[r] = -INFINITY;
        l_val[r] = 0.0f;
        for (uint c = 0u; c < 4u; c++) {
            o_reg[r][c] = 0.0f;
        }
    }

    const float scale_log2e = p.scale * VISION_LOG2E;
    const uint total_kv_elems = VIS_FA_BK * d_pad8;

    // 2. Stream BK = 32 KV blocks bidirectionally over [kv_begin, kv_end).
    for (uint kv_block_start = kv_begin; kv_block_start < kv_end; kv_block_start += VIS_FA_BK) {
        const uint kv_block_end = min(kv_block_start + VIS_FA_BK, kv_end);

        // 2a. Cooperative load of K [32, d_pad8].
        for (uint idx = tid; idx < total_kv_elems; idx += 128u) {
            const uint row = idx / d_pad8;
            const uint col = idx - row * d_pad8;
            const uint ki = kv_block_start + row;
            if (ki < kv_block_end && col < p.head_dim) {
                KVs[row * d_pad + col] = k[(ulong)ki * tok_stride + head_off + col];
            } else {
                KVs[row * d_pad + col] = half(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2b. S = Q[8, D] @ K[32, D]^T via hardware simdgroup_half8x8 MMA.
        simdgroup_float8x8 s_frag[4];
        for (uint bj = 0u; bj < 4u; bj++) {
            s_frag[bj] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
        for (uint kt = 0u; kt < d_tiles8; kt++) {
            const uint kk = kt * 8u;
            for (uint bj = 0u; bj < 4u; bj++) {
                simdgroup_half8x8 k_frag;
                simdgroup_load(k_frag, KVs + (bj * 8u) * d_pad + kk, d_pad, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(s_frag[bj], q_frag[kt], k_frag, s_frag[bj]);
            }
        }
        for (uint bj = 0u; bj < 4u; bj++) {
            simdgroup_store(s_frag[bj], Ss + r0 * VIS_FA_BK + bj * 8u, VIS_FA_BK);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // 2c. Bidirectional (non-causal) full/window mask & FP32 base-2 online softmax.
        float alpha[8];
        const uint kv_pos = kv_block_start + lane;
        const bool in_blk = (kv_pos < kv_block_end);
        for (uint r = 0u; r < 8u; r++) {
            const uint qi = q_block_base + r0 + r;
            const bool valid = (qi < p.seq_len) && in_blk &&
                               (kv_pos >= q_kv_start[r]) && (kv_pos < q_kv_end[r]);
            const float raw_s = Ss[(r0 + r) * VIS_FA_BK + lane];
            const float score = valid ? (raw_s * scale_log2e) : -INFINITY;
            const float row_m = simd_max(score);
            float p_val = 0.0f;
            if (row_m > -INFINITY) {
                const float m_new = max(m_val[r], row_m);
                const float a = (l_val[r] > 0.0f) ? exp2(m_val[r] - m_new) : 0.0f;
                p_val = valid ? exp2(score - m_new) : 0.0f;
                const float row_l = simd_sum(p_val);
                l_val[r] = fma(l_val[r], a, row_l);
                m_val[r] = m_new;
                alpha[r] = a;
            } else {
                alpha[r] = 1.0f;
            }
            Ps[(r0 + r) * 40u + lane] = half(p_val);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // 2d. Cooperative load of V [32, d_pad8].
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tid; idx < total_kv_elems; idx += 128u) {
            const uint row = idx / d_pad8;
            const uint col = idx - row * d_pad8;
            const uint vi = kv_block_start + row;
            if (vi < kv_block_end && col < p.head_dim) {
                KVs[row * d_pad + col] = v[(ulong)vi * tok_stride + head_off + col];
            } else {
                KVs[row * d_pad + col] = half(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2e. dO = P[8, 32] @ V[32, D] via simdgroup_half8x8 MMA, 32 cols/pass.
        for (uint c = 0u; c < d_cols32; c++) {
            const uint d_blk = c * 32u;
            simdgroup_float8x8 pv[4];
            for (uint bj = 0u; bj < 4u; bj++) {
                pv[bj] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            }
            for (uint kk = 0u; kk < VIS_FA_BK; kk += 8u) {
                simdgroup_half8x8 p_frag;
                simdgroup_load(p_frag, Ps + r0 * 40u + kk, 40u);
                for (uint bj = 0u; bj < 4u; bj++) {
                    const uint col_off = d_blk + bj * 8u;
                    if (col_off < d_pad8) {
                        simdgroup_half8x8 v_frag;
                        simdgroup_load(v_frag, KVs + kk * d_pad + col_off, d_pad);
                        simdgroup_multiply_accumulate(pv[bj], p_frag, v_frag, pv[bj]);
                    }
                }
            }
            for (uint bj = 0u; bj < 4u; bj++) {
                simdgroup_store(pv[bj], Ss + r0 * 32u + bj * 8u, 32u);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            for (uint r = 0u; r < 8u; r++) {
                o_reg[r][c] = fma(o_reg[r][c], alpha[r], Ss[(r0 + r) * 32u + lane]);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 3. Finalize normalized FP16 output rows.
    for (uint r = 0u; r < 8u; r++) {
        const uint qi = q_block_base + r0 + r;
        if (qi >= p.seq_len) {
            continue;
        }
        const float inv_l = (l_val[r] > 0.0f) ? (1.0f / l_val[r]) : 0.0f;
        const ulong out_base = (ulong)qi * tok_stride + head_off;
        for (uint c = 0u; c < d_cols32; c++) {
            const uint col = c * 32u + lane;
            if (col < p.head_dim) {
                out[out_base + col] = half(o_reg[r][c] * inv_l);
            }
        }
    }
}

// Apple10+ GPU Neural Accelerator (`NAX`) bidirectional FlashAttention-2 hook
// using `mpp::tensor_ops::matmul2d` (`execution_simdgroups<4>`). Instantiated
// lazily only on `gpu_apple_family >= 10` (0 overhead on Apple7-9 / M1-M4).
kernel void ami_vision_bidir_flash_attn_nax(const device half *q [[buffer(0)]],
                                            const device half *k [[buffer(1)]],
                                            const device half *v [[buffer(2)]],
                                            const device uint2 *window_bounds [[buffer(3)]],
                                            device half *out [[buffer(4)]],
                                            constant VisionAttnParams &p [[buffer(5)]],
                                            uint2 tg_pos [[threadgroup_position_in_grid]],
                                            uint tid [[thread_index_in_threadgroup]],
                                            uint sg_id [[simdgroup_index_in_threadgroup]],
                                            uint lane [[thread_index_in_simdgroup]]) {
    const uint head = tg_pos.x;
    const uint q_block_base = tg_pos.y * VIS_FA_BQ;
    if (head >= p.num_heads || q_block_base >= p.seq_len || p.head_dim == 0u || p.head_dim > 128u) {
        return;
    }

    const uint d_cols32 = (p.head_dim + 31u) >> 5;
    const ulong tok_stride = (ulong)p.num_heads * (ulong)p.head_dim;
    const ulong head_off = (ulong)head * (ulong)p.head_dim;

    threadgroup half KVs[VIS_FA_BK * 32u];
    threadgroup float Ss[VIS_FA_BQ * VIS_FA_BK];
    threadgroup half Ps[VIS_FA_BQ * VIS_FA_BK];

    const uint r0 = sg_id * 8u;
    uint q_kv_start[8];
    uint q_kv_end[8];
    uint local_min = p.seq_len;
    uint local_max = 0u;
    for (uint r = 0u; r < 8u; r++) {
        const uint qi = q_block_base + r0 + r;
        if (qi < p.seq_len) {
            if (p.has_window_bounds != 0u) {
                const uint2 wb = window_bounds[qi];
                q_kv_start[r] = min(wb.x, p.seq_len);
                q_kv_end[r] = min(wb.y, p.seq_len);
            } else {
                q_kv_start[r] = 0u;
                q_kv_end[r] = p.seq_len;
            }
            if (q_kv_start[r] < q_kv_end[r]) {
                local_min = min(local_min, q_kv_start[r]);
                local_max = max(local_max, q_kv_end[r]);
            }
        } else {
            q_kv_start[r] = 0u;
            q_kv_end[r] = 0u;
        }
    }
    if (lane == 0u) {
        Ss[sg_id] = as_type<float>(local_min);
        Ss[4u + sg_id] = as_type<float>(local_max);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint kv_begin = p.seq_len;
    uint kv_end = 0u;
    for (uint s = 0u; s < 4u; s++) {
        kv_begin = min(kv_begin, as_type<uint>(Ss[s]));
        kv_end = max(kv_end, as_type<uint>(Ss[4u + s]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_val[8];
    float l_val[8];
    float o_reg[8][4];
    for (uint r = 0u; r < 8u; r++) {
        m_val[r] = -INFINITY;
        l_val[r] = 0.0f;
        for (uint c = 0u; c < 4u; c++) {
            o_reg[r][c] = 0.0f;
        }
    }

    constexpr auto qk_desc = mpp::tensor_ops::matmul2d_descriptor(
        VIS_FA_BQ, VIS_FA_BK, 32, false, true, false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<qk_desc, metal::execution_simdgroups<4>> qk_op;

    constexpr auto pv_desc = mpp::tensor_ops::matmul2d_descriptor(
        VIS_FA_BQ, 32, VIS_FA_BK, false, false, false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<pv_desc, metal::execution_simdgroups<4>> pv_op;

    const float scale_log2e = p.scale * VISION_LOG2E;

    using QkTileA = tensor<threadgroup half, extents<int, 32, int(VIS_FA_BQ)>, tensor_inline>;
    using QkTileB = tensor<threadgroup half, extents<int, 32, int(VIS_FA_BK)>, tensor_inline>;
    using QkTileC = tensor<threadgroup float, extents<int, int(VIS_FA_BK), int(VIS_FA_BQ)>, tensor_inline>;
    using PvTileA = tensor<threadgroup half, extents<int, int(VIS_FA_BK), int(VIS_FA_BQ)>, tensor_inline>;
    using PvTileB = tensor<threadgroup half, extents<int, 32, int(VIS_FA_BK)>, tensor_inline>;
    using PvTileC = tensor<threadgroup float, extents<int, 32, int(VIS_FA_BQ)>, tensor_inline>;

    for (uint kv_block_start = kv_begin; kv_block_start < kv_end; kv_block_start += VIS_FA_BK) {
        const uint kv_block_end = min(kv_block_start + VIS_FA_BK, kv_end);

        auto s_acc = qk_op.get_destination_cooperative_tensor<QkTileA, QkTileB, float>();
        for (uint16_t i = 0u; i < s_acc.get_capacity(); i++) {
            if (s_acc.is_valid_element(i)) {
                s_acc[i] = 0.0f;
            }
        }
        for (uint c = 0u; c < d_cols32; c++) {
            const uint d_blk = c * 32u;
            for (uint idx = tid; idx < VIS_FA_BQ * 32u; idx += 128u) {
                const uint r = idx >> 5;
                const uint j = idx & 31u;
                const uint qi = q_block_base + r;
                const uint ki = kv_block_start + r;
                const uint col = d_blk + j;
                Ps[idx] = (qi < p.seq_len && col < p.head_dim)
                    ? q[(ulong)qi * tok_stride + head_off + col]
                    : half(0.0h);
                KVs[idx] = (ki < kv_block_end && col < p.head_dim)
                    ? k[(ulong)ki * tok_stride + head_off + col]
                    : half(0.0h);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            QkTileA q_t(Ps, extents<int, 32, int(VIS_FA_BQ)>());
            QkTileB k_t(KVs, extents<int, 32, int(VIS_FA_BK)>());
            qk_op.run(q_t, k_t, s_acc);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        QkTileC s_t(Ss, extents<int, int(VIS_FA_BK), int(VIS_FA_BQ)>());
        s_acc.store(s_t);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float alpha[8];
        const uint kv_pos = kv_block_start + lane;
        const bool in_blk = (kv_pos < kv_block_end);
        for (uint r = 0u; r < 8u; r++) {
            const uint qi = q_block_base + r0 + r;
            const bool valid = (qi < p.seq_len) && in_blk &&
                               (kv_pos >= q_kv_start[r]) && (kv_pos < q_kv_end[r]);
            const float raw_s = Ss[(r0 + r) * VIS_FA_BK + lane];
            const float score = valid ? (raw_s * scale_log2e) : -INFINITY;
            const float row_m = simd_max(score);
            float p_val = 0.0f;
            if (row_m > -INFINITY) {
                const float m_new = max(m_val[r], row_m);
                const float a = (l_val[r] > 0.0f) ? exp2(m_val[r] - m_new) : 0.0f;
                p_val = valid ? exp2(score - m_new) : 0.0f;
                const float row_l = simd_sum(p_val);
                l_val[r] = fma(l_val[r], a, row_l);
                m_val[r] = m_new;
                alpha[r] = a;
            } else {
                alpha[r] = 1.0f;
            }
            Ps[(r0 + r) * VIS_FA_BK + lane] = half(p_val);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint c = 0u; c < d_cols32; c++) {
            const uint d_blk = c * 32u;
            for (uint idx = tid; idx < VIS_FA_BK * 32u; idx += 128u) {
                const uint r = idx >> 5;
                const uint j = idx & 31u;
                const uint vi = kv_block_start + r;
                const uint col = d_blk + j;
                KVs[idx] = (vi < kv_block_end && col < p.head_dim)
                    ? v[(ulong)vi * tok_stride + head_off + col]
                    : half(0.0h);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            PvTileA p_t(Ps, extents<int, int(VIS_FA_BK), int(VIS_FA_BQ)>());
            PvTileB v_t(KVs, extents<int, 32, int(VIS_FA_BK)>());
            auto pv_acc = pv_op.get_destination_cooperative_tensor<PvTileA, PvTileB, float>();
            for (uint16_t i = 0u; i < pv_acc.get_capacity(); i++) {
                if (pv_acc.is_valid_element(i)) {
                    pv_acc[i] = 0.0f;
                }
            }
            pv_op.run(p_t, v_t, pv_acc);
            PvTileC o_t(Ss, extents<int, 32, int(VIS_FA_BQ)>());
            pv_acc.store(o_t);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint r = 0u; r < 8u; r++) {
                o_reg[r][c] = fma(o_reg[r][c], alpha[r], Ss[(r0 + r) * 32u + lane]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    for (uint r = 0u; r < 8u; r++) {
        const uint qi = q_block_base + r0 + r;
        if (qi >= p.seq_len) {
            continue;
        }
        const float inv_l = (l_val[r] > 0.0f) ? (1.0f / l_val[r]) : 0.0f;
        const ulong out_base = (ulong)qi * tok_stride + head_off;
        for (uint c = 0u; c < d_cols32; c++) {
            const uint col = c * 32u + lane;
            if (col < p.head_dim) {
                out[out_base + col] = half(o_reg[r][c] * inv_l);
            }
        }
    }
}

// ===========================================================================
// M7.8 G22: Tiled MMA (`f16`/`bf16` & `q4_affine`) and NAX Vision Linear
// Computes `Y[M, N] = act(X[M, K] @ W[N, K]^T + bias) + residual`.
// Grid: (ceil(N / 32), ceil(M / 32), 1), threadgroup: (128, 1, 1) (MMA)
// Grid: (ceil(N / 64), ceil(M / 64), 1), threadgroup: (128, 1, 1) (NAX)
// ===========================================================================

struct VisionLinearParams {
    uint m;
    uint n;
    uint k;
    uint num_groups;
    uint row_bytes;
    uint has_bias;
    uint has_residual;
    uint act_kind; // 0 = none, 1 = gelu_tanh (Qwen3.8), 2 = gelu_exact (erf)
};

static_assert(sizeof(VisionLinearParams) == 32, "VisionLinearParams must be 32 bytes");

kernel void ami_vision_linear_mma(const device half *x [[buffer(0)]],
                                  const device uchar *weight [[buffer(1)]],
                                  const device uchar *bias [[buffer(2)]],
                                  const device half *residual [[buffer(3)]],
                                  device half *out [[buffer(4)]],
                                  constant VisionLinearParams &p [[buffer(5)]],
                                  uint2 tg_pos [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]],
                                  uint sg_id [[simdgroup_index_in_threadgroup]]) {
    constexpr uint BM = 32u;
    constexpr uint BN = 32u;
    constexpr uint BK = 32u;
    constexpr uint STRIDE = 40u;

    const uint n_base = tg_pos.x * BN;
    const uint m_base = tg_pos.y * BM;
    if (m_base >= p.m || n_base >= p.n) {
        return;
    }

    threadgroup half As[BM * STRIDE];
    threadgroup half Bs[BN * STRIDE];
    threadgroup float Cs[BM * BN];

    const uint sg_m = (sg_id >> 1) * 16u;
    const uint sg_n = (sg_id & 1u) * 16u;

    simdgroup_float8x8 acc[2][2];
    for (uint mi = 0u; mi < 2u; mi++) {
        for (uint ni = 0u; ni < 2u; ni++) {
            acc[mi][ni] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    for (uint k_base = 0u; k_base < p.k; k_base += BK) {
        for (uint idx = tid; idx < BM * BK; idx += 128u) {
            const uint r = idx >> 5;
            const uint c = idx & 31u;
            const uint gm = m_base + r;
            const uint gk = k_base + c;
            As[r * STRIDE + c] = (gm < p.m && gk < p.k) ? x[(ulong)gm * p.k + gk] : half(0.0h);

            const uint gn = n_base + r;
            if (gn < p.n && gk < p.k) {
                Bs[r * STRIDE + c] = half(vision_load_f16_or_bf16(weight, (ulong)gn * p.k + gk, VISION_EFF_W_BF16));
            } else {
                Bs[r * STRIDE + c] = half(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0u; kk < BK; kk += 8u) {
            simdgroup_half8x8 a_frag[2];
            simdgroup_half8x8 b_frag[2];
            for (uint mi = 0u; mi < 2u; mi++) {
                simdgroup_load(a_frag[mi], As + (sg_m + mi * 8u) * STRIDE + kk, STRIDE);
            }
            for (uint ni = 0u; ni < 2u; ni++) {
                simdgroup_load(b_frag[ni], Bs + (sg_n + ni * 8u) * STRIDE + kk, STRIDE, ulong2(0, 0), true);
            }
            for (uint mi = 0u; mi < 2u; mi++) {
                for (uint ni = 0u; ni < 2u; ni++) {
                    simdgroup_multiply_accumulate(acc[mi][ni], a_frag[mi], b_frag[ni], acc[mi][ni]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint mi = 0u; mi < 2u; mi++) {
        for (uint ni = 0u; ni < 2u; ni++) {
            simdgroup_store(acc[mi][ni], Cs + (sg_m + mi * 8u) * BN + (sg_n + ni * 8u), BN);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tid; idx < BM * BN; idx += 128u) {
        const uint r = idx >> 5;
        const uint c = idx & 31u;
        const uint gm = m_base + r;
        const uint gn = n_base + c;
        if (gm < p.m && gn < p.n) {
            float v = Cs[idx];
            if (p.has_bias != 0u) {
                v += vision_load_f16_or_bf16(bias, (ulong)gn, VISION_EFF_W_BF16);
            }
            v = vision_apply_act(v, p.act_kind);
            const ulong out_idx = (ulong)gm * p.n + gn;
            if (p.has_residual != 0u) {
                v += float(residual[out_idx]);
            }
            out[out_idx] = half(v);
        }
    }
}

// Q4_AFFINE (group_size = 64) dequantizing tiled MMA vision linear kernel.
kernel void ami_vision_linear_q4_mma(const device half *x [[buffer(0)]],
                                     const device uchar *qweight [[buffer(1)]],
                                     const device half2 *scale_bias [[buffer(2)]],
                                     const device uchar *bias [[buffer(3)]],
                                     const device half *residual [[buffer(4)]],
                                     device half *out [[buffer(5)]],
                                     constant VisionLinearParams &p [[buffer(6)]],
                                     uint2 tg_pos [[threadgroup_position_in_grid]],
                                     uint tid [[thread_index_in_threadgroup]],
                                     uint sg_id [[simdgroup_index_in_threadgroup]]) {
    constexpr uint BM = 32u;
    constexpr uint BN = 32u;
    constexpr uint BK = 32u;
    constexpr uint STRIDE = 40u;

    const uint n_base = tg_pos.x * BN;
    const uint m_base = tg_pos.y * BM;
    if (m_base >= p.m || n_base >= p.n) {
        return;
    }

    threadgroup half As[BM * STRIDE];
    threadgroup half Bs[BN * STRIDE];
    threadgroup float Cs[BM * BN];

    const uint sg_m = (sg_id >> 1) * 16u;
    const uint sg_n = (sg_id & 1u) * 16u;

    simdgroup_float8x8 acc[2][2];
    for (uint mi = 0u; mi < 2u; mi++) {
        for (uint ni = 0u; ni < 2u; ni++) {
            acc[mi][ni] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    for (uint k_base = 0u; k_base < p.k; k_base += BK) {
        const uint grp = k_base >> 6; // group_size = 64
        for (uint idx = tid; idx < BM * BK; idx += 128u) {
            const uint r = idx >> 5;
            const uint c = idx & 31u;
            const uint gm = m_base + r;
            const uint gk = k_base + c;
            As[r * STRIDE + c] = (gm < p.m && gk < p.k) ? x[(ulong)gm * p.k + gk] : half(0.0h);

            const uint gn = n_base + r;
            if (gn < p.n && gk < p.k) {
                const uchar packed = qweight[(ulong)gn * p.row_bytes + (gk >> 1)];
                const uint nibble = ((gk & 1u) == 0u) ? uint(packed & 0x0Fu) : uint(packed >> 4);
                const half2 sb = scale_bias[(ulong)gn * p.num_groups + grp];
                Bs[r * STRIDE + c] = half(fma(float(nibble), float(sb.x), float(sb.y)));
            } else {
                Bs[r * STRIDE + c] = half(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0u; kk < BK; kk += 8u) {
            simdgroup_half8x8 a_frag[2];
            simdgroup_half8x8 b_frag[2];
            for (uint mi = 0u; mi < 2u; mi++) {
                simdgroup_load(a_frag[mi], As + (sg_m + mi * 8u) * STRIDE + kk, STRIDE);
            }
            for (uint ni = 0u; ni < 2u; ni++) {
                simdgroup_load(b_frag[ni], Bs + (sg_n + ni * 8u) * STRIDE + kk, STRIDE, ulong2(0, 0), true);
            }
            for (uint mi = 0u; mi < 2u; mi++) {
                for (uint ni = 0u; ni < 2u; ni++) {
                    simdgroup_multiply_accumulate(acc[mi][ni], a_frag[mi], b_frag[ni], acc[mi][ni]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint mi = 0u; mi < 2u; mi++) {
        for (uint ni = 0u; ni < 2u; ni++) {
            simdgroup_store(acc[mi][ni], Cs + (sg_m + mi * 8u) * BN + (sg_n + ni * 8u), BN);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tid; idx < BM * BN; idx += 128u) {
        const uint r = idx >> 5;
        const uint c = idx & 31u;
        const uint gm = m_base + r;
        const uint gn = n_base + c;
        if (gm < p.m && gn < p.n) {
            float v = Cs[idx];
            if (p.has_bias != 0u) {
                v += vision_load_f16_or_bf16(bias, (ulong)gn, VISION_EFF_W_BF16);
            }
            v = vision_apply_act(v, p.act_kind);
            const ulong out_idx = (ulong)gm * p.n + gn;
            if (p.has_residual != 0u) {
                v += float(residual[out_idx]);
            }
            out[out_idx] = half(v);
        }
    }
}

// Apple10+ GPU Neural Accelerator (`NAX`) vision linear projection hook (`BM=64, BN=64, BK=32`).
kernel void ami_vision_linear_nax(const device half *x [[buffer(0)]],
                                  const device uchar *weight [[buffer(1)]],
                                  const device uchar *bias [[buffer(2)]],
                                  const device half *residual [[buffer(3)]],
                                  device half *out [[buffer(4)]],
                                  constant VisionLinearParams &p [[buffer(5)]],
                                  uint2 tg_pos [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint BM = 64u;
    constexpr uint BN = 64u;
    constexpr uint BK = 32u;

    const uint n_base = tg_pos.x * BN;
    const uint m_base = tg_pos.y * BM;
    if (m_base >= p.m || n_base >= p.n) {
        return;
    }

    union VisNaxSmem {
        struct {
            half As[BM * BK];
            half Bs[BN * BK];
        } ab;
        float Cs[BM * BN];
    };
    threadgroup VisNaxSmem smem;

    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        BM, BN, BK, false, true, false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroups<4>> op;

    using TileA = tensor<threadgroup half, extents<int, int(BK), int(BM)>, tensor_inline>;
    using TileB = tensor<threadgroup half, extents<int, int(BK), int(BN)>, tensor_inline>;
    using TileC = tensor<threadgroup float, extents<int, int(BN), int(BM)>, tensor_inline>;
    TileA a_t(smem.ab.As, extents<int, int(BK), int(BM)>());
    TileB b_t(smem.ab.Bs, extents<int, int(BK), int(BN)>());
    auto acc = op.get_destination_cooperative_tensor<TileA, TileB, float>();
    for (uint16_t i = 0u; i < acc.get_capacity(); i++) {
        if (acc.is_valid_element(i)) {
            acc[i] = 0.0f;
        }
    }

    for (uint k_base = 0u; k_base < p.k; k_base += BK) {
        for (uint idx = tid; idx < BM * BK; idx += 128u) {
            const uint r = idx >> 5;
            const uint c = idx & 31u;
            const uint gm = m_base + r;
            const uint gk = k_base + c;
            smem.ab.As[idx] = (gm < p.m && gk < p.k) ? x[(ulong)gm * p.k + gk] : half(0.0h);

            const uint gn = n_base + r;
            if (gn < p.n && gk < p.k) {
                smem.ab.Bs[idx] = half(vision_load_f16_or_bf16(weight, (ulong)gn * p.k + gk, VISION_EFF_W_BF16));
            } else {
                smem.ab.Bs[idx] = half(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(a_t, b_t, acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    TileC c_t(smem.Cs, extents<int, int(BN), int(BM)>());
    acc.store(c_t);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tid; idx < BM * BN; idx += 128u) {
        const uint r = idx >> 6;
        const uint c = idx & 63u;
        const uint gm = m_base + r;
        const uint gn = n_base + c;
        if (gm < p.m && gn < p.n) {
            float v = smem.Cs[idx];
            if (p.has_bias != 0u) {
                v += vision_load_f16_or_bf16(bias, (ulong)gn, VISION_EFF_W_BF16);
            }
            v = vision_apply_act(v, p.act_kind);
            const ulong out_idx = (ulong)gm * p.n + gn;
            if (p.has_residual != 0u) {
                v += float(residual[out_idx]);
            }
            out[out_idx] = half(v);
        }
    }
}

// ===========================================================================
// M7.8 G22: SwiGLU Activation & `spatial_merge x spatial_merge` Patch Concat
// ===========================================================================

struct VisionSwiGluParams {
    uint count;
    float clamp_limit; // <= 0.0f disables clamping; > 0.0f clamps gate <= limit and up in [-limit, limit] (Glm5Next: 10.0)
    uint _pad1;
    uint _pad2;
};

static_assert(sizeof(VisionSwiGluParams) == 16, "VisionSwiGluParams must be 16 bytes");

kernel void ami_vision_swiglu(const device half *gate [[buffer(0)]],
                              const device half *up [[buffer(1)]],
                              device half *out [[buffer(2)]],
                              constant VisionSwiGluParams &p [[buffer(3)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= p.count) {
        return;
    }
    float g = float(gate[gid]);
    float u = float(up[gid]);
    if (p.clamp_limit > 0.0f) {
        g = min(g, p.clamp_limit);
        u = clamp(u, -p.clamp_limit, p.clamp_limit);
    }
    const float sig = (g >= 0.0f)
        ? (1.0f / (1.0f + exp(-min(g, 40.0f))))
        : (exp(max(g, -40.0f)) / (1.0f + exp(max(g, -40.0f))));
    out[gid] = half(g * sig * u);
}

struct VisionMergeParams {
    uint grid_h;
    uint grid_w;
    uint spatial_merge;
    uint vis_hidden;
    uint tile_count;
    uint has_cls_token;     // 0 = no CLS prefix per tile, 1 = skip leading [CLS] token per tile
    uint preordered_blocks; // 0 = raster (ph, pw) input tokens, 1 = spatial-merge-block preordered tokens
    uint channel_first;     // 0 = [sm, sm, vis_hidden] (Qwen/DeepSeek/Inkling2), 1 = [vis_hidden, sm, sm] (Glm5Next Conv2d)
};

static_assert(sizeof(VisionMergeParams) == 32, "VisionMergeParams must be 32 bytes");

kernel void ami_vision_spatial_merge_concat(const device half *in_tokens [[buffer(0)]],
                                            device half *out_merged [[buffer(1)]],
                                            constant VisionMergeParams &p [[buffer(2)]],
                                            uint out_tok [[threadgroup_position_in_grid]],
                                            uint tid [[thread_index_in_threadgroup]],
                                            uint tg_size [[threads_per_threadgroup]]) {
    const uint sm = p.spatial_merge;
    const uint merged_h = p.grid_h / sm;
    const uint merged_w = p.grid_w / sm;
    const uint merged_per_tile = merged_h * merged_w;
    const uint total_merged = p.tile_count * merged_per_tile;
    if (out_tok >= total_merged) {
        return;
    }

    const uint tile_idx = out_tok / merged_per_tile;
    const uint local_tok = out_tok - tile_idx * merged_per_tile;
    const uint mh = local_tok / merged_w;
    const uint mw = local_tok - mh * merged_w;

    const uint in_per_tile = p.grid_h * p.grid_w + p.has_cls_token;
    const uint merge_sq = sm * sm;
    const uint out_dim = merge_sq * p.vis_hidden;
    const ulong out_base = (ulong)out_tok * (ulong)out_dim;

    for (uint idx = tid; idx < out_dim; idx += tg_size) {
        uint sub;
        uint d;
        if (p.channel_first != 0u) {
            d = idx / merge_sq;
            sub = idx - d * merge_sq;
        } else {
            sub = idx / p.vis_hidden;
            d = idx - sub * p.vis_hidden;
        }
        ulong in_tok;
        if (p.preordered_blocks != 0u) {
            in_tok = (ulong)tile_idx * (ulong)in_per_tile + (ulong)p.has_cls_token +
                     (ulong)local_tok * (ulong)merge_sq + (ulong)sub;
        } else {
            const uint sy = sub / sm;
            const uint sx = sub - sy * sm;
            const uint ph = mh * sm + sy;
            const uint pw = mw * sm + sx;
            in_tok = (ulong)tile_idx * (ulong)in_per_tile + (ulong)p.has_cls_token +
                     (ulong)(ph * p.grid_w + pw);
        }
        out_merged[out_base + idx] = in_tokens[in_tok * (ulong)p.vis_hidden + d];
    }
}

// ===========================================================================
// M9.7: Inkling 2 (`inkling2_vision`) LayerScale Residual Add
//   out[tok, d] = half(float(residual[tok, d]) + float(x[tok, d]) * gamma[d])
// Safe when `out == residual` (in-place update).
// ===========================================================================

struct VisionLayerScaleParams {
    uint num_tokens;
    uint dim;
    uint _pad0;
    uint _pad1;
};

static_assert(sizeof(VisionLayerScaleParams) == 16, "VisionLayerScaleParams must be 16 bytes");

kernel void ami_vision_layerscale_residual_add(const device half *x [[buffer(0)]],
                                               const device half *residual [[buffer(1)]],
                                               const device uchar *gamma [[buffer(2)]],
                                               device half *out [[buffer(3)]],
                                               constant VisionLayerScaleParams &p [[buffer(4)]],
                                               uint gid [[thread_position_in_grid]]) {
    const uint total = p.num_tokens * p.dim;
    if (gid >= total) {
        return;
    }
    const uint d = gid % p.dim;
    const float g = vision_load_f16_or_bf16(gamma, (ulong)d, VISION_EFF_W_BF16);
    const float y = fma(float(x[gid]), g, float(residual[gid]));
    out[gid] = half(y);
}

// ===========================================================================
// M9.7: Inkling 2 (`inkling2_vision`) HWC Patchify + [CLS] + 2D Bilinear
//       Position Embedding Interpolation (`align_corners=False`, border pad)
//
// Upstream reference:
//   - `Inkling2ImageProcessor._preprocess`: right/bottom pad to multiples of
//     `patch_size * spatial_merge` (`40`) with fill `-1.0` on `0..255` scale,
//     rescale `/255`, normalize `(x - mean) * inv_std`, HWC per patch.
//   - `Inkling2VisionEmbeddings.forward`: spatial-merge-block patch order
//     `(block_row, block_col, in_row, in_col)`, HWC flatten `[ph, pw, 3]`,
//     bilinear 2D interpolation of `[posemb_side, posemb_side, hidden]`,
//     leading `[CLS] = class_embedding + cls_position_embedding` at `tok == 0`.
// ===========================================================================

struct VisionInkling2PatchParams {
    uint src_width;
    uint src_height;
    uint pad_width;
    uint pad_height;
    uint patch_size;
    uint spatial_merge;
    uint hidden_size;
    uint posemb_side;
    uint token_offset;
    uint has_cls_token;
    uint has_bias;
    uint is_full_range;
    uint use_bilinear_resize;
    float pad_fill_norm;
    float mean_r;
    float mean_g;
    float mean_b;
    float inv_std_r;
    float inv_std_g;
    float inv_std_b;
};

static_assert(sizeof(VisionInkling2PatchParams) == 80, "VisionInkling2PatchParams must be 80 bytes");

static inline float3 vision_inkling2_norm_rgb(float3 rgb, constant VisionInkling2PatchParams &p) {
    return float3(
        (rgb.x - p.mean_r) * p.inv_std_r,
        (rgb.y - p.mean_g) * p.inv_std_g,
        (rgb.z - p.mean_b) * p.inv_std_b
    );
}

static inline float3 vision_inkling2_sample_bgra8(texture2d<float, access::read> tex,
                                                  uint dst_x,
                                                  uint dst_y,
                                                  constant VisionInkling2PatchParams &p) {
    if (p.use_bilinear_resize == 0u) {
        if (dst_x < p.src_width && dst_y < p.src_height) {
            return vision_inkling2_norm_rgb(tex.read(uint2(dst_x, dst_y)).rgb, p);
        }
        return float3(p.pad_fill_norm);
    }
    float fx = (float(dst_x) + 0.5f) * (float(p.src_width) / float(p.pad_width)) - 0.5f;
    float fy = (float(dst_y) + 0.5f) * (float(p.src_height) / float(p.pad_height)) - 0.5f;
    fx = clamp(fx, 0.0f, float(p.src_width - 1u));
    fy = clamp(fy, 0.0f, float(p.src_height - 1u));
    const uint x0 = uint(floor(fx));
    const uint y0 = uint(floor(fy));
    const uint x1 = min(x0 + 1u, p.src_width - 1u);
    const uint y1 = min(y0 + 1u, p.src_height - 1u);
    const float wx = fx - float(x0);
    const float wy = fy - float(y0);
    const float3 c00 = tex.read(uint2(x0, y0)).rgb;
    const float3 c10 = tex.read(uint2(x1, y0)).rgb;
    const float3 c01 = tex.read(uint2(x0, y1)).rgb;
    const float3 c11 = tex.read(uint2(x1, y1)).rgb;
    return vision_inkling2_norm_rgb(mix(mix(c00, c10, wx), mix(c01, c11, wx), wy), p);
}

static inline float3 vision_inkling2_sample_nv12(texture2d<float, access::read> y_tex,
                                                 texture2d<float, access::read> uv_tex,
                                                 uint dst_x,
                                                 uint dst_y,
                                                 constant VisionInkling2PatchParams &p) {
    if (p.use_bilinear_resize == 0u) {
        if (dst_x < p.src_width && dst_y < p.src_height) {
            const float3 rgb = vision_bt709_nv12_to_rgb(
                y_tex.read(uint2(dst_x, dst_y)).r,
                uv_tex.read(uint2(dst_x >> 1, dst_y >> 1)).rg,
                p.is_full_range
            );
            return vision_inkling2_norm_rgb(rgb, p);
        }
        return float3(p.pad_fill_norm);
    }
    float fx = (float(dst_x) + 0.5f) * (float(p.src_width) / float(p.pad_width)) - 0.5f;
    float fy = (float(dst_y) + 0.5f) * (float(p.src_height) / float(p.pad_height)) - 0.5f;
    fx = clamp(fx, 0.0f, float(p.src_width - 1u));
    fy = clamp(fy, 0.0f, float(p.src_height - 1u));
    const uint x0 = uint(floor(fx));
    const uint y0 = uint(floor(fy));
    const uint x1 = min(x0 + 1u, p.src_width - 1u);
    const uint y1 = min(y0 + 1u, p.src_height - 1u);
    const float wx = fx - float(x0);
    const float wy = fy - float(y0);
    const float3 c00 = vision_bt709_nv12_to_rgb(y_tex.read(uint2(x0, y0)).r, uv_tex.read(uint2(x0 >> 1, y0 >> 1)).rg, p.is_full_range);
    const float3 c10 = vision_bt709_nv12_to_rgb(y_tex.read(uint2(x1, y0)).r, uv_tex.read(uint2(x1 >> 1, y0 >> 1)).rg, p.is_full_range);
    const float3 c01 = vision_bt709_nv12_to_rgb(y_tex.read(uint2(x0, y1)).r, uv_tex.read(uint2(x0 >> 1, y1 >> 1)).rg, p.is_full_range);
    const float3 c11 = vision_bt709_nv12_to_rgb(y_tex.read(uint2(x1, y1)).r, uv_tex.read(uint2(x1 >> 1, y1 >> 1)).rg, p.is_full_range);
    return vision_inkling2_norm_rgb(mix(mix(c00, c10, wx), mix(c01, c11, wx), wy), p);
}

static inline float vision_inkling2_interp_posemb(const device uchar *pos_embed,
                                                  uint patch_row,
                                                  uint patch_col,
                                                  uint grid_h,
                                                  uint grid_w,
                                                  uint h,
                                                  constant VisionInkling2PatchParams &p) {
    const float side_f = float(p.posemb_side);
    const float src_h = (float(patch_row) + 0.5f) * side_f / float(grid_h) - 0.5f;
    const float src_w = (float(patch_col) + 0.5f) * side_f / float(grid_w) - 0.5f;
    const float fh = floor(src_h);
    const float fw = floor(src_w);
    const int side_m1 = int(p.posemb_side) - 1;
    const uint h0 = uint(clamp(int(fh), 0, side_m1));
    const uint h1 = uint(clamp(int(fh) + 1, 0, side_m1));
    const uint w0 = uint(clamp(int(fw), 0, side_m1));
    const uint w1 = uint(clamp(int(fw) + 1, 0, side_m1));
    const float wh1 = src_h - fh;
    const float wh0 = 1.0f - wh1;
    const float ww1 = src_w - fw;
    const float ww0 = 1.0f - ww1;
    const ulong hs = (ulong)p.hidden_size;
    const float pe00 = vision_load_f16_or_bf16(pos_embed, (ulong)(h0 * p.posemb_side + w0) * hs + (ulong)h, VISION_EFF_W_BF16);
    const float pe01 = vision_load_f16_or_bf16(pos_embed, (ulong)(h0 * p.posemb_side + w1) * hs + (ulong)h, VISION_EFF_W_BF16);
    const float pe10 = vision_load_f16_or_bf16(pos_embed, (ulong)(h1 * p.posemb_side + w0) * hs + (ulong)h, VISION_EFF_W_BF16);
    const float pe11 = vision_load_f16_or_bf16(pos_embed, (ulong)(h1 * p.posemb_side + w1) * hs + (ulong)h, VISION_EFF_W_BF16);
    return (wh0 * ww0) * pe00 + (wh0 * ww1) * pe01 + (wh1 * ww0) * pe10 + (wh1 * ww1) * pe11;
}

kernel void ami_vision_inkling2_patchify_embed_bgra8(const device uchar *weight [[buffer(0)]],
                                                     const device uchar *bias [[buffer(1)]],
                                                     const device uchar *pos_embed [[buffer(2)]],
                                                     const device uchar *class_embed [[buffer(3)]],
                                                     const device uchar *cls_pos_embed [[buffer(4)]],
                                                     device uchar *out [[buffer(5)]],
                                                     constant VisionInkling2PatchParams &p [[buffer(6)]],
                                                     texture2d<float, access::read> frame0 [[texture(0)]],
                                                     uint2 tg_pos [[threadgroup_position_in_grid]],
                                                     uint tid [[thread_index_in_threadgroup]],
                                                     uint sg_id [[simdgroup_index_in_threadgroup]],
                                                     uint lane [[thread_index_in_simdgroup]]) {
    const uint grid_h = p.pad_height / p.patch_size;
    const uint grid_w = p.pad_width / p.patch_size;
    const uint num_patches = grid_h * grid_w;
    const uint total_seq = num_patches + p.has_cls_token;
    const uint tok = tg_pos.y;
    if (tok >= total_seq) {
        return;
    }

    const uint h = tg_pos.x * 8u + sg_id;
    const bool valid_h = (h < p.hidden_size);

    if (p.has_cls_token != 0u && tok == 0u) {
        if (valid_h && lane == 0u) {
            const float cls_v = vision_load_f16_or_bf16(class_embed, (ulong)h, VISION_EFF_W_BF16) +
                                vision_load_f16_or_bf16(cls_pos_embed, (ulong)h, VISION_EFF_W_BF16);
            const ulong out_idx = (ulong)p.token_offset * (ulong)p.hidden_size + (ulong)h;
            vision_store_f16_or_bf16(out, out_idx, cls_v, VISION_EFF_O_BF16);
        }
        return;
    }

    const uint p_idx = tok - p.has_cls_token;
    const uint sm = p.spatial_merge;
    const uint blocks_w = grid_w / sm;
    const uint sm_sq = sm * sm;
    const uint block_idx = p_idx / sm_sq;
    const uint sub = p_idx - block_idx * sm_sq;
    const uint block_row = block_idx / blocks_w;
    const uint block_col = block_idx - block_row * blocks_w;
    const uint in_row = sub / sm;
    const uint in_col = sub - in_row * sm;
    const uint patch_row = block_row * sm + in_row;
    const uint patch_col = block_col * sm + in_col;

    const uint patch_area = p.patch_size * p.patch_size;
    const uint token_in_dim = patch_area * 3u;
    const ulong w_row = (ulong)h * (ulong)token_in_dim;

    threadgroup float tg_rgb[256u * 3u];
    float acc = 0.0f;

    for (uint tile_start = 0u; tile_start < patch_area; tile_start += 256u) {
        const uint tile_count = min(256u, patch_area - tile_start);
        if (tid < tile_count) {
            const uint pix = tile_start + tid;
            const uint dy = pix / p.patch_size;
            const uint dx = pix - dy * p.patch_size;
            const uint dst_y = patch_row * p.patch_size + dy;
            const uint dst_x = patch_col * p.patch_size + dx;
            const float3 norm = vision_inkling2_sample_bgra8(frame0, dst_x, dst_y, p);
            tg_rgb[tid * 3u + 0u] = norm.x;
            tg_rgb[tid * 3u + 1u] = norm.y;
            tg_rgb[tid * 3u + 2u] = norm.z;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (valid_h) {
            for (uint i = lane; i < tile_count; i += 32u) {
                const ulong kb = w_row + (ulong)(tile_start + i) * 3ul;
                const float wr = vision_load_f16_or_bf16(weight, kb + 0ul, VISION_EFF_W_BF16);
                const float wg = vision_load_f16_or_bf16(weight, kb + 1ul, VISION_EFF_W_BF16);
                const float wb = vision_load_f16_or_bf16(weight, kb + 2ul, VISION_EFF_W_BF16);
                acc = fma(tg_rgb[i * 3u + 0u], wr, acc);
                acc = fma(tg_rgb[i * 3u + 1u], wg, acc);
                acc = fma(tg_rgb[i * 3u + 2u], wb, acc);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (valid_h) {
        acc = simd_sum(acc);
        if (lane == 0u) {
            if (p.has_bias != 0u) {
                acc += vision_load_f16_or_bf16(bias, (ulong)h, VISION_EFF_W_BF16);
            }
            if (p.posemb_side > 0u) {
                acc += vision_inkling2_interp_posemb(pos_embed, patch_row, patch_col, grid_h, grid_w, h, p);
            }
            const ulong out_idx = (ulong)(p.token_offset + tok) * (ulong)p.hidden_size + (ulong)h;
            vision_store_f16_or_bf16(out, out_idx, acc, VISION_EFF_O_BF16);
        }
    }
}

kernel void ami_vision_inkling2_patchify_embed_nv12(const device uchar *weight [[buffer(0)]],
                                                    const device uchar *bias [[buffer(1)]],
                                                    const device uchar *pos_embed [[buffer(2)]],
                                                    const device uchar *class_embed [[buffer(3)]],
                                                    const device uchar *cls_pos_embed [[buffer(4)]],
                                                    device uchar *out [[buffer(5)]],
                                                    constant VisionInkling2PatchParams &p [[buffer(6)]],
                                                    texture2d<float, access::read> y0 [[texture(0)]],
                                                    texture2d<float, access::read> cbcr0 [[texture(1)]],
                                                    uint2 tg_pos [[threadgroup_position_in_grid]],
                                                    uint tid [[thread_index_in_threadgroup]],
                                                    uint sg_id [[simdgroup_index_in_threadgroup]],
                                                    uint lane [[thread_index_in_simdgroup]]) {
    const uint grid_h = p.pad_height / p.patch_size;
    const uint grid_w = p.pad_width / p.patch_size;
    const uint num_patches = grid_h * grid_w;
    const uint total_seq = num_patches + p.has_cls_token;
    const uint tok = tg_pos.y;
    if (tok >= total_seq) {
        return;
    }

    const uint h = tg_pos.x * 8u + sg_id;
    const bool valid_h = (h < p.hidden_size);

    if (p.has_cls_token != 0u && tok == 0u) {
        if (valid_h && lane == 0u) {
            const float cls_v = vision_load_f16_or_bf16(class_embed, (ulong)h, VISION_EFF_W_BF16) +
                                vision_load_f16_or_bf16(cls_pos_embed, (ulong)h, VISION_EFF_W_BF16);
            const ulong out_idx = (ulong)p.token_offset * (ulong)p.hidden_size + (ulong)h;
            vision_store_f16_or_bf16(out, out_idx, cls_v, VISION_EFF_O_BF16);
        }
        return;
    }

    const uint p_idx = tok - p.has_cls_token;
    const uint sm = p.spatial_merge;
    const uint blocks_w = grid_w / sm;
    const uint sm_sq = sm * sm;
    const uint block_idx = p_idx / sm_sq;
    const uint sub = p_idx - block_idx * sm_sq;
    const uint block_row = block_idx / blocks_w;
    const uint block_col = block_idx - block_row * blocks_w;
    const uint in_row = sub / sm;
    const uint in_col = sub - in_row * sm;
    const uint patch_row = block_row * sm + in_row;
    const uint patch_col = block_col * sm + in_col;

    const uint patch_area = p.patch_size * p.patch_size;
    const uint token_in_dim = patch_area * 3u;
    const ulong w_row = (ulong)h * (ulong)token_in_dim;

    threadgroup float tg_rgb[256u * 3u];
    float acc = 0.0f;

    for (uint tile_start = 0u; tile_start < patch_area; tile_start += 256u) {
        const uint tile_count = min(256u, patch_area - tile_start);
        if (tid < tile_count) {
            const uint pix = tile_start + tid;
            const uint dy = pix / p.patch_size;
            const uint dx = pix - dy * p.patch_size;
            const uint dst_y = patch_row * p.patch_size + dy;
            const uint dst_x = patch_col * p.patch_size + dx;
            const float3 norm = vision_inkling2_sample_nv12(y0, cbcr0, dst_x, dst_y, p);
            tg_rgb[tid * 3u + 0u] = norm.x;
            tg_rgb[tid * 3u + 1u] = norm.y;
            tg_rgb[tid * 3u + 2u] = norm.z;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (valid_h) {
            for (uint i = lane; i < tile_count; i += 32u) {
                const ulong kb = w_row + (ulong)(tile_start + i) * 3ul;
                const float wr = vision_load_f16_or_bf16(weight, kb + 0ul, VISION_EFF_W_BF16);
                const float wg = vision_load_f16_or_bf16(weight, kb + 1ul, VISION_EFF_W_BF16);
                const float wb = vision_load_f16_or_bf16(weight, kb + 2ul, VISION_EFF_W_BF16);
                acc = fma(tg_rgb[i * 3u + 0u], wr, acc);
                acc = fma(tg_rgb[i * 3u + 1u], wg, acc);
                acc = fma(tg_rgb[i * 3u + 2u], wb, acc);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (valid_h) {
        acc = simd_sum(acc);
        if (lane == 0u) {
            if (p.has_bias != 0u) {
                acc += vision_load_f16_or_bf16(bias, (ulong)h, VISION_EFF_W_BF16);
            }
            if (p.posemb_side > 0u) {
                acc += vision_inkling2_interp_posemb(pos_embed, patch_row, patch_col, grid_h, grid_w, h, p);
            }
            const ulong out_idx = (ulong)(p.token_offset + tok) * (ulong)p.hidden_size + (ulong)h;
            vision_store_f16_or_bf16(out, out_idx, acc, VISION_EFF_O_BF16);
        }
    }
}
