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
using namespace metal;

constant bool VISION_WEIGHT_BF16 [[function_constant(0)]];
constant bool VISION_OUT_BF16 [[function_constant(1)]];

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
