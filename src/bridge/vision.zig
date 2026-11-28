//! Zig wrapper for pure in-memory ImageIO image/GIF decoding and ISO-BMFF /
//! Annex-B VideoToolbox hardware decoding directly into 16 KiB-aligned
//! `IOSurface`-backed `VisionFrame`s (`M9.6 C29` Item 7, zero temp files).

const std = @import("std");
const c = @import("apple_bridge");
const metal = @import("metal.zig");
const page = @import("../mem/page.zig");

pub const MAX_IMAGE_DIM: u32 = c.AMI_VISION_MAX_BYTES_IMAGE_DIM;
pub const MAX_IMAGE_PIXELS: u64 = c.AMI_VISION_MAX_IMAGE_PIXELS;
pub const MAX_VIDEO_FRAMES: u32 = c.AMI_VISION_MAX_VIDEO_FRAMES;
pub const MAX_TOTAL_SURFACE_BYTES: u64 = c.AMI_VISION_MAX_TOTAL_SURFACE_BYTES;
pub const MAX_DECODED_VIDEO_SAMPLES: u32 = c.AMI_VISION_MAX_DECODED_VIDEO_SAMPLES;

pub const Device = metal.Device;
pub const VisionFrame = metal.VisionFrame;
pub const VisionFormat = metal.VisionFormat;
pub const VisionFrameInfo = metal.VisionFrameInfo;
pub const VtClip = metal.VtClip;
pub const VtCodec = metal.VtCodec;
pub const Error = metal.Error;

fn check(status: c.ami_status) Error!void {
    return switch (status) {
        c.AMI_OK => {},
        c.AMI_ERR_INVALID_ARGUMENT => error.InvalidArgument,
        c.AMI_ERR_NO_DEVICE => error.NoDevice,
        c.AMI_ERR_OUT_OF_MEMORY => error.OutOfMemory,
        c.AMI_ERR_LIBRARY => error.LibraryLoadFailed,
        c.AMI_ERR_PIPELINE => error.PipelineCreationFailed,
        c.AMI_ERR_BUFFER => error.BufferWrapFailed,
        c.AMI_ERR_COMPUTE => error.ComputeFailed,
        c.AMI_ERR_TIMEOUT => error.Timeout,
        c.AMI_ERR_UNSUPPORTED => error.Unsupported,
        c.AMI_ERR_TILE => error.TileCreationFailed,
        c.AMI_ERR_ANE => error.AneFailed,
        c.AMI_ERR_VISION => error.VisionFailed,
        else => error.UnexpectedStatus,
    };
}

/// Decodes in-memory image bytes (PNG, JPEG, WebP, HEIC, GIF, BMP, TIFF) via
/// ImageIO directly into a 16 KiB-aligned `BGRA8` `IOSurface`-backed `VisionFrame`
/// with zero temporary files.
pub fn decodeImageBytes(device: Device, bytes: []const u8) Error!VisionFrame {
    if (bytes.len == 0) return error.InvalidArgument;
    var handle: ?*c.ami_vision_frame = null;
    try check(c.ami_image_decode_bytes_iosurface(
        device.handle,
        bytes.ptr,
        bytes.len,
        &handle,
    ));
    return .{ .handle = handle orelse return error.NullHandle };
}

/// Encodes a `BGRA8` `VisionFrame` into an in-memory PNG byte slice owned by `allocator`.
pub fn encodePngBytesAlloc(frame: VisionFrame, allocator: std.mem.Allocator) Error![]u8 {
    var raw_ptr: ?[*]u8 = null;
    var raw_len: usize = 0;
    try check(c.ami_image_encode_png_alloc(frame.handle, &raw_ptr, &raw_len));
    defer if (raw_ptr) |p| c.ami_vision_free_bytes(p);

    const ptr = raw_ptr orelse return error.NullHandle;
    if (raw_len == 0) return error.VisionFailed;
    return allocator.dupe(u8, ptr[0..raw_len]) catch return error.OutOfMemory;
}

/// Encodes a `BGRA8` `VisionFrame` into an in-memory byte slice for `uti`
/// (e.g. `"public.png"`, `"public.jpeg"`, `"public.tiff"`), owned by `allocator`.
pub fn encodeImageBytesAlloc(
    frame: VisionFrame,
    uti: [:0]const u8,
    allocator: std.mem.Allocator,
) Error![]u8 {
    if (uti.len == 0) return error.InvalidArgument;
    var handles: [1]?*c.ami_vision_frame = .{frame.handle};
    var raw_ptr: ?[*]u8 = null;
    var raw_len: usize = 0;
    try check(c.ami_image_encode_uti_alloc(&handles, 1, uti.ptr, &raw_ptr, &raw_len));
    defer if (raw_ptr) |p| c.ami_vision_free_bytes(p);

    const ptr = raw_ptr orelse return error.NullHandle;
    if (raw_len == 0) return error.VisionFailed;
    return allocator.dupe(u8, ptr[0..raw_len]) catch return error.OutOfMemory;
}

/// Encodes 1..128 `BGRA8` `VisionFrame`s into an in-memory multi-frame animated GIF
/// byte slice owned by `allocator`.
pub fn encodeAnimatedGifBytesAlloc(
    frames: []const VisionFrame,
    allocator: std.mem.Allocator,
) Error![]u8 {
    if (frames.len == 0 or frames.len > 128) return error.InvalidArgument;
    var handles: [128]?*c.ami_vision_frame = undefined;
    for (frames, 0..) |f, i| {
        handles[i] = f.handle;
    }
    var raw_ptr: ?[*]u8 = null;
    var raw_len: usize = 0;
    try check(c.ami_image_encode_uti_alloc(
        &handles,
        @intCast(frames.len),
        "com.compuserve.gif",
        &raw_ptr,
        &raw_len,
    ));
    defer if (raw_ptr) |p| c.ami_vision_free_bytes(p);

    const ptr = raw_ptr orelse return error.NullHandle;
    if (raw_len == 0) return error.VisionFailed;
    return allocator.dupe(u8, ptr[0..raw_len]) catch return error.OutOfMemory;
}

/// Muxes a hardware-encoded H.264 or HEVC `VtClip` into a compact, standard
/// ISO-BMFF (`.mp4`) byte buffer in memory, owned by `allocator`.
pub fn exportClipMp4BytesAlloc(clip: VtClip, allocator: std.mem.Allocator) Error![]u8 {
    var raw_ptr: ?[*]u8 = null;
    var raw_len: usize = 0;
    try check(c.ami_vt_clip_export_mp4_bytes(clip.handle, &raw_ptr, &raw_len));
    defer if (raw_ptr) |p| c.ami_vision_free_bytes(p);

    const ptr = raw_ptr orelse return error.NullHandle;
    if (raw_len == 0) return error.VisionFailed;
    return allocator.dupe(u8, ptr[0..raw_len]) catch return error.OutOfMemory;
}

/// Decodes up to `out_frames.len` (`1..64`, evenly sampled across total frames
/// when the clip has more than `out_frames.len` frames) from an in-memory video
/// or animated image payload (ISO-BMFF MP4/MOV H.264/HEVC, raw Annex-B, or
/// ImageIO animated GIF/APNG/WebP/single-image) directly into 16 KiB-aligned
/// `IOSurface`-backed `VisionFrame`s, returning the populated prefix slice.
pub fn decodeVideoBytesInto(
    device: Device,
    bytes: []const u8,
    out_frames: []VisionFrame,
) Error![]VisionFrame {
    if (bytes.len == 0 or out_frames.len == 0 or out_frames.len > MAX_VIDEO_FRAMES) {
        return error.InvalidArgument;
    }

    var raw_frames: [MAX_VIDEO_FRAMES]?*c.ami_vision_frame = @splat(null);
    var count: u32 = 0;
    try check(c.ami_video_decode_bytes_iosurface(
        device.handle,
        bytes.ptr,
        bytes.len,
        @intCast(out_frames.len),
        &raw_frames,
        &count,
    ));

    if (count > out_frames.len) {
        for (raw_frames[0..out_frames.len]) |opt_h| {
            if (opt_h) |h| c.ami_vision_frame_destroy(h);
        }
        return error.VisionFailed;
    }

    var built: usize = 0;
    errdefer for (out_frames[0..built]) |f| f.close();
    for (0..count) |i| {
        const handle = raw_frames[i] orelse {
            for (raw_frames[i + 1 .. count]) |rem| {
                if (rem) |h| c.ami_vision_frame_destroy(h);
            }
            return error.NullHandle;
        };
        out_frames[i] = .{ .handle = handle };
        built += 1;
    }
    return out_frames[0..count];
}

/// Allocates and returns up to `max_frames` (`1..64`) evenly sampled decoded
/// `VisionFrame`s from an in-memory video/animated payload. Caller must close
/// each frame and free the returned slice (e.g. via `freeDecodedFrames`).
pub fn decodeVideoBytesAlloc(
    device: Device,
    bytes: []const u8,
    max_frames: u32,
    allocator: std.mem.Allocator,
) Error![]VisionFrame {
    if (bytes.len == 0 or max_frames == 0 or max_frames > MAX_VIDEO_FRAMES) {
        return error.InvalidArgument;
    }

    var stack_frames: [MAX_VIDEO_FRAMES]VisionFrame = undefined;
    const decoded = try decodeVideoBytesInto(device, bytes, stack_frames[0..max_frames]);
    errdefer for (decoded) |f| f.close();

    const out = allocator.alloc(VisionFrame, decoded.len) catch return error.OutOfMemory;
    @memcpy(out, decoded);
    return out;
}

/// Closes every decoded `VisionFrame` in `frames` and frees the slice.
pub fn freeDecodedFrames(allocator: std.mem.Allocator, frames: []VisionFrame) void {
    for (frames) |f| f.close();
    allocator.free(frames);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "bridge.vision: in-memory PNG & JPEG decode -> 16 KiB IOSurface, pixel accuracy, and MTLTexture residency" {
    const gpa = std.testing.allocator;
    const device = try Device.open();
    defer device.close();

    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();

    const w: u32 = 64;
    const h: u32 = 48;
    const src = try VisionFrame.init(device, .bgra8, w, h);
    defer src.close();

    {
        var g = try src.lockPlane(0, false);
        errdefer g.release();
        for (0..h) |y| {
            const row = g.rowBytes(@intCast(y));
            for (0..w) |x| {
                if (x < 16 and y < 16) {
                    row[x * 4 + 0] = 24;
                    row[x * 4 + 1] = 16;
                    row[x * 4 + 2] = 240;
                    row[x * 4 + 3] = 255;
                } else if (x >= w - 16 and y >= h - 16) {
                    row[x * 4 + 0] = 236;
                    row[x * 4 + 1] = 32;
                    row[x * 4 + 2] = 20;
                    row[x * 4 + 3] = 255;
                } else {
                    row[x * 4 + 0] = @intCast((x * 3 + y * 5) % 256);
                    row[x * 4 + 1] = @intCast((x * 7 + y * 2) % 256);
                    row[x * 4 + 2] = @intCast((x * 11 + y * 3) % 256);
                    row[x * 4 + 3] = 255;
                }
            }
        }
        try g.unlock();
    }

    // 1. In-memory PNG encode -> decode round-trip (exact pixel parity).
    const png_bytes = try encodePngBytesAlloc(src, gpa);
    defer gpa.free(png_bytes);
    try std.testing.expect(png_bytes.len > 16);
    try std.testing.expectEqualSlices(u8, "\x89PNG\r\n\x1a\n", png_bytes[0..8]);

    const png_frame = try decodeImageBytes(device, png_bytes);
    defer png_frame.close();
    const pi = try png_frame.info();
    try std.testing.expectEqual(VisionFormat.bgra8, pi.format());
    try std.testing.expectEqual(w, pi.width());
    try std.testing.expectEqual(h, pi.height());
    try std.testing.expect(pi.zeroCopyVerified());
    try std.testing.expect(pi.isPageAligned());
    try std.testing.expectEqual(@as(u64, 0), pi.baseAddress() % page.APPLE_PAGE_SIZE);
    try std.testing.expectEqual(@as(u64, 0), pi.surfaceAllocSize() % page.APPLE_PAGE_SIZE);
    try std.testing.expect(pi.planeResourceId(0) != 0);

    try compute.makeVisionFramesResident(&.{png_frame});
    try compute.removeVisionFramesResident(&.{png_frame});

    {
        var lg = try png_frame.lockPlane(0, true);
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

    // 2. In-memory JPEG encode -> decode round-trip + repeated decode autorelease check.
    const jpeg_bytes = try encodeImageBytesAlloc(src, "public.jpeg", gpa);
    defer gpa.free(jpeg_bytes);
    try std.testing.expect(jpeg_bytes.len > 16);
    try std.testing.expectEqual(@as(u8, 0xFF), jpeg_bytes[0]);
    try std.testing.expectEqual(@as(u8, 0xD8), jpeg_bytes[1]);

    for (0..8) |_| {
        const jf = try decodeImageBytes(device, jpeg_bytes);
        defer jf.close();
        const ji = try jf.info();
        try std.testing.expectEqual(VisionFormat.bgra8, ji.format());
        try std.testing.expectEqual(w, ji.width());
        try std.testing.expectEqual(h, ji.height());
        try std.testing.expect(ji.zeroCopyVerified());
        try std.testing.expect(ji.isPageAligned());
        try std.testing.expectEqual(@as(u64, 0), ji.baseAddress() % page.APPLE_PAGE_SIZE);
        try std.testing.expectEqual(@as(u64, 0), ji.surfaceAllocSize() % page.APPLE_PAGE_SIZE);
        try std.testing.expect(ji.planeResourceId(0) != 0);

        try compute.makeVisionFramesResident(&.{jf});
        try compute.removeVisionFramesResident(&.{jf});

        var jg = try jf.lockPlane(0, true);
        errdefer jg.release();
        const jtop = jg.rowBytes(0);
        try std.testing.expect(jtop[2] > jtop[0] + 100); // strong red at (0,0)
        const jbot = jg.rowBytes(h - 1);
        try std.testing.expect(jbot[(w - 1) * 4 + 0] > jbot[(w - 1) * 4 + 2] + 100); // strong blue at (w-1,h-1)
        try jg.unlock();
    }

    // 3. In-memory TIFF & BMP encode -> decode round-trip.
    for ([_][:0]const u8{ "public.tiff", "com.microsoft.bmp" }) |uti| {
        const enc = try encodeImageBytesAlloc(src, uti, gpa);
        defer gpa.free(enc);
        const dec = try decodeImageBytes(device, enc);
        defer dec.close();
        const di = try dec.info();
        try std.testing.expectEqual(VisionFormat.bgra8, di.format());
        try std.testing.expectEqual(w, di.width());
        try std.testing.expectEqual(h, di.height());
        try std.testing.expect(di.zeroCopyVerified());
        try std.testing.expect(di.isPageAligned());
    }
}

test "bridge.vision: hardware H.264 & HEVC VtClip -> exportClipMp4BytesAlloc -> decodeVideoBytesInto (12 -> 4)" {
    const gpa = std.testing.allocator;
    const device = try Device.open();
    defer device.close();

    var compute = try metal.Compute.initWithFrames(device, 2);
    defer compute.close();

    const width: u32 = 112;
    const height: u32 = 112;
    const codecs = [_]VtCodec{ .h264, .hevc };

    for (codecs) |codec| {
        const clip = try VtClip.encodeTestClip(codec, width, height, 12, 30);
        defer clip.close();
        const ci = try clip.info();
        try std.testing.expectEqual(@as(u32, 12), ci.frameCount());
        try std.testing.expect(ci.usingHardwareEncoder());

        const mp4_bytes = try exportClipMp4BytesAlloc(clip, gpa);
        defer gpa.free(mp4_bytes);
        try std.testing.expect(mp4_bytes.len > 64);
        try std.testing.expectEqualStrings("ftyp", mp4_bytes[4..8]);

        var out_buf: [4]VisionFrame = undefined;
        const frames = try decodeVideoBytesInto(device, mp4_bytes, &out_buf);
        defer for (frames) |f| f.close();
        try std.testing.expectEqual(@as(usize, 4), frames.len);

        try compute.makeVisionFramesResident(frames);
        defer compute.removeVisionFramesResident(frames) catch |err| {
            std.debug.panic("removeVisionFramesResident failed: {s}", .{@errorName(err)});
        };

        var prev_pts: i64 = -1;
        for (frames) |f| {
            const fi = try f.info();
            try std.testing.expectEqual(VisionFormat.nv12, fi.format());
            try std.testing.expectEqual(width, fi.width());
            try std.testing.expectEqual(height, fi.height());
            try std.testing.expect(fi.isVtDecoded());
            try std.testing.expect(fi.zeroCopyVerified());
            try std.testing.expectEqual(@as(u64, 0), fi.baseAddress() % page.APPLE_PAGE_SIZE);
            try std.testing.expect(fi.surfaceAllocSize() > 0);
            try std.testing.expect(fi.planeResourceId(0) != 0);
            try std.testing.expect(fi.planeResourceId(1) != 0);
            try std.testing.expect(fi.ptsValue() > prev_pts);
            prev_pts = fi.ptsValue();
        }

        // Also test allocating variant requesting 6 frames from the 12-frame MP4.
        const alloc_frames = try decodeVideoBytesAlloc(device, mp4_bytes, 6, gpa);
        defer freeDecodedFrames(gpa, alloc_frames);
        try std.testing.expectEqual(@as(usize, 6), alloc_frames.len);
    }
}

test "bridge.vision: multi-frame animated GIF & single-image fallback via decodeVideoBytesInto" {
    const gpa = std.testing.allocator;
    const device = try Device.open();
    defer device.close();

    const w: u32 = 32;
    const h: u32 = 32;
    var src_frames: [6]VisionFrame = undefined;
    var built: usize = 0;
    defer for (src_frames[0..built]) |f| f.close();

    for (0..6) |idx| {
        const f = try VisionFrame.init(device, .bgra8, w, h);
        src_frames[idx] = f;
        built += 1;

        var g = try f.lockPlane(0, false);
        errdefer g.release();
        const r_val: u8 = @intCast(30 + idx * 38);
        const b_val: u8 = @intCast(230 - idx * 36);
        for (0..h) |y| {
            const row = g.rowBytes(@intCast(y));
            for (0..w) |x| {
                row[x * 4 + 0] = b_val;
                row[x * 4 + 1] = 80;
                row[x * 4 + 2] = r_val;
                row[x * 4 + 3] = 255;
            }
        }
        try g.unlock();
    }

    const gif_bytes = try encodeAnimatedGifBytesAlloc(&src_frames, gpa);
    defer gpa.free(gif_bytes);
    try std.testing.expect(gif_bytes.len > 32);
    try std.testing.expectEqualStrings("GIF8", gif_bytes[0..4]);

    var gif_out: [3]VisionFrame = undefined;
    const gif_decoded = try decodeVideoBytesInto(device, gif_bytes, &gif_out);
    defer for (gif_decoded) |f| f.close();
    try std.testing.expectEqual(@as(usize, 3), gif_decoded.len);

    var prev_pts: i64 = -1;
    var prev_r: u8 = 0;
    for (gif_decoded, 0..) |f, i| {
        const fi = try f.info();
        try std.testing.expectEqual(VisionFormat.bgra8, fi.format());
        try std.testing.expectEqual(w, fi.width());
        try std.testing.expectEqual(h, fi.height());
        try std.testing.expect(fi.zeroCopyVerified());
        try std.testing.expect(fi.isPageAligned());
        try std.testing.expectEqual(@as(u64, 0), fi.baseAddress() % page.APPLE_PAGE_SIZE);
        try std.testing.expectEqual(@as(u64, 0), fi.surfaceAllocSize() % page.APPLE_PAGE_SIZE);
        try std.testing.expect(fi.ptsValue() > prev_pts);
        prev_pts = fi.ptsValue();

        var g = try f.lockPlane(0, true);
        errdefer g.release();
        const r_chan = g.rowBytes(0)[2];
        if (i > 0) {
            try std.testing.expect(r_chan > prev_r + 20);
        }
        prev_r = r_chan;
        try g.unlock();
    }

    // Single-image PNG fallback via decodeVideoBytesInto returns 1 frame.
    const png_bytes = try encodePngBytesAlloc(src_frames[0], gpa);
    defer gpa.free(png_bytes);
    var single_out: [4]VisionFrame = undefined;
    const single_decoded = try decodeVideoBytesInto(device, png_bytes, &single_out);
    defer for (single_decoded) |f| f.close();
    try std.testing.expectEqual(@as(usize, 1), single_decoded.len);
    const si = try single_decoded[0].info();
    try std.testing.expectEqual(VisionFormat.bgra8, si.format());
    try std.testing.expectEqual(w, si.width());
    try std.testing.expectEqual(h, si.height());
}

test "bridge.vision: rejects empty, invalid image, and truncated/corrupt ISO-BMFF payloads cleanly" {
    const gpa = std.testing.allocator;
    const device = try Device.open();
    defer device.close();

    var out_buf: [4]VisionFrame = undefined;
    var oversize_buf: [MAX_VIDEO_FRAMES + 1]VisionFrame = undefined;

    // 1. Argument bounds
    try std.testing.expectError(error.InvalidArgument, decodeImageBytes(device, &.{}));
    try std.testing.expectError(error.InvalidArgument, decodeVideoBytesInto(device, &.{}, &out_buf));
    try std.testing.expectError(error.InvalidArgument, decodeVideoBytesInto(device, "abc", out_buf[0..0]));
    try std.testing.expectError(error.InvalidArgument, decodeVideoBytesInto(device, "abc", &oversize_buf));
    try std.testing.expectError(error.InvalidArgument, decodeVideoBytesAlloc(device, "abc", 0, gpa));
    try std.testing.expectError(error.InvalidArgument, decodeVideoBytesAlloc(device, "abc", MAX_VIDEO_FRAMES + 1, gpa));

    // 2. Invalid / truncated image bytes
    try std.testing.expectError(error.VisionFailed, decodeImageBytes(device, "not-an-image-payload"));
    try std.testing.expectError(error.VisionFailed, decodeVideoBytesInto(device, "not-a-video-payload", &out_buf));

    const tmp_frame = try VisionFrame.init(device, .bgra8, 32, 32);
    defer tmp_frame.close();
    const png_bytes = try encodePngBytesAlloc(tmp_frame, gpa);
    defer gpa.free(png_bytes);
    try std.testing.expectError(error.VisionFailed, decodeImageBytes(device, png_bytes[0..12]));

    // 3. Corrupt / truncated ISO-BMFF MP4 payloads
    const clip = try VtClip.encodeTestClip(.h264, 64, 64, 4, 30);
    defer clip.close();
    const mp4_bytes = try exportClipMp4BytesAlloc(clip, gpa);
    defer gpa.free(mp4_bytes);

    // Truncated ftyp-only
    const ftyp_len = std.mem.readInt(u32, mp4_bytes[0..4], .big);
    try std.testing.expectError(error.VisionFailed, decodeVideoBytesInto(device, mp4_bytes[0..ftyp_len], &out_buf));

    // Truncated mid-mdat
    try std.testing.expectError(error.VisionFailed, decodeVideoBytesInto(device, mp4_bytes[0 .. mp4_bytes.len - 12], &out_buf));

    // Corrupt zero/invalid box size in moov
    const bad_box = [_]u8{ 0, 0, 0, 4, 'm', 'o', 'o', 'v' };
    try std.testing.expectError(error.VisionFailed, decodeVideoBytesInto(device, &bad_box, &out_buf));

    // Invalid raw Annex-B stream missing SPS/PPS
    const bad_annexb = [_]u8{ 0, 0, 0, 1, 0x65, 0x88, 0x84, 0x21 };
    try std.testing.expectError(error.VisionFailed, decodeVideoBytesInto(device, &bad_annexb, &out_buf));

    // Corrupt stsz sample count inside valid MP4 copy
    const mutated = try gpa.dupe(u8, mp4_bytes);
    defer gpa.free(mutated);
    if (std.mem.indexOf(u8, mutated, "stsz")) |stsz_off| {
        // stsz payload: version+flags(4), sample_size(4), sample_count(4)
        std.mem.writeInt(u32, mutated[stsz_off + 12 ..][0..4], 0x7FFF_FFFF, .big);
        try std.testing.expectError(error.VisionFailed, decodeVideoBytesInto(device, mutated, &out_buf));
    } else {
        return error.TestUnexpectedResult;
    }

    // Corrupt stco chunk offset pointing past EOF
    const mutated_stco = try gpa.dupe(u8, mp4_bytes);
    defer gpa.free(mutated_stco);
    if (std.mem.indexOf(u8, mutated_stco, "stco")) |stco_off| {
        // stco payload: version+flags(4), entry_count(4), chunk_offset[0](4)
        std.mem.writeInt(u32, mutated_stco[stco_off + 12 ..][0..4], 0x7FFF_0000, .big);
        try std.testing.expectError(error.VisionFailed, decodeVideoBytesInto(device, mutated_stco, &out_buf));
    } else {
        return error.TestUnexpectedResult;
    }
}

fn appendPngChunk(buf: *std.ArrayList(u8), allocator: std.mem.Allocator, tag: *const [4]u8, payload: []const u8) !void {
    var len_be: [4]u8 = undefined;
    std.mem.writeInt(u32, &len_be, @intCast(payload.len), .big);
    try buf.appendSlice(allocator, &len_be);
    try buf.appendSlice(allocator, tag);
    try buf.appendSlice(allocator, payload);
    var crc = std.hash.Crc32.init();
    crc.update(tag);
    crc.update(payload);
    var crc_be: [4]u8 = undefined;
    std.mem.writeInt(u32, &crc_be, crc.final(), .big);
    try buf.appendSlice(allocator, &crc_be);
}

fn makeTinyPngWithDimsAlloc(allocator: std.mem.Allocator, width: u32, height: u32) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "\x89PNG\r\n\x1a\n");
    var ihdr: [13]u8 = undefined;
    std.mem.writeInt(u32, ihdr[0..4], width, .big);
    std.mem.writeInt(u32, ihdr[4..8], height, .big);
    ihdr[8] = 8; // bit depth
    ihdr[9] = 6; // RGBA
    ihdr[10] = 0;
    ihdr[11] = 0;
    ihdr[12] = 0;
    try appendPngChunk(&out, allocator, "IHDR", &ihdr);
    // Minimal zlib stream (valid empty/tiny deflate block) + IEND so ImageIO parses header cleanly.
    const idat = [_]u8{ 0x78, 0x9C, 0x63, 0x00, 0x00, 0x00, 0x02, 0x00, 0x01 };
    try appendPngChunk(&out, allocator, "IDAT", &idat);
    try appendPngChunk(&out, allocator, "IEND", &.{});
    return out.toOwnedSlice(allocator);
}

fn appendIsoBox(buf: *std.ArrayList(u8), allocator: std.mem.Allocator, tag: *const [4]u8, payload: []const u8) !void {
    var hdr: [8]u8 = undefined;
    std.mem.writeInt(u32, hdr[0..4], @intCast(8 + payload.len), .big);
    @memcpy(hdr[4..8], tag);
    try buf.appendSlice(allocator, &hdr);
    try buf.appendSlice(allocator, payload);
}

test "bridge.vision: F8 rejects oversized PNG header bombs (16384x16384 and 8192x8192) before pixel allocation" {
    const gpa = std.testing.allocator;
    const device = try Device.open();
    defer device.close();

    var out_buf: [2]VisionFrame = undefined;

    // 1. 16384x16384 exceeds MAX_IMAGE_DIM (8192).
    const bomb_16k = try makeTinyPngWithDimsAlloc(gpa, 16384, 16384);
    defer gpa.free(bomb_16k);
    try std.testing.expectError(error.VisionFailed, decodeImageBytes(device, bomb_16k));
    try std.testing.expectError(error.VisionFailed, decodeVideoBytesInto(device, bomb_16k, &out_buf));

    // 2. 8192x8192 fits per-axis 8192 but exceeds MAX_IMAGE_PIXELS (32 MP / 128 MiB RGBA).
    const bomb_64mp = try makeTinyPngWithDimsAlloc(gpa, 8192, 8192);
    defer gpa.free(bomb_64mp);
    try std.testing.expectError(error.VisionFailed, decodeImageBytes(device, bomb_64mp));
    try std.testing.expectError(error.VisionFailed, decodeVideoBytesInto(device, bomb_64mp, &out_buf));
}

test "bridge.vision: F9 rejects aliased stco chunks and excessive GOP decode amplification, and N5 supports HEIC fallback" {
    const gpa = std.testing.allocator;
    const device = try Device.open();
    defer device.close();

    // Build a valid baseline MP4 to borrow genuine avcC and sample bytes.
    const clip = try VtClip.encodeTestClip(.h264, 64, 64, 2, 30);
    defer clip.close();
    const valid_mp4 = try exportClipMp4BytesAlloc(clip, gpa);
    defer gpa.free(valid_mp4);

    const avcc_tag_pos = std.mem.indexOf(u8, valid_mp4, "avcC") orelse return error.TestUnexpectedResult;
    const avcc_len = std.mem.readInt(u32, valid_mp4[avcc_tag_pos - 4 ..][0..4], .big);
    const avcc_payload = valid_mp4[avcc_tag_pos + 4 .. avcc_tag_pos - 4 + avcc_len];

    const mdat_tag_pos = std.mem.indexOf(u8, valid_mp4, "mdat") orelse return error.TestUnexpectedResult;
    const mdat_payload_off: u32 = @intCast(mdat_tag_pos + 4);
    const stsz_tag_pos = std.mem.indexOf(u8, valid_mp4, "stsz") orelse return error.TestUnexpectedResult;
    const sample0_sz = std.mem.readInt(u32, valid_mp4[stsz_tag_pos + 16 ..][0..4], .big);

    // Helper closure-style builder for crafted MP4 tables (`stsz`, `stco`, `stsc`, optional `stss`).
    const Builder = struct {
        fn buildMp4(
            alloc: std.mem.Allocator,
            avcc: []const u8,
            sample_count: u32,
            uniform_sz: u32,
            chunk_offsets: []const u32,
            stsc_entries: []const [3]u32,
            stss_syncs: ?[]const u32,
            mdat_bytes: []const u8,
        ) ![]u8 {
            var stsd: std.ArrayList(u8) = .empty;
            defer stsd.deinit(alloc);
            var avc1: std.ArrayList(u8) = .empty;
            defer avc1.deinit(alloc);
            try avc1.appendNTimes(alloc, 0, 24);
            var wh: [4]u8 = undefined;
            std.mem.writeInt(u16, wh[0..2], 64, .big);
            std.mem.writeInt(u16, wh[2..4], 64, .big);
            try avc1.appendSlice(alloc, &wh);
            try avc1.appendNTimes(alloc, 0, 50);
            try appendIsoBox(&avc1, alloc, "avcC", avcc);

            var stsd_head: [8]u8 = .{ 0, 0, 0, 0, 0, 0, 0, 1 };
            try stsd.appendSlice(alloc, &stsd_head);
            try appendIsoBox(&stsd, alloc, "avc1", avc1.items);

            var stsz: std.ArrayList(u8) = .empty;
            defer stsz.deinit(alloc);
            var stsz_hdr: [12]u8 = @splat(0);
            std.mem.writeInt(u32, stsz_hdr[4..8], uniform_sz, .big);
            std.mem.writeInt(u32, stsz_hdr[8..12], sample_count, .big);
            try stsz.appendSlice(alloc, &stsz_hdr);

            var stsc: std.ArrayList(u8) = .empty;
            defer stsc.deinit(alloc);
            var stsc_hdr: [8]u8 = @splat(0);
            std.mem.writeInt(u32, stsc_hdr[4..8], @intCast(stsc_entries.len), .big);
            try stsc.appendSlice(alloc, &stsc_hdr);
            for (stsc_entries) |e| {
                var rec: [12]u8 = undefined;
                std.mem.writeInt(u32, rec[0..4], e[0], .big);
                std.mem.writeInt(u32, rec[4..8], e[1], .big);
                std.mem.writeInt(u32, rec[8..12], e[2], .big);
                try stsc.appendSlice(alloc, &rec);
            }

            var stco: std.ArrayList(u8) = .empty;
            defer stco.deinit(alloc);
            var stco_hdr: [8]u8 = @splat(0);
            std.mem.writeInt(u32, stco_hdr[4..8], @intCast(chunk_offsets.len), .big);
            try stco.appendSlice(alloc, &stco_hdr);
            for (chunk_offsets) |off| {
                var rec: [4]u8 = undefined;
                std.mem.writeInt(u32, &rec, off, .big);
                try stco.appendSlice(alloc, &rec);
            }

            var stbl: std.ArrayList(u8) = .empty;
            defer stbl.deinit(alloc);
            try appendIsoBox(&stbl, alloc, "stsd", stsd.items);
            try appendIsoBox(&stbl, alloc, "stsz", stsz.items);
            try appendIsoBox(&stbl, alloc, "stsc", stsc.items);
            try appendIsoBox(&stbl, alloc, "stco", stco.items);
            if (stss_syncs) |syncs| {
                var stss: std.ArrayList(u8) = .empty;
                defer stss.deinit(alloc);
                var stss_hdr: [8]u8 = @splat(0);
                std.mem.writeInt(u32, stss_hdr[4..8], @intCast(syncs.len), .big);
                try stss.appendSlice(alloc, &stss_hdr);
                for (syncs) |s1| {
                    var rec: [4]u8 = undefined;
                    std.mem.writeInt(u32, &rec, s1, .big);
                    try stss.appendSlice(alloc, &rec);
                }
                try appendIsoBox(&stbl, alloc, "stss", stss.items);
            }

            var minf: std.ArrayList(u8) = .empty;
            defer minf.deinit(alloc);
            try appendIsoBox(&minf, alloc, "stbl", stbl.items);

            var mdia: std.ArrayList(u8) = .empty;
            defer mdia.deinit(alloc);
            var mdhd: [20]u8 = @splat(0);
            std.mem.writeInt(u32, mdhd[12..16], 30000, .big);
            try appendIsoBox(&mdia, alloc, "mdhd", &mdhd);
            try appendIsoBox(&mdia, alloc, "minf", minf.items);

            var trak: std.ArrayList(u8) = .empty;
            defer trak.deinit(alloc);
            try appendIsoBox(&trak, alloc, "mdia", mdia.items);

            var moov: std.ArrayList(u8) = .empty;
            defer moov.deinit(alloc);
            try appendIsoBox(&moov, alloc, "trak", trak.items);

            var file: std.ArrayList(u8) = .empty;
            errdefer file.deinit(alloc);
            try appendIsoBox(&file, alloc, "ftyp", "isom\x00\x00\x02\x00isom");
            try appendIsoBox(&file, alloc, "mdat", mdat_bytes);
            try appendIsoBox(&file, alloc, "moov", moov.items);
            return file.toOwnedSlice(alloc);
        }
    };

    var out_buf: [2]VisionFrame = undefined;
    const mdat_sample = valid_mp4[mdat_payload_off..][0..sample0_sz];

    // 1. Aliased/overlapping stco chunk offsets pointing to the same offset (16 = inside mdat payload)
    const aliased_mp4 = try Builder.buildMp4(
        gpa,
        avcc_payload,
        2,
        sample0_sz,
        &.{ 28, 28 },
        &.{.{ 1, 1, 1 }},
        null,
        mdat_sample,
    );
    defer gpa.free(aliased_mp4);
    try std.testing.expectError(error.VisionFailed, decodeVideoBytesInto(device, aliased_mp4, &out_buf));

    // 2. Crafted MP4 exceeding MAX_DECODED_VIDEO_SAMPLES (2048) with 1 sync frame at sample 1
    //    (non-overlapping uniform 4-byte samples inside a ~9 KB mdat).
    const big_count: u32 = MAX_DECODED_VIDEO_SAMPLES + 16;
    const fake_mdat = try gpa.alloc(u8, big_count * 4);
    defer gpa.free(fake_mdat);
    @memset(fake_mdat, 0);
    const amp_mp4 = try Builder.buildMp4(
        gpa,
        avcc_payload,
        big_count,
        4,
        &.{28},
        &.{.{ 1, big_count, 1 }},
        &.{1},
        fake_mdat,
    );
    defer gpa.free(amp_mp4);
    try std.testing.expectError(error.VisionFailed, decodeVideoBytesInto(device, amp_mp4, &out_buf));

    // 3. N5: Single HEIC image (`ftyp` major brand `heic`) via decodeVideoBytesInto succeeds!
    const src_f = try VisionFrame.init(device, .bgra8, 32, 32);
    defer src_f.close();
    {
        var g = try src_f.lockPlane(0, false);
        errdefer g.release();
        for (0..32) |y| {
            const row = g.rowBytes(@intCast(y));
            for (0..32) |x| {
                row[x * 4 + 0] = 40;
                row[x * 4 + 1] = 180;
                row[x * 4 + 2] = 90;
                row[x * 4 + 3] = 255;
            }
        }
        try g.unlock();
    }
    const heic_bytes = try encodeImageBytesAlloc(src_f, "public.heic", gpa);
    defer gpa.free(heic_bytes);
    try std.testing.expect(heic_bytes.len > 16);
    try std.testing.expectEqualStrings("ftyp", heic_bytes[4..8]);
    const heic_frames = try decodeVideoBytesInto(device, heic_bytes, &out_buf);
    defer for (heic_frames) |f| f.close();
    try std.testing.expectEqual(@as(usize, 1), heic_frames.len);
    const hi = try heic_frames[0].info();
    try std.testing.expectEqual(@as(u32, 32), hi.width());
    try std.testing.expectEqual(@as(u32, 32), hi.height());
}
