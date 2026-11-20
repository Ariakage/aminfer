//! Zig view of the Metal half of the bridge (`apple_bridge.h`). Converts
//! status codes into an explicit error set and null-checks every handle the
//! bridge returns.

const std = @import("std");
const c = @import("apple_bridge");
const page = @import("../mem/page.zig");

/// The kernel library compiled from `src/kernels/*.metal` by `build.zig`.
const embedded_metallib = @embedFile("aminfer_metallib");

pub const Error = error{
    InvalidArgument,
    NoDevice,
    OutOfMemory,
    LibraryLoadFailed,
    PipelineCreationFailed,
    BufferWrapFailed,
    ComputeFailed,
    Timeout,
    Unsupported,
    /// The bridge returned a status this wrapper does not know.
    UnexpectedStatus,
    /// The bridge reported success but returned a null handle.
    NullHandle,
};

pub const WrapError = page.ShareError || Error;

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
        else => error.UnexpectedStatus,
    };
}

pub const DeviceInfo = struct {
    raw: c.ami_device_info,

    pub fn name(self: *const DeviceInfo) []const u8 {
        return std.mem.sliceTo(&self.raw.name, 0);
    }
    pub fn recommendedMaxWorkingSetSize(self: DeviceInfo) u64 {
        return self.raw.recommended_max_working_set_size;
    }
    pub fn maxBufferLength(self: DeviceInfo) u64 {
        return self.raw.max_buffer_length;
    }
    pub fn hasUnifiedMemory(self: DeviceInfo) bool {
        return self.raw.has_unified_memory != 0;
    }
    pub fn supportsMetal4(self: DeviceInfo) bool {
        return self.raw.supports_metal4 != 0;
    }
    pub fn appleGpuFamily(self: DeviceInfo) u32 {
        return self.raw.apple_gpu_family;
    }
    pub fn aneCoreCount(self: DeviceInfo) u32 {
        return self.raw.ane_core_count;
    }
};

pub const Device = struct {
    handle: *c.ami_device,

    pub fn open() Error!Device {
        var handle: ?*c.ami_device = null;
        try check(c.ami_device_create(&handle));
        return .{ .handle = handle orelse return error.NullHandle };
    }

    pub fn close(self: Device) void {
        c.ami_device_destroy(self.handle);
    }

    pub fn info(self: Device) Error!DeviceInfo {
        var raw: c.ami_device_info = undefined;
        try check(c.ami_device_query(self.handle, &raw));
        return .{ .raw = raw };
    }
};

pub const Library = struct {
    handle: *c.ami_library,

    /// Loads the metallib embedded at build time.
    pub fn openEmbedded(device: Device) Error!Library {
        var handle: ?*c.ami_library = null;
        try check(c.ami_library_create(device.handle, embedded_metallib.ptr, embedded_metallib.len, &handle));
        return .{ .handle = handle orelse return error.NullHandle };
    }

    pub fn close(self: Library) void {
        c.ami_library_destroy(self.handle);
    }

    pub fn hasFunction(self: Library, function_name: [:0]const u8) Error!bool {
        var present: u32 = 0;
        try check(c.ami_library_has_function(self.handle, function_name.ptr, &present));
        return present != 0;
    }
};

/// Zero-copy `MTLBuffer` view over caller-owned, 16 KiB-aligned wired memory
/// (ADR-0002). The backing region must outlive this buffer and any residency
/// set referencing it.
pub const Buffer = struct {
    handle: *c.ami_buffer,

    pub fn wrap(device: Device, bytes: []u8) WrapError!Buffer {
        try page.checkShareable(bytes.ptr, bytes.len);
        var handle: ?*c.ami_buffer = null;
        try check(c.ami_buffer_wrap(device.handle, bytes.ptr, bytes.len, &handle));
        return .{ .handle = handle orelse return error.NullHandle };
    }

    pub fn close(self: Buffer) void {
        c.ami_buffer_destroy(self.handle);
    }

    pub fn gpuAddress(self: Buffer) u64 {
        return c.ami_buffer_gpu_address(self.handle);
    }
};

/// Compute pipeline built at startup through the Metal 4 compiler.
pub const Pipeline = struct {
    handle: *c.ami_pipeline,

    pub fn init(device: Device, library: Library, function_name: [:0]const u8) Error!Pipeline {
        var handle: ?*c.ami_pipeline = null;
        try check(c.ami_pipeline_create(device.handle, library.handle, function_name.ptr, &handle));
        return .{ .handle = handle orelse return error.NullHandle };
    }

    pub fn close(self: Pipeline) void {
        c.ami_pipeline_destroy(self.handle);
    }
};

/// Reusable Metal 4 compute submission context (ADR-0011).
pub const Compute = struct {
    handle: *c.ami_compute,

    pub fn init(device: Device) Error!Compute {
        var handle: ?*c.ami_compute = null;
        try check(c.ami_compute_create(device.handle, &handle));
        return .{ .handle = handle orelse return error.NullHandle };
    }

    pub fn close(self: Compute) void {
        c.ami_compute_destroy(self.handle);
    }

    /// Adds buffers to this context's residency set and commits residency.
    /// Zero heap allocation: submits in fixed stack batches.
    pub fn makeResident(self: Compute, buffers: []const Buffer) Error!void {
        const batch_cap = 64;
        var offset: usize = 0;
        while (offset < buffers.len) {
            const count = @min(batch_cap, buffers.len - offset);
            var handles: [batch_cap]?*c.ami_buffer = undefined;
            for (buffers[offset .. offset + count], 0..) |buf, i| {
                handles[i] = buf.handle;
            }
            try check(c.ami_compute_make_resident(self.handle, &handles, count));
            offset += count;
        }
    }

    /// Removes all allocations from this context's residency set and commits.
    pub fn clearResidency(self: Compute) Error!void {
        try check(c.ami_compute_clear_residency(self.handle));
    }

    /// Encodes a 1D dispatch on the reusable command buffer and waits for
    /// shared-event completion. `buffers[i]` binds to `[[buffer(i)]]`.
    pub fn dispatch1dSync(
        self: Compute,
        pipeline: Pipeline,
        buffers: []const Buffer,
        threads: u64,
        timeout_ms: u64,
    ) Error!void {
        if (threads == 0 or buffers.len > c.AMI_MAX_BUFFER_BINDINGS) {
            return error.InvalidArgument;
        }
        var handles: [c.AMI_MAX_BUFFER_BINDINGS]?*c.ami_buffer = undefined;
        for (buffers, 0..) |buf, i| {
            handles[i] = buf.handle;
        }
        const ptr: ?[*]const ?*c.ami_buffer = if (buffers.len > 0) &handles else null;
        try check(c.ami_compute_dispatch_1d_sync(
            self.handle,
            pipeline.handle,
            ptr,
            buffers.len,
            threads,
            timeout_ms,
        ));
    }
};

test "default device is a unified-memory Apple GPU with Metal 4" {
    const device = try Device.open();
    defer device.close();
    const info = try device.info();
    try std.testing.expect(info.name().len > 0);
    try std.testing.expect(info.hasUnifiedMemory());
    try std.testing.expect(info.supportsMetal4());
    try std.testing.expect(info.appleGpuFamily() >= 7);
    try std.testing.expect(info.recommendedMaxWorkingSetSize() > 0);
    try std.testing.expect(info.maxBufferLength() > 0);
}

test "embedded metallib exposes the smoke kernel" {
    const device = try Device.open();
    defer device.close();
    const library = try Library.openEmbedded(device);
    defer library.close();
    try std.testing.expect(try library.hasFunction("ami_smoke_iota"));
    try std.testing.expect(!try library.hasFunction("ami_no_such_kernel"));
}

test "a wired region is shared with the GPU without copying" {
    const device = try Device.open();
    defer device.close();
    const library = try Library.openEmbedded(device);
    defer library.close();
    const pipeline = try Pipeline.init(device, library, "ami_smoke_iota");
    defer pipeline.close();

    // Backing region and buffer outlive `compute` so residency is torn down
    // before the wired pages are unmapped.
    const threads = 4096;
    const region = try page.WiredRegion.init(threads * @sizeOf(u32));
    defer region.deinit();
    const buffer = try Buffer.wrap(device, region.bytes);
    defer buffer.close();
    try std.testing.expect(buffer.gpuAddress() != 0);

    const compute = try Compute.init(device);
    defer compute.close();
    try compute.makeResident(&.{buffer});

    try compute.dispatch1dSync(pipeline, &.{buffer}, threads, 5000);
    const words: []const u32 = @ptrCast(region.bytes);
    for (words, 0..) |word, i| try std.testing.expectEqual(@as(u32, @intCast(i)) ^ 0xA5A5A5A5, word);

    // The second dispatch reuses the same queue, command buffer and allocator.
    @memset(region.bytes, 0);
    try compute.dispatch1dSync(pipeline, &.{buffer}, threads / 2, 5000);
    for (words[0 .. threads / 2], 0..) |word, i| try std.testing.expectEqual(@as(u32, @intCast(i)) ^ 0xA5A5A5A5, word);
    for (words[threads / 2 ..]) |word| try std.testing.expectEqual(@as(u32, 0), word);
}

test "spans that break the 16 KiB rule never reach Metal" {
    const device = try Device.open();
    defer device.close();
    const region = try page.WiredRegion.init(2 * page.APPLE_PAGE_SIZE);
    defer region.deinit();
    try std.testing.expectError(error.ZeroLength, Buffer.wrap(device, region.bytes[0..0]));
    try std.testing.expectError(error.MisalignedLength, Buffer.wrap(device, region.bytes[0..100]));
    try std.testing.expectError(error.MisalignedAddress, Buffer.wrap(device, region.bytes[64 .. 64 + page.APPLE_PAGE_SIZE]));
}

test "a missing kernel is a pipeline error" {
    const device = try Device.open();
    defer device.close();
    const library = try Library.openEmbedded(device);
    defer library.close();
    try std.testing.expectError(error.PipelineCreationFailed, Pipeline.init(device, library, "ami_no_such_kernel"));
}

test "dispatch validates arguments and residency can be cleared" {
    const device = try Device.open();
    defer device.close();
    const library = try Library.openEmbedded(device);
    defer library.close();
    const pipeline = try Pipeline.init(device, library, "ami_smoke_iota");
    defer pipeline.close();
    const region = try page.WiredRegion.init(page.APPLE_PAGE_SIZE);
    defer region.deinit();
    const buffer = try Buffer.wrap(device, region.bytes);
    defer buffer.close();

    const compute = try Compute.init(device);
    defer compute.close();
    try compute.makeResident(&.{buffer});
    try std.testing.expectError(error.InvalidArgument, compute.dispatch1dSync(pipeline, &.{buffer}, 0, 1000));
    try compute.clearResidency();
}
