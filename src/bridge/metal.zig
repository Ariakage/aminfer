//! Zig view of the Metal half of the bridge (`apple_bridge.h`). Converts
//! status codes into an explicit error set and null-checks every handle the
//! bridge returns.

const std = @import("std");
const c = @import("apple_bridge");
const page = @import("../mem/page.zig");

/// The kernel library compiled from `src/kernels/*.metal` by `build.zig`.
const embedded_metallib = @embedFile("aminfer_metallib");

pub const MAX_BUFFER_BINDINGS: usize = c.AMI_MAX_BUFFER_BINDINGS;
pub const MAX_FRAMES_IN_FLIGHT: u32 = c.AMI_MAX_FRAMES_IN_FLIGHT;
pub const MAX_UNIFORM_BYTES: usize = c.AMI_UNIFORM_STRIDE;

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

/// Buffer plus byte offset for Metal 4 argument-table GPU-address binding.
pub const BufferBinding = struct {
    buffer: Buffer,
    offset: u64 = 0,
};

/// Startup Metal function-constant value for compile-time GPU specialization (ADR-0014).
pub const FnConstant = struct {
    index: u32,
    value: union(enum) {
        u32: u32,
        bool: bool,
    },
};

/// Compute pipeline built at startup through the Metal 4 compiler.
pub const Pipeline = struct {
    handle: *c.ami_pipeline,

    pub fn init(device: Device, library: Library, function_name: [:0]const u8) Error!Pipeline {
        return initSpecialized(device, library, function_name, &.{});
    }

    pub fn initSpecialized(
        device: Device,
        library: Library,
        function_name: [:0]const u8,
        constants: []const FnConstant,
    ) Error!Pipeline {
        const max_constants = 16;
        if (constants.len > max_constants) return error.InvalidArgument;
        var raw: [max_constants]c.ami_fn_constant = undefined;
        for (constants, 0..) |fc, i| {
            raw[i] = switch (fc.value) {
                .u32 => |v| .{ .index = fc.index, .type = c.AMI_CONST_U32, .value = v },
                .bool => |b| .{ .index = fc.index, .type = c.AMI_CONST_BOOL, .value = if (b) 1 else 0 },
            };
        }
        const ptr: ?[*]const c.ami_fn_constant = if (constants.len > 0) &raw else null;
        var handle: ?*c.ami_pipeline = null;
        try check(c.ami_pipeline_create_specialized(
            device.handle,
            library.handle,
            function_name.ptr,
            ptr,
            constants.len,
            &handle,
        ));
        return .{ .handle = handle orelse return error.NullHandle };
    }

    pub fn close(self: Pipeline) void {
        c.ami_pipeline_destroy(self.handle);
    }

    pub fn maxThreadsPerThreadgroup(self: Pipeline) u32 {
        return c.ami_pipeline_max_threads_per_threadgroup(self.handle);
    }
};

/// Reusable multi-frame Metal 4 compute submission context (ADR-0011).
pub const Compute = struct {
    handle: *c.ami_compute,

    /// An open command-buffer frame; call `commit` or `abort`.
    pub const Frame = struct {
        compute: *Compute,
        slot: u32,
        open: bool,

        /// Encodes a dispatch whose grid is specified in total threads.
        pub fn encodeThreads(
            self: *Frame,
            pipeline: Pipeline,
            bindings: []const BufferBinding,
            uniform_bytes: ?[]const u8,
            grid: [3]u64,
            threadgroup: [3]u64,
            insert_barrier_before: bool,
        ) Error!void {
            return self.encode(pipeline, bindings, uniform_bytes, grid, threadgroup, c.AMI_DISPATCH_THREADS, insert_barrier_before);
        }

        /// Encodes a dispatch whose grid is specified in threadgroups.
        pub fn encodeThreadgroups(
            self: *Frame,
            pipeline: Pipeline,
            bindings: []const BufferBinding,
            uniform_bytes: ?[]const u8,
            grid_groups: [3]u64,
            threadgroup: [3]u64,
            insert_barrier_before: bool,
        ) Error!void {
            return self.encode(pipeline, bindings, uniform_bytes, grid_groups, threadgroup, c.AMI_DISPATCH_THREADGROUPS, insert_barrier_before);
        }

        fn encode(
            self: *Frame,
            pipeline: Pipeline,
            bindings: []const BufferBinding,
            uniform_bytes: ?[]const u8,
            grid: [3]u64,
            threadgroup: [3]u64,
            mode: u32,
            insert_barrier_before: bool,
        ) Error!void {
            if (!self.open) return error.InvalidArgument;
            if (bindings.len > MAX_BUFFER_BINDINGS) return error.InvalidArgument;
            var raw: [MAX_BUFFER_BINDINGS]c.ami_buffer_binding = undefined;
            for (bindings, 0..) |b, i| {
                raw[i] = .{ .buffer = b.buffer.handle, .offset = b.offset };
            }
            const b_ptr: ?[*]const c.ami_buffer_binding = if (bindings.len > 0) &raw else null;
            const u_ptr: ?*const anyopaque = if (uniform_bytes) |u| u.ptr else null;
            const u_len: usize = if (uniform_bytes) |u| u.len else 0;
            try check(c.ami_compute_encode_dispatch(
                self.compute.handle,
                pipeline.handle,
                b_ptr,
                bindings.len,
                u_ptr,
                u_len,
                grid[0],
                grid[1],
                grid[2],
                threadgroup[0],
                threadgroup[1],
                threadgroup[2],
                mode,
                if (insert_barrier_before) 1 else 0,
            ));
        }

        /// Ends encoding, commits the command buffer and returns its completion signal.
        pub fn commit(self: *Frame) Error!u64 {
            if (!self.open) return error.InvalidArgument;
            var signal: u64 = 0;
            self.open = false;
            try check(c.ami_compute_commit_frame(self.compute.handle, &signal));
            return signal;
        }

        /// Aborts an uncommitted frame and resets its allocator. Safe to call in `errdefer`.
        pub fn abort(self: *Frame) void {
            if (!self.open) return;
            self.open = false;
            c.ami_compute_abort_frame(self.compute.handle);
        }
    };

    pub fn init(device: Device) Error!Compute {
        return initWithFrames(device, 2);
    }

    pub fn initWithFrames(device: Device, frames_in_flight: u32) Error!Compute {
        var handle: ?*c.ami_compute = null;
        try check(c.ami_compute_create_with_frames(device.handle, frames_in_flight, &handle));
        return .{ .handle = handle orelse return error.NullHandle };
    }

    pub fn close(self: Compute) void {
        c.ami_compute_destroy(self.handle);
    }

    /// Adds buffers to this context's user residency set and commits residency.
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

    /// Removes buffers from this context's user residency set and commits.
    pub fn removeResident(self: Compute, buffers: []const Buffer) Error!void {
        const batch_cap = 64;
        var offset: usize = 0;
        while (offset < buffers.len) {
            const count = @min(batch_cap, buffers.len - offset);
            var handles: [batch_cap]?*c.ami_buffer = undefined;
            for (buffers[offset .. offset + count], 0..) |buf, i| {
                handles[i] = buf.handle;
            }
            try check(c.ami_compute_remove_resident(self.handle, &handles, count));
            offset += count;
        }
    }

    /// Removes all caller allocations from this context's user residency set.
    pub fn clearResidency(self: Compute) Error!void {
        try check(c.ami_compute_clear_residency(self.handle));
    }

    /// Acquires the next reusable frame slot, waiting only if that slot's prior
    /// GPU work has not yet finished.
    pub fn beginFrame(self: *Compute, timeout_ms: u64) Error!Frame {
        var slot: u32 = 0;
        try check(c.ami_compute_begin_frame(self.handle, timeout_ms, &slot));
        return .{ .compute = self, .slot = slot, .open = true };
    }

    /// Waits on the shared event until `signal` has been reached by the GPU.
    pub fn wait(self: Compute, signal: u64, timeout_ms: u64) Error!void {
        try check(c.ami_compute_wait(self.handle, signal, timeout_ms));
    }

    /// Encodes a single 1D dispatch on the reusable command buffer and waits for
    /// shared-event completion. `buffers[i]` binds to `[[buffer(i)]]`.
    pub fn dispatch1dSync(
        self: Compute,
        pipeline: Pipeline,
        buffers: []const Buffer,
        threads: u64,
        timeout_ms: u64,
    ) Error!void {
        if (threads == 0 or buffers.len > MAX_BUFFER_BINDINGS) {
            return error.InvalidArgument;
        }
        var handles: [MAX_BUFFER_BINDINGS]?*c.ami_buffer = undefined;
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

test "specialized pipeline, intra-encoder barrier, uniforms and multi-frame ring work end-to-end" {
    const device = try Device.open();
    defer device.close();
    const library = try Library.openEmbedded(device);
    defer library.close();

    const iota_pipe = try Pipeline.init(device, library, "ami_smoke_iota");
    defer iota_pipe.close();
    const scale_pipe = try Pipeline.initSpecialized(
        device,
        library,
        "ami_smoke_scale_add",
        &.{.{ .index = 0, .value = .{ .u32 = 3 } }},
    );
    defer scale_pipe.close();

    const threads = 1024;
    const region = try page.WiredRegion.init(2 * page.APPLE_PAGE_SIZE);
    defer region.deinit();
    const buffer = try Buffer.wrap(device, region.bytes);
    defer buffer.close();

    // Two frames in flight; run 6 frames to wrap the allocator ring multiple times.
    var compute = try Compute.initWithFrames(device, 2);
    defer compute.close();
    try compute.makeResident(&.{buffer});

    const SmokeParams = extern struct { addend: u32 };
    var last_sig: u64 = 0;

    for (0..6) |round| {
        const offset: u64 = (round % 2) * page.APPLE_PAGE_SIZE;
        var frame = try compute.beginFrame(5000);
        errdefer frame.abort();
        try frame.encodeThreads(
            iota_pipe,
            &.{.{ .buffer = buffer, .offset = offset }},
            null,
            .{ threads, 1, 1 },
            .{ 256, 1, 1 },
            false,
        );
        const params: SmokeParams = .{ .addend = @intCast(round + 7) };
        try frame.encodeThreads(
            scale_pipe,
            &.{.{ .buffer = buffer, .offset = offset }},
            std.mem.asBytes(&params),
            .{ threads, 1, 1 },
            .{ 256, 1, 1 },
            true,
        );
        last_sig = try frame.commit();
    }

    try compute.wait(last_sig, 5000);

    // Round 4 wrote page 0 (addend = 11); round 5 wrote page 1 (addend = 12).
    const page0: []const u32 = @ptrCast(region.bytes[0..page.APPLE_PAGE_SIZE]);
    const page1: []const u32 = @ptrCast(region.bytes[page.APPLE_PAGE_SIZE..]);
    for (0..threads) |i| {
        const base: u32 = @as(u32, @intCast(i)) ^ 0xA5A5A5A5;
        try std.testing.expectEqual(base *% 3 +% 11, page0[i]);
        try std.testing.expectEqual(base *% 3 +% 12, page1[i]);
    }

    try compute.removeResident(&.{buffer});
}
