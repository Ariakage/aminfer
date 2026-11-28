//! Zig view of the Metal half of the bridge (`apple_bridge.h`). Converts
//! status codes into an explicit error set and null-checks every handle the
//! bridge returns.

const std = @import("std");
const c = @import("apple_bridge");
const page = @import("../mem/page.zig");
const paths = @import("../platform/paths.zig");

/// The kernel library compiled from `src/kernels/*.metal` by `build.zig`.
const embedded_metallib = @embedFile("aminfer_metallib");

pub const MAX_BUFFER_BINDINGS: usize = c.AMI_MAX_BUFFER_BINDINGS;
pub const MAX_TEXTURE_BINDINGS: usize = c.AMI_MAX_TEXTURE_BINDINGS;
pub const MAX_TEXTURE_DIM: u32 = c.AMI_MAX_TEXTURE_DIM;
pub const KV_TILE_ROW_ALIGN_F16: u32 = c.AMI_KV_TILE_ROW_ALIGN_F16;
pub const MAX_FRAMES_IN_FLIGHT: u32 = c.AMI_MAX_FRAMES_IN_FLIGHT;
pub const MAX_UNIFORM_BYTES: usize = c.AMI_UNIFORM_STRIDE;
pub const UNIFORM_SLOTS_PER_FRAME: u32 = c.AMI_UNIFORM_SLOTS_PER_FRAME;
pub const ANE_MAX_DIMS: usize = c.AMI_ANE_MAX_DIMS;
pub const ANE_MAX_BINDINGS: usize = c.AMI_ANE_MAX_BINDINGS;

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
    TileCreationFailed,
    AneFailed,
    VisionFailed,
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
        c.AMI_ERR_TILE => error.TileCreationFailed,
        c.AMI_ERR_ANE => error.AneFailed,
        c.AMI_ERR_VISION => error.VisionFailed,
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
    pub fn appleGpuFamily(self: DeviceInfo) u8 {
        if (self.raw.gpu_apple_family != 0) return self.raw.gpu_apple_family;
        return @intCast(@min(self.raw.apple_gpu_family, 255));
    }
    pub fn aneCoreCount(self: DeviceInfo) u32 {
        return self.raw.ane_core_count;
    }
    pub fn supportsApple9(self: DeviceInfo) bool {
        return self.raw.supports_apple9 != 0 or self.appleGpuFamily() >= 9;
    }
    pub fn supportsApple10(self: DeviceInfo) bool {
        return self.raw.supports_apple10 != 0 or self.appleGpuFamily() >= 10;
    }
    pub fn supportsApple11(self: DeviceInfo) bool {
        return self.raw.supports_apple11 != 0 or self.appleGpuFamily() >= 11;
    }
    /// Returns true on Apple10+ GPUs (M5+) that provide GPU Neural Accelerator
    /// (`NAX`) hardware backing Metal 4 `mpp::tensor_ops::matmul2d` (`ADR-0021`).
    pub fn supportsNaxTensorOps(self: DeviceInfo) bool {
        return self.appleGpuFamily() >= 10 or self.raw.supports_apple10 != 0;
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

/// Matches Metal's `MTLDispatchThreadgroupsIndirectArguments` (`[3]u32`, 12 bytes, 4-byte aligned).
pub const IndirectDispatchArgs = extern struct {
    threadgroups_per_grid: [3]u32,

    pub const ALIGN: usize = 4;
    pub const SIZE: usize = 12;

    pub inline fn init(x: u32, y: u32, z: u32) IndirectDispatchArgs {
        return .{ .threadgroups_per_grid = .{ x, y, z } };
    }

    comptime {
        if (@sizeOf(IndirectDispatchArgs) != 12 or @alignOf(IndirectDispatchArgs) != 4) {
            @compileError("IndirectDispatchArgs must match MTLDispatchThreadgroupsIndirectArguments (12 bytes, align 4)");
        }
        if (@sizeOf(c.ami_indirect_dispatch_args) != @sizeOf(IndirectDispatchArgs)) {
            @compileError("ami_indirect_dispatch_args ABI mismatch");
        }
    }
};

/// Metal texture view format over an IOSurface-backed fp16 KV tile (ADR-0004).
pub const TextureView = enum {
    /// RGBA16Float view (`width = row_f16 / 4`, default for decode attention).
    rgba16f,
    /// R16Float view (`width = row_f16`, scalar fp16 view).
    r16f,

    fn toRaw(self: TextureView) u32 {
        return switch (self) {
            .rgba16f => c.AMI_TEX_VIEW_RGBA16F,
            .r16f => c.AMI_TEX_VIEW_R16F,
        };
    }
};

/// Texture view binding for Metal 4 argument-table `[[texture(i)]]` slots.
pub const TextureBinding = struct {
    tile: KvTile,
    view: TextureView = .rgba16f,
};

/// Optional features when creating an IOSurface-backed `KvTile`.
pub const KvTileOptions = struct {
    /// Also create a zero-copy `MTLBuffer` alias over the 16 KiB-aligned pages.
    alias_buffer: bool = false,
    /// Attempt to `mlock` the IOSurface mapped pages.
    mlock: bool = false,
};

pub const KvTileInfo = struct {
    raw: c.ami_kv_tile_info,

    pub fn rowF16(self: KvTileInfo) u32 {
        return self.raw.row_f16;
    }
    pub fn rows(self: KvTileInfo) u32 {
        return self.raw.rows;
    }
    pub fn bytesPerRow(self: KvTileInfo) u32 {
        return self.raw.bytes_per_row;
    }
    pub fn hasAliasBuffer(self: KvTileInfo) bool {
        return self.raw.has_alias_buffer != 0;
    }
    pub fn isMlocked(self: KvTileInfo) bool {
        return self.raw.is_mlocked != 0;
    }
    pub fn dataBytes(self: KvTileInfo) u64 {
        return self.raw.data_bytes;
    }
    pub fn allocBytes(self: KvTileInfo) u64 {
        return self.raw.alloc_bytes;
    }
    pub fn surfaceAllocSize(self: KvTileInfo) u64 {
        return self.raw.surface_alloc_size;
    }
    pub fn rgbaResourceId(self: KvTileInfo) u64 {
        return self.raw.rgba_resource_id;
    }
    pub fn r16ResourceId(self: KvTileInfo) u64 {
        return self.raw.r16_resource_id;
    }
    pub fn aliasGpuAddress(self: KvTileInfo) u64 {
        return self.raw.alias_gpu_address;
    }
};

/// IOSurface-backed fp16 KV cache tile shared zero-copy across CoreML/ANE
/// (`CVPixelBuffer` `OneComponent16Half`), Metal `RGBA16Float` / `R16Float`
/// textures, and an optional zero-copy `MTLBuffer` alias (ADR-0004).
pub const KvTile = struct {
    handle: *c.ami_kv_tile,

    /// Scoped CPU lock over the tile's IOSurface pages (tests/inspection only).
    pub const LockGuard = struct {
        tile: KvTile,
        read_only: bool,
        locked: bool,
        bytes: []align(page.APPLE_PAGE_SIZE) u8,
        bytes_per_row: usize,

        pub fn f16Slice(self: LockGuard) []f16 {
            return @alignCast(std.mem.bytesAsSlice(f16, self.bytes));
        }

        pub fn unlock(self: *LockGuard) Error!void {
            if (!self.locked) return error.InvalidArgument;
            self.locked = false;
            try check(c.ami_kv_tile_unlock(self.tile.handle, if (self.read_only) 1 else 0));
        }

        /// Safe cleanup for `errdefer` when an error occurs before `unlock`.
        pub fn release(self: *LockGuard) void {
            if (!self.locked) return;
            self.locked = false;
            const st = c.ami_kv_tile_unlock(self.tile.handle, if (self.read_only) 1 else 0);
            std.debug.assert(st == c.AMI_OK);
        }
    };

    pub fn init(device: Device, row_f16: u32, rows: u32, options: KvTileOptions) Error!KvTile {
        if (row_f16 == 0 or rows == 0 or
            (row_f16 % KV_TILE_ROW_ALIGN_F16) != 0 or
            row_f16 > MAX_TEXTURE_DIM or
            rows > MAX_TEXTURE_DIM)
        {
            return error.InvalidArgument;
        }
        var flags: u32 = c.AMI_KV_TILE_FLAG_NONE;
        if (options.alias_buffer) flags |= c.AMI_KV_TILE_FLAG_ALIAS_BUFFER;
        if (options.mlock) flags |= c.AMI_KV_TILE_FLAG_MLOCK;

        var handle: ?*c.ami_kv_tile = null;
        try check(c.ami_kv_tile_create(device.handle, row_f16, rows, flags, &handle));
        return .{ .handle = handle orelse return error.NullHandle };
    }

    pub fn close(self: KvTile) void {
        c.ami_kv_tile_destroy(self.handle);
    }

    pub fn info(self: KvTile) Error!KvTileInfo {
        var raw: c.ami_kv_tile_info = undefined;
        try check(c.ami_kv_tile_query(self.handle, &raw));
        return .{ .raw = raw };
    }

    pub fn rgbaResourceId(self: KvTile) u64 {
        return c.ami_kv_tile_rgba_resource_id(self.handle);
    }

    pub fn r16ResourceId(self: KvTile) u64 {
        return c.ami_kv_tile_r16_resource_id(self.handle);
    }

    /// Returns a borrowed `Buffer` view over the IOSurface allocation when
    /// created with `options.alias_buffer = true`. Do not call `close()` on it;
    /// the owning `KvTile` releases it on `KvTile.close()`.
    pub fn aliasBuffer(self: KvTile) ?Buffer {
        const ptr = c.ami_kv_tile_alias_buffer(self.handle) orelse return null;
        return .{ .handle = ptr };
    }

    /// Locks the IOSurface for CPU access (test initialization and verification).
    pub fn lock(self: KvTile, read_only: bool) Error!LockGuard {
        var base_ptr: ?*anyopaque = null;
        var bpr: usize = 0;
        var data_bytes: usize = 0;
        try check(c.ami_kv_tile_lock(
            self.handle,
            if (read_only) 1 else 0,
            &base_ptr,
            &bpr,
            &data_bytes,
        ));
        const raw_ptr = base_ptr orelse return error.NullHandle;
        if (!std.mem.isAligned(@intFromPtr(raw_ptr), page.APPLE_PAGE_SIZE)) {
            _ = c.ami_kv_tile_unlock(self.handle, if (read_only) 1 else 0);
            return error.TileCreationFailed;
        }
        const aligned_ptr: [*]align(page.APPLE_PAGE_SIZE) u8 = @ptrCast(@alignCast(raw_ptr));
        return .{
            .tile = self,
            .read_only = read_only,
            .locked = true,
            .bytes = aligned_ptr[0..data_bytes],
            .bytes_per_row = bpr,
        };
    }
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
            return self.encodeEx(pipeline, bindings, &.{}, uniform_bytes, grid, threadgroup, c.AMI_DISPATCH_THREADS, insert_barrier_before);
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
            return self.encodeEx(pipeline, bindings, &.{}, uniform_bytes, grid_groups, threadgroup, c.AMI_DISPATCH_THREADGROUPS, insert_barrier_before);
        }

        /// Encodes a GPU-driven indirect threadgroup dispatch (`G20`, ADR-0023).
        /// `indirect.offset` must be 4-byte aligned with at least 12 bytes
        /// (`IndirectDispatchArgs`) remaining inside the resident `indirect.buffer`.
        pub fn encodeThreadgroupsIndirect(
            self: *Frame,
            pipeline: Pipeline,
            bindings: []const BufferBinding,
            uniform_bytes: ?[]const u8,
            indirect: BufferBinding,
            threadgroup: [3]u64,
            insert_barrier_before: bool,
        ) Error!void {
            if (!self.open) return error.InvalidArgument;
            return self.compute.dispatchIndirect(
                pipeline,
                bindings,
                uniform_bytes,
                indirect,
                threadgroup,
                insert_barrier_before,
            );
        }

        /// Encodes a threads-grid dispatch with explicit texture bindings.
        pub fn encodeThreadsWithTextures(
            self: *Frame,
            pipeline: Pipeline,
            bindings: []const BufferBinding,
            textures: []const TextureBinding,
            uniform_bytes: ?[]const u8,
            grid: [3]u64,
            threadgroup: [3]u64,
            insert_barrier_before: bool,
        ) Error!void {
            return self.encodeEx(pipeline, bindings, textures, uniform_bytes, grid, threadgroup, c.AMI_DISPATCH_THREADS, insert_barrier_before);
        }

        /// Encodes a threadgroups-grid dispatch with explicit texture bindings.
        pub fn encodeThreadgroupsWithTextures(
            self: *Frame,
            pipeline: Pipeline,
            bindings: []const BufferBinding,
            textures: []const TextureBinding,
            uniform_bytes: ?[]const u8,
            grid_groups: [3]u64,
            threadgroup: [3]u64,
            insert_barrier_before: bool,
        ) Error!void {
            return self.encodeEx(pipeline, bindings, textures, uniform_bytes, grid_groups, threadgroup, c.AMI_DISPATCH_THREADGROUPS, insert_barrier_before);
        }

        /// Encodes a threads-grid dispatch with explicit zero-copy vision plane texture bindings.
        pub fn encodeThreadsWithVisionTextures(
            self: *Frame,
            pipeline: Pipeline,
            bindings: []const BufferBinding,
            vision_textures: []const VisionTextureBinding,
            uniform_bytes: ?[]const u8,
            grid: [3]u64,
            threadgroup: [3]u64,
            insert_barrier_before: bool,
        ) Error!void {
            return self.encodeVisionEx(pipeline, bindings, vision_textures, uniform_bytes, grid, threadgroup, c.AMI_DISPATCH_THREADS, insert_barrier_before);
        }

        /// Encodes a threadgroups-grid dispatch with explicit zero-copy vision plane texture bindings.
        pub fn encodeThreadgroupsWithVisionTextures(
            self: *Frame,
            pipeline: Pipeline,
            bindings: []const BufferBinding,
            vision_textures: []const VisionTextureBinding,
            uniform_bytes: ?[]const u8,
            grid_groups: [3]u64,
            threadgroup: [3]u64,
            insert_barrier_before: bool,
        ) Error!void {
            return self.encodeVisionEx(pipeline, bindings, vision_textures, uniform_bytes, grid_groups, threadgroup, c.AMI_DISPATCH_THREADGROUPS, insert_barrier_before);
        }

        fn encodeVisionEx(
            self: *Frame,
            pipeline: Pipeline,
            bindings: []const BufferBinding,
            vision_textures: []const VisionTextureBinding,
            uniform_bytes: ?[]const u8,
            grid: [3]u64,
            threadgroup: [3]u64,
            mode: u32,
            insert_barrier_before: bool,
        ) Error!void {
            if (!self.open) return error.InvalidArgument;
            if (bindings.len > MAX_BUFFER_BINDINGS or vision_textures.len > MAX_TEXTURE_BINDINGS) {
                return error.InvalidArgument;
            }
            var raw_bufs: [MAX_BUFFER_BINDINGS]c.ami_buffer_binding = undefined;
            for (bindings, 0..) |b, i| {
                raw_bufs[i] = .{ .buffer = b.buffer.handle, .offset = b.offset };
            }
            var raw_texs: [MAX_TEXTURE_BINDINGS]c.ami_vision_texture_binding = undefined;
            for (vision_textures, 0..) |t, i| {
                raw_texs[i] = .{ .frame = t.frame.handle, .plane = t.plane, .reserved = 0 };
            }
            const b_ptr: ?[*]const c.ami_buffer_binding = if (bindings.len > 0) &raw_bufs else null;
            const t_ptr: ?[*]const c.ami_vision_texture_binding = if (vision_textures.len > 0) &raw_texs else null;
            const u_ptr: ?*const anyopaque = if (uniform_bytes) |u| u.ptr else null;
            const u_len: usize = if (uniform_bytes) |u| u.len else 0;
            try check(c.ami_compute_encode_dispatch_vision(
                self.compute.handle,
                pipeline.handle,
                b_ptr,
                bindings.len,
                t_ptr,
                vision_textures.len,
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

        fn encodeEx(
            self: *Frame,
            pipeline: Pipeline,
            bindings: []const BufferBinding,
            textures: []const TextureBinding,
            uniform_bytes: ?[]const u8,
            grid: [3]u64,
            threadgroup: [3]u64,
            mode: u32,
            insert_barrier_before: bool,
        ) Error!void {
            if (!self.open) return error.InvalidArgument;
            if (bindings.len > MAX_BUFFER_BINDINGS or textures.len > MAX_TEXTURE_BINDINGS) {
                return error.InvalidArgument;
            }
            var raw_bufs: [MAX_BUFFER_BINDINGS]c.ami_buffer_binding = undefined;
            for (bindings, 0..) |b, i| {
                raw_bufs[i] = .{ .buffer = b.buffer.handle, .offset = b.offset };
            }
            var raw_texs: [MAX_TEXTURE_BINDINGS]c.ami_texture_binding = undefined;
            for (textures, 0..) |t, i| {
                raw_texs[i] = .{ .tile = t.tile.handle, .view = t.view.toRaw(), .reserved = 0 };
            }
            const b_ptr: ?[*]const c.ami_buffer_binding = if (bindings.len > 0) &raw_bufs else null;
            const t_ptr: ?[*]const c.ami_texture_binding = if (textures.len > 0) &raw_texs else null;
            const u_ptr: ?*const anyopaque = if (uniform_bytes) |u| u.ptr else null;
            const u_len: usize = if (uniform_bytes) |u| u.len else 0;
            try check(c.ami_compute_encode_dispatch_ex(
                self.compute.handle,
                pipeline.handle,
                b_ptr,
                bindings.len,
                t_ptr,
                textures.len,
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

    /// Adds KV tiles' zero-copy textures (and alias buffers if present) to the
    /// user residency set and commits residency. Zero heap allocation.
    pub fn makeTilesResident(self: Compute, tiles: []const KvTile) Error!void {
        const batch_cap = 64;
        var offset: usize = 0;
        while (offset < tiles.len) {
            const count = @min(batch_cap, tiles.len - offset);
            var handles: [batch_cap]?*c.ami_kv_tile = undefined;
            for (tiles[offset .. offset + count], 0..) |tile, i| {
                handles[i] = tile.handle;
            }
            try check(c.ami_compute_make_tiles_resident(self.handle, &handles, count));
            offset += count;
        }
    }

    /// Removes KV tiles' allocations from the user residency set and commits.
    pub fn removeTilesResident(self: Compute, tiles: []const KvTile) Error!void {
        const batch_cap = 64;
        var offset: usize = 0;
        while (offset < tiles.len) {
            const count = @min(batch_cap, tiles.len - offset);
            var handles: [batch_cap]?*c.ami_kv_tile = undefined;
            for (tiles[offset .. offset + count], 0..) |tile, i| {
                handles[i] = tile.handle;
            }
            try check(c.ami_compute_remove_tiles_resident(self.handle, &handles, count));
            offset += count;
        }
    }

    /// Adds zero-copy vision frames' plane textures to the user residency set
    /// and commits residency. Zero heap allocation.
    pub fn makeVisionFramesResident(self: Compute, frames: []const VisionFrame) Error!void {
        const batch_cap = 64;
        var offset: usize = 0;
        while (offset < frames.len) {
            const count = @min(batch_cap, frames.len - offset);
            var handles: [batch_cap]?*c.ami_vision_frame = undefined;
            for (frames[offset .. offset + count], 0..) |frame, i| {
                handles[i] = frame.handle;
            }
            try check(c.ami_compute_make_vision_frames_resident(self.handle, &handles, count));
            offset += count;
        }
    }

    /// Removes zero-copy vision frames' plane textures from the user residency set.
    pub fn removeVisionFramesResident(self: Compute, frames: []const VisionFrame) Error!void {
        const batch_cap = 64;
        var offset: usize = 0;
        while (offset < frames.len) {
            const count = @min(batch_cap, frames.len - offset);
            var handles: [batch_cap]?*c.ami_vision_frame = undefined;
            for (frames[offset .. offset + count], 0..) |frame, i| {
                handles[i] = frame.handle;
            }
            try check(c.ami_compute_remove_vision_frames_resident(self.handle, &handles, count));
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

    /// Waits with bounded retries until all committed frames complete or `timeout_ms` elapses.
    pub fn waitIdle(self: Compute, timeout_ms: u64) Error!void {
        try check(c.ami_compute_wait_idle(self.handle, timeout_ms));
    }

    /// Best-effort bounded wait for any in-flight GPU work before tearing down
    /// residency, NoCopy wired arenas, or IOSurface vision frames after a timeout.
    pub fn waitIdleBestEffort(self: Compute) void {
        c.ami_compute_wait_idle_best_effort(self.handle);
    }

    /// Encodes a GPU-driven indirect threadgroup dispatch into the active Metal 4
    /// frame (`G20`, ADR-0023). Zero dynamic heap allocation; validates 4-byte
    /// indirect-buffer alignment, binding limits, and buffer bounds.
    pub fn dispatchIndirect(
        self: *Compute,
        pipeline: Pipeline,
        bindings: []const BufferBinding,
        uniform_bytes: ?[]const u8,
        indirect: BufferBinding,
        threadgroup: [3]u64,
        insert_barrier_before: bool,
    ) Error!void {
        if ((indirect.offset & (IndirectDispatchArgs.ALIGN - 1)) != 0) {
            return error.InvalidArgument;
        }
        if (threadgroup[0] == 0 or threadgroup[1] == 0 or threadgroup[2] == 0) {
            return error.InvalidArgument;
        }
        const extra: usize = if (uniform_bytes != null and uniform_bytes.?.len > 0) 1 else 0;
        if (bindings.len + extra > MAX_BUFFER_BINDINGS) {
            return error.InvalidArgument;
        }
        if (uniform_bytes) |u| {
            if (u.len > MAX_UNIFORM_BYTES) return error.InvalidArgument;
        }
        var raw_bufs: [MAX_BUFFER_BINDINGS]c.ami_buffer_binding = undefined;
        for (bindings, 0..) |b, i| {
            raw_bufs[i] = .{ .buffer = b.buffer.handle, .offset = b.offset };
        }
        const b_ptr: ?[*]const c.ami_buffer_binding = if (bindings.len > 0) &raw_bufs else null;
        const u_ptr: ?*const anyopaque = if (uniform_bytes) |u| u.ptr else null;
        const u_len: usize = if (uniform_bytes) |u| u.len else 0;
        try check(c.ami_mtl4_compute_dispatch_indirect(
            self.handle,
            pipeline.handle,
            b_ptr,
            bindings.len,
            u_ptr,
            u_len,
            indirect.buffer.handle,
            indirect.offset,
            threadgroup[0],
            threadgroup[1],
            threadgroup[2],
            if (insert_barrier_before) 1 else 0,
        ));
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

// ---------------------------------------------------------------------------
// CoreML / Apple Neural Engine zero-copy IOSurface bridge (ADR-0004, D5, S4)
// ---------------------------------------------------------------------------

/// CoreML `MLComputeUnits` selection for `AneModel` loading.
pub const AneComputeUnits = enum {
    cpu_and_neural_engine,
    all,
    cpu_only,
    cpu_and_gpu,

    fn toRaw(self: AneComputeUnits) u32 {
        return switch (self) {
            .cpu_and_neural_engine => c.AMI_ANE_COMPUTE_CPU_AND_NE,
            .all => c.AMI_ANE_COMPUTE_ALL,
            .cpu_only => c.AMI_ANE_COMPUTE_CPU_ONLY,
            .cpu_and_gpu => c.AMI_ANE_COMPUTE_CPU_AND_GPU,
        };
    }
};

/// `MLComputePlan` placement summary for a loaded CoreML model.
pub const AneModelInfo = struct {
    raw: c.ami_ane_model_info,

    pub fn totalLayers(self: AneModelInfo) u32 {
        return self.raw.total_layers;
    }
    pub fn anePreferredLayers(self: AneModelInfo) u32 {
        return self.raw.ane_preferred_layers;
    }
    pub fn gpuPreferredLayers(self: AneModelInfo) u32 {
        return self.raw.gpu_preferred_layers;
    }
    pub fn cpuPreferredLayers(self: AneModelInfo) u32 {
        return self.raw.cpu_preferred_layers;
    }
    pub fn usesNeuralEngine(self: AneModelInfo) bool {
        return self.raw.ane_preferred_layers > 0;
    }
};

/// Result metadata from one CoreML prediction call.
pub const AnePredictInfo = struct {
    raw: c.ami_ane_predict_info,

    pub fn outputsZeroCopyMatched(self: AnePredictInfo) u32 {
        return self.raw.outputs_zero_copy_matched;
    }
    pub fn outputsTotal(self: AnePredictInfo) u32 {
        return self.raw.outputs_total;
    }
    pub fn allOutputsZeroCopy(self: AnePredictInfo) bool {
        return self.raw.outputs_total > 0 and
            self.raw.outputs_zero_copy_matched == self.raw.outputs_total;
    }
};

/// Zero-copy `MLMultiArray` view over an IOSurface-backed `KvTile`
/// (`kCVPixelFormatType_OneComponent16Half`, ADR-0004).
pub const AneTensor = struct {
    handle: *c.ami_ane_tensor,

    /// Wraps `tile`'s IOSurface `CVPixelBuffer` with `shape` (2..5 dims).
    /// Requires `shape[shape.len - 1] == tile.rowF16` and
    /// `product(shape[0..shape.len - 1]) == tile.rows`.
    pub fn fromKvTile(tile: KvTile, shape: []const u32) Error!AneTensor {
        if (shape.len < 2 or shape.len > ANE_MAX_DIMS) {
            return error.InvalidArgument;
        }
        var handle: ?*c.ami_ane_tensor = null;
        try check(c.ami_ane_tensor_from_kv_tile(tile.handle, shape.ptr, shape.len, &handle));
        return .{ .handle = handle orelse return error.NullHandle };
    }

    /// Wraps an `F16` (`OneComponent16Half`) `VisionFrame`'s IOSurface
    /// `CVPixelBuffer` as a zero-copy fp16 `MLMultiArray` (`ADR-0004`, `G14`).
    pub fn fromVisionFrame(frame: VisionFrame, shape: []const u32) Error!AneTensor {
        if (shape.len < 2 or shape.len > ANE_MAX_DIMS) {
            return error.InvalidArgument;
        }
        var handle: ?*c.ami_ane_tensor = null;
        try check(c.ami_ane_tensor_from_vision_frame(frame.handle, shape.ptr, shape.len, &handle));
        return .{ .handle = handle orelse return error.NullHandle };
    }

    pub fn close(self: AneTensor) void {
        c.ami_ane_tensor_destroy(self.handle);
    }
};

/// Named tensor binding for CoreML inputs or zero-copy `outputBackings`.
pub const AneBinding = struct {
    name: [:0]const u8,
    tensor: AneTensor,
};

/// Preallocated CoreML input provider + `MLPredictionOptions.outputBackings`
/// built once at startup so hot-path ANE predictions do not rebuild Foundation
/// dictionaries on every call.
pub const AnePlan = struct {
    handle: *c.ami_ane_plan,

    pub fn init(inputs: []const AneBinding, outputs: []const AneBinding) Error!AnePlan {
        if (inputs.len == 0 or inputs.len > ANE_MAX_BINDINGS or
            outputs.len == 0 or outputs.len > ANE_MAX_BINDINGS)
        {
            return error.InvalidArgument;
        }
        var raw_in: [ANE_MAX_BINDINGS]c.ami_ane_binding = undefined;
        for (inputs, 0..) |b, i| {
            raw_in[i] = .{ .name = b.name.ptr, .tensor = b.tensor.handle };
        }
        var raw_out: [ANE_MAX_BINDINGS]c.ami_ane_binding = undefined;
        for (outputs, 0..) |b, i| {
            raw_out[i] = .{ .name = b.name.ptr, .tensor = b.tensor.handle };
        }
        var handle: ?*c.ami_ane_plan = null;
        try check(c.ami_ane_plan_create(&raw_in, inputs.len, &raw_out, outputs.len, &handle));
        return .{ .handle = handle orelse return error.NullHandle };
    }

    pub fn close(self: AnePlan) void {
        c.ami_ane_plan_destroy(self.handle);
    }
};

/// Compiled and loaded CoreML model targeting the Apple Neural Engine (or CPU/GPU).
pub const AneModel = struct {
    handle: *c.ami_ane_model,

    pub const IoCounts = struct {
        inputs: u32,
        outputs: u32,
    };

    /// Compiles (`.mlmodel` / `.mlpackage`) and loads, or directly loads
    /// (`.mlmodelc`), a CoreML model from disk. Startup only.
    pub fn loadPath(path: [:0]const u8, units: AneComputeUnits) Error!AneModel {
        return loadPathWithFunction(path, null, units);
    }

    /// Like `loadPath`, optionally selecting a named function from a
    /// multi-function CoreML `.mlpackage` / `.mlmodelc` bundle. Startup only.
    pub fn loadPathWithFunction(
        path: [:0]const u8,
        function_name: ?[:0]const u8,
        units: AneComputeUnits,
    ) Error!AneModel {
        if (path.len == 0) return error.InvalidArgument;
        const fn_ptr: ?[*:0]const u8 = if (function_name) |f| f.ptr else null;
        var handle: ?*c.ami_ane_model = null;
        try check(c.ami_ane_model_load_path_ex(path.ptr, fn_ptr, units.toRaw(), &handle));
        return .{ .handle = handle orelse return error.NullHandle };
    }

    /// Compiles and loads a CoreML `.mlmodel` protobuf directly from in-memory
    /// bytes via `MLModelAsset` (zero temporary disk files). Startup only.
    pub fn loadSpecData(spec_bytes: []const u8, units: AneComputeUnits) Error!AneModel {
        if (spec_bytes.len == 0) return error.InvalidArgument;
        var handle: ?*c.ami_ane_model = null;
        try check(c.ami_ane_model_load_spec_data(spec_bytes.ptr, spec_bytes.len, units.toRaw(), &handle));
        return .{ .handle = handle orelse return error.NullHandle };
    }

    pub fn close(self: AneModel) void {
        c.ami_ane_model_destroy(self.handle);
    }

    pub fn info(self: AneModel) Error!AneModelInfo {
        var raw: c.ami_ane_model_info = undefined;
        try check(c.ami_ane_model_query(self.handle, &raw));
        return .{ .raw = raw };
    }

    /// Returns the number of declared input and output features on this model.
    pub fn ioCounts(self: AneModel) Error!IoCounts {
        var ins: u32 = 0;
        var outs: u32 = 0;
        try check(c.ami_ane_model_io_counts(self.handle, &ins, &outs));
        return .{ .inputs = ins, .outputs = outs };
    }

    /// Copies the `index`-th lexicographically sorted input (`is_output == false`)
    /// or output (`is_output == true`) feature name into `buf` and returns it as
    /// a NUL-terminated slice.
    pub fn ioName(self: AneModel, is_output: bool, index: u32, buf: []u8) Error![:0]const u8 {
        if (buf.len < 2) return error.InvalidArgument;
        try check(c.ami_ane_model_io_name(
            self.handle,
            if (is_output) 1 else 0,
            index,
            buf.ptr,
            buf.len,
        ));
        const slice = std.mem.sliceTo(buf, 0);
        if (slice.len >= buf.len) return error.AneFailed;
        return buf[0..slice.len :0];
    }

    /// Executes synchronous prediction using a preallocated `AnePlan` with
    /// zero-copy IOSurface input tensors and `outputBackings`.
    pub fn predict(self: AneModel, plan: AnePlan) Error!AnePredictInfo {
        var raw: c.ami_ane_predict_info = undefined;
        try check(c.ami_ane_model_predict(self.handle, plan.handle, &raw));
        return .{ .raw = raw };
    }

    /// One-shot prediction convenience helper (builds a transient `AnePlan`).
    pub fn predictOneShot(
        self: AneModel,
        inputs: []const AneBinding,
        outputs: []const AneBinding,
    ) Error!AnePredictInfo {
        if (inputs.len == 0 or inputs.len > ANE_MAX_BINDINGS or
            outputs.len == 0 or outputs.len > ANE_MAX_BINDINGS)
        {
            return error.InvalidArgument;
        }
        var raw_in: [ANE_MAX_BINDINGS]c.ami_ane_binding = undefined;
        for (inputs, 0..) |b, i| {
            raw_in[i] = .{ .name = b.name.ptr, .tensor = b.tensor.handle };
        }
        var raw_out: [ANE_MAX_BINDINGS]c.ami_ane_binding = undefined;
        for (outputs, 0..) |b, i| {
            raw_out[i] = .{ .name = b.name.ptr, .tensor = b.tensor.handle };
        }
        var raw: c.ami_ane_predict_info = undefined;
        try check(c.ami_ane_model_predict_oneshot(
            self.handle,
            &raw_in,
            inputs.len,
            &raw_out,
            outputs.len,
            &raw,
        ));
        return .{ .raw = raw };
    }
};

/// Row-major FP32 GEMM via Apple Accelerate `cblas_sgemm` (Spike S7 baseline;
/// ADR-0001 routes all Apple framework calls through `apple_bridge.m`).
/// Computes `out_c[m, n] = alpha * a[m, k] * op(b) + beta * out_c[m, n]`,
/// where `op(b)` is `b[n, k]^T` when `transpose_b` is true, or `b[k, n]` otherwise.
pub fn accelerateSgemm(
    m: u32,
    n: u32,
    k: u32,
    alpha: f32,
    a: []const f32,
    b: []const f32,
    beta: f32,
    out_c: []f32,
    transpose_b: bool,
) Error!void {
    if (m == 0 or n == 0 or k == 0) return error.InvalidArgument;
    const mk = std.math.mul(usize, m, k) catch return error.InvalidArgument;
    const nk = std.math.mul(usize, n, k) catch return error.InvalidArgument;
    const mn = std.math.mul(usize, m, n) catch return error.InvalidArgument;
    if (a.len < mk or b.len < nk or out_c.len < mn) return error.InvalidArgument;
    try check(c.ami_accelerate_sgemm(
        m,
        n,
        k,
        alpha,
        a.ptr,
        b.ptr,
        beta,
        out_c.ptr,
        if (transpose_b) 1 else 0,
    ));
}

// ---------------------------------------------------------------------------
// Zero-copy multimodal vision/video ingress bridge (M6.5 G14, ADR-0002/0004/0019)
// ---------------------------------------------------------------------------

/// Pixel layout of an IOSurface-backed `VisionFrame`.
pub const VisionFormat = enum {
    /// 1-plane `kCVPixelFormatType_32BGRA` -> `MTLPixelFormatBGRA8Unorm`.
    bgra8,
    /// 2-plane `kCVPixelFormatType_420YpCbCr8BiPlanar{Full,Video}Range` -> `R8Unorm` + `RG8Unorm`.
    nv12,
    /// 1-plane `kCVPixelFormatType_OneComponent16Half` -> `R16Float` + zero-copy CoreML `MLMultiArray`.
    f16,

    fn toRaw(self: VisionFormat) u32 {
        return switch (self) {
            .bgra8 => c.AMI_VISION_FMT_BGRA8,
            .nv12 => c.AMI_VISION_FMT_NV12,
            .f16 => c.AMI_VISION_FMT_F16,
        };
    }

    fn fromRaw(raw: u32) VisionFormat {
        return switch (raw) {
            c.AMI_VISION_FMT_NV12 => .nv12,
            c.AMI_VISION_FMT_F16 => .f16,
            else => .bgra8,
        };
    }
};

/// Hardware VideoToolbox codec selection for compressed video clips.
pub const VtCodec = enum {
    h264,
    hevc,

    fn toRaw(self: VtCodec) u32 {
        return switch (self) {
            .h264 => c.AMI_VT_CODEC_H264,
            .hevc => c.AMI_VT_CODEC_HEVC,
        };
    }

    fn fromRaw(raw: u32) VtCodec {
        return switch (raw) {
            c.AMI_VT_CODEC_HEVC => .hevc,
            else => .h264,
        };
    }
};

/// Binding of one zero-copy IOSurface plane `MTLTexture` into `[[texture(i)]]`.
pub const VisionTextureBinding = struct {
    frame: VisionFrame,
    plane: u32 = 0,
};

/// Physical-page and zero-copy Metal metadata for a `VisionFrame`.
pub const VisionFrameInfo = struct {
    raw: c.ami_vision_frame_info,

    pub fn format(self: VisionFrameInfo) VisionFormat {
        return VisionFormat.fromRaw(self.raw.format);
    }
    pub fn cvPixelFormat(self: VisionFrameInfo) u32 {
        return self.raw.cv_pixel_format;
    }
    pub fn width(self: VisionFrameInfo) u32 {
        return self.raw.width;
    }
    pub fn height(self: VisionFrameInfo) u32 {
        return self.raw.height;
    }
    pub fn planeCount(self: VisionFrameInfo) u32 {
        return self.raw.plane_count;
    }
    pub fn isFullRange(self: VisionFrameInfo) bool {
        return self.raw.is_full_range != 0;
    }
    pub fn zeroCopyVerified(self: VisionFrameInfo) bool {
        return self.raw.zero_copy_verified != 0;
    }
    pub fn isPageAligned(self: VisionFrameInfo) bool {
        return self.raw.is_page_aligned != 0;
    }
    pub fn iosurfaceId(self: VisionFrameInfo) u32 {
        return self.raw.iosurface_id;
    }
    pub fn isVtDecoded(self: VisionFrameInfo) bool {
        return self.raw.vt_decoded != 0;
    }
    pub fn baseAddress(self: VisionFrameInfo) u64 {
        return self.raw.base_address;
    }
    pub fn allocSize(self: VisionFrameInfo) u64 {
        return self.raw.alloc_size;
    }
    pub fn surfaceAllocSize(self: VisionFrameInfo) u64 {
        return self.raw.surface_alloc_size;
    }
    pub fn planeWidth(self: VisionFrameInfo, plane: u32) u32 {
        return if (plane < 2) self.raw.plane_width[plane] else 0;
    }
    pub fn planeHeight(self: VisionFrameInfo, plane: u32) u32 {
        return if (plane < 2) self.raw.plane_height[plane] else 0;
    }
    pub fn planeBytesPerRow(self: VisionFrameInfo, plane: u32) u32 {
        return if (plane < 2) self.raw.plane_bytes_per_row[plane] else 0;
    }
    pub fn planeResourceId(self: VisionFrameInfo, plane: u32) u64 {
        return if (plane < 2) self.raw.plane_resource_id[plane] else 0;
    }
    pub fn ptsValue(self: VisionFrameInfo) i64 {
        return self.raw.pts_value;
    }
    pub fn ptsTimescale(self: VisionFrameInfo) i32 {
        return self.raw.pts_timescale;
    }
};

/// CoreML / ANE zero-copy compatibility probe result for a `VisionFrame`.
pub const VisionAneProbe = struct {
    raw: c.ami_vision_ane_probe,

    pub fn pixelBufferFeatureOk(self: VisionAneProbe) bool {
        return self.raw.pixel_buffer_feature_ok != 0;
    }
    pub fn multiArrayZeroCopyOk(self: VisionAneProbe) bool {
        return self.raw.multi_array_zero_copy_ok != 0;
    }
    pub fn iosurfaceIdentityMatched(self: VisionAneProbe) bool {
        return self.raw.iosurface_identity_matched != 0;
    }
};

/// Zero-copy IOSurface-backed `CVPixelBuffer` (`BGRA8`, `NV12`, or `F16`) with
/// per-plane `MTLTexture` views sharing the exact same physical pages (`G14`).
pub const VisionFrame = struct {
    handle: *c.ami_vision_frame,

    /// Scoped CPU lock over one plane of the frame's IOSurface (tests/offline setup only).
    pub const PlaneLockGuard = struct {
        frame: VisionFrame,
        read_only: bool,
        locked: bool,
        bytes: []u8,
        bytes_per_row: usize,
        height: u32,

        pub fn rowBytes(self: PlaneLockGuard, y: u32) []u8 {
            const start = @as(usize, y) * self.bytes_per_row;
            return self.bytes[start .. start + self.bytes_per_row];
        }

        pub fn unlock(self: *PlaneLockGuard) Error!void {
            if (!self.locked) return error.InvalidArgument;
            self.locked = false;
            try check(c.ami_vision_frame_unlock(self.frame.handle, if (self.read_only) 1 else 0));
        }

        /// Safe cleanup for `errdefer` when an error occurs before `unlock`.
        pub fn release(self: *PlaneLockGuard) void {
            if (!self.locked) return;
            self.locked = false;
            const st = c.ami_vision_frame_unlock(self.frame.handle, if (self.read_only) 1 else 0);
            std.debug.assert(st == c.AMI_OK);
        }
    };

    /// Creates a 16 KiB-aligned IOSurface-backed `VisionFrame` with zero-copy Metal views.
    pub fn init(device: Device, fmt: VisionFormat, width: u32, height: u32) Error!VisionFrame {
        if (width == 0 or height == 0 or width > MAX_TEXTURE_DIM or height > MAX_TEXTURE_DIM) {
            return error.InvalidArgument;
        }
        var handle: ?*c.ami_vision_frame = null;
        try check(c.ami_vision_frame_create(device.handle, fmt.toRaw(), width, height, &handle));
        return .{ .handle = handle orelse return error.NullHandle };
    }

    /// Loads a local `.png`/`.jpg` image via ImageIO directly into a 16 KiB-aligned `BGRA8` IOSurface.
    pub fn loadImage(device: Device, path: [:0]const u8) Error!VisionFrame {
        if (path.len == 0) return error.InvalidArgument;
        var handle: ?*c.ami_vision_frame = null;
        try check(c.ami_image_load_iosurface(device.handle, path.ptr, &handle));
        return .{ .handle = handle orelse return error.NullHandle };
    }

    /// Writes a `BGRA8` `VisionFrame` to a local `.png` file via ImageIO (test/offline helper).
    pub fn writePng(self: VisionFrame, path: [:0]const u8) Error!void {
        if (path.len == 0) return error.InvalidArgument;
        try check(c.ami_image_write_png(self.handle, path.ptr));
    }

    pub fn close(self: VisionFrame) void {
        c.ami_vision_frame_destroy(self.handle);
    }

    pub fn info(self: VisionFrame) Error!VisionFrameInfo {
        var raw: c.ami_vision_frame_info = undefined;
        try check(c.ami_vision_frame_query(self.handle, &raw));
        return .{ .raw = raw };
    }

    pub fn planeResourceId(self: VisionFrame, plane: u32) u64 {
        return c.ami_vision_frame_plane_resource_id(self.handle, plane);
    }

    pub fn probeAne(self: VisionFrame) Error!VisionAneProbe {
        var raw: c.ami_vision_ane_probe = undefined;
        try check(c.ami_vision_frame_probe_ane(self.handle, &raw));
        return .{ .raw = raw };
    }

    pub fn toAneTensor(self: VisionFrame, shape: []const u32) Error!AneTensor {
        return AneTensor.fromVisionFrame(self, shape);
    }

    pub fn lockPlane(self: VisionFrame, plane: u32, read_only: bool) Error!PlaneLockGuard {
        var base_ptr: ?*anyopaque = null;
        var bpr: usize = 0;
        var h: u32 = 0;
        try check(c.ami_vision_frame_lock_plane(
            self.handle,
            plane,
            if (read_only) 1 else 0,
            &base_ptr,
            &bpr,
            &h,
        ));
        const raw_ptr = base_ptr orelse {
            _ = c.ami_vision_frame_unlock(self.handle, if (read_only) 1 else 0);
            return error.NullHandle;
        };
        const total = std.math.mul(usize, bpr, h) catch {
            _ = c.ami_vision_frame_unlock(self.handle, if (read_only) 1 else 0);
            return error.VisionFailed;
        };
        const byte_ptr: [*]u8 = @ptrCast(raw_ptr);
        return .{
            .frame = self,
            .read_only = read_only,
            .locked = true,
            .bytes = byte_ptr[0..total],
            .bytes_per_row = bpr,
            .height = h,
        };
    }
};

/// Metadata for a hardware-encoded VideoToolbox clip.
pub const VtClipInfo = struct {
    raw: c.ami_vt_clip_info,

    pub fn codec(self: VtClipInfo) VtCodec {
        return VtCodec.fromRaw(self.raw.codec);
    }
    pub fn width(self: VtClipInfo) u32 {
        return self.raw.width;
    }
    pub fn height(self: VtClipInfo) u32 {
        return self.raw.height;
    }
    pub fn frameCount(self: VtClipInfo) u32 {
        return self.raw.frame_count;
    }
    pub fn fps(self: VtClipInfo) u32 {
        return self.raw.fps;
    }
    pub fn usingHardwareEncoder(self: VtClipInfo) bool {
        return self.raw.using_hardware_encoder != 0;
    }
    pub fn totalCompressedBytes(self: VtClipInfo) u64 {
        return self.raw.total_compressed_bytes;
    }
};

/// Hardware-compressed H.264 / HEVC clip of `CMSampleBuffer`s produced via VideoToolbox.
pub const VtClip = struct {
    handle: *c.ami_vt_clip,

    pub fn encodeTestClip(
        codec: VtCodec,
        width: u32,
        height: u32,
        frame_count: u32,
        fps: u32,
    ) Error!VtClip {
        var handle: ?*c.ami_vt_clip = null;
        try check(c.ami_vt_encode_test_clip(codec.toRaw(), width, height, frame_count, fps, &handle));
        return .{ .handle = handle orelse return error.NullHandle };
    }

    pub fn encodeFrames(codec: VtCodec, frames: []const VisionFrame, fps: u32) Error!VtClip {
        if (frames.len == 0 or frames.len > 128) return error.InvalidArgument;
        var handles: [128]?*c.ami_vision_frame = undefined;
        for (frames, 0..) |frame, i| {
            handles[i] = frame.handle;
        }
        var handle: ?*c.ami_vt_clip = null;
        try check(c.ami_vt_encode_frames(
            codec.toRaw(),
            &handles,
            @intCast(frames.len),
            fps,
            &handle,
        ));
        return .{ .handle = handle orelse return error.NullHandle };
    }

    pub fn close(self: VtClip) void {
        c.ami_vt_clip_destroy(self.handle);
    }

    pub fn info(self: VtClip) Error!VtClipInfo {
        var raw: c.ami_vt_clip_info = undefined;
        try check(c.ami_vt_clip_query(self.handle, &raw));
        return .{ .raw = raw };
    }
};

/// Metadata for a hardware VideoToolbox decompression session.
pub const VtDecodeSessionInfo = struct {
    raw: c.ami_vt_decode_session_info,

    pub fn codec(self: VtDecodeSessionInfo) VtCodec {
        return VtCodec.fromRaw(self.raw.codec);
    }
    pub fn outputFormat(self: VtDecodeSessionInfo) VisionFormat {
        return VisionFormat.fromRaw(self.raw.output_format);
    }
    pub fn width(self: VtDecodeSessionInfo) u32 {
        return self.raw.width;
    }
    pub fn height(self: VtDecodeSessionInfo) u32 {
        return self.raw.height;
    }
    pub fn usingHardwareDecoder(self: VtDecodeSessionInfo) bool {
        return self.raw.using_hardware_decoder != 0;
    }
};

/// Hardware VideoToolbox `VTDecompressionSession` decoding compressed H.264/HEVC
/// frames directly into zero-copy IOSurface-backed `VisionFrame`s (`G14`).
pub const VtDecodeSession = struct {
    handle: *c.ami_vt_decode_session,

    pub fn init(device: Device, clip: VtClip, output_format: VisionFormat) Error!VtDecodeSession {
        if (output_format != .nv12 and output_format != .bgra8) {
            return error.InvalidArgument;
        }
        var handle: ?*c.ami_vt_decode_session = null;
        try check(c.ami_vt_decode_session_create(
            device.handle,
            clip.handle,
            output_format.toRaw(),
            &handle,
        ));
        return .{ .handle = handle orelse return error.NullHandle };
    }

    pub fn close(self: VtDecodeSession) void {
        c.ami_vt_decode_session_destroy(self.handle);
    }

    pub fn info(self: VtDecodeSession) Error!VtDecodeSessionInfo {
        var raw: c.ami_vt_decode_session_info = undefined;
        try check(c.ami_vt_decode_session_query(self.handle, &raw));
        return .{ .raw = raw };
    }

    /// Decodes `frame_index` from `clip` directly into a zero-copy IOSurface `VisionFrame`.
    pub fn decodeFrame(self: VtDecodeSession, clip: VtClip, frame_index: u32) Error!VisionFrame {
        var handle: ?*c.ami_vision_frame = null;
        try check(c.ami_vt_decode_clip_frame(self.handle, clip.handle, frame_index, &handle));
        return .{ .handle = handle orelse return error.NullHandle };
    }
};

// ---------------------------------------------------------------------------
// Pure-Zig minimal CoreML .mlmodel protobuf emitter (zero Python/coremltools)
// ---------------------------------------------------------------------------

pub const CoreMlLinearOp = enum {
    /// Native ANE 1x1 2D convolution over 4D `[1, C_in, 1, S]` -> `[1, C_out, 1, S]`.
    conv1x1,
    /// Dense `innerProduct` over 2D `[M, C_in]` -> `[M, C_out]`.
    inner_product,
};

pub const CoreMlLinearLayer = struct {
    c_in: u32,
    c_out: u32,
    /// Row-major `[c_out, c_in]` fp16 weights (`len == c_out * c_in`).
    weights_f16: []const f16,
};

pub const CoreMlLinearModelSpec = struct {
    op: CoreMlLinearOp = .conv1x1,
    /// For `.conv1x1`: spatial width `S` (multiple of 32 for 64 B IOSurface rows).
    /// For `.inner_product`: batch rows `M` (`>= 1`).
    batch_or_spatial: u32 = 32,
    /// CoreML protobuf specificationVersion (`7` required for `FLOAT16` I/O,
    /// `4` for `FLOAT32` I/O with fp16 weights).
    specification_version: u32 = 7,
    /// ArrayFeatureType dataType: `65552` (`FLOAT16`) or `65568` (`FLOAT32`).
    io_float16: bool = true,
    input_name: []const u8 = "x",
    output_name: []const u8 = "y",
    layers: []const CoreMlLinearLayer,
};

pub const ProtoEmitError = error{
    InvalidSpec,
    OutOfMemory,
};

fn pbAppendVarint(allocator: std.mem.Allocator, out: *std.ArrayList(u8), val: u64) !void {
    var v = val;
    while (v >= 0x80) {
        try out.append(allocator, @as(u8, @intCast((v & 0x7F) | 0x80)));
        v >>= 7;
    }
    try out.append(allocator, @as(u8, @intCast(v)));
}

fn pbAppendTag(allocator: std.mem.Allocator, out: *std.ArrayList(u8), field: u32, wire_type: u3) !void {
    try pbAppendVarint(allocator, out, (@as(u64, field) << 3) | wire_type);
}

fn pbAppendVarintField(allocator: std.mem.Allocator, out: *std.ArrayList(u8), field: u32, val: u64) !void {
    try pbAppendTag(allocator, out, field, 0);
    try pbAppendVarint(allocator, out, val);
}

fn pbAppendBytesField(allocator: std.mem.Allocator, out: *std.ArrayList(u8), field: u32, payload: []const u8) !void {
    try pbAppendTag(allocator, out, field, 2);
    try pbAppendVarint(allocator, out, payload.len);
    try out.appendSlice(allocator, payload);
}

fn pbBuildFeatureDesc(
    allocator: std.mem.Allocator,
    name: []const u8,
    shape: []const u32,
    io_float16: bool,
) ![]u8 {
    // ArrayFeatureType: shape (1), dataType (2: FLOAT16 = 65552, FLOAT32 = 65568)
    var arr_type: std.ArrayList(u8) = .empty;
    defer arr_type.deinit(allocator);
    for (shape) |d| {
        try pbAppendVarintField(allocator, &arr_type, 1, d);
    }
    const dtype_val: u64 = if (io_float16) 65552 else 65568;
    try pbAppendVarintField(allocator, &arr_type, 2, dtype_val);

    // FeatureType: multiArrayType (5)
    var feat_type: std.ArrayList(u8) = .empty;
    defer feat_type.deinit(allocator);
    try pbAppendBytesField(allocator, &feat_type, 5, arr_type.items);

    // FeatureDescription: name (1), type (3)
    var desc: std.ArrayList(u8) = .empty;
    errdefer desc.deinit(allocator);
    try pbAppendBytesField(allocator, &desc, 1, name);
    try pbAppendBytesField(allocator, &desc, 3, feat_type.items);
    return desc.toOwnedSlice(allocator);
}

/// Emits a valid CoreML `.mlmodel` protobuf byte slice containing one or more
/// stacked fp16 linear layers (`1x1 convolution` or `innerProduct`), suitable
/// for direct in-memory loading with `AneModel.loadSpecData` or disk saving.
pub fn emitCoreMlLinearModel(
    allocator: std.mem.Allocator,
    spec: CoreMlLinearModelSpec,
) ProtoEmitError![]u8 {
    if (spec.layers.len == 0 or spec.batch_or_spatial == 0 or
        spec.input_name.len == 0 or spec.output_name.len == 0)
    {
        return error.InvalidSpec;
    }
    for (spec.layers, 0..) |layer, idx| {
        if (layer.c_in == 0 or layer.c_out == 0) return error.InvalidSpec;
        const expected_len = std.math.mul(usize, layer.c_out, layer.c_in) catch return error.InvalidSpec;
        if (layer.weights_f16.len != expected_len) return error.InvalidSpec;
        if (idx > 0 and spec.layers[idx - 1].c_out != layer.c_in) return error.InvalidSpec;
    }

    const first_in = spec.layers[0].c_in;
    const last_out = spec.layers[spec.layers.len - 1].c_out;

    const in_shape_3d = [_]u32{ first_in, 1, spec.batch_or_spatial };
    const out_shape_3d = [_]u32{ last_out, 1, spec.batch_or_spatial };
    const in_shape_2d = [_]u32{ spec.batch_or_spatial, first_in };
    const out_shape_2d = [_]u32{ spec.batch_or_spatial, last_out };

    // Always use 2D [M, C] with EXACT_ARRAY_MAPPING for innerProduct so
    // CVPixelBuffer-backed MLMultiArray tensors always have rank >= 2.
    const use_exact_mapping = (spec.op == .inner_product);
    const in_shape: []const u32 = switch (spec.op) {
        .conv1x1 => &in_shape_3d,
        .inner_product => &in_shape_2d,
    };
    const out_shape: []const u32 = switch (spec.op) {
        .conv1x1 => &out_shape_3d,
        .inner_product => &out_shape_2d,
    };

    const in_feat = try pbBuildFeatureDesc(allocator, spec.input_name, in_shape, spec.io_float16);
    defer allocator.free(in_feat);
    const out_feat = try pbBuildFeatureDesc(allocator, spec.output_name, out_shape, spec.io_float16);
    defer allocator.free(out_feat);

    // ModelDescription: input (1), output (10)
    var model_desc: std.ArrayList(u8) = .empty;
    defer model_desc.deinit(allocator);
    try pbAppendBytesField(allocator, &model_desc, 1, in_feat);
    try pbAppendBytesField(allocator, &model_desc, 10, out_feat);

    // NeuralNetwork: repeated NeuralNetworkLayer layers (1), arrayInputShapeMapping (5)
    var nn: std.ArrayList(u8) = .empty;
    defer nn.deinit(allocator);
    if (use_exact_mapping) {
        // EXACT_ARRAY_MAPPING = 1 (allows 2D [M, C] inputs in CoreML NeuralNetwork)
        try pbAppendVarintField(allocator, &nn, 5, 1);
    }

    var name_buf: [32]u8 = undefined;
    var prev_buf: [32]u8 = undefined;
    var next_buf: [32]u8 = undefined;

    for (spec.layers, 0..) |layer, idx| {
        const layer_name = std.fmt.bufPrint(&name_buf, "linear_{d}", .{idx}) catch return error.InvalidSpec;
        const in_blob = if (idx == 0)
            spec.input_name
        else
            std.fmt.bufPrint(&prev_buf, "h_{d}", .{idx - 1}) catch return error.InvalidSpec;
        const out_blob = if (idx + 1 == spec.layers.len)
            spec.output_name
        else
            std.fmt.bufPrint(&next_buf, "h_{d}", .{idx}) catch return error.InvalidSpec;

        // WeightParams: float16Value (2)
        var w_params: std.ArrayList(u8) = .empty;
        defer w_params.deinit(allocator);
        try pbAppendBytesField(allocator, &w_params, 2, std.mem.sliceAsBytes(layer.weights_f16));

        var layer_msg: std.ArrayList(u8) = .empty;
        defer layer_msg.deinit(allocator);
        try pbAppendBytesField(allocator, &layer_msg, 1, layer_name);
        try pbAppendBytesField(allocator, &layer_msg, 2, in_blob);
        try pbAppendBytesField(allocator, &layer_msg, 3, out_blob);

        switch (spec.op) {
            .conv1x1 => {
                // ConvolutionLayerParams:
                // outputChannels(1), kernelChannels(2), nGroups(10),
                // kernelSize(20) x2, stride(30) x2, valid(50), hasBias(70), weights(90)
                var conv: std.ArrayList(u8) = .empty;
                defer conv.deinit(allocator);
                try pbAppendVarintField(allocator, &conv, 1, layer.c_out);
                try pbAppendVarintField(allocator, &conv, 2, layer.c_in);
                try pbAppendVarintField(allocator, &conv, 10, 1);
                try pbAppendVarintField(allocator, &conv, 20, 1);
                try pbAppendVarintField(allocator, &conv, 20, 1);
                try pbAppendVarintField(allocator, &conv, 30, 1);
                try pbAppendVarintField(allocator, &conv, 30, 1);
                try pbAppendBytesField(allocator, &conv, 50, &.{});
                try pbAppendVarintField(allocator, &conv, 70, 0);
                try pbAppendBytesField(allocator, &conv, 90, w_params.items);
                try pbAppendBytesField(allocator, &layer_msg, 100, conv.items);
            },
            .inner_product => {
                // InnerProductLayerParams:
                // inputChannels(1), outputChannels(2), hasBias(10), weights(20)
                var ip: std.ArrayList(u8) = .empty;
                defer ip.deinit(allocator);
                try pbAppendVarintField(allocator, &ip, 1, layer.c_in);
                try pbAppendVarintField(allocator, &ip, 2, layer.c_out);
                try pbAppendVarintField(allocator, &ip, 10, 0);
                try pbAppendBytesField(allocator, &ip, 20, w_params.items);
                try pbAppendBytesField(allocator, &layer_msg, 140, ip.items);
            },
        }

        try pbAppendBytesField(allocator, &nn, 1, layer_msg.items);
    }

    // Root Model: specificationVersion (1), description (2), neuralNetwork (500)
    var root: std.ArrayList(u8) = .empty;
    errdefer root.deinit(allocator);
    try pbAppendVarintField(allocator, &root, 1, spec.specification_version);
    try pbAppendBytesField(allocator, &root, 2, model_desc.items);
    try pbAppendBytesField(allocator, &root, 500, nn.items);
    return root.toOwnedSlice(allocator);
}

fn pbAppendSignedVarintField(allocator: std.mem.Allocator, out: *std.ArrayList(u8), field: u32, val: i64) !void {
    try pbAppendTag(allocator, out, field, 0);
    try pbAppendVarint(allocator, out, @bitCast(val));
}

fn pbAppendF32Field(allocator: std.mem.Allocator, out: *std.ArrayList(u8), field: u32, val: f32) !void {
    try pbAppendTag(allocator, out, field, 5);
    const bits: u32 = @bitCast(val);
    var le: [4]u8 = undefined;
    std.mem.writeInt(u32, &le, bits, .little);
    try out.appendSlice(allocator, &le);
}

fn pbAppendLayer(
    allocator: std.mem.Allocator,
    nn: *std.ArrayList(u8),
    name: []const u8,
    inputs: []const []const u8,
    outputs: []const []const u8,
    layer_field: u32,
    layer_body: []const u8,
) !void {
    var layer_msg: std.ArrayList(u8) = .empty;
    defer layer_msg.deinit(allocator);
    try pbAppendBytesField(allocator, &layer_msg, 1, name);
    for (inputs) |inp| try pbAppendBytesField(allocator, &layer_msg, 2, inp);
    for (outputs) |out_name| try pbAppendBytesField(allocator, &layer_msg, 3, out_name);
    try pbAppendBytesField(allocator, &layer_msg, layer_field, layer_body);
    try pbAppendBytesField(allocator, nn, 1, layer_msg.items);
}

fn pbAppendConv1x1Layer(
    allocator: std.mem.Allocator,
    nn: *std.ArrayList(u8),
    name: []const u8,
    in_blob: []const u8,
    out_blob: []const u8,
    c_in: u32,
    c_out: u32,
    weights_f16: []const f16,
) !void {
    var w_params: std.ArrayList(u8) = .empty;
    defer w_params.deinit(allocator);
    try pbAppendBytesField(allocator, &w_params, 2, std.mem.sliceAsBytes(weights_f16));

    var conv: std.ArrayList(u8) = .empty;
    defer conv.deinit(allocator);
    try pbAppendVarintField(allocator, &conv, 1, c_out);
    try pbAppendVarintField(allocator, &conv, 2, c_in);
    try pbAppendVarintField(allocator, &conv, 10, 1);
    try pbAppendVarintField(allocator, &conv, 20, 1);
    try pbAppendVarintField(allocator, &conv, 20, 1);
    try pbAppendVarintField(allocator, &conv, 30, 1);
    try pbAppendVarintField(allocator, &conv, 30, 1);
    try pbAppendBytesField(allocator, &conv, 50, &.{});
    try pbAppendVarintField(allocator, &conv, 70, 0);
    try pbAppendBytesField(allocator, &conv, 90, w_params.items);
    try pbAppendLayer(allocator, nn, name, &.{in_blob}, &.{out_blob}, 100, conv.items);
}

/// Emits a CoreML `BatchedMatMulLayerParams` (field 1045) with constant fp16
/// weights supplied in row-major `[n_out, k_in]` (standard `.amf` / PyTorch
/// linear layout). CoreML stores `BatchedMatMulLayerParams.weights` in
/// column-major `(weightMatrixFirstDimension=k_in, weightMatrixSecondDimension=n_out)`
/// order, which is identical to row-major `[n_out, k_in]`.
fn pbAppendBatchedMatmulConstLayer(
    allocator: std.mem.Allocator,
    nn: *std.ArrayList(u8),
    name: []const u8,
    in_blob: []const u8,
    out_blob: []const u8,
    k_in: u32,
    n_out: u32,
    weights_nk: []const f16,
) !void {
    const total = std.math.mul(usize, k_in, n_out) catch return error.InvalidSpec;
    if (weights_nk.len != total) return error.InvalidSpec;

    var w_params: std.ArrayList(u8) = .empty;
    defer w_params.deinit(allocator);
    try pbAppendBytesField(allocator, &w_params, 2, std.mem.sliceAsBytes(weights_nk));

    // BatchedMatMulLayerParams:
    // weightMatrixFirstDimension(5), weightMatrixSecondDimension(6), hasBias(7), weights(8)
    var bmm: std.ArrayList(u8) = .empty;
    defer bmm.deinit(allocator);
    try pbAppendVarintField(allocator, &bmm, 5, k_in);
    try pbAppendVarintField(allocator, &bmm, 6, n_out);
    try pbAppendVarintField(allocator, &bmm, 7, 0);
    try pbAppendBytesField(allocator, &bmm, 8, w_params.items);
    try pbAppendLayer(allocator, nn, name, &.{in_blob}, &.{out_blob}, 1045, bmm.items);
}

fn pbAppendBatchedMatmul2InLayer(
    allocator: std.mem.Allocator,
    nn: *std.ArrayList(u8),
    name: []const u8,
    in_a: []const u8,
    in_b: []const u8,
    out_blob: []const u8,
    transpose_a: bool,
    transpose_b: bool,
) !void {
    var bmm: std.ArrayList(u8) = .empty;
    defer bmm.deinit(allocator);
    if (transpose_a) try pbAppendVarintField(allocator, &bmm, 1, 1);
    if (transpose_b) try pbAppendVarintField(allocator, &bmm, 2, 1);
    try pbAppendLayer(allocator, nn, name, &.{ in_a, in_b }, &.{out_blob}, 1045, bmm.items);
}

fn pbAppendReshapeStaticLayer(
    allocator: std.mem.Allocator,
    nn: *std.ArrayList(u8),
    name: []const u8,
    in_blob: []const u8,
    out_blob: []const u8,
    target_shape: []const u32,
) !void {
    var body: std.ArrayList(u8) = .empty;
    defer body.deinit(allocator);
    for (target_shape) |d| try pbAppendVarintField(allocator, &body, 1, d);
    try pbAppendLayer(allocator, nn, name, &.{in_blob}, &.{out_blob}, 1140, body.items);
}

fn pbAppendTransposeLayer(
    allocator: std.mem.Allocator,
    nn: *std.ArrayList(u8),
    name: []const u8,
    in_blob: []const u8,
    out_blob: []const u8,
    axes: []const u32,
) !void {
    var body: std.ArrayList(u8) = .empty;
    defer body.deinit(allocator);
    for (axes) |a| try pbAppendVarintField(allocator, &body, 1, a);
    try pbAppendLayer(allocator, nn, name, &.{in_blob}, &.{out_blob}, 985, body.items);
}

fn pbAppendConcatNdLayer(
    allocator: std.mem.Allocator,
    nn: *std.ArrayList(u8),
    name: []const u8,
    in_a: []const u8,
    in_b: []const u8,
    out_blob: []const u8,
    axis: i64,
) !void {
    var body: std.ArrayList(u8) = .empty;
    defer body.deinit(allocator);
    try pbAppendSignedVarintField(allocator, &body, 1, axis);
    try pbAppendLayer(allocator, nn, name, &.{ in_a, in_b }, &.{out_blob}, 980, body.items);
}

fn pbAppendSoftmaxNdLayer(
    allocator: std.mem.Allocator,
    nn: *std.ArrayList(u8),
    name: []const u8,
    in_blob: []const u8,
    out_blob: []const u8,
    axis: i64,
) !void {
    var body: std.ArrayList(u8) = .empty;
    defer body.deinit(allocator);
    try pbAppendSignedVarintField(allocator, &body, 1, axis);
    try pbAppendLayer(allocator, nn, name, &.{in_blob}, &.{out_blob}, 950, body.items);
}

fn pbAppendActivationLinearLayer(
    allocator: std.mem.Allocator,
    nn: *std.ArrayList(u8),
    name: []const u8,
    in_blob: []const u8,
    out_blob: []const u8,
    alpha: f32,
) !void {
    // ActivationParams (130) -> linear (5) -> alpha (1)
    var lin: std.ArrayList(u8) = .empty;
    defer lin.deinit(allocator);
    try pbAppendF32Field(allocator, &lin, 1, alpha);

    var act: std.ArrayList(u8) = .empty;
    defer act.deinit(allocator);
    try pbAppendBytesField(allocator, &act, 5, lin.items);
    try pbAppendLayer(allocator, nn, name, &.{in_blob}, &.{out_blob}, 130, act.items);
}

fn pbAppendActivationReluLayer(
    allocator: std.mem.Allocator,
    nn: *std.ArrayList(u8),
    name: []const u8,
    in_blob: []const u8,
    out_blob: []const u8,
) !void {
    var act: std.ArrayList(u8) = .empty;
    defer act.deinit(allocator);
    try pbAppendBytesField(allocator, &act, 10, &.{});
    try pbAppendLayer(allocator, nn, name, &.{in_blob}, &.{out_blob}, 130, act.items);
}

fn pbAppendActivationSigmoidLayer(
    allocator: std.mem.Allocator,
    nn: *std.ArrayList(u8),
    name: []const u8,
    in_blob: []const u8,
    out_blob: []const u8,
) !void {
    var act: std.ArrayList(u8) = .empty;
    defer act.deinit(allocator);
    try pbAppendBytesField(allocator, &act, 40, &.{});
    try pbAppendLayer(allocator, nn, name, &.{in_blob}, &.{out_blob}, 130, act.items);
}

/// Specification for a CoreML ANE chunked GQA attention block (A1, A3, S6).
///
/// Tensor contract (all `FLOAT16` / `OneComponent16Half` zero-copy `KvTile`s):
/// - Inputs:
///   - `"x_in"`:   `[1, chunk_tokens, hidden_size]`
///   - `"k_past"`: `[1, num_kv_heads, kv_bucket_tokens - chunk_tokens, head_dim]` (omitted when `kv_bucket_tokens == chunk_tokens`)
///   - `"v_past"`: `[1, num_kv_heads, kv_bucket_tokens - chunk_tokens, head_dim]` (omitted when `kv_bucket_tokens == chunk_tokens`)
///   - `"mask"`:   `[1, 1, chunk_tokens, kv_bucket_tokens]` (0.0 for valid causal keys, `-10000.0` for masked keys)
/// - Outputs (all backed zero-copy via `outputBackings`):
///   - `"attn_out"`:   `[1, chunk_tokens, hidden_size]`
///   - `"k_tile_out"`: `[1, num_kv_heads, chunk_tokens, head_dim]` (exact ADR-0004 token-major `KvTile`)
///   - `"v_tile_out"`: `[1, num_kv_heads, chunk_tokens, head_dim]` (exact ADR-0004 token-major `KvTile`)
///
/// All linear weights are row-major `[out_dim, in_dim]` fp16.
pub const CoreMlAttentionSpec = struct {
    hidden_size: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    chunk_tokens: u32 = 512,
    /// Total KV bucket length `B >= chunk_tokens` (e.g. `512, 1024, 2048, 4096`).
    kv_bucket_tokens: u32 = 512,
    /// Softmax scale; defaults to `1 / sqrt(head_dim)` when `null`.
    scale: ?f32 = null,
    /// Use ANE-native 1x1 convolutions for Q/K/V/O projections (recommended on M4).
    use_conv_proj: bool = true,
    /// `[num_q_heads * head_dim, hidden_size]` fp16.
    w_q: []const f16,
    /// `[num_kv_heads * head_dim, hidden_size]` fp16.
    w_k: []const f16,
    /// `[num_kv_heads * head_dim, hidden_size]` fp16.
    w_v: []const f16,
    /// `[hidden_size, num_q_heads * head_dim]` fp16.
    w_o: []const f16,
};

/// Emits a pure-4D GQA chunked attention CoreML `.mlmodel` protobuf (`specVersion = 7`,
/// `FLOAT16` zero-copy IOSurface I/O) that runs 100% on the Apple Neural Engine
/// without K/V head duplication (`Q` folded as `[1, kv_heads, gqa_ratio * S, D]`
/// and pre-scaled before `Q @ K^T` to avoid spilling a full scores tensor to DRAM).
pub fn emitCoreMlAttentionBlock(
    allocator: std.mem.Allocator,
    spec: CoreMlAttentionSpec,
) ProtoEmitError![]u8 {
    if (spec.hidden_size == 0 or spec.num_q_heads == 0 or spec.num_kv_heads == 0 or
        spec.head_dim == 0 or spec.chunk_tokens == 0 or
        spec.kv_bucket_tokens < spec.chunk_tokens or
        (spec.num_q_heads % spec.num_kv_heads) != 0 or
        (spec.hidden_size % KV_TILE_ROW_ALIGN_F16) != 0 or
        (spec.head_dim % KV_TILE_ROW_ALIGN_F16) != 0 or
        (spec.kv_bucket_tokens % KV_TILE_ROW_ALIGN_F16) != 0)
    {
        return error.InvalidSpec;
    }

    const h = spec.hidden_size;
    const qh = spec.num_q_heads;
    const kvh = spec.num_kv_heads;
    const d = spec.head_dim;
    const s = spec.chunk_tokens;
    const b = spec.kv_bucket_tokens;
    const past = b - s;
    const gqa = qh / kvh;
    const q_dim = std.math.mul(u32, qh, d) catch return error.InvalidSpec;
    const kv_dim = std.math.mul(u32, kvh, d) catch return error.InvalidSpec;
    const gs = std.math.mul(u32, gqa, s) catch return error.InvalidSpec;

    const exp_q = std.math.mul(usize, q_dim, h) catch return error.InvalidSpec;
    const exp_kv = std.math.mul(usize, kv_dim, h) catch return error.InvalidSpec;
    const exp_o = std.math.mul(usize, h, q_dim) catch return error.InvalidSpec;
    if (spec.w_q.len != exp_q or spec.w_k.len != exp_kv or
        spec.w_v.len != exp_kv or spec.w_o.len != exp_o)
    {
        return error.InvalidSpec;
    }

    const scale: f32 = spec.scale orelse (1.0 / @sqrt(@as(f32, @floatFromInt(d))));
    if (!std.math.isFinite(scale) or !(scale > 0.0)) return error.InvalidSpec;

    // ModelDescription: inputs (1), outputs (10)
    var model_desc: std.ArrayList(u8) = .empty;
    defer model_desc.deinit(allocator);

    {
        const f_x = try pbBuildFeatureDesc(allocator, "x_in", &.{ 1, s, h }, true);
        defer allocator.free(f_x);
        try pbAppendBytesField(allocator, &model_desc, 1, f_x);
    }
    if (past > 0) {
        const f_kp = try pbBuildFeatureDesc(allocator, "k_past", &.{ 1, kvh, past, d }, true);
        defer allocator.free(f_kp);
        try pbAppendBytesField(allocator, &model_desc, 1, f_kp);

        const f_vp = try pbBuildFeatureDesc(allocator, "v_past", &.{ 1, kvh, past, d }, true);
        defer allocator.free(f_vp);
        try pbAppendBytesField(allocator, &model_desc, 1, f_vp);
    }
    {
        const f_m = try pbBuildFeatureDesc(allocator, "mask", &.{ 1, 1, s, b }, true);
        defer allocator.free(f_m);
        try pbAppendBytesField(allocator, &model_desc, 1, f_m);

        const f_out = try pbBuildFeatureDesc(allocator, "attn_out", &.{ 1, s, h }, true);
        defer allocator.free(f_out);
        try pbAppendBytesField(allocator, &model_desc, 10, f_out);

        const f_ko = try pbBuildFeatureDesc(allocator, "k_tile_out", &.{ 1, kvh, s, d }, true);
        defer allocator.free(f_ko);
        try pbAppendBytesField(allocator, &model_desc, 10, f_ko);

        const f_vo = try pbBuildFeatureDesc(allocator, "v_tile_out", &.{ 1, kvh, s, d }, true);
        defer allocator.free(f_vo);
        try pbAppendBytesField(allocator, &model_desc, 10, f_vo);
    }

    var nn: std.ArrayList(u8) = .empty;
    defer nn.deinit(allocator);
    try pbAppendVarintField(allocator, &nn, 5, 1); // EXACT_ARRAY_MAPPING

    if (spec.use_conv_proj) {
        try pbAppendTransposeLayer(allocator, &nn, "x_tr", "x_in", "x_hs", &.{ 0, 2, 1 });
        try pbAppendReshapeStaticLayer(allocator, &nn, "x_4d", "x_hs", "x_nchw", &.{ 1, h, 1, s });
        try pbAppendConv1x1Layer(allocator, &nn, "q_proj", "x_nchw", "q_nchw", h, q_dim, spec.w_q);
        try pbAppendActivationLinearLayer(allocator, &nn, "q_scale", "q_nchw", "q_scaled", scale);
        try pbAppendConv1x1Layer(allocator, &nn, "k_proj", "x_nchw", "k_nchw", h, kv_dim, spec.w_k);
        try pbAppendConv1x1Layer(allocator, &nn, "v_proj", "x_nchw", "v_nchw", h, kv_dim, spec.w_v);

        try pbAppendReshapeStaticLayer(allocator, &nn, "q_r4", "q_scaled", "q_hds", &.{ 1, qh, d, s });
        try pbAppendTransposeLayer(allocator, &nn, "q_tr", "q_hds", "q_hsd", &.{ 0, 1, 3, 2 });
        try pbAppendReshapeStaticLayer(allocator, &nn, "q_grp", "q_hsd", "q_g", &.{ 1, kvh, gs, d });

        try pbAppendReshapeStaticLayer(allocator, &nn, "k_r4", "k_nchw", "k_hds", &.{ 1, kvh, d, s });
        try pbAppendTransposeLayer(allocator, &nn, "k_tr", "k_hds", "k_tile_out", &.{ 0, 1, 3, 2 });

        try pbAppendReshapeStaticLayer(allocator, &nn, "v_r4", "v_nchw", "v_hds", &.{ 1, kvh, d, s });
        try pbAppendTransposeLayer(allocator, &nn, "v_tr", "v_hds", "v_tile_out", &.{ 0, 1, 3, 2 });
    } else {
        try pbAppendBatchedMatmulConstLayer(allocator, &nn, "q_proj", "x_in", "q_flat", h, q_dim, spec.w_q);
        try pbAppendActivationLinearLayer(allocator, &nn, "q_scale", "q_flat", "q_scaled", scale);
        try pbAppendBatchedMatmulConstLayer(allocator, &nn, "k_proj", "x_in", "k_flat", h, kv_dim, spec.w_k);
        try pbAppendBatchedMatmulConstLayer(allocator, &nn, "v_proj", "x_in", "v_flat", h, kv_dim, spec.w_v);

        try pbAppendReshapeStaticLayer(allocator, &nn, "q_rs", "q_scaled", "q_4d", &.{ 1, s, qh, d });
        try pbAppendTransposeLayer(allocator, &nn, "q_tr", "q_4d", "q_h", &.{ 0, 2, 1, 3 });
        try pbAppendReshapeStaticLayer(allocator, &nn, "q_grp", "q_h", "q_g", &.{ 1, kvh, gs, d });

        try pbAppendReshapeStaticLayer(allocator, &nn, "k_rs", "k_flat", "k_4d", &.{ 1, s, kvh, d });
        try pbAppendTransposeLayer(allocator, &nn, "k_tr", "k_4d", "k_tile_out", &.{ 0, 2, 1, 3 });

        try pbAppendReshapeStaticLayer(allocator, &nn, "v_rs", "v_flat", "v_4d", &.{ 1, s, kvh, d });
        try pbAppendTransposeLayer(allocator, &nn, "v_tr", "v_4d", "v_tile_out", &.{ 0, 2, 1, 3 });
    }

    const k_src: []const u8 = if (past > 0) blk: {
        try pbAppendConcatNdLayer(allocator, &nn, "k_cat", "k_past", "k_tile_out", "k_all", 2);
        try pbAppendConcatNdLayer(allocator, &nn, "v_cat", "v_past", "v_tile_out", "v_all", 2);
        break :blk "k_all";
    } else "k_tile_out";
    const v_src: []const u8 = if (past > 0) "v_all" else "v_tile_out";

    // Pure-4D GQA attention without repeating K/V heads.
    try pbAppendBatchedMatmul2InLayer(allocator, &nn, "qk", "q_g", k_src, "scores_raw", false, true);
    try pbAppendReshapeStaticLayer(allocator, &nn, "s_r4", "scores_raw", "scores_4d", &.{ kvh, gqa, s, b });
    try pbAppendLayer(allocator, &nn, "add_mask", &.{ "scores_4d", "mask" }, &.{"scores_m"}, 880, &.{});
    try pbAppendSoftmaxNdLayer(allocator, &nn, "sm", "scores_m", "probs_4d", -1);
    try pbAppendReshapeStaticLayer(allocator, &nn, "p_r4", "probs_4d", "probs", &.{ 1, kvh, gs, b });
    try pbAppendBatchedMatmul2InLayer(allocator, &nn, "pv", "probs", v_src, "ctx_g", false, false);

    if (spec.use_conv_proj) {
        try pbAppendReshapeStaticLayer(allocator, &nn, "ctx_r4", "ctx_g", "ctx_hsd", &.{ 1, qh, s, d });
        try pbAppendTransposeLayer(allocator, &nn, "ctx_tr", "ctx_hsd", "ctx_hds", &.{ 0, 1, 3, 2 });
        try pbAppendReshapeStaticLayer(allocator, &nn, "ctx_nchw", "ctx_hds", "ctx_4d", &.{ 1, q_dim, 1, s });
        try pbAppendConv1x1Layer(allocator, &nn, "o_proj", "ctx_4d", "out_nchw", q_dim, h, spec.w_o);
        try pbAppendReshapeStaticLayer(allocator, &nn, "out_r3", "out_nchw", "out_hs", &.{ 1, h, s });
        try pbAppendTransposeLayer(allocator, &nn, "out_tr", "out_hs", "attn_out", &.{ 0, 2, 1 });
    } else {
        try pbAppendReshapeStaticLayer(allocator, &nn, "ctx_r4", "ctx_g", "ctx_h", &.{ 1, qh, s, d });
        try pbAppendTransposeLayer(allocator, &nn, "ctx_tr", "ctx_h", "ctx_t", &.{ 0, 2, 1, 3 });
        try pbAppendReshapeStaticLayer(allocator, &nn, "ctx_rs", "ctx_t", "ctx_flat", &.{ 1, s, q_dim });
        try pbAppendBatchedMatmulConstLayer(allocator, &nn, "o_proj", "ctx_flat", "attn_out", q_dim, h, spec.w_o);
    }

    var root: std.ArrayList(u8) = .empty;
    errdefer root.deinit(allocator);
    try pbAppendVarintField(allocator, &root, 1, 7);
    try pbAppendBytesField(allocator, &root, 2, model_desc.items);
    try pbAppendBytesField(allocator, &root, 500, nn.items);
    return root.toOwnedSlice(allocator);
}

/// Specification for a single fixed-shape ANE QKV projection & IOSurface KV-write
/// block (`A3` split ANE QKV + GPU split-KV attention strategy).
///
/// Tensor contract (all zero-copy `OneComponent16Half` `KvTile`s):
/// - Input: `"x_in"` `[1, chunk_tokens, hidden_size]`
/// - Outputs:
///   - `"q_out"`:      `[1, chunk_tokens, num_q_heads * head_dim]`
///   - `"k_tile_out"`: `[1, num_kv_heads, chunk_tokens, head_dim]`
///   - `"v_tile_out"`: `[1, num_kv_heads, chunk_tokens, head_dim]`
pub const CoreMlQkvSpec = struct {
    hidden_size: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    chunk_tokens: u32 = 512,
    w_q: []const f16,
    w_k: []const f16,
    w_v: []const f16,
};

/// Emits a fixed-shape ANE QKV projection CoreML `.mlmodel` protobuf (`specVersion = 7`)
/// writing `q_out`, `k_tile_out`, and `v_tile_out` zero-copy into IOSurface `KvTile`s.
pub fn emitCoreMlQkvProjectionBlock(
    allocator: std.mem.Allocator,
    spec: CoreMlQkvSpec,
) ProtoEmitError![]u8 {
    if (spec.hidden_size == 0 or spec.num_q_heads == 0 or spec.num_kv_heads == 0 or
        spec.head_dim == 0 or spec.chunk_tokens == 0 or
        (spec.hidden_size % KV_TILE_ROW_ALIGN_F16) != 0 or
        (spec.head_dim % KV_TILE_ROW_ALIGN_F16) != 0)
    {
        return error.InvalidSpec;
    }

    const h = spec.hidden_size;
    const qh = spec.num_q_heads;
    const kvh = spec.num_kv_heads;
    const d = spec.head_dim;
    const s = spec.chunk_tokens;
    const q_dim = std.math.mul(u32, qh, d) catch return error.InvalidSpec;
    const kv_dim = std.math.mul(u32, kvh, d) catch return error.InvalidSpec;

    const exp_q = std.math.mul(usize, q_dim, h) catch return error.InvalidSpec;
    const exp_kv = std.math.mul(usize, kv_dim, h) catch return error.InvalidSpec;
    if (spec.w_q.len != exp_q or spec.w_k.len != exp_kv or spec.w_v.len != exp_kv) {
        return error.InvalidSpec;
    }

    var model_desc: std.ArrayList(u8) = .empty;
    defer model_desc.deinit(allocator);
    {
        const f_x = try pbBuildFeatureDesc(allocator, "x_in", &.{ 1, s, h }, true);
        defer allocator.free(f_x);
        try pbAppendBytesField(allocator, &model_desc, 1, f_x);

        const f_q = try pbBuildFeatureDesc(allocator, "q_out", &.{ 1, s, q_dim }, true);
        defer allocator.free(f_q);
        try pbAppendBytesField(allocator, &model_desc, 10, f_q);

        const f_k = try pbBuildFeatureDesc(allocator, "k_tile_out", &.{ 1, kvh, s, d }, true);
        defer allocator.free(f_k);
        try pbAppendBytesField(allocator, &model_desc, 10, f_k);

        const f_v = try pbBuildFeatureDesc(allocator, "v_tile_out", &.{ 1, kvh, s, d }, true);
        defer allocator.free(f_v);
        try pbAppendBytesField(allocator, &model_desc, 10, f_v);
    }

    var nn: std.ArrayList(u8) = .empty;
    defer nn.deinit(allocator);
    try pbAppendVarintField(allocator, &nn, 5, 1);

    try pbAppendTransposeLayer(allocator, &nn, "x_tr", "x_in", "x_hs", &.{ 0, 2, 1 });
    try pbAppendReshapeStaticLayer(allocator, &nn, "x_4d", "x_hs", "x_nchw", &.{ 1, h, 1, s });
    try pbAppendConv1x1Layer(allocator, &nn, "q_proj", "x_nchw", "q_nchw", h, q_dim, spec.w_q);
    try pbAppendConv1x1Layer(allocator, &nn, "k_proj", "x_nchw", "k_nchw", h, kv_dim, spec.w_k);
    try pbAppendConv1x1Layer(allocator, &nn, "v_proj", "x_nchw", "v_nchw", h, kv_dim, spec.w_v);

    try pbAppendReshapeStaticLayer(allocator, &nn, "q_r3", "q_nchw", "q_qs", &.{ 1, q_dim, s });
    try pbAppendTransposeLayer(allocator, &nn, "q_tr", "q_qs", "q_out", &.{ 0, 2, 1 });

    try pbAppendReshapeStaticLayer(allocator, &nn, "k_r4", "k_nchw", "k_hds", &.{ 1, kvh, d, s });
    try pbAppendTransposeLayer(allocator, &nn, "k_tr", "k_hds", "k_tile_out", &.{ 0, 1, 3, 2 });

    try pbAppendReshapeStaticLayer(allocator, &nn, "v_r4", "v_nchw", "v_hds", &.{ 1, kvh, d, s });
    try pbAppendTransposeLayer(allocator, &nn, "v_tr", "v_hds", "v_tile_out", &.{ 0, 1, 3, 2 });

    var root: std.ArrayList(u8) = .empty;
    errdefer root.deinit(allocator);
    try pbAppendVarintField(allocator, &root, 1, 7);
    try pbAppendBytesField(allocator, &root, 2, model_desc.items);
    try pbAppendBytesField(allocator, &root, 500, nn.items);
    return root.toOwnedSlice(allocator);
}

/// Activation for an optional 2-layer bottleneck MLP router predictor (A2).
pub const CoreMlRouterActivation = enum {
    none,
    relu,
    silu,
};

/// Specification for an ANE Multi-Head MoE Router Predictor (`[M, hidden_size]`
/// -> `[M, out_experts]` fp16, where `out_experts = lookahead_layers * num_experts`
/// is a multiple of 32 for zero-copy `KvTile` `outputBackings`).
pub const CoreMlRouterPredictorSpec = struct {
    /// Batch rows `M >= 1` (`1` for single-token decode, `32`/`64` for tree verification).
    batch_size: u32,
    hidden_size: u32,
    /// Total output router logits per token (`lookahead_layers * num_experts`, multiple of 32).
    out_experts: u32,
    /// `0` for direct 1-layer multi-head linear (`[out_experts, hidden_size]`);
    /// `> 0` for a 2-layer bottleneck MLP (`hidden_size -> bottleneck_dim -> out_experts`).
    bottleneck_dim: u32 = 0,
    activation: CoreMlRouterActivation = .silu,
    input_name: []const u8 = "x",
    output_name: []const u8 = "logits",
    /// Row-major `[out_experts, hidden_size]` if `bottleneck_dim == 0`,
    /// or `[bottleneck_dim, hidden_size]` if `bottleneck_dim > 0`.
    weights_layer1: []const f16,
    /// Row-major `[out_experts, bottleneck_dim]`, required iff `bottleneck_dim > 0`.
    weights_layer2: ?[]const f16 = null,
};

/// Emits an ANE-native CoreML `.mlmodel` protobuf (`specVersion = 7`, `FLOAT16` I/O)
/// for cross-layer MoE expert prediction (`A2`).
pub fn emitCoreMlRouterPredictor(
    allocator: std.mem.Allocator,
    spec: CoreMlRouterPredictorSpec,
) ProtoEmitError![]u8 {
    if (spec.batch_size == 0 or spec.hidden_size == 0 or spec.out_experts == 0 or
        (spec.hidden_size % KV_TILE_ROW_ALIGN_F16) != 0 or
        (spec.out_experts % KV_TILE_ROW_ALIGN_F16) != 0 or
        spec.input_name.len == 0 or spec.output_name.len == 0)
    {
        return error.InvalidSpec;
    }

    const in_feat = try pbBuildFeatureDesc(allocator, spec.input_name, &.{ spec.batch_size, spec.hidden_size }, true);
    defer allocator.free(in_feat);
    const out_feat = try pbBuildFeatureDesc(allocator, spec.output_name, &.{ spec.batch_size, spec.out_experts }, true);
    defer allocator.free(out_feat);

    var model_desc: std.ArrayList(u8) = .empty;
    defer model_desc.deinit(allocator);
    try pbAppendBytesField(allocator, &model_desc, 1, in_feat);
    try pbAppendBytesField(allocator, &model_desc, 10, out_feat);

    var nn: std.ArrayList(u8) = .empty;
    defer nn.deinit(allocator);
    try pbAppendVarintField(allocator, &nn, 5, 1);

    if (spec.bottleneck_dim == 0) {
        if (spec.weights_layer2 != null) return error.InvalidSpec;
        try pbAppendBatchedMatmulConstLayer(
            allocator,
            &nn,
            "router_head",
            spec.input_name,
            spec.output_name,
            spec.hidden_size,
            spec.out_experts,
            spec.weights_layer1,
        );
    } else {
        const w2 = spec.weights_layer2 orelse return error.InvalidSpec;
        try pbAppendBatchedMatmulConstLayer(
            allocator,
            &nn,
            "mlp_fc1",
            spec.input_name,
            "h_pre",
            spec.hidden_size,
            spec.bottleneck_dim,
            spec.weights_layer1,
        );
        const h_act_name: []const u8 = switch (spec.activation) {
            .none => "h_pre",
            .relu => blk: {
                try pbAppendActivationReluLayer(allocator, &nn, "mlp_relu", "h_pre", "h_act");
                break :blk "h_act";
            },
            .silu => blk: {
                try pbAppendActivationSigmoidLayer(allocator, &nn, "mlp_sig", "h_pre", "h_sig");
                try pbAppendLayer(allocator, &nn, "mlp_silu", &.{ "h_pre", "h_sig" }, &.{"h_act"}, 900, &.{});
                break :blk "h_act";
            },
        };
        try pbAppendBatchedMatmulConstLayer(
            allocator,
            &nn,
            "mlp_fc2",
            h_act_name,
            spec.output_name,
            spec.bottleneck_dim,
            spec.out_experts,
            w2,
        );
    }

    var root: std.ArrayList(u8) = .empty;
    errdefer root.deinit(allocator);
    try pbAppendVarintField(allocator, &root, 1, 7);
    try pbAppendBytesField(allocator, &root, 2, model_desc.items);
    try pbAppendBytesField(allocator, &root, 500, nn.items);
    return root.toOwnedSlice(allocator);
}

test "default device is a unified-memory Apple GPU with Metal 4" {
    const device = try Device.open();
    defer device.close();
    const info = try device.info();
    try std.testing.expect(info.name().len > 0);
    try std.testing.expect(info.hasUnifiedMemory());
    try std.testing.expect(info.supportsMetal4());
    try std.testing.expect(info.appleGpuFamily() >= 7);
    try std.testing.expect(info.appleGpuFamily() <= 11);
    try std.testing.expectEqual(info.appleGpuFamily() >= 9, info.supportsApple9());
    try std.testing.expectEqual(info.appleGpuFamily() >= 10, info.supportsApple10());
    try std.testing.expectEqual(info.appleGpuFamily() >= 11, info.supportsApple11());
    try std.testing.expectEqual(info.appleGpuFamily() >= 10, info.supportsNaxTensorOps());
    if (std.mem.indexOf(u8, info.name(), "M4") != null) {
        try std.testing.expectEqual(@as(u8, 9), info.appleGpuFamily());
        try std.testing.expect(info.supportsApple9());
        try std.testing.expect(!info.supportsApple10());
        try std.testing.expect(!info.supportsApple11());
        try std.testing.expect(!info.supportsNaxTensorOps());
    }
    try std.testing.expect(info.recommendedMaxWorkingSetSize() > 0);
    try std.testing.expect(info.maxBufferLength() > 0);

    // Synthetic Apple7..Apple11 capability probe invariants.
    var fake: DeviceInfo = .{ .raw = std.mem.zeroes(c.ami_device_info) };
    fake.raw.gpu_apple_family = 9;
    fake.raw.apple_gpu_family = 9;
    fake.raw.supports_apple9 = 1;
    try std.testing.expectEqual(@as(u8, 9), fake.appleGpuFamily());
    try std.testing.expect(fake.supportsApple9());
    try std.testing.expect(!fake.supportsNaxTensorOps());

    fake.raw.gpu_apple_family = 10;
    fake.raw.apple_gpu_family = 10;
    fake.raw.supports_apple10 = 1;
    try std.testing.expectEqual(@as(u8, 10), fake.appleGpuFamily());
    try std.testing.expect(fake.supportsApple10());
    try std.testing.expect(fake.supportsNaxTensorOps());

    fake.raw.gpu_apple_family = 11;
    fake.raw.apple_gpu_family = 11;
    fake.raw.supports_apple11 = 1;
    try std.testing.expectEqual(@as(u8, 11), fake.appleGpuFamily());
    try std.testing.expect(fake.supportsApple11());
    try std.testing.expect(fake.supportsNaxTensorOps());
}

test "embedded metallib exposes smoke and M6.9 NAX TensorOps kernels without compiling PSOs" {
    const device = try Device.open();
    defer device.close();
    const library = try Library.openEmbedded(device);
    defer library.close();
    try std.testing.expect(try library.hasFunction("ami_smoke_iota"));
    try std.testing.expect(try library.hasFunction("ami_dense_gemm_nax"));
    try std.testing.expect(try library.hasFunction("ami_dequant_gemm_nax"));
    try std.testing.expect(try library.hasFunction("ami_fp4_e2m1_gemm_nax"));
    try std.testing.expect(!try library.hasFunction("ami_no_such_kernel"));
    try std.testing.expect(!try library.hasFunction("ami_nonexistent_kernel"));
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

test "KvTile enforces 16 KiB alignment, 64-byte row stride, and dimension limits" {
    const device = try Device.open();
    defer device.close();

    // Invalid geometries rejected at the bridge boundary.
    try std.testing.expectError(error.InvalidArgument, KvTile.init(device, 0, 64, .{}));
    try std.testing.expectError(error.InvalidArgument, KvTile.init(device, 16, 64, .{}));
    try std.testing.expectError(error.InvalidArgument, KvTile.init(device, 128, 0, .{}));
    try std.testing.expectError(error.InvalidArgument, KvTile.init(device, 128, MAX_TEXTURE_DIM + 1, .{}));
    try std.testing.expectError(error.InvalidArgument, KvTile.init(device, MAX_TEXTURE_DIM + 32, 8, .{}));

    // Qwen3-30B-A3B KV chunk tile: 4 KV heads * 512 tokens = 2048 rows, 128 fp16/row = 512 KiB.
    const tile = try KvTile.init(device, 128, 2048, .{ .alias_buffer = true });
    defer tile.close();

    const meta = try tile.info();
    try std.testing.expectEqual(@as(u32, 128), meta.rowF16());
    try std.testing.expectEqual(@as(u32, 2048), meta.rows());
    try std.testing.expectEqual(@as(u32, 256), meta.bytesPerRow());
    try std.testing.expectEqual(@as(u32, 0), meta.bytesPerRow() % 64);
    try std.testing.expectEqual(@as(u64, 512 * 1024), meta.dataBytes());
    try std.testing.expectEqual(@as(u64, 512 * 1024), meta.allocBytes());
    try std.testing.expectEqual(@as(u64, 0), meta.allocBytes() % page.APPLE_PAGE_SIZE);
    try std.testing.expect(meta.surfaceAllocSize() >= meta.allocBytes());
    try std.testing.expect(meta.rgbaResourceId() != 0);
    try std.testing.expect(meta.r16ResourceId() != 0);
    try std.testing.expect(meta.rgbaResourceId() != meta.r16ResourceId());
    try std.testing.expectEqual(meta.rgbaResourceId(), tile.rgbaResourceId());
    try std.testing.expectEqual(meta.r16ResourceId(), tile.r16ResourceId());
    try std.testing.expect(meta.hasAliasBuffer());
    try std.testing.expect(meta.aliasGpuAddress() != 0);

    // Non-page-multiple data size still rounds alloc_bytes up to 16 KiB.
    const small = try KvTile.init(device, 32, 10, .{});
    defer small.close();
    const small_meta = try small.info();
    try std.testing.expectEqual(@as(u64, 640), small_meta.dataBytes());
    try std.testing.expectEqual(@as(u64, page.APPLE_PAGE_SIZE), small_meta.allocBytes());
    try std.testing.expect(!small_meta.hasAliasBuffer());
    try std.testing.expectEqual(@as(?Buffer, null), small.aliasBuffer());
}

test "KvTile CPU lock and alias MTLBuffer share physical pages zero-copy with GPU" {
    const device = try Device.open();
    defer device.close();
    const library = try Library.openEmbedded(device);
    defer library.close();
    const iota_pipe = try Pipeline.init(device, library, "ami_smoke_iota");
    defer iota_pipe.close();

    // 128 fp16 * 64 rows = 16 KiB (4096 u32 words).
    const tile = try KvTile.init(device, 128, 64, .{ .alias_buffer = true });
    defer tile.close();
    const alias = tile.aliasBuffer() orelse return error.TestUnexpectedResult;

    {
        var guard = try tile.lock(false);
        errdefer guard.release();
        try std.testing.expect(std.mem.isAligned(@intFromPtr(guard.bytes.ptr), page.APPLE_PAGE_SIZE));
        try std.testing.expectEqual(@as(usize, 256), guard.bytes_per_row);
        @memset(guard.bytes, 0);
        try guard.unlock();
    }

    var compute = try Compute.init(device);
    defer compute.close();
    try compute.makeTilesResident(&.{tile});

    const threads: u64 = 4096;
    var frame = try compute.beginFrame(5000);
    errdefer frame.abort();
    try frame.encodeThreads(
        iota_pipe,
        &.{.{ .buffer = alias, .offset = 0 }},
        null,
        .{ threads, 1, 1 },
        .{ 256, 1, 1 },
        false,
    );
    const sig = try frame.commit();
    try compute.wait(sig, 5000);

    {
        var guard = try tile.lock(true);
        errdefer guard.release();
        const words: []const u32 = @ptrCast(guard.bytes);
        for (words, 0..) |word, i| {
            try std.testing.expectEqual(@as(u32, @intCast(i)) ^ 0xA5A5A5A5, word);
        }
        try guard.unlock();
    }

    try compute.removeTilesResident(&.{tile});
}

test "KvTile texture binding via MTL4ArgumentTable writes RGBA16Float zero-copy to IOSurface" {
    const device = try Device.open();
    defer device.close();
    const library = try Library.openEmbedded(device);
    defer library.close();
    const pipe = try Pipeline.init(device, library, "ami_qknorm_rope_kv_write");
    defer pipe.close();

    const Params = extern struct {
        num_q_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        pos: u32,
        chunk_tokens: u32,
        slot_in_chunk: u32,
        rms_eps: f32,
        rope_theta: f32,
        apply_qk_norm: u32,
        _pad0: u32 = 0,
        _pad1: u32 = 0,
        _pad2: u32 = 0,
    };

    const head_dim: u32 = 64;
    const chunk_tokens: u32 = 32;
    const slot: u32 = 5;

    const region = try page.WiredRegion.init(page.APPLE_PAGE_SIZE);
    defer region.deinit();
    const vals: []f16 = @alignCast(std.mem.bytesAsSlice(f16, region.bytes));
    @memset(vals, @floatCast(0.0));
    // Q [0..64), K [64..128), V [128..192)
    for (0..head_dim) |d| {
        vals[d] = @floatCast(@as(f32, @floatFromInt(d + 1)) * 0.25);
        vals[head_dim + d] = @floatCast(@as(f32, @floatFromInt(d + 1)) * 0.5);
        vals[2 * head_dim + d] = @floatCast(@as(f32, @floatFromInt(d + 1)) * -0.125);
    }

    const gpu_buf = try Buffer.wrap(device, region.bytes);
    defer gpu_buf.close();
    const k_tile = try KvTile.init(device, head_dim, chunk_tokens, .{});
    defer k_tile.close();
    const v_tile = try KvTile.init(device, head_dim, chunk_tokens, .{});
    defer v_tile.close();

    var compute = try Compute.init(device);
    defer compute.close();
    try compute.makeResident(&.{gpu_buf});
    try compute.makeTilesResident(&.{ k_tile, v_tile });

    // pos = 0, apply_qk_norm = 0 => RoPE is identity so K and V copy directly into row `slot`.
    const params: Params = .{
        .num_q_heads = 1,
        .num_kv_heads = 1,
        .head_dim = head_dim,
        .pos = 0,
        .chunk_tokens = chunk_tokens,
        .slot_in_chunk = slot,
        .rms_eps = 1e-6,
        .rope_theta = 1_000_000.0,
        .apply_qk_norm = 0,
    };
    const q_bind = BufferBinding{ .buffer = gpu_buf, .offset = 0 };
    const k_bind = BufferBinding{ .buffer = gpu_buf, .offset = head_dim * @sizeOf(f16) };
    const v_bind = BufferBinding{ .buffer = gpu_buf, .offset = 2 * head_dim * @sizeOf(f16) };

    var frame = try compute.beginFrame(5000);
    errdefer frame.abort();
    try frame.encodeThreadgroupsWithTextures(
        pipe,
        &.{ q_bind, k_bind, v_bind, q_bind, q_bind },
        &.{
            .{ .tile = k_tile, .view = .rgba16f },
            .{ .tile = v_tile, .view = .rgba16f },
        },
        std.mem.asBytes(&params),
        .{ 2, 1, 1 },
        .{ 32, 1, 1 },
        false,
    );
    const sig = try frame.commit();
    try compute.wait(sig, 5000);

    var kg = try k_tile.lock(true);
    errdefer kg.release();
    var vg = try v_tile.lock(true);
    errdefer vg.release();
    const k_row = kg.f16Slice()[slot * head_dim .. (slot + 1) * head_dim];
    const v_row = vg.f16Slice()[slot * head_dim .. (slot + 1) * head_dim];
    for (0..head_dim) |d| {
        try std.testing.expectEqual(vals[head_dim + d], k_row[d]);
        try std.testing.expectEqual(vals[2 * head_dim + d], v_row[d]);
    }
    try vg.unlock();
    try kg.unlock();

    try compute.removeTilesResident(&.{ k_tile, v_tile });
    try compute.removeResident(&.{gpu_buf});
}

test "UNIFORM_SLOTS_PER_FRAME accommodates 48-layer tree verification and in-CB drafting" {
    try std.testing.expectEqual(@as(u32, 16384), UNIFORM_SLOTS_PER_FRAME);
}

test "CoreML ANE zero-copy IOSurface conv1x1 and innerProduct match CPU reference with zero-copy outputBackings" {
    const gpa = std.testing.allocator;
    const device = try Device.open();
    defer device.close();

    // Case 1: 1x1 convolution [1, C_in=64, 1, S=32] -> [1, C_out=64, 1, S=32]
    // Native ANE layout with 64-byte aligned IOSurface rows (32 f16 = 64 B).
    {
        const c_in: u32 = 64;
        const c_out: u32 = 64;
        const s_dim: u32 = 32;

        const w = try gpa.alloc(f16, @as(usize, c_out) * c_in);
        defer gpa.free(w);
        for (0..c_out) |co| {
            for (0..c_in) |ci| {
                const v: i32 = @as(i32, @intCast((co * 13 + ci * 7 + 3) % 19)) - 9;
                w[co * c_in + ci] = @floatCast(@as(f32, @floatFromInt(v)) * 0.03125);
            }
        }

        const spec_bytes = try emitCoreMlLinearModel(gpa, .{
            .op = .conv1x1,
            .batch_or_spatial = s_dim,
            .layers = &.{.{ .c_in = c_in, .c_out = c_out, .weights_f16 = w }},
        });
        defer gpa.free(spec_bytes);

        const model = try AneModel.loadSpecData(spec_bytes, .cpu_and_neural_engine);
        defer model.close();
        const minfo = try model.info();
        try std.testing.expect(minfo.totalLayers() >= 1);
        if ((try device.info()).aneCoreCount() > 0) {
            try std.testing.expect(minfo.usesNeuralEngine());
        }

        const in_tile = try KvTile.init(device, s_dim, c_in, .{ .alias_buffer = true });
        defer in_tile.close();
        const out_tile = try KvTile.init(device, s_dim, c_out, .{ .alias_buffer = true });
        defer out_tile.close();

        // Shape validation rejects mismatched tile geometry.
        try std.testing.expectError(error.InvalidArgument, AneTensor.fromKvTile(in_tile, &.{ c_in, 1, s_dim + 32 }));
        try std.testing.expectError(error.InvalidArgument, AneTensor.fromKvTile(in_tile, &.{}));

        const in_tensor = try AneTensor.fromKvTile(in_tile, &.{ c_in, 1, s_dim });
        defer in_tensor.close();
        const out_tensor = try AneTensor.fromKvTile(out_tile, &.{ c_out, 1, s_dim });
        defer out_tensor.close();

        const plan = try AnePlan.init(
            &.{.{ .name = "x", .tensor = in_tensor }},
            &.{.{ .name = "y", .tensor = out_tensor }},
        );
        defer plan.close();

        const x_host = try gpa.alloc(f16, @as(usize, c_in) * s_dim);
        defer gpa.free(x_host);
        {
            var g = try in_tile.lock(false);
            errdefer g.release();
            const dst = g.f16Slice();
            for (0..c_in) |ci| {
                for (0..s_dim) |si| {
                    const v: i32 = @as(i32, @intCast((ci * 11 + si * 5 + 1) % 23)) - 11;
                    const h: f16 = @floatCast(@as(f32, @floatFromInt(v)) * 0.0625);
                    dst[ci * s_dim + si] = h;
                    x_host[ci * s_dim + si] = h;
                }
            }
            try g.unlock();
        }
        {
            var g = try out_tile.lock(false);
            errdefer g.release();
            @memset(g.bytes, 0);
            try g.unlock();
        }

        const pred_info = try model.predict(plan);
        try std.testing.expectEqual(@as(u32, 1), pred_info.outputsTotal());
        try std.testing.expect(pred_info.allOutputsZeroCopy());

        // Read output directly from out_tile's IOSurface pages and verify W @ X.
        var og = try out_tile.lock(true);
        errdefer og.release();
        const got_slice = og.f16Slice();
        for (0..c_out) |co| {
            for (0..s_dim) |si| {
                var acc: f32 = 0.0;
                for (0..c_in) |ci| {
                    acc += @as(f32, @floatCast(w[co * c_in + ci])) *
                        @as(f32, @floatCast(x_host[ci * s_dim + si]));
                }
                const got: f32 = @floatCast(got_slice[co * s_dim + si]);
                const tol: f32 = @max(0.02, @abs(acc) * 0.02);
                try std.testing.expectApproxEqAbs(acc, got, tol);
            }
        }
        try og.unlock();
    }

    // Case 2: innerProduct [M=2, C_in=64] -> [M=2, C_out=64] + loadPath from disk
    {
        const m_dim: u32 = 2;
        const c_in: u32 = 64;
        const c_out: u32 = 64;

        const w = try gpa.alloc(f16, @as(usize, c_out) * c_in);
        defer gpa.free(w);
        for (0..c_out) |co| {
            for (0..c_in) |ci| {
                const v: i32 = @as(i32, @intCast((co * 5 + ci * 9 + 7) % 17)) - 8;
                w[co * c_in + ci] = @floatCast(@as(f32, @floatFromInt(v)) * 0.03125);
            }
        }

        const spec_bytes = try emitCoreMlLinearModel(gpa, .{
            .op = .inner_product,
            .batch_or_spatial = m_dim,
            .layers = &.{.{ .c_in = c_in, .c_out = c_out, .weights_f16 = w }},
        });
        defer gpa.free(spec_bytes);

        var path_buf: [512]u8 = undefined;
        const path_z = try paths.tmpPathFmtZ(&path_buf, "aminfer_test_ip_{d}.mlmodel", .{std.c.getpid()});
        const fd = std.c.open(path_z.ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o644));
        try std.testing.expect(fd >= 0);
        defer _ = std.c.unlink(path_z.ptr);
        {
            defer _ = std.c.close(fd);
            var written: usize = 0;
            while (written < spec_bytes.len) {
                const n = std.c.write(fd, spec_bytes.ptr + written, spec_bytes.len - written);
                try std.testing.expect(n > 0);
                written += @intCast(n);
            }
        }

        const model = try AneModel.loadPath(path_z, .cpu_and_neural_engine);
        defer model.close();

        const in_tile = try KvTile.init(device, c_in, m_dim, .{});
        defer in_tile.close();
        const out_tile = try KvTile.init(device, c_out, m_dim, .{});
        defer out_tile.close();

        const in_tensor = try AneTensor.fromKvTile(in_tile, &.{ m_dim, c_in });
        defer in_tensor.close();
        const out_tensor = try AneTensor.fromKvTile(out_tile, &.{ m_dim, c_out });
        defer out_tensor.close();

        const x_host = try gpa.alloc(f16, @as(usize, m_dim) * c_in);
        defer gpa.free(x_host);
        {
            var g = try in_tile.lock(false);
            errdefer g.release();
            const dst = g.f16Slice();
            for (0..m_dim) |mi| {
                for (0..c_in) |ci| {
                    const v: i32 = @as(i32, @intCast((mi * 17 + ci * 3 + 2) % 21)) - 10;
                    const h: f16 = @floatCast(@as(f32, @floatFromInt(v)) * 0.0625);
                    dst[mi * c_in + ci] = h;
                    x_host[mi * c_in + ci] = h;
                }
            }
            try g.unlock();
        }

        const pred_info = try model.predictOneShot(
            &.{.{ .name = "x", .tensor = in_tensor }},
            &.{.{ .name = "y", .tensor = out_tensor }},
        );
        try std.testing.expectEqual(@as(u32, 1), pred_info.outputsTotal());

        var og = try out_tile.lock(true);
        errdefer og.release();
        const got_slice = og.f16Slice();
        for (0..m_dim) |mi| {
            for (0..c_out) |co| {
                var acc: f32 = 0.0;
                for (0..c_in) |ci| {
                    acc += @as(f32, @floatCast(x_host[mi * c_in + ci])) *
                        @as(f32, @floatCast(w[co * c_in + ci]));
                }
                const got: f32 = @floatCast(got_slice[mi * c_out + co]);
                const tol: f32 = @max(0.02, @abs(acc) * 0.02);
                try std.testing.expectApproxEqAbs(acc, got, tol);
            }
        }
        try og.unlock();
    }
}

fn fillTestF16(dst: []f16, seed: usize, scale: f32) void {
    for (dst, 0..) |*v, i| {
        const val: i32 = @as(i32, @intCast((i * 13 + seed * 29 + 5) % 19)) - 9;
        v.* = @floatCast(@as(f32, @floatFromInt(val)) * scale);
    }
}

test "emitCoreMlAttentionBlock multi-input/multi-output GQA matches CPU fp32 reference with 3/3 zero-copy KvTiles on ANE" {
    const gpa = std.testing.allocator;
    const device = try Device.open();
    defer device.close();
    const dev_info = try device.info();

    const h: u32 = 64;
    const qh: u32 = 4;
    const kvh: u32 = 2;
    const d: u32 = 32;
    const s: u32 = 32;
    const q_dim: u32 = qh * d;
    const kv_dim: u32 = kvh * d;
    const gqa: u32 = qh / kvh;
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(d)));

    const w_q = try gpa.alloc(f16, @as(usize, q_dim) * h);
    defer gpa.free(w_q);
    const w_k = try gpa.alloc(f16, @as(usize, kv_dim) * h);
    defer gpa.free(w_k);
    const w_v = try gpa.alloc(f16, @as(usize, kv_dim) * h);
    defer gpa.free(w_v);
    const w_o = try gpa.alloc(f16, @as(usize, h) * q_dim);
    defer gpa.free(w_o);
    fillTestF16(w_q, 1, 0.03125);
    fillTestF16(w_k, 2, 0.03125);
    fillTestF16(w_v, 3, 0.03125);
    fillTestF16(w_o, 4, 0.03125);

    for ([_]u32{ 32, 64 }) |b_len| {
        const past = b_len - s;
        const spec_bytes = try emitCoreMlAttentionBlock(gpa, .{
            .hidden_size = h,
            .num_q_heads = qh,
            .num_kv_heads = kvh,
            .head_dim = d,
            .chunk_tokens = s,
            .kv_bucket_tokens = b_len,
            .scale = scale,
            .use_conv_proj = true,
            .w_q = w_q,
            .w_k = w_k,
            .w_v = w_v,
            .w_o = w_o,
        });
        defer gpa.free(spec_bytes);

        const model = try AneModel.loadSpecData(spec_bytes, .cpu_and_neural_engine);
        defer model.close();
        const minfo = try model.info();
        try std.testing.expect(minfo.totalLayers() >= 15);
        if (dev_info.aneCoreCount() > 0) {
            try std.testing.expect(minfo.usesNeuralEngine());
        }

        const counts = try model.ioCounts();
        try std.testing.expectEqual(if (past > 0) @as(u32, 4) else @as(u32, 2), counts.inputs);
        try std.testing.expectEqual(@as(u32, 3), counts.outputs);
        var name_buf: [64]u8 = undefined;
        try std.testing.expectEqualStrings("attn_out", try model.ioName(true, 0, &name_buf));
        try std.testing.expectEqualStrings("k_tile_out", try model.ioName(true, 1, &name_buf));
        try std.testing.expectEqualStrings("v_tile_out", try model.ioName(true, 2, &name_buf));

        const x_tile = try KvTile.init(device, h, s, .{ .alias_buffer = true });
        defer x_tile.close();
        const mask_tile = try KvTile.init(device, b_len, s, .{});
        defer mask_tile.close();
        const out_tile = try KvTile.init(device, h, s, .{ .alias_buffer = true });
        defer out_tile.close();
        const k_out_tile = try KvTile.init(device, d, kvh * s, .{});
        defer k_out_tile.close();
        const v_out_tile = try KvTile.init(device, d, kvh * s, .{});
        defer v_out_tile.close();

        const x_host = try gpa.alloc(f16, @as(usize, s) * h);
        defer gpa.free(x_host);
        fillTestF16(x_host, 10 + b_len, 0.0625);
        {
            var g = try x_tile.lock(false);
            errdefer g.release();
            @memcpy(g.f16Slice()[0..x_host.len], x_host);
            try g.unlock();
        }

        const mask_host = try gpa.alloc(f16, @as(usize, s) * b_len);
        defer gpa.free(mask_host);
        const valid_past: u32 = if (past > 0) past - 5 else 0;
        for (0..s) |row| {
            for (0..b_len) |col| {
                const allow = if (col < past)
                    (col < valid_past)
                else
                    (col - past <= row);
                mask_host[row * b_len + col] = if (allow) @as(f16, @floatCast(0.0)) else @as(f16, @floatCast(-10000.0));
            }
        }
        {
            var g = try mask_tile.lock(false);
            errdefer g.release();
            @memcpy(g.f16Slice()[0..mask_host.len], mask_host);
            try g.unlock();
        }

        const x_t = try AneTensor.fromKvTile(x_tile, &.{ 1, s, h });
        defer x_t.close();
        const mask_t = try AneTensor.fromKvTile(mask_tile, &.{ 1, 1, s, b_len });
        defer mask_t.close();
        const out_t = try AneTensor.fromKvTile(out_tile, &.{ 1, s, h });
        defer out_t.close();
        const ko_t = try AneTensor.fromKvTile(k_out_tile, &.{ 1, kvh, s, d });
        defer ko_t.close();
        const vo_t = try AneTensor.fromKvTile(v_out_tile, &.{ 1, kvh, s, d });
        defer vo_t.close();

        var kp_tile_opt: ?KvTile = null;
        var vp_tile_opt: ?KvTile = null;
        var kp_t_opt: ?AneTensor = null;
        var vp_t_opt: ?AneTensor = null;
        defer if (vp_t_opt) |t| t.close();
        defer if (kp_t_opt) |t| t.close();
        defer if (vp_tile_opt) |t| t.close();
        defer if (kp_tile_opt) |t| t.close();

        const kp_host = try gpa.alloc(f16, @as(usize, kvh) * past * d);
        defer gpa.free(kp_host);
        const vp_host = try gpa.alloc(f16, @as(usize, kvh) * past * d);
        defer gpa.free(vp_host);

        var in_binds: [4]AneBinding = undefined;
        var in_count: usize = 0;
        in_binds[in_count] = .{ .name = "x_in", .tensor = x_t };
        in_count += 1;
        if (past > 0) {
            fillTestF16(kp_host, 21, 0.0625);
            fillTestF16(vp_host, 22, 0.0625);
            kp_tile_opt = try KvTile.init(device, d, kvh * past, .{});
            vp_tile_opt = try KvTile.init(device, d, kvh * past, .{});
            {
                var g = try kp_tile_opt.?.lock(false);
                errdefer g.release();
                @memcpy(g.f16Slice()[0..kp_host.len], kp_host);
                try g.unlock();
            }
            {
                var g = try vp_tile_opt.?.lock(false);
                errdefer g.release();
                @memcpy(g.f16Slice()[0..vp_host.len], vp_host);
                try g.unlock();
            }
            kp_t_opt = try AneTensor.fromKvTile(kp_tile_opt.?, &.{ 1, kvh, past, d });
            vp_t_opt = try AneTensor.fromKvTile(vp_tile_opt.?, &.{ 1, kvh, past, d });
            in_binds[in_count] = .{ .name = "k_past", .tensor = kp_t_opt.? };
            in_count += 1;
            in_binds[in_count] = .{ .name = "v_past", .tensor = vp_t_opt.? };
            in_count += 1;
        }
        in_binds[in_count] = .{ .name = "mask", .tensor = mask_t };
        in_count += 1;

        const out_binds = [_]AneBinding{
            .{ .name = "attn_out", .tensor = out_t },
            .{ .name = "k_tile_out", .tensor = ko_t },
            .{ .name = "v_tile_out", .tensor = vo_t },
        };
        const plan = try AnePlan.init(in_binds[0..in_count], &out_binds);
        defer plan.close();

        const pinfo = try model.predict(plan);
        try std.testing.expectEqual(@as(u32, 3), pinfo.outputsTotal());
        try std.testing.expect(pinfo.allOutputsZeroCopy());

        // CPU fp64 reference: Q, K_cur, V_cur, concat past, causal GQA attention, O_proj.
        const q_ref = try gpa.alloc(f64, @as(usize, qh) * s * d);
        defer gpa.free(q_ref);
        const k_cur_ref = try gpa.alloc(f64, @as(usize, kvh) * s * d);
        defer gpa.free(k_cur_ref);
        const v_cur_ref = try gpa.alloc(f64, @as(usize, kvh) * s * d);
        defer gpa.free(v_cur_ref);

        for (0..s) |ti| {
            for (0..qh) |head| {
                for (0..d) |di| {
                    var acc: f64 = 0.0;
                    const row_w = head * d + di;
                    for (0..h) |ci| {
                        acc += @as(f64, @floatCast(x_host[ti * h + ci])) * @as(f64, @floatCast(w_q[row_w * h + ci]));
                    }
                    q_ref[(head * s + ti) * d + di] = acc;
                }
            }
            for (0..kvh) |head| {
                for (0..d) |di| {
                    var ak: f64 = 0.0;
                    var av: f64 = 0.0;
                    const row_w = head * d + di;
                    for (0..h) |ci| {
                        const xv: f64 = @floatCast(x_host[ti * h + ci]);
                        ak += xv * @as(f64, @floatCast(w_k[row_w * h + ci]));
                        av += xv * @as(f64, @floatCast(w_v[row_w * h + ci]));
                    }
                    k_cur_ref[(head * s + ti) * d + di] = ak;
                    v_cur_ref[(head * s + ti) * d + di] = av;
                }
            }
        }

        // Verify k_tile_out and v_tile_out directly on IOSurface pages.
        {
            var kg = try k_out_tile.lock(true);
            errdefer kg.release();
            var vg = try v_out_tile.lock(true);
            errdefer vg.release();
            for (k_cur_ref, kg.f16Slice()[0..k_cur_ref.len]) |exp, got_h| {
                try std.testing.expectApproxEqAbs(@as(f32, @floatCast(exp)), @as(f32, @floatCast(got_h)), 0.02);
            }
            for (v_cur_ref, vg.f16Slice()[0..v_cur_ref.len]) |exp, got_h| {
                try std.testing.expectApproxEqAbs(@as(f32, @floatCast(exp)), @as(f32, @floatCast(got_h)), 0.02);
            }
            try vg.unlock();
            try kg.unlock();
        }

        // Compute CPU GQA context & O projection and compare with attn_out.
        const ctx_ref = try gpa.alloc(f64, @as(usize, s) * q_dim);
        defer gpa.free(ctx_ref);
        const scores = try gpa.alloc(f64, b_len);
        defer gpa.free(scores);

        for (0..qh) |head| {
            const kvh_i = head / gqa;
            for (0..s) |ti| {
                var max_s: f64 = -std.math.inf(f64);
                for (0..b_len) |bi| {
                    var dot_v: f64 = 0.0;
                    for (0..d) |di| {
                        const qv = q_ref[(head * s + ti) * d + di];
                        const kv = if (bi < past)
                            @as(f64, @floatCast(kp_host[(kvh_i * past + bi) * d + di]))
                        else
                            k_cur_ref[(kvh_i * s + (bi - past)) * d + di];
                        dot_v += qv * kv;
                    }
                    const sc = dot_v * @as(f64, scale) + @as(f64, @floatCast(mask_host[ti * b_len + bi]));
                    scores[bi] = sc;
                    if (sc > max_s) max_s = sc;
                }
                var sum_e: f64 = 0.0;
                for (0..b_len) |bi| {
                    scores[bi] = @exp(scores[bi] - max_s);
                    sum_e += scores[bi];
                }
                for (0..d) |di| {
                    var acc: f64 = 0.0;
                    for (0..b_len) |bi| {
                        const vv = if (bi < past)
                            @as(f64, @floatCast(vp_host[(kvh_i * past + bi) * d + di]))
                        else
                            v_cur_ref[(kvh_i * s + (bi - past)) * d + di];
                        acc += (scores[bi] / sum_e) * vv;
                    }
                    ctx_ref[ti * q_dim + head * d + di] = acc;
                }
            }
        }

        var og = try out_tile.lock(true);
        errdefer og.release();
        const got_out = og.f16Slice();
        for (0..s) |ti| {
            for (0..h) |ci| {
                var acc: f64 = 0.0;
                for (0..q_dim) |qi| {
                    acc += ctx_ref[ti * q_dim + qi] * @as(f64, @floatCast(w_o[ci * q_dim + qi]));
                }
                const got: f32 = @floatCast(got_out[ti * h + ci]);
                const exp: f32 = @floatCast(acc);
                try std.testing.expectApproxEqAbs(exp, got, @max(0.025, @abs(exp) * 0.03));
            }
        }
        try og.unlock();
    }
}

test "emitCoreMlQkvProjectionBlock, emitCoreMlRouterPredictor (linear & SiLU MLP), and .mlpackage loadPath work zero-copy on ANE" {
    const gpa = std.testing.allocator;
    const device = try Device.open();
    defer device.close();
    const dev_info = try device.info();

    // 1. Fixed-shape QKV projection block
    {
        const h: u32 = 64;
        const qh: u32 = 4;
        const kvh: u32 = 2;
        const d: u32 = 32;
        const s: u32 = 32;
        const q_dim: u32 = qh * d;
        const kv_dim: u32 = kvh * d;

        const w_q = try gpa.alloc(f16, @as(usize, q_dim) * h);
        defer gpa.free(w_q);
        const w_k = try gpa.alloc(f16, @as(usize, kv_dim) * h);
        defer gpa.free(w_k);
        const w_v = try gpa.alloc(f16, @as(usize, kv_dim) * h);
        defer gpa.free(w_v);
        fillTestF16(w_q, 5, 0.03125);
        fillTestF16(w_k, 6, 0.03125);
        fillTestF16(w_v, 7, 0.03125);

        const spec_bytes = try emitCoreMlQkvProjectionBlock(gpa, .{
            .hidden_size = h,
            .num_q_heads = qh,
            .num_kv_heads = kvh,
            .head_dim = d,
            .chunk_tokens = s,
            .w_q = w_q,
            .w_k = w_k,
            .w_v = w_v,
        });
        defer gpa.free(spec_bytes);

        const model = try AneModel.loadSpecData(spec_bytes, .cpu_and_neural_engine);
        defer model.close();
        if (dev_info.aneCoreCount() > 0) try std.testing.expect((try model.info()).usesNeuralEngine());

        const x_tile = try KvTile.init(device, h, s, .{});
        defer x_tile.close();
        const q_tile = try KvTile.init(device, q_dim, s, .{});
        defer q_tile.close();
        const k_tile = try KvTile.init(device, d, kvh * s, .{});
        defer k_tile.close();
        const v_tile = try KvTile.init(device, d, kvh * s, .{});
        defer v_tile.close();

        const x_host = try gpa.alloc(f16, @as(usize, s) * h);
        defer gpa.free(x_host);
        fillTestF16(x_host, 8, 0.0625);
        {
            var g = try x_tile.lock(false);
            errdefer g.release();
            @memcpy(g.f16Slice()[0..x_host.len], x_host);
            try g.unlock();
        }

        const x_t = try AneTensor.fromKvTile(x_tile, &.{ 1, s, h });
        defer x_t.close();
        const q_t = try AneTensor.fromKvTile(q_tile, &.{ 1, s, q_dim });
        defer q_t.close();
        const k_t = try AneTensor.fromKvTile(k_tile, &.{ 1, kvh, s, d });
        defer k_t.close();
        const v_t = try AneTensor.fromKvTile(v_tile, &.{ 1, kvh, s, d });
        defer v_t.close();

        const plan = try AnePlan.init(
            &.{.{ .name = "x_in", .tensor = x_t }},
            &.{
                .{ .name = "q_out", .tensor = q_t },
                .{ .name = "k_tile_out", .tensor = k_t },
                .{ .name = "v_tile_out", .tensor = v_t },
            },
        );
        defer plan.close();

        const pinfo = try model.predict(plan);
        try std.testing.expectEqual(@as(u32, 3), pinfo.outputsTotal());
        try std.testing.expect(pinfo.allOutputsZeroCopy());

        var qg = try q_tile.lock(true);
        errdefer qg.release();
        const q_got = qg.f16Slice();
        for (0..s) |ti| {
            for (0..q_dim) |qi| {
                var acc: f32 = 0.0;
                for (0..h) |ci| {
                    acc += @as(f32, @floatCast(x_host[ti * h + ci])) * @as(f32, @floatCast(w_q[qi * h + ci]));
                }
                try std.testing.expectApproxEqAbs(acc, @as(f32, @floatCast(q_got[ti * q_dim + qi])), 0.02);
            }
        }
        try qg.unlock();
    }

    // 2. Multi-head router predictor (1-layer linear + 2-layer SiLU bottleneck MLP) + real .mlpackage loadPath
    {
        // 2a. 1-layer linear predictor (in-memory)
        {
            const m: u32 = 4;
            const h: u32 = 64;
            const out_e: u32 = 64;
            const w1 = try gpa.alloc(f16, @as(usize, out_e) * h);
            defer gpa.free(w1);
            fillTestF16(w1, 9, 0.0625);

            const spec_lin = try emitCoreMlRouterPredictor(gpa, .{
                .batch_size = m,
                .hidden_size = h,
                .out_experts = out_e,
                .weights_layer1 = w1,
            });
            defer gpa.free(spec_lin);

            const model = try AneModel.loadSpecData(spec_lin, .cpu_and_neural_engine);
            defer model.close();
            if (dev_info.aneCoreCount() > 0) try std.testing.expect((try model.info()).usesNeuralEngine());

            const x_tile = try KvTile.init(device, h, m, .{});
            defer x_tile.close();
            const y_tile = try KvTile.init(device, out_e, m, .{});
            defer y_tile.close();

            const x_host = try gpa.alloc(f16, @as(usize, m) * h);
            defer gpa.free(x_host);
            fillTestF16(x_host, 10, 0.0625);
            {
                var g = try x_tile.lock(false);
                errdefer g.release();
                @memcpy(g.f16Slice()[0..x_host.len], x_host);
                try g.unlock();
            }

            const x_t = try AneTensor.fromKvTile(x_tile, &.{ m, h });
            defer x_t.close();
            const y_t = try AneTensor.fromKvTile(y_tile, &.{ m, out_e });
            defer y_t.close();
            const plan = try AnePlan.init(&.{.{ .name = "x", .tensor = x_t }}, &.{.{ .name = "logits", .tensor = y_t }});
            defer plan.close();

            const pinfo = try model.predict(plan);
            try std.testing.expect(pinfo.allOutputsZeroCopy());

            var yg = try y_tile.lock(true);
            errdefer yg.release();
            const y_got = yg.f16Slice();
            for (0..m) |mi| {
                for (0..out_e) |ei| {
                    var acc: f64 = 0.0;
                    for (0..h) |ci| {
                        acc += @as(f64, @floatCast(x_host[mi * h + ci])) * @as(f64, @floatCast(w1[ei * h + ci]));
                    }
                    const exp: f32 = @floatCast(acc);
                    try std.testing.expectApproxEqAbs(exp, @as(f32, @floatCast(y_got[mi * out_e + ei])), @max(0.02, @abs(exp) * 0.02));
                }
            }
            try yg.unlock();
        }

        // 2b. 2-layer SiLU bottleneck MLP loaded from real .mlpackage directory on disk
        // (sized at realistic hidden=2048 so CoreML's planner places all 4 layers on ANE)
        const m: u32 = 4;
        const h: u32 = 2048;
        const bn: u32 = 256;
        const out_e: u32 = 256; // 2 lookahead layers * 128 experts

        const w1 = try gpa.alloc(f16, @as(usize, bn) * h);
        defer gpa.free(w1);
        const w2 = try gpa.alloc(f16, @as(usize, out_e) * bn);
        defer gpa.free(w2);
        fillTestF16(w1, 11, 0.0078125);
        fillTestF16(w2, 12, 0.015625);

        const spec_bytes = try emitCoreMlRouterPredictor(gpa, .{
            .batch_size = m,
            .hidden_size = h,
            .out_experts = out_e,
            .bottleneck_dim = bn,
            .activation = .silu,
            .weights_layer1 = w1,
            .weights_layer2 = w2,
        });
        defer gpa.free(spec_bytes);

        // Write a real CoreML .mlpackage directory bundle in TMPDIR and load via AneModel.loadPath!
        const pid = std.c.getpid();
        var pkg_buf: [512]u8 = undefined;
        var data_dir_buf: [512]u8 = undefined;
        var coreml_dir_buf: [512]u8 = undefined;
        var manifest_buf: [512]u8 = undefined;
        var model_buf: [512]u8 = undefined;

        const pkg_z = try paths.tmpPathFmtZ(&pkg_buf, "aminfer_test_pred_{d}.mlpackage", .{pid});
        const pkg_str: []const u8 = pkg_z;

        const data_dir = try std.fmt.bufPrint(data_dir_buf[0 .. data_dir_buf.len - 1], "{s}/Data", .{pkg_str});
        data_dir_buf[data_dir.len] = 0;
        const coreml_dir = try std.fmt.bufPrint(coreml_dir_buf[0 .. coreml_dir_buf.len - 1], "{s}/Data/com.apple.CoreML", .{pkg_str});
        coreml_dir_buf[coreml_dir.len] = 0;
        const manifest_path = try std.fmt.bufPrint(manifest_buf[0 .. manifest_buf.len - 1], "{s}/Manifest.json", .{pkg_str});
        manifest_buf[manifest_path.len] = 0;
        const model_path = try std.fmt.bufPrint(model_buf[0 .. model_buf.len - 1], "{s}/Data/com.apple.CoreML/model.mlmodel", .{pkg_str});
        model_buf[model_path.len] = 0;

        _ = std.c.mkdir(pkg_z.ptr, 0o755);
        _ = std.c.mkdir(@ptrCast(data_dir_buf[0..data_dir.len :0].ptr), 0o755);
        _ = std.c.mkdir(@ptrCast(coreml_dir_buf[0..coreml_dir.len :0].ptr), 0o755);
        defer {
            _ = std.c.unlink(@ptrCast(model_buf[0..model_path.len :0].ptr));
            _ = std.c.unlink(@ptrCast(manifest_buf[0..manifest_path.len :0].ptr));
            _ = std.c.rmdir(@ptrCast(coreml_dir_buf[0..coreml_dir.len :0].ptr));
            _ = std.c.rmdir(@ptrCast(data_dir_buf[0..data_dir.len :0].ptr));
            _ = std.c.rmdir(pkg_z.ptr);
        }

        const manifest_json =
            \\{
            \\  "fileFormatVersion": "1.0.0",
            \\  "itemInfoEntries": {
            \\    "00000000-0000-0000-0000-000000000001": {
            \\      "author": "com.apple.CoreML",
            \\      "description": "CoreML Model Specification",
            \\      "name": "model.mlmodel",
            \\      "path": "com.apple.CoreML/model.mlmodel"
            \\    }
            \\  },
            \\  "rootModelIdentifier": "00000000-0000-0000-0000-000000000001"
            \\}
        ;
        inline for (.{
            .{ manifest_buf[0..manifest_path.len :0], manifest_json },
            .{ model_buf[0..model_path.len :0], spec_bytes },
        }) |pair| {
            const fd = std.c.open(pair[0].ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o644));
            try std.testing.expect(fd >= 0);
            defer _ = std.c.close(fd);
            var written: usize = 0;
            while (written < pair[1].len) {
                const n = std.c.write(fd, pair[1].ptr + written, pair[1].len - written);
                try std.testing.expect(n > 0);
                written += @intCast(n);
            }
        }

        const model = try AneModel.loadPath(pkg_z, .cpu_and_neural_engine);
        defer model.close();
        if (dev_info.aneCoreCount() > 0) try std.testing.expect((try model.info()).usesNeuralEngine());

        const counts = try model.ioCounts();
        try std.testing.expectEqual(@as(u32, 1), counts.inputs);
        try std.testing.expectEqual(@as(u32, 1), counts.outputs);
        var nbuf: [64]u8 = undefined;
        try std.testing.expectEqualStrings("x", try model.ioName(false, 0, &nbuf));
        try std.testing.expectEqualStrings("logits", try model.ioName(true, 0, &nbuf));

        const x_tile = try KvTile.init(device, h, m, .{});
        defer x_tile.close();
        const y_tile = try KvTile.init(device, out_e, m, .{});
        defer y_tile.close();

        const x_host = try gpa.alloc(f16, @as(usize, m) * h);
        defer gpa.free(x_host);
        fillTestF16(x_host, 13, 0.0625);
        {
            var g = try x_tile.lock(false);
            errdefer g.release();
            @memcpy(g.f16Slice()[0..x_host.len], x_host);
            try g.unlock();
        }

        const x_t = try AneTensor.fromKvTile(x_tile, &.{ m, h });
        defer x_t.close();
        const y_t = try AneTensor.fromKvTile(y_tile, &.{ m, out_e });
        defer y_t.close();
        const plan = try AnePlan.init(&.{.{ .name = "x", .tensor = x_t }}, &.{.{ .name = "logits", .tensor = y_t }});
        defer plan.close();

        const pinfo = try model.predict(plan);
        try std.testing.expectEqual(@as(u32, 1), pinfo.outputsTotal());
        try std.testing.expect(pinfo.allOutputsZeroCopy());

        var yg = try y_tile.lock(true);
        errdefer yg.release();
        const y_got = yg.f16Slice();
        var hidden_tmp: [bn]f64 = undefined;
        for (0..m) |mi| {
            for (0..bn) |bi| {
                var acc: f64 = 0.0;
                for (0..h) |ci| {
                    acc += @as(f64, @floatCast(x_host[mi * h + ci])) * @as(f64, @floatCast(w1[bi * h + ci]));
                }
                hidden_tmp[bi] = acc / (1.0 + @exp(-acc));
            }
            for (0..out_e) |ei| {
                var acc: f64 = 0.0;
                for (0..bn) |bi| {
                    acc += hidden_tmp[bi] * @as(f64, @floatCast(w2[ei * bn + bi]));
                }
                const exp: f32 = @floatCast(acc);
                const got: f32 = @floatCast(y_got[mi * out_e + ei]);
                try std.testing.expectApproxEqAbs(exp, got, @max(0.02, @abs(exp) * 0.02));
            }
        }
        try yg.unlock();
    }
}

test "accelerateSgemm matches fp32 scalar reference for transposed and non-transposed B" {
    const m: u32 = 5;
    const n: u32 = 7;
    const k: u32 = 13;
    var a: [m * k]f32 = undefined;
    var b_nk: [n * k]f32 = undefined;
    var b_kn: [k * n]f32 = undefined;
    for (&a, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 11)) - 5)) * 0.25;
    for (0..n) |ni| {
        for (0..k) |ki| {
            const val = @as(f32, @floatFromInt(@as(i32, @intCast((ni * 7 + ki * 3) % 13)) - 6)) * 0.125;
            b_nk[ni * k + ki] = val;
            b_kn[ki * n + ni] = val;
        }
    }

    var c_trans: [m * n]f32 = @splat(0.0);
    var c_notrans: [m * n]f32 = @splat(0.0);
    try accelerateSgemm(m, n, k, 1.0, &a, &b_nk, 0.0, &c_trans, true);
    try accelerateSgemm(m, n, k, 1.0, &a, &b_kn, 0.0, &c_notrans, false);

    for (0..m) |mi| {
        for (0..n) |ni| {
            var expected: f32 = 0.0;
            for (0..k) |ki| expected += a[mi * k + ki] * b_nk[ni * k + ki];
            try std.testing.expectApproxEqAbs(expected, c_trans[mi * n + ni], 1e-4);
            try std.testing.expectApproxEqAbs(expected, c_notrans[mi * n + ni], 1e-4);
        }
    }

    try std.testing.expectError(error.InvalidArgument, accelerateSgemm(0, n, k, 1.0, &a, &b_nk, 0.0, &c_trans, true));
    try std.testing.expectError(error.InvalidArgument, accelerateSgemm(m, n, k, 1.0, a[0..1], &b_nk, 0.0, &c_trans, true));
}

test "G20 Metal 4 Compute.dispatchIndirect executes non-zero/zero grids and validates alignment and bounds" {
    try std.testing.expectEqual(@as(usize, 12), @sizeOf(IndirectDispatchArgs));
    try std.testing.expectEqual(@as(usize, 4), @alignOf(IndirectDispatchArgs));

    const device = try Device.open();
    defer device.close();
    const lib = try Library.openEmbedded(device);
    defer lib.close();

    const iota_pipe = try Pipeline.init(device, lib, "ami_smoke_iota");
    defer iota_pipe.close();
    const scale_pipe = try Pipeline.initSpecialized(
        device,
        lib,
        "ami_smoke_scale_add",
        &.{.{ .index = 0, .value = .{ .u32 = 3 } }},
    );
    defer scale_pipe.close();

    const data_region = try page.WiredRegion.init(page.APPLE_PAGE_SIZE);
    defer data_region.deinit();
    const ind_region = try page.WiredRegion.init(page.APPLE_PAGE_SIZE);
    defer ind_region.deinit();

    const data_u32: []u32 = @alignCast(std.mem.bytesAsSlice(u32, data_region.bytes));
    @memset(data_u32, 0);

    // Place 4 IndirectDispatchArgs records at 16-byte stride inside `ind_region`.
    const ind_bytes = ind_region.bytes;
    @memset(ind_bytes, 0);
    const args0: *IndirectDispatchArgs = @ptrCast(@alignCast(ind_bytes.ptr + 0));
    const args1: *IndirectDispatchArgs = @ptrCast(@alignCast(ind_bytes.ptr + 16));
    const args_zero_y: *IndirectDispatchArgs = @ptrCast(@alignCast(ind_bytes.ptr + 32));
    const args_zero_all: *IndirectDispatchArgs = @ptrCast(@alignCast(ind_bytes.ptr + 48));
    args0.* = IndirectDispatchArgs.init(4, 1, 1); // 4 * 32 = 128 threads
    args1.* = IndirectDispatchArgs.init(2, 1, 1); // 2 * 32 = 64 threads
    args_zero_y.* = IndirectDispatchArgs.init(4, 0, 1); // 0 threadgroups (empty bucket pattern)
    args_zero_all.* = IndirectDispatchArgs.init(0, 0, 0); // 0 threadgroups

    const data_buf = try Buffer.wrap(device, data_region.bytes);
    defer data_buf.close();
    const ind_buf = try Buffer.wrap(device, ind_region.bytes);
    defer ind_buf.close();

    var compute = try Compute.initWithFrames(device, 2);
    defer compute.close();
    try compute.makeResident(&.{ data_buf, ind_buf });

    // 1. Calling dispatchIndirect outside an active frame must fail cleanly.
    try std.testing.expectError(error.InvalidArgument, compute.dispatchIndirect(
        iota_pipe,
        &.{.{ .buffer = data_buf, .offset = 0 }},
        null,
        .{ .buffer = ind_buf, .offset = 0 },
        .{ 32, 1, 1 },
        false,
    ));

    // 2. Execute chained non-zero indirect dispatches with intra-encoder barrier & uniform,
    //    plus zero-grid indirect dispatches (`{4, 0, 1}` and `{0, 0, 0}`) targeting untouched tail.
    const SmokeParams = extern struct { addend: u32 };
    const params: SmokeParams = .{ .addend = 11 };

    var frame = try compute.beginFrame(5000);
    errdefer frame.abort();

    // Alignment / bounds / zero-tg validation inside an active frame:
    try std.testing.expectError(error.InvalidArgument, frame.encodeThreadgroupsIndirect(
        iota_pipe,
        &.{.{ .buffer = data_buf, .offset = 0 }},
        null,
        .{ .buffer = ind_buf, .offset = 2 }, // unaligned (not multiple of 4)
        .{ 32, 1, 1 },
        false,
    ));
    try std.testing.expectError(error.InvalidArgument, frame.encodeThreadgroupsIndirect(
        iota_pipe,
        &.{.{ .buffer = data_buf, .offset = 0 }},
        null,
        .{ .buffer = ind_buf, .offset = page.APPLE_PAGE_SIZE - 8 }, // < 12 bytes remaining
        .{ 32, 1, 1 },
        false,
    ));
    try std.testing.expectError(error.InvalidArgument, frame.encodeThreadgroupsIndirect(
        iota_pipe,
        &.{.{ .buffer = data_buf, .offset = 0 }},
        null,
        .{ .buffer = ind_buf, .offset = page.APPLE_PAGE_SIZE }, // out of bounds
        .{ 32, 1, 1 },
        false,
    ));
    try std.testing.expectError(error.InvalidArgument, frame.encodeThreadgroupsIndirect(
        iota_pipe,
        &.{.{ .buffer = data_buf, .offset = 0 }},
        null,
        .{ .buffer = ind_buf, .offset = 0 },
        .{ 0, 1, 1 }, // zero threadgroup dim
        false,
    ));

    // Actual indirect dispatches:
    try frame.encodeThreadgroupsIndirect(
        iota_pipe,
        &.{.{ .buffer = data_buf, .offset = 0 }},
        null,
        .{ .buffer = ind_buf, .offset = 0 },
        .{ 32, 1, 1 },
        false,
    );
    try compute.dispatchIndirect(
        scale_pipe,
        &.{.{ .buffer = data_buf, .offset = 0 }},
        std.mem.asBytes(&params),
        .{ .buffer = ind_buf, .offset = 16 },
        .{ 32, 1, 1 },
        true,
    );
    // Zero-threadgroup indirect dispatches aimed at offset 512.. (elements 128..) must write nothing.
    try frame.encodeThreadgroupsIndirect(
        iota_pipe,
        &.{.{ .buffer = data_buf, .offset = 128 * @sizeOf(u32) }},
        null,
        .{ .buffer = ind_buf, .offset = 32 },
        .{ 32, 1, 1 },
        true,
    );
    try frame.encodeThreadgroupsIndirect(
        iota_pipe,
        &.{.{ .buffer = data_buf, .offset = 128 * @sizeOf(u32) }},
        null,
        .{ .buffer = ind_buf, .offset = 48 },
        .{ 32, 1, 1 },
        false,
    );

    const sig = try frame.commit();
    try compute.wait(sig, 5000);

    // Closed frame must reject further indirect dispatches.
    try std.testing.expectError(error.InvalidArgument, frame.encodeThreadgroupsIndirect(
        iota_pipe,
        &.{.{ .buffer = data_buf, .offset = 0 }},
        null,
        .{ .buffer = ind_buf, .offset = 0 },
        .{ 32, 1, 1 },
        false,
    ));

    for (0..64) |i| {
        const expected = (@as(u32, @intCast(i)) ^ 0xA5A5_A5A5) *% 3 +% 11;
        try std.testing.expectEqual(expected, data_u32[i]);
    }
    for (64..128) |i| {
        const expected = @as(u32, @intCast(i)) ^ 0xA5A5_A5A5;
        try std.testing.expectEqual(expected, data_u32[i]);
    }
    for (128..256) |i| {
        try std.testing.expectEqual(@as(u32, 0), data_u32[i]);
    }
}

test "F15.4 & L1: Compute.waitIdle, waitIdleBestEffort, and cross-frame queue barrier ordering" {
    const device = try Device.open();
    defer device.close();
    const lib = try Library.openEmbedded(device);
    defer lib.close();

    const iota_pipe = try Pipeline.init(device, lib, "ami_smoke_iota");
    defer iota_pipe.close();
    const scale_pipe = try Pipeline.initSpecialized(
        device,
        lib,
        "ami_smoke_scale_add",
        &.{.{ .index = 0, .value = .{ .u32 = 2 } }},
    );
    defer scale_pipe.close();

    const region = try page.WiredRegion.init(page.APPLE_PAGE_SIZE);
    defer region.deinit();
    const words: []u32 = @alignCast(std.mem.bytesAsSlice(u32, region.bytes));
    @memset(words, 0);

    const buf = try Buffer.wrap(device, region.bytes);
    defer buf.close();

    var compute = try Compute.initWithFrames(device, 2);
    defer compute.close();
    try compute.makeResident(&.{buf});

    // Frame 1 writes 64 words; Frame 2 immediately reads & scales them without explicit caller wait.
    var f1 = try compute.beginFrame(5000);
    errdefer f1.abort();
    try f1.encodeThreads(iota_pipe, &.{.{ .buffer = buf, .offset = 0 }}, null, .{ 64, 1, 1 }, .{ 32, 1, 1 }, false);
    _ = try f1.commit();

    const SmokeParams = extern struct { addend: u32 };
    const params: SmokeParams = .{ .addend = 7 };
    var f2 = try compute.beginFrame(5000);
    errdefer f2.abort();
    try f2.encodeThreads(scale_pipe, &.{.{ .buffer = buf, .offset = 0 }}, std.mem.asBytes(&params), .{ 64, 1, 1 }, .{ 32, 1, 1 }, false);
    _ = try f2.commit();

    try compute.waitIdle(5000);
    compute.waitIdleBestEffort();

    for (0..64) |i| {
        const expected = (@as(u32, @intCast(i)) ^ 0xA5A5_A5A5) *% 2 +% 7;
        try std.testing.expectEqual(expected, words[i]);
    }
}
