//! Zig view of the Metal half of the bridge (`apple_bridge.h`). Converts
//! status codes into an explicit error set and null-checks every handle the
//! bridge returns.

const std = @import("std");
const c = @import("apple_bridge");
const page = @import("../mem/page.zig");

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

    /// Compiles (`.mlmodel` / `.mlpackage`) and loads, or directly loads
    /// (`.mlmodelc`), a CoreML model from disk. Startup only.
    pub fn loadPath(path: [:0]const u8, units: AneComputeUnits) Error!AneModel {
        if (path.len == 0) return error.InvalidArgument;
        var handle: ?*c.ami_ane_model = null;
        try check(c.ami_ane_model_load_path(path.ptr, units.toRaw(), &handle));
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
    try std.testing.expectEqual(@as(u32, 2048), UNIFORM_SLOTS_PER_FRAME);
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

        const tmp_dir = std.c.getenv("TMPDIR") orelse "/Volumes/Akari/.tmp";
        var path_buf: [512]u8 = undefined;
        const path_slice = try std.fmt.bufPrint(path_buf[0 .. path_buf.len - 1], "{s}/aminfer_test_ip_{d}.mlmodel", .{
            std.mem.sliceTo(tmp_dir, 0),
            std.c.getpid(),
        });
        path_buf[path_slice.len] = 0;
        const path_z: [:0]const u8 = path_buf[0..path_slice.len :0];
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
