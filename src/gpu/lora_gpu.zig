//! Zero-flush Metal 4 GPU Low-Rank (`r = 1..64`) + PEFT DoRA operators (`W30`, M9.5).
//!
//! Architecture & engineering invariants:
//! - Wraps `LoraRegistry.poolSlice()` directly via `metal.Buffer.wrap`
//!   (`newBufferWithBytesNoCopy`, 16 KiB page-aligned, **0 `memcpy`**) and
//!   registers it in the caller's Metal 4 `metal.Compute` residency set.
//! - Preallocates `z_f32` intermediate scratch and per-frame-slot descriptor +
//!   heterogeneous `row_slots` staging inside one startup `arena_mod.WiredArena`
//!   (`0` hot-path heap allocations, `0` mid-frame `flushFrame()` calls).
//! - Encodes two compute dispatches (`ami_lora_down_a` -> intra-pass compute
//!   barrier -> `ami_lora_up_b_dora`) directly into the caller's active
//!   `*metal.Compute.Frame`, automatically chunking arbitrary `M > max_rows`
//!   prefill spans with intra-pass barriers only.
//! - Supports `r = 1..64`, `row_major` and `sme2_panel64` (`f16` / `bf16` weights),
//!   `x` in `f16` / `bf16` / `f32`, in-place `y` in `f16` / `bf16` / `f32`, exact
//!   PEFT DoRA row scaling (`dora_row_scale_f32`), and `C14` heterogeneous per-row
//!   multi-adapter batching.

const std = @import("std");
const metal = @import("../bridge/metal.zig");
const page = @import("../mem/page.zig");
const arena_mod = @import("../mem/arena.zig");
const budget_mod = @import("../mem/budget.zig");
const amf = @import("../format/amf.zig");
const quant = @import("../quant/spec.zig");
const lora = @import("../model/lora.zig");

pub const MAX_LORA_RANK: u32 = lora.MAX_LORA_RANK;
pub const Z_ROW_STRIDE: u32 = 64;
pub const BATCH_SLOTS_MODE: u32 = 0xFFFFFFFE;

pub const LORA_FLAG_W_BF16: u32 = 1 << 0;
pub const LORA_FLAG_PANEL64: u32 = 1 << 1;
pub const LORA_FLAG_HAS_DORA: u32 = 1 << 2;
pub const LORA_FLAG_ACTIVE: u32 = 1 << 3;

pub const LORA_PARAM_VEC4_X_SAFE: u32 = 1 << 0;

pub const ActivationDType = enum(u32) {
    f16 = 0,
    bf16 = 1,
    f32 = 2,

    pub fn elemBytes(self: ActivationDType) usize {
        return switch (self) {
            .f16, .bf16 => 2,
            .f32 => 4,
        };
    }
};

/// Exact ABI match with `LoraSlotDesc` in `src/kernels/lora_ops.metal`.
pub const LoraSlotDesc = extern struct {
    a_byte_offset: u64 = 0,
    b_byte_offset: u64 = 0,
    dora_scale_byte_offset: u64 = 0,
    scale: f32 = 0.0,
    rank: u32 = 0,
    flags: u32 = 0,
    _pad0: u32 = 0,
};

/// Exact ABI match with `LoraParams` in `src/kernels/lora_ops.metal`.
pub const LoraParams = extern struct {
    m: u32,
    k: u32,
    n: u32,
    max_rank: u32,
    z_stride: u32,
    x_dtype: u32,
    y_dtype: u32,
    single_slot: u32,
    num_slots: u32,
    x_row_stride: u32,
    y_row_stride: u32,
    param_flags: u32,
};

comptime {
    if (@sizeOf(LoraSlotDesc) != 40) @compileError("LoraSlotDesc ABI size mismatch");
    if (@sizeOf(LoraParams) != 48) @compileError("LoraParams ABI size mismatch");
}

pub const WorkspaceInitError = metal.WrapError || arena_mod.WiredArena.InitError || arena_mod.WiredArena.AllocError || error{
    InvalidConfig,
};

pub const EncodeError = metal.Error || error{
    ZeroDimension,
    InvalidConfig,
    DimensionMismatch,
    UnsupportedRank,
    UnsupportedDType,
    StagingExhausted,
    PoolMismatch,
    Overflow,
};

pub const GpuLoraWorkspaceConfig = struct {
    /// Maximum rows processed per chunk in `z_f32` scratch (arbitrary `M > max_rows`
    /// is automatically chunked with intra-pass barriers and zero flushes).
    max_rows: u32 = 64,
    /// Preallocated descriptor staging capacity per in-flight frame slot (bytes, 64B aligned).
    desc_staging_bytes_per_frame: usize = 256 * 1024,
    /// Preallocated per-row slot-ID staging capacity per in-flight frame slot (`u32` elements).
    row_slots_per_frame: u32 = 16384,
    /// Number of disjoint staging rings matching `metal.Compute` frames-in-flight (`1..MAX_FRAMES_IN_FLIGHT`).
    frames_in_flight: u32 = 2,
    /// Optional `R9` wired budget checking workspace scratch memory.
    wired_budget: ?budget_mod.Budget = null,
};

/// Preallocated zero-copy GPU LoRA workspace bound to a `lora.LoraRegistry`'s
/// wired `page.WiredRegion` (0 `memcpy`) plus a single `WiredArena` for `z_f32`
/// and per-frame descriptor/slot staging.
pub const GpuLoraWorkspace = struct {
    arena: arena_mod.WiredArena,
    scratch_buf: metal.Buffer,
    pool_buf: metal.Buffer,
    pool_base_ptr: [*]const u8,
    pool_len: usize,
    z_off: u64,
    max_rows: u32,
    frames_in_flight: u32,
    desc_base_off: [metal.MAX_FRAMES_IN_FLIGHT]u64,
    desc_cap_bytes: usize,
    desc_cursor: [metal.MAX_FRAMES_IN_FLIGHT]usize,
    row_slots_base_off: [metal.MAX_FRAMES_IN_FLIGHT]u64,
    row_slots_cap: u32,
    row_slots_cursor: [metal.MAX_FRAMES_IN_FLIGHT]u32,

    pub fn init(
        device: metal.Device,
        compute: metal.Compute,
        reg: *const lora.LoraRegistry,
        cfg: GpuLoraWorkspaceConfig,
    ) WorkspaceInitError!GpuLoraWorkspace {
        if (cfg.max_rows == 0 or cfg.desc_staging_bytes_per_frame < 64 or cfg.row_slots_per_frame == 0) {
            return error.InvalidConfig;
        }
        if (cfg.frames_in_flight == 0 or cfg.frames_in_flight > metal.MAX_FRAMES_IN_FLIGHT) {
            return error.InvalidConfig;
        }

        const pool_slice = reg.poolSlice();
        const pool_buf = try metal.Buffer.wrap(device, pool_slice);
        errdefer pool_buf.close();

        const z_bytes = std.mem.alignForward(
            usize,
            @as(usize, cfg.max_rows) * Z_ROW_STRIDE * @sizeOf(f32),
            256,
        );
        const desc_cap = std.mem.alignForward(usize, cfg.desc_staging_bytes_per_frame, 64);
        const row_slots_bytes = std.mem.alignForward(usize, @as(usize, cfg.row_slots_per_frame) * @sizeOf(u32), 64);
        const frames: usize = cfg.frames_in_flight;
        const total_raw = z_bytes + frames * (desc_cap + row_slots_bytes) + 1024;
        const total_bytes = try page.alignUp(total_raw);

        var arena = if (cfg.wired_budget) |b|
            try arena_mod.WiredArena.init(total_bytes, b)
        else
            arena_mod.WiredArena{
                .region = try page.WiredRegion.init(total_bytes),
                .used = 0,
                .owns_mapping = true,
            };
        errdefer arena.deinit();

        const z_slice = try arena.allocAligned(z_bytes, 256);
        const z_off: u64 = @intCast(arena.offsetOf(z_slice.ptr));

        var desc_base_off: [metal.MAX_FRAMES_IN_FLIGHT]u64 = @splat(0);
        var row_slots_base_off: [metal.MAX_FRAMES_IN_FLIGHT]u64 = @splat(0);
        for (0..frames) |fi| {
            const d_sl = try arena.allocAligned(desc_cap, 64);
            @memset(d_sl, 0);
            desc_base_off[fi] = @intCast(arena.offsetOf(d_sl.ptr));

            const rs_sl = try arena.allocAligned(row_slots_bytes, 64);
            @memset(rs_sl, 0);
            row_slots_base_off[fi] = @intCast(arena.offsetOf(rs_sl.ptr));
        }

        const scratch_buf = try metal.Buffer.wrap(device, arena.bytes());
        errdefer scratch_buf.close();

        try compute.makeResident(&.{ pool_buf, scratch_buf });

        return .{
            .arena = arena,
            .scratch_buf = scratch_buf,
            .pool_buf = pool_buf,
            .pool_base_ptr = pool_slice.ptr,
            .pool_len = pool_slice.len,
            .z_off = z_off,
            .max_rows = cfg.max_rows,
            .frames_in_flight = cfg.frames_in_flight,
            .desc_base_off = desc_base_off,
            .desc_cap_bytes = desc_cap,
            .desc_cursor = @splat(0),
            .row_slots_base_off = row_slots_base_off,
            .row_slots_cap = cfg.row_slots_per_frame,
            .row_slots_cursor = @splat(0),
        };
    }

    pub fn deinit(self: *GpuLoraWorkspace, compute: ?metal.Compute) void {
        if (compute) |comp| {
            comp.removeResident(&.{ self.pool_buf, self.scratch_buf }) catch |err| {
                std.debug.panic("removeResident failed in GpuLoraWorkspace.deinit: {}", .{err});
            };
        }
        self.scratch_buf.close();
        self.pool_buf.close();
        self.arena.deinit();
    }

    pub fn makeResident(self: *const GpuLoraWorkspace, compute: metal.Compute) metal.Error!void {
        try compute.makeResident(&.{ self.pool_buf, self.scratch_buf });
    }

    pub fn removeResident(self: *const GpuLoraWorkspace, compute: metal.Compute) metal.Error!void {
        try compute.removeResident(&.{ self.pool_buf, self.scratch_buf });
    }

    /// Resets per-frame staging cursors for `frame_slot` (call right after `compute.beginFrame`).
    pub fn beginFrame(self: *GpuLoraWorkspace, frame_slot: u32) void {
        const fi: usize = frame_slot % self.frames_in_flight;
        self.desc_cursor[fi] = 0;
        self.row_slots_cursor[fi] = 0;
    }

    fn verifyPoolPointer(self: *const GpuLoraWorkspace, ptr: [*]const u8, len: usize) EncodeError!u64 {
        const base = @intFromPtr(self.pool_base_ptr);
        const addr = @intFromPtr(ptr);
        if (addr < base or addr + len > base + self.pool_len) return error.PoolMismatch;
        return @intCast(addr - base);
    }

    fn stageDescriptors(
        self: *GpuLoraWorkspace,
        frame_slot: u32,
        descs: []const LoraSlotDesc,
    ) EncodeError!metal.BufferBinding {
        const fi: usize = frame_slot % self.frames_in_flight;
        const raw_bytes = descs.len * @sizeOf(LoraSlotDesc);
        const aligned_bytes = std.mem.alignForward(usize, raw_bytes, 64);
        const cur = self.desc_cursor[fi];
        if (cur + aligned_bytes > self.desc_cap_bytes) return error.StagingExhausted;
        const abs_off: usize = @intCast(self.desc_base_off[fi] + cur);
        const dst_bytes = self.arena.bytes()[abs_off .. abs_off + raw_bytes];
        @memcpy(dst_bytes, std.mem.sliceAsBytes(descs));
        self.desc_cursor[fi] = cur + aligned_bytes;
        return .{ .buffer = self.scratch_buf, .offset = @intCast(abs_off) };
    }

    fn stageRowSlots(
        self: *GpuLoraWorkspace,
        frame_slot: u32,
        slots: []const u32,
    ) EncodeError!metal.BufferBinding {
        const fi: usize = frame_slot % self.frames_in_flight;
        const count_u32: u32 = @intCast(slots.len);
        const aligned_count: u32 = std.mem.alignForward(u32, count_u32, 16);
        const cur = self.row_slots_cursor[fi];
        if (@as(u64, cur) + aligned_count > self.row_slots_cap) return error.StagingExhausted;
        const abs_off: usize = @intCast(self.row_slots_base_off[fi] + @as(u64, cur) * @sizeOf(u32));
        const dst_bytes = self.arena.bytes()[abs_off .. abs_off + slots.len * @sizeOf(u32)];
        @memcpy(dst_bytes, std.mem.sliceAsBytes(slots));
        self.row_slots_cursor[fi] = cur + aligned_count;
        return .{ .buffer = self.scratch_buf, .offset = @intCast(abs_off) };
    }

    fn dummyRowSlotsBinding(self: *const GpuLoraWorkspace, frame_slot: u32) metal.BufferBinding {
        const fi: usize = frame_slot % self.frames_in_flight;
        return .{ .buffer = self.scratch_buf, .offset = self.row_slots_base_off[fi] };
    }

    fn zBinding(self: *const GpuLoraWorkspace) metal.BufferBinding {
        return .{ .buffer = self.scratch_buf, .offset = self.z_off };
    }

    fn poolBinding(self: *const GpuLoraWorkspace) metal.BufferBinding {
        return .{ .buffer = self.pool_buf, .offset = 0 };
    }
};

pub const ProjectionArgs = struct {
    x: metal.BufferBinding,
    y: metal.BufferBinding,
    m: u32,
    /// Input feature dim `K` (`0` infers from the active adapter pair).
    k: u32 = 0,
    /// Output feature dim `N` (`0` infers from the active adapter pair).
    n: u32 = 0,
    x_dtype: ActivationDType = .f16,
    y_dtype: ActivationDType = .f16,
    /// Row stride in elements (`0` defaults to `k` / `n`).
    x_row_stride: u32 = 0,
    y_row_stride: u32 = 0,
    /// Whether to emit an intra-pass barrier before the first LoRA dispatch.
    insert_barrier_before: bool = true,
};

pub const GpuLoraPipelines = struct {
    down_a: metal.Pipeline,
    up_b_dora: metal.Pipeline,

    pub fn init(device: metal.Device, library: metal.Library) metal.Error!GpuLoraPipelines {
        const down_a = try metal.Pipeline.init(device, library, "ami_lora_down_a");
        errdefer down_a.close();
        const up_b_dora = try metal.Pipeline.init(device, library, "ami_lora_up_b_dora");
        errdefer up_b_dora.close();
        return .{
            .down_a = down_a,
            .up_b_dora = up_b_dora,
        };
    }

    pub fn deinit(self: GpuLoraPipelines) void {
        self.up_b_dora.close();
        self.down_a.close();
    }

    fn buildDescriptor(ws: *const GpuLoraWorkspace, pair: *const lora.LoraLinearPair) EncodeError!LoraSlotDesc {
        if (pair.rank == 0 or pair.rank > MAX_LORA_RANK) return error.UnsupportedRank;
        if (pair.weight_dtype == .q2_affine) return error.UnsupportedDType;
        if (pair.k == 0 or pair.n == 0) return error.ZeroDimension;

        const a_off = try ws.verifyPoolPointer(pair.a_blob.ptr, pair.a_blob.len);
        const b_off = try ws.verifyPoolPointer(pair.b_blob.ptr, pair.b_blob.len);
        var dora_off: u64 = 0;
        var flags: u32 = LORA_FLAG_ACTIVE;
        if (pair.dtype == .bf16) flags |= LORA_FLAG_W_BF16;
        if (pair.layout == .sme2_panel64) flags |= LORA_FLAG_PANEL64;
        if (pair.use_dora) {
            if (pair.dora_row_scale_f32) |ds| {
                const ds_bytes = std.mem.sliceAsBytes(ds);
                dora_off = try ws.verifyPoolPointer(ds_bytes.ptr, ds_bytes.len);
                flags |= LORA_FLAG_HAS_DORA;
            }
        }
        return .{
            .a_byte_offset = a_off,
            .b_byte_offset = b_off,
            .dora_scale_byte_offset = dora_off,
            .scale = pair.scale,
            .rank = pair.rank,
            .flags = flags,
        };
    }

    fn encodeDispatchesChunked(
        self: GpuLoraPipelines,
        frame: *metal.Compute.Frame,
        ws: *const GpuLoraWorkspace,
        desc_bind: metal.BufferBinding,
        row_slots_bind: metal.BufferBinding,
        single_slot: u32,
        num_slots: u32,
        max_rank: u32,
        k: u32,
        n: u32,
        args: ProjectionArgs,
    ) EncodeError!void {
        const x_stride: u32 = if (args.x_row_stride != 0) args.x_row_stride else k;
        const y_stride: u32 = if (args.y_row_stride != 0) args.y_row_stride else n;
        if (x_stride < k or y_stride < n) return error.DimensionMismatch;

        const x_elem_bytes: u64 = @intCast(args.x_dtype.elemBytes());
        const y_elem_bytes: u64 = @intCast(args.y_dtype.elemBytes());
        const x_row_bytes = std.math.mul(u64, x_stride, x_elem_bytes) catch return error.Overflow;
        const y_row_bytes = std.math.mul(u64, y_stride, y_elem_bytes) catch return error.Overflow;

        const rank_groups: u64 = (@as(u64, max_rank) + 3) / 4;
        const n_groups: u64 = (@as(u64, n) + 255) / 256;

        var row_off: u32 = 0;
        while (row_off < args.m) {
            const chunk_m: u32 = @min(args.m - row_off, ws.max_rows);
            const x_chunk_off = std.math.add(
                u64,
                args.x.offset,
                std.math.mul(u64, row_off, x_row_bytes) catch return error.Overflow,
            ) catch return error.Overflow;
            const y_chunk_off = std.math.add(
                u64,
                args.y.offset,
                std.math.mul(u64, row_off, y_row_bytes) catch return error.Overflow,
            ) catch return error.Overflow;
            const rs_chunk_off: u64 = if (single_slot == BATCH_SLOTS_MODE)
                row_slots_bind.offset + @as(u64, row_off) * @sizeOf(u32)
            else
                row_slots_bind.offset;

            const vec4_x_safe = ((x_chunk_off & 15) == 0) and ((k & 3) == 0) and ((x_stride & 3) == 0);
            const params: LoraParams = .{
                .m = chunk_m,
                .k = k,
                .n = n,
                .max_rank = max_rank,
                .z_stride = Z_ROW_STRIDE,
                .x_dtype = @backingInt(args.x_dtype),
                .y_dtype = @backingInt(args.y_dtype),
                .single_slot = single_slot,
                .num_slots = num_slots,
                .x_row_stride = x_stride,
                .y_row_stride = y_stride,
                .param_flags = if (vec4_x_safe) LORA_PARAM_VEC4_X_SAFE else 0,
            };

            const x_chunk: metal.BufferBinding = .{ .buffer = args.x.buffer, .offset = x_chunk_off };
            const y_chunk: metal.BufferBinding = .{ .buffer = args.y.buffer, .offset = y_chunk_off };
            const rs_chunk: metal.BufferBinding = .{ .buffer = row_slots_bind.buffer, .offset = rs_chunk_off };

            // Dispatch 1: z_f32[row, j] = scale * dot(x[row, 0..K], A[j, 0..K])
            try frame.encodeThreadgroups(
                self.down_a,
                &.{
                    x_chunk,
                    ws.poolBinding(),
                    desc_bind,
                    rs_chunk,
                    ws.zBinding(),
                },
                std.mem.asBytes(&params),
                .{ rank_groups, chunk_m, 1 },
                .{ 128, 1, 1 },
                if (row_off == 0) args.insert_barrier_before else true,
            );

            // Dispatch 2: y[row, col] = dora_scale[col] * (y[row, col] + dot(z_f32[row, 0..r], B[col, 0..r]))
            // Zero flushFrame(): only an intra-pass compute memory barrier!
            try frame.encodeThreadgroups(
                self.up_b_dora,
                &.{
                    ws.zBinding(),
                    ws.poolBinding(),
                    desc_bind,
                    rs_chunk,
                    y_chunk,
                },
                std.mem.asBytes(&params),
                .{ n_groups, chunk_m, 1 },
                .{ 256, 1, 1 },
                true,
            );

            row_off += chunk_m;
        }
    }

    /// Directly encodes a single `LoraLinearPair` into the active Metal 4 `frame`
    /// with zero flushes.
    pub fn encodePairDirect(
        self: GpuLoraPipelines,
        frame: *metal.Compute.Frame,
        ws: *GpuLoraWorkspace,
        pair: *const lora.LoraLinearPair,
        args: ProjectionArgs,
    ) EncodeError!void {
        if (args.m == 0) return error.ZeroDimension;
        const k: u32 = if (args.k != 0) args.k else pair.k;
        const n: u32 = if (args.n != 0) args.n else pair.n;
        if (k != pair.k or n != pair.n) return error.DimensionMismatch;
        if (pair.scale == 0.0 and !pair.use_dora) return;

        const desc = try buildDescriptor(ws, pair);
        const desc_arr = [_]LoraSlotDesc{desc};
        const desc_bind = try ws.stageDescriptors(frame.slot, &desc_arr);
        try self.encodeDispatchesChunked(
            frame,
            ws,
            desc_bind,
            ws.dummyRowSlotsBinding(frame.slot),
            0,
            1,
            pair.rank,
            k,
            n,
            args,
        );
    }

    /// Encodes a single active adapter slot's projection delta (+ optional DoRA scale)
    /// into `frame` with ZERO `flushFrame()` calls. Returns `false` (no dispatches)
    /// if `slot_id == NO_ADAPTER` or the adapter does not target `(layer_idx, target)`.
    pub fn encodeSingleAdapterProjection(
        self: GpuLoraPipelines,
        frame: *metal.Compute.Frame,
        ws: *GpuLoraWorkspace,
        reg: *const lora.LoraRegistry,
        slot_id: u32,
        layer_idx: usize,
        target: lora.TargetProjection,
        args: ProjectionArgs,
    ) EncodeError!bool {
        if (slot_id == lora.NO_ADAPTER or args.m == 0) return false;
        const pair = reg.pairAt(slot_id, layer_idx, target) orelse return false;
        if (pair.scale == 0.0 and !pair.use_dora) return false;
        try self.encodePairDirect(frame, ws, pair, args);
        return true;
    }

    /// Encodes heterogeneous `C14` continuous-batching LoRA + DoRA projections where
    /// row `r` selects `batch_lora_slots[r]` (`NO_ADAPTER` rows stay bit-exact untouched).
    /// Returns `false` if no row in the batch targets `(layer_idx, target)`.
    pub fn encodeBatchProjection(
        self: GpuLoraPipelines,
        frame: *metal.Compute.Frame,
        ws: *GpuLoraWorkspace,
        reg: *const lora.LoraRegistry,
        batch_lora_slots: []const u32,
        layer_idx: usize,
        target: lora.TargetProjection,
        args: ProjectionArgs,
    ) EncodeError!bool {
        if (args.m == 0 or batch_lora_slots.len == 0) return false;
        if (batch_lora_slots.len != args.m) return error.InvalidConfig;

        var descs: [lora.MAX_ADAPTERS]LoraSlotDesc = @splat(.{});
        var desc_built: [lora.MAX_ADAPTERS]bool = @splat(false);
        var max_rank: u32 = 0;
        var eff_k: u32 = args.k;
        var eff_n: u32 = args.n;
        var highest_slot_plus_one: u32 = 0;
        var any_active = false;

        for (batch_lora_slots) |sid| {
            if (sid == lora.NO_ADAPTER or sid >= lora.MAX_ADAPTERS) continue;
            if (desc_built[sid]) continue;
            desc_built[sid] = true;

            const pair = reg.pairAt(sid, layer_idx, target) orelse continue;
            if (pair.scale == 0.0 and !pair.use_dora) continue;

            if (eff_k == 0) eff_k = pair.k;
            if (eff_n == 0) eff_n = pair.n;
            if (pair.k != eff_k or pair.n != eff_n) return error.DimensionMismatch;

            descs[sid] = try buildDescriptor(ws, pair);
            if (pair.rank > max_rank) max_rank = pair.rank;
            if (sid + 1 > highest_slot_plus_one) highest_slot_plus_one = sid + 1;
            any_active = true;
        }
        if (!any_active) return false;

        const desc_bind = try ws.stageDescriptors(frame.slot, descs[0..highest_slot_plus_one]);
        const rs_bind = try ws.stageRowSlots(frame.slot, batch_lora_slots);
        try self.encodeDispatchesChunked(
            frame,
            ws,
            desc_bind,
            rs_bind,
            BATCH_SLOTS_MODE,
            highest_slot_plus_one,
            max_rank,
            eff_k,
            eff_n,
            args,
        );
        return true;
    }
};

// ===========================================================================
// Unit Tests (`W30`, Metal 4 Zero-Flush LoRA + DoRA + Heterogeneous Batch)
// ===========================================================================

fn fillPattern(buf: []f32, seed: usize, scale: f32) void {
    for (buf, 0..) |*v, i| {
        const s: i32 = @intCast(((i * 17 + seed * 31) % 19));
        v.* = @as(f32, @floatFromInt(s - 9)) * scale;
    }
}

fn refDoraMergedRow(
    x_f32: []const f32,
    w0_f32: []const f32,
    a_f32: []const f32,
    b_f32: []const f32,
    mag_f32: ?[]const f32,
    scale: f32,
    n: usize,
    k: usize,
    r: usize,
    out_f32: []f32,
) void {
    for (0..n) |i| {
        var norm_sq: f64 = 0.0;
        var dot_val: f64 = 0.0;
        for (0..k) |t| {
            var ba: f64 = 0.0;
            for (0..r) |j| ba += @as(f64, b_f32[i * r + j]) * @as(f64, a_f32[j * k + t]);
            const w_eff: f64 = @as(f64, w0_f32[i * k + t]) + @as(f64, scale) * ba;
            norm_sq += w_eff * w_eff;
            dot_val += w_eff * @as(f64, x_f32[t]);
        }
        if (mag_f32) |m| {
            const gamma: f64 = @as(f64, m[i]) / @sqrt(norm_sq + 1.0e-12);
            out_f32[i] = @floatCast(gamma * dot_val);
        } else {
            out_f32[i] = @floatCast(dot_val);
        }
    }
}

test "gpu.lora_gpu. r=8/16/64 row_major & sme2_panel64 f16/bf16 x(f16/bf16/f32) y(f16/f32) and chunked M>max_rows" {
    const gpa = std.testing.allocator;
    const device = try metal.Device.open();
    defer device.close();
    const lib = try metal.Library.openEmbedded(device);
    defer lib.close();
    var compute = try metal.Compute.init(device);
    defer compute.close();

    const pipes = try GpuLoraPipelines.init(device, lib);
    defer pipes.deinit();

    const base_shape: lora.BaseModelShape = .{
        .num_layers = 4,
        .hidden_size = 128,
        .flexible_dims = true,
    };
    const pool_bytes: usize = 32 * 1024 * 1024; // 32 MiB > 4 MiB legacy cap
    const wired_budget = budget_mod.Budget.compute(.{
        .memory_bytes = 256 * 1024 * 1024,
        .user_wire_limit = 128 * 1024 * 1024,
        .global_user_wire_limit = 128 * 1024 * 1024,
        .gpu_working_set_bytes = 128 * 1024 * 1024,
        .configured_cap = null,
        .headroom_bytes = page.APPLE_PAGE_SIZE,
    });
    var reg = try lora.LoraRegistry.init(.{
        .pool_bytes = pool_bytes,
        .allocator = gpa,
        .wired_budget = wired_budget,
        .base_shape = base_shape,
    });
    defer reg.deinit(gpa);

    var ws = try GpuLoraWorkspace.init(device, compute, &reg, .{
        .max_rows = 16, // Forces chunking when M = 40 > 16 with zero flushes!
        .wired_budget = wired_budget,
    });
    defer ws.deinit(compute);

    const Case = struct {
        rank: u32,
        m: u32,
        k: u32,
        n: u32,
        layout: amf.TensorLayout,
        w_dt: quant.FloatDType,
        x_dt: ActivationDType,
        y_dt: ActivationDType,
        target: lora.TargetProjection,
        custom_sfx: ?[]const u8,
        alpha: f32,
    };
    const cases = [_]Case{
        .{ .rank = 8, .m = 1, .k = 128, .n = 96, .layout = .row_major, .w_dt = .f16, .x_dt = .f16, .y_dt = .f16, .target = .q_proj, .custom_sfx = "attn.wq_du", .alpha = 16.0 },
        .{ .rank = 16, .m = 5, .k = 128, .n = 192, .layout = .row_major, .w_dt = .bf16, .x_dt = .bf16, .y_dt = .f32, .target = .gate_attn, .custom_sfx = "attn.gate_attn", .alpha = 16.0 },
        .{ .rank = 64, .m = 40, .k = 128, .n = 128, .layout = .row_major, .w_dt = .f16, .x_dt = .f32, .y_dt = .f32, .target = .o_proj, .custom_sfx = "attn.wo_ud", .alpha = 32.0 },
        .{ .rank = 8, .m = 4, .k = 128, .n = 80, .layout = .sme2_panel64, .w_dt = .f16, .x_dt = .f16, .y_dt = .f16, .target = .k_proj, .custom_sfx = null, .alpha = 8.0 },
        .{ .rank = 64, .m = 20, .k = 128, .n = 96, .layout = .sme2_panel64, .w_dt = .bf16, .x_dt = .f16, .y_dt = .f16, .target = .gate_up_proj, .custom_sfx = "ffn.w13", .alpha = 64.0 },
        .{ .rank = 7, .m = 3, .k = 70, .n = 53, .layout = .row_major, .w_dt = .f16, .x_dt = .f16, .y_dt = .f16, .target = .wr_du, .custom_sfx = "attn.wr_du", .alpha = 7.0 },
    };

    const io_bytes = try page.alignUp(2 * 1024 * 1024);
    var io_region = try page.WiredRegion.init(io_bytes);
    defer io_region.deinit();
    const io_buf = try metal.Buffer.wrap(device, io_region.bytes);
    defer io_buf.close();
    try compute.makeResident(&.{io_buf});
    defer compute.removeResident(&.{io_buf}) catch |err| {
        std.debug.panic("removeResident(io_buf) failed in lora_gpu test: {}", .{err});
    };

    for (cases, 0..) |cs, ci| {
        const a_f32 = try gpa.alloc(f32, @as(usize, cs.rank) * cs.k);
        defer gpa.free(a_f32);
        const b_f32 = try gpa.alloc(f32, @as(usize, cs.n) * cs.rank);
        defer gpa.free(b_f32);
        fillPattern(a_f32, ci * 3 + 1, 0.03125);
        fillPattern(b_f32, ci * 3 + 2, 0.03125);

        const img = try lora.buildSyntheticAmloraBytes(gpa, "gpu-test", cs.rank, cs.alpha, &.{
            .{
                .layer = 1,
                .target = cs.target,
                .n = cs.n,
                .k = cs.k,
                .rank = cs.rank,
                .layout = cs.layout,
                .dtype = cs.w_dt,
                .a_f32 = a_f32,
                .b_f32 = b_f32,
                .custom_suffix = cs.custom_sfx,
            },
        });
        defer gpa.free(img);

        const slot = try reg.loadAdapterFromBytes(gpa, "adp", img);
        defer reg.unloadSlot(gpa, slot) catch unreachable;

        const x_off: usize = 0;
        const y_off: usize = 512 * 1024;
        const x_elems: usize = @as(usize, cs.m) * cs.k;
        const y_elems: usize = @as(usize, cs.m) * cs.n;

        const x_ref = try gpa.alloc(f32, x_elems);
        defer gpa.free(x_ref);
        const y_init = try gpa.alloc(f32, y_elems);
        defer gpa.free(y_init);
        fillPattern(x_ref, ci + 11, 0.125);
        fillPattern(y_init, ci + 29, 0.25);

        switch (cs.x_dt) {
            .f16 => {
                const dst: []u16 = @alignCast(std.mem.bytesAsSlice(u16, io_region.bytes[x_off..][0 .. x_elems * 2]));
                for (x_ref, 0..) |v, idx| {
                    dst[idx] = quant.f32ToFloatBits(.f16, v);
                    x_ref[idx] = quant.floatBitsToF32(.f16, dst[idx]);
                }
            },
            .bf16 => {
                const dst: []u16 = @alignCast(std.mem.bytesAsSlice(u16, io_region.bytes[x_off..][0 .. x_elems * 2]));
                for (x_ref, 0..) |v, idx| {
                    dst[idx] = quant.f32ToFloatBits(.bf16, v);
                    x_ref[idx] = quant.floatBitsToF32(.bf16, dst[idx]);
                }
            },
            .f32 => {
                const dst: []f32 = @alignCast(std.mem.bytesAsSlice(f32, io_region.bytes[x_off..][0 .. x_elems * 4]));
                @memcpy(dst, x_ref);
            },
        }
        switch (cs.y_dt) {
            .f16 => {
                const dst: []u16 = @alignCast(std.mem.bytesAsSlice(u16, io_region.bytes[y_off..][0 .. y_elems * 2]));
                for (y_init, 0..) |v, idx| {
                    dst[idx] = quant.f32ToFloatBits(.f16, v);
                    y_init[idx] = quant.floatBitsToF32(.f16, dst[idx]);
                }
            },
            .bf16 => {
                const dst: []u16 = @alignCast(std.mem.bytesAsSlice(u16, io_region.bytes[y_off..][0 .. y_elems * 2]));
                for (y_init, 0..) |v, idx| {
                    dst[idx] = quant.f32ToFloatBits(.bf16, v);
                    y_init[idx] = quant.floatBitsToF32(.bf16, dst[idx]);
                }
            },
            .f32 => {
                const dst: []f32 = @alignCast(std.mem.bytesAsSlice(f32, io_region.bytes[y_off..][0 .. y_elems * 4]));
                @memcpy(dst, y_init);
            },
        }

        var frame = try compute.beginFrame(5_000);
        errdefer frame.abort();
        ws.beginFrame(frame.slot);
        const encoded = try pipes.encodeSingleAdapterProjection(&frame, &ws, &reg, slot, 1, cs.target, .{
            .x = .{ .buffer = io_buf, .offset = x_off },
            .y = .{ .buffer = io_buf, .offset = y_off },
            .m = cs.m,
            .k = cs.k,
            .n = cs.n,
            .x_dtype = cs.x_dt,
            .y_dtype = cs.y_dt,
            .insert_barrier_before = false,
        });
        try std.testing.expect(encoded);
        const sig = try frame.commit();
        try compute.wait(sig, 5_000);

        const scale: f32 = cs.alpha / @as(f32, @floatFromInt(cs.rank));
        const tol: f32 = if (cs.w_dt == .bf16 or cs.x_dt == .bf16 or cs.y_dt == .f16) 0.03 else 0.005;
        const z_ref = try gpa.alloc(f32, cs.rank);
        defer gpa.free(z_ref);

        for (0..cs.m) |row| {
            for (0..cs.rank) |j| {
                var acc: f32 = 0.0;
                for (0..cs.k) |t| {
                    const aw = quant.floatBitsToF32(cs.w_dt, quant.f32ToFloatBits(cs.w_dt, a_f32[j * cs.k + t]));
                    acc += x_ref[row * cs.k + t] * aw;
                }
                z_ref[j] = acc * scale;
            }
            for (0..cs.n) |o| {
                var expected = y_init[row * cs.n + o];
                for (0..cs.rank) |j| {
                    const bw = quant.floatBitsToF32(cs.w_dt, quant.f32ToFloatBits(cs.w_dt, b_f32[o * cs.rank + j]));
                    expected += z_ref[j] * bw;
                }
                const got: f32 = switch (cs.y_dt) {
                    .f16 => quant.floatBitsToF32(.f16, (@as([]const u16, @alignCast(std.mem.bytesAsSlice(u16, io_region.bytes[y_off..][0 .. y_elems * 2]))))[row * cs.n + o]),
                    .bf16 => quant.floatBitsToF32(.bf16, (@as([]const u16, @alignCast(std.mem.bytesAsSlice(u16, io_region.bytes[y_off..][0 .. y_elems * 2]))))[row * cs.n + o]),
                    .f32 => (@as([]const f32, @alignCast(std.mem.bytesAsSlice(f32, io_region.bytes[y_off..][0 .. y_elems * 4]))))[row * cs.n + o],
                };
                try std.testing.expectApproxEqAbs(expected, got, tol);
            }
        }
    }
}

test "gpu.lora_gpu. exact DoRA equivalence, heterogeneous C14 batching, zero-flush multi-layer frame, and hot-reload" {
    const gpa = std.testing.allocator;
    const device = try metal.Device.open();
    defer device.close();
    const lib = try metal.Library.openEmbedded(device);
    defer lib.close();
    var compute = try metal.Compute.init(device);
    defer compute.close();

    const pipes = try GpuLoraPipelines.init(device, lib);
    defer pipes.deinit();

    var reg = try lora.LoraRegistry.init(.{
        .pool_bytes = 8 * 1024 * 1024,
        .allocator = gpa,
        .base_shape = .{ .num_layers = 2, .hidden_size = 128, .flexible_dims = true },
    });
    defer reg.deinit(gpa);

    var ws = try GpuLoraWorkspace.init(device, compute, &reg, .{ .max_rows = 16 });
    defer ws.deinit(compute);

    const k: usize = 128;
    const n: usize = 96;
    const r1: usize = 8;
    const r2: usize = 16;

    const w0 = try gpa.alloc(f32, n * k);
    defer gpa.free(w0);
    fillPattern(w0, 101, 0.0625);

    const a1 = try gpa.alloc(f32, r1 * k);
    defer gpa.free(a1);
    const b1 = try gpa.alloc(f32, n * r1);
    defer gpa.free(b1);
    const mag1 = try gpa.alloc(f32, n);
    defer gpa.free(mag1);
    fillPattern(a1, 103, 0.03125);
    fillPattern(b1, 107, 0.03125);
    for (mag1, 0..) |*v, i| v.* = 0.75 + @as(f32, @floatFromInt(i % 7)) * 0.125;

    const a2 = try gpa.alloc(f32, r2 * k);
    defer gpa.free(a2);
    const b2 = try gpa.alloc(f32, n * r2);
    defer gpa.free(b2);
    const mag2 = try gpa.alloc(f32, n);
    defer gpa.free(mag2);
    fillPattern(a2, 109, 0.03125);
    fillPattern(b2, 113, 0.03125);
    for (mag2, 0..) |*v, i| v.* = 1.1 + @as(f32, @floatFromInt(i % 5)) * 0.15;

    // Adapter 1: DoRA row_major f16 on layer 0 gate_attn AND layer 1 q_proj
    const img1 = try lora.buildSyntheticAmloraBytes(gpa, "dora-rm", r1, 16.0, &.{
        .{ .layer = 0, .target = .gate_attn, .n = n, .k = k, .rank = r1, .layout = .row_major, .dtype = .f16, .a_f32 = a1, .b_f32 = b1, .dora_mag_f32 = mag1, .dora_mag_dtype = .f32 },
        .{ .layer = 1, .target = .q_proj, .n = n, .k = k, .rank = r1, .layout = .row_major, .dtype = .f16, .a_f32 = a1, .b_f32 = b1, .dora_mag_f32 = mag1, .dora_mag_dtype = .f16 },
    });
    defer gpa.free(img1);

    // Adapter 2: DoRA sme2_panel64 bf16 on layer 0 gate_attn (different rank r=16!)
    const img2 = try lora.buildSyntheticAmloraBytes(gpa, "dora-p64", r2, 16.0, &.{
        .{ .layer = 0, .target = .gate_attn, .n = n, .k = k, .rank = r2, .layout = .sme2_panel64, .dtype = .bf16, .a_f32 = a2, .b_f32 = b2, .dora_mag_f32 = mag2, .dora_mag_dtype = .f32 },
    });
    defer gpa.free(img2);

    const slot1 = try reg.loadAdapterFromBytes(gpa, "dora1", img1);
    const slot2 = try reg.loadAdapterFromBytes(gpa, "dora2", img2);
    try reg.calibrateDoraPairFromBaseRows(slot1, 0, .gate_attn, w0);
    try reg.calibrateDoraPairFromBaseRows(slot1, 1, .q_proj, w0);
    try reg.calibrateDoraPairFromBaseRows(slot2, 0, .gate_attn, w0);

    // Heterogeneous batch M=4:
    //   row 0 = NO_ADAPTER (bit-exact base W0 @ x)
    //   row 1 = slot1 (r=8, row_major, f16, DoRA)
    //   row 2 = slot2 (r=16, sme2_panel64, bf16, DoRA)
    //   row 3 = slot1 (r=8, row_major, f16, DoRA)
    const m: usize = 4;
    const io_bytes = try page.alignUp(512 * 1024);
    var io_region = try page.WiredRegion.init(io_bytes);
    defer io_region.deinit();
    const io_buf = try metal.Buffer.wrap(device, io_region.bytes);
    defer io_buf.close();
    try compute.makeResident(&.{io_buf});
    defer compute.removeResident(&.{io_buf}) catch |err| {
        std.debug.panic("removeResident(io_buf) failed in lora_gpu test: {}", .{err});
    };

    const x_off: usize = 0;
    const y0_off: usize = 64 * 1024;
    const y1_off: usize = 128 * 1024;
    const x_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, io_region.bytes[x_off..][0 .. m * k * 4]));
    const y0_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, io_region.bytes[y0_off..][0 .. m * n * 4]));
    const y1_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, io_region.bytes[y1_off..][0 .. m * n * 4]));
    fillPattern(x_f32, 211, 0.2);

    // Pre-populate y0 and y1 with exact base W0 @ x.
    for (0..m) |row| {
        for (0..n) |i| {
            var acc: f32 = 0.0;
            for (0..k) |t| acc += w0[i * k + t] * x_f32[row * k + t];
            y0_f32[row * n + i] = acc;
            y1_f32[row * n + i] = acc;
        }
    }
    var y0_row0_orig: [n]f32 = undefined;
    @memcpy(&y0_row0_orig, y0_f32[0..n]);

    // Encode BOTH layer 0 (heterogeneous batch) AND layer 1 (single-slot) into ONE single frame (0 flushes)!
    const batch_slots = [_]u32{ lora.NO_ADAPTER, slot1, slot2, slot1 };
    var frame = try compute.beginFrame(5_000);
    errdefer frame.abort();
    ws.beginFrame(frame.slot);

    const enc0 = try pipes.encodeBatchProjection(&frame, &ws, &reg, &batch_slots, 0, .gate_attn, .{
        .x = .{ .buffer = io_buf, .offset = x_off },
        .y = .{ .buffer = io_buf, .offset = y0_off },
        .m = m,
        .k = k,
        .n = n,
        .x_dtype = .f32,
        .y_dtype = .f32,
        .insert_barrier_before = false,
    });
    const enc1 = try pipes.encodeSingleAdapterProjection(&frame, &ws, &reg, slot1, 1, .q_proj, .{
        .x = .{ .buffer = io_buf, .offset = x_off },
        .y = .{ .buffer = io_buf, .offset = y1_off },
        .m = m,
        .k = k,
        .n = n,
        .x_dtype = .f32,
        .y_dtype = .f32,
        .insert_barrier_before = true,
    });
    try std.testing.expect(enc0 and enc1);
    const sig = try frame.commit();
    try compute.wait(sig, 5_000);

    // Verify row 0 (NO_ADAPTER) is bit-exact untouched!
    try std.testing.expectEqualSlices(f32, &y0_row0_orig, y0_f32[0..n]);

    // Quantized-weight exact merged DoRA references for adapter 1 (f16) and adapter 2 (bf16).
    const a1_q = try gpa.alloc(f32, r1 * k);
    defer gpa.free(a1_q);
    const b1_q = try gpa.alloc(f32, n * r1);
    defer gpa.free(b1_q);
    for (a1, 0..) |v, idx| a1_q[idx] = quant.floatBitsToF32(.f16, quant.f32ToFloatBits(.f16, v));
    for (b1, 0..) |v, idx| b1_q[idx] = quant.floatBitsToF32(.f16, quant.f32ToFloatBits(.f16, v));

    const a2_q = try gpa.alloc(f32, r2 * k);
    defer gpa.free(a2_q);
    const b2_q = try gpa.alloc(f32, n * r2);
    defer gpa.free(b2_q);
    for (a2, 0..) |v, idx| a2_q[idx] = quant.floatBitsToF32(.bf16, quant.f32ToFloatBits(.bf16, v));
    for (b2, 0..) |v, idx| b2_q[idx] = quant.floatBitsToF32(.bf16, quant.f32ToFloatBits(.bf16, v));

    var ref_row: [n]f32 = undefined;
    // Row 1 & Row 3 (slot1, scale = 16/8 = 2.0, DoRA)
    for ([_]usize{ 1, 3 }) |row| {
        refDoraMergedRow(x_f32[row * k ..][0..k], w0, a1_q, b1_q, mag1, 2.0, n, k, r1, &ref_row);
        for (0..n) |i| try std.testing.expectApproxEqAbs(ref_row[i], y0_f32[row * n + i], 1e-3);
    }
    // Row 2 (slot2, scale = 16/16 = 1.0, DoRA, panel64 bf16)
    refDoraMergedRow(x_f32[2 * k ..][0..k], w0, a2_q, b2_q, mag2, 1.0, n, k, r2, &ref_row);
    for (0..n) |i| try std.testing.expectApproxEqAbs(ref_row[i], y0_f32[2 * n + i], 1e-3);

    // Layer 1 (slot1 across all 4 rows, f16-stored mag1)
    var mag1_f16: [n]f32 = undefined;
    for (mag1, 0..) |v, idx| mag1_f16[idx] = quant.floatBitsToF32(.f16, quant.f32ToFloatBits(.f16, v));
    for (0..m) |row| {
        refDoraMergedRow(x_f32[row * k ..][0..k], w0, a1_q, b1_q, &mag1_f16, 2.0, n, k, r1, &ref_row);
        for (0..n) |i| try std.testing.expectApproxEqAbs(ref_row[i], y1_f32[row * n + i], 1e-3);
    }

    // Hot-unload slot1 & reload a new adapter into the SAME wired pool without recreating GpuLoraWorkspace!
    try reg.unloadSlot(gpa, slot1);
    const slot_re = try reg.loadAdapterFromBytes(gpa, "dora1-reloaded", img2);
    try reg.calibrateDoraPairFromBaseRows(slot_re, 0, .gate_attn, w0);
    for (0..n) |i| {
        var acc: f32 = 0.0;
        for (0..k) |t| acc += w0[i * k + t] * x_f32[t];
        y0_f32[i] = acc;
    }
    var frame2 = try compute.beginFrame(5_000);
    errdefer frame2.abort();
    ws.beginFrame(frame2.slot);
    try std.testing.expect(try pipes.encodeSingleAdapterProjection(&frame2, &ws, &reg, slot_re, 0, .gate_attn, .{
        .x = .{ .buffer = io_buf, .offset = x_off },
        .y = .{ .buffer = io_buf, .offset = y0_off },
        .m = 1,
        .k = k,
        .n = n,
        .x_dtype = .f32,
        .y_dtype = .f32,
        .insert_barrier_before = false,
    }));
    const sig2 = try frame2.commit();
    try compute.wait(sig2, 5_000);
    refDoraMergedRow(x_f32[0..k], w0, a2_q, b2_q, mag2, 1.0, n, k, r2, &ref_row);
    for (0..n) |i| try std.testing.expectApproxEqAbs(ref_row[i], y0_f32[i], 1e-3);

    // Zero-scale DoRA (alpha=0 => scale=0, dora_row_scale still scales y_base!)
    const img_dora_zero = try lora.buildSyntheticAmloraBytes(gpa, "dora-zero", r1, 0.0, &.{
        .{ .layer = 0, .target = .o_proj, .n = n, .k = k, .rank = r1, .layout = .row_major, .dtype = .f16, .a_f32 = a1, .b_f32 = b1, .dora_mag_f32 = mag1, .dora_mag_dtype = .f32 },
    });
    defer gpa.free(img_dora_zero);
    const slot_z = try reg.loadAdapterFromBytes(gpa, "dora-zero", img_dora_zero);
    try reg.calibrateDoraPairFromBaseRows(slot_z, 0, .o_proj, w0);
    for (0..n) |i| {
        var acc: f32 = 0.0;
        for (0..k) |t| acc += w0[i * k + t] * x_f32[t];
        y0_f32[i] = acc;
    }
    var frame3 = try compute.beginFrame(5_000);
    errdefer frame3.abort();
    ws.beginFrame(frame3.slot);
    try std.testing.expect(try pipes.encodeSingleAdapterProjection(&frame3, &ws, &reg, slot_z, 0, .o_proj, .{
        .x = .{ .buffer = io_buf, .offset = x_off },
        .y = .{ .buffer = io_buf, .offset = y0_off },
        .m = 1,
        .k = k,
        .n = n,
        .x_dtype = .f32,
        .y_dtype = .f32,
        .insert_barrier_before = false,
    }));
    const sig3 = try frame3.commit();
    try compute.wait(sig3, 5_000);
    refDoraMergedRow(x_f32[0..k], w0, a1_q, b1_q, mag1, 0.0, n, k, r1, &ref_row);
    for (0..n) |i| try std.testing.expectApproxEqAbs(ref_row[i], y0_f32[i], 1e-3);
}
