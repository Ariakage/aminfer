//! C26 L2-Resident SME2 `gate_rank = 64` Loop Controllers (`Inkling 2`, ADR-0024, ADR-0025).
//!
//! Executes `controllers.{attn,k,v,mlp}.{down_weight,forget_weight,gain_weight}`
//! (`gate_rank = 64`, `sme2_panel64` layout inside a single 16 KiB-aligned
//! `WiredRegion` fitting within M4's 16 MiB P-cluster L2 cache) on CPU SME2
//! (`Placement.cpu_sme2`, 0 Metal dispatches, 0 GPU bus contention, clean
//! automatic NEON/scalar fallback on M1-M3).
//!
//! Recurrent controller update per branch:
//!   z      = down_weight * rmsnorm(h)                 ([m, 64])
//!   forget = sigmoid(forget_weight * z + forget_bias) ([m, state_dim])
//!   gain   = act(gain_weight * z + gain_bias)         ([m, state_dim], tanh or silu)
//!   s'     = forget * s + gain * value_x              ([m, state_dim])
//!
//! Engineering invariants:
//! - Zero dynamic heap allocations on the step hot path.
//! - All packed `sme2_panel64` weights, biases, and intermediate scratch buffers
//!   live in a single 16 KiB (`APPLE_PAGE_SIZE`) page-aligned `WiredArena`.
//! - All projections reuse `src/cpu/sme.zig` (`evalLinearBatchHoisted`) so
//!   `forget` and `gain` share one hoisted `smstart..smstop` window and packed
//!   activation strip.

const std = @import("std");
const page = @import("../mem/page.zig");
const arena_mod = @import("../mem/arena.zig");
const budget_mod = @import("../mem/budget.zig");
const sysinfo = @import("../platform/sysinfo.zig");
const vmstat = @import("../platform/vmstat.zig");
const quant = @import("../quant/spec.zig");
const amf = @import("../format/amf.zig");
const sme = @import("sme.zig");

/// Fixed Inkling-2 loop controller bottleneck rank matching `sme2_panel64` `GROUP_SIZE`.
pub const GATE_RANK: usize = sme.GROUP_SIZE; // 64

/// Four loop controller branches per recurrent Inkling-2 layer.
pub const NUM_BRANCHES: usize = 4;

/// Maximum batch rows per controller step (`M = 1` decode .. `M = 64` verify).
pub const MAX_BATCH_M: usize = sme.MAX_STEP_LORA_M;

/// Default P-cluster L2 residency budget on M4 (16 MiB).
pub const DEFAULT_L2_LIMIT_BYTES: usize = sme.MAX_L2_DRAFT_PROJ_BYTES;

const SVL_ALIGN: usize = sme.SVL_BYTES;

pub const Error = sme.Error ||
    arena_mod.WiredArena.InitError ||
    arena_mod.WiredArena.AllocError ||
    sysinfo.Error ||
    vmstat.Error ||
    error{
        InvalidConfig,
        InvalidWeights,
        ExceedsL2Budget,
        MissingTensor,
        DimensionMismatch,
    };

/// Logical controller branch inside an Inkling-2 recurrent layer.
pub const ControllerBranch = enum(u2) {
    attn = 0,
    k = 1,
    v = 2,
    mlp = 3,

    pub fn label(self: ControllerBranch) []const u8 {
        return switch (self) {
            .attn => "attn",
            .k => "k",
            .v => "v",
            .mlp => "mlp",
        };
    }

    pub fn fromLabel(str: []const u8) error{InvalidConfig}!ControllerBranch {
        if (std.mem.eql(u8, str, "attn")) return .attn;
        if (std.mem.eql(u8, str, "k")) return .k;
        if (std.mem.eql(u8, str, "v")) return .v;
        if (std.mem.eql(u8, str, "mlp")) return .mlp;
        return error.InvalidConfig;
    }
};

/// Nonlinearity applied to the controller gain gate.
pub const GainActivation = enum(u8) {
    tanh = 0,
    silu = 1,
};

/// Effective controller execution placement (matches `Placement.cpu_sme2` vs fallback).
pub const Placement = enum(u8) {
    cpu_sme2 = 0,
    cpu_scalar_fallback = 1,

    pub fn fromBackend(backend: sme.ExecutionBackend) Placement {
        return switch (backend) {
            .sme2 => .cpu_sme2,
            .scalar_fallback => .cpu_scalar_fallback,
        };
    }
};

/// Configuration for one controller branch (`attn`, `k`, `v`, or `mlp`).
pub const BranchConfig = struct {
    /// Width of recurrent state `s` and gated output (`hidden_size` or `kv_dim`).
    state_dim: u32,
    down_dtype: sme.PanelDType = .bf16,
    forget_dtype: sme.PanelDType = .bf16,
    gain_dtype: sme.PanelDType = .bf16,
    gain_activation: GainActivation = .tanh,
    has_norm_weight: bool = false,
    has_forget_bias: bool = false,
    has_gain_bias: bool = false,

    pub fn validate(self: BranchConfig) Error!void {
        if (self.state_dim == 0) return error.InvalidConfig;
        inline for (.{ self.down_dtype, self.forget_dtype, self.gain_dtype }) |dt| {
            if (!isSupportedControllerDType(dt)) return error.InvalidDType;
        }
    }
};

pub fn isSupportedControllerDType(dtype: sme.PanelDType) bool {
    return switch (dtype) {
        .bf16, .f16, .q4_affine => true,
        else => false,
    };
}

/// Top-level configuration for `Inkling2LoopController`.
pub const ControllerConfig = struct {
    /// Input hidden dimension (`H > 0`, must be a multiple of `GROUP_SIZE = 64`).
    hidden_size: u32,
    /// Controller bottleneck width (must equal `GATE_RANK = 64` for `sme2_panel64`).
    gate_rank: u32 = GATE_RANK,
    /// Maximum token rows processed per step (`1..=64`).
    max_m: u32 = 8,
    rms_eps: f32 = 1e-6,
    /// L2 cache residency limit for packed controller weights (default 16 MiB).
    l2_limit_bytes: usize = DEFAULT_L2_LIMIT_BYTES,
    branches: [NUM_BRANCHES]BranchConfig,
    budget_override: ?budget_mod.Budget = null,

    pub fn uniform(
        hidden_size: u32,
        state_dim: u32,
        dtype: sme.PanelDType,
        gain_activation: GainActivation,
        max_m: u32,
    ) ControllerConfig {
        const bc: BranchConfig = .{
            .state_dim = state_dim,
            .down_dtype = dtype,
            .forget_dtype = dtype,
            .gain_dtype = dtype,
            .gain_activation = gain_activation,
        };
        return .{
            .hidden_size = hidden_size,
            .gate_rank = GATE_RANK,
            .max_m = max_m,
            .branches = .{ bc, bc, bc, bc },
        };
    }

    pub fn validate(self: ControllerConfig) Error!void {
        if (self.hidden_size == 0 or (self.hidden_size % GATE_RANK) != 0) {
            return error.InvalidConfig;
        }
        if (self.gate_rank != GATE_RANK) return error.InvalidConfig;
        if (self.max_m == 0 or self.max_m > MAX_BATCH_M) return error.InvalidConfig;
        if (!std.math.isFinite(self.rms_eps) or self.rms_eps <= 0.0) {
            return error.InvalidConfig;
        }
        if (self.l2_limit_bytes == 0) return error.InvalidConfig;
        for (self.branches) |bc| {
            try bc.validate();
        }
    }
};

/// Input weight matrix format packed into `sme2_panel64` inside `WiredArena` at init.
pub const WeightMatrixInput = union(enum) {
    /// Row-major `[rows, cols]` FP32 weights; converted to `bf16`, `f16`, or `q4_affine`.
    dense_f32: []const f32,
    /// Row-major `[rows, cols]` raw 16-bit (`bf16` or `f16`) weights.
    dense_u16: []const u16,
    /// Unpacked `0..15` Q4 integer codes `[rows, cols]` + per-64-group `ScaleBias`.
    q4_affine: struct {
        codes_nk: []const u8,
        scale_bias: []const quant.ScaleBias,
    },
    /// Pre-packed `sme2_panel64` bytes copied directly into wired L2 memory.
    sme2_panel64: []const u8,
};

/// Startup weights and optional biases/norms for a single controller branch.
pub const BranchWeightInput = struct {
    down_weight: WeightMatrixInput,
    forget_weight: WeightMatrixInput,
    gain_weight: WeightMatrixInput,
    norm_weight: ?[]const f32 = null,
    forget_bias: ?[]const f32 = null,
    gain_bias: ?[]const f32 = null,
};

const CompiledBranch = struct {
    config: BranchConfig,
    down_weight: sme.SmeLinearWeight,
    forget_weight: sme.SmeLinearWeight,
    gain_weight: sme.SmeLinearWeight,
    norm_weight: ?[]const f32,
    forget_bias: ?[]const f32,
    gain_bias: ?[]const f32,
    weight_bytes: usize,
};

const ScratchLayout = struct {
    need_norm_bf16: bool,
    need_norm_f16: bool,
    need_z_bf16: bool,
    need_z_f16: bool,
    max_state_dim: usize,
};

fn alignUp64(n: usize) Error!usize {
    const added = std.math.add(usize, n, SVL_ALIGN - 1) catch return error.Overflow;
    return added & ~(SVL_ALIGN - 1);
}

fn defaultWiredBudget() sysinfo.Error!budget_mod.Budget {
    const facts = try sysinfo.Facts.read();
    return budget_mod.Budget.compute(.{
        .memory_bytes = facts.memory_bytes,
        .user_wire_limit = facts.user_wire_limit,
        .global_user_wire_limit = facts.global_user_wire_limit,
        .gpu_working_set_bytes = facts.memory_bytes,
        .configured_cap = null,
        .headroom_bytes = budget_mod.DEFAULT_HEADROOM_BYTES,
    });
}

fn analyzeScratch(cfg: ControllerConfig) ScratchLayout {
    var layout: ScratchLayout = .{
        .need_norm_bf16 = false,
        .need_norm_f16 = false,
        .need_z_bf16 = false,
        .need_z_f16 = false,
        .max_state_dim = 0,
    };
    for (cfg.branches) |bc| {
        if (bc.down_dtype.activationIsBf16()) {
            layout.need_norm_bf16 = true;
        } else {
            layout.need_norm_f16 = true;
        }
        if (bc.forget_dtype.activationIsBf16() or bc.gain_dtype.activationIsBf16()) {
            layout.need_z_bf16 = true;
        }
        if (!bc.forget_dtype.activationIsBf16() or !bc.gain_dtype.activationIsBf16()) {
            layout.need_z_f16 = true;
        }
        layout.max_state_dim = @max(layout.max_state_dim, bc.state_dim);
    }
    return layout;
}

/// L2-resident P-core SME2 loop controller (`gate_rank = 64`, 4 branches:
/// `attn`, `k`, `v`, `mlp`) executing with zero Metal dispatches and zero
/// hot-path dynamic heap allocations.
pub const Inkling2LoopController = struct {
    arena: arena_mod.WiredArena,
    config: ControllerConfig,
    branches: [NUM_BRANCHES]CompiledBranch,
    x_norm_bf16: ?[]align(SVL_ALIGN) u16,
    x_norm_f16: ?[]align(SVL_ALIGN) u16,
    z_f32: []align(SVL_ALIGN) f32,
    z_u16_bf16: ?[]align(SVL_ALIGN) u16,
    z_u16_f16: ?[]align(SVL_ALIGN) u16,
    forget_raw_f32: []align(SVL_ALIGN) f32,
    gain_raw_f32: []align(SVL_ALIGN) f32,
    total_weight_bytes: usize,
    passes_executed: u64 = 0,
    sme_ops_dispatched: u64 = 0,
    metal_dispatches: u64 = 0,

    /// Computes exact 16 KiB-aligned `WiredArena` capacity for weights, optional
    /// norms/biases, and hot-path scratch buffers.
    pub fn requiredBytes(cfg: ControllerConfig) Error!usize {
        try cfg.validate();
        const h: usize = cfg.hidden_size;
        const r: usize = GATE_RANK;
        const m: usize = cfg.max_m;

        var weight_bytes: usize = 0;
        var extra_bytes: usize = 0;
        for (cfg.branches) |bc| {
            const sd: usize = bc.state_dim;
            const d_bytes = try sme.packedBlobBytes(bc.down_dtype, r, h);
            const f_bytes = try sme.packedBlobBytes(bc.forget_dtype, sd, r);
            const g_bytes = try sme.packedBlobBytes(bc.gain_dtype, sd, r);
            weight_bytes = std.math.add(usize, weight_bytes, try alignUp64(d_bytes)) catch return error.Overflow;
            weight_bytes = std.math.add(usize, weight_bytes, try alignUp64(f_bytes)) catch return error.Overflow;
            weight_bytes = std.math.add(usize, weight_bytes, try alignUp64(g_bytes)) catch return error.Overflow;

            if (bc.has_norm_weight) {
                extra_bytes = std.math.add(usize, extra_bytes, try alignUp64(h * @sizeOf(f32))) catch return error.Overflow;
            }
            if (bc.has_forget_bias) {
                extra_bytes = std.math.add(usize, extra_bytes, try alignUp64(sd * @sizeOf(f32))) catch return error.Overflow;
            }
            if (bc.has_gain_bias) {
                extra_bytes = std.math.add(usize, extra_bytes, try alignUp64(sd * @sizeOf(f32))) catch return error.Overflow;
            }
        }
        if (weight_bytes > cfg.l2_limit_bytes) return error.ExceedsL2Budget;

        const sl = analyzeScratch(cfg);
        var scratch_bytes: usize = 0;
        if (sl.need_norm_bf16) {
            scratch_bytes = std.math.add(usize, scratch_bytes, try alignUp64(m * h * @sizeOf(u16))) catch return error.Overflow;
        }
        if (sl.need_norm_f16) {
            scratch_bytes = std.math.add(usize, scratch_bytes, try alignUp64(m * h * @sizeOf(u16))) catch return error.Overflow;
        }
        scratch_bytes = std.math.add(usize, scratch_bytes, try alignUp64(m * r * @sizeOf(f32))) catch return error.Overflow;
        if (sl.need_z_bf16) {
            scratch_bytes = std.math.add(usize, scratch_bytes, try alignUp64(m * r * @sizeOf(u16))) catch return error.Overflow;
        }
        if (sl.need_z_f16) {
            scratch_bytes = std.math.add(usize, scratch_bytes, try alignUp64(m * r * @sizeOf(u16))) catch return error.Overflow;
        }
        const raw_gate_bytes = try alignUp64((std.math.mul(usize, m, sl.max_state_dim) catch return error.Overflow) * @sizeOf(f32));
        scratch_bytes = std.math.add(usize, scratch_bytes, raw_gate_bytes) catch return error.Overflow;
        scratch_bytes = std.math.add(usize, scratch_bytes, raw_gate_bytes) catch return error.Overflow;

        const total = std.math.add(usize, std.math.add(usize, weight_bytes, extra_bytes) catch return error.Overflow, scratch_bytes) catch return error.Overflow;
        return page.alignUp(total) catch return error.Overflow;
    }

    /// Initializes the controller, packing all 4 branches into a single 16 KiB-aligned
    /// `WiredRegion` verified to fit inside `cfg.l2_limit_bytes` (default 16 MiB L2).
    pub fn init(
        cfg: ControllerConfig,
        inputs: [NUM_BRANCHES]BranchWeightInput,
    ) Error!Inkling2LoopController {
        // Resolve effective config flags for optional norms/biases from actual inputs.
        var eff_cfg = cfg;
        for (0..NUM_BRANCHES) |i| {
            eff_cfg.branches[i].has_norm_weight = inputs[i].norm_weight != null;
            eff_cfg.branches[i].has_forget_bias = inputs[i].forget_bias != null;
            eff_cfg.branches[i].has_gain_bias = inputs[i].gain_bias != null;
        }

        const cap_bytes = try requiredBytes(eff_cfg);
        const wired_budget = eff_cfg.budget_override orelse try defaultWiredBudget();
        try wired_budget.checkPlan(cap_bytes);

        var mem_arena = try arena_mod.WiredArena.init(cap_bytes, wired_budget);
        errdefer mem_arena.deinit();

        const h: usize = eff_cfg.hidden_size;
        const r: usize = GATE_RANK;
        const m: usize = eff_cfg.max_m;

        var compiled: [NUM_BRANCHES]CompiledBranch = undefined;
        var total_weight_bytes: usize = 0;

        for (0..NUM_BRANCHES) |bi| {
            const bc = eff_cfg.branches[bi];
            const inp = inputs[bi];
            const sd: usize = bc.state_dim;

            const w_down = try packMatrixIntoArena(&mem_arena, bc.down_dtype, inp.down_weight, r, h);
            const w_forget = try packMatrixIntoArena(&mem_arena, bc.forget_dtype, inp.forget_weight, sd, r);
            const w_gain = try packMatrixIntoArena(&mem_arena, bc.gain_dtype, inp.gain_weight, sd, r);

            const norm_w = if (inp.norm_weight) |nw| blk: {
                if (nw.len != h) return error.InvalidWeights;
                break :blk try copyF32IntoArena(&mem_arena, nw);
            } else null;
            const forget_b = if (inp.forget_bias) |fb| blk: {
                if (fb.len != sd) return error.InvalidWeights;
                break :blk try copyF32IntoArena(&mem_arena, fb);
            } else null;
            const gain_b = if (inp.gain_bias) |gb| blk: {
                if (gb.len != sd) return error.InvalidWeights;
                break :blk try copyF32IntoArena(&mem_arena, gb);
            } else null;

            const branch_w_bytes = w_down.byteSize() + w_forget.byteSize() + w_gain.byteSize();
            total_weight_bytes += branch_w_bytes;

            compiled[bi] = .{
                .config = bc,
                .down_weight = w_down,
                .forget_weight = w_forget,
                .gain_weight = w_gain,
                .norm_weight = norm_w,
                .forget_bias = forget_b,
                .gain_bias = gain_b,
                .weight_bytes = branch_w_bytes,
            };
        }

        if (total_weight_bytes > eff_cfg.l2_limit_bytes) {
            return error.ExceedsL2Budget;
        }

        const sl = analyzeScratch(eff_cfg);
        const x_norm_bf16 = if (sl.need_norm_bf16) try allocAlignedSlice(u16, &mem_arena, m * h) else null;
        const x_norm_f16 = if (sl.need_norm_f16) try allocAlignedSlice(u16, &mem_arena, m * h) else null;
        const z_f32 = try allocAlignedSlice(f32, &mem_arena, m * r);
        const z_u16_bf16 = if (sl.need_z_bf16) try allocAlignedSlice(u16, &mem_arena, m * r) else null;
        const z_u16_f16 = if (sl.need_z_f16) try allocAlignedSlice(u16, &mem_arena, m * r) else null;
        const forget_raw_f32 = try allocAlignedSlice(f32, &mem_arena, m * sl.max_state_dim);
        const gain_raw_f32 = try allocAlignedSlice(f32, &mem_arena, m * sl.max_state_dim);

        if (x_norm_bf16) |s| @memset(s, 0);
        if (x_norm_f16) |s| @memset(s, 0);
        @memset(z_f32, 0.0);
        if (z_u16_bf16) |s| @memset(s, 0);
        if (z_u16_f16) |s| @memset(s, 0);
        @memset(forget_raw_f32, 0.0);
        @memset(gain_raw_f32, 0.0);

        return .{
            .arena = mem_arena,
            .config = eff_cfg,
            .branches = compiled,
            .x_norm_bf16 = x_norm_bf16,
            .x_norm_f16 = x_norm_f16,
            .z_f32 = z_f32,
            .z_u16_bf16 = z_u16_bf16,
            .z_u16_f16 = z_u16_f16,
            .forget_raw_f32 = forget_raw_f32,
            .gain_raw_f32 = gain_raw_f32,
            .total_weight_bytes = total_weight_bytes,
        };
    }

    /// Loads `layers.{layer}.controllers.{attn,k,v,mlp}.{down_weight,forget_weight,gain_weight}`
    /// (plus optional `norm_weight`, `forget_bias`, `gain_bias`) from an opened `.amf`
    /// file into a 16 KiB-aligned `WiredRegion` fitting in P-cluster L2.
    pub fn loadFromAmfLayer(
        allocator: std.mem.Allocator,
        amf_file: *const amf.AmfFile,
        layer: u16,
        cfg: ControllerConfig,
    ) (Error || amf.ParseError || amf.IoError || error{OutOfMemory})!Inkling2LoopController {
        try cfg.validate();
        const h: usize = cfg.hidden_size;
        const r: usize = GATE_RANK;

        var loaded_bufs: [NUM_BRANCHES * 6]?[]align(4) u8 = @splat(null);
        var buf_count: usize = 0;
        defer {
            for (0..buf_count) |i| {
                if (loaded_bufs[i]) |b| allocator.free(b);
            }
        }

        var inputs: [NUM_BRANCHES]BranchWeightInput = undefined;
        var name_buf: [64]u8 = undefined;

        inline for (0..NUM_BRANCHES) |bi| {
            const b_enum: ControllerBranch = @fromBackingInt(@intCast(bi));
            const b_name = b_enum.label();
            const bc = cfg.branches[bi];
            const sd: usize = bc.state_dim;

            const down_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.controllers.{s}.down_weight", .{ layer, b_name }) catch return error.Overflow;
            const down_in = try loadAmfWeightInput(allocator, amf_file, down_nm, bc.down_dtype, r, h, &loaded_bufs, &buf_count);

            const forget_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.controllers.{s}.forget_weight", .{ layer, b_name }) catch return error.Overflow;
            const forget_in = try loadAmfWeightInput(allocator, amf_file, forget_nm, bc.forget_dtype, sd, r, &loaded_bufs, &buf_count);

            const gain_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.controllers.{s}.gain_weight", .{ layer, b_name }) catch return error.Overflow;
            const gain_in = try loadAmfWeightInput(allocator, amf_file, gain_nm, bc.gain_dtype, sd, r, &loaded_bufs, &buf_count);

            const norm_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.controllers.{s}.norm_weight", .{ layer, b_name }) catch return error.Overflow;
            const norm_w = try loadOptionalAmfF32(allocator, amf_file, norm_nm, h, &loaded_bufs, &buf_count);

            const fb_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.controllers.{s}.forget_bias", .{ layer, b_name }) catch return error.Overflow;
            const fb_w = try loadOptionalAmfF32(allocator, amf_file, fb_nm, sd, &loaded_bufs, &buf_count);

            const gb_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.controllers.{s}.gain_bias", .{ layer, b_name }) catch return error.Overflow;
            const gb_w = try loadOptionalAmfF32(allocator, amf_file, gb_nm, sd, &loaded_bufs, &buf_count);

            inputs[bi] = .{
                .down_weight = down_in,
                .forget_weight = forget_in,
                .gain_weight = gain_in,
                .norm_weight = norm_w,
                .forget_bias = fb_w,
                .gain_bias = gb_w,
            };
        }

        return init(cfg, inputs);
    }

    pub fn deinit(self: *Inkling2LoopController) void {
        self.x_norm_bf16 = null;
        self.x_norm_f16 = null;
        self.z_f32 = &.{};
        self.z_u16_bf16 = null;
        self.z_u16_f16 = null;
        self.forget_raw_f32 = &.{};
        self.gain_raw_f32 = &.{};
        self.arena.deinit();
    }

    pub fn totalWeightBytes(self: *const Inkling2LoopController) usize {
        return self.total_weight_bytes;
    }

    pub fn totalWiredBytes(self: *const Inkling2LoopController) u64 {
        return self.arena.bytes().len;
    }

    pub fn fitsL2(self: *const Inkling2LoopController, limit_override: ?usize) bool {
        return self.total_weight_bytes <= (limit_override orelse self.config.l2_limit_bytes);
    }

    pub fn effectivePlacement(
        dispatcher: *const sme.SmeDispatcher,
        backend_override: ?sme.ExecutionBackend,
    ) Error!Placement {
        const backend = try dispatcher.caps.resolveBackend(backend_override);
        return Placement.fromBackend(backend);
    }

    /// Executes one recurrent controller branch (`attn`, `k`, `v`, or `mlp`)
    /// when `state_dim == hidden_size`, updating `s_next = sigmoid(...) * s_prev + act(...) * h`.
    /// Aliasing `s_prev` and `s_next` for in-place state update is supported.
    /// Zero dynamic heap allocations, zero Metal dispatches.
    pub fn stepBranch(
        self: *Inkling2LoopController,
        dispatcher: *sme.SmeDispatcher,
        branch: ControllerBranch,
        h_f32: []const f32,
        s_prev_f32: []const f32,
        s_next_f32: []f32,
        m: usize,
        backend_override: ?sme.ExecutionBackend,
    ) Error!void {
        const bc = self.branches[@backingInt(branch)].config;
        if (bc.state_dim != self.config.hidden_size) return error.DimensionMismatch;
        return self.stepBranchWithValue(dispatcher, branch, h_f32, h_f32, s_prev_f32, s_next_f32, m, backend_override);
    }

    /// Executes one controller branch where `h_f32` (`[m, hidden_size]`) drives
    /// the bottleneck `z = down_weight * rmsnorm(h)` and `value_x_f32` (`[m, state_dim]`)
    /// is gated into `s_next = forget * s_prev + gain * value_x`.
    /// Zero dynamic heap allocations, zero Metal dispatches.
    pub fn stepBranchWithValue(
        self: *Inkling2LoopController,
        dispatcher: *sme.SmeDispatcher,
        branch: ControllerBranch,
        h_f32: []const f32,
        value_x_f32: []const f32,
        s_prev_f32: []const f32,
        s_next_f32: []f32,
        m: usize,
        backend_override: ?sme.ExecutionBackend,
    ) Error!void {
        if (m == 0 or m > self.config.max_m) return error.InvalidArgument;
        const cb = &self.branches[@backingInt(branch)];
        const h: usize = self.config.hidden_size;
        const r: usize = GATE_RANK;
        const sd: usize = cb.config.state_dim;

        const mh = std.math.mul(usize, m, h) catch return error.Overflow;
        const mr = std.math.mul(usize, m, r) catch return error.Overflow;
        const msd = std.math.mul(usize, m, sd) catch return error.Overflow;

        if (h_f32.len < mh or value_x_f32.len < msd or s_prev_f32.len < msd or s_next_f32.len < msd) {
            return error.BufferTooSmall;
        }

        // 1. Vectorized RMSNorm(h) -> u16 activations matching `down_weight` dtype.
        const down_is_bf16 = cb.down_weight.dtype.activationIsBf16();
        const x_norm = if (down_is_bf16) self.x_norm_bf16.?[0..mh] else self.x_norm_f16.?[0..mh];
        const act_dt_down: quant.FloatDType = if (down_is_bf16) .bf16 else .f16;
        for (0..m) |row| {
            rmsNormRowFast(
                h_f32[row * h ..][0..h],
                cb.norm_weight,
                self.config.rms_eps,
                act_dt_down,
                x_norm[row * h ..][0..h],
            );
        }

        // 2. z = down_weight * rmsnorm(h) ([m, 64]) via hoisted SME2 (or fallback).
        const z_out = self.z_f32[0..mr];
        const down_op = [_]sme.BatchLinearOp{.{
            .weight = cb.down_weight,
            .a_u16 = x_norm,
            .out_bytes = std.mem.sliceAsBytes(z_out),
            .m = m,
            .out_dtype = .f32,
            .cache_hint = .temporal,
        }};
        try dispatcher.evalLinearBatchHoisted(&down_op, backend_override);

        // 3. Convert z ([m, 64]) to activation format(s) required by forget & gain weights.
        const forget_is_bf16 = cb.forget_weight.dtype.activationIsBf16();
        const gain_is_bf16 = cb.gain_weight.dtype.activationIsBf16();

        const z_forget_u16 = if (forget_is_bf16) self.z_u16_bf16.?[0..mr] else self.z_u16_f16.?[0..mr];
        convertF32ToFloatBitsSlice(z_out, if (forget_is_bf16) .bf16 else .f16, z_forget_u16);

        const z_gain_u16 = if (gain_is_bf16 == forget_is_bf16)
            z_forget_u16
        else blk: {
            const dst = if (gain_is_bf16) self.z_u16_bf16.?[0..mr] else self.z_u16_f16.?[0..mr];
            convertF32ToFloatBitsSlice(z_out, if (gain_is_bf16) .bf16 else .f16, dst);
            break :blk dst;
        };

        // 4. Hoisted `smstart..smstop` batch for `forget_weight * z` and `gain_weight * z`,
        //    sharing packed activation strip whenever both weights share activation format.
        const forget_raw = self.forget_raw_f32[0..msd];
        const gain_raw = self.gain_raw_f32[0..msd];
        const gate_ops = [_]sme.BatchLinearOp{
            .{
                .weight = cb.forget_weight,
                .a_u16 = z_forget_u16,
                .out_bytes = std.mem.sliceAsBytes(forget_raw),
                .m = m,
                .out_dtype = .f32,
                .cache_hint = .temporal,
            },
            .{
                .weight = cb.gain_weight,
                .a_u16 = z_gain_u16,
                .out_bytes = std.mem.sliceAsBytes(gain_raw),
                .m = m,
                .out_dtype = .f32,
                .cache_hint = .temporal,
            },
        };
        try dispatcher.evalLinearBatchHoisted(&gate_ops, backend_override);

        // 5. Fused gated recurrent state update: s' = sigmoid(f) * s + act(g) * value_x.
        applyGatedStateUpdate(
            forget_raw,
            gain_raw,
            cb.forget_bias,
            cb.gain_bias,
            cb.config.gain_activation,
            s_prev_f32[0..msd],
            value_x_f32[0..msd],
            s_next_f32[0..msd],
            m,
            sd,
        );

        self.passes_executed += 1;
        self.sme_ops_dispatched += 3;
    }

    /// Executes exact `Inkling2LoopMessageGate` (HuggingFace `modeling_inkling2_loop.py`):
    ///   `gate_in = [UnweightedRMSNorm(cur), UnweightedRMSNorm(msg)]` (`[m, 2*state_dim]`),
    ///   `z = down_weight @ gate_in` (`[m, 64]`, L2-resident SME2),
    ///   `forget = tanh(forget_weight @ z)`, `gain = tanh(gain_weight @ z)` (`[m, state_dim]`, SME2),
    ///   `out = (1 - forget) * cur + gain * msg`.
    /// Zero GPU dispatches and zero dynamic heap allocations.
    pub fn stepMessageGate(
        self: *Inkling2LoopController,
        dispatcher: *sme.SmeDispatcher,
        branch: ControllerBranch,
        cur_f32: []const f32,
        msg_f32: []const f32,
        out_f32: []f32,
        m: usize,
        backend_override: ?sme.ExecutionBackend,
    ) Error!void {
        if (m == 0 or m > self.config.max_m) return error.InvalidArgument;
        const cb = &self.branches[@backingInt(branch)];
        const h: usize = self.config.hidden_size;
        const r: usize = GATE_RANK;
        const sd: usize = cb.config.state_dim;

        const mh = std.math.mul(usize, m, h) catch return error.Overflow;
        const mr = std.math.mul(usize, m, r) catch return error.Overflow;
        const msd = std.math.mul(usize, m, sd) catch return error.Overflow;
        if (cur_f32.len < msd or msg_f32.len < msd or out_f32.len < msd) {
            return error.BufferTooSmall;
        }

        const down_is_bf16 = cb.down_weight.dtype.activationIsBf16();
        const x_norm = if (down_is_bf16) self.x_norm_bf16.?[0..mh] else self.x_norm_f16.?[0..mh];
        const act_dt_down: quant.FloatDType = if (down_is_bf16) .bf16 else .f16;
        for (0..m) |row| {
            const c_row = cur_f32[row * sd ..][0..sd];
            const m_row = msg_f32[row * sd ..][0..sd];
            if (h == sd * 2) {
                rmsNormRowFast(c_row, null, self.config.rms_eps, act_dt_down, x_norm[row * h ..][0..sd]);
                rmsNormRowFast(m_row, null, self.config.rms_eps, act_dt_down, x_norm[row * h + sd ..][0..sd]);
            } else if (h == sd) {
                rmsNormRowFast(c_row, null, self.config.rms_eps, act_dt_down, x_norm[row * h ..][0..sd]);
            } else {
                return error.InvalidArgument;
            }
        }

        const z_out = self.z_f32[0..mr];
        const down_op = [_]sme.BatchLinearOp{.{
            .weight = cb.down_weight,
            .a_u16 = x_norm,
            .out_bytes = std.mem.sliceAsBytes(z_out),
            .m = m,
            .out_dtype = .f32,
            .cache_hint = .temporal,
        }};
        try dispatcher.evalLinearBatchHoisted(&down_op, backend_override);

        const forget_is_bf16 = cb.forget_weight.dtype.activationIsBf16();
        const gain_is_bf16 = cb.gain_weight.dtype.activationIsBf16();
        const z_forget_u16 = if (forget_is_bf16) self.z_u16_bf16.?[0..mr] else self.z_u16_f16.?[0..mr];
        convertF32ToFloatBitsSlice(z_out, if (forget_is_bf16) .bf16 else .f16, z_forget_u16);
        const z_gain_u16 = if (gain_is_bf16 == forget_is_bf16)
            z_forget_u16
        else blk: {
            const dst = if (gain_is_bf16) self.z_u16_bf16.?[0..mr] else self.z_u16_f16.?[0..mr];
            convertF32ToFloatBitsSlice(z_out, if (gain_is_bf16) .bf16 else .f16, dst);
            break :blk dst;
        };

        const forget_raw = self.forget_raw_f32[0..msd];
        const gain_raw = self.gain_raw_f32[0..msd];
        const gate_ops = [_]sme.BatchLinearOp{
            .{
                .weight = cb.forget_weight,
                .a_u16 = z_forget_u16,
                .out_bytes = std.mem.sliceAsBytes(forget_raw),
                .m = m,
                .out_dtype = .f32,
                .cache_hint = .temporal,
            },
            .{
                .weight = cb.gain_weight,
                .a_u16 = z_gain_u16,
                .out_bytes = std.mem.sliceAsBytes(gain_raw),
                .m = m,
                .out_dtype = .f32,
                .cache_hint = .temporal,
            },
        };
        try dispatcher.evalLinearBatchHoisted(&gate_ops, backend_override);

        for (0..msd) |i| {
            const fg = std.math.tanh(forget_raw[i]);
            const mg = std.math.tanh(gain_raw[i]);
            out_f32[i] = (1.0 - fg) * cur_f32[i] + mg * msg_f32[i];
        }

        self.passes_executed += 1;
        self.sme_ops_dispatched += 3;
    }

    /// Executes all 4 branches (`attn`, `k`, `v`, `mlp`) sequentially on shared `h_f32`
    /// when each branch has `state_dim == hidden_size`.
    pub fn stepAllBranches(
        self: *Inkling2LoopController,
        dispatcher: *sme.SmeDispatcher,
        h_f32: []const f32,
        s_prev_all: [NUM_BRANCHES][]const f32,
        s_next_all: [NUM_BRANCHES][]f32,
        m: usize,
        backend_override: ?sme.ExecutionBackend,
    ) Error!void {
        inline for (0..NUM_BRANCHES) |bi| {
            try self.stepBranch(
                dispatcher,
                @fromBackingInt(@intCast(bi)),
                h_f32,
                s_prev_all[bi],
                s_next_all[bi],
                m,
                backend_override,
            );
        }
    }
};

/// Standalone FP32 reference implementation for one Inkling-2 loop controller branch.
pub fn stepBranchFp32Reference(
    h_f32: []const f32,
    value_x_f32: []const f32,
    s_prev_f32: []const f32,
    down_weight_f32: []const f32,
    forget_weight_f32: []const f32,
    gain_weight_f32: []const f32,
    norm_weight_f32: ?[]const f32,
    forget_bias_f32: ?[]const f32,
    gain_bias_f32: ?[]const f32,
    gain_activation: GainActivation,
    rms_eps: f32,
    m: usize,
    hidden_size: usize,
    state_dim: usize,
    s_next_f32: []f32,
) Error!void {
    if (m == 0 or hidden_size == 0 or state_dim == 0) return error.InvalidArgument;
    if (down_weight_f32.len != GATE_RANK * hidden_size or
        forget_weight_f32.len != state_dim * GATE_RANK or
        gain_weight_f32.len != state_dim * GATE_RANK)
    {
        return error.InvalidWeights;
    }
    if (h_f32.len < m * hidden_size or
        value_x_f32.len < m * state_dim or
        s_prev_f32.len < m * state_dim or
        s_next_f32.len < m * state_dim)
    {
        return error.BufferTooSmall;
    }

    var x_norm_stack: [4096]f32 = undefined;
    if (hidden_size > x_norm_stack.len) return error.BufferTooSmall;
    const x_norm = x_norm_stack[0..hidden_size];

    var z: [GATE_RANK]f32 = undefined;
    for (0..m) |row| {
        const h_row = h_f32[row * hidden_size ..][0..hidden_size];
        var sum_sq: f64 = 0.0;
        for (h_row) |v| {
            const vf: f64 = v;
            sum_sq += vf * vf;
        }
        const inv_rms: f32 = @floatCast(1.0 / @sqrt(sum_sq / @as(f64, @floatFromInt(hidden_size)) + @as(f64, rms_eps)));
        if (norm_weight_f32) |nw| {
            for (0..hidden_size) |d| x_norm[d] = (h_row[d] * inv_rms) * nw[d];
        } else {
            for (0..hidden_size) |d| x_norm[d] = h_row[d] * inv_rms;
        }

        for (0..GATE_RANK) |r_idx| {
            const w_row = down_weight_f32[r_idx * hidden_size ..][0..hidden_size];
            var acc: f32 = 0.0;
            for (0..hidden_size) |d| acc += w_row[d] * x_norm[d];
            z[r_idx] = acc;
        }

        const base = row * state_dim;
        for (0..state_dim) |d| {
            const fw_row = forget_weight_f32[d * GATE_RANK ..][0..GATE_RANK];
            const gw_row = gain_weight_f32[d * GATE_RANK ..][0..GATE_RANK];
            var f_acc: f32 = if (forget_bias_f32) |fb| fb[d] else 0.0;
            var g_acc: f32 = if (gain_bias_f32) |gb| gb[d] else 0.0;
            for (0..GATE_RANK) |r_idx| {
                f_acc += fw_row[r_idx] * z[r_idx];
                g_acc += gw_row[r_idx] * z[r_idx];
            }
            const f_gate = sigmoidF32(f_acc);
            const g_gate = switch (gain_activation) {
                .tanh => std.math.tanh(g_acc),
                .silu => siluF32(g_acc),
            };
            s_next_f32[base + d] = f_gate * s_prev_f32[base + d] + g_gate * value_x_f32[base + d];
        }
    }
}

/// Quantizes a row-major `[rows, cols]` FP32 matrix into per-64-group affine Q4
/// codes (`0..15`) and `ScaleBias` headers for `sme2_panel64` packing.
pub fn quantizeF32ToQ4Affine(
    src_f32: []const f32,
    rows: usize,
    cols: usize,
    out_codes_nk: []u8,
    out_scale_bias: []quant.ScaleBias,
) Error!void {
    if (rows == 0 or cols == 0) return error.ZeroDimension;
    if ((cols % GATE_RANK) != 0) return error.UnalignedK;
    const groups = cols / GATE_RANK;
    if (src_f32.len < rows * cols or out_codes_nk.len < rows * cols or out_scale_bias.len < rows * groups) {
        return error.BufferTooSmall;
    }
    for (0..rows) |r| {
        for (0..groups) |g| {
            const src_grp = src_f32[r * cols + g * GATE_RANK ..][0..GATE_RANK];
            var min_v: f32 = src_grp[0];
            var max_v: f32 = src_grp[0];
            for (src_grp[1..]) |v| {
                if (v < min_v) min_v = v;
                if (v > max_v) max_v = v;
            }
            const span = max_v - min_v;
            const scale_f32: f32 = if (span > 1e-12) span / 15.0 else 0.0;
            const sb: quant.ScaleBias = .{
                .scale = @floatCast(scale_f32),
                .bias = @floatCast(min_v),
            };
            out_scale_bias[r * groups + g] = sb;
            const eff_scale: f32 = @floatCast(sb.scale);
            const eff_bias: f32 = @floatCast(sb.bias);
            const dst_grp = out_codes_nk[r * cols + g * GATE_RANK ..][0..GATE_RANK];
            for (0..GATE_RANK) |j| {
                const q: u8 = if (eff_scale > 0.0) blk: {
                    const scaled = std.math.clamp(@round((src_grp[j] - eff_bias) / eff_scale), 0.0, 15.0);
                    break :blk @intFromFloat(scaled);
                } else 0;
                dst_grp[j] = q;
            }
        }
    }
}

/// Decodes packed Q4 codes + `ScaleBias` back to FP32 for quantized parity checks.
pub fn dequantizeQ4AffineToF32(
    codes_nk: []const u8,
    scale_bias: []const quant.ScaleBias,
    rows: usize,
    cols: usize,
    out_f32: []f32,
) Error!void {
    if (rows == 0 or cols == 0) return error.ZeroDimension;
    if ((cols % GATE_RANK) != 0) return error.UnalignedK;
    const groups = cols / GATE_RANK;
    if (codes_nk.len < rows * cols or scale_bias.len < rows * groups or out_f32.len < rows * cols) {
        return error.BufferTooSmall;
    }
    for (0..rows) |r| {
        for (0..groups) |g| {
            const sb = scale_bias[r * groups + g];
            const s: f32 = @floatCast(sb.scale);
            const b: f32 = @floatCast(sb.bias);
            const src_grp = codes_nk[r * cols + g * GATE_RANK ..][0..GATE_RANK];
            const dst_grp = out_f32[r * cols + g * GATE_RANK ..][0..GATE_RANK];
            for (0..GATE_RANK) |j| {
                dst_grp[j] = @as(f32, @floatFromInt(src_grp[j] & 0x0F)) * s + b;
            }
        }
    }
}

pub const BenchmarkReport = struct {
    hidden_size: u32,
    gate_rank: u32,
    dtype: sme.PanelDType,
    m: usize,
    iters: usize,
    us_per_branch: f64,
    us_per_full_pass_4_branches: f64,
    total_weight_bytes: usize,
    placement: Placement,
};

/// Benchmarks controller pass latency (`us/pass` for single branch & 4-branch full pass).
pub fn benchmarkControllerPass(
    ctrl: *Inkling2LoopController,
    dispatcher: *sme.SmeDispatcher,
    m: usize,
    iters: usize,
    backend_override: ?sme.ExecutionBackend,
) Error!BenchmarkReport {
    if (m == 0 or m > ctrl.config.max_m or iters == 0) return error.InvalidArgument;
    const h: usize = ctrl.config.hidden_size;
    const sd: usize = ctrl.branches[0].config.state_dim;
    if (h > 4096 or sd > 4096 or m > 16) return error.BufferTooSmall;

    var h_buf: [16 * 4096]f32 = undefined;
    var s_buf: [NUM_BRANCHES][16 * 4096]f32 = undefined;
    for (0..m * h) |i| {
        h_buf[i] = @sin(@as(f32, @floatFromInt((i % 97) + 1))) * 0.25;
    }
    for (0..NUM_BRANCHES) |bi| {
        for (0..m * sd) |i| {
            s_buf[bi][i] = @cos(@as(f32, @floatFromInt((i + bi * 13) % 89))) * 0.25;
        }
    }

    const s_prev = [NUM_BRANCHES][]const f32{
        s_buf[0][0 .. m * sd],
        s_buf[1][0 .. m * sd],
        s_buf[2][0 .. m * sd],
        s_buf[3][0 .. m * sd],
    };
    const s_next = [NUM_BRANCHES][]f32{
        s_buf[0][0 .. m * sd],
        s_buf[1][0 .. m * sd],
        s_buf[2][0 .. m * sd],
        s_buf[3][0 .. m * sd],
    };

    // Warmup pass.
    try ctrl.stepAllBranches(dispatcher, h_buf[0 .. m * h], s_prev, s_next, m, backend_override);

    const t0 = try vmstat.nowNs();
    for (0..iters) |_| {
        try ctrl.stepAllBranches(dispatcher, h_buf[0 .. m * h], s_prev, s_next, m, backend_override);
    }
    const t1 = try vmstat.nowNs();
    const ns: f64 = @floatFromInt(t1 -| t0);
    const us_per_full_pass = ns / (@as(f64, @floatFromInt(iters)) * 1000.0);

    return .{
        .hidden_size = ctrl.config.hidden_size,
        .gate_rank = ctrl.config.gate_rank,
        .dtype = ctrl.branches[0].down_weight.dtype,
        .m = m,
        .iters = iters,
        .us_per_branch = us_per_full_pass / @as(f64, NUM_BRANCHES),
        .us_per_full_pass_4_branches = us_per_full_pass,
        .total_weight_bytes = ctrl.total_weight_bytes,
        .placement = try Inkling2LoopController.effectivePlacement(dispatcher, backend_override),
    };
}

fn allocAlignedSlice(
    comptime T: type,
    arena: *arena_mod.WiredArena,
    count: usize,
) Error![]align(SVL_ALIGN) T {
    const raw_bytes = std.math.mul(usize, count, @sizeOf(T)) catch return error.Overflow;
    const raw = try arena.allocAligned(raw_bytes, SVL_ALIGN);
    const typed: []align(SVL_ALIGN) T = @alignCast(std.mem.bytesAsSlice(T, raw));
    return typed[0..count];
}

fn copyF32IntoArena(arena: *arena_mod.WiredArena, src: []const f32) Error![]const f32 {
    const dst = try allocAlignedSlice(f32, arena, src.len);
    @memcpy(dst, src);
    return dst;
}

fn packMatrixIntoArena(
    arena: *arena_mod.WiredArena,
    dtype: sme.PanelDType,
    input: WeightMatrixInput,
    n: usize,
    k: usize,
) Error!sme.SmeLinearWeight {
    const packed_bytes = try sme.packedBlobBytes(dtype, n, k);
    const dst_raw = try arena.allocAligned(packed_bytes, SVL_ALIGN);

    switch (input) {
        .dense_f32 => |w_f32| {
            const nk = std.math.mul(usize, n, k) catch return error.Overflow;
            if (w_f32.len != nk) return error.InvalidWeights;
            if (dtype == .q4_affine) {
                if ((k % GATE_RANK) != 0) return error.UnalignedK;
                const groups = k / GATE_RANK;
                // Quantize groupwise using fixed stack/inline chunking when small, or
                // pack directly group-by-group without dynamic heap allocation!
                try packF32AsQ4Panel64NoHeap(w_f32, n, k, groups, dst_raw[0..packed_bytes]);
            } else {
                try sme.packDenseFromF32Panel64(dtype, w_f32, n, k, dst_raw[0..packed_bytes]);
            }
        },
        .dense_u16 => |w_u16| {
            if (dtype.isQuantized()) return error.InvalidDType;
            const nk = std.math.mul(usize, n, k) catch return error.Overflow;
            if (w_u16.len != nk) return error.InvalidWeights;
            try sme.packDensePanel64(dtype, w_u16, n, k, dst_raw[0..packed_bytes]);
        },
        .q4_affine => |q| {
            if (dtype != .q4_affine) return error.InvalidDType;
            try sme.packQuantAffinePanel64(.q4_affine, q.codes_nk, q.scale_bias, n, k, dst_raw[0..packed_bytes]);
        },
        .sme2_panel64 => |blob| {
            if (blob.len != packed_bytes) return error.InvalidWeights;
            @memcpy(dst_raw[0..packed_bytes], blob);
        },
    }

    return .{
        .dtype = dtype,
        .param_kind = .affine_scale_bias,
        .cache_hint = .temporal,
        .n = n,
        .k = k,
        .w_packed = dst_raw[0..packed_bytes],
    };
}

/// Packs FP32 `[n, k]` row-major weights directly into canonical Q4 `sme2_panel64`
/// layout (`640 B/group`: 64 B duplicated `f16` scales + 64 B duplicated `f16` biases
/// + 512 B packed 4-bit nibbles, matching `sme.packQuantAffinePanel64`) panel-by-panel
/// using strictly fixed stack buffers (`0` heap allocations).
fn packF32AsQ4Panel64NoHeap(
    w_f32: []const f32,
    n: usize,
    k: usize,
    groups: usize,
    dst: []u8,
) Error!void {
    comptime {
        if (GATE_RANK != sme.GROUP_SIZE) @compileError("GATE_RANK must equal sme.GROUP_SIZE");
        if (sme.groupBytes(.q4_affine) != sme.QUANT_HEADER_BYTES + 32 * sme.PANEL_COLS) {
            @compileError("q4_affine panel64 group stride mismatch");
        }
    }
    const panel_cols = sme.PANEL_COLS; // 16
    const panels = sme.panelCount(n);
    const grp_bytes: usize = sme.groupBytes(.q4_affine); // 640 B = 128 B header + 512 B codes
    const pg = std.math.mul(usize, panels, groups) catch return error.Overflow;
    const total_bytes = std.math.mul(usize, pg, grp_bytes) catch return error.Overflow;
    if (dst.len < total_bytes) return error.BufferTooSmall;

    var out_off: usize = 0;
    for (0..panels) |p| {
        for (0..groups) |g| {
            const k_base = g * GATE_RANK;
            const grp_slice = dst[out_off .. out_off + grp_bytes];
            var col_codes: [panel_cols][GATE_RANK]u8 = @splat(@splat(0));

            for (0..panel_cols) |c| {
                const col = p * panel_cols + c;
                var sb: quant.ScaleBias = .{ .scale = 0.0, .bias = 0.0 };
                if (col < n) {
                    const row_grp = w_f32[col * k + k_base ..][0..GATE_RANK];
                    var min_v: f32 = row_grp[0];
                    var max_v: f32 = row_grp[0];
                    for (row_grp[1..]) |v| {
                        if (v < min_v) min_v = v;
                        if (v > max_v) max_v = v;
                    }
                    const span = max_v - min_v;
                    const scale_f32: f32 = if (span > 1e-12) span / 15.0 else 0.0;
                    sb = .{ .scale = @floatCast(scale_f32), .bias = @floatCast(min_v) };
                    const eff_scale: f32 = @floatCast(sb.scale);
                    const eff_bias: f32 = @floatCast(sb.bias);
                    for (0..GATE_RANK) |j| {
                        if (eff_scale > 0.0) {
                            const scaled = std.math.clamp(@round((row_grp[j] - eff_bias) / eff_scale), 0.0, 15.0);
                            col_codes[c][j] = @intFromFloat(scaled);
                        }
                    }
                }
                const s_bits: u16 = @bitCast(sb.scale);
                const b_bits: u16 = @bitCast(sb.bias);
                std.mem.writeInt(u16, grp_slice[c * 4 + 0 ..][0..2], s_bits, .little);
                std.mem.writeInt(u16, grp_slice[c * 4 + 2 ..][0..2], s_bits, .little);
                std.mem.writeInt(u16, grp_slice[64 + c * 4 + 0 ..][0..2], b_bits, .little);
                std.mem.writeInt(u16, grp_slice[64 + c * 4 + 2 ..][0..2], b_bits, .little);
            }

            const code_dst = grp_slice[sme.QUANT_HEADER_BYTES..];
            for (0..32) |kp| {
                const k0 = 2 * kp;
                const k1 = k0 + 1;
                for (0..panel_cols) |c| {
                    const q0 = col_codes[c][k0] & 0x0F;
                    const q1 = col_codes[c][k1] & 0x0F;
                    code_dst[kp * panel_cols + c] = q0 | (q1 << 4);
                }
            }
            out_off += grp_bytes;
        }
    }
    if (out_off < dst.len) {
        @memset(dst[out_off..], 0);
    }
}

fn readAmfBlobAlloc(
    allocator: std.mem.Allocator,
    amf_file: *const amf.AmfFile,
    entry: *const amf.TensorEntry,
) (amf.IoError || amf.ParseError || error{OutOfMemory})![]align(4) u8 {
    const blob_len: usize = @intCast(entry.blob_bytes);
    const raw = try allocator.alignedAlloc(u8, .fromByteUnits(4), blob_len);
    errdefer allocator.free(raw);

    var done: usize = 0;
    while (done < blob_len) {
        const rc = std.c.pread(amf_file.fd, raw.ptr + done, blob_len - done, @intCast(entry.blob_offset + done));
        if (rc < 0) {
            if (std.c.errno(rc) == .INTR) continue;
            return error.ReadFailed;
        }
        if (rc == 0) return error.ShortRead;
        done += @intCast(rc);
    }
    if (std.hash.Crc32.hash(raw) != entry.blob_crc32) {
        return error.BlobChecksumMismatch;
    }
    return raw;
}

fn loadAmfWeightInput(
    allocator: std.mem.Allocator,
    amf_file: *const amf.AmfFile,
    name: []const u8,
    target_dtype: sme.PanelDType,
    rows: usize,
    cols: usize,
    buf_pool: *[NUM_BRANCHES * 6]?[]align(4) u8,
    buf_count: *usize,
) (Error || amf.ParseError || amf.IoError || error{OutOfMemory})!WeightMatrixInput {
    const entry = amf_file.parsed.find(name) orelse return error.MissingTensor;
    if (entry.rank != 2 or entry.shape[0] != rows or entry.shape[1] != cols) {
        return error.DimensionMismatch;
    }
    const raw = try readAmfBlobAlloc(allocator, amf_file, entry);
    buf_pool[buf_count.*] = raw;
    buf_count.* += 1;

    const layout = try entry.tensorLayout();
    const dt = try entry.dataType();
    if (layout == .sme2_panel64) {
        const panel_dt = try sme.panelDTypeFromAmf(dt);
        if (panel_dt != target_dtype) return error.InvalidDType;
        const need = try sme.packedBlobBytes(panel_dt, rows, cols);
        if (entry.data_bytes != need or raw.len < need) return error.InvalidWeights;
        return .{ .sme2_panel64 = raw[0..need] };
    }

    switch (dt) {
        .f32 => {
            const need = std.math.mul(usize, rows, cols) catch return error.Overflow;
            if (raw.len < need * 4) return error.InvalidWeights;
            const s: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, raw[0 .. need * 4]));
            return .{ .dense_f32 = s };
        },
        .f16, .bf16 => {
            const expected_amf: amf.DType = if (target_dtype == .bf16) .bf16 else .f16;
            if (dt != expected_amf) return error.InvalidDType;
            const need = std.math.mul(usize, rows, cols) catch return error.Overflow;
            if (raw.len < need * 2) return error.InvalidWeights;
            const s: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, raw[0 .. need * 2]));
            return .{ .dense_u16 = s };
        },
        else => return error.InvalidDType,
    }
}

fn loadOptionalAmfF32(
    allocator: std.mem.Allocator,
    amf_file: *const amf.AmfFile,
    name: []const u8,
    expected_len: usize,
    buf_pool: *[NUM_BRANCHES * 6]?[]align(4) u8,
    buf_count: *usize,
) (Error || amf.ParseError || amf.IoError || error{OutOfMemory})!?[]const f32 {
    const entry = amf_file.parsed.find(name) orelse return null;
    if (entry.rank != 1 or entry.shape[0] != expected_len) return error.DimensionMismatch;
    if ((try entry.dataType()) != .f32) return error.InvalidDType;
    const raw = try readAmfBlobAlloc(allocator, amf_file, entry);
    buf_pool[buf_count.*] = raw;
    buf_count.* += 1;
    if (raw.len < expected_len * 4) return error.InvalidWeights;
    const s: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, raw[0 .. expected_len * 4]));
    return s;
}

inline fn rmsNormRowFast(
    src: []const f32,
    norm_w: ?[]const f32,
    rms_eps: f32,
    out_dt: quant.FloatDType,
    dst: []u16,
) void {
    const dim = src.len;
    var acc0: @Vector(8, f32) = @splat(0.0);
    var i: usize = 0;
    while (i + 8 <= dim) : (i += 8) {
        const v: @Vector(8, f32) = src[i..][0..8].*;
        acc0 += v * v;
    }
    var sum_sq: f32 = @reduce(.Add, acc0);
    while (i < dim) : (i += 1) {
        const v = src[i];
        sum_sq += v * v;
    }
    const inv_rms: f32 = 1.0 / @sqrt(sum_sq / @as(f32, @floatFromInt(dim)) + rms_eps);
    if (norm_w) |nw| {
        for (0..dim) |d| {
            dst[d] = quant.f32ToFloatBits(out_dt, (src[d] * inv_rms) * nw[d]);
        }
    } else {
        for (0..dim) |d| {
            dst[d] = quant.f32ToFloatBits(out_dt, src[d] * inv_rms);
        }
    }
}

inline fn convertF32ToFloatBitsSlice(src: []const f32, out_dt: quant.FloatDType, dst: []u16) void {
    for (src, dst) |v, *out| {
        out.* = quant.f32ToFloatBits(out_dt, v);
    }
}

inline fn sigmoidF32(x: f32) f32 {
    if (x >= 0.0) {
        return 1.0 / (1.0 + @exp(-x));
    }
    const e = @exp(x);
    return e / (1.0 + e);
}

inline fn siluF32(x: f32) f32 {
    return x * sigmoidF32(x);
}

fn applyGatedStateUpdate(
    forget_raw: []const f32,
    gain_raw: []const f32,
    forget_bias: ?[]const f32,
    gain_bias: ?[]const f32,
    gain_act: GainActivation,
    s_prev: []const f32,
    value_x: []const f32,
    s_next: []f32,
    m: usize,
    sd: usize,
) void {
    for (0..m) |row| {
        const base = row * sd;
        var d: usize = 0;
        while (d + 4 <= sd) : (d += 4) {
            const idx = base + d;
            var f_vec: @Vector(4, f32) = forget_raw[idx..][0..4].*;
            var g_vec: @Vector(4, f32) = gain_raw[idx..][0..4].*;
            if (forget_bias) |fb| {
                const fb_v: @Vector(4, f32) = fb[d..][0..4].*;
                f_vec += fb_v;
            }
            if (gain_bias) |gb| {
                const gb_v: @Vector(4, f32) = gb[d..][0..4].*;
                g_vec += gb_v;
            }
            const f_gate = @as(@Vector(4, f32), @splat(1.0)) / (@as(@Vector(4, f32), @splat(1.0)) + @exp(-f_vec));
            const g_gate: @Vector(4, f32) = switch (gain_act) {
                .tanh => blk: {
                    const zeros: @Vector(4, f32) = @splat(0.0);
                    const ones: @Vector(4, f32) = @splat(1.0);
                    const abs2 = @min(@abs(g_vec) * @as(@Vector(4, f32), @splat(2.0)), @as(@Vector(4, f32), @splat(80.0)));
                    const e = @exp(-abs2);
                    const mag = (ones - e) / (ones + e);
                    break :blk @select(f32, g_vec < zeros, -mag, mag);
                },
                .silu => g_vec / (@as(@Vector(4, f32), @splat(1.0)) + @exp(-g_vec)),
            };
            const sp: @Vector(4, f32) = s_prev[idx..][0..4].*;
            const vx: @Vector(4, f32) = value_x[idx..][0..4].*;
            const res = f_gate * sp + g_gate * vx;
            s_next[idx..][0..4].* = res;
        }
        while (d < sd) : (d += 1) {
            const idx = base + d;
            const fb_val = if (forget_bias) |fb| fb[d] else 0.0;
            const gb_val = if (gain_bias) |gb| gb[d] else 0.0;
            const f_gate = sigmoidF32(forget_raw[idx] + fb_val);
            const g_in = gain_raw[idx] + gb_val;
            const g_gate = switch (gain_act) {
                .tanh => std.math.tanh(g_in),
                .silu => siluF32(g_in),
            };
            s_next[idx] = f_gate * s_prev[idx] + g_gate * value_x[idx];
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

fn testBudget(cap_bytes: usize) budget_mod.Budget {
    return budget_mod.Budget.compute(.{
        .memory_bytes = cap_bytes * 8,
        .user_wire_limit = cap_bytes * 2,
        .global_user_wire_limit = cap_bytes * 2,
        .gpu_working_set_bytes = cap_bytes * 2,
        .configured_cap = null,
        .headroom_bytes = page.APPLE_PAGE_SIZE,
    });
}

fn fillDeterministicMatrix(dst: []f32, seed: u64, scale: f32) void {
    var s = seed;
    for (dst) |*v| {
        s = s *% 6364136223846793005 +% 1442695040888963407;
        const raw_i: i32 = @as(i32, @intCast((s >> 33) % 2001)) - 1000;
        v.* = @as(f32, @floatFromInt(raw_i)) * 0.001 * scale;
    }
}

fn quantizeWeightsToFloatPrecision(w: []f32, dt: sme.PanelDType) void {
    const fdt: quant.FloatDType = if (dt == .bf16) .bf16 else .f16;
    for (w) |*v| {
        v.* = quant.floatBitsToF32(fdt, quant.f32ToFloatBits(fdt, v.*));
    }
}

test "C26 Inkling2LoopController enforces 16 KiB WiredRegion, L2 residency, BF16/F16 FP32 parity (<= 2e-3), and zero Metal dispatches" {
    const gpa = std.testing.allocator;
    const h: usize = 128;
    const sd: usize = 128;
    const r: usize = GATE_RANK;
    const max_m: usize = 8;

    const dtypes = [_]sme.PanelDType{ .bf16, .f16 };
    const acts = [_]GainActivation{ .tanh, .silu };

    var disp = try sme.SmeDispatcher.init(gpa, .{ .total_threads = 2 });
    defer disp.deinit();

    for (dtypes) |dt| {
        for (acts) |act| {
            var cfg = ControllerConfig.uniform(@intCast(h), @intCast(sd), dt, act, @intCast(max_m));
            cfg.budget_override = testBudget(4 * 1024 * 1024);

            const down_w = try gpa.alloc(f32, r * h);
            defer gpa.free(down_w);
            const forget_w = try gpa.alloc(f32, sd * r);
            defer gpa.free(forget_w);
            const gain_w = try gpa.alloc(f32, sd * r);
            defer gpa.free(gain_w);
            const norm_w = try gpa.alloc(f32, h);
            defer gpa.free(norm_w);
            const forget_b = try gpa.alloc(f32, sd);
            defer gpa.free(forget_b);
            const gain_b = try gpa.alloc(f32, sd);
            defer gpa.free(gain_b);

            fillDeterministicMatrix(down_w, 101 + @as(u64, @backingInt(dt)), 1.0 / @sqrt(@as(f32, @floatFromInt(h))));
            fillDeterministicMatrix(forget_w, 202 + @as(u64, @backingInt(act)), 1.0 / @sqrt(@as(f32, @floatFromInt(r))));
            fillDeterministicMatrix(gain_w, 303 + @as(u64, @backingInt(dt)), 1.0 / @sqrt(@as(f32, @floatFromInt(r))));
            for (0..h) |i| norm_w[i] = 0.8 + 0.003 * @as(f32, @floatFromInt(i % 64));
            for (0..sd) |i| {
                forget_b[i] = -0.2 + 0.01 * @as(f32, @floatFromInt(i % 32));
                gain_b[i] = 0.1 - 0.01 * @as(f32, @floatFromInt(i % 24));
            }

            const branch_in: BranchWeightInput = .{
                .down_weight = .{ .dense_f32 = down_w },
                .forget_weight = .{ .dense_f32 = forget_w },
                .gain_weight = .{ .dense_f32 = gain_w },
                .norm_weight = norm_w,
                .forget_bias = forget_b,
                .gain_bias = gain_b,
            };

            var ctrl = try Inkling2LoopController.init(cfg, .{ branch_in, branch_in, branch_in, branch_in });
            defer ctrl.deinit();

            try std.testing.expectEqual(@as(usize, 0), ctrl.totalWiredBytes() % page.APPLE_PAGE_SIZE);
            try std.testing.expect(std.mem.isAligned(@intFromPtr(ctrl.arena.bytes().ptr), page.APPLE_PAGE_SIZE));
            try std.testing.expect(ctrl.fitsL2(null));
            try std.testing.expectEqual(@as(u64, 0), ctrl.metal_dispatches);

            for ([_]usize{ 1, 8 }) |m| {
                const h_in = try gpa.alloc(f32, m * h);
                defer gpa.free(h_in);
                const s_prev = try gpa.alloc(f32, m * sd);
                defer gpa.free(s_prev);
                const s_got = try gpa.alloc(f32, m * sd);
                defer gpa.free(s_got);
                const s_ref = try gpa.alloc(f32, m * sd);
                defer gpa.free(s_ref);

                fillDeterministicMatrix(h_in, 777 + m, 0.5);
                fillDeterministicMatrix(s_prev, 888 + m, 0.5);

                // Run SME2 (or native host backend) and verify zero Metal dispatches.
                try ctrl.stepBranch(&disp, .attn, h_in, s_prev, s_got, m, null);
                try std.testing.expectEqual(@as(u64, 0), ctrl.metal_dispatches);

                // Also verify M1-M3 scalar fallback parity.
                const s_fb = try gpa.alloc(f32, m * sd);
                defer gpa.free(s_fb);
                try ctrl.stepBranch(&disp, .attn, h_in, s_prev, s_fb, m, .scalar_fallback);

                try stepBranchFp32Reference(
                    h_in,
                    h_in,
                    s_prev,
                    down_w,
                    forget_w,
                    gain_w,
                    norm_w,
                    forget_b,
                    gain_b,
                    act,
                    cfg.rms_eps,
                    m,
                    h,
                    sd,
                    s_ref,
                );

                var max_err: f32 = 0.0;
                var max_fb_err: f32 = 0.0;
                for (0..m * sd) |idx| {
                    max_err = @max(max_err, @abs(s_got[idx] - s_ref[idx]));
                    max_fb_err = @max(max_fb_err, @abs(s_got[idx] - s_fb[idx]));
                }
                try std.testing.expect(max_err <= 2e-3);
                try std.testing.expect(max_fb_err <= 1e-4);
            }
        }
    }
}

test "C26 Inkling2LoopController Q4_affine packing, 4-branch hoisted execution, in-place aliasing, and L2 limit rejection" {
    const gpa = std.testing.allocator;
    const h: usize = 128;
    const sd: usize = 64;
    const r: usize = GATE_RANK;

    var disp = try sme.SmeDispatcher.init(gpa, .{ .total_threads = 2 });
    defer disp.deinit();

    var cfg = ControllerConfig.uniform(@intCast(h), @intCast(sd), .q4_affine, .silu, 4);
    cfg.budget_override = testBudget(4 * 1024 * 1024);

    const down_f32 = try gpa.alloc(f32, r * h);
    defer gpa.free(down_f32);
    const forget_f32 = try gpa.alloc(f32, sd * r);
    defer gpa.free(forget_f32);
    const gain_f32 = try gpa.alloc(f32, sd * r);
    defer gpa.free(gain_f32);

    fillDeterministicMatrix(down_f32, 41, 0.25);
    fillDeterministicMatrix(forget_f32, 42, 0.25);
    fillDeterministicMatrix(gain_f32, 43, 0.25);

    const down_codes = try gpa.alloc(u8, r * h);
    defer gpa.free(down_codes);
    const down_sb = try gpa.alloc(quant.ScaleBias, r * (h / GATE_RANK));
    defer gpa.free(down_sb);
    try quantizeF32ToQ4Affine(down_f32, r, h, down_codes, down_sb);

    const down_deq = try gpa.alloc(f32, r * h);
    defer gpa.free(down_deq);
    try dequantizeQ4AffineToF32(down_codes, down_sb, r, h, down_deq);

    const fg_codes = try gpa.alloc(u8, sd * r);
    defer gpa.free(fg_codes);
    const fg_sb = try gpa.alloc(quant.ScaleBias, sd);
    defer gpa.free(fg_sb);
    try quantizeF32ToQ4Affine(forget_f32, sd, r, fg_codes, fg_sb);
    const forget_deq = try gpa.alloc(f32, sd * r);
    defer gpa.free(forget_deq);
    try dequantizeQ4AffineToF32(fg_codes, fg_sb, sd, r, forget_deq);

    const gg_codes = try gpa.alloc(u8, sd * r);
    defer gpa.free(gg_codes);
    const gg_sb = try gpa.alloc(quant.ScaleBias, sd);
    defer gpa.free(gg_sb);
    try quantizeF32ToQ4Affine(gain_f32, sd, r, gg_codes, gg_sb);
    const gain_deq = try gpa.alloc(f32, sd * r);
    defer gpa.free(gain_deq);
    try dequantizeQ4AffineToF32(gg_codes, gg_sb, sd, r, gain_deq);

    const branch_in: BranchWeightInput = .{
        .down_weight = .{ .q4_affine = .{ .codes_nk = down_codes, .scale_bias = down_sb } },
        .forget_weight = .{ .q4_affine = .{ .codes_nk = fg_codes, .scale_bias = fg_sb } },
        .gain_weight = .{ .q4_affine = .{ .codes_nk = gg_codes, .scale_bias = gg_sb } },
    };

    var ctrl = try Inkling2LoopController.init(cfg, .{ branch_in, branch_in, branch_in, branch_in });
    defer ctrl.deinit();

    const m: usize = 3;
    var h_in: [m * h]f32 = undefined;
    var val_x: [m * sd]f32 = undefined;
    var s_state: [m * sd]f32 = undefined;
    var s_orig: [m * sd]f32 = undefined;
    var s_ref: [m * sd]f32 = undefined;
    fillDeterministicMatrix(&h_in, 901, 0.5);
    fillDeterministicMatrix(&val_x, 902, 0.5);
    fillDeterministicMatrix(&s_orig, 903, 0.5);
    @memcpy(&s_state, &s_orig);

    // In-place update (s_prev == s_next) with distinct `state_dim < hidden_size`!
    try ctrl.stepBranchWithValue(&disp, .k, &h_in, &val_x, &s_state, &s_state, m, null);
    try stepBranchFp32Reference(
        &h_in,
        &val_x,
        &s_orig,
        down_deq,
        forget_deq,
        gain_deq,
        null,
        null,
        null,
        .silu,
        cfg.rms_eps,
        m,
        h,
        sd,
        &s_ref,
    );

    var max_err: f32 = 0.0;
    for (0..m * sd) |idx| {
        max_err = @max(max_err, @abs(s_state[idx] - s_ref[idx]));
    }
    try std.testing.expect(max_err <= 2e-3);

    // Enforcing tight L2 ceiling rejects oversized controller cleanly.
    var tight_cfg = cfg;
    tight_cfg.l2_limit_bytes = 1024;
    try std.testing.expectError(error.ExceedsL2Budget, Inkling2LoopController.requiredBytes(tight_cfg));
}

test "C26 Inkling-2-Mini (hidden=2048, gate_rank=64) L2 residency, parity, and us/pass latency" {
    const gpa = std.testing.allocator;
    const h: usize = 2048;
    const sd: usize = 2048;
    const r: usize = GATE_RANK;

    var disp = try sme.SmeDispatcher.init(gpa, .{});
    defer disp.deinit();

    var cfg = ControllerConfig.uniform(@intCast(h), @intCast(sd), .bf16, .tanh, 8);
    cfg.budget_override = testBudget(8 * 1024 * 1024);

    const down_w = try gpa.alloc(f32, r * h);
    defer gpa.free(down_w);
    const forget_w = try gpa.alloc(f32, sd * r);
    defer gpa.free(forget_w);
    const gain_w = try gpa.alloc(f32, sd * r);
    defer gpa.free(gain_w);

    fillDeterministicMatrix(down_w, 501, 1.0 / @sqrt(@as(f32, @floatFromInt(h))));
    fillDeterministicMatrix(forget_w, 502, 1.0 / @sqrt(@as(f32, @floatFromInt(r))));
    fillDeterministicMatrix(gain_w, 503, 1.0 / @sqrt(@as(f32, @floatFromInt(r))));

    const branch_in: BranchWeightInput = .{
        .down_weight = .{ .dense_f32 = down_w },
        .forget_weight = .{ .dense_f32 = forget_w },
        .gain_weight = .{ .dense_f32 = gain_w },
    };

    var ctrl = try Inkling2LoopController.init(cfg, .{ branch_in, branch_in, branch_in, branch_in });
    defer ctrl.deinit();

    // 4 branches * 3 * (64 * 2048 * 2 B) = 3.0 MiB << 16 MiB P-cluster L2.
    try std.testing.expect(ctrl.totalWeightBytes() == 3 * 1024 * 1024);
    try std.testing.expect(ctrl.fitsL2(null));

    const h_in = try gpa.alloc(f32, h);
    defer gpa.free(h_in);
    const s_prev = try gpa.alloc(f32, sd);
    defer gpa.free(s_prev);
    const s_got = try gpa.alloc(f32, sd);
    defer gpa.free(s_got);
    const s_ref = try gpa.alloc(f32, sd);
    defer gpa.free(s_ref);

    fillDeterministicMatrix(h_in, 601, 0.5);
    fillDeterministicMatrix(s_prev, 602, 0.5);

    // Quantize reference weights to exact BF16 representation to isolate kernel error vs BF16 rounding.
    quantizeWeightsToFloatPrecision(down_w, .bf16);
    quantizeWeightsToFloatPrecision(forget_w, .bf16);
    quantizeWeightsToFloatPrecision(gain_w, .bf16);

    try ctrl.stepBranch(&disp, .mlp, h_in, s_prev, s_got, 1, null);
    try stepBranchFp32Reference(
        h_in,
        h_in,
        s_prev,
        down_w,
        forget_w,
        gain_w,
        null,
        null,
        null,
        .tanh,
        cfg.rms_eps,
        1,
        h,
        sd,
        s_ref,
    );

    var max_err: f32 = 0.0;
    for (0..sd) |idx| {
        max_err = @max(max_err, @abs(s_got[idx] - s_ref[idx]));
    }
    try std.testing.expect(max_err <= 2e-3);

    const rep_m1 = try benchmarkControllerPass(&ctrl, &disp, 1, 25, null);
    const rep_m8 = try benchmarkControllerPass(&ctrl, &disp, 8, 25, null);
    try std.testing.expect(rep_m1.us_per_full_pass_4_branches > 0.0);
    try std.testing.expect(rep_m8.us_per_full_pass_4_branches > 0.0);
    try std.testing.expectEqual(@as(u64, 0), ctrl.metal_dispatches);
}

test "C26 Inkling2LoopController loadFromAmfLayer loads sme2_panel64 blobs from F_NOCACHE AMF and matches FP32" {
    const gpa = std.testing.allocator;
    const h: usize = 64;
    const sd: usize = 64;
    const r: usize = GATE_RANK;

    var disp = try sme.SmeDispatcher.init(gpa, .{ .total_threads = 2 });
    defer disp.deinit();

    try std.testing.expectEqual(ControllerBranch.attn, try ControllerBranch.fromLabel("attn"));
    try std.testing.expectEqual(ControllerBranch.mlp, try ControllerBranch.fromLabel("mlp"));

    var cfg = ControllerConfig.uniform(@intCast(h), @intCast(sd), .bf16, .tanh, 4);
    cfg.budget_override = testBudget(4 * 1024 * 1024);

    var down_f32: [r * h]f32 = undefined;
    var forget_f32: [sd * r]f32 = undefined;
    var gain_f32: [sd * r]f32 = undefined;
    fillDeterministicMatrix(&down_f32, 31, 0.125);
    fillDeterministicMatrix(&forget_f32, 32, 0.125);
    fillDeterministicMatrix(&gain_f32, 33, 0.125);

    const packed_down_bytes = try sme.packedBlobBytes(.bf16, r, h);
    const packed_fg_bytes = try sme.packedBlobBytes(.bf16, sd, r);
    const packed_down = try gpa.alloc(u8, packed_down_bytes);
    defer gpa.free(packed_down);
    const packed_forget = try gpa.alloc(u8, packed_fg_bytes);
    defer gpa.free(packed_forget);
    const packed_gain = try gpa.alloc(u8, packed_fg_bytes);
    defer gpa.free(packed_gain);

    try sme.packDenseFromF32Panel64(.bf16, &down_f32, r, h, packed_down);
    try sme.packDenseFromF32Panel64(.bf16, &forget_f32, sd, r, packed_forget);
    try sme.packDenseFromF32Panel64(.bf16, &gain_f32, sd, r, packed_gain);

    // Build an F_NOCACHE .amf image with all 4 branches (12 sme2_panel64 tensors) for layer 6.
    const count: usize = NUM_BRANCHES * 3;
    const dir_offset: usize = page.APPLE_PAGE_SIZE;
    const dir_raw = count * @sizeOf(amf.TensorEntry);
    const dir_bytes = try page.alignUp(dir_raw);

    var entries: [count]amf.TensorEntry = undefined;
    var blob_slices: [count][]const u8 = undefined;
    var nbuf: [48]u8 = undefined;
    var next_off: usize = dir_offset + dir_bytes;

    for (0..NUM_BRANCHES) |bi| {
        const b_name = (@as(ControllerBranch, @fromBackingInt(@intCast(bi)))).label();
        const names = [_][]const u8{ "down_weight", "forget_weight", "gain_weight" };
        const datas = [_][]const u8{ packed_down, packed_forget, packed_gain };
        const shapes = [_][2]u32{ .{ r, h }, .{ sd, r }, .{ sd, r } };
        for (0..3) |mi| {
            const idx = bi * 3 + mi;
            const tname = try std.fmt.bufPrint(&nbuf, "layers.6.controllers.{s}.{s}", .{ b_name, names[mi] });
            var e = amf.TensorEntry.zeroed(tname, .dense, .bf16);
            e.rank = 2;
            e.layer = 6;
            e.shape = .{ shapes[mi][0], shapes[mi][1], 0, 0 };
            e.data_bytes = datas[mi].len;
            e.blob_offset = next_off;
            e.blob_bytes = try page.alignUp(datas[mi].len);
            e.setLayoutAndParamKind(.sme2_panel64, .affine_scale_bias);
            entries[idx] = e;
            blob_slices[idx] = datas[mi];
            next_off += @intCast(e.blob_bytes);
        }
    }

    const image = try gpa.alloc(u8, next_off);
    defer gpa.free(image);
    @memset(image, 0);
    for (0..count) |i| {
        const boff: usize = @intCast(entries[i].blob_offset);
        const blen: usize = @intCast(entries[i].blob_bytes);
        @memcpy(image[boff..][0..blob_slices[i].len], blob_slices[i]);
        entries[i].blob_crc32 = std.hash.Crc32.hash(image[boff..][0..blen]);
        @memcpy(image[dir_offset + i * @sizeOf(amf.TensorEntry) ..][0..@sizeOf(amf.TensorEntry)], std.mem.asBytes(&entries[i]));
    }
    var hdr: amf.Header = .{
        .magic = amf.MAGIC,
        .version = amf.FORMAT_VERSION,
        .flags = 0,
        .page_size = page.APPLE_PAGE_SIZE,
        .checksum_algo = amf.CHECKSUM_CRC32_IEEE,
        .tensor_count = @intCast(count),
        .reserved0 = 0,
        .directory_offset = dir_offset,
        .directory_bytes = dir_bytes,
        .directory_crc32 = std.hash.Crc32.hash(image[dir_offset..][0..dir_raw]),
        .header_crc32 = 0,
        .model_id = @splat(0),
        .reserved1 = 0,
    };
    const mname = "inkling2-ctrl-c26";
    @memcpy(hdr.model_id[0..mname.len], mname);
    hdr.header_crc32 = amf.computeHeaderCrc32(&hdr);
    @memcpy(image[0..@sizeOf(amf.Header)], std.mem.asBytes(&hdr));

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "ctrl-l6.amf", .data = image });
    var p_buf: [std.fs.max_path_bytes]u8 = undefined;
    const p_raw = try std.fmt.bufPrint(p_buf[0 .. p_buf.len - 1], ".zig-cache/tmp/{s}/ctrl-l6.amf", .{tmp.sub_path});
    p_buf[p_raw.len] = 0;

    var af = try amf.AmfFile.open(gpa, p_buf[0..p_raw.len :0]);
    defer af.close(gpa);

    var ctrl = try Inkling2LoopController.loadFromAmfLayer(gpa, &af, 6, cfg);
    defer ctrl.deinit();

    var h_in: [h]f32 = undefined;
    var s_prev: [sd]f32 = undefined;
    var s_got: [sd]f32 = undefined;
    var s_ref: [sd]f32 = undefined;
    fillDeterministicMatrix(&h_in, 77, 0.5);
    fillDeterministicMatrix(&s_prev, 88, 0.5);

    try ctrl.stepBranch(&disp, .v, &h_in, &s_prev, &s_got, 1, null);
    try stepBranchFp32Reference(&h_in, &h_in, &s_prev, &down_f32, &forget_f32, &gain_f32, null, null, null, .tanh, cfg.rms_eps, 1, h, sd, &s_ref);
    for (0..sd) |d| {
        try std.testing.expectApproxEqAbs(s_ref[d], s_got[d], 2e-3);
    }
}

test "F11: dense_f32 -> q4_affine packF32AsQ4Panel64NoHeap matches 640B canonical layout and FP32 parity" {
    const gpa = std.testing.allocator;

    // 1. Direct binary layout & zero-tail check on non-multiple-of-16 `n` (`n = 21, k = 128`)
    //    into a poisoned buffer (`0xAA`) so any unwritten header/padding byte fails immediately.
    {
        const n_odd: usize = 21;
        const k_odd: usize = 128;
        const groups = k_odd / GATE_RANK;
        const w_f32 = try gpa.alloc(f32, n_odd * k_odd);
        defer gpa.free(w_f32);
        fillDeterministicMatrix(w_f32, 313, 0.35);

        const blob_bytes = try sme.packedBlobBytes(.q4_affine, n_odd, k_odd);
        try std.testing.expectEqual(sme.panelCount(n_odd) * groups * 640, blob_bytes);

        const got_blob = try gpa.alloc(u8, blob_bytes + 64);
        defer gpa.free(got_blob);
        @memset(got_blob, 0xAA);
        try packF32AsQ4Panel64NoHeap(w_f32, n_odd, k_odd, groups, got_blob);

        // Verify tail bytes beyond packed payload are strictly zeroed.
        for (got_blob[blob_bytes..]) |b| {
            try std.testing.expectEqual(@as(u8, 0), b);
        }

        // Compare byte-for-byte against canonical `quantizeF32ToQ4Affine + sme.packQuantAffinePanel64`.
        const ref_codes = try gpa.alloc(u8, n_odd * k_odd);
        defer gpa.free(ref_codes);
        const ref_sb = try gpa.alloc(quant.ScaleBias, n_odd * groups);
        defer gpa.free(ref_sb);
        try quantizeF32ToQ4Affine(w_f32, n_odd, k_odd, ref_codes, ref_sb);

        const ref_blob = try gpa.alloc(u8, blob_bytes);
        defer gpa.free(ref_blob);
        @memset(ref_blob, 0x55);
        try sme.packQuantAffinePanel64(.q4_affine, ref_codes, ref_sb, n_odd, k_odd, ref_blob);
        try std.testing.expectEqualSlices(u8, ref_blob, got_blob[0..blob_bytes]);
    }

    // 2. End-to-end `Inkling2LoopController` initialized from `.dense_f32` -> `.q4_affine`
    //    verifying numerical parity against FP32 reference across native SME2 and scalar fallback.
    {
        const h: usize = 128;
        const sd: usize = 64;
        const r: usize = GATE_RANK;

        var disp = try sme.SmeDispatcher.init(gpa, .{ .total_threads = 2 });
        defer disp.deinit();

        var cfg = ControllerConfig.uniform(@intCast(h), @intCast(sd), .q4_affine, .silu, 4);
        cfg.budget_override = testBudget(4 * 1024 * 1024);

        const down_f32 = try gpa.alloc(f32, r * h);
        defer gpa.free(down_f32);
        const forget_f32 = try gpa.alloc(f32, sd * r);
        defer gpa.free(forget_f32);
        const gain_f32 = try gpa.alloc(f32, sd * r);
        defer gpa.free(gain_f32);
        fillDeterministicMatrix(down_f32, 611, 0.25);
        fillDeterministicMatrix(forget_f32, 612, 0.25);
        fillDeterministicMatrix(gain_f32, 613, 0.25);

        const branch_in: BranchWeightInput = .{
            .down_weight = .{ .dense_f32 = down_f32 },
            .forget_weight = .{ .dense_f32 = forget_f32 },
            .gain_weight = .{ .dense_f32 = gain_f32 },
        };

        var ctrl = try Inkling2LoopController.init(cfg, .{ branch_in, branch_in, branch_in, branch_in });
        defer ctrl.deinit();

        // Build dequantized Q4 reference weights to check exact SME2 vs reference parity,
        // plus raw FP32 reference to bound quantization error.
        const down_codes = try gpa.alloc(u8, r * h);
        defer gpa.free(down_codes);
        const down_sb = try gpa.alloc(quant.ScaleBias, r * (h / GATE_RANK));
        defer gpa.free(down_sb);
        try quantizeF32ToQ4Affine(down_f32, r, h, down_codes, down_sb);
        const down_deq = try gpa.alloc(f32, r * h);
        defer gpa.free(down_deq);
        try dequantizeQ4AffineToF32(down_codes, down_sb, r, h, down_deq);

        const fg_codes = try gpa.alloc(u8, sd * r);
        defer gpa.free(fg_codes);
        const fg_sb = try gpa.alloc(quant.ScaleBias, sd);
        defer gpa.free(fg_sb);
        try quantizeF32ToQ4Affine(forget_f32, sd, r, fg_codes, fg_sb);
        const forget_deq = try gpa.alloc(f32, sd * r);
        defer gpa.free(forget_deq);
        try dequantizeQ4AffineToF32(fg_codes, fg_sb, sd, r, forget_deq);

        const gg_codes = try gpa.alloc(u8, sd * r);
        defer gpa.free(gg_codes);
        const gg_sb = try gpa.alloc(quant.ScaleBias, sd);
        defer gpa.free(gg_sb);
        try quantizeF32ToQ4Affine(gain_f32, sd, r, gg_codes, gg_sb);
        const gain_deq = try gpa.alloc(f32, sd * r);
        defer gpa.free(gain_deq);
        try dequantizeQ4AffineToF32(gg_codes, gg_sb, sd, r, gain_deq);

        const m: usize = 3;
        var h_in: [m * h]f32 = undefined;
        var val_x: [m * sd]f32 = undefined;
        var s_prev: [m * sd]f32 = undefined;
        var s_got: [m * sd]f32 = undefined;
        var s_fb: [m * sd]f32 = undefined;
        var s_qref: [m * sd]f32 = undefined;
        var s_fref: [m * sd]f32 = undefined;
        fillDeterministicMatrix(&h_in, 701, 0.5);
        fillDeterministicMatrix(&val_x, 702, 0.5);
        fillDeterministicMatrix(&s_prev, 703, 0.5);

        try ctrl.stepBranchWithValue(&disp, .attn, &h_in, &val_x, &s_prev, &s_got, m, null);
        try ctrl.stepBranchWithValue(&disp, .attn, &h_in, &val_x, &s_prev, &s_fb, m, .scalar_fallback);
        try stepBranchFp32Reference(&h_in, &val_x, &s_prev, down_deq, forget_deq, gain_deq, null, null, null, .silu, cfg.rms_eps, m, h, sd, &s_qref);
        try stepBranchFp32Reference(&h_in, &val_x, &s_prev, down_f32, forget_f32, gain_f32, null, null, null, .silu, cfg.rms_eps, m, h, sd, &s_fref);

        for (0..m * sd) |idx| {
            try std.testing.expect(std.math.isFinite(s_got[idx]));
            try std.testing.expectApproxEqAbs(s_qref[idx], s_got[idx], 2e-3);
            try std.testing.expectApproxEqAbs(s_fb[idx], s_got[idx], 1e-4);
            try std.testing.expectApproxEqAbs(s_fref[idx], s_got[idx], 0.15);
        }
    }
}
