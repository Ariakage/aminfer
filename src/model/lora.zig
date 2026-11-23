//! Runtime multi-LoRA hot-loading, preallocated wired L2 pool, model-selector
//! routing (`"<base>:<adapter>"`), and zero-allocation rank-`r` projection delta
//! evaluation (`W19`, `ADR-0022`).
//!
//! Design invariants:
//! - Hard wired pool ceiling `MAX_LORA_POOL_BYTES = 4 MiB` (`25%` of M4's
//!   `16 MiB` P-core L2), accounted inside the `R9` wired budget at startup and
//!   never unpinning base model weights.
//! - Preallocated 16 KiB-page wired pool (`page.WiredRegion`) with bitmap page
//!   reclamation so adapters hot-load and hot-unload in arbitrary order without
//!   OS `mmap`/`mlock` churn or fragmentation leaks.
//! - Strict zero-overhead bypass outside the transformer layer loop when
//!   `activeAdapterCount() == 0` (`0` instructions added to base decode/prefill).
//! - Supports both `sme2_panel64` (accelerating via `W16` L2-resident
//!   `sme.SmeStepLoraAdapter` when `K % 64 == 0`) and `row_major` (`f16`/`bf16`)
//!   `.amlora` (`.amf`-based) bundles across single-request and multi-slot `C14`
//!   heterogeneous batches.

const std = @import("std");
const page = @import("../mem/page.zig");
const budget_mod = @import("../mem/budget.zig");
const amf = @import("../format/amf.zig");
const quant = @import("../quant/spec.zig");
const sme = @import("../cpu/sme.zig");

/// Hard cap on P-core L2-resident LoRA wired pool (`4 MiB`, ADR-0022).
pub const MAX_LORA_POOL_BYTES: usize = 4 * 1024 * 1024;
pub const MAX_POOL_PAGES: usize = MAX_LORA_POOL_BYTES / page.APPLE_PAGE_SIZE;

/// Maximum simultaneously hot-loaded LoRA adapters in `LoraRegistry`.
pub const MAX_ADAPTERS: usize = 16;
pub const MAX_ADAPTER_NAME_LEN: usize = 64;
pub const MAX_LORA_RANK: u32 = sme.MAX_STEP_LORA_RANK; // 64
pub const MAX_LORA_BATCH_ROWS: usize = sme.MAX_STEP_LORA_M; // 64

/// Sentinel indicating no active LoRA adapter (base model only).
pub const NO_ADAPTER: u32 = std.math.maxInt(u32);

pub const TargetProjection = enum(u8) {
    q_proj = 0,
    k_proj = 1,
    v_proj = 2,
    o_proj = 3,
    gate_proj = 4,
    up_proj = 5,
    down_proj = 6,

    pub const COUNT: usize = 7;

    pub fn label(self: TargetProjection) []const u8 {
        return switch (self) {
            .q_proj => "q_proj",
            .k_proj => "k_proj",
            .v_proj => "v_proj",
            .o_proj => "o_proj",
            .gate_proj => "gate_proj",
            .up_proj => "up_proj",
            .down_proj => "down_proj",
        };
    }

    pub fn suffix(self: TargetProjection) []const u8 {
        return switch (self) {
            .q_proj => "self_attn.q_proj",
            .k_proj => "self_attn.k_proj",
            .v_proj => "self_attn.v_proj",
            .o_proj => "self_attn.o_proj",
            .gate_proj => "mlp.gate_proj",
            .up_proj => "mlp.up_proj",
            .down_proj => "mlp.down_proj",
        };
    }

    pub fn bit(self: TargetProjection) u8 {
        return @as(u8, 1) << @intCast(@backingInt(self));
    }

    pub fn parseModuleName(name: []const u8) ?TargetProjection {
        if (std.mem.eql(u8, name, "q_proj")) return .q_proj;
        if (std.mem.eql(u8, name, "k_proj")) return .k_proj;
        if (std.mem.eql(u8, name, "v_proj")) return .v_proj;
        if (std.mem.eql(u8, name, "o_proj")) return .o_proj;
        if (std.mem.eql(u8, name, "gate_proj")) return .gate_proj;
        if (std.mem.eql(u8, name, "up_proj")) return .up_proj;
        if (std.mem.eql(u8, name, "down_proj")) return .down_proj;
        return null;
    }
};

pub const ModelSelector = struct {
    base: []const u8,
    adapter: ?[]const u8,

    pub fn hasAdapter(self: ModelSelector) bool {
        return self.adapter != null;
    }
};

/// Parses OpenAI/Anthropic `"<base>:<adapter>"` model selectors (`W19`).
/// Plain `"<base>"` returns `.adapter = null`. Splits on the last `:` so base
/// identifiers containing colons (e.g. `"org/model:q4:sql-lora"`) still route
/// cleanly when an adapter suffix is present, while trailing `":"` yields `null`.
pub fn parseModelSelector(raw: []const u8) ModelSelector {
    const trimmed = std.mem.trim(u8, raw, " \t\r\n");
    if (std.mem.lastIndexOfScalar(u8, trimmed, ':')) |colon| {
        const base = trimmed[0..colon];
        const adp = trimmed[colon + 1 ..];
        if (base.len > 0 and adp.len > 0) {
            return .{ .base = base, .adapter = adp };
        }
        return .{ .base = if (base.len > 0) base else trimmed, .adapter = null };
    }
    return .{ .base = trimmed, .adapter = null };
}

pub const LoraError = amf.ParseError || amf.IoError || page.WiredRegion.InitError || budget_mod.PlanError || sme.Error || error{
    InvalidConfig,
    PoolCapExceeded,
    PoolOutOfSpace,
    RegistryFull,
    DuplicateAdapterName,
    AdapterNotFound,
    InvalidAdapterName,
    InvalidAdapterConfig,
    UnsupportedRank,
    UnsupportedDType,
    MissingPairedTensor,
    DimensionMismatch,
    LayerOutOfRange,
    InvalidTargetProjection,
    BufferTooSmall,
    OutOfMemory,
};

pub const AdapterConfigMeta = struct {
    rank: u32,
    lora_alpha: f32,
    scale: f32,
    target_mask: u8,
};

/// Parses PEFT `adapter_config.json` (either embedded in `__meta.adapter_config_json`
/// or sidecar) extracting `r`, `lora_alpha`, derived `scale = lora_alpha / r`,
/// and `target_modules` bitmask.
pub fn parseAdapterConfigJson(allocator: std.mem.Allocator, json_bytes: []const u8) LoraError!AdapterConfigMeta {
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, json_bytes, .{
        .ignore_unknown_fields = true,
    }) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.InvalidAdapterConfig,
    };
    defer parsed.deinit();
    if (parsed.value != .object) return error.InvalidAdapterConfig;
    const obj = parsed.value.object;

    const r_val = obj.get("r") orelse return error.InvalidAdapterConfig;
    const rank: u32 = switch (r_val) {
        .integer => |v| if (v >= 1 and v <= MAX_LORA_RANK) @intCast(v) else return error.UnsupportedRank,
        else => return error.InvalidAdapterConfig,
    };

    const alpha_val = obj.get("lora_alpha");
    const lora_alpha: f32 = if (alpha_val) |av| switch (av) {
        .float => |f| @floatCast(f),
        .integer => |v| @floatFromInt(v),
        else => return error.InvalidAdapterConfig,
    } else @floatFromInt(rank);
    if (!std.math.isFinite(lora_alpha)) return error.InvalidAdapterConfig;

    var target_mask: u8 = 0;
    if (obj.get("target_modules")) |tm_val| {
        switch (tm_val) {
            .array => |arr| {
                for (arr.items) |item| {
                    if (item == .string) {
                        if (TargetProjection.parseModuleName(item.string)) |tp| {
                            target_mask |= tp.bit();
                        }
                    }
                }
            },
            .string => |s| {
                if (TargetProjection.parseModuleName(s)) |tp| {
                    target_mask |= tp.bit();
                }
            },
            else => {},
        }
    }

    const scale: f32 = lora_alpha / @as(f32, @floatFromInt(rank));
    return .{
        .rank = rank,
        .lora_alpha = lora_alpha,
        .scale = scale,
        .target_mask = target_mask,
    };
}

/// Geometry expectations for validating a loaded `.amlora` against the active base model.
pub const BaseModelShape = struct {
    num_layers: u32,
    hidden_size: u32,
    q_dim: u32,
    kv_dim: u32,
    intermediate_size: u32,

    pub fn expectedDims(self: BaseModelShape, target: TargetProjection) struct { n: u32, k: u32 } {
        return switch (target) {
            .q_proj => .{ .n = self.q_dim, .k = self.hidden_size },
            .k_proj, .v_proj => .{ .n = self.kv_dim, .k = self.hidden_size },
            .o_proj => .{ .n = self.hidden_size, .k = self.q_dim },
            .gate_proj, .up_proj => .{ .n = self.intermediate_size, .k = self.hidden_size },
            .down_proj => .{ .n = self.hidden_size, .k = self.intermediate_size },
        };
    }
};

pub const LoraLinearPair = struct {
    rank: u32,
    n: u32,
    k: u32,
    scale: f32,
    layout: amf.TensorLayout,
    dtype: quant.FloatDType,
    a_blob: []const u8,
    b_blob: []const u8,
    /// Present when `layout == .sme2_panel64` (fast L2-resident SME2 `ld1b` path).
    sme_adapter: ?sme.SmeStepLoraAdapter = null,

    /// Evaluates `y_f16[row, 0..n] += scale * ((x_f16[row, 0..k] @ A^T) @ B^T)` for
    /// a single row in-place with ZERO heap allocations using caller scratch `z_f32[0..rank]`.
    pub fn applySingleRowF16(
        self: *const LoraLinearPair,
        x_row: []const f16,
        y_row: []f16,
        z_scratch: []f32,
        z_u16_scratch: []u16,
        dispatcher: ?*sme.SmeDispatcher,
    ) sme.Error!void {
        if (self.scale == 0.0) return;
        const r: usize = self.rank;
        const n: usize = self.n;
        const k: usize = self.k;
        std.debug.assert(x_row.len >= k and y_row.len >= n and z_scratch.len >= r);

        switch (self.layout) {
            .row_major => {
                const a_u16: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, self.a_blob[0 .. r * k * 2]));
                const b_u16: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, self.b_blob[0 .. n * r * 2]));

                // 1. z[j] = scale * dot(x[0..k], A[j, 0..k])
                for (0..r) |j| {
                    const a_row = a_u16[j * k ..][0..k];
                    var acc: f32 = 0.0;
                    for (0..k) |t| {
                        const xv: f32 = @floatCast(x_row[t]);
                        const av: f32 = quant.floatBitsToF32(self.dtype, a_row[t]);
                        acc += xv * av;
                    }
                    z_scratch[j] = acc * self.scale;
                }

                // 2. y[o] += dot(z[0..r], B[o, 0..r])
                for (0..n) |o| {
                    const b_row = b_u16[o * r ..][0..r];
                    var acc: f32 = @floatCast(y_row[o]);
                    for (0..r) |j| {
                        const bv: f32 = quant.floatBitsToF32(self.dtype, b_row[j]);
                        acc += z_scratch[j] * bv;
                    }
                    y_row[o] = @floatCast(acc);
                }
            },
            .sme2_panel64 => {
                if (dispatcher != null and self.sme_adapter != null and self.dtype == .f16 and z_u16_scratch.len >= sme.GROUP_SIZE) {
                    const x_u16: []const u16 = @ptrCast(x_row[0..k]);
                    const adp = self.sme_adapter.?;
                    try sme.evalSmeLinear(
                        dispatcher.?,
                        adp.a_weight,
                        x_u16,
                        null,
                        std.mem.sliceAsBytes(z_scratch[0..r]),
                        1,
                        .{ .cache_hint = .temporal, .epilogue = .none, .out_dtype = .f32 },
                    );
                    for (0..r) |j| {
                        z_u16_scratch[j] = quant.f32ToFloatBits(.f16, z_scratch[j] * self.scale);
                    }
                    @memset(z_u16_scratch[r..sme.GROUP_SIZE], 0);
                    self.addPanel64BToF16Row(z_u16_scratch[0..sme.GROUP_SIZE], y_row);
                    return;
                }
                self.applyPanel64ScalarF16(x_row, y_row, z_scratch);
            },
        }
    }

    fn applyPanel64ScalarF16(
        self: *const LoraLinearPair,
        x_row: []const f16,
        y_row: []f16,
        z_scratch: []f32,
    ) void {
        const r: usize = self.rank;
        const n: usize = self.n;
        const k: usize = self.k;
        const k_groups: usize = k / sme.GROUP_SIZE;
        const g_bytes: usize = @intCast(amf.SME2_F16_GROUP_BYTES);

        // 1. z[j] = scale * (x @ A_panel64^T)[j]
        @memset(z_scratch[0..r], 0.0);
        const a_panels = (r + sme.PANEL_COLS - 1) / sme.PANEL_COLS;
        for (0..a_panels) |p| {
            for (0..k_groups) |g| {
                const grp_off = (p * k_groups + g) * g_bytes;
                const grp = self.a_blob[grp_off .. grp_off + g_bytes];
                const x_base = g * sme.GROUP_SIZE;
                for (0..32) |kp| {
                    const x0: f32 = @floatCast(x_row[x_base + 2 * kp]);
                    const x1: f32 = @floatCast(x_row[x_base + 2 * kp + 1]);
                    const chunk_off = kp * 64;
                    for (0..sme.PANEL_COLS) |c| {
                        const row_idx = p * sme.PANEL_COLS + c;
                        if (row_idx >= r) break;
                        const w0 = std.mem.readInt(u16, grp[chunk_off + c * 4 + 0 ..][0..2], .little);
                        const w1 = std.mem.readInt(u16, grp[chunk_off + c * 4 + 2 ..][0..2], .little);
                        z_scratch[row_idx] += x0 * quant.floatBitsToF32(self.dtype, w0) +
                            x1 * quant.floatBitsToF32(self.dtype, w1);
                    }
                }
            }
        }
        for (0..r) |j| z_scratch[j] *= self.scale;

        // 2. y[o] += (z_padded @ B_panel64^T)[o]
        const b_panels = (n + sme.PANEL_COLS - 1) / sme.PANEL_COLS;
        for (0..b_panels) |p| {
            const grp_off = p * g_bytes;
            const grp = self.b_blob[grp_off .. grp_off + g_bytes];
            var acc: [sme.PANEL_COLS]f32 = @splat(0.0);
            const kp_limit: usize = (r + 1) / 2;
            for (0..kp_limit) |kp| {
                const z0: f32 = z_scratch[2 * kp];
                const z1: f32 = if (2 * kp + 1 < r) z_scratch[2 * kp + 1] else 0.0;
                const chunk_off = kp * 64;
                for (0..sme.PANEL_COLS) |c| {
                    const col = p * sme.PANEL_COLS + c;
                    if (col >= n) break;
                    const w0 = std.mem.readInt(u16, grp[chunk_off + c * 4 + 0 ..][0..2], .little);
                    const w1 = std.mem.readInt(u16, grp[chunk_off + c * 4 + 2 ..][0..2], .little);
                    acc[c] += z0 * quant.floatBitsToF32(self.dtype, w0) +
                        z1 * quant.floatBitsToF32(self.dtype, w1);
                }
            }
            for (0..sme.PANEL_COLS) |c| {
                const col = p * sme.PANEL_COLS + c;
                if (col >= n) break;
                y_row[col] = @floatCast(@as(f32, @floatCast(y_row[col])) + acc[c]);
            }
        }
    }

    fn addPanel64BToF16Row(self: *const LoraLinearPair, z_u16: []const u16, y_row: []f16) void {
        const n: usize = self.n;
        const r: usize = self.rank;
        const g_bytes: usize = @intCast(amf.SME2_F16_GROUP_BYTES);
        const b_panels = (n + sme.PANEL_COLS - 1) / sme.PANEL_COLS;
        const kp_limit: usize = (r + 1) / 2;
        for (0..b_panels) |p| {
            const grp = self.b_blob[p * g_bytes ..][0..g_bytes];
            var acc: [sme.PANEL_COLS]f32 = @splat(0.0);
            for (0..kp_limit) |kp| {
                const z0 = quant.floatBitsToF32(self.dtype, z_u16[2 * kp]);
                const z1 = quant.floatBitsToF32(self.dtype, z_u16[2 * kp + 1]);
                const chunk_off = kp * 64;
                for (0..sme.PANEL_COLS) |c| {
                    const col = p * sme.PANEL_COLS + c;
                    if (col >= n) break;
                    const w0 = std.mem.readInt(u16, grp[chunk_off + c * 4 + 0 ..][0..2], .little);
                    const w1 = std.mem.readInt(u16, grp[chunk_off + c * 4 + 2 ..][0..2], .little);
                    acc[c] += z0 * quant.floatBitsToF32(self.dtype, w0) +
                        z1 * quant.floatBitsToF32(self.dtype, w1);
                }
            }
            for (0..sme.PANEL_COLS) |c| {
                const col = p * sme.PANEL_COLS + c;
                if (col >= n) break;
                y_row[col] = @floatCast(@as(f32, @floatCast(y_row[col])) + acc[c]);
            }
        }
    }
};

pub const LayerAdapters = struct {
    /// Indexed by `@backingInt(TargetProjection)`.
    projections: [TargetProjection.COUNT]?LoraLinearPair = @splat(null),
    target_mask: u8 = 0,

    pub fn hasTarget(self: *const LayerAdapters, target: TargetProjection) bool {
        return (self.target_mask & target.bit()) != 0;
    }

    pub fn hasAnyQkv(self: *const LayerAdapters) bool {
        const mask = TargetProjection.q_proj.bit() | TargetProjection.k_proj.bit() | TargetProjection.v_proj.bit();
        return (self.target_mask & mask) != 0;
    }

    pub fn hasAnyGateUp(self: *const LayerAdapters) bool {
        const mask = TargetProjection.gate_proj.bit() | TargetProjection.up_proj.bit();
        return (self.target_mask & mask) != 0;
    }
};

pub const LoadedAdapter = struct {
    in_use: bool = false,
    name: [MAX_ADAPTER_NAME_LEN]u8 = @splat(0),
    name_len: u8 = 0,
    rank: u32 = 0,
    lora_alpha: f32 = 0.0,
    scale: f32 = 0.0,
    target_mask: u8 = 0,
    wired_bytes: usize = 0,
    num_layers: u32 = 0,
    layers: []LayerAdapters = &.{},

    pub fn nameSlice(self: *const LoadedAdapter) []const u8 {
        return self.name[0..self.name_len];
    }

    pub fn layerAdapters(self: *const LoadedAdapter, layer_idx: usize) ?*const LayerAdapters {
        if (!self.in_use or layer_idx >= self.layers.len) return null;
        return &self.layers[layer_idx];
    }
};

pub const LoraRegistryConfig = struct {
    /// Preallocated wired pool capacity in bytes (`> 0`, 16 KiB aligned, `<= 4 MiB`).
    pool_bytes: usize = MAX_LORA_POOL_BYTES,
    /// `R9` wired budget enforcing that this LoRA pool stays inside startup wired headroom
    /// without ever unpinning base model weights.
    wired_budget: ?budget_mod.Budget = null,
    /// Base model shape against which every hot-loaded `.amlora` is validated.
    base_shape: BaseModelShape,
    /// Optional shared SME2 dispatcher for `sme2_panel64` LoRA acceleration.
    sme_dispatcher: ?*sme.SmeDispatcher = null,
};

pub const LoraRegistryStats = struct {
    active_adapters: u32,
    max_adapters: u32,
    pool_capacity_bytes: usize,
    pool_used_bytes: usize,
    pool_free_bytes: usize,
    total_loads: u64,
    total_unloads: u64,
};

/// Preallocated wired-RAM multi-LoRA registry (`W19`, `ADR-0022`).
///
/// Owns ONE fixed `page.WiredRegion` (`<= 4 MiB`, `mlock`ed once at `init`) and
/// manages hot-load / hot-unload of up to `MAX_ADAPTERS` `.amlora` bundles via
/// a 16 KiB page-ownership bitmap. Zero OS `mlock`/`munlock` churn at runtime,
/// zero base-weight unpinning, and zero heap allocations during per-token delta
/// evaluation.
pub const LoraRegistry = struct {
    const FREE_PAGE_OWNER: u8 = 0xFF;

    pool: page.WiredRegion,
    total_pages: usize,
    page_owner: [MAX_POOL_PAGES]u8,
    used_pages: usize,

    base_shape: BaseModelShape,
    sme_dispatcher: ?*sme.SmeDispatcher,
    slots: [MAX_ADAPTERS]LoadedAdapter,
    active_count: u32,
    total_loads: u64,
    total_unloads: u64,

    // Preallocated zero-alloc evaluation scratch (inside struct — no runtime heap alloc).
    z_f32_scratch: [MAX_LORA_RANK]f32,
    z_u16_scratch: [sme.GROUP_SIZE]u16,

    pub fn init(cfg: LoraRegistryConfig) LoraError!LoraRegistry {
        if (cfg.pool_bytes == 0 or (cfg.pool_bytes % page.APPLE_PAGE_SIZE) != 0) {
            return error.InvalidConfig;
        }
        if (cfg.pool_bytes > MAX_LORA_POOL_BYTES) {
            return error.PoolCapExceeded;
        }
        if (cfg.base_shape.num_layers == 0 or cfg.base_shape.hidden_size == 0) {
            return error.InvalidConfig;
        }
        if (cfg.wired_budget) |b| {
            try b.checkPlan(cfg.pool_bytes);
        }

        const region = try page.WiredRegion.init(cfg.pool_bytes);
        return .{
            .pool = region,
            .total_pages = cfg.pool_bytes / page.APPLE_PAGE_SIZE,
            .page_owner = @splat(FREE_PAGE_OWNER),
            .used_pages = 0,
            .base_shape = cfg.base_shape,
            .sme_dispatcher = cfg.sme_dispatcher,
            .slots = @splat(.{}),
            .active_count = 0,
            .total_loads = 0,
            .total_unloads = 0,
            .z_f32_scratch = @splat(0.0),
            .z_u16_scratch = @splat(0),
        };
    }

    pub fn deinit(self: *LoraRegistry, allocator: std.mem.Allocator) void {
        for (&self.slots) |*slot| {
            if (slot.in_use and slot.layers.len > 0) {
                allocator.free(slot.layers);
                slot.* = .{};
            }
        }
        self.active_count = 0;
        self.used_pages = 0;
        self.pool.deinit();
    }

    /// O(1) hot-path gate checked strictly OUTSIDE transformer layer loops.
    pub inline fn activeAdapterCount(self: *const LoraRegistry) u32 {
        return self.active_count;
    }

    pub fn stats(self: *const LoraRegistry) LoraRegistryStats {
        const used_bytes = self.used_pages * page.APPLE_PAGE_SIZE;
        const cap_bytes = self.total_pages * page.APPLE_PAGE_SIZE;
        return .{
            .active_adapters = self.active_count,
            .max_adapters = MAX_ADAPTERS,
            .pool_capacity_bytes = cap_bytes,
            .pool_used_bytes = used_bytes,
            .pool_free_bytes = cap_bytes - used_bytes,
            .total_loads = self.total_loads,
            .total_unloads = self.total_unloads,
        };
    }

    /// Resolves an adapter name to its fixed slot index (`null` if not loaded).
    pub fn findSlotByName(self: *const LoraRegistry, name: []const u8) ?u32 {
        if (self.active_count == 0 or name.len == 0) return null;
        for (&self.slots, 0..) |*slot, i| {
            if (slot.in_use and std.mem.eql(u8, slot.nameSlice(), name)) {
                return @intCast(i);
            }
        }
        return null;
    }

    /// Parses `"<base>:<adapter>"` and resolves `<adapter>` to a loaded slot index.
    /// Returns `null` for plain `"<base>"`, or `error.AdapterNotFound` if a named
    /// adapter suffix is not currently loaded.
    pub fn resolveModelSelector(self: *const LoraRegistry, model_str: []const u8) LoraError!?u32 {
        const sel = parseModelSelector(model_str);
        const adp_name = sel.adapter orelse return null;
        return self.findSlotByName(adp_name) orelse error.AdapterNotFound;
    }

    pub fn adapterAt(self: *const LoraRegistry, slot_id: u32) ?*const LoadedAdapter {
        if (slot_id >= MAX_ADAPTERS) return null;
        if (!self.slots[slot_id].in_use) return null;
        return &self.slots[slot_id];
    }

    /// Hot-loads an `.amlora` (`.amf`-based) bundle from disk (`F_NOCACHE`) into
    /// the preallocated wired L2 pool under `name`. Returns the assigned slot ID.
    pub fn loadAdapterFromFile(
        self: *LoraRegistry,
        allocator: std.mem.Allocator,
        name: []const u8,
        amlora_path: [:0]const u8,
    ) LoraError!u32 {
        var file = try amf.AmfFile.open(allocator, amlora_path);
        defer file.close(allocator);
        return self.loadAdapterInternal(allocator, name, .{ .file = &file });
    }

    /// Hot-loads an `.amlora` (`.amf`-based) image from memory into the
    /// preallocated wired L2 pool under `name`. Returns the assigned slot ID.
    pub fn loadAdapterFromBytes(
        self: *LoraRegistry,
        allocator: std.mem.Allocator,
        name: []const u8,
        amlora_bytes: []const u8,
    ) LoraError!u32 {
        var parsed = try amf.Parsed.parseBytes(allocator, amlora_bytes);
        defer parsed.deinit(allocator);
        return self.loadAdapterInternal(allocator, name, .{ .bytes = .{ .parsed = &parsed, .raw = amlora_bytes } });
    }

    /// Hot-unloads adapter `name`, immediately returning all of its wired 16 KiB
    /// pages back to the preallocated pool without touching base weights.
    pub fn unloadAdapter(self: *LoraRegistry, allocator: std.mem.Allocator, name: []const u8) LoraError!void {
        const slot_id = self.findSlotByName(name) orelse return error.AdapterNotFound;
        try self.unloadSlot(allocator, slot_id);
    }

    pub fn unloadSlot(self: *LoraRegistry, allocator: std.mem.Allocator, slot_id: u32) LoraError!void {
        if (slot_id >= MAX_ADAPTERS or !self.slots[slot_id].in_use) return error.AdapterNotFound;
        const owner: u8 = @intCast(slot_id);
        for (0..self.total_pages) |p| {
            if (self.page_owner[p] == owner) {
                self.page_owner[p] = FREE_PAGE_OWNER;
                self.used_pages -= 1;
            }
        }
        if (self.slots[slot_id].layers.len > 0) {
            allocator.free(self.slots[slot_id].layers);
        }
        self.slots[slot_id] = .{};
        self.active_count -= 1;
        self.total_unloads += 1;
    }

    /// Returns `true` if `adapter_slot` targets any projection in `(layer_idx, mask)`.
    pub fn slotTargetsLayerMask(self: *const LoraRegistry, slot_id: u32, layer_idx: usize, mask: u8) bool {
        const adp = self.adapterAt(slot_id) orelse return false;
        const la = adp.layerAdapters(layer_idx) orelse return false;
        return (la.target_mask & mask) != 0;
    }

    /// Returns `true` if ANY slot in `batch_lora_slots` targets `(layer_idx, mask)`.
    pub fn batchTargetsLayerMask(self: *const LoraRegistry, batch_lora_slots: []const u32, layer_idx: usize, mask: u8) bool {
        for (batch_lora_slots) |sid| {
            if (sid == NO_ADAPTER) continue;
            if (self.slotTargetsLayerMask(sid, layer_idx, mask)) return true;
        }
        return false;
    }

    /// Applies a single adapter's rank-`r` projection delta across `num_rows`
    /// (`m`) rows of `[num_rows, k]` -> `[num_rows, n]` in-place (`0` heap allocs).
    pub fn applySingleAdapterProjectionF16(
        self: *LoraRegistry,
        slot_id: u32,
        layer_idx: usize,
        target: TargetProjection,
        x_f16: []const f16,
        y_f16: []f16,
        num_rows: usize,
    ) sme.Error!void {
        if (slot_id == NO_ADAPTER) return;
        const adp = self.adapterAt(slot_id) orelse return;
        const la = adp.layerAdapters(layer_idx) orelse return;
        const pair = la.projections[@backingInt(target)] orelse return;

        const k: usize = pair.k;
        const n: usize = pair.n;
        for (0..num_rows) |row| {
            try pair.applySingleRowF16(
                x_f16[row * k ..][0..k],
                y_f16[row * n ..][0..n],
                &self.z_f32_scratch,
                &self.z_u16_scratch,
                self.sme_dispatcher,
            );
        }
    }

    /// Applies per-row heterogeneous LoRA deltas across a `C14` continuous-batching
    /// step where row `b` uses `batch_lora_slots[b]` (`NO_ADAPTER` for base-only
    /// requests, or distinct loaded adapter slots across concurrent requests!).
    pub fn applyBatchProjectionF16(
        self: *LoraRegistry,
        batch_lora_slots: []const u32,
        layer_idx: usize,
        target: TargetProjection,
        x_f16: []const f16,
        y_f16: []f16,
    ) sme.Error!void {
        const dims = self.base_shape.expectedDims(target);
        const k: usize = dims.k;
        const n: usize = dims.n;
        for (batch_lora_slots, 0..) |sid, row| {
            if (sid == NO_ADAPTER) continue;
            const adp = self.adapterAt(sid) orelse continue;
            const la = adp.layerAdapters(layer_idx) orelse continue;
            const pair = la.projections[@backingInt(target)] orelse continue;
            try pair.applySingleRowF16(
                x_f16[row * k ..][0..k],
                y_f16[row * n ..][0..n],
                &self.z_f32_scratch,
                &self.z_u16_scratch,
                self.sme_dispatcher,
            );
        }
    }

    // -----------------------------------------------------------------------
    // Internal `.amlora` Loader & Page-Bitmap Allocator
    // -----------------------------------------------------------------------

    const Source = union(enum) {
        file: *const amf.AmfFile,
        bytes: struct { parsed: *const amf.Parsed, raw: []const u8 },

        fn parsed(self: Source) *const amf.Parsed {
            return switch (self) {
                .file => |f| &f.parsed,
                .bytes => |b| b.parsed,
            };
        }

        fn readMetadataAlloc(self: Source, allocator: std.mem.Allocator, name: []const u8) LoraError!?[]u8 {
            return switch (self) {
                .file => |f| try f.readMetadataAlloc(allocator, name),
                .bytes => |b| blk: {
                    const slice = (try b.parsed.extractMetadataFromBytes(b.raw, name)) orelse break :blk null;
                    break :blk try allocator.dupe(u8, slice);
                },
            };
        }

        fn readBlobInto(self: Source, entry: *const amf.TensorEntry, dst: []align(page.APPLE_PAGE_SIZE) u8) LoraError!void {
            const len: usize = @intCast(entry.blob_bytes);
            if (dst.len < len) return error.BufferTooSmall;
            switch (self) {
                .bytes => |b| {
                    const start: usize = @intCast(entry.blob_offset);
                    if (start + len > b.raw.len) return error.BlobOutOfBounds;
                    @memcpy(dst[0..len], b.raw[start .. start + len]);
                },
                .file => |f| {
                    var done: usize = 0;
                    while (done < len) {
                        const rc = std.c.pread(f.fd, dst.ptr + done, len - done, @intCast(entry.blob_offset + done));
                        if (rc < 0) {
                            const err = std.c.errno(rc);
                            if (err == .INTR) continue;
                            return error.ReadFailed;
                        }
                        if (rc == 0) return error.ShortRead;
                        done += @intCast(rc);
                    }
                },
            }
            if (std.hash.Crc32.hash(dst[0..len]) != entry.blob_crc32) {
                return error.BlobChecksumMismatch;
            }
        }
    };

    fn allocPoolPages(self: *LoraRegistry, owner: u8, num_pages: usize) LoraError![]align(page.APPLE_PAGE_SIZE) u8 {
        if (num_pages == 0 or num_pages > self.total_pages) return error.PoolOutOfSpace;
        var run_start: usize = 0;
        var run_len: usize = 0;
        for (0..self.total_pages) |p| {
            if (self.page_owner[p] == FREE_PAGE_OWNER) {
                if (run_len == 0) run_start = p;
                run_len += 1;
                if (run_len == num_pages) {
                    for (run_start..run_start + num_pages) |q| {
                        self.page_owner[q] = owner;
                    }
                    self.used_pages += num_pages;
                    const byte_off = run_start * page.APPLE_PAGE_SIZE;
                    const byte_len = num_pages * page.APPLE_PAGE_SIZE;
                    return @alignCast(self.pool.bytes[byte_off .. byte_off + byte_len]);
                }
            } else {
                run_len = 0;
            }
        }
        return error.PoolOutOfSpace;
    }

    fn releaseOwnerPages(self: *LoraRegistry, owner: u8) void {
        for (0..self.total_pages) |p| {
            if (self.page_owner[p] == owner) {
                self.page_owner[p] = FREE_PAGE_OWNER;
                self.used_pages -= 1;
            }
        }
    }

    fn parseLoraTensorName(name: []const u8) ?struct { layer: u32, target: TargetProjection, is_a: bool } {
        if (!std.mem.startsWith(u8, name, "layers.")) return null;
        const rest = name["layers.".len..];
        const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
        const layer_idx = std.fmt.parseInt(u32, rest[0..dot], 10) catch return null;
        const tail = rest[dot + 1 ..];

        const is_a = std.mem.endsWith(u8, tail, ".lora_a");
        const is_b = std.mem.endsWith(u8, tail, ".lora_b");
        if (!is_a and !is_b) return null;
        const mod_path = tail[0 .. tail.len - ".lora_a".len];

        const all_targets = [_]TargetProjection{
            .q_proj,
            .k_proj,
            .v_proj,
            .o_proj,
            .gate_proj,
            .up_proj,
            .down_proj,
        };
        for (all_targets) |tp| {
            if (std.mem.eql(u8, mod_path, tp.suffix())) {
                return .{ .layer = layer_idx, .target = tp, .is_a = is_a };
            }
        }
        return null;
    }

    fn loadAdapterInternal(
        self: *LoraRegistry,
        allocator: std.mem.Allocator,
        name: []const u8,
        source: Source,
    ) LoraError!u32 {
        if (name.len == 0 or name.len > MAX_ADAPTER_NAME_LEN) return error.InvalidAdapterName;
        if (std.mem.indexOfScalar(u8, name, ':') != null) return error.InvalidAdapterName;
        if (self.findSlotByName(name) != null) return error.DuplicateAdapterName;

        var slot_id_opt: ?u32 = null;
        for (&self.slots, 0..) |*slot, i| {
            if (!slot.in_use) {
                slot_id_opt = @intCast(i);
                break;
            }
        }
        const slot_id = slot_id_opt orelse return error.RegistryFull;
        const owner: u8 = @intCast(slot_id);

        const parsed = source.parsed();
        var cfg_meta: ?AdapterConfigMeta = null;
        if (try source.readMetadataAlloc(allocator, amf.META_ADAPTER_CONFIG_JSON)) |cfg_json| {
            defer allocator.free(cfg_json);
            cfg_meta = try parseAdapterConfigJson(allocator, cfg_json);
        }

        const num_layers: usize = self.base_shape.num_layers;
        const layers = try allocator.alloc(LayerAdapters, num_layers);
        errdefer allocator.free(layers);
        @memset(layers, .{});

        // Pre-check total wired pages required before mutating pool state.
        var needed_pages: usize = 0;
        var found_any_pair = false;
        for (parsed.entries) |*e| {
            if (amf.isMetaEntry(e)) continue;
            if ((try e.tensorKind()) != .loop_adapter) continue;
            const info = parseLoraTensorName(e.nameSlice()) orelse continue;
            if (info.layer >= self.base_shape.num_layers) return error.LayerOutOfRange;
            needed_pages += @intCast(e.blob_bytes / page.APPLE_PAGE_SIZE);
        }
        if (needed_pages == 0) return error.MissingPairedTensor;
        if (self.used_pages + needed_pages > self.total_pages) return error.PoolOutOfSpace;

        errdefer self.releaseOwnerPages(owner);

        var inferred_rank: u32 = if (cfg_meta) |m| m.rank else 0;
        var effective_scale: f32 = if (cfg_meta) |m| m.scale else 1.0;
        var effective_alpha: f32 = if (cfg_meta) |m| m.lora_alpha else 0.0;
        var combined_target_mask: u8 = 0;
        var wired_bytes: usize = 0;

        for (parsed.entries) |*a_entry| {
            if (amf.isMetaEntry(a_entry)) continue;
            if ((try a_entry.tensorKind()) != .loop_adapter) continue;
            const info = parseLoraTensorName(a_entry.nameSlice()) orelse continue;
            if (!info.is_a) continue;

            var b_name_buf: [64]u8 = undefined;
            const b_name = std.fmt.bufPrint(&b_name_buf, "layers.{d}.{s}.lora_b", .{
                info.layer,
                info.target.suffix(),
            }) catch return error.InvalidAdapterName;
            const b_entry = parsed.find(b_name) orelse return error.MissingPairedTensor;
            if ((try b_entry.tensorKind()) != .loop_adapter) return error.InvalidTensorKind;

            const a_dt = try a_entry.dataType();
            const b_dt = try b_entry.dataType();
            if (a_dt != b_dt) return error.UnsupportedDType;
            const fdt: quant.FloatDType = switch (a_dt) {
                .f16 => .f16,
                .bf16 => .bf16,
                else => return error.UnsupportedDType,
            };

            const a_lay = try a_entry.tensorLayout();
            const b_lay = try b_entry.tensorLayout();
            if (a_lay != b_lay) return error.InvalidTensorLayout;
            if (a_entry.rank != 2 or b_entry.rank != 2) return error.InvalidRank;

            const rank = a_entry.shape[0];
            if (rank == 0 or rank > MAX_LORA_RANK) return error.UnsupportedRank;
            if (inferred_rank == 0) {
                inferred_rank = rank;
                if (cfg_meta == null) {
                    effective_alpha = @floatFromInt(rank);
                    effective_scale = 1.0;
                }
            } else if (rank != inferred_rank) {
                return error.UnsupportedRank;
            }

            const expected = self.base_shape.expectedDims(info.target);
            const k = a_entry.shape[1];
            const n = b_entry.shape[0];
            if (n != expected.n or k != expected.k) return error.DimensionMismatch;

            switch (a_lay) {
                .row_major => {
                    if (b_entry.shape[1] != rank) return error.DimensionMismatch;
                },
                .sme2_panel64 => {
                    if ((k % sme.GROUP_SIZE) != 0) return error.DimensionMismatch;
                    if (b_entry.shape[1] != sme.GROUP_SIZE) return error.DimensionMismatch;
                },
            }

            const a_pages: usize = @intCast(a_entry.blob_bytes / page.APPLE_PAGE_SIZE);
            const b_pages: usize = @intCast(b_entry.blob_bytes / page.APPLE_PAGE_SIZE);
            const a_dst = try self.allocPoolPages(owner, a_pages);
            const b_dst = try self.allocPoolPages(owner, b_pages);
            try source.readBlobInto(a_entry, a_dst);
            try source.readBlobInto(b_entry, b_dst);
            wired_bytes += a_dst.len + b_dst.len;

            var sme_adp: ?sme.SmeStepLoraAdapter = null;
            if (a_lay == .sme2_panel64) {
                sme_adp = try sme.SmeStepLoraAdapter.fromAmfEntries(
                    a_entry,
                    a_dst,
                    b_entry,
                    b_dst,
                    effective_scale,
                );
            }

            const t_idx: usize = @backingInt(info.target);
            layers[info.layer].projections[t_idx] = .{
                .rank = rank,
                .n = n,
                .k = k,
                .scale = effective_scale,
                .layout = a_lay,
                .dtype = fdt,
                .a_blob = a_dst,
                .b_blob = b_dst,
                .sme_adapter = sme_adp,
            };
            layers[info.layer].target_mask |= info.target.bit();
            combined_target_mask |= info.target.bit();
            found_any_pair = true;
        }

        if (!found_any_pair) return error.MissingPairedTensor;

        var slot: LoadedAdapter = .{
            .in_use = true,
            .name = @splat(0),
            .name_len = @intCast(name.len),
            .rank = inferred_rank,
            .lora_alpha = effective_alpha,
            .scale = effective_scale,
            .target_mask = combined_target_mask,
            .wired_bytes = wired_bytes,
            .num_layers = self.base_shape.num_layers,
            .layers = layers,
        };
        @memcpy(slot.name[0..name.len], name);
        self.slots[slot_id] = slot;
        self.active_count += 1;
        self.total_loads += 1;
        return slot_id;
    }
};

// ===========================================================================
// Synthetic `.amlora` Builder Helper & Unit Tests
// ===========================================================================

pub const SyntheticLoraSpec = struct {
    layer: u32,
    target: TargetProjection,
    n: u32,
    k: u32,
    rank: u32,
    layout: amf.TensorLayout = .row_major,
    dtype: quant.FloatDType = .f16,
    a_f32: []const f32,
    b_f32: []const f32,
};

/// Builds a valid 16 KiB-aligned `.amlora` (`.amf` v0) byte image embedding
/// `__meta.adapter_config_json` and `TensorKind.loop_adapter` `lora_a`/`lora_b`
/// blobs (supports both `row_major` and `sme2_panel64`).
pub fn buildSyntheticAmloraBytes(
    allocator: std.mem.Allocator,
    adapter_id: []const u8,
    rank: u32,
    lora_alpha: f32,
    specs: []const SyntheticLoraSpec,
) ![]u8 {
    var cfg_buf: [512]u8 = undefined;
    const cfg_json = try std.fmt.bufPrint(
        &cfg_buf,
        "{{\"peft_type\":\"LORA\",\"r\":{d},\"lora_alpha\":{d:.6},\"target_modules\":[\"q_proj\",\"k_proj\",\"v_proj\",\"o_proj\",\"gate_proj\",\"up_proj\",\"down_proj\"]}}",
        .{ rank, lora_alpha },
    );

    const total_entries = 1 + specs.len * 2;
    const entries = try allocator.alloc(amf.TensorEntry, total_entries);
    defer allocator.free(entries);

    const blobs = try allocator.alloc([]u8, total_entries);
    var built: usize = 0;
    defer {
        for (0..built) |i| allocator.free(blobs[i]);
        allocator.free(blobs);
    }

    // Entry 0: embedded `__meta.adapter_config_json`
    {
        var meta_e = try amf.makeMetadataEntry(amf.META_ADAPTER_CONFIG_JSON, cfg_json.len);
        const aligned = try page.alignUp(cfg_json.len);
        const blob = try allocator.alloc(u8, aligned);
        @memset(blob, 0);
        @memcpy(blob[0..cfg_json.len], cfg_json);
        meta_e.blob_bytes = aligned;
        meta_e.blob_crc32 = std.hash.Crc32.hash(blob);
        entries[0] = meta_e;
        blobs[0] = blob;
        built += 1;
    }

    const dir_offset: usize = page.APPLE_PAGE_SIZE;
    const dir_bytes: usize = try page.alignUp(total_entries * @sizeOf(amf.TensorEntry));

    for (specs, 0..) |sp, si| {
        const amf_dt: amf.DType = if (sp.dtype == .bf16) .bf16 else .f16;
        const pdt: sme.PanelDType = if (sp.dtype == .bf16) .bf16 else .f16;

        var a_name_buf: [48]u8 = undefined;
        const a_name = try std.fmt.bufPrint(&a_name_buf, "layers.{d}.{s}.lora_a", .{ sp.layer, sp.target.suffix() });
        var b_name_buf: [48]u8 = undefined;
        const b_name = try std.fmt.bufPrint(&b_name_buf, "layers.{d}.{s}.lora_b", .{ sp.layer, sp.target.suffix() });

        var ae = amf.TensorEntry.zeroed(a_name, .loop_adapter, amf_dt);
        ae.rank = 2;
        ae.layer = @intCast(sp.layer);
        ae.shape = .{ sp.rank, sp.k, 0, 0 };
        ae.setLayoutAndParamKind(sp.layout, .affine_scale_bias);

        var be = amf.TensorEntry.zeroed(b_name, .loop_adapter, amf_dt);
        be.rank = 2;
        be.layer = @intCast(sp.layer);
        be.setLayoutAndParamKind(sp.layout, .affine_scale_bias);

        const a_data_bytes: usize = switch (sp.layout) {
            .row_major => @as(usize, sp.rank) * sp.k * 2,
            .sme2_panel64 => try sme.packedBlobBytes(pdt, sp.rank, sp.k),
        };
        const a_blob_len = try page.alignUp(a_data_bytes);
        const a_blob = try allocator.alloc(u8, a_blob_len);
        @memset(a_blob, 0);
        switch (sp.layout) {
            .row_major => {
                const dst_u16: []u16 = @alignCast(std.mem.bytesAsSlice(u16, a_blob[0..a_data_bytes]));
                for (sp.a_f32[0 .. @as(usize, sp.rank) * sp.k], 0..) |v, idx| {
                    dst_u16[idx] = quant.f32ToFloatBits(sp.dtype, v);
                }
            },
            .sme2_panel64 => {
                try sme.packDenseFromF32Panel64(pdt, sp.a_f32, sp.rank, sp.k, a_blob[0..a_data_bytes]);
            },
        }
        ae.data_bytes = a_data_bytes;
        ae.blob_bytes = a_blob_len;
        ae.blob_crc32 = std.hash.Crc32.hash(a_blob);

        const b_data_bytes: usize = switch (sp.layout) {
            .row_major => blk: {
                be.shape = .{ sp.n, sp.rank, 0, 0 };
                break :blk @as(usize, sp.n) * sp.rank * 2;
            },
            .sme2_panel64 => blk: {
                be.shape = .{ sp.n, sme.GROUP_SIZE, 0, 0 };
                break :blk try sme.packedBlobBytes(pdt, sp.n, sme.GROUP_SIZE);
            },
        };
        const b_blob_len = try page.alignUp(b_data_bytes);
        const b_blob = try allocator.alloc(u8, b_blob_len);
        @memset(b_blob, 0);
        switch (sp.layout) {
            .row_major => {
                const dst_u16: []u16 = @alignCast(std.mem.bytesAsSlice(u16, b_blob[0..b_data_bytes]));
                for (sp.b_f32[0 .. @as(usize, sp.n) * sp.rank], 0..) |v, idx| {
                    dst_u16[idx] = quant.f32ToFloatBits(sp.dtype, v);
                }
            },
            .sme2_panel64 => {
                try sme.packStepLoraBDenseFromF32Panel64(pdt, sp.b_f32, sp.n, sp.rank, b_blob[0..b_data_bytes]);
            },
        }
        be.data_bytes = b_data_bytes;
        be.blob_bytes = b_blob_len;
        be.blob_crc32 = std.hash.Crc32.hash(b_blob);

        entries[1 + si * 2] = ae;
        blobs[1 + si * 2] = a_blob;
        built += 1;
        entries[1 + si * 2 + 1] = be;
        blobs[1 + si * 2 + 1] = b_blob;
        built += 1;
    }

    var cursor: usize = dir_offset + dir_bytes;
    for (0..total_entries) |i| {
        entries[i].blob_offset = cursor;
        cursor += blobs[i].len;
    }

    const image = try allocator.alloc(u8, cursor);
    errdefer allocator.free(image);
    @memset(image, 0);

    for (0..total_entries) |i| {
        const off: usize = @intCast(entries[i].blob_offset);
        @memcpy(image[off .. off + blobs[i].len], blobs[i]);
    }
    const dir_slice = std.mem.sliceAsBytes(entries);
    @memcpy(image[dir_offset .. dir_offset + dir_slice.len], dir_slice);

    var hdr: amf.Header = .{
        .magic = amf.MAGIC,
        .version = amf.FORMAT_VERSION,
        .flags = amf.FLAG_EMBEDDED_METADATA,
        .page_size = page.APPLE_PAGE_SIZE,
        .checksum_algo = amf.CHECKSUM_CRC32_IEEE,
        .tensor_count = @intCast(total_entries),
        .reserved0 = 0,
        .directory_offset = dir_offset,
        .directory_bytes = dir_bytes,
        .directory_crc32 = std.hash.Crc32.hash(dir_slice),
        .header_crc32 = 0,
        .model_id = @splat(0),
        .reserved1 = 0,
    };
    const mid_len = @min(adapter_id.len, hdr.model_id.len - 1);
    @memcpy(hdr.model_id[0..mid_len], adapter_id[0..mid_len]);
    hdr.header_crc32 = amf.computeHeaderCrc32(&hdr);
    @memcpy(image[0..@sizeOf(amf.Header)], std.mem.asBytes(&hdr));
    return image;
}

test "W19 parseModelSelector handles base, base:adapter, and colon-containing base names" {
    const s0 = parseModelSelector("qwen3-30b");
    try std.testing.expectEqualStrings("qwen3-30b", s0.base);
    try std.testing.expect(s0.adapter == null);

    const s1 = parseModelSelector("qwen3-30b:sql-v1");
    try std.testing.expectEqualStrings("qwen3-30b", s1.base);
    try std.testing.expectEqualStrings("sql-v1", s1.adapter.?);

    const s2 = parseModelSelector("qwen3:30b:code-lora");
    try std.testing.expectEqualStrings("qwen3:30b", s2.base);
    try std.testing.expectEqualStrings("code-lora", s2.adapter.?);

    const s3 = parseModelSelector("qwen3:");
    try std.testing.expectEqualStrings("qwen3", s3.base);
    try std.testing.expect(s3.adapter == null);
}

test "W19 LoraRegistry enforces 4 MiB wired cap, hot-loads/unloads without leaks, and matches FP32 reference across row_major & sme2_panel64" {
    const gpa = std.testing.allocator;
    const base_shape: BaseModelShape = .{
        .num_layers = 2,
        .hidden_size = 64,
        .q_dim = 64,
        .kv_dim = 32,
        .intermediate_size = 128,
    };

    // 1. Reject pool > 4 MiB L2 cap.
    try std.testing.expectError(error.PoolCapExceeded, LoraRegistry.init(.{
        .pool_bytes = MAX_LORA_POOL_BYTES + page.APPLE_PAGE_SIZE,
        .base_shape = base_shape,
    }));

    const pool_bytes: usize = 16 * page.APPLE_PAGE_SIZE; // 256 KiB preallocated wired pool
    const wired_budget = budget_mod.Budget.compute(.{
        .memory_bytes = 64 * 1024 * 1024,
        .user_wire_limit = 32 * 1024 * 1024,
        .global_user_wire_limit = 32 * 1024 * 1024,
        .gpu_working_set_bytes = 32 * 1024 * 1024,
        .configured_cap = null,
        .headroom_bytes = page.APPLE_PAGE_SIZE,
    });
    var reg = try LoraRegistry.init(.{
        .pool_bytes = pool_bytes,
        .wired_budget = wired_budget,
        .base_shape = base_shape,
    });
    defer reg.deinit(gpa);
    try std.testing.expectEqual(@as(u32, 0), reg.activeAdapterCount());

    // Build two deterministic adapters: adapter A (row_major f16) and adapter B (sme2_panel64 f16).
    const r: usize = 4;
    const n: usize = 64;
    const k: usize = 64;
    var a1: [r * k]f32 = undefined;
    var b1: [n * r]f32 = undefined;
    var a2: [r * k]f32 = undefined;
    var b2: [n * r]f32 = undefined;
    for (&a1, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 7)) - 3)) * 0.125;
    for (&b1, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 5)) - 2)) * 0.25;
    for (&a2, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 3) % 9)) - 4)) * 0.125;
    for (&b2, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 2) % 7)) - 3)) * 0.25;

    const img_a = try buildSyntheticAmloraBytes(gpa, "sql-lora", r, 8.0, &.{
        .{ .layer = 0, .target = .q_proj, .n = n, .k = k, .rank = r, .layout = .row_major, .dtype = .f16, .a_f32 = &a1, .b_f32 = &b1 },
    });
    defer gpa.free(img_a);

    const img_b = try buildSyntheticAmloraBytes(gpa, "code-lora", r, 4.0, &.{
        .{ .layer = 0, .target = .q_proj, .n = n, .k = k, .rank = r, .layout = .sme2_panel64, .dtype = .f16, .a_f32 = &a2, .b_f32 = &b2 },
    });
    defer gpa.free(img_b);

    const slot_a = try reg.loadAdapterFromBytes(gpa, "sql", img_a);
    const slot_b = try reg.loadAdapterFromBytes(gpa, "code", img_b);
    try std.testing.expectEqual(@as(u32, 2), reg.activeAdapterCount());
    try std.testing.expectEqual(@as(?u32, slot_a), try reg.resolveModelSelector("qwen3:sql"));
    try std.testing.expectEqual(@as(?u32, slot_b), try reg.resolveModelSelector("qwen3:code"));
    try std.testing.expectEqual(@as(?u32, null), try reg.resolveModelSelector("qwen3"));
    try std.testing.expectError(error.AdapterNotFound, reg.resolveModelSelector("qwen3:missing"));

    // Verify 3-row heterogeneous batch: row 0 = base (NO_ADAPTER), row 1 = sql (row_major), row 2 = code (sme2_panel64).
    var x_batch: [3 * k]f16 = undefined;
    var y_batch: [3 * n]f16 = undefined;
    var y_base: [3 * n]f16 = undefined;
    for (0..3) |row| {
        for (0..k) |j| {
            x_batch[row * k + j] = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast((row * 13 + j) % 11)) - 5)) * 0.25);
        }
        for (0..n) |o| {
            const init_val: f16 = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast((row + o) % 5)) - 2)) * 0.5);
            y_batch[row * n + o] = init_val;
            y_base[row * n + o] = init_val;
        }
    }

    const batch_slots = [_]u32{ NO_ADAPTER, slot_a, slot_b };
    try reg.applyBatchProjectionF16(&batch_slots, 0, .q_proj, &x_batch, &y_batch);

    // Row 0 (NO_ADAPTER) must remain bit-exact untouched!
    try std.testing.expectEqualSlices(f16, y_base[0..n], y_batch[0..n]);

    // Compare row 1 (scale = 8/4 = 2.0) and row 2 (scale = 4/4 = 1.0) against FP32 reference.
    const specs_ref = [_]struct { row: usize, scale: f32, a: []const f32, b: []const f32 }{
        .{ .row = 1, .scale = 2.0, .a = &a1, .b = &b1 },
        .{ .row = 2, .scale = 1.0, .a = &a2, .b = &b2 },
    };
    for (specs_ref) |sr| {
        var z_ref: [r]f32 = undefined;
        for (0..r) |j| {
            var acc: f32 = 0.0;
            for (0..k) |t| acc += @as(f32, @floatCast(x_batch[sr.row * k + t])) * sr.a[j * k + t];
            z_ref[j] = acc * sr.scale;
        }
        for (0..n) |o| {
            var acc: f32 = @floatCast(y_base[sr.row * n + o]);
            for (0..r) |j| acc += z_ref[j] * sr.b[o * r + j];
            const got: f32 = @floatCast(y_batch[sr.row * n + o]);
            try std.testing.expectApproxEqAbs(acc, got, 0.05);
        }
    }

    // Hot-unload out of order ("sql" first), verify immediate page reclamation and reload reuse.
    const used_before = reg.stats().pool_used_bytes;
    try reg.unloadAdapter(gpa, "sql");
    try std.testing.expectEqual(@as(u32, 1), reg.activeAdapterCount());
    try std.testing.expect(reg.stats().pool_used_bytes < used_before);
    _ = try reg.loadAdapterFromBytes(gpa, "sql-reloaded", img_a);
    try std.testing.expectEqual(used_before, reg.stats().pool_used_bytes);
    try reg.unloadAdapter(gpa, "code");
    try reg.unloadAdapter(gpa, "sql-reloaded");
    try std.testing.expectEqual(@as(u32, 0), reg.activeAdapterCount());
    try std.testing.expectEqual(@as(usize, 0), reg.stats().pool_used_bytes);
}
