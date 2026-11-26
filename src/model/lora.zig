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
const chip_profile = @import("../platform/chip_profile.zig");

/// Default L2-friendly wired pool capacity (`4 MiB`, ADR-0022).
pub const DEFAULT_LORA_POOL_BYTES: usize = 4 * 1024 * 1024;
/// Hard wired pool ceiling for universal multi-LoRA + DoRA (`256 MiB`, W30),
/// strictly bounded further by startup `R9` wired headroom (`wired_budget`).
pub const MAX_LORA_POOL_BYTES: usize = 256 * 1024 * 1024;
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
    gate_attn = 7,
    wr_du = 8,
    qkv_proj = 9,
    gate_up_proj = 10,
    q_a_proj = 11,
    q_b_proj = 12,
    kv_a_proj = 13,
    kv_b_proj = 14,
    eh_proj = 15,

    pub const COUNT: usize = 16;

    pub fn label(self: TargetProjection) []const u8 {
        return switch (self) {
            .q_proj => "q_proj",
            .k_proj => "k_proj",
            .v_proj => "v_proj",
            .o_proj => "o_proj",
            .gate_proj => "gate_proj",
            .up_proj => "up_proj",
            .down_proj => "down_proj",
            .gate_attn => "gate_attn",
            .wr_du => "wr_du",
            .qkv_proj => "qkv_proj",
            .gate_up_proj => "gate_up_proj",
            .q_a_proj => "q_a_proj",
            .q_b_proj => "q_b_proj",
            .kv_a_proj => "kv_a_proj",
            .kv_b_proj => "kv_b_proj",
            .eh_proj => "eh_proj",
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
            .gate_attn => "self_attn.gate_attn",
            .wr_du => "self_attn.wr_du",
            .qkv_proj => "self_attn.qkv_proj",
            .gate_up_proj => "mlp.gate_up_proj",
            .q_a_proj => "self_attn.q_a_proj",
            .q_b_proj => "self_attn.q_b_proj",
            .kv_a_proj => "self_attn.kv_a_proj",
            .kv_b_proj => "self_attn.kv_b_proj",
            .eh_proj => "self_attn.eh_proj",
        };
    }

    pub fn bit(self: TargetProjection) u16 {
        return @as(u16, 1) << @intCast(@backingInt(self));
    }

    pub fn parseModuleName(name: []const u8) ?TargetProjection {
        if (std.mem.eql(u8, name, "q_proj") or
            std.mem.eql(u8, name, "self_attn.q_proj") or
            std.mem.eql(u8, name, "self_attn.q_proj.linear") or
            std.mem.eql(u8, name, "linear_attn.q_proj") or
            std.mem.eql(u8, name, "linear_attn.q_proj.linear") or
            std.mem.eql(u8, name, "attn.wq") or
            std.mem.eql(u8, name, "attn.wq_du") or
            std.mem.eql(u8, name, "wq") or
            std.mem.eql(u8, name, "wq_du")) return .q_proj;
        if (std.mem.eql(u8, name, "k_proj") or
            std.mem.eql(u8, name, "self_attn.k_proj") or
            std.mem.eql(u8, name, "self_attn.k_proj.linear") or
            std.mem.eql(u8, name, "linear_attn.k_proj") or
            std.mem.eql(u8, name, "linear_attn.k_proj.linear") or
            std.mem.eql(u8, name, "attn.wk") or
            std.mem.eql(u8, name, "attn.wk_dv") or
            std.mem.eql(u8, name, "wk") or
            std.mem.eql(u8, name, "wk_dv")) return .k_proj;
        if (std.mem.eql(u8, name, "v_proj") or
            std.mem.eql(u8, name, "self_attn.v_proj") or
            std.mem.eql(u8, name, "self_attn.v_proj.linear") or
            std.mem.eql(u8, name, "linear_attn.v_proj") or
            std.mem.eql(u8, name, "linear_attn.v_proj.linear") or
            std.mem.eql(u8, name, "attn.wv") or
            std.mem.eql(u8, name, "attn.wv_dv") or
            std.mem.eql(u8, name, "wv") or
            std.mem.eql(u8, name, "wv_dv")) return .v_proj;
        if (std.mem.eql(u8, name, "o_proj") or
            std.mem.eql(u8, name, "self_attn.o_proj") or
            std.mem.eql(u8, name, "self_attn.o_proj.linear") or
            std.mem.eql(u8, name, "linear_attn.o_proj") or
            std.mem.eql(u8, name, "linear_attn.out_proj") or
            std.mem.eql(u8, name, "out_proj") or
            std.mem.eql(u8, name, "attn.wo") or
            std.mem.eql(u8, name, "attn.wo_ud") or
            std.mem.eql(u8, name, "wo") or
            std.mem.eql(u8, name, "wo_ud")) return .o_proj;
        if (std.mem.eql(u8, name, "gate_proj") or
            std.mem.eql(u8, name, "mlp.gate_proj") or
            std.mem.eql(u8, name, "ffn.w1") or
            std.mem.eql(u8, name, "w1")) return .gate_proj;
        if (std.mem.eql(u8, name, "up_proj") or
            std.mem.eql(u8, name, "mlp.up_proj") or
            std.mem.eql(u8, name, "ffn.w3") or
            std.mem.eql(u8, name, "w3")) return .up_proj;
        if (std.mem.eql(u8, name, "down_proj") or
            std.mem.eql(u8, name, "mlp.down_proj") or
            std.mem.eql(u8, name, "ffn.w2") or
            std.mem.eql(u8, name, "w2")) return .down_proj;
        if (std.mem.eql(u8, name, "gate_attn") or
            std.mem.eql(u8, name, "self_attn.gate_attn") or
            std.mem.eql(u8, name, "attn.gate_attn") or
            std.mem.eql(u8, name, "q_gate_proj") or
            std.mem.eql(u8, name, "self_attn.q_gate_proj") or
            std.mem.eql(u8, name, "linear_attn.g_proj") or
            std.mem.eql(u8, name, "g_proj") or
            std.mem.eql(u8, name, "linear_attn.in_proj_z") or
            std.mem.eql(u8, name, "in_proj_z") or
            std.mem.eql(u8, name, "self_attn.o_proj.gate_proj") or
            std.mem.eql(u8, name, "o_proj.gate_proj")) return .gate_attn;
        if (std.mem.eql(u8, name, "wr_du") or
            std.mem.eql(u8, name, "self_attn.wr_du") or
            std.mem.eql(u8, name, "attn.wr") or
            std.mem.eql(u8, name, "attn.wr_du") or
            std.mem.eql(u8, name, "wr")) return .wr_du;
        if (std.mem.eql(u8, name, "qkv_proj") or
            std.mem.eql(u8, name, "self_attn.qkv_proj") or
            std.mem.eql(u8, name, "linear_attn.in_proj_qkv") or
            std.mem.eql(u8, name, "in_proj_qkv") or
            std.mem.eql(u8, name, "attn.wqkv") or
            std.mem.eql(u8, name, "wqkv")) return .qkv_proj;
        if (std.mem.eql(u8, name, "gate_up_proj") or
            std.mem.eql(u8, name, "mlp.gate_up_proj") or
            std.mem.eql(u8, name, "ffn.w13") or
            std.mem.eql(u8, name, "w13")) return .gate_up_proj;
        if (std.mem.eql(u8, name, "q_a_proj") or
            std.mem.eql(u8, name, "self_attn.q_a_proj") or
            std.mem.eql(u8, name, "attn.q_a_proj")) return .q_a_proj;
        if (std.mem.eql(u8, name, "q_b_proj") or
            std.mem.eql(u8, name, "self_attn.q_b_proj") or
            std.mem.eql(u8, name, "attn.q_b_proj")) return .q_b_proj;
        if (std.mem.eql(u8, name, "kv_a_proj") or
            std.mem.eql(u8, name, "self_attn.kv_a_proj") or
            std.mem.eql(u8, name, "self_attn.kv_a_proj_with_mqa") or
            std.mem.eql(u8, name, "kv_a_proj_with_mqa") or
            std.mem.eql(u8, name, "attn.kv_a_proj")) return .kv_a_proj;
        if (std.mem.eql(u8, name, "kv_b_proj") or
            std.mem.eql(u8, name, "self_attn.kv_b_proj") or
            std.mem.eql(u8, name, "attn.kv_b_proj")) return .kv_b_proj;
        if (std.mem.eql(u8, name, "eh_proj") or
            std.mem.eql(u8, name, "self_attn.eh_proj") or
            std.mem.eql(u8, name, "attn.eh_proj") or
            std.mem.eql(u8, name, "mlp.eh_proj")) return .eh_proj;
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

pub const LoraError = amf.ParseError || amf.IoError || page.WiredRegion.InitError || budget_mod.PlanError || sme.Error || sme.WiredScratchError || error{
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
    target_mask: u16,
    use_dora: bool = false,
};
pub const AdapterMeta = AdapterConfigMeta;

/// Parses PEFT `adapter_config.json` (either embedded in `__meta.adapter_config_json`
/// or sidecar) extracting `r`, `lora_alpha`, derived `scale = lora_alpha / r`,
/// `use_dora`, and `target_modules` bitmask.
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

    const use_dora: bool = if (obj.get("use_dora")) |dv| switch (dv) {
        .bool => |b| b,
        .null => false,
        else => return error.InvalidAdapterConfig,
    } else false;

    var target_mask: u16 = 0;
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
        .use_dora = use_dora,
    };
}

/// Geometry expectations for validating a loaded `.amlora` against the active base model.
pub const BaseModelShape = struct {
    num_layers: u32,
    hidden_size: u32,
    q_dim: u32 = 0,
    kv_dim: u32 = 0,
    intermediate_size: u32 = 0,
    /// When `true` (`W30`), validates `lora_a [rank, K]` and `lora_b [N, rank|64]`
    /// with `K > 0`, `N > 0`, and `layer < num_layers` without requiring Qwen3-only
    /// fixed `q_dim`/`kv_dim`/`intermediate_size` (for heterogeneous hybrid/graph models).
    flexible_dims: bool = false,

    pub fn expectedDims(self: BaseModelShape, target: TargetProjection) struct { n: u32, k: u32 } {
        return switch (target) {
            .q_proj => .{ .n = self.q_dim, .k = self.hidden_size },
            .k_proj, .v_proj => .{ .n = self.kv_dim, .k = self.hidden_size },
            .o_proj => .{ .n = self.hidden_size, .k = self.q_dim },
            .gate_proj, .up_proj => .{ .n = self.intermediate_size, .k = self.hidden_size },
            .down_proj => .{ .n = self.hidden_size, .k = self.intermediate_size },
            .gate_attn, .wr_du => .{ .n = self.q_dim, .k = self.hidden_size },
            .qkv_proj => .{ .n = self.q_dim + 2 * self.kv_dim, .k = self.hidden_size },
            .gate_up_proj => .{ .n = 2 * self.intermediate_size, .k = self.hidden_size },
            .q_a_proj, .q_b_proj, .kv_a_proj, .kv_b_proj, .eh_proj => .{ .n = 0, .k = 0 },
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
    weight_dtype: quant.WeightDType = .f16,
    param_kind: amf.QuantParamKind = .affine_scale_bias,
    a_blob: []const u8,
    b_blob: []const u8,
    /// Present when `layout == .sme2_panel64` (fast L2-resident SME2 `ld1b` / `luti2` path).
    sme_adapter: ?sme.SmeStepLoraAdapter = null,
    /// `true` when this projection carries a PEFT DoRA magnitude vector (`W30`).
    use_dora: bool = false,
    /// Wired `[n]f32` DoRA magnitude vector `m` (owned by adapter pages in `LoraRegistry.pool`).
    dora_mag_f32: ?[]f32 = null,
    /// Wired `[n]f32` precomputed exact PEFT DoRA row scale
    /// `gamma_i = m_i / sqrt(||W_0[i] + scale * (B @ A)[i]||_2^2 + 1e-12)`,
    /// initialized to `1.0` until calibrated against base weights.
    dora_row_scale_f32: ?[]f32 = null,

    pub fn readAValue(self: *const LoraLinearPair, j: usize, t: usize) f32 {
        const r: usize = self.rank;
        const k: usize = self.k;
        std.debug.assert(j < r and t < k);
        switch (self.layout) {
            .row_major => {
                const a_u16: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, self.a_blob[0 .. r * k * 2]));
                return quant.floatBitsToF32(self.dtype, a_u16[j * k + t]);
            },
            .sme2_panel64 => {
                const k_groups = k / sme.GROUP_SIZE;
                const g_bytes: usize = @intCast(amf.SME2_F16_GROUP_BYTES);
                const p = j / sme.PANEL_COLS;
                const pc = j % sme.PANEL_COLS;
                const g = t / sme.GROUP_SIZE;
                const cg = t % sme.GROUP_SIZE;
                const kp = cg / 2;
                const sub = cg % 2;
                const off = (p * k_groups + g) * g_bytes + kp * 64 + pc * 4 + sub * 2;
                const bits = std.mem.readInt(u16, self.a_blob[off..][0..2], .little);
                return quant.floatBitsToF32(self.dtype, bits);
            },
        }
    }

    pub fn readBValue(self: *const LoraLinearPair, o: usize, j: usize) f32 {
        const r: usize = self.rank;
        const n: usize = self.n;
        std.debug.assert(o < n and j < r);
        switch (self.layout) {
            .row_major => {
                const b_u16: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, self.b_blob[0 .. n * r * 2]));
                return quant.floatBitsToF32(self.dtype, b_u16[o * r + j]);
            },
            .sme2_panel64 => {
                const g_bytes: usize = @intCast(amf.SME2_F16_GROUP_BYTES);
                const p = o / sme.PANEL_COLS;
                const pc = o % sme.PANEL_COLS;
                const kp = j / 2;
                const sub = j % 2;
                const off = p * g_bytes + kp * 64 + pc * 4 + sub * 2;
                const bits = std.mem.readInt(u16, self.b_blob[off..][0..2], .little);
                return quant.floatBitsToF32(self.dtype, bits);
            },
        }
    }

    /// Calibrates a single output row `row_idx` of `dora_row_scale_f32` directly from
    /// dequantized base weight row `w0_row_f32[0..k]`:
    /// `gamma[row] = m[row] / sqrt(||W_0[row] + scale * (B @ A)[row]||_2^2 + 1e-12)`.
    pub fn calibrateDoraRowFromBaseRow(
        self: *LoraLinearPair,
        row_idx: usize,
        w0_row_f32: []const f32,
    ) LoraError!void {
        const mag = self.dora_mag_f32 orelse return;
        const out_scale = self.dora_row_scale_f32 orelse return;
        if (!self.use_dora) return;
        if (self.weight_dtype == .q2_affine) return error.UnsupportedDType;
        if (row_idx >= self.n) return error.DimensionMismatch;
        if (w0_row_f32.len < self.k) return error.BufferTooSmall;

        const r: usize = self.rank;
        const k: usize = self.k;
        var b_row: [MAX_LORA_RANK]f32 = undefined;
        for (0..r) |j| b_row[j] = self.readBValue(row_idx, j);

        const s64: f64 = @floatCast(self.scale);
        var norm_sq: f64 = 0.0;
        for (0..k) |t| {
            var ba_t: f64 = 0.0;
            for (0..r) |j| {
                ba_t += @as(f64, b_row[j]) * @as(f64, self.readAValue(j, t));
            }
            const eff: f64 = @as(f64, w0_row_f32[t]) + s64 * ba_t;
            norm_sq += eff * eff;
        }
        const denom: f64 = @sqrt(norm_sq + 1.0e-12);
        out_scale[row_idx] = @floatCast(@as(f64, mag[row_idx]) / denom);
    }

    /// Calibrates all `N` rows of `dora_row_scale_f32` once at adapter load/bind time
    /// from row-major base weight matrix `base_w_f32` (`[n, k]`).
    pub fn calibrateDoraPairFromBaseRows(
        self: *LoraLinearPair,
        base_w_f32: []const f32,
    ) LoraError!void {
        if (!self.use_dora or self.dora_mag_f32 == null or self.dora_row_scale_f32 == null) return;
        const n: usize = self.n;
        const k: usize = self.k;
        const nk = std.math.mul(usize, n, k) catch return error.DimensionMismatch;
        if (base_w_f32.len < nk) return error.BufferTooSmall;
        for (0..n) |i| {
            try self.calibrateDoraRowFromBaseRow(i, base_w_f32[i * k ..][0..k]);
        }
    }

    /// Calibrates all `N` rows of `dora_row_scale_f32` from precomputed base row L2 norms
    /// `||W_0[i]||_2` and optional exact cross inner products `dot(W_0[i], scale * (B @ A)[i])`.
    /// When `cross_dots_f32` is provided, matches `calibrateDoraPairFromBaseRows` exactly via
    /// `||W_0[i] + s(BA)[i]||^2 = ||W_0[i]||^2 + s^2 * b_i^T (A A^T) b_i + 2 * cross_dots[i]`.
    pub fn calibrateDoraPairFromBaseRowNorms(
        self: *LoraLinearPair,
        base_row_norms_f32: []const f32,
        cross_dots_f32: ?[]const f32,
    ) LoraError!void {
        const mag = self.dora_mag_f32 orelse return;
        const out_scale = self.dora_row_scale_f32 orelse return;
        if (!self.use_dora) return;
        if (self.weight_dtype == .q2_affine) return error.UnsupportedDType;

        const n: usize = self.n;
        const k: usize = self.k;
        const r: usize = self.rank;
        if (base_row_norms_f32.len < n) return error.BufferTooSmall;
        if (cross_dots_f32) |cd| {
            if (cd.len < n) return error.BufferTooSmall;
        }

        // Compute Gram matrix G_A = A @ A^T in f64 (`r <= 64`).
        var gram_a: [MAX_LORA_RANK][MAX_LORA_RANK]f64 = undefined;
        for (0..r) |j1| {
            for (j1..r) |j2| {
                var acc: f64 = 0.0;
                for (0..k) |t| {
                    acc += @as(f64, self.readAValue(j1, t)) * @as(f64, self.readAValue(j2, t));
                }
                gram_a[j1][j2] = acc;
                gram_a[j2][j1] = acc;
            }
        }

        const s64: f64 = @floatCast(self.scale);
        const s_sq: f64 = s64 * s64;
        var b_row: [MAX_LORA_RANK]f64 = undefined;
        for (0..n) |i| {
            for (0..r) |j| b_row[j] = @as(f64, self.readBValue(i, j));
            var quad: f64 = 0.0;
            for (0..r) |j1| {
                var inner: f64 = 0.0;
                for (0..r) |j2| inner += gram_a[j1][j2] * b_row[j2];
                quad += b_row[j1] * inner;
            }
            const w0_norm: f64 = @as(f64, base_row_norms_f32[i]);
            const cross_term: f64 = if (cross_dots_f32) |cd| 2.0 * @as(f64, cd[i]) else 0.0;
            const eff_sq: f64 = @max(0.0, w0_norm * w0_norm + s_sq * quad + cross_term);
            const denom: f64 = @sqrt(eff_sq + 1.0e-12);
            out_scale[i] = @floatCast(@as(f64, mag[i]) / denom);
        }
    }

    /// Alias for `applySingleRowF16`.
    pub inline fn evalRowF16(
        self: *const LoraLinearPair,
        x_row: []const f16,
        y_row: []f16,
        z_scratch: []f32,
        z_u16_scratch: []u16,
        dispatcher: ?*sme.SmeDispatcher,
    ) sme.Error!void {
        return self.applySingleRowF16(x_row, y_row, z_scratch, z_u16_scratch, dispatcher);
    }

    /// Evaluates `y_f16[row, 0..n] = gamma * (y_f16[row, 0..n] + scale * ((x_f16[row, 0..k] @ A^T) @ B^T))`
    /// (with `gamma = dora_row_scale_f32` when `use_dora == true`, else `1.0`) for
    /// a single row in-place with ZERO heap allocations using caller scratch `z_f32[0..rank]`.
    pub fn applySingleRowF16(
        self: *const LoraLinearPair,
        x_row: []const f16,
        y_row: []f16,
        z_scratch: []f32,
        z_u16_scratch: []u16,
        dispatcher: ?*sme.SmeDispatcher,
    ) sme.Error!void {
        const dora_scale: ?[]const f32 = if (self.use_dora) self.dora_row_scale_f32 else null;
        const r: usize = self.rank;
        const n: usize = self.n;
        const k: usize = self.k;
        if (self.scale == 0.0) {
            if (dora_scale) |ds| {
                std.debug.assert(y_row.len >= n and ds.len >= n);
                for (0..n) |o| {
                    y_row[o] = @floatCast(@as(f32, @floatCast(y_row[o])) * ds[o]);
                }
            }
            return;
        }
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

                // 2. y[o] = (y[o] + dot(z[0..r], B[o, 0..r])) * (has_dora ? dora_scale[o] : 1.0)
                for (0..n) |o| {
                    const b_row = b_u16[o * r ..][0..r];
                    var acc: f32 = @floatCast(y_row[o]);
                    for (0..r) |j| {
                        const bv: f32 = quant.floatBitsToF32(self.dtype, b_row[j]);
                        acc += z_scratch[j] * bv;
                    }
                    if (dora_scale) |ds| acc *= ds[o];
                    y_row[o] = @floatCast(acc);
                }
            },
            .sme2_panel64 => {
                if (dispatcher != null and self.sme_adapter != null and z_u16_scratch.len >= sme.GROUP_SIZE) {
                    const x_u16: []const u16 = @ptrCast(x_row[0..k]);
                    const adp = self.sme_adapter.?;
                    if (!adp.a_weight.dtype.activationIsBf16()) {
                        try sme.evalSmeLinear(
                            dispatcher.?,
                            adp.a_weight,
                            x_u16,
                            null,
                            std.mem.sliceAsBytes(z_scratch[0..r]),
                            1,
                            .{ .cache_hint = .temporal, .epilogue = .none, .out_dtype = .f32 },
                        );
                        const b_act_dt: quant.FloatDType = if (adp.b_weight.dtype.activationIsBf16()) .bf16 else .f16;
                        for (0..r) |j| {
                            z_u16_scratch[j] = quant.f32ToFloatBits(b_act_dt, z_scratch[j] * self.scale);
                        }
                        @memset(z_u16_scratch[r..sme.GROUP_SIZE], 0);
                        if (adp.b_weight.dtype.isQuantized()) {
                            if (n > 8192) return error.BufferTooSmall;
                            var y_f32_tmp: [8192]f32 = undefined;
                            try sme.evalSmeLinear(
                                dispatcher.?,
                                adp.b_weight,
                                z_u16_scratch[0..sme.GROUP_SIZE],
                                null,
                                std.mem.sliceAsBytes(y_f32_tmp[0..n]),
                                1,
                                .{ .cache_hint = .temporal, .epilogue = .none, .out_dtype = .f32 },
                            );
                            for (0..n) |o| {
                                var val = @as(f32, @floatCast(y_row[o])) + y_f32_tmp[o];
                                if (dora_scale) |ds| val *= ds[o];
                                y_row[o] = @floatCast(val);
                            }
                            return;
                        }
                        self.addPanel64BToF16Row(z_u16_scratch[0..sme.GROUP_SIZE], y_row);
                        return;
                    }
                }
                if (self.sme_adapter) |adp| {
                    if (adp.a_weight.dtype.isQuantized() or adp.b_weight.dtype.isQuantized() or self.weight_dtype == .q2_affine) {
                        if (n > 8192 or z_u16_scratch.len < sme.GROUP_SIZE) return error.BufferTooSmall;
                        const x_u16: []const u16 = @ptrCast(x_row[0..k]);
                        sme.matmulPanel64ScalarRange(
                            adp.a_weight.dtype,
                            adp.a_weight.param_kind,
                            .none,
                            .f32,
                            x_u16,
                            adp.a_weight.w_packed,
                            null,
                            std.mem.sliceAsBytes(z_scratch[0..r]),
                            0,
                            1,
                            0,
                            sme.panelCount(r),
                            r,
                            k,
                        );
                        for (0..r) |j| {
                            z_u16_scratch[j] = quant.f32ToFloatBits(.f16, z_scratch[j] * self.scale);
                        }
                        @memset(z_u16_scratch[r..sme.GROUP_SIZE], 0);
                        var y_f32_tmp: [8192]f32 = undefined;
                        sme.matmulPanel64ScalarRange(
                            adp.b_weight.dtype,
                            adp.b_weight.param_kind,
                            .none,
                            .f32,
                            z_u16_scratch[0..sme.GROUP_SIZE],
                            adp.b_weight.w_packed,
                            null,
                            std.mem.sliceAsBytes(y_f32_tmp[0..n]),
                            0,
                            1,
                            0,
                            sme.panelCount(n),
                            n,
                            sme.GROUP_SIZE,
                        );
                        for (0..n) |o| {
                            var val = @as(f32, @floatCast(y_row[o])) + y_f32_tmp[o];
                            if (dora_scale) |ds| val *= ds[o];
                            y_row[o] = @floatCast(val);
                        }
                        return;
                    }
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
        const dora_scale: ?[]const f32 = if (self.use_dora) self.dora_row_scale_f32 else null;
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

        // 2. y[o] = (y[o] + (z_padded @ B_panel64^T)[o]) * (has_dora ? dora_scale[o] : 1.0)
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
                var val = @as(f32, @floatCast(y_row[col])) + acc[c];
                if (dora_scale) |ds| val *= ds[col];
                y_row[col] = @floatCast(val);
            }
        }
    }

    fn addPanel64BToF16Row(self: *const LoraLinearPair, z_u16: []const u16, y_row: []f16) void {
        const dora_scale: ?[]const f32 = if (self.use_dora) self.dora_row_scale_f32 else null;
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
                var val = @as(f32, @floatCast(y_row[col])) + acc[c];
                if (dora_scale) |ds| val *= ds[col];
                y_row[col] = @floatCast(val);
            }
        }
    }
};

pub const LayerAdapters = struct {
    /// Indexed by `@backingInt(TargetProjection)`.
    projections: [TargetProjection.COUNT]?LoraLinearPair = @splat(null),
    target_mask: u16 = 0,

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
    target_mask: u16 = 0,
    use_dora: bool = false,
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

    pub fn layerAdaptersMut(self: *LoadedAdapter, layer_idx: usize) ?*LayerAdapters {
        if (!self.in_use or layer_idx >= self.layers.len) return null;
        return &self.layers[layer_idx];
    }
};

pub const LoraRegistryConfig = struct {
    /// Preallocated wired pool capacity in bytes (`> 0`, 16 KiB aligned, `<= 256 MiB`).
    /// Defaults to `DEFAULT_LORA_POOL_BYTES` (`4 MiB`).
    pool_bytes: usize = DEFAULT_LORA_POOL_BYTES,
    /// Optional allocator for the startup `page_owner` bitmap (`total_pages` bytes);
    /// defaults to `std.heap.page_allocator` and is freed automatically in `deinit`.
    allocator: ?std.mem.Allocator = null,
    /// `R9` wired budget enforcing that this LoRA pool stays inside startup wired headroom
    /// without ever unpinning base model weights.
    wired_budget: ?budget_mod.Budget = null,
    /// Base model shape against which every hot-loaded `.amlora` is validated.
    base_shape: BaseModelShape,
    /// Optional shared SME2 dispatcher for `sme2_panel64` LoRA acceleration (`C24`).
    sme_dispatcher: ?*sme.SmeDispatcher = null,
    /// Optional target chip profile for `C13`/`C16`/`C24` unified memory bus gating.
    chip_profile: ?chip_profile.ChipProfile = null,
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

/// Preallocated wired-RAM multi-LoRA + DoRA registry (`W19`, `W30`, `ADR-0022`, `C24`).
///
/// Owns ONE fixed `page.WiredRegion` (`<= 256 MiB`, `mlock`ed once at `init`,
/// bounded by `R9` `wired_budget`) and manages hot-load / hot-unload of up to
/// `MAX_ADAPTERS` `.amlora` bundles via a dynamically-sized 16 KiB page-ownership
/// bitmap. When an SME2 dispatcher is attached, also maintains wired
/// `StepLoraScratch` + `f32` accumulation buffers to run
/// `sme.SmeStepLoraAdapter.evalMultiAdapterHoisted` (`<= 4 MiB` L2 cap & `C13`
/// bus gate) with zero hot-path allocations.
pub const LoraRegistry = struct {
    const FREE_PAGE_OWNER: u8 = 0xFF;

    pool: page.WiredRegion,
    total_pages: usize,
    page_owner: []u8,
    page_owner_allocator: std.mem.Allocator,
    used_pages: usize,

    base_shape: BaseModelShape,
    sme_dispatcher: ?*sme.SmeDispatcher,
    chip_profile: ?chip_profile.ChipProfile,
    gpu_bus_saturated: bool,
    measured_coexec_speedup: ?f64,
    swapouts_delta: u64,
    sme_scratch: ?sme.StepLoraScratch,
    delta_f32_region: ?page.WiredRegion,
    slots: [MAX_ADAPTERS]LoadedAdapter,
    active_count: u32,
    total_loads: u64,
    total_unloads: u64,

    // Preallocated zero-alloc evaluation scratch (inside struct — no runtime heap alloc).
    z_f32_scratch: [MAX_LORA_RANK]f32,
    z_u16_scratch: [sme.GROUP_SIZE]u16,

    fn maxOutDim(base: BaseModelShape) usize {
        return @max(
            @as(usize, base.hidden_size),
            @max(@as(usize, base.q_dim), @max(@as(usize, base.kv_dim), @as(usize, base.intermediate_size))),
        );
    }

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

        const total_pages = cfg.pool_bytes / page.APPLE_PAGE_SIZE;
        const po_alloc = cfg.allocator orelse std.heap.page_allocator;
        const page_owner = try po_alloc.alloc(u8, total_pages);
        errdefer po_alloc.free(page_owner);
        @memset(page_owner, FREE_PAGE_OWNER);

        const region = try page.WiredRegion.init(cfg.pool_bytes);
        errdefer region.deinit();

        var sme_scratch: ?sme.StepLoraScratch = null;
        errdefer if (sme_scratch) |*sc| sc.deinit();
        var delta_region: ?page.WiredRegion = null;
        errdefer if (delta_region) |dr| dr.deinit();

        if (cfg.sme_dispatcher != null) {
            sme_scratch = try sme.StepLoraScratch.init(MAX_LORA_BATCH_ROWS, MAX_LORA_RANK, 0, cfg.wired_budget);
            const max_out = maxOutDim(cfg.base_shape);
            const delta_bytes = try page.alignUp(max_out * @sizeOf(f32));
            delta_region = try page.WiredRegion.init(delta_bytes);
        }

        return .{
            .pool = region,
            .total_pages = total_pages,
            .page_owner = page_owner,
            .page_owner_allocator = po_alloc,
            .used_pages = 0,
            .base_shape = cfg.base_shape,
            .sme_dispatcher = cfg.sme_dispatcher,
            .chip_profile = cfg.chip_profile,
            .gpu_bus_saturated = false,
            .measured_coexec_speedup = null,
            .swapouts_delta = 0,
            .sme_scratch = sme_scratch,
            .delta_f32_region = delta_region,
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
        if (self.page_owner.len > 0) {
            self.page_owner_allocator.free(self.page_owner);
            self.page_owner = &.{};
        }
        if (self.sme_scratch) |*sc| {
            sc.deinit();
            self.sme_scratch = null;
        }
        if (self.delta_f32_region) |dr| {
            dr.deinit();
            self.delta_f32_region = null;
        }
        self.pool.deinit();
    }

    /// Returns the 16 KiB-aligned backing wired pool slice for zero-copy Metal wrapping.
    pub fn poolSlice(self: *const LoraRegistry) []align(page.APPLE_PAGE_SIZE) u8 {
        return self.pool.bytes;
    }

    /// Computes byte offset of a pointer living inside `self.pool.bytes`.
    pub fn poolByteOffset(self: *const LoraRegistry, ptr: [*]const u8) usize {
        const base = @intFromPtr(self.pool.bytes.ptr);
        const addr = @intFromPtr(ptr);
        std.debug.assert(addr >= base and addr < base + self.pool.bytes.len);
        return addr - base;
    }

    /// Returns a const pointer to the loaded `LoraLinearPair` at `(slot_id, layer_idx, target)`.
    pub fn pairAt(self: *const LoraRegistry, slot_id: u32, layer_idx: usize, target: TargetProjection) ?*const LoraLinearPair {
        const adp = self.adapterAt(slot_id) orelse return null;
        const la = adp.layerAdapters(layer_idx) orelse return null;
        if (la.projections[@backingInt(target)]) |*pair| return pair;
        return null;
    }

    /// Returns a mutable pointer to the loaded `LoraLinearPair` at `(slot_id, layer_idx, target)`.
    pub fn pairAtMut(self: *LoraRegistry, slot_id: u32, layer_idx: usize, target: TargetProjection) ?*LoraLinearPair {
        if (slot_id >= MAX_ADAPTERS or !self.slots[slot_id].in_use) return null;
        const la = self.slots[slot_id].layerAdaptersMut(layer_idx) orelse return null;
        if (la.projections[@backingInt(target)]) |*pair| return pair;
        return null;
    }

    /// Calibrates a loaded DoRA pair's wired `dora_row_scale_f32` from full base weight rows `[n, k]`.
    pub fn calibrateDoraPairFromBaseRows(
        self: *LoraRegistry,
        slot_id: u32,
        layer_idx: usize,
        target: TargetProjection,
        base_w_f32: []const f32,
    ) LoraError!void {
        const pair = self.pairAtMut(slot_id, layer_idx, target) orelse return error.AdapterNotFound;
        try pair.calibrateDoraPairFromBaseRows(base_w_f32);
    }

    /// Calibrates a loaded DoRA pair's wired `dora_row_scale_f32` from base row norms and optional cross dots.
    pub fn calibrateDoraPairFromBaseRowNorms(
        self: *LoraRegistry,
        slot_id: u32,
        layer_idx: usize,
        target: TargetProjection,
        base_row_norms_f32: []const f32,
        cross_dots_f32: ?[]const f32,
    ) LoraError!void {
        const pair = self.pairAtMut(slot_id, layer_idx, target) orelse return error.AdapterNotFound;
        try pair.calibrateDoraPairFromBaseRowNorms(base_row_norms_f32, cross_dots_f32);
    }

    /// Updates live `C13` / `C16` unified memory bus pressure telemetry used by `evalMultiAdapterHoisted`.
    pub fn setBusContentionState(
        self: *LoraRegistry,
        gpu_bus_saturated: bool,
        measured_coexec_speedup: ?f64,
        swapouts_delta: u64,
    ) void {
        self.gpu_bus_saturated = gpu_bus_saturated;
        self.measured_coexec_speedup = measured_coexec_speedup;
        self.swapouts_delta = swapouts_delta;
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
    pub fn slotTargetsLayerMask(self: *const LoraRegistry, slot_id: u32, layer_idx: usize, mask: u16) bool {
        const adp = self.adapterAt(slot_id) orelse return false;
        const la = adp.layerAdapters(layer_idx) orelse return false;
        return (la.target_mask & mask) != 0;
    }

    /// Returns `true` if ANY slot in `batch_lora_slots` targets `(layer_idx, mask)`.
    pub fn batchTargetsLayerMask(self: *const LoraRegistry, batch_lora_slots: []const u32, layer_idx: usize, mask: u16) bool {
        for (batch_lora_slots) |sid| {
            if (sid == NO_ADAPTER) continue;
            if (self.slotTargetsLayerMask(sid, layer_idx, mask)) return true;
        }
        return false;
    }

    /// Alias for `applySingleAdapterProjectionF16`.
    pub inline fn applyProjectionF16(
        self: *LoraRegistry,
        slot_id: u32,
        layer_idx: usize,
        target: TargetProjection,
        x_f16: []const f16,
        y_f16: []f16,
        num_rows: usize,
    ) (sme.Error || error{BusContentionDeferred})!void {
        return self.applySingleAdapterProjectionF16(slot_id, layer_idx, target, x_f16, y_f16, num_rows);
    }

    /// Evaluates a single adapter's rank-`r` projection delta (plus optional PEFT DoRA
    /// row scaling) across `num_rows` (`m`) rows of `[num_rows, k]` -> `[num_rows, n]`
    /// in-place (`0` heap allocs), routing `.sme2_panel64` adapters (including `q2_affine`
    /// / `LUT4`) through hoisted `sme.SmeStepLoraAdapter.evalMultiAdapterHoisted`
    /// (`<= 4 MiB` L2 cap & `C13` bus gate) when SME2 hoisted runtime scratch is attached.
    pub fn applySingleAdapterProjectionF16(
        self: *LoraRegistry,
        slot_id: u32,
        layer_idx: usize,
        target: TargetProjection,
        x_f16: []const f16,
        y_f16: []f16,
        num_rows: usize,
    ) (sme.Error || error{BusContentionDeferred})!void {
        if (slot_id == NO_ADAPTER or num_rows == 0) return;
        const adp = self.adapterAt(slot_id) orelse return;
        const la = adp.layerAdapters(layer_idx) orelse return;
        const pair = la.projections[@backingInt(target)] orelse return;
        if (pair.scale == 0.0 and !pair.use_dora) return;

        const k: usize = pair.k;
        const n: usize = pair.n;
        const dora_scale: ?[]const f32 = if (pair.use_dora) pair.dora_row_scale_f32 else null;
        if (pair.scale != 0.0 and pair.layout == .sme2_panel64 and pair.sme_adapter != null and
            self.sme_dispatcher != null and self.sme_scratch != null and self.delta_f32_region != null)
        {
            const sme_adp = pair.sme_adapter.?.withAlpha(pair.scale);
            if (!sme_adp.a_weight.dtype.activationIsBf16()) {
                const delta_all: []f32 = @alignCast(std.mem.bytesAsSlice(f32, self.delta_f32_region.?.bytes));
                const cap_rows: usize = @min(MAX_LORA_BATCH_ROWS, delta_all.len / n);
                if (cap_rows > 0) {
                    const adps = [_]sme.SmeStepLoraAdapter{sme_adp};
                    var row_off: usize = 0;
                    while (row_off < num_rows) {
                        const chunk_m: usize = @min(num_rows - row_off, cap_rows);
                        const mn: usize = chunk_m * n;
                        const delta_slice = delta_all[0..mn];
                        @memset(delta_slice, 0.0);
                        const x_u16: []const u16 = @ptrCast(x_f16[row_off * k ..][0 .. chunk_m * k]);
                        try sme.SmeStepLoraAdapter.evalMultiAdapterHoisted(
                            &adps,
                            self.sme_dispatcher.?,
                            &self.sme_scratch.?,
                            x_u16,
                            delta_slice,
                            chunk_m,
                            self.chip_profile,
                            self.gpu_bus_saturated,
                            self.measured_coexec_speedup,
                            self.swapouts_delta,
                            null,
                        );
                        const y_dst = y_f16[row_off * n ..][0..mn];
                        if (dora_scale) |ds| {
                            for (0..chunk_m) |r_i| {
                                const rb = r_i * n;
                                for (0..n) |o| {
                                    y_dst[rb + o] = @floatCast((@as(f32, @floatCast(y_dst[rb + o])) + delta_slice[rb + o]) * ds[o]);
                                }
                            }
                        } else {
                            for (0..mn) |i| {
                                y_dst[i] = @floatCast(@as(f32, @floatCast(y_dst[i])) + delta_slice[i]);
                            }
                        }
                        row_off += chunk_m;
                    }
                    return;
                }
            }
        }

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

    /// Evaluates multiple active LoRA adapters stacked on the SAME `[num_rows, k]`
    /// input in hoisted `sme.SmeStepLoraAdapter.evalMultiAdapterHoisted` waves (`C24`),
    /// enforcing `<= 4 MiB` P-core L2 cap and `C13` unified memory bus gating with zero
    /// hot-path heap allocations.
    pub fn applyMultiAdapterProjectionF16(
        self: *LoraRegistry,
        slot_ids: []const u32,
        layer_idx: usize,
        target: TargetProjection,
        x_f16: []const f16,
        y_f16: []f16,
        num_rows: usize,
    ) (sme.Error || error{BusContentionDeferred})!void {
        if (slot_ids.len == 0 or num_rows == 0) return;

        var k: usize = 0;
        var n: usize = 0;
        var hoisted: [MAX_ADAPTERS]sme.SmeStepLoraAdapter = undefined;
        var hoisted_count: usize = 0;
        var fallback_slots: [MAX_ADAPTERS]u32 = undefined;
        var fallback_count: usize = 0;

        for (slot_ids) |sid| {
            if (sid == NO_ADAPTER) continue;
            const adp = self.adapterAt(sid) orelse continue;
            const la = adp.layerAdapters(layer_idx) orelse continue;
            const pair = la.projections[@backingInt(target)] orelse continue;
            if (pair.scale == 0.0 and !pair.use_dora) continue;
            if (k == 0) {
                k = pair.k;
                n = pair.n;
            }
            if (!pair.use_dora and pair.scale != 0.0 and pair.layout == .sme2_panel64 and pair.sme_adapter != null and
                self.sme_dispatcher != null and self.sme_scratch != null and self.delta_f32_region != null and
                !pair.sme_adapter.?.a_weight.dtype.activationIsBf16() and hoisted_count < MAX_ADAPTERS)
            {
                hoisted[hoisted_count] = pair.sme_adapter.?.withAlpha(pair.scale);
                hoisted_count += 1;
            } else if (fallback_count < MAX_ADAPTERS) {
                fallback_slots[fallback_count] = sid;
                fallback_count += 1;
            }
        }

        if (hoisted_count > 0) {
            const delta_all: []f32 = @alignCast(std.mem.bytesAsSlice(f32, self.delta_f32_region.?.bytes));
            const cap_rows: usize = @min(MAX_LORA_BATCH_ROWS, delta_all.len / n);
            if (cap_rows == 0) return error.BufferTooSmall;
            var row_off: usize = 0;
            while (row_off < num_rows) {
                const chunk_m: usize = @min(num_rows - row_off, cap_rows);
                const mn: usize = chunk_m * n;
                const delta_slice = delta_all[0..mn];
                @memset(delta_slice, 0.0);
                const x_u16: []const u16 = @ptrCast(x_f16[row_off * k ..][0 .. chunk_m * k]);
                try sme.SmeStepLoraAdapter.evalMultiAdapterHoisted(
                    hoisted[0..hoisted_count],
                    self.sme_dispatcher.?,
                    &self.sme_scratch.?,
                    x_u16,
                    delta_slice,
                    chunk_m,
                    self.chip_profile,
                    self.gpu_bus_saturated,
                    self.measured_coexec_speedup,
                    self.swapouts_delta,
                    null,
                );
                const y_dst = y_f16[row_off * n ..][0..mn];
                for (0..mn) |i| {
                    y_dst[i] = @floatCast(@as(f32, @floatCast(y_dst[i])) + delta_slice[i]);
                }
                row_off += chunk_m;
            }
        }

        for (fallback_slots[0..fallback_count]) |sid| {
            try self.applySingleAdapterProjectionF16(sid, layer_idx, target, x_f16, y_f16, num_rows);
        }
    }

    /// Applies per-row heterogeneous LoRA / DoRA deltas across a `C14` continuous-batching
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
        for (batch_lora_slots, 0..) |sid, row| {
            if (sid == NO_ADAPTER) continue;
            const adp = self.adapterAt(sid) orelse continue;
            const la = adp.layerAdapters(layer_idx) orelse continue;
            const pair = la.projections[@backingInt(target)] orelse continue;
            const k: usize = pair.k;
            const n: usize = pair.n;
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

    const LoraTensorRole = enum { lora_a, lora_b, dora_mag };

    const ParsedLoraTensorName = struct {
        layer: u32,
        target: TargetProjection,
        mod_path: []const u8,
        role: LoraTensorRole,
    };

    fn parseLoraTensorName(name: []const u8) ?ParsedLoraTensorName {
        if (!std.mem.startsWith(u8, name, "layers.")) return null;
        const rest = name["layers.".len..];
        const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
        const layer_idx = std.fmt.parseInt(u32, rest[0..dot], 10) catch return null;
        const tail = rest[dot + 1 ..];

        const role_path: struct { role: LoraTensorRole, mod_path: []const u8 } =
            if (std.mem.endsWith(u8, tail, ".lora_a"))
                .{ .role = .lora_a, .mod_path = tail[0 .. tail.len - ".lora_a".len] }
            else if (std.mem.endsWith(u8, tail, ".lora_b"))
                .{ .role = .lora_b, .mod_path = tail[0 .. tail.len - ".lora_b".len] }
            else if (std.mem.endsWith(u8, tail, ".dora_mag"))
                .{ .role = .dora_mag, .mod_path = tail[0 .. tail.len - ".dora_mag".len] }
            else if (std.mem.endsWith(u8, tail, ".lora_mag"))
                .{ .role = .dora_mag, .mod_path = tail[0 .. tail.len - ".lora_mag".len] }
            else if (std.mem.endsWith(u8, tail, ".lora_magnitude_vector"))
                .{ .role = .dora_mag, .mod_path = tail[0 .. tail.len - ".lora_magnitude_vector".len] }
            else
                return null;

        const target = TargetProjection.parseModuleName(role_path.mod_path) orelse return null;
        return .{
            .layer = layer_idx,
            .target = target,
            .mod_path = role_path.mod_path,
            .role = role_path.role,
        };
    }

    fn findPairedEntry(
        parsed: *const amf.Parsed,
        info: ParsedLoraTensorName,
        role: LoraTensorRole,
    ) ?*const amf.TensorEntry {
        var name_buf: [64]u8 = undefined;
        const suffixes: []const []const u8 = switch (role) {
            .lora_a => &.{"lora_a"},
            .lora_b => &.{"lora_b"},
            .dora_mag => &.{ "dora_mag", "lora_mag", "lora_magnitude_vector" },
        };
        // Fast O(1)-ish exact lookup trying source mod_path first, then canonical suffix.
        for (suffixes) |sfx| {
            if (std.fmt.bufPrint(&name_buf, "layers.{d}.{s}.{s}", .{ info.layer, info.mod_path, sfx })) |nm| {
                if (nm.len < 48) {
                    if (parsed.find(nm)) |e| return e;
                }
            } else |_| {}
            if (!std.mem.eql(u8, info.mod_path, info.target.suffix())) {
                if (std.fmt.bufPrint(&name_buf, "layers.{d}.{s}.{s}", .{ info.layer, info.target.suffix(), sfx })) |nm| {
                    if (nm.len < 48) {
                        if (parsed.find(nm)) |e| return e;
                    }
                } else |_| {}
            }
        }
        // Fallback scan matching any alias resolving to the same `(layer, target, role)`.
        for (parsed.entries) |*e| {
            if (amf.isMetaEntry(e)) continue;
            const parsed_e = parseLoraTensorName(e.nameSlice()) orelse continue;
            if (parsed_e.layer == info.layer and parsed_e.target == info.target and parsed_e.role == role) {
                return e;
            }
        }
        return null;
    }

    fn doraWiredBytes(n: usize) LoraError!usize {
        const vec_bytes = std.math.mul(usize, n, @sizeOf(f32)) catch return error.DimensionMismatch;
        const scale_off = std.mem.alignForward(usize, vec_bytes, 16);
        const total = std.math.add(usize, scale_off, vec_bytes) catch return error.DimensionMismatch;
        return try page.alignUp(total);
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

        // Pre-check total wired pages required (including DoRA wired vectors) before mutating pool state.
        var needed_pages: usize = 0;
        var found_any_pair = false;
        for (parsed.entries) |*e| {
            if (amf.isMetaEntry(e)) continue;
            if ((try e.tensorKind()) != .loop_adapter) continue;
            const info = parseLoraTensorName(e.nameSlice()) orelse continue;
            if (info.layer >= self.base_shape.num_layers) return error.LayerOutOfRange;
            if (info.role != .lora_a) continue;
            const b_entry = findPairedEntry(parsed, info, .lora_b) orelse return error.MissingPairedTensor;
            if ((try b_entry.tensorKind()) != .loop_adapter) return error.InvalidTensorKind;
            if (b_entry.rank != 2 or b_entry.shape[0] == 0) return error.DimensionMismatch;

            needed_pages += @intCast(e.blob_bytes / page.APPLE_PAGE_SIZE);
            needed_pages += @intCast(b_entry.blob_bytes / page.APPLE_PAGE_SIZE);
            if (findPairedEntry(parsed, info, .dora_mag)) |mag_entry| {
                if ((try mag_entry.tensorKind()) != .loop_adapter) return error.InvalidTensorKind;
                const d_bytes = try doraWiredBytes(b_entry.shape[0]);
                needed_pages += d_bytes / page.APPLE_PAGE_SIZE;
            }
        }
        if (needed_pages == 0) return error.MissingPairedTensor;
        if (self.used_pages + needed_pages > self.total_pages) return error.PoolOutOfSpace;

        errdefer self.releaseOwnerPages(owner);

        var inferred_rank: u32 = if (cfg_meta) |m| m.rank else 0;
        var effective_scale: f32 = if (cfg_meta) |m| m.scale else 1.0;
        var effective_alpha: f32 = if (cfg_meta) |m| m.lora_alpha else 0.0;
        var any_dora: bool = if (cfg_meta) |m| m.use_dora else false;
        var combined_target_mask: u16 = 0;
        var wired_bytes: usize = 0;

        for (parsed.entries) |*a_entry| {
            if (amf.isMetaEntry(a_entry)) continue;
            if ((try a_entry.tensorKind()) != .loop_adapter) continue;
            const info = parseLoraTensorName(a_entry.nameSlice()) orelse continue;
            if (info.role != .lora_a) continue;

            const b_entry = findPairedEntry(parsed, info, .lora_b) orelse return error.MissingPairedTensor;
            if ((try b_entry.tensorKind()) != .loop_adapter) return error.InvalidTensorKind;

            const a_dt = try a_entry.dataType();
            const b_dt = try b_entry.dataType();
            const a_ok = (a_dt == .f16 or a_dt == .bf16 or a_dt == .q2_affine);
            const b_ok = (b_dt == .f16 or b_dt == .bf16 or b_dt == .q2_affine);
            if (!a_ok or !b_ok) return error.UnsupportedDType;

            const a_lay = try a_entry.tensorLayout();
            const b_lay = try b_entry.tensorLayout();
            if (a_lay != b_lay) return error.InvalidTensorLayout;
            if ((a_dt == .q2_affine or b_dt == .q2_affine) and a_lay != .sme2_panel64) {
                return error.InvalidTensorLayout;
            }
            if (a_dt != .q2_affine and b_dt != .q2_affine and a_dt != b_dt) {
                return error.UnsupportedDType;
            }
            const fdt: quant.FloatDType = if (a_dt == .bf16 or b_dt == .bf16) .bf16 else .f16;
            const b_wdt: quant.WeightDType = switch (b_dt) {
                .f16 => .f16,
                .bf16 => .bf16,
                .q2_affine => .q2_affine,
                else => return error.UnsupportedDType,
            };
            const b_pk = try b_entry.quantParamKind();
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

            const k = a_entry.shape[1];
            const n = b_entry.shape[0];
            if (k == 0 or n == 0) return error.DimensionMismatch;
            if (!self.base_shape.flexible_dims) {
                const expected = self.base_shape.expectedDims(info.target);
                if (n != expected.n or k != expected.k) return error.DimensionMismatch;
            }

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

            var pair_use_dora = false;
            var dora_mag_slice: ?[]f32 = null;
            var dora_scale_slice: ?[]f32 = null;
            if (findPairedEntry(parsed, info, .dora_mag)) |mag_entry| {
                if ((try mag_entry.tensorKind()) != .loop_adapter) return error.InvalidTensorKind;
                if (mag_entry.rank != 1 or mag_entry.shape[0] != n) return error.DimensionMismatch;
                if ((try mag_entry.tensorLayout()) != .row_major) return error.InvalidTensorLayout;
                const mag_dt = try mag_entry.dataType();
                if (mag_dt != .f32 and mag_dt != .f16 and mag_dt != .bf16) return error.UnsupportedDType;

                const n_us: usize = n;
                const vec_bytes = n_us * @sizeOf(f32);
                const scale_off = std.mem.alignForward(usize, vec_bytes, 16);
                const d_bytes = try doraWiredBytes(n_us);
                const d_dst = try self.allocPoolPages(owner, d_bytes / page.APPLE_PAGE_SIZE);
                wired_bytes += d_dst.len;

                const mag_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, d_dst[0..vec_bytes]));
                const scale_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, d_dst[scale_off .. scale_off + vec_bytes]));

                if (mag_dt == .f32) {
                    if (mag_entry.blob_bytes > d_dst.len) return error.BufferTooSmall;
                    try source.readBlobInto(mag_entry, d_dst);
                } else {
                    const tmp_len: usize = @intCast(mag_entry.blob_bytes);
                    const tmp = try allocator.alignedAlloc(u8, .fromByteUnits(page.APPLE_PAGE_SIZE), tmp_len);
                    defer allocator.free(tmp);
                    try source.readBlobInto(mag_entry, tmp);
                    const src_u16: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, tmp[0 .. n_us * 2]));
                    const mag_fdt: quant.FloatDType = if (mag_dt == .bf16) .bf16 else .f16;
                    for (0..n_us) |i| {
                        mag_f32[i] = quant.floatBitsToF32(mag_fdt, src_u16[i]);
                    }
                }
                for (mag_f32) |mv| {
                    if (!std.math.isFinite(mv)) return error.InvalidAdapterConfig;
                }
                @memset(scale_f32, 1.0);
                pair_use_dora = true;
                any_dora = true;
                dora_mag_slice = mag_f32;
                dora_scale_slice = scale_f32;
            }

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
                .weight_dtype = b_wdt,
                .param_kind = b_pk,
                .a_blob = a_dst,
                .b_blob = b_dst,
                .sme_adapter = sme_adp,
                .use_dora = pair_use_dora,
                .dora_mag_f32 = dora_mag_slice,
                .dora_row_scale_f32 = dora_scale_slice,
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
            .use_dora = any_dora,
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
    b_weight_dtype: ?quant.WeightDType = null,
    b_lut4: bool = false,
    a_f32: []const f32,
    b_f32: []const f32,
    /// Optional 1D `[n]` DoRA magnitude vector (`W30`).
    dora_mag_f32: ?[]const f32 = null,
    dora_mag_dtype: amf.DType = .f32,
    /// Tensor suffix for the DoRA magnitude vector (canonical `"dora_mag"`, or alias `"lora_mag"`).
    dora_mag_suffix: []const u8 = "dora_mag",
    /// Optional architecture-native module suffix override (e.g. `"attn.wq_du"`, `"attn.gate_attn"`).
    custom_suffix: ?[]const u8 = null,
};

/// Builds a valid 16 KiB-aligned `.amlora` (`.amf` v0) byte image embedding
/// `__meta.adapter_config_json` and `TensorKind.loop_adapter` `lora_a`/`lora_b`
/// (and optional `dora_mag`) blobs (supports both `row_major` and `sme2_panel64`).
pub fn buildSyntheticAmloraBytes(
    allocator: std.mem.Allocator,
    adapter_id: []const u8,
    rank: u32,
    lora_alpha: f32,
    specs: []const SyntheticLoraSpec,
) ![]u8 {
    var any_dora = false;
    var extra_entries: usize = 0;
    var tm_mask: u16 = 0;
    for (specs) |sp| {
        tm_mask |= sp.target.bit();
        if (sp.dora_mag_f32 != null) {
            any_dora = true;
            extra_entries += 1;
        }
    }
    var tm_buf: [512]u8 = undefined;
    var tm_pos: usize = 0;
    const all_targets = [_]TargetProjection{
        .q_proj,    .k_proj,    .v_proj,       .o_proj,
        .gate_proj, .up_proj,   .down_proj,    .gate_attn,
        .wr_du,     .qkv_proj,  .gate_up_proj, .q_a_proj,
        .q_b_proj,  .kv_a_proj, .kv_b_proj,    .eh_proj,
    };
    const emit_mask: u16 = if (tm_mask != 0) tm_mask else (TargetProjection.q_proj.bit() |
        TargetProjection.k_proj.bit() | TargetProjection.v_proj.bit() | TargetProjection.o_proj.bit() |
        TargetProjection.gate_proj.bit() | TargetProjection.up_proj.bit() | TargetProjection.down_proj.bit());
    for (all_targets) |tp| {
        if ((emit_mask & tp.bit()) == 0) continue;
        const part = try std.fmt.bufPrint(tm_buf[tm_pos..], "{s}\"{s}\"", .{
            if (tm_pos == 0) "" else ",",
            tp.label(),
        });
        tm_pos += part.len;
    }

    var cfg_buf: [1024]u8 = undefined;
    const cfg_json = try std.fmt.bufPrint(
        &cfg_buf,
        "{{\"peft_type\":\"LORA\",\"r\":{d},\"lora_alpha\":{d:.6},\"use_dora\":{s},\"target_modules\":[{s}]}}",
        .{ rank, lora_alpha, if (any_dora) "true" else "false", tm_buf[0..tm_pos] },
    );

    const total_entries = 1 + specs.len * 2 + extra_entries;
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

    var entry_idx: usize = 1;
    for (specs) |sp| {
        const amf_dt: amf.DType = if (sp.dtype == .bf16) .bf16 else .f16;
        const pdt: sme.PanelDType = if (sp.dtype == .bf16) .bf16 else .f16;
        const b_wdt: quant.WeightDType = sp.b_weight_dtype orelse if (sp.dtype == .bf16) .bf16 else .f16;
        const b_amf_dt: amf.DType = switch (b_wdt) {
            .f16 => .f16,
            .bf16 => .bf16,
            .q2_affine => .q2_affine,
            else => return error.UnsupportedDType,
        };
        const b_pk: amf.QuantParamKind = if (b_wdt == .q2_affine and sp.b_lut4)
            .lut16_codebook
        else
            .affine_scale_bias;

        const mod_suffix = sp.custom_suffix orelse sp.target.suffix();
        var a_name_buf: [48]u8 = undefined;
        const a_name = try std.fmt.bufPrint(&a_name_buf, "layers.{d}.{s}.lora_a", .{ sp.layer, mod_suffix });
        var b_name_buf: [48]u8 = undefined;
        const b_name = try std.fmt.bufPrint(&b_name_buf, "layers.{d}.{s}.lora_b", .{ sp.layer, mod_suffix });

        var ae = amf.TensorEntry.zeroed(a_name, .loop_adapter, amf_dt);
        ae.rank = 2;
        ae.layer = @intCast(sp.layer);
        ae.shape = .{ sp.rank, sp.k, 0, 0 };
        ae.setLayoutAndParamKind(sp.layout, .affine_scale_bias);

        var be = amf.TensorEntry.zeroed(b_name, .loop_adapter, b_amf_dt);
        be.rank = 2;
        be.layer = @intCast(sp.layer);
        be.setLayoutAndParamKind(sp.layout, b_pk);
        if (b_wdt == .q2_affine) {
            be.quant_bits = 2;
            be.group_size = sme.GROUP_SIZE;
        }

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
                if (b_wdt == .q2_affine) return error.InvalidTensorLayout;
                be.shape = .{ sp.n, sp.rank, 0, 0 };
                break :blk @as(usize, sp.n) * sp.rank * 2;
            },
            .sme2_panel64 => blk: {
                be.shape = .{ sp.n, sme.GROUP_SIZE, 0, 0 };
                const b_pdt: sme.PanelDType = if (b_wdt == .q2_affine) .q2_affine else pdt;
                break :blk try sme.packedBlobBytes(b_pdt, sp.n, sme.GROUP_SIZE);
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
                if (b_wdt == .q2_affine) {
                    const n_us: usize = sp.n;
                    const r_us: usize = sp.rank;
                    const codes = try allocator.alloc(u8, n_us * r_us);
                    defer allocator.free(codes);
                    if (sp.b_lut4) {
                        const panels = sme.panelCount(n_us);
                        const cbs = try allocator.alloc([4]f16, panels);
                        defer allocator.free(cbs);
                        for (0..panels) |p| {
                            const col_start = p * sme.PANEL_COLS;
                            const col_end = @min(col_start + sme.PANEL_COLS, n_us);
                            var lo: f32 = std.math.inf(f32);
                            var hi: f32 = -std.math.inf(f32);
                            for (col_start..col_end) |row| {
                                for (0..r_us) |col| {
                                    const v = sp.b_f32[row * r_us + col];
                                    if (v < lo) lo = v;
                                    if (v > hi) hi = v;
                                }
                            }
                            if (!std.math.isFinite(lo)) {
                                lo = 0.0;
                                hi = 0.0;
                            }
                            const step = if (hi > lo) (hi - lo) / 3.0 else 0.0;
                            cbs[p] = .{
                                @floatCast(lo),
                                @floatCast(lo + step),
                                @floatCast(lo + 2.0 * step),
                                @floatCast(hi),
                            };
                            for (col_start..col_end) |row| {
                                for (0..r_us) |col| {
                                    const v = sp.b_f32[row * r_us + col];
                                    var best_q: u8 = 0;
                                    var best_err: f32 = std.math.inf(f32);
                                    for (0..4) |qi| {
                                        const err = @abs(v - @as(f32, @floatCast(cbs[p][qi])));
                                        if (err < best_err) {
                                            best_err = err;
                                            best_q = @intCast(qi);
                                        }
                                    }
                                    codes[row * r_us + col] = best_q;
                                }
                            }
                        }
                        try sme.packStepLoraBQ2Lut4Panel64(codes, cbs, null, n_us, r_us, b_blob[0..b_data_bytes]);
                    } else {
                        const sbs = try allocator.alloc(quant.ScaleBias, n_us);
                        defer allocator.free(sbs);
                        for (0..n_us) |row| {
                            var lo: f32 = std.math.inf(f32);
                            var hi: f32 = -std.math.inf(f32);
                            for (0..r_us) |col| {
                                const v = sp.b_f32[row * r_us + col];
                                if (v < lo) lo = v;
                                if (v > hi) hi = v;
                            }
                            const scale_f32 = if (hi > lo) (hi - lo) / 3.0 else 1.0;
                            const scale_h: f16 = @floatCast(scale_f32);
                            const bias_h: f16 = @floatCast(lo);
                            const inv_s = 1.0 / @as(f32, @floatCast(scale_h));
                            sbs[row] = .{ .scale = scale_h, .bias = bias_h };
                            for (0..r_us) |col| {
                                const v = sp.b_f32[row * r_us + col];
                                const qf = std.math.clamp(@round((v - @as(f32, @floatCast(bias_h))) * inv_s), 0.0, 3.0);
                                codes[row * r_us + col] = @intFromFloat(qf);
                            }
                        }
                        try sme.packStepLoraBQ2AffinePanel64(codes, sbs, n_us, r_us, b_blob[0..b_data_bytes]);
                    }
                } else {
                    try sme.packStepLoraBDenseFromF32Panel64(pdt, sp.b_f32, sp.n, sp.rank, b_blob[0..b_data_bytes]);
                }
            },
        }
        be.data_bytes = b_data_bytes;
        be.blob_bytes = b_blob_len;
        be.blob_crc32 = std.hash.Crc32.hash(b_blob);

        entries[entry_idx] = ae;
        blobs[entry_idx] = a_blob;
        entry_idx += 1;
        built += 1;
        entries[entry_idx] = be;
        blobs[entry_idx] = b_blob;
        entry_idx += 1;
        built += 1;

        if (sp.dora_mag_f32) |mag_src| {
            var m_name_buf: [48]u8 = undefined;
            const m_name = try std.fmt.bufPrint(&m_name_buf, "layers.{d}.{s}.{s}", .{ sp.layer, mod_suffix, sp.dora_mag_suffix });
            var me = amf.TensorEntry.zeroed(m_name, .loop_adapter, sp.dora_mag_dtype);
            me.rank = 1;
            me.layer = @intCast(sp.layer);
            me.shape = .{ sp.n, 0, 0, 0 };
            me.setLayoutAndParamKind(.row_major, .affine_scale_bias);
            const elem_bytes: usize = switch (sp.dora_mag_dtype) {
                .f32 => 4,
                .f16, .bf16 => 2,
                else => return error.UnsupportedDType,
            };
            const m_data_bytes: usize = @as(usize, sp.n) * elem_bytes;
            const m_blob_len = try page.alignUp(m_data_bytes);
            const m_blob = try allocator.alloc(u8, m_blob_len);
            @memset(m_blob, 0);
            switch (sp.dora_mag_dtype) {
                .f32 => {
                    const dst_f32: []f32 = @alignCast(std.mem.bytesAsSlice(f32, m_blob[0..m_data_bytes]));
                    @memcpy(dst_f32, mag_src[0..sp.n]);
                },
                .f16 => {
                    const dst_u16: []u16 = @alignCast(std.mem.bytesAsSlice(u16, m_blob[0..m_data_bytes]));
                    for (mag_src[0..sp.n], 0..) |v, idx| dst_u16[idx] = quant.f32ToFloatBits(.f16, v);
                },
                .bf16 => {
                    const dst_u16: []u16 = @alignCast(std.mem.bytesAsSlice(u16, m_blob[0..m_data_bytes]));
                    for (mag_src[0..sp.n], 0..) |v, idx| dst_u16[idx] = quant.f32ToFloatBits(.bf16, v);
                },
                else => return error.UnsupportedDType,
            }
            me.data_bytes = m_data_bytes;
            me.blob_bytes = m_blob_len;
            me.blob_crc32 = std.hash.Crc32.hash(m_blob);
            entries[entry_idx] = me;
            blobs[entry_idx] = m_blob;
            entry_idx += 1;
            built += 1;
        }
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

test "C24 LoraRegistry supports Q2 affine & LUT4 Step-LoRA, hoisted multi-adapter waves, and C13 bus gate" {
    const gpa = std.testing.allocator;
    var disp = try sme.SmeDispatcher.init(gpa, .{});
    defer disp.deinit();
    const base_shape: BaseModelShape = .{
        .num_layers = 1,
        .hidden_size = 64,
        .q_dim = 64,
        .kv_dim = 64,
        .intermediate_size = 128,
    };
    const pool_bytes: usize = 512 * 1024;
    const wired_budget = budget_mod.Budget.compute(.{
        .memory_bytes = 64 * 1024 * 1024,
        .user_wire_limit = 32 * 1024 * 1024,
        .global_user_wire_limit = 32 * 1024 * 1024,
        .gpu_working_set_bytes = 32 * 1024 * 1024,
        .configured_cap = null,
        .headroom_bytes = page.APPLE_PAGE_SIZE,
    });
    const m4_prof: chip_profile.ChipProfile = .{
        .generation = .m4,
        .tier = .base,
        .gpu_apple_family = 9,
        .p_cores = 4,
        .e_cores = 6,
        .has_sme2 = true,
        .has_nax = false,
        .spec_bus_gbs = 120.0,
        .bandwidth_source = .apple_spec_table,
        .c13_min_coexec_speedup = 1.05,
        .ane_wavefront_min_tokens_default = 256,
    };
    var reg = try LoraRegistry.init(.{
        .pool_bytes = pool_bytes,
        .wired_budget = wired_budget,
        .base_shape = base_shape,
        .sme_dispatcher = &disp,
        .chip_profile = m4_prof,
    });
    defer reg.deinit(gpa);

    const r: usize = 4;
    const n: usize = 64;
    const k: usize = 64;
    const m: usize = 2;
    const q2_levels = [_]f32{ -0.75, -0.25, 0.25, 0.75 };

    var a_aff: [r * k]f32 = undefined;
    var b_aff: [n * r]f32 = undefined;
    var a_lut: [r * k]f32 = undefined;
    var b_lut: [n * r]f32 = undefined;
    for (&a_aff, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 5)) - 2)) * 0.25;
    for (&a_lut, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 3) % 7)) - 3)) * 0.125;
    for (&b_aff, 0..) |*v, i| v.* = q2_levels[i % 4];
    for (&b_lut, 0..) |*v, i| v.* = q2_levels[(i * 3 + 1) % 4];

    const img_aff = try buildSyntheticAmloraBytes(gpa, "q2-affine", r, 4.0, &.{
        .{
            .layer = 0,
            .target = .q_proj,
            .n = n,
            .k = k,
            .rank = r,
            .layout = .sme2_panel64,
            .dtype = .f16,
            .b_weight_dtype = .q2_affine,
            .b_lut4 = false,
            .a_f32 = &a_aff,
            .b_f32 = &b_aff,
        },
    });
    defer gpa.free(img_aff);

    const img_lut = try buildSyntheticAmloraBytes(gpa, "q2-lut4", r, 8.0, &.{
        .{
            .layer = 0,
            .target = .q_proj,
            .n = n,
            .k = k,
            .rank = r,
            .layout = .sme2_panel64,
            .dtype = .f16,
            .b_weight_dtype = .q2_affine,
            .b_lut4 = true,
            .a_f32 = &a_lut,
            .b_f32 = &b_lut,
        },
    });
    defer gpa.free(img_lut);

    const slot_aff = try reg.loadAdapterFromBytes(gpa, "aff", img_aff);
    const slot_lut = try reg.loadAdapterFromBytes(gpa, "lut", img_lut);

    var x_in: [m * k]f16 = undefined;
    var y_hoisted: [m * n]f16 = undefined;
    var y_base: [m * n]f16 = undefined;
    for (0..m) |row| {
        for (0..k) |j| {
            x_in[row * k + j] = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast((row * 7 + j) % 9)) - 4)) * 0.25);
        }
        for (0..n) |o| {
            const init_val: f16 = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast((row + o) % 5)) - 2)) * 0.5);
            y_hoisted[row * n + o] = init_val;
            y_base[row * n + o] = init_val;
        }
    }

    // 1) Multi-adapter hoisted wave combining Q2 affine (scale=1.0) + Q2 LUT4 (scale=2.0).
    const multi_slots = [_]u32{ slot_aff, slot_lut };
    try reg.applyMultiAdapterProjectionF16(&multi_slots, 0, .q_proj, &x_in, &y_hoisted, m);

    const refs = [_]struct { scale: f32, a: []const f32, b: []const f32 }{
        .{ .scale = 1.0, .a = &a_aff, .b = &b_aff },
        .{ .scale = 2.0, .a = &a_lut, .b = &b_lut },
    };
    for (0..m) |row| {
        for (0..n) |o| {
            var expected: f32 = @floatCast(y_base[row * n + o]);
            for (refs) |rf| {
                for (0..r) |j| {
                    var acc: f32 = 0.0;
                    for (0..k) |t| acc += @as(f32, @floatCast(x_in[row * k + t])) * rf.a[j * k + t];
                    expected += acc * rf.scale * rf.b[o * r + j];
                }
            }
            const got: f32 = @floatCast(y_hoisted[row * n + o]);
            try std.testing.expectApproxEqAbs(expected, got, 0.08);
        }
    }

    // 2) C13 bus contention gate:
    //    - Active wave `<= 4 MiB` stays L2-resident and executes even under saturated GPU bus.
    //    - Active wave `> 4 MiB` defers cleanly (`error.BusContentionDeferred`) under GPU saturation.
    reg.setBusContentionState(true, 1.0, 1);
    try reg.applyMultiAdapterProjectionF16(&multi_slots, 0, .q_proj, &x_in, &y_hoisted, m);

    const large_k: usize = 1024;
    const large_n: usize = 4096;
    const large_r: usize = 64;
    const a_large = try gpa.alloc(f32, large_r * large_k);
    defer gpa.free(a_large);
    const b_large = try gpa.alloc(f32, large_n * large_r);
    defer gpa.free(b_large);
    @memset(a_large, 0.0625);
    @memset(b_large, 0.0625);
    const img_large = try buildSyntheticAmloraBytes(gpa, "large-lora", large_r, 64.0, &.{
        .{ .layer = 0, .target = .q_proj, .n = large_n, .k = large_k, .rank = large_r, .layout = .sme2_panel64, .dtype = .f16, .a_f32 = a_large, .b_f32 = b_large },
    });
    defer gpa.free(img_large);

    var reg_large = try LoraRegistry.init(.{
        .pool_bytes = 2 * 1024 * 1024,
        .wired_budget = wired_budget,
        .base_shape = .{ .num_layers = 1, .hidden_size = large_k, .q_dim = large_n, .kv_dim = large_n, .intermediate_size = large_n },
        .sme_dispatcher = &disp,
        .chip_profile = m4_prof,
    });
    defer reg_large.deinit(gpa);
    const slot_large = try reg_large.loadAdapterFromBytes(gpa, "lg", img_large);
    const x_large = try gpa.alloc(f16, large_k);
    defer gpa.free(x_large);
    const y_large = try gpa.alloc(f16, large_n);
    defer gpa.free(y_large);
    @memset(x_large, 0.5);
    @memset(y_large, 0.0);

    reg_large.setBusContentionState(true, 1.0, 1);
    const heavy_wave = [_]u32{
        slot_large, slot_large, slot_large, slot_large,
        slot_large, slot_large, slot_large, slot_large,
    }; // 8 * 640 KiB = 5.0 MiB > 4 MiB L2 cap
    try std.testing.expectError(
        error.BusContentionDeferred,
        reg_large.applyMultiAdapterProjectionF16(&heavy_wave, 0, .q_proj, x_large, y_large, 1),
    );
    reg_large.setBusContentionState(false, null, 0);
    try reg_large.applyMultiAdapterProjectionF16(&heavy_wave, 0, .q_proj, x_large, y_large, 1);
    reg.setBusContentionState(false, null, 0);
}

test "W30 universal projections, Inkling-2 aliases, 32 MiB R9 pool, and exact DoRA calibration & CPU equivalence" {
    const gpa = std.testing.allocator;

    // 1. TargetProjection 16-bit mask and canonical + architecture-native aliases
    try std.testing.expectEqual(@as(usize, 16), TargetProjection.COUNT);
    try std.testing.expectEqual(@as(?TargetProjection, .q_proj), TargetProjection.parseModuleName("attn.wq_du"));
    try std.testing.expectEqual(@as(?TargetProjection, .q_proj), TargetProjection.parseModuleName("self_attn.q_proj.linear"));
    try std.testing.expectEqual(@as(?TargetProjection, .k_proj), TargetProjection.parseModuleName("attn.wk_dv"));
    try std.testing.expectEqual(@as(?TargetProjection, .v_proj), TargetProjection.parseModuleName("attn.wv_dv"));
    try std.testing.expectEqual(@as(?TargetProjection, .o_proj), TargetProjection.parseModuleName("attn.wo_ud"));
    try std.testing.expectEqual(@as(?TargetProjection, .gate_attn), TargetProjection.parseModuleName("attn.gate_attn"));
    try std.testing.expectEqual(@as(?TargetProjection, .gate_attn), TargetProjection.parseModuleName("self_attn.o_proj.gate_proj"));
    try std.testing.expectEqual(@as(?TargetProjection, .wr_du), TargetProjection.parseModuleName("attn.wr_du"));
    try std.testing.expectEqual(@as(?TargetProjection, .qkv_proj), TargetProjection.parseModuleName("attn.wqkv"));
    try std.testing.expectEqual(@as(?TargetProjection, .gate_proj), TargetProjection.parseModuleName("ffn.w1"));
    try std.testing.expectEqual(@as(?TargetProjection, .up_proj), TargetProjection.parseModuleName("ffn.w3"));
    try std.testing.expectEqual(@as(?TargetProjection, .down_proj), TargetProjection.parseModuleName("ffn.w2"));
    try std.testing.expectEqual(@as(?TargetProjection, .gate_up_proj), TargetProjection.parseModuleName("ffn.w13"));
    try std.testing.expectEqual(@as(?TargetProjection, .kv_a_proj), TargetProjection.parseModuleName("self_attn.kv_a_proj_with_mqa"));
    try std.testing.expectEqual(@as(?TargetProjection, .eh_proj), TargetProjection.parseModuleName("eh_proj"));
    try std.testing.expectEqual(@as(u16, 1 << 15), TargetProjection.eh_proj.bit());
    // M9.1 mimo_v2 & qwen3_5 aliases (q_gate_proj + linear_attn GDN projections)
    try std.testing.expectEqual(@as(?TargetProjection, .gate_attn), TargetProjection.parseModuleName("self_attn.q_gate_proj"));
    try std.testing.expectEqual(@as(?TargetProjection, .gate_attn), TargetProjection.parseModuleName("q_gate_proj"));
    try std.testing.expectEqual(@as(?TargetProjection, .q_proj), TargetProjection.parseModuleName("linear_attn.q_proj"));
    try std.testing.expectEqual(@as(?TargetProjection, .k_proj), TargetProjection.parseModuleName("linear_attn.k_proj"));
    try std.testing.expectEqual(@as(?TargetProjection, .v_proj), TargetProjection.parseModuleName("linear_attn.v_proj"));
    try std.testing.expectEqual(@as(?TargetProjection, .gate_attn), TargetProjection.parseModuleName("linear_attn.g_proj"));
    try std.testing.expectEqual(@as(?TargetProjection, .o_proj), TargetProjection.parseModuleName("linear_attn.o_proj"));
    try std.testing.expectEqual(@as(?TargetProjection, .o_proj), TargetProjection.parseModuleName("linear_attn.out_proj"));
    try std.testing.expectEqual(@as(?TargetProjection, .qkv_proj), TargetProjection.parseModuleName("linear_attn.in_proj_qkv"));

    // 2. parseAdapterConfigJson parses use_dora and widened u16 target_mask
    const cfg_meta = try parseAdapterConfigJson(
        gpa,
        "{\"r\":64,\"lora_alpha\":128,\"use_dora\":true,\"target_modules\":[\"attn.wq_du\",\"attn.gate_attn\",\"attn.wr_du\",\"ffn.w13\",\"eh_proj\"]}",
    );
    try std.testing.expectEqual(@as(u32, 64), cfg_meta.rank);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), cfg_meta.scale, 1e-6);
    try std.testing.expect(cfg_meta.use_dora);
    const expected_mask: u16 = TargetProjection.q_proj.bit() |
        TargetProjection.gate_attn.bit() |
        TargetProjection.wr_du.bit() |
        TargetProjection.gate_up_proj.bit() |
        TargetProjection.eh_proj.bit();
    try std.testing.expectEqual(expected_mask, cfg_meta.target_mask);

    // 3. R9 wired budget enforcement + > 4 MiB (32 MiB) wired pool allocation & reclamation
    const tight_budget = budget_mod.Budget.compute(.{
        .memory_bytes = 32 * 1024 * 1024,
        .user_wire_limit = 8 * 1024 * 1024,
        .global_user_wire_limit = 8 * 1024 * 1024,
        .gpu_working_set_bytes = 8 * 1024 * 1024,
        .configured_cap = null,
        .headroom_bytes = page.APPLE_PAGE_SIZE,
    });
    const flex_shape: BaseModelShape = .{
        .num_layers = 4,
        .hidden_size = 64,
        .flexible_dims = true,
    };
    try std.testing.expectError(error.BudgetExceeded, LoraRegistry.init(.{
        .pool_bytes = 32 * 1024 * 1024,
        .allocator = gpa,
        .wired_budget = tight_budget,
        .base_shape = flex_shape,
    }));

    const roomy_budget = budget_mod.Budget.compute(.{
        .memory_bytes = 256 * 1024 * 1024,
        .user_wire_limit = 128 * 1024 * 1024,
        .global_user_wire_limit = 128 * 1024 * 1024,
        .gpu_working_set_bytes = 128 * 1024 * 1024,
        .configured_cap = null,
        .headroom_bytes = page.APPLE_PAGE_SIZE,
    });
    var reg = try LoraRegistry.init(.{
        .pool_bytes = 32 * 1024 * 1024,
        .allocator = gpa,
        .wired_budget = roomy_budget,
        .base_shape = flex_shape,
    });
    defer reg.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 32 * 1024 * 1024), reg.stats().pool_capacity_bytes);

    // 4. Load Inkling-2 native suffixes + DoRA (.dora_mag f32, .lora_mag bf16, .lora_magnitude_vector f16)
    //    at r=16 & r=64 across heterogeneous per-layer dims (`flexible_dims = true`).
    const k: usize = 64;
    const n_ga: usize = 96;
    const r: usize = 16;

    const w0 = try gpa.alloc(f32, n_ga * k);
    defer gpa.free(w0);
    const a_rm = try gpa.alloc(f32, r * k);
    defer gpa.free(a_rm);
    const b_rm = try gpa.alloc(f32, n_ga * r);
    defer gpa.free(b_rm);
    const mag = try gpa.alloc(f32, n_ga);
    defer gpa.free(mag);
    for (w0, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 7 + 3) % 13)) - 6)) * 0.125;
    for (a_rm, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 5 + 1) % 11)) - 5)) * 0.0625;
    for (b_rm, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 3 + 2) % 9)) - 4)) * 0.0625;
    for (mag, 0..) |*v, i| v.* = 0.8 + @as(f32, @floatFromInt(i % 7)) * 0.15;

    const img_dora = try buildSyntheticAmloraBytes(gpa, "inkling-dora", r, 32.0, &.{
        .{
            .layer = 0,
            .target = .gate_attn,
            .n = n_ga,
            .k = k,
            .rank = r,
            .layout = .row_major,
            .dtype = .f16,
            .a_f32 = a_rm,
            .b_f32 = b_rm,
            .dora_mag_f32 = mag,
            .dora_mag_dtype = .f32,
            .dora_mag_suffix = "dora_mag",
            .custom_suffix = "attn.gate_attn",
        },
        .{
            .layer = 1,
            .target = .q_proj,
            .n = n_ga,
            .k = k,
            .rank = r,
            .layout = .sme2_panel64,
            .dtype = .bf16,
            .a_f32 = a_rm,
            .b_f32 = b_rm,
            .dora_mag_f32 = mag,
            .dora_mag_dtype = .bf16,
            .dora_mag_suffix = "lora_mag",
            .custom_suffix = "attn.wq_du",
        },
        .{
            .layer = 2,
            .target = .o_proj,
            .n = n_ga,
            .k = k,
            .rank = r,
            .layout = .row_major,
            .dtype = .f16,
            .a_f32 = a_rm,
            .b_f32 = b_rm,
            .dora_mag_f32 = mag,
            .dora_mag_dtype = .f16,
            .dora_mag_suffix = "lora_magnitude_vector",
            .custom_suffix = "attn.wo_ud",
        },
    });
    defer gpa.free(img_dora);

    const slot = try reg.loadAdapterFromBytes(gpa, "ink-dora", img_dora);
    const adp = reg.adapterAt(slot).?;
    try std.testing.expect(adp.use_dora);
    try std.testing.expect(reg.slotTargetsLayerMask(slot, 0, TargetProjection.gate_attn.bit()));
    try std.testing.expect(reg.slotTargetsLayerMask(slot, 1, TargetProjection.q_proj.bit()));
    try std.testing.expect(reg.slotTargetsLayerMask(slot, 2, TargetProjection.o_proj.bit()));

    // Before calibration, dora_row_scale_f32 defaults to 1.0.
    const pair0 = reg.pairAt(slot, 0, .gate_attn).?;
    try std.testing.expect(pair0.use_dora);
    for (pair0.dora_row_scale_f32.?) |s| try std.testing.expectEqual(@as(f32, 1.0), s);

    // Calibrate layer 0 via BaseRows and layer 2 via BaseRowNorms + cross_dots -> must agree closely!
    try reg.calibrateDoraPairFromBaseRows(slot, 0, .gate_attn, w0);
    try reg.calibrateDoraPairFromBaseRows(slot, 1, .q_proj, w0);

    var w0_norms: [n_ga]f32 = undefined;
    var cross_dots: [n_ga]f32 = undefined;
    const scale: f32 = 32.0 / @as(f32, @floatFromInt(r)); // 2.0
    for (0..n_ga) |i| {
        var wn_sq: f64 = 0.0;
        var cd: f64 = 0.0;
        for (0..k) |t| {
            const wt: f64 = w0[i * k + t];
            wn_sq += wt * wt;
            var ba: f64 = 0.0;
            for (0..r) |j| {
                const av = quant.floatBitsToF32(.f16, quant.f32ToFloatBits(.f16, a_rm[j * k + t]));
                const bv = quant.floatBitsToF32(.f16, quant.f32ToFloatBits(.f16, b_rm[i * r + j]));
                ba += @as(f64, bv) * @as(f64, av);
            }
            cd += wt * @as(f64, scale) * ba;
        }
        w0_norms[i] = @floatCast(@sqrt(wn_sq));
        cross_dots[i] = @floatCast(cd);
    }
    try reg.calibrateDoraPairFromBaseRowNorms(slot, 2, .o_proj, &w0_norms, &cross_dots);

    // Verify CPU applyProjectionF16 matches merged (m / ||W0 + s BA||) * (W0 + s BA) @ x!
    var x_f16: [k]f16 = undefined;
    var y0_f16: [n_ga]f16 = undefined;
    var y1_f16: [n_ga]f16 = undefined;
    for (0..k) |t| x_f16[t] = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast((t * 3) % 9)) - 4)) * 0.25);
    for (0..n_ga) |i| {
        var base_y: f32 = 0.0;
        for (0..k) |t| base_y += w0[i * k + t] * @as(f32, @floatCast(x_f16[t]));
        y0_f16[i] = @floatCast(base_y);
        y1_f16[i] = @floatCast(base_y);
    }

    try reg.applyProjectionF16(slot, 0, .gate_attn, &x_f16, &y0_f16, 1);
    try reg.applyProjectionF16(slot, 1, .q_proj, &x_f16, &y1_f16, 1);

    for (0..n_ga) |i| {
        // Exact merged DoRA reference for layer 0 (f16 weights, f32 mag)
        var norm_sq: f64 = 0.0;
        var dot_val: f64 = 0.0;
        for (0..k) |t| {
            var ba: f64 = 0.0;
            for (0..r) |j| {
                const av = quant.floatBitsToF32(.f16, quant.f32ToFloatBits(.f16, a_rm[j * k + t]));
                const bv = quant.floatBitsToF32(.f16, quant.f32ToFloatBits(.f16, b_rm[i * r + j]));
                ba += @as(f64, bv) * @as(f64, av);
            }
            const eff = @as(f64, w0[i * k + t]) + @as(f64, scale) * ba;
            norm_sq += eff * eff;
            dot_val += eff * @as(f64, @floatCast(x_f16[t]));
        }
        const expected0: f32 = @floatCast((@as(f64, mag[i]) / @sqrt(norm_sq + 1.0e-12)) * dot_val);
        try std.testing.expectApproxEqAbs(expected0, @as(f32, @floatCast(y0_f16[i])), 0.03);

        // Layer 1 (sme2_panel64 bf16 weights, bf16 mag)
        var norm_sq1: f64 = 0.0;
        var dot_val1: f64 = 0.0;
        for (0..k) |t| {
            var ba: f64 = 0.0;
            for (0..r) |j| {
                const av = quant.floatBitsToF32(.bf16, quant.f32ToFloatBits(.bf16, a_rm[j * k + t]));
                const bv = quant.floatBitsToF32(.bf16, quant.f32ToFloatBits(.bf16, b_rm[i * r + j]));
                ba += @as(f64, bv) * @as(f64, av);
            }
            const eff = @as(f64, w0[i * k + t]) + @as(f64, scale) * ba;
            norm_sq1 += eff * eff;
            dot_val1 += eff * @as(f64, @floatCast(x_f16[t]));
        }
        const mag_bf16 = quant.floatBitsToF32(.bf16, quant.f32ToFloatBits(.bf16, mag[i]));
        const expected1: f32 = @floatCast((@as(f64, mag_bf16) / @sqrt(norm_sq1 + 1.0e-12)) * dot_val1);
        try std.testing.expectApproxEqAbs(expected1, @as(f32, @floatCast(y1_f16[i])), 0.04);

        // Layer 2 (calibrated via BaseRowNorms + cross_dots) must match layer 0 scale (up to f16 mag rounding)
        const s0 = pair0.dora_row_scale_f32.?[i];
        const s2 = reg.pairAt(slot, 2, .o_proj).?.dora_row_scale_f32.?[i];
        try std.testing.expectApproxEqAbs(s0, s2, 0.01);
    }

    // 5. Unload reclaims all wired pages including DoRA wired pages!
    try std.testing.expect(reg.stats().pool_used_bytes > 0);
    try reg.unloadAdapter(gpa, "ink-dora");
    try std.testing.expectEqual(@as(usize, 0), reg.stats().pool_used_bytes);
    try std.testing.expectEqual(@as(u32, 0), reg.activeAdapterCount());
}
