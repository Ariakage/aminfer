//! Milestone M8.2 (`ADR-0024`, `ADR-0025`) Unified Runtime Executor for
//! `Inkling 2` (`inkling2`, `inkling2_loop`) and `Hy4` (`hy_v4`, `hy_v4_vl`, `hy_v4_flash`).
//!
//! Integrates all Phase-1 M8.2 primitives inside `GraphModel` with zero dynamic
//! heap allocations on the hot path, strict 16 KiB (`APPLE_PAGE_SIZE = 16384`)
//! wired alignment, and `swapouts_delta == 0`:
//! - `G25` `Inkling2Hy4Pipelines` (`src/gpu/inkling2_hy4.zig`): fused
//!   `RMSNorm + sconv4`, single-pass relative-position `1024`-SWA / global GQA,
//!   softcap MoE router/SwiGLU/epilogue, 6-layer `AttnRes`, `IndexCache`
//!   (period-4 `.full`/`.shared` `0`-dispatch reuse), `Hy4-flash` `IndexPool`
//!   (`compress_ratio = 4`), `LatentMoE` (`routed_expert_{down,up}_proj` +
//!   `combine_output_norm` + sigmoid router + `e_score_correction_bias`),
//!   `swiglu_limit = 10.0` + `linear_gate`, `sandwich_norm`, and sparse MLA
//!   `learnable_sink_param` attention.
//! - `C26` `Inkling2LoopController` (`src/cpu/inkling2_ctrl.zig`): `gate_rank = 64`
//!   `sme2_panel64` L2 `SME2` controllers (`0` GPU dispatches) +
//!   `middle_cycle_cross_pass_{kv_}queries`.
//! - `C26` `fp8_e4m3_row` Engram `F_NOCACHE` streaming (`src/storage/ple.zig`):
//!   `r = 0` fetch, `0` NVMe reads on `r >= 1`.
//! - `G23` multi-scheme `NVFP4`/`MXFP8`/`FP8`/`Q2..Q8`/`BF16`/`F16` weights +
//!   `C3`–`C5` `F_NOCACHE` cold-expert NVMe streaming over 2-tensor (`w13`/`gate_up_proj`
//!   + `down_proj`) or 3-tensor (`gate_proj` + `up_proj` + `down_proj`) layouts.
//! - `D14` `M = 8` batched `Core x 3` `DFlare` block/tree verification &
//!   `DFly2` (`block_size = 7`) tree/block verification with bit-exact `T = 0`
//!   greedy parity against serial `stepDecode`.

const std = @import("std");
const metal = @import("../bridge/metal.zig");
const page = @import("../mem/page.zig");
const arena_mod = @import("../mem/arena.zig");
const budget_mod = @import("../mem/budget.zig");
const quant = @import("../quant/spec.zig");
const amf = @import("../format/amf.zig");
const dequant = @import("../gpu/dequant.zig");
const quant_formats = @import("../gpu/quant_formats.zig");
const attention = @import("../gpu/attention.zig");
const moe = @import("../gpu/moe.zig");
const hybrid = @import("../gpu/hybrid.zig");
const ink_gpu = @import("../gpu/inkling2_hy4.zig");
const ink_ctrl = @import("../cpu/inkling2_ctrl.zig");
const sme = @import("../cpu/sme.zig");
const aio = @import("../storage/aio.zig");
const ple_mod = @import("../storage/ple.zig");
const dflare_dfly = @import("../spec/dflare_dfly.zig");
const qwen3 = @import("qwen3.zig");
const lora = @import("lora.zig");
const graph = @import("graph.zig");

const APPLE_PAGE_SIZE = page.APPLE_PAGE_SIZE;

// ============================================================================
// M8.2 Architecture Family, Config & Telemetry
// ============================================================================

pub const M82Family = enum(u8) {
    none = 0,
    inkling2 = 1,
    inkling2_loop = 2,
    hy4 = 3,
    hy4_flash = 4,

    pub fn isInkling2(self: M82Family) bool {
        return self == .inkling2 or self == .inkling2_loop;
    }

    pub fn isHy4(self: M82Family) bool {
        return self == .hy4 or self == .hy4_flash;
    }

    pub fn label(self: M82Family) []const u8 {
        return switch (self) {
            .none => "none",
            .inkling2 => "inkling2",
            .inkling2_loop => "inkling2_loop",
            .hy4 => "hy_v4",
            .hy4_flash => "hy_v4_flash",
        };
    }
};

pub const M82Config = struct {
    // Inkling 2 geometry & operator knobs
    d_rel: u32 = ink_gpu.DEFAULT_REL_DIM,
    rel_extent: u32 = ink_gpu.DEFAULT_REL_EXTENT,
    sconv_kernel_size: u32 = ink_gpu.INKLING2_SCONV_K,
    sliding_window_size: u32 = 1024,
    local_layer_flags: []bool = &.{},
    n_layers_per_attn_res_block: u32 = ink_gpu.INKLING2_DEFAULT_ATTN_RES_BLOCKS,
    attn_res_normalize_values: bool = true,
    qk_l2_clip_threshold: f32 = ink_gpu.INKLING2_DEFAULT_QK_L2_CLIP,
    rel_l2_clip_threshold: f32 = ink_gpu.INKLING2_DEFAULT_REL_L2_CLIP,
    log_scaling_n_floor: u32 = 8192,
    log_scaling_alpha: f32 = 0.1,
    use_value_norm_no_gamma: bool = true,
    mlp_gate_softcap: f32 = 4.0,
    mlp_up_softcap: f32 = 25.0,
    final_logit_softcapping: f32 = 30.0,
    route_scale: f32 = ink_gpu.INKLING2_DEFAULT_ROUTE_SCALE,
    use_gate_bias: bool = true,
    unpadded_vocab_size: u32 = 0,
    loop_gate_rank: u32 = ink_ctrl.GATE_RANK,

    // DFlare (`D14`) target-tap metadata from `mtp_config`
    dflare_num_layers: u8 = 0,
    dflare_block_size: u8 = dflare_dfly.DFLARE_BLOCK_SIZE,
    dflare_markov_rank: u32 = dflare_dfly.DFLARE_MARKOV_RANK,
    dflare_target_layer_count: u8 = 0,
    dflare_target_layer_ids: [dflare_dfly.MAX_TARGET_TAPS]u32 = @splat(0),

    // Hy4 geometry & operator knobs
    enable_ihc: bool = true,
    hc_mult: u32 = 4,
    hc_eps: f32 = 1e-6,
    hc_magnitude: f32 = 2.0,
    gated_mla: bool = true,
    learnable_sink: bool = true,
    swiglu_limit: f32 = ink_gpu.HY4_DEFAULT_SWIGLU_LIMIT,
    routed_scaling_factor: f32 = ink_gpu.HY4_DEFAULT_ROUTED_SCALING_FACTOR,
    moe_router_use_sigmoid: bool = true,
    sandwich_norm: bool = false,
    sliding_window: u32 = 512,
    index_n_heads: u32 = 32,
    index_head_dim: u32 = 128,
    index_topk: u32 = 2048,
    use_compressed_indexer: bool = false,
    compress_ratio: u32 = ink_gpu.HY4_DEFAULT_COMPRESS_RATIO,
    group_topk: u32 = 512,
    candidate_topk: u32 = 2048,
    indexer_modes: []ink_gpu.IndexCacheMode = &.{},
    is_swa_layer: []bool = &.{},
    is_sparse_mlp: []bool = &.{},
    engram_layer_flags: []bool = &.{},
    engram_fp8_embedding: bool = false,
    engram_embed_dim: u32 = 0,

    // DFly2 (`D14`) target KV layer metadata
    dfly_num_layers: u8 = 0,
    dfly_block_size: u8 = dflare_dfly.DFLY2_BLOCK_SIZE,
    dfly_target_kv_count: u8 = 0,
    dfly_target_kv_layers: [dflare_dfly.MAX_LAYERS]u32 = @splat(0),

    pub fn deinit(self: *M82Config, allocator: std.mem.Allocator) void {
        if (self.local_layer_flags.len > 0) {
            allocator.free(self.local_layer_flags);
            self.local_layer_flags = &.{};
        }
        if (self.indexer_modes.len > 0) {
            allocator.free(self.indexer_modes);
            self.indexer_modes = &.{};
        }
        if (self.is_swa_layer.len > 0) {
            allocator.free(self.is_swa_layer);
            self.is_swa_layer = &.{};
        }
        if (self.is_sparse_mlp.len > 0) {
            allocator.free(self.is_sparse_mlp);
            self.is_sparse_mlp = &.{};
        }
        if (self.engram_layer_flags.len > 0) {
            allocator.free(self.engram_layer_flags);
            self.engram_layer_flags = &.{};
        }
    }

    pub fn isM82ConfigJson(json_bytes: []const u8) bool {
        const markers = [_][]const u8{
            "\"inkling2\"",
            "\"inkling2_loop\"",
            "\"inkling2_text\"",
            "\"inkling2_loop_text\"",
            "\"hy_v4\"",
            "\"hy_v4_vl\"",
            "\"hy_v4_flash\"",
            "\"rel_gqa_swa_sconv\"",
            "\"hy4_index_pool_swa\"",
            "\"hy4_index_cache\"",
        };
        for (markers) |m| {
            if (std.mem.indexOf(u8, json_bytes, m) != null) return true;
        }
        return false;
    }

    pub fn detectFamily(root: std.json.ObjectMap, obj: std.json.ObjectMap) M82Family {
        const mt_top = jsonStr(root, "model_type") orelse "";
        const mt_txt = jsonStr(obj, "model_type") orelse "";
        if (std.mem.eql(u8, mt_top, "inkling2_loop") or std.mem.eql(u8, mt_txt, "inkling2_loop") or std.mem.eql(u8, mt_txt, "inkling2_loop_text")) {
            return .inkling2_loop;
        }
        if (std.mem.eql(u8, mt_top, "inkling2") or std.mem.eql(u8, mt_txt, "inkling2") or std.mem.eql(u8, mt_txt, "inkling2_text")) {
            if (obj.get("loop_config")) |lc| {
                if (lc == .object) {
                    const cs = jsonU32(lc.object, "core_start") orelse 0;
                    const ce = jsonU32(lc.object, "core_end") orelse 0;
                    const np = jsonU32(lc.object, "num_passes") orelse 1;
                    if (ce > cs and np > 1) return .inkling2_loop;
                }
            }
            return .inkling2;
        }
        if (std.mem.eql(u8, mt_top, "hy_v4_flash") or std.mem.eql(u8, mt_txt, "hy_v4_flash")) {
            return .hy4_flash;
        }
        if (std.mem.eql(u8, mt_top, "hy_v4") or std.mem.eql(u8, mt_top, "hy_v4_vl") or
            std.mem.eql(u8, mt_txt, "hy_v4") or std.mem.eql(u8, mt_txt, "hy_v4_vl"))
        {
            if (jsonBool(obj, "use_compressed_indexer") orelse false) return .hy4_flash;
            return .hy4;
        }
        if (root.get("operator_axes")) |oa_v| {
            if (oa_v == .object) {
                const att = jsonStr(oa_v.object, "attention") orelse "";
                if (std.mem.eql(u8, att, "rel_gqa_swa_sconv")) {
                    if (root.get("loop_schedule") != null or obj.get("loop_config") != null) return .inkling2_loop;
                    return .inkling2;
                }
                if (std.mem.eql(u8, att, "hy4_index_pool_swa")) return .hy4_flash;
                if (std.mem.eql(u8, att, "hy4_index_cache")) return .hy4;
            }
        }
        return .none;
    }

    pub fn parseFromJson(
        allocator: std.mem.Allocator,
        family: M82Family,
        root: std.json.ObjectMap,
        obj: std.json.ObjectMap,
        num_layers: u32,
        num_experts: u32,
    ) error{ InvalidConfig, OutOfMemory }!M82Config {
        var cfg: M82Config = .{};
        errdefer cfg.deinit(allocator);

        if (family.isInkling2()) {
            cfg.d_rel = jsonU32(obj, "d_rel") orelse ink_gpu.DEFAULT_REL_DIM;
            cfg.rel_extent = jsonU32(obj, "rel_extent") orelse ink_gpu.DEFAULT_REL_EXTENT;
            cfg.sconv_kernel_size = jsonU32(obj, "sconv_kernel_size") orelse ink_gpu.INKLING2_SCONV_K;
            cfg.sliding_window_size = jsonU32(obj, "sliding_window_size") orelse 1024;
            cfg.n_layers_per_attn_res_block = @max(@as(u32, 1), jsonU32(obj, "n_layers_per_attn_res_block") orelse ink_gpu.INKLING2_DEFAULT_ATTN_RES_BLOCKS);
            cfg.attn_res_normalize_values = jsonBool(obj, "attn_res_normalize_values") orelse true;
            cfg.qk_l2_clip_threshold = jsonF32(obj, "qk_l2_clip_threshold") orelse ink_gpu.INKLING2_DEFAULT_QK_L2_CLIP;
            cfg.rel_l2_clip_threshold = jsonF32(obj, "rel_l2_clip_threshold") orelse ink_gpu.INKLING2_DEFAULT_REL_L2_CLIP;
            cfg.log_scaling_n_floor = jsonU32(obj, "log_scaling_n_floor") orelse 8192;
            cfg.log_scaling_alpha = jsonF32(obj, "log_scaling_alpha") orelse 0.1;
            cfg.use_value_norm_no_gamma = jsonBool(obj, "use_value_norm_no_gamma") orelse true;
            cfg.mlp_gate_softcap = jsonF32(obj, "mlp_gate_softcap") orelse 4.0;
            cfg.mlp_up_softcap = jsonF32(obj, "mlp_up_softcap") orelse 25.0;
            cfg.final_logit_softcapping = jsonF32(obj, "final_logit_softcapping") orelse 30.0;
            cfg.route_scale = jsonF32(obj, "route_scale") orelse ink_gpu.INKLING2_DEFAULT_ROUTE_SCALE;
            cfg.use_gate_bias = jsonBool(obj, "use_gate_bias") orelse true;
            cfg.unpadded_vocab_size = jsonU32(obj, "unpadded_vocab_size") orelse 0;

            if (obj.get("loop_config") orelse root.get("loop_schedule")) |lc| {
                if (lc == .object) {
                    cfg.loop_gate_rank = jsonU32(lc.object, "gate_rank") orelse ink_ctrl.GATE_RANK;
                }
            }

            cfg.local_layer_flags = try allocator.alloc(bool, num_layers);
            @memset(cfg.local_layer_flags, false);
            if (obj.get("local_layer_ids") orelse root.get("local_layer_ids")) |lid_val| {
                if (lid_val == .array) {
                    for (lid_val.array.items) |it| {
                        if (jsonItemU32(it)) |idx| {
                            if (idx < num_layers) cfg.local_layer_flags[idx] = true;
                        }
                    }
                }
            }

            if (root.get("mtp_config") orelse obj.get("mtp_config")) |mc| {
                if (mc == .object) {
                    const mco = mc.object;
                    cfg.dflare_num_layers = @intCast(@min(@as(u32, dflare_dfly.MAX_LAYERS), jsonU32(mco, "num_hidden_layers") orelse 0));
                    cfg.dflare_block_size = @intCast(@min(@as(u32, dflare_dfly.MAX_BLOCK_SIZE), @max(@as(u32, 1), jsonU32(mco, "block_size") orelse dflare_dfly.DFLARE_BLOCK_SIZE)));
                    cfg.dflare_markov_rank = jsonU32(mco, "markov_rank") orelse dflare_dfly.DFLARE_MARKOV_RANK;
                    if (mco.get("target_layer_ids")) |tids| {
                        if (tids == .array) {
                            var cnt: u8 = 0;
                            for (tids.array.items) |it| {
                                if (cnt >= dflare_dfly.MAX_TARGET_TAPS) break;
                                if (jsonItemU32(it)) |v| {
                                    cfg.dflare_target_layer_ids[cnt] = v;
                                    cnt += 1;
                                }
                            }
                            cfg.dflare_target_layer_count = cnt;
                        }
                    }
                }
            }
        } else if (family.isHy4()) {
            cfg.enable_ihc = jsonBool(obj, "enable_ihc") orelse true;
            cfg.hc_mult = @max(@as(u32, 1), @min(@as(u32, 8), jsonU32(obj, "hc_mult") orelse 4));
            cfg.hc_eps = jsonF32(obj, "hc_eps") orelse 1e-6;
            cfg.hc_magnitude = jsonF32(obj, "hc_magnitude") orelse 2.0;
            cfg.gated_mla = jsonBool(obj, "gated_mla") orelse true;
            cfg.learnable_sink = jsonBool(obj, "learnable_sink") orelse true;
            cfg.swiglu_limit = jsonF32(obj, "swiglu_limit") orelse ink_gpu.HY4_DEFAULT_SWIGLU_LIMIT;
            cfg.routed_scaling_factor = jsonF32(obj, "routed_scaling_factor") orelse
                (if (family == .hy4_flash) @as(f32, 3.998) else ink_gpu.HY4_DEFAULT_ROUTED_SCALING_FACTOR);
            cfg.moe_router_use_sigmoid = jsonBool(obj, "moe_router_use_sigmoid") orelse true;
            cfg.sandwich_norm = jsonBool(obj, "sandwich_norm") orelse (family == .hy4_flash);
            cfg.sliding_window = jsonU32(obj, "sliding_window") orelse 512;
            cfg.index_n_heads = jsonU32(obj, "index_n_heads") orelse 32;
            cfg.index_head_dim = jsonU32(obj, "index_head_dim") orelse 128;
            cfg.index_topk = jsonU32(obj, "index_topk") orelse 2048;
            cfg.use_compressed_indexer = jsonBool(obj, "use_compressed_indexer") orelse (family == .hy4_flash);
            cfg.compress_ratio = @max(@as(u32, 1), jsonU32(obj, "compressed_indexer_compress_ratio") orelse ink_gpu.HY4_DEFAULT_COMPRESS_RATIO);
            cfg.group_topk = @max(@as(u32, 1), jsonU32(obj, "compressed_indexer_group_topk") orelse 512);
            cfg.candidate_topk = @max(@as(u32, 1), jsonU32(obj, "compressed_indexer_candidate_topk") orelse cfg.index_topk);
            cfg.engram_fp8_embedding = jsonBool(obj, "engram_fp8_embedding") orelse false;

            cfg.indexer_modes = try allocator.alloc(ink_gpu.IndexCacheMode, num_layers);
            @memset(cfg.indexer_modes, .full);
            if (obj.get("indexer_types") orelse root.get("indexer_types")) |it_val| {
                if (it_val == .array) {
                    for (it_val.array.items, 0..) |item, l| {
                        if (l >= num_layers) break;
                        if (item == .string and std.mem.eql(u8, item.string, "shared")) {
                            cfg.indexer_modes[l] = .shared;
                        } else {
                            cfg.indexer_modes[l] = .full;
                        }
                    }
                }
            }

            cfg.is_swa_layer = try allocator.alloc(bool, num_layers);
            @memset(cfg.is_swa_layer, false);
            if (obj.get("layer_types") orelse root.get("layer_types")) |lt_val| {
                if (lt_val == .array) {
                    for (lt_val.array.items, 0..) |item, l| {
                        if (l >= num_layers) break;
                        if (item == .string and std.mem.eql(u8, item.string, "sliding_attention")) {
                            cfg.is_swa_layer[l] = true;
                        }
                    }
                }
            }

            cfg.is_sparse_mlp = try allocator.alloc(bool, num_layers);
            const first_k_dense = jsonU32(obj, "first_k_dense_replace") orelse 1;
            for (0..num_layers) |l| {
                cfg.is_sparse_mlp[l] = (num_experts > 0 and l >= first_k_dense);
            }
            if (obj.get("mlp_layer_types") orelse root.get("mlp_layer_types")) |mlt_val| {
                if (mlt_val == .array) {
                    for (mlt_val.array.items, 0..) |item, l| {
                        if (l >= num_layers) break;
                        if (item == .string) {
                            cfg.is_sparse_mlp[l] = std.mem.eql(u8, item.string, "sparse");
                        }
                    }
                }
            }

            cfg.engram_layer_flags = try allocator.alloc(bool, num_layers);
            @memset(cfg.engram_layer_flags, false);
            if (obj.get("engram_layer_ids") orelse root.get("engram_layer_ids")) |eg_val| {
                if (eg_val == .array) {
                    for (eg_val.array.items) |it| {
                        if (jsonItemU32(it)) |idx| {
                            if (idx < num_layers) cfg.engram_layer_flags[idx] = true;
                        }
                    }
                }
            }

            cfg.dfly_num_layers = @intCast(@min(@as(u32, dflare_dfly.MAX_LAYERS), jsonU32(root, "dfly_num_hidden_layers") orelse jsonU32(obj, "num_nextn_predict_layers") orelse 0));
            cfg.dfly_block_size = @intCast(@min(@as(u32, dflare_dfly.MAX_BLOCK_SIZE), @max(@as(u32, 1), jsonU32(root, "dfly_block_size") orelse dflare_dfly.DFLY2_BLOCK_SIZE)));
            if (root.get("dfly_target_kv_layers") orelse obj.get("dfly_target_kv_layers")) |kvl| {
                if (kvl == .array) {
                    var cnt: u8 = 0;
                    for (kvl.array.items) |it| {
                        if (cnt >= dflare_dfly.MAX_LAYERS) break;
                        if (jsonItemU32(it)) |v| {
                            cfg.dfly_target_kv_layers[cnt] = v;
                            cnt += 1;
                        }
                    }
                    cfg.dfly_target_kv_count = cnt;
                }
            }
        }
        return cfg;
    }
};

pub const M82Stats = struct {
    inkling2_sconv_dispatches: u64 = 0,
    inkling2_rel_gqa_dispatches: u64 = 0,
    inkling2_attn_res_dispatches: u64 = 0,
    inkling2_sme2_ctrl_calls: u64 = 0,
    inkling2_gpu_ctrl_calls: u64 = 0,
    inkling2_indirect_moe_dispatches: u64 = 0,
    inkling2_grouped_moe_dispatches: u64 = 0,
    mid_pass_flushes: u64 = 0,
    inkling2_core3_batched_verify_calls: u64 = 0,
    inkling2_swa_kv_bytes: u64 = 0,
    inkling2_global_kv_bytes: u64 = 0,
    hy4_index_full_dispatches: u64 = 0,
    hy4_index_shared_reuses: u64 = 0,
    hy4_index_pool_compresses: u64 = 0,
    hy4_latent_moe_dispatches: u64 = 0,
    hy4_engram_r0_reads: u64 = 0,
    hy4_engram_r1_plus_reads: u64 = 0,
    cold_expert_fetches: u64 = 0,
    cold_expert_cache_hits: u64 = 0,
    cold_expert_bytes_read: u64 = 0,
    cold_expert_io_ns: u64 = 0,

    pub fn resetCounters(self: *M82Stats) void {
        const swa = self.inkling2_swa_kv_bytes;
        const glob = self.inkling2_global_kv_bytes;
        self.* = .{
            .inkling2_swa_kv_bytes = swa,
            .inkling2_global_kv_bytes = glob,
        };
    }
};

// ============================================================================
// Cold Expert 2-Blob / 3-Blob NVMe Cache (`C3`-`C5`, F_NOCACHE, 16 KiB-aligned)
// ============================================================================

pub const M82ColdExpertMeta = struct {
    layer: u16,
    expert: u16,
    /// Fused `w13`/`gate_up_proj` (or `gate_proj` when `has_separate_up == true`).
    w1_offset: u64,
    w1_bytes: u64,
    w1_params_offset: u64,
    w1_params_bytes: u64 = 0,
    w1_amf_dtype: amf.DType,
    w1_layout: amf.TensorLayout = .row_major,
    w1_param_kind: amf.QuantParamKind = .affine_scale_bias,
    w1_group_size: u16,
    w1_n: u32,
    w1_k: u32,
    /// Optional separate `up_proj` (zero when fused `w13`/`gate_up_proj`).
    w_up_offset: u64 = 0,
    w_up_bytes: u64 = 0,
    w_up_params_offset: u64 = 0,
    w_up_params_bytes: u64 = 0,
    w_up_amf_dtype: amf.DType = .bf16,
    w_up_layout: amf.TensorLayout = .row_major,
    w_up_param_kind: amf.QuantParamKind = .affine_scale_bias,
    w_up_group_size: u16 = 64,
    has_separate_up: bool = false,
    /// `down_proj`
    w2_offset: u64,
    w2_bytes: u64,
    w2_params_offset: u64,
    w2_params_bytes: u64 = 0,
    w2_amf_dtype: amf.DType,
    w2_layout: amf.TensorLayout = .row_major,
    w2_param_kind: amf.QuantParamKind = .affine_scale_bias,
    w2_group_size: u16,
    w2_n: u32,
    w2_k: u32,
};

const ColdSlotState = struct {
    valid: bool = false,
    layer: u16 = 0,
    expert: u16 = 0,
    last_use: u64 = 0,
};

const ColdSlotBindings = struct {
    w1: qwen3.LinearBinding,
    w_up: ?qwen3.LinearBinding,
    w2: qwen3.LinearBinding,
};

pub const ColdPrefetchHandle = struct {
    total_reqs: usize = 0,
    submitted: usize = 0,
    completed: usize = 0,
    miss_count: usize = 0,
    total_read_bytes: u64 = 0,
    t0: std.Io.Timestamp = .{ .nanoseconds = 0 },
};

pub const M82ColdExpertCache = struct {
    amf_file: amf.AmfFile,
    reader: aio.AsyncExpertReader,
    pool_arena: arena_mod.WiredArena,
    pool_buf: metal.Buffer,
    num_slots: u32,
    slot_w1_bytes: usize,
    slot_w_up_bytes: usize,
    slot_w2_bytes: usize,
    slot_stride_bytes: usize,
    slots: []ColdSlotState,
    slot_pinned: []bool,
    req_buf: []aio.ReadRequest,
    miss_slots: []u32,
    miss_metas: []M82ColdExpertMeta,
    clock: u64 = 0,

    pub fn init(
        allocator: std.mem.Allocator,
        device: metal.Device,
        compute: metal.Compute,
        amf_path: [:0]const u8,
        num_slots: u32,
        max_w1_bytes: usize,
        max_w_up_bytes: usize,
        max_w2_bytes: usize,
        budget: budget_mod.Budget,
    ) !M82ColdExpertCache {
        var amf_file = try amf.AmfFile.open(allocator, amf_path);
        errdefer amf_file.close(allocator);

        var reader = try aio.AsyncExpertReader.init(allocator, .{ .max_inflight = 64 });
        errdefer reader.deinit(allocator);

        const w1_al = try page.alignUp(@max(max_w1_bytes, APPLE_PAGE_SIZE));
        const wu_al = if (max_w_up_bytes > 0) try page.alignUp(max_w_up_bytes) else 0;
        const w2_al = try page.alignUp(@max(max_w2_bytes, APPLE_PAGE_SIZE));
        const stride = w1_al + wu_al + w2_al;
        const total = try page.alignUp(@as(usize, num_slots) * stride);

        var pool_arena = try arena_mod.WiredArena.init(total, budget);
        errdefer pool_arena.deinit();
        const pool_buf = try metal.Buffer.wrap(device, pool_arena.bytes());
        errdefer pool_buf.close();
        try compute.makeResident(&.{pool_buf});

        const slots = try allocator.alloc(ColdSlotState, num_slots);
        errdefer allocator.free(slots);
        @memset(slots, .{});

        const slot_pinned = try allocator.alloc(bool, num_slots);
        errdefer allocator.free(slot_pinned);
        @memset(slot_pinned, false);

        const req_buf = try allocator.alloc(aio.ReadRequest, @as(usize, num_slots) * 3);
        errdefer allocator.free(req_buf);

        const miss_slots = try allocator.alloc(u32, num_slots);
        errdefer allocator.free(miss_slots);

        const miss_metas = try allocator.alloc(M82ColdExpertMeta, num_slots);
        errdefer allocator.free(miss_metas);

        return .{
            .amf_file = amf_file,
            .reader = reader,
            .pool_arena = pool_arena,
            .pool_buf = pool_buf,
            .num_slots = num_slots,
            .slot_w1_bytes = w1_al,
            .slot_w_up_bytes = wu_al,
            .slot_w2_bytes = w2_al,
            .slot_stride_bytes = stride,
            .slots = slots,
            .slot_pinned = slot_pinned,
            .req_buf = req_buf,
            .miss_slots = miss_slots,
            .miss_metas = miss_metas,
        };
    }

    pub fn deinit(self: *M82ColdExpertCache, allocator: std.mem.Allocator, compute: metal.Compute) void {
        compute.removeResident(&.{self.pool_buf}) catch |err| {
            std.debug.panic("M82ColdExpertCache removeResident failed: {}", .{err});
        };
        self.pool_buf.close();
        self.pool_arena.deinit();
        self.reader.deinit(allocator);
        self.amf_file.close(allocator);
        allocator.free(self.miss_metas);
        allocator.free(self.miss_slots);
        allocator.free(self.req_buf);
        allocator.free(self.slot_pinned);
        allocator.free(self.slots);
    }

    pub fn totalWiredBytes(self: *const M82ColdExpertCache) u64 {
        return self.pool_arena.bytes().len;
    }

    /// Starts asynchronous `F_NOCACHE` prefetch for a window of `metas` (`metas.len <= num_slots`),
    /// pinning cache hits first so misses never overwrite another expert in the same window.
    /// Zero dynamic heap allocations; pair with `finishPrefetchWindow` before encoding GPU kernels
    /// over `out_bindings`.
    pub fn beginPrefetchWindow(
        self: *M82ColdExpertCache,
        metas: []const M82ColdExpertMeta,
        out_bindings: []ColdSlotBindings,
        stats: *M82Stats,
    ) !ColdPrefetchHandle {
        if (metas.len > self.num_slots or out_bindings.len < metas.len) return error.InvalidArgument;
        self.clock += 1;
        @memset(self.slot_pinned, false);

        var is_hit: [256]bool = @splat(false);
        if (metas.len > is_hit.len) return error.InvalidArgument;

        // Pass 1: resolve and pin all cache hits in this window first.
        for (metas, 0..) |meta, mi| {
            for (self.slots, 0..) |*s, idx| {
                if (s.valid and s.layer == meta.layer and s.expert == meta.expert) {
                    s.last_use = self.clock;
                    self.slot_pinned[idx] = true;
                    is_hit[mi] = true;
                    stats.cold_expert_cache_hits += 1;
                    out_bindings[mi] = self.bindingsForSlot(@intCast(idx), meta);
                    break;
                }
            }
        }

        // Pass 2: assign unpinned LRU victim slots for all misses and queue F_NOCACHE reads.
        var handle: ColdPrefetchHandle = .{
            .t0 = std.Io.Timestamp.now(std.Options.debug_io, .awake),
        };
        const pool_bytes = self.pool_arena.bytes();
        for (metas, 0..) |meta, mi| {
            if (is_hit[mi]) continue;
            var victim_opt: ?usize = null;
            var oldest: u64 = std.math.maxInt(u64);
            for (self.slots, 0..) |s, idx| {
                if (self.slot_pinned[idx]) continue;
                if (!s.valid) {
                    victim_opt = idx;
                    break;
                }
                if (s.last_use < oldest) {
                    oldest = s.last_use;
                    victim_opt = idx;
                }
            }
            const victim = victim_opt orelse return error.InvalidArgument;
            self.slot_pinned[victim] = true;
            self.slots[victim].valid = false;
            self.miss_slots[handle.miss_count] = @intCast(victim);
            self.miss_metas[handle.miss_count] = meta;
            handle.miss_count += 1;

            const base_off: usize = victim * self.slot_stride_bytes;
            const w1_al: usize = page.alignUp(@intCast(meta.w1_bytes)) catch return error.Overflow;
            const w1_dst: []align(APPLE_PAGE_SIZE) u8 = @alignCast(pool_bytes[base_off..][0..w1_al]);
            self.req_buf[handle.total_reqs] = .{
                .fd = self.amf_file.fd,
                .offset = meta.w1_offset,
                .dst = w1_dst,
                .user_data = handle.total_reqs,
            };
            handle.total_reqs += 1;
            handle.total_read_bytes += w1_al;

            if (meta.has_separate_up and meta.w_up_bytes > 0) {
                const wu_al: usize = page.alignUp(@intCast(meta.w_up_bytes)) catch return error.Overflow;
                const wu_dst: []align(APPLE_PAGE_SIZE) u8 = @alignCast(pool_bytes[base_off + self.slot_w1_bytes ..][0..wu_al]);
                self.req_buf[handle.total_reqs] = .{
                    .fd = self.amf_file.fd,
                    .offset = meta.w_up_offset,
                    .dst = wu_dst,
                    .user_data = handle.total_reqs,
                };
                handle.total_reqs += 1;
                handle.total_read_bytes += wu_al;
            }

            const w2_al: usize = page.alignUp(@intCast(meta.w2_bytes)) catch return error.Overflow;
            const w2_dst: []align(APPLE_PAGE_SIZE) u8 = @alignCast(pool_bytes[base_off + self.slot_w1_bytes + self.slot_w_up_bytes ..][0..w2_al]);
            self.req_buf[handle.total_reqs] = .{
                .fd = self.amf_file.fd,
                .offset = meta.w2_offset,
                .dst = w2_dst,
                .user_data = handle.total_reqs,
            };
            handle.total_reqs += 1;
            handle.total_read_bytes += w2_al;

            out_bindings[mi] = self.bindingsForSlot(@intCast(victim), meta);
        }

        // Submit as many reads as fit in the async queue right now so NVMe I/O overlaps GPU work.
        while (handle.submitted < handle.total_reqs and self.reader.availableSlots() > 0) {
            try self.reader.submit(self.req_buf[handle.submitted]);
            handle.submitted += 1;
        }
        return handle;
    }

    pub fn finishPrefetchWindow(
        self: *M82ColdExpertCache,
        handle: *ColdPrefetchHandle,
        stats: *M82Stats,
    ) !void {
        if (handle.miss_count == 0) {
            @memset(self.slot_pinned, false);
            return;
        }
        var comps: [32]aio.Completion = undefined;
        while (handle.completed < handle.total_reqs) {
            while (handle.submitted < handle.total_reqs and self.reader.availableSlots() > 0) {
                try self.reader.submit(self.req_buf[handle.submitted]);
                handle.submitted += 1;
            }
            const want = @min(handle.total_reqs - handle.completed, comps.len);
            const got = try self.reader.wait(comps[0..want], null);
            for (comps[0..got]) |c| {
                _ = try c.check();
            }
            handle.completed += got;
        }
        const t1 = std.Io.Timestamp.now(std.Options.debug_io, .awake);
        const dt_ns: u64 = @intCast(@max(@as(i96, 0), handle.t0.durationTo(t1).toNanoseconds()));

        for (0..handle.miss_count) |mi| {
            const slot_idx = self.miss_slots[mi];
            const meta = self.miss_metas[mi];
            self.slots[slot_idx] = .{
                .valid = true,
                .layer = meta.layer,
                .expert = meta.expert,
                .last_use = self.clock,
            };
        }
        @memset(self.slot_pinned, false);
        stats.cold_expert_fetches += handle.miss_count;
        stats.cold_expert_bytes_read += handle.total_read_bytes;
        stats.cold_expert_io_ns += dt_ns;
    }

    pub fn pinExpert(
        self: *M82ColdExpertCache,
        meta: M82ColdExpertMeta,
        stats: *M82Stats,
    ) !ColdSlotBindings {
        self.clock += 1;
        // 1. Cache lookup
        for (self.slots, 0..) |*s, idx| {
            if (s.valid and s.layer == meta.layer and s.expert == meta.expert) {
                s.last_use = self.clock;
                stats.cold_expert_cache_hits += 1;
                return self.bindingsForSlot(@intCast(idx), meta);
            }
        }
        // 2. Evict LRU slot
        var victim: usize = 0;
        var oldest: u64 = std.math.maxInt(u64);
        for (self.slots, 0..) |s, idx| {
            if (!s.valid) {
                victim = idx;
                break;
            }
            if (s.last_use < oldest) {
                oldest = s.last_use;
                victim = idx;
            }
        }

        const base_off: usize = victim * self.slot_stride_bytes;
        const pool_bytes = self.pool_arena.bytes();
        const t0 = std.Io.Timestamp.now(std.Options.debug_io, .awake);

        // Issue F_NOCACHE direct pread batch for all blobs of this cold expert (16 KiB-aligned).
        var reqs: [3]aio.ReadRequest = undefined;
        var req_cnt: usize = 0;

        const w1_al: usize = page.alignUp(@intCast(meta.w1_bytes)) catch return error.Overflow;
        const w1_dst: []align(APPLE_PAGE_SIZE) u8 = @alignCast(pool_bytes[base_off..][0..w1_al]);
        reqs[req_cnt] = .{
            .fd = self.amf_file.fd,
            .offset = meta.w1_offset,
            .dst = w1_dst,
            .user_data = 0,
        };
        req_cnt += 1;

        var total_read: u64 = w1_al;
        if (meta.has_separate_up and meta.w_up_bytes > 0) {
            const wu_al: usize = page.alignUp(@intCast(meta.w_up_bytes)) catch return error.Overflow;
            const wu_dst: []align(APPLE_PAGE_SIZE) u8 = @alignCast(pool_bytes[base_off + self.slot_w1_bytes ..][0..wu_al]);
            reqs[req_cnt] = .{
                .fd = self.amf_file.fd,
                .offset = meta.w_up_offset,
                .dst = wu_dst,
                .user_data = 1,
            };
            req_cnt += 1;
            total_read += wu_al;
        }
        const w2_al: usize = page.alignUp(@intCast(meta.w2_bytes)) catch return error.Overflow;
        const w2_dst: []align(APPLE_PAGE_SIZE) u8 = @alignCast(pool_bytes[base_off + self.slot_w1_bytes + self.slot_w_up_bytes ..][0..w2_al]);
        reqs[req_cnt] = .{
            .fd = self.amf_file.fd,
            .offset = meta.w2_offset,
            .dst = w2_dst,
            .user_data = 2,
        };
        req_cnt += 1;
        total_read += w2_al;

        for (reqs[0..req_cnt]) |req| {
            try self.reader.submit(req);
        }
        var comps: [3]aio.Completion = undefined;
        var done_cnt: usize = 0;
        while (done_cnt < req_cnt) {
            const got = try self.reader.wait(comps[0 .. req_cnt - done_cnt], null);
            for (comps[0..got]) |c| {
                _ = try c.check();
            }
            done_cnt += got;
        }

        const t1 = std.Io.Timestamp.now(std.Options.debug_io, .awake);
        const dt_ns: u64 = @intCast(@max(@as(i96, 0), t0.durationTo(t1).toNanoseconds()));

        self.slots[victim] = .{
            .valid = true,
            .layer = meta.layer,
            .expert = meta.expert,
            .last_use = self.clock,
        };
        stats.cold_expert_fetches += 1;
        stats.cold_expert_bytes_read += total_read;
        stats.cold_expert_io_ns += dt_ns;
        return self.bindingsForSlot(@intCast(victim), meta);
    }

    fn bindingsForSlot(
        self: *const M82ColdExpertCache,
        slot_idx: u32,
        meta: M82ColdExpertMeta,
    ) ColdSlotBindings {
        const base: u64 = @as(u64, slot_idx) * self.slot_stride_bytes;
        const w1_bind: qwen3.LinearBinding = .{
            .weight = .{ .buffer = self.pool_buf, .offset = base },
            .scale_bias = if (meta.w1_params_offset > 0) .{ .buffer = self.pool_buf, .offset = base + meta.w1_params_offset } else null,
            .n = meta.w1_n,
            .k = meta.w1_k,
            .dtype = amfToWeightDType(meta.w1_amf_dtype),
            .amf_dtype = meta.w1_amf_dtype,
            .group_size = meta.w1_group_size,
            .tensor_scale_per_row = (meta.w1_amf_dtype == .fp8_e4m3_tensor and meta.w1_params_bytes >= @as(u64, meta.w1_n) * 4),
            .alg_bytes = meta.w1_bytes,
            .layout = meta.w1_layout,
            .param_kind = meta.w1_param_kind,
        };
        const w_up_bind: ?qwen3.LinearBinding = if (meta.has_separate_up) .{
            .weight = .{ .buffer = self.pool_buf, .offset = base + self.slot_w1_bytes },
            .scale_bias = if (meta.w_up_params_offset > 0) .{ .buffer = self.pool_buf, .offset = base + self.slot_w1_bytes + meta.w_up_params_offset } else null,
            .n = meta.w1_n,
            .k = meta.w1_k,
            .dtype = amfToWeightDType(meta.w_up_amf_dtype),
            .amf_dtype = meta.w_up_amf_dtype,
            .group_size = meta.w_up_group_size,
            .tensor_scale_per_row = (meta.w_up_amf_dtype == .fp8_e4m3_tensor and meta.w_up_params_bytes >= @as(u64, meta.w1_n) * 4),
            .alg_bytes = meta.w_up_bytes,
            .layout = meta.w_up_layout,
            .param_kind = meta.w_up_param_kind,
        } else null;
        const w2_off = base + self.slot_w1_bytes + self.slot_w_up_bytes;
        const w2_bind: qwen3.LinearBinding = .{
            .weight = .{ .buffer = self.pool_buf, .offset = w2_off },
            .scale_bias = if (meta.w2_params_offset > 0) .{ .buffer = self.pool_buf, .offset = w2_off + meta.w2_params_offset } else null,
            .n = meta.w2_n,
            .k = meta.w2_k,
            .dtype = amfToWeightDType(meta.w2_amf_dtype),
            .amf_dtype = meta.w2_amf_dtype,
            .group_size = meta.w2_group_size,
            .tensor_scale_per_row = (meta.w2_amf_dtype == .fp8_e4m3_tensor and meta.w2_params_bytes >= @as(u64, meta.w2_n) * 4),
            .alg_bytes = meta.w2_bytes,
            .layout = meta.w2_layout,
            .param_kind = meta.w2_param_kind,
        };
        return .{ .w1 = w1_bind, .w_up = w_up_bind, .w2 = w2_bind };
    }
};

// ============================================================================
// Expert Weight Representation (Hot Wired or Cold NVMe, Fused or Split)
// ============================================================================

const M82ExpertWeight = union(enum) {
    hot: struct {
        w1: qwen3.LinearBinding, // fused [2*inter, in] or gate [inter, in]
        w_up: ?qwen3.LinearBinding = null,
        w2: qwen3.LinearBinding, // down [out, inter]
    },
    cold: M82ColdExpertMeta,
};

// ============================================================================
// Inkling 2 Layer & Step Structures
// ============================================================================

const Inkling2GpuBranchCtrl = struct {
    down_weight: metal.BufferBinding,
    forget_weight: metal.BufferBinding,
    gain_weight: metal.BufferBinding,
    weight_dtype: quant.FloatDType,
    state_dim: u32,
    down_in_dim: u32,
    gate_rank: u32,
};

const Inkling2GpuController = struct {
    branches: [ink_ctrl.NUM_BRANCHES]Inkling2GpuBranchCtrl,
};

const Inkling2IndirectMoeMeta = struct {
    weights_base: metal.BufferBinding,
    expert_table: metal.BufferBinding,
    expert_schemes: ?metal.BufferBinding = null,
    is_heterogeneous: bool = false,
    pair_interleaved: bool = true,
    bucket_pair_interleaved: ?[moe.NUM_SCHEME_BUCKETS]bool = null,
    scheme_counts: [moe.NUM_SCHEME_BUCKETS]u32 = .{ 0, 0, 0, 0, 0 },
    scheme_descs: [moe.NUM_SCHEME_BUCKETS]moe.MoeSchemeDesc = .{ .{}, .{}, .{}, .{}, .{} },
};

const Inkling2LayerWeights = struct {
    attn_norm: qwen3.NormBinding,
    q_input_norm: ?qwen3.NormBinding = null,
    k_input_norm: ?qwen3.NormBinding = null,
    v_input_norm: ?qwen3.NormBinding = null,
    mlp_norm: qwen3.NormBinding,

    // Pre-converted f32 static sconv weights `[dim, 4]` inside M82 wired arena
    attn_sconv_w: ?metal.BufferBinding = null,
    mlp_sconv_w: ?metal.BufferBinding = null,
    k_sconv_w: ?metal.BufferBinding = null,
    v_sconv_w: ?metal.BufferBinding = null,
    k_sconv_dyn_proj: ?qwen3.LinearBinding = null,
    v_sconv_dyn_proj: ?qwen3.LinearBinding = null,

    // Attention projections
    wq: qwen3.LinearBinding,
    wk: qwen3.LinearBinding,
    wv: qwen3.LinearBinding,
    wo: qwen3.LinearBinding,
    wr: ?qwen3.LinearBinding = null,
    rel_emb_f16: ?metal.BufferBinding = null, // `[rel_extent, d_rel]` f16
    attn_sink_proj: ?qwen3.LinearBinding = null,
    gate_attn: ?qwen3.LinearBinding = null,
    q_l2_clip_scale: f32 = 1.0,
    rel_l2_clip_scale: f32 = 1.0,
    q_l2_scale_vec: ?[]const f32 = null, // `[q_dim]` per-head per-channel scale on real Inkling-2-Mini
    rel_l2_scale_vec: ?[]const f32 = null, // `[n_q * d_rel]` per-head per-dim scale on real Inkling-2-Mini
    q_l2_scale_vec_gpu: ?metal.BufferBinding = null, // GPU f32 binding for `encodeInkling2AttnPostProj`
    rel_l2_scale_vec_gpu: ?metal.BufferBinding = null, // GPU f32 binding for `encodeInkling2AttnPostProj`

    // AttnRes queries (`f16 [passes, H]` pre-converted in M82 arena)
    attn_res_attn_q: ?metal.BufferBinding = null,
    attn_res_attn_q_passes: u32 = 1,
    attn_res_mlp_q: ?metal.BufferBinding = null,
    attn_res_mlp_q_passes: u32 = 1,

    // Loop Controllers (GPU zero-flush default `G29`, or opt-in C26 L2 SME2 CPU fallback)
    loop_ctrl: ?ink_ctrl.Inkling2LoopController = null,
    gpu_ctrl: ?Inkling2GpuController = null,

    // MoE router + experts (+ optional zero-readback `G29` indirect descriptor when `cold == 0`)
    router: qwen3.LinearBinding,
    router_bias_f32: ?metal.BufferBinding = null, // `[E]` f32
    experts: []M82ExpertWeight,
    indirect_moe: ?Inkling2IndirectMoeMeta = null,
    cold_expert_count: u32 = 0,
    is_local: bool = false,
};

const Inkling2StepState = struct {
    kv_ring: hybrid.HybridKvRing,
    tree_k_off: u64,
    tree_v_off: u64,
    tree_attn_sconv_in_off: u64,
    tree_mlp_sconv_in_off: u64,
    tree_k_sconv_in_off: u64,
    tree_v_sconv_in_off: u64,
    sconv_attn_slot: u32,
    sconv_mlp_slot: u32,
    sconv_k_slot: u32,
    sconv_v_slot: u32,
    is_local: bool,
};

// ============================================================================
// Hy4 Layer Structures
// ============================================================================

const Hy4HcWeights = struct {
    hc_base: []const f32, // `[2 * hc_mult]` or `[hc_mult]`
    hc_fn: []const f32, // `[2 * hc_mult, in_dim]` or `[hc_mult, in_dim]`
    hc_scale: []const f32, // `[2]` or `[1]`
    in_dim: u32,
};

const Hy4IndexerWeights = struct {
    q_proj: qwen3.LinearBinding, // `[idx_h * idx_d, q_lora_rank]`
    k_proj: qwen3.LinearBinding, // `[idx_d, H]`
    k_norm: ?qwen3.NormBinding = null,
    k_norm_bias_f32: ?[]const f32 = null,
    head_w_proj: ?qwen3.LinearBinding = null, // `[idx_h, H]`
    kv_proj: ?qwen3.LinearBinding = null,
    kg_proj: ?qwen3.LinearBinding = null,
    key_norm: ?qwen3.NormBinding = null,
    is_compressed: bool = false,
};

const Hy4EngramLayerWeights = struct {
    key_proj: qwen3.LinearBinding, // `[H, engram_dim]`
    value_proj: qwen3.LinearBinding, // `[H, engram_dim]`
    norm_key: ?qwen3.NormBinding = null,
    norm_query: ?qwen3.NormBinding = null,
    hc_scales: ?[]const f32 = null, // `[hc_mult]`
    pre_gamma: ?[]const f32 = null, // `[H]`
    engram_dim: u32,
};

const Hy4LayerWeights = struct {
    input_layernorm: qwen3.NormBinding,
    post_attention_layernorm: qwen3.NormBinding,
    hc_attn: ?Hy4HcWeights = null,
    hc_mlp: ?Hy4HcWeights = null,

    // MLA weights
    q_a_proj: qwen3.LinearBinding,
    q_a_layernorm: ?qwen3.NormBinding = null,
    q_b_proj: qwen3.LinearBinding,
    kv_a_proj: qwen3.LinearBinding,
    kv_a_layernorm: ?qwen3.NormBinding = null,
    /// Pre-unpacked per-head `W_uk^T [n_q][kv_lora_rank, qk_nope]` and `W_uv [n_q][v_head_dim, kv_lora_rank]` in f16 wired arena.
    w_uk_t_off: u64,
    w_uv_off: u64,
    o_proj: qwen3.LinearBinding,
    linear_gate: ?qwen3.LinearBinding = null, // `[n_q * v_head_dim, H]`
    sink_params_f32: ?metal.BufferBinding = null, // `[n_q]` f32
    post_proj_norm: ?qwen3.NormBinding = null,

    // IndexCache / IndexPool
    indexer_mode: ink_gpu.IndexCacheMode = .full,
    is_swa: bool = false,
    indexer: ?Hy4IndexerWeights = null,
    k_fine_cache_off: u64 = 0, // `[max_seq_len, idx_d]` f16 for `.full` sparse layers

    // MLA KV cache in M82 arena
    kv_latent_off: u64, // `[max_seq_len, kv_lora_rank]` f16
    kv_rope_off: u64, // `[max_seq_len, qk_rope_dim]` f16
    tree_kv_latent_off: u64, // `[max_tree_nodes, kv_lora_rank]` f16
    tree_kv_rope_off: u64, // `[max_tree_nodes, qk_rope_dim]` f16

    // Optional C26 Engram per-layer projections
    engram: ?Hy4EngramLayerWeights = null,

    // FFN: Dense OR Sparse MoE / LatentMoE
    is_sparse_mlp: bool = false,
    dense_gate: ?qwen3.LinearBinding = null,
    dense_up: ?qwen3.LinearBinding = null,
    dense_down: ?qwen3.LinearBinding = null,
    post_fc2_norm: ?qwen3.NormBinding = null,

    router: ?qwen3.LinearBinding = null,
    e_score_bias_f32: ?metal.BufferBinding = null,
    latent_down_proj: ?qwen3.LinearBinding = null, // `[latent_dim, H]`
    latent_up_proj: ?qwen3.LinearBinding = null, // `[H, latent_dim]`
    latent_dim: u32 = 0,
    combine_output_norm: ?qwen3.NormBinding = null,
    shared_gate: ?qwen3.LinearBinding = null,
    shared_up: ?qwen3.LinearBinding = null,
    shared_down: ?qwen3.LinearBinding = null,
    experts: []M82ExpertWeight = &.{},
};

// ============================================================================
// Load Context Passed From `GraphModel.loadImpl`
// ============================================================================

pub const M82LoadContext = struct {
    source: *const graph.GraphModel.AmfSource,
    amf_path: ?[:0]const u8,
    config: *const graph.GraphConfig,
    device: metal.Device,
    library: metal.Library,
    compute: metal.Compute,
    weight_shards: []const graph.WeightShard,
    entry_shard: []const u16,
    entry_offsets: []const u64,
    entry_slices: []const []u8,
    entry_base_ptr: usize,
    max_seq_len: u32,
    max_tree_nodes: u32,
    chunk_tokens: u32,
    wired_budget: budget_mod.Budget,
    cold_slot_count: u32,
    use_sme2_controllers: bool = false,
    enable_router_trace: bool = false,
};

// ============================================================================
// M82Runtime Unified Executor
// ============================================================================

pub const M82Runtime = struct {
    family: M82Family,
    pipes: ink_gpu.Inkling2Hy4Pipelines,
    arena: arena_mod.WiredArena,
    buf: metal.Buffer,
    max_seq_len: u32,
    max_tree_nodes: u32,
    effective_steps: u32,
    use_sme2_controllers: bool = false,
    enable_router_trace: bool = false,
    sconv_pos: u32 = 0,
    stats: M82Stats = .{},

    // Optional cold-expert NVMe streaming cache (`C3`-`C5`)
    cold_cache: ?M82ColdExpertCache = null,
    sme_dispatcher: ?sme.SmeDispatcher = null,

    // Inkling 2 state
    ink_layers: []Inkling2LayerWeights = &.{},
    ink_steps: []Inkling2StepState = &.{},
    ink_moe_ws: ?moe.MoeWorkspace = null,
    ink_embed_norm: ?qwen3.NormBinding = null,
    ink_embed_norm_f32: ?[]const f32 = null,
    ink_attn_res_out_q: ?metal.BufferBinding = null,
    ink_cross_pass_q_f32: ?[]const f32 = null,
    ink_cross_pass_kv_q_f32: ?[]const f32 = null,
    ink_cross_pass_q_gpu: ?metal.BufferBinding = null,
    ink_cross_pass_q_rows: u32 = 0,
    ink_cross_pass_kv_q_gpu: ?metal.BufferBinding = null,
    ink_cross_pass_kv_q_rows: u32 = 0,
    ink_msg_bank_gpu_off: u64 = 0,
    ink_msg_buf_gpu_off: u64 = 0,
    ink_ctrl_z_gpu_off: u64 = 0,
    x_ctrl_q_off: u64 = 0,
    x_ctrl_k_off: u64 = 0,
    x_ctrl_v_off: u64 = 0,
    sconv_h_ring: ?ink_gpu.SconvRingArena = null,
    sconv_kv_ring: ?ink_gpu.SconvRingArena = null,
    attn_res_ring: ?ink_gpu.AttnResRingArena = null,
    /// Preallocated CPU scratch for opt-in `C26` L2 `SME2` controllers (`0` hot-path allocs).
    ctrl_x_f32: []f32 = &.{},
    ctrl_val_f32: []f32 = &.{},
    ctrl_out_f32: []f32 = &.{},
    ctrl_state_attn: []f32 = &.{},
    ctrl_state_k: []f32 = &.{},
    ctrl_state_v: []f32 = &.{},
    ctrl_state_mlp: []f32 = &.{},
    /// Preallocated CPU scratch for opt-in `SME2` `Inkling2Loop` cross-pass message bank & 4 attended messages.
    ink_msg_bank_f32: []f32 = &.{},
    ink_msg_buf_f32: []f32 = &.{},
    ink_prelude_cur_off: u64 = 0,
    ink_inp_sa_off: u64 = 0,
    /// Multi-layer `DFlare` target hidden taps:
    /// - GPU wired `[num_taps, max_tree_nodes, H]` `f16` at `dflare_taps_gpu_off` (`0` mid-pass flushes)
    /// - CPU `f32` mirror `[num_taps][H]` for fallback `dflareTargetTaps()`.
    dflare_taps_gpu_off: u64 = 0,
    dflare_tap_gpu_binds: [dflare_dfly.MAX_TARGET_TAPS]metal.BufferBinding = undefined,
    dflare_tap_storage: []f32 = &.{},
    dflare_tap_slices: [dflare_dfly.MAX_TARGET_TAPS][]const f32 = @splat(&.{}),
    dflare_tap_count: usize = 0,
    /// Full committed-token `DFlare` context tap history `[cap, num_taps, H]` `f16`
    /// stored zero-copy inside `self.arena` (`self.buf`) for direct `D16` `GpuContextHistory`!
    dflare_ctx_hist_off: u64 = 0,
    dflare_ctx_hist_cap: u32 = 0,
    dflare_ctx_hist_len: u32 = 0,
    dflare_ctx_hist_taps: u8 = 0,
    dflare_ctx_hist_hidden: u32 = 0,
    dflare_ctx_hist_epoch: u64 = 0,
    /// Preallocated `W8` loop-folded router trace buffers (`0` hot-path allocs).
    router_trace_topks: []u32 = &.{},
    router_trace_valid: []bool = &.{},
    router_trace_top_k: u32 = 0,
    router_trace_gpu_off: u64 = 0,
    router_trace_gpu_mask: u64 = 0,

    // Hy4 state
    hy4_layers: []Hy4LayerWeights = &.{},
    hy4_hc_head: ?Hy4HcWeights = null,
    hy4_index_pool: ?ink_gpu.IndexPoolBuffer = null,
    hy4_index_n_heads: u32 = 32,
    hy4_index_head_dim: u32 = 128,
    hy4_hc_streams_off: u64 = 0, // `[max_tree_nodes, hc_mult, H]` f16
    hy4_hc_streams_host: []f16 = &.{},
    hy4_hc_tmp_f32: []f32 = &.{},
    hy4_q_a_off: u64 = 0,
    hy4_q_b_off: u64 = 0,
    hy4_q_nope_off: u64 = 0,
    hy4_q_pe_off: u64 = 0,
    hy4_q_abs_off: u64 = 0,
    hy4_kv_a_off: u64 = 0,
    hy4_kv_lat_cur_off: u64 = 0,
    hy4_kv_rope_cur_off: u64 = 0,
    hy4_mla_out_lat_off: u64 = 0,
    hy4_attn_v_off: u64 = 0,
    hy4_lin_gate_off: u64 = 0,
    hy4_q_idx_off: u64 = 0,
    hy4_k_idx_off: u64 = 0,
    hy4_hw_idx_off: u64 = 0,
    hy4_latent_x_off: u64 = 0,
    hy4_latent_exp_out_off: u64 = 0,
    hy4_latent_comb_off: u64 = 0,
    hy4_shared_out_off: u64 = 0,
    hy4_engram_in_f16_off: u64 = 0,
    hy4_engram_k_off: u64 = 0,
    hy4_engram_v_off: u64 = 0,
    /// `DFly2` target KV f32 views for `dfly_target_kv_layers` (zero alloc).
    dfly_kv_k_buf: []f32 = &.{},
    dfly_kv_v_buf: []f32 = &.{},
    dfly_kv_views: [dflare_dfly.MAX_LAYERS]dflare_dfly.TargetKvLayerView = undefined,
    dfly_kv_view_count: usize = 0,
    dfly_kv_dim: u32 = 0,

    // Shared M8.2 GPU scratch offsets (inside `self.buf`)
    x_alt_off: u64 = 0, // `[max_tree_nodes, H]` f16
    x_norm2_off: u64 = 0, // `[max_tree_nodes, H]` f16
    x_norm_k_off: u64 = 0, // `[max_tree_nodes, H]` f16
    x_norm_v_off: u64 = 0, // `[max_tree_nodes, H]` f16
    q_rel_off: u64 = 0, // `[max_tree_nodes, n_q * d_rel]` f16
    sink_f16_off: u64 = 0, // `[max_tree_nodes, n_q]` f16
    sink_f32_off: u64 = 0, // `[max_tree_nodes, n_q]` f32
    gate_attn_off: u64 = 0, // `[max_tree_nodes, q_dim]` f16
    dyn_sconv_w_off: u64 = 0, // `[max_tree_nodes, kv_dim * 4]` f16
    dyn_sconv_proj_off: u64 = 0, // `[max_tree_nodes, 4]` f16
    router_logits_off: u64 = 0, // `[max_tree_nodes, E]` f16
    topk_ids_off: u64 = 0, // `[max_tree_nodes, top_k]` u32
    topk_w_off: u64 = 0, // `[max_tree_nodes, top_k]` f32
    exp_gate_up_off: u64 = 0, // `[max_tree_nodes * 2 * inter]` f16
    exp_up_off: u64 = 0, // `[max_tree_nodes * inter]` f16
    exp_act_off: u64 = 0, // `[max_tree_nodes * inter]` f16
    exp_out_off: u64 = 0, // `[max_tree_nodes * top_k, H]` f16
    exp_gathered_x_off: u64 = 0, // `[max_tree_nodes, H]` f16
    exp_down_tmp_off: u64 = 0, // `[max_tree_nodes, H]` f16
    exp_tok_idx_off: u64 = 0, // `[max_tree_nodes * top_k]` u32
    exp_dst_slot_off: u64 = 0, // `[max_tree_nodes * top_k]` u32
    last_hidden_f32: []f32 = &.{}, // `[max_tree_nodes * H]` f32
    owned_f32_slices: [][]f32 = &.{},
    total_kv_wired_bytes: u64 = 0,
    active_weight_bytes_per_tok: u64 = 0,

    pub fn totalWiredBytes(self: *const M82Runtime) u64 {
        var total: u64 = self.arena.bytes().len + self.total_kv_wired_bytes;
        if (self.cold_cache) |*cc| total += cc.totalWiredBytes();
        for (self.ink_layers) |*l| {
            if (l.loop_ctrl) |*ctrl| total += ctrl.totalWiredBytes();
        }
        return total;
    }

    pub fn activeDecodeBytes(self: *const M82Runtime, cfg: *const graph.GraphConfig, seq_len: u32) u64 {
        var kv_bytes: u64 = 0;
        if (self.family.isInkling2()) {
            const per_tok: u64 = 2 * @as(u64, cfg.num_key_value_heads) * cfg.head_dim * @sizeOf(f16);
            for (self.ink_steps) |st| {
                const vis: u64 = if (st.kv_ring.window_size > 0)
                    @min(@as(u64, seq_len), st.kv_ring.window_size)
                else
                    @as(u64, seq_len);
                kv_bytes += vis * per_tok;
            }
        } else if (self.family.isHy4()) {
            const per_tok: u64 = (@as(u64, cfg.kv_lora_rank) + cfg.qk_rope_head_dim) * @sizeOf(f16);
            kv_bytes = @as(u64, self.hy4_layers.len) * @as(u64, seq_len) * per_tok;
        }
        return self.active_weight_bytes_per_tok + kv_bytes;
    }

    pub fn lastHiddenF32(self: *const M82Runtime, node_idx: usize, h: usize) []const f32 {
        const off = node_idx * h;
        if (off + h > self.last_hidden_f32.len) return &.{};
        return self.last_hidden_f32[off .. off + h];
    }

    pub fn dflareTargetTaps(self: *const M82Runtime) []const []const f32 {
        return self.dflare_tap_slices[0..self.dflare_tap_count];
    }

    pub fn hasDFlareCtxHistory(self: *const M82Runtime) bool {
        return self.dflare_ctx_hist_cap > 0 and self.dflare_ctx_hist_off > 0;
    }

    pub fn dflareGpuTargetTaps(self: *const M82Runtime) ?dflare_dfly.GpuTargetTaps {
        if (self.dflare_tap_count == 0 or self.dflare_taps_gpu_off == 0) return null;
        return .{
            .dtype = .f16,
            .taps = self.dflare_tap_gpu_binds[0..self.dflare_tap_count],
        };
    }

    pub fn dflareGpuContextHistory(self: *const M82Runtime) ?dflare_dfly.GpuContextHistory {
        if (!self.hasDFlareCtxHistory()) return null;
        return .{
            .taps_ring = self.bindAt(self.dflare_ctx_hist_off),
            .dtype = .f16,
            .cap = self.dflare_ctx_hist_cap,
            .len = self.dflare_ctx_hist_len,
            .epoch = self.dflare_ctx_hist_epoch,
        };
    }

    /// Reads one historical committed token's `DFlare` target tap `[H]` into `dst_f32`
    /// with zero heap allocations (used by `DFlareProposer`'s CPU fallback context KV cache).
    pub fn readDFlareCtxTapF32(self: *const M82Runtime, pos: u32, tap_idx: usize, dst_f32: []f32) bool {
        if (!self.hasDFlareCtxHistory()) return false;
        if (pos >= self.dflare_ctx_hist_len) return false;
        if (self.dflare_ctx_hist_len > self.dflare_ctx_hist_cap and pos + self.dflare_ctx_hist_cap < self.dflare_ctx_hist_len) {
            return false;
        }
        const taps: usize = self.dflare_ctx_hist_taps;
        const h: usize = self.dflare_ctx_hist_hidden;
        if (tap_idx >= taps or dst_f32.len < h) return false;
        const slot: usize = @intCast(pos % self.dflare_ctx_hist_cap);
        const total_elems: usize = @as(usize, self.dflare_ctx_hist_cap) * taps * h;
        const hist_f16 = self.sliceF16(self.dflare_ctx_hist_off, total_elems);
        const base: usize = (slot * taps + tap_idx) * h;
        const src = hist_f16[base .. base + h];
        for (src, 0..) |hv, i| {
            dst_f32[i] = @floatCast(hv);
        }
        return true;
    }

    fn recordDFlareCtxTapRow(self: *M82Runtime, pos: u32, node_idx: usize, h: usize) void {
        if (!self.hasDFlareCtxHistory() or self.dflare_taps_gpu_off == 0) return;
        const taps: usize = @min(self.dflare_tap_count, @as(usize, self.dflare_ctx_hist_taps));
        if (taps == 0 or h != self.dflare_ctx_hist_hidden) return;
        const stride: usize = @as(usize, self.max_tree_nodes) * h;
        if ((node_idx + 1) * h > stride) return;
        const slot: usize = @intCast(pos % self.dflare_ctx_hist_cap);
        const total_hist: usize = @as(usize, self.dflare_ctx_hist_cap) * @as(usize, self.dflare_ctx_hist_taps) * h;
        const hist_f16 = self.sliceF16(self.dflare_ctx_hist_off, total_hist);
        const all_taps = self.sliceF16(self.dflare_taps_gpu_off, @as(usize, self.dflare_ctx_hist_taps) * stride);
        for (0..taps) |ti| {
            const src = all_taps[ti * stride + node_idx * h ..][0..h];
            const dst = hist_f16[(slot * @as(usize, self.dflare_ctx_hist_taps) + ti) * h ..][0..h];
            @memcpy(dst, src);
        }
        if (pos + 1 > self.dflare_ctx_hist_len) {
            self.dflare_ctx_hist_len = pos + 1;
        }
    }

    fn syncDFlareActiveRow(self: *M82Runtime, row_idx: usize, h: usize) void {
        if (self.dflare_tap_count == 0 or self.dflare_taps_gpu_off == 0) return;
        const stride: usize = @as(usize, self.max_tree_nodes) * h;
        if ((row_idx + 1) * h > stride) return;
        const row_bytes: u64 = @as(u64, row_idx) * h * @sizeOf(f16);
        const stride_bytes: u64 = @as(u64, stride) * @sizeOf(f16);
        for (0..self.dflare_tap_count) |ti| {
            const gpu_off = self.dflare_taps_gpu_off + @as(u64, ti) * stride_bytes + row_bytes;
            self.dflare_tap_gpu_binds[ti] = self.bindAt(gpu_off);
            if (self.dflare_tap_storage.len >= (ti + 1) * stride) {
                const src_f16 = self.sliceF16(gpu_off, h);
                const dst_f32 = self.dflare_tap_storage[ti * stride ..][0..h];
                f16SliceToF32(src_f16, dst_f32);
                self.dflare_tap_slices[ti] = dst_f32;
            }
        }
    }

    /// Returns the last-step router top-k expert IDs for effective step `eff_step`
    /// (`0 .. self.effective_steps`) when `enable_router_trace` is enabled (`W8`).
    pub fn lastStepRouterTopKForStep(self: *const M82Runtime, eff_step: u32) ?[]const u32 {
        if (!self.enable_router_trace or self.router_trace_top_k == 0) return null;
        if (eff_step >= self.router_trace_valid.len or !self.router_trace_valid[eff_step]) return null;
        const tk: usize = self.router_trace_top_k;
        const base: usize = @as(usize, eff_step) * tk;
        if (base + tk > self.router_trace_topks.len) return null;
        return self.router_trace_topks[base .. base + tk];
    }

    pub fn physicalLayerForStepPublic(self: *const M82Runtime, step_idx: u32, cfg: *const graph.GraphConfig) usize {
        return self.physicalLayerForStep(step_idx, cfg);
    }

    /// Resolves the base linear weight binding at `(layer_idx, target)` for one-time
    /// PEFT DoRA row-scale calibration (`W30`, 0 hot-path cost).
    pub fn baseLinearBindingForLora(
        self: *const M82Runtime,
        layer_idx: usize,
        target: lora.TargetProjection,
    ) ?qwen3.LinearBinding {
        if (self.family.isInkling2()) {
            if (layer_idx >= self.ink_layers.len) return null;
            const ink = &self.ink_layers[layer_idx];
            return switch (target) {
                .q_proj => ink.wq,
                .k_proj => ink.wk,
                .v_proj => ink.wv,
                .o_proj => ink.wo,
                .wr_du => ink.wr,
                .gate_attn => ink.gate_attn,
                else => null,
            };
        }
        if (self.family.isHy4()) {
            if (layer_idx >= self.hy4_layers.len) return null;
            const hy = &self.hy4_layers[layer_idx];
            return switch (target) {
                .q_a_proj => hy.q_a_proj,
                .q_b_proj => hy.q_b_proj,
                .kv_a_proj => hy.kv_a_proj,
                .o_proj => hy.o_proj,
                .gate_attn => hy.linear_gate,
                .gate_proj => hy.dense_gate,
                .up_proj => hy.dense_up,
                .down_proj => hy.dense_down,
                else => null,
            };
        }
        return null;
    }

    pub fn dflyTargetKvViews(self: *const M82Runtime) []const dflare_dfly.TargetKvLayerView {
        return self.dfly_kv_views[0..self.dfly_kv_view_count];
    }

    pub fn resetKvCache(self: *M82Runtime) void {
        self.sconv_pos = 0;
        if (self.sconv_h_ring) |*r| @memset(r.bytes(), 0);
        if (self.sconv_kv_ring) |*r| @memset(r.bytes(), 0);
        if (self.attn_res_ring) |*r| {
            @memset(r.bytes(), 0);
            r.count = 0;
            r.head = 0;
        }
        if (self.hy4_index_pool) |*ip| @memset(ip.bytes(), 0);
        self.dfly_kv_view_count = 0;
        if (self.ctrl_state_attn.len > 0) @memset(self.ctrl_state_attn, 0);
        if (self.ctrl_state_k.len > 0) @memset(self.ctrl_state_k, 0);
        if (self.ctrl_state_v.len > 0) @memset(self.ctrl_state_v, 0);
        if (self.ctrl_state_mlp.len > 0) @memset(self.ctrl_state_mlp, 0);
        self.dflare_ctx_hist_len = 0;
        self.dflare_ctx_hist_epoch +%= 1;
    }

    pub fn deinit(self: *M82Runtime, allocator: std.mem.Allocator, compute: metal.Compute) void {
        for (self.ink_steps) |*st| {
            st.kv_ring.removeResident(compute) catch |err| {
                std.debug.panic("M82Runtime kv_ring.removeResident failed: {}", .{err});
            };
            st.kv_ring.deinit(allocator);
        }
        if (self.ink_steps.len > 0) allocator.free(self.ink_steps);
        for (self.ink_layers) |*l| {
            if (l.loop_ctrl) |*ctrl| ctrl.deinit();
            if (l.experts.len > 0) allocator.free(l.experts);
        }
        if (self.ink_layers.len > 0) allocator.free(self.ink_layers);
        for (self.hy4_layers) |*l| {
            if (l.experts.len > 0) allocator.free(l.experts);
        }
        if (self.hy4_layers.len > 0) allocator.free(self.hy4_layers);
        if (self.cold_cache) |*cc| cc.deinit(allocator, compute);
        if (self.sme_dispatcher) |*sd| sd.deinit();

        compute.removeResident(&.{self.buf}) catch |err| {
            std.debug.panic("M82Runtime removeResident failed: {}", .{err});
        };
        self.buf.close();
        self.arena.deinit();
        self.pipes.close();

        if (self.ctrl_x_f32.len > 0) allocator.free(self.ctrl_x_f32);
        if (self.ctrl_val_f32.len > 0) allocator.free(self.ctrl_val_f32);
        if (self.ctrl_out_f32.len > 0) allocator.free(self.ctrl_out_f32);
        if (self.ctrl_state_attn.len > 0) allocator.free(self.ctrl_state_attn);
        if (self.ctrl_state_k.len > 0) allocator.free(self.ctrl_state_k);
        if (self.ctrl_state_v.len > 0) allocator.free(self.ctrl_state_v);
        if (self.ctrl_state_mlp.len > 0) allocator.free(self.ctrl_state_mlp);
        if (self.ink_msg_bank_f32.len > 0) allocator.free(self.ink_msg_bank_f32);
        if (self.ink_msg_buf_f32.len > 0) allocator.free(self.ink_msg_buf_f32);
        if (self.dflare_tap_storage.len > 0) allocator.free(self.dflare_tap_storage);
        if (self.router_trace_topks.len > 0) allocator.free(self.router_trace_topks);
        if (self.router_trace_valid.len > 0) allocator.free(self.router_trace_valid);
        if (self.hy4_hc_tmp_f32.len > 0) allocator.free(self.hy4_hc_tmp_f32);
        if (self.dfly_kv_k_buf.len > 0) allocator.free(self.dfly_kv_k_buf);
        if (self.dfly_kv_v_buf.len > 0) allocator.free(self.dfly_kv_v_buf);
        if (self.last_hidden_f32.len > 0) allocator.free(self.last_hidden_f32);
        for (self.owned_f32_slices) |s| allocator.free(s);
        if (self.owned_f32_slices.len > 0) allocator.free(self.owned_f32_slices);
        allocator.destroy(self);
    }

    // ------------------------------------------------------------------------
    // Startup Loader (16 KiB-aligned WiredArena, zero hot-path allocations)
    // ------------------------------------------------------------------------

    pub fn load(
        allocator: std.mem.Allocator,
        ctx: M82LoadContext,
    ) !*M82Runtime {
        const cfg = ctx.config;
        const family = cfg.m82_family;
        const h: u32 = cfg.hidden_size;
        const num_layers: u32 = cfg.num_hidden_layers;
        const max_m: u32 = ctx.max_tree_nodes;

        const ls = cfg.axes.loop_schedule;
        const core_start: u32 = if (ls.isRecurrent()) ls.start_layer else 0;
        const core_end: u32 = if (ls.isRecurrent()) (if (ls.end_layer > ls.start_layer) ls.end_layer else num_layers) else 0;
        const num_passes: u32 = if (ls.isRecurrent()) ls.max_loops else 1;
        const eff_steps: u32 = if (ls.isRecurrent())
            num_layers + (core_end - core_start) * (num_passes - 1)
        else
            num_layers;

        var pipes = try ink_gpu.Inkling2Hy4Pipelines.init(ctx.device, ctx.library);
        errdefer pipes.close();

        var f32_pool: std.ArrayListUnmanaged([]f32) = .empty;
        errdefer {
            for (f32_pool.items) |s| allocator.free(s);
            f32_pool.deinit(allocator);
        }

        // Scan for cold experts (2-blob fused or 3-blob split) so we size M82ColdExpertCache.
        var has_cold = false;
        var max_cold_w1: usize = 0;
        var max_cold_w_up: usize = 0;
        var max_cold_w2: usize = 0;
        for (ctx.source.entries()) |*e| {
            if ((try e.tensorKind()) != .cold_expert) continue;
            has_cold = true;
            const nm = e.nameSlice();
            const al = try page.alignUp(@intCast(e.blob_bytes));
            if (std.mem.endsWith(u8, nm, ".w13.weight") or std.mem.endsWith(u8, nm, ".w13") or
                std.mem.endsWith(u8, nm, ".gate_up_proj.weight") or std.mem.endsWith(u8, nm, ".gate_up_proj") or
                std.mem.endsWith(u8, nm, ".gate_proj.weight") or std.mem.endsWith(u8, nm, ".gate_proj"))
            {
                max_cold_w1 = @max(max_cold_w1, al);
            } else if (std.mem.endsWith(u8, nm, ".up_proj.weight") or std.mem.endsWith(u8, nm, ".up_proj")) {
                max_cold_w_up = @max(max_cold_w_up, al);
            } else if (std.mem.endsWith(u8, nm, ".down_proj.weight") or std.mem.endsWith(u8, nm, ".down_proj")) {
                max_cold_w2 = @max(max_cold_w2, al);
            }
        }

        var cold_cache: ?M82ColdExpertCache = null;
        errdefer if (cold_cache) |*cc| cc.deinit(allocator, ctx.compute);
        if (has_cold) {
            const path = ctx.amf_path orelse return error.InvalidConfig;
            const min_slots: u32 = @max(@as(u32, 16), cfg.num_experts_per_tok * 2);
            const est_stride: usize = @max(max_cold_w1, APPLE_PAGE_SIZE) + max_cold_w_up + @max(max_cold_w2, APPLE_PAGE_SIZE);
            // Size default LRU slots for `inkling2_loop` to keep Core 6..17 active experts warm across all 3 passes
            // (up to 104 slots, capped at 512 MiB so 24 GB M4 stays comfortably within live zero-swap RAM!).
            const auto_target: u32 = if (family == .inkling2_loop) blk: {
                const max_by_mem: u32 = @intCast(@max(@as(usize, min_slots), (512 * 1024 * 1024) / @max(est_stride, APPLE_PAGE_SIZE)));
                break :blk @min(@as(u32, 104), max_by_mem);
            } else @max(@as(u32, 32), min_slots);
            const n_slots: u32 = if (ctx.cold_slot_count > 0) @max(ctx.cold_slot_count, min_slots) else @max(auto_target, min_slots);
            cold_cache = try M82ColdExpertCache.init(
                allocator,
                ctx.device,
                ctx.compute,
                path,
                n_slots,
                max_cold_w1,
                max_cold_w_up,
                max_cold_w2,
                ctx.wired_budget,
            );
        }

        // Compute exact wired arena capacity needed for M8.2 scratch, rings, small converted weights & Hy4 KV.
        const arena_bytes = try computeArenaBytes(cfg, ctx, eff_steps);
        var arena = try arena_mod.WiredArena.init(arena_bytes, ctx.wired_budget);
        errdefer arena.deinit();
        const buf = try metal.Buffer.wrap(ctx.device, arena.bytes());
        errdefer buf.close();
        try ctx.compute.makeResident(&.{buf});

        const rt = try allocator.create(M82Runtime);
        errdefer allocator.destroy(rt);
        rt.* = .{
            .family = family,
            .pipes = pipes,
            .arena = arena,
            .buf = buf,
            .max_seq_len = ctx.max_seq_len,
            .max_tree_nodes = max_m,
            .effective_steps = eff_steps,
            .use_sme2_controllers = ctx.use_sme2_controllers,
            .enable_router_trace = ctx.enable_router_trace,
            .cold_cache = cold_cache,
        };
        errdefer {
            // Avoid double-free of fields transferred into `rt`
            rt.cold_cache = null;
        }

        // Allocate shared scratch bindings inside `rt.arena`.
        const q_dim = cfg.qDim();
        const kv_dim = cfg.kvDim();
        const max_inter = @max(@max(cfg.intermediate_size, cfg.moe_intermediate_size), @as(u32, 64));
        const num_experts = @max(cfg.num_experts, @as(u32, 1));
        const top_k = @max(cfg.num_experts_per_tok, @as(u32, 1));

        rt.x_alt_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * h * 2);
        rt.x_norm2_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * h * 2);
        rt.x_norm_k_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * h * 2);
        rt.x_norm_v_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * h * 2);
        rt.router_logits_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * num_experts * 2);
        rt.topk_ids_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * top_k * 4);
        rt.topk_w_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * top_k * 4);
        rt.exp_gate_up_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * 2 * max_inter * 2);
        rt.exp_up_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * max_inter * 2);
        rt.exp_act_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * max_inter * 2);
        rt.exp_out_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * top_k * h * 2);
        rt.exp_gathered_x_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * h * 2);
        rt.exp_down_tmp_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * h * 2);
        rt.exp_tok_idx_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * top_k * 4);
        rt.exp_dst_slot_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * top_k * 4);
        rt.last_hidden_f32 = try allocator.alloc(f32, @as(usize, max_m) * h);
        @memset(rt.last_hidden_f32, 0);

        if (ctx.enable_router_trace) {
            rt.router_trace_topks = try allocator.alloc(u32, @as(usize, eff_steps) * top_k);
            @memset(rt.router_trace_topks, 0);
            rt.router_trace_valid = try allocator.alloc(bool, eff_steps);
            @memset(rt.router_trace_valid, false);
            rt.router_trace_top_k = top_k;
            rt.router_trace_gpu_off = try allocArenaOff(&rt.arena, @as(usize, eff_steps) * top_k * @sizeOf(u32));
        }

        if (family.isInkling2()) {
            try loadInkling2(allocator, rt, ctx, &f32_pool, q_dim, kv_dim, eff_steps);
        } else if (family.isHy4()) {
            try loadHy4(allocator, rt, ctx, &f32_pool);
        } else {
            return error.InvalidConfig;
        }

        // Precompute active weight bytes streamed per `M = 1` decode token across all effective steps.
        var active_w: u64 = 0;
        for (ctx.source.entries()) |*e| {
            if (amf.isMetaEntry(e)) continue;
            const tk = e.tensorKind() catch continue;
            const nm = e.nameSlice();
            if (tk == .embedding or std.mem.indexOf(u8, nm, "lm_head") != null or std.mem.indexOf(u8, nm, "embed_tokens") != null) {
                continue;
            }
            const nb: u64 = e.data_bytes + e.params_bytes;
            const mult: u64 = if (ls.isRecurrent() and e.layer != 0xFFFF_FFFF and e.layer >= core_start and e.layer < core_end)
                num_passes
            else
                1;
            if (tk == .hot_expert or tk == .cold_expert) {
                active_w += (nb * top_k * mult) / num_experts;
            } else if (std.mem.indexOf(u8, nm, ".controllers.") != null) {
                const c_mult: u64 = if (num_passes > 1) (num_passes - 1) else 0;
                active_w += nb * c_mult;
            } else {
                active_w += nb * mult;
            }
        }
        rt.active_weight_bytes_per_tok = active_w;

        rt.owned_f32_slices = try f32_pool.toOwnedSlice(allocator);
        return rt;
    }

    // ------------------------------------------------------------------------
    // Unified Forward Execution (`stepDecode`, `prefillChunk`, `verifyTree`)
    // ------------------------------------------------------------------------

    pub fn forward(
        self: *M82Runtime,
        model: *graph.GraphModel,
        token_ids: []const u32,
        positions: []const u32,
        ancestor_masks: []const u64,
        base_pos: u32,
        num_loops: u32,
        compute_logits: bool,
        commit_state: bool,
        vis_override_src: ?metal.BufferBinding,
        vis_src_row_off: u32,
        vis_local_start: u32,
        vis_count: u32,
    ) graph.GraphStepError!qwen3.VerifyTreeResult {
        if (self.family.isInkling2()) {
            return self.forwardInkling2(
                model,
                token_ids,
                positions,
                ancestor_masks,
                base_pos,
                num_loops,
                compute_logits,
                commit_state,
                vis_override_src,
                vis_src_row_off,
                vis_local_start,
                vis_count,
            );
        }
        return self.forwardHy4(
            model,
            token_ids,
            positions,
            ancestor_masks,
            base_pos,
            compute_logits,
            commit_state,
            vis_override_src,
            vis_src_row_off,
            vis_local_start,
            vis_count,
        );
    }

    // ------------------------------------------------------------------------
    // Inkling 2 (`inkling2`, `inkling2_loop`) End-to-End Forward
    // ------------------------------------------------------------------------

    fn forwardInkling2(
        self: *M82Runtime,
        model: *graph.GraphModel,
        token_ids: []const u32,
        positions: []const u32,
        ancestor_masks: []const u64,
        base_pos: u32,
        num_loops: u32,
        compute_logits: bool,
        commit_state: bool,
        vis_override_src: ?metal.BufferBinding,
        vis_src_row_off: u32,
        vis_local_start: u32,
        vis_count: u32,
    ) graph.GraphStepError!qwen3.VerifyTreeResult {
        const cfg = &model.config;
        const m82c = &cfg.m82_cfg;
        const m: u32 = @intCast(token_ids.len);
        const h: u32 = cfg.hidden_size;
        const q_dim: u32 = cfg.qDim();
        const kv_dim: u32 = cfg.kvDim();
        const ls = cfg.axes.loop_schedule;
        const R: u32 = if (ls.isRecurrent()) @max(@as(u32, 1), num_loops) else 1;
        const core_start: u32 = if (R > 1) ls.start_layer else 0;
        const core_end: u32 = if (R > 1) (if (ls.end_layer > ls.start_layer) ls.end_layer else cfg.num_hidden_layers) else 0;

        if (!commit_state and m > 1 and R > 1) {
            self.stats.inkling2_core3_batched_verify_calls += 1;
        }

        for (0..m) |i| {
            model.token_ids_host[i] = token_ids[i];
            model.positions_host[i] = positions[i];
            model.ancestor_masks_host[i] = if (commit_state and m == 1) @as(u64, 1) else ancestor_masks[i];
        }

        var frame_opt: ?metal.Compute.Frame = null;
        errdefer if (frame_opt) |*f| f.abort();

        // 1. Token embedding + optional vision override + optional `embed_norm`.
        {
            const f = try ensureFrame(model, &frame_opt);
            try model.encodeEmbedLookupUnified(f, .{
                .weight = model.embed.weight,
                .scale_bias = model.embed.scale_bias,
                .tokens = .{ .token_ids = model.token_ids_bind },
                .out = model.x_bind,
                .m = m,
                .vocab_size = cfg.vocab_size,
                .hidden_size = h,
            }, model.token_ids_host, model.x_host, false);

            if (vis_override_src) |vsrc| {
                const row_bytes: u64 = @as(u64, h) * @sizeOf(f16);
                try model.moe_pipes.encodeCopyStrided(f, .{
                    .src = .{ .buffer = vsrc.buffer, .offset = vsrc.offset + @as(u64, vis_src_row_off) * row_bytes },
                    .dst = .{ .buffer = model.x_bind.buffer, .offset = model.x_bind.offset + @as(u64, vis_local_start) * row_bytes },
                    .num_rows = vis_count,
                    .copy_cols = h,
                    .src_stride = h,
                    .dst_stride = h,
                }, true);
            }

            if (self.ink_embed_norm) |enorm| {
                try model.attn_pipes.encodeRmsNorm(f, .{
                    .x = model.x_bind,
                    .w = enorm.binding,
                    .y = model.x_bind,
                    .num_rows = m,
                    .hidden_dim = h,
                    .eps = cfg.rms_norm_eps,
                    .w_dtype = enorm.dtype,
                }, true);
            }
        }

        // Reset AttnRes block stack for this token batch (starts EMPTY at layer 0 per Inkling2 spec).
        const block_layers: u32 = @max(@as(u32, 1), m82c.n_layers_per_attn_res_block);
        var num_blocks: u32 = 0;
        if (self.attn_res_ring) |*ring| {
            ring.count = 0;
            ring.head = 0;
        }
        self.dflare_tap_count = 0;
        self.router_trace_gpu_mask = 0;
        if (self.enable_router_trace and self.router_trace_valid.len > 0) {
            @memset(self.router_trace_valid, false);
        }

        // Prefill GPU watchdog guard (ADR-0026): fuse all 48 effective steps into ONE
        // `MTL4CommandBuffer` for `m <= 16` (decode & `DFlare` verify), while splitting
        // large prefill chunks (`m > 16`) every 6 effective steps so no single command
        // buffer approaches macOS's ~5 s GPU watchdog timeout.
        const watchdog_flush_interval: u32 = if (m > 16) 6 else 0;

        // 2. Execute schedule: Prelude -> Core x R (with pass restart + message bank) -> Coda.
        var eff_step: u32 = 0;
        if (R > 1) {
            // Prelude [0 .. core_start)
            for (0..core_start) |l| {
                if (watchdog_flush_interval > 0 and eff_step > 0 and (eff_step % watchdog_flush_interval) == 0) {
                    try self.flushMidPass(model, &frame_opt);
                }
                try self.maybeCaptureDFlareTap(model, &frame_opt, eff_step, m, model.x_bind);
                try self.execInkling2Step(model, &frame_opt, l, eff_step, 0, 0, base_pos, positions, m, commit_state, &num_blocks, block_layers, q_dim, kv_dim, 0);
                eff_step += 1;
            }

            // Save prelude output (`prelude_cur`) and block count (`prelude_num_blocks`) so passes `1..R-1` restart cleanly.
            const prelude_num_blocks: u32 = num_blocks;
            {
                const f = try ensureFrame(model, &frame_opt);
                try model.moe_pipes.encodeCopyStrided(f, .{
                    .src = model.x_bind,
                    .dst = self.bindAt(self.ink_prelude_cur_off),
                    .num_rows = m,
                    .copy_cols = h,
                    .src_stride = h,
                    .dst_stride = h,
                }, true);
            }

            // Core [core_start .. core_end) x R passes
            var r: u32 = 0;
            while (r < R) : (r += 1) {
                var bank_len: usize = 0;
                if (r > 0) {
                    // Build normalized message bank from previous pass: `blocks[prelude_num_blocks + 1 .. num_blocks]` + `cur`.
                    if (self.attn_res_ring) |*ring| {
                        const start_blk: u32 = @min(prelude_num_blocks + 1, num_blocks);
                        if (self.use_sme2_controllers) {
                            try self.flushMidPass(model, &frame_opt);
                            bank_len = stageMsgBankF32(
                                ring,
                                self.arena.bytes(),
                                model.x_host[0 .. @as(usize, m) * h],
                                start_blk,
                                num_blocks,
                                m,
                                h,
                                cfg.rms_norm_eps,
                                self.ink_msg_bank_f32,
                            );
                        } else {
                            const f = try ensureFrame(model, &frame_opt);
                            bank_len = try self.pipes.encodeInkling2MsgBankNorm(f, .{
                                .ring = ring,
                                .ring_buf = self.buf,
                                .cur = model.x_bind,
                                .msg_bank_out = self.bindAt(self.ink_msg_bank_gpu_off),
                                .m = m,
                                .start_slot = start_blk,
                                .end_slot = num_blocks,
                                .append_cur = true,
                                .rms_eps = cfg.rms_norm_eps,
                            }, true);
                        }
                    }
                    // Restart `cur = prelude_cur` and `blocks = prelude_blocks`.
                    num_blocks = prelude_num_blocks;
                    const f = try ensureFrame(model, &frame_opt);
                    try model.moe_pipes.encodeCopyStrided(f, .{
                        .src = self.bindAt(self.ink_prelude_cur_off),
                        .dst = model.x_bind,
                        .num_rows = m,
                        .copy_cols = h,
                        .src_stride = h,
                        .dst_stride = h,
                    }, true);
                }

                for (core_start..core_end) |l| {
                    if (watchdog_flush_interval > 0 and eff_step > 0 and (eff_step % watchdog_flush_interval) == 0) {
                        try self.flushMidPass(model, &frame_opt);
                    }
                    try self.maybeCaptureDFlareTap(model, &frame_opt, eff_step, m, model.x_bind);
                    try self.execInkling2Step(model, &frame_opt, l, eff_step, r, l - core_start, base_pos, positions, m, commit_state, &num_blocks, block_layers, q_dim, kv_dim, bank_len);
                    eff_step += 1;
                }
            }

            // Coda [core_end .. num_hidden_layers)
            for (core_end..cfg.num_hidden_layers) |l| {
                if (watchdog_flush_interval > 0 and eff_step > 0 and (eff_step % watchdog_flush_interval) == 0) {
                    try self.flushMidPass(model, &frame_opt);
                }
                try self.maybeCaptureDFlareTap(model, &frame_opt, eff_step, m, model.x_bind);
                try self.execInkling2Step(model, &frame_opt, l, eff_step, 0, 0, base_pos, positions, m, commit_state, &num_blocks, block_layers, q_dim, kv_dim, 0);
                eff_step += 1;
            }
        } else {
            for (0..cfg.num_hidden_layers) |l| {
                if (watchdog_flush_interval > 0 and eff_step > 0 and (eff_step % watchdog_flush_interval) == 0) {
                    try self.flushMidPass(model, &frame_opt);
                }
                try self.maybeCaptureDFlareTap(model, &frame_opt, eff_step, m, model.x_bind);
                try self.execInkling2Step(model, &frame_opt, l, eff_step, 0, 0, base_pos, positions, m, commit_state, &num_blocks, block_layers, q_dim, kv_dim, 0);
                eff_step += 1;
            }
        }

        // 3. Final AttnRes output query + final RMSNorm + GPU DFlare final tap (`eff_steps`, e.g. 48)
        //    all encoded inside the SAME `MTL4CommandBuffer` as `lm_head` (`0` mid-pass flushes!).
        const attn_res_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(h)));
        if (self.attn_res_ring) |*ring| {
            if (self.ink_attn_res_out_q) |oq| {
                const f = try ensureFrame(model, &frame_opt);
                const partial_slot = @min(num_blocks, ring.spec.max_blocks - 1);
                try model.moe_pipes.encodeCopyStrided(f, .{
                    .src = model.x_bind,
                    .dst = attnResBlockBind(ring, self.buf, partial_slot),
                    .num_rows = m,
                    .copy_cols = h,
                    .src_stride = h,
                    .dst_stride = h,
                }, true);
                ring.count = partial_slot + 1;
                ring.head = 0;
                try self.pipes.encodeInkling2AttnRes(f, .{
                    .ring = ring,
                    .ring_buf = self.buf,
                    .query = oq,
                    .out = model.x_bind,
                    .m = m,
                    .num_blocks_override = partial_slot + 1,
                    .apply_rms_norm = true,
                    .normalize_values = m82c.attn_res_normalize_values,
                    .rms_eps = cfg.rms_norm_eps,
                    .scale = attn_res_scale,
                }, true);
                self.stats.inkling2_attn_res_dispatches += 1;
            }
        }
        {
            const f = try ensureFrame(model, &frame_opt);
            try model.attn_pipes.encodeRmsNorm(f, .{
                .x = model.x_bind,
                .w = model.final_norm.binding,
                .y = model.x_norm_bind,
                .num_rows = m,
                .hidden_dim = h,
                .eps = cfg.rms_norm_eps,
                .w_dtype = model.final_norm.dtype,
            }, true);
        }
        try self.maybeCaptureDFlareTap(model, &frame_opt, eff_step, m, model.x_norm_bind);

        // Execute `lm_head` + SINGLE end-of-forward commit+wait.
        const res = try self.finalizeLogits(model, &frame_opt, m, compute_logits, m82c.final_logit_softcapping, m82c.unpadded_vocab_size, true);

        // Unified memory is now coherent: harvest zero-flush GPU router-trace selections (if enabled).
        if (self.enable_router_trace and self.router_trace_gpu_mask != 0 and self.router_trace_gpu_off > 0) {
            const tk_us: usize = self.router_trace_top_k;
            const gpu_topks = self.sliceU32(self.router_trace_gpu_off, self.router_trace_valid.len * tk_us);
            const limit: usize = @min(self.router_trace_valid.len, @as(usize, 64));
            for (0..limit) |si| {
                if ((self.router_trace_gpu_mask & (@as(u64, 1) << @intCast(si))) != 0) {
                    @memcpy(self.router_trace_topks[si * tk_us ..][0..tk_us], gpu_topks[si * tk_us ..][0..tk_us]);
                }
            }
        }

        // Unified memory is now coherent: snapshot final normed hidden state (`f32`) and update DFlare taps / history.
        const h_us: usize = h;
        f16SliceToF32(model.x_norm_host[0 .. @as(usize, m) * h_us], self.last_hidden_f32[0 .. @as(usize, m) * h_us]);
        if (commit_state) {
            if (m > 1) {
                const last_row: usize = @as(usize, m) - 1;
                @memcpy(self.last_hidden_f32[0..h_us], self.last_hidden_f32[last_row * h_us ..][0..h_us]);
            }
            if (base_pos == 0) {
                self.dflare_ctx_hist_len = 0;
                self.dflare_ctx_hist_epoch +%= 1;
            }
            if (self.hasDFlareCtxHistory()) {
                for (0..m) |tok| {
                    self.recordDFlareCtxTapRow(positions[tok], tok, h_us);
                }
            }
            self.syncDFlareActiveRow(if (m > 0) @as(usize, m) - 1 else 0, h_us);
            self.sconv_pos = base_pos + m;
        }
        return res;
    }

    fn execInkling2Step(
        self: *M82Runtime,
        model: *graph.GraphModel,
        frame_opt: *?metal.Compute.Frame,
        l: usize,
        eff_step: u32,
        loop_pass: u32,
        loop_layer_idx: usize,
        base_pos: u32,
        positions: []const u32,
        m: u32,
        commit_state: bool,
        num_blocks: *u32,
        block_layers: u32,
        q_dim: u32,
        kv_dim: u32,
        msg_bank_len: usize,
    ) graph.GraphStepError!void {
        _ = positions;
        _ = q_dim;
        const cfg = &model.config;
        const m82c = &cfg.m82_cfg;
        const h: u32 = cfg.hidden_size;
        const mh: usize = @as(usize, m) * h;
        const layer = &self.ink_layers[l];
        const step_st = &self.ink_steps[eff_step];
        const attn_res_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(h)));
        const is_block_boundary: bool = ((l % block_layers) == 0);
        const has_ctrl = if (self.use_sme2_controllers) (layer.loop_ctrl != null) else (layer.gpu_ctrl != null);
        const has_cp_q = if (self.use_sme2_controllers) (self.ink_cross_pass_q_f32 != null) else (self.ink_cross_pass_q_gpu != null);
        const use_messages: bool = (loop_pass >= 1 and has_ctrl and msg_bank_len > 0 and has_cp_q);

        // Compute `[msg_q, msg_k, msg_v, msg_mlp]` from the previous pass's bank if `loop_pass >= 1`.
        if (use_messages) {
            if (self.use_sme2_controllers) {
                computeCrossPassMessagesF32(
                    self.ink_msg_bank_f32,
                    msg_bank_len,
                    self.ink_cross_pass_q_f32.?,
                    self.ink_cross_pass_kv_q_f32,
                    loop_layer_idx,
                    m,
                    h,
                    self.ink_msg_buf_f32,
                );
            } else {
                const f = try ensureFrame(model, frame_opt);
                try self.pipes.encodeInkling2CrossPassMsgs(f, .{
                    .msg_bank = self.bindAt(self.ink_msg_bank_gpu_off),
                    .q_src = self.ink_cross_pass_q_gpu.?,
                    .kv_q_src = self.ink_cross_pass_kv_q_gpu,
                    .queries_f16 = false,
                    .out_msgs = self.bindAt(self.ink_msg_buf_gpu_off),
                    .m = m,
                    .hidden_dim = h,
                    .bank_len = @intCast(msg_bank_len),
                    .q_rows = self.ink_cross_pass_q_rows,
                    .kv_rows = self.ink_cross_pass_kv_q_rows,
                    .loop_layer_idx = @intCast(loop_layer_idx),
                }, true);
            }
        }

        // Save `inpSA` (`block_residual = cur`) before attention.
        {
            const f = try ensureFrame(model, frame_opt);
            try model.moe_pipes.encodeCopyStrided(f, .{
                .src = model.x_bind,
                .dst = self.bindAt(self.ink_inp_sa_off),
                .num_rows = m,
                .copy_cols = h,
                .src_stride = h,
                .dst_stride = h,
            }, true);
        }

        // --- A. Pre-Attention AttnRes (`G25`, 1 query on SWA or 3 queries [Q,K,V] on global) ---
        var pre_q_bind = model.x_bind;
        var pre_k_bind = model.x_bind;
        var pre_v_bind = model.x_bind;
        var kv_differ: bool = false;
        if (self.attn_res_ring) |*ring| {
            if (layer.attn_res_attn_q) |aq| {
                const f = try ensureFrame(model, frame_opt);
                const partial_slot = @min(num_blocks.*, ring.spec.max_blocks - 1);
                try model.moe_pipes.encodeCopyStrided(f, .{
                    .src = model.x_bind,
                    .dst = attnResBlockBind(ring, self.buf, partial_slot),
                    .num_rows = m,
                    .copy_cols = h,
                    .src_stride = h,
                    .dst_stride = h,
                }, true);
                ring.count = partial_slot + 1;
                ring.head = 0;

                if (layer.attn_res_attn_q_passes >= 3) {
                    const out_binds = [3]metal.BufferBinding{
                        self.bindAt(self.x_alt_off),
                        self.bindAt(self.x_norm_k_off),
                        self.bindAt(self.x_norm_v_off),
                    };
                    for (0..3) |qi| {
                        try self.pipes.encodeInkling2AttnRes(f, .{
                            .ring = ring,
                            .ring_buf = self.buf,
                            .query = .{ .buffer = aq.buffer, .offset = aq.offset + @as(u64, qi) * h * 2 },
                            .out = out_binds[qi],
                            .m = m,
                            .num_blocks_override = partial_slot + 1,
                            .apply_rms_norm = true,
                            .normalize_values = m82c.attn_res_normalize_values,
                            .rms_eps = cfg.rms_norm_eps,
                            .scale = attn_res_scale,
                        }, qi == 0);
                        self.stats.inkling2_attn_res_dispatches += 1;
                    }
                    pre_q_bind = out_binds[0];
                    pre_k_bind = out_binds[1];
                    pre_v_bind = out_binds[2];
                    kv_differ = true;
                } else {
                    const out_b = self.bindAt(self.x_alt_off);
                    try self.pipes.encodeInkling2AttnRes(f, .{
                        .ring = ring,
                        .ring_buf = self.buf,
                        .query = aq,
                        .out = out_b,
                        .m = m,
                        .num_blocks_override = partial_slot + 1,
                        .apply_rms_norm = true,
                        .normalize_values = m82c.attn_res_normalize_values,
                        .rms_eps = cfg.rms_norm_eps,
                        .scale = attn_res_scale,
                    }, true);
                    self.stats.inkling2_attn_res_dispatches += 1;
                    pre_q_bind = out_b;
                    pre_k_bind = out_b;
                    pre_v_bind = out_b;
                }
            }
        }

        // --- B. Optional Cross-Pass Message Gate (`G29` GPU zero-flush default, or opt-in C26 SME2) ---
        if (use_messages) {
            if (self.use_sme2_controllers) {
                try self.flushMidPass(model, frame_opt);
                const ctrl = &(layer.loop_ctrl.?);
                const q_src_f16: []const f16 = if (pre_q_bind.buffer.handle == self.buf.handle)
                    self.sliceF16(pre_q_bind.offset, mh)
                else
                    model.x_host[0..mh];
                const k_src_f16: []const f16 = if (kv_differ) self.sliceF16(pre_k_bind.offset, mh) else q_src_f16;
                const v_src_f16: []const f16 = if (kv_differ) self.sliceF16(pre_v_bind.offset, mh) else q_src_f16;

                // Compute K and V gates BEFORE overwriting `x_alt_off` (in case SWA shares `q_src_f16 == x_alt_off`).
                if (ctrl.branches[1].config.state_dim == h) {
                    try self.applySme2MessageGate(ctrl, .k, k_src_f16, self.ink_msg_buf_f32[1 * mh ..][0..mh], self.sliceF16(self.x_norm_k_off, mh), m, h);
                    pre_k_bind = self.bindAt(self.x_norm_k_off);
                } else {
                    @memcpy(self.sliceF16(self.x_norm_k_off, mh), k_src_f16);
                    pre_k_bind = self.bindAt(self.x_norm_k_off);
                }
                if (ctrl.branches[2].config.state_dim == h) {
                    try self.applySme2MessageGate(ctrl, .v, v_src_f16, self.ink_msg_buf_f32[2 * mh ..][0..mh], self.sliceF16(self.x_norm_v_off, mh), m, h);
                    pre_v_bind = self.bindAt(self.x_norm_v_off);
                } else {
                    @memcpy(self.sliceF16(self.x_norm_v_off, mh), v_src_f16);
                    pre_v_bind = self.bindAt(self.x_norm_v_off);
                }
                try self.applySme2MessageGate(ctrl, .attn, q_src_f16, self.ink_msg_buf_f32[0..mh], self.sliceF16(self.x_alt_off, mh), m, h);
                pre_q_bind = self.bindAt(self.x_alt_off);
                kv_differ = true;
            } else {
                const f = try ensureFrame(model, frame_opt);
                const gctrl = &(layer.gpu_ctrl.?);
                const msg_stride_b: u64 = @as(u64, mh) * @sizeOf(f32);
                const z_b = self.bindAt(self.ink_ctrl_z_gpu_off);

                const q_out_b = self.bindAt(self.x_ctrl_q_off);
                const br_q = gctrl.branches[0];
                try self.pipes.encodeInkling2MessageGate(f, .{
                    .cur = pre_q_bind,
                    .msg = self.bindAt(self.ink_msg_buf_gpu_off + 0 * msg_stride_b),
                    .down_weight = br_q.down_weight,
                    .forget_weight = br_q.forget_weight,
                    .gain_weight = br_q.gain_weight,
                    .weight_dtype = br_q.weight_dtype,
                    .z_scratch = z_b,
                    .out = q_out_b,
                    .m = m,
                    .state_dim = br_q.state_dim,
                    .down_in_dim = br_q.down_in_dim,
                    .gate_rank = br_q.gate_rank,
                    .gain_act = .tanh,
                    .rms_eps = cfg.rms_norm_eps,
                }, true);
                self.stats.inkling2_gpu_ctrl_calls += 1;

                const br_k = gctrl.branches[1];
                const k_out_b = self.bindAt(self.x_ctrl_k_off);
                if (br_k.state_dim == h) {
                    try self.pipes.encodeInkling2MessageGate(f, .{
                        .cur = pre_k_bind,
                        .msg = self.bindAt(self.ink_msg_buf_gpu_off + 1 * msg_stride_b),
                        .down_weight = br_k.down_weight,
                        .forget_weight = br_k.forget_weight,
                        .gain_weight = br_k.gain_weight,
                        .weight_dtype = br_k.weight_dtype,
                        .z_scratch = z_b,
                        .out = k_out_b,
                        .m = m,
                        .state_dim = br_k.state_dim,
                        .down_in_dim = br_k.down_in_dim,
                        .gate_rank = br_k.gate_rank,
                        .gain_act = .tanh,
                        .rms_eps = cfg.rms_norm_eps,
                    }, true);
                    self.stats.inkling2_gpu_ctrl_calls += 1;
                    pre_k_bind = k_out_b;
                }

                const br_v = gctrl.branches[2];
                const v_out_b = self.bindAt(self.x_ctrl_v_off);
                if (br_v.state_dim == h) {
                    try self.pipes.encodeInkling2MessageGate(f, .{
                        .cur = pre_v_bind,
                        .msg = self.bindAt(self.ink_msg_buf_gpu_off + 2 * msg_stride_b),
                        .down_weight = br_v.down_weight,
                        .forget_weight = br_v.forget_weight,
                        .gain_weight = br_v.gain_weight,
                        .weight_dtype = br_v.weight_dtype,
                        .z_scratch = z_b,
                        .out = v_out_b,
                        .m = m,
                        .state_dim = br_v.state_dim,
                        .down_in_dim = br_v.down_in_dim,
                        .gate_rank = br_v.gate_rank,
                        .gain_act = .tanh,
                        .rms_eps = cfg.rms_norm_eps,
                    }, true);
                    self.stats.inkling2_gpu_ctrl_calls += 1;
                    pre_v_bind = v_out_b;
                }
                pre_q_bind = q_out_b;
                kv_differ = true;
            }
        }

        // --- C. Pre-Attention Input RMSNorms (`attn_norm` on SWA, separate `{q,k,v}_input_norm` on global) ---
        const q_in_bind = model.x_norm_bind;
        var k_in_bind = model.x_norm_bind;
        var v_in_bind = model.x_norm_bind;
        {
            const f = try ensureFrame(model, frame_opt);
            const q_norm = layer.q_input_norm orelse layer.attn_norm;
            const k_norm = layer.k_input_norm orelse layer.attn_norm;
            const v_norm = layer.v_input_norm orelse layer.attn_norm;
            try model.attn_pipes.encodeRmsNorm(f, .{
                .x = pre_q_bind,
                .w = q_norm.binding,
                .y = q_in_bind,
                .num_rows = m,
                .hidden_dim = h,
                .eps = cfg.rms_norm_eps,
                .w_dtype = q_norm.dtype,
            }, true);
            if (kv_differ or layer.k_input_norm != null or layer.v_input_norm != null) {
                k_in_bind = self.bindAt(self.x_norm_k_off);
                v_in_bind = self.bindAt(self.x_norm_v_off);
                try model.attn_pipes.encodeRmsNorm(f, .{
                    .x = pre_k_bind,
                    .w = k_norm.binding,
                    .y = k_in_bind,
                    .num_rows = m,
                    .hidden_dim = h,
                    .eps = cfg.rms_norm_eps,
                    .w_dtype = k_norm.dtype,
                }, false);
                try model.attn_pipes.encodeRmsNorm(f, .{
                    .x = pre_v_bind,
                    .w = v_norm.binding,
                    .y = v_in_bind,
                    .num_rows = m,
                    .hidden_dim = h,
                    .eps = cfg.rms_norm_eps,
                    .w_dtype = v_norm.dtype,
                }, false);
            }
        }

        // --- D. Q, K, V, Q_rel, Sink, Gate projections (`G23` multi-scheme) ---
        {
            const f = try ensureFrame(model, frame_opt);
            try model.encodeLinearBatched(f, layer.wq, q_in_bind, model.q_bind, m, true);
            try model.encodeLinearBatched(f, layer.wk, k_in_bind, model.k_bind, m, false);
            try model.encodeLinearBatched(f, layer.wv, v_in_bind, model.v_bind, m, false);
            if (layer.wr) |wr| {
                try model.encodeLinearBatched(f, wr, q_in_bind, self.bindAt(self.q_rel_off), m, false);
            }
            if (layer.attn_sink_proj) |sp| {
                try model.encodeLinearBatched(f, sp, q_in_bind, self.bindAt(self.sink_f16_off), m, false);
            }
            if (layer.gate_attn) |ga| {
                try model.encodeLinearBatched(f, ga, q_in_bind, self.bindAt(self.gate_attn_off), m, false);
            }
            if (model.hasActiveLora()) {
                _ = try model.encodeLoraSingle(f, l, .q_proj, q_in_bind, model.q_bind, m, layer.wq.k, layer.wq.n, true);
                _ = try model.encodeLoraSingle(f, l, .k_proj, k_in_bind, model.k_bind, m, layer.wk.k, layer.wk.n, true);
                _ = try model.encodeLoraSingle(f, l, .v_proj, v_in_bind, model.v_bind, m, layer.wv.k, layer.wv.n, true);
                if (layer.wr) |wr| {
                    _ = try model.encodeLoraSingle(f, l, .wr_du, q_in_bind, self.bindAt(self.q_rel_off), m, wr.k, wr.n, true);
                }
                if (layer.gate_attn) |ga| {
                    _ = try model.encodeLoraSingle(f, l, .gate_attn, q_in_bind, self.bindAt(self.gate_attn_off), m, ga.k, ga.n, true);
                }
            }
        }

        // K/V `sconv4` (static `/ sqrt(5)` on SWA layers, dynamic `weight_proj` `/ 3.0` on global layers).
        if (layer.k_sconv_w != null or layer.k_sconv_dyn_proj != null) {
            try self.applyInkling2KvSconv(model, frame_opt, layer, step_st, true, k_in_bind, model.k_bind, step_st.tree_k_sconv_in_off, m, kv_dim, commit_state);
        }
        if (layer.v_sconv_w != null or layer.v_sconv_dyn_proj != null) {
            try self.applyInkling2KvSconv(model, frame_opt, layer, step_st, false, v_in_bind, model.v_bind, step_st.tree_v_sconv_in_off, m, kv_dim, commit_state);
        }

        // Exact Q/K/R ScaledL2Clip + optional LogScale (`!is_local`) + V UnweightedRMSNorm + sink f16->f32 on GPU (`G29`, `0` CPU flush).
        {
            const f = try ensureFrame(model, frame_opt);
            const use_log = (!layer.is_local and m82c.log_scaling_n_floor > 0 and m82c.log_scaling_alpha > 0.0);
            try self.pipes.encodeInkling2AttnPostProj(f, .{
                .q = model.q_bind,
                .k = model.k_bind,
                .q_rel = if (layer.wr != null) self.bindAt(self.q_rel_off) else null,
                .v = if (m82c.use_value_norm_no_gamma) model.v_bind else null,
                .q_scale_vec = layer.q_l2_scale_vec_gpu,
                .rel_scale_vec = layer.rel_l2_scale_vec_gpu,
                .positions = if (use_log) model.positions_bind else null,
                .sink_f16 = if (layer.attn_sink_proj != null) self.bindAt(self.sink_f16_off) else null,
                .sink_f32 = if (layer.attn_sink_proj != null) self.bindAt(self.sink_f32_off) else null,
                .m = m,
                .num_q_heads = cfg.num_attention_heads,
                .num_kv_heads = cfg.num_key_value_heads,
                .head_dim = cfg.head_dim,
                .d_rel = m82c.d_rel,
                .q_clip = m82c.qk_l2_clip_threshold * layer.q_l2_clip_scale,
                .k_clip = m82c.qk_l2_clip_threshold,
                .rel_clip = m82c.rel_l2_clip_threshold * layer.rel_l2_clip_scale,
                .use_log_scaling = use_log,
                .base_pos = base_pos,
                .log_floor = m82c.log_scaling_n_floor,
                .log_alpha = m82c.log_scaling_alpha,
                .do_v_norm = m82c.use_value_norm_no_gamma,
                .rms_eps = cfg.rms_norm_eps,
            }, true);
        }

        // --- E. KV Cache Write / Tree Scratch + Single-Pass `G25` `encodeInkling2RelGqaAttn` + `wo_ud` ---
        {
            const f = try ensureFrame(model, frame_opt);
            const kv_ring = &step_st.kv_ring;
            if (commit_state) {
                try model.hybrid_pipes.encodeQsaPrefillKvRingCommit(f, .{
                    .k = model.k_bind,
                    .v = model.v_bind,
                    .kv_tiles = kv_ring.tileTableBinding(self.buf),
                    .num_tokens = m,
                    .num_kv_heads = cfg.num_key_value_heads,
                    .head_dim = cfg.head_dim,
                    .start_pos = base_pos,
                    .chunk_tokens = kv_ring.chunk_tokens,
                    .ring_capacity = kv_ring.ring_capacity,
                }, true);

                try self.pipes.encodeInkling2RelGqaAttn(f, .{
                    .q = model.q_bind,
                    .q_rel = if (layer.wr != null and layer.rel_emb_f16 != null) self.bindAt(self.q_rel_off) else null,
                    .rel_emb = if (layer.wr != null) layer.rel_emb_f16 else null,
                    .kv_ring = kv_ring,
                    .kv_ring_buf = self.buf,
                    .sink_logits = if (layer.attn_sink_proj != null) self.bindAt(self.sink_f32_off) else null,
                    .sink_per_token = true,
                    .gate_attn = if (layer.gate_attn != null) self.bindAt(self.gate_attn_off) else null,
                    .gate_mode = .sigmoid,
                    .out = model.attn_out_bind,
                    .num_nodes = m,
                    .num_q_heads = cfg.num_attention_heads,
                    .kv_len = base_pos + m,
                    .causal_q_start_pos = base_pos,
                    .d_rel = if (layer.wr != null and layer.rel_emb_f16 != null) m82c.d_rel else 0,
                    .rel_extent = m82c.rel_extent,
                    .rel_zero_oob = true,
                    .use_log_scaling = false,
                    .qk_l2_clip = 0.0,
                    .rel_l2_clip = 0.0,
                    .rel_scale = 1.0,
                    .attn_softcap = 0.0,
                }, true);
            } else {
                const tk_bind = self.bindAt(step_st.tree_k_off);
                const tv_bind = self.bindAt(step_st.tree_v_off);
                try model.moe_pipes.encodeCopyStrided(f, .{
                    .src = model.k_bind,
                    .dst = tk_bind,
                    .num_rows = m,
                    .copy_cols = kv_dim,
                    .src_stride = kv_dim,
                    .dst_stride = kv_dim,
                }, true);
                try model.moe_pipes.encodeCopyStrided(f, .{
                    .src = model.v_bind,
                    .dst = tv_bind,
                    .num_rows = m,
                    .copy_cols = kv_dim,
                    .src_stride = kv_dim,
                    .dst_stride = kv_dim,
                }, false);

                try self.pipes.encodeInkling2RelGqaAttn(f, .{
                    .q = model.q_bind,
                    .q_rel = if (layer.wr != null and layer.rel_emb_f16 != null) self.bindAt(self.q_rel_off) else null,
                    .rel_emb = if (layer.wr != null) layer.rel_emb_f16 else null,
                    .kv_ring = kv_ring,
                    .kv_ring_buf = self.buf,
                    .k_tree = tk_bind,
                    .v_tree = tv_bind,
                    .ancestor_masks = model.ancestor_masks_bind,
                    .sink_logits = if (layer.attn_sink_proj != null) self.bindAt(self.sink_f32_off) else null,
                    .sink_per_token = true,
                    .gate_attn = if (layer.gate_attn != null) self.bindAt(self.gate_attn_off) else null,
                    .gate_mode = .sigmoid,
                    .out = model.attn_out_bind,
                    .num_nodes = m,
                    .num_q_heads = cfg.num_attention_heads,
                    .kv_len = base_pos,
                    .d_rel = if (layer.wr != null and layer.rel_emb_f16 != null) m82c.d_rel else 0,
                    .rel_extent = m82c.rel_extent,
                    .rel_zero_oob = true,
                    .use_log_scaling = false,
                    .qk_l2_clip = 0.0,
                    .rel_l2_clip = 0.0,
                    .rel_scale = 1.0,
                    .attn_softcap = 0.0,
                }, true);
            }
            self.stats.inkling2_rel_gqa_dispatches += 1;

            // Output projection `wo_ud` -> `model.proj_out_bind` (separate from residual, so DoRA runs strictly before residual add)
            try model.encodeLinearBatched(f, layer.wo, model.attn_out_bind, model.proj_out_bind, m, true);
            if (model.hasActiveLora()) {
                _ = try model.encodeLoraSingle(f, l, .o_proj, model.attn_out_bind, model.proj_out_bind, m, layer.wo.k, layer.wo.n, true);
            }

            // Post-attention convolution (`attn_sconv.weight`: `(conv(y) + y) / sqrt(5)`)!
            if (layer.attn_sconv_w) |sw| {
                const sring = &(self.sconv_h_ring.?);
                const state_b = try sring.layerBinding(step_st.sconv_attn_slot, self.buf);
                if (!commit_state) {
                    try model.moe_pipes.encodeCopyStrided(f, .{
                        .src = model.proj_out_bind,
                        .dst = self.bindAt(step_st.tree_attn_sconv_in_off),
                        .num_rows = m,
                        .copy_cols = h,
                        .src_stride = h,
                        .dst_stride = h,
                    }, true);
                }
                try self.pipes.encodeRmsNormSconv4(f, .{
                    .x = if (!commit_state) self.bindAt(step_st.tree_attn_sconv_in_off) else model.proj_out_bind,
                    .conv_w = sw,
                    .conv_w_f16 = false,
                    .conv_state_in = state_b,
                    .conv_state_out = state_b,
                    .out = model.proj_out_bind,
                    .ancestor_masks = if (!commit_state) model.ancestor_masks_bind else null,
                    .m = m,
                    .hidden_dim = h,
                    .state_pos = self.sconv_pos,
                    .write_state = commit_state,
                    .apply_rms_norm = false,
                    .apply_silu = false,
                    .post_residual_scale = 1.0 / @sqrt(@as(f32, 5.0)),
                }, true);
                self.stats.inkling2_sconv_dispatches += 1;
            }

            // Block boundary rule: if `l % block_layers == 0`, push `inpSA` to `blocks` and start fresh partial block (`cur = attn_out`);
            // otherwise accumulate into partial block (`cur = inpSA + attn_out`).
            if (is_block_boundary) {
                if (self.attn_res_ring) |*ring| {
                    if (num_blocks.* < ring.spec.max_blocks - 1) {
                        try model.moe_pipes.encodeCopyStrided(f, .{
                            .src = self.bindAt(self.ink_inp_sa_off),
                            .dst = attnResBlockBind(ring, self.buf, num_blocks.*),
                            .num_rows = m,
                            .copy_cols = h,
                            .src_stride = h,
                            .dst_stride = h,
                        }, true);
                        num_blocks.* += 1;
                    }
                }
                try model.moe_pipes.encodeCopyStrided(f, .{
                    .src = model.proj_out_bind,
                    .dst = model.x_bind,
                    .num_rows = m,
                    .copy_cols = h,
                    .src_stride = h,
                    .dst_stride = h,
                }, true);
            } else {
                try model.moe_pipes.encodeResidualAdd(f, .{
                    .x = model.x_bind,
                    .delta = model.proj_out_bind,
                    .count = m * h,
                }, true);
            }
        }

        // --- F. MLP AttnRes Query + optional SME2 Message Gate + `mlp_norm` + Multi-Scheme MoE + `mlp_sconv` ---
        var pre_mlp_bind: metal.BufferBinding = model.x_bind;
        if (self.attn_res_ring) |*ring| {
            if (layer.attn_res_mlp_q) |mq| {
                const f = try ensureFrame(model, frame_opt);
                const partial_slot = @min(num_blocks.*, ring.spec.max_blocks - 1);
                try model.moe_pipes.encodeCopyStrided(f, .{
                    .src = model.x_bind,
                    .dst = attnResBlockBind(ring, self.buf, partial_slot),
                    .num_rows = m,
                    .copy_cols = h,
                    .src_stride = h,
                    .dst_stride = h,
                }, true);
                ring.count = partial_slot + 1;
                ring.head = 0;
                const out_b = self.bindAt(self.x_alt_off);
                try self.pipes.encodeInkling2AttnRes(f, .{
                    .ring = ring,
                    .ring_buf = self.buf,
                    .query = mq,
                    .out = out_b,
                    .m = m,
                    .num_blocks_override = partial_slot + 1,
                    .apply_rms_norm = true,
                    .normalize_values = m82c.attn_res_normalize_values,
                    .rms_eps = cfg.rms_norm_eps,
                    .scale = attn_res_scale,
                }, true);
                self.stats.inkling2_attn_res_dispatches += 1;
                pre_mlp_bind = out_b;
            }
        }

        if (use_messages) {
            if (self.use_sme2_controllers) {
                try self.flushMidPass(model, frame_opt);
                const ctrl = &(layer.loop_ctrl.?);
                const mlp_src_f16: []const f16 = if (pre_mlp_bind.buffer.handle == self.buf.handle)
                    self.sliceF16(pre_mlp_bind.offset, mh)
                else
                    model.x_host[0..mh];
                try self.applySme2MessageGate(ctrl, .mlp, mlp_src_f16, self.ink_msg_buf_f32[3 * mh ..][0..mh], self.sliceF16(self.x_alt_off, mh), m, h);
                pre_mlp_bind = self.bindAt(self.x_alt_off);
            } else {
                const f = try ensureFrame(model, frame_opt);
                const gctrl = &(layer.gpu_ctrl.?);
                const br_m = gctrl.branches[3];
                const msg_stride_b: u64 = @as(u64, mh) * @sizeOf(f32);
                const mlp_out_b = self.bindAt(self.x_norm2_off);
                try self.pipes.encodeInkling2MessageGate(f, .{
                    .cur = pre_mlp_bind,
                    .msg = self.bindAt(self.ink_msg_buf_gpu_off + 3 * msg_stride_b),
                    .down_weight = br_m.down_weight,
                    .forget_weight = br_m.forget_weight,
                    .gain_weight = br_m.gain_weight,
                    .weight_dtype = br_m.weight_dtype,
                    .z_scratch = self.bindAt(self.ink_ctrl_z_gpu_off),
                    .out = mlp_out_b,
                    .m = m,
                    .state_dim = br_m.state_dim,
                    .down_in_dim = br_m.down_in_dim,
                    .gate_rank = br_m.gate_rank,
                    .gain_act = .tanh,
                    .rms_eps = cfg.rms_norm_eps,
                }, true);
                self.stats.inkling2_gpu_ctrl_calls += 1;
                pre_mlp_bind = mlp_out_b;
            }
        }

        const top_k = cfg.num_experts_per_tok;
        const inter = cfg.intermediate_size;
        const qf_ptr: ?*const quant_formats.QuantFormatPipelines = if (model.qf_pipes) |*qfp| qfp else null;
        const act_b: ?metal.BufferBinding = if (model.act_quant_buf) |ab| .{ .buffer = ab, .offset = 0 } else null;
        const tile_b: ?metal.BufferBinding = if (model.qf_tile_buf) |tb| .{ .buffer = tb, .offset = 0 } else null;

        {
            const f = try ensureFrame(model, frame_opt);
            try model.attn_pipes.encodeRmsNorm(f, .{
                .x = pre_mlp_bind,
                .w = layer.mlp_norm.binding,
                .y = model.x_norm_bind,
                .num_rows = m,
                .hidden_dim = h,
                .eps = cfg.rms_norm_eps,
                .w_dtype = layer.mlp_norm.dtype,
            }, true);

            // Router logits (`model.x_norm_bind` -> `router_logits_off`)
            try model.encodeLinearBatched(f, layer.router, model.x_norm_bind, self.bindAt(self.router_logits_off), m, true);
        }

        const use_indirect_moe = (layer.cold_expert_count == 0) and
            (!self.enable_router_trace or self.router_trace_gpu_off > 0) and
            (layer.indirect_moe != null) and
            (self.ink_moe_ws != null);

        if (use_indirect_moe) {
            // Zero-CPU-readback `G29` indirect Softcap MoE (`cold == 0`, 0 mid-pass flushes!).
            const f = try ensureFrame(model, frame_opt);
            const im = layer.indirect_moe.?;
            const ws = &(self.ink_moe_ws.?);
            try self.pipes.encodeInkling2MoeLayerIndirect(f, .{
                .moe_pipes = &model.moe_pipes,
                .qf_pipes = qf_ptr,
                .dequant_pipes = &model.dequant_pipes,
                .x = model.x_norm_bind,
                .router = .{ .logits = .{
                    .binding = self.bindAt(self.router_logits_off),
                    .f16 = true,
                } },
                .bias = if (m82c.use_gate_bias) layer.router_bias_f32 else null,
                .bias_f16 = false,
                .bias_add_to_logits = false,
                .scoring_sigmoid = true,
                .norm_topk_prob = cfg.norm_topk_prob,
                .route_scale = m82c.route_scale,
                .router_softcap = 0.0,
                .gate_softcap = m82c.mlp_gate_softcap,
                .up_softcap = m82c.mlp_up_softcap,
                .pair_interleaved = im.pair_interleaved,
                .bucket_pair_interleaved = im.bucket_pair_interleaved,
                .weights_base = im.weights_base,
                .expert_table = im.expert_table,
                .workspace = ws,
                .workspace_buf = self.buf,
                .is_heterogeneous = im.is_heterogeneous,
                .expert_schemes = im.expert_schemes,
                .scheme_counts = im.scheme_counts,
                .scheme_descs = im.scheme_descs,
                .residual = null,
                .out = model.proj_out_bind,
                .m = m,
                .num_experts = cfg.num_experts,
                .top_k = top_k,
                .hidden_size = h,
                .intermediate_size = inter,
                .act_quant_policy = model.act_quant_policy,
                .act_scratch = act_b,
                .act_scratch_bytes = model.act_quant_bytes,
                .router_tile_f16 = tile_b,
                .router_tile_f16_bytes = model.qf_tile_bytes,
            }, true);
            if (self.enable_router_trace and m == 1 and eff_step < self.router_trace_valid.len and eff_step < 64 and top_k <= self.router_trace_top_k and self.router_trace_gpu_off > 0) {
                const dst_off = self.router_trace_gpu_off + @as(u64, eff_step) * @as(u64, self.router_trace_top_k) * @sizeOf(u32);
                try model.moe_pipes.encodeCopyStrided(f, .{
                    .src = .{ .buffer = self.buf, .offset = ws.scratch_topk_experts_off },
                    .dst = self.bindAt(dst_off),
                    .num_rows = 1,
                    .copy_cols = top_k * 2,
                    .src_stride = top_k * 2,
                    .dst_stride = top_k * 2,
                }, false);
                self.router_trace_valid[eff_step] = true;
                self.router_trace_gpu_mask |= (@as(u64, 1) << @intCast(eff_step));
            }
            self.stats.inkling2_indirect_moe_dispatches += 1;
        } else {
            // Router top-k + CPU readback path for `cold > 0` (with windowed async NVMe prefetch
            // overlapping hot GPU execution + grouped `M_e`-row dispatch) or `W8` router trace capture.
            {
                const f = try ensureFrame(model, frame_opt);
                try self.pipes.encodeInkling2MoeRoute(f, .{
                    .logits = self.bindAt(self.router_logits_off),
                    .logits_f16 = true,
                    .bias = if (m82c.use_gate_bias) layer.router_bias_f32 else null,
                    .bias_f16 = false,
                    .bias_add_to_logits = false,
                    .scoring_sigmoid = true,
                    .topk_experts = self.bindAt(self.topk_ids_off),
                    .topk_weights = self.bindAt(self.topk_w_off),
                    .m = m,
                    .num_experts = cfg.num_experts,
                    .top_k = top_k,
                    .norm_topk_prob = cfg.norm_topk_prob,
                    .route_scale = m82c.route_scale,
                    .router_softcap = 0.0,
                }, true);
            }

            try self.flushMidPass(model, frame_opt);
            const topk_ids = self.sliceU32(self.topk_ids_off, @as(usize, m) * top_k);

            if (self.enable_router_trace and m == 1 and eff_step < self.router_trace_valid.len and top_k <= self.router_trace_top_k) {
                const tk_us: usize = self.router_trace_top_k;
                @memcpy(self.router_trace_topks[@as(usize, eff_step) * tk_us ..][0..top_k], topk_ids[0..top_k]);
                self.router_trace_valid[eff_step] = true;
            }

            // Deduplicate active experts across all `m * top_k` assignments and partition into hot vs cold,
            // packing disjoint `(tok_idx, dst_slot)` slices so each active expert streams ONCE per step (`M_e` rows).
            const total_assign = @as(usize, m) * top_k;
            const tok_idx_all = self.sliceU32(self.exp_tok_idx_off, total_assign);
            const dst_slot_all = self.sliceU32(self.exp_dst_slot_off, total_assign);
            var unique_eids: [256]u32 = undefined;
            var unique_cnt: usize = 0;
            for (0..total_assign) |ai| {
                const eid = topk_ids[ai];
                if (eid >= layer.experts.len) return error.InvalidArgument;
                var seen = false;
                for (unique_eids[0..unique_cnt]) |u| {
                    if (u == eid) {
                        seen = true;
                        break;
                    }
                }
                if (!seen and unique_cnt < unique_eids.len) {
                    unique_eids[unique_cnt] = eid;
                    unique_cnt += 1;
                }
            }

            const GroupedAssign = struct {
                eid: u32,
                start: u32,
                count: u32,
                single_tok: u32,
                single_slot: u32,
            };
            var hot_list: [256]GroupedAssign = undefined;
            var hot_cnt: usize = 0;
            var cold_list: [256]GroupedAssign = undefined;
            var cold_cnt: usize = 0;
            var cursor: u32 = 0;
            for (unique_eids[0..unique_cnt]) |eid| {
                const start = cursor;
                var single_tok: u32 = 0;
                var single_slot: u32 = 0;
                for (0..m) |tok| {
                    for (0..top_k) |ki| {
                        const slot_idx: u32 = @intCast(tok * top_k + ki);
                        if (topk_ids[slot_idx] != eid) continue;
                        tok_idx_all[cursor] = @intCast(tok);
                        dst_slot_all[cursor] = slot_idx;
                        single_tok = @intCast(tok);
                        single_slot = slot_idx;
                        cursor += 1;
                    }
                }
                const ga: GroupedAssign = .{
                    .eid = eid,
                    .start = start,
                    .count = cursor - start,
                    .single_tok = single_tok,
                    .single_slot = single_slot,
                };
                switch (layer.experts[eid]) {
                    .hot => {
                        hot_list[hot_cnt] = ga;
                        hot_cnt += 1;
                    },
                    .cold => {
                        cold_list[cold_cnt] = ga;
                        cold_cnt += 1;
                    },
                }
            }

            const x_row_bytes: u64 = @as(u64, h) * 2;
            const encodeOneGrouped = struct {
                fn run(
                    rt_s: *M82Runtime,
                    m_ptr: *graph.GraphModel,
                    f: *metal.Compute.Frame,
                    ga: GroupedAssign,
                    resolved: ColdSlotBindings,
                    h_dim: u32,
                    inter_dim: u32,
                    row_bytes: u64,
                    gate_sc: f32,
                    up_sc: f32,
                    qfp: ?*const quant_formats.QuantFormatPipelines,
                    tile_opt: ?metal.BufferBinding,
                    act_opt: ?metal.BufferBinding,
                ) graph.GraphStepError!void {
                    if (ga.count == 0) return;
                    const single_out_bind = rt_s.bindAt(rt_s.exp_out_off + @as(u64, ga.single_slot) * row_bytes);
                    try rt_s.pipes.encodeInkling2GroupedExpertMlp(f, .{
                        .dequant_pipes = &m_ptr.dequant_pipes,
                        .qf_pipes = qfp,
                        .x = m_ptr.x_norm_bind,
                        .w1 = linearToInkling2Desc(resolved.w1),
                        .w_up = if (resolved.w_up) |wu| linearToInkling2Desc(wu) else null,
                        .w2 = linearToInkling2Desc(resolved.w2),
                        .m_e = ga.count,
                        .hidden_size = h_dim,
                        .intermediate_size = inter_dim,
                        .single_token_idx = ga.single_tok,
                        .token_indices_dev = if (ga.count > 1) rt_s.bindAt(rt_s.exp_tok_idx_off + @as(u64, ga.start) * 4) else null,
                        .dst_slots_or_tok_dev = if (ga.count > 1) rt_s.bindAt(rt_s.exp_dst_slot_off + @as(u64, ga.start) * 4) else null,
                        .slot_k_indices_dev = null,
                        .slot_stride = 0,
                        .gathered_x = if (ga.count > 1) rt_s.bindAt(rt_s.exp_gathered_x_off) else null,
                        .gate_up_buf = rt_s.bindAt(rt_s.exp_gate_up_off),
                        .up_buf = if (resolved.w_up != null) rt_s.bindAt(rt_s.exp_up_off) else null,
                        .act_buf = rt_s.bindAt(rt_s.exp_act_off),
                        .down_tmp_buf = if (ga.count > 1) rt_s.bindAt(rt_s.exp_down_tmp_off) else null,
                        .expert_out = if (ga.count == 1) single_out_bind else rt_s.bindAt(rt_s.exp_out_off),
                        .gate_softcap = gate_sc,
                        .up_softcap = up_sc,
                        .tile_f16 = tile_opt,
                        .tile_f16_bytes = m_ptr.qf_tile_bytes,
                        .act_quant_policy = m_ptr.act_quant_policy,
                        .act_scratch = act_opt,
                        .act_scratch_bytes = m_ptr.act_quant_bytes,
                    }, true);
                    rt_s.stats.inkling2_grouped_moe_dispatches += 1;
                }
            }.run;

            if (cold_cnt == 0) {
                const f = try ensureFrame(model, frame_opt);
                for (hot_list[0..hot_cnt]) |ga| {
                    const hw = layer.experts[ga.eid].hot;
                    try encodeOneGrouped(self, model, f, ga, .{ .w1 = hw.w1, .w_up = hw.w_up, .w2 = hw.w2 }, h, inter, x_row_bytes, m82c.mlp_gate_softcap, m82c.mlp_up_softcap, qf_ptr, tile_b, act_b);
                }
            } else {
                // Windowed async cold-expert prefetch (`W = min(remaining_cold, cold_cache.num_slots)`)
                // submitting window 0's `F_NOCACHE` reads BEFORE encoding hot experts on GPU so NVMe I/O
                // overlaps hot GPU compute, while preventing `ColdCache` slot overwrite across windows!
                const cc = &(self.cold_cache.?);
                const max_win: usize = @max(@as(usize, 1), @min(@as(usize, cc.num_slots), 256));
                var win_metas: [256]M82ColdExpertMeta = undefined;
                var win_binds: [256]ColdSlotBindings = undefined;

                const w0_len: usize = @min(cold_cnt, max_win);
                for (0..w0_len) |wi| {
                    win_metas[wi] = layer.experts[cold_list[wi].eid].cold;
                }
                var w0_handle = cc.beginPrefetchWindow(win_metas[0..w0_len], win_binds[0..w0_len], &self.stats) catch return error.IoFailed;

                // Encode hot experts and submit asynchronously while window 0 NVMe reads complete!
                var hot_sig_opt: ?u64 = null;
                if (hot_cnt > 0) {
                    const hf = try ensureFrame(model, frame_opt);
                    for (hot_list[0..hot_cnt]) |ga| {
                        const hw = layer.experts[ga.eid].hot;
                        try encodeOneGrouped(self, model, hf, ga, .{ .w1 = hw.w1, .w_up = hw.w_up, .w2 = hw.w2 }, h, inter, x_row_bytes, m82c.mlp_gate_softcap, m82c.mlp_up_softcap, qf_ptr, tile_b, act_b);
                    }
                    hot_sig_opt = try hf.commit();
                    frame_opt.* = null;
                    self.stats.mid_pass_flushes += 1;
                }

                // Wait for window 0 NVMe reads (overlapping with GPU hot-expert execution above!),
                // then ensure hot GPU scratch is complete before dispatching window 0 cold experts.
                cc.finishPrefetchWindow(&w0_handle, &self.stats) catch return error.IoFailed;
                if (hot_sig_opt) |hsig| {
                    try model.compute.wait(hsig, model.timeout_ms);
                }

                {
                    const cf = try ensureFrame(model, frame_opt);
                    for (0..w0_len) |wi| {
                        try encodeOneGrouped(self, model, cf, cold_list[wi], win_binds[wi], h, inter, x_row_bytes, m82c.mlp_gate_softcap, m82c.mlp_up_softcap, qf_ptr, tile_b, act_b);
                    }
                }

                // Remaining cold windows (when `cold_cnt > cc.num_slots`): flush previous window's GPU
                // command buffer before reusing `ColdCache` slots so no in-flight kernel reads overwritten weights.
                var w_start: usize = w0_len;
                while (w_start < cold_cnt) {
                    try self.flushMidPass(model, frame_opt);
                    const w_len: usize = @min(cold_cnt - w_start, max_win);
                    for (0..w_len) |wi| {
                        win_metas[wi] = layer.experts[cold_list[w_start + wi].eid].cold;
                    }
                    var wh = cc.beginPrefetchWindow(win_metas[0..w_len], win_binds[0..w_len], &self.stats) catch return error.IoFailed;
                    cc.finishPrefetchWindow(&wh, &self.stats) catch return error.IoFailed;
                    const cf = try ensureFrame(model, frame_opt);
                    for (0..w_len) |wi| {
                        try encodeOneGrouped(self, model, cf, cold_list[w_start + wi], win_binds[wi], h, inter, x_row_bytes, m82c.mlp_gate_softcap, m82c.mlp_up_softcap, qf_ptr, tile_b, act_b);
                    }
                    w_start += w_len;
                }
            }

            // `G25` MoE weighted combine -> `model.proj_out_bind`
            {
                const f = try ensureFrame(model, frame_opt);
                try self.pipes.encodeInkling2MoeEpilogue(f, .{
                    .expert_out = self.bindAt(self.exp_out_off),
                    .topk_weights = self.bindAt(self.topk_w_off),
                    .weights_f16 = false,
                    .residual = null,
                    .out = model.proj_out_bind,
                    .m = m,
                    .hidden_dim = h,
                    .top_k = top_k,
                    .mlp_softcap = 0.0,
                }, true);
            }
        }

        // Optional post-MLP `mlp_sconv` (`/ sqrt(5)`) -> residual add (`cur = ffn_inp + mlp_out`).
        {
            const f = try ensureFrame(model, frame_opt);
            if (layer.mlp_sconv_w) |sw| {
                const sring = &(self.sconv_h_ring.?);
                const state_b = try sring.layerBinding(step_st.sconv_mlp_slot, self.buf);
                if (!commit_state) {
                    try model.moe_pipes.encodeCopyStrided(f, .{
                        .src = model.proj_out_bind,
                        .dst = self.bindAt(step_st.tree_mlp_sconv_in_off),
                        .num_rows = m,
                        .copy_cols = h,
                        .src_stride = h,
                        .dst_stride = h,
                    }, true);
                }
                try self.pipes.encodeRmsNormSconv4(f, .{
                    .x = if (!commit_state) self.bindAt(step_st.tree_mlp_sconv_in_off) else model.proj_out_bind,
                    .conv_w = sw,
                    .conv_w_f16 = false,
                    .conv_state_in = state_b,
                    .conv_state_out = state_b,
                    .out = model.proj_out_bind,
                    .ancestor_masks = if (!commit_state) model.ancestor_masks_bind else null,
                    .m = m,
                    .hidden_dim = h,
                    .state_pos = self.sconv_pos,
                    .write_state = commit_state,
                    .apply_rms_norm = false,
                    .apply_silu = false,
                    .post_residual_scale = 1.0 / @sqrt(@as(f32, 5.0)),
                }, true);
                self.stats.inkling2_sconv_dispatches += 1;
            }

            try model.moe_pipes.encodeResidualAdd(f, .{
                .x = model.x_bind,
                .delta = model.proj_out_bind,
                .count = m * h,
            }, true);
        }
    }

    fn applyInkling2KvSconv(
        self: *M82Runtime,
        model: *graph.GraphModel,
        frame_opt: *?metal.Compute.Frame,
        layer: *const Inkling2LayerWeights,
        step_st: *const Inkling2StepState,
        is_k: bool,
        kv_in_bind: metal.BufferBinding,
        kv_bind: metal.BufferBinding,
        tree_in_off: u64,
        m: u32,
        kv_dim: u32,
        commit_state: bool,
    ) graph.GraphStepError!void {
        const sring = &(self.sconv_kv_ring.?);
        const slot = if (is_k) step_st.sconv_k_slot else step_st.sconv_v_slot;
        const state_b = try sring.layerBinding(slot, self.buf);
        const dyn_proj = if (is_k) layer.k_sconv_dyn_proj else layer.v_sconv_dyn_proj;
        const static_w = if (is_k) layer.k_sconv_w else layer.v_sconv_w;

        if (!commit_state) {
            const f = try ensureFrame(model, frame_opt);
            try model.moe_pipes.encodeCopyStrided(f, .{
                .src = kv_bind,
                .dst = self.bindAt(tree_in_off),
                .num_rows = m,
                .copy_cols = kv_dim,
                .src_stride = kv_dim,
                .dst_stride = kv_dim,
            }, true);
        }

        if (dyn_proj) |dp| {
            // Global layer `Inkling2LoopDynamicShortConvolution` on GPU (`G29`, `0` CPU flush):
            // `weight_proj` outputs lag `0..3` (lag 0 = current token), reversed to `[t-3..t]` on GPU, scaled by `1 / 3`.
            const f = try ensureFrame(model, frame_opt);
            try model.encodeLinearBatched(f, dp, kv_in_bind, self.bindAt(self.dyn_sconv_proj_off), m, true);
            try self.pipes.encodeInkling2DynSconvAssemble(f, .{
                .proj = self.bindAt(self.dyn_sconv_proj_off),
                .static_w = static_w,
                .static_w_f16 = false,
                .dyn_w = self.bindAt(self.dyn_sconv_w_off),
                .m = m,
                .kv_dim = kv_dim,
                .num_kv_heads = model.config.num_key_value_heads,
                .head_dim = model.config.head_dim,
                .proj_n = dp.n,
                .weight_scale = 1.0,
            }, true);
            try self.pipes.encodeRmsNormSconv4(f, .{
                .x = if (!commit_state) self.bindAt(tree_in_off) else kv_bind,
                .dyn_w = self.bindAt(self.dyn_sconv_w_off),
                .conv_state_in = state_b,
                .conv_state_out = state_b,
                .out = kv_bind,
                .ancestor_masks = if (!commit_state) model.ancestor_masks_bind else null,
                .m = m,
                .hidden_dim = kv_dim,
                .state_pos = self.sconv_pos,
                .write_state = commit_state,
                .is_dynamic = true,
                .apply_rms_norm = false,
                .apply_silu = false,
                .post_residual_scale = 1.0 / 3.0,
            }, true);
            self.stats.inkling2_sconv_dispatches += 1;
        } else if (static_w) |sw| {
            const f = try ensureFrame(model, frame_opt);
            try self.pipes.encodeRmsNormSconv4(f, .{
                .x = if (!commit_state) self.bindAt(tree_in_off) else kv_bind,
                .conv_w = sw,
                .conv_w_f16 = false,
                .conv_state_in = state_b,
                .conv_state_out = state_b,
                .out = kv_bind,
                .ancestor_masks = if (!commit_state) model.ancestor_masks_bind else null,
                .m = m,
                .hidden_dim = kv_dim,
                .state_pos = self.sconv_pos,
                .write_state = commit_state,
                .is_dynamic = false,
                .apply_rms_norm = false,
                .apply_silu = false,
                .post_residual_scale = 1.0 / @sqrt(@as(f32, 5.0)),
            }, true);
            self.stats.inkling2_sconv_dispatches += 1;
        }
    }

    fn applySme2MessageGate(
        self: *M82Runtime,
        ctrl: *ink_ctrl.Inkling2LoopController,
        branch: ink_ctrl.ControllerBranch,
        cur_f16: []const f16,
        msg_f32: []const f32,
        out_f16: []f16,
        m: u32,
        dim: u32,
    ) graph.GraphStepError!void {
        const mn: usize = @as(usize, m) * dim;
        f16SliceToF32(cur_f16[0..mn], self.ctrl_val_f32[0..mn]);
        const disp = &(self.sme_dispatcher.?);
        ctrl.stepMessageGate(
            disp,
            branch,
            self.ctrl_val_f32[0..mn],
            msg_f32[0..mn],
            self.ctrl_out_f32[0..mn],
            m,
            null,
        ) catch return error.InvalidArgument;
        f32SliceToF16(self.ctrl_out_f32[0..mn], out_f16[0..mn]);
        self.stats.inkling2_sme2_ctrl_calls += 1;
    }

    fn flushMidPass(self: *M82Runtime, model: *graph.GraphModel, frame_opt: *?metal.Compute.Frame) graph.GraphStepError!void {
        if (frame_opt.* != null) {
            try flushFrame(model, frame_opt);
            self.stats.mid_pass_flushes += 1;
        }
    }

    fn maybeCaptureDFlareTap(
        self: *M82Runtime,
        model: *graph.GraphModel,
        frame_opt: *?metal.Compute.Frame,
        visit_idx: u32,
        m: u32,
        src_bind: metal.BufferBinding,
    ) graph.GraphStepError!void {
        const m82c = &model.config.m82_cfg;
        if (m82c.dflare_target_layer_count == 0 or self.dflare_taps_gpu_off == 0) return;
        const h: u32 = model.config.hidden_size;
        const stride_bytes: u64 = @as(u64, self.max_tree_nodes) * h * @sizeOf(f16);
        for (0..m82c.dflare_target_layer_count) |ti| {
            if (m82c.dflare_target_layer_ids[ti] == visit_idx) {
                const f = try ensureFrame(model, frame_opt);
                const dst_off = self.dflare_taps_gpu_off + @as(u64, ti) * stride_bytes;
                try model.moe_pipes.encodeCopyStrided(f, .{
                    .src = src_bind,
                    .dst = self.bindAt(dst_off),
                    .num_rows = m,
                    .copy_cols = h,
                    .src_stride = h,
                    .dst_stride = h,
                }, true);
                self.dflare_tap_gpu_binds[ti] = self.bindAt(dst_off);
                if (ti + 1 > self.dflare_tap_count) self.dflare_tap_count = ti + 1;
            }
        }
    }

    // ------------------------------------------------------------------------
    // Hy4 (`hy_v4`, `hy_v4_vl`, `hy_v4_flash`) End-to-End Forward
    // ------------------------------------------------------------------------

    fn forwardHy4(
        self: *M82Runtime,
        model: *graph.GraphModel,
        token_ids: []const u32,
        positions: []const u32,
        ancestor_masks: []const u64,
        base_pos: u32,
        compute_logits: bool,
        commit_state: bool,
        vis_override_src: ?metal.BufferBinding,
        vis_src_row_off: u32,
        vis_local_start: u32,
        vis_count: u32,
    ) graph.GraphStepError!qwen3.VerifyTreeResult {
        const cfg = &model.config;
        const m82c = &cfg.m82_cfg;
        const m: u32 = @intCast(token_ids.len);
        const h: u32 = cfg.hidden_size;
        const hc_mult: u32 = if (m82c.enable_ihc) m82c.hc_mult else 1;

        for (0..m) |i| {
            model.token_ids_host[i] = token_ids[i];
            model.positions_host[i] = positions[i];
            model.ancestor_masks_host[i] = if (commit_state and m == 1) @as(u64, 1) else ancestor_masks[i];
        }

        if (model.ple_rt) |prt| {
            prt.scratch.resetStep();
        }

        var frame_opt: ?metal.Compute.Frame = null;
        errdefer if (frame_opt) |*f| f.abort();

        // 1. Embedding lookup + optional ViT override + iHC 4-stream init.
        {
            const f = try ensureFrame(model, &frame_opt);
            try model.encodeEmbedLookupUnified(f, .{
                .weight = model.embed.weight,
                .scale_bias = model.embed.scale_bias,
                .tokens = .{ .token_ids = model.token_ids_bind },
                .out = model.x_bind,
                .m = m,
                .vocab_size = cfg.vocab_size,
                .hidden_size = h,
            }, model.token_ids_host, model.x_host, false);

            if (vis_override_src) |vsrc| {
                const row_bytes: u64 = @as(u64, h) * @sizeOf(f16);
                try model.moe_pipes.encodeCopyStrided(f, .{
                    .src = .{ .buffer = vsrc.buffer, .offset = vsrc.offset + @as(u64, vis_src_row_off) * row_bytes },
                    .dst = .{ .buffer = model.x_bind.buffer, .offset = model.x_bind.offset + @as(u64, vis_local_start) * row_bytes },
                    .num_rows = vis_count,
                    .copy_cols = h,
                    .src_stride = h,
                    .dst_stride = h,
                }, true);
            }
        }
        try flushFrame(model, &frame_opt);

        // Initialize `hc_streams[m, hc_mult, h]` with embedded `x`.
        if (m82c.enable_ihc) {
            for (0..m) |tok| {
                const src = model.x_host[tok * h ..][0..h];
                for (0..hc_mult) |s| {
                    @memcpy(self.hy4_hc_streams_host[(tok * hc_mult + s) * h ..][0..h], src);
                }
            }
        }

        // 2. Execute all Hy4 layers `0 .. num_hidden_layers`.
        var last_full_idx_layer: ?usize = null;
        for (0..cfg.num_hidden_layers) |l| {
            try self.execHy4Layer(
                model,
                &frame_opt,
                l,
                token_ids,
                positions,
                ancestor_masks,
                base_pos,
                m,
                hc_mult,
                commit_state,
                &last_full_idx_layer,
            );
        }

        // 3. Final iHC head collapse (`hc_head.*`) -> `x_bind`.
        try flushFrame(model, &frame_opt);
        if (m82c.enable_ihc) {
            if (self.hy4_hc_head) |hh| {
                applyHy4HcHead(self.hy4_hc_streams_host, model.x_host, hh, m, hc_mult, h, m82c.hc_eps, self.hy4_hc_tmp_f32);
            } else {
                const inv_s: f32 = 1.0 / @as(f32, @floatFromInt(hc_mult));
                for (0..m) |tok| {
                    for (0..h) |c| {
                        var sum: f32 = 0.0;
                        for (0..hc_mult) |s| {
                            sum += @as(f32, @floatCast(self.hy4_hc_streams_host[(tok * hc_mult + s) * h + c]));
                        }
                        model.x_host[tok * h + c] = @floatCast(sum * inv_s);
                    }
                }
            }
        }

        // Capture final hidden state & DFly2 target KV views.
        f16SliceToF32(model.x_host[0 .. @as(usize, m) * h], self.last_hidden_f32[0 .. @as(usize, m) * h]);
        if (commit_state) {
            self.refreshDFly2TargetKvViews(cfg, base_pos + m);
        }

        return self.finalizeLogits(model, &frame_opt, m, compute_logits, 0.0, 0, false);
    }

    fn execHy4Layer(
        self: *M82Runtime,
        model: *graph.GraphModel,
        frame_opt: *?metal.Compute.Frame,
        l: usize,
        token_ids: []const u32,
        positions: []const u32,
        ancestor_masks: []const u64,
        base_pos: u32,
        m: u32,
        hc_mult: u32,
        commit_state: bool,
        last_full_idx_layer: *?usize,
    ) graph.GraphStepError!void {
        const cfg = &model.config;
        const m82c = &cfg.m82_cfg;
        const h: u32 = cfg.hidden_size;
        const layer = &self.hy4_layers[l];
        const n_q = cfg.num_attention_heads;
        const q_rank = cfg.q_lora_rank;
        const kv_rank = cfg.kv_lora_rank;
        const qk_nope = cfg.qk_nope_head_dim;
        const qk_rope = cfg.qk_rope_head_dim;
        const v_dim = cfg.v_head_dim;

        // --- A. Optional C26 `fp8_e4m3_row` Engram `F_NOCACHE` lookup (`0` r>=1 reads) ---
        if (layer.engram) |eg| {
            if (model.ple_rt) |prt| {
                try flushFrame(model, frame_opt);
                try self.execHy4Engram(model, frame_opt, prt, l, eg, token_ids, ancestor_masks, base_pos, m, hc_mult, h);
            }
        }

        // --- B. iHC Pre-Attention Stream Mix + `input_layernorm` ---
        var attn_post_weights: [dflare_dfly.MAX_BLOCK_SIZE * 8]f32 = undefined;
        if (m82c.enable_ihc) {
            try flushFrame(model, frame_opt);
            if (layer.hc_attn) |hca| {
                applyHy4HcPre(
                    self.hy4_hc_streams_host,
                    model.x_host,
                    hca,
                    m,
                    hc_mult,
                    h,
                    m82c.hc_eps,
                    m82c.hc_magnitude,
                    self.hy4_hc_tmp_f32,
                    attn_post_weights[0 .. @as(usize, m) * hc_mult],
                );
            } else {
                const inv_s: f32 = 1.0 / @as(f32, @floatFromInt(hc_mult));
                for (0..m) |tok| {
                    for (0..hc_mult) |s| attn_post_weights[tok * hc_mult + s] = 1.0;
                    for (0..h) |c| {
                        var sum: f32 = 0.0;
                        for (0..hc_mult) |s| sum += @as(f32, @floatCast(self.hy4_hc_streams_host[(tok * hc_mult + s) * h + c]));
                        model.x_host[tok * h + c] = @floatCast(sum * inv_s);
                    }
                }
            }
        }

        // Detect causal linear chain (`positions[i] == base_pos + i` & `ancestor_masks[i] == (1<<(i+1))-1`):
        // covers 100% of prefill, single-step decode, and DFly2 linear blocks, giving exact bit-identical
        // `causal_q_start_pos = base_pos` indexer + sparse-MLA attention across every node `0..m-1`.
        var is_linear_chain = commit_state;
        if (!commit_state and (base_pos + m) <= self.max_seq_len) {
            is_linear_chain = true;
            for (0..m) |ni| {
                const exp_mask: u64 = (@as(u64, 1) << @intCast(ni + 1)) - 1;
                if (positions[ni] != base_pos + @as(u32, @intCast(ni)) or ancestor_masks[ni] != exp_mask) {
                    is_linear_chain = false;
                    break;
                }
            }
        }

        {
            const f = try ensureFrame(model, frame_opt);
            try model.attn_pipes.encodeRmsNorm(f, .{
                .x = model.x_bind,
                .w = layer.input_layernorm.binding,
                .y = model.x_norm_bind,
                .num_rows = m,
                .hidden_dim = h,
                .eps = cfg.rms_norm_eps,
                .w_dtype = layer.input_layernorm.dtype,
            }, true);

            // --- C. MLA Q & KV Projections ---
            try model.encodeLinearBatched(f, layer.q_a_proj, model.x_norm_bind, self.bindAt(self.hy4_q_a_off), m, true);
            if (model.hasActiveLora()) {
                _ = try model.encodeLoraSingle(f, l, .q_a_proj, model.x_norm_bind, self.bindAt(self.hy4_q_a_off), m, layer.q_a_proj.k, layer.q_a_proj.n, true);
            }
            if (layer.q_a_layernorm) |qan| {
                try model.attn_pipes.encodeRmsNorm(f, .{
                    .x = self.bindAt(self.hy4_q_a_off),
                    .w = qan.binding,
                    .y = self.bindAt(self.hy4_q_a_off),
                    .num_rows = m,
                    .hidden_dim = q_rank,
                    .eps = cfg.rms_norm_eps,
                    .w_dtype = qan.dtype,
                }, true);
            }
            try model.encodeLinearBatched(f, layer.q_b_proj, self.bindAt(self.hy4_q_a_off), self.bindAt(self.hy4_q_b_off), m, true);
            try model.encodeLinearBatched(f, layer.kv_a_proj, model.x_norm_bind, self.bindAt(self.hy4_kv_a_off), m, false);
            if (layer.linear_gate) |lg| {
                try model.encodeLinearBatched(f, lg, model.x_norm_bind, self.bindAt(self.hy4_lin_gate_off), m, false);
            }
            if (model.hasActiveLora()) {
                _ = try model.encodeLoraSingle(f, l, .q_b_proj, self.bindAt(self.hy4_q_a_off), self.bindAt(self.hy4_q_b_off), m, layer.q_b_proj.k, layer.q_b_proj.n, true);
                _ = try model.encodeLoraSingle(f, l, .kv_a_proj, model.x_norm_bind, self.bindAt(self.hy4_kv_a_off), m, layer.kv_a_proj.k, layer.kv_a_proj.n, true);
                if (layer.linear_gate) |lg| {
                    _ = try model.encodeLoraSingle(f, l, .gate_attn, model.x_norm_bind, self.bindAt(self.hy4_lin_gate_off), m, lg.k, lg.n, true);
                }
            }
        }

        // Split `q_b` -> `(q_nope, q_pe)`, split `kv_a` -> `(kv_latent, k_pe)`, apply RoPE,
        // and apply `kv_a_layernorm` to `kv_latent`.
        try flushFrame(model, frame_opt);
        {
            const qb = self.sliceF16(self.hy4_q_b_off, @as(usize, m) * n_q * (qk_nope + qk_rope));
            const qn_dst = self.sliceF16(self.hy4_q_nope_off, @as(usize, m) * n_q * qk_nope);
            const qp_dst = self.sliceF16(self.hy4_q_pe_off, @as(usize, m) * n_q * qk_rope);
            const q_hd = qk_nope + qk_rope;
            for (0..m) |tok| {
                for (0..n_q) |qh| {
                    const src = qb[(tok * n_q + qh) * q_hd ..];
                    @memcpy(qn_dst[(tok * n_q + qh) * qk_nope ..][0..qk_nope], src[0..qk_nope]);
                    if (qk_rope > 0) {
                        @memcpy(qp_dst[(tok * n_q + qh) * qk_rope ..][0..qk_rope], src[qk_nope .. qk_nope + qk_rope]);
                        applyRopeInPlaceF16(qp_dst[(tok * n_q + qh) * qk_rope ..][0..qk_rope], positions[tok], cfg.rope_theta);
                    }
                }
            }

            const kva = self.sliceF16(self.hy4_kv_a_off, @as(usize, m) * (kv_rank + qk_rope));
            const kl_cur = self.sliceF16(self.hy4_kv_lat_cur_off, @as(usize, m) * kv_rank);
            const kr_cur = self.sliceF16(self.hy4_kv_rope_cur_off, @as(usize, m) * qk_rope);
            for (0..m) |tok| {
                const src = kva[tok * (kv_rank + qk_rope) ..];
                @memcpy(kl_cur[tok * kv_rank ..][0..kv_rank], src[0..kv_rank]);
                if (qk_rope > 0) {
                    @memcpy(kr_cur[tok * qk_rope ..][0..qk_rope], src[kv_rank .. kv_rank + qk_rope]);
                    applyRopeInPlaceF16(kr_cur[tok * qk_rope ..][0..qk_rope], positions[tok], cfg.rope_theta);
                }
            }
        }

        // Apply `kv_a_layernorm` on GPU, absorb `q_nope @ W_uk` -> `q_abs [m, n_q, kv_rank]`,
        // and write `kv_latent` / `kv_rope` into committed cache (or tree scratch).
        {
            const f = try ensureFrame(model, frame_opt);
            if (layer.kv_a_layernorm) |kvn| {
                try model.attn_pipes.encodeRmsNorm(f, .{
                    .x = self.bindAt(self.hy4_kv_lat_cur_off),
                    .w = kvn.binding,
                    .y = self.bindAt(self.hy4_kv_lat_cur_off),
                    .num_rows = m,
                    .hidden_dim = kv_rank,
                    .eps = cfg.rms_norm_eps,
                    .w_dtype = kvn.dtype,
                }, true);
            }

            // Copy into committed MLA KV cache (for commit_state or linear-chain uncommitted tail)
            // and also into per-layer tree scratch when `!commit_state` so `commitTreeKv` is always valid.
            if (is_linear_chain) {
                try model.moe_pipes.encodeCopyStrided(f, .{
                    .src = self.bindAt(self.hy4_kv_lat_cur_off),
                    .dst = self.bindAt(layer.kv_latent_off + @as(u64, base_pos) * kv_rank * 2),
                    .num_rows = m,
                    .copy_cols = kv_rank,
                    .src_stride = kv_rank,
                    .dst_stride = kv_rank,
                }, true);
                if (qk_rope > 0) {
                    try model.moe_pipes.encodeCopyStrided(f, .{
                        .src = self.bindAt(self.hy4_kv_rope_cur_off),
                        .dst = self.bindAt(layer.kv_rope_off + @as(u64, base_pos) * qk_rope * 2),
                        .num_rows = m,
                        .copy_cols = qk_rope,
                        .src_stride = qk_rope,
                        .dst_stride = qk_rope,
                    }, false);
                }
            }
            if (!commit_state) {
                try model.moe_pipes.encodeCopyStrided(f, .{
                    .src = self.bindAt(self.hy4_kv_lat_cur_off),
                    .dst = self.bindAt(layer.tree_kv_latent_off),
                    .num_rows = m,
                    .copy_cols = kv_rank,
                    .src_stride = kv_rank,
                    .dst_stride = kv_rank,
                }, !is_linear_chain);
                if (qk_rope > 0) {
                    try model.moe_pipes.encodeCopyStrided(f, .{
                        .src = self.bindAt(self.hy4_kv_rope_cur_off),
                        .dst = self.bindAt(layer.tree_kv_rope_off),
                        .num_rows = m,
                        .copy_cols = qk_rope,
                        .src_stride = qk_rope,
                        .dst_stride = qk_rope,
                    }, false);
                }
            }

            // Absorb `q_nope` into `kv_lora_rank` per head using pre-unpacked `W_uk_t[qh] [kv_rank, qk_nope]`.
            const w_uk_head_bytes: u64 = @as(u64, kv_rank) * qk_nope * 2;
            for (0..m) |tok| {
                for (0..n_q) |qh| {
                    const qn_b = self.bindAt(self.hy4_q_nope_off + @as(u64, tok * n_q + qh) * qk_nope * 2);
                    const qa_b = self.bindAt(self.hy4_q_abs_off + @as(u64, tok * n_q + qh) * kv_rank * 2);
                    const w_b = self.bindAt(layer.w_uk_t_off + @as(u64, qh) * w_uk_head_bytes);
                    try model.dequant_pipes.encodeDenseGemv(f, .f16, .{
                        .x = qn_b,
                        .weight = w_b,
                        .y = qa_b,
                        .n = kv_rank,
                        .k = qk_nope,
                    }, tok == 0 and qh == 0);
                }
            }
        }

        // --- D. `IndexCache` (period-4 `.full`/`.shared`) & `IndexPool` (`compress_ratio = 4`) ---
        const idx_pool = &(self.hy4_index_pool.?);
        var eff_swa_window: u32 = if (layer.is_swa) m82c.sliding_window else 0;
        if (!layer.is_swa) {
            if (layer.indexer_mode == .full and layer.indexer != null) {
                const ix = layer.indexer.?;
                const idx_d = self.hy4_index_head_dim;
                const idx_h = self.hy4_index_n_heads;
                const f = try ensureFrame(model, frame_opt);
                try model.encodeLinearBatched(f, ix.q_proj, self.bindAt(self.hy4_q_a_off), self.bindAt(self.hy4_q_idx_off), m, true);
                try model.encodeLinearBatched(f, ix.k_proj, model.x_norm_bind, self.bindAt(self.hy4_k_idx_off), m, false);
                if (ix.k_norm) |kn| {
                    try model.attn_pipes.encodeRmsNorm(f, .{
                        .x = self.bindAt(self.hy4_k_idx_off),
                        .w = kn.binding,
                        .y = self.bindAt(self.hy4_k_idx_off),
                        .num_rows = m,
                        .hidden_dim = idx_d,
                        .eps = cfg.rms_norm_eps,
                        .w_dtype = kn.dtype,
                    }, true);
                }
                if (ix.head_w_proj) |hwp| {
                    try model.encodeLinearBatched(f, hwp, model.x_norm_bind, self.bindAt(self.hy4_hw_idx_off), m, false);
                }
                if (is_linear_chain) {
                    // Write `k_fine` at `base_pos..base_pos+m` (uncommitted tail is safe when `!commit_state`).
                    try model.moe_pipes.encodeCopyStrided(f, .{
                        .src = self.bindAt(self.hy4_k_idx_off),
                        .dst = self.bindAt(layer.k_fine_cache_off + @as(u64, base_pos) * idx_d * 2),
                        .num_rows = m,
                        .copy_cols = idx_d,
                        .src_stride = idx_d,
                        .dst_stride = idx_d,
                    }, true);

                    const disp = try self.pipes.encodeHy4IndexLayer(f, .{
                        .mode = .full,
                        .compress_keys = ix.is_compressed,
                        .pool_mode = .mean,
                        .q_idx = self.bindAt(self.hy4_q_idx_off),
                        .head_weights = if (ix.head_w_proj != null) self.bindAt(self.hy4_hw_idx_off) else null,
                        .k_fine = self.bindAt(layer.k_fine_cache_off),
                        .index_pool = idx_pool,
                        .index_metal_buf = self.buf,
                        .num_nodes = m,
                        .num_q_heads = idx_h,
                        .kv_len = base_pos + m,
                        .causal_q_start_pos = base_pos,
                        .score_mode = .relu_weighted_sum,
                        .swa_window = 0,
                    }, true);
                    self.stats.hy4_index_full_dispatches += disp.total();
                    self.stats.hy4_index_pool_compresses += disp.compress_dispatches;
                } else {
                    // Branching tree: compute exact per-node ancestor-path top-K in reverse order `m-1..0`
                    // so `selected_tokens[0..m]` holds every node's exact indices for `.shared` reuse!
                    try flushFrame(model, frame_opt);
                    const k_idx_src = self.sliceF16(self.hy4_k_idx_off, @as(usize, m) * idx_d);
                    const k_fine_dst = self.sliceF16(layer.k_fine_cache_off, @as(usize, self.max_seq_len) * idx_d);
                    const cand_k: usize = idx_pool.spec.candidate_topk;
                    var rev: usize = m;
                    while (rev > 0) {
                        rev -= 1;
                        const ni = rev;
                        const depth = stageAncestorRowsF16(k_fine_dst, k_idx_src, base_pos, self.max_seq_len, idx_d, ancestor_masks[ni], m);
                        const bf = try ensureFrame(model, frame_opt);
                        const disp = try self.pipes.encodeHy4IndexLayer(bf, .{
                            .mode = .full,
                            .compress_keys = ix.is_compressed,
                            .pool_mode = .mean,
                            .q_idx = self.bindAt(self.hy4_q_idx_off + @as(u64, ni) * idx_h * idx_d * 2),
                            .head_weights = if (ix.head_w_proj != null)
                                self.bindAt(self.hy4_hw_idx_off + @as(u64, ni) * idx_h * 2)
                            else
                                null,
                            .k_fine = self.bindAt(layer.k_fine_cache_off),
                            .index_pool = idx_pool,
                            .index_metal_buf = self.buf,
                            .num_nodes = 1,
                            .num_q_heads = idx_h,
                            .kv_len = base_pos + depth,
                            .causal_q_start_pos = null,
                            .score_mode = .relu_weighted_sum,
                            .swa_window = 0,
                        }, false);
                        self.stats.hy4_index_full_dispatches += disp.total();
                        self.stats.hy4_index_pool_compresses += disp.compress_dispatches;
                        try flushFrame(model, frame_opt);
                        if (ni > 0) {
                            @memcpy(
                                idx_pool.selected_tokens[ni * cand_k ..][0..cand_k],
                                idx_pool.selected_tokens[0..cand_k],
                            );
                            idx_pool.selected_counts[ni] = idx_pool.selected_counts[0];
                        }
                    }
                    // Restore tree node `k_fine` rows at `base_pos..base_pos+m` for `commitTreeKv` compaction.
                    for (0..m) |ni| {
                        const pos = base_pos + @as(u32, @intCast(ni));
                        if (pos < self.max_seq_len) {
                            @memcpy(k_fine_dst[@as(usize, pos) * idx_d ..][0..idx_d], k_idx_src[ni * idx_d ..][0..idx_d]);
                        }
                    }
                }
                last_full_idx_layer.* = l;
            } else if (layer.indexer_mode == .shared and last_full_idx_layer.* != null) {
                // `0`-dispatch `.shared` reuse of the active `IndexPoolBuffer`!
                const f = try ensureFrame(model, frame_opt);
                const disp = try self.pipes.encodeHy4IndexLayer(f, .{
                    .mode = .shared,
                    .q_idx = self.bindAt(self.hy4_q_idx_off),
                    .k_fine = self.bindAt(self.hy4_k_idx_off),
                    .index_pool = idx_pool,
                    .index_metal_buf = self.buf,
                    .num_nodes = m,
                    .num_q_heads = self.hy4_index_n_heads,
                    .kv_len = base_pos + m,
                }, false);
                std.debug.assert(disp.total() == 0);
                self.stats.hy4_index_shared_reuses += 1;
            } else {
                // Fallback dense MLA window covering full prefix
                eff_swa_window = self.max_seq_len;
            }
        }

        // --- E. Sparse MLA + `learnable_sink_param` Attention (`G25`) + `W_uv` + `linear_gate` + `o_proj` ---
        {
            const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(qk_nope + qk_rope)));
            if (is_linear_chain or layer.is_swa) {
                const use_branch_tree: bool = !is_linear_chain;
                const f = try ensureFrame(model, frame_opt);
                const kv_len_attn: u32 = if (use_branch_tree) base_pos else (base_pos + m);
                try self.pipes.encodeHy4MlaSinkAttn(f, .{
                    .q_nope = self.bindAt(self.hy4_q_abs_off),
                    .q_pe = if (qk_rope > 0) self.bindAt(self.hy4_q_pe_off) else null,
                    .kv_latent = self.bindAt(layer.kv_latent_off),
                    .kv_rope = if (qk_rope > 0) self.bindAt(layer.kv_rope_off) else null,
                    .index_pool = idx_pool,
                    .index_metal_buf = self.buf,
                    .kv_tree_latent = if (use_branch_tree) self.bindAt(layer.tree_kv_latent_off) else null,
                    .kv_tree_rope = if (qk_rope > 0 and use_branch_tree) self.bindAt(layer.tree_kv_rope_off) else null,
                    .ancestor_masks = if (use_branch_tree) model.ancestor_masks_bind else null,
                    .sink_params = if (m82c.learnable_sink) layer.sink_params_f32 else null,
                    .linear_gate = null, // Applied in `n_q * v_dim` space right after `W_uv`!
                    .out = self.bindAt(self.hy4_mla_out_lat_off),
                    .num_nodes = m,
                    .num_q_heads = n_q,
                    .kv_lora_rank = kv_rank,
                    .qk_rope_dim = qk_rope,
                    .kv_len = kv_len_attn,
                    .causal_q_start_pos = if (use_branch_tree) null else base_pos,
                    .swa_window = eff_swa_window,
                    .scale = scale,
                }, true);
            } else {
                // Branching tree sparse MLA: run each node against its exact staged ancestor path
                // (`has_tree = 0`), swapping `selected_tokens[0 <-> ni]` to preserve `.shared` rows!
                try flushFrame(model, frame_opt);
                const tree_lat = self.sliceF16(layer.tree_kv_latent_off, @as(usize, self.max_tree_nodes) * kv_rank);
                const kv_lat = self.sliceF16(layer.kv_latent_off, @as(usize, self.max_seq_len) * kv_rank);
                const tree_rope: ?[]f16 = if (qk_rope > 0) self.sliceF16(layer.tree_kv_rope_off, @as(usize, self.max_tree_nodes) * qk_rope) else null;
                const kv_rope: ?[]f16 = if (qk_rope > 0) self.sliceF16(layer.kv_rope_off, @as(usize, self.max_seq_len) * qk_rope) else null;
                const cand_k: usize = idx_pool.spec.candidate_topk;

                for (0..m) |ni| {
                    const depth = stageAncestorRowsF16(kv_lat, tree_lat, base_pos, self.max_seq_len, kv_rank, ancestor_masks[ni], m);
                    if (qk_rope > 0) {
                        _ = stageAncestorRowsF16(kv_rope.?, tree_rope.?, base_pos, self.max_seq_len, qk_rope, ancestor_masks[ni], m);
                    }
                    if (ni > 0) {
                        swapSelectedRow(idx_pool, 0, ni, cand_k);
                    }
                    const bf = try ensureFrame(model, frame_opt);
                    try self.pipes.encodeHy4MlaSinkAttn(bf, .{
                        .q_nope = self.bindAt(self.hy4_q_abs_off + @as(u64, ni) * n_q * kv_rank * 2),
                        .q_pe = if (qk_rope > 0) self.bindAt(self.hy4_q_pe_off + @as(u64, ni) * n_q * qk_rope * 2) else null,
                        .kv_latent = self.bindAt(layer.kv_latent_off),
                        .kv_rope = if (qk_rope > 0) self.bindAt(layer.kv_rope_off) else null,
                        .index_pool = idx_pool,
                        .index_metal_buf = self.buf,
                        .kv_tree_latent = null,
                        .kv_tree_rope = null,
                        .ancestor_masks = null,
                        .sink_params = if (m82c.learnable_sink) layer.sink_params_f32 else null,
                        .linear_gate = null,
                        .out = self.bindAt(self.hy4_mla_out_lat_off + @as(u64, ni) * n_q * kv_rank * 2),
                        .num_nodes = 1,
                        .num_q_heads = n_q,
                        .kv_lora_rank = kv_rank,
                        .qk_rope_dim = qk_rope,
                        .kv_len = base_pos + depth,
                        .causal_q_start_pos = null,
                        .swa_window = eff_swa_window,
                        .scale = scale,
                    }, false);
                    try flushFrame(model, frame_opt);
                    if (ni > 0) {
                        swapSelectedRow(idx_pool, 0, ni, cand_k);
                    }
                }
            }

            // Un-absorb `out_latent @ W_uv^T` -> `attn_v [m, n_q * v_dim]`.
            const f = try ensureFrame(model, frame_opt);
            const w_uv_head_bytes: u64 = @as(u64, v_dim) * kv_rank * 2;
            for (0..m) |tok| {
                for (0..n_q) |qh| {
                    const ol_b = self.bindAt(self.hy4_mla_out_lat_off + @as(u64, tok * n_q + qh) * kv_rank * 2);
                    const av_b = self.bindAt(self.hy4_attn_v_off + @as(u64, tok * n_q + qh) * v_dim * 2);
                    const w_b = self.bindAt(layer.w_uv_off + @as(u64, qh) * w_uv_head_bytes);
                    try model.dequant_pipes.encodeDenseGemv(f, .f16, .{
                        .x = ol_b,
                        .weight = w_b,
                        .y = av_b,
                        .n = v_dim,
                        .k = kv_rank,
                    }, tok == 0 and qh == 0);
                }
            }
        }

        // Apply `linear_gate` (`sigmoid(gate) * attn_v`) if present, then `o_proj` + optional `post_proj_norm`.
        if (layer.linear_gate != null and m82c.gated_mla) {
            try flushFrame(model, frame_opt);
            const total_v: usize = @as(usize, m) * n_q * v_dim;
            const av = self.sliceF16(self.hy4_attn_v_off, total_v);
            const lg = self.sliceF16(self.hy4_lin_gate_off, total_v);
            for (0..total_v) |i| {
                const gv: f32 = @floatCast(lg[i]);
                const sig: f32 = 1.0 / (1.0 + @exp(-std.math.clamp(gv, -20.0, 20.0)));
                av[i] = @floatCast(@as(f32, @floatCast(av[i])) * sig);
            }
        }
        {
            const f = try ensureFrame(model, frame_opt);
            try model.encodeLinearBatched(f, layer.o_proj, self.bindAt(self.hy4_attn_v_off), model.proj_out_bind, m, true);
            if (model.hasActiveLora()) {
                _ = try model.encodeLoraSingle(f, l, .o_proj, self.bindAt(self.hy4_attn_v_off), model.proj_out_bind, m, layer.o_proj.k, layer.o_proj.n, true);
            }
            if (layer.post_proj_norm) |ppn| {
                try model.attn_pipes.encodeRmsNorm(f, .{
                    .x = model.proj_out_bind,
                    .w = ppn.binding,
                    .y = model.proj_out_bind,
                    .num_rows = m,
                    .hidden_dim = h,
                    .eps = cfg.rms_norm_eps,
                    .w_dtype = ppn.dtype,
                }, true);
            }
        }

        // --- F. iHC Post-Attention Stream Update + Pre-MLP Stream Mix ---
        var mlp_post_weights: [dflare_dfly.MAX_BLOCK_SIZE * 8]f32 = undefined;
        if (m82c.enable_ihc) {
            try flushFrame(model, frame_opt);
            applyHy4HcPost(self.hy4_hc_streams_host, model.proj_out_host, attn_post_weights[0 .. @as(usize, m) * hc_mult], m, hc_mult, h);
            if (layer.hc_mlp) |hcm| {
                applyHy4HcPre(
                    self.hy4_hc_streams_host,
                    model.x_host,
                    hcm,
                    m,
                    hc_mult,
                    h,
                    m82c.hc_eps,
                    m82c.hc_magnitude,
                    self.hy4_hc_tmp_f32,
                    mlp_post_weights[0 .. @as(usize, m) * hc_mult],
                );
            } else {
                const inv_s: f32 = 1.0 / @as(f32, @floatFromInt(hc_mult));
                for (0..m) |tok| {
                    for (0..hc_mult) |s| mlp_post_weights[tok * hc_mult + s] = 1.0;
                    for (0..h) |c| {
                        var sum: f32 = 0.0;
                        for (0..hc_mult) |s| sum += @as(f32, @floatCast(self.hy4_hc_streams_host[(tok * hc_mult + s) * h + c]));
                        model.x_host[tok * h + c] = @floatCast(sum * inv_s);
                    }
                }
            }
        } else {
            const f = try ensureFrame(model, frame_opt);
            try model.moe_pipes.encodeResidualAdd(f, .{
                .x = model.x_bind,
                .delta = model.proj_out_bind,
                .count = m * h,
            }, true);
        }

        // --- G. Post-Attention RMSNorm + Dense / MoE / LatentMoE (`G25`) ---
        {
            const f = try ensureFrame(model, frame_opt);
            try model.attn_pipes.encodeRmsNorm(f, .{
                .x = model.x_bind,
                .w = layer.post_attention_layernorm.binding,
                .y = model.x_norm_bind,
                .num_rows = m,
                .hidden_dim = h,
                .eps = cfg.rms_norm_eps,
                .w_dtype = layer.post_attention_layernorm.dtype,
            }, true);
        }

        if (!layer.is_sparse_mlp) {
            // Dense clamped SwiGLU (`swiglu_limit = 10.0`) + optional `post_fc2_norm`.
            const inter = cfg.intermediate_size;
            const f = try ensureFrame(model, frame_opt);
            try model.encodeLinearBatched(f, layer.dense_gate.?, model.x_norm_bind, model.dense_gate_bind, m, true);
            try model.encodeLinearBatched(f, layer.dense_up.?, model.x_norm_bind, model.dense_up_bind, m, false);
            if (model.hasActiveLora()) {
                _ = try model.encodeLoraSingle(f, l, .gate_proj, model.x_norm_bind, model.dense_gate_bind, m, layer.dense_gate.?.k, layer.dense_gate.?.n, true);
                _ = try model.encodeLoraSingle(f, l, .up_proj, model.x_norm_bind, model.dense_up_bind, m, layer.dense_up.?.k, layer.dense_up.?.n, true);
            }
            try self.pipes.encodeHy4SwigluClampedLinearGate(f, .{
                .gate_in = model.dense_gate_bind,
                .up_in = model.dense_up_bind,
                .out = model.dense_act_bind,
                .num_rows = m,
                .intermediate_size = inter,
                .interleaved_gate_up = false,
                .clamp_mode = .gate_and_up,
                .swiglu_limit = m82c.swiglu_limit,
            }, true);
            try model.encodeLinearBatched(f, layer.dense_down.?, model.dense_act_bind, model.proj_out_bind, m, true);
            if (model.hasActiveLora()) {
                _ = try model.encodeLoraSingle(f, l, .down_proj, model.dense_act_bind, model.proj_out_bind, m, layer.dense_down.?.k, layer.dense_down.?.n, true);
            }
            if (layer.post_fc2_norm) |pfn| {
                try model.attn_pipes.encodeRmsNorm(f, .{
                    .x = model.proj_out_bind,
                    .w = pfn.binding,
                    .y = model.proj_out_bind,
                    .num_rows = m,
                    .hidden_dim = h,
                    .eps = cfg.rms_norm_eps,
                    .w_dtype = pfn.dtype,
                }, true);
            }
        } else {
            // Sparse MoE or `LatentMoE` (`routed_expert_{down,up}_proj` + `combine_output_norm`).
            const is_latent = (layer.latent_down_proj != null and layer.latent_up_proj != null);
            const exp_dim: u32 = if (is_latent) layer.latent_dim else h;
            const top_k = cfg.num_experts_per_tok;
            const moe_inter = cfg.moe_intermediate_size;

            {
                const f = try ensureFrame(model, frame_opt);
                try model.encodeLinearBatched(f, layer.router.?, model.x_norm_bind, self.bindAt(self.router_logits_off), m, true);
                try self.pipes.encodeHy4LatentMoeRoute(f, .{
                    .logits = self.bindAt(self.router_logits_off),
                    .logits_f16 = true,
                    .e_score_correction_bias = layer.e_score_bias_f32,
                    .bias_f16 = false,
                    .topk_experts = self.bindAt(self.topk_ids_off),
                    .topk_weights = self.bindAt(self.topk_w_off),
                    .m = m,
                    .num_experts = cfg.num_experts,
                    .top_k = top_k,
                    .norm_topk_prob = cfg.norm_topk_prob,
                    .routed_scaling_factor = m82c.routed_scaling_factor,
                }, true);

                if (is_latent) {
                    try model.encodeLinearBatched(f, layer.latent_down_proj.?, model.x_norm_bind, self.bindAt(self.hy4_latent_x_off), m, false);
                    self.stats.hy4_latent_moe_dispatches += 1;
                }
                // Shared expert (if present) runs on full `x_norm_bind`.
                if (layer.shared_gate) |sg| {
                    const sh_inter = sg.n;
                    try model.encodeLinearBatched(f, sg, model.x_norm_bind, model.dense_gate_bind, m, false);
                    try model.encodeLinearBatched(f, layer.shared_up.?, model.x_norm_bind, model.dense_up_bind, m, false);
                    try self.pipes.encodeHy4SwigluClampedLinearGate(f, .{
                        .gate_in = model.dense_gate_bind,
                        .up_in = model.dense_up_bind,
                        .out = model.dense_act_bind,
                        .num_rows = m,
                        .intermediate_size = sh_inter,
                        .interleaved_gate_up = false,
                        .clamp_mode = .gate_and_up,
                        .swiglu_limit = m82c.swiglu_limit,
                    }, true);
                    try model.encodeLinearBatched(f, layer.shared_down.?, model.dense_act_bind, self.bindAt(self.hy4_shared_out_off), m, true);
                }
            }

            try flushFrame(model, frame_opt);
            const topk_ids = self.sliceU32(self.topk_ids_off, @as(usize, m) * top_k);
            const exp_in_base: metal.BufferBinding = if (is_latent) self.bindAt(self.hy4_latent_x_off) else model.x_norm_bind;
            const exp_out_base_off: u64 = if (is_latent) self.hy4_latent_exp_out_off else self.exp_out_off;
            const exp_row_bytes: u64 = @as(u64, exp_dim) * 2;

            const total_assign = @as(usize, m) * top_k;
            var unique_eids: [256]u32 = undefined;
            var unique_cnt: usize = 0;
            for (0..total_assign) |ai| {
                const eid = topk_ids[ai];
                if (eid >= layer.experts.len) return error.InvalidArgument;
                var seen = false;
                for (unique_eids[0..unique_cnt]) |u| {
                    if (u == eid) {
                        seen = true;
                        break;
                    }
                }
                if (!seen and unique_cnt < unique_eids.len) {
                    unique_eids[unique_cnt] = eid;
                    unique_cnt += 1;
                }
            }
            for (unique_eids[0..unique_cnt]) |eid| {
                const ew = layer.experts[eid];
                const resolved: ColdSlotBindings = switch (ew) {
                    .hot => |hw| .{ .w1 = hw.w1, .w_up = hw.w_up, .w2 = hw.w2 },
                    .cold => |cm| blk: {
                        try flushFrame(model, frame_opt);
                        const cc = &(self.cold_cache.?);
                        const pinned = cc.pinExpert(cm, &self.stats) catch return error.IoFailed;
                        break :blk pinned;
                    },
                };
                const f = try ensureFrame(model, frame_opt);
                for (0..m) |tok| {
                    const x_tok_bind: metal.BufferBinding = .{
                        .buffer = exp_in_base.buffer,
                        .offset = exp_in_base.offset + @as(u64, tok) * exp_row_bytes,
                    };
                    for (0..top_k) |ki| {
                        if (topk_ids[tok * top_k + ki] != eid) continue;
                        const out_slot = self.bindAt(exp_out_base_off + @as(u64, tok * top_k + ki) * exp_row_bytes);
                        if (resolved.w_up) |wu| {
                            try model.encodeLinearGemv(f, resolved.w1, x_tok_bind, self.bindAt(self.exp_gate_up_off), true);
                            try model.encodeLinearGemv(f, wu, x_tok_bind, self.bindAt(self.exp_up_off), false);
                            try self.pipes.encodeHy4SwigluClampedLinearGate(f, .{
                                .gate_in = self.bindAt(self.exp_gate_up_off),
                                .up_in = self.bindAt(self.exp_up_off),
                                .out = self.bindAt(self.exp_act_off),
                                .num_rows = 1,
                                .intermediate_size = moe_inter,
                                .interleaved_gate_up = false,
                                .clamp_mode = .gate_and_up,
                                .swiglu_limit = m82c.swiglu_limit,
                            }, true);
                        } else {
                            try model.encodeLinearGemv(f, resolved.w1, x_tok_bind, self.bindAt(self.exp_gate_up_off), true);
                            try self.pipes.encodeHy4SwigluClampedLinearGate(f, .{
                                .gate_in = self.bindAt(self.exp_gate_up_off),
                                .out = self.bindAt(self.exp_act_off),
                                .num_rows = 1,
                                .intermediate_size = moe_inter,
                                .interleaved_gate_up = true,
                                .clamp_mode = .gate_and_up,
                                .swiglu_limit = m82c.swiglu_limit,
                            }, true);
                        }
                        try model.encodeLinearGemv(f, resolved.w2, self.bindAt(self.exp_act_off), out_slot, true);
                    }
                }
            }

            const f = try ensureFrame(model, frame_opt);
            if (is_latent) {
                // Combine routed outputs in `latent_dim`, project up via `routed_expert_up_proj`,
                // then add shared expert + `combine_output_norm` via `encodeHy4CombineSandwichNorm`!
                try self.pipes.encodeHy4CombineSandwichNorm(f, .{
                    .expert_or_x = self.bindAt(self.hy4_latent_exp_out_off),
                    .topk_weights = self.bindAt(self.topk_w_off),
                    .weights_f16 = false,
                    .out = self.bindAt(self.hy4_latent_comb_off),
                    .m = m,
                    .hidden_dim = exp_dim,
                    .top_k = top_k,
                    .rms_eps = cfg.rms_norm_eps,
                }, true);
                try model.encodeLinearBatched(f, layer.latent_up_proj.?, self.bindAt(self.hy4_latent_comb_off), model.proj_out_bind, m, true);
                try self.pipes.encodeHy4CombineSandwichNorm(f, .{
                    .expert_or_x = model.proj_out_bind,
                    .shared_out = if (layer.shared_gate != null) self.bindAt(self.hy4_shared_out_off) else null,
                    .combine_norm_w = if (layer.combine_output_norm) |cn| cn.binding else null,
                    .norm_dtype = if (layer.combine_output_norm) |cn| cn.dtype else .bf16,
                    .out = model.proj_out_bind,
                    .m = m,
                    .hidden_dim = h,
                    .top_k = 0,
                    .rms_eps = cfg.rms_norm_eps,
                }, true);
            } else {
                try self.pipes.encodeHy4CombineSandwichNorm(f, .{
                    .expert_or_x = self.bindAt(self.exp_out_off),
                    .topk_weights = self.bindAt(self.topk_w_off),
                    .weights_f16 = false,
                    .shared_out = if (layer.shared_gate != null) self.bindAt(self.hy4_shared_out_off) else null,
                    .combine_norm_w = if (layer.combine_output_norm) |cn| cn.binding else null,
                    .norm_dtype = if (layer.combine_output_norm) |cn| cn.dtype else .bf16,
                    .out = model.proj_out_bind,
                    .m = m,
                    .hidden_dim = h,
                    .top_k = top_k,
                    .rms_eps = cfg.rms_norm_eps,
                }, true);
            }
        }

        // --- H. iHC Post-MLP Update ---
        if (m82c.enable_ihc) {
            try flushFrame(model, frame_opt);
            applyHy4HcPost(self.hy4_hc_streams_host, model.proj_out_host, mlp_post_weights[0 .. @as(usize, m) * hc_mult], m, hc_mult, h);
        } else {
            const f = try ensureFrame(model, frame_opt);
            try model.moe_pipes.encodeResidualAdd(f, .{
                .x = model.x_bind,
                .delta = model.proj_out_bind,
                .count = m * h,
            }, true);
        }
    }

    fn execHy4Engram(
        self: *M82Runtime,
        model: *graph.GraphModel,
        frame_opt: *?metal.Compute.Frame,
        prt: *graph.PleRuntime,
        l: usize,
        eg: Hy4EngramLayerWeights,
        token_ids: []const u32,
        ancestor_masks: []const u64,
        base_pos: u32,
        m: u32,
        hc_mult: u32,
        h: u32,
    ) graph.GraphStepError!void {
        const d = eg.engram_dim;
        const before_reads = prt.store.stats().nvme_reads;
        const dst_f16 = self.sliceF16(self.hy4_engram_in_f16_off, @as(usize, m) * d);
        for (0..m) |tok| {
            // Build causal token window ending at `tok`.
            var win: [ple_mod.ENGRAM_MAX_ORDER]u32 = @splat(0);
            var win_len: usize = 0;
            if (base_pos > 0 and model.token_history.len > 0) {
                const start = if (base_pos >= 2) base_pos - 2 else 0;
                for (start..base_pos) |p| {
                    if (win_len < win.len - 1) {
                        win[win_len] = model.token_history[p];
                        win_len += 1;
                    }
                }
            }
            var mask = ancestor_masks[tok];
            while (mask != 0) {
                const j: usize = @intCast(@ctz(mask));
                mask &= (mask - 1);
                if (j < token_ids.len and win_len < win.len) {
                    win[win_len] = token_ids[j];
                    win_len += 1;
                }
            }
            if (win_len == 0) {
                win[0] = token_ids[tok];
                win_len = 1;
            }
            const row_idx = ple_mod.hashPleNgram(win[0..win_len], @intCast(@min(win_len, 2)), prt.store.layerTotalRows(@intCast(l)) catch 1);
            const row_arr = [_]u64{row_idx};
            const f32_out = prt.f32_tmp[0..d];
            @memset(f32_out, 0);
            prt.store.lookupForLoopStep(@intCast(l), &row_arr, 1.0, 0, &prt.scratch, f32_out) catch return error.IoFailed;
            f32SliceToF16(f32_out, dst_f16[tok * d ..][0..d]);
        }
        const after_reads = prt.store.stats().nvme_reads;
        self.stats.hy4_engram_r0_reads += (after_reads -| before_reads);

        // Project looked-up Engram rows through wired `key_proj` and `value_proj`.
        {
            const f = try ensureFrame(model, frame_opt);
            try model.encodeLinearBatched(f, eg.key_proj, self.bindAt(self.hy4_engram_in_f16_off), self.bindAt(self.hy4_engram_k_off), m, true);
            try model.encodeLinearBatched(f, eg.value_proj, self.bindAt(self.hy4_engram_in_f16_off), self.bindAt(self.hy4_engram_v_off), m, false);
        }
        try flushFrame(model, frame_opt);

        const k_h = self.sliceF16(self.hy4_engram_k_off, @as(usize, m) * h);
        const v_h = self.sliceF16(self.hy4_engram_v_off, @as(usize, m) * h);
        const inv_sqrt_h: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(h)));
        for (0..m) |tok| {
            // Compute scalar context gate between stream 0 and projected key.
            var dot_qk: f32 = 0.0;
            for (0..h) |c| {
                const qv: f32 = @floatCast(self.hy4_hc_streams_host[tok * hc_mult * h + c]);
                const kv: f32 = @floatCast(k_h[tok * h + c]);
                const pg: f32 = if (eg.pre_gamma) |g| g[c] else 1.0;
                dot_qk += (qv * pg) * kv;
            }
            const gate: f32 = 1.0 / (1.0 + @exp(-std.math.clamp(dot_qk * inv_sqrt_h, -15.0, 15.0)));
            for (0..hc_mult) |s| {
                const sc: f32 = if (eg.hc_scales) |hs| (if (s < hs.len) hs[s] else 1.0) else 1.0;
                const fac = gate * sc;
                for (0..h) |c| {
                    const idx = (tok * hc_mult + s) * h + c;
                    const cur: f32 = @floatCast(self.hy4_hc_streams_host[idx]);
                    const vv: f32 = @floatCast(v_h[tok * h + c]);
                    self.hy4_hc_streams_host[idx] = @floatCast(cur + fac * vv);
                }
            }
        }
    }

    // ------------------------------------------------------------------------
    // Tree Commit (`commitTreeKv`) — Bit-Exact State Advancement for DFlare/DFly2
    // ------------------------------------------------------------------------

    fn stageAncestorRowsF16(
        dst_cache: []f16,
        src_tree: []const f16,
        base_pos: u32,
        max_seq_len: u32,
        row_elems: u32,
        anc_mask: u64,
        m: u32,
    ) u32 {
        var depth: u32 = 0;
        const re: usize = row_elems;
        for (0..m) |anc| {
            if (((anc_mask >> @intCast(anc)) & 1) == 0) continue;
            const pos = base_pos + depth;
            if (pos < max_seq_len) {
                @memcpy(dst_cache[@as(usize, pos) * re ..][0..re], src_tree[anc * re ..][0..re]);
            }
            depth += 1;
        }
        return depth;
    }

    fn swapSelectedRow(idx_pool: *ink_gpu.IndexPoolBuffer, a: usize, b: usize, cand_k: usize) void {
        const ra = idx_pool.selected_tokens[a * cand_k ..][0..cand_k];
        const rb = idx_pool.selected_tokens[b * cand_k ..][0..cand_k];
        for (0..cand_k) |i| {
            const tmp = ra[i];
            ra[i] = rb[i];
            rb[i] = tmp;
        }
        const tc = idx_pool.selected_counts[a];
        idx_pool.selected_counts[a] = idx_pool.selected_counts[b];
        idx_pool.selected_counts[b] = tc;
    }

    pub fn commitTreeKv(
        self: *M82Runtime,
        model: *graph.GraphModel,
        base_pos: u32,
        accepted_nodes: []const u16,
    ) graph.GraphStepError!void {
        if (accepted_nodes.len == 0) return;
        const k_len: u32 = @intCast(accepted_nodes.len);
        const cfg = &model.config;

        if (self.family.isInkling2()) {
            const h = cfg.hidden_size;
            const kv_dim = cfg.kvDim();
            var frame = try model.compute.beginFrame(model.timeout_ms);
            errdefer frame.abort();
            var had_dispatch = false;

            for (self.ink_steps) |*st| {
                const slot0 = try st.kv_ring.slotForPos(base_pos);
                const end_slot = try st.kv_ring.slotForPos(base_pos + k_len - 1);
                const wraps_tile = (end_slot.tile_idx != slot0.tile_idx);
                try model.hybrid_pipes.encodeQsaKvTreeCommit(&frame, .{
                    .k_tree = self.bindAt(st.tree_k_off),
                    .v_tree = self.bindAt(st.tree_v_off),
                    .accepted_nodes = .{ .inline_u16 = accepted_nodes },
                    .accepted_count = k_len,
                    .k_tile0 = st.kv_ring.k_tiles[slot0.tile_idx],
                    .v_tile0 = st.kv_ring.v_tiles[slot0.tile_idx],
                    .k_tile1 = if (wraps_tile) st.kv_ring.k_tiles[end_slot.tile_idx] else null,
                    .v_tile1 = if (wraps_tile) st.kv_ring.v_tiles[end_slot.tile_idx] else null,
                    .num_kv_heads = cfg.num_key_value_heads,
                    .head_dim = cfg.head_dim,
                    .base_pos = base_pos,
                    .chunk_tokens = st.kv_ring.chunk_tokens,
                    .ring_capacity = st.kv_ring.ring_capacity,
                }, had_dispatch);
                had_dispatch = true;
            }
            const sig = try frame.commit();
            try model.compute.wait(sig, model.timeout_ms);

            // Commit `SconvRingArena` states bit-exact along `accepted_nodes`!
            const raw = self.arena.bytes();
            if (self.sconv_h_ring) |*sring| {
                for (self.ink_steps, 0..) |*st, step_idx| {
                    const phys_l = self.physicalLayerForStep(@intCast(step_idx), cfg);
                    const lw = &self.ink_layers[phys_l];
                    if (lw.attn_sconv_w != null) {
                        commitSconvRingAccepted(raw, sring, st.sconv_attn_slot, st.tree_attn_sconv_in_off, self.sconv_pos, accepted_nodes, h);
                    }
                    if (lw.mlp_sconv_w != null) {
                        commitSconvRingAccepted(raw, sring, st.sconv_mlp_slot, st.tree_mlp_sconv_in_off, self.sconv_pos, accepted_nodes, h);
                    }
                }
            }
            if (self.sconv_kv_ring) |*kvring| {
                for (self.ink_steps, 0..) |*st, step_idx| {
                    const phys_l = self.physicalLayerForStep(@intCast(step_idx), cfg);
                    const lw = &self.ink_layers[phys_l];
                    if (lw.k_sconv_w != null or lw.k_sconv_dyn_proj != null) {
                        commitSconvRingAccepted(raw, kvring, st.sconv_k_slot, st.tree_k_sconv_in_off, self.sconv_pos, accepted_nodes, kv_dim);
                    }
                    if (lw.v_sconv_w != null or lw.v_sconv_dyn_proj != null) {
                        commitSconvRingAccepted(raw, kvring, st.sconv_v_slot, st.tree_v_sconv_in_off, self.sconv_pos, accepted_nodes, kv_dim);
                    }
                }
            }
            self.sconv_pos = base_pos + k_len;

            // Record all accepted tree nodes into full DFlare context history before compacting.
            const h_us: usize = h;
            if (self.hasDFlareCtxHistory()) {
                for (accepted_nodes, 0..) |nid, idx| {
                    self.recordDFlareCtxTapRow(base_pos + @as(u32, @intCast(idx)), nid, h_us);
                }
                self.dflare_ctx_hist_len = base_pos + @as(u32, @intCast(accepted_nodes.len));
            }

            // Compact `last_hidden_f32` and point GPU + CPU `dflare_taps` at the last accepted node!
            const last_node: usize = accepted_nodes[accepted_nodes.len - 1];
            if (last_node != 0 and (last_node + 1) * h_us <= self.last_hidden_f32.len) {
                @memcpy(self.last_hidden_f32[0..h_us], self.last_hidden_f32[last_node * h_us ..][0..h_us]);
            }
            self.syncDFlareActiveRow(last_node, h_us);
        } else if (self.family.isHy4()) {
            const kv_rank = cfg.kv_lora_rank;
            const qk_rope = cfg.qk_rope_head_dim;
            const idx_d = self.hy4_index_head_dim;
            for (self.hy4_layers) |*layer| {
                const src_lat = self.sliceF16(layer.tree_kv_latent_off, @as(usize, self.max_tree_nodes) * kv_rank);
                const dst_lat = self.sliceF16(layer.kv_latent_off, @as(usize, self.max_seq_len) * kv_rank);
                for (accepted_nodes, 0..) |nid, idx| {
                    const pos = base_pos + @as(u32, @intCast(idx));
                    @memcpy(dst_lat[@as(usize, pos) * kv_rank ..][0..kv_rank], src_lat[@as(usize, nid) * kv_rank ..][0..kv_rank]);
                }
                if (qk_rope > 0) {
                    const src_r = self.sliceF16(layer.tree_kv_rope_off, @as(usize, self.max_tree_nodes) * qk_rope);
                    const dst_r = self.sliceF16(layer.kv_rope_off, @as(usize, self.max_seq_len) * qk_rope);
                    for (accepted_nodes, 0..) |nid, idx| {
                        const pos = base_pos + @as(u32, @intCast(idx));
                        @memcpy(dst_r[@as(usize, pos) * qk_rope ..][0..qk_rope], src_r[@as(usize, nid) * qk_rope ..][0..qk_rope]);
                    }
                }
                // Compact `k_fine` cache when tree nodes were non-contiguous.
                if (layer.indexer_mode == .full and layer.indexer != null and !layer.is_swa) {
                    const kf = self.sliceF16(layer.k_fine_cache_off, @as(usize, self.max_seq_len) * idx_d);
                    for (accepted_nodes, 0..) |nid, idx| {
                        if (nid != idx) {
                            const src_pos = base_pos + @as(u32, nid);
                            const dst_pos = base_pos + @as(u32, @intCast(idx));
                            if (src_pos < self.max_seq_len and dst_pos < self.max_seq_len) {
                                @memcpy(kf[@as(usize, dst_pos) * idx_d ..][0..idx_d], kf[@as(usize, src_pos) * idx_d ..][0..idx_d]);
                            }
                        }
                    }
                }
            }
            const last_node: usize = accepted_nodes[accepted_nodes.len - 1];
            const h_us: usize = cfg.hidden_size;
            if (last_node != 0 and (last_node + 1) * h_us <= self.last_hidden_f32.len) {
                @memcpy(self.last_hidden_f32[0..h_us], self.last_hidden_f32[last_node * h_us ..][0..h_us]);
            }
            self.refreshDFly2TargetKvViews(cfg, base_pos + k_len);
        }
    }

    // ------------------------------------------------------------------------
    // Logits Finalization & Helpers
    // ------------------------------------------------------------------------

    fn finalizeLogits(
        self: *M82Runtime,
        model: *graph.GraphModel,
        frame_opt: *?metal.Compute.Frame,
        m: u32,
        compute_logits: bool,
        logit_softcap: f32,
        unpadded_vocab_size: u32,
        already_normed: bool,
    ) graph.GraphStepError!qwen3.VerifyTreeResult {
        const cfg = &model.config;
        const h = cfg.hidden_size;
        const v = cfg.vocab_size;
        if (!compute_logits) {
            try flushFrame(model, frame_opt);
            @memset(model.out_token_ids_host[0..m], 0);
            return .{
                .num_nodes = m,
                .argmax_tokens = model.out_token_ids_host[0..m],
                .logits_f16 = model.logits_host[0 .. @as(usize, m) * v],
                .total_unique_experts = 0,
                .avg_unique_experts_per_layer = 0.0,
                .actual_bytes_moved = 0,
            };
        }

        const eff_vocab: u32 = if (unpadded_vocab_size > 0 and unpadded_vocab_size < v) unpadded_vocab_size else v;

        {
            const f = try ensureFrame(model, frame_opt);
            if (!already_normed) {
                try model.attn_pipes.encodeRmsNorm(f, .{
                    .x = model.x_bind,
                    .w = model.final_norm.binding,
                    .y = model.x_norm_bind,
                    .num_rows = m,
                    .hidden_dim = h,
                    .eps = cfg.rms_norm_eps,
                    .w_dtype = model.final_norm.dtype,
                }, true);
            }
            try model.encodeLinearBatched(f, model.lm_head, model.x_norm_bind, model.logits_bind, m, true);
            // GPU logit softcap + padded-vocab `-inf` mask + per-row greedy argmax inside the
            // same `MTL4CommandBuffer` (`0` scalar CPU `tanh` loops over `m * vocab_size`!).
            try self.pipes.encodeInkling2LogitFinalize(f, .{
                .logits = model.logits_bind,
                .out_tokens = model.out_token_id_bind,
                .m = m,
                .vocab_size = v,
                .eff_vocab = eff_vocab,
                .logit_softcap = logit_softcap,
            }, true);
        }
        try flushFrame(model, frame_opt);

        model.last_logits_row = if (m > 0) (m - 1) else 0;
        return .{
            .num_nodes = m,
            .argmax_tokens = model.out_token_ids_host[0..m],
            .logits_f16 = model.logits_host[0 .. @as(usize, m) * v],
            .total_unique_experts = 0,
            .avg_unique_experts_per_layer = 0.0,
            .actual_bytes_moved = 0,
        };
    }

    fn refreshDFly2TargetKvViews(self: *M82Runtime, cfg: *const graph.GraphConfig, kv_len: u32) void {
        const m82c = &cfg.m82_cfg;
        if (m82c.dfly_target_kv_count == 0 or self.dfly_kv_k_buf.len == 0 or self.dfly_kv_dim == 0) return;
        const max_len = @min(kv_len, self.max_seq_len);
        const cap_per_layer = self.dfly_kv_k_buf.len / m82c.dfly_target_kv_count;
        const kv_dim = self.dfly_kv_dim;
        const max_toks = @min(@as(usize, max_len), cap_per_layer / kv_dim);
        const kv_rank = cfg.kv_lora_rank;
        const copy_d = @min(kv_dim, kv_rank);

        var cnt: usize = 0;
        for (0..m82c.dfly_target_kv_count) |vi| {
            const l_idx = m82c.dfly_target_kv_layers[vi];
            if (l_idx >= self.hy4_layers.len) continue;
            const layer = &self.hy4_layers[l_idx];
            const src_lat = self.sliceF16(layer.kv_latent_off, @as(usize, self.max_seq_len) * kv_rank);
            const k_dst = self.dfly_kv_k_buf[vi * cap_per_layer ..][0 .. max_toks * kv_dim];
            const v_dst = self.dfly_kv_v_buf[vi * cap_per_layer ..][0 .. max_toks * kv_dim];
            for (0..max_toks) |t| {
                const s_row = src_lat[t * kv_rank ..];
                const kd = k_dst[t * kv_dim ..][0..kv_dim];
                const vd = v_dst[t * kv_dim ..][0..kv_dim];
                for (0..copy_d) |c| {
                    const fv: f32 = @floatCast(s_row[c]);
                    kd[c] = fv;
                    vd[c] = fv;
                }
                if (copy_d < kv_dim) {
                    @memset(kd[copy_d..kv_dim], 0);
                    @memset(vd[copy_d..kv_dim], 0);
                }
            }
            self.dfly_kv_views[cnt] = .{
                .k = k_dst,
                .v = v_dst,
                .kv_len = max_toks,
            };
            cnt += 1;
        }
        self.dfly_kv_view_count = cnt;
    }

    fn physicalLayerForStep(self: *const M82Runtime, step_idx: u32, cfg: *const graph.GraphConfig) usize {
        _ = self;
        const ls = cfg.axes.loop_schedule;
        if (!ls.isRecurrent()) return @min(step_idx, cfg.num_hidden_layers - 1);
        const cs = ls.start_layer;
        const ce = if (ls.end_layer > cs) ls.end_layer else cfg.num_hidden_layers;
        const core_len = ce - cs;
        if (step_idx < cs) return step_idx;
        const after_pre = step_idx - cs;
        const total_core = core_len * ls.max_loops;
        if (after_pre < total_core) return cs + (after_pre % core_len);
        return @min(ce + (after_pre - total_core), cfg.num_hidden_layers - 1);
    }

    inline fn bindAt(self: *const M82Runtime, offset: u64) metal.BufferBinding {
        return .{ .buffer = self.buf, .offset = offset };
    }

    inline fn sliceF16(self: *const M82Runtime, offset: u64, count: usize) []f16 {
        const raw = self.arena.bytes()[@intCast(offset)..][0 .. count * 2];
        return @alignCast(std.mem.bytesAsSlice(f16, raw));
    }

    inline fn sliceF32(self: *const M82Runtime, offset: u64, count: usize) []f32 {
        const raw = self.arena.bytes()[@intCast(offset)..][0 .. count * 4];
        return @alignCast(std.mem.bytesAsSlice(f32, raw));
    }

    inline fn sliceU32(self: *const M82Runtime, offset: u64, count: usize) []u32 {
        const raw = self.arena.bytes()[@intCast(offset)..][0 .. count * 4];
        return @alignCast(std.mem.bytesAsSlice(u32, raw));
    }
};

// ============================================================================
// Private Loaders (`Inkling 2` & `Hy4`)
// ============================================================================

fn inferHy4IndexDims(cfg: *const graph.GraphConfig, source: *const graph.GraphModel.AmfSource) struct { idx_h: u32, idx_d: u32 } {
    var idx_h: u32 = @max(cfg.m82_cfg.index_n_heads, @as(u32, 1));
    var idx_d: u32 = @max(cfg.m82_cfg.index_head_dim, @as(u32, 16));
    var nb_k: [96]u8 = undefined;
    var nb_q: [96]u8 = undefined;
    for (0..cfg.num_hidden_layers) |l| {
        const cidx_k = std.fmt.bufPrint(&nb_k, "layers.{d}.self_attn.cidx.tk_proj", .{l}) catch unreachable;
        const cidx_q = std.fmt.bufPrint(&nb_q, "layers.{d}.self_attn.cidx.query_projection", .{l}) catch unreachable;
        if (findLinearEntry(source, cidx_k)) |ke| {
            if (ke.shape[0] > 0) idx_d = ke.shape[0];
            if (findLinearEntry(source, cidx_q)) |qe| {
                if (idx_d > 0 and qe.shape[0] >= idx_d) idx_h = qe.shape[0] / idx_d;
            }
            return .{ .idx_h = idx_h, .idx_d = idx_d };
        }
        const std_k = std.fmt.bufPrint(&nb_k, "layers.{d}.self_attn.indexer.wk", .{l}) catch unreachable;
        const std_q = std.fmt.bufPrint(&nb_q, "layers.{d}.self_attn.indexer.wq_b", .{l}) catch unreachable;
        if (findLinearEntry(source, std_k)) |ke| {
            if (ke.shape[0] > 0) idx_d = ke.shape[0];
            if (findLinearEntry(source, std_q)) |qe| {
                if (idx_d > 0 and qe.shape[0] >= idx_d) idx_h = qe.shape[0] / idx_d;
            }
            return .{ .idx_h = idx_h, .idx_d = idx_d };
        }
    }
    return .{ .idx_h = idx_h, .idx_d = idx_d };
}

fn computeArenaBytes(
    cfg: *const graph.GraphConfig,
    ctx: M82LoadContext,
    eff_steps: u32,
) !usize {
    const m82c = &cfg.m82_cfg;
    const h: usize = cfg.hidden_size;
    const max_m: usize = ctx.max_tree_nodes;
    const num_layers: usize = cfg.num_hidden_layers;
    const max_seq: usize = ctx.max_seq_len;
    const q_dim: usize = cfg.qDim();
    const kv_dim: usize = cfg.kvDim();
    const inter: usize = @max(@max(cfg.intermediate_size, cfg.moe_intermediate_size), @as(u32, 64));
    const num_exp: usize = @max(cfg.num_experts, @as(u32, 1));
    const top_k: usize = @max(cfg.num_experts_per_tok, @as(u32, 1));

    var bytes: usize = 64 * 1024;
    // Shared scratch
    bytes += 4 * amf.align64(max_m * h * 2);
    bytes += amf.align64(max_m * num_exp * 2);
    bytes += 2 * amf.align64(max_m * top_k * 4);
    bytes += amf.align64(max_m * 2 * inter * 2);
    bytes += 2 * amf.align64(max_m * inter * 2);
    bytes += amf.align64(max_m * top_k * h * 2);
    bytes += 2 * amf.align64(max_m * h * 2); // exp_gathered_x + exp_down_tmp
    bytes += 2 * amf.align64(max_m * top_k * 4); // exp_tok_idx + exp_dst_slot
    if (ctx.enable_router_trace) {
        bytes += amf.align64(@as(usize, eff_steps) * top_k * @sizeOf(u32));
    }

    if (cfg.m82_family.isInkling2()) {
        bytes += 2 * amf.align64(max_m * h * 2); // ink_prelude_cur + ink_inp_sa
        bytes += 3 * amf.align64(max_m * h * 2); // x_ctrl_{q,k,v}
        bytes += amf.align64(max_m * cfg.num_attention_heads * m82c.d_rel * 2);
        bytes += amf.align64(max_m * cfg.num_attention_heads * 6);
        bytes += amf.align64(max_m * q_dim * 2);
        const dyn_proj_max: usize = @max(@as(usize, cfg.num_key_value_heads) * 4, @as(usize, 4));
        bytes += amf.align64(max_m * kv_dim * 4 * 2) + amf.align64(max_m * dyn_proj_max * 2);
        // Sconv rings (`H` and `kv_dim`)
        bytes += try ink_gpu.SconvRingArena.requiredBytes(.{ .num_layers = eff_steps * 2, .hidden_dim = cfg.hidden_size });
        bytes += try ink_gpu.SconvRingArena.requiredBytes(.{ .num_layers = eff_steps * 2, .hidden_dim = @max(cfg.kvDim(), @as(u32, 4)) });
        // AttnRes ring
        const max_blks: u32 = @min(@as(u32, ink_gpu.MAX_ATTN_RES_BLOCKS), (eff_steps / @max(@as(u32, 1), m82c.n_layers_per_attn_res_block)) + 3);
        bytes += try ink_gpu.AttnResRingArena.requiredBytes(.{
            .max_blocks = max_blks,
            .max_tokens = ctx.max_tree_nodes,
            .hidden_dim = cfg.hidden_size,
        });
        // GPU cross-pass message bank / attended messages / controller z scratch + cross-pass queries
        if (cfg.m82_family == .inkling2_loop) {
            bytes += amf.align64(@as(usize, max_blks) * max_m * h * 4);
            bytes += amf.align64(4 * max_m * h * 4);
            bytes += amf.align64(max_m * @max(@as(usize, m82c.loop_gate_rank), 64) * 4);
            if (ctx.source.find("middle_cycle_cross_pass_queries")) |cpq_e| {
                bytes += amf.align64(@as(usize, entryElemCount(cpq_e)) * 4);
            }
            if (ctx.source.find("middle_cycle_cross_pass_kv_queries")) |cpkv_e| {
                bytes += amf.align64(@as(usize, entryElemCount(cpkv_e)) * 4);
            }
            if (!ctx.use_sme2_controllers) {
                for (ctx.source.entries()) |*e| {
                    const nm = e.nameSlice();
                    if (std.mem.indexOf(u8, nm, ".controllers.") != null and
                        (std.mem.endsWith(u8, nm, ".down_weight") or
                            std.mem.endsWith(u8, nm, ".forget_weight") or
                            std.mem.endsWith(u8, nm, ".gain_weight")))
                    {
                        const layout = e.tensorLayout() catch .row_major;
                        if (layout == .sme2_panel64) {
                            bytes += amf.align64(@as(usize, entryElemCount(e)) * 2) + 64;
                        }
                    }
                }
            }
        }
        // `G29` indirect MoE workspace + per-layer expert offset/scheme tables
        bytes += (try moe.MoeWorkspace.requiredBytes(.{
            .max_tokens = ctx.max_tree_nodes,
            .num_experts = cfg.num_experts,
            .top_k = cfg.num_experts_per_tok,
            .hidden_size = cfg.hidden_size,
            .intermediate_size = cfg.intermediate_size,
            .shared_intermediate_size = 0,
        })) + 4096;
        bytes += num_layers * (amf.align64(num_exp * @sizeOf(moe.ExpertWeights)) + amf.align64(num_exp) + 128);
        // GPU DFlare taps (`[num_taps, max_m, H]` f16) and committed context ring (`[hist_cap, num_taps, H]` f16)
        if (m82c.dflare_target_layer_count > 0) {
            bytes += amf.align64(@as(usize, m82c.dflare_target_layer_count) * max_m * h * 2);
            if (max_m > 1) {
                const hist_cap: usize = @min(max_seq, if (h >= 1024) @as(usize, 1024) else @as(usize, 128));
                if (hist_cap > 0) {
                    bytes += amf.align64(hist_cap * @as(usize, m82c.dflare_target_layer_count) * h * 2);
                }
            }
        }
        // Per-layer converted small weights (`sconv`, `rel_emb`, multi-pass `attn_res`, `bias`, `l2_scale_vec`)
        const per_l_small = 2 * amf.align64(h * 4 * 4) + 2 * amf.align64(kv_dim * 4 * 4) +
            amf.align64(@as(usize, m82c.rel_extent) * m82c.d_rel * 2) +
            amf.align64(q_dim * 4) + amf.align64(@as(usize, cfg.num_attention_heads) * m82c.d_rel * 4) +
            6 * amf.align64(h * 2) + amf.align64(num_exp * 4) + 4096;
        bytes += num_layers * per_l_small + amf.align64(h * 2);
        // Per-step KV tile tables + tree scratch
        const max_tiles: usize = @max((max_seq + 511) / 512, (m82c.sliding_window_size + 127) / 128) + 2;
        const per_step = amf.align64(max_tiles * @sizeOf(attention.KvTilePair)) +
            2 * amf.align64(max_m * kv_dim * 2) +
            2 * amf.align64(max_m * h * 2) +
            2 * amf.align64(max_m * kv_dim * 2) + 512;
        bytes += @as(usize, eff_steps) * per_step;
    } else if (cfg.m82_family.isHy4()) {
        const hc_mult: usize = m82c.hc_mult;
        const n_q: usize = cfg.num_attention_heads;
        const q_rank: usize = @max(cfg.q_lora_rank, @as(u32, 1));
        const kv_rank: usize = @max(cfg.kv_lora_rank, @as(u32, 1));
        const qk_nope: usize = @max(cfg.qk_nope_head_dim, @as(u32, 1));
        const qk_rope: usize = cfg.qk_rope_head_dim;
        const v_dim: usize = @max(cfg.v_head_dim, @as(u32, 1));
        const idx_dims = inferHy4IndexDims(cfg, ctx.source);
        const idx_d: usize = idx_dims.idx_d;
        const idx_h: usize = idx_dims.idx_h;

        bytes += amf.align64(max_m * hc_mult * h * 2);
        bytes += amf.align64(max_m * q_rank * 2);
        bytes += amf.align64(max_m * n_q * (qk_nope + qk_rope) * 2);
        bytes += amf.align64(max_m * n_q * qk_nope * 2);
        bytes += amf.align64(max_m * n_q * @max(qk_rope, 1) * 2);
        bytes += amf.align64(max_m * n_q * kv_rank * 2) * 2;
        bytes += amf.align64(max_m * (kv_rank + qk_rope) * 2);
        bytes += amf.align64(max_m * kv_rank * 2);
        bytes += amf.align64(max_m * @max(qk_rope, 1) * 2);
        bytes += amf.align64(max_m * n_q * v_dim * 2) * 2;
        bytes += amf.align64(max_m * idx_h * idx_d * 2);
        bytes += amf.align64(max_m * idx_d * 2);
        bytes += amf.align64(max_m * idx_h * 2);
        // LatentMoE + shared + Engram scratch
        bytes += 4 * amf.align64(max_m * h * 2);
        bytes += amf.align64(max_m * top_k * h * 2);
        bytes += amf.align64(max_m * @max(h, @as(usize, 256)) * 2);

        // IndexPoolBuffer
        const cand_k = @min(m82c.candidate_topk, @max(ctx.max_seq_len, @as(u32, 16)));
        const grp_k = if (m82c.use_compressed_indexer)
            @min(m82c.group_topk, @max((ctx.max_seq_len + m82c.compress_ratio - 1) / m82c.compress_ratio, @as(u32, 1)))
        else
            0;
        bytes += try ink_gpu.IndexPoolBuffer.requiredBytes(.{
            .max_kv_len = ctx.max_seq_len,
            .max_nodes = ctx.max_tree_nodes,
            .idx_dim = idx_dims.idx_d,
            .compress_ratio = if (m82c.use_compressed_indexer) m82c.compress_ratio else 1,
            .group_topk = grp_k,
            .candidate_topk = cand_k,
        });

        // Per-layer MLA KV caches + W_uk_t/W_uv + k_fine cache + sink + bias
        for (0..num_layers) |l| {
            bytes += amf.align64(max_seq * kv_rank * 2);
            bytes += amf.align64(max_seq * @max(qk_rope, 1) * 2);
            bytes += amf.align64(max_m * kv_rank * 2);
            bytes += amf.align64(max_m * @max(qk_rope, 1) * 2);
            bytes += amf.align64(n_q * kv_rank * qk_nope * 2);
            bytes += amf.align64(n_q * v_dim * kv_rank * 2);
            bytes += amf.align64(n_q * 4) + amf.align64(num_exp * 4) + 512;
            const is_full_sparse = (l < m82c.indexer_modes.len and m82c.indexer_modes[l] == .full) and
                !(l < m82c.is_swa_layer.len and m82c.is_swa_layer[l]);
            if (is_full_sparse) {
                bytes += amf.align64(max_seq * idx_d * 2);
            }
        }
    }

    // Add 128 KiB alignment slack and round up to 16 KiB page boundary.
    return page.alignUp(bytes + 128 * 1024);
}

fn loadInkling2(
    allocator: std.mem.Allocator,
    rt: *M82Runtime,
    ctx: M82LoadContext,
    f32_pool: *std.ArrayListUnmanaged([]f32),
    q_dim: u32,
    kv_dim: u32,
    eff_steps: u32,
) !void {
    const cfg = ctx.config;
    const m82c = &cfg.m82_cfg;
    const h = cfg.hidden_size;
    const max_m = ctx.max_tree_nodes;
    const num_layers = cfg.num_hidden_layers;
    const num_experts = cfg.num_experts;

    rt.ink_prelude_cur_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * h * 2);
    rt.ink_inp_sa_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * h * 2);
    rt.x_ctrl_q_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * h * 2);
    rt.x_ctrl_k_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * h * 2);
    rt.x_ctrl_v_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * h * 2);
    rt.q_rel_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * cfg.num_attention_heads * m82c.d_rel * 2);
    rt.sink_f16_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * cfg.num_attention_heads * 2);
    rt.sink_f32_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * cfg.num_attention_heads * 4);
    rt.gate_attn_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * q_dim * 2);
    rt.dyn_sconv_w_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * kv_dim * 4 * 2);
    const dyn_proj_max: usize = @max(@as(usize, cfg.num_key_value_heads) * 4, @as(usize, 4));
    rt.dyn_sconv_proj_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * dyn_proj_max * 2);

    rt.ink_moe_ws = try moe.MoeWorkspace.init(&rt.arena, .{
        .max_tokens = max_m,
        .num_experts = num_experts,
        .top_k = cfg.num_experts_per_tok,
        .hidden_size = h,
        .intermediate_size = cfg.intermediate_size,
        .shared_intermediate_size = 0,
    });

    rt.sconv_h_ring = try ink_gpu.SconvRingArena.init(&rt.arena, .{ .num_layers = eff_steps * 2, .hidden_dim = h });
    @memset(rt.sconv_h_ring.?.bytes(), 0);
    rt.sconv_kv_ring = try ink_gpu.SconvRingArena.init(&rt.arena, .{ .num_layers = eff_steps * 2, .hidden_dim = @max(kv_dim, @as(u32, 4)) });
    @memset(rt.sconv_kv_ring.?.bytes(), 0);
    const max_blks: u32 = @min(@as(u32, ink_gpu.MAX_ATTN_RES_BLOCKS), (eff_steps / @max(@as(u32, 1), m82c.n_layers_per_attn_res_block)) + 3);
    rt.attn_res_ring = try ink_gpu.AttnResRingArena.init(&rt.arena, .{
        .max_blocks = max_blks,
        .max_tokens = max_m,
        .hidden_dim = h,
    });
    @memset(rt.attn_res_ring.?.bytes(), 0);

    // Optional top-level `embed_norm.weight`, `attn_res.output_query`, `middle_cycle_cross_pass_*`.
    if (ctx.source.find("embed_norm.weight") != null) {
        rt.ink_embed_norm = try bindNormCtx(ctx, "embed_norm.weight", h);
        rt.ink_embed_norm_f32 = try loadTensorF32Slice(allocator, ctx, f32_pool, "embed_norm.weight", h);
    }
    if (ctx.source.find("attn_res.output_query") != null) {
        rt.ink_attn_res_out_q = try loadConvertedF16Binding(ctx, &rt.arena, rt.buf, "attn_res.output_query", h);
    }
    if (ctx.source.find("middle_cycle_cross_pass_queries")) |cpq_e| {
        const elems = entryElemCount(cpq_e);
        rt.ink_cross_pass_q_f32 = try loadTensorF32Slice(allocator, ctx, f32_pool, "middle_cycle_cross_pass_queries", elems);
        rt.ink_cross_pass_q_gpu = try loadConvertedF32Binding(ctx, &rt.arena, rt.buf, "middle_cycle_cross_pass_queries", elems);
        rt.ink_cross_pass_q_rows = @max(@as(u32, 1), elems / h);
    }
    if (ctx.source.find("middle_cycle_cross_pass_kv_queries")) |cpkv_e| {
        const elems = entryElemCount(cpkv_e);
        rt.ink_cross_pass_kv_q_f32 = try loadTensorF32Slice(allocator, ctx, f32_pool, "middle_cycle_cross_pass_kv_queries", elems);
        rt.ink_cross_pass_kv_q_gpu = try loadConvertedF32Binding(ctx, &rt.arena, rt.buf, "middle_cycle_cross_pass_kv_queries", elems);
        rt.ink_cross_pass_kv_q_rows = @max(@as(u32, 1), elems / h);
    }

    // Preallocate GPU cross-pass scratch (default zero-flush `G29`) or CPU SME2 scratch (opt-in `--sme2-controllers`).
    if (cfg.m82_family == .inkling2_loop) {
        rt.ink_msg_bank_gpu_off = try allocArenaOff(&rt.arena, @as(usize, max_blks) * max_m * h * 4);
        rt.ink_msg_buf_gpu_off = try allocArenaOff(&rt.arena, 4 * @as(usize, max_m) * h * 4);
        rt.ink_ctrl_z_gpu_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * @max(@as(usize, m82c.loop_gate_rank), 64) * 4);
        if (rt.use_sme2_controllers) {
            rt.sme_dispatcher = try sme.SmeDispatcher.init(allocator, .{});
            const max_kv_h = @max(h, kv_dim);
            rt.ctrl_x_f32 = try allocator.alloc(f32, @as(usize, max_m) * h * 2);
            rt.ctrl_val_f32 = try allocator.alloc(f32, @as(usize, max_m) * max_kv_h);
            rt.ctrl_out_f32 = try allocator.alloc(f32, @as(usize, max_m) * max_kv_h);
            rt.ctrl_state_attn = try allocator.alloc(f32, @as(usize, max_m) * h);
            rt.ctrl_state_k = try allocator.alloc(f32, @as(usize, max_m) * max_kv_h);
            rt.ctrl_state_v = try allocator.alloc(f32, @as(usize, max_m) * max_kv_h);
            rt.ctrl_state_mlp = try allocator.alloc(f32, @as(usize, max_m) * h);
            rt.ink_msg_bank_f32 = try allocator.alloc(f32, @as(usize, ink_gpu.MAX_ATTN_RES_BLOCKS) * max_m * h);
            rt.ink_msg_buf_f32 = try allocator.alloc(f32, 4 * @as(usize, max_m) * h);
            @memset(rt.ctrl_state_attn, 0);
            @memset(rt.ctrl_state_k, 0);
            @memset(rt.ctrl_state_v, 0);
            @memset(rt.ctrl_state_mlp, 0);
            @memset(rt.ink_msg_bank_f32, 0);
            @memset(rt.ink_msg_buf_f32, 0);
        }
    }

    // Preallocate `DFlare` GPU target taps (`f16`) + zero-copy GPU committed context tap ring (`1024`-tok SWA ring) + CPU mirror.
    if (m82c.dflare_target_layer_count > 0) {
        const ntaps: usize = m82c.dflare_target_layer_count;
        rt.dflare_taps_gpu_off = try allocArenaOff(&rt.arena, ntaps * max_m * h * 2);
        rt.dflare_tap_storage = try allocator.alloc(f32, ntaps * max_m * h);
        @memset(rt.dflare_tap_storage, 0);
        if (max_m > 1) {
            const hist_cap: u32 = @min(ctx.max_seq_len, if (h >= 1024) @as(u32, 1024) else @as(u32, 128));
            if (hist_cap > 0) {
                rt.dflare_ctx_hist_off = try allocArenaOff(&rt.arena, @as(usize, hist_cap) * ntaps * h * 2);
                rt.dflare_ctx_hist_cap = hist_cap;
                rt.dflare_ctx_hist_taps = m82c.dflare_target_layer_count;
                rt.dflare_ctx_hist_hidden = h;
            }
        }
    }

    // Load physical layers `0 .. num_layers`.
    rt.ink_layers = try allocator.alloc(Inkling2LayerWeights, num_layers);
    var built_layers: usize = 0;
    errdefer {
        for (0..built_layers) |i| {
            if (rt.ink_layers[i].loop_ctrl) |*c| c.deinit();
            allocator.free(rt.ink_layers[i].experts);
        }
        allocator.free(rt.ink_layers);
        rt.ink_layers = &.{};
    }

    var nb: [96]u8 = undefined;
    for (0..num_layers) |l| {
        const qn_nm = std.fmt.bufPrint(&nb, "layers.{d}.q_input_norm.weight", .{l}) catch unreachable;
        const q_in_norm = if (ctx.source.find(qn_nm) != null) try bindNormCtx(ctx, qn_nm, h) else null;
        const kn_nm = std.fmt.bufPrint(&nb, "layers.{d}.k_input_norm.weight", .{l}) catch unreachable;
        const k_in_norm = if (ctx.source.find(kn_nm) != null) try bindNormCtx(ctx, kn_nm, h) else null;
        const vn_nm = std.fmt.bufPrint(&nb, "layers.{d}.v_input_norm.weight", .{l}) catch unreachable;
        const v_in_norm = if (ctx.source.find(vn_nm) != null) try bindNormCtx(ctx, vn_nm, h) else null;

        const attn_norm_nm = std.fmt.bufPrint(&nb, "layers.{d}.attn_norm.weight", .{l}) catch unreachable;
        const attn_norm = if (ctx.source.find(attn_norm_nm) != null)
            try bindNormCtx(ctx, attn_norm_nm, h)
        else
            q_in_norm orelse return error.MissingTensor;
        const mlp_norm_nm = std.fmt.bufPrint(&nb, "layers.{d}.mlp_norm.weight", .{l}) catch unreachable;
        const mlp_norm = try bindNormCtx(ctx, mlp_norm_nm, h);

        const as_nm = std.fmt.bufPrint(&nb, "layers.{d}.attn_sconv.weight", .{l}) catch unreachable;
        const attn_sconv_w = if (ctx.source.find(as_nm) != null) try loadConvertedF32Binding(ctx, &rt.arena, rt.buf, as_nm, h * 4) else null;
        const ms_nm = std.fmt.bufPrint(&nb, "layers.{d}.mlp_sconv.weight", .{l}) catch unreachable;
        const mlp_sconv_w = if (ctx.source.find(ms_nm) != null) try loadConvertedF32Binding(ctx, &rt.arena, rt.buf, ms_nm, h * 4) else null;

        const ksw_a = std.fmt.bufPrint(&nb, "layers.{d}.attn.k_sconv.static_w", .{l}) catch unreachable;
        var k_sconv_w: ?metal.BufferBinding = if (ctx.source.find(ksw_a)) |ke|
            try loadSconvF32TransposedIfNeeded(allocator, ctx, &rt.arena, rt.buf, ke, kv_dim)
        else
            null;
        if (k_sconv_w == null) {
            const ksw_b = std.fmt.bufPrint(&nb, "layers.{d}.attn.k_sconv.weight", .{l}) catch unreachable;
            if (ctx.source.find(ksw_b)) |ke| {
                k_sconv_w = try loadSconvF32TransposedIfNeeded(allocator, ctx, &rt.arena, rt.buf, ke, kv_dim);
            }
        }
        const vsw_a = std.fmt.bufPrint(&nb, "layers.{d}.attn.v_sconv.static_w", .{l}) catch unreachable;
        var v_sconv_w: ?metal.BufferBinding = if (ctx.source.find(vsw_a)) |ve|
            try loadSconvF32TransposedIfNeeded(allocator, ctx, &rt.arena, rt.buf, ve, kv_dim)
        else
            null;
        if (v_sconv_w == null) {
            const vsw_b = std.fmt.bufPrint(&nb, "layers.{d}.attn.v_sconv.weight", .{l}) catch unreachable;
            if (ctx.source.find(vsw_b)) |ve| {
                v_sconv_w = try loadSconvF32TransposedIfNeeded(allocator, ctx, &rt.arena, rt.buf, ve, kv_dim);
            }
        }
        const kdp_nm = std.fmt.bufPrint(&nb, "layers.{d}.attn.k_sconv.weight_proj", .{l}) catch unreachable;
        const k_dyn_p = if (findLinearEntry(ctx.source, kdp_nm)) |ke| try bindLinearEntry(ctx, ke, ke.shape[0], h) else null;
        const vdp_nm = std.fmt.bufPrint(&nb, "layers.{d}.attn.v_sconv.weight_proj", .{l}) catch unreachable;
        const v_dyn_p = if (findLinearEntry(ctx.source, vdp_nm)) |ve| try bindLinearEntry(ctx, ve, ve.shape[0], h) else null;

        const wq = try bindLinearCtx(ctx, std.fmt.bufPrint(&nb, "layers.{d}.attn.wq_du", .{l}) catch unreachable, q_dim, h);
        const wk = try bindLinearCtx(ctx, std.fmt.bufPrint(&nb, "layers.{d}.attn.wk_dv", .{l}) catch unreachable, kv_dim, h);
        const wv = try bindLinearCtx(ctx, std.fmt.bufPrint(&nb, "layers.{d}.attn.wv_dv", .{l}) catch unreachable, kv_dim, h);
        const wo = try bindLinearCtx(ctx, std.fmt.bufPrint(&nb, "layers.{d}.attn.wo_ud", .{l}) catch unreachable, h, q_dim);

        const wr_nm = std.fmt.bufPrint(&nb, "layers.{d}.attn.wr_du", .{l}) catch unreachable;
        const wr = if (findLinearEntry(ctx.source, wr_nm) != null) try bindLinearCtx(ctx, wr_nm, cfg.num_attention_heads * m82c.d_rel, h) else null;
        const rel_nm = std.fmt.bufPrint(&nb, "layers.{d}.attn.rel_logits_proj.proj", .{l}) catch unreachable;
        const rel_emb_f16 = if (ctx.source.find(rel_nm)) |re|
            try loadRelEmbF16(allocator, ctx, &rt.arena, rt.buf, re, m82c.rel_extent, m82c.d_rel)
        else
            null;
        const sp_nm = std.fmt.bufPrint(&nb, "layers.{d}.attn.attn_sink_proj", .{l}) catch unreachable;
        const sp = if (findLinearEntry(ctx.source, sp_nm) != null) try bindLinearCtx(ctx, sp_nm, cfg.num_attention_heads, h) else null;
        const ga_nm = std.fmt.bufPrint(&nb, "layers.{d}.attn.gate_attn", .{l}) catch unreachable;
        const ga = if (findLinearEntry(ctx.source, ga_nm) != null) try bindLinearCtx(ctx, ga_nm, q_dim, h) else null;

        const qclip_nm = std.fmt.bufPrint(&nb, "layers.{d}.attn.q_l2_clip.scale", .{l}) catch unreachable;
        var qclip_scale: f32 = 1.0;
        var q_l2_scale_vec: ?[]const f32 = null;
        var q_l2_scale_vec_gpu: ?metal.BufferBinding = null;
        if (ctx.source.find(qclip_nm)) |qe| {
            const qelems = entryElemCount(qe);
            if (qelems == q_dim) {
                q_l2_scale_vec = try loadTensorF32Slice(allocator, ctx, f32_pool, qclip_nm, q_dim);
                q_l2_scale_vec_gpu = try loadConvertedF32Binding(ctx, &rt.arena, rt.buf, qclip_nm, q_dim);
            } else {
                qclip_scale = readOptionalScalarF32(ctx, qclip_nm) orelse 1.0;
            }
        }
        const rclip_nm = std.fmt.bufPrint(&nb, "layers.{d}.attn.rel_l2_clip.scale", .{l}) catch unreachable;
        var rclip_scale: f32 = 1.0;
        var rel_l2_scale_vec: ?[]const f32 = null;
        var rel_l2_scale_vec_gpu: ?metal.BufferBinding = null;
        const rel_q_elems: u32 = cfg.num_attention_heads * m82c.d_rel;
        if (ctx.source.find(rclip_nm)) |re| {
            const relems = entryElemCount(re);
            if (relems == rel_q_elems) {
                rel_l2_scale_vec = try loadTensorF32Slice(allocator, ctx, f32_pool, rclip_nm, rel_q_elems);
                rel_l2_scale_vec_gpu = try loadConvertedF32Binding(ctx, &rt.arena, rt.buf, rclip_nm, rel_q_elems);
            } else {
                rclip_scale = readOptionalScalarF32(ctx, rclip_nm) orelse 1.0;
            }
        }

        // Real Inkling-2-Mini has 23 `attn_res.attn_queries.0..22` (for layers 1..23; layer 0 has no attn query),
        // while synth fixtures emit `0..num_layers-1` (1-to-1). Detect by checking `attn_queries.{num_layers-1}`.
        var nb_last_aq: [64]u8 = undefined;
        const last_aq_nm = std.fmt.bufPrint(&nb_last_aq, "attn_res.attn_queries.{d}", .{num_layers - 1}) catch unreachable;
        const aq_shifted_by_1 = (ctx.source.find(last_aq_nm) == null);
        var ar_attn_q: ?metal.BufferBinding = null;
        var ar_attn_passes: u32 = 1;
        if (!aq_shifted_by_1 or l > 0) {
            const aq_idx = if (aq_shifted_by_1) (l - 1) else l;
            const ara_nm = std.fmt.bufPrint(&nb, "attn_res.attn_queries.{d}", .{aq_idx}) catch unreachable;
            if (ctx.source.find(ara_nm)) |ae| {
                ar_attn_passes = if (ae.rank >= 2 and ae.shape[1] == h) ae.shape[0] else 1;
                ar_attn_q = try loadConvertedF16Binding(ctx, &rt.arena, rt.buf, ara_nm, ar_attn_passes * h);
            }
        }
        const arm_nm = std.fmt.bufPrint(&nb, "attn_res.mlp_queries.{d}", .{l}) catch unreachable;
        var ar_mlp_q: ?metal.BufferBinding = null;
        var ar_mlp_passes: u32 = 1;
        if (ctx.source.find(arm_nm)) |me| {
            ar_mlp_passes = if (me.rank >= 2 and me.shape[1] == h) me.shape[0] else 1;
            ar_mlp_q = try loadConvertedF16Binding(ctx, &rt.arena, rt.buf, arm_nm, ar_mlp_passes * h);
        }

        // Loop Controller (`G29` GPU default with one-time `sme2_panel64` deinterleave, or opt-in C26 L2 SME2)
        const ctrl_probe = std.fmt.bufPrint(&nb, "layers.{d}.controllers.attn.down_weight", .{l}) catch unreachable;
        var loop_ctrl: ?ink_ctrl.Inkling2LoopController = null;
        var gpu_ctrl: ?Inkling2GpuController = null;
        if (ctx.source.find(ctrl_probe)) |probe_e| {
            const p_dt: sme.PanelDType = switch (try probe_e.dataType()) {
                .f16 => .f16,
                .q4_affine => .q4_affine,
                .q3_affine => .q3_affine,
                .q2_affine => .q2_affine,
                else => .bf16,
            };
            const ctrl_in_dim: u32 = if (probe_e.rank >= 2 and probe_e.shape[1] > 0) probe_e.shape[1] else h;
            const gate_rank: u32 = if (probe_e.rank >= 2 and probe_e.shape[0] > 0) probe_e.shape[0] else ink_ctrl.GATE_RANK;
            var nb_kfg: [96]u8 = undefined;
            const kfg_nm = std.fmt.bufPrint(&nb_kfg, "layers.{d}.controllers.k.forget_weight", .{l}) catch unreachable;
            const kv_ctrl_dim: u32 = if (ctx.source.find(kfg_nm)) |ke| ke.shape[0] else kv_dim;
            const out_dims = [ink_ctrl.NUM_BRANCHES]u32{ h, kv_ctrl_dim, kv_ctrl_dim, h };
            if (rt.use_sme2_controllers) {
                var branches: [ink_ctrl.NUM_BRANCHES]ink_ctrl.BranchConfig = undefined;
                for (0..ink_ctrl.NUM_BRANCHES) |bi| {
                    branches[bi] = .{
                        .state_dim = out_dims[bi],
                        .down_dtype = p_dt,
                        .forget_dtype = p_dt,
                        .gain_dtype = p_dt,
                        .gain_activation = .silu,
                    };
                }
                const ctrl_cfg: ink_ctrl.ControllerConfig = .{
                    .hidden_size = ctrl_in_dim,
                    .gate_rank = gate_rank,
                    .max_m = max_m,
                    .rms_eps = cfg.rms_norm_eps,
                    .l2_limit_bytes = 8 * 1024 * 1024,
                    .branches = branches,
                    .budget_override = ctx.wired_budget,
                };
                if (ctx.amf_path) |ap| {
                    var af = try amf.AmfFile.open(allocator, ap);
                    defer af.close(allocator);
                    loop_ctrl = try ink_ctrl.Inkling2LoopController.loadFromAmfLayer(allocator, &af, @intCast(l), ctrl_cfg);
                } else {
                    loop_ctrl = try loadControllerFromParsedBytes(allocator, ctx, @intCast(l), ctrl_cfg);
                }
            } else {
                gpu_ctrl = try loadGpuControllerToArena(ctx, &rt.arena, rt.buf, @intCast(l), ctrl_in_dim, gate_rank, out_dims);
            }
        }

        // Router + experts + optional zero-readback `G29` indirect MoE descriptor (`cold == 0`)
        const router = try bindLinearCtx(ctx, std.fmt.bufPrint(&nb, "layers.{d}.mlp.gate", .{l}) catch unreachable, num_experts, h);
        const rb_nm = std.fmt.bufPrint(&nb, "layers.{d}.mlp.gate.bias", .{l}) catch unreachable;
        const rb_f32 = if (ctx.source.find(rb_nm) != null) try loadConvertedF32Binding(ctx, &rt.arena, rt.buf, rb_nm, num_experts) else null;

        const experts = try bindLayerExperts(allocator, ctx, l, num_experts, h, cfg.intermediate_size, "w13");
        var cold_cnt_l: u32 = 0;
        for (experts) |ew| {
            if (ew == .cold) cold_cnt_l += 1;
        }
        const indirect_moe = if (cold_cnt_l == 0)
            try buildInkling2IndirectMoeMeta(&rt.arena, rt.buf, experts)
        else
            null;

        rt.ink_layers[l] = .{
            .attn_norm = attn_norm,
            .q_input_norm = q_in_norm,
            .k_input_norm = k_in_norm,
            .v_input_norm = v_in_norm,
            .mlp_norm = mlp_norm,
            .attn_sconv_w = attn_sconv_w,
            .mlp_sconv_w = mlp_sconv_w,
            .k_sconv_w = k_sconv_w,
            .v_sconv_w = v_sconv_w,
            .k_sconv_dyn_proj = k_dyn_p,
            .v_sconv_dyn_proj = v_dyn_p,
            .wq = wq,
            .wk = wk,
            .wv = wv,
            .wo = wo,
            .wr = wr,
            .rel_emb_f16 = rel_emb_f16,
            .attn_sink_proj = sp,
            .gate_attn = ga,
            .q_l2_clip_scale = if (qclip_scale > 0.0 and std.math.isFinite(qclip_scale)) qclip_scale else 1.0,
            .rel_l2_clip_scale = if (rclip_scale > 0.0 and std.math.isFinite(rclip_scale)) rclip_scale else 1.0,
            .q_l2_scale_vec = q_l2_scale_vec,
            .rel_l2_scale_vec = rel_l2_scale_vec,
            .q_l2_scale_vec_gpu = q_l2_scale_vec_gpu,
            .rel_l2_scale_vec_gpu = rel_l2_scale_vec_gpu,
            .attn_res_attn_q = ar_attn_q,
            .attn_res_attn_q_passes = ar_attn_passes,
            .attn_res_mlp_q = ar_mlp_q,
            .attn_res_mlp_q_passes = ar_mlp_passes,
            .loop_ctrl = loop_ctrl,
            .gpu_ctrl = gpu_ctrl,
            .router = router,
            .router_bias_f32 = rb_f32,
            .experts = experts,
            .indirect_moe = indirect_moe,
            .cold_expert_count = cold_cnt_l,
            .is_local = (l < m82c.local_layer_flags.len and m82c.local_layer_flags[l]),
        };
        built_layers += 1;
    }

    // Allocate per-effective-step `HybridKvRing`s:
    // Local SWA layers (`40/48` in Mini) -> fixed `1024`-token circular `HybridKvRing`!
    // Global layers (`8/48` in Mini) -> full `512`-token paged `HybridKvRing`!
    rt.ink_steps = try allocator.alloc(Inkling2StepState, eff_steps);
    var built_steps: usize = 0;
    errdefer {
        for (0..built_steps) |si| {
            rt.ink_steps[si].kv_ring.removeResident(ctx.compute) catch |err| {
                std.debug.panic("errdefer kv_ring.removeResident failed: {}", .{err});
            };
            rt.ink_steps[si].kv_ring.deinit(allocator);
        }
        allocator.free(rt.ink_steps);
        rt.ink_steps = &.{};
    }

    const eff_swa_win: u32 = @max(@as(u32, 1), @min(m82c.sliding_window_size, ctx.max_seq_len));
    const swa_tiles: u32 = @max(@as(u32, 1), (eff_swa_win + hybrid.QSA_SWA_WINDOW_TOKENS - 1) / hybrid.QSA_SWA_WINDOW_TOKENS);
    const full_tiles: u32 = @max(@as(u32, 1), (ctx.max_seq_len + hybrid.QSA_FULL_CHUNK_TOKENS - 1) / hybrid.QSA_FULL_CHUNK_TOKENS);

    for (0..eff_steps) |si| {
        const phys_l = rt.physicalLayerForStep(@intCast(si), cfg);
        const is_local = rt.ink_layers[phys_l].is_local;
        var kv_ring = try hybrid.HybridKvRing.init(allocator, ctx.device, &rt.arena, .{
            .mode = if (is_local) .sliding_window_128 else .full_chunk_512,
            .num_kv_heads = cfg.num_key_value_heads,
            .head_dim = cfg.head_dim,
            .num_tiles = if (is_local) swa_tiles else full_tiles,
            .window_size_override = if (is_local) eff_swa_win else null,
        });
        errdefer kv_ring.deinit(allocator);
        try kv_ring.makeResident(ctx.compute);
        var ring_bytes: u64 = 0;
        for (kv_ring.k_tiles) |kt| ring_bytes += (try kt.info()).allocBytes();
        for (kv_ring.v_tiles) |vt| ring_bytes += (try vt.info()).allocBytes();
        rt.total_kv_wired_bytes += ring_bytes;
        if (is_local) {
            rt.stats.inkling2_swa_kv_bytes += ring_bytes;
        } else {
            rt.stats.inkling2_global_kv_bytes += ring_bytes;
        }

        rt.ink_steps[si] = .{
            .kv_ring = kv_ring,
            .tree_k_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * kv_dim * 2),
            .tree_v_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * kv_dim * 2),
            .tree_attn_sconv_in_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * h * 2),
            .tree_mlp_sconv_in_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * h * 2),
            .tree_k_sconv_in_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * kv_dim * 2),
            .tree_v_sconv_in_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * kv_dim * 2),
            .sconv_attn_slot = @intCast(si * 2),
            .sconv_mlp_slot = @intCast(si * 2 + 1),
            .sconv_k_slot = @intCast(si * 2),
            .sconv_v_slot = @intCast(si * 2 + 1),
            .is_local = is_local,
        };
        built_steps += 1;
    }
}

fn loadHy4(
    allocator: std.mem.Allocator,
    rt: *M82Runtime,
    ctx: M82LoadContext,
    f32_pool: *std.ArrayListUnmanaged([]f32),
) !void {
    const cfg = ctx.config;
    const m82c = &cfg.m82_cfg;
    const h = cfg.hidden_size;
    const max_m = ctx.max_tree_nodes;
    const max_seq = ctx.max_seq_len;
    const num_layers = cfg.num_hidden_layers;
    const n_q = cfg.num_attention_heads;
    const q_rank = cfg.q_lora_rank;
    const kv_rank = cfg.kv_lora_rank;
    const qk_nope = cfg.qk_nope_head_dim;
    const qk_rope = cfg.qk_rope_head_dim;
    const v_dim = cfg.v_head_dim;
    const idx_dims = inferHy4IndexDims(cfg, ctx.source);
    const idx_d = idx_dims.idx_d;
    const idx_h = idx_dims.idx_h;
    rt.hy4_index_head_dim = idx_d;
    rt.hy4_index_n_heads = idx_h;
    const hc_mult = m82c.hc_mult;
    const top_k = @max(cfg.num_experts_per_tok, @as(u32, 1));

    rt.hy4_hc_streams_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * hc_mult * h * 2);
    rt.hy4_hc_streams_host = rt.sliceF16(rt.hy4_hc_streams_off, @as(usize, max_m) * hc_mult * h);
    rt.hy4_hc_tmp_f32 = try allocator.alloc(f32, @as(usize, hc_mult) * h);
    rt.hy4_q_a_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * q_rank * 2);
    rt.hy4_q_b_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * n_q * (qk_nope + qk_rope) * 2);
    rt.hy4_q_nope_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * n_q * qk_nope * 2);
    rt.hy4_q_pe_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * n_q * @max(qk_rope, @as(u32, 1)) * 2);
    rt.hy4_q_abs_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * n_q * kv_rank * 2);
    rt.hy4_kv_a_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * (kv_rank + qk_rope) * 2);
    rt.hy4_kv_lat_cur_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * kv_rank * 2);
    rt.hy4_kv_rope_cur_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * @max(qk_rope, @as(u32, 1)) * 2);
    rt.hy4_mla_out_lat_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * n_q * kv_rank * 2);
    rt.hy4_attn_v_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * n_q * v_dim * 2);
    rt.hy4_lin_gate_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * n_q * v_dim * 2);
    rt.hy4_q_idx_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * idx_h * idx_d * 2);
    rt.hy4_k_idx_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * idx_d * 2);
    rt.hy4_hw_idx_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * idx_h * 2);
    rt.hy4_latent_x_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * h * 2);
    rt.hy4_latent_exp_out_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * top_k * h * 2);
    rt.hy4_latent_comb_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * h * 2);
    rt.hy4_shared_out_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * h * 2);
    rt.hy4_engram_in_f16_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * @max(h, @as(u32, 256)) * 2);
    rt.hy4_engram_k_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * h * 2);
    rt.hy4_engram_v_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * h * 2);

    const cand_k = @min(m82c.candidate_topk, @max(max_seq, @as(u32, 16)));
    const comp_r = if (m82c.use_compressed_indexer) m82c.compress_ratio else 1;
    const grp_k = if (m82c.use_compressed_indexer)
        @min(m82c.group_topk, @max((max_seq + comp_r - 1) / comp_r, @as(u32, 1)))
    else
        0;
    rt.hy4_index_pool = try ink_gpu.IndexPoolBuffer.init(&rt.arena, .{
        .max_kv_len = max_seq,
        .max_nodes = max_m,
        .idx_dim = idx_d,
        .compress_ratio = comp_r,
        .group_topk = grp_k,
        .candidate_topk = cand_k,
    });
    @memset(rt.hy4_index_pool.?.bytes(), 0);

    // Optional top-level `hc_head.*`
    if (ctx.source.find("hc_head.hc_head_base")) |_| {
        const fn_e = ctx.source.find("hc_head.hc_head_fn") orelse return error.MissingTensor;
        const in_dim: u32 = fn_e.shape[1];
        rt.hy4_hc_head = .{
            .hc_base = try loadTensorF32Slice(allocator, ctx, f32_pool, "hc_head.hc_head_base", hc_mult),
            .hc_fn = try loadTensorF32Slice(allocator, ctx, f32_pool, "hc_head.hc_head_fn", hc_mult * in_dim),
            .hc_scale = try loadTensorF32Slice(allocator, ctx, f32_pool, "hc_head.hc_head_scale", 1),
            .in_dim = in_dim,
        };
    }

    // DFly2 target KV f32 scratch buffers
    if (m82c.dfly_target_kv_count > 0) {
        const kv_d = @min(kv_rank, @max(cfg.num_key_value_heads * v_dim, @as(u32, 32)));
        rt.dfly_kv_dim = kv_d;
        const cap_toks = @min(max_seq, @as(u32, 256));
        const total_elems = @as(usize, m82c.dfly_target_kv_count) * cap_toks * kv_d;
        rt.dfly_kv_k_buf = try allocator.alloc(f32, total_elems);
        rt.dfly_kv_v_buf = try allocator.alloc(f32, total_elems);
        @memset(rt.dfly_kv_k_buf, 0);
        @memset(rt.dfly_kv_v_buf, 0);
    }

    rt.hy4_layers = try allocator.alloc(Hy4LayerWeights, num_layers);
    var built: usize = 0;
    errdefer {
        for (0..built) |i| {
            if (rt.hy4_layers[i].experts.len > 0) allocator.free(rt.hy4_layers[i].experts);
        }
        allocator.free(rt.hy4_layers);
        rt.hy4_layers = &.{};
    }

    var nb: [96]u8 = undefined;
    for (0..num_layers) |l| {
        const in_norm = try bindNormCtx(ctx, std.fmt.bufPrint(&nb, "layers.{d}.input_layernorm.weight", .{l}) catch unreachable, h);
        const post_norm = try bindNormCtx(ctx, std.fmt.bufPrint(&nb, "layers.{d}.post_attention_layernorm.weight", .{l}) catch unreachable, h);

        // Optional iHC pre/post weights
        var hc_attn: ?Hy4HcWeights = null;
        const hca_b = std.fmt.bufPrint(&nb, "layers.{d}.hc_attn_layer.hc_pre.hc_base", .{l}) catch unreachable;
        if (ctx.source.find(hca_b) != null) {
            const fn_nm = std.fmt.bufPrint(&nb, "layers.{d}.hc_attn_layer.hc_pre.hc_fn", .{l}) catch unreachable;
            const fn_e = ctx.source.find(fn_nm) orelse return error.MissingTensor;
            const in_d: u32 = fn_e.shape[1];
            hc_attn = .{
                .hc_base = try loadTensorF32Slice(allocator, ctx, f32_pool, std.fmt.bufPrint(&nb, "layers.{d}.hc_attn_layer.hc_pre.hc_base", .{l}) catch unreachable, 2 * hc_mult),
                .hc_fn = try loadTensorF32Slice(allocator, ctx, f32_pool, std.fmt.bufPrint(&nb, "layers.{d}.hc_attn_layer.hc_pre.hc_fn", .{l}) catch unreachable, 2 * hc_mult * in_d),
                .hc_scale = try loadTensorF32Slice(allocator, ctx, f32_pool, std.fmt.bufPrint(&nb, "layers.{d}.hc_attn_layer.hc_pre.hc_scale", .{l}) catch unreachable, 2),
                .in_dim = in_d,
            };
        }
        var hc_mlp: ?Hy4HcWeights = null;
        const hcm_b = std.fmt.bufPrint(&nb, "layers.{d}.hc_mlp_layer.hc_pre.hc_base", .{l}) catch unreachable;
        if (ctx.source.find(hcm_b) != null) {
            const fn_nm = std.fmt.bufPrint(&nb, "layers.{d}.hc_mlp_layer.hc_pre.hc_fn", .{l}) catch unreachable;
            const fn_e = ctx.source.find(fn_nm) orelse return error.MissingTensor;
            const in_d: u32 = fn_e.shape[1];
            hc_mlp = .{
                .hc_base = try loadTensorF32Slice(allocator, ctx, f32_pool, std.fmt.bufPrint(&nb, "layers.{d}.hc_mlp_layer.hc_pre.hc_base", .{l}) catch unreachable, 2 * hc_mult),
                .hc_fn = try loadTensorF32Slice(allocator, ctx, f32_pool, std.fmt.bufPrint(&nb, "layers.{d}.hc_mlp_layer.hc_pre.hc_fn", .{l}) catch unreachable, 2 * hc_mult * in_d),
                .hc_scale = try loadTensorF32Slice(allocator, ctx, f32_pool, std.fmt.bufPrint(&nb, "layers.{d}.hc_mlp_layer.hc_pre.hc_scale", .{l}) catch unreachable, 2),
                .in_dim = in_d,
            };
        }

        // MLA projections
        const q_a = try bindLinearCtx(ctx, std.fmt.bufPrint(&nb, "layers.{d}.self_attn.q_a_proj", .{l}) catch unreachable, q_rank, h);
        const qa_n_nm = std.fmt.bufPrint(&nb, "layers.{d}.self_attn.q_a_layernorm.weight", .{l}) catch unreachable;
        const q_a_norm = if (ctx.source.find(qa_n_nm) != null) try bindNormCtx(ctx, qa_n_nm, q_rank) else null;
        const q_b = try bindLinearCtx(ctx, std.fmt.bufPrint(&nb, "layers.{d}.self_attn.q_b_proj", .{l}) catch unreachable, n_q * (qk_nope + qk_rope), q_rank);

        const kva_stem_a = std.fmt.bufPrint(&nb, "layers.{d}.self_attn.kv_a_proj_with_mqa", .{l}) catch unreachable;
        const kv_a = if (findLinearEntry(ctx.source, kva_stem_a) != null)
            try bindLinearCtx(ctx, kva_stem_a, kv_rank + qk_rope, h)
        else
            try bindLinearCtx(ctx, std.fmt.bufPrint(&nb, "layers.{d}.self_attn.kv_a_mqa", .{l}) catch unreachable, kv_rank + qk_rope, h);
        const kva_n_nm = std.fmt.bufPrint(&nb, "layers.{d}.self_attn.kv_a_layernorm.weight", .{l}) catch unreachable;
        const kv_a_norm = if (ctx.source.find(kva_n_nm) != null) try bindNormCtx(ctx, kva_n_nm, kv_rank) else null;

        // Unpack `kv_b_proj [n_q * (qk_nope + v_dim), kv_rank]` into GPU f16 `W_uk_t` & `W_uv`.
        const kvb_nm = std.fmt.bufPrint(&nb, "layers.{d}.self_attn.kv_b_proj", .{l}) catch unreachable;
        const w_uk_t_off = try allocArenaOff(&rt.arena, @as(usize, n_q) * kv_rank * qk_nope * 2);
        const w_uv_off = try allocArenaOff(&rt.arena, @as(usize, n_q) * v_dim * kv_rank * 2);
        try unpackHy4KvBProj(allocator, ctx, kvb_nm, n_q, qk_nope, v_dim, kv_rank, rt.sliceF16(w_uk_t_off, @as(usize, n_q) * kv_rank * qk_nope), rt.sliceF16(w_uv_off, @as(usize, n_q) * v_dim * kv_rank));

        const o_proj = try bindLinearCtx(ctx, std.fmt.bufPrint(&nb, "layers.{d}.self_attn.o_proj", .{l}) catch unreachable, h, n_q * v_dim);
        const lg_nm = std.fmt.bufPrint(&nb, "layers.{d}.self_attn.linear_gate", .{l}) catch unreachable;
        const lin_gate = if (findLinearEntry(ctx.source, lg_nm) != null) try bindLinearCtx(ctx, lg_nm, n_q * v_dim, h) else null;
        const sink_nm = std.fmt.bufPrint(&nb, "layers.{d}.self_attn.learnable_sink_param", .{l}) catch unreachable;
        const sink_f32 = if (ctx.source.find(sink_nm) != null) try loadConvertedF32Binding(ctx, &rt.arena, rt.buf, sink_nm, n_q) else null;
        const ppn_nm = std.fmt.bufPrint(&nb, "layers.{d}.self_attn.post_proj_norm.weight", .{l}) catch unreachable;
        const post_proj_norm = if (ctx.source.find(ppn_nm) != null) try bindNormCtx(ctx, ppn_nm, h) else null;

        // IndexCache / IndexPool weights
        const idx_mode: ink_gpu.IndexCacheMode = if (l < m82c.indexer_modes.len) m82c.indexer_modes[l] else .full;
        const is_swa: bool = (l < m82c.is_swa_layer.len and m82c.is_swa_layer[l]);
        var indexer: ?Hy4IndexerWeights = null;
        var k_fine_off: u64 = 0;
        if (!is_swa and idx_mode == .full) {
            var nb_cidx: [96]u8 = undefined;
            var nb_std: [96]u8 = undefined;
            const cidx_q = std.fmt.bufPrint(&nb_cidx, "layers.{d}.self_attn.cidx.query_projection", .{l}) catch unreachable;
            const std_q = std.fmt.bufPrint(&nb_std, "layers.{d}.self_attn.indexer.wq_b", .{l}) catch unreachable;
            if (findLinearEntry(ctx.source, cidx_q) != null) {
                const q_p = try bindLinearCtx(ctx, cidx_q, idx_h * idx_d, q_rank);
                const tk_p = try bindLinearCtx(ctx, std.fmt.bufPrint(&nb, "layers.{d}.self_attn.cidx.tk_proj", .{l}) catch unreachable, idx_d, h);
                const tkn_nm = std.fmt.bufPrint(&nb, "layers.{d}.self_attn.cidx.token_key_norm.weight", .{l}) catch unreachable;
                const tkn = if (ctx.source.find(tkn_nm) != null) try bindNormCtx(ctx, tkn_nm, idx_d) else null;
                const hw_nm = std.fmt.bufPrint(&nb, "layers.{d}.self_attn.cidx.hw_proj", .{l}) catch unreachable;
                const hwp = if (findLinearEntry(ctx.source, hw_nm) != null) try bindLinearCtx(ctx, hw_nm, idx_h, h) else null;
                indexer = .{
                    .q_proj = q_p,
                    .k_proj = tk_p,
                    .k_norm = tkn,
                    .head_w_proj = hwp,
                    .is_compressed = true,
                };
                k_fine_off = try allocArenaOff(&rt.arena, @as(usize, max_seq) * idx_d * 2);
            } else if (findLinearEntry(ctx.source, std_q) != null) {
                const q_p = try bindLinearCtx(ctx, std_q, idx_h * idx_d, q_rank);
                const wk_p = try bindLinearCtx(ctx, std.fmt.bufPrint(&nb, "layers.{d}.self_attn.indexer.wk", .{l}) catch unreachable, idx_d, h);
                const kn_nm = std.fmt.bufPrint(&nb, "layers.{d}.self_attn.indexer.k_norm.weight", .{l}) catch unreachable;
                const kn = if (ctx.source.find(kn_nm) != null) try bindNormCtx(ctx, kn_nm, idx_d) else null;
                const wp_nm = std.fmt.bufPrint(&nb, "layers.{d}.self_attn.indexer.weights_proj", .{l}) catch unreachable;
                const wp = if (findLinearEntry(ctx.source, wp_nm) != null) try bindLinearCtx(ctx, wp_nm, idx_h, h) else null;
                indexer = .{
                    .q_proj = q_p,
                    .k_proj = wk_p,
                    .k_norm = kn,
                    .head_w_proj = wp,
                    .is_compressed = false,
                };
                k_fine_off = try allocArenaOff(&rt.arena, @as(usize, max_seq) * idx_d * 2);
            }
        }

        // Per-layer MLA KV caches & uncommitted tree scratch
        const kv_lat_off = try allocArenaOff(&rt.arena, @as(usize, max_seq) * kv_rank * 2);
        const kv_rope_off = try allocArenaOff(&rt.arena, @as(usize, max_seq) * @max(qk_rope, @as(u32, 1)) * 2);
        const tree_lat_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * kv_rank * 2);
        const tree_rope_off = try allocArenaOff(&rt.arena, @as(usize, max_m) * @max(qk_rope, @as(u32, 1)) * 2);

        // Optional C26 Engram layer projections
        var engram_w: ?Hy4EngramLayerWeights = null;
        const eg_kp_nm = std.fmt.bufPrint(&nb, "layers.{d}.engram_layer.key_projs.0", .{l}) catch unreachable;
        if (findLinearEntry(ctx.source, eg_kp_nm)) |kpe| {
            const eg_dim = kpe.shape[1];
            const kp = try bindLinearCtx(ctx, eg_kp_nm, h, eg_dim);
            const vp = try bindLinearCtx(ctx, std.fmt.bufPrint(&nb, "layers.{d}.engram_layer.value_proj", .{l}) catch unreachable, h, eg_dim);
            const hcs_nm = std.fmt.bufPrint(&nb, "layers.{d}.engram_layer.unified_per_hc_scales.0", .{l}) catch unreachable;
            const hcs = if (ctx.source.find(hcs_nm) != null) try loadTensorF32Slice(allocator, ctx, f32_pool, hcs_nm, hc_mult) else null;
            const pg_nm = std.fmt.bufPrint(&nb, "layers.{d}.engram_layer.unified_pre_gamma", .{l}) catch unreachable;
            const pg = if (ctx.source.find(pg_nm) != null) try loadTensorF32Slice(allocator, ctx, f32_pool, pg_nm, h) else null;
            engram_w = .{
                .key_proj = kp,
                .value_proj = vp,
                .hc_scales = hcs,
                .pre_gamma = pg,
                .engram_dim = eg_dim,
            };
        }

        // FFN: Dense OR Sparse MoE / LatentMoE
        const is_sparse = (l < m82c.is_sparse_mlp.len and m82c.is_sparse_mlp[l]);
        var lw: Hy4LayerWeights = .{
            .input_layernorm = in_norm,
            .post_attention_layernorm = post_norm,
            .hc_attn = hc_attn,
            .hc_mlp = hc_mlp,
            .q_a_proj = q_a,
            .q_a_layernorm = q_a_norm,
            .q_b_proj = q_b,
            .kv_a_proj = kv_a,
            .kv_a_layernorm = kv_a_norm,
            .w_uk_t_off = w_uk_t_off,
            .w_uv_off = w_uv_off,
            .o_proj = o_proj,
            .linear_gate = lin_gate,
            .sink_params_f32 = sink_f32,
            .post_proj_norm = post_proj_norm,
            .indexer_mode = idx_mode,
            .is_swa = is_swa,
            .indexer = indexer,
            .k_fine_cache_off = k_fine_off,
            .kv_latent_off = kv_lat_off,
            .kv_rope_off = kv_rope_off,
            .tree_kv_latent_off = tree_lat_off,
            .tree_kv_rope_off = tree_rope_off,
            .engram = engram_w,
            .is_sparse_mlp = is_sparse,
        };

        if (!is_sparse) {
            lw.dense_gate = try bindLinearCtx(ctx, std.fmt.bufPrint(&nb, "layers.{d}.mlp.gate_proj", .{l}) catch unreachable, cfg.intermediate_size, h);
            lw.dense_up = try bindLinearCtx(ctx, std.fmt.bufPrint(&nb, "layers.{d}.mlp.up_proj", .{l}) catch unreachable, cfg.intermediate_size, h);
            lw.dense_down = try bindLinearCtx(ctx, std.fmt.bufPrint(&nb, "layers.{d}.mlp.down_proj", .{l}) catch unreachable, h, cfg.intermediate_size);
            const pfn_nm = std.fmt.bufPrint(&nb, "layers.{d}.mlp.post_fc2_norm.weight", .{l}) catch unreachable;
            if (ctx.source.find(pfn_nm) != null) lw.post_fc2_norm = try bindNormCtx(ctx, pfn_nm, h);
        } else {
            lw.router = try bindLinearCtx(ctx, std.fmt.bufPrint(&nb, "layers.{d}.mlp.gate", .{l}) catch unreachable, cfg.num_experts, h);
            const eb_nm = std.fmt.bufPrint(&nb, "layers.{d}.mlp.gate.e_score_correction_bias", .{l}) catch unreachable;
            if (ctx.source.find(eb_nm) != null) {
                lw.e_score_bias_f32 = try loadConvertedF32Binding(ctx, &rt.arena, rt.buf, eb_nm, cfg.num_experts);
            }
            var nb_ld: [96]u8 = undefined;
            var nb_lu: [96]u8 = undefined;
            const ld_nm = std.fmt.bufPrint(&nb_ld, "layers.{d}.mlp.routed_expert_down_proj", .{l}) catch unreachable;
            const lu_nm = std.fmt.bufPrint(&nb_lu, "layers.{d}.mlp.routed_expert_up_proj", .{l}) catch unreachable;
            var exp_in_dim: u32 = h;
            if (findLinearEntry(ctx.source, ld_nm)) |lde| {
                const lat_d = lde.shape[0];
                lw.latent_down_proj = try bindLinearCtx(ctx, ld_nm, lat_d, h);
                lw.latent_up_proj = try bindLinearCtx(ctx, lu_nm, h, lat_d);
                lw.latent_dim = lat_d;
                exp_in_dim = lat_d;
            }
            const cn_nm = std.fmt.bufPrint(&nb, "layers.{d}.mlp.combine_output_norm.weight", .{l}) catch unreachable;
            if (ctx.source.find(cn_nm) != null) lw.combine_output_norm = try bindNormCtx(ctx, cn_nm, h);

            // Optional shared expert
            var nb_sga: [96]u8 = undefined;
            var nb_sgb: [96]u8 = undefined;
            const sg_a = std.fmt.bufPrint(&nb_sga, "layers.{d}.mlp.shared_experts.gate_proj", .{l}) catch unreachable;
            const sg_b = std.fmt.bufPrint(&nb_sgb, "layers.{d}.mlp.shared_exp.gate_proj", .{l}) catch unreachable;
            const sg_entry_opt = findLinearEntry(ctx.source, sg_a) orelse findLinearEntry(ctx.source, sg_b);
            if (sg_entry_opt) |sge| {
                const use_long = (findLinearEntry(ctx.source, sg_a) != null);
                const sh_inter = sge.shape[0];
                if (use_long) {
                    lw.shared_gate = try bindLinearCtx(ctx, std.fmt.bufPrint(&nb, "layers.{d}.mlp.shared_experts.gate_proj", .{l}) catch unreachable, sh_inter, h);
                    lw.shared_up = try bindLinearCtx(ctx, std.fmt.bufPrint(&nb, "layers.{d}.mlp.shared_experts.up_proj", .{l}) catch unreachable, sh_inter, h);
                    lw.shared_down = try bindLinearCtx(ctx, std.fmt.bufPrint(&nb, "layers.{d}.mlp.shared_experts.down_proj", .{l}) catch unreachable, h, sh_inter);
                } else {
                    lw.shared_gate = try bindLinearCtx(ctx, std.fmt.bufPrint(&nb, "layers.{d}.mlp.shared_exp.gate_proj", .{l}) catch unreachable, sh_inter, h);
                    lw.shared_up = try bindLinearCtx(ctx, std.fmt.bufPrint(&nb, "layers.{d}.mlp.shared_exp.up_proj", .{l}) catch unreachable, sh_inter, h);
                    lw.shared_down = try bindLinearCtx(ctx, std.fmt.bufPrint(&nb, "layers.{d}.mlp.shared_exp.down_proj", .{l}) catch unreachable, h, sh_inter);
                }
            }

            lw.experts = try bindLayerExperts(allocator, ctx, l, cfg.num_experts, exp_in_dim, cfg.moe_intermediate_size, "gate_up_proj");
        }

        rt.hy4_layers[l] = lw;
        built += 1;
    }
}

fn bindLayerExperts(
    allocator: std.mem.Allocator,
    ctx: M82LoadContext,
    l: usize,
    num_experts: u32,
    in_dim: u32,
    inter: u32,
    fused_stem: []const u8,
) ![]M82ExpertWeight {
    const out = try allocator.alloc(M82ExpertWeight, num_experts);
    errdefer allocator.free(out);
    var nb_fused: [96]u8 = undefined;
    var nb_alt: [96]u8 = undefined;
    var nb_down: [96]u8 = undefined;
    var nb_gate: [96]u8 = undefined;
    var nb_up: [96]u8 = undefined;

    for (0..num_experts) |e| {
        // Check fused `w13` / `gate_up_proj` first, then split `gate_proj` + `up_proj`.
        const fused_nm = std.fmt.bufPrint(&nb_fused, "layers.{d}.mlp.experts.{d}.{s}", .{ l, e, fused_stem }) catch unreachable;
        const alt_fused_nm = std.fmt.bufPrint(&nb_alt, "layers.{d}.mlp.experts.{d}.{s}", .{
            l,
            e,
            if (std.mem.eql(u8, fused_stem, "w13")) "gate_up_proj" else "w13",
        }) catch unreachable;
        const down_nm = std.fmt.bufPrint(&nb_down, "layers.{d}.mlp.experts.{d}.down_proj", .{ l, e }) catch unreachable;
        const gate_nm = std.fmt.bufPrint(&nb_gate, "layers.{d}.mlp.experts.{d}.gate_proj", .{ l, e }) catch unreachable;
        const up_nm = std.fmt.bufPrint(&nb_up, "layers.{d}.mlp.experts.{d}.up_proj", .{ l, e }) catch unreachable;

        const w1_e_opt = findLinearEntry(ctx.source, fused_nm) orelse findLinearEntry(ctx.source, alt_fused_nm);
        const w2_e = findLinearEntry(ctx.source, down_nm) orelse return error.MissingTensor;
        const is_cold = (try w2_e.tensorKind()) == .cold_expert;

        if (w1_e_opt) |w1_e| {
            if (is_cold or (try w1_e.tensorKind()) == .cold_expert) {
                out[e] = .{ .cold = .{
                    .layer = @intCast(l),
                    .expert = @intCast(e),
                    .w1_offset = w1_e.blob_offset,
                    .w1_bytes = w1_e.blob_bytes,
                    .w1_params_offset = w1_e.params_offset,
                    .w1_params_bytes = w1_e.params_bytes,
                    .w1_amf_dtype = try w1_e.dataType(),
                    .w1_layout = try w1_e.tensorLayout(),
                    .w1_param_kind = try w1_e.quantParamKind(),
                    .w1_group_size = if (w1_e.group_size > 0) w1_e.group_size else 64,
                    .w1_n = 2 * inter,
                    .w1_k = in_dim,
                    .has_separate_up = false,
                    .w2_offset = w2_e.blob_offset,
                    .w2_bytes = w2_e.blob_bytes,
                    .w2_params_offset = w2_e.params_offset,
                    .w2_params_bytes = w2_e.params_bytes,
                    .w2_amf_dtype = try w2_e.dataType(),
                    .w2_layout = try w2_e.tensorLayout(),
                    .w2_param_kind = try w2_e.quantParamKind(),
                    .w2_group_size = if (w2_e.group_size > 0) w2_e.group_size else 64,
                    .w2_n = in_dim,
                    .w2_k = inter,
                } };
            } else {
                const w1_b = try bindLinearEntry(ctx, w1_e, 2 * inter, in_dim);
                const w2_b = try bindLinearEntry(ctx, w2_e, in_dim, inter);
                out[e] = .{ .hot = .{ .w1 = w1_b, .w_up = null, .w2 = w2_b } };
            }
        } else {
            const ge = findLinearEntry(ctx.source, gate_nm) orelse return error.MissingTensor;
            const ue = findLinearEntry(ctx.source, up_nm) orelse return error.MissingTensor;
            if (is_cold or (try ge.tensorKind()) == .cold_expert) {
                out[e] = .{ .cold = .{
                    .layer = @intCast(l),
                    .expert = @intCast(e),
                    .w1_offset = ge.blob_offset,
                    .w1_bytes = ge.blob_bytes,
                    .w1_params_offset = ge.params_offset,
                    .w1_params_bytes = ge.params_bytes,
                    .w1_amf_dtype = try ge.dataType(),
                    .w1_layout = try ge.tensorLayout(),
                    .w1_param_kind = try ge.quantParamKind(),
                    .w1_group_size = if (ge.group_size > 0) ge.group_size else 64,
                    .w1_n = inter,
                    .w1_k = in_dim,
                    .w_up_offset = ue.blob_offset,
                    .w_up_bytes = ue.blob_bytes,
                    .w_up_params_offset = ue.params_offset,
                    .w_up_params_bytes = ue.params_bytes,
                    .w_up_amf_dtype = try ue.dataType(),
                    .w_up_layout = try ue.tensorLayout(),
                    .w_up_param_kind = try ue.quantParamKind(),
                    .w_up_group_size = if (ue.group_size > 0) ue.group_size else 64,
                    .has_separate_up = true,
                    .w2_offset = w2_e.blob_offset,
                    .w2_bytes = w2_e.blob_bytes,
                    .w2_params_offset = w2_e.params_offset,
                    .w2_params_bytes = w2_e.params_bytes,
                    .w2_amf_dtype = try w2_e.dataType(),
                    .w2_layout = try w2_e.tensorLayout(),
                    .w2_param_kind = try w2_e.quantParamKind(),
                    .w2_group_size = if (w2_e.group_size > 0) w2_e.group_size else 64,
                    .w2_n = in_dim,
                    .w2_k = inter,
                } };
            } else {
                const gb = try bindLinearEntry(ctx, ge, inter, in_dim);
                const ub = try bindLinearEntry(ctx, ue, inter, in_dim);
                const db = try bindLinearEntry(ctx, w2_e, in_dim, inter);
                out[e] = .{ .hot = .{ .w1 = gb, .w_up = ub, .w2 = db } };
            }
        }
    }
    return out;
}

// ============================================================================
// Tensor Loading, Unpacking & Math Helpers
// ============================================================================

fn ensureFrame(model: *graph.GraphModel, frame_opt: *?metal.Compute.Frame) graph.GraphStepError!*metal.Compute.Frame {
    if (frame_opt.* == null) {
        frame_opt.* = try model.compute.beginFrame(model.timeout_ms);
        model.beginLoraFrame(frame_opt.*.?.slot);
    }
    return &frame_opt.*.?;
}

fn flushFrame(model: *graph.GraphModel, frame_opt: *?metal.Compute.Frame) graph.GraphStepError!void {
    if (frame_opt.*) |*f| {
        const sig = try f.commit();
        frame_opt.* = null;
        try model.compute.wait(sig, model.timeout_ms);
    }
}

fn allocArenaOff(arena: *arena_mod.WiredArena, bytes: usize) !u64 {
    const slice = try arena.allocAligned(@max(bytes, 64), 64);
    @memset(slice, 0);
    return @intCast(arena.offsetOf(slice.ptr));
}

fn findLinearEntry(source: *const graph.GraphModel.AmfSource, stem: []const u8) ?*const amf.TensorEntry {
    if (source.find(stem)) |e| return e;
    var buf: [128]u8 = undefined;
    if (stem.len + 7 <= buf.len) {
        const with_w = std.fmt.bufPrint(&buf, "{s}.weight", .{stem}) catch unreachable;
        if (source.find(with_w)) |e| return e;
    }
    return null;
}

fn entryIndex(ctx: M82LoadContext, entry: *const amf.TensorEntry) usize {
    return (@intFromPtr(entry) - ctx.entry_base_ptr) / @sizeOf(amf.TensorEntry);
}

fn bindLinearCtx(ctx: M82LoadContext, stem: []const u8, expected_n: u32, expected_k: u32) !qwen3.LinearBinding {
    const e = findLinearEntry(ctx.source, stem) orelse return error.MissingTensor;
    return bindLinearEntry(ctx, e, expected_n, expected_k);
}

fn bindLinearEntry(ctx: M82LoadContext, e: *const amf.TensorEntry, expected_n: u32, expected_k: u32) !qwen3.LinearBinding {
    if (e.shape[0] != expected_n or e.shape[1] != expected_k) return error.TensorShapeMismatch;
    const idx = entryIndex(ctx, e);
    const sh = ctx.entry_shard[idx];
    if (sh == std.math.maxInt(u16)) return error.MissingTensor;
    const off = ctx.entry_offsets[idx];
    const dt = try e.dataType();
    const gs: u16 = if (e.group_size > 0) e.group_size else 64;
    const has_params = (e.params_bytes > 0 and e.params_offset + e.params_bytes <= e.blob_bytes);
    const w_buf = ctx.weight_shards[sh].buffer;
    const raw_slice = ctx.entry_slices[idx];
    return .{
        .weight = .{ .buffer = w_buf, .offset = off },
        .scale_bias = if (has_params) .{ .buffer = w_buf, .offset = off + e.params_offset } else null,
        .n = expected_n,
        .k = expected_k,
        .dtype = amfToWeightDType(dt),
        .amf_dtype = dt,
        .group_size = gs,
        .tensor_scale_per_row = (dt == .fp8_e4m3_tensor and e.params_bytes >= @as(u64, expected_n) * 4),
        .alg_bytes = e.blob_bytes,
        .layout = try e.tensorLayout(),
        .param_kind = try e.quantParamKind(),
        .weight_host = raw_slice[0..@min(raw_slice.len, @as(usize, @intCast(e.data_bytes)))],
        .scale_bias_host = if (has_params and raw_slice.len >= e.params_offset + e.params_bytes)
            raw_slice[@intCast(e.params_offset)..][0..@intCast(e.params_bytes)]
        else
            null,
    };
}

fn bindNormCtx(ctx: M82LoadContext, name: []const u8, expected_len: u32) !qwen3.NormBinding {
    const e = ctx.source.find(name) orelse return error.MissingTensor;
    if (e.shape[0] != expected_len) return error.TensorShapeMismatch;
    const idx = entryIndex(ctx, e);
    const sh = ctx.entry_shard[idx];
    if (sh == std.math.maxInt(u16)) return error.MissingTensor;
    const dt = try e.dataType();
    const fdt: quant.FloatDType = switch (dt) {
        .f16 => .f16,
        .bf16 => .bf16,
        else => return error.UnsupportedDType,
    };
    return .{
        .binding = .{ .buffer = ctx.weight_shards[sh].buffer, .offset = ctx.entry_offsets[idx] },
        .dtype = fdt,
    };
}

fn decodeEntryToF32(e: *const amf.TensorEntry, raw: []const u8, dst: []f32) !void {
    const dt = try e.dataType();
    switch (dt) {
        .f32 => {
            if (raw.len < dst.len * 4) return error.TensorShapeMismatch;
            const s: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, raw[0 .. dst.len * 4]));
            @memcpy(dst, s);
        },
        .f16 => {
            if (raw.len < dst.len * 2) return error.TensorShapeMismatch;
            const s: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, raw[0 .. dst.len * 2]));
            for (s, 0..) |v, i| dst[i] = @floatCast(v);
        },
        .bf16 => {
            if (raw.len < dst.len * 2) return error.TensorShapeMismatch;
            const s: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, raw[0 .. dst.len * 2]));
            for (s, 0..) |v, i| dst[i] = @bitCast(@as(u32, v) << 16);
        },
        else => {
            const qf_dt = quant_formats.DType.fromAmf(dt) orelse return error.UnsupportedDType;
            const rows: usize = e.shape[0];
            const cols: usize = if (e.rank >= 2) e.shape[1] else 1;
            if (rows * cols != dst.len) return error.TensorShapeMismatch;
            const params_slice: []const u8 = if (e.params_offset > 0 and e.params_offset <= raw.len) raw[@intCast(e.params_offset)..] else &.{};
            const gs: u16 = if (e.group_size > 0) e.group_size else 64;
            for (0..rows) |r| {
                quant_formats.dequantRowCpuRef(qf_dt, gs, raw, params_slice, r, rows, cols, dst[r * cols ..][0..cols]) catch return error.UnsupportedDType;
            }
        },
    }
}

fn entryElemCount(e: *const amf.TensorEntry) u32 {
    var total: u64 = 1;
    for (0..e.rank) |d| {
        if (e.shape[d] > 0) total *= e.shape[d];
    }
    return @intCast(total);
}

fn loadSconvF32TransposedIfNeeded(
    allocator: std.mem.Allocator,
    ctx: M82LoadContext,
    arena: *arena_mod.WiredArena,
    buf: metal.Buffer,
    e: *const amf.TensorEntry,
    dim: u32,
) !metal.BufferBinding {
    const idx = entryIndex(ctx, e);
    const total: usize = @as(usize, dim) * 4;
    const off = try allocArenaOff(arena, total * 4);
    const dst: []f32 = @alignCast(std.mem.bytesAsSlice(f32, arena.bytes()[@intCast(off)..][0 .. total * 4]));
    if (e.rank == 2 and e.shape[0] == 4 and e.shape[1] == dim) {
        const tmp = try allocator.alloc(f32, total);
        defer allocator.free(tmp);
        try decodeEntryToF32(e, ctx.entry_slices[idx], tmp);
        // `static_w` `[4, dim]` stores lag 0 (current token) at row 0 and lag 3 at row 3;
        // reverse into `[t-3, t-2, t-1, t]` kernel tap order.
        for (0..dim) |c| {
            for (0..4) |lag| dst[c * 4 + (3 - lag)] = tmp[lag * dim + c];
        }
    } else {
        try decodeEntryToF32(e, ctx.entry_slices[idx], dst);
    }
    return .{ .buffer = buf, .offset = off };
}

fn loadRelEmbF16(
    allocator: std.mem.Allocator,
    ctx: M82LoadContext,
    arena: *arena_mod.WiredArena,
    buf: metal.Buffer,
    e: *const amf.TensorEntry,
    rel_extent: u32,
    d_rel: u32,
) !metal.BufferBinding {
    const idx = entryIndex(ctx, e);
    const total: usize = @as(usize, rel_extent) * d_rel;
    const tmp = try allocator.alloc(f32, total);
    defer allocator.free(tmp);
    try decodeEntryToF32(e, ctx.entry_slices[idx], tmp);
    const off = try allocArenaOff(arena, total * 2);
    const dst: []f16 = @alignCast(std.mem.bytesAsSlice(f16, arena.bytes()[@intCast(off)..][0 .. total * 2]));
    if (e.rank == 2 and e.shape[0] == d_rel and e.shape[1] == rel_extent) {
        for (0..rel_extent) |dist| {
            for (0..d_rel) |r| dst[dist * d_rel + r] = @floatCast(tmp[r * rel_extent + dist]);
        }
    } else {
        for (0..total) |i| dst[i] = @floatCast(tmp[i]);
    }
    return .{ .buffer = buf, .offset = off };
}

fn loadConvertedF32Binding(
    ctx: M82LoadContext,
    arena: *arena_mod.WiredArena,
    buf: metal.Buffer,
    name: []const u8,
    count: u32,
) !metal.BufferBinding {
    const e = ctx.source.find(name) orelse return error.MissingTensor;
    const idx = entryIndex(ctx, e);
    const off = try allocArenaOff(arena, @as(usize, count) * 4);
    const dst: []f32 = @alignCast(std.mem.bytesAsSlice(f32, arena.bytes()[@intCast(off)..][0 .. @as(usize, count) * 4]));
    try decodeEntryToF32(e, ctx.entry_slices[idx], dst);
    return .{ .buffer = buf, .offset = off };
}

fn loadConvertedF16Binding(
    ctx: M82LoadContext,
    arena: *arena_mod.WiredArena,
    buf: metal.Buffer,
    name: []const u8,
    count: u32,
) !metal.BufferBinding {
    const e = ctx.source.find(name) orelse return error.MissingTensor;
    const idx = entryIndex(ctx, e);
    const raw = ctx.entry_slices[idx];
    const off = try allocArenaOff(arena, @as(usize, count) * 2);
    const dst: []f16 = @alignCast(std.mem.bytesAsSlice(f16, arena.bytes()[@intCast(off)..][0 .. @as(usize, count) * 2]));
    const dt = try e.dataType();
    switch (dt) {
        .f16 => {
            const s: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, raw[0 .. @as(usize, count) * 2]));
            @memcpy(dst, s);
        },
        .bf16 => {
            const s: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, raw[0 .. @as(usize, count) * 2]));
            for (s, 0..) |v, i| dst[i] = @floatCast(@as(f32, @bitCast(@as(u32, v) << 16)));
        },
        .f32 => {
            const s: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, raw[0 .. @as(usize, count) * 4]));
            for (s, 0..) |v, i| dst[i] = @floatCast(v);
        },
        else => return error.UnsupportedDType,
    }
    return .{ .buffer = buf, .offset = off };
}

fn loadTensorF32Slice(
    allocator: std.mem.Allocator,
    ctx: M82LoadContext,
    pool: *std.ArrayListUnmanaged([]f32),
    name: []const u8,
    count: u32,
) ![]const f32 {
    const e = ctx.source.find(name) orelse return error.MissingTensor;
    const idx = entryIndex(ctx, e);
    const dst = try allocator.alloc(f32, count);
    errdefer allocator.free(dst);
    try decodeEntryToF32(e, ctx.entry_slices[idx], dst);
    try pool.append(allocator, dst);
    return dst;
}

fn readOptionalScalarF32(ctx: M82LoadContext, name: []const u8) ?f32 {
    const e = ctx.source.find(name) orelse return null;
    const idx = entryIndex(ctx, e);
    var out: [1]f32 = .{1.0};
    decodeEntryToF32(e, ctx.entry_slices[idx], &out) catch return null;
    return out[0];
}

fn unpackHy4KvBProj(
    allocator: std.mem.Allocator,
    ctx: M82LoadContext,
    stem: []const u8,
    n_q: u32,
    qk_nope: u32,
    v_dim: u32,
    kv_rank: u32,
    w_uk_t_dst: []f16,
    w_uv_dst: []f16,
) !void {
    const e = findLinearEntry(ctx.source, stem) orelse return error.MissingTensor;
    const total_rows: u32 = n_q * (qk_nope + v_dim);
    if (e.shape[0] != total_rows or e.shape[1] != kv_rank) return error.TensorShapeMismatch;
    const idx = entryIndex(ctx, e);
    const tmp = try allocator.alloc(f32, @as(usize, total_rows) * kv_rank);
    defer allocator.free(tmp);
    try decodeEntryToF32(e, ctx.entry_slices[idx], tmp);

    const per_head = qk_nope + v_dim;
    for (0..n_q) |qh| {
        const head_base = qh * per_head;
        // `W_uk[qh]` is `[qk_nope, kv_rank]` -> transpose to `W_uk_t[qh]` `[kv_rank, qk_nope]`
        const uk_dst = w_uk_t_dst[qh * kv_rank * qk_nope ..];
        for (0..qk_nope) |d| {
            const src_row = tmp[(head_base + d) * kv_rank ..];
            for (0..kv_rank) |r| {
                uk_dst[r * qk_nope + d] = @floatCast(src_row[r]);
            }
        }
        // `W_uv[qh]` is `[v_dim, kv_rank]` -> direct copy
        const uv_dst = w_uv_dst[qh * v_dim * kv_rank ..];
        for (0..v_dim) |v| {
            const src_row = tmp[(head_base + qk_nope + v) * kv_rank ..];
            for (0..kv_rank) |r| {
                uv_dst[v * kv_rank + r] = @floatCast(src_row[r]);
            }
        }
    }
}

fn linearToInkling2Desc(lin: qwen3.LinearBinding) ink_gpu.Inkling2LinearDesc {
    return .{
        .amf_dtype = lin.amf_dtype,
        .group_size = lin.group_size,
        .tensor_scale_per_row = lin.tensor_scale_per_row,
        .weight = lin.weight,
        .params = lin.scale_bias,
        .n = lin.n,
        .k = lin.k,
    };
}

fn loadControllerMatrixToGpu(
    ctx: M82LoadContext,
    arena: *arena_mod.WiredArena,
    buf: metal.Buffer,
    e: *const amf.TensorEntry,
    rows: u32,
    cols: u32,
    out_dt: quant.FloatDType,
) !metal.BufferBinding {
    const idx = entryIndex(ctx, e);
    const raw = ctx.entry_slices[idx];
    const lay = try e.tensorLayout();
    const dt = try e.dataType();
    if (lay == .sme2_panel64) {
        const p_dt: sme.PanelDType = switch (dt) {
            .f16 => .f16,
            .bf16 => .bf16,
            .q4_affine => .q4_affine,
            .q3_affine => .q3_affine,
            .q2_affine => .q2_affine,
            else => return error.UnsupportedDType,
        };
        const dw = try ink_gpu.deinterleavePanel64ToArena(arena, p_dt, raw, rows, cols, out_dt);
        return dw.binding(buf);
    }
    if (lay == .row_major and ((out_dt == .f16 and dt == .f16) or (out_dt == .bf16 and dt == .bf16))) {
        return .{
            .buffer = ctx.weight_shards[ctx.entry_shard[idx]].buffer,
            .offset = ctx.entry_offsets[idx],
        };
    }
    const count: usize = @as(usize, rows) * cols;
    const off = try allocArenaOff(arena, count * @sizeOf(u16));
    const dst_u16: []u16 = @alignCast(std.mem.bytesAsSlice(u16, arena.bytes()[@intCast(off)..][0 .. count * 2]));
    switch (dt) {
        .f16 => {
            const src: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, raw[0 .. count * 2]));
            if (out_dt == .f16) {
                @memcpy(@as([]f16, @ptrCast(dst_u16)), src);
            } else {
                for (src, 0..) |v, i| {
                    const fv: f32 = @floatCast(v);
                    dst_u16[i] = quant.f32ToBf16(fv);
                }
            }
        },
        .bf16 => {
            const src: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, raw[0 .. count * 2]));
            if (out_dt == .bf16) {
                @memcpy(dst_u16, src);
            } else {
                const dst_f16: []f16 = @ptrCast(dst_u16);
                for (src, 0..) |v, i| dst_f16[i] = @floatCast(quant.bf16ToF32(v));
            }
        },
        .f32 => {
            const src: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, raw[0 .. count * 4]));
            if (out_dt == .f16) {
                const dst_f16: []f16 = @ptrCast(dst_u16);
                for (src, 0..) |v, i| dst_f16[i] = @floatCast(v);
            } else {
                for (src, 0..) |v, i| dst_u16[i] = quant.f32ToBf16(v);
            }
        },
        else => return error.UnsupportedDType,
    }
    return .{ .buffer = buf, .offset = off };
}

fn loadGpuControllerToArena(
    ctx: M82LoadContext,
    arena: *arena_mod.WiredArena,
    buf: metal.Buffer,
    layer_idx: u32,
    ctrl_in_dim: u32,
    gate_rank: u32,
    out_dims: [ink_ctrl.NUM_BRANCHES]u32,
) !Inkling2GpuController {
    var name_buf: [96]u8 = undefined;
    var out: Inkling2GpuController = undefined;
    for (0..ink_ctrl.NUM_BRANCHES) |bi| {
        const br: ink_ctrl.ControllerBranch = @fromBackingInt(@intCast(bi));
        const sd = out_dims[bi];
        const dn_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.controllers.{s}.down_weight", .{ layer_idx, br.label() }) catch unreachable;
        const dn_e = ctx.source.find(dn_nm) orelse return error.MissingTensor;
        const fg_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.controllers.{s}.forget_weight", .{ layer_idx, br.label() }) catch unreachable;
        const fg_e = ctx.source.find(fg_nm) orelse return error.MissingTensor;
        const gn_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.controllers.{s}.gain_weight", .{ layer_idx, br.label() }) catch unreachable;
        const gn_e = ctx.source.find(gn_nm) orelse return error.MissingTensor;

        const out_dt: quant.FloatDType = if ((try dn_e.dataType()) == .f16) .f16 else .bf16;
        const dn_bind = try loadControllerMatrixToGpu(ctx, arena, buf, dn_e, gate_rank, ctrl_in_dim, out_dt);
        const fg_bind = try loadControllerMatrixToGpu(ctx, arena, buf, fg_e, sd, gate_rank, out_dt);
        const gn_bind = try loadControllerMatrixToGpu(ctx, arena, buf, gn_e, sd, gate_rank, out_dt);

        out.branches[bi] = .{
            .down_weight = dn_bind,
            .forget_weight = fg_bind,
            .gain_weight = gn_bind,
            .weight_dtype = out_dt,
            .state_dim = sd,
            .down_in_dim = ctrl_in_dim,
            .gate_rank = gate_rank,
        };
    }
    return out;
}

fn buildInkling2IndirectMoeMeta(
    arena: *arena_mod.WiredArena,
    arena_buf: metal.Buffer,
    experts: []const M82ExpertWeight,
) !?Inkling2IndirectMoeMeta {
    if (experts.len == 0) return null;
    const first_hot = switch (experts[0]) {
        .hot => |hw| hw,
        .cold => return null,
    };
    const base_buf = first_hot.w1.weight.buffer;

    var scheme_descs: [moe.NUM_SCHEME_BUCKETS]moe.MoeSchemeDesc = .{ .{}, .{}, .{}, .{}, .{} };
    var scheme_counts: [moe.NUM_SCHEME_BUCKETS]u32 = .{ 0, 0, 0, 0, 0 };
    var bucket_pair: [moe.NUM_SCHEME_BUCKETS]bool = .{ true, true, true, true, true };
    var bucket_count: usize = 0;

    // Pre-scan to validate single-buffer residency, row_major layout, supported dtypes, and <= 5 scheme buckets.
    for (experts) |ew| {
        const hw = switch (ew) {
            .hot => |h| h,
            .cold => return null,
        };
        if (hw.w1.layout != .row_major or hw.w2.layout != .row_major) return null;
        if (hw.w1.amf_dtype == .f32 or hw.w1.amf_dtype == .u32) return null;
        if (hw.w2.amf_dtype == .f32 or hw.w2.amf_dtype == .u32) return null;
        if (hw.w1.weight.buffer.handle != base_buf.handle or hw.w2.weight.buffer.handle != base_buf.handle) return null;
        if (hw.w1.scale_bias) |sb| if (sb.buffer.handle != base_buf.handle) return null;
        if (hw.w2.scale_bias) |sb| if (sb.buffer.handle != base_buf.handle) return null;
        if (hw.w_up) |wu| {
            if (wu.layout != .row_major or wu.weight.buffer.handle != base_buf.handle) return null;
            if (wu.scale_bias) |sb| if (sb.buffer.handle != base_buf.handle) return null;
            if (wu.amf_dtype != hw.w1.amf_dtype or wu.group_size != hw.w1.group_size or
                wu.tensor_scale_per_row != hw.w1.tensor_scale_per_row)
            {
                return null;
            }
        }
        const desc: moe.MoeSchemeDesc = .{
            .gate_up_dtype = hw.w1.amf_dtype,
            .gate_up_group_size = hw.w1.group_size,
            .gate_up_tensor_scale_per_row = hw.w1.tensor_scale_per_row,
            .down_dtype = hw.w2.amf_dtype,
            .down_group_size = hw.w2.group_size,
            .down_tensor_scale_per_row = hw.w2.tensor_scale_per_row,
        };
        const is_pair = (hw.w_up == null);
        var found: ?usize = null;
        for (0..bucket_count) |bi| {
            if (scheme_descs[bi].eql(desc) and bucket_pair[bi] == is_pair) {
                found = bi;
                break;
            }
        }
        if (found == null) {
            if (bucket_count >= moe.NUM_SCHEME_BUCKETS) return null;
            scheme_descs[bucket_count] = desc;
            bucket_pair[bucket_count] = is_pair;
            bucket_count += 1;
        }
    }
    if (bucket_count == 0) return null;

    const tbl_off = try allocArenaOff(arena, experts.len * @sizeOf(moe.ExpertWeights));
    const tbl: []moe.ExpertWeights = @alignCast(std.mem.bytesAsSlice(
        moe.ExpertWeights,
        arena.bytes()[@intCast(tbl_off)..][0 .. experts.len * @sizeOf(moe.ExpertWeights)],
    ));
    const is_hetero = (bucket_count > 1);
    var schemes_bind: ?metal.BufferBinding = null;
    var schemes_slice: []u8 = &.{};
    if (is_hetero) {
        const sc_off = try allocArenaOff(arena, amf.align64(experts.len));
        schemes_slice = arena.bytes()[@intCast(sc_off)..][0..experts.len];
        schemes_bind = .{ .buffer = arena_buf, .offset = sc_off };
    }

    for (experts, 0..) |ew, ei| {
        const hw = ew.hot;
        tbl[ei] = .{
            .gate_qweight_offset = hw.w1.weight.offset,
            .gate_sb_offset = if (hw.w1.scale_bias) |sb| sb.offset else 0,
            .up_qweight_offset = if (hw.w_up) |wu| wu.weight.offset else 0,
            .up_sb_offset = if (hw.w_up) |wu| (if (wu.scale_bias) |sb| sb.offset else 0) else 0,
            .down_qweight_offset = hw.w2.weight.offset,
            .down_sb_offset = if (hw.w2.scale_bias) |sb| sb.offset else 0,
        };
        const desc: moe.MoeSchemeDesc = .{
            .gate_up_dtype = hw.w1.amf_dtype,
            .gate_up_group_size = hw.w1.group_size,
            .gate_up_tensor_scale_per_row = hw.w1.tensor_scale_per_row,
            .down_dtype = hw.w2.amf_dtype,
            .down_group_size = hw.w2.group_size,
            .down_tensor_scale_per_row = hw.w2.tensor_scale_per_row,
        };
        const is_pair = (hw.w_up == null);
        var bid: usize = 0;
        for (0..bucket_count) |bi| {
            if (scheme_descs[bi].eql(desc) and bucket_pair[bi] == is_pair) {
                bid = bi;
                break;
            }
        }
        scheme_counts[bid] += 1;
        if (is_hetero) {
            schemes_slice[ei] = @intCast(bid);
        }
    }

    return .{
        .weights_base = .{ .buffer = base_buf, .offset = 0 },
        .expert_table = .{ .buffer = arena_buf, .offset = tbl_off },
        .expert_schemes = schemes_bind,
        .is_heterogeneous = is_hetero,
        .pair_interleaved = bucket_pair[0],
        .bucket_pair_interleaved = if (is_hetero) bucket_pair else null,
        .scheme_counts = scheme_counts,
        .scheme_descs = scheme_descs,
    };
}

fn loadControllerFromParsedBytes(
    allocator: std.mem.Allocator,
    ctx: M82LoadContext,
    layer_idx: u32,
    ctrl_cfg: ink_ctrl.ControllerConfig,
) !ink_ctrl.Inkling2LoopController {
    var name_buf: [96]u8 = undefined;
    var inputs: [ink_ctrl.NUM_BRANCHES]ink_ctrl.BranchWeightInput = undefined;
    var owned_f32s: [ink_ctrl.NUM_BRANCHES * 3]?[]f32 = @splat(null);
    defer {
        for (owned_f32s) |maybe_s| {
            if (maybe_s) |s| allocator.free(s);
        }
    }

    const h = ctrl_cfg.hidden_size;
    for (0..ink_ctrl.NUM_BRANCHES) |bi| {
        const br: ink_ctrl.ControllerBranch = @fromBackingInt(@intCast(bi));
        const sd = ctrl_cfg.branches[bi].state_dim;
        const dn_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.controllers.{s}.down_weight", .{ layer_idx, br.label() }) catch unreachable;
        const dn_e = ctx.source.find(dn_nm) orelse return error.MissingTensor;
        const fg_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.controllers.{s}.forget_weight", .{ layer_idx, br.label() }) catch unreachable;
        const fg_e = ctx.source.find(fg_nm) orelse return error.MissingTensor;
        const gn_nm = std.fmt.bufPrint(&name_buf, "layers.{d}.controllers.{s}.gain_weight", .{ layer_idx, br.label() }) catch unreachable;
        const gn_e = ctx.source.find(gn_nm) orelse return error.MissingTensor;

        const dn_lay = try dn_e.tensorLayout();
        const fg_lay = try fg_e.tensorLayout();
        const gn_lay = try gn_e.tensorLayout();

        const dn_in: ink_ctrl.WeightMatrixInput = if (dn_lay == .sme2_panel64)
            .{ .sme2_panel64 = ctx.entry_slices[entryIndex(ctx, dn_e)] }
        else blk: {
            const s = try allocator.alloc(f32, @as(usize, ink_ctrl.GATE_RANK) * h);
            owned_f32s[bi * 3 + 0] = s;
            try decodeEntryToF32(dn_e, ctx.entry_slices[entryIndex(ctx, dn_e)], s);
            break :blk .{ .dense_f32 = s };
        };
        const fg_in: ink_ctrl.WeightMatrixInput = if (fg_lay == .sme2_panel64)
            .{ .sme2_panel64 = ctx.entry_slices[entryIndex(ctx, fg_e)] }
        else blk: {
            const s = try allocator.alloc(f32, @as(usize, sd) * ink_ctrl.GATE_RANK);
            owned_f32s[bi * 3 + 1] = s;
            try decodeEntryToF32(fg_e, ctx.entry_slices[entryIndex(ctx, fg_e)], s);
            break :blk .{ .dense_f32 = s };
        };
        const gn_in: ink_ctrl.WeightMatrixInput = if (gn_lay == .sme2_panel64)
            .{ .sme2_panel64 = ctx.entry_slices[entryIndex(ctx, gn_e)] }
        else blk: {
            const s = try allocator.alloc(f32, @as(usize, sd) * ink_ctrl.GATE_RANK);
            owned_f32s[bi * 3 + 2] = s;
            try decodeEntryToF32(gn_e, ctx.entry_slices[entryIndex(ctx, gn_e)], s);
            break :blk .{ .dense_f32 = s };
        };

        inputs[bi] = .{
            .down_weight = dn_in,
            .forget_weight = fg_in,
            .gain_weight = gn_in,
        };
    }

    return ink_ctrl.Inkling2LoopController.init(ctrl_cfg, inputs);
}

fn amfToWeightDType(dt: amf.DType) quant.WeightDType {
    return switch (dt) {
        .f16 => .f16,
        .bf16 => .bf16,
        .q4_affine => .q4_affine,
        .q3_affine => .q3_affine,
        .q2_affine => .q2_affine,
        else => .bf16,
    };
}

fn f16SliceToF32(src: []const f16, dst: []f32) void {
    for (src, dst) |v, *o| o.* = @floatCast(v);
}

fn f32SliceToF16(src: []const f32, dst: []f16) void {
    for (src, dst) |v, *o| o.* = @floatCast(v);
}

fn applyUnitRmsNormPerHeadF16(buf: []f16, m: u32, num_heads: u32, head_dim: u32, eps: f32) void {
    for (0..m) |tok| {
        for (0..num_heads) |nh| {
            const row = buf[(tok * num_heads + nh) * head_dim ..][0..head_dim];
            var sq: f32 = 0.0;
            for (row) |v| {
                const fv: f32 = @floatCast(v);
                sq += fv * fv;
            }
            const inv_rms: f32 = 1.0 / @sqrt(sq / @as(f32, @floatFromInt(head_dim)) + eps);
            for (row) |*v| {
                v.* = @floatCast(@as(f32, @floatCast(v.*)) * inv_rms);
            }
        }
    }
}

fn applyScaledL2ClipPerHeadF16(
    buf: []f16,
    scale_vec: ?[]const f32,
    m: u32,
    num_heads: u32,
    head_dim: u32,
    threshold: f32,
    log_positions: ?[]const u32,
    log_floor: u32,
    log_alpha: f32,
) void {
    const hd: usize = head_dim;
    const nh_total: usize = num_heads;
    const floor_f: f32 = if (log_floor > 0) @floatFromInt(log_floor) else 1.0;
    for (0..m) |tok| {
        const tau: f32 = if (log_positions) |lp| blk: {
            const ratio: f32 = @max(1.0, (@as(f32, @floatFromInt(lp[tok])) + 1.0) / floor_f);
            break :blk 1.0 + log_alpha * @log(ratio);
        } else 1.0;
        for (0..nh_total) |nh| {
            const row = buf[(tok * nh_total + nh) * hd ..][0..hd];
            const sv_off = nh * hd;
            var sq: f32 = 0.0;
            for (0..hd) |i| {
                var fv: f32 = @floatCast(row[i]);
                if (scale_vec) |sv| fv *= sv[sv_off + i];
                sq += fv * fv;
            }
            const norm = @sqrt(sq);
            const clip: f32 = if (threshold > 0.0) @min(1.0, threshold / (norm + 1e-6)) else 1.0;
            const fac: f32 = clip * tau;
            for (0..hd) |i| {
                var fv: f32 = @floatCast(row[i]);
                if (scale_vec) |sv| fv *= sv[sv_off + i];
                row[i] = @floatCast(fv * fac);
            }
        }
    }
}

fn stageMsgBankF32(
    ring: *const ink_gpu.AttnResRingArena,
    raw: []const u8,
    cur_host: []const f16,
    start_slot: u32,
    end_slot: u32,
    m: u32,
    h: u32,
    rms_eps: f32,
    dst_bank_norm: []f32,
) usize {
    const h_us: usize = h;
    const m_us: usize = m;
    const mh: usize = m_us * h_us;
    var bank_len: usize = 0;

    var slot: u32 = start_slot;
    while (slot < end_slot and bank_len + 1 < ink_gpu.MAX_ATTN_RES_BLOCKS) : (slot += 1) {
        const b_off: usize = @intCast(ring.base_off + @as(u64, slot) * ring.block_stride_elems * 2);
        const src: []const f16 = @alignCast(std.mem.bytesAsSlice(f16, raw[b_off..][0 .. mh * 2]));
        const dst = dst_bank_norm[bank_len * mh ..][0..mh];
        for (0..m_us) |tok| {
            var sq: f32 = 0.0;
            for (0..h_us) |c| {
                const v: f32 = @floatCast(src[tok * h_us + c]);
                dst[tok * h_us + c] = v;
                sq += v * v;
            }
            const inv_rms = 1.0 / @sqrt(sq / @as(f32, @floatFromInt(h)) + rms_eps);
            for (0..h_us) |c| dst[tok * h_us + c] *= inv_rms;
        }
        bank_len += 1;
    }

    // Append `previous_hidden_states` (`cur_host`).
    const dst_cur = dst_bank_norm[bank_len * mh ..][0..mh];
    for (0..m_us) |tok| {
        var sq: f32 = 0.0;
        for (0..h_us) |c| {
            const v: f32 = @floatCast(cur_host[tok * h_us + c]);
            dst_cur[tok * h_us + c] = v;
            sq += v * v;
        }
        const inv_rms = 1.0 / @sqrt(sq / @as(f32, @floatFromInt(h)) + rms_eps);
        for (0..h_us) |c| dst_cur[tok * h_us + c] *= inv_rms;
    }
    bank_len += 1;
    return bank_len;
}

fn computeCrossPassMessagesF32(
    bank_norm_f32: []const f32,
    bank_len: usize,
    q_src: []const f32,
    kv_q_src: ?[]const f32,
    loop_layer_idx: usize,
    m: u32,
    h: u32,
    out_msgs: []f32,
) void {
    const h_us: usize = h;
    const m_us: usize = m;
    const mh: usize = m_us * h_us;
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(h)));
    const q_rows: usize = @max(@as(usize, 1), q_src.len / h_us);
    const kv_slice = kv_q_src orelse q_src;
    const kv_rows: usize = @max(@as(usize, 1), kv_slice.len / h_us);

    // Per `Inkling2LoopMergeMessageQueries`:
    //   `middle_cycle_cross_pass_queries` stores `[query, mlp]` for each loop layer (`2*k`, `2*k+1`);
    //   `middle_cycle_cross_pass_kv_queries` stores `[key, value]` for each loop layer (`2*k`, `2*k+1`).
    const queries = [4][]const f32{
        q_src[((2 * loop_layer_idx + 0) % q_rows) * h_us ..][0..h_us],
        kv_slice[((2 * loop_layer_idx + 0) % kv_rows) * h_us ..][0..h_us],
        kv_slice[((2 * loop_layer_idx + 1) % kv_rows) * h_us ..][0..h_us],
        q_src[((2 * loop_layer_idx + 1) % q_rows) * h_us ..][0..h_us],
    };

    var scores: [ink_gpu.MAX_ATTN_RES_BLOCKS]f32 = undefined;
    var weights: [ink_gpu.MAX_ATTN_RES_BLOCKS]f32 = undefined;
    for (0..4) |qi| {
        const qv = queries[qi];
        for (0..m_us) |tok| {
            var max_s: f32 = -std.math.inf(f32);
            for (0..bank_len) |b| {
                const bv = bank_norm_f32[b * mh + tok * h_us ..][0..h_us];
                var dot_v: f32 = 0.0;
                for (0..h_us) |c| dot_v += qv[c] * bv[c];
                const s = dot_v * scale;
                scores[b] = s;
                if (s > max_s) max_s = s;
            }
            var exp_sum: f32 = 0.0;
            for (0..bank_len) |b| {
                const e = @exp(scores[b] - max_s);
                weights[b] = e;
                exp_sum += e;
            }
            const inv_sum: f32 = if (exp_sum > 0.0) 1.0 / exp_sum else 0.0;
            for (0..bank_len) |b| weights[b] *= inv_sum;

            const dst = out_msgs[qi * mh + tok * h_us ..][0..h_us];
            @memset(dst, 0.0);
            for (0..bank_len) |b| {
                const w = weights[b];
                const bv = bank_norm_f32[b * mh + tok * h_us ..][0..h_us];
                for (0..h_us) |c| dst[c] += w * bv[c];
            }
        }
    }
}

fn applyRopeInPlaceF16(vec: []f16, pos: u32, rope_theta: f32) void {
    const half = vec.len / 2;
    const pos_f: f32 = @floatFromInt(pos);
    const dim_f: f32 = @floatFromInt(vec.len);
    for (0..half) |i| {
        const exp_val: f32 = -2.0 * @as(f32, @floatFromInt(i)) / dim_f;
        const freq: f32 = std.math.pow(f32, rope_theta, exp_val);
        const angle: f32 = pos_f * freq;
        const c = @cos(angle);
        const s = @sin(angle);
        const x0: f32 = @floatCast(vec[i]);
        const x1: f32 = @floatCast(vec[i + half]);
        vec[i] = @floatCast(x0 * c - x1 * s);
        vec[i + half] = @floatCast(x0 * s + x1 * c);
    }
}

fn attnResBlockBind(ring: *const ink_gpu.AttnResRingArena, buf: metal.Buffer, slot: u32) metal.BufferBinding {
    return .{
        .buffer = buf,
        .offset = ring.base_off + @as(u64, slot) * @as(u64, ring.block_stride_elems) * @sizeOf(f16),
    };
}

fn commitSconvRingAccepted(
    arena_bytes: []u8,
    ring: *const ink_gpu.SconvRingArena,
    slot_idx: u32,
    tree_in_off: u64,
    prev_pos: u32,
    accepted_nodes: []const u16,
    dim: u32,
) void {
    const dim_us: usize = dim;
    const state_slice = ring.layerSlice(slot_idx) catch return;
    const tree_src: []const f16 = @alignCast(std.mem.bytesAsSlice(
        f16,
        arena_bytes[@intCast(tree_in_off)..],
    ));
    var cur_pos = prev_pos;
    for (accepted_nodes) |nid| {
        const dst_slot = cur_pos % ink_gpu.INKLING2_SCONV_RING_SLOTS;
        @memcpy(
            state_slice[dst_slot * dim_us ..][0..dim_us],
            tree_src[@as(usize, nid) * dim_us ..][0..dim_us],
        );
        cur_pos += 1;
    }
}

fn applyHy4HcPre(
    streams: []const f16,
    x_out: []f16,
    hc: Hy4HcWeights,
    m: u32,
    hc_mult: u32,
    h: u32,
    eps: f32,
    magnitude: f32,
    tmp_u: []f32,
    post_weights_out: []f32,
) void {
    const in_dim: usize = hc.in_dim;
    const use_concat = (in_dim == @as(usize, hc_mult) * h);
    for (0..m) |tok| {
        // Build normalized iHC gate input `u[0..in_dim]`.
        var sq: f32 = 0.0;
        if (use_concat) {
            const base = tok * hc_mult * h;
            for (0..in_dim) |i| {
                const fv: f32 = @floatCast(streams[base + i]);
                tmp_u[i] = fv;
                sq += fv * fv;
            }
        } else {
            const inv_s: f32 = 1.0 / @as(f32, @floatFromInt(hc_mult));
            const lim = @min(in_dim, @as(usize, h));
            for (0..lim) |c| {
                var sum: f32 = 0.0;
                for (0..hc_mult) |s| sum += @as(f32, @floatCast(streams[(tok * hc_mult + s) * h + c]));
                const mv = sum * inv_s;
                tmp_u[c] = mv;
                sq += mv * mv;
            }
            if (lim < in_dim) @memset(tmp_u[lim..in_dim], 0);
        }
        const inv_rms: f32 = 1.0 / @sqrt(sq / @as(f32, @floatFromInt(@max(in_dim, 1))) + eps);
        for (0..in_dim) |i| tmp_u[i] *= inv_rms;

        var pre_w: [8]f32 = @splat(0);
        var pre_sum: f32 = 0.0;
        const sc_pre: f32 = if (hc.hc_scale.len > 0) hc.hc_scale[0] else 1.0;
        const sc_post: f32 = if (hc.hc_scale.len > 1) hc.hc_scale[1] else sc_pre;
        for (0..hc_mult) |s| {
            var dot_pre: f32 = 0.0;
            var dot_post: f32 = 0.0;
            const row_pre = hc.hc_fn[s * in_dim ..][0..in_dim];
            const row_post = hc.hc_fn[(hc_mult + s) * in_dim ..][0..in_dim];
            for (0..in_dim) |i| {
                dot_pre += row_pre[i] * tmp_u[i];
                dot_post += row_post[i] * tmp_u[i];
            }
            const logit_pre = hc.hc_base[s] + sc_pre * std.math.tanh(dot_pre);
            const logit_post = hc.hc_base[hc_mult + s] + sc_post * std.math.tanh(dot_post);
            const wp = 1.0 / (1.0 + @exp(-std.math.clamp(logit_pre, -15.0, 15.0)));
            pre_w[s] = wp;
            pre_sum += wp;
            post_weights_out[tok * hc_mult + s] = magnitude / (1.0 + @exp(-std.math.clamp(logit_post, -15.0, 15.0)));
        }
        const norm_pre: f32 = if (pre_sum > 1e-6) (1.0 / pre_sum) else (1.0 / @as(f32, @floatFromInt(hc_mult)));
        for (0..h) |c| {
            var acc: f32 = 0.0;
            for (0..hc_mult) |s| {
                acc += (pre_w[s] * norm_pre) * @as(f32, @floatCast(streams[(tok * hc_mult + s) * h + c]));
            }
            x_out[tok * h + c] = @floatCast(acc);
        }
    }
}

fn applyHy4HcPost(
    streams: []f16,
    delta: []const f16,
    post_weights: []const f32,
    m: u32,
    hc_mult: u32,
    h: u32,
) void {
    for (0..m) |tok| {
        for (0..hc_mult) |s| {
            const pw = post_weights[tok * hc_mult + s];
            const dst = streams[(tok * hc_mult + s) * h ..][0..h];
            const src = delta[tok * h ..][0..h];
            for (0..h) |c| {
                dst[c] = @floatCast(@as(f32, @floatCast(dst[c])) + pw * @as(f32, @floatCast(src[c])));
            }
        }
    }
}

fn applyHy4HcHead(
    streams: []const f16,
    x_out: []f16,
    hh: Hy4HcWeights,
    m: u32,
    hc_mult: u32,
    h: u32,
    eps: f32,
    tmp_u: []f32,
) void {
    const in_dim: usize = hh.in_dim;
    const use_concat = (in_dim == @as(usize, hc_mult) * h);
    const sc: f32 = if (hh.hc_scale.len > 0) hh.hc_scale[0] else 1.0;
    for (0..m) |tok| {
        var sq: f32 = 0.0;
        if (use_concat) {
            const base = tok * hc_mult * h;
            for (0..in_dim) |i| {
                const fv: f32 = @floatCast(streams[base + i]);
                tmp_u[i] = fv;
                sq += fv * fv;
            }
        } else {
            const inv_s: f32 = 1.0 / @as(f32, @floatFromInt(hc_mult));
            const lim = @min(in_dim, @as(usize, h));
            for (0..lim) |c| {
                var sum: f32 = 0.0;
                for (0..hc_mult) |s| sum += @as(f32, @floatCast(streams[(tok * hc_mult + s) * h + c]));
                const mv = sum * inv_s;
                tmp_u[c] = mv;
                sq += mv * mv;
            }
            if (lim < in_dim) @memset(tmp_u[lim..in_dim], 0);
        }
        const inv_rms: f32 = 1.0 / @sqrt(sq / @as(f32, @floatFromInt(@max(in_dim, 1))) + eps);
        for (0..in_dim) |i| tmp_u[i] *= inv_rms;

        var w: [8]f32 = @splat(0);
        var max_l: f32 = -std.math.inf(f32);
        for (0..hc_mult) |s| {
            var d: f32 = 0.0;
            const row = hh.hc_fn[s * in_dim ..][0..in_dim];
            for (0..in_dim) |i| d += row[i] * tmp_u[i];
            const lv = hh.hc_base[s] + sc * std.math.tanh(d);
            w[s] = lv;
            if (lv > max_l) max_l = lv;
        }
        var sum_e: f32 = 0.0;
        for (0..hc_mult) |s| {
            w[s] = @exp(w[s] - max_l);
            sum_e += w[s];
        }
        const inv_e: f32 = if (sum_e > 1e-6) (1.0 / sum_e) else (1.0 / @as(f32, @floatFromInt(hc_mult)));
        for (0..h) |c| {
            var acc: f32 = 0.0;
            for (0..hc_mult) |s| acc += (w[s] * inv_e) * @as(f32, @floatCast(streams[(tok * hc_mult + s) * h + c]));
            x_out[tok * h + c] = @floatCast(acc);
        }
    }
}

// ============================================================================
// JSON Helpers (matching `graph.zig` conventions)
// ============================================================================

fn jsonStr(obj: std.json.ObjectMap, key: []const u8) ?[]const u8 {
    const val = obj.get(key) orelse return null;
    return if (val == .string) val.string else null;
}

fn jsonBool(obj: std.json.ObjectMap, key: []const u8) ?bool {
    const val = obj.get(key) orelse return null;
    return if (val == .bool) val.bool else null;
}

fn jsonU32(obj: std.json.ObjectMap, key: []const u8) ?u32 {
    const val = obj.get(key) orelse return null;
    return jsonItemU32(val);
}

fn jsonItemU32(val: std.json.Value) ?u32 {
    return switch (val) {
        .integer => |v| if (v >= 0 and v <= std.math.maxInt(u32)) @intCast(v) else null,
        .float => |f| if (std.math.isFinite(f) and f >= 0.0 and f <= @as(f64, @floatFromInt(std.math.maxInt(u32)))) @intFromFloat(f) else null,
        else => null,
    };
}

fn jsonF32(obj: std.json.ObjectMap, key: []const u8) ?f32 {
    const val = obj.get(key) orelse return null;
    return switch (val) {
        .float => |f| @floatCast(f),
        .integer => |i| @floatFromInt(i),
        else => null,
    };
}
